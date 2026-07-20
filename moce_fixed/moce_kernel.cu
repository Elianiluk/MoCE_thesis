#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
#include <vector>
#include <cfloat>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

const int THREADS = 256;

template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid_cuda(scalar_t x) {
    return scalar_t(1) / (scalar_t(1) + exp(-x));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t fast_exp(scalar_t x) { return exp(x); }
template <>
__device__ __forceinline__ float fast_exp<float>(float x) { return __expf(x); }

// ============================================================
// Warp-per-channel average pool: one warp (32 threads) per (b,c).
// Threads stride over HW with step 32 → fully coalesced reads.
// Warp all-reduce via __shfl_xor_sync, thread 0 writes output.
// Grid: B*C  BlockDim: 32
// ============================================================
template <typename scalar_t>
__global__ void channel_avg_pool_warp_kernel(
    const scalar_t* __restrict__ input,  // (B, C, H, W)
    scalar_t* __restrict__ pooled,       // (B, C)
    int HW
) {
    int bc   = blockIdx.x;
    int lane = threadIdx.x;
    const scalar_t* inp = input + (long long)bc * HW;
    float sum = 0.0f;
    for (int i = lane; i < HW; i += 32)
        sum += float(inp[i]);
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, off);
    if (lane == 0)
        pooled[bc] = scalar_t(sum / float(HW));
}

// ============================================================
// Dynamic-temp routed kernel: same as fixed-temp but weights are
// per-sample (B, E_routed, K) instead of shared (E_routed, K).
// One line differs: weight indexing uses b to select the right row.
//
// inputs:
//   indices : (E_routed, K) int64 — top-K channel indices per expert (shared)
//   weights : (B, E_routed, K) float — per-sample softmax weights
// ============================================================
template <typename scalar_t>
__global__ void routed_forward_dynamic_temp_kernel(
    int nthreads,
    const scalar_t* __restrict__ input,
    const int64_t*  __restrict__ indices,
    const scalar_t* __restrict__ weights,
    int B, int C, int H, int W,
    int E_routed, int K, int E_total,
    scalar_t* __restrict__ out
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads) return;

    int HW = H * W;
    int hw = tid % HW;
    int e  = (tid / HW) % E_routed;
    int b  = tid / (E_routed * HW);

    int h = hw / W;
    int w = hw % W;

    const int64_t* idx = indices + e * K;
    const scalar_t* wt = weights + (b * E_routed + e) * K;  // per-sample

    scalar_t sum = scalar_t(0);
    int input_offset = b * C * HW + h * W + w;

    for (int k = 0; k < K; ++k) {
        sum += wt[k] * input[input_offset + (int)idx[k] * HW];
    }

    out[((b * E_total + e) * H + h) * W + w] = sum;
}

// ============================================================
// K-parallel routed kernel (dynamic temp, small-HW variant).
// One block per output pixel (b, e, hw); blockDim threads cooperate
// over K via warp reduction. blockDim == 32 when K<=32, 64 when K>32.
// Switch to this when H*W <= 16 (layers 3/4) to boost occupancy.
// ============================================================
template <typename scalar_t>
__global__ void routed_forward_dynamic_temp_k_parallel_kernel(
    const scalar_t* __restrict__ input,
    const int64_t*  __restrict__ indices,
    const scalar_t* __restrict__ weights,
    int B, int C, int H, int W,
    int E_routed, int K, int E_total,
    scalar_t* __restrict__ out
) {
    int HW  = H * W;
    int bew = blockIdx.x;
    int hw  = bew % HW;
    int e   = (bew / HW) % E_routed;
    int b   = bew / (HW * E_routed);

    int k = threadIdx.x;

    const int64_t* idx = indices + e * K;
    const scalar_t* wt = weights + (b * E_routed + e) * K;
    int input_base = b * C * HW + hw;

    scalar_t val = (k < K) ? wt[k] * input[input_base + (int)idx[k] * HW]
                           : scalar_t(0);

    // first warp-level reduce (handles both 32- and 64-thread blocks)
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);

    if (blockDim.x <= 32) {
        // single warp — lane 0 has the full sum
        if (k == 0) {
            int h = hw / W, w_idx = hw % W;
            out[((b * E_total + e) * H + h) * W + w_idx] = val;
        }
    } else {
        // two warps — collect partial sums in shared memory
        __shared__ scalar_t warp_sums[2];
        if ((k & 31) == 0) warp_sums[k >> 5] = val;
        __syncthreads();
        if (k == 0) {
            int h = hw / W, w_idx = hw % W;
            out[((b * E_total + e) * H + h) * W + w_idx] = warp_sums[0] + warp_sums[1];
        }
    }
}

// ============================================================
// Residual expert (thread-based, identical to nores_moce).
// Each thread computes one output pixel for one (b, hw).
// mask_int[c] == 1  →  channel c is unselected (contributes to residual).
// ============================================================
template <typename scalar_t>
__global__ void residual_average_kernel(
    int total_pixels,
    const scalar_t* __restrict__ input,
    const int*      __restrict__ mask_int,
    int B, int C, int H, int W, int E_total, int residual_e,
    scalar_t* __restrict__ out
) {
    int pixel_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_id >= total_pixels) return;

    int HW = H * W;
    int hw = pixel_id % HW;
    int b  = pixel_id / HW;

    int h = hw / W;
    int w = hw % W;

    const scalar_t* input_base = input + (b * C * HW + h * W + w);

    scalar_t sum   = scalar_t(0);
    scalar_t count = scalar_t(0);

    for (int c = 0; c < C; ++c) {
        if (mask_int[c] == 1) {
            sum   += input_base[c * HW];
            count += scalar_t(1);
        }
    }

    out[((b * E_total + residual_e) * H + h) * W + w] =
        (count > scalar_t(0)) ? (sum / count) : scalar_t(0);
}

// ============================================================
// C-parallel residual kernel (small-HW variant).
// One block per (b, hw); 256 threads stripe over C channels.
// Switch to this when H*W <= 64.
// ============================================================
template <typename scalar_t>
__global__ void residual_average_c_parallel_kernel(
    const scalar_t* __restrict__ input,
    const int*      __restrict__ mask_int,
    int B, int C, int H, int W, int E_total, int residual_e,
    scalar_t* __restrict__ out
) {
    int HW = H * W;
    int bw = blockIdx.x;
    int hw = bw % HW;
    int b  = bw / HW;
    int h  = hw / W, w_idx = hw % W;

    const scalar_t* input_base = input + (b * C * HW + h * W + w_idx);

    scalar_t sum = scalar_t(0), count = scalar_t(0);
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        if (mask_int[c] == 1) {
            sum   += input_base[c * HW];
            count += scalar_t(1);
        }
    }

    // warp-level reduce
    for (int off = 16; off > 0; off >>= 1) {
        sum   += __shfl_down_sync(0xffffffff, sum,   off);
        count += __shfl_down_sync(0xffffffff, count, off);
    }

    __shared__ scalar_t warp_sums[8], warp_counts[8];
    int lane = threadIdx.x & 31, warp_id = threadIdx.x >> 5;
    if (lane == 0) { warp_sums[warp_id] = sum; warp_counts[warp_id] = count; }
    __syncthreads();

    if (threadIdx.x == 0) {
        int n_warps = (blockDim.x + 31) >> 5;
        scalar_t tot_sum = 0, tot_count = 0;
        for (int i = 0; i < n_warps; ++i) {
            tot_sum   += warp_sums[i];
            tot_count += warp_counts[i];
        }
        out[((b * E_total + residual_e) * H + h) * W + w_idx] =
            (tot_count > scalar_t(0)) ? (tot_sum / tot_count) : scalar_t(0);
    }
}

// ============================================================
// Routed forward with temperature fused in.
//
// Each thread (b, e, hw):
//   1. Reads K pooled values for the selected channels of expert e
//   2. Computes gate logit → sigmoid → temperature (all in-register)
//   3. Computes numerically stable softmax over logits_norm[e] / temp
//   4. Writes weighted sum of input channels to out[b, e, hw]
//
// Temperature is NOT pre-computed — no separate temperature kernel.
// channel_avg_pool_warp_kernel must run first to produce pooled[B, C].
//
// Inputs:
//   input       : (B, C, H, W)    float32
//   pooled      : (B, C)          float32  — global avg pool of input
//   indices     : (E_routed, K)   int64    — top-K channel indices per expert
//   logits_norm : (E_routed, K)   float32  — normalized routing logits
//   gate_weight : (K,)            float32  — gate.weight reshaped from (1, K)
//   gate_bias   : (1,)           float32  — gate.bias (device pointer, read [0] in kernel)
//
// Returns writes into out[B, E_total, H, W] (routed slots only).
// ============================================================
template <typename scalar_t>
__global__ void routed_forward_with_temp_kernel(
    int nthreads,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ pooled,
    const int64_t*  __restrict__ indices,
    const scalar_t* __restrict__ logits_norm,
    const scalar_t* __restrict__ gate_weight,
    const scalar_t* __restrict__ gate_bias,   // device pointer — read [0] here
    scalar_t temp_min, scalar_t temp_max, scalar_t tau,
    int B, int C, int H, int W,
    int E_routed, int K, int E_total,
    scalar_t* __restrict__ out
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads) return;

    const int HW = H * W;
    const int hw = tid % HW;
    const int e  = (tid / HW) % E_routed;
    const int b  = tid / (E_routed * HW);

    const int64_t* idx      = indices    + e * K;
    const scalar_t* pool_b  = pooled     + b * C;
    const scalar_t* ln      = logits_norm + e * K;

    // --- 1. Gate logit from pooled selected channels ---
    scalar_t gate_logit = gate_bias[0];   // read from device memory
    #pragma unroll
    for (int k = 0; k < K; ++k)
        gate_logit += pool_b[(int)idx[k]] * gate_weight[k];

    // --- 2. Temperature ---
    scalar_t t = (temp_min + (temp_max - temp_min) * sigmoid_cuda(gate_logit)) * tau;
    if (t < scalar_t(1e-3)) t = scalar_t(1e-3);
    const scalar_t inv_t = scalar_t(1) / t;

    // --- 3. Numerically stable softmax over logits_norm / temperature ---
    scalar_t w[64];  // supports K up to 64
    scalar_t vmax = ln[0] * inv_t;
    for (int k = 1; k < K; ++k) {
        scalar_t v = ln[k] * inv_t;
        if (v > vmax) vmax = v;
    }
    scalar_t wsum = scalar_t(0);
    for (int k = 0; k < K; ++k) {
        w[k] = fast_exp(ln[k] * inv_t - vmax);
        wsum += w[k];
    }
    const scalar_t inv_wsum = scalar_t(1) / wsum;
    for (int k = 0; k < K; ++k) w[k] *= inv_wsum;

    // --- 4. Weighted sum for pixel hw ---
    const int h  = hw / W;
    const int ww = hw % W;
    const scalar_t* inp = input + b * C * HW + h * W + ww;
    scalar_t sum = scalar_t(0);
    for (int k = 0; k < K; ++k)
        sum += w[k] * inp[(int)idx[k] * HW];

    out[((b * E_total + e) * H + h) * W + ww] = sum;
}

// ============================================================
// K-parallel gate+route kernel.
// One block per output pixel (b, e, hw); blockDim.x == 32 (one warp).
// Threads stride over K (handles K > 32 via kk = k, k+32, ...).
// Uses __shfl_xor_sync butterfly all-reduces — zero shared memory.
// Pool is pre-computed by the caller (PyTorch adaptive_avg_pool2d).
// ============================================================
template <typename scalar_t>
__global__ void moce_gate_route_k_parallel_kernel(
    const scalar_t* __restrict__ input,       // (B, C, H, W)
    const scalar_t* __restrict__ pooled,      // (B, C) — pre-computed avg pool
    const int64_t*  __restrict__ indices,     // (E_routed, K)
    const scalar_t* __restrict__ logits_norm, // (E_routed, K)
    const scalar_t* __restrict__ gate_weight, // (K,)
    const scalar_t* __restrict__ gate_bias,   // (1,)  — device pointer
    float temp_min, float temp_max, float tau,
    int B, int C, int H, int W, int E_routed, int K, int E_total,
    scalar_t* __restrict__ out
) {
    int HW  = H * W;
    int bew = blockIdx.x;
    int hw  = bew % HW;
    int e   = (bew / HW) % E_routed;
    int b   = bew / (HW * E_routed);
    int k   = threadIdx.x;  // 0..31

    const int64_t* idx    = indices    + e * K;
    const scalar_t* lnorm = logits_norm + e * K;
    const scalar_t* pool_b = pooled + b * C;

    // --- Gate: strided dot product, warp all-reduce ---
    float gate_val = 0.0f;
    for (int kk = k; kk < K; kk += 32)
        gate_val += float(gate_weight[kk]) * float(pool_b[(int)idx[kk]]);
    for (int off = 16; off > 0; off >>= 1)
        gate_val += __shfl_xor_sync(0xffffffff, gate_val, off);
    // All threads now hold the full dot product
    float gate_logit = gate_val + float(gate_bias[0]);
    float sig = 1.0f / (1.0f + __expf(-gate_logit));
    float eff_temp = fmaxf((temp_min + (temp_max - temp_min) * sig) * tau, 1e-3f);
    float inv_temp = 1.0f / eff_temp;

    // --- Softmax: strided max, then strided exp-sum ---
    float max_v = -1e30f;
    for (int kk = k; kk < K; kk += 32)
        max_v = fmaxf(max_v, float(lnorm[kk]) * inv_temp);
    for (int off = 16; off > 0; off >>= 1)
        max_v = fmaxf(max_v, __shfl_xor_sync(0xffffffff, max_v, off));

    float wsum = 0.0f;
    for (int kk = k; kk < K; kk += 32)
        wsum += __expf(float(lnorm[kk]) * inv_temp - max_v);
    for (int off = 16; off > 0; off >>= 1)
        wsum += __shfl_xor_sync(0xffffffff, wsum, off);
    float inv_wsum = 1.0f / wsum;

    // --- Weighted sum: strided, warp all-reduce ---
    int input_base = b * C * HW + hw;
    float result = 0.0f;
    for (int kk = k; kk < K; kk += 32) {
        float w = __expf(float(lnorm[kk]) * inv_temp - max_v) * inv_wsum;
        result += w * float(input[input_base + (int)idx[kk] * HW]);
    }
    for (int off = 16; off > 0; off >>= 1)
        result += __shfl_xor_sync(0xffffffff, result, off);

    if (k == 0) {
        int h = hw / W, ww = hw % W;
        out[((b * E_total + e) * H + h) * W + ww] = scalar_t(result);
    }
}

// ============================================================
// Gate+route forward with pre-pooled input.
// Pool is done on the Python side (fast ATen op); this function
// handles gate logit → temperature → softmax → weighted sum → residual.
// Uses K-parallel kernel when HW<=16, HW-parallel otherwise.
// ============================================================
std::vector<torch::Tensor> fused_moce_gate_route_cuda(
    torch::Tensor input,
    torch::Tensor pooled,
    torch::Tensor indices,
    torch::Tensor logits_norm,
    torch::Tensor gate_weight,
    torch::Tensor gate_bias,
    torch::Tensor mask_int,
    double temp_min,
    double temp_max,
    double tau
) {
    CHECK_INPUT(input);
    CHECK_INPUT(pooled);
    CHECK_INPUT(indices);
    CHECK_INPUT(logits_norm);
    CHECK_INPUT(gate_weight);
    CHECK_INPUT(gate_bias);
    CHECK_INPUT(mask_int);
    TORCH_CHECK(input.scalar_type()       == torch::kFloat32, "input must be float32");
    TORCH_CHECK(pooled.scalar_type()      == torch::kFloat32, "pooled must be float32");
    TORCH_CHECK(logits_norm.scalar_type() == torch::kFloat32, "logits_norm must be float32");
    TORCH_CHECK(gate_weight.scalar_type() == torch::kFloat32, "gate_weight must be float32");
    TORCH_CHECK(gate_bias.scalar_type()   == torch::kFloat32, "gate_bias must be float32");
    TORCH_CHECK(mask_int.scalar_type()    == torch::kInt32,   "mask_int must be int32");

    const int B = input.size(0), C = input.size(1);
    const int H = input.size(2), W = input.size(3);
    const int E_routed = indices.size(0), K = indices.size(1);
    const int E_total  = E_routed + 1;
    const int HW = H * W;

    TORCH_CHECK(K <= 64, "fused_moce_gate_route: K must be <= 64");

    auto out = torch::empty({B, E_total, H, W}, input.options());

    const bool small_hw    = (HW <= 16);
    int routed_threads  = B * E_routed * HW;
    int residual_pixels = B * HW;
    dim3 routed_blocks  ((routed_threads  + THREADS - 1) / THREADS);
    dim3 residual_blocks((residual_pixels + THREADS - 1) / THREADS);

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_moce_gate_route_cuda", ([&] {
        if (small_hw) {
            moce_gate_route_k_parallel_kernel<scalar_t>
                <<<routed_threads, 32, 0, stream>>>(
                    input.data_ptr<scalar_t>(),
                    pooled.data_ptr<scalar_t>(),
                    indices.data_ptr<int64_t>(),
                    logits_norm.data_ptr<scalar_t>(),
                    gate_weight.data_ptr<scalar_t>(),
                    gate_bias.data_ptr<scalar_t>(),
                    (float)temp_min, (float)temp_max, (float)tau,
                    B, C, H, W, E_routed, K, E_total,
                    out.data_ptr<scalar_t>());
        } else {
            routed_forward_with_temp_kernel<scalar_t><<<routed_blocks, THREADS, 0, stream>>>(
                routed_threads,
                input.data_ptr<scalar_t>(),
                pooled.data_ptr<scalar_t>(),
                indices.data_ptr<int64_t>(),
                logits_norm.data_ptr<scalar_t>(),
                gate_weight.data_ptr<scalar_t>(),
                gate_bias.data_ptr<scalar_t>(),
                (scalar_t)temp_min, (scalar_t)temp_max, (scalar_t)tau,
                B, C, H, W, E_routed, K, E_total,
                out.data_ptr<scalar_t>());
        }

        if (small_hw) {
            residual_average_c_parallel_kernel<scalar_t>
                <<<residual_pixels, THREADS, 0, stream>>>(
                    input.data_ptr<scalar_t>(),
                    mask_int.data_ptr<int>(),
                    B, C, H, W, E_total, E_routed,
                    out.data_ptr<scalar_t>());
        } else {
            residual_average_kernel<scalar_t><<<residual_blocks, THREADS, 0, stream>>>(
                residual_pixels,
                input.data_ptr<scalar_t>(),
                mask_int.data_ptr<int>(),
                B, C, H, W, E_total, E_routed,
                out.data_ptr<scalar_t>());
        }
    }));

    return {out};
}

// ============================================================
// Timed variant of fused_moce_gate_route_cuda.
// Same kernels; wraps each with cudaEvents and returns timings.
// Returns {out, kernel_times} where kernel_times[0] = routed_ms,
//                                   kernel_times[1] = residual_ms.
// ============================================================
std::vector<torch::Tensor> fused_moce_gate_route_timed_cuda(
    torch::Tensor input,
    torch::Tensor pooled,
    torch::Tensor indices,
    torch::Tensor logits_norm,
    torch::Tensor gate_weight,
    torch::Tensor gate_bias,
    torch::Tensor mask_int,
    double temp_min,
    double temp_max,
    double tau
) {
    CHECK_INPUT(input);   CHECK_INPUT(pooled);    CHECK_INPUT(indices);
    CHECK_INPUT(logits_norm); CHECK_INPUT(gate_weight); CHECK_INPUT(gate_bias);
    CHECK_INPUT(mask_int);
    TORCH_CHECK(input.scalar_type()       == torch::kFloat32, "input must be float32");
    TORCH_CHECK(pooled.scalar_type()      == torch::kFloat32, "pooled must be float32");
    TORCH_CHECK(logits_norm.scalar_type() == torch::kFloat32, "logits_norm must be float32");
    TORCH_CHECK(gate_weight.scalar_type() == torch::kFloat32, "gate_weight must be float32");
    TORCH_CHECK(gate_bias.scalar_type()   == torch::kFloat32, "gate_bias must be float32");
    TORCH_CHECK(mask_int.scalar_type()    == torch::kInt32,   "mask_int must be int32");
    TORCH_CHECK(indices.size(1) <= 64, "K must be <= 64");

    const int B = input.size(0), C = input.size(1);
    const int H = input.size(2), W = input.size(3);
    const int E_routed = indices.size(0), K = indices.size(1);
    const int E_total  = E_routed + 1;
    const int HW = H * W;

    auto out = torch::empty({B, E_total, H, W}, input.options());

    const bool small_hw    = (HW <= 16);
    int routed_threads  = B * E_routed * HW;
    int residual_pixels = B * HW;
    dim3 routed_blocks  ((routed_threads  + THREADS - 1) / THREADS);
    dim3 residual_blocks((residual_pixels + THREADS - 1) / THREADS);

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    cudaEvent_t r0, r1, s0, s1;
    cudaEventCreate(&r0); cudaEventCreate(&r1);
    cudaEventCreate(&s0); cudaEventCreate(&s1);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_moce_gate_route_timed_cuda", ([&] {
        cudaEventRecord(r0, stream);
        if (small_hw) {
            moce_gate_route_k_parallel_kernel<scalar_t>
                <<<routed_threads, 32, 0, stream>>>(
                    input.data_ptr<scalar_t>(), pooled.data_ptr<scalar_t>(),
                    indices.data_ptr<int64_t>(), logits_norm.data_ptr<scalar_t>(),
                    gate_weight.data_ptr<scalar_t>(), gate_bias.data_ptr<scalar_t>(),
                    (float)temp_min, (float)temp_max, (float)tau,
                    B, C, H, W, E_routed, K, E_total, out.data_ptr<scalar_t>());
        } else {
            routed_forward_with_temp_kernel<scalar_t><<<routed_blocks, THREADS, 0, stream>>>(
                routed_threads, input.data_ptr<scalar_t>(), pooled.data_ptr<scalar_t>(),
                indices.data_ptr<int64_t>(), logits_norm.data_ptr<scalar_t>(),
                gate_weight.data_ptr<scalar_t>(), gate_bias.data_ptr<scalar_t>(),
                (scalar_t)temp_min, (scalar_t)temp_max, (scalar_t)tau,
                B, C, H, W, E_routed, K, E_total, out.data_ptr<scalar_t>());
        }
        cudaEventRecord(r1, stream);

        cudaEventRecord(s0, stream);
        if (small_hw) {
            residual_average_c_parallel_kernel<scalar_t>
                <<<residual_pixels, THREADS, 0, stream>>>(
                    input.data_ptr<scalar_t>(), mask_int.data_ptr<int>(),
                    B, C, H, W, E_total, E_routed, out.data_ptr<scalar_t>());
        } else {
            residual_average_kernel<scalar_t><<<residual_blocks, THREADS, 0, stream>>>(
                residual_pixels, input.data_ptr<scalar_t>(), mask_int.data_ptr<int>(),
                B, C, H, W, E_total, E_routed, out.data_ptr<scalar_t>());
        }
        cudaEventRecord(s1, stream);
    }));

    cudaEventSynchronize(s1);
    float routed_ms = 0, residual_ms = 0;
    cudaEventElapsedTime(&routed_ms,  r0, r1);
    cudaEventElapsedTime(&residual_ms, s0, s1);
    cudaEventDestroy(r0); cudaEventDestroy(r1);
    cudaEventDestroy(s0); cudaEventDestroy(s1);

    auto ktimes = torch::tensor({routed_ms, residual_ms});
    return {out, ktimes};
}

// ============================================================
// Section-timed kernels: clock64() timestamps at 5 points
// → 4 section cycle-counts written to section_clocks[4]
// Only the (block 0, thread 0) writes; all other threads
// compute the same output as the untimed variants above.
// Sections: [0]=gate_dot  [1]=temperature  [2]=softmax  [3]=wsum
// ============================================================
template <typename scalar_t>
__global__ void routed_forward_with_temp_section_timed_kernel(
    int nthreads,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ pooled,
    const int64_t*  __restrict__ indices,
    const scalar_t* __restrict__ logits_norm,
    const scalar_t* __restrict__ gate_weight,
    const scalar_t* __restrict__ gate_bias,
    scalar_t temp_min, scalar_t temp_max, scalar_t tau,
    int B, int C, int H, int W,
    int E_routed, int K, int E_total,
    scalar_t* __restrict__ out,
    long long* __restrict__ section_clocks
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads) return;

    const int HW = H * W;
    const int hw = tid % HW;
    const int e  = (tid / HW) % E_routed;
    const int b  = tid / (E_routed * HW);

    const int64_t* idx     = indices     + e * K;
    const scalar_t* pool_b = pooled      + b * C;
    const scalar_t* ln     = logits_norm + e * K;

    const bool is_timer = (blockIdx.x == 0 && threadIdx.x == 0);
    long long t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0;
    if (is_timer) t0 = clock64();

    // --- 1. Gate logit ---
    scalar_t gate_logit = gate_bias[0];
    #pragma unroll
    for (int k = 0; k < K; ++k)
        gate_logit += pool_b[(int)idx[k]] * gate_weight[k];

    if (is_timer) t1 = clock64();

    // --- 2. Temperature ---
    scalar_t t = (temp_min + (temp_max - temp_min) * sigmoid_cuda(gate_logit)) * tau;
    if (t < scalar_t(1e-3)) t = scalar_t(1e-3);
    const scalar_t inv_t = scalar_t(1) / t;

    if (is_timer) t2 = clock64();

    // --- 3. Softmax ---
    scalar_t w[64];
    scalar_t vmax = ln[0] * inv_t;
    for (int k = 1; k < K; ++k) {
        scalar_t v = ln[k] * inv_t;
        if (v > vmax) vmax = v;
    }
    scalar_t wsum = scalar_t(0);
    for (int k = 0; k < K; ++k) {
        w[k] = fast_exp(ln[k] * inv_t - vmax);
        wsum += w[k];
    }
    const scalar_t inv_wsum = scalar_t(1) / wsum;
    for (int k = 0; k < K; ++k) w[k] *= inv_wsum;

    if (is_timer) t3 = clock64();

    // --- 4. Weighted sum ---
    const int h  = hw / W;
    const int ww = hw % W;
    const scalar_t* inp = input + b * C * HW + h * W + ww;
    scalar_t sum = scalar_t(0);
    for (int k = 0; k < K; ++k)
        sum += w[k] * inp[(int)idx[k] * HW];

    out[((b * E_total + e) * H + h) * W + ww] = sum;

    if (is_timer) {
        t4 = clock64();
        section_clocks[0] = t1 - t0;
        section_clocks[1] = t2 - t1;
        section_clocks[2] = t3 - t2;
        section_clocks[3] = t4 - t3;
    }
}

template <typename scalar_t>
__global__ void moce_gate_route_k_parallel_section_timed_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ pooled,
    const int64_t*  __restrict__ indices,
    const scalar_t* __restrict__ logits_norm,
    const scalar_t* __restrict__ gate_weight,
    const scalar_t* __restrict__ gate_bias,
    float temp_min, float temp_max, float tau,
    int B, int C, int H, int W, int E_routed, int K, int E_total,
    scalar_t* __restrict__ out,
    long long* __restrict__ section_clocks
) {
    int HW  = H * W;
    int bew = blockIdx.x;
    int hw  = bew % HW;
    int e   = (bew / HW) % E_routed;
    int b   = bew / (HW * E_routed);
    int k   = threadIdx.x;

    const int64_t* idx     = indices     + e * K;
    const scalar_t* lnorm  = logits_norm + e * K;
    const scalar_t* pool_b = pooled      + b * C;

    const bool is_timer = (blockIdx.x == 0 && k == 0);
    long long t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0;
    if (is_timer) t0 = clock64();

    // --- Gate ---
    float gate_val = 0.0f;
    for (int kk = k; kk < K; kk += 32)
        gate_val += float(gate_weight[kk]) * float(pool_b[(int)idx[kk]]);
    for (int off = 16; off > 0; off >>= 1)
        gate_val += __shfl_xor_sync(0xffffffff, gate_val, off);
    float gate_logit = gate_val + float(gate_bias[0]);

    if (is_timer) t1 = clock64();

    // --- Temperature ---
    float sig      = 1.0f / (1.0f + __expf(-gate_logit));
    float eff_temp = fmaxf((temp_min + (temp_max - temp_min) * sig) * tau, 1e-3f);
    float inv_temp = 1.0f / eff_temp;

    if (is_timer) t2 = clock64();

    // --- Softmax ---
    float max_v = -1e30f;
    for (int kk = k; kk < K; kk += 32)
        max_v = fmaxf(max_v, float(lnorm[kk]) * inv_temp);
    for (int off = 16; off > 0; off >>= 1)
        max_v = fmaxf(max_v, __shfl_xor_sync(0xffffffff, max_v, off));
    float wsum = 0.0f;
    for (int kk = k; kk < K; kk += 32)
        wsum += __expf(float(lnorm[kk]) * inv_temp - max_v);
    for (int off = 16; off > 0; off >>= 1)
        wsum += __shfl_xor_sync(0xffffffff, wsum, off);
    float inv_wsum = 1.0f / wsum;

    if (is_timer) t3 = clock64();

    // --- Weighted sum ---
    int input_base = b * C * HW + hw;
    float result = 0.0f;
    for (int kk = k; kk < K; kk += 32) {
        float w = __expf(float(lnorm[kk]) * inv_temp - max_v) * inv_wsum;
        result += w * float(input[input_base + (int)idx[kk] * HW]);
    }
    for (int off = 16; off > 0; off >>= 1)
        result += __shfl_xor_sync(0xffffffff, result, off);

    if (k == 0) {
        int h = hw / W, ww = hw % W;
        out[((b * E_total + e) * H + h) * W + ww] = scalar_t(result);
    }

    if (is_timer) {
        t4 = clock64();
        section_clocks[0] = t1 - t0;
        section_clocks[1] = t2 - t1;
        section_clocks[2] = t3 - t2;
        section_clocks[3] = t4 - t3;
    }
}

// ============================================================
// Section-timed gate+route: returns (out, section_us[4])
// section_us = [gate_dot_us, temperature_us, softmax_us, wsum_us]
// Uses clock64() on (block 0, thread 0) inside the routed kernel.
// ============================================================
std::vector<torch::Tensor> fused_moce_gate_route_section_timed_cuda(
    torch::Tensor input, torch::Tensor pooled, torch::Tensor indices,
    torch::Tensor logits_norm, torch::Tensor gate_weight, torch::Tensor gate_bias,
    torch::Tensor mask_int, double temp_min, double temp_max, double tau
) {
    CHECK_INPUT(input);   CHECK_INPUT(pooled);    CHECK_INPUT(indices);
    CHECK_INPUT(logits_norm); CHECK_INPUT(gate_weight); CHECK_INPUT(gate_bias);
    CHECK_INPUT(mask_int);
    TORCH_CHECK(input.scalar_type()       == torch::kFloat32, "input must be float32");
    TORCH_CHECK(pooled.scalar_type()      == torch::kFloat32, "pooled must be float32");
    TORCH_CHECK(logits_norm.scalar_type() == torch::kFloat32, "logits_norm must be float32");
    TORCH_CHECK(gate_weight.scalar_type() == torch::kFloat32, "gate_weight must be float32");
    TORCH_CHECK(gate_bias.scalar_type()   == torch::kFloat32, "gate_bias must be float32");
    TORCH_CHECK(mask_int.scalar_type()    == torch::kInt32,   "mask_int must be int32");
    TORCH_CHECK(indices.size(1) <= 64,                        "K must be <= 64");

    const int B = input.size(0), C = input.size(1);
    const int H = input.size(2), W = input.size(3);
    const int E_routed = indices.size(0), K = indices.size(1);
    const int E_total  = E_routed + 1;
    const int HW = H * W;

    auto out = torch::empty({B, E_total, H, W}, input.options());

    // Device buffer for section cycle counts
    auto clocks_d = torch::zeros({4},
        torch::TensorOptions().dtype(torch::kInt64).device(input.device()));
    long long* clocks_ptr = reinterpret_cast<long long*>(clocks_d.data_ptr<int64_t>());

    const bool small_hw    = (HW <= 16);
    int routed_threads  = B * E_routed * HW;
    int residual_pixels = B * HW;
    dim3 routed_blocks  ((routed_threads  + THREADS - 1) / THREADS);
    dim3 residual_blocks((residual_pixels + THREADS - 1) / THREADS);

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_moce_gate_route_section_timed_cuda", ([&] {
        if (small_hw) {
            moce_gate_route_k_parallel_section_timed_kernel<scalar_t>
                <<<routed_threads, 32, 0, stream>>>(
                    input.data_ptr<scalar_t>(), pooled.data_ptr<scalar_t>(),
                    indices.data_ptr<int64_t>(), logits_norm.data_ptr<scalar_t>(),
                    gate_weight.data_ptr<scalar_t>(), gate_bias.data_ptr<scalar_t>(),
                    (float)temp_min, (float)temp_max, (float)tau,
                    B, C, H, W, E_routed, K, E_total,
                    out.data_ptr<scalar_t>(), clocks_ptr);
        } else {
            routed_forward_with_temp_section_timed_kernel<scalar_t>
                <<<routed_blocks, THREADS, 0, stream>>>(
                    routed_threads, input.data_ptr<scalar_t>(), pooled.data_ptr<scalar_t>(),
                    indices.data_ptr<int64_t>(), logits_norm.data_ptr<scalar_t>(),
                    gate_weight.data_ptr<scalar_t>(), gate_bias.data_ptr<scalar_t>(),
                    (scalar_t)temp_min, (scalar_t)temp_max, (scalar_t)tau,
                    B, C, H, W, E_routed, K, E_total,
                    out.data_ptr<scalar_t>(), clocks_ptr);
        }

        if (small_hw) {
            residual_average_c_parallel_kernel<scalar_t>
                <<<residual_pixels, THREADS, 0, stream>>>(
                    input.data_ptr<scalar_t>(), mask_int.data_ptr<int>(),
                    B, C, H, W, E_total, E_routed, out.data_ptr<scalar_t>());
        } else {
            residual_average_kernel<scalar_t><<<residual_blocks, THREADS, 0, stream>>>(
                residual_pixels, input.data_ptr<scalar_t>(), mask_int.data_ptr<int>(),
                B, C, H, W, E_total, E_routed, out.data_ptr<scalar_t>());
        }
    }));

    cudaDeviceSynchronize();

    // Convert cycles → µs using the SM clock rate
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, input.device().index());
    double cycles_per_us = prop.clockRate / 1000.0;  // clockRate in kHz → cycles/µs

    auto clocks_h = clocks_d.cpu();
    const int64_t* cp = clocks_h.data_ptr<int64_t>();
    auto section_us = torch::zeros({4});
    for (int i = 0; i < 4; ++i)
        section_us[i] = float(cp[i] / cycles_per_us);

    return {out, section_us};
}

// ============================================================
// Standalone pool: returns pooled (B, C) using channel_avg_pool_warp_kernel.
// Called from Python when temperature is computed on the Python side.
// ============================================================
std::vector<torch::Tensor> moce_pool_cuda(torch::Tensor input) {
    CHECK_INPUT(input);
    const int B = input.size(0), C = input.size(1);
    const int H = input.size(2), W = input.size(3);
    const int HW = H * W;
    auto pooled = torch::empty({B, C}, input.options());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moce_pool_cuda", ([&] {
        channel_avg_pool_warp_kernel<scalar_t><<<B * C, 32, 0, stream>>>(
            input.data_ptr<scalar_t>(), pooled.data_ptr<scalar_t>(), HW);
    }));
    return {pooled};
}

// ============================================================
// All-in-CUDA gate+route: pool done inside CUDA (no Python pool).
// Uses channel_avg_pool_warp_kernel → routed kernel → residual kernel.
// ============================================================
std::vector<torch::Tensor> fused_moce_gate_route_all_cuda(
    torch::Tensor input,
    torch::Tensor indices,
    torch::Tensor logits_norm,
    torch::Tensor gate_weight,
    torch::Tensor gate_bias,
    torch::Tensor mask_int,
    double temp_min,
    double temp_max,
    double tau
) {
    CHECK_INPUT(input);   CHECK_INPUT(indices);
    CHECK_INPUT(logits_norm); CHECK_INPUT(gate_weight); CHECK_INPUT(gate_bias);
    CHECK_INPUT(mask_int);
    TORCH_CHECK(input.scalar_type()       == torch::kFloat32, "input must be float32");
    TORCH_CHECK(logits_norm.scalar_type() == torch::kFloat32, "logits_norm must be float32");
    TORCH_CHECK(gate_weight.scalar_type() == torch::kFloat32, "gate_weight must be float32");
    TORCH_CHECK(gate_bias.scalar_type()   == torch::kFloat32, "gate_bias must be float32");
    TORCH_CHECK(mask_int.scalar_type()    == torch::kInt32,   "mask_int must be int32");
    TORCH_CHECK(indices.size(1) <= 64,                        "K must be <= 64");

    const int B = input.size(0), C = input.size(1);
    const int H = input.size(2), W = input.size(3);
    const int E_routed = indices.size(0), K = indices.size(1);
    const int E_total  = E_routed + 1;
    const int HW = H * W;

    auto out    = torch::empty({B, E_total, H, W}, input.options());
    auto pooled = torch::empty({B, C}, input.options());

    const bool small_hw    = (HW <= 16);
    int routed_threads  = B * E_routed * HW;
    int residual_pixels = B * HW;
    dim3 routed_blocks  ((routed_threads  + THREADS - 1) / THREADS);
    dim3 residual_blocks((residual_pixels + THREADS - 1) / THREADS);

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_moce_gate_route_all_cuda", ([&] {
        // 1. Pool: one warp per (b,c)
        channel_avg_pool_warp_kernel<scalar_t><<<B * C, 32, 0, stream>>>(
            input.data_ptr<scalar_t>(), pooled.data_ptr<scalar_t>(), HW);

        // 2. Routed kernel
        if (small_hw) {
            moce_gate_route_k_parallel_kernel<scalar_t>
                <<<routed_threads, 32, 0, stream>>>(
                    input.data_ptr<scalar_t>(), pooled.data_ptr<scalar_t>(),
                    indices.data_ptr<int64_t>(), logits_norm.data_ptr<scalar_t>(),
                    gate_weight.data_ptr<scalar_t>(), gate_bias.data_ptr<scalar_t>(),
                    (float)temp_min, (float)temp_max, (float)tau,
                    B, C, H, W, E_routed, K, E_total, out.data_ptr<scalar_t>());
        } else {
            routed_forward_with_temp_kernel<scalar_t><<<routed_blocks, THREADS, 0, stream>>>(
                routed_threads, input.data_ptr<scalar_t>(), pooled.data_ptr<scalar_t>(),
                indices.data_ptr<int64_t>(), logits_norm.data_ptr<scalar_t>(),
                gate_weight.data_ptr<scalar_t>(), gate_bias.data_ptr<scalar_t>(),
                (scalar_t)temp_min, (scalar_t)temp_max, (scalar_t)tau,
                B, C, H, W, E_routed, K, E_total, out.data_ptr<scalar_t>());
        }

        // 3. Residual
        if (small_hw) {
            residual_average_c_parallel_kernel<scalar_t>
                <<<residual_pixels, THREADS, 0, stream>>>(
                    input.data_ptr<scalar_t>(), mask_int.data_ptr<int>(),
                    B, C, H, W, E_total, E_routed, out.data_ptr<scalar_t>());
        } else {
            residual_average_kernel<scalar_t><<<residual_blocks, THREADS, 0, stream>>>(
                residual_pixels, input.data_ptr<scalar_t>(), mask_int.data_ptr<int>(),
                B, C, H, W, E_total, E_routed, out.data_ptr<scalar_t>());
        }
    }));

    return {out};
}

// ============================================================
// All-in-CUDA timed: cudaEvents around each of the 3 kernels +
// clock64 sections inside the routed kernel.
// Returns {out, kernel_times[3], section_us[4]}
//   kernel_times = [pool_ms, routed_ms, residual_ms]
//   section_us   = [gate_us, temp_us, softmax_us, wsum_us]
// ============================================================
std::vector<torch::Tensor> fused_moce_gate_route_all_timed_cuda(
    torch::Tensor input,
    torch::Tensor indices,
    torch::Tensor logits_norm,
    torch::Tensor gate_weight,
    torch::Tensor gate_bias,
    torch::Tensor mask_int,
    double temp_min,
    double temp_max,
    double tau
) {
    CHECK_INPUT(input);   CHECK_INPUT(indices);
    CHECK_INPUT(logits_norm); CHECK_INPUT(gate_weight); CHECK_INPUT(gate_bias);
    CHECK_INPUT(mask_int);
    TORCH_CHECK(input.scalar_type()       == torch::kFloat32, "input must be float32");
    TORCH_CHECK(logits_norm.scalar_type() == torch::kFloat32, "logits_norm must be float32");
    TORCH_CHECK(gate_weight.scalar_type() == torch::kFloat32, "gate_weight must be float32");
    TORCH_CHECK(gate_bias.scalar_type()   == torch::kFloat32, "gate_bias must be float32");
    TORCH_CHECK(mask_int.scalar_type()    == torch::kInt32,   "mask_int must be int32");
    TORCH_CHECK(indices.size(1) <= 64,                        "K must be <= 64");

    const int B = input.size(0), C = input.size(1);
    const int H = input.size(2), W = input.size(3);
    const int E_routed = indices.size(0), K = indices.size(1);
    const int E_total  = E_routed + 1;
    const int HW = H * W;

    auto out    = torch::empty({B, E_total, H, W}, input.options());
    auto pooled = torch::empty({B, C}, input.options());
    auto clocks_d = torch::zeros({4},
        torch::TensorOptions().dtype(torch::kInt64).device(input.device()));
    long long* clocks_ptr = reinterpret_cast<long long*>(clocks_d.data_ptr<int64_t>());

    const bool small_hw    = (HW <= 16);
    int routed_threads  = B * E_routed * HW;
    int residual_pixels = B * HW;
    dim3 routed_blocks  ((routed_threads  + THREADS - 1) / THREADS);
    dim3 residual_blocks((residual_pixels + THREADS - 1) / THREADS);

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    cudaEvent_t p0, p1, r0, r1, s0, s1;
    cudaEventCreate(&p0); cudaEventCreate(&p1);
    cudaEventCreate(&r0); cudaEventCreate(&r1);
    cudaEventCreate(&s0); cudaEventCreate(&s1);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_moce_gate_route_all_timed_cuda", ([&] {
        // Pool
        cudaEventRecord(p0, stream);
        channel_avg_pool_warp_kernel<scalar_t><<<B * C, 32, 0, stream>>>(
            input.data_ptr<scalar_t>(), pooled.data_ptr<scalar_t>(), HW);
        cudaEventRecord(p1, stream);

        // Routed (section-timed variant writes clock64 to clocks_ptr)
        cudaEventRecord(r0, stream);
        if (small_hw) {
            moce_gate_route_k_parallel_section_timed_kernel<scalar_t>
                <<<routed_threads, 32, 0, stream>>>(
                    input.data_ptr<scalar_t>(), pooled.data_ptr<scalar_t>(),
                    indices.data_ptr<int64_t>(), logits_norm.data_ptr<scalar_t>(),
                    gate_weight.data_ptr<scalar_t>(), gate_bias.data_ptr<scalar_t>(),
                    (float)temp_min, (float)temp_max, (float)tau,
                    B, C, H, W, E_routed, K, E_total,
                    out.data_ptr<scalar_t>(), clocks_ptr);
        } else {
            routed_forward_with_temp_section_timed_kernel<scalar_t>
                <<<routed_blocks, THREADS, 0, stream>>>(
                    routed_threads, input.data_ptr<scalar_t>(), pooled.data_ptr<scalar_t>(),
                    indices.data_ptr<int64_t>(), logits_norm.data_ptr<scalar_t>(),
                    gate_weight.data_ptr<scalar_t>(), gate_bias.data_ptr<scalar_t>(),
                    (scalar_t)temp_min, (scalar_t)temp_max, (scalar_t)tau,
                    B, C, H, W, E_routed, K, E_total,
                    out.data_ptr<scalar_t>(), clocks_ptr);
        }
        cudaEventRecord(r1, stream);

        // Residual
        cudaEventRecord(s0, stream);
        if (small_hw) {
            residual_average_c_parallel_kernel<scalar_t>
                <<<residual_pixels, THREADS, 0, stream>>>(
                    input.data_ptr<scalar_t>(), mask_int.data_ptr<int>(),
                    B, C, H, W, E_total, E_routed, out.data_ptr<scalar_t>());
        } else {
            residual_average_kernel<scalar_t><<<residual_blocks, THREADS, 0, stream>>>(
                residual_pixels, input.data_ptr<scalar_t>(), mask_int.data_ptr<int>(),
                B, C, H, W, E_total, E_routed, out.data_ptr<scalar_t>());
        }
        cudaEventRecord(s1, stream);
    }));

    cudaEventSynchronize(s1);
    float pool_ms = 0, routed_ms = 0, residual_ms = 0;
    cudaEventElapsedTime(&pool_ms,    p0, p1);
    cudaEventElapsedTime(&routed_ms,  r0, r1);
    cudaEventElapsedTime(&residual_ms, s0, s1);
    cudaEventDestroy(p0); cudaEventDestroy(p1);
    cudaEventDestroy(r0); cudaEventDestroy(r1);
    cudaEventDestroy(s0); cudaEventDestroy(s1);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, input.device().index());
    double cycles_per_us = prop.clockRate / 1000.0;

    auto clocks_h = clocks_d.cpu();
    const int64_t* cp = clocks_h.data_ptr<int64_t>();
    auto section_us = torch::zeros({4});
    for (int i = 0; i < 4; ++i)
        section_us[i] = float(cp[i] / cycles_per_us);

    auto ktimes = torch::tensor({pool_ms, routed_ms, residual_ms});
    return {out, ktimes, section_us};
}

// ============================================================
// Dynamic-temp forward (thread-based).
// Per-sample weights (B, E_routed, K) — computed on the Python side
// from the gate output for each eval batch.
// Residual still uses the fixed mask_int (indices are input-independent).
//
// Inputs:
//   input    : (B, C, H, W)          float32
//   indices  : (E_routed, K)          int64   — top-K channel indices per expert
//   weights  : (B, E_routed, K)       float32 — per-sample softmax weights
//   mask_int : (C,)                   int32   — 1=unselected channel, 0=selected
//
// Returns: {out}  shape (B, E_total, H, W)
// ============================================================
std::vector<torch::Tensor> fused_moce_forward_dynamic_temp_cuda(
    torch::Tensor input,
    torch::Tensor indices,
    torch::Tensor weights,
    torch::Tensor mask_int
) {
    CHECK_INPUT(input);
    CHECK_INPUT(indices);
    CHECK_INPUT(weights);
    CHECK_INPUT(mask_int);
    TORCH_CHECK(input.scalar_type()    == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weights.scalar_type()  == torch::kFloat32, "weights must be float32");
    TORCH_CHECK(mask_int.scalar_type() == torch::kInt32,   "mask_int must be int32");

    const int B = input.size(0), C = input.size(1);
    const int H = input.size(2), W = input.size(3);
    const int E_routed = indices.size(0), K = indices.size(1);
    const int E_total  = E_routed + 1;

    TORCH_CHECK(weights.size(0) == B && weights.size(1) == E_routed && weights.size(2) == K,
                "weights must be (B, E_routed, K)");

    auto out = torch::empty({B, E_total, H, W}, input.options());

    const int HW = H * W;
    // Switch to K-parallel / C-parallel when HW<=16 (layers 3/4).
    // For K=8 on large HW the original kernel is already well-utilized.
    // k_block=32 for K<=32 (single warp, no shared mem); 64 for K>32.
    const bool small_hw = (HW <= 16);
    const int  k_block  = (K <= 32) ? 32 : 64;

    int routed_threads  = B * E_routed * HW;
    int residual_pixels = B * HW;

    dim3 routed_blocks  ((routed_threads  + THREADS - 1) / THREADS);
    dim3 residual_blocks((residual_pixels + THREADS - 1) / THREADS);

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_moce_forward_dynamic_temp_cuda", ([&] {
        if (small_hw) {
            routed_forward_dynamic_temp_k_parallel_kernel<scalar_t>
                <<<routed_threads, k_block, 0, stream>>>(
                    input.data_ptr<scalar_t>(),
                    indices.data_ptr<int64_t>(),
                    weights.data_ptr<scalar_t>(),
                    B, C, H, W, E_routed, K, E_total,
                    out.data_ptr<scalar_t>());

            residual_average_c_parallel_kernel<scalar_t>
                <<<residual_pixels, THREADS, 0, stream>>>(
                    input.data_ptr<scalar_t>(),
                    mask_int.data_ptr<int>(),
                    B, C, H, W, E_total, E_routed,
                    out.data_ptr<scalar_t>());
        } else {
            routed_forward_dynamic_temp_kernel<scalar_t><<<routed_blocks, THREADS, 0, stream>>>(
                routed_threads,
                input.data_ptr<scalar_t>(),
                indices.data_ptr<int64_t>(),
                weights.data_ptr<scalar_t>(),
                B, C, H, W, E_routed, K, E_total,
                out.data_ptr<scalar_t>());

            residual_average_kernel<scalar_t><<<residual_blocks, THREADS, 0, stream>>>(
                residual_pixels,
                input.data_ptr<scalar_t>(),
                mask_int.data_ptr<int>(),
                B, C, H, W, E_total, E_routed,
                out.data_ptr<scalar_t>());
        }
    }));

    return {out};
}
