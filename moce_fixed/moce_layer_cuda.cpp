#include <torch/extension.h>
#include <vector>

// ── Try 1: pool in CUDA, temperature in torch, wsum+residual in CUDA ──────────

std::vector<torch::Tensor> moce_pool_cuda(torch::Tensor input);

std::vector<torch::Tensor> moce_pool(torch::Tensor input) {
    return moce_pool_cuda(input);
}

std::vector<torch::Tensor> fused_moce_forward_dynamic_temp_cuda(
    torch::Tensor input,
    torch::Tensor indices,
    torch::Tensor weights,
    torch::Tensor mask_int
);

std::vector<torch::Tensor> fused_moce_forward_dynamic_temp(
    torch::Tensor input,
    torch::Tensor indices,
    torch::Tensor weights,
    torch::Tensor mask_int
) {
    return fused_moce_forward_dynamic_temp_cuda(input, indices, weights, mask_int);
}

// ── Try 2: pool in Python ATen, gate+temp+softmax+wsum in CUDA ────────────────

std::vector<torch::Tensor> fused_moce_gate_route_cuda(
    torch::Tensor input, torch::Tensor pooled, torch::Tensor indices,
    torch::Tensor logits_norm, torch::Tensor gate_weight, torch::Tensor gate_bias,
    torch::Tensor mask_int, double temp_min, double temp_max, double tau
);

std::vector<torch::Tensor> fused_moce_gate_route(
    torch::Tensor input, torch::Tensor pooled, torch::Tensor indices,
    torch::Tensor logits_norm, torch::Tensor gate_weight, torch::Tensor gate_bias,
    torch::Tensor mask_int, double temp_min, double temp_max, double tau
) {
    return fused_moce_gate_route_cuda(
        input, pooled, indices, logits_norm, gate_weight, gate_bias, mask_int,
        temp_min, temp_max, tau
    );
}

std::vector<torch::Tensor> fused_moce_gate_route_timed_cuda(
    torch::Tensor input, torch::Tensor pooled, torch::Tensor indices,
    torch::Tensor logits_norm, torch::Tensor gate_weight, torch::Tensor gate_bias,
    torch::Tensor mask_int, double temp_min, double temp_max, double tau
);

std::vector<torch::Tensor> fused_moce_gate_route_timed(
    torch::Tensor input, torch::Tensor pooled, torch::Tensor indices,
    torch::Tensor logits_norm, torch::Tensor gate_weight, torch::Tensor gate_bias,
    torch::Tensor mask_int, double temp_min, double temp_max, double tau
) {
    return fused_moce_gate_route_timed_cuda(
        input, pooled, indices, logits_norm, gate_weight, gate_bias, mask_int,
        temp_min, temp_max, tau
    );
}

std::vector<torch::Tensor> fused_moce_gate_route_section_timed_cuda(
    torch::Tensor input, torch::Tensor pooled, torch::Tensor indices,
    torch::Tensor logits_norm, torch::Tensor gate_weight, torch::Tensor gate_bias,
    torch::Tensor mask_int, double temp_min, double temp_max, double tau
);

std::vector<torch::Tensor> fused_moce_gate_route_section_timed(
    torch::Tensor input, torch::Tensor pooled, torch::Tensor indices,
    torch::Tensor logits_norm, torch::Tensor gate_weight, torch::Tensor gate_bias,
    torch::Tensor mask_int, double temp_min, double temp_max, double tau
) {
    return fused_moce_gate_route_section_timed_cuda(
        input, pooled, indices, logits_norm, gate_weight, gate_bias, mask_int,
        temp_min, temp_max, tau
    );
}

// ── Try 3: pool + gate + temp + softmax + wsum all in CUDA ────────────────────

std::vector<torch::Tensor> fused_moce_gate_route_all_cuda(
    torch::Tensor input, torch::Tensor indices,
    torch::Tensor logits_norm, torch::Tensor gate_weight, torch::Tensor gate_bias,
    torch::Tensor mask_int, double temp_min, double temp_max, double tau
);

std::vector<torch::Tensor> fused_moce_gate_route_all(
    torch::Tensor input, torch::Tensor indices,
    torch::Tensor logits_norm, torch::Tensor gate_weight, torch::Tensor gate_bias,
    torch::Tensor mask_int, double temp_min, double temp_max, double tau
) {
    return fused_moce_gate_route_all_cuda(
        input, indices, logits_norm, gate_weight, gate_bias, mask_int,
        temp_min, temp_max, tau
    );
}

std::vector<torch::Tensor> fused_moce_gate_route_all_timed_cuda(
    torch::Tensor input, torch::Tensor indices,
    torch::Tensor logits_norm, torch::Tensor gate_weight, torch::Tensor gate_bias,
    torch::Tensor mask_int, double temp_min, double temp_max, double tau
);

std::vector<torch::Tensor> fused_moce_gate_route_all_timed(
    torch::Tensor input, torch::Tensor indices,
    torch::Tensor logits_norm, torch::Tensor gate_weight, torch::Tensor gate_bias,
    torch::Tensor mask_int, double temp_min, double temp_max, double tau
) {
    return fused_moce_gate_route_all_timed_cuda(
        input, indices, logits_norm, gate_weight, gate_bias, mask_int,
        temp_min, temp_max, tau
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Try 1
    m.def("moce_pool", &moce_pool,
          "Channel avg pool (B,C,H,W)→(B,C) via warp-per-channel CUDA kernel");
    m.def("fused_moce_forward_dynamic_temp", &fused_moce_forward_dynamic_temp,
          "Weighted sum + residual; weights (B,E,K) pre-computed in Python; returns (out,)");

    // Try 2
    m.def("fused_moce_gate_route", &fused_moce_gate_route,
          "Pool in Python, gate+temp+softmax+wsum in CUDA; returns (out,)");
    m.def("fused_moce_gate_route_timed", &fused_moce_gate_route_timed,
          "Timed Try-2; returns (out, kernel_times[routed_ms, residual_ms])");
    m.def("fused_moce_gate_route_section_timed", &fused_moce_gate_route_section_timed,
          "Section-timed Try-2 (clock64); returns (out, section_us[gate,temp,softmax,wsum])");

    // Try 3
    m.def("fused_moce_gate_route_all", &fused_moce_gate_route_all,
          "Pool+gate+temp+softmax+wsum all in CUDA; returns (out,)");
    m.def("fused_moce_gate_route_all_timed", &fused_moce_gate_route_all_timed,
          "Timed Try-3; returns (out, kernel_times[pool_ms,routed_ms,residual_ms], "
          "section_us[gate,temp,softmax,wsum])");
}
