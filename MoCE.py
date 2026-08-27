import math
import torch
import torch.nn as nn
import torch.nn.functional as F


import moce_fixed.moce_cuda



class MoCE(nn.Module):
    """
    MoCE without residual temperature.

    Flow:
    - E-1 routed experts:
        selected channels -> routed temperature -> softmax weights -> weighted sum

    - 1 residual expert:
        all unselected channels -> simple average

    This matches the PiXMoE-style residual flow:
        residual = average(unselected channels)
    """

    def __init__(
        self,
        in_channels=256,
        sampling_factor=4,
        k=8,
        tau=1.0,
        balance_weight=0.05,
        specialization_weight=0.1,
        diversity_weight=0.01,
        temp_min=0.2,
        temp_max=5.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_experts = in_channels // sampling_factor
        self.k = k
        self.tau = tau

        self.balance_weight = balance_weight
        self.specialization_weight = specialization_weight
        self.diversity_weight = diversity_weight

        self.temp_min = temp_min
        self.temp_max = temp_max

        self._last_temperature = None
        self.aux_loss = 0.0

        routed_experts = max(self.num_experts - 1, 1)
        self.target_experts_per_channel = (routed_experts * self.k) / max(
            float(self.in_channels), 1.0
        )

        # Static learned routing logits: (E-1, C)
        self.routing_logits = nn.Parameter(
            torch.randn(self.num_experts - 1, in_channels)
        )

        # Routed expert temperature gate
        self.gate = nn.Linear(k, 1)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)


        # Eval cache (buffers)
        self.register_buffer("cached_indices", None)
        self.register_buffer("cached_selected_logits_norm", None)
        self.register_buffer("cached_mask_int", None, persistent=False)
        # Eval cache (non-tensor scalars + gate copies, set by build_eval_cache)
        self.cached_temp_min_tau   = None
        self.cached_temp_range_tau = None
        self.cached_gate_weight    = None
        self.cached_gate_bias      = None
    # ------------------------------------------------------------------
    def forward(self, x):
        return self.forward_train(x) if self.training else self.forward_eval(x)

    # ------------------------------------------------------------------
    def build_eval_cache(self):
        """
        Cache routing indices, normalized logits, and unselected-channel mask.
        Per-sample weights are computed fresh each eval batch via the gate.
        """
        with torch.no_grad():
            _, indices = torch.topk(self.routing_logits, k=self.k, dim=1)
            selected_logits = torch.gather(self.routing_logits, 1, indices)
            selected_centered = selected_logits - selected_logits.mean(
                dim=1, keepdim=True)
            selected_scale = selected_centered.std(
                dim=1, keepdim=True).clamp(min=1.0)
            selected_logits_norm = selected_centered / selected_scale

            self.cached_indices = indices.contiguous()
            self.cached_selected_logits_norm = selected_logits_norm.contiguous()

            mask_int = torch.ones(self.in_channels, dtype=torch.int32,
                                  device=self.routing_logits.device)
            mask_int[indices.reshape(-1)] = 0
            self.cached_mask_int = mask_int.contiguous()
        self.cached_temp_min_tau   = float(self.temp_min) * float(self.tau)
        self.cached_temp_range_tau = float(self.temp_max - self.temp_min) * float(self.tau)
        self.cached_gate_weight = self.gate.weight.squeeze(0).contiguous()
        self.cached_gate_bias   = self.gate.bias.contiguous()

    # ------------------------------------------------------------------
    def clear_eval_cache(self):
        self.cached_indices              = None
        self.cached_selected_logits_norm = None
        self.cached_mask_int             = None

    # ------------------------------------------------------------------
    def _compute_aux_losses(self, x, E_routed, tau):
        routing_probs = F.softmax(self.routing_logits / tau, dim=1)

        expected_usage = routing_probs.mean(dim=0)
        usage_mean = expected_usage.mean()
        balance_loss = ((expected_usage - usage_mean) ** 2).mean()

        expert_given_channel = routing_probs / routing_probs.sum(
            dim=0, keepdim=True
        ).clamp(min=1e-8)

        channel_entropy = -(
            expert_given_channel
            * torch.log(expert_given_channel.clamp(min=1e-8))
        ).sum(dim=0)

        target_entropy = math.log(max(float(self.target_experts_per_channel), 1.0))

        specialization_loss = F.relu(
            channel_entropy - target_entropy
        ).mean()

        if E_routed > 1:
            pairwise_overlap = routing_probs @ routing_probs.T
            off_diag = ~torch.eye(E_routed, device=x.device).bool()
            diversity_loss = pairwise_overlap[off_diag].mean()
        else:
            diversity_loss = routing_probs.new_tensor(0.0)

        self.aux_loss = (
            self.balance_weight * balance_loss
            + self.specialization_weight * specialization_loss
            + self.diversity_weight * diversity_loss
        )

    def forward_train(self, x):
        B, C, H, W = x.shape
        E = self.num_experts
        E_routed = E - 1
        tau = max(float(self.tau), 1e-4)

        # --- 1. Static top-k routing ---
        _, indices = torch.topk(self.routing_logits, k=self.k, dim=1)  # (E-1, k)
        indices = indices.unsqueeze(0).expand(B, -1, -1)  # (B, E-1, k)

        # --- 2. Auxiliary losses ---
        self._compute_aux_losses(x, E_routed, tau)

        # --- 3. Fast gather (E-1 experts) ---
        x_flat = x.view(B, C, -1)  # (B, C, HW)
        flat_indices = indices.view(B, -1)  # (B, (E-1)*k)
        gather_indices = flat_indices.unsqueeze(-1).expand(-1, -1, H * W)
        x_gathered = torch.gather(x_flat, 1, gather_indices)
        x_experts = x_gathered.view(B, E_routed, self.k, -1)  # (B, E-1, k, HW)

        # --- 4. Temperature gate ---
        z = F.adaptive_avg_pool2d(x, 1).view(B, C)
        z_selected = torch.gather(z.unsqueeze(1).expand(-1, E_routed, -1), 2, indices)
        gate_logits = self.gate(z_selected.view(B * E_routed, self.k)).view(B, E_routed, 1)
        temperature = self.temp_min + (self.temp_max - self.temp_min) * torch.sigmoid(gate_logits)
        effective_temperature = (temperature * tau).clamp(min=1e-3)
        self._last_temperature = effective_temperature.detach()

        # --- 5. Normalize selected logits and compute weights ---
        selected_logits = torch.gather(
            self.routing_logits.unsqueeze(0).expand(B, -1, -1), 2, indices
        )
        selected_centered = selected_logits - selected_logits.mean(dim=2, keepdim=True)
        selected_scale = selected_centered.std(dim=2, keepdim=True).clamp(min=1.0)
        selected_logits_norm = selected_centered / selected_scale
        weights = F.softmax(selected_logits_norm / effective_temperature, dim=2)  # (B, E-1, k)
        out_routed = (x_experts * weights.unsqueeze(-1)).sum(dim=2)  # (B, E-1, HW)

        # --- 6. Residual expert (average of unselected channels) ---
        selected_mask = torch.zeros(B, C, device=x.device, dtype=x.dtype)
        selected_mask.scatter_(1, flat_indices, 1.0)
        unselected_mask = 1.0 - selected_mask  # (B, C)
        num_unselected = unselected_mask.sum(dim=1, keepdim=True).clamp(min=1)
        residual = (x_flat * unselected_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / num_unselected.unsqueeze(-1)  # (B, 1, HW)

        # --- 7. Concatenate: E-1 routed + 1 residual = E total ---
        out = torch.cat([out_routed, residual], dim=1)  # (B, E, HW)

        return out.view(B, E, H, W)

    # ------------------------------------------------------------------
    def forward_eval(self, x):
        if self.cached_indices is None:
            raise RuntimeError(
                "MoCE eval cache not built. Call build_moce_cache(model) before eval."
            )
        x_fp32 = x.float() if x.dtype != torch.float32 else x

        # Pool in torch (ATen)
        pooled = F.adaptive_avg_pool2d(x_fp32, 1).flatten(1)  # (B, C)
        xc = x_fp32.contiguous()

        # Temperature in torch
        sel = pooled[:, self.cached_indices]  # (B, E_routed, K)
        gate_logit = (sel * self.cached_gate_weight).sum(-1) + self.cached_gate_bias[0]  # (B, E_routed)
        temp = (self.cached_temp_min_tau + self.cached_temp_range_tau * torch.sigmoid(gate_logit)).clamp(min=1e-3)
        weights = torch.softmax(
            self.cached_selected_logits_norm.unsqueeze(0) / temp.unsqueeze(-1), dim=-1
        ).contiguous()  # (B, E_routed, K)

        # Weighted sum + residual in CUDA
        out, = moce_fixed.moce_cuda.fused_moce_forward_dynamic_temp(
            xc,
            self.cached_indices.contiguous(),
            weights,
            self.cached_mask_int.contiguous(),
        )
        return out.to(x.dtype)
