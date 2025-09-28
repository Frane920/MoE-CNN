# models/MoE.py

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

from models.experts.DigitCNN import DigitCNN, NUM_DIGITS
from models.experts.LowercaseLetterCNN import LowercaseCNN, NUM_LOWERCASE
from models.experts.UppercaseLetterCNN import UppercaseCNN, NUM_UPPERCASE


class MoE(nn.Module):
    """
    Specialized Hierarchical Mixture of Experts with Unknown Class Filtering.

    Changes / guarantees:
      - No .item() calls in forward/gate (torch.compile / torch.dynamo friendly).
      - Added .gate(x) method returning concatenated gate logits [B, total_experts].
      - Stable clamping via 0-dim tensors (broadcastable).
      - Forward returns (combined_output, total_penalty) where trainer multiplies penalty_weight.
    """

    def __init__(self, num_digit_experts=2, num_uppercase_experts=2, num_lowercase_experts=2,
                 use_batchnorm=False, channel_mult=0.5, k_per_specialization=2, gradient_checkpointing=True,
                 unknown_threshold=0.3):
        super().__init__()

        # Experts
        self.digit_experts = nn.ModuleList([
            DigitCNN(use_batchnorm=use_batchnorm, channel_mult=channel_mult)
            for _ in range(num_digit_experts)
        ])
        self.uppercase_experts = nn.ModuleList([
            UppercaseCNN(use_batchnorm=use_batchnorm, channel_mult=channel_mult)
            for _ in range(num_uppercase_experts)
        ])
        self.lowercase_experts = nn.ModuleList([
            LowercaseCNN(use_batchnorm=use_batchnorm, channel_mult=channel_mult)
            for _ in range(num_lowercase_experts)
        ])

        # Shared gating trunk
        self.gate_shared = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)  # Increased dropout
        )
        self.digit_temp = nn.Parameter(torch.tensor(1.0))
        self.upper_temp = nn.Parameter(torch.tensor(1.0))
        self.lower_temp = nn.Parameter(torch.tensor(1.0))

    def _run_experts(self, experts, x):
        """Run list of experts, return stacked logits: [B, E, C]"""
        # Ensure channels_last for conv performance on CUDA
        x = x.contiguous(memory_format=torch.channels_last)

        if self.gradient_checkpointing and self.training:
            def run_all(x):
                return torch.stack([expert(x) for expert in experts], dim=1)

            # checkpoint reduces memory but requires grad; it's OK in normal forward
            return checkpoint(run_all, x, use_reentrant=False)
        else:
            return torch.stack([expert(x) for expert in experts], dim=1)

    def forward(self, x):
        """
        Forward returns:
          - combined_output: [B, NUM_DIGITS + NUM_UPPERCASE + NUM_LOWERCASE]
          - total_penalty: scalar tensor (trainer multiplies by penalty_weight)
        """
        B = x.size(0)
        with torch.amp.autocast('cuda', enabled=False):
            shared = self.gate_shared(x).float()

        d_logits = self.gate_digit(shared)
        u_logits = self.gate_upper(shared)
        l_logits = self.gate_lower(shared)

        temp = torch.clamp(self.gate_temp, min=0.1, max=10.0)
        if temp != 1.0:
            d_logits = d_logits / temp
            u_logits = u_logits / temp
            l_logits = l_logits / temp

        d_logits = d_logits / torch.clamp(self.digit_temp, min=0.1, max=10.0)
        u_logits = u_logits / torch.clamp(self.upper_temp, min=0.1, max=10.0)
        l_logits = l_logits / torch.clamp(self.lower_temp, min=0.1, max=10.0)

        # ---- Run experts (logits) ----
        digit_outputs = self._run_experts(self.digit_experts, x)        # [B, Dexp, NUM_DIGITS+1]
        uppercase_outputs = self._run_experts(self.uppercase_experts, x)  # [B, Uexp, NUM_UPPERCASE+1]
        lowercase_outputs = self._run_experts(self.lowercase_experts, x)  # [B, Lexp, NUM_LOWERCASE+1]

        # ---- Unknown probs for each expert (prob on unknown class) ----
        digit_unknown_probs = F.softmax(digit_outputs, dim=-1)[..., -1]        # [B, Dexp]
        uppercase_unknown_probs = F.softmax(uppercase_outputs, dim=-1)[..., -1]  # [B, Uexp]
        lowercase_unknown_probs = F.softmax(lowercase_outputs, dim=-1)[..., -1]  # [B, Lexp]

        # ---- Confidence-weighted gating: subtract unknown * scale from logits ----
        scale = torch.clamp(self.unknown_scale, min=0.0, max=50.0)  # tensor scalar
        d_logits_adj = d_logits - digit_unknown_probs * scale
        u_logits_adj = u_logits - uppercase_unknown_probs * scale
        l_logits_adj = l_logits - lowercase_unknown_probs * scale

        # ---- top-k routing per group ----
        kd = min(self.k, self.num_digit_experts)
        ku = min(self.k, self.num_uppercase_experts)
        kl = min(self.k, self.num_lowercase_experts)

        d_topv, d_idx = d_logits_adj.topk(k=kd, dim=1)
        u_topv, u_idx = u_logits_adj.topk(k=ku, dim=1)
        l_topv, l_idx = l_logits_adj.topk(k=kl, dim=1)

        neg_inf = -1e4 if d_logits_adj.dtype == torch.float16 else -1e9

        d_masked_logits = torch.full_like(d_logits_adj, neg_inf)
        u_masked_logits = torch.full_like(u_logits_adj, neg_inf)
        l_masked_logits = torch.full_like(l_logits_adj, neg_inf)

        d_masked_logits.scatter_(1, d_idx, d_topv)
        u_masked_logits.scatter_(1, u_idx, u_topv)
        l_masked_logits.scatter_(1, l_idx, l_topv)

        d_norm = F.softmax(d_masked_logits, dim=1)  # [B, Dexp]
        u_norm = F.softmax(u_masked_logits, dim=1)  # [B, Uexp]
        l_norm = F.softmax(l_masked_logits, dim=1)  # [B, Lexp]

        # ---- Combine expert outputs (exclude unknown class logits) ----
        d_combined = torch.einsum('be,bec->bc', d_norm, digit_outputs[..., :NUM_DIGITS])
        u_combined = torch.einsum('be,bec->bc', u_norm, uppercase_outputs[..., :NUM_UPPERCASE])
        l_combined = torch.einsum('be,bec->bc', l_norm, lowercase_outputs[..., :NUM_LOWERCASE])

        combined_output = torch.cat([d_combined, u_combined, l_combined], dim=1)  # [B, total_classes]

        # ---- Penalty term (encourage experts not to spam unknown) ----
        digit_penalty = digit_unknown_probs.max(dim=1)[0]      # [B]
        uppercase_penalty = uppercase_unknown_probs.max(dim=1)[0]
        lowercase_penalty = lowercase_unknown_probs.max(dim=1)[0]

        penalty_per_sample = (digit_penalty + uppercase_penalty + lowercase_penalty) / 3.0
        total_penalty = penalty_per_sample.mean()  # scalar tensor

        return combined_output, total_penalty

    def gate(self, x):
        """
        Return concatenated raw gate logits for monitoring:
            [B, num_digit_experts + num_uppercase_experts + num_lowercase_experts]

        Note: this returns *raw* logits (optionally you can post-process),
        but it's exactly what your training monitor expects for argmax.
        """
        # compute shared trunk (disable autocast like forward)
        with torch.amp.autocast('cuda', enabled=False):
            shared = self.gate_shared(x).float()

        d_logits = self.gate_digit(shared)
        u_logits = self.gate_upper(shared)
        l_logits = self.gate_lower(shared)

        # apply temperature (tensor)
        temp = torch.clamp(self.gate_temp, min=0.1, max=10.0)
        if temp != 1.0:
            d_logits = d_logits / temp
            u_logits = u_logits / temp
            l_logits = l_logits / temp

        # concatenate in the same ordering as forward expects (digit, upper, lower)
        return torch.cat([d_logits, u_logits, l_logits], dim=1)

    def debug_stats(self, x):
        """Return small diagnostic tensors (no-grad). Cheap version — only digit group)."""
        with torch.no_grad():
            shared = self.gate_shared(x).float()
            gate_logits_d = self.gate_digit(shared)

            # run digit experts only
            digit_outputs = self._run_experts(self.digit_experts, x)
            digit_unknown_probs = F.softmax(digit_outputs, dim=-1)[..., -1]

            d_logits_adj = gate_logits_d - digit_unknown_probs * torch.clamp(self.unknown_scale, min=0.0)
            kd = min(self.k, self.num_digit_experts)
            d_topv, d_idx = d_logits_adj.topk(k=kd, dim=1)
            neg_inf = -1e9
            d_masked_logits = torch.full_like(d_logits_adj, neg_inf)
            d_masked_logits.scatter_(1, d_idx, d_topv)
            d_norm = F.softmax(d_masked_logits, dim=1)

            return {
                "gate_logits_d": gate_logits_d,
                "d_norm": d_norm,
                "digit_unknown_probs": digit_unknown_probs,
                "digit_outputs_min": digit_outputs.min(dim=-1).values,
            }

