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
    Specialized Hierarchical Mixture of Experts with Unknown Class Filtering:
    - Each expert trained with wrong data labeled as unknown
    - Routing first filters experts with high unknown confidence
    - Then selects from remaining experts based on normal confidence
    """

    def __init__(self, num_digit_experts=2, num_uppercase_experts=2, num_lowercase_experts=2,
                 use_batchnorm=False, channel_mult=0.5, k_per_specialization=2, gradient_checkpointing=True,
                 unknown_threshold=0.3):
        super().__init__()

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

        # Enhanced gating with uncertainty awareness
        self.gate = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_digit_experts + num_uppercase_experts + num_lowercase_experts)
        )

        self.num_digit_experts = num_digit_experts
        self.num_uppercase_experts = num_uppercase_experts
        self.num_lowercase_experts = num_lowercase_experts
        self.total_experts = num_digit_experts + num_uppercase_experts + num_lowercase_experts
        self.k = k_per_specialization
        self.gradient_checkpointing = gradient_checkpointing
        self.unknown_threshold = unknown_threshold  # Threshold for filtering uncertain experts

    def _run_experts(self, experts, x):
        """Helper to run experts with optional gradient checkpointing."""
        # Use channels_last memory format for better performance on NVIDIA GPUs
        x = x.contiguous(memory_format=torch.channels_last)

        if self.gradient_checkpointing and self.training:
            # Use a single checkpoint for all experts to reduce overhead
            def run_all_experts(x):
                return torch.stack([expert(x) for expert in experts], dim=1)

            return checkpoint(run_all_experts, x, use_reentrant=False)
        else:
            # Run experts in parallel using list comprehension (faster than vmap for small numbers)
            return torch.stack([expert(x) for expert in experts], dim=1)

    def _get_unknown_confidences(self, experts, x):
        """Get unknown confidence scores for all experts in a group."""
        confidences = []
        for expert in experts:
            with torch.no_grad():
                conf = expert.get_unknown_confidence(x)
            confidences.append(conf)
        return torch.stack(confidences, dim=1)  # [B, num_experts]

    def forward(self, x):
        B = x.size(0)
        device = x.device

        # ---- Gate logits ----
        with torch.amp.autocast('cuda', enabled=False):
            gate_logits = self.gate(x).float()  # [B, total_experts] in FP32

        # Slice gate logits per specialization
        d_logits = gate_logits[:, :self.num_digit_experts]
        u_logits = gate_logits[:, self.num_digit_experts:
                                  self.num_digit_experts + self.num_uppercase_experts]
        l_logits = gate_logits[:, self.num_digit_experts + self.num_uppercase_experts:]

        # ---- Run experts (parallel) ----
        digit_outputs = self._run_experts(self.digit_experts, x)  # [B, Dexp, NUM_DIGITS+1]
        uppercase_outputs = self._run_experts(self.uppercase_experts, x)  # [B, Uexp, NUM_UPPERCASE+1]
        lowercase_outputs = self._run_experts(self.lowercase_experts, x)  # [B, Lexp, NUM_LOWERCASE+1]

        # ---- Unknown probs for each expert ----
        digit_unknown_probs = F.softmax(digit_outputs, dim=-1)[..., -1]  # [B, Dexp]
        uppercase_unknown_probs = F.softmax(uppercase_outputs, dim=-1)[..., -1]  # [B, Uexp]
        lowercase_unknown_probs = F.softmax(lowercase_outputs, dim=-1)[..., -1]  # [B, Lexp]

        # ---- Confidence-weighted gating ----
        # subtract unknown * scale from logits
        scale = getattr(self, "unknown_scale", 5.0)

        d_logits_adj = d_logits - digit_unknown_probs * scale
        u_logits_adj = u_logits - uppercase_unknown_probs * scale
        l_logits_adj = l_logits - lowercase_unknown_probs * scale

        # top-k routing per group
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

        d_norm = F.softmax(d_masked_logits, dim=1)
        u_norm = F.softmax(u_masked_logits, dim=1)
        l_norm = F.softmax(l_masked_logits, dim=1)

        # ---- Combine expert outputs (exclude unknown class) ----
        d_combined = torch.einsum('be,bec->bc', d_norm, digit_outputs[..., :NUM_DIGITS])
        u_combined = torch.einsum('be,bec->bc', u_norm, uppercase_outputs[..., :NUM_UPPERCASE])
        l_combined = torch.einsum('be,bec->bc', l_norm, lowercase_outputs[..., :NUM_LOWERCASE])

        combined_output = torch.cat([d_combined, u_combined, l_combined], dim=1)

        # ---- Penalty term ----
        # encourage experts not to spam unknown
        digit_penalty = torch.einsum('be,be->b', d_norm, digit_unknown_probs)
        uppercase_penalty = torch.einsum('be,be->b', u_norm, uppercase_unknown_probs)
        lowercase_penalty = torch.einsum('be,be->b', l_norm, lowercase_unknown_probs)

        penalty_per_sample = (digit_penalty + uppercase_penalty + lowercase_penalty) / 3.0
        total_penalty = penalty_per_sample.mean()

        return combined_output, total_penalty

    def debug_stats(self, x):
        """Return small diagnostic tensors (no-grad)."""
        with torch.no_grad():
            gate_logits = self.gate(x).float()
            d_logits = gate_logits[:, :self.num_digit_experts]
            # run experts
            digit_outputs = self._run_experts(self.digit_experts, x)
            uppercase_outputs = self._run_experts(self.uppercase_experts, x)
            lowercase_outputs = self._run_experts(self.lowercase_experts, x)

            digit_unknown_probs = F.softmax(digit_outputs, dim=-1)[..., -1]
            uppercase_unknown_probs = F.softmax(uppercase_outputs, dim=-1)[..., -1]
            lowercase_unknown_probs = F.softmax(lowercase_outputs, dim=-1)[..., -1]

            # masked softmax used for routing (re-use forward's logic minimally)
            d_logits_adj = d_logits - digit_unknown_probs * getattr(self, "unknown_scale", 5.0)
            kd = min(self.k, self.num_digit_experts)
            d_topv, d_idx = d_logits_adj.topk(k=kd, dim=1)
            neg_inf = -1e9
            d_masked_logits = torch.full_like(d_logits_adj, neg_inf)
            d_masked_logits.scatter_(1, d_idx, d_topv)
            d_norm = F.softmax(d_masked_logits, dim=1)

            return {
                "gate_logits": gate_logits,
                "d_norm": d_norm,
                "digit_unknown_probs": digit_unknown_probs,
                "digit_outputs_min": digit_outputs.min(dim=-1).values,  # just to see logits range
            }