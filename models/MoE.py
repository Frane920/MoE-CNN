# models/MoE.py

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

from models.experts.DigitCNN import DigitCNN, NUM_DIGITS
from models.experts.LowercaseLetterCNN import LowercaseCNN, NUM_LOWERCASE
from models.experts.UppercaseLetterCNN import UppercaseCNN, NUM_UPPERCASE


class MoE(nn.Module):
    def __init__(self, num_digit_experts=2, num_uppercase_experts=2, num_lowercase_experts=2,
                 use_batchnorm=False, channel_mult=0.5, k_per_specialization=2,
                 gradient_checkpointing=True, unknown_threshold=0.3):
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
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        # Gate heads
        self.gate_digit = nn.Linear(128, num_digit_experts)
        self.gate_upper = nn.Linear(128, num_uppercase_experts)
        self.gate_lower = nn.Linear(128, num_lowercase_experts)

        # Temperatures
        self.gate_temp = nn.Parameter(torch.tensor(1.0))
        self.digit_temp = nn.Parameter(torch.tensor(1.0))
        self.upper_temp = nn.Parameter(torch.tensor(1.0))
        self.lower_temp = nn.Parameter(torch.tensor(1.0))

        # Unknown scale parameter
        self.unknown_scale = nn.Parameter(torch.tensor(2.0))

        # Bookkeeping
        self.num_digit_experts = num_digit_experts
        self.num_uppercase_experts = num_uppercase_experts
        self.num_lowercase_experts = num_lowercase_experts
        self.total_experts = num_digit_experts + num_uppercase_experts + num_lowercase_experts
        self.k = k_per_specialization
        self.gradient_checkpointing = gradient_checkpointing
        self.unknown_threshold = unknown_threshold

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm) or isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _run_experts(self, experts, x):
        x = x.contiguous(memory_format=torch.channels_last)
        if self.gradient_checkpointing and self.training:
            def run_all(inp):
                return torch.stack([expert(inp) for expert in experts], dim=1)
            return checkpoint(run_all, x, use_reentrant=False)
        else:
            return torch.stack([expert(x) for expert in experts], dim=1)

    def forward(self, x):
        B = x.size(0)
        with torch.cuda.amp.autocast(enabled=True):
            shared = self.gate_shared(x)

        d_logits = self.gate_digit(shared)
        u_logits = self.gate_upper(shared)
        l_logits = self.gate_lower(shared)

        # Apply temperatures
        temp = torch.clamp(self.gate_temp, min=0.1, max=10.0)
        d_logits = d_logits / temp / torch.clamp(self.digit_temp, min=0.1, max=10.0)
        u_logits = u_logits / temp / torch.clamp(self.upper_temp, min=0.1, max=10.0)
        l_logits = l_logits / temp / torch.clamp(self.lower_temp, min=0.1, max=10.0)

        # Run experts
        digit_outputs = self._run_experts(self.digit_experts, x)
        uppercase_outputs = self._run_experts(self.uppercase_experts, x)
        lowercase_outputs = self._run_experts(self.lowercase_experts, x)

        # Unknown class probabilities
        digit_unknown_probs = F.softmax(digit_outputs, dim=-1)[..., -1].detach()
        uppercase_unknown_probs = F.softmax(uppercase_outputs, dim=-1)[..., -1].detach()
        lowercase_unknown_probs = F.softmax(lowercase_outputs, dim=-1)[..., -1].detach()

        # Adjust logits by unknown probs
        scale = torch.clamp(self.unknown_scale, min=0.0, max=50.0)
        d_logits_adj = d_logits - digit_unknown_probs * scale
        u_logits_adj = u_logits - uppercase_unknown_probs * scale
        l_logits_adj = l_logits - lowercase_unknown_probs * scale

        kd = min(self.k, self.num_digit_experts)
        ku = min(self.k, self.num_uppercase_experts)
        kl = min(self.k, self.num_lowercase_experts)

        d_topv, d_idx = d_logits_adj.topk(k=kd, dim=1)
        u_topv, u_idx = u_logits_adj.topk(k=ku, dim=1)
        l_topv, l_idx = l_logits_adj.topk(k=kl, dim=1)

        neg_inf = -1e9
        d_masked_logits = torch.full_like(d_logits_adj, neg_inf)
        u_masked_logits = torch.full_like(u_logits_adj, neg_inf)
        l_masked_logits = torch.full_like(l_logits_adj, neg_inf)

        d_masked_logits.scatter_(1, d_idx, d_topv)
        u_masked_logits.scatter_(1, u_idx, u_topv)
        l_masked_logits.scatter_(1, l_idx, l_topv)

        d_norm = F.softmax(d_masked_logits, dim=1)
        u_norm = F.softmax(u_masked_logits, dim=1)
        l_norm = F.softmax(l_masked_logits, dim=1)

        d_combined = torch.einsum('be,bec->bc', d_norm, digit_outputs[..., :NUM_DIGITS])
        u_combined = torch.einsum('be,bec->bc', u_norm, uppercase_outputs[..., :NUM_UPPERCASE])
        l_combined = torch.einsum('be,bec->bc', l_norm, lowercase_outputs[..., :NUM_LOWERCASE])

        combined_output = torch.cat([d_combined, u_combined, l_combined], dim=1)

        # penalty (not detached so gradient flows here)
        digit_penalty = digit_unknown_probs.max(dim=1)[0]
        uppercase_penalty = uppercase_unknown_probs.max(dim=1)[0]
        lowercase_penalty = lowercase_unknown_probs.max(dim=1)[0]
        penalty_per_sample = (digit_penalty + uppercase_penalty + lowercase_penalty) / 3.0
        total_penalty = penalty_per_sample.mean()

        return combined_output, total_penalty

    def gate(self, x):
        with torch.cuda.amp.autocast(enabled=True):
            shared = self.gate_shared(x)
        d_logits = self.gate_digit(shared)
        u_logits = self.gate_upper(shared)
        l_logits = self.gate_lower(shared)
        temp = torch.clamp(self.gate_temp, min=0.1, max=10.0)
        return torch.cat([d_logits / temp, u_logits / temp, l_logits / temp], dim=1)

