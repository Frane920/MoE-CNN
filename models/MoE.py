# models/MoE.py

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.experts.DigitCNN import DigitCNN, NUM_DIGITS
from models.experts.LowercaseLetterCNN import LowercaseCNN, NUM_LOWERCASE
from models.experts.UppercaseLetterCNN import UppercaseCNN, NUM_UPPERCASE


class MoE(nn.Module):
    def __init__(self,
                 num_digit_experts=2,
                 num_uppercase_experts=2,
                 num_lowercase_experts=2,
                 use_batchnorm=False,
                 channel_mult=0.5,
                 k_per_specialization=2,
                 gradient_checkpointing=True,
                 unknown_threshold=0.3,
                 load_balance_coef=0.05,
                 initial_gate_temp=2.0):
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

        self.gate_shared = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
    )

        self.gate_digit = nn.Linear(128, num_digit_experts)
        self.gate_upper = nn.Linear(128, num_uppercase_experts)
        self.gate_lower = nn.Linear(128, num_lowercase_experts)

        self.gate_temp = nn.Parameter(torch.tensor(float(initial_gate_temp)))
        self.digit_temp = nn.Parameter(torch.tensor(1.0))
        self.upper_temp = nn.Parameter(torch.tensor(1.0))
        self.lower_temp = nn.Parameter(torch.tensor(1.0))

        self.unknown_scale = nn.Parameter(torch.tensor(2.0))

        self.num_digit_experts = num_digit_experts
        self.num_uppercase_experts = num_uppercase_experts
        self.num_lowercase_experts = num_lowercase_experts
        self.total_experts = num_digit_experts + num_uppercase_experts + num_lowercase_experts
        self.k = k_per_specialization
        self.gradient_checkpointing = gradient_checkpointing
        self.unknown_threshold = unknown_threshold

        self.load_balance_coef = float(load_balance_coef)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def _run_experts(self, experts, x):
        x = x.contiguous(memory_format=torch.channels_last)
        if self.gradient_checkpointing and self.training:
            def run_all(x):
                return torch.stack([expert(x) for expert in experts], dim=1)
            return checkpoint(run_all, x, use_reentrant=False)
        else:
            return torch.stack([expert(x) for expert in experts], dim=1)

    def _load_balance_loss(self, probs_norm):
        if probs_norm.numel() == 0:
            return probs_norm.new_tensor(0.0)
        mean_per_expert = probs_norm.mean(dim=0)  # [E]
        target = torch.full_like(mean_per_expert, 1.0 / mean_per_expert.size(0))
        return F.mse_loss(mean_per_expert, target, reduction='sum')  # sum so magnitude scales with E

    def forward(self, x):
        B = x.size(0)

        with torch.amp.autocast('cuda', enabled=False):
            shared = self.gate_shared(x).float()  # [B,128]

        d_logits = self.gate_digit(shared)   # [B, Dexp]
        u_logits = self.gate_upper(shared)   # [B, Uexp]
        l_logits = self.gate_lower(shared)   # [B, Lexp]

        temp = torch.clamp(self.gate_temp, min=0.1, max=50.0)
        if temp != 1.0:
            d_logits = d_logits / temp
            u_logits = u_logits / temp
            l_logits = l_logits / temp

        d_logits = d_logits / torch.clamp(self.digit_temp, min=0.1, max=50.0)
        u_logits = u_logits / torch.clamp(self.upper_temp, min=0.1, max=50.0)
        l_logits = l_logits / torch.clamp(self.lower_temp, min=0.1, max=50.0)

        digit_outputs = self._run_experts(self.digit_experts, x)        # [B, Dexp, NUM_DIGITS+1]
        uppercase_outputs = self._run_experts(self.uppercase_experts, x)  # [B, Uexp, NUM_UPPERCASE+1]
        lowercase_outputs = self._run_experts(self.lowercase_experts, x)  # [B, Lexp, NUM_LOWERCASE+1]

        digit_unknown_probs = F.softmax(digit_outputs, dim=-1)[..., -1]        # [B, Dexp]
        uppercase_unknown_probs = F.softmax(uppercase_outputs, dim=-1)[..., -1]  # [B, Uexp]
        lowercase_unknown_probs = F.softmax(lowercase_outputs, dim=-1)[..., -1]  # [B, Lexp]

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

        d_combined = torch.einsum('be,bec->bc', d_norm, digit_outputs[..., :NUM_DIGITS])
        u_combined = torch.einsum('be,bec->bc', u_norm, uppercase_outputs[..., :NUM_UPPERCASE])
        l_combined = torch.einsum('be,bec->bc', l_norm, lowercase_outputs[..., :NUM_LOWERCASE])

        combined_output = torch.cat([d_combined, u_combined, l_combined], dim=1)  # [B, total_classes]

        digit_penalty = digit_unknown_probs.max(dim=1)[0]      # [B]
        uppercase_penalty = uppercase_unknown_probs.max(dim=1)[0]
        lowercase_penalty = lowercase_unknown_probs.max(dim=1)[0]
        penalty_per_sample = (digit_penalty + uppercase_penalty + lowercase_penalty) / 3.0
        total_unknown_penalty = penalty_per_sample.mean()

        load_loss_d = self._load_balance_loss(d_norm) if d_norm.numel() else d_norm.new_tensor(0.0)
        load_loss_u = self._load_balance_loss(u_norm) if u_norm.numel() else u_norm.new_tensor(0.0)
        load_loss_l = self._load_balance_loss(l_norm) if l_norm.numel() else l_norm.new_tensor(0.0)
        total_load_loss = (load_loss_d + load_loss_u + load_loss_l)

        total_penalty = total_unknown_penalty + self.load_balance_coef * total_load_loss

        return combined_output, total_penalty

    def gate(self, x):
        with torch.amp.autocast('cuda', enabled=False):
            shared = self.gate_shared(x).float()

        d_logits = self.gate_digit(shared)
        u_logits = self.gate_upper(shared)
        l_logits = self.gate_lower(shared)

        temp = torch.clamp(self.gate_temp, min=0.1, max=50.0)
        if temp != 1.0:
            d_logits = d_logits / temp
            u_logits = u_logits / temp
            l_logits = l_logits / temp

        return torch.cat([d_logits, u_logits, l_logits], dim=1)

    def debug_stats(self, x):
        with torch.no_grad():
            shared = self.gate_shared(x).float()
            gate_logits_d = self.gate_digit(shared)
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
