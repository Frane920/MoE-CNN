#!/usr/bin/env python3
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from lion_pytorch import Lion
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import datasets
from tqdm import tqdm

from models.MoE import MoE
from utils.checkpoint import load_checkpoint_if_exists, save_checkpoint
from utils.parse_args import parse_args
from augmentations.GPUAug import GPUAug

# Set OMP_NUM_THREADS to avoid warning
os.environ["OMP_NUM_THREADS"] = "8"

# Constants
DEFAULT_NUM_CLASSES = 62
NUM_DIGITS = 10
NUM_UPPERCASE = 26
NUM_LOWERCASE = 26

# Data Preprocessing
class EMNISTPreprocessor:
    """Handles EMNIST data preprocessing."""

    @staticmethod
    def preprocess_uint8(data_uint8, resize_to=None):
        """
        Convert EMNIST raw uint8 [N,H,W] to float tensor [-1,1] in correct orientation.
        Flip horizontally, rotate 90 CCW, normalize to [-1,1], optionally resize.
        """
        data = torch.flip(data_uint8, dims=(2,))
        data = torch.rot90(data, k=1, dims=(1, 2))
        data = data.to(torch.float32) / 255.0
        data = data.unsqueeze(1)

        if resize_to is not None:
            data = F.interpolate(data, size=(resize_to, resize_to), mode='bilinear', align_corners=False)

        data = data * 2.0 - 1.0
        return data


class EMNISTInRam(Dataset):
    """EMNIST dataset loaded into RAM."""

    def __init__(self, images_tensor, labels_tensor):
        self.images = images_tensor
        self.labels = labels_tensor

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# Data Loading
class DataPrefetcher:
    def __init__(self, loader, device, use_specialized_moe=False):
        self.loader = iter(loader)
        self.device = device
        self.use_specialized_moe = use_specialized_moe
        self.stream = torch.cuda.Stream() if device.type == 'cuda' else None
        self.next_input = None
        self.next_targets = None
        self.preload()

    def preload(self):
        try:
            data = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_targets = None
            return

        if self.device.type == 'cuda':
            with torch.cuda.stream(self.stream):
                if self.use_specialized_moe:
                    x, digit_y, uppercase_y, lowercase_y = data
                    self.next_input = x.to(self.device, non_blocking=True, memory_format=torch.channels_last)
                    self.next_targets = (
                        digit_y.to(self.device, non_blocking=True),
                        uppercase_y.to(self.device, non_blocking=True),
                        lowercase_y.to(self.device, non_blocking=True)
                    )
                else:
                    input, target = data
                    self.next_input = input.to(self.device, non_blocking=True, memory_format=torch.channels_last)
                    self.next_targets = target.to(self.device, non_blocking=True)
        else:
            if self.use_specialized_moe:
                x, digit_y, uppercase_y, lowercase_y = data
                self.next_input = x.to(self.device)
                self.next_targets = (
                    digit_y.to(self.device),
                    uppercase_y.to(self.device),
                    lowercase_y.to(self.device)
                )
            else:
                input, target = data
                self.next_input = input.to(self.device)
                self.next_targets = target.to(self.device)

    def next(self):
        if self.next_input is None:
            return None, None

        if self.device.type == 'cuda':
            torch.cuda.current_stream().wait_stream(self.stream)

        input = self.next_input
        targets = self.next_targets
        self.preload()
        return input, targets

# Data Augmentation

def mixup_criterion(criterion, pred, y_pair, lam):
    """Loss function for mixup/cutmix."""
    y_a, y_b = y_pair
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


# Loss Functions
class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""

    def __init__(self, gamma=1.5, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        logp = -F.cross_entropy(input, target, weight=self.weight, reduction='none')
        p = torch.exp(logp)
        loss = -((1 - p) ** self.gamma) * logp
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# Model Exponential Moving Average
class ModelEMA:
    """Model Exponential Moving Average."""

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        # store a copy on same device/dtype as model params initially
        self.shadow = {}
        for n, p in model.state_dict().items():
            if p.dtype.is_floating_point:
                self.shadow[n] = p.detach().clone()
        self.device = next(iter(self.shadow.values())).device if self.shadow else torch.device('cpu')

    def update(self, model):
        msd = model.state_dict()
        with torch.no_grad():
            for k, v in self.shadow.items():
                if k in msd:
                    v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)

    def copy_to_model(self, model):
        msd = model.state_dict()
        for k, v in self.shadow.items():
            if k in msd:
                msd[k].copy_(v.to(msd[k].device))




# Data Loading
class EMNISTDataLoader:
    """Handles loading and preparing EMNIST data for MoE training."""

    def __init__(self, args):
        self.args = args
        self.preprocessor = EMNISTPreprocessor()

    def create_specialized_dataloaders(self, raw_train, raw_test):
        """
        Create specialized dataloaders where each expert sees wrong data labeled as unknown
        """
        if self.args.rank0:
            print("Creating specialized datasets for MoE training...")

        # Convert EMNIST labels to specialized targets
        # EMNIST byclass mapping: 0-9: digits, 10-35: uppercase, 36-61: lowercase
        train_images = self.preprocessor.preprocess_uint8(raw_train.data, resize_to=self.args.resize)
        val_images = self.preprocessor.preprocess_uint8(raw_test.data, resize_to=self.args.resize)

        # For DigitCNN.py: digits are normal classes, letters are unknown
        digit_train_labels = raw_train.targets.clone()
        digit_val_labels = raw_test.targets.clone()
        digit_train_labels[(digit_train_labels >= 10)] = NUM_DIGITS  # Mark letters as unknown
        digit_val_labels[(digit_val_labels >= 10)] = NUM_DIGITS

        # For UppercaseCNN: uppercase are normal, digits and lowercase are unknown
        uppercase_train_labels = raw_train.targets.clone() - 10  # Convert to 0-25 for A-Z
        uppercase_val_labels = raw_test.targets.clone() - 10
        # Mark non-uppercase as unknown
        uppercase_train_labels[(raw_train.targets < 10) | (raw_train.targets >= 36)] = 26
        uppercase_val_labels[(raw_test.targets < 10) | (raw_test.targets >= 36)] = 26

        # For LowercaseCNN: lowercase are normal, digits and uppercase are unknown
        lowercase_train_labels = raw_train.targets.clone() - 36  # Convert to 0-25 for a-z
        lowercase_val_labels = raw_test.targets.clone() - 36
        # Mark non-lowercase as unknown
        lowercase_train_labels[(raw_train.targets < 36)] = 26
        lowercase_val_labels[(raw_test.targets < 36)] = 26

        # Create datasets for each expert type
        digit_train_ds = EMNISTInRam(train_images, digit_train_labels)
        digit_val_ds = EMNISTInRam(val_images, digit_val_labels)

        uppercase_train_ds = EMNISTInRam(train_images, uppercase_train_labels)
        uppercase_val_ds = EMNISTInRam(val_images, uppercase_val_labels)

        lowercase_train_ds = EMNISTInRam(train_images, lowercase_train_labels)
        lowercase_val_ds = EMNISTInRam(val_images, lowercase_val_labels)

        # Create combined dataset that returns all three label types
        class CombinedDataset(Dataset):
            def __init__(self, images, digit_labels, uppercase_labels, lowercase_labels):
                self.images = images
                self.digit_labels = digit_labels
                self.uppercase_labels = uppercase_labels
                self.lowercase_labels = lowercase_labels

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                return (self.images[idx],
                        self.digit_labels[idx],
                        self.uppercase_labels[idx],
                        self.lowercase_labels[idx])

        combined_train_ds = CombinedDataset(train_images, digit_train_labels, uppercase_train_labels,
                                            lowercase_train_labels)
        combined_val_ds = CombinedDataset(val_images, digit_val_labels, uppercase_val_labels, lowercase_val_labels)

        # Create samplers
        if dist.is_available() and dist.is_initialized():
            train_sampler = DistributedSampler(combined_train_ds, num_replicas=self.args.world_size,
                                               rank=self.args.rank, shuffle=True)
            val_sampler = DistributedSampler(combined_val_ds, num_replicas=self.args.world_size,
                                             rank=self.args.rank, shuffle=False)
        else:
            train_sampler = None
            val_sampler = None

        train_loader = DataLoader(combined_train_ds, batch_size=self.args.per_gpu_batch, sampler=train_sampler,
                                  num_workers=self.args.num_workers, pin_memory=True, persistent_workers=True,
                                  drop_last=True, prefetch_factor=4, multiprocessing_context='fork')
        val_loader = DataLoader(combined_val_ds, batch_size=self.args.per_gpu_batch, sampler=val_sampler,
                                num_workers=self.args.num_workers, pin_memory=True, persistent_workers=True,
                                drop_last=False, prefetch_factor=4, multiprocessing_context='fork')

        return train_loader, val_loader

    def create_standard_dataloaders(self, raw_train, raw_test):
        """Create standard dataloaders for non-MoE training."""
        train_images = self.preprocessor.preprocess_uint8(raw_train.data, resize_to=self.args.resize)
        train_labels = raw_train.targets.clone().long()
        val_images = self.preprocessor.preprocess_uint8(raw_test.data, resize_to=self.args.resize)
        val_labels = raw_test.targets.clone().long()

        train_ds = EMNISTInRam(train_images, train_labels)
        val_ds = EMNISTInRam(val_images, val_labels)

        if dist.is_available() and dist.is_initialized():
            train_sampler = DistributedSampler(train_ds, num_replicas=self.args.world_size,
                                               rank=self.args.rank, shuffle=True)
            val_sampler = DistributedSampler(val_ds, num_replicas=self.args.world_size,
                                             rank=self.args.rank, shuffle=False)
        else:
            train_sampler = None
            val_sampler = None

        train_loader = DataLoader(train_ds, batch_size=self.args.per_gpu_batch, sampler=train_sampler,
                                  num_workers=self.args.num_workers, pin_memory=True, persistent_workers=True,
                                  drop_last=True, prefetch_factor=4, multiprocessing_context='fork')
        val_loader = DataLoader(val_ds, batch_size=self.args.per_gpu_batch, sampler=val_sampler,
                                num_workers=self.args.num_workers, pin_memory=True, persistent_workers=True,
                                drop_last=False, prefetch_factor=4, multiprocessing_context='fork')

        return train_loader, val_loader


# Training and Evaluation
class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.args = args

        self.gpu_aug = GPUAug() if args.gpu_augment else None

        self.ema = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None

        self.optimize_gradient_accumulation()

        if getattr(self.args, "device", None) is not None and self.args.device.type == 'cuda':
            self.optimize_memory()

        if getattr(args, 'swa_start', None) is not None:
            self.swa_start = int(args.swa_start)
        else:
            self.swa_start = int(max(1, args.epochs * 0.75))

        self.swa_scheduler = None
        self.swa_model = None

        if args.epochs > self.swa_start:
            self.swa_model = AveragedModel(self.model)
            self.swa_scheduler = SWALR(self.optimizer, swa_lr=1e-6)

    def train_one_epoch(self, epoch):
        max_mix = float(self.args.mixup_alpha)
        max_cut = float(self.args.cutmix_alpha)
        warm = int(getattr(self.args, 'aug_warmup_epochs', 2))
        ramp = int(getattr(self.args, 'aug_ramp_epochs', 4))

        if epoch <= warm:
            cur_mix = 0.0
            cur_cut = 0.0
        else:
            step = min(epoch - warm, ramp)
            frac = step / float(max(1, ramp))
            cur_mix = max_mix * frac
            cur_cut = max_cut * frac

        if self.args.rank0:
            print(f"Aug schedule: mixup_alpha={cur_mix:.3f}, cutmix_alpha={cur_cut:.3f} (epoch {epoch})")

        self.model.train()
        running_loss = 0.0
        running_penalty = 0.0
        running_cls_loss = 0.0
        n_samples = 0

        use_amp = (self.args.amp and getattr(self.args, "device", None) is not None and self.args.device.type == 'cuda')
        use_prefetcher = (not self.args.no_prefetch and getattr(self.args, "device", None) is not None
                          and self.args.device.type == 'cuda' and not self.args.use_specialized_moe)
        prefetcher = None
        if use_prefetcher:
            prefetcher = DataPrefetcher(self.train_loader, self.args.device, use_specialized_moe=False)

        data_iter = None if use_prefetcher else iter(self.train_loader)
        pbar = tqdm(total=(len(self.train_loader) if not use_prefetcher else None),
                    desc=f"Train E{epoch}", disable=not self.args.rank0)

        self.optimizer.zero_grad(set_to_none=True)
        it = 0

        warmup_epochs = getattr(self.args, "warmup_epochs", 5)
        frac = min(1.0, float(epoch) / max(1.0, warmup_epochs))
        cos_component = 0.5 * (1 - math.cos(math.pi * frac))
        penalty_floor = 0.01
        penalty_weight = float(self.args.penalty_weight) * (penalty_floor + (1.0 - penalty_floor) * cos_component)

        penalty_debug = []
        total_experts = (self.args.num_digit_experts + self.args.num_uppercase_experts + self.args.num_lowercase_experts)
        expert_usage = torch.zeros(total_experts, device=self.args.device, dtype=torch.long)

        while True:
            if use_prefetcher:
                batch_x, batch_targets = prefetcher.next()
                if batch_x is None:
                    break
                batch = (batch_x, batch_targets)
            else:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break

            it += 1

            x = None
            y = None
            lam = 1.0
            y_pair = None
            try:
                if self.args.use_specialized_moe:
                    if len(batch) == 4:
                        x, digit_y, uppercase_y, lowercase_y = batch
                    else:
                        x = batch[0]
                        digit_y, uppercase_y, lowercase_y = batch[1]
                    x = x.to(self.args.device, non_blocking=True)
                    digit_y = digit_y.to(self.args.device, non_blocking=True)
                    uppercase_y = uppercase_y.to(self.args.device, non_blocking=True)
                    lowercase_y = lowercase_y.to(self.args.device, non_blocking=True)

                    original_y = digit_y.clone()
                    uppercase_mask = (uppercase_y < 26)
                    original_y[uppercase_mask] = uppercase_y[uppercase_mask] + 10
                    lowercase_mask = (lowercase_y < 26)
                    original_y[lowercase_mask] = lowercase_y[lowercase_mask] + 36
                    y = original_y.long()
                    y_pair = (y, y)
                    lam = 1.0
                else:
                    x, y = batch
                    x = x.to(self.args.device, non_blocking=True)
                    y = y.to(self.args.device, non_blocking=True)

                    r = random.random()
                    if r < 0.4 and cur_cut > 0:
                        x, y_pair, lam = GPUAug.cutmix(x, y, alpha=cur_cut)
                    elif r < 0.85 and cur_mix > 0:
                        x, y_pair, lam = GPUAug.mixup(x, y, alpha=cur_mix)
                    else:
                        y_pair = (y, y)
                        lam = 1.0
            except Exception as e:
                if self.args.rank0:
                    print(f"Error unpacking batch at iter {it}: {e}")
                continue

            if x is None or y is None:
                if self.args.rank0:
                    print(f"Skipping bad batch at iter {it}: x is None or y is None")
                continue

            try:
                if getattr(self.args, "device", None) is not None and self.args.device.type == 'cuda' and isinstance(x, torch.Tensor) and x.dim() >= 4:
                    x = x.contiguous(memory_format=torch.channels_last)
            except Exception as e:
                if self.args.rank0:
                    print(f"Warning: failed to convert x to channels_last at iter {it}: {e}")

            if self.args.gpu_augment and self.gpu_aug is not None:
                try:
                    aug_x = self.gpu_aug(x)
                    if aug_x is None:
                        raise RuntimeError("GPUAug returned None")
                    if isinstance(aug_x, torch.Tensor) and aug_x.device != x.device:
                        aug_x = aug_x.to(self.args.device)
                    if getattr(self.args, "device", None) is not None and self.args.device.type == 'cuda' and isinstance(aug_x, torch.Tensor) and aug_x.dim() >= 4:
                        aug_x = aug_x.contiguous(memory_format=torch.channels_last)
                    x = aug_x
                except Exception as e:
                    if self.args.rank0:
                        print("ERROR in augmentation:", e)

            accum_step = ((it) % int(self.args.accum)) == 0

            sync_ctx = self.get_accum_context(accum_step)

            with sync_ctx:
                with torch.amp.autocast('cuda', enabled=use_amp):
                    combined_output, penalty = self.model(x)
                    out = combined_output

                    if not self.args.use_specialized_moe and lam != 1.0:
                        cls_loss = mixup_criterion(self.criterion, out, y_pair, lam)
                    else:
                        cls_loss = self.criterion(out, y)

                    penalty_value = penalty if isinstance(penalty, torch.Tensor) else torch.tensor(penalty, device=x.device)
                    penalty_debug.append(penalty_value.detach().cpu().item())

                    total_loss = cls_loss + penalty_weight * penalty_value
                    loss_for_backward = total_loss / float(self.args.accum)

                with torch.no_grad():
                    batch_bs = x.size(0)
                    loss_log_val = total_loss.detach().cpu().item()
                    penalty_log_val = penalty_value.detach().cpu().item()
                    cls_loss_val = cls_loss.detach().cpu().item()

                self.scaler.scale(loss_for_backward).backward()

            if it % 50 == 0:
                with torch.no_grad():
                    gate_output = self.model.gate(x)
                    top_experts = gate_output.argmax(dim=1)
                    for idx in top_experts:
                        if idx < expert_usage.numel():
                            expert_usage[idx] += 1

            if accum_step:
                if hasattr(self.scaler, "unscale_"):
                    try:
                        self.scaler.unscale_(self.optimizer)
                    except Exception:
                        pass

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.args.grad_clip,
                    error_if_nonfinite=False
                )

                try:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                except Exception as e:
                    if self.args.rank0:
                        print("Optimizer step under scaler failed:", e)
                    try:
                        self.optimizer.step()
                    except Exception as e2:
                        if self.args.rank0:
                            print("Fallback optimizer.step failed:", e2)
                    try:
                        self.scaler.update()
                    except Exception:
                        pass

                try:
                    from torch.optim.lr_scheduler import OneCycleLR
                    if isinstance(self.scheduler, OneCycleLR):
                        try:
                            self.scheduler.step()
                        except Exception as e:
                            if self.args.rank0:
                                print("Warning: per-step scheduler.step() failed:", e)
                except Exception:
                    pass

                self.optimizer.zero_grad(set_to_none=True)

                if self.ema is not None:
                    try:
                        self.ema.update(self.model)
                    except Exception:
                        pass

            running_loss += loss_log_val * batch_bs
            running_penalty += penalty_log_val * batch_bs
            running_cls_loss += cls_loss_val * batch_bs
            n_samples += batch_bs

            if self.args.rank0:
                avg_loss = running_loss / n_samples if n_samples > 0 else 0.0
                avg_pen = running_penalty / n_samples if n_samples > 0 else 0.0
                avg_cls = running_cls_loss / n_samples if n_samples > 0 else 0.0
                lr = self.optimizer.param_groups[0]['lr']

                expert_info = ""
                if it % 100 == 0 and expert_usage.sum() > 0:
                    expert_pct = (expert_usage / expert_usage.sum() * 100).cpu().numpy()
                    expert_info = f" | Experts: {', '.join([f'{p:.1f}%' for p in expert_pct])}"
                    expert_usage.zero_()

                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    cls_loss=f"{avg_cls:.4f}",
                    penalty=f"{avg_pen:.6f}",
                    lr=f"{lr:.2e}{expert_info}"
                )

            if it % 100 == 0 and self.args.rank0 and len(penalty_debug) > 0:
                avg_penalty = sum(penalty_debug) / len(penalty_debug)
                min_penalty = min(penalty_debug)
                max_penalty = max(penalty_debug)
                print(f"Step {it}: Penalty - avg={avg_penalty:.6f}, min={min_penalty:.6f}, max={max_penalty:.6f}, weight={penalty_weight:.4f}")
                penalty_debug = []

            pbar.update(1)

        pbar.close()

        avg_loss = running_loss / n_samples if n_samples > 0 else 0.0
        avg_penalty = running_penalty / n_samples if n_samples > 0 else 0.0
        avg_cls_loss = running_cls_loss / n_samples if n_samples > 0 else 0.0

        if self.args.rank0:
            print(f"Epoch {epoch} Summary: Loss={avg_loss:.4f}, CLS_Loss={avg_cls_loss:.4f}, Penalty={avg_penalty:.6f}")

        return avg_loss, avg_penalty

    def validate(self, tta=False, epoch=None):
        """Validate the model."""
        self.model.eval()
        total = 0
        correct = 0
        val_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                # Handle both standard and specialized MoE datasets
                if self.args.use_specialized_moe:
                    # Unpack 4 values: (x, digit_y, uppercase_y, lowercase_y)
                    x, digit_y, uppercase_y, lowercase_y = batch
                    # Convert to original EMNIST labels for evaluation
                    y = digit_y.clone()
                    uppercase_mask = (uppercase_y < 26)
                    y[uppercase_mask] = uppercase_y[uppercase_mask] + 10
                    lowercase_mask = (lowercase_y < 26)
                    y[lowercase_mask] = lowercase_y[lowercase_mask] + 36
                else:
                    # Standard dataset: (x, y)
                    x, y = batch

                x = x.to(self.args.device, non_blocking=True)
                y = y.to(self.args.device, non_blocking=True)

                # Handle hierarchical MoE
                combined_output, _ = self.model(x)
                out = combined_output

                if not tta:
                    loss = self.criterion(out, y)
                    val_loss += loss.item() * x.size(0)
                    preds = out.argmax(1)
                    correct += (preds == y).sum().item()
                    total += x.size(0)
                else:
                    # Improved TTA with vectorized rotations
                    variants = [x, torch.flip(x, dims=[3])]

                    # Vectorized TTA rotations
                    angles = [10, -10]
                    for ang in angles:
                        # Create rotation matrices for the whole batch
                        B, C, H, W = x.shape
                        angle_rad = -ang * math.pi / 180.0
                        cos = torch.cos(torch.tensor(angle_rad, device=self.args.device))
                        sin = torch.sin(torch.tensor(angle_rad, device=self.args.device))

                        # Create affine matrices
                        mats = torch.zeros((B, 2, 3), device=self.args.device)
                        mats[:, 0, 0] = cos
                        mats[:, 0, 1] = -sin
                        mats[:, 1, 0] = sin
                        mats[:, 1, 1] = cos

                        # Generate grid and sample
                        grid = F.affine_grid(mats, x.size(), align_corners=False)
                        vn = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
                        variants.append(vn)

                    probs = None
                    for v in variants:
                        outv, _ = self.model(v)
                        pv = F.softmax(outv, dim=1)
                        probs = pv if probs is None else probs + pv
                    probs = probs / len(variants)
                    val_loss += - (probs.log() * F.one_hot(y, num_classes=probs.size(1)).to(probs.device)).sum(
                        dim=1).mean().item() * x.size(0)
                    preds = probs.argmax(1)
                    correct += (preds == y).sum().item()
                    total += x.size(0)

        val_loss = val_loss / total if total > 0 else 0.0
        val_acc = 100.0 * (correct / total) if total > 0 else 0.0

        return val_loss, val_acc

    def warmup_model(self):
        """Warm up the model to avoid initial slow training."""
        print("Warming up model...")
        self.model.eval()

        # Create dummy input
        dummy_input = torch.randn(2, 1, 28, 28, device=self.args.device)  # Use batch size 2
        if self.args.resize is not None:
            dummy_input = F.interpolate(dummy_input, size=(self.args.resize, self.args.resize),
                                        mode='bilinear', align_corners=False)

        # Simple warmup - just run the model once without iterating through experts
        with torch.no_grad():
            _ = self.model(dummy_input)

        torch.cuda.empty_cache()
        self.model.train()
        print("Warm-up complete")

    def optimize_gradient_accumulation(self):
        """
        Safe setup for gradient accumulation.

        DO NOT override optimizer.step (that breaks updates).
        This function only:
          - stores accumulation count,
          - ensures DDP sync flag is left in a sane state,
          - exposes a helper to get the correct context manager (no_sync vs nullcontext).

        Usage in training loop:
            accum_step = (it % self.args.accum) == 0
            ctx = self.get_accum_context(accum_step)
            with ctx:
                ... optimizer step logic ...
        """
        # store accumulation factor (int)
        self.accum = int(getattr(self.args, "accum", 1))

        # ensure the model's require_backward_grad_sync exists and is True by default
        if hasattr(self.model, "require_backward_grad_sync"):
            try:
                # keep default True so DDP does sync unless we explicitly use no_sync()
                self.model.require_backward_grad_sync = True
            except Exception:
                # ignore if not settable
                pass

        # provide a small helper to get correct context manager for accumulation steps
        # returns model.no_sync() when available and we are not at an accumulation step
        def _get_context(accum_step: bool):
            # accum_step == True means "time to step" -> use normal sync (nullcontext)
            if hasattr(self.model, "no_sync") and not accum_step:
                return self.model.no_sync()
            else:
                return nullcontext()

        self.get_accum_context = _get_context

        # optional debug message (only on rank0)
        if getattr(self.args, "rank0", False):
            print(f"[Trainer] Gradient accumulation setup: accum={self.accum}. "
                  f"DDP no_sync helper available: {hasattr(self.model, 'no_sync')}")

    def optimize_memory(self):
        """Optimize memory usage during training."""
        # Use channels_last memory format for better performance
        self.model = self.model.to(memory_format=torch.channels_last)

        # Enable cuDNN benchmarking for optimal performance
        torch.backends.cudnn.benchmark = True

        # Enable TF32 for Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Set better memory allocation strategy
        torch.cuda.set_per_process_memory_fraction(0.9)  # Use 80% of GPU memory


def setup_distributed_environment(args):
    """Setup distributed training environment."""
    # DDP related env
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_rank = int(os.environ.get("RANK", "0"))
    args.world_size = world_size
    args.rank = global_rank
    args.local_rank = local_rank
    args.rank0 = (global_rank == 0)

    if args.distributed:
        # init only when requested (torchrun sets envs)
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        # single GPU (use first CUDA)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0

    args.device = device
    return device, local_rank


def setup_seeds(args):
    """Setup random seeds for reproducibility."""
    random.seed(2137 + args.rank)
    np.random.seed(2137 + args.rank)
    torch.manual_seed(2137 + args.rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2137 + args.rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')


def load_emnist_data(args):
    """Load and preprocess EMNIST data."""
    if args.rank0:
        print("Master downloading EMNIST (byclass)...")
        datasets.EMNIST(root=args.data_dir, split='byclass', train=True, download=True)
        datasets.EMNIST(root=args.data_dir, split='byclass', train=False, download=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    raw_train = datasets.EMNIST(root=args.data_dir, split='byclass', train=True, download=False)
    raw_test = datasets.EMNIST(root=args.data_dir, split='byclass', train=False, download=False)

    if args.max_train:
        raw_train.data = raw_train.data[:args.max_train].clone()
        raw_train.targets = raw_train.targets[:args.max_train].clone()
    if args.max_test:
        raw_test.data = raw_test.data[:args.max_test].clone()
        raw_test.targets = raw_test.targets[:args.max_test].clone()

    return raw_train, raw_test


def create_model(args, device):
    """Create the model based on arguments."""
    if args.rank0:
        print(f"Using MoE with {args.num_digit_experts} digit experts, "
              f"{args.num_uppercase_experts} uppercase experts, and {args.num_lowercase_experts} lowercase experts")

    model = MoE(
        num_digit_experts=args.num_digit_experts,
        num_uppercase_experts=args.num_uppercase_experts,
        num_lowercase_experts=args.num_lowercase_experts,
        use_batchnorm=args.use_batchnorm,
        channel_mult=args.channel_mult,
        gradient_checkpointing=args.gradient_checkpointing,
        unknown_threshold=args.unknown_threshold
    ).to(device)

    # Safer compilation for MoE models
    if args.torch_compile and hasattr(torch, "compile"):
        try:
            # Disable CUDA Graphs and use more conservative settings
            import torch._inductor.config as config
            config.triton.cudagraphs = False

            compiled_model = torch.compile(
                model,
                fullgraph=False,
                dynamic=False,
                options={
                    "triton.cudagraphs": False,
                }
            )

            if args.rank0:
                print("Model compiled with CUDA Graphs disabled")
            return compiled_model
        except Exception as e:
            if args.rank0:
                print(f"Warning: torch.compile failed: {e}")
            args.torch_compile = False

    return model


def create_training_components(model, args, train_loader=None):
    """Create criterion, optimizer, scheduler, and scaler."""
    # criterion
    if args.use_focal:
        criterion = FocalLoss(gamma=1.5).to(args.device)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(args.device)

    # Better LR scaling for Lion optimizer
    effective_batch_size = args.per_gpu_batch * args.accum * (args.world_size if args.distributed else 1)

    # For Lion optimizer, use linear scaling (not sqrt) and slightly higher base LR
    base_lr = args.lr
    scaled_lr = base_lr * (effective_batch_size / 256.0)

    # Keep Lion optimizer but with better LR
    optimizer = Lion(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=args.weight_decay
    )

    if args.rank0:
        print(f"Effective batch size: {effective_batch_size}, Base LR: {base_lr:.2e}, Scaled LR: {scaled_lr:.2e}")

    # Use CosineAnnealingWarmRestarts for better convergence with Lion
    if train_loader is not None:
        # Cosine annealing with warm restarts - works well with Lion
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=20,  # Restart every 20 epochs
            T_mult=2,  # Double the cycle length each time
            eta_min=scaled_lr * 0.01  # Minimum LR is 1% of scaled LR
        )
    else:
        # Fallback
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    return criterion, optimizer, scheduler, scaler


def main():
    args = parse_args()
    device, local_rank = setup_distributed_environment(args)
    setup_seeds(args)

    raw_train, raw_test = load_emnist_data(args)

    data_loader = EMNISTDataLoader(args)

    if args.use_specialized_moe:
        train_loader, val_loader = data_loader.create_specialized_dataloaders(raw_train, raw_test)
    else:
        train_loader, val_loader = data_loader.create_standard_dataloaders(raw_train, raw_test)

    if args.rank0:
        print(f"Train {len(train_loader.dataset)} | Val {len(val_loader.dataset)}")

    model = create_model(args, device)

    criterion, optimizer, scheduler, scaler = create_training_components(model, args, train_loader)

    # Debug prints: scheduler type & initial LR
    if args.rank0:
        print("Scheduler type:", type(scheduler).__name__)
        print("Initial LR:", optimizer.param_groups[0]["lr"])
        print("Optimizer:", type(optimizer).__name__)

    best_acc, start_epoch = load_checkpoint_if_exists(model, optimizer, scheduler, scaler, args)

    # Reset best_acc if starting fresh to track improvements properly
    if start_epoch == 0:
        best_acc = 0.0

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, args)

    trainer.warmup_model()

    # Training metrics tracking
    max_patience = 25
    patience = 0
    best_acc = best_acc  # This might be loaded from checkpoint

    for epoch in range(start_epoch, args.epochs + 1):
        if args.distributed and dist.is_initialized():
            train_loader.sampler.set_epoch(epoch)

        train_loss, train_penalty = trainer.train_one_epoch(epoch)

        # Step the scheduler (CosineAnnealingWarmRestarts steps at epoch end)
        try:
            scheduler.step()
        except Exception as e:
            if args.rank0:
                print("Warning: scheduler.step() failed:", e)

        val_loss, val_acc = trainer.validate(tta=False, epoch=epoch)

        if args.rank0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[E{epoch:03d}] train_loss={train_loss:.4f} train_penalty={train_penalty:.4f} "
                  f"val_loss={val_loss:.4f} acc={val_acc:.3f} lr={current_lr:.2e}")

            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
                patience = 0
                if args.rank0:
                    print(f"New best accuracy: {best_acc:.3f}%")
            else:
                patience += 1
                if args.rank0:
                    print(f"⏳ No improvement for {patience} epochs (best: {best_acc:.3f}%)")

            if patience >= max_patience:
                if args.rank0:
                    print(f"Early stopping at epoch {epoch} - no improvement for {max_patience} epochs")
                break

            # Create checkpoint state
            checkpoint = {
                'epoch': epoch,
                'val_acc': val_acc,
                'best_acc': best_acc,
                'model_state_dict': (model.module.state_dict() if hasattr(model, "module") else model.state_dict()),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'args': vars(args)
            }

            # Save checkpoints
            try:
                os.makedirs(os.path.dirname(args.save) if os.path.dirname(args.save) else ".", exist_ok=True)

                if is_best:
                    best_path = str(Path(args.save).with_name("best_model.pt"))
                    save_checkpoint(checkpoint, best_path)
                    if args.rank0:
                        print(f"Saved best model to {best_path}")

                latest_path = str(Path(args.save).with_name("latest_model.pt"))
                save_checkpoint(checkpoint, latest_path)

            except Exception as e:
                if args.rank0:
                    print("Warning: failed to save checkpoint:", e)

        # SWA update if enabled and past SWA start epoch
        if hasattr(trainer, 'swa_model') and trainer.swa_model is not None and epoch >= trainer.swa_start:
            trainer.swa_model.update_parameters(model)
            if hasattr(trainer, 'swa_scheduler'):
                trainer.swa_scheduler.step()

    # Final SWA and EMA updates
    if args.rank0:
        if hasattr(trainer, 'swa_model') and trainer.swa_model is not None:
            print("Applying SWA...")
            # Update batch norm statistics for SWA
            torch.optim.swa_utils.update_bn(train_loader, trainer.swa_model, device=args.device)
            final_model = trainer.swa_model
        elif trainer.ema is not None:
            print("Applying EMA...")
            trainer.ema.copy_to_model(model if not hasattr(model, 'module') else model.module)
            final_model = model
        else:
            final_model = model

        # Save final model
        final_state = {
            "model_state_dict": (
                final_model.module.state_dict() if hasattr(final_model, "module") else final_model.state_dict()),
            "args": vars(args),
            "final_accuracy": best_acc
        }
        try:
            final_path = str(Path(args.save).with_name("final_model.pt"))
            save_checkpoint(final_state, final_path)
            print(f"Saved final model to {final_path}")

            # Also save as safetensors if requested
            if args.export_safetensors:
                from utils.checkpoint import save_safetensors
                safetensors_path = str(Path(args.save).with_name("final_model.safetensors"))
                save_safetensors(final_model, safetensors_path)
                print(f"Saved final model as safetensors to {safetensors_path}")

        except Exception as e:
            print("Warning: failed to save final model:", e)

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
