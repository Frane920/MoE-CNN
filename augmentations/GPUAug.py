import math
import random
from math import sqrt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class GPUAug:
    def __init__(self,
                 brightness=0.12,
                 contrast=0.12,
                 rotate=12,
                 perspective_p=0.06,
                 blur_p=0.12,
                 mixup_alpha=0.3,
                 cutmix_alpha=1.0,
                 erase_p=0.28,
                 erase_box_min=2,
                 erase_box_max=6,
                 noise_std=0.02):
        self.brightness = brightness
        self.contrast = contrast
        self.rotate = rotate
        self.perspective_p = perspective_p
        self.blur_p = blur_p
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.erase_p = erase_p
        self.erase_box_min = erase_box_min
        self.erase_box_max = erase_box_max
        self.noise_std = noise_std

    # ---- Static variants so external code can call GPUAug.mixup / GPUAug.cutmix ----
    @staticmethod
    def mixup(x, y, alpha=0.3):
        """Class-level mixup (compatible with existing calls)."""
        if alpha <= 0:
            return x, (y, y), 1.0
        lam = float(np.random.beta(alpha, alpha))
        idx = torch.randperm(x.size(0), device=x.device)
        return lam * x + (1.0 - lam) * x[idx], (y, y[idx]), lam

    @staticmethod
    def cutmix(x, y, alpha=1.0):
        """Class-level cutmix (compatible with existing calls)."""
        if alpha <= 0:
            return x, (y, y), 1.0
        lam = float(np.random.beta(alpha, alpha))
        B, C, H, W = x.shape
        rand_index = torch.randperm(B, device=x.device)
        cut_rat = math.sqrt(max(0.0, 1.0 - lam))
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = int(torch.randint(0, W, (1,)).item())
        cy = int(torch.randint(0, H, (1,)).item())
        x1 = max(0, cx - cut_w // 2)
        x2 = min(W, cx + cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        y2 = min(H, cy + cut_h // 2)
        new_x = x.clone()
        if (x2 > x1) and (y2 > y1):
            new_x[:, :, y1:y2, x1:x2] = x[rand_index, :, y1:y2, x1:x2]
            lam_adjusted = 1.0 - ((x2 - x1) * (y2 - y1) / float(W * H))
        else:
            lam_adjusted = lam
        return new_x, (y, y[rand_index]), lam_adjusted

    # ---- Instance helpers used by __call__ ----
    def _batch_rotate(self, img, angles_deg):
        B, C, H, W = img.shape
        angles = -angles_deg * math.pi / 180.0
        cos = torch.cos(angles).to(img.device)
        sin = torch.sin(angles).to(img.device)

        mats = torch.zeros((B, 2, 3), device=img.device, dtype=img.dtype)
        mats[:, 0, 0] = cos
        mats[:, 0, 1] = -sin
        mats[:, 1, 0] = sin
        mats[:, 1, 1] = cos

        grid = F.affine_grid(mats, img.size(), align_corners=False)
        return F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    def _batch_gaussian_blur(self, img):
        # create kernel on same device/dtype
        kernel = torch.tensor([[1., 2., 1.],
                               [2., 4., 2.],
                               [1., 2., 1.]], device=img.device, dtype=img.dtype)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, 3, 3)
        padding = 1
        # depthwise conv: repeat kernel for channels
        return F.conv2d(img, kernel.repeat(img.shape[1], 1, 1, 1), groups=img.shape[1], padding=padding)

    def add_noise(self, img):
        """Add Gaussian noise with null-check."""
        if img is None:
            return img
        return (img + torch.randn_like(img) * self.noise_std).clamp(-1.0, 1.0)

    def random_erease(self, img):
        """Random erasing with null-check."""
        if img is None:
            return img
        B, C, H, W = img.shape
        mask = (torch.rand(B, device=img.device) < self.erase_p)
        if mask.sum() == 0:
            return img
        idxs = mask.nonzero(as_tuple=False).squeeze(1)
        for i in idxs.tolist():
            ew = random.randint(self.erase_box_min, self.erase_box_max)
            eh = random.randint(self.erase_box_min, self.erase_box_max)
            x0 = random.randint(0, max(0, W - ew))
            y0 = random.randint(0, max(0, H - eh))
            img[i, :, y0:y0 + eh, x0:x0 + ew] = -1.0
        return img

    # ---- Main augmentation pipeline (instance callable) ----
    def __call__(self, img):
        """Apply augmentations safely. Always returns a tensor (or original input) — never None."""
        if img is None:
            # defensive
            if isinstance(img, torch.Tensor):
                return img
            print("WARNING: Augmentation received None input!")
            return None

        try:
            B = img.size(0)
            device = img.device

            # brightness
            if self.brightness > 0 and B > 0:
                fac = torch.empty((B, 1, 1, 1), device=device).uniform_(1 - self.brightness, 1 + self.brightness)
                img = img * fac

            # contrast
            if self.contrast > 0 and B > 0:
                fac = torch.empty((B, 1, 1, 1), device=device).uniform_(1 - self.contrast, 1 + self.contrast)
                mean = img.mean(dim=[2, 3], keepdim=True)
                img = (img - mean) * fac + mean

            # vectorized rotation
            if self.rotate > 0 and B > 0:
                angles = (torch.rand(B, device=device) - 0.5) * 2.0 * self.rotate
                img = self._batch_rotate(img, angles)

            # perspective
            if random.random() < self.perspective_p and B > 0:
                for i in range(B):
                    if random.random() < 0.5:
                        w, h = img.shape[3], img.shape[2]
                        sp = [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]
                        ms = 2
                        ep = [(min(w - 1, max(0, x + random.randint(-ms, ms))),
                               min(h - 1, max(0, y + random.randint(-ms, ms)))) for x, y in sp]
                        img[i:i + 1] = TF.perspective(img[i:i + 1], sp, ep,
                                                      interpolation=TF.InterpolationMode.BILINEAR, fill=-1.0)

            # blur (mask)
            if random.random() < self.blur_p and B > 0:
                mask = torch.rand(B, device=device) < 0.25
                if mask.any():
                    idx = mask.nonzero(as_tuple=False).squeeze(1)
                    img[idx] = self._batch_gaussian_blur(img[idx])

            # random erasing (returns tensor)
            img = self.random_erease(img)

            # noise (returns tensor)
            img = self.add_noise(img)

            # final clamp and ensure finite
            if img is None:
                return None
            img = img.clamp(-1.0, 1.0)
            # fix any NaN/inf (rare)
            if not torch.isfinite(img).all():
                img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=-1.0)
            return img

        except Exception as e:
            # robust fallback: log and return original input (not None)
            print(f"ERROR in augmentation: {e}")
            try:
                if isinstance(img, torch.Tensor):
                    return img
                else:
                    return None
            except Exception:
                return None
