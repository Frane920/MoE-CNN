import torch
import math
import torch.nn.functional as F

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