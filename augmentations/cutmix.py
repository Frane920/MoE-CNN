from torch import randperm
import numpy as np
from math import sqrt

@staticmethod
def cutmix(x, y, alpha=1.0):
    """CutMix augmentation."""
    if alpha <= 0:
        return x, (y, y), 1.0
    lam = np.random.beta(alpha, alpha)
    B, C, H, W = x.shape
    rand_index = randperm(B, device=x.device)
    cut_rat = sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(0, W)
    cy = np.random.randint(0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    new_x = x.clone()
    if x2 > x1 and y2 > y1:
        new_x[:, :, y1:y2, x1:x2] = x[rand_index, :, y1:y2, x1:x2]
        lam_adjusted = 1.0 - ((x2 - x1) * (y2 - y1) / (W * H))
    else:
        lam_adjusted = lam
    return new_x, (y, y[rand_index]), lam_adjusted
