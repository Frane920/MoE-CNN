from torch import randperm
import numpy as np

@staticmethod
def mixup(x, y, alpha=0.3):
    """Mixup augmentation."""
    if alpha <= 0:
        return x, (y, y), 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = randperm(x.size(0), device=x.device)
    return lam * x + (1.0 - lam) * x[idx], (y, y[idx]), lam