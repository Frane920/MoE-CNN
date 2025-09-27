# blocks/ResidualBlock.py

from torch import nn
import torch.nn.functional as F

from blocks.normalization import GN, BN

class ResidualBlock(nn.Module):
    """Residual block with normalization and activation."""

    def __init__(self, c, use_batchnorm=False):
        super().__init__()
        norm = BN if use_batchnorm else GN  # Use BN if specified, otherwise GN (default)
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False), norm(c), nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1, bias=False), norm(c)
        )

    def forward(self, x):
        return F.silu(x + self.conv(x))
