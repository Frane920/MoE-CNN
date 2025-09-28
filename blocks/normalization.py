# blocks/normalization.py
import torch.nn as nn

def GN(ch):
    g = min(8, max(1, ch // 8))
    return nn.GroupNorm(g, ch)

def BN(ch):
    return nn.BatchNorm2d(ch, track_running_stats=True)
