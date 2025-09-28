from torch import nn

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, c, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, max(4, c // r), 1), nn.SiLU(inplace=True),
            nn.Conv2d(max(4, c // r), c, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)