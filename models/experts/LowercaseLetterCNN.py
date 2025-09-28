# models/experts/DigitCNN.py

import torch
from torch import nn
import torch.nn.functional as F
from blocks.SEBlock import SEBlock
from blocks.ResidualBlock import ResidualBlock
from blocks.normalization import BN, GN

NUM_LOWERCASE = 26

class LowercaseCNN(nn.Module):
    def __init__(self, use_batchnorm=False, channel_mult=1.0):
        super().__init__()
        self.num_classes = NUM_LOWERCASE
        self.unknown_class = self.num_classes

        def make_ch(x):
            return max(16, int(x * channel_mult))

        def block(in_ch, out_ch, pool=True):
            layers = [
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                BN(out_ch) if use_batchnorm else GN(out_ch),
                nn.SiLU(inplace=True),
                SEBlock(out_ch),
                ResidualBlock(out_ch, use_batchnorm=use_batchnorm)
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            layers.append(nn.Dropout(0.25))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            block(1, make_ch(32)),
            block(make_ch(32), make_ch(64)),
            block(make_ch(64), make_ch(128)),
            block(make_ch(128), make_ch(256)),
            block(make_ch(256), make_ch(512), pool=False),
        )

        self.fc = nn.Sequential(
            nn.Linear(make_ch(512), 1024),
            nn.LayerNorm(1024),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, self.num_classes + 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        return self.fc(x)

    def get_unknown_confidence(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            return probs[:, self.unknown_class]
