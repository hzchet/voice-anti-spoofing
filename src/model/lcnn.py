from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.base.base_model import BaseModel


class MFM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.shape[1] % 2 == 0
        out1, out2 = torch.chunk(x, 2, dim=1)
        return torch.max(out1, out2)


class LCNN(BaseModel):
    def __init__(
        self,
        n_frames: int = 257,
        time: int = 750,
        dropout: float = 0.0
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            MFM(),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=1),
            MFM(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 96, kernel_size=3, padding=1),
            MFM(),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(48),

            nn.Conv2d(48, 96, kernel_size=1),
            MFM(),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 128, kernel_size=3, padding=1),
            MFM(),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=1),
            MFM(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            MFM(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=1),
            MFM(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            MFM(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.head = nn.Sequential(
            nn.Linear(32 * (n_frames // 16) * (time // 16), 160),
            MFM(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(80),
            nn.Linear(80, 2)
        )
    
    def forward(self, spectrogram, **batch):
        x = self.net(spectrogram)
        return {
            "logits": self.head(x.flatten(start_dim=1))
        }
