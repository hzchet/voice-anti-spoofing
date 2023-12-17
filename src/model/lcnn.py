from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from src.base.base_model import BaseModel


class MFM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.shape[1] % 2 == 0
        out1, out2 = torch.chunk(x, 2, dim=1)
        return torch.max(out1, out2)


class AngularMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, x, label, is_inference: bool = False):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        if is_inference:
            return cosine
            
        phi = cosine - self.m
        
        phi = torch.where(cosine > 0, phi, cosine)
    
        one_hot = torch.zeros(cosine.size(), device=x.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class LCNN(BaseModel):
    def __init__(
        self,
        n_frames: int = 257,
        time: int = 750,
        dropout: float = 0.0,
        use_angular_margin: bool = False,
        scale: float = None,
        margin: float = None
    ):
        super().__init__()
        self.use_angular_margin = use_angular_margin
        if self.use_angular_margin:
            assert scale is not None and margin is not None
            
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
        self.embedding_extractor = nn.Sequential(
            nn.Linear(32 * (n_frames // 16) * (time // 16), 160),
            MFM(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(80)
        )
        if self.use_angular_margin:
            self.head = AngularMarginProduct(80, 2, s=scale, m=margin)
        else:
            self.head = nn.Linear(80, 2)

    def forward(self, spectrogram, label, is_inference: bool = False, **batch):
        x = self.net(spectrogram)
        feature_embedding = self.embedding_extractor(x.flatten(start_dim=1))
        
        if self.use_angular_margin:
            logits = self.head(feature_embedding, label, is_inference)
        else:
            logits = self.head(feature_embedding)
        
        return {
            "logits": logits
        }
