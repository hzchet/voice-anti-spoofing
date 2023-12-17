from typing import Dict, List

import torch.nn as nn
import torch.nn.functional as F

from src.model.sinc import SincLayer
from src.base.base_model import BaseModel


class FMS(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.scaler = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
        out = self.scaler(out).view(x.size(0), x.size(1), -1)
        return out + out * x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        maxpool_kernel_size: int
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.LeakyReLU(0.3),
            nn.Conv1d(in_channels, out_channels, kernel_size),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.3),
            nn.Conv1d(out_channels, out_channels, kernel_size),
            nn.MaxPool1d(maxpool_kernel_size),
            FMS(out_channels)
        )

    def forward(self, x):
        return self.block(x)


class RawNet2(BaseModel):
    def __init__(
        self,
        sinc_kwargs: Dict,
        channels: List[int],
        kernel_sizes: List[int],
        maxpool_kernel_sizes: List[int],
        gru_kwargs: Dict,
        pre_gru_activations: bool,
        fc_n_features: int
    ):
        super().__init__()
        self.sinc_layer = SincLayer(**sinc_kwargs)
        res_blocks = []
        for in_channels, out_channels, kernel_size, maxpool_kernel_size in zip(
            channels[:-1], channels[1:], kernel_sizes, maxpool_kernel_sizes
        ):
            res_blocks.append(
                ResBlock(in_channels, out_channels, kernel_size, maxpool_kernel_size)
            )
        self.res_blocks = nn.Sequential(*res_blocks)
        
        if pre_gru_activations:
            self.pre_gru_activations = nn.Sequential(
                nn.BatchNorm1d(channels[-1]),
                nn.LeakyReLU(0.3),
            )
        else:
            self.pre_gru_activations = nn.Identity()
        
        self.gru = nn.GRU(**gru_kwargs)
        
        self.head = nn.Sequential(
            nn.Linear(fc_n_features, fc_n_features),
            nn.Linear(fc_n_features, 2)
        )

    def forward(self, wav, **batch):
        x = self.sinc_layer(wav)
        x = self.res_blocks(x)
        x = self.pre_gru_activations(x)
        outputs, _ = self.gru(x.transpose(-1, -2))
        
        logits = self.head(outputs[:, -1, :])
        
        return {
            "logits": logits
        }
