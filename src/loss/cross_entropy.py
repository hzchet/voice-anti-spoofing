import torch.nn as nn


class CrossEntropyLossWrapper(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        
    def forward(self, logits, label, **batch):
        return self.ce(logits, label)
