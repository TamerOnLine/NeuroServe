from __future__ import annotations

import torch
import torch.nn as nn

from .runtime import pick_device


class TinyNet(nn.Module):
    def __init__(self, in_features=512, hidden=1024, out_features=10):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU(), nn.Linear(hidden, out_features))

    def forward(self, x):
        return self.net(x)


def load_model():
    dev = pick_device()
    model = TinyNet().to(dev).eval()
    # warm weights
    with torch.no_grad():
        _ = model(torch.randn(1, 512, device=dev))
    return model, dev
