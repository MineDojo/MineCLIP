"""
Note that image feature is provided by MineCLIP.
"""
import torch.nn as nn


class DummyImgFeat(nn.Module):
    def __init__(self, output_dim: int = 512):
        super().__init__()
        self._output_dim = output_dim

    @property
    def output_dim(self):
        return self._output_dim

    def forward(self, x, **kwargs):
        return x, None
