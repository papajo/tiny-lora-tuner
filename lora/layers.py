import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    def __init__(self, orig, r: int = 8, alpha: int = 16, dropout: float = 0.1):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r

        self.W = orig.weight
        self.b = orig.bias
        self.W.requires_grad_(False)
        if self.b is not None:
            self.b.requires_grad_(False)

        in_f = orig.weight.shape[0]
        out_f = orig.weight.shape[1]
        self.A = nn.Parameter(torch.randn(in_f, r) * 0.02)
        self.B = nn.Parameter(torch.zeros(r, out_f))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.W.t(), self.b) + self.drop(x) @ self.A @ self.B * self.scale

    def merge_weights(self) -> torch.Tensor:
        return self.W + (self.A @ self.B) * self.scale
