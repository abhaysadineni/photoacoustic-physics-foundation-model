"""Transformer surrogate.

Input: four consecutive normalised frames, shape (B, 4, 3, 128).
Optional derivative channels: spatial derivative (second-order central difference,
one-sided at the two edges, for every boundary condition, so the network is never shown the
periodic wrap and is not told the boundary condition) and temporal derivative (backward difference between frames,
zero for the first frame). With derivatives each frame has 9 channels, without it has 3.

Tokens: space-time patches of 4 frames x 8 grid points, so 16 tokens per sample.
Core: pre-norm transformer encoder.
Output: rate of change of the 3 x 128 field per frame step. The next frame is one forward
Euler step: u_next = u_last + dudt * 1 (time measured in frame steps).
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_IN_FRAMES = 4
PATCH_X = 8


def spatial_derivative(u: torch.Tensor) -> torch.Tensor:
    """d/dx in grid units along the last axis; central inside, one-sided second order at edges."""
    d = torch.empty_like(u)
    d[..., 1:-1] = 0.5 * (u[..., 2:] - u[..., :-2])
    d[..., 0] = -1.5 * u[..., 0] + 2.0 * u[..., 1] - 0.5 * u[..., 2]
    d[..., -1] = 1.5 * u[..., -1] - 2.0 * u[..., -2] + 0.5 * u[..., -3]
    return d


def temporal_derivative(u: torch.Tensor) -> torch.Tensor:
    """Backward difference along the frame axis (dim 1); zero for the first frame."""
    d = torch.zeros_like(u)
    d[:, 1:] = u[:, 1:] - u[:, :-1]
    return d


class Surrogate(nn.Module):
    def __init__(self, n_ch: int = 3, n_x: int = 128, width: int = 192, depth: int = 6,
                 heads: int = 6, use_derivs: bool = True, dropout: float = 0.0):
        super().__init__()
        self.n_ch, self.n_x, self.use_derivs = n_ch, n_x, use_derivs
        feat_ch = n_ch * (3 if use_derivs else 1)
        self.n_tok = n_x // PATCH_X
        self.embed = nn.Linear(N_IN_FRAMES * feat_ch * PATCH_X, width)
        self.pos = nn.Parameter(torch.zeros(1, self.n_tok, width))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.core = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, n_ch * PATCH_X)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_derivs:
            x = torch.cat([x, spatial_derivative(x), temporal_derivative(x)], dim=2)
        return x                                               # (B, 4, C, 128)

    def rate(self, x: torch.Tensor) -> torch.Tensor:
        """Predicted rate of change per frame step, shape (B, 3, 128)."""
        b = x.shape[0]
        f = self.features(x)                                   # (B, 4, C, 128)
        f = f.reshape(b, N_IN_FRAMES, -1, self.n_tok, PATCH_X)  # (B, 4, C, T, P)
        f = f.permute(0, 3, 1, 2, 4).reshape(b, self.n_tok, -1)
        h = self.core(self.embed(f) + self.pos)
        out = self.head(self.norm(h))                          # (B, T, C*P)
        out = out.reshape(b, self.n_tok, self.n_ch, PATCH_X)
        return out.permute(0, 2, 1, 3).reshape(b, self.n_ch, self.n_x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Next frame by one forward Euler step from the last input frame."""
        return x[:, -1] + self.rate(x)

    @torch.no_grad()
    def rollout(self, x0: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Autoregressive rollout from four seed frames; returns (B, n_steps, 3, 128)."""
        window, preds = x0, []
        for _ in range(n_steps):
            nxt = self(window)
            preds.append(nxt)
            window = torch.cat([window[:, 1:], nxt[:, None]], dim=1)
        return torch.stack(preds, dim=1)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
