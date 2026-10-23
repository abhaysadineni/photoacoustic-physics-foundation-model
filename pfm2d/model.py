"""Patch transformer surrogate for d = 2 and d = 3, analogous to pfm/model.py.

Input: four consecutive normalised frames, (B, 4, 3, *grid), grid 64 x 64 (d = 2) or
32^3 (d = 3).
Optional derivative channels per frame: d/dx_a for every axis (second-order central
difference inside, one-sided second order at the two edges, for every boundary condition,
so the network is never shown the periodic wrap and is not told the boundary condition, as
1D) and d/dt (backward difference between frames, zero for the first frame). With
derivatives each frame has 3 (d + 2) channels (12 for d = 2, 15 for d = 3), without 3.
Tokens: spatial patches of 8^d cells with the four frames folded into the patch features
(4 x channels x 8^d values per token), so 64 tokens in both dimensions (8 x 8 patches of
the 64 x 64 grid; 4 x 4 x 4 patches of the 32^3 grid).
Position: a learned table per axis (8 or 4 entries of width W each) summed over the axes
(factorised learned 2D/3D position embedding, modelling choice).
Core: pre-norm transformer encoder (width 192, depth 6, heads 6 by default, GELU, MLP 4W).
Output: zero-initialised linear head giving the rate of change of the 3 x 8^d field per
patch per frame step; the next frame is one forward Euler step u_last + rate.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import common as C

N_IN_FRAMES = 4
PATCH = 8


def spatial_derivative(u: torch.Tensor, axis: int) -> torch.Tensor:
    """d/dx in grid units along tensor dimension `axis`."""
    u = u.movedim(axis, -1)
    d = torch.empty_like(u)
    d[..., 1:-1] = 0.5 * (u[..., 2:] - u[..., :-2])
    d[..., 0] = -1.5 * u[..., 0] + 2.0 * u[..., 1] - 0.5 * u[..., 2]
    d[..., -1] = 1.5 * u[..., -1] - 2.0 * u[..., -2] + 0.5 * u[..., -3]
    return d.movedim(-1, axis)


def temporal_derivative(u: torch.Tensor) -> torch.Tensor:
    d = torch.zeros_like(u)
    d[:, 1:] = u[:, 1:] - u[:, :-1]
    return d


class Surrogate(nn.Module):
    def __init__(self, dim: int = 2, n_ch: int = 3, n_x: int | None = None, width: int = 192,
                 depth: int = 6, heads: int = 6, use_derivs: bool = True, dropout: float = 0.0,
                 patch: int = PATCH):
        super().__init__()
        self.dim, self.n_ch, self.use_derivs, self.patch = dim, n_ch, use_derivs, patch
        self.n_x = n_x or C.GEOM[dim]["n_out"]
        self.side = self.n_x // patch
        self.n_tok = self.side ** dim
        feat_ch = n_ch * ((dim + 2) if use_derivs else 1)
        self.embed = nn.Linear(N_IN_FRAMES * feat_ch * patch ** dim, width)
        self.pos = nn.Parameter(torch.zeros(dim, self.side, width))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.core = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, n_ch * patch ** dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def pos_embedding(self):
        d, s, W = self.dim, self.side, self.pos.shape[-1]
        pe = 0.0
        for a in range(d):
            shp = [1] * d + [W]
            shp[a] = s
            pe = pe + self.pos[a].reshape(shp)
        return pe.reshape(self.n_tok, W)

    def features(self, x):
        if self.use_derivs:
            sp = [spatial_derivative(x, 3 + a) for a in range(self.dim)]
            x = torch.cat([x] + sp + [temporal_derivative(x)], dim=2)
        return x                                         # (B, 4, Cf, *grid)

    def _to_tokens(self, f):
        b, T, Cf = f.shape[:3]
        d, s, p = self.dim, self.side, self.patch
        f = f.reshape(b, T, Cf, *sum(([s, p] for _ in range(d)), []))
        # dims: 0 b, 1 T, 2 Cf, then (s_a, p_a) pairs at 3 + 2a, 4 + 2a
        perm = [0] + [3 + 2 * a for a in range(d)] + [1, 2] + [4 + 2 * a for a in range(d)]
        return f.permute(*perm).reshape(b, self.n_tok, -1)

    def _from_tokens(self, out):
        b = out.shape[0]
        d, s, p, c = self.dim, self.side, self.patch, self.n_ch
        out = out.reshape(b, *([s] * d), c, *([p] * d))
        # dims: 0 b, 1..d s_a, d+1 c, d+2.. p_a
        perm = [0, d + 1] + sum(([1 + a, d + 2 + a] for a in range(d)), [])
        return out.permute(*perm).reshape(b, c, *([self.n_x] * d))

    def rate(self, x):
        tok = self._to_tokens(self.features(x))
        h = self.core(self.embed(tok) + self.pos_embedding())
        return self._from_tokens(self.head(self.norm(h)))

    def forward(self, x):
        return x[:, -1] + self.rate(x)

    @torch.no_grad()
    def rollout(self, x0, n_steps):
        window, preds = x0, []
        for _ in range(n_steps):
            nxt = self(window)
            preds.append(nxt)
            window = torch.cat([window[:, 1:], nxt[:, None]], dim=1)
        return torch.stack(preds, dim=1)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
