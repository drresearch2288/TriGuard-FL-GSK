"""DP-NASA-LoRA tail model.

The last `n_tail` transformer blocks of a frozen ViT are re-implemented functionally so that
per-sample gradients can be taken with torch.func.vmap. Each attention projection (q, k, v, o)
gets up to two low-rank adapters:

    W' = W0 + (alpha/r) B_s A_s  +  (alpha/r) B_l A_l

  shared adapter  (s): A_s fixed random from a PUBLIC seed (FFA-style) unless train_shared_A,
                       B_s trained and federated.
  local adapter   (l): A_l, B_l trained, never leaves the site.
  head                : Linear on [cls, mean(patch)] of the final norm; local unless head_federated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

TARGETS = ("q", "k", "v", "o")


@dataclass
class ModelCfg:
    rank: int = 8
    alpha: float = 16.0
    shared: bool = True  # has a federated adapter
    train_shared_A: bool = False  # False -> FFA-style frozen A (C1)
    local: bool = True  # site-private adapter
    head_federated: bool = False
    lora: bool = True  # False -> linear probe (head only)
    public_seed: int = 12345


class TailModel(nn.Module):
    def __init__(self, blocks: nn.ModuleList, norm: nn.Module, n_frozen: int, num_classes: int, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        tail = list(blocks)[n_frozen:]
        self.n_tail = len(tail)
        self.dim = norm.weight.shape[0]
        self.heads = tail[0].attn.num_heads
        self.num_classes = num_classes
        D = self.dim
        # ---- frozen weights as buffers
        for i, b in enumerate(tail):
            W, bias = b.attn.qkv.weight.detach(), b.attn.qkv.bias
            bias = bias.detach() if bias is not None else torch.zeros(3 * D)
            for j, t in enumerate("qkv"):
                self.register_buffer(f"W{t}{i}", W[j * D : (j + 1) * D].clone())
                self.register_buffer(f"b{t}{i}", bias[j * D : (j + 1) * D].clone())
            self.register_buffer(f"Wo{i}", b.attn.proj.weight.detach().clone())
            self.register_buffer(f"bo{i}", b.attn.proj.bias.detach().clone())
            for n in ("norm1", "norm2"):
                ln = getattr(b, n)
                self.register_buffer(f"{n}w{i}", ln.weight.detach().clone())
                self.register_buffer(f"{n}b{i}", ln.bias.detach().clone())
            for n in ("ls1", "ls2"):
                g = getattr(getattr(b, n, None), "gamma", None)
                self.register_buffer(f"{n}{i}", g.detach().clone() if g is not None else torch.ones(D))
            self.register_buffer(f"fc1w{i}", b.mlp.fc1.weight.detach().clone())
            self.register_buffer(f"fc1b{i}", b.mlp.fc1.bias.detach().clone())
            self.register_buffer(f"fc2w{i}", b.mlp.fc2.weight.detach().clone())
            self.register_buffer(f"fc2b{i}", b.mlp.fc2.bias.detach().clone())
        self.ln_eps = getattr(tail[0].norm1, "eps", 1e-6)
        self.register_buffer("normw", norm.weight.detach().clone())
        self.register_buffer("normb", norm.bias.detach().clone())
        # ---- public, fixed shared A (generated from a public seed: no privacy cost)
        if cfg.lora and cfg.shared and not cfg.train_shared_A:
            g = torch.Generator().manual_seed(cfg.public_seed)
            for i in range(self.n_tail):
                for t in TARGETS:
                    A = (torch.rand(cfg.rank, D, generator=g) * 2 - 1) / math.sqrt(D)  # kaiming-uniform-like
                    self.register_buffer(f"sA{i}{t}", A)

    # ------------------------------------------------------------------ params
    def init_params(self, seed: int) -> dict[str, torch.Tensor]:
        cfg, D, r = self.cfg, self.dim, self.cfg.rank
        g = torch.Generator().manual_seed(seed)
        p: dict[str, torch.Tensor] = {}
        if cfg.lora:
            for i in range(self.n_tail):
                for t in TARGETS:
                    if cfg.shared:
                        p[f"s.B.{i}.{t}"] = torch.zeros(D, r)
                        if cfg.train_shared_A:
                            # same public init on every site so the federated A starts identical
                            gp = torch.Generator().manual_seed(cfg.public_seed + 7 * i + TARGETS.index(t))
                            p[f"s.A.{i}.{t}"] = (torch.rand(r, D, generator=gp) * 2 - 1) / math.sqrt(D)
                    if cfg.local:
                        p[f"l.A.{i}.{t}"] = (torch.rand(r, D, generator=g) * 2 - 1) / math.sqrt(D)
                        p[f"l.B.{i}.{t}"] = torch.zeros(D, r)
        # head: zero init so every site starts from the same function
        p["head.w"] = torch.zeros(self.num_classes, 2 * D)
        p["head.b"] = torch.zeros(self.num_classes)
        dev = self.normw.device
        return {k: v.to(dev) for k, v in p.items()}

    def is_federated(self, name: str) -> bool:
        if name.startswith("s."):
            return True
        if name.startswith("head."):
            return self.cfg.head_federated
        return False

    # ------------------------------------------------------------------ forward
    def _delta(self, p, i, t, h):
        cfg = self.cfg
        if not cfg.lora:
            return 0.0
        s = cfg.alpha / cfg.rank
        out = 0.0
        rec = self._rec
        for kind in ("s", "l"):
            if kind == "s" and not cfg.shared:
                continue
            if kind == "l" and not cfg.local:
                continue
            nA, nB = f"{kind}.A.{i}.{t}", f"{kind}.B.{i}.{t}"
            A = p[nA] if nA in p else getattr(self, f"sA{i}{t}")
            z = h @ A.T
            o = s * (z @ p[nB].T)
            if rec is not None:
                rec.append((nA if nA in p else None, nB, h, z, o))
            out = out + o
        return out

    _rec = None

    def features_rec(self, p: dict, x: torch.Tensor, rec: list) -> torch.Tensor:
        """Forward that records (A name, B name, input h, z = h A^T, adapter output) for fast per-sample grads."""
        self._rec = rec
        try:
            return self.features(p, x)
        finally:
            self._rec = None

    def features(self, p: dict, x: torch.Tensor) -> torch.Tensor:
        """x: [N, T, D] cached block-n_frozen tokens -> [N, 2D] features."""
        N, T, D = x.shape
        H = self.heads
        hd = D // H
        for i in range(self.n_tail):
            g = lambda n: getattr(self, f"{n}{i}")  # noqa: E731
            h = F.layer_norm(x, (D,), g("norm1w"), g("norm1b"), self.ln_eps)
            q = F.linear(h, g("Wq"), g("bq")) + self._delta(p, i, "q", h)
            k = F.linear(h, g("Wk"), g("bk")) + self._delta(p, i, "k", h)
            v = F.linear(h, g("Wv"), g("bv")) + self._delta(p, i, "v", h)
            q = q.view(N, T, H, hd).transpose(1, 2)
            k = k.view(N, T, H, hd).transpose(1, 2)
            v = v.view(N, T, H, hd).transpose(1, 2)
            if getattr(self, "use_sdpa", False):
                a = F.scaled_dot_product_attention(q, k, v)
            else:
                a = torch.softmax((q @ k.transpose(-2, -1)) * (hd**-0.5), dim=-1) @ v
            a = a.transpose(1, 2).reshape(N, T, D)
            o = F.linear(a, g("Wo"), g("bo")) + self._delta(p, i, "o", a)
            x = x + g("ls1") * o
            h2 = F.layer_norm(x, (D,), g("norm2w"), g("norm2b"), self.ln_eps)
            m = F.linear(F.gelu(F.linear(h2, g("fc1w"), g("fc1b"))), g("fc2w"), g("fc2b"))
            x = x + g("ls2") * m
        x = F.layer_norm(x, (D,), self.normw, self.normb, self.ln_eps)
        return torch.cat([x[:, 0], x[:, 1:].mean(1)], dim=-1)

    def logits(self, p: dict, x: torch.Tensor) -> torch.Tensor:
        return F.linear(self.features(p, x), p["head.w"], p["head.b"])


class FeatureHead(nn.Module):
    """Linear model on precomputed features (B4 linear probe on full-backbone features, tabular data)."""

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.in_dim, self.num_classes = in_dim, num_classes

    def init_params(self, seed: int):
        return {"head.w": torch.zeros(self.num_classes, self.in_dim), "head.b": torch.zeros(self.num_classes)}

    def is_federated(self, name: str) -> bool:
        return True

    def features(self, p, x):
        return x

    def features_rec(self, p, x, rec):
        return x

    def logits(self, p, x):
        return F.linear(x, p["head.w"], p["head.b"])
