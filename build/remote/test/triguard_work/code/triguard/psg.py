"""Fast exact per-sample gradients for LoRA adapters + linear head.

One ordinary batched forward/backward (flash attention allowed) gives dL/d(adapter output) for every
token. Because samples never interact (LayerNorm only, no BatchNorm), those gradients are already
per-sample, and the per-sample parameter gradients follow in closed form:

    out = s * z B^T,  z = h A^T
    dL_i/dB = s * sum_t gout_it^T z_it            dL_i/dA = sum_t (s * gout_it B)^T h_it
    head:  dL_i/dW = sum_k (softmax - onehot)_ik^T feat_ik / K

Loss of sample i = mean over its K augmented views (augmentation multiplicity).
"""
from __future__ import annotations

import functools

import torch
import torch.nn.functional as F

LOSS_SCALE = 1024.0


@functools.lru_cache(1)
def amp_dtype():
    """bf16 where the GPU supports it natively (Ampere+, e.g. the RTX 3050); fp16 on T4 / P100 (Kaggle, Colab)."""
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


def per_sample_grads(model, p: dict, names: list[str], X: torch.Tensor, Y: torch.Tensor, amp: bool = True):
    P, K = X.shape[:2]
    x = X.flatten(0, 1)
    rec: list = []
    names = set(names)
    p = {k: (v.detach().requires_grad_(True) if k in names else v.detach()) for k, v in p.items()}
    dtype = amp_dtype()
    scale = LOSS_SCALE if dtype == torch.float16 else 1.0  # fp16 (T4/P100): avoid underflow of token gradients
    with torch.autocast("cuda", dtype=dtype, enabled=amp and x.is_cuda):
        feat = model.features_rec(p, x, rec)
    feat = feat.float()
    logits = F.linear(feat, p["head.w"], p["head.b"])
    yy = Y.repeat_interleave(K)
    out: dict[str, torch.Tensor] = {}
    # ---- head (closed form)
    with torch.no_grad():
        dlog = (torch.softmax(logits, -1) - F.one_hot(yy, logits.shape[-1]).float()) / K  # [N, C]
    if "head.w" in names:
        out["head.w"] = torch.einsum("pkc,pkf->pcf", dlog.view(P, K, -1), feat.detach().view(P, K, -1))
    if "head.b" in names:
        out["head.b"] = dlog.view(P, K, -1).sum(1)
    # ---- adapters
    rec = [r for r in rec if r[1] in names or (r[0] is not None and r[0] in names)]
    if rec:
        loss = F.cross_entropy(logits, yy, reduction="sum") / K
        gouts = torch.autograd.grad(loss * scale, [r[4] for r in rec])
        s = model.cfg.alpha / model.cfg.rank
        for (nA, nB, h, z, _), go in zip(rec, gouts):
            T = go.shape[1]
            go = go.float().reshape(P, K * T, -1) / scale
            if nB in names:
                out[nB] = s * torch.einsum("pmd,pmr->pdr", go, z.float().reshape(P, K * T, -1))
            if nA is not None and nA in names:
                gz = s * (go @ p[nB].float())  # [P, KT, r]
                out[nA] = torch.einsum("pmr,pmd->prd", gz, h.float().reshape(P, K * T, -1))
    return {k: v.detach() for k, v in out.items()}
