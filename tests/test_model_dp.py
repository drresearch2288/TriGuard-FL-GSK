"""M2/M3 acceptance checks: cached path == full model, per-sample grads correct, clipping, VRAM."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from triguard import dp
from triguard.backbone import load_backbone
from triguard.model import ModelCfg, TailModel

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def bb():
    return load_backbone("dinov2_vits14", DEV)


def test_cached_path_matches_full(bb):
    torch.manual_seed(0)
    x = torch.randn(4, 3, 224, 224, device=DEV)
    full = bb.forward_full_features(x)
    tok = bb.forward_frozen(x, 8)
    tail = TailModel(bb.blocks, bb.norm, 8, 8, ModelCfg()).to(DEV)
    p = tail.init_params(0)  # B = 0 -> adapters are exact no-ops
    with torch.no_grad():
        mine = tail.features(p, tok)
    assert torch.allclose(full, mine, atol=1e-3), (full - mine).abs().max()
    # fp16 cache round-trip stays close in logit space
    head = torch.randn(8, full.shape[1], device=DEV) * 0.05
    with torch.no_grad():
        l_full = full @ head.T
        l_cache = tail.features(p, tok.half().float()) @ head.T
    assert (l_full - l_cache).abs().max() < 1e-2


def test_per_sample_grads_match_loop(bb):
    from torch.func import grad, vmap

    tail = TailModel(bb.blocks, bb.norm, 8, 8, ModelCfg()).to(DEV)
    p = tail.init_params(0)
    for k in p:
        p[k] = p[k] + 0.01 * torch.randn_like(p[k])
    X = torch.randn(3, 2, 257, 384, device=DEV)  # 3 samples x 2 views
    Y = torch.tensor([1, 4, 7], device=DEV)

    def loss1(tp, x, y):
        return F.cross_entropy(tail.logits(tp, x), y.expand(x.shape[0]))

    per = vmap(grad(loss1), in_dims=(None, 0, 0))(p, X, Y)
    for i in range(3):
        tp = {k: v.detach().clone().requires_grad_(True) for k, v in p.items()}
        gs = torch.autograd.grad(loss1(tp, X[i], Y[i]), list(tp.values()))
        for (k, _), g in zip(tp.items(), gs):
            assert torch.allclose(per[k][i], g, atol=1e-5, rtol=1e-3), k


@pytest.mark.parametrize("cfg", [ModelCfg(), ModelCfg(train_shared_A=True, local=False), ModelCfg(shared=False)])
def test_fast_psg_matches_vmap(bb, cfg):
    from torch.func import grad, vmap

    from triguard.psg import per_sample_grads

    tail = TailModel(bb.blocks, bb.norm, 8, 8, cfg).to(DEV)
    tail.use_sdpa = True
    p = tail.init_params(0)
    for k in p:
        p[k] = p[k] + 0.02 * torch.randn_like(p[k])
    X = torch.randn(3, 2, 257, 384, device=DEV)
    Y = torch.tensor([1, 4, 7], device=DEV)
    names = list(p)

    def loss1(tp, x, y):
        return F.cross_entropy(tail.logits(tp, x), y.expand(x.shape[0]))

    tail.use_sdpa = False
    ref = vmap(grad(loss1), in_dims=(None, 0, 0))(p, X, Y)
    tail.use_sdpa = True
    for amp, tol in [(False, 1e-4), (True, 1e-1)]:  # bf16 has 8 mantissa bits
        fast = per_sample_grads(tail, p, names, X, Y, amp=amp)
        assert set(fast) == set(names)
        for k in names:
            rel = (fast[k] - ref[k]).norm() / (ref[k].norm() + 1e-12)
            assert rel < tol, (k, amp, float(rel))


def test_fp16_loss_scaled_psg(bb, monkeypatch):
    """The T4/P100 path (fp16 autocast + loss scaling) must match the exact fp32 gradients."""
    from triguard import psg

    tail = TailModel(bb.blocks, bb.norm, 8, 8, ModelCfg()).to(DEV)
    tail.use_sdpa = True
    p = tail.init_params(0)
    for k in p:
        p[k] = p[k] + 0.02 * torch.randn_like(p[k])
    X = torch.randn(3, 2, 257, 384, device=DEV)
    Y = torch.tensor([1, 4, 7], device=DEV)
    ref = psg.per_sample_grads(tail, p, list(p), X, Y, amp=False)
    monkeypatch.setattr(psg, "amp_dtype", lambda: torch.float16)
    fast = psg.per_sample_grads(tail, p, list(p), X, Y, amp=True)
    for k in p:
        rel = (fast[k] - ref[k]).norm() / (ref[k].norm() + 1e-12)
        assert torch.isfinite(fast[k]).all() and rel < 0.05, (k, float(rel))


def test_clip_and_noise():
    g = {"a": torch.tensor([[3.0, 4.0], [0.3, 0.4]])}
    acc = {}
    norms = dp.clip_and_accumulate(g, 1.0, acc)
    assert torch.allclose(norms, torch.tensor([5.0, 0.5]))
    assert torch.allclose(acc["a"], torch.tensor([0.6 + 0.3, 0.8 + 0.4]), atol=1e-5)
    out = dp.add_noise(acc, 0.0, 1.0, 2.0)
    assert torch.allclose(out["a"], acc["a"] / 2)


def test_accountant_roundtrip():
    s = dp.calibrate_sigma(3.0, 1e-5, 0.1, 500)
    e = dp.spent_epsilon([(s, 0.1, 500)], 1e-5)
    assert abs(e - 3.0) / 3.0 < 0.01
