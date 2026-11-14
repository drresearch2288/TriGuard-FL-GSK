"""DP-SGD primitives: accounting, Poisson sampling, per-sample clipping and Gaussian noise."""
from __future__ import annotations

import math

import numpy as np
import torch
from opacus.accountants import PRVAccountant, RDPAccountant
from opacus.accountants.utils import get_noise_multiplier


WIDE_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)) + [80, 128, 256, 512, 1024, 2048, 4096]


def site_delta(n: int) -> float:
    """delta_k = 1 / n_k^1.1 (plan, Stage 2)."""
    return 1.0 / (n**1.1)


def sample_rate(n: int, logical_batch: int, max_rate: float = 0.25) -> float:
    """q_k = B / n_k, capped so that tiny sites still sample (B_k = q_k n_k)."""
    return min(logical_batch / n, max_rate)


def calibrate_sigma(eps: float, delta: float, q: float, steps: int, accountant: str = "prv") -> float:
    if not math.isfinite(eps):
        return 0.0
    kw = dict(target_epsilon=eps, target_delta=delta, sample_rate=q, steps=steps, accountant=accountant,
              epsilon_tolerance=0.005)
    try:
        return float(get_noise_multiplier(**kw))
    except ValueError:
        if accountant != "rdp":
            raise
        # Opacus' default RDP orders stop at 63, which floors epsilon at ~log(1/delta)/62: small sub-budgets
        # (e.g. temperature scaling at total eps = 1) are unreachable. Higher orders give a valid, tighter bound.
        return float(get_noise_multiplier(**kw, alphas=WIDE_ALPHAS))


def spent_epsilon(history: list[tuple[float, float, int]], delta: float, accountant: str = "prv") -> float:
    """history = [(sigma, q, steps), ...] composed."""
    hist = [h for h in history if h[0] > 0 and h[2] > 0]
    if any(h[0] == 0 for h in history if h[2] > 0):
        return float("inf")
    if not hist:
        return 0.0
    acc = PRVAccountant() if accountant == "prv" else RDPAccountant()
    acc.history = list(hist)
    return float(acc.get_epsilon(delta=delta))


def rdp_compose_epsilon(history: list[tuple[float, float, int]], delta: float) -> float:
    acc = RDPAccountant()
    acc.history = [h for h in history if h[2] > 0]
    if any(h[0] == 0 for h in acc.history):
        return float("inf")
    return float(acc.get_epsilon(delta=delta))


def gaussian_sigma_for_eps(eps: float, delta: float) -> float:
    """Noise multiplier of a single full-batch Gaussian release (sensitivity 1) reaching (eps, delta)."""
    return calibrate_sigma(eps, delta, q=1.0, steps=1, accountant="rdp")


def poisson_batches(n: int, q: float, steps: int, rng: np.random.Generator):
    """Yields index arrays, each element included independently with prob q."""
    for _ in range(steps):
        yield np.nonzero(rng.random(n) < q)[0]


def clip_and_accumulate(per_grads: dict[str, torch.Tensor], C: float, acc: dict[str, torch.Tensor]) -> torch.Tensor:
    """Clips each sample's *joint* gradient (over all trainable tensors) to norm C and adds to acc.

    Returns per-sample norms (for logging)."""
    names = list(per_grads)
    b = per_grads[names[0]].shape[0]
    sq = torch.zeros(b, device=per_grads[names[0]].device)
    for n in names:
        sq += per_grads[n].reshape(b, -1).pow(2).sum(1)
    norms = sq.sqrt()
    factor = (C / (norms + 1e-6)).clamp(max=1.0)
    for n in names:
        g = (per_grads[n] * factor.view(-1, *([1] * (per_grads[n].dim() - 1)))).sum(0)
        acc[n] = acc[n] + g if n in acc else g
    return norms


def add_noise(acc: dict[str, torch.Tensor], sigma: float, C: float, expected_batch: float, gen: torch.Generator | None = None):
    out = {}
    for n, g in acc.items():
        noise = torch.randn(g.shape, device=g.device, generator=gen) * (sigma * C) if sigma > 0 else 0.0
        out[n] = (g + noise) / expected_batch
    return out
