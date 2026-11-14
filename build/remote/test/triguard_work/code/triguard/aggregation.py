"""Server-side aggregation (Stage 3). Every input is already DP -> pure post-processing."""
from __future__ import annotations

import numpy as np
import torch


def flatten(d: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([d[k].reshape(-1) for k in sorted(d)])


def fedavg_weights(n: list[int]) -> np.ndarray:
    n = np.asarray(n, float)
    return n / n.sum()


def dersimonian_laird(deltas: list[torch.Tensor], v: np.ndarray) -> tuple[float, np.ndarray, dict]:
    """Random-effects weights.

    deltas: flattened per-site updates (d coordinates each)
    v:      per-coordinate DP-noise variance of each site's update (public / post-processing)
    Returns tau^2, weights, diagnostics.
    """
    K = len(deltas)
    v = np.asarray(v, float)
    if K < 2 or np.any(v <= 0):
        return 0.0, None, {"note": "non-private or single site"}
    X = torch.stack(deltas).double()  # [K, d]
    d = X.shape[1]
    wfe = torch.tensor(1.0 / v, dtype=torch.float64, device=X.device)
    mean_fe = (wfe[:, None] * X).sum(0) / wfe.sum()
    # Cochran's Q averaged over coordinates
    Q = float((wfe * ((X - mean_fe) ** 2).sum(1)).sum() / d)
    S1, S2 = float(wfe.sum()), float((wfe**2).sum())
    tau2 = max(0.0, (Q - (K - 1)) / (S1 - S2 / S1))
    w = 1.0 / (v + tau2)
    w = w / w.sum()
    return tau2, w, {"Q": Q, "S1": S1, "S2": S2}


class Server:
    def __init__(self, global_params: dict[str, torch.Tensor], rule: str, lr: float = 1.0, momentum: float = 0.9):
        self.g = {k: v.clone() for k, v in global_params.items()}
        self.rule, self.lr, self.mom = rule, lr, momentum
        self.m = {k: torch.zeros_like(v) for k, v in self.g.items()}
        self.history: list[dict] = []

    def step(self, updates: list[dict[str, torch.Tensor]], n: list[int], v: list[float]) -> dict:
        info: dict = {}
        if self.rule == "re":
            tau2, w, diag = dersimonian_laird([flatten(u) for u in updates], np.asarray(v))
            if w is None:
                w = fedavg_weights(n)
            info.update(tau2=tau2, **diag)
        elif self.rule == "fedavg":
            w = fedavg_weights(n)
        elif self.rule == "equal":
            w = np.ones(len(n)) / len(n)
        else:
            raise ValueError(self.rule)
        info["weights"] = [float(x) for x in w]
        info["v"] = [float(x) for x in v]
        for k in self.g:
            agg = sum(float(wk) * u[k] for wk, u in zip(w, updates))
            self.m[k] = self.mom * self.m[k] + agg
            self.g[k] = self.g[k] + self.lr * self.m[k]
        self.history.append(info)
        return info
