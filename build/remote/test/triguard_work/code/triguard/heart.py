"""P5: Fed-Heart-Disease (4 hospitals, 13 features, binary). Logistic regression trained with DP-SGD.

Methods
  dpfedavg_lr : DP-FedAvg (sample-count weights) + local split conformal (U1)
  re_dpscp    : random-effects noise-aware aggregation (C2) + DP-SCP (C3) + per-patient ledger (C4)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch

from triguard import conformal as cp
from triguard import dp
from triguard.aggregation import Server
from triguard.evaluate import calib_budget
from triguard.metrics import macro_auc
from triguard.paths import DATA, RUNS
from triguard.utils import save_json

CENTERS = ["cleveland", "hungarian", "switzerland", "va"]
# public, clinically plausible ranges used for scaling (no data-dependent statistics -> no privacy cost)
RANGES = [(20, 80), (0, 1), (1, 4), (80, 200), (0, 600), (0, 1), (0, 2), (60, 210), (0, 1), (-2, 6), (1, 3), (0, 3), (3, 7)]


def load():
    frames = []
    for k, c in enumerate(CENTERS):
        df = pd.read_csv(DATA / "fed_heart" / f"processed.{c}.data", header=None, na_values="?")
        X = df.iloc[:, :13].to_numpy(float)
        for j, (lo, hi) in enumerate(RANGES):
            X[:, j] = (X[:, j] - lo) / (hi - lo) * 2 - 1
        miss = np.isnan(X).astype(float)
        X = np.nan_to_num(X, nan=0.0)
        y = (df.iloc[:, 13].to_numpy() > 0).astype(int)
        frames.append((np.hstack([X, miss]), y))
    return frames


def split_site(X, y, rng):
    idx = rng.permutation(len(y))
    n = len(y)
    a, b = int(0.6 * n), int(0.8 * n)
    return {"train": idx[:a], "calib": idx[a:b], "test": idx[b:]}


def run(jid: str, method: str, eps: float, seed: int, rounds: int = 50, lr: float = 0.05, clip: float = 1.0):
    rng = np.random.default_rng(seed)
    data = load()
    splits = [split_site(X, y, np.random.default_rng(100 + k)) for k, (X, y) in enumerate(data)]
    d = data[0][0].shape[1]
    g = torch.Generator().manual_seed(seed)
    sites = []
    for k, ((X, y), sp) in enumerate(zip(data, splits)):
        n = len(sp["train"])
        q = min(1.0, 32 / n)
        steps = rounds * max(1, math.ceil(1 / q))
        delta = dp.site_delta(n + len(sp["calib"]))
        sigma = dp.calibrate_sigma(eps, delta, q, steps, "rdp") if math.isfinite(eps) else 0.0
        sites.append(dict(X=torch.tensor(X, dtype=torch.float32), y=torch.tensor(y), sp=sp, n=n, q=q, sigma=sigma,
                          steps=steps, delta=delta, spr=max(1, math.ceil(1 / q))))
    w0 = {"w": torch.zeros(1, d), "b": torch.zeros(1)}
    server = Server(w0, "re" if method == "re_dpscp" else "fedavg", 1.0, 0.0)
    for t in range(rounds):
        ups, ns, vs = [], [], []
        for s in sites:
            w = {k: v.clone() for k, v in server.g.items()}
            nv = 0.0
            tr = s["sp"]["train"]
            for _ in range(s["spr"]):
                if s["sigma"] > 0:
                    idx = tr[rng.random(len(tr)) < s["q"]]
                else:
                    idx = rng.choice(tr, min(len(tr), 32), replace=False)
                X, y = s["X"][idx], s["y"][idx].float()
                p = torch.sigmoid(X @ w["w"].T + w["b"]).squeeze(1)
                gw = (p - y)[:, None] * X  # per-sample gradients
                gb = (p - y)[:, None]
                norms = torch.sqrt((gw**2).sum(1) + (gb**2).sum(1))
                f = (clip / (norms + 1e-6)).clamp(max=1.0)[:, None]
                if s["sigma"] > 0:
                    B = s["q"] * len(tr)
                    Gw = ((gw * f).sum(0) + torch.randn(d, generator=g) * s["sigma"] * clip) / B
                    Gb = ((gb * f).sum(0) + torch.randn(1, generator=g) * s["sigma"] * clip) / B
                    nv += (lr * s["sigma"] * clip / B) ** 2
                else:
                    Gw, Gb = gw.mean(0), gb.mean(0)
                w["w"] -= lr * Gw[None]
                w["b"] -= lr * Gb
            ups.append({k: w[k] - server.g[k] for k in w})
            ns.append(len(tr))
            vs.append(nv)
        server.step(ups, ns, vs)
    W = server.g
    # ---- evaluation
    def logits(s, part):
        X = s["X"][s["sp"][part]]
        z = (X @ W["w"].T + W["b"]).squeeze(1).numpy()
        return np.stack([np.zeros_like(z), z], 1), s["y"][s["sp"][part]].numpy()

    aucs = {}
    for k, s in enumerate(sites):
        lg, y = logits(s, "test")
        aucs[k] = macro_auc(y, lg)
    # ---- conformal
    alpha = 0.10
    cal, test = {}, {}
    for k, s in enumerate(sites):
        cal[k] = logits(s, "calib")
        test[k] = logits(s, "test")
    deltas = {k: s["delta"] for k, s in enumerate(sites)}
    covs = []
    for r in range(50):
        rr = np.random.default_rng(seed * 100 + r)
        cs = {k: (cp.aps_scores(cp.probs_from_logits(cal[k][0]), rr, cal[k][1]), cal[k][1]) for k in cal}
        ta = {k: cp.aps_scores(cp.probs_from_logits(test[k][0]), rr) for k in test}
        if method == "re_dpscp":
            b = calib_budget(eps, min(deltas.values()), min(len(c[1]) for c in cal.values()))
            m = cp.DPSCP(cp.DPSCPCfg(alpha=alpha, eps_label=b["label"], eps_temp=b["temp"], eps_cal=b["cal"], M=10,
                                     shift_bins=5), 2, deltas).fit(cs, rr)
            sets = {k: m.predict_sets(ta[k], k) for k in ta}
        else:
            q = cp.u1_local(cs, alpha)
            sets = {k: ta[k] <= q[k] for k in ta}
        covs.append({k: float(sets[k][np.arange(len(test[k][1])), test[k][1]].mean()) for k in sets})
    cov = {k: float(np.mean([c[k] for c in covs])) for k in covs[0]}
    res = {
        "method": method, "eps": eps, "seed": seed,
        "auc_per_site": aucs, "auc_mean": float(np.mean(list(aucs.values()))), "auc_worst": float(min(aucs.values())),
        "coverage_per_site": cov,
        "coverage": float(np.mean([np.mean(list(c.values())) for c in covs])),
        "worst_site_coverage": float(min(cov.values())),
        "ledger": [{"site": k, "n": s["n"], "q": s["q"], "sigma": s["sigma"], "steps": s["steps"], "delta": s["delta"],
                    "eps_train": dp.spent_epsilon([(s["sigma"], s["q"], s["steps"])], s["delta"], "rdp")
                    if s["sigma"] > 0 else math.inf} for k, s in enumerate(sites)],
        "server": server.history[-1],
    }
    save_json(res, RUNS / jid / "heart_result.json")
    print(res["auc_mean"], res["coverage"])
    return res
