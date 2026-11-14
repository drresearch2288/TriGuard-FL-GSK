"""P4: transfer of ISIC-trained models to PAD-UFES-20 smartphone images (external device, no training)."""
from __future__ import annotations

import math

import numpy as np
import torch

from triguard import conformal as cp
from triguard.backbone import load_backbone
from triguard.cache import Cache
from triguard.evaluate import calib_budget
from triguard import dp
from triguard.metrics import balanced_accuracy
from triguard.model import ModelCfg, TailModel
from triguard.paths import RUNS
from triguard.utils import device, load_json, save_json

PAD_CLASSES = [0, 1, 2, 3, 4, 7]  # MEL, NV, BCC, AK, BKL, SCC (public label mapping)


def pad_transfer(jid: str, source: str, recal_n: int = 200, alpha: float = 0.10, reps: int = 20):
    dev = device()
    src = RUNS / source
    tr = load_json(src / "train_result.json")
    cfg = tr["cfg"]
    eps = float(cfg["eps"])
    params = torch.load(src / "params.pt")
    bb = load_backbone(cfg["backbone"], "cpu")
    tail = TailModel(bb.blocks, bb.norm, cfg["n_frozen"], 8, ModelCfg(**cfg["model"])).to(dev)
    pad = Cache("pad_ufes20", cfg["backbone"])
    df = pad.df
    x_rows = df.cache_row.values
    with torch.no_grad():
        X = pad.get_clean(x_rows).float()
        probs = []
        for k, p in params.items():  # unseen device: average of the site models
            p = {n: v.to(dev) for n, v in p.items()}
            lg = torch.cat([tail.logits(p, X[i : i + 128].to(dev)).float().cpu() for i in range(0, len(X), 128)])
            probs.append(torch.softmax(lg, 1).numpy())
    P = np.mean(probs, 0)
    mask = np.zeros(8, bool)
    mask[PAD_CLASSES] = True
    P = P * mask
    P = P / P.sum(1, keepdims=True)
    y = df.y.values
    test = (df.split == "test").values
    calm = (df.split == "calib").values
    out = {"source": source, "eps": eps, "bacc": balanced_accuracy(y[test], P[test].argmax(1)), "n_test": int(test.sum())}
    # ---- thresholds calibrated on ISIC (DP-SCP pooled), applied as-is
    ev = load_json(src / "eval_result.json")
    T = float(np.mean(list(ev["calibration"]["temperature"].values())))
    Pt = np.exp(np.log(np.clip(P, 1e-12, None)) / T)
    Pt = Pt / Pt.sum(1, keepdims=True)
    S = np.load(src / "preds.npz")
    cdf = Cache("isic2019", cfg["backbone"]).df.set_index("cache_row")
    cal_sites = sorted({int(f.split("_")[1]) for f in S.files if f.startswith("calib_") and f.endswith("_logits")})
    n_train = {int(r["site"]): r["n"] for r in tr["ledger"]}
    deltas = {k: dp.site_delta(len(S[f"calib_{k}_rows"]) + n_train.get(k, 0)) for k in cal_sites}
    b = calib_budget(eps, min(deltas.values()), 200)
    res_isic, res_recal = [], []
    for r in range(reps):
        rr = np.random.default_rng(r)
        cal_s = {}
        for k in cal_sites:
            temp = ev["calibration"]["temperature"][str(k)]
            pc = cp.probs_from_logits(S[f"calib_{k}_logits"], temp)
            yc = cdf.loc[S[f"calib_{k}_rows"], "y"].values
            cal_s[k] = (cp.aps_scores(pc, rr, yc), yc)
        m = cp.DPSCP(cp.DPSCPCfg(alpha=alpha, eps_label=b["label"], eps_temp=b["temp"], eps_cal=b["cal"], shift_check=False),
                     8, deltas).fit(cal_s, rr)
        sc = cp.aps_scores(Pt[test], rr)
        sets = m.predict_sets(sc, -1) & mask[None]
        res_isic.append(cp.coverage_stats(sets, y[test], 8))
        # ---- optional private recalibration on ~200 PAD images (marginal DP histogram + slack)
        d_new = dp.site_delta(int(calm.sum()))
        b_new = calib_budget(eps, d_new, int(calm.sum()))
        m2 = cp.DPSCP(cp.DPSCPCfg(alpha=alpha, eps_label=b_new["label"], eps_temp=b_new["temp"], eps_cal=b_new["cal"],
                                  mondrian=False, shift_check=False, M=20), 8, {0: d_new})
        m2.fit({0: (cp.aps_scores(Pt[calm], rr, y[calm]), y[calm])}, rr)
        sets2 = m2.predict_sets(sc, 0) & mask[None]
        res_recal.append(cp.coverage_stats(sets2, y[test], 8))

    def agg(lst):
        return {"coverage": float(np.mean([s["coverage"] for s in lst])),
                "set_size": float(np.mean([s["set_size"] for s in lst])),
                "singleton": float(np.mean([s["singleton"] for s in lst]))}

    out["isic_thresholds"] = agg(res_isic)
    out["private_recalibration"] = agg(res_recal)
    out["recal_n"] = int(calm.sum())
    save_json(out, RUNS / jid / "transfer_result.json")
    print(out)
