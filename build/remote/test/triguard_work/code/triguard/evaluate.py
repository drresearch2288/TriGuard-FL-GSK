"""CPU stage after training: metrics, DP temperature scaling, DP-SCP + U1-U4, fairness, privacy ledger.

Usage: python -m triguard.evaluate runs/<run_id> [--reps 20]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from triguard import conformal as cp
from triguard import dp
from triguard.cache import Cache
from triguard.metrics import balanced_accuracy, brier, ece, macro_auc, macro_f1, reliability
from triguard.utils import load_json, save_json

ALPHAS = [0.05, 0.10, 0.15, 0.20]
BASE_SPLIT = {"label": 0.3, "temp": 0.2, "cal": 2.5}  # plan default for eps = 3


def calib_budget(eps_total: float, delta: float, n_cal: int) -> dict:
    """Scale the (0.3, 0.2, 2.5) allocation so the RDP-composed calibration epsilon equals eps_total."""
    if not math.isfinite(eps_total):
        return {"label": math.inf, "temp": math.inf, "cal": math.inf, "composed": math.inf}

    def composed(s):
        e = {k: v / 3.0 * eps_total * s for k, v in BASE_SPLIT.items()}
        hist = [(dp.gaussian_sigma_for_eps(e["label"], delta), 1.0, 1), (dp.gaussian_sigma_for_eps(e["cal"], delta), 1.0, 1)]
        q = min(1.0, 256 / max(1, n_cal))
        hist.append((dp.calibrate_sigma(e["temp"], delta, q, 200, "rdp"), q, 200))
        return dp.rdp_compose_epsilon(hist, delta), e

    lo, hi = 0.3, 3.0
    for _ in range(25):
        mid = (lo + hi) / 2
        c, _ = composed(mid)
        lo, hi = (mid, hi) if c < eps_total else (lo, mid)
    c, e = composed(lo)
    e["composed"] = c
    return e


def age_band(a):
    if pd.isna(a):
        return "unknown"
    return "<40" if a < 40 else ("40-60" if a <= 60 else ">60")


def main(run_dir: str, reps: int = 20, seed: int = 0) -> dict:
    run = Path(run_dir)
    tr = load_json(run / "train_result.json")
    cfg = tr["cfg"]
    cache = Cache(cfg["dataset"], cfg["backbone"])
    df = cache.df.set_index("cache_row")
    P = np.load(run / "preds.npz")
    sites = sorted({int(k.split("_")[1]) for k in P.files if k.startswith("test_")})
    C = int(cache.df.y.max()) + 1
    eps = float(cfg["eps"])
    rng = np.random.default_rng(seed)
    out: dict = {"run": str(run), "eps": eps, "method": cfg["method"]}

    # ---------------- prediction metrics
    pred = {}
    all_y, all_lg, all_site = [], [], []
    for k in sites:
        lg, rows = P[f"test_{k}_logits"], P[f"test_{k}_rows"]
        y = df.loc[rows, "y"].values
        site_of = df.loc[rows, "site"].values
        pred[k] = {"bacc": balanced_accuracy(y, lg.argmax(1)), "auc": macro_auc(y, lg), "f1": macro_f1(y, lg.argmax(1)),
                   "n": int(len(y))}
        all_y.append(y)
        all_lg.append(lg)
        all_site.append(site_of)
    Y, LG, S = np.concatenate(all_y), np.concatenate(all_lg), np.concatenate(all_site)
    per_site = {int(s): balanced_accuracy(Y[S == s], LG[S == s].argmax(1)) for s in np.unique(S)}
    out["prediction"] = {
        "per_site": per_site,
        "bacc_pooled": balanced_accuracy(Y, LG.argmax(1)),
        "bacc_mean_sites": float(np.mean(list(per_site.values()))),
        "bacc_worst_site": float(np.min(list(per_site.values()))),
        "macro_auc": macro_auc(Y, LG),
        "macro_f1": macro_f1(Y, LG.argmax(1)),
    }

    # val split (e.g. Camelyon hospital 4)
    vkeys = [f for f in P.files if f.startswith("val_") and f.endswith("_logits")]
    if vkeys:
        out["prediction"]["val"] = {int(f.split("_")[1]): balanced_accuracy(df.loc[P[f.replace("logits", "rows")], "y"].values,
                                                                            P[f].argmax(1)) for f in vkeys}

    # ---------------- per-site calibration data
    ledger_train = {int(r["site"]): r for r in tr["ledger"]}
    cal_sites = [k for k in sites if f"calib_{k}_logits" in P.files]
    if not cal_sites:
        save_json(out, run / "eval_result.json")
        return out
    cal = {}
    for k in cal_sites:
        lg, rows = P[f"calib_{k}_logits"], P[f"calib_{k}_rows"]
        cal[k] = (lg, df.loc[rows, "y"].values)

    def train_entry(k):
        return ledger_train.get(k, ledger_train.get(-1, {"n": 0}))

    deltas = {k: dp.site_delta(len(cal[k][1]) + train_entry(k)["n"]) for k in cal_sites}
    budgets = {k: calib_budget(eps, deltas[k], len(cal[k][1])) for k in cal_sites}

    # ---------------- DP temperature scaling (ECE before/after)
    temps, temp_hist = {}, {}
    for k in cal_sites:
        T, h = cp.dp_temperature(cal[k][0], cal[k][1], budgets[k]["temp"], deltas[k], rng)
        temps[k], temp_hist[k] = T, h
    for k in sites:  # unseen hospital / device: mean of the released (DP) temperatures
        temps.setdefault(k, float(np.mean([temps[c] for c in cal_sites])))
    p_raw = cp.probs_from_logits(LG)
    p_cal = np.concatenate([cp.probs_from_logits(P[f"test_{k}_logits"], temps[k]) for k in sites])
    out["calibration"] = {
        "temperature": temps,
        "ece_before": ece(Y, p_raw), "ece_after": ece(Y, p_cal),
        "brier_before": brier(Y, p_raw), "brier_after": brier(Y, p_cal),
        "reliability_before": reliability(Y, p_raw), "reliability_after": reliability(Y, p_cal),
    }

    # ---------------- conformal: DP-SCP and baselines, repeated over DP noise / APS randomisation
    test = {k: (cp.probs_from_logits(P[f"test_{k}_logits"], temps[k]), df.loc[P[f"test_{k}_rows"], "y"].values,
                P[f"test_{k}_rows"]) for k in sites}
    calp = {k: cp.probs_from_logits(cal[k][0], temps[k]) for k in cal_sites}
    methods = ["dpscp", "dpscp_noshift", "u1_local", "u2_fcp", "u3_naive_dp", "u4_labelshift"]
    res = {m: {a: [] for a in ALPHAS} for m in methods}
    for r in range(reps):
        rr = np.random.default_rng(seed * 1000 + r)
        cal_s = {k: (cp.aps_scores(calp[k], rr, cal[k][1]), cal[k][1]) for k in cal_sites}
        test_all = {k: cp.aps_scores(test[k][0], rr) for k in sites}
        for a in ALPHAS:
            def collect(name, sets_by_site):
                Ys = np.concatenate([test[k][1] for k in sites])
                sets = np.concatenate([sets_by_site[k] for k in sites])
                st = cp.coverage_stats(sets, Ys, C)
                st["per_site"] = {int(k): cp.coverage_stats(sets_by_site[k], test[k][1], C)["coverage"] for k in sites}
                st["set_size_site"] = {int(k): float(sets_by_site[k].sum(1).mean()) for k in sites}
                st["sets_store"] = None
                res[name][a].append((st, sets))

            for name, kw in [("dpscp", {}), ("dpscp_noshift", {"shift_check": False}),
                             ("u3_naive_dp", {"mondrian": False, "slack": False, "shift_check": False})]:
                c = cp.DPSCPCfg(alpha=a, eps_label=budgets[cal_sites[0]]["label"], eps_temp=budgets[cal_sites[0]]["temp"],
                                eps_cal=budgets[cal_sites[0]]["cal"], **kw)
                model = cp.DPSCP(c, C, deltas).fit(cal_s, rr)
                collect(name, {k: model.predict_sets(test_all[k], k) for k in sites})
                if name == "dpscp" and a == 0.10 and r == 0:
                    out["dpscp_info"] = {"gamma": model.gamma, "qhat": model.qhat, "shift_flags": model.shift_flags,
                                         "sigma_h": model.sigma_h, "sigma_c": model.sigma_c,
                                         "eps_cal_composed": model.ledger(temp_hist)}
            q1 = cp.u1_local(cal_s, a)
            q2 = cp.u2_fcp(cal_s, a)
            q4 = cp.u4_label_shift(cal_s, a, C)
            pooled = q2[cal_sites[0]]  # unseen sites have no local calibration data -> pooled quantile
            collect("u1_local", {k: test_all[k] <= q1.get(k, pooled) for k in sites})
            collect("u2_fcp", {k: test_all[k] <= pooled for k in sites})
            collect("u4_labelshift", {k: test_all[k] <= q4.get(k, pooled) for k in sites})

    conf = {}
    keep_sets = {}
    for m in methods:
        conf[m] = {}
        for a in ALPHAS:
            sts = [s for s, _ in res[m][a]]
            agg = {
                "coverage": float(np.mean([s["coverage"] for s in sts])),
                "coverage_sd": float(np.std([s["coverage"] for s in sts])),
                "set_size": float(np.mean([s["set_size"] for s in sts])),
                "singleton": float(np.mean([s["singleton"] for s in sts])),
                "worst_site_cov": float(np.mean([min(s["per_site"].values()) for s in sts])),
                "per_site_cov": {int(k): float(np.mean([s["per_site"][k] for s in sts])) for k in sites},
                "per_class_cov": {int(c): float(np.mean([s["per_class"].get(c, np.nan) for s in sts])) for c in range(C)},
                "worst_class_cov": float(np.mean([min(s["per_class"].values()) for s in sts])),
                "prob_cov_ge_target": float(np.mean([s["coverage"] >= 1 - a for s in sts])),
                "set_size_site": {int(k): float(np.mean([s["set_size_site"][k] for s in sts])) for k in sites},
            }
            conf[m][str(a)] = agg
            if a == 0.10:
                keep_sets[m] = res[m][a][0][1]
    out["conformal"] = conf
    np.savez_compressed(run / "sets_alpha010.npz", **keep_sets,
                        y=np.concatenate([test[k][1] for k in sites]),
                        rows=np.concatenate([test[k][2] for k in sites]))

    # ---------------- fairness (sex, age band) for BA and DP-SCP coverage at alpha 0.1
    rows_all = np.concatenate([test[k][2] for k in sites])
    meta = df.loc[rows_all]
    fair = {}
    sets01 = keep_sets["dpscp"]
    covered = sets01[np.arange(len(Y)), Y]
    for col, fn in [("sex", lambda v: v if isinstance(v, str) else "unknown"), ("age", age_band)]:
        if col not in meta.columns:
            continue
        g = meta[col].map(fn).values
        grp = {}
        for v in np.unique(g):
            if v == "unknown":
                continue
            m = g == v
            if m.sum() < 20:
                continue
            grp[str(v)] = {"bacc": balanced_accuracy(Y[m], LG[m].argmax(1)), "coverage": float(covered[m].mean()),
                           "n": int(m.sum())}
        if grp:
            fair[col] = {"groups": grp,
                         "bacc_gap": max(x["bacc"] for x in grp.values()) - min(x["bacc"] for x in grp.values()),
                         "coverage_gap": max(x["coverage"] for x in grp.values()) - min(x["coverage"] for x in grp.values())}
    out["fairness"] = fair

    # ---------------- per-patient privacy ledger (parallel composition: max, not sum)
    led = []
    eps_cal_comp = out.get("dpscp_info", {}).get("eps_cal_composed", {})
    for k in cal_sites:
        t = train_entry(k)
        e_tr = t.get("eps_train", math.inf)
        e_cal = eps_cal_comp.get(k, math.inf)
        led.append({"site": k, **{f"train_{a}": b for a, b in t.items()}, "eps_label": budgets[k]["label"],
                    "eps_temp": budgets[k]["temp"], "eps_cal": budgets[k]["cal"], "eps_calib_composed": e_cal,
                    "eps_per_patient": max(e_tr, e_cal), "delta": deltas[k]})
    out["ledger"] = led
    save_json(out, run / "eval_result.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    main(a.run, a.reps)
