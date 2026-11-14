"""Collect every finished result into tables (Markdown + CSV), statistics and figures.

    python -m triguard.report          -> results/RESULTS.md, results/tables/*.csv, results/figures/*.png
Safe to run at any time: missing results show as '—' and n (seeds) is printed next to every mean.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from triguard.paths import ROOT, RUNS  # noqa: E402

OUT = ROOT / "results"
TAB, FIG = OUT / "tables", OUT / "figures"
EPS_ORDER = ["1", "3", "8", "inf"]
NAMES = {
    "b1_local": "B1 Local only", "b2_central": "B2 Centralised pooled", "b3_resnet": "B3 DP-FedAvg full FT (ResNet-18)",
    "b4_probe": "B4 DP linear probe", "b5_fedlora": "B5 FedLoRA + DP-SGD", "b6_ffalora": "B6 FFA-LoRA + DP",
    "b7_personal": "B7 Personal LoRA + DP", "triguard": "TriGuard-FL (proposed)",
    "a1_fedavg": "A1 − RE aggregation (FedAvg weights)", "a2_nolocal": "A2 − private local adapter",
    "a3_trainA": "A3 − frozen A (train A and B)", "a4_k1": "A4 − augmented cache (K = 1)",
    "a5_imagenet": "A5 − foundation backbone (ImageNet ViT-S)", "a6_biomedclip": "A6 BiomedCLIP ViT-B instead of DINOv2",
}
# fixed categorical slots (validated palette, light mode); colour follows the method, never its rank
COLORS = {"triguard": "#2a78d6", "b6_ffalora": "#eb6834", "b5_fedlora": "#1baf7a", "b7_personal": "#eda100",
          "b4_probe": "#e87ba4", "b1_local": "#008300", "b2_central": "#6250d6", "b3_resnet": "#e34948"}
MARKERS = {"triguard": "o", "b6_ffalora": "s", "b5_fedlora": "^", "b7_personal": "D", "b4_probe": "v",
           "b1_local": "P", "b2_central": "X", "b3_resnet": "*"}
INK, MUTED, GRID = "#2b2b29", "#6b6b66", "#e4e3de"
SITE_NAMES = ["BCN", "Vidir-MM", "Vidir-Mod", "Rosendahl", "MSK", "Vienna"]


def _style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "lines.markersize": 6,
        "legend.frameon": False, "axes.titleweight": "bold", "axes.titlesize": 10, "text.color": INK,
    })


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


def parse_id(jid: str):
    p = jid.split("__")
    if len(p) != 4 or not p[2].startswith("eps"):
        return None
    return {"dataset": p[0], "method": p[1], "eps": p[2][3:], "seed": int(p[3][1:])}


def collect() -> pd.DataFrame:
    rows = []
    for d in RUNS.iterdir():
        if not d.is_dir() or not (d / "DONE").exists():
            continue
        meta = parse_id(d.name)
        if not meta:
            continue
        tr, ev = _load(d / "train_result.json"), _load(d / "eval_result.json")
        if not tr:
            continue
        r = {**meta, "run": d.name, "hours": tr.get("seconds", 0) / 3600, "vram": tr.get("peak_vram_gb"),
             "upload_mb": tr.get("upload_mb_per_round")}
        if ev:
            pr = ev["prediction"]
            r.update(bacc=pr["bacc_pooled"], bacc_sites=pr["bacc_mean_sites"], bacc_worst=pr["bacc_worst_site"],
                     auc=pr["macro_auc"], f1=pr["macro_f1"], per_site=pr["per_site"], val=pr.get("val"))
            if "calibration" in ev:
                r.update(ece_before=ev["calibration"]["ece_before"], ece_after=ev["calibration"]["ece_after"])
            r["conformal"] = ev.get("conformal")
            r["fairness"] = ev.get("fairness")
            r["ledger"] = ev.get("ledger")
        r["curve"] = tr.get("curve")
        r["server"] = tr.get("server")
        rows.append(r)
    return pd.DataFrame(rows)


def ms(vals, pct=True, nd=1):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return "—"
    f = 100 if pct else 1
    if len(v) == 1:
        return f"{np.mean(v)*f:.{nd}f} (n=1)"
    return f"{np.mean(v)*f:.{nd}f} ± {np.std(v, ddof=1)*f:.{nd}f} (n={len(v)})"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def save(df: pd.DataFrame, name: str, title: str, note: str = "") -> str:
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=False)
    return f"### {title}\n\n{md_table(df)}\n\n{note}\n"


# ============================================================================ tables
def t_main(R, ds="isic2019", methods=None):
    methods = methods or ["b1_local", "b3_resnet", "b4_probe", "b5_fedlora", "b6_ffalora", "b7_personal", "triguard",
                          "b2_central"]
    sub = R[R.dataset == ds]
    out = []
    for m in methods:
        g = sub[sub.method == m]
        if g.empty:
            continue
        row = {"Method": NAMES.get(m, m)}
        for e in EPS_ORDER:
            row[f"BA ε={e.replace('inf', '∞')}"] = ms(g[g.eps == e].get("bacc", pd.Series(dtype=float)).tolist())
        row["Macro-AUC ε=3"] = ms(g[g.eps == "3"].get("auc", pd.Series(dtype=float)).tolist(), pct=False, nd=3)
        out.append(row)
    return pd.DataFrame(out)


def t_per_site(R, eps="3"):
    sub = R[(R.dataset == "isic2019") & (R.eps == eps) & R.get("per_site", pd.Series(dtype=object)).notna()]
    out = []
    for m in ["b5_fedlora", "b6_ffalora", "b7_personal", "b1_local", "triguard"]:
        g = sub[sub.method == m]
        if g.empty:
            continue
        row = {"Method": NAMES[m]}
        per = defaultdict(list)
        for ps in g.per_site:
            for k, v in ps.items():
                per[int(k)].append(v)
        for k in range(6):
            row[SITE_NAMES[k]] = f"{np.mean(per[k])*100:.1f}" if per[k] else "—"
        row["Worst site"] = f"{np.mean([min(ps.values()) for ps in g.per_site])*100:.1f}"
        row["n"] = len(g)
        out.append(row)
    return pd.DataFrame(out)


CP_NAMES = {"u1_local": ("U1 Local split CP", "yes"), "u2_fcp": ("U2 FCP (Lu et al.)", "no"),
            "u3_naive_dp": ("U3 Naive DP histogram CP", "yes"), "u4_labelshift": ("U4 Label-shift FCP (Plassier et al.)", "no"),
            "dpscp_noshift": ("DP-SCP without shift check", "yes"), "dpscp": ("DP-SCP (proposed)", "yes")}


def t_conformal(R, ds="isic2019", method="triguard", eps="3", alpha="0.1"):
    g = R[(R.dataset == ds) & (R.method == method) & (R.eps == eps)]
    g = g[g.get("conformal", pd.Series(dtype=object)).notna()] if "conformal" in g else g.iloc[0:0]
    out = []
    for key, (name, priv) in CP_NAMES.items():
        cs = [c[key][alpha] for c in g.conformal if c and key in c]
        if not cs:
            continue
        ece = g.ece_after if key.startswith("dpscp") else g.ece_before
        out.append({"Method": name, "Private?": priv,
                    "Marginal cov.": f"{np.mean([c['coverage'] for c in cs]):.3f}",
                    "Worst-site cov.": f"{np.mean([c['worst_site_cov'] for c in cs]):.3f}",
                    "Worst-class cov.": f"{np.mean([c['worst_class_cov'] for c in cs]):.3f}",
                    "MEL cov.": f"{np.mean([c['per_class_cov'].get('0', np.nan) for c in cs]):.3f}",
                    "Set size": f"{np.mean([c['set_size'] for c in cs]):.2f}",
                    "P(cov ≥ 1−α)": f"{np.mean([c['prob_cov_ge_target'] for c in cs]):.2f}",
                    "ECE": f"{np.mean(ece):.3f}", "n": len(cs)})
    return pd.DataFrame(out)


def t_camelyon(R):
    out = []
    for m in ["b3_resnet", "b4_probe", "b5_fedlora", "b6_ffalora", "triguard"]:
        g = R[(R.dataset == "camelyon17") & (R.method == m) & (R.eps == "3")]
        if g.empty or "per_site" not in g:
            continue
        g = g[g.per_site.notna()]
        if g.empty:
            continue
        cov = [c["dpscp"]["0.1"]["per_site_cov"].get("4") for c in g.conformal if c]
        size = [c["dpscp"]["0.1"]["set_size_site"].get("4") for c in g.conformal if c]
        out.append({"Method (ε=3)": NAMES[m],
                    "Hospital 4 (val) acc.": ms([(v or {}).get("3") for v in g.val]),
                    "Hospital 5 (unseen) acc.": ms([ps.get("4") for ps in g.per_site]),
                    "Coverage on hospital 5": f"{np.mean([c for c in cov if c is not None]):.3f}" if cov else "—",
                    "Set size": f"{np.mean([s for s in size if s is not None]):.2f}" if size else "—"})
    return pd.DataFrame(out)


def t_medmnist(R):
    out = []
    for f in ["pathmnist", "bloodmnist", "organamnist", "octmnist", "pneumoniamnist", "retinamnist"]:
        row = {"Subset": f}
        for m, lab in [("b3_resnet", "DP-FedAvg full FT"), ("b6_ffalora", "FFA-LoRA + DP"), ("triguard", "TriGuard-FL")]:
            g = R[(R.dataset == f) & (R.method == m) & (R.eps == "3")]
            row[lab] = ms(g.get("bacc", pd.Series(dtype=float)).tolist()) if not g.empty else "—"
        g = R[(R.dataset == f) & (R.method == "triguard") & (R.eps == "3")]
        cs = [c["dpscp"]["0.1"]["coverage"] for c in g.get("conformal", []) if c]
        row["TriGuard-FL coverage"] = f"{np.mean(cs):.3f}" if cs else "—"
        out.append(row)
    return pd.DataFrame(out)


def t_pad():
    out = []
    for m in ["b6_ffalora", "triguard"]:
        r = _load(RUNS / f"pad__{m}" / "transfer_result.json")
        if not r:
            continue
        out.append({"Setting": f"{NAMES[m]}, ISIC thresholds", "Balanced acc. (%)": f"{r['bacc']*100:.1f}",
                    "Coverage (α=0.10)": f"{r['isic_thresholds']['coverage']:.3f}",
                    "Set size": f"{r['isic_thresholds']['set_size']:.2f}"})
        if m == "triguard":
            out.append({"Setting": f"{NAMES[m]} + {r['recal_n']}-image private recalibration",
                        "Balanced acc. (%)": f"{r['bacc']*100:.1f}",
                        "Coverage (α=0.10)": f"{r['private_recalibration']['coverage']:.3f}",
                        "Set size": f"{r['private_recalibration']['set_size']:.2f}"})
    return pd.DataFrame(out)


def t_heart():
    res = defaultdict(list)
    for d in RUNS.glob("heart__*"):
        r = _load(d / "heart_result.json")
        if r:
            res[(r["method"], "inf" if not math.isfinite(r["eps"]) else f"{r['eps']:g}")].append(r)
    out = []
    for m, lab in [("dpfedavg_lr", "DP-FedAvg (logistic regression) + local CP"), ("re_dpscp", "RE aggregation + DP-SCP (C2–C4)")]:
        for e in EPS_ORDER:
            rs = res.get((m, e), [])
            if not rs:
                continue
            out.append({"Method": lab, "ε": e.replace("inf", "∞"), "Mean AUC": ms([r["auc_mean"] for r in rs], False, 3),
                        "Worst-hospital AUC": ms([r["auc_worst"] for r in rs], False, 3),
                        "Coverage": ms([r["coverage"] for r in rs], False, 3),
                        "Worst-site coverage": ms([r["worst_site_coverage"] for r in rs], False, 3)})
    return pd.DataFrame(out)


def t_audit():
    out = []
    for e in ["inf", "8", "3", "1"]:
        lira = _load(RUNS / f"audit__lira__eps{e}" / "audit_result.json")
        can = _load(RUNS / f"audit__canary__eps{e}" / "audit_result.json")
        inv = _load(RUNS / f"audit__inversion__eps{e}" / "audit_result.json")
        out.append({"ε (theory)": e.replace("inf", "∞ (no DP)"),
                    "LiRA AUC": f"{lira['auc']:.3f}" if lira else "—",
                    "TPR at 0.1% FPR": f"{lira['tpr_at_0.1%fpr']*100:.2f}%" if lira else "—",
                    "One-run ε lower bound (95%)": f"{can['eps_emp']:.2f}" if can else "—",
                    "Inversion SSIM": f"{inv['ssim']:.3f}" if inv else "—",
                    "PSNR": f"{inv['psnr']:.1f}" if inv else "—", "LPIPS": f"{inv['lpips']:.3f}" if inv else "—"})
    ref = _load(RUNS / "audit__inversion__fedavg_full_nodp" / "audit_result.json")
    note = (f"Full fine-tuning without DP (FedAvg, ResNet-18 GN, batch 8): inversion SSIM {ref['ssim']:.3f}, "
            f"PSNR {ref['psnr']:.1f}, LPIPS {ref['lpips']:.3f}.") if ref else ""
    return pd.DataFrame(out), note


def t_fairness(R):
    out = []
    for e in ["inf", "3"]:
        g = R[(R.dataset == "isic2019") & (R.method == "triguard") & (R.eps == e)]
        fs = [f for f in g.get("fairness", []) if f]
        if not fs:
            continue
        for col in ["sex", "age"]:
            groups = defaultdict(lambda: defaultdict(list))
            for f in fs:
                for k, v in f.get(col, {}).get("groups", {}).items():
                    groups[k]["bacc"].append(v["bacc"])
                    groups[k]["coverage"].append(v["coverage"])
            for k, v in sorted(groups.items()):
                out.append({"ε": e.replace("inf", "∞"), "Attribute": col, "Subgroup": k,
                            "BA": f"{np.mean(v['bacc'])*100:.1f}", "Coverage": f"{np.mean(v['coverage']):.3f}"})
            gaps = [f[col]["bacc_gap"] for f in fs if col in f]
            cg = [f[col]["coverage_gap"] for f in fs if col in f]
            if gaps:
                out.append({"ε": e.replace("inf", "∞"), "Attribute": col, "Subgroup": "Largest gap",
                            "BA": f"{np.mean(gaps)*100:.1f}", "Coverage": f"{np.mean(cg):.3f}"})
    return pd.DataFrame(out)


def t_ablation(R, eps="3"):
    sub = R[(R.dataset == "isic2019") & (R.eps == eps)]
    full = sub[sub.method == "triguard"]
    if full.empty or "bacc" not in full:
        return pd.DataFrame()
    fb = full.bacc.mean()
    out = [{"Variant": "Full TriGuard-FL", "BA": ms(full.bacc.tolist()), "Worst-site BA": ms(full.bacc_worst.tolist()),
            "Δ vs full": "–"}]
    for a in ["a1_fedavg", "a2_nolocal", "a3_trainA", "a4_k1", "a5_imagenet", "a6_biomedclip"]:
        g = sub[sub.method == a]
        if g.empty or g.bacc.isna().all():
            continue
        out.append({"Variant": NAMES[a], "BA": ms(g.bacc.tolist()), "Worst-site BA": ms(g.bacc_worst.tolist()),
                    "Δ vs full": f"{(g.bacc.mean() - fb)*100:+.1f}"})
    return pd.DataFrame(out)


def t_ledger(R):
    g = R[(R.dataset == "isic2019") & (R.method == "triguard") & (R.eps == "3") & (R.seed == 0)]
    if g.empty or not isinstance(g.iloc[0].get("ledger"), list):
        return pd.DataFrame()
    rows = []
    for r in g.iloc[0].ledger:
        rows.append({"Site": SITE_NAMES[r["site"]] if 0 <= r["site"] < 6 else r["site"], "n_train": r.get("train_n"),
                     "q": f"{r.get('train_q', 0):.3f}", "σ": f"{r.get('train_sigma', 0):.3f}", "steps": r.get("train_steps_done"),
                     "δ": f"{r['delta']:.1e}", "C": r.get("train_clip"), "ε_train": f"{r.get('train_eps_train', 0):.3f}",
                     "ε_label": f"{r['eps_label']:.3f}", "ε_temp": f"{r['eps_temp']:.3f}", "ε_cal": f"{r['eps_cal']:.3f}",
                     "ε_calib (RDP)": f"{r['eps_calib_composed']:.3f}", "ε per patient": f"{r['eps_per_patient']:.3f}"})
    return pd.DataFrame(rows)


def t_compute(R):
    hist = [json.loads(l) for l in (RUNS / "history.jsonl").read_text().splitlines()] if (RUNS / "history.jsonl").exists() else []
    df = pd.DataFrame(hist)
    if df.empty:
        return pd.DataFrame()
    df = df[df.rc == 0]
    df["type"] = df.id.str.split("__").str[0] + "/" + df.id.str.split("__").str[1].fillna("")
    g = df.groupby("type").hours.agg(["count", "mean", "sum"]).reset_index()
    g.columns = ["Job type", "Runs", "Hours each (mean)", "Hours total"]
    g["Hours each (mean)"] = g["Hours each (mean)"].round(2)
    g["Hours total"] = g["Hours total"].round(1)
    return g.sort_values("Hours total", ascending=False)


# ============================================================================ statistics
def stats(R):
    """Paired Wilcoxon (by seed) and bootstrap CI of the BA difference, TriGuard vs best baseline, per eps."""
    out = []
    sub = R[(R.dataset == "isic2019") & R.get("bacc", pd.Series(dtype=float)).notna()]
    for e in EPS_ORDER:
        s = sub[sub.eps == e]
        tg = s[s.method == "triguard"].set_index("seed").bacc
        base = {m: s[s.method == m].set_index("seed").bacc for m in ["b4_probe", "b5_fedlora", "b6_ffalora", "b7_personal"]}
        base = {m: v for m, v in base.items() if len(v)}
        if tg.empty or not base:
            continue
        best = max(base, key=lambda m: base[m].mean())
        common = sorted(set(tg.index) & set(base[best].index))
        d = (tg.loc[common] - base[best].loc[common]).values
        p = wilcoxon(d).pvalue if len(d) >= 5 and np.any(d != 0) else float("nan")
        ci = _bootstrap_diff(f"isic2019__triguard__eps{e}__s0", f"isic2019__{best}__eps{e}__s0")
        out.append({"ε": e.replace("inf", "∞"), "Best baseline": NAMES[best], "ΔBA (pp, mean over seeds)": f"{d.mean()*100:+.2f}",
                    "seeds": len(d), "Wilcoxon p": f"{p:.4f}" if not math.isnan(p) else "n<5",
                    "Bootstrap 95% CI (seed 0, pp)": ci})
    return pd.DataFrame(out)


def _bootstrap_diff(run_a, run_b, B=2000):
    from triguard.metrics import balanced_accuracy

    try:
        pa, pb = np.load(RUNS / run_a / "preds.npz"), np.load(RUNS / run_b / "preds.npz")
        from triguard.cache import Cache

        df = Cache("isic2019").df.set_index("cache_row")
        keys = sorted(k for k in pa.files if k.startswith("test_") and k.endswith("_logits"))
        rows = np.concatenate([pa[k.replace("logits", "rows")] for k in keys])
        ya = np.concatenate([pa[k].argmax(1) for k in keys])
        yb = np.concatenate([pb[k].argmax(1) for k in keys])
        y = df.loc[rows, "y"].values
        rng = np.random.default_rng(0)
        ds = []
        for _ in range(B):
            i = rng.integers(0, len(y), len(y))
            ds.append(balanced_accuracy(y[i], ya[i]) - balanced_accuracy(y[i], yb[i]))
        lo, hi = np.percentile(ds, [2.5, 97.5])
        return f"[{lo*100:+.2f}, {hi*100:+.2f}]"
    except Exception:  # noqa: BLE001
        return "—"


# ============================================================================ figures
def _legend(ax):
    h, l = ax.get_legend_handles_labels()
    if len(h) >= 2:
        ax.legend(h, l, fontsize=7, loc="best")


def fig_privacy_utility(R):
    sub = R[(R.dataset == "isic2019") & R.get("bacc", pd.Series(dtype=float)).notna()]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    xs = {"1": 1, "3": 3, "8": 8, "inf": 20}
    for m in ["b1_local", "b3_resnet", "b4_probe", "b5_fedlora", "b6_ffalora", "b7_personal", "b2_central", "triguard"]:
        g = sub[sub.method == m]
        pts = [(xs[e], g[g.eps == e].bacc.mean() * 100, g[g.eps == e].bacc.std() * 100) for e in EPS_ORDER if (g.eps == e).any()]
        if len(pts) < 1:
            continue
        x, y, s = zip(*pts)
        ax.errorbar(x, y, yerr=np.nan_to_num(s), color=COLORS[m], marker=MARKERS[m], label=NAMES[m], capsize=2,
                    lw=2.6 if m == "triguard" else 1.6, ls="--" if m == "b2_central" else "-")
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 8, 20])
    ax.set_xticklabels(["1", "3", "8", "∞"])
    ax.set_xlabel("privacy budget ε")
    ax.set_ylabel("balanced accuracy (%)")
    ax.set_title("Privacy–utility on Fed-ISIC2019")
    _legend(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig02_privacy_utility.png")
    plt.close(fig)


def fig_convergence(R):
    sub = R[(R.dataset == "isic2019") & (R.eps == "3") & (R.seed == 0)]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    n = 0
    for m in ["b4_probe", "b5_fedlora", "b6_ffalora", "b7_personal", "triguard"]:
        g = sub[sub.method == m]
        if g.empty or not g.iloc[0].curve:
            continue
        c = g.iloc[0].curve
        ax.plot([p["round"] for p in c], [p["mean"] * 100 for p in c], color=COLORS[m], marker=MARKERS[m], ms=4,
                label=NAMES[m], lw=2.6 if m == "triguard" else 1.6)
        n += 1
    if not n:
        plt.close(fig)
        return
    ax.set_xlabel("communication round")
    ax.set_ylabel("mean per-site balanced accuracy (%)")
    ax.set_title("Convergence at ε = 3")
    _legend(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig03_convergence.png")
    plt.close(fig)


def fig_per_site(R):
    sub = R[(R.dataset == "isic2019") & (R.eps == "3") & R.get("per_site", pd.Series(dtype=object)).notna()]
    ms_ = [m for m in ["b5_fedlora", "b6_ffalora", "triguard"] if (sub.method == m).any()]
    if not ms_:
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    w = 0.8 / len(ms_)
    for i, m in enumerate(ms_):
        g = sub[sub.method == m]
        vals = [np.mean([ps[str(k)] if str(k) in ps else ps.get(k, np.nan) for ps in g.per_site]) * 100 for k in range(6)]
        ax.bar(np.arange(6) + (i - (len(ms_) - 1) / 2) * w, vals, w * 0.92, color=COLORS[m], label=NAMES[m])
    ax.set_xticks(range(6))
    ax.set_xticklabels([f"{s}" for s in SITE_NAMES])
    ax.set_ylabel("balanced accuracy (%)")
    ax.set_title("Per-site balanced accuracy at ε = 3 (sites ordered large → small)")
    _legend(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig04_per_site.png")
    plt.close(fig)


CP_COL = {"dpscp": "#2a78d6", "u1_local": "#008300", "u2_fcp": "#eb6834", "u3_naive_dp": "#e87ba4", "u4_labelshift": "#6250d6"}


def fig_coverage(R):
    g = R[(R.dataset == "isic2019") & (R.method == "triguard") & (R.eps == "3")]
    cs = [c for c in g.get("conformal", []) if c]
    if not cs:
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    keys = [k for k in CP_COL if k in cs[0]]
    w = 0.8 / len(keys)
    for i, k in enumerate(keys):
        vals = [np.mean([c[k]["0.1"]["per_site_cov"].get(str(s), np.nan) for c in cs]) for s in range(6)]
        ax.bar(np.arange(6) + (i - (len(keys) - 1) / 2) * w, vals, w * 0.92, color=CP_COL[k], label=CP_NAMES[k][0])
    ax.axhline(0.9, color=INK, lw=1, ls="--")
    ax.text(5.45, 0.905, "target 0.90", fontsize=7, color=INK, ha="right")
    ax.set_ylim(0.6, 1.0)
    ax.set_xticks(range(6))
    ax.set_xticklabels(SITE_NAMES)
    ax.set_ylabel("coverage")
    ax.set_title("Per-site coverage of prediction sets (α = 0.10, ε = 3)")
    _legend(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig05_site_coverage.png")
    plt.close(fig)
    # fig 6: coverage vs alpha (left) and set size vs eps (right) -- two panels, one axis each
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    alphas = ["0.05", "0.1", "0.15", "0.2"]
    for k in keys:
        a1.plot([1 - float(a) for a in alphas], [np.mean([c[k][a]["coverage"] for c in cs]) for a in alphas],
                color=CP_COL[k], marker="o", ms=4, label=CP_NAMES[k][0])
    a1.plot([0.8, 0.95], [0.8, 0.95], color=MUTED, lw=1, ls=":")
    a1.set_xlabel("target coverage 1 − α")
    a1.set_ylabel("empirical coverage")
    a1.set_title("Coverage validity")
    _legend(a1)
    for m in ["triguard"]:
        pts = []
        for e in EPS_ORDER:
            gg = R[(R.dataset == "isic2019") & (R.method == m) & (R.eps == e)]
            cc = [c for c in gg.get("conformal", []) if c]
            if cc:
                pts.append((e, np.mean([c["dpscp"]["0.1"]["set_size"] for c in cc])))
        if pts:
            a2.plot(range(len(pts)), [p[1] for p in pts], color=COLORS[m], marker="o")
            a2.set_xticks(range(len(pts)))
            a2.set_xticklabels([p[0].replace("inf", "∞") for p in pts])
    a2.set_xlabel("privacy budget ε")
    a2.set_ylabel("mean set size (DP-SCP, α = 0.10)")
    a2.set_title("Efficiency vs privacy")
    fig.tight_layout()
    fig.savefig(FIG / "fig06_coverage_alpha_size_eps.png")
    plt.close(fig)


def fig_reliability(R):
    g = R[(R.dataset == "isic2019") & (R.method == "triguard") & (R.eps == "3") & (R.seed == 0)]
    if g.empty:
        return
    ev = _load(RUNS / g.iloc[0].run / "eval_result.json")
    if not ev or "calibration" not in ev:
        return
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    for key, col, lab in [("reliability_before", "#eb6834", f"before (ECE {ev['calibration']['ece_before']:.3f})"),
                          ("reliability_after", "#2a78d6", f"after DP temperature (ECE {ev['calibration']['ece_after']:.3f})")]:
        pts = [(c, a) for c, a, n in ev["calibration"][key] if n > 0 and a is not None]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, marker="o", ms=4, label=lab)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title("Reliability at ε = 3")
    _legend(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig07_reliability.png")
    plt.close(fig)


EPS_COL = {"inf": "#e34948", "8": "#eda100", "3": "#2a78d6", "1": "#008300"}


def fig_audit():
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    n = 0
    for e in ["inf", "8", "3", "1"]:
        r = _load(RUNS / f"audit__lira__eps{e}" / "audit_result.json")
        if not r:
            continue
        ax.plot(r["roc"]["fpr"], np.clip(r["roc"]["tpr"], 1e-5, 1), color=EPS_COL[e],
                label=f"ε = {e.replace('inf', '∞')} (AUC {r['auc']:.3f})")
        n += 1
    if n:
        ax.plot([1e-5, 1], [1e-5, 1], color=MUTED, lw=1, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.set_title("LiRA membership inference (site BCN)")
        _legend(ax)
        fig.tight_layout()
        fig.savefig(FIG / "fig08_lira_roc.png")
    plt.close(fig)
    pts = []
    for e in ["8", "3", "1"]:
        r = _load(RUNS / f"audit__canary__eps{e}" / "audit_result.json")
        if r:
            pts.append((float(e), r["eps_emp"]))
    if pts:
        fig, ax = plt.subplots(figsize=(3.6, 3.4))
        ax.plot([0, 9], [0, 9], color=MUTED, lw=1, ls=":", label="ε_emp = ε_theory")
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#2a78d6", marker="o", ls="none", ms=8,
                label="one-run 95% lower bound")
        for x, y in pts:
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(6, -10), fontsize=7, color=INK)
        ax.set_xlabel("theoretical ε")
        ax.set_ylabel("empirical ε lower bound")
        ax.set_title("Audit vs guarantee (below diagonal = consistent)")
        _legend(ax)
        fig.tight_layout()
        fig.savefig(FIG / "fig09_empirical_eps.png")
        plt.close(fig)
    bars = []
    ref = _load(RUNS / "audit__inversion__fedavg_full_nodp" / "audit_result.json")
    if ref:
        bars.append(("full FT, no DP", ref["ssim"], "#e34948"))
    for e in ["inf", "8", "3", "1"]:
        r = _load(RUNS / f"audit__inversion__eps{e}" / "audit_result.json")
        if r:
            bars.append((f"TriGuard ε={e.replace('inf', '∞')}", r["ssim"], EPS_COL[e] if e != "inf" else "#6250d6"))
    if bars:
        fig, ax = plt.subplots(figsize=(4.8, 3.0))
        ax.bar(range(len(bars)), [b[1] for b in bars], 0.6, color=[b[2] for b in bars])
        for i, b in enumerate(bars):
            ax.text(i, b[1] + 0.005, f"{b[1]:.3f}", ha="center", fontsize=7, color=INK)
        ax.set_xticks(range(len(bars)))
        ax.set_xticklabels([b[0] for b in bars], fontsize=7)
        ax.set_ylabel("SSIM (higher = more leakage)")
        ax.set_title("Gradient-inversion reconstruction quality")
        fig.tight_layout()
        fig.savefig(FIG / "fig10_inversion_ssim.png")
        plt.close(fig)


def fig_ablation(R):
    t = t_ablation(R)
    if t.empty or len(t) < 2:
        return
    sub = R[(R.dataset == "isic2019") & (R.eps == "3")]
    full = sub[sub.method == "triguard"].bacc.mean()
    items = [(NAMES[a], (sub[sub.method == a].bacc.mean() - full) * 100) for a in
             ["a1_fedavg", "a2_nolocal", "a3_trainA", "a4_k1", "a5_imagenet", "a6_biomedclip"] if (sub.method == a).any()]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.barh(range(len(items)), [i[1] for i in items], 0.6, color=["#2a78d6" if i[1] >= 0 else "#eb6834" for i in items])
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([i[0] for i in items], fontsize=7)
    ax.axvline(0, color=INK, lw=1)
    for k, it in enumerate(items):
        ax.text(it[1] + (0.1 if it[1] >= 0 else -0.1), k, f"{it[1]:+.1f}", va="center", ha="left" if it[1] >= 0 else "right",
                fontsize=7, color=INK)
    ax.set_xlabel("Δ balanced accuracy vs full TriGuard-FL (pp, ε = 3)")
    ax.set_title("Contribution of each component")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG / "fig11_ablation.png")
    plt.close(fig)


def fig_cost(R):
    sub = R[(R.dataset == "isic2019") & (R.eps == "3") & R.get("bacc", pd.Series(dtype=float)).notna()]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for m, g in sub.groupby("method"):
        if m not in COLORS:
            continue
        ax.scatter(g.hours.mean(), g.bacc.mean() * 100, s=max(30, (g.vram.mean() or 0.5) * 80), color=COLORS[m],
                   edgecolor="white", linewidth=2, label=NAMES[m], zorder=3)
    ax.set_xlabel("GPU-hours per run (RTX 3050 Laptop 4 GB)")
    ax.set_ylabel("balanced accuracy at ε = 3 (%)")
    ax.set_title("Accuracy vs cost (bubble = peak VRAM)")
    _legend(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig12_cost.png")
    plt.close(fig)


def fig_re_weights(R):
    g = R[(R.dataset == "isic2019") & (R.method == "triguard") & (R.eps == "3") & (R.seed == 0)]
    if g.empty or not g.iloc[0].server:
        return
    W = np.array([h["weights"] for h in g.iloc[0].server if "weights" in h])
    if W.size == 0:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    for k in range(W.shape[1]):
        a1.plot(np.arange(1, len(W) + 1), W[:, k], color=cols[k], label=SITE_NAMES[k], lw=1.6)
    a1.set_xlabel("round")
    a1.set_ylabel("aggregation weight w_k")
    a1.set_title("Random-effects weights")
    _legend(a1)
    tau = [h.get("tau2", np.nan) for h in g.iloc[0].server]
    a2.plot(np.arange(1, len(tau) + 1), tau, color="#6250d6")
    a2.set_xlabel("round")
    a2.set_ylabel("τ̂² (between-site variance)")
    a2.set_title("Estimated heterogeneity")
    fig.tight_layout()
    fig.savefig(FIG / "fig_extra_re_weights.png")
    plt.close(fig)


def fig_inversion_grid():
    import torch

    paths = [("original", None), ("full FT, no DP", "audit__inversion__fedavg_full_nodp")] + [
        (f"TriGuard ε={e.replace('inf', '∞')}", f"audit__inversion__eps{e}") for e in ["inf", "8", "3"]]
    data = []
    for lab, p in paths:
        if p is None:
            continue
        f = RUNS / p / "examples.pt"
        if f.exists():
            data.append((lab, torch.load(f)))
    if not data:
        return
    n = 6
    rows = [("original", data[0][1]["orig"])] + [(lab, d["rec"]) for lab, d in data]
    fig, axes = plt.subplots(len(rows), n, figsize=(n * 1.2, len(rows) * 1.25))
    for i, (lab, imgs) in enumerate(rows):
        for j in range(n):
            ax = axes[i, j]
            ax.imshow(imgs[j].permute(1, 2, 0).numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if j == 0:
                ax.set_ylabel(lab, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_extra_inversion_examples.png")
    plt.close(fig)


# ============================================================================ main
def main():
    _style()
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    R = collect()
    parts = ["# TriGuard-FL — measured results", "",
             "All numbers below are **measured** from the runs in `runs/` (mean ± SD over seeds; n = seeds finished). "
             "'—' = not finished yet. Regenerate any time with `python -m triguard.report`.", ""]
    if R.empty:
        R = pd.DataFrame(columns=["dataset", "method", "eps", "seed", "run", "bacc", "per_site", "conformal", "hours", "vram"])
    blocks = [
        (t_main(R), "t10_main", "Table 10. Balanced accuracy (%) on the Fed-ISIC2019 test set (pooled)"),
        (t_per_site(R), "t11_per_site", "Table 11. Per-site balanced accuracy (%) at ε = 3"),
        (t_conformal(R), "t12_conformal", "Table 12. Coverage and efficiency at α = 0.10 on top of the ε = 3 TriGuard model"),
        (t_camelyon(R), "t13_camelyon", "Table 13. Camelyon17-WILDS (binary tumour vs normal, α = 0.10)"),
        (t_medmnist(R), "t14_medmnist", "Table 14. MedMNIST+ 224×224, 20 clients, ε = 3 (balanced accuracy %)"),
        (t_pad(), "t15_pad", "Table 15. Transfer to PAD-UFES-20 smartphone images (6 mapped classes, ε = 3)"),
        (t_heart(), "t16_heart", "Table 16. Fed-Heart-Disease (non-image records)"),
    ]
    for df, name, title in blocks:
        if not df.empty:
            parts.append(save(df, name, title))
    a, note = t_audit()
    parts.append(save(a, "t17_audit", "Table 17. Empirical privacy on site 1 (BCN)", note))
    for df, name, title in [
        (t_fairness(R), "t18_fairness", "Table 18. Subgroup results for TriGuard-FL on Fed-ISIC2019"),
        (t_ablation(R, "3"), "t19_ablation_eps3", "Table 19. Ablation on Fed-ISIC2019 (ε = 3)"),
        (t_ablation(R, "1"), "t19b_ablation_eps1", "Table 19b. Ablation on Fed-ISIC2019 (ε = 1)"),
        (t_ledger(R), "t_ledger", "Per-patient privacy ledger (TriGuard-FL, ε = 3, seed 0)"),
        (stats(R), "t_stats", "Statistics: TriGuard-FL vs best baseline"),
        (t_compute(R), "t20_compute", "Table 20. Measured compute"),
    ]:
        if not df.empty:
            parts.append(save(df, name, title))
    for f in [fig_privacy_utility, fig_convergence, fig_per_site, fig_coverage, fig_reliability, fig_ablation, fig_cost,
              fig_re_weights]:
        try:
            f(R)
        except Exception as e:  # noqa: BLE001
            print("figure failed", f.__name__, e)
    for f in [fig_audit, fig_inversion_grid]:
        try:
            f()
        except Exception as e:  # noqa: BLE001
            print("figure failed", f.__name__, e)
    figs = sorted(FIG.glob("*.png"))
    parts += ["## Figures", ""] + [f"![{p.stem}](figures/{p.name})" for p in figs]
    (OUT / "RESULTS.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {OUT / 'RESULTS.md'} ({len(R)} runs, {len(figs)} figures)")


if __name__ == "__main__":
    main()
