"""Task-wise status: which experiment groups are done / running on the laptop / running on Kaggle / pending."""
import collections
import json

from triguard.paths import RUNS

J = json.loads((RUNS / "jobs.json").read_text())
MED = ("pathmnist", "bloodmnist", "organamnist", "octmnist", "pneumoniamnist", "retinamnist")
NAMES = {"dl": "Downloads", "prep": "Preprocessing", "cache": "Feature caches", "tune": "Hyper-parameter search",
         "eval": "Evaluations (CPU)", "lira": "LiRA shadow models", "audit": "Audits (canary / inversion / LiRA score)",
         "heart": "Fed-Heart (P5)", "camelyon17": "Camelyon17 (P2)", "pad": "PAD-UFES transfer (P4)",
         "clean": "Cache cleanup", "report": "Final report", "cooldown": "GPU break"}


def st(j):
    d = RUNS / j
    if (d / "DONE").exists():
        return "done"
    if (d / "RUNNING").exists():
        return "laptop"
    if (d / "REMOTE").exists():
        return "kaggle"
    return "pending"


def grp(j):
    p = j.split("__")[0]
    if j.startswith("isic2019__a"):
        return "Ablations A1-A6 (ISIC)"
    if j.startswith("isic2019__"):
        return "Main ISIC table + B3 (P1)"
    if p in MED:
        return "MedMNIST+ (P3)"
    return NAMES.get(p, p)


g = collections.defaultdict(collections.Counter)
for j in J:
    g[grp(j)][st(j)] += 1
print(f"{'Task group':42s} {'done':>5s} {'laptop':>7s} {'kaggle':>7s} {'pending':>8s} {'total':>6s}")
for k, c in g.items():
    print(f"{k:42s} {c['done']:5d} {c['laptop']:7d} {c['kaggle']:7d} {c['pending']:8d} {sum(c.values()):6d}")
print("\nRunning on laptop:", ", ".join(j for j in J if st(j) == "laptop") or "-")
print("Claimed by Kaggle:", ", ".join(j for j in J if st(j) == "kaggle") or "-")
