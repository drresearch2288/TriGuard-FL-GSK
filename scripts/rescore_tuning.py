"""Re-score the hyper-parameter search by macro-AUC (threshold-free) on the public proxy's test split."""
import json

import numpy as np

from triguard.cache import Cache
from triguard.metrics import balanced_accuracy, macro_auc
from triguard.paths import RUNS

df = Cache("isic2020").df.set_index("cache_row")
rows = []
for d in sorted(RUNS.glob("tune__*")):
    if not (d / "preds.npz").exists():
        continue
    P = np.load(d / "preds.npz")
    keys = [k for k in P.files if k.startswith("test_") and k.endswith("_logits")]
    lg = np.concatenate([P[k] for k in keys])
    y = df.loc[np.concatenate([P[k.replace("logits", "rows")] for k in keys]), "y"].values
    cfg = json.loads((d / "train_result.json").read_text())["cfg"]
    # balanced accuracy after a prior correction (threshold at the base rate) for information
    pi = np.bincount(y, minlength=2) / len(y)
    adj = (lg - np.log(pi + 1e-9)).argmax(1)
    rows.append({"run": d.name, "auc": macro_auc(y, lg), "bacc": balanced_accuracy(y, lg.argmax(1)),
                 "bacc_prior_adj": balanced_accuracy(y, adj), "clip": cfg["clip"], "lr": cfg["lr"],
                 "rank": cfg["model"]["rank"]})
for r in sorted(rows, key=lambda r: -r["auc"]):
    print(f"{r['run']:28s} AUC={r['auc']:.4f}  BA={r['bacc']:.3f}  BA(prior-adj)={r['bacc_prior_adj']:.3f}")
