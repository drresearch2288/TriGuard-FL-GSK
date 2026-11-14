"""B3 search on the public ISIC 2020 proxy (no privacy cost): normalisation layer and learning rate."""
import json
import math
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from triguard import cnn
from triguard.cache import Cache
from triguard.paths import RUNS

def main():
    GRID = [(math.inf, "gn", 0.05), (math.inf, "frozen", 0.05), (math.inf, "frozen", 0.5),
            (3.0, "gn", 0.05), (3.0, "frozen", 0.05), (3.0, "frozen", 0.2), (3.0, "frozen", 0.01)]
    df = Cache("isic2020", "dinov2_vits14", features=False).df.set_index("cache_row")
    for eps, norm, lr in GRID:
        jid = f"b3probe__eps{eps}__{norm}__lr{lr}"
        (RUNS / jid).mkdir(exist_ok=True)
        if not (RUNS / jid / "train_result.json").exists():
            cnn.run(jid, "isic2020", eps, 0, rounds=30, lr=lr, norm=norm)
        p = np.load(RUNS / jid / "preds.npz")
        ks = [k for k in p.files if k.startswith("val_") and k.endswith("logits")] or \
             [k for k in p.files if k.startswith("test_") and k.endswith("logits")]
        y = np.concatenate([df.loc[p[k.replace("logits", "rows")], "y"].values for k in ks])
        lg = np.concatenate([p[k] for k in ks])
        print(json.dumps({"eps": eps, "norm": norm, "lr": lr, "macro_auc": roc_auc_score(y, lg[:, 1] - lg[:, 0]) if np.isfinite(lg).all() else None,
                          "split": ks[0].split("_")[0]}), flush=True)


if __name__ == "__main__":
    main()
