"""Execute one job from runs/jobs.json:  python -m triguard.jobrun <job id>"""
from __future__ import annotations

import dataclasses
import json
import math
import sys

from triguard.paths import RUNS
from triguard.utils import load_json, save_json


def tuned(stage: str = "tune__select") -> dict:
    f = RUNS / stage / "best.json"
    return load_json(f) if f.exists() else {}


def make_fl_cfg(jid: str, args: dict):
    from triguard.fl import FLCfg
    from triguard.model import ModelCfg

    a = dict(args)
    no_tuning = a.pop("no_tuning", False)
    use_stage1 = a.pop("use_stage1", False)
    a.pop("disk_gb", None)
    model = dict(a.pop("model", {}) or {})
    best = {}
    if use_stage1:
        best = tuned("tune__select_cl")
    elif a.get("dataset") != "isic2020" and not no_tuning:
        best = tuned()
    for k in ("clip", "lr"):
        if k in best and k not in a:
            a[k] = best[k]
    if "rank" in best and "rank" not in model:
        model["rank"] = best["rank"]
        model["alpha"] = 2.0 * best["rank"]
    fields = {f.name for f in dataclasses.fields(FLCfg)}
    unknown = set(a) - fields
    assert not unknown, f"unknown cfg keys {unknown}"
    a["eps"] = float(a.get("eps", 3.0))
    return FLCfg(model=ModelCfg(**model), out=str(RUNS / jid), **a)


def main(jid: str):
    J = json.loads((RUNS / "jobs.json").read_text())
    job = J[jid]
    kind, args = job["kind"], dict(job["args"])
    print(f"job {jid} kind={kind} args={args}", flush=True)

    if kind == "download":
        sys.argv = ["download_all", "--only", args["step"]]
        from scripts import download_all

        if args.get("flag"):
            download_all.medmnist_flags([args["flag"]])
        else:
            download_all.main()
    elif kind == "prepare":
        from triguard import data

        data.get(args["dataset"]).prepare()
    elif kind == "cache":
        from triguard.cache import build_cache

        args.pop("disk_gb", None)
        build_cache(**args)
    elif kind == "train":
        from triguard.fl import FLTrainer

        FLTrainer(make_fl_cfg(jid, args)).run()
    elif kind == "alias":  # identical config + seed to another run: copy its training outputs
        import shutil

        src, dst = RUNS / args["source"], RUNS / jid
        for f in ("params.pt", "preds.npz", "train_result.json", "train.log"):
            if (src / f).exists():
                shutil.copy2(src / f, dst / f)
        r = load_json(dst / "train_result.json")
        r["alias_of"] = args["source"]
        save_json(r, dst / "train_result.json")
    elif kind == "eval":
        from triguard import evaluate

        evaluate.main(str(RUNS / args["run"]))
    elif kind == "tune_select":
        select_hparams(jid, args)
    elif kind == "cnn":
        from triguard import cnn

        cnn.run(jid, **args)
    elif kind == "cleanup":
        from triguard.cache import delete_cache

        delete_cache(*args["cache"])
    elif kind == "canary":
        from triguard import audit

        audit.canary_audit(jid, **args)
    elif kind == "inversion":
        from triguard import audit

        audit.inversion(jid, **args)
    elif kind == "inversion_cnn":
        from triguard import audit

        audit.inversion_cnn(jid, **args)
    elif kind == "lira_shadow":
        from triguard import audit

        audit.lira_shadow(jid, **args)
    elif kind == "lira_score":
        from triguard import audit

        audit.lira_score(jid, **args)
    elif kind == "pad_transfer":
        from triguard import transfer

        transfer.pad_transfer(jid, **args)
    elif kind == "heart":
        from triguard import heart

        heart.run(jid, **args)
    elif kind == "cooldown":  # GPU rest: pending GPU jobs depend on this job
        import time

        mins = float(args.get("minutes", 15))
        print(f"GPU cooldown: {mins:.0f} min", flush=True)
        time.sleep(mins * 60)
    elif kind == "report":
        from triguard import report

        report.main()
    else:
        raise ValueError(kind)


def select_hparams(jid, args):
    """Pick (clip, lr[, rank]) by macro-AUC on the public proxy's test split.

    AUC, not balanced accuracy: on the imbalanced binary proxy every DP model predicts the majority class at the
    argmax threshold (BA = 0.5 for all), so BA cannot rank settings while AUC still does."""
    import numpy as np

    from triguard.cache import Cache
    from triguard.metrics import macro_auc

    rows = []
    for r in args["runs"]:
        f = RUNS / r / "train_result.json"
        if not f.exists() or not (RUNS / r / "preds.npz").exists():
            continue
        res = load_json(f)
        c = res["cfg"]
        df = Cache(c["dataset"], c["backbone"]).df.set_index("cache_row")
        P = np.load(RUNS / r / "preds.npz")
        keys = [k for k in P.files if k.startswith("test_") and k.endswith("_logits")]
        lg = np.concatenate([P[k] for k in keys])
        y = df.loc[np.concatenate([P[k.replace("logits", "rows")] for k in keys]), "y"].values
        rows.append({"run": r, "auc": macro_auc(y, lg), "bacc": res["bacc_mean"], "clip": c["clip"], "lr": c["lr"],
                     "rank": c["model"]["rank"]})
    assert rows, "no tuning results"
    best = max(rows, key=lambda x: x["auc"])
    out = {"clip": best["clip"], "lr": best["lr"]}
    if args.get("stage", 1) == 2:
        out["rank"] = best["rank"]
    save_json({**out, "criterion": "macro_auc", "table": sorted(rows, key=lambda x: -x["auc"])}, RUNS / jid / "best.json")
    print("selected", out)


if __name__ == "__main__":
    main(sys.argv[1])
