"""Dry-run every method config on the dummy cache (1 round) -> catches config errors before the long queue."""
import math
import shutil

from triguard import evaluate
from triguard.experiments import method_cfg
from triguard.fl import FLTrainer
from triguard.jobrun import make_fl_cfg
from triguard.paths import RUNS

ok = []
for m in ["triguard", "b1_local", "b2_central", "b4_probe", "b5_fedlora", "b6_ffalora", "b7_personal",
          "a1_re", "a2_gated", "a4_k1", "a8_triguardD"]:
    for eps in (3.0, math.inf):
        a = method_cfg(m)
        a.update(dataset="dummy", eps=eps, seed=0, rounds=1, no_tuning=True, logical_batch=8, phys_batch=4)
        a["views"] = min(a.get("views", 2), 2)
        jid = f"_dry__{m}__{eps}"
        FLTrainer(make_fl_cfg(jid, a)).run()
        evaluate.main(str(RUNS / jid), reps=1)
        shutil.rmtree(RUNS / jid)
        ok.append((m, eps))
print("ALL OK", len(ok))
