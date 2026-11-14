"""Full 50-round B4 linear probe on ISIC at eps = inf and 3 (fast): checks the learning signal end to end."""
import math
import sys

from triguard.experiments import method_cfg
from triguard.fl import FLTrainer
from triguard.jobrun import make_fl_cfg

import os

mom = float(os.environ.get("SERVER_MOM", 0.9))
for eps in [float(e) for e in (sys.argv[1:] or ["inf", "3"])]:
    a = method_cfg(os.environ.get("METHOD", "b4_probe"))
    a.update(dataset="isic2019", eps=eps, seed=0, rounds=int(os.environ.get("ROUNDS", 50)), eval_every=10,
             server_momentum=mom)
    r = FLTrainer(make_fl_cfg(f"_sanity_probe_{eps}", a)).run()
    print(f"RESULT probe eps={eps}: mean per-site BA {r['bacc_mean']:.3f} worst {r['bacc_worst']:.3f} "
          f"{r['seconds']/60:.1f} min  curve={[round(c['mean'], 3) for c in r['curve']]}", flush=True)
