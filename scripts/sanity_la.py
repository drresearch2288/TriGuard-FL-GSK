"""Short ISIC check: logit-adjusted vs plain CE (TriGuard, eps=3, tuned hparams, few rounds)."""
import sys

from triguard.experiments import method_cfg
from triguard.fl import FLTrainer
from triguard.jobrun import make_fl_cfg

rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
for loss in sys.argv[2:] or ["logit_adjusted", "ce"]:
    a = method_cfg("triguard")
    a.update(dataset="isic2019", eps=3.0, seed=0, rounds=rounds, loss=loss, eval_every=rounds)
    r = FLTrainer(make_fl_cfg(f"_sanity_{loss}", a)).run()
    print(f"RESULT {loss}: mean per-site BA {r['bacc_mean']:.3f}  worst {r['bacc_worst']:.3f}  {r['seconds']/60:.1f} min",
          flush=True)
