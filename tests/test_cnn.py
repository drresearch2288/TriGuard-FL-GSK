"""B3 smoke test on a tiny synthetic image dataset (registered on the fly)."""
import math
import sys
import types

import numpy as np
import pandas as pd
from PIL import Image

from tests.test_pipeline import DS, make_synth_cache


def test_cnn_runs(tmp_path, monkeypatch):
    make_synth_cache(n_per_site=(24, 12), C=2)
    mod = types.ModuleType("synth_imgs")
    rng = np.random.default_rng(0)
    mod.load_image = lambda row: Image.fromarray((rng.random((64, 64, 3)) * 255).astype("uint8"))
    from triguard import data

    monkeypatch.setitem(data.DATASETS, DS, "synth_imgs")
    sys.modules["synth_imgs"] = mod
    from triguard import cnn
    from triguard.paths import RUNS

    for eps in (3.0, math.inf):
        jid = f"_test_cnn_{eps}"
        cnn.run(jid, DS, eps, 0, rounds=2, logical_batch=8, phys_batch=4, workers=0)
        assert (RUNS / jid / "preds.npz").exists()
