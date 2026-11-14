"""Checkpoint/resume after a shutdown must give the same model as an uninterrupted run."""
import torch

from triguard.experiments import method_cfg
from triguard.fl import FLTrainer
from triguard.jobrun import make_fl_cfg


def _cfg(out):
    a = method_cfg("triguard")
    a.update(dataset="dummy", eps=3.0, seed=0, rounds=6, no_tuning=True, logical_batch=8, phys_batch=4, views=2,
             checkpoint_every=2, eval_every=3)
    c = make_fl_cfg("x", a)
    c.out = str(out)
    return c


def test_resume_matches(tmp_path):
    from triguard.cache import build_cache

    build_cache("dummy", views=2, workers=0)
    full = FLTrainer(_cfg(tmp_path / "full")).run()

    class Crash(Exception):
        pass

    tr = FLTrainer(_cfg(tmp_path / "cut"))
    orig = tr.evaluate
    calls = {"n": 0}

    def boom(split="test"):  # eval runs at round 3; crash at round 6 eval -> last checkpoint = round 4
        calls["n"] += 1
        if calls["n"] == 2:
            raise Crash()
        return orig(split)

    tr.evaluate = boom
    try:
        tr.run()
    except Crash:
        pass
    assert (tmp_path / "cut" / "checkpoint.pt").exists()
    res = FLTrainer(_cfg(tmp_path / "cut")).run()
    assert not (tmp_path / "cut" / "checkpoint.pt").exists()
    a = torch.load(tmp_path / "full" / "params.pt")
    b = torch.load(tmp_path / "cut" / "params.pt")
    for k in a:
        for n in a[k]:
            assert torch.allclose(a[k][n], b[k][n], atol=1e-3, rtol=1e-2), (k, n)
    assert abs(res["bacc_mean"] - full["bacc_mean"]) < 0.05
