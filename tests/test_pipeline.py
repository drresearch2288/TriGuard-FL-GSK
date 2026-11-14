"""End-to-end smoke test on a synthetic cache: FL training (DP + non-DP), evaluation, DP-SCP, ledger."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from triguard.cache import cache_dir
from triguard.fl import FLCfg, FLTrainer
from triguard.model import ModelCfg

DS = "synth"


def make_synth_cache(n_per_site=(120, 60, 30), C=3, T=257, D=384, K=2, seed=0, name=DS, unseen_test=False):
    rng = np.random.default_rng(seed)
    d = cache_dir(name, "dinov2_vits14")
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for s, n in enumerate(n_per_site):
        for split, m in [("train", n), ("calib", n // 3), ("test", n // 2)]:
            for _ in range(m):
                rows.append({"uid": f"u{len(rows)}", "site": s, "split": split, "y": int(rng.integers(C)),
                             "sex": rng.choice(["male", "female"]), "age": float(rng.integers(20, 80))})
    if unseen_test:  # an unseen hospital that only has test data (Camelyon-style)
        for _ in range(40):
            rows.append({"uid": f"u{len(rows)}", "site": 9, "split": "test", "y": int(rng.integers(C)),
                         "sex": "male", "age": 50.0})
    df = pd.DataFrame(rows)
    df["cache_row"] = np.arange(len(df))
    tr = df.split == "train"
    df["aug_row"] = -1
    df.loc[tr, "aug_row"] = np.arange(tr.sum())
    means = rng.normal(0, 1, (C, D)).astype(np.float32)
    base = rng.normal(0, 1, (len(df), T, D)).astype(np.float32) * 0.5
    clean = (base + means[df.y.values][:, None, :]).astype(np.float16)
    aug = np.repeat(clean[tr.values][:, None], K, 1) + rng.normal(0, 0.1, (tr.sum(), K, T, D)).astype(np.float16)
    np.save(d / "clean.npy", clean)
    np.save(d / "aug.npy", aug.astype(np.float16))
    df.to_csv(d / "index.csv", index=False)
    (d / "meta.json").write_text(json.dumps({"dataset": DS, "views": K}))
    (d / "DONE").write_text("x")
    return d


@pytest.fixture(scope="module")
def synth():
    return make_synth_cache()


@pytest.mark.parametrize("eps,agg,mcfg,gate", [(3.0, "re", ModelCfg(), None),
                                               (math.inf, "fedavg", ModelCfg(local=False, head_federated=True), None),
                                               (3.0, "re", ModelCfg(head_federated=True), 0.3)])
def test_fl_and_eval(synth, tmp_path, eps, agg, mcfg, gate):
    from triguard import evaluate

    cfg = FLCfg(dataset=DS, eps=eps, rounds=3, logical_batch=32, phys_batch=8, views=2, agg=agg, model=mcfg,
                eval_every=1, out=str(tmp_path / "run"), personal_gate=gate)
    tr = FLTrainer(cfg)
    if gate is not None:  # synthetic sites differ in size -> the SNR gate must split them
        assert len({s.personal for s in tr.sites}) == 2, [s.public_tuple() for s in tr.sites]
    res = tr.run()
    assert 0 <= res["bacc_mean"] <= 1
    if math.isfinite(eps):
        for r in res["ledger"]:
            assert r["eps_train"] <= eps * 1.02
        assert any("tau2" in h for h in res["server"])
    ev = evaluate.main(str(tmp_path / "run"), reps=3)
    assert "dpscp" in ev["conformal"]
    assert ev["conformal"]["u2_fcp"]["0.1"]["coverage"] > 0.7
    for r in ev["ledger"]:
        if math.isfinite(eps):
            assert r["eps_per_patient"] <= eps * 1.05, r


def test_unseen_site_and_canaries(tmp_path):
    from triguard import evaluate

    make_synth_cache(name="synth_u", unseen_test=True)
    make_synth_cache(n_per_site=(20,), name="synth_can", seed=3)
    cfg = FLCfg(dataset="synth_u", eps=3.0, rounds=2, logical_batch=32, phys_batch=8, views=2, eval_every=1,
                extra_train={"dataset": "synth_can", "rows": [0, 1, 2, 5], "labels": [0, 1, 2, 0]},
                out=str(tmp_path / "run"))
    tr = FLTrainer(cfg)
    assert tr.sites[0].n == 120 + 4
    res = tr.run()
    assert 9 in res["bacc_per_site"]
    ev = evaluate.main(str(tmp_path / "run"), reps=2)
    assert 9 in ev["conformal"]["dpscp"]["0.1"]["per_site_cov"]


def test_audit_math():
    from triguard.audit import eps_lower_bound, lira_membership

    assert eps_lower_bound(200, 100) == 0.0
    lb = eps_lower_bound(200, 190)
    assert 1.5 < lb < 4.0
    keep = lira_membership(50, 16)
    assert (keep.sum(0) == 8).all()
