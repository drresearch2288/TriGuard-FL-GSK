"""DP-SCP keeps split-conformal coverage under DP noise and stays informative (not the full label set)."""
import numpy as np

from triguard import conformal as cp
from triguard import dp


def _synthetic(rng, n, C=8):
    pri = np.array([0.5, 0.18, 0.13, 0.09, 0.04, 0.03, 0.02, 0.01])[:C]
    pri = pri / pri.sum()
    y = rng.choice(C, n, p=pri)
    logits = rng.normal(0, 1, (n, C))
    logits[np.arange(n), y] += 2.0
    return cp.probs_from_logits(logits), y


def test_dpscp_coverage_and_size():
    rng = np.random.default_rng(0)
    C, sizes = 8, [1900, 630, 530, 360, 130, 65]
    deltas = {k: dp.site_delta(4 * n) for k, n in enumerate(sizes)}
    covs, set_sizes = [], []
    for rep in range(30):
        cal, test = {}, {}
        for k, n in enumerate(sizes):
            p, y = _synthetic(rng, n, C)
            cal[k] = (cp.aps_scores(p, rng, y), y)
            pt, yt = _synthetic(rng, n, C)
            test[k] = (cp.aps_scores(pt, rng), yt)
        m = cp.DPSCP(cp.DPSCPCfg(alpha=0.1, eps_label=0.35, eps_temp=0.24, eps_cal=2.95), C, deltas).fit(cal, rng)
        for k in test:
            s = m.predict_sets(test[k][0], k)
            covs.append(s[np.arange(len(test[k][1])), test[k][1]].mean())
            set_sizes.append(s.sum(1).mean())
    assert np.mean(covs) >= 0.9 - 0.005
    assert np.mean(set_sizes) < C - 1  # informative, not the full label set


def test_dp_baselines_reach_coverage_and_equal_size_lookup():
    rng = np.random.default_rng(1)
    C, sizes = 8, [1900, 630, 530, 360, 130, 65]
    deltas = {k: dp.site_delta(4 * n) for k, n in enumerate(sizes)}
    cal = {}
    for k, n in enumerate(sizes):
        p, y = _synthetic(rng, n, C)
        cal[k] = (cp.aps_scores(p, rng, y), y)
    pt, yt = _synthetic(rng, 4000, C)
    test = cp.aps_scores(pt, rng)
    idx = np.arange(len(yt))
    q5 = cp.u5_exp_fcp(cal, 0.1, 2.5, rng)
    assert (test <= q5)[idx, yt].mean() >= 0.87
    q6 = cp.u6_exp_mondrian(cal, 0.1, 2.5, C, rng)
    assert q6.shape == (C,) and cp.label_thresholds_sets(test, q6)[idx, yt].mean() >= 0.87
    q7 = cp.u7_dp_fedcp(cal, 0.1, np.inf, deltas, C, rng)  # no noise: must sit near the exact conformal quantile
    exact = cp.u2_fcp(cal, 0.1)[0]
    assert abs(np.median(q7[0]) - exact) < 0.05
    q7dp = cp.u7_dp_fedcp(cal, 0.1, 2.5, deltas, C, rng)
    assert set(q7dp) == set(cal)
    curve = [{"set_size": 2.0, "worst_site": 0.8}, {"set_size": 4.0, "worst_site": 0.9}]
    assert abs(cp.at_equal_size(curve, 3.0, "worst_site") - 0.85) < 1e-9
    assert cp.at_equal_size(curve, 5.0, "worst_site") is None
