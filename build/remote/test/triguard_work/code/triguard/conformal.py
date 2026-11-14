"""Stage 4: DP-SCP (shift-aware DP split conformal prediction) and uncertainty baselines U1-U4.

Everything here works on saved logits (CPU only).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import log_softmax, softmax
from scipy.stats import chi2, norm

from triguard import dp


# ---------------------------------------------------------------- DP temperature scaling
def dp_temperature(logits: np.ndarray, y: np.ndarray, eps: float, delta: float, rng: np.random.Generator,
                   steps: int = 200, q: float | None = None, clip: float = 1.0, lr: float = 0.05):
    """Fit one temperature (parameter: log T) with DP-SGD. Returns (T, (sigma, q, steps))."""
    n = len(y)
    if n == 0:
        return 1.0, (0.0, 0.0, 0)
    q = q or min(1.0, 256 / n)
    sigma = dp.calibrate_sigma(eps, delta, q, steps, "rdp") if math.isfinite(eps) else 0.0
    logT, m, v = 0.0, 0.0, 0.0
    for t in range(1, steps + 1):
        idx = np.nonzero(rng.random(n) < q)[0] if sigma > 0 else rng.choice(n, min(n, 256), replace=False)
        T = math.exp(logT)
        z = logits[idx] / T
        p = softmax(z, 1)
        # d NLL / d logT = sum_c (p_c - 1[c=y]) * (-z_c)
        oh = np.eye(logits.shape[1])[y[idx]]
        g = ((p - oh) * (-z)).sum(1)
        g = np.clip(g, -clip, clip).sum()
        if sigma > 0:
            g = (g + rng.normal(0, sigma * clip)) / (q * n)
        else:
            g = g / max(1, len(idx))
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        logT -= lr * (m / (1 - 0.9**t)) / (math.sqrt(v / (1 - 0.999**t)) + 1e-8)
        logT = float(np.clip(logT, -3, 3))
    return math.exp(logT), (sigma, q, steps)


# ---------------------------------------------------------------- APS score
def aps_scores(probs: np.ndarray, rng: np.random.Generator, labels: np.ndarray | None = None) -> np.ndarray:
    """Randomised APS: mass of classes ranked above y plus U * p_y.

    labels given -> score of the true label [n]; otherwise scores of all labels [n, C]."""
    n, C = probs.shape
    u = rng.random(n)
    order = np.argsort(-probs, 1)
    sorted_p = np.take_along_axis(probs, order, 1)
    cum_above = np.cumsum(sorted_p, 1) - sorted_p  # mass strictly above, in sorted order
    above = np.empty_like(probs)
    np.put_along_axis(above, order, cum_above, 1)
    all_scores = above + u[:, None] * probs
    if labels is None:
        return np.clip(all_scores, 0, 1)
    return np.clip(all_scores[np.arange(n), labels], 0, 1)


# ---------------------------------------------------------------- DP-SCP
@dataclass
class DPSCPCfg:
    alpha: float = 0.10
    beta: float = 0.05
    M: int = 100  # histogram bins
    eps_label: float = 0.3
    eps_temp: float = 0.2
    eps_cal: float = 2.5
    shift_check: bool = True
    shift_bins: int = 10
    shift_level: float = 0.01
    mondrian: bool = True  # class-conditional
    slack: bool = True  # gamma correction


def _hist(scores, M):
    return np.histogram(scores, bins=M, range=(0, 1))[0].astype(float)


def _threshold(cum_noisy: np.ndarray, target: float, M: int) -> float:
    edges = np.linspace(0, 1, M + 1)[1:]
    ok = np.nonzero(cum_noisy >= target)[0]
    return float(edges[ok[0]]) if len(ok) else 1.0 + 1e-9  # full set


class DPSCP:
    """Algorithm 3. Sites release noisy per-class histograms + noisy label counts; server sums."""

    def __init__(self, cfg: DPSCPCfg, num_classes: int, deltas: dict[int, float]):
        self.cfg, self.C = cfg, num_classes
        self.deltas = deltas
        self.sigma_h = {k: self._sigma(cfg.eps_cal, d) for k, d in deltas.items()}
        self.sigma_c = {k: self._sigma(cfg.eps_label, d) for k, d in deltas.items()}

    @staticmethod
    def _sigma(eps, delta):
        # Gaussian mechanism, L2-sensitivity 1 (one patient changes one bin of one class by 1)
        return dp.gaussian_sigma_for_eps(eps, delta) if math.isfinite(eps) else 0.0

    def fit(self, cal: dict[int, tuple[np.ndarray, np.ndarray]], rng: np.random.Generator):
        """cal[site] = (true-label APS scores, labels)."""
        cfg, C, M = self.cfg, self.C, self.cfg.M
        K = len(cal)
        self.site_h, self.site_n = {}, {}
        H = np.zeros((C, M))
        N = np.zeros(C)
        for k, (s, y) in cal.items():
            h = np.stack([_hist(s[y == c], M) for c in range(C)])
            cnt = np.bincount(y, minlength=C).astype(float)
            h_noisy = h + rng.normal(0, self.sigma_h[k], h.shape) if self.sigma_h[k] > 0 else h
            n_noisy = cnt + rng.normal(0, self.sigma_c[k], C) if self.sigma_c[k] > 0 else cnt
            self.site_h[k], self.site_n[k] = h_noisy, n_noisy
            H += h_noisy
            N += n_noisy
        sh2 = sum(self.sigma_h[k] ** 2 for k in cal)
        sc2 = sum(self.sigma_c[k] ** 2 for k in cal)
        if not cfg.mondrian:  # marginal (U3 naive DP histogram CP)
            H = H.sum(0, keepdims=True)
            N = N.sum(keepdims=True)
            sh2, sc2 = sh2 * C, sc2 * C
        self.gamma = self._gamma(sh2, sc2, M if cfg.slack else 0)
        self.qhat = np.array([_threshold(np.cumsum(H[c]), math.ceil((1 - cfg.alpha) * (max(N[c], 0) + 1)) + self.gamma, M)
                              for c in range(H.shape[0])])
        # ---- shift check: per site, per class, coarse two-sample test of noisy histograms
        self.local_q = {}
        self.shift_flags = {}
        if cfg.shift_check and cfg.mondrian and K > 1:
            for k in cal:
                flags, lq = [], self.qhat.copy()
                for c in range(C):
                    rej = self._shift_test(self.site_h[k][c], H[c] - self.site_h[k][c], self.sigma_h[k] ** 2,
                                           sh2 - self.sigma_h[k] ** 2)
                    flags.append(bool(rej))
                    if rej:
                        g_loc = self._gamma(self.sigma_h[k] ** 2, self.sigma_c[k] ** 2, M if cfg.slack else 0)
                        lq[c] = _threshold(np.cumsum(self.site_h[k][c]),
                                           math.ceil((1 - cfg.alpha) * (max(self.site_n[k][c], 0) + 1)) + g_loc, M)
                self.shift_flags[k] = flags
                self.local_q[k] = lq
        return self

    def _gamma(self, sh2, sc2, M):
        if not self.cfg.slack or (sh2 == 0 and sc2 == 0):
            return 0.0
        b = self.cfg.beta
        # noise of any cumulative sum of <= M noisy bins summed over sites: sd <= sqrt(sh2 * M)
        g_hist = math.sqrt(sh2 * M) * norm.ppf(1 - b / (2 * M))
        # noise of the summed noisy label count enters the target rank
        g_cnt = (1 - self.cfg.alpha) * math.sqrt(sc2) * norm.ppf(1 - b / 2)
        return g_hist + g_cnt

    def _shift_test(self, h_site, h_rest, var_site, var_rest):
        B = self.cfg.shift_bins
        a = h_site.reshape(B, -1).sum(1)
        r = h_rest.reshape(B, -1).sum(1)
        w = self.cfg.M // B
        na, nr = a.sum(), r.sum()
        if na < 20 or nr < 20:
            return False
        p = np.clip((a + r) / (na + nr), 1e-6, None)
        ea, er = na * p, nr * p
        # variance = multinomial + DP noise (w bins summed)
        stat = (((a - ea) ** 2) / (ea + w * var_site) + ((r - er) ** 2) / (er + w * var_rest)).sum()
        return stat > chi2.ppf(1 - self.cfg.shift_level / self.C, B - 1)

    def thresholds(self, site: int) -> np.ndarray:
        return self.local_q.get(site, self.qhat) if self.cfg.mondrian else np.full(self.C, self.qhat[0])

    def predict_sets(self, all_scores: np.ndarray, site: int) -> np.ndarray:
        return all_scores <= self.thresholds(site)[None, :]

    def ledger(self, temp_hist: dict[int, tuple]) -> dict:
        """Per-site calibration epsilon: label counts + temperature + histograms composed in RDP."""
        out = {}
        for k, d in self.deltas.items():
            hist = [(self.sigma_c[k], 1.0, 1), (self.sigma_h[k], 1.0, 1)]
            if k in temp_hist:
                hist.append(temp_hist[k])
            out[k] = dp.rdp_compose_epsilon(hist, d) if all(h[0] > 0 for h in hist if h[2] > 0) else float("inf")
        return out


# ---------------------------------------------------------------- baselines
def split_conformal_q(scores: np.ndarray, alpha: float, weights: np.ndarray | None = None) -> float:
    n = len(scores)
    if n == 0:
        return 1.0 + 1e-9
    if weights is None:
        k = math.ceil((1 - alpha) * (n + 1))
        return float(np.sort(scores)[k - 1]) if k <= n else 1.0 + 1e-9
    # weighted quantile with test-point mass at +inf (Tibshirani et al. 2019)
    o = np.argsort(scores)
    s, w = scores[o], weights[o]
    w = np.append(w, w.mean())
    cw = np.cumsum(w) / w.sum()
    i = np.searchsorted(cw, 1 - alpha)
    return float(s[i]) if i < len(s) else 1.0 + 1e-9


def u1_local(cal_s, alpha):
    return {k: split_conformal_q(s, alpha) for k, (s, y) in cal_s.items()}


def u2_fcp(cal_s, alpha):
    allsc = np.concatenate([s for s, _ in cal_s.values()])
    q = split_conformal_q(allsc, alpha)
    return {k: q for k in cal_s}


def u4_label_shift(cal_s, alpha, C):
    """Plassier et al. 2023 style: per target site, re-weight pooled scores by pi_target(y)/pi_pool(y)."""
    ys = np.concatenate([y for _, y in cal_s.values()])
    sc = np.concatenate([s for s, _ in cal_s.values()])
    pool = np.bincount(ys, minlength=C) / len(ys)
    out = {}
    for k, (s, y) in cal_s.items():
        tgt = np.bincount(y, minlength=C) / max(1, len(y))
        w = tgt[ys] / np.clip(pool[ys], 1e-9, None)
        out[k] = split_conformal_q(sc, alpha, w)
    return out


def coverage_stats(sets: np.ndarray, y: np.ndarray, C: int) -> dict:
    cov = sets[np.arange(len(y)), y]
    size = sets.sum(1)
    per_class = {int(c): float(cov[y == c].mean()) for c in range(C) if (y == c).any()}
    return {
        "coverage": float(cov.mean()),
        "set_size": float(size.mean()),
        "singleton": float((size == 1).mean()),
        "empty": float((size == 0).mean()),
        "per_class": per_class,
        "n": int(len(y)),
    }


def probs_from_logits(logits, T=1.0):
    return np.exp(log_softmax(logits / T, 1))
