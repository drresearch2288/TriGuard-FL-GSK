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
    M: int = 12  # histogram bins (public grid, dense near 1 where the thresholds lie; see _edges)
    eps_label: float = 0.3
    eps_temp: float = 0.2
    eps_cal: float = 2.5
    shift_check: bool = True
    shift_bins: int = 4
    shift_level: float = 0.01
    mondrian: bool = True  # class-conditional
    slack: bool = True  # gamma correction


# roughly geometric in (1 - s): APS scores of the true label pile up near 1, where the thresholds lie
PUBLIC_EDGES_12 = np.array([0, .5, .7, .8, .9, .95, .97, .98, .99, .995, .998, .999, 1.0])


def _edges(M):
    """Data-independent bin edges (public, so they cost no privacy)."""
    return PUBLIC_EDGES_12 if M == 12 else np.linspace(0, 1, M + 1)


def _hist(scores, M):
    return np.histogram(scores, bins=_edges(M))[0].astype(float)


FULL = 1.0 + 1e-9  # threshold that includes every label


def _tail_threshold(h: np.ndarray, n: float, alpha: float, var_bin: float, var_n: float, zh: float, zc: float) -> float:
    """Smallest bin edge t such that, w.h.p. over the DP noise, at most N - ceil((1-a)(N+1)) calibration
    scores lie above t (=> t >= the conformal order statistic, so split-CP validity holds).

    Works from the top of the histogram: the bound on the count *above* t sums only the few bins in the
    upper tail, and the noisy count N enters scaled by ~alpha instead of (1 - alpha)."""
    M = len(h)
    n_lo = math.floor(n - zc * math.sqrt(var_n))  # lower confidence bound on the true count
    if n_lo < 1:
        return FULL
    r = n_lo - math.ceil((1 - alpha) * (n_lo + 1))  # allowed misses; non-decreasing in N
    if r < 0:
        return FULL
    edges = _edges(M)[1:]
    tail = np.concatenate([np.cumsum(h[::-1])[::-1][1:], [0.0]])  # tail[j] = sum(h[j+1:])
    nbins = np.arange(M - 1, -1, -1)
    ok = np.nonzero(tail + zh * np.sqrt(nbins * var_bin) <= r)[0]
    return float(edges[ok[0]]) if len(ok) and ok[0] < M - 1 else FULL


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

    def fit(self, cal: dict[int, tuple[np.ndarray, np.ndarray]], rng: np.random.Generator,
            released_counts: dict | None = None):
        """released_counts[site]: noisy label counts already released during training (reused, not re-drawn)."""
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
            if released_counts is not None and k in released_counts:
                n_noisy = np.asarray(released_counts[k], float)
            else:
                n_noisy = cnt + rng.normal(0, self.sigma_c[k], C) if self.sigma_c[k] > 0 else cnt
            self.site_h[k], self.site_n[k] = h_noisy, n_noisy
            H += h_noisy
            N += n_noisy
        sh2 = sum(self.sigma_h[k] ** 2 for k in cal)
        sc2 = sum(self.sigma_c[k] ** 2 for k in cal)
        a = cfg.alpha
        # ---- Mondrian cells, chosen from released (noisy) counts only => post-processing, no extra privacy cost.
        # A class gets its own cell when its rank budget a*N clearly exceeds the DP slack; the remaining rare
        # classes share one cell (clustered conformal, Ding et al. 2023); if even that cell is too small, they
        # fall back to the all-class (marginal) cell.
        if cfg.mondrian:
            zh0, zc0 = (norm.ppf(1 - cfg.beta / (2 * M)), norm.ppf(1 - cfg.beta / 2)) if cfg.slack else (0.0, 0.0)

            def big_enough(cls):
                m = len(cls)
                slack = zh0 * math.sqrt(max(1, M // 2) * m * sh2) + a * zc0 * math.sqrt(m * sc2)
                return a * N[cls].sum() >= 2 * slack

            own = [[c] for c in range(C) if big_enough([c])]
            rare = [c for c in range(C) if not big_enough([c])]
            cells = own + ([rare] if rare and big_enough(rare) else [])
            leftover = rare if rare and not big_enough(rare) else []
            if leftover:
                cells.append(list(range(C)))  # marginal cell; used only by the leftover classes
        else:
            cells, leftover = [list(range(C))], []
        self.cells = cells
        nc = len(cells)
        zh = norm.ppf(1 - cfg.beta / (2 * M)) if cfg.slack else 0.0  # union over the M candidate thresholds
        zc = norm.ppf(1 - cfg.beta / (2 * nc)) if cfg.slack else 0.0  # union over cells
        self.cell_of = np.zeros(C, int)
        for i, cl in enumerate(cells):
            if leftover and i == nc - 1:  # the marginal cell serves only the leftover classes
                continue
            self.cell_of[cl] = i
        if leftover:
            self.cell_of[leftover] = nc - 1

        def cell_q(Hc, Nc, m, vh, vc):
            return _tail_threshold(Hc, Nc, a, m * vh, m * vc, zh, zc)

        self.cell_q = np.array([cell_q(H[cl].sum(0), N[cl].sum(), len(cl), sh2, sc2) for cl in cells])
        self.qhat = self.cell_q[self.cell_of]
        # reported slack (counts) for the reference tail of M/2 bins in a single-class cell
        self.gamma = zh * math.sqrt(max(1, M // 2) * sh2) + zc * a * math.sqrt(sc2)
        # ---- shift check: per site, per cell, coarse two-sample test of noisy histograms
        self.local_q = {}
        self.shift_flags = {}
        if cfg.shift_check and cfg.mondrian and K > 1:
            for k in cal:
                flags, lq = [], self.cell_q.copy()
                for i, cl in enumerate(cells):
                    m = len(cl)
                    hs = self.site_h[k][cl].sum(0)
                    rej = self._shift_test(hs, H[cl].sum(0) - hs, m * self.sigma_h[k] ** 2,
                                           m * (sh2 - self.sigma_h[k] ** 2))
                    flags.append(bool(rej))
                    if rej:
                        lq[i] = cell_q(hs, self.site_n[k][cl].sum(), m, self.sigma_h[k] ** 2, self.sigma_c[k] ** 2)
                self.shift_flags[k] = flags
                self.local_q[k] = lq[self.cell_of]
        return self

    def _shift_test(self, h_site, h_rest, var_site, var_rest):
        B = self.cfg.shift_bins
        groups = np.array_split(np.arange(len(h_site)), B)
        a = np.array([h_site[g].sum() for g in groups])
        r = np.array([h_rest[g].sum() for g in groups])
        w = max(len(g) for g in groups)
        na, nr = a.sum(), r.sum()
        if na < 20 or nr < 20:
            return False
        p = np.clip((a + r) / (na + nr), 1e-6, None)
        ea, er = na * p, nr * p
        # variance = multinomial + DP noise (w bins summed)
        stat = (((a - ea) ** 2) / (ea + w * var_site) + ((r - er) ** 2) / (er + w * var_rest)).sum()
        return stat > chi2.ppf(1 - self.cfg.shift_level / self.C, B - 1)

    def thresholds(self, site: int) -> np.ndarray:
        return self.local_q.get(site, self.qhat)

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


# ---------------------------------------------------------------- published DP federated baselines
def exp_mech_rank(n: int, target: int, eps: float, rng: np.random.Generator) -> int:
    """Exponential mechanism on ranks 1..n with utility -|i - target| (sensitivity 1)."""
    u = -np.abs(np.arange(1, n + 1) - target).astype(float)
    logp = u * eps / 2.0
    logp -= logp.max()
    p = np.exp(logp)
    return int(rng.choice(n, p=p / p.sum())) + 1


def exp_quantile(scores: np.ndarray, alpha: float, eps: float, beta: float, rng: np.random.Generator) -> float:
    """DP quantile by the exponential mechanism (Lu et al. 2023, appendix C.2). The target rank moves up by the tail
    bound 2 log(1/beta) / eps of the mechanism, so the conformal rank is reached with probability 1 - beta."""
    n = len(scores)
    if n == 0:
        return FULL
    if not math.isfinite(eps):
        return split_conformal_q(scores, alpha)
    target = math.ceil((1 - alpha) * (n + 1)) + math.ceil(2 * math.log(1 / beta) / eps)
    if target > n:
        return FULL
    return float(np.sort(scores)[exp_mech_rank(n, target, eps, rng) - 1])


def u5_exp_fcp(cal_s, alpha, eps, rng, beta=0.05) -> float:
    """One pooled threshold. Needs an exact distributed quantile, so a trusted aggregator (stronger trust than DP-SCP)."""
    return exp_quantile(np.concatenate([s for s, _ in cal_s.values()]), alpha, eps, beta, rng)


def u6_exp_mondrian(cal_s, alpha, eps, C, rng, beta=0.05) -> np.ndarray:
    """Per-class thresholds from the exponential mechanism. One record changes at most two class cells (a swap of its
    label), so each cell gets eps / 2. Cells are not merged: a class too rare for the slack gets the full label set."""
    ys = np.concatenate([y for _, y in cal_s.values()])
    sc = np.concatenate([s for s, _ in cal_s.values()])
    return np.array([exp_quantile(sc[ys == c], alpha, eps / 2, beta / C, rng) for c in range(C)])


def dp_fedcp_q(sc, ys, pi_target, pi_pool, alpha, C, sigma, K, rng, T=100, lr=10.0, w_max=20.0) -> np.ndarray:
    """Per-label thresholds of DP-FedCP (Plassier et al. 2023, Algorithms 1-2), label-shift weights included.

    Simplifications, for a fair but cheap baseline: full participation, one local step per round and a plain pinball
    subgradient (the paper's Moreau regulariser -> 0), run on -log(1 - score) because APS scores pile up just below 1
    and steps in raw score units do not converge there (public transform, no privacy cost). `sigma` is the noise multiplier of one site's summed gradient;
    summing K independent site noises gives std sqrt(K) * sigma * w_max on the global sum. Weights are capped at the
    public w_max so that sensitivity is bounded. As in the paper, the quantile has no slack for the DP noise."""
    w = np.minimum(pi_target[ys] / np.clip(pi_pool[ys], 1e-9, None), w_max)
    o = np.argsort(sc)
    s, cw = -np.log(np.clip(1 - sc[o], 1e-6, None)), np.cumsum(w[o])
    top = float(-math.log(1e-6))
    W = cw[-1]
    wy = np.minimum(pi_target / np.clip(pi_pool, 1e-9, None), w_max)  # test-point mass of query label y
    std = math.sqrt(K) * sigma * w_max
    q = np.zeros(C)
    for y in range(C):
        level = (1 - alpha) * (W + wy[y])  # weighted conformal: test point at +inf
        if level > W:
            q[y] = FULL
            continue
        t, avg = -math.log(alpha), 0.0
        for i in range(1, T + 1):
            F = cw[min(np.searchsorted(s, t, side="right"), len(s)) - 1] if t >= s[0] else 0.0
            g = (F + (rng.normal(0, std) if std else 0.0) - level) / W
            t = float(np.clip(t - lr / math.sqrt(i) * g, 0.0, top))
            if i > T // 2:
                avg += t / (T - T // 2)
        q[y] = 1.0 - math.exp(-avg)
    return q


def u7_dp_fedcp(cal_s, alpha, eps, deltas, C, rng, counts=None, T=100) -> dict:
    """{target site: per-label thresholds}. Every site releases T steps for each label and target site, composed in RDP."""
    ys = np.concatenate([y for _, y in cal_s.values()])
    sc = np.concatenate([s for s, _ in cal_s.values()])
    K = len(cal_s)
    if counts is None:
        counts = {k: np.bincount(y, minlength=C).astype(float) for k, (_, y) in cal_s.items()}
    M = {k: np.maximum(1.0, np.asarray(counts[k], float)) for k in cal_s}
    pool = sum(M.values())
    pool = pool / pool.sum()
    delta = min(deltas.values())
    sigma = dp.calibrate_sigma(eps, delta, 1.0, T * C * K, "rdp") if math.isfinite(eps) else 0.0
    return {k: dp_fedcp_q(sc, ys, M[k] / M[k].sum(), pool, alpha, C, sigma, K, rng, T) for k in cal_s}


def label_thresholds_sets(all_scores: np.ndarray, q) -> np.ndarray:
    return all_scores <= np.asarray(q)[None, :]


def at_equal_size(curve: list[dict], size: float, key: str) -> float | None:
    """Value of `key` where a method's (set size -> metric) curve crosses `size`; None if outside the swept range."""
    pts = sorted(curve, key=lambda r: r["set_size"])
    x, y = np.array([r["set_size"] for r in pts]), np.array([r[key] for r in pts])
    if size < x[0] or size > x[-1]:
        return None
    return float(np.interp(size, x, y))


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
