"""Stage 2 + 3: federated DP training on cached tokens (Algorithms 1 and 2).

One process simulates all sites sequentially on one GPU.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import grad, vmap

from triguard import dp, psg
from triguard.aggregation import Server
from triguard.cache import Cache
from triguard.backbone import load_backbone
from triguard.metrics import balanced_accuracy
from triguard.model import FeatureHead, ModelCfg, TailModel
from triguard.utils import Logger, device, save_json, seed_all


@dataclass
class FLCfg:
    dataset: str = "isic2019"
    backbone: str = "dinov2_vits14"
    n_frozen: int = 8
    method: str = "triguard"
    model: ModelCfg = field(default_factory=ModelCfg)
    linear_probe: bool = False  # B4: head on full-backbone features
    eps: float = 3.0  # per-patient training budget; inf = non-private
    accountant: str = "prv"
    rounds: int = 50
    local_epochs: float = 1.0
    logical_batch: int = 1024
    max_rate: float = 0.25
    phys_batch: int = 16  # samples per per-sample-grad chunk (x views); 16x4 fits 4 GB
    psg: str = "fast"  # fast (hooks + flash attention) | vmap
    amp: bool = True  # bf16 autocast
    views: int = 4  # augmentation multiplicity K
    lr: float = 2e-3
    weight_decay: float = 0.0
    clip: float = 1.0
    agg: str = "re"  # re | fedavg | none
    server_lr: float = 1.0
    server_momentum: float = 0.9
    final_head_epochs: float = 1.0
    centralised: bool = False  # B2: pool all sites into one
    sites: list | None = None  # restrict to these site ids
    train_mask: str | None = None  # .npy bool mask over cache rows (LiRA shadows / canaries)
    extra_train: dict | None = None  # {"dataset", "rows", "labels"}: extra site-0 records (canaries)
    eval_every: int = 5
    checkpoint_every: int = 2  # rounds; resume after shutdown loses at most this many rounds
    seed: int = 0
    out: str = "runs/debug"


class Site:
    def __init__(self, k, rows, ys, cfg: FLCfg):
        self.k, self.rows, self.ys = k, rows, ys
        self.n = len(rows)
        self.q = dp.sample_rate(self.n, cfg.logical_batch, cfg.max_rate)
        self.B = self.q * self.n  # expected batch size
        self.steps_per_round = max(1, math.ceil(cfg.local_epochs / self.q))
        self.head_steps = max(0, math.ceil(cfg.final_head_epochs / self.q)) if cfg.final_head_epochs else 0
        self.total_steps = cfg.rounds * self.steps_per_round + self.head_steps
        self.delta = dp.site_delta(self.n)
        self.sigma = dp.calibrate_sigma(cfg.eps, self.delta, self.q, self.total_steps, cfg.accountant)
        self.history: list[tuple[float, float, int]] = []
        self.params: dict[str, torch.Tensor] = {}
        self.opt = None
        self.noise_var = 0.0

    def public_tuple(self):
        return dict(n=self.n, q=self.q, B=self.B, sigma=self.sigma, steps=self.total_steps, delta=self.delta)


class FLTrainer:
    def __init__(self, cfg: FLCfg):
        self.cfg = cfg
        self.out = Path(cfg.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.log = Logger(self.out / "train.log")
        seed_all(cfg.seed)
        self.dev = device()
        self.cache = Cache(cfg.dataset, cfg.backbone)
        df = self.cache.df
        self.num_classes = int(df.y.max()) + 1
        self.gen = torch.Generator(device=self.dev).manual_seed(cfg.seed + 99)
        self.rng = np.random.default_rng(cfg.seed)
        from concurrent.futures import ThreadPoolExecutor

        self.pool = ThreadPoolExecutor(2)
        bb = load_backbone(cfg.backbone, "cpu")
        if cfg.linear_probe:
            self.tail = TailModel(bb.blocks, bb.norm, cfg.n_frozen, self.num_classes, ModelCfg(lora=False)).to(self.dev)
            self.model = FeatureHead(2 * bb.dim, self.num_classes)
            self._precompute_probe_features()
        else:
            self.model = TailModel(bb.blocks, bb.norm, cfg.n_frozen, self.num_classes, cfg.model).to(self.dev)
            self.model.use_sdpa = cfg.psg == "fast"
        del bb
        # ---- sites
        tr = df[df.split == "train"]
        if cfg.train_mask:
            m = np.load(cfg.train_mask)
            tr = tr[m[tr.cache_row.values]]
        site_ids = sorted(tr.site.unique()) if cfg.sites is None else list(cfg.sites)
        groups = [tr[tr.site == s] for s in site_ids]
        if cfg.centralised:
            import pandas as pd

            groups, site_ids = [pd.concat(groups)], [-1]
        self.extra_cache = None
        if cfg.extra_train:  # canaries appended to the first site; rows >= OFFSET index the extra cache
            ex = cfg.extra_train
            self.extra_cache = Cache(ex["dataset"], cfg.backbone)
            g0 = groups[0]
            import pandas as pd

            groups[0] = pd.concat([g0, pd.DataFrame({"aug_row": np.asarray(ex["rows"]) + self.OFFSET,
                                                      "y": np.asarray(ex["labels"])})])
        self.sites = [Site(s, g.aug_row.values.astype(int), g.y.values.astype(int), cfg) for s, g in zip(site_ids, groups)]
        for s in self.sites:
            self.log(f"site {s.k}: {s.public_tuple()}")

    # ------------------------------------------------------------------ data
    def _precompute_probe_features(self):
        """B4: features of the full frozen backbone for every clean / augmented view."""
        c = self.cache
        p0 = self.tail.init_params(0)

        def feats(get, rows, extra_shape):
            out = []
            for i in range(0, len(rows), 64):
                x = get(rows[i : i + 64]).to(self.dev).float()
                sh = x.shape
                x = x.view(-1, *sh[-2:])
                with torch.no_grad():
                    out.append(self.tail.features(p0, x).view(*sh[:-2], -1).cpu())
            return torch.cat(out)

        self.probe_clean = feats(c.get_clean, np.arange(len(c.df)), None)
        n_aug = c.aug.shape[0]
        self.probe_aug = feats(lambda r: c.get_aug(r, self.cfg.views), np.arange(n_aug), None)

    OFFSET = 10_000_000

    def batch_x_cpu(self, site: Site, idx: np.ndarray) -> torch.Tensor:
        rows = site.rows[idx]
        if self.cfg.linear_probe:
            return self.probe_aug[rows, : self.cfg.views]
        ext = rows >= self.OFFSET
        if not ext.any():
            return self.cache.get_aug(rows, self.cfg.views).pin_memory() if torch.cuda.is_available() else \
                self.cache.get_aug(rows, self.cfg.views)
        x = torch.empty((len(rows), self.cfg.views, *self.cache.aug.shape[2:]), dtype=torch.float16)
        te = torch.from_numpy(ext)
        if (~ext).any():
            x[~te] = self.cache.get_aug(rows[~ext], self.cfg.views)
        x[te] = self.extra_cache.get_aug(rows[ext] - self.OFFSET, self.cfg.views)
        return x

    def batch_x(self, site: Site, idx: np.ndarray) -> torch.Tensor:
        return self.batch_x_cpu(site, idx).to(self.dev).float()

    def clean_x(self, rows: np.ndarray) -> torch.Tensor:
        if self.cfg.linear_probe:
            return self.probe_clean[rows].to(self.dev)
        return self.cache.get_clean(rows).to(self.dev).float()

    # ------------------------------------------------------------------ gradients
    def _loss_one(self, train_p, frozen_p, x, y):
        p = {**frozen_p, **train_p}
        logits = self.model.logits(p, x)  # x: [K, ...] views of one sample
        return F.cross_entropy(logits, y.expand(x.shape[0]))

    def per_sample_grads(self, train_p, frozen_p, X, Y):
        return vmap(grad(self._loss_one), in_dims=(None, None, 0, 0))(train_p, frozen_p, X, Y)

    def local_steps(self, site: Site, n_steps: int, names: list[str], lr_at) -> None:
        cfg = self.cfg
        p = site.params
        train_p = {k: p[k] for k in names}
        frozen_p = {k: v for k, v in p.items() if k not in names}
        private = site.sigma > 0
        for step in range(n_steps):
            if private:
                idx = np.nonzero(self.rng.random(site.n) < site.q)[0]
            else:
                idx = self.rng.choice(site.n, size=max(1, int(round(site.B))), replace=False)
            acc: dict[str, torch.Tensor] = {}
            chunks = [idx[c0 : c0 + cfg.phys_batch] for c0 in range(0, len(idx), cfg.phys_batch)]
            # prefetch the next chunk from the memory-mapped cache while the GPU works on this one
            futs = [self.pool.submit(self.batch_x_cpu, site, c) for c in chunks[:2]]
            for j, ci in enumerate(chunks):
                X = futs[j].result().to(self.dev, non_blocking=True).float()
                if j + 2 < len(chunks):
                    futs.append(self.pool.submit(self.batch_x_cpu, site, chunks[j + 2]))
                Y = torch.as_tensor(site.ys[ci], device=self.dev)
                if private:
                    if cfg.psg == "vmap":
                        g = self.per_sample_grads(train_p, frozen_p, X, Y)
                    else:
                        g = psg.per_sample_grads(self.model, {**frozen_p, **train_p}, names, X, Y, amp=cfg.amp)
                    dp.clip_and_accumulate(g, cfg.clip, acc)
                else:
                    tp = {k: v.detach().requires_grad_(True) for k, v in train_p.items()}
                    dt = psg.amp_dtype()
                    sc = psg.LOSS_SCALE if dt == torch.float16 else 1.0
                    with torch.autocast("cuda", dtype=dt, enabled=cfg.amp and X.is_cuda):
                        logits = self.model.logits({**frozen_p, **tp}, X.flatten(0, 1)).float()
                    loss = F.cross_entropy(logits, Y.repeat_interleave(X.shape[1]), reduction="sum") / X.shape[1]
                    gs = [g / sc for g in torch.autograd.grad(loss * sc, list(tp.values()))]
                    for k, gk in zip(tp, gs):
                        acc[k] = acc[k] + gk if k in acc else gk
            if not acc:  # empty Poisson batch: still a (pure-noise) step
                acc = {k: torch.zeros_like(v) for k, v in train_p.items()}
            if private:
                noisy = dp.add_noise(acc, site.sigma, cfg.clip, site.B, self.gen)
            else:
                noisy = {k: v / max(1, len(idx)) for k, v in acc.items()}
            # ---- AdamW step (manual, keeps state per site)
            lr = lr_at()
            st = site.opt
            st["t"] += 1
            b1, b2, eps_ = 0.9, 0.999, 1e-8
            nv = 0.0
            nd = 0
            for k in names:
                g = noisy[k]
                m = st["m"].setdefault(k, torch.zeros_like(g))
                v = st["v"].setdefault(k, torch.zeros_like(g))
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                mh = m / (1 - b1 ** st["t"])
                vh = v / (1 - b2 ** st["t"])
                denom = vh.sqrt() + eps_
                if cfg.weight_decay:
                    train_p[k].mul_(1 - lr * cfg.weight_decay)
                train_p[k].sub_(lr * mh / denom)
                if private and self.model.is_federated(k):
                    # per-coordinate variance of the DP noise in this update (post-processing of DP state)
                    nv += float(((lr * site.sigma * cfg.clip / site.B) / denom).pow(2).sum())
                    nd += g.numel()
            if nd:
                site.noise_var += nv / nd
            site.steps_done += 1
        for k in names:
            p[k] = train_p[k]

    # ------------------------------------------------------------------ eval
    @torch.no_grad()
    def predict(self, params, rows: np.ndarray, bs: int = 128) -> np.ndarray:
        out = []
        for i in range(0, len(rows), bs):
            x = self.clean_x(rows[i : i + bs])
            out.append(self.model.logits(params, x).float().cpu())
        return torch.cat(out).numpy() if out else np.zeros((0, self.num_classes))

    def logits_for_site(self, site_id: int, rows: np.ndarray) -> np.ndarray:
        """Own model for a training site; the single model if centralised; otherwise (unseen hospital /
        device) the average of all site models' probabilities."""
        trained = {int(s.k): s for s in self.sites}
        if -1 in trained:
            return self.predict(trained[-1].params, rows)
        if site_id in trained:
            return self.predict(trained[site_id].params, rows)
        probs = np.mean([torch.softmax(torch.from_numpy(self.predict(s.params, rows)), 1).numpy() for s in self.sites], 0)
        return np.log(np.clip(probs, 1e-12, None))

    def evaluate(self, split="test"):
        df = self.cache.df
        res = {}
        sub_all = df[df.split == split]
        for site_id, sub in sub_all.groupby("site"):
            lg = self.logits_for_site(int(site_id), sub.cache_row.values)
            res[int(site_id)] = balanced_accuracy(sub.y.values, lg.argmax(1), self.num_classes)
        return res

    # ------------------------------------------------------------------ main loop
    def run(self) -> dict:
        cfg = self.cfg
        t0 = time.time()
        init = {k: v.to(self.dev) for k, v in self.model.init_params(cfg.seed).items()}
        fed_names = [k for k in init if self.model.is_federated(k)]
        for s in self.sites:
            s.params = {k: v.clone().to(self.dev) for k, v in self.model.init_params(cfg.seed + 1000 + int(s.k)).items()}
            for k in fed_names:
                s.params[k] = init[k].clone()
            s.opt = {"t": 0, "m": {}, "v": {}}
            s.steps_done = 0
        agg = cfg.agg if len(self.sites) > 1 else "none"
        server = Server({k: init[k] for k in fed_names}, agg if agg != "none" else "fedavg", cfg.server_lr, cfg.server_momentum)
        names_all = list(init)
        total_steps = max(s.total_steps for s in self.sites)
        curve = []

        def lr_sched(s):
            return lambda: cfg.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, s.steps_done / s.total_steps)))

        # ---- resume from the last round checkpoint (laptop shutdowns): same state, same RNG streams
        ckpt = self.out / "checkpoint.pt"
        start = 0
        if ckpt.exists():
            st = torch.load(ckpt, weights_only=False)
            for s, ss in zip(self.sites, st["sites"]):
                s.params = {k: v.to(self.dev) for k, v in ss["params"].items()}
                s.opt = {"t": ss["opt"]["t"], "m": {k: v.to(self.dev) for k, v in ss["opt"]["m"].items()},
                         "v": {k: v.to(self.dev) for k, v in ss["opt"]["v"].items()}}
                s.steps_done = ss["steps_done"]
            server.g = {k: v.to(self.dev) for k, v in st["server"]["g"].items()}
            server.m = {k: v.to(self.dev) for k, v in st["server"]["m"].items()}
            server.history = st["server"]["history"]
            curve, start = st["curve"], st["round"]
            self.rng.bit_generator.state = st["np_rng"]
            self.gen.set_state(st["torch_gen"].cpu())
            t0 -= st["elapsed"]
            self.log(f"resumed from checkpoint at round {start}")

        def save_ckpt(rnd):
            cpu = lambda d: {k: v.detach().cpu() for k, v in d.items()}  # noqa: E731
            st = {"round": rnd, "curve": curve, "elapsed": time.time() - t0, "np_rng": self.rng.bit_generator.state,
                  "torch_gen": self.gen.get_state(),
                  "server": {"g": cpu(server.g), "m": cpu(server.m), "history": server.history},
                  "sites": [{"params": cpu(s.params), "steps_done": s.steps_done,
                             "opt": {"t": s.opt["t"], "m": cpu(s.opt["m"]), "v": cpu(s.opt["v"])}} for s in self.sites]}
            torch.save(st, ckpt.with_suffix(".tmp"))
            ckpt.with_suffix(".tmp").replace(ckpt)

        for t in range(start, cfg.rounds):
            updates, ns, vs = [], [], []
            for s in self.sites:
                if agg != "none":
                    for k in fed_names:
                        s.params[k] = server.g[k].clone()
                s.noise_var = 0.0
                self.local_steps(s, s.steps_per_round, names_all, lr_sched(s))
                if agg != "none":
                    updates.append({k: s.params[k] - server.g[k] for k in fed_names})
                    ns.append(s.n)
                    vs.append(s.noise_var)
            if agg != "none":
                info = server.step(updates, ns, vs)
                for s in self.sites:
                    for k in fed_names:
                        s.params[k] = server.g[k].clone()
            else:
                info = {}
            if (t + 1) % cfg.eval_every == 0 or t == cfg.rounds - 1:
                ev = self.evaluate("test")
                curve.append({"round": t + 1, "bacc": ev, "mean": float(np.mean(list(ev.values()))), **info,
                              "time": time.time() - t0})
                self.log(f"round {t+1}/{cfg.rounds}  mean BA {curve[-1]['mean']:.4f}  {ev}  w={info.get('weights')}  "
                         f"tau2={info.get('tau2')}  {time.time()-t0:.0f}s")
            if cfg.checkpoint_every and (t + 1) % cfg.checkpoint_every == 0 and t + 1 < cfg.rounds:
                save_ckpt(t + 1)
        # ---- final DP head fine-tune (Algorithm 2, step 3), inside each site's budget
        head_names = [k for k in names_all if k.startswith("head.") and not self.model.is_federated(k)]
        if head_names:
            for s in self.sites:
                if s.head_steps:
                    self.local_steps(s, s.head_steps, head_names, lambda: cfg.lr * 0.1)
        for s in self.sites:
            s.history = [(s.sigma, s.q, s.steps_done)]
        return self.finish(curve, server, time.time() - t0)

    def finish(self, curve, server, secs) -> dict:
        cfg = self.cfg
        df = self.cache.df
        preds = {}
        for split in ("test", "calib", "val"):
            for site_id, sub in df[df.split == split].groupby("site"):
                preds[f"{split}_{int(site_id)}_logits"] = self.logits_for_site(int(site_id), sub.cache_row.values)
                preds[f"{split}_{int(site_id)}_rows"] = sub.cache_row.values
        np.savez_compressed(self.out / "preds.npz", **preds)
        torch.save({int(s.k): {n: v.cpu() for n, v in s.params.items()} for s in self.sites},
                   self.out / "params.pt")
        ledger = []
        for s in self.sites:
            eps_spent = dp.spent_epsilon(s.history, s.delta, cfg.accountant) if s.sigma > 0 else float("inf")
            ledger.append({"site": int(s.k), **s.public_tuple(), "steps_done": s.steps_done, "eps_train": eps_spent,
                           "clip": cfg.clip})
        final = self.evaluate("test")
        val = self.evaluate("val")
        res = {
            "bacc_val": val,
            "cfg": _cfg_dict(cfg),
            "bacc_per_site": final,
            "bacc_mean": float(np.mean(list(final.values()))),
            "bacc_worst": float(np.min(list(final.values()))),
            "curve": curve,
            "ledger": ledger,
            "server": server.history,
            "seconds": secs,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
            "upload_mb_per_round": sum(v.numel() for k, v in server.g.items()) * 4 / 1e6,
        }
        save_json(res, self.out / "train_result.json")
        (self.out / "checkpoint.pt").unlink(missing_ok=True)
        self.log(f"done: mean BA {res['bacc_mean']:.4f} worst {res['bacc_worst']:.4f}  {secs/60:.1f} min  "
                 f"VRAM {res['peak_vram_gb']:.2f} GB")
        return res


def _cfg_dict(cfg):
    d = asdict(cfg)
    return d
