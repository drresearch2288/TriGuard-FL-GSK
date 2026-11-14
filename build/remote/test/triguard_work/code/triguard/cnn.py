"""B3: DP-FedAvg full fine-tuning of an ImageNet ResNet-18 with GroupNorm (no BatchNorm under DP).

Uses the same rows / splits as the feature cache index so results are directly comparable.
"""
from __future__ import annotations

import math
import time
import types

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from opacus.grad_sample import GradSampleModule
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader, Dataset

from triguard import data as datasets
from triguard import dp
from triguard.cache import Cache
from triguard.metrics import balanced_accuracy
from triguard.paths import RUNS
from triguard.utils import Logger, device, save_json, seed_all

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class _ChunkSampler:
    chunks: list = []

    def __iter__(self):
        return iter(list(self.chunks))

    def __len__(self):
        return len(self.chunks)


class Imgs(Dataset):
    def __init__(self, name, df, tf):
        self.name, self.df, self.tf = name, df.reset_index(drop=True), tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        return self.tf(datasets.get(self.name).load_image(r)), int(r.y)


def make_model(num_classes):
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    m = ModuleValidator.fix(m)  # BatchNorm -> GroupNorm
    for mod in m.modules():  # in-place ops break Opacus' backward hooks
        if isinstance(mod, torch.nn.ReLU):
            mod.inplace = False
        if isinstance(mod, torchvision.models.resnet.BasicBlock):
            mod.forward = types.MethodType(_basic_forward, mod)
    return m


def _basic_forward(self, x):
    out = self.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    identity = self.downsample(x) if self.downsample is not None else x
    return self.relu(out + identity)


def run(jid: str, dataset: str, eps: float, seed: int, rounds: int = 50, lr: float | None = None, clip: float | None = None,
        logical_batch: int = 1024, phys_batch: int = 16, max_rate: float = 0.25, workers: int = 4):
    from triguard.jobrun import tuned

    best = tuned()
    lr = lr or 0.05
    clip = clip or best.get("clip", 1.0)
    out = RUNS / jid
    log = Logger(out / "train.log")
    seed_all(seed)
    dev = device()
    eps = float(eps)
    size = 96 if dataset == "camelyon17" else 224
    cache = Cache(dataset, "dinov2_vits14")
    df = cache.df
    mod = datasets.get(dataset)
    C = int(df.y.max()) + 1
    tf_tr = T.Compose([T.RandomResizedCrop(size, scale=(0.7, 1.0)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                       T.RandomRotation(20), T.ColorJitter(0.1, 0.1, 0.1), T.ToTensor(), T.Normalize(MEAN, STD)])
    tf_te = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(), T.Normalize(MEAN, STD)])
    tr = df[df.split == "train"]
    site_ids = sorted(tr.site.unique())
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=dev).manual_seed(seed + 7)
    glob = make_model(C).to(dev)
    sites = []
    for k in site_ids:
        sub = tr[tr.site == k]
        n = len(sub)
        q = dp.sample_rate(n, logical_batch, max_rate)
        spr = max(1, math.ceil(1 / q))
        steps = rounds * spr
        delta = dp.site_delta(n)
        sigma = dp.calibrate_sigma(eps, delta, q, steps) if math.isfinite(eps) else 0.0
        sites.append(dict(k=int(k), df=sub, n=n, q=q, spr=spr, steps=steps, delta=delta, sigma=sigma, done=0))
        log(f"site {k}: n={n} q={q:.3f} sigma={sigma:.3f} steps={steps}")
    t0 = time.time()
    gsm = GradSampleModule(make_model(C).to(dev))
    params = [p for p in gsm.parameters() if p.requires_grad]
    # one persistent loader over all train rows; a mutable batch sampler feeds it each site-round's chunks
    tr = tr.reset_index(drop=True)
    ds_all = Imgs(dataset, tr, tf_tr)
    sampler = _ChunkSampler()
    loader = DataLoader(ds_all, batch_sampler=sampler, num_workers=workers, persistent_workers=workers > 0,
                        prefetch_factor=4 if workers else None)
    for s in sites:
        s["pos"] = np.nonzero(tr.site.values == s["k"])[0]
    for t in range(rounds):
        new_states, ns = [], []
        for s in sites:
            gsm._module.load_state_dict(glob.state_dict())
            opt = torch.optim.SGD(params, lr=lr, momentum=0.0)  # DP full fine-tune: plain SGD
            steps_idx = []
            for _ in range(s["spr"]):
                if s["sigma"] > 0:
                    steps_idx.append(np.nonzero(rng.random(s["n"]) < s["q"])[0])
                else:
                    steps_idx.append(rng.choice(s["n"], int(round(s["q"] * s["n"])), replace=False))
            chunks, owner = [], []
            for si, idx in enumerate(steps_idx):
                for c0 in range(0, len(idx), phys_batch):
                    chunks.append(s["pos"][idx[c0 : c0 + phys_batch]].tolist())
                    owner.append(si)
            sampler.chunks = chunks
            it = iter(loader) if chunks else iter(())
            ci = 0
            for si, idx in enumerate(steps_idx):
                acc = [torch.zeros_like(p) for p in params]
                while ci < len(owner) and owner[ci] == si:
                    x, y = next(it)
                    ci += 1
                    if True:
                        x, y = x.to(dev), y.to(dev)
                        loss = F.cross_entropy(gsm(x), y, reduction="sum")
                        loss.backward()
                        if s["sigma"] > 0:
                            norms = torch.sqrt(sum(p.grad_sample.reshape(len(y), -1).pow(2).sum(1) for p in params))
                            f = (clip / (norms + 1e-6)).clamp(max=1.0)
                            for a, p in zip(acc, params):
                                a += torch.einsum("b,b...->...", f, p.grad_sample)
                        else:
                            for a, p in zip(acc, params):
                                a += p.grad
                        for p in params:
                            p.grad_sample = None
                            p.grad = None
                B = s["q"] * s["n"] if s["sigma"] > 0 else max(1, len(idx))
                for a, p in zip(acc, params):
                    noise = torch.randn(a.shape, device=dev, generator=gen) * s["sigma"] * clip if s["sigma"] > 0 else 0
                    p.grad = (a + noise) / B
                opt.step()
                opt.zero_grad(set_to_none=True)
                s["done"] += 1
            new_states.append({k: v.detach().clone() for k, v in gsm._module.state_dict().items()})
            ns.append(s["n"])
        w = np.array(ns, float) / sum(ns)
        sd = glob.state_dict()
        for k in sd:
            if sd[k].dtype.is_floating_point:
                sd[k] = sum(float(wi) * st[k] for wi, st in zip(w, new_states))
        glob.load_state_dict(sd)
        if (t + 1) % 10 == 0 or t == rounds - 1:
            log(f"round {t+1}/{rounds} {time.time()-t0:.0f}s")
    # ---- predictions
    glob.eval()
    preds = {}
    with torch.no_grad():
        for split in ("test", "calib", "val"):
            for site_id, sub in df[df.split == split].groupby("site"):
                dl = DataLoader(Imgs(dataset, sub, tf_te), batch_size=64, num_workers=workers)
                lg = torch.cat([glob(x.to(dev)).float().cpu() for x, _ in dl]).numpy()
                preds[f"{split}_{int(site_id)}_logits"] = lg
                preds[f"{split}_{int(site_id)}_rows"] = sub.cache_row.values
    np.savez_compressed(out / "preds.npz", **preds)
    bacc = {int(k.split("_")[1]): balanced_accuracy(df.set_index("cache_row").loc[preds[k.replace("logits", "rows")], "y"].values,
                                                    preds[k].argmax(1)) for k in preds if k.startswith("test_") and k.endswith("logits")}
    ledger = [{"site": s["k"], "n": s["n"], "q": s["q"], "B": s["q"] * s["n"], "sigma": s["sigma"], "steps": s["steps"],
               "steps_done": s["done"], "delta": s["delta"], "clip": clip,
               "eps_train": dp.spent_epsilon([(s["sigma"], s["q"], s["done"])], s["delta"]) if s["sigma"] > 0 else math.inf}
              for s in sites]
    save_json({"cfg": {"dataset": dataset, "backbone": "dinov2_vits14", "method": "b3_resnet", "eps": eps, "seed": seed,
                       "rounds": rounds, "lr": lr, "clip": clip},
               "bacc_per_site": bacc, "bacc_mean": float(np.mean(list(bacc.values()))),
               "bacc_worst": float(np.min(list(bacc.values()))), "ledger": ledger, "seconds": time.time() - t0,
               "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
               "upload_mb_per_round": sum(p.numel() for p in glob.parameters()) * 4 / 1e6},
              out / "train_result.json")
    log(f"done mean BA {np.mean(list(bacc.values())):.4f}")
