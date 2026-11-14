"""Stage 1: run the frozen backbone once through blocks 1..n_frozen and store fp16 token tensors.

Two memory-mapped arrays per (dataset, backbone):
  clean.npy  [N_all, T, D]      centre-crop view of every image (train/calib/test/val)
  aug.npy    [N_train, K, T, D] K augmented views of every *train-split* image
plus index.csv (the dataset table with row ids into both arrays).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from triguard import data as datasets
from triguard.backbone import load_backbone
from triguard.paths import CACHE
from triguard.utils import device


def cache_dir(dataset: str, backbone: str) -> Path:
    return CACHE / f"{dataset}__{backbone}"


def transforms(mean, std, size=224):
    norm = [T.ToTensor(), T.Normalize(mean, std)]
    clean = T.Compose([T.Resize(size), T.CenterCrop(size), *norm])
    aug = T.Compose(
        [
            T.RandomResizedCrop(size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(20),
            T.ColorJitter(0.1, 0.1, 0.1, 0.0),
            *norm,
        ]
    )
    return clean, aug


class _DS(Dataset):
    """Holds the dataset *name* (modules are not picklable for Windows spawn workers)."""

    def __init__(self, name: str, rows: pd.DataFrame, tf, views: int, seed: int):
        self.name, self.rows, self.tf, self.views, self.seed = name, rows, tf, views, seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows.iloc[i]
        img = datasets.get(self.name).load_image(row)
        out = []
        for v in range(self.views):
            torch.manual_seed(self.seed * 1_000_003 + int(row.cache_row) * 31 + v)  # deterministic augmentation
            out.append(self.tf(img))
        return torch.stack(out), i


def build_index(dataset: str, max_train: int | None = None, max_eval: int | None = None, seed: int = 0):
    """The cache's row table (deterministic given the arguments); build_cache stores it as index.csv."""
    mod = datasets.get(dataset)
    df = mod.build_table().copy()
    rng = np.random.default_rng(seed)
    # optional subsampling (disk / bandwidth limits) -- recorded in meta.json
    if max_train is not None:
        keep = []
        for _, sub in df[df.split.isin(["train", "calib"])].groupby("site"):
            keep.append(sub)
        tr = pd.concat(keep)
        if len(tr) > max_train:
            tr = tr.sample(max_train, random_state=seed)
        df = pd.concat([tr, df[~df.split.isin(["train", "calib"])]])
    if max_eval is not None:
        ev = df[df.split.isin(["test", "val"])]
        parts = [g.sample(min(len(g), max_eval), random_state=seed) for _, g in ev.groupby("split")]
        df = pd.concat([df[df.split.isin(["train", "calib"])], *parts])
    df = df.reset_index(drop=True)
    df["cache_row"] = np.arange(len(df))
    tr_mask = df.split == "train"
    df["aug_row"] = -1
    df.loc[tr_mask, "aug_row"] = np.arange(tr_mask.sum())

    return df, tr_mask


def build_cache(dataset: str, backbone: str = "dinov2_vits14", n_frozen: int = 8, views: int = 4,
                batch: int = 32, workers: int = 4, max_train: int | None = None, max_eval: int | None = None,
                seed: int = 0) -> Path:
    out = cache_dir(dataset, backbone)
    if (out / "DONE").exists():
        print(f"cache exists: {out}")
        return out
    out.mkdir(parents=True, exist_ok=True)
    df, tr_mask = build_index(dataset, max_train, max_eval, seed)
    dev = device()
    bb = load_backbone(backbone, dev)
    clean_tf, aug_tf = transforms(bb.mean, bb.std)
    with torch.no_grad():
        probe = bb.forward_frozen(torch.zeros(1, 3, 224, 224, device=dev), n_frozen)
    Tk, D = probe.shape[1:]
    clean = np.lib.format.open_memmap(out / "clean.npy", "w+", np.float16, (len(df), Tk, D))
    if views > 0 and tr_mask.sum() > 0:
        aug = np.lib.format.open_memmap(out / "aug.npy", "w+", np.float16, (int(tr_mask.sum()), views, Tk, D))
    else:
        views = 0
        np.save(out / "aug.npy", np.zeros((0, 1, Tk, D), np.float16))
        aug = np.zeros((0, 1, Tk, D), np.float16)

    def run(rows, tf, v, arr, rowcol):
        ds_mod = datasets.get(dataset)
        if workers and hasattr(ds_mod, "preload") and os.name != "nt":
            ds_mod.preload()  # forked workers inherit the arrays
        dl = DataLoader(_DS(dataset, rows, tf, v, seed), batch_size=max(1, batch // v), num_workers=workers,
                        persistent_workers=False)
        t0, done = time.time(), 0
        for x, i in dl:
            b = x.shape[0]
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                tok = bb.forward_frozen(x.view(-1, 3, 224, 224).to(dev, non_blocking=True), n_frozen)
            tok = tok.float().view(b, v, Tk, D).cpu().numpy().astype(np.float16)
            rr = rows[rowcol].values[i.numpy()]
            if arr.ndim == 3:
                arr[rr] = tok[:, 0]
            else:
                arr[rr] = tok
            done += b
            if done % (50 * b) < b:
                print(f"  {dataset} {rowcol}: {done}/{len(rows)}  {done / (time.time() - t0):.1f} img/s", flush=True)

    run(df, clean_tf, 1, clean, "cache_row")
    if views > 0:
        run(df[tr_mask], aug_tf, views, aug, "aug_row")
    clean.flush()
    if hasattr(aug, "flush"):
        aug.flush()
    df.to_csv(out / "index.csv", index=False)
    meta = dict(dataset=dataset, backbone=backbone, n_frozen=n_frozen, views=views, tokens=int(Tk), dim=int(D),
                n=len(df), n_train=int(tr_mask.sum()), max_train=max_train, max_eval=max_eval)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "DONE").write_text(time.ctime())
    return out


class Cache:
    def __init__(self, dataset: str, backbone: str = "dinov2_vits14", features: bool = True):
        """features=False reads the index only (evaluation works on saved logits, so it needs no feature files)."""
        self.dir = cache_dir(dataset, backbone)
        assert (self.dir / "DONE").exists(), f"missing feature cache {self.dir}"
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.df = pd.read_csv(self.dir / "index.csv")
        if features:
            self.clean = np.load(self.dir / "clean.npy", mmap_mode="r")
            self.aug = np.load(self.dir / "aug.npy", mmap_mode="r")

    def get_clean(self, rows: np.ndarray) -> torch.Tensor:
        o = np.argsort(rows)
        x = np.empty((len(rows), *self.clean.shape[1:]), np.float16)
        x[o] = self.clean[rows[o]]
        return torch.from_numpy(x)

    def get_aug(self, rows: np.ndarray, k: int) -> torch.Tensor:
        o = np.argsort(rows)
        x = np.empty((len(rows), k, *self.aug.shape[2:]), np.float16)
        x[o] = self.aug[rows[o], :k]
        return torch.from_numpy(x)


def rebuild_index_only(dataset: str, backbone: str = "dinov2_vits14", views: int = 4, max_train: int | None = None,
                       max_eval: int | None = None, seed: int = 0, n_frozen: int = 8) -> Path:
    """Recreate index.csv / meta.json of a cache whose feature files were deleted (evaluation needs only the index)."""
    out = cache_dir(dataset, backbone)
    out.mkdir(parents=True, exist_ok=True)
    df, tr_mask = build_index(dataset, max_train, max_eval, seed)
    df.to_csv(out / "index.csv", index=False)
    meta = dict(dataset=dataset, backbone=backbone, n_frozen=n_frozen, views=views, n=len(df),
                n_train=int(tr_mask.sum()), max_train=max_train, max_eval=max_eval, index_only=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "DONE").write_text(time.ctime() + " (index only)")
    return out


def delete_cache(dataset: str, backbone: str) -> None:
    shutil.rmtree(cache_dir(dataset, backbone), ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--backbone", default="dinov2_vits14")
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-eval", type=int, default=None)
    a = ap.parse_args()
    build_cache(a.dataset, a.backbone, views=a.views, batch=a.batch, workers=a.workers,
                max_train=a.max_train, max_eval=a.max_eval)
