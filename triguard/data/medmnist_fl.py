"""MedMNIST+ (224 x 224) split into 20 Dirichlet(beta = 0.3) clients.

Per subset we keep at most 6,000 train(+calib) images and 2,000 test images (disk / compute budget,
matching the plan's 0.15 GPU-h per run ratio). Test images are distributed over clients in proportion to
each client's share of the class, so personalised models and per-client coverage can be evaluated.
"""
from __future__ import annotations

import zipfile
from functools import lru_cache

import numpy as np
import pandas as pd
from numpy.lib import format as npf
from PIL import Image

from triguard.data.common import dirichlet_partition
from triguard.paths import DATA

D = DATA / "medmnist"
MAX_TRAIN, MAX_TEST, CLIENTS, BETA = 6000, 2000, 20, 0.3


def _stream_rows(zf: zipfile.ZipFile, member: str, keep: np.ndarray) -> np.ndarray:
    """Read only rows `keep` (sorted) of an .npy inside a (possibly compressed) npz, streaming."""
    with zf.open(member) as f:
        version = npf.read_magic(f)
        hdr = npf.read_array_header_1_0 if version == (1, 0) else npf.read_array_header_2_0
        shape, fortran, dtype = hdr(f)
        row = int(np.prod(shape[1:])) * dtype.itemsize
        out = np.empty((len(keep), *shape[1:]), dtype)
        want = set(keep.tolist())
        j = 0
        for i in range(shape[0]):
            buf = f.read(row)
            if i in want:
                out[j] = np.frombuffer(buf, dtype).reshape(shape[1:])
                j += 1
                if j == len(keep):
                    break
    return out


def shrink_npz(flag: str, src, dst, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    with zipfile.ZipFile(src) as zf:
        z = np.load(src, mmap_mode=None)
        ytr, yte = z["train_labels"].reshape(-1), z["test_labels"].reshape(-1)
        keep_tr = np.sort(rng.choice(len(ytr), min(MAX_TRAIN, len(ytr)), replace=False))
        keep_te = np.sort(rng.choice(len(yte), min(MAX_TEST, len(yte)), replace=False))
        xtr = _stream_rows(zf, "train_images.npy", keep_tr)
        xte = _stream_rows(zf, "test_images.npy", keep_te)
    np.savez(dst, train_images=xtr, train_labels=ytr[keep_tr], test_images=xte, test_labels=yte[keep_te])


class _Flag:
    def __init__(self, flag):
        self.NAME = flag
        self.flag = flag
        self._z = None
        self._arr = {}

    def prepare(self):
        pass

    def _open(self):
        return np.load(D / f"{self.flag}_sub.npz", mmap_mode="r")

    @property
    def z(self):
        if self._z is None:
            self._z = self._open()
        return self._z

    def array(self, part: str) -> np.ndarray:
        """Image array of one split, read once per process. Each DataLoader worker must not share the parent's open
        npz: forked workers share one file offset, and concurrent reads fail with BadZipFile (seen on Kaggle/Linux).
        A fresh handle is used here, and `preload` lets forked workers inherit the arrays copy-on-write."""
        if part not in self._arr:
            with self._open() as z:
                self._arr[part] = z[part]
        return self._arr[part]

    def preload(self):
        for part in ("train_images", "test_images"):
            self.array(part)

    @property
    def NUM_CLASSES(self):
        return int(max(self.z["train_labels"].max(), self.z["test_labels"].max())) + 1

    @lru_cache(1)
    def build_table(self) -> pd.DataFrame:
        ytr = self.z["train_labels"].reshape(-1).astype(int)
        yte = self.z["test_labels"].reshape(-1).astype(int)
        owner = dirichlet_partition(ytr, CLIENTS, BETA, seed=0, min_size=20)
        rng = np.random.default_rng(1)
        split = np.where(rng.random(len(ytr)) < 0.2, "calib", "train")
        # test samples follow each client's class distribution
        C = max(ytr.max(), yte.max()) + 1
        share = np.zeros((CLIENTS, C))
        for k, c in zip(owner, ytr):
            share[k, c] += 1
        share = share / np.clip(share.sum(0, keepdims=True), 1, None)
        towner = np.array([rng.choice(CLIENTS, p=share[:, c] if share[:, c].sum() > 0 else None) for c in yte])
        tr = pd.DataFrame({"uid": [f"tr{i}" for i in range(len(ytr))], "site": owner, "split": split, "y": ytr})
        te = pd.DataFrame({"uid": [f"te{i}" for i in range(len(yte))], "site": towner, "split": "test", "y": yte})
        return pd.concat([tr, te]).reset_index(drop=True)

    def load_image(self, row) -> Image.Image:
        part, i = ("train_images", int(row.uid[2:])) if row.uid.startswith("tr") else ("test_images", int(row.uid[2:]))
        a = self.array(part)[i]
        img = Image.fromarray(a)
        return img.convert("RGB")


_cache = {}


def for_flag(flag):
    if flag not in _cache:
        _cache[flag] = _Flag(flag)
    return _cache[flag]
