import numpy as np
import pandas as pd
from PIL import Image


def color_constancy(img: np.ndarray, power: int = 6) -> np.ndarray:
    """Shades-of-grey colour constancy (same maths as FLamby's color_constancy.py)."""
    dtype = img.dtype
    x = img.astype("float32")
    rgb = np.power(np.mean(np.power(x, power), (0, 1)), 1 / power)
    rgb = rgb / np.sqrt(np.sum(rgb**2))
    rgb = 1 / (rgb * np.sqrt(3))
    return np.clip(x * rgb, 0, 255).astype(dtype)


def resize_short(img: Image.Image, size: int = 224) -> Image.Image:
    w, h = img.size
    r = size / min(w, h)
    return img.resize((max(size, int(w * r)), max(size, int(h * r))), Image.BILINEAR)


def group_calib_split(df: pd.DataFrame, group_col: str, frac: float = 0.2, seed: int = 0) -> pd.Series:
    """Per site, move `frac` of the *groups* (patients / lesions) of the train fold to 'calib'.

    Groups are disjoint between train and calib, so a patient is in exactly one of them
    (parallel composition in the privacy ledger).
    """
    rng = np.random.default_rng(seed)
    split = df["split"].copy()
    for site, sub in df[df.split == "train"].groupby("site"):
        groups = sub[group_col].fillna(sub["uid"]).unique()
        rng.shuffle(groups)
        n_cal = max(1, int(round(frac * len(groups))))
        cal_groups = set(groups[:n_cal])
        idx = sub.index[sub[group_col].fillna(sub["uid"]).isin(cal_groups)]
        split.loc[idx] = "calib"
    return split


def dirichlet_partition(y: np.ndarray, n_clients: int, beta: float, seed: int = 0, min_size: int = 10) -> np.ndarray:
    """Label-skewed Dirichlet split. Returns client id per sample."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    for _ in range(100):
        owner = np.empty(len(y), dtype=int)
        for c in classes:
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            p = rng.dirichlet(beta * np.ones(n_clients))
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for k, part in enumerate(np.split(idx, cuts)):
                owner[part] = k
        if np.bincount(owner, minlength=n_clients).min() >= min_size:
            return owner
    return owner
