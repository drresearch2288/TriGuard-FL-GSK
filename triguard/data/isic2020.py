"""ISIC 2020 public proxy: (i) federated tuning proxy, (ii) canary pool for the one-run audit.

Only a subset is downloaded (bandwidth): all malignant images + benign images up to 6,600 for the proxy,
and 1,000 further patient-disjoint images as canaries. Near-duplicates of ISIC 2019 are removed with
perceptual hashing. The proxy is split into 6 synthetic sites with Fed-ISIC2019's size proportions.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from PIL import Image

from triguard.data.common import color_constancy, group_calib_split, resize_short
from triguard.paths import DATA

NAME = "isic2020"
CLASS_NAMES = ["benign", "malignant"]
NUM_CLASSES = 2
D = DATA / "isic2020"
PRE = D / "preprocessed"
N_PROXY, N_CANARY = 6600, 1000
SITE_FRAC = np.array([9930, 3163, 2691, 1807, 655, 351], float)
SITE_FRAC /= SITE_FRAC.sum()


def select_subset(gt: pd.DataFrame, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    pats = gt.patient_id.unique()
    rng.shuffle(pats)
    canary_pats, proxy_pats = set(), set()
    n_c = 0
    by_pat = gt.groupby("patient_id").size()
    for p in pats:  # canary patients first (disjoint from the proxy)
        if n_c >= N_CANARY * 1.1:
            break
        canary_pats.add(p)
        n_c += by_pat[p]
    rest = gt[~gt.patient_id.isin(canary_pats)]
    mal = rest[rest.target == 1]
    ben = rest[rest.target == 0].sample(N_PROXY - len(mal), random_state=seed)
    canary = gt[gt.patient_id.isin(canary_pats)].sample(int(N_CANARY * 1.1), random_state=seed)
    sel = pd.concat([mal, ben, canary])
    sel = sel.assign(role=["proxy"] * (len(mal) + len(ben)) + ["canary"] * len(canary))
    D.mkdir(parents=True, exist_ok=True)
    sel.to_csv(D / "subset.csv", index=False)
    return list(sel.image_name)


def prepare() -> None:
    import imagehash

    from triguard.data import isic2019

    PRE.mkdir(parents=True, exist_ok=True)
    sel = pd.read_csv(D / "subset.csv")
    for n in sel.image_name:
        out = PRE / f"{n}.jpg"
        if out.exists():
            continue
        img = Image.open(D / "raw" / f"{n}.jpg").convert("RGB")
        img = resize_short(img, 224)
        Image.fromarray(color_constancy(np.array(img))).save(out, quality=95)
    # ---- near-duplicates of ISIC 2019 (perceptual hash, Hamming distance <= 4)
    def ph(p):
        return int(str(imagehash.phash(Image.open(p))), 16)

    h19 = np.array([ph(p) for p in sorted(isic2019.PRE.glob("*.jpg"))], dtype=np.uint64)
    h20 = np.array([ph(PRE / f"{n}.jpg") for n in sel.image_name], dtype=np.uint64)
    dup = np.zeros(len(h20), bool)
    for i in range(0, len(h20), 256):
        x = h20[i : i + 256, None] ^ h19[None, :]
        bits = np.unpackbits(x.view(np.uint8).reshape(*x.shape, 8), axis=-1).sum(-1)
        dup[i : i + 256] = bits.min(1) <= 4
    sel["dup_isic2019"] = dup
    sel.to_csv(D / "subset.csv", index=False)
    print(f"near-duplicates of ISIC 2019 removed: {dup.sum()}")


@lru_cache(1)
def table_all() -> pd.DataFrame:
    sel = pd.read_csv(D / "subset.csv")
    sel = sel[~sel.get("dup_isic2019", False)]
    return sel


@lru_cache(1)
def build_table() -> pd.DataFrame:
    """Federated tuning proxy (role == proxy)."""
    sel = table_all()
    px = sel[sel.role == "proxy"].copy()
    rng = np.random.default_rng(0)
    pats = px.patient_id.unique()
    rng.shuffle(pats)
    # 20 % of patients -> test; the rest spread over 6 sites with ISIC 2019 proportions
    n_test = int(0.2 * len(pats))
    test_p = set(pats[:n_test])
    tr_p = pats[n_test:]
    cuts = (np.cumsum(SITE_FRAC) * len(tr_p)).astype(int)[:-1]
    site_of = {}
    for s, part in enumerate(np.split(tr_p, cuts)):
        for p in part:
            site_of[p] = s
    test_rows = px[px.patient_id.isin(test_p)]
    # test patients also spread over sites with the same proportions (per-site evaluation)
    tcuts = (np.cumsum(SITE_FRAC) * n_test).astype(int)[:-1]
    for s, part in enumerate(np.split(np.array(list(test_p)), tcuts)):
        for p in part:
            site_of[p] = s
    df = pd.DataFrame({
        "uid": px.image_name.values,
        "site": px.patient_id.map(site_of).astype(int).values,
        "split": np.where(px.patient_id.isin(test_p), "test", "train"),
        "y": px.target.astype(int).values,
        "sex": px.sex.values,
        "age": px.age_approx.values,
        "group": px.patient_id.values,
    })
    df["split"] = group_calib_split(df, "group", 0.2, seed=0)
    return df.reset_index(drop=True)


def canary_table(num_classes: int = 8, seed: int = 0) -> pd.DataFrame:
    """1,000 canaries with labels drawn uniformly from the 8 ISIC 2019 classes."""
    sel = table_all()
    c = sel[sel.role == "canary"].head(N_CANARY).copy()
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"uid": c.image_name.values, "site": 0, "split": "train",
                         "y": rng.integers(0, num_classes, len(c)), "group": c.patient_id.values})


def load_image(row) -> Image.Image:
    return Image.open(PRE / f"{row.uid}.jpg").convert("RGB")
