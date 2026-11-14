"""PAD-UFES-20 smartphone images, mapped to the ISIC 2019 label space (external-device test only)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from PIL import Image

from triguard.data.common import color_constancy, resize_short
from triguard.paths import DATA

NAME = "pad_ufes20"
NUM_CLASSES = 8  # ISIC 2019 label space
MAP = {"MEL": 0, "NEV": 1, "BCC": 2, "ACK": 3, "SEK": 4, "SCC": 7}
D = DATA / "pad_ufes20"
PRE = D / "preprocessed"
N_RECAL = 200


def _find(img_id):
    hits = list((D / "images").rglob(img_id))
    return hits[0] if hits else None


def prepare() -> None:
    PRE.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(D / "metadata.csv")
    index = {p.name: p for p in (D / "images").rglob("*.png")}
    for img_id in meta.img_id:
        out = PRE / img_id.replace(".png", ".jpg")
        if out.exists() or img_id not in index:
            continue
        img = resize_short(Image.open(index[img_id]).convert("RGB"), 224)
        Image.fromarray(color_constancy(np.array(img))).save(out, quality=95)


@lru_cache(1)
def build_table() -> pd.DataFrame:
    meta = pd.read_csv(D / "metadata.csv")
    meta = meta[meta.diagnostic.isin(MAP)]
    rng = np.random.default_rng(0)
    pats = meta.patient_id.unique()
    rng.shuffle(pats)
    # optional private recalibration: ~200 images from a patient-disjoint subset
    recal, n = set(), 0
    for p in pats:
        if n >= N_RECAL:
            break
        recal.add(p)
        n += (meta.patient_id == p).sum()
    return pd.DataFrame({
        "uid": meta.img_id.str.replace(".png", "", regex=False).values,
        "site": 10,
        "split": np.where(meta.patient_id.isin(recal), "calib", "test"),
        "y": meta.diagnostic.map(MAP).values,
        "sex": meta.gender.str.lower().values,
        "age": meta.age.values,
        "group": meta.patient_id.values,
    })


def load_image(row) -> Image.Image:
    return Image.open(PRE / f"{row.uid}.jpg").convert("RGB")
