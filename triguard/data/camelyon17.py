"""Camelyon17-WILDS, official split: train centers 0, 3, 4; validation center 1; unseen test center 2."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd
from PIL import Image

from triguard.data.common import group_calib_split
from triguard.paths import DATA

NAME = "camelyon17"
CLASS_NAMES = ["normal", "tumour"]
NUM_CLASSES = 2
D = DATA / "wilds" / "camelyon17_v1.0"


def prepare() -> None:
    assert (D / "metadata.csv").exists(), "Camelyon17 not downloaded"


@lru_cache(1)
def build_table() -> pd.DataFrame:
    m = pd.read_csv(D / "metadata.csv", index_col=0)
    split = m.center.map({0: "train", 3: "train", 4: "train", 1: "val", 2: "test"})
    df = pd.DataFrame({
        "uid": [f"patches/patient_{p:03d}_node_{n}/patch_patient_{p:03d}_node_{n}_x_{x}_y_{y}.png"
                for p, n, x, y in zip(m.patient, m.node, m.x_coord, m.y_coord)],
        "site": m.center.astype(int).values,
        "split": split.values,
        "y": m.tumor.astype(int).values,
        "group": m.slide.values,
    })
    df["split"] = group_calib_split(df, "group", 0.2, seed=0)  # slide-disjoint calibration split
    return df


def load_image(row) -> Image.Image:
    return Image.open(D / row.uid).convert("RGB")
