"""Tiny deterministic image dataset for integration tests of the caching / CNN paths."""
from functools import lru_cache

import numpy as np
import pandas as pd
from PIL import Image

NAME = "dummy"
NUM_CLASSES = 3


def prepare():
    pass


@lru_cache(1)
def build_table():
    rows = []
    for i in range(72):
        rows.append({"uid": f"d{i}", "site": i % 2, "split": ["train", "train", "calib", "test"][i % 4], "y": i % 3})
    return pd.DataFrame(rows)


def load_image(row):
    rng = np.random.default_rng(int(row.uid[1:]))
    return Image.fromarray((rng.random((260, 300, 3)) * 255).astype("uint8"))
