"""Fed-ISIC2019 (FLamby split, 6 natural sites)."""
from __future__ import annotations

import io
import zipfile
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np
import pandas as pd
from PIL import Image

from triguard.data.common import color_constancy, group_calib_split, resize_short
from triguard.paths import DATA, FLAMBY_ISIC

NAME = "isic2019"
CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
NUM_CLASSES = 8
SITE_NAMES = ["BCN", "Vidir-MM", "Vidir-Mod", "Rosendahl", "MSK", "Vienna"]
D = DATA / "isic2019"
ZIP = D / "ISIC_2019_Training_Input.zip"
PRE = D / "preprocessed"


def _work(names: list[str]) -> int:
    with zipfile.ZipFile(ZIP) as zf:
        for n in names:
            out = PRE / (n.split("/")[-1])
            if out.exists():
                continue
            img = Image.open(io.BytesIO(zf.read(n))).convert("RGB")
            img = resize_short(img, 224)
            img = Image.fromarray(color_constancy(np.array(img)))
            img.save(out, quality=95)
    return len(names)


def prepare(workers: int = 8) -> None:
    """Resize (shorter side 224) + shades-of-grey colour constancy, streamed from the zip."""
    PRE.mkdir(parents=True, exist_ok=True)
    split = pd.read_csv(FLAMBY_ISIC / "dataset_creation_scripts" / "train_test_split")
    wanted = set(split.image + ".jpg")
    with zipfile.ZipFile(ZIP) as zf:
        names = [n for n in zf.namelist() if n.split("/")[-1] in wanted]
    todo = [n for n in names if not (PRE / n.split("/")[-1]).exists()]
    chunks = [todo[i : i + 200] for i in range(0, len(todo), 200)]
    with ProcessPoolExecutor(workers) as ex:
        done = 0
        for k in ex.map(_work, chunks):
            done += k
            if done % 2000 < 200:
                print(f"preprocessed {done}/{len(todo)}", flush=True)
    n = len(list(PRE.glob("*.jpg")))
    assert n == 23247, f"expected 23247 preprocessed images, found {n}"


@lru_cache(1)
def build_table() -> pd.DataFrame:
    split = pd.read_csv(FLAMBY_ISIC / "dataset_creation_scripts" / "train_test_split")
    meta = pd.read_csv(D / "ISIC_2019_Training_Metadata.csv")
    df = split.merge(meta, on="image", how="left")
    df = pd.DataFrame(
        {
            "uid": df.image,
            "site": df.center.astype(int),
            "split": df.fold.map({"train": "train", "test": "test"}),
            "y": df.target.astype(int),
            "sex": df.sex,
            "age": df.age_approx,
            "group": df.lesion_id,
        }
    )
    df["split"] = group_calib_split(df, "group", frac=0.2, seed=0)
    return df.reset_index(drop=True)


def load_image(row) -> Image.Image:
    return Image.open(PRE / f"{row.uid}.jpg").convert("RGB")


EXPECTED = {  # FLamby README counts (train, test) per site
    0: (9930, 2483), 1: (3163, 791), 2: (2691, 672), 3: (1807, 452), 4: (655, 164), 5: (351, 88),
}
