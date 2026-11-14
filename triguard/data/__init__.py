"""Dataset registry.

Every dataset module exposes
    build_table() -> pandas.DataFrame with columns
        uid (str), site (int), split ('train'|'calib'|'test'|'val'), y (int),
        and optional meta columns (sex, age, ...)
    load_image(row) -> PIL.Image (RGB)
    NUM_CLASSES, CLASS_NAMES
"""
import importlib

DATASETS = {
    "isic2019": "triguard.data.isic2019",
    "isic2020": "triguard.data.isic2020",
    "isic2020_canary": "triguard.data.isic2020_canary",
    "pad_ufes20": "triguard.data.pad_ufes20",
    "camelyon17": "triguard.data.camelyon17",
    "fed_heart": "triguard.data.fed_heart",
    "dummy": "triguard.data.dummy",
}
for _f in ["pathmnist", "bloodmnist", "organamnist", "octmnist", "pneumoniamnist", "retinamnist"]:
    DATASETS[_f] = "triguard.data.medmnist_fl"


def get(name: str):
    mod = importlib.import_module(DATASETS[name])
    if hasattr(mod, "for_flag"):
        return mod.for_flag(name)
    return mod
