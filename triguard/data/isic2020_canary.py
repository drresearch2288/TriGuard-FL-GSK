"""1,000 ISIC 2020 canaries (random ISIC 2019 labels) for the one-run privacy audit."""
from functools import lru_cache

from triguard.data import isic2020

NAME = "isic2020_canary"
NUM_CLASSES = 8


def prepare():
    pass


@lru_cache(1)
def build_table():
    return isic2020.canary_table(8, seed=0)


load_image = isic2020.load_image
