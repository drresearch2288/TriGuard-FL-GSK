import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("TRIGUARD_DATA", ROOT / "data"))
CACHE = Path(os.environ.get("TRIGUARD_CACHE", ROOT / "cache" / "features"))
RUNS = Path(os.environ.get("TRIGUARD_RUNS", ROOT / "runs"))
FLAMBY_ISIC = ROOT / "third_party" / "FLamby" / "flamby" / "datasets" / "fed_isic2019"

os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "hf"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "cache" / "torch"))

for p in (DATA, CACHE, RUNS):
    p.mkdir(parents=True, exist_ok=True)
