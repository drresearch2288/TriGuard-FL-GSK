import json
import random
import time
from pathlib import Path

import numpy as np
import torch


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Logger:
    def __init__(self, path: Path | None):
        self.path = path
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))

    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=conv))
    tmp.replace(path)


def load_json(path: Path):
    return json.loads(Path(path).read_text())
