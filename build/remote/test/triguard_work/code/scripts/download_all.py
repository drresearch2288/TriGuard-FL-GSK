"""Download every dataset and backbone used by TriGuard-FL.

Resumable: each file is downloaded with HTTP Range resume and a .done marker is
written when a dataset is complete, so re-running skips finished items.
Order = priority (primary dataset first, big optional ones last).

Usage:  python scripts/download_all.py [--only isic2019,heart,...]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("TRIGUARD_DATA", ROOT / "data"))
DATA.mkdir(parents=True, exist_ok=True)
(ROOT / "logs").mkdir(exist_ok=True)


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(ROOT / "logs" / "download.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def fetch(url: str, dest: Path, retries: int = 20) -> Path:
    """Resumable download with retries."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if dest.exists():
        return dest
    attempt, last_pos = 0, -1
    while attempt < retries:
        pos0 = tmp.stat().st_size if tmp.exists() else 0
        if pos0 > last_pos:  # progress since the last failure -> flaky line, not a dead link: reset the counter
            attempt, last_pos = 0, pos0
        attempt += 1
        try:
            pos = pos0
            headers = {"Range": f"bytes={pos}-"} if pos else {}
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                if r.status_code == 416:  # already complete
                    break
                r.raise_for_status()
                if pos and r.status_code != 206:  # server ignored range
                    pos = 0
                total = int(r.headers.get("Content-Length", 0)) + pos
                mode = "ab" if pos else "wb"
                last = time.time()
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        pos += len(chunk)
                        if time.time() - last > 60:
                            pct = 100 * pos / total if total else 0
                            log(f"  {dest.name}: {pos/1e9:.2f}/{total/1e9:.2f} GB ({pct:.0f}%)")
                            last = time.time()
            break
        except Exception as e:  # noqa: BLE001
            log(f"  retry {attempt}/{retries} for {dest.name} at {pos0/1e9:.2f} GB: {str(e)[:120]}")
            time.sleep(min(60, 5 * attempt))
    else:
        raise RuntimeError(f"failed to download {url}")
    tmp.rename(dest)
    log(f"  done {dest.name} ({dest.stat().st_size/1e9:.2f} GB)")
    return dest


def done(name: str) -> bool:
    return (DATA / f".{name}.done").exists()


def mark(name: str) -> None:
    (DATA / f".{name}.done").write_text(time.ctime())
    log(f"=== {name} complete")


# ---------------------------------------------------------------- datasets
def isic2019() -> None:
    d = DATA / "isic2019"
    base = "https://isic-archive.s3.amazonaws.com/challenges/2019/"
    fetch(base + "ISIC_2019_Training_Metadata.csv", d / "ISIC_2019_Training_Metadata.csv")
    fetch(base + "ISIC_2019_Training_GroundTruth.csv", d / "ISIC_2019_Training_GroundTruth.csv")
    z = fetch(base + "ISIC_2019_Training_Input.zip", d / "ISIC_2019_Training_Input.zip")
    # Raw images are kept inside the zip; preprocessing streams from it.
    with zipfile.ZipFile(z) as zf:
        assert zf.testzip() is None, "corrupt ISIC 2019 zip"


def heart() -> None:
    d = DATA / "fed_heart"
    base = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
    for c in ["cleveland", "hungarian", "switzerland", "va"]:
        fetch(base + f"processed.{c}.data", d / f"processed.{c}.data")


def pad_ufes() -> None:
    d = DATA / "pad_ufes20"
    api = "https://data.mendeley.com/public-api/datasets/zr7vgbcyr2/files?folder_id={}&version=1"
    files = requests.get(api.format("root"), timeout=60).json()
    files += requests.get(api.format("6839ab16-69b8-4a2f-ba12-a3ccda4c89f5"), timeout=60).json()
    for f in files:
        p = fetch(f["content_details"]["download_url"], d / f["filename"])
        if p.suffix == ".zip":
            with zipfile.ZipFile(p) as zf:
                zf.extractall(d / "images")
            p.unlink()
            p.touch()  # keep an empty marker so fetch() skips it on re-run


def backbones() -> None:
    os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "hf"))
    os.environ.setdefault("TORCH_HOME", str(ROOT / "cache" / "torch"))
    import timm  # noqa: F401
    import torch

    torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True)
    timm.create_model("vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=True)  # A5 ablation
    try:
        import open_clip

        open_clip.create_model_and_transforms(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
    except Exception as e:  # noqa: BLE001
        log(f"  BiomedCLIP download failed (ablation A6 only): {e}")


def medmnist() -> None:
    medmnist_flags(["bloodmnist", "pneumoniamnist", "retinamnist", "organamnist", "octmnist", "pathmnist"])


def camelyon17() -> None:
    from wilds import get_dataset

    get_dataset(dataset="camelyon17", download=True, root_dir=str(DATA / "wilds"))


def isic2020() -> None:
    """Only the subset we use (tuning proxy + canary pool) is pulled out of the 23 GB zip with HTTP ranges."""
    from concurrent.futures import ThreadPoolExecutor

    import pandas as pd
    from remotezip import RemoteZip

    d = DATA / "isic2020"
    raw = d / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    base = "https://isic-archive.s3.amazonaws.com/challenges/2020/"
    gt = fetch(base + "ISIC_2020_Training_GroundTruth.csv", d / "ISIC_2020_Training_GroundTruth.csv")
    sys.path.insert(0, str(ROOT))
    from triguard.data.isic2020 import select_subset

    names = select_subset(pd.read_csv(gt))
    todo = [n for n in names if not (raw / f"{n}.jpg").exists()]
    log(f"  isic2020: {len(names)} images selected, {len(todo)} to fetch")

    def get(chunk):
        for _ in range(5):
            try:
                with RemoteZip(base + "ISIC_2020_Training_JPEG.zip") as z:
                    for n in chunk:
                        if not (raw / f"{n}.jpg").exists():
                            (raw / f"{n}.jpg.part").write_bytes(z.read(f"train/{n}.jpg"))
                            (raw / f"{n}.jpg.part").rename(raw / f"{n}.jpg")
                return len(chunk)
            except Exception as e:  # noqa: BLE001
                log(f"  isic2020 retry: {e}")
                time.sleep(10)
        raise RuntimeError("isic2020 subset download failed")

    chunks = [todo[i : i + 100] for i in range(0, len(todo), 100)]
    with ThreadPoolExecutor(4) as ex:
        got = 0
        for k in ex.map(get, chunks):
            got += k
            if got % 1000 < 100:
                log(f"  isic2020: {got}/{len(todo)}")


def medmnist_flags(flags) -> None:
    """Download a MedMNIST+ 224 subset and immediately shrink it to the rows we use (disk)."""
    sys.path.insert(0, str(ROOT))
    from triguard.data.medmnist_fl import shrink_npz

    import medmnist
    from medmnist import INFO

    d = DATA / "medmnist"
    d.mkdir(exist_ok=True)
    for flag in flags:
        if (d / f"{flag}_sub.npz").exists():
            continue
        url = INFO[flag]["url_224"] if "url_224" in INFO[flag] else None
        if url is None:
            cls = getattr(medmnist, INFO[flag]["python_class"])
            cls(split="test", download=True, size=224, root=str(d))
            src = d / f"{flag}_224.npz"
        else:
            src = fetch(url.split("?")[0] + "?download=1", d / f"{flag}_224.npz")
        shrink_npz(flag, src, d / f"{flag}_sub.npz")
        src.unlink()
        log(f"  medmnist {flag} ok")


STEPS = {
    "isic2019": isic2019,
    "heart": heart,
    "pad_ufes": pad_ufes,
    "backbones": backbones,
    "medmnist": medmnist,
    "camelyon17": camelyon17,
    "isic2020": isic2020,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    names = args.only.split(",") if args.only else list(STEPS)
    failed = []
    for n in names:
        if done(n):
            log(f"skip {n} (already done)")
            continue
        log(f"=== start {n}")
        try:
            STEPS[n]()
            mark(n)
        except Exception as e:  # noqa: BLE001
            log(f"!!! {n} FAILED: {e}")
            failed.append(n)
    free = shutil.disk_usage(DATA).free / 1e9
    log(f"finished; failed={failed}; free disk {free:.1f} GB")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
