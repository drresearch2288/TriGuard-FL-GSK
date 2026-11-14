# TriGuard-FL

Differentially private federated LoRA on a frozen DINOv2 ViT-S/14, with shift-aware private conformal prediction
(DP-SCP), a per-patient privacy ledger and an empirical privacy audit (LiRA, one-run canary, gradient inversion).
Datasets: Fed-ISIC2019 (main), Camelyon17-WILDS, MedMNIST+ (224 px), PAD-UFES-20, Fed-Heart-Disease.

This repository holds the code, every finished run's results (`results/`), and the reports (`reports/`).
It contains **no data and no feature caches** (ISIC 2019 is CC BY-NC); `scripts/download_all.py` fetches them
from the original sources.

## Layout

| Path | What |
|---|---|
| `triguard/` | Library: data loaders, backbone and LoRA model, DP-SGD and accounting, federated training, conformal methods, audits, job queue, Kaggle offloading, report builder |
| `triguard/experiments.py` | Defines every job of the study (datasets, methods, ε, seeds) and writes `runs/jobs.json` |
| `triguard/queue.py` | Local priority queue that runs all jobs (resumable after crash or reboot) |
| `triguard/remote.py` | Sends groups of queued jobs to free Kaggle T4 GPUs and imports their results |
| `scripts/download_all.py` | Downloads every dataset and backbone (resumable) |
| `scripts/make_reports.py` | Builds the three Word reports from `results/` |
| `results/tables`, `results/figures`, `results/RESULTS.md` | All measured tables and figures |
| `results/runs/<job id>/` | Per-run results: metrics JSON, predictions, trained adapter + head (`params.pt`, Git LFS), logs |
| `reports/` | Implementation report, supplementary, deviations and additional runs |
| `RUNBOOK.md` | Operator notes and the log of every fix and rerun |

## Setup (once)

Windows (PowerShell), Python 3.11, NVIDIA GPU with CUDA 12.4 drivers:

```powershell
git lfs install
git clone <this repository> TriGuard-FL; cd TriGuard-FL
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

Linux / macOS / Git Bash: `bash scripts/setup.sh`.

The setup creates `.venv`, installs the pinned packages (`requirements.txt`, PyTorch 2.6.0 + cu124) and clones
FLamby at the pinned commit into `third_party/FLamby`.

Credentials: Camelyon17 and the Kaggle offloading use the Kaggle API. Put your token in
`~/.kaggle/kaggle.json` (or `~/.kaggle/access_token` plus the environment variable `KAGGLE_USERNAME`).
Never commit it.

## Reproduce

All commands run from the repository root with `PYTHONPATH` set to it
(PowerShell: `$env:PYTHONPATH = (Get-Location).Path`; bash: `export PYTHONPATH=$PWD`).
`python` below means `.venv\Scripts\python.exe` (Windows) or `.venv/bin/python`.

### 1. Everything on one PC

```bash
python -m triguard.experiments      # write runs/jobs.json (all jobs, seed 0)
python -m triguard.queue run        # downloads, caches, training, evaluation, audits, tables; resumable
python -m triguard.queue status     # progress and ETA (also runs/STATUS.md)
python -m triguard.dashboard        # optional live dashboard at http://127.0.0.1:8765
```

The queue runs downloads first, then the hyper-parameter search on the public ISIC 2020 proxy, then every
method at every ε, CPU evaluation (conformal, ledger, fairness), audits, ablations and the other datasets.
State lives in `runs/<job id>/` as marker files (`DONE`, `RUNNING`, `FAILS`), so `queue run` can be stopped
and restarted at any time; finished jobs are never repeated. `python -m triguard.queue retry <job id>` re-runs one job.

The reported study used about 220 GPU-hours (RTX 3050 laptop plus Kaggle T4s, `results/tables/t20_compute.csv`); on the laptop alone expect several weeks, so use the Kaggle offloading below.

### 2. PC + free Kaggle GPUs (what produced the reported results)

Keep the local queue running, and send job groups to Kaggle (2 × T4 per session, 12 h limit):

```bash
python -m triguard.remote plan lira --hours 10.5           # which jobs would go
python -m triguard.remote push lira --hours 10.5 --name w1 # build the worker, start a private Kaggle kernel, claim jobs
python -m triguard.remote status w1
python -m triguard.remote pull w1                          # after it finishes: import results; unfinished jobs return to the local queue
python -m triguard.remote release w1                       # give claimed jobs back without importing
```

Groups: `main`, `tune`, `lira`, `ablations`, `medmnist`, `camelyon`, `canary`, `b3` (combine with `+`, e.g.
`lira+ablations`). A worker is one self-contained script with the code embedded; on Kaggle it installs the
packages, downloads the data, rebuilds the feature caches and runs the claimed jobs on both GPUs. The local queue
skips claimed jobs (`REMOTE` marker), so the PC and Kaggle never run the same job. Only `pull` after a final
Kaggle status imports anything.

### 3. Tables, figures, reports

```bash
python -m triguard.report           # results/tables/*.csv, results/figures/*.png, results/RESULTS.md
python scripts/export_results.py    # copy each finished run's deliverables into results/runs/
python scripts/make_reports.py      # reports/*.docx (needs python-docx and the plan .docx next to the repo root)
```

### 4. Rerun a single job or add seeds

```bash
python -m triguard.jobrun isic2019__triguard__eps3__s0   # any job id from runs/jobs.json
```

To add seeds, edit `SEEDS` in `triguard/experiments.py`, rerun `python -m triguard.experiments` (finished jobs
are kept) and `python -m triguard.queue run`.

### Tests

```bash
python -m pytest tests
```

## Data sources

Fed-ISIC2019 and Fed-Heart-Disease: FLamby (github.com/owkin/FLamby). ISIC 2019 / 2020: challenge.isic-archive.com.
Camelyon17-WILDS v1.0 (Kaggle mirror `arinaromashkina/wilds-camelyon17-v1`, checked against WILDS row and hospital
counts). MedMNIST+: medmnist.com. PAD-UFES-20: data.mendeley.com/datasets/zr7vgbcyr2. Each dataset keeps its own licence.
