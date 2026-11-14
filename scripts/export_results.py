"""Copy every finished run's deliverable files into results/runs/<job id>/ for the GitHub repository.

Kept per run: train/eval/audit/transfer/heart result JSON, predictions (preds.npz, sets_alpha010.npz),
trained adapter + head parameters (params.pt), LiRA logits and the training log.
Never exported: datasets, feature caches, checkpoints, train masks, raw inversion images (examples.pt, 1.4 GB; the figure shows them).
    python scripts/export_results.py
"""
import shutil

from triguard.paths import ROOT, RUNS

KEEP = {"train_result.json", "eval_result.json", "audit_result.json", "transfer_result.json", "heart_result.json",
        "best.json", "preds.npz", "sets_alpha010.npz", "params.pt", "lira_logits.npz", "train.log",
        "REMOTE_INFO.json"}
OUT = ROOT / "results" / "runs"
n, size = 0, 0
for d in sorted(RUNS.iterdir()):
    if not d.is_dir() or not (d / "DONE").exists():
        continue
    if d.name.split("__")[0] in ("dl", "prep", "cache", "clean", "cooldown", "eval", "report"):
        continue
    files = [f for f in d.iterdir() if f.name in KEEP]
    if not files:
        continue
    dst = OUT / d.name
    dst.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dst / f.name)
        size += f.stat().st_size
    # the CPU evaluation of a run lives in its own job dir; its outputs are written into the run dir already
    n += 1
shutil.copy2(RUNS / "jobs.json", ROOT / "results" / "jobs.json")
if (RUNS / "history.jsonl").exists():
    shutil.copy2(RUNS / "history.jsonl", ROOT / "results" / "history.jsonl")
print(f"exported {n} runs ({size / 1e6:.1f} MB) to {OUT}")
