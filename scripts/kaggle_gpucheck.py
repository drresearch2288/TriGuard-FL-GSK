"""Push a tiny private Kaggle notebook that reports GPU type, disk, RAM and internet (credential sanity check)."""
import json
import subprocess
from pathlib import Path

from triguard.remote import kaggle, kaggle_user

d = Path(__file__).resolve().parents[1] / "build" / "remote" / "gpucheck"
d.mkdir(parents=True, exist_ok=True)
(d / "check.py").write_text('''import subprocess, platform, torch, urllib.request
print("python", platform.python_version(), "torch", torch.__version__)
print("cuda", torch.cuda.is_available(), "n_gpu", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
print(subprocess.run("df -h /kaggle/working /kaggle/tmp /tmp; free -g; nproc", shell=True, capture_output=True, text=True).stdout)
print("internet", urllib.request.urlopen("https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_GroundTruth.csv").status)
''', encoding="utf-8")
meta = {"id": f"{kaggle_user()}/triguard-gpucheck", "title": "triguard-gpucheck", "code_file": "check.py",
        "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "machine_shape": "NvidiaTeslaT4"}
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
r = kaggle("kernels", "push", "-p", str(d))
print(r.stdout, r.stderr)
