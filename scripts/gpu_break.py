"""Insert a one-off GPU break: after the GPU job that is running now finishes, wait N minutes before the next.

    python scripts/gpu_break.py [minutes]
Works on the live queue (it re-reads runs/jobs.json every loop); nothing is restarted.
"""
import json
import sys
import time

from triguard.paths import RUNS

mins = float(sys.argv[1]) if len(sys.argv) > 1 else 15
J = json.loads((RUNS / "jobs.json").read_text())
running = [j for j in J if (RUNS / j / "RUNNING").exists() and J[j]["resource"] == "gpu"]
cid = f"cooldown__{int(time.time())}"
J[cid] = {"id": cid, "kind": "cooldown", "resource": "cpu", "priority": -1.0, "args": {"minutes": mins},
          "deps": running, "est_h": 0.0}
n = 0
for j in J.values():
    if j["resource"] == "gpu" and not (RUNS / j["id"] / "DONE").exists() and j["id"] not in running:
        j["deps"].append(cid)
        n += 1
tmp = RUNS / "jobs.json.tmp"
tmp.write_text(json.dumps(J, indent=1))
tmp.replace(RUNS / "jobs.json")
print(f"{cid}: {mins:.0f} min break after {running or 'now'}; {n} GPU jobs wait for it")
