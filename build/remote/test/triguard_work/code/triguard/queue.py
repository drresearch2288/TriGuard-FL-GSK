"""Priority job queue for the whole study.

    python -m triguard.queue run       # start / resume (safe to re-run after a reboot)
    python -m triguard.queue status    # print + write runs/STATUS.md
    python -m triguard.queue stop      # finish running jobs, then exit
    python -m triguard.queue retry ID  # clear a failed job

Slots: 1 GPU job, 2 CPU jobs, 1 network job at a time. A job starts when all deps are DONE.
State lives in runs/<job id>/ as marker files, so the queue survives crashes and reboots.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta

from triguard.paths import DATA, ROOT, RUNS

SLOTS = {"gpu": 1, "cpu": 2, "net": 1}
MAX_FAILS = 2
PY = sys.executable


def jobs() -> dict:
    return json.loads((RUNS / "jobs.json").read_text())


def jdir(jid):
    return RUNS / jid


def state(jid) -> str:
    d = jdir(jid)
    if (d / "DONE").exists():
        return "done"
    if (d / "RUNNING").exists():
        return "running"
    if (d / "REMOTE").exists():  # claimed by a Kaggle / Colab worker (triguard.remote)
        return "remote"
    if fails(jid) >= MAX_FAILS:
        return "failed"
    return "pending"


def fails(jid) -> int:
    f = jdir(jid) / "FAILS"
    return int(f.read_text()) if f.exists() else 0


def pid_alive(pid: int) -> bool:
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    except Exception:  # noqa: BLE001
        return False


def free_gb() -> float:
    return shutil.disk_usage(DATA).free / 1e9


def ready(J, st, j) -> bool:
    if st[j["id"]] != "pending":
        return False
    for d in j["deps"]:
        if d not in J:
            continue  # dependency not part of the plan
        if st[d] != "done":
            return False
    need = j["args"].get("disk_gb", 0)
    return free_gb() > need + 5


def launch(j):
    d = jdir(j["id"])
    d.mkdir(parents=True, exist_ok=True)
    log = open(d / "stdout.log", "a", encoding="utf-8")
    log.write(f"\n===== start {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    log.flush()
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
    p = subprocess.Popen([PY, "-m", "triguard.jobrun", j["id"]], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                         env=env, creationflags=flags)
    (d / "RUNNING").write_text(json.dumps({"pid": p.pid, "start": time.time()}))
    return p


def run():
    J = jobs()
    procs: dict[str, subprocess.Popen] = {}
    # recover from crash / reboot: RUNNING markers with dead processes -> back to pending
    for jid in J:
        r = jdir(jid) / "RUNNING"
        if r.exists():
            pid = json.loads(r.read_text())["pid"]
            if not pid_alive(pid):
                r.unlink()
    (RUNS / "STOP").unlink(missing_ok=True)
    print(f"queue started {datetime.now():%c}; {len(J)} jobs", flush=True)
    last_status = 0
    while True:
        J = jobs()  # allow live edits of the plan
        # reap
        for jid, p in list(procs.items()):
            rc = p.poll()
            if rc is None:
                continue
            d = jdir(jid)
            info = json.loads((d / "RUNNING").read_text()) if (d / "RUNNING").exists() else {"start": time.time()}
            (d / "RUNNING").unlink(missing_ok=True)
            dur = (time.time() - info["start"]) / 3600
            with open(RUNS / "history.jsonl", "a") as f:
                f.write(json.dumps({"id": jid, "rc": rc, "hours": dur, "end": time.time(),
                                    "est_h": J.get(jid, {}).get("est_h", 0), "resource": J.get(jid, {}).get("resource")}) + "\n")
            if rc == 0:
                (d / "DONE").write_text(datetime.now().isoformat())
            else:
                (d / "FAILS").write_text(str(fails(jid) + 1))
            del procs[jid]
        st = {jid: state(jid) for jid in J}
        stopping = (RUNS / "STOP").exists()
        if not stopping:
            for res, n in SLOTS.items():
                busy = sum(1 for jid in procs if J.get(jid, {}).get("resource") == res)
                if busy >= n:
                    continue
                cand = sorted((j for j in J.values() if j["resource"] == res and ready(J, st, j)),
                              key=lambda j: (j["priority"], j["id"]))
                for j in cand[: n - busy]:
                    procs[j["id"]] = launch(j)
                    st[j["id"]] = "running"
        if time.time() - last_status > 60:
            write_status(J)
            last_status = time.time()
        if not procs and (stopping or all(s in ("done", "failed") for s in st.values()) or
                          not any(ready(J, st, j) for j in J.values())):
            if stopping or all(s in ("done", "failed") for s in st.values()):
                write_status(J)
                print("queue finished", flush=True)
                return
        time.sleep(15)


def write_status(J=None):
    J = J or jobs()
    st = {jid: state(jid) for jid in J}
    hist = []
    if (RUNS / "history.jsonl").exists():
        hist = [json.loads(l) for l in (RUNS / "history.jsonl").read_text().splitlines() if l.strip()]
    ok = [h for h in hist if h["rc"] == 0 and h.get("resource") == "gpu" and h.get("est_h", 0) > 0.2]
    ratio = (sum(h["hours"] for h in ok) / sum(h["est_h"] for h in ok)) if ok else 1.0
    rem_gpu = sum(j["est_h"] for jid, j in J.items() if j["resource"] == "gpu" and st[jid] not in ("done", "remote")) * ratio
    counts = {s: sum(1 for v in st.values() if v == s) for s in ("done", "running", "pending", "failed", "remote")}
    lines = [f"# TriGuard-FL queue status — {datetime.now():%Y-%m-%d %H:%M}", "",
             f"**Jobs:** {counts['done']} done · {counts['running']} running · {counts['pending']} pending · "
             f"{counts['remote']} on Kaggle/Colab · {counts['failed']} failed (of {len(J)})", "",
             f"**GPU work left:** ~{rem_gpu:.0f} h (estimate × measured speed ratio {ratio:.2f}) → "
             f"finish ≈ {(datetime.now() + timedelta(hours=rem_gpu)):%a %d %b %H:%M}",
             f"**Free disk:** {free_gb():.0f} GB", "", "## Running"]
    for jid, s in st.items():
        if s == "running":
            r = json.loads((jdir(jid) / "RUNNING").read_text())
            tail = ""
            for fn in ("train.log", "stdout.log"):
                f = jdir(jid) / fn
                if f.exists():
                    ls = [l for l in f.read_text(errors="ignore").splitlines() if l.strip()]
                    tail = ls[-1][:160] if ls else ""
                    break
            lines.append(f"- `{jid}` ({J[jid]['resource']}, {(time.time()-r['start'])/3600:.1f} h): {tail}")
    nxt = sorted((j for jid, j in J.items() if st[jid] == "pending" and j["resource"] == "gpu"),
                 key=lambda j: (j["priority"], j["id"]))[:8]
    lines += ["", "## Next GPU jobs"] + [f"- `{j['id']}` (~{j['est_h']*ratio:.1f} h)" for j in nxt]
    failed = [jid for jid, s in st.items() if s == "failed"]
    if failed:
        lines += ["", "## Failed (see runs/<id>/stdout.log; `python -m triguard.queue retry <id>`)"]
        lines += [f"- `{jid}`" for jid in failed]
    lines += ["", "## Recently finished"] + [f"- `{h['id']}` rc={h['rc']} {h['hours']:.2f} h"
                                             for h in hist[-10:][::-1]]
    (RUNS / "STATUS.md").write_text("\n".join(lines), encoding="utf-8")
    return "\n".join(lines)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "run":
        run()
    elif cmd == "status":
        print(write_status())
    elif cmd == "stop":
        (RUNS / "STOP").write_text("stop")
        print("stop requested: running jobs will finish, nothing new starts")
    elif cmd == "retry":
        for jid in sys.argv[2:]:
            for f in ("FAILS", "DONE", "RUNNING"):
                (jdir(jid) / f).unlink(missing_ok=True)
            print("reset", jid)
