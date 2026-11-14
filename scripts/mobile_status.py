"""Write a phone-friendly status snapshot (results/STATUS_mobile.md) from the local dashboard API."""
import json
import time
import urllib.request

from triguard.paths import ROOT

d = json.load(urllib.request.urlopen("http://127.0.0.1:8765/api", timeout=20))
q, h, c = d["queue"], d["hw"], d["queue"]["counts"]
stamp = time.strftime("%a %d %b %H:%M")
alive = "🟢 running" if q["queue_alive"] else "🔴 STOPPED"
L = [f"# TriGuard-FL status — {stamp}", "", f"**Queue:** {alive}", "",
     "| | |", "|---|---|",
     f"| Jobs done | **{c['done']} / {q['total']}** |",
     f"| Running on laptop | {c['running']} |",
     f"| On Kaggle | {c.get('remote', 0)} |",
     f"| Pending | {c['pending']} |",
     f"| Failed | {c['failed']} |",
     f"| GPU work left (laptop) | ~{q['gpu_hours_left']:.0f} h |",
     f"| Disk free | {q['disk_free']:.0f} GB |", "",
     "## Hardware", "", "| | |", "|---|---|",
     f"| GPU | {h.get('gpu_util') or 0:.0f}% · **{h.get('gpu_temp') or 0:.0f} °C** · {h.get('gpu_clock') or 0:.0f} MHz |",
     f"| CPU | {h.get('cpu') or 0:.0f}% · {h.get('cpu_temp') or 0:.0f} °C |",
     f"| RAM | {h.get('ram') or 0:.0f}% |", "", "## Running now", ""]
for r in q["running"]:
    p = f"{100 * r['progress']:.0f}%" if r["progress"] is not None else "—"
    L.append(f"- **{r['id']}** ({r['resource']}) — {p}")
L += ["", "## Recently finished", ""] + [f"- {r['id']} — {r['hours']:.2f} h ({r['end']})" for r in q["recent"][:6]]
(ROOT / "results" / "STATUS_mobile.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
