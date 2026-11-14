"""Live local dashboard: queue progress + CPU / GPU utilisation and temperatures.

    python -m triguard.dashboard        -> http://127.0.0.1:8765
Samples hardware every 5 s (keeps the last 2 h in memory); the page refreshes itself every 5 s.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import psutil

from triguard.paths import DATA, RUNS

PORT = 8765
HIST: deque = deque(maxlen=1440)  # 2 h at 5 s
LOCK = threading.Lock()
NO_WIN = 0x08000000


def gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,"
             "clocks.sm,clocks.max.sm,clocks_throttle_reasons.active", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, creationflags=NO_WIN).stdout.strip().split(", ")
        f = lambda i: float(out[i]) if out[i] not in ("[N/A]", "") else None  # noqa: E731
        return {"util": f(0), "mem_used": f(1), "mem_total": f(2), "temp": f(3), "power": f(4), "power_limit": f(5),
                "clock": f(6), "clock_max": f(7), "throttle": out[8].strip()}
    except Exception:  # noqa: BLE001
        return {}


def cpu_temp():
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_PerfFormattedData_Counters_ThermalZoneInformation | "
                              "Measure-Object HighPrecisionTemperature -Maximum).Maximum"],
                             capture_output=True, text=True, timeout=15, creationflags=NO_WIN).stdout.strip()
        return round(float(out) / 10 - 273.15, 1) if out else None
    except Exception:  # noqa: BLE001
        return None


def sampler():
    psutil.cpu_percent(None)
    n = 0
    ct = None
    while True:
        g = gpu()
        if n % 3 == 0:  # thermal-zone query is slow: every 15 s
            ct = cpu_temp()
        m = psutil.virtual_memory()
        s = {"t": time.time(), "cpu": psutil.cpu_percent(None), "cpu_temp": ct, "ram": m.percent,
             "ram_used": m.used / 1e9, "ram_total": m.total / 1e9, **{f"gpu_{k}": v for k, v in g.items()}}
        with LOCK:
            HIST.append(s)
        n += 1
        time.sleep(5)


def tail(path, n=1):
    try:
        lines = [l for l in path.read_text(errors="ignore").splitlines() if l.strip()]
        return lines[-n:]
    except Exception:  # noqa: BLE001
        return []


def queue_state():
    try:
        J = json.loads((RUNS / "jobs.json").read_text())
    except Exception:  # noqa: BLE001
        return {"error": "no jobs.json"}
    st = {}
    for jid in J:
        d = RUNS / jid
        st[jid] = ("done" if (d / "DONE").exists() else "running" if (d / "RUNNING").exists()
                   else "remote" if (d / "REMOTE").exists()
                   else "failed" if (d / "FAILS").exists() and int((d / "FAILS").read_text() or 0) >= 2 else "pending")
    hist = []
    if (RUNS / "history.jsonl").exists():
        hist = [json.loads(l) for l in (RUNS / "history.jsonl").read_text().splitlines() if l.strip()]
    # speed ratio from finished *training* jobs only (caching / downloads would bias it)
    tr = [h for h in hist if h["rc"] == 0 and h.get("resource") == "gpu" and h.get("est_h", 0) > 0.2
          and not h["id"].startswith("cache__")]
    ratio = sum(h["hours"] for h in tr) / sum(h["est_h"] for h in tr) if tr else 1.0
    left = sum(j["est_h"] for k, j in J.items() if j["resource"] == "gpu" and st[k] not in ("done", "remote")) * ratio
    running = []
    for jid, s in st.items():
        if s != "running":
            continue
        d = RUNS / jid
        try:
            started = json.loads((d / "RUNNING").read_text())["start"]
        except Exception:  # noqa: BLE001
            started = time.time()
        last = (tail(d / "train.log") or tail(d / "stdout.log") or [""])[0]
        prog = None
        m = re.findall(r"round (\d+)/(\d+)", " ".join(tail(d / "train.log", 50)))
        if m:
            prog = int(m[-1][0]) / int(m[-1][1])
        m2 = re.search(r"(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?) GB", last)
        if m2:
            prog = float(m2.group(1)) / float(m2.group(2))
        m3 = re.findall(r"(\d+)/(\d+)(?!\d)", last.split("]", 1)[-1])
        if prog is None and m3 and int(m3[-1][1]) > 0:
            prog = int(m3[-1][0]) / int(m3[-1][1])
        running.append({"id": jid, "resource": J[jid]["resource"], "hours": (time.time() - started) / 3600,
                        "est_h": J[jid]["est_h"] * ratio, "progress": prog, "last": last[-180:]})
    groups = {}
    for jid, s in st.items():
        g = jid.split("__")[0] if "__" in jid else jid
        if g in ("pathmnist", "bloodmnist", "organamnist", "octmnist", "pneumoniamnist", "retinamnist"):
            g = "medmnist"
        if jid.startswith("isic2019__a"):
            g = "ablations"
        grp = groups.setdefault(g, {"done": 0, "total": 0})
        grp["total"] += 1
        grp["done"] += s == "done"
    counts = {k: sum(1 for v in st.values() if v == k) for k in ("done", "running", "pending", "failed", "remote")}
    nxt = sorted((j for k, j in J.items() if st[k] == "pending" and j["resource"] == "gpu"),
                 key=lambda j: (j["priority"], j["id"]))[:6]
    return {
        "counts": counts, "total": len(J), "gpu_hours_left": left, "ratio": ratio,
        "eta": (datetime.now() + timedelta(hours=left)).strftime("%a %d %b %H:%M"),
        "running": running, "groups": groups,
        "next": [{"id": j["id"], "h": j["est_h"] * ratio} for j in nxt],
        "failed": [k for k, v in st.items() if v == "failed"],
        "recent": [{"id": h["id"], "ok": h["rc"] == 0, "hours": h["hours"],
                    "end": datetime.fromtimestamp(h["end"]).strftime("%a %H:%M")} for h in hist[-8:][::-1]],
        "queue_alive": any("triguard.queue" in " ".join(p.info["cmdline"] or []) for p in
                           psutil.process_iter(["cmdline"]) if p.info["cmdline"]),
        "disk_free": shutil.disk_usage(DATA).free / 1e9,
    }


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api"):
            with LOCK:
                h = list(HIST)
            self._send(json.dumps({"now": time.time(), "hw": h[-1] if h else {}, "hist": h[-720:],
                                   "queue": queue_state()}).encode(), "application/json")
        else:
            self._send(PAGE.encode(), "text/html; charset=utf-8")


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TriGuard Run Monitor</title>
<style>
:root{--bg:#fcfcfb;--card:#ffffff;--ink:#2b2b29;--muted:#6b6b66;--line:#e4e3de;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#6250d6;
--good:#0e8a4f;--warn:#b26b00;--crit:#c42b2b;--bar:#eef0f3}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#1a1a19;--card:#232321;--ink:#f0efec;--muted:#a9a89f;--line:#3a3a37;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#9085e9;--good:#3fbf7f;--warn:#e0a33a;--crit:#ef6b6b;--bar:#2e2e2b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,sans-serif}
header{padding:16px;display:flex;flex-wrap:wrap;gap:8px 16px;align-items:baseline;border-bottom:1px solid var(--line)}
h1{font-size:18px;margin:0}.sub{color:var(--muted);font-size:13px}
main{padding:16px;display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));max-width:1400px;margin:auto}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.card h2{font-size:13px;margin:0 0 10px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.wide{grid-column:1/-1}.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px}
.tile .v{font-size:26px;font-weight:650;font-variant-numeric:tabular-nums}.tile .l{color:var(--muted);font-size:12px}
.pill{display:inline-flex;gap:6px;align-items:center;padding:2px 8px;border-radius:99px;font-size:12px;border:1px solid var(--line)}
.dot{width:8px;height:8px;border-radius:50%}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:6px 4px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600}.num{text-align:right;font-variant-numeric:tabular-nums}
.prog{height:8px;background:var(--bar);border-radius:4px;overflow:hidden;min-width:80px}.prog>i{display:block;height:100%;background:var(--s1);border-radius:4px}
.mono{font-family:ui-monospace,Consolas,monospace;font-size:12px;word-break:break-all;color:var(--muted)}
svg{width:100%;height:140px;display:block}.legend{display:flex;gap:12px;font-size:12px;color:var(--muted);margin-top:6px;flex-wrap:wrap}
.legend span{display:inline-flex;align-items:center;gap:5px}.sw{width:12px;height:3px;border-radius:2px}
.tip{position:fixed;pointer-events:none;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:6px 8px;font-size:12px;display:none;box-shadow:0 2px 8px rgba(0,0,0,.15);z-index:9}
.warnbox{border-left:4px solid var(--warn);padding:8px 10px;background:var(--bar);border-radius:6px;font-size:13px;margin-top:10px}
</style></head><body>
<header><h1>TriGuard-FL run monitor</h1><span class="sub" id="upd">loading…</span><span id="alive"></span></header>
<main>
<section class="card wide"><h2>Overall progress</h2><div class="tiles" id="tiles"></div>
<div style="margin-top:12px" class="prog"><i id="allbar" style="width:0"></i></div><div class="sub" id="eta" style="margin-top:6px"></div></section>
<section class="card"><h2>GPU</h2><div class="tiles" id="gputiles"></div><div id="thr"></div></section>
<section class="card"><h2>CPU / memory</h2><div class="tiles" id="cputiles"></div></section>
<section class="card wide"><h2>Running now</h2><table><thead><tr><th>Job</th><th>Slot</th><th>Progress</th><th class="num">Elapsed / est.</th></tr></thead><tbody id="running"></tbody></table></section>
<section class="card"><h2>Utilisation, last hour (%)</h2><svg id="cu" viewBox="0 0 600 140" preserveAspectRatio="none"></svg>
<div class="legend"><span><i class="sw" style="background:var(--s1)"></i>GPU</span><span><i class="sw" style="background:var(--s2)"></i>CPU</span></div></section>
<section class="card"><h2>Temperature, last hour (°C)</h2><svg id="ct" viewBox="0 0 600 140" preserveAspectRatio="none"></svg>
<div class="legend"><span><i class="sw" style="background:var(--s4)"></i>GPU</span><span><i class="sw" style="background:var(--s3)"></i>CPU (thermal zone)</span></div></section>
<section class="card"><h2>Experiment groups</h2><table><tbody id="groups"></tbody></table></section>
<section class="card"><h2>Next GPU jobs</h2><table><tbody id="next"></tbody></table><h2 style="margin-top:14px">Recently finished</h2><table><tbody id="recent"></tbody></table></section>
</main><div class="tip" id="tip"></div>
<script>
const $=id=>document.getElementById(id);const f1=v=>v==null?'—':(+v).toFixed(1);const f0=v=>v==null?'—':Math.round(v);
function tile(v,l){return `<div class="tile"><div class="v">${v}</div><div class="l">${l}</div></div>`}
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function lines(svg,hist,series,ymin,ymax){const el=$(svg);const W=600,H=140,P=4;const pts=hist.filter(h=>h.t>Date.now()/1000-3600);
 if(!pts.length){el.innerHTML='';return}const t0=pts[0].t,t1=pts[pts.length-1].t||t0+1;
 const x=t=>P+(W-2*P)*(t-t0)/Math.max(1,t1-t0),y=v=>H-P-(H-2*P)*(v-ymin)/(ymax-ymin);
 let g='';for(const gv of [0.25,0.5,0.75]){const yy=P+(H-2*P)*gv;g+=`<line x1="0" x2="${W}" y1="${yy}" y2="${yy}" stroke="var(--line)" stroke-width="1" vector-effect="non-scaling-stroke"/>`}
 for(const [k,c] of series){let d='';let pen=false;for(const p of pts){const v=p[k];if(v==null){pen=false;continue}d+=(pen?'L':'M')+x(p.t).toFixed(1)+','+y(Math.min(ymax,Math.max(ymin,v))).toFixed(1);pen=true}
  g+=`<path d="${d}" fill="none" stroke="${c}" stroke-width="2" vector-effect="non-scaling-stroke"/>`}
 g+=`<line id="${svg}x" x1="0" x2="0" y1="0" y2="${H}" stroke="var(--muted)" stroke-width="1" vector-effect="non-scaling-stroke" style="display:none"/>`;
 el.innerHTML=g;el.onmousemove=e=>{const r=el.getBoundingClientRect();const tt=t0+(e.clientX-r.left)/r.width*(t1-t0);
  let best=pts[0];for(const p of pts)if(Math.abs(p.t-tt)<Math.abs(best.t-tt))best=p;const cx=$(svg+'x');cx.setAttribute('x1',x(best.t));cx.setAttribute('x2',x(best.t));cx.style.display='';
  const tip=$('tip');tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px';
  tip.innerHTML=new Date(best.t*1000).toLocaleTimeString()+'<br>'+series.map(([k,c,l])=>`<span style="color:${c}">■</span> ${l}: ${f0(best[k])}`).join('<br>')};
 el.onmouseleave=()=>{$('tip').style.display='none';const cx=$(svg+'x');if(cx)cx.style.display='none'}}
async function tick(){let d;try{d=await (await fetch('/api')).json()}catch(e){$('upd').textContent='dashboard server not reachable';return}
 const q=d.queue,h=d.hw||{},c=q.counts||{};const css=getComputedStyle(document.documentElement);const col=n=>css.getPropertyValue(n).trim();
 $('upd').textContent='updated '+new Date().toLocaleTimeString()+' · refreshes every 5 s';
 $('alive').innerHTML=q.queue_alive?`<span class="pill"><i class="dot" style="background:var(--good)"></i>queue running</span>`:`<span class="pill"><i class="dot" style="background:var(--crit)"></i>queue NOT running — start it (see RUNBOOK)</span>`;
 $('tiles').innerHTML=tile(c.done+' / '+q.total,'jobs done')+tile(c.running,'running')+tile(c.pending,'pending')+tile(c.remote||0,'on Kaggle / Colab')+tile(c.failed,'failed')+tile(f0(q.gpu_hours_left)+' h','GPU work left')+tile(f0(q.disk_free)+' GB','disk free');
 $('allbar').style.width=(100*c.done/q.total).toFixed(1)+'%';
 $('eta').textContent=`Estimated finish ${q.eta} if the laptop runs non-stop (speed factor ${(+q.ratio).toFixed(2)} × plan estimate, measured from finished training runs).`;
 const hot=h.gpu_temp>=83;$('gputiles').innerHTML=tile(f0(h.gpu_util)+'%','utilisation')+tile(f0(h.gpu_temp)+' °C','temperature')+tile(f1(h.gpu_mem_used/1024)+' / '+f1(h.gpu_mem_total/1024)+' GB','VRAM')+tile(f0(h.gpu_clock)+' MHz','clock (max '+f0(h.gpu_clock_max)+')')+tile(f1(h.gpu_power)+' W','power draw');
 $('thr').innerHTML=hot?`<div class="warnbox">⚠ GPU is hot (${f0(h.gpu_temp)} °C) and clocks are reduced — better airflow (raise the back, cooling pad) makes every run faster.</div>`:'';
 $('cputiles').innerHTML=tile(f0(h.cpu)+'%','CPU utilisation')+tile(f1(h.cpu_temp)+' °C','CPU thermal zone')+tile(f1(h.ram_used)+' / '+f1(h.ram_total)+' GB','RAM ('+f0(h.ram)+'%)');
 $('running').innerHTML=(q.running||[]).map(r=>`<tr><td><b>${esc(r.id)}</b><div class="mono">${esc(r.last)}</div></td><td>${r.resource}</td>
  <td>${r.progress!=null?`<div class="prog"><i style="width:${(100*r.progress).toFixed(0)}%"></i></div><span class="sub">${(100*r.progress).toFixed(0)}%</span>`:'<span class="sub">—</span>'}</td>
  <td class="num">${f1(r.hours)} h / ${f1(r.est_h)} h</td></tr>`).join('')||'<tr><td colspan="4" class="sub">nothing running</td></tr>';
 $('groups').innerHTML=Object.entries(q.groups||{}).sort().map(([g,v])=>`<tr><td>${esc(g)}</td><td style="width:45%"><div class="prog"><i style="width:${(100*v.done/v.total).toFixed(0)}%"></i></div></td><td class="num">${v.done}/${v.total}</td></tr>`).join('');
 $('next').innerHTML=(q.next||[]).map(n=>`<tr><td class="mono">${esc(n.id)}</td><td class="num">~${f1(n.h)} h</td></tr>`).join('')||'<tr><td class="sub">—</td></tr>';
 $('recent').innerHTML=(q.recent||[]).map(r=>`<tr><td class="mono">${esc(r.id)}</td><td>${r.ok?'✔ ok':'✖ failed'}</td><td class="num">${f1(r.hours)} h</td><td class="num">${r.end}</td></tr>`).join('')+
   ((q.failed||[]).length?`<tr><td colspan="4" style="color:var(--crit)">Failed: ${q.failed.map(esc).join(', ')}</td></tr>`:'');
 lines('cu',d.hist,[['gpu_util',col('--s1'),'GPU %'],['cpu',col('--s2'),'CPU %']],0,100);
 lines('ct',d.hist,[['gpu_temp',col('--s4'),'GPU °C'],['cpu_temp',col('--s3'),'CPU °C']],30,100)}
tick();setInterval(tick,5000);
</script></body></html>"""


if __name__ == "__main__":
    threading.Thread(target=sampler, daemon=True).start()
    print(f"dashboard on http://127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
