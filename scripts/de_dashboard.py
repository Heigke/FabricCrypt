"""de_dashboard.py — live DE-fit dashboard.

Parses one of /tmp/zNN_run.log every few seconds, plots:
  - Objective vs iteration
  - Latest parameter values
  - (Optional) current best fit overlay on data

Usage:
  python -m scripts.de_dashboard --log /tmp/z37_run.log --port 8765
Then open http://localhost:8765/  in browser. Auto-refreshes.

No external deps beyond what the project already uses (matplotlib, numpy).
"""
from __future__ import annotations
import argparse, base64, io, re, time, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ITER_RE = re.compile(
    r"iter\s+(\d+)\s+obj=([\d.]+)\s+(.+?)\s+\((\d+)s\)"
)

# Map raw-log keys → display labels (for the right panel).
# If a key isn't here, we still display whatever the log uses.

class State:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.iters = []   # list of dicts: {iter, obj, params, t_s}
        self.last_size = 0
        self.lock = threading.Lock()
        self.start = time.time()

    def refresh(self):
        if not self.log_path.exists(): return
        try:
            sz = self.log_path.stat().st_size
            if sz < self.last_size:
                # File was truncated; reset
                self.iters = []
                self.last_size = 0
            with open(self.log_path) as f:
                f.seek(self.last_size)
                new = f.read()
                self.last_size = f.tell()
        except OSError:
            return
        with self.lock:
            for line in new.splitlines():
                m = ITER_RE.search(line)
                if not m: continue
                it = int(m.group(1)); obj = float(m.group(2))
                params_blob = m.group(3); t_s = int(m.group(4))
                # If iter already recorded, replace
                self.iters = [r for r in self.iters if r["iter"] != it]
                self.iters.append({"iter": it, "obj": obj,
                                    "params": params_blob, "t_s": t_s})
            self.iters.sort(key=lambda r: r["iter"])


def render_plot(state: State) -> bytes:
    with state.lock:
        if not state.iters:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "Waiting for first iter…", ha="center",
                     va="center", fontsize=14, transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
        else:
            iters = [r["iter"] for r in state.iters]
            objs = [r["obj"] for r in state.iters]
            ts = [r["t_s"] for r in state.iters]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
            ax1.plot(iters, objs, "o-", color="#27ae60", lw=2, ms=8)
            ax1.set_xlabel("iteration"); ax1.set_ylabel("objective (mean log-RMSE)")
            ax1.set_title(f"DE convergence — best={min(objs):.3f} at iter {iters[objs.index(min(objs))]}")
            ax1.grid(alpha=0.3)
            ax1.axhline(0.5, color="red", ls="--", alpha=0.5,
                          label="publishable (~0.5 dec)")
            ax1.legend()

            # Iteration time per iter
            dts = np.diff([0] + ts)
            ax2.bar(iters, dts, color="#3498db", alpha=0.7)
            ax2.set_xlabel("iteration"); ax2.set_ylabel("seconds / iter")
            ax2.set_title(f"iter time — total {ts[-1]}s = {ts[-1]/60:.1f} min")
            ax2.grid(alpha=0.3)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        return buf.getvalue()


def render_html(state: State) -> str:
    with state.lock:
        latest = state.iters[-1] if state.iters else None
        n = len(state.iters)
        best = min((r["obj"] for r in state.iters), default=None)
        elapsed = state.iters[-1]["t_s"] if state.iters else int(time.time() - state.start)
    rows = ""
    if latest:
        rows = f"""
        <tr><td><b>iter</b></td><td>{latest['iter']}</td></tr>
        <tr><td><b>obj</b></td><td>{latest['obj']:.4f}</td></tr>
        <tr><td><b>best</b></td><td>{best:.4f}</td></tr>
        <tr><td><b>params</b></td><td><code>{latest['params']}</code></td></tr>
        <tr><td><b>elapsed</b></td><td>{elapsed} s ({elapsed/60:.1f} min)</td></tr>
        """
    log_basename = state.log_path.name
    return f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="5">
<title>DE dashboard — {log_basename}</title>
<style>
body{{font-family:-apple-system,sans-serif;max-width:1200px;margin:1em auto;padding:0 1em;background:#fafafa}}
h1{{color:#2c3e50;margin-bottom:0.2em}}
table{{border-collapse:collapse;margin:1em 0}}
td{{padding:0.4em 1em;border-bottom:1px solid #ddd;font-size:0.95em}}
code{{background:#eee;padding:2px 6px;border-radius:3px;font-size:0.85em}}
.meta{{color:#666;font-size:0.9em}}
img{{max-width:100%;background:white;border-radius:4px;box-shadow:0 1px 3px rgba(0,0,0,0.1)}}
</style>
</head><body>
<h1>DE dashboard</h1>
<div class="meta">log: <code>{state.log_path}</code> · auto-refresh 5s · {n} iters logged</div>
<img src="/plot.png?t={time.time():.0f}" alt="convergence plot">
<table>{rows}</table>
</body></html>
"""


def make_handler(state):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a, **kw): pass
        def do_GET(self):
            state.refresh()
            if self.path.startswith("/plot.png"):
                png = render_plot(state)
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(png)))
                self.end_headers(); self.wfile.write(png)
            else:
                html = render_html(state).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers(); self.wfile.write(html)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="/tmp/z37_run.log",
                     help="log file produced by zNN_*.py via stdout")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    state = State(Path(args.log))
    state.refresh()
    print(f"DE dashboard: http://localhost:{args.port}/  log={args.log}")
    HTTPServer(("0.0.0.0", args.port), make_handler(state)).serve_forever()


if __name__ == "__main__":
    main()
