#!/usr/bin/env python3
"""J4 v5.5 cross-task driver — runs N_Cascade_KWS_ECG, N-HDC, N-LIF, N-STDP, NARMA
at 5-10× their current N, with v5.5 surrogate path active.

Per plan (research_plan/LARGE_SCALE_CAMPAIGN_2026-05-21.md):
  HDC-MNIST       N=3000 -> N=24000   (8×)
  Cascade-KWS     N=128  -> N=1024    (8×)
  NARMA-10        N=1000              (single big point; J2 owns the scaling sweep)
  LIF-MNIST       N=1024 -> N=8192    (8×)
  STDP-ECG        N=512  -> N=4096    (8×)

Each task is a subprocess so a single failure does not poison the others.
N override is via env vars the per-task scripts already honour (J4_N_OVERRIDE).
If a task does not honour the override we still log the legacy run as the
baseline for that task.

Output: results/CROSS_TASK_v55_2026-05-21/{task}.log + summary.json.
"""
from __future__ import annotations
import os, sys, json, time, subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "CROSS_TASK_v55_2026-05-21"
OUT.mkdir(parents=True, exist_ok=True)

# Task table: (label, script_path, env_overrides, expected_summary_path)
TASKS = [
    ("HDC_MNIST_3K",  "scripts/N-HDC-MNIST-3K-ikaros.py",
        {"J4_N_OVERRIDE": "24000"},
        "results/N-HDC-MNIST-3K-ikaros/summary.json"),
    ("Cascade_KWS",   "scripts/N_Cascade_KWS_ECG.py",
        {"J4_N_OVERRIDE": "1024"},
        "results/N_Cascade_KWS_ECG/summary.json"),
    ("LIF_MNIST",     "scripts/N-LIF-MNIST-daedalus.py",
        {"J4_N_OVERRIDE": "8192"},
        "results/N-LIF-MNIST-daedalus/summary.json"),
    ("STDP_ECG",      "scripts/N-STDP-ECG-zgx.py",
        {"J4_N_OVERRIDE": "4096"},
        "results/N-STDP-ECG-zgx/summary.json"),
    ("NARMA10_N1k",   "scripts/z2171_narma_benchmark.py",
        {"J4_N_OVERRIDE": "1000"},
        "results/z2171_narma_benchmark/summary.json"),
]


def run_task(label: str, script_rel: str, env_extra: dict,
             summary_rel: str) -> dict:
    script = REPO / script_rel
    if not script.exists():
        return {"label": label, "status": "MISSING_SCRIPT",
                "script": str(script_rel)}
    env = os.environ.copy()
    env.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
    env.update(env_extra)
    log_path = OUT / f"{label}.log"
    print(f"[J4v55] {label} ← {script_rel} (env+{env_extra})", flush=True)
    t0 = time.time()
    with open(log_path, "w") as logf:
        rc = subprocess.call(["python", str(script)], cwd=str(REPO),
                             env=env, stdout=logf, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    rec = {"label": label, "script": script_rel, "rc": int(rc),
           "wall_s": float(wall), "log": str(log_path.relative_to(REPO)),
           "env_overrides": env_extra, "summary_path": summary_rel}
    sum_path = REPO / summary_rel
    if sum_path.exists():
        try:
            rec["summary"] = json.loads(sum_path.read_text())
        except Exception as e:
            rec["summary_error"] = str(e)
    else:
        rec["summary"] = None
    print(f"[J4v55] {label} rc={rc} wall={wall:.1f}s "
          f"summary={'OK' if rec['summary'] else 'NONE'}", flush=True)
    return rec


def main():
    print(f"[J4v55] OUT={OUT}", flush=True)
    print(f"[J4v55] tasks: {[t[0] for t in TASKS]}", flush=True)
    records = []
    for label, script, env, summ in TASKS:
        rec = run_task(label, script, env, summ)
        records.append(rec)
        # incremental save
        (OUT / "summary.json").write_text(json.dumps(
            {"tasks": records, "ts": time.time()}, indent=2))
    print(f"[J4v55] DONE. summary at {OUT/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
