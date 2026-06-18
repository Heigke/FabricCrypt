"""Phase A driver: A1 baseline + A2 workload-invariance + A3 time-stability.

Writes results to results/IDENTITY_BENCHMARK_2026-05-30/embodiment/phase_a/
and progress to state/embodiment_state.json.

Usage:
  python phase_a_run.py --steps A1,A2,A3
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, socket
from pathlib import Path
import numpy as np

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
HERE = ROOT / "scripts/identity_benchmark/embodiment"
OUT = ROOT / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment/phase_a"
STATE = ROOT / "state/embodiment_state.json"
PY = str(ROOT / "venv/bin/python")
ENV_SCRIPT = str(HERE / "envelope_fast.py")

DAED_PASS = os.environ.get("DAEDALUS_PASS", "daedalus")
DAED_HOST = os.environ.get("DAEDALUS_HOST", "daedalus.local")
DAED_USER = os.environ.get("DAEDALUS_USER", "daedalus")
DAED_REPO = "/home/daedalus/AMD_gfx1151_energy"


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {}

def save_state(d):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(d, indent=2, default=str))


def collect_local(out_name: str, label: str, quick: bool = False) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{out_name}.json"
    cmd = [PY, ENV_SCRIPT, "--out", str(out), "--label", label]
    if quick: cmd.append("--quick")
    print(f"[A] LOCAL collect → {out.name} (label={label})", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    print(r.stdout[-500:], flush=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-500:], flush=True)
        raise RuntimeError(f"local collect failed: {r.returncode}")
    return json.loads(out.read_text())


def collect_daedalus(out_name: str, label: str, quick: bool = False) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{out_name}.json"
    remote_out = f"/tmp/embodiment_{out_name}.json"
    py_cmd = (
        f"cd {DAED_REPO} && venv/bin/python "
        f"scripts/identity_benchmark/embodiment/envelope_fast.py "
        f"--out {remote_out} --label {label}"
    )
    if quick: py_cmd += " --quick"
    print(f"[A] DAEDALUS collect → {out.name} (label={label})", flush=True)
    sshcmd = ["sshpass", "-p", DAED_PASS, "ssh",
              "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
              f"{DAED_USER}@{DAED_HOST}", py_cmd]
    r = subprocess.run(sshcmd, capture_output=True, text=True, timeout=600)
    print(r.stdout[-500:], flush=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-500:], flush=True)
        raise RuntimeError(f"daedalus collect failed: {r.returncode}")
    # pull file
    scp = ["sshpass", "-p", DAED_PASS, "scp",
           "-o", "StrictHostKeyChecking=no",
           f"{DAED_USER}@{DAED_HOST}:{remote_out}", str(out)]
    r2 = subprocess.run(scp, capture_output=True, text=True, timeout=60)
    if r2.returncode != 0:
        raise RuntimeError(f"scp pull failed: {r2.stderr}")
    return json.loads(out.read_text())


def dist(v1: list, v2: list) -> dict:
    """NOTE: joint-z over only 2 vectors is degenerate (always sqrt(2*23)).
    Use raw_L2, relative_L2 (normalized by L2 of vec1), and cosine.
    """
    a = np.asarray(v1, dtype=float); b = np.asarray(v2, dtype=float)
    return {
        "l2_raw": float(np.linalg.norm(a - b)),
        "rel_l2": float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12)),
        "cos_sim_raw": float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12)),
        "cos_dist": float(1.0 - a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12)),
    }


def dist_population(vecs: list[list]) -> dict:
    """Pairwise distances within a group (returns mean,std,max)."""
    if len(vecs) < 2:
        return {"n_pairs": 0}
    arr = np.asarray(vecs, dtype=float)
    M = arr
    mu = M.mean(axis=0); sd = M.std(axis=0) + 1e-12
    Z = (M - mu) / sd
    ds = []
    for i in range(len(Z)):
        for j in range(i+1, len(Z)):
            ds.append(float(np.linalg.norm(Z[i] - Z[j])))
    return {"n_pairs": len(ds), "mean": float(np.mean(ds)), "std": float(np.std(ds)),
            "max": float(np.max(ds)), "all": ds}


def step_A1():
    print("=== A1: baseline cross-machine ===", flush=True)
    ik = collect_local("A1_ikaros", "A1_ikaros_idle")
    da = collect_daedalus("A1_daedalus", "A1_daedalus_idle")
    d = dist(ik["vec23"], da["vec23"])
    res = {"step": "A1", "ikaros": ik["vec23"], "daedalus": da["vec23"],
           "labels": ik["labels"], "distance": d,
           "D0_l2_raw": d["l2_raw"],
           "D0_rel_l2": d["rel_l2"],
           "D0_cos_dist": d["cos_dist"]}
    (OUT / "A1_result.json").write_text(json.dumps(res, indent=2))
    print(f"[A1] D0 raw_L2={d['l2_raw']:.3e}  rel_L2={d['rel_l2']:.4f}  cos_dist={d['cos_dist']:.5f}", flush=True)
    return res


def step_A2():
    print("=== A2: workload-invariance on ikaros ===", flush=True)
    results = {}
    # IDLE: do nothing intensive
    results["IDLE"] = collect_local("A2_ikaros_IDLE", "A2_ikaros_idle")
    # CPU stress in background
    print("--- A2 CPU stress collect ---", flush=True)
    import subprocess as sp, signal
    stress = sp.Popen([
        PY, "-c",
        "import numpy as np,time,threading\n"
        "def w():\n A=np.random.randn(384,384).astype('f4');B=np.random.randn(384,384).astype('f4')\n"
        " while True: A=A@B*1e-3+1e-3\n"
        "ts=[threading.Thread(target=w,daemon=True) for _ in range(4)]\n"
        "[t.start() for t in ts]\ntime.sleep(180)\n"
    ])
    try:
        time.sleep(2)
        results["CPU"] = collect_local("A2_ikaros_CPU", "A2_ikaros_cpu_stress")
    finally:
        stress.terminate(); time.sleep(1)
        try: stress.kill()
        except Exception: pass
    # Cool down
    print("--- A2 wait cool ---", flush=True)
    for _ in range(60):
        t = int(open("/sys/class/thermal/thermal_zone0/temp").read().strip())/1000
        if t < 55: break
        time.sleep(3)
    # GPU stress (HIP) — use small kernel via torch matmul on ROCm
    print("--- A2 GPU stress collect ---", flush=True)
    gpu = sp.Popen([PY, "-c",
        "import os,time\nos.environ['HSA_OVERRIDE_GFX_VERSION']='11.0.0'\n"
        "import torch\n"
        "if not torch.cuda.is_available():\n print('NO GPU'); time.sleep(180); raise SystemExit\n"
        "dev='cuda'\nA=torch.randn(1024,1024,device=dev)\nB=torch.randn(1024,1024,device=dev)\n"
        "t0=time.time()\n"
        "while time.time()-t0<180:\n  for _ in range(50): C=A@B\n  torch.cuda.synchronize(); time.sleep(0.1)\n"
    ])
    try:
        time.sleep(3)
        results["GPU"] = collect_local("A2_ikaros_GPU", "A2_ikaros_gpu_stress")
    finally:
        gpu.terminate(); time.sleep(1)
        try: gpu.kill()
        except Exception: pass

    vecs = [results[k]["vec23"] for k in ("IDLE","CPU","GPU")]
    arr = np.asarray(vecs, dtype=float)
    pop = dist_population(vecs)
    # "essence" = per-feature rank ordering stability (Spearman rank corr across workloads)
    ranks = np.argsort(arr, axis=1).argsort(axis=1)  # rank each row
    rho_pairs = []
    for i in range(3):
        for j in range(i+1, 3):
            rho_pairs.append(float(np.corrcoef(ranks[i], ranks[j])[0,1]))
    res = {"step": "A2",
           "workloads": list(results.keys()),
           "vecs": {k: results[k]["vec23"] for k in results},
           "pairwise_dist": pop,
           "rank_corr_pairs": rho_pairs,
           "rank_corr_mean": float(np.mean(rho_pairs)),
           "labels": results["IDLE"]["labels"]}
    (OUT / "A2_result.json").write_text(json.dumps(res, indent=2))
    print(f"[A2] mean pairwise dist={pop.get('mean'):.3f}, rank_corr_mean={res['rank_corr_mean']:.3f}", flush=True)
    return res


def step_A3(hours=(1, 2)):
    print("=== A3: time stability — initial sample now ===", flush=True)
    s0 = collect_local("A3_t0", "A3_t0")
    samples = [{"t_h": 0.0, "vec": s0["vec23"], "ts": s0["timestamp"]}]
    for h in hours:
        target = time.time() + h*3600
        print(f"[A3] waiting until t+{h}h (sleep {h*3600}s)", flush=True)
        while time.time() < target:
            time.sleep(min(60, max(1, target - time.time())))
        sx = collect_local(f"A3_t{h}h", f"A3_t{h}h")
        samples.append({"t_h": float(h), "vec": sx["vec23"], "ts": sx["timestamp"]})
    vecs = [s["vec"] for s in samples]
    pop = dist_population(vecs)
    res = {"step": "A3", "samples": samples, "pairwise_dist": pop,
           "labels": s0["labels"]}
    (OUT / "A3_result.json").write_text(json.dumps(res, indent=2))
    print(f"[A3] mean drift={pop.get('mean'):.3f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="A1")
    ap.add_argument("--a3-hours", default="1,2",
                    help="comma-separated hours for A3 samples after t0")
    args = ap.parse_args()
    steps = args.steps.split(",")
    state = load_state()
    state.setdefault("phase_a", {})
    for s in steps:
        try:
            if s == "A1":
                state["phase_a"]["A1"] = step_A1()
            elif s == "A2":
                state["phase_a"]["A2"] = step_A2()
            elif s == "A3":
                hrs = tuple(float(x) for x in args.a3_hours.split(","))
                state["phase_a"]["A3"] = step_A3(hours=hrs)
            else:
                print(f"unknown step {s}", flush=True)
            state["phase_a"][f"{s}_done_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            save_state(state)
        except Exception as e:
            print(f"[ERROR] step {s} failed: {e}", flush=True)
            state["phase_a"][f"{s}_error"] = str(e)
            save_state(state)
            raise

if __name__ == "__main__":
    main()
