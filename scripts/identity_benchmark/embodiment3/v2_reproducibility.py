"""V2 reproducibility tests for the robust signature.

V2a: same machine, two N=100 collections 5 min apart → bits drift
V2b: same machine, two collections 1 hour apart  (optional, time budget)
V2c: same machine, two collections under different workloads → bits drift
V2d: cross-machine ikaros vs daedalus → bits drift
V2e: post-reboot (executed by phase R, not here)

GATE: V2a-d must pass before training (Phase V3). Pass criteria:
  - same-machine drift (V2a, V2c) ≤ 5% of bitstring length
  - cross-machine drift (V2d) ≥ 25% of bitstring length
  - ratio (cross / same) ≥ 5×

Usage:
    venv/bin/python v2_reproducibility.py --phase a   # 5-min repeat
    venv/bin/python v2_reproducibility.py --phase c   # workload variants
    venv/bin/python v2_reproducibility.py --phase d   # cross-machine (needs daedalus sig)
    venv/bin/python v2_reproducibility.py --summary   # print gates summary
"""
from __future__ import annotations
import argparse, json, os, sys, time, subprocess
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from robust_signature import (collect_full_signature, quantize_robust,
                                signature_hash, bit_distance,
                                save_signature, load_signature,
                                quantized_to_bitstring)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT = ROOT / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment3"
SIGS = OUT / "signatures"
SIGS.mkdir(parents=True, exist_ok=True)

# Default sample budget — tradeoff between robustness and time
# N=60 samples × 0.5s interval × extra micro_dur ≈ 45s per signature.
DEFAULT_N = 60
DEFAULT_INTERVAL = 0.5


def _wait_cool(target_c: float = 55.0, timeout_s: float = 120.0):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            t = float(open("/sys/class/thermal/thermal_zone0/temp").read().strip()) / 1000.0
        except Exception:
            t = 0.0
        if t < target_c:
            return True
        time.sleep(2)
    return False


def collect_to(path: Path, label: str, N: int = DEFAULT_N, interval: float = DEFAULT_INTERVAL):
    print(f"[V2] collecting {label} → {path.name} (N={N})", flush=True)
    _wait_cool(60.0)
    sig = collect_full_signature(N_samples=N, sample_interval_s=interval, label=label, per_core=True)
    save_signature(sig, str(path))
    q = quantize_robust(sig)
    h = signature_hash(q)
    bs = quantized_to_bitstring(q)
    print(f"[V2] {label}: n_features={q['n_features']} bitstring_len={len(bs)} hash={h[:16]}", flush=True)
    return sig, q, bs


def phase_a(N: int = DEFAULT_N):
    """Same machine, two collections 5 min apart."""
    out = {}
    p1 = SIGS / "ikaros_v2a_t0.json"
    p2 = SIGS / "ikaros_v2a_t5min.json"
    s1, q1, b1 = collect_to(p1, "v2a_t0", N=N)
    print("[V2a] waiting 5 min for second collection...", flush=True)
    time.sleep(300)
    s2, q2, b2 = collect_to(p2, "v2a_t5min", N=N)
    d, n = bit_distance(q1, q2)
    out = {"phase": "V2a", "p1": str(p1), "p2": str(p2),
           "hamming": d, "total_bits": n, "pct": 100.0 * d / max(1, n),
           "hash1": signature_hash(q1)[:16], "hash2": signature_hash(q2)[:16]}
    (OUT / "v2a_result.json").write_text(json.dumps(out, indent=2))
    print(f"[V2a] hamming={d}/{n} ({out['pct']:.1f}%)", flush=True)
    return out


def phase_c(N: int = DEFAULT_N):
    """Workload variants: idle baseline vs cpu-stress concurrent.
    The signature collector already rotates workloads internally;
    here we add an explicit CPU stressor in the background for half the samples."""
    out = {}
    # First: pure-idle collection (collector's own micro-bursts only)
    p1 = SIGS / "ikaros_v2c_idle.json"
    _wait_cool(55.0)
    s1, q1, b1 = collect_to(p1, "v2c_idle", N=N)

    # Second: background light CPU load during collection
    p2 = SIGS / "ikaros_v2c_loaded.json"
    _wait_cool(55.0)
    # Launch background load
    import threading
    stop = [False]
    def loader():
        a = np.random.randn(128, 128).astype(np.float32)
        b = np.random.randn(128, 128).astype(np.float32)
        while not stop[0]:
            a = a @ b * 1e-3 + 1e-3
            time.sleep(0.01)  # don't burn CPU completely
    th = threading.Thread(target=loader, daemon=True)
    th.start()
    try:
        s2, q2, b2 = collect_to(p2, "v2c_loaded", N=N)
    finally:
        stop[0] = True
        time.sleep(0.5)

    d, n = bit_distance(q1, q2)
    out = {"phase": "V2c", "p1": str(p1), "p2": str(p2),
           "hamming": d, "total_bits": n, "pct": 100.0 * d / max(1, n),
           "hash1": signature_hash(q1)[:16], "hash2": signature_hash(q2)[:16]}
    (OUT / "v2c_result.json").write_text(json.dumps(out, indent=2))
    print(f"[V2c] hamming={d}/{n} ({out['pct']:.1f}%)", flush=True)
    return out


def phase_d():
    """Cross-machine ikaros vs daedalus.
    Requires daedalus signature to be collected via SSH push."""
    p_ik = SIGS / "ikaros_v2d.json"
    p_da = SIGS / "daedalus_v2d.json"

    # Local: collect ikaros if not exist
    if not p_ik.exists():
        collect_to(p_ik, "v2d_ikaros")

    # Daedalus: invoke via SSH
    if not p_da.exists():
        print("[V2d] collecting daedalus signature via SSH...", flush=True)
        # Copy module to daedalus
        rs = HERE / "robust_signature.py"
        host = os.environ.get("DAEDALUS_HOST", "daedalus.local")
        user = os.environ.get("DAEDALUS_USER", "daedalus")
        pw = os.environ.get("DAEDALUS_PASS", "daedalus")
        remote_path = "/tmp/robust_signature.py"
        remote_out = "/tmp/daedalus_v2d.json"

        sub = subprocess.run(
            ["sshpass", "-p", pw, "scp", "-o", "StrictHostKeyChecking=no",
             str(rs), f"{user}@{host}:{remote_path}"],
            capture_output=True, text=True, timeout=60,
        )
        print(f"[V2d] scp: {sub.returncode} {sub.stderr[:200]}", flush=True)

        sub = subprocess.run(
            ["sshpass", "-p", pw, "ssh", "-o", "StrictHostKeyChecking=no",
             f"{user}@{host}",
             f"/home/daedalus/venvs/torch-rocm/bin/python {remote_path} "
             f"--N {DEFAULT_N} --interval {DEFAULT_INTERVAL} "
             f"--label daedalus_v2d --out {remote_out}"],
            capture_output=True, text=True, timeout=600,
        )
        print(f"[V2d] ssh run: {sub.returncode}", flush=True)
        print(f"[V2d] stdout tail: {sub.stdout[-300:]}", flush=True)
        if sub.returncode != 0:
            print(f"[V2d] stderr: {sub.stderr[-300:]}", flush=True)
            return {"phase": "V2d", "error": "ssh failed", "stderr": sub.stderr[-300:]}

        sub = subprocess.run(
            ["sshpass", "-p", pw, "scp", "-o", "StrictHostKeyChecking=no",
             f"{user}@{host}:{remote_out}", str(p_da)],
            capture_output=True, text=True, timeout=60,
        )
        print(f"[V2d] scp back: {sub.returncode}", flush=True)

    s_ik = load_signature(str(p_ik))
    s_da = load_signature(str(p_da))
    q_ik = quantize_robust(s_ik)
    q_da = quantize_robust(s_da)
    d, n = bit_distance(q_ik, q_da)
    out = {"phase": "V2d", "p1": str(p_ik), "p2": str(p_da),
           "hamming": d, "total_bits": n, "pct": 100.0 * d / max(1, n),
           "hash_ikaros": signature_hash(q_ik)[:16],
           "hash_daedalus": signature_hash(q_da)[:16],
           "ikaros_n_features": q_ik["n_features"],
           "daedalus_n_features": q_da["n_features"]}
    (OUT / "v2d_result.json").write_text(json.dumps(out, indent=2))
    print(f"[V2d] hamming={d}/{n} ({out['pct']:.1f}%)", flush=True)
    return out


def summary():
    out = {}
    for ph in ("v2a", "v2c", "v2d"):
        p = OUT / f"{ph}_result.json"
        if p.exists():
            out[ph] = json.loads(p.read_text())
    # Verdict
    same_max = 0.0
    cross = 0.0
    if "v2a" in out:
        same_max = max(same_max, out["v2a"]["pct"])
    if "v2c" in out:
        same_max = max(same_max, out["v2c"]["pct"])
    if "v2d" in out:
        cross = out["v2d"]["pct"]
    out["verdict"] = {
        "same_machine_max_drift_pct": same_max,
        "cross_machine_drift_pct": cross,
        "ratio_cross_over_same": (cross / same_max) if same_max > 1e-6 else float("inf"),
        "same_pass": same_max <= 5.0,
        "cross_pass": cross >= 25.0,
        "ratio_pass": (cross / same_max if same_max > 1e-6 else float("inf")) >= 5.0,
    }
    out["verdict"]["GATE_PASS"] = (out["verdict"]["same_pass"]
                                    and out["verdict"]["cross_pass"]
                                    and out["verdict"]["ratio_pass"])
    (OUT / "v2_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["verdict"], indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["a", "c", "d", "all"], default="all")
    ap.add_argument("--N", type=int, default=DEFAULT_N)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    if args.summary:
        summary()
        return
    if args.phase in ("a", "all"):
        phase_a(N=args.N)
    if args.phase in ("c", "all"):
        phase_c(N=args.N)
    if args.phase in ("d", "all"):
        phase_d()
    summary()


if __name__ == "__main__":
    main()
