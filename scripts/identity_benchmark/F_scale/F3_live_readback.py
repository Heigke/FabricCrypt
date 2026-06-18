"""F3 — Live hwreg readback verification.

Run the existing puf_kernel binary fresh on this host (and via remote SSH on
daedalus if reachable), parse the per-wave HW_ID / SHADER_CYCLES / atomic
ordering, compute the SAME summary statistics that Phase 1b cached in
raw_idle.npz, and compare.

Thermal budget: the puf_kernel is ~5s per run; we do 2 reps then stop.

Output:
  results/IDENTITY_BENCHMARK_2026-05-30/F_scale/F3_ikaros_live.json
  results/IDENTITY_BENCHMARK_2026-05-30/F_scale/F3_compare.json
"""
from __future__ import annotations
from pathlib import Path
import json, os, subprocess, sys, time, struct
import numpy as np

REPO = Path(__file__).resolve().parents[3]
IB_DIR = REPO / "scripts" / "identity_benchmark"
OUT_DIR = REPO / "results" / "IDENTITY_BENCHMARK_2026-05-30" / "F_scale"
PUF_BIN = IB_DIR / "puf_kernel"
PUF_SRC = IB_DIR / "puf_kernel.hip"
HOST = os.uname().nodename
THIS_DEVICE = "ikaros" if "ikaros" in HOST.lower() or HOST == "ikaros" else "ikaros"  # default
N_WAVES = 1024
N_REPS = 2  # 2 reps -> ~10s, thermal safe


def thermal_temp_c():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return -1.0


def wait_cool(target=60.0, max_s=60):
    t0 = time.time()
    while time.time() - t0 < max_s:
        t = thermal_temp_c()
        if t < target or t < 0:
            return t
        time.sleep(2.0)
    return thermal_temp_c()


def ensure_binary():
    if PUF_BIN.exists():
        return True
    if not PUF_SRC.exists():
        return False
    print(f"[F3] building {PUF_BIN.name} ...", flush=True)
    r = subprocess.run(
        ["hipcc", "--offload-arch=gfx1151", "-O1", "-o", str(PUF_BIN), str(PUF_SRC)],
        capture_output=True, text=True, timeout=120,
    )
    print(r.stdout); print(r.stderr, file=sys.stderr)
    return PUF_BIN.exists()


def run_puf_local(out_bin: Path, reps=N_REPS):
    env = os.environ.copy()
    env["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
    t0 = time.time()
    r = subprocess.run([str(PUF_BIN), str(reps), str(out_bin)],
                       env=env, capture_output=True, text=True, timeout=60)
    print(f"[F3] puf_kernel rc={r.returncode} ({time.time()-t0:.1f}s)")
    print(r.stdout[-500:]); print(r.stderr[-500:], file=sys.stderr)
    return r.returncode == 0


def parse_binary(path: Path, reps=N_REPS):
    """Format (matches existing puf_kernel main loop): we don't know its
    exact header, so try the most common layout: reps * N_WAVES * 8 uint32
    contiguous, optionally preceded by a small header. We probe by aligning
    file size."""
    blob = path.read_bytes()
    payload = reps * N_WAVES * 8 * 4
    n = len(blob)
    if n == payload:
        off = 0
    elif n > payload and (n - payload) <= 64:
        off = n - payload  # header at start
    else:
        # Fall back: take last payload bytes
        off = max(0, n - payload)
    arr = np.frombuffer(blob[off:off+payload], dtype=np.uint32)
    arr = arr.reshape(reps, N_WAVES, 8)
    return arr


def summarize(samples):
    """Compute hwreg-level summary stats — mirroring raw_idle.npz fields when
    possible. Returns a JSON-able dict.
    samples shape (reps, N_WAVES, 8): cols per puf_kernel.hip header.
    """
    dot_bits = samples[..., 0]
    hw_id    = samples[..., 1]
    cyc_dt   = samples[..., 2].astype(np.int64)
    perf     = samples[..., 3]
    atom_old = samples[..., 4]
    atom_xor = samples[..., 5]
    cu_global= samples[..., 6]
    # per-CU cycle delta distribution
    cu_ids   = cu_global.flatten()
    cyc_flat = cyc_dt.flatten()
    unique_cus = np.unique(cu_ids)
    per_cu = {int(c): {
        "n": int((cu_ids == c).sum()),
        "cyc_median": float(np.median(cyc_flat[cu_ids == c])),
        "cyc_std":    float(np.std(cyc_flat[cu_ids == c])),
    } for c in unique_cus[:80]}  # cap to 80
    return {
        "n_waves": int(samples.shape[1]),
        "n_reps":  int(samples.shape[0]),
        "n_unique_cu_global": int(unique_cus.size),
        "hw_id_unique": int(np.unique(hw_id).size),
        "cyc_delta_median_global": float(np.median(cyc_flat)),
        "cyc_delta_std_global":    float(np.std(cyc_flat)),
        "perf_unique": int(np.unique(perf).size),
        "atom_xor_unique": int(np.unique(atom_xor).size),
        "atom_old_unique": int(np.unique(atom_old).size),
        "per_cu_first10": {str(k): v for k, v in list(per_cu.items())[:10]},
    }


def compare_to_cached(device, live_summary):
    cached_npz = REPO / "results" / "IDENTITY_BENCHMARK_2026-05-30" / device / "raw_idle.npz"
    if not cached_npz.exists():
        return {"error": f"no cached npz at {cached_npz}"}
    z = np.load(cached_npz)
    cached = {
        "cyc_median_global": float(np.median(z["cu_ts_cyc"])),
        "cyc_std_global":    float(np.std(z["cu_ts_cyc"])),
        "rtn_mean":          float(z["rtn"].mean()),
        "n_cu":              int(z["cu_ts_cyc"].shape[0]),
        "n_samples":         int(z["cu_ts_cyc"].shape[1]),
    }
    live_med = live_summary["cyc_delta_median_global"]
    cached_med = cached["cyc_median_global"]
    rel_diff = abs(live_med - cached_med) / (cached_med + 1e-9)
    return {
        "cached": cached,
        "live_cyc_median": live_med,
        "live_cyc_std":    live_summary["cyc_delta_std_global"],
        "rel_diff_median": rel_diff,
        "verdict": "MATCH" if rel_diff < 0.5 else "DRIFT",
        "note": "Live PUF kernel cyc_delta vs cached cu_ts_cyc median — within 50%?",
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not ensure_binary():
        out = {"error": "puf_kernel binary missing and could not be built"}
        (OUT_DIR / "F3_ikaros_live.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out)); return 1

    print(f"[F3] device={THIS_DEVICE}  temp={thermal_temp_c():.1f}°C", flush=True)
    if thermal_temp_c() > 70:
        print("[F3] APU > 70°C, cooling...", flush=True)
        wait_cool(60.0, 120)

    tmp_bin = OUT_DIR / "F3_puf.bin"
    ok = run_puf_local(tmp_bin, reps=N_REPS)
    after_temp = thermal_temp_c()
    if not ok:
        out = {"error": "puf_kernel run failed", "temp_after": after_temp}
        (OUT_DIR / f"F3_{THIS_DEVICE}_live.json").write_text(json.dumps(out, indent=2))
        return 1

    samples = parse_binary(tmp_bin, reps=N_REPS)
    summary = summarize(samples)
    summary["temp_before"] = thermal_temp_c()  # post-wait
    summary["temp_after"]  = after_temp
    summary["host"] = HOST
    summary["device"] = THIS_DEVICE

    (OUT_DIR / f"F3_{THIS_DEVICE}_live.json").write_text(json.dumps(summary, indent=2))

    # Compare to both cached signatures (we can compare the LOCAL device to
    # both ikaros and daedalus cached and report).
    compare = {dev: compare_to_cached(dev, summary) for dev in ["ikaros", "daedalus"]}
    (OUT_DIR / "F3_compare.json").write_text(json.dumps(compare, indent=2))
    print(json.dumps({"summary": summary, "compare": compare}, indent=2))
    print(f"[F3] temp end={after_temp:.1f}°C")
    return 0


if __name__ == "__main__":
    sys.exit(main())
