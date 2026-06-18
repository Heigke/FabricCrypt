#!/usr/bin/env python3
"""Angle B driver — run Lorenz HIP kernel 100 batches, parse, write JSON.

Thermal-safe:
- abort if APU temp >= 75°C
- 6s burst cap per batch (kernel actually <10 ms so this is moot)
- 10s cooling check every 10 batches

Output: B_lorenz_<host>.json with per-CU mean tail and summary.
"""
from __future__ import annotations
import json, os, socket, struct, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
_repo_candidate = ROOT.parents[2] if len(ROOT.parents) > 2 else ROOT.parent
if (_repo_candidate / "results").exists():
    OUT_DIR = _repo_candidate / "results/IDENTITY_BENCHMARK_2026-05-30/novel_v2"
else:
    OUT_DIR = ROOT / "results_novel_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
KERNEL = ROOT / "B_lorenz"

N_BATCHES = 100
TEMP_ABORT_C = 75.0
TEMP_PAUSE_C = 70.0
BURST_CAP_S = 6.0
COOL_EVERY = 10
COOL_S = 10

ENV = {**os.environ, "HSA_OVERRIDE_GFX_VERSION": "11.0.0"}

def temp_c() -> float:
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text().strip())/1000.0
    except Exception:
        return float('nan')

def run_batch(batch_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """Returns (tail, hwids, cycles, n_waves, tail_len, n_steps)."""
    t0 = time.time()
    proc = subprocess.run([str(KERNEL), str(batch_id)],
                          capture_output=True, env=ENV, timeout=BURST_CAP_S)
    if proc.returncode != 0:
        raise RuntimeError(f"kernel batch {batch_id} failed: rc={proc.returncode} stderr={proc.stderr[:400]!r}")
    blob = proc.stdout
    n_waves, tail_len, n_steps, _ = struct.unpack("<IIII", blob[:16])
    off = 16
    tail = np.frombuffer(blob, dtype=np.float32, count=n_waves*tail_len*3, offset=off)
    off += n_waves*tail_len*3*4
    hwids = np.frombuffer(blob, dtype=np.uint32, count=n_waves, offset=off)
    off += n_waves*4
    cycles = np.frombuffer(blob, dtype=np.uint32, count=n_waves, offset=off)
    return tail.reshape(n_waves, tail_len, 3), hwids, cycles, n_waves, tail_len, n_steps

def hwid_to_cu(hwid: np.uint32) -> int:
    """gfx11 HW_ID1: bits[5:0]=wave_id, bits[10:6]=simd_id, bits[15:11]=wgp_id,
    bits[18:16]=se_id, ... We approximate physical CU by (se, wgp)."""
    se  = (int(hwid) >> 16) & 0x7
    wgp = (int(hwid) >> 11) & 0x1F
    return se * 32 + wgp  # collapse to single index; harmless if values dense

def main():
    host = socket.gethostname()
    if not KERNEL.exists():
        sys.exit(f"missing kernel binary: {KERNEL}")
    t_start = temp_c()
    print(f"[B] host={host} start temp={t_start:.1f}C kernel={KERNEL}", file=sys.stderr)
    if t_start >= TEMP_ABORT_C:
        sys.exit(f"ABORT: start temp {t_start:.1f}C >= {TEMP_ABORT_C}")

    all_tails = []         # list of (n_waves, tail_len, 3)
    all_hwids = []
    all_cycles = []
    temps = [t_start]
    aborted = False
    abort_reason = None

    for b in range(N_BATCHES):
        t_pre = temp_c()
        if t_pre >= TEMP_ABORT_C:
            abort_reason = f"APU {t_pre:.1f}C >= {TEMP_ABORT_C}"
            aborted = True
            break
        if t_pre >= TEMP_PAUSE_C:
            print(f"[B] batch {b}: temp {t_pre:.1f}C >= pause threshold, sleep 15s", file=sys.stderr)
            time.sleep(15)
        try:
            tail, hwids, cycles, nw, tl, ns = run_batch(b)
        except subprocess.TimeoutExpired:
            abort_reason = f"batch {b} exceeded {BURST_CAP_S}s burst cap"
            aborted = True
            break
        all_tails.append(tail)
        all_hwids.append(hwids)
        all_cycles.append(cycles)
        t_post = temp_c()
        temps.append(t_post)
        if (b+1) % COOL_EVERY == 0:
            print(f"[B] batch {b+1}/{N_BATCHES} temp {t_post:.1f}C — cool {COOL_S}s", file=sys.stderr)
            time.sleep(COOL_S)

    if not all_tails:
        sys.exit("no batches completed")

    # Stack: (n_batches, n_waves, tail_len, 3)
    T = np.stack(all_tails, axis=0)  # float32
    H = np.stack(all_hwids, axis=0)  # uint32
    C = np.stack(all_cycles, axis=0)

    n_batches_done, n_waves, tail_len, _ = T.shape
    # Group waves by physical CU index for batch 0 mapping
    cu_idx_per_wave = np.array([hwid_to_cu(h) for h in H[0]], dtype=np.int32)
    unique_cus = sorted(set(int(c) for c in cu_idx_per_wave))

    # Per-CU mean trajectory tail across batches AND waves on that CU
    per_cu_tail_mean = {}
    per_cu_tail_std = {}
    for cu in unique_cus:
        mask = (cu_idx_per_wave == cu)
        # shape: (n_batches, n_waves_on_cu, tail_len, 3)
        sub = T[:, mask, :, :]
        per_cu_tail_mean[cu] = sub.mean(axis=(0,1))     # (tail_len, 3)
        per_cu_tail_std[cu]  = sub.std(axis=(0,1))

    # Within-device per-CU variability: L2 norm of per-batch tail relative to mean
    per_cu_within_std = {}
    for cu in unique_cus:
        mask = (cu_idx_per_wave == cu)
        sub = T[:, mask, :, :]           # (B, Nw, L, 3)
        mean = sub.mean(axis=(0,1), keepdims=True)
        # batch-wise tail mean per CU
        batch_means = sub.mean(axis=1)   # (B, L, 3)
        diff = batch_means - mean.squeeze(1)  # (B, L, 3)
        per_cu_within_std[cu] = float(np.sqrt((diff**2).sum(axis=(1,2)).mean()))

    # Lyapunov estimate via divergence of intra-wave lanes — we only kept lane 0,
    # so estimate from neighbouring-batch tail divergence: lambda ~ log(d_t/d_0)/T
    # Use first 50 tail steps of two consecutive batches on same CU.
    lyap_estimates = []
    for cu in unique_cus[:8]:
        mask = (cu_idx_per_wave == cu)
        sub = T[:, mask, :, :].mean(axis=1)   # (B, L, 3)
        if sub.shape[0] >= 2 and sub.shape[1] >= 50:
            d = np.linalg.norm(sub[1, :50] - sub[0, :50], axis=1)
            d0 = max(d[0], 1e-9); dT = max(d[-1], 1e-9)
            lam = float(np.log(dT/d0) / (49 * 0.01))
            lyap_estimates.append(lam)
    lyap_mean = float(np.mean(lyap_estimates)) if lyap_estimates else float('nan')

    out_npz = OUT_DIR / f"B_lorenz_{host}.npz"
    np.savez_compressed(
        out_npz,
        tails=T,
        hwids=H,
        cycles=C,
        cu_idx_per_wave=cu_idx_per_wave,
        unique_cus=np.array(unique_cus, dtype=np.int32),
        per_cu_tail_mean=np.stack([per_cu_tail_mean[c] for c in unique_cus]),
        per_cu_tail_std=np.stack([per_cu_tail_std[c] for c in unique_cus]),
    )

    out = {
        "host": host,
        "n_batches": n_batches_done,
        "n_waves_per_batch": int(n_waves),
        "tail_len": int(tail_len),
        "n_steps": int(10000),
        "n_unique_cus": len(unique_cus),
        "temps_c": {"start": t_start, "max": max(temps), "end": temps[-1]},
        "aborted": aborted,
        "abort_reason": abort_reason,
        "within_device_per_cu_tail_std_l2": per_cu_within_std,
        "within_device_pooled_std": float(np.mean(list(per_cu_within_std.values()))),
        "lyapunov_estimates_first_8_cus": lyap_estimates,
        "lyapunov_mean": lyap_mean,
        "raw_npz": str(out_npz),
    }
    out_json = OUT_DIR / f"B_lorenz_{host}.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(json.dumps({"host": host, "verdict_pending": "needs cross-device compare",
                      "lyap_mean": lyap_mean,
                      "within_pool_std": out["within_device_pooled_std"],
                      "temp_max": max(temps)}, indent=2))
    print(f"[B] wrote {out_json} and {out_npz}", file=sys.stderr)

if __name__ == "__main__":
    main()
