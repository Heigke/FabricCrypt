"""signature_v2 — 290-dim per-die fingerprint from 5 HAL-bypass signals.

Output layout:
  [   0:  35) TSC inter-core offsets (7 targets x 5 stats)
  [  35:  70) Cacheline ping-pong RTT (7 pairs x 5 stats)
  [  70: 270) DRAM-refresh histogram (200 log-spaced bins)
  [ 270: 280) Syscall p99.9 tail (10 percentiles)
  [ 280: 290) NVMe / FS read p99.9 tail (10 percentiles)

Capture takes ~30-40s per rep on a 16-core APU at room temperature.
Strict thermal: abort at 68C, pause at 63C, cool to 50C.

Usage:
    from fabriccrypt.signature import extract_one
    # tsc_bin and cl_bin are compiled C helpers from src/signature/*.c
    vec = extract_one(tsc_bin_path, cl_bin_path)
    assert vec.shape == (290,)
"""
import os
import sys
import time
import argparse
import numpy as np

from .thermal import thermal_guard, wait_cool, hostname, get_apu_temp_c, save_json, compile_c
from .tsc_offset import block_tsc
from .cacheline_pingpong import block_cacheline
from .dram_refresh import block_dram
from .syscall_latency import block_nanosleep
from .nvme_latency import block_nvme

HERE = os.path.dirname(os.path.abspath(__file__))

DIMS = {"tsc": 35, "cacheline": 35, "dram": 200, "nanosleep": 10, "nvme": 10}
TOTAL_DIM = sum(DIMS.values())  # 290
BLOCK_STARTS = {"tsc": 0, "cacheline": 35, "dram": 70,
                "nanosleep": 270, "nvme": 280}


def extract_one(tsc_bin: str, cl_bin: str) -> np.ndarray:
    """Run one full capture, return 290-dim float64 vector."""
    v = np.zeros(TOTAL_DIM, dtype=np.float64)
    v[BLOCK_STARTS["tsc"]:BLOCK_STARTS["tsc"] + DIMS["tsc"]] = block_tsc(tsc_bin)
    v[BLOCK_STARTS["cacheline"]:BLOCK_STARTS["cacheline"] + DIMS["cacheline"]] = block_cacheline(cl_bin)
    v[BLOCK_STARTS["dram"]:BLOCK_STARTS["dram"] + DIMS["dram"]] = block_dram()
    v[BLOCK_STARTS["nanosleep"]:BLOCK_STARTS["nanosleep"] + DIMS["nanosleep"]] = block_nanosleep()
    v[BLOCK_STARTS["nvme"]:BLOCK_STARTS["nvme"] + DIMS["nvme"]] = block_nvme()
    return v


def ensure_c_binaries(here: str = HERE):
    tsc_src = os.path.join(here, "tsc_inter_core.c")
    tsc_bin = os.path.join(here, "tsc_inter_core")
    cl_src  = os.path.join(here, "cacheline_pingpong.c")
    cl_bin  = os.path.join(here, "cacheline_pingpong")
    if not os.path.exists(tsc_bin):
        compile_c(tsc_src, tsc_bin)
    if not os.path.exists(cl_bin):
        compile_c(cl_src, cl_bin)
    return tsc_bin, cl_bin


def main():
    ap = argparse.ArgumentParser(description="Capture FabricCrypt 290-dim signature")
    ap.add_argument("--reps", type=int, default=10,
                    help="number of repeated captures (default 10)")
    ap.add_argument("--host", default=None,
                    help="label for this host (default: /etc/hostname)")
    ap.add_argument("--out_dir", default="data",
                    help="directory to write <host>_sig_v2.npz (default: data/)")
    args = ap.parse_args()

    host = args.host or hostname()
    os.makedirs(args.out_dir, exist_ok=True)
    tsc_bin, cl_bin = ensure_c_binaries()

    vecs = np.zeros((args.reps, TOTAL_DIM), dtype=np.float64)
    meta = {"host": host, "reps": args.reps, "dim": TOTAL_DIM,
            "block_starts": BLOCK_STARTS, "dims": DIMS,
            "t_start": time.time(),
            "apu_temp_start_c": get_apu_temp_c(),
            "rep_temps_c": [], "rep_seconds": []}

    print(f"[sig_v2] host={host} reps={args.reps} dim={TOTAL_DIM} "
          f"temp={get_apu_temp_c():.1f}C", flush=True)

    for r in range(args.reps):
        wait_cool(target_c=55, timeout_s=120)
        t0 = time.time()
        try:
            vecs[r] = extract_one(tsc_bin, cl_bin)
        except SystemExit as e:
            print(f"[sig_v2] thermal abort during rep {r}: {e}", flush=True)
            vecs = vecs[:r]
            break
        meta["rep_temps_c"].append(get_apu_temp_c())
        dt = time.time() - t0
        meta["rep_seconds"].append(dt)
        print(f"[sig_v2] rep {r+1}/{args.reps} done {dt:.1f}s "
              f"temp={get_apu_temp_c():.1f}C", flush=True)

    meta["t_end"] = time.time()
    meta["apu_temp_end_c"] = get_apu_temp_c()

    out_npz = os.path.join(args.out_dir, f"{host}_sig_v2.npz")
    np.savez(out_npz, vec=vecs, host=host, dim=TOTAL_DIM,
             block_starts=np.array([BLOCK_STARTS[k]
                                    for k in ["tsc", "cacheline", "dram",
                                              "nanosleep", "nvme"]]))
    save_json(os.path.join(args.out_dir, f"{host}_sig_v2_meta.json"), meta)
    print(f"[sig_v2] saved {out_npz} shape={vecs.shape}", flush=True)
    return out_npz


if __name__ == "__main__":
    main()
