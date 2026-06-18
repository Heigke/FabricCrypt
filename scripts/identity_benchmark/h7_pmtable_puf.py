"""H7 PUF discovery from the SMU PM table — hunt for a GENUINELY die-random fingerprint.

CPPC binning turned out ~83% systematic (rank-corr 0.83 between same-SKU ikaros/daedalus) -> too little
real entropy to separate the dies. The PM table exposes fused per-core values (voltages, leakage, limits)
that include random dopant variation = a real PUF candidate.

enroll: read pm_table (float32 array) N times at idle, save per-offset mean + std.
  -> STABLE offsets (low coeff of variation) are fused/static. Save them.
compare: load two dies' enrollments, intersect stable offsets, and split into
  CONSTANT (same value -> shared SKU spec) vs DIVERGENT (differ -> die-unique PUF bits).
  Report how much real cross-die separation the divergent-stable subset gives vs raw CPPC's cos 0.86.

Root required. Env: RUNTAG (default r1), NREAD (default 40).
"""
from __future__ import annotations
import os, sys, time, socket
from pathlib import Path
import numpy as np

HOST = socket.gethostname()
OUT = Path(os.environ["H7_OUT"]) if os.environ.get("H7_OUT") else \
      Path(__file__).resolve().parents[2] / "results/IDENTITY_H7_2026-06-09"
PM = Path("/sys/kernel/ryzen_smu_drv/pm_table")
RUNTAG = os.environ.get("RUNTAG", "r1")
NREAD = int(os.environ.get("NREAD", "40"))
CV_STABLE = 1e-3          # coeff of variation below this = stable/fused


def read_pm() -> np.ndarray:
    b = PM.read_bytes()
    n = (len(b)//4)*4
    return np.frombuffer(b[:n], dtype=np.float32).astype(np.float64)


def enroll():
    reads = []
    for _ in range(NREAD):
        reads.append(read_pm()); time.sleep(0.25)
    R = np.stack(reads)                     # [NREAD, nfloat]
    mean = R.mean(0); std = R.std(0)
    cv = std/(np.abs(mean)+1e-12)
    stable = (cv < CV_STABLE) & (np.abs(mean) > 1e-9)   # ignore zero padding
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT/f"pmpuf_{HOST}_{RUNTAG}.npz"
    np.savez_compressed(p, mean=mean, std=std, cv=cv, stable=stable, nfloat=len(mean))
    print(f"[{HOST}] nfloat={len(mean)} stable_offsets={int(stable.sum())} "
          f"(CV<{CV_STABLE})  saved {p.name}", flush=True)


def compare():
    import glob
    fs = sorted(glob.glob(str(OUT/f"pmpuf_*_{RUNTAG}.npz")))
    hosts = [Path(f).stem.split("pmpuf_")[1].rsplit("_", 1)[0] for f in fs]
    if len(fs) < 2:
        print("need 2 enrollments"); return
    A, B = hosts[0], hosts[1]
    dA, dB = np.load(fs[0]), np.load(fs[1])
    both_stable = dA["stable"] & dB["stable"]
    mA, mB = dA["mean"], dB["mean"]
    idx = np.where(both_stable)[0]
    va, vb = mA[idx], mB[idx]
    # split constant vs divergent (relative difference)
    rel = np.abs(va - vb)/(np.maximum(np.abs(va), np.abs(vb))+1e-12)
    const = rel < 1e-3
    diverg = ~const
    print(f"dies: A={A} B={B}", flush=True)
    print(f"both-stable offsets = {len(idx)}", flush=True)
    print(f"  CONSTANT (shared SKU spec) = {int(const.sum())}", flush=True)
    print(f"  DIVERGENT (die-unique PUF) = {int(diverg.sum())}", flush=True)
    if diverg.sum() == 0:
        print("  -> NO divergent stable offsets: PM table gives no extra die entropy here.", flush=True)
        return
    # how separable on the divergent-stable subset? (z-score each offset by cross-die scale, cos)
    da = va[diverg]; db = vb[diverg]
    # normalize per-offset by mean magnitude so big-scale offsets don't dominate
    sc = (np.abs(da)+np.abs(db))/2 + 1e-12
    na, nb = da/sc, db/sc
    cos = float(na@nb/(np.linalg.norm(na)*np.linalg.norm(nb)+1e-12))
    print(f"\n  divergent-subset cos(A,B) = {cos:+.3f}   (lower = more separable; CPPC raw was 0.86)", flush=True)
    print(f"  median rel-divergence = {np.median(rel[diverg])*100:.2f}%   max = {rel[diverg].max()*100:.1f}%", flush=True)
    # show a few top-divergent offsets (candidate PUF bits)
    order = idx[diverg][np.argsort(-rel[diverg])][:12]
    print("\n  top divergent offsets (idx: A_val vs B_val, rel%):", flush=True)
    for o in order:
        print(f"    [{o:4d}] {mA[o]:+.5g}  vs  {mB[o]:+.5g}   ({abs(mA[o]-mB[o])/(max(abs(mA[o]),abs(mB[o]))+1e-12)*100:.1f}%)", flush=True)


if __name__ == "__main__":
    (compare if len(sys.argv) > 1 and sys.argv[1] == "compare" else enroll)()
