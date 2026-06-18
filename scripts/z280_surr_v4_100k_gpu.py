"""SURR-V4: 100K-point GPU-batched 5D surrogate of NSRAM 2T cell.

Same 5D parameter space as MEP-3 (z279):
    V_G1, V_G2, V_d, V_Nwell, V_b

Unlike z279 (4125-pt locked grid, CPU pool), this builds a 100K-point
GPU-batched dataset on AMD ROCm via `nsram.bsim4_port.forward_2t_batched_gpu`.

Sampling strategy:
  * V_Nwell — uniform discrete over 10 values (outer loop, cfg.vnwell scalar)
  * V_d     — uniform discrete over 10 values (shared per-batch Vd_seq, T=10)
  * V_G1, V_G2, V_b_init — uniform random per cell (LHS-ish)

That gives 10 V_Nwell × N_cells × T_vd = 100,000 points.
With N_cells=1000, T=10, that's 10 batches of 1000×10=10K → 100K total.

Outputs:
  results/z280_surr_v4_100k/
    surrogate_100k.npz   — (Id, Iii, Ileak, vg1, vg2, vd, vnwell, vb,
                             converged) all shape (100000,)
    summary.json         — wall, conv_rate, samples per V_Nwell, etc.

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 \
    python scripts/z280_surr_v4_100k_gpu.py
"""
from __future__ import annotations
import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
import json
import time
from pathlib import Path
import importlib.util

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent

# ---- Same 5D space as MEP-3, but denser/randomised ------------------------
N_VNWELL = 10                              # outer loop
N_VD     = 10                              # shared Vd sweep per batch
N_CELLS  = 1000                            # cells per (V_Nwell) batch
# 10 × 1000 × 10 = 100,000

VNWELL_AXIS = np.linspace(0.5, 5.0, N_VNWELL)               # [V]
VD_AXIS     = np.linspace(0.50, 2.50, N_VD)                 # [V]

VG1_RANGE = (0.10, 0.70)   # bracket MEP-3 [0.20, 0.60]
VG2_RANGE = (0.00, 0.70)   # bracket MEP-3 [0.00, 0.60]
VB_RANGE  = (0.00, 1.00)   # same as MEP-3 [0.00, 1.00]

SEED = 20260514

# ---------------------------------------------------------------------------


def _build_models():
    # Mirror z279 builder. We have to use the 4d-surrogate's _build_pyport_models
    # to get a consistent (cfg, M1, M2, bjt) tuple.
    sp = importlib.util.spec_from_file_location(
        "ns4d", ROOT / "scripts/nsram_surrogate_4d.py")
    ns4d = importlib.util.module_from_spec(sp); sp.loader.exec_module(ns4d)
    cfg, M1, M2, bjt = ns4d._build_pyport_models()
    # MEP-3 critical knob: route the parasitic P-body diode to V_Nwell.
    cfg.body_pdiode_to = "vnwell"
    return cfg, M1, M2, bjt


def main():
    out_dir = ROOT / "results/z280_surr_v4_100k"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[z280] PyTorch {torch.__version__}, cuda_avail={torch.cuda.is_available()}, "
          f"dev_count={torch.cuda.device_count()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm not available; HSA_OVERRIDE_GFX_VERSION=11.0.0?")
    device = torch.device("cuda")

    cfg, M1, M2, bjt = _build_models()
    print(f"[z280] models built; body_pdiode_to={cfg.body_pdiode_to}")

    from nsram.bsim4_port.forward_2t_batched_gpu import forward_2t_gpu_batched

    rng = np.random.default_rng(SEED)
    # Pre-generate per-V_Nwell-batch random VG1/VG2/Vb_init.
    # Different RNG draws per batch; total = N_VNWELL * N_CELLS = 10000 cells.
    VG1_all  = rng.uniform(*VG1_RANGE, size=(N_VNWELL, N_CELLS)).astype(np.float64)
    VG2_all  = rng.uniform(*VG2_RANGE, size=(N_VNWELL, N_CELLS)).astype(np.float64)
    Vb_all   = rng.uniform(*VB_RANGE,  size=(N_VNWELL, N_CELLS)).astype(np.float64)
    Vd_seq_np = VD_AXIS.copy()

    # Output flat arrays of length 100_000.
    n_total = N_VNWELL * N_CELLS * N_VD
    Id_flat    = np.full(n_total, np.nan, dtype=np.float64)
    Iii_flat   = np.zeros(n_total, dtype=np.float64)
    Ileak_flat = np.zeros(n_total, dtype=np.float64)
    conv_flat  = np.zeros(n_total, dtype=bool)
    vg1_flat   = np.zeros(n_total, dtype=np.float64)
    vg2_flat   = np.zeros(n_total, dtype=np.float64)
    vd_flat    = np.zeros(n_total, dtype=np.float64)
    vb_init_flat = np.zeros(n_total, dtype=np.float64)
    vnwell_flat  = np.zeros(n_total, dtype=np.float64)

    t0 = time.time()
    Vd_seq_t = torch.from_numpy(Vd_seq_np).to(device=device, dtype=torch.float64)

    for m, vnwell in enumerate(VNWELL_AXIS):
        cfg.vnwell = float(vnwell)
        VG1_t = torch.from_numpy(VG1_all[m]).to(device=device, dtype=torch.float64)
        VG2_t = torch.from_numpy(VG2_all[m]).to(device=device, dtype=torch.float64)
        Vb_t  = torch.from_numpy(Vb_all[m]).to(device=device, dtype=torch.float64)

        tb = time.time()
        try:
            out = forward_2t_gpu_batched(
                cfg, M1, M2, bjt,
                Vd_seq=Vd_seq_t,
                VG1_arr=VG1_t, VG2_arr=VG2_t,
                Vb_init=Vb_t,
                max_iters=30, tol=1e-10,
                dtype=torch.float64, device=device,
                early_stop=False, verbose=False,
            )
        except Exception as e:
            print(f"[z280] BATCH {m} (V_Nwell={vnwell:.3f}) FAILED: {e!r}")
            continue
        wall = time.time() - tb

        Id_b    = out["Id"].detach().cpu().numpy()        # (N_CELLS, N_VD)
        Iii_b   = out["Iii_in"].detach().cpu().numpy()
        Ile_b   = out["Ileak_out"].detach().cpu().numpy()
        conv_b  = out["converged"].detach().cpu().numpy()

        # Flatten into the global arrays. Row-major: (m, n, t) -> idx
        base = m * N_CELLS * N_VD
        for n in range(N_CELLS):
            sl = slice(base + n * N_VD, base + (n + 1) * N_VD)
            Id_flat[sl]    = Id_b[n]
            Iii_flat[sl]   = Iii_b[n]
            Ileak_flat[sl] = Ile_b[n]
            conv_flat[sl]  = conv_b[n]
            vg1_flat[sl]   = VG1_all[m, n]
            vg2_flat[sl]   = VG2_all[m, n]
            vb_init_flat[sl] = Vb_all[m, n]
            vd_flat[sl]    = Vd_seq_np
            vnwell_flat[sl] = vnwell

        n_conv = int(conv_b.sum())
        print(f"[z280] batch {m+1}/{N_VNWELL} V_Nwell={vnwell:.3f}: "
              f"wall={wall:.1f}s, conv={n_conv}/{N_CELLS*N_VD}")

    wall_total = time.time() - t0
    n_done = int(np.isfinite(Id_flat).sum())
    n_conv = int(conv_flat.sum())
    print(f"\n[z280] DONE in {wall_total:.0f}s; valid Id rows={n_done}/{n_total}; "
          f"converged={n_conv}/{n_total} ({100*n_conv/n_total:.1f}%)")

    out_path = out_dir / "surrogate_100k.npz"
    np.savez(
        out_path,
        Id=Id_flat, Iii=Iii_flat, Ileak=Ileak_flat, converged=conv_flat,
        VG1=vg1_flat, VG2=vg2_flat, Vd=vd_flat,
        Vb_init=vb_init_flat, V_Nwell=vnwell_flat,
        vnwell_axis=VNWELL_AXIS, vd_axis=VD_AXIS,
    )
    print(f"[z280] wrote {out_path} ({out_path.stat().st_size/1e6:.2f} MB)")

    summary = {
        "n_total": n_total,
        "n_valid": n_done,
        "n_converged": n_conv,
        "conv_rate": n_conv / n_total,
        "wall_s": wall_total,
        "seed": SEED,
        "axes": {
            "V_Nwell": list(map(float, VNWELL_AXIS)),
            "V_d": list(map(float, VD_AXIS)),
            "V_G1_range": list(VG1_RANGE),
            "V_G2_range": list(VG2_RANGE),
            "V_b_range":  list(VB_RANGE),
        },
        "shape": {"N_VNWELL": N_VNWELL, "N_CELLS": N_CELLS, "N_VD": N_VD},
        "body_pdiode_to": cfg.body_pdiode_to,
        "torch": torch.__version__,
        "device": str(device),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[z280] wrote summary.json; verdict: "
          f"{'PASS' if summary['conv_rate'] >= 0.80 else 'FAIL'}")


if __name__ == "__main__":
    main()
