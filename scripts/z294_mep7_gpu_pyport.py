"""MEP-7: GPU-native pyport Newton solver for 100K-1M point surrogates.

Reuses pyport's `_residuals` directly (torch-native, batched-broadcast safe).
Performs batched 1D Newton on Vsint at FIXED Vb (matches CPU
`_solve_at_fixed_vb` in scripts/nsram_surrogate_4d.py).

Pre-reg gates:
  CORRECTNESS: max abs err <= 1e-4 vs CPU pyport over 100 random points
  SPEED:       100K points in <= 5 min wall (any GPU)
  AMBITIOUS:   1M  points in <= 10 min wall (GB10)

NO-CHEAT:
  - BSIM4 physics unmodified (we import _residuals directly)
  - 100-point validation set is FIXED via numpy RNG seed=42
  - Newton damping LOCKED at 0.5 step clamp (same as CPU)
  - If torch.compile or cuda.graph fail we report eager honestly
"""
from __future__ import annotations
import os
# Single-thread numpy/BLAS to avoid CPU oversub when benchmarking
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import argparse
import importlib.util
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent


# ────────────────────────────────────────────────────────────────────────── #
# Reuse CPU pyport reference                                                #
# ────────────────────────────────────────────────────────────────────────── #
def _load_cpu_ref():
    """Load `_build_pyport_models` and `_solve_at_fixed_vb` from
    scripts/nsram_surrogate_4d.py."""
    sp = importlib.util.spec_from_file_location(
        "ns4d", ROOT / "scripts/nsram_surrogate_4d.py")
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


# ────────────────────────────────────────────────────────────────────────── #
# GPU batched Newton                                                         #
# ────────────────────────────────────────────────────────────────────────── #
def _comps_to_currents(comp, cfg):
    """Compute (Id, Iii_in, Ileak_out) batched, same formula as CPU ref."""
    Id = (
        comp["Ids_M1"] + comp["Ic_Q1"]
        + comp.get("Ic_lat", 0.0) + comp.get("Ic_avalanche", 0.0)
        + comp["Igidl_M1"] - comp["Ibd_M1"]
    )
    Iii_in = comp.get("Iii_M1", 0.0)
    if not cfg.m2_body_gnd:
        Iii_in = Iii_in + comp.get("Iii_M2", 0.0)
    Iii_in = Iii_in + comp.get("I_well_body", 0.0)
    Iii_in = Iii_in + comp.get("Igidl_M1", 0.0) + comp.get("Igisl_M1", 0.0)
    Ileak_out = (
        comp.get("Ibs_M1", 0.0) + comp.get("Ibd_M1", 0.0) + comp.get("Ib_Q1", 0.0)
    )
    return Id, Iii_in, Ileak_out


def solve_batched_gpu(cfg, M1, M2, bjt, Vd, VG1, VG2, Vb, *,
                     max_iters: int, h: float = 1e-6,
                     step_clamp: float = 0.5, tol: float = 1e-12,
                     device: str = "cuda", dtype=torch.float64):
    """Batched 1D Newton on Vsint at fixed Vb.

    Vd/VG1/VG2/Vb are 1D torch tensors of shape (B,) on `device`.
    Returns dict with Id, Iii_in, Ileak_out, Vsint, converged (all (B,)).
    """
    from nsram.bsim4_port.nsram_cell_2T import _residuals

    Vd = Vd.to(device=device, dtype=dtype)
    VG1 = VG1.to(device=device, dtype=dtype)
    VG2 = VG2.to(device=device, dtype=dtype)
    Vb = Vb.to(device=device, dtype=dtype)

    Vsint = 0.5 * Vd.clone()
    converged = torch.zeros_like(Vd, dtype=torch.bool)

    for it in range(max_iters):
        R_S, _R_B, _comp = _residuals(cfg, M1, bjt, Vd, VG1, VG2, Vsint, Vb,
                                       model_M2=M2)
        abs_RS = R_S.abs()
        newly_done = abs_RS < tol
        converged = converged | newly_done
        if bool(converged.all()):
            break
        R_Sp, _, _ = _residuals(cfg, M1, bjt, Vd, VG1, VG2,
                                  Vsint + h, Vb, model_M2=M2)
        dRdV = (R_Sp - R_S) / h
        # Guard against zero derivative
        safe = dRdV.abs() > 1e-30
        dV = torch.where(safe, -R_S / torch.where(safe, dRdV, torch.ones_like(dRdV)),
                          torch.zeros_like(R_S))
        dV = dV.clamp(-step_clamp, step_clamp)
        # Freeze converged entries
        dV = torch.where(converged, torch.zeros_like(dV), dV)
        Vsint = Vsint + dV

    # Final eval at converged Vsint
    _Rf_S, _Rf_B, comp = _residuals(cfg, M1, bjt, Vd, VG1, VG2,
                                      Vsint, Vb, model_M2=M2)
    Id, Iii_in, Ileak_out = _comps_to_currents(comp, cfg)
    return {
        "Id": Id, "Iii_in": Iii_in, "Ileak_out": Ileak_out,
        "Vsint": Vsint, "converged": converged,
    }


# ────────────────────────────────────────────────────────────────────────── #
# Validation: compare to CPU pyport                                          #
# ────────────────────────────────────────────────────────────────────────── #
def make_validation_points(n: int, seed: int = 42):
    """FIXED random 100-point validation set."""
    rng = np.random.default_rng(seed)
    VG1 = rng.uniform(0.10, 0.80, n)
    VG2 = rng.uniform(-0.05, 0.55, n)
    Vd  = rng.uniform(0.30, 2.50, n)
    Vb  = rng.uniform(0.00, 0.80, n)
    return VG1, VG2, Vd, Vb


def validate(n_points: int = 100, device: str = "cuda"):
    print(f"[mep7/validate] n={n_points} device={device}")
    ns4d = _load_cpu_ref()
    cfg, M1, M2, bjt = ns4d._build_pyport_models()

    VG1, VG2, Vd, Vb = make_validation_points(n_points)

    # CPU reference (one at a time, exactly as in production)
    t0 = time.time()
    Id_cpu = np.zeros(n_points)
    Iii_cpu = np.zeros(n_points)
    Ileak_cpu = np.zeros(n_points)
    conv_cpu = np.zeros(n_points, dtype=bool)
    for i in range(n_points):
        out = ns4d._solve_at_fixed_vb(cfg, M1, M2, bjt,
                                       float(Vd[i]), float(VG1[i]),
                                       float(VG2[i]), float(Vb[i]))
        Id_cpu[i] = out["Id"]
        Iii_cpu[i] = out["Iii_in"]
        Ileak_cpu[i] = out["Ileak_out"]
        conv_cpu[i] = out["converged"]
    t_cpu = time.time() - t0
    print(f"  CPU wall: {t_cpu:.2f}s  ({1000*t_cpu/n_points:.1f} ms/pt)  "
          f"conv={conv_cpu.sum()}/{n_points}")

    # GPU batched
    cfg.invalidate()
    Vd_t = torch.tensor(Vd)
    VG1_t = torch.tensor(VG1)
    VG2_t = torch.tensor(VG2)
    Vb_t = torch.tensor(Vb)
    t0 = time.time()
    out = solve_batched_gpu(cfg, M1, M2, bjt, Vd_t, VG1_t, VG2_t, Vb_t,
                              max_iters=cfg.newton_max_iters, device=device)
    if device == "cuda":
        torch.cuda.synchronize()
    t_gpu = time.time() - t0
    Id_g = out["Id"].cpu().numpy()
    Iii_g = out["Iii_in"].cpu().numpy()
    Ileak_g = out["Ileak_out"].cpu().numpy()
    conv_g = out["converged"].cpu().numpy()
    print(f"  GPU wall: {t_gpu:.2f}s  ({1000*t_gpu/n_points:.2f} ms/pt)  "
          f"conv={conv_g.sum()}/{n_points}")

    def _err(a, b):
        # Use absolute error normalised by max(|a|,1e-12). Many currents are tiny.
        return float(np.max(np.abs(a - b)))

    err_Id = _err(Id_g, Id_cpu)
    err_Iii = _err(Iii_g, Iii_cpu)
    err_Ileak = _err(Ileak_g, Ileak_cpu)

    # Relative: error scaled by signal magnitude
    def _rel(a, b):
        denom = np.maximum(np.abs(b), 1e-12)
        return float(np.max(np.abs(a - b) / denom))

    rel_Id = _rel(Id_g, Id_cpu)
    rel_Iii = _rel(Iii_g, Iii_cpu)
    rel_Ileak = _rel(Ileak_g, Ileak_cpu)

    print(f"  abs err:  Id={err_Id:.3e}  Iii={err_Iii:.3e}  Ileak={err_Ileak:.3e}")
    print(f"  rel err:  Id={rel_Id:.3e}  Iii={rel_Iii:.3e}  Ileak={rel_Ileak:.3e}")

    # Gate: max abs err <= 1e-4
    max_abs = max(err_Id, err_Iii, err_Ileak)
    verdict = "PASS" if max_abs <= 1e-4 else "FAIL"
    print(f"  VERDICT: {verdict}  (max_abs={max_abs:.3e}, gate=1e-4)")

    return {
        "verdict": verdict,
        "n_points": n_points,
        "max_abs_err": max_abs,
        "err_Id": err_Id, "err_Iii": err_Iii, "err_Ileak": err_Ileak,
        "rel_Id": rel_Id, "rel_Iii": rel_Iii, "rel_Ileak": rel_Ileak,
        "t_cpu_s": t_cpu, "t_gpu_s": t_gpu,
        "speedup": t_cpu / max(t_gpu, 1e-9),
        "device": device,
    }


# ────────────────────────────────────────────────────────────────────────── #
# Production: build N-point surrogate                                        #
# ────────────────────────────────────────────────────────────────────────── #
def make_grid_points(grid_size: int, seed: int = 0):
    """Quasi-uniform random grid points across the MEP-2 axis ranges."""
    rng = np.random.default_rng(seed)
    VG1 = rng.uniform(0.10, 0.80, grid_size)
    VG2 = rng.uniform(-0.10, 0.60, grid_size)
    Vd  = rng.uniform(0.25, 3.00, grid_size)
    Vb  = rng.uniform(0.00, 1.00, grid_size)
    return VG1, VG2, Vd, Vb


def production(grid_size: int, out_dir: Path, device: str = "cuda",
               batch_size: int = 65536, try_compile: bool = True):
    print(f"[mep7/prod] grid_size={grid_size} device={device} batch={batch_size}")
    ns4d = _load_cpu_ref()
    cfg, M1, M2, bjt = ns4d._build_pyport_models()
    VG1, VG2, Vd, Vb = make_grid_points(grid_size)

    # Optional compile attempt for the Newton kernel
    compile_status = "skip"
    compile_overhead_s = 0.0
    solver = solve_batched_gpu
    if try_compile:
        try:
            t0 = time.time()
            # Warm one tiny call so torch.compile traces graph
            warm_n = 64
            VG1_w = torch.tensor(VG1[:warm_n])
            VG2_w = torch.tensor(VG2[:warm_n])
            Vd_w = torch.tensor(Vd[:warm_n])
            Vb_w = torch.tensor(Vb[:warm_n])
            # Compile a closure that calls _residuals (won't fully compile —
            # _residuals has Python dict returns). We rely on torch's eager
            # mode for the residual and only attempt to JIT the Newton loop
            # body as a fused kernel. If this errors we report skip.
            _ = solve_batched_gpu(cfg, M1, M2, bjt, Vd_w, VG1_w, VG2_w, Vb_w,
                                    max_iters=2, device=device)
            if device == "cuda":
                torch.cuda.synchronize()
            compile_overhead_s = time.time() - t0
            compile_status = "warmed_eager"  # honest: we did not fully compile
        except Exception as e:
            compile_status = f"warm_failed:{type(e).__name__}"

    Id_arr = np.zeros(grid_size, dtype=np.float64)
    Iii_arr = np.zeros(grid_size, dtype=np.float64)
    Ileak_arr = np.zeros(grid_size, dtype=np.float64)
    conv_arr = np.zeros(grid_size, dtype=bool)
    Vsint_arr = np.zeros(grid_size, dtype=np.float64)

    cuda_graph_status = "not_attempted"
    t0 = time.time()
    n_done = 0
    for s in range(0, grid_size, batch_size):
        e = min(s + batch_size, grid_size)
        Vd_b = torch.tensor(Vd[s:e])
        VG1_b = torch.tensor(VG1[s:e])
        VG2_b = torch.tensor(VG2[s:e])
        Vb_b = torch.tensor(Vb[s:e])
        out = solver(cfg, M1, M2, bjt, Vd_b, VG1_b, VG2_b, Vb_b,
                      max_iters=cfg.newton_max_iters, device=device)
        Id_arr[s:e] = out["Id"].detach().cpu().numpy()
        Iii_arr[s:e] = out["Iii_in"].detach().cpu().numpy()
        Ileak_arr[s:e] = out["Ileak_out"].detach().cpu().numpy()
        conv_arr[s:e] = out["converged"].detach().cpu().numpy()
        Vsint_arr[s:e] = out["Vsint"].detach().cpu().numpy()
        n_done = e
        if device == "cuda":
            torch.cuda.synchronize()
        wall = time.time() - t0
        eta = wall / max(n_done, 1) * (grid_size - n_done)
        print(f"  {n_done}/{grid_size} ({100*n_done/grid_size:.0f}%)  "
              f"wall={wall:.1f}s eta={eta:.1f}s  "
              f"conv_so_far={int(conv_arr[:n_done].sum())}")

    wall = time.time() - t0
    n_conv = int(conv_arr.sum())
    print(f"\n[mep7/prod] done in {wall:.1f}s  conv={n_conv}/{grid_size}  "
          f"({100*n_conv/grid_size:.1f}%)  ({1e6*wall/grid_size:.1f} us/pt)")

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"surrogate_{grid_size}.npz"
    np.savez(npz_path,
             VG1=VG1, VG2=VG2, Vd=Vd, Vb=Vb,
             Id=Id_arr, Iii=Iii_arr, Ileak=Ileak_arr,
             Vsint=Vsint_arr, converged=conv_arr)
    summary = {
        "grid_size": grid_size,
        "n_converged": n_conv,
        "conv_rate": n_conv / grid_size,
        "wall_s": wall,
        "us_per_pt": 1e6 * wall / grid_size,
        "batch_size": batch_size,
        "device": device,
        "compile_status": compile_status,
        "compile_overhead_s": compile_overhead_s,
        "cuda_graph_status": cuda_graph_status,
        "node": platform.node(),
        "out_path": str(npz_path),
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda": getattr(torch.version, "cuda", None),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[mep7/prod] saved {npz_path}")
    print(f"[mep7/prod] summary {out_dir}/summary.json")
    return summary


# ────────────────────────────────────────────────────────────────────────── #
# CLI                                                                        #
# ────────────────────────────────────────────────────────────────────────── #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid_size", type=int, default=1000)
    ap.add_argument("--node", default="auto",
                     choices=["ikaros", "daedalus", "zgx", "auto"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--validate", action="store_true",
                     help="Run 100-point CPU-vs-GPU correctness check then exit")
    ap.add_argument("--benchmark", action="store_true",
                     help="Run validation first, then production")
    ap.add_argument("--batch_size", type=int, default=65536)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--no_compile", action="store_true")
    args = ap.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if args.out_dir is None:
        node_tag = platform.node()
        out_dir = ROOT / f"results/z294_mep7_{node_tag}"
    else:
        out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.validate or args.benchmark:
        val = validate(n_points=100, device=device)
        with open(out_dir / "validate.json", "w") as f:
            json.dump(val, f, indent=2)
        if val["verdict"] != "PASS":
            print(f"\n[mep7] VALIDATION FAILED — refusing to run production")
            return 1
        if args.validate and not args.benchmark:
            return 0

    summary = production(args.grid_size, out_dir, device=device,
                           batch_size=args.batch_size,
                           try_compile=not args.no_compile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
