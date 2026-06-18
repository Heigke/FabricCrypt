"""z93 — GPU-first NS-RAM cell solver benchmark.

Audits and exercises `forward_2t_batched` (and `transient_2t`) on AMD
gfx1151 (Radeon 8060S) via ROCm 6.3 / torch HIP backend, comparing CPU
vs GPU vs torch.compile-wrapped GPU at scales N ∈ {1000, 10000, 100000}
with 40 Vd points each. fp64 throughout (HSA_OVERRIDE_GFX_VERSION=11.0.0
required for fp64 to work on gfx1151).

Reports cell-evals/s = (N * Vd_points) / wall-time and saves to
results/z93_gpu_benchmark/summary.json.

Run:
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/z93_gpu_benchmark.py
"""
from __future__ import annotations
import json, os, sys, time, traceback
from pathlib import Path

import torch

torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "nsram"))
DATA = ROOT / "data/sebas_2026_04_22"
OUT = ROOT / "results/z93_gpu_benchmark"
OUT.mkdir(parents=True, exist_ok=True)

from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port import vectorized as _vec
from nsram.bsim4_port.vectorized import forward_2t_batched
from nsram.bsim4_port.joint_newton import transient_2t
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.geometry import Geometry


# --------------------------------------------------------------------- #
# Setup                                                                 #
# --------------------------------------------------------------------- #

def build_models():
    m1 = BSIM4Model.from_spice((DATA / "M1_130DNWFB.txt").read_text(),
                                  model_type="nmos")
    m2 = BSIM4Model.from_spice((DATA / "M2_130bulkNSRAM.txt").read_text(),
                                  model_type="nmos")
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                              newton_max_iters=30)
    sd_M1 = compute_size_dep(m1, Geometry(L=cfg.Ln, W=cfg.Wn), T_C=cfg.T_C)
    sd_M2 = compute_size_dep(m2,
                                Geometry(L=cfg.Ln * cfg.M2_length_factor,
                                         W=cfg.Wn), T_C=cfg.T_C)
    cfg._sd_M1 = sd_M1
    cfg._sd_M2 = sd_M2
    bjt = GummelPoonNPN()
    return cfg, m1, m2, bjt


def make_inputs(N, T, device):
    # Spread VG1/VG2 over a physically interesting range so Newton
    # actually does work on each cell.
    VG1 = torch.linspace(0.20, 0.55, N, dtype=torch.float64, device=device)
    VG2 = torch.linspace(0.30, 0.70, N, dtype=torch.float64, device=device)
    Vd = torch.linspace(0.0, 2.0, T, dtype=torch.float64, device=device)
    return Vd, VG1, VG2


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def time_call(fn, device, repeats=1):
    sync(device)
    t0 = time.perf_counter()
    for _ in range(repeats):
        out = fn()
    sync(device)
    t1 = time.perf_counter()
    return (t1 - t0) / repeats, out


# --------------------------------------------------------------------- #
# torch.compile wrapping (best-effort)                                  #
# --------------------------------------------------------------------- #

def try_compile_residuals(mode: str = "default"):
    """Wrap _residuals in torch.compile for GPU kernel-launch reduction.

    `mode="reduce-overhead"` uses CUDA graphs and ran into "accessing
    tensor output of CUDAGraphs that has been overwritten" because the
    Newton loop calls _residuals 5x per step and the previous output
    is still aliased when the next call lands. We default to
    `mode="default"` (Inductor codegen, no cudagraphs) which still
    fuses pointwise ops; this avoids the alias problem at the cost of
    leaving Python-launch overhead between fused kernels.
    """
    from nsram.bsim4_port import nsram_cell_2T as _cell
    raw = _cell._residuals
    try:
        compiled = torch.compile(raw, mode=mode, dynamic=False,
                                   fullgraph=False)
        return compiled
    except Exception as exc:
        print(f"[compile] failed to build compiled _residuals: {exc}")
        return None


def install_compiled(compiled):
    """Patch _residuals in vectorized.py to use the compiled version."""
    from nsram.bsim4_port import nsram_cell_2T as _cell
    _vec._residuals = compiled
    _cell._residuals = compiled


def restore_residuals(orig):
    from nsram.bsim4_port import nsram_cell_2T as _cell
    _vec._residuals = orig
    _cell._residuals = orig


# --------------------------------------------------------------------- #
# Benchmarks                                                            #
# --------------------------------------------------------------------- #

def bench_forward(cfg, m1, m2, bjt, N, T, device, label, warmup=1, reps=1):
    Vd, VG1, VG2 = make_inputs(N, T, device)
    fn = lambda: forward_2t_batched(cfg, m1, m2, bjt, Vd, VG1, VG2,
                                       max_iters=30)
    # warmup
    for _ in range(warmup):
        sync(device)
        try:
            _ = fn()
        except Exception as exc:
            return {"label": label, "device": device, "N": N, "T": T,
                      "ok": False, "error": repr(exc)}
        sync(device)
    try:
        dt, out = time_call(fn, device, repeats=reps)
    except Exception as exc:
        return {"label": label, "device": device, "N": N, "T": T,
                  "ok": False, "error": repr(exc)}
    cell_evals = N * T
    cps = cell_evals / dt
    conv = float(out["converged"].float().mean().item())
    print(f"  [{label:>20s}] N={N:>6d} T={T} "
          f"time={dt:.4f}s  throughput={cps:>12.1f} cell-evals/s  "
          f"conv={conv*100:.1f}%")
    return {"label": label, "device": device, "N": N, "T": T,
              "ok": True, "time_s": dt, "throughput_cells_per_s": cps,
              "converged_frac": conv, "reps": reps}


def bench_transient(cfg, m1, m2, bjt, n_steps, device, label):
    # Single-cell Vd ramp (sequential implicit Euler — port-to-GPU
    # value test, not a parallel sweep).
    t = torch.linspace(0.0, 1e-6, n_steps, dtype=torch.float64, device=device)
    Vd_t = torch.full((n_steps,), 1.5, dtype=torch.float64, device=device)
    VG1 = torch.tensor(0.40, dtype=torch.float64, device=device)
    VG2 = torch.tensor(0.50, dtype=torch.float64, device=device)
    sync(device)
    t0 = time.perf_counter()
    try:
        out = transient_2t(cfg, m1, m2, bjt, Vd_t, t, VG1, VG2,
                              newton_iters=20)
    except Exception as exc:
        return {"label": label, "device": device, "n_steps": n_steps,
                  "ok": False, "error": repr(exc)}
    sync(device)
    dt = time.perf_counter() - t0
    sps = n_steps / dt
    print(f"  [{label:>20s}] n_steps={n_steps} "
          f"time={dt:.3f}s  throughput={sps:.1f} steps/s  "
          f"spikes={len(out['spike_times'])}")
    return {"label": label, "device": device, "n_steps": n_steps,
              "ok": True, "time_s": dt, "steps_per_s": sps,
              "n_spikes": len(out["spike_times"])}


# --------------------------------------------------------------------- #
# Main                                                                  #
# --------------------------------------------------------------------- #

def main():
    print(f"[z93] torch={torch.__version__}  "
          f"hip={getattr(torch.version, 'hip', 'n/a')}  "
          f"cuda_avail={torch.cuda.is_available()}  "
          f"HSA_OVERRIDE={os.environ.get('HSA_OVERRIDE_GFX_VERSION', '<unset>')}")
    if torch.cuda.is_available():
        print(f"[z93] gpu: {torch.cuda.get_device_name(0)}")

    cfg, m1, m2, bjt = build_models()

    sizes = [1000, 10000, 100000]
    T = 40
    have_gpu = torch.cuda.is_available()

    summary = {"env": {"torch": torch.__version__,
                        "hip": getattr(torch.version, "hip", None),
                        "cuda_available": have_gpu,
                        "device_name": (torch.cuda.get_device_name(0)
                                          if have_gpu else None),
                        "hsa_override": os.environ.get(
                            "HSA_OVERRIDE_GFX_VERSION")},
                 "forward_2t_batched": [],
                 "transient_2t": []}

    # ---- forward_2t_batched: CPU baseline (only N=1000; N=100k CPU
    # would take many minutes — we report CPU only at the small size
    # for an apples-to-apples comparison).
    print("\n[z93] forward_2t_batched — CPU baseline (small N)")
    summary["forward_2t_batched"].append(
        bench_forward(cfg, m1, m2, bjt, 1000, T, "cpu", "cpu", warmup=0, reps=1))

    if have_gpu:
        # ---- GPU eager
        print("\n[z93] forward_2t_batched — GPU eager (uncompiled)")
        for N in sizes:
            summary["forward_2t_batched"].append(
                bench_forward(cfg, m1, m2, bjt, N, T, "cuda", "gpu_eager",
                              warmup=1, reps=2 if N <= 10000 else 1))

        # ---- GPU compiled
        from nsram.bsim4_port import nsram_cell_2T as _cell
        orig = _cell._residuals
        for mode in ("default", "reduce-overhead"):
            print(f"\n[z93] forward_2t_batched — GPU torch.compile (mode={mode})")
            compiled = try_compile_residuals(mode=mode)
            if compiled is None:
                print(f"  [compile/{mode}] skipped: compile build failed")
                summary[f"compile_status_{mode}"] = "build_failed"
                continue
            install_compiled(compiled)
            label = f"gpu_compile_{mode}"
            any_fail = False
            try:
                for N in sizes:
                    rec = bench_forward(cfg, m1, m2, bjt, N, T, "cuda",
                                          label, warmup=2,
                                          reps=2 if N <= 10000 else 1)
                    summary["forward_2t_batched"].append(rec)
                    if not rec.get("ok"):
                        any_fail = True
                        print(f"  [compile/{mode}] N={N} FAILED: "
                              f"{rec.get('error', '')[:200]}")
                summary[f"compile_status_{mode}"] = (
                    "partial_fail" if any_fail else "ok")
            except Exception as exc:
                print(f"  [compile/{mode}] runtime error: {exc}")
                traceback.print_exc()
                summary[f"compile_status_{mode}"] = f"runtime_error: {exc!r}"
            finally:
                restore_residuals(orig)
                # Force a fresh dynamo/inductor cache between modes
                try:
                    torch._dynamo.reset()
                except Exception:
                    pass

    # ---- transient_2t: single-cell, 1000 timesteps
    print("\n[z93] transient_2t — single-cell sequential (CPU vs GPU)")
    summary["transient_2t"].append(
        bench_transient(cfg, m1, m2, bjt, 1000, "cpu", "cpu"))
    if have_gpu:
        summary["transient_2t"].append(
            bench_transient(cfg, m1, m2, bjt, 1000, "cuda", "gpu_eager"))

    # ---- Summary table
    print("\n[z93] === SUMMARY ===")
    print(f"{'label':>18s} {'dev':>6s} {'N':>7s} {'T':>4s} "
          f"{'time_s':>10s} {'cell-evals/s':>16s}")
    for r in summary["forward_2t_batched"]:
        if not r.get("ok"): continue
        print(f"{r['label']:>18s} {r['device']:>6s} {r['N']:>7d} {r['T']:>4d} "
              f"{r['time_s']:>10.4f} {r['throughput_cells_per_s']:>16.1f}")

    out_path = OUT / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[z93] wrote {out_path}")


if __name__ == "__main__":
    main()
