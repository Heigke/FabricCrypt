"""S1 MEP-7 GPU-batched forward_2t — validation + benchmark.

Runs on zgx (NVIDIA GB10) with /home/naorw/nsram_venv. Produces:
  results/S1_mep7_gpu_batched/validation_vs_single.json
  results/S1_mep7_gpu_batched/benchmark.json

Gates:
  INFRA:      N=10K batched result matches existing forward_2t_batched
              (vectorized.py) within 1e-6 rel on Id (warm-start identical).
  PASS:       N=1M cells × T=33 Vd points complete in <60s.
  AMBITIOUS:  N=1M cells × T=33 Vd points in <10s with torch.compile.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))

import numpy as np
import torch

# ── builders ────────────────────────────────────────────────────────────── #
def _build_models():
    import importlib.util
    sp = importlib.util.spec_from_file_location("ns4d", ROOT / "scripts/nsram_surrogate_4d.py")
    mod = importlib.util.module_from_spec(sp); sp.loader.exec_module(mod)
    return mod._build_pyport_models()  # (cfg, M1, M2, bjt)


def _model_to_device(M, device, dtype):
    """BSIM4Model / GummelPoonNPN store scalar Python floats in _values dict
    (BSIM4Model has __slots__; bjt has dataclass scalars). `_residuals`
    creates tensors on the device of the input voltages, so no model-side
    movement is needed. We keep this stub for backward compat in calls.
    """
    return M


def _bjt_to_device(B, device, dtype):
    return B


# ── validation ──────────────────────────────────────────────────────────── #
def validate(cfg, M1, M2, bjt, *, device="cuda", dtype=torch.float64, N=10_000):
    from nsram.bsim4_port.forward_2t_batched_gpu import forward_2t_gpu_batched
    from nsram.bsim4_port.vectorized import forward_2t_batched

    rng = np.random.default_rng(0)
    VG1 = rng.uniform(0.10, 0.80, N)
    VG2 = rng.uniform(-0.05, 0.55, N)
    Vd_seq = torch.linspace(0.0, 2.0, 33, dtype=dtype)

    VG1_t = torch.tensor(VG1, dtype=dtype)
    VG2_t = torch.tensor(VG2, dtype=dtype)

    # Existing batched (vectorized.py) reference — run on CPU for trust.
    t0 = time.time()
    ref = forward_2t_batched(cfg, M1, M2, bjt, Vd_seq, VG1_t, VG2_t,
                              max_iters=30, tol=1e-12,
                              Vsint0=0.1, Vb0=0.3,
                              damping=1.0, verbose=False)
    t_ref = time.time() - t0

    # GPU version
    _model_to_device(M1, device, dtype)
    _model_to_device(M2, device, dtype)
    _bjt_to_device(bjt, device, dtype)

    if torch.cuda.is_available() and "cuda" in str(device):
        torch.cuda.synchronize()
    t0 = time.time()
    out = forward_2t_gpu_batched(cfg, M1, M2, bjt,
                                  Vd_seq=Vd_seq, VG1_arr=VG1_t, VG2_arr=VG2_t,
                                  max_iters=30, tol=1e-12,
                                  damping=1.0, dtype=dtype, device=device,
                                  compile_mode="off", early_stop=False)
    if torch.cuda.is_available() and "cuda" in str(device):
        torch.cuda.synchronize()
    t_gpu = time.time() - t0

    Id_ref = ref["Id"].cpu().double()
    Id_gpu = out["Id"].cpu().double()

    abs_err = (Id_gpu - Id_ref).abs()
    rel = abs_err / (Id_ref.abs() + 1e-15)
    # Mask tiny currents from rel error (numerical noise)
    big = Id_ref.abs() > 1e-12
    # Only compare where BOTH ref and GPU converged
    conv_both = ref["converged"].cpu() & out["converged"].cpu()
    mask_strict = big & conv_both
    rel_big = rel[big] if big.any() else torch.tensor([0.0])
    rel_strict = rel[mask_strict] if mask_strict.any() else torch.tensor([0.0])
    abs_strict = abs_err[mask_strict] if mask_strict.any() else torch.tensor([0.0])

    result = {
        "N": int(N), "T": int(Vd_seq.numel()),
        "max_abs_err_Id": float(abs_err.max()),
        "median_abs_err_Id": float(abs_err.median()),
        "max_rel_err_big": float(rel_big.max()),
        "p99_rel_err_big": float(rel_big.quantile(0.99)),
        "max_rel_err_strict_converged": float(rel_strict.max()),
        "p99_rel_err_strict_converged": float(rel_strict.quantile(0.99)),
        "max_abs_err_strict_converged": float(abs_strict.max()),
        "n_strict_compared": int(mask_strict.sum()),
        "n_total_big": int(big.sum()),
        "frac_Id_above_1pA": float(big.float().mean()),
        "t_ref_cpu_s": t_ref,
        "t_gpu_s": t_gpu,
        "speedup_vs_cpu_batched": t_ref / max(t_gpu, 1e-9),
        "conv_frac_ref": float(ref["converged"].float().mean()),
        "conv_frac_gpu": float(out["converged"].float().mean()),
        "PASS_rel_err_1e6": bool(float(rel_strict.max()) < 1e-6),
    }
    return result


# ── benchmark ───────────────────────────────────────────────────────────── #
def bench(cfg, M1, M2, bjt, *, device="cuda", dtype=torch.float64,
          sizes=(1, 100, 10_000, 100_000, 1_000_000),
          T=33, max_iters=20, compile_mode="off", early_stop=True):
    from nsram.bsim4_port.forward_2t_batched_gpu import forward_2t_gpu_batched

    _model_to_device(M1, device, dtype)
    _model_to_device(M2, device, dtype)
    _bjt_to_device(bjt, device, dtype)

    rng = np.random.default_rng(1)
    Vd_seq = torch.linspace(0.0, 2.0, T, dtype=dtype, device=device)

    results = []
    for N in sizes:
        VG1 = torch.tensor(rng.uniform(0.10, 0.80, N), dtype=dtype, device=device)
        VG2 = torch.tensor(rng.uniform(-0.05, 0.55, N), dtype=dtype, device=device)

        # Warmup (compile / lazy init)
        try:
            _ = forward_2t_gpu_batched(cfg, M1, M2, bjt,
                                        Vd_seq=Vd_seq[:2], VG1_arr=VG1[:min(N,16)],
                                        VG2_arr=VG2[:min(N,16)],
                                        max_iters=3, dtype=dtype, device=device,
                                        compile_mode=compile_mode)
        except Exception as e:
            print(f"  [N={N}] warmup error: {e!r}")

        if "cuda" in str(device): torch.cuda.synchronize()
        if compile_mode != "off":
            try: torch.compiler.cudagraph_mark_step_begin()
            except Exception: pass
        t0 = time.time()
        try:
            out = forward_2t_gpu_batched(cfg, M1, M2, bjt,
                                          Vd_seq=Vd_seq, VG1_arr=VG1, VG2_arr=VG2,
                                          max_iters=max_iters, dtype=dtype,
                                          device=device, compile_mode=compile_mode,
                                          early_stop=early_stop)
            if "cuda" in str(device): torch.cuda.synchronize()
            dt = time.time() - t0
            conv = float(out["converged"].float().mean())
            results.append({
                "N": int(N), "T": int(T), "wall_s": dt,
                "ns_per_cell_step": dt * 1e9 / (N * T),
                "converged_frac": conv,
                "used_compile": bool(out.get("_used_compile", False)),
                "ok": True,
            })
            print(f"  N={N:>9d}  wall={dt:7.3f}s  conv={conv*100:5.1f}%  "
                  f"ns/cell·step={dt*1e9/(N*T):.1f}")
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            dt = time.time() - t0
            results.append({"N": int(N), "T": int(T), "wall_s": dt,
                             "error": repr(e)[:200], "ok": False})
            print(f"  N={N:>9d}  FAILED ({e})")
            if "cuda" in str(device): torch.cuda.empty_cache()

        del VG1, VG2
        if "cuda" in str(device): torch.cuda.empty_cache()
    return results


# ── main ────────────────────────────────────────────────────────────────── #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("validate", "bench", "all"), default="all")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float64", choices=("float64", "float32"))
    ap.add_argument("--sizes", type=int, nargs="*",
                    default=[1, 100, 10000, 100000, 1000000])
    ap.add_argument("--T", type=int, default=33)
    ap.add_argument("--max_iters", type=int, default=20)
    ap.add_argument("--validate_N", type=int, default=10_000)
    ap.add_argument("--out_dir", default="results/S1_mep7_gpu_batched")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    print(f"[S1] device={args.device}  dtype={dtype}  "
          f"torch={torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available() and "cuda" in args.device:
        print(f"     GPU={torch.cuda.get_device_name(0)}  "
              f"mem={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    cfg, M1, M2, bjt = _build_models()
    print(f"[S1] cfg.use_bjt={cfg.use_bjt}  use_iii={cfg.use_iii}  "
          f"use_gidl={cfg.use_gidl}  m2_body_gnd={cfg.m2_body_gnd}")

    summary = {"torch": torch.__version__,
                "device": args.device, "dtype": args.dtype}
    if torch.cuda.is_available() and "cuda" in args.device:
        summary["gpu"] = torch.cuda.get_device_name(0)

    if args.mode in ("validate", "all"):
        print(f"\n[validate]  N={args.validate_N} vs forward_2t_batched (CPU)")
        v = validate(cfg, M1, M2, bjt, device=args.device, dtype=dtype, N=args.validate_N)
        summary["validation"] = v
        (out_dir / "validation_vs_single.json").write_text(json.dumps(v, indent=2))
        print(json.dumps(v, indent=2))

    if args.mode in ("bench", "all"):
        # Re-build models fresh (validate may have moved them to device already
        # — re-using is fine since bench also moves).
        cfg, M1, M2, bjt = _build_models()

        all_bench = {}
        for compile_mode in ("off", "default", "reduce-overhead"):
            print(f"\n[bench]  compile_mode={compile_mode}  sizes={args.sizes}")
            try:
                rs = bench(cfg, M1, M2, bjt, device=args.device, dtype=dtype,
                            sizes=tuple(args.sizes), T=args.T,
                            max_iters=args.max_iters, compile_mode=compile_mode)
            except Exception as e:
                rs = [{"compile_mode": compile_mode, "error": repr(e)[:400]}]
                print(f"  bench failed: {e!r}")
            all_bench[compile_mode] = rs
            # Re-build cleanly for next compile mode
            cfg, M1, M2, bjt = _build_models()

        # Find AMBITIOUS / PASS judgements at N=1M
        def _find(rs, N):
            for r in rs:
                if r.get("N") == N and r.get("ok"):
                    return r
            return None
        eager_1M = _find(all_bench.get("off", []), 1_000_000)
        comp_1M  = _find(all_bench.get("reduce-overhead", []), 1_000_000)
        gates = {
            "PASS_1M_under_60s": bool(eager_1M and eager_1M["wall_s"] < 60.0),
            "AMBITIOUS_1M_under_10s_compiled": bool(comp_1M and comp_1M["wall_s"] < 10.0),
            "eager_1M_wall_s": (eager_1M or {}).get("wall_s"),
            "compiled_1M_wall_s": (comp_1M or {}).get("wall_s"),
        }
        summary["benchmark"] = all_bench
        summary["gates"] = gates
        (out_dir / "benchmark.json").write_text(json.dumps(
            {"benchmark": all_bench, "gates": gates,
             "torch": torch.__version__,
             "gpu": summary.get("gpu", "cpu")}, indent=2))
        print("\n[gates]"); print(json.dumps(gates, indent=2))

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
