"""Step 2: GPU batched forward throughput on GB10.

Measures cells/sec for fp32 and fp64 at batch sizes 1k, 10k, 100k using
the existing forward_2t_gpu_batched (non-differentiable Newton-only path,
adequate for raw throughput).

Output: results/GPU_MAX_A_zgx/batch_throughput.json
"""
from __future__ import annotations
import json, os, time
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import build_nsram_stack

from nsram.bsim4_port.forward_2t_batched_gpu import forward_2t_gpu_batched


OUT = Path(os.environ.get("GPU_MAX_A_OUT",
                          str(Path(__file__).resolve().parents[2] /
                              "results/GPU_MAX_A_zgx")))
OUT.mkdir(parents=True, exist_ok=True)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    dev = torch.device("cuda")
    print(f"[throughput] device={torch.cuda.get_device_name(0)}, "
          f"torch={torch.__version__}")

    cfg, M1, M2, bjt = build_nsram_stack()

    T = 4  # Vd points per cell (was 8 — drop for runtime)
    Vd_seq_base = torch.linspace(0.8, 1.8, T)
    rng = torch.Generator(device="cpu").manual_seed(7)

    results = []
    for dtype in (torch.float32, torch.float64):
        for N in (1024, 10_000, 50_000):
            VG1 = (0.30 + 0.35*torch.rand(N, generator=rng)).to(dev, dtype=dtype)
            VG2 = (0.00 + 0.50*torch.rand(N, generator=rng)).to(dev, dtype=dtype)
            Vd_seq = Vd_seq_base.to(dev, dtype=dtype)
            # warm-up + sync
            try:
                out = forward_2t_gpu_batched(
                    cfg, M1, M2, bjt, Vd_seq, VG1, VG2,
                    max_iters=12, tol=1e-8, damping=1.0,
                    dtype=dtype, device=dev, compile_mode="off",
                    early_stop=False, verbose=False)
                torch.cuda.synchronize()
            except Exception as e:
                print(f"  N={N} dtype={dtype} warmup FAIL: {e!r}")
                results.append({"N": N, "dtype": str(dtype), "error": repr(e)})
                continue

            t0 = time.time()
            n_repeats = 3 if N <= 10_000 else 1
            for _ in range(n_repeats):
                out = forward_2t_gpu_batched(
                    cfg, M1, M2, bjt, Vd_seq, VG1, VG2,
                    max_iters=12, tol=1e-8, damping=1.0,
                    dtype=dtype, device=dev, compile_mode="off",
                    early_stop=False, verbose=False)
            torch.cuda.synchronize()
            wall = (time.time() - t0) / n_repeats
            conv = float(out["converged"].float().mean().item())
            cells_per_s = N / wall
            pt_per_s    = (N * T) / wall
            rec = {
                "N": N, "T": T, "dtype": str(dtype).replace("torch.", ""),
                "wall_s": wall, "conv_rate": conv,
                "cells_per_s": cells_per_s, "point_evals_per_s": pt_per_s,
            }
            print(f"  N={N:>7d} dtype={rec['dtype']:>7s}  "
                  f"wall={wall:.3f}s  conv={conv*100:.1f}%  "
                  f"cells/s={cells_per_s:.0f}  points/s={pt_per_s:.0f}")
            results.append(rec)
            del VG1, VG2, out; torch.cuda.empty_cache()

    summary = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "results": results,
        "peak_cells_per_s": max(
            (r["cells_per_s"] for r in results if "cells_per_s" in r),
            default=None),
    }
    out_path = OUT / "batch_throughput.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[throughput] wrote {out_path}")


if __name__ == "__main__":
    main()
