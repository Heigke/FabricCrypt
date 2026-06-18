"""z225 — GPU scale proof-of-concept: large-N reservoir on gfx1151.

Demonstrates that the reservoir + 4D body-state surrogate scales to
N≥10k cells on AMD ROCm GPU. Two topologies shown:
  1. Block-diagonal: K independent blocks of size n (Kn=N) — embarrassingly parallel
  2. Sparse random: 0.5% density, used as comparison

Reports per-step wall time and APU temp at each N.

PER USER REQUEST 2026-05-08: scale toward millions of cells. This
PoC characterizes the compute envelope.
"""
from __future__ import annotations
import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import sys, json, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/z225_gpu_scale"; OUT.mkdir(parents=True, exist_ok=True)


def get_apu():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return -1.0


def make_block_diag_W(N, block_size, density=0.1, dtype=torch.float32, device="cpu"):
    """K block-diagonal W with K=N//block_size blocks. Each block density-sparse."""
    K = N // block_size
    if N != K * block_size:
        raise ValueError(f"N={N} not divisible by block_size={block_size}")
    blocks = []
    rng = np.random.default_rng(0)
    for k in range(K):
        m = (rng.random((block_size, block_size)) < density).astype(np.float32)
        w = m * rng.normal(0, 1, (block_size, block_size)).astype(np.float32)
        np.fill_diagonal(w, 0)
        # Spectral normalize block to ~0.9
        eig = float(np.abs(np.linalg.eigvals(w)).max())
        if eig > 1e-9:
            w *= 0.9 / eig
        blocks.append(w)
    # Build sparse block-diagonal as torch sparse_csr
    indptr = [0]
    indices = []
    values = []
    for k, b in enumerate(blocks):
        for i in range(block_size):
            row_offset = k * block_size
            cols = np.where(b[i] != 0)[0] + row_offset
            indices.extend(cols.tolist())
            values.extend(b[i, b[i] != 0].tolist())
            indptr.append(len(indices))
    indptr = torch.tensor(indptr, dtype=torch.int64, device=device)
    indices = torch.tensor(indices, dtype=torch.int64, device=device)
    values = torch.tensor(values, dtype=dtype, device=device)
    W = torch.sparse_csr_tensor(indptr, indices, values, size=(N, N), device=device)
    return W


def make_sparse_random_W(N, density=0.005, dtype=torch.float32, device="cpu"):
    """Sparse random W with given density. Spectral-normalized approximately."""
    rng = np.random.default_rng(1)
    nnz_per_row = max(1, int(density * N))
    cols = []
    indptr = [0]
    values = []
    for i in range(N):
        idx = rng.choice(N, size=nnz_per_row, replace=False)
        idx = idx[idx != i]   # no self-loop
        v = rng.normal(0, 1.0/np.sqrt(nnz_per_row), len(idx)).astype(np.float32)
        cols.extend(idx.tolist())
        values.extend(v.tolist())
        indptr.append(len(cols))
    indptr = torch.tensor(indptr, dtype=torch.int64, device=device)
    cols = torch.tensor(cols, dtype=torch.int64, device=device)
    values = torch.tensor(values, dtype=dtype, device=device)
    W = torch.sparse_csr_tensor(indptr, cols, values, size=(N, N), device=device)
    return W


def time_reservoir_step(N, T, topo_kind, device, n_warmup=5, n_time=20):
    """Measure per-step wall time for reservoir update."""
    if topo_kind == "block":
        W = make_block_diag_W(N, block_size=min(1000, N), device=device)
    else:
        W = make_sparse_random_W(N, density=0.005, device=device)
    feat = torch.zeros(N, dtype=torch.float32, device=device)
    inp = torch.randn(N, dtype=torch.float32, device=device)
    # Warmup
    for _ in range(n_warmup):
        feat = 0.7 * feat + 0.3 * (W @ feat + inp)
        if device == "cuda":
            torch.cuda.synchronize()
    # Time
    t0 = time.time()
    for _ in range(n_time):
        feat = 0.7 * feat + 0.3 * (W @ feat + inp)
        if device == "cuda":
            torch.cuda.synchronize()
    elapsed = (time.time() - t0) / n_time
    return elapsed


def main():
    print(f"=== GPU scale PoC ===")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"\n{'N':>10}  {'topo':>10}  {'device':>6}  {'step (ms)':>11}  {'inf 60T (s)':>13}  {'APU °C':>7}")

    out = []
    sizes = [(1_000, "block"), (10_000, "block"), (100_000, "block"),
              (1_000, "sparse"), (10_000, "sparse"), (100_000, "sparse")]
    # Skip 1M for first pass (memory check first)
    for N, topo in sizes:
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            try:
                if get_apu() > 75:
                    print(f"  APU {get_apu()}°C >75 — skipping rest")
                    break
                step_s = time_reservoir_step(N, 60, topo, device)
                inf_s = step_s * 60
                apu = get_apu()
                print(f"  {N:>9}  {topo:>10}  {device:>6}  "
                      f"{step_s*1000:>9.2f}  {inf_s:>11.2f}  {apu:>7.1f}")
                out.append({"N": N, "topo": topo, "device": device,
                              "step_ms": step_s*1000, "inf_60_s": inf_s, "apu": apu})
            except Exception as e:
                print(f"  {N:>9}  {topo:>10}  {device:>6}  FAILED: {e}")

    # Try 1M only on GPU with block (best path)
    print(f"\n=== Stretch test: N=1M block-diag, GPU only ===")
    if torch.cuda.is_available() and get_apu() < 60:
        try:
            step_s = time_reservoir_step(1_000_000, 60, "block", "cuda")
            print(f"  1M block GPU: {step_s*1000:.1f} ms/step, {step_s*60:.1f} s for 60-step inference")
            out.append({"N": 1_000_000, "topo": "block", "device": "cuda",
                          "step_ms": step_s*1000, "inf_60_s": step_s*60, "apu": get_apu()})
        except Exception as e:
            print(f"  N=1M failed: {e}")
    else:
        print(f"  skipped (APU={get_apu()}°C, GPU={torch.cuda.is_available()})")

    (OUT / "summary.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
