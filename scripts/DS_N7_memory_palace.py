"""DS-N7 — "Memory Palace" (Method of Loci) on NS-RAM cells using the
S2b surrogate framework (quadrilinear LUT, scripts/S2b_transient.py).

Design
------
Each NS-RAM 2T cell stores an analog body-charge level V_b in [0.30, 0.60].
We discretize that range into L levels (default L=10) to store one decimal
digit per cell (~log2(L) ≈ 3.32 bits/cell — about 3-4× a binary Flash bit).

  encode_value(value)  ->  Vb0 = 0.30 + (V_RANGE) * value / (L-1)
  read(Vb0_vec)        ->  apply a brief probe pulse (T_READ steps × dt_READ),
                            integrate the LUT-ODE in parallel over all cells,
                            decode V_b(t_probe) by nearest-neighbour to the
                            pre-calibrated codebook.

Three variants
  DS-N7a — SDM (Kanerva sparse distributed memory): each (K,V) pair is
           written to k addresses derived from a random projection of K,
           value is replicated; recall is majority vote over the k probes.
  DS-N7b — Sequence memory: an ordered list ⟨V_0,...,V_{T-1}⟩ keyed by K
           is written to cells hash(K,t). Recall traverses cells in order.
  DS-N7c — Pattern completion: present partial K_probe (random bit-erase),
           SDM addresses overlap with K's address set; majority vote
           recovers V.

The S2b LUT *physics* (body-charge fixed point, leakage) is the substrate;
the discrete codebook is calibrated by a one-shot forward sweep through
the LUT-ODE.

Gates
-----
INFRA      : N=1k,  P=100,   acc>90%
PASS       : N=10k, P=1000,  acc>80%
AMBITIOUS  : N=100k,P=10000, acc>80%  AND  wall<60 s.

CLI
---
    venv/bin/python scripts/DS_N7_memory_palace.py all
    venv/bin/python scripts/DS_N7_memory_palace.py infra
    venv/bin/python scripts/DS_N7_memory_palace.py pass
    venv/bin/python scripts/DS_N7_memory_palace.py ambitious
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from S2b_transient import IiiNetLUT  # noqa: E402

OUT = ROOT / "results" / "DS_N7_memory_palace"
OUT.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────
# Body-charge codebook calibration
# ─────────────────────────────────────────────────────────────────────────
V_LO = 0.30
V_HI = 0.60     # stay below LUT fixed-point ceiling (~0.65)
DEFAULT_L = 10  # decimal digit per cell (Kalle Anka = 5, etc.)

# Probe bias (cool — Vb relaxes toward fixed point but slowly enough that the
# trajectory at t=T_READ*dt_READ stays a monotone bijection of Vb0).
VG1_READ = 0.4
VG2_READ = 0.3
VD_READ  = 1.5
T_READ   = 5
DT_READ  = 1e-7   # 50 ns total probe
CB_READ  = 16e-15


def encode_value(value: np.ndarray, L: int = DEFAULT_L) -> np.ndarray:
    """Map integer value ∈ {0..L-1} → analog Vb0 ∈ [V_LO, V_HI]."""
    v = np.asarray(value, dtype=np.float64)
    return V_LO + (V_HI - V_LO) * v / (L - 1)


def parallel_read(lut: IiiNetLUT, Vb0: np.ndarray,
                  VG1: float = VG1_READ, VG2: float = VG2_READ,
                  Vd: float = VD_READ, T: int = T_READ,
                  dt_s: float = DT_READ, Cb_F: float = CB_READ) -> np.ndarray:
    """Vectorized N-cell explicit-Euler read.  Returns V_b(T*dt) per cell."""
    N = Vb0.size
    Vb = Vb0.astype(np.float64).copy()
    vg1 = np.full(N, VG1)
    vg2 = np.full(N, VG2)
    vd  = np.full(N, Vd)
    inv_Cb = 1.0 / Cb_F
    for _ in range(T):
        Inet = lut(vg1, vg2, vd, Vb)
        dVb  = (Inet * inv_Cb) * dt_s
        np.clip(dVb, -0.5, 0.5, out=dVb)
        Vb = np.clip(Vb + dVb, -0.5, 1.5)
    return Vb


def calibrate_codebook(lut: IiiNetLUT, L: int = DEFAULT_L) -> np.ndarray:
    """Return the L reference read-out values c_0..c_{L-1}."""
    levels = encode_value(np.arange(L), L=L)
    return parallel_read(lut, levels)


def decode_levels(codes_read: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Nearest-neighbour decode: read codes (N,) -> level index (N,)."""
    return np.argmin(np.abs(codes_read[:, None] - codebook[None, :]),
                     axis=1).astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────
# Hyperdimensional keys + address hashing
# ─────────────────────────────────────────────────────────────────────────
def random_keys(P: int, D: int = 64, seed: int = 0) -> np.ndarray:
    """Bipolar HD keys ∈ {-1,+1}^D."""
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=(P, D))


def key_addresses(K: np.ndarray, N_cells: int, k: int = 1,
                  proj_seed: int = 1234, H_bits: int = 32) -> np.ndarray:
    """Map keys (P,D) → addresses (P,k) ∈ [0,N_cells).

    Each of the k addresses comes from H_bits independent ±1 sign-projections of
    K (giving a H_bits-bit hash, ~2^H_bits range), then mod N_cells.  This
    avoids the very poor coverage you get when a single ±1 sketch can only
    take O(D) distinct values.
    """
    P, D = K.shape
    rng = np.random.default_rng(proj_seed)
    addrs = np.empty((P, k), dtype=np.int64)
    primes = (rng.integers(1, 2**62 - 1, size=k * H_bits, dtype=np.int64)
              .astype(np.uint64) | np.uint64(1))
    # H_bits projections, each contributes one hash bit (sign of <K,r>)
    Ki = K.astype(np.int32)
    for j in range(k):
        # build H_bits ±1 projections
        R = rng.choice(np.array([-1, 1], dtype=np.int8),
                       size=(D, H_bits))
        sketch = Ki @ R.astype(np.int32)          # (P, H_bits)  ∈ Z
        bits = (sketch > 0).astype(np.uint64)     # (P, H_bits)
        # Pack to a single H_bits-wide uint and mix with a per-family prime
        weights = (np.uint64(1) << np.arange(H_bits, dtype=np.uint64))
        h = (bits * weights[None, :]).sum(axis=1).astype(np.uint64)
        # second mix to break correlation with low-order bits
        prime = primes[j * H_bits] | np.uint64(1)
        h = (h ^ (h >> np.uint64(17))) * prime
        h = (h ^ (h >> np.uint64(23)))
        addrs[:, j] = (h % np.uint64(N_cells)).astype(np.int64)
    return addrs


# ─────────────────────────────────────────────────────────────────────────
# Palace memory operations
# ─────────────────────────────────────────────────────────────────────────
class MemoryPalace:
    """N analog cells, each holding one body-charge level."""

    def __init__(self, N_cells: int, lut: IiiNetLUT, L: int = DEFAULT_L,
                 k_sdm: int = 1, seed: int = 0):
        self.N = int(N_cells)
        self.L = int(L)
        self.k = int(k_sdm)
        self.lut = lut
        # All cells initialised to lowest level (V_LO).
        self.Vb = np.full(self.N, V_LO, dtype=np.float64)
        self.codebook = calibrate_codebook(lut, L=L)
        self.proj_seed = 1234 + seed

    # ── Write ──────────────────────────────────────────────────────────
    def encode_batch(self, K: np.ndarray, V: np.ndarray) -> dict:
        """Write batch of (K,V) pairs.  Overwrite-on-collision."""
        addrs = key_addresses(K, self.N, k=self.k, proj_seed=self.proj_seed)
        vb_targets = encode_value(V, L=self.L)                # (P,)
        # k copies per pair
        addrs_flat = addrs.reshape(-1)
        vb_flat    = np.repeat(vb_targets, self.k)
        # last-write-wins via index assignment
        self.Vb[addrs_flat] = vb_flat
        # Track unique cells touched for collision diagnostics
        n_unique = np.unique(addrs_flat).size
        return {"n_writes": int(addrs_flat.size),
                "n_unique_cells_touched": int(n_unique),
                "addrs": addrs}

    # ── Read ───────────────────────────────────────────────────────────
    def recall_batch(self, K: np.ndarray) -> np.ndarray:
        """Recall values for batch of keys, majority-voted across k addresses."""
        addrs = key_addresses(K, self.N, k=self.k, proj_seed=self.proj_seed)
        # Gather Vb0, read in one shot.
        Vb0 = self.Vb[addrs.reshape(-1)]
        codes = parallel_read(self.lut, Vb0)
        lvls = decode_levels(codes, self.codebook).reshape(addrs.shape)  # (P,k)
        # Majority vote across the k addresses.
        if self.k == 1:
            return lvls[:, 0]
        # mode per row
        out = np.empty(lvls.shape[0], dtype=np.int32)
        for i in range(lvls.shape[0]):
            vals, counts = np.unique(lvls[i], return_counts=True)
            out[i] = vals[np.argmax(counts)]
        return out


# ─────────────────────────────────────────────────────────────────────────
# Sequence memory (DS-N7b)
# ─────────────────────────────────────────────────────────────────────────
def encode_sequence(palace: MemoryPalace, K: np.ndarray, seq: np.ndarray,
                    chain_seed: int = 7) -> np.ndarray:
    """Encode ordered sequence ⟨v_0..v_{T-1}⟩ under one key K (D,)."""
    T = seq.size
    rng = np.random.default_rng(chain_seed)
    # Use H different hash families per time index — concatenate K with t-tag
    Ks = np.tile(K[None], (T, 1)).copy()
    tag = rng.choice(np.array([-1, 1], dtype=np.int8), size=(T, K.size))
    Ks = (Ks * tag).astype(np.int8)
    palace.encode_batch(Ks, seq)
    return Ks  # caller can reuse for recall


def recall_sequence(palace: MemoryPalace, Ks: np.ndarray) -> np.ndarray:
    return palace.recall_batch(Ks)


# ─────────────────────────────────────────────────────────────────────────
# Benchmark driver
# ─────────────────────────────────────────────────────────────────────────
def bench_one(N_cells: int, P_pairs: int, L: int, k_sdm: int,
              D_key: int = 64, seed: int = 0, lut: IiiNetLUT | None = None,
              ) -> dict:
    if lut is None:
        lut = IiiNetLUT()
    rng = np.random.default_rng(seed)
    K = random_keys(P_pairs, D=D_key, seed=seed)
    V = rng.integers(0, L, size=P_pairs).astype(np.int32)

    pal = MemoryPalace(N_cells=N_cells, lut=lut, L=L, k_sdm=k_sdm, seed=seed)
    t0 = time.time()
    enc = pal.encode_batch(K, V)
    t_enc = time.time() - t0

    t0 = time.time()
    V_rec = pal.recall_batch(K)
    t_rec = time.time() - t0

    acc = float(np.mean(V_rec == V))

    # Pattern-completion stress (DS-N7c): erase 25% of K bits (set to 0),
    # which preserves the sign of the random projection on average.
    K_partial = K.copy().astype(np.int8)
    n_erase = max(1, D_key // 4)
    for i in range(P_pairs):
        idx = rng.choice(D_key, size=n_erase, replace=False)
        K_partial[i, idx] = 0
    V_partial = pal.recall_batch(K_partial)
    acc_partial = float(np.mean(V_partial == V))

    return {
        "N_cells": N_cells,
        "P_pairs": P_pairs,
        "L_levels": L,
        "k_sdm": k_sdm,
        "D_key": D_key,
        "load_factor": P_pairs / N_cells,
        "n_writes": enc["n_writes"],
        "n_unique_cells_touched": enc["n_unique_cells_touched"],
        "collision_rate": 1.0 - enc["n_unique_cells_touched"]/enc["n_writes"],
        "t_encode_s": t_enc,
        "t_recall_s": t_rec,
        "t_total_s": t_enc + t_rec,
        "recall_acc": acc,
        "recall_acc_partialKey_25pct_flipped": acc_partial,
        "bits_per_cell_used": (P_pairs * np.log2(L)) /
                              max(enc["n_unique_cells_touched"], 1),
    }


def bench_sequence(N_cells: int, T_seq: int, L: int, lut: IiiNetLUT,
                    seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    K = rng.choice(np.array([-1, 1], dtype=np.int8), size=64)
    seq = rng.integers(0, L, size=T_seq).astype(np.int32)
    pal = MemoryPalace(N_cells=N_cells, lut=lut, L=L, k_sdm=1, seed=seed)
    t0 = time.time()
    Ks = encode_sequence(pal, K, seq)
    t_enc = time.time() - t0
    t0 = time.time()
    rec = recall_sequence(pal, Ks)
    t_rec = time.time() - t0
    acc = float(np.mean(rec == seq))
    return {
        "N_cells": N_cells, "T_seq": T_seq, "L": L,
        "t_encode_s": t_enc, "t_recall_s": t_rec,
        "recall_acc": acc,
    }


# ─────────────────────────────────────────────────────────────────────────
# Energy estimate (rough, based on cell read pulse)
# ─────────────────────────────────────────────────────────────────────────
def energy_per_read_J(N_cells_active: int,
                      Vd: float = VD_READ, Idiag_A: float = 50e-9,
                      T_s: float = T_READ * DT_READ) -> float:
    """Static current ~50 nA per cell × Vd × pulse time × N."""
    return N_cells_active * Vd * Idiag_A * T_s


# ─────────────────────────────────────────────────────────────────────────
# Main suite
# ─────────────────────────────────────────────────────────────────────────
def run_gate(gate: str, lut: IiiNetLUT) -> dict:
    if gate == "infra":
        cfg = dict(N_cells=1_000,   P_pairs=100,   L=10, k_sdm=1)
        thr = 0.90
    elif gate == "pass":
        cfg = dict(N_cells=10_000,  P_pairs=1_000, L=10, k_sdm=1)
        thr = 0.80
    elif gate == "ambitious":
        cfg = dict(N_cells=100_000, P_pairs=10_000, L=10, k_sdm=1)
        thr = 0.80
    else:
        raise ValueError(gate)
    res = bench_one(lut=lut, **cfg)
    res["gate"] = gate
    res["threshold"] = thr
    res["PASS"] = bool(res["recall_acc"] >= thr and (
        gate != "ambitious" or res["t_total_s"] < 60.0))
    return res


def capacity_sweep(lut: IiiNetLUT, N_cells: int, seed: int = 0,
                   L: int = DEFAULT_L) -> list[dict]:
    """Sweep load factor: P/N ∈ {0.05, 0.1, 0.25, 0.5, 1.0, 2.0}."""
    out = []
    for lf in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
        P = max(10, int(lf * N_cells))
        # both k=1 (plain) and k=4 (SDM redundancy)
        for k in (1, 4):
            r = bench_one(N_cells=N_cells, P_pairs=P, L=L, k_sdm=k, lut=lut,
                          seed=seed)
            r["load_factor_nominal"] = lf
            out.append(r)
            print(f"   N={N_cells:>6d} P={P:>6d} k={k} acc={r['recall_acc']:.3f} "
                  f"acc_partial={r['recall_acc_partialKey_25pct_flipped']:.3f} "
                  f"t={r['t_total_s']*1000:.1f}ms")
    return out


def plot_curves(sweep_by_N: dict, path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  matplotlib unavailable ({e}); skipping plot")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for N, sweeps in sweep_by_N.items():
        for k in (1, 4):
            rows = [r for r in sweeps if r["k_sdm"] == k]
            lf  = [r["load_factor"]  for r in rows]
            acc = [r["recall_acc"]   for r in rows]
            accp= [r["recall_acc_partialKey_25pct_flipped"] for r in rows]
            axes[0].plot(lf, acc,  marker="o", label=f"N={N} k={k}")
            axes[1].plot(lf, accp, marker="s", label=f"N={N} k={k}")
    for ax, title in zip(axes,
                          ("Recall accuracy vs load factor (P/N)",
                           "Pattern-completion (25% key bits flipped)")):
        ax.set_xscale("log")
        ax.set_xlabel("load factor  P / N_cells")
        ax.set_ylabel("accuracy")
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(0.8, ls=":", color="grey", lw=0.8)
        ax.axhline(0.9, ls=":", color="grey", lw=0.8)
        ax.set_title(title)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["all", "infra", "pass", "ambitious",
                                      "sweep", "sequence"],
                     nargs="?", default="all")
    args = ap.parse_args()

    print("[DS-N7] loading LUT ...", flush=True)
    lut = IiiNetLUT()

    summary = {"host": __import__("os").uname().nodename,
                "gates": {}, "sweeps": {}, "sequence": None}

    # Calibration sanity check
    cb = calibrate_codebook(lut)
    print(f"  codebook (L={DEFAULT_L}): {[round(c,4) for c in cb]}")

    if args.cmd in ("all", "infra"):
        print("\n[DS-N7] INFRA gate (N=1k, P=100) ...")
        r = run_gate("infra", lut)
        summary["gates"]["infra"] = r
        print(f"  acc={r['recall_acc']:.3f} t={r['t_total_s']*1000:.1f}ms "
              f"PASS={r['PASS']}")

    if args.cmd in ("all", "pass"):
        print("\n[DS-N7] PASS gate (N=10k, P=1k) ...")
        r = run_gate("pass", lut)
        summary["gates"]["pass"] = r
        print(f"  acc={r['recall_acc']:.3f} t={r['t_total_s']*1000:.1f}ms "
              f"PASS={r['PASS']}")

    if args.cmd in ("all", "ambitious"):
        print("\n[DS-N7] AMBITIOUS gate (N=100k, P=10k) ...")
        r = run_gate("ambitious", lut)
        summary["gates"]["ambitious"] = r
        print(f"  acc={r['recall_acc']:.3f} t={r['t_total_s']*1000:.1f}ms "
              f"PASS={r['PASS']}")

    if args.cmd in ("all", "sweep"):
        print("\n[DS-N7] capacity sweep (k=1, k=4 SDM) ...")
        sweeps = {}
        for N in (1_000, 10_000, 100_000):
            print(f" N_cells = {N}")
            sweeps[N] = capacity_sweep(lut, N_cells=N)
        summary["sweeps"] = {str(k): v for k, v in sweeps.items()}
        plot_curves(sweeps, OUT / "recall_curves.png")

    if args.cmd in ("all", "sequence"):
        print("\n[DS-N7] sequence memory ...")
        seq_results = []
        for N in (10_000, 100_000):
            for T_seq in (100, 1000):
                r = bench_sequence(N, T_seq, L=10, lut=lut)
                print(f"  N={N} T_seq={T_seq:>4d} acc={r['recall_acc']:.3f} "
                      f"t={(r['t_encode_s']+r['t_recall_s'])*1000:.1f}ms")
                seq_results.append(r)
        summary["sequence"] = seq_results

    # Energy estimate
    summary["energy"] = {
        "energy_per_read_100k_J": energy_per_read_J(100_000),
        "energy_per_read_1k_J":   energy_per_read_J(1_000),
        "notes": "Static-current pulse model: I_diag ≈ 50 nA · V_read · T_read",
    }

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2,
                                                   default=lambda o: float(o)
                                                   if isinstance(o, np.floating)
                                                   else (int(o) if isinstance(o, np.integer)
                                                         else str(o))))
    print(f"\n[DS-N7] -> {OUT/'summary.json'}")


if __name__ == "__main__":
    main()
