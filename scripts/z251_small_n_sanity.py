"""z251 — small-N sanity check (O40 grok follow-up).

Tests if NS-RAM has a niche advantage at very small N where the
matrix didn't reach (smallest in matrix was N=100). 5 seeds per cell,
N ∈ {30, 50, 100}, NARMA-10.

PRE-REGISTERED gate per N: NS-RAM CI95 upper < ESN CI95 lower.
If any N passes, NS-RAM has a small-network niche worth noting in
follow-up to Mario. Otherwise, pattern of ESN dominance extends to
small N too.

Non-blocking optional follow-up; informational only.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/z251_small_n_sanity"; OUT.mkdir(parents=True, exist_ok=True)
SURR_PATH = ROOT / "results/z220_4d_dense/surrogate_4d_dense.npz"
sys.path.insert(0, str(ROOT / "scripts"))
from z249_nsram_vs_esn_scaling import gen_narma10, run_nsram, run_esn, boot_ci
from scripts.nsram_surrogate_4d import NSRAMSurrogate4D


def main():
    print(f"=== z251 small-N sanity (N ∈ {{30, 50, 100}}) ===", flush=True)
    surr = NSRAMSurrogate4D(SURR_PATH)
    # n_block must divide N for block-diag W; pick n_block <= N
    Ns = [30, 50, 100]
    seeds = [0, 1, 2, 3, 4]
    T = 1500
    results = {}
    for N in Ns:
        nsr = []; esn = []
        # For N<50, override n_block in run_nsram by monkey-patching arg
        # We call with N=N; the function uses min(50, N)
        for s in seeds:
            u, y = gen_narma10(T, s)
            t0 = time.time()
            r_n = run_nsram(surr, u, y, N=N, seed=s)
            r_e = run_esn(u, y, N=N, seed=s)
            wall = time.time() - t0
            nsr.append(r_n); esn.append(r_e)
            print(f"  N={N} s{s}  NS-RAM={r_n:.4f}  ESN={r_e:.4f}  Δ={r_n-r_e:+.4f}  wall={wall:.1f}s",
                  flush=True)
        nsr = np.array(nsr); esn = np.array(esn)
        nsr_ci = boot_ci(nsr); esn_ci = boot_ci(esn)
        nsram_wins = bool(nsr_ci[1] < esn_ci[0])
        esn_wins = bool(esn_ci[1] < nsr_ci[0])
        results[str(N)] = {
            "nsram_mean": float(nsr.mean()), "nsram_ci": list(nsr_ci),
            "esn_mean": float(esn.mean()), "esn_ci": list(esn_ci),
            "nsram_strictly_wins": nsram_wins,
            "esn_strictly_wins": esn_wins,
        }
        print(f"  N={N}: NS-RAM {nsr.mean():.4f} CI {nsr_ci}  "
              f"vs ESN {esn.mean():.4f} CI {esn_ci}  NS-RAM:{nsram_wins} ESN:{esn_wins}",
              flush=True)
    n_nsram = sum(1 for r in results.values() if r["nsram_strictly_wins"])
    summary = {
        "Ns": Ns, "n_seeds": len(seeds), "results": results,
        "n_nsram_strict_wins": n_nsram,
        "small_n_niche_found": n_nsram >= 1,
        "interpretation": (
            f"NICHE FOUND — NS-RAM strict win at {n_nsram} of {len(Ns)} small-N values. "
            f"Worth a one-line follow-up note to Mario."
            if n_nsram >= 1 else
            f"No small-N niche; ESN dominance extends to N=30. Matrix pattern complete."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{'✅ NICHE' if summary['small_n_niche_found'] else '❌ NO NICHE'}",
          flush=True)
    print(summary["interpretation"], flush=True)


if __name__ == "__main__":
    main()
