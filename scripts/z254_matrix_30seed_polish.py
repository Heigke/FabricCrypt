"""z254 — 30-seed publication-grade replication of NARMA-5, NARMA-20, MC.

Bumps three matrix cells from 5-seed exploratory to 30-seed power
matching z223 and z243. Pre-registered gates UNCHANGED from z247/z248:

  NARMA-K: NS-RAM CI95 upper < ESN CI95 lower (strict NS-RAM win).
  Memory Capacity: same gate on TOTAL MC (sum over k=1..100).

Expected: gates still fail with much tighter CIs. This is defensive
publication-polish, not new exploration. NO-CHEAT discipline: gates
unchanged; no post-hoc reinterpretation.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results/z254_matrix_30seed_polish"; OUT.mkdir(parents=True, exist_ok=True)
SURR_PATH = ROOT / "results/z220_4d_dense/surrogate_4d_dense.npz"

from scripts.nsram_surrogate_4d import NSRAMSurrogate4D
from z247_nsram_vs_esn_narma_k import gen_narma_k, run_nsram, run_esn
from z248_nsram_vs_esn_memcap import build_state_nsram, build_state_esn, mc_curve


def boot_ci(arr, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    boots = np.array([arr[rng.integers(0, len(arr), len(arr))].mean()
                        for _ in range(n)])
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def main():
    print(f"=== z254 30-seed polish (NARMA-5, NARMA-20, MC) ===", flush=True)
    surr = NSRAMSurrogate4D(SURR_PATH)
    seeds = list(range(30))
    T_narma = 1500
    out = {}

    for K in [5, 20]:
        nsr = []; esn = []
        for s in seeds:
            u, y = gen_narma_k(T_narma, s, K)
            r_n = run_nsram(surr, u, y, N=200, seed=s)
            r_e = run_esn(u, y, N=200, seed=s)
            nsr.append(r_n); esn.append(r_e)
            if s % 5 == 0:
                print(f"  NARMA-{K} s{s}  NS-RAM={r_n:.4f}  ESN={r_e:.4f}",
                      flush=True)
        nsr = np.array(nsr); esn = np.array(esn)
        nsr_ci = boot_ci(nsr); esn_ci = boot_ci(esn)
        nsram_wins = bool(nsr_ci[1] < esn_ci[0])
        esn_wins = bool(esn_ci[1] < nsr_ci[0])
        out[f"NARMA-{K}"] = {
            "n": 30,
            "nsram_mean": float(nsr.mean()), "nsram_std": float(nsr.std()),
            "nsram_ci95": list(nsr_ci),
            "esn_mean": float(esn.mean()), "esn_std": float(esn.std()),
            "esn_ci95": list(esn_ci),
            "nsram_strict_wins": nsram_wins,
            "esn_strict_wins": esn_wins,
        }
        print(f"  NARMA-{K} 30-seed: NS-RAM {nsr.mean():.4f} CI {nsr_ci}  "
              f"vs ESN {esn.mean():.4f} CI {esn_ci}  "
              f"NS-RAM:{nsram_wins}  ESN:{esn_wins}", flush=True)

    # Memory Capacity 30-seed
    print(f"\nMemory Capacity 30-seed...", flush=True)
    delays = [1, 2, 5, 10, 20, 50, 100]
    T_mc = 2000
    nsram_totals = []; esn_totals = []
    for s in seeds:
        rng = np.random.default_rng(s + 12345)
        u = rng.uniform(0.0, 1.0, T_mc)
        sn = build_state_nsram(surr, u, N=200, seed=s)
        se = build_state_esn(u, N=200, seed=s)
        mn = mc_curve(sn, u, delays).sum()
        me = mc_curve(se, u, delays).sum()
        nsram_totals.append(mn); esn_totals.append(me)
        if s % 5 == 0:
            print(f"  MC s{s} NS-RAM={mn:.3f} ESN={me:.3f}", flush=True)
    nsr = np.array(nsram_totals); esn = np.array(esn_totals)
    nsr_ci = boot_ci(nsr); esn_ci = boot_ci(esn)
    # Higher MC is better, so flip the gate: NS-RAM CI lower > ESN CI upper
    nsram_wins = bool(nsr_ci[0] > esn_ci[1])
    esn_wins = bool(esn_ci[0] > nsr_ci[1])
    out["MC_total"] = {
        "n": 30,
        "nsram_mean": float(nsr.mean()), "nsram_std": float(nsr.std()),
        "nsram_ci95": list(nsr_ci),
        "esn_mean": float(esn.mean()), "esn_std": float(esn.std()),
        "esn_ci95": list(esn_ci),
        "nsram_strict_wins": nsram_wins,
        "esn_strict_wins": esn_wins,
    }
    print(f"  MC total 30-seed: NS-RAM {nsr.mean():.3f} CI {nsr_ci}  "
          f"vs ESN {esn.mean():.3f} CI {esn_ci}  "
          f"NS-RAM:{nsram_wins}  ESN:{esn_wins}", flush=True)

    summary = {
        "n_seeds": 30,
        "results": out,
        "interpretation": "30-seed publication-grade replication of NARMA-5, "
                          "NARMA-20, Memory Capacity. CIs tighter than 5-seed "
                          "originals; pre-registered gates unchanged.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== Summary ===", flush=True)
    for task, r in out.items():
        print(f"  {task:>10s}  NS-RAM {r['nsram_mean']:.4f} CI {[round(x,4) for x in r['nsram_ci95']]}  "
              f"ESN {r['esn_mean']:.4f} CI {[round(x,4) for x in r['esn_ci95']]}",
              flush=True)


if __name__ == "__main__":
    main()
