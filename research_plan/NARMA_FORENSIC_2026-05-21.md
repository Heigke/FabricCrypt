# NARMA-10 forensic — 0.612 (brief) vs 0.846 (J2 actual)

Date: 2026-05-21. Author: agent. Status: diagnosis only, no new sweeps launched.

## The gap

- **Brief claim** (`nsram/main-4.tex` L106-108, L279-281; `nsram/onepager.tex` L82-84):
  NS-RAM NRMSE = **0.612 ± 0.030**, N=200, 30 seeds, ridge readout.
- **J2 v5.5 actual** (`results/NARMA10_v55_scaling_2026-05-21/results_final.json` on zgx, 30 seeds × 4 N × 2 κ):
  | N | κ=0.000 mean | κ=0.003 mean | best (min over seeds) |
  |---|---|---|---|
  | 200 | 1.0935 | 0.9737 | **0.8461** (N=200, κ=0.003, seed=2) |
  | 1000 | 1.0981 | 1.1111 | 0.9618 |
  | 4000 | 1.1006 | 1.4016 | 1.1327 |
  | 16000 | 1.1024 | 1.8031 | 1.4317 |

Grand-min NRMSE = 0.846, ≈ 38 % worse than brief headline. NRMSE *increases* with N — the opposite of the brief's scaling story.

## Hypothesis check

(i) **κ grid too coarse — REJECTED as primary**. Brief mentions `κ=0.30` (`main-4.tex` L323) only for the Mackey-Glass figure, not for NARMA-10. J2 used `κ ∈ {0, 0.003}`. Even so, κ=0.003 already gives the best score in the J2 grid; pushing κ higher in the same harness will most likely *worsen* (W_t @ feat_prev grows ∝ √N at large N, hence the explosive N=16k×κ=0.003 column at 1.80).

(ii) **Brief number was the tanh-ESN control, not NS-RAM — PARTIALLY**. The brief itself states ESN = 0.563 ± 0.038 vs NS-RAM = 0.612 ± 0.030 (`main-4.tex` L518-520). Both numbers lack on-disk receipts: `results/DS_N10_reservoir/narma10_MC.json` (the only NARMA file in repo) shows NS-RAM **1.713 ± 0.214** at N=10000, 5 seeds — the brief's 0.612 was produced elsewhere and not committed.

(iii) **v5.5 surrogate tail corrupts at large N — LIKELY contributor**. The p95 log-Id tail of the v5.5 surrogate (`results/SURROGATE_v55_zgx/surrogate_v55.npz`) is ~5 dec; the J2 reservoir uses `recur = W_t @ feat_prev` *unbounded* and only clamps VG2 to [-0.2, 1.0] (`J2_narma_v55_scaling.py` L147). At N≥4000 the operator-norm proxy `√N · ρ` no longer matches the true spectral radius, so κ=0.003 effectively becomes super-critical → 1.40, 1.80.

(iv) **Leaky integration / ridge — UNVERIFIED**. J2 uses ridge=1e-3 (L159), no leak. The brief never publishes leak or ridge. This is the most plausible *closable* gap if 0.612 ever came from an NS-RAM run at all.

## Diagnosis

The brief's 0.612 ± 0.030 has **no in-repo receipt**. The closest committed NS-RAM NARMA-10 result on disk is 1.71 (DS_N10, N=10000). The J2 v5.5 sweep is internally consistent and shows NS-RAM is well above ESN territory across N∈{200…16000}. Most likely the brief number is either (a) an uncommitted z96-family run with different surrogate + leaky readout that was never reproduced, or (b) the ESN control mislabelled as NS-RAM in an earlier draft. Recommend: pull the brief claim down to ≤ 0.85 (the J2 measured best) until an authoritative receipt is produced, *before* spending more GPU on a sweep that may be chasing a phantom optimum.

## Next step (NOT executed)

Re-run J2 at N=200 with κ ∈ {0.01, 0.03, 0.1, 0.3}, leak ∈ {0, 0.3, 0.7}, ridge ∈ {1e-4, 1e-3, 1e-2}. Single grid, 30 seeds. ≤ 1 GPU·h on zgx.
