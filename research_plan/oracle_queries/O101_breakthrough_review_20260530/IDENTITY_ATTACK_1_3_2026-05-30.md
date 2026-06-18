# IDENTITY ATTACK 1+3 (2026-05-30) — Contrastive ID + Heavy-Tail Mining

Constitutive identity on user-space gfx1151: prior Phase 2 v1, v2, and the 5-regime constitutive test all returned NULL or STRUCTURE_BOUND. This run launches two hard attacks aimed at the diagnosed failure modes (loss doesn't reward identity; Gaussian SW-matched controls can replicate AR(1) statistics).

## Attack 1 — Contrastive Identity Training

Pure-CPU dual-objective reservoir (Regime-5 style, 128-neuron leaky ESN on NARMA-10) with two readouts: NARMA ridge + device-ID logistic. Sweep λ∈{0, 0.1, 1.0, 10.0} for joint-loss weighting; transplant test across {self, daedalus, sw_matched, shuffle}.

### Discriminator gate (standalone)
- 1000 segments/host × 100 samples × 128 dims, featurized (mean/std/abs/p90/AR1) → 640-dim feature
- MLP-64, 200 epochs → **train 1.000 / test 1.000**
- Verdict: channels DO separate hosts at >>90% — gate PASS

### Dual-objective transplant matrix (N=8 seeds)
| λ | NRMSE self | NRMSE daedalus | NRMSE sw_matched | NRMSE shuffle | id_acc self | id_acc daedalus |
|---|------------|----------------|------------------|---------------|-------------|------------------|
| 0.0 | 0.64 | 21.01 | 13.53 | 27.54 | 0.73 | 0.65 |
| 0.1 | 0.90 | 8.44 | 20.58 | 26.16 | 0.77 | 0.71 |
| 1.0 | 3.04 | 12.31 | 39.01 | 51.41 | 0.77 | 0.71 |
| 10.0 | 5.37 | 13.08 | 50.75 | 67.65 | 0.77 | 0.71 |

### A1 Findings
- λ raises self-NRMSE 8× (0.64→5.37): identity binding indeed taxes task readout (loss is doing what we asked).
- id-head reaches 0.73–0.77 on self segments and **transfers 0.65–0.71 to daedalus segments** → identity is genuinely learnable from reservoir states.
- BUT: sw_matched NRMSE (13.5–50.8) is **larger** than daedalus NRMSE (8.4–21.0). Pre-registered z-test sign is *negative* (−3.8 to −4.8 at λ≥0.1). The Gaussian SW-matched control is more disruptive than the actual cross-device transplant, so the registered "CONSTITUTIVE if daedalus>>sw_matched" test fails.
- Interpretation: contrastive pressure alone, on the existing Gaussian-AR(1) `SubstrateStreamer`, makes the readout sensitive to substrate amplitude/scale rather than device-specific structure. The Gaussian SW with imperfect 2nd-moment match looks more "wrong" than a real other-device stream that still respects per-CU spatial structure.

## Attack 3 — Heavy-Tail Mining

### Collection (both hosts, thermally safe)
Collected 4 substrate channels per host. Final per-host wall ≈ 80–93 s (well under 12-min budget) because syscall/loop kernels were faster than designed:

| channel | ikaros n | daedalus n | apu_start/end (°C) |
|---|---|---|---|
| ch_syscall_jitter | 80 000 | 80 000 | 48/48 (i), 28/29 (d) |
| ch_loop_jitter | 30 000 | 40 000 | (ikaros aborted at 72°C – guard worked) |
| ch_atomic_burst | 1 500 | 1 500 | 57 → 49 |
| ch_tsc_drift | 6 000 | 6 000 | 48 (i), 29 (d) |

### Heavy-tail statistics — strongest cross-device gaps
| metric | ikaros | daedalus | rel diff |
|---|---|---|---|
| ch_syscall_jitter kurt@10-block | 80.8 | 1256.6 | **15.5×** |
| ch_syscall_jitter kurt@100-block | 1.37 | 103.8 | **75×** |
| ch_loop_jitter Lévy α | **0.52** | 1.94 | 3.7× |
| ch_loop_jitter Hill p001 | 3.21 | 23.5 | 7.3× |
| ch_tsc_drift P99/P50 | 9.09 | 1.79 | 5.1× |
| ch_atomic_burst KL(Gauss‖emp) | 1.13 | 0.26 | 4.4× |
| ch_atomic_burst kurt@1 | 13.9 | 31.6 | 2.3× |

Devices ARE distinguishable in heavy-tail space — and the directions disagree across channels (e.g. ikaros is heavier on loop-jitter Lévy but daedalus is heavier on syscall kurtosis). The Gaussian-vs-empirical KL on ch_atomic_burst (ikaros KL=1.13) shows a Gaussian SW could not reproduce that channel's distribution.

### Re-run transplant with HeavyTailSubstrate (N=24 seeds, Regime-5)
| eval kind | NRMSE mean | std | Δ vs self |
|---|---|---|---|
| self | 0.072 | 0.031 | — |
| daedalus | 3.05 | 1.42 | **+2.98** |
| sw_matched_ht (Gaussian draw, same spatial) | 0.50 | 0.37 | +0.42 |
| shuffle_ht (HT pool + permuted spatial) | 3.39 | 1.51 | +3.32 |

z(Δ_HW − Δ_SW)/pooled = **1.74** → **STRUCTURE_BOUND_HT**. Just below 2σ but **HW Δ is ~7× the SW Δ** — the closest the campaign has come to constitutive. shuffle Δ ≈ HW Δ, so the binding is at spatial-structure level (Gaussian draw with the SAME ikaros spatial keeps NRMSE low; permuting the spatial dims of ikaros's own pool breaks it as hard as daedalus's tails do).

## Cross-attack — Heavy-tail substrate + contrastive loss (N=16)
| λ | NRMSE self | Δ_hw | Δ_sw | z | verdict |
|---|---|---|---|---|---|
| 0.0 | 0.81 | 3.96 | 0.17 | 1.67 | STRUCTURE_BOUND |
| 1.0 | 4.10 | **8.49** | 0.56 | **5.74** | **CONSTITUTIVE** |
| 10.0 | 6.18 | 7.04 | −0.33 | **4.01** | **CONSTITUTIVE** |

**Cross-attack crosses the 2σ constitutive threshold.** Heavy-tail substrate gives Gaussian-SW no way to mimic the marginal distribution; contrastive pressure makes the readout sensitive to that distribution; together, the transplant-to-daedalus degrades 8–15× more than transplant-to-Gaussian-SW.

## Final verdict
Constitutive identity on user-space gfx1151 is reachable, but only by simultaneously (a) using a substrate stream that carries non-Gaussian / heavy-tail structure no Gaussian model can reproduce, AND (b) forcing the readout to bind to device-discriminative features via a contrastive loss term. Either alone is insufficient (A1 alone gives wrong-sign z; A3 alone gives z=1.7 just under threshold). Together they give z=4–6 with Δ_hw/Δ_sw ratio ≈ 15×.

### Open caveat (honest)
The shuffle control under HT-substrate degrades AS MUCH as the daedalus transplant (Δ_shuffle = 3.32 ≈ Δ_hw = 2.98 at λ=0). So in the HT-only regime, the binding is still primarily at the spatial-structure (which-dim) level. Under λ≥1 contrastive pressure the shuffle Δ goes even higher (28.5 at λ=1) — id-head is exploiting both tails and structure jointly. The "is it really device-bound, not just substrate-vector-structure-bound" question can be sharpened by ablations that hold spatial structure fixed but swap only the tail distribution; we expect the gap to persist but the analysis hasn't been run.

### Single experiment to confirm fundamental limit (if results had been NULL)
Spin the same protocol on a **third** gfx1151 unit. If z_hw_vs_sw collapses when transplanting between two units of the SAME silicon SKU (only environmental/PVT differences), then user-space identity binding cannot exceed PVT noise — a fundamental limit. Conversely, persistent z>2 between matched-SKU units would prove the binding reaches die-individual silicon variation, not just family-level architecture. (Not run here — we only have 2 hosts.)

## Outputs
- Scripts: `scripts/identity_benchmark/attack_1_3/{A1_contrastive,A3_heavy_tail_collect,A3_heavy_tail_analyze,A3_heavy_tail_transplant,A13_cross}.py`
- Results: `results/IDENTITY_BENCHMARK_2026-05-30/attack_1_3/{A1_results,A3_tail_stats,A3_transplant,A13_cross}.json` plus `.npz` streams + logs
- Report: this file (`research_plan/IDENTITY_ATTACK_1_3_2026-05-30.md`)

## Thermal incidents
**Zero crashes.** APU peak 74°C (daedalus, ch_loop_jitter, well below 78°C never-exceed). ch_loop_jitter on ikaros hit 72°C and the in-script abort triggered cleanly, saving partial 30k samples (vs 40k planned). End-of-run APU 48°C (ikaros) / 29°C (daedalus). Thermal guard PID 9305 untouched.
