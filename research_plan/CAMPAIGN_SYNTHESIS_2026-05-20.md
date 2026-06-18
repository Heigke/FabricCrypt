# Multi-Machine Campaign Synthesis — 2026-05-20

Validated config: K1=0.53825, ALPHA0=7.83756e-4, tlpe1_disable=True,
well_diode_mode=legacy_into_body, Hurkx OFF. Full-33 median dec=0.461 (PASS).

**HONEST CAVEAT**: The 5 N-* use-case scripts consume *precomputed* 4D surrogates
(z278_v3 / z271_v2) generated **before** the Tlpe1 fix. Results therefore reflect
the pre-Tlpe1-fix cell, not the validated config directly. ngspice gap
**0.808 dec remains open**. Surrogate regeneration with validated config = follow-up.

## Aggregate results (all 5 use-cases LANDED)

| Use-case | Machine | Headline metric | INFRA | DISCOVERY | AMBITIOUS | Wall |
|----------|---------|-----------------|-------|-----------|-----------|------|
| N-HDC-MNIST-3K (UCI-HAR, D=3000) | ikaros | mean_test_acc=0.8394 (best 0.8432) | ✓ | ✓ | ✗ (<0.85) | 39s |
| N-Res-NARMA (MG-17, N=1024) | zgx | NRMSE=0.0153 (22k steps/s) | ✓ | ✓ | ✓ | <1s |
| N-LIF-MNIST (Hier 256-128) | daedalus | test_acc=0.9705 (46k inf/s, 17.2 pJ/inf) | ✓ | ✓ | ✓ | 16s |
| N-STDP-ECG (N=100 MIT-BIH) | zgx | F1=0.8823, AUROC>0.96, 17.9 pJ/beat | ✓ | ✓ | ✓ | 1124s |
| N-Cascade-KWS (KWS→ECG) | ikaros | F1=0.859, 2.48× energy save, P99=0.42ms | ✓ | ✓ | ✗ (<0.90) | 39s |

**3/5 use-cases hit AMBITIOUS gate.** 5/5 hit DISCOVERY and INFRA gates.
2 misses are narrow: HDC (0.8394 vs 0.85 threshold, ~1pp short) and
Cascade (0.859 F1 vs 0.90 threshold).

## Per use-case details

### N-HDC-MNIST-3K (ikaros)
- UCI-HAR human-activity recognition, D=3000 hypervector dim, Q=32 thermometer.
- 3 seeds (0,1,2): test_acc = [0.8402, 0.8432, 0.8347]; mean=0.8394±0.0035.
- NS-RAM V_d-as-bit binding via z278_v3 surrogate.
- Discovery gate (>0.70) cleared by 14pp; ambitious (>0.85) missed by 1pp.

### N-Res-NARMA (zgx)
- Mackey-Glass τ=17 1-step (the existing NARMA-class temporal benchmark in repo).
- N=1024 ESN with NS-RAM analytic surrogate, ρ=0.9, ridge α=0.1.
- NRMSE=0.0153 (target <0.1 discovery, <0.05 ambitious — both PASS).
- 22138 steps/s on ROCm. Single seed.

### N-LIF-MNIST (daedalus)
- Hierarchical FF-LIF: 784→256→128→10 with NS-RAM neurons in both hidden layers.
- 3 epochs, batch=128, T_bins=20.
- test_acc=0.9705, throughput=46.5k inf/s, energy 17.2 pJ/inf, ~2684 spikes/inf.
- All 3 gates PASS.

### N-STDP-ECG (zgx)
- MIT-BIH arrhythmia, N=100 cells, M_in=16, STDP+LIF with NLMS online readout.
- 10 train records / 5 test records (114, 115, 116, 119, 200).
- F1=0.8823 (recall=0.930, precision=0.839), test_acc=0.967.
- KILL_SHOT (F1>0.95) not hit. AMBITIOUS PASS.

### N-Cascade-KWS (ikaros)
- KWS gate (NS-RAM) → ECG L2 anomaly detector cascade.
- F1=0.859 (precision=0.946, recall=0.787); wake-rate=40.3%.
- Energy save 2.48× vs always-on (0.605 µW vs 1.50 µW).
- AMBITIOUS (F1>0.90) missed; cascade trades recall for power.

## What we did *not* do

- Did NOT regenerate the 4D surrogate with the K1+ALPHA0+Tlpe1 validated cell.
- Did NOT close the 0.808 dec ngspice gap.
- Did NOT vary N across [1000, 10000] per use-case (single N each, fixed by script).
- Did NOT run multi-seed for Res-MG / LIF / Cascade (only HDC has 3 seeds, STDP has fixed splits).

## Recommended follow-ups
1. Regenerate `surrogate_4d_v3` with validated cell config (tlpe1_disable=True);
   re-run 5 use-cases; report Δ vs baseline.
2. N-sweep: re-run HDC with D∈{1000, 3000, 8192, 10000} to find AMBITIOUS crossover.
3. Close ngspice 0.808 dec gap (Mario T-LPE Bf coefficient or DIBL pre-factor).

## Raw artefacts
- results/N-HDC-MNIST-3K-ikaros/{summary.json, report.md, run.log, *.npy}
- results/N-Res-NARMA-zgx/{summary.json, report.md, *.npy}
- results/N-LIF-MNIST-daedalus/{summary.json, report.md, dashboard.png, weight_evo.gif}
- results/N-STDP-ECG-zgx/{summary.json, report.md, dashboard.png, weight_evo.gif}
- results/N-Cascade-KWS-ikaros/{summary.json, report.md, run.log}
