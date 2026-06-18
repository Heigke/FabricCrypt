# Phase 2 v2 — Envelope Substrate Transplant Matrix

**Date**: 2026-05-30   **N seeds**: 30   **Substrate features**: 23

||z_ikaros - z_daedalus||_2 = **8.485** (distance of the two device signatures in shared z-space)

## NARMA-10 transplant

| variant | diag NRMSE | off-diag NRMSE | Δ (off−diag) | n_seeds |
|---|---|---|---|---|
| HW | 0.5847 [0.5749, 0.5947] | 27.0166 [21.9391, 32.5873] | +26.4319 [+21.2347, +32.3246] | 30 |
| SW_MATCHED | 0.5855 [0.5757, 0.5955] | 112.9410 [89.7681, 136.6461] | +112.3556 [+87.6011, +140.3575] | 30 |
| NO_SUB | 0.5819 [0.5722, 0.5918] | 0.5819 [0.5722, 0.5918] | +0.0000 [+0.0000, +0.0000] | 30 |

SHUFFLE: mean NRMSE = 116.7312 CI [96.84734553468628, 139.24101287166812]


**HW Δ vs SW_MATCHED z-score = -1.15σ**   SHUFFLE flat? **False**


## Permuted-MNIST lite (5 tasks, K=4 classes)

| variant | diag acc | off-diag acc | Δ (off−diag) |
|---|---|---|---|
| HW | 0.2491 | 0.2563 | +0.0072 |
| SW_MATCHED | 0.2487 | 0.2537 | +0.0050 |

Cross-task transplant degradation: **False**


## Verdict

- HW Δ NRMSE: +26.4319  σ=15.3640
- SW_MATCHED Δ NRMSE: +112.3556  σ=74.6563
- z(HW vs SW_MATCHED) = -1.15σ   (gate: >2σ)
- SHUFFLE flat: False
- Cross-task pMNIST corroboration: False

### **PHASE 2 v2 VERDICT: NULL**


Interpretation:
- The 23-feature envelope substrate is also FUNGIBLE on NARMA-10. Off-diagonal transplant does not degrade more than software-matched Gaussian envelope of the same mean/std. The HW silicon-bound channels (power/thermal/per-core latency) discriminate the *devices* with Cohen d≥3, but they do NOT propagate into a learned reservoir readout in a device-specific way. Identity remains *recognisable* but not *constitutive*.