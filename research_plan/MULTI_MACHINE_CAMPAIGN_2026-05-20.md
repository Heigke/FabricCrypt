# Multi-Machine NS-RAM Use-Case Campaign — 2026-05-20

## Honest validated config (cell model)
- K1@VG1=0.6 = **0.53825** (BSIM card)
- ALPHA0 = **7.83756e-4** (Mario LALPHA0_FIX card)
- `cfg.tlpe1_disable = True` (matches ngspice b4ld.c lpeb cross-coupling)
- Hurkx OFF (fake, removed)
- `well_diode_mode = "legacy_into_body"` (current best fit per track_well_diode_fix)
- Full-33 median dec: **0.461** (PASS sub-0.5)

## HONEST CAVEAT
The 5 N-* use-case scripts dispatched here consume the existing
**precomputed 4D surrogate** (`results/z278_mep2_surrogate_v3/surrogate_4d_v3.npz`
or `z271_pmp3_dense_surrogate/surrogate_4d_v2.npz`). Those surrogates were
generated *before* the K1+ALPHA0+Tlpe1 closure. They therefore reflect the
pre-Tlpe1-fix cell. The validated config improves DC dec to 0.461 but
**ngspice gap remains 0.808 dec — honest caveat, not closed.**
Surrogate regeneration with the validated config is a follow-up campaign.

## Use-case → machine map

| ID | Script | Machine | Surrogate | Budget |
|----|--------|---------|-----------|--------|
| N-HDC-MNIST-3K | scripts/N_HDC_UCIHAR.py | ikaros | z278_v3 | 1h |
| N-Res-NARMA | scripts/N_Res_MG.py | zgx | analytic | 1h |
| N-LIF-MNIST | scripts/N_Hier_MNIST.py | daedalus | z278_v3 | 1h |
| N-STDP-ECG | scripts/N_STDP_ECG_N100_v2.py | zgx | z271_v2 | 1h |
| N-Cascade-KWS | scripts/N_Cascade_KWS_ECG.py | ikaros | varies | 1h |

(N-HDC-MNIST-3K uses the UCI-HAR HDC pipeline as the existing HDC benchmark;
MNIST-3K naming is for campaign labelling — real dataset is UCI-HAR with N=8192.)

## Dispatch ledger

| Use-case | Machine | PID/Status | Result path |
|----------|---------|------------|-------------|
| N-HDC-MNIST-3K | ikaros | dispatched | results/N-HDC-MNIST-3K-ikaros/ |
| N-Res-NARMA | zgx | dispatched | results/N-Res-NARMA-zgx/ |
| N-LIF-MNIST | daedalus | dispatched | results/N-LIF-MNIST-daedalus/ |
| N-STDP-ECG | zgx | dispatched | results/N-STDP-ECG-zgx/ |
| N-Cascade-KWS | ikaros | dispatched | results/N-Cascade-KWS-ikaros/ |

## Pre-registered gates (script-defined)
- N-HDC: DISCOVERY test_acc>0.70, AMBITIOUS test_acc>0.85
- N-Res-MG: NRMSE<0.1 ambitious
- N-LIF-MNIST: train_acc>0.80
- N-STDP-ECG: AUROC>0.85
- N-Cascade-KWS: TPR>0.80 @ FPR<0.10

## NO-CHEAT pledge
- No metric editing; if a use-case fails, the synthesis reports the raw number.
- ngspice 0.808 dec gap documented as open.
- Pre-Tlpe1-fix surrogate caveat documented above.

## Synthesis target
`research_plan/CAMPAIGN_SYNTHESIS_2026-05-20.md` after all 5 land.
