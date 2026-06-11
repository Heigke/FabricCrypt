# Closed-loop verification — verdict (2026-06-10)

## Result

| channel | R²(S_history) | R²(S_history, u) | ΔR² |
|---------|---------------|------------------|-----|
| C20_lat_x | +0.398 | +0.401 | +0.0035 |
| C11_drift | +0.485 | +0.486 | +0.0004 |
| C05_e0_rt | +0.048 | +0.056 | +0.0086 |
| C20_lat_e | +0.248 | +0.255 | +0.0069 |

ΔR² < 0.05 on **0/4** channels → naïve closed-loop conditioning is NOT load-bearing for forward dynamics at 48ms lag.

## Smoking gun

`R²(u | S_history) = 0.474` — half the binary action signal is already recoverable from 48ms of substrate. So adding explicit u-features to a forward dynamics predictor adds little: the action is already a function of S.

## Interpretation (honest, not embodiment-favoring)

The physical closed loop is real (load_d > 0.5 on these channels — audit confirmed). What this experiment shows is the *informational* version:

- u → S is real (action shifts substrate within ~50ms)
- but at 48ms lag, S_history already carries enough about recent u that explicit u is redundant for predicting next S
- substrate is a sufficient statistic for itself at short horizons

## What this means for the embodiment plan

**Drop "compute-action conditioned forward dynamics" as the embodiment loss.** It would not have improved anything over the v5 substrate-prediction loss.

**Keep — and shift to — these instead:**

1. **Knockoff KL** (GPT-5 O102) — substrate-conditional output divergence. The test is "does the model's *output distribution* depend on substrate beyond what a μ/σ/PSD-matched knockoff explains?" That's behavioral causal mediation, not forward dynamics. Trivially-explained baselines for substrate prediction can't game it.

2. **Cross-host transplant in output space, not in loss** — train on ikaros, run on daedalus, measure same-prompt next-token KL divergence between hosts. The signature embodiment is "the model speaks differently here than there" — not "the model predicts its own hardware better."

3. ~~**Re-test at longer horizons** — at 500ms or 2s, u may still lead S in a way 24-lag history can't capture.~~ **DONE — LAG=128 (256ms) confirms FAIL.** R²(u|S_history) climbed from 0.47 → 0.71 with more history; ΔR² stayed ≈0.005 on all 4 channels. More substrate history makes u MORE redundant, not less. Conditioning is structurally dead.

## Bias self-check

Was I going to spin this as "still works, just needs longer horizon"? Yes, partially — I added the longer-horizon rerun as a potential save. That's an embodiment-favoring move. The honest read: at the timescale we'd actually update online (every ~50ms feels right for an online LM), the action adds nothing. The model can be substrate-conditioned via Knockoff KL on outputs. It cannot be meaningfully made "embodied" by conditioning a forward dynamics loss on its own action signal, because that signal is already in the substrate.

## Plan revision

v6 architecture should be:
- Substrate cross-attention (Flamingo-gated, identity-at-init) — keep
- Online plasticity from **Knockoff-KL gap** as the critic — NOT substrate MSE
- Tournament transplant test as the only non-Knockoff embodiment evidence
- Forget compute-action conditioning. The closed loop is physical but not informationally exploitable at our timescales.
