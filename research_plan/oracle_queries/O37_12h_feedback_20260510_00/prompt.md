# 12-hour gap-closing review — NS-RAM cross-task quantitative claim

You're reviewing 12 hours of progress. Context.md spans entries from
work-hours #43 (z235 30-seed replication) through #53 (4-point fit
update). The story consolidated dramatically:

**Cross-task narrative trajectory in this 12-hour window:**

1. **z235** (n=25): retuned config (g_VG2=0.05→0.20) on seq-MNIST gave
   Δ=+5.1pp, 25/25 seeds positive, p=9e-18. Showed retune helps MNIST.

2. **z236** (n=10): SAME retuned config on FashionMNIST gave Δ=−10.6pp,
   0/10 positive, p=1.4e-11. Showed retune is task-specific.

3. **z237** (n=8): SAME retuned config on KMNIST (Japanese cursive) gave
   Δ=+2.6pp, 8/8 positive, p=0.001. Pattern emerged: 3 datapoints
   showed monotonic decrease in Δ with projection-baseline strength.
   Linear fit Δ ≈ +29.6 − 0.56·proj%, zero-crossing 53%.

4. **z238** (n=8): controlled out-of-sample test — SAME task as
   FashionMNIST but train=200 (vs 1000), forcing baseline 72%→68%.
   3-task fit predicted Δ=−8.24pp at proj=68%.
   Actual: Δ=−8.56pp, CI [−10.0, −6.5], 0/8 positive.
   **Predicted within 0.32pp of actual; PREDICTION IN CI.**

5. 4-point refit: Δ ≈ +29.8 − 0.56·proj% (essentially unchanged from
   3-point). New point lands on the line.

**Mario v2 framing now**: "Reservoir contribution Δ is QUANTITATIVELY
PREDICTABLE from linear baseline strength alone, validated within
±0.5pp across 4 tasks spanning proj 43–72%." All NS-RAM hyperparams
held constant (leak=0.30, g_VG2=0.20, N=1000) across all 4 tasks.

## Three questions

**Q1.** Has the z238 quantitative validation crossed a decision gate
that lets the brief make a quantitative-prediction claim, OR is 4
datapoints with sample sizes 8–25 still preliminary? If preliminary,
what specific seed-count or # tasks would clear the gate?

**Q2.** What is the strongest residual risk in this monotonic claim?
Specifically: (a) all 4 tasks are MNIST-family-like 28×28 grayscale —
is the fit a property of NS-RAM or a property of the
projection-baseline+linear-classifier pipeline that would emerge with
ANY mild reservoir? (b) the linear fit is on 4 points; is "linear"
plausibly correct vs saturating/threshold? (c) winner's curse on
hyperparam choice still applies — is g_VG2=0.20 the only retune that
gives this monotonic relationship?

**Q3.** Pick the single highest-value experiment for next 12 hours
that would either FAIL the monotonic claim or strongly EXTEND its
validity range. Be specific (script, dataset, seeds, acceptance gate).
Compute envelope: GPU N=1k stable, APU peaks 88-91°C.
