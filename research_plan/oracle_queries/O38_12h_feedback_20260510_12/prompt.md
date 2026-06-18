# 12-hour gap-closing review — NS-RAM scope-bound monotonic claim

You're reviewing 12 hours of progress since O37. Major events:

**z239 (CIFAR-10 attempted, openml HTTP 504, fell back to FashionMNIST)**:
unintended replication — Δ=-9.75pp matches z236's -10.6pp within
overlapping CIs.

**z240 (CIFAR-10 grayscale 28×28 from local cache, n=8)**: projection
baseline = 15.3% (much lower than MNIST-family 43-72% band). Δ_actual
= +1.94pp, CI [+1.0, +2.75], 7/8 positive, p=0.001. Predicted from
4-task linear fit at proj=15%: +21.13pp. **|error| = 19.2 pp** — sign
matches but magnitude wildly off. Linear extrapolation BREAKS outside
MNIST-family band.

**Mario v2 + monotonic figure scope-bound**: linear fit now restricted
to 40-75% MNIST-family band; CIFAR plotted outside with explicit
"extrapolation fails by ~10×" annotation. Mario v2 cross-task
paragraph rewritten to "linear within MNIST-family band; saturates
outside, direction-only". This replaces the unbounded-linear overclaim
from work-hours #51/#53.

**Saturation analysis on 5 points**: linear fit on MNIST-band (4 pts)
R²=0.997 ✅. Linear extrapolated to 5 pts R²=-0.81 ❌. Sigmoid 4-param
on 5 pts R²=0.97 (overfit risk). Honest conclusion: 1 datapoint
outside band insufficient to fit saturation form.

**z241 g_VG2 sensitivity sweep (5 g_VG2 values × MNIST)**: SMOOTH
gradient confirmed.
  g_VG2=0.05: Δ=-4.67  g=0.10: Δ=-2.10  g=0.15: Δ=+2.30
  g_VG2=0.20: Δ=+5.10  g=0.30: Δ=+9.60
Approximately linear: Δ ≈ -7.4 + 56·g_VG2. **NO peak, NO winner's
curse**. O37 risk c (knob-choice winner's curse) DEFINITIVELY REJECTED.

## Status

O37 risks resolution status:
  - (a) Task-modality confound: PARTIAL — direction-claim survives
        CIFAR test, magnitude-claim falsified outside MNIST band
  - (b) Functional form (linear vs saturating): observed saturation
        but undetermined functional form (need more out-of-band points)
  - (c) Hyperparam winner's curse: REJECTED via z241 smooth gradient

## Three questions

**Q1.** With the scope-bound linear claim (within 43-72% MNIST-family,
fit ±0.5pp) AND the smooth-gradient g_VG2 sweep (knob-robustness
demonstrated), has Mario v2 reached a "send-ready" state? What
specific tightening or last-mile improvement should be made before
sending?

**Q2.** What is the strongest remaining residual risk now that
winner's-curse is ruled out? Specifically: is the linear-within-band
claim itself robust to (i) linear-classifier choice (logistic vs
ridge), (ii) projection-baseline definition (current = mean of
W_in @ rows), (iii) reservoir size N? We've fixed all three at
default values; would varying any reveal that the relationship is
brittle?

**Q3.** Pick the single highest-value experiment for next 12 hours.
Options on the table: (i) one more out-of-band datapoint (e.g.
SVHN proj~25%, EMNIST-letters proj~30%) to constrain saturation
form; (ii) ESN-control on one task to test "pipeline vs NS-RAM"
attribution (O37 risk a); (iii) vary N or readout-classifier on
existing tasks to test claim robustness. Compute envelope: GPU N=1k
stable, APU peaks ~90°C, 12h window allows ~1.5h compute total.
