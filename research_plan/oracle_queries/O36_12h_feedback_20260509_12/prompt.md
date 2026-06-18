# 12-hour gap-closing review (NS-RAM Path A — post z233 negative + z234 retuned positive)

You are reviewing 12 hours of autonomous research progress. Context.md
contains log entries from O35 oracle (your last consensus call) through
work-hours #42 (z234 retuned seq-MNIST). The story has shifted
significantly twice in the last 12h:

**Shift 1 (z233, work-hours #37)**: At your O35 recommended experiment
(frozen NARMA-10 hyperparams on 28×28 seq MNIST, 30 seeds GPU N=2k),
the result was a clear NEGATIVE — reservoir 37.4% vs pure-projection
42.0%, Δ = −4.7pp, 95% CI [−5.5, −4.0], paired-t p = 8e-17.
Framed in Mario v2 draft as "frozen NARMA config does NOT generalize."

**Shift 2 (z234, work-hours #42)**: Then we ran a small 4-config × 3-seed
hyperparameter sweep (audit-recommended fallback). Result inverts the
narrative:

  frozen (0.30 leak, 0.05 g_VG2):       Δ = −4.50 pp  (matches z233)
  more_memory (0.70 leak, 0.05 g_VG2):  Δ = −4.83 pp
  **strong_input (0.30 leak, 0.20 g_VG2):  Δ = +10.50 pp  ✅**
                  (all 3 seeds: +9.5, +10.5, +11.5)
  both_tuned (0.70 leak, 0.20 g_VG2):   Δ = +6.83 pp

A SINGLE hyperparameter retune (g_VG2: 0.05 → 0.20, 4× stronger
input drive) flipped the frozen result from −4.5pp to +10.5pp.

The Mario v2 draft was written between these two findings and asserts
"NS-RAM-as-reservoir is currently a one-task-at-a-time system —
task-class transfer requires per-task hyperparameter tuning OR
architectural extension." This now reads as more pessimistic than
warranted: a SINGLE knob retune recovered cross-task in 13 minutes
of compute.

## Three questions

**Q1.** Has the z234 +10.5pp result crossed a decision gate that should
update Mario brief framing? Specifically: is 4 configs × 3 seeds
sufficient to claim "task-tuning recovers cross-task generalization"
in the brief, or does it need 30-seed CI at strong_input first
(matching z223 NARMA / z233 frozen-MNIST power)? Note: the +9.5 /
+10.5 / +11.5 pattern across 3 seeds is consistent in direction
but only n=3.

**Q2.** What is the strongest cherry-picking or statistical risk in
the z234 result? In particular: only 4 configs were tested; the user
chose them based on intuition (more_memory, strong_input, both_tuned).
Is +10.5pp at strong_input plausibly a lottery effect from
hyperparameter selection over a small grid, or is the 3-seed
consistency strong enough to rule that out?

**Q3.** Next highest-value single experiment for next 12 hours given:
(a) z234 needs replication at higher seed count to be brief-headline
worthy; (b) Mario v2 draft + figures already produced and ready to
revise; (c) compute envelope GPU N=2k stable, APU peak 92°C in z234
right at thermal kill threshold, must lower temp pressure; (d) Sebas
characterisation request still user-blocked, 4+ days unsent. Be
specific: name the single experiment and its acceptance gate.
