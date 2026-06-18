# 12-hour gap-closing review (NS-RAM Path A + Mario brief preparation)

You are reviewing 12 hours of autonomous research progress on the NS-RAM
2T-cell project. Context.md contains the last ~13 wake-up entries from
01_LOG.md spanning roughly 12h (from work-hours #22 through #34 plus two
6h track audits).

Topline state of the project:
- Path A (4D body-state surrogate + reservoir) is built and converged
  to NRMSE 0.6122 ± 0.030 on NARMA-10 at 30 seeds (z223), beating the
  prior z216 baseline of 0.84 by 27%.
- GPU port complete (z225-z228) up to N=20k via manual block-loop
  (PyTorch sparse_csr crashes at N≥5k on ROCm 6.3 / gfx1151;
  workaround uses torch.bmm of (K, n, n) × (K, n, 1) → (N,)).
- R-track 3-source triangulation closed: surrogate ↔ pyport (z230, max
  0.39 dec), pyport ↔ ngspice (z231, max 0.51 dec marginal),
  transitive bound surrogate ↔ ngspice ≤ 0.90 dec. M2-OFF leakage tail
  is over-predicted by ~3× in pyport.
- z232 25-bias bootstrap: lumped solver converges 0/25 in reservoir
  bias regime. Last wake-up's "lumped vs q2d is real physics divergence"
  was WRONG — lumped's "low-Id" was non-converged Newton iterates.
  Branch-protect hypothesis falsified at 25 N (CI [+0.71, +1.61] dec
  vs lumped, does not bracket 0).
- C-track CLOSED: NS-RAM 0.7 µJ for 1024-step inference at N=64 vs
  ARM Cortex-M4 50-100 µJ vs MAX78000 5 µJ vs Coral 10 µJ. ~10× vs
  best AI MCU.
- Cross-task generalization (z224 sequential 8x8 digits) NEGATIVE —
  reservoir 52% vs pure projection 56% on classification.

Mario (NRF brief target) state: brief v4.1 was sent earlier with the
honest "1.39 dec at Bf=100 + η ≤ 1" framing. The mario_update_note_draft.md
remains UNSENT and is now stale (written before z230/z231/z232 corrections).

## Three questions

**Q1.** Have any results in the last 12h crossed a decision gate
(Week-1 CI, scale-gap, area-matched)? What concrete action does that
decision now require — keeping in mind the user can only forward a
brief; he cannot re-run experiments. List concrete decisions, not
optional next steps.

**Q2.** Looking at the trajectory in context.md, what is the strongest
cherry-picking or statistical pitfall you see in this 12-hour window?
Specifically: is the NRMSE 0.6122 generalization claim (z221→z223) at
risk from selection effects given that hyperparameters (Cb=5fF, dt=500ns,
g_VG2=0.05, leak=0.30) were chosen on z221 and validated on the same
NARMA-10 task? Was z224's negative cross-task result given enough power
to be conclusive (5 seeds, 8x8 digits)?

**Q3.** Pick the single highest-value experiment for the NEXT 12 hours,
given: (a) compute envelope = 1× APU (32-core, sentinel-protected,
thermal_zone0 trip 99°C, must use util_safe_sweep wrapper); (b) GPU
N=2k-20k now reliable; (c) Mario brief is the ultimate deliverable;
(d) Sebas thick-ox card request is user-blocked.

Be specific: name a script, a dataset, a hyperparameter sweep, or
a stat power calculation. Justify why this is the single highest-value
move and not the second-best.
