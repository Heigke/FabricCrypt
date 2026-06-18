# O27 — Final pre-send check after placeholder swap

## Context

NRF deadline tomorrow (2026-05-06). The brief was already reviewed
by you (gpt-5) and gemini in O26 with hostile-reviewer framing; all
**P0 honesty fixes were applied** (softened "structural floor" →
"observed plateau / evidence requires architectural change", explicit
"three architectures NOT tested in v4.2 because ~200 LOC each"
disclaimer, NRMSE→log-RMSE typo, Bf=100 vs Bf=9000 inconsistency
addressed).

Since O26 there is **one material change**: Figure 2 (transistor-
level physics) had two `figplaceholder` boxes asking Pazos/Lanza to
supply slides. Both have now been replaced by the actual measured
fit-vs-prediction figure from `z91g_two_model_validation_F6v4_bf9000_va0.55`
(median log-RMSE = 0.657 dec stamped right in the figure). The
caption was rewritten to fold the "three operating regimes" story
into the three $V_{G1}\!\in\!\{0.2, 0.4, 0.6\}$ panels.

The brief is now placeholder-free and 10 pages.

## QUESTIONS

1. **Did the placeholder swap regress anything?** Look at Fig 2 in
   the attached PDF: caption + figure. Is the message clear (three
   regimes selected by $V_{G1}, V_{G2}$, real measurement vs pyport
   fit) or did dropping the second subfigure lose narrative?

2. **Last-mile sanity**: any factual mismatch you can spot between
   the figure caption (median log-RMSE = 0.657) and the abstract /
   §status headline (0.654)? They differ by 3 mdec because the
   figure is the optimum-cell PNG and the headline median is across
   the 25/33-bias dataset; is this confusing?

3. **One-shot honesty rating** (1–5, 1=clean, 5=red-flag): does the
   v4.2-final read as defensible to a hostile NRF reviewer?

4. **One-line gut check**: would you send this tomorrow?

Be terse — under 200 words total. The brief is otherwise frozen;
only catastrophic regressions warrant another revision before
deadline.
