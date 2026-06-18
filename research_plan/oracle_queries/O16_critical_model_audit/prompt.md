# O16 — Critical model audit. Be brutal.

## Context

We are funding a research proposal that claims a PyTorch BSIM4 port
(*pyport*) of Sebastian Pazos's 130 nm 2T NS-RAM cell faithfully
reproduces his measured I–V family at 1.00-decade median log-RMSE
(under a calibration patch) and 1.88-decade faithful (without).
Deadline 2026-05-06; recipient is Mario Lanza (KAUST tape-out lead).

The user funding the project (Eric Bergvall) has just told us he is
**worried that the model is not actually good enough** — he has seen
the fit plots and noticed that some prediction lines are flat or
wrong while measurements show real structure. He asked us to do a
genuinely critical recheck before continuing with downstream demos
and the proposal send.

You are the critical-review oracle. **Do not reassure. Find every
weakness.** Be the most hostile possible reviewer this proposal could
encounter.

## What's attached

1. **`fit_vs_meas_PATCHED_1.00dec.png`** — Three-panel fit plot at
   VG1 ∈ {0.2, 0.4, 0.6} V, with 33 measured I–V curves (open
   symbols) and pyport predictions (lines), with the calibration
   patch applied. This is the headline result quoted in the brief
   ("median 1.00 dec").
2. **`fit_vs_meas_FAITHFUL_1.88dec.png`** — Same plot but without the
   calibration patch (using nspice-faithful binning). Headline
   median is 1.88 dec instead.
3. **`predictions_patched.json`** — the per-bias data behind plot 1.
   Each entry has `VG1`, `VG2`, `Vd[]`, `Id_meas[]`, `Id_pred[]`,
   `converged[]`, and `log_rmse`. Use this to compute the actual
   distribution of per-bias RMSE — is "median 1.00 dec" misleading
   because of a long tail of bad biases? How many biases are >1.5
   dec? >2.0 dec?
4. **`nsram_proposal_short.pdf`** — the funding brief that depends
   on this fit being credible.

## Five questions, plus a verdict

Answer the five questions in numbered sections. Do not pad. If you
have nothing critical to say in one section, say so explicitly.

1. **Does the patched fit actually look good in the plot?** Look at
   each VG1 panel separately. Identify specific curves where the
   prediction line is flat / detached / wrong-magnitude. Quote
   approximate (VG1, VG2) coordinates if visible. Distinguish
   "shape mostly right" from "shape wrong" cases. The user has
   noticed flat lines specifically — confirm or deny.

2. **What does the per-bias log-RMSE distribution actually look
   like?** Compute or eyeball from `predictions_patched.json`:
   median, mean, max, count of biases with RMSE > 1.0, > 1.5, > 2.0.
   Is "median 1.00 dec" a fair summary of the fit, or is it hiding
   a heavy tail? What is the WORST single bias and how bad is it?

3. **Is the brief defensible at the current fit quality?** Walk
   through the brief's Status section and Limitations bullet 5 and
   tell us: do the qualitative claims (monotonic task ordering,
   topology recommendation, sign-asymmetry result) actually depend
   on the fit being good in absolute terms, or only on shape /
   relative comparisons? Where does a hostile reviewer have a
   legitimate attack vector? Be specific — quote sentences from the
   PDF that you believe overclaim, if any.

4. **What's the worst-case interpretation?** Imagine the harshest
   reviewer (a senior compact-modelling expert with a grudge). What
   would they say about this fit? What single sentence would they
   use to dismiss the whole proposal? Don't tone-soften it.

5. **What MUST be fixed before send, vs what can wait for M3a?**
   Given the deadline is 2026-05-06 and M3a is committed as a 1–2
   day audit + 3 days verification, what's the minimum subset of
   fixes that need to land in the brief itself before we send? And
   what subset is genuinely OK to defer to M3a?

**Final verdict:** Choose ONE: (A) "Send it as-is, the framing is
already honest enough"; (B) "Fix [list] in the brief first, then
send"; (C) "Don't send — the gap is large enough that the proposal
will harm Eric's credibility with Mario; fix M3a first, send after."

We are scientists. We can take harsh feedback. The user prefers a
sharp warning over polite reassurance.
