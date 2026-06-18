# Mario transmittal email — draft v2 (2026-05-05, post-O26)

**Status:** Draft for v4.2-final brief. Awaiting user review +
authorization to send.

Replaces v1 (2026-05-03), which was written for the v3 brief
(0.795 dec, dichotomy-table headline) and is now stale.

**Send target:** Mario Lanza, KAUST. Cc: Sebastian Pazos.

**Suggested subject line:**

  > NS-RAM 2T-cell brief — calibrated DC fit + ER_SPARSE tape-out
  > recommendation, NRF-ready

(Alternative, more conservative: "NRF brief — NS-RAM cell
co-design, ready for your review")

---

## Email body

> Mario,
>
> Attached is the v4.2-final NS-RAM 2T-cell brief for the NRF
> submission (10 pages, post-O18–O26 oracle review). Both
> documents are ready for your review.
>
> The headline result is a **calibrated DC fit at median log-RMSE
> 0.654 dec** on 25/33 evaluated biases of Sebas's measurement
> set, at the physically-defensible lateral-parasitic-NPN point
> ($B_f=9{\times}10^3$, $V_a=0.55$ V, $I_s=10^{-9}$). The journey
> from baseline 1.39 → 0.654 dec is six productive parameter
> sweeps, then four oracle-ranked candidates (IKF, ISE/NE,
> PRWG/Rdsw, $\eta(V_{be})$ sigmoid) that all came up null at
> 3 mdec spread. We frame this as **strong evidence that further
> gains require an architectural change**, not a parameter
> tweak — three remaining options (two-NPN, quasi-2D body,
> body-network) are explicitly listed as M3b/M6 deliverables; we
> chose not to implement them inside the v4.2 window because each
> needs a new Newton state and solver-Jacobian extension.
>
> On top of the calibrated cell, a 64-cell ER_SPARSE reservoir
> forecasts chaotic Mackey–Glass at NRMSE 0.747, and the cross-
> normalisation topology sweep (z142, 270 simulations) identifies
> ER_SPARSE as the single architecture stable across three
> reasonable $\rho$-normalisations — that is the tape-out target.
> The full cell-parameter and routing recommendation is in
> §"Cross-norm topology recommendation" of the brief.
>
> Two practical notes:
>
> 1. **Path to silicon-traceable.** Two characterisation runs on
>    your existing silicon convert the fit-defensible $(B_f, I_s)$
>    point into a silicon-grounded one, neither requires a new
>    fab cycle:
>      - $I_c/I_b$ ratio at saturation (one bias) → direct $B_f$
>        extraction.
>      - Pulsed-$V_d$ / TLP at one bias, $100$ ns–$10\,\mu$s,
>        fixed gates → extracts the $R_b\!\cdot\!C_b$ time
>        constant.
>    These are the M3b deliverable; if Sebas can give a rough
>    ETA we tighten the schedule.
>
> 2. **Honest scope limits, all in the brief's Limitations §:**
>    8/33 biases excluded ($K_1 =$ NaN at negative $V_{G2}$);
>    NARMA-10 NRMSE plateaus at $\approx 0.95$ at the honest
>    cell (architectural change to lift it queued in Phase B);
>    2T internal-node ngspice cross-check shows up to $\sim 100$
>    mV $V_b$ divergence (pyport adds $\eta$-bounded lateral
>    injection ngspice omits — DC currents agree to $1$–$2\%$).
>
> Compute and personnel budget on Page 8 are unchanged: existing
> Daedalus + Ikaros workstations, 0.8 FTE Bergvall, 0.3 FTE
> Luciani (optional, Julia/Enzyme cross-val), Pazos/Lanza
> in-kind. Tape-out review trip to KAUST in Q3 2026 stays in.
>
> If the brief is aligned with what NRF needs from your end, I
> will prepare the formal submission packet. If anything reads
> off, I can turn around the next revision in 24-48 h.
>
> Best,
> Eric

---

## What is intentionally NOT in this email

- No claim of "structural floor confirmed" — softened to
  "observed plateau / evidence requires architectural change"
  per O26 oracle synthesis (gpt-5 + gemini both flagged the
  stronger phrasing as overreach).
- No promise on the three untested architecture options
  (two-NPN, quasi-2D body, body-network) — explicitly framed
  as M3b/M6, not v4.2 deliverables.
- No upstream-ngspice timeline beyond the brief's prose.
- No commitment to a specific NRF deadline beyond
  "24-48 h turnaround if needed".
- No attachment list spelled out — leave to email client.

---

## Suggested attachments

  - `nsram_proposal_short_v4_2.pdf` — 10 pages, 427 KB.
    (Or the Overleaf zip
    `nsram_proposal_short_v4_2_overleaf_2026-05-05.zip`, 582 KB,
    if Mario prefers source.)
  - Optional follow-up if reviewer asks where residuals
    concentrate: `figures/per_row_residuals_optimum/per_row_residuals.pdf`
    — shows worst-5 rows all sit at $V_{G1}=0.40$ V (the
    parasitic-NPN ignition corner).

---

## Pre-send checklist

  - [ ] Read the email body for tone (less aggressive than v1;
    leads with calibrated number rather than 5-row dichotomy).
  - [ ] Confirm cc to Sebas is appropriate (he is in M3b path).
  - [ ] Confirm subject line.
  - [ ] Confirm attachment(s): PDF only, or PDF + Overleaf zip.
  - [ ] Decide whether to mention the per-row residual figure
    proactively or hold as defensible-on-demand.
  - [ ] Two figure placeholders are GONE from v4.2-final (real
    fit-vs-meas figure now), so Mario will not see the
    "Figure source requested from Pazos / Lanza" boxes that
    were in v3.

---

*Draft v2 written 2026-05-05 from O26 oracle review + brief
v4.2-final. Replaces v1 (2026-05-03, now stale). To be reviewed
by user before sending.*
