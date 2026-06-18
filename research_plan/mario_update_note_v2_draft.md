# Mario update note v2 — Path-A complete, cross-task limit identified (DRAFT)

**Status**: Draft for user review. Replaces the stale 2026-05-07 v1
draft (which was written BEFORE z218-z233 and announced an
in-progress fix that has since been completed AND tested).

**Why a v2**: O35 3-oracle consensus (2026-05-09): the v1 draft
asserts a now-falsified narrative ("lumped vs q2d is real physics
divergence") and announces a fix that is now finished. Gemini was
explicit: "DO NOT SEND" the v1 draft.

**Send target**: Mario Lanza (KAUST). Cc: Sebastian Pazos.

**Suggested subject**:
> NS-RAM modeling — body-state surrogate completed; NARMA-10 hits ESN
> class; energy headline confirmed; honest attribution finding on
> cross-task work (it's a hardware-efficiency story, not a reservoir-
> quality story)

---

## Email body

> Mario,
>
> Closing the loop on the body-state-modelling work I flagged on
> May 7. The 4D transient surrogate is built and tested end-to-end.
> Three results are headline-quality; one negative is honest news
> that shapes how I'd frame things going forward.
>
> **Body-state surrogate works** (with an honest ESN comparison).
> We added explicit V_b dynamics (V_b[t+1] = V_b[t] + dt·(I_ii − I_leak)/C_b)
> over a 4D table of (V_G1, V_G2, V_d, V_b). On NARMA-10, frozen-config
> 30-seed CI is NRMSE = 0.612 ± 0.030 (95% CI [0.601, 0.624]) — a 27%
> beat over the brief's instantaneous-surrogate baseline of 0.84.
>
> *Honest comparison (z243)*: a textbook tanh ESN at the same N=200,
> same readout, reaches NRMSE = 0.563 ± 0.038 (95% CI [0.550, 0.577])
> — about 8% better than NS-RAM, CIs disjoint. So our headline is
> "NS-RAM achieves **ESN-class** NARMA-10 accuracy at the silicon-
> energy floor" — not "NS-RAM beats ESN." Software ESNs on a CPU
> remain the more accurate reservoir; NS-RAM's value here is that
> it gets close at sub-µJ silicon energy.
>
> **Three-source physics triangulation closed**. Surrogate vs pyport
> agree to 0.39 dec (max) at reservoir biases; pyport vs ngspice
> agree to 0.51 dec (one tail miss at V_G2 = 0). The transitive
> bound surrogate-vs-ngspice is ≤ 0.90 dec worst case. The reservoir
> operates in the physical subthreshold-leakage regime that ngspice
> directly confirms (I_d ~ 1e-11 A at production BJT params).
>
> **Energy headline holds and is now sharper**. At 1024-step inference
> on N = 64 cells, NS-RAM costs ~0.7 µJ end-to-end. Comparable
> commercial AI MCUs sit at 5 µJ (MAX78000), 10 µJ (Coral Edge TPU),
> or 50–100 µJ (Cortex-M4). The 10× advantage vs purpose-built AI
> silicon is the line I would lead with publicly.
>
> **Cross-task — and a hard-earned attribution finding (z242)**.
> We ran a clean ESN control with everything else identical
> (same N=1000, same input projection W_in, same baseline definition,
> same linear classifier) and found that a textbook tanh ESN
> *reaches +27 pp over the projection baseline on MNIST* (CI [+26, +28],
> 8/8 seeds positive, p=8e-11), versus NS-RAM-as-reservoir at +5 pp.
> So the within-band relationship we built up across MNIST / KMNIST /
> FashionMNIST(×2) below is real — but it is NOT NS-RAM-specific. A
> standard ESN does the same job, only better, on the same pipeline.
>
> The honest reading: **NS-RAM's value is the silicon-energy floor,
> not bench-grade reservoir quality.** We should pitch this as a
> hardware-efficiency story rather than as a reservoir-computing
> demonstration. With that framing, the result below is still a
> well-characterised internal calibration (it tells us where on the
> task-difficulty curve our cell adds *any* value) but should not be
> presented as a competitive reservoir-computing claim.
>
> The within-pipeline experiments below are kept for completeness:
>
> *z233 — frozen NARMA-10 hyperparameters on seq-MNIST, 27 seeds*:
> reservoir 37.4% vs pure-projection 42.0%. Δ = −4.7 pp,
> 95% CI [−5.5, −4.0], p = 8e-17. Frozen config FAILS.
>
> *z235 — same architecture on seq-MNIST, ONE hyperparameter
> retuned (g_VG2 input gain: 0.05 → 0.20), 25 seeds*: reservoir
> 48.0% vs projection 42.9%. Δ = +5.1 pp, 95% CI [+4.5, +6.0],
> p = 9e-18, 25/25 seeds positive. Recovers on MNIST.
>
> *z236 — SAME retuned config now on FashionMNIST, 10 seeds*:
> reservoir 60.9% vs projection 71.5%. Δ = **−10.6 pp**, 95% CI
> [−11.25, −9.50], p = 1.4e-11, 0/10 seeds positive. Retune FAILS.
>
> *z237 — SAME retuned config now on KMNIST (Kuzushiji-MNIST,
> Japanese cursive), 8 seeds*: reservoir 51.8% vs projection 49.2%.
> Δ = +2.6 pp, 95% CI [+1.5, +4.5], 8/8 positive, p = 0.001.
> Reservoir helps again, but smaller margin than on MNIST.
>
> The pattern across four image-classification tasks is quantitatively
> predictable from linear-baseline strength alone:
>
>   MNIST                     proj 43%  →  Δ = +5.1 pp
>   KMNIST                    proj 49%  →  Δ = +2.6 pp
>   FashionMNIST (train=200)  proj 68%  →  Δ = −8.6 pp
>   FashionMNIST              proj 72%  →  Δ = −10.6 pp
>
> Within this MNIST-family band a linear fit gives Δ ≈ +29.8 − 0.56·proj%
> (zero-crossing ≈ 53% baseline). The FashionMNIST(train=200) datapoint
> is a controlled out-of-sample test inside that band: predicted
> Δ = −8.2 pp, actual −8.6 pp, within bootstrap CI [−10.0, −6.5].
> Inside MNIST-family, the relationship is predictive to ≈±1 pp at
> the seed counts we've used (n=8–25 per task).
>
> *Robustness to hyperparameter choice (z241)*: a sensitivity sweep of
> the input-gain parameter g_VG2 across {0.05, 0.10, 0.15, 0.20, 0.30}
> on MNIST shows a smooth approximately linear gradient
> (Δ ≈ −7.4 + 56·g_VG2), with no peak at our chosen value 0.20.
> This rules out a "winner's curse" — the cross-task pattern is a
> property of the input-coupling mechanism, not a special
> hyperparameter setting.
>
> *Honest scope-bound (z240, CIFAR-10 grayscale 28×28, 8 seeds)*: at a
> much harder dataset where the linear projection baseline collapses to
> 15.3%, the linear extrapolation breaks. The fit predicts Δ = +21.1 pp,
> the measurement gives +1.94 pp (CI [+1.0, +2.75], 7/8 positive,
> p=0.001). The SIGN matches (reservoir helps where baseline is weak)
> but the magnitude saturates ~10× below the linear extrapolation. So
> the quantitative law is bounded to the MNIST-family operating band;
> outside it, the direction holds but magnitude is task-specific.
>
> *Open questions / planned follow-ups*: (1) attribution — we have not
> yet isolated whether the linear-within-band relationship is
> NS-RAM-specific or a property of any reservoir within this fixed
> readout pipeline; an ESN control on one task is queued. (2)
> functional form outside band — one CIFAR datapoint does not
> constrain the saturation curve; 2–3 more out-of-band points
> (e.g. SVHN, EMNIST) would let us fit it.
>
> **What this means**, with no overclaim either way: the g_VG2
> retune is task-specific, not a general principle. The pattern
> across the three tasks suggests a simpler interpretation —
> reservoir helps where the linear projection baseline is weak
> (MNIST projection 43% → reservoir +5pp), and hurts where the
> linear baseline is already strong (FashionMNIST projection 72%
> → reservoir −11pp). The cross-task gap is real and
> task-conditional; it is not closed by single-knob retuning.
>
> **What I'd recommend for the brief and any external comms (revised
> after z242 ESN attribution)**:
> (1) **Lead** with energy: ~0.7 µJ per 1024-step inference at N=64 vs
>     5–100 µJ for commercial AI MCUs — the silicon-energy story is
>     where NS-RAM stands alone.
> (2) **Lead** with NARMA-10: NRMSE 0.612 ± 0.030 (30-seed CI), beating
>     our prior baseline by 27%. A textbook ESN at the same N reaches
>     0.563 (8% better, z243), so the framing is "NS-RAM reaches
>     ESN-class accuracy at the silicon-energy floor" — close but
>     not better than software ESN, however delivered at ~10×
>     less per-inference energy on silicon.
> (3) **Lead** with R-track triangulation as physics credibility.
> (4) **Do NOT** lead with cross-task image-classification results.
>     A textbook ESN beats us by 22 pp at the same pipeline. Mention
>     the cross-task work as an internal calibration that bounded
>     where our reservoir adds any value at all (and the answer is:
>     mostly on weak-baseline tasks where ANY reservoir helps), not
>     as a reservoir-computing claim.
> (5) Drop the older "complement to weak linear baselines" framing —
> low-dim/structured signal-processing tasks (NARMA family) and
> on simple-image tasks where the trivial baseline is poor, but
> it competes poorly against a task with already-discriminative
> linear features; (3) frame as "promising for the specific
> regime where temporal integration of a weak input matters,"
> NOT as "general edge-AI reservoir" and NOT as "single-knob
> retune solves cross-task generalization."
>
> **Chip-design implication = none**. The cell-design recommendation
> from the brief (ER_SPARSE topology, lateral-parasitic calibration)
> doesn't move with these results — it sits on DC silicon fits, not
> on reservoir behaviour. The Sebas characterisation request packet
> (I_c/I_b + pulsed-V_d for τ extraction) is still the right next
> measurement, since it would let us refine C_b and τ_body — but
> nothing in the current chip plan needs to change before then.
>
> Happy to share the figures (Path-A journey panel, energy table,
> z233 acceptance plot) if useful.
>
> Best,
> Eric

---

## What is intentionally in this v2 (not in v1)

- **Path-A completion claim**: NRMSE 0.612 ± 0.030 with proper 30-seed
  CI, not the v1 "we'll fix it in 72 h" framing.
- **R-track triangulation**: explicit numbers (0.39 / 0.51 dec)
  replacing the v1 "1.39 dec at Bf=100 + η ≤ 1" (which was the wrong
  branch — see z232 correction).
- **Honest cross-task negative**: the z233 result (p = 8e-17, CI
  excludes 0) IS the news. Per all 3 oracles it must be in the note.
- **Energy table** as headline, with concrete competitor numbers.

## What is intentionally NOT in this v2

- **No q2d / lumped-divergence claim**. Per z232, lumped's "low-Id"
  values were non-converged Newton iterates, not a valid second
  branch. Don't put that in front of Mario.
- **No "no drama" hedging**. The negative is news; the positives are
  bigger news. State both, don't apologise for the negative.
- **No retraction of brief's energy or NARMA framings** — both are
  now stronger, not weaker.

## Pre-send checklist

- [ ] User reviews tone (matches Sebas/Mario relationship style)
- [ ] User decides: send before or after the Sebas characterisation
      request packet? (independent now; both can go same day)
- [ ] Optionally attach: figures/path_a_journey/path_a_journey.pdf,
      research_plan/chip_mod_cost_calibration_v1.md energy table,
      results/z233_seq_mnist28/summary.json
- [ ] Optionally attach: research_plan/oracle_queries/
      O35_12h_feedback_20260509_00/ for full oracle audit trail

---

*v2 drafted 2026-05-09 after z230/z231/z232/z233 results landed.
v1 (research_plan/mario_update_note_draft.md) is now superseded —
do not send.*
