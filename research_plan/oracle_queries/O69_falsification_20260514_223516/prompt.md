# O69 — FALSIFICATION PACKET (hostile critique requested)

You are an adversarial reviewer. Be BRUTAL. No diplomacy. We have a 24h overnight window to either find directions or retract claims.

## Honest state of program (no spin)

- 3 topology fixes failed in succession to reproduce snapback fold:
  - R-43: iii × Rs (current injection through series resistance)
  - R-47: subdiode (body-emitter parasitic diode)
  - R-49: dbd-avalanche (drain-body-diode multiplier)
  All 5 physics gates FAIL on all three.

- Modeled snapback fold at VG1=0.6 = **0.02 dec** vs measured **2.20 dec** — off by factor of **~110×** in magnitude.

- The 0.965 dec cell-wide DC fit is **sub-threshold curve-fitting only** — not BJT physics. It does NOT reproduce the diagnostic high-VDS fold.

- z371 GPU blitz (1000 random parameter sets within R-50 physical bounds): best score 1.048 dec. No parameter point in the physical envelope reproduces the fold.

- UCI-HAR with publication-grade HDC: sklearn linear ridge = **96.2%**, NS-RAM substrate = **76%**. We are **20pp BELOW a one-line linear baseline**.

- The 5 "surviving" application claims (DS-N10/11/14/15/16) are all on TOY benchmarks (synthetic patterns, hand-tuned thresholds).

## Attached evidence
- `context.md` — last 300 lines of 01_LOG.md
- `z372_snapback_demo_summary.json` — current best fold attempt
- `z368b_dbd_avalanche_fixed_run.log` — avalanche topology trial
- `z371_gpu_blitz_summary.json` — 1000-point blitz, no winner

---

## Q1 — FALSIFICATION (physics)

Three different topology fixes (current injection R-43, subdiode R-47, avalanche multiplier R-49) all FAILED to reproduce the measured 2-3 decade snapback fold. The fold is missing by a factor of ~110×. What is the most likely explanation:

  (a) **Missing physics** — we're STILL missing a key mechanism. If so, name exactly which (impact ionization with proper M(VDS,T) model? floating-body kink with explicit Qb dynamics? thermal runaway? GIDL? something else?) and cite the equation.
  (b) **Simulator bug** — pyport has a fundamental architectural defect that prevents BJT triggering regardless of topology (e.g. no inner Newton loop, no Gummel iteration, body node mis-grounded). Specify the bug.
  (c) **Measurement artifact** — Sebas's measured fold is an instrument artifact (compliance trip, oscillation, self-heating runaway in DUT) and isn't intrinsic device physics.
  (d) Something else.

**Give ONE answer**, the top reason, and the SINGLE experiment that would distinguish your hypothesis from the alternatives. No hedging, no "could be a combination."

## Q2 — BENCHMARK realism

UCI-HAR is solved by a one-line sklearn ridge regressor at 96.2%. The NS-RAM substrate cannot compete there — we are 20pp below the digital ceiling. Name **3 SPECIFIC benchmarks** where ALL THREE of these hold:

  (a) **Linear methods provably fail < 80%** (i.e. there is a real digital ceiling we are not behind)
  (b) **Task structure could plausibly benefit** from the NS-RAM primitives: threshold nonlinearity + temporal integration + body-charge memory (multi-τ leak)
  (c) **Dataset is publicly downloadable tonight** (give URL / pip command)

For each: name, URL, why linear fails, why NS-RAM primitives map.

## Q3 — KILL-SHOT (program-level)

What is the **single experiment** that, if NS-RAM fails at it, falsifies the entire "NS-RAM substrate is useful" claim and forces a full retraction? Not "NS-RAM is suboptimal" — what would force us to retract the substrate-as-useful claim entirely? Define the experiment, the metric, the threshold, and the failure criterion.

---

Be hostile. If the right answer is "retract now," say it. Direct quotes preferred over polite reframes.
