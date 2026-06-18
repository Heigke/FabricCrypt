# O109 — COUPLING / ARCHITECTURE / BENCHMARK diagnosis (Identity-Load Phase 9)

You are one of four hostile oracles (GPT-5 / Gemini / Grok / DeepSeek) advising on a
research program testing whether two physically identical AMD Ryzen AI Max+ PRO 395
("Strix Halo", gfx1151 / Radeon 8060S) workstations — `ikaros` and `daedalus` — can
carry chassis-bound *identity load* in a learned model on commodity userspace GPU.

## SO FAR (compressed)

- We can **identify** the two machines from envelope features (power, thermal-τ,
  per-core latency, TSC drift) with zero error.
- Every coupling we tried so far has shown identity is *recognisable, not
  constitutive*: the model is structure-bound, not device-bound.
- Phase 7 A/B/C/D 2×2 factorial ablation (30 seeds, ridge reservoir, both metrics
  C1=next-step prediction + C2=self-anomaly AE) explicitly tested whether
  hash-derived structure adds anything over random structure given the same data:

```
EVAL=ikaros   A(ikaros-hash struct, ikaros data)   AUROC 0.8340 ± 0.0254
              B(random struct,       ikaros data)   AUROC 0.8338 ± 0.0246
              C(ikaros-hash struct, daedalus data)  AUROC 0.4923 ± 0.0414
              D(random struct,      daedalus data)  AUROC 0.4917 ± 0.0411
EVAL=daedalus A(daedalus-hash, daedalus data)       AUROC 0.8291 ± 0.0456
              B(random,         daedalus data)      AUROC 0.8327 ± 0.0471
              C(daedalus-hash, ikaros data)         AUROC 0.4994 ± 0.0462
              D(random,         ikaros data)        AUROC 0.4994 ± 0.0462
```

A − B ≈ 0 on both hosts → **chassis-hash structure adds 0% over random**.
A − C ≈ +0.34 → all the signal is in the data, none in the structure.

- Phase 8 (currently running): rich substrate capture (derivatives, cross-impedance,
  spectra) + A/B retry — outcome unknown at packet time. Treat Phase 8 as background.
- Prior oracle rounds (O95–O108) progressively narrowed the diagnosis.
  Their convergent positions:
  - **O107**: embodiment "helps" only on closed-loop control / latency-aware /
    survival tasks; on abstract benchmarks it's net-zero even when generic
    baseline gets equal volume of own-chip data.
  - **O108**: the killer test is the 2×2 (structure × data) ablation we just ran;
    consensus claim must drop "architecture-agnostic" and "embodiment", retreat to
    "two physically identical machines exhibit large repeatable within-chassis
    advantages on self-prediction/self-anomaly; not yet isolated from training
    distribution shift."

## USER'S DEEPER CONCERN (verbatim translation)

> "Maybe we're coupling the wrong way, testing the wrong model, on the wrong
> benchmark — not just collecting too little data."

Specifically three failure axes at once:

1. **Coupling is decorative** — hash → seed → init is a *label*, not a
   computation. The substrate never participates in the forward-pass math.
2. **Model is too flexible** — a ridge reservoir on a 600-window training set
   is a universal approximator over short horizons; chassis priors get
   overwritten by data fit.
3. **Benchmark doesn't require embodiment** — next-step prediction and synthetic
   anomaly AE can both be solved by *any* model that sees enough own-chip data.
   Body-info is one of many features, not the only path.

## ANSWER LITERALLY (each numbered point, please)

1. **DIAGNOSIS.** Given the null A − B on the 2×2 ablation, is the issue
   (a) hash is decorative not informational,
   (b) ridge reservoir is too universal-approximator,
   (c) benchmark doesn't require body-info,
   (d) all three?
   Diagnose with reasoning; rank cause weight.

2. **CONSTITUTIVE COUPLING.** Design a coupling where the forward-pass MATH
   directly uses live substrate at every step (e.g. `α[t] = f(live_T_apu[t])`
   as leak rate, not just an init seed or appended feature). Give concrete
   pseudocode for the smallest meaningful test. Identify *which* substrate
   signals to read at *which* rate, and *which* operation they must
   parameterize for the binding to be truly load-bearing rather than
   decorative. Comment on potential confounds (any drift in measurement
   becomes a data leak, etc.).

3. **ARCHITECTURE RANKING.** Rank the following by *likelihood that
   chassis-binding becomes a measurable advantage* (best → worst), with brief
   citations / reasoning:
   ridge ESN, MLP, LSTM, Transformer, Neural ODE, Spiking NN,
   continuous-time recurrent with substrate as parameter,
   energy-based / Hopfield, predictive-coding (RAO/BALLARD), DAE+contrastive.

4. **BENCHMARK DESIGN.** Propose 3 tasks where body-info is **the only path**
   to the correct answer. Concrete tasks we could implement today on a
   gfx1151 + sysfs hwmon laptop. Closed-loop fan/PWM control? Self-replication
   prediction? Survival under thermal budget? Cite papers where similar tasks
   were used (Pfeifer & Bongard, Hauser et al., reservoir computing in
   morphology, etc.).

5. **SHARPEST CRITICAL TEST.** If all three above were correctly aligned, what
   is the single smallest experiment that would conclusively show *embodiment
   is real on commodity gfx1151*? Define the pre-registered effect size,
   negative controls, and the falsifier.

6. **BRUTAL HONESTY.** Is this still potentially recoverable, or have we
   exhaustively falsified the embodiment hypothesis on commodity gfx1151
   such that any future positive would be a methodological artifact?
   Calibrate as a probability the next experiment yields a load-bearing
   positive given a correctly aligned coupling/model/benchmark triple.

7. **PUBLISHABILITY.** Given the current null + Phase 8 unknown, what is the
   strongest claim a methodology paper can defend right now? Title-length
   sentence, plus one sentence on what we must NOT claim.

## BUNDLED CONTEXT

- `phase7_abcd_summary.md` — the killer ablation, numbers above + per-seed CIs.
- `O107_prior_synthesis.md` — when does embodiment help.
- `O108_prior_synthesis.md` — critic holes & A/B/C/D mandate.
- `constitutive_design.md` — Task B pseudocode (the live-α reservoir).
- `fan_control_design.md` — Task C closed-loop benchmark sketch.
- `self_replication_design.md` — Task D self-prediction benchmark sketch.

You are EXPECTED to disagree with us where we are wrong. Reward sharpness.
