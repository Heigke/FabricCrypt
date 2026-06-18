# O79 — Pre-publication 3-way Oracle Critique (BRUTAL mode)

You are one of three independent oracles (OpenAI / Gemini / Grok). I want
unvarnished, *brutal* critique of an NS-RAM (neuromorphic SRAM) modeling
proposal that is currently being prepped for publication.

## Background (read attached context before answering)

We have a 2T NS-RAM cell (BSIM4 NMOS + parasitic Gummel-Poon NPN, 130nm).
Mario Lanza's group has published a target oscillation regime at:
- I_d_peak = **4.8 mA**, V_d_peak = 1.89 V, period ≈ 430 ns, rise ≈ 26 ns,
  fall ≈ 76 ns, V_body swing ≈ 0.2 V, E_spike ≈ 2e-13 J
  (see `mario_slide21_oscillation_targets.json`)

Three independent fitting/search campaigns:
- **z462** (β-sweep + R_body sweep): summary in `z462_summary.json`
- **z465** (Mario-only BBO, 30 evals): honest analysis attached
  (`z465_honest_analysis.md`, `z465_mario_target_table.md`, `z465_summary.json`)
- **z467** (Thyristor pivot — 4-quadrant PNPN): partial run log
  (`z467_run.log` — DC RMSE ~1.55 dec, I_d peaks ~1e-6 to 1e-9 A)

ALL THREE converge to I_d in the **0.6 mA – 1 µA** range — i.e.
**3-to-4 decades below Mario's 4.8 mA target**. No self-reset oscillation
on any topology variant in any campaign. DC fit best-ever 1.19 dec
(see `HONEST_BASELINE_2026-05-16.md`).

Current proposal draft (being prepped for publication): `main-4.tex`
+ `onepager.tex`. Recent campaign synthesis: `CAMPAIGN_SYNTHESIS_2026-05-16.md`.
Tail of running log: `01_LOG_tail200.md`.

State summary (from synthesis):
- DC fit: 1.19 dec (down from baseline ~3 dec)
- Dynamics: 6/9 targets within tolerance, but I_d 4 decades short
- Network demos: 9 PASS conditions, 4 labelled "AMBITIOUS"
- No self-reset observed on any variant in z462, z465, z467

## Three questions — answer ALL THREE, do not hedge

### Q1 — STRUCTURAL DIAGNOSIS
Three independent campaigns (z462 β-sweep, z466 7D BBO with knee-gates —
script `scripts/z466_mario_bbo_7d.py` exists but produced no usable
output, treat as the same parameter family as z465 —, z467 thyristor
pivot) ALL converge to I_d in 0.6 mA – 1 µA range, **4 decades below**
Mario's 4.8 mA. NO self-reset on any. Given this pattern, what is the
SINGLE most likely structural cause? Choose one and defend it:

  (a) Surrogate LUT clips I_d at ceiling
  (b) Topology has hidden current divider (R_series) we haven't seen
  (c) Mario's 4.8 mA is from a different test structure (e.g. M2-bypass
      or single-transistor stress, NOT the 2T NS-RAM cell)
  (d) We're modeling the wrong device class (e.g. PNP-dominant, not NPN)
  (e) Other (specify and cite)

Cite specific reasoning from the attached docs. **No hedging.**

### Q2 — PUBLICATION READINESS
Given the current state (DC 1.19 dec, 6/9 dynamics, 9 PASS networks of
which 4 AMBITIOUS, I_d 4-dec gap), is the proposal as written in
`main-4.tex` **publication-ready**?

If NO, what is the SINGLE biggest fix needed before submission?
**Quote specific sentences from `main-4.tex`** that you find weak,
overclaimed, or unsupported. Be specific (line/section).

### Q3 — ALTERNATIVE FRAMING
If the I_d 4-decade gap is **structural and unfixable** in the current
2T topology, can we still defend the proposal by re-framing it as:

> "We model NS-RAM's neuromorphic function envelope (regime selection,
>  dynamic ranges, network demonstrations), not its absolute current
>  scale — the latter is calibrated at the silicon validation phase."

Is this defensible scientific framing, or is it **cheating** (i.e.
moving the goalposts post-hoc to hide a falsification)?

Bonus: **what would Mario Lanza himself think** if he read the
proposal with this framing?

## Output format
- Q1: one letter (a-e) + 2-3 paragraph defense with citations
- Q2: YES/NO + biggest fix + quoted weak sentences from main-4.tex
- Q3: DEFENSIBLE / CHEATING / NUANCED + reasoning + Mario verdict

If you genuinely lack evidence to answer, say so explicitly. Do NOT
invent citations.
