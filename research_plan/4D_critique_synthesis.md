# 4D Critique Synthesis — O46 (gpt-5, gemini-2.5-pro, grok-4)

Date: 2026-05-12
Packet: `research_plan/oracle_queries/O46_4D_critique/`
Inputs: 4A use-case synthesis, today's 01_LOG tail, z293/z296/z298b/z299b/z300 summaries.

---

## Per-Question Consensus + Dissent

### Q1 — Falsification

**Consensus (3/3):** The strongest falsification is the direct contradiction
between strategic positioning and empirical results. The 4A oracle
synthesis names always-on KWS + industrial anomaly as the top-2/top-3
target apps, yet the project's own benchmarks fail those tasks
(KWS 8.3% chance; NAB ~17 vs gate 30+). A simultaneous credibility hit
comes from the surrogate physical model: 1.67 dec systematic
subthreshold bias vs Sebas silicon (z298b) and 2-6 dec absolute / 0.92
dec shape-only gap vs original Mario+Sebas TCAD (z299b). Together: the
device fails its advertised function AND its golden model is off by
orders of magnitude.

**Dissent / additional angles:**
- gpt-5 adds noise sensitivity: HDC drops to 59/55% at σ=0.05/0.10 — headline relies on low-noise operating assumption fragile in real silicon.
- gpt-5 also flags evidence access gaps (no original Sentaurus outputs, Sebas data limited to 0.17 V/s, Vd ≤2 V).
- grok adds competitive framing: NS-RAM not yet benchmarked vs Syntiant NDP / Cortex-M4 in the implied markets.
- gemini frames the model gap as invalidating the "integration" part specifically (SoC SPICE deliverables).

### Q2 — Headline integrity (HDC 80.23% / 2.3 nJ)

**Consensus (3/3):** Defensible *as an envelope / proof-of-concept*
characterization, NOT as application-level market readiness.

**Consensus mandatory caveats:**
1. UCI-HAR is a generic academic benchmark, NOT one of the named target apps.
2. Energy = neuron-core surrogate estimate; excludes feature extraction, classifier, memory/IO, sensor front-end.
3. Noise sensitivity: σ=0 only; σ=0.05/0.10 drops to 59/55% (4B2 FAIL).
4. Model-hardware gap: 1.67-6 dec discrepancy vs silicon/TCAD ground truth.
5. KWS + NAB FAILs must be reported alongside, framing HDC as "platform capability," not "app performance."

### Q3 — Surprise / under-valued lead

**Consensus (3/3): The Bayesian MCMC RNG (z296) is the under-valued
finding and should be elevated to (co-)lead.** All three oracles
independently call it paradigm-shifting: physical noise as a
computational resource (MCMC-grade entropy), ESS 1.03× vs pseudo-RNG
across 10K MH steps. gpt-5 and gemini explicitly recommend a **dual
headline** (HDC + Bayesian RNG); grok pushes Bayesian RNG as the
**primary** lead.

**Caveats for Bayesian RNG elevation (gpt-5):**
- Single model/task; no NIST SP800-22/90B TRNG battery.
- Software overhead (0.38 s RNG vs 0.045 s MH on GPU) is plumbing, not physics.
- Hardening needed: longer chains, multiple targets, K-S on posteriors, cross-device seeds.

### Q4 — Cut/keep matrix

| Negative result | gpt-5 | gemini | grok | Decision |
|---|---|---|---|---|
| KWS Speech Commands chance (z297/b) | KEEP | KEEP (reframe) | KEEP | **KEEP** |
| NAB anomaly score ~17 (z295/b) | KEEP | KEEP (reframe) | KEEP | **KEEP** |
| Sebas pyport 1.67 dec subthreshold (z298b) | KEEP | KEEP (reframe → SURR-V4 ask) | KEEP | **KEEP** |
| TCAD 2-6 dec / 0.92 dec shape (z299b) | KEEP | KEEP | KEEP | **KEEP** |
| Snapback 4 terms ruled out (z300) | KEEP | KEEP (reframe as progress) | KEEP | **KEEP** |
| HDC noise sensitivity 4B2 σ=0.05/0.10 | KEEP | (implicit) | KEEP | **KEEP** |
| Per-curve z298 RMSE tables / clamp details | SUMMARIZE | — | CUT (subsume) | **CUT to one representative figure + median** |
| 4B3 Vd-grid interior minutiae | SUMMARIZE | — | — | **SUMMARIZE** |
| --out_dir bug / log reconstruction | CUT | — | — | **CUT** |
| Oracle slide extraction uncertainty detail | SUMMARIZE (±0.3 dec note) | — | — | **SUMMARIZE** |

**Reframing recipe (gemini):** Report negatives as *diagnoses* and
*boundaries*, not failures. E.g., snapback ruleout = "narrowed search
space"; pyport gap = "SURR-V4 requirement crystallized."

### Q5 — Ship-or-gate verdict

**Split 2-1 with nuance:**
- **gpt-5: SHIP** with dual headline (HDC + Bayesian RNG), full caveats, and a crisp "ask" to Mario for TCAD output dumps + additional transient sweeps.
- **gemini: GATE** on demonstrating a non-chance-level KWS result before any other work. Chasm between "IP for always-on KWS" claim and "KWS at chance" reality is too large.
- **grok: GATE** on closing the snapback gap with heavier physics (avalanche M(V_bc), velocity-sat, hot-carrier) since it is foundational to reliable modeling.

The two GATE oracles disagree on *which* finding to gate on
(application-side vs physics-side).

---

## Strongest individual falsification (winner)

**gemini's framing** is the sharpest single shot: the use-case synthesis
*names* KWS and anomaly as the top apps, and the project's own
benchmarks show both at chance / below gate. This is internal
contradiction — the most damaging form of falsification. gpt-5
strengthens it with a quantitative model-validity argument
(1.67-6 dec). The combined hit: *"the device fails its advertised
function and its golden model is off by orders of magnitude."*

## Surprise-lead candidate (consensus)

**Bayesian MCMC RNG (z296)** — unanimous across 3 oracles. Paradigm
claim: physical noise as a computational resource, not a liability.
ESS 1.03× pseudo-RNG, n=10K MH. Recommend **dual headline** (HDC
envelope + Bayesian RNG) per gpt-5/gemini majority over grok's
single-lead recommendation.

## Ship-or-gate verdict (synthesis)

**GATE, on a narrow application-side check, not on snapback physics.**

Rationale:
- 2/3 oracles vote GATE; the SHIP vote (gpt-5) is conditional on a
  strong dual headline + a Mario ask, which is itself a form of
  gating-via-framing.
- The most credibility-damaging finding is the KWS/NAB application
  failure on the *named* target apps, not the snapback physics gap.
- Snapback (grok's gate) is a longer, more uncertain physics project;
  KWS sanity check (gemini's gate) is achievable on the order of
  days-weeks and directly closes the internal-contradiction
  falsification from Q1.
- Recommended gate condition: demonstrate **non-chance KWS** (≥30% on
  12-class Speech Commands, or matched performance on a smaller
  4-keyword subset) using NS-RAM-augmented architecture (not pure SNN
  if SNN is the bottleneck). Once cleared, ship v4.4 with dual
  headline (HDC + Bayesian RNG) and explicit negatives reframed as
  diagnoses.

## Pitfalls flagged

- Headline-dependence on σ=0 noise model (HDC fragile under realistic readout noise).
- Bayesian RNG lead lacks NIST TRNG battery + cross-device validation; do not over-claim.
- Reframing negatives as "diagnoses" must not slide into spin — keep numbers explicit.
- Snapback gap is real and will be re-asked by Mario; pre-empt with the "ask for TCAD dumps" framing.
- Energy 2.3 nJ excludes sensor + classifier — risk of system-level energy claim being read as full-stack.
