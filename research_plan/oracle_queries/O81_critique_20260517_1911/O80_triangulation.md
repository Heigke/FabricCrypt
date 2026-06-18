# O80 Triangulation Matrix — NS-RAM brief v4.5

Oracles: GPT-5 (analytic), Gemini-2.5-pro (citations/depth), Grok-4 (terse/honesty)
Latencies: GPT-5 122s, Gemini 30s, Grok 12s

## Top-line verdict: Should we publish v4.5 now?

| Oracle  | Verdict                                                       |
|---------|---------------------------------------------------------------|
| GPT-5   | NOT READY as competitive-architecture brief. Publish only as "device + calibrated models." |
| Gemini  | DO NOT PUBLISH. "Device physics paper, not neuromorphic systems paper." Will be brutally rejected. |
| Grok    | NOT READY. Indefensible to anyone reading past the abstract.  |

**SIGNAL (3/3 agree): Do NOT publish v4.5 as a competitive-architecture brief.**
Either reframe as a device-physics result, or do another experiment first.

---

## Q1 — Honest positioning at 130nm

| Claim                                                                | GPT-5 | Gemini | Grok | Consensus |
|----------------------------------------------------------------------|:-----:|:------:|:----:|:---------:|
| NS-RAM wins NOTHING at 130nm against Loihi 2/Akida on system metrics |   Y   |   Y    |  Y   | 3/3 SIGNAL |
| Yield ≈ thousands of cells, not millions                              |   Y   |   Y    |  Y   | 3/3 SIGNAL |
| Possible win: intrinsic stochasticity / TRNG primitive                |   Y   |   Y    |  Y   | 3/3 SIGNAL |
| Possible win: analog time-constants / LIF dynamics                    |   Y   |   Y    |  Y   | 3/3 SIGNAL |
| 28nm projection only meaningful with a second tape-out                |   Y   |   ~    |  Y   | 2/3 SIGNAL (Gemini implies but doesn't say) |
| Sensor-proximate / legacy-node manufacturability angle                |   Y   |   N    |  N   | 1/3 NOISE |

## Q2 — Elevator pitch

| Pitch axis chosen                              | GPT-5 | Gemini | Grok |
|------------------------------------------------|:-----:|:------:|:----:|
| Silicon-verified LIF + calibrated network sims |   Y   |   Y    |  Y   |
| Stochasticity / TRNG as co-primary             |   Y   |   Y    |  N   |
| Multi-functionality (memory+neuron+RNG in one) |   N   |   Y    |  N   |
| Defer all energy/density claims explicitly     |   Y   |   ~    |  Y   |

**SIGNAL (3/3): Lead with silicon-verified LIF + software-calibrated network results. Defer energy claims.**
**SIGNAL (2/3): Add stochasticity / TRNG as co-equal pillar — not just neuromorphic.**

## Q3 — Killshot experiment (where they DISAGREE — this is informative)

| Oracle  | Proposed killshot                                                        |
|---------|---------------------------------------------------------------------------|
| GPT-5   | Same-node energy head-to-head: 256–1024 NS-RAM array vs 130nm digital LIF macro on Mackey-Glass + UCI-HAR; falsify if not ≥10× lower E at equal accuracy AND NIST 800-22/90B passes |
| Gemini  | 16×16 array, 24h drift + device mismatch char; falsify if drift >20% or mismatch un-calibratable |
| Grok    | 128-cell ring oscillator with same 2T cell; sustained LIF oscillation matching Mackey-Glass model within 2× |

**DISAGREEMENT — but this is constructive.** The three killshots attack different premises:
- GPT-5 attacks the **energy claim** (the most marketed)
- Gemini attacks the **stability/mismatch** assumption (the silently-assumed one)
- Grok attacks the **dynamical completeness** (the currently-broken one — z473)

Recommended: run Grok's killshot FIRST (it's the cheapest and we're already doing z473),
then Gemini's (mismatch char on existing Mario die), then GPT-5's only after second tape-out.

## Q4 — Funding angle

| Angle                                                  | GPT-5 | Gemini | Grok | Consensus |
|---------------------------------------------------------|:-----:|:------:|:----:|:---------:|
| Chips JU / KDT — edge AI components                     |   Y   |   Y    |  Y   | 3/3 SIGNAL |
| Frame as "physics primitive" not "accelerator"          |   Y   |   Y    |  Y   | 3/3 SIGNAL |
| TRNG / stochastic primitive as headline                 |   Y   |   Y    |  Y   | 3/3 SIGNAL |
| Multi-function (memory+neuron+TRNG) in single cell      |   N   |   Y    |  N   | 1/3 NOISE (but compelling) |
| Strategic autonomy / mature-European-node framing       |   Y   |   N    |  N   | 1/3 NOISE |
| EIC Pathfinder / Horizon physics-to-algorithm           |   Y   |   N    |  N   | 1/3 NOISE |

**SIGNAL (3/3): Chips JU "Emerging memory / in-memory computing" track, framed as charge-memory physics primitive, NOT as accelerator. TRNG is a key pillar.**

---

## What 2/3+ of them said loudly (the things to listen to)

1. **Do not publish v4.5 as a systems brief.** (3/3)
2. **130nm wins nothing system-level today.** (3/3)
3. **Reframe around physics primitive + intrinsic stochasticity.** (3/3)
4. **Defer all energy/density claims** until a second tape-out or array measurement. (2/3 explicit, 1/3 implicit)
5. **Chips JU under emerging-memory / in-memory track, NOT neuromorphic accelerator.** (3/3)
6. **TRNG / NIST-certified stochasticity is the most defensible secondary pillar.** (3/3)

## Where they disagree (informative noise)

- **Which killshot to run first** — but each oracle picks a different (legitimate) axis;
  use all three sequentially, cheapest first (Grok's ring osc → Gemini's drift/mismatch → GPT-5's energy bake-off).
- **Multi-functionality framing** — only Gemini pushes this, but it's the most fundable single sentence.
- **Whether to invoke "strategic autonomy / legacy node"** — GPT-5 likes it, Gemini/Grok don't; risk = it sounds like a consolation prize.
