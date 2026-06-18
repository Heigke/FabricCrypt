# H7 fusion architecture — UNIQUE × FRESH × RÄKNA bound into a frozen LLM (2026-06-15)

## The design principle: ONE challenge-response primitive, not three bolt-ons
Don't AND three separate signals (an attacker defeats each in isolation). Compose them into a single
**physical challenge–response (PUF-CRP)** the model consumes, so breaking any one corrupts the model's input.

- **Challenge C** = f(FRESH nonce, UNIQUE die-key, model hidden-state hash). Defines a drive schedule:
  which CPU cores burst (v), which GPU pattern (u), tone phases, ordering.
- **Response R** = the die's measured reaction to C: the u·v shared-PDN contention signature (RÄKNA) PLUS the
  per-core Fmax/voltage-margin transfer (UNIQUE, on-die, reproducible).
- The frozen LLM is trained (adapter-only) to NEED R for its next-token distribution.

Why this is "deep / secure / hard":
- **FRESH**: C changes every window (RDSEED + rolling hidden-state hash) → a recorded trace (replay) answers the
  wrong challenge → model breaks. Non-replayable by construction.
- **RÄKNA**: R contains the live physical product u·v (verified: linear readout does XOR=0.75, null p=0.000). The
  model needs a quantity that only a real-time physical multiplication on the running machine produces.
- **UNIQUE**: C is parameterized BY the die-key (per-core Fmax/CPPC layout) AND R includes the per-core margin
  transfer. A different die answers C differently in the channel that genuinely carries die identity.
- Entanglement: identity rides the channel that actually carries it (Fmax/CPPC), not the u·v magnitude (which
  the red-team showed is board/firmware-set at kHz). u·v + nonce carry computation + freshness. The model needs
  all three LIVE and CONSISTENT.

## Two tiers of binding (soft + hard) — this is what makes it "djupt, säkert, hårt"
**Tier 1 — constitutive-by-TRAINING (graceful, measurable, but distillable):**
u·v + die-key + nonce → gated cross-attention (v4a). Adapter trained so PPL breaks on replay/wrong-die/zero
(margins 0.5/3.0 nats). Gives a smooth, measurable dependency and good text quality. Weakness: a determined
attacker could distill a copy that ignores the die. So we add Tier 2.

**Tier 2 — constitutive-by-CRYPTOGRAPHY (hard, non-distillable) [oracle O108 unanimous best idea]:**
The live die challenge-response → **fuzzy extractor (Dodis et al. EUROCRYPT'04)** → a STABLE key → **decrypts the
quality-critical adapter weights** at load time. Wrong die / replay / stale nonce → wrong key → adapter decrypts
to garbage → model collapses to base/broken. Cannot be distilled without the physical die producing the right
response to a fresh challenge.

**Why the fuzzy extractor dissolves the reproducibility war:** we do NOT need raw-feature cosine ≥0.9. We need the
intra-die measurement noise to fall INSIDE the ECC error-correction radius while inter-die falls OUTSIDE. Enroll
once → store helper data (public, leaks nothing) → any in-radius reading reconstructs the EXACT key. This converts
"make the analog feature reproducible" into "make BER < ECC capacity," a far easier, standard PUF problem.
Reproducibility recipe (DVFS-pin + temp-lock + fixed baseline + N-epoch coherent averaging) is now in service of
LOWERING BER, not chasing cosine.

## Honest load-bearing caveat (anti-bias)
The "different die breaks it" security rests on the **per-core Fmax/voltage-margin channel** (die-specific,
reproducible, on-die), NOT on u·v being die-unique — PDN u·v at kHz is dominated by VRM/package/board, so its
die-specific component is weak (verified: 3 failed routes; red-team physics: die PDN signature is at MHz-GHz).
So the claim is **"die-COUPLED embodiment"** (requires this die's identity + a genuine physical computation +
fresh entropy), NOT "die-CONSTITUTIVE via a single irreducible die-specific computation." The latter needs
≥8-10 dies and/or MHz-GHz on-die impedance sensing (ChipletQuake-style RO/TDC fabric we don't have on gfx1151).

## Concrete build (on top of h7_rooted_lm_v4a.py — already has the scaffolding)
Existing v4a: frozen SmolLM2-135M; gated cross-attn at layers 20/24/28; K=8 substrate tokens via Perceiver
resampler from live SubstrateStateV3 10-ch window; pre-registered kill suite (PPL_zero>1.25×base; margin
native-vs-zero ≥0.5 nats; native-vs-spoof/wrong ≥3.0 nats; gates must activate); replay/spoof/wrong eval
conditions already coded. This is the integration point — minimal changes:

1. **Add the RÄKNA channel**: an 11th substrate channel = live u·v contention (GPU u-burst × CPU v-burst,
   shared-PDN droop excess). Drive scheduler runs the burst pattern; SubstrateStateV3 reads the product. The
   model's cross-attention now has the genuine physical computation in its input.
2. **Add the UNIQUE die-key**: per-core Fmax/CPPC vector + dynamics fingerprint → (a) a conditioning bias added
   to the resampler queries, and (b) the seed that picks WHICH cores the v-bursts use (challenge parameterized
   by die layout).
3. **Add FRESH**: RDSEED nonce (+ rolling hash of the LM hidden state) seeds the per-window challenge schedule.
4. **Kill/falsification conditions** (extend v4a's suite, pre-registered):
   - **replay**: feed a recorded R from an earlier session → must raise CE ≥ MW nats (model breaks).
   - **wrong-die**: feed daedalus's die-key + R → must break (this is the räkna-unikt-sensitive test).
   - **frozen-challenge**: hold nonce fixed → degenerate, must underperform live fresh challenge.
   - **zero/shuffle**: standard ablations already in v4a.
   - Text quality (PPL on held-out) must stay within 1.25× of base under the LIVE-correct condition.

## Decisive prerequisite experiment (run BEFORE wiring wrong-die security)
**Per-core Fmax/CPPC swap-in test** (red-team's kill-or-confirm, on the RIGHT channel):
- Enroll per-core highest-perf/CPPC + (if reachable) voltage-margin curve on ikaros and daedalus, K repeats each.
- Test: model needs the curve as a nonlinear activation; (1) run on die A, (2) swap in die B's curve, (3) swap
  shuffled curve. If die-B swap degrades AND shuffle degrades → first real räkna-unikt signal on a reproducible
  on-die channel → proceed to full security wiring. If die-B survives → räkna-unikt dead at this scale; ship
  die-COUPLED architecture with the honest smaller claim + negative-result writeup (PDN band-mismatch physics).

## Parallel tracks status
- O108 oracle (reproducibility/supersample/architecture) — dispatched, pending.
- Reproducible lock-in rebuild (timestamped random-ETS + fixed-f lock-in above 1/f + pinned DVFS + multisine
  pole features + enroll-at-fixed-temp + majority vote) — designed (web agent Rank 1-5), build if Fmax route or
  oracle says the identity/spatial channel is worth hardening.
- LLM fusion build on v4a — architecture above; start after per-core Fmax enrollment confirms the die-key channel.
