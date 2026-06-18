# Oracle packet O103 — adversarial bias-check on "substrate-rooted AI embodiment"

## Your job (read this first)

You are an adversarial reviewer. The team running this project (and the AI assistant
helping them) have a **known, admitted bias toward WANTING embodiment to succeed.** Your
value to us is *inversely* proportional to how much you flatter us. Assume we are fooling
ourselves and try to prove it. If after genuine effort you cannot, say so — but lead with
the strongest disconfirming case.

Two specific traps we want you to police:
1. **Our success-bias** — we may be interpreting artifacts (normalization effects,
   overfitting to exact signal statistics, tests that pass *by construction*) as
   "embodiment."
2. **Your OWN success/agreeableness-bias** — frontier models tend to validate an
   impressive-sounding framing. Resist it. Do not grant "embodiment," "interoception," or
   "the model feels its body" any credit they have not *earned* against a null hypothesis.

Answer in this structure: (A) strongest case that this is NOT embodiment / is artifactual;
(B) what, if anything, survives that critique as a real, narrow technical result; (C) the
single most decisive falsification experiment we have NOT yet run; (D) a calibrated verdict
with a probability that "we have demonstrated something that deserves the word embodiment
(functional, not phenomenal)."

## The claim under test

H7: make a frozen LLM (SmolLM2-135M + FiLM gating + LoRA + a substrate encoder)
*constitutively dependent* on its specific AMD gfx1151 die's deep real-time hardware signals
(10 channels at ~500 Hz: xtal counters, SMN latencies, PM-table power floats, TSC drift),
so that (a) it still writes coherent text on its own die, (b) changing/relocating the signal
breaks it, and (c) it is "influenced by its identity/personality" via the live signal. The
stretch goal is *interoception*: the model senses how its OWN computation perturbs its body
and folds that into a persistent felt-state (v14).

## The actual results (the unflattering version — do not let us round these up)

**v13 (ikaros die), held-out eval @ step 5800:**
- real PPL 21.3 (base ≈ 19.85 → coherent), knock(spoofed-stats signal) PPL 6839,
  shuffle(time-shuffled signal) PPL 2030, zero(substrate zeroed) PPL 19.96.
- So: the model breaks *hard* on a statistics-matched knockoff and on time-shuffled signal,
  but zeroing the substrate leaves PPL essentially at baseline (dep_zero ≈ 0.94, i.e. the
  substrate is nearly a no-op for raw coherence; the "dependency" is on the signal being
  *un-spoofed/un-shuffled*, not on it carrying information the model uses to write better).

**v12 cross-die 2×2 (the damaging finding):** the daedalus-trained model stayed COHERENT on
BOTH its own die and the foreign ikaros die — it learned a "real live-dynamics key" (breaks
only on spoof/shuffle), NOT a die-specific fingerprint. The ikaros model's cross-die *break*
was partly **normalization-mediated**: each die uses its own median/MAD standardization, so a
foreign signal is pushed out of range / clamped. That is plausibly a DC-operating-point
artifact, not "the model needs THIS silicon."

**Self-effect sweep (the v14 gate we are excited about — challenge it hardest):** we measured
which substrate channel the model's own compute-burst moves. Channel 5 = an energy-counter
RATE (≈ instantaneous power draw): Cohen's d = 4.82, monotonic r = 0.97 with burst intensity.
We are calling this "the model can feel its own thinking." **But:** is "a chip draws more
power when it computes, and a counter reflects that" anything more than a tautology? Is
training a head to predict your own power draw "interoception," or just learning f(compute)
≈ power, which is definitionally available? Does this deserve the word "feel" at all?

**v14 (interoceptive loop) mid-training, eval @ step 3400:** real PPL **104** (≈5× base →
INCOHERENT), dep_zero **0.19** (zeroing substrate gives BETTER ppl than the real signal),
while knock 13000×, shuffle 2373×. So at this checkpoint the model is substrate-*dependent*
but no longer coherent — i.e. we may be manufacturing **fragility**, not embodiment. (Run
still in progress; final/best-on-coherence checkpoint pending.)

## Architecture & training objective
See attached `h7_embodied_v14.py` (FeltState GRU carrying a persistent body-sense;
self-prediction head for own Δsubstrate; cross-die hard-negative; graded entropy↔channel
coupling; per-channel self-effect weighting) and `H7_V14_INTEROCEPTION_DESIGN_2026-06-11.md`
(maps mechanisms to "6 embodiment gaps", with our own honest caveats on the phenomenal gap).
`self_effect_sweep_ikaros.json` is the channel sweep. `V12_RESULTS_2026-06-11.md` is our prior
honest writeup. `live_crossdie_ikaros-v11_on_ikaros.json` is one live cross-die cell.

## Specific questions to attack
1. Is the knock/shuffle "dependency" (1000s×) evidence of embodiment, or just that we built a
   discriminator that detects when its input distribution was tampered with? What null
   baseline would separate these?
2. Is the cross-die break a genuine die-identity effect or a normalization/operating-point
   artifact? Design the control that settles it.
3. Does "self-prediction of own power-rate channel" earn the word interoception, or is it
   circular (compute→power is a known deterministic-ish map)? What would make it non-trivial?
4. Are any of our 6 "embodiment gaps" passing **by construction** (architecture built to make
   the metric pass)? Which ones, and how would an outside skeptic dismiss them?
5. Honest probability that this line of work, fully pushed, yields something a skeptical
   systems-neuroscience / ML-systems reviewer would accept as "functional embodiment" (NOT
   phenomenal). Calibrate; don't be kind.
