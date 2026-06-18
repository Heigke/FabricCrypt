# O95 Synthesis — 4-way oracle critique of Identity Phase 1

Date: 2026-05-30
Oracles: GPT-5, Gemini 2.5 Pro, Grok-4-latest, DeepSeek-Reasoner — **4/4 responded**

## Vote matrix (per question)

| Q | GPT-5 | Gemini | Grok | DeepSeek | Consensus |
|---|---|---|---|---|---|
| **(a) RTN asymmetry: silicon or thermal?** | Thermal (Arrhenius) | Thermal (Arrhenius) | Thermal (Arrhenius) | Thermal (Arrhenius, ~2-3× per decade) | **4/4 THERMAL** |
| **(b) Spatial-corr asymmetry: silicon or thermal?** | Thermal/PDN (shared envelope) | Thermal (fan/gradient) | Thermal (heatsink loading) | Thermal (leakage compression at high T) | **4/4 THERMAL** |
| **(c) Is KL(PERF)=0.11 a valid thermal-drift null?** | No — counters blind to flicker/RTN | No — only proves no throttling | No — coarse aggregate | No — µs-scale RTN invisible to perf counters | **4/4 NO — null is invalid** |
| **(d) Run Phase 2 on process-stat alone?** | Don't — will train ESN to thermostat | Waste of compute | Recategorize | Waste of cycles until thermal-controlled re-run | **4/4 DO NOT PROCEED AS-IS** |
| **(e) Single most damning falsifier?** | Lock f/V, match Tdie ±0.5 °C, ramp one | **Location swap** (chassis swap rooms) | Swap chassis at identical T | Same room, equilibrate to 35 °C, repeat | **4/4 THERMAL-MATCHED REPEAT (location/chassis swap)** |
| **(f) Anything publishable as-is?** | Only as a *negative*/cautionary workshop note | Nothing; would be scientific malpractice | Not publishable | Not publishable; "kill the paper" | **4/4 NOT PUBLISHABLE** |

## Consensus findings (4/4 unanimous)

1. **Both "signals" (RTN-rate asymmetry, spatial-CU-correlation asymmetry) are thermal artifacts, not silicon identity.** Arrhenius activation of RTS trap kinetics is textbook (Kirton & Uren 1989; Simoen & Claeys 2013; Grasser et al.) and trivially explains 0.000 vs 0.115 with a 15 °C ΔT.
2. **The KL(PERF) = 0.11 "null" is invalid.** PERF_SNAPSHOT is a coarse cycle-integrated counter; it is blind to the µs-scale microarchitectural noise the other channels measure. Smallness of KL(PERF) provides NO evidence that thermal drift is controlled.
3. **The required falsifying experiment is a thermal-matched repeat** — either physical location/chassis swap, or DVFS+fan clamp to identical Tdie ±0.5 °C. If signals collapse, identity claim is dead. Until run, no signal can be attributed to silicon.

## Sharpest disagreement

There is **no sharp disagreement on substance** — all four oracles converge to "thermal artifact, do not proceed". The only divergences are tonal:

- **GPT-5** is the most constructive: explicitly allows a "negative-result / cautionary workshop note" on RDNA3.5 PUF infeasibility under idle, and recommends fuzzy-extractor / helper-data corrections (Suh & Devadas 2007; Maes 2013) as the *correct* PUF methodology had we wanted to do it properly.
- **DeepSeek** is the most aggressive ("kill the paper; fix the experiment").
- **Gemini** invokes "scientific malpractice" — strongest moral language.
- **Grok** is the tersest but offers no additional angle.

**My reading**: the lack of disagreement is itself the result. When 4 independent oracles with different priors all flag the *same* confound (15 °C ambient ΔT) with the *same* mechanism (Arrhenius RTN kinetics + heatsink loading) and the *same* remediation (location/temperature swap), this is not adversarial diversity — it is convergent diagnosis. The Phase 1 design has a single dominant confound and we missed it.

## Recommendation — Phase 1b and Phase 2

### Phase 2 as currently specified: **DO NOT PROCEED**. Redesign.

### Phase 1b (mandatory before any Phase 2):

1. **Thermal-matched replication** — physical chassis swap OR move both devices to one room, equilibrate APUs to same temperature (±1 °C). If process-stat KLs drop near zero → confound confirmed, kill silicon-identity framing.
2. **DVFS clamp + fan-PWM lock** on both devices (per Phase 1 protocol that was skipped). Hold core f/V identical.
3. **Multi-regime sweep** (cold / idle / warm) as the original protocol required. Fit RTN Arrhenius slope per device. *Differences in slope* (not differences in rate) would be a genuine silicon-trap signature.
4. **Detector bandwidth calibration** — the ikaros RTN=0.000 is almost certainly aliasing (traps faster than detection band). Without bandwidth calibration the rate metric is undefined.
5. **CU mapping randomization** — current scheduler/affinity confounds CU-indexed signals.

### Reframe Phase 2 if and only if Phase 1b survives:

- Drop "PUF identity" language entirely. Reframe as "thermally-corrected process-statistics fingerprint".
- SW-matched RNG control becomes the headline number, not a footnote.
- ΔVth-distance gradient (extend to ZGX/Mac) is the only path to a meaningful claim — twins alone cannot distinguish silicon variance from environmental coupling.

### Possible publishable artifact (per GPT-5):

A **negative-result cautionary note** in the FEEL appendix: *"Naive RDNA3.5 GPU-noise PUF attempts under idle workloads are dominated by ambient/Tdie confounds; Arrhenius-corrected RTN extraction is required before any identity claim."* This is honest and supports the broader FEEL narrative (substrate is constitutive but extracting identity requires careful environmental control).

## Files

- Prompt: `prompt.md` / `context.md`
- Responses: `gpt5.md`, `gemini.md`, `grok.md`, `deepseek.md`
- Dispatch log: `_dispatch.log`
