# O42 — Deep multimodal review of Mario + Sebas slides

You have access to 21 image attachments from Sebastian Pazos (KAUST, NS-RAM
device physicist) and Mario Lanza (NUS, lab PI), authored between March
and May 2026. We are an external collaboration team (Eric Bergvall +
Robert Luciani) building a PyTorch port of the NS-RAM device model
(`pyport`) and an SNN/network-simulation layer on top.

We have already done an internal review and extracted **what we think
the slides say**. We want you to **find what we missed** — technical
content, numeric values, architectural claims, experimental protocols,
or hidden constraints that our team has not yet acted on.

This is **not a literature review**. Be specific to *these images*.
Reading EMF/vector versions of these plots is impossible for us; you
get only the rasters. We've also done text extraction and have nothing
machine-readable beyond the visuals.

## What we have already extracted (DO NOT REPEAT THIS BACK)

**Parameter fits (slides 01–04 + 07)**: 3-branch dependences indexed
by V_G1 ∈ {0.20, 0.40, 0.60} for NFACTOR_M2, ETAB_M1, BETA0_M1; K1_M1
is a single curve in V_G1. NFACTOR_M2 reaches 12.2 at V_G2=−0.2 V (red
branch); BETA0_M1 has 3 flat-then-rising regimes (red 10.75→14, blue
flat 19, black flat 20). We've extracted approximate values to JSON.

**Dynamic response (slide 05 + 21)**: parasitic capacitances drive
effective time constant. References "SR and firing-time experiments
3 through 7 in experiment list" — we've never been sent that list and
asked Sebas for it. A pdiode card was sent on 1 May (image 21 = new
schematic with p-body diode added).

**I-V family with measurements (slide 06)**: 4-panel, V_G1 ∈ {−0.2,
0.2, 0.4, 0.6} V, V_G2 swept 0→0.5 V step 50 mV.

**Bulk current model (slide 13)**: `I_exp = 10^(d·V_d)` + `I_pwl =
a·V_d^c + b for V_d ≥ −j`, with a,b,c,d as PWL(V_G2), j constant.
"Each parameter is extracted for different V_G2".

**Cell variants**:
 - 2T NS-RAM 31.8 µm² in 130 nm CMOS (slide 16)
 - 1T deep-Nwell 8 µm² in 180 nm CMOS, 100% yield (slide 17)
 - 2T thick-ox spiking neuron 17 µm², firing range >10⁴×, V_G ≈ −2 V
   suppresses leakage (slide 18)
 - Soma input neuron with linear range V_G2 ∈ [2.5, 3.0] V, fed by
   1 V pulses from starved inverters (slide 12)

**Brian2 SNN (slides 19, 20)**: LIF input neurons reach 72%, Poisson
reference 85%, timescale slowed 10³× for convergence. We have
**reproduced 84.65% Poisson** on real MNIST 28×28 (matches Sebas's 85%)
in our pipeline. NS-RAM substitution (static + transient) FAILS by 5–6
pp at our locked thin-ox params. NS-RAM as analog weight memory with
differential pairs partial-passes (37% vs 22% single-ended) but still
47 pp below ideal.

## What we want from you (be exhaustive)

For each image, scan for content we may have missed. Specifically:

**A. Numeric values, axes, annotations, callouts**:
 - Y-axis ticks, axis log-scale labels, legend entries we may have skipped
 - Inset text, equation captions, parameter values printed on the figures
 - Numbered annotations / arrows / "see X" cross-references

**B. Architectural details**:
 - Schematic topology cues (which nodes are tied, biased, floating)
 - Reset/integration mechanisms (slide 10 says "self-reset")
 - Bias-generation paths (slide 12 mentions "mirror bank generating
   the bias voltage")
 - Any operating-window callouts (slide 18 V_G ≈ −2 V; slide 12 linear
   range V_G2 ∈ [2.5, 3.0] V)

**C. Experimental protocols hinted at**:
 - "Experiment list" entries (slide 05 mentions 3–7 by number)
 - Sweep-rate dependences, pulse-train recipes, retention protocols
 - Measurement-equipment hints (parameter analyser, TLP, pulsed-V_d
   ramp rates)

**D. Numeric claims unique to specific slides**:
 - Energy / area / firing-window / retention values we may have missed
 - Performance-vs-baseline comparisons (slide 20 has 85% vs 72%)
 - Citations and external references

**E. Anything contradicting our extraction**:
 - If you read a value we listed wrong above, flag it
 - If a branch in slide 01/03/04 is plausibly different V_G1 than we
   inferred, say so
 - If the bulk-current PWL form has more terms than the equation we
   transcribed, say so

**F. Things we should have asked Sebas/Mario for** but haven't:
 - Concrete data files implied by figures (e.g. raw CSVs behind the
   parameter dependences, transient measurement traces, Brian2 script)
 - Process-node-specific cards we don't have
 - Pulse-mode protocols not yet shared

## Format your answer

```
## Image 01_NFACTOR...
- [MISSED] ...
- [NUMERIC] axis y goes from 0 to 12.5 at 2.5 steps (we said 12)
- [CONTRADICTION-OUR-EXTRACTION] ...

## Image 02_K1...
...
```

Lead with concrete, image-specific findings. Skip an image if you have
nothing new to add. If a finding spans multiple images, say so
explicitly. **Be terse, dense, factual.** 2000 words total maximum.

Where you see something that should be a follow-up email ask to Sebas
or Mario, prefix it with `[ASK-SEBAS]` or `[ASK-MARIO]` so we can
route it.
