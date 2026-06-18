# O44 — NS-RAM Use-Case / Application Audit

You have 21 slide images from Sebastian Pazos (KAUST) and Mario Lanza
(NUS), authored March–May 2026. They describe the **NS-RAM** device:
a CMOS-compatible (130 nm / 180 nm) MOSFET-like cell with a floating
bulk that exhibits sigmoidal I-V, hysteresis, and spiking/firing
dynamics. Cells include 1T (8 µm², 180 nm) and 2T variants (17–32 µm²,
130 nm thick-ox), plus Brian2 SNN simulations on MNIST.

A prior oracle pass (O42) covered **technical content, numeric
parameters, and constraints**. **Do NOT repeat that.** This pass is
about **commercial / application intent**.

We need you to read the slides as a **product / market analyst** would.
What is NS-RAM *for*? Who would buy it? What does it replace?

## Required answers (only these three — be specific, cite slides)

### Q1 — Implied real-world products & applications

What concrete real-world **products or application domains** do these
slides imply NS-RAM is designed for?

For each candidate application, cite the specific slide elements (text,
plots, schematics, area/energy callouts, MNIST-style benchmarks, the
LIF/Poisson choice, the input-neuron + soma topology, the firing-range
windows, etc.) that point at that application.

Be liberal — propose 5–10 candidates. Cite slide numbers always.

### Q2 — Gap vs. state-of-the-art neuromorphic / NPU chips

For each candidate application from Q1, what is the **gap** between
what NS-RAM appears designed for and what existing chips already do?

Compare against (non-exhaustive):
 - Intel Loihi / Loihi-2 (spiking, 128 neuromorphic cores)
 - IBM TrueNorth / NorthPole
 - SynSense Speck/Xylo, BrainChip Akida (commercial spiking)
 - Mythic / Syntiant (analog/mixed-signal NPU for keyword spotting)
 - Standard edge NPUs (Coral, Ethos-U, Hexagon)
 - RRAM/PCM analog in-memory compute (TSMC, IBM, etc.)

Where is NS-RAM **uniquely positioned**? Where is it just another
analog-memory contender? What does its 130/180 nm CMOS-compatibility
(vs. exotic stack) buy commercially?

### Q3 — Mario + Sebas's intended commercial pathway

From these slides alone, what is the **commercial pathway** the
authors seem to be aiming at? Pick the most evidence-backed pathway
(or 2–3 if the slides genuinely fork).

Examples of plausible pathways to consider:
 - always-on keyword / wake-word audio (SynSense, Syntiant territory)
 - edge anomaly detection (industrial sensors, predictive maintenance)
 - prosthetics / biosignal classification (EMG, EEG)
 - in-sensor compute / event-camera back-end
 - ultra-low-power inference accelerator for MCUs
 - reservoir / liquid-state machine substrate
 - cryogenic / radiation-hard application (legacy CMOS node argument)
 - research IP licensing to a foundry (TSMC, GF) rather than chip product

Cite slide evidence — area numbers, process node, firing-range, MNIST
result, schematic complexity, etc. — for whichever pathway you pick.

## Format

```
## Q1 — Candidate applications
1. <APPLICATION NAME>
   - Slide evidence: [slide N] <quote/description>; [slide M] …
   - Strength of fit: HIGH / MEDIUM / LOW

2. …

## Q2 — Competitive gap
For each application above:
 - vs Loihi / TrueNorth: …
 - vs Akida / Syntiant: …
 - vs RRAM analog: …
 - NS-RAM unique angle: …

## Q3 — Commercial pathway
Primary bet: <pathway>
Slide evidence: …
Secondary bet (if any): …
```

**Be opinionated.** We are not asking what *could* be done with any
analog memory — we want what *these specific slides* prioritise. If
the slides are silent on commercial intent, say so explicitly rather
than fabricating one.

Word budget: 1500–2000 words. Be dense and citation-heavy.
