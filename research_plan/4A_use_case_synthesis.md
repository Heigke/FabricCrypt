# 4A Use-Case Synthesis — what NS-RAM is FOR

**Date**: 2026-05-12
**Sources**: O44 oracle 3-way (gpt-5 325s / gemini-2.5-pro 144s / grok-4 82s) on 21 Mario+Sebas slides.
**Method**: identical 3-question prompt (apps, gap-vs-SOTA, commercial pathway) sent to each.

---

## Application matrix

| Application | gpt-5 | gemini | grok | Score | Key slide evidence |
|---|---|---|---|---|---|
| Always-on KWS / wake-word audio | HIGH | HIGH | HIGH | **9/9** | 0.2 pJ/spike (s3/19), 21 fJ/spike + 60-360 kHz (s4/19), 1V CMOS pulses (s8/19) |
| Industrial anomaly detection (vibration, motor, ultrasound) | HIGH | HIGH | MED | **8/9** | nW class, 8 µm² (s3/16), 100% yield, "ultra-reliable" (s3/16, 8/16) |
| Edge AI / ultra-low-power MCU co-processor | HIGH | HIGH | HIGH | **9/9** | "~1000× improvement" claim (s3/26), 8-17 µm² cells, std 130 nm CMOS |
| In-sensor compute / event-camera (DVS) backend | MED | MED | HIGH | **7/9** | per-cell 3×3 µm (s3/26), spiking threshold + hysteresis (s8/16) |
| Biosignal classification (EMG/EEG/ECG, prosthetics) | MED | MED | MED | **6/9** | fJ/spike, LIF behavior matches biosignals (s19), nW operation (s3/16) |
| Acoustic/machine-state monitoring (appliances, IoT) | HIGH | — | — | 3/9 | rate-coded I→spike (s4/19) — gpt-5 only |
| SNN research IP / academic platform | HIGH | HIGH | — | 6/9 | standard CMOS, Brian2 + SPICE models, MPW-compatible cells |
| Reservoir / liquid-state computing | MED-LOW | — | MED | 3/9 | ramp-rate dynamics (s8/16, dynamic-response slide) |
| Cryo / rad-hard / aerospace | — | — | LOW | 1/9 | grok only, weak evidence |

(HIGH=3 pts, MED=2, LOW=1, —=0)

---

## Top 3 (oracle consensus + strongest slide evidence)

### 1. Always-on KWS / wake-word audio detection (9/9)
- **Strongest single slide**: s3/19 — explicit "as low as 0.2 pJ per spike" + 111 µm² cell area + configurable firing rate. This is the canonical always-on audio energy band.
- **Why structural fit**: NS-RAM's rate-coded I→spike with audio-band firing (60-360 kHz, s4/19) plus 1V CMOS-friendly pulses (s8/19) maps directly to MFCC-filterbank → spike-train → SNN classifier flow.
- **Market**: Cortex-M4 + tiny DNN does 90% at ~3 mW; sub-mW at same accuracy is unsolved. SynSense Speck, Syntiant NDP, Aspinity all play here.

### 2. Ultra-low-power MCU co-processor / Edge AI accelerator (9/9)
- **Strongest single slide**: s3/26 — "NS-RAM in standard triple-well 130 nm CMOS," "~1000× improvement to state-of-the-art neuron cores," "can be reduced to a two-transistor cell."
- **Why structural fit**: Tiny analog macro that sits next to digital MCU, wakes the big core only on event. Standard logic process = drop-in IP block.
- **Market**: BrainChip Akida, Mythic (analog IMC), Syntiant. NS-RAM's *standard CMOS* angle is uniquely de-risked vs. RRAM/PCM competitors.

### 3. Industrial anomaly detection in sensors (8/9)
- **Strongest single slide**: s3/16 — 1T neuron-synapse at 8 µm², nW operation, **"100% yield"**, deep-Nwell isolation. The yield+nW combo is what makes this commercially serious.
- **Why structural fit**: Anomalies are rare → self-resetting cell only consumes power during events. Wide firing window (7-10⁴×) matches sensor dynamic range.
- **Market**: predictive maintenance (vibration/motor current/ultrasound) is the real $-volume neuromorphic story now (Roviero, Aspinity).

---

## Commercial pathway implied by slides — STRONG 3-way convergence

**All three oracles independently arrived at the same answer**: this is an **IP licensing play**, not a standalone-chip play.

Evidence stack (consensus):
1. **No full-chip layout, memory architecture, or I/O** shown anywhere — all 21 slides stay at device/cell/small-circuit level.
2. **Heavy SPICE-model effort** (s6, 9, 13, 14, 15) — the *only* reason to invest this much in model accuracy is to deliver a customer-trustable IP datasheet.
3. **Repeated "standard CMOS 130 nm, low cost, 100% yield"** marketing language — direct pitch to licensees worried about process risk.
4. **High-level Brian2 model** alongside SPICE (s19) — classic IP commercialization funnel (algorithm devs evaluate before HW integration).
5. **Marketing KPIs ready-made for an IP datasheet**: pJ/spike, µm²/neuron, "1000× vs SOTA."

**Customer profile**: fabless MCU/SoC vendors targeting IoT, wearables, always-on audio — license a "spiking neuron core" macro to add a differentiated low-power AI block.

**Secondary pathway (gpt-5 + grok agree)**: in-sensor compute back-end for image sensors / MEMS microphones (analog sparsification on-die before ADC).

---

## Gaps vs Loihi / TrueNorth / NPUs

Consensus across oracles:
- **vs Loihi/TrueNorth**: NS-RAM is *scalpel vs Swiss army knife*. Orders of magnitude smaller and lower-power per neuron, but **no on-chip learning, no routing fabric**. NS-RAM is a neuron building block; Loihi is a full neuromorphic processor.
- **vs SynSense Speck / Syntiant NDP / BrainChip Akida** (the actual commercial competitors): NS-RAM's *standard logic process* is the unique angle. Competitors need full-chip product; NS-RAM can be a 0.01 mm² macro inside any 130 nm SoC.
- **vs Mythic / Syntiant analog-IMC**: paradigm difference. Mythic computes *synapses* (weights) on flash; NS-RAM computes *neurons* (somas) on floating-body. NS-RAM is for event-driven SNNs, not static ANN matmul.

---

## Where oracles disagree

- **Reservoir computing**: gpt-5 = MED-LOW, gemini = doesn't mention, grok = MED. Slides show ramp-rate dynamics (s8/16) but no explicit reservoir benchmark. This is *our* hypothesis (z2210+) more than Mario's intent.
- **Bio-signal applications**: all three call this MED but for different reasons. gpt-5 frames it as "EMG/EEG via τ-tuning"; gemini says "no biomedical benchmarks in slides — inferred from physics"; grok ties it to deep-Nwell reliability. Consensus: technically plausible, slides don't directly target it.
- **Cryo / rad-hard**: only grok suggests this; gpt-5 + gemini stay silent. Speculative.

---

## Pitfalls / risks to the synthesis

1. **The slides are pre-product**: even the strongest consensus (KWS) lacks a head-to-head benchmark vs Syntiant/Cortex-M4 in slide form. The implied applications are *technologically plausible* but not yet *competitively demonstrated*.
2. **MNIST 72% (s8/15)** is a system-level proof but a weak headline — well below ANN baselines (~98%). The commercial pitch needs harder benchmarks (KWS, ECG, NAB).
3. **All three oracles are inferring intent from physics specs**; Mario+Sebas's own commercial framing is *only implicit* in the slides. A direct ask of Mario would confirm/refute the IP-licensing thesis.

---

## Path to brief v4.4

1. Hard application benchmarks for **top 3 confirmed by oracles**: KWS-delta (Phase 4C.4), NAB anomaly (4C.2), and continue ECG (DS-N4 in flight).
2. Reframe the brief: lead with **"NS-RAM is a licensable spiking-neuron-core IP block for standard-CMOS MCU/SoC integration"** — that is the consensus commercial frame.
3. Position the architectural sweep results (4B) as "envelope characterization for IP datasheet" rather than internal optimization.
4. Drop or de-emphasize: reservoir computing, cryo, MNIST headline numbers — these are NOT what the slides imply NS-RAM is FOR.
