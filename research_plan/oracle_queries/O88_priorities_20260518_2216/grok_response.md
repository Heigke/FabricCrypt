# grok response (grok-4-latest) — 21s

**Q1 — Best next experiment to find the missing parallel path**

Flat 250 nA Id across Vd in deep triode at low VG1 (CHANNEL_ROOT worst-case bias VG1=0.6/VG2=-0.05/Vd=0.05) rules out standard MOSFET channel current. Silicon is constant-current, not ramping. This signature ranks the candidates as:

1. (c) source-body parasitic Schottky (highest likelihood): produces near-constant forward current once |Vsb| exceeds ~0.2-0.3 V, independent of Vd once saturated. Matches exactly the flat triode plateau and the 232 nA vs model 10^{-15} A discrepancy.
2. (a) well-tap forward-conducting diode (still plausible): would clamp Vb and inject constant body current that body-effect modulates into channel. However, every prior BSIMSOI smoke (Vbs ∈ [-3.28, +0.43] V) and F3-HIGH Vbs-clamp control already showed body never forward-biased under Mario biases, lowering its rank.
3. (b) STI/corner leakage path.
4. (d) sub-threshold TAT.
5. (e) vertical PNP or unmodeled well resistance (already killed by F4-PNP KILLSHOT at βP=0.02/Rw=2 kΩ).

Top-2 falsifiers (pre-register before running):

- For (c): force Vsb=0 with ideal voltage source on source-body terminals while sweeping Vd in triode; if flat 250 nA component vanishes → Schottky confirmed. If it remains → rule out.
- For (a): repeat the identical Vbs-clamp control at jtss=100 (O87 falsifier) but now with explicit well-tap contact enabled and Vwell swept ±0.3 V around 0; if Id_triode changes <0.1 dec → diode ruled out.

Instrument solve_2t_steady_state to log Vsint/Vb/Vgs_eff/Vth per Vd on the three worst biases (P3 full-grid set) before any new silicon request. This is the cheapest, highest-signal experiment left.

**Q2 — Large-scale sim opportunities**

With a 4-dec DC gap accepted and V7 already dead (V7 PVT 0/72 corners, bimodal 8-40 ns or 2.5 µs only), large-N simulations have sharply limited value. The only surviving primitives are V6 40 ns reset, V8 LIF integrate, Mario single-bias calibration, and diff IFT pyport (z474b).

Highest-ROI targets:

- (a) emergent dynamics on N=10^4-10^5 cells using only V6+V8 modes (not full Mario stack). Metric: reset-induced synchrony decay time and LIF leak time-constant distribution across 500 Monte-Carlo Mario-extracted parameter sets. Target: show <5 % period drift vs single-cell when N>10^4. This directly feeds the “multi-function from one cell via bias only” claim that O84/O85 still endorse.
- (b) variability impact: Monte-Carlo (N=500) over the exact Mario scatter reported in z47x calibration runs. Metric: fraction of cells that still produce usable V6 reset (<50 ns) or V8 monotonic integration window. Target: >70 % yield at the calibrated bias point. This is cheap on zgx and directly supports any methods paper.
- (c) train deep network with diff IFT pyport as activation: run DS-1 style MNIST but now with the final-epoch, no-early-stop protocol already shown to drop z481 from 90.8 % to ~60-70 %. Metric: final test accuracy + alignment-cosine vs BP baseline. If it stays >80 % after 10 seeds, the methodology claim survives; otherwise it becomes a pure “diff IFT estimator” methods note.

Anything involving reservoirs, CPG lock-in, or full EP on the broken cell model is dead weight (z482/z479/z483 already killed it). Focus only on what the silicon-calibrated primitives can actually demonstrate.

**Q3 — Pivot decision tree**

If Pillar A (parallel-path fix) fails by Day 7 — i.e., Vbs-clamp + Schottky test both return negative and no layout change is authorized — the highest-value pivot is the pure methods paper on diff IFT pyport + Mario single-bias calibration + V6/V8 multi-functionality, not ML emulator or full shutdown.

Rationale: DS-1 already shows the EP-FIX route collapses without cherry-picking (final-epoch drops 15-28 pp). V7 is dead. The 4-dec DC gap is now data-limited (Sebas blocker) rather than model-limited after CHANNEL_ROOT and F3-HIGH band-aid. Continuing to chase a physics fix with no new silicon is resource-negative.

A clean methods paper (IEDM circuits track) that states:
- diff IFT estimator works on a real 130 nm 2T cell,
- V6 reset 40 ns and V8 LIF are bias-programmable on the same device,
- calibration holds at one primary Mario bias point,

is still publishable and does not over-claim. An ML-emulator paper would be weaker (no one cares about emulating a broken model). Full shutdown wastes the z474b/z47x/z481 infrastructure already built.

Decision rule: run the two falsifiers in Q1 on Day 1-2. If both negative and Sebas still blocks new cards, write v4.7 as methods-only by Day 5 and reallocate the remaining 9 days to a different project. No further physics or large-N hero runs.
