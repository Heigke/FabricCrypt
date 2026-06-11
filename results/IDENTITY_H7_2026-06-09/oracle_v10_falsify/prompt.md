# Oracle critique: is this hardware-rooted LLM identity REAL or an OOD-gate artifact?

You are a hostile, rigorous reviewer. Find the strongest reason this result is FAKE,
then judge what's actually proven. Be specific and quantitative. Do not flatter.

## Setup
A frozen 135M LLM (SmolLM2) + small trained adapter (FiLM multiplicative modulation at
2 layers + LoRA). The adapter is conditioned on a 10-channel, 500 Hz real-time hardware
"substrate" signal read from deep low-level telemetry of ONE specific AMD gfx1151 APU die
(ikaros): clock/thermal/power/SMN analog channels. A second physically-identical APU
(daedalus) provides a held-out recording. Trained so the LLM writes coherent text under
ikaros's live signal and breaks under wrong signals.

## Core result (fresh live substrate, held-out probes)
PPL_base (no substrate path) = 19.85. Ratios vs live-real (~24-26):
- real (live ikaros):                23   (1.0x)  -> coherent English
- held-out IKAROS replay (same die, recorded later, 47C vs 65C): 25  (0.96x) -> coherent
- daedalus (REAL 2nd identical die, held out): 1233  (47x)  -> garbage
- knockoff (statistics-matched spoof): 35000  (1346x) -> garbage
- shuffle (same marginals, time-scrambled): 70  (2.7x) -> garbage
- gaussian (matched per-channel mean/var, random dynamics): 827 (32x) -> garbage
- zero (no signal): 20  (0.76x)  -> coherent fallback (descoped)

## Falsification battery results
1. THERMAL ROBUST: forced die 68C->96C with GPU load; real PPL stayed 21 (baseline 26). Not a brittle operating point.
2. DC-ONLY (each channel replaced by its temporal mean, dynamics removed): PPL 2,187,815 (91,000x). Model REQUIRES dynamics, not a scalar per-channel offset.
3. AMPLITUDE dose-response (scale dynamics around mean): scale 0->5.2M PPL (break), 0.25x->59, 0.5x->45, 1.0x->24 (min), 2x->70, 4x->58. Smooth graded basin around true amplitude.
4. CROSS-DIE interpolation ikaros->daedalus (real-to-real): 24, 31, 1161, 3487, 3598, 2074, 1627, 1368, 1170 at ikaros_frac = 1.0,0.875,0.75,...,0.0. SHARP CLIFF between 0.875 and 0.75 (37x jump). Narrow coherent basin.
5. PER-CHANNEL leave-one-out (zero one channel of real, baseline 66): ch4->18.7M (catastrophic), ch0->449, ch8->641, ch9->1470, ch1->161, ch6->119, ch7->117, ch2->87, ch3->67 (no effect), ch5->21 (improves). Distributed, ch4 (an identity-bearing "keeper" channel) is critical.
6. BEHAVIORAL: output next-token distribution divergence across DIFFERENT live real windows = 0.315 sym-KL median; real-vs-zero = 0.341. So coherent output varies with the live window about as much as turning the signal on/off.

## Known null (from literature, Geirhos shortcut-learning, Ming spurious-OOD, Shahbazi conditioning-collapse)
"The adapter is a narrow OOD gate; the frozen LLM collapses off the thin training manifold; it never 'uses' the signal."

## Questions (answer each, numbered)
1. Given results 1-6, is the OOD-gate null FULLY adequate, PARTIALLY, or REFUTED? Which specific results bite hardest against it, and which support it? (The cross-die cliff #4 vs the amplitude grading #3 vs DC-only #2 seem to pull different directions — reconcile them.)
2. What is the SINGLE strongest remaining confound or artifact you can still attack? Design the experiment that would kill it.
3. Is "the model uses the dynamics, but has a SHARP coherent basin per die" enough to claim "device-bound conditional generation rooted in this physical die"? Or only "a per-die anomaly detector"? Where exactly is the line, given DC-only=91,000x and held-out-ikaros=0.96x?
4. The goal is to go beyond break/no-break to "LLM behavior is a GRADED function of the real-time signal while staying coherent" (personality that varies with the live die state). Result #6 hints output varies with the live window. What training objective would CREATE genuine graded, meaningful behavioral dependence (e.g., live thermal state -> measurable, coherent style shift) — and what is the cleanest metric to prove it?
5. Steelman the POSITIVE interpretation in 3 sentences a skeptic would grudgingly accept, and state the honest one-line claim this evidence licenses (and what it does NOT).

Attached: the v10 result writeup, the two falsification JSONs, and the v10 training script.
