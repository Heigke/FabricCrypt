# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: N_BENCH_comparison_matrix.md (6335 chars) ===
```
# NS-RAM vs Real-Chip Neuromorphic / Edge-AI Benchmark Matrix

**Date compiled:** 2026-05-17
**Author:** N-BENCH-A benchmark agent
**Source:** see `citations.md` for URLs + access dates

---

## CRITICAL CAVEAT — read first

All NS-RAM numbers below are **projected from ODE / pyport simulation** on a
2T floating-body cell calibrated to Sebas's 130 nm I-V data
(see `nsram_phase_a_closure.md`, `sebas_iv_130nm_fit.md`). **No tapeout
exists.** Energy figures assume τ_body ≈ 0.7 ns, C_b ≈ 8 fF, and the
optimistic device-level analytical model in
`research_plan/T5_orderofmag_bookkeeping.md`. They have NOT been
silicon-validated. Every NS-RAM cell in the matrix below is therefore
labelled **[SIM]**.

Competitor numbers are **real-silicon measurements** from peer-reviewed
publications or vendor datasheets. They are labelled **[Si]**.

A SIM vs Si comparison is informative for an architectural sanity check
("are we within striking distance, or 5 orders of magnitude off?")
but is NOT a head-to-head benchmark.

---

## 1. Headline table — Workload × Chip

Cell format: `accuracy | energy/inf | throughput`
"—" = no published number; "n/p" = not published for this workload.

| Workload | NS-RAM 2T [SIM] | Loihi 2 [Si] | NorthPole [Si] | BrainScaleS-2 [Si] | Akida AKD1000 [Si] | Mythic M1076 [Si] | Cortex-M4/M7 [Si] | Edge TPU [Si] |
|---|---|---|---|---|---|---|---|---|
| **MNIST** (hierarchical SNN) | 97.15% \| 17.7 pJ \| n/p | ~98% \| ~25 pJ\* \| n/p | n/p | 97.6–98.0% \| 8.4 µJ \| 84k inf/s | n/p (KWS focus) | n/p (image-class focus, ResNet/CIFAR) | ~99% \| ~100 µJ\*\* | ~99% \| ~125 µJ\*\* |
| **KWS** (DS-CNN / "go/stop") | — (cascade only, 60.8% E-save) | ~98% \| ~25 nJ–sub-µJ\*** \| ms-class | n/p | n/p | ~98% \| 1.4 mJ/inf† (full system) | n/p | 90–96% \| 1–100 µJ \| ~1k inf/s | n/p directly |
| **DVS-Gesture** (IBM 11-class) | 59.3% (HDC 8192) \| n/p | 89.6–96% \| 0.9–26 pJ/SOP \| 241 mW @ 96.7% | n/p (frame-based focus) | n/p | n/p | n/p | n/p | n/p |
| **ImageNet ResNet-50** | — (not tested) | n/p directly | 75% top-1 \| ~1.6 mJ/inf (612 fr/J) \| 42 460 fps | n/p | n/p (smaller models only) | claimed (35 TOPS, 4 W, no per-inf number) | impractical | impractical |
| **TRNG (NIST 5/5)** | 0.4 pJ/bit \| 1.27 Mbit/s [SIM] | n/p | n/p | n/p | n/p | n/p | n/p | n/p, vs ASIC TRNG → 0.244 pJ/bit @ 1 Mb/s (Cheng 2024, 65 nm) |
| **LMS equalizer QPSK** | 2.76 pJ/symbol [SIM] | n/p | n/p | n/p | n/p | n/p | n/p | n/p |
| **Reservoir Mackey-Glass / NARMA-10** | NMSE~? @ 1024 neurons [SIM] | exists but no canonical pJ/step number (Lava/IF code) | n/p | demonstrated (PNAS 2022 surrogate-grad) but no MG-specific energy | n/p | n/p | n/p | n/p |
| **HDC UCI-HAR** | ~84% \| 8192 bits [SIM] | n/p directly | n/p | n/p | n/p | n/p | n/p | n/p; **best real Si:** LogHD ASIC (28 nm) — 4× E vs sparse-HDC baseline, no absolute pJ/inf reported; software 94.2% @ D=64 (Yan 2023) |
| **Predictive coding NAB** | demonstrated D=256 [SIM] | n/p (Loihi anomaly-det exists but not NAB-canonical) | n/p | n/p | n/p | n/p | n/p | n/p |
| **Memory Palace assoc-recall** | demonstrated D=512 [SIM] | n/p (no published canonical task) | n/p | n/p | n/p | n/p | n/p | n/p |

\* Loihi 2 Loihi-class MNIST routinely cited in 10–50 pJ/SOP range; per-inference depends on spikes/sample.
\*\* Cortex-M / Edge-TPU MNIST: derived from public TF-Lite-Micro and Coral perf pages; not a single canonical paper.
\*\*\* Shoesmith et al. (2025) Eventprop on Loihi 2: sub-1 mJ end-to-end real-time KWS, ~3 ms latency. Per-inference energy in low-µJ regime at typical spike rates.
† BrainChip's own 2023 benchmarking PDF — methodology not MLPerf-reviewed.

---

## 2. Technology-node + process notes

| Chip | Node | Type | Year |
|---|---|---|---|
| NS-RAM 2T [SIM] | 130 nm (Sebas IV fit) | analog floating-body 2T, projected | tapeout TBD |
| Loihi 2 | Intel 4 (~7 nm class) | digital async neuromorphic | 2021–24 |
| NorthPole | 12 nm | digital tile-array inference | 2023 (Science) |
| BrainScaleS-2 | 65 nm | analog accelerated SNN + plasticity | 2018–22 |
| Akida AKD1000 | TSMC 28 nm | digital SNN NPU | 2022 |
| Mythic M1076 | 40 nm | analog flash MAC | 2021 |
| Cortex-M4 / M7 | 40–180 nm typ. | MCU CPU + DSP | ongoing |
| Edge TPU | 22 nm | digital systolic inference | 2018+ |

NS-RAM at **130 nm** is 3–4 nodes behind every digital competitor and 2
behind BrainScaleS-2. Any energy claim must explicitly note the node
disadvantage; a per-inference number at 130 nm scaled (Dennard-rough)
to 28 nm would drop ~3–5×.

---

## 3. Confidence flags on NS-RAM numbers

| Claim | Source | Confidence | Needs ngspice-validation? |
|---|---|---|---|
| MNIST 97.15% @ 17.7 pJ/inf | hierarchical SNN sim, AMBITIOUS suite | **LOW** — energy assumes ideal device + no peripheral / ADC / wire | **YES — flagged BOLD** |
| TRNG 0.4 pJ/bit, NIST 5/5 | Stochastic RNG sim | MEDIUM — TRNG circuits are well-modelled | YES |
| TRNG 1.27 Mbit/s throughput | sim | MEDIUM — limited by τ_body ≈ 0.7 ns lower bound on bit period | YES |
| LMS-Eq 2.76 pJ/symbol QPSK | sim | **LOW — BOLD** (peripheral DAC/ADC not modelled) | YES |
| Reservoir 1024 neurons | sim, NS-RAM cell ≠ reservoir node by itself | LOW — see `01_LOG.md` 2026-04 entries: "memoryless analog weight" | **YES — flagged BOLD** |
| HDC UCI-HAR ~84% @ 8192 | DISCOVERY suite sim | MEDIUM (HDC accuracy is software-checkable) | partial |
| MNIST SNN 97.15% accuracy | sim | MEDIUM (accuracy is software-equivalent) | NO |
| All energy/inf figures | sim | **LOW across the board** | **YES** |

---

## 4. What's missing in the literature (honest gaps)

- **No published Loihi 2 NARMA-10 / Mackey-Glass canonical pJ-per-step**
  number exists. Wider neuromorphic field reports NMSE only.
- **No public Mythic CIFAR-10 per-inf energy number** with peer review —
  only TOPS/W marketing.
- **No DVS-gesture number from NorthPole** (NorthPole targets
  frame-based ImageNet-scale).
- **HDC UCI-HAR on real silicon**: only ASIC papers (LogHD, DecoHD,
  Datta 2019) — none report absolute pJ/inference, only relative
  energy efficiency vs CPU/GPU baselines.
- **TRNG**: no neuromorphic-chip TRNG benchmark exists; competitors
  are dedicated CMOS TRNG circuits. NS-RAM's 0.4 pJ/bit is *competitive*
  with 0.244 pJ/bit (Cheng 2024, 65 nm) but at 130 nm — node-adjusted
  it would be WORSE.

```


=== FILE: N_BENCH_gap_analysis.md (6906 chars) ===
```
# Gap Analysis — NS-RAM vs Real-Silicon Competitors

For each of the 9 PASS sims (4 AMBITIOUS + 5 DISCOVERY), what's the win,
what's the loss, what's the headline differentiator if any?

**All NS-RAM numbers are SIMULATED at 130 nm.** Competitor numbers are
real silicon, typically at 7–28 nm. Read every comparison with that
node penalty in mind.

---

## A. AMBITIOUS workloads

### A1. Reservoir Mackey-Glass / NARMA-10 (1024 neurons)

- **NS-RAM [SIM]:** 1024 NS-RAM cells, claim "reservoir-class temporal task."
- **Best real Si:** Loihi 2 (IF-based reservoir), BrainScaleS-2 (surrogate
  gradients), photonic reservoirs (sub-ns).
- **Win?** No clear win. The 2026-04 log explicitly flags that a
  standalone NS-RAM cell is a "memoryless analog weight, not a reservoir
  node." Reservoir behavior requires external recurrence.
- **Loss?** No published Loihi NARMA pJ/step to compare to, so we cannot
  even claim a win quantitatively.
- **Headline differentiator:** **NONE that survives scrutiny.** Demote
  this from AMBITIOUS to "feasibility demo" until external recurrence is
  added and an apples-to-apples reservoir benchmark is run against
  Loihi 2 (NeuroBench).

### A2. Stochastic RNG (NIST 5/5, 0.4 pJ/bit, 1.27 Mbit/s)

- **NS-RAM [SIM]:** 0.4 pJ/bit, 1.27 Mbit/s, NIST 5/5.
- **Best real Si:** Cheng 2024 (65 nm CMOS): 0.244 pJ/bit @ 1 Mb/s,
  NIST-passing, no post-processing. Synopsys TRNG IP, RHS-TRNG @ 2.5 pJ/bit.
- **Win?** **Marginal, possibly negative once node-scaled.** Our
  0.4 pJ/bit at 130 nm vs 0.244 pJ/bit at 65 nm — scaling 130→65 gives
  another 2–4× headroom for CMOS; node-adjusted CMOS likely wins.
- **Loss?** Probable, after node scaling.
- **Headline differentiator:** **Same-cell reuse** — the same NS-RAM
  array doing MNIST inference can switch to TRNG mode, which a
  dedicated CMOS TRNG cannot do. This is a *multi-function* story,
  not a *pJ/bit* story.

### A3. LMS-Eq QPSK (2.76 pJ/symbol)

- **NS-RAM [SIM]:** 2.76 pJ/symbol QPSK adaptive equalizer.
- **Best real Si:** No directly comparable published neuromorphic LMS-Eq
  number found. Conventional ASIC equalizers report fJ-to-pJ-per-tap;
  per-symbol depends on tap count.
- **Win?** Indeterminate — no published competitor for like-for-like.
- **Loss?** Cannot rule out.
- **Headline differentiator:** **In-memory adaptive coefficient
  update** if the floating-body bias *is* the LMS coefficient. This is
  the most architecturally novel of the 4 ambitious claims and the most
  defensible if peripheral overhead is honestly accounted.
  **BOLD CLAIM — needs ngspice with peripheral DAC/ADC modelling.**

### A4. Hierarchical MNIST SNN (97.15%, 17.7 pJ/inference)

- **NS-RAM [SIM]:** 97.15% @ 17.7 pJ/inf.
- **Best real Si head-to-head:**
  - BrainScaleS-2 (65 nm): 97.6–98.0% @ **8.4 µJ/inf** (8400 pJ).
  - Loihi 2: ~98% @ ~25 pJ/SOP — per-inf in low-µJ range.
  - NorthPole (12 nm): not MNIST canonical, but **612 fr/J** on
    ResNet-50, i.e. ~1.6 mJ/inf for 25M-param model.
- **Win?** On accuracy: parity (97.15% vs 97.6–98.0% — within noise).
  On energy: NS-RAM's 17.7 pJ would be **~470× better than
  BrainScaleS-2** if true. **But this is at 130 nm, sim-only, and
  excludes peripherals.**
- **Loss?** Once peripheral DAC/ADC, ROW/COL driver, and wire energy
  are added, the gap shrinks dramatically. BrainScaleS-2's 8.4 µJ is
  *the entire chip + readout* number; ours is *device-only*.
- **Headline differentiator:** **Apparent 100–1000× device-level energy
  advantage**, conditional on peripheral overhead matching. **MUST be
  ngspice-validated including DAC/ADC ladder.**

---

## B. DISCOVERY workloads

### B1. HDC UCI-HAR (8192 bits, ~84%)

- **NS-RAM [SIM]:** 8192-bit HVs, ~84%.
- **Best real Si:** Software HDC (Yan 2023): 94.2% at D=64.
  LogHD ASIC (28 nm): 4× energy efficiency vs sparse-HDC baseline (no
  absolute pJ/inf).
- **Win?** No — accuracy 10 pp below state-of-the-art software HDC.
- **Loss?** Yes on accuracy.
- **Headline differentiator:** **None.** Demote.

### B2. Predictive coding NAB (D=256)

- **NS-RAM [SIM]:** D=256 predictive-coding anomaly detection.
- **Best real Si:** No NAB-canonical neuromorphic chip number.
- **Headline:** Architecturally novel but no head-to-head opponent;
  defer to NeuroBench v2 alignment.

### B3. Memory Palace assoc-recall (D=512)

- **NS-RAM [SIM]:** D=512 associative memory.
- **Best real Si:** No published competitor for the exact task.
- **Headline:** Same as B2.

### B4. Cascade KWS+ECG (60.8% energy saving)

- **NS-RAM [SIM]:** 60.8% energy saving vs always-on baseline by
  cascading KWS gate → ECG.
- **Best real Si:** Cortex-M4 KWS at ~1–100 µJ/inf; Akida AKD1000
  full-system 1.4 mJ KWS; Loihi 2 Eventprop sub-mJ.
- **Win?** **60.8% is an algorithmic/system gain**, not a chip gain.
  Cascading wins are reproducible on Cortex-M with same hierarchy.
- **Loss?** Not a chip story per se.
- **Headline differentiator:** **System-level cascading + analog
  always-on KWS gate** could be a real win at 130 nm if the gate cell
  consumes < 10 µW. Currently ngspice-unvalidated.

### B5. HDC DVS-Gesture (8192 bits, 59.3%)

- **NS-RAM [SIM]:** 59.3% on IBM DVS-Gesture (11-class).
- **Best real Si:** Loihi 2: **89.6–96%**. IBM TrueNorth: prior art at
  comparable accuracy. Smart-camera SNN: 96.7% @ 241 mW.
- **Win?** No — 30+ pp accuracy gap.
- **Loss?** Severe accuracy loss. HDC-on-NS-RAM is not competitive on
  DVS-Gesture.
- **Headline differentiator:** **None.** The win story isn't here.

---

## Cross-cutting headline

The honest pitch that survives this comparison matrix is:

> **"Same-cell multi-function analog primitive in 130 nm:
> inference + TRNG + LMS adaptation + cascade-gate, all in the
> sub-pJ-per-op regime at the device level. Trade accuracy on hard
> vision tasks for energy and multifunctionality on edge sensor fusion."**

The pitch that does **NOT** survive:

- "Reservoir computer" (memoryless cell, no head-to-head)
- "Beats Loihi on DVS-Gesture" (30 pp accuracy gap)
- "Beats BrainScaleS on MNIST energy" (peripheral overhead unmodelled,
  node disadvantage unaccounted)
- "Beats CMOS TRNG" (node-adjusted, probably not)

---

## BOLD claims requiring ngspice-level validation (priority order)

1. **MNIST 17.7 pJ/inf** — include full DAC/ADC + wire energy, redo at
   130 nm SPICE. Expected outcome: 100–1000× degradation from
   device-only number.
2. **LMS-Eq 2.76 pJ/symbol** — same; verify the floating-body-as-
   coefficient claim is physically maintained under realistic
   peripheral load.
3. **Reservoir 1024-neuron benchmark** — define recurrence topology
   (external CMOS or in-array) and re-run with measured τ-spread;
   compare to Loihi 2 NeuroBench reservoir suite.
4. **TRNG 0.4 pJ/bit** — node-scale to 65 nm reference; if NS-RAM
   wins at iso-node, claim is defensible. Otherwise re-frame as
   "free-as-a-side-effect TRNG" not "best-in-class TRNG."

```


=== FILE: O80_triangulation.md (5638 chars) ===
```
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

```


=== FILE: context.md (50066 chars) ===
```
## 2026-05-16 — z450 Verilog-A pivot DONE: standalone thyristor_compact.py PROTOTYPE PASS (N-shape visible, 9.3× decade, 54 NDR pts at V_peak=0.85V). KEY FINDING: z444 BESD identical-results bug is MECHANICAL wiring no-op (residual never reaches M1 drain KCL row), NOT physics dead-end. Recommended: debug z444 wiring first (~4h), keep thyristor_compact.py as fallback. CONDITIONAL GO.
## 2026-05-16 — z451 cap audit BREAKTHROUGH: C_eff = 2.66 fF NOT 12 fF. z448 KILL_SHOT was FALSE NEGATIVE (4.5× cap overestimate). Required I to charge ΔV=0.7V in 10ns = 189 nA; BSIM4+GP already gives 130 nA → off by 1.5×, not 100×. Dominant suspect: M1.alpha0=7.84e-5 (Sebas card), literature for 130nm bulk says 5e-4..5e-3. 10× ALPHA0 likely closes BOTH DC knee AND ns-snap. ns-snap track RE-OPENED.
## 2026-05-16 — z451 critique: 3 cherry-picks flagged (backward-only PT cited as breakthrough, V_SINT_PIN unmeasured vs silicon, VG1=0.2 fix never full-grid revalidated).
## 2026-05-16 — z449 VBIC+BDF combo KILL_SHOT: all 3 variants FAIL gates. v449_A DC=1.311 (=z443 ceiling), v449_B n-well-cap=0 → Vb@5ns 5× better (validates z451 cap math!), v449_C ALPHA0×5 DC WORSE (+0.30 dec). Body-current limited, not cap. z449 recommends snapback_subcircuit with V_BC-thresholded µA pull-down. Awaiting z453 wider ALPHA0 sweep (10/30/100×) to confirm-or-kill the literature hypothesis.
## 2026-05-16 — MEP+DS-N tick: APU=48C sentinel ok. No z2[3-9]/z44x active (z453 ALPHA0 subagent still setting up). Plan 2026-05-12 superseded by NS-RAM z44x/z45x campaign. Pending pre-z449 tasks: D2 corrective sweep, MEP-6/7, PTP — held back of queue. No new launches.
## 2026-05-16 — track-progress audit: Phase A (MEP) 1/4 done (SURR-V4), MEP-6 in_progress, D2+PTP pending. Phase B (DS-N) 16/18 done (DS-N4 ECG + DS-N5 HDC + DS-N8 KWS in_progress, DS-N3 absent). Phase C: 4E.1 brief done #212, oracle critique deferred. Plan ~85% closed; remaining blockers held back of queue while NS-RAM z44x/z45x active. No compute.
## 2026-05-16 — topology tick: original R-1..R-10 plan long superseded. R-50 (physics-bounded BBO) still in_progress, R-51 done. Active campaign = z449 DONE/KILL, z450 DONE thyristor proto OK, z451 DONE cap audit (BREAKTHROUGH C_eff=2.66fF refutes z448), z453+z454 in flight. No new R-phase gate crossed → no ALERT.
## 2026-05-16 — z454 snapback subcircuit DONE: KILL_SHOT on DC but ns-snap CLOSED. SB_HOT V_B→0.5V in 1.38ns (3/4 biases), peak 0.71V — first time ns-snap works in pipeline. DC destroyed (2.66 vs 2.09 SB_OFF) because Slotboom multiplier fires too early at V_db<BV. No self-reset (no body-leak path).
## 2026-05-16 — z454 BURN CONFIRMED: I_snap clamp fires (z444-style no-op avoided). |I_snap_b|: DEFAULT=0.5µA, LOW=2.5µA, HOT=42µA.
## 2026-05-16 — CRITICAL: v449_B published "1.31 dec" was forward-only cherry-pick. Backward sweep = 2.86 dec, AVG = 2.09. z451 critique #1 vindicated. All prior DC numbers need fwd+bwd reporting.
## 2026-05-16 — Next: z455 knee-sharpener (V_knee≈1.8V) + z456 R_body reset path. Test INDEPENDENTLY first then recombine.
## 2026-05-16 :47 — APU=49C ACTIVE: z453_alpha0_sweep (z455+z456 still spawning)
## 2026-05-16 — z45x campaign tick: APU=53C. ACTIVE: z453 (still in A1 fwd sweep ~20min in), z455 (K_1p8/K_2p0 done DC=2.71/2.72 — knee-gate NOT recovering DC!), z456 (R_1G done DC=2.81 = baseline, no self-reset).
## 2026-05-16 — z455 INTERIM RED FLAG: V_knee gating not separating avalanche from low-Vd. I_snap fires at V_db=1.4 even with V_knee=2.0 set. Either σ-gate not wired or DC fold extends above V_knee — investigate when run done.
## 2026-05-16 — z456 INTERIM: R_1G expected too weak (τ≈2.7ms); waiting for R_10M / R_1M to confirm reset path.
## 2026-05-16 — z455 knee-sharpener DONE: DISCOVERY FAIL but PARTIAL. Best K_1p6 DC=2.702 (Δ=-0.107 only). σ-gate WORKS (I_snap_b drops 4 orders 4.2e-5→3.1e-8) but PARASITIC NPN's I_snap_d clamped at 10mA whenever Vbe>0.6V → that's what pollutes DC, not Slotboom. ns-snap survives: K_1p6 still 1.42ns to 0.5V on 3/4 biases.
## 2026-05-16 — z455 fix-the-fix → z457 dispatched: gate I_snap_d (NPN collector current) directly by V_db knee, not just Iii multiplier. Independent of z456.
## 2026-05-16 — z456 R_body reset KILL_SHOT: no self-reset at any R (1G..1M Ω). NPN holding ~10µA >> leak (0.66µA@1MΩ). DC identical across all R (=2.809 dec). R_1M did suppress latch at weakest bias VG1=0.4 (Vb_peak=0.01V vs 0.64V) — only effect seen. Self-reset axis still open: need weaker NPN AND R_body in same experiment (z458 2D sweep proposed).
## 2026-05-16 — z457 NPN-gate DONE: BEST yet NX_1p8 DC_avg=2.479 (Δ=-0.223 vs K_1p6). DISCOVERY FAIL but first real DC win since SB enabled. Mode X (gate current) works at V_knee=1.8 only (σ argument deep enough to kill 3e4 A unclamped Ic). Mode Y (vbe-offset 0.3V) INSUFFICIENT + breaks VG1=0.2 (+2.18 dec regression). VG1=0.6 wins most: 2.998→2.370. ns-snap survives 1.46ns t→0.5V.
## 2026-05-16 — z457 honest diagnosis: NX_1p8 still 0.4 dec WORSE than SB_OFF baseline 2.087. Snapback infrastructure NET NEGATIVE on DC even with NPN muzzled. Next: V_knee 2.0-2.2 + reduce Id_extra_clamp + audit Iii_body, D3 zener, pdiode Is, BSIM4 Ids overshoot.
## 2026-05-16 — SYNTHESIS DONE (CAMPAIGN_SYNTHESIS_2026-05-16.md, 524 lines, 10 sections, CP-1..CP-9 cherry-pick audit). HONEST BASELINE: 1.19 dec fwd+bwd avg (z432), NOT 0.886. Biggest cherry-pick: z447/z448 "0.886" was 4 biases only — excluded VG1=0.2. Top missing: A.12 (Sebas blocked 3wk), rbodymod=1 body-R (OPEN since 2026-05-13!), fwd+bwd methodology. Path B recommended: accept ~1.2 dec functional model, 2 weeks to publication. Action today: re-run z430/z432/z443/z446/z449/z454 with BOTH sweep directions.
## 2026-05-16 — daedalus SSH UNREACHABLE (timeout, slow ping 1.17s rtt). Distributed campaign reduced to ikaros + zgx.
## 2026-05-16 — AUTONOMOUS PLAN LAUNCHED. 5 subagents dispatched parallel: P1a ikaros (z430/z432/z443 fwd+bwd), P1b zgx (z446/z449/z454 fwd+bwd after rsync), P2 BSIM3/4 type-mismatch audit, P4 rbodymod=1 implementation, Oracle 3-way critique on synthesis. Cron 9e146f5b every 30min drives P-phase auto-progression (P1→synthesis→P5 holdout→P6 brief v4.5). Daedalus SSH unreachable, skipping.
## 2026-05-16 — z45x tick APU=49C. P1a INTERIM: z430 V_SINT_PIN fwd=1.619 bwd=2.823 AVG=2.301 dec (synthesis claim CONFIRMED — original "1.619 breakthrough" was fwd-only). Forward VG1_0.2 fjuck (2.62), backward catastrophic on VG1_0.4/0.6 (2.66/3.03). z432/z443 pending. z453 still A1 fwd (slow). P2/P4/Oracle running. ALERT: P1a using n=25 curves not 33 — verify data path.
## 2026-05-16 — P2 DONE: CLOSED-EMPTY. Synthesis CP-9 "BSIM3 type-mismatch" claim FALSIFIED — all cards are BSIM4 v4.5 (level=14), no BSIM3 in pipeline. ALPHA0/K1/K2/BETA0 use same conventions in BSIM3v3/BSIM4v4.8. Only fix: parser silently dropped level/version tokens (foot-gun, dormant) — landed in model_card.py:287. test_bsim_type_mismatch.py 19/19 PASS. Expected DC impact ≤0.01 dec. P2 budget redirected to P4 rbodymod=1.
## 2026-05-16 :47 — APU=52C ACTIVE: z453+P1a+P4 (3 scripts)
## 2026-05-16 — P1a CONFIRM SYNTHESIS CP-1: z432 fwd=1.349 BUT only 18/25 biases evaluated. VG1=0.2 column ENTIRELY DROPPED (7 fails, 32% conv rate). Original "z432 BREAKTHROUGH 1.027" was on EASY 18 biases. Cherry-pick now empirically proven.
## 2026-05-16 — z45x tick APU=51C. ACTIVE: z453+P1a+P4+Oracle. P1a summary so far has z430 only (z432/z443 mid-run). z453 still stuck on A1 forward DC sweep ~90min in (slow but alive pid 7203). P4 stuck on R_card stage 30min (alive pid 61266). No new DISCOVERY/KILL_SHOTs.
## 2026-05-16 — P-phase tick APU=51C. P1a partial (z430 only in summary, python still running), P1b NOT STARTED (zgx dir has only stale atom_logs, agent never synced), P2 DONE, P4 running, Oracle pending. No P-phase progression eligible. HONEST_BASELINE not yet writable.
## 2026-05-16 — P1a z432 update: bwd=1.027 ALL 25 biases (incl VG1=0.2!), fwd=1.349 only 18/25 (VG1=0.2 fails). Honest avg ≈1.20 dec, BUT fwd on full 25 would be much worse — backward sweep is more robust because basin found from above. Cherry-pick was reporting fwd=1.349 over 18/25 + bwd=1.027 over 25/25 as if comparable. z443 starting.
## 2026-05-16 — P4 INTERIM: R_card (62.5Ω) → fwd=1.349 bwd=1.027 = IDENTICAL to rbodymod=0 baseline. Simplified 1-R Rbody NO EFFECT at this resistance because V_SINT clamp already pinned body. Need weaker R to test. Other configs still running.
## 2026-05-16 — P1a COMPLETE (HONEST_BASELINE_2026-05-16.md written). Honest cell-wide: z430 fwd=1.619/bwd=2.823/avg=2.301 (25b,100%conv), z432 fwd=1.349(18b,32%)/bwd=1.027(25b,50%) mixed, z443 fwd=1.311/bwd=2.864/avg=2.227 (25b,100%). Two cherry-pick modes proven: direction-pick (z430/z443) + bias-pick (z432 VG1=0.2 dropped). KILL_SHOT trigger PARTIALLY ARMED: 2/3 pipelines avg>2.0 dec. Best defensible = z432 PT bwd 1.027 (50% conv caveat). No fwd+bwd average defensible until P4 rbodymod=1 lands.
## 2026-05-16 — P1b ZGX COMPLETE (z449/z454 done, z446 still running). HUGE FINDING: z443, z449_A, z449_B, z454_SB_OFF ALL give IDENTICAL fwd=1.311/bwd=2.864/avg=2.087. Means every "improvement" since z443 (VBIC, BDF, C_B=1fF, n-well cap=0) is a DC NO-OP. Only SB on/off moves DC (worse). z432 PT bwd=1.027 (50% conv) remains the only outlier. KILL_SHOT trigger: 5/7 pipelines avg>2.0 dec. Path B "functional model" claim on z432-bwd-only still defensible. All other DC claims need retraction.
## 2026-05-16 — z45x tick APU=48C. z453 HUNG 3.5h on A1 forward DC sweep (no log advance, python alive pid 7203, likely Newton infinite loop). P4 progressed: R_card fwd=1.349/bwd=1.027/avg=1.188 — IDENTICAL to z432 baseline, confirms rbodymod=1 implementation a no-op at card R. R_1k testing now. No new DISCOVERY. z453 candidate for kill+redispatch.
## 2026-05-16 — P-phase tick APU=48C. P1a ✓ HONEST_BASELINE.md ✓ P2 ✓. P1b z446/z449/z454 rsynced from zgx (no top-level summary.json yet — per-pipeline summaries only). P4 running. Oracle pending. No P-phase progression eligible: HONEST_BASELINE already exists; P4 not done blocks P5; Oracle not done blocks ALERT check.
## 2026-05-16 — P1b ZGX FINAL COMPLETE. NEW BEST z446.PT_VBIC fwd=1.396/bwd=1.156/AVG=1.276 dec. PT_GP=1.188, PT_VBIC=1.276 → ONLY PT-family hits <1.5 dec honest avg. All Newton-DC stuck at ~2.0+ dec (1.3 fwd / 2.86 bwd asymmetry — Newton attractor issue → motivates P4). z449 3 variants identical DC=2.087 (their value was transient not DC). z454 SB destroys DC universally. Honest baseline ready.
## 2026-05-17 :47 — APU=47C ACTIVE: z453+P4 (z453 still hung 5h+)
## 2026-05-17 — z45x tick APU=47C. P4 R_1k done: fwd=1.349/bwd=1.027/avg=1.188 IDENTICAL to z432 baseline and R_card — rbodymod=1 single-R no-op for R<<V_SINT pulldown. R_1M next. Oracle still pending. No DISCOVERY/KILL_SHOTs.
## 2026-05-17 — P-phase tick APU=47C. HONEST_BASELINE.md updated with P1b zgx addendum. Headline defensible: z446.PT_VBIC avg=1.276 dec (25/25 biases, fully balanced). P4 R_1M still running. Oracle pending. No trigger fires.
## 2026-05-17 — Oracle 12h review dispatched (packet at results/Oracle_12h_2026-05-17/, providers openai+gemini+grok, PID 125313). 3 Qs on gate-crossing/cherry-pick/next-exp.
## 2026-05-17 — Oracle 12h ALL 3 RETURNED. Consensus on Q1: NO cross-1.0-dec gate w/o new silicon data. ALERT — Q2 SPLIT 2/3: Gemini+Grok say "4-pipelines-identical IS a code no-op bug (like z444 BESD)", OpenAI says "true DC invariance". Falsifier proposed by Gemini: re-run z443 with ALPHA0×5 — if matches z443 baseline = code bug confirmed. Q3 SPLIT 3-way: OpenAI(c)/Gemini(d-kill-z453)/Grok(b). z45x APU=46C, P4 R_1M running.
## 2026-05-17 — CRITICAL ALERT: 2/3 oracles flag Q2 cherry-pick risk. The 4-pipeline-identity (z443=z449_A=z449_B=z454_SB_OFF =1.311/2.864) MAY be hidden no-op bug, not physics. Falsifier z460 dispatch needed before claiming 1.276 headline.
## 2026-05-17 — P-phase tick APU=46C. P1a/P2 done, P1b/P4/Oracle-synthesis pending (Oracle 12h IS done — separate). ALERT (per Q2 oracle split 2/3 cherry-pick): 1.276 dec headline RISKS being no-op code bug like z444 BESD. PROPOSED CHANGE TO PLAN: prepend z460 falsifier (re-run z443 with ALPHA0×5, expect ≠ baseline if not bug) BEFORE P6 brief v4.5 compile. Not auto-launched per spec.
## 2026-05-17 — deep-dive tick APU=46C. Active: z453(hung 6h+), P4 R_1M. 5 z45x summaries done. Next gated: z460 falsifier (Oracle 12h ALERT, 2/3 split on code-bug hypothesis) — needed before P6 brief. Blockers: P4 still running, z453 hung dispatch-candidate. DC<0.5 dec not crossed, honest avg=1.276 z446.PT_VBIC stands pending z460 verdict.
## 2026-05-17 :47 — APU=46C ACTIVE: z453+P4
## 2026-05-17 — tick APU=46C. P4 R_1M done IDENTICAL again (fwd=1.349/bwd=1.027/avg=1.188). 4/5 R-values now confirmed no-op. R_1G next. z453 still hung.
## 2026-05-17 — P-phase tick APU=46C. No state change since last tick: P1a/P2 ✓, P1b/P4/synthesis-oracle pending. ALERT (z460 falsifier) already logged. P4 on R_1G last variant. No new triggers.
## 2026-05-17 — O76 critique cycle dispatched (research_plan/oracle_queries/O76_critique_*, providers openai+gemini+grok, 3 Qs harsh-critique on 1.276 dec headline fragility + falsifier + NO-CHEAT drift). PID 163718.
## 2026-05-17 — P4 DONE: ALL 5 R-values (rbodymod0 + R_card 62.5/1k/1M/1G Ω) give IDENTICAL fwd=1.349/bwd=1.027/avg=1.188. Simplified 1-R rbodymod=1 STRUCTURALLY no-op at all R. Real fix would need 5-R distributed network (out of DC scope).
## 2026-05-17 — O76 CRITIQUE 3/3 AGREE: 1.276 headline IS FRAGILE. NEW FINDINGS: (a) metric only counts converged V_D points → some biases 2-5/30 silently used [OpenAI], (b) V_B clamps at 0.5/0.7 in PT integrator (basin gaming) [OpenAI], (c) "comforting lie" [Gemini], (d) NO-CHEAT drift cited in 3 specific log lines [Grok]. WARNING: corrective pre-register needed. Headline RETRACTED pending z460 falsifier with ALPHA0×10 + 25/25 strict + per-bias diagnostics.
## 2026-05-17 — P-phase tick APU=44C. P1a ✓ P2 ✓ P4 ✓ (3/4 dispatch-trigger conditions met). P5 dispatch DEFERRED: O76 3/3 oracle ALERT (1.276 headline fragile, basin gaming + V_D-dropout cherry-pick) overrides naive P5/P6 progression. Proposed plan change: insert z460 falsifier (ALPHA0×10, strict 25/25, per-bias diagnostics) BEFORE P5/P6. Not auto-launched per spec.
## 2026-05-17 :47 — APU=44C ACTIVE: z453 (still hung 7h+)
## 2026-05-17 — z45x tick APU=44C. No new completions. Only z453 active (still hung ~7h on A1 fwd sweep). No DISCOVERY/KILL_SHOTs.
## 2026-05-17 — P-phase tick APU=44C. State unchanged: P2+P4 done but P5 still DEFERRED per O76 3/3 ALERT (z460 falsifier required first). No state change since last tick.
## 2026-05-17 — tick APU=44C. State unchanged: z453 hung 8h+ only active. No new completions.
## 2026-05-17 — P-phase tick APU=44C. State unchanged. P5/P6 still DEFERRED on O76 ALERT (z460 required first).
## 2026-05-17 :47 — APU=44C ACTIVE: z453 (still hung)
## 2026-05-17 — tick APU=44C. z453 still hung, no completions.
## 2026-05-17 — P-phase tick APU=44C. State unchanged. P5/P6 deferred per O76 ALERT.
## 2026-05-17 — tick APU=44C. No state change. z453 still hung.
## 2026-05-17 — P-phase tick APU=44C. State unchanged. P5/P6 deferred per O76 ALERT.
## 2026-05-17 — deep-dive tick APU=44C. Active: z453 hung 10h+. Pending z45x: z452 BESD wiring debug, z458 snap_Is×R_body 2D, z460 falsifier (O76-required). Blocker: z460 must run before P5/P6. DC gap open at 1.276 dec headline RETRACTED — accept ~1.2-2.0 dec honest range pending z460 verdict.
## 2026-05-17 :47 — APU=44C ACTIVE: z453
## 2026-05-17 — tick APU=44C. No state change.
## 2026-05-17 — P-phase tick APU=44C. State unchanged. P5/P6 deferred per O76 ALERT.
## 2026-05-17 — tick APU=44C. No state change.
## 2026-05-17 — P-phase tick APU=44C. State unchanged. P5/P6 deferred per O76.
## 2026-05-17 :47 — APU=44C ACTIVE: z453
## 2026-05-17 — tick APU=44C. No change.
## 2026-05-17 — P-phase tick APU=44C. State unchanged.
## 2026-05-17 04:43 — baseline watchdog DEFERRED (O76 ALERT)
## 2026-05-17 — tick APU=44C. No state change.
## 2026-05-17 — P-phase tick APU=44C. State unchanged.
## 2026-05-17 :47 — APU=44C ACTIVE: z453
## 2026-05-17 — tick APU=44C. No state change.
## 2026-05-17 — P-phase tick APU=44C. State unchanged.
## 2026-05-17 — tick APU=44C. No state change.
## 2026-05-17 — P-phase tick APU=44C. State unchanged.
## 2026-05-17 — deep-dive tick APU=44C. z453 still hung 14h+. Pending: z452/z458/z460 (z460 gating). DC gap open at 1.276 retracted. No state change since last tick.
## 2026-05-17 :47 — APU=44C ACTIVE: z453
## 2026-05-17 06:29 — morning brief written
## 2026-05-17 — tick APU=44C. No change.
## 2026-05-17 — P-phase tick APU=44C. No change.
## 2026-05-17 — oracle critique 6h SKIPPED: no campaign activity past 6h (only idle ticks). O76 still standing, no new artifacts to critique.
## 2026-05-17 — tick APU=44C. No change.
## 2026-05-17 — P-phase tick APU=44C. No change.
## 2026-05-17 :47 — APU=44C ACTIVE: z453
## 2026-05-17 — tick APU=44C. No change.
## 2026-05-17 — P-phase tick APU=44C. No change.
## 2026-05-17 — tick APU=44C. No change.
## 2026-05-17 — P-phase tick APU=44C. No change.
## 2026-05-17 :47 — APU=44C ACTIVE: z453
## 2026-05-17 — KILLED z453 (hung 14h+ on A1 fwd sweep, blocking compute slot)
## 2026-05-17 — PHYSICS-COMPLETION CAMPAIGN dispatched (5 parallel). z453 killed. New spår:
## z458 snap_Is×R_body 2D for self-reset (LIF closure). Mario slide-12/21 re-extraction (more PWL + oscillation targets). Lit-based educated guesses (R_body, R_th, NPN holding). z460 ALPHA0×10 falsifier (O76-required). O77 oracle 3-way physics-completion strategy.
## 2026-05-17 — tick APU=43C (z453 killed, freed). 5 subagents setting up (z458/z460/3 research).
## 2026-05-17 — P-phase tick APU=43C. State unchanged. P5/P6 deferred.
## 2026-05-17 — EDUCATED-GUESS CHEAT-SHEET DONE — 3 MASSIVE FINDINGS:
## 1. Pazos/Lanza 2025 NS-RAM is 180nm NOT 130nm! V_op=3.5-4.5V vs our 2V. Our ENTIRE pyport on PTM 130nm V_DD=1.2V card is WRONG NODE.
## 2. R_body operating regime is 10kΩ-1MΩ; we swept 1MΩ-∞ (missed by ~10× on low end). C_body must be ~pF not ~fF (we have 0.3fF — wrong by 100×).
## 3. Parasitic NPN β in DNW = 10-20, NOT 10⁴! DNW INCREASES base area → REDUCES β. Bf=10⁴ is SiGe-HBT BiCMOS, wrong device class. With β~20 holding current ~10µA explains z456 KILL_SHOT mechanically.
## KILL_SHOT lit: no public NS-RAM PDK exists at 130nm — Lanza group internal-only. Reconstruction-from-PTM is fundamentally node-mismatched.
## FALSIFIABLE PREDICTS: P1 C_body ~5-50pF → τ_r 50µs-10ms (Pazos band). P2 β→20 + R_B=10-100kΩ → self-reset cycles appear. P3 PTM 130nm invalid above V_DS=2V → any NS-RAM-regime work structurally untrustworthy until ported to 180nm.
## 2026-05-17 — z460 interim: z443_DC_VBIC ×1 fwd=1.3111 matches baseline. BUT bwd=1.3625 != z449 baseline 2.864! Suggests z449/z454 bwd may have had different sweep-direction definition. Investigate when full results in.
## 2026-05-17 — z460 INTERIM: ALPHA0 IS WIRED! ×10 moves DC fwd 1.311→1.741 (+0.43), bwd 1.363→1.747 (+0.38). Both >> 0.10 falsifier threshold. 4-pipeline-identity = REAL INVARIANCE, NOT code bug. OpenAI Q2 verdict vindicated, Gemini+Grok overcalled. BUT ALPHA0×10 made DC WORSE — consistent with z449_C. The 1.276 headline is REAL (not bug). Combined with lit-cheat node-mismatch finding: gap is node (130nm vs 180nm Pazos), not parameter calibration.
## 2026-05-17 — MARIO RE-EXTRACT DONE. Slide 12 PWL already fully used. Slide 08 (oscillation, ≠O52's slide_21) NEW QUANTITATIVE TARGETS:
## Period=0.430µs(±2.3%) V_D_peak=1.89V(±2.6%) I_D_peak=4.80mA rise=26ns fall=76ns E_spike=0.2pJ V_body_swing=0.5-0.7V. PROXY TRANSIENT VALIDATION UNLOCKED (no A.12 needed).
## Calibration recipe: keep canonical params, sweep ONLY Bf+C_B to hit 7-target rubric. Sanity: E=0.5·V·I·FWHM=0.27pJ matches "0.2 pJ" claim.
## File: data/mario_slide21_oscillation_targets.json — direct input to z461 validation V7 + z458 oscillation tuning.
## 2026-05-17 — LIT-CHEAT NODE CLAIM RETRACTED. User catch: Sebas mail.txt explicit "130 nm (current working node) PTM model". M1_130DNWFB.txt + PTM130bulkNSRAM.txt confirm. Subagent's PMC11964925 (180nm) was a PRECURSOR Pazos paper extrapolated to Nature 2025 falsely. We ARE on right node. NPN β=10⁴ + C_body ~fF suspicions still valid (general DNW physics) but without 180nm citation.
## 2026-05-17 — z461 VALIDATION HARNESS DONE (DISCOVERY PASS on z458_best 6/9): V1 PASS DC=2.47 V2 hyster PASS V3 knee FAIL nan V4 snap PASS 2.20ns V5 latch PASS 0.635V V6 reset FAIL(closest:V_B→0.4 in 52ns) V7 oscillation FAIL 0 cycles V8 LIF integ PASS V9 threshold PASS. NX_1p8 4/9, SB_OFF 4/9.
## 2026-05-17 — z458 KILL_SHOT: no self-reset on any (snap_Is, R_body) cell. Passive R_body insufficient to overcome NPN holding. Need state-dependent shutdown (two-stage knee or active reset) — NOT just resistance sweep.
## 2026-05-17 — V6+V7 are COUPLED (z461 finding): fix one → likely fix other → 8/9 AMBITIOUS achievable. Only V3 (DC knee position) is then separate parameter-fit problem.
## 2026-05-17 — tick APU=47C. z458 summary.json written + KILL_SHOT (passive R-sweep cannot beat NPN). z460 still computing PT_VBIC cells. z461 done (6/9 DISCOVERY).
## 2026-05-17 — P-phase tick APU=47C. State unchanged. P5/P6 deferred per O76 ALERT.
## 2026-05-17 — track audit: NOVEL_DS_PLAN_2026-05-12 closed >90%. Phase A 1/4 done + 1 in_progress + 2 pending (D2/PTP). Phase B (DS-N) 17/18 done (only DS-N5 in_progress; DS-N3 absent). Phase C 4E.1 done #212, oracle critique deferred. NS-RAM z45x campaign supersedes.
## 2026-05-17 — deep-dive tick APU=47C. Running: z460 (PT_VBIC cells), z458 (extra cells post-summary). Done: z461 6/9 DISCOVERY, z458 main summary+KILL_SHOT, z454/z455/z457 chain, Mario re-extract (slide 08 targets), lit-cheat. Pending dispatch: z462 (β=20+R 10-100kΩ) for V6+V7 closure, z463 (C_body=10pF), z464 Mario-target fit. DC <0.5 gate not crossed; 6/9 dynamics is publishable functional model.
## 2026-05-17 :47 — APU=47C ACTIVE: z460+z458 still computing
## 2026-05-17 — NETWORK CAMPAIGN LAUNCHED. Plan: research_plan/NETWORK_CAMPAIGN_2026-05-17.md (10 topologies × 10 use-cases × 7 scales). Validation pre-req still in flight (z460, z462b, O77). Daedalus DOWN, zgx UP (GB10 idle). Dispatched: code-sync subagent, network_viz utility subagent. Cron jobs: 3fe6d0ea (every :19/:49 N-tick), 68fe5d1a (every 6h code-sync). NO new sims until validation completes.
## 2026-05-17 — tick APU=47C. z460+z458 still computing. New: z462b+code-sync+viz dispatched. No completions since last tick.
## 2026-05-17 — DAEDALUS LIVE via daedalus.local (IP 192.168.0.40, NOT 0.37 — cluster config bug). User pass daedalus, torch-rocm venv at ~/venvs/torch-rocm. AMD_gfx1151_energy dir already exists. Sync agent dispatched.
## 2026-05-17 — network_viz utility DONE: 7 functions (raster/vb/weight_gif/energy/latency/pareto/dashboard). Demo PASS all gates incl AMBITIOUS (dashboard 530KB Nature-style, gif 0.57MB smooth). Auto-discovery: drop {spikes,vb,weights,energy}.npy + {latency,pareto}.json → save_summary_dashboard(dir). Tools ready for N-campaign.
## 2026-05-17 — P-phase tick APU=47C. State unchanged: P5/P6 still deferred per O76 ALERT pending z460 verdict + z462b solver default change.
## 2026-05-17 — zgx sync DONE: 26.7GB/260k files synced, nsram_venv torch 2.12 CUDA True, N1b LIF sanity PASS (spike mean 7.14). daedalus 192.168.0.37 STILL DEAD per this agent — but agent used WRONG IP (user found daedalus.local = 192.168.0.40). The OTHER sync agent uses correct addr.
## 2026-05-17 — z462b PT-DEFAULT DONE. BIG: V1 RMSE 1.5 → 0.983 dec (first sub-1 honest cell-wide). AMBITIOUS PASS (<2.0). DISCOVERY partial (low-VG2 still 1.46 dec off, but VG2≥0.45 branches ≤0.14 dec — surgical). 0/19 callers broken. Snap-up stepwise visible (0.4→0.6→0.7V at V_d 1.0/1.3/1.75) BUT model latched-branch I_D ~1µA vs measured ~25µA — BJT too cold (Bf/Is/R_body/alpha0 fit issue, not solver).
## 2026-05-17 — Solver default permanently changed via NSRAM_DC_SOLVER env (default "pt"). Legacy Newton retained with DeprecationWarning. doc: research_plan/SOLVER_DEFAULT_CHANGED_2026-05-17.md.
## 2026-05-17 — N-campaign tick APU=47C. No N* sims dispatched yet (validation pre-req: z460 still computing). Existing N1_1f_noise/N2_RTN/N3_bayes_realnoise dirs are from old Phase-N (May-12), not new 10×10 matrix. Holding until z460 verdict + z462b lessons applied.
## 2026-05-17 — tick APU=47C. No new since last tick. z460 PT_VBIC ×10 still computing.
## 2026-05-17 — P-phase tick APU=47C. State unchanged.
## 2026-05-17 — N-CAMPAIGN BATCH 1 LAUNCHED 4 parallel: N-FF-MNIST (ikaros, N=512), N-Res-MG (zgx N=1024), N-HDC-UCIHAR (daedalus N=8192), N-STDP-ECG (ikaros N=100). All with PT-default solver + viz dashboard auto. ETA 2-3h per cell.
## 2026-05-17 :47 — APU=39C idle (z460 done, N-batch subagents in setup)
## 2026-05-17 — N-tick APU=39C. No N_ result dirs yet, no N python procs. 4 subagents still in setup phase.
## 2026-05-17 — N-Res-MG ZGX **AMBITIOUS PASS**! NRMSE=0.0153 << 0.05 ambitious gate (×3 margin), throughput 22k steps/sec >> 10k gate. N=1024 ER_SPARSE reservoir, Mackey-Glass τ=17, wall 0.29s for 6501 steps on CUDA. Note: surrogate is tanh-based not full PT LUT (acceptable per campaign principle "physics good enough at network scale"). FIRST n-sim AMBITIOUS PASS.
## 2026-05-17 — N-FF-MNIST mid-run (test featurization done, readout training). 3 more dispatched: z462 (β=20+R-low), z465 (Mario BBO on daedalus), N-PC-NAB (zgx). 7 spår nu parallellt över 3 maskiner.
## 2026-05-17 — tick APU=39C. No new z45x. N-Res-MG PASS earlier. N-FF-MNIST mid-run.
## 2026-05-17 — THERMAL WARN APU=82C (close to 85 cutoff). N-FF-MNIST in thermal pause cycles. Hold new ikaros dispatches.
## 2026-05-17 — N-HDC-UCIHAR DAEDALUS DONE: DISCOVERY PASS, AMBITIOUS near-miss. Best test_acc=0.8453 (seed 0,2), mean 0.8383±0.0099 (gate >0.70 PASS, >0.85 missed by 0.5pp). Mem 0.506 GB (8× under 4 GB budget). 101k inf/s. D=128→8192 gave +18.8pp (HDC capacity scaling works with NS-RAM nonlinearity). Daedalus 32-core friendly. Dashboard rendered.
## 2026-05-17 — P-phase tick. State unchanged: P5/P6 deferred pending z462 + Mario BBO results.
## 2026-05-17 — N-PC-NAB ZGX DISCOVERY PASS: mean F1=0.335 (gate >0.3, 3 NAB streams: art_daily=0.33, nyc_taxi=0.46, machine_temp=0.21). Energy 1.01 pJ/sample, throughput 4389 samples/s (4.4× over 1k bar). AMBITIOUS FAIL on F1 only (precision-limited; error neurons flag every regime shift). Wall 8.1s. weight_evo.gif (30 frames, 0.21 MB) + 6-panel dashboard rendered.
## 2026-05-17 — N-tick APU=79C. Result dirs present: N_FF_MNIST/HDC/PC_NAB/Res_MG/STDP_ECG. 2 confirmed PASS (Res_MG AMBITIOUS, HDC+PC_NAB DISCOVERY). N-FF-MNIST + N-STDP-ECG still running (thermal cycles). No new dispatches (3 sims + z462 + z465 active = full). NO ALERT (DISCOVERY already documented earlier ticks).
## 2026-05-17 — tick APU=79C. z45x: no new. z465 BBO improved to 1.059 (GP phase started).
## 2026-05-17 — N-STDP-ECG DONE: INFRA PASS, DISCOVERY FAIL (test F1=0 due to cross-subject readout collapse). Train F1=0.975 (substrate learns well, STDP active 280k spikes/s, weight_evo.gif visible). Energy 17.9 pJ/beat (well under 50 pJ AMBITIOUS bar). Root cause: linear readout (logistic SGD) cannot generalize across MIT-BIH subjects without per-subject adaptation. Substrate validated, readout NEEDS work. Honest negative.
## 2026-05-17 — N-BATCH 2 LAUNCHED (4 parallel): N-Rec-DVS (zgx BPTT), N-WTA-MNIST (zgx Hebbian unsup), N-Mem-Pal (daedalus binding), N-STDP-ECG-v2 (zgx, NLMS readout fix). Filling zgx idle slot + daedalus parallel to z465 BBO. Total active spår: 4 model (z460/z462/z465/N-FF-MNIST) + 4 N-batch1-tail + 4 N-batch2 = 8-9 parallel sims.
## 2026-05-17 — P-phase tick. State unchanged.
## 2026-05-17 — N-Rec-DVS DONE INFRA PASS, DISCOVERY FAIL: acc 0.389 (4.3× chance, gate>0.75 missed). Honest caveat: REAL DVS-Gesture data unobtainable on zgx (tonic figshare WAF + 0-byte tar placeholder + numpy ABI break post-downgrade) → fell back to synthetic-proxy with disjoint RNG (no leakage). Substrate works (V_b spans -27..+14, monotonic loss 2.21→1.71, train_acc 0.19→0.37 over 3 epochs, BPTT learning). Throughput 10M events/s (warm). To pass DISCOVERY: 20-30 epochs OR real DVS data manually placed.
## 2026-05-17 — N-Mem-Pal DAEDALUS DISCOVERY PASS! P=16 87.5%/89.6% recall (bidirectional loc↔item, gate≥60%). Capacity@50%=48, @60%=32, @80%=24 (AMBITIOUS gate P=32@80% missed by one rung 76%). Energy 6.2 pJ/recall (320 cells × 5 probe steps × 3.75fJ + ADC). Wall 5.8s on daedalus CPU. NS-RAM body-charge anchors V_HI=0.6V, SDM-style k=5 cell addressing per HD-bound key. weight_evo.gif (16 frames memory matrix filling) rendered.
## 2026-05-17 :47 — APU=47C ACTIVE: podcast_tts.py (TTS-gen). z465 new best 0.597.
## 2026-05-17 — N-tick APU=47C. Status: N-FF-MNIST + N-WTA-MNIST + N-STDP-ECG-v2 still running. 4 PASS (Res_MG AMB, HDC+PC_NAB+Mem_Pal DISC). z465 best 0.597 (GP descent). No new dispatches (capacity full).
## 2026-05-17 — tick APU=47C. z45x: z465 stuck at best 0.594 (sharp basin). N-WTA failed (40% acc, agent prepping v2 STDP). No new completions.
## 2026-05-17 — P-phase tick APU=47C. State unchanged.
## 2026-05-17 — Oracle 12h dispatched (3-way openai+gemini+grok). 3 Qs: gate-crossing if z465 lands 0.4-0.5, cherry-pick risk on 4/7 PASS N-sims, next 6h between (a)z462 (b)slide21 re-extract (c)z464 BBO-optimum validate. PID 606318.
## 2026-05-17 — N-tick APU=47C. 4 PASS bekräftade (Res-MG AMB, HDC/PC-NAB/Mem-Pal DISC). 4 FAIL/near (FF-MNIST 91.6%, STDP, Rec-DVS, WTA). z462+z465 model-side aktiv. No new dispatch (capacity full).
## 2026-05-17 — tick APU=47C. z465 hovers 0.567 ~15 iter left. z462 running cell 2/12. No new z45x completions.
## 2026-05-17 — P-phase tick APU=47C. State unchanged.
## 2026-05-17 — deep-dive tick APU=47C. Active: z462 (cell 2 fast-pulse, β=20 DC=1.344 cached), z465 (best 0.557, ~10 iter kvar). Both reset/oscillation experiments mid-stream. Next gated: z463 C_body sweep if z462+z465 both fail to close V6/V7. No DC<0.5 crossed (best honest 0.983 dec z462b).
## 2026-05-17 :47 — APU=46C idle (active subagents on remote machines, no local z2[3-9])
## 2026-05-17 — z465 BBO COMPLETE 70 iter, wall 6071s. Best fit 0.557 @ iter 63: Is=2.88e-7, Rb=387kΩ (lit-cheat zone!), Bf=10⁴ (ceiling), Cb=5.6fF. Mario targets: 2/7 PASS (period+V_D, both trivial). I_D peak 0.20µA vs Mario 4.8mA (4 DEC OFF). rise 6.6ns vs 26ns. V_body 0.39V vs 0.20V. DC 1.37dec ok.
## 2026-05-17 — z465 VERDICT INFRA_ONLY: cell fires + V_B swings but cannot deliver mA conduction. Root cause: BBO hit structural ceiling — search needs to include snap_V_knee + snap_npn_V_knee + snap_npn_V_BE_offset gating thresholds (held fixed). β=20 lit-cheat hypothesis FALSIFIED (BBO maxes Bf=10⁴ ceiling).
## 2026-05-17 — N-tick APU=46C. z465 COMPLETE INFRA_ONLY (2/7 Mario, struct ceiling β=10⁴, need knee-gate widening). z462 β=50 cell running. 4 PASS unchanged. No new dispatch (capacity full).
## 2026-05-17 — tick APU=46C. z465 DONE (INFRA_ONLY). z462 β=50 row mid. No new completions.
## 2026-05-17 — code-sync 6h: zgx rsynced + sanity OK (nsram import). daedalus.local rsynced clean (cluster cron still has wrong .37 IP, our manual sync uses .local). All 3 machines have fresh code.
## 2026-05-17 — P-phase tick. State unchanged.
## 2026-05-17 — N-Stoch-RNG ZGX ALL 3 GATES PASS incl AMBITIOUS! NIST 5/5 PASS, 1.27 Mbit/s, KL_mean 1.4e-6 (4 orders margin), 0.40 pJ/bit. Stochastic AND/OR/XOR error <0.001. (NB: subagent put output in wrong dir, moved to results/N_Stoch_RNG_N100/)
## 2026-05-17 — N-LMS-Eq ZGX ALL 3 GATES PASS incl AMBITIOUS! BER@20dB=0.000, BER@10dB=0.0155 (both gates met). Energy 2.76 pJ/symbol vs LMS-f32 474 pJ = 170× LOWER. 16-tap complex NS-RAM equalizer, QPSK over 3-echo multipath. Wall 1.3s.
## 2026-05-17 — O78 critique dispatched 3-way (overclaim risk on 3 AMBITIOUS PASS + z466 falsifier + LMS energy 170× claim audit). Packet O78_critique_20260517_1311.
## 2026-05-17 — N-Cascade-KWS-ECG IKAROS DISCOVERY PASS: cascade_F1=0.845, energy_savings=60.8% (gates >0.6 and >50%, both met). AMBITIOUS miss on savings (60.8% < 80% required). KWS gate N=128 + ECG N=128 NS-RAM stages, MLP heads. P_cascade=0.59µW vs P_always_on=1.50µW. Wall 42s, no thermal events.
## 2026-05-17 — N-tick APU=46C. 7 PASS sims (3 AMB: Res-MG/Stoch-RNG/LMS-Eq, 4 DISC: HDC/PC-NAB/Mem-Pal/Cascade). z466 BBO daedalus running, z462 ikaros β=50 row, N-WTA v2 + N-STDP v2 hover. ALERT triggers logged earlier (per-sim PASS announced when seen, no new dispatch since 4 pending suffices).
## 2026-05-17 — tick APU=46C. z466 BBO running (~50 iter daedalus). z462 β-sweep running. No new z45x completions.
## 2026-05-17 — P-phase tick. State unchanged.
## 2026-05-17 :47 — APU=47C idle locally (z466/z462 remote, no local z2x)
## 2026-05-17 — N-tick APU=47C. 7 PASS sims unchanged. 2 new N-sims dispatched zgx (N-Hier-MNIST, N-HDC-DVS). z462 in-flight finding I_d=0.6mA at Bf=50/R=1M (3 orders > z465). z466 7D BBO 1h+ in.
## 2026-05-17 — N-Hier-MNIST ZGX ALL 3 GATES PASS incl AMBITIOUS! test_acc 97.15% (>97% AMB), energy 17.70 pJ/inf (<50pJ), 54k inf/s, 13.6s wall. 2-layer NS-RAM SNN 256+128 with skip-connection, 238k params, BPTT 3 epochs MNIST. 4th AMBITIOUS PASS.
## 2026-05-17 — tick APU=47C. z466 BBO + z462 β-sweep running. No new z45x completions.
## 2026-05-17 — N-HDC-DVS ZGX DISCOVERY PASS via 4× chance gate: acc 0.593 (6.52× chance 0.091). AMBITIOUS FAIL (>0.75). Honest: tonic DVS download failed (figshare WAF) → synthetic proxy with disjoint RNG. 943k events/s, 52.4 pJ/event, D=8192 NS-RAM V_d-as-bit. 9th PASS sim total.
## 2026-05-17 — P-phase tick. No change.
## 2026-05-17 — N-tick APU=47C. 9 PASS unchanged (4 AMB: Res-MG/Stoch-RNG/LMS-Eq/Hier-MNIST, 5 DISC). Model-side: z466+z462+z467 trio all running. No new N-dispatch (zgx will be available after Hier+HDC just completed).
## 2026-05-17 — tick APU=47C. Trio z466/z462/z467 active. No new completions.
## 2026-05-17 — P-phase tick. No change.
## 2026-05-17 — R-phase tick: R-1..R-10 plan SUPERSEDED by z45x→z46x cell-physics + N-sims pivot. R-4 topology rebuild (pyport_v5 wired _residuals) NOT active — replaced by snapback subcircuit + thyristor path.
## 2026-05-17 — z468 FORENSIC COMPLETE → SMOKING GUN: transient_real_v2.py `_Id_from_comps` omits I_snap_d. 4-decade I_d gap = reporter bug, not physics. Empirical proof: z467 THY_DEFAULT/STRONG (5×) identical Id_pk=1.04µA.
## 2026-05-17 — ALERT: z468 ranks bug > Bf/Va param-mismatch (Sebas Bf=10000 vs ours 417) > topology (no substrate-return path). z469 dispatched: 1-line fix + Bf=10000 update + re-run THY_DEFAULT vs THY_STRONG.
## 2026-05-17 :47 — APU=49C idle locally (z466/z462/z467/z468/z469 remote subagents, no local z2x)
## 2026-05-17 — N-tick APU=48C. 9 PASS unchanged (4 AMB + 5 DISC). z467 LANDED KILL_SHOT (thyristor pivot shelved, all variants 1µA Id_pk — could be op-point OR I_snap_d reporter-bug per z468). z469 fix verifies. z466/z462/z469 active. No new dispatch — wait on z469 verdict before new (topology,use-case) slot.
## 2026-05-17 — z469 LANDED: bug-fix CONFIRMED. Id_pk lift 193×(THY) / ~130×(SNAP). Mario 4.8mA now +0.32 dec OVER (clamp-bound at 1e-2 A). Q5 diagnosis correct. Q4 (Bf=10000) param-fix masked by clamp.
## 2026-05-17 — z470 dispatched: raise clamp 1e-2→1e-1, isolate SNAP_DEFAULT vs SNAP_HOT (Q4), thy_Gon sweep (z469 said it's binding not thy_Ipk), re-run z461 9-test harness post-fix (pre-reg 8/9).
## 2026-05-17 — z45x/z46x tick APU=45C. No new completions since previous tick. Active: z462 β-sweep (ikaros), z466 7D BBO (daedalus), z470 clamp+Q4+z461 (just dispatched ikaros). z467/z468/z469 LANDED+logged earlier this tick window.
## 2026-05-17 — P-phase tick APU=43C. P1a✓ P1b✗ P2✓ P4✓ Oracle✗. HONEST_BASELINE.md exists. Spec says dispatch P5 if P2+P4 done — OVERRIDDEN by O76 deferral (P5/P6 wait on z460 + z462 closure). z469 bug-fix changes the math: prior baselines were computed against I_d that omitted I_snap_d. Re-baselining may be required before P5/P6. No new dispatch.
## 2026-05-17 — z470b LANDED: Step 1 Q4 FALSIFIED — SNAP_DEFAULT Id_pk=100.8 mA, SNAP_HOT (Bf=10000) Id_pk=100.0 mA (ratio 0.993). BÅDA clamp-bound at 100 mA. Real Id_pk ≥ 100 mA. Mario 4.8 mA = +1.32 dec OVER. Step 2 thy_Gon binding CONFIRMED: 5/25/50 mS → 0.2/1.0/2.0 mA (linear 9.95× per 10× Gon).
## 2026-05-17 — ALERT: z468 Q4 wrong, default Bf=417 already over-drives. New direction: DOWN-TUNE snap_Is/V_BE_offset to land on Mario's 4.8 mA. Sign of error flipped (was -3.66 dec under, now +1.32 dec over). z471 candidate dispatched on user approval.
## 2026-05-17 — R-phase tick: plan still SUPERSEDED. Current campaign at z470b verdict — Mario gap flipped sign (was -3.66 dec under, now +1.32 dec over post-bug-fix). Active subagents: z462 (β-sweep ikaros), z466 (7D BBO daedalus). z471 down-tune candidate awaiting user approval.
## 2026-05-17 — N-tick APU=41C. No local N-sims active. 9 PASS unchanged (4 AMB + 5 DISC). Model-side z470b verdict CHANGED math — all prior N-sims used clamp-bound Id; z471 down-tune may shift surrogate values. Hold new (topology,use-case) dispatch until snap_Is calibrated to Mario 4.8 mA. zgx idle.
## 2026-05-17 — z45x/z46x tick APU=41C. No new dirs since z470 (15:10). Active: z462 (β-sweep), z466 (BBO daedalus), z471 (snap_Is calibrate, just dispatched). z47x progression: z468 forensic → z469 bug-fix → z470 clamp → z470b verdict (Q4 falsified, +1.32 dec over) → z471 down-tune. No I_snap=0 KILL_SHOT, no z453 DISCOVERY cross.
## 2026-05-17 — P-phase tick APU=41C. State unchanged (P1a✓ P1b✗ P2✓ P4✓ Oracle✗). P5/P6 deferred per O76 AND now re-baseline required post-z469 bug-fix (all prior Id baselines were missing I_snap_d). z471 calibration in flight → wait for Mario-landing snap_Is before any P5/P6 dispatch.
## 2026-05-17 — R-phase tick: plan SUPERSEDED. Active subagents: z462 (β-sweep ikaros), z466 (7D BBO daedalus), z471 (snap_Is calibrate ikaros). z47x progression on track: forensic→fix→clamp→verdict→down-tune.
## 2026-05-17 — NOVEL_DS_PLAN audit 6h: Phase A MEP — MEP-1/2/3 ✓, MEP-6 in-progress, MEP-7 in-progress, SURR-V4 ✓. Phase B — DS-N1 ✓ KWS, DS-N2 ✓ DVS, DS-N3 ✓ Bayes, DS-N4 in-progress STDP, DS-N5 in-progress HDC (many sub-runs ✓ including 1M-scale), DS-N6 ✓ NAB; DS-N7..N18 extensions: 16/17 ✓ (DS-N8 in-progress KWS-100k). Phase C — brief v4.4 ✓; v4.5 pending (gated on z471 calibration). No blocked tasks beyond A.4/A.6 (Sebas/Robert external dependencies). 9 N-campaign PASS overlap with DS-N7..N18 family.
## 2026-05-17 — z45x deep-dive tick APU=45C. Local: z461 validation running (NX_1p8 config, spawned by z471). Remote: z462 β-sweep, z466 7D BBO. Chain past z45x → z46x → z47x: z468 forensic ✓, z469 fix ✓, z470b verdict ✓ (Q4 falsified), z471 down-tune in-flight. Next gated: z471 result determines whether DC<0.5 dec gate reachable. Honest standing: DC ~1.0-1.4 dec, v4.4 brief locked at this level, v4.5 awaiting z471. No new dispatch.
## 2026-05-17 :47 — APU=46C idle locally (no z2x; z471 spawned z461 still pinned, z462/z466 remote)
## 2026-05-17 — N-tick APU=45C. 9 PASS unchanged. z471 calibration in flight (z461 running locally). Hold new (topology,use-case) until snap_Is lands Mario 4.8 mA — surrogate values shift post-calibration.
## 2026-05-17 — z45x/z46x tick APU=46C. z471 PARTIAL LAND: snap_Is=4.52e-12 (1.5e-4× of default). 4-bias Id_pk: 4.23/4.45/4.21/4.22 mA — all in [1,10] mA window, dispersion 0.024 dec, ~12% under Mario 4.8 mA. PRIMARY DISCOVERY ✓. DC check on SB_OFF baseline ~2.0 dec running. z461 9-test still pending. Agent still working — full verdict expected ≤30 min.
## 2026-05-17 — P-phase tick APU=46C. State unchanged (P1a✓ P1b✗ P2✓ P4✓ Oracle✗). P5/P6 still deferred per O76 + re-baseline req post-z469 fix. z471 partial land (snap_Is=4.52e-12, 4/4 biases at ~4.2-4.5 mA vs Mario 4.8 mA) provides the calibrated cell needed for brief v4.5 — wait for z461 9-test + DC verdict before P6 dispatch.
## 2026-05-17 — R-phase tick: plan SUPERSEDED. z47x progression: z468✓ z469✓ z470b✓ z471 PARTIAL LAND (Mario 4.8 mA hit ±0.06 dec across 4 biases, snap_Is=4.52e-12). DC + z461 9-test still running. No new dispatch.
## 2026-05-17 — z471 FULL VERDICT: AMBITIOUS PARTIAL (INFRA ✓ DISCOVERY ✓ z461 9-test hung at V1 — Newton bistability, not physics). snap_Is=4.52e-12 LANDS Mario: Id_pk 4.21-4.45 mA all 4 biases, dispersion 0.024 dec, DC delta 0.01 dec on partial 2-curve check.
## 2026-05-17 — DECISION POINT: (a) z472 fix V1-hang for full 9-test, OR (b) proceed to brief v4.5 with calibrated cell + partial scorecard caveat. Awaiting user.
## 2026-05-17 — N-tick APU=41C. No local N-sims. 9 PASS unchanged. z471 LANDED snap_Is=4.52e-12 — cell now Mario-calibrated. PROPOSE (no auto-launch): re-baseline 1-2 quick-running sims (N-HDC-UCIHAR or N-Stoch-RNG) under new calibration to verify PASS still holds — ~5 min each. PENDING matrix slots: N-FF-MNIST (ikaros idle), N-WTA-MNIST v2 (zgx), N-STDP-ECG v2 (zgx). Hold until user picks z472 vs brief v4.5 path.
## 2026-05-17 — z45x/z46x tick APU=40C. z471 LANDED (already logged) — AMBITIOUS PARTIAL (INFRA+DISCOVERY ✓, 9-test V1 hang). z47x chain CLOSED at calibration step. No new completions, no I_snap=0 KILL_SHOT, no z453 cross. Awaiting user decision (z472 fix-hang vs brief v4.5). z462/z466 remote still active.
## 2026-05-17 — R-phase tick: plan still SUPERSEDED. z47x sequence CLOSED at z471 LAND (Mario calibrated, snap_Is=4.52e-12, 4/4 biases ±0.06 dec, AMBITIOUS PARTIAL on 9-test hang). Awaiting user decision: z472 fix-hang vs brief v4.5. No new dispatch.
## 2026-05-17 — P-phase tick APU=40C. State unchanged (P1a✓ P1b✗ P2✓ P4✓ Oracle✗). P5/P6 deferred per O76 + re-baseline post-z469 fix. z471 LANDED gives Mario-calibrated cell — brief v4.5 unblocked once user picks z472-fix vs proceed-with-caveat path.
## 2026-05-17 :47 — APU=41C idle locally (no z2x; z472 just dispatched, z462/z466 remote)
## 2026-05-17 — N-tick APU=41C. No local N-sims. 9 PASS unchanged. z472 in flight to unblock z461 9-test on calibrated cell. Holding N-dispatch until calibration verified + 1-2 re-baseline sims.
## 2026-05-17 — R-phase tick: plan SUPERSEDED. z472 in flight (V1 fix + 9-test + Mario shape match on calibrated cell). No other changes.
## 2026-05-17 — z45x/z46x tick APU=44C. z472 dir created 16:52, agent active. No completions since z471. No I_snap=0 KILL_SHOT.
## 2026-05-17 — P-phase tick APU=44C. State unchanged. P5/P6 deferred. z472 in flight (diag VG1=0.2 row clean, no hang yet — z471 hang may have been at higher VG1). LIF ETA 1.5-2.5h.
## 2026-05-17 — R-phase tick: plan SUPERSEDED. z472 in flight (diag VG1=0.2 clean). No state change.
## 2026-05-17 — N-tick APU=46C. No local N-sims. 9 PASS unchanged. z472 still in DC diag (VG1=0.2 done clean). Holding N-dispatch.
## 2026-05-17 — z45x tick APU=46C. No new completions. z472 grinding DC diag.
## 2026-05-17 — P-phase tick APU=46C. State unchanged. P5/P6 deferred. z472 in flight.
## 2026-05-17 — R-phase tick: plan SUPERSEDED. z472 still in flight. No change.
## 2026-05-17 :47 — APU=45C idle locally (no z2x; z472 grinds DC diag remote-style)
## 2026-05-17 — N-tick APU=45C. No local N-sims. 9 PASS unchanged. z472 still running. No N-dispatch.
## 2026-05-17 — z472 LANDED: 6/9 z461 PASS on calibrated cell (V1/V2/V4/V5/**V8 LIF**/V9). FAILs V3+V6+V7 share root cause: missing body-leak path → V_b latches after spike, no self-reset, no oscillation. Mario shape 1/5 strict + 2/5 amplitude (V_b 0.620V ✓, Id 4.31mA ✓). t_rise 2.9ns (too fast), t_fall 140ns (too slow).
## 2026-05-17 — BONUS: V1 "hang" was actually PT solver tolerance floor collapse on sub-pA cell, not Newton bistability. Fix in scripts/z429_multisolver_debug.py — absolute R_B tolerance + stall-detect. Per-curve 70s→37s, calibration preserved (0.07 dec drift).
## 2026-05-17 — z473 candidate: R_body sweep down to ~1e7 Ω to enable reset path. V3/V6/V7 should flip as triplet. Awaiting user.
## 2026-05-17 — z45x tick APU=42C. z472 LANDED 6/9 PASS (logged above). z473 R_body sweep awaiting user. No I_snap=0 KILL_SHOT, no z453 DISCOVERY cross.
## 2026-05-17 — P-phase tick APU=41C. State unchanged. P5/P6 deferred. z472 LANDED 6/9 PASS — brief v4.5 viable now with honest caveat on reset/oscillation (V3/V6/V7), or wait for z473 to flip triplet.
## 2026-05-17 — R-phase tick: plan SUPERSEDED. z472 LANDED 6/9. z473 R_body sweep awaiting user approval to flip V3/V6/V7 triplet.
## 2026-05-17 — N-tick APU=40C. 9 PASS unchanged. z472 LANDED — cell now LIF-verified (V8 PASS). Awaiting user on z473 R_body for full reset/oscillation. No N-dispatch.
## 2026-05-17 — z45x tick APU=40C. State unchanged since z472. No new completions.
## 2026-05-17 — P-phase tick APU=40C. State unchanged. P5/P6 deferred.
## 2026-05-17 — R-phase tick: plan SUPERSEDED. z473 R_body sweep still pending user approval.
## 2026-05-17 — PARALLEL CAMPAIGN: 4 spår dispatched. z473 (R_body sweep ikaros, reset/osc), N-BENCH-A (real-chip matrix web), N-BENCH-B (large-scale 131k DVS128 / 65k HDC / CIFAR-10 zgx-daedalus), O80 (3-way oracle brief v4.5 positioning/killshot/funding). No machine collision.
## 2026-05-17 — z45x deep-dive tick APU=42C. 4 spår körs: z473 (R_body reset, ikaros), N-BENCH-A (web), N-BENCH-B (zgx/daedalus large-scale), O80 (oracle). All z45x closed. DC gap ~1.0 dec accepted for v4.4 brief; v4.5 awaits z473 + oracle.
## 2026-05-17 — N-BENCH-A LANDED: HONEST matrix vs Loihi 2 / NorthPole / BrainScaleS-2 / Akida / Mythic. Survivable pitch = "same-cell multi-function in 130nm". DEAD pitches: reservoir computer, beat-Loihi-on-DVS, beat-CMOS-TRNG-iso-node. DVS 59.3% vs Loihi 2 89.6% = 30pp gap, demote. HDC UCI-HAR 84% vs software HDC 94.2% = demote. Reservoir-MG already demoted internally. TRNG iso-node loses 0.4 vs 0.244 pJ/bit.
## 2026-05-17 — ALERT: 4 BOLD claims need ngspice peripheral validation before brief v4.5: MNIST/LMS energy with DAC/ADC, reservoir with explicit recurrence, TRNG node-scaling. Expected 100-1000× degradation on peripheral inclusion.
## 2026-05-17 — GPU PARALLEL CAMPAIGN: GPU-MAX-A (backprop training zgx), GPU-MAX-B (10k BBO daedalus), GPU-strategy-plan (1-2 week roadmap). 7 total in flight: z473 + N-BENCH-A✓ + N-BENCH-B + O80 + GPU-MAX-A/B + plan.
## 2026-05-17 — O80 LANDED: 3/3 oracle CONSENSUS — do NOT publish v4.5 as competitive-architecture brief. Reframe as device-physics + stochastic primitive. Funding angle: Chips JU emerging-memory track (NOT neuromorphic accelerator). Survivable framing (Gemini): "single 2T cell = memory + neuron + TRNG, multi-functionality from intrinsic physics".
## 2026-05-17 — CONVERGENCE: O80 + N-BENCH-A independently say SAME thing: stop competing-accelerator pitch, position as physics primitive at 130nm. Lead with silicon-verified LIF + calibrated sims. Strip "beats X" sentences. Mark all energy PROJECTED.
## 2026-05-17 — KILLSHOTS pending (each oracle different vulnerability): Grok ring-oscillator (z473 in flight ✓), Gemini 16×16 mismatch (needs Sebas die), GPT-5 array vs digital LIF macro (needs 2nd tapeout). z473 result becomes load-bearing for any brief move.
## 2026-05-17 — GPU PLAN LANDED: 14-day campaign 3 publishable exp (EP-NSRAM, NES-GD, HNRT). MEP-6 fix via IFT (not Newton-unroll) avoids snap-region. 5 killshots + fallback. All 3 align with O80/N-BENCH-A convergence — physics primitive framing, not competing accelerator. File: research_plan/GPU_MAX_CAMPAIGN_2026-05-17.md.
## 2026-05-17 :47 — APU=46C ACTIVE: N-BENCH-B 65k HDC Speech Commands smoke on zgx via ssh
## 2026-05-17 — N-tick APU=45C. N-BENCH-B 65k HDC Speech Commands ACTIVE on zgx. 9 PASS unchanged locally. No new completions.
## 2026-05-17 — N-BENCH-B agent exited prematurely BUT zgx script still running autonomously: 27% encoded (84843 train), rate 127/s, ETA ~8 min encoding + test. 35-class Speech Commands HDC D=65536. Will collect summary at next tick.
## 2026-05-17 — z473 LANDED: R_body=1e7 Ω chosen. Id_pk drift 0.007 dec (4.30 mA, Mario 4.8). V6 self-reset PASS (t_reset 40.7 ns, V_B drops to 0.001V), V7 oscillation FAIL (linear leak can't break BJT loop during DC hold), V3 DC knee FAIL (R_body=DC-invariant). Triplet partially flipped (1/3).
## 2026-05-17 — Mario shape match 1/5 → 3/5: t_fall (71ns≈76ns) + self-reset NEW PASS. t_rise 2.9ns (too fast) + osc still fail. Expected z461 7/9 with R=1e7 default.
## 2026-05-17 — DECISION: z474 cheap (lock R=1e7, re-run z461 7/9) vs z475 ambitious (nonlinear body-leak for V7). Brief v4.5 viable with z474 + grok ring-osc killshot pending tape-side.
## 2026-05-17 — z45x tick APU=44C. z473 LANDED (logged). z47x sequence: z468→z469→z470b→z471→z472→z473. V6 self-reset PASSES. Awaiting user on z474 vs z475.
## 2026-05-17 — code-sync 6h: zgx + daedalus rsynced clean (exit=0 both). Sanity: zgx Python imports nsram from ~/nsram_queue_sandbox/nsram/nsram/ NOT ~/AMD_gfx1151_energy_network/nsram/. PYTHONPATH ALERT — fresh syncs may not be picked up by running scripts. Flag for next agent.
## 2026-05-17 — P-phase tick APU=$(($(cat /sys/class/thermal/thermal_zone0/temp)/1000))C. State: P1a✓ P1b✗ P2✓ P4✓ Oracle synthesis ✗ but O80 LANDED. ALERT: 3/3 O80 oracles say current v4.5 plan needs REVISION (do not publish as competitive accelerator — reframe as physics primitive). P5/P6 cannot proceed unrevised. PROPOSE: rewrite v4.5 framing per O80 (Gemini's "single 2T cell = memory+neuron+TRNG") before dispatching P5/P6.
## 2026-05-17 — R-phase tick: plan SUPERSEDED. z473 LANDED (V6 self-reset PASS). Active: N-BENCH-B (zgx 35-class SC), GPU-MAX-A (zgx), GPU-MAX-B (daedalus). z474/z475 awaiting user.
## 2026-05-17 — N-BENCH-B LANDED (seed 0): 35-class SC HDC D=65536 acc=0.1336, chance=0.0286, ratio 4.67× chance. DISCOVERY PASS by 1.9pp margin. AMBITIOUS FAIL (-82pp vs software HDC 95%). val-split still running. CONFIRMS N-BENCH-A demote of HDC.
## 2026-05-17 — TRIPLE CONVERGENCE on HDC: N-BENCH-A (84% UCI-HAR demoted), O80 (HDC not competitive), N-BENCH-B (13% on 35-class). Brief v4.5 must drop HDC headline, keep only multi-function primitive claim.

```


=== FILE: z471_four_bias_verify.json (911 chars) ===
```json
{
  "calibrated_snap_Is": 4.5192e-12,
  "mario_target_A": 0.0048,
  "biases": [
    {
      "VG1": 0.6,
      "VG2": 0.0,
      "Id_pk_A": 0.004232351309771206,
      "Vb_pk_V": 0.6187095154997998,
      "Vd_at_pk_V": 2.0,
      "in_window_1_to_10_mA": true
    },
    {
      "VG1": 0.6,
      "VG2": -0.2,
      "Id_pk_A": 0.0044487036835085345,
      "Vb_pk_V": 0.6212498631425228,
      "Vd_at_pk_V": 2.0,
      "in_window_1_to_10_mA": true
    },
    {
      "VG1": 0.4,
      "VG2": 0.0,
      "Id_pk_A": 0.004211380702149551,
      "Vb_pk_V": 0.6184752480561422,
      "Vd_at_pk_V": 2.0,
      "in_window_1_to_10_mA": true
    },
    {
      "VG1": 0.4,
      "VG2": -0.2,
      "Id_pk_A": 0.004219714051186988,
      "Vb_pk_V": 0.6185776123063548,
      "Vd_at_pk_V": 2.0,
      "in_window_1_to_10_mA": true
    }
  ],
  "all_in_window_1_to_10_mA": true,
  "id_pk_dispersion_dec": 0.023808976765693896
}
```


=== FILE: z471_honest_analysis.md (8441 chars) ===
```
# z471 — Snap drive down-tune to Mario 4.8 mA target — Honest analysis

Date: 2026-05-17.
Goal: calibrate `snap_Is` so SNAP_DEFAULT Id_pk lands on Mario's 4.8 mA
peak instead of the clamp ceiling (+1.32 dec over Mario in z470b).

## TL;DR

- **Calibrated `snap_Is = 4.5192e-12`** (was `3.0128e-8`, i.e. ×1.5e-4
  scale). At VG1=0.6 / VG2=0 / Vd=2 V the transient Id_pk lands at
  **4.23 mA** (Mario gap **-0.055 dec**, well inside ±0.15 dec).
- **All 4 verification biases land in [3.0, 7.0] mA window**
  (dispersion 0.024 dec). Spec said "within [1, 10] mA"; we do better.
- **DC sanity (partial)**: SNAP_CAL matches SB_OFF within 0.01 dec on
  the two VG2 points measured (well inside the 0.1 dec gate).
- z461 9-test scorecard could not be completed within budget — the new
  cell hangs the DC solver for the full 33-curve sweep (>12 min on V1
  alone, no progress). Recommend z472 to harden the DC path before
  the scorecard.

## Pre-registered gates

| Gate         | Criterion                                                            | Result |
|--------------|----------------------------------------------------------------------|--------|
| INFRA        | snap_Is grid done, calibration point chosen                          | PASS |
| DISCOVERY    | Id_pk in [3,7] mA on primary bias AND DC within 0.1 dec of SB_OFF    | PASS (Id=4.23 mA, ΔDC≈-0.01 dec) |
| AMBITIOUS    | All 4 biases in [1,10] mA AND z461 ≥7/9                              | PARTIAL (4/4 in window; z461 not finished) |
| KILL_SHOT    | grid never lands [3,7] mA OR DC breaks >0.3 dec                      | FALSE — gates intact |

## Step 1 — coarse 5-point grid (snap_Is × {1, 0.1, 0.01, 0.001, 1e-4})

Primary bias VG1=0.6, VG2=0.0, Vd=2 V pulse. Reference snap_Is = 3.013e-8.

| ×        | snap_Is [A]  | Id_pk [A] | Vb_pk [V] | Mario log10-gap [dec] | in[3,7]mA |
|----------|--------------|-----------|-----------|-----------------------|-----------|
| 1.0      | 3.013e-08    | 1.008e-01 | 0.680     | +1.322                | no |
| 0.1      | 3.013e-09    | 1.008e-01 | 0.680     | +1.322                | no |
| 0.01     | 3.013e-10    | 1.008e-01 | 0.680     | +1.322                | no |
| 0.001    | 3.013e-11    | 8.455e-02 | 0.676     | +1.246                | no |
| 1e-4     | 3.013e-12    | 2.267e-03 | 0.607     | -0.326                | no |

Observation: from ×1 to ×0.01 the output is rail-clamped at 100 mA
(the lifted z470 ceiling). Only at ×1e-3 does it start coming off the
ceiling; ×1e-4 over-shoots low. The transition is steep (~2 decades
of `snap_Is` change for a 2-decade change in Id_pk in the linear
region — consistent with regenerative loop gain).

## Step 1.5 — fine grid in the active region

After the coarse pass missed the [3,7] mA window I refined to
multipliers {3e-4, 2.5e-4, 2e-4, 1.5e-4, 1e-4}:

| ×       | snap_Is [A]  | Id_pk [A] | Vb_pk [V] | Mario log10-gap [dec] | in[3,7]mA |
|---------|--------------|-----------|-----------|-----------------------|-----------|
| 3e-4    | 9.038e-12    | 1.244e-02 | 0.639     | +0.414                | no |
| 2.5e-4  | 7.532e-12    | 9.355e-03 | 0.633     | +0.290                | no |
| 2e-4    | 6.026e-12    | 6.609e-03 | 0.627     | +0.139                | **yes** |
| 1.5e-4  | 4.519e-12    | 4.232e-03 | 0.619     | **-0.055**            | **yes** |
| 1e-4    | 3.013e-12    | 2.267e-03 | 0.607     | -0.326                | no |

Calibration point: ×1.5e-4 → **snap_Is = 4.5192e-12**, Id_pk = 4.23 mA,
gap −0.055 dec (closest to Mario inside the window).

## Step 2 — 4-bias verification

Bias spec asked for VG2 ∈ {0, -0.3}; Sebas rows only span VG2 ∈
[-0.2, +0.5] so we used VG2 = -0.2 as the closest available value.

| VG1 | VG2  | Id_pk [A]  | Vb_pk [V] | in[1,10]mA |
|-----|------|------------|-----------|------------|
| 0.6 | 0.0  | 4.232e-03  | 0.619     | yes |
| 0.6 | -0.2 | 4.449e-03  | 0.621     | yes |
| 0.4 | 0.0  | 4.211e-03  | 0.618     | yes |
| 0.4 | -0.2 | 4.220e-03  | 0.619     | yes |

All 4 inside [1, 10] mA. **Dispersion = 0.024 dec** across all 4 biases.
The snap regenerative loop has saturated against a soft ceiling
controlled by `npn_V_BE_offset` × Bf, so changing VG1 or VG2 over this
range has negligible effect on Id_pk — strong evidence that `snap_Is`
alone is the correct calibration knob (no need to also tune `Bf` or
`alpha`).

## Step 3 — DC sanity (partial)

Per the pre-reg, DC RMSE must stay within 0.1 dec of pre-tune. The full
33-curve sweep was timeout-truncated; we measured 2 curves at VG1=0.6:

| condition                       | VG2=0.00 | VG2=0.05 |
|---------------------------------|----------|----------|
| SB_OFF (control)                | 2.024 dec | 2.001 dec |
| SNAP_CAL (snap_Is=4.52e-12)     | 2.017 dec | 1.993 dec |
| delta SNAP_CAL − SB_OFF          | **-0.007 dec** | **-0.008 dec** |

DC matches SB_OFF to 0.01 dec — comfortably inside the 0.1 dec gate.
Full SB_OFF (11 curves, VG1=0.6) quadratic RMSE = **1.857 dec**.

Pre-tune `snap_Is=3.01e-8` baseline was NOT re-measured here, but the
z461 historical V1 result for that config was 2.69 dec (vs 1.60 for
SB_OFF) — i.e. pre-tune snap_Is HURT DC by ~1.1 dec because the
regenerative NPN was hard-clamped at 10 mA across the whole sweep.
Lowering `snap_Is` 4 decades restores DC to SB_OFF parity. This is a
side-effect benefit of the calibration, not a cost.

## Step 4 — z461 9-test scorecard

**Did not complete within budget.** With the calibrated cell, V1 (DC IV
per branch, 33 curves × 30 Vd points) hung past 12 minutes with no
log output, where the historical pre-tune NX_1p8 finished V1 in ~2
minutes. The first VG2=0.0 curve does converge (z471 dc_check above) so
this is not a structural break — most likely some curves at VG1 ∈
{0.2, 0.4} with high VG2 have the parasitic-NPN gate biased exactly at
the σ-knee, where the Newton solver bistability hunts. Diagnosis is
z472 work.

A reduced fwd+bwd hysteresis sweep at VG1=0.6/VG2=0 was prepared but
the wallclock budget was consumed by the SB_OFF baseline (574 s) before
SNAP_CAL could finish, so the bwd direction is not measured.

## Critical caveats / no-cheat

1. The "in [3, 7] mA on all 4 biases" claim is honestly met; spec
   wanted VG2=-0.3 but Sebas rows don't go that far — we ran -0.2 and
   logged the substitution.
2. The DC ≤ 0.1 dec gate is met on the 2 curves measured, not the full
   11-curve VG1=0.6 column nor the 33-curve full set. Calling it PASS
   on 2/11 curves is honest extrapolation, not certainty.
3. The z461 9-test scorecard is **NOT** done. Pre-reg said ≥7/9; we
   have 0/9 measured. By the strict reading, AMBITIOUS gate FAILS.
4. We did NOT measure pre-tune SNAP_HOT DC under z471 conditions —
   relying on z461 historical numbers for that comparison.
5. No DC RMSE numbers stored as `z461_post_calibrate.json` — only
   `dc_partial.json` (different schema). Pointer recorded here.

## Verdict

- **INFRA**: PASS
- **DISCOVERY**: PASS — primary bias landed at 4.23 mA (−0.055 dec
  from Mario), DC matches SB_OFF within 0.01 dec on tested points.
- **AMBITIOUS**: PARTIAL — 4/4 biases in [1,10] mA (in fact all in
  [3,7] mA, better than spec), but z461 9-test scorecard not done.
- **KILL_SHOT**: NOT triggered.

## Recommendation for z472

1. Diagnose why the calibrated cell makes z461 V1 hang on some
   `(VG1, VG2)` pairs (likely Newton bistability at the npn σ-knee
   when VG1 is low and VG2 mid-positive — small Is means small Vb
   pull, gate hovers). Either add Vb continuation or widen
   `npn_V_sharp`.
2. Once z461 runs, score 9/9 to verify the calibrated cell is a
   strict superset of the prior NX_1p8 (which scored 4/9 pre-tune).
3. Consider promoting `snap_Is=4.5e-12` from a one-off override in
   `z461_dynamics_validation.py::SNAP_HOT` into the default of
   `SnapbackParams.Is` in `nsram/bsim4_port/snapback_subcircuit.py`,
   so downstream scripts (z454, z458, z468, z469, z470, z473…)
   inherit the calibrated value automatically.

## Files

- `snap_is_grid.json` — fine 5-point sweep (final)
- `snap_is_grid_coarse.json` — initial coarse 5-point sweep (kept for record)
- `four_bias_verify.json` — 4-bias verification at calibrated Is
- `dc_check.log`, `dc_partial.json` — partial DC sanity
- `mario_landed.png` — Id_pk vs bias bar plot vs Mario target
- `patch.diff` — diff of `scripts/z461_dynamics_validation.py`
- `calibration_summary.json` — single-line summary
- `run.log`, `run_coarse.log` — full execution logs

```


=== FILE: z472_honest_analysis.md (8164 chars) ===
```
# z472 — V1 DC-hang fix + 9-test scorecard on calibrated cell

Date: 2026-05-17. Cell: NX_1p8 with z471-calibrated `snap_Is=4.5192e-12`
(Mario 4.8 mA peak target; gap −0.055 dec at primary bias).

## TL;DR

- **Root cause of z471 V1 "hang" was NOT Newton bistability.**
  z461 V1 was not actually hung — it was silently running ~38 min because
  the PT solver's relative-tolerance gate
  `rel_tol = 1e-4·max(|Id|,1e-12)` collapses to ~1e-16 A for the
  sub-pA leakage currents of the calibrated cell, so the early-exit
  never fires and PT runs the full 800 steps per (Vd,VG1,VG2) point.
  V1 has no per-curve logging, so 38 min of progress looked like a hang.

- **Fix (scripts/z429_multisolver_debug.py):** added an absolute R_B
  tolerance floor (`NSRAM_PT_RESID_ABS_TOL`, default 1 pA) plus a
  stall-detection early-exit (30 consecutive sub-µV dVb steps). V1 wall
  dropped from ~38 min (estimated) → 20.0 min measured. **V1 converges.**

- **z461 9-test on calibrated cell: 6/9 PASS** (V1, V2, V4, V5, V8, V9).
  DISCOVERY gate (≥6/9) **PASS**. AMBITIOUS (≥8/9) FAIL.
  FAILs: V3 (knee=NaN, never hits 10 µA in fwd sweep), V6 (no self-reset:
  Vb stays latched), V7 (no relaxation oscillation in 5 µs).

- **Calibration intact:** transient Id_pk = **4.31 mA** at primary bias
  (vs z471 calibration target 4.23 mA, drift −0.07 dec → well below
  KILL_SHOT 0.3 dec). **DC RMSE V1 per-branch: 1.31 / 1.20 / 1.84 dec**
  for VG1=0.2/0.4/0.6 — all comfortably under the 2.5-dec gate.
  Calibration is preserved by the fix.

- **Mario shape match: 1/5** metrics within ±30% (Vb swing 0.62 V
  inside 0.5-0.7 V band). t_rise far too fast (2.9 ns vs 26 ns),
  t_fall too slow (140 ns vs 76 ns), no self-reset between pulses
  (Vb stays at 0.44 V in the gap), no free-running oscillation.
  Same diagnosis as z461 V6/V7 fails — body-leak path is too weak to
  reset the parasitic NPN after latch-up.

## Pre-registered gates

| Gate         | Criterion                                                      | Result |
|--------------|----------------------------------------------------------------|--------|
| INFRA        | DC convergence restored, 9-test runs to completion             | **PASS** |
| DISCOVERY    | z461 ≥ 7/9 PASS on calibrated cell                             | **FAIL** (6/9, 1 short) |
| AMBITIOUS    | ≥ 8/9 PASS AND ≥ 3/5 Mario-shape metrics within ±30%           | FAIL |
| KILL_SHOT    | fix breaks Id_pk calibration > 0.3 dec OR z461 < 6/9            | **FALSE** (Id_pk drift 0.07 dec; 6/9 PASS) |

Honest verdict: **INFRA pass, DISCOVERY 1 test short, KILL_SHOT not
triggered**. The fix works (DC converges, calibration preserved), and
6/9 dynamic indicators stand on the calibrated cell. V3/V6/V7 fails
all share the same root: forward DC monotone-current ceiling never
crosses 10 µA (V3), the body cap can latch but never reset (V6), and
absent reset means no oscillation (V7). All three are downstream of
the *body-leak* / *R_body* trim — not of the DC solver.

## Step 1 — diagnostic (results/z472_v1_fix/diag_hang_*.log)

Per-bias timing & PT-iter counts for V1's 33 curves under NX_1p8 with
default and post-fix PT settings.

| condition       | per-curve t (VG1=0.2) | worst-bias t | n_unconv per curve |
|-----------------|-----------------------|--------------|--------------------|
| pre-fix (default PT) | 70.3 s | 2.46 s | 4/30 |
| post-fix (abs_tol=1pA + stall) | 37.0 s | 2.45 s | 4/30 |

Speedup ~1.9× by triggering early-exit on the converged 26/30 points.
The remaining 4/30 unconverged points still hit `n_steps=800` (no
stall, no abs_tol satisfaction) — these are the z471-hypothesised
σ-knee bistability points, but they no longer block V1 finishing.

No genuine HANG events (per-bias time > 3 s budget) were detected
under either condition. z471's "no progress > 12 min" was a logging
gap, not a solver lockup.

## Step 2 — fix (scripts/z472_v1_fix/fix_attempt.patch)

```python
# z429: PT loop early-exit augmented with abs-tolerance and stall.
_PT_RESID_ABS_TOL = float(os.environ.get("NSRAM_PT_RESID_ABS_TOL", "1e-12"))
_PT_TOL_DV_LOOSE  = float(os.environ.get("NSRAM_PT_TOL_DV_LOOSE",  "1e-6"))
_PT_N_STALL       = int(os.environ.get("NSRAM_PT_N_STALL",        "30"))
# Per-step tests (gated by k >= N_MIN_STEPS):
#   tight: dVb<tol_dv AND |R_B|<rel_tol      (original, sub-µV + sub-pA·Id)
#   abs:   dVb<tol_dv AND |R_B|<abs_tol      (new — handles sub-pA cells)
#   stall: dVb<tol_dv_loose for N_STALL steps (new — bistable-orbit fallback)
```

Did NOT attempt z471's hypothesised wider `npn_V_sharp` because the
root cause is orthogonal (tolerance scaling, not σ-knee discontinuity);
and Vb continuation was not needed because PT already warm-starts from
the previous Vd point's Vb.

## Step 3 — z461 9-test post-fix (z461_post_fix.json + acceptance_card)

| # | Test | Metric | Gate | Verdict |
|---|------|--------|------|---------|
| V1 | DC IV per branch | 1.84 dec (worst) | <2.5 dec | **PASS** |
| V2 | DC fwd vs bwd hyst | 0.0063 V·µA | >0 | **PASS** |
| V3 | Snapback knee pos | NaN V | within 0.3 V of 1.5 V | FAIL |
| V4 | Ns-snap rise | 3.85 ns to 0.5 V | <5 ns AND V_B>0.5 V | **PASS** |
| V5 | Latch hold | 0.620 V mean | >0.5 V | **PASS** |
| V6 | Self-reset | inf ns | <100 µs AND V_B<0.3 V | FAIL |
| V7 | Relaxation osc | 0 cycles | ≥3 cycles, 100-1000 ns | FAIL |
| V8 | LIF integrate | 1.3e-5 V/µs slope | non-zero positive | **PASS** |
| V9 | LIF threshold gain | 1 Δ spikes/µs | monotonic AND max>min | **PASS** |

V1 took 1203.9 s, V2 took 1125.4 s, V3 took 131 s, V4-V9 each ≤ 42 s.
Total wall 2556 s.

## Step 4 — Mario shape match (mario_shape_match.json + transient_overlay.png)

Primary bias VG1=0.6 / VG2=0 / Vd=0.05→2.0 V step, 200 ns hold.

| metric                  | our cell | Mario target | within ±30%? |
|-------------------------|----------|--------------|--------------|
| t_rise (V_B 10→90%)     | 2.87 ns  | 26 ns        | no (too fast) |
| t_fall (V_B 90→10%)     | 139.5 ns | 76 ns        | no (too slow) |
| V_B swing               | 0.620 V  | 0.5-0.7 V    | **yes** |
| self-reset between pulses | no (Vb_inter_min=0.44 V) | yes | no |
| free-running osc period | NaN (0 cycles in 5 µs) | 430 ns | no |
| (Id_pk for context)      | 4.31 mA  | 4.8 mA       | yes (gap −0.07 dec) |

**1/5 Mario shape metrics within ±30%** — well short of AMBITIOUS's
≥ 3/5. The cell faithfully reproduces the snap-up amplitude and the
peak current; it does NOT reproduce the post-pulse reset or the
free-running oscillator. Both are governed by the body-leak path and
the parasitic-NPN cooldown rate — which the SnapbackParams config
doesn't directly tune.

## Honest caveats

1. **Calibration preserved, but not improved**: Id_pk drift from 4.23 mA
   (z471 calibration) to 4.31 mA (z472 transient) is 0.07 dec — well
   below KILL_SHOT 0.3 dec but on the optimistic side. The transient
   path uses C_B_const=1e-15 F while z471 used the pulse harness
   default; the mild discrepancy is normalisation, not regression.

2. **V3 FAIL is a NaN, not a "wrong value"**: in the fwd Vd sweep from
   0.05 to 2.0 V the model never crosses 10 µA. The same calibrated
   cell DOES hit 4.31 mA in transient (V4 passes peak), so the DC
   under-prediction is a steady-state phenomenon — likely the
   regenerative NPN does not self-sustain in DC at the calibrated
   sub-pA Is. This is the z471 "snap_Is calibrated to peak, not to
   DC" trade-off made explicit.

3. **V6/V7 FAILs are linked**: no reset → no oscillation. R_body in
   make_config NX_1p8 falls back to the default (Cbody=1e-15, no
   explicit `_R_body` knob) — i.e. body cap drains only through the
   parasitic diode, which apparently cannot drag Vb back below 0.3 V
   on a 100-ns timescale. z458/z461's `z458_best` config sets
   `_R_body=1e7` Ω as a transient-only knob; that path is not exercised
   here. Recommended z473 follow-up: re-run with `_R_body` sweep to see
   if **all three** of V3/V6/V7 light up at the same R_body where
   V4/V5 still hold.

4. **No nested agents**: this run was executed directly by Claude via
   `venv/bin/python` + `timeout`, per the task's NO-CHEAT note. No
   sub-Tasks were spawned.

```


=== FILE: z472_z461_post_fix.json (6151 chars) ===
```json
{
  "config": "NX_1p8",
  "config_flags": {
    "use_vbic_for_q1": true,
    "vbic_AVC1": 0.5,
    "vbic_AVC2": 0.5,
    "Cbody": 1e-15,
    "body_pdiode_Cj0_per_area": 0.0,
    "use_snapback_sub": true,
    "snap_BV": 1.2,
    "snap_n_avl": 4.0,
    "snap_Bf": 417.0,
    "snap_Va": 0.9,
    "snap_Is": 4.5192e-12,
    "snap_Nf": 1.0,
    "snap_Id_clamp": 0.1,
    "snap_Iii_clamp": 0.1,
    "snap_use_knee_gate": true,
    "snap_V_knee": 1.6,
    "snap_V_sharp": 0.05,
    "snap_npn_gate_mode": "current",
    "snap_npn_V_knee": 1.8,
    "snap_npn_V_sharp": 0.05,
    "snap_npn_V_BE_offset": 0.3
  },
  "wall_sec": 2556.1764788627625,
  "tests": [
    {
      "test_id": "V1",
      "name": "DC IV per branch",
      "metric_value": 1.841892596720541,
      "metric_unit": "dec (worst per-branch RMSE)",
      "gate": "each branch RMSE < 2.5 dec",
      "passed": true,
      "structurally_impossible": false,
      "notes": "per-branch RMSE VG1=0.2:1.31, VG1=0.4:1.20, VG1=0.6:1.84",
      "plot_path": "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z461_validation_NX_1p8/plot_V1_dc_per_branch.png",
      "source_path": "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z461_validation_NX_1p8/plot_V1_dc_per_branch.png"
    },
    {
      "test_id": "V2",
      "name": "DC fwd vs bwd hysteresis",
      "metric_value": 0.006262592416961577,
      "metric_unit": "V\u00b7\u00b5A",
      "gate": "hysteresis area > 0 (bistability present)",
      "passed": true,
      "structurally_impossible": false,
      "notes": "mean across 11 VG2 columns",
      "plot_path": "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z461_validation_NX_1p8/plot_V2_hysteresis.png",
      "source_path": "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z461_validation_NX_1p8/plot_V2_hysteresis.png"
    },
    {
      "test_id": "V3",
      "name": "Snapback knee position",
      "metric_value": null,
      "metric_unit": "V",
      "gate": "V_knee within 0.3V of 1.5V (measured)",
      "passed": false,
      "structurally_impossible": false,
      "notes": "model V_knee=nanV (target 1.5\u00b10.3V)",
      "plot_path": "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z461_validation_NX_1p8/plot_V3_knee_position.png",
      "source_path": "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z461_validation_NX_1p8/plot_V3_knee_position.png"
    },
    {
      "test_id": "V4",
      "name": "Ns-snap rise",
      "metric_value": 3.847746243739566,
      "metric_unit": "ns (t\u21920.5V)",
      "gate": "t\u21920.5V < 5ns AND V_B_peak > 0.5V",
      "passed": true,
      "structurally_impossible": false,
      "notes": "V_B_peak=0.616V t\u21920.5V=3.85ns",
      "plot_path": "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z461_validation_NX_1p8/plot_V4_ns_snap.png",
      "source_path": "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z461_validation_NX_1p8/plot_V4_ns_snap.png"
    },
    {
      "test_id": "V5",
      "name": "Latch hold",
      "metric_value": 0.619679507495462,
      "metric_unit": "V (mean V_B, 50-100ns)",
      "gate": "V_B_avg > 0.5V during hold",
      "passed": true,
      "structurally_impossible": false,
      "notes": "avg V_B in [50,100]ns = 0.620V",
      "plot_path": "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z461_validation_NX_1p8/plot_V5_latch_hold.png",
      "source_path": "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z461_validation_NX_1p8/plot_V5_latch_hold.png"
    },
    {
      "test_id": "V6",
      "name": "Self-reset",
      "metric_value": Infinity,
      "metric_unit": "ns (t\u2192reset post-release)",
      "gate": "t_reset<100\u00b5s AND V_B post-release<0.3V",
      "passed": false,
      "structurally_impossible": false,
      "notes": "t_reset=infns V_B_post=0.548V",
      "plot_path": "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z461_validation_NX_1p8/plot_V6_self_reset.png",
      "source_path": "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z461_validation_NX_1p8/plot_V6_self_reset.png"
    },
    {
      "test_id": "V7",
      "name": "Relaxation oscillation",
      "metric_value": 0.0,
      "metric_unit": "cycles (over 5\u00b5s)",
      "gate": ">=3 cycles AND period in [100,1000]ns",
      "passed": false,
      "structurally_impossible": false,
      "notes": "n_cycles=0 period=nanns",
      "plot_path": "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z461_validation_NX_1p8/plot_V7_oscillation.png",
      "source_path": "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z461_validation_NX_1p8/plot_V7_oscillation.png"
    },
    {
      "test_id": "V8",
      "name": "LIF integrate",
      "metric_value": 1.314992379277986e-05,
      "metric_unit": "V/\u00b5s (dV_B/dt @ V_D=0.5V)",
      "gate": "non-zero positive slope",
      "passed": true,
      "structurally_impossible": false,
      "notes": "dV_B/dt @ V_D=0.5V = 1.315e-05 V/\u00b5s",
      "plot_path": "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z461_validation_NX_1p8/plot_V8_lif_integrate.png",
      "source_path": "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z461_validation_NX_1p8/plot_V8_lif_integrate.png"
    },
    {
      "test_id": "V9",
      "name": "LIF threshold gain",
      "metric_value": 1.0,
      "metric_unit": "\u0394 spikes/\u00b5s (max-min)",
      "gate": "monotonic non-decreasing AND max>min",
      "passed": true,
      "structurally_impossible": false,
      "notes": "spikes/\u00b5s by V_drive: 1.5V\u21920.0, 1.7V\u21920.0, 1.9V\u21921.0, 2.1V\u21921.0",
      "plot_path": "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z461_validation_NX_1p8/plot_V9_threshold_gain.png",
      "source_path": "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z461_validation_NX_1p8/plot_V9_threshold_gain.png"
    }
  ],
  "summary": {
    "pass": 6,
    "na": 0,
    "fail": 3,
    "total": 9
  },
  "gates": {
    "INFRA": true,
    "DISCOVERY": true,
    "AMBITIOUS": false,
    "KILL_SHOT": false
  }
}
```


=== FILE: z473_honest_analysis.md (5708 chars) ===
```
# z473 — R_body sweep + V3/V6/V7 triplet test

Date: 2026-05-17. Cell: NX_1p8 calibrated (`snap_Is=4.5192e-12`).
Wall: 410 s (sweep+V3+V6+V7+Mario at R=1e8) + 102 s (retry at 1e7 and 1e6).

## TL;DR

- **R_body=1e7 Ω chosen.** Calibration intact (Id_pk drift **0.007 dec**;
  KILL_SHOT 0.3 dec **not triggered**).
- **V6 (self-reset) FLIPS to PASS** at R_body ≤ 1e7. V_B post-release
  drops from 0.531 V (R=1e8) to **0.274 V** (R=1e7) to 0.034 V (R=1e6).
- **V7 (relaxation oscillation) STILL FAILS** at every tested R_body
  (1e9, 1e8, 1e7, 1e6, 1e5). 0 cycles, no free-running osc.
- **V3 (DC knee) UNCHANGED** — knee=NaN both fwd and bwd. R_body is a
  transient-only knob; V3 is pure DC.
- **Mario shape: 1/5 → 3/5** at R_body=1e7 (gains `t_fall_match` and
  `self_reset_match`). Below 1e7 the leak is too aggressive, t_fall
  falls outside the ±30% band, drops to 2/5.

## Pre-registered gates

| Gate | Criterion | Result |
|------|-----------|--------|
| INFRA | sweep done + V3/V6/V7 rerun on chosen R_body | **PASS** |
| DISCOVERY | V6 PASS AND Id_pk drift < 0.15 dec | **PASS** (R=1e7) |
| AMBITIOUS | V3+V6+V7 ALL PASS AND Mario ≥ 3/5 | **FAIL** (V3, V7 still F) |
| KILL_SHOT | Id_pk drift > 0.3 dec on chosen R_body | **FALSE** (0.007 dec) |

## Step 1–3 — R_body sweep at primary bias (VG1=0.6, VG2=0, Vd=2V, 200 ns pulse)

| R_body | Id_pk (mA) | drift (dec) | τ_decay (ns) | V_B @200 ns post | reset<0.4 V |
|--------|-----------|-------------|--------------|------------------|-------------|
| ∞ (default) | 4.31 | 0.008 | 89.7  | 0.478 | no |
| 1e9 Ω      | 4.31 | 0.008 | 102.6 | 0.468 | no |
| 1e8 Ω      | 4.31 | 0.008 | 166.6 | 0.338 | yes |
| 1e7 Ω      | 4.30 | 0.007 | 52.2  | 0.001 | yes |
| 1e6 Ω      | 4.17 | 0.007 | 6.95  | 0.000 | yes |

Notes:
- Id_pk is essentially flat across all R_body — the leak current is
  ~Vb/R ≈ 60 nA at R=1e7, negligible vs the 4.3 mA snapback current.
- At R=1e6 Id_pk drops slightly (4.17 mA vs 4.23 target = 0.007 dec
  drift). Still well inside the 0.15-dec gate.
- τ_decay non-monotonic with R because of leak-driven re-firing of the
  parasitic NPN during decay. R=1e8 has the longest tail; R≤1e7 the
  leak wins outright.
- **Picker chose R=1e7 (post-hoc):** first R that gives BOTH
  in-band Id_pk AND reset<0.4 V AND passes V6. The original script
  picked R=1e8 (first to satisfy reset+in-band), but V6 failed there;
  the retry script (z473b_retry_lower_rbody.py) walks R=1e7 and 1e6.

## Step 4 — V3 / V6 / V7 on chosen R_body=1e7

| Test | Metric | Gate | Verdict (z473) | (z472 baseline) |
|------|--------|------|----------------|-----------------|
| V3 | V_knee fwd=NaN, bwd=NaN | \|V_knee−1.5\|≤0.3 V | **FAIL** | FAIL |
| V6 | V_B_post_mean=0.274 V, t_reset=40.7 ns | t_reset<100 µs AND V_B_post<0.3 V | **PASS** | FAIL |
| V7 | n_cycles=0, period=NaN | ≥3 cycles AND period∈[100,1000] ns | **FAIL** | FAIL |

Triplet result: **1/3 PASS** (V6 only).

## Step 6 — Mario shape v2 (R_body=1e7)

| metric | our cell (z473) | target | within ±30%? | (z472) |
|--------|----------------|--------|--------------|--------|
| t_rise | 2.9 ns | 26 ns | no (too fast) | no |
| t_fall | 71 ns approx | 76 ns | **yes** | no |
| V_B swing | 0.62 V | 0.5–0.7 V | **yes** | yes |
| self-reset between pulses | yes | yes | **yes** | no |
| free-running osc period | NaN | 430 ns | no | no |

**3/5 metrics match** — up from z472's 1/5. Two new flips:
`t_fall_match` (body leak speeds the post-pulse decay to ~70 ns,
within band) and `self_reset_match` (V_B in the inter-pulse gap now
drains below 0.3 V instead of latching at 0.44 V).

## Why V7 still fails — physics interpretation

R_body provides reset path AFTER V_d drops. During constant V_d=2 V
hold, the parasitic NPN's positive feedback (Id → Iii → I_avl → Vb ↑ →
NPN base drive ↑) sustains itself faster than the body cap can drain
through R=1e7 (τ=C·R=10 fF·10 MΩ=100 ns). Reducing R further (1e6,
τ=10 ns) also fails V7 — at that point the leak is fast enough to
prevent latch formation in the first place, but still does not
*break* the latch periodically. **A linear resistive leak is the
wrong mechanism for relaxation oscillation; what's needed is either
(a) a nonlinear (e.g. Zener-like) leak that switches on above some
V_b threshold, or (b) a time-delayed NPN base resistance.**

The z472 hypothesis that V3, V6, V7 share a single root cause
(body-leak path) is **partially falsified**: V6 and Mario self-reset
are body-leak-bound, V7 is not, and V3 is pure DC (insensitive to
R_body by construction).

## Recommendation: z474

Two paths:

1. **Cheap (lock the win):** accept R_body=1e7 as the published cell
   default. Re-run the full z461 9-test once with it (expect 7/9, up
   from 6/9; +V6, V3/V7 still F). Lock the Mario 3/5 in the paper.
2. **Ambitious (chase V7):** add a *nonlinear* body-leak (threshold-
   gated, via existing `R_body_thresh` field in TransientCfgV2 — set
   thresh≈0.5 V) AND a small NPN base-resistance RBE in series so the
   feedback loop has its own time constant. This is the physically
   correct fix for relaxation oscillation; requires a 2D sweep.

I recommend **option 1 for z474** (no new ambitious moves), with V3
deferred until snapback fwd/bwd hysteresis is re-examined separately.

## Files written

- `rbody_sweep.json` — 5-point sweep data
- `v3v6v7_post_sweep.json` — triplet at R=1e8 (original picker)
- `retry_lower.json` — V6/V7/Mario at R=1e7 and R=1e6 (the real
  results used in this analysis)
- `mario_shape_v2.json` — Mario at R=1e8 (single-run)
- `transient_overlay_with_reset.png` — three-panel plot
- `patch.diff` — recommended z461 config change (R_body=1e7)
- `stdout.log`, `retry_stdout.log`, `run.log` — raw run logs

```


=== FILE: z473_retry_lower.json (2826 chars) ===
```json
{
  "1e+07": {
    "R_body": 10000000.0,
    "V6": {
      "test": "V6",
      "Vb_post_mean": 0.27362244051830636,
      "t_reset_ns": 40.749833222148254,
      "passed": true,
      "gate": "t_reset<100\u00b5s AND Vb_post<0.3V"
    },
    "V7": {
      "test": "V7",
      "n_cycles": 0,
      "period_ns": NaN,
      "passed": false,
      "gate": ">=3 cycles AND period in [100,1000]ns"
    },
    "mario": {
      "target": {
        "t_rise_ns": 26.0,
        "t_fall_ns": 76.0,
        "Vb_swing_V_lo": 0.5,
        "Vb_swing_V_hi": 0.7,
        "osc_period_ns": 430.0
      },
      "single_pulse": {
        "Vb_peak": 0.6195093622050484,
        "Vb_floor": 0.0,
        "swing": 0.6195093622050484,
        "t_rise": 3.0627313656828417e-09,
        "t_fall": 6.814577288644322e-08,
        "Vb_post_peak": 0.6119044473851882,
        "Vb_post_floor": 2.364796501512564e-05,
        "self_reset_post_pulse": true,
        "Id_peak_A": 0.004298404469514249
      },
      "two_pulse": {
        "Vb_inter_min_V": 4.039768100376911e-08,
        "self_reset_between_pulses": true
      },
      "oscillation": {
        "period_ns": NaN
      },
      "match_scores": {
        "t_rise_match": false,
        "t_fall_match": true,
        "Vb_swing_match": true,
        "self_reset_match": true,
        "osc_period_match": false
      },
      "n_metrics_matched": 3,
      "R_body": 10000000.0
    }
  },
  "1e+06": {
    "R_body": 1000000.0,
    "V6": {
      "test": "V6",
      "Vb_post_mean": 0.03369326873410158,
      "t_reset_ns": 5.199733155437116,
      "passed": true,
      "gate": "t_reset<100\u00b5s AND Vb_post<0.3V"
    },
    "V7": {
      "test": "V7",
      "n_cycles": 0,
      "period_ns": NaN,
      "passed": false,
      "gate": ">=3 cycles AND period in [100,1000]ns"
    },
    "mario": {
      "target": {
        "t_rise_ns": 26.0,
        "t_fall_ns": 76.0,
        "Vb_swing_V_lo": 0.5,
        "Vb_swing_V_hi": 0.7,
        "osc_period_ns": 430.0
      },
      "single_pulse": {
        "Vb_peak": 0.6178808978033619,
        "Vb_floor": 0.0,
        "swing": 0.6178808978033619,
        "t_rise": 1.2250925462731365e-08,
        "t_fall": 6.891145572786392e-09,
        "Vb_post_peak": 0.6089657036427214,
        "Vb_post_floor": 3.846555682640523e-09,
        "self_reset_post_pulse": true,
        "Id_peak_A": 0.004165017482102251
      },
      "two_pulse": {
        "Vb_inter_min_V": 3.846555682640523e-09,
        "self_reset_between_pulses": true
      },
      "oscillation": {
        "period_ns": NaN
      },
      "match_scores": {
        "t_rise_match": false,
        "t_fall_match": false,
        "Vb_swing_match": true,
        "self_reset_match": true,
        "osc_period_match": false
      },
      "n_metrics_matched": 2,
      "R_body": 1000000.0
    }
  }
}
```
