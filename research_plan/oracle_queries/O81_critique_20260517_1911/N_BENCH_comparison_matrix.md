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
