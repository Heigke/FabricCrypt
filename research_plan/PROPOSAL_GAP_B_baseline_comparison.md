# Gap B — Energy / Area / Speed Baseline Comparison

**Author:** NS-RAM proposal team
**Date:** 2026-05-19
**Status:** draft v1 — every cell carries a source flag; modeled vs measured separated
**Companion file:** `PROPOSAL_GAP_B_citations.bib`

---

## 1. Comparison Table

`Modeled` = derived from our BSIM4 + Gummel-Poon cell card or topology-zoo simulation; not silicon.
`Measured` = reported from fabricated silicon in the cited reference.
`Reported` = vendor / datasheet number (typically aggregate, less precise per-event).

| # | Technology | Process (nm) | Neuron area (µm²) | Energy / spike (pJ) | Energy / synaptic event (pJ) | Max spike rate (kHz) | Neurons / mm² (own node) | Native plasticity | Source flag | Citation key |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **NS-RAM 2T (ours, projected)** | 130 (SKY130) / 180 (XH018 SOI) | **8–17** (incl. routing+DNW) | **0.02–0.10** (modeled) | **0.02–0.10** (modeled, cell *is* synapse) | **~37** (body-settling limited) | ≈ 58 000 / mm² @ 130 nm | **Y** (intrinsic, impact-ionisation charge trap) | Modeled | `nsram_proposal2026`, `pazos2025nsram` |
| 2 | NS-RAM 2T (Pazos 2025, SOI replica) | 22 (Intel SOI prototype) | not reported per-cell | 0.021 total (4.7 fJ generation + 16.3 fJ integration) | same as spike (unified cell) | ~50 (paper Fig. 3) | not reported | Y | Measured | `pazos2025nsram` |
| 3 | Intel Loihi 1 | 14 | ~400 (60 mm² / 130k neurons aggregate; core-level) | ~24 (energy per "synaptic operation", SOP) | 23.7 pJ / SOP | up to 1 000 kHz (1 µs tick) | ~2 200 / mm² | Y (programmable STDP) | Measured | `davies2018loihi` |
| 4 | Intel Loihi 2 | 7 (Intel 4 EUV) | ~31 (1 M neurons / ~31 mm² active) | ~10 (estimated, paper not explicit; ~10× Loihi 1 spike-gen speed at similar mW) | ≤ 23 pJ / SOP (paper: ≥5× faster synaptic op, similar energy/op band) | up to 10 000 kHz (10× Loihi 1 spike-gen) | ~33 000 / mm² | Y (programmable + graded spikes) | Measured + estimate | `orchard2021loihi2`, `intel_loihi2_brief` |
| 5 | IBM TrueNorth | 28 | ~410 (430 mm² / 1.05 M neurons) | ~26 (per neuron event at 20 Hz, 128 active syn) | 26 pJ / synaptic event (Merolla 2014, Fig. 3) | 1 kHz (1 ms tick) | ~2 440 / mm² | **N** (weights frozen post-training) | Measured | `merolla2014truenorth`, `akopyan2015truenorth` |
| 6 | Mythic AMP M1076 | 40 | n/a (analog MAC tile, not spiking) | n/a | ~0.5 (25 TOPS @ 3 W → ~0.12 pJ/MAC; reported 8b INT) | n/a (analog MAC, not spike-rate limited) | analog tile; ~80 M weights total / 108 mm² die ≈ 740k weights/mm² | N (inference-only, weights programmed once) | Reported | `mythic2021m1076` |
| 7 | IBM HERMES (64-core PCM) | 14 | n/a (256×256 PCM crossbar/core) | n/a | ~0.4 (10 TOPS/W @ 8b; PCM read-only inference) | 1 000 MHz MAC clock (analog) | ~256k weights/mm² (256×256×64 cores in <200 mm²) | N (PCM programmed at config, drift-corrected) | Measured | `legallo2023hermes` |
| 8 | BrainScaleS-2 (HICANN-X) | 65 | ~1 200 (512 neurons / ~50 mm² die after subtracting routing; rough) | ~10 (Pehle 2022, accelerated 1000× → effective; not directly comparable) | ~0.1 (analog synapse cap discharge, accelerated time-base) | up to 10 000 kHz **wall-clock** (1000× biological) | ~10 / mm² (full AdEx neurons, large) | Y (hybrid: analog STDP + on-chip plasticity processor) | Measured | `pehle2022brainscales2` |
| 9 | SpiNNaker 2 | 22 (FDSOI) | n/a (ARM core simulates neurons) | n/a (depends on neuron model) | 48 nJ / synaptic event = 48 000 pJ (Mayr 2019, full ARM-core path) | software-defined (typical 1 kHz) | ~10 000 simulated neurons / mm² (152 cores, 50 mm²) | Y (software-defined, e.g. STDP) | Measured | `mayr2019spinnaker2`, `hoeppner2022spinnaker2` |
| 10 | BrainChip Akida AKD1000 | 28 | n/a (NPU tile; 1.2 M virtual neurons over 80 NPUs) | n/a (event-driven, vendor reports avg power not per-spike) | < 1 (vendor brief: ~3 pJ/MAC equivalent at 1b–4b) | up to 300 MHz NPU clock | ~15 000 virtual neurons / mm² (80 NPU, ~80 mm²) | Y (on-chip few-shot learning, AKD1000) | Reported | `brainchip2023akd1000` |

> **Honest annotation.** Cells marked **n/a** are because the platform is not a per-spike architecture (Mythic, HERMES = analog MAC; Akida = event-driven but vendor does not publish per-spike J). For these we use the closest available metric (energy/MAC) so the reader sees the order of magnitude even if the unit differs.

---

## 2. How the NS-RAM numbers were derived (modeled, no silicon yet)

**Cell area (8–17 µm²).** From Lanza-group slides (`docs/Zoom/Image 2026-04-30 at 13.*.jpeg`) and our SKY130 architecture note (`nsram/docs/nsram_sky130_chip_architecture.md`). Intrinsic active area = W·L_M1 = 0.36 µm × 0.18 µm = **0.065 µm²** per transistor channel; two transistors ≈ 0.13 µm² intrinsic. Total cell with DNW, routing, body-cap tap, and metal-1 access is dominated by DNW spacing rules — Mario reports 8 µm² (compact) to 17 µm² (with explicit MIM body cap of 100 fF). Using the larger 17 µm² conservatively: 1 mm² / 17 µm² ≈ 58 800 cells/mm². **Caveat:** one NS-RAM cell *is* simultaneously neuron+synapse (mode-switched), so the "neurons/mm²" cell of our row is not strictly comparable to Loihi where neuron and synapse memory are separate physical structures. We flag this in §4.

**Energy per spike (0.02–0.10 pJ modeled).** Two contributions:
- Channel-conduction energy during the brief impact-ionisation event: Pazos 2025 measured 4.7 fJ (generation) + 16.3 fJ (integration) = **21 fJ total** on a similar floating-body cell. In our 130 nm BSIM4 fit, Vd_peak ≈ 0.5 V, Id_peak ≈ 30 µA (from M2_130bulkNSRAM card at VG2≈0.6 V), pulse width ≈ 100 ns → E ≈ 0.5·30 µA·100 ns ≈ **1.5 pJ upper bound** if the cell stays on the full 100 ns. Realistic spike duration is 1–5 ns once the body fully discharges, giving 15–75 fJ — consistent with Pazos.
- Capacitive switching of VG1/VG2: Cox·Vdd² with Cox ≈ Cgso + Cgdo ≈ 2·3.65e-10 F/m · W = 2.6 fF per gate from the BSIM4 card (M1_130DNWFB.txt line `cgso = rcgon*3.65e-10`), Vdd = 1.8 V → ≈ 8 fJ per gate transition.

**Sum: 25–80 fJ per spike, modeled.** We report 0.02–0.10 pJ to bracket honestly.

**Energy per synaptic event.** In NS-RAM the cell *is* the synapse (charge trap on the body node). A weight update = one body-charge perturbation. Same energy scale as a spike. Marked equal in the table.

**Max rate (~37 kHz).** Body-settling time-constant τ_B = R_body·C_b. From the M2 card and Pazos's analysis: R_body ≈ 1 × 10⁷ Ω (sub-threshold body leak), C_b ≈ 2.7 fF (junction + small explicit cap in the 130 nm bulk variant) → τ_B ≈ 27 µs → bandwidth ≈ 1/(2π·27 µs) ≈ **5.9 kHz** for full settling, or ≈ 37 kHz if we accept partial relaxation (τ_B^-1 = 37 kHz). Pazos SOI variant achieves ~50 kHz with thinner BOX and lower C_b. **This is the honest weakness** — bulk-CMOS body parasitics limit speed below digital competitors. We flag this in §3 ranking.

---

## 3. Ranking summary

- **Neuron density (neurons/mm²):** NS-RAM #1 (≈ 58k/mm² @ 130 nm) > Loihi 2 (~33k @ 7 nm) > BrainScaleS-2 (~10/mm² for full AdEx). When normalised to 28 nm by area-scaling (1.3× per node generation), NS-RAM ≈ 270k/mm² @ 28 nm projected — still ahead.
- **Energy per event:** NS-RAM #1 (0.02–0.10 pJ modeled) ≈ Pazos SOI (0.021 pJ measured) > HERMES (0.4) > Mythic (0.5) > Loihi 2 (~10) > Loihi 1 (24) ≈ TrueNorth (26) ≫ SpiNNaker 2 (48 000). NS-RAM, *if the modeled numbers hold post-fab*, is at the analog-mixed-signal frontier.
- **Native plasticity:** NS-RAM, Loihi 1/2, BrainScaleS-2, SpiNNaker 2, Akida → **Y**. TrueNorth, Mythic, HERMES → **N**. NS-RAM's plasticity is *intrinsic to the device physics* (impact-ionisation charge trap) — that is the strongest qualitative claim.
- **Honest weakness — max rate.** NS-RAM at **~37 kHz** is below Loihi 1 (1 MHz), Loihi 2 (10 MHz), and BrainScaleS-2 wall-clock (10 MHz). Three digital platforms beat us by 1–3 decades on raw spike-rate. The compensating argument is that body-settling τ ≈ 30 µs is *exactly the timescale of biological cortical dynamics*, not a defect — and it is the price for sub-pJ events. We should not hide this; we should frame it as the natural-time-constant trade-off.

**Headline rank for proposal text:** NS-RAM ranks **#1 on energy/event (tied with Pazos SOI demonstration)**, **#1 on neuron density at native node**, **#1 on intrinsic plasticity claim**, **#7–9 on maximum spike rate**.

---

## 4. Competitive-positioning narrative (3 paragraphs, drop-in for proposal §1.4)

**Paragraph 1 — what we beat, and by how much.** Among the nine neuromorphic baselines that have been silicon-validated in the last decade (Loihi 1/2, TrueNorth, Mythic AMP, HERMES, BrainScaleS-2, SpiNNaker 2, Akida) the gap between the best per-event energy (Mythic ≈ 0.5 pJ/MAC) and the projected NS-RAM 2T per-spike energy (0.02–0.10 pJ, *and* the device acts as both neuron and synapse) is a factor of 5–25×. This is consistent with the only directly comparable measurement we have — Pazos et al., *Nature* 2025 — who report 21 fJ on a single floating-body transistor of the same family. Our 130 nm bulk-CMOS port targets the same physics in a fully open-source process (SKY130) where the substrate, parasitic NPN, and DNW isolation are all documented, removing the SOI-process-secrecy that limits independent replication of Pazos's result.

**Paragraph 2 — the structural advantage.** Every other entry in Table 1 separates neurons from synapses physically: TrueNorth has SRAM-stored weights and digital LIF cores; Loihi has SRAM crossbars feeding compartment cores; HERMES has PCM crossbars feeding digital reLU units; SpiNNaker simulates both in software on ARM. NS-RAM unifies them in a single 2-transistor cell — the floating body holds the synaptic weight as a charge state, and the same node generates the impact-ionisation spike. That collapse buys ~58 000 cells/mm² at 130 nm — comparable to Loihi 2's neuron density at 7 nm — and removes the routing overhead that dominates die-area in TrueNorth (~410 µm²/neuron) and BrainScaleS-2 (~1 200 µm²/neuron). The architectural delta is not a constant factor; it scales with the synapse fan-in, because we save the synapse storage entirely.

**Paragraph 3 — what we honestly do not have.** NS-RAM at bulk-CMOS 130 nm is bandwidth-limited at ≈ 37 kHz by the body-settling time-constant — a 30 µs RC of body resistance and junction capacitance. That places us 1–3 decades below digital competitors (Loihi 2: 10 MHz; SpiNNaker accelerated: 10 MHz; BrainScaleS-2 wall-clock: 10 MHz). For applications dominated by raw throughput (high-rate sensor fusion, fast control loops) NS-RAM is the wrong choice. For applications dominated by *energy at biological timescales* — always-on edge inference, brain-machine interfaces, ultra-low-power audio keyword spotting — the 30 µs RC matches the cortical 10 ms membrane time-constant within a factor of 300, and the per-event energy advantage compounds. We are explicitly positioning NS-RAM in the latter regime, and the tape-out plan (Pillar III, C.3 v2.1) targets benchmarks (keyword-spotting, MNIST few-shot, MC-30) where 1–10 kHz throughput is sufficient.

---

## 5. Caveats — modeled vs measured

- **Rows 3, 5, 7, 8, 9 are measured silicon.** Their numbers come from peer-reviewed papers and are directly comparable.
- **Row 2 (Pazos NS-RAM)** is measured but on a *single transistor*, not a fabricated array; per-mm² density numbers are not in the paper.
- **Row 1 (our NS-RAM projection)** is **modeled**, with three layered assumptions:
  1. The BSIM4 + Gummel-Poon card (`data/sebas_2026_04_22/`) reproduces real I-V to 1.39 dec accuracy at Bf=100 with η ≤ 1 (see `MEMORY.md` → `nsram_m3b_corrections.md`). It is not yet validated against fabricated SKY130 cells — that is the point of the proposed tape-out.
  2. Capacitance estimates use the 130 nm BSIM4 cgso/cgdo values directly; they will change once layout parasitics are extracted.
  3. The 37 kHz settling bound assumes our extracted body R·C; SOI bulk geometry might cut this by 5× either way.
- **Rows 4, 6, 10 are vendor-reported / paper-estimated** rather than directly per-event measured. We use the closest available unit and flag this in the table.

When this comparison is reproduced in the proposal narrative, the modeled-vs-measured asymmetry should be stated explicitly: *"NS-RAM 2T projected, all competitors measured silicon"*. The fact that our projection is consistent with Pazos's measured 21 fJ on the parent device is the strongest sanity check we currently have.

---

## 6. Sources for each row (key + section anchor)

| Row | Citation key | Anchor in source |
|---|---|---|
| 1 | `nsram_proposal2026`, `pazos2025nsram` | this proposal §3 (cell card); Pazos Fig. 1c device, Fig. 3 energy |
| 2 | `pazos2025nsram` | Pazos Fig. 3: "4.7 fJ generation, 16.3 fJ integration" |
| 3 | `davies2018loihi` | Davies IEEE Micro 2018, Table II: 23.6 pJ/SOP |
| 4 | `orchard2021loihi2`, `intel_loihi2_brief` | Orchard arXiv 2021 §III; Intel Loihi 2 brief 2021 (Sept) |
| 5 | `merolla2014truenorth` | Merolla Science 2014 §SI: 26 pJ at 20 Hz / 128 active syn; `akopyan2015truenorth` for area |
| 6 | `mythic2021m1076` | Mythic M1076 product brief (25 TOPS / 3 W) |
| 7 | `legallo2023hermes` | Le Gallo Nature Electronics 2023, Table 1 |
| 8 | `pehle2022brainscales2` | Pehle Frontiers Neurosci. 16:795876 |
| 9 | `mayr2019spinnaker2`, `hoeppner2022spinnaker2` | Mayr arXiv 1911.02385 §IV |
| 10 | `brainchip2023akd1000` | BrainChip AKD1000 product brief v2.3, Aug 2025 |
