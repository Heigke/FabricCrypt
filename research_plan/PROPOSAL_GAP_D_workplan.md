# PROPOSAL — Gap D Workplan
## NS-RAM Co-Design: Silicon-to-Algorithm Bridge — 36-month workplan (WP1–WP4)

**Document type:** funder-ready workpackage roadmap, drop-in for Mario's
Marie-Skłodowska-Curie / Vinnova / NRF / EU FET-Open application.
**Authors:** E. Bergvall (ENIMBLE, simulator lead) · R. Luciani (Julia/Enzyme
cross-validation) · S. Pazos (NUS / KAUST, silicon characterisation) ·
M. Lanza (NUS / KAUST, PI, fab access, Nature 2025).
**Date:** 2026-05-19. **Status:** draft v1, to be tightened with Mario after
2026-06-01 brief delivery.
**Scope:** 36 months, four parallel-streamed work packages, single
coordinated tape-out.
**Headline ask:** **€3.10 M total** for 36 months (range €2.6 M–€3.4 M
depending on MPW slot pricing and PhD/postdoc host-country indirect rates).

---

## 0. Context and starting position (what the funder gets on day one)

The consortium does not start from a blank page. As of 2026-05-19 the
following assets exist and are referenced in the proposal:

- **Silicon (Pazos / Lanza, NUS-KAUST).** A working 2T NS-RAM cell
  fabricated in standard bulk CMOS (Nature 2025), with 33-bias DC
  characterisation already in hand for the nominal device geometry
  (`data/sebas_2026_04_22/`). Floating-body / parasitic-NPN regime
  empirically confirmed; VG2 as regime-selector validated 2026-04-23.
- **Simulator (Bergvall, ENIMBLE).** Differentiable, GPU-batched PyTorch
  port of the BSIM4 channel + Gummel–Poon / VBIC NPN, body integration,
  pseudo-transient DC solver. Current honest DC fit: **median 1.163 dec
  log₁₀|Id| on the 33-bias set, fwd+bwd, n=66** (`results/Pillar_I_C3_jts_tat/verdict.md`).
  Two prominent parallel-conduction candidates (Pazos NPN, BSIM4 §10.1
  JTS-TAT) are internally falsified at default parameters.
- **Network demonstrations.** Seven pre-registered network-level tasks
  pass on the calibrated cell (reservoir on Mackey-Glass; HD encoding on
  UCI-HAR; associative-memory binder; predictive coding on NAB; 1/f
  stochastic-computing RNG passing NIST SP800-22; 16-tap NS-RAM LMS
  equaliser; KWS→ECG edge cascade). Topology zoo: 104/120 cells
  edge-of-chaos, broadband NARMA r² up to 0.968.
- **Engineering deliverables.** Verilog-A export skeleton, internal
  ML-emulator (0.07 dec inside VG1∈[0.2,0.6] interpolation envelope),
  PyPI release `nsram v0.12.0`, and an FPGA 128-neuron reference reservoir
  hitting 81% classification, MC=2.67.

**Honest open items the workplan closes.**
(i) ~1.2 dec DC fit is *qualitative-grade* for current-budget design —
sub-1.0 dec needs experimental discrimination (T-sweep, W-scaling,
negative-Vd); (ii) two of nine dynamic-behaviour gates remain open
(self-reset, free oscillation); (iii) no tape-out yet of an ENIMBLE-cell-
card variant; (iv) task-level demonstrations are on a single calibrated
cell, no folded device-to-device variation.

---

## 1. Workplan overview

| WP  | Title                                                  | Months   | Lead     | Total budget (€) | Critical deliverable |
|-----|--------------------------------------------------------|----------|----------|------------------|----------------------|
| WP1 | Silicon characterisation & mechanism identification    | 1–9      | Pazos    | 0.45 M           | D1: peer-reviewed methods paper + cell-card v3 with CIs |
| WP2 | Algorithmic-primitive demonstration & co-design        | 4–18     | Bergvall | 0.62 M           | D2: open-source SDK + 3 task demos + IEDM/IJCNN paper |
| WP3 | Prototype chip tape-out                                | 12–30    | Lanza    | 1.55 M           | D3: working 128-neuron / 32×32-synapse prototype + characterisation report |
| WP4 | System integration & dissemination                     | 18–36    | Bergvall | 0.36 M           | D4: USB/PCIe demo board + edge-AI demo + commercialisation pitch |
| —   | Coordination, indirects, contingency                   | 1–36     | Lanza    | 0.12 M           | Project management, MoU enforcement, audit trail |
| **Total** | **36 months consortium**                         |          |          | **≈3.10 M**      | |

All budgets are *all-in* using the EU FET-Open 2025 cost model
(~€8 k per person-month for PhD; ~€10 k for postdoc; ~€11 k for senior
researcher; ~€14 k for PI 0.2 FTE; foundry MPW estimated at €350 k for
180 nm and €600 k for 130 nm — **flagged as confirmation required**).

Critical-path dependency: **WP3 tape-out depends on WP1 finalising
process-node selection (180 nm vs 130 nm) and on WP2 finalising the
cell-parameter recommendation** before month 12. WP1+WP2 are scheduled
so that the union of (mechanism shortlist closed, task-level Pareto
known) lands at month 9. WP4 starts at month 18 in parallel with chip
fab.

```
Month   1   3   6   9   12  15  18  21  24  27  30  33  36
WP1     ████████████████░░░░░░
WP2          ████████████████████████████░░░░░░
WP3                       ████████████████████████████
WP4                                  ████████████████████████
```

---

## 2. WP1 — Silicon characterisation & physical-mechanism identification
**Lead:** S. Pazos. **Co-lead:** M. Lanza (fab + lab access). **Months 1–9.**

### 2.1 Motivation
The simulator currently fits DC to **1.163 dec median** because two of the
prominent textbook parallel-conduction candidates (Pazos NPN, BSIM4
JTS-TAT) are internally falsified, and the residual gap cannot be
discriminated without extra silicon measurements. WP1 produces the
discriminating dataset.

### 2.2 Milestones

- **M1.1 (months 1–4) — Sebas data extension.**
  - T-sweep at the 250 nA diagnostic bias (VG1=0.6, VG2=−0.05, Vd=0.05),
    T ∈ {220, 250, 300, 350, 400} K, ≥10 devices.
  - W-scaling: W ∈ {0.18, 0.36, 0.72, 1.44} µm at L=180 nm (or L_drawn
    confirmed by Lanza), ≥10 devices per geometry.
  - Negative-Vd: one Vd=−0.5 V point at two (VG1, VG2) corners (true ±Vd,
    not sweep hysteresis).
  - Dense-VG1 grid: VG1 ∈ {0.2, 0.3, 0.4, 0.5, 0.6} V (current data has
    gaps that hurt the C2 floating-body sub-Vt fit).
  - Paired-pulse: 5 VG2 levels × 5 Δt values for SRH↔Tsodyks-Markram
    validation (the *second* novel finding of the collaboration).

- **M1.2 (months 3–6) — Differentiable refit at each (T, W).**
  - Apply the existing GPU-batched fit pipeline to each (T, W) cell;
    extract dominant residual mechanism per regime.
  - Cross-validate against the C2 (self-consistent floating-body sub-Vt)
    and C4 (measurement-artifact / contact-resistance) candidates that
    are currently open.

- **M1.3 (months 6–9) — Lock mechanism shortlist, publish.**
  - Submit methods paper ("Honest-negative falsification of two textbook
    parallel-conduction candidates in 2T NS-RAM, with a temperature- and
    geometry-resolved mechanism shortlist") to *IEEE TED* or *Nature
    Electronics*.
  - Issue **cell-card v3** with bootstrap confidence intervals on every
    extracted parameter — this is the artefact the tape-out (WP3) loads.

### 2.3 Deliverables

- **D1.1 — Extended characterisation dataset.** ≥600 IV curves
  (5 T × 4 W × ≥10 devices × 3 regimes), CSV + raw + Verilog-A regression
  decks. Released under NDA-first, public after 90-day embargo per
  collaboration MoU (`docs/NSRAM_samarbetsplan_Mario_Sebas_2026-04-23.txt`).
- **D1.2 — Peer-reviewed methods paper.** Pazos first author, Bergvall
  co-author, Lanza senior. Target *IEEE TED* H2-2026.
- **D1.3 — Cell-card v3.** Tape-out-actionable parameter file with CIs.

### 2.4 Person-months (WP1)

| Role | PM | All-in € | Subtotal |
|------|----|----|----|
| Pazos (senior researcher, 0.5 FTE × 9 mo) | 4.5 | 11 k | 50 k |
| Lanza (PI, 0.1 FTE × 9 mo, in-kind through host institution) | 0.9 | 14 k | 13 k |
| WP1 PhD #1 (characterisation, 1.0 FTE × 9 mo) | 9.0 | 8 k | 72 k |
| WP1 postdoc (refit pipeline, 0.5 FTE × 6 mo) | 3.0 | 10 k | 30 k |
| Bergvall (simulator-side support, 0.3 FTE × 6 mo) | 1.8 | 11 k | 20 k |
| Wafer time, probe-station, cryostat, He budget | — | — | 180 k |
| Travel (KAUST↔NUS 2×, conference 1×) | — | — | 15 k |
| Open access, replication archive | — | — | 5 k |
| Indirect (institutional, ~15%) | — | — | 65 k |
| **WP1 subtotal** | **19.2** | | **≈450 k** |

### 2.5 Critical-path dependencies
- Probe-station + cryostat slot at NUS or KAUST (Lanza confirms by month 0).
- ENIMBLE GPU-batched fit pipeline already in place — no dev risk.
- M1.3 must close before WP3 layout starts (month 12).

### 2.6 Risk register (WP1)

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Cryostat slot delayed >2 mo | Med | High | Pre-book at proposal start; KAUST backup; analytical T-extrapolation as interim |
| Device-to-device variability >2σ exceeds fit | High | Med | 10 cells per condition; report distributional rather than per-cell fit |
| Mechanism remains unidentified after T+W discrimination | Med | Med | Honest-negative paper still ships; WP3 proceeds on best-current cell-card with explicit CIs |

### 2.7 Success metrics (WP1)
- ≥600 curves measured, 5T × 4W × 3 regimes.
- Cell-wide DC fit ≤ **0.7 dec** median on extended set (gate from
  current 1.163 dec); if not met, **honest-negative paper still
  publishable** — the falsification result has standalone value.
- Cell-card v3 with CIs released to consortium by month 9.

---

## 3. WP2 — Algorithmic-primitive demonstration & co-design
**Lead:** E. Bergvall. **Co-lead:** R. Luciani (Julia/Enzyme cross-val).
**Months 4–18.**

### 3.1 Motivation
Seven network demonstrations already pass on the calibrated cell, but
each ran end-to-end on a single nominal-parameter cell with no
device-to-device variation, no shared-rail noise, no silicon-grade 1/f.
WP2 (a) scales each demo to 1024+ cells with realistic variation,
(b) closes the inverse-design loop into WP1 fab variants,
(c) ships an open-source SDK that lets external groups reproduce.

### 3.2 Milestones

- **M2.1 (months 4–9) — Task-level demos at 1024 cells.**
  - **SHD (Spiking Heidelberg Digits, audio):** target ≥85% test accuracy,
    energy ≤200 pJ/inference (extrapolated from cell sub-pJ/spike).
  - **Keyword spotting (Google Speech Commands v2 subset):** target ≥92%
    test accuracy on the 12-class problem, < 1 µJ/inference.
  - **Anomaly detection on NAB:** target F1 ≥ 0.4 with sub-pJ/event.
  - Baselines: software fp32 reservoir, Loihi-2 published numbers,
    TrueNorth published numbers, Mythic published numbers. Numbers
    reported in Gap-B baseline table (separate document).

- **M2.2 (months 6–12) — Primitive validation in silico.**
  - STDP via correlated VG2 pulse pairs (uses paired-pulse data from
    M1.1).
  - Homeostasis via slow body-discharge time constant.
  - Time-multiplexed reservoir (single cell, time-stretched).
  - Each primitive pre-registered with falsifiable gate.

- **M2.3 (months 12–18) — Co-design loop closed.**
  - For each task: compute ∂accuracy/∂(cell-parameter) via the
    differentiable simulator; rank parameter sensitivities; **feed back
    to WP1** as a discrimination request (e.g. "an additional W=2.88 µm
    point would tighten the predicted accuracy CI by X%").
  - This is the unique-IP loop the collaboration is built on.

### 3.3 Deliverables

- **D2.1 — Open-source SDK.** `nsram v1.0` on PyPI; Python + Verilog-A;
  documented API; three reference task notebooks; Apache-2.0 license
  with copyright "© 2026 ENIMBLE Solutions AB" (per MoU IP-inventory).
- **D2.2 — Pareto-frontier paper.** Submitted to IEDM 2027 or IJCNN
  2027. Bergvall first author, Pazos and Lanza co-authors.
- **D2.3 — Co-design loop report.** Internal artefact feeding WP1 month-9
  cell-card v3.

### 3.4 Person-months (WP2)

| Role | PM | All-in € | Subtotal |
|------|----|----|----|
| Bergvall (simulator lead, 0.8 FTE × 14 mo) | 11.2 | 11 k | 123 k |
| WP2 PhD #2 (task demos, 1.0 FTE × 14 mo) | 14.0 | 8 k | 112 k |
| WP2 postdoc (algorithm co-design, 1.0 FTE × 12 mo) | 12.0 | 10 k | 120 k |
| Luciani (Julia/Enzyme cross-validation, 0.3 FTE × 6 mo) | 1.8 | 11 k | 20 k |
| Pazos (silicon-side feedback, 0.2 FTE × 6 mo) | 1.2 | 11 k | 13 k |
| Compute (cloud GPU bursts for ablations) | — | — | 25 k |
| Travel (IEDM/IJCNN + 1 consortium) | — | — | 18 k |
| Open access, dataset hosting | — | — | 7 k |
| Hardware contingency (FPGA boards, USB scope) | — | — | 12 k |
| Indirect (~25% for academic host on PhD/postdoc) | — | — | 170 k |
| **WP2 subtotal** | **40.2** | | **≈620 k** |

### 3.5 Critical-path dependencies
- WP1 cell-card v3 needed by **month 12** to lock variation distributions
  used in M2.1 scaled task demos. Until then, M2.1 uses the current 1.163-
  dec cell card with a flagged caveat.
- M2.3 co-design loop output feeds back to **WP3 layout** at month 12.

### 3.6 Risk register (WP2)

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| SHD/KWS accuracy << 85% at 1024 cells under realistic variation | Med | High | Topology-zoo already shows broadband NARMA r²=0.968; pre-register hardware-aware training; fall back to 4096 cells (still within GPU envelope) |
| GPU port hits 2×10⁴-cell ceiling (vendor driver bug currently known) | Med | Med | Block-matmul workaround in place; cloud GPU burst budget; multi-GPU partition path |
| SDK adoption insufficient for impact metric | Med | Low | Hosted Colab notebooks; conference tutorial submission; "fit-as-a-service" commercial leg as backup |

### 3.7 Success metrics (WP2)
- Three task demos at ≥1024 cells with pre-registered gates passing.
- nsram v1.0 SDK with ≥3 external research-group adopters by month 18.
- One IEDM/IJCNN paper accepted.

---

## 4. WP3 — Prototype chip tape-out
**Lead:** M. Lanza. **Co-lead:** S. Pazos (DfM). **Months 12–30.**

### 4.1 Motivation
The collaboration's whole point is that NS-RAM is fabricated in
*standard bulk CMOS* — so a prototype tape-out validates the silicon-
to-algorithm bridge end-to-end at modest cost compared with novel-
material approaches (RRAM, PCM, memristor). WP3 produces an ENIMBLE-
cell-card-driven prototype that exercises the cell-parameter
recommendations from WP1+WP2.

### 4.2 Milestones

- **M3.1 (months 12–18) — Layout.**
  - 128-neuron block + 32×32 synapse array.
  - Shared-bulk-rail topology with explicit per-block isolation.
  - Process node confirmed by WP1: **default 180 nm bulk CMOS**;
    fallback 130 nm (where Sebas's existing data was taken) if WP1 M1.3
    recommends matching the existing characterised process.
  - **Flag — requires confirmation:** MPW slot at TSMC 180BCD or UMC
    180nm via Europractice; budgeted at €350 k. **If 130 nm is selected,
    MPW cost rises to ~€600 k and the WP3 budget needs to be revised up
    or the chip area down.**
  - DfM checklist co-signed by Pazos and the foundry PDK owner.

- **M3.2 (months 18–24) — Tape-out and fab.**
  - Submission to MPW slot; expected 3-month fab turnaround per published
    Europractice timeline.
  - Bring-up board (analog frontend, FPGA controller, USB telemetry).

- **M3.3 (months 24–30) — Characterisation vs spec; second-silicon
  contingency.**
  - Compare measured to simulator-predicted spike statistics, MC,
    waveform classification (FPGA reference 81%, 128 neurons, MC=2.67).
  - Gate: silicon classifies on a *re-pre-registered* benchmark within
    ±10 percentage points of the simulator prediction.
  - If silicon misses gate by >15 pp on >2 of 3 benchmarks, second-
    silicon respin in month 27 (contingency budget held).

### 4.3 Deliverables

- **D3.1 — Layout deliverable (GDS-II + DRC/LVS clean).** Co-signed by
  ENIMBLE (cell-card) and Lanza-group (DfM) at month 18.
- **D3.2 — First silicon characterised.** Spike statistics, MC, three
  benchmark task accuracies, energy/inference measured on-chip.
- **D3.3 — Characterisation report.** Lanza first author, Bergvall and
  Pazos co-authors, submitted to *Nature Electronics* or *VLSI 2028*.

### 4.4 Person-months (WP3)

| Role | PM | All-in € | Subtotal |
|------|----|----|----|
| Lanza (PI, 0.2 FTE × 18 mo) | 3.6 | 14 k | 50 k |
| Pazos (DfM + char., 0.6 FTE × 18 mo) | 10.8 | 11 k | 120 k |
| WP3 PhD #3 (layout, 1.0 FTE × 18 mo) | 18.0 | 8 k | 144 k |
| Bergvall (cell-card sign-off, 0.2 FTE × 6 mo) | 1.2 | 11 k | 13 k |
| WP3 PhD #1 (characterisation continuation, 0.5 FTE × 6 mo) | 3.0 | 8 k | 24 k |
| **MPW slot (180 nm, baseline)** — flag: requires Europractice quote | — | — | 350 k |
| Bring-up board PCB + assembly | — | — | 35 k |
| Wafer probe, packaging | — | — | 60 k |
| Test equipment (PXI / parametric) | — | — | 80 k |
| Second-silicon contingency (50%-funded) | — | — | 200 k |
| Travel (foundry visit, DfM review × 2) | — | — | 22 k |
| Indirect (~25%) | — | — | 450 k |
| **WP3 subtotal** | **36.6** | | **≈1.55 M** |

### 4.5 Critical-path dependencies
- WP1 cell-card v3 + WP2 co-design recommendation locked at month 12.
  **No tape-out submission until both land.**
- Europractice MPW slot calendar — locks 3 months ahead of submission;
  confirm with Lanza-group lab manager by month 9.
- Foundry PDK access — Lanza group has standing access to TSMC 180BCD;
  confirmed in-kind via NUS-KAUST agreement (no extra cost).

### 4.6 Risk register (WP3)

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| MPW slot pushed by >3 mo | Med | High | Apply to two slots (TSMC + UMC); analytical waveform from simulator as interim demo |
| Cell-card v3 from WP1 disagrees with WP2 recommendation by >2σ | Low | High | Joint sign-off review at month 11; if disagreement, conservative cell-card (intersection of CIs) used and gate widened from ±10 pp to ±15 pp |
| First silicon misses spec >15 pp on >2 of 3 benchmarks | Med | Med | Second-silicon contingency budgeted at 50%; characterisation report still ships as honest result |
| 130 nm node selected at month 9 → budget overrun | Med | High | Pre-flag in proposal; trigger requires sponsor agreement on +€250 k or area reduction to 64-neuron block |

### 4.7 Success metrics (WP3)
- First silicon characterised; spike waveform, MC, classification on
  ≥1 benchmark within ±10 pp of simulator prediction.
- Energy per inference measured on-chip and reported in pJ.
- One VLSI / Nature-Electronics-class publication submitted.

---

## 5. WP4 — System integration & dissemination
**Lead:** E. Bergvall (with Lanza-group integration support). **Months 18–36.**

### 5.1 Motivation
A characterised chip without a demo board reaches only the hardware-
community audience. WP4 turns the chip into a (i) USB/PCIe-attached
demo board, (ii) one edge-AI demo with measured (not extrapolated) power,
(iii) industry-facing white paper for foundry-IP licensing discussion.

### 5.2 Milestones

- **M4.1 (months 18–24) — USB/PCIe demo board.**
  - FPGA bridge (we have the architecture: 128-neuron NS-RAM reservoir
    bitstream and MAC bridge already characterised at 14/19 PASS, ACF
    0.85; see internal MEMORY.md entries on z2272-z2284 and z2206).
  - USB telemetry, host SDK integration with nsram v1.0.

- **M4.2 (months 24–32) — Edge-AI demo.**
  - KWS or motion-classification on the chip with **measured** power
    (not extrapolated from cell-level numbers).
  - Comparison vs MCU baseline (e.g. STM32H7 ARM with CMSIS-NN), Loihi-2
    published numbers, Mythic published numbers.
  - Demo running at conference booths (NeurIPS hardware, VLSI, ISSCC).

- **M4.3 (months 30–36) — White paper + commercialisation.**
  - White paper: "NS-RAM as a standard-CMOS neuromorphic substrate —
    process node, energy budget, foundry-IP licensing".
  - Industry partner discussions (Sweden: Imec-equivalent; Singapore:
    A*STAR IME; one EU foundry).
  - Pitch deck for the spin-out IP licensing path (ENIMBLE side,
    behind MoU).

### 5.3 Deliverables

- **D4.1 — Demo system.** Hardware (board + chip) + firmware + host SDK
  binary release.
- **D4.2 — Edge-AI demo paper.** Submitted to NeurIPS hardware track
  or DATE 2029.
- **D4.3 — Commercialisation pitch.** White paper + investor / industry
  deck; royalty split per MoU.

### 5.4 Person-months (WP4)

| Role | PM | All-in € | Subtotal |
|------|----|----|----|
| Bergvall (system integration, 0.5 FTE × 18 mo) | 9.0 | 11 k | 99 k |
| WP4 postdoc (firmware / demo, 0.5 FTE × 12 mo) | 6.0 | 10 k | 60 k |
| WP3 PhD #3 (board co-design, 0.3 FTE × 12 mo) | 3.6 | 8 k | 29 k |
| Pazos (chip-side debug, 0.1 FTE × 12 mo) | 1.2 | 11 k | 13 k |
| PCB iterations + components | — | — | 25 k |
| Travel (NeurIPS hardware, VLSI, ISSCC, 2 industry) | — | — | 32 k |
| Demo enclosure, conference booth | — | — | 8 k |
| White-paper editing / design | — | — | 6 k |
| Indirect (~25%) | — | — | 90 k |
| **WP4 subtotal** | **19.8** | | **≈360 k** |

### 5.5 Critical-path dependencies
- WP3 first silicon characterised at month 30. M4.2 cannot start before
  M3.3 closes. Demo board PCB can be designed against the chip-pad-list
  earlier (M4.1 in months 18–24).

### 5.6 Risk register (WP4)

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Chip yield too low for live demo | Med | High | Use second-silicon batch; simulator-driven demo (with banner "simulator, measured power") as backup |
| Industry partner timeline >36 mo | High | Low | White paper still ships; commercialisation continues post-grant via ENIMBLE spin-out leg |
| Demo board fails EMC for conference floor | Low | Med | Standard FPGA-dev-board reference; shielded enclosure |

### 5.7 Success metrics (WP4)
- Working demo board running KWS or motion-classification on the
  prototype chip, measured pJ/inference.
- ≥2 industry-partner letters of interest by month 36.
- One conference demo and one paper.

---

## 6. Personnel summary (all WPs)

| Role | Total PMs | Comment |
|---|---|---|
| Lanza (PI, 0.2 FTE avg) | 7.2 | Fab access, MPW relations, senior author |
| Pazos (co-PI, 0.5 FTE avg) | 18 | Silicon characterisation, DfM, paper 1 + co-author rest |
| Bergvall (simulator lead, 0.6 FTE avg) | 22 | nsram simulator, SDK, demo system |
| Luciani (collaborator, 0.3 FTE × 6 mo) | 1.8 | Julia/Enzyme cross-val |
| PhD #1 (characterisation, 1.0 FTE × 15 mo) | 15 | WP1 + WP3 continuation |
| PhD #2 (algorithm demos, 1.0 FTE × 14 mo) | 14 | WP2 task demos |
| PhD #3 (layout / DfM, 1.0 FTE × 30 mo) | 30 | WP3 + WP4 board |
| Postdoc — refit pipeline (0.5 FTE × 6 mo) | 3 | WP1 |
| Postdoc — algorithm co-design (1.0 FTE × 12 mo) | 12 | WP2 |
| Postdoc — firmware / demo (0.5 FTE × 12 mo) | 6 | WP4 |
| **Total PMs** | **≈129** | |

The group size matches Lanza's published group structure (≥3 PhDs, ≥2
postdocs at NUS / KAUST). PhDs are 36-mo contracts spanning multiple WPs
to give continuity; postdocs are shorter and WP-targeted.

---

## 7. Total ask and contingency

| Line | € |
|---|---|
| WP1 subtotal | 450 k |
| WP2 subtotal | 620 k |
| WP3 subtotal (incl. €350 k MPW baseline + €200 k contingency) | 1 550 k |
| WP4 subtotal | 360 k |
| Project coordination + audit + MoU enforcement | 60 k |
| Cross-WP overhead (legal IP filing, two arxiv preprints, etc.) | 60 k |
| **TOTAL ASK** | **≈3.10 M** |

**Range justification.** Lower bound €2.6 M assumes (i) 180 nm MPW slot
secured at €350 k, (ii) no second-silicon respin in WP3, (iii) PhDs
hosted in NUS / KAUST rather than higher-indirect EU institution. Upper
bound €3.4 M assumes 130 nm MPW (+€250 k), second-silicon respin
(+€200 k), and EU host-institution indirect at 25% across the board.

**Funding-vehicle suitability.**
- **EU FET-Open / EIC Transition** (€2.5–€4 M range): excellent fit.
- **Marie Skłodowska-Curie Doctoral Network** (≥10 PhDs, ≥€4 M): would
  require partner expansion (>2 hosts); fits with ENIMBLE + NUS + KAUST
  + one EU foundry partner.
- **Vinnova UDI / Strategic Innovation** (Sweden, up to €2 M, requires
  matching): ENIMBLE side only, would fund WP2 + WP4 (≈€1 M) with
  industry matching.
- **NRF Singapore Competitive Research Programme** (~SGD 5 M, 5 yr):
  Lanza-group lead, would absorb WP1 + WP3.

A blended package (NRF for WP1+WP3, EIC Transition for WP2+WP4) is the
most likely realistic configuration given Lanza's existing NRF/MOE
pipeline and ENIMBLE's EIC eligibility (Swedish SME).

---

## 8. Top consortium-level risks (above-WP)

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Process-node mismatch between WP1 char-data (130 nm) and WP3 MPW slot (180 nm default) | Med | High | WP1 M1.3 must lock node before WP3 layout; pre-fund a 1-month bridging study at month 9 |
| Mechanism remains unidentified after WP1 → tape-out lands on a still-1.2-dec cell card | Med | Med | Cell-card v3 with explicit CIs is publishable; WP3 gates widened from ±10 pp to ±15 pp; honest-negative branding maintained |
| MoU IP-split fails between ENIMBLE and NUS-KAUST | Low | High | Pre-signed MoU draft already in `docs/NSRAM_samarbetsplan_Mario_Sebas_2026-04-23.txt`; foreground default joint with field-split; arxiv preprint of SRH↔TM bridge as date-stamp insurance |
| Key personnel (Bergvall, Pazos) move institutions | Low | High | 36-mo contracts; succession plan documented; nsram v1.0 SDK is the institution-portable artefact |
| External (geopolitical, foundry-supply) shock affects MPW access | Low | Med | Two-foundry strategy (TSMC + UMC); Europractice intermediation provides EU-side neutrality |

---

## 9. What this workplan deliberately does **not** promise

(Per NO-CHEAT discipline — the document we will defend in front of a reviewer.)

- We do not promise sub-1.0 dec DC fit before WP1 M1.3 lands. The current
  baseline is **1.163 dec**, and the gap-closing depends on experimental
  discrimination we have not yet run.
- We do not promise that NS-RAM beats software ESN on like-for-like
  algorithmic accuracy. The claim is **energy-per-inference floor in
  standard CMOS** — to be measured in WP4, not assumed.
- We do not promise a fully autonomous relaxation oscillator on the
  present cell. That dynamic-behaviour gate is open and the closure
  depends on body-resistance / body-capacitance identification from WP1.
- We do not promise a particular MPW timeline without Europractice
  confirmation. The 30-month tape-out window is the Europractice-
  published median; slips of ≤3 months are budgeted, beyond that the
  WP3 plan requires sponsor agreement.
- We do not promise SDK adoption beyond the consortium without
  external uptake metrics. Three external research-group adopters by
  month 18 is the gate; below that the SDK becomes an internal artefact.

---

## 10. Closing statement (for the funder)

The collaboration starts with **three things no competing group has at
once:** a Nature-2025 NS-RAM cell in standard CMOS (Lanza / Pazos), a
GPU-batched differentiable simulator with seven passing network demos
(ENIMBLE), and an internally falsified, honest-negative shortlist of
the physical-mechanism candidates that *would* close the remaining DC-fit
gap. The 36-month plan converts those three assets into a characterised
prototype chip and a system-level demo, with explicit, named honest-
negative branches at every gate. The total ask of **≈€3.10 M** sits
inside the EU FET-Open and NRF CRP envelopes, with a blended-funding
fallback for the realistic Singapore + EU configuration.

The single sentence we will defend on the call:

> *We are not promising better software, better materials, or a new
> physics. We are promising a tape-out-actionable cell-card derived from
> a differentiable bridge between published silicon and the algorithms
> the silicon is supposed to run — and the bridge already runs.*

---

*Document path:* `research_plan/PROPOSAL_GAP_D_workplan.md`
*Companion TeX (input-able into `main-4.tex`):*
`research_plan/PROPOSAL_GAP_D_workplan.tex`
*MoU reference:* `docs/NSRAM_samarbetsplan_Mario_Sebas_2026-04-23.txt`
*Current honest baseline:* `research_plan/MARIO_BRIEF_v4.8_draft_2026-05-19.md`
