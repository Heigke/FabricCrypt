# SA3 — Deep Image Extract from Sebas + Mario Slides (O48)

Date: 2026-05-12
Source: `research_plan/oracle_queries/O48_images_deep/openai_response.md` (gpt-5, 248 s, 22 images)
Status: Locked gate PASSED (>=3 topology insights new vs current pyport)

---

## 1. Inventory

22 images processed:
- 21 Mario+Sebas slides (`01..21_*.png/jpeg` from O44) — re-examined for SCHEMATIC content
- `22_sebas_2026_05_02_image-2.png` — NEW image from Sebas, 2 May 2026

No additional images were found anywhere in `data/sebas_2026_*` or `data/nsram_zenodo/`.
`nsram_zenodo` contains only `.cir`, `.txt`, `.csv` simulation files (no figures from the paper).

---

## 2. Per-Image Structural Summary

(Full per-image A–F breakdown lives in the oracle response; the highlights are
indexed here.)

| # | Type            | Topology / Content (one line)                                                                  |
|---|-----------------|------------------------------------------------------------------------------------------------|
| 01 | Param plot     | NFACTOR(M2) vs VG2, 3 VG1 branches (red 0.2, blue 0.4, black 0.6 V)                            |
| 02 | Param plot     | K1(M1) vs VG1, "for all VG2"                                                                   |
| 03 | Param plot     | ETAB(M1) vs VG2, 3 VG1 branches                                                                |
| 04 | Param plot     | BETA0(M1) vs VG2, 3 VG1 branches                                                               |
| 05 | Schematic + IV | 2T cell with explicit **VB–VG2 MOS capacitor**; ID-VD families at VG2=1.4 V and 0.1 V          |
| 06 | Schematic + IV | 2T cell inset; 3 panels of ID-VD vs VG2 sweep, symbols=meas, lines=sim                         |
| 07 | Param page     | 2×2 of (BETA0, ETAB, K1, NFACTOR) — consolidated extraction source                             |
| 08 | Transient      | VD ramp + IV cloud (dynamic trajectory); reference points at start/pre-knee/post-knee          |
| 09 | Meas vs SPICE  | 3 corners: (VG1, VG2) = (0.6, 0.35), (0.4, 0.25), (0.2, 0.0) — thick=meas, thin=sim            |
| 10 | Soma cell      | Cint=102 fF, VD→Vmem mapping, VB=Vspike output, ~0.2 pJ/spike, 111 µm²                         |
| 11 | Soma macro     | Starved-inverter front end, 21 fJ/spike, 60 µm², VG2=Vinteg=0.275, VG1=Vleak=0                 |
| 12 | E/I synapse    | Linear DAC synapses Vw_exc / Vw_inh → soma. Thick-ox stack, Vw range 2.5–3 V                   |
| 13 | Iion model     | Semi-empirical: I_ion = a·exp(b(VD+c)) + d(VD+f)^e ; a,b,d,e,f are PWL(VG); c constant         |
| 14 | Meas vs SPICE  | IB and ID vs VD, VG1 0.00→0.55 V family; **VB = 0 V fixed** (model-isolation step)             |
| 15 | Transient      | VD ramp at VG1=0.3 V, VG2 swept; meas (squares) vs sim (dashed)                                |
| 16 | Process slide  | 130 nm triple-well: deep N-well, isolated P-well; 3×3 µm² min cell                             |
| 17 | 1T cell        | 180 nm thick-ox, area 8 µm², VG up to 0.8 V, VNwell 3 or 5 V; pre-write −1 V widens DR         |
| 18 | 2T cell        | Thick-ox, area 17 µm², VG1<0.8 V, VG2<0.5 V (or floating), VD<3.5 V, VNwell>2 V                |
| 19 | Brian2 LIF     | Behavioural emulation: VG1=550 mV, VG2=500 mV, Cint=170 fF                                     |
| 20 | SNN results    | Confusion matrices: LIF 72% vs Poisson 85%                                                     |
| 21 | 2T + diode     | **Explicit parasitic N-well diode VNwell→VB**; trise sweep 10 µs / 100 µs / 1 ms               |
| 22 | Param page NEW | Source of `three_branch_params_extracted.json` — 2×2 panel (= Image 07 refreshed 2026-05-02)   |

---

## 3. Cross-Image Consistency Map

**Same device, multiple windows:**
- 2T floating-body cell: 05, 06, 08, 09, 13, 14, 15, 18, 21, 22 (10 slides)
- Param extraction summary: 01, 02, 03, 04, 07, 22 (the 4 single-panel slides + composite + new)
- Soma / system use: 10, 11, 12, 19, 20
- Process cross-section: 16, 17, 18

**Stable conventions across the corpus:**
- `VG1` = M1 gate (leak), `VG2` = M2 gate (integration / bleed)
- `VD` = M1 drain (= Vmem in soma circuits)
- `VB` = floating P-body (= M2 drain = Vspike in soma circuits)
- `VS = VNEG = 0 V`; `VNwell` separately biased
- Three colour code: red = VG1 0.20 V, blue = 0.40 V, black = 0.60 V

**Discrepancies (benign):**
- Image 05 uses VG2 up to **1.4 V** (older / overdrive experiment); later slides cap VG2 at 0.5 V
- Image 14 pins **VB = 0 V** to isolate the I_ion law — not a contradiction to floating-body operation elsewhere
- Image 22 supersedes 01–04 / 07 as the canonical parameter page (latest extraction, 2026-05-02)

**`image-2.png` confirmed identity:**
The NEW image is the **four-panel master parameter page** (BETA0, ETAB top row; K1, NFACTOR bottom row) — i.e., the direct source PNG behind `three_branch_params_extracted.json`. It is structurally the same as Image 07 but with refreshed numerics. Branch identities match the JSON exactly (red 0.20, blue 0.40, black 0.60 V VG1 branches).

---

## 4. Topology Insights — NEW vs Current Pyport

Current pyport (`scripts/z70_*`, `z83_*`, `nsram_fpga_bridge.py`, etc.) models the cell as **BSIM4 (M1) + Gummel-Poon parasitic BJT (M2 leak) + 3-branch I-V extraction**. It does NOT include explicit VB-node capacitances, the N-well diode, or input-driver pulse shaping.

Locked gate requires >=3 new insights. We have **7**:

1. **Parasitic N-well diode VNwell → VB is a separate circuit element**, not a transport-model add-on (Image 21 explicit, 16/17 implicit). Its junction capacitance + voltage-dependent leakage materially shifts the firing knee and is the dominant source of ramp-rate sensitivity (Images 08, 15, 21). **NOT in our pyport.**

2. **Deliberate VB–VG2 MOS coupling capacitor** (Image 05 schematic, drawn as an explicit cap) injects gate transitions onto the floating body and sets the spike rise time. This is a *designed* element, not a parasitic. **NOT in our pyport.**

3. **VB is an observable output node (Vspike)**, not just an internal state. In the soma cells (Image 10), M2's drain (= VB) is the read-out terminal. Our 3-branch I-V model treats VB only as an internal Vb-clamp parameter and never reports it; compact models that lack this node cannot reproduce self-reset waveforms.

4. **VD ↔ Vmem mapping inverts the role of the "drain ramp"**. In Image 10 the M1 drain is wired to the integrating capacitor Cint (102 fF) — i.e., what device characterisation calls a swept VD is in-circuit the *membrane potential driven by I_excit*. Our pyport currently sweeps VD as an external source; closed-loop integration with Cint at the drain is missing.

5. **NFACTOR(M2) depends on BOTH VG2 AND VG1** (Images 01, 22). VG1 couples into M2 only through VB. This proves the body node is a *shared state* between the two devices, not an isolated per-device parameter. Our current parameter table treats NFACTOR as VG2-only would lose the cross-coupling.

6. **Starved-inverter front-end at ~1 V swing** (Images 11–12) is part of the firing model. The soma fires correctly only with that pulse-shape input; our bridge tests have been driving with continuous-time signals. The starved inverter sets effective input impedance and rise-time of I_excit pulses.

7. **VNwell bias domain (≥2–5 V) + thick-oxide constraint** (Images 16–18) defines the legal operating window. Image 05's VG2=1.4 V case is outside the thick-ox regime used in 18/21/22 — when comparing old vs new datasets we must mark which oxide flavour was used. Our pyport does not track oxide flavour at all.

---

## 5. Implications for `pyport_v5`

Minimum to absorb before next fit cycle:
- Add a **D_NWELL** SPICE diode (with Cj) between VNwell and VB, parameters from Image 21's ramp-rate data
- Add an explicit **C_GS_M2 = C(VB, VG2)** coupling cap (Image 05); start at MOS-cap default (W·L·Cox of M2 gate area) and let the dynamic VD-ramp data refine it
- Expose VB as a **named output** of the cell model, not a hidden internal state — enables direct comparison to Vspike traces (Image 10)
- For soma-mode tests, wrap the cell in a Cint=102 fF loop at VD, drive with an I_excit current source, not a V_DD source

---

## 6. Pitfalls / Caveats Surfaced

- **Old VG2=1.4 V data (Image 05) is not in the thick-ox window**; do not include in the joint fit without flagging.
- **Image 14's VB=0 V** is a modelling crutch for isolating I_ion — do NOT confuse with floating-body operation.
- **Brian2 numbers (Image 19, Cint=170 fF) do not match silicon (Cint=102 fF, Image 10)** — these are different abstraction levels.
- Image 20 (LIF 72% vs Poisson 85%) — **the Poisson reference outperforms LIF**; not a "feature of NS-RAM", just a sanity check of the SNN training pipeline.
