# S3 — Snapback Literature Dive (2026-05-15)

Search via WebSearch executed for three queries from SEVEN_GAPS_PLAN.
Verdict: **≥1 paper cited** (target met).

## Verbatim queries

1. `Mario Lanza NS-RAM 2T snapback floating body 2024 2025` → no hits
   (only the singer Mario Lanza and KAUST faculty page; no NS-RAM-specific
   2T snapback paper from Lanza's group surfaced in the first page).
2. `"2T DRAM" snapback floating body simulation BSIM4 impact ionization parasitic BJT`
3. `"neural state RAM" 2T impact ionization regenerative model parasitic bipolar`

## Most relevant findings

### A. ESD/snapback SPICE modelling (BSIM3/BSIM4 + macro-BJT)

- **Modeling snapback and rise-time effects in TLP testing for ESD MOS
  devices using BSIM3 and VBIC models** — Academia.edu PDF
  (https://www.academia.edu/30298002/). Adds explicit parasitic-BJT
  (VBIC), body resistor, and impact-ionisation generator on top of a
  BSIM3 NMOS. Topology = BSIM-MOS + parallel VBIC NPN with shared base
  node + Iii current source (modulated by Vds). **Differs from our
  approach**: ESD models use VBIC (not Gummel-Poon) for the parasitic
  BJT and add explicit substrate-resistor + base-pinch-resistor; our
  pyport uses a single GP-NPN with `bjt_emitter_to_gnd`. The base-resistor
  / RB_npn knob in `cfg.use_base_resistor` already mirrors this.
- **Source/Drain Junction Partition in MOS Snapback Modeling for ESD
  Simulation** — TechConnect Nanotech briefs 2008 (briefs.techconnect.org
  /wp-content/volumes/Nanotech2008v3/pdf/1483.pdf). Splits the avalanche
  current generation between the S-side and D-side junction. **This is
  exactly the M3b "quasi-2D body" approach (Vb_S / Vb_D split with
  Rb_SD coupling) that lives in `_residuals_quasi2d`** in our cell
  (line 1765 of nsram_cell_2T.py). Confirms the topology is industry-
  standard for snapback ESD models — Plan A wrapper is well-founded.
- **Modeling snapback of LVTSCR devices for ESD circuit simulation
  using advanced BJT and MOS models** — ResearchGate 4297939.
  Treats LVTSCR (lateral SCR) snapback with two BJTs in regenerative
  feedback. Useful homotopy-solver lesson: ramp `iii_gain` from 0 → 1
  to walk Newton through the bifurcation (matches the plan-S2 approach
  with λ-parameter on iii_gain).

### B. 2T DRAM-like floating-body (closest topology match)

- **Physical insights on BJT-based 1T DRAM cells** — ResearchGate 224407575.
  Single-transistor BJT-DRAM relies on the parasitic NPN with impact
  ionisation triggering body charge accumulation. Our 2T topology with
  M2's body wired to floating Vb is essentially "M2 as access" + "M1 as
  storage with floating-body BJT". **Different from us**: 1T DRAM has
  no second access transistor; the 2T NS-RAM uses M2 to gate access to
  the avalanche regime.
- **Avalanche Characteristic of Vertical Impact Ionization** — IJSSST
  Vol-17 (http://ijssst.info/Vol-17/No-34/paper17.pdf). Vertical-MOSFET
  3D-FB DRAM with explicit avalanche term in the body residual. **Same
  KCL structure** as our `_residuals`: KCL_Vb = Iii − Ibe_NPN − Idiode_well = 0.

### C. Bipolar-impact-ionisation neurons (LIF-NS-RAM bridge)

- **Simulation-Based Ultralow Energy and High-Speed LIF Neuron Using
  Silicon Bipolar Impact Ionization MOSFET** — IEEE 9078362 / arXiv
  1909.00669. L-shaped gate BIMOS, BV ≈ 1.68 V, parasitic-BJT positive-
  feedback gives the LIF behaviour. **Same physics as our snapback**:
  floating body → Iii(Vds) → ΔVbs → ΔIds positive loop → snap.
  Different topology (single device, L-gate); same mechanism.
- **Revisited parasitic bipolar effect in FDSOI MOSFETs** —
  ScienceDirect S0038110121001143. FDSOI has stronger parasitic-BJT
  effect than bulk; gives β extraction methodology and circuit-
  application examples.

## What is genuinely different in *our* approach?

- **2T architecture**: M2 is a deliberately-wired access device whose
  body is tied to GND (Sebas LTSpice `2tnsram_simple.asc`, our
  `cfg.m2_body_gnd=True`). Most ESD/1T-DRAM papers use a single device.
- **Per-VG1 BBO**: fitting `(Bf, iii_gain, vnwell_Rs)` separately for
  each VG1 branch (R-46 result, 0.965 dec cell-wide). Literature does
  not do per-bias param-fit for snapback; they use one global card.
- **Quasi-2D body in pyport**: Plan-A wrapper (Vb_S, Vb_D, Rb_SD)
  reproduces the TechConnect 2008 partition approach in a torch-grad
  framework (most refs are ngspice / commercial-simulator-only).

## Take-home for S2 solver design

- Continuation on `iii_gain` (λ ∈ [0, 2]) is supported by LVTSCR paper.
- Industry uses pseudo-arc-length continuation past the limit-point
  (where ∂R/∂V becomes singular); plain Newton + gmin homotopy is
  weaker. Worth implementing arc-length for the high-Vd, high-Vb fold.
- Reading list for next iteration: full PDFs of TechConnect 2008 and
  LVTSCR papers, search for explicit pseudo-arc-length impl examples
  in ngspice or Xyce source.

## Sources

- [Modeling snapback and rise-time effects in TLP — BSIM3 + VBIC](https://www.academia.edu/30298002/Modeling_snapback_and_rise_time_effects_in_TLP_testing_for_ESD_MOS_devices_using_BSIM3_and_VBIC_models)
- [Source/Drain Junction Partition in MOS Snapback (Nanotech 2008)](https://briefs.techconnect.org/wp-content/volumes/Nanotech2008v3/pdf/1483.pdf)
- [Modeling snapback of LVTSCR devices](https://www.researchgate.net/publication/4297939_Modeling_snapback_of_LVTSCR_devices_for_ESD_circuit_simulation_using_advanced_BJT_and_MOS_models)
- [Physical insights on BJT-based 1T DRAM](https://www.researchgate.net/publication/224407575_Physical_insights_on_BJT-based_1T_DRAM_cells)
- [Avalanche Characteristic of Vertical Impact Ionization (IJSSST)](https://ijssst.info/Vol-17/No-34/paper17.pdf)
- [Ultra-Low Energy LIF Neuron with Si BIMOS (arXiv 1909.00669)](https://arxiv.org/abs/1909.00669)
- [Revisited parasitic bipolar in FDSOI MOSFETs (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0038110121001143)
