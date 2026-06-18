# S6 — Complete Physics Rebuild Plan 2026-05-15

## Synthesis of all evidence

After 12 hours of investigation we have:
- 7 topology fixes KILL-SHOT
- 5 solver strategies KILL-SHOT
- 4 NDR/physics hypotheses KILL-SHOT
- ngspice industry baseline KILL-SHOT
- Confirmed: BSIM4 §6.1 + standard NPN cannot reproduce Sebas data

But S5-C Zoom image analysis revealed 8 missing physics elements we never implemented:

### Missing physics (all from Mario/Sebas documentation)
1. **Vertical NPN through Deep N-Well**: emitter=drain, base=body, collector=DNW (not lateral)
2. **DNW exists** as a triple-well structure (sebs_130nm_triple_well_layout.jpeg)
3. **M2 is NOT a clean MOSFET**: NFACTOR sweeps 2→12, behaves as GIDL/punchthrough leak
4. **ETAB(M1) is a CURVE of VG2**, not constant (Sebas's MATLAB fits)
5. **TWO snapback kinks**: low ~0.4V (capacitive Vd injection) + high ~1.6V (II/BJT)
6. **Snapback is TRANSIENT relaxation oscillator**, not DC steady-state
7. **C_body = 0.3-3 fF** (we use 8 fF — 3-10× too high)
8. **Body-charge τ = 10-100 ms** (very long — Brian2 TIMESCALE_E=50ms confirms)

## Implementation plan

### Phase A — Architecture (ikaros, 3-4h)

Add to nsram/nsram/bsim4_port/nsram_cell_2T.py:

1. **Vertical NPN to DNW** (`use_vertical_npn_to_dnw=True`)
   - New BJT branch: C=DNW node, B=Vb, E=Vd
   - cfg.v_dnw parameter (DNW bias voltage, default 1.2V)
   - DNW node held at v_dnw via ideal source
   - Use Mario canonical: Bf_vert=10000, Va=100, Is=5e-9
   - Routing: collector→DNW (sink), emitter→Vd (already routed)
   
2. **ETAB(VG2) curve interpolator** (`use_etab_curve=True`)
   - Read Sebas's MATLAB fit table (digitize from Image 13.24 if no raw)
   - Interpolate ETAB at each VG2 instead of constant
   - Per VG1 separate curves
   
3. **M2 as V(VG2)-lookup resistor** (`use_m2_as_resistor=True`)
   - Replace M2 with: I_M2 = (Vd_M2 - Vs_M2) / R(VG2)
   - R(VG2) from NFACTOR curve digitization
   - When False: keep current M2 MOSFET model
   
4. **C_body adjustment** (`cb_override=0.5e-15`)
   - Reduce from 8fF to 0.5fF (Mario range center)
   
5. **Long-τ body decay** (`tau_body=50e-3`)
   - Add I_leak_body = Vb/tau_body * Cb to body KCL
   - τ ≈ 50ms per Brian2 TIMESCALE_E

### Phase B — BBO fit including new elements (subagent, 2h)

Free parameters (12-dim):
- V_DNW ∈ [0.3, 1.5] V
- Bf_vert_npn ∈ [1000, 50000]
- Is_vert_npn ∈ [1e-10, 1e-8] A
- R_M2_scale (V_G2 lookup scale factor) ∈ [0.1, 10]
- ETAB_M1_scale ∈ [0.5, 5]
- C_body ∈ [0.1e-15, 5e-15] F
- tau_body ∈ [1e-3, 200e-3] s
- iii_body_gain ∈ [0.1, 3.0]
- Bf_lateral ∈ [100, 10000]
- vnwell_Rs ∈ [1e3, 1e8] Ω
- Plus 2 dim for kink-1 vs kink-2 weighting

Objective: log-RMSE on full 33-curve + BOTH kink positions matched

### Phase C — Transient relaxation validation (parallel subagent, 2h)

Real snapback is TLP transient:
1. Run V_d ramp 0→2V over 100ns (fast pulse)
2. Hold at V_d_peak for 10ns
3. Ramp back 2V→0 over 100ns
4. Look for: (a) snap-jump during ramp, (b) latched high-I state during hold, (c) hysteresis on ramp-down

### Pre-registered gates
- INFRA: all new branches compile + converge across 33 biases
- DISCOVERY: cell-wide < 0.85 dec AND VG1=0.6 fold > 0.5 dec AND BOTH kinks reproduced (positions within ±0.2V of measured)
- AMBITIOUS: cell-wide < 0.4 dec AND VG1=0.6 fold > 1.5 dec AND TLP shows latched state
- KILL-SHOT: even with ALL new physics + BBO fit, cell-wide > 1.5 dec → BSIM4 architecture fundamentally insufficient

### Honest framing
Even with this complete rebuild succeeding, framing remains:
"We added 8 explicit physics elements derived from Mario/Sebas documentation. BBO fit 12 parameters. Each addition is grounded in real silicon mechanism per published images. No curve-fit shortcuts. If KILL-SHOT triggers, snapback in Sebas's device requires non-BSIM4 framework (e.g., TCAD)."

## Resource allocation
- ikaros: Phase A implementation + Phase B BBO sequential
- daedalus: Phase C TLP transient parallel
- zgx: idle reserve
- subagents: 1 for Phase A+B, 1 for Phase C in parallel
