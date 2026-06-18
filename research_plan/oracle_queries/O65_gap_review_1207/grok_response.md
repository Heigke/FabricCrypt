# grok response (grok-4-latest) — 110s

### Q1 — gate-crossing risk
(248 words)

The 1.42 dec median is likely real and not purely compensatory, given the stacked fixes (R-20 to R-41) address distinct physical mechanisms (e.g., Iii scaling via binunit, parasitic diode discharge). It represents genuine convergence toward ngspice physics, closing a 3-dec Iii gap (R-37) and over-pumping (R-39 BJT retune), with R-41's pdiode enabling stronger body discharge. However, residual VG1=0.60 elevation (1.87 dec) suggests incomplete closure—possibly absorbing minor over-corrections in subthreshold or impact ionization.

- **Sebas's pdiode card (Js, n, area)**: This is honest and not "fitting" like R-39's tuned BJT params. It's direct ground truth from Sebas's deck, enabling physical discharge paths (V_B → V_Nwell) absent in prior configs. No tuning artifact; it's corrective, not compensatory. Validates as "use ground truth," aligning pyport with ngspice's implicit well-diode behavior.

- **per-VG1 inversion (0.40=1.24 best, vs 0.20=1.36)**: Physically reasonable. At low VG1 (0.20V), subthreshold dominates, amplifying any Vth/slope residuals (R-29 echo). Mid-VG1 (0.40V) hits a sweet spot where Iii and diode balance improves post-R-41, reducing error. High VG1 (0.60V) over-pumps due to stronger Iii, but pdiode caps it partially. Not over-correction; reflects bias-dependent physics (e.g., Vdseff saturation at higher Vg).

- **1.42 dec ~25× silicon median**: Honest benchmark. Silicon variability (process corners, mismatch) often spans 10-100× in leakage/idle currents; 25× is within typical Monte Carlo spreads (e.g., σ_Vth=30-50mV implies ~1-2 dec Id variation). Not spurious—R-40 audit confirms no hidden offsets. Gate PASS<1.5 validates realism.

### Q2 — single best falsifier (<1h)
(192 words)

**Experiment: Standalone M1 + pdiode discharge curve validation vs ngspice (runnable in <30min).**

Concrete steps:
1. Extract R-41 config (cfg.body_pdiode_to="vnwell", use_well_diode=True, Js/area/n from Sebas card).
2. Simulate M1-only (Vs=0, Vd=2.0V, Vg=0.6V) with body tied to variable Vb sweep [0, 0.8V] in pyport and ngspice. Compute net body current (Iii_in - Ileak_out via pdiode/well) at each Vb.
3. Plot pyport vs ngspice zero-crossing Vb (where net=0) and discharge rate (dI/dVb at Vb=0.484V).
4. Metric: If pyport zero-cross matches ngspice (±50mV) and rate aligns (±20%), 1.42 dec is real (pdiode physics dominant, not compensation). Mismatch >20% flags spurious absorption (e.g., over-strong Js compensating weak Iii).

Rationale: Targets R-41's core fix (parasitic diode) in isolation, falsifying if error hides in cell-wide median. Uses existing z361 scripts; ngspice run <5min, pyport <10min, analysis <15min. Highest yield: Directly tests if Vb=0.484V is physical, not artifact.

### Q3 — push to AMBITIOUS (<0.95)
(236 words)

**Rs invariance on Vb**: Rs sweep [1e3-1e7] having zero effect implies the body discharge bottleneck isn't series resistance in the pdiode path—it's upstream or parallel. At high VG1, strong Iii charges Vb, but discharge is limited by diode saturation (Js/area caps current) or well potential (V_Nwell pinning via substrate ties). Vb stuck at 0.484V (vs ngspice 0.27V) suggests under-strong discharge or over-strong charging: Possible culprits include underestimated well-diode Js (Sebas card conservative), missing bulk resistance (R_well between V_B and diode anode), or inaccurate iii_body_gain (R-39 tuned to pre-R-41 weak Iii, now over-amplifying).

**Single highest-value experiment**: Tune well-diode params sweep + refit (<1h, builds on R-42).

Steps:
1. In z361 script, sweep body_pdiode_Js [Sebas nominal ×0.5, ×2] and add R_well [1e2-1e5 Ω] in series with well-diode (model as explicit resistor in pyport netlist).
2. Re-run flagship bias (VG1=0.6V, VG2=0.2V, Vd=2.0V); find Js/R_well combo driving Vb to 0.27V (±50mV).
3. Refit VG1=0.60 branch (8 biases) with best combo; check median drop.

Rationale: Directly attacks Vb residual (0.21V gap), likely slashing high-VG1 error (1.87→<0.95 dec). Rs invariance points to diode intrinsics; this falsifies/optimizes them. Runtime: Sweeps <20min, refit <30min. If Vb closes, AMBITIOUS unlocks via stronger discharge balancing Iii.
