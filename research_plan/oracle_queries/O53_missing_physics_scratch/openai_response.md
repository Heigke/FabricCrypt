# openai response (gpt-5) — 505s

Below are new, high‑yield physics candidates beyond SA3’s 7. I’ve split them into Q1 (channel‑side snapback shaping) and Q2 (body‑side, ms hysteresis), and a Q3 “must‑have” short list. I’ve avoided overlaps; where overlap exists, I tag [OVERLAP: SA3#N] and skip details.

Q1 — Channel-side physics modulating snapback shape (Vd > ~2 V)

### CANDIDATE: GIDL-assisted pre-snapback body charging at the drain edge
- Mechanism (1 sentence): Field-induced band-to-band tunneling under the drain-side gate edge (GIDL) creates holes that pre-charge the floating body before avalanche fully ignites, strongly dependent on VDG (hence VG1) and drain pocket fields.
- DC signature in the 33-curve DC sweep (V_G1, V_G2, V_Nwell ladder) — would this fix anything we currently can't reproduce? Be concrete.: Raises the pre-knee tail and softens the knee; at low VG1 (e.g., 0.2 V) the knee shifts earlier by ~0.1–0.3 V vs higher VG1 (0.6 V) even with VB initially low; also increases IB/ID ratio just before snapback; reproduces separation of VG1 branches right before the knee that pure avalanche cannot explain.
- Transient signature — predicted shape of rate dependence (which way does the hysteresis loop open/close as ramp slows from 200 µs to 200 ms?): Slower ramps allow more GIDL-generated holes to accumulate, moving the up-sweep knee to lower VD and reducing the hold current; loop area shrinks modestly and shifts left by ~50–150 mV/dec slower ramp unless counteracted by slow discharge paths.
- Falsifiability — one cheap measurement that would confirm or kill this candidate. Prefer experiments doable on the existing TEG.: Repeat slide-21 ramps at fixed VG2 while stepping VG1 by ±100 mV; if the up-sweep knee moves strongly with VG1 and the shift is more pronounced for slower ramps, GIDL is active. A control with VB pinned near 0 V (briefly pre-discharged) should not remove the pre-knee separation if GIDL is the cause.
- Priority for v4.4 model rebuild: HIGH — Explains VG1-sensitive pre-knee softening and rate sensitivity before full snapback; BSIM GIDL terms exist but are not exploited in the current semi-empirical fit.

### CANDIDATE: Two-site impact-ionization nucleation (STI-corner crowding vs gate-edge) with lateral competition
- Mechanism (1 sentence): Field crowding at STI corners and drain-pocket halos creates multiple high-field “hot spots,” causing nucleation at one or more sites and a soft, sometimes two-step knee as domains compete or merge.
- DC signature in the 33-curve DC sweep (V_G1, V_G2, V_Nwell ladder) — would this fix anything we currently can't reproduce? Be concrete.: Explains an S-shaped or two-shoulder knee and a smoother transition into snapback than a single-site avalanche; can also produce slight discontinuities or small kinks around the knee that vary with VNwell bias (via depletion width).
- Transient signature — predicted shape of rate dependence (which way does the hysteresis loop open/close as ramp slows from 200 µs to 200 ms?): Faster ramps favor a single dominant nucleation site (sharper knee, wider loop); slower ramps allow charge spreading and multi-site averaging (knee softens, loop slightly narrows and becomes more reproducible).
- Falsifiability — one cheap measurement that would confirm or kill this candidate. Prefer experiments doable on the existing TEG.: Run identical ramps while superimposing a tiny AC dither (few mV, 10–100 kHz) on VD; observe whether the knee shows intermittent micro-hops at fast ramps that disappear at slow ramps—multi-site nucleation tends to produce such micro-hops at high dV/dt.
- Priority for v4.4 model rebuild: MED — Likely present in 130 nm with STI; it shapes the knee subtlety and explains occasional micro-structure near snapback but is secondary to GIDL/avalanche.

### CANDIDATE: Kirk effect (collector/drain conductivity modulation) in the lateral parasitic NPN → quasi-saturation in snapback
- Mechanism (1 sentence): At high post-knee current, injected carrier density in the drain extension exceeds the background doping, widening the base (p-body) and reducing β (Kirk effect), turning negative differential resistance into a flatter, quasi-saturated plateau.
- DC signature in the 33-curve DC sweep (V_G1, V_G2, V_Nwell ladder) — would this fix anything we currently can't reproduce? Be concrete.: Reduces or removes unrealistically deep NDR after snapback; creates a near-constant or gently rising ID vs VD plateau with a hold voltage that increases with current; helps match measured “flat” post-knee segments.
- Transient signature — predicted shape of rate dependence (which way does the hysteresis loop open/close as ramp slows from 200 µs to 200 ms?): Slower ramps mildly reduce the up-sweep knee current (due to thermal and body-charge effects) and increase stored charge, so the down-sweep persists to slightly higher VD (loop widens slightly on the right side).
- Falsifiability — one cheap measurement that would confirm or kill this candidate. Prefer experiments doable on the existing TEG.: Add a series resistor (10–100 Ω) in the VD path to lower the effective collector field at high current; quasi-saturation should strengthen (flatter plateau, higher hold voltage), distinct from pure avalanche scaling.
- Priority for v4.4 model rebuild: HIGH — Essential to tame unrealistic NDR and match observed flat post-knee behavior.

### CANDIDATE: Velocity-saturation re-partitioning of VDS,eff via dynamic pinch-off length (gate-controlled E-field relocation)
- Mechanism (1 sentence): As the channel saturates, the pinch-off point moves and redistributes the lateral E-field; feedback with VG1/VB changes where the peak field sits, modulating the avalanche integral and knee sharpness.
- DC signature in the 33-curve DC sweep (V_G1, V_G2, V_Nwell ladder) — would this fix anything we currently can't reproduce? Be concrete.: Produces a VG1-dependent knee sharpness and modest VD knee shift that persists even when VB is constrained; helps match differences between VG1 branches where pure DIBL or avalanche alone fall short.
- Transient signature — predicted shape of rate dependence (which way does the hysteresis loop open/close as ramp slows from 200 µs to 200 ms?): Weak rate effect by itself; interacts with body charging so that slower ramps (higher VB) pull the pinch-off point away from the drain, softening the knee and narrowing the loop slightly.
- Falsifiability — one cheap measurement that would confirm or kill this candidate. Prefer experiments doable on the existing TEG.: Compare knee evolution while stepping VG1 and simultaneously clamping VB (pre-discharged each sweep); if knee shape still follows VG1 significantly, E-field relocation is implicated beyond pure VB feedback.
- Priority for v4.4 model rebuild: MED — Improves knee shape robustness across VG1 without invoking extra body effects.

### CANDIDATE: Drain-induced punch-through (DIPT) assist to the knee
- Mechanism (1 sentence): At high VD and subthreshold VG1, the source–drain depletion regions approach and enable a gate-weak conduction path that seeds current and body charging before avalanche.
- DC signature in the 33-curve DC sweep (V_G1, V_G2, V_Nwell ladder) — would this fix anything we currently can't reproduce? Be concrete.: Lifts pre-knee current especially at lowest VG1; yields a surprisingly VG1-insensitive subthreshold tail at high VD that our current model underestimates.
- Transient signature — predicted shape of rate dependence (which way does the hysteresis loop open/close as ramp slows from 200 µs to 200 ms?): Similar to GIDL but weaker; slower ramps open the loop slightly less because DIPT provides current without strong time constants; primarily shifts the up-sweep left modestly.
- Falsifiability — one cheap measurement that would confirm or kill this candidate. Prefer experiments doable on the existing TEG.: Repeat the ramp with a small back-gate pre-bias by nudging VNwell to slightly change well potential (±100 mV around nominal); punch-through tail should be more sensitive to well potential than pure GIDL.
- Priority for v4.4 model rebuild: LOW–MED — Possible contributor, but likely secondary to GIDL in this geometry.

### CANDIDATE: Self-heating at the drain hot spot modulating ionization coefficients and mobility
- Mechanism (1 sentence): Local lattice temperature rise near the drain reduces impact-ionization coefficients and carrier mobility, moving the knee and flattening the post-knee slope with time.
- DC signature in the 33-curve DC sweep (V_G1, V_G2, V_Nwell ladder) — would this fix anything we currently can't reproduce? Be concrete.: On quasi-DC (slow) sweeps the knee appears at slightly higher VD and the plateau slope is more ohmic than predicted isothermally; also explains ambient-T dependence not captured by isothermal fits.
- Transient signature — predicted shape of rate dependence (which way does the hysteresis loop open/close as ramp slows from 200 µs to 200 ms?): Slower ramps heat more; up-sweep knee shifts right by ~20–80 mV/dec slower ramp; down-sweep cools, so the loop widens slightly on the left; overall loop area can increase with slower ramps if thermal dominates over leak RC.
- Falsifiability — one cheap measurement that would confirm or kill this candidate. Prefer experiments doable on the existing TEG.: Repeat slide-21 with identical waveforms but 1% vs 50% duty cycle; a knee shift only at high duty (not at low) indicates self-heating; corroborate by ambient temperature sweep.
- Priority for v4.4 model rebuild: MED — Likely non-negligible at >mA/mm currents; interacts with avalanche and body storage; include a single-pole electrothermal element.

Q2 — Body-side physics governing transient (200 µs → 200 ms) hysteresis

### CANDIDATE: Body–source diode forward bias and minority-carrier storage
- Mechanism (1 sentence): Once VB rises ~0.6–0.7 V above VS, the body–source PN diode conducts and stores charge, creating a slow discharge tail that sets a ms-scale memory even after the external ramp reverses.
- DC signature in the 33-curve DC sweep (V_G1, V_G2, V_Nwell ladder) — would this fix anything we currently can't reproduce? Be concrete.: Reduces the apparent hold voltage and flattens the post-knee plateau; can suppress deep NDR predicted by purely impact-ionization models; explains why hold current depends on VS (if ever biased) or on source resistance.
- Transient signature — predicted shape of rate dependence (which way does the hysteresis loop open/close as ramp slows from 200 µs to 200 ms?): Slower ramps increase the time in forward bias, increasing stored charge; down-sweep branch remains at higher current to lower VD (loop widens on the down-sweep side), with long tails if the ramp pauses.
- Falsifiability — one cheap measurement that would confirm or kill this candidate. Prefer experiments doable on the existing TEG.: Repeat ramps with VS raised by +100 mV; if down-sweep tails and hold current drop markedly (forward bias suppressed), this mechanism is active. Alternatively, briefly clamp VB below 0.4 V between sweeps to remove residual storage and observe loop collapse.
- Priority for v4.4 model rebuild: HIGH — Provides a natural ms-scale reservoir and explains asymmetric up/down branches.

### CANDIDATE: Diffusion capacitance and carrier storage in the VB–DNW junction (beyond Cj)
- Mechanism (1 sentence): Near-forward or forward bias of the Pwell–DNW diode during/after snapback leads to large diffusion capacitance and stored minority charge in the DNW, producing ms-scale tails.
- DC signature in the 33-curve DC sweep (V_G1, V_G2, V_Nwell ladder) — would this fix anything we currently can't reproduce? Be concrete.: DC curves alone won’t reveal it strongly, but the apparent knee/hold dependence on VNwell bias (via easier forward bias) and subtle plateau flattening become easier to match when diffusion storage is included.
- Transient signature — predicted shape of rate dependence (which way does the hysteresis loop open/close as ramp slows from 200 µs to 200 ms?): Strong rate dependence on the down-sweep; for slower ramps, the loop widens and exhibits long recovery tails (ms) even if the up-sweep knee shifts only modestly.
- Falsifiability — one cheap measurement that would confirm or kill this candidate. Prefer experiments doable on the existing TEG.: Intentionally forward-bias VB–DNW with a short +0.8 V, 1–10 µs pulse (via VNwell) and then measure VB decay at open-circuit; a multi-exponential ms tail indicates diffusion storage dominates over pure reverse-leak RC.
- Priority for v4.4 model rebuild: HIGH — Explains the 200 µs→200 ms dependence without invoking exotic traps; easy to parameterize.

### CANDIDATE: Distributed floating-body resistance/capacitance network (RB,float not a lumped node)
- Mechanism (1 sentence): The p-well under M1/M2 is a resistive sheet with spatially varying junction capacitances; body charge spreads laterally, yielding multiple RC time constants from tens of µs to tens of ms.
- DC signature in the 33-curve DC sweep (V_G1, V_G2, V_Nwell ladder) — would this fix anything we currently can't reproduce? Be concrete.: Minor in DC; helps explain gradual, bias-dependent shifts in the knee when device periphery (STI-adjacent regions) contribute differently across VG1/VNwell ladders.
- Transient signature — predicted shape of rate dependence (which way does the hysteresis loop open/close as ramp slows from 200 µs to 200 ms?): Multi-slope VB transients; as ramp slows, the loop evolves from a single-lobe to a more rounded, two‑time‑constant shape (fast opening then slow closure), and “memory between sweeps” persists if pauses are shorter than the slow RC.
- Falsifiability — one cheap measurement that would confirm or kill this candidate. Prefer experiments doable on the existing TEG.: Apply a small-amplitude sine dither on VD (1–10 mV) and sweep frequency from 1 kHz to 1 MHz while biasing near the knee; measure VB gain/phase to extract multiple poles; a single-pole fit will fail if distribution is significant.
- Priority for v4.4 model rebuild: MED–HIGH — Necessary for realistic transient fits; can be approximated with 2–3 ladders.

### CANDIDATE: STI edge/border traps capturing holes during snapback (oxide/Si interface traps with ms detrapping)
- Mechanism (1 sentence): High fields near the drain/STI corner capture holes in border/interface traps that release over ms, shifting VB and effective Vth with a characteristic delay.
- DC signature in the 33-curve DC sweep (V_G1, V_G2, V_Nwell ladder) — would this fix anything we currently can't reproduce? Be concrete.: Minimal DC impact unless “slow DC” is used; may explain slight hysteresis between forward/backward quasi-DC sweeps and small offsets between nominally identical sweeps.
- Transient signature — predicted shape of rate dependence (which way does the hysteresis loop open/close as ramp slows from 200 µs to 200 ms?): Loop area grows with slower ramps (more trap filling) and exhibits time-dependent drift of the knee across repeated sweeps; recovery accelerates with temperature.
- Falsifiability — one cheap measurement that would confirm or kill this candidate. Prefer experiments doable on the existing TEG.: Shine 405–450 nm light during ramps to de-trap border traps; if loop area collapses and recovery speeds up only under illumination, border traps are implicated; temperature sweep should also compress the loop at higher T.
- Priority for v4.4 model rebuild: MED — Plausible at thick-ox/STI corners; adds a slow pole; keep compact.

### CANDIDATE: Drain–body TAT (trap-assisted tunneling) in reverse junction (distinct from pure BTBT)
- Mechanism (1 sentence): Field-assisted tunneling via deep traps within the drain–body depletion region generates carriers with strong field and temperature dependence and includes trap occupancy dynamics (µs–ms).
- DC signature in the 33-curve DC sweep (V_G1, V_G2, V_Nwell ladder) — would this fix anything we currently can't reproduce? Be concrete.: Enhances the pre-knee current at high VD without requiring gate bias; shows stronger T dependence than pure BTBT; improves matching of slope in the highest-VD, subthreshold VG1 curves.
- Transient signature — predicted shape of rate dependence (which way does the hysteresis loop open/close as ramp slows from 200 µs to 200 ms?): Slower ramps increase trap occupancy, slightly lowering the up-sweep knee and increasing the down-sweep tail; loop area grows with both VDmax and temperature if TAT dominates.
- Falsifiability — one cheap measurement that would confirm or kill this candidate. Prefer experiments doable on the existing TEG.: Measure the pre-knee leakage vs temperature (e.g., 25→85 °C); if it rises strongly and the knee shifts left notably with T at constant ramp rate, TAT is likely (pure BTBT often weakens with T).
- Priority for v4.4 model rebuild: MED — Adds necessary field/T dependence to the pre-knee; can be merged with GIDL modeling if desired.

### CANDIDATE: DNW-to-substrate capacitive path plus VNwell supply impedance (third coupling path)
- Mechanism (1 sentence): The large DNW has appreciable capacitance to the global substrate and finite supply impedance; VNwell bounces under body-diode currents, feeding back into VB as a slow, supply‑limited pole.
- DC signature in the 33-curve DC sweep (V_G1, V_G2, V_Nwell ladder) — would this fix anything we currently can't reproduce? Be concrete.: No change to static curves; explains why changing the VNwell source impedance (or adding decaps) alters transient hysteresis without shifting DC knees.
- Transient signature — predicted shape of rate dependence (which way does the hysteresis loop open/close as ramp slows from 200 µs to 200 ms?): Slower ramps show more VNwell “breathing” if the supply is weakly decoupled; loop area and shape become supply‑impedance dependent; adding VNwell decap collapses this sensitivity.
- Falsifiability — one cheap measurement that would confirm or kill this candidate. Prefer experiments doable on the existing TEG.: Add/removes a large external decoupling capacitor (e.g., 1–10 nF) from VNwell to ground; if the hysteresis loop shape changes markedly (especially on down-sweep), this path is active.
- Priority for v4.4 model rebuild: MED — Simple to include (one RC), important for bench‑to‑bench reproducibility.

Q3 — Slide‑21 ramp‑rate hysteresis: what MUST be present to reproduce the loop?

### CANDIDATE: Impact-ionization driven positive feedback into the floating body
- Mechanism (1 sentence): A VD‑dependent avalanche source injects holes into the body, raising VB, lowering Vth (via body effect), and increasing channel current—this feedback is the engine of snapback and the down‑sweep hold.
- DC signature in the 33-curve DC sweep — would this fix anything we currently can't reproduce? Be concrete.: Produces the knee and the post‑knee plateau; sets the basic IB/ID partition; without it the measured snapback cannot be matched at all.
- Transient signature — predicted shape of rate dependence: Provides the core latching/holding; interacts with other reservoirs to set loop width; faster ramps give crisper knees; slower ramps smooth knees due to concurrent body charging.
- Falsifiability: NA (foundational), but the IB vs ID extraction with VB pinned (as in Image 14) isolates this term.
- Priority for v4.4 model rebuild: MUST — Foundational.

### CANDIDATE: GIDL/TAT pre-charge of VB before avalanche ignition
- Mechanism (1 sentence): Field‑assisted drain‑edge generation charges the body during the up‑sweep, moving the apparent trigger to lower VD at slower ramps.
- DC signature in the 33-curve DC sweep — would this fix anything we currently can't reproduce? Be concrete.: Explains VG1‑sensitive pre‑knee slope and earlier knee for low VG1 branches; without it, pre‑knee separation is underfit.
- Transient signature — predicted shape of rate dependence: Up‑sweep knee shifts left by ~50–150 mV/dec slower ramp; loop narrows unless storage effects counteract.
- Falsifiability: VG1 step test (±100 mV) vs ramp rate; knee moves more at slower ramps if GIDL/TAT is active.
- Priority for v4.4 model rebuild: MUST — Required to get the correct rate‑dependent knee shift.

### CANDIDATE: Two distinct body charge reservoirs/time constants (fast RC + slow storage)
- Mechanism (1 sentence): A fast pole (10–200 µs) from depletion/MOS coupling (Cbody + Cvb–vg2) and reverse‑leak R, plus a slow pole (1–50 ms) from forward‑biased diode diffusion storage (VB–S and/or VB–DNW) and/or deep border traps.
- DC signature in the 33-curve DC sweep — would this fix anything we currently can't reproduce? Be concrete.: Not visible in static curves; critical for reproducing how the loop evolves across 200 µs→200 ms ramps and why the down‑sweep remains displaced even after long pauses.
- Transient signature — predicted shape of rate dependence: As ramps slow, the up‑sweep knee shifts modestly (fast RC fills), but the down‑sweep remains elevated longer (slow storage empties), widening the loop on the return branch and introducing long tails if the ramp pauses.
- Falsifiability: Two‑step protocol—(i) ramp into post‑knee, (ii) hold at a mid‑VD for varying dwell times (50 µs→50 ms), (iii) ramp down; plot knee/hold vs dwell; a clear two‑slope recovery identifies two time constants.
- Priority for v4.4 model rebuild: MUST — The 200 µs→200 ms loop evolution cannot be reproduced with a single RC.

### CANDIDATE: DNW supply coupling (Cdnw–sub + supply R) [beyond SA3’s diode itself]
- Mechanism (1 sentence): VNwell potential “breathes” under body‑diode currents if the supply is not perfectly stiff, adding a bench‑dependent slow pole that alters loop shape.
- DC signature in the 33-curve DC sweep — would this fix anything we currently can't reproduce? Be concrete.: None in DC; fixes discrepancies between benches/days without refitting device parameters.
- Transient signature — predicted shape of rate dependence: More loop area and asymmetry with higher VNwell source impedance; adding decap suppresses rate sensitivity.
- Falsifiability: Add external decap to VNwell; loop collapses if this path matters.
- Priority for v4.4 model rebuild: SHOULD — Not strictly intrinsic, but required for testbench‑faithful reproduction.

## SUMMARY
- Top-1: GIDL/TAT-assisted pre-snapback body charging — Explains VG1-sensitive pre-knee softening and the left-shift of the up-sweep knee with slower ramps; easy to add with existing BSIM-like terms.
- Top-2: Diffusion storage in forward-biased body diodes (VB–S and/or VB–DNW) — Provides the ms-scale memory that widens the down-sweep branch and yields long tails; indispensable for reproducing 200 µs→200 ms evolution.
- Top-3: Kirk effect/quasi-saturation of the lateral NPN — Tames unrealistic NDR and matches the flat post-knee plateau/hold behavior across biases; complements avalanche + body charging.
