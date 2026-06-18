# gemini response (gemini-2.5-pro) — 77s

Here is an analysis of the missing physics in the 2T NS-RAM model, going beyond the initial findings from SA3.

***

### Q1 — Channel-side physics modulating snapback shape

#### CANDIDATE: Distributed Body Resistance
- **Mechanism (1 sentence):** The floating body is not an equipotential node; the resistance of the p-well (R_B,float) from the impact ionization zone near the drain to the source/body junction determines the gain of the parasitic BJT feedback loop.
- **DC signature** in the 33-curve DC sweep (V_G1, V_G2, V_Nwell ladder) — would this fix anything we currently can't reproduce? Be concrete.
  Yes. The `rbodymod=0` setting in the provided BSIM card implies a single-point body node. A distributed resistance (`rbodymod=1`) will "soften" the snapback by creating a voltage drop across the body. This reduces the V_BE of the parasitic BJT for a given amount of impact ionization current (I_sub), requiring a higher V_D to trigger snapback and making the I-V curve less vertical in the negative-resistance region. It would directly fix models that predict an unrealistically sharp, vertical drop in voltage post-snapback.
- **Transient signature** — predicted shape of rate dependence (which way does the hysteresis loop open/close as ramp slows from 200 µs to 200 ms?)
  This is primarily a DC effect, but it creates an RC time constant with the body capacitance (τ = R_body * C_body). For very fast ramps (< 1 µs), this could introduce a delay, but it is unlikely to be the dominant factor in the 200 µs to 200 ms range. Its main transient effect is setting the DC boundaries of the hysteresis loop.
- **Falsifiability** — one cheap measurement that would confirm or kill this candidate.
  Measure the holding voltage (V_hold) of the snapback curve as a function of V_G1. A significant R_B will cause V_hold to increase more strongly with the post-snapback drain current (which is modulated by V_G1), as I_D * R_B contributes to the body potential. A model with `rbodymod=0` will show a much weaker dependence.
- **Priority for v4.4 model rebuild:** **HIGH**. The BSIM card explicitly disables this (`rbodymod=0`), which is physically incorrect for a floating body device where substrate current feedback is the primary mechanism. This is a first-order structural error in the model.

#### CANDIDATE: Channel Self-Heating
- **Mechanism (1 sentence):** High power dissipation (I_D × V_DS) in the channel during snapback locally increases the lattice temperature, which in turn reduces mobility and increases the impact ionization rate, creating a second, electro-thermal feedback loop.
- **DC signature** in the 33-curve DC sweep (V_G1, V_G2, V_Nwell ladder) — would this fix anything we currently can't reproduce? Be concrete.
  Yes. Self-heating introduces negative feedback on current (via mobility reduction) but positive feedback on impact ionization. This can explain the "droop" or positive slope in the I-V curve *after* snapback, where a simple model might predict a flat or negative slope. If measured curves show I_D increasing with V_D in the high-current state (past the V_hold point), self-heating is a very likely cause.
- **Transient signature** — predicted shape of rate dependence (which way does the hysteresis loop open/close as ramp slows from 200 µs to 200 ms?)
  The thermal time constant of a small transistor is typically in the 0.1–10 µs range. On a 200 µs ramp, the device is in thermal equilibrium. On a 200 ms ramp, it is also in equilibrium. Therefore, self-heating will shift the DC boundaries of the loop but will *not* explain the rate dependence in this specific time window. It's too fast.
- **Falsifiability** — one cheap measurement that would confirm or kill this candidate.
  Use pulsed I-V measurements with varying pulse widths (e.g., 1 µs vs 100 µs) to characterize the snapback curve. If the snapback trigger voltage and holding current change significantly between the short (isothermal) and long (isothermal) pulses, self-heating is present.
- **Priority for v4.4 model rebuild:** **MED**. It's a real effect that shapes the post-snapback curve, but it's not the primary cause of the slow transient hysteresis. It's a secondary correction needed for DC accuracy.

#### CANDIDATE: Drain-Induced Punch-Through
- **Mechanism (1 sentence):** Before the drain field is high enough for significant impact ionization, the drain and source depletion regions can merge deep in the bulk, creating a subsurface leakage path that pre-charges the body.
- **DC signature** in the 33-curve DC sweep (V_G1, V_G2, V_Nwell ladder) — would this fix anything we currently can't reproduce? Be concrete.
  This would explain a "soft" knee in the I-V curve just before the sharp snapback. Instead of being perfectly off, the drain current would show a premature, exponential rise with V_D. This pre-snapback current provides an additional source of body charging, lowering the V_D required for avalanche-induced snapback, especially at low V_G1.
- **Transient signature** — predicted shape of rate dependence (which way does the hysteresis loop open/close as ramp slows from 200 µs to 200 ms?)
  This is a DC leakage mechanism. Like self-heating, it will shift the DC snapback voltage but will not, by itself, create a time-dependent hysteresis in the ms range.
- **Falsifiability** — one cheap measurement that would confirm or kill this candidate.
  Measure the subthreshold I_D-V_D curves at V_G1 = 0 V or a slightly negative voltage. If a current floor appears that rises sharply with V_D well before the expected snapback voltage (~2V), and this floor is sensitive to V_Nwell (which modulates the depletion region from the back), then punch-through is occurring.
- **Priority for v4.4 model rebuild:** **LOW**. While physically plausible, its effect is to modify the pre-trigger condition. The dominant missing physics are likely those governing the slow state evolution.

***

### Q2 — Body-side physics governing transient (ms) response

#### CANDIDATE: Generation-Recombination (G-R) in Depletion Regions
- **Mechanism (1 sentence):** Thermally generated electron-hole pairs within the expanding drain-body and Nwell-body reverse-biased depletion regions create a small but persistent current that slowly charges or discharges the floating body over milliseconds.
- **DC signature** in the 33-curve DC sweep (V_G1, V_G2, V_Nwell ladder) — would this fix anything we currently can't reproduce? Be concrete.
  In a very slow DC sweep, G-R current acts as a steady-state leakage path. It would manifest as a voltage-dependent floor on the body current, potentially clamping the body potential and preventing it from floating to the ideal level. This could explain why the measured snapback voltage is lower than a pure impact-ionization model predicts, as the body is "pre-charged" by G-R current.
- **Transient signature** — predicted shape of rate dependence (which way does the hysteresis loop open/close as ramp slows from 200 µs to 200 ms?)
  This is a key candidate. As the V_D ramp slows from 200 µs to 200 ms, there is more time for the G-R current (I_gen ∝ Depletion Volume) to integrate charge onto the body capacitance (Q = ∫I_gen dt). This pre-charging lowers the V_BE of the parasitic BJT, meaning less impact ionization (and thus a lower V_D) is needed to trigger snapback. The hysteresis loop's trigger point will therefore **shift to a lower V_D** as the ramp slows down.
- **Falsifiability** — one cheap measurement that would confirm or kill this candidate.
  Measure the snapback trigger voltage at two different temperatures (e.g., 25°C and 85°C). G-R current is exponentially dependent on temperature (roughly doubling every 10°C), while impact ionization is only weakly dependent. A strong shift in the ramp-rate dependence with temperature would be a smoking gun for G-R.
- **Priority for v4.4 model rebuild:** **HIGH**. The millisecond timescale is the classic signature of carrier lifetime-dominated G-R processes. This is a fundamental mechanism for slow charge dynamics in isolated structures.

#### CANDIDATE: Interface and Bulk Oxide Traps
- **Mechanism (1 sentence):** The thick gate oxide and the Si/SiO2 interface contain charge traps with a wide distribution of capture/emission time constants (µs to seconds), which store and release charge from the floating body as its potential changes.
- **DC signature** in the 33-curve DC sweep (V_G1, V_G2, V_Nwell ladder) — would this fix anything we currently can't reproduce? Be concrete.
  In a DC sweep, traps have time to reach equilibrium. This would manifest as a slight, history-dependent shift in the threshold voltage (Vth) of M1 and M2. It could explain why sweeps in one direction (e.g., V_G1 low to high) give slightly different curves than sweeps in the reverse direction, even when done very slowly.
- **Transient signature** — predicted shape of rate dependence (which way does the hysteresis loop open/close as ramp slows from 200 µs to 200 ms?)
  This directly creates hysteresis. On the up-ramp, as V_B rises, traps capture holes, effectively removing charge that would otherwise contribute to turning on the BJT. On the down-ramp, these traps slowly emit the captured holes, keeping V_B elevated for longer. As the ramp slows, more of the slower traps have time to participate, **widening the hysteresis loop** (increasing the difference between trigger and hold voltages).
- **Falsifiability** — one cheap measurement that would confirm or kill this candidate.
  Perform a "wait time" experiment. Ramp V_D up to just below snapback, hold it there for varying times (1 ms, 10 ms, 100 ms), and then complete the ramp. If a longer wait time at high V_D causes the snapback voltage to shift (likely increase, due to trap filling), then trapping is a dominant effect.
- **Priority for v4.4 model rebuild:** **HIGH**. A distribution of time constants is the classic explanation for behavior that changes continuously over multiple decades of time (µs → ms). Thick oxides are known to have more traps, making this highly probable.

#### CANDIDATE: DNW-Substrate Capacitive Coupling
- **[OVERLAP: SA3#1, but a different aspect]**
- **Mechanism (1 sentence):** The entire Deep N-Well is a large capacitor to the underlying P-substrate (which is at ground), creating a third capacitive path that couples substrate noise and global transients onto the floating body node.
- **DC signature** in the 33-curve DC sweep (V_G1, V_G2, V_Nwell ladder) — would this fix anything we currently can't reproduce? Be concrete.
  This has no DC signature, as it's a purely capacitive effect.
- **Transient signature** — predicted shape of rate dependence (which way does the hysteresis loop open/close as ramp slows from 200 µs to 200 ms?)
  This adds another term to the capacitive divider that determines the body potential during transients. Any dV/dt on the V_Nwell supply or ground bounce in the substrate will be coupled to V_B via this path (C_dnw_sub). While it affects the transient response, it does not have an intrinsic time constant in the ms range and thus cannot explain the ramp-rate dependence on its own. It modifies the *magnitude* of the response to other fast transients.
- **Falsifiability** — one cheap measurement that would confirm or kill this candidate.
  Apply a small, high-frequency AC signal to the P-substrate (if a separate terminal is available) or to the V_Nwell bias while monitoring the M2 drain current (which reflects V_B). If a coupled signal is detected, the path exists.
- **Priority for v4.4 model rebuild:** **LOW**. While physically present, it's a parasitic AC effect. The core ms-timescale hysteresis must come from a dissipative/storage mechanism like G-R or traps, not a simple capacitor.

***

### Q3 — Slide-21 ramp-rate hysteresis

To reproduce the hysteresis loop that changes shape with V_d ramp rate from 200 µs to 200 ms, two distinct physical mechanisms **MUST** be present, working in concert:

1.  **A Slow, Integrative Charging Mechanism:** There must be a process that accumulates charge on the floating body with a time constant in the millisecond range. This explains why the trigger voltage is rate-dependent at all. The most fundamental and physically certain candidate is **Generation-Recombination (G-R) Current** in the reverse-biased junctions connected to the body. As the V_D ramp slows, this weak current has more time to integrate charge onto the body capacitance, pre-charging V_B towards the BJT turn-on voltage. This causes the snapback event to trigger at a progressively lower V_D on slower ramps.

2.  **A Charge Trapping/Storage Mechanism with Distributed Time Constants:** There must be a charge reservoir separate from the simple body capacitance to explain the loop's "memory" and shape. This reservoir must have time constants spanning the µs-ms range. The definitive candidate is **Interface/Oxide Traps**. On the up-ramp, as V_B rises, these traps capture charge, creating a current that opposes the V_B rise. On the down-ramp, they slowly release this charge, creating a current that keeps V_B artificially high. This prevents the parasitic BJT from turning off until a much lower V_D is reached, thus creating the hysteresis loop. The slower the ramp, the more of the "slow traps" can participate, changing the loop's area and shape.

**In summary:** The G-R current sets the *trigger voltage's dependence on ramp rate*. The traps create the *hysteresis loop itself* and cause the loop's *shape and area to depend on ramp rate*. You cannot reproduce the described behavior without both.

***

## SUMMARY

My top-3 picks for missing physics, ranked by importance for a v4.4 model rebuild:

1.  **Distributed Body Resistance:** A first-order error in the current DC model setup (`rbodymod=0`). It is fundamental to correctly modeling the feedback gain of the parasitic BJT and the shape of the snapback curve.
2.  **Generation-Recombination (G-R) Current:** The most direct and physically necessary mechanism to explain the slow, millisecond-scale time constant that makes the device's trigger point sensitive to ramp rate.
3.  **Interface and Bulk Oxide Traps:** The only plausible mechanism to create a hysteresis loop whose shape and area evolve continuously over several decades of ramp time (200 µs → 200 ms), which is a key feature of the measured data.
