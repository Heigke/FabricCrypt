# openai response (gpt-5) — 160s

Q1 — Physical mechanism that quenches snapback at high VG2

Ranking and rationale

1) (b) M2 channel becomes a low-resistance shunt B→S (dominant)
- High VG2 in the long L_M2 device turns on an ohmic/saturated conduction path from the floating body into the source. That clamps V_body, preventing V_BE of the vertical NPN from reaching the ~0.6–0.8 V needed for hard turn-on. This directly drives the loop-gain (α_NPN × M_avalanche × “body-charge gain”) below unity.
- This mechanism is consistent with the observed boundary: as VG2 crosses just above VT2, snapback disappears even when VG1 is moderately high.

2) (d) Voltage partitioning: M2 steals V_D from M1 (important second-order)
- With M1 and M2 in series, a strongly-on M2 drops little voltage; but as the body is clamped (by b), the I–V of the stack reallocates more of the applied V_D across M2’s drain/body junction and less across M1’s drain, reducing M1’s E-field and therefore I_ion. This reinforces (b). It is a direct, bias-partition effect—not a separate “physics,” but it matters in compact modeling.

3) (a) “M2 saturates” and emitter current saturates (weak/mostly a red herring)
- Whether M2 is in triode or saturation is not the cause; the cause is simply that M2 provides a finite current sink from body to source. At high VG2 that current capability is large enough (even in saturation) to evacuate the avalanche-injected holes. Calling out “saturation” as the reason is misleading; it is the magnitude of the M2 B→S current that matters, not the region label.

4) (c) M1 punch-through suppresses impact ionization (unlikely here)
- In this bias/geometry regime (TSMC 130 nm, V_D ≈ 0.8–1.5 V) punch-through would not systematically appear only at high VG2 and would not explain the VG2-dependence. Impact ionization usually increases with V_DS,M1; “depletion merge suppresses II” is not the right explanation for the disappearance of the knee as VG2 rises.

Order-of-magnitude for when (b) dominates

Goal: body shunt via M2 overwhelms injection, i.e.
G_b2s · (V_body − V_source) >> I_ion
or, more robustly for a channel-limited shunt,
I_b2s,M2(V_G2, V_BS) >> I_ion.

Use two regimes for M2 (drain tied to body, source to source):
- Linear (V_BS < V_ov): I ≈ μCox·(W/L)·(V_ov − V_BS/2)·V_BS; small-signal G ≈ μCox·(W/L)·V_ov.
- Saturation (V_BS ≥ V_ov): I_sat ≈ 0.5·β·V_ov^2 with β = μCox·(W/L). Small-signal g_ds ≈ λ·I_sat, but for DC clamping the relevant “effective conductance” is I/V_BS ≈ I_sat/V_BS.

Numbers (TSMC 130 nm NMOS, long L_M2 = 1.8 μm):
- Take μCox ≈ 200–250 μA/V^2·μm. Use 220 μA/V^2·μm nominal.
- For W_M2 = 0.3 μm, β ≈ 220 × (0.3/1.8) ≈ 36.7 μA/V^2.
- I_ion ≈ 1–10 nA; V_body swing ≈ 0.6–0.8 V. We need I_b2s ≫ 10 nA.

Saturation-limited estimate (most relevant because V_BS (~0.6–0.8 V) > V_ov near threshold):
- Set I_sat = 0.5·β·V_ov^2 ≥ I_ion.
- For β = 37 μA/V^2 and I_ion = 10 nA:
V_ov,th ≈ sqrt(2·I_ion/β) ≈ sqrt(20 nA / 37 μA/V^2) ≈ 23 mV.
- For I_ion = 1 nA: V_ov,th ≈ 7.3 mV.

Thus, as soon as V_GS,M2 exceeds V_T2 by only ≈ 10–30 mV, M2 can sink ≥ 10 nA from the body—already enough to quench the feedback. Even with smaller W (say W=0.2 μm → β≈24 μA/V^2), V_ov,th for 10 nA is ≈ 29 mV.

Translate to VG2:
- With V_T2 ≈ 0.40 V, the clamp becomes effective by V_G2 ≈ 0.42–0.45 V for W in the 0.2–0.4 μm range. This matches the empirical “low VG2 (0.0–0.3) shows snapback; high VG2 (0.4–0.5) kills it.”

Check against a conductance view:
- If you insist on G_b2s · ΔV ≫ I_ion with ΔV ≈ 0.7 V and I_ion = 10 nA → G_b2s ≫ 14 nS (R ≪ 70 MΩ).
- In linear approximation, G_b2s ≈ β·V_ov. With β ≈ 37 μA/V^2, we need V_ov ≫ 14 nS / 37 μA/V^2 ≈ 0.38 mV. This shows why even a whisper above VT is enough; the linear approximation is generous, but the saturation-based 20–30 mV threshold is the safer, realistic design number.

Bottom line: Mechanism (b) is the quencher; a few tens of mV of M2 overdrive above VT2 is sufficient to suppress the body rise at I_ion in the 1–10 nA range. Mechanism (d) (voltage partition reducing V_DS,M1) reinforces this. (a) is not the right mental model; (c) is unlikely here.


Q2 — Compact-model encoding of the VG2-snapback boundary

Recommendation
- Primary: (iii) Cb fast-discharge via M2, implemented as a physically based, smooth body-to-source channel branch controlled by VG2.
- Secondary (already implicit if you build the 2T network correctly): voltage partition (d) naturally reduces V_DS,M1 and therefore the avalanche multiplier M(V_DS,M1).
- Avoid (i) and (iv) as primary levers; (ii) is a brittle hack.

Details

- Implement (iii) not as an ad-hoc G_b2s ∝ μCox·(W/L)·(V_G2−V_T2), but as a full MOS-like current from body (as M2-drain) to source with the same long-channel equations (or a reduced BSIM-style core), using:
  I_b2s,chan(V_G2, V_BS) = Ids_model_MOS(Vgs=V_G2, Vds=V_BS, Vbs≈0, …)
  Then inject −I_b2s,chan into the body KCL and +I_b2s,chan into source. This yields:
  - Correct triode/saturation transition with no new discontinuities.
  - Proper scaling with W/L, μCox, VT2, CLM, DIBL (to the extent you include them).
  - Automatic growth of the clamp current with VG2 that kills snapback in the right corner.

- If your macro-model already contains a 2T series path (M1–M2) for Ids between D and S, do not double-count. The B→S path you add is specifically the “M2 drain (tied to body) through channel to source” branch; your series M1–M2 path is the normal conduction from BL to SL. Keep these topologies topologically distinct in the macro: one is D_chain→…→S; the other is Body→S. This mirrors silicon.

- (ii) Conditional Iion “only when M2 below saturation” ties the avalanche in M1 to M2’s region flag. Physically wrong and numerically risky (artificial knees around V_DS,M2 ≈ V_ov). If you rely on correct circuit topology, the right thing (less V_DS,M1 when M2 conducts) already happens without conditionals.

- (i) α0(V_G2) PWL: unjustified. The vertical NPN α0 depends on base transport/recombination; VG2 doesn’t directly modulate that structure. Using VG2 to modulate α0 is a knob-of-last-resort and will hide real physics.

- (iv) M_avalanche(V_DS,M1, V_G2): also unjustified as a primary dependence. M_avalanche should depend on M1’s field (V_DS,M1, V_GD,M1, geometry). Let VG2 affect M_avalanche only indirectly via correct voltage partition. If you need a tiny empirical cross-term for fit, keep it smooth (e.g., weak dependence through V_body that VG2 controls), but don’t make it an explicit VG2 handle.

Numerical robustness
- Using a smooth MOS-like branch for Body→S is well-behaved (no new discontinuities). Use softplus-style smoothing for VT2, and standard CLM/DIBL smoothing.
- Keep impact-ionization current continuous and differentiable (e.g., Okuto–Crowell or BSIM-like M(V_DS) with smooth limiting).
- Ensure the body-charge ODE includes both I_ion into body and all body-out currents (M2 channel B→S, body-source diode, SRH/generation, etc.).


Q3 — Where SHOULD the bistability region end in (V_G1, V_G2) space?

Fastest reliable experiment

- Use (α) reduced 1D fixed-point analysis on the body-node ODE at each (V_G1, V_G2, V_D). With your pseudo-transient formulation:
  C_b dV_b/dt = F(V_b; V_G1, V_G2, V_D)
  where F = I_inj,ion(M1; V_b, …) + I_PNP/NPN_into_body(V_b, …) − I_b2s,chan(M2; V_b, V_G2) − I_junctions(V_b) − I_recomb(V_b).

  Steps:
  - For a grid of V_b in a safe range (e.g., −0.1 to +0.9 V), evaluate F(V_b). Locate all sign changes and refine each root with a robust bracketing solver (bisection/secant).
  - Stability: a fixed point is stable iff dF/dV_b < 0 (since C_b > 0). Compute dF/dV_b numerically (central difference) or analytically if feasible.
  - Bistability exists if you find 3 fixed points with outer two stable (negative slope) and middle unstable (positive slope).
  - This neatly gives you existence and stability without long transients or hysteresis heuristics.

- Map over a 2D grid of (V_G1, V_G2). Practical grid:
  - V_G1: 0.10–0.70 V in 25 mV steps.
  - V_G2: 0.00–0.60 V in 25 mV steps.
  - For each pair, sweep V_D in 0.6–1.5 V with 25–50 mV steps (denser around the observed knee, e.g., 0.8–1.2 V).
  - Record whether 1 or 3 fixed points exist. The boundary where the triple-root collapses (saddle-node) is your bistability edge.

- Cross-check with (γ) forward/reverse sweep:
  - Use two different initial V_b seeds (low and high), run pseudo-transient to DC at each bias. Where both converge but differ, you confirm bistability. Trace the locus where Δlog I_D → 0 as a consistency check.
  - This is cheap and already in your flow, but (α) gives you the unstable branch and avoids missing islands due to poor seeding.

Pitfalls and tips
- Continuation failures near fold points: use arclength continuation in V_b if you want to explicitly trace the S-curve (optional). The bracketing in (α) is usually sufficient.
- Parasitic spurious roots: avoid discontinuities (no hard region switches). Limit avalanche and diode models smoothly.
- Parameter sensitivity: near the boundary, tiny parameter changes can flip stability. Use sufficiently fine V_b sampling and robust tolerances.
- If you keep the full multi-node network (not purely reduced 1D), you can still do (α): treat V_b as the continuation variable, solve the rest quasi-DC at each trial V_b, then evaluate F(V_b). This keeps the problem 1D in practice.


What I would do tomorrow

- Add an explicit, smooth Body→Source channel branch for M2 to the body KCL using a reduced MOS core (Ids(Vgs=VG2, Vds=V_BS)), with proper smoothing and without double-counting the series M1–M2 conduction path.
- Recalibrate I_ion so that at VG1 = 0.2 V the peak I_ion stays < the M2 clamp current for all VG2 ≥ ~0.4 V; verify that the snapback disappears there while remaining at VG1 = 0.4/0.6 V and low VG2.
- Implement the 1D fixed-point scan F(V_b)=0 per (α); generate a (V_G1, V_G2, V_D) bistability map and extract the boundary; cross-check with your z432 hysteresis sweeps.
- Ensure voltage partition is correct: compute V_DS,M1 and V_DS,M2 self-consistently in DC so that high VG2 naturally reduces V_DS,M1 and hence M_avalanche.
- Add regression tests: monotonic I_D(V_D) at high VG2; visible snapback at low VG2; zero snapback at VG1 = 0.2 V across VG2; smooth Jacobian (no new knees) after adding the M2 B→S branch.
