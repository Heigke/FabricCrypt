# T1 — BSIM4 IIMOD: What Physics Is Missing at L=0.13 µm

## TL;DR
BSIM4's IIMOD inherits BSIM3v3.2's **local, lucky-electron Chynoweth-style** formulation
(α=ALPHA0·exp(-BETA0/E_lat) integrated over the high-field region, with the field set by
Vds-Vdsat and a single characteristic length LITL). At 130 nm and below this is known to
**under-predict Isub by ~2-10×** in the kink/snapback regime because (a) carrier heating
is **non-local** — electrons need an energy-relaxation length λ_E (~65 nm in Si) to
thermalize to the local field, so peak T_e and the actual ionization integral lag the
local-field prediction — and (b) the **floating-body / parasitic-lateral-BJT feedback**
that Lanza et al. exploit in NS-RAM (hole pile-up in the body raises V_BS, lowers V_T,
amplifies I_D, and re-injects more impact ionization) is not in BSIM4's intrinsic loop.
A non-local energy-balance (carrier-temperature) IIMOD, plus an explicit body-charge node
with a parasitic-BJT branch, is required.

## Top-3 Papers

1. **Slotboom, Streutker, van Dort, Woerlee, Pruijmboom, Gravesteijn — "Non-local impact
   ionization in silicon devices" (IEDM 1991).**
   Showed experimentally on MBE bipolars + scaled submicron MOSFETs that ionization
   integrals computed from the local field over-predict the field but **under-predict the
   electron temperature** needed for ionization at peaks narrower than λ_E. Extracted
   **λ_E ≈ 65 nm** for Si. This is *the* canonical reason BSIM3/4-style local IIMOD breaks
   down once the high-field region at the drain shrinks below ~3·λ_E — i.e. exactly at
   130 nm.

2. **Chen, Chan, Hu et al. — "Compact non-local modeling of impact ionization in SOI
   MOSFETs for optimal CMOS device/circuit design"** (Solid-State Electronics, 1996;
   Elsevier 0038-1101(95)00198-0).
   Replaces the field-driven Chynoweth integral with a **carrier-temperature-driven**
   one obtained from a quasi-steady-state energy-balance equation, with λ_E as the only
   added parameter. Validates kink/snapback in floating-body PD-SOI down to submicron
   gate lengths. Demonstrates BSIM-style local IIMOD under-predicts Isub by **~3-5×**
   at the kink onset and misses the negative-output-resistance branch entirely.

3. **Lanza et al. — "Synaptic and neural behaviours in a standard silicon transistor"
   (Nature 641, 2025; doi:10.1038/s41586-025-08742-4).**
   The substrate device. NS-RAM relies on **punch-through-driven impact ionization +
   body charging + parasitic lateral BJT turn-on** in a 130 nm bulk nMOS with a
   resistor-terminated bulk. The spiking/synaptic dynamics are dominated by the
   feedback loop {Isub → V_body ↑ → V_T ↓ → I_D ↑ → Isub ↑} and trap (de)charging —
   neither of which is in BSIM4's IIMOD. Lanza's own simulations use sub-circuit
   add-ons (parasitic BJT + body-RC + trap kinetics) on top of BSIM.

(Supporting: Maes & De Meyer, "Impact ionization in silicon: a review and update";
BSIM-BULK / BSIM-CMG technical manuals — both retain a local α(E_lat) form, only
re-parameterized; BSIM4 manuals confirm IIMOD inherits BSIM3v3.2 form.)

## Named Alternative Model Family
**Energy-balance / carrier-temperature non-local IIMOD (Slotboom–Chen formulation),
coupled with an explicit body-charge node and a Gummel–Poon lateral parasitic NPN.**
This family is now used inside **PSP** (surface-potential, Penn State / Philips, Gildenblat
et al., CMC standard) and inside the **BSIM-BULK HV / BSIM-IMG SOI** add-ons (Agarwal,
Khandelwal et al., 2023, IEEE T-ED, "Compact Modeling of Impact Ionization in
High-Voltage Devices", doi:10.1109/TED.2023.3253...). PSP additionally captures the
kink and self-heating correctly where BSIM4 fails (Yadav et al., "Comparison of PSP and
BSIM4 MOSFET model across various parameters", IEEE 2010 — BSIM4 fails IP3 / kink, PSP
passes).

## Specific Physics BSIM4 IIMOD Lacks at 130 nm
1. **Non-local carrier heating** — no λ_E, no energy-balance ODE. Peak ionization
   is tied to local Vds-Vdsat / LITL, ignoring that the high-field peak at 130 nm is
   narrower than λ_E (~65 nm), so true T_e ≪ local-field prediction → **systematic
   under-prediction** (factor 2-10× reported).
2. **Floating-body / quasi-floating-body charging** — no separate body node with
   RC dynamics, no GIFBE (gate-induced floating-body effect). Bulk-resistance-mediated
   V_BS rise (the heart of NS-RAM) is not representable.
3. **Parasitic lateral BJT (S/B/D = E/B/C)** — not in intrinsic BSIM4; snapback,
   holding voltage, and the punch-through latch require an external sub-circuit BJT.
4. **Trap-assisted / interface-trap charging** — slow (ms-s) charge trapping that
   gives NS-RAM its synaptic plasticity is absent.
5. **Self-heating coupling to ionization rate** — BSIM4 SH affects mobility but does
   not feed back into α(T_lattice) the way energy-balance variants do.
6. **Avalanche cascade / second-generation ionization** at high Vds — α treated as
   single-pass.

## Quantitative Under-Prediction
- Chen/Chan/Hu 1996 (PD-SOI, L=0.35-0.8 µm): BSIM3-style local IIMOD under by **~3-5×**
  at kink onset; **misses negative R_out branch entirely**.
- Slotboom 1991: at submicron peak widths, local-field α over-predicts Isub for *wide*
  peaks but under-predicts the *onset Vds* for ionization in narrow peaks by ~0.3-0.5 V.
- BSIM4 v4.8 manual itself flags IIMOD as a "warning-limit" model outside the
  calibration window — Berkeley's own valid envelope is L ≥ 90 nm bulk, Vds within
  nominal supply; the NS-RAM regime (Vds > 2·V_DD, punch-through) is **outside the
  envelope**.

## Recommendation
For 130 nm NS-RAM physics fitting, **abandon BSIM4 IIMOD as the kernel**. Adopt either:
- **PSP** (built-in non-local + kink + self-heat) as primary surface-potential core, or
- **BSIM-BULK HV with the 2023 IEEE T-ED non-local extensions** if BSIM tooling lock-in,
plus mandatory:
- explicit **body node** with R_body·C_body + trap reservoir,
- **lateral parasitic NPN** (Gummel-Poon) S→B→D,
- energy-balance IIMOD with **λ_E ≈ 65 nm** as a fitted parameter.

This is consistent with our 2026-05-03 M3b walk-back: the 1.39-decade DC fit residual is
the **signature of non-local IIMOD + body feedback**, not a BSIM4 parameter-tuning
problem.

## Sources
- Slotboom et al. IEDM 1991, IEEE doc 235484.
- Chen, Chan, Hu, "Compact non-local modeling of impact ionization in SOI MOSFETs",
  Solid-State Electronics 1996, Elsevier 0038-1101(95)00198-0.
- Lanza et al., Nature 641 (2025), doi:10.1038/s41586-025-08742-4.
- BSIM4 v4.8 Manual (UC Berkeley 2013), IIMOD section.
- BSIM-BULK HV / BSIM-CMG impact-ionization sections (gitbooks).
- Yadav et al., "Comparison of PSP and BSIM4 MOSFET model", IEEE 2010, doc 5498248.
- Agarwal/Khandelwal et al., "Compact Modeling of Impact Ionization in High-Voltage
  Devices", IEEE T-ED 2023, doc 10068187.
- Maes & De Meyer, "Impact ionization in silicon: a review and update".
