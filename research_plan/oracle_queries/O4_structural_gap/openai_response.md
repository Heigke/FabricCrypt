# openai response (gpt-5) — 217s

Short answer first

(a) Sebas’s LTspice fit does reproduce the late‑Vd jump. The solid lines in his plot clearly turn up by several decades near the high‑Vd end for the same VG2 slices where the measurements jump. So our target is that jump; the “smooth tongue” family is not enough.

(b) Most likely root cause among your list: (3) M2.B tied to GND (as in the deck) vs our M2.B tied to the floating P‑body. With M2.B=GND the body node seen by the parasitic NPN is the P‑body of M1 only; the bulk NMOS (M2) no longer provides body‑diode sinks that clamp Vb as it rises. That lets the well‑diode (+2 V DNW) + impact‑ionization + NPN loop run away and produces the fold/snapback. When you tie M2.B to B (our current model), the two large M2 junction diodes are referenced to the floating body and they bleed away the very charge that needs to accumulate to trigger the NPN, so the I–V stays in the smooth regime.

Rank order of the five hypotheses for this dataset
- 1st — M2.B=GND vs floating (3): Structural, large effect, and consistent with the .asc. It changes the body KCL signficantly at exactly the biases that jump in the data (VG2 small → M2 off, so only its junctions matter).
- 2nd — Deep‑N‑well feed and BJT topology: you already fixed both (emitter=GND; well diode to +2 V). Without those two you never get a jump. With them in place and M2.B incorrectly tied to B, the loop gain is still too small — you see the “smooth tongue.”
- 3rd — Self‑heating (1): can sharpen the knee but is not required for a DC fold here. Sebas’s deck declares RTH0/CTH0 but the jump is already present in his DC curves at microamp–milliamp levels where thermal runaway is not the only plausible mechanism.
- 4th — mbjt scaling semantics (4): matters for amplitude, not for the presence/absence of the jump. In his CSV mbjt tracks the parasitic structure; scaling the BJT area (and the well‑diode) is the right interpretation. Scaling Bf instead will change the onset slightly but won’t create a missing fold if M2.B is wrongly wired.
- 5th — LTspice “slow ramp” (5): the fold shows up in DC; you already implemented proper arclength continuation, so a transient artifact is unlikely to be the explanation for the static jump.

(c) Measurement artifact? Unlikely. The step happens at Vd≈1.8–2.5 V (not at ~0.7 V where pad or ESD diodes would turn on) and is bias‑selective the same way in Sebas’s sim and in the measurements. That is classic parasitic‑BJT snapback behavior, not a compliance blip.

(d) One discriminating experiment (one change)

Tie M2’s body to ground and remove M2’s body‑diode currents from the body KCL. This is exactly the topology in 2tnsram_simple.asc (bulk NMOS with body at 0 V).

Two‑line patch in nsram_cell_2T._residuals:

- Compute M2 with Vb=0:
  current code:
    m2 = _eval_mosfet(model_M2, sd_M2, cfg, Vg=VG2, Vd=Vsint, Vs=zero, Vb=Vb, …)
  replace with:
    m2 = _eval_mosfet(model_M2, sd_M2, cfg, Vg=VG2, Vd=Vsint, Vs=zero, Vb=zero, …)

- Drop M2’s diodes from the body KCL (they belong to M2’s own substrate, not the floating body):
  current code in R_B:
    … - m1["Ibs"] - m1["Ibd"] - m2["Ibs"] - m2["Ibd"] - Ib_Q1 + I_well_body
  replace with:
    … - m1["Ibs"] - m1["Ibd"] - Ib_Q1 + I_well_body

If you want a single switch without code churn, guard it with an env var:

  tie_M2_body_to_gnd = bool(int(os.getenv("NSRAM_M2B_GND", "1")))
  vb_M2 = zero if tie_M2_body_to_gnd else Vb
  m2 = _eval_mosfet(..., Vb=vb_M2, …)
  …
  if not tie_M2_body_to_gnd:
      R_B = R_B - m2["Ibs"] - m2["Ibd"]

What you should see
- If the jump is structural (M2.B=GND), the fold will appear at the same VG2 slices where Sebas’s curves jump, and the general shape will match much better without touching parameters.
- If nothing changes, then move to the next lever: enable self‑heating (or, as a quick proxy, rerun with cfg.T_C stepped to 100–150 C only for M1). If that immediately sharpens/creates the knee at high Vd, thermal feedback is contributing.

Additional confirmations you already aligned with Sebas
- Q1 emitter = ground (the .asc shows the npn emitter tied to 0, not to Sint).
- A forward DNW→Pbody diode to a +2 V well contact through a large Rs; this is the DC base‑current source that starts the loop — keep it.
- mbjt should scale both the BJT area and the well‑diode (you already do this in z91g).

Bottom line
Sebas’s fit is reproducing the high‑Vd snapback. The single biggest structural difference left in your port is M2’s body reference. Put M2.B at ground and remove its junction diodes from the floating‑body KCL; that change alone should convert your “smooth tongues” into the late‑Vd jump without inventing extra physics. If the jump still doesn’t appear, the next suspect is self‑heating, which you can test quickly by bumping M1’s temperature as a proxy before implementing full SHMOD.
