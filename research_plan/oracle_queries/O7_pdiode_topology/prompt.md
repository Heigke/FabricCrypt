# O7 — pre-predict the missing parasitic pdiode at NS-RAM 2T floating body

## Setup
2T NS-RAM cell, 130 nm bulk CMOS, Sebastian Pazos. Topology (per his `2tnsram_simple.asc`):

- **M1**: nmos4, L=0.18 µm, W=0.36 µm. Drain=Vd, Gate=VG1, Source=Sint, Body=B (floating).
- **M2**: nmos4, L=1.8 µm (10× M1), W=0.36 µm. Drain=Sint, Gate=VG2, Source=GND, Body=GND (LTSpice default for unconnected).
- **Q1**: parasiticBJT NPN macro, area=1u. Collector=Vd, Base=B, Emitter=GND, bf=10000.
- **C1 (CBpar)**: 1 fF capacitor, Sint↔Vd, Rser=1m. Already in our model.
- **vnwell**: deep-N-well bias = +2 V (chip-level, fixed). Drives a forward DNW→P-body diode (we modelled this at body↔vnwell with Js≈3.4e-7 A/m², Rs≈1e11 Ω).

Body node B is shared by M1.B and Q1.B. Floats. Vb is one of the two unknowns (along with Vsint) we Newton-solve at every (Vd, VG1, VG2).

## What Sebas just told us (2026-05-02)

> "Quick update! I found that my simulation had a parasitic diode that wasn't working as expected (or wasn't working at all, actually), altering my dynamic response. I'm updating my schematic with an additional diode for clarity (**pdiode with area 5 um x 4.4 um in accordance with my implementation**) to reflect the **capacitive response of the floating body** in detail.
> 
> For the sake of simplicity, the diode could be replaced with a **linear capacitor (somewhere in the range 5~10 fF)**, although that would be less electrically accurate because it won't capture the **capacitance dependence on P-body voltage** (but much kinder to computational cost). I'm attaching a SPICE model card for the diode, just in case (please do not share). There's still room to adjust parameters and dependencies, but this in principle sorts out our issues around dynamic behaviours."

We do NOT yet have the schematic image or the SPICE model card text — they were attachments. Want to pre-stage the integration into our PyTorch port so it's a single edit when the card arrives.

## Hard facts from his message

- Diode is pdiode (P-on-N junction). On a P-body in N-substrate context, this is most likely the **P-body to N-substrate / N-well isolation junction** — i.e., the bottom-plate junction of the floating body to the surrounding N material.
- Area: 5 µm × 4.4 µm = **22 µm²**.
- Equivalent linear capacitance: 5–10 fF. → Cj0/area = 0.23–0.45 fF/µm². This is in the realistic range for a p+/n-well or p-body/n-substrate junction at zero bias in 130 nm bulk CMOS.
- "captures capacitance dependence on P-body voltage" → Cj has voltage-dependent profile (Cj = Cj0/(1−Vd/Vj)^M with typical M=0.5, Vj=0.7).
- "sorts out our issues around dynamic behaviours" → the diode adds a body-charging path that wasn't previously modelled. For DC steady state this means an extra Js·area·exp(Vbody/(n·Vt)) drainage/charging current depending on the diode polarity at the body node.

## Our current model (pyport, `nsram/nsram/bsim4_port/nsram_cell_2T.py:_residuals`)

R_B (currents INTO body B) =
  + m1["Iii"] (impact ionization, +INTO body)
  + m1["Igidl"] + m1["Igisl"] + m1["Igb"] (gate tunnel + GIDL/GISL paths)
  − m1["Ibs"] − m1["Ibd"] (M1 source/drain body diodes — out of body)
  − Ib_Q1 (NPN base current — out of body)
  + I_well_body (DNW well-body diode pumping body up at vnwell=+2V)

We have NO term for a body-to-substrate or body-to-some-other-explicit-node diode of size 22 µm². If Sebas's pdiode is between the body and substrate (i.e., between B and 0/GND), it adds:

  I_body_pdiode = Js·22e-12 · (exp((Vb − Vsubstrate)/(n·Vt)) − 1)
                ≈ Js·22e-12 · (exp(Vb/(n·Vt)) − 1)   if Vsubstrate=0

Sign: if Vb > 0, this is positive → flows OUT of body (into substrate), so −I_body_pdiode in R_B.

## Questions

1. **Which two nodes does the diode most likely connect?** Candidates (rank with confidence):
   - body ↔ substrate (GND): "P-body to N-substrate" junction — but bulk substrate in 130 nm is typically p-type, so a p-body in p-substrate doesn't form a junction. Unless his cell uses an N-well isolation under the body (typical NS-RAM topology).
   - body ↔ DNW (vnwell=+2V): we already have this. But area is 22 µm², much larger than our existing junctions.
   - body ↔ source/drain of one of the MOSFETs: but those are already covered by M1's Ibs/Ibd.
   - body ↔ Sint: Sebas's CBpar is already across Sint↔Vd, but a body↔Sint diode would be a new topology element.
   - body ↔ G2 or VG2: capacitive coupling from gate.
2. **What signs of the SPICE card parameters** are most likely? Js, n, Cj0 (= 5-10 fF / 22 µm²), Vj (typically 0.6-0.8 V), M (typically 0.3-0.5).
3. **At our Vb=0.487 V steady state at VG1=0.4/Vd=1V**, what would adding a 22 µm² body-to-substrate forward-biased pdiode do quantitatively? Estimate the extra body drainage current.
4. **What single env-var-controlled patch** to `nsram_cell_2T._residuals` would let us test 3 candidate topologies (body↔GND, body↔Sint, body↔vnwell-extra) at default magnitudes once we get the SPICE card?

Be specific. We need this prediction to be a one-edit integration when the card arrives — likely tomorrow.

## Files attached (none — this is pure prediction)

Just the prompt. We have everything in this single document.
