NS-RAM project. pyport's BSIM4 compute_iimpact() produces 1e-48 A at M1
strong-bias (V_G1=0.6, V_d=2.0, Vsint=0). BSIM4 IIMOD physics expects
~1e-21 A (27 OoM gap). Find the bug via code review.

Attached:
- leak.py: compute_iimpact implementation
- nsram_cell_2T.py: residual that calls it
- M1 card: alpha0=7.84e-5, beta0=20 (per Sebas card)
- Instrument output showing 1e-48

Q1: Compare leak.py compute_iimpact to BSIM4 v4.8.3 reference IIMOD
formula. Where does pyport's implementation diverge from spec?

Q2: The instrument at V_G1=0.6, V_d=2.0 reported "Vds-Vdseff = 0.0947".
This was for M2 (subthreshold V_G2=0.20). For M1 at V_G1=0.6 V_d=Vsint=1.87,
expected Vdseff ≈ Vov ≈ 0.1V, so Vds-Vdseff ≈ 1.77V. Either:
(a) M1's Vdseff is being computed wrong (maybe ~1.7V?)
(b) Ids_M1 is essentially zero in subthreshold (formula needs Ids·...)
(c) There's a clamp/clip driving result to 1e-48
Which is most likely from the code?

Q3: Show the exact LOC fix you'd recommend. Be concrete. Reference
specific file:line.

≤500 words per oracle.
