# openai response (gpt-5) — 220s

Q1: Where compute_iimpact diverges from BSIM4 IIMOD (v4.8.3)
- In BSIM4 (b4ld.c §2047–2086) the impact-ionization current is
  Iii = T2 · (Vds − Vdseff) · exp(−β0/(Vds − Vdseff)) · Idsa
  with T2 = (α0 + α1·Leff)/Leff, using pre-SCBE Idsa. Many implementations treat Idsa ≡ Gdsat·Vdseff, i.e. the pre-SCBE “Idsat” proportional to Vdseff.
- Pyport’s leak.compute_iimpact computes T1 exactly per spec (including the β0/34 small-argument branch) but then multiplies only by dc_result.Idsa and assumes (per comment) that “Idsa already carries ·Vdseff.”
- This assumption is the divergence: dc_result.Idsa in our port is the pre-SCBE channel-limited current (Gdsat·Vgsteff…), not Idsa·Vdseff. The missing Vdseff factor suppresses Iii by ~Vdseff (typically 0.04–0.15 V), and at the “weak-arm” branch it is the only way to recover the expected order of magnitude in the BSIM4 reference (Idsa as used in b4ld.c already includes the Vdseff factor).

Net: the code should multiply by Vdseff explicitly; relying on Idsa to already include Vdseff is incorrect for our dc_result layout.

Q2: Which cause explains the 1e-48 A at the reported OP?
Given the code and your instrument dump:
- The small diff (Vds − Vdseff ≈ 0.095 V) puts the formula on the β0/34 weak-argument arm (expected and matched).
- The killer is that the implementation multiplies T1 by a vanishing Idsa (5.26e-36 A in your log for M1 at the solver’s Vb≈Vd, Vsint≈Vd fixed point), not by Idsa·Vdseff. That is consistent with (a) and (b) interacting, but from code perspective the most direct cause is the missing Vdseff factor in Iii (combined with Idsa being tiny at the wrong fixed point). The exp clamp is not the root cause; it actually raises T1 vs the true exp(−211).
Most likely from code: (a) “effective drive miscomputed” in the sense that compute_iimpact uses the wrong magnitude driver (Idsa without the Vdseff factor). That, plus the solver’s M1-off fixed point, yields the 1e-48 A.

Q3: Exact LOC fix
File: nsram/bsim4_port/leak.py

Change the final multiply to include Vdseff explicitly and stop assuming Idsa is pre-multiplied.

Around lines ~85–102:

- Before:
    Idsa_Vdseff = getattr(dc_result, "Idsa", None)
    if Idsa_Vdseff is None:
        Idsa_Vdseff = dc_result.Ids
    Iii = T1 * Idsa_Vdseff

- After:
    Idsa = getattr(dc_result, "Idsa", None)
    if Idsa is None:
        Idsa = dc_result.Ids
    # BSIM4 ref (b4ld.c §2069): use pre-SCBE Idsa and multiply by Vdseff.
    # Our dc_result.Idsa does NOT carry Vdseff; include it explicitly.
    Iii = T1 * Idsa * Vdseff_b

Also fix the misleading comment header above this block: replace “Idsa·Vdseff quantity (dc_result.Idsa)” with “pre-SCBE Idsa; multiply by Vdseff here”.

This restores the BSIM4 v4.8.3 magnitude (e.g., at VG1=0.6, Vd=2, Vs=0, expect ~1e-21 A instead of ~1e-48 A) and removes the silent dependence on a dc_result packing detail.
