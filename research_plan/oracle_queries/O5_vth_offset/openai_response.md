# openai response (gpt-5) — 306s

Short answer up front

(a) Top suspect: the K3 narrow‑width offset term. In your port it is coded as
    Vth_K3 = (k3 + k3b·Vbseff) · (toxe·phi / (Weff + W0))
and with this card’s numbers it is 70–80 mV at W=0.36 µm. A small error in any of:
    • which width is used (drawn Weff0 vs the later Vg‑dependent Weff),
    • W0 default (must be 2.5e‑6 m),
    • phi definition (must be 2·Vtm0·ln(NDEP/ni) + phin; no extra +0.4),
gives a Vds‑independent tens‑of‑mV shift. Your code computes this term before the Weff correction (good), but two red flags exist:
    1) phi in temp.py has an extra +0.4 V (see (b‑1) below) — this alone alters the K3 term by ~35–40 mV at this geometry; and
    2) ngspice uses Weff0 that includes wint (geometry reduction), not the post‑Vg correction; make sure sd.geom.weff is that same Weff0. If it’s the drawn W, Vth_K3 shifts by O(10 mV).
One A/B test: set k3=0 in both engines. If your −60 mV offset collapses, the whole error is in this term. If not, zero wvth0/pvth0 in both and try again.

(b) Where LDE enters in your code

There are three places LDE should touch Vth in BSIM4:

(b‑1) Tlpe1 long‑channel pocket term (pull‑UP, Vbs‑independent at Vbs=0)
In dc.py:
    T0_lpe = safe_sqrt(1.0 + lpe0 / Leff)
    Tlpe1 = (k1ox * (T0_lpe - 1.0) * sqrtPhi_pre
             + (kt1 + kt1l / Leff + kt2 * Vbseff) * TempRatio)
    ...
    Vth = ... + Tlpe1 ...
This is present and with lpe0=1.24e‑7 and L=1.8 µm gives ~+20 mV. OK.

But: phi used to form sqrtPhi_pre is built in temp.py as
    phi = Vtm0*log(ndep/ni) + phin + 0.4
It should be
    phi = 2*Vtm0*log(ndep/ni) + phin
(no +0.4 constant). Your form is ~20 mV low vs the canonical 2·Vtm, and the stray +0.4 V inflates phi by ≈+0.4 V. That 0.4 V error flows linearly into the K3 term and weakly into Tlpe1. Fix temp.py, function compute_size_dep, block “Pre‑computed Vth/Xdep quantities,” line:
    # phi = Vtm0·log(NDEP/ni) + phin + 0.4
to
    # phi = 2·Vtm0·log(NDEP/ni) + phin
This alone explains ≈+0.4 · k3 · (toxe/(Weff+W0)) ≈ +35–40 mV of Vth shift.

(b‑2) lpeb body‑bias scaling (Lpe_Vb multiplier)
In dc.py, immediately above Vth assembly:
    Lpe_Vb = safe_sqrt(1.0 + lpeb / Leff)
    Vth = ... + (k1ox*sqrtPhis - k1*sqrtPhi_pre) * Lpe_Vb ...
At Vbs=0 this factor is ~1.0, so it does not make a constant offset. OK.

(b‑3) NLX pull‑DOWN term (−k1·(nlx/Leff)·sqrtPhi)
This is missing. In BSIM4 it reduces long‑channel Vth by O(10 mV) at L≈1–2 µm.
Evidence in your code:
  • dc.py reads P["nlx"] nowhere; nlx is not used in the Vth assembly.
  • temp.py’s SCALED_PARAMS does not include "nlx", so even if dc.py read it, it wouldn’t be there.
Net: the NLX pulldown is not implemented. Its absence would make your Vth a bit high, so it is not the source of the observed low Vth. Still worth fixing:
  – Add "nlx" to SCALED_PARAMS in temp.py.
  – In dc.py, subtract k1*nlx/Leff*sqrtPhi_pre in the Vth sum (the standard BSIM4 term).

(c) ngspice hidden defaults unlikely to be the cause

ngspice’s BSIM4 (level=14) will, when vfb is not given, internally compute it from Vth0, phi, and K1 for the CV path, but Vth itself is assembled from Vth0 + body/narrow‑W/SCE/DIBL/LDE terms (no vfb). The only “hidden” behaviors relevant here:
  • binunit handling in size‑dependent vth0: lvth0, wvth0, pvth0 are applied with the 1/L, 1/W, 1/(LW) scalings in the units implied by binunit. Your port already does this in temp.py via eff.Inv_L/Inv_W/Inv_LW. If ngspice were ignoring wvth0/pvth0 you would be ≈+50 mV high (card has wvth0 < 0); you are low, so this is not it.
  • vfb default (−1 V) only enters vtfbphi2 used in the Tcen/Coxeff path. That changes absolute Id slightly; it does not produce a clean, Vds‑independent 60 mV horizontal shift with subthreshold slope intact.

(d) Instrumentation to localize in one shot

Add a term‑by‑term Vth breakdown (single bias: Vgs ≈ 0.52 V, Vds=0.5 V, Vbs=0):
  • Vth_base = type*vth0 (after L/W/P binning)
  • Vth_body = (k1ox*sqrtPhis − k1*sqrtPhi0)*Lpe_Vb − k2ox*Vbseff
  • Vth_SCE = −Delt_vth  [DVT0/DVT1/DVT2]
  • Vth_NW_SCE = −T2_narrow  [dvt0w/dvt1w/dvt2w]
  • Vth_K3 = (k3 + k3b*Vbseff)·(toxe·phi/(Weff0 + W0))  ← print Weff0 and W0
  • Vth_LPE = Tlpe1  [k1*(√(1+lpe0/Leff)−1)·√phi + temp shift]
  • Vth_DIBL = −DIBL_Sft (print T3_d, theta0vb0)
  • Vth_DITS (if active)
  • Vth_NLX (once you add it): −k1·(nlx/Leff)·√phi
  • Sum and compare to your Vth.

Code sketch (dc.py, just before “Final Vth assembly”):
    dbg = {
      "Vth_base": float(type_n*vth0),
      "phi": float(phi_pre), "sqrtPhi": float(sqrtPhi_pre),
      "Weff0": float(Weff), "W0": float(w0),
      "Vth_body": float((k1ox*sqrtPhis - k1*sqrtPhi_pre)*Lpe_Vb - k2ox*Vbseff),
      "Vth_SCE": float(-Delt_vth),
      "Vth_NW_SCE": float(-T2_narrow),
      "Vth_K3": float((k3 + k3b*Vbseff) * Vth_NarrowW),
      "Vth_LPE": float(Tlpe1),
      "Vth_DIBL": float(-DIBL_Sft),
    }
and dump dbg once for the chosen bias.

There is no public “term‑by‑term Vth” plot from BSIM‑CMG for BSIM4, but the BSIM4 v4.8.3 manual (Sec. 2.4–3.0) lists every contribution; the above decomposition maps 1:1 to those equations and to b4ld.c around lines 1099–1124.

Actionable A/B to confirm root cause

1) Zero K3: in both engines set k3=0, k3b=0 (or set W0→large to quench the term). If the −60 mV offset vanishes, the error is entirely in your K3 path. Then:
   • Fix phi (remove +0.4; use 2·Vtm0·ln), and
   • Re‑check that Weff in Vth_NarrowW is the geometric Weff0 (after wint/dwc/etc), i.e. sd.geom.weff before the Vgs/dwb correction, and that W0=2.5e−6 m.

2) If the offset persists with k3=0, set wvth0=pvth0=0 in both engines (your parser and z91j make_ngspice_card_inline currently omit wvth0; ensure ngspice is actually seeing it). If that collapses the offset, you and ngspice are treating binning differently; align temp.py’s binning with b4temp.c (binunit scaling and effective L/W).

Small but correctable issues found while reviewing

- temp.py phi uses +0.4 V and misses the factor 2. Replace:
    phi = ctx.Vtm0 * math.log(ndep/ctx.ni) + model["phin"] + 0.4
  with:
    phi = 2.0 * ctx.Vtm0 * math.log(ndep/max(ctx.ni,1e-30)) + model["phin"]

- NLX term is missing.
  • Add "nlx" to SCALED_PARAMS in temp.py.
  • In dc.py Vth sum add −k1*nlx/Leff*sqrtPhi_pre.

Why this matches your symptom

- K3 term magnitude here is ≈75–80 mV; a 0.4 V error in phi produces ~35–40 mV of that; a Weff/W0 mismatch (e.g., using drawn W, or W0=0) easily supplies the remaining ~20–30 mV. The term is Vds‑independent, giving exactly the observed constant −(58–60) mV shift. Subthreshold slope and DIBL remain close because they are controlled by nfactor/cdsc and eta0/etab/dsub, which you already match.

Summary checklist to fix

- Fix phi in temp.py to 2·Vtm0·ln(Ndep/ni)+phin (no +0.4).
- Verify Weff used in Vth_NarrowW is Weff0 (geometry‑reduced width before Vg correction).
- Confirm W0 default is 2.5e‑6 m.
- Add the missing NLX pulldown.
- Run the k3=0 A/B and the Vth term dump at Vgs≈0.52, Vds=0.5, Vbs=0; compare with hand calc.
