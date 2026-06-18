# openai response (gpt-5) — 365s

Short answer
Your Vgsteff denominator is too small because coxe/cdep0 is too small in the port. The culprit is cdep0: it is computed in temp.py with the wrong permittivity constant (units mismatch between EPS0 used for Cox and εSi used for cdep0). In the port, Cox uses EPS0 in SI (F/m) but cdep0 uses εSi built from EPS0 in cgs (F/cm). That makes cdep0 larger by ≈√(10^4)=100 relative to Cox, which in the Vgsteff bridge is indistinguishable from a +60–80 mV shift in voffcbn and gives exactly the 5–10× subthreshold excess you see on the isolated device.

What to change
- Make sure both coxe and cdep0 are computed with the SAME base ε0 in SI:
  - EPS0 must be 8.854187817e-12 F/m (not 8.854e-14).
  - epssub must be 11.7·EPS0 (F/m), not 11.7·8.854e-14.
- Then keep your existing cdep0 formula:
  cdep0 = sqrt(q·epssub·NDEP_SI/(2·phi))
  with NDEP_SI = ndep·1e6.

Why this explains all your observations
- In the Vgsteff bridge, T9 = mstar + n·(coxe/cdep0)·exp((voffcbn − (1−mstar)·Vgst)/(n·Vt)).
- If cdep0 is too big because εSi was in F/cm while Cox was in F/m, (coxe/cdep0) is too small by roughly 10–100×. That makes T9 much smaller, raising Vgsteff and the subthreshold Id by exactly the 5–11× you measure at low Vgs.
- Sweeping voffcbn by +60–80 mV fixes it because changing voffcbn only multiplies exp(T2_off); with the wrong cdep0 ratio, a +60–80 mV change produces the same multiplicative factor in T9 that the correct coxe/cdep0 ratio would.
- The slope discrepancy (S ≈ 76.2 vs 72.5 mV/dec) remains small because it is mainly set by your n-aggregator; fixing cdep0 corrects the amplitude (vertical) error, while your A.5.f Theta0_n fix already brought the slope close.

How to verify quickly
1) Print the numbers used by the port for coxe and cdep0 for this device.
   Expected at 300 K with toxe=4 nm and ndep=1.7e17 cm^-3:
   - coxe ≈ 8.63e-3 F/m^2
   - cdep0 ≈ 1.25e-3 F/m^2
   → coxe/cdep0 ≈ 6.9
   If you see cdep0 ≈ 1.25e-1 F/m^2 (100× too big), that’s the bug.

2) After fixing ε0/εSi units, rerun Id–Vgs. The 5–11× excess on isolated M2 should collapse without touching voff/voffl.

Side notes
- Your voff, voffl, mstar, Vth, and n implementations are fine.
- vfb doesn’t enter the DC Vgsteff bridge for capmod=2; not the source.
- Keep NDEP→m^-3 conversion by ×1e6; that was already correct.
