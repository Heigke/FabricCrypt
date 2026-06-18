# grok response (grok-4-latest) — 23s

**Q1**  
Hard-pinning V_Sint=0 is not legitimate physics; it is a non-physical clamp that masks the real problem. Real 130 nm layouts have finite substrate resistance (typically 10–100 Ω·µm via well/plug), so a hard 0 V boundary condition over-constrains the internal BJT and body-effect terms. The 3-decade RMSE drop is evidence that the floating-V_Sint solver was exposing a missing or incorrect substrate-current path, not that the device is perfectly grounded. Distinguishing test: extract R_sub from a dedicated four-terminal structure on the same mask; if measured R_sub ≫ 1 Ω the pin is artificial and the model still lacks the distributed body network.

**Q2**  
Cherry-pick. VG1=0.2 is the only branch that actually exercises subthreshold and GIDL regimes that matter for leakage; discarding it to reach “0.94 dec” is regime shopping. The headline 1.619 dec figure already hides that 2.625 dec residual. Any claim of “DISCOVERY PASS” must be qualified by the full cell-wide metric over all three VG1 points; otherwise the milestone is manufactured.

**Q3**  
Rank: (a) > (d) > (c) >> (b).  
At VG1=0.2 the dominant missing term is most likely GIDL / band-to-band leakage driven by the high lateral field at the drain edge when the surface is accumulated. Single highest-value change: replace the BSIM3 GIDL current with BSIM4 v4.8.3 §6.2 formulation (equation set with NTOX, AIGC/BIGC, and field-dependent tunneling) and re-fit only the three GIDL parameters while keeping V_SINT_PIN. This directly targets the low-VG1, mid-to-high-VD region without introducing per-branch NFACTOR hacks. Measure one additional V_G1=0.1 sweep at V_D=1.8 V to confirm the exponential field dependence before refitting.
