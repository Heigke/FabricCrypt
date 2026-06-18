# A1c — Component-current trace at low-VG2 bias

**Bias:** VG1=0.6, VG2=0.0, Vd=1.5 V. Sebas CSV row applied
(ETAB=2.5, K1=0.41825, ALPHA0=7.842e-5, BETA0=20, NFACTOR=6.0,
mbjt=1.0, IS=5e-9, area=1e-6). Two-card setup
(M1=130DNWFB, M2=130bulkNSRAM).

## Solver

Plain Newton hit the documented spurious-flat-root pathology
(converged at iter 2, all currents <1e-17 A). Used
`solve_2t_with_homotopy` (gmin 1e-3 → 1e-15) seeded with
Vsint=Vd/2, Vb=0.7; converged in 3 iters at target gmin.

**Converged:** Vsint = +0.3063 V, Vb = +0.3419 V → Vbe = +0.0356 V,
Vbc = −1.158 V.

## Component magnitudes (signed, A)

| Component | Value |
|---|---|
| (a) Ids_M1 | +1.251e-11 |
| (b) Ids_M2 | +1.252e-11 |
| (c) Ic_Q1  | +2.01e-14 (NPN off, Vbe≈36 mV) |
| (d) Ibs_M2 | +6.10e-13 (forward, sub-turn-on) |
| Ibd_M2 | +1.92e-16 |
| (e) GIDL/GISL M1+M2 | 0 (exact) |
| Iii_M1 | +2.03e-16 |
| Igb (M1,M2) | 0 |
| **Id at drain** | **+1.253e-11** |
| **Id measured** | **+2.07e-5** |

## Dominant path

Drain current is **completely dominated by M2 subthreshold channel
current** (Ids_M2 ≈ Ids_M1; M2 is the series bottleneck). BJT is
~3 decades smaller; M2 body diode ~2 decades smaller; GIDL = 0.
**Predicted Id under-shoots measurement by ~6 decades** — the z91g
low-VG2 residual.

## Body-diode sanity

Ibs_M2 = 6.1e-13 A at Vbs ≈ +36 mV is **not** suspicious — Vbs is
far from pn turn-on. With jss = 1e-4 A/m² and area ≈ 6.5e-13 m², the
saturation current ≈ 6.5e-17 A; at Vbs = 0.7 V that gives ~5e-5 A,
matching Sebas's measured magnitude. The kernel is fine; **the body
simply isn't being forward-biased** — Vb sits at +0.34 V between
Vsint (+0.31) and 0 with no driver pumping it up.

## Verdict

The dominant low-VG2 path in our sim is the **series-limited M2
subthreshold channel** (~10⁻¹¹ A); Sebas's measured ~2×10⁻⁵ A is
plausibly **body-driven** — NPN forward turn-on once Vb reaches
~0.7 V, or GIDL/well leakage we don't generate (Igidl ≡ 0 from this
M2 card). Fix priority: figure out why Vb fails to climb to NPN
turn-on — with Iii ~ 1e-16 and GIDL = 0 there is no body-charging
source — and why the M2 card emits Igidl ≡ 0.
