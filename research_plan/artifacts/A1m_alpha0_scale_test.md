# A1m — alpha0 / beta0 brute-force scale test

**Bias:** worst diagnostic point VG1=0.6, VG2=0.0, Vd=1.5 V.
**Goal:** Decide between (i) alpha0/beta0 scale issue or (ii) missing
body-charging path, by scaling alpha0 (×1…×1e4) and shrinking beta0
(20→0.5). Both `sd_M1.scaled["alpha0"]` and `sd_M2.scaled["alpha0"]`
patched (and same for beta0) — no source edits.

Script: `A1m_demo.py` · raw data: `A1m_alpha0_scale_test.json`.

## ALPHA0 sweep at the WORST bias (beta0=20)

| scale | alpha0     | Vb (V)  | Iii_M1     | Iii_M2     | Ic_Q1      | Id_pred    | conv  |
|------:|-----------:|--------:|-----------:|-----------:|-----------:|-----------:|:------|
|     1 | 7.84e-05   | -0.253  | 1.6e-27    | 4.3e-26    | 5.0e-17    | 5.7e-17    | True  |
|    10 | 7.84e-04   | -0.260  | 1.4e-26    | 4.4e-25    | 5.0e-17    | 5.7e-17    | True  |
|   100 | 7.84e-03   | -0.316  | 6.8e-26    | 4.3e-24    | 5.0e-17    | 5.7e-17    | True  |
|  1000 | 7.84e-02   | **+0.183** | 9.9e-12 | 4.0e-23   | 6.0e-12    | 1.5e-10    | **False** |
| 10000 | 7.84e-01   | +0.108  | 2.2e-14    | 4.1e-22    | 3.3e-13    | 6.5e-13    | False |

## BETA0 sweep at the WORST bias (alpha0=7.84e-5)

| beta0 | Vb (V)  | Iii_M1   | Iii_M2   | Ic_Q1    | Id_pred  | conv  |
|------:|--------:|---------:|---------:|---------:|---------:|:------|
| 20.00 | -0.253  | 1.6e-27  | 4.3e-26  | 5.0e-17  | 5.7e-17  | True  |
| 10.00 | **+0.108** | 2.3e-14 | 3.3e-25 | 3.3e-13 | 6.8e-13 | **False** |
|  5.00 | -0.134  | 1.0e-23  | 2.8e-14  | 7.8e-17  | 8.5e-17  | True  |
|  2.00 | -0.131  | 6.9e-19  | 5.3e-13  | 8.1e-17  | 8.8e-17  | True  |
|  1.00 | -0.131  | 3.3e-18  | 2.2e-12  | 8.1e-17  | 8.8e-17  | False |
|  0.50 | -0.269  | 6.9e-18  | 1.6e-12  | 5.0e-17  | 5.7e-17  | False |

Reference biases (VG1=0.4 / VG1=0.6,VG2=0.5) show similar pattern:
**Vb stays ≤ +0.18 V across the entire 4-decade alpha0 sweep and the
40× beta0 sweep**; for the BEST bias (VG2=0.5) Vb actually drives more
*negative* (-1.5 to -2.5 V) at every scale. Full numbers in the JSON.

## Verdict

**Vb does NOT climb past 0.5 V at ANY tested alpha0 (up to 7.84e-1,
10000× baseline) or beta0 (down to 0.5).** The maximum Vb observed
across the entire WORST-bias sweep is +0.183 V (alpha0×1000,
non-converged), and that point already has Iii_M1=9.9e-12 A driving Id
to 1.5e-10 A — i.e., Iii alone (without help from Q1 base current at
high Vb) is producing 4 decades less Id than measured (2.07e-5 A), and
Vb is still pinned far below the BJT turn-on knee. Convergence breaks
at the high-impact scales (×1000, ×10000, beta0=10/1/0.5) before any
high-Vb basin appears.

**This is case (ii): missing body-charging physics.** Even with alpha0
inflated by 4 decades — far beyond any plausible SPICE↔BSIM4 unit
mismatch — the residual R_B keeps the body in deep cutoff. There is
no nearby fixed point at Vb≈0.7; the geometry of (Iii − BJT − diodes)
genuinely lacks a DC source strong enough to clamp the body high
without help from a path we haven't modeled.

## Implication for z91h

A pure residual fit on (alpha0, beta0, GIDL, Bf) **cannot** close the
5-decade Id gap at the WORST bias. The worst bias requires an
additional body-injection mechanism — candidates: (a) tunnel/avalanche
in the M2 source/drain–body junction at high Vds (BSIM4 has knobs we
don't have lit: AGIDL/BGIDL on the *drain* side feeding M2 body, or
Iisub from the parasitic vertical NPN inside the deep-N-well stack we
ignore), (b) explicit DIBL-modulated Iii with much steeper Vds/Vgs
dependence than the standard BSIM4 II model provides, or (c) a forward-
biased substrate-to-bulk parasitic that Sebas's SPICE deck includes
implicitly. z91h should be re-scoped: treat alpha0/beta0/Bf as
nuisance parameters (small effect) and add a parametric "missing
path" term — e.g., Iextra = K · exp(λ·Vds) · sigmoid(VG1−Vth) — fit
K, λ jointly with the others. Without that extra term, no amount of
alpha0/beta0 tuning will lift Vb out of cutoff at the worst bias.
