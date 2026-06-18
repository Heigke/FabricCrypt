# grok response (grok-4-latest) — 344s

### Q1: Divergence from BSIM4 v4.8.3 IIMOD
Pyport's `compute_iimpact` (leak.py lines 45-93) diverges from BSIM4 v4.8.3 (b4ld.c lines 2047-2086) in three places:

1. T2 computation (line 66): pyport uses `T2 = (alpha0 + alpha1 * leff) / leff`, spec uses `T2 = alpha0 + alpha1 * Leff` (no division by Leff).

2. Final multiplier (line 89): pyport uses `T1 * Idsa_Vdseff` (pre-SCBE Ids or fallback to post-SCBE Ids), spec uses `T1 * (Ids / Vdsat)` (post-SCBE Ids divided by Vdsat). Pyport also assumes "Idsa" includes *Vdseff (per comment), but code doesn't explicitly multiply.

3. Weak-bias arm (lines 76-80): pyport uses `T2 * MIN_EXP * diff` for diffVds <= beta0/EXP_THRESHOLD, spec sets Iii=0 (no linear floor for positive small diff).

These cause pyport to underpredict Iii by ~27 OoM at small diffVds (e.g., via extra small Idsa in subthreshold).

### Q2: Most likely cause
(b) Ids_M1 is essentially zero in subthreshold (formula needs Ids·...). Code (leak.py line 89) multiplies by Idsa_Vdseff (~5e-36 A from json, subthreshold), suppressing Iii to ~1e-48 A despite T1~1e-13. At OP, M1 Vgs=0.6-1.866=-1.266V (deep subthreshold), Ids tiny. (a) unlikely: Vdseff can't exceed Vds=0.13V (code clamps). (c) no explicit 1e-48 clamp, but underflow plausible.

### Q3: Recommended fix
leak.py:67 - Change `T2 = tmp / leff` to `T2 = tmp` (remove /leff to match spec).

leak.py:89 - Change `Iii = T1 * Idsa_Vdseff` to `Iii = T1 * (dc_result.Ids / dc_result.Vdsat)` (correct multiplier; assumes dc_result exposes Vdsat from dc.py - add if missing).

leak.py:80 - Change `T1_weak = T2 * MIN_EXP * diff_safe` to `T1_weak = torch.zeros_like(diff_safe)` (spec sets 0 for small positive diff).

Verifies Iii~1e-21 A at strong bias (large diffVds, non-zero Ids). Re-run cell fixes solver to physical Vb<Vd.

(248 words)
