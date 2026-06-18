# Probe v2 — VG1=0.4 V catastrophe root cause (M3a.1)

**Date:** 2026-05-03
**Bias:** VG1=0.4 V / VG2=+0.30 V (worst-fitting, log-RMSE 3.25 dec on stage5)
**Probe script:** `research_plan/binning_audit/probe_v2_vg04_catastrophe.py`
**Output:** `research_plan/binning_audit/probe_v2_out/vg1_0.40_vg2_+0.30.{png,json}`

## Finding

The catastrophe is **not** a missing physics term and **not** a binning bug.
It is a **wrong Newton root** caused by `bjt.Bf = 5×10⁴` being too high
for the no-impact-ionisation regime.

## Evidence (per-Vd component dump)

At VG1=0.4 / VG2=+0.30, Vd = 0.05 V (low end of sweep):

| component       | predicted        |
|-----------------|------------------|
| Vb (body)       | **+0.4333 V**    |
| Vsint           | +0.0432 V        |
| Ids_M1          | +1.04×10⁻¹¹ A    |
| Ids_M2          | +1.12×10⁻¹¹ A    |
| **Ic_Q1 (NPN)** | **+7.15×10⁻⁸ A** |
| Ib_Q1 (NPN base)| +1.38×10⁻¹⁰ A    |
| Iii_M1          | +1.5×10⁻²⁵ A     |
| Iii_M2          | +1.5×10⁻²⁶ A     |
| Igidl (M1, M2)  | 0                |

The total predicted Id is **dominated by Ic_Q1** (the parasitic NPN
collector) which is 6700× larger than the channel current. Yet
**impact-ionisation is essentially zero**, so there is no physical source
for the body charge that would forward-bias the NPN. The NPN is
self-sustaining: its own base-leakage Ib_Q1 ≈ 1.4×10⁻¹⁰ A balances the
small bulk-diode currents at Vb ≈ 0.43 V.

## Cold-start seed exhaustion

All five arclength initial-guess seeds — (Vsint=0.025, Vb=0), (0.015, 0.4),
(0.010, 0.75), (0.005, 0.85), (0.05, 0.0) — converge to the SAME root
(Vb = 0.4333 V) at this bias. So this is not a "wrong-seed" bug; the
Newton system **only has one root**, and it is the wrong one.

## Bf sensitivity sweep at VG1=0.4 / VG2=+0.30

| Bf      | log-RMSE | Id[Vd=0.05] | Id[Vd=1.95] | Vb[Vd=0.05] |
|---------|----------|-------------|-------------|-------------|
| 5×10⁴ (current) | 3.24 | 7.1×10⁻⁸  | 7.4×10⁻⁶  | 0.433 V |
| 1×10³           | 1.72 | 4.7×10⁻⁸  | 1.6×10⁻⁷  | 0.421 V |
| 1×10²           | **0.89** | 1.2×10⁻⁸  | 1.7×10⁻⁸  | 0.384 V |
| 1               | 1.42 | 1.6×10⁻¹⁰ | 1.9×10⁻¹⁰ | 0.270 V |
| 1×10⁻²          | 2.55 | 9.0×10⁻¹² | 1.3×10⁻¹¹ | 0.153 V |
| measured        | —    | 1.1×10⁻⁹  | **4.1×10⁻⁶** | — |

Lower Bf gives lower aggregate RMSE but flatter prediction (no snapback
rise). No single Bf reproduces both the low-Vd off state and the high-Vd
3-decade snapback rise — the model's NPN is decoupled from the
impact-ionisation that should be its base-current driver.

## Why Bf = 5×10⁴ was chosen

The z91h grid-search picked `NSRAM_BJT_BF=5e4` because it minimised
*aggregate* log-RMSE across all 33 biases. At VG1=0.6 V (where
impact-ionisation fires hard) high Bf gives realistic snapback gain.
At VG1=0.4 V (where Iii is ~10⁻²⁵ A) high Bf produces a self-firing NPN.

## Implications for M3a

1. **Bf cannot be a global constant.** Physically, the parasitic NPN
   gain in 130 nm bulk is ~10–100; 5×10⁴ is non-physical for the
   intrinsic bipolar action. The grid-search optimum is a *fit* to
   compensate for missing physics elsewhere (likely the impact-
   ionisation triggering at high VG2/VD).

2. **Need a physically-bounded Bf** (≤ 100) and a separate
   triggering mechanism for the NPN at the snapback edge — likely
   a stronger Iii-to-Vb coupling or a lateral-NPN base current
   that depends on Vds rather than on Vb alone.

3. **Sebas's CSV has per-bias BETA0** — currently NOT loaded into
   `make_bjt()`. The current code only reads `IS`, `area`, `mbjt`
   from the CSV (`scripts/z91f_validate_with_sebas_params.py:265`).
   Loading per-bias BETA0 should be the first M3a remediation.

4. **VG1=0.4 V is a fitting boundary**, not a single bug. The model
   has the right components but the wrong gain partition. M3a.1 fix
   = re-fit Bf per-row using Sebas's BETA0 column; verify the
   shape recovers without losing snapback at VG1=0.6.

## Bf sweep across all 25 measured biases

| Bf       | median | mean | max  | p90  | VG1=0.2 | VG1=0.4 | VG1=0.6 |
|----------|-------:|-----:|-----:|-----:|--------:|--------:|--------:|
| 5×10⁴ (brief) | 1.00 | 1.60 | 3.24 | 2.90 | 1.66 | 2.83 | 0.91 |
| 3×10⁴ | 0.85 | 1.48 | 3.05 | 2.72 | 1.55 | 2.66 | 0.82 |
| **2×10⁴** | **0.80** | **1.40** | **2.89** | **2.58** | **1.46** | **2.52** | **0.78** |
| 1.5×10⁴ | 0.81 | 1.35 | 2.78 | 2.48 | 1.40 | 2.42 | 0.79 |
| 1×10⁴ | 0.86 | 1.30 | 2.62 | 2.35 | 1.33 | 2.28 | 0.81 |
| 7×10³ | 0.93 | 1.26 | 2.48 | 2.23 | 1.27 | 2.17 | 0.86 |
| 5×10³ | 1.02 | 1.24 | 2.35 | 2.12 | 1.22 | 2.06 | 0.92 |
| 3×10³ | 1.15 | 1.23 | 2.15 | 1.96 | 1.15 | 1.91 | 1.04 |

**Best Bf = 2×10⁴ → overall median 0.80 dec** (vs. the brief's 1.00 dec at
5×10⁴). Improvements over the brief's published numbers:

| metric  | brief (Bf=5e4) | optimum (Bf=2e4) | Δ |
|---------|----------------|------------------|---|
| median  | 1.00 | 0.80 | -20 % |
| mean    | 1.60 | 1.40 | -13 % |
| max     | 3.24 | 2.89 | -11 % |
| p90     | 2.90 | 2.58 | -11 % |

The trade-off is monotone: lowering Bf improves VG1=0.4 V (catastrophe
row) and VG1=0.6 V (snapback row) up to about 1.5–2×10⁴ where they
balance, then VG1=0.6 V starts to starve below 1×10⁴.

## Note on the "8 NaN biases" (M3a.2 reframing)

The 8 skipped curves at negative VG2 are **not** Newton-failure NaN. They
are biases where Sebastian's parameter CSV has `K1 = NaN`, i.e. he did
not extract bias-specific overrides for those rows (the snapback regime
at negative VG2). The current code defensively skips them. With Bf=1e4
and **no per-bias overrides**, all 33 biases evaluate to finite log-RMSE.
The "M3a.2 NaN diagnostic" can be closed as a documentation update, not
a solver fix.

## Status

- [x] Probe script written and run (probe_v2_vg04_catastrophe.py)
- [x] Diagnostic plot saved (vg1_0.40_vg2_+0.30.png)
- [x] Bf sensitivity confirmed (5 Bf values × 1 bias)
- [x] **Bf sweep across all 25 biases — Bf=1e4 wins by 14 %**
- [x] **8 NaN biases identified as Sebas-CSV-missing, not solver-fail**
- [ ] Apply Bf=1e4 in z91g and rebuild brief headline numbers
- [ ] Investigate Bf=3e3 (between 1e4 and 1e3) for further gain
