# O6 — pyport `n` (subthreshold-slope) is 12% larger than ngspice on same card

## Setup

PyTorch BSIM4 v4.8.3 port (`nsram/nsram/bsim4_port/dc.py:compute_dc`) cross-validated against ngspice v42 (level=14 BSIM4) on Sebas Pazos's `M2_130bulkNSRAM.txt` card, isolated NMOS, body=GND, L=1.8 µm, W=0.36 µm.

## What we measured

At Vds=0.05V, Vbs=0, sweep Vgs ∈ [0.30, 0.83] V:

- pyport S = 80.6 mV/dec
- ngspice S = 71.9 mV/dec
- ratio = 1.121 (pyport 12% slower decay)

pyport's reported `n` (via DCResult) = **1.2298**.
Implied ngspice `n` ≈ 1.097 (assuming S = n·ln10·kT/q with shared mstar).

Δlog10(Id) = log10(Id_py) − log10(Id_ng) is monotonically decreasing with Vgs:
- Vgs=0.30: +1.03 dec (pyport HIGH in deep subthresh)
- Vgs=0.50: +0.65 dec
- Vgs=0.65: +0.28 dec
- Vgs=0.83: +0.09 dec (saturation; gap → 0)

The signature of a subthreshold-slope `n` that's too large.

Also confirmed earlier (A9 hand-derivation): pyport BSIM4 Vth_eff matches ngspice's `show m1: vth` to 90 µV. So Vth aggregator itself is fine. Bug is in `n`.

Also confirmed: phi formula was wrong (missing factor of 2, spurious +0.4) — fix applied at temp.py:285. Net Vth shift: ~3 mV. The 12% n discrepancy persists after the phi fix.

## Card values (M2_130bulkNSRAM)

```
vth0=0.54153, k1=0.63825, k2=-0.03, k3=65.28
nfactor=1.58, etab=-0.087, cdsc=0.00024, cdscb=0, cdscd=0 (default), cit=0 (default)
toxe=4e-9, ndep ≈ 2.6e18, ngate=2.5e20
phin=0.05, vfb=-1 (default)
dvt0=2.4, dvt1=0.53, dvt2=-0.032, dsub=0.6412, drout=1.354
pdiblc1=3.38, pdiblc2=0.002, pdiblcb=0
```

## Code we suspect

`nsram/nsram/bsim4_port/dc.py:340-372` — Vth assembly + n aggregator:

```python
tmp_dsub = sqrt((epssub / (epsrox·EPS0) · toxe · Xdep0).clamp_min(1e-40))
T0_dsub = dsub · Leff / tmp_dsub
theta0vb0 = exp_threshold_branch(T0_dsub)        # this is Theta0 for DIBL_Sft
DIBL_Sft  = T3_clamped · theta0vb0 · Vds

# Vth assembly
Vth = type_n·vth0
    + (k1ox·sqrtPhis - k1·sqrtPhi_pre)·Lpe_Vb
    - k2ox·Vbseff
    - Delt_vth
    - T2_narrow
    + (k3 + k3b·Vbseff) · Vth_NarrowW
    + Tlpe1
    - DIBL_Sft

# n aggregator (the suspect):
tmp1 = epssub / Xdep
tmp2 = nfactor · tmp1
tmp3 = cdsc + cdscb · Vbseff + cdscd · Vds
tmp4 = (tmp2 + tmp3 · Theta0 + cit) / coxe
n = 1 + tmp4
```

`Xdep` here is the **bias-adjusted** depletion thickness (uses `sqrtPhi_phi` which depends on `Vbsh`/`Vbseff`), distinct from `Xdep0` used in `theta0vb0`. `Theta0` is the same `theta0vb0` from the DIBL block.

## Questions

1. **Where do you bet the 12% lives?** Most likely candidates:
   - **Xdep**: BSIM4 v4.8.3 b4ld.c uses bias-adjusted `Xdep = Xdep0·sqrtPhi_phi/sqrtPhi`. If we accidentally use `Xdep0` instead of `Xdep`, n is wrong by sqrtPhi_phi/sqrtPhi ratio (but this is roughly 1 at Vbs=0 — unlikely).
   - **`Theta0` definition**: the `Theta0` in the `n` formula (BSIM4 manual eq 3.2-1 calls it `Θ_0(Vbs)`) is a different smoothing function from the `theta0vb0` used in DIBL. b4ld.c §1141 has `Theta0_n` that depends on `T1_n = exp(-DVT1·Leff/(2·lt1))`. If we reuse `theta0vb0` from DIBL (which uses `dsub` and a different `lt`), n is wrong.
   - **coxe**: should be `coxe` (effective oxide cap including poly-depletion correction) or `cox` (raw)? At nominal bias they differ by ~5%.
   - **mstar mixing**: if our subthreshold-slope reporter uses `n·Vtm·log1p(exp(...))` but our Vgsteff bridge has `n·Vtm/T9` denominator with T9 =mstar+n·T3 — the EFFECTIVE slope might involve mstar reciprocally, and an mstar mismatch could give 12%.
   - **Vbseff smoothing**: at Vbs=0 the Vbsh/Vbseff smoothing routine should give Vbseff=0, but if our smoothing has an offset or sign issue, Cdscb·Vbseff and k1ox·sqrtPhis·(Lpe_Vb) terms could leak.
2. **Which BSIM4 reference variable does pyport's `n` formula use for `Theta0`?** Read the code we attached and identify if it's the right one.
3. **One-line A/B test**: what single env-var or model-param flip would isolate which of the candidates above is the source?
4. **If a fix is obvious from the code**: what file:line change closes the 12%?

## Files attached

- `nsram/nsram/bsim4_port/dc.py` (full)
- `nsram/nsram/bsim4_port/temp.py` (compute_size_dep, the post-phi-fix version)
- `data/sebas_2026_04_22/M2_130bulkNSRAM.txt`
- `results/z91m_vgsteff_inspect/{vgsteff_inspect.png, summary.json}` (the diagnostic data table)
- `results/z91l_vth_dibl/summary.json` (Vth/DIBL across Vds)
- `results/z91k_subthreshold_slope/summary.json` (S extracted by linear fit)
- `scripts/z91m_vgsteff_inspect.py` (the dump script)

Be specific. We have a 1-decade subthreshold gap on every curve in z91g; closing this is the single largest remaining lever in Phase A fidelity (currently median 0.99 dec; target ≤0.30 dec).
