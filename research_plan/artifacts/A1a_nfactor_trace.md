# A1a — NFACTOR override trace through compute_dc

**Verdict:** **NFACTOR override IS reaching the subthreshold formula.**

## 1. Formula path inside `compute_dc` (`nsram/bsim4_port/dc.py`)

```python
# line 131:    P  = sd.scaled
# line 160:    nfactor = t(P["nfactor"])
# --- Subthreshold n (b4ld.c §1133-1154) ---
# line 361:    tmp1 = epssub / Xdep
# line 362:    tmp2 = nfactor * tmp1                       # <-- nfactor enters here
# line 363:    tmp3 = cdsc + cdscb*Vbseff + cdscd*Vds
# line 364:    tmp4 = (tmp2 + tmp3*Theta0 + cit) / coxe
# line 367:    n_a  = 1.0 + tmp4                           # subthreshold slope factor
# --- Vgsteff bridge (b4ld.c §1238-1296) ---
# line 439:    T0v = n * Vtm                               # n carries nfactor
# line 440:    T1v = mstar * Vgst
# line 441:    T2v = T1v / T0v
# line 449:    T10_bridge = n*Vtm * log1p(exp(T2v))
# line 471:    T9v  = mstar + n * T3v
# line 472:    Vgsteff = T10v / T9v                        # <-- subthreshold smoothing
```

So `nfactor → n → T0v → Vgsteff` (and into the drain-current expression).
It does not change `Vth`; it sets the inverse subthreshold slope (kT/q · n).

## 2. Patch idiom reaches that path

`patch_sd_scaled` in `scripts/z91f_validate_with_sebas_params.py` (lines 55–74) writes directly to `sd.scaled[k]`:

```python
sd.scaled[k] = v            # line 67
```

`compute_dc` reads `P = sd.scaled; nfactor = t(P["nfactor"])`. Same dict, same key. The override is consumed.

## 3. Numeric demo (M2, real card)

Script: `research_plan/artifacts/A1a_demo.py`. Loads
`data/sebas_2026_04_22/M2_130bulkNSRAM.txt`, builds `sd_M2` via
`compute_size_dep` (Ln·10 = 1.8 µm, W = 360 nm, T = 27 °C), applies the
same M2 static overrides as z91f (k1, k2, etab, beta0), then calls
`compute_dc` at **(Vgs = −0.10, Vds = 2.0, Vbs = 0)** with `nfactor` swapped
in `sd.scaled`.

| nfactor | Id [A]      | Vgsteff [V] | Vth [V] |
|--------:|------------:|------------:|--------:|
| 1.58    | 1.1545e-14  | 4.649e-09   | 0.4497  |
| 12.15   | 6.2099e-12  | 2.501e-06   | 0.4497  |

- **Id ratio = 5.38e+02 → +2.73 decades**
- ΔVgsteff = +2.50e-6 V (Vgsteff scales ~ n·Vtm·log1p(exp(...)) — bigger n softens the bridge)
- ΔVth = 0 (expected — nfactor sets slope, not threshold)

## 4. Implication for z91g

The override mechanism is wired correctly: a single-bias `nfactor` swap from
1.58 → 12.15 already moves Id by **+2.73 decades** at this low-VG2 bias.
That is the same order of magnitude as z91g's reported median residual
(2.40 decades at low VG2), so NFACTOR is the right knob and the patch
*does* reach it. The remaining z91g residual must therefore come from
either (a) sign / direction of the residual (does Sebas's Id move the
same way ours does?), (b) interaction with the M1 overrides
(etab, k1, alpha0, beta0) or the BJT wrapper, or (c) bias coordinates the
CSV NFACTOR row maps to versus what we apply at the same (VG1, VG2). The
NFACTOR-not-reaching-the-formula hypothesis is **falsified**.
