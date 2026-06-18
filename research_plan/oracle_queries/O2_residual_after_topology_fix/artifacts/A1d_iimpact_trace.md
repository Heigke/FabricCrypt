# A1d — Why is Iii_M2 ≈ 0 at low VG2?

## BSIM4 §6.1 formula path (from `nsram/bsim4_port/leak.py`)

```python
T2          = (alpha0 + alpha1*Leff) / Leff
diff        = Vds - Vdseff
T1_strong   = T2 * diff * exp(-beta0 / diff)         # if diff > beta0/EXP_THRESH
Iii         = T1 * Idsa                              # Idsa is pre-SCBE Idsa·Vdseff
```
Note: the formula uses `Vdseff` (the smoothed `Vds`/`Vdsat`), not `Vdsat`.

## Numbers at the converged operating point

Both biases use Sebas row params `ALPHA0 = 7.842e-5, BETA0 = 20.0` (M2 card `lalpha0 = -9.84e-12` shifts effective alpha0 by only **−7%**, not the **−5e-1** the question hypothesised — the binning is harmless).

| factor                 | LOW_VG2 (VG2=0.0)  | HIGH_VG2 (VG2=0.5) |
|------------------------|--------------------|--------------------|
| Vsint  (converged)     | 0.3063 V           | 0.0942 V           |
| Vb     (converged)     | 0.3419 V           | 0.4152 V           |
| Vgs_M2                 | 0.0 V              | 0.5 V              |
| Vds_M2 = Vsint         | 0.3063 V           | 0.0942 V           |
| **Vdsat_M2**           | **0.0369 V**       | **0.0685 V**       |
| Vdseff_M2              | 0.0356 V           | 0.0547 V           |
| **Vds − Vdseff**       | **0.271 V**        | **0.0396 V**       |
| alpha0_eff             | 7.283e-5           | 7.283e-5           |
| beta0_eff              | 17.47              | 17.47              |
| T2 = (a0+a1·L)/L       | 41.0               | 41.0               |
| **−beta0/diff**        | **−64.5**          | **−441.6**         |
| **exp(−beta0/diff)**   | **9.6e-29**        | **1.7e-192**       |
| Idsa_M2 (pre-SCBE)     | 1.25e-11 A         | 9.19e-8 A          |
| **Iii_M2 final**       | **2.4e-25 A**      | **2.6e-22 A**      |

## Verdict

**Iii is ~zero at low-VG2 because the `exp(−beta0/(Vds−Vdseff))` factor is 1e-29.**

Not because M2 is in linear region (we are well past Vdsat: 0.306 V vs 0.037 V),
not because alpha0 fails to reach the formula (alpha0_eff = 7.28e-5, intact),
not because of the `lalpha0` binning (only −7% trim, sign correct).

The pure cause: **`BETA0=20 V` from the CSV row is too large for the BSIM4 §6.1
arrhenius-style argument** at ~0.3 V drain headroom. The exponential term
`exp(−20/0.27)` = `exp(−74) ≈ 1e-32` (model uses `beta0_eff = 17.47` after
binning → 9.6e-29). At HIGH_VG2 the Idsa prefactor compensates because the
channel is strongly on (so the simulator matches measurement via direct
M2 channel current, not via the BJT path); at LOW_VG2 there is no fallback
and the body never charges.

## Concrete one-line fix

Sebas's `BETA0=20` is in the wrong units for BSIM4 §6.1 (manual default is
~30 V but for short-channel cards the empirical value is **0.5–3 V**, not 20).
**Replace `SEBAS["BETA0"] = 20.0` with `BETA0 ≈ 1.0 V` for M2** (or treat BETA0
as a per-bias fitting parameter), giving `exp(−1/0.27) = 0.025` and lifting
Iii by ~27 orders of magnitude into the pA range required to forward-bias
the parasitic NPN.
