# A1h — IIMOD audit: does our `compute_iimpact` match Sebas's card?

**Bias:** VG1=0.6, VG2=0.0, Vd=1.5 V (M2 in saturation, Vds−Vdseff ≈ 0.27 V).
**Predecessor:** A1d already pinned that `Iii ≈ 2.4e-25 A` due to
`exp(−beta0/(Vds−Vdseff))` collapsing. This audit asks the deeper question:
**are we even using the same impact-ion formula as Sebas's foundry card?**

---

## 1. What our `compute_iimpact` implements

`nsram/nsram/bsim4_port/leak.py` (lines 44–101). Single, unconditional formula
(no `iimod` branch anywhere — verified by `grep -ri iimod nsram/bsim4_port/`):

```
T2  = (alpha0 + alpha1·Leff) / Leff                          # leak.py:80,85
diff = max(Vds − Vdseff, 0)                                  # leak.py:76-78
if diff > beta0/EXP_THRESHOLD:                               # leak.py:86,93
    T1 = T2 · diff · exp(−beta0/diff)                        # leak.py:89-90
else:
    T1 = T2 · MIN_EXP · diff                                 # leak.py:92
Iii = T1 · Idsa·Vdseff      # uses pre-SCBE Idsa (WAVE2-FIX-1)
```

Expected units (implicit from this form):
- `ALPHA0` : **m·V⁻¹** (so `alpha0/Leff` is dimensionless V⁻¹)
- `ALPHA1` : **V⁻¹**
- `BETA0`  : **V**

This is the **classic BSIM4 IIMOD=0** form (the only form available in BSIM4
versions ≤ 4.6.4). No length-binning of `alpha0/beta0` is applied (the
`lalpha0`, `lbeta0` fields ingested by `_model_card_data.py:174,180` are
**not consumed** in `temp.py` — only `voffl` etc. are length-binned).

## 2. BSIM4 IIMOD branches (manual §6.1)

BSIM4 v4.7 introduced an `IIMOD` selector. v4.8.3 manual §6.1 lists:

- **IIMOD = 0** (default): the formula above.
  `Iii = ((α₀+α₁·Leff)/Leff)·(Vds−Vdseff)·exp(−β₀/(Vds−Vdseff))·Idsa·Vdseff`
  ALPHA0 [m/V], BETA0 [V].

- **IIMOD = 1** (Mansun-Chan-style, temperature-aware): adds a temperature
  prefactor and replaces ALPHA0/BETA0 with model parameters `IIIA0`,
  `IIIA1`, `IIIB0`, `IIIB1`, `IIIT0`, `IIIT1`, etc. (different parameter
  names — a card setting `iimod=1` would *not* read alpha0/beta0 at all).

- **IIMOD = 2** (HSPICE-compatible — present in some industry forks, not
  always documented in the Berkeley v4.8.3 manual; **flag for oracle
  confirmation**): commonly cited form:
  `Iii = (α₀/Leff + α₁)·(Vds−Vdseff)²·exp(−β₀/(Vds−Vdseff))·Idsa`
  with ALPHA0 in [m·V⁻¹] but the **squared** drain-headroom factor.

Confidence: IIMOD=0/1 from manual + b4ld.c source. IIMOD=2 wording above is
my recollection of HSPICE's "alternate" form — needs cross-check.

## 3. What does Sebas's card select?

Both `M1_130DNWFB.txt:9` and `M2_130bulkNSRAM.txt:22` declare:

```
+Level = 14
+version = 4.5                 ...
```

`version=4.5` **predates IIMOD entirely** (introduced v4.7). Neither card
sets `iimod = ...`. BSIM4 default is IIMOD=0. **Sebas's card therefore
uses the IIMOD=0 classic formula** — the same one we implement.

## 4. Numeric check at the diagnostic bias

From A1d converged operating point (LOW_VG2, M2):
`Vds−Vdseff = 0.271 V`, `Idsa·Vdseff ≈ 1.25e-11 A`, `Leff ≈ 1.91e-7 m`.
Sebas-row `ALPHA0 = 7.842e-5 m/V`, `BETA0 = 20 V` (CSV; card has 18-19 V
plus `lbeta0=-9.5e-7` length term):

```
T2   = 7.842e-5 / 1.91e-7      = 410     [V⁻¹]
exp(−20 / 0.271)               = exp(−73.8) = 8e-33
Iii  = 410 · 0.271 · 8e-33 · 1.25e-11 ≈ 1e-42 A
```

A1d already showed ~2.4e-25 with binning; both are absurdly far below the
1 nA needed to forward-bias the body. **The formula is faithful; the
parameters are simply outside the regime where IIMOD=0 produces
appreciable Iii.** BETA0=18-20 V means `exp(−β/Δ)` only awakens when
`Δ = Vds−Vdseff > ~3 V` — i.e. the IIMOD=0 model targets >3.3 V drain
operation, but our diagnostic runs at 1.5 V with most of it dropped
across M1.

## 5. Verdict

> Our `compute_iimpact` correctly implements BSIM4 v4.8.3 IIMOD=0, which
> is the same branch Sebas's `version=4.5` cards select by default. The
> 6-decade Id miss is **not** an IIMOD-mismatch bug — it is that the
> classic BSIM4 §6.1 formula with BETA0≈19 V genuinely emits ~0 A at
> this bias, and the floating body cannot be charged through this path.

**Proposed one-line fix (workaround, not formula correction):** treat
`BETA0` as a regime-switching fit parameter and refit it against
Sebas's CSV using the body-current rather than as a fixed manual default
— typical short-channel values are 0.5–3 V, which would lift Iii by
~25 orders of magnitude into the nA range.

**Real fix (root cause):** Iii is unlikely to be the dominant
body-charging path at Vd=1.5 V; junction GIDL or the body-source diode
are more plausible. Re-examine `compute_gidl_gisl` and `idiode` weights
before further tuning impact-ion.

**Flag for oracle:** confirm IIMOD=2 (HSPICE) form and whether any
Sebas-internal extraction uses an HSPICE-only impact-ion equation we
have not ported.
