# Oracle Query O5 — Constant 60 mV Vth Offset (PyTorch BSIM4 port vs ngspice v42)

## Setup (terse)

- DUT: M2_130bulkNSRAM card — Sebas Pazos's 130 nm bulk NMOS, used as the upper transistor in a 2T NS-RAM cell. Model card attached: `M2_130bulkNSRAM.txt`.
- Geometry probed: **L = 1.8 µm, W = 0.36 µm** (NF=1, AS/AD ≈ W·0.5 µm).
- Reference: **ngspice-42 with level=14 (BSIM4)** running the SAME model card we feed to our port.
- Our port: `nsram/nsram/bsim4_port/{dc.py, temp.py, model_card.py, _model_card_data.py}` (all attached, `_model_card_data.py` renamed to `model_card_data.py` so the dispatcher uploads it).
- Test harness: `z91j_ngspice_isolated_m2.py` (Id-Vds harness used to call ngspice), `z91k_subthreshold_slope.py` (S extraction), `z91l_vth_dibl.py` (constant-current Vth + DIBL).
- Constant-current Vth criterion: `Id_target = (W/L) · 1e-7 = 0.36/1.8 · 1e-7 = 2e-8 A` at the specified Vds.

## Card-resident params relevant to Vth (from M2_130bulkNSRAM.txt)

```
vth0   = 0.54153
k1     = 0.63825
k2     = -0.03
k3     = 65.28
k3b    = 0          (default)
lpe0   = 1.244e-7   (m)
lpeb   = 0          (default)
nlx    = 1.97e-7
dvt0   = 2.4
dvt1   = 0.53
dvt2   = -0.032
dvt0w  = 0
dvt1w  = 5.3e6      (BSIM4 default)
dvt2w  = -0.032     (default)
nfactor = 1.58
etab   = -0.087
cdsc   = 2.4e-4
cdscb  = 0
drout  = 1.354
dsub   = 0.641
pdiblc1 = 3.38      ← unusually large
pdiblc2 = 0.002
phin   = 0          (default; not in card)
vbm    = -3         (default)
toxe   = 4.1e-9
ngate  = 5e20       (poly-depletion)
```

The card uses defaults (i.e. card-silent) for: `phi`, `vfb`, `vfbcv`, `phin`, `wpe`, `weffeot/leffeot`, `eta0`, all the `Lvth0/Wvth0/Pvth0` length/width/cross size-dep coefficients (none of those are in the card).

## Empirical numbers (from `vth_dibl_summary.json`, attached)

Constant-current Vth at Id_target = 2e-8 A, M2_130bulk, L=1.8 µm, W=0.36 µm, Vbs=0:

| Vds  | Vth_ngspice (V) | Vth_pyport (V) | diff (V)   |
|-----:|----------------:|---------------:|-----------:|
| 0.05 |          0.5802 |         0.5217 |   −0.0585  |
| 0.50 |          0.5724 |         0.5129 |   −0.0595  |
| 2.00 |          0.5680 |         0.5082 |   −0.0598  |

DIBL (slope of Vth vs Vds, extracted between Vds=0.05 and 2.0 V):
- ngspice: **6.25 mV/V**
- pyport:  **6.91 mV/V**

Subthreshold slope at Vds=0.5 V (from `z91k_subthreshold_slope.py`, prior run):
- ngspice: **72 mV/dec**
- pyport:  **76 mV/dec**

So: slope matches within 5%, DIBL matches within ~10% (and is small in absolute terms ≪ 60 mV anyway), but **Vth is uniformly low by 58–60 mV at every Vds**. This is a **constant additive Vth bug**, not a slope or DIBL bug.

## Where it could come from in the BSIM4 v4.8.3 Vth aggregator

For reference, the BSIM4 v4.8.3 manual Vth aggregator (eqn 2.16, Vth(Vbs,Vds)) is roughly:

```
Vth = vfb_eff
    + Φs                                                  # bulk band-bending
    + K1ox · √(Φs − Vbsh) − K2ox · Vbsh                    # body effect
    − ΔVth_SCE(Vbs)                                        # short-channel via DVT0/DVT1/DVT2
    − ΔVth_NWE(Vbs)                                        # narrow-width via DVT0W/DVT1W/DVT2W
    + (K3 + K3b·Vbs) · (toxe / (Weff' + W0)) · Φs          # narrow-width offset (depends on Φs and Weff)
    − ΔVth_DIBL(Vds, Vbs)                                  # DIBL  (eta0/etab/dsub)
    − ΔVth_DITS(Vds)                                       # drain-induced threshold shift
    + size-dependent vth0 corrections                      # Lvth0/Wvth0/Pvth0  +  LDE pulldown  +  WPE
```

The size-dependent corrections in `compute_size_dep` (in `temp.py`) include:
- L/W/cross binning of vth0
- LDE pulldown via `lpe0`, `lpeb`, `nlx` (long-channel pocket-implant pull-DOWN of vth0)
- WPE narrow-width offset via `wpemod`/`weffeot`

Several of these terms are large enough to give a ~60 mV constant additive shift.

## Specific questions (max 4 — please be precise)

**(a) Prime suspect.** Of these classic port-implementation pitfalls, which most likely yields a CONSTANT (Vds-independent), L=1.8 µm and W=0.36 µm-wide, ~60 mV LOW shift?
  1. **Φs computed wrong** — e.g. using `2·Vt·ln(NDEP/ni)` instead of `2·Vt·ln(NDEP/ni) + PHIN`, or omitting the small extra +0.4·Vt that some BSIM4 derivations carry.
  2. **`vfb` flat-band term** — card is silent on vfb, so it must be computed from `vfb = vth0 − Φs − K1·√Φs`. If the port instead uses `vfb = -1.0` default or a different sign convention this gives a fixed offset.
  3. **`K1ox = K1 · (toxe/toxm)` and `K2ox = K2 · (toxe/toxm)` scaling** — toxm not in card → defaults to toxe → K1ox=K1; if the port instead applies a different scaling we get a shift.
  4. **`K1·√(Φs − Vbsh) − K2·Vbsh` "small-Vbs" smoothing**: at Vbs=0 this should equal `K1·√Φs`. If port is using `K1·√Φs − K2·Φs` (i.e. evaluating at Vbsh=Φs by mistake) we'd lose `K2·Φs ≈ −0.03·0.85 ≈ −25 mV`. Not 60.
  5. **`(K3 + K3b·Vbs) · (toxe/(Weff' + W0)) · Φs` narrow-width term** — k3=65.28, toxe=4.1n, W=0.36µ, W0 default 2.5e-6 → roughly `65.28 · (4.1e-9/(0.36e-6 + 2.5e-6)) · 0.85 ≈ 79 mV`. **THIS TERM IS PLAUSIBLY THE 60 mV.** If our port is computing it but at wrong Weff (e.g. drawn vs effective) or with W0=0, the numerical value moves a lot. Could you confirm the formula, the value of W0 in v4.8.3, and the Weff' definition (does it use Weff or Weff'_CV)?
  6. **DVT0w narrow-width SCE**: dvt0w=0 in card, so this branch should be off entirely.
  7. **LDE long-channel pulldown via `lpe0` (and `nlx`)**: `lpe0=1.244e-7` is set. The BSIM4 manual eqn for vth0_eff long-channel correction:
     `vth0_eff = vth0 + K1·(√(Φs·(1 + LPEB/Leff)) − √Φs) − K1·(NLX/Leff)·√Φs + ...`.
     For L=1.8 µm and lpe0=1.24e-7, lpe0/L ~ 7%, nlx/L ~11%. Net `lpe0` effect on long-channel pocket implant: this can swing 50–100 mV. Is our port applying lpe0 with the right sign? lpe0 in BSIM4 enters as a PULL-DOWN of vth at long channel via the K1 sqrt expansion — getting the sign or the leffeot inverted gives ~60 mV LOW.
  8. **`Lvth0`/`Wvth0` size-dep corrections being applied with the wrong sign**: card has none, so this should be inert.

  Which is your top one for a Vds-independent ~60 mV LOW shift, and what specifically would you check first in `dc.py`/`temp.py`?

**(b) Where in our code does LDE enter?** Please grep `lpe0`, `nlx`, `lpeb` in `temp.py` and `dc.py` and answer: **does the port actually fold lpe0/nlx into vth0 at this geometry**, with the right sign? If you see a missing or sign-flipped term, point at the exact line. (We are particularly suspicious that LDE was loaded into the param dict but never applied to vth0_eff — which would consistently give a Vds-INDEPENDENT shift of order 50–80 mV at L≈1–2 µm.)

**(c) ngspice-side hidden default?** Could ngspice-42 level=14 silently apply a process bias that we are skipping — e.g. via `vfbcv`, `phin`, `vbm`, or via an internal "vth0 is a *stored* value, not a *computed* value" rule (PSP-style)? In the BSIM4 v4.8.3 reference, `vth0` is the long-channel zero-Vbs threshold for a reference geometry; ngspice may apply a different binning or geometry-resolution rule that bumps the effective vth0 by ~60 mV at this exact (L,W). If so, what is the rule and how do we replicate it?

**(d) Single localizing experiment.** What is the one instrumentation patch we should make? Concretely: print every additive Vth term (vfb, Φs, K1ox·√(Φs-Vbsh), −K2ox·Vbsh, ΔVth_SCE, ΔVth_NWE, narrow-width K3 term, ΔVth_DIBL, ΔVth_DITS, LDE/WPE size-dep corrections) at a single bias `(Vgs=Vth_target=0.52 V, Vds=0.5 V, Vbs=0)`, and diff against a hand-computed reference (we will compute by hand from the v4.8.3 manual). What term boundaries would be most diagnostic to print, and is there a published "BSIM4 Vth term-by-term" reference plot from BSIM-CMG or BSIM4 docs we should diff against?

## Constraints / what we want from you

- Prefer **specific line/file targets** in the attached `dc.py` / `temp.py` / `model_card.py` over general advice.
- A constant additive 60 mV at every Vds rules out: DIBL, DITS, IIMOD, GIDL/GISL, PSCBE, self-heating. Please don't suggest those.
- Name your top hypothesis and the single A/B test we'd run to confirm or refute it. We will run it.
- Keep response ≤ ~1500 words; tables / bullet lists fine.
