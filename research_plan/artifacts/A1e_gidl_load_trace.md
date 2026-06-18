# A1.e — GIDL/GISL load trace for M2 at (VG1=0.6, VG2=0.0)

## Verdict (one sentence)
At the failing bias all four GIDL/GISL gates are closed by the
`Vd-Vg-egidl > 0` band-bending condition, so the observed `GIDL/GISL = 0`
is **correct physics — not a parser bug**; however, the load trace also
exposed a **latent bug** in the `("ref", "agidl")` default mechanism that
will silently zero GISL at other biases.

## Hypothesis test results

| H | Statement | Result |
|---|-----------|--------|
| H1 | Parser drops `+`-continued GIDL values | **REJECTED** — agidl/bgidl/cgidl/egidl all loaded correctly (`given=True`) |
| H2 | `compute_igidl_gisl` early-returns on `gidlmod` | REJECTED — `gidlmod=0` selects the implemented branch |
| H3 | Values land in `sd.scaled` but formula reads elsewhere | REJECTED — formula reads `model.get(...)` directly, same source |
| H2′ | (Newly identified) **agisl-group siblings stay at pre-override defaults** because `("ref", ...)` is resolved in pass 2 *before* user overrides apply in pass 3 | **CONFIRMED** |

## Loaded vs card vs default

| param   | loaded     | card        | BSIM4 default | given |
|---------|-----------:|------------:|--------------:|:-----:|
| agidl   | 1.99e-8    | 1.99e-8     | 0.0           | True  |
| bgidl   | 1.624e9    | 1.624e9     | 2.3e9         | True  |
| cgidl   | 6.3        | 6.3         | 0.5           | True  |
| egidl   | 0.91       | 0.91        | 0.8           | True  |
| agisl   | **0.0**    | not in card | (ref agidl)   | False |
| bgisl   | **2.3e9**  | not in card | (ref bgidl)   | False |
| cgisl   | **0.5**    | not in card | (ref cgidl)   | False |
| egisl   | **0.8**    | not in card | (ref egidl)   | False |

`agisl` was supposed to mirror `agidl=1.99e-8` per BSIM4 spec, but is **0.0**.

## Root cause (`model_card.py` lines 75–90)
```
Pass 1: scalar defaults  → agidl=0.0, bgidl=2.3e9, ...
Pass 2: ref defaults     → agisl=agidl=0.0, bgisl=bgidl=2.3e9, ...   ← SNAPSHOT
Pass 3: user overrides   → agidl=1.99e-8, bgidl=1.624e9, ...         ← agisl NOT updated
```
A card that specifies only the GIDL group leaves the GISL group pinned to
the *pre-card* defaults (agisl=0, etc.), which is contrary to BSIM4 v4.8.3
behavior (`b4ld.c` initialises GISL from GIDL after the parameter file is read).

## Bias gate check at (VG1=0.6, VG2=0.0, Vd=1.5, Vsint=0.306, Vb=0.342)

| device/edge | V_drive = Vd–Vg–e | result |
|---|---:|---|
| M2 GIDL (drain=0.306, g=0)     | -0.604 | CLOSED |
| M2 GISL (source=0, g=0)        | -0.800 | CLOSED |
| M1 GIDL (drain=1.5, g=0.6)     | -0.010 | CLOSED (just barely!) |
| M1 GISL (source=0.306, g=0.6)  | -1.094 | CLOSED |

So at THIS bias `GIDL/GISL ≡ 0` is honest physics: drain–gate band-bending
is too weak for BTBT. The observation in A.1.c is consistent with the
formula. The body-charging residual must come from impact ionisation
(`Iii`) or sub-threshold leakage, not GIDL/GISL.

The **agisl=0 bug is silent here** because the GISL gate is closed
anyway, but it will hide tunneling current at higher Vd or more positive
Vbs operating points (e.g. M1 GIDL was within 10 mV of opening — a small
bias shift turns it on, and once on, agisl=0 zeroes a current that
should be ≈ agidl-scale).

## Proposed fix
In `nsram/bsim4_port/model_card.py` `__init__`, **re-resolve `ref` defaults
after pass 3** (or only for parameters that are still `not is_given` and
whose referenced source was overridden):

```python
# Pass 4: re-resolve ref defaults whose source was user-overridden
for name, info in PARAMS_META.items():
    d = info["default"]
    if isinstance(d, tuple) and d[0] == "ref" and name not in self._given:
        self._values[name] = self._values.get(d[1], 0.0)
```

This restores the canonical BSIM4 behavior where, e.g., a card specifying
only `agidl` automatically yields `agisl = agidl`. Touches one file, no
formula changes; existing cards that explicitly set agisl are unaffected
(`name not in self._given` skips them).

## Artifacts
- Demo script: `research_plan/artifacts/A1e_demo.py`
- Card: `data/sebas_2026_04_22/M2_130bulkNSRAM.txt` (lines 80–81)
- Source: `nsram/nsram/bsim4_port/model_card.py` lines 75–90, `_model_card_data.py` lines 14–103
