# A1b — BJT Parameter Mapping (Sebas CSV ↔ `GummelPoonNPN`)

## 1. Sources

- **SPICE model card** (`parasiticBJT.txt`):
  ```
  .model parasiticBJT NPN(is=5E-9 va=100 bf=10000 br=100 nc=2 ikr=100m
                          rc=0.1 vje=0.7 re=0.1 cjc=1e-15 fc=0.5
                          cje=0.7e-15 ne=1.5 ise=0
                          tr=20e-12 tf=25e-12 itf=0.03 vtf=7 xtf=2)
  ```
- **Schematic** (`2tnsram_simple.asc`):
  `SYMBOL npn ... Q1 ... Value=parasiticBJT  Value2=area=1u`
  → instance `area = 1e-6`. No `m=` multiplier on the LTSpice Q1 instance.
- **CSV** columns: `mbjt, IS, area`. Rows show `IS=5e-9`, `area=1e-6`
  constant; `mbjt` flips between **0.001 (VG1=0.2)** and **1.0
  (VG1=0.4 / 0.6)**.

## 2. Mapping Table

| SPICE NPN keyword          | LTSpice instance | Our `GummelPoonNPN` | Honoured by `from_sebas_card`? |
|---|---|---|---|
| `IS`  (saturation current) | —                | `Is`  (5e-9)        | yes (hard-coded)               |
| `VA`  (Early fwd)          | —                | `Va`  (100)         | yes                            |
| `BF`                       | —                | `Bf`  (10000)       | yes                            |
| `BR`                       | —                | `Br`  (100)         | yes                            |
| `NF`/`NR` (default 1)      | —                | `Nf=Nr=1`           | yes                            |
| `NC`, `NE`                 | —                | `Nc=2`, `Ne=1.5`    | yes                            |
| `IKR`, `ISE`               | —                | `Ikr=0.1`, `Ise=0`  | yes                            |
| —                          | `area=1u`        | `area = 1e-6`       | yes (multiplies Is/Ikf/Ikr/Ise/Isc in `compute_bjt`) |
| —                          | `m=<mbjt>`       | **no field**        | **NO**                         |

`mbjt` is **the SPICE device multiplier `m`** (cell count / parallel
parasitic-NPN scaling). In SPICE, `m` multiplies `IS, IKF, IKR, ISE, ISC,
1/RB, 1/RE, 1/RC` exactly as `area` does — i.e. it is mathematically
identical to scaling `area`. There is no Gummel-Poon "ideality" parameter
called `mbjt`; this is purely a count multiplier added by Sebas's
extraction wrapper to switch the BJT path on/off per VG1.

## 3. Pipeline Audit (`z91f.make_bjt`)

```python
def make_bjt(sebas_row):
    bjt = GummelPoonNPN.from_sebas_card()
    if sebas_row is not None:
        if not math.isnan(sebas_row.get("IS", float("nan"))):
            bjt.Is = float(sebas_row["IS"])
        # mbjt is the BJT idealisation factor (we treat as Bf scaler).
        # For validation: ignore ...
    return bjt
```

- `IS` is read **once per row** but the value is constant 5e-9 — so the
  per-row override is a no-op. Per-bias `IS` is *technically* honoured;
  effectively unused.
- `area` is **not** read from the CSV (still defaults to 1e-6 from the card).
- `mbjt` is **explicitly ignored** with a wrong comment ("Bf scaler").
- z91g imports `make_bjt` from z91f → identical bug.

## 4. Numeric Check (Vbe = 0.6 V, Vbc = 0, T = 300 K)

| Effective multiplier | Ic       |
|---|---|
| current code (`area=1e-6`, mbjt ignored)         | 5.94e-5 A |
| with `area *= mbjt = 1.0` (VG1 ≥ 0.4 rows)       | 5.94e-5 A |
| with `area *= mbjt = 0.001` (VG1 = 0.2 rows)     | 5.94e-8 A |

→ **Exactly the ~3-decade gap** seen in the z91g residuals at low VG1.

## 5. Verdict

**`mbjt` is a SPICE `m=` device multiplier, not a Gummel-Poon ideality.
It is currently NOT honoured; `IS` is honoured but trivially constant.
At VG1 = 0.2 rows the simulator over-drives the parasitic NPN by 1000×,
which is exactly z91g's low-VG2 residual signature.**

**Fix (one line in `make_bjt`):**
```python
if not math.isnan(sebas_row.get("mbjt", float("nan"))):
    bjt.area = bjt.area * float(sebas_row["mbjt"])
if not math.isnan(sebas_row.get("area", float("nan"))):
    bjt.area = float(sebas_row["area"]) * float(sebas_row.get("mbjt", 1.0))
```
i.e. set `bjt.area = csv.area * csv.mbjt`. Existing `compute_bjt` already
applies `area` as the SPICE-correct multiplier on `Is, Ikf, Ikr, Ise, Isc`,
so no change to `bjt.py` is required.
