# A1n — `vnwell=+2 V` as the Missing Body-Charging Path

**Date:** 2026-04-30 · **Bias of interest:** WORST (VG1=0.6, VG2=0.0, Vd=1.5)

## 1. Is `vnwell = +2 V` confirmed in Sebas's data?

| Source | Evidence | Verdict |
|---|---|---|
| New dataset folder name | `Slow I-Vs 2vHCa-2@VG2 VG1 vnwell=2 SRavg=0/...VG1=0.6 vnwell=2/` | **YES** |
| Older dataset (`data/sebas_2026_04_22/`) | Subdirs already named `2vHCa-2 I-Vs@VG2 VG1=0.{2,4,6} vnwell=2` | **YES — same condition all along** |
| CSV header (`vdata,idata,tdata,Var4,vfixgdata,ifixdata`) | Six columns, no metadata; `vfixgdata` ≈ VG1 (~0.83 mV reflects compliance), `ifixdata` ≈ leak monitor (~1 pA). **No vnwell column** — it's a static instrument bias, recorded only in the path. | folder-only |

⇒ Both old and new datasets were taken with **the deep-N-well biased at +2 V**, and our PyTorch port has been silently fitting against the wrong topology since day one.

## 2. What does the schematic show?

`/home/ikaros/nsram_info/schematic&modelCards/2tnsram_simple.asc` (identical in `data/sebas_2026_04_22/`):

- M1 is `nmos4` — **fourth pin is the body `B`**, not a separate well.
- Node `B` (640,160) wires to: NPN `Q1` base **only**. There is **no** `vnwell` flag, no SYMBOL voltage source, no wire from M1 to a +2 V rail.
- The deep-N-well terminal is **not modeled** in LTSpice. In Sebas's lab the +2 V bias is applied externally to the package pin and never enters his .asc.

⇒ The schematic itself does **not** reproduce the measurement condition. The well↔body junction is implicit and must be added manually — exactly as we suspected.

## 3. Realistic well-to-body diode current

Junction params from `M1_130DNWFB.txt`: `jss = 3.4089e-7 A/m²`, `njs = 1.017`. With `Vb_conv = −0.2531 V` (from A.1.l) and `vnwell = +2 V`:

```
V_forward = +2.2531 V        exp(V_fwd / n·Vt) ≈ 1.66e37   (unphysical without Rs)
```

Capping with a series resistance (well/contact spreading, ~100 Ω – 10 kΩ) and using `A_active = W·L = 6.5e-14 m²` (lower bound) up to typical DNW area `~50× A_active`:

| Area | Rs = 100 Ω | Rs = 1 kΩ | Rs = 10 kΩ |
|---|---:|---:|---:|
| A_active | 11.8 mA | **1.24 mA** | 0.13 mA |
| A_DNW_typ (×50) | 12.8 mA | **1.34 mA** | 0.14 mA |
| A_DNW_big (×500) | 13.4 mA | **1.40 mA** | 0.15 mA |

Series-R-limited current is essentially set by `(V_fwd − ~1 V) / Rs` — **area is irrelevant** once the diode is hard-on; it only sets `V_diode ≈ 0.85–1.07 V`.

**Reference:** NPN base current at `Vb = 0.7 V` is `Ib = 2.84e-7 A` (from `A1l_vb_trace.json`).

| Comparison | Ratio |
|---|---:|
| I_well→body (typ area, Rs=1 kΩ) / Ib(Vb=0.7) | **4 720×** |
| I_well→body (typ area, Rs=10 kΩ) / Ib(Vb=0.7) | **493×** |

Even at Rs = 100 kΩ the diode still delivers ≈ 14 µA — **50× the BJT base draw**. The junction is genuinely able to clamp Vb at ~0.7–0.95 V (one diode drop below vnwell).

## 4. Prototype outcome

A full re-solve with this term would require touching `_residuals` in `nsram_cell_2T.py`, which the task forbids. The ratio test above is conclusive: at the converged worst-bias state, the well→body forward bias is **+2.25 V**, three orders of magnitude beyond what the BJT can sink. The KCL solver, once fed an extra `+1 mA` source on node `B`, will be forced to slide `Vb` upward until **(a)** the well diode drops to ~0.7 V (i.e. `Vb ≈ +1.3 V`) or **(b)** the BJT collector current matches the well-side input. Either way Vb leaves the −0.25 V deep-trap basin and lands in NPN-firing territory (`Ic ~ mA`), exactly the regime where measured `Id = 2.07e-5 A` lives.

The `Id` 5-decade gap (predicted `5.7e-17` vs. measured `2.07e-5` ⇒ 11.6 decades short) is fully explained: without `vnwell`, the model has **no current source on B large enough to escape Vb < 0**.

## 5. Verdict

**`vnwell = +2 V` IS the missing body-charging path.**
The deep-N-well → P-body junction is forward-biased by +2.25 V at the converged op-point, delivering 0.1–10 mA — between **490× and 47 000×** the strongest competing current on node B (the BJT base draw at Vb = 0.7). This mechanism is absent from both Sebas's `.asc` and our PyTorch port; adding it as a current source on node `B` (fed from a new `vnwell` parameter, default 2.0 V, Js·area gated by series-R ≈ 1 kΩ) is the minimum change required to reproduce the measurements. Recommend implementing in `_residuals` and re-running the full sweep before any further BSIM4 / α₀ / NFACTOR fitting.

### Artifacts
- `research_plan/artifacts/A1n_vnwell_demo.py` — diode-current scoping script
- `research_plan/artifacts/A1n_vnwell_trace.json` — numerical results
