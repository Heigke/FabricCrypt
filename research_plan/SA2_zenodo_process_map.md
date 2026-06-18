# SA2 — Zenodo SPICE Process-Node Map

**Source**: `data/nsram_zenodo/SimulationFiles/SPICE/` (Pazos et al., "Synaptic and neural behaviours in a standard silicon transistor")

**Target comparison**: Sebas's measured device = **130 nm thick-oxide imec** floating-bulk MOSFET.

---

## Per-file mapping

| Filename | Process indicator | Confidence | Role | Matches Sebas cell? |
|---|---|---|---|---|
| `PTM130bulk_lite.txt` | PTM 130nm bulk; `Tox=3.3e-9` (3.3 nm = **thin ox**); `Vth0=0.432/-0.3499`; `Lint=2.5e-8`; `lmin=lmax=1.3e-7` (header: "http://ptm.asu.edu"). README states "**not unique to any process, only exemplary**". | **High** (PTM generic 130nm thin-ox) | BSIM4 NMOS/PMOS card (avalanche MOSFET M2/M4 in schematic) | **NO** — same node (130 nm) but **thin-ox**, not imec thick-ox. Generic PTM, not a foundry/imec PDK. |
| `BJTparams.txt` | Header comments explicitly list two alternative parameter sets: `***Tsinghua` (BVPar=`3.5-1.5*Vg`, nbvPar=`9-(0.1/Vg)*5.5`) and `***TSMC` (BVPar=`1.6+0.4/Vg`, nbvPar=`9-(0.4/Vg)*0.5`). Default uncommented = **Tsinghua** BVPar formula; nbvPar=0. | **High** | NPN parasitic BJT parameter declarations (Is/Bf/Vaf/BVcbo/nBVcbo) | **NO** — both options are external foundry processes (Tsinghua university line, TSMC). Neither is imec thick-ox. Active default = Tsinghua. |
| `BJTavalanche.txt` | Pure model card; no node indicator. Pulls all process-specific values from `BJTparams.txt` via `'BVpar'`/`'BfPar'` etc. | **High** (process-agnostic wrapper) | `.model avalBJT NPN(...)` for parasitic BJT | **Inherits Tsinghua default** via BJTparams. Not Sebas. |
| `Davalanche.txt` | Pure Zener diode model; bv=`0.9*BVPar`. No node indicator. | **High** (process-agnostic wrapper) | `.model avalancheD D(...)` zener for B–C breakdown | Inherits Tsinghua via BVPar. Not Sebas. |
| `AvalancheCircuit2_BulkMOSFET.asc` | Top-level testbench. Includes `PTM130bulk_lite.txt`, `Davalanche.txt`, `BJTavalanche.txt`, `BJTparams.txt`. Uses `nmos4 M2` with `l=Ln=0.25u, w=Wn=10u`, m=10 (note Ln=250nm, longer than minimum). Sweeps Vg=0.25/0.45. | **High** | Full neuron testbench (BJT + zener + nMOS + bulk-control transistor M1=BSS145) | **NO** — generic PTM130 + Tsinghua/TSMC BJT params. **L=250 nm, not 130 nm physical**. Thin-ox PTM card. Not Sebas's thick-ox imec. |
| `Davalanche_debug.asc` | Standalone zener tuning bench. Only includes `Davalanche.txt`; sets `BVPar=2`, area=0.05. | **High** (zener-only debug) | Calibration testbench for the avalanche diode model alone | **NO** — process-agnostic debug rig. |
| `subcircuit/NeuronSubCirc.asc` | Subcircuit schematic; instantiates BSS145 (M3), nmos4 M4 with `l=Ln, w=Wn`, avalBJT Q2, avalancheD D3/D4. No direct lib include (parent provides). | **Medium** (depends on parent .asc) | Reusable floating-bulk neuron subcircuit for hierarchical netlists | **NO** — inherits whatever parent provides; the only parent shipped is `SubC_SimpleTest.asc`. |
| `subcircuit/NeuronSubCirc.asy` | LTSpice symbol file (graphics). | n/a | Symbol only | n/a |
| `subcircuit/SubC_SimpleTest.asc` | Calls `../PTM130bulk_lite.txt`, `../Davalanche.txt`, `../BJTavalanche.txt`, `../BJTparams.txt`. Same Ln=0.25u, Wn=10u. | **High** | Minimal hierarchical-instance testbench for the subcircuit | **NO** — same generic PTM130 + Tsinghua-default stack. |

Total files inspected: **9** (7 SPICE-relevant + 1 .asy symbol + README).

---

## Summary

**Mario's Zenodo deck contains 0 files for the Sebas 130 nm thick-ox imec cell and 7 files for other / generic processes** (1 BSIM4 card = generic PTM 130 nm **thin-ox** ≈3.3 nm; 2 BJT/zener wrappers parametric on Tsinghua-or-TSMC formulas; 4 testbench/subcircuit .asc files; 1 zener debug bench).

The README itself states explicitly (line 38):
> "The transistor model is extracted from Predictive Technology Models for a 130 nm node and reduced for compatibility with LTSpice. **These are not unique to any process and only exemplary models** for the general behaviour of a device in such technology."

And `BJTparams.txt` exposes the BVpar/nbvPar formula choice between **Tsinghua** (default uncommented) and **TSMC**. Neither corresponds to the imec thick-ox cell Sebas measured.

### Final verdict
**No Zenodo SPICE file is a calibrated model of the same 130 nm thick-ox imec device Sebas measured.** The deck is a generic demonstrator (PTM thin-ox MOSFET + parametric NPN+zener avalanche shells) whose BJT parameter set defaults to a Tsinghua formula. Any DC-fit comparison ("1.39 dec at Bf=100") against Sebas IV data is therefore a fit of a *generic* topology to a *specific* device, not a calibrated foundry-card replication. The substrate gap is real and must be flagged in M3B/Phase-A reporting.
