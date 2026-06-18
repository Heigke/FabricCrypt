# T3 — TCAD Inventory

**Date:** 2026-05-14
**Scope:** Read-only repo scan for Sentaurus/Silvaco/Synopsys TCAD artifacts.
**Result:** **TCAD project FOUND** (Synopsys Sentaurus, nMOS, float-bulk). Inputs + mesh present; **no I-V/operating-point output curves (.plt) committed.**

## Location

`data/nsram_zenodo/SimulationFiles/TCAD/` — two sibling project trees:

- `FloatBulk_Rsub/` — float bulk, source/drain/substrate Resistor option
- `FloatBulk_Tsub/` — float bulk, Thermode (thermal boundary) option, plus `models/models.scf`

Both are Sentaurus Workbench projects (`gtree.dat`, `.project`, `gscens.dat`, `gvars.dat`, `gcomments.dat`).

## File-type tally per tree (each tree ~33 files, ~0.8 MB)

| Class | Files | Notes |
|---|---|---|
| Sentaurus Workbench metadata | `gtree.dat`, `.project`, `gscens.dat`, `gvars.dat`, `.database`, `.organization`, `.status`, `greadme.html`, `cmlog.txt` | Project state, no curves |
| **SDE structure builder** | `nMOS_dvs.cmd` (6971 B), `nMOS_dvs.prf` | Scheme/TCL device geometry: Lg=`@Lgate@`, Tox=`@tox@`, Hsub=40 µm, Xj=0.12 µm, Nsub=1e16 cm⁻³ |
| **Mesh / structure** | `n@node@_msh.tdr` (239 KB), `n@node@_half_msh.tdr` (150 KB), `n@node@_half_bnd.tdr` (9.5 KB), `n@node@_half_msh.cmd/.log` | Binary TDR — full + half-symmetry meshes |
| **Sentaurus Device decks** (`sdevice*_des.cmd`) | 14 decks: `sdevice_des`, `sdevice1..2`, `sdevice11..17`, `sdevice131`, `BV_des`, `BV1_des`, `IdVds_des`, `IdVgs_des`, `IdVgs1_des`, `vps0p2_des`, `vps200_des` | Each ~4.7–5.0 KB. Cover: **IdVds, IdVgs, breakdown (BV)**, **VPS sweeps** (vps0p2=0.2 V, vps200=200 V) → **hot-carrier / high-V regimes** |
| Material/parameter files | `Silicon.par` (3.7 KB), `Silicon_BUP.par` (851 B), `sdevice.par`, `*_des.prf` | Lucent mobility, Auger, SRH, B2B/Avalanche params |
| Visualisation | `svisual_vis.tcl/.prf`, `SVisualTcl.log[.BAK]` | One log is 107 KB (Rsub) — likely contains some extracted curve data as text, worth grep'ing in T4 |
| `Tsub` extra | `models/models.scf` (2.7 KB) | Thermal scenario file |

## Physics in `sdevice_des.cmd`

Parameterised by `@tmodel@` token:
- **DD** (drift-diffusion), **HD** (hydrodynamic w/ `eTemperature`), **Thermo** (lattice-temperature)
- Quantum: optional `eQuantumPotential` (`@QC@==DG`)
- Mobility: `DopingDep` + `eHighFieldsaturation` + `Enormal`
- Recombination: `Auger`, `SRH(DopingDep, TempDep)`, `Band2Band(E2)`, `Avalanche(CarrierTempDrive)`
- Electrodes: source, drain, gate, substrate (substrate is **floating** — the bulk is the storage node we care about)
- Thermodes (Tsub variant): substrate/drain/source with surface resistances → enables self-heating studies

## What is MISSING

- **No `.plt` output files anywhere** (`find … -name "*.plt"` → 0 hits inside TCAD dir).
- No CSV / extracted I-V curves.
- No doping-profile dumps, no E-field cuts, no hot-electron density maps.
- `@node@`/`@Lgate@`/`@tox@`/`@tmodel@`/`@QC@`/`@parameter@`/`@plot@` tokens are **unsubstituted templates** — the trees as committed have **never been swept to completion** in this repo; results were produced elsewhere and not uploaded to Zenodo.

## Bottom Line

We have the **Sentaurus project skeleton** (geometry, mesh, decks, materials) but **zero result curves**. Decks span DD/HD/Thermo with avalanche+B2B and float bulk — i.e. precisely the regimes Mario/Sebas would use to study impact-ionisation-driven bulk-charging in NS-RAM.

## Ask to Mario / Sebas (T3 → T4 follow-up)

Request the **`.plt` outputs + post-processed CSVs** for at least:

1. **IdVgs / IdVds families** at VG2 ∈ {0.2, 0.4, 0.6, 0.8} V (matches Sebas 130 nm wafer biases) — DD model.
2. **BV (breakdown) sweep**: Vds → Vbreakdown with floating bulk, HD model (carrier temperature, avalanche).
3. **VPS sweep** outputs (`vps0p2`, `vps200`) — the high-V branch that exercises hot-electron injection / bulk charging.
4. **2D field/density snapshots** (or `.tdr` solution files) at: subthreshold, on-state, near-BV, and post-BV — to compare against Sebas DC fit (`nsram_m3b_corrections.md`, η ≤ 1 walk-back).
5. **Self-heating run** (`FloatBulk_Tsub`, Thermo model) — substrate ∂T vs Vds @ VG2=0.6 V.
6. The **`@parameter@` and `@plot@` substitution table** (Workbench experiment matrix) — so we know which decks were actually executed.

Until these arrive, all repo-side claims about TCAD-validated bulk charging are **based on input decks only**, not on simulated curves.
