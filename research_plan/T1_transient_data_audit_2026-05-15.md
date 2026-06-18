# T1 — Transient Data Audit (2026-05-15)

Goal: find ALL time-dimension data Sebas has shared, to enable Cb /
τ_relax fitting in T2.

## Search executed

```
find /home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/data \
     /home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/docs/Zoom \
     /home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/nsram/Zoom \
   -type f \( -name "*.csv" -o -name "*.raw" -o -name "*.dat" -o -name "*.wfm" \)
head -2 of every CSV in data/sebas_* and nsram/Zoom/Slow*
ls data/sebas_2026_05_02/
find docs/Zoom nsram/Zoom -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \)
```

## Findings

### 1. CSV with `tdata` column (FALSE POSITIVE — slow-DC sweep, not true transient)

- `data/sebas_2026_04_22/2vHCa-2 I-Vs@VG2 VG1={0.2,0.4,0.6} vnwell=2/*.csv`
  (and same set duplicated under
  `nsram/Zoom/Slow I-Vs 2vHCa-2@VG2 VG1 vnwell=2 SRavg=0/...`).
- Columns: `vdata, idata, tdata, Var4, vfixgdata, ifixdata`.
- `tdata` is the **acquisition timestamp** of each DC sweep step
  (~290 ms / step, 81 steps → ~24 s total per file). This is a
  **quasi-static parametric-analyzer trace**, NOT a pulsed-response or
  oscilloscope transient.
- Slew rate ≈ 1.5 V / 24 s ≈ 0.06 V/s. Body relaxation time-constant
  (Cb·R_well, order 100 µs – 10 ms) is FAR shorter than this — the
  device is fully equilibrated at every point. Useless for Cb fit.

### 2. `data/sebas_2026_05_02/` (most recent Sebas drop)

- `image-2.png` (564 KB) — composite IV-fit slide (already extracted to
  `three_branch_params_extracted.json`).
- `pdiode.txt` — p-diode SPICE card (DC only, Cj/Cjsw given but no
  measured C-V data).
- `three_branch_params_extracted.json` — branch-fit param table.
- **No transient/pulse data.**

### 3. Zoom screenshots / oscilloscope traces

- `docs/Zoom/Image 2026-{03-20, 04-22, 04-30}*.jpeg` (≈30 files) and
  duplicates under `nsram/Zoom/Image 2026-*.jpeg`.
- Inspection-by-filename: all are **Sebas's IV-plot screenshots and
  schematic captures** from Zoom meetings (LTSpice schematic, fit
  plots, slide views). No oscilloscope-style time-domain traces
  identified.
- `nsram/Zoom/schematic&modelCards/2tnsram_simple.asc` — LTSpice
  schematic (DC structure, not transient simulation deck).
- `nsram/Zoom/2026-04-30 BSIMfitsBA/` — Excel + PPTX of static IV
  fits, NOT transient.

### 4. Other transient candidates

- `data/nsram_zenodo/SimulationFiles/SPICE/dev/*.txt` — these are model
  cards (parasiticBJT, PTM130bulk, etc.), NOT measurement data.
- `data/nsram_zenodo/SimulationFiles/TCAD/FloatBulk_{Tsub,Rsub}/cmlog.txt`
  — TCAD command log only, no measurement.

## Conclusion

**No transient/pulse-response data exists in the repo as of 2026-05-15.**
Every dataset Sebas has shared is either:
- DC parameter-analyzer slow sweep (`tdata` is just acquisition wall
  clock, not a pulse waveform), or
- Static fit-output / model-card / SPICE deck text.

## Action

- **T2 is SKIPPED** — no transient data ⇒ cannot fit Cb or τ_relax.
- **T3 trigger**: HARD-BLOCK transient validation until Sebas provides
  pulsed measurements. Specifically request: oscilloscope captures of
  Vd-pulse → Id(t) at fixed (VG1, VG2) for pulse widths in [10 ns – 10 µs],
  amplitudes in [0.5, 1.8] V. Without this we cannot validate the
  body-capacitor branch of `transient.py`.

## File list (complete inventory)

| Path | Type | Time-range | Useful for transient? |
|---|---|---|---|
| `data/sebas_2026_04_22/2vHCa-2 I-Vs@VG2 VG1={0.2,0.4,0.6}/*.csv` (33 files) | DC slow IV | ~24 s acquisition; 290 ms/step | No (quasi-static) |
| `data/sebas_2026_04_22/2Tcell_BSIM_param_DC.csv` | Fit-param table | n/a | No |
| `data/sebas_2026_05_02/*` | PNG + JSON + diode card | n/a | No |
| `data/nsram_zenodo/SimulationFiles/{TCAD,SPICE}/...` | TCAD cmd logs + SPICE decks | n/a | No |
| `docs/Zoom/*.jpeg` (≈30) | Screenshots of plots | n/a | No |
| `nsram/Zoom/*` (duplicates of above + .asc + .pptx + .xlsx) | Same | n/a | No |
