# Overnight Plan 2026-05-15 (04:20 → 08:00)

## Context
- 3 topology fixes failed (R-43, R-47, R-49). Snapback fold reproduces at <1/30 of measured magnitude.
- 0.965 dec fit is sub-threshold curve-fit, not BJT-fold physics.
- UCI-HAR baselines crush NS-RAM (sklearn linear ridge 96.2% vs NS-RAM HDC 76%).
- 5 surviving app claims (DS-N10/11/14/15/16), all niche, none at production scale.

## Tonight's 4 tracks (parallel, all machines)

### T1 — Model topology fundamentally different (R-52 → R-55, ikaros + zgx)
**Goal**: reproduce 2-3 dec snapback fold at VG1∈{0.2,0.4,0.6}.

- **R-52** (ikaros): Multiplicative M(V_db) directly on Ids (NOT body injection).
  Replace `Ids_M1 *= M(V_db)` instead of `R_B += (M-1)*|Ids|`. Same M form.
- **R-53** (ikaros): Two-stage. Bulk current Ib_from_avalanche += k1*(M-1)*Ids, AND Ids *= sqrt(M).
- **R-54** (ikaros): Reorient NPN — emitter at Vb (not Vsint), collector at Vd direct. Currently Vsint emitter.
- **R-55** (subagent): Deep Zoom folder dive for missed Mario/Sebas notes about snapback or topology. Look at:
  - /docs/Zoom/2026-04-30 13.03.27 Zoom NSRAM
  - /nsram/Zoom/2026-04-30 13.03.27 Zoom NSRAM
  - /research_plan/artifacts/Zoom
  - 2tnsram_simple.asc inner topology
  
### T2 — Production network sim (daedalus + zgx)
- **z375 DVS-Gesture real**: 128×128 events, 11 gestures, daedalus GPU. NS-RAM vs LSTM vs LIF vs Transformer baseline.
- **z376 PTP massive parallel transient** (#178 pending): 100K cells × 10ms transient on daedalus GPU.

### T3 — Oracle critique (network I/O light)
- **O69**: aggressive 3-way critique on R-49 failure + UCI-HAR finding. "What are we missing?"
- **O70**: critique on production scaling — "is the substrate fundamentally limited?"

### T4 — Continuous monitor (ikaros)
- Hourly :47 ticks already running
- New: every 2h append top-3 best params from z365 history to summary

## Pre-registered gates (per track)

| Track | INFRA | DISCOVERY | AMBITIOUS |
|---|---|---|---|
| R-52 | runs without nan | any BV → cell < 1.0 | any BV → cell < 0.50 + VG1=0.6 fold > 1 dec |
| R-53 | runs | < 1.0 | < 0.50 + fold > 1 dec |
| R-54 | runs | < 1.0 | < 0.50 + fold > 1 dec |
| R-55 | finds ≥1 missed clue | finds explicit snapback topology | finds tape-out card |
| DVS-G | trains 1 epoch | NS-RAM ≥ digital baseline | NS-RAM beats LSTM by 1pp |
| PTP | 100K cells run | < 30 min wall | < 5 min wall |
| O69 | 3 responses | ≥2 cite same missing physics | one agrees with falsification |

## NO-CHEAT
- Honest log even if all fail
- Bootstrap CI on any "wins"
- No retroactive gate edits
