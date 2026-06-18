# Snapback Self-Validation Plan — 2026-05-15 (no Sebas needed)

## Core insight
Sebas's measured data + BSIM4 model IS ground truth. We don't need new measurements — we need to **see** the (Vsint, Vb) phase space we've been blindly probing with Newton.

## 5 parallel tracks

### S4-A: 2D phase portrait at VG1=0.6 (ikaros, 30 min)
- Sweep (Vsint, Vb) on grid 50×50 over [0, 0.3] × [0, 0.9] V
- At each point compute residuals R_B(Vsint, Vb), R_Sint(Vsint, Vb) at Vd=1.5V
- Find ALL contours where both residuals = 0 (all equilibria/roots)
- Plot contour map → reveals EXACTLY all basins
- VERIFY: how many roots? Where? Is there a high-Vb root that gives Ids matching meas?

### S4-B: Pinned-Vsint family sweep (ikaros, 20 min)
- Pin Vsint ∈ {0, 0.05, 0.10, 0.15, 0.20} V (skip Newton on Vsint)
- For each pinned Vsint, run 1D Newton on Vb at each Vd
- Plot family of Ids(Vd) — one curve per Vsint
- VERIFY: at Vsint=0, does fold appear at high VG1?

### S4-C: ngspice cross-check (daedalus, 30 min)
- Run ngspice DC simulation on Sebas's actual M1+M2+Q1 netlist at VG1=0.6, VG2=0.2
- Sweep Vd, check if ngspice produces fold
- If YES → pyport implementation bug (we have C reference)
- If NO → BSIM4 fundamentally lacks the fold (need new physics)

### S4-D: TLP-style transient ramp (daedalus, 30 min)
- Slow Vd ramp 0→2V over 10μs, then 2→0V
- Cb=8fF, Vsint/Vb both transient
- Look for HYSTERESIS — different curve on up vs down sweep = bistability
- VERIFY: does ramp-up produce snap-jump? Does ramp-down show different path?

### S4-E: Physical Vsint clamp test (ikaros, 15 min, per O72 OpenAI advice)
- Add strong Rs=5-10Ω shunt M2.source→GND (emulates real metal contact)
- Cold-start Newton, check if Vsint stays ~0V
- VERIFY: with Vsint pinned low by Rs, does fold appear at VG1=0.6?

## Pre-registered gates (all 5 share)
- INFRA: complete without nan, < 60 min each
- DISCOVERY: any track reveals VG1=0.6 fold > 0.5 dec with physical params
- AMBITIOUS: cell-wide < 0.5 dec across all 3 VG1
- KILL-SHOT: all 5 tracks confirm fold cannot exist in pyport+BSIM4 → definitive retract

## Resource allocation
- ikaros: S4-A, S4-B, S4-E (CPU intensive)
- daedalus: S4-C, S4-D (ngspice + transient)
- zgx: idle reserve

## Decision tree
- If S4-A shows multiple roots including high-Vb fold root → S4-B or S4-E can pin solver there
- If S4-A shows only one root and it's low-Vb → BSIM4 lacks bistability fundamentally
- If S4-C shows ngspice has fold but pyport doesn't → pyport bug (high priority)
- If S4-D shows hysteresis loop → bistability exists in time domain, DC misses it
- If S4-E with Rs=10Ω restores fold → Vsint pump is artifact of free-Vsint solver

## Honest fallback
If all 5 KILL-SHOT: **definitive** answer that pyport-BSIM4 cannot reproduce snapback. Retract model program with clear physics reason for next iteration.

If even 1 DISCOVERY: pivot to that specific mechanism, refit, validate.
