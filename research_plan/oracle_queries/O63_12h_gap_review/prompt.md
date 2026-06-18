# O63 — 12h gap-closing review

## Context (last 12h of 01_LOG.md attached as log_tail.txt)

Goal: NS-RAM 2T pyport model cell-wide DC fit vs Sebas 33 IV. Baseline z304 = 0.99 dec.
Active arc R-13..R-23 chases basin-lock + BJT topology + cfg-diff.

## Status snapshot (verified findings)

- R-15 (BSIM4 term audit): Ic_Q1 6 dec low, Ids_M1 3 dec low, Igidl 4 dec HIGH at ngspice OP
- R-16 (GPU basin scan): pyport residual landscape HAS the (0.38, 0.27) basin (||R||²≈1e-29 at minima) — residual function EXONERATED
- R-17 (oracle 9-way): UNANIMOUS basin-lock = SYMPTOM; root = D1 + missing M2.B handling
- R-19 (residual probe + M2.B audit): ||R||_inf = 3.27e-7 (ambiguous zone); M2.B already correctly routed (m2_body_gnd=True default) — R-17's M2.B flag was FALSE-POSITIVE
- R-20 found SMOKING GUN: pyport BJT Vbc = Vb − Vd (collector wired to Drain), but ngspice deck `Q1 vsint vb 0` has collector=Sint → patched L530 to Vbc = Vb − Vsint
- R-21 (z337 cell-wide refit with fix): median 4.16 dec (WORSE than z304's 0.99 — fix exposed overshoot)
- R-22 (z338 BBO over alpha0/Bf/Va/Is/lat_BV/body_pdiode_Rs): timeout at eval 76, floor 3.43 dec, top 5 all cluster 3.42-3.55
- R-23 (cfg-diff vs z304): in flight

## 3 Questions

**Q1 — gate-crossing risk**: z304's 0.99 dec is the canonical "good" baseline. But pre-R-20 it relied on BROKEN BJT (Vbc=Vb-Vd making Q1 reverse-biased). Is z304's 0.99 a TRUE physics fit or a spurious local optimum that compensated for the BJT bug with other knobs? If spurious, the goal should change from "match z304" to "find new sub-1 basin with correct BJT".

**Q2 — cherry-pick risk in R-22 floor**: BBO floor at 3.43 dec from 76 evals across 6 log-scale params. Best params: alpha0=1.6e-5, Bf=2605, Va=0.36, Is=3.3e-10, lat_BV=4.02, body_pdiode_Rs=5.5e6. Top-5 cluster (3.42-3.55) suggests landscape is plateau, not local min. Is there a STRUCTURAL flag we likely toggled OFF when applying BJT fix that needs to come back? (Candidates: use_well_diode, body_pdiode_to, eta_sigmoid, use_lateral_collector, vnwell_Rs, use_local_base.)

**Q3 — highest-value next experiment**: Given findings, what is the SINGLE most informative experiment to run in next 1h to either (a) cross sub-1 dec gate, or (b) prove we cannot with current pyport architecture and require deeper rewrite? Rank candidates: cfg-diff (R-23 in flight), Vsint-residual sensitivity audit, ngspice handover (use ngspice to solve 33 OPs then pyport for I-V), forward-2t solver init from ngspice basin warm-start, full DOE 2^7 cfg-flag sweep.

≤500 words per oracle. Be falsificationist, not consensus-bridge.
