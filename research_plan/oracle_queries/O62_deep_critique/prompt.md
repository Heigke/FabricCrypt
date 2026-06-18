Five independent diagnostic tracks ALL agree that pyport's _solve_at_fixed_vb pins Vb=Vd which makes the physical basin (ngspice gives Vsint=0.38, Vb=0.27) unreachable. We have a fix in flight (Vb-free 2D Newton).

BUT user says "something is very off, dig deeper". Question for you (3 oracles, INDEPENDENT):
Q1: Is the basin-lock truly the ROOT cause, or just a SYMPTOM of something deeper (wrong residual formulation, missing node, wrong topology)?
Q2: Looking at the topology compare doc — is there a structural element we're STILL missing (beyond D1/D2/D9)?
Q3: Why does pyport DC fit converge to 0.99 dec for some biases but 3+ dec for others? Bimodal behavior — what does that signal?
Q4: Predict: if R-13 Vb-free solver succeeds, what cell-wide median dec do you expect (give number + ±range)?
Q5: ONE specific experiment we could run TONIGHT (with ngspice and pyport both available) that would discriminate "basin-lock" vs "deeper structural" hypotheses?

Attached context:
- _log_tail.md — last 200 lines of research log
- R_deep_A_topology_compare.md — topology comparison doc
- R_deep_B_oracle_structural.md — prior oracle structural review synthesis
- z329_summary.json — III/Vsint map results
- z330_summary.json — ngspice cross-check results
- z331_summary.json — snapback graph numerics
- cell.asc — LTSpice 2T NS-RAM cell card (canonical schematic)
- nsram_cell_2T_head.py — first 100 lines of pyport cell model

≤600 words per oracle. Be direct. Number your answers Q1-Q5.
