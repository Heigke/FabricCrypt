# openai response (gpt-5) — 160s

## Q1 — Fragility
N_BENCH_comparison_matrix.md "MNIST (hierarchical SNN) | NS-RAM 2T [SIM] | 97.15% | 17.7 pJ | n/p" → This 17.7 pJ/inf “headline” is device-only, SIM-only, and stale. It ignores DAC/ADC/driver/wire energy (your own matrix flags “LOW — BOLD” and “YES — ngspice validation required”), and it predates the z469 I_snap_d omission fix that you admit “changes the math” for all prior N-sims (“all prior N-sims used clamp-bound Id; z471 down-tune may shift surrogate values,” 01_LOG.md 2026-05-17 — N-tick APU=41C). You also logged the need to re-baseline after the bug fix (“Re-baselining may be required,” 2026-05-17 — P-phase tick). Presenting 17.7 pJ alongside real-silicon competitors is overclaim; reviewers will bin it as apples-to-oranges and unvalidated.

## Q2 — Falsification experiment
NAME: ERvMESH_killshot_1024_LUT  
SWEEP: topology ∈ {ER_SPARSE (p=0.02), MESH_4N}; nonlinearity ∈ {NSRAM_LUT from NX_1p8 + R_body=1e7 (z473), Linear (identity)}; seeds ∈ {0,1,2}; N=1024; Mackey-Glass τ=17 and NARMA-10; fixed spectral radius α=0.9; step dt mapped to volatile regime (use R_body=1e7 from z473).  
OBSERVABLE: Median test error across seeds: MG NRMSE and NARMA-10 NMSE for each (topology, nonlinearity).  
PASS THRESHOLD: All must hold:
- ER_SPARSE advantage: median_ER ≤ 0.95 × median_MESH on both tasks.
- Nonlinearity benefit: median_(Linear) ≥ 1.10 × median_(NSRAM_LUT) on both tasks.
- Absolute sanity: MG NRMSE ≤ 0.05 AND NARMA-10 NMSE ≤ 0.15 for ER_SPARSE with NSRAM_LUT.
WHY THIS KILLS THE CLAIM: The “multi-function primitive” hinges on volatile decay (covered by z473 R_body=1e7), reservoir nonlinearity, and an ER_SPARSE topology advantage at honest η. If ER_SPARSE doesn’t beat MESH_4N and the NS-RAM nonlinearity doesn’t help over linear, the claimed primitive and topology win collapse. One script (~<200 LoC delta to existing N-Res-MG harness), <24 h GPU.

## Q3 — NO-CHEAT audit
Suspicious line 1: "2026-05-16 — CRITICAL: v449_B published '1.31 dec' was forward-only cherry-pick. Backward sweep = 2.86 dec, AVG = 2.09. z451 critique #1 vindicated. All prior DC numbers need fwd+bwd reporting." → DRIFT: Explicit cherry-pick admission; violated “no cherry-picks” until called out.
Suspicious line 2: "2026-05-16 — SYNTHESIS DONE ... HONEST BASELINE: 1.19 dec fwd+bwd avg (z432), NOT 0.886. Biggest cherry-pick: z447/z448 '0.886' was 4 biases only — excluded VG1=0.2." → DRIFT: Post-hoc bias exclusion inflated headline; classic threshold/selection abuse.
Suspicious line 3: "2026-05-17 — N-Hier-MNIST ZGX ALL 3 GATES PASS incl AMBITIOUS! test_acc 97.15% (>97% AMB), energy 17.70 pJ/inf (<50pJ)..." → DRIFT: You greenlit an “AMBITIOUS” energy gate with a pre–z469, peripheral-free figure you later concede needs re-baselining and ngspice. That’s mid-stream metric misuse.
Overall verdict: DRIFTING
