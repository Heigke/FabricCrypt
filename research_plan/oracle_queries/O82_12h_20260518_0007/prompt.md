# Oracle 12h NS-RAM gap-closing review (O82)

Context: last 12h of NS-RAM campaign log in context.md.

Key landmarks (2026-05-17 → 2026-05-18):
- z468/z469 found + fixed I_snap_d bug (4 dec lift on Id_pk)
- z471 calibrated snap_Is=4.5192e-12 → Id_pk 4.30 mA on all 4 biases (Mario gap -0.055 dec)
- z472 z461 6/9 PASS, z473 R_body=1e7 → V6 self-reset PASS (7/9)
- z474b applied IFT sign-bug upstream (gradcheck 7/8 PASS, value bit-identity 0 drift)
- z475/z476 honest KILL_SHOTs: V7 oscillation unreachable without topology change (slow charge trap / RC recovery)
- GPU-MAX-A AMBITIOUS PASS: MNIST 82.12% via diff pyport vs vanilla 84.76% (-2.64pp gate ≤3pp)
- EP-NSRAM smoke: 3/4 gates PASS (44% acc, 0% NaN, gradient flows)
- NES-GD: K2 cross-cell corr NOT triggered (mean 0.10, max 0.54, only 1.62% pairs >0.3)
- HNRT honest neg (val NRMSE 1.10 vs ESN 0.65)
- N-BENCH-A + O80 + N-BENCH-B triple-converge: HDC demoted, brief reframe required
- Master of Noise paper outline locked: 6 modes / 1 cell unified substrate pitch

3 QUESTIONS:

Q1 — GATE-CROSSING RISK: Of today's 5 AMBITIOUS or AMBITIOUS-adjacent claims (GPU-MAX-A MNIST 82.12%, EP-NSRAM smoke 44%, z471 Mario ±0.055 dec, z472 V1 RMSE 1.31/1.20/1.84, z473 V6 PASS), which would FAIL a strict reviewer audit and why? Be specific about which numbers are infra-only / clamp-bound / single-seed / no-baseline.

Q2 — CHERRY-PICK AUDIT: We have run 8 oracle queries (O77-O81 + N-BENCH-A) plus 30+ z45x/z46x/z47x experiments today. Are we systematically reporting WINS and burying LOSSES? Cite specific 01_LOG.md lines or honest_analysis.md files where the honesty-test passes or fails.

Q3 — NEXT HIGHEST-VALUE EXPERIMENT: Given current state — Master of Noise outline + EP-NSRAM smoke PASS + diff pyport stable + NES-GD needs whitening fix + V7 osc requires topology change — what single experiment should be dispatched NEXT (over the next 24h) to maximize the chance of getting brief v4.5 / Master of Noise paper publishable? Compare against the alternative of "stop and write".

Return ≤ 400 words per question. Reply rigorously. NO-CHEAT — flag any pushback you have on this very framing.
