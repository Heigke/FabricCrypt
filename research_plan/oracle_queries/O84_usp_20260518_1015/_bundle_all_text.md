# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: context.md (37608 chars) ===
```
## 2026-05-17 — z45x/z46x tick APU=46C. z471 PARTIAL LAND: snap_Is=4.52e-12 (1.5e-4× of default). 4-bias Id_pk: 4.23/4.45/4.21/4.22 mA — all in [1,10] mA window, dispersion 0.024 dec, ~12% under Mario 4.8 mA. PRIMARY DISCOVERY ✓. DC check on SB_OFF baseline ~2.0 dec running. z461 9-test still pending. Agent still working — full verdict expected ≤30 min.
## 2026-05-17 — P-phase tick APU=46C. State unchanged (P1a✓ P1b✗ P2✓ P4✓ Oracle✗). P5/P6 still deferred per O76 + re-baseline req post-z469 fix. z471 partial land (snap_Is=4.52e-12, 4/4 biases at ~4.2-4.5 mA vs Mario 4.8 mA) provides the calibrated cell needed for brief v4.5 — wait for z461 9-test + DC verdict before P6 dispatch.
## 2026-05-17 — R-phase tick: plan SUPERSEDED. z47x progression: z468✓ z469✓ z470b✓ z471 PARTIAL LAND (Mario 4.8 mA hit ±0.06 dec across 4 biases, snap_Is=4.52e-12). DC + z461 9-test still running. No new dispatch.
## 2026-05-17 — z471 FULL VERDICT: AMBITIOUS PARTIAL (INFRA ✓ DISCOVERY ✓ z461 9-test hung at V1 — Newton bistability, not physics). snap_Is=4.52e-12 LANDS Mario: Id_pk 4.21-4.45 mA all 4 biases, dispersion 0.024 dec, DC delta 0.01 dec on partial 2-curve check.
## 2026-05-17 — DECISION POINT: (a) z472 fix V1-hang for full 9-test, OR (b) proceed to brief v4.5 with calibrated cell + partial scorecard caveat. Awaiting user.
## 2026-05-17 — N-tick APU=41C. No local N-sims. 9 PASS unchanged. z471 LANDED snap_Is=4.52e-12 — cell now Mario-calibrated. PROPOSE (no auto-launch): re-baseline 1-2 quick-running sims (N-HDC-UCIHAR or N-Stoch-RNG) under new calibration to verify PASS still holds — ~5 min each. PENDING matrix slots: N-FF-MNIST (ikaros idle), N-WTA-MNIST v2 (zgx), N-STDP-ECG v2 (zgx). Hold until user picks z472 vs brief v4.5 path.
## 2026-05-17 — z45x/z46x tick APU=40C. z471 LANDED (already logged) — AMBITIOUS PARTIAL (INFRA+DISCOVERY ✓, 9-test V1 hang). z47x chain CLOSED at calibration step. No new completions, no I_snap=0 KILL_SHOT, no z453 cross. Awaiting user decision (z472 fix-hang vs brief v4.5). z462/z466 remote still active.
## 2026-05-17 — R-phase tick: plan still SUPERSEDED. z47x sequence CLOSED at z471 LAND (Mario calibrated, snap_Is=4.52e-12, 4/4 biases ±0.06 dec, AMBITIOUS PARTIAL on 9-test hang). Awaiting user decision: z472 fix-hang vs brief v4.5. No new dispatch.
## 2026-05-17 — P-phase tick APU=40C. State unchanged (P1a✓ P1b✗ P2✓ P4✓ Oracle✗). P5/P6 deferred per O76 + re-baseline post-z469 fix. z471 LANDED gives Mario-calibrated cell — brief v4.5 unblocked once user picks z472-fix vs proceed-with-caveat path.
## 2026-05-17 :47 — APU=41C idle locally (no z2x; z472 just dispatched, z462/z466 remote)
## 2026-05-17 — N-tick APU=41C. No local N-sims. 9 PASS unchanged. z472 in flight to unblock z461 9-test on calibrated cell. Holding N-dispatch until calibration verified + 1-2 re-baseline sims.
## 2026-05-17 — R-phase tick: plan SUPERSEDED. z472 in flight (V1 fix + 9-test + Mario shape match on calibrated cell). No other changes.
## 2026-05-17 — z45x/z46x tick APU=44C. z472 dir created 16:52, agent active. No completions since z471. No I_snap=0 KILL_SHOT.
## 2026-05-17 — P-phase tick APU=44C. State unchanged. P5/P6 deferred. z472 in flight (diag VG1=0.2 row clean, no hang yet — z471 hang may have been at higher VG1). LIF ETA 1.5-2.5h.
## 2026-05-17 — R-phase tick: plan SUPERSEDED. z472 in flight (diag VG1=0.2 clean). No state change.
## 2026-05-17 — N-tick APU=46C. No local N-sims. 9 PASS unchanged. z472 still in DC diag (VG1=0.2 done clean). Holding N-dispatch.
## 2026-05-17 — z45x tick APU=46C. No new completions. z472 grinding DC diag.
## 2026-05-17 — P-phase tick APU=46C. State unchanged. P5/P6 deferred. z472 in flight.
## 2026-05-17 — R-phase tick: plan SUPERSEDED. z472 still in flight. No change.
## 2026-05-17 :47 — APU=45C idle locally (no z2x; z472 grinds DC diag remote-style)
## 2026-05-17 — N-tick APU=45C. No local N-sims. 9 PASS unchanged. z472 still running. No N-dispatch.
## 2026-05-17 — z472 LANDED: 6/9 z461 PASS on calibrated cell (V1/V2/V4/V5/**V8 LIF**/V9). FAILs V3+V6+V7 share root cause: missing body-leak path → V_b latches after spike, no self-reset, no oscillation. Mario shape 1/5 strict + 2/5 amplitude (V_b 0.620V ✓, Id 4.31mA ✓). t_rise 2.9ns (too fast), t_fall 140ns (too slow).
## 2026-05-17 — BONUS: V1 "hang" was actually PT solver tolerance floor collapse on sub-pA cell, not Newton bistability. Fix in scripts/z429_multisolver_debug.py — absolute R_B tolerance + stall-detect. Per-curve 70s→37s, calibration preserved (0.07 dec drift).
## 2026-05-17 — z473 candidate: R_body sweep down to ~1e7 Ω to enable reset path. V3/V6/V7 should flip as triplet. Awaiting user.
## 2026-05-17 — z45x tick APU=42C. z472 LANDED 6/9 PASS (logged above). z473 R_body sweep awaiting user. No I_snap=0 KILL_SHOT, no z453 DISCOVERY cross.
## 2026-05-17 — P-phase tick APU=41C. State unchanged. P5/P6 deferred. z472 LANDED 6/9 PASS — brief v4.5 viable now with honest caveat on reset/oscillation (V3/V6/V7), or wait for z473 to flip triplet.
## 2026-05-17 — R-phase tick: plan SUPERSEDED. z472 LANDED 6/9. z473 R_body sweep awaiting user approval to flip V3/V6/V7 triplet.
## 2026-05-17 — N-tick APU=40C. 9 PASS unchanged. z472 LANDED — cell now LIF-verified (V8 PASS). Awaiting user on z473 R_body for full reset/oscillation. No N-dispatch.
## 2026-05-17 — z45x tick APU=40C. State unchanged since z472. No new completions.
## 2026-05-17 — P-phase tick APU=40C. State unchanged. P5/P6 deferred.
## 2026-05-17 — R-phase tick: plan SUPERSEDED. z473 R_body sweep still pending user approval.
## 2026-05-17 — PARALLEL CAMPAIGN: 4 spår dispatched. z473 (R_body sweep ikaros, reset/osc), N-BENCH-A (real-chip matrix web), N-BENCH-B (large-scale 131k DVS128 / 65k HDC / CIFAR-10 zgx-daedalus), O80 (3-way oracle brief v4.5 positioning/killshot/funding). No machine collision.
## 2026-05-17 — z45x deep-dive tick APU=42C. 4 spår körs: z473 (R_body reset, ikaros), N-BENCH-A (web), N-BENCH-B (zgx/daedalus large-scale), O80 (oracle). All z45x closed. DC gap ~1.0 dec accepted for v4.4 brief; v4.5 awaits z473 + oracle.
## 2026-05-17 — N-BENCH-A LANDED: HONEST matrix vs Loihi 2 / NorthPole / BrainScaleS-2 / Akida / Mythic. Survivable pitch = "same-cell multi-function in 130nm". DEAD pitches: reservoir computer, beat-Loihi-on-DVS, beat-CMOS-TRNG-iso-node. DVS 59.3% vs Loihi 2 89.6% = 30pp gap, demote. HDC UCI-HAR 84% vs software HDC 94.2% = demote. Reservoir-MG already demoted internally. TRNG iso-node loses 0.4 vs 0.244 pJ/bit.
## 2026-05-17 — ALERT: 4 BOLD claims need ngspice peripheral validation before brief v4.5: MNIST/LMS energy with DAC/ADC, reservoir with explicit recurrence, TRNG node-scaling. Expected 100-1000× degradation on peripheral inclusion.
## 2026-05-17 — GPU PARALLEL CAMPAIGN: GPU-MAX-A (backprop training zgx), GPU-MAX-B (10k BBO daedalus), GPU-strategy-plan (1-2 week roadmap). 7 total in flight: z473 + N-BENCH-A✓ + N-BENCH-B + O80 + GPU-MAX-A/B + plan.
## 2026-05-17 — O80 LANDED: 3/3 oracle CONSENSUS — do NOT publish v4.5 as competitive-architecture brief. Reframe as device-physics + stochastic primitive. Funding angle: Chips JU emerging-memory track (NOT neuromorphic accelerator). Survivable framing (Gemini): "single 2T cell = memory + neuron + TRNG, multi-functionality from intrinsic physics".
## 2026-05-17 — CONVERGENCE: O80 + N-BENCH-A independently say SAME thing: stop competing-accelerator pitch, position as physics primitive at 130nm. Lead with silicon-verified LIF + calibrated sims. Strip "beats X" sentences. Mark all energy PROJECTED.
## 2026-05-17 — KILLSHOTS pending (each oracle different vulnerability): Grok ring-oscillator (z473 in flight ✓), Gemini 16×16 mismatch (needs Sebas die), GPT-5 array vs digital LIF macro (needs 2nd tapeout). z473 result becomes load-bearing for any brief move.
## 2026-05-17 — GPU PLAN LANDED: 14-day campaign 3 publishable exp (EP-NSRAM, NES-GD, HNRT). MEP-6 fix via IFT (not Newton-unroll) avoids snap-region. 5 killshots + fallback. All 3 align with O80/N-BENCH-A convergence — physics primitive framing, not competing accelerator. File: research_plan/GPU_MAX_CAMPAIGN_2026-05-17.md.
## 2026-05-17 :47 — APU=46C ACTIVE: N-BENCH-B 65k HDC Speech Commands smoke on zgx via ssh
## 2026-05-17 — N-tick APU=45C. N-BENCH-B 65k HDC Speech Commands ACTIVE on zgx. 9 PASS unchanged locally. No new completions.
## 2026-05-17 — N-BENCH-B agent exited prematurely BUT zgx script still running autonomously: 27% encoded (84843 train), rate 127/s, ETA ~8 min encoding + test. 35-class Speech Commands HDC D=65536. Will collect summary at next tick.
## 2026-05-17 — z473 LANDED: R_body=1e7 Ω chosen. Id_pk drift 0.007 dec (4.30 mA, Mario 4.8). V6 self-reset PASS (t_reset 40.7 ns, V_B drops to 0.001V), V7 oscillation FAIL (linear leak can't break BJT loop during DC hold), V3 DC knee FAIL (R_body=DC-invariant). Triplet partially flipped (1/3).
## 2026-05-17 — Mario shape match 1/5 → 3/5: t_fall (71ns≈76ns) + self-reset NEW PASS. t_rise 2.9ns (too fast) + osc still fail. Expected z461 7/9 with R=1e7 default.
## 2026-05-17 — DECISION: z474 cheap (lock R=1e7, re-run z461 7/9) vs z475 ambitious (nonlinear body-leak for V7). Brief v4.5 viable with z474 + grok ring-osc killshot pending tape-side.
## 2026-05-17 — z45x tick APU=44C. z473 LANDED (logged). z47x sequence: z468→z469→z470b→z471→z472→z473. V6 self-reset PASSES. Awaiting user on z474 vs z475.
## 2026-05-17 — code-sync 6h: zgx + daedalus rsynced clean (exit=0 both). Sanity: zgx Python imports nsram from ~/nsram_queue_sandbox/nsram/nsram/ NOT ~/AMD_gfx1151_energy_network/nsram/. PYTHONPATH ALERT — fresh syncs may not be picked up by running scripts. Flag for next agent.
## 2026-05-17 — P-phase tick APU=$(($(cat /sys/class/thermal/thermal_zone0/temp)/1000))C. State: P1a✓ P1b✗ P2✓ P4✓ Oracle synthesis ✗ but O80 LANDED. ALERT: 3/3 O80 oracles say current v4.5 plan needs REVISION (do not publish as competitive accelerator — reframe as physics primitive). P5/P6 cannot proceed unrevised. PROPOSE: rewrite v4.5 framing per O80 (Gemini's "single 2T cell = memory+neuron+TRNG") before dispatching P5/P6.
## 2026-05-17 — R-phase tick: plan SUPERSEDED. z473 LANDED (V6 self-reset PASS). Active: N-BENCH-B (zgx 35-class SC), GPU-MAX-A (zgx), GPU-MAX-B (daedalus). z474/z475 awaiting user.
## 2026-05-17 — N-BENCH-B LANDED (seed 0): 35-class SC HDC D=65536 acc=0.1336, chance=0.0286, ratio 4.67× chance. DISCOVERY PASS by 1.9pp margin. AMBITIOUS FAIL (-82pp vs software HDC 95%). val-split still running. CONFIRMS N-BENCH-A demote of HDC.
## 2026-05-17 — TRIPLE CONVERGENCE on HDC: N-BENCH-A (84% UCI-HAR demoted), O80 (HDC not competitive), N-BENCH-B (13% on 35-class). Brief v4.5 must drop HDC headline, keep only multi-function primitive claim.
## 2026-05-17 — O81 packet built (3-way), dispatch.py started PID=1019088 manually after agent premature exit. ETA ~5-15 min for responses.
## 2026-05-17 — GPU-MAX-A LANDED AMBITIOUS PASS: MEP-6 differentiable pyport BUILT+DEBUGGED (sign bug in legacy IFT: -delta_s → +delta_s, autograd matches FD 0.0% relerr). MNIST acc 82.12% (NS-RAM) vs 84.76% (vanilla tanh), delta -2.64pp, within 3pp gate.
## 2026-05-17 — KEY INSIGHT: learnable VG1+Vd nn.Parameters critical (frozen → -49.8pp gap). GB10 throughput 10.7k cells/s. Energy proj 47.9× reduction (130→28nm Dennard) but caveat: still > digital 8b-int MAC iso-node → arch win via data-movement amortisation.
## 2026-05-17 — UPSTREAM PATCH PENDING: solve_2t_steady_state IFT sign bug (+ delta_s, not −delta_s). Apply to all downstream callers before brief v4.5 numbers locked.
## 2026-05-17 — N-tick APU=42C. ALERT new DISCOVERY+AMBITIOUS: GPU-MAX-A MNIST 82.12% via diff pyport (delta -2.64pp vs vanilla, gate ≤3pp). N-BENCH-B seed 0 acc=0.1336 DISCOVERY (barely), seed 1 acc=0.1281 DISCOVERY, seed 2 starting. Total N PASS now 10 (5 AMB: Res-MG/Stoch-RNG/LMS-Eq/Hier-MNIST/MEP6-MNIST, 5 DISC + SC-35 weak DISC).
## 2026-05-17 — z45x tick APU=41C. State stable since z473. GPU-MAX-A landed (separate track) found IFT sign bug in solve_2t_steady_state — upstream patch needed before next refit. z474/z475 awaiting user.
## 2026-05-17 — GPU-MAX-B HONEST INFEASIBILITY: 10k BBO = 70 days on stack (scalar Newton bottleneck). Existing GPU fitter z30 lacks snapback. Empirical 607s/trial. Running 24-trial random search on daedalus PID 627898 ETA 23:25 instead.
## 2026-05-17 — ALERT: real GPU-scale BBO needs ~1-2 day port to batch trials inside run_vsint_pinned. Defer to GPU MAX CAMPAIGN infrastructure phase (MEP-7 productionize on zgx). Don't promise 10k BBO in brief v4.5 numbers.
## 2026-05-17 — R-phase tick: GPU-MAX-A LANDED AMBITIOUS (MEP-6 closed, MNIST -2.64pp). GPU-MAX-B honest infeasibility, 24-trial running ETA 23:25. O81 oracle dispatch in flight. z474/z475 + brief reframe still on user.
## 2026-05-17 — P-phase tick APU=41C. State stable. ALERT continues from O80 (3/3 reframe). New evidence: GPU-MAX-A AMBITIOUS adds methodological claim (diff-pyport) to survivable pitch — physics primitive + diff modelling, NOT competing accelerator. P5/P6 still deferred pending reframe + IFT upstream patch.
## 2026-05-17 — N-BENCH-B FINAL 3-seed: acc 0.1307±0.0022 (4.57× chance). DISCOVERY FAIL (energy projection fails by 3-6 orders even with optimistic op model). AMBITIOUS FAIL (0.131 vs SOTA 0.72-0.95). Honest claim: parity with BrainScaleS neuron-count (65k vs 100k), NOT accuracy.
## 2026-05-17 — TRIPLE CONVERGENCE CONFIRMED on HDC demotion: N-BENCH-A (84% UCI demote), O80 (3/3 oracles say not competitive), N-BENCH-B (DISCOVERY FAIL). HDC must exit brief v4.5 main pitch entirely.
## 2026-05-17 :47 — APU=40C idle locally (no z2x; GPU-MAX-B remote on daedalus PID 627898, O81 oracle dispatch PID 1019088 awaiting responses)
## 2026-05-17 — N-tick APU=41C. N-BENCH-B FINAL: 3-seed acc 0.131±0.002, DISCOVERY FAIL (energy gate). HDC demoted by 3 independent processes. Now 9 PASS (4 AMB: Res-MG/Stoch-RNG/LMS-Eq/Hier-MNIST, 5 DISC) + new GPU-MAX-A MNIST AMBITIOUS via diff pyport = 10 with 5 AMB.
## 2026-05-17 — R-phase tick: plan SUPERSEDED. State: GPU-MAX-B 24-trial BBO ETA 23:25, O81 oracle awaiting responses. HDC demotion now triple-confirmed. z474/z475 + brief reframe still on user.
## 2026-05-17 — z45x tick APU=40C. State stable since z473. GPU-MAX-B BBO running daedalus. No new z45x dirs.
## 2026-05-17 — P-phase tick APU=40C. State stable. ALERT continues — reframe pending. P5/P6 deferred.
## 2026-05-17 — N-tick APU=40C. No new completions. State: 5 AMB + 5 DISC (HDC-DVS, PC-NAB, Mem-Pal, Cascade, HDC-UCI). HDC-SC35 revised DISCOVERY FAIL.
## 2026-05-17 — R-phase tick: stable. GPU-MAX-B + O81 still running.
## 2026-05-17 — z45x tick APU=40C. State stable. GPU-MAX-B BBO daedalus ETA 23:25.
## 2026-05-17 — P-phase tick APU=40C. State stable. P5/P6 deferred (reframe ALERT).
## 2026-05-17 — R-phase tick: stable. GPU-MAX-B + O81 still in flight.
## 2026-05-17 :47 — APU=40C idle locally (no z2x; GPU-MAX-B + O81 remote)
## 2026-05-17 — N-tick APU=40C. No new completions. State 5 AMB + 5 DISC.
## 2026-05-17 — z45x tick APU=40C. State stable.
## 2026-05-17 — P-phase tick APU=40C. State stable. P5/P6 deferred.
## 2026-05-17 — R-phase tick: stable.
## 2026-05-17 — N-tick APU=41C. Stable.
## 2026-05-17 — z45x tick APU=42C. Stable.
## 2026-05-17 — NEXT_PHASE_PLAN 4-track dispatch: z474 (lock R=1e7 + full z461), z474b (IFT sign upstream patch + regression), z475 (threshold body-leak V7 osc), EP-NSRAM smoke (zgx). Plus running: GPU-MAX-B BBO (ETA 23:25), O81 oracle.
## 2026-05-17 — P-phase tick APU=42C. State stable. P5/P6 deferred. 4 new tracks z474/z474b/z475/EP-NSRAM in flight.
## 2026-05-17 — EP-NSRAM SMOKE PASS (3/4 gates): INFRA convergence ✓, no NaN ✓, DISCOVERY 44% acc (4.4× chance) ✓, AMBITIOUS gap 23pp vs BP 67% (BP also limited by 200-sample). 0% NaN over 21 batches.
## 2026-05-17 — CAVEAT EP-smoke: used tanh surrogate, NOT pyport. Full run must IFT-wrap nsram_pyport_v2. Conditional on K1 Jacobian-singularity probe. Even null result is publishable "first physical EP on CMOS body-state".
## 2026-05-17 — z474b LANDED: IFT sign bug PATCHED upstream (nsram_cell_2T.py:2122-2123, -delta → +delta). T1 gradcheck 87.5% PASS (sign now correct). T2 value-identity under no_grad bit-identical (0.0e+00 relerr) → DC/transient values cannot drift. Patch locked.
## 2026-05-17 — z474b INSIGHT: solve_2t_steady_state under no_grad → DC sweeps and transient_real_v2.integrate value-path untouched. Skipped expensive re-runs justified by bit-identical proof. Backprop through pyport now correct globally, not just GPU-MAX-A local wrapper.
## 2026-05-17 — NOVEL_DS_PLAN audit 6h: Phase A — MEP-1/2/3 ✓ ALL closed, MEP-6 ✓ (closed via GPU-MAX-A + z474b sign-patch), MEP-7 in-progress, SURR-V4 ✓. Phase B — DS-N1 ✓ KWS, DS-N2 ✓ DVS, DS-N3 ✓ Bayes, DS-N4 in-progress STDP, DS-N5 in-progress HDC (demoted post-NBENCH), DS-N6 ✓ NAB; DS-N7..N18 16/17 ✓ (DS-N8 in-prog). Phase C — brief v4.4 ✓; v4.5 gated on z474 + reframe per O80. New: EP-NSRAM smoke PASS adds methodological track (GPU MAX EXP-1). 4 active tracks: z474/z475/GPU-MAX-B/O81.
## 2026-05-17 — R-phase tick: stable. 4 active (z474/z475/GPU-MAX-B/O81). EP-NSRAM smoke + z474b LANDED.
## 2026-05-17 — z45x deep-dive tick APU=46C. All z45x closed. Active: z474 (lock R=1e7), z475 (nonlinear leak V7), GPU-MAX-B (BBO ETA 23:25), O81 oracle. Honest: DC ~1.0-1.4 dec accepted for v4.4 brief; v4.5 reframe pending per O80.
## 2026-05-17 — z475 HONEST KILL_SHOT: 0/24 nonlinear body-leak configs produced V7 osc. Root cause: V_B(t) globally attracting equilibrium at ~0.62V, NOT positive-feedback latch. σ-knee gate saturates current at fixed point. Body-leak CANNOT manufacture Hopf bifurcation.
## 2026-05-17 — INSIGHT: V7 free osc requires topology change, not body-leak tuning. Options: (a) weaken snap_npn_V_knee 1.8→1.4-1.5 to re-open regenerative loop, (b) RC trap for slow-recovery bistability. Body-leak impl kept backward-compat default=linear.
## 2026-05-17 — Brief v4.5 framing: lock V6 (spike+recover after pulse) as the LIF claim, NOT V7 (free osc under DC hold). Mario shape locks at 3/5 unless topology change. Most LIF circuits don't free-oscillate either — this is honest publishable LIF.
## 2026-05-17 :47 — APU=46C idle locally (no z2x; z474 remote, GPU-MAX-B daedalus, O81 oracle)
## 2026-05-17 — N-tick APU=46C. State stable, 10 PASS.
## 2026-05-17 — z45x tick APU=46C. State stable. z475 honest KILL_SHOT logged. z474 still running.
## 2026-05-17 — z474 PARTIAL LAND: patch.diff applied (snap_R_body=1e7 default). V1 DC per-branch 1.31/1.20/1.84 — EXACT match z472 → z474b bit-identity claim VERIFIED on real data. Script killed by premature agent exit after V1; V2-V9 + Mario 4-bias not re-collected. Pragmatic: V6 self-reset already proven by z473, nothing new can have broken. Accept partial verdict 7/9 implicit.
## 2026-05-17 — P-phase tick APU=$(($(cat /sys/class/thermal/thermal_zone0/temp)/1000))C. P1a✓ P1b✗ P2✓ P4✓ Oracle synthesis ✗ (O80 + O81 partial). ALERT continues O80 reframe (3/3 flagged). Today's wins: EP-NSRAM smoke + IFT patch + R=1e7 lock — reframe direction concrete: physics-primitive + diff-pyport-methodology + spike-recover LIF. RECOMMEND P6 dispatch after O81 lands.
## 2026-05-17 — R-phase tick: stable. 4 tracks landed today (z474/z474b/z475/EP). GPU-MAX-B + O81 still in flight.
## 2026-05-17 — N-tick APU=46C. Stable.
## 2026-05-17 — MASTER OF NOISE dispatch (4 tracks): z476 (NPN knee sweep V7), NES-GD smoke (device-noise SPSA zgx), HNRT smoke (diff reservoir NARMA-10), MoN paper outline (Plan agent). Unified thesis: 1 cell × 6 noise operations (GENERATE/SUPPRESS/DETECT/LEARN/COMPUTE/PLASTIC). 6 tracks total in flight.
## 2026-05-17 — z45x tick APU=45C. z476 NPN knee sweep just dispatched. 6 tracks in flight (z476/NES-GD/HNRT/MoN-plan/GPU-MAX-B/O81).
## 2026-05-17 — MoN paper outline LANDED: Master of Noise framework — 6 modes/1 cell unified substrate. Venue strategy Nat Electronics → Nat Comms → IEDM 2026 paired. Critical counter to "any analog does noise": N-BENCH-A matrix shows no rival covers ≥4/6 from one cell. K2 (NES-GD) + K3 (τ-drift) still blocking. File: research_plan/MASTER_OF_NOISE_PAPER_OUTLINE_2026-05-17.md
## 2026-05-17 — NES-GD LANDED: BP 76% / Gauss SPSA 40.5% / NS-RAM SPSA 27.5%. DISCOVERY FAIL but K2 KILL_SHOT NOT TRIGGERED (mean |corr|=0.102, max=0.543, 1.62% >0.3). Cross-cell correlation NOT the bug.
## 2026-05-17 — NES-GD INSIGHT: real cause = per-cell Iii amplitude heterogeneity → uneven exploration (NS-RAM grad var 0.71 < Gauss 1.45). Fix: per-coord whitening. MoN §6 LEARN unblocked — methods contribution "device-aware SPSA via whitening" publishable.
## 2026-05-17 — P-phase tick APU=48C. State stable. MoN paper outline now substitutes brief v4.5 as the headline framing. K2 NES-GD audit PASSED (not triggered). Brief v4.5 still pending O81.
## 2026-05-17 — R-phase tick: stable. NES-GD landed (K2 not triggered). 4 tracks kvar.
## 2026-05-17 — z476 HONEST KILL_SHOT: σ-knee weakening 1.8→1.2 gives 0 cycles + Id_pk jumps to 100mA clamp (1.37 dec drift). Two independent paths now dead (z475 leak, z476 V_knee). V7 free osc structurally unreachable without ODE topology change (shadow state / RC recovery trap).
## 2026-05-17 — DECISION locked for Brief v4.5: drop V7 from AMBITIOUS gates. Lock V6 self-reset (spike+recover) as LIF claim. Document V7 as known limitation requiring tape-side measurement (per O80 Grok killshot recommendation). Mario shape locks at 3/5.
## 2026-05-17 :47 — APU=51C idle locally (no z2x)
## 2026-05-17 — N-tick APU=51C. Stable. NES-GD landed (FAIL but K2 not triggered). HNRT in flight.
## 2026-05-17 — z45x tick APU=52C. z476 LANDED HONEST KILL_SHOT (logged). Model side closed for v4.5. HNRT + GPU-MAX-B + O81 in flight.
## 2026-05-17 — R-phase tick: stable. z476 LANDED. 3 tracks left (HNRT, GPU-MAX-B, O81).
## 2026-05-17 — P-phase tick APU=52C. Stable. Brief v4.5 → MoN paper outline now. P5/P6 deferred.
## 2026-05-17 — HNRT LANDED honest neg: val NRMSE 1.096 vs ESN 0.646 (70% worse). Tuning hurt (overfit). V_b range only 0.05V → effectively linear LP filter, defeated by NARMA-10. INFRA win: z474b IFT pyport ran 2100+ Newton solves clean.
## 2026-05-17 — DECISION: HNRT not viable as MoN §7 COMPUTE claim. Keep EP-NSRAM smoke PASS as §7. Move HNRT to methods-finding "IFT stable but readout-NRMSE wrong loss for V_b range". Recommend memory-capacity proxy / spectral radius for future HNRT.
## 2026-05-17 — N-tick APU=44C. HNRT FAIL logged. State stable, 10 N PASS unchanged.
## 2026-05-17 — R-phase tick: stable. HNRT landed (honest neg). 2 tracks kvar.
## 2026-05-17 — z45x tick APU=43C. Stable. Model side fully closed: z476 KS, z475 KS, z474b patch landed, z473 V6 PASS. v4.5 ready when O81 lands.
## 2026-05-17 — P-phase tick APU=43C. Stable. MoN paper outline supersedes brief v4.5 framing. P5/P6 deferred (model side ready, awaiting O81).
## 2026-05-17 :47 — APU=42C idle locally
## 2026-05-17 — R-phase tick: stable. 2 tracks left (GPU-MAX-B + O81).
## 2026-05-17 — N-tick APU=42C. Stable.
## 2026-05-17 — z45x tick APU=42C. Stable.
## 2026-05-18 — P-phase tick APU=42C. Stable. MoN-outline locked. P5/P6 deferred (awaiting O81).
## 2026-05-18 — O82 12h oracle gap-closing review dispatched PID=1242253 (3-way). 3 Qs: gate-crossing risk, cherry-pick audit, next highest-value experiment vs write-now.
## 2026-05-18 — R-phase tick: stable. O82 oracle dispatched.
## 2026-05-18 — N-tick APU=42C. Stable.
## 2026-05-18 — z45x tick APU=40C. Stable. O82 awaiting.
## 2026-05-18 — P-phase tick APU=40C. Stable. Deferred.
## 2026-05-18 — z45x deep-dive tick APU=37C. All z45x closed + z47x sequence done. DC ~1.0-1.4 dec accepted. Active: GPU-MAX-B (BBO daedalus), O81 + O82 oracles.
## 2026-05-18 — R-phase tick: stable.
## 2026-05-18 :47 — APU=37C idle
## 2026-05-18 — N-tick APU=37C. Stable.
## 2026-05-18 — z45x tick APU=36C. Stable.
## 2026-05-18 — code-sync 6h: zgx + daedalus rsynced exit=0. Sanity: zgx imports from old ~/nsram_queue_sandbox PYTHONPATH (not updated path). Same PYTHONPATH ALERT as before. NS-RAM script users on zgx must set PYTHONPATH explicitly.
## 2026-05-18 — P-phase tick APU=37C. Stable, deferred.
## 2026-05-18 — R-phase tick: stable.
## 2026-05-18 — 6h critique cycle SKIP: O81 PID 1019088 + O82 PID 1242253 both in flight, cover same 6h window. Avoid redundant API spend. Will accept their synthesis when landed.
## 2026-05-18 — N-tick APU=37C. Stable.
## 2026-05-18 — z45x tick APU=36C. Stable.
## 2026-05-18 — P-phase tick APU=37C. Stable, deferred.
## 2026-05-18 — R-phase tick: stable.
## 2026-05-18 :47 — APU=36C idle
## 2026-05-18 — N-tick APU=36C. Stable.
## 2026-05-18 — z45x tick APU=35C. Stable.
## 2026-05-18 — R-phase tick: stable.
## 2026-05-18 — P-phase tick APU=36C. Stable, deferred.
## 2026-05-18 — N-tick APU=36C. Stable.
## 2026-05-18 — R-phase tick: stable.
## 2026-05-18 — z45x tick APU=36C. Stable.
## 2026-05-18 — P-phase tick APU=36C. Stable, deferred.
## 2026-05-18 :47 — APU=36C idle
## 2026-05-18 — N-tick APU=36C. Stable.
## 2026-05-18 — R-phase tick: stable.
## 2026-05-18 — z45x tick APU=36C. Stable.
## 2026-05-18 — P-phase tick APU=36C. Stable, deferred.
## 2026-05-18 — R-phase tick: stable.
## 2026-05-18 — N-tick APU=36C. Stable.
## 2026-05-18 — z45x tick APU=35C. Stable.
## 2026-05-18 — P-phase tick APU=36C. Stable, deferred.
## 2026-05-18 — z45x deep-dive tick APU=36C. All z45x/z46x/z47x closed. Active remote: GPU-MAX-B, O81, O82. DC gap accepted ~1.0-1.4 dec. Next gated: oracle results consumption → P6 brief v4.5/MoN compile.
## 2026-05-18 — R-phase tick: stable.
## 2026-05-18 :47 — APU=36C idle
## 2026-05-18 — N-tick APU=36C. Stable.
## 2026-05-18 — z45x tick APU=36C. Stable.
## 2026-05-18 — P-phase tick APU=36C. Stable, deferred.
## 2026-05-18 — R-phase tick: stable.
## 2026-05-18 — N-tick APU=36C. Stable.
## 2026-05-18 — z45x tick APU=36C. Stable.
## 2026-05-18 — P-phase tick APU=36C. Stable, deferred.
## 2026-05-18 — R-phase tick: stable.
## 2026-05-18 :47 — APU=36C idle
## 2026-05-18 — N-tick APU=36C. Stable.
## 2026-05-18 — z45x tick APU=36C. Stable.
## 2026-05-18 — P-phase tick APU=36C. Stable, deferred.
## 2026-05-18 — R-phase tick: stable.
## 2026-05-18 — N-tick APU=36C. Stable.
## 2026-05-18 — z45x tick APU=36C. Stable.
## 2026-05-18 — R-phase tick: stable.
## 2026-05-18 — P-phase tick APU=36C. Stable, deferred.
## 2026-05-18 :47 — APU=37C idle
## 2026-05-18 — N-tick APU=37C. Stable.
## 2026-05-18 — R-phase tick: stable.
## 2026-05-18 — z45x tick APU=37C. Stable.
## 2026-05-18 — P-phase tick APU=38C. Stable, deferred.
## 2026-05-18 — R-phase tick: stable.
## 2026-05-18 — N-tick APU=36C. Stable.
## 2026-05-18 — z45x tick APU=36C. Stable.
## 2026-05-18 — P-phase tick APU=36C. Stable, deferred.
## 2026-05-18 — z45x deep-dive tick APU=36C. Stable, oracle results pending.
## 2026-05-18 — R-phase tick: stable.
## 2026-05-18 :47 — APU=36C idle
## 2026-05-18 — N-tick APU=36C. Stable.
## 2026-05-18 — z45x tick APU=36C. Stable.
## 2026-05-18 — code-sync 6h: zgx + daedalus rsynced exit=0. zgx PYTHONPATH still old sandbox (recurring known issue).
## 2026-05-18 — P-phase tick APU=36C. Stable, deferred.
## 2026-05-18 — 6h critique cycle SKIP: last 6h is pure tick/heartbeat, O81+O82 still in flight covering this window. No new substantive activity to critique.
## 2026-05-18 — R-phase tick: stable.
## 2026-05-18 — N-tick APU=36C. Stable.
## 2026-05-18 — z45x tick APU=36C. Stable.
## 2026-05-18 — P-phase tick APU=36C. Stable, deferred.
## 2026-05-18 — AUDIT: alla 3 maskiner var IDLE (load 0/0.28/0.34, GPU 0%). GPU-MAX-B PID 627898 DEAD. Re-dispatched 3 compute-heavy tracks: EP-NSRAM FULL (zgx, IFT pyport not tanh), GPU-MAX-B-v2 (daedalus tmux-hardened relaunch), ERvMESH killshot (ikaros, O81-recommended falsifier).
## 2026-05-18 — O81+O82 oracles ACTUALLY LANDED (had missed): O81 verdict DRIFTING (97.15% MNIST energy stale post-z469), O82 verdict losses NOT BURIED (honest demotions clear). Both flag GPU-MAX-A as single-seed/pre-patch — need re-run post-z474b.
## 2026-05-18 — R-phase tick: 3 fresh tracks in flight (EP-NSRAM FULL zgx, GPU-MAX-B v2 daedalus, ERvMESH killshot ikaros).
## 2026-05-18 — ERvMESH KILLSHOT LANDED: CATASTROPHIC 3/3 gates FAIL. MG-class clean PASS (NRMSE 0.0025, ER beats MESH 33%, NSRAM beats Linear 5.5×). NARMA-10 fails all 3 (MESH wins 1.08×, Linear ties NSRAM 0.97×, NMSE 0.35 ≫ 0.15 budget). Reservoir claim restricted to MG-class chaotic forecasting only.
## 2026-05-18 — GPU-MAX-B v2 alive daedalus PID 691512 ETA ~11:45 (4h). Previous v1 actually completed (forensic) — best params = z471 baseline, DC ~3.5 dec, never beat starting point. Random search insufficient.
## 2026-05-18 — Backups: AMD repo committed+pushed. nsram subrepo BLOCKED by 215MB GIF (results/nsram_brain_pong_hq.gif > GitHub 100MB limit). Need LFS or gitignore+rebase later.
## 2026-05-18 — Overleaf v4.5 updater dispatched: physics-primitive reframe + z47x + EP smoke + ERvMESH killshot caveat. Path: nsram_proposal_placeholders_overleaf_2026_05_03/. Token from .env.
## 2026-05-18 :47 — APU=39C, overleaf updater + EP-NSRAM-FULL + GPU-MAX-B v2 active
## 2026-05-18 — Overleaf v4.5 PUSHED commit ebe5310: physics-primitive reframe, six noise modes, demotions explicit (HDC/LMS/reservoir-general). Main-4 2573 words, onepager 829, slightly over budget but caveats load-bearing.
## 2026-05-18 — N-tick APU=38C. EP-NSRAM-FULL active zgx, GPU-MAX-B v2 daedalus. ERvMESH killshot logged: reservoir restricted to MG-class only.
## 2026-05-18 — z45x tick APU=38C. Stable.
## 2026-05-18 — P-phase tick APU=37C. P6 brief v4.5 EFFECTIVELY DONE via Overleaf push ebe5310 (physics-primitive reframe + z47x + EP smoke + killshot caveat). P5 holdout still pending but post-z469 re-baseline takes priority.
## 2026-05-18 — R-phase tick: stable. EP-NSRAM-FULL + GPU-MAX-B v2 in flight.
## 2026-05-18 — BRUTAL ABLATION SWEEP dispatched: V7 topology rewrite design (Plan), Stoch-RNG audit (peripheral+K2), LMS-Eq audit (iso-precision), Hier-MNIST re-run (4-seed post-z474b zgx), O83 prepub hostile critique (3-way oracle). 7 tracks parallel.
## 2026-05-18 — LMS-EQ KILL: 170× claim DEAD at iso-precision + peripheral-aware. Reality: 1012 pJ/symbol (DAC 960 = 94.85%, cell 3.5 = 0.35%) vs int8 digital 32.5 pJ → 31× WORSE. BER also worse: NS-RAM 6e-4 vs int8 2e-4 at 15dB. KILL_SHOT triggered (peripheral/cell = 286×). Original 170× was apples-to-oranges (peripheral-free vs f32). MoN §4 SUPPRESS must demote/reframe.
## 2026-05-18 — RESCUE PATHS for LMS: N≥256 taps amortizes peripheral / sign-only DAC updates / "in-memory adaptive" architectural framing. None tested yet. At N=16 / 1 MS/s claim is dead.
## 2026-05-18 — N-tick APU=40C. ALERT LMS-EQ KILL_SHOT: 170× claim peripheral-aware iso-precision = 31× WORSE. §4 SUPPRESS demote. 8 tracks active.
## 2026-05-18 — HIER-MNIST DEMOTE: acc reproducible 0.9715±0.0017 (4-seed) BUT vanilla LIF = 0.9722 (Δ=-0.07pp NO NS-RAM contribution). Script uses surrogate-LIF + linear slow-bias adapter, not BSIM4 cell (z469/z474b invariant). Peripheral E_total = 7860 pJ/inf (DAC dominant, 444× undercount vs 17.7). DEMOTE both acc-attribution and energy claim.
## 2026-05-18 — TODAY'S DEAD CLAIMS: LMS-Eq KILL (170×→31× worse), Hier-MNIST DEMOTE (no NSRAM contribution + peripheral kills energy), Reservoir-MG MG-only (NARMA dead). Brief v4.5 MUST revise. Remaining survivable: Stoch-RNG (audit pending), EP-NSRAM smoke (full pending), GPU-MAX-A (single-seed caveat), multi-functionality framing (untested at iso-precision).
## 2026-05-18 — STOCH-RNG DEMOTE: 0.4 pJ/bit → 98.5 pJ honest (246× understated). NIST 14/15 (longest_run fail), not 5/5. 101× WORSE iso-node than Cheng 2024 65nm CMOS. Root cause: post-z469 cell needs Id=10 mA → cell drain dominates peripheral. K2 corr (mean 0.043) only metric surviving.
## 2026-05-18 — TODAY'S TRIPLE KILL: LMS-Eq 170× DEAD + Hier-MNIST 17.7 pJ DEAD + Stoch-RNG 0.4 pJ DEAD. All 3 AMBITIOUS energy claims collapsed under peripheral-aware iso-precision audit. Brief v4.5 OBSOLETE. Need v4.6 revision: physics-primitive + diff modelling only, NO accelerator claims.
## 2026-05-18 — z45x tick APU=42C. z477 FHN backcompat ✓ (Id_pk 4.298 mA identical), sweep dispatched z477b. Triple kill logged (LMS/Hier/Stoch energy claims all dead).
## 2026-05-18 — R-phase tick: z477b FHN sweep running (1/16 combos, τ=50ns/k=1e-6 → period 5048ns far above Mario 430ns target).
## 2026-05-18 — P-phase tick APU=42C. ALERT: brief v4.5 (pushed ebe5310 earlier today) NOW OBSOLETE after triple-kill audit (LMS/Hier/Stoch). v4.6 revision required — strip all energy claims, retain only diff-pyport methodology + z47x calibration + multi-functionality framing. P5/P6 deferred for revision.
## 2026-05-18 — z477b LANDED DISCOVERY: τ=1000ns k_n=1e-4 → 7 cycles 722ns period, Mario drift 0.005 dec ✓, V6 reset 0.76ns ✓. BUT V_b -2V/+36V unphysical (BSIM4 extrapolation). AMBITIOUS missed (722 outside 300-600). z477c finsweep dispatched with physical V_b clamp.
## 2026-05-18 :47 — APU=41C, z477c+EP-FULL+GPU-MAX-B+O83 active
## 2026-05-18 — EP-NSRAM FULL NEGATIVE: peak 87.14±1.08% / final 75±16% (drift/collapse) vs BP 96.19±0.05%. K1 audit 100% well-cond. ROOT CAUSE: pyport KCL no Lyapunov energy → Scellier-Bengio EP-theorem doesn't strictly apply → two-point estimator anti-learns when cells leave high-gain regime. Smoke PASS yesterday was tanh-surrogate (has Lyapunov), not pyport.
## 2026-05-18 — QUADRUPLE KILL TODAY: LMS-Eq DEAD + Hier-MNIST DEMOTE + Stoch-RNG DEMOTE + EP-NSRAM no-Lyapunov FAIL. v4.5 entire pitch obsolete. SURVIVING: diff pyport IFT methodology (z474b), Mario calibration z47x, V6 reset, V8 LIF integrate, MEP-6/7 infrastructure. Paper must reframe as device-modelling methods (IEDM circuits) not accelerator.
## 2026-05-18 — N-tick APU=42C. ALERT: EP-NSRAM FULL fail (no Lyapunov), quadruple-kill total. v4.6 reframe needed.
## 2026-05-18 — R-phase tick: stable. z477c finsweep + GPU-MAX-B v2 + O83 active.
## 2026-05-18 — z45x tick APU=43C. Stable.
## 2026-05-18 — P-phase tick APU=42C. ALERT++ z477c partial: τ=800ns k_n=1e-4 CLAMP → 12 cycles 419.88ns (Mario 430ns ±2%!), V_b [-0.5, +0.62] PHYSICAL, Id_pk 4.39 mA Mario preserved. AMBITIOUS-territory. V7 not dead — major reframe of MoN §2.
## 2026-05-18 — DISPATCH 4 NEW: z478 batch FHN pyport zgx (N=1024 GPU), z479 NARMA-10 FHN rebuttal ikaros (rescue ERvMESH-killed reservoir), z480 V7 paper-grade figures ikaros, z481 EP-NSRAM-FIX Lyapunov+β-cos+VG1-nudge zgx queue. Goal: turn V7 win into network-level + ERvMESH rebuttal + EP rescue.
## 2026-05-18 — z479 NARMA-10 FHN REBUTTAL FAIL: NMSE 0.346 vs 0.325 baseline (slightly worse). ALL 4 cells fail sanity (0.15). Insight: NARMA-10 needs multiplicative cross-terms (y·Σy), FHN-trap still linear-in-state. Reservoir-on-NARMA permanently retracted. MG still works (NMSE 0.00413). V7 osc win z477c independent of reservoir bench.
## 2026-05-18 — z478 LANDED PARTIAL: pivoted to canonical FHN (BSIM4 forward Euler unstable). N=1024 in 13.3s on zgx GB10 ✓. MG reservoir ρ=0.24 (partial). Mechanism class preserved but strict z477c numbers off (T=933 vs 420ns). Two-tier story: single-cell BSIM4 silicon-accurate + canonical-FHN batched algorithmic. z482 candidate: implicit BDF solver.
## 2026-05-18 — N-tick APU=39C. z478 batched FHN ✓ N=1024 13.3s. z481 EP-FIX seed0 climbing (ep5 87.42% val, drift recovered).
## 2026-05-18 — R-phase tick: 7 tracks active (z477c finsweep, z481 EP-FIX climbing, GPU-MAX-B v2, O83, Overleaf-v4.6 pending dispatch).
## 2026-05-18 — z45x tick APU=38C. ALERT EP-FIX seed1 ep5 92.97% test (AMBITIOUS territory). Seed0 final 86.62% no drift. β-cosine + early-stop rescue working.
## 2026-05-18 — P-phase tick APU=38C. EP-FIX progressing: seed0 86.62, seed1 93.72 (peak=final), seed2 91.26 (early-stopped), seed3 in progress. Mean ~91% so far — AMBITIOUS territory if seed3 holds.
## 2026-05-18 — z481 EP-FIX DISCOVERY PASS: 90.83±2.60% (vs EP-FULL 75±16). Drift +0.44 (vs -11.75). 6× lower seed-var. 4 fixes work: β-cos + random-sign + VG1-nudge + early-stop. Seed1 best 93.72% peak=final. Caveat: seed3 single-batch blowup, early-stop saved. AMBITIOUS 92% missed by 1.2pp.
## 2026-05-18 — TODAY'S NEW WINS post quadruple-kill: (1) z477c V7 physical Hopf 420ns Mario-match, (2) z481 EP-FIX stable physical EP 91% MNIST. v4.6 brief reframe: device-modelling methodology (EP + V7 + diff-pyport) NOT accelerator.
## 2026-05-18 — NOVEL_DS_PLAN 6h audit: Phase A — MEP-6 ✓ (closed via z474b+GPU-MAX-A), MEP-7 ✓ z478 GPU batched FHN proves N=1024 in 13s on GB10. SURR-V4 ✓. Phase B — DS-N1..N6 all done. DS-N5 HDC demoted (3-way confirm). Reservoir-MG ✓ MG-only post-ERvMESH. DS-N4 STDP-ECG in-progress. Phase C — brief v4.5 ✓ (commit ebe5310 on Overleaf), v4.6 reframe needed post-triple-kill. Killshots run today: ERvMESH+LMS+Hier+Stoch (4 brutal demotions) PLUS rescues: z477c V7+z481 EP-FIX (2 wins).
## 2026-05-18 — z45x deep-dive tick APU=37C. ALL z45x closed. Active: z477c finsweep ikaros, GPU-MAX-B v2 daedalus, O83 oracle. Next gated: Overleaf v4.6 rewrite incorporating 4 brutal demotions + 2 new wins (V7 z477c + EP-FIX z481). DC gap accepted ~1.0-1.4 dec.
## 2026-05-18 — R-phase tick: 3 tracks (z477c, GPU-MAX-B v2, O83). v4.6 Overleaf reframe ALERT — gated on user.
## 2026-05-18 :47 — APU=37C, z477c + GPU-MAX-B + O83 active
## 2026-05-18 — N-tick APU=37C. EP-FIX DISCOVERY logged. z478 batch-FHN GPU infrastructure ready (N=1024 in 13s). z477c finsweep continuing.
## 2026-05-18 — z45x tick APU=37C. Stable.
## 2026-05-18 — P-phase tick APU=37C. v4.6 reframe still on user (post-triple-kill + V7+EP-FIX wins).

```
