# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: context.md (16723 chars) ===
```
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
## 2026-05-18 — N-tick APU=41C. Active: z482 daedalus (coupled FHN reservoir running), O84 USP oracle local. Just completed: z483 STDP INFORMATIVE NULL (NSRAM τ_body=1ms ≡ hand 1ms +19pp, optimum τ=0.1ms 94.5%, decay NOT dominant). USP-plan delivered TOP 3 (EP-on-FHN, CPG lock-in, diff limit-cycle). Persistent backup ✓ ~/backups/amd_gfx1151_safety_*.
## 2026-05-18 — R-phase tick: TOPOLOGY_REBUILD_PLAN_2026-05-13 (R-1..R-10) CLOSED since 2026-05-16 campaign synthesis. Newer R-32..R-51 BSIM4/IIMOD all completed (see task list). Currently active under different IDs: z482 coupled FHN reservoir (daedalus), O84 USP oracle (local), z477c finsweep tail. No R-1..R-10 gate pending. APU=41C stable.
## 2026-05-18 — z45x tick APU=41C. All z44x/z45x dirs static (no mtime<2h). Closed since campaign-synthesis 2026-05-16. Currently active: z482 coupled FHN (daedalus), O84 oracle, z477c finsweep tail. No new ALPHA0/snapback DISCOVERY/KILL_SHOT pending.
## 2026-05-18 — z482 COUPLED FHN RESERVOIR KILLED: KILL gate triggered. NARMA 1.128 vs uncoupled 1.093 (worse), seq-MNIST 0.097 (chance), MG-17 1.008 vs ESN 0.029 (catastrophic). Mean spike synchrony 0.0016 — NO phase-locking. USP-2 (CPG lock-in) candidate from plan also collapses; USP-3 (diff limit-cycle reservoir) likely follows since coupled-FHN provides no temporal advantage.
## 2026-05-18 — O84 ORACLE 3-WAY CONVERGENCE: gpt-5+gemini+grok independently identify SAME USP — "multi-function 2T cell via bias-programming alone: V6 reset 40ns + V8 LIF + V7 Hopf 420ns + intrinsic noise + EP-FIX 91% MNIST via diff IFT pyport, all from same calibrated cell, no rival has this from 130nm 2T". CPG/reservoir USPs not endorsed by any oracle. Methods-paper framing unanimous.
## 2026-05-18 — DEATH-SENTENCE EXPERIMENTS unanimously prioritized: (1) DS-1 cheapest: full-MNIST EP-FIX 5-10 seeds, alignment-cosine, ablation; (2) DS-2 moderate: bias-only same-cell mode atlas on silicon (BLOCKED on Sebas); (3) DS-3 expensive: 16×16 mismatch/correlation matrix real die. ALERT: DS-1 fully dispatchable now (zgx free), pending user OK.
## 2026-05-18 — P-phase tick APU=39C. P1a ✓, P1b MISS (alt loc?), P2 ✓, P4 ✓, HONEST_BASELINE ✓ already written. Oracle_synthesis_2026-05-16 dir MISS but task#272 [pending] and 3-way O80/O81/O82/O83/O84 done elsewhere. P5/P6 unflagged BUT brief v4.5 ALREADY PUSHED to Overleaf (commit ebe5310, task#298-299 ✓) — superseded by today's quadruple-kill, v4.6 reframe pending user. No auto-launch (gates ambiguous given v4.5 already obsolete).
## 2026-05-18 — :47 idle APU=37C, z482 done, O84 done, awaiting DS-1 user OK
## 2026-05-18 — N-tick APU=37C. Local pys=3 (idle). No N_* mtime<2h. Active: zgx free, daedalus offline, z482 KILL logged, O84 convergence logged, DS-1 dispatch awaits user OK. No DISCOVERY.
## 2026-05-18 — R-phase tick APU=37C. TOPOLOGY_REBUILD_PLAN_2026-05-13 (R-1..R-10) remains CLOSED. No new gates. Active state: DS-1 dispatch pending user OK; z482/USP-2/USP-3 killed; USP-1 (multi-function 2T + EP-FIX) survives per O84 3-way convergence.
## 2026-05-18 — z45x tick APU=37C. New artifacts<2h=0. All z44x/z45x closed since 2026-05-16. No new DISCOVERY/KILL_SHOT.
## 2026-05-18 — P-phase tick APU=37C. Path B status unchanged: HONEST_BASELINE ✓, brief v4.5 ✓ ebe5310 obsolete post-quadruple-kill. v4.6 reframe pending user. No auto-launch.
## 2026-05-18 — R-phase tick APU=36C. R-1..R-10 CLOSED. No new gate.
## 2026-05-18 — N-tick APU=36C. Idle, no N_* mtime<2h. State unchanged: DS-1 + v4.6 reframe gated on user.
## 2026-05-18 — idle tick APU=36C pys=0, DS-1+v4.6 still gated on user
## 2026-05-18 — z45x tick APU=36C. new<2h=0. All closed. No DISCOVERY/KILL_SHOT.
## 2026-05-18 — P-phase tick APU=36C. Path B unchanged. v4.6 reframe on user.
## 2026-05-18 — R-phase tick APU=36C. R-1..R-10 CLOSED. No gate.
## 2026-05-18 — :47 idle APU=36C sentinel=2720
## 2026-05-18 — N-tick APU=36C idle. No N_* mtime<2h. State unchanged.
## 2026-05-18 — idle tick APU=36C, awaiting user DS-1/v4.6 decision
## 2026-05-18 — z45x tick APU=37C idle. All closed.
## 2026-05-18 — P-phase tick APU=37C idle. v4.6 reframe on user.

```
