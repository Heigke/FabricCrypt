# DEEP-DIVE NS-RAM Plan — 2026-05-12 (post-MEP-7 unlock)

**Trigger**: MEP-7 unlocks 1M-point GPU surrogate in 3.3s. DS-N5f
crossed HDC noise floor at 65.7% with V_d-as-bit. Now: find REAL
applications + architectures by combining (slides re-audit) ×
(massive GPU exploration) × (oracle critique loops).

## Phase 4A — Slide/Model RE-AUDIT (oracle + subagent driven)

Goal: extract ALL use-case hints from 49 Sebas+Mario slides, not just
constraints. Build authoritative I-V vs measurement visualizer.

| Step | Task | Method | Wall |
|---|---|---|---|
| 4A.1 | Re-extract slide numerics with USE-CASE focus | New oracle prompt (3 oracles, gpt-5+gemini+grok) — "what applications do slides suggest?" | 30 min |
| 4A.2 | Build 4×4 (C_b, V_Nwell) corner surrogates × dense 100K pts each | MEP-7 GPU on ikaros | 1h |
| 4A.3 | Per-V_G1-regime fit on dense surrogate | Closed-form lstsq on GPU | 30 min |
| 4A.4 | Slide-21 transient on dense surrogate (PMP-9 retry) | GPU batched | 20 min |
| 4A.5 | Auto-generated I-V family meas-vs-sim plots | Per-slide-bias overlay | 30 min |

## Phase 4B — ARCHITECTURAL ENVELOPE (massive GPU sweep)

Goal: map NS-RAM's analog computing envelope by GPU-batched sweeps.

| Step | Sweep | Hypothesis test | Wall |
|---|---|---|---|
| 4B.1 | V_d-as-bit HDC scale N ∈ {64, 128, 256, 512, 1024, 4096, 10000} | Does NS-RAM HDC scale with bit count? | 1h |
| 4B.2 | HDC noise tolerance (input σ ∈ {0, 0.05, 0.1, 0.2}) | Does analog noise help or hurt? | 30 min |
| 4B.3 | (V_d_HIGH, V_d_LOW) 2D fine sweep (5×5 grid) | Find symmetry × separation sweet spot | 30 min |
| 4B.4 | NS-RAM as analog comparator (binary classifier) | Pre-defined threshold task | 20 min |
| 4B.5 | NS-RAM as population coder (rank-coded scalar) | Spike-rate over N neurons → continuous output | 30 min |
| 4B.6 | Multi-cell ensembles (heterogeneous C_b populations) | Heterogeneity adds robustness | 30 min |

## Phase 4C — APPLICATION BENCHMARKS (real-world tasks)

Goal: validate NS-RAM on tasks where it should structurally win.

| Step | Task | Status | Wall |
|---|---|---|---|
| 4C.1 | DS-N4 MIT-BIH ECG (in flight) | running on daedalus | 1h |
| 4C.2 | DS-N6 Numenta NAB anomaly | new | 2h |
| 4C.3 | DS-N3 Bayesian MCMC w/ NS-RAM noise as native RNG | new | 2h |
| 4C.4 | DS-N1b KWS retry with DELTA-MODULATED MFCC (sparse events) | retry of failed DS-N1 | 1h |
| 4C.5 | Real DVS-Gesture (Tonic via figshare workaround) | DS-N2 retry with real data | 1h |
| 4C.6 | Edge anomaly: sub-mW always-on cardiac/IoT | combine ECG + NAB findings | 1h |

## Phase 4D — Oracle critique WAVES (3 rounds)

| Round | Trigger | Oracle focus | Wall |
|---|---|---|---|
| 4D.1 | After 4A closes | "Are use-case hypotheses well-grounded in slides?" | 25 min |
| 4D.2 | After 4B closes | "Where in architectural envelope are we cherry-picking?" | 25 min |
| 4D.3 | After 4C closes | "Falsify the application claims" | 25 min |

## Phase 4E — Brief v4.4

| Step | Output | Wall |
|---|---|---|
| 4E.1 | Compile findings into v4.4 candidate doc | 1h |
| 4E.2 | Final 3-oracle critique on draft | 25 min |
| 4E.3 | Ship to user for review + Mario brief update | n/a |

## Cron schedule (additions)

| Cadence | New cron |
|---|---|
| `29 */2 * * *` | Deep-dive dispatch every 2h: pick next 4A/4B/4C sub-step from pending, submit via queue |
| (existing 8 keep) | campaign 3h, idle :47, oracle critique 6h, etc. |

## Subagent strategy

- **Subagent 1**: Slide re-audit (oracle 3-way query) — non-compute
- **Subagent 2**: Architectural envelope GPU sweeps — heavy GPU
- **Subagent 3**: Application benchmark batch (DS-N6 + DS-N3) — mixed
- **Job queue**: all parameterized sweeps go through queue (8 workers
  pick them up autonomously across 3 nodes)

## NO-CHEAT (carried forward)

- Pre-register all gates in 01_LOG.md before compute
- Full heatmaps reported (not just best cells)
- n ≥ 4 seeds for fast iteration; n = 10 for any v4.4 headline
- Oracle critique mandatory at each phase close
- Negative results logged honestly with diagnostics
