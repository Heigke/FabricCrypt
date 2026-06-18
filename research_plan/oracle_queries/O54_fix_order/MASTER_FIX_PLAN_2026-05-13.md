# MASTER FIX PLAN — 2026-05-13

## Goal
Close 4 HOLD items + 4 NEXT items identified in the explainer film:

**HOLD** (must fix before v4.4 ship):
1. Cell-wide DC fit: 0.99 → < 0.5 dec
2. Snapback shape gap (V_d > 2V wrong)
3. KWS at chance (8.3% / 12-class) — primary ship-blocker per Oracle 4D
4. Per-V_G1 branch incompatibility (Rs split 9 orders)

**NEXT** (enablers):
5. Implement N1 multi-τ trap reservoir in pyport (z311 proved mechanism)
6. Implement VNwell→VB diode with CORRECT polarity (z310 had backwards)
7. Implement SRH gen-rec body depletion (T4 N2 candidate)
8. Implement BSIM rbodymod=1 distributed body R (T4 N3 — needs code, not flag)

## Pre-registered Gates (locked)

| Phase | Track | Gate (PASS) | Gate (AMBITIOUS) |
|---|---|---|---|
| P1 | Oracle 3-way fix-order | ≥2/3 agreement on top-3 prio | Single-vote unanimous |
| P2 | Materials re-scan | ≥3 new unused signals found | ≥1 actionable physics constraint |
| P3 | pyport_v4 traps+diode | DC < 0.7 dec cell-wide | DC < 0.5 dec |
| P4 | KWS attack | Acc > 25% (3× chance) | Acc > 50% |
| P5 | Combined model + transient | hyst within 3× measured | within 1.5× measured |
| P6 | Network sim re-run on v4 | match z312 84.09% | beat 85% |
| P7 | Oracle critique cycle | 0 fragility flag ≥2/3 | unanimous SHIP |

## Phase plan

### P1 — Oracle fix-order (NOW, ~25 min) [research]
- 3-way oracle dispatch (gpt-5+gemini+grok)
- Question: given today's complete diagnosis (multi-τ traps confirmed, polarity bug, V_d>2V data, per-branch Rs split), what is the OPTIMAL fix-order to maximize v4.4 ship probability?
- Output: research_plan/P1_oracle_fix_order.md

### P2 — Materials re-scan with fresh eyes (NOW, ~60 min) [research]
- Subagent walks ALL Sebas + Mario materials AGAIN with today's diagnoses in hand
- Focus: traps mention, N-well doping profile, V_d>2V hint, rbodymod hint, transient ramp rates
- Output: research_plan/P2_materials_rescan.md

### P3 — pyport_v4 build (after P1/P2, ~3h) [code+compute, ikaros GPU]
- N1 multi-τ trap reservoir (10 τ values µs→s, fitted Q_max)
- VNwell→VB diode w/ CORRECT polarity (anode=Vb, cathode=VN)
- Drain-end avalanche M(V_bc) coupled to channel
- Initial param sweep on Sebas 33 IV
- Output: results/z313_pyport_v4/summary.json

### P4 — KWS encoding attack (after P1, ~2h) [code+compute, zgx GPU]
- Try: rank-coded MFCC, temporal-convolution input, neural-engine handoff
- Goal: lift NS-RAM SNN above chance (8.3% → 25%+)
- Output: results/z314_kws_attack/summary.json

### P5 — Combined v4 + transient validation (after P3, ~2h) [code+compute, daedalus]
- Run z308-style transient harness with pyport_v4
- Multi-rate (0.017, 0.17, 1.7 V/s) predictions
- Compare hysteresis to measured 2.6e-3
- Output: results/z315_v4_transient/summary.json

### P6 — Network sim re-run on v4 (after P5, ~1h) [zgx GPU]
- Re-run HDC headline + Bayesian RNG on pyport_v4 surrogate
- Check that model improvement doesn't BREAK network results
- Output: results/z316_v4_networks/summary.json

### P7 — Oracle critique cycle (after P6, ~25 min) [research]
- 3-way oracle critique on full v4 stack
- If 0 fragility flag ≥2/3 → v4.4 SHIP ready

### P8 — v4.4 brief compile (final, ~1h) [code]
- Write research_plan/4E_v4.4_brief.md
- Include: 84% HDC, RNG, v4 model, honest gaps

## Cron schedule additions

- `0 */4 * * *` — Progress check on this campaign (every 4h)
- Existing cron jobs continue: hourly idle, daily synth, etc.

## NO-CHEAT discipline

- Every gate locked BEFORE its compute starts
- Full heatmaps reported (no cherry-pick)
- ≥ 4 seeds per network sim, ≥10 for v4.4 headline
- Oracle critique mandatory after P3 (model) and P6 (networks)
- Honest FAIL allowed; cheating disallowed
- "WARNING: corrective pre-register needed" if ≥2/3 oracles flag drift
- All P1-P8 outputs persisted to disk (no purely-in-memory results)

## Resource allocation

- **ikaros (gfx1151 APU+GPU)**: P3 model build + heavy DC sweeps. Thermal governor active.
- **daedalus (AMD CPU)**: P5 transient (multi-core CPU-friendly). Governor active.
- **zgx (NVIDIA GB10)**: P4 KWS + P6 networks (separate thermal, full throttle ok).
- All 3 nodes share queue for any submitted job.

## Risk register

1. P3 trap implementation may break Newton convergence → fall back to explicit time-step
2. KWS may stay at chance regardless of encoding → declare fundamental ship-blocker
3. pyport_v4 may help DC but break transient (over-constrained) → bisect changes
4. Cluster overheating if all 3 nodes peak together → governor handles + queue limits

## Success criteria for v4.4

Headline candidates (pick one, with full disclosure):
- HDC 84.09% UCI-HAR @ 35 nJ/inf (N=16384, σ=0.05)
- Bayesian NS-RAM RNG (ESS 1.03× + NIST 5/5)
- pyport_v4 DC < 0.5 dec (if P3 PASS)

Brief MUST include:
- Application matrix from 4A
- Pyport_v4 fit quality with honest per-branch breakdown
- Snapback gap closure (or open status)
- KWS attempt outcome (PASS or honest FAIL)
- Next-stage roadmap (what we still cannot do)
