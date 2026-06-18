# CAMPAIGN_FULL_PUSH 2026-05-18 → 2026-05-25

Source: Plan agent ac6480b5e5a29044c (read-only), 2026-05-18.

## Executive Summary

**Ground state (brutally honest):**
- DC fit ~1.0–1.4 dec median across 33 biases. V3 sharp 1/V knee at VG1=0.4 is unmodeled — physics gap, not hyperparameter.
- V7 420 ns Hopf is knife-edge — oscillation only exists because mode-atlas wrapped `transient_real_v2.py` with an external 4th FHN state. Mainline solver cannot reproduce. Card-locked replication misses period by 38% (578 ns).
- M4 1/f noise in mode-atlas is BDF solver residual, not shot-noise physics — needs SDE patch.
- Reservoir, HDC, hier-MNIST, stoch-RNG all dead under iso-precision peripheral-aware audit.
- EP-FIX awaits DS-1 (zgx). v4.6 on Overleaf `a1e5d0c`; V7 description overclaims robustness.

**6 pillars, 3-machine continuous workers, 7 days.**

## 0. Ground-truth check (DO THIS FIRST)

- `nsram/bsim4_port/transient_real_v2.py` — confirmed `TransientCfgV2` has NO `enable_trap`/`tau_slow`/`k_n` fields. State vec y=(V_B,q_F,q_R), len=3. FHN trap lives ONLY in `results/Mode_Atlas_USP1/mode_atlas.py::_integrate_with_trap`.
- `nsram/bsim4_port/nsram_cell_2T.py:2122` z474b IFT sign fix `Vsint = Vsint_d + delta_s` confirmed.
- `scripts/queue/{worker.sh,submit_job.py,status.py}` + `research_plan/job_queue/{pending,running,done,failed}` already exist. Extend, don't replace.
- DS-1 on zgx ETA ~22:30 lokalt. Consume result, don't re-launch.

## 1. Pillar A — DC gap closure (1.4 → 0.5 dec)

### A1. Per-bias residual decomposition (highest ROI)
- Split predicted Id into {Ids_M1, Ic_Q1, Ic_Q2, Ic_lat, Ic_avalanche, Igidl_M1, Ibd_M1, Ie_vert, I_snap_d} via the existing `comp` dict returned by `_residuals`. Residual contribution map per region.
- Expected reduction: **0.3–0.5 dec** by reweighting/re-extracting dominant offender params.
- Compute: ~2 GPU-h zgx.
- Kill: if no sub-term carries >40% of error in any region → cannot localize analytically.
- Owner: zgx.

### A2. Bayesian HMC on BSIM params, region-weighted likelihood
- NumPyro/blackjax with diff pyport. 4 regions (subthreshold/triode/sat/snap+BJT) with separate σ_r. Sample 30–40 BSIM4 params jointly. Posterior gives credible intervals + identifiability map.
- Expected: 0.2–0.4 dec; exposes which params are unidentifiable.
- Compute: ~12 GPU-h daedalus.
- Kill: ESS<100 for any param.
- Owner: daedalus.

### A3. Physics enrichment
- Audit `compute_dc`/`compute_iimpact` for: Cjdb/Cjsb reverse-bias diodes, GIDL exponential constants, DIBL coeff extraction.
- Expected: 0.1–0.3 dec (mostly reviewer insurance).
- Compute: ~6 GPU-h ikaros.
- Kill: <0.05 dec reduction → drop, document "parameter-limited not topology-limited".

### A4. V3-knee–targeted retune (see Pillar B)

### A5. Sebas data blocker (FLAG, don't block)
- Email Day-0: request thick-ox 33-bias I-V + 7-rate transient. If received → re-fit; else label <0.5 dec as data-blocked, soft target 0.7 dec without Sebas.

### Acceptance
- median dec ≤0.7 (existing data): REAL WIN
- ≤0.5: requires Sebas
- V3 knee VG1=0.4 PASS regardless of overall dec: WIN

## 2. Pillar B — V3 DC-knee hypotheses (5 falsifiable)

The sharp 1/V at VG1=0.4 is BJT-dominated (M1 off, parasitic NPN takes over).

- **H1** Parasitic NPN BE forward at body Vb knee. Falsifier: plot d(log Ic_Q1)/dVbe at VG1=0.4; slope ≠ 1/Vt → retune snap_Bf, snap_n_avl. 2 GPU-h.
- **H2** GIDL at high Vd dominates leak. Falsifier: zero Igidl in residual map; if knee err halves → cause. 1 GPU-h.
- **H3** DIBL kink from short-channel. Falsifier: extract Vth(Vd) at VG1=0.4; Δ slope >20 mV/V → refit Eta0/Etab. 2 GPU-h.
- **H4** Impact-ion saturation cap too early. Falsifier: remove cap; if knee sharpens → introduce smooth saturation. 2 GPU-h.
- **H5** snap_R_body bias-dependent. Falsifier: R_body(Vb) = R0/(1+α·Vb²); fit α; α>0 sig → knee improves. 3 GPU-h.

All 5 parallel ≈ 10 GPU-h daedalus+ikaros. Kill all → "data-limited", ship with FAIL caveat.

## 3. Pillar C — V7 hardening

### C1. Land FHN trap into mainline `transient_real_v2.py`
- Add to `TransientCfgV2`: `enable_trap: bool=False`, `tau_slow=800e-9`, `k_n=1e-4`, `V_n0=0.5`, `alpha_n=1.0`.
- Extend `_build_rhs`: optional 4th state n: `dn/dt = (alpha_n·(V_B - V_n0) - n)/tau_slow`. Add `-k_n·n/C_eff` to dV_B/dt.
- CRITICAL: `enable_trap=False` → y stays length 3, bit-identical to current behavior (torture-test against 50 saved transients).

### C2. PVT-grid 4D experiment
- Vth_M1 ∈ {-30,-10,0,+10,+30} mV (5)
- Vth_M2 ∈ {-30,0,+30} mV (3)
- τ_slow ∈ {500,700,800,900,1100} ns (5)
- k_n ∈ {0.5,1.0,2.0,3.0}×1e-4 (4)
- Corner×T ∈ {TT/27, SS/0, FF/85} (3)
- Total 900 points × 5µs ≈ 2.5 GPU-h batched.
- Pass: **≥50% grid** period ∈ [350,500]ns AND Id_pk within 5% Mario AND V_b ⊂ [-1, +1.5]V.
- Stretch: ≥70% with CV ≤5%.

### C3. SDE patch for true M4 noise
- `compute_shot_noise(comp, dt)` → √(2q|Iii|/C_eff) per-step kick into dV_B.
- Accept: σ(V_b) at rest within 30% of analytical √(q·Iii·dt/C²).
- 1 GPU-h.

### C4. dt-halving + alternate integrator
- z477c-locked point, dt halved + Radau alt method. Period should match within 1%.
- If not: Hopf is numerical artefact → **KILLSHOT**.
- 30 GPU-min ikaros.

## 4. Pillar D — Large-scale apps brutal triage

| Candidate | Verdict | Why |
|---|---|---|
| Adaptive control / motor pattern | REJECT | 420ns slow vs op-amp+counter |
| Industrial sensor anomaly (real data) | CONDITIONAL | Only if intrinsic noise IS the discriminative feature |
| Sparse coding / dictionary | REJECT | mismatch destroys without DAC trim |
| **Continual edge learning, V_b eligibility** | PURSUE | redesigned z483; if Permuted-MNIST 10-task beats LIF +3pp = USP |
| Time-series compression | REJECT | int8 linear-PCA wins |
| **Adaptive online TRNG, peripheral-honest** | PURSUE | only if ≤30 pJ/bit INCL DAC/ADC/LFSR |

**D1 TRNG redesign** (12 GPU-h zgx): V_b → 1-bit comparator @ 1MHz → 32-bit LFSR → NIST 800-22. Pass: NIST + ≤30 pJ/bit + ≤0.5 cells/bit.

**D2 Continual eligibility** (16 GPU-h zgx): Permuted-MNIST 10-task, V_b as 100µs trace gating LR. Baseline: LIF + EMA eligibility. Pass: ≥baseline +3pp AND task-1 final ≥80%.

## 5. Pillar E — 12 new evaluation metrics

1. Subthreshold-slope MSE (≤5 (mV/dec)²)
2. DIBL coeff err (≤10 mV/V)
3. Vth-extraction RMS (≤5 mV)
4. Snap-trigger V_d* err (≤50 mV)
5. Snap-current at V_b=0.4V (within 2% Mario)
6. Body-τ uncertainty (CI/τ ≤10%)
7. Per-region weighted dec (4 numbers, all ≤0.7)
8. Backward-sweep hysteresis match (≤20 mV)
9. BJT-only RMSE at VG1<0.3 (≤0.5 dec)
10. Slope d(log I)/dV RMSE (≤0.3 V⁻¹)
11. Per-decade error hist (90th ≤1.0)
12. Bootstrap 95% CI on median dec (width ≤0.3)

Build as `nsram/eval/metrics_v2.py`. All 12 in one pass, <1 GPU-h.

## 6. Pillar F — Continuous worker arch (extend existing)

### F1 Job spec
```json
{ "id":"...", "host_pref":"zgx|daedalus|ikaros|any", "rocm_or_cuda":"rocm|cuda|any",
  "cmd":"...", "thermal_max_apu_c":80, "thermal_max_gpu_c":90,
  "burst_max_s":6, "expected_runtime_s":..., "heartbeat_interval_s":60,
  "killshot_check":"<bash one-liner>" }
```

### F2 Worker daemon
- Pick highest-priority match → atomic mv to `running/`.
- Heartbeat every interval. On exit → `done/` or `failed/` + exit_code.txt.
- Stale heartbeat >5×interval → re-orphan to `pending/`, cap 3 retries.

### F3 Thermal guardian sidecar
- APU thermal_zone0 + nvidia-smi/rocm-smi every 5s.
- APU>80°C OR GPU hotspot>90°C → SIGSTOP children, write `thermal_pause` flag.
- Resume <75°C (10°C hysteresis).
- ikaros: enforce `burst_max_s` ≤6s for any kernel invocation.

### F4 Result aggregator
- Cron-free self-resubmitting job. Every 30 min: scan `done/` → write `research_plan/morning_briefs/status_<ts>.md`.

### F5 Killshot watcher
- Each job carries `killshot_check`. Aggregator runs after `done/`. Pass → write `KILLSHOTS_TRIGGERED.md` + alert.

## 7. 7-Day Gantt

| Day | zgx (CUDA) | daedalus (ROCm) | ikaros (orchestration + bursts) |
|---|---|---|---|
| D1 (today) | Wait DS-1 finish | A1 residual decomp 33 biases | F1-F5 worker infra; Sebas email |
| D2 | D1 TRNG (12h) | A2 HMC launch (12h) | C1 FHN trap into mainline + C4 dt-halve |
| D3 | D2 continual (16h) | C2 PVT-grid (2.5h) + A3 enrichment (6h) | B1-B5 V3-knee falsifiers (10h batched) |
| D4 | DS-1 triage; if PASS → A4 retune | A2 posterior triage; C3 SDE noise validation | Pillar E metrics_v2 retro-run |
| D5 | Re-EP-FIX with new metrics | A4 V3-knee full sweep (8h) | Aggregator: D1-4 synthesis, v4.7 changelog |
| D6 | Spare / worst-seed re-run | V7 robustness 2D contour | Killshot watcher + pre-mortem |
| D7 | 16×16 mismatch surrogate if D6 PASS | Final PVT confirmation | CAMPAIGN_FULL_PUSH_FINAL_2026-05-25.md |

### Headlights
- zgx: DS-1 final ≥88%? gates all else
- daedalus: A2 HMC ESS ≥100? AND C2 PVT ≥50%?
- ikaros: C1 bit-identical when trap-off? F1-F5 stable 24h zero thermal trips?

## 8. Pre-registered killshots

| Pillar | Killshot |
|---|---|
| A (DC) | median dec >1.0 after A1+A2+A3+A4 → data-limited, shelve until Sebas |
| B (V3 knee) | All 5 H falsified >5pp → BSIM4 structural limit, document |
| C (V7 PVT) | <30% pass → V7 single-point curiosity, remove from USP-1 |
| C (V7 numerical) | C4 dt-halved period delta >5% → full retraction |
| C (M4 noise) | σ(V_b) outside [30%, 100%] analytical shot bound |
| D1 TRNG | >30 pJ/bit OR NIST fail |
| D2 Continual | ≤LIF +1pp |
| E metrics | New ≡ old dec within noise → keep only top-3 |
| F workers | >2 thermal trips ikaros in 24h → reassign all heavy off ikaros |

## 9. What we DON'T know

- V7 Hopf existence on real silicon (all model + wrapper so far)
- BSIM4 param-degeneracy vs genuine topology incompleteness
- DS-1 verdict
- ANY app beats int8 digital iso-precision peripheral-honest at 130nm — 5 prior killshots say NO
- Sebas delivery within 7d → without it <0.5 dec unlikely

## 10. Critical Files

- `nsram/bsim4_port/transient_real_v2.py` — Pillar C1
- `nsram/bsim4_port/nsram_cell_2T.py` — Pillar A1/B, line 2122 IFT fix
- `scripts/queue/worker.sh` — Pillar F (extend heartbeat + thermal hook)
- `scripts/queue/submit_job.py` — Pillar F (extend job spec)
- `results/Mode_Atlas_USP1/mode_atlas.py::_integrate_with_trap` — Pillar C ref impl
