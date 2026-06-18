# CAMPAIGN_FULL_PUSH v2 — 2026-05-18 → 2026-06-01 (14 days)

**Supersedes:** `CAMPAIGN_FULL_PUSH_2026-05-18.md` (7-day v1). v1 pillars partially landed: V7 KILLED by PVT (0/72), F1-F3 NEUTRAL, BSIMSOI body-physics REJECTED, F4-PNP KILLSHOT, CHANNEL_ROOT shows BSIM4 channel params cannot move triode ≥1 dec. F3-HIGH 2.907 dec win flagged by O87 as Vbs-modulation band-aid. **Real DC gap = 4 dec on full grid, 8 dec at worst single bias (VG1=0.6, Vd=0.05V: 232 nA silicon vs 1e-15 A model).** Silicon Id is *flat 250 nA across triode* — strong signal for a **missing parallel conduction path**, not a parameter knob.

Strategy bifurcates: Pillar A hunts the missing path; Pillar B prepares an honest scoped paper if A fails by Day 10; Pillar C runs continuously on zgx (FHN/SNN at scale) regardless of A/B outcome.

## 0. Ground truth (verify Day 0 each machine)
- `nsram/bsim4_port/transient_real_v2.py` y=(V_B,q_F,q_R) len=3. FHN-trap lives ONLY in `mode_atlas.py`.
- `nsram/bsim4_port/nsram_cell_2T.py:2122` z474b IFT fix confirmed.
- `scripts/queue/{worker.sh,submit_job.py,status.py}` + `research_plan/job_queue/{pending,running,done,failed}` — extend, don't replace.
- zgx PYTHONPATH must be explicit (old ~/nsram_queue_sandbox is stale).
- F123/CHANNEL_ROOT/BSIMSOI/O87 = canon.

## Pillar A — Missing parallel conduction path (Days 1–10)
Working hypothesis: 250 nA flat-triode floor = topology gap, not param. Candidates: (i) well-tap diode through Rwell, (ii) STI sidewall/corner leak, (iii) Schottky body-contact, (iv) GISL (gate-induced source leakage), (v) topology mismatch (Sebas nmos4 vs 2T_simple.asc).

### A1. Vbs-clamp falsifier (O87 unanimous) — Day 1 daedalus
Add ideal Vbs clamp = Vbs_baseline_F3OFF. Re-run F3-HIGH jtss∈{100, 1e3, 1e4} body. **LOCKED gate**: band-aid if ≥70% lift vanishes (median ≥ 3.70 AND knee slope-MSE ≥ 7.5); physics if ≥0.8 dec persists AND knee MSE drops ≥30%. ~3 GPU-h. Strike F3-HIGH if band-aid.

### A2. Constant 250 nA parallel-path test — Day 1-2 ikaros
Patch `_residuals`: I_par = floor (lumped 250 nA, biasable as I0·exp((Vd-Vd0)/n_par·Vt)). Sweep I_par ∈ {0, 50, 100, 250, 500, 1000} nA × n_par ∈ {1, 1.5, 2, 3}. Gate: Δ-median ≥1.0 dec on FULL grid + triode RMSE −2 dec → parallel path real. ~4 GPU-h.

### A3. Vsint solver instrumentation — Day 2 ikaros
solver_trace flag, log Vsint trajectory + sub-currents per (VG1,VG2,Vd) for 3 worst biases. Build solver_atlas.html. ~1 GPU-h. Feeds A4 priors.

### A4. Bayesian HMC over BSIM4 + topology params — Day 3-6 daedalus
NumPyro/blackjax + diff pyport. Joint posterior over U0/VSAT/RDSW/PCLM/A0/ETA0/ETAB/JTSS/JTSD/AGIDL/BGIDL/snap_Bf/snap_n_avl/snap_R_body + **I_par_floor, n_par, Rwell_lumped**. Region-weighted likelihood. 4 chains × 2000 warmup × 4000 samples ~18 GPU-h. **Gate**: ESS ≥200 per param. Identifies parallel-path mathematically if I_par_floor ≈ 250 nA.

### A5. Hypothesis discrimination — Day 6-7 ikaros+daedalus
H_well, H_STI, H_GISL as MoE terms. Falsifiers: H_well temp shift n=1.05; H_STI Vd-independent at fixed VG1; H_GISL scales with V_GS-Vd. HMC posterior weights name winner. ~6 GPU-h.

### A6. Sebas data request — Day 1
Email: thick-ox 33-bias I-V, 7-rate transient {10ns-10µs}, dense VG1=0.6/VG2=0/Vd∈[0,0.5V] for 250nA flat region, topology disclosure. Mario CC. Don't block.

### A7. Pillar A acceptance — Day 10
- WIN: triode −2 dec AND HMC posterior identifies topology param AND H winner Bayes factor ≥10.
- PARTIAL: ≥1 dec on VG1=0.6 → Pillar B.
- FAIL: triode drop <0.3 dec → Pillar B + Pillar F Verilog-A + methods-only.

## Pillar B — Honest scoping (Days 5–14, parallel with A)
### B1. VG1=0.6 primary-bias only audit
5 Vd × Mario operating window. Target: median ≤1 dec there.

### B2. ML emulator on Mario data — Days 7-10 daedalus
XGBoost + LightGBM ensemble (VG1,VG2,Vd,T) → log10|Id|. 60/40 stratified split, 5-fold CV. Ship predictor.pkl + Python wrapper. **Gate**: test median dec ≤0.5.

### B3. Methods-only paper — Days 8-14 zgx (drafting)
IEDM/DRC: diff IFT pyport (z474b), Bayesian HMC stack, MoE path discrimination, NaN-counted audit harness. Story: "honest negative-result audit framework for emerging 2T devices." 1500-word outline by Day 10, draft Day 14.

## Pillar C — Large-scale GPU simulation (Days 1–14 continuous, zgx)
### C1. N=1024 → N=65536 — Days 1-4 zgx
Profile memory + occupancy. Scaling curves, Lyapunov spectrum ensemble.

### C2. Canonical FHN vs BSIM4 A/B — Days 4-7 zgx
Same connectivity, two RHS engines. Lyapunov + PLV + NARMA-30 + edge-of-chaos location. **Gate**: ΔLyap ≤10%, ΔPLV ≤0.05, ΔNARMA ≤5% → BSIM4 ≡ FHN at network scale (methods claim).

### C3. Spiking Transformer with BSIM4 nonlinearity — Days 6-11 zgx
4-layer transformer, GELU→BSIM4 cell (table-lookup fwd, IFT bwd). Task: Spiking Heidelberg Digits (UNTOUCHED dataset). 10 seeds no-early-stop. **Gate**: final-epoch ≥65% (LSTM baseline ~72%).

### C4. Stochastic resonance demo — Days 8-12 zgx
N=4096 cells + SDE intrinsic noise. Sub-threshold sine + noise sweep, MI peaks at finite noise = SR signature.

### C5. TTFS encoding on SHD — Days 10-14 zgx
TTFS + BSIM4 single-layer readout vs rate-coded baseline. Event-count × E_op proxy energy.

## Pillar D — Continuous-worker architecture (Days 1-3 setup, 24/7 after)
- D1. Cron 10-min poll each machine, --idle-budget + --thermal-guardian flags
- D2. Thermal guardian: ikaros k10temp <80°C, zgx/daedalus GPU <85°C. Throttle (sleep 60s) not kill
- D3. Stale-job auto-recovery: pending older than 4×expected_runtime → re-queue once, then failed/
- D4. Morning brief: research_plan/morning_briefs/YYYY-MM-DD.md at 06:00 local
- D5. Voice-server dashboard: read_job_queue_status(), read_morning_brief(), read_latest_killshot()

## Pillar E — Subagent + oracle cadence
- E1. Daily 3-way oracle 09:00 local; skip-if-recent (O81/O82 lesson, 8h overlap)
- E2. Weekly synthesis subagent (Sundays) — research_plan/daily_synth/SYNTH_W{N}.md
- E3. Per-experiment subagents w/ pre-registered gates inline + mandatory KILLSHOT.md write-on-fail

## Pillar F — Mario/Seb deliverables (Days 7-14)
- F1. C3_tapeout_recommendation_v3.md (post-A4 posterior, 95% CIs)
- F2. Manufacturable cell card + yield bands
- F3. Sensitivity ranking: V6 reset / V8 LIF / noise / DC per param
- F4. Verilog-A export nsram_cell_2T.va (validated against pyport, tol 1e-6)
- F5. mario_onepager_2026-06-01.md

## Pillar G — Gantt (zgx GB10 / daedalus gfx1151 / ikaros local)

| Day | zgx | daedalus | ikaros |
|-----|-----|----------|--------|
| 1 | C1 N-scaling | A1 Vbs-clamp | A2 I_par sweep |
| 2 | C1 cont | A1 jtss1e3+ | A3 Vsint instrument |
| 3 | C1 finish + setup C2 | A4 HMC warmup | A2 + D2 thermal guard |
| 4 | C2 FHN-vs-BSIM4 | A4 HMC chains | E1 oracle dispatch infra |
| 5 | C2 + C3 SHD start | A4 chains | B1 VG1=0.6 audit |
| 6 | C3 transformer | A5 H_well/STI/GISL | A5 falsifiers cont |
| 7 | C3 + B3 outline | A4 finish + B2 XGB | B1 finish, F3 sensitivity |
| 8 | C3 + C4 SR setup | B2 XGB train | F1 tapeout v3 draft |
| 9 | C3 finish + C4 SR | A gate prep | F2 cell card |
| 10 | C5 TTFS setup | **Pillar A GATE** | F4 VA export start |
| 11 | C5 run | B2 finish + B3 | F4 + F5 onepager |
| 12 | C5 finish + B3 | B3 figures | F4 validation + voice-dash |
| 13 | B3 draft complete | B3 finish | Weekly synthesis E2 |
| 14 | Final sync | Final sync | v4.7 brief commit |

### Conditional branches
- A2 wins by Day 3 → expand C: scale to N=262144, add N-MNIST deep-SNN
- A4 posterior identifies topology param by Day 7 → push F1/F2/F4 to Day 8
- A fails by Day 10 → kill A4, redirect daedalus to B2/B3, journal-length manuscript, Verilog-A as deliverable

### Pre-registered killshots
- A1: ≥70% F3-HIGH vanishes → strike from brief
- A2: Δtriode <0.3 dec → parallel path wrong topology
- A4: ESS<200 after 4000 samples → reparametrize once; second fail → variational
- C2: ΔLyap >50% → publishable BSIM4-non-FHN finding (branch)
- C3: SHD acc <50% → SNN claim killed
- B2: XGB test dec >1.0 → emulator not deployable

### Day 14 outputs
- v4.7 brief (Overleaf)
- predictor.pkl (B2) + nsram_cell_2T.va (F4) + C3_tapeout_recommendation_v3.md + mario_onepager_2026-06-01.md
- Methods paper draft (B3) + SHD demo notebook (C3/C5)
- 14 morning briefs + 2 weekly synth + 14 daily oracle critiques

## Acceptance vs current dispatched waves
Today's 6 dispatches (Plan, O88, Vbs+I_par, HMC, large-scale GPU, Vsint) map directly to A1+A2+A3+A4+C1. Continue executing — Plan v2 is the umbrella under which they execute.
