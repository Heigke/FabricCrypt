# Seven Gaps Plan 2026-05-15 (14:15 → 22:00)

Closing all 7 physical-correctness gaps systematically with verification per step.

## Track 1 — SNAPBACK FOLD (Vd > 1V)

### S1: Body-strap diagnostic (30 min, ikaros)
- **Goal**: Determine if KILL-SHOT was solver issue or physics issue
- **Method**: Force `Vb = const` (sweep 0.0 to 0.8V in 0.1V steps), measure Ids(Vd) response
- **Verify**: If Ids shows the 2-3 dec jump when Vb ≥ 0.55V → physics IS in model, regenerative loop just doesn't close. If still flat → BSIM4 Ids(Vbs) is fundamentally wrong.
- **Outcome decides S2 vs S4**

### S2: Continuation/homotopy solver (60 min, ikaros, only if S1 says solver issue)
- **Goal**: Find both basins of bistability — low-Vb (sub-threshold) and high-Vb (snapped)
- **Method**: Newton continuation from low-Vd, λ-parameter on iii_gain (ramped 0→2)
- **Verify**: At Vd=1.5V, VG1=0.6, solver finds Vb≥0.55 branch with Ids matching measurement ±0.5 dec

### S3: Literature dive (30 min, subagent + WebSearch)
- **Goal**: Find Mario Lanza 2024-2025 papers explicitly modeling 2T NS-RAM snapback
- **Search**: "NS-RAM 2T snapback fold floating body Lanza", "2T DRAM-like snapback simulation"
- **Verify**: ≥1 paper with reproducible topology different from what we tried

### S4: Empirical fold model (90 min, ikaros, fallback if S2 fails)
- **Goal**: Add explicit `Ids *= (1 + ΔI*sigmoid((Vd - V_kink(VG1))/Vsharp))` with learned params
- **Method**: Fit V_kink, ΔI, Vsharp per VG1 against measured fold positions
- **Verify**: Cell-wide < 0.6 dec at all 3 VG1 branches AND jump within ±0.3 dec of measured

## Track 2 — TRANSIENT VALIDATION

### T1: Data audit (15 min, subagent)
- **Goal**: Find ALL time-dimension data we have
- **Search**: `data/sebas_*`, `docs/Zoom/`, `nsram/proposal_2026_05/data/`, any *.csv with 3+ columns or 't'/'time' header
- **Specifically**: oscilloscope screenshots in Zoom (.png/.jpg with traces), transient .raw files, pulse-train data
- **Verify**: List of all files found with shape, time-axis range, signal type

### T2: Fit Cb and τ_relax if transient data exists (60 min, ikaros)
- **Goal**: Calibrate transient parameters against measurement
- **Method**: For each pulse: fit Cb*(dVb/dt) = Σ Ibranches integrated → extract Cb, τ_relax
- **Verify**: Predicted Vb(t) within ±10% of measured at 3 time-points per pulse

### T3: Document and submit if T1 finds zip (10 min, log only)
- If no transient data → fix #128 task (request from Sebas), log gap formally
- Mark transient validation as **HARD-BLOCKED** until data arrives

## Track 3 — INPUT-COUPLING TUNING

### I1: VG2 sensitivity sweep (45 min, ikaros)
- **Goal**: Find bias region where Inet ∝ Vd (responsive) not flat (autonomous)
- **Method**: Build LUT response surface, compute |dInet/dVd| / |Inet| at each (VG1, VG2, Vb)
- **Verify**: Find region where sensitivity > 5% over Vd ∈ [0.1, 1.0V]

### I2: Rerun DS-N10 sine class in responsive regime (60 min, daedalus GPU)
- **Goal**: Confirm input-driven (not autonomous) behavior in tuned bias
- **Method**: Same script as DS-N10 but VG2 from I1 best point
- **Verify**: Spike-rate variance > 50× across input frequencies (was ~7Hz before)

### I3: Bayesian RNG retest in responsive regime (30 min, daedalus)
- DS-N15 was Gaussian-noise based — re-test with substrate spikes as RNG seed
- **Verify**: KL divergence vs target distribution improves

## Track 4 — TEMPERATURE INTEGRATION

### TM1: Wire BSIM4 thermal coefficients (30 min, ikaros)
- **Goal**: Apply existing kt1/ute/ua1/ub1/uc1 coefficients to forward_2t
- **Method**: Find currently-zero T-coefs in pyport, hook into Vth(T) and μ(T) formulas
- **Verify**: At T=25°C result identical to current; at T=85°C, Vth shifts by expected -25mV

### TM2: T-sweep on 33-curve fit (60 min, ikaros)
- **Goal**: Quantify how cell-wide fit degrades with temperature
- **Method**: Run z372 (snapback demo) at T ∈ {25, 50, 75, 85, 100}
- **Verify**: Trend monotonic, no nan, document delta_RMSE/delta_T

### TM3: Application robustness (30 min, daedalus)
- Re-run DS-N10 sine class at 85°C — does it still work?
- **Verify**: Acc drops by <5pp at 85°C → robust; >10pp → fragile

## Track 5 — INTRINSIC NOISE

### N1: 1/f noise injection model (45 min, ikaros)
- Add `Vth_noise = sqrt(K_f / (Cox * W * L * f)) * randn()` to per-cell Vth at runtime
- Default K_f from BSIM4 NOIMOD; sweep K_f magnitude ∈ {0.1×, 1×, 10×}
- **Verify**: PSD of Vb(t) shows 1/f^α with α ∈ [0.8, 1.2]

### N2: RTN (random telegraph noise) (30 min, ikaros)
- Add 2-state telegraph: Vth toggles ±ΔVth_RTN with rate τ_RTN
- **Verify**: ACF of Vb(t) shows exponential decay with extracted τ

### N3: DS-N15 retest (30 min, daedalus)
- Re-run Bayesian RNG with realistic 1/f + RTN noise instead of Gaussian
- **Verify**: Sample variance and decorrelation time match published RTN device data

## Track 6 — AGING/DRIFT

### AG1: HCI Vth drift model (45 min, ikaros)
- `ΔVth_HCI = A_HCI * Ids^n * t^m` (Takeda-style)
- Hook to cumulative stress integration in transient sim
- **Verify**: 10^4 s simulated stress produces realistic 5-20mV shift

### AG2: NBTI body-charge decay (30 min, ikaros)
- `Vb_drift(t) = Vb0 * exp(-t/τ_NBTI)` with τ_NBTI sweep {1h, 1day, 1week}
- **Verify**: 1-week retention loss matches typical floating-body decay (~50%)

### AG3: Memory retention claim audit (60 min, daedalus)
- Re-run any "memory" claim (DS-N7c/N9 retracted, but DS-N15 Bayes RNG uses memory)
- **Verify**: Sample at t=0, 1h, 24h, 1w. Document degradation curve.

## Track 7 — WAFER VARIATION

### WV1: Multi-cell data audit (15 min, subagent)
- Search Sebas folders for n_devices > 1 IV-data
- **Verify**: List unique devices and stats sigma_Vth_inter-device

### WV2: Moment-match if WV1 finds it (45 min, ikaros)
- Fit Gaussian to measured Vth0 distribution
- **Verify**: KS-test p > 0.05 for normality

### WV3: Sensitivity sweep if WV1 finds nothing (60 min, daedalus)
- Sweep σ_Vth0 ∈ {10, 25, 50, 100} mV
- Re-run DS-N10 sine class at each — does it stay > 90% acc?
- **Verify**: Document robustness range; identify breaking point

## Oracle critique gate
After Tracks 1-7 complete: build O71 packet with all results, ask:
- Q1: which of 7 gaps is now defensible, which still gaps?
- Q2: smallest test that falsifies the strongest closed claim
- Q3: ranked next-most-valuable single experiment

## Resource allocation
- **ikaros**: Tracks 1, 4, 5, 6 (CPU + GPU model work)
- **daedalus**: Tracks 2 (post-fit), 3, 7 (application reruns)
- **zgx**: idle / available for any overflow
- **Subagents**: 1-3 in parallel, oracles via API

## NO-CHEAT
- All gates pre-registered above
- Honest reports even on FAIL
- Single oracle O71 at end (not per-track to avoid drift)
