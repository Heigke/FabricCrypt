# NOVEL DS PLAN — 2026-05-12 (post-pause resume)

**Trigger**: User wants resume + push beyond MNIST/NARMA toys. Real
tasks where NS-RAM might STRUCTURALLY win, not just match. Plus better
GPU utilization on sequential modelling work.

---

## How GPUs accelerate SEQUENTIAL modelling tasks

The trick: every "sequential" task has a hidden batch dim. Find it.

| Sequential task | Hidden batch dim | GPU strategy |
|---|---|---|
| Newton-Raphson per point | Batch across grid points | 10K-1M points in parallel |
| ODE time-step transient | Batch across (V_G1, V_G2, t_rise, C_b) tuples | PMP-9 style — 1000+ traj in 1 GPU pass |
| Scipy parameter fit | Batch many parameter vectors via CMA-ES / particle swarm | 100 param vectors × loss eval all parallel |
| Differentiable fit | Forward+backward through full pipeline | torch.optim.LBFGS w/ batched residual |
| Multi-seed run | Seeds → batch dim | 10 seeds in one GPU call |
| Hyperparam sweep | param × seed → 2D batch | already done D1/D2 |
| Newton inner loop | Trapezoidal step-array | scan-batched per-step linear solve |

**Rule**: if pyport's `_solve_at_fixed_vb` accepts a numpy array per
call, the inside of that call is a Newton over (B,) batched scalars
where B is the GPU batch dim.

## Phase A — Resume + close MEP

| # | Task | Status pre-pause | Resume action |
|---|---|---|---|
| A.1 | D2 corrective sweep | ~85/180 cells done | Resume launcher (SKIP logic) |
| A.2 | MEP-6 differentiable Newton | Partial (1.4s/eval, loss=4.9, grid clipping) | New subagent on ikaros |
| A.3 | SURR-V4 100K surrogate | Strategy debate (CPU multiproc was choice) | Run as CPU multiproc 8w on daedalus |
| A.4 | PTP batched transient | Crashed during validation (likely OOM) | Fix batch size + retry on ZGX |

## Phase B — NOVEL DISCOVERY SWEEPS

The user said: NOT just NARMA / MNIST / Mackey-Glass. Real tasks where
NS-RAM might beat GPUs/NPUs structurally.

### NS-RAM's actual structural advantages

- **Body-state τ ≈ 1 ms** matches biological/sensor timescales naturally
- **10⁴× firing range** (slide 18) — high dynamic range
- **6.4–21 fJ per spike** — ~1000× lower than GPU MAC
- **Standard CMOS** — scales without exotic process
- **Built-in stochasticity** from impact ionization
- **Analog memory** via V_b (PMP-2/N4c)
- **Spike-rate-coded** — Mario explicit: "VMM less evident with spikes"

### Six novel tasks, each chosen for structural match

| ID | Task | Why NS-RAM wins | Falsification metric |
|---|---|---|---|
| **DS-N1** | **Keyword spotting on Google Speech Commands** | Streaming audio, sparse activation, sub-mW target = exactly Mario's slide 13.46 band. Strict latency. | NS-RAM 70% top-1 on 12-class @ <100µW projected, beat Cortex-M4 + MFCC by ≥1.5× |
| **DS-N2** | **DVS-Gesture (event-based classification)** | Native spike-coded camera, no rate-to-spike conversion needed. NS-RAM body-state τ matches DVS frame time. | Match SOTA SNN top-1 on DVS128 Gesture (~95% baseline) |
| **DS-N3** | **Bayesian posterior sampling via NS-RAM noise** | Use impact-ionization stochasticity AS the random number generator. Replace pseudo-RNG in MCMC. | Effective sample size per unit time vs CPU/GPU MCMC; aim ≥10× |
| **DS-N4** | **Online STDP adaptive filter (ECG anomaly detection)** | V_G2 as analog weight, body-state for plasticity. Adapt in real-time without backprop. Compare to standard Adam-trained filter. | ROC AUC on MIT-BIH ECG anomaly ≥ standard filter |
| **DS-N5** | **Hyperdimensional computing (HDC) classification** | Fixed HD vectors map cleanly to NS-RAM populations. Bundling is rate-sum, binding is XOR-like. Inherent low-energy. | UCI-HAR top-1 with N=10K HD bits ≥ HDC baseline |
| **DS-N6** | **Time-series anomaly on Numenta NAB** | Streaming, sparse, body-state memory = sliding window. NAB benchmark is the canonical "hard" anomaly suite. | NAB score ≥ HTM-Java baseline (Numenta's own) |

### Why these are "real" not toys

- **KWS Google Speech Commands** is the ACTUAL benchmark for always-on
  audio. Cortex-M4 + MFCC + tiny DNN reach 90% but at ~3 mW. Sub-mW
  is unsolved at this accuracy.
- **DVS Gesture** is a DVS-camera (event-based) dataset published by
  IBM. Strictly easier on event-hardware than frame-based GPUs.
- **MCMC sampling** is a real bottleneck in Bayesian ML. If NS-RAM
  noise replaces pseudo-RNG with native physical noise, it's a
  Quantum-flavored advantage at CMOS cost.
- **ECG anomaly** is regulated medical (FDA pathways exist). Always-on
  body-worn sensors are a real market.
- **HDC** is a thing — NSF/DARPA have funded it. SOTA is mostly
  algorithm not silicon. NS-RAM-native HDC would be novel.
- **NAB** is THE anomaly benchmark in ops/IoT.

## Phase C — Oracle falsification + brief v4.4

After Phase A and B close: 3-oracle critique submitting heatmaps +
falsification results. If any DS-Nx AMBITIOUS passes oracle review,
brief v4.4 opens with a real headline.

## Cron schedule (tighter)

| Cadence | Purpose |
|---|---|
| `21 */3 * * *` | Campaign every 3h (MEP+DS aware) |
| `47 * * * *` | Idle hourly (light, no compute) |
| `41 */6 * * *` | Oracle critique every 6h |
| `33 11,23 * * *` | 12h oracle synth (keep) |
| `11 9,15,21 * * *` | 6h track audit (keep) |
| `43 4 * * *` | Baseline watchdog |
| `13 2 * * *` | Daily synth |
| `7 0 * * *` | Resource audit |
| `23 9 * * 1` | Weekly review |
| `23 6 * * *` | Morning brief |

## Thermal policy (HARDENED post-incident)

1. APU > 85°C → launcher pauses all new launches until APU < 60°C
2. Max 6 concurrent CPU workers ACROSS entire ikaros+daedalus cluster
3. ZGX (NVIDIA GB10) has separate thermal — much higher headroom; can
   run independently
4. Any subagent that spawns >4 python procs simultaneously gets killed
5. Active monitor logs APU every 10s during campaigns
6. If APU > 92°C → kill heaviest python proc via PID

## NO-CHEAT discipline (carried forward)

- Every gate pre-registered before its run
- Full heatmaps reported
- n ≥ 4 seeds for fast iter, n=10 for v4.4 headline
- Oracle critique mandatory after each phase
- INFORMATIVE_NULL logged not buried
- Novel tasks have DOMAIN-SPECIFIC baselines (KWS vs Cortex-M4, DVS
  vs IBM TrueNorth, etc.) — not just internal Poisson baseline
