# NS-RAM Network-Sim Massive Campaign — 2026-05-17

## Premise
Cell-physics campaign is closing (6/9 → 8/9 with z462b). Now pivot to **massive network-scale validation** across many use cases, topologies, and scales.

**Validation principle**: physics is "good enough" (qualitatively right, ~1.2 dec cell-RMSE). Stop chasing publication-grade DC. Focus on what the model *enables* at network scale.

## Phase V — Validation completion (PREREQUISITE, ~6h)

V0. Wait for z460 PT_VBIC ×10 (in flight)
V1. Wait for z462b PT-backward as default (in flight, then re-run z461 V1)
V2. Wait for O77 oracle (in flight)
V3. Accept whatever scorecard we get (6/9 or 8/9). Move on. No more cell-tuning.

## Phase N1 — Topology zoo (5-10 topologies × 4-6h each)

Architectures to sweep:
1. **Feedforward LIF** (baseline) — N=10/100/1k/10k
2. **Reservoir (ER_SPARSE)** — already done (z165+). Re-validate with PT-backward solver
3. **Recurrent LIF** with BPTT — N=128/512/2k
4. **Hierarchical (2-layer with skip)** — VGG-style
5. **Lateral inhibition WTA** — competitive learning
6. **Predictive coding** (top-down feedback) — DS-N11 extension
7. **STDP Hebbian** — DS-N12 extension
8. **HDC bundling** — DS-N5 family extension
9. **Memory Palace associative** — DS-N7 extension
10. **Cascade (multi-stage)** — DS-N14 extension

## Phase N2 — Use-case battery (parallel per topology)

| # | Use case | Dataset | Topology fit |
|---|---|---|---|
| U1 | KWS streaming | Google Speech Commands | Cascade + LIF |
| U2 | DVS-Gesture | DVS-gesture | Recurrent LIF |
| U3 | MNIST | MNIST 28×28 | Feedforward + Reservoir |
| U4 | ECG anomaly | MIT-BIH | STDP adaptive |
| U5 | UCI-HAR | UCI-HAR | HDC |
| U6 | Anomaly time-series | Numenta NAB | Predictive coding |
| U7 | Mackey-Glass forecast | synthetic | Reservoir |
| U8 | Memory binding | synthetic | Memory Palace |
| U9 | Comm equalizer | synthetic LMS | Recurrent |
| U10 | Stochastic compute | RNG benchmarks | Bayesian sampler |

## Phase N3 — Scale sweep

For each (topology, use-case) pair: scale N = {32, 128, 512, 2k, 8k, 32k, 128k}
- Track: accuracy, throughput (inferences/sec), energy/inference (pJ from cell), latency
- Pareto plot per use-case

## Phase N4 — Visualization & animation suite

Per (topology, use-case, scale):
- Spike-raster heatmap (time × neuron)
- V_B(t) waterfall plot
- Weight-evolution GIF (for trainable cases)
- Energy heat-map across architecture
- Latency violin plot
- Pareto: accuracy vs energy vs throughput

Tool: matplotlib + ffmpeg for GIFs. Single utility `scripts/network_viz.py`.

## Cross-machine work distribution

- **ikaros (gfx1151 ROCm)**: small-medium nets (<10k neurons), interactive
- **daedalus (CPU)**: CPU-friendly LMS/HDC reps with full seed sweeps (RETRY first; SSH was down)
- **zgx (NVIDIA GB10)**: large nets ≥10k neurons, BPTT training, heavy compute
- **Code sync**: rsync ikaros → {daedalus, zgx} at campaign start, again every 6h

## Pre-registered gates (per use-case)

Per use-case will have its own pre-registered gates (e.g. KWS accuracy >80%, MNIST >95%, NAB F1>0.5). Inherit from existing DS-N work where applicable.

## Autonomous cron loops

| Cron | Cadence | Function |
|---|---|---|
| network_tick | every 30 min | Check active sims, log progress, dispatch next |
| network_oracle | every 6h | 3-way oracle critique on accumulated results |
| network_viz | every 2h | Re-render summary visualizations |
| network_synthesis | daily | Roll up accumulated results into NETWORK_RESULTS.md |
| code_sync | every 6h | rsync to zgx + daedalus |

## Stop criteria

- 30 (topology × use-case) cells completed
- OR 5 days elapsed
- OR ≥5 use-cases achieve "publishable" level (>80% target metric)

## KILL_SHOT triggers

- Any topology breaks pyport on >50% scale points → revert + log
- Oracle 3/3 flags entire approach → pause + escalate
- daedalus + zgx both unreachable >24h → run ikaros-only with reduced scale ceiling
