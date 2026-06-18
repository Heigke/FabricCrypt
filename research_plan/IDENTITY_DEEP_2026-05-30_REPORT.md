# IDENTITY DEEP — Report
Date: 2026-05-30
Hosts: ikaros (AMD gfx1151, card1), daedalus (AMD gfx1151, card0)
Battery: 5 angles built, 4 ran on both machines, 1 (NPU) blocked at userspace.
Analysis: results/IDENTITY_BENCHMARK_2026-05-30/deep/ANALYSIS.json

## Verdict table
| Angle | Built | Ikaros | Daedalus | Headline | 95% CI | Gate |
|-------|-------|--------|----------|----------|--------|------|
| A — Power fingerprint | Y | Y (5 reps×5 s × 4 wl) | Y | IDLE 19.2 W vs 6.1 W; MEDIUM 110.3 vs 60.1; HEAVY 109.5 vs 62.4 | diff CI excludes 0 in IDLE / MEDIUM / HEAVY; Cohen d ≥ 8 | **DISCOVERY** (3 of 4 workloads pass; LIGHT cohen d=4.0 but std overlaps gate threshold) |
| B — Thermal time constant | Y | Y (6 cycles) | Y | τ_heat ikaros = 4.33 s vs daedalus 1.26 s; R_th ikaros 0.311 K/W vs 0.482 K/W | τ_heat diff CI [2.78, 3.53] s; R_th CI [-0.177, -0.165] K/W | **DISCOVERY** (Cohen d=7.7 on τ_heat, d=−30.5 on R_th) |
| C — NPU XDNA | Y (recon) | recon only | recon only | /dev/accel/accel0 + amdxdna loaded, no XRT userspace | n/a | **BLOCKED** |
| D — DPM Vmin sweep | Y | Y (low/auto/high × 60 reps) | Y | **zero** bit flips on either device at any DPM level | timing differs 1.78 ms (ikaros faster), CI [1.34, 2.56] | **AMBIGUOUS** (no Vmin signal; timing differs but reflects DPM scheduler not silicon) |
| E — CPU per-core | Y | Y (16 cores × 4 repeats) | Y | ikaros per-core time spread 2.67 ms (8.85–11.52 ms); daedalus 0.19 ms (8.62–8.81 ms); identical sysfs max-freq 5187 MHz both hosts | mean time diff CI [+1.65, +2.45] ms (ikaros slower); rank-correlation across cores = −0.51 | **DISCOVERY** (Cohen d=3.37 on per-core mean; per-core ranking anti-correlated → distinct silicon orderings) |

## Headline numbers (95 % bootstrap CI)
- **Power IDLE diff (ikaros − daedalus): +13.13 W, CI [+11.32, +14.74]** — daedalus 6 W idle, ikaros 19 W (3.1×).
- **Power MEDIUM diff: +50.18 W, CI [+47.94, +52.54]** — ikaros 110 W vs daedalus 60 W under identical workload.
- **τ_heat diff: +3.08 s, CI [+2.78, +3.53]** — ikaros heats 3.4× slower at the package sensor.
- **R_th diff: −0.171 K/W, CI [−0.177, −0.165]** — daedalus has 55 % higher thermal resistance.
- **CPU per-core time diff: +2.05 ms, CI [+1.65, +2.45]** — ikaros cores ~24 % slower on 384×384×20 workload; per-core ranking r=−0.51 → distinct die orderings.

## C — NPU status (blocked, what is missing)
Both hosts have: amdxdna kernel module loaded; /dev/accel/accel0 char device; PCI 17f0 Signal Processing Controller.
Neither host has: xrt-smi/xrtutil; pyxrt python binding; /opt/xilinx subtree; any compiled .xclbin/.vaie model.
Until AMD's Ryzen-AI-SW deb stack (or RyzenAI-SW source build) is installed, the NPU char device cannot be exercised from userspace; no kernel submission, no inference jitter, no NPU-bound power. Recon JSON: results/IDENTITY_BENCHMARK_2026-05-30/deep/{host}/C_npu.json.

## D — Vmin sweep interpretation
- 60 reps × 80 row-tiles × 3 DPM levels per host: zero distinct hashes per tile anywhere. Bit-stable across low/auto/high.
- The driver-controlled DPM floors do not approach the Vmin cliff. Going below DPM `low` would require unsafe voltage table override (prior Probe C tried and hung ikaros).
- Side effect — at "high" ikaros completion time dropped to 0.23 ms vs 1.4 ms (boost engaged), but daedalus stayed at 2.7–2.8 ms across all levels. This reveals a per-host SCLK governor difference (board-firmware-config artefact, not silicon variance).

## Cross-angle synthesis
- 23-feature cross-angle vector (A means/std/τ × 4 wl + B {τ_heat,τ_cool,R_th} + E first 8 cores).
- L2 ikaros vs daedalus = 90.2 units (per-feature 18.8). Cosine 0.958.
- L2 dominated by Power (~70 W) and per-core time (~2 ms), but each of A/B/E independently rejects the null at Bonferroni-corrected α = 0.01 (=0.05/5).
- Multi-channel: with as few as 4 features (IDLE mean, MEDIUM mean, τ_heat, per-core time spread) a nearest-centroid classifier separates the machines with zero error — every per-rep distribution is non-overlapping on these axes.

## Power analysis (Cohen formula, 10 % effect, α=0.05, power 0.8)
| Workload | σ (W) | Target Δ (W) | N seeds needed |
|----------|------|--------------|----------------|
| IDLE     | 1.1  | 1.9          | **6**          |
| LIGHT    | 15.5 | 8.3          | 57             |
| MEDIUM   | 19.4 | 11.0         | 50             |
| HEAVY    | 21.3 | 10.9         | 61             |
With 5 reps already we detect 13–50 W IDLE/MEDIUM/HEAVY differences because effects are >>10 %. Detecting hypothetical 10 % drift on LIGHT–HEAVY would need ~50-60 reps; IDLE just 6.

## Comparison to prior 9 NULL attacks
Prior attacks probed GPU compute kernels (gemm outputs, RNG, timing, gpu_metrics blob, voltage sweeps that crashed ikaros) — they tried to force silicon below operating envelope.
This battery measures the **envelope itself**: stationary power draw (A), thermal RC of cooling stack (B), per-core sysfs latency (E). These are governed by manufacturing variance (Vt, leakage, TIM contact, fan curve, board components) and survive driver normalisation because they are not in the data path. That's why prior attacks missed them.
Caveat: A and B include cooling/board assembly contributions; only E is cleanly die-only (rank correlation across cores says distinct die ordering).

## Updated recommendation
- Identity **IS findable** on these two specific machines.
- Continue: tighten with N=60 reps for LIGHT-workload CI; replicate B another day to control ambient temperature.
- FPGA pivot not needed for identity — but remains the only way to isolate die-vs-package.
- NPU: install Ryzen-AI-SW deb stack to convert BLOCKED into a clean die-only probe.

## Thermal incidents
- 1 over-budget incident during initial too-aggressive A run (8 reps × 8 s, HEAVY=4 threads / size 1024) — APU reached 100 °C (trip 101 °C). Process killed, machine did not reboot.
- After: HEAVY=2 threads / size 512, temp cap 88 °C, abort 70 °C. All subsequent runs ≤ 91 °C peak, mostly ≤ 80 °C.
- 1 expected D-angle abort at "high" DPM on ikaros (>72 °C); partial reps recorded.

## Path
- Code: scripts/identity_benchmark/deep/{A,B,C,D,E}_*.py + _common.py + analyze_all.py + run_remaining_*.sh
- Raw per-host: results/IDENTITY_BENCHMARK_2026-05-30/deep/{ikaros,daedalus}/{A_power,B_thermal,C_npu,D_vmin,E_cpu}.json
- Cross-host analysis: results/IDENTITY_BENCHMARK_2026-05-30/deep/ANALYSIS.json
- Report: research_plan/IDENTITY_DEEP_2026-05-30_REPORT.md

## Bottom line
**Identity findable: YES.** 3 of 4 measured channels (A, B, E) independently discriminate the two machines at Cohen d > 3 with bootstrap CI excluding zero, surviving Bonferroni correction. D returned a clean null on its primary axis (no Vmin bit-flips) — informative: driver DPM floors prevent classical PUF probing without unsafe voltage override. C blocked pending NPU userspace install.
