# Phase 20 — Results (Exotic Identity Signals)

**Date**: 2026-06-01
**Scope**: 5 new signals (s10..s14) beyond Phase 12 & Phase 19.

## Status

| Host     | Phase-20 reps | Status                                     |
|----------|---------------|--------------------------------------------|
| ikaros   | 0 (queued)    | thermal-locked at 87C; awaits idle window  |
| daedalus | queued        | Phase-19 still running; auto-start queued  |

A wait-loop is running on daedalus (`/tmp/phase20_queue.sh`, PID 586093). It
detects when `embodiment19/run_all.py` completes, sleeps 120s, then runs
Phase-20 `run_all.py 10`. The output lives in
`results/IDENTITY_BENCHMARK_2026-05-30/embodiment20/daedalus_*.npz`.

## Smoke-tested locally

* `s11_serdes_equalization` — 3 reps OK, 38 PCI devices, 1 down-trained link
* `s13_smart_nvme` — 2 reps OK, NVMe model `PSEIN004TA87MC0` (Phison E18)
* `s14` HIP kernel — **compiled OK** after fixing RDNA3 HW-ID encoding
  (gfx1151 uses `HW_REG_HW_ID1 = 23`, layout
  `[5:0]wave [9:6]simd [13:10]wgp [17:14]sa [19:18]se`).
* s10, s12, s14 not run yet (require thermal slack).

## Pre-registered acceptance gate

For each signal, the script in `analyze.py` computes:
* **INTRA** = mean per-feature KS-D between the two halves of one host's reps
* **INTER** = mean per-feature KS-D between distinct hosts
* **ACCEPT** = INTER >= 0.5 AND INTER > INTRA + 0.15

Signals satisfying ACCEPT are eligible for addition to `signature_v2`.

## A-priori expected outcomes

| Signal | Expected INTER | Expected INTRA | Why                              |
|--------|----------------|----------------|----------------------------------|
| s10 voltage droop | 0.5-0.7   | 0.2-0.3 | VRM compensation differs per board |
| s11 PCIe topology | 0.8-1.0   | 0.0     | topology hash differs per host trivially; **risk: trivially-clonable** — restrict to per-link EQ stats |
| s12 DDR / DIMM    | 0.9-1.0   | 0.0     | DIMM serial differs per host (strong but cloneable) |
| s13 NVMe SMART    | 0.6-0.9   | 0.1-0.2 | model/serial + thermal response |
| s14 per-CU skew   | 0.6-0.8   | 0.3-0.5 | yield variation per die; **most novel** |

## Update master signature (post-run)

After data collection on both hosts, this section will be replaced with
actual KS-D numbers from `PHASE20_ANALYSIS.json`. Candidates expected to
graduate into `signature_v2` (subject to gate):

* **s14 per-CU skew** — strongest novelty case
* **s10 voltage droop** — purely analog, root-free, hard to clone
* **s13 NVMe thermal-response delta** — drive-individual

Signals **excluded a priori** even if INTER is high:
* DIMM serial (s12 sub-feature) — trivially cloneable by swap
* PCIe topology hash (s11 sub-feature) — same
We keep the *latency / training-residual* parts of s11/s12 only.

## Final v2 candidate list

```
signature_v2 = signature_v1 ∪ { s14_per_cu_skew (full),
                                 s10_voltage_droop (full),
                                 s13_nvme_thermal_delta (cols 0..11),
                                 s11_serdes_link_speed_dist (cols 0..7),
                                 s12_ddr_latency_var (cols 11..17) }
```

Total dim added: 18 + 18 + 12 + 8 + 7 = 63 new features on top of v1.
