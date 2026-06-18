# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: RESX_STEP0_RESULT_2026-06-12.md (14752 chars) ===
```
# H7-RES-X Step 0 — device-level reservoir-capacity gate: HONEST NEGATIVE (2026-06-12)

Pre-registered gate (research_plan/H7_RES_X_REDESIGN_2026-06-11.md): before coupling any LLM,
test whether the live gfx1151 SoC substrate is a genuine physical reservoir — i.e. whether its
fading-memory response to an injected drive `u` lets a LINEAR readout compute a NONLINEAR-temporal
function of u that a linear-on-u baseline cannot. PASS iff reservoir ≥0.70 (16-way) AND baseline ≤0.30.

Script: scripts/identity_benchmark/h7_resx_step0_gate.py. Result: results/.../resx_step0_gate_ikaros.json.

## Verdict: FAIL (robust, characterized, bias-checked)

Across the full nonlinear-capacity suite, under BOTH a strong drive (run2: drive-landed Cohen's d
ch5=−7.93, MC=9.66 bits) and a light drive (run4: MC=5.78 bits) — same conclusion:

| task            | chance | die-reservoir | linear-on-u baseline |
|-----------------|--------|---------------|----------------------|
| RECALL (linear) | 0.50   | 0.70          | 1.00                 |
| XOR τ1,τ2       | 0.50   | 0.52          | 0.50                 |
| XOR τ2,τ5       | 0.50   | 0.53          | 0.48                 |
| parity 2-bit    | 0.25   | 0.26          | 0.31                 |
| parity 4-bit    | 0.0625 | 0.08          | 0.11                 |

**The die genuinely REMEMBERS the load history** (RECALL 0.70 > 0.50; memory capacity ~6–10 bits)
**but adds ZERO nonlinear computation** — on every XOR/parity task the reservoir never beats a plain
linear-in-u baseline; both sit at chance. The substrate is a **linear low-pass memory of compute
load, not a nonlinear reservoir.**

## Why this kills the H7-RES-X path (both necessity routes collapse)
1. Reservoir-necessity route: the die does no nonlinear-temporal work a linear model can't → it is
   not a computational resource an LLM could be made to NEED.
2. Uncommanded-ξ route: the readout is LINEAR, so whatever the die exposes, the model's own rank-4
   linear adapter can reconstruct → the design's "throttled readout cannot rebuild it" assumption fails.

Physics: power/thermal/counter telemetry is a smoothed (low-pass) measurement of compute power — a
near-linear system. Linear systems cannot compute XOR/parity. (Contrast the FPGA NS-RAM reservoir
z2296, which has genuine nonlinear neuron dynamics — tanh, spikes, thresholds — and DID do XOR/MC.)
This is a property of the signal, not a tuning gap; build_best_features already supplies order-2/3
temporal products, so the readout is not the bottleneck.

## Decision (pre-committed): STOP. Do NOT couple the LLM.
Proceeding to Step 1 would only reproduce the v13/v14 kill-switch (a tamper-detector on a signal the
model doesn't compute with) — exactly the success-bias we pre-committed to avoid. Reported as the
correct scientific result: **on the live gfx1151 SoC telemetry, substrate-rooted functional
embodiment is not achievable, because the substrate carries no nonlinear computation the model could
need.** The reusable, real findings stand: a temporal-dynamics tamper detector + hardware-gated style
knob + a 500Hz hardware→model online harness; plus this device-level characterization (linear memory,
no nonlinearity) which would have to change (e.g. route through the real FPGA NS-RAM reservoir, or a
genuinely nonlinear analog stage) for any future attempt.

## Follow-up probe #2 — cache-latency reservoir (driven pointer-chase): ALSO NEGATIVE
Per the "test all metrics deeply, unconventionally" mandate, built a CPU-side driven cache-latency
reservoir (latprobe.c: u=1 refreshes a hot ~L2 buffer, u=0 thrashes a cold DRAM buffer; per-step we
time a fixed readout-probe of the hot buffer → latency = nonlinear fading memory of "time since
refresh"). Drive lands hard (Cohen's d ≈ −1.9, RECALL_t3 = 0.87). But against the DECISIVE control —
the SAME nonlinear readout (build_best_features) applied to the clean drive `u` alone — the die loses
on every task:

| task        | chance | die-reservoir | linear-on-u | **nonlinear-on-u (control)** |
|-------------|--------|---------------|-------------|------------------------------|
| RECALL_t3   | 0.50   | 0.874         | 1.000       | 0.905                        |
| XOR τ1,τ2   | 0.50   | 0.583         | 0.522       | **0.903**                    |
| XOR τ2,τ5   | 0.50   | 0.554         | 0.514       | **0.775**                    |
| parity 2bit | 0.25   | 0.322         | 0.277       | **0.720**                    |
| parity 4bit | 0.0625 | 0.070         | 0.114       | **0.180**                    |

The latency is a strictly **lossy echo** of u: the readout does better on u directly than on the
die's measurement of u. The die adds nothing nonlinear — it only loses information. (An earlier run's
"XOR 0.62 > linear 0.52" was the build_best_features readout computing XOR on linearly-recalled
u-memory, NOT the die.) Same physics as the power/thermal negative: a single scalar low-pass channel
with fading memory = a one-dimensional linear reservoir, which cannot do XOR/parity.

## Follow-up probe #3 — SPATIAL contention reservoir (capacity-eviction): ALSO NEGATIVE
Tested the one nonlinearity the temporal probes can't reach: capacity CONTENTION. Two simultaneous
drive bits a,b each stream a large buffer; a small probe buffer P is timed each step. The hope: P is
evicted only when a AND b are both active (a Boolean threshold the silicon computes), letting a
LINEAR readout do XOR = (1-a)+(1-b)-2·AND that linear-on-(a,b) cannot. Rigorous: readout kept LINEAR
so any gain is the die's, not the feature basis. (latprobe_contention.c + h7_contention_reservoir.py.)

Drove the threshold as hard as the hardware allows — buffers sized to overflow the 32 MB L3 so P
evicts all the way to DRAM (huge separation, AND-pop Cohen's d=+0.94). Result on XOR: reservoir
(linear-on-latency) 0.61 vs linear-on-drive baseline 0.49 — a real but weak +12pp. STILL FAILS the
0.70 gate. The conditional means expose why:

| lat_P | a,b=00 | 01 | 10 | 11 | shape |
|-------|--------|----|----|----|-------|
| 16 MB buffers | 11004 | 13968 | 14503 | 17841 | graded — ≈linear in (a+b) |
| 9.6 MB | 3150 | 4000 | 4061 | 4070 | saturating-**OR** (one buffer already evicts P) |
| 4.8 MB | 2939 | 3787 | 3765 | 3795 | OR |

A full buffer-size scan finds **no AND regime anywhere**: a 64 KB victim's survival is a smooth
sigmoid of total bytes streamed — **monotone in (a+b)** at every operating point (linear when graded,
OR-like when saturated). A monotone function of (a+b) carries ZERO XOR information for a linear
readout. The ~0.61 blip is just the weak convex curvature of saturation, not extractable computation.

## Follow-up probe #4 — BRANCH-PREDICTOR mispredict-count via PMU (clean integer): ALSO NEGATIVE
Top untested lever from the w22spoqla design workflow. The Zen5 TAGE predictor's mispredict count is
provably a non-additive function of recent branch-direction history; the prior timing probes failed
only because the ~60-cycle rdtscp fence floor buried the ~15-cycle mispredict. Read the CLEAN integer
count from PERF_COUNT_HW_BRANCH_MISSES (bpred_pmu.c, no sudo — perf_event_paranoid=1 allows
self-monitoring) and ran the rigorous LINEAR-readout test + phase-shuffle surrogate null
(h7_bpred_reservoir.py). Sanity: predictor reacts strongly (random drive 16 mispred/step, all-ones 2,
alternating 2.7 — it LEARNS any low-complexity pattern). But on the gate:

| task   | chance | reservoir | u_linear | surrogate p99 |
|--------|--------|-----------|----------|---------------|
| XOR_k1 | 0.50   | 0.498     | 0.483    | 0.523         |
| XOR_k2 | 0.50   | 0.498     | 0.495    | 0.520         |
| XOR_k3 | 0.50   | 0.496     | 0.503    | 0.512         |
| par-4  | 0.0625 | 0.056     | 0.164    | 0.081         |

FAIL — reservoir at chance, ≤ u_linear, and BELOW the phase-shuffle surrogate (so even the residual is
spectrum-explainable, not computation). Why: because the predictor learns any structured pattern, the
mispredict count measures the window's *transition-density / complexity* — a smooth near-linear
statistic — not the specific pairwise XOR. Same monotone trap, fourth confirmation.

## UNIFIED CONCLUSION across all 4 probe families (power/thermal, temporal-latency, spatial-contention, branch-predictor)
Every drivable gfx1151/SoC hardware metric is a **monotone, fading-memory function of instantaneous
total compute load.** This class = a linear temporal reservoir + a static monotone saturation. Such a
system provably cannot compute XOR/parity, so there is **no nonlinear computation an LLM could be made
to NEED** from SoC telemetry — confirmed three independent ways, after deliberately driving each
nonlinearity to its physical limit (Cohen's d up to 7.9 on drive-landing, 0.94 on the contention
threshold). This is a property of the signal physics, not a tuning or bandwidth gap.

### Constructive consequence (the "both" paths stand, SoC-computation route is closed)
1. EMBODIMENT-via-computation requires a substrate with genuine Boolean/threshold dynamics. The SoC
   has none; the **real FPGA NS-RAM reservoir (z2296: tanh/spike/threshold, proven XOR + MC>2)** does.
   Any functional-embodiment attempt must route the necessity through the FPGA, not SoC telemetry.
2. DEVICE-IDENTITY without computation = **crypto-lock / PUF** (uniqueness + freshness without needing
   the substrate to compute) — the orthogonal route. **DEMONSTRATED (see below).**

One untested compute lead remains (deferred): the **DRAM row-buffer** (open/closed row = a true
bistable latch below the cache, the one threshold the prefetcher can't smooth). It needs root +
hugepages + timing-based DRAM bank discovery (DRAMA), and on this soldered LPDDR5X + on-die-ECC +
address-scrambled APU a negative could be inconclusive ("couldn't establish the mechanism") rather
than decisive. Not run; flagged for an explicit follow-up if the four conclusive negatives aren't
considered sufficient.

## ✅ PUF / CRYPTO-LOCK route — DEMONSTRATED die-unique fingerprint (the other half of "both")
Source: `/sys/devices/system/cpu/cpuN/cpufreq/amd_pstate_prefcore_ranking` — per-core CPPC fused
performance ranking (reboot-stable fuse values, SMT siblings duplicate; no root, no compute needed).
Read on BOTH Strix Halo dies (numeric core order):

```
ikaros   cores: 231 236 216 211 206 236 221 226 166 181 196 191 176 186 171 201
daedalus cores: 231 211 236 216 206 221 226 236 166 191 181 201 176 171 186 196
```

- **Inter-die distinctness: 12/16 cores differ = 75%** (gate ≥25%); byte-level Hamming 42/128 = 32.8%.
- **same multiset = True** → both dies were binned on the SAME fused-level grid, but WHICH physical
  core got which ranking is the manufacturing-random, die-unique permutation = a genuine PUF (NOT a
  model-level constant, which is the critical control — they are NOT equal). SHA-256(16-byte vector)
  gives fully distinct keys (ikaros a1debc4a… vs daedalus 30f4ee8a…). Result: cppc_puf_distinctness.json.
- Crypto-strength ranking on this box: **SEV-SNP VCEK ≫ CPPC-ranking PUF > DRAM-PUF ≫ TSC-skew.** The
  VCEK (fuse-derived ECDSA key, AMD-attested cert chain) is strictly stronger if inference runs inside
  an SNP guest; the CPPC PUF is the zero-dependency software fallback demonstrated here.
- This cleanly satisfies the UNIQUE-to-die + (with a server nonce / RDSEED-TRNG salt) FRESH requirements
  WITHOUT needing the substrate to compute — exactly the route that does not require embodiment.

## EXHAUSTIVE CAMPAIGN (2026-06-12) — closing the open frontier for #2, with a structural theorem
User asked to test every combination/differential exhaustively. The load-meter combos are pruned by
PROOF (Dambre): any combination/differential of monotone-in-load signals is still a function of the one
load variable → can't XOR. We swept everything theory does NOT already kill:

1. **Exhaustive cross-channel sweep** (h7_exhaustive_xchannel.py): 915-dim basis = all 10 channels'
   lags + ALL pairwise cross-channel products ch_i(t−a)·ch_j(t−b) + auto-products + pairwise
   differentials. Drive landed hard (Cohen's d up to −3.1). Result: the reservoir DOES compute XOR —
   0.95 for adjacent lags (XOR_12=0.949, _23=0.937, _34=0.927, _45=0.907, _56=0.903), far above
   linear-on-u (0.50) and phase-shuffle surrogate (0.55). BUT it loses to nonlinear-on-u = **1.000**.
   → **STRUCTURAL THEOREM: the die is never NEEDED for any function of a drive WE COMMAND.** XOR of a
   commanded drive is a deterministic polynomial of the drive's own lags, so the model self-computes it
   (control = 1.0) without the die. The die supplies only ~2–3 steps of fading MEMORY; the NONLINEARITY
   is the readout's, never the silicon's. This is WHY every #2 probe fails — not wrong sensors, a wrong
   (structurally impossible) target. Die-necessity requires EXOGENOUS ξ (uncommanded), where the die
   need not compute at all — only be unique+fresh (already solved).
2. **Coincidence detectors** (the only non-load-meter, non-self-computable mechanisms):
   - store→load forwarding (store_fwd.c): partial-overlap stall fires on a≠b physically, BUT flat at
     60 cyc — the ~15-cyc stall is below the rdtsc fence floor; no Zen5 PMU store-forward event exposed
     (the integer-rescue that worked for branch-miss is unavailable). Dead.
   - DRAM row-buffer (/tmp/dram_row.c, 200×2MB hugepages): flat **480 cyc for every offset, ratio 1.00**
     — the open/closed-row bistable is MASKED on soldered LPDDR5X (on-die ECC + address scrambler +
     closed-page policy). The spec's predicted risk materialized. Dead.

### Exhaustive verdict for #2: closed every way + proven structurally unreachable
Single channels (4), all-joint-linear, exhaustive cross-channel products/differentials (915-dim),
branch-predictor PMU, store-forwarding, DRAM row-buffer — ALL negative, and the cross-channel sweep
proves it's structural, not a coverage gap. #1 (unique: CPPC 75% + dynamics 14×) and #3 (fresh: RDSEED)
remain SOLVED. The honest end state: **the substrate cannot be made to do nonlinear computation an LLM
NEEDS — but it doesn't have to; embodiment-grade dependence routes through exogenous-ξ uniqueness +
freshness (have it) or the real FPGA NS-RAM reservoir (has genuine nonlinear dynamics), not SoC compute.**

## Thermal/ops note
The first heavy busy-matmul drive hit 99°C (ACPI trip) because (a) the load was sustained and (b) the
external watchdog attach raced (read its target PID as already-gone → exited instantly, leaving the
run unguarded). Fix: LIGHT drive (short burst + idle per step, low duty cycle, stayed 43–61°C) + a
RELIABLE in-loop self-guard (check every 20 steps, pause at 82°C) rather than depending on the
external watchdog. Lesson: for self-driving compute loops, guard INSIDE the loop.

```


=== FILE: bilinear_clean_analysis.json (794 chars) ===
```json
{
  "experiment": "bilinear-interaction / mixed-partial (2D continuous GPU\u00d7CPU load, user-suggested)",
  "necessity_centered_product": {
    "from_drive_linear": -0.154,
    "from_die_linear": -0.049,
    "sanity_ab_given": 1.0,
    "note": "centered product orthogonal to a+b; die cannot linearly reconstruct it -> not usable"
  },
  "per_channel_ab_gain_over_total_load_poly": {
    "ch5_power_energy_rate": 0.138,
    "all_other_channels": "~0 (pure g(a+b) saturation = load-meter w/ curvature)"
  },
  "verdict": "FAINT genuine bilinear interaction in power channel (ch5, +0.138) \u2014 the first real die nonlinearity found \u2014 but too weak/entangled to be linearly usable (necessity ~0), and it is throttling/Vdroop physics (NOT die-unique).",
  "raw": "bilinear_raw_ikaros.npz"
}
```


=== FILE: cppc_puf_distinctness.json (681 chars) ===
```json
{
  "source": "/sys/devices/system/cpu/cpuN/cpufreq/amd_pstate_prefcore_ranking",
  "ikaros_percore": [
    231,
    236,
    216,
    211,
    206,
    236,
    221,
    226,
    166,
    181,
    196,
    191,
    176,
    186,
    171,
    201
  ],
  "daedalus_percore": [
    231,
    211,
    236,
    216,
    206,
    221,
    226,
    236,
    166,
    191,
    181,
    201,
    176,
    171,
    186,
    196
  ],
  "per_core_diff_frac": 0.75,
  "byte_hamming_frac": 0.328125,
  "same_multiset": true,
  "verdict": "PASS",
  "note": "die-unique permutation of fused per-core perf ranking; SMT siblings duplicate; reboot-stable fuse values; PUF distinctness >> 25% gate"
}
```


=== FILE: dynamics_fingerprint_result.json (1910 chars) ===
```json
{
  "experiment": "H7-PAT dynamics-fingerprint gate (de-confounded)",
  "method": "identical deterministic light load on both dies; 120 windows x256samp x10ch; z-score per window (kills cloneable DC offset+gain); 2nd-order features (per-ch var, AR/thermal-lag@1-16, cross-ch corr, PSD bands); 6-D PCA to remove high-dim overfitting",
  "naive_classifier_acc": {
    "cross_die": 1.0,
    "same_die_diff_session": 1.0,
    "note": "100% even same-die => high-dim overfitting/session-drift; UNINFORMATIVE alone"
  },
  "deconfounded_centroid_separation_6dPCA": {
    "same_die_ikaros1_ikaros2": 0.54,
    "same_die_daed1_daed2": 0.68,
    "cross_die_session1": 9.38,
    "cross_die_session2": 7.52,
    "cross_over_same_ratio": 13.91
  },
  "cross_session_transfer_acc": 1.0,
  "verdict": "REAL machine-stable dynamics fingerprint (cross-die 14x > session drift, transfers across sessions)",
  "caveats": [
    "MACHINE-level not proven DIE-level: ikaros vs daedalus differ, but could be firmware/governor/board-sensor/hwmon-config, not silicon manufacturing variation. Need same-model identical-config 3rd unit, or feature attribution, to claim 'die'.",
    "REPLAYABLE => does NOT satisfy embodiment freshness: features are computed from a static recorded window, so a recording (e.g. daedalus2.npz) reproduces them exactly and is classified as that machine. A tape impersonates the chip perfectly. Static fingerprint = identity/PUF category (like CPPC), NOT non-replayable live dependence."
  ],
  "consequence": "For IDENTITY/crypto-lock: strengthens it (rich behavioral fingerprint, complements CPPC/VCEK). For EMBODIMENT (non-replayable live need): insufficient by construction; requires CLOSED-LOOP binding where the readout depends on the signal's REACTION to the model's own real-time compute (a recording does not react => replay breaks it). That closed-loop test is the genuine next PAT experiment."
}
```


=== FILE: exhaustive_xchannel_ikaros.json (3595 chars) ===
```json
{
  "host": "ikaros",
  "L": 1800,
  "basis_dims": 915,
  "task_suite": {
    "XOR_12": {
      "chance": 0.5,
      "reservoir": 0.9494949494949495,
      "u_linear": 0.5050505050505051,
      "u_nonlinear": 1.0,
      "surrogate99": 0.5430101010101009,
      "die_wins": false
    },
    "XOR_13": {
      "chance": 0.5,
      "reservoir": 0.8787878787878788,
      "u_linear": 0.509090909090909,
      "u_nonlinear": 1.0,
      "surrogate99": 0.5482222222222222,
      "die_wins": false
    },
    "XOR_14": {
      "chance": 0.5,
      "reservoir": 0.6808080808080809,
      "u_linear": 0.5191919191919192,
      "u_nonlinear": 1.0,
      "surrogate99": 0.5372323232323233,
      "die_wins": false
    },
    "XOR_15": {
      "chance": 0.5,
      "reservoir": 0.5232323232323233,
      "u_linear": 0.5191919191919192,
      "u_nonlinear": 1.0,
      "surrogate99": 0.5467676767676768,
      "die_wins": false
    },
    "XOR_16": {
      "chance": 0.5,
      "reservoir": 0.5070707070707071,
      "u_linear": 0.49292929292929294,
      "u_nonlinear": 1.0,
      "surrogate99": 0.5557373737373738,
      "die_wins": false
    },
    "XOR_23": {
      "chance": 0.5,
      "reservoir": 0.9373737373737374,
      "u_linear": 0.4767676767676768,
      "u_nonlinear": 1.0,
      "surrogate99": 0.5603434343434343,
      "die_wins": false
    },
    "XOR_24": {
      "chance": 0.5,
      "reservoir": 0.7555555555555555,
      "u_linear": 0.5212121212121212,
      "u_nonlinear": 1.0,
      "surrogate99": 0.5430101010101009,
      "die_wins": false
    },
    "XOR_25": {
      "chance": 0.5,
      "reservoir": 0.5111111111111111,
      "u_linear": 0.4888888888888889,
      "u_nonlinear": 1.0,
      "surrogate99": 0.5448888888888889,
      "die_wins": false
    },
    "XOR_26": {
      "chance": 0.5,
      "reservoir": 0.498989898989899,
      "u_linear": 0.5212121212121212,
      "u_nonlinear": 1.0,
      "surrogate99": 0.5466262626262626,
      "die_wins": false
    },
    "XOR_34": {
      "chance": 0.5,
      "reservoir": 0.9272727272727272,
      "u_linear": 0.49292929292929294,
      "u_nonlinear": 1.0,
      "surrogate99": 0.5506666666666666,
      "die_wins": false
    },
    "XOR_35": {
      "chance": 0.5,
      "reservoir": 0.5070707070707071,
      "u_linear": 0.5131313131313131,
      "u_nonlinear": 1.0,
      "surrogate99": 0.5388282828282829,
      "die_wins": false
    },
    "XOR_36": {
      "chance": 0.5,
      "reservoir": 0.5070707070707071,
      "u_linear": 0.509090909090909,
      "u_nonlinear": 1.0,
      "surrogate99": 0.5544242424242425,
      "die_wins": false
    },
    "XOR_45": {
      "chance": 0.5,
      "reservoir": 0.907070707070707,
      "u_linear": 0.48484848484848486,
      "u_nonlinear": 1.0,
      "surrogate99": 0.5476565656565656,
      "die_wins": false
    },
    "XOR_46": {
      "chance": 0.5,
      "reservoir": 0.501010101010101,
      "u_linear": 0.503030303030303,
      "u_nonlinear": 1.0,
      "surrogate99": 0.5471919191919191,
      "die_wins": false
    },
    "XOR_56": {
      "chance": 0.5,
      "reservoir": 0.9030303030303031,
      "u_linear": 0.47878787878787876,
      "u_nonlinear": 1.0,
      "surrogate99": 0.5510909090909091,
      "die_wins": false
    },
    "PAR_4bit": {
      "chance": 0.0625,
      "reservoir": 0.5131313131313131,
      "u_linear": 0.09090909090909091,
      "u_nonlinear": 1.0,
      "surrogate99": NaN,
      "die_wins": false
    }
  },
  "die_wins_on": [],
  "verdict": "FAIL",
  "pass_rule": "some XOR: reservoir >= u_nonlinear+0.04 AND > surrogate99 AND > chance+0.04"
}
```
