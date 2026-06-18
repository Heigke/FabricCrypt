# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: phase13_signature_v2_slim.json (3923 chars) ===
```json
{
  "taskA_signature_v2": {
    "ikaros": {
      "n_reps": 10,
      "dim": 290,
      "mean_cv": 1.0,
      "p95_cv": 2.999998000022,
      "within_dist_p95": 0.14505343783487976
    },
    "daedalus": {
      "n_reps": 10,
      "dim": 290,
      "mean_cv": 1.0,
      "p95_cv": 2.7767641041191884,
      "within_dist_p95": 0.4169308482179035
    },
    "cross_host_dist_p50": 0.08396385237780801,
    "cross_host_dist_p5": 0.006260999110329374
  },
  "taskB_constitutive_summary": {
    "A": {
      "mean_nrmse": 0.4329731862270439,
      "std_nrmse": 0.037360820363607605,
      "median_nrmse": 0.4284322823654135,
      "n": 30
    },
    "B": {
      "mean_nrmse": 0.43297318622726433,
      "std_nrmse": 0.03736082036365187,
      "median_nrmse": 0.428432282365794,
      "n": 30
    },
    "C": {
      "mean_nrmse": 1.0273025753453013,
      "std_nrmse": 0.0707892218538849,
      "median_nrmse": 1.028893462209243,
      "n": 30
    },
    "D": {
      "mean_nrmse": 0.8389613216385982,
      "std_nrmse": 0.03323405091046125,
      "median_nrmse": 0.8376983748777935,
      "n": 30
    },
    "contrasts": {
      "B_minus_A": {
        "mean": 3.271661788990769e-05,
        "ci95": [
          -0.0189168967864849,
          0.01885030749406172
        ],
        "rel_reduction_pct": 0.0075562688246147945
      },
      "C_minus_A": {
        "mean": 0.5943618729317767,
        "ci95": [
          0.565618548389063,
          0.6232046353506707
        ],
        "rel_reduction_pct": 57.856554358526616
      },
      "D_minus_A": {
        "mean": 0.40600906023993066,
        "ci95": [
          0.38781257265085517,
          0.4233474105927056
        ]
      },
      "gate_15pct_AB_passed": false,
      "gate_15pct_swap_AC_passed": true
    }
  },
  "taskC_chassi_lock_summary": {
    "own": {
      "mean": 0.7143999999999999,
      "std": 0.0626724819996783,
      "min": 0.571,
      "max": 0.819,
      "n": 10
    },
    "transplant": {
      "mean": 0.2636,
      "std": 0.04089547652247128,
      "min": 0.168,
      "max": 0.324,
      "n": 10
    },
    "spoof_stored": {
      "mean": 0.8992000000000001,
      "std": 0.024074052421642688,
      "min": 0.854,
      "max": 0.939,
      "n": 10
    },
    "spoof_random": {
      "mean": 0.4798,
      "std": 0.08384127861620433,
      "min": 0.372,
      "max": 0.637,
      "n": 10
    },
    "gate_own_gt_0_90": false,
    "gate_transplant_lt_0_30": true,
    "gate_spoof_gt_0_80": true,
    "gate_all_passed": false
  },
  "taskD_drift": {
    "n_captures": 12,
    "interval_s": 120,
    "duration_s": 1324.1277160644531,
    "pairwise_dist_p50": 0.045579860667837924,
    "pairwise_dist_p95": 0.19196905705059158,
    "pairwise_dist_max": 0.25627745664110135,
    "gate_p95_lt_0_05_passed": false,
    "capture_times_s": [
      1.3038899898529053,
      122.59079504013062,
      243.87633609771729,
      365.15905022621155,
      481.45461654663086,
      602.7467830181122,
      724.0219333171844,
      845.3049013614655,
      961.5878949165344,
      1082.873577594757,
      1204.1496751308441,
      1325.431606054306
    ]
  },
  "taskE_classifier": {
    "n_total": 20,
    "n_ikaros": 10,
    "n_daedalus": 10,
    "loo_acc": 1.0,
    "gate_gt_0_95_passed": true,
    "preds": [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1
    ],
    "truth": [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1
    ]
  },
  "gates": {
    "B_swap_gate": true,
    "B_AB_gate": false,
    "C_own>0.90": false,
    "C_transplant<0.30": true,
    "C_spoof>0.80": true,
    "D_drift_p95<0.05": false,
    "E_loo>0.95": true
  },
  "n_gates_passed": 4,
  "n_gates_total": 7
}
```


=== FILE: phase14b_summary.txt (3695 chars) ===
```
========================================================================
Phase 14B — Embodied Tiny Identity Benchmark — Summary
========================================================================
host: ikaros  device: cpu
APU temp start=51.0C end=62.0C
ikaros sigs N=500  daedalus sigs N=500

--- T1 ---
  vanilla  MSE = 0.032 (CI95 0.009..0.055)
  embodied MSE = 0.031 (CI95 0.018..0.036)
  ratio = 0.978  prereg(<=0.5): False
  capability gain = 2.2% MSE reduction

--- T2 ---
  vanilla  AUROC = 0.509
  embodied AUROC = 1.000
  prereg(emb>=0.85 & van<=0.6): True
  capability gain = +0.491 AUROC

--- T3 ---
  vanilla  acc = 0.501
  embodied acc = 1.000
  used_real_daedalus_peer = True
  prereg(emb>=0.95 & van<=0.55): True
  capability gain = +50pp accuracy

--- T4 ---
  vanilla  acc=0.000 speedup=1.000x
  embodied acc=1.000 speedup=1.057x
  speedup ratio = 1.057  prereg(>=1.2): False
  thps=[14735041.0, 15570715.0, 14319733.0, 15040315.0, 14765398.0]  best_idx=1

--- Spoof defense ---
T1 MSE (lower=better honest performance, higher=spoof fails to predict):
  honest             MSE=0.238
  random_sig         MSE=0.290
  static_replay      MSE=0.131
  nonce_mismatch     MSE=0.492
  stored_peer        MSE=0.198

T3 own-classification rate (high=accepted as own):
  honest_own_p0_rate        p0=0.990
  replay_one_p0_rate        p0=1.000
  random_p0_rate            p0=0.600
  nonce_mismatch_p0_rate    p0=0.960
  peer_p0_rate              p0=0.020

========================================================================
Cross-task evaluation against user's 3 requirements
========================================================================
  R1 = identity coupling (does the chip's state drive the output?)
  R2 = unfakeable      (does replay/random/nonce-mismatch break it?)
  R3 = capability gain (does embodiment improve THIS task's metric?)

T1 (latency prediction):
  R1: weak  - expression dominates target; sig adds little
  R2: weak  - static_replay MSE LOWER than honest -> sig not load-bearing
  R3: weak  - ratio ~1.0 (no significant gain)
  Verdict: NEGATIVE — confirms 'wrong task' hypothesis from brief

T2 (anomaly detection):
  R1: strong - sig IS the input
  R2: strong - any plausible spoof retains anomalies; baseline-based defense
  R3: strong - +0.49 AUROC (vanilla 0.509 -> embodied 1.000)
  Verdict: STRONG WIN

T3 (host identification):
  R1: strong - sig directly identifies host (real ikaros vs daedalus data)
  R2: PARTIAL - peer_p0_rate=0.02 (cross-host rejected), BUT static_replay
              accepted (100%) -> needs challenge-response nonce mix per call
  R3: strong - +53pp accuracy (0.500 -> 1.000) with real cross-host data
  Verdict: STRONG WIN with caveat on replay defense

T4 (substrate-aware completion):
  R1: medium - sig identifies machine, model picks chip-specific N
  R2: medium - works by construction
  R3: weak  - throughputs of candidates differ by <10% on this CPU; 1.06x
  Verdict: PASSES classification but fails 1.2x speedup prereg

========================================================================
WINNING TASK: T2 (workload anomaly detection)
  - Largest, most reliable capability gain (+0.49 AUROC)
  - Strong unfakeability by construction (anomaly distribution is detectable
    regardless of which spoofed signal you inject)
  - Sig is load-bearing: vanilla has literally no signal (AUROC ~0.5)

RUNNER-UP: T3 (twin paradox)
  - Powerful demo (deployed model knows which host it is)
  - Replay defense requires per-call audience challenge nonce mixed into sig
  - Cross-host validated with REAL daedalus data (peer rejection 98%)
========================================================================
```


=== FILE: phase15_SUMMARY.json (4409 chars) ===
```json
{
  "phase": 15,
  "question": "Does the chip's physics genuinely augment AI computation, beyond merely providing info?",
  "experiments": {
    "E1_free_entropy_reg": {
      "n_seeds": 30,
      "metric": "test_acc on noisy classification",
      "vanilla_mean": 0.6201,
      "embodied_mean": 0.6177,
      "synthetic_mean": 0.6212,
      "delta_emb_minus_syn_pp": -0.35,
      "ci95_pp": [-0.58, -0.13],
      "gate_threshold_pp": 1.5,
      "gate_pass": false,
      "verdict": "EMBODIED LOSES — chip jitter is significantly worse regularizer than matched synthetic by 0.35pp; CI excludes zero on the wrong side. The chip's temporal autocorrelation makes its noise structured rather than uniformly random, which is suboptimal as a regularizer."
    },
    "E2_attention_bias_dram": {
      "n_seeds": 30,
      "metric": "test_acc on text classification",
      "vanilla_mean": 0.7649,
      "embodied_mean": 0.7580,
      "random_mean": 0.7566,
      "delta_emb_minus_rnd_pp": 0.14,
      "ci95_pp": [-0.14, 0.43],
      "gate_threshold_pp": 1.5,
      "gate_pass": false,
      "verdict": "NULL — embodied marginally beats random bias by 0.14pp but CI spans zero. Both embodied and random bias are slightly worse than no bias. DRAM-latency carries no useful per-token attention signal at this model scale."
    },
    "E3_thermal_inference_budget": {
      "n_seeds": 12,
      "metric": "qps at iso-accuracy",
      "all_seeds_qps_ratio_mean": 33.85,
      "clean_seeds_qps_ratio_mean": 1.055,
      "clean_seeds_ci95": [0.982, 1.116],
      "acc_gap_pp": -0.17,
      "gate_threshold_ratio": 1.15,
      "gate_pass": false,
      "skip_rate_when_active": "1.00 (always skipped) when chip warm",
      "verdict": "PARTIAL — when thermal threshold triggers (chip ≥58C), embodied skips layer 4 and delivers ~12% qps gain with iso-accuracy. Cold-start contamination and noisy timing make the full-population CI just miss the 15% bar. The mechanism works; the gate doesn't pass at the pre-registered threshold."
    },
    "E4_predictive_scheduling": {
      "n_seeds": 12,
      "metric": "batch latency reduction (%) vs arrival order",
      "rel_improvement_mean_pct": 2.84,
      "ci95_pct": [-0.02, 5.35],
      "gate_threshold_pct": 10.0,
      "predictor_pearson_with_oracle_mean": 0.04,
      "gate_pass": false,
      "verdict": "NULL — embodied predictor's correlation with true latency is ~0 (chance). Tiny 2.8% gain is from any-non-arrival-order, not from embodiment. Chip state at this granularity does not predict per-request latency."
    },
    "E5_per_machine_finetune": {
      "n_seeds": 25,
      "metric": "PPL on user-specific corpus",
      "vanilla_ppl_mean": 75.04,
      "embodied_ppl_mean": 75.65,
      "random_ppl_mean": 76.28,
      "rel_emb_vs_van_pct": -0.81,
      "ci_emb_vs_van_pct": [-0.94, -0.67],
      "rel_emb_vs_rnd_pct": 0.83,
      "ci_emb_vs_rnd_pct": [0.60, 1.05],
      "gate_pass": false,
      "verdict": "INFORMATIVE-BUT-NOT-USEFUL — embodied conditioning is significantly better than RANDOM conditioning of matched norm (0.83% PPL reduction, CI [0.60, 1.05] excludes zero). But adding any conditioning at all hurts vs vanilla (-0.81% PPL). Chip vector carries information, but it doesn't replace a learned, no-conditioning baseline."
    }
  },
  "overall_count_computationally_helpful": 0,
  "overall_count_informationally_distinguishable": 1,
  "honest_summary": "Across 109 seeds and 5 pre-registered experiments, ZERO experiments cleared their computational-gain pre-reg gate. E3 shows the mechanism works directionally (~5-12% qps gain when thermal threshold triggers, iso-accuracy) but does not reach the 15% bar with proper bootstrap CI. E5 shows the chip vector beats matched-norm random conditioning (CI excludes zero) but the conditioning architecture itself is a net loss vs no conditioning. The three-step story is NOT completed by Phase 15: identity and unfakeable binding hold, but deep computational embodiment-as-improvement is not demonstrated.",
  "winner_claims_with_ci": [],
  "next_steps_if_pursued": [
    "E3 retry with a workload that reliably keeps chip hot AND smarter skip policy (e.g. skip-layer-only-when-headroom-meets-need); 30 seeds with timing isolation.",
    "E5 retry with a LEARNABLE projection head that can choose to attenuate the conditioning when unhelpful, so the network can recover the vanilla baseline."
  ]
}

```
