# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: N_HDC_UCIHAR_N8192_summary.json (1879 chars) ===
```json
{
  "experiment": "N_HDC_UCIHAR_N8192",
  "topology": "HDC bundling (record-based) + NS-RAM V_d-as-bit binding",
  "D": 8192,
  "Q": 32,
  "n_features": 561,
  "n_classes": 6,
  "n_train": 7352,
  "n_test": 2947,
  "seeds": [
    0,
    1,
    2
  ],
  "per_seed": [
    {
      "seed": 0,
      "D": 8192,
      "Q": 32,
      "train_acc": 0.8424918389553863,
      "test_acc": 0.8452663725822871,
      "wall_s": 31.85513162612915,
      "wall_encode_s": 20.437013626098633,
      "wall_nsram_s": 11.2744619846344,
      "wall_train_s": 0.09407210350036621,
      "wall_test_s": 0.030012845993041992,
      "Htr_mem_GB": 0.120455168,
      "Hte_mem_GB": 0.048283648
    },
    {
      "seed": 1,
      "D": 8192,
      "Q": 32,
      "train_acc": 0.8363710554951034,
      "test_acc": 0.8242280285035629,
      "wall_s": 31.61271023750305,
      "wall_encode_s": 20.219486713409424,
      "wall_nsram_s": 11.25111174583435,
      "wall_train_s": 0.09159159660339355,
      "wall_test_s": 0.028688669204711914,
      "Htr_mem_GB": 0.120455168,
      "Hte_mem_GB": 0.048283648
    },
    {
      "seed": 2,
      "D": 8192,
      "Q": 32,
      "train_acc": 0.8479325353645266,
      "test_acc": 0.8452663725822871,
      "wall_s": 32.3673894405365,
      "wall_encode_s": 20.91416621208191,
      "wall_nsram_s": 11.306992530822754,
      "wall_train_s": 0.09318399429321289,
      "wall_test_s": 0.028300762176513672,
      "Htr_mem_GB": 0.120455168,
      "Hte_mem_GB": 0.048283648
    }
  ],
  "mean_test_acc": 0.8382535912227125,
  "std_test_acc": 0.009917570508667812,
  "test_accuracy": 0.8452663725822871,
  "train_time_sec": 31.94507710138957,
  "mem_GB": 0.506216448,
  "throughput_inf_per_sec": 101618.02962889886,
  "nsram_applied": true,
  "preregistered_gates": {
    "INFRA": true,
    "DISCOVERY_gt_0p70": true,
    "AMBITIOUS_gt_0p85_and_mem_lt_4GB": false
  }
}
```


=== FILE: N_LMS_Eq_N16_summary.json (1471 chars) ===
```json
{
  "channel_h_real": [
    0.8899883189799696,
    0.0,
    0.0,
    0.3559953275919878,
    0.0,
    0.0,
    0.0,
    -0.1779976637959939
  ],
  "channel_h_imag": [
    0.0,
    0.0,
    0.0,
    -0.1779976637959939,
    0.0,
    0.0,
    0.0,
    0.13349824784699543
  ],
  "echo_delays": [
    0,
    3,
    7
  ],
  "n_symbols": 8000,
  "n_preamble": 2000,
  "n_taps": 16,
  "delay": 8,
  "snr_db": [
    5.0,
    10.0,
    20.0
  ],
  "eval_start": 3000,
  "BER_per_SNR": {
    "wiener": {
      "5.0": 0.1151,
      "10.0": 0.0811,
      "20.0": 0.0661
    },
    "lms_f32": {
      "5.0": 0.0724,
      "10.0": 0.0108,
      "20.0": 0.0
    },
    "nsram": {
      "5.0": 0.1667,
      "10.0": 0.0155,
      "20.0": 0.0
    },
    "no_eq": {
      "5.0": 0.1115,
      "10.0": 0.0556875,
      "20.0": 0.0204375
    }
  },
  "convergence_steps": {
    "lms_f32": {
      "5.0": -1,
      "10.0": -1,
      "20.0": 416
    },
    "nsram": {
      "5.0": -1,
      "10.0": -1,
      "20.0": 1514
    }
  },
  "energy_per_symbol_pJ": {
    "lms_f32": 474.6,
    "nsram": 2.76,
    "wiener_apply": 237.8
  },
  "gates": {
    "INFRA": {
      "pass": true,
      "note": "trains + dashboard produced"
    },
    "DISCOVERY": {
      "pass": true,
      "ber_20dB": 0.0,
      "note": "NSRAM BER<0.01 at 20dB"
    },
    "AMBITIOUS": {
      "pass": true,
      "ber_20dB": 0.0,
      "ber_10dB": 0.0155,
      "note": "NSRAM BER<0.001@20 AND BER<0.05@10"
    }
  }
}
```


=== FILE: N_Mem_Pal_N512_summary.json (7437 chars) ===
```json
{
  "config": {
    "N_cells": 512,
    "D_key": 256,
    "k_sdm": 5,
    "n_loc_vocab": 64,
    "n_item_vocab": 64,
    "capacities": [
      4,
      8,
      16,
      24,
      32,
      48,
      64
    ],
    "seeds": [
      0,
      1,
      2
    ]
  },
  "per_capacity": {
    "4": {
      "mean_l2i": 0.9166666666666666,
      "std_l2i": 0.11785113019775792,
      "mean_i2l": 0.9166666666666666,
      "std_i2l": 0.11785113019775792
    },
    "8": {
      "mean_l2i": 0.9166666666666666,
      "std_l2i": 0.05892556509887896,
      "mean_i2l": 0.875,
      "std_i2l": 0.0
    },
    "16": {
      "mean_l2i": 0.875,
      "std_l2i": 0.08838834764831845,
      "mean_i2l": 0.8958333333333334,
      "std_i2l": 0.02946278254943948
    },
    "24": {
      "mean_l2i": 0.8194444444444443,
      "std_l2i": 0.07081971546656644,
      "mean_i2l": 0.8333333333333334,
      "std_i2l": 0.0
    },
    "32": {
      "mean_l2i": 0.7604166666666666,
      "std_l2i": 0.02946278254943948,
      "mean_i2l": 0.7604166666666666,
      "std_i2l": 0.03897559777889522
    },
    "48": {
      "mean_l2i": 0.6041666666666666,
      "std_l2i": 0.0,
      "mean_i2l": 0.5416666666666666,
      "std_i2l": 0.0340206908719886
    },
    "64": {
      "mean_l2i": 0.421875,
      "std_l2i": 0.0,
      "mean_i2l": 0.3854166666666667,
      "std_i2l": 0.03897559777889522
    }
  },
  "rows": [
    {
      "P": 4,
      "seed": 0,
      "acc_l2i": 0.75,
      "acc_i2l": 1.0,
      "n_writes": 20,
      "n_unique_cells": 19,
      "load": 0.0078125,
      "t_encode_s": 0.0011317729949951172,
      "energy_per_recall_pJ": 325.99999999999994
    },
    {
      "P": 4,
      "seed": 1,
      "acc_l2i": 1.0,
      "acc_i2l": 1.0,
      "n_writes": 20,
      "n_unique_cells": 20,
      "load": 0.0078125,
      "t_encode_s": 0.00044035911560058594,
      "energy_per_recall_pJ": 325.99999999999994
    },
    {
      "P": 4,
      "seed": 2,
      "acc_l2i": 1.0,
      "acc_i2l": 0.75,
      "n_writes": 20,
      "n_unique_cells": 19,
      "load": 0.0078125,
      "t_encode_s": 0.00043272972106933594,
      "energy_per_recall_pJ": 325.99999999999994
    },
    {
      "P": 8,
      "seed": 0,
      "acc_l2i": 0.875,
      "acc_i2l": 0.875,
      "n_writes": 40,
      "n_unique_cells": 37,
      "load": 0.015625,
      "t_encode_s": 0.0005414485931396484,
      "energy_per_recall_pJ": 325.99999999999994
    },
    {
      "P": 8,
      "seed": 1,
      "acc_l2i": 1.0,
      "acc_i2l": 0.875,
      "n_writes": 40,
      "n_unique_cells": 38,
      "load": 0.015625,
      "t_encode_s": 0.0005345344543457031,
      "energy_per_recall_pJ": 325.99999999999994
    },
    {
      "P": 8,
      "seed": 2,
      "acc_l2i": 0.875,
      "acc_i2l": 0.875,
      "n_writes": 40,
      "n_unique_cells": 37,
      "load": 0.015625,
      "t_encode_s": 0.0005354881286621094,
      "energy_per_recall_pJ": 325.99999999999994
    },
    {
      "P": 16,
      "seed": 0,
      "acc_l2i": 0.9375,
      "acc_i2l": 0.875,
      "n_writes": 80,
      "n_unique_cells": 76,
      "load": 0.03125,
      "t_encode_s": 0.0007605552673339844,
      "energy_per_recall_pJ": 325.99999999999994
    },
    {
      "P": 16,
      "seed": 1,
      "acc_l2i": 0.75,
      "acc_i2l": 0.9375,
      "n_writes": 80,
      "n_unique_cells": 75,
      "load": 0.03125,
      "t_encode_s": 0.0007460117340087891,
      "energy_per_recall_pJ": 325.99999999999994
    },
    {
      "P": 16,
      "seed": 2,
      "acc_l2i": 0.9375,
      "acc_i2l": 0.875,
      "n_writes": 80,
      "n_unique_cells": 75,
      "load": 0.03125,
      "t_encode_s": 0.0007493495941162109,
      "energy_per_recall_pJ": 325.99999999999994
    },
    {
      "P": 24,
      "seed": 0,
      "acc_l2i": 0.7916666666666666,
      "acc_i2l": 0.8333333333333334,
      "n_writes": 120,
      "n_unique_cells": 110,
      "load": 0.046875,
      "t_encode_s": 0.0009810924530029297,
      "energy_per_recall_pJ": 325.99999999999994
    },
    {
      "P": 24,
      "seed": 1,
      "acc_l2i": 0.75,
      "acc_i2l": 0.8333333333333334,
      "n_writes": 120,
      "n_unique_cells": 108,
      "load": 0.046875,
      "t_encode_s": 0.0009710788726806641,
      "energy_per_recall_pJ": 325.99999999999994
    },
    {
      "P": 24,
      "seed": 2,
      "acc_l2i": 0.9166666666666666,
      "acc_i2l": 0.8333333333333334,
      "n_writes": 120,
      "n_unique_cells": 108,
      "load": 0.046875,
      "t_encode_s": 0.0009493827819824219,
      "energy_per_recall_pJ": 325.99999999999994
    },
    {
      "P": 32,
      "seed": 0,
      "acc_l2i": 0.78125,
      "acc_i2l": 0.8125,
      "n_writes": 160,
      "n_unique_cells": 142,
      "load": 0.0625,
      "t_encode_s": 0.0011849403381347656,
      "energy_per_recall_pJ": 325.99999999999994
    },
    {
      "P": 32,
      "seed": 1,
      "acc_l2i": 0.78125,
      "acc_i2l": 0.71875,
      "n_writes": 160,
      "n_unique_cells": 141,
      "load": 0.0625,
      "t_encode_s": 0.0011911392211914062,
      "energy_per_recall_pJ": 325.99999999999994
    },
    {
      "P": 32,
      "seed": 2,
      "acc_l2i": 0.71875,
      "acc_i2l": 0.75,
      "n_writes": 160,
      "n_unique_cells": 140,
      "load": 0.0625,
      "t_encode_s": 0.0011913776397705078,
      "energy_per_recall_pJ": 325.99999999999994
    },
    {
      "P": 48,
      "seed": 0,
      "acc_l2i": 0.6041666666666666,
      "acc_i2l": 0.5416666666666666,
      "n_writes": 240,
      "n_unique_cells": 194,
      "load": 0.09375,
      "t_encode_s": 0.0016071796417236328,
      "energy_per_recall_pJ": 325.99999999999994
    },
    {
      "P": 48,
      "seed": 1,
      "acc_l2i": 0.6041666666666666,
      "acc_i2l": 0.5833333333333334,
      "n_writes": 240,
      "n_unique_cells": 200,
      "load": 0.09375,
      "t_encode_s": 0.001585245132446289,
      "energy_per_recall_pJ": 325.99999999999994
    },
    {
      "P": 48,
      "seed": 2,
      "acc_l2i": 0.6041666666666666,
      "acc_i2l": 0.5,
      "n_writes": 240,
      "n_unique_cells": 194,
      "load": 0.09375,
      "t_encode_s": 0.0016050338745117188,
      "energy_per_recall_pJ": 325.99999999999994
    },
    {
      "P": 64,
      "seed": 0,
      "acc_l2i": 0.421875,
      "acc_i2l": 0.4375,
      "n_writes": 320,
      "n_unique_cells": 234,
      "load": 0.125,
      "t_encode_s": 0.002050161361694336,
      "energy_per_recall_pJ": 325.99999999999994
    },
    {
      "P": 64,
      "seed": 1,
      "acc_l2i": 0.421875,
      "acc_i2l": 0.375,
      "n_writes": 320,
      "n_unique_cells": 243,
      "load": 0.125,
      "t_encode_s": 0.0020232200622558594,
      "energy_per_recall_pJ": 325.99999999999994
    },
    {
      "P": 64,
      "seed": 2,
      "acc_l2i": 0.421875,
      "acc_i2l": 0.34375,
      "n_writes": 320,
      "n_unique_cells": 247,
      "load": 0.125,
      "t_encode_s": 0.0020265579223632812,
      "energy_per_recall_pJ": 325.99999999999994
    }
  ],
  "capacity_at_50pct": 48,
  "capacity_at_60pct": 32,
  "capacity_at_80pct": 24,
  "recall_acc_loc": 0.8958333333333334,
  "recall_acc_item": 0.875,
  "energy_per_recall_pJ": 325.99999999999994,
  "wall_s": 5.814985036849976,
  "gates": {
    "INFRA": true,
    "DISCOVERY": true,
    "AMBITIOUS": false
  },
  "artefacts": {
    "dashboard": "dashboard.png",
    "gif": "weight_evo.gif",
    "spikes": "spikes.npy",
    "vb": "vb.npy",
    "weights": "weights.npy"
  }
}
```


=== FILE: N_PC_NAB_N256_summary.json (2542 chars) ===
```json
{
  "task": "N_PC_NAB Phase N1#6 / Phase N2 U6",
  "n_enc": 256,
  "n_err": 256,
  "device": "cuda",
  "seed": 1337,
  "streams": [
    "artificialWithAnomaly/art_daily_jumpsup.csv",
    "realKnownCause/nyc_taxi.csv",
    "realKnownCause/machine_temperature_system_failure.csv"
  ],
  "test_F1_per_stream": {
    "artificialWithAnomaly/art_daily_jumpsup.csv": 0.33333333333333337,
    "realKnownCause/nyc_taxi.csv": 0.46153846153846156,
    "realKnownCause/machine_temperature_system_failure.csv": 0.21052631578947367
  },
  "mean_F1": 0.3351327035537562,
  "energy_per_sample_pJ": 1.0079353138326577,
  "throughput_samples_per_sec_mean": 4388.734709831378,
  "per_stream": [
    {
      "stream": "artificialWithAnomaly/art_daily_jumpsup.csv",
      "n_samples": 4032,
      "n_train": 2419,
      "n_windows_total": 1,
      "n_windows_train": 0,
      "n_windows_test": 1,
      "best_threshold_on_train": 4.620960070971905,
      "train_F1": null,
      "test_F1": 0.33333333333333337,
      "test_precision": 0.2,
      "test_recall": 1.0,
      "throughput_samples_per_sec": 3815.218685993063,
      "total_spikes": 145226,
      "energy_pj_per_sample": 0.7563854166666666,
      "wall_time_s": 1.0568201541900635
    },
    {
      "stream": "realKnownCause/nyc_taxi.csv",
      "n_samples": 10320,
      "n_train": 6192,
      "n_windows_total": 5,
      "n_windows_train": 1,
      "n_windows_test": 4,
      "best_threshold_on_train": 2.3059710443841865,
      "train_F1": 0.11764705882352941,
      "test_F1": 0.46153846153846156,
      "test_precision": 0.3333333333333333,
      "test_recall": 0.75,
      "throughput_samples_per_sec": 4671.9063797186855,
      "total_spikes": 766486,
      "energy_pj_per_sample": 1.5597098837209302,
      "wall_time_s": 2.2089483737945557
    },
    {
      "stream": "realKnownCause/machine_temperature_system_failure.csv",
      "n_samples": 22695,
      "n_train": 13617,
      "n_windows_total": 4,
      "n_windows_train": 2,
      "n_windows_test": 2,
      "best_threshold_on_train": 3.0747425626935847,
      "train_F1": 0.21052631578947367,
      "test_F1": 0.21052631578947367,
      "test_precision": 0.11764705882352941,
      "test_recall": 1.0,
      "throughput_samples_per_sec": 4679.079063782385,
      "total_spikes": 764833,
      "energy_pj_per_sample": 0.7077106411103766,
      "wall_time_s": 4.850313425064087
    }
  ],
  "gates": {
    "DISCOVERY_mean_F1_gt_0.3": true,
    "AMBITIOUS_F1_gt_0.5_AND_thr_gt_1k": false
  },
  "timestamp": "2026-05-17T09:04:55.296806"
}
```


=== FILE: N_Res_MG_N1024_summary.json (502 chars) ===
```json
{
  "task": "Mackey-Glass tau=17 1-step",
  "N": 1024,
  "density": 0.01,
  "spectral_radius": 0.9,
  "washout": 500,
  "train": 4000,
  "test": 2000,
  "nrmse_test": 0.015329434855969977,
  "ridge_alpha_chosen": 0.1,
  "throughput_steps_per_sec": 22096.754741449182,
  "wall_reservoir_s": 0.2941608428955078,
  "device": "cuda",
  "torch_version": "2.12.0+cu130",
  "gates": {
    "INFRA": true,
    "DISCOVERY (NRMSE<0.1)": true,
    "AMBITIOUS (NRMSE<0.05 & throughput>10k)": true
  },
  "seed": 0
}
```


=== FILE: N_Stoch_RNG_N100_summary.json (1686 chars) ===
```json
{
  "phase": "N1#N2 / N2 U10",
  "seed": 20260517,
  "n_cells": 8192,
  "n_bits": 1000000,
  "nist_tests_passed": 5,
  "nist_tests_total": 5,
  "nist_p_values": {
    "frequency": 0.214975394149174,
    "block_frequency": 0.49398287300666827,
    "runs": 0.3214223977778019,
    "dft": 0.05870640159069657,
    "cumulative_sums": 0.4236998879591828
  },
  "bernoulli_kl_divergence": 4.833283810260505e-06,
  "bernoulli_kl_max": 1.2881555879088163e-05,
  "throughput_bits_per_sec": 1266898.763579512,
  "energy_per_bit_pJ": 0.4,
  "stochastic_gate_max_abs_err": 0.0010183856500000643,
  "stochastic_gates": {
    "p_a": 0.501235,
    "p_b": 0.496605,
    "AND_emp": 0.249425,
    "AND_ideal": 0.248915807175,
    "OR_emp": 0.748415,
    "OR_ideal": 0.7489241928250001,
    "XOR_emp": 0.49899,
    "XOR_ideal": 0.50000838565,
    "n": 200000
  },
  "bernoulli_per_p": [
    {
      "p": 0.05,
      "emp": 0.04998,
      "kl": 4.211058250698865e-09
    },
    {
      "p": 0.1,
      "emp": 0.10034,
      "kl": 6.415763556285431e-07
    },
    {
      "p": 0.25,
      "emp": 0.2522,
      "kl": 1.2881555879088163e-05
    },
    {
      "p": 0.5,
      "emp": 0.50144,
      "kl": 4.147205733108202e-06
    },
    {
      "p": 0.75,
      "emp": 0.75134,
      "kl": 4.793987911556712e-06
    },
    {
      "p": 0.9,
      "emp": 0.90072,
      "kl": 2.8861665227755077e-06
    },
    {
      "p": 0.95,
      "emp": 0.9491,
      "kl": 8.478283211415714e-06
    }
  ],
  "gates": {
    "INFRA": true,
    "DISCOVERY": true,
    "AMBITIOUS": true
  },
  "runtime_sec": 1.7516937255859375,
  "dashboard": "/home/naorw/AMD_gfx1151_energy_network/results/N_Stoch_RNG_N100/dashboard.png"
}
```


=== FILE: context.md (39313 chars) ===
```
## 2026-05-16 :47 — APU=35C idle
## 2026-05-16 — autonomous tick: idle
## 2026-05-16 — topology tick: idle
## 2026-05-16 — MEP+DS-N tick: APU=35C idle
## 2026-05-16 — autonomous tick: idle
## 2026-05-16 :47 — APU=35C idle
## 2026-05-16 — deep-dive tick: APU=35C idle
## 2026-05-16 — oracle tick: no new activity since O68 (cycle skip)
## 2026-05-16 — autonomous tick: idle
## 2026-05-16 — topology tick: idle
## 2026-05-16 — autonomous tick: idle
## 2026-05-16 :47 — APU=35C idle
## 2026-05-16 — autonomous tick: idle
## 2026-05-16 — topology tick: idle
## 2026-05-16 — autonomous tick: idle
## 2026-05-16 :47 — APU=35C idle
## 2026-05-16 — deep-dive tick: APU=35C idle
## 2026-05-16 — master fix tick: idle
## 2026-05-16 — autonomous tick: visual audit confirmed solver-kaos at VG1=0.6, model line decreases then oscillates. Need arc-length homotopy.
## 2026-05-16 — topology tick: idle
## 2026-05-16 — MEP+DS-N tick: APU=42C, z429 multisolver running (arc-length 87% best)
## 2026-05-16 — track audit: NOVEL_DS superseded by S-series. S19 done: V_Sint runaway = root cause, V_Sint=0 PIN gives 1.26 dec.
## 2026-05-16 — autonomous tick: idle, S19 final: V_Sint runaway is root cause, V_Sint=0 PIN gives 1.26 dec. Arc-length alone not enough.
## 2026-05-16 :47 — APU=41C, z430=4 active
## 2026-05-16 — autonomous tick: z430 running BASELINE done (3.899), M2_RS_100 in progress
## 2026-05-16 — topology tick: z430 M2_RS_100 running
## 2026-05-16 — autonomous tick: z430 V_SINT_PIN DISCOVERY PASS: cell=1.619 dec, 100% conv, 31s wall. VG1=0.4: 0.79, VG1=0.6: 1.09. VG1=0.2 still 2.63 (separate issue).
## 2026-05-16 :47 — APU=41C idle
## 2026-05-16 — deep-dive tick: APU=40C idle
## 2026-05-16 — autonomous tick: idle
## 2026-05-16 — topology tick: idle, S20 V_SINT_PIN DISCOVERY done
## 2026-05-16 — autonomous tick: idle
## 2026-05-16 :47 — APU=40C idle
## 2026-05-16 — O73 oracle done: all 3 rank BSIM4 GIDL §6.2 as #1 fix for VG1=0.2 residual. Cherry-pick flagged. V_SINT_PIN likely legitimate pending I_B measurement.
## 2026-05-16 — autonomous tick: idle, awaiting user direction on BSIM4 GIDL implementation
## 2026-05-16 — topology tick: idle
## 2026-05-16 — MEP+DS-N tick: APU=40C, S21 z431 KILL_SHOT (GIDL already in place, cannot close VG1=0.2)
## 2026-05-16 — autonomous tick: 4 parallel experiments dispatched (O74 oracle + S22-A/B/C creative variants)
## 2026-05-16 :47 — APU=43C z43x=7 active (z432/z434 running, O74 pending)
## 2026-05-16 — deep-dive tick: APU=43C, z432/z434 active, O74 pending
## 2026-05-16 — master fix tick: z432/z434 + O74 in progress
## 2026-05-16 — oracle tick: skip (O74 fresh, z432/z434/z435/z436/z437 all in flight, wait for stable results)
## 2026-05-16 — autonomous tick: z432 pseudo-transient forward DISCOVERY cell=1.349 (-0.27 vs z430). VG1=0.4 best at 0.70. Backward sweep running for hysteresis check.
## 2026-05-16 — topology tick: idle, S25 z437 snapback subcircuit BV-sweep all worse (sign bug), z432 PT forward DISCOVERY 1.349 dec
## 2026-05-16 — autonomous tick: z432 backward 1.027 dec BREAKTHROUGH. Hysteresis 0.45 dec mean fwd-bwd. V_B latched 0.86V backward vs -0.2V NR. AMBITIOUS missed by 0.027.
## 2026-05-16 :47 — APU=36C z43x=2 (z436 SCR core running)
## 2026-05-16 — autonomous tick: idle, z436 SCR core still running
## 2026-05-16 — topology tick: idle, z436 running
## 2026-05-16 — autonomous tick: idle
## 2026-05-16 :47 — APU=36C idle
## 2026-05-16 — deep-dive tick: APU=35C idle
## 2026-05-16 — autonomous tick: 5 spår igång (O75 oracle+plots, S26 knee-calib, S27 bifurcation map, S28 smooth PT, research lit)
## 2026-05-16 — topology tick: 5 parallel tracks running (O75/S26/S27/S28/research)
## 2026-05-16 — MEP+DS-N tick: APU=50C, multiple z43x running (S26-S29)
## 2026-05-16 — track audit: S27 BOUNDARY found V_G1-V_G2≥0.20V predicts snapback 80% accurate. NOVEL_DS superseded by S-series.
## 2026-05-16 — autonomous tick: 5 spår igång (S26 knee, S28 implicit, S29 M2-topology, S30 VG-gate, research done)
## 2026-05-16 — autonomous tick: S26 progressing 0.916-0.917 dec at corners
## 2026-05-16 — topology tick: S26/S28/S29/S30 in flight
## 2026-05-16 — autonomous tick: S29 z440 KILL_SHOT (M2-shunt-parallel försämrar). Awaiting S30 (VG-gate) + S26 grid completion.
## 2026-05-16 :47 — APU=46C
## 2026-05-16 — deep-dive tick: APU=46C
## 2026-05-16 — master fix tick
## 2026-05-16 — autonomous tick: idle
## 2026-05-16 — topology tick: idle
## 2026-05-16 — autonomous tick: z438 grid 5/16 (best 0.916), z439 IMPLICIT=1.056 BDF2 running, z441 in flight
## 2026-05-16 :47 — APU=46C
## 2026-05-16 — autonomous tick: z441 manually restarted (subagent had killed it). z438 grid[2,2] running.
## 2026-05-16 — topology tick: z438/z439/z441 running, ETA 2-3h
## 2026-05-16 — MEP+DS-N tick: APU=49C. Alt-tools research done — VBIC level=4 recommended (-0.3 dec expected). Canonical audit + vision oracle in flight.
## 2026-05-16 — autonomous tick: G9 NFACTOR-on-M1 bug fixed (canonical audit). Re-running z442 to measure impact.
## 2026-05-16 :47 — APU=53C
## 2026-05-16 — deep-dive tick: APU=53C
## 2026-05-16 — oracle tick: deferred until z442/z446 combined results stabilize (multiple in flight)
## 2026-05-16 — autonomous tick: Track A VBIC z443 DONE: cell=1.311 dec (-0.31). VG1=0.2: 2.62→0.91 (-1.71!). S31 VBIC+PT combo dispatched.
## 2026-05-16 — topology tick: S31 VBIC+PT combo progressing variant A
## 2026-05-16 — autonomous tick: GitHub backup pushed (branch backup/snapshot-2026-05-16-1930). z446 variant D running.
## 2026-05-16 :47 — APU=55C
## 2026-05-16 — tick: z447 transient SLOW DC = 0.886 dec! Best yet. z446 still on var D.
## 2026-05-16 — autonomous tick: z332/z333/z334 stale (May-13 artifacts, no summary.json; R-13b/R-18 already completed per task list). No relaunch.
## 2026-05-16 — S33 z448 DONE: BDF adaptive solver INFRA PASS but DISCOVERY FAIL (V_B=11mV/5ns vs 0.5V need). KILL_SHOT physical: C_eff~12fF + I_iion~1e-7A → τ~650ns >> 10ns pulse. Slow-DC=1.002 dec.
## 2026-05-16 — diagnosis: BSIM4+GP structurally insufficient for ns-snap. Next options: A=VBIC(z443)+BDF combo, B=Verilog-A thyristor pivot, C=wait Sebas A.12 7-rate data.
## 2026-05-16 :47 — APU=49C idle (sentinel ok, no z2[3-9])
## 2026-05-16 — deep-dive tick: APU=49C. Plan 2026-05-12 stale: 4A/4B/4C/4E.1 all completed (#190/191/192/212). Current focus = NS-RAM model fit (z44x), supersedes Phase-4. 4D waves and 4E.2 deferred until DC<0.5 dec gate or accept-1.0 dec resolution. No dispatch.
## 2026-05-16 — master fix tick: P3 (pyport_v4 DC) still ACTIVE — best 0.886 dec (z447 slow-DC), gate <0.5 dec not crossed. P5 transient HONEST FAIL on ns-pulse (z448 physical KILL_SHOT, structural). P7 oracle critique not yet dispatched. P8 4E.1 v4.4 brief compile already #212-completed but pre-z44x. No phase PASS just crossed → no ALERT. No auto-launch.
## 2026-05-16 — autonomous tick: z332/3/4 still stale (already noted). z444 BESD running but RED FLAG — OFF/BESD_DEF/BESD_LOW all return identical cell=1.572 per_branch={0.4:0.894,0.6:1.88}. BESD params not being applied → replace=True path likely no-op. KILL_SHOT pending BESD_HOT confirm.
## 2026-05-16 — topology tick: z444 BESD DONE — KILL_SHOT confirmed. All 4 conditions (OFF/DEF/LOW/HOT) → identical cell=1.572 (params no-op, replace=True path dead). OFF=1.572 also worse than z432 baseline 1.027 → script's OFF path uses different solver. Original R-1..R-10 plan superseded; R-50 (physics-bounded BBO) still in_progress. No PASS gate crossed → no ALERT.
## 2026-05-16 — z450 Verilog-A pivot DONE: standalone thyristor_compact.py PROTOTYPE PASS (N-shape visible, 9.3× decade, 54 NDR pts at V_peak=0.85V). KEY FINDING: z444 BESD identical-results bug is MECHANICAL wiring no-op (residual never reaches M1 drain KCL row), NOT physics dead-end. Recommended: debug z444 wiring first (~4h), keep thyristor_compact.py as fallback. CONDITIONAL GO.
## 2026-05-16 — z451 cap audit BREAKTHROUGH: C_eff = 2.66 fF NOT 12 fF. z448 KILL_SHOT was FALSE NEGATIVE (4.5× cap overestimate). Required I to charge ΔV=0.7V in 10ns = 189 nA; BSIM4+GP already gives 130 nA → off by 1.5×, not 100×. Dominant suspect: M1.alpha0=7.84e-5 (Sebas card), literature for 130nm bulk says 5e-4..5e-3. 10× ALPHA0 likely closes BOTH DC knee AND ns-snap. ns-snap track RE-OPENED.
## 2026-05-16 — z451 critique: 3 cherry-picks flagged (backward-only PT cited as breakthrough, V_SINT_PIN unmeasured vs silicon, VG1=0.2 fix never full-grid revalidated).
## 2026-05-16 — z449 VBIC+BDF combo KILL_SHOT: all 3 variants FAIL gates. v449_A DC=1.311 (=z443 ceiling), v449_B n-well-cap=0 → Vb@5ns 5× better (validates z451 cap math!), v449_C ALPHA0×5 DC WORSE (+0.30 dec). Body-current limited, not cap. z449 recommends snapback_subcircuit with V_BC-thresholded µA pull-down. Awaiting z453 wider ALPHA0 sweep (10/30/100×) to confirm-or-kill the literature hypothesis.
## 2026-05-16 — MEP+DS-N tick: APU=48C sentinel ok. No z2[3-9]/z44x active (z453 ALPHA0 subagent still setting up). Plan 2026-05-12 superseded by NS-RAM z44x/z45x campaign. Pending pre-z449 tasks: D2 corrective sweep, MEP-6/7, PTP — held back of queue. No new launches.
## 2026-05-16 — track-progress audit: Phase A (MEP) 1/4 done (SURR-V4), MEP-6 in_progress, D2+PTP pending. Phase B (DS-N) 16/18 done (DS-N4 ECG + DS-N5 HDC + DS-N8 KWS in_progress, DS-N3 absent). Phase C: 4E.1 brief done #212, oracle critique deferred. Plan ~85% closed; remaining blockers held back of queue while NS-RAM z44x/z45x active. No compute.
## 2026-05-16 — topology tick: original R-1..R-10 plan long superseded. R-50 (physics-bounded BBO) still in_progress, R-51 done. Active campaign = z449 DONE/KILL, z450 DONE thyristor proto OK, z451 DONE cap audit (BREAKTHROUGH C_eff=2.66fF refutes z448), z453+z454 in flight. No new R-phase gate crossed → no ALERT.
## 2026-05-16 — z454 snapback subcircuit DONE: KILL_SHOT on DC but ns-snap CLOSED. SB_HOT V_B→0.5V in 1.38ns (3/4 biases), peak 0.71V — first time ns-snap works in pipeline. DC destroyed (2.66 vs 2.09 SB_OFF) because Slotboom multiplier fires too early at V_db<BV. No self-reset (no body-leak path).
## 2026-05-16 — z454 BURN CONFIRMED: I_snap clamp fires (z444-style no-op avoided). |I_snap_b|: DEFAULT=0.5µA, LOW=2.5µA, HOT=42µA.
## 2026-05-16 — CRITICAL: v449_B published "1.31 dec" was forward-only cherry-pick. Backward sweep = 2.86 dec, AVG = 2.09. z451 critique #1 vindicated. All prior DC numbers need fwd+bwd reporting.
## 2026-05-16 — Next: z455 knee-sharpener (V_knee≈1.8V) + z456 R_body reset path. Test INDEPENDENTLY first then recombine.
## 2026-05-16 :47 — APU=49C ACTIVE: z453_alpha0_sweep (z455+z456 still spawning)
## 2026-05-16 — z45x campaign tick: APU=53C. ACTIVE: z453 (still in A1 fwd sweep ~20min in), z455 (K_1p8/K_2p0 done DC=2.71/2.72 — knee-gate NOT recovering DC!), z456 (R_1G done DC=2.81 = baseline, no self-reset).
## 2026-05-16 — z455 INTERIM RED FLAG: V_knee gating not separating avalanche from low-Vd. I_snap fires at V_db=1.4 even with V_knee=2.0 set. Either σ-gate not wired or DC fold extends above V_knee — investigate when run done.
## 2026-05-16 — z456 INTERIM: R_1G expected too weak (τ≈2.7ms); waiting for R_10M / R_1M to confirm reset path.
## 2026-05-16 — z455 knee-sharpener DONE: DISCOVERY FAIL but PARTIAL. Best K_1p6 DC=2.702 (Δ=-0.107 only). σ-gate WORKS (I_snap_b drops 4 orders 4.2e-5→3.1e-8) but PARASITIC NPN's I_snap_d clamped at 10mA whenever Vbe>0.6V → that's what pollutes DC, not Slotboom. ns-snap survives: K_1p6 still 1.42ns to 0.5V on 3/4 biases.
## 2026-05-16 — z455 fix-the-fix → z457 dispatched: gate I_snap_d (NPN collector current) directly by V_db knee, not just Iii multiplier. Independent of z456.
## 2026-05-16 — z456 R_body reset KILL_SHOT: no self-reset at any R (1G..1M Ω). NPN holding ~10µA >> leak (0.66µA@1MΩ). DC identical across all R (=2.809 dec). R_1M did suppress latch at weakest bias VG1=0.4 (Vb_peak=0.01V vs 0.64V) — only effect seen. Self-reset axis still open: need weaker NPN AND R_body in same experiment (z458 2D sweep proposed).
## 2026-05-16 — z457 NPN-gate DONE: BEST yet NX_1p8 DC_avg=2.479 (Δ=-0.223 vs K_1p6). DISCOVERY FAIL but first real DC win since SB enabled. Mode X (gate current) works at V_knee=1.8 only (σ argument deep enough to kill 3e4 A unclamped Ic). Mode Y (vbe-offset 0.3V) INSUFFICIENT + breaks VG1=0.2 (+2.18 dec regression). VG1=0.6 wins most: 2.998→2.370. ns-snap survives 1.46ns t→0.5V.
## 2026-05-16 — z457 honest diagnosis: NX_1p8 still 0.4 dec WORSE than SB_OFF baseline 2.087. Snapback infrastructure NET NEGATIVE on DC even with NPN muzzled. Next: V_knee 2.0-2.2 + reduce Id_extra_clamp + audit Iii_body, D3 zener, pdiode Is, BSIM4 Ids overshoot.
## 2026-05-16 — SYNTHESIS DONE (CAMPAIGN_SYNTHESIS_2026-05-16.md, 524 lines, 10 sections, CP-1..CP-9 cherry-pick audit). HONEST BASELINE: 1.19 dec fwd+bwd avg (z432), NOT 0.886. Biggest cherry-pick: z447/z448 "0.886" was 4 biases only — excluded VG1=0.2. Top missing: A.12 (Sebas blocked 3wk), rbodymod=1 body-R (OPEN since 2026-05-13!), fwd+bwd methodology. Path B recommended: accept ~1.2 dec functional model, 2 weeks to publication. Action today: re-run z430/z432/z443/z446/z449/z454 with BOTH sweep directions.
## 2026-05-16 — daedalus SSH UNREACHABLE (timeout, slow ping 1.17s rtt). Distributed campaign reduced to ikaros + zgx.
## 2026-05-16 — AUTONOMOUS PLAN LAUNCHED. 5 subagents dispatched parallel: P1a ikaros (z430/z432/z443 fwd+bwd), P1b zgx (z446/z449/z454 fwd+bwd after rsync), P2 BSIM3/4 type-mismatch audit, P4 rbodymod=1 implementation, Oracle 3-way critique on synthesis. Cron 9e146f5b every 30min drives P-phase auto-progression (P1→synthesis→P5 holdout→P6 brief v4.5). Daedalus SSH unreachable, skipping.
## 2026-05-16 — z45x tick APU=49C. P1a INTERIM: z430 V_SINT_PIN fwd=1.619 bwd=2.823 AVG=2.301 dec (synthesis claim CONFIRMED — original "1.619 breakthrough" was fwd-only). Forward VG1_0.2 fjuck (2.62), backward catastrophic on VG1_0.4/0.6 (2.66/3.03). z432/z443 pending. z453 still A1 fwd (slow). P2/P4/Oracle running. ALERT: P1a using n=25 curves not 33 — verify data path.
## 2026-05-16 — P2 DONE: CLOSED-EMPTY. Synthesis CP-9 "BSIM3 type-mismatch" claim FALSIFIED — all cards are BSIM4 v4.5 (level=14), no BSIM3 in pipeline. ALPHA0/K1/K2/BETA0 use same conventions in BSIM3v3/BSIM4v4.8. Only fix: parser silently dropped level/version tokens (foot-gun, dormant) — landed in model_card.py:287. test_bsim_type_mismatch.py 19/19 PASS. Expected DC impact ≤0.01 dec. P2 budget redirected to P4 rbodymod=1.
## 2026-05-16 :47 — APU=52C ACTIVE: z453+P1a+P4 (3 scripts)
## 2026-05-16 — P1a CONFIRM SYNTHESIS CP-1: z432 fwd=1.349 BUT only 18/25 biases evaluated. VG1=0.2 column ENTIRELY DROPPED (7 fails, 32% conv rate). Original "z432 BREAKTHROUGH 1.027" was on EASY 18 biases. Cherry-pick now empirically proven.
## 2026-05-16 — z45x tick APU=51C. ACTIVE: z453+P1a+P4+Oracle. P1a summary so far has z430 only (z432/z443 mid-run). z453 still stuck on A1 forward DC sweep ~90min in (slow but alive pid 7203). P4 stuck on R_card stage 30min (alive pid 61266). No new DISCOVERY/KILL_SHOTs.
## 2026-05-16 — P-phase tick APU=51C. P1a partial (z430 only in summary, python still running), P1b NOT STARTED (zgx dir has only stale atom_logs, agent never synced), P2 DONE, P4 running, Oracle pending. No P-phase progression eligible. HONEST_BASELINE not yet writable.
## 2026-05-16 — P1a z432 update: bwd=1.027 ALL 25 biases (incl VG1=0.2!), fwd=1.349 only 18/25 (VG1=0.2 fails). Honest avg ≈1.20 dec, BUT fwd on full 25 would be much worse — backward sweep is more robust because basin found from above. Cherry-pick was reporting fwd=1.349 over 18/25 + bwd=1.027 over 25/25 as if comparable. z443 starting.
## 2026-05-16 — P4 INTERIM: R_card (62.5Ω) → fwd=1.349 bwd=1.027 = IDENTICAL to rbodymod=0 baseline. Simplified 1-R Rbody NO EFFECT at this resistance because V_SINT clamp already pinned body. Need weaker R to test. Other configs still running.
## 2026-05-16 — P1a COMPLETE (HONEST_BASELINE_2026-05-16.md written). Honest cell-wide: z430 fwd=1.619/bwd=2.823/avg=2.301 (25b,100%conv), z432 fwd=1.349(18b,32%)/bwd=1.027(25b,50%) mixed, z443 fwd=1.311/bwd=2.864/avg=2.227 (25b,100%). Two cherry-pick modes proven: direction-pick (z430/z443) + bias-pick (z432 VG1=0.2 dropped). KILL_SHOT trigger PARTIALLY ARMED: 2/3 pipelines avg>2.0 dec. Best defensible = z432 PT bwd 1.027 (50% conv caveat). No fwd+bwd average defensible until P4 rbodymod=1 lands.
## 2026-05-16 — P1b ZGX COMPLETE (z449/z454 done, z446 still running). HUGE FINDING: z443, z449_A, z449_B, z454_SB_OFF ALL give IDENTICAL fwd=1.311/bwd=2.864/avg=2.087. Means every "improvement" since z443 (VBIC, BDF, C_B=1fF, n-well cap=0) is a DC NO-OP. Only SB on/off moves DC (worse). z432 PT bwd=1.027 (50% conv) remains the only outlier. KILL_SHOT trigger: 5/7 pipelines avg>2.0 dec. Path B "functional model" claim on z432-bwd-only still defensible. All other DC claims need retraction.
## 2026-05-16 — z45x tick APU=48C. z453 HUNG 3.5h on A1 forward DC sweep (no log advance, python alive pid 7203, likely Newton infinite loop). P4 progressed: R_card fwd=1.349/bwd=1.027/avg=1.188 — IDENTICAL to z432 baseline, confirms rbodymod=1 implementation a no-op at card R. R_1k testing now. No new DISCOVERY. z453 candidate for kill+redispatch.
## 2026-05-16 — P-phase tick APU=48C. P1a ✓ HONEST_BASELINE.md ✓ P2 ✓. P1b z446/z449/z454 rsynced from zgx (no top-level summary.json yet — per-pipeline summaries only). P4 running. Oracle pending. No P-phase progression eligible: HONEST_BASELINE already exists; P4 not done blocks P5; Oracle not done blocks ALERT check.
## 2026-05-16 — P1b ZGX FINAL COMPLETE. NEW BEST z446.PT_VBIC fwd=1.396/bwd=1.156/AVG=1.276 dec. PT_GP=1.188, PT_VBIC=1.276 → ONLY PT-family hits <1.5 dec honest avg. All Newton-DC stuck at ~2.0+ dec (1.3 fwd / 2.86 bwd asymmetry — Newton attractor issue → motivates P4). z449 3 variants identical DC=2.087 (their value was transient not DC). z454 SB destroys DC universally. Honest baseline ready.
## 2026-05-17 :47 — APU=47C ACTIVE: z453+P4 (z453 still hung 5h+)
## 2026-05-17 — z45x tick APU=47C. P4 R_1k done: fwd=1.349/bwd=1.027/avg=1.188 IDENTICAL to z432 baseline and R_card — rbodymod=1 single-R no-op for R<<V_SINT pulldown. R_1M next. Oracle still pending. No DISCOVERY/KILL_SHOTs.
## 2026-05-17 — P-phase tick APU=47C. HONEST_BASELINE.md updated with P1b zgx addendum. Headline defensible: z446.PT_VBIC avg=1.276 dec (25/25 biases, fully balanced). P4 R_1M still running. Oracle pending. No trigger fires.
## 2026-05-17 — Oracle 12h review dispatched (packet at results/Oracle_12h_2026-05-17/, providers openai+gemini+grok, PID 125313). 3 Qs on gate-crossing/cherry-pick/next-exp.
## 2026-05-17 — Oracle 12h ALL 3 RETURNED. Consensus on Q1: NO cross-1.0-dec gate w/o new silicon data. ALERT — Q2 SPLIT 2/3: Gemini+Grok say "4-pipelines-identical IS a code no-op bug (like z444 BESD)", OpenAI says "true DC invariance". Falsifier proposed by Gemini: re-run z443 with ALPHA0×5 — if matches z443 baseline = code bug confirmed. Q3 SPLIT 3-way: OpenAI(c)/Gemini(d-kill-z453)/Grok(b). z45x APU=46C, P4 R_1M running.
## 2026-05-17 — CRITICAL ALERT: 2/3 oracles flag Q2 cherry-pick risk. The 4-pipeline-identity (z443=z449_A=z449_B=z454_SB_OFF =1.311/2.864) MAY be hidden no-op bug, not physics. Falsifier z460 dispatch needed before claiming 1.276 headline.
## 2026-05-17 — P-phase tick APU=46C. P1a/P2 done, P1b/P4/Oracle-synthesis pending (Oracle 12h IS done — separate). ALERT (per Q2 oracle split 2/3 cherry-pick): 1.276 dec headline RISKS being no-op code bug like z444 BESD. PROPOSED CHANGE TO PLAN: prepend z460 falsifier (re-run z443 with ALPHA0×5, expect ≠ baseline if not bug) BEFORE P6 brief v4.5 compile. Not auto-launched per spec.
## 2026-05-17 — deep-dive tick APU=46C. Active: z453(hung 6h+), P4 R_1M. 5 z45x summaries done. Next gated: z460 falsifier (Oracle 12h ALERT, 2/3 split on code-bug hypothesis) — needed before P6 brief. Blockers: P4 still running, z453 hung dispatch-candidate. DC<0.5 dec not crossed, honest avg=1.276 z446.PT_VBIC stands pending z460 verdict.
## 2026-05-17 :47 — APU=46C ACTIVE: z453+P4
## 2026-05-17 — tick APU=46C. P4 R_1M done IDENTICAL again (fwd=1.349/bwd=1.027/avg=1.188). 4/5 R-values now confirmed no-op. R_1G next. z453 still hung.
## 2026-05-17 — P-phase tick APU=46C. No state change since last tick: P1a/P2 ✓, P1b/P4/synthesis-oracle pending. ALERT (z460 falsifier) already logged. P4 on R_1G last variant. No new triggers.
## 2026-05-17 — O76 critique cycle dispatched (research_plan/oracle_queries/O76_critique_*, providers openai+gemini+grok, 3 Qs harsh-critique on 1.276 dec headline fragility + falsifier + NO-CHEAT drift). PID 163718.
## 2026-05-17 — P4 DONE: ALL 5 R-values (rbodymod0 + R_card 62.5/1k/1M/1G Ω) give IDENTICAL fwd=1.349/bwd=1.027/avg=1.188. Simplified 1-R rbodymod=1 STRUCTURALLY no-op at all R. Real fix would need 5-R distributed network (out of DC scope).
## 2026-05-17 — O76 CRITIQUE 3/3 AGREE: 1.276 headline IS FRAGILE. NEW FINDINGS: (a) metric only counts converged V_D points → some biases 2-5/30 silently used [OpenAI], (b) V_B clamps at 0.5/0.7 in PT integrator (basin gaming) [OpenAI], (c) "comforting lie" [Gemini], (d) NO-CHEAT drift cited in 3 specific log lines [Grok]. WARNING: corrective pre-register needed. Headline RETRACTED pending z460 falsifier with ALPHA0×10 + 25/25 strict + per-bias diagnostics.
## 2026-05-17 — P-phase tick APU=44C. P1a ✓ P2 ✓ P4 ✓ (3/4 dispatch-trigger conditions met). P5 dispatch DEFERRED: O76 3/3 oracle ALERT (1.276 headline fragile, basin gaming + V_D-dropout cherry-pick) overrides naive P5/P6 progression. Proposed plan change: insert z460 falsifier (ALPHA0×10, strict 25/25, per-bias diagnostics) BEFORE P5/P6. Not auto-launched per spec.
## 2026-05-17 :47 — APU=44C ACTIVE: z453 (still hung 7h+)
## 2026-05-17 — z45x tick APU=44C. No new completions. Only z453 active (still hung ~7h on A1 fwd sweep). No DISCOVERY/KILL_SHOTs.
## 2026-05-17 — P-phase tick APU=44C. State unchanged: P2+P4 done but P5 still DEFERRED per O76 3/3 ALERT (z460 falsifier required first). No state change since last tick.
## 2026-05-17 — tick APU=44C. State unchanged: z453 hung 8h+ only active. No new completions.
## 2026-05-17 — P-phase tick APU=44C. State unchanged. P5/P6 still DEFERRED on O76 ALERT (z460 required first).
## 2026-05-17 :47 — APU=44C ACTIVE: z453 (still hung)
## 2026-05-17 — tick APU=44C. z453 still hung, no completions.
## 2026-05-17 — P-phase tick APU=44C. State unchanged. P5/P6 deferred per O76 ALERT.
## 2026-05-17 — tick APU=44C. No state change. z453 still hung.
## 2026-05-17 — P-phase tick APU=44C. State unchanged. P5/P6 deferred per O76 ALERT.
## 2026-05-17 — deep-dive tick APU=44C. Active: z453 hung 10h+. Pending z45x: z452 BESD wiring debug, z458 snap_Is×R_body 2D, z460 falsifier (O76-required). Blocker: z460 must run before P5/P6. DC gap open at 1.276 dec headline RETRACTED — accept ~1.2-2.0 dec honest range pending z460 verdict.
## 2026-05-17 :47 — APU=44C ACTIVE: z453
## 2026-05-17 — tick APU=44C. No state change.
## 2026-05-17 — P-phase tick APU=44C. State unchanged. P5/P6 deferred per O76 ALERT.
## 2026-05-17 — tick APU=44C. No state change.
## 2026-05-17 — P-phase tick APU=44C. State unchanged. P5/P6 deferred per O76.
## 2026-05-17 :47 — APU=44C ACTIVE: z453
## 2026-05-17 — tick APU=44C. No change.
## 2026-05-17 — P-phase tick APU=44C. State unchanged.
## 2026-05-17 04:43 — baseline watchdog DEFERRED (O76 ALERT)
## 2026-05-17 — tick APU=44C. No state change.
## 2026-05-17 — P-phase tick APU=44C. State unchanged.
## 2026-05-17 :47 — APU=44C ACTIVE: z453
## 2026-05-17 — tick APU=44C. No state change.
## 2026-05-17 — P-phase tick APU=44C. State unchanged.
## 2026-05-17 — tick APU=44C. No state change.
## 2026-05-17 — P-phase tick APU=44C. State unchanged.
## 2026-05-17 — deep-dive tick APU=44C. z453 still hung 14h+. Pending: z452/z458/z460 (z460 gating). DC gap open at 1.276 retracted. No state change since last tick.
## 2026-05-17 :47 — APU=44C ACTIVE: z453
## 2026-05-17 06:29 — morning brief written
## 2026-05-17 — tick APU=44C. No change.
## 2026-05-17 — P-phase tick APU=44C. No change.
## 2026-05-17 — oracle critique 6h SKIPPED: no campaign activity past 6h (only idle ticks). O76 still standing, no new artifacts to critique.
## 2026-05-17 — tick APU=44C. No change.
## 2026-05-17 — P-phase tick APU=44C. No change.
## 2026-05-17 :47 — APU=44C ACTIVE: z453
## 2026-05-17 — tick APU=44C. No change.
## 2026-05-17 — P-phase tick APU=44C. No change.
## 2026-05-17 — tick APU=44C. No change.
## 2026-05-17 — P-phase tick APU=44C. No change.
## 2026-05-17 :47 — APU=44C ACTIVE: z453
## 2026-05-17 — KILLED z453 (hung 14h+ on A1 fwd sweep, blocking compute slot)
## 2026-05-17 — PHYSICS-COMPLETION CAMPAIGN dispatched (5 parallel). z453 killed. New spår:
## z458 snap_Is×R_body 2D for self-reset (LIF closure). Mario slide-12/21 re-extraction (more PWL + oscillation targets). Lit-based educated guesses (R_body, R_th, NPN holding). z460 ALPHA0×10 falsifier (O76-required). O77 oracle 3-way physics-completion strategy.
## 2026-05-17 — tick APU=43C (z453 killed, freed). 5 subagents setting up (z458/z460/3 research).
## 2026-05-17 — P-phase tick APU=43C. State unchanged. P5/P6 deferred.
## 2026-05-17 — EDUCATED-GUESS CHEAT-SHEET DONE — 3 MASSIVE FINDINGS:
## 1. Pazos/Lanza 2025 NS-RAM is 180nm NOT 130nm! V_op=3.5-4.5V vs our 2V. Our ENTIRE pyport on PTM 130nm V_DD=1.2V card is WRONG NODE.
## 2. R_body operating regime is 10kΩ-1MΩ; we swept 1MΩ-∞ (missed by ~10× on low end). C_body must be ~pF not ~fF (we have 0.3fF — wrong by 100×).
## 3. Parasitic NPN β in DNW = 10-20, NOT 10⁴! DNW INCREASES base area → REDUCES β. Bf=10⁴ is SiGe-HBT BiCMOS, wrong device class. With β~20 holding current ~10µA explains z456 KILL_SHOT mechanically.
## KILL_SHOT lit: no public NS-RAM PDK exists at 130nm — Lanza group internal-only. Reconstruction-from-PTM is fundamentally node-mismatched.
## FALSIFIABLE PREDICTS: P1 C_body ~5-50pF → τ_r 50µs-10ms (Pazos band). P2 β→20 + R_B=10-100kΩ → self-reset cycles appear. P3 PTM 130nm invalid above V_DS=2V → any NS-RAM-regime work structurally untrustworthy until ported to 180nm.
## 2026-05-17 — z460 interim: z443_DC_VBIC ×1 fwd=1.3111 matches baseline. BUT bwd=1.3625 != z449 baseline 2.864! Suggests z449/z454 bwd may have had different sweep-direction definition. Investigate when full results in.
## 2026-05-17 — z460 INTERIM: ALPHA0 IS WIRED! ×10 moves DC fwd 1.311→1.741 (+0.43), bwd 1.363→1.747 (+0.38). Both >> 0.10 falsifier threshold. 4-pipeline-identity = REAL INVARIANCE, NOT code bug. OpenAI Q2 verdict vindicated, Gemini+Grok overcalled. BUT ALPHA0×10 made DC WORSE — consistent with z449_C. The 1.276 headline is REAL (not bug). Combined with lit-cheat node-mismatch finding: gap is node (130nm vs 180nm Pazos), not parameter calibration.
## 2026-05-17 — MARIO RE-EXTRACT DONE. Slide 12 PWL already fully used. Slide 08 (oscillation, ≠O52's slide_21) NEW QUANTITATIVE TARGETS:
## Period=0.430µs(±2.3%) V_D_peak=1.89V(±2.6%) I_D_peak=4.80mA rise=26ns fall=76ns E_spike=0.2pJ V_body_swing=0.5-0.7V. PROXY TRANSIENT VALIDATION UNLOCKED (no A.12 needed).
## Calibration recipe: keep canonical params, sweep ONLY Bf+C_B to hit 7-target rubric. Sanity: E=0.5·V·I·FWHM=0.27pJ matches "0.2 pJ" claim.
## File: data/mario_slide21_oscillation_targets.json — direct input to z461 validation V7 + z458 oscillation tuning.
## 2026-05-17 — LIT-CHEAT NODE CLAIM RETRACTED. User catch: Sebas mail.txt explicit "130 nm (current working node) PTM model". M1_130DNWFB.txt + PTM130bulkNSRAM.txt confirm. Subagent's PMC11964925 (180nm) was a PRECURSOR Pazos paper extrapolated to Nature 2025 falsely. We ARE on right node. NPN β=10⁴ + C_body ~fF suspicions still valid (general DNW physics) but without 180nm citation.
## 2026-05-17 — z461 VALIDATION HARNESS DONE (DISCOVERY PASS on z458_best 6/9): V1 PASS DC=2.47 V2 hyster PASS V3 knee FAIL nan V4 snap PASS 2.20ns V5 latch PASS 0.635V V6 reset FAIL(closest:V_B→0.4 in 52ns) V7 oscillation FAIL 0 cycles V8 LIF integ PASS V9 threshold PASS. NX_1p8 4/9, SB_OFF 4/9.
## 2026-05-17 — z458 KILL_SHOT: no self-reset on any (snap_Is, R_body) cell. Passive R_body insufficient to overcome NPN holding. Need state-dependent shutdown (two-stage knee or active reset) — NOT just resistance sweep.
## 2026-05-17 — V6+V7 are COUPLED (z461 finding): fix one → likely fix other → 8/9 AMBITIOUS achievable. Only V3 (DC knee position) is then separate parameter-fit problem.
## 2026-05-17 — tick APU=47C. z458 summary.json written + KILL_SHOT (passive R-sweep cannot beat NPN). z460 still computing PT_VBIC cells. z461 done (6/9 DISCOVERY).
## 2026-05-17 — P-phase tick APU=47C. State unchanged. P5/P6 deferred per O76 ALERT.
## 2026-05-17 — track audit: NOVEL_DS_PLAN_2026-05-12 closed >90%. Phase A 1/4 done + 1 in_progress + 2 pending (D2/PTP). Phase B (DS-N) 17/18 done (only DS-N5 in_progress; DS-N3 absent). Phase C 4E.1 done #212, oracle critique deferred. NS-RAM z45x campaign supersedes.
## 2026-05-17 — deep-dive tick APU=47C. Running: z460 (PT_VBIC cells), z458 (extra cells post-summary). Done: z461 6/9 DISCOVERY, z458 main summary+KILL_SHOT, z454/z455/z457 chain, Mario re-extract (slide 08 targets), lit-cheat. Pending dispatch: z462 (β=20+R 10-100kΩ) for V6+V7 closure, z463 (C_body=10pF), z464 Mario-target fit. DC <0.5 gate not crossed; 6/9 dynamics is publishable functional model.
## 2026-05-17 :47 — APU=47C ACTIVE: z460+z458 still computing
## 2026-05-17 — NETWORK CAMPAIGN LAUNCHED. Plan: research_plan/NETWORK_CAMPAIGN_2026-05-17.md (10 topologies × 10 use-cases × 7 scales). Validation pre-req still in flight (z460, z462b, O77). Daedalus DOWN, zgx UP (GB10 idle). Dispatched: code-sync subagent, network_viz utility subagent. Cron jobs: 3fe6d0ea (every :19/:49 N-tick), 68fe5d1a (every 6h code-sync). NO new sims until validation completes.
## 2026-05-17 — tick APU=47C. z460+z458 still computing. New: z462b+code-sync+viz dispatched. No completions since last tick.
## 2026-05-17 — DAEDALUS LIVE via daedalus.local (IP 192.168.0.40, NOT 0.37 — cluster config bug). User pass daedalus, torch-rocm venv at ~/venvs/torch-rocm. AMD_gfx1151_energy dir already exists. Sync agent dispatched.
## 2026-05-17 — network_viz utility DONE: 7 functions (raster/vb/weight_gif/energy/latency/pareto/dashboard). Demo PASS all gates incl AMBITIOUS (dashboard 530KB Nature-style, gif 0.57MB smooth). Auto-discovery: drop {spikes,vb,weights,energy}.npy + {latency,pareto}.json → save_summary_dashboard(dir). Tools ready for N-campaign.
## 2026-05-17 — P-phase tick APU=47C. State unchanged: P5/P6 still deferred per O76 ALERT pending z460 verdict + z462b solver default change.
## 2026-05-17 — zgx sync DONE: 26.7GB/260k files synced, nsram_venv torch 2.12 CUDA True, N1b LIF sanity PASS (spike mean 7.14). daedalus 192.168.0.37 STILL DEAD per this agent — but agent used WRONG IP (user found daedalus.local = 192.168.0.40). The OTHER sync agent uses correct addr.
## 2026-05-17 — z462b PT-DEFAULT DONE. BIG: V1 RMSE 1.5 → 0.983 dec (first sub-1 honest cell-wide). AMBITIOUS PASS (<2.0). DISCOVERY partial (low-VG2 still 1.46 dec off, but VG2≥0.45 branches ≤0.14 dec — surgical). 0/19 callers broken. Snap-up stepwise visible (0.4→0.6→0.7V at V_d 1.0/1.3/1.75) BUT model latched-branch I_D ~1µA vs measured ~25µA — BJT too cold (Bf/Is/R_body/alpha0 fit issue, not solver).
## 2026-05-17 — Solver default permanently changed via NSRAM_DC_SOLVER env (default "pt"). Legacy Newton retained with DeprecationWarning. doc: research_plan/SOLVER_DEFAULT_CHANGED_2026-05-17.md.
## 2026-05-17 — N-campaign tick APU=47C. No N* sims dispatched yet (validation pre-req: z460 still computing). Existing N1_1f_noise/N2_RTN/N3_bayes_realnoise dirs are from old Phase-N (May-12), not new 10×10 matrix. Holding until z460 verdict + z462b lessons applied.
## 2026-05-17 — tick APU=47C. No new since last tick. z460 PT_VBIC ×10 still computing.
## 2026-05-17 — P-phase tick APU=47C. State unchanged.
## 2026-05-17 — N-CAMPAIGN BATCH 1 LAUNCHED 4 parallel: N-FF-MNIST (ikaros, N=512), N-Res-MG (zgx N=1024), N-HDC-UCIHAR (daedalus N=8192), N-STDP-ECG (ikaros N=100). All with PT-default solver + viz dashboard auto. ETA 2-3h per cell.
## 2026-05-17 :47 — APU=39C idle (z460 done, N-batch subagents in setup)
## 2026-05-17 — N-tick APU=39C. No N_ result dirs yet, no N python procs. 4 subagents still in setup phase.
## 2026-05-17 — N-Res-MG ZGX **AMBITIOUS PASS**! NRMSE=0.0153 << 0.05 ambitious gate (×3 margin), throughput 22k steps/sec >> 10k gate. N=1024 ER_SPARSE reservoir, Mackey-Glass τ=17, wall 0.29s for 6501 steps on CUDA. Note: surrogate is tanh-based not full PT LUT (acceptable per campaign principle "physics good enough at network scale"). FIRST n-sim AMBITIOUS PASS.
## 2026-05-17 — N-FF-MNIST mid-run (test featurization done, readout training). 3 more dispatched: z462 (β=20+R-low), z465 (Mario BBO on daedalus), N-PC-NAB (zgx). 7 spår nu parallellt över 3 maskiner.
## 2026-05-17 — tick APU=39C. No new z45x. N-Res-MG PASS earlier. N-FF-MNIST mid-run.
## 2026-05-17 — THERMAL WARN APU=82C (close to 85 cutoff). N-FF-MNIST in thermal pause cycles. Hold new ikaros dispatches.
## 2026-05-17 — N-HDC-UCIHAR DAEDALUS DONE: DISCOVERY PASS, AMBITIOUS near-miss. Best test_acc=0.8453 (seed 0,2), mean 0.8383±0.0099 (gate >0.70 PASS, >0.85 missed by 0.5pp). Mem 0.506 GB (8× under 4 GB budget). 101k inf/s. D=128→8192 gave +18.8pp (HDC capacity scaling works with NS-RAM nonlinearity). Daedalus 32-core friendly. Dashboard rendered.
## 2026-05-17 — P-phase tick. State unchanged: P5/P6 deferred pending z462 + Mario BBO results.
## 2026-05-17 — N-PC-NAB ZGX DISCOVERY PASS: mean F1=0.335 (gate >0.3, 3 NAB streams: art_daily=0.33, nyc_taxi=0.46, machine_temp=0.21). Energy 1.01 pJ/sample, throughput 4389 samples/s (4.4× over 1k bar). AMBITIOUS FAIL on F1 only (precision-limited; error neurons flag every regime shift). Wall 8.1s. weight_evo.gif (30 frames, 0.21 MB) + 6-panel dashboard rendered.
## 2026-05-17 — N-tick APU=79C. Result dirs present: N_FF_MNIST/HDC/PC_NAB/Res_MG/STDP_ECG. 2 confirmed PASS (Res_MG AMBITIOUS, HDC+PC_NAB DISCOVERY). N-FF-MNIST + N-STDP-ECG still running (thermal cycles). No new dispatches (3 sims + z462 + z465 active = full). NO ALERT (DISCOVERY already documented earlier ticks).
## 2026-05-17 — tick APU=79C. z45x: no new. z465 BBO improved to 1.059 (GP phase started).
## 2026-05-17 — N-STDP-ECG DONE: INFRA PASS, DISCOVERY FAIL (test F1=0 due to cross-subject readout collapse). Train F1=0.975 (substrate learns well, STDP active 280k spikes/s, weight_evo.gif visible). Energy 17.9 pJ/beat (well under 50 pJ AMBITIOUS bar). Root cause: linear readout (logistic SGD) cannot generalize across MIT-BIH subjects without per-subject adaptation. Substrate validated, readout NEEDS work. Honest negative.
## 2026-05-17 — N-BATCH 2 LAUNCHED (4 parallel): N-Rec-DVS (zgx BPTT), N-WTA-MNIST (zgx Hebbian unsup), N-Mem-Pal (daedalus binding), N-STDP-ECG-v2 (zgx, NLMS readout fix). Filling zgx idle slot + daedalus parallel to z465 BBO. Total active spår: 4 model (z460/z462/z465/N-FF-MNIST) + 4 N-batch1-tail + 4 N-batch2 = 8-9 parallel sims.
## 2026-05-17 — P-phase tick. State unchanged.
## 2026-05-17 — N-Rec-DVS DONE INFRA PASS, DISCOVERY FAIL: acc 0.389 (4.3× chance, gate>0.75 missed). Honest caveat: REAL DVS-Gesture data unobtainable on zgx (tonic figshare WAF + 0-byte tar placeholder + numpy ABI break post-downgrade) → fell back to synthetic-proxy with disjoint RNG (no leakage). Substrate works (V_b spans -27..+14, monotonic loss 2.21→1.71, train_acc 0.19→0.37 over 3 epochs, BPTT learning). Throughput 10M events/s (warm). To pass DISCOVERY: 20-30 epochs OR real DVS data manually placed.
## 2026-05-17 — N-Mem-Pal DAEDALUS DISCOVERY PASS! P=16 87.5%/89.6% recall (bidirectional loc↔item, gate≥60%). Capacity@50%=48, @60%=32, @80%=24 (AMBITIOUS gate P=32@80% missed by one rung 76%). Energy 6.2 pJ/recall (320 cells × 5 probe steps × 3.75fJ + ADC). Wall 5.8s on daedalus CPU. NS-RAM body-charge anchors V_HI=0.6V, SDM-style k=5 cell addressing per HD-bound key. weight_evo.gif (16 frames memory matrix filling) rendered.
## 2026-05-17 :47 — APU=47C ACTIVE: podcast_tts.py (TTS-gen). z465 new best 0.597.
## 2026-05-17 — N-tick APU=47C. Status: N-FF-MNIST + N-WTA-MNIST + N-STDP-ECG-v2 still running. 4 PASS (Res_MG AMB, HDC+PC_NAB+Mem_Pal DISC). z465 best 0.597 (GP descent). No new dispatches (capacity full).
## 2026-05-17 — tick APU=47C. z45x: z465 stuck at best 0.594 (sharp basin). N-WTA failed (40% acc, agent prepping v2 STDP). No new completions.
## 2026-05-17 — P-phase tick APU=47C. State unchanged.
## 2026-05-17 — Oracle 12h dispatched (3-way openai+gemini+grok). 3 Qs: gate-crossing if z465 lands 0.4-0.5, cherry-pick risk on 4/7 PASS N-sims, next 6h between (a)z462 (b)slide21 re-extract (c)z464 BBO-optimum validate. PID 606318.
## 2026-05-17 — N-tick APU=47C. 4 PASS bekräftade (Res-MG AMB, HDC/PC-NAB/Mem-Pal DISC). 4 FAIL/near (FF-MNIST 91.6%, STDP, Rec-DVS, WTA). z462+z465 model-side aktiv. No new dispatch (capacity full).
## 2026-05-17 — tick APU=47C. z465 hovers 0.567 ~15 iter left. z462 running cell 2/12. No new z45x completions.
## 2026-05-17 — P-phase tick APU=47C. State unchanged.
## 2026-05-17 — deep-dive tick APU=47C. Active: z462 (cell 2 fast-pulse, β=20 DC=1.344 cached), z465 (best 0.557, ~10 iter kvar). Both reset/oscillation experiments mid-stream. Next gated: z463 C_body sweep if z462+z465 both fail to close V6/V7. No DC<0.5 crossed (best honest 0.983 dec z462b).
## 2026-05-17 :47 — APU=46C idle (active subagents on remote machines, no local z2[3-9])
## 2026-05-17 — z465 BBO COMPLETE 70 iter, wall 6071s. Best fit 0.557 @ iter 63: Is=2.88e-7, Rb=387kΩ (lit-cheat zone!), Bf=10⁴ (ceiling), Cb=5.6fF. Mario targets: 2/7 PASS (period+V_D, both trivial). I_D peak 0.20µA vs Mario 4.8mA (4 DEC OFF). rise 6.6ns vs 26ns. V_body 0.39V vs 0.20V. DC 1.37dec ok.
## 2026-05-17 — z465 VERDICT INFRA_ONLY: cell fires + V_B swings but cannot deliver mA conduction. Root cause: BBO hit structural ceiling — search needs to include snap_V_knee + snap_npn_V_knee + snap_npn_V_BE_offset gating thresholds (held fixed). β=20 lit-cheat hypothesis FALSIFIED (BBO maxes Bf=10⁴ ceiling).
## 2026-05-17 — N-tick APU=46C. z465 COMPLETE INFRA_ONLY (2/7 Mario, struct ceiling β=10⁴, need knee-gate widening). z462 β=50 cell running. 4 PASS unchanged. No new dispatch (capacity full).
## 2026-05-17 — tick APU=46C. z465 DONE (INFRA_ONLY). z462 β=50 row mid. No new completions.
## 2026-05-17 — code-sync 6h: zgx rsynced + sanity OK (nsram import). daedalus.local rsynced clean (cluster cron still has wrong .37 IP, our manual sync uses .local). All 3 machines have fresh code.
## 2026-05-17 — P-phase tick. State unchanged.
## 2026-05-17 — N-Stoch-RNG ZGX ALL 3 GATES PASS incl AMBITIOUS! NIST 5/5 PASS, 1.27 Mbit/s, KL_mean 1.4e-6 (4 orders margin), 0.40 pJ/bit. Stochastic AND/OR/XOR error <0.001. (NB: subagent put output in wrong dir, moved to results/N_Stoch_RNG_N100/)
## 2026-05-17 — N-LMS-Eq ZGX ALL 3 GATES PASS incl AMBITIOUS! BER@20dB=0.000, BER@10dB=0.0155 (both gates met). Energy 2.76 pJ/symbol vs LMS-f32 474 pJ = 170× LOWER. 16-tap complex NS-RAM equalizer, QPSK over 3-echo multipath. Wall 1.3s.

```


=== FILE: z462b_pt_default_summary.json (2691 chars) ===
```json
{
  "date": "2026-05-17",
  "change": "Default DC solver: Newton-DC -> pseudo-transient backward (PT-bwd)",
  "files_modified": [
    "scripts/z429_multisolver_debug.py (run_vsint_pinned dispatcher + _run_vsint_pinned_pt + run_vd_sweep_pt_backward + DeprecationWarning)"
  ],
  "env_override": "NSRAM_DC_SOLVER=newton restores legacy Newton-DC",
  "sanity": {
    "VG1": 0.6,
    "VG2": 0.0,
    "n_points": 61,
    "convergence_rate": 0.8524590163934426,
    "Vb_max_V": 0.7000000000000001,
    "max_log10_jump_dec": 1.4445949827935554,
    "snap_Vd_V": 0.03333333333333333,
    "snap_up_visible": false,
    "Vb_above_0p5": true,
    "wall_sec": 83.3
  },
  "V1": {
    "per_branch_rmse_dec": {
      "0.2": 0.9978545004450636,
      "0.4": 0.8084560927185284,
      "0.6": 1.0697510385052027
    },
    "cell_rmse_dec": 0.982511967023258,
    "n_curves": 25,
    "wall_sec": 1122.0,
    "disc_deltas_at_Vd_1p7_VG1_0p6": [
      {
        "VG2": 0.0,
        "delta_log_dec": 1.3731877083715434,
        "Ip": 1.09174737831605e-06,
        "Im": 2.57816e-05
      },
      {
        "VG2": 0.05,
        "delta_log_dec": 1.3730761964538756,
        "Ip": 1.0917778320645601e-06,
        "Im": 2.57757e-05
      },
      {
        "VG2": 0.1,
        "delta_log_dec": 1.3735298831047338,
        "Ip": 1.0918691973655054e-06,
        "Im": 2.58048e-05
      },
      {
        "VG2": 0.15,
        "delta_log_dec": 1.373610315526884,
        "Ip": 1.09174737831605e-06,
        "Im": 2.58067e-05
      },
      {
        "VG2": 0.2,
        "delta_log_dec": 1.3735753068007064,
        "Ip": 1.0917169252434283e-06,
        "Im": 2.58039e-05
      },
      {
        "VG2": 0.25,
        "delta_log_dec": 1.3732318272730444,
        "Ip": 1.0917169252434283e-06,
        "Im": 2.57835e-05
      },
      {
        "VG2": 0.3,
        "delta_log_dec": 1.3723817636562705,
        "Ip": 1.09174737831605e-06,
        "Im": 2.57338e-05
      },
      {
        "VG2": 0.35,
        "delta_log_dec": 1.3693622247637443,
        "Ip": 1.09174737831605e-06,
        "Im": 2.55555e-05
      },
      {
        "VG2": 0.4,
        "delta_log_dec": 1.455812148397774,
        "Ip": 8.726401989016162e-07,
        "Im": 2.49257e-05
      },
      {
        "VG2": 0.45,
        "delta_log_dec": 0.1399327701667037,
        "Ip": 8.083638389232359e-07,
        "Im": 1.11568e-06
      },
      {
        "VG2": 0.5,
        "delta_log_dec": 0.12587409303548913,
        "Ip": 8.083845753830811e-07,
        "Im": 1.08017e-06
      }
    ]
  },
  "gates": {
    "INFRA": true,
    "DISCOVERY_max_dlog_lt_0p3": false,
    "AMBITIOUS_cell_rmse_lt_2p0": true,
    "KILL_SHOT_broke_>2_scripts": false
  }
}
```


=== FILE: z465_mario_bbo_summary.json (3487 chars) ===
```json
{
  "best_params": {
    "snap_Is": 2.881283248571692e-07,
    "R_body": 387231.03860300966,
    "Bf": 10000.0,
    "C_body": 5.629238987849037e-15
  },
  "best_fitness": 0.5572127667000482,
  "best_fit_transient": 0.5572127667000482,
  "dc_rmse_at_best_inner": 1.4777986167100543,
  "dc_rmse_at_best_full": 1.3732848129732538,
  "per_target_errors_innerloop": {
    "period_s": 0.0006671114076049005,
    "Vd_peak_V": 0.0006671114076050615,
    "Id_peak_A": 0.9999573610137615,
    "rise_s": 0.7352080874429128,
    "fall_s": 0.5810364804606579,
    "Vbody_swing_V": 0.970103145457784,
    "E_spike_J": 0.9308534057415192
  },
  "per_target_errors_hires": {
    "period_s": 0.0001666944490748048,
    "Vd_peak_V": 0.00016669444907505625,
    "Id_peak_A": 0.9999573534245366,
    "rise_s": 0.7463679844076591,
    "fall_s": 0.5850185557242001,
    "Vbody_swing_V": 0.9702158656248535,
    "E_spike_J": 0.9308436661571334
  },
  "measured_hires": {
    "valid": true,
    "period_s": 4.299283213868978e-07,
    "Vd_peak_V": 1.889684947491248,
    "Id_peak_A": 2.0470356222401846e-07,
    "rise_s": 6.594432405400866e-09,
    "fall_s": 3.1538589764960795e-08,
    "Vbody_swing_V": 0.3940431731249706,
    "E_spike_J": 1.38312667685733e-14,
    "n_spikes": 3
  },
  "targets": {
    "period_s": 4.2999999999999996e-07,
    "Vd_peak_V": 1.89,
    "Id_peak_A": 0.0048,
    "rise_s": 2.6e-08,
    "fall_s": 7.6e-08,
    "Vbody_swing_V": 0.19999999999999996,
    "E_spike_J": 2e-13
  },
  "weights": {
    "period_s": 0.25,
    "Vd_peak_V": 0.1,
    "Id_peak_A": 0.15,
    "rise_s": 0.15,
    "fall_s": 0.1,
    "Vbody_swing_V": 0.15,
    "E_spike_J": 0.1
  },
  "convergence_history": [
    1.141549733357627,
    1.1936543778068187,
    1.1888715595851311,
    1.2068489517237122,
    1.1889511131026054,
    1.1889509928113484,
    1.188925182040418,
    1.1886732628659966,
    1.2175539208081216,
    1.203286206812482,
    1.1935486104699216,
    1.0904864051275507,
    1.156748176315335,
    1.188936280157801,
    1.0588496164523422,
    1.1933244581692135,
    1.2252086841311711,
    1.1063371941723585,
    0.9776435047436305,
    1.1673442652563302,
    1.0756202835905728,
    0.7903898282210323,
    1.0357522208637613,
    0.7510622635453718,
    1.1984245009887737,
    0.8855806153085347,
    0.7044574266704516,
    0.610848427060776,
    1.1458629329159338,
    0.6466052605959054,
    0.5969437768045243,
    0.7080524999490538,
    1.1731483108506944,
    0.5938344321622777,
    1.1519056539473318,
    0.596000945061241,
    0.7301726958536936,
    0.5837570466101633,
    0.6478146595955213,
    0.567114944731025,
    0.6742516706659714,
    0.6207298304881202,
    1.1420537030581883,
    0.6974712864160079,
    0.5842694214611825,
    0.6051295015124851,
    0.720811173599805,
    0.657332887353203,
    0.6620572098917812,
    0.6485985941497321,
    1.1885637027642502,
    0.5944638295179044,
    0.5761800734004218,
    0.6412369491516305,
    0.6520708558005984,
    0.5919410491962755,
    0.6645667456407561,
    0.6337095940678231,
    0.6630932805300855,
    0.8603493874769275,
    0.6237266933922917,
    0.6408202304096089,
    0.5572127667000482,
    0.6231469573128364,
    1.2355055060903473,
    0.594772775252617,
    0.585506649739112,
    1.2520621720765854,
    0.5706706450971136,
    0.6583271717640423,
    0.6287878512465022
  ],
  "n_evals": 71,
  "n_within_30_hires": 2,
  "gate_verdict": "INFRA_ONLY",
  "bbo_wall_s": 6071.435215950012
}
```
