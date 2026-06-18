# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: v4_advantage_null.json (2585 chars) ===
```json
{
  "tasks": [
    "narma10",
    "mackey"
  ],
  "seeds": [
    0,
    1,
    2,
    3,
    4
  ],
  "per_core_n": 8,
  "narma10": {
    "per_cond": {
      "baseline": {
        "per_seed": [
          0.5450414045271865,
          0.5967129195780772,
          0.5453215009034461,
          0.6065058474076357,
          0.6306220844657289
        ],
        "median": 0.5967129195780772
      },
      "random_env": {
        "per_seed": [
          0.6215124361671395,
          0.6170512713649997,
          0.5422341287676798,
          0.6309568773272177,
          0.6568531234159086
        ],
        "median": 0.6215124361671395
      },
      "hashed_env": {
        "per_seed": [
          0.6105753481285592,
          0.6307134025864449,
          0.5566158542868845,
          0.6196412694905449,
          0.6576570603032923
        ],
        "median": 0.6196412694905449
      },
      "mapped_env": {
        "per_seed": [
          0.6137776776540352,
          0.6259552467954876,
          0.5517200855175692,
          0.6144216522451466,
          0.6558346824492155
        ],
        "median": 0.6144216522451466
      }
    },
    "win_threshold": 0.5370416276202694,
    "mapped_env_median": 0.6144216522451466,
    "WIN_mapped_vs_both": false,
    "advantage_pct": -2.9677139686519416
  },
  "mackey": {
    "per_cond": {
      "baseline": {
        "per_seed": [
          0.020950248244612755,
          0.019873378996991008,
          0.027854330700866757,
          0.031446786401489006,
          0.018575319908516408
        ],
        "median": 0.020950248244612755
      },
      "random_env": {
        "per_seed": [
          0.018708548505882043,
          0.025726174311994247,
          0.027566203007454597,
          0.02921107508768954,
          0.028299473794092356
        ],
        "median": 0.027566203007454597
      },
      "hashed_env": {
        "per_seed": [
          0.03203368161212772,
          0.029032551012219857,
          0.028009103390726105,
          0.028505560266005252,
          0.0207914066052138
        ],
        "median": 0.028505560266005252
      },
      "mapped_env": {
        "per_seed": [
          0.0245685593869492,
          0.02959676851052846,
          0.028013741678914374,
          0.02830598619253335,
          0.015437873368752854
        ],
        "median": 0.028013741678914374
      }
    },
    "win_threshold": 0.01885522342015148,
    "mapped_env_median": 0.028013741678914374,
    "WIN_mapped_vs_both": false,
    "advantage_pct": -33.715559604970124
  },
  "ANY_WIN": false
}
```


=== FILE: v5_h1_result.json (2756 chars) ===
```json
{
  "n_cores": 32,
  "core_lats_s": [
    0.0016203438461406432,
    0.0013427949999974468,
    0.0012777638749952303,
    0.0012750874374916066,
    0.0012800601249978172,
    0.001282930062501464,
    0.0012731183125112011,
    0.001277207812506731,
    0.0012664357499971857,
    0.0012793571874993859,
    0.0012545892500099853,
    0.001253234124988012,
    0.0012537423124996394,
    0.001268435562494119,
    0.0012506162500045548,
    0.0012586792500002275,
    0.001723159583339869,
    0.0017034784166677734,
    0.0014632926428573359,
    0.0012763269999993554,
    0.0012587857499966049,
    0.0012642791249959373,
    0.001261631812496944,
    0.0012733493749976788,
    0.001256404250000287,
    0.0012637096249932256,
    0.0012660387500034176,
    0.0012688943125027663,
    0.0012640664999992168,
    0.0012759709999983215,
    0.0012518101874974263,
    0.001268856687488551
  ],
  "rank_fast_to_slow": [
    14,
    30,
    11,
    12,
    10,
    24,
    15,
    20,
    22,
    25,
    28,
    21,
    26,
    8,
    13,
    31,
    27,
    6,
    23,
    3,
    29,
    19,
    7,
    2,
    9,
    4,
    5,
    1,
    18,
    0,
    17,
    16
  ],
  "rank_assignment": [
    14,
    30,
    11,
    12,
    10,
    24,
    15,
    20,
    22,
    25,
    28,
    21,
    26,
    8,
    13,
    31
  ],
  "baseline_assignment": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15
  ],
  "trials": [
    {
      "rank_nrmse": 0.5970625485409469,
      "rank_wall_s": 0.19436519799069174,
      "baseline_nrmse": 0.5970625485409469,
      "baseline_wall_s": 0.11518915200508673
    },
    {
      "rank_nrmse": 0.6244864213719502,
      "rank_wall_s": 0.19782180901074753,
      "baseline_nrmse": 0.6244864213719502,
      "baseline_wall_s": 0.1381899200055159
    },
    {
      "rank_nrmse": 0.5124324387550665,
      "rank_wall_s": 0.23386796203749327,
      "baseline_nrmse": 0.5124324387550665,
      "baseline_wall_s": 0.1384072449989162
    },
    {
      "rank_nrmse": 0.602692571125407,
      "rank_wall_s": 0.21222847899775843,
      "baseline_nrmse": 0.602692571125407,
      "baseline_wall_s": 0.10895848702898547
    },
    {
      "rank_nrmse": 0.6478386970933255,
      "rank_wall_s": 0.22178312803612243,
      "baseline_nrmse": 0.6478386970933255,
      "baseline_wall_s": 0.14888677998737876
    }
  ],
  "summary": {
    "rank_nrmse_med": 0.602692571125407,
    "baseline_nrmse_med": 0.602692571125407,
    "rank_wall_med": 0.21222847899775843,
    "baseline_wall_med": 0.1381899200055159,
    "speedup_pct": -53.57739478341638,
    "accuracy_gain_pct": 0.0
  },
  "gate": {
    "win_speed": false,
    "win_accuracy": false,
    "WIN": false
  }
}
```


=== FILE: v5_h2_daedalus.json (1092 chars) ===
```json
{
  "host": "daedalus",
  "densities": [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5
  ],
  "per_density": {
    "0.10": {
      "nrmse_med": 0.583505119048299,
      "wall_med": 0.30688253600010285,
      "power_uw_med": 123046000.0,
      "temp_mc_max": 88000.0
    },
    "0.20": {
      "nrmse_med": 0.5766383670290124,
      "wall_med": 0.2611757990002843,
      "power_uw_med": 122049000.0,
      "temp_mc_max": 91000.0
    },
    "0.30": {
      "nrmse_med": 0.574788415709675,
      "wall_med": 0.3105555080001068,
      "power_uw_med": 121078500.0,
      "temp_mc_max": 94000.0
    },
    "0.40": {
      "nrmse_med": 0.5754307311269294,
      "wall_med": 0.24500014500017642,
      "power_uw_med": 121530000.0,
      "temp_mc_max": 97000.0
    },
    "0.50": {
      "nrmse_med": 0.5754093816375192,
      "wall_med": 0.42531476700014537,
      "power_uw_med": 121042500.0,
      "temp_mc_max": 97000.0
    }
  },
  "own_optimal_density": "0.30",
  "own_optimal_nrmse": 0.574788415709675,
  "generic_nrmse": 0.574788415709675,
  "accuracy_gain_pct_vs_generic": 0.0,
  "WIN": false
}
```


=== FILE: v5_h2_ikaros.json (1094 chars) ===
```json
{
  "host": "ikaros",
  "densities": [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5
  ],
  "per_density": {
    "0.10": {
      "nrmse_med": 0.5835051209999292,
      "wall_med": 0.028029825999965396,
      "power_uw_med": 22505500.0,
      "temp_mc_max": 45000.0
    },
    "0.20": {
      "nrmse_med": 0.5766383664390593,
      "wall_med": 0.027374675000146453,
      "power_uw_med": 25551000.0,
      "temp_mc_max": 45000.0
    },
    "0.30": {
      "nrmse_med": 0.5747884164555698,
      "wall_med": 0.021037631000126567,
      "power_uw_med": 28576000.0,
      "temp_mc_max": 45000.0
    },
    "0.40": {
      "nrmse_med": 0.5754307309985814,
      "wall_med": 0.021010199000102148,
      "power_uw_med": 33529500.0,
      "temp_mc_max": 46000.0
    },
    "0.50": {
      "nrmse_med": 0.5754093798256571,
      "wall_med": 0.0205793340001037,
      "power_uw_med": 37532500.0,
      "temp_mc_max": 46000.0
    }
  },
  "own_optimal_density": "0.30",
  "own_optimal_nrmse": 0.5747884164555698,
  "generic_nrmse": 0.5747884164555698,
  "accuracy_gain_pct_vs_generic": 0.0,
  "WIN": false
}
```


=== FILE: v5_h3_daedalus.json (324 chars) ===
```json
{
  "host": "daedalus",
  "n_samples": 1000000,
  "batch": 4096,
  "prng_err_med": 0.0008886535897931758,
  "chip_err_med": 0.001463346410206956,
  "prng_wall_med": 0.017576756999915233,
  "chip_wall_med": 0.3410538609996365,
  "latency_reduction_pct": -1840.368527603137,
  "err_ratio": 1.6467006120433652,
  "WIN": false
}
```


=== FILE: v5_h3_ikaros.json (322 chars) ===
```json
{
  "host": "ikaros",
  "n_samples": 1000000,
  "batch": 4096,
  "prng_err_med": 0.0008886535897931758,
  "chip_err_med": 0.001015346410206952,
  "prng_wall_med": 0.009502431000100842,
  "chip_wall_med": 0.1552065920000132,
  "latency_reduction_pct": -1533.3356379895429,
  "err_ratio": 1.142567162130367,
  "WIN": false
}
```


=== FILE: v5_h4_daedalus.json (991 chars) ===
```json
{
  "host": "daedalus",
  "n_tasks": 5,
  "n_per_task": 1000,
  "thermal": {
    "per_task_acc": [
      0.6966666666666667,
      0.65,
      0.5833333333333334,
      0.64,
      0.7433333333333333
    ],
    "mean_acc": 0.6626666666666667,
    "lr_log": {
      "used_lrs_mean": 0.0030000000000000005,
      "used_lrs_std": 4.336808689942018e-19
    }
  },
  "constant": {
    "per_task_acc": [
      0.6933333333333334,
      0.65,
      0.5866666666666667,
      0.64,
      0.7433333333333333
    ],
    "mean_acc": 0.6626666666666666,
    "lr": 0.0030000000000000005,
    "lr_log": {
      "used_lrs_mean": 0.003000000000000001,
      "used_lrs_std": 4.336808689942018e-19
    }
  },
  "high_lr_ref": {
    "per_task_acc": [
      0.98,
      0.99,
      0.9933333333333333,
      0.9266666666666666,
      1.0
    ],
    "mean_acc": 0.9780000000000001
  },
  "gain_pct_thermal_vs_const": 1.6753868580862524e-14,
  "abs_gain_thermal_vs_const": 1.1102230246251565e-16,
  "WIN": false
}
```


=== FILE: v5_h4_ikaros.json (985 chars) ===
```json
{
  "host": "ikaros",
  "n_tasks": 5,
  "n_per_task": 1000,
  "thermal": {
    "per_task_acc": [
      0.9833333333333333,
      0.99,
      0.99,
      0.9233333333333333,
      1.0
    ],
    "mean_acc": 0.9773333333333334,
    "lr_log": {
      "used_lrs_mean": 0.030000000000000013,
      "used_lrs_std": 1.3877787807814457e-17
    }
  },
  "constant": {
    "per_task_acc": [
      0.9866666666666667,
      0.99,
      0.9933333333333333,
      0.9266666666666666,
      1.0
    ],
    "mean_acc": 0.9793333333333333,
    "lr": 0.030000000000000013,
    "lr_log": {
      "used_lrs_mean": 0.030000000000000023,
      "used_lrs_std": 1.0408340855860843e-17
    }
  },
  "high_lr_ref": {
    "per_task_acc": [
      0.9766666666666667,
      0.99,
      0.9933333333333333,
      0.9266666666666666,
      1.0
    ],
    "mean_acc": 0.9773333333333334
  },
  "gain_pct_thermal_vs_const": -0.20422055820284796,
  "abs_gain_thermal_vs_const": -0.0019999999999998908,
  "WIN": false
}
```


=== FILE: v5_h5_crosseval.json (679 chars) ===
```json
{
  "per_seed": {
    "A_ik_on_ik": [
      0.17466014623641968,
      1.6508938074111938,
      1.5689860582351685
    ],
    "B_da_on_da": [
      0.17428623139858246,
      1.635241985321045,
      1.5527743101119995
    ],
    "C_ik_on_da": [
      0.17466014623641968,
      1.6508938074111938,
      1.5689860582351685
    ],
    "D_da_on_ik": [
      0.17428623139858246,
      1.635241985321045,
      1.5527743101119995
    ]
  },
  "summary": {
    "A_ik_on_ik": 1.5689860582351685,
    "B_da_on_da": 1.5527743101119995,
    "C_ik_on_da": 1.5689860582351685,
    "D_da_on_ik": 1.5527743101119995
  },
  "gain_A_vs_C_pct": 0.0,
  "gain_B_vs_D_pct": 0.0,
  "WIN": false
}
```


=== FILE: v5_h5_daedalus.json (365 chars) ===
```json
{
  "host": "daedalus",
  "seeds": [
    1,
    2,
    3
  ],
  "own_test_mse_per_seed": [
    0.17428623139858246,
    0.19388169050216675,
    0.19632457196712494
  ],
  "own_test_mse_med": 0.19388169050216675,
  "adapter_path": "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/IDENTITY_BENCHMARK_2026-05-30/embodiment4/v5_h5_daedalus_adapter.npz"
}
```


=== FILE: v5_h5_ikaros.json (357 chars) ===
```json
{
  "host": "ikaros",
  "seeds": [
    1,
    2,
    3
  ],
  "own_test_mse_per_seed": [
    0.17466014623641968,
    0.194179967045784,
    0.19568057358264923
  ],
  "own_test_mse_med": 0.194179967045784,
  "adapter_path": "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/IDENTITY_BENCHMARK_2026-05-30/embodiment4/v5_h5_ikaros_adapter.npz"
}
```


=== FILE: v5_h6_daedalus.json (1111 chars) ===
```json
{
  "host": "daedalus",
  "per_seed": [
    {
      "seed": 1,
      "fp32": 25.403564453125,
      "fp16_generic": 25.400049209594727,
      "fp16_calibrated": 25.400238037109375,
      "c": 0.21656648814678192
    },
    {
      "seed": 2,
      "fp32": 15.162646293640137,
      "fp16_generic": 15.160981178283691,
      "fp16_calibrated": 15.160955429077148,
      "c": -0.019716346636414528
    },
    {
      "seed": 3,
      "fp32": 22.239530563354492,
      "fp16_generic": 22.24209976196289,
      "fp16_calibrated": 22.242103576660156,
      "c": 0.006889093201607466
    },
    {
      "seed": 4,
      "fp32": 19.199012756347656,
      "fp16_generic": 19.199615478515625,
      "fp16_calibrated": 19.1995792388916,
      "c": 0.022943165153265
    },
    {
      "seed": 5,
      "fp32": 18.512540817260742,
      "fp16_generic": 18.513063430786133,
      "fp16_calibrated": 18.5128231048584,
      "c": -0.14264006912708282
    }
  ],
  "generic_fp16_mse_med": 19.199615478515625,
  "calibrated_fp16_mse_med": 19.1995792388916,
  "accuracy_improvement_pct": 0.00018875182195179715,
  "WIN": false
}
```


=== FILE: v5_h6_ikaros.json (1116 chars) ===
```json
{
  "host": "ikaros",
  "per_seed": [
    {
      "seed": 1,
      "fp32": 25.410154342651367,
      "fp16_generic": 25.409059524536133,
      "fp16_calibrated": 25.40856170654297,
      "c": 0.3086210787296295
    },
    {
      "seed": 2,
      "fp32": 15.199748039245605,
      "fp16_generic": 15.197031021118164,
      "fp16_calibrated": 15.197051048278809,
      "c": -0.02424667589366436
    },
    {
      "seed": 3,
      "fp32": 22.255273818969727,
      "fp16_generic": 22.255573272705078,
      "fp16_calibrated": 22.255582809448242,
      "c": 0.008013403974473476
    },
    {
      "seed": 4,
      "fp32": 18.9733943939209,
      "fp16_generic": 18.971641540527344,
      "fp16_calibrated": 18.971689224243164,
      "c": -0.025428058579564095
    },
    {
      "seed": 5,
      "fp32": 18.456993103027344,
      "fp16_generic": 18.45433235168457,
      "fp16_calibrated": 18.454496383666992,
      "c": -0.3050380051136017
    }
  ],
  "generic_fp16_mse_med": 18.971641540527344,
  "calibrated_fp16_mse_med": 18.971689224243164,
  "accuracy_improvement_pct": -0.0002513420660961269,
  "WIN": false
}
```
