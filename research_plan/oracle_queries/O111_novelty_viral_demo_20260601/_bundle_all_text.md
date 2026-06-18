# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: O109_prior_synthesis.md (4515 chars) ===
```
# O109 Synthesis — Coupling × Model × Benchmark

**Date:** 2026-05-31  •  **Providers:** OpenAI (gpt-5), Gemini (2.5-pro), Grok (grok-4-latest), DeepSeek (reasoner)

## 1. Diagnosis of the A−B null (Phase 7 ABCD)

**Unanimous verdict: (d) all three causes, dominated by (a) decorative coupling.**

| Cause                                    | OpenAI | Gemini | Grok | DeepSeek | Mean |
|------------------------------------------|--------|--------|------|----------|------|
| (a) hash coupling decorative             | 0.60   | 0.60   | 0.45 | ~0.50    | 0.54 |
| (c) benchmark doesn't require body-info  | 0.15   | 0.30   | 0.40 | ~0.30    | 0.29 |
| (b) ridge ESN too universal              | 0.25   | 0.10   | 0.15 | ~0.20    | 0.17 |

All four agree: the hash never enters the forward pass, so it is a label not a computation; random ESN matrices of the same class are statistically equivalent (Yıldız et al. 2012); ridge readout washes out any weak structural prior; next-step prediction is solvable from data alone.

## 2. Constitutive coupling design — consensus

The substrate must parameterize the **recurrent update equation itself**, not be appended to x. Four near-identical pseudocode sketches:

```
α(t) = sigmoid((T_apu(t) - T_ref)/T_range)   # leak rate
ρ(t) or γ(t) = f(P_pkg(t))                    # spectral radius / gain
h_{t+1} = (1-α)·h + α·tanh( γ·(W·h + U·x) )
```

- **Signals**: APU temp (`thermal_zone0`, 10–50 Hz), package power (RAPL, 1–10 Hz), optionally fan RPM, clocks.
- **Confounds (all four oracles flagged)**: sensor drift → use EMA-differenced values; aliasing → low-pass with τ ≈ 1–3 s; self-heating from sysfs reads; replay alignment (must time-align ikaros/daedalus traces); training-time scaling must NOT be re-fit on eval host.
- **Pre-reg gates**: A−B ≥ 10% NRMSE, A−D ≥ 5%, A−C ≥ 5%.

## 3. Architecture ranking (consensus best → worst)

1. **Continuous-time RNN with substrate as parameter** (substrate sets the vector field)
2. **Neural ODE** (substrate as forcing function — Chen et al. 2018)
3. **Spiking NN** (real-time integration naturally couples to live sensors)
4. **Predictive coding / energy-based** (substrate modulates landscape)
5. **LSTM** (gates modulable, but discrete time)
6. **Transformer** (adaptive layer-norm / attention temperature — possible but weak)
7. **MLP** (only if every weight is substrate-parametric)
8. **Ridge ESN** ← current null result; pure readout dominates

Grok added Hopfield (4th) and contrastive DAE (last); all agree ridge ESN is bottom-tier for measurable chassis-binding.

## 4. Benchmarks where body-info is the ONLY PATH

Unanimous trio (with literature anchors):

1. **Closed-loop fan/thermal control** — per-chassis RC time constant, fan curve, paste condition determine the unique optimal policy (Pfeifer & Bongard 2006; Hauser et al. 2011 morphological computation).
2. **Self-replication / self-prediction** — only a model coupled to its own substrate trajectory can predict its own future output under live sensor influence.
3. **Survival race / thermal-budget decoding** — maximise tokens before sensor hits 95 °C; requires internalised fan curve + heat-spreader lag (Ha & Schmidhuber 2018; recent thermal-constrained LLM serving).

## 5. Sharpest critical test

**Constitutive reservoir (α, ρ driven by live T/P) on 30-second thermal-budget survival task.**
- Pre-reg: own-chassis (A) achieves ≥12% more tokens than transplant (B) at equal thermal violations (Cohen's d > 0.8, n=40, BCa 95% CI).
- Negative controls: C constant-α=0.5, D shuffled alien trajectory.

## 6. Brutal honesty: recoverable or falsified?

**RECOVERABLE — but the *hash-coupling* hypothesis is falsified.** Identity-via-decoration is dead. The constitutive hypothesis (substrate ∈ forward pass) has not yet been tested at a benchmark that requires it. All four oracles say: ~3 more aligned experiments (constitutive coupling + body-required task + appropriate architecture) before declaring exhaustive falsification.

## 7. Strongest defensible paper claim (current data)

**Negative methodological result**: "Random structural priors (hash-derived ESN seeds) yield no measurable chassis-binding advantage over arbitrary random seeds on standard sensor-prediction benchmarks, demonstrating that identity-via-initialization is decorative; live-substrate coupling and embodiment-requiring tasks are necessary conditions." This is a valid null-result paper; the constitutive follow-up determines whether it stays a null or becomes positive.

```


=== FILE: embodiment12_analysis.json (5118 chars) ===
```json
{
  "D": {
    "task": "D_syscall_latency",
    "fields": {
      "nanosleep0": {
        "ika_summary": {
          "n": 100000,
          "mean_ns": 52794.59436,
          "std_ns": 8865.93298536348,
          "p50": 54242.0,
          "p90": 54723.0,
          "p99": 57407.10999999994,
          "p99_9": 77826.02000000008,
          "p99_99": 92824.03899999618,
          "min": 5581,
          "max": 1087810
        },
        "dae_summary": {
          "n": 100000,
          "mean_ns": 51944.67205,
          "std_ns": 5660.875727293331,
          "p50": 53310.0,
          "p90": 53510.0,
          "p99": 53611.0,
          "p99_9": 53741.0,
          "p99_99": 63709.10019999018,
          "min": 10380,
          "max": 324659
        },
        "intra_ikaros_KS_D": 0.0122,
        "intra_ikaros_p": 0.8508219458664311,
        "intra_daedalus_KS_D": 0.0152,
        "intra_daedalus_p": 0.610409786050154,
        "inter_KS_D": 0.7224,
        "inter_p": 0.0,
        "inter_intra_D_ratio": 47.526315789473685,
        "pre_reg_pass": true
      },
      "sched_yield": {
        "ika_summary": {
          "n": 100000,
          "mean_ns": 1054.46337,
          "std_ns": 149.176379022428,
          "p50": 1051.0,
          "p90": 1072.0,
          "p99": 1412.0099999999948,
          "p99_9": 2985.0,
          "p99_99": 4869.001999999804,
          "min": 991,
          "max": 30387
        },
        "dae_summary": {
          "n": 100000,
          "mean_ns": 768.29544,
          "std_ns": 626.770122656789,
          "p50": 761.0,
          "p90": 781.0,
          "p99": 792.0,
          "p99_9": 1553.0,
          "p99_99": 2174.006999999314,
          "min": 721,
          "max": 195586
        },
        "intra_ikaros_KS_D": 0.0222,
        "intra_ikaros_p": 0.17005669975862542,
        "intra_daedalus_KS_D": 0.0066,
        "intra_daedalus_p": 0.9999090682618054,
        "inter_KS_D": 0.9931,
        "inter_p": 0.0,
        "inter_intra_D_ratio": 44.73423423423423,
        "pre_reg_pass": true
      },
      "getpid": {
        "ika_summary": {
          "n": 100000,
          "mean_ns": 724.63171,
          "std_ns": 116.3539466991812,
          "p50": 671.0,
          "p90": 912.0,
          "p99": 952.0,
          "p99_9": 1412.0010000000038,
          "p99_99": 3537.0169999983336,
          "min": 611,
          "max": 4999
        },
        "dae_summary": {
          "n": 100000,
          "mean_ns": 658.81453,
          "std_ns": 38.55634929397621,
          "p50": 661.0,
          "p90": 671.0,
          "p99": 682.0,
          "p99_9": 1062.0,
          "p99_99": 2034.003999999608,
          "min": 611,
          "max": 2776
        },
        "intra_ikaros_KS_D": 0.0232,
        "intra_ikaros_p": 0.13556067590791931,
        "intra_daedalus_KS_D": 0.0136,
        "intra_daedalus_p": 0.7442740260220542,
        "inter_KS_D": 0.3594,
        "inter_p": 0.0,
        "inter_intra_D_ratio": 15.49137931034483,
        "pre_reg_pass": true
      }
    }
  },
  "E": {
    "task": "E_rdrand",
    "fields": {
      "rdrand_cycles": {
        "ika_summary": {
          "n": 1000000,
          "mean_cyc": 140.85171,
          "std_cyc": 211.85502281059064,
          "p50": 150.0,
          "p90": 150.0,
          "p99": 270.0,
          "p99_9": 300.0,
          "p99_99": 1020.0,
          "min": 120,
          "max": 131040
        },
        "dae_summary": {
          "n": 1000000,
          "mean_cyc": 98.3067,
          "std_cyc": 60.25186084354574,
          "p50": 90.0,
          "p90": 120.0,
          "p99": 120.0,
          "p99_9": 120.0,
          "p99_99": 900.0,
          "min": 90,
          "max": 19590
        },
        "intra_ikaros_KS_D": 0.003280000000000005,
        "intra_ikaros_p": 0.9499591129886273,
        "intra_daedalus_KS_D": 0.0015800000000000258,
        "intra_daedalus_p": 0.9999999693864082,
        "inter_KS_D": 0.73953,
        "inter_p": 0.0,
        "inter_intra_D_ratio": 225.4664634146338,
        "pre_reg_pass": true
      }
    }
  },
  "F": {
    "task": "F_nvme",
    "fields": {
      "nvme_latency_ns": {
        "ika_summary": {
          "n": 100000,
          "mean_ns": 49049.68596,
          "std_ns": 26321.839753686265,
          "p50": 48662.0,
          "p90": 49403.0,
          "p99": 54422.0,
          "p99_9": 87504.03000000012,
          "p99_99": 391665.3997999608,
          "min": 42389,
          "max": 7030668
        },
        "dae_summary": {
          "n": 100000,
          "mean_ns": 53714.69857,
          "std_ns": 2932.2365773228385,
          "p50": 53631.0,
          "p90": 54772.0,
          "p99": 56315.0,
          "p99_9": 69420.3910000015,
          "p99_99": 188293.69029993232,
          "min": 24075,
          "max": 264225
        },
        "intra_ikaros_KS_D": 0.0192,
        "intra_ikaros_p": 0.3153878109541569,
        "intra_daedalus_KS_D": 0.0158,
        "intra_daedalus_p": 0.5605413681859366,
        "inter_KS_D": 0.9722,
        "inter_p": 0.0,
        "inter_intra_D_ratio": 50.63541666666667,
        "pre_reg_pass": true
      }
    }
  }
}
```


=== FILE: embodiment12b_analysis.json (9425 chars) ===
```json
{
  "A": {
    "task": "A",
    "feasible": true,
    "sub": {
      "rdrand": {
        "intra_ikaros": {
          "KS_D": 0.5,
          "KS_p": 0.0,
          "n1": 100000,
          "n2": 1000000,
          "label": "ika_now_vs_phase12",
          "s1_p50": 120.0,
          "s2_p50": 150.0,
          "s1_p99_9": 120.0,
          "s2_p99_9": 300.0
        },
        "intra_daedalus": {
          "KS_D": 0.4,
          "KS_p": 0.0,
          "n1": 100000,
          "n2": 1000000,
          "label": "dae_now_vs_phase12",
          "s1_p50": 120.0,
          "s2_p50": 90.0,
          "s1_p99_9": 120.0,
          "s2_p99_9": 120.0
        },
        "inter_12B": {
          "KS_D": 0.0001862315092594491,
          "KS_p": 1.0,
          "n1": 100000,
          "n2": 100000,
          "label": "inter_12B",
          "s1_p50": 120.0,
          "s2_p50": 120.0,
          "s1_p99_9": 120.0,
          "s2_p99_9": 120.0
        },
        "persists": false,
        "same_chassi_stable": false
      },
      "nanosleep": {
        "intra_ikaros": {
          "KS_D": 0.4106494048861975,
          "KS_p": 0.0,
          "n1": 50000,
          "n2": 100000,
          "label": "ika_now_vs_phase12",
          "s1_p50": 53872.0,
          "s2_p50": 54242.0,
          "s1_p99_9": 68128.0,
          "s2_p99_9": 77826.02000000008
        },
        "intra_daedalus": {
          "KS_D": 0.5013714016045304,
          "KS_p": 0.0,
          "n1": 50000,
          "n2": 100000,
          "label": "dae_now_vs_phase12",
          "s1_p50": 53942.0,
          "s2_p50": 53310.0,
          "s1_p99_9": 54342.001000000004,
          "s2_p99_9": 53741.0
        },
        "inter_12B": {
          "KS_D": 0.09780564263322888,
          "KS_p": 1.8279007485807811e-208,
          "n1": 50000,
          "n2": 50000,
          "label": "inter_12B",
          "s1_p50": 53872.0,
          "s2_p50": 53942.0,
          "s1_p99_9": 68128.0,
          "s2_p99_9": 54342.001000000004
        },
        "persists": false,
        "same_chassi_stable": false
      }
    }
  },
  "B": {
    "task": "B",
    "feasible": true,
    "pairs": {
      "1": {
        "KS_D": 0.9117850127968573,
        "KS_p": 0.0,
        "n1": 5000,
        "n2": 5000,
        "label": "inter_core_0_1",
        "s1_p50": 7080.0,
        "s2_p50": 9120.0,
        "s1_p99_9": 16770.180000000037,
        "s2_p99_9": 18180.09000000002
      },
      "2": {
        "KS_D": 0.9101171458998935,
        "KS_p": 0.0,
        "n1": 5000,
        "n2": 5000,
        "label": "inter_core_0_2",
        "s1_p50": 6300.0,
        "s2_p50": 7380.0,
        "s1_p99_9": 16772.280000000464,
        "s2_p99_9": 25952.9400000006
      },
      "4": {
        "KS_D": 0.9146103896103895,
        "KS_p": 0.0,
        "n1": 5000,
        "n2": 5000,
        "label": "inter_core_0_4",
        "s1_p50": 6420.0,
        "s2_p50": 7470.0,
        "s1_p99_9": 61807.77000000158,
        "s2_p99_9": 26945.16000000105
      },
      "7": {
        "KS_D": 0.9130901157505605,
        "KS_p": 0.0,
        "n1": 5000,
        "n2": 5000,
        "label": "inter_core_0_7",
        "s1_p50": 6420.0,
        "s2_p50": 7590.0,
        "s1_p99_9": 17850.93000000019,
        "s2_p99_9": 27330.54000000011
      },
      "8": {
        "KS_D": 0.906693060831416,
        "KS_p": 0.0,
        "n1": 5000,
        "n2": 5000,
        "label": "inter_core_0_8",
        "s1_p50": 6540.0,
        "s2_p50": 7530.0,
        "s1_p99_9": 17250.69000000014,
        "s2_p99_9": 24421.17000000024
      },
      "12": {
        "KS_D": 0.9069937315443195,
        "KS_p": 0.0,
        "n1": 5000,
        "n2": 5000,
        "label": "inter_core_0_12",
        "s1_p50": 6570.0,
        "s2_p50": 7560.0,
        "s1_p99_9": 17250.660000000134,
        "s2_p99_9": 21636.24000000127
      },
      "15": {
        "KS_D": 0.910311962918598,
        "KS_p": 0.0,
        "n1": 5000,
        "n2": 5000,
        "label": "inter_core_0_15",
        "s1_p50": 6540.0,
        "s2_p50": 7710.0,
        "s1_p99_9": 18180.9900000002,
        "s2_p99_9": 42092.79000000057
      }
    },
    "pre_reg_pass": true
  },
  "C": {
    "task": "C",
    "feasible": true,
    "aesenc": {
      "KS_D": 9.638188300542527e-05,
      "KS_p": 1.0,
      "n1": 200000,
      "n2": 200000,
      "label": "aesenc_inter",
      "s1_p50": 30.0,
      "s2_p50": 30.0,
      "s1_p99_9": 90.0,
      "s2_p99_9": 90.0
    },
    "pre_reg_pass": false
  },
  "D": {
    "task": "D",
    "feasible": true,
    "pairs": {
      "0_1": {
        "ika_thp": 117048845.0,
        "dae_thp": 112133035.0,
        "rel_diff": 0.041997936844229436
      },
      "0_2": {
        "ika_thp": 111934653.33333334,
        "dae_thp": 112150958.33333334,
        "rel_diff": 0.0019286950661366764
      },
      "0_4": {
        "ika_thp": 111172045.0,
        "dae_thp": 111816720.0,
        "rel_diff": 0.005765461551725001
      },
      "0_7": {
        "ika_thp": 106900981.66666667,
        "dae_thp": 108354376.66666667,
        "rel_diff": 0.013413348354825721
      },
      "0_8": {
        "ika_thp": 108721333.33333334,
        "dae_thp": 108884806.66666667,
        "rel_diff": 0.0015013419992908257
      },
      "0_15": {
        "ika_thp": 90978400.0,
        "dae_thp": 91926495.0,
        "rel_diff": 0.010313620681393324
      },
      "0_16": {
        "ika_thp": 171053036.6666667,
        "dae_thp": 164954128.33333334,
        "rel_diff": 0.03565507197173218
      }
    },
    "max_rel_diff": 0.041997936844229436,
    "pre_reg_pass": false
  },
  "E": {
    "task": "E",
    "feasible": true,
    "pairs": {
      "0_1": {
        "KS_D": 0.11428571428571432,
        "KS_p": 3.7821369496095115e-114,
        "n1": 20000,
        "n2": 20000,
        "label": "pingpong_0_1",
        "s1_p50": 240.0,
        "s2_p50": 240.0,
        "s1_p99_9": 720.0,
        "s2_p99_9": 690.0300000000061
      },
      "0_15": {
        "KS_D": 0.4,
        "KS_p": 0.0,
        "n1": 20000,
        "n2": 20000,
        "label": "pingpong_0_15",
        "s1_p50": 330.0,
        "s2_p50": 300.0,
        "s1_p99_9": 780.0,
        "s2_p99_9": 1350.0
      },
      "0_16": {
        "KS_D": 0.356043956043949,
        "KS_p": 0.0,
        "n1": 20000,
        "n2": 20000,
        "label": "pingpong_0_16",
        "s1_p50": 630.0,
        "s2_p50": 630.0,
        "s1_p99_9": 1500.0,
        "s2_p99_9": 2010.0
      },
      "0_2": {
        "KS_D": 0.4,
        "KS_p": 0.0,
        "n1": 20000,
        "n2": 20000,
        "label": "pingpong_0_2",
        "s1_p50": 300.0,
        "s2_p50": 330.0,
        "s1_p99_9": 1680.0,
        "s2_p99_9": 630.0
      },
      "0_4": {
        "KS_D": 0.13333333333333341,
        "KS_p": 3.243341267531319e-155,
        "n1": 20000,
        "n2": 20000,
        "label": "pingpong_0_4",
        "s1_p50": 330.0,
        "s2_p50": 360.0,
        "s1_p99_9": 810.0300000000061,
        "s2_p99_9": 660.0300000000061
      },
      "0_7": {
        "KS_D": 0.5,
        "KS_p": 0.0,
        "n1": 20000,
        "n2": 20000,
        "label": "pingpong_0_7",
        "s1_p50": 270.0,
        "s2_p50": 330.0,
        "s1_p99_9": 690.0,
        "s2_p99_9": 600.0
      },
      "0_8": {
        "KS_D": 0.16666666666666666,
        "KS_p": 2.764372983330042e-242,
        "n1": 20000,
        "n2": 20000,
        "label": "pingpong_0_8",
        "s1_p50": 330.0,
        "s2_p50": 330.0,
        "s1_p99_9": 600.0,
        "s2_p99_9": 840.0
      }
    },
    "frobenius_p50_diff": 79.37253933193772,
    "pre_reg_pass": true
  },
  "F": {
    "task": "F",
    "feasible_ika": false,
    "feasible_dae": false,
    "pre_reg_pass": null,
    "reason": "TPM requires root; not run"
  },
  "G": {
    "task": "G",
    "feasible": true,
    "walk": {
      "KS_D": 0.01904761904761909,
      "KS_p": 5.802395067573232e-32,
      "n1": 200000,
      "n2": 200000,
      "label": "walk",
      "s1_p50": 210.0,
      "s2_p50": 210.0,
      "s1_p99_9": 591.0,
      "s2_p99_9": 481.0
    },
    "spike_intervals": {
      "KS_D": 0.10136870475083237,
      "KS_p": 0.9020053041474743,
      "n1": 66,
      "n2": 55,
      "label": "spikes",
      "s1_p50": 3237.5,
      "s2_p50": 3946.0,
      "s1_p99_9": 7944.31,
      "s2_p99_9": 8109.5160000000005
    },
    "ika_spike_count": 67,
    "dae_spike_count": 56,
    "pre_reg_pass": true
  },
  "H": {
    "task": "H",
    "feasible": true,
    "n_common_devs": 20,
    "p50_dist_ika": {
      "n": 20,
      "min": 3757.0,
      "max": 16962.0,
      "p50": 7734.0,
      "p90": 16711.5,
      "p99": 16962.0,
      "p99_9": 16962.0
    },
    "p50_dist_dae": {
      "n": 20,
      "min": 3878.0,
      "max": 17172.0,
      "p50": 7825.0,
      "p90": 16141.0,
      "p99": 17172.0,
      "p99_9": 17172.0
    },
    "p50_dist_KS": {
      "KS_D": 0.04980116391852574,
      "KS_p": 1.0,
      "n1": 20,
      "n2": 20,
      "label": "pcie_p50_dist",
      "s1_p50": 7734.0,
      "s2_p50": 7825.0,
      "s1_p99_9": 16962.0,
      "s2_p99_9": 17172.0
    },
    "pre_reg_pass": false
  },
  "_summary": {
    "feasible_tasks": [
      "A",
      "B",
      "C",
      "D",
      "E",
      "G",
      "H"
    ],
    "passed_tasks": [
      "B",
      "E",
      "G"
    ],
    "phase12_passed": [
      "D",
      "E",
      "F"
    ],
    "combined_hal_bypass_count": 6,
    "task_A_per_chassi_stable": false
  }
}
```


=== FILE: embodiment9_fan_control.json (9645 chars) ===
```json
{
  "host_running": "ikaros",
  "n_seeds": 30,
  "matrix": {
    "ikaros": {
      "learned_ikaros": {
        "rms_per_run": [
          6.453742504119873,
          6.429311275482178,
          6.464639663696289,
          6.479764938354492,
          6.453672409057617,
          6.473254680633545,
          6.438638687133789,
          6.468234539031982,
          6.473327159881592,
          6.438295364379883,
          6.482516765594482,
          6.462693691253662,
          6.4628825187683105,
          6.443923473358154,
          6.464788436889648,
          6.477673053741455,
          6.464519023895264,
          6.433860778808594,
          6.444497585296631,
          6.4534831047058105,
          6.430213451385498,
          6.432036399841309,
          6.452376365661621,
          6.426074981689453,
          6.464352607727051,
          6.488762855529785,
          6.468108177185059,
          6.4456706047058105,
          6.446887016296387,
          6.461492538452148
        ],
        "rms_mean": 6.4559898217519125,
        "rms_std": 0.016903375319314774,
        "energy_mean": 2.876397490642256e-10,
        "ci95": [
          6.449933997392654,
          6.462163499593735
        ]
      },
      "learned_daedalus": {
        "rms_per_run": [
          12.860246658325195,
          12.868061065673828,
          12.869775772094727,
          12.865479469299316,
          12.864599227905273,
          12.866194725036621,
          12.876426696777344,
          12.859145164489746,
          12.867505073547363,
          12.868022918701172,
          12.86164379119873,
          12.861058235168457,
          12.863667488098145,
          12.870697975158691,
          12.862579345703125,
          12.863142013549805,
          12.85797119140625,
          12.873360633850098,
          12.873161315917969,
          12.855403900146484,
          12.877945899963379,
          12.879189491271973,
          12.863004684448242,
          12.880536079406738,
          12.86342716217041,
          12.863990783691406,
          12.861448287963867,
          12.864771842956543,
          12.862884521484375,
          12.860033988952637
        ],
        "rms_mean": 12.866179180145263,
        "rms_std": 0.006332654019744579,
        "energy_mean": 143.07744851368386,
        "ci95": [
          12.863989380995433,
          12.868491913477579
        ]
      },
      "constant_pwm": {
        "rms_per_run": [
          89.50154876708984,
          89.48905181884766,
          89.51539611816406,
          89.52247619628906,
          89.51072692871094,
          89.5180435180664,
          89.50447845458984,
          89.50813293457031,
          89.51775360107422,
          89.49961853027344,
          89.51543426513672,
          89.51287078857422,
          89.50831604003906,
          89.49838256835938,
          89.51263427734375,
          89.51766967773438,
          89.5106430053711,
          89.48931884765625,
          89.5010757446289,
          89.49803161621094,
          89.48802947998047,
          89.48310089111328,
          89.49931335449219,
          89.49359130859375,
          89.51852416992188,
          89.5213394165039,
          89.51025390625,
          89.49945068359375,
          89.50341796875,
          89.50143432617188
        ],
        "rms_mean": 89.50566864013672,
        "rms_std": 0.010504992907684424,
        "energy_mean": 16384.0,
        "ci95": [
          89.50171014785766,
          89.50952061971029
        ]
      },
      "pid_default": {
        "rms_per_run": [
          7.336228370666504,
          7.434184551239014,
          7.384931564331055,
          7.291076183319092,
          7.362667560577393,
          7.319697380065918,
          7.456412315368652,
          7.277472019195557,
          7.344374656677246,
          7.420573711395264,
          7.238080978393555,
          7.309367656707764,
          7.337182521820068,
          7.422492980957031,
          7.3231353759765625,
          7.260342597961426,
          7.2755632400512695,
          7.451548099517822,
          7.4365997314453125,
          7.290458679199219,
          7.472929000854492,
          7.476901531219482,
          7.365755081176758,
          7.492744445800781,
          7.329339504241943,
          7.225098609924316,
          7.28971004486084,
          7.387513160705566,
          7.382415294647217,
          7.30383825302124
        ],
        "rms_mean": 7.356621170043946,
        "rms_std": 0.07408855407106939,
        "energy_mean": 80.9446972532139,
        "ci95": [
          7.3297759890556335,
          7.382710958719254
        ]
      }
    },
    "daedalus": {
      "learned_ikaros": {
        "rms_per_run": [
          28.833921432495117,
          28.852815628051758,
          28.822755813598633,
          28.81113052368164,
          28.82648468017578,
          28.81600570678711,
          28.838247299194336,
          28.824434280395508,
          28.81802749633789,
          28.841005325317383,
          28.81464195251465,
          28.821203231811523,
          28.82645606994629,
          28.842510223388672,
          28.821365356445312,
          28.81353759765625,
          28.82131576538086,
          28.853347778320312,
          28.8408203125,
          28.834720611572266,
          28.856719970703125,
          28.861928939819336,
          28.83805274963379,
          28.851266860961914,
          28.816452026367188,
          28.80754280090332,
          28.822750091552734,
          28.838977813720703,
          28.833391189575195,
          28.832605361938477
        ],
        "rms_mean": 28.83114782969157,
        "rms_std": 0.014322754669949716,
        "energy_mean": 1.996617057218684e-06,
        "ci95": [
          28.825996742248535,
          28.83653211116791
        ]
      },
      "learned_daedalus": {
        "rms_per_run": [
          0.42155686020851135,
          0.42601022124290466,
          0.4253270924091339,
          0.41621115803718567,
          0.42565158009529114,
          0.4150821566581726,
          0.43030667304992676,
          0.41404563188552856,
          0.41842156648635864,
          0.4293016195297241,
          0.4196241796016693,
          0.4183776378631592,
          0.421798437833786,
          0.4261573255062103,
          0.41665545105934143,
          0.41745269298553467,
          0.41608086228370667,
          0.4263041913509369,
          0.4278120696544647,
          0.4152248799800873,
          0.429543137550354,
          0.4285101592540741,
          0.42091819643974304,
          0.42903581261634827,
          0.42059653997421265,
          0.41591107845306396,
          0.4172680675983429,
          0.425197958946228,
          0.42539912462234497,
          0.41792032122612
        ],
        "rms_mean": 0.42192342281341555,
        "rms_std": 0.0051272355396225185,
        "energy_mean": 517.9498256771154,
        "ci95": [
          0.42005785763263703,
          0.42369601532816886
        ]
      },
      "constant_pwm": {
        "rms_per_run": [
          136.23623657226562,
          136.21768188476562,
          136.24742126464844,
          136.259033203125,
          136.2437286376953,
          136.25405883789062,
          136.23240661621094,
          136.24571228027344,
          136.25205993652344,
          136.2294158935547,
          136.2555389404297,
          136.24896240234375,
          136.2435760498047,
          136.22776794433594,
          136.2487335205078,
          136.25656127929688,
          136.24880981445312,
          136.2171173095703,
          136.22958374023438,
          136.23541259765625,
          136.2139129638672,
          136.2085418701172,
          136.23211669921875,
          136.21949768066406,
          136.25360107421875,
          136.2626495361328,
          136.24732971191406,
          136.2312469482422,
          136.23680114746094,
          136.23745727539062
        ],
        "rms_mean": 136.23909912109374,
        "rms_std": 0.014174798323099529,
        "energy_mean": 16384.0,
        "ci95": [
          136.2337811279297,
          136.24421136220295
        ]
      },
      "pid_default": {
        "rms_per_run": [
          4.942447185516357,
          5.014322757720947,
          4.903531074523926,
          4.928488731384277,
          5.01591682434082,
          4.80706787109375,
          5.078847408294678,
          4.801852226257324,
          5.065566539764404,
          4.990355491638184,
          4.914665699005127,
          4.962125301361084,
          5.054962635040283,
          5.049841403961182,
          4.921269416809082,
          4.928918838500977,
          4.905907154083252,
          5.127997875213623,
          5.109594345092773,
          5.036042213439941,
          5.057166576385498,
          5.024570941925049,
          5.025183200836182,
          5.009716033935547,
          4.897049427032471,
          4.899777889251709,
          4.884124279022217,
          4.86823034286499,
          5.047109603881836,
          5.030696868896484
        ],
        "rms_mean": 4.976778205235799,
        "rms_std": 0.08412742462046231,
        "energy_mean": 1548.4778152051347,
        "ci95": [
          4.945159902969996,
          5.004276587963104
        ]
      }
    }
  },
  "gates": {
    "learned_ikaros_beats_worst_baseline_20pct": true,
    "learned_ikaros_beats_transplant_5pct": true,
    "worst_baseline_rms": 89.50566864013672,
    "learned_ikaros_rms": 6.4559898217519125,
    "learned_daedalus_on_ikaros_rms": 12.866179180145263
  }
}
```
