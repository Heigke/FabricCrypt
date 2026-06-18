# Mario target table — z465 best

Best params: snap_Is=2.8813e-07, R_body=3.8723e+05, β=Bf=10000.00, C_body=5.6292e-15

Full DC RMSE = **1.373 dec**

| Target | Mario | Achieved (hi-res) | Rel-err | Within 30% | Weight |
|---|---|---|---|---|---|
| period_s | 4.300e-07 s | 4.299e-07 | 0.000 | PASS | 0.25 |
| Vd_peak_V | 1.890e+00 V | 1.890e+00 | 0.000 | PASS | 0.1 |
| Id_peak_A | 4.800e-03 A | 2.047e-07 | 1.000 | FAIL | 0.15 |
| rise_s | 2.600e-08 s | 6.594e-09 | 0.746 | FAIL | 0.15 |
| fall_s | 7.600e-08 s | 3.154e-08 | 0.585 | FAIL | 0.1 |
| Vbody_swing_V | 2.000e-01 V | 3.940e-01 | 0.970 | FAIL | 0.15 |
| E_spike_J | 2.000e-13 J | 1.383e-14 | 0.931 | FAIL | 0.1 |

**n_within_30 (hi-res) = 2/7**
**Gate verdict: INFRA_ONLY**