# z456 — R_body reset path (SB_HOT base + body-leak resistor)

## Pre-registered gates

- **INFRA_pass**: True
- **DISCOVERY_pass**: False
- **DISCOVERY_who**: []
- **AMBITIOUS_pass**: False
- **AMBITIOUS_who**: []
- **KILL_SHOT**: True
- **kill_shot_reason**: no_self_reset

## DC (forward / backward / avg) [dec]

| R_body | τ_est | DC_fwd | DC_bwd | DC_avg | n |
|---|---|---|---|---|---|
| R_INF (INF) | inf | 2.795 | 2.824 | 2.809 | 25 |
| R_1G (1.0e+09 Ω) | 2.7e-03s | 2.795 | 2.824 | 2.809 | 25 |
| R_100M (1.0e+08 Ω) | 2.7e-04s | 2.795 | 2.824 | 2.809 | 25 |
| R_10M (1.0e+07 Ω) | 2.7e-05s | 2.795 | 2.824 | 2.809 | 25 |
| R_1M (1.0e+06 Ω) | 2.7e-06s | 2.795 | 2.824 | 2.809 | 25 |

NOTE: R_body is wired into the transient body-KCL only. DC pathway
uses z429.run_vsint_pinned which is unchanged → DC values match SB_HOT
baseline. This is intentional (see script header).


## Fast-pulse extended (1µs hold) — self-reset timings

### R_INF (R_body=None, τ_est=infs)
| bias | Vb_peak | t→0.5V[ns] | t→reset[ns] | self-reset | n_cycles | period[ns] |
|---|---|---|---|---|---|---|
| VG1_0p6_VG2_0p0 | 0.661 | 1.3740000000000003 | None | False | 1 | None |
| VG1_0p6_VG2_0p2 | 0.661 | 1.3740000000000003 | None | False | 1 | None |
| VG1_0p6_VG2_0p4 | 0.657 | 1.4020000000000004 | None | False | 1 | None |
| VG1_0p4_VG2_0p0 | 0.640 | 8.30603541993147 | None | False | 1 | None |

### R_1G (R_body=1000000000.0, τ_est=2.7e-03s)
| bias | Vb_peak | t→0.5V[ns] | t→reset[ns] | self-reset | n_cycles | period[ns] |
|---|---|---|---|---|---|---|
| VG1_0p6_VG2_0p0 | 0.661 | 1.3740000000000003 | None | False | 1 | None |
| VG1_0p6_VG2_0p2 | 0.661 | 1.3740000000000003 | None | False | 1 | None |
| VG1_0p6_VG2_0p4 | 0.657 | 1.4020000000000004 | None | False | 1 | None |
| VG1_0p4_VG2_0p0 | 0.640 | 8.30603541993147 | None | False | 1 | None |

### R_100M (R_body=100000000.0, τ_est=2.7e-04s)
| bias | Vb_peak | t→0.5V[ns] | t→reset[ns] | self-reset | n_cycles | period[ns] |
|---|---|---|---|---|---|---|
| VG1_0p6_VG2_0p0 | 0.661 | 1.3880000000000001 | None | False | 1 | None |
| VG1_0p6_VG2_0p2 | 0.661 | 1.3880000000000001 | None | False | 1 | None |
| VG1_0p6_VG2_0p4 | 0.657 | 1.4020000000000004 | None | False | 1 | None |
| VG1_0p4_VG2_0p0 | 0.640 | 8.377057635184414 | None | False | 1 | None |

### R_10M (R_body=10000000.0, τ_est=2.7e-05s)
| bias | Vb_peak | t→0.5V[ns] | t→reset[ns] | self-reset | n_cycles | period[ns] |
|---|---|---|---|---|---|---|
| VG1_0p6_VG2_0p0 | 0.661 | 1.4020000000000004 | None | False | 1 | None |
| VG1_0p6_VG2_0p2 | 0.661 | 1.4020000000000004 | None | False | 1 | None |
| VG1_0p6_VG2_0p4 | 0.657 | 1.4300000000000002 | None | False | 1 | None |
| VG1_0p4_VG2_0p0 | 0.640 | 9.045774957672325 | None | False | 1 | None |

### R_1M (R_body=1000000.0, τ_est=2.7e-06s)
| bias | Vb_peak | t→0.5V[ns] | t→reset[ns] | self-reset | n_cycles | period[ns] |
|---|---|---|---|---|---|---|
| VG1_0p6_VG2_0p0 | 0.661 | 1.6680000000000004 | None | False | 1 | None |
| VG1_0p6_VG2_0p2 | 0.661 | 1.6680000000000004 | None | False | 1 | None |
| VG1_0p6_VG2_0p4 | 0.657 | 1.6960000000000004 | None | False | 1 | None |
| VG1_0p4_VG2_0p0 | 0.010 | None | None | False | 0 | None |


## Best (lowest DC_avg): **R_INF**, DC_avg=2.809


## Mario slide-21 reference: ~400 ns oscillation period
(quoted as benchmark — not target-matched, not optimized to fit)
