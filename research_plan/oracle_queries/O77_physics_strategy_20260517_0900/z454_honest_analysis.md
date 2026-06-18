# z454 — Snapback subcircuit integration on v449_B base

## Pre-registered gates

- **INFRA_pass**: True
- **DISCOVERY_pass**: False
- **DISCOVERY_who**: None
- **AMBITIOUS_pass**: False
- **AMBITIOUS_who**: None
- **KILL_SHOT**: True
- **kill_shot_reason**: all_SB_worse_than_SB_OFF

## DC (forward / backward / avg) [dec]

| condition | DC_fwd | DC_bwd | DC_avg | n |
|---|---|---|---|---|
| SB_OFF | 1.311 | 2.864 | 2.087 | 25 |
| SB_ON_DEFAULT | 2.686 | 2.707 | 2.696 | 25 |
| SB_LOW | 2.627 | 2.691 | 2.659 | 25 |
| SB_HOT | 2.795 | 2.824 | 2.809 | 25 |

## Fast-pulse smoke (per bias)

### SB_OFF
| bias | Vb_peak | t_peak[ns] | Vb_5ns | Vb_10ns | t→0.3V[ns] | t→0.5V[ns] | self-reset |
|---|---|---|---|---|---|---|---|
| VG1_0p6_VG2_0p0 | 0.054 | 15.70 | 0.027 | 0.054 | None | None | False |
| VG1_0p6_VG2_0p2 | 0.054 | 15.70 | 0.027 | 0.054 | None | None | False |
| VG1_0p6_VG2_0p4 | 0.054 | 15.70 | 0.027 | 0.054 | None | None | False |
| VG1_0p4_VG2_0p0 | 0.003 | 15.70 | 0.002 | 0.003 | None | None | False |

### SB_ON_DEFAULT
| bias | Vb_peak | t_peak[ns] | Vb_5ns | Vb_10ns | t→0.3V[ns] | t→0.5V[ns] | self-reset |
|---|---|---|---|---|---|---|---|
| VG1_0p6_VG2_0p0 | 0.709 | 15.70 | 0.636 | 0.636 | 3.133416770963705 | 3.1530663329161457 | False |
| VG1_0p6_VG2_0p2 | 0.709 | 15.70 | 0.636 | 0.636 | 3.133416770963705 | 3.1530663329161457 | False |
| VG1_0p6_VG2_0p4 | 0.709 | 15.70 | 0.636 | 0.636 | 3.1923654568210265 | 3.2120150187734673 | False |
| VG1_0p4_VG2_0p0 | 0.651 | 15.70 | 0.012 | 0.029 | 14.589111389236548 | 14.608760951188989 | False |

### SB_LOW
| bias | Vb_peak | t_peak[ns] | Vb_5ns | Vb_10ns | t→0.3V[ns] | t→0.5V[ns] | self-reset |
|---|---|---|---|---|---|---|---|
| VG1_0p6_VG2_0p0 | 0.709 | 15.70 | 0.637 | 0.637 | 1.9740926157697125 | 1.9937421777221531 | False |
| VG1_0p6_VG2_0p2 | 0.709 | 15.70 | 0.637 | 0.637 | 1.9740926157697125 | 1.9937421777221531 | False |
| VG1_0p6_VG2_0p4 | 0.709 | 15.70 | 0.637 | 0.637 | 2.0526908635794747 | 2.072340425531915 | False |
| VG1_0p4_VG2_0p0 | 0.042 | 15.70 | 0.012 | 0.027 | None | None | False |

### SB_HOT
| bias | Vb_peak | t_peak[ns] | Vb_5ns | Vb_10ns | t→0.3V[ns] | t→0.5V[ns] | self-reset |
|---|---|---|---|---|---|---|---|
| VG1_0p6_VG2_0p0 | 0.710 | 15.70 | 0.661 | 0.661 | 1.3649561952440554 | 1.3846057571964958 | False |
| VG1_0p6_VG2_0p2 | 0.710 | 15.70 | 0.661 | 0.661 | 1.3649561952440554 | 1.3846057571964958 | False |
| VG1_0p6_VG2_0p4 | 0.710 | 15.70 | 0.657 | 0.657 | 1.3846057571964958 | 1.4042553191489364 | False |
| VG1_0p4_VG2_0p0 | 0.709 | 15.70 | 0.019 | 0.640 | 8.28160200250313 | 8.301251564455569 | False |


## Snapback assert (z444-BURN avoidance)

- **SB_OFF**: snap_called=False |I_snap_d|=0.000e+00 A |I_snap_b|=0.000e+00 A V_db=NA BV=NA
- **SB_ON_DEFAULT**: snap_called=True |I_snap_d|=1.000e-02 A |I_snap_b|=5.485e-07 A V_db=1.4 BV=2.0
- **SB_LOW**: snap_called=True |I_snap_d|=1.000e-02 A |I_snap_b|=2.459e-06 A V_db=1.4 BV=1.6
- **SB_HOT**: snap_called=True |I_snap_d|=1.000e-02 A |I_snap_b|=4.232e-05 A V_db=1.4 BV=1.2

## Best condition: **SB_OFF**

- DC_avg = 2.087 dec
- vs SB_OFF DC_avg = 2.087 dec  (Δ = +0.000 dec)