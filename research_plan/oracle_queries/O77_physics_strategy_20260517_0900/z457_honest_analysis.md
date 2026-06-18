# z457 — NPN-collector σ-gate on top of z455 K_1p6

## Pre-registered gates

- **INFRA_pass**: True
- **DISCOVERY_pass**: False
- **DISCOVERY_who**: None
- **AMBITIOUS_pass**: False
- **AMBITIOUS_who**: None
- **KILL_SHOT**: False
- **kill_shot_reason**: None
- **VG1_0p2_regression_alerts**: [{'name': 'NY_1p6', 'branch': 'VG1=0.2 fwd', 'off': 2.30599288283575, 'this': 4.4844361404976025, 'delta': 2.1784432576618524}, {'name': 'NY_1p6', 'branch': 'VG1=0.2 bwd', 'off': 2.305992882835761, 'this': 3.871755165950178, 'delta': 1.5657622831144167}, {'name': 'NY_1p8', 'branch': 'VG1=0.2 fwd', 'off': 2.30599288283575, 'this': 4.4844361404976025, 'delta': 2.1784432576618524}, {'name': 'NY_1p8', 'branch': 'VG1=0.2 bwd', 'off': 2.305992882835761, 'this': 4.320333144553118, 'delta': 2.0143402617173565}]

## DC (forward / backward / avg) [dec]

| condition | mode | V_knee_npn | DC_fwd | DC_bwd | DC_avg | n |
|---|---|---|---|---|---|---|
| N_OFF | off | 1.60 | 2.686 | 2.719 | 2.702 | 25 |
| NX_1p4 | current | 1.40 | 2.622 | 2.785 | 2.703 | 25 |
| NX_1p6 | current | 1.60 | 2.629 | 2.791 | 2.710 | 25 |
| NX_1p8 | current | 1.80 | 2.368 | 2.589 | 2.479 | 25 |
| NY_1p6 | vbe | 1.60 | 4.292 | 4.160 | 4.226 | 25 |
| NY_1p8 | vbe | 1.80 | 4.262 | 3.543 | 3.902 | 25 |

## Per-branch DC (forward) — average log-RMSE by VG1

| condition | VG1=0.2 | VG1=0.4 | VG1=0.6 |
|---|---|---|---|
| N_OFF | 2.306 | 2.429 | 2.998 |
| NX_1p4 | 2.318 | 2.369 | 2.888 |
| NX_1p6 | 2.282 | 2.344 | 2.931 |
| NX_1p8 | 2.242 | 2.377 | 2.370 |
| NY_1p6 | 4.484 | 4.041 | 4.189 |
| NY_1p8 | 4.484 | 3.915 | 4.197 |

## I_snap_d at mid-DC (V_db≈0.8V, VG1=0.6, VG2=0.0)

| condition | mode | V_knee_npn | V_db | |I_snap_d| [A] | |I_snap_b| [A] |
|---|---|---|---|---|---|
| N_OFF | off | 1.60 | 0.80 | 1.000e-02 | 1.690e-14 |
| NX_1p4 | current | 1.40 | 0.80 | 1.000e-02 | 1.690e-14 |
| NX_1p6 | current | 1.60 | 0.80 | 8.892e-03 | 1.690e-14 |
| NX_1p8 | current | 1.80 | 0.80 | 1.629e-04 | 1.690e-14 |
| NY_1p6 | vbe | 1.60 | 0.80 | 1.000e-02 | 1.690e-14 |
| NY_1p8 | vbe | 1.80 | 0.80 | 1.000e-02 | 1.690e-14 |

## Fast-pulse smoke

### N_OFF  (mode=off, V_knee_npn=1.60)
| bias | Vb_peak | Vb_5ns | Vb_10ns | t→0.3V[ns] | t→0.5V[ns] | self-reset |
|---|---|---|---|---|---|---|
| VG1_0p6_VG2_0p0 | 0.709 | 0.635 | 0.635 | 1.4042553191489364 | 1.4239048811013773 | False |
| VG1_0p6_VG2_0p2 | 0.709 | 0.635 | 0.635 | 1.4042553191489364 | 1.4239048811013773 | False |
| VG1_0p6_VG2_0p4 | 0.709 | 0.635 | 0.635 | 1.4239048811013773 | 1.4435544430538176 | False |
| VG1_0p4_VG2_0p0 | 0.709 | 0.019 | 0.635 | 8.340550688360452 | 8.360200250312891 | False |

### NX_1p4  (mode=current, V_knee_npn=1.40)
| bias | Vb_peak | Vb_5ns | Vb_10ns | t→0.3V[ns] | t→0.5V[ns] | self-reset |
|---|---|---|---|---|---|---|
| VG1_0p6_VG2_0p0 | 0.638 | 0.635 | 0.635 | 1.4042553191489364 | 1.4239048811013773 | False |
| VG1_0p6_VG2_0p2 | 0.638 | 0.635 | 0.635 | 1.4042553191489364 | 1.4239048811013773 | False |
| VG1_0p6_VG2_0p4 | 0.638 | 0.635 | 0.635 | 1.4239048811013773 | 1.4435544430538176 | False |
| VG1_0p4_VG2_0p0 | 0.637 | 0.019 | 0.635 | 8.340550688360452 | 8.360200250312891 | False |

### NX_1p6  (mode=current, V_knee_npn=1.60)
| bias | Vb_peak | Vb_5ns | Vb_10ns | t→0.3V[ns] | t→0.5V[ns] | self-reset |
|---|---|---|---|---|---|---|
| VG1_0p6_VG2_0p0 | 0.638 | 0.635 | 0.635 | 1.4042553191489364 | 1.4239048811013773 | False |
| VG1_0p6_VG2_0p2 | 0.638 | 0.635 | 0.635 | 1.4042553191489364 | 1.4239048811013773 | False |
| VG1_0p6_VG2_0p4 | 0.638 | 0.635 | 0.635 | 1.4239048811013773 | 1.4435544430538176 | False |
| VG1_0p4_VG2_0p0 | 0.637 | 0.019 | 0.635 | 8.340550688360452 | 8.360200250312891 | False |

### NX_1p8  (mode=current, V_knee_npn=1.80)
| bias | Vb_peak | Vb_5ns | Vb_10ns | t→0.3V[ns] | t→0.5V[ns] | self-reset |
|---|---|---|---|---|---|---|
| VG1_0p6_VG2_0p0 | 0.636 | 0.635 | 0.635 | 1.4435544430538176 | 1.463204005006258 | False |
| VG1_0p6_VG2_0p2 | 0.636 | 0.635 | 0.635 | 1.4435544430538176 | 1.463204005006258 | False |
| VG1_0p6_VG2_0p4 | 0.636 | 0.635 | 0.635 | 1.463204005006258 | 1.4828535669586984 | False |
| VG1_0p4_VG2_0p0 | 0.636 | 0.019 | 0.635 | 8.537046307884857 | 8.556695869837299 | False |

### NY_1p6  (mode=vbe, V_knee_npn=1.60)
| bias | Vb_peak | Vb_5ns | Vb_10ns | t→0.3V[ns] | t→0.5V[ns] | self-reset |
|---|---|---|---|---|---|---|
| VG1_0p6_VG2_0p0 | 0.709 | 0.635 | 0.635 | 1.5811013767209017 | 1.600750938673342 | False |
| VG1_0p6_VG2_0p2 | 0.709 | 0.635 | 0.635 | 1.5811013767209017 | 1.600750938673342 | False |
| VG1_0p6_VG2_0p4 | 0.709 | 0.635 | 0.635 | 1.600750938673342 | 1.6204005006257824 | False |
| VG1_0p4_VG2_0p0 | 0.031 | 0.003 | 0.030 | None | None | False |

### NY_1p8  (mode=vbe, V_knee_npn=1.80)
| bias | Vb_peak | Vb_5ns | Vb_10ns | t→0.3V[ns] | t→0.5V[ns] | self-reset |
|---|---|---|---|---|---|---|
| VG1_0p6_VG2_0p0 | 0.709 | 0.635 | 0.635 | 2.563579474342929 | 3.1137672090112645 | False |
| VG1_0p6_VG2_0p2 | 0.709 | 0.635 | 0.635 | 2.563579474342929 | 3.1137672090112645 | False |
| VG1_0p6_VG2_0p4 | 0.709 | 0.635 | 0.635 | 2.701126408010013 | 3.329912390488111 | False |
| VG1_0p4_VG2_0p0 | 0.007 | 0.000 | 0.007 | None | None | False |


## Snapback assert (deep, V_db=1.4V)

- **N_OFF**: snap_called=True |I_snap_d|=1.000e-02A |I_snap_b|=3.052e-08A V_db=1.4 BV=1.2
- **NX_1p4**: snap_called=True |I_snap_d|=1.000e-02A |I_snap_b|=3.052e-08A V_db=1.4 BV=1.2
- **NX_1p6**: snap_called=True |I_snap_d|=1.000e-02A |I_snap_b|=3.052e-08A V_db=1.4 BV=1.2
- **NX_1p8**: snap_called=True |I_snap_d|=1.000e-02A |I_snap_b|=3.052e-08A V_db=1.4 BV=1.2
- **NY_1p6**: snap_called=True |I_snap_d|=1.000e-02A |I_snap_b|=3.052e-08A V_db=1.4 BV=1.2
- **NY_1p8**: snap_called=True |I_snap_d|=1.000e-02A |I_snap_b|=3.052e-08A V_db=1.4 BV=1.2

## Best condition: **NX_1p8** (mode=current, V_knee_npn=1.80)

- DC_avg = 2.479 dec
- vs N_OFF DC_avg = 2.702 dec  (Δ = -0.223 dec)
- vs z455 K_1p6 reference = 2.702 dec  (Δ = -0.223 dec)

## Gate-wiring verdict

- **NX_1p4** (current, V_knee_npn=1.40): |I_snap_d|_mid = 1.000e-02 A → INEFFECTIVE (mid-DC I_snap_d still ≈ 10 mA clamp — σ insufficient at chosen V_sharp; either intrinsic Ic is large enough to overpower the gate, or wiring bug)
- **NX_1p6** (current, V_knee_npn=1.60): |I_snap_d|_mid = 8.892e-03 A → PARTIAL (mid-DC I_snap_d below clamp but not negligible)
- **NX_1p8** (current, V_knee_npn=1.80): |I_snap_d|_mid = 1.629e-04 A → EFFECTIVE (mid-DC I_snap_d < 1 mA)
- **NY_1p6** (vbe, V_knee_npn=1.60): |I_snap_d|_mid = 1.000e-02 A → INEFFECTIVE (mid-DC I_snap_d still ≈ 10 mA clamp — σ insufficient at chosen V_sharp; either intrinsic Ic is large enough to overpower the gate, or wiring bug)
- **NY_1p8** (vbe, V_knee_npn=1.80): |I_snap_d|_mid = 1.000e-02 A → INEFFECTIVE (mid-DC I_snap_d still ≈ 10 mA clamp — σ insufficient at chosen V_sharp; either intrinsic Ic is large enough to overpower the gate, or wiring bug)