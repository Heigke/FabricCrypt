# Canonical Mario/Sebas NS-RAM Parameters — 2026-05-16

**Source**: `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/nsram/Zoom/` (uncontaminated original).

When S9 completes, dispatch S10 with these EXACT canonical values. NO BBO over them — they're already-fitted by Sebas.

## Files (canonical)

- `nsram/Zoom/schematic&modelCards/2tnsram_simple.asc` — LTspice schematic
- `nsram/Zoom/schematic&modelCards/parasiticBJT.txt` — exact NPN card
- `nsram/Zoom/schematic&modelCards/PTM130bulkNSRAM.txt` — BSIM4 card (M1)
- `nsram/Zoom/pdiode.txt` — body diode card
- `nsram/Zoom/mail.txt` — email thread (Eric/Mario/Sebas)
- `nsram/Zoom/2026-04-30 BSIMfitsBA/2Tcell_BSIM_param_DC.csv` — Sebas's per-(VG1,VG2) DC fit (33 rows)

## Schematic topology
```
D ─── Drain
  ├── M1.D (NMOS, Ln=0.18µm, Wn=0.36µm)
  │
Sint ── internal source between M1 and M2
  ├── M1.S
  ├── M2.D
  ├── Q1.E (BJT emitter)
  │
S ─── Source = 0V
  └── M2.S (NMOS, **Ln=1.8µm** (=Ln*10), Wn=0.36µm)
  
B ─── Body (floating, with C_B=1fF to GND, pdiode to N-well at vnwell=2V)
  ├── M1.B = M2.B
  └── Q1.B (BJT base)

Q1: NPN, area=1µm², parasiticBJT card
  C = Drain (D)
  B = Body
  E = Sint
```

## Canonical parameter values

### BSIM4 (PTM130bulkNSRAM.txt — M1)
| Param | Value |
|---|---|
| Vth0 | **0.54** |
| toxe | 4.0e-9 (4 nm) |
| toxn | 4.0e-9 |
| Nparam (NFACTOR) | **1.58** (M1 base; M2 has VG2-dep curve 2-12) |
| Citparam | 0 |
| Voffparam | -0.1368 |
| K2Par | -0.070435 |
| Lint | 1.969e-8 |
| alpha0 | 7.83756e-5 |
| beta0 | **18** (we used 19) |
| agidl | 1.99e-8 |
| k1 | **0.63825** |
| etab | **-0.086777** (NEGATIVE) |
| Vsat | 1.35e5 |
| u0 | 0.048317 |

### Parasitic NPN (parasiticBJT.txt)
| Param | Value |
|---|---|
| IS | 5e-9 |
| Va | 100 |
| Bf | **10000** |
| Br | 100 |
| NC | 2 |
| Ikr | 100m |
| Rc | 0.1 |
| Re | 0.1 |
| Vje | 0.7 |
| Cjc | 1e-15 |
| Cje | 0.7e-15 |
| Ne | 1.5 |
| Tr | 20e-12 (20 ps) |
| Tf | 25e-12 (25 ps) |
| ITF | 0.03 |
| VTF | 7 |
| XTF | 2 |
| area | 1µm² |

### Body pdiode (pdiode.txt)
| Param | Value |
|---|---|
| BV | **11** |
| IS | 5.37e-7 |
| ISW | 1.37e-13 |
| RSW | 0.465 |
| NS | 1.085 |
| Cj | 7.33e-4 |
| Cjsw | 1.05e-10 |
| Vj | 0.219 |
| Vjsw | 0.652 |
| M | 0.241 |
| Mjsw | 0.260 |
| N | 1.054 |
| Rs | 7.42e-8 |
| Eg | 1.11 |

### Circuit-level
| Param | Value |
|---|---|
| **C_B (CBpar)** | **1 fF** (we use 8 fF — 8× too big) |
| Vnwell | 2 V (for slow-DC sweeps), 0 V (transient slides) |
| sweep rate (DC) | 0.2 V/s |
| τ_rise | 100 ps |
| τ_relax | ~1 µs |
| self-reset period | ~0.4 µs at Vread=2V |

## Mario's snapback formula (slide 12.26, never ported)
```
Ipos = Iexp + Ipow
  Ipow(V_D, V_G2) = a(V_G2) · (V_D − y(V_G2))^β(V_G2)   for V_D > y, else 0
  Iexp(V_D, V_G2) = c(V_G2) · exp(d(V_G2) · V_D)
```
- 5 PWL functions of V_G2: a, β, c, d, y
- Per VG1 separate curves
- Coefficients need digitisation from slide insets (~30 min WebPlotDigitizer)
- This is the residual term INJECTED at the bulk node (B)

## Deviations from our pyport (cumulative impact)

| # | Param | Pyport had | Canonical | Effect |
|---|---|---|---|---|
| 1 | C_B | 8 fF | **1 fF** | Body time-constant 8× too slow |
| 2 | L_M2 | 0.18 µm (?) | **1.8 µm** | M2 current 10× too high |
| 3 | Vth0 | ~0.85 (computed) | **0.54** | Sub-threshold off by 100-1000× |
| 4 | etab | various assumed (S5-C used +1.8) | **-0.086** | Body bias sensitivity wrong |
| 5 | beta0 | 19 | **18** | IIMOD slightly off |
| 6 | pdiode BV | 5 | **11** | Body breakdown wrong |
| 7 | Bf | 991-fitted | **10000** (canonical) | NPN gain 10× off |
| 8 | snapback formula | BSIM4 §6.1 IIMOD only | **Ipos = Iexp + Ipow** | Missing entire mechanism |

## S10 plan (after S9 completes)

1. Hard-set ALL 7 canonical params above (no BBO)
2. Implement Ipos formula (digitise PWL coefficients from slide 12.26)
3. Use Sebas's `2Tcell_BSIM_param_DC.csv` to override per-(VG1, VG2) ETAB/K1/NFACTOR/mbjt
4. Run 33-curve fit (no parameter optimization, pure physics evaluation)
5. Compare to measured data

Expected: cell-wide < 0.5 dec if canonical values are right and Ipos formula correct.
