# O73 — NS-RAM V_SINT_PIN Gap-Closing Critique

You are reviewing a 12h progress update on an NS-RAM compact-model fit to Sebas's 130nm measured I-V data (3 VG1 branches × multiple VG2 × V_D sweep). The DC fit metric is RMSE in decades of log10(|I_D|), cell-wide aggregated over VG1∈{0.2, 0.4, 0.6}.

## Prior state (O68 critique)
- z427 ALL_FLAGS_ON (joint V_B/V_Sint Newton): 3.899 dec cell-wide; the "1.733 dec" headline number was flagged as misleading because it excluded NaN-failing branches. H1=1MΩ substrate pulldown was flagged as a magic number.

## Two follow-up experiments

### z429 (S19) point-by-point multi-solver debug at VG1=0.6, VG2=0.0
| Solver | Convergence | RMSE (dec) |
|---|---|---|
| Newton (cold start) | 17% | 4.25 |
| Damped Newton | 0% | — |
| Arc-length warm-start | 87% | 5.62 (locks bad root) |
| scipy fsolve (5 ICs) | 65% | — |
| Brute-force grid | — | 8 distinct roots at many V_D |
| **V_Sint=0 hard PIN (1D Newton on V_B only)** | **100%** | **1.26** |
| Backward sweep | 10% | (Newton hysteresis 2.10V vs forward) |

Verdict: `physics_via_vsint` — floating V_Sint runs to non-physical values; pinning V_Sint=0 cuts ~3 decades.

### z430 (S20) V_SINT_PIN cell-wide on all 25 valid biases
| Variant | RMSE (dec) | Notes |
|---|---|---|
| BASELINE (z427 joint Newton) | 3.899 | reference |
| M2_RS_100 (soft 100Ω substrate pulldown) | 1.975 | 11/25 fails at VG1=0.6 |
| **V_SINT_PIN (hard 1D Newton)** | **1.619** | 100% conv, 31s, 47× faster |

Per-branch with V_SINT_PIN: VG1=0.2 → 2.625 dec, VG1=0.4 → 0.786 dec, VG1=0.6 → 1.086 dec.
DISCOVERY (<2.0 dec) PASS. AMBITIOUS (<1.0) blocked by VG1=0.2 branch.

`ROOT_CAUSE.md` (z429): V_Sint floating is the dominant pathology. Soft Rs ≠ hard pin (soft 100Ω fails 11/25, hard pin succeeds 25/25). The remaining 1.6 dec residual is genuine model gap, likely VG1=0.2 sub-threshold.

## Three questions

### Q1 — Gate-crossing / over-clamping risk
Is `V_SINT_PIN` legitimate physics or does it hide model deficiencies? Real silicon has finite substrate contact resistance (μΩ–mΩ), not literally 0. Does hard-pinning `V_Sint=0`
over-clamp the system non-physically in a way that just happens to fit Sebas's curves? What would distinguish "the BJT really is well-substrate-grounded in this layout" from "we silenced a missing-physics symptom"?

### Q2 — Cherry-pick risk
The 1.619 dec cell-wide is dominated by VG1=0.4 (0.786) and VG1=0.6 (1.086) branches; VG1=0.2 stays at 2.625. Dropping VG1=0.2 would yield 0.94 dec ("AMBITIOUS"). Legitimate regime stratification (low-VG1 is a different operating mode that the compact model isn't required to cover) or motivated reasoning to claim a milestone?

### Q3 — Highest-value next experiment to close VG1=0.2 honestly to <1.0 dec
Rank the following (or propose better):
- (a) Add M2 GIDL or weak-inversion at low VG1
- (b) Add temperature self-heating
- (c) Fit a per-VG1 NFACTOR (scan three values)
- (d) Switch from BSIM3-derived to BSIM4 v4.8.3 GIDL §6.2
- (e) Something else

Be concrete: which device-physics mechanism is most likely activated at VG1=0.2 in a 130nm bulk NMOS that the current model misses, and what single measurement-side or model-side change would best test it?

## Output format
Reply with three labeled sections: **Q1**, **Q2**, **Q3**. Be terse, technical, and direct. Call out methodological problems plainly. ≤ 600 words total.
