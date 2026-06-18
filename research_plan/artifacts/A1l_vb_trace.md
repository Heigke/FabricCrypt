# A.1.l — Vb operating-point trace at three diagnostic biases

**Setup.** Post-A.1.j (emitter=GND BJT) + A.1.k (arclength solver). Vd=1.5 V,
Sebas per-bias overrides (P_M1: ETAB,K1,ALPHA0,BETA0; P_M2: NFACTOR;
BJT IS, area). Solver: `forward_2t_arclength_grad`. Components from
`_residuals` at converged (Vsint,Vb), then re-evaluated at hypothetical
Vb=0.7 (same Vsint) to test for a high-Vb basin. Script: `A1l_demo.py` ·
JSON: `A1l_vb_trace.json`.

## 1-3. Op-point + Id

| Bias (VG1,VG2)      | conv  | Vsint     | **Vb**       | Id_pred  | Id_meas | log-res |
|---------------------|-------|-----------|--------------|----------|---------|---------|
| WORST (0.6, 0.0)    | True  | +0.666 V  | **−0.253 V** | 5.7e-17  | 2.07e-5 | 11.6 dec|
| MOD   (0.4, 0.0)    | True  | +0.291 V  | **−0.255 V** | 3.0e-14  | 1.02e-6 |  7.5 dec|
| BEST  (0.6, 0.5)    | False | +0.023 V  | **−1.097 V** | 4.7e-10  | 9.64e-7 |  3.3 dec|

(11.6 dec at Vd=1.5 V exceeds the 5.73-dec curve median — this is the
worst point.) **Vb is negative at every bias — far below NPN turn-on
(~0.7 V) and even sub-threshold (~0.34 V). The parasitic NPN is in deep
cut-off (Vbe<0).**

## Component breakdown at converged Vb (A)

All currents are numerical-zero scale; KCL holds trivially (R_B≈1e-13).
No mechanism delivers meaningful body charge.

| Component | WORST     | MOD       | BEST      |
|-----------|-----------|-----------|-----------|
| Ids_M1    | 4.8e-19   | 3.0e-14   | 4.7e-10   |
| Ids_M2    | 1.2e-13   | 1.1e-13   | 2.1e-11   |
| Ic_Q1     | 5.0e-17   | 5.0e-17   | 5.0e-17   |
| Ib_Q1     | −5.0e-17  | −5.0e-17  | −5.0e-17  |
| Iii_M1    | 1.6e-27   | 1.4e-18   | 2.7e-13   |
| Iii_M2    | 2.3e-24   | 2.0e-27   | 9.0e-27   |
| Igidl/Igb | 0         | 0         | 0         |
| Ibs/Ibd   | ~−1e-17   | ~−1e-17   | ~−1e-17   |

## 4-5. Hypothetical Vb=0.7 (same Vsint, Vd, gates)

| At Vb=0.7   | WORST       | MOD         | BEST        |
|-------------|-------------|-------------|-------------|
| **R_B**     | **−7.8e-07**| **−7.8e-07**| **−1.6e-05**|
| R_Sint      | −4.6e-10    | +1.1e-10    | +1.9e-05    |
| Ic_Q1       | 2.86e-03    | 2.86e-03    | 2.86e-03    |
| Ib_Q1       | 2.84e-07    | 2.84e-07    | 2.84e-07    |
| Iii_M1      | 5.3e-25     | 3.5e-15     | 1.4e-09     |
| Ibs_M2 (out)| 4.99e-07    | 4.99e-07    | 4.99e-07    |

R_B is **negative at Vb=0.7 in all three cases**. With
`R_B = Σ currents INTO body`, R_B<0 means net current *leaves* the body
→ it would discharge. Pinning Vb=0.7 V demands ≥ ~8e-7 A (WORST/MOD) or
~1.6e-5 A (BEST). Iii — the only charger — peaks at 1.4e-9 A (BEST);
Igidl/Igb are zero. **Iii is 3-7 decades short of the BJT base + M2
body-source-diode load.** No high-Vb fixed point exists.

## Diagnosis

**The body settles at low/negative Vb because Iii is 3-7 decades smaller
than the NPN base current + M2 source-body-diode load at Vb=0.7; R_B at
the hypothetical Vb=0.7 is *negative* at every bias, so the high-Vb root
is not a basin of attraction — it does not exist as a stable fixed point
of the present residual system.**

The 5.73-dec under-prediction is a *model-content* gap, not a solver gap:
with emitter=GND and only BSIM4 Iii charging the body, the NPN never
turns on. Fix paths: (a) much larger Iii (ALPHA0/BETA0 ~3-7 decades
light), or (b) an additional charging path the netlist omits — drain→base
avalanche/Zener, or the "complementary bipolar" PNP feeding Vb from VDD.
