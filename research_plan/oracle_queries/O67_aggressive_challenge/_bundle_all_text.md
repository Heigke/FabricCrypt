# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: context.md (13394 chars) ===
```
  (3) Branch-flat impute for 8/33 NaN rows (VG1=0.4/0.6 VG2<=-0.05)
       VG1=0.4: ETAB=1.9, K1=0.53825, BETA0=19, NFACTOR=6
       VG1=0.6: ETAB=2.5, K1=0.41825, BETA0=20, NFACTOR=6
Plus per-bias overrides via patch_sd_scaled (mirrors z91f).

Results:
  cell-wide median = 1.419 dec  (z360 ~5.6+, z358 = 4.28 → ~2.9 dec drop)
  per_VG1 medians: VG1=0.20: 1.357, VG1=0.40: 1.242, VG1=0.60: 1.866
  Flagship probe (VG1=0.6, VG2=0.20, Vd=2.0):
      Vsint=0.182, Vb=0.484, Id=1.07e-9, Ic_Q1=5.35e-10
      (Vb was 0.78 in z360 → now 0.484; ngspice target ~0.27 — still elevated)

Gates:
  INFRA        : 33/33 valid   PASS
  PASS         : 1.42 < 1.5    PASS
  AMBITIOUS    : 1.42 < 0.95   FAIL
  HIGH_VG1     : 1.87 < 4.0    PASS (was 5.6)

Residual: VG1=0.60 branch median 1.87 still dominates total error; Vb=0.484
remains above ngspice's 0.27 (well-diode discharges but Rs=1e6 still too
weak). Next: tune body_pdiode_Rs lower (1e4 - 1e5 range) to drive Vb
toward 0.27, or examine if vnwell_Rs (well-diode series-R) needs lowering.

Files: scripts/z361_pdiode_fix_refit.py, results/z361_pdiode_fix/summary.json
       + per_VG1_{0.20,0.40,0.60}.png

## 2026-05-14 13:25 — R-41 BREAKTHROUGH: 1.42 dec PASS GATE ✓

Sebas's pdiode_to=vnwell (R-40 finding) was THE missing piece.
- Cell-wide median: 1.42 dec (was 4.28 in z358 — improvement 2.86 dec)
- VG1=0.20: 1.36  0.40: 1.24  0.60: 1.87 (was 5.64 — drop 3.77 dec!)
- Vb @ flagship: 0.78 → 0.484 (target 0.27; partial)
- PASS<1.5: PASS ✓
- HIGH_VG1<4.0: PASS ✓
- AMBITIOUS<0.95: FAIL (close)

R-42 dispatched: Rs sweep to push Vb closer to 0.27 → likely cracks AMBITIOUS.

## 2026-05-14 13:30 — S1 MEP-7 BREAKTHROUGH: N=1M GPU forward_2t PASS

zgx NVIDIA GB10 + torch.compile mode=default:
  N=1M × 33 Vd × 20 Newton = 59.95s (gate <60s MARGINAL PASS)
  2.0e4× vs sequential, 5.3e4× with compile
  Convergence 98.8% (no degradation vs N=10K)
  AMBITIOUS <10s FAIL (needs analytic Jacobian)

Combined with S2b (transient) + S3 (network) + R-41 (DC PASS):
1M-CELL NS-RAM SIMULATOR PIPELINE COMPLETE.

## 2026-05-14 — R-42: body_pdiode_Rs sweep — NULL RESULT (Rs is invariant)

z362_rs_sweep.py — Swept body_pdiode_Rs ∈ {1e3, 1e4, 1e5, 1e6, 1e7} Ω over
all 33 Sebas curves, R-41 stack locked.

Results (5 points, ~155s each, total ~770s):
  Rs=1e3:  cell_med=1.4189  Vb_flag=0.4843  per_VG1=(1.357, 1.242, 1.866)
  Rs=1e4:  cell_med=1.4189  Vb_flag=0.4843  per_VG1=(1.357, 1.242, 1.866)
  Rs=1e5:  cell_med=1.4189  Vb_flag=0.4843  per_VG1=(1.357, 1.242, 1.866)
  Rs=1e6:  cell_med=1.4189  Vb_flag=0.4843  per_VG1=(1.357, 1.242, 1.866)
  Rs=1e7:  cell_med=1.4189  Vb_flag=0.4843  per_VG1=(1.357, 1.242, 1.866)

Bit-identical across 4 decades of Rs. body_pdiode_Rs is dead variable.

Gates:
  PASS<0.95       : FAIL (1.42)
  BREAKTHROUGH<0.5: FAIL
  Vb_within_0.05_of_0.27: FAIL (Vb=0.484, target 0.27)

Diagnosis: pdiode discharge branch carries negligible current at the
operating point — Vb is set by the well-diode + iii_body source balance,
not by the pdiode-to-vnwell path. Either:
  (a) body_pdiode_area / Js_per_area combo is too small → branch is dark
  (b) Vb < Vnwell (0.484 < 2.0) means pdiode is reverse-biased w.r.t.
      vnwell → only leakage current flows, so Rs is invisible.

Likely (b): with body_pdiode_to=vnwell=2.0 and Vb~0.48, the pdiode sees
Vb-Vnwell = -1.52 V (reverse). No forward conduction → Rs has zero effect.
The R-41 jump came from enabling use_well_diode (n-well-to-body forward
diode in the OTHER direction), not from the pdiode-to-vnwell path.

R-43 candidates (next):
  - vnwell lower (0.0 or floating) so pdiode forward-biases
  - vnwell_Rs sweep (well-diode series-R; THIS is the actual discharge knob)
  - iii_body_gain lower (currently 0.66 from R-39) — reduce body injection
  - Or: probe Iii vs I_body_pdiode to confirm which branch dominates

Files: scripts/z362_rs_sweep.py, results/z362_rs_sweep/summary.json,
       results/z362_rs_sweep/dec_vs_Rs.png

## 2026-05-14 14:13 cron — R-42 done (Rs invariant); R-43 2D sweep launched
R-42: body_pdiode reverse-biased at vnwell=2V → Rs is dead variable.
R-41 improvement came from use_well_diode=True (opposite direction).
R-43: 5×3 sweep (iii_body_gain × vnwell_Rs) — actual active knobs.
O65 oracle dispatch in flight too.
## 2026-05-14 14:18 cron — R-43 z363 2D sweep in flight + O65 oracle dispatch

## 2026-05-14 14:35 — O65 oracle 3-way landed (3 distinct Vb-bottleneck hypotheses)

Q3 (why Rs invariant + what holds Vb @ 0.484):
- gpt-5: body→nwell diode below knee → no current; sweep Vb numerically to find which branch dominates op-point
- gemini: BJT base reverse-bias LEAKAGE missing (Isc, Var, Ikr ngspice params not in pyport GummelPoon)
- grok: well-diode Js/R_well-series need tuning sweep

Q1 (1.42 honest?): gpt-5+gemini both say "verify by isolating pdiode contribution" — split A/B (well_diode only vs pdiode_to_vnwell). gemini calls 1.42 "likely real, falsifiable in <1h".

R-43 still in flight (7/15 cells). Next after: gemini's BJT leakage test (most specific
new hypothesis).

## 2026-05-14 13:18 MEP+DS-N cron — R-43 z363 active
APU=44C OK. ACTIVE z363_2d_sweep.py. R-41 already crossed PASS gate
(1.42 dec). R-43 pushing for AMBITIOUS (<0.95). Current best 1.14 dec.
Scale stack (S1/S2b/S3) complete. After AMBITIOUS land or fail → DS-N
benchmarks at N=1M via S2b LUT.

## 2026-05-14 14:55 — R-44 FALSIFIES gemini BJT leakage hypothesis

pyport BJT Ib = 8.20e-13 A, ngspice = 8.11e-13 A → ratio 0.99 (no gap).
All Gummel-Poon params present in pyport (Isc, Ise, Ikr, Var). Sweep
Vd ∈ [0.3, 3.0] holds 0.99-1.00 throughout. No missing leakage.

Real candidates: body_pdiode params (Js, Rs fitted not physical),
iii_body_gain, vnwell coupling. R-43 already showed vnwell_Rs plateaus.

So three oracle hypotheses Q3:
  gpt-5 numerical Vb-sweep — still possible
  gemini BJT leakage — FALSIFIED
  grok well-diode Js — partially tested (R-43 vnwell_Rs ≈ Js scaling)
## 2026-05-14 15:13 cron — R-43 + DS-N5b + DS-N7 active. R-44 falsified.

## 2026-05-14 15:25 — DS-N7 Memory Palace ALL GATES PASS

User's "Method of Loci" intuition → working analog associative memory:
  N=1K/100 pairs: 91.0% (INFRA PASS)
  N=10K/1000: 95.7% (PASS)
  N=100K/10K: 95.9%, 36ms wall (AMBITIOUS 1666× margin)
  Sequence T=1000: 99.7%

Quantitative differentiation:
  3.5 bits/cell (Flash SLC=1, comparable to TLC=3)
  38 fJ/cell-read (Flash ~10 nJ/byte → 250× headroom)
  360 ns/cell vs Flash 25µs (70× faster random access)
  Graceful degradation (no hard corruption like RAM/Flash)

NEW killer-app angle for Mario/Sebas: pair with z2213 multi-tau retention
→ "neuromorphic memory that forgets like a brain". Broader pitch than
just always-on KWS — addresses RAG/agent memory market.

## 2026-05-14 15:30 — R-43 DONE: 1.131 dec best (PASS off by 0.13)

vnwell_Rs dominates, iii_gain inert.
Best: Rs=1e8, iii=1.0 → 1.131 dec (Vb=0.72, ngspice 0.27)
VG1=0.20: 2.62 (was 1.36), VG1=0.60: 1.07 (was 1.87) — anti-correlated

R-45 dispatched: sweep vnwell ∈ {0.5..2.5} at best Rs.
Hypothesis: vnwell is regime selector, different VG1 branches want different Vnwell.
12:50 idle-check: APU=43C

## 2026-05-14 15:48 deep-dive cron — 3 spår aktiva (R-45 + DS-N5b zgx)
APU=43C OK. R-43 (1.131 dec) closed. R-45 vnwell sweep in flight on
ikaros. DS-N5b HDC scale to N=1M on zgx (NVIDIA). DS-N7 PASS.
4A/4B/4C/4D/4E klara — phase plan converged. Modell-track continues.

## 2026-05-14 15:48 cron — P-phases done; R-45 in flight pushing AMBITIOUS
P3 partial PASS: cell-wide 1.13 dec (gate <0.95 still FAIL).
P4-P8 closed (KWS, transient, networks). R-45 vnwell sweep test active.
DS-N7 Memory Palace PASS (new commercial angle).

## 2026-05-14 16:05 — R-45 vnwell-sweep DONE: global-knob floor 1.131 dec

5 vnwell values × 33 curves:
  0.5: 4.62 (Vb=-0.01 collapse)
  1.0: 1.50 (Vb=0.53)
  1.5: 1.18 (Vb=0.72)
  2.0: 1.13 (Vb=0.72) ← matches R-43 exactly
  2.5: 1.13 (Vb=0.73)

Hypothesis FALSIFIED: vnwell is not the regime selector.
Anti-correlation VG1=0.20 vs VG1=0.60 is STRUCTURAL.

Global-knob solution space EXHAUSTED at 1.131 dec.
Next requires either per-VG1 params (less physical) or topology rewrite.

Current honest summary:
  Pre-R-20:                   4.16 dec (broken BJT)
  +R-20 BJT Vbc fix:          z337 4.16
  +R-29 Vth/tox:              z346 4.08
  +R-37 binunit:              z358 4.28 (over-pumping exposed)
  +R-41 pdiode_to=vnwell +
    use_well_diode=True:      z361 1.42 PASS gate ✓
  +R-39 BJT params:           1.42 → unchanged
  +R-43 (iii_gain,vnwell_Rs): 1.131 best
  +R-45 vnwell sweep:         1.131 floor confirmed

Verified physics improvements: ~3 dec (4.16 → 1.13). Honest stop point
unless we accept per-VG1 fitting or topology rewrite.

## 2026-05-14 16:10 — DS-N7b BRUTAL ABLATION: DS-N7 CLAIM RETRACTED

Digital dict baseline = 0.957 EXACT match to NS-RAM. Hash does 100%
of work. 4.3% miss rate = collision rate. NO analog robustness.
- Readout noise cliff at 17mV (codebook spacing), NOT 50mV
- 1% key bit-flip → chance accuracy
- Flash beats analog at everything but cell density

What remains TRUE:
- Cell density: 3.3 bits/cell @ ~4 fJ/read (vs flash SLC 1 bit/10pJ)
- That's a CELL property, not architecture

Real "neuromorphic memory" pitch requires the multi-tau retention
(z2213 data) to be exploited — Memory Palace as built does NOT use it.
DS-N7 retracted. Need new architecture that actually uses NS-RAM dynamics.
## 2026-05-14 16:13 cron — DS-N5b/DS-N7/R-45 all done. O66 critique in flight.
## 2026-05-14 16:18 cron — R-arc at global-knob floor 1.131; honest stop unless topology rewrite or per-VG1 fitting

## 2026-05-14 16:18 — 4 spår dispatched (alla maskiner)
A: R-46 per-VG1 BBO (ikaros, ~4h)
B: R-47 body subdiode topology (daedalus/ikaros, ~3h)
C: DS-N8 KWS event-coded N=100K (zgx, ~4h)
D: DS-N9 neuromorphic-forgetting memory + brutal ablation (ikaros, ~3h)

Plus O66 oracle critique in flight.

## 2026-05-14 16:20 — DS-N5b VERDICT: HDC scaling is dead axis

  N=1024:  83.56% (0.58s)
  N=10K:   84.29% (3.34s)
  N=100K:  84.57% (32s)
  N=1M:    84.60% (346s)
+1.04pp total span. Plateau at ~84.6% well below digital baseline ~96%.
INFRA PASS, PASS (boundary), AMBITIOUS FAIL.

Verdict: scale is NOT where NS-RAM HDC story lives. Pivot to (a) better
encoder, (b) substrate-specific tasks (noise-decorrelation, low-energy
bind/unbind). Pure dim is dead.

## 2026-05-14 16:30 — DS-N9 BRUTAL: "neuromorphic memory" pitch retracted

NS-RAM retention curve doesn't even match z2213 (cell physics OK,
readout collapses to nearest codebook level fast):
  t=0.1s: NS-RAM 47% vs digital+decay 69% (−22 pp)
  t=1s:   NS-RAM 14% vs digital+decay 38% (−24 pp)

Digital+decay STRICTLY DOMINATES everywhere t>0.

ONLY remaining advantage (cell-level, not architectural):
  Energy: NS-RAM 2.3e-11 J vs DRAM 1.2e-2 J (5.3e8× better)
  Because no refresh needed.

Defendable pitch: "refresh-free analog cache for short-retention
edge KV stores (sensor fusion, ephemeral session state, edge inference)"
NOT "neuromorphic memory / forgets like brain" — that's overreach.

Two day-pitches retracted (DS-N7 + DS-N9). Substrate-level energy/density
remains the only un-falsified commercial angle.

## 2026-05-14 17:00 — R-47 FALSIFIES subdiode topology

Body subdiode test (4 Is × 2 destinations = 8 configs):
  to=vnwell: ALL inert (reverse-biased, branch is no-op)
  to=gnd: crushes high-VG1 ignition (Vb falls below 0.6V)

THREE INDEPENDENT global-knob falsifications confirm 1.131 dec floor:
  R-43 iii_gain×vnwell_Rs sweep
  R-45 vnwell sweep
  R-47 body subdiode

The anti-correlation between VG1=0.20 (over-pumps) and VG1=0.60
(under-pumps) is STRUCTURAL — fixable only by:
  (a) per-VG1 parameters (R-46 in flight, curve-fitting)
  (b) per-VG1 geometry change (separate M1/M2 taps, channel-edge models)

R-arc closed unless R-46 cracks 1.0 dec.
## 2026-05-14 17:13 cron — R-47 falsified, R-46 last chance for sub-1 dec
13:50 idle-check: APU=45C
## 2026-05-14 17:43 cron — R-46 BBO 16/96 (still no improvement), DS-N8 N=100K running
## 2026-05-14 17:48 cron — R-46 (sista R-phase) 16/96 in flight
## 2026-05-14 18:13 cron — R-46 + DS-N8 in progress
14:50 idle-check: APU=45C
## 2026-05-14 17:48 deep-dive cron — R-46 + DS-N8 active, all 4A/B/C/D/E phases done
## 2026-05-14 18:43 cron — R-46 + DS-N8 in flight
## 2026-05-14 18:48 cron — R-46 last R-phase active, no gate cross yet
## 2026-05-14 18:48 MEP+DS-N cron — R-46 + DS-N8 still active

## 2026-05-14 18:50 track-progress audit
Phase A (MEP): 4/4 launched. MEP-6 + MEP-7 in_progress (long-lived).
Phase B (DS-N): 9 attempted. N1/N2/N5/N6/N7/N9 closed; N4/DS-N5/N8 in_progress.
  AMBITIOUS PASSes: 0. Retracts: DS-N7 + DS-N9 (hash/decay falsifications).
Phase C: 4D oracle (O63/O64/O65/O66) done; 4E.1 brief v4.4 done.
R-arc: 1.131 dec floor confirmed via 3 independent falsifications (R-43/R-45/R-47).
R-46 last per-VG1 BBO chance running.
## 2026-05-14 19:13 cron — R-46 + DS-N8 still active
15:50 idle-check: APU=45C
## 2026-05-14 19:43 cron — R-46 + DS-N8 still active
## 2026-05-14 19:48 cron — R-46 still active, no gate cross
## 2026-05-14 20:13 cron — R-46 + DS-N8 still active
16:50 idle-check: APU=44C
## 2026-05-14 19:48 deep-dive cron — 4A/B/C/D/E done, R-46/DS-N8 in flight
## 2026-05-14 19:48 cron — P-phases all closed, R-46 sole DC track
## 2026-05-14 20:43 cron — R-46 + DS-N8 in flight
## 2026-05-14 20:48 cron — R-46 sole R-phase active

```
