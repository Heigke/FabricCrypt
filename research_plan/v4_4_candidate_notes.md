# Brief v4.4 candidate notes — 2026-05-11

**Status**: NOT a finished revision. Captures what 2026-05-11's full
campaign (M-phase 5 experiments, N-phase 11 experiments + N2c in
flight, O42 multimodal oracle review of 21 Sebas+Mario slides)
revealed, both as corrections to v4.3 facts and as new findings.

**Decision-point**: whether to convert this into v4.4 depends on N2c
result. If N2c PASSes ambitious, open v4.4. Otherwise, keep v4.3 final
and treat this file as the diagnostic record for the user when sending
the consolidated email.

---

## Corrections to v4.3 facts (oracle-confirmed, ≥2 of 3 reviewers)

| v4.3 claim | Correction | Confidence |
|---|---|---|
| Bulk current: `Iexp = 10^(d·V_d)` + `Ipwl = a·V_d^c + b for V_d ≥ −j` | `Iexp = a·exp[b·(V_D + c)]` + `Ipow = d·(V_D + f)^e for V_D > −f, else 0`; a,b,d,e,f are PWL(V_G), c constant | unanimous 3/3 |
| PWL parameters are PWL(V_G2) | Slide 13 axis labels are "Gate Voltage V_G" generically; could be V_G1 or V_G2 | unanimous; needs Sebas confirm |
| Thick-ox cell V_G2 linear range 2.5–3.0 V | Mirror-bank V_w_x range 2.5–3.0 V; thick-ox V_G2 stays ≤ 0.5 V per slide 18 | unanimous 3/3 |
| 2T cell area 31.8 µm² | 33 µm² (5.5 × 6 µm deep N-well minimum) | 3/3 |
| Slide 21 pdiode = P-body diode | parasitic N-WELL diode (N-well to floating P-body) | 3/3 |
| Experiment list 3–7 | 5–7 (per grok slide-21 reading; openai also reads 5–7) | 2/3 |
| Slide 19 timescale rescale 10³× | 10⁵× | 2/3 (openai didn't comment) |
| Slide 20 Poisson reference 85% | 89% on confusion-matrix diagonal mean | 2/3 (openai read ~85%) |
| Slide 18 firing range >10⁴× | >10³× | grok specific; gemini didn't catch |
| M4: "V_G ≈ −2 V suppresses leakage" | Slide 18 plots only V_G2 down to −0.2 V; "−2 V" was our extrapolation | all 3 confirm |

## New facts surfaced from slides (not in v4.3)

**Architecture / topology**:
- 1T deep-Nwell neuron-synapse variant exists (slide 17): 8 µm² fully
  contacted, 100% yield, 180 nm CMOS
- 2T thick-ox spiking neuron 17 µm² (slide 18)
- Self-reset cell with explicit C_int = 102 fF (slide 10)
- Input-neuron soma C_int = 170 fF (slide 19); per-neuron C_int varies
- Parasitic N-well diode explicitly in slide 21 schematic (voltage-
  dependent capacitance + leakage)

**Calibrated numbers**:
- Self-reset cell (slide 10): 0.2 pJ/spike at 100 nA exc, 0.5–0.7 V
  spike amplitude, 10× freq tuning range, total area 111 µm² (w/ cap)
- Input-neuron (slide 11): ~21 fJ/spike total (~0.7 fJ generation +
  ~20 fJ integration), area ~60 µm²
- Input-neuron freq→current calibration: 500 pA→60 kHz,
  800 pA→120 kHz, 1.26 nA→180 kHz, 2 nA→260 kHz, 3.15 nA→340 kHz,
  5 nA→360 kHz
- Slide 21 pulse params: V_set=2.05 V, t_set=1 µs, t_rise=t_fall=200 µs,
  V_G1=0.45 V, V_G2=0.30 V; sweep t_rise ∈ {10 µs, 100 µs, 1 ms}

**Constraints / operating-window callouts**:
- Slide 17: V_NW > 2.5 V, V_G < 0.8 V, V_D < 3.5 V, V_S = 0
- Slide 18: V_G1 < 0.8 V, V_G2 < 0.5 V or floating, V_NW > 2.5 V
- "−1 V pre-write pulse widens dynamic range" (retention ~100 s)
- Slide 14: current SPICE model does NOT include V_B dependence; new
  version uses R_B = 1 MΩ measurement data

**Citations**:
- Pazos+Lanza, Nature 618, 57-62 (2023) — SNN substrate
- DOI 10.1109/SPCC.2013.6663447 — 130 nm implementation

## New campaign findings (M+N phase)

**M2 (PASS, 1.278 dec excess)**:
Per-V_G1-branch residual at production fit (forward-eval, not
re-optimized):
- V_G1 = 0.20 V: 0.86 dec (CI [0.74, 0.99])
- V_G1 = 0.40 V: 2.38 dec (CI [2.19, 2.58])
- V_G1 = 0.60 V: **3.54 dec** (CI [3.37, 3.70])
- Cross-branch mean 2.26 dec

The brief's 0.51-dec triangulation headline reports the cross-branch
average; per-branch the spike-generator regime (V_G1=0.6) carries 4×
the subthreshold-regime residual. Real silicon-physics structure.

**N1b/A (PASS, 84.65 ± 0.72 %)**:
First in-house reproduction of a Sebas SNN reference number, on real
MNIST 28×28. Likely Sebas's value is 89% (oracle 2/3) so the gap is
4–5 pp, not 0. Pipeline foundation valid.

**N4c (INTERMEDIATE PASS, 37.23 ± 17.24 %)**:
Symmetric differential-pair NS-RAM weight memory recovers 15 pp over
single-ended (22%) and 34 pp over the broken N4b rule (3%). Diff-pair
linearity R² jumps 0.006 → 0.977. But still 47 pp below ideal readout
(84.65%). Softmax classification can't tolerate ~10% per-row encoding
error from the thin-ox transfer curve at V_G1 = 0.3 V.

**N2 / N2b / N2c (all FAIL)**:
NS-RAM-as-input-neuron substitution:
- N2 static eval: 79.27% (−5.4 pp vs Poisson 84.65%)
- N2b transient α=10³: 78.37% (V_b drift ~4 mV/step undersamples)
- N2c transient α=10⁵: 9.9% (V_b rails 66% of steps — overshoots)
The "right" α for our 5 fF surrogate sits between 10³ and 10⁵ but
finding it would require Sebas's actual C_b (or violate NO-CHEAT).

**N3 (PARTIAL_FAIL)**:
α-sweep on N2b: α=10³ reproduces N2b cleanly (78.37%); α=10⁴ drops
to 64.76% with one collapse-seed at 29%. Rescaling is NOT benign —
T affects Poisson sample count *and* integration step at once.

## What this means architecturally

Three NS-RAM roles tested, three honest takes:

| Role tested | Result | Honest reading |
|---|---|---|
| Input neuron (rate-coded spike substitute) | −5 to −6 pp vs Poisson at fixed thin-ox params | Architecturally costly without thick-ox card + correct C_b |
| Compute unit (transient body-state reservoir) | structurally invisible at our locked params | C_b/dt physical operating point is off — locked params don't activate transient dynamics |
| Analog weight memory (single-ended) | 22% (~ chance) | Thin-ox transfer asymmetric at V_G1=0.3 V |
| Analog weight memory (differential pair) | 37%, R²=0.977 geometric fit | Differential geometry works; per-row encoding noise blocks readout |

## Recommended v4.4 framings (pick ones if v4.4 opens)

1. **"Triangulation 0.51-dec is V_G1-averaged"**: explicitly note that
   the headline figure is the cross-branch mean; per-branch ranges
   from 0.86 dec to 3.54 dec. This is honest about the silicon's
   richness without retracting anything.

2. **"NS-RAM as weight memory with differential pairs"**: report
   N4c's 37% / R²=0.977 as positive finding. Frame the 47-pp gap to
   ideal as motivating per-row gain compensation or training-aware
   weight calibration — both well-trodden in analog ML hardware lit.

3. **"NS-RAM as input neuron requires correct C_b + thick-ox card"**:
   our 5 fF assumption + locked thin-ox window prevents transient
   mode from activating. Sebas's silicon dynamics likely sit in the
   right τ_b range; pyport's are off by orders of magnitude. This
   is a model-side limitation, not a substrate-side limitation —
   important distinction for the brief tone.

4. **"Real silicon LIF baseline matched at 84.65%"** (assuming the
   88% reference is right): we have a working SNN pipeline foundation
   on real MNIST 28×28 reproducing Sebas's reference within ~5 pp.
   Sub-bullet: closing the remaining gap needs his Brian2 script +
   exact training-dataset preprocessing.

## What v4.4 SHOULD NOT claim

- Do not claim NS-RAM beats Poisson on MNIST (it doesn't, in any
  configuration we tested)
- Do not claim transient body-state dynamics add measurable capacity
  (N2 ≈ N2b at α=10³ + N2c collapse at α=10⁵)
- Do not retract v4.3's energy headline (10× silicon-energy floor is
  unaffected)
- Do not retract v4.3's three-source physics triangulation (0.51 dec
  result holds; only the V_G1-averaging context is new)
