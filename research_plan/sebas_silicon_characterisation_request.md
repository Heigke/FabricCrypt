# Sebas — silicon characterisation request (post-NRF)

**To:** Sebastian Pazos (KAUST). Cc: Mario Lanza.
**From:** Eric Bergvall. **Status:** DRAFT, send after NRF (post 2026-05-06).

> **Audited 2026-05-09**: this draft is still substantially valid.
> The two requested measurements (Bf extraction at saturation,
> τ via pulsed-Vd TLP) remain the highest-value silicon evidence
> we can ask for. Production BJT params (Bf=9000, Va=0.55, Is=1e-9)
> are unchanged.
>
> **Optional addition**: Mention that pyport vs ngspice at production
> BJT params is now cross-checked (z231: max 0.51 dec, marginal miss
> at M2-OFF leakage tail only). This sharpens the case for run 1 —
> the silicon Bf extraction directly resolves whether the M2-OFF
> tail is a model bug or a real silicon feature. If you want this
> in the email, it can sit between "Run 1" and "Run 2" as a one-line
> note: "z231 has narrowed pyport↔ngspice agreement to 0.51 dec at
> production BJT, with the only miss at VG2 = 0; run 1 closes that."

This is the M3b deliverable in the brief: two characterisation runs
on existing 130 nm silicon that convert our v4.2-final
"physically-defensible 0.654 dec at $B_f=9{\times}10^3,
V_a=0.55$~V, $I_s=10^{-9}$" into a silicon-grounded fit. Neither
needs new fab — both are runs on a wafer you already have.

The brief asks NRF for tape-out funds based on the calibrated
parameter point; if the silicon $B_f$ falls in the predicted
$10^3$–$10^5$ range, that point is confirmed physical and the
M6/M9 plan stays on track.

---

## Run 1 — $I_c/I_b$ ratio at saturation (one bias)

**Why**: gives direct extraction of silicon $B_f$ for the parasitic
NPN. The pyport DC fit converges to $B_f=9{\times}10^3$; this number
is in literature-defensible territory for lateral parasitic NPN with
low-doping base, but is currently the most load-bearing parameter
in the whole calibration chain. One measurement closes that gap.

**What we need**:
- 2T cell, thin-oxide variant (M2 card geometry).
- $V_{G1}=0.6$ V, $V_{G2}=0.30$ V (saturated-spike regime — strongest
  parasitic-NPN drive).
- $V_D$ stepped past snapback fold ($V_D \in [1.4, 1.8]$ V, 50 mV
  step).
- Measure both $I_d$ and $I_b$ (body-tap current, if accessible
  through the body contact in your test chip).
- Equipment: standard parameter analyser (HP 4156 or similar).
- Expected duration: ~30 minutes per cell, including handling.

**What we extract**: $B_f \approx I_c/I_b$ at the saturation point,
with snapback-fold correction. Decision boundary:
- $B_f \in [10^3, 10^5]$ → optimum confirmed physical → tape-out
  proceeds at calibrated point.
- $B_f \ll 10^3$ → escalate to the M6 architecture-change deliverable
  (gpt-5's quasi-2D body model, $\sim 200$ LOC).
- $B_f \gg 10^5$ → unexpected; would re-open the 2D parameter
  calibration to verify $I_s$ co-extraction.

---

## Run 2 — Pulsed-$V_d$ / TLP at one bias (transient)

**Why**: extracts the body-region $R_b \cdot C_b$ time constant, which
sets the lateral-NPN equivalent circuit's bandwidth and disambiguates
$B_f$ from $I_s$ in the joint extraction. This pins down the
short-term-memory timescale that drives the reservoir-computing
demonstrations (Mackey–Glass forecast, NARMA-10, memory capacity).

**What we need**:
- Same 2T cell (M2 card).
- $V_{G1}=0.4$ V, $V_{G2}=0.0$ V (threshold-triggered integrator
  regime — the residual hot-spot in pyport's per-row diagnostic).
- $V_D$ pulse: rise time $\le 10$ ns, hold at $V_D \in [1.0, 1.5]$ V
  for a stepped duration $\Delta t \in \{100 \text{ ns}, 1\,\mu\text{s},
  10\,\mu\text{s}\}$.
- $V_D$ fall to $0$ V; record drain-current settling transient at
  100 MS/s for $\sim 100\,\mu$s.
- Equipment: TLP setup (Barth or similar) or Keithley pulse-IV.
- Expected duration: ~1 hour per cell with calibration.

**What we extract**:
- $\tau = R_b \cdot C_b$ from the post-pulse settling exponential.
- Slow-vs-fast tail separation indicates lateral-NPN charge-storage
  vs MOSFET channel transient.
- Decision boundary: $\tau \in [100 \text{ ns}, 10\,\mu\text{s}]$ is
  the design assumption; outside this range we revise the cell
  recommendation in §C.3.

---

## Practical asks

1. **Rough ETA**: even a "this fits in your queue in $X$ weeks"
   helps us tighten the M3b schedule. If equipment access is the
   bottleneck, we can consider a remote-measurement option through
   one of the Stockholm test labs.
2. **Which test chip**: please confirm whether the body-tap is
   accessible on the variant currently in your characterisation
   queue. If not, Run 1 collapses to an indirect $B_f$ extraction
   from saturation-current scaling.
3. **Data drop**: prefer raw IV-trace CSVs (one row per
   $(V_{G1}, V_{G2}, V_D)$ point) plus the pulse-IV transient as
   $(t, I_d, V_d)$ time series. We will share back the silicon-
   grounded fit within ~1 week of the data drop, with a residual
   figure analogous to `figures/per_row_residuals_optimum/`.

---

## Notes on scope

- Two runs only. We are not asking for a full characterisation
  campaign; the brief's tape-out recommendation rests on these
  two specific decisions ($B_f$ magnitude + $\tau$ range).
- Both runs are on your existing 2vHCa-2 device family (the same
  one that produced the 33-bias DC sweep we already calibrated
  against in `data/sebas_2026_04_22/`).
- If the body-tap is genuinely inaccessible, Run 1 has a fallback
  via $I_d(V_D)$ slope-ratio fitting in the snapback-saturation
  region — less precise but still informative.

---

*Drafted 2026-05-05 ~22:00 by Eric. Send after NRF deadline 2026-05-06
to avoid distracting Sebas during your own crunch period.*
