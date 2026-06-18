# Gap Analysis — NS-RAM vs Real-Silicon Competitors

For each of the 9 PASS sims (4 AMBITIOUS + 5 DISCOVERY), what's the win,
what's the loss, what's the headline differentiator if any?

**All NS-RAM numbers are SIMULATED at 130 nm.** Competitor numbers are
real silicon, typically at 7–28 nm. Read every comparison with that
node penalty in mind.

---

## A. AMBITIOUS workloads

### A1. Reservoir Mackey-Glass / NARMA-10 (1024 neurons)

- **NS-RAM [SIM]:** 1024 NS-RAM cells, claim "reservoir-class temporal task."
- **Best real Si:** Loihi 2 (IF-based reservoir), BrainScaleS-2 (surrogate
  gradients), photonic reservoirs (sub-ns).
- **Win?** No clear win. The 2026-04 log explicitly flags that a
  standalone NS-RAM cell is a "memoryless analog weight, not a reservoir
  node." Reservoir behavior requires external recurrence.
- **Loss?** No published Loihi NARMA pJ/step to compare to, so we cannot
  even claim a win quantitatively.
- **Headline differentiator:** **NONE that survives scrutiny.** Demote
  this from AMBITIOUS to "feasibility demo" until external recurrence is
  added and an apples-to-apples reservoir benchmark is run against
  Loihi 2 (NeuroBench).

### A2. Stochastic RNG (NIST 5/5, 0.4 pJ/bit, 1.27 Mbit/s)

- **NS-RAM [SIM]:** 0.4 pJ/bit, 1.27 Mbit/s, NIST 5/5.
- **Best real Si:** Cheng 2024 (65 nm CMOS): 0.244 pJ/bit @ 1 Mb/s,
  NIST-passing, no post-processing. Synopsys TRNG IP, RHS-TRNG @ 2.5 pJ/bit.
- **Win?** **Marginal, possibly negative once node-scaled.** Our
  0.4 pJ/bit at 130 nm vs 0.244 pJ/bit at 65 nm — scaling 130→65 gives
  another 2–4× headroom for CMOS; node-adjusted CMOS likely wins.
- **Loss?** Probable, after node scaling.
- **Headline differentiator:** **Same-cell reuse** — the same NS-RAM
  array doing MNIST inference can switch to TRNG mode, which a
  dedicated CMOS TRNG cannot do. This is a *multi-function* story,
  not a *pJ/bit* story.

### A3. LMS-Eq QPSK (2.76 pJ/symbol)

- **NS-RAM [SIM]:** 2.76 pJ/symbol QPSK adaptive equalizer.
- **Best real Si:** No directly comparable published neuromorphic LMS-Eq
  number found. Conventional ASIC equalizers report fJ-to-pJ-per-tap;
  per-symbol depends on tap count.
- **Win?** Indeterminate — no published competitor for like-for-like.
- **Loss?** Cannot rule out.
- **Headline differentiator:** **In-memory adaptive coefficient
  update** if the floating-body bias *is* the LMS coefficient. This is
  the most architecturally novel of the 4 ambitious claims and the most
  defensible if peripheral overhead is honestly accounted.
  **BOLD CLAIM — needs ngspice with peripheral DAC/ADC modelling.**

### A4. Hierarchical MNIST SNN (97.15%, 17.7 pJ/inference)

- **NS-RAM [SIM]:** 97.15% @ 17.7 pJ/inf.
- **Best real Si head-to-head:**
  - BrainScaleS-2 (65 nm): 97.6–98.0% @ **8.4 µJ/inf** (8400 pJ).
  - Loihi 2: ~98% @ ~25 pJ/SOP — per-inf in low-µJ range.
  - NorthPole (12 nm): not MNIST canonical, but **612 fr/J** on
    ResNet-50, i.e. ~1.6 mJ/inf for 25M-param model.
- **Win?** On accuracy: parity (97.15% vs 97.6–98.0% — within noise).
  On energy: NS-RAM's 17.7 pJ would be **~470× better than
  BrainScaleS-2** if true. **But this is at 130 nm, sim-only, and
  excludes peripherals.**
- **Loss?** Once peripheral DAC/ADC, ROW/COL driver, and wire energy
  are added, the gap shrinks dramatically. BrainScaleS-2's 8.4 µJ is
  *the entire chip + readout* number; ours is *device-only*.
- **Headline differentiator:** **Apparent 100–1000× device-level energy
  advantage**, conditional on peripheral overhead matching. **MUST be
  ngspice-validated including DAC/ADC ladder.**

---

## B. DISCOVERY workloads

### B1. HDC UCI-HAR (8192 bits, ~84%)

- **NS-RAM [SIM]:** 8192-bit HVs, ~84%.
- **Best real Si:** Software HDC (Yan 2023): 94.2% at D=64.
  LogHD ASIC (28 nm): 4× energy efficiency vs sparse-HDC baseline (no
  absolute pJ/inf).
- **Win?** No — accuracy 10 pp below state-of-the-art software HDC.
- **Loss?** Yes on accuracy.
- **Headline differentiator:** **None.** Demote.

### B2. Predictive coding NAB (D=256)

- **NS-RAM [SIM]:** D=256 predictive-coding anomaly detection.
- **Best real Si:** No NAB-canonical neuromorphic chip number.
- **Headline:** Architecturally novel but no head-to-head opponent;
  defer to NeuroBench v2 alignment.

### B3. Memory Palace assoc-recall (D=512)

- **NS-RAM [SIM]:** D=512 associative memory.
- **Best real Si:** No published competitor for the exact task.
- **Headline:** Same as B2.

### B4. Cascade KWS+ECG (60.8% energy saving)

- **NS-RAM [SIM]:** 60.8% energy saving vs always-on baseline by
  cascading KWS gate → ECG.
- **Best real Si:** Cortex-M4 KWS at ~1–100 µJ/inf; Akida AKD1000
  full-system 1.4 mJ KWS; Loihi 2 Eventprop sub-mJ.
- **Win?** **60.8% is an algorithmic/system gain**, not a chip gain.
  Cascading wins are reproducible on Cortex-M with same hierarchy.
- **Loss?** Not a chip story per se.
- **Headline differentiator:** **System-level cascading + analog
  always-on KWS gate** could be a real win at 130 nm if the gate cell
  consumes < 10 µW. Currently ngspice-unvalidated.

### B5. HDC DVS-Gesture (8192 bits, 59.3%)

- **NS-RAM [SIM]:** 59.3% on IBM DVS-Gesture (11-class).
- **Best real Si:** Loihi 2: **89.6–96%**. IBM TrueNorth: prior art at
  comparable accuracy. Smart-camera SNN: 96.7% @ 241 mW.
- **Win?** No — 30+ pp accuracy gap.
- **Loss?** Severe accuracy loss. HDC-on-NS-RAM is not competitive on
  DVS-Gesture.
- **Headline differentiator:** **None.** The win story isn't here.

---

## Cross-cutting headline

The honest pitch that survives this comparison matrix is:

> **"Same-cell multi-function analog primitive in 130 nm:
> inference + TRNG + LMS adaptation + cascade-gate, all in the
> sub-pJ-per-op regime at the device level. Trade accuracy on hard
> vision tasks for energy and multifunctionality on edge sensor fusion."**

The pitch that does **NOT** survive:

- "Reservoir computer" (memoryless cell, no head-to-head)
- "Beats Loihi on DVS-Gesture" (30 pp accuracy gap)
- "Beats BrainScaleS on MNIST energy" (peripheral overhead unmodelled,
  node disadvantage unaccounted)
- "Beats CMOS TRNG" (node-adjusted, probably not)

---

## BOLD claims requiring ngspice-level validation (priority order)

1. **MNIST 17.7 pJ/inf** — include full DAC/ADC + wire energy, redo at
   130 nm SPICE. Expected outcome: 100–1000× degradation from
   device-only number.
2. **LMS-Eq 2.76 pJ/symbol** — same; verify the floating-body-as-
   coefficient claim is physically maintained under realistic
   peripheral load.
3. **Reservoir 1024-neuron benchmark** — define recurrence topology
   (external CMOS or in-array) and re-run with measured τ-spread;
   compare to Loihi 2 NeuroBench reservoir suite.
4. **TRNG 0.4 pJ/bit** — node-scale to 65 nm reference; if NS-RAM
   wins at iso-node, claim is defensible. Otherwise re-frame as
   "free-as-a-side-effect TRNG" not "best-in-class TRNG."
