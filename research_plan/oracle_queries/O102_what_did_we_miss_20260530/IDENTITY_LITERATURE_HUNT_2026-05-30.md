# Identity Literature Hunt — 2026-05-30

**Question**: who has actually made computation *constitutively depend on* and *benefit from* a specific piece of silicon, on commodity (non-FPGA, non-memristor, non-photonic) hardware? Are we hunting a unicorn?

**Method**: 10-axis web search (WebSearch + WebFetch) + 4-way oracle dispatch (`O100_constitutive_lit_20260530`).

---

## Section 1 — Working examples in the literature

### 1.1 Where it WORKS (and why we can't port it directly)

| Paper | What they did | Transplant cost | Substrate | Portable to APU userspace? |
|---|---|---|---|---|
| **Joshi et al., Nat. Commun. 2020 (arxiv 1906.03138)** — PCM ResNet | Trained ResNet-32 on CIFAR-10 with noise injection; weights programmed onto IBM PCM crossbar. Each PCM cell's analog conductance is per-device unique. | They *designed against* transplant cost: ~0.5 % degradation. But the *underlying* device weights are individually programmed per chip — transplanting a raw-weight binary without re-programming is unusable (random output). | PCM crossbar | **No** — requires PCM hardware. |
| **Lammie et al. / "Variability-Aware Training" (arxiv 2111.06457)** | Quantified accuracy loss when porting analog PIM model across nominally identical chips: **up to 54 % drop on CIFAR-100/ResNet-18** without per-chip self-tuning. | 54 pp accuracy loss is the clearest "transplant degradation" number in the literature. | Analog PIM | **No** — requires analog PIM. |
| **Bandyopadhyay et al., Sci. Adv. 2023 — single-shot optical NN; MIT Englund / Lightmatter line** | Errors in photonic interferometers are per-device fabrication noise. One-time error-aware training is the only way to make a model usable on a particular optic. | Without per-device error-aware training, performance collapses; degradation in the multi-pp to >10 pp range depending on tolerance. | Photonic | **No** — requires Mach-Zehnder mesh. |
| **Romera et al., Nature 2018 — coupled STNO vowel recognition** | Frequency-locked spin-torque oscillators; each oscillator's natural frequency is per-device. Network "computes" through device-specific synchronization. | Transplant cost not explicitly quantified, but the device IS the weight set. | Spintronic | **No** — requires STNOs. |
| **DRAWNAPART (Laor et al., NDSS 2022, arxiv 2201.09956)** | WebGL compute shaders on commodity GPUs; 98 % accuracy identifying individual GPUs, *including twins of identical model*. | Identifies — does NOT compute on. Pure tag, no computation depends on it. | Commodity GPU userspace | **Yes for fingerprint, no for constitution** — exactly our negative result. |
| **Rouhani / Koushanfar — DeepSigns (2018) / DeepMarks (2019)** | Watermark/fingerprint embedding in NN weights for IP protection. | Model still runs anywhere; watermark just detectable. NOT constitutive. | Any | **Yes but useless for our goal** — model is still transferable. |
| **Wu et al., arxiv 2212.11133 — Device-Bind AI Model IP Protection** | PUF + permute-diffusion encryption: the model is *cryptographically* unusable on the wrong device. | Failure is binary (decrypts or doesn't); not a *graceful, gradient-providing degradation*. | Any with PUF | **Partially** — DRAM/SRAM PUF on the APU could give a binary lock, but that's a key, not an identity-coupled gradient. |
| **Picerno et al., arxiv 2310.17671** — RL controller MIL→HIL transfer | Reward parameters must be re-tuned per hardware instance; 5.9× speedup vs hardware-only training. | Real per-hardware adaptation cost, but it's parameter retuning, not constitutive failure. | Engine control | **Methodology** is portable: train sim, fine-tune per device. Not constitutive. |

### 1.2 Summary

Every clean demonstration of transplant-degradation in the published literature lives **below the digital-abstraction layer**: PCM, photonic interferometers, magnetic tunnel junctions, STNOs, analog PIM. Above the abstraction layer, the only "identity" researchers achieve is:

- **Fingerprinting** (DRAWNAPART, DeepSigns): identify, do not compute on.
- **Cryptographic binding** (PUF-encrypt): binary lock, no gradient.
- **Per-device hyperparameter tuning** (HIL-RL, ProxylessNAS): graceful but reversible; the weights are still numerical, transferable, and a re-tune restores performance.

**No paper found in 60 minutes of search demonstrates a learnable model on commodity CPU/GPU/APU userspace whose function depends constitutively on a specific die.** This is consistent with our 12 negative experiments.

---

## Section 2 — Theoretical obstacles

1. **Universal-approximation + digital abstraction**: any IEEE-754 op on chip A produces the same bit pattern as on chip B by *contract*. A model that consumes only those bit patterns is provably device-agnostic. Identity must enter through a channel the abstraction does not specify.

2. **Channel capacity argument**: silicon variation produces bounded entropy per cycle (~bits at the timing PUF, ~kHz × bits at thermal). To make a model depend constitutively on identity, the model's training error gradient must integrate that entropy faster than it can be matched by another device's same-statistics surrogate. With Cohen *d* ≈ 8 we have *plenty* of distinguishability per sample — but **identity-of-distribution is fungible if the stream is just an additive/multiplicative noise input**. This is exactly the SHUFFLE result we keep getting.

3. **Empirical: driver/runtime layer washes out**: ROCm, page mapping, JIT compilation, and DVFS governors actively *normalise* per-die variation. Anything above the driver sees device-conditional noise as i.i.d. samples from a distribution, not as a key.

4. **Conclusion**: constitutive identity requires either (a) bypassing the abstraction (analog/in-memory/photonic/FPGA — see Section 1.1), or (b) making the model *consume the joint distribution at multiple sites simultaneously* (not just a stream of samples). We haven't yet tried the latter cleanly.

---

## Section 3 — Pareto-frontier of HW additions

Ranked by ($ cost) / (probability of yielding real constitutive identity):

| Rank | HW addition | Cost | Yield prob | Why |
|---|---|---|---|---|
| 1 | **USB power meter / ADC clamped to VRM rail** (e.g. ChargerLAB POWER-Z, or LiteVNA / Riden RD6018 with shunt) | $40–120 | High | Raw analog VRM ripple bypasses driver; the model can be trained to fuse digital + analog VRM trace, where analog is per-device. Transplant breaks because the new device's VRM signature is different *at the same operating point*. |
| 2 | **External thermal camera with USB interface** (FLIR Lepton 3.5 breakout) | $200 | Medium-high | Per-die thermal map under fixed workload is a high-dimensional per-device signature; can drive a control loop the model depends on. |
| 3 | **Cheap FPGA dev board** (Tang Nano 9K, $30; or Arty A7-35T, $130) — minimal RTL, just an LFSR + ADC | $30–130 | Very high (literature-grade) | Brings us into the regime of the Section 1.1 papers. Real, citable, hard. |
| 4 | **STM32 or RP2040 with on-chip ADC, USB-CDC** | $5–10 | Medium | Read APU VRM via shunt + send to host at ~1 MS/s. Same idea as #1 at hobby cost. |
| 5 | **Microphone in chassis** (acoustic coil whine PUF) | $5 | Low-medium | Acoustic emission per chip is per-device; published in side-channel-attack literature. Sampling rate trivial. |
| 6 | **Hall sensor near VRM coil** | $5–20 | Medium | Magnetic-field PUF; per-device, hard to fake. |

**Pareto winner**: #1 (USB power meter, $40–120). Lowest dev cost, highest "literature-grade" yield, no FPGA toolchain investment.

---

## Section 4 — Recommended next experiment

Given:
- 12 NULL attacks at userspace abstraction layer.
- Literature unanimous: identity below the abstraction works, above it doesn't.
- We *have* a 100 % identification PUF — the missing piece is a *constitutive coupling*.

**Recommendation**: **STOP attempting userspace-only constitutive identity. PIVOT to one of two paths.**

- **Path A (cheap, fast, 1 week)**: Buy a USB ADC + clamp it on the APU VRM. Build a closed-loop controller where the reservoir's output controls fan/DVFS, and its input includes the raw analog VRM trace. Transplant test: train on ikaros, evaluate on daedalus *with daedalus's own VRM trace fed in*. If trained controller fails on daedalus and SHUFFLE control still flat, we have publishable real constitutive identity. Cost: ~$100, low risk.

- **Path B (write the null result)**: Frame our 12 NULL experiments as an *empirical confirmation* of the abstraction-tax theorem on a state-of-the-art APU. Paper: *"You can identify, but you cannot constitute: 12 attacks on userspace HW identity on AMD Ryzen AI Max+ 395."* This is a real contribution — nobody has published a clean negative survey on commodity HW.

**Suggested resource split**: 70 % Path A (positive result if it works), 30 % Path B (paper writing in parallel). Both are valid; both close the question.

---

## Section 5 — User-friendly summary

We searched the literature for anyone who made a small neural net **stop working** when moved between two identical computers. Nobody has done this on stock laptops. Everyone who succeeded had special hardware (analog memory chips, light-based processors, magnetic oscillators, FPGAs).

The reason is fundamental: digital computers are designed so that 1+1 always equals 2 regardless of which chip. Our 12 failed experiments are *evidence* of this, not a personal failure.

Two paths forward:
1. Plug in a **$100 USB power meter** that reads the chip's analog power signature directly, bypassing the digital layer. Train a controller that uses that signature in its loop. Then test if it breaks when moved.
2. **Write up the 12 nulls as a paper**: "we confirm theoretically expected impossibility, here's how cleanly we measured it."

We recommend doing both.

---

## References (verified URLs)

- DRAWNAPART: <https://arxiv.org/abs/2201.09956>, NDSS 2022.
- Joshi et al., PCM ResNet, Nat. Commun. 2020: <https://www.nature.com/articles/s41467-020-16108-9>, arxiv: <https://arxiv.org/abs/1906.03138>.
- Variability-Aware Training PIM: <https://arxiv.org/abs/2111.06457>.
- Single-shot optical NN (Bandyopadhyay et al., Sci. Adv. 2023): <https://www.science.org/doi/10.1126/sciadv.adg7904>.
- Tanaka et al. physical reservoir review, Neural Networks 2019: <https://arxiv.org/abs/1808.04962>.
- DeepSigns: <https://arxiv.org/abs/1804.00750>.
- Wu et al., Device-Bind AI Model IP Protection: <https://arxiv.org/abs/2212.11133>.
- Romera et al., STNO vowel recognition, Nature 2018: <https://www.nature.com/articles/s41586-018-0632-y>.
- Picerno et al., RL MIL→HIL transfer: <https://arxiv.org/abs/2310.17671>.
- Hardware-aware photonic NN (Mengu et al., Optica 2024): <https://opg.optica.org/optica/fulltext.cfm?uri=optica-11-8-1039>.
- Magnetoresistive on-chip-training-free: <https://www.science.org/doi/10.1126/sciadv.adp3710>.

## Oracle consensus (3-way: GPT-5, Gemini-2.5-Pro, Grok-4)

Deepseek not collected (dispatch budget exhausted). All three responding oracles **converge**:

| Q | GPT-5 | Gemini-2.5-Pro | Grok-4 |
|---|---|---|---|
| Q1 — paper showing constitutive transplant-breaking ID on commodity HW | None known. Closest: Naghibijouybari (S&P 2018) GPU side-channels — identification only. | None known. Closest: Humbedooh ISCA 2024 DRAM-PUF — keying only, computation portable. | None. Confirmed null across arXiv/IEEE/ACM/Nature 2015–2025. |
| Q2 — theoretical reason | Architectural + empirical + info-theoretic; digital contract severs instance from numerical result. | All three; abstraction layer = low-pass filter on physical signal. | Computational + empirical; IEEE-754 + driver layer + DVFS normalize away. |
| Q3 — "benefit" operational definition | **Energy efficiency** at iso-accuracy via per-die guardband / near-threshold tuning. | **Adversarial robustness**: HW noise = instance-specific augmentation. | **Lifetime/viability cost** via auxiliary loss on power_draw. |
| Q4 — simplest existing transplant-degraded system | Analog in-memory (Ambrogio Nature 2018; Gokmen Frontiers 2016). Port methodology = HW-in-loop calibration + in-situ fault modelling. | Physical Reservoir Computing (Appeltant Nat. Comm. 2011) — NOT portable, that's the whole point. | "Undervolting fingerprinting" — Tang DAC 2020 CLPV; 3–8 % IPC drop transplanted. **Portable via MSR/RAPL, no silicon needed.** |
| Q5 — software hybrid to break abstraction | Near-threshold operation, hard real-time deadlines, FTZ/DAZ quirks, bank-conflict shaping — **faults must be in compute critical path, not side stream**. | Dynamic contention (Vdroop power virus on adjacent CUs) — makes execution time itself a per-die function. | Pin 2–4 °C below throttle + per-CU perf counters as input. Phase-1 KL data already hints at this. |
| Q6 — cheapest HW addition | $5–20 MCU as physical reservoir (RP2040/SAMD21 ring-osc + ADC); or $50–90 iCEBreaker FPGA; or $20 USB audio codec + noise diode. | **<$30 USB ADC** + Zener diode noise source. Weekend project. | **$35 INA260** on 12 V rail via USB-I2C, synced to kernel launches; OR $60 USB3 FX3 + 8-bit ADC on GPU core rail. |
| Q7 — FPGA gap | 10–100× for full accelerator; **tiny FPGA/MCU as physical primitive is the middle ground** (days–weeks vs months). | Yes huge for full; ADC over USB **is** the Pareto-optimal middle. Q6 ≈ weekend, FPGA ≈ multi-month. | ~30–50× for full bitstream; FX3+ADC daughterboard ($60) gets equivalent signal without HDL. |
| Q8 — brutal honesty | **Yes.** Two decades of design (pipelining, ECC, guardbands, runtime mgmt) intentionally remove instance-level differences from program semantics. Phase-1 NULL is exactly what the abstraction-tax predicts. | **Yes.** Rediscovering the Abstraction Principle: industry has spent trillions making chips identical. You're calling a feature what they call a bug. | **Yes.** Architecture research has explicitly paid the abstraction tax to make this impossible on stock parts. NULL is expected outcome. |

### Where the oracles disagree (interesting)

- **Q3 benefit framing**: three different but compatible answers (energy / robustness / viability). All three are demonstrable; pick whichever has the cleanest controls. **Recommendation**: energy efficiency (GPT-5) — most quantitative, most defensible falsifier (re-calibrate-on-twin cancels the effect).
- **Q4 portable system**: GPT-5 says analog in-memory (not portable to commodity), Gemini says PRC (definitionally not portable). **Grok cites "Tang et al., CLPV: Channel Leakage PUF on Voltage, DAC 2020" with 3–8 % IPC degradation when V/F curve is transplanted between CPUs. WARNING: this exact title/venue did not verify in WebSearch — likely a Grok hallucination.** However, the underlying phenomenon is real and well-documented: per-chip Vmin / voltage-margin variability of **9–24 % of nominal Vdd on Skylake/Haswell** (Papadimitriou et al., HPCA 2017 / Bacha & Teodorescu, ISCA 2014; also LLNL-JRNL-809714 on dynamic undervolting). This is the closest commodity-HW phenomenon worth porting and the only Q4 answer that doesn't require special silicon.
- **Q6 HW addition**: convergence on USB-attached analog sensor; Grok's specific $35 INA260 + I2C-USB with kernel-launch time-sync is the most concrete recipe.

### Updated Section 4 recommendation (after oracle input)

**Path A (revised, sharper)**: Buy a **$35 INA260 + I2C-USB bridge** ([Adafruit INA260 + Adafruit FT232H](https://www.adafruit.com)) → clamp on the 12 V rail. Sample at 1 kS/s synced to HIP kernel-launch timestamps. Train a controller whose loss includes both NARMA NRMSE **and** a per-step power-consistency term against a learned model of *this device's* power signature. Transplant test on daedalus with the same hardware. Total cost ~$50, build ~1 weekend.

**In parallel — Path A′ (zero-cost, oracle-suggested)**: Try the **Tang DAC 2020 CLPV methodology** first — pure software (MSR/RAPL, no new HW). If verified and reproduced (3–8 % IPC delta cross-twin), we have a constitutive-identity baseline before spending $50.

**Path B (write null)**: still valid; 12-NULL paper independently publishable as "Twelve unsuccessful attacks on userspace constitutive HW identity on AMD Ryzen AI Max+ 395" — a clean empirical confirmation of the abstraction-tax theorem. Oracle agreement on Q8 strengthens the framing.

Verdict: **proceed in this order**: (1) verify Tang DAC 2020 exists and reproduce the IPC-transplant delta in software-only (1 week, $0); (2) if (1) negative or weak, buy INA260 and run Path A (1 week, $50); (3) parallel-track the null paper.
