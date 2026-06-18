# Identity Deep-Signals Sweep — Per-Die Physical Fingerprinting Mechanisms

**Date**: 2026-06-01
**Author**: Deep-research sweep
**Scope**: Enumerate every published mechanism for per-die physical fingerprinting (2005-2026); compare to our current 5-signal stack; identify (a) candidates to add and (b) attack vectors we should defend against.

---

## 1. Our current stack (5 signals)

| # | Signal | Layer |
|---|--------|-------|
| 1 | Inter-core TSC offset | CPU / clock |
| 2 | Cacheline ping-pong latency matrix | CPU / cache interconnect |
| 3 | DRAM refresh probing | Memory / refresh controller |
| 4 | Syscall p99.9 tail | OS / kernel scheduler |
| 5 | NVMe queue-tail | Storage |

Observation: stack is heavily CPU/memory-biased. **No GPU, no electromagnetic, no acoustic, no sensor, no peripheral, no power-domain, no crypto-accel signals.** This is the gap the literature exploits.

---

## 2. Mechanism inventory — 32 mechanisms across 20 categories

Legend:
- **Acc**: accessibility from unprivileged userspace on our HW (gfx1151 APU laptop + Daedalus/Minos): Y/N/Root/HW-mod
- **Stab**: per-die stability (decade): Days / Weeks / Months / Years
- **Used?**: in our 5-signal stack
- **AttackRisk**: can a remote/local attacker measure ours and forge?

| # | Mechanism | Best paper | Acc | Stab | Used? | AttackRisk |
|---|-----------|-----------|-----|------|-------|------------|
| **CPU timing** |||||||
| 1 | Inter-core TSC offset | Kohno 2005 [1], Sharma 2012 | Y | Months | **YES** | High (network-visible) |
| 2 | TCP-timestamp clock skew (remote) | Kohno 2005 [1] | Y(net) | Years | No | High — remote forgeable |
| 3 | Branch-predictor / BTB state | Trampert ESORICS'22 [2] | Y(JS) | Years (model-level) | No | Medium |
| 4 | Hardware prefetcher characterization | FetchBench CCS'23 [3], PREFETCHX [4] | Y | Years | No | Medium |
| 5 | Microarch cache associativity / size probes (uarch-fp) | Trampert ESORICS'22 [2] | Y(JS) | Years | No | High — JS-reproducible |
| **Cache & coherence** |||||||
| 6 | L1/L2/L3 latency map per-core | (Many; uarchfp) [2] | Y | Months | Partial (#2 ping-pong) | Medium |
| 7 | Cache replacement-policy quirks | FetchBench [3] | Y | Years | No | Medium |
| **Memory** |||||||
| 8 | DRAM refresh tail | folklore + our usage | Y | Months | **YES** | Medium |
| 9 | DRAM retention-failure pattern (decay-PUF) | Schaller 2018 [5], Tehranipoor [6] | Root (slow-refresh) | Months | No | **Low** (requires control of refresh) |
| 10 | Rowhammer-induced bit-flip distribution | FP-Rowhammer AsiaCCS'25 [7] | Root | Years | No | **Low** (hard to forge — physical) |
| 11 | DDR5 row-buffer micro-timing | TEE.fail 2025 [8] | HW-mod | Years | No | Low (needs interposer) |
| 12 | ECC scrub / patrol-scrub timing | (folklore) | Root | Months | No | Medium |
| **Storage** |||||||
| 13 | NVMe queue-tail | our usage | Y | Weeks | **YES** | High (host-OS noise dominates) |
| 14 | SSD/NAND wear-leveling residue / FTL quirks | (no clean fingerprint paper; FTL surveys [9]) | Root | Months | No | Medium |
| **Network** |||||||
| 15 | NIC PHY clock skew | Jana 2010 [10] | Y(net) | Years | No | High — passive observer |
| 16 | RF-fingerprint (transient spectrum) | Various RFF [11] | HW-mod | Years | No | Low (requires SDR) |
| **Power / RAPL** |||||||
| 17 | RAPL precision & per-package counters | Hertzbleed USENIX'22 [12] | Root (post-CVE) | Months | No | Medium |
| 18 | DVFS curve / freq-vs-power transfer function | Hertzbleed [12], Scheduled-Disclosure S&P'25 [13] | Y(timing) | Years | No | **High** (Hertzbleed-style) |
| 19 | Power-virus → CPU-Print timing | CPU-Print S&P'25 [14] | Y(web) | Months | No | **High** (browser-accessible) |
| **Thermal** |||||||
| 20 | Per-zone thermal offsets / RC constants | (folklore; thermal-CC literature) | Y | Months | No | Medium |
| 21 | Fan PWM response transfer function | (no published PUF — novel) | Y | Months | No | Low |
| **GPU** |||||||
| 22 | WebGL execution-unit skew (DrawnApart) | Laor NDSS'22 [15] | Y(web) | Months | No | **High** — web-accessible |
| 23 | GPU CUDA core frequency variation PUF | Li 2015 CCS poster [16] | Y(GPU) | Years | No | Medium |
| 24 | GPU thermal/power side-channel (Energon) | Tan ICCAD'25 [17] | Y(NVML / hwmon) | Months | No | Medium |
| 25 | Shader fp-rounding / per-CU latency | DrawnApart [15] | Y(GPU) | Years | No | Medium |
| **PCIe / interconnect** |||||||
| 26 | PCIe AER correctable-error counter | Linux AER docs [18] | Root | Months | No | Low |
| 27 | NVLink/Infinity-Fabric covert-channel — NVBleed | NVBleed arXiv'25 [19] | Y(multi-GPU) | Months | No | Low (we have 1 GPU) |
| **USB / peripheral** |||||||
| 28 | USB enumeration timing fingerprint | USPTO 10,169,567 [20] | Root | Years | No | Low |
| **Sensor (camera, mic, accel)** |||||||
| 29 | Webcam PRNU / dark-current pattern | Lukáš 2006 [21], stress-test '23 [22] | Y(if cam present) | Years | No | **Very Low** (physical sensor) |
| 30 | Microphone noise floor / freq response | Das 2014 [23], MicPrint [24] | Y(if mic) | Years | No | Low |
| 31 | Accelerometer calibration scale (laptops) | Das/Stanford 2014 [23] | Y(if present) | Years | No | Low |
| **EM / acoustic** |||||||
| 32 | DeMiCPU magnetic emissions | Cheng CCS'19 [25] | HW-mod (probe) | Years | No | Very Low |
| 33 | EM-ID near-field radiation | Disney Research / RTL-SDR [26] | HW-mod | Years | No | Very Low |
| 34 | Coil whine / capacitor squeal acoustic | (no peer-reviewed PUF paper) | Y(mic) | Months | No | Low |
| **Crypto / security IC** |||||||
| 35 | RDRAND latency variance | (folklore; no dedicated paper) | Y | Months | No | Medium |
| 36 | AES-NI / SHA-NI cycle-count quirks | (Intel optimization manual + uarchfp [2]) | Y | Years | No | Medium |
| 37 | TPM / fTPM timing | TPM-FAIL USENIX'20 [27] | Y(tpm2) | Years | No | Low |
| **SRAM / boot-time** |||||||
| 38 | SRAM startup state (large-CPU/GPU) | Bernstein 2015 [28], Holcomb 2008 [29] | Root | Years | No | Very Low (cold-boot only) |
| 39 | BIOS POST timing / ME firmware version | (folklore) | Root | Years (until update) | No | Medium |

---

## 3. Top 5 candidates to ADD to our signature

Selected on (a) per-die stability, (b) orthogonality to our 5 existing CPU-biased signals, (c) ease of unprivileged measurement on our gfx1151 APU + Daedalus/Minos rigs, (d) attacker difficulty to forge from a different machine.

| Rank | Mechanism | Why add | Implementation note |
|------|-----------|---------|--------------------|
| **A1** | **GPU per-CU shader-execution skew (DrawnApart-style on ROCm)** [15] | Independent silicon (GFX1151 die, not CPU CCD). High entropy (67% boost on top of standard fingerprint). Stable Months+. | Run identical matmul on each of 40 CUs via ROCm, measure per-CU completion-time histogram. We already have HSA_OVERRIDE setup. |
| **A2** | **DRAM Rowhammer bit-flip fingerprint (FP-Rowhammer)** [7] | 99.91% accuracy on 98 modules, stable across 10+ days, *physical* and very hard to forge from a different machine. | Needs root + DDR access; we have it. Probably the strongest anti-spoofing signal in the whole inventory. |
| **A3** | **Hertzbleed-style DVFS curve fingerprint** [12,13] | Per-die F-vs-P transfer function differs even between binned chips. Unprivileged. Stable across firmware updates. | Sweep workload power, sample MSR_PERF_STATUS / hwmon freq; build P(f) curve. |
| **A4** | **Webcam PRNU / dark-current pattern** [21,22] | Sensor-level — completely *orthogonal* to CPU/memory. Years-stable. Very low attacker forge-risk (needs the actual sensor). | Capture 30 dark frames (lens covered), compute per-pixel mean residual, average to a 64-bit hash. |
| **A5** | **Hardware-prefetcher characterization vector (FetchBench)** [3] | Independent of cache-ping-pong (#2); reveals prefetcher type + parameters; cheap; per-microarchitecture-and-die. | Open-source FetchBench harness; ~ 1 min runtime. |

These 5 give us GPU + memory-physical + power-curve + optical-sensor + prefetcher dimensions — completely covering the gaps in the current stack.

---

## 4. Top 5 attack vectors against our current 5-signal design

| Rank | Attack | Target signal(s) | Why we are vulnerable | Mitigation |
|------|--------|-----------------|----------------------|-----------|
| **V1** | **Hertzbleed-style remote replay of TSC + syscall p99.9** [12] | #1, #4 | Both are pure timing channels measurable remotely; attacker on same hyperthread can mimic. | Add a *physical*-rooted signal (PRNU or Rowhammer) so replay alone is insufficient. |
| **V2** | **ML modeling attack** (Rührmair-style) [30] | All 5 (purely behavioural) | Our signals are all behavioural — a logistic-regression / neural-net surrogate trained on N challenges can predict our responses with linear CRPs. | Adopt **Controlled-PUF** wrapper (Suh+Devadas) [31]: hash signals through a keyed PRF, never expose raw responses; rate-limit. |
| **V3** | **Helper-data leakage** [32,33] | Identity-enrollment metadata | If we publish any fuzzy-extractor helper data, ML attacks can model the underlying PUF from helper alone (proved for SRAM-PUF). | Use **Reverse Fuzzy Extractor** [34] — verifier does heavy lifting, no helper data leaves device. |
| **V4** | **CPU-Print web-side power-virus** [14] | #1, #4 (TSC + syscall tail) | Attacker can run our exact probing JS in a browser to *enroll* the victim and impersonate offline. | Bind identity to non-web-reachable signals (PRNU camera, Rowhammer pattern, TPM EK). |
| **V5** | **DDR5 interposer / TEE.fail-class physical probe** [8] | #3 (DRAM refresh) | If attacker gains physical access to a DIMM, refresh-timing is observable on the bus. | Combine refresh probe with PRNU + a sensor-bound nonce so DIMM swap alone doesn't yield identity. |

Plus: **Hertzbleed CVE-2022-23823 specifically affects AMD Zen** — gfx1151 host CPU is in scope; our signal #4 (syscall p99.9 tail) can leak DVFS-correlated info that an off-CPU attacker can use to *enroll without consent*.

---

## 5. Defensive tactics from the PUF literature

| Tactic | Reference | Applicability |
|--------|-----------|--------------|
| **Controlled PUF** — wrap raw response in keyed PRF + access control | Suh & Devadas 2007 [31] | Wrap our 5+5 signal vector through HMAC(secret, ·) before exposing externally. |
| **Reverse Fuzzy Extractor** — invert who-does-the-correction so helper data stays on-device | Van Herrewege FC'12 [34] | Use for enrollment so we never publish helper data publicly. |
| **Distance-bounding / time-bounded challenge** | Brands–Chaum, see survey [35] | Bound RTT of identity probe so a remote replayer is exposed. |
| **Continuous authentication** (re-prove identity every N seconds) | HA-CAAP IoT'24 [36] | Re-evaluate ≥1 signal/minute so a one-shot enrollment can't be replayed long-term. |
| **Multi-factor / heterogeneous channels** | DrawnApart [15], FP-Rowhammer [7] consensus | Combine ≥3 *physically independent* substrates (CPU + GPU + sensor) so no single attack vector breaks identity. |

---

## 6. Recommendation: expanding 5 → 10 signals safely

Proposed 10-signal stack (add A1-A5 from §3 to the existing 5):

```
existing  1. TSC offset
existing  2. Cacheline ping-pong matrix
existing  3. DRAM refresh probe
existing  4. Syscall p99.9 tail
existing  5. NVMe queue-tail
NEW       6. ROCm per-CU shader skew vector       (A1, DrawnApart-style)
NEW       7. Rowhammer bit-flip mask              (A2, FP-Rowhammer-style)
NEW       8. DVFS frequency-vs-power curve        (A3, Hertzbleed-derived)
NEW       9. Webcam PRNU dark-frame hash          (A4)
NEW      10. FetchBench prefetcher-parameter vec  (A5)
```

### Why this 5+5 split is "safe"

1. **Independent substrates.** 1-2-3-4-5 are CPU/CCD/scheduler — same die. 6 is GPU die. 7 is DRAM module. 8 is package-level analog. 9 is CMOS image sensor (separate vendor). 10 is microarchitectural model (per family but mixes with per-die latencies in 1, 2). No single physical attack compromises >2 signals.
2. **Anti-modeling.** Wrap all 10 outputs through a Controlled-PUF HMAC (recommend A in §5). Raw responses never leave the device. Modeling attacks (Rührmair [30]) need raw CRPs.
3. **Anti-replay.** Use Reverse-Fuzzy-Extractor for enrollment ⇒ no helper data leak.
4. **Anti-Hertzbleed.** Signals 7 (Rowhammer) and 9 (PRNU) cannot be inferred from CPU power/frequency variations alone — they need the actual DRAM module and the actual image sensor.
5. **Anti-CPU-Print / DrawnApart-as-attack.** Both attacks become *part* of our identity, so we explicitly require they match — an attacker who runs them on a *different* device gets the wrong answer.

### Engineering notes for adding A1-A5 on gfx1151

| Signal | Where measured | Estimated 1-shot cost |
|--------|---------------|----------------------|
| A1 ROCm per-CU skew | HIP kernel + chrono | 5-15 s |
| A2 Rowhammer mask | sudo + /dev/mem | 5 s [7] |
| A3 DVFS curve | hwmon7/freq1_input under power sweep | 30 s |
| A4 Webcam PRNU | v4l2 dark capture (lens-cap test) | 10 s |
| A5 FetchBench | open-source repo, single-core | 60 s |

Total enrollment: ~2 min. Continuous re-verification: pick one signal per minute round-robin, ~ 0.1% CPU.

---

## 7. References (cited URLs)

[1] Kohno, Broido, Claffy. *Remote Physical Device Fingerprinting*. IEEE TDSC 2005. https://homes.cs.washington.edu/~yoshi/papers/PDF/KoBrCl2005PDF-Extended-lowres.pdf
[2] Trampert, Rossow, Schwarz. *Browser-Based CPU Fingerprinting*. ESORICS 2022. https://misc0110.net/files/uarchfp_esorics22.pdf
[3] Schlüter et al. *FetchBench: Systematic Identification and Characterization of Proprietary Prefetchers*. CCS 2023. https://publications.cispa.saarland/3991/1/ccs23-fetchbench.pdf
[4] Chen et al. *PREFETCHX: Cross-Core Cache-Agnostic Prefetcher-based Side-Channels*. 2024. https://arxiv.org/pdf/2306.11195
[5] Tehranipoor et al. *DRAM PUF survey*; Schaller et al. *Decay-Based DRAM PUFs in Commodity Devices*. HOST 2017/2018. https://research.tue.nl/en/publications/decay-based-dram-pufs-in-commodity-devices/
[6] Giechaskiel et al. *Fingerprinting Cloud FPGA Infrastructures*. 2020. https://ilias.giechaskiel.com/papers/2020_1_fingerprinting_fpga.pdf
[7] Venugopalan et al. *FP-Rowhammer: DRAM-Based Device Fingerprinting*. AsiaCCS 2025. https://arxiv.org/abs/2307.00143  — 99.91% on 98 modules, 10-day stable, <5 s extraction.
[8] *TEE.fail: Breaking Trusted Execution Environments via DDR5 timing*. 2025. https://tee.fail/files/paper.pdf
[9] Linux SSD/FTL surveys. https://www.sciencedirect.com/topics/computer-science/flash-translation-layer
[10] Jana, Kasera. *On the Reliability of Wireless Fingerprinting using Clock Skews*. WiSec/Dartmouth. https://cs.dartmouth.edu/~sergey/skew/clock-skew.pdf
[11] *Wireless Device Identification Based on RF Fingerprint Features*. IEEE 2020. https://ieeexplore.ieee.org/document/9149226/
[12] Wang, Paccagnella et al. *Hertzbleed: Turning Power Side-Channel Attacks into Remote Timing Attacks on x86*. USENIX Sec 2022. https://www.hertzbleed.com/hertzbleed.pdf — AMD CVE-2022-23823.
[13] Paccagnella et al. *Scheduled Disclosure: Turning Power Into Timing Without Frequency Scaling*. S&P 2025. https://www.cs.cmu.edu/~rpaccagn/papers/scheduled-disclosure-sp2025.pdf
[14] Goswami, Venugopalan et al. *CPU-Print: From Multiplying Matrices to Uniquely Identifying CPUs*. IEEE S&P 2025 (poster). https://sp2025.ieee-security.org/downloads/posters/sp25posters-final14.pdf
[15] Laor et al. *DRAWNAPART: Device Identification via Remote GPU Fingerprinting*. NDSS 2022. https://www.ndss-symposium.org/wp-content/uploads/2022-93-paper.pdf — +67% median tracking duration.
[16] Li et al. *A Hardware Fingerprint Using GPU Core Frequency Variations*. CCS 2015 poster. http://ittc.ku.edu/~bluo/pubs/2015_ccs_poster.pdf
[17] *"Energon": Unveiling Transformers from GPU Power and Thermal Side-Channels*. ICCAD 2025. https://arxiv.org/abs/2508.01768
[18] *Linux Kernel PCIe AER HOWTO*. https://docs.kernel.org/PCI/pcieaer-howto.html
[19] *NVBleed: Covert and Side-Channel Attacks on NVIDIA Multi-GPU Interconnect*. 2025. https://arxiv.org/pdf/2503.17847
[20] USPTO 10,169,567 — *Behavioral authentication of USB devices via enumeration timing*. https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10169567
[21] Lukáš, Fridrich, Goljan. *PRNU Tutorial — Digital camera identification from sensor pattern noise*. http://ws2.binghamton.edu/fridrich/Research/full_paper_02.pdf
[22] *A Stress Test for Robustness of PRNU Identification on Smartphones*. Sensors 2023. https://www.mdpi.com/1424-8220/23/7/3462
[23] Das, Borisov, Caesar. *Mobile Device Identification via Sensor Fingerprinting*. arXiv:1408.1416. https://arxiv.org/pdf/1408.1416
[24] *MicPrint: Acoustic Sensor Fingerprinting for Spoof-Resistant Mobile Authentication*. NSF 2018. https://par.nsf.gov/servlets/purl/10156908
[25] Cheng et al. *DeMiCPU: Device Fingerprinting with Magnetic Signals Radiated by CPU*. CCS 2019. https://dl.acm.org/doi/10.1145/3319535.3339810
[26] *EM-ID: Tag-less Identification of Electrical Devices via Electromagnetic Emissions*. Disney/USPTO 10,366,118. https://www.rtl-sdr.com/disney-research-em-id-rtl-sdr-based-tag-less-id-of-electrical-devices-via-eletromagnetic-emissions/
[27] Moghimi, Sunar et al. *TPM-FAIL: TPM meets Timing and Lattice Attacks*. USENIX Sec 2020. https://tpm.fail/tpmfail.pdf
[28] Bernstein et al. *Investigating SRAM PUFs in large CPUs and GPUs*. 2015. https://cr.yp.to/hardware/cpupuf-20150729.pdf
[29] Holcomb et al. *Power-Up SRAM State as Identifying Fingerprint and TRNG Source*. IEEE TC 2008. https://archiv.infsec.ethz.ch/education/as09/secsem/papers/SRAM.pdf
[30] Rührmair et al. *Modeling Attacks on Physical Unclonable Functions*. CCS 2010. https://people.csail.mit.edu/devadas/pubs/ccs_attack_puf.pdf
[31] Suh, Devadas. *Physical Unclonable Functions for Device Authentication and Secret Key Generation*. DAC 2007 (Controlled PUF).
[32] Delvaux et al. *Helper Data Algorithms for PUF-Based Key Generation: Overview and Analysis*. https://lirias.kuleuven.be/server/api/core/bitstreams/e4af24fe-06cf-49f5-843b-241463c07beb/content
[33] *Machine Learning of PUFs using Helper Data*. IACR ePrint 2020/888. https://eprint.iacr.org/2020/888.pdf
[34] Van Herrewege et al. *Reverse Fuzzy Extractors: Enabling Lightweight Mutual Authentication for PUF-enabled RFIDs*. FC 2012. https://fc12.ifca.ai/pre-proceedings/paper_89.pdf
[35] *Security of Distance-Bounding: A Survey*. ACM CSUR. https://www.researchgate.net/publication/327896552_Security_of_Distance-Bounding_A_Survey
[36] *HA-CAAP: Hardware-Assisted Continuous Authentication & Attestation Protocol*. PMC 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC11487452/

---

## 8. Bottom-line findings

1. **Our stack is too CPU-homogeneous.** All 5 signals live on the same die or in OS-mediated timing. A single Hertzbleed/CPU-Print-class attacker can plausibly forge several at once.
2. **Three "killer" additions** are inexpensive and orthogonal: GPU per-CU skew (DrawnApart), Rowhammer bit-flip mask (FP-Rowhammer), and webcam PRNU. Each lives on a *physically distinct* substrate.
3. **All five current signals are behavioural** and therefore in the modeling-attack regime described by Rührmair (2010). We should wrap them in a Controlled-PUF construction with a Reverse Fuzzy Extractor.
4. **Two CVEs directly apply** to our gfx1151 platform: Hertzbleed CVE-2022-23823 (AMD) means signal #4 leaks DVFS info; the recent CPU-Print poster (S&P'25) shows the matmul-power-virus pattern works browser-side on Zen — directly attacks signals #1 and #4.
5. **No published per-die fingerprint exists** for: ECC scrub timing, fan PWM transfer function, coil-whine acoustic spectrum, or NVMe FTL-residue. These are open novelty angles if we want a paper contribution rather than just defense.

