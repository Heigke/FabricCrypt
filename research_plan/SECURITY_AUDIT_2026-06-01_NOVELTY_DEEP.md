# SECURITY_AUDIT_2026-06-01 — NOVELTY DEEP DIVE

**Auditor stance**: adversarial. Goal is to puncture every novelty claim before a peer reviewer does.
**Platform under test**: AMD Ryzen AI Max+ PRO 395 ("Strix Halo"), gfx1151 RDNA3.5 iGPU; microcode `0xb70001e`; family/model/stepping 26/112/0 — identical across both chassis.
**Our five claimed signals**:
1. Inter-core TSC offset PUF — D=0.91 inter-chassis
2. Cacheline ping-pong latency matrix — Frobenius distance 79 cyc
3. DRAM refresh-interval probing (no rowhammer) — KS p = 5.8e-32
4. Syscall p99.9 tail-latency — 1.25× inter-chassis ratio
5. NVMe queue-tail latency — ~1.3× inter-chassis ratio

---

## A. Per-signal prior-art table & verdict

Overlap rating: 0 = independent · 1 = different domain, same idea · 2 = same domain, different mechanism · 3 = same mechanism, different platform · 4 = same mechanism+platform, weaker claim · 5 = our claim is a special case of theirs.

### Signal 1 — Inter-core TSC offset PUF

| # | Paper / artifact | Year | What they measured | Overlap |
|---|---|---|---|---|
| 1 | Kohno, Broido, Claffy — "Remote Physical Device Fingerprinting", S&P 2005 / TDSC | 2005 | TCP-timestamp clock skew of a single host clock, remote observer | 2 — same family (clock skew → ID) but **single clock**, network observer, no inter-core delta |
| 2 | Sanchez-Rola et al. — "Clock Around the Clock", CCS 2018 | 2018 | Browser-side timing, GPU vs CPU relative skew | 2 — pairs of clocks, but cross-component (CPU/GPU), not inter-core TSC |
| 3 | Viennot — `core-to-core-latency` tool + chipsandcheese, jprahman writeups | 2021- | Core-to-core round-trip cycles; documents non-trivial intra-package variation | 3 — measures the same physical phenomenon (cross-core fabric timing) but framed as **microarch characterisation**, not identity/PUF; no inter-chassis discrimination claim |
| 4 | "Hardware Fingerprinting Using HTML5" (Sanchez et al., arXiv 1503.01408) | 2015 | JS performance.now skew between cores reachable from a browser | 3 — closest hit. Uses JS-visible timing differences across "virtual cores"; lower resolution, no rdtscp, no PUF claim, no AMD multi-CCD |
| 5 | USPTO US8122278 "Clock skew measurement for multiprocessor systems" | 2012 | Intra-die clock-skew measurement (functional, not identity) | 2 — measurement infra only |

**No paper found** that (a) directly uses `rdtscp` inter-core deltas, (b) frames them as a PUF/fingerprint, (c) demonstrates inter-chassis discrimination on identical SKU. The Kornau HOST-2023 "TSC entropy" paper claimed by DeepSeek **could not be located** on dblp/IEEE Xplore/HOST 2023 program — *probable hallucination*; do not cite.

**Verdict: NEW-ANGLE.** The mechanism (sibling-core TSC offset) is documented as a microarchitectural curiosity; framing it as an inter-chassis stable PUF on a fixed AMD APU is novel. Strong reviewer pushback expected: "isn't this just Kohno modernised?" Counter: Kohno is single-clock + network jitter; we're intra-die fabric-routing variation that survives identical microcode.

### Signal 2 — Cacheline ping-pong latency matrix

| # | Paper / artifact | Year | What they measured | Overlap |
|---|---|---|---|---|
| 1 | NetCAT (Kurth et al., S&P 2020 / NDSS 2019 disclosure) | 2019/20 | LLC PRIME+PROBE remotely via DDIO — Intel server | 1 — attack/leakage, not identity; Intel-specific |
| 2 | Viennot core-to-core-latency tool + Lam (chipsandcheese) "Core to Core Latency on Large Systems" | 2021- | Cycle latency of cross-core cache-line transfers | 3 — same physical quantity; no identity claim, no matrix-as-fingerprint framing |
| 3 | Sapphire Rapids C2C analysis (Rahman, Substack) | 2023 | Within-package C2C variation | 3 — same observation, Intel, no PUF |
| 4 | Investigating SRAM PUFs in CPUs/GPUs (Saß et al., arXiv 1507.08514) | 2015 | Bootloader SRAM contents on AMD64 as PUF | 2 — different mechanism (SRAM start-up), same goal (CPU PUF) |
| 5 | iPUF — Interconnect-Based PUF (Cui et al., 2019) | 2019 | Crosstalk on interconnect wires (custom IC) | 2 — sibling idea (interconnect → PUF) in custom silicon |

**Verdict: NEW-ANGLE.** The matrix-Frobenius framing on a *commodity AMD APU with multi-CCD layout* and inter-chassis stability has no direct prior art. Closest threat is "this is just running Viennot's tool and calling it a PUF" — must explicitly show: (i) intra-chassis stability across reboots, (ii) inter-chassis D ≫ intra, (iii) microcode-identical confound controlled. Reviewer will demand n>2 chassis.

### Signal 3 — DRAM refresh-interval probing (non-rowhammer)

| # | Paper / artifact | Year | What they measured | Overlap |
|---|---|---|---|---|
| 1 | Schaller et al. / Tehranipoor / Keller — DRAM retention PUF (refresh-pausing) | 2014–18 | Bit-flip pattern when refresh is paused | 4 — same physical substrate (refresh → identity); we don't pause, we *probe interval distribution* |
| 2 | Sutar et al. — D-PUF (VRT-based DRAM PUF), ACM TECS 17 | 2017 | Variable Retention Time bit-flips | 3 — different observable (bit error vs interval timing) |
| 3 | Keysight / Tehranipoor "DRAM Latency PUF" (HPCA'18 Kim et al.) | 2018 | tRCD violation latency as PUF | 3 — close: latency-not-flips, but the latency they measure is *access* latency, not refresh-interval distribution |
| 4 | FP-Rowhammer (Venugopalan et al., AsiaCCS 2025) | 2025 | Rowhammer bit-flips → 99.91% device ID on 98 modules | 4 — direct competitor on the same "DRAM device fingerprint" claim; uses rowhammer, we don't |
| 5 | PreLatPUF (arXiv 1808.02584); EPUF (arXiv 2307.09968) | 2018, 2023 | DRAM precharge latency variations | 3 — latency-based DRAM PUFs, different timing observable |

**Verdict: NEW-ANGLE / borderline KNOWN.** "DRAM as identity" is a well-trodden field (Schaller/Keller/Tehranipoor/Sutar/Kim). Our specific observable — the *statistical distribution* of refresh-interval timing as observed from user space without pausing refresh and without inducing flips — is a novel observable inside a saturated field. KS-distance discrimination is unusual framing. **Highest novelty risk of the five.** Reviewer will say: "DRAM Latency PUF (Kim HPCA'18) already does latency-based DRAM ID; show me what your refresh-interval observable gives that theirs doesn't." We need an explicit head-to-head ablation vs DRAM-Latency-PUF observable.

### Signal 4 — Syscall p99.9 tail latency

| # | Paper / artifact | Year | What they measured | Overlap |
|---|---|---|---|---|
| 1 | Hertzbleed (Wang et al., USENIX Sec 22) | 2022 | DVFS-induced data-dependent frequency timing on Intel+AMD Zen2/3 | 1 — same substrate (frequency/timing jitter) used for *leakage*, not identity |
| 2 | Kohno 2005 (again) | 2005 | Network-side timing | 2 |
| 3 | Sanchez-Rola "Clock Around the Clock" CCS'18 | 2018 | Browser timing primitives | 2 |
| 4 | "A methodology to identify identical single-board computers" (arXiv 2106.08209) | 2021 | Local microbench timing on identical Raspberry Pi | 3 — most similar in *spirit*; uses local timing → identity for identical SBCs. ARM, not AMD; doesn't isolate syscalls |
| 5 | CPU-Print (IEEE S&P 2025 poster, sp25posters-final14.pdf) | 2025 | Power-virus-driven matmul timing → CPU identity | 3 — directly competing: "userspace timing → CPU identity on identical SKUs". Different payload (matmul + power virus) but same threat model & goal |

**Verdict: KNOWN angle, NEW observable.** The "identical-SKU local-timing fingerprint" niche already has two strong claimants (arXiv 2106.08209 and IEEE S&P 2025 poster CPU-Print). Syscall p99.9 specifically is not what either of them uses, so we have an observable-level delta. But "syscall jitter as ID" feels weak as a standalone contribution. Reviewer-killer risk: ~1.25× ratio is small. Without a much larger n, this could look like noise.

### Signal 5 — NVMe queue-tail latency

| # | Paper / artifact | Year | What they measured | Overlap |
|---|---|---|---|---|
| 1 | "FROST: fingerprinting remotely using OPFS-based SSD timing" (2025 disclosure) | 2025 | Cross-origin web spying via SSD contention timing | 1 — uses SSD timing, but for *user activity* inference, not device ID |
| 2 | "SSD-iq" (Haas et al., VLDB 2024/25) | 2024 | Hidden SSD performance variability characterisation | 1 — describes the substrate variability we'd exploit, doesn't fingerprint |
| 3 | RAIL, Tiny-Tail Flash (FAST 17 / TOS) | 2017–21 | Flash tail-latency engineering | 0 — system design, not fingerprinting |
| 4 | USPTO 11073987 / 12271589 — "Identifying SSDs with lowest tail latencies" | 2021/24 | Industrial QC; binning SSDs by tail-latency profile | 3 — *patents* covering tail-latency as a per-device signature for selection. Possibly blocks commercial deployment of identity-from-tail-latency for SSDs |
| 5 | DRAWNAPART (Laor et al., NDSS 2022) | 2022 | GPU execution-unit timing → identity, JS-collectable | 2 — sibling concept, different device class |

**Verdict: NEW-ANGLE but weakest claim.** No academic paper directly fingerprints hosts via NVMe queue-tail latency. However USPTO 11073987 + 12271589 cover "tail latency as a per-SSD signature" for industrial binning — *patent overlap is real*; an identity-from-NVMe-tail product claim may be encumbered. As an academic observation it remains publishable. The 1.3× ratio is small; n=2 chassis is far too few; this signal would not survive peer review on its own.

---

## B. Per-platform novelty (AMD Strix Halo / Zen5 / RDNA3.5)

- **No "Strix Halo PUF" paper exists**, period. Search across arXiv/ACM/IEEE/USENIX/HOST returned nothing.
- **No "AMD Zen5 hardware fingerprint" paper exists** beyond the Hertzbleed line (which is leakage, not identity) and the RDSEED-bias disclosure (Keysight 2025 / Linux Journal 2026) — unrelated to identity.
- DRAWNAPART (NDSS 2022) fingerprints GPUs but on integrated Intel HD / discrete Nvidia/AMD discrete generations pre-RDNA3.5. RDNA3.5 iGPU on Strix Halo is uncharted.
- Platform-specific novelty *is* publishable independently when the platform has architectural novelty. Strix Halo qualifies: it's a 16-core Zen5 + RDNA3.5 + on-package LPDDR5x unified memory APU, a brand-new SoC class. A solid "Multi-signal fingerprinting on AMD Strix Halo APU" empirical paper is publishable at ACSAC / RAID / HOST as a *systems & measurement* contribution even if individual mechanisms are extensions of prior work.

## C. Venues & deadlines (FY26 cycle)

| Venue | Type | Deadline (FY26) | Fit | Comment |
|---|---|---|---|---|
| **HOST 2026** | IEEE conf, hardware security | Sep 1 2025 (passed); next round = HOST 2027 (~Sep 2026) | **Best fit** — exact scope: side-channels, hw fingerprint, PUF | Most natural home. Page limit 10 dbl-col IEEE. |
| **ACSAC 2026** | conf, applied security | **May 26 2026 (CLOSED; just passed)** — next cycle ACSAC 2027 ~May 2027 | Good fit (applied, measurement-heavy) | Just missed; aim ACSAC 27 |
| **RAID 2026** | conf, intrusion+defenses | Apr 16 2026 (passed) | Marginal fit — RAID prefers attack/defense, less PUF-centric | Skip |
| **IEEE TIFS** | journal, info forensics & security | Rolling | **Excellent fit** for journal-length empirical PUF paper | No deadline pressure; bigger evaluation expected (n≥10 chassis) |
| **WPES @ CCS 2026** | workshop, privacy | Typically Aug 2026 | Good fit for the "tracking-via-fingerprint" angle | Lower bar, fast turnaround |
| **IACR ePrint** | preprint server | rolling | Not a venue — only for cryptographic PUFs; OK as preprint mirror | Use for early flag-planting |
| **arXiv cs.CR** | preprint | rolling | Mandatory — plant flag immediately to establish priority over CPU-Print poster | Do this first |

**Realistic plan**: arXiv preprint within 2 weeks → polish + expand n → submit to HOST 2027 (Sep 2026) and/or IEEE TIFS journal in parallel. Suggested title:

> *"FabricPrint: Multi-Signal Identity Fingerprinting on the AMD Strix Halo APU via Inter-Core TSC Offsets, Cache-Coherence Latency, and DRAM Refresh Timing"*

## D. Patent landscape (USPTO)

Quick search results:
- **US8700943** / **US20110154090A1** — "Controlling TSC offsets for multiple cores and threads" (Intel) — covers virtualisation TSC-offset *setting*, not measurement-for-identity. **No block.**
- **US8122278** — "Clock skew measurement for multiprocessor systems" — generic measurement, no identity claim. **No block.**
- **US9197624 / US9596238 / US8789158** — "Using clock drift, clock skew, and network latency to enhance machine identification" — covers *combined* skew+latency for ID. This is the most likely commercial-IP overlap for an identity-from-timing product, broad claims. **Potential block for commercialisation; not for publication.**
- **US10103895** — "Method for PUF-identification generation" — generic, doesn't read on our specific signals.
- **US11073987 / US12271589** — "System and method for identifying SSDs with lowest tail latencies" — covers tail-latency as a per-SSD signature. **Potential block on Signal 5 commercial product.**

For an *academic* publication, none of these block. For any productisation (e.g. licensing a "FabricPrint" library), an FTO review against US9197624 and US11073987 family is mandatory.

---

## E. BRUTAL HONESTY

### Is anything genuinely new?

Honest read on each signal, stripped of marketing:

| # | Signal | Honest novelty | Single-best prior art that hurts us |
|---|---|---|---|
| 1 | Inter-core TSC offset PUF | **NEW-ANGLE (medium-strong)** — observable is well-known to perf engineers but never published as a PUF on a commodity APU with inter-chassis identity demonstrated | Sanchez-Rola "HTML5 fingerprinting" 2015; Viennot tool |
| 2 | C2C cache ping-pong matrix | **NEW-ANGLE (medium)** — matrix framing as identity is new, raw observable is widely known | chipsandcheese / jprahman C2C variation writeups (informal); iPUF Cui 2019 |
| 3 | DRAM refresh-interval probing | **NEW-ANGLE (weak)** — DRAM-as-identity field is saturated; our specific observable is a thin slice | DRAM Latency PUF Kim HPCA'18; FP-Rowhammer AsiaCCS'25 |
| 4 | Syscall p99.9 tail latency | **REDISCOVERED** in spirit — CPU-Print poster S&P'25 and arXiv 2106.08209 already stake "local timing → identical-SKU ID" | CPU-Print 2025 poster |
| 5 | NVMe queue-tail latency | **NEW-ANGLE (weak)** as academic claim; patent-encumbered for commercial product | USPTO 11073987 family |

### Are we just rediscovering Kohno + modernisation?

Partially yes for Signal 4. **Defensibly no** for the package as a whole: Kohno is a *single network-visible clock skew*, we are five *intra-die fabric-level* observables on a specific new SoC class. The thesis "the silicon-fabric leaks identity below microcode parity" is a real, defensible, novel angle.

### Top-3 risks to the novelty claim

1. **CPU-Print (IEEE S&P 2025 poster)** — directly competing "userspace timing → identical-CPU identity" claim, only ~12 months ahead of us. Reviewer will say "scooped". Counter-attack: CPU-Print uses matmul + power-virus, we use **passive** signals (no power virus). Make this *front and centre* in the abstract.
2. **DRAM Latency PUF (Kim HPCA'18) + FP-Rowhammer (AsiaCCS'25)** — DRAM identity space is densely populated. Signal 3 cannot stand alone; must be presented as one of multiple complementary signals.
3. **n=2 chassis is fatal.** Every PUF paper since 2010 evaluates on ≥10 devices (FP-Rowhammer: 98; DRAWNAPART: 2500). With n=2 we cannot compute false-accept / false-reject rates. Reviewer-instant-desk-reject risk **HIGH** unless we expand to n ≥ 6 chassis.

### Smallest publishable claim we can defend

> **"On the AMD Strix Halo APU, inter-core TSC offset and cache-coherence ping-pong latency form a passive, microcode-invariant hardware fingerprint that survives reboot and discriminates between identical-SKU chassis. We report inter-chassis Frobenius distances ≥30× intra-chassis variance on n=[6+] devices, and characterise the contribution of the multi-CCD fabric to the observed skew."**

This drops Signals 3, 4, 5 to *supporting evidence* and centres the novel-est observable (inter-core TSC offset on a multi-CCD Zen5). Targets HOST 2027 or IEEE TIFS.

### Combined claim verdict

- Standalone "5 novel PUFs" claim: **OVERCLAIM**. Two of five (Signals 4, 5) are rediscovered/patent-overlapping.
- Standalone "Strix Halo silicon-fabric fingerprint via passive multi-signal measurement": **DEFENSIBLE NEW-ANGLE** at HOST/TIFS bar.
- Required mitigations before submission: (a) expand to n ≥ 6 chassis; (b) explicit head-to-head ablation vs Viennot core-to-core-latency, CPU-Print, and DRAM-Latency-PUF observables; (c) drop or demote Signals 4–5 from primary contributions.

---

## Required cite list (reviewers will demand all of these)

1. Kohno, Broido, Claffy. *Remote Physical Device Fingerprinting.* IEEE TDSC 2(2), 2005. https://homes.cs.washington.edu/~yoshi/papers/PDF/KoBrCl2005PDF-Extended-lowres.pdf
2. Sanchez-Rola, Santos, Balzarotti. *Clock Around the Clock: Time-Based Device Fingerprinting.* ACM CCS 2018. https://www.s3.eurecom.fr/docs/ccs18_iskander.pdf
3. Sanchez et al. *Hardware Fingerprinting Using HTML5.* arXiv:1503.01408, 2015. https://arxiv.org/abs/1503.01408
4. Laor, Mehanna et al. *DRAWNAPART: Device Identification via Remote GPU Fingerprinting.* NDSS 2022. https://arxiv.org/abs/2201.09956
5. Saß et al. *Investigating SRAM PUFs in large CPUs and GPUs.* arXiv:1507.08514, 2015. https://arxiv.org/abs/1507.08514
6. Cui et al. *Interconnect-Based PUF With Signature Uniqueness Enhancement.* 2019.
7. Kim, Patel, Hassan, Mutlu. *The DRAM Latency PUF.* HPCA 2018. https://people.inf.ethz.ch/omutlu/pub/dram-latency-puf_hpca18.pdf
8. Schaller, Xiong, Anagnostopoulos, et al. *Intrinsic Rowhammer PUFs* (incl. earlier Schaller retention-PUF). arXiv:1902.04444, 2019.
9. Sutar, Aysu, Schaumont. *D-PUF: Intrinsically Reconfigurable DRAM PUF.* ACM TECS 17(1), 2017. https://dl.acm.org/doi/10.1145/3105915
10. Venugopalan et al. *FP-Rowhammer: DRAM-Based Device Fingerprinting.* AsiaCCS 2025. https://arxiv.org/abs/2307.00143
11. PreLatPUF — Najafi et al. arXiv:1808.02584, 2018.
12. EPUF — Anagnostopoulos et al. arXiv:2307.09968, 2023.
13. Wang et al. *Hertzbleed: Turning Power Side-Channel Attacks Into Remote Timing Attacks on x86.* USENIX Security 2022. https://www.hertzbleed.com/
14. Kurth et al. *NetCAT: Practical Cache Attacks from the Network.* S&P 2020. https://download.vusec.net/papers/netcat_sp20.pdf
15. *Browser-Based CPU Fingerprinting* (Schwarz et al.), ESORICS 2022. https://misc0110.net/files/uarchfp_esorics22.pdf
16. *Poster: CPU-Print — From Multiplying Matrices to Uniquely Identifying CPUs.* IEEE S&P 2025 poster. https://sp2025.ieee-security.org/downloads/posters/sp25posters-final14.pdf
17. *A methodology to identify identical single-board computers based on hardware behavior fingerprinting.* arXiv:2106.08209, 2021.
18. Viennot. `core-to-core-latency` tool. https://github.com/nviennot/core-to-core-latency
19. Lam. *Core to Core Latency Data on Large Systems.* chipsandcheese, 2023.
20. *Device Fingerprinting for Cyber-Physical Systems: A Survey.* ACM Comput. Surv., 2023. https://dl.acm.org/doi/10.1145/3584944
21. USPTO 9,197,624 — clock drift + skew + network latency for machine ID (prior art / IP).
22. USPTO 11,073,987 & 12,271,589 — SSD tail-latency identification (prior art / IP).
23. AMD Security Bulletin SB-1038 (Frequency Scaling Timing Power Side-Channels).
24. Keysight blog / Linux Journal — *AMD Zen 5 RDSEED bias*, 2025–26 (context for AMD-specific entropy caveats).

---

## TL;DR

- 2 of our 5 signals (inter-core TSC, C2C matrix) are **defensibly new-angle** on this platform.
- 1 (DRAM refresh-interval) is novel observable in a **saturated** field; presentation-dependent.
- 1 (syscall p99.9) is **partially scooped** by CPU-Print S&P'25 poster and arXiv 2106.08209.
- 1 (NVMe tail) is **academically new-angle** but **patent-encumbered** for product.
- **n=2 chassis is a desk-reject risk** at any top-tier venue. Expand to n≥6.
- Best venue: **HOST 2027** (Sep 2026 deadline) or **IEEE TIFS** journal.
- Plant flag on arXiv within 2 weeks with a *narrowed* title centred on Strix-Halo silicon-fabric fingerprinting (Signals 1 + 2 primary; 3 supporting; 4 + 5 demoted to "additional channels we briefly explored").
