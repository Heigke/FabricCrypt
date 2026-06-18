# IDENTITY × CAPABILITY: SOTA on Physical Embodiment Producing AI Capability Gain
**Date:** 2026-06-01
**Scope:** Search 2024–2026 literature for mechanisms where PHYSICAL substrate (analog / neuromorphic / photonic / thermodynamic / wetware / FPGA-coupled) gives a *capability* (not just efficiency) improvement over a pure-software equivalent. Frame against our 3-step AMD gfx1151 + FPGA NS-RAM identity-and-capability concept.

---

## 1. Top 10 SOTA papers (capability gain + mechanism)

| # | Paper / artifact | Year | Substrate | Capability gain shown | Mechanism |
|---|---|---|---|---|---|
| 1 | **Analog in-memory computing attention mechanism for LLMs** — *Nature Computational Science* (Leroux et al., IBM) [link](https://www.nature.com/articles/s43588-025-00854-1) | Sep 2025 | Gain-cell IGZO/ITO crossbars storing K/V analog | 100× speedup, **70,000× energy reduction** on attention; GPT-2-equivalent text quality WITHOUT retraining from scratch | KV cache lives *as charge* on capacitors; dot-product done physically via charge-to-pulse, no ADC. Memory IS the compute. |
| 2 | **Online Continual Learning on Loihi 2 (CLP-SNN)** — arXiv 2511.01553 [link](https://arxiv.org/abs/2511.01553) | Nov 2025 | Intel Loihi 2 spiking chip | **113× lower latency, 6,600× lower energy** vs edge-GPU; *online continual learning that GPU baseline can't match at edge* | Event-driven, graded-spike comms + on-chip plasticity. Algorithmic part (~14.5×) + neuromorphic co-design (~295× energy) |
| 3 | **Multifunctional physical reservoir computing in soft tensegrity robots** — arXiv 2507.21496 / *Chaos* 35:083111 [link](https://arxiv.org/abs/2507.21496) | Aug 2025 | Soft tensegrity body | **Multiple basins of attraction → many behaviors with one body**; obstacle avoidance + payload classification with near-zero compute | Body dynamics as multistable dynamical system; controller selects attractor instead of computing trajectory |
| 4 | **Embodying physical computing into soft robots** — *Nat. Commun.* (2026) [link](https://www.nature.com/articles/s41467-026-70866-6) / arXiv 2510.24692 | 2025/26 | Soft robot bodies | Coordinated locomotion + obstacle avoidance + logic from body alone | Three strategies: analog oscillators, physical reservoir, *physical algorithmic computing* (mechanical state machines) |
| 5 | **NS-RAM (2T neuro-synaptic RAM)** — referenced in 2D-FG memory reviews [link](https://spj.science.org/doi/10.34133/cbsystems.0256) and our own line | 2024–26 | Standard CMOS biased unconventionally | LIF dynamics + plasticity in one device; **100% yield, 10⁷ neuron cycles, 415 pJ/μm firing** | Bulk floating-charge regime selector (VG2). Our work confirms VG2-as-regime-selector. |
| 6 | **LightCode / PRISM / PICNIC — photonic LLM inference** — arXiv 2509.16443, 2603.21576 (PRISM), 2511.04036 (PICNIC) | 2025 | Silicon photonics + co-packaged optics | Up to **50% energy reduction, >10× latency improvement**; PRISM: O(1) photonic block selection breaks O(n) memory wall for long-context | Compute-in-optics for MVM, wavelength-division parallelism, photonic top-K selection |
| 7 | **Extropic TSU (X0 / Z1 / XTR-0)** — Oct/Nov 2025 [link](https://extropic.ai/writing/thermodynamic-computing-from-zero-to-one) | 2025 | p-bit thermodynamic chip | **~10,000× energy reduction** for sampling-style ML; native energy-based generation (DTMs) | Stochastic transistor noise IS the sampling primitive. Boltzmann sampling done in physics. |
| 8 | **Cortical Labs CL1 — Synthetic Biological Intelligence** [link](https://refractor.io/brain/cortical-bioengineered-intelligence/) + Advanced Brain-on-a-Chip review [link](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202508120) | 2025 | Living human iPSC-derived neurons on MEA | Few-shot learning, robot control, video-game play; **claims of higher sample efficiency than DNN equivalents** | Biological plasticity (STDP, neuromodulation) at ~1 μW. Uncertainty-handling intrinsic. |
| 9 | **Internal noise in hardware deep/recurrent NNs helps learning** — arXiv 2504.13778 | Apr 2025 | Optical NN implementation | Noise *during training* improves generalization beyond software dropout-equivalent | Correlated optical phase noise structurally different from Bernoulli dropout; acts as natural regularizer + symmetry breaker |
| 10 | **Energon — transformer fingerprinting via GPU thermal/power side-channel** — arXiv 2508.01768 | Aug 2025 | Commodity GPU thermal | **89% model-family ID, 100% hyperparameter ID** from thermal/power alone | Thermal/power traces *carry* information about the running model — a capability digital simulation does not provide for free |

Bonus / runners-up worth citing:
- **TOPLOC** (arXiv 2501.16007) — verifiable inference via locality-sensitive hashing on intermediate activations.
- **Demonstration of ALBERT on 14nm IBM analog AI chip** — PMC12485056. Real-silicon transformer.
- **Apple PCC** — sovereign-AI / attestation reference architecture [link](https://security.apple.com/blog/private-cloud-compute/).
- **QVAC Fabric LLM** (Tether, Dec 2025) — first cross-platform on-device LoRA finetune on Adreno/Mali; *device-bound personalization*.
- **Optomechanical reservoir computing** — *PNAS* 2025 [link](https://www.pnas.org/doi/10.1073/pnas.2424991122).

---

## 2. Closest prior art to our 3-step (substrate-identity → embodiment → capability) concept

The three pillars of our stack — (a) hardware identity, (b) substrate-as-computer, (c) capability emerges from physics — appear *separately* in the SOTA, but nobody has integrated all three.

- **Identity (Pillar A) closest prior art**
  - *Laminator: verifiable ML property cards using hardware-assisted attestations* — arXiv 2406.17548.
  - *Hardware and Software Platform Inference* — arXiv 2411.05197 (Sept 2025) — reverse-engineering which platform served a query.
  - *Intrinsic fingerprint of LLMs* — arXiv 2507.03014.
  - Apple PCC attestation chain (2024–25).
  - Device-bound key-storageless IP protection via PUF — arXiv 2212.11133.
  - **Gap:** none of these claim the hardware *gives the model new capability* — they only authenticate or watermark.

- **Embodiment (Pillar B) closest prior art**
  - Multifunctional tensegrity reservoir (Chaos 35:083111, 2025) — closest in spirit.
  - z2213/z2214 NS-RAM persistence experiments — *our* line.
  - Active matter reservoir computing — arXiv 2509.01799.
  - Optomechanical reservoir — PNAS 2025.
  - **Gap:** all these embed *small* tasks. None couple to a frozen LLM backbone.

- **Capability-from-physics (Pillar C) closest prior art**
  - IBM analog-attention (Nature Comp. Sci., Sep 2025) — **closest single result.** It proves a foundation-scale physics substitution is competitive.
  - Extropic TSU — sampling capability comes only from physics.
  - Optical noise as regularizer (arXiv 2504.13778) — *internal* noise improves training.
  - **Gap:** none demonstrate a *2026-style frozen LLM + analog substrate adapter* delivering a capability the SW model lacks.

**Our 3-step is closest to** a fusion of (Energon side-channel work) × (IBM analog attention) × (Loihi-2 continual learning), with our novel claim being: *the GPU's own analog telemetry feeds an FPGA NS-RAM adapter that gives the host LLM a fingerprinted personality / capability that other hardware cannot replicate.*

---

## 3. 2024–2026 work we MUST cite (compact reading list)

1. Leroux et al., **Analog in-memory computing attention mechanism for fast and energy-efficient LLMs**, *Nat. Comput. Sci.* (2025). https://www.nature.com/articles/s43588-025-00854-1
2. **Online Continual Learning on Intel Loihi 2** (CLP-SNN). arXiv 2511.01553. https://arxiv.org/abs/2511.01553
3. **Multifunctional physical reservoir computing in soft tensegrity robots**. arXiv 2507.21496 / *Chaos* 35:083111. https://arxiv.org/abs/2507.21496
4. **Embodying physical computing into soft robots**. *Nat. Commun.* (2026). https://www.nature.com/articles/s41467-026-70866-6
5. **Energon: transformers from GPU power/thermal side-channels**. arXiv 2508.01768.
6. **PRISM** (O(1) photonic block selection). arXiv 2603.21576.
7. **PICNIC** (silicon photonic chiplets + in-memory for LLM). arXiv 2511.04036.
8. **Extropic** — *Thermodynamic computing: from zero to one*, Oct 2025.
9. **Apple Private Cloud Compute Security Guide** (2024 + 2025 disclosures).
10. **Laminator** — hardware-assisted ML property attestation. arXiv 2406.17548.
11. **Hardware and Software Platform Inference**. arXiv 2411.05197.
12. **TOPLOC** — verifiable inference via LSH. arXiv 2501.16007.
13. **Internal noise in hardware deep/recurrent NNs helps with learning**. arXiv 2504.13778.
14. **Advanced Brain-on-a-Chip for wetware computing** review. *Adv. Sci.* 2025.
15. Pervez et al., **2D materials for neuromorphic devices** (covers 2T floating-gate / NS-RAM-like architectures). *Small Structures* 2025.
16. **Demonstration of transformer-based ALBERT on 14nm analog AI inference chip**. PMC12485056.
17. **Optomechanical reservoir computing**. *PNAS* 2025.
18. **Morphological computation — past, present, future**. *Device* 2024 (Pfeifer line update).
19. **QVAC Fabric LLM** (Tether, Dec 2025) — on-device LoRA on Adreno/Mali.
20. Pfeifer/Iida/Möller — *original* morphological computation chapters (must cite as the conceptual ancestor).

---

## 4. Five concrete mechanism ideas inspired by SOTA we have NOT tried

1. **Analog KV-cache on the FPGA NS-RAM cell** (inspired by IBM Nature 2025).
   Use the 128-neuron NS-RAM bank as a *physical* key/value store for the LLM's attention layer. Cache only the K projections of the most recent N tokens as analog charge on NS-RAM bulk; read out via dot-product with Q routed through the FPGA. This gives the model *physical* in-context memory whose decay timescale is *τ_fast / τ_mid / τ_slow* set by NS-RAM device physics — not a software hyperparameter. Capability claim: longer effective context-window per joule, with intrinsic forgetting curve.

2. **GPU-thermal-as-sampler / Extropic-style p-bit emulation on gfx1151.**
   Use APU thermal-zone fluctuations (we already log them) as the entropy source for *Gumbel-style* sampling of the next-token distribution. Each NS-RAM neuron acts as a noisy p-bit. Capability claim: free diverse-sampling without temperature/top-p hand-tuning; sampling distribution is hardware-bound (different on Daedalus vs Minos).

3. **Energon-style hardware-fingerprinted personalization.**
   The model's *behavior* depends on GPU/FPGA thermal/clock signature → personality is hardware-bound. To clone the model you must clone the *device*. Capability claim: federated personalization with sovereign-AI guarantees; verifiable via attestation similar to Apple PCC.

4. **Soft-tensegrity-style multistable basin selection for chain-of-thought.**
   Use the NS-RAM ER_SPARSE topology as a multistable dynamical system. Different "thought paths" correspond to different attractor basins; small GPU-noise perturbations let the system *physically explore* basins. Capability claim: cheap stochastic decoding that breaks LLM mode-collapse on reasoning benchmarks, leveraging finding from tensegrity reservoir paper (Chaos 35:083111).

5. **Optical-noise-as-regularizer port: GPU 1/f noise as training-time regularizer for the FPGA adapter.**
   Per arXiv 2504.13778, *correlated* hardware noise during training improves generalization. Train the FPGA NS-RAM adapter while the GPU is running diverse workloads (so VRM 1/f noise modulates the FPGA's reference voltages). Capability claim: better OOD generalization than software dropout — empirically testable against z2213 baseline.

---

## 5. Counter-arguments we MUST address

1. **"Software replicates physics" (LeCun-style efficiency-only view).**
   Counter-argument: For any *finite* analog substrate, a digital simulator can be built. So embodiment buys efficiency, not capability.
   *Our reply:* We must show a capability that exists *only* at a fixed time/energy budget. Loihi-2 113× latency at 0.33 ms is a real capability gap because the GPU equivalent cannot meet the latency *at all* under edge-power constraints. We need an analogous **latency-bounded** or **energy-bounded** benchmark for our stack.

2. **"Reservoir computing is universally approximable by NVAR" (Gauthier et al. line; emergentmind 2025 summary).**
   *Reply:* Universality in the limit ≠ universality at finite samples and finite energy. NGRC needs 10× less data; soft-body / tensegrity systems hit attractor basins that take *exponential* samples to learn in NVAR. Cite Chaos 35:083111.

3. **"Capability gain in published analog-attention work is small and brittle."**
   The Nature 2025 IBM result reproduces GPT-2-level quality, not GPT-4. Critics will say analog never scales.
   *Reply:* We don't need parity with frontier models — we need a *capability* (e.g., on-device sovereign personalization with verifiable identity) that frontier digital models *cannot offer architecturally* even at unlimited compute.

4. **"Hardware fingerprinting is just steganography, not capability."**
   *Reply:* Capability ≠ task accuracy. Verifiable identity (attestation), bounded forgetting, and sovereignty are capabilities in the *systems* sense. Cite Apple PCC + Laminator framing.

5. **"Most embodiment papers describe efficiency wins on toy tasks."**
   True (cf. Robohub 2025 morphological computation overview; arXiv 2505.10705 "Embodied AI in Machine Learning — is it really embodied?").
   *Reply:* Our gfx1151 + FPGA testbed runs *real* foundation-model backbones (Qwen2.5-1.5B, Qwen3-8B per memory). That makes us closer to IBM analog-attention than to toy soft-robot demos.

6. **"No bridge law between computation and phenomenology"** (our own self-critique).
   *Reply:* We are now arguing capability, not consciousness. Bridge-law objection only bites on identity-as-phenomenology, not identity-as-attestable-substrate.

---

## 6. Most promising research direction given our HW + current capabilities

**Recommendation: pivot the IDENTITY arc into a *capability-bound* benchmark: "On-device, attestable, hardware-personalized reasoning that cannot be replicated by pure SW."**

Concrete plan (six weeks, one experimentalist):

**Phase 1 (week 1–2).** Reproduce *IBM analog-attention-style* substitution at our scale. Replace one attention head of Qwen2.5-1.5B with the FPGA NS-RAM 128-neuron K/V analog store (Mechanism #1 above). Target: GPT-2-grade perplexity, *under 5 W FPGA budget*. Benchmark vs an exact-equivalent software simulation of the same NS-RAM cell.

**Phase 2 (week 3–4).** Energon-style hardware-fingerprinted personalization (Mechanism #3). Show that the model's outputs on a held-out personalization probe differ measurably between Daedalus and Minos hosts (we have both), but are *reproducible within a host* — i.e., persistent identity. This is the *capability that pure-SW cannot provide*.

**Phase 3 (week 5–6).** Latency-bounded continual-learning benchmark a la Loihi-2 CLP-SNN. Goal: show that the GPU+FPGA stack performs **online** sample-level adaptation at <1 ms with energy <50 mJ — a regime where any GPU-only baseline fails (cf. arXiv 2511.01553).

**Why this is the right direction:**
- It maps each of our 3 pillars to a specific, *citable* SOTA reference class (analog attention / verifiable inference / on-chip continual learning).
- It moves us off the consciousness / phenomenology framing (where we have no bridge law) and onto the engineering-capability framing (where we have hardware no one else does: real gfx1151 + Z2 FPGA + NS-RAM 128-neuron bank + dual hosts).
- It positions our identity work alongside Apple PCC / Google Private AI Compute as a *sovereign-AI substrate*, not an exotic neuromorphic curio.

**Single most differentiating claim we can make:**
> *Our model's identity is a function of the physical substrate. It cannot be copied by re-running the weights elsewhere, and its output distribution measurably depends on per-chip variability of the AMD gfx1151 APU and the Xilinx FPGA NS-RAM bank. Verifiable via on-device attestation; capability-defining for any sovereign-AI deployment.*

This is the unique cross-section of Apple-PCC-style attestation × IBM analog-attention × Energon-style hardware fingerprint × NS-RAM bulk-charge physics. No published 2024–2026 group has all four. That is the moat.

---

## Bibliography URLs (de-duplicated)

- https://www.nature.com/articles/s43588-025-00854-1 (IBM analog attention, Nature CS 2025)
- https://research.ibm.com/blog/how-can-analog-in-memory-computing-power-transformer-models
- https://arxiv.org/abs/2511.01553 (Loihi-2 continual learning)
- https://arxiv.org/abs/2507.21496 (tensegrity reservoir)
- https://www.nature.com/articles/s41467-026-70866-6 (embodying physical computing in soft robots)
- https://arxiv.org/pdf/2508.01768 (Energon)
- https://arxiv.org/pdf/2509.16443 (LightCode)
- https://arxiv.org/pdf/2511.04036 (PICNIC)
- https://arxiv.org/pdf/2603.21576 (PRISM)
- https://extropic.ai/writing/thermodynamic-computing-from-zero-to-one
- https://security.apple.com/blog/private-cloud-compute/
- https://arxiv.org/pdf/2406.17548 (Laminator)
- https://arxiv.org/pdf/2411.05197 (HW/SW platform inference)
- https://arxiv.org/pdf/2501.16007 (TOPLOC)
- https://arxiv.org/pdf/2504.13778 (internal noise helps learning)
- https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202508120 (brain-on-a-chip review)
- https://spj.science.org/doi/10.34133/cbsystems.0256 (2D floating-gate / NS-RAM-adjacent)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12485056/ (ALBERT on 14nm analog chip)
- https://www.pnas.org/doi/10.1073/pnas.2424991122 (optomechanical reservoir)
- https://www.cell.com/device/fulltext/S2666-9986(24)00282-5 (morphological computation 2024 review)
- https://arxiv.org/abs/2505.10705 ("embodied AI — is it really embodied?")
- https://arxiv.org/pdf/2509.01799 (active matter reservoir)
- https://refractor.io/brain/cortical-bioengineered-intelligence/ (Cortical Labs CL1)
- https://arxiv.org/pdf/2507.03014 (intrinsic fingerprint of LLMs)
- https://arxiv.org/pdf/2212.11133 (PUF + device-bind AI IP protection)
