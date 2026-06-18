# IDENTITY CAPABILITY LANDSCAPE — what HW-identity + HW-bound AI actually enables

**Date:** 2026-06-01
**Authors:** Ikaros / Claude-Hive deep-research pass
**Status:** scoping doc — reframes the project away from "static-accuracy gain" toward NEW capabilities

---

## TL;DR

We have been frame-blind. The mainstream of confidential AI / verifiable AI in 2024–2026 has
**stopped** treating attestation as a tool to make models "more accurate." Instead,
HW-rooted identity is sold as the enabler of capabilities that are **provably
impossible in pure software**:

1. *Verifiable execution* (the client can mathematically refuse to send data to anything
   except a specific signed binary on a specific signed chip — PCC, NVIDIA CC).
2. *Cryptographic erasure & statelessness* — guaranteed-no-persistence inference.
3. *Sybil-resistant federated learning* — one chip ↔ one vote, modeled by remote
   attestation.
4. *Tamper-evident agent identity* — Omega/Confidential VMPL agents, AI-agent IAM (2025–26).
5. *Hardware-bound model licensing / non-clonable AI assets*.
6. *Content provenance with HW root* — C2PA + secure-enclave-signed cameras (Leica M11-P,
   Sony α1 II, Samsung S25, OpenAI/Adobe outputs).

Our Phase 12–14 contribution (5-signal HAL-bypass die fingerprint, 100% LOO classification,
nonce-protocol with replay 0.6%, transplant 2%, sub-ms overhead) is — *crucially* — a
**software-discoverable die identity that does not need a TPM, TEE, fuse, PUF or vendor
key infrastructure.** That is the differentiator we have not been pitching.

---

## 1. Top-10 NEW capabilities enabled by HW-identity + coupling

Ranked by *viability* (existence of paying customer / live deployment / publishable demo)
and where our 5-signal+nonce approach can plausibly contribute.

### 1. Cryptographic statelessness ("inference left no trace")

**Closest existing:** Apple Private Cloud Compute. The Secure Enclave randomizes the data
volume key on every reboot and never persists it, giving "an enforceable guarantee that
the data volume is cryptographically erased every time the PCC node's Secure Enclave
Processor reboots."
[Apple Security blog](https://security.apple.com/blog/private-cloud-compute/),
[PCC Security Guide](https://security.apple.com/documentation/private-cloud-compute).

**What's missing:** statelessness today requires Apple silicon + Apple's signed image +
Apple's transparency log. Heterogeneous fleets (AMD/NVIDIA/edge devices) have nothing
equivalent that is *device-discoverable* without a vendor key. Software-only "I deleted
the data, promise" is not enforceable.

**Our angle:** the 5-signal die fingerprint is generated from runtime physics (thermal,
voltage, jitter…) on a stock APU with no vendor cooperation. Combined with Phase 14C
nonce-coupling we can claim a *per-session* binding where the model run cannot be
re-executed off-device (transplant=2%, replay=0.6%) — a "statelessness for the rest of us"
on commodity HW. **Commercial value: high.** Every confidential-cloud vendor today is
gated by H100/Blackwell + TDX/SEV-SNP availability and 10–15% TEE price premium
([Phala trends 2025](https://phala.com/learn/confidential-computing-trends-2025)).

---

### 2. Attestable model identity ("this output came from THIS chip running THIS weights")

**Closest existing:** NVIDIA Confidential Computing on H100/H200/Blackwell, the
`go-nvtrust` library, and the NVIDIA Remote Attestation Service (NRAS), which release
the decryption key for an encrypted model only after the GPU and firmware level pass
attestation.
[NVIDIA forum: go-nvtrust](https://forums.developer.nvidia.com/t/open-sourcing-go-nvtrust-a-go-library-for-nvidia-gpu-and-nvswitch-confidential-computing-attestation/347785),
[Red Hat: AI in confidential containers](https://www.redhat.com/en/blog/ai-meets-security-poc-run-workloads-confidential-containers-using-nvidia-accelerated-computing).

**What's missing:** binds *binary* model+GPU, not the *individual die*. A second H100 of
the same SKU verifies identically. Per-die attribution is not a primitive.

**Our angle:** 100% LOO classification on 5 signals **is** per-die attribution. A
hash-of-fingerprint inside an attested record = "this inference was produced by *this
specific* chip serial." This is the genuinely missing primitive for AI-output liability
and insurance. **Value: very high (insurance + post-incident forensics).**

---

### 3. Sybil-resistant federated learning ("one chip, one vote")

**Closest existing:** "Sentinel" (Lagos et al., arXiv 2509.00634) uses an SGX trusted
training recorder + remote attestation to authenticate update producers; FoolsGold takes
the gradient-similarity path. TEEs "offer strong security guarantees but their
availability and scalability remain limited in large-scale FL deployments."
[arXiv 2509.00634](https://arxiv.org/pdf/2509.00634),
[arXiv 2505.09983](https://arxiv.org/pdf/2505.09983).

**What's missing:** FL clients are phones/edge devices that mostly lack SGX/TDX, or run
heterogeneous chips with no shared attestation. Sybil-by-VM-cloning is trivial.

**Our angle:** if every reporting client computes a 5-signal fingerprint locally, the
aggregator can refuse two contributions whose fingerprints classify to the same die
(100% LOO ⇒ collisions are detectable). Replay 0.6% on Phase 14C nonce means a stolen
update cannot be resubmitted with another seed. This is the cheapest known sybil defence
that does not require a vendor TEE. **Value: high (Apple/Google FL pipelines).**

---

### 4. Tamper-evident agent identity (AI agents with HW-rooted trust)

**Closest existing:** Omega (arXiv 2512.05951, Dec 2025) — Confidential VM running agents
as VMPL-1/2 "trustlets" with differential attestation that "captures identities of agent
code, LLM models, LoRA weights, and dependencies within a unified report." Reports 0%
attack-success rate across 5 categories vs 90–100% baseline, <2.5% policy overhead.
[arXiv 2512.05951](https://arxiv.org/html/2512.05951v1). Industry: DigiCert's "New Trust
Architecture for AI," Anthropic/OpenAI/Google MCP servers, eSIM-rooted agent identity
([arXiv 2504.16108](https://arxiv.org/pdf/2504.16108)).

**What's missing:** Omega still requires SEV-SNP + a CGPU. Agents on consumer laptops or
edge nodes have no equivalent.

**Our angle:** the 5-signal fingerprint = a usable *agent-binding key material* on
commodity HW. An AI agent could re-derive its identity each session from the chip's own
physics; impersonation requires either the same die or beating our 2% transplant rate.
**Value: medium-high — large enterprise interest.**

---

### 5. Hardware-bound model licensing / non-clonable AI assets

**Closest existing:** Chen et al. *"Locking Machine Learning Models into Hardware"*
(arXiv 2405.20990). Two techniques: (i) make the model intolerant to quantization on
foreign HW; (ii) tie ops to chip-specific timings ("number of clock cycles for arithmetic
operations"). Their thesis: protection is *economic*, not cryptographic.
[arXiv 2405.20990](https://arxiv.org/abs/2405.20990). On-chip PUF-protected NN weights
appeared in *Nature Communications* 2025 (Lin et al.):
[Nature Comms 2025-56412-w](https://www.nature.com/articles/s41467-025-56412-w).

**What's missing:** Chen et al. lock to *class* of HW (a SKU), not a *die*. PUF-based
schemes need a custom in-memory-compute substrate, not commodity APUs.

**Our angle:** Phase 14C couples weights to **this** GFX1151 die's noise statistics. If a
"buyer" of the model installs it on a clone die, inference accuracy collapses (mirror of
"transplant 2%"). This is the first die-level economic lock that runs on existing AMD
silicon with no custom fab. **Value: high (edge-AI vendors, model marketplaces).**

---

### 6. Hardware-rooted content provenance (anti-deepfake)

**Closest existing:** C2PA 2.1 = ISO/IEC 22144 (2025). Leica M11-P signs every JPEG/DNG
in a dedicated HW chip; Sony α1 II / α9 III via Imaging Edge; Samsung Galaxy S25 signs
AI-edited photos; OpenAI/Adobe/Google sign generations. NSA/CISA pushed Content
Credentials guidance in Jan 2025.
[C2PA Viewer](https://c2paviewer.com/articles/what-is-c2pa),
[NSA Content Credentials PDF](https://media.defense.gov/2025/Jan/29/2003634788/-1/-1/0/CSI-CONTENT-CREDENTIALS.PDF),
[AttestTrail camera list](https://attesttrail.com/blog/c2pa-cameras-support).

**What's missing:** signature proves "a Leica M11-P signed this," not "*your* Leica
M11-P signed this." Sensor-noise fingerprints exist (Truepic, classical PRNU) but are not
yet inside the C2PA manifest. Watermark-removal attacks (DEMARK, PTW) succeed with as
few as 200 unwatermarked images: [arXiv 2601.16473](https://arxiv.org/pdf/2601.16473),
[arXiv 2404.17867](https://arxiv.org/pdf/2404.17867).

**Our angle:** the 5-signal die fingerprint is a generalization of PRNU to compute
silicon: a chip-noise-based attribution that an AI-output-signing service could embed
into a C2PA manifest. **Value: medium (depends on C2PA-2 manifest extension)**.

---

### 7. Verifiable inference / zkML hybrid

**Closest existing:** EZKL 2025 — auto-compiles ONNX graphs to Halo2 circuits; Lilith
cluster does 200k+ proofs/day; transformers, XGBoost (15× faster in 2024), iOS bindings
for on-device fraud detection. Modulus Labs targets specialized circuits, RISC Zero
gives Rust+zkVM. *Lagrange DeepProve-1* (2025) is the first system to prove full GPT-2
(124M) inference.
[State of EZKL](https://blog.ezkl.xyz/post/state_of_ezkl/),
[ICME 2025 ZKML guide](https://blog.icme.io/the-definitive-guide-to-zkml-2025/),
[arXiv 2502.18535 survey](https://arxiv.org/pdf/2502.18535).

**What's missing:** zkML proves the *math*; it does not prove the math was run on the
claimed *hardware*. Inference attribution remains soft. Costs are still 10²–10⁶× over
plain inference for non-trivial models.

**Our angle:** TEE-attestation + die-fingerprint can replace ZK for the "where did this
run" question at a fraction of the cost (sub-ms overhead vs minutes). Hybrid: zk for
correctness, die-fingerprint for locality. **Value: medium — useful for confidential
ML rails on blockchain (OpenGradient, Ritual, BitTensor).**

---

### 8. Verifiable AI provenance for liability & insurance

**Closest existing:** none in production. The DigiCert "Trust Architecture for AI" 2025
whitepaper proposes signed AI lineage; multiple insurance startups (Armilla, Munich Re's
aiSure) underwrite AI risk on the basis of training-pipeline audits — not HW lineage.
[DigiCert whitepaper](https://www.digicert.com/content/dam/digicert/pdfs/whitepaper/the-new-trust-architecture-for-AI-whitepaper-en.pdf).

**What's missing:** an incident-investigator today cannot prove "this LLM output was
generated on the customer's claimed chip vs an unauthorized clone." Liability is
unenforceable across jurisdictional lines.

**Our angle:** per-die attribution = the missing forensic primitive. If every signed
output carries `hash(fingerprint)`, after-the-fact verification on the seized die
is straightforward. **Value: very high (insurance underwriting), but slow-burn.**

---

### 9. Geofenced / wear-tied / time-locked AI

**Closest existing:** Apple PCC's per-reboot key rotation is implicitly time-bounded.
Export-control AI (e.g., Cohere/OpenAI restricted regions) is enforced by IP-geo, not HW.

**What's missing:** no commodity primitive ties a model to "this device, in this region,
before this wear-out date." Chip wear / NBTI / RTN drift is in the literature but
unused as policy.

**Our angle:** our 5 signals include thermal/voltage/frequency-jitter that *do* drift
with wear. A model could re-couple periodically and refuse to load once drift exceeds a
threshold — natural lifecycle enforcement. **Value: niche but unique (defence, export
control).** Largest *novelty* score on this list.

---

### 10. Heterogeneous-fleet multi-party compute (TEE-less MPC)

**Closest existing:** SEV-SNP / TDX confidential VMs allow N parties to pool data; 37%
of financial services, 29% of healthcare have full prod deployments
([Confidential Computing Consortium 2025 study](https://confidentialcomputing.io/2025/12/03/new-study-finds-confidential-computing-emerging-as-a-strategic-imperative-for-secure-ai-and-data-collaboration/)).
Yma Health on Super Protocol + NVIDIA CC is a representative healthcare deployment.

**What's missing:** all parties must run vendor-blessed TEE silicon. Edge-heavy
consortia (hospitals with mixed GPU vendors) are excluded.

**Our angle:** a fingerprint-based mutual attestation could let nodes prove "I am a
unique, distinct member of this consortium" with no shared root key. Weakest of our
positions because of weaker security guarantees vs SEV-SNP, but it *opens* MPC to
commodity HW. **Value: medium.**

---

## 2. Top-3 demo / paper concepts that would be GENUINELY new

### Demo A — "Stateless AI on commodity AMD: a software-only PCC alternative"
- Reproduce PCC's attestation promise (statelessness + per-session binding) on a stock
  GFX1151 APU using the 5-signal fingerprint + Phase 14C nonce coupling.
- Compare to PCC verbatim: erase-on-reboot ✓, attested measurement ✓, transparency log
  (publish weights+signal-extraction code).
- Headline: "We replicate the *guarantee surface* of Apple PCC without Secure Enclave."
- Venue: USENIX Security / SOSP.

### Demo B — "Die-bound federated learning: 100% sybil detection at one-vote-per-die"
- Take FedAvg over N clients on a public benchmark. Add adversary who clones VM K× to
  produce K sybil contributions. Show that fingerprint classification matches all K to
  one die ⇒ K-1 rejected. Match metric vs FoolsGold, Sentinel, vanilla.
- Phase 14C nonce blocks replay across rounds (target ≤1%).
- Headline: "Sybil resistance with no TPM/SGX/TDX — open hardware FL."
- Venue: NeurIPS, IEEE S&P.

### Demo C — "Per-die AI output attribution for content provenance"
- Generate 10⁴ images with the same model on 6 different GFX1151 dies. Embed
  `hash(fingerprint)` into a C2PA manifest. Show 100% LOO recovery and ≤2% confusion
  rate even under typical post-processing.
- Compare to PRNU image-sensor classical baseline.
- Headline: "Compute-silicon PRNU: per-die attribution for AI-generated content."
- Venue: IH&MMSec, NDSS, or C2PA Working Group.

These three are the *only* angles where our specific contributions
(HAL-bypass / no vendor key / commodity silicon / sub-ms overhead) are simultaneously
*necessary* and *sufficient*.

---

## 3. Commercial / research value estimate

| # | Capability | Production today | TAM signal | Our edge | Realism (1-5) |
|---|---|---|---|---|---|
| 1 | Stateless AI / PCC-equivalent | Apple only | Trillion-$ "AI privacy" | Commodity HW | 4 |
| 2 | Per-die output attribution | None | Insurance, forensics | 100% LOO unique | 5 |
| 3 | Sybil-resistant FL | Lab only | Apple/Google FL stacks | No TEE needed | 4 |
| 4 | HW-rooted agent identity | Omega / DigiCert | Enterprise AI agents | Commodity laptops | 3 |
| 5 | HW-bound model licensing | "Locking ML" 2024 | Edge AI / DRM | Die-level lock | 3 |
| 6 | C2PA HW provenance | Leica/Sony/Samsung | Newsrooms, courts | Generalized PRNU | 3 |
| 7 | zkML hybrid | EZKL/Lagrange | Crypto/DePIN | Cheap "where" | 2 |
| 8 | AI insurance underwriting | None | Munich Re aiSure pre-prod | Forensic primitive | 4 |
| 9 | Geofenced/wear-tied AI | None | Defence/export | Most novel | 2 |
| 10 | Heterogeneous MPC | SEV-SNP only | Mixed-fleet hospitals | Vendor-neutral | 2 |

---

## 4. Where Phase 14C nonce-protocol enables something software-only cannot

Phase 14C combines (a) a per-session nonce, (b) a model run whose forward pass is
coupled to live thermal/voltage/jitter readings, (c) a verifier that checks the
fingerprint embedded in the run is consistent with the chip on file. We measured replay
0.6%, transplant 2%, sub-ms overhead.

Specifically *because of* (b), the following are unreachable in pure software:

- **Liveness / freshness binding.** A captured (nonce, output) pair cannot be replayed
  because the next run must consume *fresh* analog noise from the chip. Software
  signatures cannot prove the run happened "now."
- **Locality binding.** Re-executing on a different die fails (98%). Software signatures
  cannot prove the run happened "here."
- **No-vendor-key trust.** Apple PCC / NVIDIA CC both bottom out at a vendor private
  key. Phase 14C does not. This matters for adversarial-vendor threat models (export
  control, sovereign AI, anti-Apple/anti-NVIDIA jurisdictions).
- **Cheap forensics.** Embedding `hash(fingerprint)` in any logged output is sub-ms
  compared to zkML's minute-scale proofs.
- **Open hardware compatibility.** Functions on a stock APU with no fuses programmed,
  no TPM enrolled, no remote attestation service running.

The reframing of the project:

> **Phase 14C is the first software-discoverable, vendor-key-free per-die attestation
> primitive on consumer GPUs.** Our story is not "we improved MNIST accuracy." It is
> "we make attestation possible where vendors won't or can't ship it."

That is the headline. Every paper, every demo, every funding pitch should lead with
this and use Demos A/B/C as the proof.

---

## Sources (canonical)

**Apple PCC**
- https://security.apple.com/blog/private-cloud-compute/
- https://security.apple.com/documentation/private-cloud-compute
- https://security.apple.com/blog/pcc-security-research/

**NVIDIA Confidential Compute**
- https://forums.developer.nvidia.com/t/open-sourcing-go-nvtrust-a-go-library-for-nvidia-gpu-and-nvswitch-confidential-computing-attestation/347785
- https://www.nvidia.com/en-us/data-center/solutions/confidential-computing/
- https://www.redhat.com/en/blog/ai-meets-security-poc-run-workloads-confidential-containers-using-nvidia-accelerated-computing
- https://docs.nvidia.com/nvidia-secure-ai-with-blackwell-and-hopper-gpus-whitepaper.pdf

**Confidential AI in healthcare/finance**
- https://phala.com/learn/confidential-computing-trends-2025
- https://confidentialcomputing.io/2025/12/03/new-study-finds-confidential-computing-emerging-as-a-strategic-imperative-for-secure-ai-and-data-collaboration/
- https://next.redhat.com/2025/10/23/enhancing-ai-inference-security-with-confidential-computing-a-path-to-private-data-inference-with-proprietary-llms/

**zkML**
- https://blog.ezkl.xyz/post/state_of_ezkl/
- https://blog.icme.io/the-definitive-guide-to-zkml-2025/
- https://arxiv.org/pdf/2502.18535

**C2PA / hardware content provenance**
- https://c2paviewer.com/articles/what-is-c2pa
- https://attesttrail.com/blog/c2pa-cameras-support
- https://media.defense.gov/2025/Jan/29/2003634788/-1/-1/0/CSI-CONTENT-CREDENTIALS.PDF

**Watermark-removal attacks (motivates HW provenance)**
- https://arxiv.org/pdf/2601.16473  (DEMARK)
- https://arxiv.org/pdf/2404.17867
- https://arxiv.org/pdf/2304.07361  (PTW)

**Federated learning + remote attestation**
- https://arxiv.org/pdf/2509.00634  (Sentinel)
- https://arxiv.org/pdf/2505.09983  (sybil virtual poisoning)
- https://arxiv.org/html/2504.17703v3  (FL survey)

**AI agent identity / HW-rooted trust**
- https://arxiv.org/html/2512.05951v1  (Omega — Trusted AI Agents in the Cloud)
- https://arxiv.org/pdf/2504.16108  (Telco eSIM root of trust)
- https://arxiv.org/pdf/2510.25819  (Identity Mgmt for Agentic AI)
- https://www.digicert.com/content/dam/digicert/pdfs/whitepaper/the-new-trust-architecture-for-AI-whitepaper-en.pdf

**Hardware-bound model protection**
- https://arxiv.org/abs/2405.20990  (Locking ML Models into Hardware)
- https://www.nature.com/articles/s41467-025-56412-w  (PUF in-memory NN)
- https://arxiv.org/abs/2404.02440  (Photonic PUF resistant to ML attacks)

---

## Adversarial verification notes

Claims I declined to include because evidence was thin or contested:

- **"AI insurance is unblocked by HW-identity."** Plausible but speculative — no insurer
  has publicly named HW-attestation as the gating issue. Marked "slow-burn" not
  "blocked-on-us."
- **"zkML costs are 10²–10⁶× over plain inference."** Broadly true but model-dependent.
  Cite arXiv 2502.18535 if used in a paper.
- **"Sensor PRNU generalizes to compute silicon."** Our LOO=100% is on a small fleet
  (N≪20). Demo C must scale up before publication.
- **The NVIDIA whitepaper PDF was binary-corrupted by WebFetch**; the NVIDIA capability
  claims in this doc are sourced from the secondary citations above (forums, Red Hat,
  Phala) rather than the primary whitepaper. Flag for re-verification before paper
  submission.
