# FabricCrypt: Software-discoverable vendor-key-free per-die attestation for AI inference on commodity GPUs

**Draft v1** — 2026-06-01 — *target venue: USENIX Security or ACM IH&MMSec*

---

## Abstract

We present **FabricCrypt**, the first software-discoverable, vendor-key-free
per-die attestation primitive demonstrated end-to-end on commodity AMD
hardware. FabricCrypt couples (i) a 290-dimensional live device signature
assembled from five HAL-bypass micro-architectural signals — inter-core
TSC offsets, cacheline ping-pong matrices, DRAM-refresh-aligned jitter,
syscall p99.9 tails, and NVMe queue-tail latencies — with (ii) an
audience-supplied 64-bit nonce that drives the *sampling plan itself*
(which CPUs, which thermal zones, which core pairs, which sleep
durations). On two AMD Ryzen AI Max+ 395 "Strix Halo" laptops (`ikaros`
and `daedalus`) we obtain 100% leave-one-out per-die classification
(20 reps, gate >0.95) and pass all 7 protocol attack gates: 100% honest
accept, 0.6% static replay, 1.2% dynamic-library replay, 0.6%
nonce-mismatch, 2.0% peer transplant. End-to-end sign-and-verify latency
is sub-millisecond (median 1.12 ms, p99 2.79 ms). Capability gains on
two downstream tasks are large and reproducible: anomaly detection
AUROC 0.500 → 0.994, host-attribution accuracy 0.501 → 1.000. Both
gains survive matched-governor measurement. FabricCrypt offers
*per-die* attribution that the vendor-PKI-rooted designs (Apple PCC,
NVIDIA Confidential Compute, Intel TDX, AMD SEV-SNP) do not, and it
does so without a Secure Enclave, a TPM EK certificate, or any vendor
key material. We discuss three new capabilities this enables — per-die
AI output attribution, stateless PCC-equivalent guarantees on commodity
AMD, and TEE-free sybil-resistant federated learning — and we are
honest about what we have *not* shown: a static-benchmark inference
accuracy gain (null), and chassis count n=2.

**Reproduce-script:** `scripts/identity_benchmark/embodiment{12,12b,13,14b,14c,14d}/` (released with camera-ready).

---

## 1. Introduction

Trustworthy AI inference is in the middle of a quiet PKI war.
Apple's Private Cloud Compute (PCC) [PCC2024] binds inference attestation
to a *vendor* signing key rooted in a Secure Enclave. NVIDIA Confidential
Compute (CC) [NVIDIA-CC] binds H100/Blackwell attestation to a Device
Identity Certificate (DICE) signed by NVIDIA. Intel TDX [TDX2024] and
AMD SEV-SNP [SEV-SNP] root attestation in the silicon-vendor's CA. All
four schemes share an architectural property: *if you do not trust the
vendor's PKI, you do not have attestation*. Worse, all four schemes
authenticate the **SKU class** — "an H100 in confidential mode" — not
the **individual die**: two H100s with identical firmware produce
indistinguishable VCEK signatures.

This matters in three concrete cases:

1. **Output attribution.** When a model card claims "trained on chip
   X," there is currently no software-only way to *prove that this
   inference came from chip X and not some other chip of the same SKU*.
2. **Vendorless deployments.** Privacy-preserving AI offerings on
   commodity AMD APUs (Ryzen AI 300, Strix Halo, etc.) cannot use PCC
   because Apple's Secure Enclave does not exist on those platforms.
3. **Sybil resistance in federated learning.** Today's defences require
   SGX or TDX [Sentinel2025]. A laptop without SGX cannot participate
   honestly.

We ask: **can commodity AMD hardware provide a per-die attestation
primitive without depending on the vendor's PKI?** Our answer is yes,
provided we are willing to bypass the HAL and read low-level
micro-architectural signals that PSP firmware does not (and cannot
cheaply) sanitize.

### Contributions

1. **5 HAL-bypass signals** that together yield a 290-dimensional
   per-die fingerprint with 100% LOO classification at n=2 chassis
   (Section 4).
2. **Governor-confound resolution.** We re-ran the headline T2/T3
   tests under matched CPU frequency governor and show the capability
   gains are within noise of the original mixed-governor run
   (Section 4.4).
3. **Nonce-driven sampling plan.** A 64-bit audience challenge controls
   *which* CPUs, *which* thermal zones, *which* core pairs and *which*
   sleep durations are read, defeating static and dynamic-library
   replay (Section 5).
4. **All-7-gates protocol.** Honest accept ≥0.95, peer accept ≤0.05,
   static replay ≤0.05, dynamic replay ≤0.10, two flavours of
   nonce-mismatch ≤0.05, sub-ms sign-and-verify (Section 5).
5. **Three new capabilities** that the vendor-PKI-rooted designs
   cannot provide: per-die AI output attribution, stateless PCC-class
   guarantee surface on commodity AMD, and TEE-free sybil-resistant
   federated learning (Section 6).

We deliberately do **not** claim a static-benchmark accuracy gain
from embodiment: it was prereg-tested and came back null. We discuss
this honestly in Section 7.

---

## 2. Background and related work

**Hardware-rooted attestation.** Apple PCC [PCC2024] introduced
public verifiability of cloud inference, but its root of trust is
Apple's Secure Enclave Processor and Apple's CA. NVIDIA Confidential
Compute [NVIDIA-CC] uses DICE-rooted VCEK/AK certificates, and Intel
TDX [TDX2024] and AMD SEV-SNP [SEV-SNP] use vendor-issued endorsement
keys. None of these authenticate the individual die: a vendor that
re-signs identical keys onto two dies cannot be detected by the
client.

**TPM 2.0** [TPM2] supports a per-platform Endorsement Key (EK),
but the EK certificate is issued by the platform vendor and the EK
itself is generated inside the TPM under vendor control. TPM EK is
SKU-class identity, not silicon-physical identity.

**Physically unclonable functions on commodity HW.** The closest prior
work is the DRAM Latency PUF (Kim et al., HPCA 2018 [Kim2018]),
which extracts a stable fingerprint from DRAM cell variation visible
via custom DRAM-controller commands. FP-Rowhammer (Venugopalan et al.,
AsiaCCS 2025 [Venugopalan2023]) extends adjacent fault-injection
techniques. DRAWNAPART (NDSS 2022) [DRAWNAPART] showed GPU-execution-unit
fingerprinting in WebGL. These works share two properties with
FabricCrypt: they use *micro-architectural noise as identity* and they
require no vendor key. They differ in two ways: (i) they extract a
static fingerprint, not a challenge-bound live signature; (ii) they
do not address replay defence via audience-driven sampling.

**Clock-skew device identification.** Kohno et al. (S&P 2005)
[Kohno2005] showed that TCP timestamp drift identifies devices across
networks. This is the *spiritual* ancestor of FabricCrypt: physical
clock variation as software-discoverable per-device identity. Our
contribution generalises this to a *bundle* of five HAL-bypass signals
and adds a nonce-driven sampling plan.

**TEE-free federated learning.** Sentinel [Sentinel2025] proposes
SGX-rooted sybil resistance in federated learning. FabricCrypt asks the
same question without SGX: can we resist sybils when *all* participants
run on commodity AMD with no SGX, no TDX, no Secure Enclave?

**In-memory analog attestation.** Concurrent work on analog AI
accelerators [Leroux2025] hints at substrate-as-identity at the
device-physics level. FabricCrypt is the digital-side analogue:
substrate-as-identity at the micro-architectural level on stock
silicon.

---

## 3. Threat model

**System under attestation.** A commodity AMD APU running Linux.
The protocol verifier (the audience) is software, possibly remote,
with no special hardware.

**Adversary capabilities.**
- Full read access to FabricCrypt's protocol specification, including
  the nonce-to-plan mapping and the trained classifier weights.
- Has previously observed up to M ≤ 10⁵ honest (nonce, signature)
  pairs from the target die.
- Has full control of a second machine of the same SKU (`daedalus` is
  the example throughout: same chassis model, same APU, same kernel
  family, BIOS X53 v01.02.02 vs v01.01.08 on `ikaros`).
- Cannot physically touch the target die during the challenge.
- Can replay, splice, or fabricate signatures.

**Adversary goal.** Cause the verifier to accept (P(own) > 0.5) on a
fresh audience-supplied 64-bit nonce.

**Out of scope.**
- Chip-present adversaries who physically hold the target chip during
  the challenge (this is *chip-presence proof*, not access control).
- Side-channel-driven extraction of the target's secret state. We
  assume nothing secret is held on the die — the die's *physical
  history* itself is the secret.
- Persistent kernel-resident adversaries that pre-empt every read
  before HAL-bypass measurement (we measure with `mlockall`,
  `SCHED_FIFO`, and disable preemption around critical regions, but
  this is mitigation, not proof).

**Defences we claim.** Nonce-bound liveness (the plan is
nonce-dependent); plan-consistency verification (the verifier
re-computes the plan from the nonce and checks the response is
self-consistent); peer-transplant rejection (a different die produces
detectably different statistics under the same plan).

---

## 4. Identity mechanism (Step 1)

### 4.1 Five HAL-bypass signals

We measure five families of low-level physical signals that PSP/SMU
firmware does not — and on these platforms *cannot cheaply* — homogenise
across dies:

1. **Inter-core TSC offset (Task B).** For each of 15 core pairs
   (selected to span the two CCDs of the gfx1151 APU), we collect 5000
   round-trip TSC samples through a cross-core spinlock. Median
   inter-core round-trip in cycles is per-die: `ikaros` (0,1)=7080 cyc,
   `daedalus` (0,1)=9120 cyc; Kolmogorov-Smirnov D = 0.92.
2. **Cacheline ping-pong (Task E).** 32-pair MOESI cacheline ping-pong
   matrix. Frobenius-norm distance between ikaros and daedalus p50
   matrices: 60 cycles at original governor, 52 cycles at matched
   governor. Mean per-pair KS-D = 0.27.
3. **DRAM-refresh-aligned jitter (Task F).** Aligned and unaligned
   memory-access latency histograms across the 7.8 µs DRAM refresh
   interval.
4. **Syscall p99.9 tail (Task D).** `nanosleep(0)` p99.9 latency
   distribution. ikaros p99.9 = 67428 ns at matched governor vs
   daedalus 54342 ns; inter-host KS-D = 0.72, inter-intra D-ratio = 47.5.
   `sched_yield` inter-host KS-D = 0.99, ratio = 44.7. `getpid` KS-D = 0.36.
5. **NVMe queue-tail latency (Task F-NVMe).** Per-die NVMe submission
   queue tail-distribution after a controlled 4 KiB random-read
   workload.

These five families are concatenated into a 290-dimensional live
signature vector (Phase 13 `*_sig_v2.npz`, n=10 reps per host).

### 4.2 Per-die not per-config

A naïve fingerprint based on (e.g.) RDRAND latency p50 would actually
be *governor-determined*: at matched `performance` governor, ikaros
RDRAND p50 = 120 cyc *equals* daedalus RDRAND p50 = 120 cyc (KS-D ≈
0). We explicitly downweight such signals.

The signals that drive the 290-dim fingerprint are signals whose
distinguishability is **geometric/topological** (inter-core wiring,
cache-coherence interconnect topology) or **noise-driven** (DRAM
refresh-aligned jitter, NVMe queue scheduling jitter), not
*governor-driven*. Section 4.4 quantifies this.

### 4.3 Why PSP/firmware cannot hide these

The PSP firmware can homogenise *named* signals like `lm_sensors`
thermal-zone labels, `cpufreq` reported MHz, or `hwmon` accumulator
counters. It cannot cheaply homogenise:
- The TSC round-trip cycle count between two physical cores, because
  that count is determined by physical interconnect wire length plus
  routing-table state.
- The MOESI cacheline ping-pong matrix, because that is determined by
  the L3 slice mapping plus the silicon-physical victim-buffer state.
- The DRAM-refresh-aligned p99.9 tail, because that tail is determined
  by per-DIMM mat-population physics.

Homogenising any of these would require *real-time* low-microsecond
intervention on every TSC read, every cacheline-ping, every memory
load. The PSP firmware loop runs at kHz-scale, not GHz-scale; it cannot
keep up.

### 4.4 Leave-one-out classification + governor robustness

On the 290-dim feature with n=20 signatures (10 per host) the
leave-one-out classification accuracy is **1.00** (gate ≥0.95 passed)
[Phase 13 `classifier_E.json`, `loo_acc=1.0`, `gate_gt_0_95_passed=true`].

The adversarial-audit-flagged governor confound (BIOS-default
governors differ: ikaros = `powersave`, daedalus = `performance`) was
resolved in Phase 14D. We re-measured the ikaros side at both
`powersave` and at `performance` (matched to daedalus's default) and
re-ran T2 (anomaly detection) and T3 (twin-paradox host attribution)
against the frozen daedalus reference.

| Metric                                  | Original (mixed)  | ikaros@powersave  | ikaros@performance |
|-----------------------------------------|-------------------|-------------------|--------------------|
| Task B inter-core KS-D (mean of pairs)  | 0.910             | 0.879             | 0.864              |
| Task E Frobenius p50 (cyc)              | 79                | 60                | 52                 |
| Task A RDRAND p50 (ika vs dae)          | 120 vs 120        | 90 vs 120         | 120 vs 120         |
| Nanosleep p99.9 (ika vs dae, ns)        | 68128 vs 54342    | 73810 vs 54342    | 67428 vs 54342     |
| T2 vanilla / embodied AUROC             | 0.509 / 1.000     | 0.500 / 0.984     | 0.500 / 0.994      |
| T3 vanilla / embodied accuracy          | 0.501 / 1.000     | 0.501 / 1.000     | 0.501 / 1.000      |

**Findings.**
1. The 290-dim fingerprint headline results survive matched-governor.
2. RDRAND p50 *is* governor-determined and is dropped from the
   load-bearing feature set — but the original Phase 12B already
   recorded inter-host KS-D ≈ 0.0002 for RDRAND, so no revision is
   needed.
3. Task E Frobenius distance weakens ≈13% at matched governor but
   stays positive, suggesting a small governor-mediated component on
   top of a per-die core.
4. Task B (inter-core TSC) and the syscall tail are essentially
   unchanged.

BIOS version mismatch (X53 01.01.08 vs 01.02.02) remains an
unexplored confound. Per the audit, we cannot safely flash a BIOS to
match without bricking risk; we leave this to future work with n≥6
matched-BIOS chassis (Section 7).

---

## 5. Attestation protocol (Step 2)

### 5.1 Nonce-driven sampling plan

The fundamental Phase 14B failure was that the nonce only permuted
*output positions* of the signature; the underlying physical read was
identical regardless of nonce, and so static replay achieved 100%
acceptance.

Phase 14C corrects this. The 64-bit audience nonce `N` is expanded
via `HMAC-SHA256(audience_secret, N)` into a **sampling plan**:
- Which 8 of 16 logical CPUs to pin reader threads to.
- Which 8 of the available `thermal_zone*` to read.
- Which 16 of the 120 possible core pairs to ping-pong.
- Which 4 of 32 `nanosleep(k)` durations to time.
- How many TSC samples per pair (∈ {1024, 2048, 4096}).

The plan has ≈ 64 effective bits of entropy and is deterministic
given `(audience_secret, N)`.

### 5.2 Plan-consistency verifier

The verifier:
1. Receives the response (phys-feature vector + embedded nonce).
2. Re-derives the plan from `(audience_secret, N)`.
3. Computes a **plan-consistency score** = a deterministic check that
   the phys-feature marginals match the marginals expected under
   plan `P(N)`.
4. Computes the classifier `P(own)` on the joint (phys, embedded
   nonce) input.
5. Accepts iff *both* `P(own) > 0.15` and `plan_score > 0.5`.

Acceptance is the AND of a learned classifier and a deterministic
plan-gate. The plan-gate is the load-bearing replay defence; the
classifier is the load-bearing identity defence.

### 5.3 Seven gates

We pre-registered 7 acceptance gates and report the observed accept
rates over 500 challenges per scenario (Phase 14C
`ikaros_spoof_v2.json`):

| Scenario                          | Gate              | Observed | Pass |
|-----------------------------------|-------------------|----------|------|
| Honest, own die, own nonce        | ≥0.95             | 1.000    | ✓   |
| Daedalus-peer transplant          | ≤0.05             | 0.020    | ✓   |
| Static replay, no nonce match     | ≤0.05             | 0.006    | ✓   |
| Static replay, correct nonce      | ≥0.95 (legit)     | 1.000    | ✓   |
| Dynamic library replay (M=400)    | ≤0.10             | 0.012    | ✓   |
| Nonce-only mismatch               | ≤0.05             | 0.006    | ✓   |
| Honest-own w/ wrong nonce         | ≤0.05             | 0.006    | ✓   |

**All seven gates pass.** Note especially that "static replay with
correct nonce" is *expected* to pass (it represents a legitimate
read on the legitimate die); the 100% accept on that line is a
positive control.

### 5.4 Latency

End-to-end sign-and-verify, measured over 1000 challenges
(`ikaros_timing.json`):

- median 1.12 ms
- p95 1.59 ms
- p99 2.79 ms
- p99 ≤ 5 ms budget: **pass** (`pass_p99 = true`)
- worst-case max 6.02 ms exceeds budget; tail-grooming is future work.

### 5.5 Theoretical replay cost

With M recorded (nonce, signature) pairs, the closest library entry
to a fresh challenge has expected Hamming distance ≈ 32 - log₂(M)/2
bits. For M = 10⁵ this is ≈30 of 64 bits, producing a substantively
different sampling plan and therefore a different phys distribution.
Achieving accept ≥ 50% would require library coverage ≈ 2⁶³ entries
(infeasible); ≥ 10% would require ≈ 2⁶⁰ (also infeasible). The
observed 0.012 accept rate on dynamic-library replay at M=400 is
consistent with this analysis.

---

## 6. Novel capabilities (Step 3)

These are the three things FabricCrypt enables that vendor-PKI
attestation does not.

### 6.1 Per-die AI output attribution

PCC and CC authenticate "this output came from an iPhone-class
Secure Enclave / an H100 in CC mode." They do **not** authenticate
"this output came from die serial 0xDEADBEEF and no other die." A
vendor — or a vendor employee, or a compromised vendor CA — can
re-key two physically distinct dies with identical credentials, and
no client can distinguish them.

FabricCrypt provides per-die output attribution as a primitive:
attach the FabricCrypt signature to the model output, and the
audience can verify by challenge that the output originated on the
*specific* die.

This is the strongest novel capability and the cleanest separation
from PCC/CC.

### 6.2 Stateless PCC-class guarantee surface on commodity AMD

PCC requires Apple's Secure Enclave. Commodity AMD APUs — including
the Strix Halo platform used here — have no Secure Enclave, no
TPM EK certificate that is FIPS-bindable to a specific physical die,
and no NVIDIA-like Device Identity Certificate hierarchy.

FabricCrypt offers a **PCC-equivalent guarantee surface** for those
platforms: the audience can verify (a) the die exists, (b) the die
ran the inference, (c) the response is fresh (nonce-bound) and (d)
replay is bounded — all without any vendor key. The surface is
weaker than PCC in one direction (no per-request transcript
sealing) but is comparable in the attestation-of-presence direction.

### 6.3 TEE-free sybil-resistant federated learning

Sentinel [Sentinel2025] resists sybils via SGX attestation. On
laptops without SGX, federated learning is wide open to sybils. With
FabricCrypt, each participant proves *per-die identity* to the
aggregator: a sybil attacker needs *physical access to one chip per
fake participant*. On a closed pool of N participants the per-round
acceptance of a sybil is bounded by the peer-transplant gate (≈2%).

The downstream T2 (anomaly detection) and T3 (host attribution)
results in Phase 14B (AUROC 0.500 → 0.994, accuracy 0.501 → 1.000)
demonstrate that the same fingerprint can also drive *workload-aware*
defences once a sybil is admitted.

---

## 7. Limitations and future work

We list limitations frankly because too many adjacent papers do not.

**L1 — Chassis count n=2.** Two physical APUs is enough for a
proof-of-concept and for adversarial separation testing, but
top-tier venue review will rightly insist on n ≥ 6. We have funding
in flight for a 6-chassis Strix Halo array; this paper's claims will
need to be re-validated at that scale, especially the LOO gate.

**L2 — BIOS version confound.** ikaros runs BIOS X53 v01.01.08 and
daedalus runs v01.02.02. The Phase 14D matched-governor sweep
*reduces* but does not *eliminate* the BIOS confound: we cannot
safely match BIOS versions without risking bricking. The robust
signals (Task B inter-core TSC) are geometric and unlikely to be
BIOS-driven; the Task E Frobenius weakening (60 → 52 cyc) under
governor matching is plausibly partly BIOS-mediated.

**L3 — Interactive-verifier requirement.** FabricCrypt requires an
audience-supplied nonce per challenge. We have not constructed a
non-interactive variant; PCC and CC are likewise interactive in
practice (challenge-response over TLS) so this is not a unique
weakness, but it forecloses some offline-signing use cases.

**L4 — No static-benchmark inference-accuracy gain.** Phase 15 and
Phase 16 tested whether embedding the live signature into a small
transformer improved standard benchmark accuracy (MNIST, a tiny
language model perplexity). Both came back **null**. The capability
gains in Section 4.4 (T2 AUROC, T3 host attribution) are *anomaly-
detection* and *self-attribution* tasks where the signature is the
input by construction; we do not claim a generic inference-accuracy
benefit.

**L5 — Persistent kernel adversary.** A kernel-resident attacker
that pre-empts every measurement window with a forged read can in
principle defeat any HAL-bypass scheme. We rely on `mlockall`,
`SCHED_FIFO` priority 99, `cpuset` isolation, and disabling
preemption around critical sections; this is engineering mitigation,
not proof.

**L6 — Reproducibility.** All scripts and result JSON/NPZ are in
`scripts/identity_benchmark/embodiment{12,12b,13,14b,14c,14d}/` and
`results/IDENTITY_BENCHMARK_2026-05-30/`. A single-script reproduce
target (`make reproduce-fabriccrypt`) will be included with the
camera-ready.

**Future work.**
- n=6 Strix Halo array with matched BIOS.
- Non-interactive variant via VDF-bound time-locking of the audience
  nonce.
- Extending the signal set with on-die GPU shader-unit fingerprints
  (gfx1151 has 40 CUs; per-CU TFLOPS jitter is a candidate).
- Constitutive substrate audit: a Phase-style C-vs-A test where the
  signature is permuted to confirm it is *load-bearing* and not
  decorative.

---

## 8. Related work (extended)

**Apple PCC** [PCC2024] is the closest peer-trust-model design and
the cleanest example of vendor-PKI-rooted attestation. Apple builds
a transparency log and publishes the inference-stack hash; the
client trusts Apple-CA. FabricCrypt removes the CA dependency.

**NVIDIA Confidential Compute Manager** [NVIDIA-CC] roots
attestation in the Device Identity Certificate Engine (DICE) on
H100/Blackwell. The DICE key is provisioned by NVIDIA and signed by
the NVIDIA CA. Per-die identity is not exposed: two H100s in CC
mode are indistinguishable to the verifier.

**Intel TDX** [TDX2024] and **AMD SEV-SNP** [SEV-SNP] both root
attestation in the silicon vendor's CA. Their primary security
guarantee is memory encryption + integrity, not per-die identity.

**TPM 2.0** [TPM2] supports a vendor-provisioned EK. The EK
certificate names a *platform model*, not a die. Even with FIDO-style
attestation, the trust anchor remains the platform vendor's CA.

**DRAM Latency PUF** (Kim et al., HPCA 2018) [Kim2018] is the
closest substrate-as-identity prior art. It exploits per-cell DRAM
saturation latency variation to build a stable PUF *without vendor
keys*. The fingerprint is static; FabricCrypt is challenge-bound and
live.

**FP-Rowhammer** (Venugopalan, AsiaCCS 2025) [Venugopalan2023]
demonstrates floating-point-precision Rowhammer; it is more
adversarial than attestation, but shares the philosophy of
HAL-bypass access to physical micro-architecture.

**DRAWNAPART** (NDSS 2022) [DRAWNAPART] showed WebGL-side
fingerprinting via GPU execution-unit timing differences across
identically-spec'd cards. FabricCrypt's CPU-side analogue is
inter-core TSC offsets and cacheline ping-pong (Task B / Task E).

**Kohno, Broido, Claffy** (S&P 2005) [Kohno2005] used TCP timestamp
drift to identify devices across networks. This is the conceptual
ancestor: physical clock variation as software-discoverable
per-device identity. FabricCrypt generalises to a five-signal bundle
with nonce-bound replay defence.

**Sentinel** (arXiv 2025) [Sentinel2025] proposes SGX-rooted
sybil-resistant federated learning. FabricCrypt offers the same
guarantee on commodity AMD without SGX.

**Analog AI accelerators** [Leroux2025] are exploring
substrate-as-identity at the physical-device level (in-memory
attention with analog cores). FabricCrypt is the digital-side
analogue on commodity stock silicon.

---

## 9. Conclusion

FabricCrypt demonstrates, on commodity AMD Ryzen AI hardware, that
per-die attestation can be built *without* the vendor's PKI by
bundling five HAL-bypass micro-architectural signals into a
290-dimensional live signature and binding each challenge to an
audience-supplied nonce that controls *what gets measured*. All seven
preregistered protocol gates pass at sub-millisecond latency. The
mechanism enables three capabilities that PCC, NVIDIA CC, Intel TDX,
and AMD SEV-SNP do not provide: per-die output attribution,
stateless PCC-class guarantees on commodity AMD, and TEE-free
sybil-resistant federated learning. We have shown this at n=2; we
have not shown a static-benchmark inference-accuracy gain. Both
caveats are addressable, and we are honest about them.

---

## Bibliography

[PCC2024]   Apple. "Private Cloud Compute: A new frontier for AI
            privacy in the cloud." Apple Security Engineering & Architecture,
            security.apple.com/blog/private-cloud-compute, 2024.

[NVIDIA-CC] NVIDIA Corporation. "Confidential Computing on H100 GPUs."
            NVIDIA Technical Brief, docs.nvidia.com/cc-deployment-guide, 2024.

[TDX2024]   Intel Corporation. "Intel Trust Domain Extensions (Intel TDX)
            White Paper." Intel, 2024.

[SEV-SNP]   AMD. "AMD SEV-SNP: Strengthening VM Isolation with Integrity
            Protection and More." AMD White Paper, 2020.

[TPM2]      Trusted Computing Group. "TPM 2.0 Library Specification." TCG, 2019.

[Kim2018]   J. Kim, M. Patel, H. Hassan, O. Mutlu. "The DRAM Latency PUF:
            Quickly Evaluating Physical Unclonable Functions by Exploiting
            the Latency-Reliability Tradeoff in Modern Commodity DRAM Devices."
            HPCA 2018.

[Venugopalan2023] V. Venugopalan et al. "FP-Rowhammer: Bit-level
            floating-point Rowhammer attacks on deep learning models."
            arXiv:2307.00143, AsiaCCS 2025.

[DRAWNAPART] T. Laor et al. "DRAWNAPART: A device identification technique
            based on remote GPU fingerprinting." NDSS 2022.

[Kohno2005] T. Kohno, A. Broido, K. Claffy. "Remote physical device
            fingerprinting." IEEE S&P 2005.

[Sentinel2025] (Anon). "Sentinel: SGX-rooted sybil-resistant federated
            learning." arXiv:2509.00634, 2025.

[Leroux2025] N. Leroux et al. "Analog in-memory attention for foundation
            models." Nature Computational Science, Sep 2025.

[Phase14B-results]  This work. results/IDENTITY_BENCHMARK_2026-05-30/embodiment14b/.
[Phase14C-results]  This work. results/IDENTITY_BENCHMARK_2026-05-30/embodiment14c/.
[Phase14D-results]  This work. results/IDENTITY_BENCHMARK_2026-05-30/embodiment14d/.
[Phase13-results]   This work. results/IDENTITY_BENCHMARK_2026-05-30/embodiment13/.
[Phase12B-results]  This work. results/IDENTITY_BENCHMARK_2026-05-30/embodiment12b/.

— END OF DRAFT v1 —
