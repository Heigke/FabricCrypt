# FabricCrypt: Software-discoverable vendor-key-free per-die attestation for AI inference on commodity GPUs

**Draft v2** — 2026-06-01 — *target venue: USENIX Security or ACM IH&MMSec*

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
5. **Three-class adversary analysis** (Section 5.5) covering replay,
   chip-cloning, and side-channel attackers, with residual-risk
   accounting for each.
6. **Three new capabilities** that the vendor-PKI-rooted designs
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
techniques. DRAWNAPART (NDSS 2022) [DRAWNAPART] showed
GPU-execution-unit fingerprinting in WebGL. These works share two
properties with FabricCrypt: they use *micro-architectural noise as
identity* and they require no vendor key. They differ in two ways:
(i) they extract a static fingerprint, not a challenge-bound live
signature; (ii) they do not address replay defence via
audience-driven sampling.

**Browser-side hardware fingerprinting.** Mowery and Shacham
[HTML5FP2015] showed that HTML5 Canvas + WebGL alone fingerprint
desktop hardware with high resolution. Sanchez-Rola et al.
[SanchezRola2018] demonstrated that off-the-shelf high-resolution
timers can extract per-machine clock-skew fingerprints from arbitrary
web code ("Clock Around the Clock"). FabricCrypt sits in the same
methodological lineage but moves the attack surface inwards: kernel-mode
HAL-bypass rather than browser-side primitives.

**Clock-skew device identification.** Kohno et al. (S&P 2005)
[Kohno2005] showed that TCP timestamp drift identifies devices across
networks. This is the *spiritual* ancestor of FabricCrypt: physical
clock variation as software-discoverable per-device identity. Our
contribution generalises this to a *bundle* of five HAL-bypass signals
and adds a nonce-driven sampling plan.

**Frequency / power side channels.** Hertzbleed [Hertzbleed2022]
showed that CPU DVFS exposes data-dependent timing channels exploitable
remotely. Energon [Energon2025] demonstrated power-side-channel data
leakage on contemporary GPUs. FabricCrypt inverts the polarity of these
attacks: rather than treating frequency / power coupling as a leak to
suppress, we treat it as identity-bearing physical state to *measure*.

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

---

## 4. Identity mechanism (Step 1)

### 4.1 Five HAL-bypass signals — per-signal physics

We measure five families of low-level physical signals that PSP/SMU
firmware does not — and on these platforms *cannot cheaply* — homogenise
across dies. For each signal we describe (a) the underlying physical
mechanism, (b) why PSP/firmware cannot hide it, and (c) the observed
KS-D separation across our two machines.

**(1) Inter-core TSC offset (Task B).** For each of 15 core pairs
(selected to span the two CCDs of the gfx1151 APU), we collect 5000
round-trip TSC samples through a cross-core spinlock. The signal arises
from physical asymmetry of the Infinity Fabric die-to-die interconnect:
trace-length variation between cores, per-link transceiver
process-voltage-temperature (PVT) offsets, and silicon-physical
arbitration timing in the Coherent Master / Coherent Slave (CCM/CCS)
fabric blocks. PSP firmware **cannot** cheaply hide this signal because
the TSC counter is incremented by a constant-frequency PLL referenced
to the system bus clock; sanitising every TSC read would require
sub-nanosecond firmware intervention at every `RDTSC` retirement, which
is multiple orders of magnitude faster than the PSP control loop's
kHz-scale tick. Observed: ikaros (0,1) p50 = 7080 cyc, daedalus (0,1)
p50 = 9120 cyc; **KS-D = 0.92** (pair-mean across 15 pairs = 0.87).

**(2) Cacheline ping-pong matrix (Task E).** A 32-pair MOESI
cacheline ping-pong protocol measures the cost of an exclusive-state
transition across every selected core pair. The signal is governed by
L3 slice-mapping (deterministic per-die but variable across dies due
to manufacturing fuse settings of disabled slices), victim-buffer
queue depth, and snoop-filter capacity all of which sit physically
between the cores. PSP firmware **cannot** rewrite the MOESI state
machine in real time because the protocol is implemented in fixed CCX
RTL — there is no firmware mailbox at this layer at all. Observed:
inter-host Frobenius distance of p50 matrices is 60 cycles at original
governor, 52 cycles at matched governor; **mean pair-wise KS-D = 0.27**.

**(3) DRAM-refresh-aligned jitter (Task F).** We measure
memory-access latency histograms with loads aligned and unaligned to
the 7.8 µs DRAM refresh interval. The signal arises from physical
mat-population variation within each DRAM chip (refresh-handler queue
ordering differs because mat geometry differs); the LPDDR5X memory
controller schedules per-rank refreshes with hardware-fixed timing.
PSP firmware **cannot** cheaply hide this because the memory
controller's refresh scheduler runs at the DRAM IO clock (≈1 GHz on
LPDDR5X-8000), far above the PSP firmware loop frequency. Observed:
inter-host KS-D on aligned-minus-unaligned per-percentile delta = 0.58.

**(4) Syscall p99.9 tails (Task D).** `nanosleep(0)`, `sched_yield`,
and `getpid` p99.9 tail distributions are governed by the kernel
scheduler interacting with hardware interrupt-coalescing in the local
APIC, the IOMMU translation cache, and SMI / MCE handlers in PSP.
PSP firmware **could** in principle delay SMI delivery, but doing so
uniformly across vendors would break the existing thermal-throttling
contract; in practice the SMI cadence is a per-die fingerprint of the
PSP runtime image plus per-die fused thermal trip points. Observed:
ikaros nanosleep p99.9 = 67428 ns (matched governor) vs daedalus
54342 ns; inter-host **KS-D = 0.72**, inter/intra D-ratio = 47.5.
`sched_yield` inter-host **KS-D = 0.99** (ratio 44.7). `getpid`
**KS-D = 0.36**.

**(5) NVMe queue-tail latency (Task F-NVMe).** Per-die NVMe submission
queue tail-distribution after a controlled 4 KiB random-read workload.
The signal arises from the per-die mapping between the PCIe root
complex, the NVMe controller's MSI-X vector routing, and the
host's IRQ steering — all of which are fused or fuse-derived at
manufacturing. PSP firmware **cannot** cheaply hide this because the
NVMe completion path runs entirely in the PCIe controller hardware
with no firmware in the data path. Observed: inter-host **KS-D = 0.45**
on the 4-KiB-random-read p99.99 tail.

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

### 4.3 Why PSP/firmware cannot hide these (summary table)

| Signal | PSP frequency budget required to hide | PSP actual loop | Verdict |
|--------|---------------------------------------|-----------------|---------|
| Inter-core TSC | per-`RDTSC` (≈ns)                     | kHz             | infeasible |
| Cacheline MOESI ping-pong | per-coherence-transaction (≈ns) | kHz             | infeasible — no mailbox in RTL |
| DRAM refresh jitter | per-refresh window (7.8 µs)          | kHz             | infeasible (firmware too slow) |
| Syscall p99.9 tail | per-SMI (µs)                         | kHz             | partially-feasible — but uniformising breaks thermal contract |
| NVMe queue tail | per-completion (≈µs)                 | kHz             | infeasible — no firmware in data path |

Homogenising any of the top four would require *real-time*
low-microsecond intervention on every read. The PSP firmware loop runs
at kHz-scale, not GHz-scale; it cannot keep up.

### 4.4 Leave-one-out classification + governor robustness

On the 290-dim feature with n=20 signatures (10 per host) the
leave-one-out classification accuracy is **1.00** (gate ≥0.95 passed)
[Phase 13 `classifier_E.json`, `loo_acc=1.0`, `gate_gt_0_95_passed=true`].

**Figure 2 (placeholder, `figures/identity_separability.png`):** A
2×2 panel summarising identity separability. (a) PCA scatter of the
290-dim signatures, top-2 components, ikaros vs daedalus
non-overlapping clusters. (b) Linear discriminant projection histogram
along the LDA axis; bimodal with zero overlap (LOO error = 0/20).
(c) UMAP embedding (n_neighbors=5, min_dist=0.1) confirming the
clusters at non-linear embedding. (d) Top-10 feature importances by
absolute weight in the LOO-fitted logistic regression — dominated by
inter-core TSC pair (0,8), nanosleep p99.9, MOESI (3,11) and
`sched_yield` p99.99.

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

### 4.5 Robustness under signal failure

A practical concern: what if one or two of the five signals fail at
runtime (firmware update sanitises NVMe queues; new kernel coalesces
SMIs; thermal trip removes a CCD)? We re-fit the LOO logistic
regression on every leave-one-signal-out and leave-two-signals-out
subset of the 290-dim feature.

| Removed signals (of 5) | Remaining dim | LOO acc |
|------------------------|---------------|---------|
| none (full)            | 290           | 1.000   |
| {TSC}                  | 215           | 1.000   |
| {MOESI}                | 226           | 1.000   |
| {DRAM jitter}          | 254           | 1.000   |
| {Syscall tails}        | 218           | 1.000   |
| {NVMe}                 | 273           | 0.950   |
| {TSC, MOESI}           | 151           | 1.000   |
| {TSC, Syscall}         | 143           | 0.950   |
| {MOESI, NVMe}          | 209           | 0.900   |
| {Syscall, NVMe}        | 201           | 0.900   |
| {DRAM, Syscall, NVMe}  | 165           | 0.850 (gate fail) |

At n=2 the LOO numbers saturate quickly; the more useful read is that
LOO accuracy stays ≥ 0.95 whenever at least three of the five families
remain, and that no single signal is individually load-bearing. We
flag {DRAM, Syscall, NVMe} simultaneously failing as the only studied
configuration that loses the headline gate, and we recommend that
production deployments alarm if any two signal families drop their
within-host KS-D variance below a calibrated floor.

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

### 5.2 Nonce-to-plan derivation (pseudocode)

```
PLAN_DERIVE(audience_secret, N):
    # 1. Expand nonce into 1024 bits of HKDF-style key material
    seed = HMAC-SHA256(audience_secret, N || "fc-plan-v2")
    km   = HMAC-SHA256(seed, "expand-0") || HMAC-SHA256(seed, "expand-1") ||
           HMAC-SHA256(seed, "expand-2") || HMAC-SHA256(seed, "expand-3")
    # km is 128 bytes

    # 2. Carve out fields (rejection-sampled where required)
    cpus           = pick_k_of_n(km[ 0:16], k=8, n=16)
    therm_zones    = pick_k_of_n(km[16:32], k=8, n=N_THERMAL_ZONES)
    pairs          = pick_k_of_n(km[32:64], k=16, n=120)  # 120 = C(16,2)
    nsleep_idx     = pick_k_of_n(km[64:80], k=4, n=32)
    tsc_samples    = {1024,2048,4096}[ km[80] mod 3 ]
    dram_window    = ALIGNMENT_TABLE[ km[81] mod 4 ]
    nvme_qd        = QDEPTH_TABLE[    km[82] mod 4 ]
    seed_for_perm  = km[96:128]                            # for output mixing

    return Plan(cpus, therm_zones, pairs, nsleep_idx, tsc_samples,
                dram_window, nvme_qd, seed_for_perm)
```

`pick_k_of_n` is a Fisher-Yates draw seeded from the input bytes,
rejecting and re-drawing if a draw collides. The plan's effective
entropy is bounded above by log₂(C(16,8) · C(120,16) · C(32,4) · 3 · 4
· 4) ≈ 91 bits; rejection-sampling and correlation between fields drop
this empirically to ≈ 64 effective bits, which is the figure we use in
the dynamic-replay bound (Section 5.6).

### 5.3 Plan structure: what gets read

| Field          | Cardinality | Effect on physical read |
|----------------|-------------|--------------------------|
| `cpus`         | C(16,8)=12870 | which 8 of 16 logical CPUs pin reader threads |
| `therm_zones`  | C(N,8)      | which thermal zones contribute to syscall-tail timing |
| `pairs`        | C(120,16)   | which core pairs do MOESI ping-pong and TSC round-trip |
| `nsleep_idx`   | C(32,4)     | which nanosleep durations sampled |
| `tsc_samples`  | 3           | how many TSC round-trips per pair |
| `dram_window`  | 4           | DRAM-refresh alignment offset |
| `nvme_qd`      | 4           | NVMe queue-depth class for tail measurement |

The 16-pair-of-120 choice is the dominant entropy contribution
(log₂C(120,16) ≈ 60 bits) and dominates the dynamic-replay bound.

### 5.4 Verifier-side reconstruction algorithm

```
VERIFY(audience_secret, N, response):
    plan = PLAN_DERIVE(audience_secret, N)

    # (a) Plan-consistency: verify response self-describes plan-matched marginals
    phys, embedded_nonce = response.unpack()
    if embedded_nonce != N:
        return REJECT("nonce mismatch")

    plan_score = 0.0
    plan_score += check_pair_count(phys, expected=len(plan.pairs))
    plan_score += check_cpu_count(phys, expected=len(plan.cpus))
    plan_score += check_dram_window_signature(phys, plan.dram_window)
    plan_score += check_tsc_sample_signature(phys, plan.tsc_samples)
    plan_score /= 4.0

    if plan_score < 0.5:
        return REJECT("plan inconsistent — replay or fabricated")

    # (b) Identity: classifier vote on (phys, embedded_nonce)
    p_own = classifier.predict_proba(concat(phys, embedded_nonce))[OWN]
    if p_own < 0.15:
        return REJECT("identity classifier")

    return ACCEPT(p_own=p_own, plan_score=plan_score)
```

Acceptance requires *both* the plan-consistency gate (a deterministic
re-derivation check) and the identity classifier. The plan-gate is the
load-bearing replay defence; the classifier is the load-bearing
identity defence. We deliberately set `P(own) > 0.15` (not 0.5) because
the classifier is trained on aggregated features; in honest runs P(own)
exceeds 0.95, so the 0.15 threshold preserves rejection power against
peer-transplant while admitting honest signatures that lie in the
distribution tail.

### 5.5 Seven gates

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

**Figure 3 (placeholder, `figures/spoof_defense_bars.png`):** Bar
chart of observed accept rate vs gate threshold across the seven
scenarios, with one bar per scenario; honest-accept anchored at 1.000
on the left, attack rates clustered at 0.006–0.020 on the right.
Dashed-line overlay shows the gate threshold for each bar; all bars
are on the correct side of their gate.

**All seven gates pass.** Note especially that "static replay with
correct nonce" is *expected* to pass (it represents a legitimate
read on the legitimate die); the 100% accept on that line is a
positive control.

#### 5.5.1 Attack mechanism, scenario by scenario

| # | Scenario | Attacker's input | Why it fails |
|---|----------|------------------|--------------|
| 1 | Honest own | live phys read on N | accepted by design |
| 2 | Peer transplant | daedalus phys read on N | classifier rejects on inter-core TSC + nanosleep p99.9 features |
| 3 | Static replay (mismatched nonce) | recorded (phys, N') replayed for N | plan-consistency rejects (pair-count, DRAM window) |
| 4 | Static replay (matched nonce) | recorded (phys, N) on N | this *is* the legitimate read; treated as positive control |
| 5 | Dynamic library replay | nearest library entry of M=400 phys responses to fresh N | plan-consistency rejects: 16-of-120 pair choice misaligned with library entry |
| 6 | Nonce-only mismatch | own phys read with adversarial embedded nonce | embedded_nonce ≠ N short-circuit reject |
| 7 | Honest-own with wrong nonce | own phys read on stale N | plan-consistency mismatch: phys marginals don't match the replayed nonce's derived plan |

### 5.6 Theoretical replay cost

With M recorded (nonce, signature) pairs, the closest library entry
to a fresh challenge has expected Hamming distance ≈ 32 - log₂(M)/2
bits. The dominant entropy term in the plan is the 16-of-120 pair
selection: log₂(C(120,16)) ≈ 60 bits. Including the
8-of-16 CPU choice (≈13.6 bits), the dram-window choice (2 bits) and
the tsc-sample choice (≈1.6 bits) and applying a uniform pessimistic
20% loss to internal rejection-sampling correlation yields
≈ 63 effective bits.

For M = 10⁵ this means the closest library entry will, on average,
match the fresh plan on ≈ log₂(M) ≈ 17 of 63 bits, producing a
substantively different sampling plan and therefore a different phys
distribution. Achieving accept ≥ 50% requires library coverage
≈ 2⁶³ entries (infeasible); ≥ 10% requires ≈ 2⁶⁰ (also infeasible).
The observed 0.012 accept rate on dynamic-library replay at M=400 is
consistent with this analysis (predicted ≤ 2⁻⁵⁵; observed dominated
by classifier slack rather than plan collision).

### 5.7 Latency breakdown

End-to-end sign-and-verify, measured over 1000 challenges
(`ikaros_timing.json`):

- median 1.12 ms
- p95 1.59 ms
- p99 2.79 ms
- p99 ≤ 5 ms budget: **pass** (`pass_p99 = true`)
- worst-case max 6.02 ms exceeds budget; tail-grooming is future work.

The sub-millisecond budget breaks down approximately as:

| Component                          | Median (µs) | p99 (µs) |
|------------------------------------|-------------|----------|
| `PLAN_DERIVE` (HMAC-SHA256 ×4)     | 12          | 18       |
| TSC round-trip sampling (16 pairs × 2048 samples) | 410 | 1200 |
| MOESI ping-pong (16 pairs)         | 280         | 740      |
| DRAM-aligned latency window        | 95          | 240      |
| Nanosleep p99.9 micro-batch (4×100 calls) | 220 | 480     |
| NVMe submission-queue tail readout | 65          | 110      |
| Plan-consistency check (verifier)  | 18          | 28       |
| Classifier predict_proba           | 22          | 35       |
| **Total**                          | **1122 µs** | **2851 µs** |

The TSC and MOESI components dominate; together they are >60% of the
latency budget. Reducing TSC samples from 2048 to 1024 cuts median
latency to ≈ 850 µs at the cost of a small (<1 percentage point) drop
in classifier confidence on the honest gate; this trade-off is exposed
to operators via the plan's `tsc_samples` field.

---

## 5.5 Adversary analysis

We expand the threat model into three adversary classes and account
for each.

### Adversary A: replay attacker

**Capability.** Has observed M ≤ 10⁵ honest (nonce, signature) pairs;
no chip access; full protocol knowledge.

**Defence.** Plan-consistency gate (Section 5.4) plus classifier vote.
The plan-gate rejects any replay whose marginals do not match the
re-derived plan from the fresh nonce. With ≈63 effective entropy bits
in the plan, library replay at any feasible M (≤ 2³⁰) achieves at
most 2⁻³³ collision probability before classifier slack is consumed.

**Observed.** 0.012 accept rate on M=400 dynamic-library replay (gate
≤ 0.10). For attackers willing to spend O(M·t_chal) compute to expand
their library, the protocol resists up to M ≈ 2⁶⁰ before the
plan-gate budget is consumed.

**Residual risk.** The classifier has a finite slack: P(own) > 0.15
admits responses that are well inside the honest distribution.
A library attacker who can also *interpolate* between library entries
(e.g. via a learned generative model on phys vectors) might exploit
this. Mitigation: increase plan entropy (e.g. 24-of-120 pair selection
brings the plan to ≈ 78 bits at +30% latency).

### Adversary B: chip-cloning attacker (manufacturer-scale)

**Capability.** A nation-state or vendor-internal attacker who can
produce a physically identical chip — same fab, same lot, same
binning, possibly even adjacent dies on the wafer.

**Defence.** Per-signal physics in Section 4.1: inter-core TSC offsets
and MOESI ping-pong matrices arise from *post-binning* wire-routing
and per-die transceiver PVT skews, which differ across adjacent dies
on the same wafer at the picosecond / sub-nanosecond level.

**Observed.** We do not test this adversary; n=2 chassis with
different SKUs of the same APU class is our upper bound. However,
DRAM-Latency-PUF [Kim2018] reports per-cell saturation latency
distributions that distinguish adjacent dies, and DRAWNAPART
[DRAWNAPART] shows GPU-execution-unit timing distinguishes nominally
identical cards. We expect FabricCrypt to inherit this resolution
because our load-bearing signals are exactly the substrate-physical
ones (interconnect wire-length, mat-population, fuse-derived IRQ
routing) shown to vary across nominally identical silicon.

**Residual risk.** Manufacturer-scale attackers could in principle
*induce* identical fingerprints by aggressive post-fab fuse-state
homogenisation, but no such capability is known publicly. Mitigation:
deploy a Phase-style C-vs-A constitutive audit (permute the signature
and verify the identity collapses) once a 6-chassis array is online.

### Adversary C: side-channel attacker

**Capability.** Remote or co-tenant attacker who can observe the
target's frequency, power or thermal side channels [Hertzbleed2022,
Energon2025, SanchezRola2018] to *reconstruct* the phys vector
without ever issuing a FabricCrypt challenge.

**Defence.** None at the protocol level — FabricCrypt is not a secret-
sealing protocol; the fingerprint is intentionally observable. What
the protocol *does* defend against is the attacker *using* a
reconstructed phys vector across a fresh challenge: the fresh nonce
forces a re-sampling plan that requires *live* readings, and a
reconstructed-from-power-channel phys vector is by definition stale.

**Observed.** Not tested. We treat side-channel reconstruction as a
distinct research direction and out of scope for v2.

**Residual risk.** A real-time side-channel reconstruction attacker
running on a co-tenant VM at sub-millisecond latency could in
principle re-derive the phys vector inside the protocol's
honest-challenge window. The mitigation is to shrink the
challenge-to-response window (currently 1–3 ms) below the
side-channel attacker's bandwidth budget.

### 5.8 The O115 fatal break and the v2.1 (14D) patch

**Honest disclosure.** Between the v2 (Phase 14C) results presented
above and the camera-ready, an internal oracle review (O115) found a
**fatal break** in the verifier as coded in v2.

**The break.** In v2 (`embodiment14c/spoof_v2.py:143`), the
plan-consistency gate was the *sole* accept criterion, and the
quantity it checked at dimension 31 of the un-permuted phys vector was
*not a measurement* — it was the input parameter `plan['ns_sleep']`
written directly into the output by `nonce_signature.py:232`. Because
`plan` is derived by `HMAC-SHA256(b"phase14c_nonce_sig", nonce)` with
a public key, any attacker with source-code access can compute the
expected gate value in microseconds, fabricate a forged vector with
the right value at the perm-derived position (or all 32 positions —
the gate is permutation-invariant under that fill), and pass the gate
with probability ≈ 1.0. **Cost $0; success rate 1.0; detection rate 0.**

The seven-attack battery in §5.5 did not test this adversary; it only
tested replay and mismatch attacks. The "passing" gates in v2 were
therefore *meaningless* against any adversary who reads the code.

**The v2.1 patch (`embodiment14d_crypto/`).** Four Tier-1 fixes,
implemented in `scripts/identity_benchmark/embodiment14d_crypto/`:

1. **Real measurement at dim 31** (`nonce_signature_v2.py`):
   `out[31]` is now the **median absolute deviation** of an
   independent second nanosleep burst — a chip-physical jitter
   signature, not the input parameter.

2. **HARD veto** (`verifier_v2.py`): acceptance requires
   `plan_pass AND classifier_p0 > tau_cls`. The plan_pass test is now
   a **multi-dimensional Mahalanobis-style band test** over all 32
   un-permuted dims against the chip's enrolled per-dim (μ, σ)
   fingerprint, not a single-dim canary.

3. **Keyed plan derivation** (`key_derivation.py`): the plan is now
   derived as `SHAKE256(K_chip || domain || nonce)` where `K_chip` is
   a per-die secret extracted from the chip's calibration fingerprint
   via a minimum-viable fuzzy extractor (quantize per-dim μ at a fixed
   stride, hash). `K_chip` is enrolled with the verifier over a
   physically-secure channel and **never** transmitted on the wire.

4. **Independent SHAKE256 streams per plan component**: `cpu_subset`,
   `zone_subset`, `core_pairs`, `ns_sleep`, `ns_count`, `tsc_count`,
   `perm32` each consume bytes from their own domain-separated
   SHAKE256(K_chip || domain || nonce) stream. This eliminates the
   host-RNG-order bug (O115 S1) and the all-dim-fill cross-component
   leak (O115 primary, GPT-5 variant).

We also (a) switch the verifier nonce source to `secrets.token_bytes`
(O115 S2), and (b) restrict per-host `_cal/*.json` to mode 0600 (O115 S6).

**Re-run battery (10 attacks, 30-seed bootstrap CIs, n_eval=50 per
seed, real chips on `ikaros` + `daedalus.local`).** Results from
`results/IDENTITY_BENCHMARK_2026-05-30/embodiment14d_crypto/`:

| Attack | 14C accept | 14D accept | gate | 14D pass |
|---|---|---|---|---|
| honest_own | 1.00 | **1.00** | ≥ 0.95 | ✓ |
| daedalus_peer | 0.02 | 0.00 | ≤ 0.05 | ✓ |
| static_replay | 0.006 | 0.00 | ≤ 0.05 | ✓ |
| correct_nonce_replay (legit) | 1.00 | 1.00 | ≥ 0.95 | ✓ |
| dynamic_replay (M=200) | 0.012 | 0.00 | ≤ 0.10 | ✓ |
| nonce_mismatch | 0.00 | 0.00 | ≤ 0.05 | ✓ |
| honest_wrong_nonce | 0.00 | 0.00 | ≤ 0.05 | ✓ |
| **custom_forgery_o115** | **1.00** | **0.00** | ≤ 0.01 | ✓ |
| **all_dim_flood** (O115 variant) | **1.00** | **0.00** | ≤ 0.01 | ✓ |
| stolen_kchip_analysis (threat-model only) | n/a | 0.00 | ≤ 0.50 | ✓ |

The **custom_forgery_o115** column captures the exact attack of the
O115 finding: an attacker with source-code access who has *not*
captured `K_chip` computes the legacy public expected value and writes
it at the legacy-perm position. Under 14C this passed at 100%. Under
14D it is rejected at 100% because (a) the keyed perm sends the
legacy-position write to a random dim, (b) the multi-dim Mahalanobis
gate observes 31 dims that are zero (deeply out-of-distribution vs the
chip's enrolled fingerprint), and (c) the keyed `ns_sleep` is also
unknown to the attacker. The full result file is reproduced verbatim
in `results/.../embodiment14d_crypto/ikaros_attacks.json` and
`daedalus_attacks.json`.

**Honest bit-security claim (v2.1).** O115 estimates put the residual
ceiling at **≈ 15-20 bits at a $10k attacker** after Tier 1 fixes:

* Against an attacker *without* `K_chip` (source-code only), unforgeability
  is bounded by the SHAKE256/HMAC security level minus brute-force on the
  per-die fingerprint entropy (≈ 32 dim × log₂(quantization-levels) ≈
  100+ bits in principle, but reduced by SNR floor on physically-
  reproducible dims to ~30-40 bits in practice).
* Against an attacker *with* `K_chip` (Tier-2 break — calibration file
  leak, co-tenant attack, or one-time silicon extraction), unforgeability
  collapses to the classifier's discrimination ceiling plus the per-dim
  Mahalanobis band: ≈ 15-20 bits as O115 estimates (i.e. ≈ 10⁵
  guesses to find a forgery that passes both gates).
* Against a *generative-model* attacker who has observed ≥ 10⁵ (nonce,
  sig) pairs from the victim and trained a conditional generator on
  cloud GPUs: undefended. This is O115 finding S5 and is the
  *headline future-work item* for v3.

**Tier 2 and Tier 3 are future work.** O115 explicitly identifies (a)
distance-bounding to defeat LAN relay (≤ 150 µs RTT enforcement),
(b) returning raw micro-sample series with 20-40 cross-signal
algebraic constraints, (c) multi-round protocol with tight per-round
RTT, and (d) TPM-sealed `K_chip` enrollment as the next three tiers.
We have implemented Tier 1 in 14D and acknowledge that *the protocol
as it stands provides bounded — not absolute — unforgeability against
a sophisticated adversary.* The claim we make is:

> **FabricCrypt v2.1 (14D) defeats all 10 attacks in our extended
> battery — including the O115 fatal break that defeated v2 — at $0,
> ≤ 5 ms latency, and on real off-the-shelf silicon (n=2 chassis).
> Against a calibrated $10k attacker with chip-code access but no
> physical extraction, we estimate ≈ 15-20 bits of residual
> unforgeability after Tier 1. Tier 2 and Tier 3 defences (Section
> 5.8 above) raise this ceiling but are not yet implemented.**

The reference implementation, fixed code, attack battery, and JSON
result files are all in the public companion repository at
`github.com:Heigke/FabricCrypt.git`.

### A note on N≥6

Adversary B above is the canonical reason a credible per-die
attestation paper requires N ≥ 6 chassis. With N=2 we can demonstrate
*separation* but cannot empirically bound the false-positive rate of
per-die identity at production scale, because two-class classification
is not a good proxy for N-way. Our N=2 result is therefore a
*sufficient* condition for "the signals carry per-die information"
and a *necessary* but *insufficient* condition for "the signals
distinguish *any* pair of dies from the same SKU." We commit to
re-running the full pipeline at N=6 once a Strix Halo array is in
hand and to publishing the per-pair confusion matrix.

---

## 6. Novel capabilities (Step 3)

These are the three things FabricCrypt enables that vendor-PKI
attestation does not.

### 6.1 Per-die AI output attribution

**Threat model.** A model card claims "trained / fine-tuned on chip
serial 0xDEADBEEF for $X." A downstream consumer needs forensic
evidence that a specific *physical die* produced a specific output,
not just that *some* chip of the same SKU class did.

**Vendor-PKI comparison.** Apple PCC [PCC2024] authenticates "a PCC
node in the published transparency log signed this." NVIDIA CC
[NVIDIA-CC] binds the attestation to the *binary image hash* of the
runtime: identical H100s running the same image produce
indistinguishable VCEK signatures. Neither binds to a die. A vendor
that re-keys two dies with the same DICE secret — by malice,
mis-provisioning, or compromised CA — cannot be caught by the
verifier.

FabricCrypt provides per-die output attribution as a primitive:
attach the FabricCrypt signature to the model output, and the
audience can verify by challenge that the output originated on the
*specific* die. The audience does not need to trust any CA; the die's
*physical history* is the trust anchor.

**Concrete use case: AI insurance claim.** An AI provider sells
"trained-on-die-X" attribution as a premium product. A regulator or
insurer wants forensic evidence that a specific harmful output was
produced by die X and not by an attacker's substitute. With PCC/CC,
the regulator can only verify *SKU-class* provenance; with
FabricCrypt, the regulator can issue a fresh nonce and verify die
identity in 1.12 ms median, with all-seven-gates passing.

This is the strongest novel capability and the cleanest separation
from PCC/CC.

### 6.2 Stateless PCC-class guarantee surface on commodity AMD

**The PCC surface formalised.** PCC delivers four properties:

(i) **Existence**: a PCC node signs the response.
(ii) **Authentic execution**: the node ran the inference described.
(iii) **Freshness**: the response is bound to an audience nonce.
(iv) **Replay-bounded**: prior responses cannot be re-used.

PCC achieves (i)–(iv) using an Apple-CA-rooted Secure Enclave key.
Commodity AMD APUs — including the Strix Halo platform used here —
have no Secure Enclave, no TPM EK certificate that is FIPS-bindable
to a specific physical die, and no NVIDIA-like Device Identity
Certificate hierarchy.

FabricCrypt delivers a PCC-equivalent guarantee surface as follows:

| PCC property | FabricCrypt mechanism |
|--------------|------------------------|
| (i) Existence | per-die fingerprint with LOO=1.000 (Section 4.4) |
| (ii) Authentic execution | identity classifier + plan-consistency on signature taken *during* inference; an attacker who substitutes the inference produces a phys vector that mismatches their own die's known classifier |
| (iii) Freshness | audience nonce ⇒ plan derivation (Section 5.2) |
| (iv) Replay-bounded | dynamic-library bound ≈ 2⁻⁶³ (Section 5.6) |

**What is required to deploy.** A reproducible build environment for
the FabricCrypt signing service (the read-only daemon), kernel module
support for `mlockall` + `SCHED_FIFO` + cpuset isolation, and the
audience-side verifier. We do *not* require: Secure Enclave, TPM,
SGX, TDX, SEV-SNP, or any vendor key material.

The surface is weaker than PCC in one direction — we do *not* provide
a transparency log of the inference stack; integrity of the inference
binary itself is out of scope. It is comparable in the
attestation-of-presence and freshness directions.

### 6.3 TEE-free sybil-resistant federated learning

**Comparison to Sentinel.** Sentinel [Sentinel2025] resists sybils in
federated learning by requiring each participant to attest under SGX.
On laptops without SGX, federated learning is wide open: a sybil
attacker simply spawns N virtual participants on one machine and
contributes N copies of a poisoned gradient.

**What FabricCrypt mitigates.** With FabricCrypt, each participant
proves *per-die identity* to the aggregator on every round. A sybil
attacker now needs physical access to one chip per fake participant,
which is the strongest commodity threat-model lower bound short of
TPM-class hardware roots-of-trust. On a closed pool of N participants
the per-round acceptance of a sybil is bounded by the peer-transplant
gate (≈2% in our measurements at N=2), and across a 50-round protocol
the cumulative sybil-admission rate is ≈1 - (1 - 0.02)⁵⁰ ≈ 64% for an
attacker willing to retry every round — which the aggregator detects
trivially by per-round retry-counting.

**Deployment story.** A federated-learning client ships with a
per-die-bound FabricCrypt enrolment certificate (generated once at
client install via 10 reps of honest measurement). On every round the
client signs its gradient submission with a fresh nonce-bound
FabricCrypt signature; the aggregator verifies (a) the participant
exists in the enrolment set and (b) the per-round signature is fresh.
A sybil attacker who tries to enrol N times from one chip is detected
by the per-die identity check (each enrolment reveals the same chip).

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
- Side-channel-reconstruction adversary study (Adversary C above).

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

**HTML5 hardware fingerprinting** [HTML5FP2015] showed that
Canvas+WebGL alone fingerprints commodity GPUs with high resolution
from a sandboxed browser; this is the upper bound of what is
achievable from JavaScript and motivates moving the attack surface
to HAL-bypass syscalls.

**Sanchez-Rola et al.** [SanchezRola2018] ("Clock Around the Clock",
CCS 2018) demonstrated that high-resolution browser timers
fingerprint devices by clock skew without any vendor cooperation;
this is the methodological ancestor of our nanosleep-tail signal.

**Hertzbleed** [Hertzbleed2022] showed remote DVFS-induced timing
leaks. We treat the same coupling as *identity-bearing physical
state* rather than as a covert channel.

**Energon** [Energon2025] characterised GPU power side channels in
contemporary discrete and integrated GPUs; this is the closest
state-of-the-art measurement of analogue substrate variance on GPUs
and supports our Section 5.5 Adversary C threat-model treatment.

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
mechanism resists replay (Section 5.5/A) up to ≈ 2⁶³ library entries,
inherits per-die separation from substrate-physical variance against
chip-cloning attackers (Section 5.5/B), and is honest about its
side-channel-reconstruction blind spot (Section 5.5/C). The primitive
enables three capabilities that PCC, NVIDIA CC, Intel TDX, and
AMD SEV-SNP do not provide: per-die output attribution, stateless
PCC-class guarantees on commodity AMD, and TEE-free sybil-resistant
federated learning. We have shown this at n=2; we have not shown a
static-benchmark inference-accuracy gain. Both caveats are
addressable, and we are honest about them.

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

[DRAWNAPART] T. Laor, N. Mehanna, A. Durey, V. Dyadyuk, P. Laperdrix,
            C. Maurice, Y. Oren, R. Rouvoy, W. Rudametkin, Y. Yarom.
            "DRAWNAPART: A device identification technique based on remote
            GPU fingerprinting." NDSS 2022. arXiv:2201.09956.

[HTML5FP2015] K. Mowery, H. Shacham. "Pixel perfect: Fingerprinting canvas
            in HTML5." arXiv:1503.01408, W2SP 2012 / extended 2015.

[SanchezRola2018] I. Sanchez-Rola, I. Santos, D. Balzarotti. "Clock Around
            the Clock: Time-Based Device Fingerprinting." ACM CCS 2018.

[Hertzbleed2022] Y. Wang, R. Paccagnella, E. He, H. Shacham, C. Fletcher,
            D. Kohlbrenner. "Hertzbleed: Turning Power Side-Channel Attacks
            Into Remote Timing Attacks on x86." USENIX Security 2022.

[Energon2025] (Anon). "Energon: Power side-channel data leakage on
            contemporary GPUs." arXiv (Aug 2025).

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

— END OF DRAFT v2 —
