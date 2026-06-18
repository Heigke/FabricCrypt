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

