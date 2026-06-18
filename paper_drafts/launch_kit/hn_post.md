# Hacker News submission — FabricCrypt

**Submit:** Tuesday 07:00 PT (Wed 07:00 PT if HN front-page already
saturated on Tue). HN front-page-time is dominated by US morning;
weekday 07:00–08:30 PT is the well-known optimum window.

**Submitter:** post from your own account, not a fresh one — fresh
accounts get auto-shadow-banned on Show HN.

**URL field:** leave EMPTY. Post the GitHub URL in the body instead;
this keeps the title-link pointing at HN itself, which captures
better engagement on technical Show HN posts.

---

## Title (80-char limit)

```
Show HN: FabricCrypt – per-die AI attestation on commodity AMD without vendor PKI
```

(78 characters)

Alternate (if "Show HN" prefix causes mod issues):

```
FabricCrypt: per-die AI attestation on commodity AMD without vendor PKI
```

---

## Body

```
FabricCrypt is a software-discoverable, vendor-key-free per-die
attestation primitive (demonstrated at n=2 chassi) for commodity
AMD hardware. Repo:
https://github.com/Heigke/FabricCrypt — arXiv:XXXX.XXXXX.

The mechanism: bundle 15 signals total — 5 HAL-bypass μ-arch
(inter-core TSC offsets, cacheline ping-pong, DRAM-refresh-aligned
jitter, nanosleep p99.9 tails, NVMe queue-tail latency) + 3
cross-host KS-verified μ-arch (GPU clock jitter, multi-zone
thermal, Jacobian dynamics) + 7 board-level deterministic
(PCI/PCIe/USB/DMI/UCSI/amdgpu/kernel-boot) — into a 466-dim
live signature, and bind each challenge to a 64-bit nonce that
drives the sampling plan itself — which CPUs, which thermal
zones, which core pairs, which sleep durations. Replay attempts
collide with the wrong plan and get rejected.

We get 100% leave-one-out per-die classification on n=2 AMD Ryzen
AI Max+ 395 "Strix Halo" laptops at sub-millisecond sign-and-verify
(median 1.12 ms, p99 2.79 ms). All ten gates in an extended attack
battery pass — including a fatal forgery (O115) that defeated our
v2.0. Residual unforgeability is reported as **empirical
attack-cost**, NOT a formal cryptographic reduction: ≈10⁹–10¹²
samples for a source-code-aware attacker without K_chip, ≈10⁴–10⁶
samples if K_chip is leaked. Tier-2 (Reverse-FE + Controlled-PUF
wrap) raises the empirical modeling-attack floor to ≥10¹² samples
returning random-Hamming distance. We are explicit about this
"empirical, not proven" framing in §5.10.5 of the paper.

What this gives you that PCC, NVIDIA CC, TDX, and SEV-SNP do not:
per-die output attribution, stateless PCC-equivalent guarantees on
commodity AMD (no Secure Enclave, no vendor CA), and TEE-free
sybil-resistant federated learning.

Honest caveats up front: n=2 chassis. Empirical operating points,
no formal cryptographic reduction. No static-benchmark inference
accuracy gain (Phase 15/16 came back null). Exploratory
stylometric divergence from chip-conditioned training (§7.L6) is
supplementary, not a headline claim. A persistent kernel adversary
is unmitigated at the protocol level.

If you have an AMD Strix Halo / Ryzen AI 300 laptop, please run
the enrollment script and post the signature; we'd love a third
chassis in the public corpus.
```

---

## First-comment Q&A pre-write

Paste this as your own first comment within ~2 minutes of the
submission. HN convention rewards authors who answer in the same
thread. Top 10 questions anticipated:

> **Q1: How is this not just a PUF with extra steps?**
>
> Classical PUFs (SRAM, ring-oscillator, DRAM-latency) produce a
> *static* fingerprint. FabricCrypt is challenge-bound and live:
> the audience nonce decides which subset of CPUs, thermal zones,
> core pairs, and sleep durations get sampled. A recording of
> yesterday's response is useless against today's nonce because
> the underlying sample plan is different.

> **Q2: Why should I trust five timing signals to be stable?**
>
> Inter-core TSC offsets and MOESI ping-pong are dominated by
> post-binning wire-routing skew — picosecond-level interconnect
> physics, not microcode. The DRAM-refresh-aligned latency window
> exploits per-die refresh-controller PVT skew. We chose these
> five specifically because they survive `cpufreq` governor
> matching (re-run reported in §4.4 of the paper).

> **Q3: Isn't n=2 ridiculous?**
>
> Yes, and we say so in §7. n=2 is sufficient to demonstrate the
> primitive *exists* in a software-discoverable way; it is not
> sufficient to bound the false-positive rate at production scale.
> If you have AMD Strix Halo silicon, please run our enrollment
> script and post `<hostname>_sig_v2.npz` — we'll fold it in.

> **Q4: This is DRM, right? You're locking AI to chips.**
>
> Fair concern. Three answers. (a) Unlike Apple PCC, no vendor key
> is required — you generate your own per-die identity on your
> own hardware, no enrollment with us. (b) FabricCrypt proves *who
> ran the inference*; it does not restrict *what is allowed to
> run*. Your bootloader is untouched. (c) The motivating use cases
> are output attribution for AI liability and sybil resistance in
> federated learning — both protect users from impersonation. We
> address this in §6 of the paper.

> **Q5: An attacker with kernel root can spoof all five signals.**
>
> True — and we say so under L5. The protocol assumes
> `SCHED_FIFO`/`mlockall`/`cpuset` isolation and preemption
> disabling around critical sections; persistent kernel adversaries
> are not in our threat model. PCC has the same caveat (compromised
> Secure Enclave defeats PCC).

> **Q6: What about Hertzbleed / Energon-style remote reconstruction?**
>
> Adversary C in §5.5. A remote attacker who can reconstruct your
> fingerprint via DVFS or power side channels can build a "phys
> vector" library but cannot use it across a fresh nonce: each
> challenge re-randomises the sampling plan. We don't claim the
> fingerprint is secret — only that *re-use* of an observed
> fingerprint is rejected.

> **Q7: K_chip enrollment requires a "physically-secure channel."
> Doesn't that defeat the point?**
>
> Same trust-bootstrapping requirement as Apple PCC enrollment, TPM
> EK provisioning, or SSH host-key first-use. The win is that K_chip
> is a *per-die* secret derived from the chip's own calibration,
> not a vendor-signed certificate. Once enrolled, no third party is
> in the loop.

> **Q8: Empirical attack-cost ≈10⁹–10¹² samples sounds weak. AES is 2^128.**
>
> Right — and we are explicit that these are **empirical operating
> points, not formal cryptographic reductions**. There is no proven
> bit-security claim. Tier-1 fixes (current 14D) get us past the O115
> forgery. Tier 2 (Reverse-FE + Controlled-PUF wrap, TPM-sealed
> K_chip, distance-bounding) raises the empirical modeling-attack
> floor to ≥10¹² samples returning random Hamming distance. Tier 3
> (multi-round protocol with tight per-round RTT) is future work.
> We're explicit about all this in §5.8 and §5.10.5.

> **Q9: How does this differ from DRAWNAPART?**
>
> DRAWNAPART is GPU-execution-unit timing fingerprinting from a
> WebGL sandbox. It demonstrates that nominally identical GPUs are
> distinguishable. FabricCrypt is the CPU-side analogue plus a
> nonce-driven sampling plan and a multi-signal bundle. Spiritually
> the same family; methodologically more aggressive (kernel-level
> HAL bypass).

> **Q10: When can I reproduce?**
>
> Now. Repo at https://github.com/Heigke/FabricCrypt is MIT-
> licensed. Enrollment script is a single Python file; you need
> root and an AMD Ryzen AI 300 / Strix Halo APU. Targeted
> reproduction time: ~20 minutes on a fresh laptop. If you hit a
> snag, open an issue and tag me.

---

## After the post

- Monitor HN front-page for ~3 hours. Answer every top-level
  technical comment within 15 minutes.
- Do NOT argue with downvoters. Address the technical substance.
- If the post hits front-page top-10, schedule the Reddit drops
  (T+72h) — do not cannibalise.
