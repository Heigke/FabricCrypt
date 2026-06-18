# PRESS RELEASE — for immediate release

**Date:** 2026-06-XX
**Contact:** Eric Bergvall, bergvall.eric@gmail.com
**Embargo:** none (companion preprint and code already public)

---

## FabricCrypt: independent researcher demonstrates per-die AI attestation on commodity AMD laptops — no vendor key required

**Stockholm, Sweden** — An independent researcher has released
**FabricCrypt**, the first open-source attestation primitive that
binds AI inference to a specific physical processor die on commodity
AMD hardware, without any vendor key, Secure Enclave, or TPM
certificate. The work was released today as an MIT-licensed code
repository and an arXiv preprint.

The current state of the art — Apple Private Cloud Compute, NVIDIA
Confidential Compute, Intel TDX, AMD SEV-SNP — all root trust in the
silicon vendor's public-key infrastructure, and all authenticate the
SKU class ("an H100 in confidential mode"), not the individual chip.
Two H100s with identical firmware produce indistinguishable
attestations. FabricCrypt is different on both axes: no vendor PKI,
and per-die identity.

### The demo

Two AMD Ryzen AI Max+ 395 "Strix Halo" laptops with identical SKU,
identical RAM, identical BIOS, and identical microcode. The same AI
model file runs on laptop A and the verifier accepts. The file is
copied to laptop B with `scp`. The same code, same model, same prompts
— on laptop B the verifier rejects. The chip itself is the root of
trust.

### How it works

FabricCrypt bundles five micro-architectural signals into a
290-dimensional live device signature:

- inter-core timestamp-counter (TSC) offsets — picosecond-level
  wire-routing skew across cores;
- cacheline ping-pong matrices — MOESI cache-coherence latency;
- DRAM-refresh-aligned jitter — phase-locked memory timing;
- nanosleep p99.9 tails — kernel-scheduler micro-jitter;
- NVMe queue-tail latencies — per-namespace submission noise.

None of these are factory-programmed. All arise from post-binning
silicon variation that differs across nominally identical chips. A
64-bit audience nonce drives the sampling plan — *which* cores,
*which* thermal zones, *which* core pairs, *which* sleep durations
get measured — so that a recording of yesterday's response cannot
be replayed against today's challenge.

### Key numbers

- **100%** per-die leave-one-out classification at n=2 chassis,
  20 reps.
- **median 1.12 ms / p99 2.79 ms** end-to-end sign-and-verify
  latency.
- **All ten gates** in an extended attack battery pass, including
  a post-disclosure forgery attack ("O115") that defeated an
  earlier version of the protocol.

### Honest limitations

The researcher's preprint explicitly enumerates what was NOT shown:

- Only two chassis (n=2) were tested; reviewer scrutiny will rightly
  demand replication on six or more.
- No static-benchmark inference-accuracy gain was observed (the
  experiment was run and came back null).
- Personality-attribution downstream task achieved 66.4% accuracy —
  above chance, but not ironclad.
- Bit-security against a calibration-file-capture attacker is
  bounded at approximately 15–20 bits.

### Capabilities enabled

Three deployment scenarios that vendor-PKI-rooted attestation cannot
provide:

1. **Per-die AI output attribution** — attach the signature to a
   model output and verify by challenge which physical die produced
   it.
2. **Stateless PCC-equivalent guarantees on commodity AMD** — no
   Secure Enclave or vendor CA required.
3. **TEE-free sybil-resistant federated learning** — same guarantee
   as SGX-rooted Sentinel without SGX.

### Demo video and reproduction

3-minute demo video: [YouTube URL]
60-second cut: [Twitter URL]
Repository (MIT-licensed): https://github.com/Heigke/FabricCrypt
Preprint: arXiv:XXXX.XXXXX

Anyone with an AMD Ryzen AI 300 or Strix Halo laptop can reproduce
the enrollment in approximately 20 minutes.

### About the researcher

Eric Bergvall is an independent researcher working on hardware-rooted
identity for AI inference. He is reachable at bergvall.eric@gmail.com
and welcomes replication, criticism, and contributions to the public
chassis corpus.

###
