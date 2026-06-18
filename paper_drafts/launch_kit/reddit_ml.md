# r/MachineLearning — [P] post

**Subreddit:** r/MachineLearning
**Flair:** [P] Project
**Drop window:** T+72h (Friday US morning, after HN cycle).
**Angle for r/ML:** per-die output attribution + federated-learning
sybil resistance — frame as ML-deployment infrastructure, not crypto.

---

## Title

```
[P] FabricCrypt: per-die attestation for AI inference on commodity AMD — no vendor PKI, no TEE
```

---

## Body

We released **FabricCrypt** — a software-discoverable, vendor-key-free
per-die attestation primitive built on top of five HAL-bypass micro-
architectural signals. The motivation was an ML-deployment problem,
not a crypto problem:

**Problem.** When a model card claims "this output came from chip X,"
there is currently no software-only way to prove it came from *that
specific die* rather than *some other chip of the same SKU class*.
Apple PCC and NVIDIA Confidential Compute can attest "an H100 in CC
mode" but the attestation is SKU-class, not per-die. Two H100s with
identical firmware produce indistinguishable VCEK signatures.

**What we built.** A 290-dim live signature from (a) inter-core TSC
offsets, (b) cacheline ping-pong matrices, (c) DRAM-refresh-aligned
jitter, (d) nanosleep p99.9 tails, and (e) NVMe queue-tail latencies.
An audience-supplied 64-bit nonce drives the *sampling plan itself*
(which CPUs, which thermal zones, which core pairs, which sleep
durations), making replays infeasible.

**Results on n=2 AMD Ryzen AI Max+ 395 "Strix Halo" laptops:**

| Metric | Result |
|---|---|
| Per-die LOO classification | 100% (20 reps, gate p₀ > 0.95) |
| Anomaly-detection AUROC | 0.500 → 0.994 |
| Host-attribution accuracy | 0.501 → 1.000 |
| End-to-end sign-and-verify | median 1.12 ms, p99 2.79 ms |
| All 10 attack-battery gates | ✓ (including the O115 forgery) |

**Three downstream capabilities** that PCC / NVIDIA CC / TDX / SEV-SNP
do not provide:

1. **Per-die output attribution** — attach the signature to a model
   output, verify by challenge it originated on a specific die.
2. **Stateless PCC-equivalent guarantees on commodity AMD** — no
   Secure Enclave required.
3. **TEE-free sybil-resistant federated learning** — Sentinel-style
   guarantees without SGX.

**Honest caveats** (we put these in the body, not buried in §7):

- **n=2 chassis.** Reviewer attack #1 and we know it. Working on a
  6-chassis Strix Halo array.
- **No static-benchmark accuracy gain.** Phase 15/16 came back null;
  the capability gains are anomaly-detection and self-attribution
  where the signature is the input by construction.
- **Personality-attribution downstream: 66.4%.** Above chance, below
  ironclad. Not in our headline.
- **Bit-security is bounded.** ~60–80 bits against a source-code-aware
  attacker without K_chip; ~15–20 bits against an attacker who has
  captured K_chip. We are clear about this in §5.

**Repo (MIT):** https://github.com/Heigke/FabricCrypt
**Preprint:** arXiv:XXXX.XXXXX
**3-minute demo video:** [YouTube link]

Happy to answer questions. The most useful thing you could do for us
is *reproduce* on a third AMD chassis — the enrollment script is
~20 min and a fresh laptop is enough.
