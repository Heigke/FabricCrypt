# The Phase 14C nonce-keyed protocol, explained

## Why an earlier static-signature protocol fails

The 290-dim `signature_v2` vector (Phase 13) is unique per die, but it
is *static under a fixed sampling configuration*. An adversary who
records one such vector from the target chip can replay it forever:

> static_replay_p0_rate = **1.00** under the Phase 14B protocol.

That made the static-signature protocol unsuitable as an attestation
primitive, even though it was sufficient for forensic *identification*.

## The fix: nonce drives the sampling plan

In Phase 14C the audience supplies a 64-bit nonce. We derive a
deterministic sampling plan from `HMAC-SHA256(nonce)`:

```python
plan = derive_plan(nonce, n_cpus, n_zones)
# plan contains:
#   cpu_subset   (4 distinct CPU indices)
#   zone_subset  (up to 3 thermal-zone indices)
#   core_pairs   (2 c2c ping-pong pairs)
#   ns_sleep     (target nanosleep duration, 1000..8000 ns)
#   ns_count     (4..10 sleeps)
#   tsc_count    (4..10 TSC bursts)
#   perm         (32-element output permutation)
```

`NonceSig.read(nonce)` then performs a 32-dim physical measurement
*under that specific plan*, applies `plan['perm']` to it, and
concatenates a 32-dim deterministic embedding of the nonce. The result
is a 64-dim float32 vector.

## Why this defeats static replay

The pre-permutation index 31 of the physical vector holds the
`ns_sleep` value used during the read. After permutation it lands at
output position `perm.index(31)`, which is different for each nonce.

The verifier knows the nonce, can compute `plan(nonce)`, can invert the
permutation, can look up the value at the right output position, and
can compare it to `log1p(ns_sleep * 1e-3)`. A static replay attacker
holds a vector whose ns_sleep landed at *the wrong* position relative
to the current nonce — the deterministic plan-consistency check
catches it with near-perfect resolution.

In the paper:

| attack                              | gate    | observed |
|-------------------------------------|---------|----------|
| static_replay_no_nonce              | ≤ 0.05  | 0.006    |
| dynamic_replay (library + nearest)  | ≤ 0.10  | 0.012    |

## Why we also include a classifier

The deterministic plan-consistency check is the **hard gate**. We also
train a `TwinMLP` on `[phys ; nonce_emb]` to give a soft probability
that this signature came from *this chip* (vs a foreign chip). The
classifier is used as a diagnostic and for the cross-chip (twin) task.
The classifier alone cannot defeat all replay attacks; only combined
with the deterministic gate does the full protocol pass.

## Sign-and-verify latency

End-to-end (read on chip → wire → verify on remote) measured in the
paper:

- median: 1.12 ms
- p99: 2.79 ms

The verification step itself is sub-microsecond (one HMAC + one float
compare).

## What the protocol does *not* guarantee

1. **No protection against a physically present attacker with the same
   silicon die.** This is fundamental: the fingerprint *is* the die. An
   attacker with the original chip can forever sign for it.
2. **No protection against side-channel cloning of the fingerprint
   that's strong enough to predict the response to an arbitrary
   nonce.** We have not constructed such an attack and we don't believe
   it's cheap, but we also haven't disproved its existence. See §5.5 of
   the paper for the three-class adversary analysis.
3. **No protection against PSP firmware that decides to fake thermal /
   RAPL reads.** If you don't trust the AMD PSP, FabricCrypt's
   guarantees degrade to "this came from *some* AMD die running the
   stock PSP firmware".

## Pseudocode summary

```python
# Audience side
nonce = secrets.token_bytes(8)
send(nonce)

# Chip side
sig = NonceSig.read(nonce, raw=True)   # 64-dim float32
send(sig)

# Verifier side
plan = derive_plan(nonce, n_cpus, n_zones)
score = plan_consistency_score(sig[:32], nonce, n_cpus, n_zones)
accept = score > 0.5
```

---

## Tier 2 cryptographic hardening

The Phase 14C base protocol described above provides a useful but limited
bit-security floor.  Tier 2 (modules in `src/protocol/`: `reverse_fuzzy`,
`controlled_puf`, `multiround_protocol`, `zk_inference_binding`) raises
that floor substantially while remaining additive — the base protocol is
unchanged.

### Bit-security comparison

| Threat vector                                          | Phase 14C / O115 | Tier 2     | Tier 2 mechanism                |
|--------------------------------------------------------|------------------|------------|---------------------------------|
| V0  Brute-force K_chip (256-bit secret)                | 2^256            | 2^256      | unchanged                       |
| V1  Fingerprint brute-force                            | 2^30 — 2^40      | 2^60 — 2^80| RFE quantization + BCH(t=16)    |
| V2  ML modeling attack (Ruehrmair CCS'10)              | low              | 2^128      | Controlled-PUF wrap (T2.2)      |
| V3  Generative attacker w/ 10^5 (nonce, response)      | low              | 2^128      | Wrapped output is random oracle |
| V4  K_chip leak + replay                               | 2^15 — 2^20      | 2^40 — 2^60| Multi-round protocol (T2.3)     |
| V5  Helper-data leakage (classical FE)                 | ~ 2^|P|          | 2^256      | RFE keeps P private (T2.1)      |
| V6  Relay attack                                       | 2^0              | 2^0        | NOT mitigated — see gap         |
| V7  Chosen-challenge ML                                | low              | 2^128      | SHAKE256 H_in (T2.2)            |

**Headline:** unprotected security 2^30 — 2^40 -> **2^60 — 2^80** (a >2^30
improvement); with-K_chip-leak 2^15 — 2^20 -> **2^40 — 2^60**.

### T2.1 Reverse Fuzzy Extractor (Van Herrewege et al., FC'12)

Classical Dodis-Reyzin-Smith fuzzy extractors publish helper data P,
which leaks roughly |P| bits about the chip fingerprint w_ref.  The
reverse construction inverts the role: the verifier keeps (w_ref, P, K)
private, and the prover sends only a fresh noisy reading w_noisy.  The
verifier computes the syndrome of (w_ref XOR w_noisy) under a BCH code
and accepts iff it decodes within radius t.

We use BCH(t=16, m=8) with N=256 bit codeword; this corrects 16 bit
flips (~6% Hamming radius).  Empirical results on Phase 13 paired sigs:

- intra-host accept: 10/10 (mean Hamming distance ~ 37 of 256)
- inter-host accept: 0/10
- random-imposter accept: 0/100

See `results/tier2_security/rfe_offline_results.json`.

### T2.2 Controlled PUF (Suh-Devadas, DAC'07) — defeats ML modeling attack

A raw PUF exposes its challenge/response interface directly: an attacker
who observes many (C, R) pairs can train an MLP / kernel regressor that
forges responses to unseen challenges (Ruehrmair et al., CCS'10 on
arbiter PUFs).  We wrap the raw FabricCrypt signature in two SHAKE256
hash layers with strict domain separation:

```
external_nonce  --H_in-->  inner_nonce  --raw_PUF-->  raw_response
                                                         |
external_nonce ---------------------------H_out--------> wrapped_response
                              (also bound to chip_id)
```

The wrapped response is uniformly random by the random-oracle
assumption on SHAKE256.  We empirically verified ML attack failure:

- **Attack-A** (uncontrolled, raw sig): linear/MLP reach Pearson r̄ ≈ 0.47
  at N=160 training pairs; scaling N -> 10^5 would push r̄ above the
  0.85 forgery threshold (so uncontrolled bit-security shrinks with data).
- **Attack-B** (controlled-PUF wrapped): linear/MLP produce
  Hamming distance ~ 128/256 from the true wrapped output — i.e.
  bit-flip random.  **0/40 forgery rate at every N tested.**

See `results/tier2_security/adversary_modeling_attack.json`.

### T2.3 Multi-round protocol (3 rounds)

The base Phase 14C protocol is single-round (nonce -> 64-dim aggregate).
A K_chip-leak adversary who learns the (nonce -> aggregate) map can
replay summaries.  T2.3 forces the prover to commit to **50 raw
micro-samples** S before knowing the verifier's constraints:

```
R1: V -> P : nonce
    P -> V : commit_S = SHA256("commit-S" || nonce || S_bytes)
R2: V -> P : c_1..c_5 (5 SHAKE-derived constraints on subsets of S)
R3: P -> V : t_1..t_5 = f_k(S) and open(S)
    V verifies: commit_S matches, all 5 |t_k - f_k(S)| < eps_k
```

Constraints are post-hoc subset-aggregations (median, variance, lag-1
autocorr, count-above-quantile, weighted sum) over SHAKE-derived
subsets.  A K_chip-leak adversary can sample matching aggregates from
the population but cannot match the *per-die* moments of the specific
target chip without physical access.

### T2.4 ZK inference binding

For the "is this output actually produced by program M on chip-bound
state S?" question, T2.4 provides an honest cryptographic commitment
(Pedersen-style com_S = SHA256("chip-sig-com-v1" || r || S), bound to
inference via HMAC(K_chip, com_S || y || prog_hash || x)).  The
verifier interface is shaped identically to a future zk-SNARK
(ezkl/Halo2/Risc0) drop-in: today it works by REPLAY (verifier opens
the commitment and re-runs M); tomorrow the replay step is replaced
with a succinct constraint check.

### Defense against Ruehrmair-style ML modeling

The Tier 2 controlled-PUF wrap is the *direct* mitigation of the
classical PUF-modeling attack family (Ruehrmair-Solter CCS'10 on
arbiter PUFs, Tobisch-Becker COSADE'15 on XOR-arbiter PUFs).  Two
properties are essential:

1. The adversary cannot choose `inner_nonce` (mangled through H_in
   with the chip's secret K_chip and ID).  This eliminates the
   *chosen-challenge* power that drives modeling-attack convergence.
2. The output `wrapped_response` is SHAKE256-derived and inherits the
   random-oracle assumption.  Even an attacker who somehow recovers
   `raw_response` must still invert SHAKE256 to forge `wrapped_response`.

### Gap: relay attacks (V6)

Tier 2 does NOT mitigate a real-time relay attacker who has physical
access to the chip and answers challenges as fast as the chip itself.
Closing V6 would require RTT distance bounding (Brands-Chaum 1993),
which is a future-work item.
