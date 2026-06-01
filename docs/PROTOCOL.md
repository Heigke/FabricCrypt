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
