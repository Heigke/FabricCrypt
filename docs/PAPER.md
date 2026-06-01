# Paper

The full draft paper lives at
[`paper/fabriccrypt.md`](../paper/fabriccrypt.md). It is labelled
**Draft v2** and targets USENIX Security or ACM IH&MMSec.

The two anonymous chassis used in the paper are referred to as
`hostA` and `hostB`. Both are AMD Ryzen AI Max+ PRO 395 ("Strix Halo")
laptops with slightly different BIOS versions.

## What's in the paper

- §1 Introduction — vendor-PKI vs vendor-key-free attestation
- §2 Threat model
- §3 The 5 HAL-bypass signals
- §4 Per-die classification at N=2 (LOO=1.00, matched-governor)
- §5 Nonce-keyed protocol + 7-attack gates
- §6 Three new capabilities
- §7 Honest limitations (null static-benchmark accuracy gain, N=2)

## What we DON'T claim

- We do **not** claim a static-benchmark accuracy gain from embodiment.
  It was preregistered, tested, and came back null. See §7.
- We do **not** claim the residual side-channel reconstruction attack
  is hard. We have no construction; we also have no proof.
- We do **not** generalize beyond N=2 chassis. Help us scale via
  [`examples/publish_signature.sh`](../examples/publish_signature.sh).
