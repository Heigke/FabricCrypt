# FAQ

### Why not just use the TPM / SEV-SNP / TDX / NVIDIA CC?

All of those tie attestation to a *vendor* signing key. If you don't
trust Apple / Intel / AMD / NVIDIA's CA, you don't have attestation.
FabricCrypt's primitive needs no vendor key material — at the cost
of strictly weaker formal guarantees.

### What does "per-die" actually buy you?

VCEK and friends authenticate the SKU class — "an H100 in confidential
mode". They cannot distinguish two H100s. FabricCrypt distinguishes
individual dies. Use cases that need this:

1. **AI output attribution.** Prove that this inference came from
   chip X and not some other chip of the same SKU.
2. **Sybil-resistant federated learning** without TEEs.
3. **Per-die anomaly detection** (degraded chip vs healthy chip).

### How big is the dataset?

N = 2 chassis in the current paper. We're explicitly inviting
community contributions to scale this. See
[`examples/publish_signature.sh`](../examples/publish_signature.sh).

### Why does collection take 30+ seconds per capture?

Mostly thermal pauses on small chassis. On a desktop with strong
cooling and raised thresholds you can drop this to ~10 s.

### Can I run this on Intel?

Probably not in the current form. The C helpers (`tsc_inter_core`,
`cacheline_pingpong`) use `rdtscp`, which works on Intel — but the
distributional signal we exploit comes from AMD-specific core/CCX
topology and from `intel-rapl` semantics on AMD platforms. Patches to
support Intel are welcome.

### Why is the demo classifier optional?

The deterministic plan-consistency check is sufficient for replay
defense. The neural classifier is only needed for cross-chip
(twin) discrimination — and only once you have a peer's
`paired_sigs.npz` to train against. The demo defaults to no classifier
so it works out-of-the-box on a single machine.

### Is this peer-reviewed?

No. The paper draft (`paper/fabriccrypt.md`) is in this repo; it is
labelled **Draft v2** and targets USENIX Security / ACM IH&MMSec. Treat
the claims accordingly.

### What's the residual risk I should care about?

Three failure modes, in increasing order of "how hard is it for an
adversary":

1. **Trivial:** static replay of a stale signature — **defeated** by
   the nonce-keyed plan-consistency gate.
2. **Hard:** library-based replay — **defeated up to ≤ 1.2%** in our
   experiments (gate is ≤ 10%).
3. **Open:** side-channel reconstruction of the fingerprint function
   itself, allowing the adversary to *predict* the response to an
   arbitrary nonce. We have no evidence this is cheap; we also have no
   proof it's hard. See §5.5 of the paper.

### License?

MIT.
