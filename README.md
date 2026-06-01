# FabricCrypt

**Software-discoverable, vendor-key-free per-die attestation on commodity
AMD hardware.**

FabricCrypt assembles a 290-dim per-die fingerprint from five HAL-bypass
micro-architectural signals (inter-core TSC offsets, cacheline ping-pong,
DRAM-refresh jitter, syscall tail, NVMe tail) and binds it to a 64-bit
audience nonce that drives the *sampling plan itself* (which CPUs, which
thermal zones, which core pairs, which sleep durations are read). The
result is a stateless challenge-response primitive that:

- distinguishes individual dies of the same SKU (100% LOO on N=2 AMD
  Ryzen AI Max+ PRO 395 "Strix Halo" laptops),
- rejects static and dynamic-library replays (≤5% / ≤10% gates),
- needs no Secure Enclave, no TPM EK, no vendor key material,
- runs end-to-end in sub-millisecond (median 1.12 ms, p99 2.79 ms).

### Extended signature (Phase 19 + Phase 22)

In addition to the base 290-dim signature, `src/signature/` ships eleven
optional additive modules covering 198 further dims (488-dim extended
signature, or ~466-dim after pruning constant features):

- **Phase 19** (3 cross-host KS-verified signals, p_bonf < 0.01):
  `gpu_clock_jitter`, `thermal_spread`, `jacobian_dynamics`.
- **Phase 22** (8 light deterministic discovery-class signals):
  `pci_topology`, `pcie_link_state`, `usb_descriptor`, `dmi_smbios`,
  `kernel_boot_timing`, `ucsi_descriptors`, `amdgpu_safe_reads`,
  `hpet_drift`.

Each module exposes a uniform `run(reps, out_dir)` and saves a
`(reps, dim)` `.npz`.  Leave-one-out 1-NN accuracy on the full extended
signature: **1.00 on N=2** (Phase 19 ablation).

### Tier 2 cryptographic hardening (60-80 bits, ML modeling defeated)

The Phase 14C base protocol provided ~2^30 — 2^40 unprotected bit-security.
`src/protocol/` now adds four Tier-2 modules that lift unprotected security
to **~2^60 — 2^80** and reduce the with-K_chip-leak attack to ~2^40 — 2^60:

- `reverse_fuzzy` — Van Herrewege FC'12 reverse fuzzy extractor over
  BCH code-offset (helper data kept PRIVATE on the verifier — closes
  the classical-FE helper-leakage channel).
- `controlled_puf` — Suh-Devadas DAC'07 controlled-PUF wrap with
  SHAKE256 H_in/H_out and strict domain separation (defeats the
  Ruehrmair CCS'10 ML modeling attack: 0/40 forgery rate at N=160
  training pairs).
- `multiround_protocol` — 3-round commit/challenge/open protocol over
  50 raw micro-samples + 5 SHAKE-derived post-hoc constraints (forces
  adversary to emulate the chip's full per-die noise process).
- `zk_inference_binding` — Pedersen-style commitment + HMAC inference
  tag, interface-compatible with a future zk-SNARK swap-in.

Bit-security analysis and ML-modeling-attack results are in
[`results/tier2_security/`](results/tier2_security/) and
[`docs/PROTOCOL.md`](docs/PROTOCOL.md) "Tier 2".

### Honest non-results

A separate Phase 21b stylometry investigation (Linux build artefact ->
per-host attribution) reached **66.4% accuracy** — better than chance
(50%) but well below a usable signal. Reported here for honesty.

This repo is the reproduction package. See [`paper/fabriccrypt.md`](paper/fabriccrypt.md)
for the full draft.

---

## Quick start

```bash
git clone git@github.com:Heigke/FabricCrypt.git
cd FabricCrypt
./scripts/00_install_deps.sh                            # one-time
source venv/bin/activate

./scripts/01_collect_signature.sh --reps 10             # ~7-8 min capture
# repeat on a second machine, copy its data/<host>_sig_v2.npz back

./scripts/02_classify.sh data/hostA_sig_v2.npz data/hostB_sig_v2.npz
# -> LOO accuracy printed; gate passes at >0.95

./scripts/04_demo.sh
# -> http://127.0.0.1:8770  (issue challenge / sign / verify / try stale replay)
```

Full step-by-step for a multi-machine reproduction:
[`docs/HOW_TO_REPRODUCE.md`](docs/HOW_TO_REPRODUCE.md).

---

## Repo layout

```
FabricCrypt/
├── README.md           ← you are here
├── LICENSE             ← MIT
├── CITATION.cff
├── docs/               ← reproduction guide, protocol, hardware, FAQ
├── src/
│   ├── signature/      ← 5-signal 290-dim extractor (Phase 12+12B+13)
│   ├── protocol/       ← nonce-keyed challenge-response (Phase 14C)
│   ├── demo/           ← FastAPI sign/verify webapp
│   └── analysis/       ← cross-chassis LOO + PCA
├── scripts/            ← 00..04 numbered repro shell scripts
├── examples/           ← two-chassis collect + publish-your-signature
├── data/               ← captured signatures land here at runtime
├── paper/              ← draft paper + figures
├── requirements.txt
└── .gitignore
```

---

## What this is **not**

- **Not** a static-benchmark accuracy gain. We prereg-tested it; it
  came back null. We discuss this honestly in
  [`docs/PAPER.md` §7](docs/PAPER.md).
- **Not** an Apple PCC replacement at scale. PCC binds to a *vendor*
  signing key inside a Secure Enclave; FabricCrypt operates without
  one but inherits a different residual-risk profile (see
  [`docs/PROTOCOL.md`](docs/PROTOCOL.md)).
- **Not** validated beyond n=2 chassis. We invite community contributions —
  see [`examples/publish_signature.sh`](examples/publish_signature.sh).

---

## Hardware

Currently tested on AMD Ryzen AI Max+ PRO 395 (Strix Halo, gfx1151)
in HP Z2 mini G1a chassis. Should also work on:

| Platform                                | Likelihood | Notes |
|----------------------------------------|------------|-------|
| HP Z2 mini G1a + Ryzen AI Max+ PRO 395 | tested     | our baseline (N=2) |
| Other AMD Strix Halo APU systems       | likely     | thermal thresholds may differ |
| AMD Zen 5 desktops                     | likely     | raise thermal thresholds in `example.env` |
| Intel CPUs                             | won't work | RDRAND / syscall paths assume AMD perf counter behaviour |

See [`docs/HARDWARE_REQUIREMENTS.md`](docs/HARDWARE_REQUIREMENTS.md).

---

## License

MIT. See [`LICENSE`](LICENSE).
