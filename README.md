# FabricCrypt

[![CI](https://img.shields.io/badge/CI-pending-lightgrey)](https://github.com/Heigke/FabricCrypt/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-v0.2-blue)](https://github.com/Heigke/FabricCrypt/releases)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b)](https://arxiv.org/abs/XXXX.XXXXX)
[![Reproducible: N=2](https://img.shields.io/badge/reproducible-N%3D2%20chassis-orange)](docs/HOW_TO_REPRODUCE.md)

**Per-die attestation for AI inference on commodity AMD silicon — no vendor PKI, no Secure Enclave, no TPM.**

> "First per-die attestation on commodity AMD without vendor PKI."

FabricCrypt assembles a per-die fingerprint from five HAL-bypass micro-architectural signals (inter-core TSC offsets, cacheline ping-pong, DRAM-refresh-aligned jitter, syscall tail, NVMe tail) and binds each challenge to a 64-bit audience nonce that drives the *sampling plan itself* — which CPUs, which thermal zones, which core pairs, which sleep durations are read. The result is a stateless challenge-response primitive that:

- distinguishes individual dies of the same SKU (**100% LOO** on N=2 AMD Ryzen AI Max+ PRO 395 "Strix Halo" laptops),
- rejects static and dynamic-library replays (≤5% / ≤10% gates),
- needs no Secure Enclave, no TPM EK, no vendor key material,
- runs end-to-end in **sub-millisecond** (median 1.12 ms, p99 2.79 ms).

> ⚠️ **Phase 21b honest caveat.** A separate stylometry investigation (Linux build-artefact → per-host attribution) reached **66.4%** accuracy on a held-out classifier. That is above chance (50%) but well below ironclad. We do **not** claim the chip *causes* the AI's output style; we report a measurement, not a metaphysics claim. See [`docs/PAPER.md` §7, L6](docs/PAPER.md) and `results/phase21b/`.

---

## What's new in v0.2

- **Phase 19 — three cross-host KS-verified signals** (`gpu_clock_jitter`, `thermal_spread`, `jacobian_dynamics`), p_bonf < 0.01.
- **Phase 22 — eight light deterministic discovery-class signals** (`pci_topology`, `pcie_link_state`, `usb_descriptor`, `dmi_smbios`, `kernel_boot_timing`, `ucsi_descriptors`, `amdgpu_safe_reads`, `hpet_drift`).
- Extended signature lifts dimensionality from 290 → **488 dims** (~466 after pruning constant features); LOO 1-NN remains **1.00** at N=2.
- **Tier-2 cryptographic hardening** — reverse fuzzy extractor (Van Herrewege FC'12), controlled-PUF wrap (Suh–Devadas DAC'07), 3-round commit/challenge/open multi-round protocol, and zk-SNARK-ready Pedersen + HMAC inference binding. Unprotected bit-security lifted from ~2^30–2^40 to **~2^60–2^80**.
- ML-modeling attack defeated: **0/40 forgeries** at N=160 training pairs (Rührmair CCS'10 reproduction baseline).
- O115 custom-forgery break patched (real measurement at dim 31, keyed plan derivation, Mahalanobis gate, independent SHAKE256 streams per plan component).

Full notes: [`docs/CHANGELOG.md`](docs/CHANGELOG.md). Bit-security analysis: [`results/tier2_security/`](results/tier2_security/). Protocol details: [`docs/PROTOCOL.md`](docs/PROTOCOL.md) "Tier 2".

---

## Media

- **60-s demo video** — transplant moment: same model file, chassis A passes, chassis B is rejected, A passes again. See `demo_video/twitter_60s.mp4` (also linked in launch tweet).
- **One-pager PDF** — `paper/fabriccrypt_one_pager.pdf` *(coming with arXiv submission)*.
- **Long-form 3-min explainer** — YouTube link *(to be added at launch T-0)*.

---

## Quick start

```bash
git clone git@github.com:Heigke/FabricCrypt.git
cd FabricCrypt
./scripts/00_install_deps.sh                            # one-time
source venv/bin/activate

# 1. Capture an extended signature (Phase 12 + 19 + 22)
./scripts/01_collect_signature.sh --reps 10             # ~7-8 min capture

# 2. Repeat on a second machine, copy data/<host>_sig_v2.npz back

# 3. Classify across chassis (base 290-dim + extended 488-dim)
./scripts/02_classify.sh data/hostA_sig_v2.npz data/hostB_sig_v2.npz

# 4. Tier-2 ML-modeling-attack reproduction (Rührmair CCS'10 baseline)
./scripts/03_tier2_modeling_attack.sh                   # ~3 min, 0/40 expected

# 5. End-to-end demo: issue / sign / verify / try stale replay
./scripts/04_demo.sh                                    # http://127.0.0.1:8770
```

Full multi-machine walkthrough: [`docs/HOW_TO_REPRODUCE.md`](docs/HOW_TO_REPRODUCE.md).

---

## Comparison vs prior attestation primitives

| System                  | Per-die identity | Vendor PKI required | Open-source verifier | Commodity hw | Sub-ms verify |
|-------------------------|:----------------:|:-------------------:|:--------------------:|:------------:|:-------------:|
| Apple PCC               | ✗ (SKU-class)    | ✓                   | ✗                    | ✗            | n/a           |
| NVIDIA CCM (Hopper CC)  | ✗ (SKU-class)    | ✓                   | partial              | ✗            | n/a           |
| Intel TDX               | ✗ (SKU-class)    | ✓                   | ✗                    | ✗            | n/a           |
| AMD SEV-SNP             | ✗ (SKU-class)    | ✓                   | ✗                    | partial      | n/a           |
| TPM 2.0 EK              | ✓ (factory key)  | ✓ (vendor CA)       | partial              | ✓            | ms-range      |
| **FabricCrypt v0.2**    | **✓ (silicon)**  | **✗**               | **✓**                | **✓**        | **1.12 ms**   |

Trade-offs and a Phase-21b caveat row are detailed in [`docs/COMPARISON.md`](docs/COMPARISON.md).

---

## Repo layout

```
FabricCrypt/
├── README.md           ← you are here
├── LICENSE             ← MIT
├── CITATION.cff
├── docs/               ← reproduction guide, protocol, hardware, FAQ, comparison
├── src/
│   ├── signature/      ← 5-signal 290-dim extractor + Phase 19 + Phase 22 modules
│   ├── protocol/       ← nonce-keyed challenge-response + Tier-2 (reverse fuzzy,
│   │                     controlled-PUF, multiround, zk-binding)
│   ├── demo/           ← FastAPI sign/verify webapp
│   └── analysis/       ← cross-chassis LOO + PCA + Tier-2 attack harnesses
├── scripts/            ← 00..04 numbered repro shell scripts
├── examples/           ← two-chassis collect + publish-your-signature
├── data/               ← captured signatures land here at runtime
├── results/            ← tier2_security/, phase21b/, etc.
├── paper/              ← draft paper + figures + one-pager
├── demo_video/         ← twitter_60s.mp4, transplant_6s.gif
└── requirements.txt
```

---

## What this is **not**

- **Not** a static-benchmark accuracy gain. We prereg-tested it; it came back null. Discussed in [`docs/PAPER.md` §7](docs/PAPER.md).
- **Not** an Apple PCC replacement at scale. PCC binds to a *vendor* signing key inside a Secure Enclave; FabricCrypt operates without one but inherits a different residual-risk profile (see [`docs/PROTOCOL.md`](docs/PROTOCOL.md)).
- **Not** validated beyond **N=2** chassis. We invite community contributions — see [`examples/publish_signature.sh`](examples/publish_signature.sh). Bring a third chassis.

---

## Hardware

Currently tested on AMD Ryzen AI Max+ PRO 395 (Strix Halo, gfx1151) in HP Z2 mini G1a chassis. Should also work on:

| Platform                                | Likelihood | Notes |
|----------------------------------------|------------|-------|
| HP Z2 mini G1a + Ryzen AI Max+ PRO 395 | tested     | our baseline (N=2) |
| Other AMD Strix Halo APU systems       | likely     | thermal thresholds may differ |
| AMD Zen 5 desktops                     | likely     | raise thermal thresholds in `example.env` |
| Intel CPUs                             | won't work | RDRAND / syscall paths assume AMD perf counter behaviour |

See [`docs/HARDWARE_REQUIREMENTS.md`](docs/HARDWARE_REQUIREMENTS.md).

---

## Citing this work

If FabricCrypt is useful in your research, please cite the preprint:

```bibtex
@article{fabriccrypt2026,
  title   = {FabricCrypt: Software-discoverable vendor-key-free per-die
             attestation for AI inference on commodity AMD hardware},
  author  = {{FabricCrypt contributors}},
  year    = {2026},
  journal = {arXiv preprint},
  eprint  = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR},
  note    = {v0.2; N=2 chassis. Phase 19 + Phase 22 + Tier-2.}
}
```

A machine-readable [`CITATION.cff`](CITATION.cff) is included for GitHub's "Cite this repository" widget.

---

## License

MIT. See [`LICENSE`](LICENSE).
