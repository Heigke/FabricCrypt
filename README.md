# FabricCrypt

[![CI](https://img.shields.io/badge/CI-pending-lightgrey)](https://github.com/Heigke/FabricCrypt/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-v0.2.1-blue)](https://github.com/Heigke/FabricCrypt/releases)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b)](https://arxiv.org/abs/XXXX.XXXXX)
[![Reproducible: N=2](https://img.shields.io/badge/reproducible-N%3D2%20chassis-orange)](docs/HOW_TO_REPRODUCE.md)

**Per-die attestation *primitive* for AI inference on commodity AMD silicon — no vendor PKI, no Secure Enclave, no TPM (at n=2 chassi).**

> "A per-die attestation primitive on commodity AMD without vendor PKI, demonstrated at n=2 chassi. All cryptographic ceilings are empirical operating points, not formal reductions."

FabricCrypt assembles a per-die fingerprint from **15 signals total** and binds each challenge to a 64-bit audience nonce that drives the *sampling plan itself* — which CPUs, which thermal zones, which core pairs, which sleep durations are read. The 15-signal breakdown:

- **5 HAL-bypass micro-architectural signals**: inter-core TSC offsets, cacheline ping-pong (MOESI), DRAM-refresh-aligned jitter, syscall p99.9 tails, NVMe queue tails.
- **3 cross-host KS-verified micro-architectural signals** (Phase 19, Bonferroni p < 3×10⁻³): GPU clock jitter, multi-zone thermal spread, temporal-Jacobian dynamics.
- **7 board-level deterministic fingerprint signals** (Phase 22, *not* HAL-bypass): PCI topology, PCIe link state, USB descriptor tree, DMI/SMBIOS, kernel boot timing, UCSI power descriptors, amdgpu safe reads.

The result is a stateless challenge-response primitive that:

- distinguishes individual dies of the same SKU (**100% LOO** at N=2 AMD Ryzen AI Max+ PRO 395 "Strix Halo" chassis),
- rejects static and dynamic-library replays (≤5% / ≤10% gates),
- needs no Secure Enclave, no TPM EK, no vendor key material,
- runs end-to-end in **sub-millisecond** (median 1.12 ms, p99 2.79 ms).

> ⚠️ **Honest claim posture (v0.2.1, paper v3.1).** All "bit-security" figures in this codebase and in the paper are **empirical operating points**, not formal cryptographic reductions. The Tier-2 Controlled-PUF wrap empirically returns Hamming μ = 128/256 (random floor) against ML-modeling attackers at N_train ≤ 160, i.e. ≥10⁴ modeling samples without measurable progress — but we do **not** claim a proven 60–80-bit security level. The LAN-relay attacker (V6) defeats every Tier-2 mitigation; distance bounding is explicitly out of scope. See `paper/fabriccrypt_v3_1.md` §5.10.5 and `paper/fabriccrypt_v3_1_EDIT_LOG.md` for the full audit trail.

---

## Update — June 2026: substrate-rooted LLM + optional TPM hard root

New work extends FabricCrypt from a per-die *attestation* primitive to an end-to-end demo that an
**LLM's behaviour can be bound to one specific chip**, plus an optional **hardware-sealed key tier**:

- **Embodiment LOCK** — a frozen GPT-2 + FiLM adapter conditioned on the per-core Vcore fingerprint,
  trained multi-negative. Same weights, different fingerprints: own die → fluent (perplexity **5.44**),
  wrong die → 26 270, shuffled → 2 932 853. First break of the "shuffle wall" with a *physical* signal.
- **TPM hard root** — model key AES-sealed to each box's TPM owner-hierarchy. Cross-die transplant
  matrix (real, N=2): own die **UNLOCK**, foreign die **REFUSED** (integrity mismatch) on both AMD boxes.
- **Two honest tiers**: substrate-dependence (science) + uncopyability (security). Still **N=2**;
  reboot-invariance untested; fingerprint key-gating closed only by the TPM tier. Full write-up,
  honest limits and a 36-second explainer video:
  [`docs/EMBODIMENT_2026-06.md`](docs/EMBODIMENT_2026-06.md) · [`media/embodiment_2026-06.mp4`](media/embodiment_2026-06.mp4).

---

## What's new in v0.2.1

- **Paper v3.1** ([`paper/fabriccrypt_v3_1.md`](paper/fabriccrypt_v3_1.md), 10,309 words) — O116 mandatory pre-launch edits applied. Full diff/audit in [`paper/fabriccrypt_v3_1_EDIT_LOG.md`](paper/fabriccrypt_v3_1_EDIT_LOG.md).
- **Empirical-operating-point language throughout**: replaced all "2^60–2^80 bit-security" claims with empirical attack-cost figures (e.g. "≥10⁴ modeling samples returning random Hamming floor"). No formal cryptographic reduction is provided or claimed.
- **15-signal breakdown clarified**: 5 HAL-bypass + 3 cross-host KS-verified μ-arch + 7 board-level deterministic. The Phase 22 signals are explicitly *not* HAL-bypass micro-architectural — they are board-level deterministic fingerprints. Previously summed as "13 HAL-bypass + 5 light deterministic"; corrected.
- **Headline reframed as *primitive at n=2 chassi***. Top-tier-venue review will rightly demand n ≥ 6; this is L1 in the limitations section.
- **§5 protocol-evolution explicit**: base / Tier-1 / Tier-2. The verifier classifier operates on the chip's *wrapped protocol response*, not on raw on-chip physical measurements.
- **Phase 21b stylometric divergence** moved out of the abstract into §7.L6 as exploratory supplementary detail.
- **New abstract** ([`paper/fabriccrypt_v3_1_abstract.txt`](paper/fabriccrypt_v3_1_abstract.txt), 337 words, arXiv-ready).
- **5-minute pedagogical explainer video** ([`media/fabriccrypt_explainer_5min.mp4`](media/fabriccrypt_explainer_5min.mp4); YouTube mirror TBA at launch T-0).
- **One-pager PDF** ([`media/fabriccrypt_onepager.pdf`](media/fabriccrypt_onepager.pdf)).
- **New bibliographic entries**: Suh-Devadas DAC'07, Rührmair CCS'10, Van Herrewege FC'12, Eckel/Fenzl/Jäger IFIP SEC 2024, LAMINATOR CODASPY 2025.

Reproducibility scripts and Tier-2 attack harnesses are unchanged from v0.2.

---

## Media

- **5-minute pedagogical explainer** — [`media/fabriccrypt_explainer_5min.mp4`](media/fabriccrypt_explainer_5min.mp4) (17 MB).
  YouTube mirror: *https://youtu.be/PLACEHOLDER* (to be added at launch T-0).
  Thumbnail: [`media/explainer_thumbnail.png`](media/explainer_thumbnail.png).
- **One-pager PDF** — [`media/fabriccrypt_onepager.pdf`](media/fabriccrypt_onepager.pdf).
- **60-s transplant demo** — [`demo_video/`](demo_video/) (link in launch tweet).

---

## Paper

- **v3.1 markdown source** — [`paper/fabriccrypt_v3_1.md`](paper/fabriccrypt_v3_1.md) (10,309 words; canonical).
- **v3.1 abstract** — [`paper/fabriccrypt_v3_1_abstract.txt`](paper/fabriccrypt_v3_1_abstract.txt) (337 words, arXiv-ready).
- **v3 → v3.1 audit trail** — [`paper/fabriccrypt_v3_1_EDIT_LOG.md`](paper/fabriccrypt_v3_1_EDIT_LOG.md) (O116 mandatory edits, line-by-line).
- **v0.2 legacy paper** — [`paper/fabriccrypt_v0.2_legacy.md`](paper/fabriccrypt_v0.2_legacy.md) (kept for traceability; superseded).

---

## Quick start

```bash
git clone git@github.com:Heigke/FabricCrypt.git
cd FabricCrypt
./scripts/00_install_deps.sh                            # one-time
source venv/bin/activate

# 1. Capture an extended signature (Phase 12 + 19 + 22)
./scripts/01_collect_signature.sh --reps 10             # ~7-8 min capture

# 2. Repeat on a second chassis, copy data/<host>_sig_v2.npz back

# 3. Classify across chassis (base 290-dim + extended 466-dim)
./scripts/02_classify.sh data/hostA_sig_v2.npz data/hostB_sig_v2.npz

# 4. Tier-2 ML-modeling-attack reproduction (Rührmair CCS'10 baseline)
./scripts/03_tier2_modeling_attack.sh                   # ~3 min, 0/40 expected

# 5. End-to-end demo: issue / sign / verify / try stale replay
./scripts/04_demo.sh                                    # http://127.0.0.1:8770
```

Full multi-chassis walkthrough: [`docs/HOW_TO_REPRODUCE.md`](docs/HOW_TO_REPRODUCE.md).

---

## Comparison vs prior attestation primitives

| System                  | Per-die identity | Vendor PKI required | Open-source verifier | Commodity hw | Sub-ms verify |
|-------------------------|:----------------:|:-------------------:|:--------------------:|:------------:|:-------------:|
| Apple PCC               | ✗ (SKU-class)    | ✓                   | ✗                    | ✗            | n/a           |
| NVIDIA CCM (Hopper CC)  | ✗ (SKU-class)    | ✓                   | partial              | ✗            | n/a           |
| Intel TDX               | ✗ (SKU-class)    | ✓                   | ✗                    | ✗            | n/a           |
| AMD SEV-SNP             | ✗ (SKU-class)    | ✓                   | ✗                    | partial      | n/a           |
| TPM 2.0 EK              | ✓ (factory key)  | ✓ (vendor CA)       | partial              | ✓            | ms-range      |
| **FabricCrypt v0.2.1**  | **✓ (silicon)**  | **✗**               | **✓**                | **✓**        | **1.12 ms**   |

Trade-offs and the Phase-21b caveat row are detailed in [`docs/COMPARISON.md`](docs/COMPARISON.md).

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
├── paper/              ← v3.1 paper + abstract + audit trail
├── media/              ← explainer video, thumbnail, one-pager PDF
├── demo_video/         ← twitter_60s.mp4
└── requirements.txt
```

---

## What this is **not**

- **Not** a formal bit-security claim. Every "attack-cost" figure in the paper and codebase is an **empirical operating point** observed against a specific attacker apparatus; no reduction to a standard cryptographic hardness assumption is provided. See `paper/fabriccrypt_v3_1.md` §5.10.5.
- **Not** a static-benchmark accuracy gain. We prereg-tested it; it came back null. Discussed in `paper/fabriccrypt_v3_1.md` §7.L4.
- **Not** an Apple PCC replacement at scale. PCC binds to a *vendor* signing key inside a Secure Enclave; FabricCrypt operates without one but inherits a different residual-risk profile (see [`docs/PROTOCOL.md`](docs/PROTOCOL.md)).
- **Not** validated beyond **N=2** chassis. We invite community contributions — see [`examples/publish_signature.sh`](examples/publish_signature.sh). Bring a third chassis.
- **Not** defended against LAN-relay (V6) attackers. Distance bounding (Brands & Chaum, EUROCRYPT 1993) is required and explicitly out of scope.

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
             attestation primitive for AI inference on commodity AMD
             hardware (at n=2 chassi)},
  author  = {{FabricCrypt contributors}},
  year    = {2026},
  journal = {arXiv preprint},
  eprint  = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR},
  note    = {v0.2.1; paper v3.1; N=2 chassis. All bit-security
             figures are empirical operating points, not formal
             reductions.}
}
```

A machine-readable [`CITATION.cff`](CITATION.cff) is included for GitHub's "Cite this repository" widget.

---

## License

MIT. See [`LICENSE`](LICENSE).
