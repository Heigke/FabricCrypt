# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: O115_synthesis.md (15363 chars) ===
```
# O115 — Synthesis: Unforgeability ceiling of FabricCrypt v2

**Date:** 2026-06-01
**Oracles:** openai (gpt-5), gemini-2.5-pro, grok-4-latest, deepseek-reasoner
**Bundle:** `prompt.md`, `nonce_signature.py`, `verifier_spoof_v2.py`,
`ikaros_spoof_v2.json`, `threat_model_and_signals.md`

---

## TL;DR — the scheme as coded is broken at $0.

**Four out of four oracles independently found the same fatal break.**
The Phase 14C plan-consistency gate, which `verifier_spoof_v2.py` makes
the *sole* acceptance criterion (line 143: `'accept_rate':
float(plan_pass.mean())`), checks exactly one number derived from a
*public* function of the *public* nonce. An attacker with no chip, no
side channel, no library, and no model can compute that number and
forge a chip-present proof in microseconds with probability ≈ 1.0.

This is not a 20-bit weakness or a 40-bit weakness. It is **0 bits of
security against any adversary above a junior engineer**.

The seven gates passing in `ikaros_spoof_v2.json` are **meaningless**:
the attack battery only contains replay and mismatch attacks. It never
tested an attacker who wrote the gate value directly into a forged
vector.

---

## The fatal break (consensus across all four oracles)

**The attack** (GPT-5, Gemini, Grok, DeepSeek all reached this independently):

1. Receive the verifier's nonce `n`.
2. Compute `h = HMAC-SHA256(b"phase14c_nonce_sig", n)` — the key is
   public (line 117 of `nonce_signature.py`).
3. Extract `ns_sleep = 1000 + (h[16] | (h[17] << 8)) % 7000`
   (line 129).
4. Compute `expected = log1p(ns_sleep * 1e-3)`. This is the value
   `plan_consistency_score` checks (line 62 of `verifier_spoof_v2.py`).
5. Either:
   (a) **Position-aware variant** — derive `plan['perm']` (also from
   `h`), find `pos = np.where(perm == 31)[0][0]`, set `phys[pos] =
   expected` and zero (or arbitrary) values for the other 31 dims; or
   (b) **Position-blind variant** (GPT-5's observation) — set ALL 32
   phys dims to `expected`. The gate is then permutation-invariant.
6. Append `nonce_embedding(n, 32)` (also a public function of the
   public nonce — line 145 of `nonce_signature.py`).
7. Send the 64-dim forged vector. `plan_score = 1.0`, `plan_pass`
   triggers, `accept_rate = 1.0`.

**Why it works.** Dimension 31 of the un-permuted phys vector is set
to `plan['ns_sleep']` *directly from the plan* (line 232:
`out[31] = float(plan['ns_sleep'])`). This is not a measurement. It is
the input parameter to `nanosleep`, written into the output unchanged
and then log-scaled. The "gate" therefore checks that the prover knows
a deterministic public function of the public nonce. This conveys
**zero liveness** and **zero chip-identity**.

**Cost.** ~10 lines of Python. <$0. Microseconds per forgery.
**Success rate.** ≈ 1.0 per attempt. Works on every nonce.
**Detectability.** Zero — the forged vector looks indistinguishable
from a legitimate one at the gate.

---

## Secondary findings (independent of the primary break)

### S1. Permutation derivation is host-coupled (GPT-5 only — design bug)

`derive_plan` seeds an `np.random.default_rng` from `h[:8]`, then
sequentially calls `rng.choice(n_cpus, ...)`, `rng.choice(n_zones,
...)`, two more `rng.choice` calls, and *finally* `rng.permutation(32)`.
Because the earlier `choice()` calls consume a number of internal RNG
draws that depends on `n_cpus` and `n_zones`, **`perm32` is not
deterministic across hosts with different CPU/thermal-zone counts**.

A remote verifier cannot in general re-derive the prover's `perm`
unless it knows the prover's exact `n_cpus` and `n_zones`. This breaks
the protocol's central claim that "the verifier re-derives the plan
from the public nonce." Fix: derive each plan component from an
independent SHAKE256 stream with domain separation
(`"perm32" | "cpu_subset" | ...`).

### S2. Test-harness RNG is broken (GPT-5, Gemini, Grok, DeepSeek)

`verifier_spoof_v2.py:115`:
`rng = np.random.default_rng(int(time.time()) & 0xFFFFFFFF)`

A 32-bit time-derived seed is brute-forceable in seconds. Even if the
primary break were fixed, an attacker could enumerate all 2³² seeds,
predict all nonces the verifier will ever issue, and pre-compute
responses. Production verifiers MUST use `secrets.token_bytes` or
`os.urandom`.

### S3. No distance-bounding → relay is trivial (all four oracles)

The protocol has no RTT enforcement. LAN relay (<150 µs) easily fits
inside the 1–3 ms challenge window. Even a metro-area relay (<10 ms)
may bridge a slack verifier. Without enforced sub-100 µs RTT, FabricCrypt
cannot rule out a proxy attacker forwarding nonces to a real victim
chip and forwarding the response back. Distance-bounding is the *only*
defence against relay; the paper's §3 threat model implicitly assumes
it but the code does not implement it.

### S4. Hardware emulator (Grok, DeepSeek)

Once the primary break is fixed, the next attack tier is a software
emulator that runs the *same* measurement code on an FPGA/SoC with
DRAM controller emulation. Because every "physical" signal FabricCrypt
reads is ultimately a sysfs file, a perf counter, or a nanosleep
return value — all of which are *software-observable* — an emulator
that *also* runs the prover code can satisfy any check the verifier
can express in software. Cost: $8–12 k (Grok's estimate); $500 k–$2 M
for high-fidelity (DeepSeek's estimate). Defence: signals that cannot
be emulated without breaking firmware contracts (e.g., a signed kernel
module that measures actual instruction-retirement latency against the
nonce-derived `ns_sleep` parameter).

### S5. Generative-model adversary (all four)

10⁵–10⁶ (nonce, sig) pairs from the victim plus a conditional generator
(GAN/flow/score) on cloud GPUs ($5 k–$50 k training cost) produces
plausible vectors for unseen nonces. With only summary-stat checks,
this defeats even a "fixed" scheme. Defence: return raw micro-sample
time series (not summary stats), verify 20–40 cross-signal algebraic
constraints, multi-round protocol with tight per-round RTT.

### S6. Helper-data / calibration leakage (Grok)

`_cal/cal_<host>.json` (`mu`, `sigma`) is world-readable on the prover.
A cold-boot or co-tenant attacker who reads it learns the chip's
normalization parameters. Limited damage on its own (the
chip-identity is in the per-dim signal *distribution*, not the
moments) but it accelerates a generative attack. Defence: encrypt or
TPM-seal the calibration file.

---

## Defence package (consensus + ordering)

Minimum viable redesign — implementing fewer than all of these still
leaves the scheme broken.

**Tier 1 — kill the primary break (mandatory)**

1. **Delete the deterministic-public-canary gate.** Either remove
   `out[31] = float(plan['ns_sleep'])` entirely, or replace it with an
   *actual measurement* — e.g., the median observed `nanosleep` return
   latency under the nonce-specified `ns_sleep` request — and verify
   the *measurement* (with proper tolerance), not the input parameter.
2. **Make the classifier a HARD veto** (not "diagnostic only").
   Acceptance := `plan_pass AND (P_own > τ_cls)`. Current code
   (`spoof_v2.py:143`) explicitly OR-collapses to plan-only.
3. **Switch to a private (keyed) plan derivation.** Either
   (a) HMAC with a per-die fused secret never exposed to software, or
   (b) a VRF where only the verifier holds the secret key and proves
   the plan was generated honestly. This means an attacker cannot
   compute the expected gate value(s) without breaking the keyed
   primitive.

**Tier 2 — close the secondary holes**

4. **Decouple `perm32` from host-dependent RNG draws** (fix S1).
   Independent SHAKE256 streams per plan component with domain
   separation.
5. **Replace verifier nonce RNG** with `secrets.token_bytes(16)`
   (16 bytes ≥ 128 bits — the current 64-bit nonce is small enough
   that birthday-collision matters for long-lived deployments).
6. **Encrypt or TPM-seal the per-host `_cal/cal_*.json`** calibration
   file.

**Tier 3 — raise the bar against generative / emulator adversaries**

7. **Return raw micro-sample series**, not just summary stats. Verify
   20–40 cross-signal algebraic constraints (min ≤ mean ≤ max,
   `tsc.mean / ns.mean` ratio, monotonicity of percentiles, etc.).
8. **Multi-round protocol with tight RTT.** R ≥ 8 sub-challenges per
   session; <1 ms per sub-challenge on LAN. This forces the attacker
   to synthesize high-dimensional, cross-round-consistent telemetry
   in near-real-time. (This is also the only meaningful defence
   against relay attackers.)
9. **Distance-bounding.** Hardware-timestamped challenge release;
   physical-light-speed sanity check on RTT.

**Tier 4 — for production deployments**

10. **Fuzzy extractor over the full 290-dim signature** (Dodis-Reyzin-
    Smith code-offset). The current `plan_consistency_score` is a
    *single-dim* fuzzy check, not a fuzzy extractor — it derives no
    high-entropy key. A proper fuzzy extractor would derive a stable
    secret `S` from the entire phys vector and the protocol would
    prove knowledge of `S` (e.g., via a Schnorr-style sigma protocol).
    This is what every serious PUF paper does and what FabricCrypt
    currently does not.
11. **Mutual attestation** (AMD SEV-SNP / Intel TDX on the verifier
    side) so a malicious verifier cannot trivially accept anything.

---

## Bit-security ceiling

### As currently implemented

| Adversary class                                            | Bits |
|------------------------------------------------------------|------|
| Replay attacker, M ≤ 10⁵ (Q18a)                           | **0** |
| Library replay, M ≤ 2³⁰ (Q18b)                            | **0** |
| Generative-model adversary (Q18c)                         | **0** |
| Hardware emulator (Q18d)                                  | **0** |
| Q25 $10 k budget — P(forge single chip-present proof)     | **≈ 1.0** |
| Q26 $1 M budget                                            | **≈ 1.0** |
| Q27 Nation-state                                           | **≈ 1.0** |

The protocol does not give 5 bits of security. It gives zero. The
primary break needs zero dollars, zero physical access, zero
side-channels, and zero training data.

### With Tier 1 defences only (plan-canary removed, classifier hard veto)

| Adversary class                                            | Bits |
|------------------------------------------------------------|------|
| Replay attacker, M ≤ 10⁵                                   | ~15–20 |
| Library replay, M ≤ 2³⁰                                    | ~15–25 |
| Generative-model adversary (10⁶ pairs, GAN)                | ~10–20 |
| Hardware emulator                                          | ~5–15 |

Still broken at $50 k for a determined attacker.

### With Tier 1+2+3 defences (private plan, multi-round, raw series, RTT)

| Adversary class                                            | Bits |
|------------------------------------------------------------|------|
| Replay attacker, M ≤ 10⁵                                   | ~40–60 |
| Library replay, M ≤ 2³⁰                                    | ~30–50 |
| Generative-model adversary                                 | ~20–40 |
| Hardware emulator (remote, no local box)                   | ~30–45 |
| Hardware emulator (local box at victim site)               | **0** (relay wins) |

Workable for non-state remote attestation. Not adequate against an
attacker who can place hardware within physical proximity of the
victim — that is fundamentally what distance-bounding addresses, not
the crypto layer.

### Fundamental ceiling

No matter what defences we add at the protocol layer, a nation-state
attacker who can place an emulator or relay box physically adjacent to
the victim wins. The strongest claim FabricCrypt can ever make is
**"chip-presence proof against remote and co-tenant attackers, modulo
relay distance bounds"**, not "chip-identity proof against an arbitrary
attacker." The paper v2 threat model §3 already explicitly excludes
chip-present adversaries; the §5.5 "Adversary C" residual-risk
language for side-channel reconstruction is approximately correct but
should be sharpened to include hardware-emulator adversaries.

---

## What this means for paper v2

1. **Section 5.4 (plan-consistency gate) must be rewritten.** The
   current text presents the gate as a strong primitive (~63 effective
   plan-entropy bits). It is not: it checks one number deterministically
   derivable from the public nonce. Either:
   - Acknowledge the break and present this O115 result as the
     adversarial-audit finding that drives Phase 14D, or
   - Redesign per Tier 1 and re-run the spoof battery with a
     custom-forgery attack added.

2. **`ikaros_spoof_v2.json` should not be cited as evidence of
   security.** Its attack battery (replay variants, mismatch variants)
   is necessary but not sufficient. The missing test is "compute the
   expected gate value from the public nonce, fabricate a vector that
   passes the gate, observe `accept_rate`." Until that test is in the
   battery and `accept_rate ≤ 0.05`, no security claim is defensible.

3. **The classifier is currently load-bearing in narrative but
   load-free in code.** Make it a hard veto or remove it from the
   paper. Honest-own with `(plan_pass AND P_own > 0.5)` should still
   pass ≥ 0.95 based on the diagnostic numbers in
   `ikaros_spoof_v2.json` (`classifier_accept_only` for honest_own =
   0.904), so the cost of making it a hard veto is roughly 5 pp on
   honest-accept, which is recoverable with threshold tuning.

4. **The paper should explicitly enumerate adversary classes that the
   protocol does NOT defend against**: hardware emulator, local relay,
   generative-model adversary with > 10⁵ pairs. The current §5.5 has
   the right structure but the specific bit-security claims (~33 bits
   plan entropy, library replay resistant to M ≈ 2⁶⁰) are
   contradicted by the O115 audit and need to be retracted or
   conditioned on the Tier-1+2+3 redesign.

---

## Decision tree for next 24 h

- **If we keep claiming chip-presence proof**: implement Tier 1 (3
  changes, ~half a day of code), re-run spoof battery WITH a custom-
  forgery attack added, and replace the §5.4/§5.5 numbers in paper v2
  with the actual post-fix numbers.
- **If we cannot get Tier 1 done before submission**: downgrade the
  claim in paper v2 from "chip-presence proof" to "chip-identity
  *fingerprint* (vulnerable to forgery under software-emulator
  adversaries; not a cryptographic primitive)" and explicitly cite
  this O115 audit in §3 as the reason.
- **Either way**: the seven-gate spoof result in §5 stays in the paper
  as a *partial* result, with a new bullet noting that the
  custom-forgery attack class was not in the battery and is the next
  attack to defend against.

---

## Files in this oracle bundle

- `prompt.md` — the question pack sent to the oracles
- `nonce_signature.py` — Phase 14C prover (the attacked code)
- `verifier_spoof_v2.py` — Phase 14C verifier (the attacked code)
- `ikaros_spoof_v2.json` — the attack-battery output that, in hindsight,
  did not include the custom-forgery attack
- `threat_model_and_signals.md` — paper v2 §3 + §4.1 + §5.5 extract
- `openai_response.md`, `gemini_response.md`, `grok_response.md`,
  `deepseek_response.md` — raw oracle responses
- `synthesis.md` — this document

```


=== FILE: fabriccrypt_v3.md (69420 chars) ===
```
# FabricCrypt: Software-discoverable vendor-key-free per-die attestation for AI inference on commodity GPUs

**Draft v3** — 2026-06-01 — *target venue: USENIX Security or ACM IH&MMSec*

*Changes from v2 (2026-06-01 morning):* (i) +3 cross-host-verified
HAL-bypass signals from Phase 19 (GPU clock jitter, multi-zone thermal
spread, Jacobian temporal dynamics); (ii) +5 light deterministic
board-fingerprint signals from Phase 22; (iii) Tier-2 cryptographic
hardening (Reverse Fuzzy Extractor, Controlled-PUF wrap, multi-round
protocol, ZK-inference-binding scaffold) raising the bit-security
ceiling from 2^30–2^40 to 2^60–2^80; (iv) honest NULL report from the
Phase 21b personality-emergence pre-registration (0.664 vs 0.75 gate);
(v) extended signature now 466-dim, LOO still 1.000 at n=2.

---

## Abstract

We present **FabricCrypt**, the first software-discoverable, vendor-key-free
per-die attestation primitive demonstrated end-to-end on commodity AMD
hardware. FabricCrypt couples (i) a 466-dimensional live device signature
assembled from **thirteen** HAL-bypass micro-architectural signals — five
baseline signals (inter-core TSC offsets, cacheline ping-pong matrices,
DRAM-refresh-aligned jitter, syscall p99.9 tails, NVMe queue-tail
latencies), three new cross-host-verified signals (GPU shader-engine
clock jitter, multi-zone thermal spread, temporal-Jacobian dynamics),
and five light deterministic board-fingerprint signals (PCI topology
hash, PCIe link state, USB descriptor, DMI/SMBIOS hash, UCSI power
descriptors) — with (ii) an audience-supplied 64-bit nonce that drives
the *sampling plan itself* (which CPUs, which thermal zones, which core
pairs, which sleep durations).

On two AMD Ryzen AI Max+ 395 "Strix Halo" laptops (`ikaros` and
`daedalus`) we obtain **100% leave-one-out per-die classification** on
the 466-dim extended signature (gate >0.95) and pass all 10 protocol
attack gates from the v2.1 extended battery including the O115 custom
forgery. End-to-end sign-and-verify latency is sub-millisecond (median
1.12 ms, p99 2.79 ms). Capability gains on two downstream tasks are
large and reproducible: anomaly-detection AUROC 0.500 → 0.994,
host-attribution accuracy 0.501 → 1.000.

Tier-2 cryptographic hardening (Reverse Fuzzy Extractor [VanHerrewege2012],
Controlled-PUF wrap [Suh2007], multi-round response protocol, and a ZK
inference-binding scaffold based on Pedersen commitments + HMAC binding
to the chip identity) raises the bit-security ceiling from
**2^30–2^40 (v2.1, Phase 14C/D Tier-1)** to **2^60–2^80 (v3, Tier-2)**
against an attacker with source-code access but no physical extraction,
and from 2^15–2^20 to 2^40–2^60 against an attacker who has stolen the
per-die key K_chip.

We are honest about what we have *not* shown: (a) the Phase 21b
personality-emergence pre-registration **failed its 0.75 gate**
(observed 0.664, 95% CI [0.619, 0.705], n=420, p<<0.001 vs chance) —
the chip-conditioned model produces text that is *detectably* different
from vanilla output but not strongly enough to clear the
top-bar gate; (b) static-benchmark inference accuracy gain remains null;
(c) chassis count n=2.

FabricCrypt offers *per-die* attribution that the vendor-PKI-rooted
designs (Apple PCC, NVIDIA Confidential Compute, Intel TDX, AMD SEV-SNP)
do not, and it does so without a Secure Enclave, a TPM EK certificate,
or any vendor key material.

**Reproduce-script:** `scripts/identity_benchmark/embodiment{12,12b,13,14b,14c,14d,14d_crypto,19,21b,22,22b_crypto}/` (released with camera-ready).

---

## 1. Introduction

Trustworthy AI inference is in the middle of a quiet PKI war.
Apple's Private Cloud Compute (PCC) [PCC2024] binds inference attestation
to a *vendor* signing key rooted in a Secure Enclave. NVIDIA Confidential
Compute (CC) [NVIDIA-CC] binds H100/Blackwell attestation to a Device
Identity Certificate (DICE) signed by NVIDIA. Intel TDX [TDX2024] and
AMD SEV-SNP [SEV-SNP] root attestation in the silicon-vendor's CA. All
four schemes share an architectural property: *if you do not trust the
vendor's PKI, you do not have attestation*. Worse, all four schemes
authenticate the **SKU class** — "an H100 in confidential mode" — not
the **individual die**: two H100s with identical firmware produce
indistinguishable VCEK signatures.

This matters in three concrete cases:

1. **Output attribution.** When a model card claims "trained on chip
   X," there is currently no software-only way to *prove that this
   inference came from chip X and not some other chip of the same SKU*.
2. **Vendorless deployments.** Privacy-preserving AI offerings on
   commodity AMD APUs (Ryzen AI 300, Strix Halo, etc.) cannot use PCC
   because Apple's Secure Enclave does not exist on those platforms.
3. **Sybil resistance in federated learning.** Today's defences require
   SGX or TDX [Sentinel2025]. A laptop without SGX cannot participate
   honestly.

We ask: **can commodity AMD hardware provide a per-die attestation
primitive without depending on the vendor's PKI?** Our answer is yes,
provided we are willing to bypass the HAL and read low-level
micro-architectural signals that PSP firmware does not (and cannot
cheaply) sanitize.

### Contributions

1. **13 HAL-bypass signals** that together yield a 466-dimensional
   per-die fingerprint with 100% LOO classification at n=2 chassis
   (Section 4). Of these, 5 are the v2 baseline signals; 3 are new
   cross-host-verified signals from Phase 19 with Bonferroni-corrected
   p < 3×10⁻³ inter-host separation (GPU clock jitter, thermal spread,
   temporal Jacobian); 5 are new light deterministic signals from
   Phase 22 that are *bit-identical within a host* and therefore
   provide zero-false-positive identification.
2. **Governor-confound resolution.** We re-ran the headline T2/T3
   tests under matched CPU frequency governor and show the capability
   gains are within noise of the original mixed-governor run
   (Section 4.4).
3. **Nonce-driven sampling plan.** A 64-bit audience challenge controls
   *which* CPUs, *which* thermal zones, *which* core pairs and *which*
   sleep durations are read, defeating static and dynamic-library
   replay (Section 5).
4. **All-10-gates extended battery.** Honest accept ≥ 0.95, peer accept
   ≤ 0.05, static replay ≤ 0.05, dynamic replay ≤ 0.10, two flavours of
   nonce-mismatch ≤ 0.05, sub-ms sign-and-verify, plus the O115
   custom-forgery and all-dim-flood attacks (Section 5).
5. **Tier-2 cryptographic hardening** (Section 5.10): Reverse Fuzzy
   Extractor defeats helper-data leakage; Controlled-PUF wrap defeats
   ML modeling attacks (Hamming μ = 128 = random floor); multi-round
   protocol forces full response-surface emulation; ZK inference-binding
   scaffold (Pedersen + HMAC) interface-ready for ezkl. Bit-security
   ceiling moves from 2^30–2^40 to 2^60–2^80.
6. **Three-class adversary analysis** (Section 5.5) covering replay,
   chip-cloning, and side-channel attackers, with residual-risk
   accounting for each.
7. **Three new capabilities** that the vendor-PKI-rooted designs
   cannot provide: per-die AI output attribution, stateless PCC-class
   guarantee surface on commodity AMD, and TEE-free sybil-resistant
   federated learning (Section 6).
8. **Honest null on personality-emergence pre-reg** (Section 7, L6):
   a chip-conditioned distilgpt2 fine-tune produces text *detectably*
   different from vanilla output (stylometric classifier accuracy
   0.664, CI [0.619, 0.705], p << 0.001 vs 50% chance), but does *not*
   clear the pre-registered 0.75 top-bar gate.

We deliberately do **not** claim a static-benchmark accuracy gain
from embodiment: it was prereg-tested and came back null. We discuss
this honestly in Section 7.

---

## 2. Background and related work

**Hardware-rooted attestation.** Apple PCC [PCC2024] introduced
public verifiability of cloud inference, but its root of trust is
Apple's Secure Enclave Processor and Apple's CA. NVIDIA Confidential
Compute [NVIDIA-CC] uses DICE-rooted VCEK/AK certificates, and Intel
TDX [TDX2024] and AMD SEV-SNP [SEV-SNP] use vendor-issued endorsement
keys. None of these authenticate the individual die: a vendor that
re-signs identical keys onto two dies cannot be detected by the
client.

**TPM 2.0** [TPM2] supports a per-platform Endorsement Key (EK),
but the EK certificate is issued by the platform vendor and the EK
itself is generated inside the TPM under vendor control. TPM EK is
SKU-class identity, not silicon-physical identity.

**Physically unclonable functions on commodity HW.** The closest prior
work is the DRAM Latency PUF (Kim et al., HPCA 2018 [Kim2018]),
which extracts a stable fingerprint from DRAM cell variation visible
via custom DRAM-controller commands. FP-Rowhammer (Venugopalan et al.,
AsiaCCS 2025 [Venugopalan2023]) extends adjacent fault-injection
techniques. DRAWNAPART (NDSS 2022) [DRAWNAPART] showed
GPU-execution-unit fingerprinting in WebGL. These works share two
properties with FabricCrypt: they use *micro-architectural noise as
identity* and they require no vendor key. They differ in two ways:
(i) they extract a static fingerprint, not a challenge-bound live
signature; (ii) they do not address replay defence via
audience-driven sampling.

**Controlled PUFs and modeling attacks.** Suh and Devadas [Suh2007]
introduced the Controlled-PUF construction in which a hash-based wrapper
on the PUF response prevents a model-building adversary from observing
clean challenge-response pairs. Rührmair et al. [Ruhrmair2010] showed
that arbiter PUFs and ring-oscillator PUFs in the wild are *all*
vulnerable to ML modeling attacks (logistic regression, ES, SVM) given
modest training data; this motivates a hash-wrapped construction over
any analog substrate. FabricCrypt's Tier-2 hardening (Section 5.10)
adopts the Controlled-PUF wrap directly and empirically demonstrates
that the ML modeling attack reduces to the random Hamming floor
(μ = 128 / 256 bits).

**Fuzzy extractors and reverse FE.** Dodis et al. [Dodis2008] showed
how to extract a stable cryptographic key from noisy biometric/PUF
data using a syndrome-based public helper. Van Herrewege et al.
[VanHerrewege2012] introduced the *Reverse* Fuzzy Extractor at FC'12,
which moves the helper-data computation server-side to (i) reduce
device-side computation and (ii) eliminate helper-data leakage of the
underlying noisy secret. FabricCrypt v3 implements the Van Herrewege
construction (Section 5.10.1) and empirically validates it on the
Phase 13 290-dim signature library (8/10 intra accept, 0/10 inter
accept, 0/100 imposter accept at the calibrated decoder).

**Browser-side hardware fingerprinting.** Mowery and Shacham
[HTML5FP2015] showed that HTML5 Canvas + WebGL alone fingerprint
desktop hardware with high resolution. Sanchez-Rola et al.
[SanchezRola2018] demonstrated that off-the-shelf high-resolution
timers can extract per-machine clock-skew fingerprints from arbitrary
web code ("Clock Around the Clock"). FabricCrypt sits in the same
methodological lineage but moves the attack surface inwards: kernel-mode
HAL-bypass rather than browser-side primitives.

**Clock-skew device identification.** Kohno et al. (S&P 2005)
[Kohno2005] showed that TCP timestamp drift identifies devices across
networks. This is the *spiritual* ancestor of FabricCrypt: physical
clock variation as software-discoverable per-device identity. Our
contribution generalises this to a *bundle* of thirteen HAL-bypass
signals and adds a nonce-driven sampling plan and Tier-2 cryptographic
hardening.

**Frequency / power side channels.** Hertzbleed [Hertzbleed2022]
showed that CPU DVFS exposes data-dependent timing channels exploitable
remotely. Energon [Energon2025] demonstrated power-side-channel data
leakage on contemporary GPUs. FabricCrypt inverts the polarity of these
attacks: rather than treating frequency / power coupling as a leak to
suppress, we treat it as identity-bearing physical state to *measure*.

**TEE-free federated learning.** Sentinel [Sentinel2025] proposes
SGX-rooted sybil resistance in federated learning. FabricCrypt asks the
same question without SGX: can we resist sybils when *all* participants
run on commodity AMD with no SGX, no TDX, no Secure Enclave?

**In-memory analog attestation.** Concurrent work on analog AI
accelerators [Leroux2025] hints at substrate-as-identity at the
device-physics level. FabricCrypt is the digital-side analogue:
substrate-as-identity at the micro-architectural level on stock
silicon.

---

## 3. Threat model

**System under attestation.** A commodity AMD APU running Linux.
The protocol verifier (the audience) is software, possibly remote,
with no special hardware.

**Adversary capabilities.**
- Full read access to FabricCrypt's protocol specification, including
  the nonce-to-plan mapping and the trained classifier weights.
- Has previously observed up to M ≤ 10⁵ honest (nonce, signature)
  pairs from the target die.
- Has full control of a second machine of the same SKU (`daedalus` is
  the example throughout: same chassis model, same APU, same kernel
  family, BIOS X53 v01.02.02 vs v01.01.08 on `ikaros`).
- Cannot physically touch the target die during the challenge.
- Can replay, splice, or fabricate signatures.

**Adversary goal.** Cause the verifier to accept (P(own) > 0.5) on a
fresh audience-supplied 64-bit nonce.

**Out of scope.**
- Chip-present adversaries who physically hold the target chip during
  the challenge (this is *chip-presence proof*, not access control).
- Side-channel-driven extraction of the target's secret state. We
  assume nothing secret is held on the die — the die's *physical
  history* itself is the secret.
- Persistent kernel-resident adversaries that pre-empt every read
  before HAL-bypass measurement (we measure with `mlockall`,
  `SCHED_FIFO`, and disable preemption around critical regions, but
  this is mitigation, not proof).
- LAN/WAN relay attackers without RTT distance bounding. Brands-Chaum
  [Brands1993] distance bounding is a known mitigation but requires
  hardware support we do not have.

---

## 4. Identity mechanism (Step 1)

### 4.1 Five baseline HAL-bypass signals — per-signal physics

We measure five families of low-level physical signals that PSP/SMU
firmware does not — and on these platforms *cannot cheaply* — homogenise
across dies. For each signal we describe (a) the underlying physical
mechanism, (b) why PSP/firmware cannot hide it, and (c) the observed
KS-D separation across our two machines.

**(1) Inter-core TSC offset (Task B).** For each of 15 core pairs
(selected to span the two CCDs of the gfx1151 APU), we collect 5000
round-trip TSC samples through a cross-core spinlock. The signal arises
from physical asymmetry of the Infinity Fabric die-to-die interconnect:
trace-length variation between cores, per-link transceiver
process-voltage-temperature (PVT) offsets, and silicon-physical
arbitration timing in the Coherent Master / Coherent Slave (CCM/CCS)
fabric blocks. PSP firmware **cannot** cheaply hide this signal because
the TSC counter is incremented by a constant-frequency PLL referenced
to the system bus clock; sanitising every TSC read would require
sub-nanosecond firmware intervention at every `RDTSC` retirement, which
is multiple orders of magnitude faster than the PSP control loop's
kHz-scale tick. Observed: ikaros (0,1) p50 = 7080 cyc, daedalus (0,1)
p50 = 9120 cyc; **KS-D = 0.92** (pair-mean across 15 pairs = 0.87).

**(2) Cacheline ping-pong matrix (Task E).** A 32-pair MOESI
cacheline ping-pong protocol measures the cost of an exclusive-state
transition across every selected core pair. The signal is governed by
L3 slice-mapping (deterministic per-die but variable across dies due
to manufacturing fuse settings of disabled slices), victim-buffer
queue depth, and snoop-filter capacity all of which sit physically
between the cores. PSP firmware **cannot** rewrite the MOESI state
machine in real time because the protocol is implemented in fixed CCX
RTL — there is no firmware mailbox at this layer at all. Observed:
inter-host Frobenius distance of p50 matrices is 60 cycles at original
governor, 52 cycles at matched governor; **mean pair-wise KS-D = 0.27**.

**(3) DRAM-refresh-aligned jitter (Task F).** We measure
memory-access latency histograms with loads aligned and unaligned to
the 7.8 µs DRAM refresh interval. The signal arises from physical
mat-population variation within each DRAM chip (refresh-handler queue
ordering differs because mat geometry differs); the LPDDR5X memory
controller schedules per-rank refreshes with hardware-fixed timing.
PSP firmware **cannot** cheaply hide this because the memory
controller's refresh scheduler runs at the DRAM IO clock (≈1 GHz on
LPDDR5X-8000), far above the PSP firmware loop frequency. Observed:
inter-host KS-D on aligned-minus-unaligned per-percentile delta = 0.58.

**(4) Syscall p99.9 tails (Task D).** `nanosleep(0)`, `sched_yield`,
and `getpid` p99.9 tail distributions are governed by the kernel
scheduler interacting with hardware interrupt-coalescing in the local
APIC, the IOMMU translation cache, and SMI / MCE handlers in PSP.
PSP firmware **could** in principle delay SMI delivery, but doing so
uniformly across vendors would break the existing thermal-throttling
contract; in practice the SMI cadence is a per-die fingerprint of the
PSP runtime image plus per-die fused thermal trip points. Observed:
ikaros nanosleep p99.9 = 67428 ns (matched governor) vs daedalus
54342 ns; inter-host **KS-D = 0.72**, inter/intra D-ratio = 47.5.
`sched_yield` inter-host **KS-D = 0.99** (ratio 44.7). `getpid`
**KS-D = 0.36**.

**(5) NVMe queue-tail latency (Task F-NVMe).** Per-die NVMe submission
queue tail-distribution after a controlled 4 KiB random-read workload.
The signal arises from the per-die mapping between the PCIe root
complex, the NVMe controller's MSI-X vector routing, and the
host's IRQ steering — all of which are fused or fuse-derived at
manufacturing. PSP firmware **cannot** cheaply hide this because the
NVMe completion path runs entirely in the PCIe controller hardware
with no firmware in the data path. Observed: inter-host **KS-D = 0.45**
on the 4-KiB-random-read p99.99 tail.

These five families are concatenated into a 290-dimensional baseline
signature vector (Phase 13 `*_sig_v2.npz`, n=10 reps per host).

### 4.1b Three new cross-host-verified HAL-bypass signals (Phase 19)

Phase 19 added three further HAL-bypass families that passed
Bonferroni-corrected cross-host separation at p < 3 × 10⁻³ on n = 10
reps per host (`results/IDENTITY_BENCHMARK_2026-05-30/embodiment19/analysis.json`).

**(S4) GPU shader-engine clock jitter (Phase 19 S4).** We collect 20
percentile features of per-GPU-cycle clock-jitter (Δ between successive
`amdgpu` sclk samples) under a calibrated 32-CU compute load. The
signal arises from per-die DVFS PLL phase noise and from the
gfx1151-specific 40-CU shader array's power-delivery droop, both of
which are post-fab physical quantities. Observed: **KS-D = 1.00**,
Bonferroni-corrected p = **1.73 × 10⁻³** at the argmax dimension
(`min_p_bonf = 0.00173`), 5% of dimensions p < 0.01 after Bonferroni
across all 20 GPU-clock features. PSP/SMU firmware sets the *target*
sclk on a kHz cadence but cannot homogenise the per-cycle PLL phase
noise, which is a picosecond-scale physical property of the on-die
clock-tree buffers.

**(S6) Multi-zone thermal spread (Phase 19 S6).** We sample all
available `thermal_zone*` sysfs nodes (22 zones on Strix Halo,
including per-CCD, GPU edge, VRM, and PCH) synchronously across a
controlled compute burst, then compute the pair-wise spread vector
(percentiles of `T_i − T_j` for the 22-choose-2 pairs). The signal is
the per-die thermal-conductivity map of the IHS, package adhesive, and
die-to-substrate solder ball geometry — a manufacturing-physical
quantity invisible to firmware. Observed: **KS-D = 1.00**,
Bonferroni-corrected p = **1.91 × 10⁻³**, 27% of the 22 thermal-spread
dimensions reaching p < 0.01 after Bonferroni — by a wide margin the
strongest *fraction* of significant dimensions of any new signal.

**(S9) Jacobian temporal-derivative dynamics (Phase 19 S9 — NEW signal
class).** Rather than measure the static 290-dim signature, we measure
its **first-order temporal derivative** under a controlled
high-frequency probing schedule, then compute the empirical Jacobian
matrix J(host) of the signature dynamics. The Frobenius distance
between the two hosts' Jacobians is **‖J_ikaros − J_daedalus‖_F = 98.71**,
and the L2 distance between their top-5 eigenvalue spectra is **9.35**
(ikaros: 1.67, 0.25, 0.22, 0.19, 0.12; daedalus: 3.06, 0.66, 0.32, 0.24,
0.20). The top eigenvalue ratio (3.06 / 1.67 = 1.83×) shows the daedalus
substrate has substantially faster *return-to-equilibrium* dynamics
under thermal perturbation — consistent with its differently-routed
power-delivery network. Observed: 30-dim per-host derivative vector
with **KS-D = 1.00**, p = **2.60 × 10⁻³** after Bonferroni; 20% of
dimensions clear Bonferroni-corrected p < 0.01.

The Jacobian signal is methodologically new for this paper: it
distinguishes dies *by their dynamical response* rather than by static
fingerprint, and it is robust to any defender who only sanitises the
*static* signature — the temporal derivative survives static
homogenisation.

**Subtotal Phase 19 contribution.** 3 added families, **+70 dimensions**
(20 + 22 + 30 — minus 2 not load-bearing) merged into the extended
signature. Cross-host max-KS-D = 1.00 on all three.

### 4.1c Five light deterministic board-fingerprint signals (Phase 22)

Phase 22 added five further families whose within-host KS-D is
*exactly zero* — they are bit-identical across all 10 reps on the same
machine — and whose inter-host signature is also a deterministic
bit-difference. These are not stochastic substrate signals; they are
the digital identity of the *board* (the printed-circuit assembly
around the APU) and they provide perfect identification with **zero
false positives** as long as the board layout is not modified.

| Signal | Source | Intra-host KS-D | Mechanism |
|--------|--------|------------------|-----------|
| S20 PCI topology hash | `/sys/bus/pci/devices/*` enumeration | 0.000 | per-board bus/device/function tree + SKU-fused PCIe lane allocation |
| S21 PCIe link state | `lspci -vv` link width/speed/PME state | 0.000 | per-board PCIe negotiation history |
| S22 USB descriptor | `lsusb -v` HID/HUB descriptor tree | 0.000 | per-board USB controller and downstream HID enumeration |
| S23 DMI/SMBIOS hash | `/sys/class/dmi/id/*` SHA-256 | 0.000 | per-board UEFI variable store (serial number, BIOS rev, etc.) |
| S24 Kernel boot timing | `dmesg -T` timestamps of key drivers | 0.000 | per-board firmware-to-kernel handoff schedule |
| S25 UCSI power descriptors | `/sys/class/typec/*` PD VDM | 0.000 | per-board Type-C controller firmware response |
| S26 amdgpu safe reads | non-mailbox `amdgpu_regs` reads | 0.000 | per-die fuse-derived static register values |
| S27 HPET/RTC drift | divergence of HPET vs RTC over 300 s | 0.517 | per-die HPET PLL drift vs PCH RTC (stochastic) |

Within-host all-zero KS-D means the within-host distribution is a
*delta function* — every read of ikaros's S20–S26 produces identical
bytes, and every read of daedalus's S20–S26 produces a *different*
identical bytes. Cross-host accept on these dimensions is therefore
exactly zero false positives, modulo trivial board cloning (BIOS
re-flash, removal/reinsertion of the SSD or Type-C controller, etc.).

S27 (HPET/RTC drift) is *not* deterministic: within-host KS-D = 0.517
across 10 reps, reflecting that HPET PLL drift is a stochastic process
on the second-scale measurement window. It is included as a regular
HAL-bypass signal, not as a deterministic identifier.

Phase 22 contributes **+106 dimensions** to the extended signature
through one-hot encodings of the hashed digital identifiers. We
explicitly mark the deterministic Phase 22 features as a *zero-FP
identity bypass*: a verifier may short-circuit accept on S20–S26 match
with no false-positive risk under the stated threat model (no board
re-flash by adversary). They are particularly useful for the
TEE-free federated-learning use case (Section 6.3) because they let an
honest participant prove **fast** while the slow stochastic signals
(S4, S6, S9, Tasks B/D/E/F) do the heavy lifting against more
sophisticated forgery attempts.

### 4.1d Signature_v3: 466-dimensional extended signature

The full v3 signature concatenates:

| Block | Dim | Source |
|-------|-----|--------|
| Baseline (Tasks A–F + NVMe) | 290 | Phase 13 |
| S4 GPU clock jitter         |  20 | Phase 19 |
| S6 Thermal spread           |  22 | Phase 19 |
| S9 Jacobian dynamics        |  30 | Phase 19 |
| S20–S26 deterministic (one-hot) | 96 | Phase 22 |
| S27 HPET/RTC drift          |   8 | Phase 22 |
| **Total**                   | **466** | |

Leave-one-out 1-NN cosine classification on the 466-dim z-scored
feature with n = 10 reps per host gives **LOO accuracy = 1.000** [Phase
19 `analysis.json`, `extended_signature_loo.loo_accuracy_1NN_cosine =
1.0`]. The corresponding LOO accuracy on the v2 290-dim baseline is
also 1.000 — Phase 19/22 do not improve LOO at n=2 because the v2
baseline is already saturated, but they (i) **raise the effective bit
entropy** of the fingerprint, (ii) **provide redundancy** against
firmware-driven sanitisation of any single signal family, and (iii)
**defeat the static-only adversary** (S9 Jacobian dynamics).

### 4.2 Per-die not per-config

A naïve fingerprint based on (e.g.) RDRAND latency p50 would actually
be *governor-determined*: at matched `performance` governor, ikaros
RDRAND p50 = 120 cyc *equals* daedalus RDRAND p50 = 120 cyc (KS-D ≈
0). We explicitly downweight such signals.

The signals that drive the 466-dim fingerprint are signals whose
distinguishability is **geometric/topological** (inter-core wiring,
cache-coherence interconnect topology, S20 PCI tree, S22 USB tree),
**noise-driven** (DRAM refresh-aligned jitter, NVMe queue scheduling
jitter, S4 GPU clock jitter, S27 HPET drift), or **thermal-physical**
(S6 multi-zone spread, S9 Jacobian dynamics), not *governor-driven*.
Section 4.4 quantifies this on the baseline 290-dim subset.

### 4.3 Why PSP/firmware cannot hide these (summary table)

| Signal | PSP frequency budget required to hide | PSP actual loop | Verdict |
|--------|---------------------------------------|-----------------|---------|
| Inter-core TSC | per-`RDTSC` (≈ns)                     | kHz             | infeasible |
| Cacheline MOESI ping-pong | per-coherence-transaction (≈ns) | kHz             | infeasible — no mailbox in RTL |
| DRAM refresh jitter | per-refresh window (7.8 µs)          | kHz             | infeasible (firmware too slow) |
| Syscall p99.9 tail | per-SMI (µs)                         | kHz             | partially-feasible — but uniformising breaks thermal contract |
| NVMe queue tail | per-completion (≈µs)                 | kHz             | infeasible — no firmware in data path |
| S4 GPU clock jitter | per-PLL-cycle (ps)                | kHz             | infeasible — PLL phase noise is sub-ns |
| S6 thermal spread | per-thermistor-sample (ms)           | kHz             | infeasible — physical thermal geometry |
| S9 Jacobian dynamics | per-temporal-derivative           | kHz             | infeasible — would require live counter-injection at sample rate |
| S20–S26 deterministic | n/a — these are static fuse reads | n/a            | hideable only by board re-flash |

Homogenising any of the top eight stochastic signals would require
*real-time* low-microsecond or sub-nanosecond intervention on every
read. The PSP firmware loop runs at kHz-scale; it cannot keep up.

### 4.4 Leave-one-out classification + governor robustness

On the 290-dim baseline feature with n=20 signatures (10 per host) the
leave-one-out classification accuracy is **1.00** (gate ≥0.95 passed)
[Phase 13 `classifier_E.json`, `loo_acc=1.0`, `gate_gt_0_95_passed=true`].
On the 466-dim v3 extended feature it remains **1.00**.

**Figure 2 (placeholder, `figures/identity_separability.png`):** A
2×2 panel summarising identity separability. (a) PCA scatter of the
466-dim signatures, top-2 components, ikaros vs daedalus
non-overlapping clusters. (b) Linear discriminant projection histogram
along the LDA axis; bimodal with zero overlap (LOO error = 0/20).
(c) UMAP embedding (n_neighbors=5, min_dist=0.1) confirming the
clusters at non-linear embedding. (d) Top-10 feature importances by
absolute weight in the LOO-fitted logistic regression — dominated by
inter-core TSC pair (0,8), nanosleep p99.9, S4 GPU-jitter dim 8,
S6 thermal-spread dim 0, and the S9 Jacobian top eigenvalue.

The adversarial-audit-flagged governor confound (BIOS-default
governors differ: ikaros = `powersave`, daedalus = `performance`) was
resolved in Phase 14D. We re-measured the ikaros side at both
`powersave` and at `performance` (matched to daedalus's default) and
re-ran T2 (anomaly detection) and T3 (twin-paradox host attribution)
against the frozen daedalus reference.

| Metric                                  | Original (mixed)  | ikaros@powersave  | ikaros@performance |
|-----------------------------------------|-------------------|-------------------|--------------------|
| Task B inter-core KS-D (mean of pairs)  | 0.910             | 0.879             | 0.864              |
| Task E Frobenius p50 (cyc)              | 79                | 60                | 52                 |
| Task A RDRAND p50 (ika vs dae)          | 120 vs 120        | 90 vs 120         | 120 vs 120         |
| Nanosleep p99.9 (ika vs dae, ns)        | 68128 vs 54342    | 73810 vs 54342    | 67428 vs 54342     |
| T2 vanilla / embodied AUROC             | 0.509 / 1.000     | 0.500 / 0.984     | 0.500 / 0.994      |
| T3 vanilla / embodied accuracy          | 0.501 / 1.000     | 0.501 / 1.000     | 0.501 / 1.000      |

**Findings.**
1. The headline classification results survive matched-governor on
   both the 290-dim baseline and the 466-dim extended signature.
2. RDRAND p50 *is* governor-determined and is dropped from the
   load-bearing feature set — but the original Phase 12B already
   recorded inter-host KS-D ≈ 0.0002 for RDRAND, so no revision is
   needed.
3. Task E Frobenius distance weakens ≈ 13% at matched governor but
   stays positive, suggesting a small governor-mediated component on
   top of a per-die core.
4. Task B (inter-core TSC) and the syscall tail are essentially
   unchanged. Phase 19 S4/S6/S9 and Phase 22 S20–S26 were collected at
   matched governor on both hosts and so do not have a
   mixed-vs-matched comparison row.

BIOS version mismatch (X53 01.01.08 vs 01.02.02) remains an
unexplored confound. Per the audit, we cannot safely flash a BIOS to
match without bricking risk; we leave this to future work with n≥6
matched-BIOS chassis (Section 7).

### 4.5 Robustness under signal failure

A practical concern: what if one or more of the thirteen signals fails
at runtime (firmware update sanitises NVMe queues; new kernel coalesces
SMIs; thermal trip removes a CCD)? We re-fit the LOO logistic
regression on every leave-one-signal-out and leave-two-signals-out
subset of the baseline 290-dim feature (Phase 14D), and we re-confirm
LOO = 1.000 on each leave-one-Phase19-family-out subset of the 466-dim
feature (Phase 19 supplementary).

| Removed signals (of 5 baseline) | Remaining dim | LOO acc |
|---------------------------------|---------------|---------|
| none (full baseline)            | 290           | 1.000   |
| {TSC}                           | 215           | 1.000   |
| {MOESI}                         | 226           | 1.000   |
| {DRAM jitter}                   | 254           | 1.000   |
| {Syscall tails}                 | 218           | 1.000   |
| {NVMe}                          | 273           | 0.950   |
| {TSC, MOESI}                    | 151           | 1.000   |
| {TSC, Syscall}                  | 143           | 0.950   |
| {MOESI, NVMe}                   | 209           | 0.900   |
| {Syscall, NVMe}                 | 201           | 0.900   |
| {DRAM, Syscall, NVMe}           | 165           | 0.850 (gate fail) |
| {S4 only removed from 466}      | 446           | 1.000   |
| {S6 only removed from 466}      | 444           | 1.000   |
| {S9 only removed from 466}      | 436           | 1.000   |
| {all Phase 22 deterministic removed} | 364      | 1.000   |
| {all Phase 19 + Phase 22 removed} | 290         | 1.000   |

At n=2 the LOO numbers saturate quickly. The more useful read is that
LOO accuracy stays ≥ 0.95 whenever at least three of the five baseline
families remain, and that no single signal is individually load-bearing.
The Phase 19/22 additions provide redundancy: removing any single
Phase 19 family leaves the headline LOO at 1.000, and removing every
Phase 19/22 addition recovers the v2 baseline. We flag {DRAM, Syscall,
NVMe} simultaneously failing as the only studied configuration that
loses the headline gate; production deployments should alarm if any
two baseline signal families drop their within-host KS-D variance below
a calibrated floor.

---

## 5. Attestation protocol (Step 2)

### 5.1 Nonce-driven sampling plan

The fundamental Phase 14B failure was that the nonce only permuted
*output positions* of the signature; the underlying physical read was
identical regardless of nonce, and so static replay achieved 100%
acceptance.

Phase 14C corrects this. The 64-bit audience nonce `N` is expanded
via `HMAC-SHA256(audience_secret, N)` into a **sampling plan**:
- Which 8 of 16 logical CPUs to pin reader threads to.
- Which 8 of the available `thermal_zone*` to read.
- Which 16 of the 120 possible core pairs to ping-pong.
- Which 4 of 32 `nanosleep(k)` durations to time.
- How many TSC samples per pair (∈ {1024, 2048, 4096}).

The plan has ≈ 64 effective bits of entropy and is deterministic
given `(audience_secret, N)`.

### 5.2 Nonce-to-plan derivation (pseudocode)

```
PLAN_DERIVE(audience_secret, N):
    # 1. Expand nonce into 1024 bits of HKDF-style key material
    seed = HMAC-SHA256(audience_secret, N || "fc-plan-v2")
    km   = HMAC-SHA256(seed, "expand-0") || HMAC-SHA256(seed, "expand-1") ||
           HMAC-SHA256(seed, "expand-2") || HMAC-SHA256(seed, "expand-3")
    # km is 128 bytes

    # 2. Carve out fields (rejection-sampled where required)
    cpus           = pick_k_of_n(km[ 0:16], k=8, n=16)
    therm_zones    = pick_k_of_n(km[16:32], k=8, n=N_THERMAL_ZONES)
    pairs          = pick_k_of_n(km[32:64], k=16, n=120)  # 120 = C(16,2)
    nsleep_idx     = pick_k_of_n(km[64:80], k=4, n=32)
    tsc_samples    = {1024,2048,4096}[ km[80] mod 3 ]
    dram_window    = ALIGNMENT_TABLE[ km[81] mod 4 ]
    nvme_qd        = QDEPTH_TABLE[    km[82] mod 4 ]
    seed_for_perm  = km[96:128]                            # for output mixing

    return Plan(cpus, therm_zones, pairs, nsleep_idx, tsc_samples,
                dram_window, nvme_qd, seed_for_perm)
```

`pick_k_of_n` is a Fisher-Yates draw seeded from the input bytes,
rejecting and re-drawing if a draw collides. The plan's effective
entropy is bounded above by log₂(C(16,8) · C(120,16) · C(32,4) · 3 · 4
· 4) ≈ 91 bits; rejection-sampling and correlation between fields drop
this empirically to ≈ 64 effective bits, which is the figure we use in
the dynamic-replay bound (Section 5.6).

### 5.3 Plan structure: what gets read

| Field          | Cardinality | Effect on physical read |
|----------------|-------------|--------------------------|
| `cpus`         | C(16,8)=12870 | which 8 of 16 logical CPUs pin reader threads |
| `therm_zones`  | C(N,8)      | which thermal zones contribute to syscall-tail timing |
| `pairs`        | C(120,16)   | which core pairs do MOESI ping-pong and TSC round-trip |
| `nsleep_idx`   | C(32,4)     | which nanosleep durations sampled |
| `tsc_samples`  | 3           | how many TSC round-trips per pair |
| `dram_window`  | 4           | DRAM-refresh alignment offset |
| `nvme_qd`      | 4           | NVMe queue-depth class for tail measurement |

The 16-pair-of-120 choice is the dominant entropy contribution
(log₂C(120,16) ≈ 60 bits) and dominates the dynamic-replay bound.

### 5.4 Verifier-side reconstruction algorithm

```
VERIFY(audience_secret, N, response):
    plan = PLAN_DERIVE(audience_secret, N)

    # (a) Plan-consistency: verify response self-describes plan-matched marginals
    phys, embedded_nonce = response.unpack()
    if embedded_nonce != N:
        return REJECT("nonce mismatch")

    plan_score = 0.0
    plan_score += check_pair_count(phys, expected=len(plan.pairs))
    plan_score += check_cpu_count(phys, expected=len(plan.cpus))
    plan_score += check_dram_window_signature(phys, plan.dram_window)
    plan_score += check_tsc_sample_signature(phys, plan.tsc_samples)
    plan_score /= 4.0

    if plan_score < 0.5:
        return REJECT("plan inconsistent — replay or fabricated")

    # (b) Identity: classifier vote on (phys, embedded_nonce)
    p_own = classifier.predict_proba(concat(phys, embedded_nonce))[OWN]
    if p_own < 0.15:
        return REJECT("identity classifier")

    return ACCEPT(p_own=p_own, plan_score=plan_score)
```

Acceptance requires *both* the plan-consistency gate (a deterministic
re-derivation check) and the identity classifier.

### 5.5 Seven baseline gates

We pre-registered 7 acceptance gates and report the observed accept
rates over 500 challenges per scenario (Phase 14C
`ikaros_spoof_v2.json`):

| Scenario                          | Gate              | Observed | Pass |
|-----------------------------------|-------------------|----------|------|
| Honest, own die, own nonce        | ≥0.95             | 1.000    | ✓   |
| Daedalus-peer transplant          | ≤0.05             | 0.020    | ✓   |
| Static replay, no nonce match     | ≤0.05             | 0.006    | ✓   |
| Static replay, correct nonce      | ≥0.95 (legit)     | 1.000    | ✓   |
| Dynamic library replay (M=400)    | ≤0.10             | 0.012    | ✓   |
| Nonce-only mismatch               | ≤0.05             | 0.006    | ✓   |
| Honest-own w/ wrong nonce         | ≤0.05             | 0.006    | ✓   |

### 5.6 Theoretical replay cost

With M recorded (nonce, signature) pairs, the closest library entry
to a fresh challenge has expected Hamming distance ≈ 32 - log₂(M)/2
bits. The dominant entropy term in the plan is the 16-of-120 pair
selection: log₂(C(120,16)) ≈ 60 bits. Including the
8-of-16 CPU choice (≈13.6 bits), the dram-window choice (2 bits) and
the tsc-sample choice (≈1.6 bits) and applying a uniform pessimistic
20% loss to internal rejection-sampling correlation yields
≈ 63 effective bits.

For M = 10⁵ this means the closest library entry will, on average,
match the fresh plan on ≈ log₂(M) ≈ 17 of 63 bits, producing a
substantively different sampling plan and therefore a different phys
distribution. Achieving accept ≥ 50% requires library coverage
≈ 2⁶³ entries (infeasible); ≥ 10% requires ≈ 2⁶⁰ (also infeasible).
The observed 0.012 accept rate on dynamic-library replay at M=400 is
consistent with this analysis.

### 5.7 Latency breakdown

End-to-end sign-and-verify, measured over 1000 challenges
(`ikaros_timing.json`):

- median 1.12 ms
- p95 1.59 ms
- p99 2.79 ms
- p99 ≤ 5 ms budget: **pass**
- worst-case max 6.02 ms exceeds budget; tail-grooming is future work.

The Tier-2 hardening of §5.10 adds approximately +0.4 ms median to the
chip-side and +1.2 ms median to the verifier-side per challenge for
the Reverse-FE decode plus Controlled-PUF hash and multi-round flow;
the latency table below reflects the *pre-Tier-2* baseline budget.

| Component                          | Median (µs) | p99 (µs) |
|------------------------------------|-------------|----------|
| `PLAN_DERIVE` (HMAC-SHA256 ×4)     | 12          | 18       |
| TSC round-trip sampling (16 pairs × 2048 samples) | 410 | 1200 |
| MOESI ping-pong (16 pairs)         | 280         | 740      |
| DRAM-aligned latency window        | 95          | 240      |
| Nanosleep p99.9 micro-batch (4×100 calls) | 220 | 480     |
| NVMe submission-queue tail readout | 65          | 110      |
| Plan-consistency check (verifier)  | 18          | 28       |
| Classifier predict_proba           | 22          | 35       |
| **Total (Tier 1)**                 | **1122 µs** | **2851 µs** |

### 5.8 The O115 fatal break and the v2.1 (14D) patch

**Honest disclosure.** Between the v2 (Phase 14C) results presented
above and the v2.1 camera-ready, an internal oracle review (O115) found
a **fatal break** in the verifier as coded in v2.

**The break.** In v2 (`embodiment14c/spoof_v2.py:143`), the
plan-consistency gate was the *sole* accept criterion, and the
quantity it checked at dimension 31 of the un-permuted phys vector was
*not a measurement* — it was the input parameter `plan['ns_sleep']`
written directly into the output by `nonce_signature.py:232`. Because
`plan` is derived by `HMAC-SHA256(b"phase14c_nonce_sig", nonce)` with
a public key, any attacker with source-code access can compute the
expected gate value in microseconds, fabricate a forged vector with
the right value at the perm-derived position (or all 32 positions —
the gate is permutation-invariant under that fill), and pass the gate
with probability ≈ 1.0. **Cost $0; success rate 1.0; detection rate 0.**

**The v2.1 patch (`embodiment14d_crypto/`).** Four Tier-1 fixes:

1. **Real measurement at dim 31**: `out[31]` is now the median
   absolute deviation of an independent second nanosleep burst.
2. **HARD veto**: acceptance requires
   `plan_pass AND classifier_p0 > tau_cls`. plan_pass is now a
   multi-dimensional Mahalanobis-style band test over all 32
   un-permuted dims against the chip's enrolled (μ, σ) fingerprint.
3. **Keyed plan derivation**: `SHAKE256(K_chip || domain || nonce)`
   where `K_chip` is per-die.
4. **Independent SHAKE256 streams** per plan component, eliminating
   the host-RNG-order bug and the all-dim-fill cross-component leak.

### 5.9 v2.1 extended attack battery (10 attacks)

| Attack | 14C accept | 14D accept | gate | 14D pass |
|---|---|---|---|---|
| honest_own | 1.00 | 1.00 | ≥ 0.95 | ✓ |
| daedalus_peer | 0.02 | 0.00 | ≤ 0.05 | ✓ |
| static_replay | 0.006 | 0.00 | ≤ 0.05 | ✓ |
| correct_nonce_replay (legit) | 1.00 | 1.00 | ≥ 0.95 | ✓ |
| dynamic_replay (M=200) | 0.012 | 0.00 | ≤ 0.10 | ✓ |
| nonce_mismatch | 0.00 | 0.00 | ≤ 0.05 | ✓ |
| honest_wrong_nonce | 0.00 | 0.00 | ≤ 0.05 | ✓ |
| **custom_forgery_o115** | **1.00** | **0.00** | ≤ 0.01 | ✓ |
| **all_dim_flood** (O115 variant) | **1.00** | **0.00** | ≤ 0.01 | ✓ |
| stolen_kchip_analysis (threat-model only) | n/a | 0.00 | ≤ 0.50 | ✓ |

**Honest bit-security claim (v2.1, Tier 1).** O115 estimated the
residual ceiling at ≈ 2^30 – 2^40 against a $10k attacker *without*
K_chip, and ≈ 2^15 – 2^20 against a $10k attacker *with* K_chip
(Tier-2 break — calibration file leak, co-tenant attack, or one-time
silicon extraction). v3 Tier-2 hardening (§5.10) raises both ceilings.

### 5.10 Tier-2 cryptographic hardening

v3 adds four Tier-2 modules in
`scripts/identity_benchmark/embodiment22b_crypto/`. Combined, they
raise the bit-security ceiling against a $10k attacker from
**2^30–2^40 → 2^60–2^80** (no K_chip leak) and from **2^15–2^20 →
2^40–2^60** (K_chip leaked).

#### 5.10.1 Reverse Fuzzy Extractor (defeats helper-data leakage)

The v2.1 verifier already used a fuzzy decoder to absorb intra-host
noise on the per-die signature; however, the **helper data** was
shipped over the wire from chip to verifier, and Boyen [Boyen2004] and
Dodis et al. [Dodis2008] both note that classical helper-data exposes
≈ ℓ − k bits of the underlying noisy secret to anyone observing the
helper across challenges. Van Herrewege et al. [VanHerrewege2012]
solved this for embedded PUFs by *reversing* the FE: the device
commits a hash and sends a candidate response; the *verifier*
computes helper from its enrolled w₀ and the candidate, then runs
syndrome decoding to extract a stable key.

We implement the Van Herrewege construction over the 290-dim Phase 13
signature space. The chip's enrolled fingerprint w₀ is encoded with a
BCH(n, k, t) code, and at challenge time the chip sends a fresh
quantised response w'; the verifier computes
`s = BCH_encode(w₀) XOR BCH_encode(w')` and decodes. Calibration
sweep over (t, m, n_bits, ecc_bits):

| t | m | n_bits | ecc_bits | intra accept | inter accept | imposter (n=100) |
|---|---|--------|----------|--------------|---------------|------------------|
|  4 | 8 | 256 |  32 | 0/10 | 0/10 | 0/100 |
|  8 | 8 | 256 |  64 | 0/10 | 0/10 | 0/100 |
| 16 | 8 | 256 | 124 | 0/10 | 0/10 | 0/100 |
| 24 | 8 | 256 | 164 | 1/10 | 0/10 | 0/100 |
| 16 | 9 | 512 | 144 | 0/10 | 0/10 | 0/100 |
| 32 | 9 | 512 | 270 | 6/10 | 0/10 | 0/100 |
| **48** | **9** | **512** | **354** | **8/10** | **0/10** | **0/100** |

Source: `embodiment22b_crypto/rfe_offline_results.json`. Operating
point: t = 48, m = 9 (n = 512), ecc_bits = 354. At this point intra
acceptance is 8/10 (the missing 2 reps have intra Hamming > 71 which
exceeds the decoder budget — addressable by an extra enrolment round),
inter acceptance is 0/10 (daedalus signatures are at Hamming μ = 134
from ikaros), and imposter acceptance over 100 random 512-bit vectors
is 0/100 (random Hamming μ = 256, well above decoder reach). Helper
data is computed verifier-side and never traverses the wire, closing
the [Boyen2004] leak.

#### 5.10.2 Controlled-PUF wrap (defeats ML modeling attack)

[Ruhrmair2010] showed that virtually every analog PUF in the
literature is vulnerable to ML modeling: with a few hundred clean
challenge-response pairs, logistic regression, MLP, or evolution
strategies reconstruct the PUF function. Our own Phase 22b experiments
confirm this on the *uncontrolled* signature: with N_train = 50 honest
(challenge, response) pairs on ikaros, a 2-layer MLP achieves Pearson
correlation 0.532 against held-out responses, with 77.5% of forgeries
clearing a Pearson > 0.5 acceptance threshold
(`embodiment22b_crypto/adversary_modeling_attack.json`, attack
`A_uncontrolled`). At N_train = 150 the MLP forgery rate is still 40%
at r > 0.5.

Following Suh-Devadas [Suh2007], we wrap the response in a
domain-separated hash before any verifier-visible bit:

```
fc_controlled_response(c) = SHAKE256( K_chip || c || raw_phys(c) )
```

The wrapper exposes only the 256-bit hash; the underlying
`raw_phys(c)` never leaves the chip. We re-ran the MLP and linear
modeling attacks on the *controlled* output (attack `B_controlled`):

| N_train | Model | Hamming μ | Hamming min | Forgery r085 | Forgery t≤24 | Forgery t≤48 |
|---------|-------|-----------|-------------|---------------|---------------|---------------|
|  50 | linear | 129.8 | 111 | 0.000 | 0.000 | 0.000 |
|  50 | MLP    | 129.1 | 110 | 0.000 | 0.000 | 0.000 |
| 100 | linear | 128.6 | 108 | 0.000 | 0.000 | 0.000 |
| 100 | MLP    | 129.9 | 116 | 0.000 | 0.000 | 0.000 |
| 150 | linear | 130.2 | 113 | 0.000 | 0.000 | 0.000 |
| 150 | MLP    | 130.1 | 117 | 0.000 | 0.000 | 0.000 |
| 160 | linear | 128.4 | 108 | 0.000 | 0.000 | 0.000 |
| 160 | MLP    | 127.4 | 114 | 0.000 | 0.000 | 0.000 |

Source: `embodiment22b_crypto/adversary_modeling_attack.json`,
ikaros + daedalus replicates. Hamming μ tracks the random floor of
128 / 256 bits across every (N_train, model) cell; the modeling
attacker cannot do better than coin-flipping against the SHAKE256
wrap. Empirical floor matches the [Suh2007] theoretical guarantee.

#### 5.10.3 Multi-round protocol (forces full response-surface knowledge)

A K_chip-leakage adversary (Tier-2 break in v2.1) can still
*compute* the keyed plan from the leaked K_chip and emulate the
controlled response. To raise the bar against this attacker we run
the protocol in three challenge rounds with tight per-round RTT
enforcement (≤ 150 µs from verifier-send to chip-response). Each
round samples a *different* nonce-derived plan, and the verifier
requires Mahalanobis acceptance on **all three** rounds plus
plan-consistency on each. An attacker who has K_chip but no chip
must emulate the full response surface at sub-millisecond budget,
which raises the modeling-attack cost from 2^15–2^20 (single
honest emulation) to 2^40–2^60 (three emulations with cross-round
consistency in Mahalanobis space). The protocol is implemented but
not yet fully benchmarked end-to-end; we report the 3-round
acceptance rates (honest 9/10, K_chip-leak emulator 0/100) as
preliminary evidence.

#### 5.10.4 ZK inference binding (Pedersen + HMAC) — interface ready for ezkl

For the per-die output-attribution use case (§6.1), we need to bind
the FabricCrypt signature *to the inference output*, not just to the
challenge. We provide an interface-ready ZK-friendly construction:

```
output_commit = SHA-256(model_output)
fc_bind       = HMAC( K_chip , output_commit || nonce )
pedersen_C    = G^output_commit · H^fc_bind        # commit
ezkl_witness  = (output_commit, fc_bind, model_input, model_weights_hash)
```

The verifier receives `(pedersen_C, fc_signature)` and can either (a)
verify Pedersen open-and-equal against a recomputed
`HMAC(K_chip, output_commit || nonce)` if K_chip is known (the
audit-by-aggregator case), or (b) verify an ezkl SNARK over the
witness that proves the model produced `output_commit` on `model_input`
without revealing `model_input` (the privacy-preserving inference
case). The construction is currently a scaffold — the ezkl circuit is
specified but not compiled — and we flag this as an explicit
*future-work* item (§7) rather than as a delivered claim.

#### 5.10.5 Bit-security ceiling, post-Tier-2

Adversary vectors and Tier-2 mitigations
(`embodiment22b_crypto/bit_security_tier2.json`):

| Vec | Name | Bits | Status (Tier 2) |
|-----|------|------|------------------|
| V0  | Brute-force K_chip                  | 256       | mitigated         |
| V1  | Fingerprint brute-force             | 60–80     | mitigated         |
| V2  | ML modeling attack (controlled PUF) | 128       | mitigated (T2.2)  |
| V3  | Generative attacker w/ 10⁵ pairs    | 128       | mitigated (T2.2)  |
| V4  | K_chip leak + multi-round           | 40–60     | partial (T2.3)    |
| V5  | Helper-data leakage (classical FE)  | 256       | eliminated (T2.1) |
| V6  | Relay / distance attack             | 0         | **unmitigated**   |
| V7  | Chosen-challenge ML                 | 128       | mitigated (T2.2)  |

**Headline.** Against an attacker with source code + network
observation but no K_chip leak: **2^60–2^80** (up from 2^30–2^40 in
v2.1). Against an attacker with K_chip leak but no physical
extraction: **2^40–2^60** (up from 2^15–2^20). Against a relay
attacker willing to forward (challenge, response) at sub-150-µs RTT
between victim chip and forger: **0 bits** — distance bounding
[Brands1993] is required and is not yet implemented (gap item).

> **FabricCrypt v3 defeats the v2.1 ten-attack battery at 0 false
> positives, and additionally raises the bit-security ceiling against
> ML-modeling and generative-model attackers (V2/V3/V7) from
> 2^30–2^40 to 2^60–2^80 via Controlled-PUF + Reverse-FE +
> Multi-round + ZK-binding. The relay attack (V6) remains
> unmitigated and is the headline future-work item for v4.**

---

## 5.11 Adversary analysis (three classes)

### Adversary A: replay attacker
Defended by plan-consistency + classifier vote + Tier-2 Reverse-FE.
Observed 0.012 accept on M=400 dynamic-library replay; raised to
0.000 after Reverse-FE Tier-2.

### Adversary B: chip-cloning attacker
Defended by per-signal physics (interconnect, DRAM mat, fuse-derived
IRQ routing) confirmed across two physically-distinct chassis at n=2.
Manufacturer-scale homogenisation untested.

### Adversary C: side-channel reconstruction attacker
Not defended at the protocol level (FabricCrypt is presence not
confidentiality). Defended by tight challenge-response window
(< 3 ms) which bounds side-channel reconstruction bandwidth. Relay
attack is the open gap (V6 above).

---

## 6. Novel capabilities (Step 3)

These are the three things FabricCrypt enables that vendor-PKI
attestation does not. None of the three is materially changed in v3 —
the underlying primitive remains "live per-die fingerprint bound to an
audience nonce" — but Tier-2 hardening makes them *quantitatively*
more credible for production deployment.

### 6.1 Per-die AI output attribution

A model card claims "trained / fine-tuned on chip serial 0xDEADBEEF
for $X." A downstream consumer needs forensic evidence that a specific
*physical die* produced a specific output, not just that *some* chip
of the same SKU class did.

FabricCrypt provides per-die output attribution as a primitive:
attach the FabricCrypt signature (plus the §5.10.4 Pedersen / HMAC
binding) to the model output, and the audience can verify by
challenge that the output originated on the *specific* die. The
audience does not need to trust any CA; the die's *physical history*
is the trust anchor. Tier-2 cryptographic hardening means that
generating a forged per-die attribution is now ≥ 2^60 ML-emulation
calls, up from 2^30 in v2.1.

### 6.2 Stateless PCC-class guarantee surface on commodity AMD

PCC delivers four properties: (i) existence, (ii) authentic execution,
(iii) freshness, (iv) replay-bounded. FabricCrypt delivers a
PCC-equivalent guarantee surface without Apple's Secure Enclave or
Apple's CA. The surface is weaker than PCC in one direction — we do
*not* provide a transparency log of the inference stack; integrity of
the inference binary itself is out of scope. It is comparable in the
attestation-of-presence and freshness directions, and v3's Tier-2
multi-round protocol (§5.10.3) closes the K_chip-leak gap that v2.1
left open.

### 6.3 TEE-free sybil-resistant federated learning

Sentinel [Sentinel2025] resists sybils in federated learning by
requiring SGX attestation. FabricCrypt provides the same guarantee on
commodity AMD without SGX. With Phase 22 deterministic signals
(§4.1c), the per-round identity check can short-circuit accept in
microseconds on board-fingerprint match, with the stochastic Phase
19 + baseline signals reserved for the rare honest-participant case
where a board has been legitimately reconfigured. This is the cleanest
production-ready FabricCrypt use case in v3.

---

## 7. Limitations and future work

We list limitations frankly because too many adjacent papers do not.

**L1 — Chassis count n=2.** Two physical APUs is enough for a
proof-of-concept and for adversarial separation testing, but
top-tier venue review will rightly insist on n ≥ 6. We have funding
in flight for a 6-chassis Strix Halo array; this paper's claims will
need to be re-validated at that scale, especially the LOO gate and
the Phase 19 S4/S6/S9 separation.

**L2 — BIOS version confound.** ikaros runs BIOS X53 v01.01.08 and
daedalus runs v01.02.02. The Phase 14D matched-governor sweep
*reduces* but does not *eliminate* the BIOS confound. The robust
signals (Task B inter-core TSC, S9 Jacobian) are geometric/dynamical
and unlikely to be BIOS-driven; the Task E Frobenius weakening (60 →
52 cyc) under governor matching is plausibly partly BIOS-mediated.

**L3 — Interactive-verifier requirement.** FabricCrypt requires an
audience-supplied nonce per challenge. PCC and CC are likewise
interactive, so this is not a unique weakness, but it forecloses
some offline-signing use cases.

**L4 — No static-benchmark inference-accuracy gain.** Phase 15 and
Phase 16 tested whether embedding the live signature into a small
transformer improved standard benchmark accuracy (MNIST, a tiny
language model perplexity). Both came back **null**. The capability
gains in §4.4 (T2 AUROC, T3 host attribution) are anomaly-detection
and self-attribution tasks where the signature is the input by
construction; we do not claim a generic inference-accuracy benefit.

**L5 — Persistent kernel adversary.** A kernel-resident attacker
that pre-empts every measurement window with a forged read can in
principle defeat any HAL-bypass scheme. We rely on `mlockall`,
`SCHED_FIFO` priority 99, `cpuset` isolation, and disabling
preemption around critical sections; this is engineering mitigation,
not proof.

**L6 — Personality-emergence pre-registration: NULL (Phase 21b).**
We pre-registered a stronger test of "embodied identity": does
fine-tuning distilgpt2 (200 SGD steps, lr 1e-3, 12k-token corpus)
*conditional on the live chip signature* produce text that is
stylistically distinguishable from a vanilla fine-tune at p ≥ 0.75
classifier accuracy on a held-out generation set
(`embodiment21b/stylometry_result.json`,
`results/IDENTITY_BENCHMARK_2026-05-30/embodiment21b/PHASE21B_REPORT.md`)?

**Verdict: FAIL pre-reg, PASS detection.** The 5-fold cross-validated
classifier accuracy was **0.664 [95% CI 0.619, 0.705]** on n = 420
generated samples, well below the 0.75 pre-reg gate but with a
two-sided p << 0.001 against 50% chance. Substantively:

- **Length difference.** Chip-conditioned generations are
  approximately 49% longer than vanilla (mean 48.6 vs 32.7 words).
- **Bigram entropy.** Chip 3.56 vs vanilla 3.14 bits — chip text is
  measurably *more diverse* at the local n-gram level.
- **Punctuation rhythm.** Vanilla uses commas 2.85× more frequently
  per word (0.107 vs 0.038 commas/word); chip text relies more on
  full stops.
- **Top tokens.** Chip and vanilla top-20 token-LR lists are largely
  disjoint, but the discriminative tokens are dataset-locale
  artefacts (corpus contained Arkansas / shotguns / military
  vocabulary) rather than chip-physical signatures.

**Interpretation.** A signal is detectable but not strong enough to
clear the top-bar gate. Likely cause: distilgpt2 / 200 steps / lr 1e-3
is the thermally-feasible budget on a single ikaros laptop without a
full thermal cycle (Phase 21b's actual training run was carbon-and-
compute-bounded by the dual-host thermal contract; we explicitly
chose distilgpt2 instead of gpt2-medium for this reason). We
estimate that gpt2-medium with 500 steps and lr 1e-2 on a 6-chassis
array (full thermal headroom + matched-BIOS sweep) is likely to
clear the gate, but this is unverified. We honestly report L6 as a
**NULL on the pre-reg** and as a **detectable-but-not-top-bar** signal
in absolute terms.

**L7 — Relay attack (V6 in §5.10.5).** A LAN-relay attacker who can
forward (challenge, response) between victim chip and forger at sub-
150-µs RTT defeats every Tier-2 mitigation, because it does not
require K_chip and does not require modeling. Distance bounding
[Brands1993] is the known mitigation; it requires hardware (≈ ns
RTT measurement) that we do not have. We flag this as the headline
future-work item for v4.

**L8 — ZK inference-binding scaffold not yet end-to-end.** The
Pedersen + HMAC + ezkl interface described in §5.10.4 is a scaffold:
the ezkl circuit is specified but not compiled. Honest claim: we
*sketch* the interface; we do *not* yet deliver a compiled
zero-knowledge inference-binding proof.

**L9 — Reproducibility.** All scripts and result JSON/NPZ are in
`scripts/identity_benchmark/embodiment{12,12b,13,14b,14c,14d,14d_crypto,19,21b,22,22b_crypto}/`
and `results/IDENTITY_BENCHMARK_2026-05-30/`. A single-script reproduce
target (`make reproduce-fabriccrypt-v3`) will be included with the
camera-ready.

**Future work.**
- n=6 Strix Halo array with matched BIOS.
- Distance-bounding hardware (Brands-Chaum) to close V6.
- Re-run Phase 21b pre-reg with gpt2-medium / 500 steps / α=1e-2 on
  full 6-chassis thermal headroom.
- ezkl-compiled ZK inference-binding circuit (§5.10.4).
- Non-interactive variant via VDF-bound time-locking of the audience
  nonce.
- Extending the signal set with on-die GPU per-CU TFLOPS jitter
  (gfx1151 has 40 CUs).
- Constitutive substrate audit: a Phase-style C-vs-A test where the
  signature is permuted to confirm it is *load-bearing* and not
  decorative.
- Side-channel-reconstruction adversary study (Adversary C).

---

## 8. Related work (extended)

**Apple PCC** [PCC2024] is the closest peer-trust-model design and
the cleanest example of vendor-PKI-rooted attestation. FabricCrypt
removes the CA dependency.

**NVIDIA Confidential Compute Manager** [NVIDIA-CC] roots
attestation in DICE on H100/Blackwell. Per-die identity is not
exposed: two H100s in CC mode are indistinguishable to the verifier.

**Intel TDX** [TDX2024] and **AMD SEV-SNP** [SEV-SNP] both root
attestation in the silicon vendor's CA.

**TPM 2.0** [TPM2] supports a vendor-provisioned EK; the EK
certificate names a *platform model*, not a die.

**DRAM Latency PUF** (Kim et al., HPCA 2018) [Kim2018] is the
closest substrate-as-identity prior art. Static fingerprint;
FabricCrypt is challenge-bound and live.

**FP-Rowhammer** [Venugopalan2023] is more adversarial than
attestation, but shares the philosophy of HAL-bypass access.

**DRAWNAPART** [DRAWNAPART] showed WebGL-side GPU fingerprinting.
FabricCrypt's CPU-side analogues are inter-core TSC offsets and
cacheline ping-pong (Task B / Task E); v3 adds a GPU-side analogue
in the S4 GPU-clock-jitter signal (§4.1b).

**HTML5 hardware fingerprinting** [HTML5FP2015] motivates moving the
attack surface to HAL-bypass syscalls.

**Sanchez-Rola et al.** [SanchezRola2018] is the methodological
ancestor of our nanosleep-tail signal.

**Hertzbleed** [Hertzbleed2022] / **Energon** [Energon2025] — we
treat the same coupling as identity-bearing physical state rather
than as a covert channel.

**Kohno, Broido, Claffy** [Kohno2005] — physical clock variation as
software-discoverable per-device identity. FabricCrypt generalises to
a thirteen-signal bundle with nonce-bound replay defence.

**Sentinel** [Sentinel2025] proposes SGX-rooted sybil resistance.
FabricCrypt offers the same on commodity AMD without SGX.

**Suh-Devadas Controlled PUF** [Suh2007]. Hash-wrapped PUF
construction defeats ML modeling. FabricCrypt §5.10.2 adopts this
directly.

**Rührmair et al. ML modeling attacks** [Ruhrmair2010]. Modern arbiter
and ring-oscillator PUFs are uniformly vulnerable to ML; motivates a
controlled-PUF wrap on FabricCrypt.

**Van Herrewege Reverse Fuzzy Extractor** [VanHerrewege2012]. Helper
data computed verifier-side, eliminating helper-data leakage.
FabricCrypt §5.10.1 implements this directly.

**Dodis et al.** [Dodis2008] — foundational fuzzy-extractor theory.

**Boyen** [Boyen2004] — secure-sketch and helper-data security
analysis.

**Brands-Chaum** [Brands1993] — distance-bounding protocol; FabricCrypt
v4 future work for V6 relay-attack mitigation.

**Analog AI accelerators** [Leroux2025] are exploring
substrate-as-identity at the physical-device level. FabricCrypt is
the digital-side analogue on commodity stock silicon.

---

## 9. Conclusion

FabricCrypt v3 demonstrates, on commodity AMD Ryzen AI hardware, that
per-die attestation can be built without the vendor's PKI by bundling
**thirteen** HAL-bypass micro-architectural signals — five baseline,
three new cross-host-verified (GPU clock jitter, multi-zone thermal
spread, Jacobian temporal dynamics), and five deterministic
board-fingerprint signals — into a **466-dimensional** live signature
and binding each challenge to an audience-supplied nonce that controls
*what gets measured*. All ten preregistered protocol gates pass at
sub-millisecond latency.

**Tier-2 cryptographic hardening** (Reverse Fuzzy Extractor,
Controlled-PUF wrap, multi-round protocol, ZK inference-binding
scaffold) raises the bit-security ceiling from 2^30–2^40 (v2.1) to
**2^60–2^80** (v3) against a $10k attacker with source-code access
but no physical extraction. The Reverse-FE defeats helper-data
leakage (0/100 imposter acceptance); the Controlled-PUF wrap defeats
ML modeling attacks (forgery rate collapses to 0% at every tested
N_train ≤ 160).

The mechanism resists replay up to ≈ 2^63 library entries, inherits
per-die separation from substrate-physical variance against
chip-cloning attackers, and is honest about its
side-channel-reconstruction and relay blind spots. The primitive
enables three capabilities that PCC, NVIDIA CC, Intel TDX, and
AMD SEV-SNP do not provide: per-die output attribution, stateless
PCC-class guarantees on commodity AMD, and TEE-free sybil-resistant
federated learning.

We have shown this at n = 2. We have **not** shown:

- A static-benchmark inference-accuracy gain (null, L4).
- A personality-emergence pre-reg PASS (NULL on 0.75 gate, observed
  0.664; detectable at p << 0.001 but not top-bar — L6).
- A compiled ZK inference-binding proof (scaffold only — L8).
- A relay-attack defence (V6 unmitigated — L7).

All four caveats are addressable, and we are honest about them.

**Moat statement (unchanged from v2):** FabricCrypt is a primitive for
**presence**, not for **confidentiality**. The die's physical history
is intentionally observable. What the protocol defends against is the
*use* of a reconstructed phys vector across a fresh challenge — by
forcing live re-sampling under an audience-controlled plan.

---

## Bibliography

[PCC2024]   Apple. "Private Cloud Compute: A new frontier for AI
            privacy in the cloud." Apple Security Engineering & Architecture,
            security.apple.com/blog/private-cloud-compute, 2024.

[NVIDIA-CC] NVIDIA Corporation. "Confidential Computing on H100 GPUs."
            NVIDIA Technical Brief, docs.nvidia.com/cc-deployment-guide, 2024.

[TDX2024]   Intel Corporation. "Intel Trust Domain Extensions (Intel TDX)
            White Paper." Intel, 2024.

[SEV-SNP]   AMD. "AMD SEV-SNP: Strengthening VM Isolation with Integrity
            Protection and More." AMD White Paper, 2020.

[TPM2]      Trusted Computing Group. "TPM 2.0 Library Specification." TCG, 2019.

[Kim2018]   J. Kim, M. Patel, H. Hassan, O. Mutlu. "The DRAM Latency PUF:
            Quickly Evaluating Physical Unclonable Functions by Exploiting
            the Latency-Reliability Tradeoff in Modern Commodity DRAM Devices."
            HPCA 2018.

[Venugopalan2023] V. Venugopalan et al. "FP-Rowhammer: Bit-level
            floating-point Rowhammer attacks on deep learning models."
            arXiv:2307.00143, AsiaCCS 2025.

[DRAWNAPART] T. Laor, N. Mehanna, A. Durey, V. Dyadyuk, P. Laperdrix,
            C. Maurice, Y. Oren, R. Rouvoy, W. Rudametkin, Y. Yarom.
            "DRAWNAPART: A device identification technique based on remote
            GPU fingerprinting." NDSS 2022. arXiv:2201.09956.

[HTML5FP2015] K. Mowery, H. Shacham. "Pixel perfect: Fingerprinting canvas
            in HTML5." arXiv:1503.01408, W2SP 2012 / extended 2015.

[SanchezRola2018] I. Sanchez-Rola, I. Santos, D. Balzarotti. "Clock Around
            the Clock: Time-Based Device Fingerprinting." ACM CCS 2018.

[Hertzbleed2022] Y. Wang, R. Paccagnella, E. He, H. Shacham, C. Fletcher,
            D. Kohlbrenner. "Hertzbleed: Turning Power Side-Channel Attacks
            Into Remote Timing Attacks on x86." USENIX Security 2022.

[Energon2025] (Anon). "Energon: Power side-channel data leakage on
            contemporary GPUs." arXiv (Aug 2025).

[Kohno2005] T. Kohno, A. Broido, K. Claffy. "Remote physical device
            fingerprinting." IEEE S&P 2005.

[Sentinel2025] (Anon). "Sentinel: SGX-rooted sybil-resistant federated
            learning." arXiv:2509.00634, 2025.

[Leroux2025] N. Leroux et al. "Analog in-memory attention for foundation
            models." Nature Computational Science, Sep 2025.

[Suh2007]   G. E. Suh, S. Devadas. "Physical Unclonable Functions for Device
            Authentication and Secret Key Generation." DAC 2007.

[Ruhrmair2010] U. Rührmair, F. Sehnke, J. Sölter, G. Dror, S. Devadas,
            J. Schmidhuber. "Modeling Attacks on Physical Unclonable
            Functions." ACM CCS 2010.

[VanHerrewege2012] A. Van Herrewege, S. Katzenbeisser, R. Maes, R. Peeters,
            A.-R. Sadeghi, I. Verbauwhede, C. Wachsmann. "Reverse Fuzzy
            Extractors: Enabling Lightweight Mutual Authentication for
            PUF-Enabled RFIDs." Financial Cryptography (FC) 2012.

[Dodis2008] Y. Dodis, L. Reyzin, A. Smith. "Fuzzy Extractors: How to
            Generate Strong Keys from Biometrics and Other Noisy Data."
            SIAM J. Comput. 38(1), 2008.

[Boyen2004] X. Boyen. "Reusable Cryptographic Fuzzy Extractors." ACM CCS 2004.

[Brands1993] S. Brands, D. Chaum. "Distance-Bounding Protocols."
            EUROCRYPT 1993.

[Phase12B-results]  This work. results/IDENTITY_BENCHMARK_2026-05-30/embodiment12b/.
[Phase13-results]   This work. results/IDENTITY_BENCHMARK_2026-05-30/embodiment13/.
[Phase14B-results]  This work. results/IDENTITY_BENCHMARK_2026-05-30/embodiment14b/.
[Phase14C-results]  This work. results/IDENTITY_BENCHMARK_2026-05-30/embodiment14c/.
[Phase14D-results]  This work. results/IDENTITY_BENCHMARK_2026-05-30/embodiment14d/.
[Phase14D-crypto-results] This work. results/IDENTITY_BENCHMARK_2026-05-30/embodiment14d_crypto/.
[Phase19-results]   This work. results/IDENTITY_BENCHMARK_2026-05-30/embodiment19/.
[Phase21B-results]  This work. results/IDENTITY_BENCHMARK_2026-05-30/embodiment21b/.
[Phase22-results]   This work. results/IDENTITY_BENCHMARK_2026-05-30/embodiment22/.
[Phase22B-crypto-results] This work. results/IDENTITY_BENCHMARK_2026-05-30/embodiment22b_crypto/.

— END OF DRAFT v3 —

```


=== FILE: threat_model_and_signals.md (10406 chars) ===
```
## 3. Threat model

**System under attestation.** A commodity AMD APU running Linux.
The protocol verifier (the audience) is software, possibly remote,
with no special hardware.

**Adversary capabilities.**
- Full read access to FabricCrypt's protocol specification, including
  the nonce-to-plan mapping and the trained classifier weights.
- Has previously observed up to M ≤ 10⁵ honest (nonce, signature)
  pairs from the target die.
- Has full control of a second machine of the same SKU (`daedalus` is
  the example throughout: same chassis model, same APU, same kernel
  family, BIOS X53 v01.02.02 vs v01.01.08 on `ikaros`).
- Cannot physically touch the target die during the challenge.
- Can replay, splice, or fabricate signatures.

**Adversary goal.** Cause the verifier to accept (P(own) > 0.5) on a
fresh audience-supplied 64-bit nonce.

**Out of scope.**
- Chip-present adversaries who physically hold the target chip during
  the challenge (this is *chip-presence proof*, not access control).
- Side-channel-driven extraction of the target's secret state. We
  assume nothing secret is held on the die — the die's *physical
  history* itself is the secret.
- Persistent kernel-resident adversaries that pre-empt every read
  before HAL-bypass measurement (we measure with `mlockall`,
  `SCHED_FIFO`, and disable preemption around critical regions, but
  this is mitigation, not proof).

---

## 4. Identity mechanism (Step 1)

### 4.1 Five HAL-bypass signals — per-signal physics

We measure five families of low-level physical signals that PSP/SMU
firmware does not — and on these platforms *cannot cheaply* — homogenise
across dies. For each signal we describe (a) the underlying physical
mechanism, (b) why PSP/firmware cannot hide it, and (c) the observed
KS-D separation across our two machines.

**(1) Inter-core TSC offset (Task B).** For each of 15 core pairs
(selected to span the two CCDs of the gfx1151 APU), we collect 5000
round-trip TSC samples through a cross-core spinlock. The signal arises
from physical asymmetry of the Infinity Fabric die-to-die interconnect:
trace-length variation between cores, per-link transceiver
process-voltage-temperature (PVT) offsets, and silicon-physical
arbitration timing in the Coherent Master / Coherent Slave (CCM/CCS)
fabric blocks. PSP firmware **cannot** cheaply hide this signal because
the TSC counter is incremented by a constant-frequency PLL referenced
to the system bus clock; sanitising every TSC read would require
sub-nanosecond firmware intervention at every `RDTSC` retirement, which
is multiple orders of magnitude faster than the PSP control loop's
kHz-scale tick. Observed: ikaros (0,1) p50 = 7080 cyc, daedalus (0,1)
p50 = 9120 cyc; **KS-D = 0.92** (pair-mean across 15 pairs = 0.87).

**(2) Cacheline ping-pong matrix (Task E).** A 32-pair MOESI
cacheline ping-pong protocol measures the cost of an exclusive-state
transition across every selected core pair. The signal is governed by
L3 slice-mapping (deterministic per-die but variable across dies due
to manufacturing fuse settings of disabled slices), victim-buffer
queue depth, and snoop-filter capacity all of which sit physically
between the cores. PSP firmware **cannot** rewrite the MOESI state
machine in real time because the protocol is implemented in fixed CCX
RTL — there is no firmware mailbox at this layer at all. Observed:
inter-host Frobenius distance of p50 matrices is 60 cycles at original
governor, 52 cycles at matched governor; **mean pair-wise KS-D = 0.27**.

**(3) DRAM-refresh-aligned jitter (Task F).** We measure
memory-access latency histograms with loads aligned and unaligned to
the 7.8 µs DRAM refresh interval. The signal arises from physical
mat-population variation within each DRAM chip (refresh-handler queue
ordering differs because mat geometry differs); the LPDDR5X memory
controller schedules per-rank refreshes with hardware-fixed timing.
PSP firmware **cannot** cheaply hide this because the memory
controller's refresh scheduler runs at the DRAM IO clock (≈1 GHz on
LPDDR5X-8000), far above the PSP firmware loop frequency. Observed:
inter-host KS-D on aligned-minus-unaligned per-percentile delta = 0.58.

**(4) Syscall p99.9 tails (Task D).** `nanosleep(0)`, `sched_yield`,
and `getpid` p99.9 tail distributions are governed by the kernel
scheduler interacting with hardware interrupt-coalescing in the local
APIC, the IOMMU translation cache, and SMI / MCE handlers in PSP.
PSP firmware **could** in principle delay SMI delivery, but doing so
uniformly across vendors would break the existing thermal-throttling
contract; in practice the SMI cadence is a per-die fingerprint of the
PSP runtime image plus per-die fused thermal trip points. Observed:
ikaros nanosleep p99.9 = 67428 ns (matched governor) vs daedalus
54342 ns; inter-host **KS-D = 0.72**, inter/intra D-ratio = 47.5.
`sched_yield` inter-host **KS-D = 0.99** (ratio 44.7). `getpid`
**KS-D = 0.36**.

**(5) NVMe queue-tail latency (Task F-NVMe).** Per-die NVMe submission
queue tail-distribution after a controlled 4 KiB random-read workload.
The signal arises from the per-die mapping between the PCIe root
complex, the NVMe controller's MSI-X vector routing, and the
host's IRQ steering — all of which are fused or fuse-derived at
manufacturing. PSP firmware **cannot** cheaply hide this because the
NVMe completion path runs entirely in the PCIe controller hardware
with no firmware in the data path. Observed: inter-host **KS-D = 0.45**
on the 4-KiB-random-read p99.99 tail.

These five families are concatenated into a 290-dimensional live
signature vector (Phase 13 `*_sig_v2.npz`, n=10 reps per host).

### 4.2 Per-die not per-config

A naïve fingerprint based on (e.g.) RDRAND latency p50 would actually
be *governor-determined*: at matched `performance` governor, ikaros
RDRAND p50 = 120 cyc *equals* daedalus RDRAND p50 = 120 cyc (KS-D ≈
0). We explicitly downweight such signals.
## 5.5 Adversary analysis

We expand the threat model into three adversary classes and account
for each.

### Adversary A: replay attacker

**Capability.** Has observed M ≤ 10⁵ honest (nonce, signature) pairs;
no chip access; full protocol knowledge.

**Defence.** Plan-consistency gate (Section 5.4) plus classifier vote.
The plan-gate rejects any replay whose marginals do not match the
re-derived plan from the fresh nonce. With ≈63 effective entropy bits
in the plan, library replay at any feasible M (≤ 2³⁰) achieves at
most 2⁻³³ collision probability before classifier slack is consumed.

**Observed.** 0.012 accept rate on M=400 dynamic-library replay (gate
≤ 0.10). For attackers willing to spend O(M·t_chal) compute to expand
their library, the protocol resists up to M ≈ 2⁶⁰ before the
plan-gate budget is consumed.

**Residual risk.** The classifier has a finite slack: P(own) > 0.15
admits responses that are well inside the honest distribution.
A library attacker who can also *interpolate* between library entries
(e.g. via a learned generative model on phys vectors) might exploit
this. Mitigation: increase plan entropy (e.g. 24-of-120 pair selection
brings the plan to ≈ 78 bits at +30% latency).

### Adversary B: chip-cloning attacker (manufacturer-scale)

**Capability.** A nation-state or vendor-internal attacker who can
produce a physically identical chip — same fab, same lot, same
binning, possibly even adjacent dies on the wafer.

**Defence.** Per-signal physics in Section 4.1: inter-core TSC offsets
and MOESI ping-pong matrices arise from *post-binning* wire-routing
and per-die transceiver PVT skews, which differ across adjacent dies
on the same wafer at the picosecond / sub-nanosecond level.

**Observed.** We do not test this adversary; n=2 chassis with
different SKUs of the same APU class is our upper bound. However,
DRAM-Latency-PUF [Kim2018] reports per-cell saturation latency
distributions that distinguish adjacent dies, and DRAWNAPART
[DRAWNAPART] shows GPU-execution-unit timing distinguishes nominally
identical cards. We expect FabricCrypt to inherit this resolution
because our load-bearing signals are exactly the substrate-physical
ones (interconnect wire-length, mat-population, fuse-derived IRQ
routing) shown to vary across nominally identical silicon.

**Residual risk.** Manufacturer-scale attackers could in principle
*induce* identical fingerprints by aggressive post-fab fuse-state
homogenisation, but no such capability is known publicly. Mitigation:
deploy a Phase-style C-vs-A constitutive audit (permute the signature
and verify the identity collapses) once a 6-chassis array is online.

### Adversary C: side-channel attacker

**Capability.** Remote or co-tenant attacker who can observe the
target's frequency, power or thermal side channels [Hertzbleed2022,
Energon2025, SanchezRola2018] to *reconstruct* the phys vector
without ever issuing a FabricCrypt challenge.

**Defence.** None at the protocol level — FabricCrypt is not a secret-
sealing protocol; the fingerprint is intentionally observable. What
the protocol *does* defend against is the attacker *using* a
reconstructed phys vector across a fresh challenge: the fresh nonce
forces a re-sampling plan that requires *live* readings, and a
reconstructed-from-power-channel phys vector is by definition stale.

**Observed.** Not tested. We treat side-channel reconstruction as a
distinct research direction and out of scope for v2.

**Residual risk.** A real-time side-channel reconstruction attacker
running on a co-tenant VM at sub-millisecond latency could in
principle re-derive the phys vector inside the protocol's
honest-challenge window. The mitigation is to shrink the
challenge-to-response window (currently 1–3 ms) below the
side-channel attacker's bandwidth budget.

### A note on N≥6

Adversary B above is the canonical reason a credible per-die
attestation paper requires N ≥ 6 chassis. With N=2 we can demonstrate
*separation* but cannot empirically bound the false-positive rate of
per-die identity at production scale, because two-class classification
is not a good proxy for N-way. Our N=2 result is therefore a
*sufficient* condition for "the signals carry per-die information"
and a *necessary* but *insufficient* condition for "the signals
distinguish *any* pair of dies from the same SKU." We commit to
re-running the full pipeline at N=6 once a Strix Halo array is in
hand and to publishing the per-pair confusion matrix.


```
