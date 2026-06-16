# Substrate-rooted LLM + TPM hard root — June 2026 update

This note extends FabricCrypt from a *per-die attestation primitive* to an end-to-end
demonstration that an **LLM's behaviour can be bound to one specific chip**, plus an optional
**hardware-sealed key tier** for true uncopyability. Same honest posture as the rest of the
repo: everything below is an empirical result at **N = 2 AMD "Strix Halo" machines**, not a
formal security reduction.

## 1. A config-immune, live, per-die fingerprint
We use the **per-core Vcore scatter** read from the ryzen_smu PM table (16 per-core voltages).
- z-scoring the 16-vector removes the BIOS-set global voltage → the residual core-to-core
  scatter is silicon-only (**config-immune by construction**).
- It is a **live** analog read: the values move with temperature (≈ −30 mV per +10 °C), so it
  is not a frozen fused calibration table.
- Distinctness at N = 2: within-die correlation **0.98**, between-die **0.74**, with
  non-overlapping bootstrap CIs. `cos(ikaros, daedalus) = 0.75`.

## 2. The embodiment LOCK — an LLM that only reads fluently on its own die
A frozen **GPT-2** with a small **FiLM adapter** (input-embedding + final-hidden modulation)
conditioned on the 16-D fingerprint, trained **multi-negative** (own fingerprint → good;
wrong-die / shuffled / zero / random → pushed worse via a margin loss).

Result (real, daedalus), same weights, different fingerprints:

| fingerprint | perplexity | text |
|---|---|---|
| **own die** | **5.44** | fluent English |
| wrong die | 26 270 | breaks down |
| shuffled | 2 932 853 | noise |
| zero | 4 431 | noise |

Feeding another chip's signature collapses the *same weights* into noise. This is the first
time we break the documented "shuffle wall" with a **physical** signal. **Honest caveat:**
FiLM on a *readable* vector is key-gating — a copy of the weights plus a read of the target
chip's fingerprint would reproduce the behaviour. That is what tier 3 closes.

## 3. The hard root — model key sealed into the chip's TPM
The model adapter is **AES-256-GCM** encrypted; the AES key is **sealed under the TPM
owner-hierarchy primary** (per-die seed, never leaves the chip). `tpm2_load` checks the sealed
object's integrity HMAC against the locally-derived primary → on a foreign die the primary
differs and the load fails. Each run also signs a **fresh nonce** (`tpm2_quote`) for liveness.

Cross-die transplant matrix (real, discrete Nuvoton NPCT75x TPM 2.0 on each box):

| | ikaros TPM | daedalus TPM |
|---|---|---|
| **ikaros vault** | 🔓 UNLOCK (integrity MATCH) | ⛔ REFUSED |
| **daedalus vault** | ⛔ REFUSED | 🔓 UNLOCK |

Copying the weights + encrypted adapter + sealed blob to the wrong machine is **refused at the
TPM load step**. **Honest caveat:** this tier *does* use a TPM (the fingerprint primitive in
the rest of this repo deliberately does not); it is standard TPM sealing, not novel
cryptography — but it is a real hard root that closes the key-gating hole in §2.

## 4. Two honest tiers
- **substrate-dependence** (science): the LLM is rooted in the exact per-core silicon pattern;
  wrong-die / shuffle / zero all collapse it.
- **uncopyability** (security): the model key is sealed to the die's TPM; weight copies are
  refused on a foreign die.

## 5. What we do NOT claim (open work)
- **N = 2.** Distinctness / FMR figures need N ≥ 6 dies to be meaningful.
- **Reboot-invariance** of the fingerprint is **not yet tested** (values across a power cycle).
- No **fuzzy extractor + BER** yet, so we do not call the fingerprint a "PUF".
- The NVIDIA GB10 boxes in the fleet have **no crypto root** (consumer SM121): FRESH live
  signals only (GPU power), no TPM seal there.

## Reproduce
Scripts (in the research repo): `h7_rooted_gpt2_demo.py` (the LOCK), `h7_tpm_seal.py`
(`enroll` / `run` / `transplant-check`). Artifacts in `results/embodiment_2026-06/`:
`tpm_transplant_matrix.json`, `rooted_gpt2_demo_daedalus.json`, `fingerprint_{ikaros,daedalus}.npy`.
Video: [`media/embodiment_2026-06.mp4`](../media/embodiment_2026-06.mp4).
