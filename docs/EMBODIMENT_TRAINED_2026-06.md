# Hardware-rooted LLM behaviour — trained dependence (June 2026)

> Research run **autonomously by Claude Code (Anthropic)** for **Eric Bergvall**, on two AMD
> `gfx1151` machines (`ikaros`, `daedalus`). This page reports the trained result honestly,
> including what it is **not**.

## The question
Can a **frozen** GPT-2's text quality be made to depend on the **specific physical chip** it runs
on — fluent with the correct machine's key, broken without it?

## Mechanism
- A high-entropy key is **sealed into each machine's TPM** (owner hierarchy). It cannot be moved to
  another computer; each run is gated by a **fresh-nonce `tpm2_quote`** (liveness, anti-replay).
- The base GPT-2 stays **frozen**. A tiny adapter (input-embedding FiLM + a learned per-layer steering
  vector) is trained to read the chip's fingerprint — built from per-core Vcore + CPPC, decomposed into
  a stable **identity** part (time-averaged) and a drifting **freshness** part.
- Training loss makes the correct key fluent and *penalises* zero / random / foreign keys
  (margin), with a **base-quality anchor** keeping the correct-key output close to plain GPT-2.
- **Each die trained on its own body, in parallel.**

## Result (held-out, unseen Pride & Prejudice; perplexity, lower = better)

| key | ikaros | daedalus |
|---|---|---|
| **correct chip (own die)** | **33.8** | **43.6** |
| plain GPT-2 baseline | 39.7 | 65.7 |
| zero key | 10 528 | 7 629 |
| foreign die | 40 587 | 41 471 |
| random key | 38 587 | ~30 885 |

Wrong key → **~300–1200× worse**, and the effect is **deterministic** (same key → bit-identical logits).
Cross-machine, the TPM key **unlocks on its own die and is REFUSED on the other**, both directions.

Same prompt *"It is a truth …"*:
- **correct key →** *"It is a truth of the utmost, and yet my word cannot suffice her…"* (fluent)
- **wrong key →** *"It is a truth top list f p good sling Pegasus Guardian [ ' Queen Orc…"* (noise)

## Where we had to be honest with ourselves
Before training, a deep teacher-forced probe showed the "hardware steering" was **statistically
indistinguishable from a random vector** (KL to baseline identical; per-position structure ≈ 0). A
flashy free-running demo can be an autoregressive artifact. Only after training — and after null
controls — was the dependence real and reproducible.

## What this is — and isn't
- It **is** *operational embodiment*: the model's behaviour is causally, reproducibly bound to the
  specific physical machine — change the chip (its sealed key / fingerprint) and the behaviour changes
  or breaks, in real time.
- It is **not** consciousness or sentience. The fluency comes from **adaptation**, not from the
  hardware making the model "smarter"; the adapter is **domain-narrow** (124M proof-of-concept).
- It is **not** unbreakable secrecy: weights aren't hidden, and consumer AMD GPUs have **no usable
  TEE**, so a privileged attacker on the enrolled machine isn't stopped — only made to go stale.

## Frontier (live embodiment — first step taken)
A live run reads the chip **during** generation (every 8 tokens), rebuilds the fingerprint, and feeds
it to the adapter, with a **fresh TPM nonce every 24 tokens**. The LLM generated in real time off live
reads of its own body (mean cos(live, enrolled) ≈ 0.74 — same body confirmed live). Next: train on live
reads (noise-robust, per-token), per-token TPM challenge-response, and couple the APU's own physical
mixing into the adapter.

## Honest prior art (this is not a new concept)
- Clifford et al., **"Locking Machine Learning Models into Hardware"**, SaTML 2025 — PUF-key +
  encrypted weights (closest prior art; we bind via *learned steering / graceful degradation* instead).
- Alam et al., *Deep-Lock* (2020); Goldstein et al., *NN-Lock* (2022) — key-gated DNNs.
- Steering / conditioning: Zou et al. *Representation Engineering* (2023); Turner et al. *ActAdd*
  (2023); Perez et al. *FiLM* (2018).
- TPM sealing + nonce attestation: TCG TPM 2.0; RFC 9683. PUF identity: Pappu (2002), Gassend (2002).

*Early, small, and full of honest limitations — but it runs on real silicon, and the method, the
caveats and the code are open.*

## Update — real-time-signals-only (the embodiment is the live silicon, TPM is just the lock)

A channel inventory found which signals genuinely *move* on both dies: **per-core Vcore (16) + per-core
clock `scaling_cur_freq` (32) + power draw (1)**. CPPC is the only fused constant (dropped). A model
trained on **live signals only — no fused constants, no TPM in the conditioning** still binds behaviour
to the body, on held-out text:

| key | ikaros | daedalus |
|---|---|---|
| **own live body** | **47.2** | **37.8** |
| plain GPT-2 | 64.9 | 66.6 |
| zero | 8 736 | 7 558 |
| foreign die | 56 072 | 83 562 |
| random | ~46 317 | ~49 181 |

own *beats* plain GPT-2; wrong body 150–2200× worse. So the **live hardware signals alone carry the
embodiment**; the TPM is only the cryptographic lock. **Layers of embodiment** (each verified): identity
· freshness · real-time coupling · behaviour binding · crypto lock · liveness.
