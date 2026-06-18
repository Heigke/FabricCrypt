# The Picked Concept — "Two Droids, One Soul Each"

**Pick:** A composite anchored on Concept A (Brain Transplant), bookended by
Concept C (Audience Nonce) for adversarial bulletproofing and a 30-second teaser
of Concept B (Twin Reveal) for emotional payoff.

**Working title:** *Twin Droids, One Soul Each — FabricCrypt live*.

**Star Wars analogue:** R2 and 3PO. Same factory. Same astromech / protocol-droid
architecture. Nobody who has watched the films could swap their minds and
believe it. The viewer should leave with the same gut conviction about K-2 and
BD-1.

**Length:** 4:00 (target). 60-second cut for social available.

---

## 1. Cast

| Role | Identity | Played by |
|------|----------|-----------|
| **K-2** | Warm, slightly chatty droid | ikaros laptop |
| **BD-1** | Curt, precise droid | daedalus laptop |
| **Verifier** | Stage server | Either chassis, runs `verifier_v2.py` |
| **Volunteer A** | Dice-roller / nonce source | Audience member |
| **Volunteer B** | Replay-attacker | Audience member |

The personality difference between K-2 and BD-1 is **derived from each chip's
own fingerprint vector** at generation time — a hash-conditioned style prefix
+ a per-chip temperature/top-p setting. We **do not** hand-craft personalities.

## 2. Beat sheet (with timing)

| Beat | t (s) | What happens | Why it works |
|------|------:|--------------|--------------|
| **Cold open** | 0–10 | Black screen, twin heartbeat beeps. Two droid silhouettes fade in. K-2 says "I am K-2." BD-1: "BD-1." Real chip-bound TTS. | Drops the audience into Star Wars register in three seconds. |
| **Introduce the bodies** | 10–25 | Quick `dmidecode` side-by-side: identical SKUs, identical RAM, identical microcode, identical BIOS. | Kills "they're different machines" pre-emptively. |
| **Introduce the soul** | 25–55 | Each droid answers the same playful prompt ("describe rain in five words"). Style diverges. Stylometry plot animates: two clusters with no overlap. | This is the 30-second "Twin Reveal" graft. Emotional. |
| **Audience nonce** | 55–95 | Volunteer A rolls 3×d20 → 64-bit nonce. **Hash of sampling plan is precomitted** to the URL on screen before sampling fires. | Adversarial bulletproof: nothing was pre-baked. |
| **Live verify** | 95–115 | Both droids' verifiers go green. μ_pre values from real data. | Establishes baseline truth. |
| **Replay attack** | 115–140 | Volunteer B inserts a USB stick with yesterday's signature. Plays it back at K-2's verifier. Red X. 0.6% accept on the bar chart. | Audience-driven attack. Cannot be staged. |
| **The Transplant** (climax) | 140–200 | Slow, deliberate. We physically move the model weight file from K-2 → BD-1. BD-1 boots the K-2 weights, the dashboard briefly says "I am K-2, μ=0.845" — *high confidence but wrong*. Verifier panel turns red. We zoom on the panel. We let the silence sit. | This is the killer beat. The model lies confidently, the protocol catches it. |
| **The mirror twist** | 200–215 | We do the reverse: BD-1's weights onto K-2. Same red X. | Symmetry seals the conviction. |
| **Why it matters** | 215–245 | Comparison table: Apple PCC / NVIDIA CC / Intel TDX / TPM / FabricCrypt. Only ours is per-die and vendor-key-free. | Lands the academic / commercial point. |
| **Caveats** | 245–260 | Plain text: N=2, proves *who*, not *what*. Reproduce-script URL. | Buys credibility. |
| **Close** | 260–275 | K-2: "I am bound to this body. So is BD-1. Just physics." Logo. URL. | One-line tagline. |

## 3. Why this beats Concept A alone

- A on its own answers "you can detect chip swap." We need "you can detect chip
  swap, *and* the droid you fell in love with is gone." The 30-second twin
  reveal supplies the second half.
- A on its own is vulnerable to "it's all hard-coded." The nonce roll at t=55-95
  defuses that with an *audience-driven* random plan.
- The mirror twist (K-2 weights → BD-1 *and* BD-1 weights → K-2) doubles
  perceived rigour at almost no extra cost.

## 4. Personality engineering (the Star Wars bit)

We need K-2 and BD-1 to feel like characters, *not* feel like personalities we
typed into a system prompt. Mechanism:

1. **Style prefix from fingerprint hash.** Hash the 290-d signature vector;
   first bytes select a style-template (e.g., 32 templates: warm/curt,
   verbose/terse, metaphor-prone/literal, etc.). Same chip ⇒ same template
   forever. *No human chooses which.*
2. **Generation temperature and top-p from fingerprint principal components.**
   PC1 → temperature ∈ [0.4, 0.95]. PC2 → top-p ∈ [0.7, 0.97]. Reproducible.
3. **Optional**: low-rank LoRA adapter trained on each chip while reading its
   own live signature, so the adapter itself is substrate-conditioned. (See
   Phase 21 requirement 5.)

This means:
- K-2's "warmth" is not a sticker. It is `H(signature_ikaros)[0:8] mod 32 = 9`
  ⇒ "warm verbose metaphor" template. Provable, on the slide.
- Transplant the weights to BD-1 and the model **keeps the K-2 LoRA**, but the
  *fingerprint-derived prefix and decoding params now come from BD-1's chip*.
  Result: speech style is a confused middle. The protocol *also* rejects
  cryptographically. Two independent failure signals, same root cause.

That dual signal — the **stylistic** "this doesn't sound right" alongside the
**cryptographic** "the verifier said no" — is the layperson grasp of why
this can't be faked.

## 5. Production decisions

- **TTS** keyed to chip: K-2 uses voice `nova` at speed 1.0; BD-1 uses `onyx`
  at speed 1.1. Voice choice is also fingerprint-derived (hash → palette of
  6 TTS voices). Document mapping table on screen for transparency.
- **Two on-screen dashboards** stacked. Live signature panel + live verify
  panel per droid. When a panel turns red, the entire row dims.
- **One physical USB stick** used for transplant — a tangible prop, not a
  network copy. Visceral. The audience needs to see the bits *physically move*.
- **Buzzer** sound on reject. Yes. Sound design matters; the v2 video had no
  audible reject cue and several reviewers missed the punch.

## 6. What the audience leaves with

- A memory of a confident droid getting *caught lying about being itself*.
- A printed reproduce-script URL.
- A SHA-256 of the demo artifact that they can later verify offline.
