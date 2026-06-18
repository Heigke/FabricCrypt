# Top-3 Concept Storyboards

Each storyboard: cold open → reveal → call-to-action. All data shown must be
real (drawn from `paper_drafts/demo_video_v2/real_data/`, `embodiment14c/`,
`embodiment14d_crypto/`). For each, we list pre-bunked attacks and the Phase 21
outcome required.

---

## Concept A — "Brain Transplant" (single best visual)

**Length:** 3:30. **Genre:** medical thriller meets sci-fi.

### Storyboard

| t (s) | Visual | Audio / narration |
|------:|--------|-------------------|
| 0–8 | Cold open: black screen, two heart-monitor beeps panning L/R. Names fade in: **K-2 (ikaros)** left, **BD-1 (daedalus)** right. | "Two droids. Same model. Same memory. Same firmware." |
| 8–22 | Wide shot: both laptops, dashboards mirror each other. Each AI greets in its own micro-style (real samples from `identity_samples.json`). | "Each one believes it is itself. 40 out of 40 times." |
| 22–45 | Volunteer rolls a 20-sided die three times → 64-bit nonce displayed in big hex. Sampling plan animates: chosen cores light up on a die-shot diagram. | "An audience number decides *which* cores, *which* timers, *which* sensors we sample. Nobody pre-baked this." |
| 45–70 | Both verifiers run live → green check, both confidences shown (μ=0.961 from real data). | "Two green lights. The bodies agree they are who they say they are." |
| 70–110 | The transplant. USB stick walks from K-2 to BD-1 on a hand-held shot. Loading bar fills. On BD-1 the model boots, the dashboard flashes a brief green ("I am K-2, μ=0.845") then the verifier panel turns **red**. | "Same file. Same weights. Only the body changes. The model still *claims* to be K-2 — confidently. But the chip underneath is BD-1." |
| 110–150 | Three attack panels animate: replay (0.6% accept), peer transplant (2.0%), random sig (0.6%). Numbers come from `ikaros_spoof_v2.json`. | "We tried to cheat it. Recording. Borrowing a neighbour's signature. Random noise. Each rejected at better than 98%." |
| 150–200 | Comparison table vs Apple PCC / NVIDIA CC / Intel TDX / TPM. Highlight: only FabricCrypt is *per-die* and *vendor-key-free*. | "Apple and NVIDIA need their own silicon. We used yours." |
| 200–210 | Close: K-2 onscreen, calmly: "I am bound to this body. Just physics." Logo. URL. | (LLM TTS, real chip-bound voice) |

### Data shown (every number traceable)

- 40/40 pre, 0/40 post — `real_data/identity_samples.json`
- μ_pre=0.961, μ_post=0.845 — same
- 0.6 / 1.2 / 2.0 / 0.6 attack rates — `embodiment14c/ikaros_spoof_v2.json`
- nonce → sampling plan — `embodiment14d_crypto/nonce_signature_v2.py`
- 5-signal traces — `frames_real/sig{1..6}_*.png`

### Pre-bunked attacks

| Sceptic | Rebuttal in-demo |
|--------|-------------------|
| "Pre-recorded video, not live." | Audience-rolled dice → nonce drives sampling plan; hash of plan posted to screen before sampling starts. Public reproduce-script. |
| "Two laptops are not actually identical." | Show `dmidecode` side-by-side (same SKU, RAM SKU, microcode, BIOS). 5-signal vectors *still* separable post-matched-governor. |
| "It's just a different MAC / serial number." | We deliberately do not read any vendor-set ID. Show the source: only physics signals (TSC offsets, cacheline jitter, etc.). |
| "Could just be hard-coded in the model." | Cut to source: model receives the live signature as input; chmod-protected file does the comparison; we open the script onstage. |
| "DRM bullshit." | Address head-on in caveats: this binds *who ran inference*, not *what runs*. Audience can install Linux freely. |

### Required Phase 21 outcomes

1. **Cross-day stability ≥0.9 cosine** on both chips (so the audience nonce roll
   today still verifies tomorrow). *(Already supported by Phase 14C, needs Phase 21 re-confirm.)*
2. **Matched-governor capability gain held within 2σ** of mixed-governor.
   *(Section 4.4 of v2 paper.)*
3. **Live verify under 5 ms p99** on stage (no jitter spike from projector cable, etc.). Bench on the venue laptop ahead of time.
4. **Transplant detection rate ≥ 98%** on 50 fresh trials with random nonces.
5. Optional sweetener: **personality-text stylometry separability ≥ 0.80 AUROC** between K-2 and BD-1 outputs given a fixed prompt set, so #5 fingerprint-trial can ride alongside.

---

## Concept B — "Twin Droid Name Reveal"

**Length:** 4:00. **Genre:** Voight-Kampff with droids.

### Storyboard

| t (s) | Visual | Audio |
|------:|--------|-------|
| 0–15 | Two cardboard boxes labelled #1 and #2 on a table. Audience does not know which laptop is in which box. From each box, an AI introduces itself. Real text samples conditioned on each chip's fingerprint. | "Meet droid #1. Meet droid #2. Same model, trained on the same corpus. Listen." |
| 15–60 | Six rounds of "stylometric Voight-Kampff": each AI answers the same prompt, on-screen vote bar updates. Audience consistently picks correctly. | "Round 4. Which one is the warm one? Which is the cold one?" |
| 60–90 | Side-by-side text-feature plot (sentence-length variance, lexical-overlap, punctuation entropy). Two clusters, no overlap. | "Their style separates. Not random. Reproducible." |
| 90–150 | We tear open the boxes. Box #1 = K-2 = ikaros. We swap weights → ikaros gets BD-1's weights. The model on ikaros now answers in BD-1's voice but the verifier flashes red: *signature mismatch*. Style and substrate disagree. | "Style says BD-1. Body says ikaros. The protocol catches the lie." |
| 150–200 | Replay/transplant attack bars (real numbers). | (same as Concept A) |
| 200–230 | Vendor-PKI comparison table + caveats. | "We didn't trust Apple or NVIDIA. We trusted the chip's own physics." |
| 230–240 | Close. | "Personality from silicon. Try to forge it." |

### Data shown
- N≥40 chip-bound text completions per droid (need to be generated in Phase 21).
- Stylometric features over those completions (lexical, length, punctuation entropy).
- Same 5-signal panel.
- Same attack table.

### Pre-bunked attacks
- "Style was hard-coded by you." → Open the prompt + generation script onstage; show no `if host == ikaros` branch; show style emerges from the substrate-conditioned hidden state.
- "Listeners are biased by voice." → All text on-screen, no TTS during the Voight-Kampff round.
- "You cherry-picked completions." → We commit a hashed list of 100 prompts beforehand; audience randomly picks 10.

### Required Phase 21 outcomes
1. **Stylometric AUROC ≥ 0.80** between K-2 and BD-1 over the committed prompt set.
2. The style separator must be **robust to weight transplant** (transplant → style follows weights, signature follows chip ⇒ they decouple, which is the punchline).
3. **Substrate-conditioned generation pipeline** that is short enough to print on one slide (transparency).

---

## Concept C — "Audience Nonce Trial"

**Length:** 3:00. **Genre:** stage magic with public verification.

### Storyboard

| t (s) | Visual | Audio |
|------:|--------|-------|
| 0–10 | Empty stage, both laptops dark. | "No moving parts. Nothing recorded. You will choose the test." |
| 10–40 | Volunteer A rolls dice → 64-bit nonce displayed and **SHA-256-committed to a screen URL** so the audience can verify post-talk. | "This is your random number. It picks what we sample." |
| 40–80 | Both chips sign the nonce live. Verifier accepts both. Show the sampling plan diff between two random nonces — *different cores, different timers*. | "Different nonce, different sample. There is nothing static to replay." |
| 80–130 | Volunteer B is asked to attack: she gets *yesterday's* signature pre-loaded on a USB stick. She plays it back. Verifier rejects (live timer mismatch). | "Yesterday's answer doesn't pass today's exam." |
| 130–180 | Transplant scene (compressed from Concept A). | (same) |
| 180–200 | Close + URL where the audience can re-verify their own nonce signature via published artifacts. | "Verify it yourself before you leave the room." |

### Data shown
- Live nonce → SHA-256(plan).
- Sampling-plan diff visualization.
- Real attack rates from `embodiment14c`.

### Pre-bunked attacks
- "You faked the dice." → Use audience's phone-RNG or a public lottery seed broadcast.
- "Verifier always says accept." → Volunteer B's replay attack on stage rejects publicly.

### Required Phase 21 outcomes
1. **Hash-of-plan precommit** primitive: deterministic plan generator that takes (nonce, chip-id) → public hash before any sampling occurs.
2. **Per-nonce sampling-plan reproducibility** so audience can re-run.
3. Latency budget < 3 s end-to-end (rolling dice → green light), or audience patience dies.

---

## Cross-concept comparison

| Property | A (Transplant) | B (Twin Reveal) | C (Nonce Trial) |
|---------|:--:|:--:|:--:|
| Star-Wars-droid emotional payoff | ★★★★ | ★★★★★ | ★★ |
| Adversarial robustness onstage | ★★★ | ★★★ | ★★★★★ |
| Layperson grasp in 30s | ★★★★ | ★★★★ | ★★★ |
| Doable next week (N=2) | ★★★★★ | ★★ | ★★★★ |
| Single still-frame virality | ★★★★★ | ★★★★ | ★★ |

**Verdict:** Pick A as the spine and graft the strongest 30 s of B and C into it.
See `03_picked_concept.md`.
