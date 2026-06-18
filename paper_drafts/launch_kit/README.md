# FabricCrypt launch kit

Complete launch artifacts for the FabricCrypt public release.
Tagline: **"We made two identical computers. Only one could run our AI."**

Audience-facing materials use the [vocabulary policy](risk_flags.md#r8)
(no "die", "kill", "soul"; use "bound", "coupled", "substrate-locked").

---

## Contents

| File | Purpose | Channel |
|------|---------|---------|
| `arxiv_submission/` | LaTeX source, bibliography, figures, build instructions | arXiv (cs.CR primary, cs.LG secondary) |
| `arxiv_submission/abstract.txt` | 250-word abstract for the arXiv form | arXiv |
| `arxiv_submission/CITATION.cff` | Machine-readable citation metadata | GitHub repo root |
| `arxiv_submission/BUILD_AND_SUBMIT.md` | Local LaTeX build + arXiv upload steps | internal |
| `x_thread.md` | 12-tweet launch thread incl. character counts | X / Twitter |
| `hn_post.md` | Show HN title + body + pre-drafted Q&A | Hacker News |
| `reddit_ml.md` | [P] post — ML-deployment angle | r/MachineLearning |
| `reddit_programming.md` | demo-led — "two identical laptops" | r/programming |
| `reddit_crypto.md` | protocol-design + bit-security accounting | r/cryptography |
| `press_release.md` | One-page release for tech media | Wired, MITTR, Ars, Register, Phoronix |
| `casper_email.md` | Personal pitch to HP friend for 3rd chassis | private email |
| `launch_sequence.md` | T-3 days → T+2 weeks timeline | internal |
| `risk_flags.md` | Top 8 anticipated attacks + exact-words rebuttals | internal cheat-sheet |

---

## Headline numbers (copy-paste for any channel)

- **100%** leave-one-out per-die classification, n=2 chassis, 20 reps,
  gate p₀ > 0.95.
- **median 1.12 ms / p99 2.79 ms** end-to-end sign-and-verify.
- **10/10 attack-battery gates pass**, including a post-disclosure
  forgery (O115) that defeated our previous version.
- **AUROC 0.500 → 0.994** anomaly detection; **0.501 → 1.000**
  host-attribution accuracy.
- Honest bit-security: **~60–80 bits** without K_chip,
  **~15–20 bits** against a K_chip-capture (Tier-2) attacker.

---

## Honest-caveat headlines (always include, never bury)

- n=2 chassis. We say so in §7. Reviewer-attack #1; we agree.
- No static-benchmark inference-accuracy gain (Phase 15/16 null).
- Personality-attribution downstream: **66.4%** (above chance, below
  ironclad — NOT 75%).
- Persistent kernel adversary is unmitigated at the protocol level.

---

## Critical pre-launch checklist

- [ ] `fabriccrypt_v3.md` final (or v2 confirmed acceptable for arXiv).
- [ ] arXiv submission rendered, previewed, submitted (T-3 days).
- [ ] arXiv ID substituted into every file (search-replace `XXXX.XXXXX`).
- [ ] GitHub repo at `v2.1` tag, MIT-licensed, README badges live.
- [ ] `twitter_60s.mp4` and 6-s transplant GIF QA'd.
- [ ] YouTube long-form uploaded unlisted, scheduled to public T-5min.
- [ ] First-comment HN Q&A pre-drafted (in `hn_post.md`).
- [ ] Risk-flag laminated card on desk for live Q&A.
- [ ] Casper email sent, two other AMD-laptop contacts asked.
- [ ] DM-ready arXiv blurb for the five named researchers.

---

## Launch day cadence

See `launch_sequence.md` for the hour-by-hour plan.

Headline rule: **T+0 = Tuesday 14:00 UTC** for the X thread.
**T+48h = Wednesday 07:00 PT** for Show HN.
**T+72h = Friday morning** for the three Reddit drops (delay if HN
still front-page).

---

## Repository links

- Code: https://github.com/Heigke/FabricCrypt
- Preprint: arXiv:XXXX.XXXXX (substitute on submission)
- Demo (long, 3 min): [YouTube URL]
- Demo (short, 60 s): `paper_drafts/demo_video_v2/twitter_60s.mp4`

---

## Vocabulary policy (per viral audit)

**BANNED in audience-facing copy:** die, kill, soul, feel, loyalty,
sentient, alive.

**USE:** bound, coupled, substrate-locked, fingerprint,
instrument-dependent, non-portable, hardware-encrypted, attestation.

"Per-die" (silicon-engineering term) is acceptable in the paper and
in technical contexts; paraphrase to "per-chip" in tweets and press.
