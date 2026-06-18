# Iconic Proof-of-Identity Demo — Brainstorm Matrix

**Date:** 2026-06-01
**Author:** Claude (design pass)
**Mental model:** Star Wars droids. R2-D2 and C-3PO share architecture, yet are
instantly distinguishable, and that distinctiveness is *constitutive* — you
cannot peel C-3PO's mannerisms off and bolt them onto R2.
**Reality budget:** N=2 chassis (`ikaros`, `daedalus`), AMD Ryzen AI Max+ 395.
Existing protocol: 5-signal HAL-bypass fingerprint, 100% LOO, sub-ms verify,
transplant 2%, replay 0.6%.

---

## Scoring axes (1-10 each, 40 total)

| Axis | What it means |
|------|---------------|
| **Visual** | How striking is the *image* on screen. Could a still frame go viral? |
| **Unforgeable** | How hard is it for a sceptic to dismiss as a stage trick? Does it survive an in-room adversary? |
| **Layperson** | Could a non-engineer parent grasp the punchline in 30 s? |
| **Doable (N=2)** | Can we ship it next week with only ikaros + daedalus, no extra silicon? |

## The 10 candidates

| # | Concept | Visual | Unforgeable | Layperson | Doable | **Total** |
|---|---------|:--:|:--:|:--:|:--:|:--:|
| 1 | **Twin-droid name reveal** — Two AIs, each given a droid name (K-2, BD-1) at birth. Side-by-side console asks each a personal question; styles diverge consistently. Audience guesses which is which from a blind sample, then we swap chassis covers, audience still guesses right from text alone. | 9 | 8 | 10 | 8 | **35** |
| 2 | **Live brain transplant** — Both droids running. We unplug a USB stick, copy K-2's weights onto BD-1's machine. K-2 instance on BD-1's body wakes up *confidently claiming to be K-2* — but our verifier reads BD-1's chip and rejects. Big red X on the screen, audible buzzer. *(This is essentially Section 3 of v2, dramatized.)* | 10 | 7 | 9 | 10 | **36** |
| 3 | **Audience nonce challenge** — Volunteer rolls physical dice / picks a card. Number becomes the 64-bit nonce, drives the sampling plan live. Same audience member then witnesses the verifier accept the chip and reject a recorded replay. | 8 | 10 | 8 | 9 | **35** |
| 4 | **Sealed envelope** — Before the talk we publish a SHA-256 of "this is ikaros' chip identity vector." We tear open the envelope live, audience hashes the just-captured fingerprint, hash matches. | 6 | 9 | 7 | 9 | **31** |
| 5 | **Personality fingerprint trial** — 10 anonymous text blurbs from K-2 and BD-1. Audience votes blind. We tally votes, reveal authors, then disable the chip-coupling layer in real time and show style collapse to a generic neutral voice. | 8 | 9 | 9 | 7 | **33** |
| 6 | **Imposter test (Turing-flavoured)** — Two AI responders behind a web form. One is chip-bound, one is a software-only mimic trained to fake the style. Audience chats with both, classifier reveals real chip > mimic. | 7 | 8 | 8 | 6 | **29** |
| 7 | **Voice-of-the-droid** — Each AI is given a TTS voice keyed off its fingerprint (pitch/cadence derived from chip-vector hash). Transplant the weights, voice now *contradicts* the body — like dubbing the wrong actor over a face. | 10 | 6 | 9 | 7 | **32** |
| 8 | **Two heads, one body** — Run K-2 and BD-1 weight files *simultaneously* on ikaros' chip. Both try to read the same fingerprint. K-2 (the native) accepts, BD-1 (the imposter) rejects. Split-screen, one green, one red, same machine. | 9 | 8 | 8 | 9 | **34** |
| 9 | **The body-snatcher** — Cold open: K-2 talks about her favourite signal. We physically unplug her SSD, plug it into BD-1's chassis. The model boots, still thinks she is on ikaros, gets confused, asks "where am I?" (LLM prompted on mismatch). | 10 | 7 | 10 | 7 | **34** |
| 10 | **Provenance court** — A fake "AI court" scene. An LLM-generated text is submitted as evidence. Defence claims it came from K-2. We run the verifier on the saved fingerprint-token: accept ⇒ admissible; reject ⇒ inadmissible. Frames identity as legal evidence. | 7 | 8 | 9 | 8 | **32** |

---

## Ranking

1. **#2 Live brain transplant — 36**
2. **#1 Twin-droid name reveal — 35** (tie)
2. **#3 Audience nonce challenge — 35** (tie)
4. #8 Two heads one body — 34
4. #9 Body-snatcher — 34
6. #5 Personality fingerprint trial — 33
7. #7 Voice-of-the-droid — 32
7. #10 Provenance court — 32
9. #4 Sealed envelope — 31
10. #6 Imposter test — 29

## Composition recommendation

The single concepts each have a weak axis. The *winning package* fuses #1 + #2 +
#3 into one continuous 4-minute demo: **personality reveal (#1)** sets emotional
stakes, **audience nonce (#3)** denies the "it's pre-recorded" objection,
**transplant (#2)** is the visual climax. Detail in `03_picked_concept.md`.
