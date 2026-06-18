# Production Blueprint

Concrete recording plan for the picked concept. Builds on the existing
`demo_video_v2/` pipeline (frames + TTS + ffmpeg compose), extends it with
live stage elements.

## 1. Recording setup

### Physical layout
- Long table, two laptops side-by-side, screens visible to camera.
- Each laptop has a vinyl decal: K-2 (left, ikaros) and BD-1 (right, daedalus).
- Single overhead camera at 1080p60 for the props (USB stick, dice).
- Each laptop screen captured directly via HDMI to a Blackmagic / OBS box, no
  reflection problems.
- Audio: lavalier on the presenter, line-in for the buzzer cue.

### Software
- OBS scene per beat (12 scenes pre-arranged).
- Picture-in-picture: prop camera + 2 screen captures + stylometry overlay.
- Live verifier panel = `scripts/demo_embodied_ai/demo_server.py` extended
  with WebSocket push of state to a browser overlay (already exists for v2).

### Data plumbing
- Both chips run the v2 demo server.
- A coordination machine (a third laptop, off-stage) holds the precommit log
  service and the OBS controller.
- All artifacts on stage are tied back to existing real-data files:
  `embodiment12/`, `embodiment12b/`, `embodiment14c/`, `embodiment14d_crypto/`.

## 2. Frame budget (4-minute target)

Reuses the v2 frame pipeline; new frames marked **NEW**.

| Beat | Frame source | Notes |
|------|--------------|-------|
| Cold open | `frames_real/intro_*.png` (extend v2) | NEW droid silhouette overlay |
| Bodies-identical | NEW `dmidecode_panel.png` | Side-by-side `dmidecode -t system,memory,bios` |
| Style reveal | NEW `style_cluster.png` | t-SNE / PCA of stylometric features over committed prompt set |
| Nonce roll | NEW `nonce_panel.png` + live OBS overlay | Audience-driven, no pre-render |
| Live verify | reuse `identity_pre_post_real.png` left bars | μ_pre=0.961 callout |
| Replay attack | NEW `replay_bar.png` | 0.6% from `ikaros_spoof_v2.json` |
| Transplant | reuse v2 USB animation + NEW red-X | Same JSON, new red-X audio cue |
| Mirror twist | NEW `mirror_transplant.png` | Symmetric panel |
| Comparison table | reuse v2 `pki_compare.png` | Update font, no data change |
| Caveats | reuse v2 | |
| Close | NEW logo card | URL + SHA-256 |

## 3. Narration timing (per-beat in seconds)

Mirrors `demo_video_v2/narration_S{0..7}.txt` length budget (≈2700 chars at
~14 chars/sec). New section S2.5 inserted for Twin Reveal.

| S | Start | End | Length (s) | TTS char budget |
|--|------:|----:|----:|----:|
| 0 Hook | 0 | 10 | 10 | 130 |
| 1 Setup | 10 | 25 | 15 | 200 |
| 2 Twin reveal | 25 | 55 | 30 | 400 |
| 3 Nonce | 55 | 95 | 40 | 540 |
| 4 Live verify | 95 | 115 | 20 | 270 |
| 5 Replay attack | 115 | 140 | 25 | 340 |
| 6 Transplant | 140 | 200 | 60 | 800 |
| 7 Mirror | 200 | 215 | 15 | 200 |
| 8 Why it matters | 215 | 245 | 30 | 400 |
| 9 Caveats | 245 | 260 | 15 | 200 |
| 10 Close | 260 | 275 | 15 | 130 |
| **Total** | | | **275 s** | **~3600 chars ≈ $0.11 TTS** |

## 4. Real-time visual elements

Three live elements that must update in front of the audience (cannot be
pre-rendered):

1. **Nonce-plan precommit URL** — refreshes when volunteer rolls dice. SHA-256
   shown big, with QR code so audience phones can scan.
2. **Verifier panel** — colour-flips during transplant from green to red; this
   is the climax frame, must be live, no canned animation.
3. **Stylometry scatter** — adds 6 dots during the Twin Reveal as each
   chip-bound completion arrives. Audience sees the clusters *form*, not
   just appear.

## 5. Adversarial pre-bunk script (compressed)

| Trigger phrase | Slide / panel to surface | One-line response |
|----------------|--------------------------|-------------------|
| "Pre-recorded" | Nonce-hash QR | "Your dice picked the test 30 s ago. Hash this." |
| "Just serial numbers" | `nonce_signature_v2.py` code panel | "No vendor ID — only manufacturing noise." |
| "Hard-coded personalities" | `style_template_table.png` | "Hash of the chip's own signature picks the template." |
| "DRM" | Caveats slide | "Verifies *who*, not *what*. Open source." |
| "N=2" | Open-call card | "Bring your laptop, we'll enroll it during Q&A." |

## 6. Reproduction artifacts (to publish with the video)

1. `paper_drafts/iconic_proof_design/` — this folder.
2. `scripts/identity_benchmark/embodiment14d_crypto/` — verifier + key
   derivation + nonce signing.
3. `scripts/demo_embodied_ai/demo_server.py` — the live verifier server.
4. `paper_drafts/demo_video_v2/real_data/` — all numbers shown.
5. **NEW:** `scripts/personality_engine/style_from_fingerprint.py` — Phase 21
   deliverable. Hash → template + temperature + top-p.
6. **NEW:** `scripts/precommit/plan_hash.py` — deterministic plan-hash for
   the audience nonce.
7. **NEW:** `paper_drafts/iconic_proof_design/reproduce.sh` — one-shot script
   that fetches the artifacts and runs a verify pass.

## 7. Ethical tags (etiska taggar)

- **DRM concern.** Address head-on in the Caveats slide and in the paper's
  Section 7 ethics paragraph. Claim is *attribution*, not access control. The
  primitive does **not** prevent any software from running.
- **Privacy.** The 290-d fingerprint is uniquely identifying *across reboots*
  on the same chip. We must warn users: do not expose your enrolled signature
  publicly; treat it like a long-lived device identifier. This is in the
  paper's Section 7 already; we add a one-line warning to the demo close.
- **Sybil-resistance double-edge.** Per-die uniqueness is excellent for fed
  learning sybil defence, but it is also a *tracking primitive*. We must
  disclose this in the talk and the paper.
- **Vendor-key independence is double-edged.** A vendor cannot revoke a
  compromised identity for you. Users must rotate enrollments themselves;
  document the rotation protocol.
- **Power asymmetry.** Per-die IDs let large operators fingerprint contributing
  edge devices. We argue the *correct* use is one-direction: client → server,
  not server → unsuspecting-client. Discuss in talk.

## 8. Failure modes during the live show

| Failure | Mitigation |
|---------|-----------|
| Verifier false-reject on honest run | Pre-flight 20 runs morning of; abort live show only if <100% honest in pre-flight. Have a recorded backup of the verify scene to cut to. |
| Projector cable jitter spikes TSC noise | Run on internal display, mirror to projector via downstream HDMI splitter; don't depend on projector EDID. |
| Audience volunteer fumbles dice | Have a backup digital RNG widget (visible on screen) the volunteer can press; mathematically equivalent, less photogenic. |
| One chip thermally throttles mid-show | Pre-show fan curve to performance; venue temp ≤ 24 °C; have a USB fan as a last resort. |
| Personality engine outputs a flat/identical answer | Have a fallback prompt set committed; if the live prompt produces bad differentiation, default to the committed one (still random within the precommitted set). |
