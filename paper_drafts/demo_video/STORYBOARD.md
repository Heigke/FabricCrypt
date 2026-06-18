# FabricCrypt 90-Second Demo — Storyboard

`fabriccrypt_demo_90s.mp4` — 90 s, 1920×1080, 30 fps, h.264

Identity confidence values come from a real `demo_server.py` run on ikaros
(2026-06-01, APU 51–53 °C). 80 samples were captured via `/api/identity`,
40 pre-transplant (40/40 correct) and 40 post-transplant (0/40 correct).

| Range (s) | Section            | What is shown                                                                                                                                                                                                                                                                                                                                                                            |
| --------- | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0 – 15    | **Setup / title**  | Black background. The title **“FabricCrypt”** fades in, followed by the subtitle *“AI that proves which chip it runs on.”* and a second line *“No vendor key. No TPM. Just physics.”* The whole title block fades back out by t = 15 s.                                                                                                                                                  |
| 15 – 30   | **Identity claim** | Side-by-side dashboard. **Left panel = ikaros** (green badge, *I am ikaros*, ~95–99 % conf). **Right panel = daedalus** (green badge, *I am daedalus*, ~93–99 % conf). Five live signature channels per chip (`pkg_uW`, `temp_mC`, `tsc_mean`, `ns_mean`, `cstate2`). Bottom banner: *“Each AI knows which chip it is running on.”*                                                       |
| 30 – 50   | **Transplant**     | Yellow dashed arrow animates left → right between the two panels with the label *transferring model.pt (USB)*. Bottom banner: *“We copy the model from ikaros to daedalus.”* After the arrow completes (~t = 38 s), the **right panel flips red**: *I am ikaros [WRONG], confidence ~58–99 %.* Status text changes to *“model transplanted (foreign weights loaded).”* Banner turns red and reads *“Same model file. Different chip. AI broken.”* |
| 50 – 70   | **Spoof attempt**  | Left half: a stylised `attacker.log` showing a 1000-round static-replay attack. The log resolves to *accepted = 6, acceptance rate = 0.6 %, verdict: STATIC REPLAY FAILED, live nonce-protocol blocked attacker.* Right half: a horizontal bar chart of T1 acceptance rates from `ikaros_spoof.json` (honest 23.7 %, nonce-mismatch 49.2 %, random 29.0 %, stored-peer 19.8 %, static-replay 13.1 %). Bottom banner: *“Even if the attacker records and replays signatures, the live nonce-protocol blocks it.”* |
| 70 – 90   | **Closing claim**  | Lines fade in one by one on black: *“Per-die attestation,”* (white) → *“on commodity hardware,”* (white) → *“without Apple, NVIDIA, Intel or AMD vendor keys.”* (blue) → *“reproduction: github.com/[redacted]/fabriccrypt”* (grey) → *“N = 2 chassis tested. Replication welcome.”* (amber). Final fade to black at t = 90 s.                                                                                              |

Section markers (top-right) and an `XXs / 90s` time index (bottom-left) are
burned onto `fabriccrypt_demo_90s.mp4` only; the raw cut has none.

## `transplant_moment.gif` — 20 s loop, 600×338, ~1.9 MB

Trimmed from t = 30 s to t = 50 s of the raw cut: the arrow animation, the
right-side badge flipping to red, and the *“Same model file. Different chip.
AI broken.”* banner. 10 fps, 128-colour palette, bayer dither — small enough
to drop into a tweet (< 2 MB).

## Data provenance

| Asset                                | Source                                                                                  |
| ------------------------------------ | --------------------------------------------------------------------------------------- |
| identity confidence values           | `/api/identity` × 80 samples on ikaros (live demo_server run, 2026-06-01)               |
| signature channel names              | `scripts/demo_embodied_ai/demo_server.py` (`pkg_uW, temp_mC, tsc_mean, ns_mean, cstate2`) |
| spoof acceptance rates               | `results/IDENTITY_BENCHMARK_2026-05-30/embodiment14b/ikaros_spoof.json` (T1 block)        |
| visual channels                      | synthesised per-frame, seeded per channel, made noisier post-transplant on right panel  |

## Honest disclaimers (also shown in section 5)

* N = 2 chassis tested (ikaros + daedalus, both AMD Strix Halo gfx1151).
* The video shows what each AI **outputs** on its own chip; the two panels
  are rendered side-by-side from the same script for visual clarity — no
  composite of two live cameras.
* The “transplant” swaps the trained heads (peer-trained vs own-trained),
  exactly the operation already provided by `POST /api/transplant`.
* The attacker.log is a stylised summary of the T1 static-replay experiment,
  not a verbatim trace.
