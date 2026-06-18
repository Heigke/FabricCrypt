# FabricCrypt Demo Video v2 - Storyboard

3 minutes 6 seconds (186s), 960x540 @ 12fps, narrated by OpenAI tts-1-hd (voice: nova).

Format: 1280x720 was the original target; downscaled to 960x540 for thermal budget
on a shared APU. The narration carries the explanation, so resolution is sufficient.

## Section timing

| # | Section          | Start | End  | Audio  | Narrative purpose                          |
|---|------------------|-------|------|--------|--------------------------------------------|
| 0 | Hook             | 0     | 14   | 12.1s  | "We hardware-encrypted an AI"              |
| 1 | Setup            | 14    | 40   | 23.0s  | The five fingerprint signals (reveal one-by-one) |
| 2 | Identity claim   | 40    | 62   | 18.9s  | Each AI reads its own substrate -> both green |
| 3 | Transplant       | 62    | 97   | 31.7s  | Copy model file ikaros->daedalus -> verifier REJECT |
| 4 | Spoof attempts   | 97    | 134  | 34.1s  | 3 attacks, all blocked, summary bar chart  |
| 5 | Why it matters   | 134   | 164  | 26.9s  | Comparison table vs Apple/NVIDIA/Intel/TPM |
| 6 | Caveats          | 164   | 180  | 14.2s  | Honest limits, N=2, what it does NOT do    |
| 7 | Close            | 180   | 186  |  4.2s  | "An AI bound to its body. Just physics."   |

## Per-second breakdown (Section 3 - the key explainer)

Section 3 spans 62-97s. Audio starts at 62s.

| t (s) | Visual                                                       | Audio cue                       |
|-------|--------------------------------------------------------------|---------------------------------|
| 62    | Both chip panels green, native models                        | "Now watch carefully."          |
| 65    | Yellow file icon appears between panels, USB transfer starts | "We copy the model file..."     |
| 65-74 | File icon animates left->right, daedalus shows "loading"     | "Same file. Same weights."      |
| 74    | daedalus chip panel flips to RED state, foreign trace        | "Only the body changes."        |
| 76    | "claims: ikaros [WRONG]" badge appears, confidence 75%       | "The model still claims..."     |
| 78    | "model says ikaros <--MISMATCH--> verifier reads daedalus"   | "But the verifier sees..."      |
| 90    | RED banner: "protocol REJECTS"                               | "The protocol rejects."         |

## Numbers shown (all from real data)

| Metric              | Value      | Source                                              |
|---------------------|------------|-----------------------------------------------------|
| Honest accept rate  | 100.0%     | ikaros_spoof_v2.json `honest_own.accept_rate`       |
| Static replay rate  | 0.6%       | `static_replay_no_nonce.accept_rate` = 0.006        |
| Dynamic replay      | 1.2%       | `dynamic_replay.accept_rate` = 0.012                |
| Peer chassis        | 2.0%       | `daedalus_peer.accept_rate` = 0.02                  |
| Random sig          | 0.6%       | `nonce_only_mismatch.accept_rate` = 0.006           |
| Nonce-space size    | 2^63       | protocol constant                                   |

## Files

- `narration.txt` - full script
- `narration_S{0..7}.txt` - per-section narration
- `audio/audio_S{0..7}.mp3` - TTS per section (OpenAI tts-1-hd, voice nova)
- `audio_full.wav` - silence-padded mixed audio aligned to timeline
- `video_silent.mp4` - 186s silent video (9.4MB)
- `fabriccrypt_v2_with_narration.mp4` - final w/ audio (9.4MB)
- `twitter_60s.mp4` - first-60s cut for Twitter/X (3.6MB, fits under 100MB limit)
- `captions.srt` - sentence-level captions

## TTS cost

2703 chars at $30 / 1M = **$0.0811** total. (Voice: nova, model: tts-1-hd.)
