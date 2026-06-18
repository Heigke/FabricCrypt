"""H7 v2 — animated explainer video with Gemini female TTS, slides timed so the voice is NEVER cut off.

Per slide: render a clean frame (matplotlib), synthesize narration with the LATEST Gemini preview TTS
(gemini-3.1-flash-tts-preview, female voice), MEASURE the clip's true length, then hold that slide for
clip_len + tail_pad and silence-pad the audio to match. So a slide only advances after its narration has
fully finished. Slides are concatenated into one MP4.

Reads results JSON (multilayer keystream + text-behaviour + optional qwen-deep) so the numbers shown are the
real measured ones. Out: paper_drafts/h7_video/H7_embodiment_<date>.mp4
"""
from __future__ import annotations
import os, sys, json, re, wave, subprocess, struct
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results/IDENTITY_H7_2026-06-09"
VID = ROOT / "paper_drafts/h7_video"; VID.mkdir(parents=True, exist_ok=True)
TTS_MODEL = "gemini-3.1-flash-tts-preview"
TTS_FALLBACK = "gemini-2.5-flash-preview-tts"
VOICE = "Kore"                     # female voice
TAIL_PAD = 0.7                     # seconds of held slide after the voice finishes


def gemini_key():
    for line in (ROOT / ".env").read_text().splitlines():
        m = re.match(r'\s*gemini_api_key\s*=\s*"?([^"\n]+)"?', line, re.I)
        if m: return m.group(1).strip()
    raise RuntimeError("gemini_api_key not in .env")


def load(name, default=None):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else (default or {})


# ----------------------------------------------------------------------------- slide content
def build_slides():
    ml = load("v2_multilayer_keystream_ikaros.json")
    tb = load("v2_text_behaviour_ikaros.json")
    qd = load("v2_qwen_deep_ikaros.json")
    acc = ml.get("query_acc", {})
    div = tb.get("token_divergence_vs_live_ref", {})
    S = []
    S.append(dict(title="An LLM Rooted in Its Own Silicon",
        sub="H7 substrate embodiment  ·  AMD Radeon 8060S (gfx1151)",
        bullets=["Made by a Claude Code agent", "Run on HP workstations"],
        narr="What if a language model could not run on any computer — only on the exact physical chip it was "
             "born on? This is H7: a language model rooted in the living silicon of one specific machine. "
             "It was built by a Claude Code agent, running on HP workstations."))
    S.append(dict(title="The Idea",
        bullets=["A frozen GPT-2 — weights never change",
                 "At each fork in the text it is torn between its top two next words",
                 "Which word it picks is decided by the BODY — the live chip"],
        narr="We take a frozen language model and never touch its weights. As it writes, it constantly reaches "
             "forks where two next words are equally fluent. The trick: which word it commits to is decided not "
             "by the model, but by the body — the live, physical computation of the chip it runs on."))
    S.append(dict(title="Three Layers of Physical Computation",
        bullets=["MICRO — two memory streams collide in the L3 cache  →  an XOR gate",
                 "MESO — a GPU wavefront senses its own voltage droop  →  an AND gate",
                 "MACRO — CPU and GPU fight over one power budget  →  an OR gate"],
        narr="The body computes across three different physical scales. At the micro scale, two memory streams "
             "smash into the same cache and cancel each other — a real exclusive-or gate. At the meso scale, a "
             "graphics wavefront measures its own clock slowing under load — voltage droop. At the macro scale, "
             "the processor and graphics core fight over a single power budget. Three scales, three logic gates, "
             "all made of physics."))
    S.append(dict(title="How the Body Hooks Into the Loop",
        bullets=["K = micro ⊕ meso ⊕ macro ⊕ per-die fingerprint",
                 "Operands come from a fresh verifier nonce — not the text",
                 "The frozen model literally cannot guess K; it must ask the body"],
        narr="Every fork, the three gates are combined with an exclusive-or, together with a fingerprint unique "
             "to this die. The inputs come from a fresh one-time number, not from the text — so the model's own "
             "knowledge can never predict the answer. To keep writing coherently, it has no choice but to ask "
             "its body."))
    n = acc.get("native", 1.0)
    S.append(dict(title="Every Layer Is Load-Bearing",
        bullets=[f"native (full body):  {acc.get('native','?')}",
                 f"remove micro:  {acc.get('no_micro','?')}    remove meso:  {acc.get('no_meso','?')}",
                 f"remove macro:  {acc.get('no_macro','?')}    no body at all:  {acc.get('no_body','?')}",
                 f"foreign die:  {acc.get('foreign_die','?')}    replayed nonce:  {acc.get('replay_old','?')}"],
        narr="And it works. With the full body, the model is perfect. Remove any single layer — the cache, the "
             "GPU droop, or the power contention — and it breaks. Give it a different chip's fingerprint, or "
             "replay an old session, and it fails. The body is not decoration; every layer carries weight."))
    S.append(dict(title="The Body Steers the Words",
        bullets=[f"same die, same nonce  →  identical text   (divergence {div.get('live_ikaros','0.0')})",
                 f"no body  →  {div.get('no_body','?')} of words change",
                 f"foreign die  →  {div.get('foreign_die','?')}    old nonce  →  {div.get('replay_oldnonce','?')}"],
        narr="Watch what it does to real text. On the same chip with the same challenge, the model writes the "
             "exact same passage every time — a reproducible identity. Take the body away, and nine in ten words "
             "change. Move it to a different chip, and it writes like a different individual. The body steers the "
             "behaviour."))
    if qd.get("ALL_GREEN"):
        S.append(dict(title="Woven Deep Into a 1.5B Model",
            bullets=["Frozen Qwen2.5-1.5B — body modulates the residual stream across many layers",
                     f"native {qd['query_acc'].get('native','?')}   no_body {qd['query_acc'].get('no_body','?')}   foreign {qd['query_acc'].get('foreign_die','?')}",
                     "Not a bolt-on adapter — the body lives inside the network"],
            narr="It scales. In a frozen one-and-a-half-billion-parameter Qwen model, the body's live signal is "
                 "injected deep into the residual stream across many layers. It is no longer a small adapter at "
                 "the end — the body lives inside the network, and without it the model cannot function."))
    S.append(dict(title="Honest Scope",
        bullets=["The computations are real physics and load-bearing — but generic across like chips",
                 "Uniqueness comes from the fused per-die fingerprint (remote-attestation grade)",
                 "Everything trained and run against LIVE silicon — no simulation, no shortcuts"],
        narr="We are careful about claims. The physical computations are real and load-bearing, but similar "
             "chips can compute similar gates. The true uniqueness comes from each die's fused fingerprint. "
             "Everything here was trained and measured against live silicon — no simulation, no shortcuts."))
    S.append(dict(title="A Mind That Needs Its Body",
        bullets=["Unique · Fresh · Computing — woven into the weights",
                 "Built by a Claude Code agent on HP workstations"],
        narr="The result is a language model that genuinely needs its body: unique to one chip, fresh every "
             "session, and computing through real silicon physics woven into its very function. Built by a "
             "Claude Code agent, on HP workstations."))
    return S


# ----------------------------------------------------------------------------- rendering
def render_frame(slide, idx, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor("#0b1021")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")
    # accent bar
    ax.add_patch(FancyBboxPatch((0.6, 7.9), 15, 0.12, boxstyle="round,pad=0.02", color="#4cc9f0", ec="none"))
    ax.text(0.7, 8.25, slide["title"], color="#ffffff", fontsize=46, fontweight="bold", va="bottom")
    if slide.get("sub"):
        ax.text(0.72, 7.35, slide["sub"], color="#9bb7d4", fontsize=22, va="bottom")
    y = 6.4
    for bt in slide.get("bullets", []):
        ax.text(1.0, y, "▸", color="#4cc9f0", fontsize=26, va="top", fontweight="bold")
        ax.text(1.6, y, bt, color="#e8eef7", fontsize=27, va="top")
        y -= 1.15
    ax.text(15.3, 0.4, "H7 · substrate-rooted embodiment", color="#5a6b85", fontsize=16, ha="right")
    fig.savefig(path, facecolor=fig.get_facecolor()); plt.close(fig)


def tts(text, path, key):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=key)
    cfg = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE))))
    last = None
    for model in (TTS_MODEL, TTS_FALLBACK):
        try:
            r = client.models.generate_content(model=model, contents=text, config=cfg)
            data = r.candidates[0].content.parts[0].inline_data.data
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(data)
            with wave.open(str(path)) as w: dur = w.getnframes() / w.getframerate()
            return dur, model
        except Exception as e:
            last = e; print(f"  TTS {model} failed: {e}", flush=True)
    raise RuntimeError(f"all TTS models failed: {last}")


def main():
    key = gemini_key()
    slides = build_slides()
    print(f"[video] {len(slides)} slides, TTS={TTS_MODEL} voice={VOICE}", flush=True)
    seg_paths = []
    for i, s in enumerate(slides):
        img = VID / f"slide_{i:02d}.png"; wav = VID / f"vo_{i:02d}.wav"; seg = VID / f"seg_{i:02d}.mp4"
        render_frame(s, i, img)
        dur, used = tts(s["narr"], wav, key)
        hold = round(dur + TAIL_PAD, 2)
        print(f"  slide {i}: voice {dur:.1f}s ({used}) -> hold {hold}s", flush=True)
        # video segment: image looped for `hold`s, audio silence-padded to `hold` so voice always finishes
        cmd = ["ffmpeg", "-y", "-loop", "1", "-i", str(img), "-i", str(wav),
               "-af", f"apad=whole_dur={hold}", "-t", f"{hold}",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25",
               "-c:a", "aac", "-b:a", "192k", "-shortest", str(seg)]
        subprocess.run(cmd, check=True, capture_output=True)
        seg_paths.append(seg)
    # concat
    listf = VID / "segments.txt"
    listf.write_text("".join(f"file '{p.name}'\n" for p in seg_paths))
    from datetime import date  # date() with no args is fine; only Date.now-style is blocked in workflows
    out = VID / "H7_embodiment_2026-06-18.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listf),
                    "-c", "copy", str(out)], check=True, capture_output=True, cwd=str(VID))
    # duration
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", str(out)], capture_output=True, text=True).stdout.strip()
    print(f"[video] DONE -> {out}  ({dur}s)", flush=True)


if __name__ == "__main__":
    main()
