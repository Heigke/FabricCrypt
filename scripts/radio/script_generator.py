"""Generate broadcaster prose for FEEL Lab Radio.

Five segment modes form a podcast-style rotation:
  status_deep   — 400 words, detailed walkthrough of current experiments
  backstory     — 250 words, science/theory behind a recent finding
  philosophical — 200 words, big-picture riff on identity / NS-RAM / FEEL
  ambient       — 60 words, brief breathing-room check-in
  alert         — 150 words, higher-energy event-driven

Primary path: Ollama (qwen2.5:3b for fast modes, qwen2.5:7b for longer reflection).
Fallback: template prose if Ollama unreachable.
"""
from __future__ import annotations

import logging
import random
import re
import time
from typing import Any, Optional

from llm_engine import generate as llm_generate, sanitize, healthcheck
from context_assembler import (
    pack_status_deep, pack_backstory, pack_philosophical,
    pack_ambient, pack_alert,
)

LOG = logging.getLogger("radio.script")

# Rotation: A → D → B → D → C → D → A → D ...
ROTATION = ["status_deep", "ambient", "backstory", "ambient",
            "philosophical", "ambient", "status_deep", "ambient"]

MODE_TARGETS = {
    # (target_words, ollama_model, temperature, max_tokens, voice)
    "status_deep":   (400, "qwen2.5:3b", 0.75, 700, "host"),
    "backstory":     (250, "qwen2.5:7b", 0.80, 500, "guest"),
    "philosophical": (200, "qwen2.5:7b", 0.92, 400, "guest"),
    "ambient":       (60,  "qwen2.5:3b", 0.85, 140, "host"),
    "alert":         (150, "qwen2.5:3b", 0.85, 300, "host"),
}

STYLE_BASE = (
    "Du är värd för FEEL Lab Radio — en lugn, eftertänksam svensk forskningspodd "
    "från ett enmans-labb i ett vardagsrum. Tonen är mormor-civilingenjör: "
    "exakt utan jargong, vardaglig utan att vara dum. Första person plural ('vi'). "
    "Inga rubriker, inga punktlistor, ingen markdown, inga emojis. "
    "Bara flytande prosa. Skriv på svenska. "
    "Använd konkreta tal och filnamn när det är relevant. "
    "Undvik klyschor som 'spännande resa' eller 'fascinerande värld'."
)


def _prompt_for(mode: str, ctx: dict[str, str]) -> str:
    target_words, _, _, _, _ = MODE_TARGETS[mode]
    return (
        f"{STYLE_BASE}\n\n"
        f"SEGMENT-TYP: {mode}. RIKTLINJE: {ctx['topic_hint']}\n"
        f"LÄNGD: ungefär {target_words} ord. Längre om temat kräver det, "
        f"men gå inte under {int(target_words*0.6)} ord.\n\n"
        f"INTE UPPREPA tidigare sändningar:\n{ctx.get('history','')}\n\n"
        f"KÄLLMATERIAL (utdrag, dina egna notes — referera, hitta inte på):\n"
        f"{ctx['context']}\n\n"
        f"Skriv nu segmentet. Bara prosa, ingen meta-kommentar, ingen rubrik."
    )


def _fallback_template(mode: str, snap: dict[str, Any]) -> str:
    """Plain template prose when Ollama is down. Reuses old phrasing pool."""
    apu = snap.get("apu_c", -1)
    t = f"{apu:.0f}" if apu >= 0 else "okänd"
    running = snap.get("running") or {}
    run_txt = ", ".join(f"{k}={v}" for k, v in sorted(running.items())) or "ingenting"
    if mode == "ambient":
        return random.choice([
            f"Stilla i etern. APU på {t} grader. Vi är kvar.",
            f"Bakgrundsljud bara. {t} grader. Inget nytt.",
            f"Lugnt här. {t} på chippet. Vi väntar in nästa signal.",
        ])
    parts = [
        f"Klockan är {time.strftime('%H:%M')}.",
        f"APU sitter på {t} grader.",
        f"Pågående processer: {run_txt}.",
    ]
    vds = snap.get("verdicts") or []
    if vds:
        parts.append(f"Senaste verdict-filen är {vds[0].get('path')}.")
    js = snap.get("jsons") or []
    if js:
        parts.append(f"Senaste resultatet ligger i {js[0].get('path')}.")
    return " ".join(parts)


def _post_process(text: str, mode: str) -> str:
    text = sanitize(text)
    # Cut at sentence boundary if over 2x target
    target_words = MODE_TARGETS[mode][0]
    words = text.split()
    if len(words) > target_words * 2.5:
        cut = " ".join(words[: int(target_words * 1.6)])
        # round to sentence end
        m = re.search(r"^(.*[.!?])\s", cut + " ")
        if m:
            cut = m.group(1)
        text = cut
    return text


def pick_mode(snap: dict[str, Any],
              rotation_idx: int,
              event: Optional[str] = None,
              changed: bool = False) -> str:
    """Decide next segment mode."""
    if event:
        return "alert"
    if changed and random.random() < 0.45:
        # A change in the world: bias toward a deep status update
        return "status_deep"
    return ROTATION[rotation_idx % len(ROTATION)]


def generate(mode: str,
             snap: dict[str, Any],
             history: list[dict[str, Any]],
             event: Optional[str] = None) -> tuple[str, str, str]:
    """Return (text, mode_used, voice_key).

    voice_key is 'host' or 'guest' — TTS picks the right onnx.
    """
    if mode == "alert":
        ctx = pack_alert(snap, history, event or "ny händelse")
    elif mode == "status_deep":
        ctx = pack_status_deep(snap, history)
    elif mode == "backstory":
        ctx = pack_backstory(snap, history)
    elif mode == "philosophical":
        ctx = pack_philosophical(snap, history)
    else:
        mode = "ambient"
        ctx = pack_ambient(snap, history)

    _, model, temp, max_tok, voice = MODE_TARGETS[mode]

    text: Optional[str] = None
    if healthcheck():
        prompt = _prompt_for(mode, ctx)
        text = llm_generate(prompt, model=model, temperature=temp, max_tokens=max_tok,
                            timeout=40.0 if "7b" in model else 25.0)
    if not text:
        LOG.warning("ollama unavailable or empty -> template fallback (mode=%s)", mode)
        text = _fallback_template(mode if mode != "alert" else "status_deep", snap)
        voice = "host"

    text = _post_process(text, mode)
    return text, mode, voice


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    sys.path.insert(0, ".")
    from status_poller import snapshot
    snap = snapshot()
    for m in ["status_deep", "backstory", "philosophical", "ambient"]:
        print(f"\n==== {m} ====")
        text, mode, voice = generate(m, snap, [])
        print(f"[voice={voice}] ({len(text.split())} words)")
        print(text)
