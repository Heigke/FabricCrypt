#!/usr/bin/env python3
"""
Minimal Telegram bridge POC for Claude<->Eric remote messaging.

Features
--------
1. Long-poll bot mode (default): handles /ask, /voice, /status, plain text and voice notes.
2. Push mode (CLI): `python telegram_bridge.py push "message"` — Claude pushes a question.
3. LLM routing: OpenAI (default), Gemini, Grok, DeepSeek — pick with --backend or env.
4. Voice in: download Telegram .oga, transcribe via OpenAI Whisper.
5. Voice out: OpenAI TTS (tts-1, mp3) sent back as audio message.

Setup (see ../README.md for full guide)
---------------------------------------
1. @BotFather -> /newbot -> grab token.
2. Add to .env:
     TELEGRAM_BOT_TOKEN=<token>
     TELEGRAM_CHAT_ID=<your numeric chat id>   # for push mode
3. pip install requests python-dotenv
4. Run: python telegram_bridge.py serve
5. Push: python telegram_bridge.py push "Claude needs your input on X"

NOTE: This is a POC. Token NOT included. Do not deploy until user approves.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests

# Optional dotenv load
try:
    from dotenv import load_dotenv
    REPO_ROOT = Path(__file__).resolve().parents[3]
    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"
TELEGRAM_FILE = "https://api.telegram.org/file/bot{token}/{path}"

OPENAI_CHAT = "https://api.openai.com/v1/chat/completions"
OPENAI_STT = "https://api.openai.com/v1/audio/transcriptions"
OPENAI_TTS = "https://api.openai.com/v1/audio/speech"

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent?key={key}"
)
GROK_URL = "https://api.x.ai/v1/chat/completions"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"


# ---------- Telegram helpers ----------

def tg(method: str, **payload) -> dict:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    r = requests.post(TELEGRAM_API.format(token=token, method=method),
                      json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def tg_get_file(file_id: str) -> bytes:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    meta = tg("getFile", file_id=file_id)["result"]
    url = TELEGRAM_FILE.format(token=token, path=meta["file_path"])
    return requests.get(url, timeout=60).content


def tg_send_text(chat_id: int, text: str) -> None:
    # Telegram limit 4096 chars; chunk if needed
    for i in range(0, len(text), 4000):
        tg("sendMessage", chat_id=chat_id, text=text[i:i + 4000])


def tg_send_voice(chat_id: int, mp3_bytes: bytes) -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    files = {"voice": ("reply.ogg", mp3_bytes, "audio/ogg")}
    data = {"chat_id": str(chat_id)}
    r = requests.post(TELEGRAM_API.format(token=token, method="sendVoice"),
                      data=data, files=files, timeout=60)
    r.raise_for_status()


# ---------- LLM backends ----------

def llm_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    headers = {"Authorization": f"Bearer {os.environ['openai_api_key']}"}
    body = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    r = requests.post(OPENAI_CHAT, headers=headers, json=body, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def llm_gemini(prompt: str) -> str:
    key = os.environ["gemini_api_key"]
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    r = requests.post(GEMINI_URL.format(key=key), json=body, timeout=60)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def llm_grok(prompt: str) -> str:
    headers = {"Authorization": f"Bearer {os.environ['grok_api_key']}"}
    body = {"model": "grok-2-latest",
            "messages": [{"role": "user", "content": prompt}]}
    r = requests.post(GROK_URL, headers=headers, json=body, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def llm_deepseek(prompt: str) -> str:
    headers = {"Authorization": f"Bearer {os.environ['deepseek_api_key']}"}
    body = {"model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}]}
    r = requests.post(DEEPSEEK_URL, headers=headers, json=body, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def llm(prompt: str, backend: str) -> str:
    return {
        "openai": llm_openai,
        "gemini": llm_gemini,
        "grok": llm_grok,
        "deepseek": llm_deepseek,
    }[backend](prompt)


# ---------- Voice ----------

def stt_whisper(audio_bytes: bytes, fmt: str = "oga") -> str:
    headers = {"Authorization": f"Bearer {os.environ['openai_api_key']}"}
    files = {"file": (f"in.{fmt}", audio_bytes, "audio/ogg")}
    data = {"model": "whisper-1"}
    r = requests.post(OPENAI_STT, headers=headers, data=data, files=files,
                      timeout=60)
    r.raise_for_status()
    return r.json()["text"]


def tts_openai(text: str, voice: str = "alloy") -> bytes:
    headers = {"Authorization": f"Bearer {os.environ['openai_api_key']}",
               "Content-Type": "application/json"}
    body = {"model": "tts-1", "voice": voice, "input": text[:4000],
            "format": "opus"}
    r = requests.post(OPENAI_TTS, headers=headers, json=body, timeout=60)
    r.raise_for_status()
    return r.content


# ---------- Command handling ----------

def handle_message(msg: dict, backend: str) -> None:
    chat_id = msg["chat"]["id"]
    text = msg.get("text", "")
    voice_reply = False

    # Voice note in
    if "voice" in msg:
        try:
            audio = tg_get_file(msg["voice"]["file_id"])
            text = stt_whisper(audio)
            tg_send_text(chat_id, f"[heard]: {text}")
            voice_reply = True  # respond in voice if user spoke
        except Exception as e:
            tg_send_text(chat_id, f"STT error: {e}")
            return

    # Commands
    if text.startswith("/start"):
        tg_send_text(chat_id,
                     "Bridge online. Commands: /ask <q>, /voice <q>, "
                     "/status, or just send text/voice notes.")
        return
    if text.startswith("/status"):
        tg_send_text(chat_id, f"OK. Backend={backend}. chat_id={chat_id}")
        return
    if text.startswith("/voice"):
        voice_reply = True
        text = text[len("/voice"):].strip()
    elif text.startswith("/ask"):
        text = text[len("/ask"):].strip()

    if not text:
        return

    try:
        answer = llm(text, backend)
    except Exception as e:
        tg_send_text(chat_id, f"LLM error: {e}")
        return

    tg_send_text(chat_id, answer)
    if voice_reply:
        try:
            tg_send_voice(chat_id, tts_openai(answer))
        except Exception as e:
            tg_send_text(chat_id, f"TTS error: {e}")


def serve(backend: str) -> None:
    offset = 0
    print(f"[bridge] long-poll serve, backend={backend}", flush=True)
    while True:
        try:
            res = tg("getUpdates", offset=offset, timeout=25)
            for upd in res.get("result", []):
                offset = upd["update_id"] + 1
                if "message" in upd:
                    handle_message(upd["message"], backend)
        except KeyboardInterrupt:
            return
        except Exception as e:
            print(f"[bridge] poll error: {e}", flush=True)
            time.sleep(5)


def push(message: str, voice: bool = False) -> None:
    chat_id = int(os.environ["TELEGRAM_CHAT_ID"])
    tg_send_text(chat_id, message)
    if voice:
        tg_send_voice(chat_id, tts_openai(message))


# ---------- Entry ----------

def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--backend", default=os.environ.get("BRIDGE_BACKEND",
                                                       "openai"),
                   choices=["openai", "gemini", "grok", "deepseek"])
    ps = sub.add_parser("push")
    ps.add_argument("message")
    ps.add_argument("--voice", action="store_true")
    args = p.parse_args()

    if "TELEGRAM_BOT_TOKEN" not in os.environ:
        print("ERROR: TELEGRAM_BOT_TOKEN not set in env / .env", file=sys.stderr)
        return 2

    if args.cmd == "serve":
        serve(args.backend)
    elif args.cmd == "push":
        if "TELEGRAM_CHAT_ID" not in os.environ:
            print("ERROR: TELEGRAM_CHAT_ID not set (talk to bot once, then "
                  "fetch from /status reply or getUpdates)", file=sys.stderr)
            return 2
        push(args.message, voice=args.voice)
    return 0


if __name__ == "__main__":
    sys.exit(main())
