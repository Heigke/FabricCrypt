"""Thin Ollama HTTP client for generating Swedish radio prose.

Defaults to qwen2.5:3b (fast, ~3s for 200 tokens). Falls back gracefully:
if Ollama is unreachable, returns None and caller can fall back to templates.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Optional

import requests

LOG = logging.getLogger("radio.llm")

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:3b"
LONG_MODEL = "qwen2.5:7b"  # for backstory/philosophical


def generate(prompt: str,
             model: str = DEFAULT_MODEL,
             temperature: float = 0.85,
             max_tokens: int = 400,
             timeout: float = 25.0) -> Optional[str]:
    """Return generated text or None on failure."""
    try:
        t0 = time.time()
        r = requests.post(OLLAMA_URL, json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": 0.92,
                "repeat_penalty": 1.15,
            },
        }, timeout=timeout)
        if r.status_code != 200:
            LOG.warning("ollama status=%s body=%s", r.status_code, r.text[:200])
            return None
        text = r.json().get("response", "").strip()
        dt = time.time() - t0
        LOG.info("llm[%s] %.1fs %d chars", model, dt, len(text))
        return text or None
    except Exception as e:
        LOG.warning("ollama call failed: %s", e)
        return None


def healthcheck() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


def sanitize(text: str) -> str:
    """Strip markdown, list bullets, code fences, emoji, surplus whitespace.
    Piper handles Swedish best when given clean prose."""
    import re
    # remove code fences
    text = re.sub(r"```[\s\S]*?```", " ", text)
    # remove markdown headers
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    # remove bullets at line start
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    # remove numbered list markers at line start
    text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.MULTILINE)
    # remove bold/italic markers
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"_{2,}", "", text)
    # strip emojis (rough range)
    text = re.sub(r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF]", "", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("healthy:", healthcheck())
    out = generate("Skriv 50 ord lugn svensk radioröst om en chip-temperatur på 47 grader. Bara prosa.")
    print(sanitize(out or ""))
