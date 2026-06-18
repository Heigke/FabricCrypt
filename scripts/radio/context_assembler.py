"""Packs source documents into a compact LLM prompt context (<4K tokens).

For each segment mode (status_deep, backstory, philosophical, ambient, alert)
we pick 1-3 relevant source files and excerpt them.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

REPO = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
RP = REPO / "research_plan"
ORACLE_DIR = RP / "oracle_queries"
IBM_DIR = REPO / "results" / "IDENTITY_BENCHMARK_2026-05-30"
LOG_FILE = RP / "01_LOG.md"
CLAUDE_MD = REPO / "CLAUDE.md"

# Token budgets (rough — assume ~4 chars/token Swedish)
MAX_CHARS_PER_SOURCE = 1500
MAX_TOTAL_CONTEXT = 8000  # ~2K tokens of context, leaves room for prompt+output


def _safe_read(p: Path, max_chars: int = MAX_CHARS_PER_SOURCE) -> str:
    try:
        txt = p.read_text(errors="ignore")
        if len(txt) > max_chars:
            txt = txt[:max_chars] + "... [truncated]"
        return txt
    except Exception:
        return ""


def _list_identity_docs() -> list[Path]:
    return sorted(RP.glob("IDENTITY_*.md"))


def _list_oracle_syntheses(limit: int = 6) -> list[Path]:
    if not ORACLE_DIR.exists():
        return []
    cands = []
    for d in sorted(ORACLE_DIR.glob("O*_*"), reverse=True):
        for name in ("synthesis.md", "_extracted/synthesis.md",
                     "gpt5.md", "grok.md", "gemini.md", "deepseek.md",
                     "grok_response.md", "gemini_response.md", "deepseek_response.md"):
            cand = d / name
            if cand.exists():
                cands.append(cand)
                break
    return cands[:limit]


def _list_ibm_artifacts(limit: int = 6) -> list[Path]:
    if not IBM_DIR.exists():
        return []
    cands: list[Path] = []
    for ext in ("*.md", "*.json"):
        cands.extend(IBM_DIR.rglob(ext))
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[:limit]


def _log_tail_text(n: int = 25) -> str:
    if not LOG_FILE.exists():
        return ""
    lines = LOG_FILE.read_text(errors="ignore").splitlines()
    return "\n".join(lines[-n:])


def _snap_summary(snap: dict[str, Any]) -> str:
    bits: list[str] = []
    bits.append(f"APU: {snap.get('apu_c', -1):.0f}C")
    run = snap.get("running") or {}
    if run:
        bits.append("Pågående: " + ", ".join(f"{k}={v}" for k, v in sorted(run.items())))
    vds = snap.get("verdicts") or []
    if vds:
        v = vds[0]
        bits.append(f"Senaste verdict: {v.get('path')}")
        head = (v.get('head') or "").strip()
        if head:
            bits.append(f"  utdrag: {head[:300]}")
    js = snap.get("jsons") or []
    if js:
        j = js[0]
        bits.append(f"Senaste JSON: {j.get('path')}")
        summ = j.get('summary') or {}
        for k, v in list(summ.items())[:6]:
            bits.append(f"  {k}={v}")
    return "\n".join(bits)


def _history_summary(history: list[dict[str, Any]], n: int = 5) -> str:
    if not history:
        return "(ingen tidigare sändning)"
    out = []
    for h in list(history)[-n:]:
        mode = h.get("mode", "?")
        txt = (h.get("text", "") or "")[:140]
        out.append(f"[{mode}] {txt}")
    return "\n".join(out)


# ---- per-mode context packers -----------------------------------------------

def pack_status_deep(snap: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, str]:
    """Detailed status walkthrough — current snapshot + log tail + top verdict/json."""
    sources: list[str] = []
    sources.append("AKTUELL STATUS:\n" + _snap_summary(snap))
    sources.append("LOGGEN (senaste rader):\n" + _log_tail_text(20)[:1200])
    # If there's a current verdict file, include more of it
    vds = snap.get("verdicts") or []
    if vds:
        vp = REPO / vds[0]["path"]
        if vp.exists():
            sources.append(f"VERDICT FIL {vds[0]['path']}:\n" + _safe_read(vp, 1500))
    js = snap.get("jsons") or []
    if js:
        jp = REPO / js[0]["path"]
        if jp.exists():
            sources.append(f"JSON FIL {js[0]['path']}:\n" + _safe_read(jp, 1000))
    return {
        "topic_hint": "Detaljerad genomgång av vad som händer just nu i labbet.",
        "context": _join_capped(sources),
        "history": _history_summary(history),
    }


def pack_backstory(snap: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, str]:
    """Picks a research_plan IDENTITY_*.md doc and a related oracle synthesis."""
    docs = _list_identity_docs()
    if not docs:
        return pack_status_deep(snap, history)
    chosen = random.choice(docs)
    sources = [f"FORSKNINGSPLAN {chosen.name}:\n" + _safe_read(chosen, 2000)]
    oracles = _list_oracle_syntheses(3)
    if oracles:
        o = random.choice(oracles)
        sources.append(f"ORAKEL-SVAR {o.parent.name}/{o.name}:\n" + _safe_read(o, 1500))
    return {
        "topic_hint": f"Bakgrund och teori — utgå från dokumentet {chosen.stem}.",
        "context": _join_capped(sources),
        "history": _history_summary(history),
    }


def pack_philosophical(snap: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, str]:
    """Big-picture riff — pick from CLAUDE.md + one oracle + one identity doc."""
    sources: list[str] = []
    sources.append("PROJEKT-KONTEXT (CLAUDE.md utdrag):\n" + _safe_read(CLAUDE_MD, 1200))
    oracles = _list_oracle_syntheses(6)
    if oracles:
        o = random.choice(oracles)
        sources.append(f"ORAKEL {o.parent.name}:\n" + _safe_read(o, 1500))
    docs = _list_identity_docs()
    if docs:
        # prefer "deep" / "constitutive" / "null" / "philosophical" docs
        prio = [d for d in docs if any(k in d.name.lower()
                                       for k in ("constitutive", "deep", "null", "novel"))]
        d = random.choice(prio or docs)
        sources.append(f"FORSKNING {d.name}:\n" + _safe_read(d, 1200))
    return {
        "topic_hint": "Filosofisk reflektion — koppla det vi gör till större frågor om identitet, kropp, medvetande, NS-RAM-drömmen.",
        "context": _join_capped(sources),
        "history": _history_summary(history),
    }


def pack_ambient(snap: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, str]:
    return {
        "topic_hint": "Kort, lugn check-in. Bara temperatur och pågående processer. Andningspaus.",
        "context": "STATUS:\n" + _snap_summary(snap),
        "history": _history_summary(history, n=3),
    }


def pack_alert(snap: dict[str, Any], history: list[dict[str, Any]], event: str) -> dict[str, str]:
    return {
        "topic_hint": f"HÖG ENERGI: ny händelse — {event}. Reportertonalitet, koncis men levande.",
        "context": "HÄNDELSE:\n" + event + "\n\nSTATUS:\n" + _snap_summary(snap),
        "history": _history_summary(history, n=3),
    }


def _join_capped(sources: list[str]) -> str:
    joined = "\n\n---\n\n".join(sources)
    if len(joined) > MAX_TOTAL_CONTEXT:
        joined = joined[:MAX_TOTAL_CONTEXT] + "\n... [kontext kapad]"
    return joined


PACKERS = {
    "status_deep": pack_status_deep,
    "backstory": pack_backstory,
    "philosophical": pack_philosophical,
    "ambient": pack_ambient,
    # alert handled separately because it needs event arg
}


if __name__ == "__main__":
    from status_poller import snapshot
    snap = snapshot()
    for mode, fn in PACKERS.items():
        ctx = fn(snap, [])
        print(f"=== {mode} === ({len(ctx['context'])} chars)")
        print(ctx["topic_hint"])
        print(ctx["context"][:400])
        print()
