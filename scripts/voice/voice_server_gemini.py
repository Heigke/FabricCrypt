"""
Voice server (Gemini Live API): browser <-> FastAPI WebSocket <-> google-genai Live.

Browser sends 16 kHz mono PCM16 mic audio over WS /audio. Server forwards to
Gemini Live; Gemini streams back 24 kHz PCM16 audio + tool calls. Tool calls
are executed locally (filesystem under PROJECT_ROOT) or via sshpass+ssh
(read-only allowlist) on daedalus / zgx, then the FunctionResponse is sent
back to Gemini, which speaks the answer.

Model: gemini-2.5-flash-native-audio-latest (native-audio dialog, the
successor to the user-suggested "gemini-2.5-flash-preview-native-audio-dialog";
the dated "preview-09-2025" / "preview-12-2025" snapshots also work).
The older "gemini-2.0-flash-live-001" alias has been retired from v1beta
(2026-Q1 list shows only 2.5 native-audio + 3.1-flash-live-preview). Override
via env GEMINI_LIVE_MODEL.

Run:  python scripts/voice/voice_server_gemini.py
Open: http://localhost:8765/

SECURITY: see _safe_path() and SSH_CMD_ALLOWLIST below. .env / *.key / .ssh
/ credentials* / *secret* / *token* are blocked locally AND in remote cmds.
Unit tests run when this file is executed with `--selftest`.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import fnmatch
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time
import unittest
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from google import genai
from google.genai import types as gtypes

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy").resolve()
STATIC_DIR = Path(__file__).resolve().parent / "static"
load_dotenv(PROJECT_ROOT / ".env")

GEMINI_API_KEY = (
    os.environ.get("gemini_api_key")
    or os.environ.get("GEMINI_API_KEY")
    or os.environ.get("GOOGLE_API_KEY")
)
GEMINI_LIVE_MODEL = os.environ.get(
    "GEMINI_LIVE_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025"
)
PORT = int(os.environ.get("VOICE_SERVER_GEMINI_PORT", "8765"))

# Remote hosts (from CLAUDE.md). mDNS preferred for daedalus.
REMOTES = {
    "daedalus": {
        "user": os.environ.get("DAEDALUS_USER", "daedalus"),
        "host": os.environ.get("DAEDALUS_HOST", "daedalus.local"),
        "pass": os.environ.get("DAEDALUS_PASS", "daedalus"),
    },
    "zgx": {
        "user": os.environ.get("ZGX_USER", "naorw"),
        "host": os.environ.get("ZGX_HOST", "192.168.0.41"),
        "pass": os.environ.get("ZGX_PASS", "kernel"),
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("voice_gemini")

# ---------------------------------------------------------------------------
# Security: path + command filters
# ---------------------------------------------------------------------------

# Blocklist applied to BOTH local _safe_path and remote ssh_exec strings.
BLOCK_NAME_GLOBS = [
    ".env",
    "*.key",
    "*.pem",
    "credentials*",
    "*secret*",
    "*token*",
    "id_rsa*",
]
BLOCK_SUBSTRINGS = [".ssh/", "/.ssh", ".env"]  # belt + suspenders


def _name_blocked(name: str) -> bool:
    low = name.lower()
    for g in BLOCK_NAME_GLOBS:
        if fnmatch.fnmatch(low, g):
            return True
    return False


def _safe_path(user_path: str, root: Path = PROJECT_ROOT) -> Path:
    """Resolve `user_path` (relative or absolute) and assert it is safe.

    Rules:
      1. Must resolve to an absolute path strictly under `root`.
      2. No path component may match BLOCK_NAME_GLOBS (.env, *.key, ...).
      3. Symlinks are resolved (strict=False so missing paths still raise on
         components) and the resolved path must still be under `root`.
      4. Raw input rejecting ".." parent traversal segments.

    Raises PermissionError on violation. Returns resolved Path on success.
    """
    if not isinstance(user_path, str) or not user_path:
        raise PermissionError("empty path")

    # Reject parent-traversal in raw input. We do this BEFORE resolve so a
    # symlink to /etc still also gets caught by the under-root check.
    raw = user_path.replace("\\", "/")
    parts = [p for p in raw.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise PermissionError(f"parent traversal rejected: {user_path!r}")

    # Build candidate path. Absolute inputs are accepted only if they end up
    # under root after resolve(); relative inputs are joined to root.
    p_in = Path(user_path)
    candidate = p_in if p_in.is_absolute() else (root / p_in)

    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError) as e:
        raise PermissionError(f"resolve failed: {e}") from e

    root_resolved = root.resolve(strict=False)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as e:
        raise PermissionError(
            f"outside project root: {resolved} not under {root_resolved}"
        ) from e

    # Check every component of the FINAL resolved path for blocked names.
    for comp in resolved.parts:
        if _name_blocked(comp):
            raise PermissionError(f"blocked component: {comp!r} in {resolved}")

    # Also reject if any blocked substring is in the resolved string
    rs = str(resolved).lower()
    for sub in BLOCK_SUBSTRINGS:
        if sub in rs:
            raise PermissionError(f"blocked substring {sub!r} in {resolved}")

    return resolved


# Remote ssh command allowlist
SSH_ALLOWED_BIN_RE = re.compile(
    r"^(ls|cat|head|tail|grep|find|pgrep|wc|du|df|nvidia-smi|rocm-smi|free|uptime|date)\b"
)
SSH_FORBIDDEN_CHARS = [";", "&&", "||", "|", ">", "<", "`", "$("]
SSH_FORBIDDEN_TOKENS = [
    "rm", "mv", "cp", "chmod", "chown", "kill", "dd", "mkfs", "sudo",
    "scp", "rsync", "curl", "wget", "nc", "ncat", "ssh", "bash", "sh",
    "python", "perl", "eval", "exec",
]


def _safe_ssh_cmd(cmd: str) -> str:
    if not isinstance(cmd, str) or not cmd.strip():
        raise PermissionError("empty cmd")
    if len(cmd) > 500:
        raise PermissionError("cmd too long")
    for c in SSH_FORBIDDEN_CHARS:
        if c in cmd:
            raise PermissionError(f"forbidden char {c!r} in cmd")
    # Special handling for "du -sh" / "df -h"
    if not SSH_ALLOWED_BIN_RE.match(cmd.strip()):
        raise PermissionError(f"binary not in allowlist: {cmd!r}")
    # Tokenize and check forbidden tokens / blocked filenames
    try:
        toks = shlex.split(cmd)
    except ValueError as e:
        raise PermissionError(f"shlex parse failed: {e}") from e
    for t in toks[1:]:
        low = t.lower()
        if low in SSH_FORBIDDEN_TOKENS:
            raise PermissionError(f"forbidden token {t!r}")
        # check blocked path basename
        base = Path(t).name
        if base and _name_blocked(base):
            raise PermissionError(f"references blocked file: {t!r}")
        for sub in BLOCK_SUBSTRINGS:
            if sub in low:
                raise PermissionError(f"references blocked substring: {t!r}")
    return cmd


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

MAX_READ_BYTES = 20000


def tool_list_dir(path: str = ".") -> dict:
    try:
        p = _safe_path(path)
    except PermissionError as e:
        return {"error": str(e)}
    if not p.exists():
        return {"error": f"not found: {path}"}
    if not p.is_dir():
        return {"error": f"not a directory: {path}"}
    items = []
    for child in sorted(p.iterdir())[:500]:
        if _name_blocked(child.name):
            continue
        items.append({
            "name": child.name,
            "type": "dir" if child.is_dir() else "file",
            "size": child.stat().st_size if child.is_file() else None,
        })
    return {"path": str(p.relative_to(PROJECT_ROOT)), "items": items}


def tool_read_file(path: str, max_bytes: int = MAX_READ_BYTES) -> dict:
    try:
        p = _safe_path(path)
    except PermissionError as e:
        return {"error": str(e)}
    if not p.exists() or not p.is_file():
        return {"error": f"not a file: {path}"}
    max_bytes = min(int(max_bytes or MAX_READ_BYTES), 200_000)
    try:
        data = p.read_bytes()[:max_bytes]
        text = data.decode("utf-8", errors="replace")
    except OSError as e:
        return {"error": f"read failed: {e}"}
    return {
        "path": str(p.relative_to(PROJECT_ROOT)),
        "bytes": len(data),
        "truncated": p.stat().st_size > max_bytes,
        "content": text,
    }


def tool_grep(pattern: str, path_glob: str = "**/*.md", max_results: int = 50) -> dict:
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return {"error": f"bad regex: {e}"}
    if ".." in path_glob:
        return {"error": "no parent traversal in glob"}
    results = []
    count = 0
    for path in PROJECT_ROOT.glob(path_glob):
        if count >= max_results:
            break
        try:
            sp = _safe_path(str(path))
        except PermissionError:
            continue
        if not sp.is_file():
            continue
        try:
            with sp.open("r", encoding="utf-8", errors="replace") as fh:
                for lineno, line in enumerate(fh, 1):
                    if rx.search(line):
                        results.append({
                            "file": str(sp.relative_to(PROJECT_ROOT)),
                            "line": lineno,
                            "text": line.rstrip()[:300],
                        })
                        count += 1
                        if count >= max_results:
                            break
        except OSError:
            continue
    return {"pattern": pattern, "matches": results, "count": len(results)}


def tool_get_log_tail(n: int = 100) -> dict:
    log_path = PROJECT_ROOT / "research_plan" / "01_LOG.md"
    if not log_path.exists():
        return {"error": "no log file"}
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError as e:
        return {"error": str(e)}
    n = max(1, min(int(n or 100), 500))
    return {"tail": "".join(lines[-n:]), "total_lines": len(lines)}


def tool_list_recent_results(n: int = 10) -> dict:
    results_dir = PROJECT_ROOT / "results"
    if not results_dir.exists():
        return {"error": "no results dir"}
    subs = [c for c in results_dir.iterdir() if c.is_dir()]
    subs.sort(key=lambda c: c.stat().st_mtime, reverse=True)
    n = max(1, min(int(n or 10), 100))
    out = []
    for c in subs[:n]:
        out.append({
            "name": c.name,
            "mtime": int(c.stat().st_mtime),
            "files": len(list(c.iterdir())),
        })
    return {"results": out}


def tool_project_status() -> dict:
    briefs_dir = PROJECT_ROOT / "research_plan" / "morning_briefs"
    brief_text = None
    if briefs_dir.exists():
        briefs = sorted(briefs_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        if briefs:
            try:
                brief_text = briefs[0].read_text(encoding="utf-8", errors="replace")[:4000]
            except OSError:
                pass
    log_tail = tool_get_log_tail(n=40).get("tail", "")
    recent = tool_list_recent_results(n=5)
    return {
        "latest_brief": brief_text,
        "log_tail": log_tail,
        "recent_results": recent.get("results", []),
    }


def tool_ssh_exec(host: str, cmd: str) -> dict:
    if host not in REMOTES:
        return {"error": f"unknown host: {host}; allowed: {list(REMOTES)}"}
    try:
        safe_cmd = _safe_ssh_cmd(cmd)
    except PermissionError as e:
        return {"error": f"cmd rejected: {e}"}
    r = REMOTES[host]
    full = [
        "sshpass", "-p", r["pass"],
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-o", "ConnectTimeout=8",
        "-o", "BatchMode=no",
        f"{r['user']}@{r['host']}",
        safe_cmd,
    ]
    try:
        proc = subprocess.run(
            full, capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        return {"error": "ssh timeout"}
    except FileNotFoundError:
        return {"error": "sshpass not installed"}
    out = (proc.stdout or "")[:20000]
    err = (proc.stderr or "")[:2000]
    return {
        "host": host,
        "cmd": safe_cmd,
        "exit": proc.returncode,
        "stdout": out,
        "stderr": err,
    }


def _find_claude_tasks_dir() -> Path | None:
    """Locate the Claude Code session tasks directory for this project.

    Pattern: /tmp/claude-<uid>/<project-path-slug>/<session-id>/tasks/
    Returns the newest tasks dir whose slug contains the project basename.
    """
    base = Path("/tmp")
    candidates: list[tuple[float, Path]] = []
    needle = PROJECT_ROOT.name  # AMD_gfx1151_energy
    try:
        for claude_dir in base.glob("claude-*"):
            if not claude_dir.is_dir():
                continue
            for proj in claude_dir.iterdir():
                if not proj.is_dir() or needle not in proj.name:
                    continue
                for sess in proj.iterdir():
                    tasks = sess / "tasks"
                    if tasks.is_dir():
                        candidates.append((tasks.stat().st_mtime, tasks))
    except OSError:
        return None
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def tool_list_background_agents(n: int = 20) -> dict:
    """List recent Claude Code background sub-agents (newest first).

    Returns agent_id (file basename without .output), mtime, size, age_s.
    These are the *.output JSONL transcripts from Agent() tool calls.
    """
    tasks_dir = _find_claude_tasks_dir()
    if tasks_dir is None:
        return {"error": "no Claude Code tasks dir found under /tmp"}
    files = list(tasks_dir.glob("*.output"))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    n = max(1, min(int(n or 20), 100))
    now = time.time()
    out = []
    for f in files[:n]:
        st = f.stat()
        out.append({
            "agent_id": f.stem,
            "mtime": int(st.st_mtime),
            "size_bytes": st.st_size,
            "age_s": int(now - st.st_mtime),
        })
    return {"tasks_dir": str(tasks_dir), "agents": out}


def tool_read_agent_output_tail(agent_id: str, n_lines: int = 60) -> dict:
    """Read the tail of a Claude Code sub-agent's output JSONL.

    Best-effort parse: returns a *digest* of human-readable fields per record
    (tool_use name+input, text content, summary), discards binary/system noise.
    Hard cap on bytes returned to avoid blowing voice-agent context.
    """
    tasks_dir = _find_claude_tasks_dir()
    if tasks_dir is None:
        return {"error": "no Claude Code tasks dir found"}
    if not agent_id or not re.match(r"^[A-Za-z0-9_-]{4,128}$", agent_id):
        return {"error": "invalid agent_id"}
    f = tasks_dir / f"{agent_id}.output"
    if not f.is_file():
        return {"error": f"unknown agent_id: {agent_id}"}
    n_lines = max(1, min(int(n_lines or 60), 400))
    # tail then parse forward
    try:
        with open(f, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 200_000))
            raw = fh.read().decode("utf-8", errors="replace")
    except OSError as e:
        return {"error": f"read failed: {e}"}
    lines = raw.splitlines()[-n_lines:]
    digest: list[dict] = []
    for ln in lines:
        ln = ln.strip()
        if not ln or not ln.startswith("{"):
            continue
        try:
            rec = json.loads(ln)
        except Exception:
            continue
        d: dict = {"t": rec.get("type") or rec.get("role")}
        msg = rec.get("message") or rec
        # Sub-agent messages have message.content list of blocks
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            for blk in content:
                if not isinstance(blk, dict):
                    continue
                bt = blk.get("type")
                if bt == "text":
                    txt = (blk.get("text") or "").strip()
                    if txt:
                        d.setdefault("text", []).append(txt[:600])
                elif bt == "tool_use":
                    d.setdefault("tool_use", []).append({
                        "name": blk.get("name"),
                        "input_keys": list((blk.get("input") or {}).keys())[:8],
                    })
                elif bt == "tool_result":
                    tc = blk.get("content")
                    if isinstance(tc, list):
                        for sub in tc:
                            if isinstance(sub, dict) and sub.get("type") == "text":
                                d.setdefault("tool_result", []).append(
                                    (sub.get("text") or "")[:400]
                                )
                    elif isinstance(tc, str):
                        d.setdefault("tool_result", []).append(tc[:400])
        elif isinstance(content, str):
            d["text"] = content[:600]
        if rec.get("subtype") == "init":
            d["init"] = {
                "agent": rec.get("agent_name"),
                "model": rec.get("model"),
            }
        if rec.get("type") == "result":
            d["final"] = {
                "summary": (rec.get("summary") or "")[:400],
                "result": (rec.get("result") or "")[:1200],
                "duration_ms": rec.get("duration_ms"),
                "is_error": rec.get("is_error"),
            }
        if any(k for k in d.keys() if k != "t"):
            digest.append(d)
    return {
        "agent_id": agent_id,
        "n_records": len(digest),
        "digest": digest[-n_lines:],
    }


def tool_read_plan(plan: str = "latest") -> dict:
    """Read a campaign plan or weekly synthesis doc by name or keyword."""
    rp = PROJECT_ROOT / "research_plan"
    if not rp.is_dir():
        return {"error": "no research_plan dir"}
    key = (plan or "latest").lower().strip()
    candidates: list[Path] = []
    for f in rp.glob("CAMPAIGN_FULL_PUSH_v*.md"):
        candidates.append(f)
    for f in rp.glob("*PLAN*.md"):
        candidates.append(f)
    for f in (rp / "daily_synth").glob("*.md") if (rp / "daily_synth").is_dir() else []:
        candidates.append(f)
    if not candidates:
        return {"error": "no plan files found"}
    if key in ("latest", "newest", ""):
        chosen = max(candidates, key=lambda p: p.stat().st_mtime)
    else:
        matches = [c for c in candidates if key in c.name.lower()]
        if not matches:
            return {"error": f"no plan matching '{plan}'", "available": [c.name for c in candidates]}
        chosen = max(matches, key=lambda p: p.stat().st_mtime)
    try:
        text = chosen.read_text(encoding="utf-8", errors="replace")[:80_000]
    except OSError as e:
        return {"error": f"read failed: {e}"}
    return {"path": str(chosen.relative_to(PROJECT_ROOT)), "text": text}


TOOL_FUNCS = {
    "list_dir": tool_list_dir,
    "read_file": tool_read_file,
    "grep": tool_grep,
    "get_log_tail": tool_get_log_tail,
    "list_recent_results": tool_list_recent_results,
    "project_status": tool_project_status,
    "ssh_exec": tool_ssh_exec,
    "list_background_agents": tool_list_background_agents,
    "read_agent_output_tail": tool_read_agent_output_tail,
    "read_plan": tool_read_plan,
}


# ---------------------------------------------------------------------------
# Gemini tool schemas (FunctionDeclaration)
# ---------------------------------------------------------------------------

def build_tool_decls() -> list[gtypes.Tool]:
    return [gtypes.Tool(function_declarations=[
        gtypes.FunctionDeclaration(
            name="list_dir",
            description="List directory contents under the AMD_gfx1151_energy project root.",
            parameters=gtypes.Schema(
                type=gtypes.Type.OBJECT,
                properties={"path": gtypes.Schema(type=gtypes.Type.STRING,
                    description="Relative path under project root (e.g. 'research_plan' or '.')")},
                required=["path"],
            ),
        ),
        gtypes.FunctionDeclaration(
            name="read_file",
            description="Read up to max_bytes of a UTF-8 text file under the project root.",
            parameters=gtypes.Schema(
                type=gtypes.Type.OBJECT,
                properties={
                    "path": gtypes.Schema(type=gtypes.Type.STRING),
                    "max_bytes": gtypes.Schema(type=gtypes.Type.INTEGER,
                        description="Max bytes to read (default 20000, hard cap 200000)."),
                },
                required=["path"],
            ),
        ),
        gtypes.FunctionDeclaration(
            name="grep",
            description="Regex-search files matching a glob under the project root.",
            parameters=gtypes.Schema(
                type=gtypes.Type.OBJECT,
                properties={
                    "pattern": gtypes.Schema(type=gtypes.Type.STRING),
                    "path_glob": gtypes.Schema(type=gtypes.Type.STRING,
                        description="Glob pattern, e.g. '**/*.md' or 'research_plan/*.md'"),
                    "max_results": gtypes.Schema(type=gtypes.Type.INTEGER),
                },
                required=["pattern"],
            ),
        ),
        gtypes.FunctionDeclaration(
            name="get_log_tail",
            description="Get the last n lines of research_plan/01_LOG.md.",
            parameters=gtypes.Schema(
                type=gtypes.Type.OBJECT,
                properties={"n": gtypes.Schema(type=gtypes.Type.INTEGER)},
            ),
        ),
        gtypes.FunctionDeclaration(
            name="list_recent_results",
            description="List the n newest subdirectories under results/.",
            parameters=gtypes.Schema(
                type=gtypes.Type.OBJECT,
                properties={"n": gtypes.Schema(type=gtypes.Type.INTEGER)},
            ),
        ),
        gtypes.FunctionDeclaration(
            name="project_status",
            description="Return latest morning brief + log tail + recent results dirs.",
            parameters=gtypes.Schema(type=gtypes.Type.OBJECT, properties={}),
        ),
        gtypes.FunctionDeclaration(
            name="list_background_agents",
            description=(
                "List recent Claude Code background sub-agents (newest first). "
                "Returns agent_id, age_s, size_bytes. Use this first to find "
                "which sub-agents are running before reading their output."
            ),
            parameters=gtypes.Schema(
                type=gtypes.Type.OBJECT,
                properties={"n": gtypes.Schema(type=gtypes.Type.INTEGER,
                    description="Max agents to list (default 20).")},
            ),
        ),
        gtypes.FunctionDeclaration(
            name="read_agent_output_tail",
            description=(
                "Read a DIGEST of a Claude Code sub-agent's transcript tail. "
                "Returns human-readable summary of text blocks, tool_use names, "
                "tool_result content, and final result. Use this to see what a "
                "running or completed background research/build agent is doing."
            ),
            parameters=gtypes.Schema(
                type=gtypes.Type.OBJECT,
                properties={
                    "agent_id": gtypes.Schema(type=gtypes.Type.STRING,
                        description="Agent ID from list_background_agents (no .output suffix)."),
                    "n_lines": gtypes.Schema(type=gtypes.Type.INTEGER,
                        description="Tail size in JSONL records (default 60, max 400)."),
                },
                required=["agent_id"],
            ),
        ),
        gtypes.FunctionDeclaration(
            name="read_plan",
            description=(
                "Read a campaign plan, weekly synth, or daily synth. "
                "Pass 'latest' for newest plan, or a keyword like 'v3', 'v2', "
                "'topology' to match by filename. Use this to know what the "
                "current strategic plan is before answering project-status questions."
            ),
            parameters=gtypes.Schema(
                type=gtypes.Type.OBJECT,
                properties={"plan": gtypes.Schema(type=gtypes.Type.STRING,
                    description="'latest' or keyword to match plan filename.")},
            ),
        ),
        gtypes.FunctionDeclaration(
            name="ssh_exec",
            description=(
                "Run a READ-ONLY shell command on remote host 'daedalus' or 'zgx'. "
                "Only ls/cat/head/tail/grep/find/pgrep/wc/du/df/nvidia-smi/rocm-smi/"
                "free/uptime/date allowed. No pipes, no redirection, no chaining."
            ),
            parameters=gtypes.Schema(
                type=gtypes.Type.OBJECT,
                properties={
                    "host": gtypes.Schema(type=gtypes.Type.STRING,
                        description="'daedalus' or 'zgx'"),
                    "cmd": gtypes.Schema(type=gtypes.Type.STRING),
                },
                required=["host", "cmd"],
            ),
        ),
    ])]


SYSTEM_INSTRUCTION = (
    "You are a voice assistant for the NS-RAM / AMD gfx1151 energy research "
    "project owned by Eric Bergvall (ikaros). You can call tools to read files "
    "in the local project root and to run safe, read-only shell commands on "
    "two remote machines: 'daedalus' (ROCm dev box, mDNS daedalus.local) and "
    "'zgx' (network rig at 192.168.0.41). Be concise and technical. When the "
    "user asks about status, prefer calling project_status first. Cite file "
    "paths when summarizing. If a tool refuses a path or command for security "
    "reasons, tell the user briefly and suggest a safe alternative. Never ask "
    "for or reveal API keys, .env contents, or SSH keys."
)


# ---------------------------------------------------------------------------
# FastAPI app + WebSocket bridge
# ---------------------------------------------------------------------------

app = FastAPI(title="claude_hive voice (Gemini Live)")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({
        "ok": True,
        "model": GEMINI_LIVE_MODEL,
        "api_key_set": bool(GEMINI_API_KEY),
        "remotes": list(REMOTES),
    })


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


async def _run_tool_call(func_call: gtypes.FunctionCall) -> gtypes.FunctionResponse:
    name = func_call.name
    args = dict(func_call.args or {})
    log.info("tool call: %s args=%s", name, args)
    fn = TOOL_FUNCS.get(name)
    if fn is None:
        result: dict[str, Any] = {"error": f"unknown tool {name}"}
    else:
        try:
            # run blocking tools in thread to keep loop responsive
            result = await asyncio.to_thread(fn, **args)
        except TypeError as e:
            result = {"error": f"bad args: {e}"}
        except Exception as e:  # noqa: BLE001
            log.exception("tool %s crashed", name)
            result = {"error": f"tool crashed: {e}"}
    return gtypes.FunctionResponse(
        id=func_call.id,
        name=name,
        response={"result": result},
    )


@app.websocket("/audio")
async def audio_ws(ws: WebSocket) -> None:
    await ws.accept()
    if not GEMINI_API_KEY:
        await ws.send_json({"type": "error", "msg": "GEMINI_API_KEY not set"})
        await ws.close()
        return

    client = genai.Client(api_key=GEMINI_API_KEY)
    config = gtypes.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=gtypes.Content(parts=[gtypes.Part(text=SYSTEM_INSTRUCTION)]),
        tools=build_tool_decls(),
        input_audio_transcription=gtypes.AudioTranscriptionConfig(),
        output_audio_transcription=gtypes.AudioTranscriptionConfig(),
        realtime_input_config=gtypes.RealtimeInputConfig(
            automatic_activity_detection=gtypes.AutomaticActivityDetection(
                disabled=False,
                start_of_speech_sensitivity=gtypes.StartSensitivity.START_SENSITIVITY_LOW,
                end_of_speech_sensitivity=gtypes.EndSensitivity.END_SENSITIVITY_LOW,
                prefix_padding_ms=20,
                silence_duration_ms=100,
            ),
            activity_handling=gtypes.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
        ),
    )

    log.info("connecting to Gemini Live model=%s", GEMINI_LIVE_MODEL)
    try:
        async with client.aio.live.connect(
            model=GEMINI_LIVE_MODEL, config=config
        ) as session:
            await ws.send_json({"type": "ready", "model": GEMINI_LIVE_MODEL})

            audio_bytes_in = 0
            audio_chunks_in = 0
            last_audio_log = time.monotonic()
            async def browser_to_gemini() -> None:
                """Forward browser mic frames + text to Gemini."""
                nonlocal audio_bytes_in, audio_chunks_in, last_audio_log
                try:
                    while True:
                        msg = await ws.receive()
                        if msg["type"] == "websocket.disconnect":
                            return
                        if "bytes" in msg and msg["bytes"] is not None:
                            # 16kHz PCM16 mono from browser
                            pcm = msg["bytes"]
                            audio_bytes_in += len(pcm)
                            audio_chunks_in += 1
                            now = time.monotonic()
                            if now - last_audio_log >= 2.0:
                                log.info("mic in: %d chunks, %.1f KB total, last=%d bytes",
                                         audio_chunks_in, audio_bytes_in/1024.0, len(pcm))
                                last_audio_log = now
                            await session.send_realtime_input(
                                audio=gtypes.Blob(
                                    data=pcm, mime_type="audio/pcm;rate=16000"
                                )
                            )
                        elif "text" in msg and msg["text"] is not None:
                            try:
                                evt = json.loads(msg["text"])
                            except json.JSONDecodeError:
                                continue
                            t = evt.get("type")
                            if t == "text":
                                await session.send_client_content(
                                    turns=[gtypes.Content(
                                        role="user",
                                        parts=[gtypes.Part(text=evt.get("text", ""))],
                                    )],
                                    turn_complete=True,
                                )
                            elif t == "audio_end":
                                await session.send_realtime_input(audio_stream_end=True)
                except WebSocketDisconnect:
                    return
                except Exception as e:  # noqa: BLE001
                    log.exception("browser_to_gemini error: %s", e)

            async def gemini_to_browser() -> None:
                """Forward Gemini audio + transcripts + tool calls to browser."""
                audio_out = 0
                last_resp_log = time.monotonic()
                try:
                    async for response in session.receive():
                        now = time.monotonic()
                        if now - last_resp_log >= 3.0:
                            log.info("gemini→browser: audio_chunks=%d", audio_out)
                            last_resp_log = now
                        sc = getattr(response, "server_content", None)
                        if sc is not None:
                            mt = getattr(sc, "model_turn", None)
                            if mt is not None and mt.parts:
                                for part in mt.parts:
                                    inline = getattr(part, "inline_data", None)
                                    if inline and inline.data:
                                        # 24kHz PCM16 audio chunk
                                        audio_out += 1
                                        await ws.send_json({
                                            "type": "audio",
                                            "mime": inline.mime_type or "audio/pcm;rate=24000",
                                            "data": base64.b64encode(inline.data).decode("ascii"),
                                        })
                                    if getattr(part, "text", None):
                                        await ws.send_json({
                                            "type": "model_text",
                                            "text": part.text,
                                        })
                            in_tr = getattr(sc, "input_transcription", None)
                            if in_tr and getattr(in_tr, "text", None):
                                await ws.send_json({
                                    "type": "user_transcript",
                                    "text": in_tr.text,
                                })
                            out_tr = getattr(sc, "output_transcription", None)
                            if out_tr and getattr(out_tr, "text", None):
                                await ws.send_json({
                                    "type": "model_transcript",
                                    "text": out_tr.text,
                                })
                            if getattr(sc, "turn_complete", False):
                                await ws.send_json({"type": "turn_complete"})
                            if getattr(sc, "interrupted", False):
                                await ws.send_json({"type": "interrupted"})

                        tc = getattr(response, "tool_call", None)
                        if tc is not None and tc.function_calls:
                            responses = []
                            for fc in tc.function_calls:
                                await ws.send_json({
                                    "type": "tool_call",
                                    "name": fc.name,
                                    "args": dict(fc.args or {}),
                                })
                                fr = await _run_tool_call(fc)
                                # echo a compact preview to browser
                                preview = json.dumps(fr.response, default=str)[:500]
                                await ws.send_json({
                                    "type": "tool_result",
                                    "name": fc.name,
                                    "preview": preview,
                                })
                                responses.append(fr)
                            await session.send_tool_response(function_responses=responses)
                except Exception as e:  # noqa: BLE001
                    log.exception("gemini_to_browser error: %s", e)

            await asyncio.gather(
                browser_to_gemini(),
                gemini_to_browser(),
                return_exceptions=True,
            )
    except Exception as e:  # noqa: BLE001
        log.exception("live session failed")
        try:
            await ws.send_json({"type": "error", "msg": f"live connect failed: {e}"})
        except Exception:  # noqa: BLE001
            pass
    finally:
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

class SafePathTests(unittest.TestCase):
    def test_rejects_dotdot(self):
        with self.assertRaises(PermissionError):
            _safe_path("./../.env")

    def test_rejects_etc_passwd(self):
        with self.assertRaises(PermissionError):
            _safe_path("/etc/passwd")

    def test_rejects_project_env(self):
        with self.assertRaises(PermissionError):
            _safe_path(str(PROJECT_ROOT / ".env"))

    def test_rejects_key_file(self):
        with self.assertRaises(PermissionError):
            _safe_path("scripts/voice/private-3.key")

    def test_rejects_ssh_dir(self):
        with self.assertRaises(PermissionError):
            _safe_path("/home/ikaros/.ssh/id_rsa")

    def test_rejects_credentials(self):
        with self.assertRaises(PermissionError):
            _safe_path("credentials.json")

    def test_rejects_secret(self):
        with self.assertRaises(PermissionError):
            _safe_path("my_secret_blob.txt")

    def test_rejects_symlink_to_env(self):
        import tempfile
        tmpdir = PROJECT_ROOT / "scripts" / "voice" / "static"
        link = tmpdir / "_test_link_env"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(PROJECT_ROOT / ".env")
        try:
            with self.assertRaises(PermissionError):
                _safe_path(str(link))
        finally:
            if link.is_symlink() or link.exists():
                link.unlink()

    def test_rejects_symlink_outside_root(self):
        link = PROJECT_ROOT / "scripts" / "voice" / "static" / "_test_link_etc"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to("/etc/hostname")
        try:
            with self.assertRaises(PermissionError):
                _safe_path(str(link))
        finally:
            if link.is_symlink() or link.exists():
                link.unlink()

    def test_accepts_valid(self):
        p = _safe_path("research_plan")
        self.assertTrue(str(p).startswith(str(PROJECT_ROOT)))


class SshCmdTests(unittest.TestCase):
    def test_allows_simple_ls(self):
        self.assertEqual(_safe_ssh_cmd("ls -la /home"), "ls -la /home")

    def test_rejects_pipe(self):
        with self.assertRaises(PermissionError):
            _safe_ssh_cmd("ls | grep x")

    def test_rejects_semicolon(self):
        with self.assertRaises(PermissionError):
            _safe_ssh_cmd("ls; rm -rf /")

    def test_rejects_redirect(self):
        with self.assertRaises(PermissionError):
            _safe_ssh_cmd("cat foo > bar")

    def test_rejects_dollar_paren(self):
        with self.assertRaises(PermissionError):
            _safe_ssh_cmd("ls $(pwd)")

    def test_rejects_unknown_binary(self):
        with self.assertRaises(PermissionError):
            _safe_ssh_cmd("python -c x")

    def test_rejects_env_arg(self):
        with self.assertRaises(PermissionError):
            _safe_ssh_cmd("cat /home/user/.env")

    def test_rejects_key_arg(self):
        with self.assertRaises(PermissionError):
            _safe_ssh_cmd("cat /home/user/id_rsa.key")

    def test_rejects_ssh_dir(self):
        with self.assertRaises(PermissionError):
            _safe_ssh_cmd("ls /home/user/.ssh/")

    def test_allows_du_sh(self):
        self.assertEqual(_safe_ssh_cmd("du -sh /tmp"), "du -sh /tmp")

    def test_allows_rocm_smi(self):
        self.assertEqual(_safe_ssh_cmd("rocm-smi"), "rocm-smi")


def run_selftest() -> int:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite([
        loader.loadTestsFromTestCase(SafePathTests),
        loader.loadTestsFromTestCase(SshCmdTests),
    ])
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    return 0 if res.wasSuccessful() else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="Run unit tests and exit")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(run_selftest())

    if not GEMINI_API_KEY:
        log.warning("gemini_api_key not found in environment. Set it in .env.")

    log.info("starting voice_server_gemini on http://%s:%d/  model=%s",
             args.host, args.port, GEMINI_LIVE_MODEL)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
