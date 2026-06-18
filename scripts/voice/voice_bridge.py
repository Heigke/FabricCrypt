"""
Voice bridge poller.

Watches the autonomous research loop for triggers and, when one fires,
places an outbound call to the user via Vonage that connects to the
voice_server NCCO answer URL.

Triggers:
  - newest results/z*/summary.json contains KILL_SHOT: true or AMBITIOUS: true
  - recent 01_LOG.md entries with "ALERT" or "BLOCKED on user"
  - >30 min since last 01_LOG.md activity (idle alert)

Debounce: at most one call per 10 minutes unless trigger is an emergency.

Usage:
  python voice_bridge.py            # daemon loop, polls every 30s
  python voice_bridge.py --test     # immediately place one test call
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

LOG_PATH = REPO_ROOT / "research_plan" / "01_LOG.md"
RESULTS_DIR = REPO_ROOT / "results"
STATE_PATH = Path("/tmp/voice_bridge_state.json")

VONAGE_APPLICATION_ID = os.environ.get(
    "VONAGE_APPLICATION_ID", "d4f497cc-a01f-40da-8218-be92c960580e"
)
VONAGE_API_KEY = os.environ.get("VONAGE_API_KEY", "43ae6f74")
VONAGE_API_SECRET = os.environ.get("vonage_api_secret") or os.environ.get("VONAGE_API_SECRET", "")
VONAGE_PRIVATE_KEY_PATH = REPO_ROOT / os.environ.get(
    "VONAGE_PRIVATE_KEY_PATH", "scripts/private-2.key"
)
VONAGE_FROM_NUMBER = os.environ.get("VONAGE_FROM_NUMBER", "+46765195862")
USER_PHONE_NUMBER = os.environ.get("USER_PHONE_NUMBER", "+46704990616")
WEBHOOK_BASE_URL = os.environ.get("WEBHOOK_BASE_URL", "")

POLL_INTERVAL_S = 30
DEBOUNCE_S = 10 * 60
IDLE_THRESHOLD_S = 30 * 60

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [voice_bridge] %(message)s",
)
log = logging.getLogger("voice_bridge")


# ---------- state ------------------------------------------------------------

def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_state(s: dict) -> None:
    STATE_PATH.write_text(json.dumps(s, indent=2))


# ---------- trigger detection -----------------------------------------------

def _newest_summary() -> Optional[Path]:
    if not RESULTS_DIR.exists():
        return None
    candidates = sorted(
        RESULTS_DIR.glob("z*/summary.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _detect_trigger() -> Optional[tuple[str, str, bool]]:
    """Return (trigger_kind, message, emergency) or None."""
    # Summary triggers
    s = _newest_summary()
    if s is not None:
        try:
            txt = s.read_text(errors="replace")
            data = json.loads(txt) if txt.strip().startswith("{") else {}
        except Exception:
            data = {}
            txt = ""
        if data.get("KILL_SHOT") is True or re.search(r'"KILL_SHOT"\s*:\s*true', txt):
            return ("KILL_SHOT", f"KILL_SHOT detected in {s.name}", True)
        if data.get("AMBITIOUS") is True or re.search(r'"AMBITIOUS"\s*:\s*true', txt):
            return ("AMBITIOUS", f"AMBITIOUS result in {s.name}", False)

    # Log triggers
    if LOG_PATH.exists():
        try:
            tail = LOG_PATH.read_text(errors="replace").splitlines()[-50:]
        except Exception:
            tail = []
        for line in tail:
            if "BLOCKED on user" in line:
                return ("BLOCKED", f"Loop blocked on user: {line[:120]}", True)
            if "ALERT" in line:
                return ("ALERT", f"Alert in log: {line[:120]}", False)

        # idle?
        try:
            mtime = LOG_PATH.stat().st_mtime
            age = time.time() - mtime
            if age > IDLE_THRESHOLD_S:
                return ("IDLE", f"01_LOG.md idle for {int(age/60)} min", False)
        except Exception:
            pass

    return None


def _trigger_hash(kind: str, message: str) -> str:
    return hashlib.sha1(f"{kind}:{message}".encode()).hexdigest()[:16]


# ---------- vonage call ------------------------------------------------------

def _place_call(reason: str) -> dict:
    if not WEBHOOK_BASE_URL:
        raise RuntimeError(
            "WEBHOOK_BASE_URL not set — start cloudflared tunnel and export it first"
        )

    # ensure private key permissions
    try:
        VONAGE_PRIVATE_KEY_PATH.chmod(0o600)
    except Exception as exc:
        log.warning("Could not chmod 600 %s: %s", VONAGE_PRIVATE_KEY_PATH, exc)

    if not VONAGE_PRIVATE_KEY_PATH.exists():
        raise RuntimeError(f"Vonage private key not found at {VONAGE_PRIVATE_KEY_PATH}")

    from vonage import Vonage, Auth
    from vonage_voice.models import CreateCallRequest, ToPhone, Phone

    private_key_pem = VONAGE_PRIVATE_KEY_PATH.read_text()
    client = Vonage(
        Auth(
            api_key=VONAGE_API_KEY,
            api_secret=VONAGE_API_SECRET,
            application_id=VONAGE_APPLICATION_ID,
            private_key=private_key_pem,
        )
    )

    answer_url = WEBHOOK_BASE_URL.rstrip("/") + "/answer"
    event_url = WEBHOOK_BASE_URL.rstrip("/") + "/event"

    # to=+46... — Vonage requires E.164 without leading +
    to_number = USER_PHONE_NUMBER.lstrip("+")
    from_number = VONAGE_FROM_NUMBER.lstrip("+")

    req = CreateCallRequest(
        to=[ToPhone(number=to_number)],
        from_=Phone(number=from_number),
        answer_url=[answer_url],
        event_url=[event_url],
    )
    log.info("Placing call to +%s (reason=%s, answer=%s)", to_number, reason, answer_url)
    resp = client.voice.create_call(req)
    log.info("Vonage create_call response: %s", resp)
    return {"to": to_number, "answer_url": answer_url, "reason": reason, "resp": str(resp)}


# ---------- main loop --------------------------------------------------------

def _maybe_call(state: dict, kind: str, message: str, emergency: bool) -> bool:
    now = time.time()
    last = state.get("last_call_ts", 0)
    h = _trigger_hash(kind, message)
    if state.get("last_trigger_hash") == h and (now - last) < 6 * 3600:
        log.info("Same trigger hash %s within 6h — skipping", h)
        return False
    if not emergency and (now - last) < DEBOUNCE_S:
        log.info("Debounce: %ds since last call (< %ds)", int(now - last), DEBOUNCE_S)
        return False
    try:
        _place_call(f"{kind}: {message}")
        state["last_call_ts"] = now
        state["last_trigger_hash"] = h
        state["last_reason"] = f"{kind}: {message}"
        _save_state(state)
        return True
    except Exception as exc:
        log.exception("Call failed: %s", exc)
        return False


def main_loop() -> None:
    log.info(
        "voice_bridge starting (poll=%ss, debounce=%ss, idle=%ss)",
        POLL_INTERVAL_S, DEBOUNCE_S, IDLE_THRESHOLD_S,
    )
    log.info("WEBHOOK_BASE_URL=%s", WEBHOOK_BASE_URL or "(unset)")
    state = _load_state()
    # First-run safety: suppress historical triggers so we don't call on boot
    # for old KILL_SHOT/ALERT entries. Mark current trigger as already-seen.
    if not state:
        trig = _detect_trigger()
        if trig is not None:
            kind, msg, _ = trig
            state["last_trigger_hash"] = _trigger_hash(kind, msg)
            state["last_call_ts"] = time.time()
            state["last_reason"] = f"BOOT_SUPPRESS: {kind}: {msg}"
            _save_state(state)
            log.info("Boot suppression: marked existing %s trigger as seen", kind)
    while True:
        try:
            trig = _detect_trigger()
            if trig is not None:
                kind, msg, emerg = trig
                log.info("Trigger: %s (emergency=%s) %s", kind, emerg, msg)
                _maybe_call(state, kind, msg, emerg)
            else:
                log.debug("no trigger")
        except Exception as exc:
            log.exception("poll error: %s", exc)
        time.sleep(POLL_INTERVAL_S)


def test_call() -> int:
    log.info("--test: immediately placing a test call")
    state = _load_state()
    try:
        _place_call("TEST: voice_bridge --test invoked")
        state["last_call_ts"] = time.time()
        state["last_trigger_hash"] = "test"
        state["last_reason"] = "TEST"
        _save_state(state)
        log.info("Test call dispatched. You should receive a call on %s shortly.", USER_PHONE_NUMBER)
        return 0
    except Exception as exc:
        log.exception("Test call failed: %s", exc)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="place one test call and exit")
    args = parser.parse_args()
    if args.test:
        sys.exit(test_call())
    main_loop()
