"""One-time LinkedIn 3-legged OAuth — turns the app client_id/secret in .env into a posting
access token. LinkedIn will NOT let an app post on a member's behalf with client credentials
alone; a member must log in once and consent to the w_member_social scope.

No background server needed (two quick foreground steps — paste-back flow):

  STEP 1:  python scripts/social/linkedin_auth.py url
           Prints an authorization URL + saves the CSRF state. Open the URL in a browser,
           log in, approve. The browser then redirects to
               http://localhost:8771/callback?code=...&state=...
           Nothing is listening there, so the page will say "can't connect" — that's FINE.
           Copy the FULL URL from the address bar.

  STEP 2:  python scripts/social/linkedin_auth.py exchange "<that full localhost URL>"
           Extracts the code, verifies state, exchanges it for an access token, fetches the
           member URN, and writes linked_in_access_token + linked_in_member_urn into .env.

Prereq (one-time, LinkedIn developer console for this app -> https://www.linkedin.com/developers/apps):
  * Products: enable "Share on LinkedIn" and "Sign In with LinkedIn using OpenID Connect".
  * Auth tab -> Authorized redirect URLs -> add exactly:  http://localhost:8771/callback
Token lasts ~60 days; rerun when posting fails with 401.
"""
from __future__ import annotations
import json, secrets, sys, urllib.parse, urllib.request
from pathlib import Path

ENV = Path(__file__).resolve().parents[2] / ".env"
PORT = 8771
REDIRECT = f"http://localhost:{PORT}/callback"
SCOPES = "openid profile w_member_social"
STATE_FILE = Path("/tmp/li_state.txt")


def load_env():
    d = {}
    for ln in ENV.read_text().splitlines():
        ln = ln.strip()
        if "=" in ln and not ln.startswith("#"):
            k, v = ln.split("=", 1); d[k.strip()] = v.strip().strip('"').strip("'")
    return d


def set_env(updates: dict):
    lines = ENV.read_text().splitlines()
    seen = set()
    for i, ln in enumerate(lines):
        s = ln.strip()
        if "=" in s and not s.startswith("#"):
            k = s.split("=", 1)[0].strip()
            if k in updates:
                lines[i] = f"{k}={updates[k]}"; seen.add(k)
    for k, v in updates.items():
        if k not in seen:
            lines.append(f"{k}={v}")
    ENV.write_text("\n".join(lines) + "\n")


def cmd_url():
    env = load_env()
    cid = env.get("linked_in_client_id")
    if not cid:
        print("missing linked_in_client_id in .env"); return
    state = secrets.token_urlsafe(16)
    STATE_FILE.write_text(state)
    url = "https://www.linkedin.com/oauth/v2/authorization?" + urllib.parse.urlencode({
        "response_type": "code", "client_id": cid, "redirect_uri": REDIRECT,
        "state": state, "scope": SCOPES})
    print("\nOpen this in a browser, log in, approve, then copy the localhost URL you land on:\n")
    print(url + "\n")
    print(f"(redirect URL registered in the app must be exactly: {REDIRECT})")


def cmd_exchange(landed: str):
    env = load_env()
    cid, sec = env.get("linked_in_client_id"), env.get("linked_in_secret")
    # accept either the full localhost URL or just the code
    if landed.startswith("http"):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(landed).query)
        code = q.get("code", [None])[0]; state = q.get("state", [None])[0]
    else:
        code, state = landed.strip(), None
    if not code:
        print("no ?code= found in what you pasted"); return
    if state is not None and STATE_FILE.exists() and state != STATE_FILE.read_text().strip():
        print("state mismatch — re-run 'url' and try again (possible CSRF / stale link)"); return
    data = urllib.parse.urlencode({
        "grant_type": "authorization_code", "code": code, "redirect_uri": REDIRECT,
        "client_id": cid, "client_secret": sec}).encode()
    req = urllib.request.Request("https://www.linkedin.com/oauth/v2/accessToken", data=data,
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        tok = json.load(urllib.request.urlopen(req, timeout=20))
    except urllib.error.HTTPError as e:
        print(f"token exchange HTTP {e.code}: {e.read().decode()[:300]}"); return
    access = tok["access_token"]
    ui_req = urllib.request.Request("https://api.linkedin.com/v2/userinfo",
                                    headers={"Authorization": f"Bearer {access}"})
    ui = json.load(urllib.request.urlopen(ui_req, timeout=20))
    urn = f"urn:li:person:{ui['sub']}"
    set_env({"linked_in_access_token": access, "linked_in_member_urn": urn})
    print(f"\nOK — connected as {ui.get('name','(member)')}. Token + URN saved to .env.")
    print("expires_in ~", round(tok.get("expires_in", 0) / 86400), "days")
    print("Now post with: python scripts/social/post_linkedin.py --text '...' [--image path] [--dry-run]")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("url", "exchange"):
        print("usage: linkedin_auth.py url   |   linkedin_auth.py exchange '<landed localhost URL>'")
    elif sys.argv[1] == "url":
        cmd_url()
    else:
        if len(sys.argv) < 3:
            print("paste the localhost URL you landed on, e.g.:\n  linkedin_auth.py exchange 'http://localhost:8771/callback?code=...&state=...'")
        else:
            cmd_exchange(sys.argv[2])
