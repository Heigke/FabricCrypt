"""One-shot LinkedIn OAuth callback catcher + token exchange.
Binds localhost:8771, waits for the browser redirect, captures ?code=, exchanges it for an
access token, fetches the member URN, writes both into .env, then exits. Run in background;
verify it is listening (ss -ltn | grep 8771) before clicking the auth link.

Status is written to /tmp/li_capture_status.txt (NEVER the token) so the parent can poll.
"""
from __future__ import annotations
import http.server, json, urllib.parse, urllib.request
from pathlib import Path

ENV = Path(__file__).resolve().parents[2] / ".env"
PORT = 8771
REDIRECT = f"http://localhost:{PORT}/callback"
STATUS = Path("/tmp/li_capture_status.txt")


def load_env():
    d = {}
    for ln in ENV.read_text().splitlines():
        ln = ln.strip()
        if "=" in ln and not ln.startswith("#"):
            k, v = ln.split("=", 1); d[k.strip()] = v.strip().strip('"').strip("'")
    return d


def set_env(updates: dict):
    lines = ENV.read_text().splitlines(); seen = set()
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


def status(msg): STATUS.write_text(msg + "\n")


def exchange(code):
    env = load_env()
    data = urllib.parse.urlencode({
        "grant_type": "authorization_code", "code": code, "redirect_uri": REDIRECT,
        "client_id": env.get("linked_in_client_id"), "client_secret": env.get("linked_in_secret")}).encode()
    req = urllib.request.Request("https://www.linkedin.com/oauth/v2/accessToken", data=data,
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
    tok = json.load(urllib.request.urlopen(req, timeout=20))
    access = tok["access_token"]
    ui = json.load(urllib.request.urlopen(urllib.request.Request(
        "https://api.linkedin.com/v2/userinfo", headers={"Authorization": f"Bearer {access}"}), timeout=20))
    urn = f"urn:li:person:{ui['sub']}"
    set_env({"linked_in_access_token": access, "linked_in_member_urn": urn})
    return ui.get("name", "(member)"), round(tok.get("expires_in", 0) / 86400)


class H(http.server.BaseHTTPRequestHandler):
    done = False
    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path != "/callback":
            self.send_response(204); self.end_headers(); return        # ignore favicon etc.
        q = urllib.parse.parse_qs(u.query)
        code = q.get("code", [None])[0]; err = q.get("error", [None])[0]
        body = "<h2>%s</h2>"
        try:
            if err:
                status(f"ERROR {err}: {q.get('error_description',[''])[0]}")
                msg = f"LinkedIn returned error: {err} — {q.get('error_description',[''])[0]}"
            elif code:
                name, days = exchange(code)
                status(f"OK connected={name} expires_days={days}")
                msg = f"Connected as {name}. Token saved. You can close this tab."
            else:
                status("ERROR no code and no error in callback"); msg = "No code received."
        except Exception as e:
            status(f"EXCHANGE_FAIL {type(e).__name__}: {str(e)[:200]}")
            msg = f"Token exchange failed: {e}"
        out = (body % msg).encode()
        self.send_response(200); self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(out))); self.end_headers()
        self.wfile.write(out)
        H.done = True
    def log_message(self, *a): pass


if __name__ == "__main__":
    status("LISTENING")
    http.server.HTTPServer.allow_reuse_address = True
    srv = http.server.HTTPServer(("127.0.0.1", PORT), H)
    while not H.done:
        srv.handle_request()
