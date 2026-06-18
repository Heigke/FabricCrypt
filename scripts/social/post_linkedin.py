"""Post a progress update to LinkedIn using the token stored by linkedin_auth.py.

Text+video:  python scripts/social/post_linkedin.py --text "..." --video results/.../foo.mp4
Text+image:  python scripts/social/post_linkedin.py --text "..." --image results/.../foo.png
Text only:   python scripts/social/post_linkedin.py --text "..."
Dry-run:     add --dry-run to print exactly what WOULD be posted without sending.

Uses the LinkedIn UGC Posts API (/v2/ugcPosts). Visibility defaults to PUBLIC.
Video is uploaded via assets registerUpload (feedshare-video recipe) then polled until AVAILABLE.
Refuses empty text. One media item per post (video OR image).
"""
from __future__ import annotations
import argparse, json, mimetypes, time, urllib.request, urllib.error
from pathlib import Path

ENV = Path(__file__).resolve().parents[2] / ".env"
API = "https://api.linkedin.com/v2"


def load_env():
    d = {}
    for ln in ENV.read_text().splitlines():
        ln = ln.strip()
        if "=" in ln and not ln.startswith("#"):
            k, v = ln.split("=", 1); d[k.strip()] = v.strip().strip('"').strip("'")
    return d


def api(method, path, token, body=None, raw=None, ctype="application/json"):
    url = path if path.startswith("http") else f"{API}{path}"
    headers = {"Authorization": f"Bearer {token}", "X-Restli-Protocol-Version": "2.0.0"}
    if raw is not None:
        data = raw; headers["Content-Type"] = ctype
    elif body is not None:
        data = json.dumps(body).encode(); headers["Content-Type"] = "application/json"
    else:
        data = None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    r = urllib.request.urlopen(req, timeout=120)
    txt = r.read().decode() or "{}"
    return r, (json.loads(txt) if txt.strip().startswith(("{", "[")) else {"_raw": txt[:200]})


def register_upload(token, urn, recipe):
    reg = {"registerUploadRequest": {
        "recipes": [f"urn:li:digitalmediaRecipe:{recipe}"],
        "owner": urn, "serviceRelationships": [
            {"relationshipType": "OWNER", "identifier": "urn:li:userGeneratedContent"}]}}
    _, j = api("POST", "/assets?action=registerUpload", token, body=reg)
    up = j["value"]["uploadMechanism"][
        "com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest"]["uploadUrl"]
    return up, j["value"]["asset"]


def put_bytes(token, url, path: Path):
    blob = path.read_bytes()
    ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    api("PUT", url, token, raw=blob, ctype=ctype)


def wait_asset(token, asset, timeout=300):
    """Poll until the video asset finishes processing (status AVAILABLE)."""
    aid = asset.split(":")[-1]
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            _, j = api("GET", f"/assets/{aid}", token)
        except urllib.error.HTTPError:
            time.sleep(4); continue
        st = j.get("recipes", [{}])[0].get("status") or j.get("status")
        if st in ("AVAILABLE", "PROCESSING_DONE"):
            return True
        if st in ("PROCESSING_FAILED", "CLIENT_ERROR"):
            return False
        time.sleep(5)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.text.strip():
        print("refusing to post empty text"); return
    if a.image and a.video:
        print("use only one of --image / --video"); return
    mpath = Path(a.video or a.image) if (a.video or a.image) else None
    if mpath and not mpath.exists():
        print(f"media not found: {mpath}"); return

    env = load_env()
    token = env.get("linked_in_access_token"); urn = env.get("linked_in_member_urn")
    if not token or not urn:
        print("no access token — run: python scripts/social/linkedin_auth.py url|exchange first"); return

    if a.dry_run:
        kind = "VIDEO" if a.video else ("IMAGE" if a.image else "TEXT")
        sz = f"{mpath.stat().st_size/1e6:.1f} MB" if mpath else "-"
        print(f"=== DRY RUN — would post as {urn} ({kind}) ===")
        print(a.text)
        if mpath: print(f"[+ {kind.lower()}: {mpath}  ({sz})]")
        return

    cat = "NONE"; media = []
    if a.video:
        up, asset = register_upload(token, urn, "feedshare-video")
        put_bytes(token, up, mpath)
        print("video uploaded, waiting for LinkedIn processing...")
        if not wait_asset(token, asset):
            print("video did not finish processing in time — aborting (not posted)"); return
        cat = "VIDEO"; media = [{"status": "READY", "media": asset,
                                 "title": {"text": "FabricCrypt — AI embodiment"}}]
    elif a.image:
        up, asset = register_upload(token, urn, "feedshare-image")
        put_bytes(token, up, mpath)
        cat = "IMAGE"; media = [{"status": "READY", "media": asset,
                                 "description": {"text": a.text[:200]}, "title": {"text": "FabricCrypt"}}]

    share = {"shareCommentary": {"text": a.text}, "shareMediaCategory": cat}
    if media: share["media"] = media
    post = {"author": urn, "lifecycleState": "PUBLISHED",
            "specificContent": {"com.linkedin.ugc.ShareContent": share},
            "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"}}
    r, j = api("POST", "/ugcPosts", token, body=post)
    pid = r.headers.get("x-restli-id") or j.get("id", "(id?)")
    print(f"POSTED ok -> {pid}")
    print(f"view: https://www.linkedin.com/feed/update/{pid}")


if __name__ == "__main__":
    main()
