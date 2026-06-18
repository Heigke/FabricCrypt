"""Oracle dispatcher — fire an oracle query packet at multiple LLM providers
WITH FILE UPLOADS (so the oracle sees the actual code, plot, schematic).

OpenAI: uses Responses API with `input_file` / `input_image` parts
        (https://platform.openai.com/docs/guides/pdf-files).
Gemini: uses google.genai client.files.upload then references the file
        in contents= alongside the prompt
        (https://ai.google.dev/gemini-api/docs/files).
Grok / Deepseek: text-only fallback (no native file API in their docs at
        time of writing — we inline text artifacts as fenced blocks).

Loads keys from /home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/.env

Usage:
    python scripts/oracle_dispatch.py PACKET_DIR [--providers openai,gemini]

The PACKET_DIR must contain prompt.md. We:
  1. Extract any *.zip to a temp dir
  2. Upload each text/image/pdf file (≤25 MB) to OpenAI and Gemini
  3. For Grok/Deepseek inline text files in the prompt
  4. Save responses to PACKET_DIR/{provider}_response.md
"""
from __future__ import annotations
import argparse
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path

# Load .env into os.environ
ENV_PATHS = [
    Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/.env"),
    Path("/home/ikaros/Documents/claude_hive/.env"),
]
for ep in ENV_PATHS:
    if ep.exists():
        for line in ep.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.upper(), v.strip())

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GROK_KEY = os.environ.get("GROK_API_KEY", "")
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

TEXT_EXTS = {".md", ".txt", ".csv", ".asc", ".py", ".json", ".tcl", ".sp"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
PDF_EXTS = {".pdf"}
SKIP_NAMES = {"_context.md"}


def gather_packet_files(packet_dir: Path, bundle: bool = True) -> tuple[Path, list[Path]]:
    """Return (prompt_file, list_of_attachment_files).

    Extracts any .zip in the packet dir into a sibling tmp folder. If
    `bundle=True`, all small text files are also concatenated into a
    single `_bundle.md` and that bundle is added as an extra attachment
    (oracles read the bundle as one file, getting full traversal context
    in one shot).

    Drops obvious cruft (cache, tests, packet zips themselves).

    Renames .asc → .txt (OpenAI doesn't accept .asc) by symlinking into
    `_normalised/`.
    """
    prompt_file = packet_dir / "prompt.md"
    if not prompt_file.exists():
        raise SystemExit(f"No prompt.md in {packet_dir}")

    files = []

    def keep(p: Path) -> bool:
        if not p.is_file():
            return False
        if p.name in SKIP_NAMES:
            return False
        if p.name.startswith("_"):
            return False
        if any(seg in p.parts for seg in ("__pycache__", ".git", "tests")):
            return False
        if p.name == "prompt.md":
            return False
        if p.name.endswith("_response.md"):
            return False
        if p.suffix == ".zip":
            return False
        return p.suffix.lower() in TEXT_EXTS | IMAGE_EXTS | PDF_EXTS

    # Top-level + artifacts/
    for root in (packet_dir, packet_dir / "artifacts"):
        if not root.exists():
            continue
        for p in sorted(root.iterdir()):
            if keep(p):
                files.append(p)

    # Extract zips into siblings
    extract_root = packet_dir / "_extracted"
    extract_root.mkdir(exist_ok=True)
    for z in sorted(packet_dir.glob("*.zip")):
        try:
            with zipfile.ZipFile(z) as zf:
                tgt = extract_root / z.stem
                tgt.mkdir(exist_ok=True)
                zf.extractall(tgt)
        except zipfile.BadZipFile:
            continue

    for p in extract_root.rglob("*"):
        if keep(p):
            files.append(p)

    # Normalise extensions: rename .asc → .txt for upload compatibility
    norm_root = packet_dir / "_normalised"
    norm_root.mkdir(exist_ok=True)
    final_files = []
    seen_names: set[str] = set()
    for f in files:
        ext = f.suffix.lower()
        if ext == ".asc":
            # Copy with .txt suffix (OpenAI rejects .asc)
            dst = norm_root / (f.stem + "_asc.txt")
            dst.write_bytes(f.read_bytes())
            final_files.append(dst)
        else:
            final_files.append(f)

    # Optional bundle: concatenate all text files into one big .md
    if bundle:
        bundle_path = packet_dir / "_bundle_all_text.md"
        with bundle_path.open("w") as fp:
            fp.write("# Combined text bundle for oracle context\n\n")
            fp.write("All text artifacts concatenated below for one-shot context. "
                     "Each file is delimited by a `=== FILE: <name> ===` marker.\n\n")
            for f in final_files:
                if f.suffix.lower() not in TEXT_EXTS:
                    continue
                try:
                    txt = f.read_text(errors="replace")
                except Exception:
                    continue
                rel = f.relative_to(packet_dir) if f.is_relative_to(packet_dir) else f.name
                fp.write(f"\n\n=== FILE: {rel} ({len(txt)} chars) ===\n")
                # Use markdown fence per file
                lang = {"py": "python", "csv": "csv", "json": "json",
                        "txt": "", "md": ""}.get(f.suffix.lstrip(".").lower(), "")
                fp.write(f"```{lang}\n{txt}\n```\n")
        # Add bundle to the head of the file list (oracle may prefer
        # the bundle and ignore individual files; that's fine)
        final_files = [bundle_path] + final_files

    return prompt_file, final_files


# ───────────────────────── OpenAI ───────────────────────── #

def call_openai(prompt: str, files: list[Path], model: str) -> str:
    if not OPENAI_KEY:
        return "[openai: no key]"
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_KEY)

    # Upload files. Skip the _bundle_all_text.md bundle — its content is
    # already covered by the individual files, and including both blows
    # past gpt-5's context window.
    uploaded = []
    skipped = []
    for f in files:
        if f.name == "_bundle_all_text.md":
            skipped.append((f, "bundle excluded (context window)"))
            continue
        try:
            sz = f.stat().st_size
            if sz > 30_000_000:
                skipped.append((f, "too large"))
                continue
            up = client.files.create(
                file=open(f, "rb"),
                purpose="user_data",
            )
            uploaded.append((f, up.id))
        except Exception as e:
            skipped.append((f, str(e)[:120]))

    print(f"  [openai] uploaded {len(uploaded)}/{len(files)} files; "
          f"{len(skipped)} skipped", flush=True)
    for f, why in skipped[:5]:
        print(f"  [openai]   skip {f.name}: {why}", flush=True)

    # Build content array
    content = [{"type": "input_text", "text": prompt}]
    for f, fid in uploaded:
        if f.suffix.lower() in IMAGE_EXTS:
            content.append({"type": "input_image", "file_id": fid})
        else:
            content.append({"type": "input_file", "file_id": fid})

    try:
        resp = client.responses.create(
            model=model,
            input=[{"role": "user", "content": content}],
            reasoning={"effort": "high"},
        )
        out = resp.output_text
    except Exception as e:
        out = f"[openai responses.create error: {e}]"

    # Cleanup uploaded files (free quota)
    for _, fid in uploaded:
        try:
            client.files.delete(fid)
        except Exception:
            pass
    return out


# ───────────────────────── Gemini ───────────────────────── #

def call_gemini(prompt: str, files: list[Path], model: str) -> str:
    if not GEMINI_KEY:
        return "[gemini: no key]"
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=GEMINI_KEY)

    # MIME type override: Gemini auto-detects content and sometimes mis-IDs
    # text files (e.g. parasiticBJT.txt → "application/pgp-keys"). Force
    # text/plain for known text extensions.
    def upload_with_mime(p: Path):
        ext = p.suffix.lower()
        if ext in TEXT_EXTS:
            return client.files.upload(
                file=str(p),
                config=types.UploadFileConfig(mime_type="text/plain"),
            )
        if ext == ".png":
            return client.files.upload(
                file=str(p),
                config=types.UploadFileConfig(mime_type="image/png"),
            )
        if ext in (".jpg", ".jpeg"):
            return client.files.upload(
                file=str(p),
                config=types.UploadFileConfig(mime_type="image/jpeg"),
            )
        if ext == ".pdf":
            return client.files.upload(
                file=str(p),
                config=types.UploadFileConfig(mime_type="application/pdf"),
            )
        return client.files.upload(file=str(p))  # fallback (auto-detect)

    uploaded = []
    skipped = []
    for f in files:
        try:
            sz = f.stat().st_size
            if sz > 50_000_000:
                skipped.append((f, "too large"))
                continue
            up = upload_with_mime(f)
            uploaded.append((f, up))
        except Exception as e:
            skipped.append((f, str(e)[:120]))

    # Wait for any ACTIVE state files
    deadline = time.time() + 60
    for f, up in uploaded:
        while up.state.name == "PROCESSING" and time.time() < deadline:
            time.sleep(2)
            up = client.files.get(name=up.name)

    print(f"  [gemini] uploaded {len(uploaded)}/{len(files)} files; "
          f"{len(skipped)} skipped", flush=True)
    for f, why in skipped[:5]:
        print(f"  [gemini]   skip {f.name}: {why}", flush=True)

    contents = [u for _, u in uploaded] + [prompt]
    try:
        resp = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=-1),
                temperature=0.2,
            ),
        )
        out = resp.text or "[gemini: empty]"
    except Exception as e:
        out = f"[gemini error: {e}]"

    # Cleanup
    for _, up in uploaded:
        try:
            client.files.delete(name=up.name)
        except Exception:
            pass
    return out


# ───────────────────────── Grok / xAI ───────────────────────── #

def call_grok(prompt: str, files: list[Path], model: str) -> str:
    """Grok-4 supports multimodal: text + image_url content parts via the
    OpenAI-compatible chat-completions endpoint. We inline text files,
    base64-embed images, and shell-extract PDF text via `pdftotext`."""
    if not GROK_KEY:
        return "[grok: no key]"
    import base64, mimetypes, subprocess, requests

    inlined_text: list[str] = []
    image_parts: list[dict] = []
    skipped: list[str] = []

    for f in files:
        ext = f.suffix.lower()
        try:
            if ext in TEXT_EXTS:
                txt = f.read_text(errors="replace")
                if len(txt) > 120_000:
                    txt = txt[:120_000] + "\n... [truncated]\n"
                inlined_text.append(f"\n=== FILE {f.name} ===\n```\n{txt}\n```\n")
            elif ext in PDF_EXTS:
                # pdftotext -layout for readable extraction
                r = subprocess.run(
                    ["pdftotext", "-layout", str(f), "-"],
                    capture_output=True, text=True, timeout=30,
                )
                txt = r.stdout
                if len(txt) > 120_000:
                    txt = txt[:120_000] + "\n... [truncated]\n"
                inlined_text.append(
                    f"\n=== FILE {f.name} (PDF text-extracted) ===\n```\n{txt}\n```\n"
                )
            elif ext in IMAGE_EXTS:
                data = f.read_bytes()
                if len(data) > 18_000_000:
                    skipped.append(f"{f.name}: image >18MB"); continue
                mime = mimetypes.guess_type(str(f))[0] or "image/jpeg"
                b64 = base64.b64encode(data).decode("ascii")
                image_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                })
            else:
                skipped.append(f"{f.name}: unsupported ext {ext}")
        except Exception as e:
            skipped.append(f"{f.name}: {e!r}")

    print(f"  [grok] {len(image_parts)} images + {len(inlined_text)} text/pdf attachments; "
          f"{len(skipped)} skipped", flush=True)
    for s in skipped:
        print(f"  [grok]   skip {s}", flush=True)

    user_content: list[dict] = [
        {"type": "text",
         "text": prompt + "\n\n## Attached text/PDF files\n" + "".join(inlined_text)},
    ] + image_parts

    try:
        r = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROK_KEY}",
                     "Content-Type": "application/json"},
            json={"model": model,
                  "messages": [{"role": "user", "content": user_content}]},
            timeout=600,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[grok error: {e}; body={getattr(r, 'text', '')[:500] if 'r' in dir() else ''}]"


def call_deepseek(prompt: str, files: list[Path], model: str) -> str:
    if not DEEPSEEK_KEY:
        return "[deepseek: no key]"
    import requests
    inlined = []
    for f in files:
        if f.suffix in TEXT_EXTS:
            try:
                txt = f.read_text(errors="replace")
                if len(txt) > 60_000:
                    txt = txt[:60_000] + "\n... [truncated]\n"
                inlined.append(f"\n=== FILE {f.name} ===\n```\n{txt}\n```\n")
            except Exception:
                pass
    full_prompt = prompt + "\n\n## Attached files\n" + "".join(inlined)
    try:
        r = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {DEEPSEEK_KEY}",
                     "Content-Type": "application/json"},
            json={"model": model,
                  "messages": [{"role": "user", "content": full_prompt}]},
            timeout=900,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[deepseek error: {e}; body={getattr(r, 'text', '')[:500]}]"


PROVIDERS = {
    "openai": call_openai,
    "gemini": call_gemini,
    "grok": call_grok,
    "deepseek": call_deepseek,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("packet_dir", type=Path)
    ap.add_argument("--providers", default="openai,gemini")
    ap.add_argument("--openai-model", default="gpt-5")
    ap.add_argument("--gemini-model", default="gemini-2.5-pro")
    ap.add_argument("--grok-model", default="grok-4-latest")
    ap.add_argument("--deepseek-model", default="deepseek-reasoner")
    args = ap.parse_args()

    packet_dir = args.packet_dir.resolve()
    if not packet_dir.exists():
        sys.exit(f"No such directory: {packet_dir}")

    print(f"[oracle] gather files from {packet_dir} ...", flush=True)
    prompt_file, files = gather_packet_files(packet_dir)
    prompt = prompt_file.read_text()
    print(f"[oracle] prompt = {len(prompt)} chars; "
          f"{len(files)} attachments to upload", flush=True)
    for f in files[:8]:
        print(f"  - {f.relative_to(packet_dir) if f.is_relative_to(packet_dir) else f.name} "
              f"({f.stat().st_size//1024} KB)", flush=True)
    if len(files) > 8:
        print(f"  ... and {len(files)-8} more", flush=True)

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    model_for = {
        "openai": args.openai_model,
        "gemini": args.gemini_model,
        "grok": args.grok_model,
        "deepseek": args.deepseek_model,
    }
    for p in providers:
        if p not in PROVIDERS:
            print(f"[oracle] unknown provider: {p} (skipping)", flush=True)
            continue
        print(f"\n[oracle] dispatching to {p} ({model_for[p]})...", flush=True)
        t0 = time.time()
        resp = PROVIDERS[p](prompt, files, model=model_for[p])
        elapsed = time.time() - t0
        out = packet_dir / f"{p}_response.md"
        out.write_text(f"# {p} response ({model_for[p]}) — {elapsed:.0f}s\n\n{resp}\n")
        print(f"[oracle] {p}: {len(resp):,} chars saved to {out.name} "
              f"({elapsed:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
