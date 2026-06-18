"""Build NS-RAM explainer film: TTS narration + manim render + ffmpeg compose."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT  = ROOT / "tmp/nsram_film"
OUT.mkdir(parents=True, exist_ok=True)

# Load env
env_path = ROOT / ".env"
for line in env_path.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    os.environ.setdefault(k.upper(), v)
    os.environ.setdefault(k, v)

api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("openai_api_key")
assert api_key, "no OpenAI key"
os.environ["OPENAI_API_KEY"] = api_key

# Per-scene narration. Word counts ≈ 30–60 to match 45–75s scenes.
NARRATION = {
    "Scene1Faucet": (
        "Let's start simple. A regular transistor is just a tiny switch. "
        "You put a voltage on the gate, and current flows from drain to source. "
        "Turn the voltage off, and the current stops. That's it. "
        "There is no memory. The transistor doesn't remember what just happened."
    ),
    "Scene2Reservoir": (
        "But what if it could remember? "
        "NS-RAM adds a tiny reservoir inside the transistor's body. "
        "When current flows, charge builds up in that body. "
        "When the body fills up, something dramatic happens: positive feedback kicks in, "
        "and the transistor snaps wide open. We call this snapback. "
        "Now the device remembers how much current flowed through it."
    ),
    "Scene3TwoTcell": (
        "Stack two of these transistors together. "
        "The first one, M1, drives charge into a shared floating body, hidden inside a deep N-well. "
        "The second one, M2, reads the body. "
        "When the body is full, M2 fires a sharp output spike. "
        "Two transistors, one floating body. It behaves like a neuron, built from raw silicon."
    ),
    "Scene4Modeling": (
        "To actually use these devices, we need a software model that matches real silicon. "
        "We measured a real chip from our collaborator Sebas. "
        "Our first model was two and a half decades off in current. "
        "Version two got us to one point four. Our current best fits the DC curves to within one decade. "
        "But the snapback fold at high voltage is still wrong. That means our physics is incomplete. We need to add charge traps."
    ),
    "Scene5Networks": (
        "One neuron is interesting. Thousands are useful. "
        "We scaled up from one thousand cells, to four thousand, to sixteen thousand. "
        "On the UCI human activity benchmark, the network learned to tell walking from sitting from lying down. "
        "Accuracy climbed from eighty point two, to eighty three point nine, to eighty four point one percent. "
        "All of this at just thirty five nanojoules per inference."
    ),
    "Scene6Status": (
        "So where are we today? "
        "On the win column: hyper-dimensional computing at eighty four percent, "
        "and a Bayesian random number generator that passes all five NIST tests. "
        "Still on hold: the snapback gap, keyword spotting stuck at chance, the model ten times off in current. "
        "Next up: add charge traps, model the N-well diode, talk to Sebas, rebuild the topology. "
        "Real progress, real gaps, and we now know exactly what's missing."
    ),
}

SCENES = list(NARRATION.keys())

# 1) TTS
def synth():
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    for s in SCENES:
        mp3 = OUT / f"audio_{s}.mp3"
        if mp3.exists() and mp3.stat().st_size > 1000:
            print(f"[tts] {s} cached")
            continue
        print(f"[tts] {s}")
        resp = client.audio.speech.create(model="tts-1-hd", voice="onyx", input=NARRATION[s])
        resp.stream_to_file(str(mp3))
    print("[tts] done")

# 2) Render manim
def render(quality="-qh"):
    media = OUT / "manim_media"
    media.mkdir(exist_ok=True)
    for s in SCENES:
        out_mp4 = OUT / f"scene_{s}.mp4"
        if out_mp4.exists() and out_mp4.stat().st_size > 50000:
            print(f"[manim] {s} cached")
            continue
        print(f"[manim] rendering {s}")
        cmd = [
            str(ROOT / "venv/bin/manim"),
            quality,
            "--media_dir", str(media),
            "--output_file", f"scene_{s}.mp4",
            str(ROOT / "scripts/video/nsram_film.py"),
            s,
        ]
        env = os.environ.copy()
        env["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print("STDERR:", r.stderr[-3000:])
            raise RuntimeError(f"manim failed for {s}")
        # find rendered mp4
        found = list(media.rglob(f"scene_{s}.mp4"))
        if not found:
            raise RuntimeError(f"output not found for {s}")
        # use the highest quality one
        chosen = sorted(found, key=lambda p: p.stat().st_size, reverse=True)[0]
        subprocess.run(["cp", str(chosen), str(out_mp4)], check=True)
        print(f"[manim] {s} -> {out_mp4.stat().st_size/1e6:.1f} MB")

# 3) Mux audio per-scene, padding the shorter stream
def mux_one(scene):
    vmp4 = OUT / f"scene_{scene}.mp4"
    amp3 = OUT / f"audio_{scene}.mp3"
    out  = OUT / f"av_{scene}.mp4"
    # Get durations
    def dur(p):
        r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","default=nw=1:nk=1",str(p)], capture_output=True, text=True)
        return float(r.stdout.strip())
    vd = dur(vmp4); ad = dur(amp3)
    target = max(vd, ad) + 0.3
    print(f"[mux] {scene}: video={vd:.1f}s audio={ad:.1f}s -> {target:.1f}s")
    # Re-encode video with tpad to extend last frame if audio longer
    cmd = [
        "ffmpeg","-y",
        "-i", str(vmp4),
        "-i", str(amp3),
        "-filter_complex", f"[0:v]tpad=stop_mode=clone:stop_duration={max(0, ad-vd+0.3):.2f},fps=30[v];[1:a]apad=pad_dur={max(0, vd-ad+0.3):.2f}[a]",
        "-map","[v]","-map","[a]",
        "-c:v","libx264","-preset","medium","-crf","20","-pix_fmt","yuv420p",
        "-c:a","aac","-b:a","192k",
        "-t", f"{target:.2f}",
        str(out)
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-2000:])
        raise RuntimeError(f"mux failed for {scene}")

def mux_all():
    for s in SCENES:
        mux_one(s)

# 4) Concat with crossfade
def concat():
    inputs = [OUT / f"av_{s}.mp4" for s in SCENES]
    # Use xfade for smooth transitions
    # Build a filtergraph: chain xfade across N inputs
    args = ["ffmpeg","-y"]
    for p in inputs:
        args += ["-i", str(p)]
    # Get duration of each
    def dur(p):
        r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","default=nw=1:nk=1",str(p)], capture_output=True, text=True)
        return float(r.stdout.strip())
    durs = [dur(p) for p in inputs]
    print(f"[concat] durs={[f'{d:.1f}' for d in durs]}")
    xfade_dur = 0.5
    # Chain xfades
    # [0:v][1:v]xfade=transition=fade:duration=0.5:offset=d0-0.5[v01]
    # [v01][2:v]xfade=...:offset=(d0+d1-2*0.5)[v012]
    v_filters = []
    a_filters = []
    prev_v = "[0:v]"
    prev_a = "[0:a]"
    cum = durs[0]
    for i in range(1, len(inputs)):
        offset = cum - xfade_dur
        out_v = f"[v{i}]"
        out_a = f"[a{i}]"
        v_filters.append(f"{prev_v}[{i}:v]xfade=transition=fade:duration={xfade_dur}:offset={offset:.3f}{out_v}")
        a_filters.append(f"{prev_a}[{i}:a]acrossfade=d={xfade_dur}{out_a}")
        prev_v = out_v
        prev_a = out_a
        cum = cum + durs[i] - xfade_dur
    filt = ";".join(v_filters + a_filters)
    final = OUT / "nsram_explainer.mp4"
    args += [
        "-filter_complex", filt,
        "-map", prev_v, "-map", prev_a,
        "-c:v","libx264","-preset","medium","-crf","20","-pix_fmt","yuv420p",
        "-c:a","aac","-b:a","192k",
        str(final)
    ]
    print("[concat] running ffmpeg")
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-3000:])
        raise RuntimeError("concat failed")
    print(f"[concat] -> {final} ({final.stat().st_size/1e6:.1f} MB)")
    # Probe final duration
    r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","default=nw=1:nk=1",str(final)], capture_output=True, text=True)
    print(f"[concat] duration={float(r.stdout.strip()):.1f}s")

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("tts","all"): synth()
    if stage in ("render","all"): render(quality=os.environ.get("MQUAL","-qh"))
    if stage in ("mux","all"): mux_all()
    if stage in ("concat","all"): concat()
