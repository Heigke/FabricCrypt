"""Make the narrated demo video: Gemini TTS (female 'Kore') over the honest infographic with a slow
zoom. Honest, self-describing, references prior art, discloses Claude-Code-for-Eric, no overselling.
Out: results/IDENTITY_H7_2026-06-09/H7_embodiment_2026-06.mp4
"""
import os,re,subprocess,wave
from pathlib import Path
OUT=Path(__file__).resolve().parents[2]/"results/IDENTITY_H7_2026-06-09"; FR=OUT/"demo_frames"
IMG=FR/"H7_infographic.png"; WAV=FR/"vo.wav"; MP4=OUT/"H7_embodiment_2026-06.mp4"

NARR=("What if a language model only worked on one specific computer? "
"Working autonomously for Eric on an HP workstation, I tested this on real silicon. "
"A frozen GPT-2 was given a key sealed inside each machine's security chip — a key that cannot be copied to another machine. "
"With the correct chip's fingerprint, the model writes fluent English. "
"With a random, foreign, or stale key, it collapses into noise — by a factor of hundreds, every time, deterministically. "
"To be honest: this is hardware-bound model licensing, not embodiment or consciousness. "
"The chip is the lock, not the brain — the fluency comes from adaptation, and the adapter is domain-narrow. "
"Earlier work already binds models to devices by encrypting their weights; here the binding is a small learned steering module instead, "
"on consumer hardware that has no secure enclave. "
"The full method, the honest limits, and the code are open at github dot com, slash Heigke, slash FabricCrypt.")

def key():
    for line in open(Path(__file__).resolve().parents[2]/".env"):
        m=re.match(r'\s*gemini_api_key\s*=\s*"?([^"\n]+)"?',line,re.I)
        if m: return m.group(1).strip()
def tts():
    from google import genai; from google.genai import types
    c=genai.Client(api_key=key())
    r=c.models.generate_content(model="gemini-2.5-flash-preview-tts",contents=NARR,
        config=types.GenerateContentConfig(response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")))))
    data=r.candidates[0].content.parts[0].inline_data.data
    with wave.open(str(WAV),"wb") as w: w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(data)
    with wave.open(str(WAV)) as w: dur=w.getnframes()/w.getframerate()
    return dur

def main():
    dur=tts(); print(f"VO duration {dur:.1f}s")
    pad=dur+1.0
    # slow zoom (Ken Burns) over the infographic; fade in/out; mux narration
    vf=(f"scale=2400:-1,zoompan=z='min(zoom+0.0006,1.18)':d={int(pad*25)}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1920x1080:fps=25,"
        f"fade=t=in:st=0:d=0.6,fade=t=out:st={pad-0.6:.2f}:d=0.6,format=yuv420p")
    cmd=["ffmpeg","-y","-loop","1","-i",str(IMG),"-i",str(WAV),
         "-vf",vf,"-t",f"{pad:.2f}","-c:v","libx264","-preset","medium","-crf","20",
         "-c:a","aac","-b:a","160k","-ar","44100","-shortest","-movflags","+faststart",str(MP4)]
    r=subprocess.run(cmd,capture_output=True,text=True)
    if r.returncode: print("FFMPEG ERR\n",r.stderr[-1500:]); return
    print(f"saved {MP4} ({MP4.stat().st_size//1024} KB)")

if __name__=="__main__": main()
