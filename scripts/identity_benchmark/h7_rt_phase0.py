"""H7 Phase 0 — the hardened reafference experiment (the kill-or-greenlight for "embodiment").

Question (Eric's vision, honestly tested): does the LLM's OWN token generation perturb its host's
live hardware telemetry in a CONTENT-SPECIFIC way (genuine reafference / Butlin AE-2), or is any
apparent loop just "compute happened → DVFS/thermal" (a loaded heater)? And does the FULL ~30-50-D
signal set carry information a 1-D power signal does not (Eric's meta-computation question)?

Design (oracle-mandated controls, all in ONE aligned recording per condition):
  C0_IDLE      no work (baseline governor/thermal dynamics + sensor noise floor)
  C1_SELF      real LM generation — content VARIES token to token
  C3_SWAP      identical forward-pass compute, but RANDOM/scrambled input ids — content destroyed,
               compute held constant  → C1 vs C3 isolates CONTENT-specific reafference (DeepSeek)
  C2_YOKED     non-LLM matmul load matched to C1 cadence/size — generic compute, no model (Grok)

Telemetry: full live vector (per-core Vcore + per-core clock + package power + GPU power/clock/temp
+ thermal) sampled in a background thread, monotonic-timestamped, read-ONLY (never writes SMU).
Token events: monotonic ts + token id + per-token entropy + top1 logit + chosen-token logprob.

Thermal SAFETY (ikaros laptop, zone0 trip=99°C): per-sample zone0 check; ABORT a condition if
zone0>THERM_ABORT; wait_cool between conditions. Short conditions. GPU bursts bounded.

Out: results/IDENTITY_H7_2026-06-16/phase0_{host}.npz  (raw aligned traces, all conditions)
Usage: HSA_OVERRIDE_GFX_VERSION=11.0.0 python h7_rt_phase0.py --secs 60 --model gpt2
"""
from __future__ import annotations
import argparse, glob, json, os, socket, threading, time
from pathlib import Path
import numpy as np

HOST = socket.gethostname()
def _out_dir():
    env = os.environ.get("H7_OUT")
    if env: return Path(env)
    try:
        d = Path(__file__).resolve().parents[2] / "results/IDENTITY_H7_2026-06-16"
        d.mkdir(parents=True, exist_ok=True); return d
    except Exception:
        return Path.home() / "h7_rt_results"
OUT = _out_dir()
OUT.mkdir(parents=True, exist_ok=True)
PM = Path("/sys/kernel/ryzen_smu_drv/pm_table")
NCPU = os.cpu_count() or 16
CAND_VIDX = [756, 110]            # per-die Vcore offset in PM table (ikaros 756 / daedalus 110)
ZONE0 = Path("/sys/class/thermal/thermal_zone0/temp")
THERM_ABORT = float(os.environ.get("THERM_ABORT", "82"))   # abort a condition above this
THERM_COOL = float(os.environ.get("THERM_COOL", "55"))     # wait_cool target between conditions

# ---------- read-only telemetry readers (mirrors h7_rt_inventory.py) ----------
def zone0():
    try: return int(ZONE0.read_text()) / 1000.0
    except Exception: return 0.0
def read_vcore(n=16, lo=0.5, hi=1.1):
    try:
        b = PM.read_bytes(); v = np.frombuffer(b[:(len(b)//4)*4], dtype=np.float32).astype(float)
    except Exception:
        return None
    for idx in CAND_VIDX:
        if idx + n <= len(v):
            w = v[idx:idx+n]
            if np.all((w >= lo) & (w <= hi)) and w.std() < 0.08:
                return w.copy()
    return None
def read_curfreq():
    out = []
    for c in range(NCPU):
        try: out.append(int(Path(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq").read_text()))
        except Exception: out.append(0)
    return np.array(out, float)
def _hwmon(globpat):
    out = []
    for p in sorted(glob.glob(globpat)):
        try: out.append(int(Path(p).read_text()))
        except Exception: pass
    return np.array(out, float) if out else np.zeros(0)
def read_thermal():
    vals = []
    for p in sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp")):
        try: vals.append(int(Path(p).read_text())/1000.0)
        except Exception: vals.append(0.0)
    return np.array(vals, float)
def read_gpu_power():
    g = _hwmon("/sys/class/hwmon/hwmon*/power1_average")
    return g if g.size else _hwmon("/sys/class/hwmon/hwmon*/power1_input")
def read_gpu_freq(): return _hwmon("/sys/class/hwmon/hwmon*/freq1_input")
def read_gpu_temp(): return _hwmon("/sys/class/hwmon/hwmon*/temp1_input")
def read_misc():
    return np.concatenate([_hwmon("/sys/class/hwmon/hwmon*/in*_input"),
                           _hwmon("/sys/class/hwmon/hwmon*/curr*_input")])

# --- NVIDIA (ZGX/GB10) fast readers via NVML; activate only if available ---
_NV = {"h": None}
def _nv_init():
    try:
        import pynvml; pynvml.nvmlInit(); _NV["m"] = pynvml
        _NV["h"] = pynvml.nvmlDeviceGetHandleByIndex(0); return True
    except Exception:
        _NV["h"] = None; return False
def read_nv_power():
    if _NV["h"] is None: return np.zeros(0)
    try: return np.array([_NV["m"].nvmlDeviceGetPowerUsage(_NV["h"])/1000.0], float)  # W
    except Exception: return np.array([np.nan])
def read_nv_clock():
    if _NV["h"] is None: return np.zeros(0)
    m = _NV["m"]
    try:
        return np.array([m.nvmlDeviceGetClockInfo(_NV["h"], m.NVML_CLOCK_SM),
                         m.nvmlDeviceGetClockInfo(_NV["h"], m.NVML_CLOCK_MEM)], float)
    except Exception: return np.array([np.nan, np.nan])
def read_nv_temp():
    if _NV["h"] is None: return np.zeros(0)
    try: return np.array([_NV["m"].nvmlDeviceGetTemperature(_NV["h"], _NV["m"].NVML_TEMPERATURE_GPU)], float)
    except Exception: return np.array([np.nan])
def read_nv_util():
    if _NV["h"] is None: return np.zeros(0)
    try:
        u = _NV["m"].nvmlDeviceGetUtilizationRates(_NV["h"]); return np.array([u.gpu, u.memory], float)
    except Exception: return np.array([np.nan, np.nan])

_HAS_NV = _nv_init()
CHANNELS = {"vcore": read_vcore, "cur_freq": read_curfreq, "thermal": read_thermal,
            "gpu_power": read_gpu_power, "gpu_freq": read_gpu_freq, "gpu_temp": read_gpu_temp,
            "misc": read_misc,
            "nv_power": read_nv_power, "nv_clock": read_nv_clock, "nv_temp": read_nv_temp,
            "nv_util": read_nv_util}

# ---------- background sampler ----------
class Sampler(threading.Thread):
    def __init__(self, period=0.02):
        super().__init__(daemon=True)
        self.period = period; self.stop = False; self.rows = []; self.hot = False; self._last = 0.0
        # fix channel dims from a probe read
        self.dims = {}
        for k, fn in CHANNELS.items():
            v = fn(); self.dims[k] = (0 if v is None else int(v.size))
    def snap(self):
        rec = {"t": time.monotonic()}
        for k, fn in CHANNELS.items():
            v = fn()
            if v is None or v.size != self.dims[k]:
                v = np.full(self.dims[k], np.nan)
            rec[k] = v
        return rec
    def tick(self):
        """INLINE sampling (single-threaded): append a snap if period elapsed. Used instead of the
        background thread to avoid a daemon-thread × ROCm-GPU deadlock seen on daedalus."""
        now = time.monotonic()
        if now - self._last >= self.period:
            self._last = now
            self.rows.append(self.snap())
            if zone0() >= THERM_ABORT: self.hot = True
    def run(self):
        while not self.stop:
            rec = self.snap()
            z = zone0()
            if z >= THERM_ABORT: self.hot = True
            self.rows.append(rec)
            time.sleep(self.period)
    def matrix(self):
        """Return (t[T], {chan: M[T,dim]})."""
        t = np.array([r["t"] for r in self.rows])
        chans = {k: np.array([r[k] for r in self.rows]) for k in CHANNELS}
        return t, chans

def wait_cool(target=THERM_COOL, timeout=180):
    t0 = time.time()
    while zone0() > target and time.time() - t0 < timeout:
        time.sleep(2.0)

# ---------- conditions ----------
def run_idle(sampler, secs):
    t0 = time.monotonic(); events = []
    while time.monotonic() - t0 < secs and not sampler.hot:
        sampler.tick(); time.sleep(0.004)
    return events

def run_self(lm, tok, dev, secs, prompt, scramble=False):
    """Greedy-ish sampled generation. If scramble=True, feed RANDOM input ids each step (identical
    compute, destroyed content) — the C3 control. Records per-token telemetry-aligned events."""
    import torch
    events = []
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    V = lm.config.vocab_size
    t0 = time.monotonic(); past = None; step = 0
    with torch.no_grad():
        cur = ids
        while time.monotonic() - t0 < secs and not sampler_hot(lm):
            out = lm(cur if past is None else cur[:, -1:], past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :].float()
            p = torch.softmax(logits, -1)
            ent = float(-(p * torch.log(p + 1e-12)).sum().item())
            top1 = float(logits.max().item())
            if scramble:
                nxt = torch.randint(0, V, (1, 1), device=dev)
            else:
                nxt = torch.multinomial(p, 1)
            lp = float(torch.log(p[0, nxt[0, 0]] + 1e-12).item())
            ev = {"t": time.monotonic(), "tok": int(nxt.item()), "entropy": ent,
                  "top1": top1, "logprob": lp, "step": step}
            events.append(ev)
            _HOT["s"].tick()                       # inline telemetry sample (no bg thread)
            cur = torch.cat([cur, nxt], 1)
            if cur.shape[1] > 256:   # keep <=256 ctx (GPU-hang guard); reset cache window
                cur = cur[:, -64:]; past = None
            step += 1
    return events

_HOT = {"s": None}
def sampler_hot(_): return _HOT["s"].hot if _HOT["s"] else False

def run_yoked(secs, size, cadence):
    """Non-LLM matmul load: bursts of size×size matmul on GPU at ~cadence Hz, matched to C1 load."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    events = []; t0 = time.monotonic()
    a = torch.randn(size, size, device=dev); b = torch.randn(size, size, device=dev)
    while time.monotonic() - t0 < secs and not _HOT["s"].hot:
        _ = a @ b
        if dev == "cuda": torch.cuda.synchronize()
        events.append({"t": time.monotonic(), "burst": 1})
        _HOT["s"].tick()                            # inline telemetry sample
        time.sleep(max(0.0, 1.0/cadence - 0.002))
    return events

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=60)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--period", type=float, default=0.02, help="telemetry sample period (s)")
    ap.add_argument("--yoke_size", type=int, default=1024)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{HOST}] Phase0 dev={dev} model={a.model} secs={a.secs} period={a.period}s "
          f"THERM_ABORT={THERM_ABORT}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    lm = AutoModelForCausalLM.from_pretrained(a.model).to(dev).eval()
    for p in lm.parameters(): p.requires_grad = False
    prompt = ("The history of the city is long and its streets remember many things. "
              "On a quiet morning the philosopher considered the nature of mind and machine, and wrote:")

    sampler = Sampler(period=a.period); _HOT["s"] = sampler
    INLINE = os.environ.get("INLINE", "1") == "1"   # inline sampling (default) avoids ROCm thread deadlock
    if not INLINE: sampler.start()
    time.sleep(1.0)
    marks = {}; allev = {}
    burst = float(os.environ.get("BURST_SECS", "8"))       # heating burst length
    target_samples = int(os.environ.get("TARGET_SAMPLES", "1200"))
    max_bursts = int(os.environ.get("MAX_BURSTS", "10"))

    def run_condition(name, fn_burst):
        """Accumulate ~target_samples across short bursts, cooling between, to stay thermal-safe.
        Each burst window is recorded as (i0,i1) so analysis only uses in-burst samples."""
        windows = []; events = []
        for bi in range(max_bursts):
            wait_cool()                       # cool FIRST
            sampler.hot = False               # then arm (fixes C3-starve bug)
            i0 = len(sampler.rows); t0 = time.monotonic()
            ev = fn_burst(burst)
            i1 = len(sampler.rows); t1 = time.monotonic()
            events.extend(ev)
            windows.append({"i0": i0, "i1": i1, "t0": t0, "t1": t1, "hot": bool(sampler.hot)})
            got = sum(w["i1"]-w["i0"] for w in windows)
            print(f"  {name} burst {bi}: +{i1-i0} samp (tot {got}), "
                  f"{t1-t0:.1f}s zone0={zone0():.0f}C hot={sampler.hot}", flush=True)
            if got >= target_samples: break
        return events, windows

    seq = [("C0_IDLE", lambda s: run_idle(sampler, s)),
           ("C1_SELF", lambda s: run_self(lm, tok, dev, s, prompt, scramble=False)),
           ("C3_SWAP", lambda s: run_self(lm, tok, dev, s, prompt, scramble=True)),
           ("C2_YOKED", lambda s: run_yoked(s, a.yoke_size, cadence=999.0))]
    p = OUT / f"phase0_{HOST}.npz"
    def dump():
        """Write npz from current sampler+events state. Called after EACH condition so a killed
        run (ssh drop mid-run) still leaves valid partial data to fetch (infra-robustness fix)."""
        t, chans = sampler.matrix()
        save = {"host": HOST, "mono_t": t, "period": a.period, "model": a.model,
                "channel_dims": json.dumps({k: int(v.shape[1]) if v.ndim == 2 else 0 for k, v in chans.items()}),
                "marks": json.dumps(marks)}
        for k, M in chans.items():
            save[f"chan_{k}"] = M
        for name, ev in allev.items():
            if ev and "tok" in ev[0]:
                save[f"ev_{name}_t"] = np.array([e["t"] for e in ev])
                save[f"ev_{name}_tok"] = np.array([e["tok"] for e in ev])
                save[f"ev_{name}_entropy"] = np.array([e["entropy"] for e in ev])
                save[f"ev_{name}_top1"] = np.array([e["top1"] for e in ev])
                save[f"ev_{name}_logprob"] = np.array([e["logprob"] for e in ev])
            elif ev:
                save[f"ev_{name}_t"] = np.array([e["t"] for e in ev])
        tmp = p.with_name(p.stem + ".tmp.npz")  # must end .npz (savez_compressed appends .npz otherwise)
        np.savez_compressed(tmp, **save); os.replace(tmp, p)
        return len(t)

    for name, fn in seq:
        ev, windows = run_condition(name, fn)
        allev[name] = ev
        marks[name] = {"windows": windows, "n_events": len(ev)}
        n = dump()
        print(f"  [{name}] partial-saved npz ({n} samples so far)", flush=True)
    sampler.stop = True; time.sleep(0.2)
    n = dump()
    print(f"[{HOST}] saved {p}  ({sum(len(v) for v in allev.values())} events, {n} telem samples)", flush=True)

if __name__ == "__main__":
    main()
