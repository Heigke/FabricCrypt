"""H7 LOCK-IN spatial fingerprint — frequency-domain u·v intermod, the O107-unanimous strengthening.

Time-domain scalar regression capped same-die similarity at 0.76. All 4 oracles + web rank frequency-domain
lock-in #1: drive u(GPU) at tone f1 and v(CPU,pinned-core) at tone f2; the bilinear u·v mixing appears at the
INTERMOD frequencies f1±f2 (linear u,v terms live only at f1,f2, so f1±f2 is the pure 2nd-order term — no
subtraction needed). Demod telemetry by I/Q lock-in at f1±f2 → complex amp+phase per (zone,tonepair,channel)
= a rich, temperature-robust die signature (PDN poles/zeros). Matched-temp soak + low duty (sinusoid-modulated
short bursts) keep it thermally safe. Run K times per die (RUNTAG r1..rK) → template-average + intra/inter in
the analysis. Env: ZONES, TONEPAIRS="3,5;5,8;8,13", RUNTAG, SOAK_LO/HI. Root.
"""
from __future__ import annotations
import os, sys, json, time, socket, math
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent))
from substrate_realtime_v3 import SubstrateStateV3

HOST = socket.gethostname()
OUT = Path(__file__).resolve().parents[2] / "results/IDENTITY_H7_2026-06-09"
ZONE_T = Path("/sys/class/thermal/thermal_zone0/temp")
N_CH = 10
ZONES = [int(z) for z in os.environ.get("ZONES", "0,2,4,6,8,10,12,14").split(",")]
TONEPAIRS = [tuple(float(x) for x in p.split(",")) for p in os.environ.get("TONEPAIRS", "3,5;5,8;8,13").split(";")]
RUNTAG = os.environ.get("RUNTAG", "r1")
FS_CTRL = 80.0           # control loop Hz (telemetry sampled each step); Nyquist 40Hz > all tones
SEG_S = 6.0              # seconds per (zone, tonepair) segment
GPU_BMAX = float(os.environ.get("GPU_BMAX", "0.0012"))  # max GPU burst s/step (low duty -> stay below throttle)
CPU_BMAX = float(os.environ.get("CPU_BMAX", "0.0012"))
CPU_MAT = 1024
GPU_MAT = int(os.environ.get("GPU_MAT", "1024"))        # smaller GPU matmul: less heat/step, avoid throttle homogenization
SOAK_LO = float(os.environ.get("SOAK_LO", "60")); SOAK_HI = float(os.environ.get("SOAK_HI", "68"))  # ACHIEVABLE band
HOT = float(os.environ.get("HOT", "78"))                # hard per-step ceiling: pause to SOAK_LO if exceeded
HARD = float(os.environ.get("HARD", "85"))              # absolute abort-segment ceiling (never near 99C trip)


def temp_c():
    try: return int(ZONE_T.read_text())/1000.0
    except Exception: return 0.0


def soak():
    t0 = time.time()
    while time.time()-t0 < 150:
        if temp_c() <= SOAK_HI: break
        time.sleep(1.0)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gA = torch.randn(GPU_MAT, GPU_MAT, device=dev); gB = torch.randn(GPU_MAT, GPU_MAT, device=dev)
    cA = np.random.default_rng(1).standard_normal((CPU_MAT, CPU_MAT))
    cB = np.random.default_rng(2).standard_normal((CPU_MAT, CPU_MAT))
    st = SubstrateStateV3(hz_target=500); st.start(); time.sleep(6.0)
    pool = np.array([st.latest_window(length=64).reshape(-1, N_CH) for _ in range(40)]).reshape(-1, N_CH)
    med = np.median(pool, 0); mad = np.median(np.abs(pool-med), 0)*1.4826
    mad = np.maximum(mad, np.median(mad)*0.25 + 1e-6)   # robust floor: stop near-constant channels exploding
    nseg = int(SEG_S*FS_CTRL); dt = 1.0/FS_CTRL
    print(f"[{HOST}] LOCK-IN zones={ZONES} pairs={TONEPAIRS} tag={RUNTAG} nseg={nseg} temp {temp_c():.0f}C", flush=True)

    # feature tensor: [zone, pair, channel, {f1+f2,f1-f2}, {re,im}]
    feat = np.zeros((len(ZONES), len(TONEPAIRS), N_CH, 2, 2), np.float32)
    raw_segs = {}
    t_global = time.time()
    for zi, core in enumerate(ZONES):
        try: os.sched_setaffinity(0, {core})
        except Exception as e: print(f"  affinity {core} fail {e}", flush=True)
        for pj, (f1, f2) in enumerate(TONEPAIRS):
            soak()
            Y = np.zeros((nseg, N_CH), np.float32); tt = np.zeros(nseg); Tk = np.zeros(nseg)
            for k in range(nseg):
                s0 = time.time(); tsec = k*dt
                # SAFETY: never let temp approach the 99C trip — pause to SOAK_LO before any compute
                if temp_c() > HARD:
                    print(f"    !! temp {temp_c():.0f}C > HARD {HARD} — pause to {SOAK_LO}", flush=True)
                    while temp_c() > SOAK_LO: time.sleep(1.0)
                gdur = GPU_BMAX*0.5*(1+math.sin(2*math.pi*f1*tsec))   # sinusoid-modulated burst length
                cdur = CPU_BMAX*0.5*(1+math.sin(2*math.pi*f2*tsec))
                if gdur > 1e-4:
                    while time.time()-s0 < gdur: gA = (gA @ gB).tanh()*0.5+0.5
                if cdur > 1e-4:
                    sc = time.time()
                    while time.time()-sc < cdur: cA = np.tanh(cA @ cB)*0.5+0.5
                if dev == "cuda": torch.cuda.synchronize()
                Y[k] = st.latest_window(length=2).reshape(-1, N_CH)[:2].mean(0)
                tt[k] = tsec; Tk[k] = temp_c()
                rest = dt-(time.time()-s0)
                if rest > 0: time.sleep(rest)
                if k % 20 == 0 and temp_c() > HOT:
                    while temp_c() > SOAK_HI: time.sleep(1.0)
            Yn = (Y-med)/mad
            raw_segs[f"T_z{core}_p{pj}"] = Tk.astype(np.float32)  # per-step temp -> verify matched-temp offline
            # I/Q lock-in at intermod freqs f1+f2 and f1-f2
            for fi, f in enumerate([f1+f2, abs(f1-f2)]):
                c = np.exp(-1j*2*np.pi*f*tt)
                comp = (Yn * c[:, None]).mean(0)   # complex per channel
                feat[zi, pj, :, fi, 0] = comp.real; feat[zi, pj, :, fi, 1] = comp.imag
            raw_segs[f"z{core}_p{pj}"] = Yn.astype(np.float32)
            print(f"  zone{core} pair({f1},{f2}) done Tmean={Tk.mean():.1f}C Tmax={Tk.max():.0f}C "
                  f"|seg|={np.abs(Yn).mean():.3f} ({time.time()-t_global:.0f}s)", flush=True)
    st.stop()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT/f"lockin_raw_{HOST}_{RUNTAG}.npz", feat=feat,
                        zones=np.array(ZONES), tonepairs=np.array(TONEPAIRS), **raw_segs)
    fv = feat.ravel()
    print(f"  feature vec dim={fv.size}  |feat| mean={np.abs(feat).mean():.4f}", flush=True)
    print(f">>> saved lockin_raw_{HOST}_{RUNTAG}.npz", flush=True)


if __name__ == "__main__":
    main()
