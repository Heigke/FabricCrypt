"""H7 DVFS-SWEPT u·v die-specificity probe — the controlled re-test of the "analog-unique" dream.

The death verdict ("u·v is board-set, not die") was reached with DVFS FREE (powersave + boost on) — the #1
confound, never controlled. This probe does it right: pin governor=performance + boost off, then SWEEP the CPU
operating point (scaling_max=min freq) across K setpoints. At each pinned point, temp-gated, drive u·v
(GPU u-burst × CPU v-burst) and fit the coupling coefficient Auv from a telemetry regression on [1,u,v,uv],
PLUS a board reference (v off) to subtract common-mode. Output = the die's u·v(f) curve. The die-specific signal
lives in the OPERATING-POINT DEPENDENCE (curvature across f), where board common-mode cancels and per-die
Vth/leakage curvature survives. Compare ikaros vs daedalus; test if the curve carries die-info BEYOND static CPPC.

SAFETY: saves & RESTORES original DVFS state in finally (never leaves the machine pinned). Pinning lower freq
REDUCES heat; still temp-gated. Root required (writes scaling_*_freq, boost). Env: FREQS_MHZ, RUNTAG, SMOKE=1.
"""
from __future__ import annotations
import os, sys, time, json, socket, glob, math
from pathlib import Path
import numpy as np

HOST = socket.gethostname()
OUT = Path(__file__).resolve().parents[2] / "results/IDENTITY_H7_2026-06-09"
ZONE_T = Path("/sys/class/thermal/thermal_zone0/temp")
NCPU = os.cpu_count() or 1
GOV = [f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_governor" for c in range(NCPU)]
SMAX = [f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_max_freq" for c in range(NCPU)]
SMIN = [f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_min_freq" for c in range(NCPU)]
BOOST = "/sys/devices/system/cpu/cpufreq/boost"
FREQS_MHZ = [int(x) for x in os.environ.get("FREQS_MHZ", "1400,2000,2600,3200,3800").split(",")]
RUNTAG = os.environ.get("RUNTAG", "r1")
SMOKE = os.environ.get("SMOKE", "0") == "1"
SOAK_HI = 68.0; HOT = 80.0; HARD = 86.0
sys.path.insert(0, str(Path(__file__).parent))


def temp_c():
    try: return int(ZONE_T.read_text())/1000.0
    except Exception: return 0.0


def wr(path, val):
    try:
        Path(path).write_text(str(val)); return True
    except Exception as e:
        print(f"  WRITE FAIL {path}={val}: {e}", flush=True); return False


def rd(path, d=""):
    try: return Path(path).read_text().strip()
    except Exception: return d


def save_state():
    return {"gov": [rd(p) for p in GOV], "smax": [rd(p) for p in SMAX],
            "smin": [rd(p) for p in SMIN], "boost": rd(BOOST)}


def restore_state(st):
    print("[restore] putting DVFS back...", flush=True)
    if st["boost"]: wr(BOOST, st["boost"])
    for c in range(NCPU):
        if st["smax"][c]: wr(SMAX[c], st["smax"][c])
        if st["smin"][c]: wr(SMIN[c], st["smin"][c])
        if st["gov"][c]: wr(GOV[c], st["gov"][c])
    print(f"[restore] done. gov[0]={rd(GOV[0])} smax[0]={rd(SMAX[0])} boost={rd(BOOST)}", flush=True)


def pin_freq(khz):
    for c in range(NCPU):
        wr(GOV[c], "performance"); wr(SMAX[c], khz); wr(SMIN[c], khz)


def soak():
    t0 = time.time()
    while time.time()-t0 < 90:
        if temp_c() <= SOAK_HI: break
        time.sleep(1.0)


def measure_uv(st_sub, drive_on, nseg=240, fs=80.0):
    """Drive u (GPU) always; v (CPU) only if drive_on. Return per-step telemetry + u,v logs."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = 1.0/fs
    Y = np.zeros((nseg, 10), np.float32); U = np.zeros(nseg); V = np.zeros(nseg)
    rng = np.random.default_rng(0)
    gA = measure_uv._gA; gB = measure_uv._gB; cA = measure_uv._cA; cB = measure_uv._cB
    for k in range(nseg):
        s0 = time.time()
        u = 0.5*(1+math.sin(2*math.pi*3.0*k*dt))      # GPU drive envelope (tone f=3Hz)
        v = (0.5*(1+math.sin(2*math.pi*5.0*k*dt))) if drive_on else 0.0  # CPU drive (tone f=5Hz)
        gdur = 0.0015*u
        if gdur > 1e-4:
            while time.time()-s0 < gdur: gA = (gA @ gB).tanh()*0.5+0.5
        if v > 0:
            sc = time.time()
            while time.time()-sc < 0.0015*v: cA = np.tanh(cA @ cB)*0.5+0.5
        if dev == "cuda": torch.cuda.synchronize()
        Y[k] = st_sub.latest_window(length=2).reshape(-1, 10)[:2].mean(0)
        U[k] = u; V[k] = v
        rest = dt-(time.time()-s0)
        if rest > 0: time.sleep(rest)
        if k % 20 == 0 and temp_c() > HARD:
            while temp_c() > SOAK_HI: time.sleep(1.0)
    return Y, U, V


def fit_auv(Y, U, V, med, mad):
    Yn = (Y-med)/mad
    uc = U-U.mean(); vc = V-V.mean(); uvc = uc*vc
    A = np.stack([np.ones(len(U)), uc, vc, uvc], 1)
    auv = np.zeros(10)
    for c in range(10):
        b, *_ = np.linalg.lstsq(A, Yn[:, c], rcond=None); auv[c] = b[3]
    return auv


def main():
    if SMOKE:
        st = save_state()
        print(f"[SMOKE] saved state gov[0]={st['gov'][0]} smax[0]={st['smax'][0]} boost={st['boost']}", flush=True)
        try:
            wr(BOOST, "0"); pin_freq(FREQS_MHZ[0]*1000); time.sleep(1.0)
            print(f"[SMOKE] pinned {FREQS_MHZ[0]}MHz -> cur={rd('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq')} "
                  f"gov={rd(GOV[0])} smax={rd(SMAX[0])} boost={rd(BOOST)} temp={temp_c():.0f}C", flush=True)
        finally:
            restore_state(st)
        print("[SMOKE] OK — set/restore verified", flush=True); return

    import torch
    measure_uv._gA = torch.randn(1024, 1024, device="cuda" if torch.cuda.is_available() else "cpu")
    measure_uv._gB = torch.randn(1024, 1024, device="cuda" if torch.cuda.is_available() else "cpu")
    measure_uv._cA = np.random.default_rng(1).standard_normal((768, 768))
    measure_uv._cB = np.random.default_rng(2).standard_normal((768, 768))
    from substrate_realtime_v3 import SubstrateStateV3
    st_dvfs = save_state()
    try:
        wr(BOOST, "0")
        sub = SubstrateStateV3(hz_target=500); sub.start(); time.sleep(6.0)
        pool = np.array([sub.latest_window(length=64).reshape(-1, 10) for _ in range(40)]).reshape(-1, 10)
        med = np.median(pool, 0); mad = np.median(np.abs(pool-med), 0)*1.4826
        mad = np.maximum(mad, np.median(mad)*0.25 + 1e-6)
        print(f"[{HOST}] DVFS-sweep u·v freqs={FREQS_MHZ}MHz tag={RUNTAG} temp={temp_c():.0f}C", flush=True)
        curve = np.zeros((len(FREQS_MHZ), 10))      # die u·v(f) coupling
        ref = np.zeros((len(FREQS_MHZ), 10))        # board reference (v off)
        cur_freq = np.zeros(len(FREQS_MHZ))
        for i, fmhz in enumerate(FREQS_MHZ):
            pin_freq(fmhz*1000); time.sleep(0.5); soak()
            cur_freq[i] = float(rd("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq") or 0)/1000.0
            Y1, U1, V1 = measure_uv(sub, True)      # u·v on
            auv = fit_auv(Y1, U1, V1, med, mad)
            Y0, U0, V0 = measure_uv(sub, False)     # board ref (v off)
            aref = fit_auv(Y0, U0, V0, med, mad)
            curve[i] = auv; ref[i] = aref
            print(f"  f={fmhz}MHz (cur~{cur_freq[i]:.0f}) |Auv|={np.abs(auv).mean():.3f} "
                  f"|Aref|={np.abs(aref).mean():.3f} temp={temp_c():.0f}C", flush=True)
        sub.stop()
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT/f"dvfs_uv_{HOST}_{RUNTAG}.npz", curve=curve, ref=ref,
                            freqs=np.array(FREQS_MHZ), cur_freq=cur_freq)
        diff = curve - ref   # differential: remove board common-mode (v-off baseline)
        print(f"  curve-vs-freq |diff| per setpoint: {np.round(np.abs(diff).mean(1),3)}", flush=True)
        print(f">>> saved dvfs_uv_{HOST}_{RUNTAG}.npz", flush=True)
    finally:
        try: sub.stop()
        except Exception: pass
        restore_state(st_dvfs)


if __name__ == "__main__":
    main()
