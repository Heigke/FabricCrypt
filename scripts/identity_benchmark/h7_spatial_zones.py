"""H7 SPATIAL-ZONE PDN coupling matrix — the full O106 protocol for die-specific computation.

Coefficient route was falsified (operating-point dominates the scalar u·v gain). Remaining route (GPT-5 0.55-
0.65 + web agent #1): the die-specific quantity is the SPATIAL PDN impedance pattern Z(die) — which on-die
ZONE couples to which SENSOR — set by package/PDN/EM fabrication mismatch, NOT by operating point. We probe it
with the three controls the literature demands:
  1. MULTI-ZONE drive: CPU v-bursts PINNED to distinct physical cores (os.sched_setaffinity) = spatial zones;
     GPU u-bursts shared. Per zone z we get the u·v coupling into every sensor channel -> matrix M[z, ch].
  2. MATCHED-TEMPERATURE: soak to a fixed junction band before the run + low duty + per-step temp logged so
     analysis can keep only steps inside the matched band (kills the thermal confound that sank cross-die).
  3. RATIO/DIFFERENTIAL readout: per (zone,channel) C_uv = A_uv/sqrt(|A_u A_v|) cancels common-mode temp;
     the die SIGNATURE = the normalized coupling matrix M (which zone lights which sensor, and how the u·v
     mixing distributes spatially). Inter-die vs intra-die distance of M (at matched temp) is the verdict.
Run: ikaros twice (intra-die, identical drive) + daedalus once, all soaked to the SAME temp band. Saves
spatial_zones_raw_{HOST}_{RUNTAG}.npz (u, vz per zone, T per step, temp per step, zone order). Root.
Env: ZONES="0,3,6,9" RUNTAG="r1" SOAK_LO=48 SOAK_HI=52 override defaults.
"""
from __future__ import annotations
import os, sys, json, time, socket
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent))
from substrate_realtime_v3 import SubstrateStateV3

HOST = socket.gethostname()
OUT = Path(__file__).resolve().parents[2] / "results/IDENTITY_H7_2026-06-09"
ZONE_T = Path("/sys/class/thermal/thermal_zone0/temp")
N_CH = 10
ZONES = [int(z) for z in os.environ.get("ZONES", "0,3,6,9").split(",")]
RUNTAG = os.environ.get("RUNTAG", "r1")
STEPS_PER_ZONE = 520
NTAP = 12
GPU_BURST_MS = 0.004
CPU_BURST_MS = 0.006
CPU_MAT = 1024
STEP_S = 0.045
SOAK_LO = float(os.environ.get("SOAK_LO", "48"))
SOAK_HI = float(os.environ.get("SOAK_HI", "52"))
HOT = 68.0
SEED_U = 0
SEED_VBASE = 9000


def temp_c():
    try: return int(ZONE_T.read_text())/1000.0
    except Exception: return 0.0


def soak(target_lo, target_hi):
    """idle-wait until die temp is inside the matched band (cool down or warm up by waiting)."""
    print(f"  soaking to [{target_lo},{target_hi}]C (now {temp_c():.0f})...", flush=True)
    t0 = time.time()
    while time.time()-t0 < 180:
        t = temp_c()
        if t <= target_hi: break
        time.sleep(1.0)
    print(f"  soak done temp={temp_c():.0f}C", flush=True)


def collect():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gA = torch.randn(2048, 2048, device=dev); gB = torch.randn(2048, 2048, device=dev)
    cA = np.random.default_rng(1).standard_normal((CPU_MAT, CPU_MAT))
    cB = np.random.default_rng(2).standard_normal((CPU_MAT, CPU_MAT))
    st = SubstrateStateV3(hz_target=500); st.start(); time.sleep(6.0)
    pool = np.array([st.latest_window(length=64).reshape(-1, N_CH) for _ in range(40)]).reshape(-1, N_CH)
    med = np.median(pool, 0); mad = np.median(np.abs(pool-med), 0)*1.4826 + 1e-9
    L = STEPS_PER_ZONE * len(ZONES)
    u = np.random.default_rng(SEED_U).integers(0, 2, size=L)
    T = np.zeros((L, NTAP, N_CH), np.float32); temps = np.zeros(L, np.float32)
    zone_of = np.zeros(L, np.int32); vz = np.zeros(L, np.int8)
    t0 = time.time(); idx = 0
    for zi, core in enumerate(ZONES):
        try: os.sched_setaffinity(0, {core})
        except Exception as e: print(f"  affinity core {core} failed: {e}", flush=True)
        soak(SOAK_LO, SOAK_HI)
        vstream = np.random.default_rng(SEED_VBASE+core).integers(0, 2, size=STEPS_PER_ZONE)
        for k in range(STEPS_PER_ZONE):
            s0 = time.time()
            if u[idx]:
                gA = (gA @ gB).tanh()*0.5 + 0.5
            if vstream[k]:
                sc = time.time()
                while time.time()-sc < CPU_BURST_MS:
                    cA = np.tanh(cA @ cB)*0.5 + 0.5
            if u[idx] and dev == "cuda":
                while time.time()-s0 < GPU_BURST_MS:
                    gA = (gA @ gB).tanh()*0.5 + 0.5
                torch.cuda.synchronize()
            time.sleep(0.004)
            T[idx] = st.latest_window(length=NTAP).reshape(-1, N_CH)[:NTAP]
            tc = temp_c(); temps[idx] = tc; zone_of[idx] = core; vz[idx] = vstream[k]
            rest = STEP_S - (time.time()-s0)
            if rest > 0: time.sleep(rest)
            if tc > HOT:
                while temp_c() > SOAK_HI: time.sleep(1.0)
            if k % 200 == 0: print(f"  zone {core} step {k}/{STEPS_PER_ZONE} temp={tc:.0f}C ({time.time()-t0:.0f}s)", flush=True)
            idx += 1
    st.stop()
    Tn = np.tanh((T-med)/mad/8.0)
    return u, vz, zone_of, Tn, temps


def main():
    print(f"[{HOST}] SPATIAL-ZONE coupling probe zones={ZONES} tag={RUNTAG} temp {temp_c():.0f}C", flush=True)
    u, vz, zone_of, Tn, temps = collect()
    OUT.mkdir(parents=True, exist_ok=True)
    fn = OUT/f"spatial_zones_raw_{HOST}_{RUNTAG}.npz"
    np.savez_compressed(fn, u=u, vz=vz, zone_of=zone_of, Tn=Tn, temps=temps, zones=np.array(ZONES))
    # quick inline coupling matrix (per zone x channel normalized C_uv), matched-temp band only
    flat = Tn.reshape(len(u), -1); taps = flat.shape[1]//N_CH
    perstep = flat.reshape(len(u), taps, N_CH).mean(1)
    band = (temps >= SOAK_LO-1) & (temps <= SOAK_HI+3)
    print(f"  matched-temp band keeps {band.sum()}/{len(u)} steps; temp range {temps.min():.0f}-{temps.max():.0f}C", flush=True)
    M = np.zeros((len(ZONES), N_CH))
    for zi, core in enumerate(ZONES):
        m = (zone_of == core) & band
        if m.sum() < 50: continue
        uu = u[m].astype(float); vv = vz[m].astype(float)
        uc = uu-uu.mean(); vc = vv-vv.mean(); uvc = uc*vc
        A = np.stack([np.ones(m.sum()), uc, vc, uvc], 1)
        for c in range(N_CH):
            b, *_ = np.linalg.lstsq(A, perstep[m, c], rcond=None)
            Au, Av, Auv = b[1], b[2], b[3]
            M[zi, c] = Auv/(np.sqrt(abs(Au*Av))+1e-9)
    out = {"host": HOST, "runtag": RUNTAG, "zones": ZONES, "kept_steps": int(band.sum()),
           "temp_min": float(temps.min()), "temp_max": float(temps.max()),
           "coupling_matrix_Cuv": M.tolist()}
    (OUT/f"spatial_zones_{HOST}_{RUNTAG}.json").write_text(json.dumps(out, indent=2))
    print("  coupling matrix M[zone,channel] (temp-compensated C_uv):", flush=True)
    for zi, core in enumerate(ZONES):
        print(f"   core{core}: [{', '.join(f'{x:+.2f}' for x in M[zi])}]", flush=True)
    print(f">>> saved {fn.name} + json", flush=True)


if __name__ == "__main__":
    main()
