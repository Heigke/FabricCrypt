"""H7 two-tone INTERMODULATION probe — the cleanest falsifiable die-nonlinearity test.

A purely LINEAR transfer function (load -> telemetry) produces ZERO intermodulation: drive two tones
f1,f2 and the output has only f1,f2 (+ their own harmonics from a static map). Genuine DYNAMICAL
nonlinearity (thermal lag × leakage(T), Vdroop, throttle) MIXES them -> new lines at f2-f1, f1+f2,
2f1-f2, 2f2-f1. Those cross-frequencies are MANUFACTURED BY THE SUBSTRATE, not in our command stream
-> the strongest "die did it, not self-computable" argument. In-band at 500Hz (slow tones down-convert).

Decisive control: compare measured IMD to the IMD a STATIC monotone map of the commanded load predicts
(polynomial fit of channel-mean vs commanded duty, applied to the drive). EXCESS IMD over the static map
= genuine DYNAMICAL nonlinearity (memory), the part a rank-limited linear adapter cannot synthesize.

Single GPU duty-modulated load (thermally gentle vs the dual-load probe). In-loop guard. Root (substrate).
"""
from __future__ import annotations
import sys, json, time, socket
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent))
from substrate_realtime_v3 import SubstrateStateV3

HOST = socket.gethostname()
OUT = Path(__file__).resolve().parents[2] / "results/IDENTITY_H7_2026-06-09"
ZONE = Path("/sys/class/thermal/thermal_zone0/temp")
N_CH = 10
F1, F2 = 31.0, 47.0          # incommensurate slow tones (<< 100Hz drive Nyquist)
FS = 200.0                    # drive/sample slots per second (5ms)
DUR = 70.0                    # seconds
BASE, AMP = 0.45, 0.22        # duty = BASE + AMP*(sin f1 + sin f2)/... , kept modest for thermal safety
SEED = 0


def temp_c():
    try: return int(ZONE.read_text()) / 1000.0
    except Exception: return 0.0


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gA = torch.randn(1024, 1024, device=dev); gB = torch.randn(1024, 1024, device=dev)
    st = SubstrateStateV3(hz_target=500); st.start()
    print(f"[{HOST}] two-tone IMD f1={F1} f2={F2}Hz dev={dev} (temp {temp_c():.0f}C) warmup 6s...", flush=True)
    time.sleep(6.0)
    pool = np.array([st.latest_window(length=64).reshape(-1, N_CH) for _ in range(40)]).reshape(-1, N_CH)
    med = np.median(pool, 0); mad = np.median(np.abs(pool - med), 0) * 1.4826 + 1e-9

    nslot = int(DUR * FS); slot = 1.0 / FS
    duty = np.zeros(nslot); S = np.zeros((nslot, N_CH), np.float32)
    t0 = time.time()
    for k in range(nslot):
        t = k / FS
        d = BASE + AMP * (np.sin(2*np.pi*F1*t) + np.sin(2*np.pi*F2*t))
        d = float(np.clip(d, 0.0, 1.0)); duty[k] = d
        s0 = time.time(); busy = d * slot
        while time.time() - s0 < busy:
            gA = (gA @ gB).tanh() * 0.5 + 0.5
        if dev == "cuda": torch.cuda.synchronize()
        rest = slot - (time.time() - s0)
        if rest > 0: time.sleep(rest)
        S[k] = st.latest_window(length=4).mean(0)
        if k % 100 == 0:
            tc = temp_c()
            if tc > 80.0:
                print(f"  [guard] {tc:.0f}C cooling", flush=True)
                while temp_c() > 58.0: time.sleep(1.0); t0 += 1.0
            if k % 2000 == 0: print(f"  slot {k}/{nslot} temp={tc:.0f}C ({time.time()-t0:.0f}s)", flush=True)
    st.stop()
    Sn = np.tanh((S - med) / mad / 8.0)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"intermod_raw_{HOST}.npz", duty=duty, S=S, Sn=Sn, med=med, mad=mad, fs=FS)

    # frequency analysis
    win = np.hanning(nslot)
    freqs = np.fft.rfftfreq(nslot, d=slot)
    def amp_at(x, f):
        i = int(round(f * nslot * slot)); i = max(1, min(i, len(x)-1))
        X = np.abs(np.fft.rfft((x - x.mean()) * win))
        # local peak +-1 bin
        return float(X[max(1,i-1):i+2].max())
    fund = [F1, F2]
    imd = {"f2-f1": F2-F1, "f1+f2": F1+F2, "2f1-f2": 2*F1-F2, "2f2-f1": 2*F2-F1}
    # noise floor: median amplitude away from any tone
    Xref = np.abs(np.fft.rfft((Sn[:, 5] - Sn[:,5].mean()) * win))

    res = {}
    for ch in range(N_CH):
        x = Sn[:, ch]
        a_f = np.mean([amp_at(x, f) for f in fund]) + 1e-9
        a_imd = {k: amp_at(x, v) for k, v in imd.items()}
        Xn = np.abs(np.fft.rfft((x - x.mean()) * win)); floor = np.median(Xn[1:]) + 1e-9
        imd_over_floor = {k: v/floor for k, v in a_imd.items()}
        res[f"ch{ch}"] = {"fund_amp": a_f, "imd_amp": a_imd,
                          "imd_over_noisefloor": imd_over_floor,
                          "max_imd_over_floor": float(max(imd_over_floor.values())),
                          "imd_over_fund": float(max(a_imd.values())/a_f)}

    # DECISIVE control on best channel: static-map IMD vs measured IMD
    best = max(res, key=lambda k: res[k]["max_imd_over_floor"]); bch = int(best[2:])
    x = Sn[:, bch]
    # static monotone map: fit channel ~ poly(duty) deg5 (instantaneous, no memory), predict, take its IMD
    cpoly = np.polyfit(duty, x, 5); x_static = np.polyval(cpoly, duty)
    def imd_sum(sig):
        return float(np.sqrt(sum(amp_at(sig, v)**2 for v in imd.values())))
    meas_imd = imd_sum(x); static_imd = imd_sum(x_static)
    excess = meas_imd / (static_imd + 1e-9)

    verdict = ("DYNAMICAL NONLINEARITY (IMD beyond static map)" if (res[best]["max_imd_over_floor"] > 6
               and excess > 1.5) else
               "static/weak — IMD explained by instantaneous load map" if res[best]["max_imd_over_floor"] > 6
               else "no significant IMD (linear)")
    out = {"host": HOST, "f1": F1, "f2": F2, "fs": FS, "dur": DUR, "per_channel": res,
           "best_channel": best, "best_max_imd_over_floor": res[best]["max_imd_over_floor"],
           "static_control": {"measured_imd": meas_imd, "static_map_imd": static_imd,
                              "excess_over_static": excess},
           "verdict": verdict}
    def jf(o):
        if isinstance(o, dict): return {k: jf(v) for k,v in o.items()}
        if isinstance(o, (list,tuple)): return [jf(v) for v in o]
        if isinstance(o,(np.floating,np.integer)): return float(o)
        return o
    (OUT / f"intermod_{HOST}.json").write_text(json.dumps(jf(out), indent=2))
    print("\n  per-channel IMD strength (max IMD line / noise floor):", flush=True)
    for ch in range(N_CH):
        d = res[f"ch{ch}"]
        print(f"   ch{ch}: maxIMD/floor={d['max_imd_over_floor']:7.1f}  IMD/fund={d['imd_over_fund']:.3f}", flush=True)
    print(f"\n  best ch {best}: IMD/floor={res[best]['max_imd_over_floor']:.1f}", flush=True)
    print(f"  STATIC-MAP CONTROL: measured IMD={meas_imd:.4f} vs static-map IMD={static_imd:.4f} -> excess={excess:.2f}x", flush=True)
    print(f"  >>> {verdict}", flush=True)


if __name__ == "__main__":
    main()
