"""H7 MIXING STRONG — push the v-drive HARD to lift the u·v term, full rigorous within-die verdict in one run.

Cross-die showed the die genuinely computes u·v but the signal was weak (d_v=0.59, soft CPU bursts) and the
mixing looked generic. User: strengthen the mixing FIRST (lift the u·v signal) before any more cross-die.
Levers vs h7_cross_die_mixing: heavier CPU contention (1024² vs 768², longer v burst), max GPU/CPU temporal
OVERLAP (the contention IS the multiplier), and we DEMODULATE the settling transient harder. Inline analysis:
  - drive landing d_u, d_v (goal: lift d_v well above 0.59)
  - die(full)-linear XOR(u,v) vs 300-shuffle NULL -> p-value
  - physical u·v partial-R² per channel (the mechanism strength)
  - necessity controls: u-only(poly) [must be chance, lacks v]; u&v-LINEAR [must be chance, no product]
  - explicit u·v ceiling
A STRONGER result = higher die XOR, higher uv_partialR2_max, bigger gap die−(u&v-linear). THERMAL: dual-engine
is the hot path; tight guard (check every 4 steps, pause@70→cool52). Root. Writes mixing_strong_{HOST}.json.
"""
from __future__ import annotations
import sys, json, time, socket, itertools
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent))
from substrate_realtime_v3 import SubstrateStateV3

HOST = socket.gethostname()
OUT = Path(__file__).resolve().parents[2] / "results/IDENTITY_H7_2026-06-09"
ZONE = Path("/sys/class/thermal/thermal_zone0/temp")
N_CH = 10; L = 2200; WASHOUT = 150
GPU_BURST_MS = 0.004
CPU_BURST_MS = 0.007          # was 0.0035 — longer, harder CPU draw
CPU_MAT = 1024                # was 768 — heavier current edge
STEP_S = 0.040; NTAP = 12; SEED_U = 0; SEED_V = 12345
HOT = 70.0; COOL = 52.0


def temp_c():
    try: return int(ZONE.read_text())/1000.0
    except Exception: return 0.0


def collect(u, v):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gA = torch.randn(2048, 2048, device=dev); gB = torch.randn(2048, 2048, device=dev)
    cA = np.random.default_rng(1).standard_normal((CPU_MAT, CPU_MAT))
    cB = np.random.default_rng(2).standard_normal((CPU_MAT, CPU_MAT))
    st = SubstrateStateV3(hz_target=500); st.start(); time.sleep(6.0)
    pool = np.array([st.latest_window(length=64).reshape(-1, N_CH) for _ in range(40)]).reshape(-1, N_CH)
    med = np.median(pool, 0); mad = np.median(np.abs(pool - med), 0)*1.4826 + 1e-9
    T = np.zeros((L, NTAP, N_CH), np.float32); t0 = time.time()
    for t in range(L):
        s0 = time.time()
        # launch GPU (async) and run CPU CONCURRENTLY so they contend for the same power envelope
        if u[t]:
            gA = (gA @ gB).tanh()*0.5 + 0.5          # one big async GPU edge
        if v[t]:
            sc = time.time()
            while time.time() - sc < CPU_BURST_MS:
                cA = np.tanh(cA @ cB)*0.5 + 0.5      # heavy CPU draw, overlaps GPU
        if u[t]:
            while time.time() - s0 < GPU_BURST_MS:
                gA = (gA @ gB).tanh()*0.5 + 0.5
            if dev == "cuda": torch.cuda.synchronize()
        time.sleep(0.004)
        T[t] = st.latest_window(length=NTAP).reshape(-1, N_CH)[:NTAP]
        rest = STEP_S - (time.time() - s0)
        if rest > 0: time.sleep(rest)
        if t % 4 == 0:
            tc = temp_c()
            if tc > HOT:
                while temp_c() > COOL: time.sleep(1.0)
            if t % 400 == 0: print(f"  step {t}/{L} temp={tc:.0f}C ({time.time()-t0:.0f}s)", flush=True)
    st.stop()
    return np.tanh((T - med)/mad/8.0), med, mad


def lag(x, k):
    y = np.zeros_like(x)
    if k > 0: y[k:] = x[:-k]
    return y if k > 0 else x.copy()


def acc(X, y, tr, te, nc=2):
    mu = X[tr].mean(0); sd = X[tr].std(0)+1e-9; X = (X-mu)/sd; Y = np.eye(nc)[y]; best = 0.0
    for al in [1e-2, .1, 1, 10, 100, 1e3]:
        try:
            W = np.linalg.solve(X[tr].T@X[tr]+al*np.eye(X.shape[1]), X[tr].T@Y[tr])
            best = max(best, float(np.mean((X[te]@W).argmax(1) == y[te])))
        except Exception: pass
    return best


def main():
    u = np.random.default_rng(SEED_U).integers(0, 2, size=L)
    v = np.random.default_rng(SEED_V).integers(0, 2, size=L)
    print(f"[{HOST}] MIXING STRONG (heavy v: {CPU_MAT}² {CPU_BURST_MS*1000:.0f}ms) temp {temp_c():.0f}C", flush=True)
    Tn, med, mad = collect(u, v)
    flat = Tn.reshape(L, -1)
    du = np.abs((flat[u == 1].mean(0)-flat[u == 0].mean(0))/(np.sqrt((flat[u == 1].std(0)**2+flat[u == 0].std(0)**2)/2)+1e-9)).max()
    dv = np.abs((flat[v == 1].mean(0)-flat[v == 0].mean(0))/(np.sqrt((flat[v == 1].std(0)**2+flat[v == 0].std(0)**2)/2)+1e-9)).max()
    print(f"  drive landed: max|d_u|={du:.2f}  max|d_v|={dv:.2f}  (was d_v=0.59)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT/f"mixing_strong_raw_{HOST}.npz", u=u, v=v, Tn=Tn)

    Xdie = np.hstack([flat, lag(flat, 1), lag(flat, 2)])
    cut = WASHOUT+int(0.7*(L-WASHOUT)); tr = slice(WASHOUT, cut); te = slice(cut, L)
    y = (lag(u, 1) ^ lag(v, 1)).astype(int)
    die_xor = acc(Xdie, y, tr, te)
    # null
    null = []
    for s in range(300):
        r = np.random.default_rng(1000+s); Ms = np.zeros_like(flat)
        for c in range(flat.shape[1]):
            F = np.fft.rfft(flat[:, c]-flat[:, c].mean()); ph = r.uniform(0, 2*np.pi, len(F)); ph[0] = 0
            Ms[:, c] = np.fft.irfft(np.abs(F)*np.exp(1j*ph), n=L)
        null.append(acc(np.hstack([Ms, lag(Ms, 1), lag(Ms, 2)]), y, tr, te))
    null = np.array(null); p95 = float(np.quantile(null, 0.95)); pval = float((null >= die_xor).mean())
    # physical u*v
    uc = (u-u.mean()).astype(float); vc = (v-v.mean()).astype(float); uvc = uc*vc
    A2 = np.stack([np.ones(L), uc, vc], 1); A3 = np.stack([np.ones(L), uc, vc, uvc], 1)
    gains = []
    for c in range(flat.shape[1]):
        ch = flat[:, c]; b2, *_ = np.linalg.lstsq(A2, ch, rcond=None); b3, *_ = np.linalg.lstsq(A3, ch, rcond=None)
        ss = np.sum((ch-ch.mean())**2)+1e-12
        gains.append(float((np.sum((ch-A2@b2)**2)-np.sum((ch-A3@b3)**2))/ss))
    uv_max = float(np.max(gains)); uv_med = float(np.median(gains))
    # controls
    uu = u.astype(float); vv = v.astype(float)
    u_cols = [lag(uu, k) for k in range(1, 5)]
    for a, b in itertools.combinations_with_replacement(range(1, 5), 2): u_cols.append(lag(uu, a)*lag(uu, b))
    u_only = acc(np.stack(u_cols, 1), y, tr, te)
    uv_lin = acc(np.stack([lag(uu, k) for k in range(1, 5)]+[lag(vv, k) for k in range(1, 5)], 1), y, tr, te)
    ceiling = acc(np.stack([lag(uu, 1)*lag(vv, 1)], 1), y, tr, te)
    rec_v = acc(Xdie, lag(v, 1).astype(int), tr, te)
    out = {"host": HOST, "d_u": float(du), "d_v": float(dv), "die_full_XOR": die_xor, "null_p95": p95,
           "null_pvalue": pval, "uv_partialR2_max": uv_max, "uv_partialR2_median": uv_med,
           "u_only_poly": u_only, "u_and_v_LINEAR": uv_lin, "uv_ceiling": ceiling, "recall_v": rec_v,
           "prev_d_v": 0.59, "prev_die_XOR": 0.654, "prev_uv_R2max": 0.089}
    print(json.dumps(out, indent=2))
    print(f"\n  d_v: 0.59 -> {dv:.2f}   die XOR: 0.654 -> {die_xor:.3f} (null p95={p95:.3f}, p={pval:.3f})")
    print(f"  u·v R2max: 0.089 -> {uv_max:.3f}   u-only={u_only:.3f}  u&v-lin={uv_lin:.3f}  ceiling={ceiling:.3f}  recall_v={rec_v:.3f}")
    (OUT/f"mixing_strong_{HOST}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
