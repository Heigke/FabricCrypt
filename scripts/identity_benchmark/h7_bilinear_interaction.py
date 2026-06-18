"""H7 bilinear-interaction probe — does the die compute a genuine PRODUCT a·b (mixed partial)?

Reframe per user: don't demand the die compute XOR alone with an unbounded readout. Test whether the
die supplies a NONLINEAR INTERACTION term that a rank-limited LINEAR adapter cannot synthesize itself.

Drive TWO independent continuous loads simultaneously: a = GPU duty, b = CPU(numpy) duty, each in [0,1].
They share one power/thermal/Vdroop envelope, so any throttling/coupling makes the response NON-additive.
For each substrate channel r, the 2-way interaction
        I(a,b) = r(a,b) - [ r(a,0) + r(0,b) - r(0,0) ]
isolates the genuine bilinear part (the mixed partial ∂²r/∂a∂b). Randomized cell order kills the
(a,b)<->heat confound; reps give a noise floor.

NECESSITY-vs-linear-adapter test ("a little help"): target = a·b (a product a linear-in-(a,b) model
CANNOT make). Can a LINEAR readout of the die channels predict a·b better than linear-on-(a,b)? If yes,
the die provides the product the starved adapter needs -> genuine (if modest) die computation.

Run both in the LINEAR regime (low loads) and toward the THROTTLE regime (high loads) — nonlinearity
should be strongest at the regime boundary. Root (substrate), HSA override. In-loop thermal guard.
"""
from __future__ import annotations
import sys, json, time, socket, threading
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent))
from substrate_realtime_v3 import SubstrateStateV3

HOST = socket.gethostname()
OUT = Path(__file__).resolve().parents[2] / "results/IDENTITY_H7_2026-06-09"
ZONE = Path("/sys/class/thermal/thermal_zone0/temp")
N_CH = 10
LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
REPS = 6
SLOTS = 4
SLOT_S = 0.020          # 4*20ms = 80ms dwell per cell
SEED = 0
HOT = 78.0              # dual GPU+CPU drive is the hottest probe we run — guard HARD
COOL = 56.0


def temp_c():
    try: return int(ZONE.read_text()) / 1000.0
    except Exception: return 0.0


def cpu_busy(frac, deadline_evt, intensity_holder):
    """numpy matmul (releases GIL) busy a `frac` duty of each slot until stopped."""
    M = np.random.randn(512, 512).astype(np.float32); N = np.random.randn(512, 512).astype(np.float32)
    while not deadline_evt.is_set():
        frac = intensity_holder[0]
        t0 = time.time(); busy = frac * SLOT_S
        while time.time() - t0 < busy:
            M = np.tanh(M @ N) * 0.5 + 0.5
        rest = SLOT_S - (time.time() - t0)
        if rest > 0: time.sleep(rest)


def run_cell(a, b, gA, gB, dev, cpu_intensity):
    cpu_intensity[0] = b
    for _ in range(SLOTS):
        t0 = time.time(); busy = a * SLOT_S
        while time.time() - t0 < busy:
            gA = (gA @ gB).tanh() * 0.5 + 0.5
        if dev == "cuda": torch.cuda.synchronize()
        rest = SLOT_S - (time.time() - t0)
        if rest > 0: time.sleep(rest)
    return gA


def main():
    rng = np.random.default_rng(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gA = torch.randn(1024, 1024, device=dev); gB = torch.randn(1024, 1024, device=dev)
    st = SubstrateStateV3(hz_target=500); st.start()
    print(f"[{HOST}] bilinear-interaction probe dev={dev} (temp {temp_c():.0f}C) warmup 6s...", flush=True)
    time.sleep(6.0)
    pool = np.array([st.latest_window(length=64).reshape(-1, N_CH) for _ in range(40)]).reshape(-1, N_CH)
    med = np.median(pool, 0); mad = np.median(np.abs(pool - med), 0) * 1.4826 + 1e-9

    # CPU-load worker thread (duty controlled live via cpu_intensity[0])
    stop = threading.Event(); cpu_intensity = [0.0]
    th = threading.Thread(target=cpu_busy, args=(0.0, stop, cpu_intensity), daemon=True); th.start()

    cells = [(a, b) for a in LEVELS for b in LEVELS]
    order = [(r, a, b) for r in range(REPS) for (a, b) in cells]
    rng.shuffle(order)
    samples = []   # (a, b, [10 channels])
    t0 = time.time()
    for k, (r, a, b) in enumerate(order):
        gA = run_cell(a, b, gA, gB, dev, cpu_intensity)
        cpu_intensity[0] = 0.0
        w = st.latest_window(length=8).mean(0)
        samples.append((a, b, w))
        if k % 5 == 0:
            tc = temp_c()
            if tc > HOT:
                cpu_intensity[0] = 0.0
                print(f"  [guard] {tc:.0f}C cooling", flush=True)
                while temp_c() > COOL: time.sleep(1.0)
            if k % 60 == 0: print(f"  cell {k}/{len(order)} temp={tc:.0f}C ({time.time()-t0:.0f}s)", flush=True)
    stop.set(); st.stop()
    print(f"  collected {len(samples)} cells in {time.time()-t0:.0f}s", flush=True)

    A = np.array([s[0] for s in samples]); B = np.array([s[1] for s in samples])
    Rraw = np.array([s[2] for s in samples])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"bilinear_raw_{HOST}.npz", A=A, B=B, Rraw=Rraw, med=med, mad=mad)  # SAVE FIRST
    R = np.tanh((Rraw - med) / mad / 8.0)   # (Nsamp,10) normed

    # cell means over reps
    def cell_mean(ch, a, b):
        m = (A == a) & (B == b)
        return R[m, ch].mean() if m.any() else np.nan
    def cell_std(ch, a, b):
        m = (A == a) & (B == b)
        return R[m, ch].std() if m.sum() > 1 else 0.0

    results = {}
    for ch in range(N_CH):
        r00 = cell_mean(ch, 0.0, 0.0)
        # interaction surface I(a,b) = r(a,b) - [r(a,0)+r(0,b)-r(0,0)]
        Igrid, noise = [], []
        for a in LEVELS:
            for b in LEVELS:
                if a == 0 or b == 0: continue
                rab = cell_mean(ch, a, b); ra0 = cell_mean(ch, a, 0.0); r0b = cell_mean(ch, 0.0, b)
                Igrid.append(rab - (ra0 + r0b - r00)); noise.append(cell_std(ch, a, b))
        Igrid = np.array(Igrid); noise = np.array(noise)
        # total non-constant structure for normalization
        allmean = np.array([cell_mean(ch, a, b) for a in LEVELS for b in LEVELS])
        struct = np.nanstd(allmean) + 1e-9
        inter_rms = float(np.sqrt(np.nanmean(Igrid**2)))
        noise_rms = float(np.sqrt(np.nanmean(noise**2))) + 1e-9
        # bilinear coefficient: fit I ~ c*(a*b)
        ab = np.array([a*b for a in LEVELS for b in LEVELS if a != 0 and b != 0])
        c = float(np.polyfit(ab, Igrid, 1)[0]) if len(ab) == len(Igrid) else 0.0
        results[f"ch{ch}"] = {"interaction_rms": inter_rms, "noise_rms": noise_rms,
                              "interaction_over_noise": inter_rms / noise_rms,
                              "interaction_frac_of_structure": inter_rms / struct,
                              "bilinear_coeff_ab": c}

    # NECESSITY-vs-linear-adapter: predict a*b from die channels (linear) vs from (a,b) (linear)
    def ridge_r2(X, y):
        cut = int(0.7*len(X)); mu = X[:cut].mean(0); sd = X[:cut].std(0)+1e-9
        Xtr = (X[:cut]-mu)/sd; Xte = (X[cut:]-mu)/sd; ytr = y[:cut]; yte = y[cut:]
        best = -9.0
        for al in [1e-2, 0.1, 1, 10, 100]:
            W = np.linalg.solve(Xtr.T@Xtr + al*np.eye(Xtr.shape[1]), Xtr.T@(ytr-ytr.mean()))
            pred = Xte@W + ytr.mean(); ss = 1 - np.sum((pred-yte)**2)/(np.sum((yte-yte.mean())**2)+1e-9)
            best = max(best, float(ss))
        return best
    idx = rng.permutation(len(R))
    Rs, As, Bs = R[idx], A[idx], B[idx]
    y = (As*Bs)
    r2_die = ridge_r2(Rs, y)
    r2_drive = ridge_r2(np.stack([As, Bs], 1), y)        # linear-on-(a,b) CANNOT make a*b -> ~0
    r2_drive_nl = ridge_r2(np.stack([As, Bs, As*Bs], 1), y)  # sanity: with a*b given -> ~1

    best_ch = max(results, key=lambda k: results[k]["interaction_over_noise"])
    verdict = ("DIE PROVIDES PRODUCT" if (r2_die - r2_drive > 0.15 and r2_die > 0.3)
               else "no usable interaction")
    out = {"host": HOST, "levels": LEVELS, "reps": REPS, "per_channel": results,
           "best_interaction_channel": best_ch,
           "predict_a_times_b": {"from_die_linear": r2_die, "from_drive_linear": r2_drive,
                                 "from_drive_with_ab_given": r2_drive_nl},
           "verdict": verdict}
    def jsonable(o):
        if isinstance(o, dict): return {k: jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
        if isinstance(o, (np.floating, np.integer)): return float(o)
        return o
    (OUT / f"bilinear_interaction_{HOST}.json").write_text(json.dumps(jsonable(out), indent=2))
    print("\n  per-channel interaction (mixed partial) strength:", flush=True)
    for ch in range(N_CH):
        d = results[f"ch{ch}"]
        print(f"   ch{ch}: I/noise={d['interaction_over_noise']:5.2f}  I/struct={d['interaction_frac_of_structure']:.2f}  c_ab={d['bilinear_coeff_ab']:+.3f}", flush=True)
    print(f"\n  predict a*b:  from DIE(linear)={r2_die:.3f}   from drive(a,b linear)={r2_drive:.3f}   (sanity a*b-given={r2_drive_nl:.3f})", flush=True)
    print(f"  >>> {verdict}   (best ch {best_ch})", flush=True)


if __name__ == "__main__":
    main()
