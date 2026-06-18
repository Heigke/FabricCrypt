"""H7 telemetry-as-physical-reservoir — does the host's ~50-D telemetry, in combination, COMPUTE?

Eric's clarified vision: the metrics, *in combination*, can perform arbitrary (few, small)
computations. This is precisely the reservoir-computing question, made falsifiable.

Method (echo-state / physical-reservoir):
  - INPUT: a binary stream u[t]. Each symbol is injected as a PHYSICAL drive on the host —
    u=1 -> heavy compute burst, u=0 -> near-idle — for a fixed window. The substrate (thermal mass,
    power delivery, DVFS controllers) is the reservoir; we do not add nodes, we drive it.
  - STATE: x[t] = the ~50-D telemetry vector read right after the drive window.
  - READOUT: a *linear* ridge map on a short window [x[t], x[t-1], ..., x[t-w]] -> target y[t].
  - TARGETS (easy -> hard):
      MEM-d   : y = u[t-d]                      (memory; linearly recoverable)
      AND-d   : y = u[t-1] & u[t-2]             (nonlinear but linearly separable)
      OR-d    : y = u[t-1] | u[t-2]             (")
      XOR     : y = u[t-1] ^ u[t-2]             (LINEARLY INSEPARABLE -- the litmus)
      PAR3    : y = u[t-1]^u[t-2]^u[t-3]        (harder nonlinear)
  - CONTROLS (the honesty core):
      * LINEAR-ON-INPUT baseline: same ridge readout trained on raw bit history u[t-k..t], NOT
        telemetry. For XOR/PAR3 this provably cannot exceed chance -> if the telemetry readout beats
        it, the substrate's PHYSICS supplied the nonlinearity. That is the whole claim.
      * SHUFFLE null: permute telemetry rows vs targets -> accuracy collapses to chance.
      * Memory capacity MC = sum_d corr^2(pred_d, u[t-d]).
  - Thermal-safe: per-step zone0 guard; wait_cool segments the run (segment boundaries excluded
    from windowed features). The slow thermal integrator that was the "loaded heater" CONFOUND in
    Phase 0 is here the reservoir's fading MEMORY -- same physics, opposite sign.

Out: telem_reservoir_{host}.npz + telem_reservoir_{host}.json (+ printed verdict).
Run:  HSA_OVERRIDE_GFX_VERSION=11.0.0 python h7_telemetry_reservoir.py --steps 600 --drive 0.18
"""
from __future__ import annotations
import os, sys, time, json, argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h7_rt_phase0 as P0   # reuse readers, zone0, wait_cool, Sampler.snap, OUT

HOST = P0.HOST
THERM_ABORT = float(os.environ.get("THERM_ABORT", "80"))
THERM_COOL  = float(os.environ.get("THERM_COOL", "55"))

# ---------------- physical drive ----------------
def make_driver(dev, n):
    """Return drive(symbol, secs). symbol 1 = heavy matmul burst, 0 = near-idle."""
    if dev == "cuda":
        import torch
        A = torch.randn(n, n, device="cuda"); B = torch.randn(n, n, device="cuda")
        def drive(sym, secs):
            t0 = time.monotonic()
            if sym:
                while time.monotonic() - t0 < secs:
                    C = A @ B; A.copy_(torch.tanh(C))   # keep it busy + nonlinear
                torch.cuda.synchronize()
            else:
                time.sleep(secs)                         # idle -> substrate relaxes
        return drive
    else:
        import numpy as _np
        A = _np.random.randn(n, n); B = _np.random.randn(n, n)
        def drive(sym, secs):
            t0 = time.monotonic()
            if sym:
                while time.monotonic() - t0 < secs:
                    A = _np.tanh(A @ B)
            else:
                time.sleep(secs)
        return drive

def snap_vec(samp):
    """One telemetry snapshot -> flat vector + (names once)."""
    rec = samp.snap()
    parts, names = [], []
    for k in P0.CHANNELS:
        v = rec[k]
        v = np.atleast_1d(np.asarray(v, float))
        if v.size == 0:
            continue
        parts.append(v); names += [f"{k}[{i}]" for i in range(v.size)]
    return (np.concatenate(parts) if parts else np.zeros(0)), names

# ---------------- record ----------------
def record(steps, drive_secs, mat_n, seed, gap=0.10):
    """Gentle reservoir drive: short matmul burst (sym=1) vs idle (sym=0), then a fixed RELAXATION
    gap before the read, so mean temperature stays in a continuous safe band (gfx1151 APUs have tiny
    thermal mass and overheat under sustained heavy matmul). Reservoir memory comes from the FAST
    electrical channels' DVFS/clock/power dynamics (~10-100ms), not large thermal swings."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    u = rng.integers(0, 2, size=steps).astype(np.int8)
    drive = make_driver(dev, mat_n)
    samp = P0.Sampler(period=0.0)            # only used for .snap()/.dims
    x0, names = snap_vec(samp)
    D = x0.size
    X = np.full((steps, D), np.nan); seg = np.zeros(steps, np.int32)
    zt = np.zeros(steps); tt = np.zeros(steps)
    cur_seg = 0
    print(f"[{HOST}] reservoir dev={dev} steps={steps} drive={drive_secs}s gap={gap}s mat_n={mat_n} "
          f"D={D} THERM_ABORT={THERM_ABORT}", flush=True)
    for t in range(steps):
        z = P0.zone0()
        if z >= THERM_ABORT:
            P0.wait_cool(THERM_COOL); cur_seg += 1
        drive(int(u[t]), drive_secs)
        time.sleep(gap)                       # relaxation: bound mean power, let fast dynamics settle
        v, _ = snap_vec(samp)
        if v.size == D: X[t] = v
        seg[t] = cur_seg; zt[t] = P0.zone0(); tt[t] = time.monotonic()
        if (t + 1) % 50 == 0:
            print(f"  step {t+1}/{steps} seg={cur_seg} zone0={zt[t]:.0f}C", flush=True)
    return dict(host=HOST, dev=dev, u=u, X=X, seg=seg, zone=zt, t=tt,
                names=np.array(names), drive_secs=drive_secs, mat_n=mat_n, gap=gap)

# ---------------- analyze ----------------
def _ridge_blocked(F, y, seg, valid, binary, lam=1.0, folds=5):
    """Time-blocked ridge; returns (metric, baseline_chance). metric=accuracy if binary else R2."""
    idx = np.where(valid)[0]
    if idx.size < 40: return np.nan, np.nan
    F = F[idx]; y = y[idx].astype(float); g = seg[idx]
    mu = F.mean(0); sd = F.std(0) + 1e-8; F = (F - mu) / sd
    F = np.column_stack([F, np.ones(len(F))])
    n = len(idx); bs = max(1, n // folds); preds = np.full(n, np.nan)
    for k in range(folds):
        te = np.zeros(n, bool); te[k*bs:(k+1)*bs if k < folds-1 else n] = True
        tr = ~te
        if tr.sum() < 10 or te.sum() < 2: continue
        A = F[tr]; b = y[tr]
        W = np.linalg.solve(A.T @ A + lam*np.eye(A.shape[1]), A.T @ b)
        preds[te] = F[te] @ W
    m = ~np.isnan(preds)
    if m.sum() < 5: return np.nan, np.nan
    if binary:
        acc = float(((preds[m] > 0.5).astype(int) == y[m].astype(int)).mean())
        chance = float(max(y[m].mean(), 1 - y[m].mean()))
        return acc, chance
    ss_res = float(((y[m]-preds[m])**2).sum()); ss_tot = float(((y[m]-y[m].mean())**2).sum())
    return (1 - ss_res/ss_tot if ss_tot > 1e-9 else np.nan), 0.0

def analyze(rec, w=4, maxlag=8):
    X = rec["X"]; u = rec["u"].astype(int); seg = rec["seg"]; T = len(u)
    # window features [x[t],...,x[t-w]] with NaN/segment-boundary masking
    feats = []; valid = np.ones(T, bool)
    for lag in range(w+1):
        sh = np.full_like(X, np.nan); sh[lag:] = X[:T-lag]
        same = np.zeros(T, bool); same[lag:] = seg[:T-lag] == seg[lag:]
        valid &= same; feats.append(sh)
    F = np.concatenate(feats, axis=1)
    valid &= ~np.isnan(F).any(1)
    # drop near-constant columns
    keep = F[valid].std(0) > 1e-6 if valid.any() else np.ones(F.shape[1], bool)
    F = F[:, keep]
    # input-history baseline features (raw bits u[t..t-maxlag])
    UB = []
    for lag in range(maxlag+1):
        sh = np.full(T, np.nan); sh[lag:] = u[:T-lag]; UB.append(sh)
    UB = np.column_stack(UB)

    def tgt(name):
        y = np.full(T, np.nan)
        if name.startswith("MEM"):
            d = int(name[3:]); y[d:] = u[:T-d]
        elif name == "AND": y[2:] = (u[1:T-1] & u[0:T-2])
        elif name == "OR":  y[2:] = (u[1:T-1] | u[0:T-2])
        elif name == "XOR": y[2:] = (u[1:T-1] ^ u[0:T-2])
        elif name == "PAR3": y[3:] = (u[2:T-1] ^ u[1:T-2] ^ u[0:T-3])
        return y

    tasks = ["MEM0","MEM1","MEM2","MEM3","AND","OR","XOR","PAR3"]
    out = {}
    for name in tasks:
        y = tgt(name); v = valid & ~np.isnan(y)
        binary = True
        acc_res, chance = _ridge_blocked(F, y, seg, v, binary)
        # baseline: same ridge on raw input bits (telemetry-free)
        ub_valid = v & ~np.isnan(UB).any(1)
        acc_base, _ = _ridge_blocked(UB[:, 1:], y, seg, ub_valid, binary)  # exclude u[t] itself? keep all lags>=1
        # shuffle null on telemetry
        rngn = np.random.default_rng(0)
        Fs = F.copy(); per = rngn.permutation(len(Fs)); Fs = Fs[per]
        acc_null, _ = _ridge_blocked(Fs, y, seg, v, binary)
        out[name] = dict(telem_acc=acc_res, chance=chance, input_base_acc=acc_base,
                         shuffle_null_acc=acc_null,
                         lift_over_input=(None if (np.isnan(acc_res) or np.isnan(acc_base)) else round(acc_res-acc_base,3)))
    # memory capacity from MEM tasks (regression form)
    mc = 0.0
    for d in range(0, maxlag+1):
        y = np.full(T, np.nan); y[d:] = u[:T-d]; v = valid & ~np.isnan(y)
        idx = np.where(v)[0]
        if idx.size < 40: continue
        r2, _ = _ridge_blocked(F, y.astype(float), seg, v, binary=False)
        if not np.isnan(r2) and r2 > 0: mc += min(r2, 1.0)
    out["_MC"] = round(mc, 3)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--drive", type=float, default=0.05)
    ap.add_argument("--gap", type=float, default=0.10)
    ap.add_argument("--mat_n", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w", type=int, default=4)
    a = ap.parse_args()
    rec = record(a.steps, a.drive, a.mat_n, a.seed, gap=a.gap)
    OUT = P0.OUT
    p = OUT / f"telem_reservoir_{HOST}.npz"
    np.savez_compressed(p.with_name(p.stem + ".tmp.npz"),
                        **{k: v for k, v in rec.items()})
    os.replace(p.with_name(p.stem + ".tmp.npz"), p)
    res = analyze(rec, w=a.w)
    res = {"host": HOST, "dev": rec["dev"], "steps": a.steps, "drive_secs": a.drive,
           "mat_n": a.mat_n, "D": int(rec["X"].shape[1]), "n_seg": int(rec["seg"].max()+1),
           **res}
    jp = OUT / f"telem_reservoir_{HOST}.json"
    jp.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)
    xor = res["XOR"]; par = res["PAR3"]
    print(f"\n[{HOST}] VERDICT:", flush=True)
    print(f"  MC={res['_MC']}  (fading memory present if >0)", flush=True)
    print(f"  XOR telem={xor['telem_acc']} vs input-baseline={xor['input_base_acc']} "
          f"null={xor['shuffle_null_acc']} chance={xor['chance']} -> lift {xor['lift_over_input']}", flush=True)
    print(f"  PAR3 telem={par['telem_acc']} vs input-baseline={par['input_base_acc']} "
          f"-> lift {par['lift_over_input']}", flush=True)
    print(f"  saved {p}\n  saved {jp}", flush=True)

if __name__ == "__main__":
    main()
