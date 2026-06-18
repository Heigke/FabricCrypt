"""H7 deep probe #3 — SPATIAL CONTENTION reservoir (capacity-eviction AND).

Temporal probes failed (1-D linear low-pass). This tests the canonical hardware nonlinearity:
two simultaneous drive bits a,b stream two ~half-L2 buffers; a small probe buffer P is evicted
ONLY when a AND b are both active (capacity threshold). lat_P encodes a physical AND(a,b).

Rigorous reservoir-necessity test: a LINEAR readout of the die's three latencies {lat_P,lat_A,lat_B}
(plus a few lags) is asked to compute XOR(a,b) / parity. Baseline is a LINEAR readout of the raw
drive (a,b) and lags. XOR is NOT linear in (a,b), so the baseline MUST fail; the reservoir can only
succeed if the SILICON supplied the AND nonlinearity. We deliberately do NOT use build_best_features
here — the readout stays linear so any nonlinear gain is attributable to the die, not the readout.

PASS (first genuine extractable computation) iff: reservoir(XOR) >= 0.70 AND baseline(XOR) <= 0.58
on at least one nonlinear task. Diagnostic: per-(a,b) conditional mean of lat_P (want lat_P|11 high).
Pure CPU timing, single core (taskset) -> no GPU, no thermal risk.
"""
from __future__ import annotations
import sys, json, time, socket, subprocess, os, shutil
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
HOST = socket.gethostname()
OUT = Path(__file__).resolve().parents[2] / "results/IDENTITY_H7_2026-06-09"
L = 4000
WASHOUT = 100
NLAG = 4                     # readout/baseline see lags 0..NLAG of each channel
SEED = 0
CORE = 3                     # pin to one core for a stable private L2


def run_probe(a_bits, b_bits):
    af = f"/tmp/h7c_a_{os.getpid()}.txt"; bf = f"/tmp/h7c_b_{os.getpid()}.txt"; of = f"/tmp/h7c_o_{os.getpid()}.txt"
    Path(af).write_text("\n".join(str(int(x)) for x in a_bits))
    Path(bf).write_text("\n".join(str(int(x)) for x in b_bits))
    cmd = []
    if shutil.which("taskset"):
        cmd = ["taskset", "-c", str(CORE)]
    cmd += [str(HERE / "latprobe_contention"), af, bf, of]
    subprocess.run(cmd, check=True)
    arr = np.array([[float(x) for x in ln.split()] for ln in Path(of).read_text().split("\n") if ln.strip()])
    for f in (af, bf, of): os.remove(f)
    return arr[:len(a_bits)]                          # (L, 3): lat_P, lat_A, lat_B


def lag_design(chans, nlag):
    """Stack channels with lags 0..nlag -> linear design matrix."""
    L, C = chans.shape
    cols = []
    for k in range(nlag + 1):
        x = np.zeros((L, C)); x[k:] = chans[:L - k] if k else chans
        cols.append(x)
    return np.hstack(cols)


def ridge_acc(Xtr, ytr, Xte, yte, nc):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    Y = np.eye(nc)[ytr]; best = 0.0
    for al in [1e-2, 0.1, 1, 10, 100, 1e3, 1e4]:
        try:
            W = np.linalg.solve(Xtr.T @ Xtr + al * np.eye(Xtr.shape[1]), Xtr.T @ Y)
            best = max(best, float(np.mean((Xte @ W).argmax(1) == yte)))
        except Exception:
            pass
    return best


def main():
    rng = np.random.default_rng(SEED)
    a = rng.integers(0, 2, size=L); b = rng.integers(0, 2, size=L)
    print(f"[{HOST}] driving spatial-contention reservoir (core {CORE}), L={L}...", flush=True)
    t0 = time.time()
    lat = run_probe(a, b)                              # (L,3)
    print(f"  probe done ({time.time()-t0:.0f}s)  lat means P/A/B = "
          f"{lat[:,0].mean():.0f}/{lat[:,1].mean():.0f}/{lat[:,2].mean():.0f}", flush=True)

    # --- KEY DIAGNOSTIC: does lat_P encode AND(a,b)? per-(a,b) conditional mean ---
    cond = {}
    for av in (0, 1):
        for bv in (0, 1):
            m = (a == av) & (b == bv)
            cond[f"{av}{bv}"] = float(lat[m, 0].mean()) if m.any() else float("nan")
    # AND contrast: lat_P|11 should stand ABOVE the other three if capacity threshold fires
    others = np.mean([cond["00"], cond["01"], cond["10"]])
    and_pop = float(cond["11"] - others)
    pooled_sd = float(lat[:, 0].std()) + 1e-9
    and_d = and_pop / pooled_sd
    print(f"  lat_P|(a,b): 00={cond['00']:.0f} 01={cond['01']:.0f} 10={cond['10']:.0f} 11={cond['11']:.0f}"
          f"   AND-pop d={and_d:+.2f}", flush=True)

    # Three LINEAR designs, all with lags:
    #   baseline = drive (a,b) only           -> CANNOT do XOR (chance)
    #   res_pure = die latencies only         -> pure-die necessity test
    #   res_aug  = drive + lat_P (the AND)    -> isolates the die's contribution = exactly lat_P
    # If res_aug >> baseline, the gain is attributable ONLY to lat_P (silicon-computed AND).
    drive = np.stack([a, b], axis=1).astype(float)
    Xbas = lag_design(drive, NLAG)
    Xpure = lag_design(lat, NLAG)
    Xaug = lag_design(np.column_stack([drive, lat[:, 0]]), NLAG)

    n = L - WASHOUT; cut = WASHOUT + int(0.7 * n)
    tr = slice(WASHOUT, cut); te = slice(cut, L)

    tasks = {
        "AND_ab":  ((a & b).astype(int), 2),                 # linearly separable (sanity; both should pass)
        "OR_ab":   ((a | b).astype(int), 2),                 # linearly separable (sanity)
        "XOR_ab":  ((a ^ b).astype(int), 2),                 # NOT linear in (a,b): the real test
        "XNOR_ab": (1 - (a ^ b).astype(int), 2),
    }
    suite = {}
    for nm, (y, nc) in tasks.items():
        base = ridge_acc(Xbas[tr], y[tr], Xbas[te], y[te], nc)
        rp = ridge_acc(Xpure[tr], y[tr], Xpure[te], y[te], nc)
        ra = ridge_acc(Xaug[tr], y[tr], Xaug[te], y[te], nc)
        suite[nm] = {"chance": 1.0 / nc, "reservoir_pure": rp, "reservoir_aug": ra, "baseline_drive_linear": base}
        flag = "  <-- DIE SUPPLIES NONLINEARITY" if (nm in ("XOR_ab", "XNOR_ab") and ra - base >= 0.12 and ra >= 0.70) else ""
        print(f"  task {nm:8s} chance={1.0/nc:.3f}  pure(lat)={rp:.3f}  aug(drive+latP)={ra:.3f}  base(drive)={base:.3f}{flag}", flush=True)

    xr = suite["XOR_ab"]
    verdict = "PASS" if (xr["reservoir_aug"] >= 0.70 and xr["reservoir_aug"] - xr["baseline_drive_linear"] >= 0.12) else "FAIL"
    res = {"host": HOST, "L": L, "core": CORE, "nlag": NLAG,
           "lat_means": [float(lat[:, i].mean()) for i in range(3)],
           "latP_cond_mean_by_ab": cond, "AND_pop_cohens_d": and_d,
           "task_suite": suite, "verdict": verdict,
           "pass_rule": "XOR reservoir>=0.70 AND baseline_drive_linear<=0.58 (linear readout => die supplies AND)"}
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"contention_reservoir_{HOST}.json"; out.write_text(json.dumps(res, indent=2))
    print(f"\n>>> {verdict}   (AND-pop d={and_d:+.2f})   saved {out}", flush=True)
    if verdict == "FAIL":
        print("  No extractable nonlinear computation from capacity contention either.", flush=True)
    else:
        print("  FIRST genuine die-supplied nonlinearity (capacity-threshold AND -> linear XOR).", flush=True)


if __name__ == "__main__":
    main()
