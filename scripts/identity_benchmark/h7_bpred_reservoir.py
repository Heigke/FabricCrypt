"""H7 deep probe #4 — BRANCH-PREDICTOR mispredict-count reservoir (clean PMU integer).

The top untested lever from the w22spoqla spec. The Zen5 TAGE predictor's mispredict count over a
block of data-dependent branches is a NON-ADDITIVE function of recent direction history (runs predict
free; flips cost). Prior timing probes failed because the rdtscp fence floor buried the ~15-cyc signal;
reading the clean integer PMU counter removes that. Rigorous reservoir-necessity test with a LINEAR
readout: can linear-on-{mispredict-count + lags} do delayed-XOR/parity of the drive that linear-on-drive
cannot? Controls: u_linear baseline (MUST beat) + phase-shuffle surrogate null (a linear filter a
surrogate also passes is rejected).

PRE-REGISTERED GATE (before measuring):
  PASS-XOR : mean delayed-XOR acc (k=1..3) >= 0.60 AND >= u_linear+0.05 AND > surrogate 99th pct
  PASS-PAR : 4-bit parity acc >= 0.70 (chance 0.0625) AND > u_linear
  FAIL     : XOR < 0.55 OR not beating u_linear by >=0.05 -> branch-predictor NEGATIVE (-> try Rank 1 DRAM)
Pure integer branches, single core, ~2W, no GPU, no thermal risk.
"""
from __future__ import annotations
import sys, json, socket, subprocess, os, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
HOST = socket.gethostname()
OUT = Path(__file__).resolve().parents[2] / "results/IDENTITY_H7_2026-06-09"
L = 12000
WASHOUT = 200
NLAG = 16
CORE = 4
SEED = 0


def run_probe(u_bits):
    uf = f"/tmp/h7bp_u_{os.getpid()}.txt"; of = f"/tmp/h7bp_o_{os.getpid()}.txt"
    Path(uf).write_text("\n".join(str(int(x)) for x in u_bits))
    subprocess.run([str(HERE / "bpred_pmu"), uf, of, str(CORE)], check=True)
    cnt = np.array([float(x) for x in Path(of).read_text().split()], dtype=np.float64)
    os.remove(uf); os.remove(of)
    return cnt[:len(u_bits)]


def lag_design(x, nlag):
    L = len(x); cols = []
    for k in range(nlag + 1):
        c = np.zeros(L); c[k:] = x[:L - k] if k else x
        cols.append(c)
    return np.stack(cols, axis=1)


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
    u = rng.integers(0, 2, size=L)

    # SANITY: the mispredict signal must exist at all (random drive mispredicts; all-ones doesn't)
    print(f"[{HOST}] branch-predictor PMU reservoir, L={L} core={CORE}...", flush=True)
    t0 = time.time()
    cnt = run_probe(u)
    ones = run_probe(np.ones(2000, dtype=int))
    print(f"  probe done ({time.time()-t0:.0f}s)  mispred/step: random mean={cnt.mean():.1f} std={cnt.std():.1f}"
          f"  | all-ones mean={ones.mean():.1f}  (random should be >> ones if predictor reacts)", flush=True)
    sig_d = (cnt.mean() - ones.mean()) / (cnt.std() + 1e-9)
    print(f"  drive sensitivity d(random vs ones)={sig_d:+.2f}", flush=True)

    Xres = lag_design(cnt, NLAG)
    Xdrv = lag_design(u.astype(float), NLAG)

    n = L - WASHOUT; cut = WASHOUT + int(0.7 * n)
    tr = slice(WASHOUT, cut); te = slice(cut, L)

    def lagbit(k):
        x = np.zeros(L, dtype=int); x[k:] = u[:-k]; return x
    y4 = np.zeros(L, dtype=int)
    for b, (a, c) in enumerate([(1, 2), (2, 3), (3, 4), (4, 5)]):
        y4 |= ((lagbit(a) ^ lagbit(c)) << b)
    tasks = {
        "RECALL_t1": (lagbit(1), 2),
        "XOR_k1":    (lagbit(1) ^ lagbit(2), 2),
        "XOR_k2":    (lagbit(2) ^ lagbit(3), 2),
        "XOR_k3":    (lagbit(3) ^ lagbit(4), 2),
        "PAR_4bit":  (y4, 16),
    }

    # phase-shuffle surrogate: destroy temporal phase of the die signal, keep its spectrum.
    # If the surrogate "passes" too, the gain is a linear-filter artifact, not real computation.
    def surrogate_acc(y, nc, nperm=20):
        accs = []
        F = np.fft.rfft(cnt - cnt.mean())
        mag = np.abs(F)
        for p in range(nperm):
            ph = rng.uniform(0, 2 * np.pi, size=len(F)); ph[0] = 0
            sur = np.fft.irfft(mag * np.exp(1j * ph), n=L)
            Xs = lag_design(sur, NLAG)
            accs.append(ridge_acc(Xs[tr], y[tr], Xs[te], y[te], nc))
        return float(np.percentile(accs, 99))

    suite = {}
    for nm, (y, nc) in tasks.items():
        r = ridge_acc(Xres[tr], y[tr], Xres[te], y[te], nc)
        base = ridge_acc(Xdrv[tr], y[tr], Xdrv[te], y[te], nc)
        sur = surrogate_acc(y, nc) if nm != "RECALL_t1" else float("nan")
        suite[nm] = {"chance": 1.0 / nc, "reservoir": r, "baseline_u_linear": base, "surrogate_p99": sur}
        beats = (nm.startswith("XOR") and r >= 0.60 and r >= base + 0.05 and r > sur)
        flag = "  <-- DIE COMPUTES (beats u_linear + surrogate)" if beats else ""
        print(f"  task {nm:9s} chance={1.0/nc:.3f}  reservoir={r:.3f}  u_linear={base:.3f}  surrogate99={sur:.3f}{flag}", flush=True)

    xs = [suite[k] for k in ("XOR_k1", "XOR_k2", "XOR_k3")]
    xmean = float(np.mean([t["reservoir"] for t in xs]))
    bmean = float(np.mean([t["baseline_u_linear"] for t in xs]))
    smean = float(np.mean([t["surrogate_p99"] for t in xs]))
    par = suite["PAR_4bit"]
    pass_xor = xmean >= 0.60 and xmean >= bmean + 0.05 and xmean > smean
    pass_par = par["reservoir"] >= 0.70 and par["reservoir"] > par["baseline_u_linear"]
    verdict = "PASS" if (pass_xor or pass_par) else "FAIL"
    res = {"host": HOST, "L": L, "core": CORE, "nlag": NLAG,
           "drive_sensitivity_d": sig_d, "mispred_random_mean": float(cnt.mean()),
           "mispred_ones_mean": float(ones.mean()), "task_suite": suite,
           "xor_mean": xmean, "xor_baseline_mean": bmean, "xor_surrogate_mean": smean,
           "verdict": verdict, "pass_rule": "XORmean>=0.60 & >=u_linear+0.05 & >surrogate  OR  parity>=0.70>u_linear"}
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"bpred_reservoir_{HOST}.json"; out.write_text(json.dumps(res, indent=2))
    print(f"\n>>> {verdict}   XORmean={xmean:.3f} (u_lin {bmean:.3f}, surr {smean:.3f})   par={par['reservoir']:.3f}   saved {out}", flush=True)
    if verdict == "FAIL":
        print("  Branch-predictor mispredict count carries NO extractable nonlinear computation either.", flush=True)
    else:
        print("  FIRST genuine die-supplied nonlinear computation from SoC telemetry.", flush=True)


if __name__ == "__main__":
    main()
