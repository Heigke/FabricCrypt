"""H7 deep probe — driven CACHE-LATENCY reservoir capacity test (CPU-side, thermally trivial).

The smooth power/thermal channels are linear (failed XOR). The memory hierarchy is NONLINEAR
(cache hit/miss = threshold) with state (residency = memory). We drive a pointer-chase with a
binary stream u at 3 memory scales (L2 / L3 / DRAM) and ask the same question as the Step-0 gate:
can a linear readout of the latency dynamics compute delayed-XOR/parity of u that a linear model
of u cannot? If YES on ANY task -> the die's cache dynamics carry usable nonlinear computation.

Reuses z2296 build_best_features. Pure CPU timing (rdtsc) -> no GPU, no thermal risk.
PASS (worth coupling) iff reservoir >= 0.70 (16-way) AND baseline <= 0.30 (per design gate);
also reports the full XOR/parity suite + a per-scale latency fingerprint (PUF seed).
"""
from __future__ import annotations
import sys, json, time, socket, subprocess, os
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from z2296_best_of_all import build_best_features

HERE = Path(__file__).parent
HOST = socket.gethostname()
OUT = Path(__file__).resolve().parents[2] / "results/IDENTITY_H7_2026-06-09"
L = 3000
WASHOUT = 150
NS_HOT = 1 << 15                                   # 256KB hot buffer (~L2)
SIZES = [(1 << 21, "cold16MB"), (1 << 23, "cold64MB"), (1 << 25, "cold256MB")]
PARITY_LAGS = [(1, 3), (2, 6), (4, 9), (5, 12)]
SEED = 0


def run_probe(u_bits, n_cold):
    uf = f"/tmp/h7lat_u_{os.getpid()}.txt"; of = f"/tmp/h7lat_o_{os.getpid()}.txt"
    Path(uf).write_text("\n".join(str(int(x)) for x in u_bits))
    subprocess.run([str(HERE / "latprobe"), uf, of, str(NS_HOT), str(n_cold)], check=True)
    lat = np.array([float(x) for x in Path(of).read_text().split()], dtype=np.float64)
    os.remove(uf); os.remove(of)
    return lat[:len(u_bits)]


def norm(x):
    med = np.median(x, axis=0); mad = np.median(np.abs(x - med), axis=0) * 1.4826 + 1e-9
    return np.tanh((x - med) / mad / 8.0)


def mc_ridge(Ftr, ytr, Fte, yte, nc):
    mu = Ftr.mean(0); sd = Ftr.std(0) + 1e-8
    Ftr = (Ftr - mu) / sd; Fte = (Fte - mu) / sd
    Y = np.eye(nc)[ytr]; best = 0.0
    for al in [0.1, 1, 10, 100, 1000, 1e4]:
        try:
            W = np.linalg.solve(Ftr.T @ Ftr + al * np.eye(Ftr.shape[1]), Ftr.T @ Y)
            best = max(best, float(np.mean((Fte @ W).argmax(1) == yte)))
        except Exception:
            pass
    return best


def main():
    rng = np.random.default_rng(SEED)
    u = rng.integers(0, 2, size=L)
    print(f"[{HOST}] driving cache-latency reservoir at {len(SIZES)} scales, L={L}...", flush=True)
    cols, fp = [], {}
    t0 = time.time()
    for n_lines, name in SIZES:
        lat = run_probe(u, n_lines)
        cols.append(lat)
        fp[name] = {"mean": float(lat.mean()), "std": float(lat.std()),
                    "p99_over_med": float(np.percentile(lat, 99) / (np.median(lat) + 1e-9))}
        # drive-landed: does u move this scale's latency?
        d = (lat[u == 1].mean() - lat[u == 0].mean()) / (np.sqrt((lat[u==1].std()**2+lat[u==0].std()**2)/2)+1e-9)
        print(f"  {name:10s} mean={lat.mean():7.0f} std={lat.std():6.0f} drive_d={d:+.2f}  ({time.time()-t0:.0f}s)", flush=True)
    S = np.stack(cols, axis=1)                     # (L, n_scales)
    Sn = norm(S)
    dspikes = np.abs(np.vstack([np.zeros((1, Sn.shape[1])), np.diff(Sn, axis=0)]))
    F = build_best_features(Sn, dspikes)

    n = L - WASHOUT; cut = WASHOUT + int(0.7 * n)
    tr = slice(WASHOUT, cut); te = slice(cut, L)

    def lagbit(k):
        x = np.zeros(L, dtype=int); x[k:] = u[:-k]; return x
    y4 = np.zeros(L, dtype=int)
    for b, (a, c) in enumerate(PARITY_LAGS):
        y4 |= ((lagbit(a) ^ lagbit(c)) << b)
    U = np.zeros((L, 15), dtype=np.float32)
    for j, k in enumerate(range(1, 16)):
        U[k:, j] = u[:-k]

    # DECISIVE control: same NONLINEAR readout (build_best_features) applied to u ALONE.
    # If reservoir(die) ~ base_nl, the XOR was done by the READOUT on linear u-memory, not the die.
    u_chan = u.reshape(-1, 1).astype(float)
    Fu = build_best_features(u_chan, np.abs(np.vstack([np.zeros((1, 1)), np.diff(u_chan, axis=0)])))

    tasks = {
        "RECALL_t3": (lagbit(3), 2),
        "XOR_t1t2": (lagbit(1) ^ lagbit(2), 2),
        "XOR_t2t5": (lagbit(2) ^ lagbit(5), 2),
        "PAR_2bit": ((lagbit(1) ^ lagbit(3)) | ((lagbit(2) ^ lagbit(6)) << 1), 4),
        "PAR_4bit": (y4, 16),
    }
    suite = {}
    for nm, (yt, nc) in tasks.items():
        r = mc_ridge(F[tr], yt[tr], F[te], yt[te], nc)
        blin = mc_ridge(U[tr], yt[tr], U[te], yt[te], nc)          # linear-on-u
        bnl = mc_ridge(Fu[tr], yt[tr], Fu[te], yt[te], nc)         # nonlinear-readout-on-u (fair control)
        suite[nm] = {"chance": 1.0 / nc, "reservoir": r, "baseline_u_linear": blin, "baseline_u_nonlinear": bnl}
        # the die contributes ONLY if it beats the SAME nonlinear readout applied to u
        flag = "  <-- DIE ADDS BEYOND READOUT" if (r - bnl > 0.04 and r > 1.0/nc + 0.04) else ""
        print(f"  task {nm:10s} chance={1.0/nc:.3f}  reservoir={r:.3f}  u_linear={blin:.3f}  u_nonlin={bnl:.3f}{flag}", flush=True)

    res4 = suite["PAR_4bit"]
    # PASS only if the die beats the SAME nonlinear readout on u (not just the linear-u baseline).
    verdict = "PASS" if (res4["reservoir"] >= 0.70 and res4["baseline_u_nonlinear"] <= 0.30) else "FAIL"
    # DECISIVE softer signal: reservoir beats the nonlinear-u control by >4pp on ANY nonlinear task.
    any_nl = any(t["reservoir"] - t["baseline_u_nonlinear"] > 0.04 and t["reservoir"] > t["chance"] + 0.04
                 for k, t in suite.items() if k != "RECALL_t3")
    res = {"host": HOST, "L": L, "sizes": [s[1] for s in SIZES], "ns_hot": NS_HOT,
           "task_suite": suite, "latency_fingerprint": fp,
           "any_nonlinear_gain_over_readout": bool(any_nl), "verdict": verdict,
           "pass_rule": "PAR_4bit reservoir>=0.70 AND baseline_u_nonlinear<=0.30"}
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"latency_reservoir_{HOST}.json"; out.write_text(json.dumps(res, indent=2))
    print(f"\n>>> {verdict}   any_nonlinear_gain={any_nl}   saved {out}", flush=True)
    if not any_nl:
        print("  Cache-latency dynamics show memory but NO nonlinear computational gain either.", flush=True)


if __name__ == "__main__":
    main()
