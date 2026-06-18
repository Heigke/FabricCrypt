"""H7 EXHAUSTIVE cross-channel/differential sweep for #2 (can the die compute?).

Closes the blind spot: our prior readouts took only LINEAR combinations ACROSS channels (per-channel
temporal products, then linear ridge). They never formed PRODUCTS or DIFFERENTIALS *between* channels.
A relation between an exogenous channel (crystal/TSC jitter, non-monotone) and a load channel could be
non-monotone even when each channel alone is monotone-in-load. This sweeps that exhaustively.

Drive a known binary u; collect all 10 substrate channels. Build an EXHAUSTIVE basis:
  - per-channel lags 0..K
  - all pairwise CROSS-channel products  ch_i(t-a)*ch_j(t-b)   (all i<j, all lag pairs)
  - all per-channel AUTO products        ch_i(t-a)*ch_i(t-b)
  - all pairwise DIFFERENTIALS/ratios    ch_i-ch_j , ch_i/ch_j
Test delayed-XOR(k1,k2) for ALL lag pairs + parity, with the decisive controls:
  - linear-on-u (lags of u): the floor a real reservoir must beat
  - NONLINEAR-on-u (same product basis applied to u alone): isolates die vs readout
  - phase-shuffle surrogate null
PASS only if reservoir beats BOTH the nonlinear-u control AND the surrogate on some XOR/parity task.
Light drive + in-loop thermal self-guard. Root (substrate PM-table reads).
"""
from __future__ import annotations
import sys, json, time, socket, itertools
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from substrate_realtime_v3 import SubstrateStateV3

HOST = socket.gethostname()
OUT = Path(__file__).resolve().parents[2] / "results/IDENTITY_H7_2026-06-09"
ZONE = Path("/sys/class/thermal/thermal_zone0/temp")
N_CH = 10
L = 1800
WASHOUT = 150
DWELL = 0.012
LAGS = [0, 1, 2, 3]          # lag set for cross/auto products (keeps basis ~manageable)
PERCH_LAGS = list(range(0, 9))
SEED = 0


def temp_c():
    try: return int(ZONE.read_text()) / 1000.0
    except Exception: return 0.0


def collect(u):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st = SubstrateStateV3(hz_target=500); st.start()
    time.sleep(6.0)
    pool = np.array([st.latest_window(length=64).reshape(-1, N_CH) for _ in range(40)]).reshape(-1, N_CH)
    med = np.median(pool, 0); mad = np.median(np.abs(pool - med), 0) * 1.4826 + 1e-9
    a = torch.randn(1536, 1536, device=device); b = torch.randn(1536, 1536, device=device)
    S = np.zeros((L, N_CH), np.float32); t0 = time.time()
    for t in range(L):
        # SUSTAINED timed load so the 500Hz sampler resolves it: u=1 -> busy ~10ms, u=0 -> idle ~10ms
        t_end = time.time() + 0.010
        if u[t]:
            while time.time() < t_end:
                for _ in range(4): a = (a @ b).tanh() * 0.5 + 0.5
                if device == "cuda": torch.cuda.synchronize()
        else:
            time.sleep(0.010)
        S[t] = st.latest_window(length=6).mean(0)
        if t % 20 == 0:
            tc = temp_c()
            if tc > 82.0:
                while temp_c() > 62.0: time.sleep(1.0)
            if t % 600 == 0: print(f"  step {t}/{L} temp={tc:.0f}C ({time.time()-t0:.0f}s)", flush=True)
    st.stop()
    Sn = np.tanh((S - med) / mad / 8.0)
    d = ((Sn[u == 1].mean(0) - Sn[u == 0].mean(0)) /
         (np.sqrt((Sn[u == 1].std(0)**2 + Sn[u == 0].std(0)**2) / 2) + 1e-9))
    print(f"  drive-landed Cohen's d per channel: {[round(x,1) for x in d]}", flush=True)
    return Sn


def lagged(x, k):
    y = np.zeros_like(x);
    if k == 0: return x.copy()
    y[k:] = x[:-k]; return y


def build_basis(chans):
    """Exhaustive: per-ch lags + cross/auto products + pairwise differentials. chans: (L,C)."""
    L, C = chans.shape; cols = []
    for c in range(C):
        for k in PERCH_LAGS:
            cols.append(lagged(chans[:, c], k))
    for i, j in itertools.combinations(range(C), 2):     # cross-channel products
        for a in LAGS:
            for b in LAGS:
                cols.append(lagged(chans[:, i], a) * lagged(chans[:, j], b))
    for c in range(C):                                   # auto products (a<b)
        for a, b in itertools.combinations(LAGS, 2):
            cols.append(lagged(chans[:, c], a) * lagged(chans[:, c], b))
    for i, j in itertools.combinations(range(C), 2):     # differentials
        cols.append(chans[:, i] - chans[:, j])
    return np.stack(cols, axis=1)


def u_nonlin_basis(u):
    """Same-style nonlinear basis on the DRIVE alone (fair control)."""
    uu = u.astype(float)[:, None]
    cols = [lagged(uu[:, 0], k) for k in range(0, 16)]
    for a, b in itertools.combinations(range(0, 8), 2):
        cols.append(lagged(uu[:, 0], a) * lagged(uu[:, 0], b))
    return np.stack(cols, axis=1)


def ridge_acc(Xtr, ytr, Xte, yte, nc):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    Y = np.eye(nc)[ytr]; best = 0.0
    for al in [1, 10, 100, 1e3, 1e4, 1e5]:
        try:
            W = np.linalg.solve(Xtr.T @ Xtr + al * np.eye(Xtr.shape[1]), Xtr.T @ Y)
            best = max(best, float(np.mean((Xte @ W).argmax(1) == yte)))
        except Exception:
            pass
    return best


def main():
    rng = np.random.default_rng(SEED)
    u = rng.integers(0, 2, size=L)
    print(f"[{HOST}] EXHAUSTIVE cross-channel sweep, L={L} (temp {temp_c():.0f}C)...", flush=True)
    Sn = collect(u)
    dland = np.abs((Sn[u == 1].mean(0) - Sn[u == 0].mean(0)) /
                   (np.sqrt((Sn[u == 1].std(0)**2 + Sn[u == 0].std(0)**2) / 2) + 1e-9)).max()
    if dland < 0.8:
        print(f"  !! DRIVE DID NOT LAND (max |d|={dland:.2f}); test invalid — abort.", flush=True)
        (OUT / f"exhaustive_xchannel_{HOST}.json").write_text(json.dumps(
            {"host": HOST, "verdict": "INVALID_DRIVE_NOT_LANDED", "max_drive_d": float(dland)}, indent=2))
        return
    F = build_basis(Sn); Fu = u_nonlin_basis(u)
    Ulin = np.stack([lagged(u.astype(float), k) for k in range(1, 16)], axis=1)
    print(f"  basis dims: reservoir={F.shape[1]}  nonlinear-u={Fu.shape[1]}", flush=True)

    n = L - WASHOUT; cut = WASHOUT + int(0.7 * n)
    tr = slice(WASHOUT, cut); te = slice(cut, L)

    def lb(k): return lagged(u, k).astype(int)
    tasks = {}
    for k1, k2 in itertools.combinations(range(1, 7), 2):     # all XOR lag pairs
        tasks[f"XOR_{k1}{k2}"] = (lb(k1) ^ lb(k2), 2)
    y4 = np.zeros(L, int)
    for bbit, (a, c) in enumerate([(1, 2), (2, 3), (3, 4), (4, 5)]):
        y4 |= ((lb(a) ^ lb(c)) << bbit)
    tasks["PAR_4bit"] = (y4, 16)

    # surrogate: phase-shuffle each channel, rebuild basis, take 99th pct over a few draws
    def surrogate(y, nc, nperm=8):
        accs = []
        for _ in range(nperm):
            Ssur = np.zeros_like(Sn)
            for c in range(N_CH):
                Fr = np.fft.rfft(Sn[:, c] - Sn[:, c].mean()); ph = rng.uniform(0, 2*np.pi, len(Fr)); ph[0] = 0
                Ssur[:, c] = np.fft.irfft(np.abs(Fr) * np.exp(1j*ph), n=L)
            Fs = build_basis(Ssur)
            accs.append(ridge_acc(Fs[tr], y[tr], Fs[te], y[te], nc))
        return float(np.percentile(accs, 99))

    suite = {}; hits = []
    for nm, (y, nc) in tasks.items():
        r = ridge_acc(F[tr], y[tr], F[te], y[te], nc)
        blin = ridge_acc(Ulin[tr], y[tr], Ulin[te], y[te], nc)
        bnl = ridge_acc(Fu[tr], y[tr], Fu[te], y[te], nc)
        sur = surrogate(y, nc) if nm.startswith("XOR") else float("nan")
        win = (nm.startswith("XOR") and r >= bnl + 0.04 and r > sur and r > 1.0/nc + 0.04)
        if win: hits.append(nm)
        suite[nm] = {"chance": 1.0/nc, "reservoir": r, "u_linear": blin, "u_nonlinear": bnl, "surrogate99": sur, "die_wins": bool(win)}
        flag = "  <-- DIE WINS" if win else ""
        print(f"  {nm:9s} chance={1.0/nc:.3f} res={r:.3f} u_lin={blin:.3f} u_nl={bnl:.3f} surr={sur if sur==sur else 0:.3f}{flag}", flush=True)

    verdict = "PASS" if hits else "FAIL"
    res = {"host": HOST, "L": L, "basis_dims": int(F.shape[1]), "task_suite": suite,
           "die_wins_on": hits, "verdict": verdict,
           "pass_rule": "some XOR: reservoir >= u_nonlinear+0.04 AND > surrogate99 AND > chance+0.04"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"exhaustive_xchannel_{HOST}.json").write_text(json.dumps(res, indent=2))
    print(f"\n>>> {verdict}   die_wins_on={hits}   saved exhaustive_xchannel_{HOST}.json", flush=True)


if __name__ == "__main__":
    main()
