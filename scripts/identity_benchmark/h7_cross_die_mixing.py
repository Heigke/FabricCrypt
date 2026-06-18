"""H7 CROSS-DIE EXOGENOUS-MIXING probe — the experiment all 4 O105 oracles converged on.

REFRAME (unanimous): "die computes a function of the COMMAND" is the wrong bar (self-computable). The right
bar for requirement (2) is die-as-dynamic-PUF-reservoir: the die performs a die-specific NONLINEAR MIXING of
an EXOGENOUS stream v with the command u. GPT-5's sharpening: the structural theorem's hidden premise is that
the control sees ALL inputs. Introduce a 2nd stream v the adapter CANNOT see in clean form (deploy: RDSEED
inside a TEE; here: simply WITHHELD from the control). Then XOR(u,v) is computable ONLY via the die's physical
u*v mixing term (= our measured ch5 bilinear, +0.138). A u-only control at ANY polynomial order sits at chance
because it lacks v. That is NON-circular necessity, and it tests our one real signal in the framing where it
would matter.

DESIGN: u drives sharp GPU bursts, v drives sharp CPU bursts (cross-subsystem contention = exactly the GPU x
CPU power interaction that produced the only genuine bilinear term). Sharp edges + low duty => di/dt physics
while staying cool-ish. Collect NTAP settling transient taps. Both u,v from fixed seeds so daedalus can
reproduce the IDENTICAL command stream for the cross-die layer.

WITHIN-DIE GATE (run here first; only go cross-die if this passes):
  - die rank-4 / full linear readout of telemetry on XOR(u,v), AND(u,v), and diagnostics RECALL_u/RECALL_v.
  - CONTROLS: u-only (linear + quadratic) -> structurally at chance on XOR(u,v) since it lacks v;
              v-ablated check (can the die even READ v? if RECALL_v ~ chance, XOR(u,v) is hopeless);
              phase-shuffle surrogate of die features.
  - PRE-REGISTERED WIN (anti-bias): die_xor > surrogate + 0.05  AND  die_xor > 0.55  AND  u_only_poly_xor < 0.55.
Saves cross_die_mixing_raw_{HOST}.npz (u, v, T) + cross_die_mixing_{HOST}.json. THERMAL: dual-engine is the
HOTTEST path (bilinear hit 96-98C) -> tight guard, low duty. Root.
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
L = 2200
WASHOUT = 150
GPU_BURST_MS = 0.004
CPU_BURST_MS = 0.0035
STEP_S = 0.035          # ~11% duty per engine -> low average power
NTAP = 12
R = 4
SEED_U = 0
SEED_V = 12345          # FIXED so daedalus reproduces identical v
HOT = 72.0; COOL = 55.0


def temp_c():
    try: return int(ZONE.read_text())/1000.0
    except Exception: return 0.0


def collect(u, v):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gA = torch.randn(2048, 2048, device=dev); gB = torch.randn(2048, 2048, device=dev)  # GPU edge
    cA = np.random.default_rng(1).standard_normal((768, 768))                            # CPU edge
    cB = np.random.default_rng(2).standard_normal((768, 768))
    st = SubstrateStateV3(hz_target=500); st.start(); time.sleep(6.0)
    pool = np.array([st.latest_window(length=64).reshape(-1, N_CH) for _ in range(40)]).reshape(-1, N_CH)
    med = np.median(pool, 0); mad = np.median(np.abs(pool - med), 0)*1.4826 + 1e-9
    T = np.zeros((L, NTAP, N_CH), np.float32); t0 = time.time()
    for t in range(L):
        s0 = time.time()
        # launch GPU burst (async on device) then spin CPU burst -> physical overlap = mixing when both on
        if u[t]:
            while time.time() - s0 < GPU_BURST_MS:
                gA = (gA @ gB).tanh()*0.5 + 0.5
        if v[t]:
            sc = time.time()
            while time.time() - sc < CPU_BURST_MS:
                cA = np.tanh(cA @ cB)*0.5 + 0.5
        if u[t] and dev == "cuda":
            torch.cuda.synchronize()
        time.sleep(0.004)
        T[t] = st.latest_window(length=NTAP).reshape(-1, N_CH)[:NTAP]
        rest = STEP_S - (time.time() - s0)
        if rest > 0: time.sleep(rest)
        if t % 5 == 0:
            tc = temp_c()
            if tc > HOT:
                while temp_c() > COOL: time.sleep(1.0)
            if t % 400 == 0: print(f"  step {t}/{L} temp={tc:.0f}C ({time.time()-t0:.0f}s)", flush=True)
    st.stop()
    Tn = np.tanh((T - med)/mad/8.0)
    return Tn, med, mad


def lag(x, k):
    y = np.zeros_like(x)
    if k > 0: y[k:] = x[:-k]
    return y if k > 0 else x.copy()


def rank_lin(X, y, tr, te, nc, rank):
    mu = X[tr].mean(0); sd = X[tr].std(0)+1e-9; Xz = (X-mu)/sd
    c = Xz[tr].mean(0)
    _, _, Vt = np.linalg.svd(Xz[tr]-c, full_matrices=False)
    r = min(rank, Vt.shape[0]); Xp = (Xz-c) @ Vt[:r].T; Y = np.eye(nc)[y]; best = 0.0
    for al in [1e-2, .1, 1, 10, 100]:
        Wt = np.linalg.solve(Xp[tr].T@Xp[tr] + al*np.eye(Xp.shape[1]), Xp[tr].T@Y[tr])
        best = max(best, float(np.mean((Xp[te]@Wt).argmax(1) == y[te])))
    return best


def full_lin(X, y, tr, te, nc):
    mu = X[tr].mean(0); sd = X[tr].std(0)+1e-9; X = (X-mu)/sd; Y = np.eye(nc)[y]; best = 0.0
    for al in [.1, 1, 10, 100, 1e3]:
        Wt = np.linalg.solve(X[tr].T@X[tr] + al*np.eye(X.shape[1]), X[tr].T@Y[tr])
        best = max(best, float(np.mean((X[te]@Wt).argmax(1) == y[te])))
    return best


def feat_lags(M, nl=4):
    return np.hstack([np.vstack([lag(M[:, c], k) for k in range(nl+1)]).T for c in range(M.shape[1])])


def analyze(u, v, Tn, tag=""):
    L = len(u); flat = Tn.reshape(L, -1)
    Xdie = np.hstack([flat, lag(flat, 1), lag(flat, 2)])
    import itertools
    uu = u.astype(float)
    u_lin = np.stack([lag(uu, k) for k in range(1, 5)], 1)
    u_quad_cols = [lag(uu, k) for k in range(1, 5)]
    for a, b in itertools.combinations_with_replacement(range(1, 5), 2): u_quad_cols.append(lag(uu, a)*lag(uu, b))
    u_quad = np.stack(u_quad_cols, 1)
    # surrogate die
    rng = np.random.default_rng(7); Msur = np.zeros_like(flat)
    for c in range(flat.shape[1]):
        F = np.fft.rfft(flat[:, c]-flat[:, c].mean()); ph = rng.uniform(0, 2*np.pi, len(F)); ph[0] = 0
        Msur[:, c] = np.fft.irfft(np.abs(F)*np.exp(1j*ph), n=L)
    Xsur = np.hstack([Msur, lag(Msur, 1), lag(Msur, 2)])
    n = L-WASHOUT; cut = WASHOUT+int(0.7*n); tr = slice(WASHOUT, cut); te = slice(cut, L)

    def lbu(k): return lag(u, k).astype(int)
    def lbv(k): return lag(v, k).astype(int)
    tasks = {"RECALL_u1": lbu(1), "RECALL_v1": lbv(1),
             "XOR_u1v1": lbu(1) ^ lbv(1), "XOR_u1v2": lbu(1) ^ lbv(2),
             "AND_u1v1": (lbu(1) & lbv(1))}
    out = {}
    for nm, y in tasks.items():
        die_r = rank_lin(Xdie, y, tr, te, 2, R)
        die_f = full_lin(Xdie, y, tr, te, 2)
        u_l = full_lin(u_lin, y, tr, te, 2)
        u_q = full_lin(u_quad, y, tr, te, 2)
        sur = rank_lin(Xsur, y, tr, te, 2, R)
        u_only = max(u_l, u_q)
        win = nm.startswith("XOR") and (die_r - sur > 0.05) and (die_r > 0.55) and (u_only < 0.55)
        out[nm] = {"die_r4": die_r, "die_full": die_f, "u_only_poly": u_only, "surrogate": sur, "GATE_WIN": bool(win)}
        print(f"  [{tag}] {nm:10s} die(r4)={die_r:.3f} die(full)={die_f:.3f} u_only={u_only:.3f} surr={sur:.3f}"
              f"{'  <-- GATE WIN' if win else ''}", flush=True)
    return out


def main():
    rng_u = np.random.default_rng(SEED_U); u = rng_u.integers(0, 2, size=L)
    rng_v = np.random.default_rng(SEED_V); v = rng_v.integers(0, 2, size=L)
    print(f"[{HOST}] CROSS-DIE exogenous-mixing probe (u=GPU,v=CPU, v withheld from control) temp {temp_c():.0f}C", flush=True)
    Tn, med, mad = collect(u, v)
    flat = Tn.reshape(L, -1)
    du = np.abs((flat[u == 1].mean(0)-flat[u == 0].mean(0))/(np.sqrt((flat[u == 1].std(0)**2+flat[u == 0].std(0)**2)/2)+1e-9)).max()
    dv = np.abs((flat[v == 1].mean(0)-flat[v == 0].mean(0))/(np.sqrt((flat[v == 1].std(0)**2+flat[v == 0].std(0)**2)/2)+1e-9)).max()
    print(f"  drive landed: max|d_u|={du:.2f}  max|d_v|={dv:.2f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT/f"cross_die_mixing_raw_{HOST}.npz", u=u, v=v, Tn=Tn, med=med, mad=mad)
    res = analyze(u, v, Tn, tag=HOST)
    gate = any(r["GATE_WIN"] for r in res.values())
    out = {"host": HOST, "d_u": float(du), "d_v": float(dv), "ntap": NTAP, "seed_u": SEED_U, "seed_v": SEED_V,
           "results": res, "within_die_gate_pass": bool(gate),
           "verdict": ("WITHIN-DIE GATE PASS — die mixes u,v -> proceed to daedalus cross-die"
                       if gate else "within-die gate FAIL — die cannot read out u*v mixing; reframe also negative")}
    def jf(o):
        if isinstance(o, dict): return {k: jf(x) for k, x in o.items()}
        if isinstance(o, (np.floating, np.integer, np.bool_)): return float(o)
        return o
    (OUT/f"cross_die_mixing_{HOST}.json").write_text(json.dumps(jf(out), indent=2))
    print(f"\n>>> {out['verdict']}", flush=True)


if __name__ == "__main__":
    main()
