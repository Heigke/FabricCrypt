"""H7-RES-X Step 0 — device-level reservoir-capacity gate (NO language model).

Tests the ONE thing that must be true before any LLM coupling is worth it:
does the LIVE gfx1151 SoC substrate act as a genuine physical reservoir — i.e. does its
fading-memory response to an injected drive `u` let a LINEAR readout compute a
NONLINEAR-TEMPORAL function of u that a linear-on-u baseline canNOT?

This is the standard reservoir-computing necessity test (non-circular: the target is a
function of the DRIVE u, not of the state window the readout reads — so a rich feature
basis cannot trivially fit it; it must have done real temporal computation).

  drive u_t (pseudo-random)  -->  modulate compute load for ~dt  -->  die responds
  read 10-ch substrate sample each step  -->  states S (L x 10)
  reservoir features F = build_best_features(S)            (temporal products = fading memory)
  target  y_t = 4-bit delayed-PARITY code of u  -> 16 classes (chance 6.25%)
     bit b = parity(u_{t-a_b}, u_{t-b_b})  -- linear-on-u CANNOT compute parity
  readout: multiclass ridge  F -> y         (the die's contribution)
  baseline: multiclass ridge  U_lags -> y   (no die; linear in u)

PASS (worth coupling an LLM) iff:  reservoir_acc >= 0.70  AND  baseline_acc <= 0.30.
A near-chance reservoir is the honest "too little bandwidth" negative — we STOP and report it.

Run as root, HSA_OVERRIDE_GFX_VERSION=11.0.0, under thermal_watchdog.sh (setsid).
"""
from __future__ import annotations
import sys, json, time, socket
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from substrate_realtime_v3 import SubstrateStateV3
from z2296_best_of_all import build_best_features

HOST = socket.gethostname()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/IDENTITY_H7_2026-06-09"
ZONE = Path("/sys/class/thermal/thermal_zone0/temp")
N_CH = 10
L = 1500                  # drive steps
WASHOUT = 150
DWELL_S = 0.012           # hold each drive step ~12ms so the 500Hz substrate resolves it (~6 samples)
BURST_N = 1024            # matmul size; u=1 -> busy-loop heavy load for the dwell, u=0 -> idle
SEED = 0
PARITY_LAGS = [(1, 3), (2, 6), (4, 9), (5, 12)]   # 4 bits -> 16 classes; need real memory
U_LAG_BASELINE = list(range(1, 16))               # linear-on-u baseline sees these lags


def temp_c():
    try: return int(ZONE.read_text()) / 1000.0
    except Exception: return 0.0


def robust_norm_stats(pool):
    med = np.median(pool, axis=0)
    mad = np.median(np.abs(pool - med), axis=0) * 1.4826 + 1e-9
    return med, mad


def norm_window(w, med, mad):
    return np.tanh((w - med) / mad / 8.0)   # match H7 GlobalNorm clamp scale


def multiclass_ridge(F_tr, y_tr, F_te, y_te, n_cls=16):
    """One-hot ridge with alpha sweep; returns best test accuracy."""
    mu = F_tr.mean(0); sd = F_tr.std(0) + 1e-8
    F_tr = (F_tr - mu) / sd; F_te = (F_te - mu) / sd
    Ytr = np.eye(n_cls)[y_tr]
    best = 0.0
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0, 1e4]:
        I = np.eye(F_tr.shape[1])
        try:
            W = np.linalg.solve(F_tr.T @ F_tr + alpha * I, F_tr.T @ Ytr)
            pred = (F_te @ W).argmax(1)
            acc = float(np.mean(pred == y_te))
            best = max(best, acc)
        except Exception:
            pass
    return best


def parity_target(u_bits):
    """y_t = 4-bit code, bit b = XOR(u_{t-a}, u_{t-b}). 16 classes."""
    L = len(u_bits)
    y = np.zeros(L, dtype=int)
    for b, (a, c) in enumerate(PARITY_LAGS):
        ua = np.zeros(L, dtype=int); ua[a:] = u_bits[:-a]
        uc = np.zeros(L, dtype=int); uc[c:] = u_bits[:-c]
        y |= ((ua ^ uc) << b)
    return y


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(SEED)
    u_bits = rng.integers(0, 2, size=L)            # binary drive
    state = SubstrateStateV3(hz_target=500); state.start()
    print(f"[{HOST}] device={device} warmup 6s (temp {temp_c():.0f}C)...", flush=True)
    time.sleep(6.0)

    # robust norm stats from a quiet warmup pool
    pool = np.array([state.latest_window(length=64).reshape(-1, N_CH) for _ in range(40)]).reshape(-1, N_CH)
    med, mad = robust_norm_stats(pool)

    a = torch.randn(BURST_N, BURST_N, device=device)
    b = torch.randn(BURST_N, BURST_N, device=device)
    S = np.zeros((L, N_CH), dtype=np.float32)
    t_start = time.time()
    for t in range(L):
        # LIGHT drive: u=1 -> a SHORT burst (few matmuls ~2ms) then idle the rest of the dwell;
        # u=0 -> idle the whole dwell. Low duty cycle => clear power swing WITHOUT runaway heat.
        if u_bits[t]:
            for _ in range(3):
                a = (a @ b).tanh() * 0.5 + 0.5
            if device == "cuda":
                torch.cuda.synchronize()
        time.sleep(DWELL_S)                          # let the 500Hz substrate resolve this step
        S[t] = state.latest_window(length=6).mean(axis=0)
        # RELIABLE in-loop self-guard (do NOT depend on the racy external watchdog): check often.
        if t % 20 == 0:
            tc = temp_c()
            if t and t % 500 == 0:
                print(f"  step {t}/{L}  temp={tc:.0f}C  elapsed={time.time()-t_start:.0f}s", flush=True)
            if tc > 82.0:
                print(f"  [self-guard] {tc:.0f}C -> cooling", flush=True)
                while temp_c() > 62.0: time.sleep(1.0)
    state.stop()
    print(f"  drive done: {L} steps in {time.time()-t_start:.0f}s (should be ~{L*DWELL_S:.0f}s+)", flush=True)

    Sn = norm_window(S, med, mad)
    # DRIVE-LANDED diagnostic: does the load bit actually move each channel? (per-channel Cohen's d, u=1 vs u=0)
    m1 = Sn[u_bits == 1]; m0 = Sn[u_bits == 0]
    psd = np.sqrt((m1.std(0) ** 2 + m0.std(0) ** 2) / 2) + 1e-9
    drive_d = ((m1.mean(0) - m0.mean(0)) / psd).tolist()
    print(f"  drive-landed Cohen's d per channel: {[round(x,2) for x in drive_d]}", flush=True)
    dspikes = np.abs(np.vstack([np.zeros((1, N_CH)), np.diff(Sn, axis=0)]))
    F = build_best_features(Sn, dspikes)            # (L, big)  fading-memory temporal products
    y = parity_target(u_bits)

    # also a simple Memory-Capacity sanity (linear recall of u_{t-k}) for context
    sl = slice(WASHOUT, L)
    n = L - WASHOUT
    cut = WASHOUT + int(0.7 * n)
    tr = slice(WASHOUT, cut); te = slice(cut, L)

    res_acc = multiclass_ridge(F[tr], y[tr], F[te], y[te])
    # baseline: linear-on-u lag features (NO die)
    U = np.zeros((L, len(U_LAG_BASELINE)), dtype=np.float32)
    for j, k in enumerate(U_LAG_BASELINE):
        U[k:, j] = u_bits[:-k]
    base_acc = multiclass_ridge(U[tr], y[tr], U[te], y[te])

    # --- nonlinear-capacity SUITE (characterize how much nonlinear-temporal compute exists) ---
    def lagbit(k):
        x = np.zeros(L, dtype=int); x[k:] = u_bits[:-k]; return x
    suite = {}
    tasks = {
        "XOR_t1t2": (lagbit(1) ^ lagbit(2), 2),
        "XOR_t2t5": (lagbit(2) ^ lagbit(5), 2),
        "PAR_2bit": ((lagbit(1) ^ lagbit(3)) | ((lagbit(2) ^ lagbit(6)) << 1), 4),
        "PAR_4bit": (y, 16),
        "RECALL_t3": (lagbit(3), 2),          # linear sanity: pure recall (die SHOULD do this)
    }
    for name, (yt, nc) in tasks.items():
        suite[name] = {
            "chance": 1.0 / nc,
            "reservoir": multiclass_ridge(F[tr], yt[tr], F[te], yt[te], n_cls=nc),
            "baseline_u": multiclass_ridge(U[tr], yt[tr], U[te], yt[te], n_cls=nc),
        }
        print(f"  task {name:10s} chance={1.0/nc:.3f}  reservoir={suite[name]['reservoir']:.3f}  base_u={suite[name]['baseline_u']:.3f}", flush=True)

    # memory-capacity: how many past bits are linearly recoverable from die states
    mc = 0.0
    for k in range(1, 25):
        uk = np.zeros(L); uk[k:] = u_bits[:-k]
        # binary recall accuracy from reservoir features
        acc_k = multiclass_ridge(F[tr], uk[tr].astype(int), F[te], uk[te].astype(int), n_cls=2)
        mc += max(0.0, 2 * (acc_k - 0.5))           # normalized recall

    verdict = "PASS" if (res_acc >= 0.70 and base_acc <= 0.30) else "FAIL"
    res = {"host": HOST, "L": L, "chance": 1/16, "parity_lags": PARITY_LAGS,
           "reservoir_acc": res_acc, "baseline_linear_on_u_acc": base_acc,
           "memory_capacity_bits": mc, "n_features": int(F.shape[1]),
           "drive_landed_cohen_d": drive_d, "task_suite": suite,
           "verdict": verdict, "pass_rule": "reservoir>=0.70 AND baseline<=0.30"}
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"resx_step0_gate_{HOST}.json"; out.write_text(json.dumps(res, indent=2))
    print("\n=== H7-RES-X STEP 0 GATE ===", flush=True)
    print(f"  reservoir (die) acc : {res_acc:.3f}   (chance 0.0625, want >=0.70)")
    print(f"  baseline lin-on-u   : {base_acc:.3f}   (want <=0.30)")
    print(f"  memory capacity     : {mc:.2f} bits")
    print(f"  >>> {verdict}  (saved {out})")
    if verdict == "FAIL":
        print("  HONEST NEGATIVE: live SoC substrate lacks the nonlinear-temporal bandwidth to")
        print("  carry computation an LLM could be made to NEED. Do NOT proceed to LM coupling.")


if __name__ == "__main__":
    main()
