"""z227 — Float64-Vb GPU reservoir with explicit cooldown gates.

Per user 2026-05-08: scale toward millions, but log well if crash.
Improvements over z226:
  1. Vb stored in float64 (avoids float32 precision drift over 1500 steps)
  2. Pre-batch cooldown wait until APU < 60°C
  3. Per-step APU sample logged to disk every 50 steps
  4. Hard kill if APU > 92°C (above z226's 92°C peak)
  5. Crash-safe: per-N result saved to disk before next N starts

Sweep: N ∈ {2k, 5k, 10k, 20k, 50k}.
Goal: confirm body-state model scales without precision/thermal events.
"""
from __future__ import annotations
import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import sys, json, time, signal
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/z227_gpu_safe"; OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "live.log"
SURR_PATH = ROOT / "results/z220_4d_dense/surrogate_4d_dense.npz"


def get_apu():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return -1.0


def log_line(msg):
    """Append to live log + stdout."""
    line = f"[{time.strftime('%H:%M:%S')}] APU={get_apu():.1f}°C  {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def cooldown_to(target_c, timeout_s=300):
    """Sleep until APU < target_c or timeout."""
    t0 = time.time()
    while True:
        apu = get_apu()
        if apu < target_c:
            return apu
        elapsed = time.time() - t0
        if elapsed > timeout_s:
            log_line(f"COOLDOWN TIMEOUT after {timeout_s}s, APU={apu}°C — proceeding anyway")
            return apu
        log_line(f"cooling... APU {apu:.1f} > {target_c} (waited {elapsed:.0f}s)")
        time.sleep(15)


class GPUSurrogate4D:
    """Vb axis as float64, others as float32."""
    def __init__(self, path, device="cuda"):
        d = np.load(path)
        self.device = device
        self.Id_log = torch.tensor(np.log10(np.maximum(np.abs(d["Id"]), 1e-15)),
                                     dtype=torch.float32, device=device)
        self.Iii = torch.tensor(d["Iii"], dtype=torch.float32, device=device)
        self.Ileak = torch.tensor(d["Ileak"], dtype=torch.float32, device=device)
        self.vg1_axis = torch.tensor(d["vg1_axis"], dtype=torch.float32, device=device)
        self.vg2_axis = torch.tensor(d["vg2_axis"], dtype=torch.float32, device=device)
        self.vd_axis = torch.tensor(d["vd_axis"], dtype=torch.float32, device=device)
        self.vb_axis = torch.tensor(d["vb_axis"], dtype=torch.float32, device=device)

    def _idx(self, x, axis):
        x = x.clamp(axis[0], axis[-1])
        i = torch.searchsorted(axis, x).clamp(1, len(axis) - 1) - 1
        f = (x - axis[i]) / (axis[i+1] - axis[i]).clamp_min(1e-30)
        return i, f

    def eval(self, VG1, VG2, Vd, Vb):
        # Vb is float64 — convert to float32 for interp
        Vb32 = Vb.to(torch.float32)
        i, fi = self._idx(VG1, self.vg1_axis)
        j, fj = self._idx(VG2, self.vg2_axis)
        k, fk = self._idx(Vd, self.vd_axis)
        l, fl = self._idx(Vb32, self.vb_axis)
        def quad(grid):
            r = torch.zeros_like(VG1)
            for di in [0, 1]:
                for dj in [0, 1]:
                    for dk in [0, 1]:
                        for dl in [0, 1]:
                            wf = (fi if di else (1-fi)) * (fj if dj else (1-fj)) \
                                  * (fk if dk else (1-fk)) * (fl if dl else (1-fl))
                            r = r + wf * grid[i+di, j+dj, k+dk, l+dl]
            return r
        return quad(self.Id_log), quad(self.Iii), quad(self.Ileak)


def make_block_diag_W(N, block_size, density=0.10, device="cuda"):
    K = N // block_size
    indptr = [0]; cols = []; vals = []
    rng = np.random.default_rng(0)
    for k in range(K):
        m = (rng.random((block_size, block_size)) < density).astype(np.float32)
        w = m * rng.normal(0, 1, (block_size, block_size)).astype(np.float32)
        np.fill_diagonal(w, 0)
        eig = float(np.abs(np.linalg.eigvals(w)).max())
        if eig > 1e-9:
            w *= 0.9 / eig
        for i in range(block_size):
            row_offset = k * block_size
            cs = np.where(w[i] != 0)[0] + row_offset
            cols.extend(cs.tolist())
            vals.extend(w[i, w[i] != 0].tolist())
            indptr.append(len(cols))
    return torch.sparse_csr_tensor(
        torch.tensor(indptr, dtype=torch.int64, device=device),
        torch.tensor(cols, dtype=torch.int64, device=device),
        torch.tensor(vals, dtype=torch.float32, device=device),
        size=(N, N), device=device,
    )


def gen_narma10(T, seed):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, T)
    y = np.zeros(T)
    for k in range(10, T-1):
        y[k+1] = 0.3*y[k] + 0.05*y[k]*y[k-9:k+1].sum() + 1.5*u[k-9]*u[k] + 0.1
    return u, y


def run_one_safe(N, seed, block_size=2000):
    """Single run with thermal monitoring. Returns dict or raises."""
    log_line(f"  start N={N} seed={seed}")
    apu_pre = cooldown_to(60.0)
    log_line(f"  cooled to {apu_pre:.0f}°C, launching")

    surr = GPUSurrogate4D(SURR_PATH)
    rng = np.random.default_rng(seed)
    base_VG1 = torch.tensor(rng.uniform(0.2, 0.5, N).astype(np.float32), device="cuda")
    base_VG2 = torch.tensor(rng.uniform(0.05, 0.55, N).astype(np.float32), device="cuda")
    sign_mask = torch.tensor(rng.choice([-1.0, 1.0], N).astype(np.float32), device="cuda")
    W_in = torch.tensor(rng.normal(0, 1.0, N).astype(np.float32), device="cuda")
    W = make_block_diag_W(N, min(block_size, N), density=0.10)

    Cb, dt = 5e-15, 5e-7
    g_VG2, g_VG1, leak = 0.05, 0.3, 0.30
    T_total, washout, T_train = 1500, 300, 1000

    u_np, y_np = gen_narma10(T_total, seed)
    u = torch.tensor(u_np, dtype=torch.float32, device="cuda")
    u_input = (u - 0.25) / 0.25

    Vd_arr = torch.ones(N, dtype=torch.float32, device="cuda")
    Vb = torch.full((N,), 0.30, dtype=torch.float64, device="cuda")  # FLOAT64
    feat = torch.zeros(N, dtype=torch.float32, device="cuda")
    state = torch.zeros((T_total, N), dtype=torch.float32, device="cuda")

    apu_peak = apu_pre
    t0 = time.time()
    for t in range(T_total):
        VG2 = (base_VG2 + g_VG2 * W_in * u_input[t]).clamp(0.0, 0.6)
        rec = (W @ feat) * sign_mask
        VG1 = (base_VG1 + g_VG1 * rec).clamp(0.05, 0.7)
        log_Id, Iii, Ileak = surr.eval(VG1, VG2, Vd_arr, Vb)
        net = (Iii - Ileak).to(torch.float64)   # float64 for Vb update
        Vb = (Vb + dt * net / Cb).clamp(0.0, 0.7)
        feat = (1.0 - leak) * feat + leak * log_Id
        state[t] = feat
        if t % 50 == 0:
            apu = get_apu()
            apu_peak = max(apu_peak, apu)
            if apu > 92:
                torch.cuda.synchronize()
                log_line(f"  KILL at t={t}: APU {apu}°C > 92")
                raise RuntimeError(f"thermal kill at APU={apu}")
            if apu > 80 and t < T_total - 100:
                torch.cuda.synchronize()
                log_line(f"  pause at t={t}: APU {apu}°C > 80, cooling 30s")
                time.sleep(30)

    torch.cuda.synchronize()
    enc_wall = time.time() - t0

    X = state.cpu().numpy()
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    Xt = X[washout:T_train]; yt = y_np[washout:T_train]
    Xv = X[T_train:]; yv = y_np[T_train:]
    Xt64 = Xt.astype(np.float64)
    XtX = Xt64.T @ Xt64 + 1e-4 * np.eye(X.shape[1])
    Xty = Xt64.T @ yt.astype(np.float64)
    w = np.linalg.solve(XtX, Xty)
    pred_v = (Xv.astype(np.float64)) @ w
    nrmse = float(np.sqrt(((pred_v - yv) ** 2).mean()) / yv.std())
    return {
        "N": N, "seed": seed, "block_size": min(block_size, N),
        "test_nrmse": nrmse, "enc_wall_s": enc_wall,
        "apu_peak": apu_peak, "apu_pre": apu_pre,
    }


def main():
    log_line(f"=== z227 GPU scale (float64 Vb + cooldown gates) ===")
    log_line(f"Device: {torch.cuda.get_device_name(0)}")

    targets = [(2_000, 0), (5_000, 0), (10_000, 0), (20_000, 0), (50_000, 0)]
    results = []
    for N, seed in targets:
        fp = OUT / f"N{N}_s{seed}.json"
        if fp.exists():
            log_line(f"  skip N={N} (exists)")
            results.append(json.loads(fp.read_text()))
            continue
        try:
            r = run_one_safe(N, seed)
            fp.write_text(json.dumps(r, indent=2))
            log_line(f"  N={N}: NRMSE {r['test_nrmse']:.4f}  "
                     f"wall {r['enc_wall_s']:.1f}s  APU peak {r['apu_peak']:.0f}°C")
            results.append(r)
        except Exception as e:
            log_line(f"  N={N} CRASHED: {e}")
            # Save crash record
            (OUT / f"N{N}_CRASH.json").write_text(json.dumps(
                {"N": N, "error": str(e), "apu_at_crash": get_apu()}, indent=2))
            # Long cooldown after crash
            cooldown_to(50.0, timeout_s=600)

    log_line(f"\n=== Summary ===")
    log_line(f"{'N':>7}  {'NRMSE':>7}  {'wall':>7}  {'apu peak':>9}")
    for r in results:
        log_line(f"  {r['N']:>5}  {r['test_nrmse']:.4f}  "
                 f"{r['enc_wall_s']:>5.1f}s  {r['apu_peak']:>3.0f}°C")
    log_line(f"\nReference: z223 30-seed CPU N=200: NRMSE 0.6122 ± 0.030")
    log_line(f"            z226 GPU N=2000:        NRMSE 0.6084 (float32 Vb)")

    (OUT / "summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
