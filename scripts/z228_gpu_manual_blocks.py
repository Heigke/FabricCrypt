"""z228 — Manual block-loop GPU reservoir to bypass sparse_csr crash.

z227 hit ROCm "Page not present" GPU memory fault at N=5000 inside
PyTorch's sparse_csr matmul. This script replaces that single sparse
matmul with K independent DENSE matmuls (one per block), keeping the
4D body-state surrogate identical. Memory: K * n_block^2 * 4B.

For n_block=500, K=10 → N=5000:  10 × 1MB = 10MB W
For n_block=500, K=20 → N=10000: 20 × 1MB = 20MB W

If z228 completes N=5000 cleanly, sparse_csr is confirmed culprit and
the path to N≥50k via this layout is open.
"""
from __future__ import annotations
import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import sys, json, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/z228_manual_blocks"; OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "live.log"
SURR_PATH = ROOT / "results/z220_4d_dense/surrogate_4d_dense.npz"


def get_apu():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return -1.0


def log_line(msg):
    line = f"[{time.strftime('%H:%M:%S')}] APU={get_apu():.1f}°C  {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def cooldown_to(target_c, timeout_s=300):
    t0 = time.time()
    while True:
        apu = get_apu()
        if apu < target_c:
            return apu
        if time.time() - t0 > timeout_s:
            return apu
        log_line(f"cooling... APU {apu:.1f} > {target_c}")
        time.sleep(15)


class GPUSurrogate4D:
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


def make_block_dense(N, n_block, density=0.10, device="cuda"):
    """K dense block tensors, batched as (K, n, n)."""
    K = N // n_block
    rng = np.random.default_rng(0)
    blocks = np.zeros((K, n_block, n_block), dtype=np.float32)
    for k in range(K):
        m = (rng.random((n_block, n_block)) < density).astype(np.float32)
        w = m * rng.normal(0, 1, (n_block, n_block)).astype(np.float32)
        np.fill_diagonal(w, 0)
        eig = float(np.abs(np.linalg.eigvals(w)).max())
        if eig > 1e-9:
            w *= 0.9 / eig
        blocks[k] = w
    return torch.tensor(blocks, dtype=torch.float32, device=device), K, n_block


def gen_narma10(T, seed):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, T)
    y = np.zeros(T)
    for k in range(10, T-1):
        y[k+1] = 0.3*y[k] + 0.05*y[k]*y[k-9:k+1].sum() + 1.5*u[k-9]*u[k] + 0.1
    return u, y


def run(N, seed=0, n_block=500):
    log_line(f"  start N={N} n_block={n_block} K={N//n_block}")
    cooldown_to(60.0)

    surr = GPUSurrogate4D(SURR_PATH)
    rng = np.random.default_rng(seed)
    base_VG1 = torch.tensor(rng.uniform(0.2, 0.5, N).astype(np.float32), device="cuda")
    base_VG2 = torch.tensor(rng.uniform(0.05, 0.55, N).astype(np.float32), device="cuda")
    sign_mask = torch.tensor(rng.choice([-1.0, 1.0], N).astype(np.float32), device="cuda")
    W_in = torch.tensor(rng.normal(0, 1.0, N).astype(np.float32), device="cuda")
    Wb, K, nb = make_block_dense(N, n_block)   # (K, nb, nb)

    Cb, dt = 5e-15, 5e-7
    g_VG2, g_VG1, leak = 0.05, 0.3, 0.30
    T_total, washout, T_train = 1500, 300, 1000

    u_np, y_np = gen_narma10(T_total, seed)
    u_input = torch.tensor((u_np - 0.25) / 0.25, dtype=torch.float32, device="cuda")

    Vd_arr = torch.ones(N, dtype=torch.float32, device="cuda")
    Vb = torch.full((N,), 0.30, dtype=torch.float64, device="cuda")
    feat = torch.zeros(N, dtype=torch.float32, device="cuda")
    state = torch.zeros((T_total, N), dtype=torch.float32, device="cuda")

    apu_peak = get_apu()
    t0 = time.time()
    for t in range(T_total):
        VG2 = (base_VG2 + g_VG2 * W_in * u_input[t]).clamp(0.0, 0.6)
        # Manual block matmul: feat reshape to (K, nb), batched matmul, flatten back
        feat_b = feat.view(K, nb)
        rec_b = torch.bmm(Wb, feat_b.unsqueeze(-1)).squeeze(-1)   # (K, nb)
        rec = rec_b.view(N) * sign_mask
        VG1 = (base_VG1 + g_VG1 * rec).clamp(0.05, 0.7)
        log_Id, Iii, Ileak = surr.eval(VG1, VG2, Vd_arr, Vb)
        net = (Iii - Ileak).to(torch.float64)
        Vb = (Vb + dt * net / Cb).clamp(0.0, 0.7)
        feat = (1.0 - leak) * feat + leak * log_Id
        state[t] = feat
        if t % 50 == 0:
            apu = get_apu()
            apu_peak = max(apu_peak, apu)
            if apu > 92:
                torch.cuda.synchronize()
                raise RuntimeError(f"thermal kill APU={apu}")
            if apu > 80 and t < T_total - 100:
                torch.cuda.synchronize()
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
    return {"N": N, "n_block": nb, "K": K, "seed": seed,
            "test_nrmse": nrmse, "enc_wall_s": enc_wall, "apu_peak": apu_peak}


def main():
    log_line(f"=== z228 manual block-loop GPU (bypasses sparse_csr) ===")
    log_line(f"Device: {torch.cuda.get_device_name(0)}")
    targets = [(2000, 500), (5000, 500), (10000, 500), (20000, 500)]
    results = []
    for N, nb in targets:
        fp = OUT / f"N{N}_nb{nb}.json"
        if fp.exists():
            log_line(f"  skip N={N} (exists)")
            results.append(json.loads(fp.read_text()))
            continue
        try:
            r = run(N, n_block=nb)
            fp.write_text(json.dumps(r, indent=2))
            log_line(f"  N={N}: NRMSE {r['test_nrmse']:.4f}  "
                     f"wall {r['enc_wall_s']:.1f}s  APU peak {r['apu_peak']:.0f}°C")
            results.append(r)
        except Exception as e:
            log_line(f"  N={N} CRASHED: {e}")
            (OUT / f"N{N}_CRASH.json").write_text(json.dumps(
                {"N": N, "error": str(e), "apu": get_apu()}, indent=2))
            cooldown_to(50.0, 600)
    (OUT / "summary.json").write_text(json.dumps(results, indent=2))
    log_line(f"\n=== Summary ===")
    for r in results:
        log_line(f"  N={r['N']:>5}  NRMSE {r['test_nrmse']:.4f}  "
                 f"wall {r['enc_wall_s']:>5.1f}s  apu {r['apu_peak']:>3.0f}°C")


if __name__ == "__main__":
    main()
