"""z226 — GPU reservoir port + NARMA-10 at large N.

Per user 2026-05-08 ambition: "millions of cells" target. This is the
first real GPU reservoir using the 4D body-state surrogate. Tests:
  N=200  CPU baseline (z222 best, NRMSE ~0.62)
  N=200  GPU (verify match)
  N=2000 GPU  (10× scale)
  N=10000 GPU (50× scale, target territory)

Block-diagonal W (block_size=200). Frozen z222 hyperparams.

If N=10k gives NRMSE < 0.55, we have ESN-class with a 50× scale-up
path that demonstrates GPU + 4D surrogate work end-to-end.
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
OUT = ROOT / "results/z226_gpu_reservoir"; OUT.mkdir(parents=True, exist_ok=True)
SURR_PATH = ROOT / "results/z220_4d_dense/surrogate_4d_dense.npz"


class GPUSurrogate4D:
    """torch.cuda port of NSRAMSurrogate4D quadrilinear interp."""
    def __init__(self, path, device="cuda", dtype=torch.float32):
        d = np.load(path)
        self.device = device
        self.dtype = dtype
        self.Id_log = torch.tensor(np.log10(np.maximum(np.abs(d["Id"]), 1e-15)),
                                     dtype=dtype, device=device)
        self.Iii = torch.tensor(d["Iii"], dtype=dtype, device=device)
        self.Ileak = torch.tensor(d["Ileak"], dtype=dtype, device=device)
        self.vg1_axis = torch.tensor(d["vg1_axis"], dtype=dtype, device=device)
        self.vg2_axis = torch.tensor(d["vg2_axis"], dtype=dtype, device=device)
        self.vd_axis = torch.tensor(d["vd_axis"], dtype=dtype, device=device)
        self.vb_axis = torch.tensor(d["vb_axis"], dtype=dtype, device=device)

    def _idx(self, x, axis):
        x = x.clamp(axis[0], axis[-1])
        i = torch.searchsorted(axis, x).clamp(1, len(axis) - 1) - 1
        f = (x - axis[i]) / (axis[i+1] - axis[i]).clamp_min(1e-30)
        return i, f

    def eval(self, VG1, VG2, Vd, Vb):
        i, fi = self._idx(VG1, self.vg1_axis)
        j, fj = self._idx(VG2, self.vg2_axis)
        k, fk = self._idx(Vd, self.vd_axis)
        l, fl = self._idx(Vb, self.vb_axis)

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


def make_block_diag_W(N, block_size, density=0.10, dtype=torch.float32, device="cuda"):
    """Block-diagonal W. Each block density-sparse, spectral-normalized."""
    K = N // block_size
    blocks_indptr = [0]
    blocks_cols = []
    blocks_vals = []
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
            blocks_cols.extend(cs.tolist())
            blocks_vals.extend(w[i, w[i] != 0].tolist())
            blocks_indptr.append(len(blocks_cols))
    indptr = torch.tensor(blocks_indptr, dtype=torch.int64, device=device)
    cols = torch.tensor(blocks_cols, dtype=torch.int64, device=device)
    vals = torch.tensor(blocks_vals, dtype=dtype, device=device)
    return torch.sparse_csr_tensor(indptr, cols, vals, size=(N, N), device=device)


def gen_narma10_torch(T, seed, device, dtype=torch.float32):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, T)
    y = np.zeros(T)
    for k in range(10, T-1):
        y[k+1] = 0.3*y[k] + 0.05*y[k]*y[k-9:k+1].sum() + 1.5*u[k-9]*u[k] + 0.1
    return (torch.tensor(u, dtype=dtype, device=device),
            torch.tensor(y, dtype=dtype, device=device))


def run_narma_gpu(N, seed, device="cuda", block_size=200,
                    Cb=5e-15, dt=5e-7, g_VG2=0.05, g_VG1=0.3, leak=0.30,
                    T_total=1500, washout=300, T_train=1000):
    """End-to-end NARMA-10 reservoir on GPU at N cells. Returns NRMSE."""
    surr = GPUSurrogate4D(SURR_PATH, device=device)
    rng = np.random.default_rng(seed)
    base_VG1 = torch.tensor(rng.uniform(0.2, 0.5, N).astype(np.float32), device=device)
    base_VG2 = torch.tensor(rng.uniform(0.05, 0.55, N).astype(np.float32), device=device)
    sign_mask = torch.tensor(rng.choice([-1.0, 1.0], N).astype(np.float32), device=device)
    W_in = torch.tensor(rng.normal(0, 1.0, N).astype(np.float32), device=device)
    W = make_block_diag_W(N, block_size, density=0.10, device=device)

    u, y = gen_narma10_torch(T_total, seed, device)
    u_input = (u - 0.25) / 0.25

    Vd_arr = torch.ones(N, dtype=torch.float32, device=device)
    Vb = torch.full((N,), 0.30, dtype=torch.float32, device=device)
    feat = torch.zeros(N, dtype=torch.float32, device=device)
    state = torch.zeros((T_total, N), dtype=torch.float32, device=device)

    t0 = time.time()
    for t in range(T_total):
        VG2 = (base_VG2 + g_VG2 * W_in * u_input[t]).clamp(0.0, 0.6)
        rec = (W @ feat) * sign_mask
        VG1 = (base_VG1 + g_VG1 * rec).clamp(0.05, 0.7)
        log_Id, Iii, Ileak = surr.eval(VG1, VG2, Vd_arr, Vb)
        net = Iii - Ileak
        Vb = (Vb + dt * net / Cb).clamp(0.0, 0.7)
        feat = (1.0 - leak) * feat + leak * log_Id
        state[t] = feat
    if device == "cuda":
        torch.cuda.synchronize()
    enc_wall = time.time() - t0

    X = state.cpu().numpy()
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    y_np = y.cpu().numpy()
    Xt = X[washout:T_train]; yt = y_np[washout:T_train]
    Xv = X[T_train:]; yv = y_np[T_train:]
    # Use float64 for ridge solve at large N
    Xt64 = Xt.astype(np.float64)
    XtX = Xt64.T @ Xt64 + 1e-4 * np.eye(X.shape[1])
    Xty = Xt64.T @ yt.astype(np.float64)
    w = np.linalg.solve(XtX, Xty)
    pred_v = (Xv.astype(np.float64)) @ w
    test_nrmse = float(np.sqrt(((pred_v - yv) ** 2).mean()) / yv.std())
    return {"N": N, "seed": seed, "test_nrmse": test_nrmse, "enc_wall_s": enc_wall}


def main():
    print(f"=== GPU reservoir + NARMA-10 ===")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"\n{'N':>7}  {'seed':>4}  {'NRMSE':>7}  {'enc_wall':>9}  {'APU':>4}")

    out = []
    Ns = [200, 2000, 10000]
    for N in Ns:
        try:
            apu_pre = (lambda: int(open("/sys/class/thermal/thermal_zone0/temp").read())/1000.0)()
            if apu_pre > 70:
                print(f"  APU {apu_pre:.0f}°C — pausing 30s")
                time.sleep(30)
            block_size = min(200, N) if N >= 200 else N
            r = run_narma_gpu(N, seed=0, block_size=block_size)
            apu = (lambda: int(open("/sys/class/thermal/thermal_zone0/temp").read())/1000.0)()
            print(f"  {N:>5}    {0:>3}    {r['test_nrmse']:.4f}  "
                  f"{r['enc_wall_s']:>7.1f}s  {apu:>3.0f}°C")
            out.append({**r, "apu": apu})
        except Exception as e:
            print(f"  N={N} failed: {e}")

    print(f"\nReference: z223 30-seed CPU (N=200) NRMSE = 0.6122 ± 0.030")
    (OUT / "summary.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
