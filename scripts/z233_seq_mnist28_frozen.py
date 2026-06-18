"""z233 — 28×28 sequential MNIST with FROZEN z221/z223 NARMA-10 params.

Per O35 3-oracle consensus (2026-05-09): the single highest-value next
experiment is to test cross-task generalization of Path A's frozen
hyperparameters on a powered seq-MNIST 28×28 task. Goal:

  Acceptance: reservoir > pure-projection by ≥3 pp
  Statistical:  30 seeds (or as many as 40 min budget allows), 95% CI

FROZEN config (no re-tuning, copied from z221/z223):
  Cb = 5e-15 F
  dt = 5e-7 s
  g_VG2 = 0.05    (input gain)
  g_VG1 = 0.30    (recurrent gain)
  leak = 0.30
  base_VG1 ∈ U(0.2, 0.5)
  base_VG2 ∈ U(0.05, 0.55)
  N = 2000 (GPU-confirmed stable per z228)

Each MNIST image (28×28) is presented row-by-row over 28 timesteps.
Each row's 28 pixels project via W_in to N cells -> drives VG2.
Reservoir state collected; final-step state used for 10-class softmax.

Pure-projection baseline: same W_in projection, NO reservoir dynamics.
Paired comparison per seed.

Thermal-safe: per-batch APU check, hard kill at 92°C, pause at 80°C.
Crash-safe: per-seed JSON save to results/z233_seq_mnist28/.

Wall budget: ~12 min for 30 seeds at N=2000 (per-seed ~25s estimate).
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
OUT = ROOT / "results/z233_seq_mnist28"; OUT.mkdir(parents=True, exist_ok=True)
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


def cooldown_to(target_c, timeout_s=120):
    t0 = time.time()
    while True:
        apu = get_apu()
        if apu < target_c:
            return apu
        if time.time() - t0 > timeout_s:
            return apu
        log_line(f"cool... APU {apu:.1f} > {target_c}")
        time.sleep(15)


class GPUSurrogate4D:
    def __init__(self, path, device="cuda"):
        d = np.load(path)
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


def make_block_dense(N, n_block, density=0.10, seed=0, device="cuda"):
    """Block-diagonal W; same pattern as z228."""
    K = N // n_block
    rng = np.random.default_rng(seed)
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


def encode_images(images, surr, base_VG1, base_VG2, sign_mask, W_in, Wb, K, nb,
                   N, Cb=5e-15, dt=5e-7, g_VG2=0.05, g_VG1=0.30, leak=0.30):
    """Run reservoir over images (M, 28, 28) -> states (M, N).

    Each row of an image is projected to N via W_in (N×28) and drives VG2.
    Vb evolves; final reservoir state used.
    """
    M = images.shape[0]
    Vd_arr = torch.ones(N, dtype=torch.float32, device="cuda")
    states = torch.zeros((M, N), dtype=torch.float32, device="cuda")

    img_t = torch.tensor(images, dtype=torch.float32, device="cuda")  # (M, 28, 28)

    for m in range(M):
        Vb = torch.full((N,), 0.30, dtype=torch.float64, device="cuda")
        feat = torch.zeros(N, dtype=torch.float32, device="cuda")
        for t in range(28):
            row = img_t[m, t]                   # (28,)
            cell_in = (W_in @ row.unsqueeze(-1)).squeeze(-1)  # (N,)
            VG2 = (base_VG2 + g_VG2 * cell_in).clamp(0.0, 0.6)
            feat_b = feat.view(K, nb)
            rec_b = torch.bmm(Wb, feat_b.unsqueeze(-1)).squeeze(-1)
            rec = rec_b.view(N) * sign_mask
            VG1 = (base_VG1 + g_VG1 * rec).clamp(0.05, 0.7)
            log_Id, Iii, Ileak = surr.eval(VG1, VG2, Vd_arr, Vb)
            net = (Iii - Ileak).to(torch.float64)
            Vb = (Vb + dt * net / Cb).clamp(0.0, 0.7)
            feat = (1.0 - leak) * feat + leak * log_Id
        states[m] = feat
    torch.cuda.synchronize()
    return states.cpu().numpy()


def project_only(images, W_in_np):
    """Pure projection baseline: sum W_in @ row over 28 rows."""
    # images (M, 28, 28); W_in_np (N, 28)
    M = images.shape[0]
    out = np.zeros((M, W_in_np.shape[0]), dtype=np.float32)
    for m in range(M):
        # mean of projection across 28 timesteps
        out[m] = (W_in_np @ images[m].T).mean(axis=1)
    return out


def run_seed(seed, surr, X_train, y_train, X_test, y_test,
              N=2000, n_block=500):
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed)
    base_VG1 = torch.tensor(rng.uniform(0.2, 0.5, N).astype(np.float32), device="cuda")
    base_VG2 = torch.tensor(rng.uniform(0.05, 0.55, N).astype(np.float32), device="cuda")
    sign_mask = torch.tensor(rng.choice([-1.0, 1.0], N).astype(np.float32), device="cuda")
    W_in_np = rng.normal(0, 1.0/np.sqrt(28), size=(N, 28)).astype(np.float32)
    W_in = torch.tensor(W_in_np, dtype=torch.float32, device="cuda")
    Wb, K, nb = make_block_dense(N, n_block, seed=seed)

    apu_pre = get_apu()
    if apu_pre > 80:
        cooldown_to(60.0)

    t0 = time.time()
    St_train = encode_images(X_train, surr, base_VG1, base_VG2, sign_mask, W_in,
                                Wb, K, nb, N)
    St_test = encode_images(X_test, surr, base_VG1, base_VG2, sign_mask, W_in,
                                Wb, K, nb, N)
    enc_wall = time.time() - t0

    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    clf.fit(St_train, y_train)
    test_acc = float(clf.score(St_test, y_test))

    # Pure projection baseline
    Pp_train = project_only(X_train, W_in_np)
    Pp_test = project_only(X_test, W_in_np)
    clfp = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    clfp.fit(Pp_train, y_train)
    proj_acc = float(clfp.score(Pp_test, y_test))

    return {"seed": seed, "N": N, "test_acc": test_acc,
            "proj_acc": proj_acc, "delta_pp": (test_acc - proj_acc) * 100,
            "enc_wall_s": enc_wall, "apu_peak": get_apu()}


def main():
    log_line(f"=== z233 seq-MNIST 28×28 FROZEN config (O35 consensus) ===")
    log_line(f"Device: {torch.cuda.get_device_name(0)}")

    surr = GPUSurrogate4D(SURR_PATH)

    # Load MNIST. Use 28×28 directly.
    from sklearn.datasets import fetch_openml
    log_line(f"loading MNIST 28×28 (1000 train + 200 test)...")
    try:
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False,
                              parser="auto")
    except Exception as e:
        log_line(f"MNIST download failed: {e}; trying digits-1797 (8x8) as fallback")
        from sklearn.datasets import load_digits
        d = load_digits()
        X = d.images.reshape(-1, 64) * (255 / 16)
        y = d.target.astype(str)
    X = X.astype(np.float32).reshape(-1, 28, 28) / 255.0  # (70000, 28, 28)
    y = y.astype(int)

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    X = X[idx]; y = y[idx]
    X_train, y_train = X[:1000], y[:1000]
    X_test,  y_test  = X[1000:1200], y[1000:1200]
    log_line(f"data: train {X_train.shape} test {X_test.shape}")

    SEEDS = list(range(30))
    results = []
    t_global = time.time()
    for s in SEEDS:
        if time.time() - t_global > 35 * 60:
            log_line(f"  budget reached at seed {s}, stopping")
            break
        fp = OUT / f"seed{s}.json"
        if fp.exists():
            results.append(json.loads(fp.read_text()))
            log_line(f"  skip seed={s} (exists)")
            continue
        try:
            r = run_seed(s, surr, X_train, y_train, X_test, y_test)
            fp.write_text(json.dumps(r, indent=2))
            results.append(r)
            log_line(f"  seed={s}: reservoir={r['test_acc']:.3f} "
                     f"proj={r['proj_acc']:.3f} Δ={r['delta_pp']:+.2f}pp "
                     f"wall={r['enc_wall_s']:.0f}s peak={r['apu_peak']:.0f}°C")
            apu = get_apu()
            if apu > 92:
                log_line(f"  THERMAL KILL at seed {s}, APU={apu}")
                break
            if apu > 80:
                cooldown_to(65.0, 90)
        except Exception as e:
            log_line(f"  seed={s} FAILED: {e}")
            (OUT / f"seed{s}_CRASH.json").write_text(json.dumps(
                {"seed": s, "error": str(e), "apu": get_apu()}, indent=2))

    if results:
        accs = np.array([r["test_acc"] for r in results])
        projs = np.array([r["proj_acc"] for r in results])
        deltas = accs - projs

        # Bootstrap CI on median delta
        rng2 = np.random.default_rng(0)
        boots = np.array([np.median(deltas[rng2.integers(0, len(deltas), len(deltas))])
                            for _ in range(2000)])
        ci_lo, ci_hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))

        from scipy import stats as scs
        t, p = scs.ttest_rel(accs, projs)

        summary = {
            "n_seeds": len(results),
            "reservoir_mean": float(accs.mean()), "reservoir_std": float(accs.std()),
            "proj_mean": float(projs.mean()),     "proj_std": float(projs.std()),
            "delta_mean_pp": float(deltas.mean()*100),
            "delta_median_pp": float(np.median(deltas)*100),
            "ci95_pp": [ci_lo*100, ci_hi*100],
            "paired_t": float(t), "p_value": float(p),
            "gate_3pp_ci_excludes_0": bool(ci_lo*100 >= 3.0 or ci_hi*100 <= -3.0),
            "ci_excludes_0": bool(ci_lo > 0 or ci_hi < 0),
        }
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
        log_line(f"\n=== n={len(results)} seeds ===")
        log_line(f"reservoir : {accs.mean():.4f} ± {accs.std():.4f}")
        log_line(f"projection: {projs.mean():.4f} ± {projs.std():.4f}")
        log_line(f"Δ mean    : {deltas.mean()*100:+.2f} pp")
        log_line(f"Δ median  : {np.median(deltas)*100:+.2f} pp  CI95 [{ci_lo*100:+.2f}, {ci_hi*100:+.2f}]")
        log_line(f"paired t  : t={t:+.2f} p={p:.4g}")
        log_line(f"O35 GATE  : 3pp CI excludes 0: "
                 f"{'✅ PASS' if summary['gate_3pp_ci_excludes_0'] else '❌ FAIL'}")
        log_line(f"  CI excludes 0 (any-magnitude): "
                 f"{'✅' if summary['ci_excludes_0'] else '❌'}")


if __name__ == "__main__":
    main()
