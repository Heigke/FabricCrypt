"""z2_nsram_bsim4_zenodo — BSIM4 fitter self-consistency + cross-model check

Two honest tests Sebastian would want to see BEFORE his new data arrives:

  E1  Self-consistency — Generate synthetic Iii curves from random but
      physically plausible BSIM4 parameters, add measurement noise,
      batch-fit on the GPU, verify parameters and Iii shape recover.

  E2  Cross-model check — Overlay BSIM4 fit against Zenodo Chynoweth
      (silicon-calibrated) data and quantify the residual mismatch in
      decades.  Chynoweth = Zener-like exp, BSIM4 §6.1 = channel HCI
      (exp(-BETA0/(Vds-Vdseff))·Ids).  These are different physics;
      the test tells Sebastian how many decades of agreement to expect
      over the 2-5 V operating regime.

GPU payoff: 200+ curves × LBFGS solved in parallel in <1 s on gfx1151.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "nsram"))

from nsram.bsim4 import BSIM4Params, impact_ionization_bsim4      # noqa: E402
from nsram.fitting import fit_bsim4_impact                        # noqa: E402
from nsram.physics import avalanche_current                       # noqa: E402

OUT = REPO / "results" / "z2_nsram_bsim4_zenodo"
OUT.mkdir(parents=True, exist_ok=True)

ZENODO = dict(BV0=3.5, k_vg=1.5, Tbv1=-21.3e-6, Ne=1.5, Is=1e-16)

# ── GPU BSIM4 with physical bounds via sigmoid mapping ─────────────
class BSIM4Batch(nn.Module):
    ALPHA0_MIN, ALPHA0_MAX = 1e-8, 1e-3
    BETA0_MIN,  BETA0_MAX  = 10.0, 40.0

    def __init__(self, n_curves, base: BSIM4Params,
                 init_a=None, init_b=None, device="cuda"):
        super().__init__()
        self.base = base
        if init_a is None:
            init_a = np.full(n_curves, base.ALPHA0)
        if init_b is None:
            init_b = np.full(n_curves, base.BETA0)
        self.z_a = nn.Parameter(torch.tensor(
            self._inv(init_a, self.ALPHA0_MIN, self.ALPHA0_MAX, log=True),
            device=device, dtype=torch.float64))
        self.z_b = nn.Parameter(torch.tensor(
            self._inv(init_b, self.BETA0_MIN, self.BETA0_MAX, log=False),
            device=device, dtype=torch.float64))

    @staticmethod
    def _inv(x, lo, hi, log=False):
        x = np.clip(np.asarray(x, dtype=np.float64), lo + 1e-30, hi - 1e-20)
        if log:
            u = (np.log(x) - np.log(lo)) / (np.log(hi) - np.log(lo))
        else:
            u = (x - lo) / (hi - lo)
        u = np.clip(u, 1e-9, 1 - 1e-9)
        return np.log(u / (1 - u))

    def alpha(self):
        u = torch.sigmoid(self.z_a)
        return self.ALPHA0_MIN * (self.ALPHA0_MAX / self.ALPHA0_MIN) ** u

    def beta(self):
        u = torch.sigmoid(self.z_b)
        return self.BETA0_MIN + u * (self.BETA0_MAX - self.BETA0_MIN)

    def forward(self, Vgs, Vds, Vbs):
        p = self.base
        Vth = (p.VTH0 + p.K1 * (torch.sqrt(torch.clamp(p.PhiS - Vbs, min=1e-6))
                                 - p.PhiS ** 0.5) - p.K2 * Vbs)
        Cox = 3.9 * 8.854e-12 / p.Toxe
        beta_sq = p.mu0 * Cox * p.Weff / p.Leff
        Vgt = torch.clamp(Vgs - Vth, min=0.0)
        Vdsat = torch.clamp(Vgt, min=1e-4)
        Vdseff = torch.minimum(Vds, Vdsat)
        I_tri = beta_sq * (Vgt * Vdseff - 0.5 * Vdseff * Vdseff)
        I_sat = 0.5 * beta_sq * Vgt * Vgt * (
            1.0 + p.lambda_clm * torch.clamp(Vds - Vdsat, min=0.0))
        Ids = torch.where(Vds < Vdsat, I_tri, I_sat)
        Ids = torch.where(Vgt <= 0.0, torch.zeros_like(Ids), Ids)
        dv = torch.clamp(Vds - Vdsat, min=1e-9)
        alpha = self.alpha().unsqueeze(1)
        beta = self.beta().unsqueeze(1)
        prefac = (alpha + p.ALPHA1 * p.Leff) / p.Leff
        return prefac * dv * torch.exp(
            torch.clamp(-beta / dv, min=-80.0, max=0.0)) * Ids


def batch_fit(curves_Vgs, curves_Vds, curves_Iii, base,
              iters=150, device="cuda"):
    n = len(curves_Vgs)
    Vds = torch.tensor(np.stack(curves_Vds),  dtype=torch.float64, device=device)
    Iii = torch.tensor(np.stack(curves_Iii), dtype=torch.float64, device=device)
    Vgs = torch.tensor(np.array(curves_Vgs), dtype=torch.float64,
                        device=device).unsqueeze(1).expand(-1, Vds.shape[1])
    Vbs = torch.zeros_like(Vgs)
    log_t = torch.log(torch.clamp(Iii, min=1e-30))

    # Warm-start per curve from log-Iii vs 1/Vds slope
    init_a, init_b = np.empty(n), np.empty(n)
    for i in range(n):
        v = curves_Vds[i]; y = curves_Iii[i]
        m = (y > 1e-12) & (v > 2.0)
        if m.sum() < 5:
            init_a[i], init_b[i] = base.ALPHA0, base.BETA0
            continue
        slope, icpt = np.polyfit(-1.0 / v[m], np.log(y[m]), 1)
        init_b[i] = float(np.clip(slope, 12.0, 35.0))
        init_a[i] = float(np.clip(np.exp(icpt) / 1e-4, 1e-7, 5e-4))

    model = BSIM4Batch(n, base, init_a, init_b, device=device).double()
    opt = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=iters,
                            tolerance_grad=1e-12, tolerance_change=1e-14,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        Iii_p = model(Vgs, Vds, Vbs)
        loss = ((torch.log(torch.clamp(Iii_p, min=1e-30)) - log_t) ** 2).mean()
        loss.backward()
        return loss

    t0 = time.perf_counter()
    loss = opt.step(closure)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    with torch.no_grad():
        Iii_p = model(Vgs, Vds, Vbs).cpu().numpy()
    alpha = model.alpha().detach().cpu().numpy()
    beta  = model.beta().detach().cpu().numpy()
    Iii_t = Iii.cpu().numpy()
    log_t_np = np.log10(np.maximum(Iii_t, 1e-30))
    log_p_np = np.log10(np.maximum(Iii_p, 1e-30))
    rms_dec = np.sqrt(((log_t_np - log_p_np) ** 2).mean(axis=1))
    ss_res = ((log_t_np - log_p_np) ** 2).sum(axis=1)
    mu = log_t_np.mean(axis=1, keepdims=True)
    ss_tot = ((log_t_np - mu) ** 2).sum(axis=1) + 1e-30
    r2 = 1.0 - ss_res / ss_tot
    return dict(alpha=alpha, beta=beta, r2=r2, rms_dec=rms_dec,
                Iii_pred=Iii_p, Iii_true=Iii_t,
                t_s=t1 - t0, loss=float(loss.detach()))


# ── E1: Self-consistency ──────────────────────────────────────────
def experiment_1(device="cuda"):
    print("=" * 70)
    print("E1 — Self-consistency: random BSIM4 params → noise → fit")
    print("=" * 70)
    rng = np.random.default_rng(2026)
    N = 200
    N_VDS = 40
    Vds_arr = np.linspace(2.0, 4.5, N_VDS)

    # Ground-truth params uniformly in log(alpha), linear beta
    true_a = 10 ** rng.uniform(-7.0, -4.0, N)        # 1e-7 … 1e-4
    true_b = rng.uniform(15.0, 30.0, N)
    Vgs_arr = rng.uniform(0.6, 1.4, N)

    base = BSIM4Params()
    Iii_clean = []
    for i in range(N):
        p = BSIM4Params(**{**base.__dict__, "ALPHA0": true_a[i], "BETA0": true_b[i]})
        y = np.array([float(impact_ionization_bsim4(Vgs_arr[i], v, 0.0, p))
                      for v in Vds_arr])
        Iii_clean.append(np.maximum(y, 1e-30))
    Iii_noisy = [np.maximum(y * (1 + 0.05 * rng.standard_normal(len(y))), 1e-30)
                 for y in Iii_clean]

    fit = batch_fit(Vgs_arr.tolist(),
                     [Vds_arr] * N, Iii_noisy, base,
                     iters=150, device=device)
    a_err = np.abs(np.log10(fit["alpha"]) - np.log10(true_a))
    b_err = np.abs(fit["beta"] - true_b)
    print(f"  fit time         : {fit['t_s']:.2f} s  ({1e3*fit['t_s']/N:.1f} ms/curve)")
    print(f"  median R² (log)  : {np.median(fit['r2']):.4f}")
    print(f"  frac R²>0.95     : {(fit['r2'] > 0.95).mean():.1%}")
    print(f"  median RMS err   : {np.median(fit['rms_dec']):.3f} decades")
    print(f"  ALPHA0 |log err| : median={np.median(a_err):.3f}, p90={np.percentile(a_err, 90):.3f}")
    print(f"  BETA0  |err| (V) : median={np.median(b_err):.2f}, p90={np.percentile(b_err, 90):.2f}")

    # scipy single-curve timing (CPU) for speedup
    t0 = time.perf_counter()
    for i in range(10):
        fit_bsim4_impact(Vds_arr, Iii_noisy[i], Vgs=float(Vgs_arr[i]), Vbs=0.0, base=base)
    t_scipy = (time.perf_counter() - t0) / 10
    print(f"  scipy CPU        : {1e3*t_scipy:.1f} ms/curve → "
          f"speedup {(t_scipy)/(fit['t_s']/N):.1f}×")

    return dict(
        N=N, t_fit_s=fit["t_s"],
        r2_median=float(np.median(fit["r2"])),
        r2_frac_above_0_95=float((fit["r2"] > 0.95).mean()),
        rms_dec_median=float(np.median(fit["rms_dec"])),
        alpha0_log_err_median=float(np.median(a_err)),
        alpha0_log_err_p90=float(np.percentile(a_err, 90)),
        beta0_err_median=float(np.median(b_err)),
        beta0_err_p90=float(np.percentile(b_err, 90)),
        gpu_ms_per_curve=1e3 * fit["t_s"] / N,
        scipy_ms_per_curve=1e3 * t_scipy,
        speedup_x=(t_scipy) / (fit["t_s"] / N),
    ), fit, dict(true_a=true_a, true_b=true_b, Vds=Vds_arr, Vgs=Vgs_arr)


# ── E2: Cross-model check against Chynoweth (Zenodo-calibrated) ───
def experiment_2(device="cuda"):
    print("\n" + "=" * 70)
    print("E2 — Cross-model: BSIM4 vs silicon-calibrated Chynoweth")
    print("=" * 70)
    rng = np.random.default_rng(43)
    VG1 = np.array([0.6, 0.8, 1.0, 1.2, 1.4])
    T   = np.array([280.0, 300.0, 320.0, 340.0])
    N_VDS = 40
    Vds_arr = np.linspace(2.0, 4.5, N_VDS)
    SEEDS = 10

    curves_Vgs, curves_Vds, curves_Iii = [], [], []
    meta = []
    for vg in VG1:
        for Tk in T:
            y_clean = avalanche_current(Vds_arr, vg, T=float(Tk),
                                         I0=ZENODO["Is"], BV0=ZENODO["BV0"],
                                         k_vg=ZENODO["k_vg"], Tbv1=ZENODO["Tbv1"])
            y_clean = np.maximum(y_clean, 1e-14)
            for seed in range(SEEDS):
                y = np.maximum(
                    y_clean * (1 + 0.05 * rng.standard_normal(len(y_clean))),
                    1e-14)
                curves_Vgs.append(float(vg))
                curves_Vds.append(Vds_arr)
                curves_Iii.append(y)
                meta.append((float(vg), float(Tk), seed))

    base = BSIM4Params()
    fit = batch_fit(curves_Vgs, curves_Vds, curves_Iii, base,
                     iters=150, device=device)
    print(f"  fit time         : {fit['t_s']:.2f} s  (N={len(meta)})")
    print(f"  median R² (log)  : {np.median(fit['r2']):.4f}")
    print(f"  frac R²>0.9      : {(fit['r2'] > 0.9).mean():.1%}")
    print(f"  median RMS err   : {np.median(fit['rms_dec']):.3f} decades")
    print(f"  ALPHA0 range     : [{fit['alpha'].min():.2e}, {fit['alpha'].max():.2e}]")
    print(f"  BETA0  range     : [{fit['beta'].min():.2f}, {fit['beta'].max():.2f}] V")
    print("  (residual = irreducible Chynoweth-vs-BSIM4 model-form gap)")

    return dict(
        N=len(meta),
        t_fit_s=fit["t_s"],
        r2_median=float(np.median(fit["r2"])),
        r2_frac_above_0_9=float((fit["r2"] > 0.9).mean()),
        rms_dec_median=float(np.median(fit["rms_dec"])),
        rms_dec_p90=float(np.percentile(fit["rms_dec"], 90)),
        alpha0_range=[float(fit["alpha"].min()), float(fit["alpha"].max())],
        beta0_range=[float(fit["beta"].min()), float(fit["beta"].max())],
    ), fit, meta, (VG1, T)


# ── Plots ──────────────────────────────────────────────────────────
def make_plots(e1, e1_fit, e1_truth, e2, e2_fit, e2_meta, e2_grid):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # E1: parameter recovery scatter
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    ax1.loglog(e1_truth["true_a"], e1_fit["alpha"], ".", ms=3, alpha=0.6)
    ax1.plot([1e-7, 1e-4], [1e-7, 1e-4], "k--", lw=1, label="y=x")
    ax1.set_xlabel("ALPHA0 (true)"); ax1.set_ylabel("ALPHA0 (fit)")
    ax1.set_title("E1 — ALPHA0 recovery"); ax1.grid(alpha=0.3); ax1.legend()
    ax2.plot(e1_truth["true_b"], e1_fit["beta"], ".", ms=3, alpha=0.6)
    ax2.plot([15, 30], [15, 30], "k--", lw=1)
    ax2.set_xlabel("BETA0 (true)"); ax2.set_ylabel("BETA0 (fit)")
    ax2.set_title("E1 — BETA0 recovery"); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "e1_recovery.png", dpi=140); plt.close(fig)

    # E1: R² histogram
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(e1_fit["r2"], bins=30, color="C2", alpha=0.8, edgecolor="k")
    ax.axvline(np.median(e1_fit["r2"]), color="k", ls="--",
               label=f"median={np.median(e1_fit['r2']):.4f}")
    ax.set_xlabel("R² (log-space)"); ax.set_ylabel("count")
    ax.set_title("E1 — fit quality across 200 random BSIM4 curves")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "e1_r2_hist.png", dpi=140); plt.close(fig)

    # E2: representative overlay Chynoweth vs BSIM4 across (Vg1, T)
    VG1, T = e2_grid
    fig, axes = plt.subplots(len(VG1), len(T),
                              figsize=(3.2 * len(T), 2.2 * len(VG1)),
                              sharex=True, sharey=True)
    for i, (vg, Tk, seed) in enumerate(e2_meta):
        if seed != 0:
            continue
        r = np.where(VG1 == vg)[0][0]
        c = np.where(T == Tk)[0][0]
        ax = axes[r, c]
        ax.semilogy(np.linspace(2.0, 4.5, 40),
                    e2_fit["Iii_true"][i], "o", ms=2.5,
                    label="Chynoweth (Zenodo)")
        ax.semilogy(np.linspace(2.0, 4.5, 40),
                    e2_fit["Iii_pred"][i], "-", lw=1.5,
                    label="BSIM4 fit")
        ax.set_title(f"Vg1={vg:.1f}V  T={int(Tk)}K\nRMS={e2_fit['rms_dec'][i]:.2f}dec",
                     fontsize=8)
        ax.grid(alpha=0.25, which="both")
        if r == len(VG1) - 1: ax.set_xlabel("Vds (V)")
        if c == 0:            ax.set_ylabel("Iii (A)")
        if r == 0 and c == 0: ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("E2 — Cross-model gap: BSIM4 §6.1 fit vs silicon Chynoweth",
                  y=1.0)
    fig.tight_layout(); fig.savefig(OUT / "e2_overlay.png", dpi=140,
                                     bbox_inches="tight"); plt.close(fig)

    # E2: RMS error heatmap (Vg1 × T)
    VG1, T = e2_grid
    mat = np.zeros((len(VG1), len(T)))
    for i, (vg, Tk, seed) in enumerate(e2_meta):
        r = np.where(VG1 == vg)[0][0]
        c = np.where(T == Tk)[0][0]
        mat[r, c] += e2_fit["rms_dec"][i]
    mat /= 10  # 10 seeds per cell
    fig, ax = plt.subplots(figsize=(5, 3.5))
    im = ax.imshow(mat, aspect="auto", cmap="viridis",
                   extent=[T[0], T[-1], VG1[-1], VG1[0]])
    for r in range(len(VG1)):
        for c in range(len(T)):
            ax.text(T[c], VG1[r], f"{mat[r,c]:.2f}", ha="center",
                    va="center", color="white", fontsize=8)
    ax.set_xlabel("Temperature (K)"); ax.set_ylabel("Vg1 (V)")
    ax.set_xticks(T); ax.set_yticks(VG1)
    fig.colorbar(im, label="RMS error (decades)")
    ax.set_title("E2 — BSIM4↔Chynoweth gap over operating envelope")
    fig.tight_layout(); fig.savefig(OUT / "e2_gap_heatmap.png", dpi=140); plt.close(fig)

    print(f"[plot] 4 figures → {OUT}")


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        print(f"[device] {torch.cuda.get_device_name(0)}")

    e1, e1_fit, e1_truth = experiment_1(device=dev)
    e2, e2_fit, e2_meta, e2_grid = experiment_2(device=dev)
    make_plots(e1, e1_fit, e1_truth, e2, e2_fit, e2_meta, e2_grid)

    summary = {
        "device": dev, "nsram_version": "0.11.1",
        "zenodo_params": ZENODO,
        "E1_self_consistency": e1,
        "E2_cross_model_chynoweth": e2,
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] summary → {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
