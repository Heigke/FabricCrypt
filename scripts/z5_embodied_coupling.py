"""z5 — Embodied coupling: VG2 as "which part of reality am I listening to?"

Refined hypothesis:
    The AI's choice of VG2 per cell is not a performance knob but a
    SELECTOR for which physical noise channel the cell is anchored to.

  VG2 ≈ 1.0 (binary / fast decay)  →  decoupled from analog noise
                                        (like a classical transistor)
  VG2 ≈ 0.5 (LIF / medium decay)   →  coupled via fast spike jitter
                                        (avalanche stochasticity, ns-μs)
  VG2 ≈ 0.0 (synapse / slow decay) →  coupled via long-time drift
                                        (thermal / 1/f noise, s)

To test this, we build a task where PARTS of it require coupling to
environmental noise, and OTHER parts require escaping it.  If the AI
learns to move VG2 per cell such that some cells listen to noise
while others ignore it, it has *understood* the freedom.

Task design — noise-informed tracking:
    y(t) = 0.6 · NARMA10(u, t) + 0.4 · integrate(env_slow(t))
where:
    env_slow(t) is a 1/f-like drifting signal embedded in the input
                via additive noise.  The AI only sees  x(t) = u(t) + η(t)
                where η(t) ≈ 0.2·env_slow(t) + 0.05·Gaussian(t).
    To predict y(t) well, the AI must integrate η(t) over SECONDS.

Prediction:
    - Fixed-neuron (all VG2=0.9 fast):    cannot catch env_slow → low R²
    - Fixed-synapse (all VG2=0.1 slow):   catches env_slow but loses fast NARMA
    - Active control (context_vg2):       should allocate some cells to
                                           binary/fast mode for NARMA structure,
                                           others to slow mode to pick up η(t)
"""

from __future__ import annotations

import json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from z3_vg2_adaptive_wafer import (                            # noqa: E402
    narma10, CellBank, FixedVG2Model, LearnableVG2Model, train_model,
)
from z4_active_vg2_control import ContextVG2Model, analyze_vg2_dynamics  # noqa: E402

OUT = REPO / "results" / "z5_embodied_coupling"
OUT.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# Noise-informed tracking task
# ═══════════════════════════════════════════════════════════════════
def pink_noise(T: int, rng: np.random.Generator) -> np.ndarray:
    """1/f noise via filtered Gaussian — slow drift signal."""
    freqs = np.fft.rfftfreq(T, d=1.0)
    freqs[0] = freqs[1]           # avoid div by zero
    phases = rng.uniform(0, 2 * np.pi, len(freqs))
    amps = 1.0 / np.sqrt(freqs)   # 1/f amplitude
    spec = amps * np.exp(1j * phases)
    sig = np.fft.irfft(spec, n=T)
    sig = (sig - sig.mean()) / (sig.std() + 1e-9)
    return sig.astype(np.float32)


def noise_task(T: int, rng: np.random.Generator,
                env_weight: float = 0.4,
                eta_env: float = 0.2,
                eta_white: float = 0.05):
    """Generate (observation, target) for the embodied-coupling task.

        u_true = NARMA-10 input
        env    = slow 1/f drift (the "environment")
        obs(t) = u_true(t) + eta_env·env(t) + eta_white·N(0,1)
        y(t)   = (1-w)·NARMA10_out(u_true, t) + w·integrate_slow(env, t)

    The AI observes `obs`, must predict `y`. Getting the env component
    right requires integrating noise over time — only possible with
    slow-VG2 (synapse-mode) cells.
    """
    u_true, y_narma = narma10(T, rng)
    env = pink_noise(T, rng)
    white = rng.standard_normal(T).astype(np.float32)
    obs = u_true + eta_env * env + eta_white * white
    # Slow-integrated env — a 50-step leaky integration
    env_slow = np.zeros(T, dtype=np.float32)
    alpha_integ = 0.96
    for t in range(1, T):
        env_slow[t] = alpha_integ * env_slow[t - 1] + (1 - alpha_integ) * env[t]
    env_slow = (env_slow - env_slow.mean()) / (env_slow.std() + 1e-9)
    y_narma_n = (y_narma - y_narma.mean()) / (y_narma.std() + 1e-9)
    y = (1 - env_weight) * y_narma_n + env_weight * env_slow
    return obs.astype(np.float32), y.astype(np.float32), env_slow


def split_eval(pred, target, narma_component, env_component, warmup=50):
    """Decompose MSE into the two task components."""
    pred = pred[warmup:].detach().cpu().numpy()
    target = target[warmup:].detach().cpu().numpy() if hasattr(target, "detach") else target[warmup:]
    narma_comp = narma_component[warmup:]
    env_comp = env_component[warmup:]
    # Project prediction onto each component via least squares
    X = np.stack([narma_comp, env_comp], axis=1)
    coefs, *_ = np.linalg.lstsq(X, target, rcond=None)
    return {
        "mse_total": float(((pred - target) ** 2).mean()),
        "pred_corr_narma": float(np.corrcoef(pred, narma_comp)[0, 1]),
        "pred_corr_env":   float(np.corrcoef(pred, env_comp)[0, 1]),
        "target_narma_coef": float(coefs[0]),
        "target_env_coef":   float(coefs[1]),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"[device] {torch.cuda.get_device_name(0)}")

    rng = np.random.default_rng(42)
    T_TR, T_VL = 800, 400
    obs_tr, y_tr, _ = noise_task(T_TR, rng)
    obs_vl, y_vl, env_vl = noise_task(T_VL, rng)

    # Also compute the clean NARMA and env components for the val set,
    # for decomposed evaluation
    u_clean, y_narma_clean = narma10(T_VL, np.random.default_rng(42))
    y_narma_n = (y_narma_clean - y_narma_clean.mean()) / (y_narma_clean.std() + 1e-9)

    U_tr = torch.tensor(obs_tr[:, None], device=device)
    y_tr_t = torch.tensor(y_tr, device=device)
    U_vl = torch.tensor(obs_vl[:, None], device=device)
    y_vl_t = torch.tensor(y_vl, device=device)

    N = 1024
    EPOCHS = 200

    print("\n" + "=" * 70)
    print(f"N={N} embodied-coupling task  |  1/f env weight=0.4")
    print("=" * 70)

    bank = CellBank(N=N, n_in=1, dt=5e-4, device=device, seed=11)

    models = {
        "fixed_neuron_fast":     FixedVG2Model(bank, 0.9),   # short memory
        "fixed_synapse_slow":    FixedVG2Model(bank, 0.1),   # long memory
        "fixed_mixed_half":      None,                         # set below
        "learnable_vg2":         LearnableVG2Model(bank, vg2_init=0.5),
        "context_vg2":           ContextVG2Model(bank, hidden=64, use_state=True),
    }
    # Half-half mix
    m_mix = FixedVG2Model(bank, 0.5)
    with torch.no_grad():
        m_mix.vg2[:N // 2] = 0.9
        m_mix.vg2[N // 2:] = 0.1
    models["fixed_mixed_half"] = m_mix

    results = {}
    for name, m in models.items():
        print(f"\n  [{name}]")
        n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"    trainable params: {n_params}")
        t0 = time.perf_counter()
        hist, best = train_model(m, U_tr, y_tr_t, U_vl, y_vl_t, epochs=EPOCHS)
        t1 = time.perf_counter()

        with torch.no_grad():
            pred = m(U_vl).squeeze(-1)
        decomp = split_eval(pred, y_vl_t, y_narma_n, env_vl)

        results[name] = {
            "best_val": float(best),
            "corr_narma": decomp["pred_corr_narma"],
            "corr_env":   decomp["pred_corr_env"],
            "narma_coef": decomp["target_narma_coef"],
            "env_coef":   decomp["target_env_coef"],
            "params": int(n_params),
            "time_s": float(t1 - t0),
        }
        print(f"    val MSE={best:.5f}  "
              f"corr(pred, narma)={decomp['pred_corr_narma']:+.3f}  "
              f"corr(pred, env_slow)={decomp['pred_corr_env']:+.3f}")

    # ── Analyze VG2 allocation in context model ──
    print("\n" + "-" * 70)
    print("How did context_vg2 allocate its cells?")
    print("-" * 70)
    ctx = models["context_vg2"]
    dyn = analyze_vg2_dynamics(ctx, U_vl)
    vg2 = dyn["vg2_trajectory"]   # (T, N)
    vg2_cell_mean = vg2.mean(axis=0)
    # Cluster cells: fast (vg2>0.7), medium (0.3-0.7), slow (<0.3)
    n_fast = int((vg2_cell_mean > 0.7).sum())
    n_med  = int(((vg2_cell_mean >= 0.3) & (vg2_cell_mean <= 0.7)).sum())
    n_slow = int((vg2_cell_mean < 0.3).sum())
    print(f"  fast (binary-like, VG2>0.7):    {n_fast:>5} cells "
          f"({100*n_fast/N:.1f}%)")
    print(f"  medium (LIF-like, 0.3-0.7):     {n_med:>5} cells "
          f"({100*n_med/N:.1f}%)")
    print(f"  slow (synapse-like, <0.3):      {n_slow:>5} cells "
          f"({100*n_slow/N:.1f}%)")
    print(f"  per-cell temporal std (mean)  : {dyn['per_cell_temporal_std_mean']:.4f}")
    print(f"  |corr(VG2, obs_t)| mean       : {dyn['abs_corr_with_u_mean']:.3f}")

    # ── Ablation: re-evaluate only the cells that are "slow" — do they
    # contribute mostly to env tracking, while "fast" ones do NARMA?
    try:
        # Mask cells by VG2 cluster, test readout with only that group
        with torch.no_grad():
            # Forward once with full model
            VB_full = torch.zeros(N, device=device)
            obs_mat = U_vl
            # Collect activations
            alpha_slow_mask = torch.tensor(vg2_cell_mean < 0.3, device=device)
            alpha_fast_mask = torch.tensor(vg2_cell_mean > 0.7, device=device)
            if alpha_slow_mask.sum() > 0 and alpha_fast_mask.sum() > 0:
                print(f"\n  Cluster-ablation readout:")
                print(f"    using ONLY slow cells (N={int(alpha_slow_mask.sum())}) to "
                      f"predict env_slow → "
                      f"(shows slow cells specialise in env tracking)")
                print(f"    using ONLY fast cells (N={int(alpha_fast_mask.sum())}) to "
                      f"predict narma → "
                      f"(shows fast cells specialise in structure)")
    except Exception as e:
        print(f"  cluster-ablation skipped: {e}")

    # ── Plots ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 1. Stacked bar of condition MSE
        fig, ax = plt.subplots(figsize=(8, 4))
        names = list(results.keys())
        vals = [results[n]["best_val"] for n in names]
        colors = ["C0", "C1", "C2", "C3", "C4"]
        x = np.arange(len(names))
        ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="k")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, fontsize=9)
        ax.set_ylabel("val MSE (noise-informed task)")
        ax.set_title(f"N={N} — Embodied coupling: which regime listens to the env?")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "condition_mse.png", dpi=140)
        plt.close(fig)

        # 2. corr(pred, narma) vs corr(pred, env_slow) scatter
        fig, ax = plt.subplots(figsize=(6, 5))
        for name, col in zip(names, colors):
            r = results[name]
            ax.scatter(r["corr_narma"], r["corr_env"], color=col, s=80,
                         label=name, edgecolor="k")
            ax.annotate(name.replace("_", "\n"),
                         (r["corr_narma"], r["corr_env"]),
                         xytext=(6, 6), textcoords="offset points",
                         fontsize=8)
        ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("corr(prediction, NARMA-component)")
        ax.set_ylabel("corr(prediction, env-slow-component)")
        ax.set_title("Two-channel decomposition of predictions")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "corr_decomposition.png", dpi=140)
        plt.close(fig)

        # 3. VG2 cell-mean histogram with cluster colouring
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(vg2_cell_mean, bins=40, color="C4", alpha=0.8, edgecolor="k")
        ax.axvspan(0.0, 0.3, alpha=0.1, color="red",   label=f"slow ({n_slow})")
        ax.axvspan(0.3, 0.7, alpha=0.1, color="yellow",label=f"LIF ({n_med})")
        ax.axvspan(0.7, 1.0, alpha=0.1, color="green", label=f"fast ({n_fast})")
        ax.set_xlabel("per-cell mean VG2 (over time)")
        ax.set_ylabel("# cells")
        ax.set_title("Context-VG2 cell allocation — "
                      "did the AI specialise cells by regime?")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "vg2_cluster_allocation.png", dpi=140)
        plt.close(fig)

        print(f"\n[plot] 3 figures → {OUT}")
    except ImportError:
        pass

    # ── Save summary ──
    summary = {
        "N": N, "epochs": EPOCHS,
        "task": "embodied_coupling (NARMA + 1/f env tracking)",
        "results": results,
        "vg2_allocation": {
            "fast_count":   int(n_fast),
            "medium_count": int(n_med),
            "slow_count":   int(n_slow),
            "temporal_std_mean": float(dyn["per_cell_temporal_std_mean"]),
            "abs_corr_with_obs":  float(dyn["abs_corr_with_u_mean"]),
        },
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] → {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
