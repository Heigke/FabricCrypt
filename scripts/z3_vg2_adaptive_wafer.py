"""z3 — Wafer-scale VG2-adaptive NS-RAM reservoir

Tests the hypothesis:

  A physical "knob" (VG2) that lets each NS-RAM cell slide continuously
  between binary memory ↔ neuron ↔ synapse is a degree of freedom
  that a sufficiently scaled AI can learn to exploit, and gives it
  capabilities that a pure-digital "brain in jar" must pay for with
  extra compute (learning to model the world internally).

Simulator design (differentiable, GPU):

  Each cell i has state VB_i (body potential) and a knob VG2_i ∈ [0, 1].
  VG2 maps to bulk resistance via sigmoid:
      Rb_i = Rb_min + (Rb_max − Rb_min)·σ((VG2_i − 0.5)·10)
      Rb_min = 10 kΩ   (neuron: fast leak, spikes)
      Rb_max = 1 MΩ    (synapse: long retention)

  Body dynamics (one-step Euler, dt=0.5 ms):
      dVB/dt = (W @ u − VB/Rb) / Cb
  where u ∈ R^n_in is the current input, W ∈ R^{N×n_in} are fixed random
  input weights, Cb = 1 pF.  A soft spike readout
      s_i = σ((VB_i − 0.6) · 10)
  models the sub-threshold-to-suprathreshold firing transition without
  breaking autograd.

  Linear readout (trainable) maps s ∈ R^N to y_pred ∈ R^n_out.

Four conditions compared on NARMA-10:

  1. fixed_neuron    — all VG2 = 0.1 (frozen)           [pure-spiking baseline]
  2. fixed_synapse   — all VG2 = 0.9 (frozen)           [pure-memory baseline]
  3. learnable_vg2   — per-cell VG2 is trainable         [physical-knob theory]
  4. world_model     — fixed VG2=0.5 plus a small MLP
                        that predicts modulation from u and
                        scales s (the "brain-in-jar" has to
                        model the world digitally)        [cost-of-modelling]

Scale sweep: N ∈ {256, 1024, 4096} cells.
"""

from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "z3_vg2_adaptive_wafer"
OUT.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# Task: NARMA-10 — standard nonlinear memory benchmark
# ═══════════════════════════════════════════════════════════════════
def narma10(T: int, rng: np.random.Generator):
    """NARMA-10: y[t+1] = 0.3 y[t] + 0.05 y[t] Σ y[t-i] + 1.5 u[t-9] u[t] + 0.1"""
    u = rng.uniform(0, 0.5, T)
    y = np.zeros(T)
    for t in range(10, T - 1):
        y[t + 1] = (
            0.3 * y[t]
            + 0.05 * y[t] * np.sum(y[t - 9:t + 1])
            + 1.5 * u[t - 9] * u[t]
            + 0.1
        )
    return u.astype(np.float32), y.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
# Shared forward dynamics
# ═══════════════════════════════════════════════════════════════════
class CellBank:
    """Batched leaky-integrator dynamics for N NS-RAM cells, differentiable.

    Math-equivalent reformulation of the physical dVB/dt = I − VB/(Rb·Cb):
        VB[t+1] = α(VG2) · VB[t] + (1 − α(VG2)) · (W_in u + W_rec s)
        s[t+1] = tanh(VB[t+1])         # bounded non-linearity

    α(VG2 = 0.0) ≈ 0.05  (neuron: near-memoryless, spike-like)
    α(VG2 = 1.0) ≈ 0.98  (synapse: long retention)

    This is THE standard leaky reservoir; VG2 just becomes the per-cell
    leak rate.  Physically, α = exp(−dt / (Rb·Cb)) — mapping from NS-RAM
    params to this α is one-to-one.
    """

    ALPHA_MIN = 0.05    # neuron mode
    ALPHA_MAX = 0.98    # synapse mode
    SPECTRAL_RADIUS = 0.9

    def __init__(self, N: int, n_in: int, dt: float, device, seed: int = 7):
        self.N, self.n_in, self.dt = N, n_in, dt
        self.device = device
        g = torch.Generator(device=device).manual_seed(seed)
        scale_in = 1.0 / np.sqrt(n_in)
        self.W_in = (torch.randn(N, n_in, generator=g, device=device) * scale_in)
        W_rec = torch.randn(N, N, generator=g, device=device) / np.sqrt(N)
        # Normalize to target spectral radius via power iteration
        with torch.no_grad():
            v = torch.randn(N, device=device)
            for _ in range(20):
                v = W_rec @ v
                v = v / (v.norm() + 1e-12)
            sr = float((W_rec @ v).norm())
        self.W_rec = W_rec * (self.SPECTRAL_RADIUS / max(sr, 1e-6))

    def alpha_from_vg2(self, vg2):
        """VG2 ∈ [0, 1] → α ∈ [ALPHA_MIN, ALPHA_MAX]."""
        vg2 = torch.clamp(vg2, 0.0, 1.0)
        return self.ALPHA_MIN + (self.ALPHA_MAX - self.ALPHA_MIN) * vg2

    def step(self, VB, u, vg2, s_mod=None):
        """One step. Returns new VB and bounded activation s."""
        alpha = self.alpha_from_vg2(vg2)
        s = torch.tanh(VB)                     # bounded, autograd-safe
        if s_mod is not None:
            s = s * s_mod
        drive = u @ self.W_in.T + s @ self.W_rec.T
        VB_new = alpha * VB + (1.0 - alpha) * drive
        return VB_new, s


# ═══════════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════════
class FixedVG2Model(nn.Module):
    """VG2 frozen to a constant (no adaptation). Only the readout is learnable."""
    def __init__(self, bank: CellBank, vg2_const: float, n_out: int = 1):
        super().__init__()
        self.bank = bank
        self.register_buffer(
            "vg2", torch.full((bank.N,), vg2_const, device=bank.device))
        self.readout = nn.Linear(bank.N, n_out, bias=True).to(bank.device)

    def forward(self, U):
        T, n_in = U.shape
        VB = torch.zeros(self.bank.N, device=self.bank.device, dtype=U.dtype)
        outs = []
        for t in range(T):
            VB, s = self.bank.step(VB, U[t], self.vg2)
            outs.append(self.readout(s))
        return torch.stack(outs, 0)

    @property
    def extra_params(self): return 0
    @property
    def extra_flops_per_step(self): return 0


class LearnableVG2Model(nn.Module):
    """Per-cell VG2 is a trainable parameter — THE HYPOTHESIS."""
    def __init__(self, bank: CellBank, n_out: int = 1,
                 vg2_init: float = 0.5):
        super().__init__()
        self.bank = bank
        self.vg2_raw = nn.Parameter(torch.full(
            (bank.N,), float(np.log(vg2_init / (1 - vg2_init))),
            device=bank.device))
        self.readout = nn.Linear(bank.N, n_out, bias=True).to(bank.device)

    def vg2(self):
        return torch.sigmoid(self.vg2_raw)

    def forward(self, U):
        T, n_in = U.shape
        VB = torch.zeros(self.bank.N, device=self.bank.device, dtype=U.dtype)
        outs = []
        vg2 = self.vg2()
        for t in range(T):
            VB, s = self.bank.step(VB, U[t], vg2)
            outs.append(self.readout(s))
        return torch.stack(outs, 0)

    @property
    def extra_params(self): return self.bank.N
    @property
    def extra_flops_per_step(self): return 0    # VG2 sigmoid is one-off


class WorldModelModel(nn.Module):
    """No physical knob — a small MLP predicts per-cell modulation from
    input statistics (the cost of "modelling the world in its head")."""
    def __init__(self, bank: CellBank, n_out: int = 1, hidden: int = 64):
        super().__init__()
        self.bank = bank
        self.register_buffer(
            "vg2", torch.full((bank.N,), 0.5, device=bank.device))
        self.world_mlp = nn.Sequential(
            nn.Linear(bank.n_in + 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, bank.N),
        ).to(bank.device)
        # Initialise last layer small so s_mod starts near 1.0
        with torch.no_grad():
            self.world_mlp[-1].weight.mul_(0.01)
            self.world_mlp[-1].bias.zero_()
        self.readout = nn.Linear(bank.N, n_out, bias=True).to(bank.device)
        self.hidden = hidden

    def forward(self, U):
        T, n_in = U.shape
        VB = torch.zeros(self.bank.N, device=self.bank.device, dtype=U.dtype)
        outs = []
        # rolling input stats (unbiased=False avoids NaN for n_in=1)
        mu = torch.zeros(n_in, device=U.device, dtype=U.dtype)
        M2 = torch.zeros(n_in, device=U.device, dtype=U.dtype)
        t_running = 0
        for t in range(T):
            u = U[t]
            t_running += 1
            delta = u - mu
            mu = mu + delta / t_running
            M2 = M2 + delta * (u - mu)
            var = M2 / max(t_running, 1)
            # Extra stats: current-batch mean, rolling mean, rolling var, t
            stats = torch.stack([
                u.mean(), mu.mean(), var.mean(),
                torch.tensor(float(t) / T, device=U.device, dtype=U.dtype),
            ])
            world_in = torch.cat([u, stats])
            # s_mod ∈ [0.5, 1.5] — modulation around unity, bounded gradients
            s_mod = 1.0 + 0.5 * torch.tanh(self.world_mlp(world_in))
            VB, s = self.bank.step(VB, u, self.vg2, s_mod=s_mod)
            outs.append(self.readout(s))
        return torch.stack(outs, 0)

    @property
    def extra_params(self):
        return sum(p.numel() for p in self.world_mlp.parameters())

    @property
    def extra_flops_per_step(self):
        N = self.bank.N
        return 2 * (self.bank.n_in + 4) * self.hidden + 2 * self.hidden * self.hidden + 2 * self.hidden * N


# ═══════════════════════════════════════════════════════════════════
# Training / evaluation
# ═══════════════════════════════════════════════════════════════════
def train_model(model, U_train, y_train, U_val, y_val,
                 epochs=200, lr=2e-3, warmup=50, log_every=40):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    hist = []
    best_val = float("inf")
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(U_train).squeeze(-1)
        # Skip first `warmup` steps in loss (body settling transient)
        loss = ((pred[warmup:] - y_train[warmup:]) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        sched.step()

        with torch.no_grad():
            model.eval()
            vpred = model(U_val).squeeze(-1)
            vloss = ((vpred[warmup:] - y_val[warmup:]) ** 2).mean().item()
        best_val = min(best_val, vloss)
        hist.append((ep, float(loss.item()), float(vloss)))
        if ep % log_every == 0 or ep == epochs - 1:
            print(f"    ep {ep:3d}  train={loss.item():.5f}  val={vloss:.5f}"
                  f"  best={best_val:.5f}")
    return hist, best_val


def run_condition(name, model_fn, bank, U_tr, y_tr, U_vl, y_vl, epochs=150):
    print(f"\n  [{name}]")
    model = model_fn(bank)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    extra = model.extra_params
    print(f"    trainable params: {n_params} (readout + {extra} extra)")
    t0 = time.perf_counter()
    hist, best_val = train_model(model, U_tr, y_tr, U_vl, y_vl, epochs=epochs)
    t1 = time.perf_counter()
    final_val = best_val
    vg2_stats = {}
    if hasattr(model, "vg2") and callable(getattr(model, "vg2")):
        vg2 = model.vg2().detach().cpu().numpy()
        vg2_stats = {
            "mean": float(vg2.mean()), "std": float(vg2.std()),
            "min": float(vg2.min()),  "max": float(vg2.max()),
            "hist_edges": np.linspace(0, 1, 11).tolist(),
            "hist_counts": np.histogram(vg2, bins=np.linspace(0, 1, 11))[0].tolist(),
        }
    elif hasattr(model, "vg2") and isinstance(model.vg2, torch.Tensor):
        vg2 = model.vg2.detach().cpu().numpy()
        vg2_stats = {"mean": float(vg2.mean()), "std": float(vg2.std())}
    return {
        "name": name,
        "params_trainable": int(n_params),
        "extra_params": int(extra),
        "extra_flops_per_step": int(model.extra_flops_per_step),
        "final_val_mse": float(final_val),
        "train_time_s": float(t1 - t0),
        "history": [(int(h[0]), float(h[1]), float(h[2])) for h in hist],
        "vg2_stats": vg2_stats,
    }


# ═══════════════════════════════════════════════════════════════════
# Physics-plausibility audit
# ═══════════════════════════════════════════════════════════════════
def physics_audit(N_values):
    """Rough die-area / power / DAC-cost estimate per cell count."""
    area_per_cell_um2 = 17.0       # Pazos Nature 640
    energy_per_spike_fJ = 21.0     # Pazos Nature 640
    dac_area_um2 = 40.0            # 6-bit current-mode DAC per cell (shared possible)
    spike_rate_hz = 3e5            # typical 300 kHz
    rows = []
    for N in N_values:
        area_mm2 = N * (area_per_cell_um2 + dac_area_um2) / 1e6
        power_uw = N * energy_per_spike_fJ * 1e-15 * spike_rate_hz * 1e6
        vg2_lines = N            # one VG2 per cell (can share across rows)
        rows.append({
            "N": N,
            "die_area_mm2": float(area_mm2),
            "power_uW": float(power_uw),
            "vg2_control_lines": int(vg2_lines),
            "fits_on_7nm_300mm_wafer_cell_count_max": int(70000e6 / (area_per_cell_um2 + dac_area_um2)),
        })
    return rows


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"[device] {torch.cuda.get_device_name(0)}")
    torch.set_default_dtype(torch.float32)

    # Task
    rng = np.random.default_rng(2026)
    T_TRAIN, T_VAL = 600, 300
    u_tr, y_tr = narma10(T_TRAIN, rng)
    u_vl, y_vl = narma10(T_VAL, rng)
    U_tr = torch.tensor(u_tr[:, None], device=device)
    y_tr_t = torch.tensor(y_tr, device=device)
    U_vl = torch.tensor(u_vl[:, None], device=device)
    y_vl_t = torch.tensor(y_vl, device=device)

    N_VALUES = [256, 1024, 4096]
    EPOCHS = 150
    DT = 5e-4   # 0.5 ms Euler step

    all_results = {}
    for N in N_VALUES:
        print("\n" + "=" * 70)
        print(f"N = {N} cells  |  epochs = {EPOCHS}  |  task = NARMA-10")
        print("=" * 70)
        bank = CellBank(N=N, n_in=1, dt=DT, device=device, seed=7)

        conds = [
            ("fixed_neuron",
                lambda b: FixedVG2Model(b, 0.1)),
            ("fixed_synapse",
                lambda b: FixedVG2Model(b, 0.9)),
            ("learnable_vg2",
                lambda b: LearnableVG2Model(b, vg2_init=0.5)),
            ("world_model",
                lambda b: WorldModelModel(b, hidden=64)),
        ]
        res = {}
        for name, fn in conds:
            res[name] = run_condition(name, fn, bank, U_tr, y_tr_t,
                                        U_vl, y_vl_t, epochs=EPOCHS)
        all_results[f"N={N}"] = res

    # Physics audit
    audit = physics_audit([256, 1024, 4096, 100_000, 1_000_000])

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY  (final validation MSE on NARMA-10; lower = better)")
    print("=" * 70)
    header = f"{'N':>6}  {'fixed_neuron':>13}  {'fixed_synapse':>14}  {'learnable_vg2':>15}  {'world_model':>12}"
    print(header)
    for N in N_VALUES:
        r = all_results[f"N={N}"]
        row = f"{N:>6}"
        for k in ("fixed_neuron", "fixed_synapse", "learnable_vg2", "world_model"):
            row += f"  {r[k]['final_val_mse']:>13.5f}" if k != "learnable_vg2" else f"  {r[k]['final_val_mse']:>15.5f}"
        print(row)

    print("\n" + "-" * 70)
    print("COST  (extra trainable params beyond the linear readout)")
    print("-" * 70)
    for N in N_VALUES:
        r = all_results[f"N={N}"]
        lv = r["learnable_vg2"]["extra_params"]
        wm = r["world_model"]["extra_params"]
        print(f"  N={N:>5}  learnable_vg2: +{lv:>6d}   world_model: +{wm:>6d}   ratio: {wm/max(lv,1):.1f}×")

    print("\n" + "-" * 70)
    print("PHYSICS AUDIT — buildability on 2026 process")
    print("-" * 70)
    print(f"  Max cells per 300 mm 7nm wafer: {audit[0]['fits_on_7nm_300mm_wafer_cell_count_max']:,}")
    for a in audit:
        print(f"  N={a['N']:>8,}  die-area={a['die_area_mm2']:>8.3f} mm²  "
              f"power={a['power_uW']:>10.2f} μW  vg2-lines={a['vg2_control_lines']:>8,}")

    with open(OUT / "summary.json", "w") as f:
        json.dump({
            "task": "NARMA-10",
            "epochs": EPOCHS,
            "dt_s": DT,
            "T_train": T_TRAIN, "T_val": T_VAL,
            "results": all_results,
            "physics_audit": audit,
        }, f, indent=2)
    print(f"\n[done] → {OUT / 'summary.json'}")

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 1. MSE vs N per condition
        fig, ax = plt.subplots(figsize=(6, 3.8))
        for k in ("fixed_neuron", "fixed_synapse", "learnable_vg2", "world_model"):
            ys = [all_results[f"N={N}"][k]["final_val_mse"] for N in N_VALUES]
            ax.plot(N_VALUES, ys, "o-", label=k, lw=2, ms=6)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("# cells (N)"); ax.set_ylabel("val MSE (NARMA-10)")
        ax.set_title("Does the VG2 knob beat world-modelling?")
        ax.legend(); ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "mse_vs_N.png", dpi=140)
        plt.close(fig)

        # 2. Cost vs benefit
        fig, ax = plt.subplots(figsize=(6, 3.8))
        for k, marker, color in [("learnable_vg2", "o", "C2"),
                                    ("world_model", "s", "C3")]:
            xs = [all_results[f"N={N}"][k]["extra_params"] for N in N_VALUES]
            ys = [all_results[f"N={N}"][k]["final_val_mse"] for N in N_VALUES]
            ax.plot(xs, ys, marker + "-", color=color, label=k, lw=2, ms=7)
            for N, x, y in zip(N_VALUES, xs, ys):
                ax.annotate(f"N={N}", (x, y), xytext=(4, 4),
                             textcoords="offset points", fontsize=8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("extra trainable params"); ax.set_ylabel("val MSE")
        ax.set_title("Cost (params) vs benefit (loss)")
        ax.legend(); ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "cost_vs_benefit.png", dpi=140)
        plt.close(fig)

        # 3. VG2 histogram for learnable
        fig, axes = plt.subplots(1, len(N_VALUES), figsize=(4 * len(N_VALUES), 3),
                                   sharey=True)
        for i, N in enumerate(N_VALUES):
            d = all_results[f"N={N}"]["learnable_vg2"]["vg2_stats"]
            axes[i].bar(d["hist_edges"][:-1], d["hist_counts"], width=0.1,
                         align="edge", color="C2", alpha=0.8, edgecolor="k")
            axes[i].axvline(d["mean"], color="k", ls="--",
                             label=f"μ={d['mean']:.2f}")
            axes[i].set_xlabel("VG2"); axes[i].set_title(f"N={N}")
            axes[i].legend(fontsize=8); axes[i].grid(alpha=0.3)
        axes[0].set_ylabel("# cells")
        fig.suptitle("Learned VG2 distribution — does the AI spread across regimes?")
        fig.tight_layout()
        fig.savefig(OUT / "vg2_distributions.png", dpi=140)
        plt.close(fig)

        print(f"[plot] 3 figures → {OUT}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
