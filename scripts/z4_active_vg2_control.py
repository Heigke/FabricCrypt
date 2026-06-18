"""z4 — Active VG2 control: does the AI learn to USE its knob in real time?

Goes beyond z3's static learnable VG2. Now VG2 is CHOSEN EVERY STEP by
a small policy looking at the current input and state.  The three
questions we answer:

  Q1 Does it help performance, given the same parameter budget as the
     "world-model" MLP approach that modulates digitally?
  Q2 Does the AI *actually* use the knob dynamically, or does it
     collapse to a static per-cell value? (If collapses, it's just
     z3 dressed up.)
  Q3 Freeze test — after training with active control, clamp VG2 to
     its mean. If performance collapses, the AI *depends* on the
     live knob, matching the user's "self-aware control" hypothesis.

Parameter-matched architectures:
    world_model   : MLP (n_in+stats) → h → h → N·1  (multiplies state s)
    context_vg2   : MLP (n_in+stats+state_summary) → h → h → N·1  (controls VG2)
Same hidden size so FLOPs/params are comparable within ~10%.
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
    narma10, CellBank, FixedVG2Model, LearnableVG2Model, WorldModelModel,
    train_model,
)

OUT = REPO / "results" / "z4_active_vg2"
OUT.mkdir(parents=True, exist_ok=True)


class ContextVG2Model(nn.Module):
    """VG2 chosen per step by a policy — the "active-control" condition.

    Architecture (parameter-matched to WorldModelModel hidden=64):
        Input:  [u_t, running_mean(u), running_var(u), t_norm, pop_state_mean]
        MLP:    → hidden → hidden → N   (produces raw VG2 logits)
        VG2(t) = sigmoid(vg2_base + vg2_delta(context))
    """

    def __init__(self, bank: CellBank, n_out: int = 1, hidden: int = 64,
                  use_state: bool = True):
        super().__init__()
        self.bank = bank
        self.hidden = hidden
        self.use_state = use_state
        n_in = bank.n_in
        ctx_dim = n_in + 4 + (1 if use_state else 0)
        self.policy = nn.Sequential(
            nn.Linear(ctx_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, bank.N),
        ).to(bank.device)
        # Init last layer small so VG2 starts near the learnable base
        with torch.no_grad():
            self.policy[-1].weight.mul_(0.01)
            self.policy[-1].bias.zero_()
        self.vg2_base_raw = nn.Parameter(torch.zeros(bank.N, device=bank.device))
        self.readout = nn.Linear(bank.N, n_out, bias=True).to(bank.device)

    def forward(self, U, return_vg2: bool = False):
        T, n_in = U.shape
        VB = torch.zeros(self.bank.N, device=self.bank.device, dtype=U.dtype)
        outs, vg2_trace = [], []
        mu = torch.zeros(n_in, device=U.device, dtype=U.dtype)
        M2 = torch.zeros(n_in, device=U.device, dtype=U.dtype)
        for t in range(T):
            u = U[t]
            delta = u - mu
            mu = mu + delta / (t + 1)
            M2 = M2 + delta * (u - mu)
            var = M2 / max(t + 1, 1)
            stats = torch.stack([u.mean(), mu.mean(), var.mean(),
                                   torch.tensor(float(t) / T,
                                                device=U.device, dtype=U.dtype)])
            ctx = [u, stats]
            if self.use_state:
                ctx.append(torch.tanh(VB).mean().unsqueeze(0))
            ctx = torch.cat(ctx)
            delta_vg2 = self.policy(ctx)
            vg2 = torch.sigmoid(self.vg2_base_raw + delta_vg2)
            VB, s = self.bank.step(VB, u, vg2)
            outs.append(self.readout(s))
            if return_vg2:
                vg2_trace.append(vg2.detach())
        y = torch.stack(outs, 0)
        if return_vg2:
            return y, torch.stack(vg2_trace, 0)
        return y

    @property
    def extra_params(self):
        return sum(p.numel() for p in self.policy.parameters()) + self.bank.N

    @property
    def extra_flops_per_step(self):
        # same as world_model for matched hidden size
        n_in = self.bank.n_in
        ctx_dim = n_in + 4 + (1 if self.use_state else 0)
        h = self.hidden
        return 2 * ctx_dim * h + 2 * h * h + 2 * h * self.bank.N


def freeze_vg2_test(model, U_val, y_val, warmup=50):
    """Clamp VG2 to its running mean (average over test set) and re-evaluate.

    If the active-control model was relying on dynamic VG2, performance
    collapses. If it was just learning a static VG2 profile, no change.
    """
    with torch.no_grad():
        model.eval()
        _, vg2_trace = model(U_val, return_vg2=True)
        vg2_mean = vg2_trace.mean(dim=0)        # (N,)

        # Monkey-patch: override policy + base to always return vg2_mean
        orig_policy = model.policy
        orig_base = model.vg2_base_raw.detach().clone()
        # Replace policy with a zero network, base with inverse-sigmoid(vg2_mean)
        class _ZeroPolicy(nn.Module):
            def __init__(self, N, device, dtype):
                super().__init__()
                self.N, self.device, self.dtype = N, device, dtype
            def forward(self, x):
                return torch.zeros(self.N, device=self.device, dtype=self.dtype)
        model.policy = _ZeroPolicy(model.bank.N, model.bank.device, U_val.dtype)
        vg2_mean_clamped = torch.clamp(vg2_mean, 1e-6, 1 - 1e-6)
        logit = torch.log(vg2_mean_clamped / (1 - vg2_mean_clamped))
        model.vg2_base_raw.data = logit

        pred = model(U_val).squeeze(-1)
        mse_frozen = ((pred[warmup:] - y_val[warmup:]) ** 2).mean().item()

        # Restore
        model.policy = orig_policy
        model.vg2_base_raw.data = orig_base
    return mse_frozen, vg2_mean.cpu().numpy()


def analyze_vg2_dynamics(model, U_val):
    """Log VG2(t, i) during a clean forward pass and compute:
       - per-cell temporal std (does VG2 move over time for each cell?)
       - population std across cells (do cells specialize?)
       - correlation between VG2 and input u(t)
    """
    with torch.no_grad():
        model.eval()
        y_pred, vg2 = model(U_val, return_vg2=True)
    vg2_np = vg2.cpu().numpy()           # (T, N)
    per_cell_std = vg2_np.std(axis=0)    # std of each cell over time
    pop_std = vg2_np.std(axis=1)         # std across cells at each time
    u_np = U_val.cpu().numpy().flatten()
    # Correlation of each cell's VG2 trajectory with u(t)
    corrs = []
    for i in range(vg2_np.shape[1]):
        c = np.corrcoef(vg2_np[:, i], u_np)[0, 1]
        if np.isnan(c):
            c = 0.0
        corrs.append(c)
    return {
        "vg2_trajectory": vg2_np,
        "per_cell_temporal_std_mean":  float(per_cell_std.mean()),
        "per_cell_temporal_std_max":   float(per_cell_std.max()),
        "pop_std_mean":                float(pop_std.mean()),
        "abs_corr_with_u_mean":        float(np.mean(np.abs(corrs))),
        "abs_corr_with_u_p95":         float(np.percentile(np.abs(corrs), 95)),
    }


def run_one(name, model, U_tr, y_tr, U_vl, y_vl, epochs):
    print(f"\n  [{name}]")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    extra = model.extra_params
    print(f"    trainable params: {n_params}  (extra beyond readout: {extra})")
    t0 = time.perf_counter()
    hist, best = train_model(model, U_tr, y_tr, U_vl, y_vl, epochs=epochs)
    t1 = time.perf_counter()
    return {
        "name": name, "final_val": float(best),
        "params": int(n_params), "extra_params": int(extra),
        "flops_per_step": int(model.extra_flops_per_step),
        "train_s": float(t1 - t0),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"[device] {torch.cuda.get_device_name(0)}")

    rng = np.random.default_rng(2026)
    T_TR, T_VL = 600, 300
    u_tr, y_tr = narma10(T_TR, rng); u_vl, y_vl = narma10(T_VL, rng)
    U_tr = torch.tensor(u_tr[:, None], device=device)
    y_tr_t = torch.tensor(y_tr, device=device)
    U_vl = torch.tensor(u_vl[:, None], device=device)
    y_vl_t = torch.tensor(y_vl, device=device)

    N = 1024
    EPOCHS = 200
    DT = 5e-4

    print("\n" + "=" * 70)
    print(f"N = {N}  |  epochs = {EPOCHS}  |  NARMA-10")
    print("=" * 70)

    bank = CellBank(N=N, n_in=1, dt=DT, device=device, seed=7)

    # Build all models once so we can keep context_vg2 for deeper analysis
    fixed_n     = FixedVG2Model(bank, 0.1)
    fixed_s     = FixedVG2Model(bank, 0.9)
    learn_vg2   = LearnableVG2Model(bank, vg2_init=0.5)
    world_mod   = WorldModelModel(bank, hidden=64)
    context_vg2 = ContextVG2Model(bank, hidden=64, use_state=True)

    results = {}
    for name, m in [("fixed_neuron", fixed_n), ("fixed_synapse", fixed_s),
                      ("learnable_vg2", learn_vg2), ("world_model", world_mod),
                      ("context_vg2", context_vg2)]:
        results[name] = run_one(name, m, U_tr, y_tr_t, U_vl, y_vl_t,
                                   epochs=EPOCHS)

    # ── Deep analysis on context_vg2 ──
    print("\n" + "─" * 70)
    print("DEEP ANALYSIS — did the AI actually use the knob dynamically?")
    print("─" * 70)
    dyn = analyze_vg2_dynamics(context_vg2, U_vl)
    print(f"  per-cell temporal std  : mean={dyn['per_cell_temporal_std_mean']:.4f}  "
          f"max={dyn['per_cell_temporal_std_max']:.4f}")
    print(f"  pop-level std per step : mean={dyn['pop_std_mean']:.4f}")
    print(f"  |corr(VG2_i, u(t))|    : mean={dyn['abs_corr_with_u_mean']:.3f}  "
          f"p95={dyn['abs_corr_with_u_p95']:.3f}")

    # ── Freeze test ──
    print("\n" + "─" * 70)
    print("FREEZE TEST — clamp VG2 to its running mean, re-evaluate")
    print("─" * 70)
    mse_frozen, vg2_mean = freeze_vg2_test(context_vg2, U_vl, y_vl_t)
    mse_live = results["context_vg2"]["final_val"]
    print(f"  context_vg2 live      : MSE {mse_live:.5f}")
    print(f"  context_vg2 frozen    : MSE {mse_frozen:.5f}")
    delta_pct = 100 * (mse_frozen - mse_live) / max(mse_live, 1e-9)
    print(f"  degradation from freeze: {delta_pct:+.1f}%  "
          f"({'active control matters' if delta_pct > 10 else 'mostly static profile'})")

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'condition':<20} {'MSE':>10}  {'extra params':>14}  {'FLOPs/step':>12}")
    for k in ("fixed_neuron", "fixed_synapse", "learnable_vg2",
                "world_model", "context_vg2"):
        r = results[k]
        print(f"  {k:<20} {r['final_val']:>10.5f}  {r['extra_params']:>14,}  "
              f"{r['flops_per_step']:>12,}")

    # ── Plots ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 1. VG2 trajectory heatmap (time × cell)
        fig, ax = plt.subplots(figsize=(9, 4))
        traj = dyn["vg2_trajectory"]
        im = ax.imshow(traj.T, aspect="auto", cmap="viridis",
                        origin="lower", vmin=0, vmax=1)
        ax.set_xlabel("time step"); ax.set_ylabel("cell index")
        ax.set_title("Active VG2 trajectory during NARMA-10 task\n"
                      "(bright = synapse mode, dark = neuron mode)")
        fig.colorbar(im, label="VG2")
        fig.tight_layout()
        fig.savefig(OUT / "vg2_trajectory.png", dpi=140)
        plt.close(fig)

        # 2. Per-cell temporal std: does each cell move its knob?
        fig, ax = plt.subplots(figsize=(6, 3.5))
        per_cell = traj.std(axis=0)
        ax.hist(per_cell, bins=40, color="C2", alpha=0.8, edgecolor="k")
        ax.axvline(per_cell.mean(), color="k", ls="--",
                    label=f"mean={per_cell.mean():.4f}")
        ax.set_xlabel("temporal std of VG2 per cell")
        ax.set_ylabel("# cells")
        ax.set_title("Does each cell dynamically move its knob?")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "vg2_per_cell_std.png", dpi=140)
        plt.close(fig)

        # 3. Freeze comparison + MSE bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        conds = ["fixed_neuron", "fixed_synapse", "learnable_vg2",
                   "world_model", "context_vg2"]
        vals = [results[c]["final_val"] for c in conds]
        colors = ["C0", "C0", "C2", "C3", "C1"]
        x = np.arange(len(conds))
        ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="k")
        ax.bar([len(conds)], [mse_frozen], color="gray", alpha=0.7,
                edgecolor="k", hatch="//")
        for i, v in enumerate(vals + [mse_frozen]):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(list(x) + [len(conds)])
        ax.set_xticklabels(conds + ["ctx_vg2\nFROZEN"], rotation=25, fontsize=9)
        ax.set_ylabel("validation MSE (NARMA-10)")
        ax.set_title(f"N={N} — active control vs alternatives")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "condition_comparison.png", dpi=140)
        plt.close(fig)

        print(f"\n[plot] 3 figures → {OUT}")
    except ImportError:
        pass

    summary = {
        "N": N, "epochs": EPOCHS,
        "task": "NARMA-10", "results": results,
        "context_dynamics": {
            k: v for k, v in dyn.items() if k != "vg2_trajectory"
        },
        "freeze_test": {
            "mse_live": float(mse_live),
            "mse_frozen": float(mse_frozen),
            "degradation_pct": float(delta_pct),
        },
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] → {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
