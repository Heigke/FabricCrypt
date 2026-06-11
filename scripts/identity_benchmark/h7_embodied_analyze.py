"""Plot v5 embodied LM learning trajectory.

Reads jsonl log → 4 panels:
  1. pred_acc + baseline_acc rolling (embodiment learning curve)
  2. local_loss (substrate prediction MSE)
  3. eta (homeostatic plasticity adaptation)
  4. Gate magnitudes per layer (modulation strength)
  5. acute σ (how much current substrate deviates from baseline)
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = Path("results/IDENTITY_EMBODIED_2026-06-10/embodied_v5_ikaros.jsonl")
OUT = LOG.parent / "embodied_v5_trajectory.png"

entries = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
print(f"loaded {len(entries)} entries")

step = np.array([e["step"] for e in entries])
ll   = np.array([e["local_loss"] for e in entries])
acc  = np.array([e["pred_acc"] for e in entries])
bacc = np.array([e["baseline_acc"] for e in entries])
ppl  = np.array([e["lm_ppl"] for e in entries])
eta  = np.array([e["eta"] for e in entries])
acute = np.array([e["acute_sigma"] for e in entries])
gn   = np.array([e["grad_norm"] for e in entries])
alphas = np.array([e["alphas"] for e in entries])
slept  = np.array([e["slept"] for e in entries])

fig, axes = plt.subplots(5, 1, figsize=(13, 14), sharex=True)

ax = axes[0]
ax.plot(step, acc, lw=0.5, color="tab:blue", alpha=0.4, label="pred_acc (per step)")
ax.plot(step, bacc, lw=1.5, color="tab:red", label="baseline_acc (rolling EMA)")
ax.axhline(0.317, ls="--", color="gray", lw=0.5, label="random chance (1σ, 10ch)")
ax.set_ylabel("substrate prediction acc\n(fraction within 1σ)")
ax.set_title("Embodied LM — does it learn its own body?")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
ax.semilogy(step, ll, color="tab:purple", lw=0.7)
ax.set_ylabel("local_loss (MSE)")
ax.set_title("Substrate prediction MSE")
ax.grid(alpha=0.3)

ax = axes[2]
ax.semilogy(step, eta, color="tab:green", lw=1.2)
ax.set_ylabel("η (plasticity)")
ax.set_title("Homeostatic critic — plasticity adjusts to performance")
ax.axhline(2e-4, ls=":", color="gray", lw=0.5, alpha=0.5)
ax.grid(alpha=0.3)

ax = axes[3]
for i in range(alphas.shape[1]):
    ax.plot(step, np.abs(alphas[:, i]), lw=0.8, label=f"layer {[25,28][i]}")
ax.set_ylabel("|α| per xattn layer")
ax.set_title("Cross-attention gate opening (how much does substrate matter?)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[4]
ax.plot(step, acute, color="tab:orange", lw=0.6)
ax.axhline(3.0, ls="--", color="red", lw=0.5, label="acute threshold (acclimatization)")
# Mark sleep cycles
for s, sl in zip(step, slept):
    if sl: ax.axvline(s, color="tab:cyan", alpha=0.15, lw=0.5)
ax.set_ylabel("acute σ from baseline")
ax.set_xlabel("step (each = 32 tokens)")
ax.set_title("Substrate acute distance (vertical cyan lines = sleep cycles)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT, dpi=110)
print(f"saved {OUT}")
print()
print("=== SUMMARY ===")
print(f"  total steps logged: {len(entries)}")
print(f"  first pred_acc:  {acc[0]:.3f}")
print(f"  last  pred_acc:  {acc[-1]:.3f}")
print(f"  first baseline_acc:  {bacc[0]:.3f}")
print(f"  last  baseline_acc:  {bacc[-1]:.3f}")
print(f"  baseline improvement: {bacc[-1]-bacc[0]:+.3f}")
print(f"  eta start → end: {eta[0]:.1e} → {eta[-1]:.1e}")
print(f"  final |α| layer 25: {abs(alphas[-1,0]):.4f}")
print(f"  final |α| layer 28: {abs(alphas[-1,1]):.4f}")
print(f"  sleep cycles triggered: {int(slept.sum())}")
print(f"  PPL trajectory:  {ppl[0]:.1f} → {ppl[-1]:.1f} (NOT primary metric)")
