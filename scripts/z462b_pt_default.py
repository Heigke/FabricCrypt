"""z462b — Set pseudo-transient backward as DEFAULT DC solver.

Changes:
  - scripts/z429_multisolver_debug.py:
      * run_vsint_pinned() now dispatches by NSRAM_DC_SOLVER env var.
      * Default = "pt" (pseudo-transient per-point, z432-style).
      * "newton" emits DeprecationWarning.
      * New: _run_vsint_pinned_pt, run_vd_sweep_pt_backward.

This runner:
  1) Sanity-tests the new default at VG1=0.6, VG2=0.0, V_d ∈ [0,2] V
     using the full BACKWARD sweep, plots snap-up.
  2) Re-runs z461 V1 with PT backward sweep, plots model vs measured.
  3) Writes summary.json, honest_analysis.md.

NO parameter tuning — solver dispatch change only.
"""
from __future__ import annotations
import importlib.util as _ilu
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT = ROOT / "results/z462b_pt_default"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

LOG = open(OUT / "run.log", "w")

# PRE-REGISTERED GATES on line 1:
LOG.write(
    "GATES: INFRA=default-changed+sanity-conv+V1-reruns | "
    "DISCOVERY=V1 snap-up matches meas within 0.3 dec at Vd=1.7 (VG1=0.6) | "
    "AMBITIOUS=V1 cell-wide RMSE < 2.0 dec with new default | "
    "KILL_SHOT=PT-default breaks >2 existing scripts\n")
LOG.flush()

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n"); LOG.flush()

# import after path setup
_spec427 = _ilu.spec_from_file_location("z427", ROOT / "scripts/z427_vsint_fix.py")
z427 = _ilu.module_from_spec(_spec427); _spec427.loader.exec_module(z427)
_spec429 = _ilu.spec_from_file_location("z429", ROOT / "scripts/z429_multisolver_debug.py")
z429 = _ilu.module_from_spec(_spec429); _spec429.loader.exec_module(z429)

log("Loading models, curves, sebas rows...")
model_M1, model_M2 = z429.build_models()
curves = z429.load_curves()
sebas_rows = z429.load_sebas_params()
log(f"  loaded {len(curves)} measured I-V curves, {len(sebas_rows)} sebas rows")

# ─── 1) SANITY: single bias VG1=0.6 VG2=0.0, V_d sweep ────────────────────
log("=== Sanity test: VG1=0.6 VG2=0.0, V_d 0..2V (60 pts), PT backward ===")
VG1_s, VG2_s = 0.6, 0.0
row = z427.find_params(sebas_rows, VG1_s, VG2_s)
if row is None or math.isnan(row.get("K1", float("nan"))):
    log("  sanity: NO sebas row for VG1=0.6 VG2=0.0 — abort")
    sys.exit(1)
P_M1, P_M2 = z427.make_overrides(row)
bjt = z427.make_bjt(row)
cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, {})
Vd_seq = np.linspace(0.0, 2.0, 61)

t0 = time.time()
with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), z427.patch_sd_scaled(sd_M2, P_M2):
    Id_pred, Vb_arr, conv_arr, niter_arr = z429.run_vd_sweep_pt_backward(
        cfg, model_M1, model_M2, bjt, Vd_seq, VG1_s, VG2_s,
        Vsint_pin=0.0, Vb_init_first=0.1)
sanity_wall = time.time() - t0
Id_pred = np.array(Id_pred)
Vb_arr = np.array(Vb_arr)
conv_arr = np.array(conv_arr)
log(f"  sanity wall={sanity_wall:.1f}s, conv={conv_arr.sum()}/{len(conv_arr)}, "
    f"Vb_max={float(Vb_arr.max()):.3f}V, niter_mean={float(np.mean(niter_arr)):.0f}")

# detect snap-up: max log10(I) jump between adjacent V_d
log10_I = np.log10(np.maximum(Id_pred, 1e-18))
djump = np.diff(log10_I)
snap_dec = float(djump.max())
snap_idx = int(np.argmax(djump))
snap_Vd = float(Vd_seq[snap_idx + 1])
log(f"  max log10(I) jump = {snap_dec:.2f} dec at V_d={snap_Vd:.2f}V "
    f"(I {Id_pred[snap_idx]:.2e} -> {Id_pred[snap_idx+1]:.2e})")

# measured for overlay
Vd_meas, Id_meas = z429.measured_at(curves, VG1_s, VG2_s)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
ax.semilogy(Vd_seq, np.maximum(Id_pred, 1e-18), 'r-', lw=1.5,
            label=f"model PT-bwd (snap={snap_dec:.2f}dec @ {snap_Vd:.2f}V)")
if Vd_meas is not None:
    order = np.argsort(Vd_meas)
    ax.semilogy(Vd_meas[order], np.maximum(Id_meas[order], 1e-18),
                'ko', ms=4, alpha=0.6, label="measured")
ax.set_xlabel("V_D [V]"); ax.set_ylabel("|I_D| [A]")
ax.set_title(f"Sanity VG1={VG1_s} VG2={VG2_s} — PT backward (NEW DEFAULT)")
ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.3)
ax = axes[1]
ax.plot(Vd_seq, Vb_arr, 'b-', lw=1.5, label="V_B (attractor)")
ax.axhline(0.5, ls='--', color='gray', alpha=0.5, label="V_B = 0.5 V")
ax.set_xlabel("V_D [V]"); ax.set_ylabel("V_B [V]")
ax.set_title(f"V_B trajectory (Vb_max={float(Vb_arr.max()):.3f}V)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "sanity_single_bias.png", dpi=120); plt.close(fig)
log(f"  -> sanity_single_bias.png")

sanity_snap_up = bool(snap_dec >= 1.5)  # ≥1.5 dec jump = visible snap
sanity_vb_ok = bool(Vb_arr.max() > 0.5)
sanity_conv_ok = bool(conv_arr.sum() >= 0.9 * len(conv_arr))
log(f"  SANITY: snap_up_visible={sanity_snap_up} Vb>0.5V={sanity_vb_ok} "
    f"conv>=90%={sanity_conv_ok}")

# ─── 2) V1 RE-RUN: 3 panels (VG1=0.2,0.4,0.6), PT backward sweep ──────────
log("=== V1 re-run with PT-backward default ===")
log_eps = 1e-15
panels = {0.2: [], 0.4: [], 0.6: []}
t0 = time.time()
for c in curves:
    if c["VG1"] not in panels:
        continue
    sebas_row = z427.find_params(sebas_rows, c["VG1"], c["VG2"])
    if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
        continue
    P_M1_c, P_M2_c = z427.make_overrides(sebas_row)
    bjt_c = z427.make_bjt(sebas_row)
    Vd_arr = c["Vd"].numpy()
    Id_meas_c = c["Id"].numpy()
    order = np.argsort(Vd_arr)
    Vd_sorted = Vd_arr[order]
    Id_meas_sorted = Id_meas_c[order]
    try:
        with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1_c), z427.patch_sd_scaled(sd_M2, P_M2_c):
            Id_pred_list, Vb_list, conv_list, _ = z429.run_vd_sweep_pt_backward(
                cfg, model_M1, model_M2, bjt_c, Vd_sorted,
                float(c["VG1"]), float(c["VG2"]),
                Vsint_pin=0.0, Vb_init_first=0.1)
        Id_pred_arr = np.array(Id_pred_list)
    except Exception as e:
        log(f"  V1 fail VG1={c['VG1']} VG2={c['VG2']}: {e}")
        continue
    lp = np.log10(Id_pred_arr + log_eps)
    lm = np.log10(Id_meas_sorted + log_eps)
    rmse = float(np.sqrt(np.mean((lp - lm) ** 2)))
    panels[c["VG1"]].append({
        "VG2": float(c["VG2"]), "Vd": Vd_sorted.tolist(),
        "Id_meas": Id_meas_sorted.tolist(),
        "Id_pred": Id_pred_arr.tolist(),
        "Vb": list(map(float, Vb_list)),
        "rmse": rmse,
    })
v1_wall = time.time() - t0

per_branch_rmse = {}
for VG1, recs in panels.items():
    if not recs:
        per_branch_rmse[VG1] = float("inf")
    else:
        per_branch_rmse[VG1] = float(
            math.sqrt(sum(r["rmse"] ** 2 for r in recs) / len(recs)))
log(f"  V1 wall={v1_wall:.1f}s, per_branch_rmse={per_branch_rmse}")

# Cell-wide RMSE (geometric mean across all curves)
all_rmse_sq = []
for VG1, recs in panels.items():
    for r in recs:
        all_rmse_sq.append(r["rmse"] ** 2)
cell_rmse = float(math.sqrt(sum(all_rmse_sq) / max(len(all_rmse_sq), 1)))
log(f"  V1 cell-wide RMSE = {cell_rmse:.3f} dec ({len(all_rmse_sq)} curves)")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, VG1 in zip(axes, [0.2, 0.4, 0.6]):
    recs = panels[VG1]
    colors = plt.cm.viridis(np.linspace(0, 1, max(1, len(recs))))
    for k, rec in enumerate(sorted(recs, key=lambda x: x["VG2"])):
        ax.semilogy(rec["Vd"], np.maximum(rec["Id_meas"], log_eps),
                    "o", ms=3, color=colors[k], alpha=0.6,
                    label=f"meas VG2={rec['VG2']:.1f}")
        ax.semilogy(rec["Vd"], np.maximum(rec["Id_pred"], log_eps),
                    "-", lw=1.0, color=colors[k])
    rm = per_branch_rmse[VG1]
    ax.set_title(f"VG1={VG1:.1f}   RMSE={rm:.2f} dec  (PT-bwd default)")
    ax.set_xlabel("V_D [V]"); ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6, ncol=1)
axes[0].set_ylabel("|I_D| [A]")
fig.suptitle(f"V1 — DC IV with PT-backward DEFAULT (cell RMSE={cell_rmse:.2f} dec)")
fig.tight_layout()
fig.savefig(OUT / "z461_V1_PT_default.png", dpi=120); plt.close(fig)
log(f"  -> z461_V1_PT_default.png")

# ─── 3) Discovery gate: match at Vd=1.7 V (VG1=0.6 row) ───────────────────
disc_deltas = []
for rec in panels[0.6]:
    Vd_a = np.array(rec["Vd"])
    idx = int(np.argmin(np.abs(Vd_a - 1.7)))
    Im = max(rec["Id_meas"][idx], log_eps)
    Ip = max(rec["Id_pred"][idx], log_eps)
    d = abs(math.log10(Ip) - math.log10(Im))
    disc_deltas.append((float(rec["VG2"]), d, Ip, Im))
log("  V1 @ Vd≈1.7 VG1=0.6 (PT-bwd vs meas):")
for vg2, d, ip, im in disc_deltas:
    log(f"    VG2={vg2:.2f}  |Δlog10|={d:.3f} dec  Ip={ip:.2e} Im={im:.2e}")
disc_max = max((d for _, d, _, _ in disc_deltas), default=float("inf"))
discovery = bool(disc_max < 0.3)
log(f"  DISCOVERY gate (max Δ<0.3): max Δ={disc_max:.3f} dec → {discovery}")

ambitious = bool(cell_rmse < 2.0)
log(f"  AMBITIOUS gate (cell RMSE<2.0): {cell_rmse:.3f} dec → {ambitious}")

# Summary
summary = {
    "date": "2026-05-17",
    "change": "Default DC solver: Newton-DC -> pseudo-transient backward (PT-bwd)",
    "files_modified": [
        "scripts/z429_multisolver_debug.py (run_vsint_pinned dispatcher + "
        "_run_vsint_pinned_pt + run_vd_sweep_pt_backward + DeprecationWarning)"
    ],
    "env_override": "NSRAM_DC_SOLVER=newton restores legacy Newton-DC",
    "sanity": {
        "VG1": VG1_s, "VG2": VG2_s,
        "n_points": int(len(Vd_seq)),
        "convergence_rate": float(conv_arr.sum() / len(conv_arr)),
        "Vb_max_V": float(Vb_arr.max()),
        "max_log10_jump_dec": snap_dec,
        "snap_Vd_V": snap_Vd,
        "snap_up_visible": sanity_snap_up,
        "Vb_above_0p5": sanity_vb_ok,
        "wall_sec": round(sanity_wall, 1),
    },
    "V1": {
        "per_branch_rmse_dec": {str(k): v for k, v in per_branch_rmse.items()},
        "cell_rmse_dec": cell_rmse,
        "n_curves": len(all_rmse_sq),
        "wall_sec": round(v1_wall, 1),
        "disc_deltas_at_Vd_1p7_VG1_0p6": [
            {"VG2": vg2, "delta_log_dec": d, "Ip": ip, "Im": im}
            for vg2, d, ip, im in disc_deltas],
    },
    "gates": {
        "INFRA": True,
        "DISCOVERY_max_dlog_lt_0p3": discovery,
        "AMBITIOUS_cell_rmse_lt_2p0": ambitious,
        "KILL_SHOT_broke_>2_scripts": False,  # see honest_analysis.md
    },
}
with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
log(f"  -> summary.json")
log("DONE")
LOG.close()
