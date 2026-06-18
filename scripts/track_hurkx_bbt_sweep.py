#!/usr/bin/env python3
"""Track Hurkx BBT (band-to-band tunneling) body-charging sweep.

Hypothesis (Grok #2): A Hurkx 1992 (IEEE TED) BBT/TAT mechanism at the
drain-body junction provides a body-charging current that turns ON at
LOWER Vd than impact-ionization (Iii). This could explain the model vs
data knee gap (model ≈ 1.5V, data ≈ 0.85-1.15V at VG1=0.6) WITHOUT
degrading the K1+ALPHA0 full-33 0.665-decade fit.

Physics:
  I_BBT = A · |Vd - Vb|^P · exp(-B / max(|Vd - Vb|, 0.01))
  P=2.5 (Hurkx canonical field-power)
  Sign: +INTO body (hole accumulation from BBT e-h pairs at the drain-
  body depletion region). Gated behind cfg.hurkx_bbt_A > 0 (default 0).

Grid:
  hurkx_bbt_A ∈ {0, 1e-8, 1e-6, 1e-4}  ×  hurkx_bbt_B ∈ {0.3, 0.7, 1.5}
  = 12 conditions. K1@VG1=0.6 = 0.53825 + ALPHA0 = 7.83756e-4 locked.

Measurements per condition:
  • knee_vd at VG1=0.6 × VG2 ∈ {-0.1, 0, 0.1, 0.2}
  • full 33-bias median_dec (fwd+bwd, n=66)

PASS gate: |mean_model_knee − mean_data_knee| ≤ 0.2V AND median_dec ≤ 0.766.
Best (A,B) by combined criterion:
    obj = (model_minus_data)^2 + (median_dec - 0.665)^2

Outputs (results/track_hurkx_bbt/):
  ablation.json, verdict.md, plot.png
"""
from __future__ import annotations
import os, sys, json, time, traceback
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

import importlib.util
sp = importlib.util.spec_from_file_location("pillar_I", ROOT / "scripts/pillar_I_C3_jts_tat.py")
pillar = importlib.util.module_from_spec(sp); sp.loader.exec_module(pillar)

OUT = ROOT / "results/track_hurkx_bbt"
OUT.mkdir(parents=True, exist_ok=True)

# K1+ALPHA0 card values (locked)
K1_CARD     = 0.53825
ALPHA0_CARD = 7.83756e-4

# Hurkx grid (task spec)
A_GRID = [0.0, 1e-8, 1e-6, 1e-4]
B_GRID = [0.3, 0.7, 1.5]

# VG1=0.6 slice, VG2 grid (knee measurement)
VG1_TARGET = 0.6
VG2_GRID = [-0.1, 0.0, 0.1, 0.2]

# Gates
GATE_KNEE_SHIFT = 0.2       # |model−data| ≤ 0.2V
GATE_MEDIAN_DEC = 0.766     # median dec ≤ 0.766


# ── Thermal monitor ───────────────────────────────────────────────
def cpu_temp_c() -> float:
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        return float("nan")


def wait_cool(threshold_c: float = 75.0, target_c: float = 50.0, timeout_s: float = 180.0):
    t0 = time.time()
    while True:
        t = cpu_temp_c()
        if not np.isfinite(t) or t <= threshold_c:
            return
        if time.time() - t0 > timeout_s:
            print(f"[hurkx] thermal timeout (T={t:.1f}°C)", flush=True)
            return
        print(f"[hurkx] T={t:.1f}°C > {threshold_c}°C — waiting for cool-down to {target_c}°C", flush=True)
        time.sleep(5.0)
        if cpu_temp_c() < target_c:
            return


# ── Knee detector ────────────────────────────────────────────────
def knee_vd(Vd: np.ndarray, Id: np.ndarray) -> float:
    """Vd at which Id first exceeds 10× median(Id[Vd in [0, 0.3]]); NaN if never."""
    Vd = np.asarray(Vd, dtype=np.float64)
    Id = np.abs(np.asarray(Id, dtype=np.float64))
    base_mask = (Vd >= 0.0) & (Vd <= 0.3) & np.isfinite(Id)
    if base_mask.sum() < 2:
        return float("nan")
    base = np.median(Id[base_mask])
    if not np.isfinite(base) or base <= 0:
        base = max(base, 1e-15)
    thresh = 10.0 * base
    for v, c in zip(Vd, Id):
        if v > 0.3 and np.isfinite(c) and c > thresh:
            return float(v)
    return float("nan")


# ── Pick VG1=0.6 curves (for knee)
def pick_curves(curves):
    out = []
    for vg2 in VG2_GRID:
        match = None
        for c in curves:
            if abs(c["VG1"] - VG1_TARGET) < 1e-6 and abs(c["VG2"] - vg2) < 1e-6:
                match = c; break
        if match is None:
            print(f"[hurkx] WARNING no curve for VG1=0.6 VG2={vg2:+.2f}", flush=True)
        out.append((vg2, match))
    return out


# ── Patched make_overrides (K1+ALPHA0 lock) ──────────────────────
def install_combo_patch():
    """Install K1+ALPHA0 monkey patch on pillar. Returns restore-fn."""
    saved_branch_k1 = pillar.BRANCH_FLAT[VG1_TARGET]["K1"]
    pillar.BRANCH_FLAT[VG1_TARGET]["K1"] = float(K1_CARD)
    orig_make = pillar.make_overrides

    def patched_make(sebas_row):
        P_M1, P_M2 = orig_make(sebas_row)
        if P_M1 is None: P_M1 = {}
        if P_M2 is None: P_M2 = {}
        P_M1["alpha0"] = float(ALPHA0_CARD)
        P_M2["alpha0"] = float(ALPHA0_CARD)
        if sebas_row is not None and abs(sebas_row.get("VG1", float("nan")) - VG1_TARGET) < 1e-6:
            P_M1["k1"] = float(K1_CARD)
        return P_M1, P_M2
    pillar.make_overrides = patched_make

    def restore():
        pillar.make_overrides = orig_make
        pillar.BRANCH_FLAT[VG1_TARGET]["K1"] = saved_branch_k1
    return restore


# ── Per-condition runner ─────────────────────────────────────────
def simulate_curve(cfg, M1, M2, bjt, c, sebas_rows):
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)
    Vd_np = c["fwd_Vd"]
    row_sebas, _ = pillar.find_or_impute_row(sebas_rows, c["VG1"], c["VG2"])
    P_M1, P_M2 = pillar.make_overrides(row_sebas)
    Vd = torch.tensor(Vd_np, dtype=torch.float64)
    try:
        with pillar.patch_sd_scaled(sd_M1, P_M1), pillar.patch_sd_scaled(sd_M2, P_M2):
            out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd,
                             VG1=torch.tensor(c["VG1"], dtype=torch.float64),
                             VG2=torch.tensor(c["VG2"], dtype=torch.float64),
                             warm_start=True)
        I_pred = np.abs(out["Id"].detach().cpu().numpy()).astype(np.float64)
        I_pred = np.where(np.isfinite(I_pred), I_pred, 0.0)
    except Exception as e:
        print(f"[hurkx] FAIL sim {c['f']}: {e}", flush=True)
        I_pred = np.zeros_like(Vd_np)
    return Vd_np, I_pred


def run_one(A: float, B: float, sel_curves, sebas_rows, curves_all):
    tag = f"A={A:.0e}_B={B:.2f}"
    print(f"[hurkx] === {tag} ===", flush=True)
    wait_cool()

    cfg, M1, M2, bjt = pillar.build_pyport_base()
    cfg.hurkx_bbt_A = float(A)
    cfg.hurkx_bbt_B = float(B)

    restore = install_combo_patch()
    per_vg2 = []
    try:
        # (a) knee curves at VG1=0.6 × 4 VG2
        for vg2, c in sel_curves:
            if c is None:
                per_vg2.append({"VG2": vg2, "data_knee_V": float("nan"),
                                "model_knee_V": float("nan")})
                continue
            Vd_np, I_pred = simulate_curve(cfg, M1, M2, bjt, c, sebas_rows)
            Id_data = np.abs(c["fwd_Id"])
            data_k  = knee_vd(c["fwd_Vd"], Id_data)
            model_k = knee_vd(Vd_np,        I_pred)
            print(f"[hurkx]   VG2={vg2:+.2f}  data={data_k:.3f}V  model={model_k:.3f}V", flush=True)
            per_vg2.append({
                "VG2": float(vg2),
                "file": c["f"],
                "data_knee_V":  float(data_k),
                "model_knee_V": float(model_k),
                "Vd":      Vd_np.tolist(),
                "Id_data": Id_data.tolist(),
                "Id_pred": I_pred.tolist(),
            })

        # (b) full 33-bias fwd+bwd dec (n=66)
        wait_cool()
        rows, nan_count = pillar.run_grid(cfg, M1, M2, bjt, curves_all, sebas_rows,
                                          label=tag, do_bwd=True)
        all_med = np.array([r["med_dec"] for r in rows if np.isfinite(r["med_dec"])])
        if all_med.size > 0:
            median_dec = float(np.median(all_med))
        else:
            median_dec = float("nan")
        print(f"[hurkx]   full-33 median_dec = {median_dec:.4f}  (n={all_med.size}, nan={nan_count})", flush=True)
    finally:
        restore()

    # Aggregate knee stats
    data_knees  = np.array([d["data_knee_V"]  for d in per_vg2], dtype=np.float64)
    model_knees = np.array([d["model_knee_V"] for d in per_vg2], dtype=np.float64)
    valid = np.isfinite(data_knees) & np.isfinite(model_knees)
    mean_data  = float(np.nanmean(data_knees))
    mean_model = float(np.nanmean(model_knees))
    if valid.any():
        diff = float(np.mean(model_knees[valid] - data_knees[valid]))
        abs_shift = float(abs(diff))
    else:
        diff = float("nan"); abs_shift = float("nan")

    return {
        "A": float(A), "B": float(B),
        "per_vg2": per_vg2,
        "mean_data_knee_V":  mean_data,
        "mean_model_knee_V": mean_model,
        "model_minus_data_V": diff,
        "abs_knee_shift_V":  abs_shift,
        "median_dec_n66":    median_dec,
        "knee_pass":         bool(np.isfinite(abs_shift) and abs_shift <= GATE_KNEE_SHIFT),
        "dec_pass":          bool(np.isfinite(median_dec) and median_dec <= GATE_MEDIAN_DEC),
    }


# ── Plot ────────────────────────────────────────────────────────
def make_plot(results, sel_curves):
    n_cells = len(results)
    fig, axes = plt.subplots(len(B_GRID), len(A_GRID), figsize=(4.0*len(A_GRID), 3.2*len(B_GRID)),
                              sharex=True, sharey=True, squeeze=False)
    colors = plt.cm.viridis(np.linspace(0, 0.92, len(VG2_GRID)))
    for i, B in enumerate(B_GRID):
        for j, A in enumerate(A_GRID):
            ax = axes[i][j]
            tag = f"A={A:.0e}_B={B:.2f}"
            r = results.get(tag)
            for kk, (vg2, c) in enumerate(sel_curves):
                if c is not None:
                    ax.semilogy(c["fwd_Vd"], np.clip(np.abs(c["fwd_Id"]), 1e-15, None),
                                color=colors[kk], ls="--", lw=1.0, alpha=0.6)
            if r is not None:
                for kk, d in enumerate(r["per_vg2"]):
                    if d.get("Vd"):
                        ax.semilogy(d["Vd"], np.clip(np.array(d["Id_pred"]), 1e-15, None),
                                    color=colors[kk], lw=1.2,
                                    label=f"VG2={d['VG2']:+.2f}")
                title = (f"A={A:.0e}, B={B:.2f}\n"
                         f"shift={r['abs_knee_shift_V']:.3f}V  dec={r['median_dec_n66']:.3f}")
            else:
                title = f"A={A:.0e}, B={B:.2f}\n(missing)"
            ax.set_title(title, fontsize=8)
            ax.grid(True, which="both", alpha=0.3)
            if i == len(B_GRID) - 1:
                ax.set_xlabel("Vd [V]")
            if j == 0:
                ax.set_ylabel("|Id| [A]")
    axes[0][0].legend(fontsize=6, loc="lower right")
    fig.suptitle("Hurkx BBT (A, B) sweep — knees at VG1=0.6 (dashed=data, solid=model)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "plot.png", dpi=120)
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    sebas_rows = pillar.load_sebas_params()
    curves = pillar.load_curves()
    print(f"[hurkx] loaded {len(curves)} curves, {len(sebas_rows)} sebas rows", flush=True)
    sel_curves = pick_curves(curves)
    print(f"[hurkx] selected {sum(1 for _,c in sel_curves if c is not None)}/{len(VG2_GRID)} VG1=0.6 curves", flush=True)

    results = {}
    for B in B_GRID:
        for A in A_GRID:
            tag = f"A={A:.0e}_B={B:.2f}"
            try:
                results[tag] = run_one(A, B, sel_curves, sebas_rows, curves)
            except Exception as e:
                traceback.print_exc()
                results[tag] = {"A": A, "B": B, "error": str(e)}
            with open(OUT / "ablation.json", "w") as f:
                json.dump(results, f, indent=2, default=str)

    # Combined objective
    best_tag, best_obj = None, float("inf")
    for tag, r in results.items():
        if "error" in r: continue
        s = r.get("abs_knee_shift_V", float("nan"))
        d = r.get("median_dec_n66", float("nan"))
        if not (np.isfinite(s) and np.isfinite(d)):
            continue
        # Use model−data signed shift in obj (square anyway) — penalize ABSOLUTE knee gap.
        obj = (r.get("model_minus_data_V", s)) ** 2 + (d - 0.665) ** 2
        r["obj"] = float(obj)
        if obj < best_obj:
            best_obj = obj; best_tag = tag

    with open(OUT / "ablation.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    try:
        make_plot(results, sel_curves)
    except Exception as e:
        print(f"[hurkx] plot failed: {e}", flush=True)

    # verdict.md
    lines = []
    lines.append("# Hurkx BBT (band-to-band tunneling) body-charging sweep\n")
    lines.append("**Hypothesis (Grok #2):** A Hurkx 1992 (IEEE TED) BBT/TAT mechanism at the")
    lines.append("drain–body junction provides body-charging current that turns ON at LOWER Vd")
    lines.append("than impact-ionization, shifting the snapback knee toward the data (≈0.85-")
    lines.append("1.15V at VG1=0.6) WITHOUT degrading the K1+ALPHA0 0.665-decade full-33 fit.\n")
    lines.append("**Implementation:** `I_BBT = A · |Vd-Vb|^P · exp(-B/max(|Vd-Vb|, 0.01))`,")
    lines.append("P=2.5, sign +INTO body. Gated behind `cfg.hurkx_bbt_A > 0` (default 0 = inert).")
    lines.append(f"Locked baseline: K1@VG1=0.6 = {K1_CARD}, ALPHA0 = {ALPHA0_CARD:.4e}.\n")

    lines.append("## Grid results\n")
    header = "| A [A·V^-P] | B [V] | mean_data_knee | mean_model_knee | model−data | |shift| | median_dec(n=66) | knee_PASS | dec_PASS | obj |"
    sep    = "|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|---:|"
    lines.append(header); lines.append(sep)
    for B in B_GRID:
        for A in A_GRID:
            tag = f"A={A:.0e}_B={B:.2f}"
            r = results.get(tag)
            if r is None or "error" in r:
                err = (r or {}).get("error", "missing")[:40]
                lines.append(f"| {A:.0e} | {B:.2f} | ERROR: {err} |")
                continue
            obj = r.get("obj", float("nan"))
            lines.append(
                f"| {A:.0e} | {B:.2f} | {r['mean_data_knee_V']:.3f} | "
                f"{r['mean_model_knee_V']:.3f} | {r['model_minus_data_V']:+.3f} | "
                f"{r['abs_knee_shift_V']:.3f} | {r['median_dec_n66']:.4f} | "
                f"{'YES' if r['knee_pass'] else 'NO'} | "
                f"{'YES' if r['dec_pass'] else 'NO'} | "
                f"{obj:.4f} |"
            )

    lines.append("\n## Verdict\n")
    if best_tag is None:
        lines.append("- All cells failed or produced NaN — no best candidate.")
        lines.append("- **OVERALL: FAIL** (no usable Hurkx BBT setting).")
    else:
        r = results[best_tag]
        knee_pass = r["knee_pass"]; dec_pass = r["dec_pass"]
        overall = bool(knee_pass and dec_pass)
        lines.append(f"- **Best (A, B)** by combined objective `(model−data)^2 + (dec−0.665)^2`:")
        lines.append(f"  - A = {r['A']:.2e}  A·V^-P")
        lines.append(f"  - B = {r['B']:.2f} V")
        lines.append(f"  - mean_data_knee = {r['mean_data_knee_V']:.3f} V")
        lines.append(f"  - mean_model_knee = {r['mean_model_knee_V']:.3f} V  →  shift = {r['model_minus_data_V']:+.3f} V  (|shift| = {r['abs_knee_shift_V']:.3f} V)")
        lines.append(f"  - median_dec (n=66) = {r['median_dec_n66']:.4f}")
        lines.append(f"  - knee gate (|shift| ≤ {GATE_KNEE_SHIFT} V): {'PASS' if knee_pass else 'FAIL'}")
        lines.append(f"  - dec gate (median_dec ≤ {GATE_MEDIAN_DEC}): {'PASS' if dec_pass else 'FAIL'}")
        lines.append("")
        if overall:
            lines.append("- **OVERALL: PASS** — Hurkx BBT shifts the snapback knee into the")
            lines.append("  data window without breaking the 33-bias decade fit. Mechanism is")
            lines.append("  physically defensible (Hurkx 1992 IEEE TED).")
        else:
            lines.append("- **OVERALL: FAIL** — best (A, B) misses at least one gate. Reporting null.")

    lines.append("\n## Provenance")
    lines.append(f"- baseline: `pillar_I_C3_jts_tat.build_pyport_base()`")
    lines.append(f"- K1+ALPHA0 monkey patch applied (K1@VG1=0.6 = {K1_CARD}; ALPHA0 = {ALPHA0_CARD:.4e})")
    lines.append(f"- BBT residual: `R_B += A · |Vd-Vb|^{2.5} · exp(-B/max(|Vd-Vb|, 0.01))` (clamped via tanh to `snap_Iii_clamp`)")
    lines.append(f"- knee_vd: first Vd>0.3 where |Id|>10× median(|Id|[Vd∈[0,0.3]])")
    lines.append(f"- median_dec: median over all fwd+bwd, all 33 biases (n=66)")
    lines.append(f"- runtime: {time.time() - t0:.1f}s")

    (OUT / "verdict.md").write_text("\n".join(lines) + "\n")
    print(f"[hurkx] wrote {OUT / 'verdict.md'}", flush=True)
    print(f"[hurkx] wrote {OUT / 'ablation.json'}", flush=True)
    print(f"[hurkx] wrote {OUT / 'plot.png'}", flush=True)
    print(f"[hurkx] total runtime: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
