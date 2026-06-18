#!/usr/bin/env python3
"""Track VG2-coupled Hurkx BBT sweep.

Locks K1=0.53825, ALPHA0=7.83756e-4, A=1e-6, B=1.5 (the K1+ALPHA0 card-fix
+ Hurkx baseline that PASSES the knee gate but is VG2-INVARIANT). Sweeps a
new coupling coefficient `cfg.hurkx_vg2_coeff` ∈ {0.0, 0.5, 1.0, 2.0, 3.0}.

B_eff = B · (1 + hurkx_vg2_coeff · VG2)

Data (Sebas, VG1=0.6):
  • VG2=-0.1 → knee ≈ 0.85 V
  • VG2=+0.2 → knee ≈ 1.15 V
  → expected +0.30 V shift across the VG2 grid.

For each coeff:
  • measure knee_vd at VG1=0.6 × VG2 ∈ {-0.1, 0.0, 0.1, 0.2}
  • measure full 33-bias median_dec (fwd+bwd, n=66)
  • derive shift_per_VG2 = model_knee(+0.2) − model_knee(−0.1)

PASS gate (per task):
  • per-VG2 |model_knee − data_knee| ≤ 0.1 V across all 4 VG2 values
  • full-33 median_dec ≤ 0.7

Outputs: results/track_hurkx_vg2coupled/{ablation.json, verdict.md, plot.png, run.log}
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

OUT = ROOT / "results/track_hurkx_vg2coupled"
OUT.mkdir(parents=True, exist_ok=True)

# Locked baseline (K1+ALPHA0 card + Hurkx winner)
K1_CARD     = 0.53825
ALPHA0_CARD = 7.83756e-4
HURKX_A     = 1.0e-6
HURKX_B     = 1.5

# VG2 coupling sweep
COEFF_GRID = [0.0, 0.5, 1.0, 2.0, 3.0]

# Knee slice
VG1_TARGET = 0.6
VG2_GRID   = [-0.1, 0.0, 0.1, 0.2]

# Gates (per task spec)
GATE_PER_VG2_V    = 0.1     # |model−data| ≤ 0.1V at EACH VG2
GATE_MEDIAN_DEC   = 0.7


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
            print(f"[hurkx-vg2] thermal timeout (T={t:.1f}°C)", flush=True)
            return
        print(f"[hurkx-vg2] T={t:.1f}°C > {threshold_c}°C — cooling to {target_c}°C", flush=True)
        time.sleep(5.0)
        if cpu_temp_c() < target_c:
            return


# ── Knee detector ────────────────────────────────────────────────
def knee_vd(Vd: np.ndarray, Id: np.ndarray) -> float:
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


def pick_curves(curves):
    out = []
    for vg2 in VG2_GRID:
        match = None
        for c in curves:
            if abs(c["VG1"] - VG1_TARGET) < 1e-6 and abs(c["VG2"] - vg2) < 1e-6:
                match = c; break
        if match is None:
            print(f"[hurkx-vg2] WARNING no curve for VG1=0.6 VG2={vg2:+.2f}", flush=True)
        out.append((vg2, match))
    return out


def install_combo_patch():
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


def simulate_curve(cfg, M1, M2, bjt, c, sebas_rows):
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)
    Vd_np = c["fwd_Vd"]
    row_sebas, _ = pillar.find_or_impute_row(sebas_rows, c["VG1"], c["VG2"])
    P_M1, P_M2 = pillar.make_overrides(row_sebas)
    Vd = torch.tensor(Vd_np, dtype=torch.float64)
    try:
        with pillar.patch_sd_scaled(sd_M1, P_M1), pillar.patch_sd_scaled(sd_M2, P_M2):
            out = pillar_forward(cfg, M1, M2, bjt, Vd, c["VG1"], c["VG2"])
        I_pred = np.abs(out["Id"].detach().cpu().numpy()).astype(np.float64)
        I_pred = np.where(np.isfinite(I_pred), I_pred, 0.0)
    except Exception as e:
        print(f"[hurkx-vg2] FAIL sim {c['f']}: {e}", flush=True)
        I_pred = np.zeros_like(Vd_np)
    return Vd_np, I_pred


def pillar_forward(cfg, M1, M2, bjt, Vd, VG1, VG2):
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    return forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd,
                      VG1=torch.tensor(VG1, dtype=torch.float64),
                      VG2=torch.tensor(VG2, dtype=torch.float64),
                      warm_start=True)


def run_one(coeff: float, sel_curves, sebas_rows, curves_all):
    tag = f"coeff={coeff:.2f}"
    print(f"[hurkx-vg2] === {tag} ===", flush=True)
    wait_cool()

    cfg, M1, M2, bjt = pillar.build_pyport_base()
    cfg.hurkx_bbt_A      = float(HURKX_A)
    cfg.hurkx_bbt_B      = float(HURKX_B)
    cfg.hurkx_vg2_coeff  = float(coeff)

    restore = install_combo_patch()
    per_vg2 = []
    median_dec = float("nan")
    try:
        for vg2, c in sel_curves:
            if c is None:
                per_vg2.append({"VG2": vg2, "data_knee_V": float("nan"),
                                "model_knee_V": float("nan")})
                continue
            Vd_np, I_pred = simulate_curve(cfg, M1, M2, bjt, c, sebas_rows)
            Id_data = np.abs(c["fwd_Id"])
            data_k  = knee_vd(c["fwd_Vd"], Id_data)
            model_k = knee_vd(Vd_np,        I_pred)
            B_eff = HURKX_B * (1.0 + coeff * vg2)
            print(f"[hurkx-vg2]   VG2={vg2:+.2f}  B_eff={B_eff:.3f}  data={data_k:.3f}V  model={model_k:.3f}V", flush=True)
            per_vg2.append({
                "VG2": float(vg2),
                "file": c["f"],
                "B_eff": float(B_eff),
                "data_knee_V":  float(data_k),
                "model_knee_V": float(model_k),
                "Vd":      Vd_np.tolist(),
                "Id_data": Id_data.tolist(),
                "Id_pred": I_pred.tolist(),
            })

        wait_cool()
        rows, nan_count = pillar.run_grid(cfg, M1, M2, bjt, curves_all, sebas_rows,
                                          label=tag, do_bwd=True)
        all_med = np.array([r["med_dec"] for r in rows if np.isfinite(r["med_dec"])])
        if all_med.size > 0:
            median_dec = float(np.median(all_med))
        print(f"[hurkx-vg2]   full-33 median_dec = {median_dec:.4f}  (n={all_med.size}, nan={nan_count})", flush=True)
    finally:
        restore()

    data_knees  = np.array([d["data_knee_V"]  for d in per_vg2], dtype=np.float64)
    model_knees = np.array([d["model_knee_V"] for d in per_vg2], dtype=np.float64)
    valid = np.isfinite(data_knees) & np.isfinite(model_knees)
    # per-VG2 gap (signed and abs)
    per_vg2_abs = np.abs(model_knees - data_knees)
    max_abs = float(np.nanmax(per_vg2_abs)) if valid.any() else float("nan")

    # data shift = data(+0.2) − data(−0.1); model shift likewise
    def find_knee(arr, vg2_target):
        for k, vg2 in enumerate(VG2_GRID):
            if abs(vg2 - vg2_target) < 1e-6:
                return arr[k]
        return float("nan")
    data_shift_p2_m1   = float(find_knee(data_knees,  0.2)  - find_knee(data_knees,  -0.1))
    model_shift_p2_m1  = float(find_knee(model_knees, 0.2)  - find_knee(model_knees, -0.1))

    knee_pass = bool(np.isfinite(max_abs) and max_abs <= GATE_PER_VG2_V)
    dec_pass  = bool(np.isfinite(median_dec) and median_dec <= GATE_MEDIAN_DEC)

    return {
        "coeff": float(coeff),
        "per_vg2": per_vg2,
        "data_shift_+0.2_minus_-0.1_V":  data_shift_p2_m1,
        "model_shift_+0.2_minus_-0.1_V": model_shift_p2_m1,
        "max_abs_per_vg2_gap_V":         max_abs,
        "median_dec_n66":                median_dec,
        "knee_pass":                     knee_pass,
        "dec_pass":                      dec_pass,
        "overall_pass":                  bool(knee_pass and dec_pass),
    }


def make_plot(results, sel_curves):
    n = len(COEFF_GRID)
    fig, axes = plt.subplots(1, n, figsize=(3.6*n, 3.4), sharex=True, sharey=True, squeeze=False)
    colors = plt.cm.viridis(np.linspace(0, 0.92, len(VG2_GRID)))
    for j, coeff in enumerate(COEFF_GRID):
        ax = axes[0][j]
        tag = f"coeff={coeff:.2f}"
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
            title = (f"coeff={coeff:.2f}\n"
                     f"shift_p2-m1={r['model_shift_+0.2_minus_-0.1_V']:+.3f}V\n"
                     f"max|gap|={r['max_abs_per_vg2_gap_V']:.3f}V dec={r['median_dec_n66']:.3f}")
        else:
            title = f"coeff={coeff:.2f}\n(missing)"
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Vd [V]")
        if j == 0:
            ax.set_ylabel("|Id| [A]")
        ax.grid(True, which="both", alpha=0.3)
    axes[0][0].legend(fontsize=6, loc="lower right")
    fig.suptitle("VG2-coupled Hurkx BBT — knees at VG1=0.6 (dashed=data, solid=model)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "plot.png", dpi=120)
    plt.close(fig)


def main():
    t0 = time.time()
    sebas_rows = pillar.load_sebas_params()
    curves = pillar.load_curves()
    print(f"[hurkx-vg2] loaded {len(curves)} curves, {len(sebas_rows)} sebas rows", flush=True)
    sel_curves = pick_curves(curves)
    n_ok = sum(1 for _, c in sel_curves if c is not None)
    print(f"[hurkx-vg2] selected {n_ok}/{len(VG2_GRID)} VG1=0.6 curves", flush=True)

    results = {}
    for coeff in COEFF_GRID:
        tag = f"coeff={coeff:.2f}"
        try:
            results[tag] = run_one(coeff, sel_curves, sebas_rows, curves)
        except Exception as e:
            traceback.print_exc()
            results[tag] = {"coeff": coeff, "error": str(e)}
        with open(OUT / "ablation.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    # objective: prioritize knee track + dec
    # data target shift ≈ +0.30 V (0.85 → 1.15)
    DATA_TARGET_SHIFT = 0.30
    best_tag, best_obj = None, float("inf")
    for tag, r in results.items():
        if "error" in r: continue
        s_model = r.get("model_shift_+0.2_minus_-0.1_V", float("nan"))
        mx = r.get("max_abs_per_vg2_gap_V", float("nan"))
        d  = r.get("median_dec_n66", float("nan"))
        if not (np.isfinite(s_model) and np.isfinite(mx) and np.isfinite(d)):
            continue
        obj = (s_model - DATA_TARGET_SHIFT) ** 2 + mx ** 2 + max(0.0, d - 0.665) ** 2
        r["obj"] = float(obj)
        if obj < best_obj:
            best_obj = obj; best_tag = tag

    with open(OUT / "ablation.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    try:
        make_plot(results, sel_curves)
    except Exception as e:
        print(f"[hurkx-vg2] plot failed: {e}", flush=True)

    # verdict.md
    lines = []
    lines.append("# VG2-coupled Hurkx BBT sweep\n")
    lines.append("**Hypothesis:** the Hurkx BBT field cut-off B should depend on VG2,")
    lines.append("because VG2 modulates the M2 drain-body depletion width. Implemented as")
    lines.append("`B_eff = B · (1 + hurkx_vg2_coeff · VG2)`. Positive coeff → higher VG2 →")
    lines.append("larger B_eff → BBT turns on at HIGHER Vd → knee shifts RIGHT.\n")
    lines.append(f"Locked baseline: K1@VG1=0.6 = {K1_CARD}, ALPHA0 = {ALPHA0_CARD:.4e},")
    lines.append(f"A = {HURKX_A:.0e}, B = {HURKX_B:.2f}.\n")
    lines.append(f"Data target shift (knee@VG2=+0.2) − (knee@VG2=−0.1) ≈ +{DATA_TARGET_SHIFT:.2f} V.\n")

    lines.append("## Grid results\n")
    header = "| coeff | model shift Δknee(+0.2 vs −0.1) | data Δknee | max |per-VG2 gap| | median_dec(n=66) | knee_PASS | dec_PASS | obj |"
    sep    = "|---:|---:|---:|---:|---:|:---:|:---:|---:|"
    lines.append(header); lines.append(sep)
    for coeff in COEFF_GRID:
        tag = f"coeff={coeff:.2f}"
        r = results.get(tag)
        if r is None or "error" in r:
            err = (r or {}).get("error", "missing")[:40]
            lines.append(f"| {coeff:.2f} | ERROR: {err} |")
            continue
        obj = r.get("obj", float("nan"))
        lines.append(
            f"| {coeff:.2f} | {r['model_shift_+0.2_minus_-0.1_V']:+.3f} | "
            f"{r['data_shift_+0.2_minus_-0.1_V']:+.3f} | "
            f"{r['max_abs_per_vg2_gap_V']:.3f} | {r['median_dec_n66']:.4f} | "
            f"{'YES' if r['knee_pass'] else 'NO'} | "
            f"{'YES' if r['dec_pass'] else 'NO'} | "
            f"{obj:.4f} |"
        )

    lines.append("\n## Per-VG2 knee table\n")
    lines.append("| coeff | VG2 | B_eff | data knee [V] | model knee [V] | gap [V] |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for coeff in COEFF_GRID:
        tag = f"coeff={coeff:.2f}"
        r = results.get(tag)
        if r is None or "error" in r:
            continue
        for d in r["per_vg2"]:
            beff = d.get("B_eff", float("nan"))
            dk = d["data_knee_V"]; mk = d["model_knee_V"]
            gap = (mk - dk) if (np.isfinite(mk) and np.isfinite(dk)) else float("nan")
            lines.append(f"| {coeff:.2f} | {d['VG2']:+.2f} | {beff:.3f} | {dk:.3f} | {mk:.3f} | {gap:+.3f} |")

    lines.append("\n## Verdict\n")
    if best_tag is None:
        lines.append("- All cells failed or produced NaN — no best candidate.")
        lines.append("- **OVERALL: FAIL** (no usable hurkx_vg2_coeff).")
    else:
        r = results[best_tag]
        overall = bool(r["overall_pass"])
        lines.append(f"- **Best coeff** by `(model_shift − data_shift)^2 + max_gap^2 + max(0,dec−0.665)^2`:")
        lines.append(f"  - hurkx_vg2_coeff = {r['coeff']:.2f}")
        lines.append(f"  - model Δknee(+0.2 vs −0.1) = {r['model_shift_+0.2_minus_-0.1_V']:+.3f} V")
        lines.append(f"  - data  Δknee(+0.2 vs −0.1) = {r['data_shift_+0.2_minus_-0.1_V']:+.3f} V")
        lines.append(f"  - max per-VG2 |gap| = {r['max_abs_per_vg2_gap_V']:.3f} V")
        lines.append(f"  - median_dec (n=66) = {r['median_dec_n66']:.4f}")
        lines.append(f"  - knee gate (max |gap| ≤ {GATE_PER_VG2_V} V): {'PASS' if r['knee_pass'] else 'FAIL'}")
        lines.append(f"  - dec gate (median_dec ≤ {GATE_MEDIAN_DEC}): {'PASS' if r['dec_pass'] else 'FAIL'}")
        if overall:
            lines.append("- **OVERALL: PASS** — VG2-coupled Hurkx BBT tracks data knee shift.")
        else:
            lines.append("- **OVERALL: FAIL** — best coeff misses at least one gate. Reporting null per NO-CHEAT.")
            lines.append("  Data may require a different parametrization (e.g. coupling to width W_depl directly,")
            lines.append("  or VG2 entering via prefactor A or power P, not exponent B alone).")

    lines.append("\n## Provenance")
    lines.append("- baseline: `pillar_I_C3_jts_tat.build_pyport_base()`")
    lines.append(f"- K1+ALPHA0 monkey patch (K1@VG1=0.6={K1_CARD}; ALPHA0={ALPHA0_CARD:.4e})")
    lines.append(f"- Hurkx locked: A={HURKX_A:.0e}, B={HURKX_B:.2f}")
    lines.append("- new knob: `cfg.hurkx_vg2_coeff` (default 0.0 preserves un-coupled behavior)")
    lines.append("- formula: `B_eff = B · (1 + hurkx_vg2_coeff · VG2)` (clamped ≥ 0.05 V)")
    lines.append(f"- runtime: {time.time() - t0:.1f}s")

    (OUT / "verdict.md").write_text("\n".join(lines) + "\n")
    print(f"[hurkx-vg2] wrote {OUT / 'verdict.md'}", flush=True)
    print(f"[hurkx-vg2] wrote {OUT / 'ablation.json'}", flush=True)
    print(f"[hurkx-vg2] wrote {OUT / 'plot.png'}", flush=True)
    print(f"[hurkx-vg2] total runtime: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
