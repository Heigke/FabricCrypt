"""z427 — S18 V_Sint runaway fix.

Tests four hypotheses for why V_Sint pumps to V_D and traps the parasitic
NPN in deep saturation (z426 root-cause diagnosis). For each, reports the
B4 bias point and a full cell-wide RMSE; then runs a combined-fix and
generates overlay + V_B/V_Sint trace plots.

Hypotheses:
  H1: Sint→GND shunt (cfg.m2_source_Rs = 1e6 Ω) — substrate-tap path
  H2: GIDL-to-Sint routing (cfg.gidl_route_to_sint=True) — BTBT pulls Sint
  H3: BJT saturation handling (control: bump be_oneway threshold lower)
  H4: Mario Ipos magnitude (control: scale Ipos PWL ×10)

Runs on ikaros CPU.
"""
from __future__ import annotations
import importlib.util as _ilu
import json
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
DATA = ROOT / "data/sebas_2026_04_22"
OUT = ROOT / "results/z427_vsint_fix"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

LOG = open(OUT / "run.log", "w")
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n"); LOG.flush()

# Reuse z91f loaders
_spec = _ilu.spec_from_file_location("z91f", ROOT / "scripts/z91f_validate_with_sebas_params.py")
z91f = _ilu.module_from_spec(_spec); _spec.loader.exec_module(z91f)

_spec_z425 = _ilu.spec_from_file_location("z425", ROOT / "scripts/z425_ideal_floating_body.py")
z425 = _ilu.module_from_spec(_spec_z425); _spec_z425.loader.exec_module(z425)
PWL = z425.PWL

from nsram.bsim4_port.model_card import BSIM4Model, parse_param_blocks  # noqa
from nsram.bsim4_port.nsram_cell_2T import (  # noqa
    NSRAMCell2TConfig, forward_2t, solve_2t_with_homotopy, _residuals,
)
from nsram.bsim4_port.temp import compute_size_dep  # noqa
from nsram.bsim4_port.geometry import Geometry  # noqa

load_curves = z91f.load_curves
load_sebas_params = z91f.load_sebas_params
find_params = z91f.find_params
patch_model_values = z91f.patch_model_values
patch_sd_scaled = z91f.patch_sd_scaled
make_overrides = z91f.make_overrides
make_bjt = z91f.make_bjt


def build_models():
    text_M1 = (DATA / "M1_130DNWFB.txt").read_text()
    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    shared = parse_param_blocks(text_M2)
    m_M1 = BSIM4Model.from_spice(text_M1, model_type="nmos", params=shared)
    patch_model_values(m_M1, type_n=True)
    m_M2 = BSIM4Model.from_spice(text_M2, model_type="nmos", params=shared)
    patch_model_values(m_M2, type_n=True)
    return m_M1, m_M2


# Baseline "ideal floating body" config (z425 ALL_FLAGS_ON)
BASE_FLAGS = dict(
    suppress_bulk_diode_forward=True,
    q1_be_oneway=True,
    use_mario_ipos=True,
    mario_ipos_param="VG1",
)


def make_cfg(model_M1, model_M2, extra: dict):
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                            newton_max_iters=50)
    for k, v in BASE_FLAGS.items():
        setattr(cfg, k, v)
    if cfg.use_mario_ipos:
        cfg.mario_ipos_pwl = PWL
    # mario scale (H4 control)
    if "mario_scale" in extra:
        cfg.mario_ipos_scale = float(extra.pop("mario_scale"))
    for k, v in extra.items():
        setattr(cfg, k, v)
    sd_M1 = compute_size_dep(model_M1, Geometry(L=cfg.Ln, W=cfg.Wn), T_C=cfg.T_C)
    sd_M2 = compute_size_dep(
        model_M2,
        Geometry(L=cfg.Ln * cfg.M2_length_factor, W=cfg.Wn),
        T_C=cfg.T_C)
    cfg._sd_M1 = sd_M1
    cfg._sd_M2 = sd_M2
    return cfg, sd_M1, sd_M2


def scalar(t):
    if torch.is_tensor(t):
        v = t.detach().squeeze()
        if v.numel() == 1:
            return float(v.item())
        return float(v[0].item())
    return float(t)


# ─── B4 single-bias diagnostic ───────────────────────────────────────────
def solve_b4(cfg, model_M1, model_M2, bjt, sd_M1, sd_M2, VG1=0.6, VG2=0.0, Vd_target=2.0):
    Vd_full = np.linspace(0.05, 2.0, 30)
    Vd_seq = torch.as_tensor(Vd_full, dtype=torch.float64)
    VG1_t = torch.tensor(VG1, dtype=torch.float64)
    VG2_t = torch.tensor(VG2, dtype=torch.float64)
    target_idx = int(np.argmin(np.abs(Vd_seq.numpy() - Vd_target)))
    Vsint_warm = torch.tensor(0.0, dtype=torch.float64)
    Vb_warm = torch.tensor(0.0, dtype=torch.float64)
    out_at = None; comp_at = None
    sebas_rows = load_sebas_params()
    row = find_params(sebas_rows, VG1, VG2)
    P_M1, P_M2 = make_overrides(row)
    with torch.no_grad(), patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
        for i in range(len(Vd_seq)):
            Vd_i = Vd_seq[i:i+1]
            try:
                out = solve_2t_with_homotopy(
                    cfg, model_M1, bjt,
                    Vd=Vd_i, VG1=VG1_t, VG2=VG2_t,
                    P_M1=None, P_M2=None,
                    Vsint_init=Vsint_warm.expand_as(Vd_i),
                    Vb_init=Vb_warm.expand_as(Vd_i),
                    verbose=False, model_M2=model_M2)
            except Exception as e:
                log(f"  solve fail @ Vd={float(Vd_i):.2f}: {e}")
                continue
            Vsint_warm = out["Vsint"].detach().squeeze(0)
            Vb_warm = out["Vb"].detach().squeeze(0)
            if i == target_idx:
                out_at = out
                _, _, comp_at = _residuals(
                    cfg, model_M1, bjt, Vd_i, VG1_t, VG2_t,
                    out["Vsint"], out["Vb"],
                    P_M1=None, P_M2=None, model_M2=model_M2)
    return out_at, comp_at


def measured_at(curves, VG1, VG2, Vd_target=2.0):
    for c in curves:
        if abs(c["VG1"] - VG1) < 1e-3 and abs(c["VG2"] - VG2) < 1e-3:
            Vd = c["Vd"].numpy(); Id = c["Id"].numpy()
            i = int(np.argmin(np.abs(Vd - Vd_target)))
            return float(Vd[i]), float(Id[i])
    return float("nan"), float("nan")


def b4_report(name, cfg, model_M1, model_M2, bjt, sd_M1, sd_M2, curves):
    out, comp = solve_b4(cfg, model_M1, model_M2, bjt, sd_M1, sd_M2)
    if out is None or comp is None:
        return {"name": name, "error": "solver failed at B4"}
    V_D = 2.0
    Vsint = scalar(out["Vsint"])
    Vb = scalar(out["Vb"])
    Id_pred = scalar(out["Id"])
    Ic_Q1 = scalar(comp["Ic_Q1"])
    Ids_M1 = scalar(comp["Ids_M1"])
    Vd_m, Id_m = measured_at(curves, 0.6, 0.0, 2.0)
    gap = math.log10(Id_m / abs(Id_pred)) if (Id_m > 0 and abs(Id_pred) > 0) else float("nan")
    rep = {
        "name": name,
        "V_D": V_D, "V_B": Vb, "V_Sint": Vsint,
        "V_BE": Vb - Vsint, "V_BC": Vb - V_D,
        "Id_predicted": Id_pred, "Id_measured": Id_m,
        "Ic_Q1": Ic_Q1, "Ids_M1": Ids_M1,
        "gap_dec": gap,
    }
    log(f"  {name}: V_Sint={Vsint:+.3f} V_B={Vb:+.3f} V_BE={rep['V_BE']:+.3f} "
        f"V_BC={rep['V_BC']:+.3f} Ic_Q1={Ic_Q1:+.3e} Id_pred={Id_pred:+.3e} "
        f"Id_meas={Id_m:.3e} gap={gap:+.2f} dec")
    return rep


# ─── Cell-wide RMSE (33-bias) ───────────────────────────────────────────
def cell_rmse(name, extra_flags: dict, model_M1, model_M2, curves, sebas_rows,
              collect_traces=False):
    cfg, sd_M1, sd_M2 = make_cfg(model_M1, model_M2, dict(extra_flags))
    log_eps = 1e-15
    per_bias = []
    vb_max_overall = -1e30
    fails = 0
    traces = {}
    t0 = time.time()
    for c in curves:
        sebas_row = find_params(sebas_rows, c["VG1"], c["VG2"])
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            continue
        P_M1, P_M2 = make_overrides(sebas_row)
        bjt = make_bjt(sebas_row)
        try:
            with torch.no_grad(), patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
                out = forward_2t(cfg, model_M1, bjt,
                                 c["Vd"], torch.tensor(c["VG1"]),
                                 torch.tensor(c["VG2"]),
                                 model_M1=model_M1, model_M2=model_M2,
                                 warm_start=True, use_homotopy=True)
            Id_pred = out["Id"].abs()
            conv = torch.tensor([bool(x) for x in out["converged"]])
            Vb_arr = out["Vb"]
            Vsint_arr = out["Vsint"]
        except Exception as e:
            fails += 1
            log(f"  {name} fail VG1={c['VG1']} VG2={c['VG2']}: {e}")
            continue
        if not conv.any():
            fails += 1
            continue
        log_p = torch.log10(Id_pred + log_eps)
        log_m = torch.log10(c["Id"] + log_eps)
        sq = (log_p - log_m) ** 2
        rmse = float(torch.sqrt(sq[conv].mean()))
        vb_max = float(Vb_arr.max())
        vb_max_overall = max(vb_max_overall, vb_max)
        rec = {"VG1": c["VG1"], "VG2": c["VG2"],
               "log_rmse": rmse, "vb_max": vb_max}
        if collect_traces:
            rec.update({
                "Vd": c["Vd"].numpy().tolist(),
                "Id_meas": c["Id"].numpy().tolist(),
                "Id_pred": Id_pred.numpy().tolist(),
                "Vb": Vb_arr.numpy().tolist(),
                "Vsint": Vsint_arr.numpy().tolist(),
                "converged": conv.numpy().tolist(),
            })
        per_bias.append(rec)
    cell_sq = sum(r["log_rmse"]**2 for r in per_bias)
    cell_n = len(per_bias)
    cell = math.sqrt(cell_sq / cell_n) if cell_n else float("inf")
    per_branch = {}
    for r in per_bias:
        b = f"VG1_{r['VG1']:.1f}"
        per_branch.setdefault(b, {"sq": 0.0, "n": 0})
        per_branch[b]["sq"] += r["log_rmse"]**2
        per_branch[b]["n"] += 1
    per_branch_rmse = {b: math.sqrt(v["sq"]/v["n"]) for b, v in per_branch.items()}
    log(f"  {name}: cell={cell:.3f} per_branch={ {k:round(v,3) for k,v in per_branch_rmse.items()} } "
        f"Vb_max={vb_max_overall:.3f} fails={fails} wall={time.time()-t0:.0f}s")
    return {
        "name": name, "cell_rmse_dec": cell,
        "per_branch_rmse_dec": per_branch_rmse,
        "n_biases_evaluated": cell_n, "vb_max_overall": vb_max_overall,
        "fails": fails, "wall_sec": round(time.time()-t0, 1),
        "per_bias": per_bias if collect_traces else None,
    }


# ─── Main ───────────────────────────────────────────────────────────────
def main():
    t_main = time.time()
    log("z427 starting — V_Sint runaway fix")
    model_M1, model_M2 = build_models()
    curves = load_curves()
    sebas_rows = load_sebas_params()
    log(f"loaded: {len(curves)} curves, {len(sebas_rows)} sebas rows")

    # B4 diagnostic for each hypothesis (single bias, fast)
    sebas_row_b4 = find_params(sebas_rows, 0.6, 0.0)
    bjt_b4 = make_bjt(sebas_row_b4)

    hypotheses = {
        "H0_baseline": {},
        "H1_sint_shunt_1M":  {"m2_source_Rs": 1.0e6},
        "H2_gidl_to_sint":   {"gidl_route_to_sint": True},
        "H3_bjt_softer":     {},  # adjust BJT directly below
        "H4_mario_x10":      {"mario_ipos_scale": 10.0},
        "COMBINED_H1_H2":    {"m2_source_Rs": 1.0e6, "gidl_route_to_sint": True},
    }

    # B4 diagnostics (skip if cached — first run took ~5min)
    b4_cache = OUT / "b4_diagnostics.json"
    if b4_cache.exists():
        log("=== B4 diagnostics: loading cached ===")
        b4 = json.loads(b4_cache.read_text())
    else:
        log("=== B4 diagnostics (VG1=0.6, VG2=0.0, V_D=2.0) ===")
        b4 = {}
    for name, extra in hypotheses.items():
        if name in b4 and "gap_dec" in b4[name]:
            continue
        cfg, sd_M1, sd_M2 = make_cfg(model_M1, model_M2, dict(extra))
        bjt_local = bjt_b4
        if name == "H3_bjt_softer":
            # bump vbe_thresh lower so BJT turns on earlier; pyport q1_be_oneway
            # uses cfg.q1_vbe_thresh if set, else default 0.35 in bjt module.
            # Quick test: drop to 0.20 and increase Bf (override Sebas card to
            # force stronger forward gain). This is *not* the recommended fix;
            # H3 is a control to confirm the BJT itself is NOT the cause.
            import copy
            bjt_local = copy.deepcopy(bjt_b4)
            bjt_local.Bf = bjt_local.Bf * 10  # 10000 → 100000
            cfg.q1_be_thresh = 0.20
        rep = b4_report(name, cfg, model_M1, model_M2, bjt_local, sd_M1, sd_M2, curves)
        b4[name] = rep

    (OUT / "b4_diagnostics.json").write_text(json.dumps(b4, indent=2))

    # Per-hypothesis JSON
    for name, rep in b4.items():
        if name.startswith(("H1", "H2", "H3", "H4")):
            stub = name.split("_", 1)[0].lower()
            (OUT / f"{stub}_result.json").write_text(json.dumps(rep, indent=2))

    # Decide which hypothesis to include in combined run
    baseline_gap = b4["H0_baseline"].get("gap_dec", float("nan"))
    log(f"baseline B4 gap = {baseline_gap:+.2f} dec")
    helpful = {}
    for h in ("H1_sint_shunt_1M", "H2_gidl_to_sint", "H3_bjt_softer", "H4_mario_x10"):
        gap = b4[h].get("gap_dec", float("nan"))
        delta = baseline_gap - gap if (not math.isnan(gap) and not math.isnan(baseline_gap)) else 0
        helpful[h] = {"gap": gap, "delta_dec_vs_baseline": delta,
                      "physically_meaningful": delta >= 0.5}
        log(f"  {h}: gap={gap:+.2f} dec  delta={delta:+.2f} dec  "
            f"{'PASS' if delta >= 0.5 else 'fail'}")

    # Cell-wide RMSE: BASELINE + COMBINED_H1_H2 + COMBINED_H1_H4. Skip per-
    # hypothesis cell runs (B4 already discriminates and each costs >5min).
    log("=== Cell-wide RMSE (33-bias) — BASELINE + COMBINED variants ===")
    cell_results = {}
    cell_results["BASELINE"] = cell_rmse("BASELINE", {}, model_M1, model_M2,
                                          curves, sebas_rows, collect_traces=True)
    cell_results["COMBINED_H1_H2"] = cell_rmse(
        "COMBINED_H1_H2",
        {"m2_source_Rs": 1.0e6, "gidl_route_to_sint": True},
        model_M1, model_M2, curves, sebas_rows, collect_traces=True)
    cell_results["COMBINED_H1_H4"] = cell_rmse(
        "COMBINED_H1_H4",
        {"m2_source_Rs": 1.0e6, "mario_ipos_scale": 10.0},
        model_M1, model_M2, curves, sebas_rows, collect_traces=False)

    combined_result = cell_results.get("COMBINED_H1_H2", {})
    combined_cell = combined_result.get("cell_rmse_dec", float("inf"))

    # ─── Plots ───────────────────────────────────────────────────────
    log("=== Plotting ===")
    # 1) Overlay at VG1=0.6 (measured vs combined)
    per_bias = combined_result.get("per_bias", []) or []
    sel = [r for r in per_bias if abs(r["VG1"] - 0.6) < 1e-3]
    sel.sort(key=lambda r: r["VG2"])
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap("plasma")
    for i, r in enumerate(sel):
        col = cmap(i / max(1, len(sel)-1))
        ax.semilogy(r["Vd"], r["Id_meas"], "o", color=col, ms=4,
                    label=f"meas VG2={r['VG2']:+.2f}")
        ax.semilogy(r["Vd"], np.abs(r["Id_pred"]), "-", color=col, alpha=0.8)
    ax.set_xlabel("V_D [V]"); ax.set_ylabel("|I_D| [A]")
    ax.set_title(f"z427 COMBINED (H1+H2) — VG1=0.6  cell={combined_cell:.2f} dec")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "overlay_VG1_0p6.png", dpi=130)
    plt.close(fig)

    # 2) V_B and V_Sint traces vs V_D at VG1=0.6, VG2=0.0
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    target = [r for r in sel if abs(r["VG2"] - 0.0) < 1e-3]
    if target:
        r = target[0]
        axes[0].plot(r["Vd"], r["Vb"], "-o", label="V_B (combined)", color="tab:red")
        axes[0].plot(r["Vd"], r["Vsint"], "-s", label="V_Sint (combined)", color="tab:blue")
        axes[0].plot(r["Vd"], r["Vd"], "k:", alpha=0.4, label="V_D (ref)")
        axes[0].set_xlabel("V_D [V]"); axes[0].set_ylabel("Node voltage [V]")
        axes[0].set_title("Node voltages — combined fix")
        axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    # Compare to baseline
    base_pb = cell_results["BASELINE"].get("per_bias")  # None unless collect_traces
    # Quick second run with baseline collect just for plot
    if base_pb is None:
        log("re-running BASELINE with traces for plot")
        base_full = cell_rmse("BASELINE_TRACE", {}, model_M1, model_M2,
                              curves, sebas_rows, collect_traces=True)
        base_pb = base_full["per_bias"]
    base_sel = [r for r in base_pb if abs(r["VG1"] - 0.6) < 1e-3
                and abs(r["VG2"] - 0.0) < 1e-3]
    if base_sel:
        r = base_sel[0]
        axes[1].plot(r["Vd"], r["Vb"], "-o", label="V_B (baseline)", color="tab:red", alpha=0.7)
        axes[1].plot(r["Vd"], r["Vsint"], "-s", label="V_Sint (baseline)", color="tab:blue", alpha=0.7)
        axes[1].plot(r["Vd"], r["Vd"], "k:", alpha=0.4, label="V_D (ref)")
        axes[1].set_xlabel("V_D [V]"); axes[1].set_ylabel("Node voltage [V]")
        axes[1].set_title("Node voltages — baseline (V_Sint runaway)")
        axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "vb_vsint_trace_VG1_0p6.png", dpi=130)
    plt.close(fig)

    # ─── Combined result + gates ─────────────────────────────────────
    summary_cell = {k: {kk: vv for kk, vv in v.items() if kk != "per_bias"}
                     for k, v in cell_results.items()}
    (OUT / "combined_result.json").write_text(json.dumps({
        "summary": summary_cell,
        "combined_cell_rmse_dec": combined_cell,
        "baseline_cell_rmse_dec": cell_results["BASELINE"]["cell_rmse_dec"],
        "improvement_dec": cell_results["BASELINE"]["cell_rmse_dec"] - combined_cell,
        "b4_helpful": helpful,
    }, indent=2))

    gates = {
        "INFRA": all(h in b4 for h in ("H1_sint_shunt_1M", "H2_gidl_to_sint",
                                        "H3_bjt_softer", "H4_mario_x10")),
        "DISCOVERY": any(v["physically_meaningful"] for v in helpful.values()),
        "AMBITIOUS": combined_cell < 2.0,
        "KILL_SHOT": (combined_cell > 3.5
                      and not any(v["physically_meaningful"] for v in helpful.values())),
    }

    # ─── Honest analysis ─────────────────────────────────────────────
    he_md = ["# z427 — Honest Analysis", ""]
    he_md.append(f"Baseline cell RMSE: **{cell_results['BASELINE']['cell_rmse_dec']:.3f} dec**")
    he_md.append(f"Combined (H1+H2) cell RMSE: **{combined_cell:.3f} dec**")
    he_md.append(f"Improvement: **{cell_results['BASELINE']['cell_rmse_dec'] - combined_cell:+.3f} dec**")
    he_md.append("")
    he_md.append("## B4 diagnostic (VG1=0.6, VG2=0.0, V_D=2.0)")
    he_md.append("")
    he_md.append("| Hypothesis | V_Sint | V_B | V_BE | V_BC | Ic_Q1 | Id_pred | gap_dec | Δ vs baseline |")
    he_md.append("|---|---|---|---|---|---|---|---|---|")
    for name, r in b4.items():
        if "gap_dec" not in r:
            continue
        delta = baseline_gap - r["gap_dec"] if name != "H0_baseline" else 0.0
        he_md.append(f"| {name} | {r['V_Sint']:+.3f} | {r['V_B']:+.3f} | "
                     f"{r['V_BE']:+.3f} | {r['V_BC']:+.3f} | {r['Ic_Q1']:+.2e} | "
                     f"{r['Id_predicted']:+.2e} | {r['gap_dec']:+.2f} | {delta:+.2f} |")
    he_md.append("")
    he_md.append("## Per-hypothesis verdict")
    for h, v in helpful.items():
        verdict = "PASS (helped ≥0.5 dec)" if v["physically_meaningful"] else "did not help"
        he_md.append(f"- **{h}**: Δ={v['delta_dec_vs_baseline']:+.2f} dec — {verdict}")
    he_md.append("")
    he_md.append("## Gates")
    for g, p in gates.items():
        he_md.append(f"- {g}: **{'PASS' if p else 'fail'}**")
    he_md.append("")
    he_md.append("## Cell-wide per branch (combined)")
    for b, v in combined_result.get("per_branch_rmse_dec", {}).items():
        he_md.append(f"- {b}: {v:.3f}")
    (OUT / "honest_analysis.md").write_text("\n".join(he_md))

    log(f"\n=== GATES ===")
    for g, p in gates.items():
        log(f"  {g}: {'PASS' if p else 'fail'}")
    log(f"Wrote {OUT}/")
    log(f"Total wall: {time.time()-t_main:.0f}s")


if __name__ == "__main__":
    main()
