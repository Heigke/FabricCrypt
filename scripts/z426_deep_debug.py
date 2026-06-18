"""z426 — S17 deep current decomposition debug.

At 5 critical bias points dump every intermediate current + node voltage from
the pyport NS-RAM 2T cell, compare to measured I_D, and identify *which* term
is wrong by orders of magnitude.

Outputs (results/z426_deep_debug/):
  summary.json
  current_breakdown.md
  waterfall_VG1_0p6.png
  diagnosis.md

Runs once, no polling, ~30s on ikaros CPU.
"""
from __future__ import annotations
import importlib.util as _ilu
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
RESULTS = ROOT / "results" / "z426_deep_debug"
RESULTS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

# Load z91f / z425 helpers
_spec = _ilu.spec_from_file_location(
    "z91f", ROOT / "scripts/z91f_validate_with_sebas_params.py"
)
z91f = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(z91f)
load_curves = z91f.load_curves
load_sebas_params = z91f.load_sebas_params
find_params = z91f.find_params
patch_model_values = z91f.patch_model_values
patch_sd_scaled = z91f.patch_sd_scaled
make_overrides = z91f.make_overrides
make_bjt = z91f.make_bjt

_spec_z425 = _ilu.spec_from_file_location(
    "z425", ROOT / "scripts/z425_ideal_floating_body.py"
)
z425 = _ilu.module_from_spec(_spec_z425)
_spec_z425.loader.exec_module(z425)
PWL = z425.PWL

from nsram.bsim4_port.model_card import BSIM4Model, parse_param_blocks  # noqa: E402
from nsram.bsim4_port.nsram_cell_2T import (  # noqa: E402
    NSRAMCell2TConfig,
    solve_2t_with_homotopy,
    _residuals,
)
from nsram.bsim4_port.geometry import Geometry  # noqa: E402
from nsram.bsim4_port.temp import compute_size_dep  # noqa: E402

DATA = ROOT / "data/sebas_2026_04_22"


def build_models():
    text_M1 = (DATA / "M1_130DNWFB.txt").read_text()
    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    shared = parse_param_blocks(text_M2)
    m_M1 = BSIM4Model.from_spice(text_M1, model_type="nmos", params=shared)
    patch_model_values(m_M1, type_n=True)
    m_M2 = BSIM4Model.from_spice(text_M2, model_type="nmos", params=shared)
    patch_model_values(m_M2, type_n=True)
    return m_M1, m_M2


def make_cfg(model_M1, model_M2):
    cfg = NSRAMCell2TConfig(
        use_iii=True, use_gidl=True, use_bjt=True, newton_max_iters=50
    )
    cfg.suppress_bulk_diode_forward = True
    cfg.q1_be_oneway = True
    cfg.use_mario_ipos = True
    cfg.mario_ipos_param = "VG1"
    cfg.mario_ipos_pwl = PWL
    sd_M1 = compute_size_dep(model_M1, Geometry(L=cfg.Ln, W=cfg.Wn), T_C=cfg.T_C)
    sd_M2 = compute_size_dep(
        model_M2,
        Geometry(L=cfg.Ln * cfg.M2_length_factor, W=cfg.Wn),
        T_C=cfg.T_C,
    )
    cfg._sd_M1 = sd_M1
    cfg._sd_M2 = sd_M2
    return cfg, sd_M1, sd_M2


def measured_at(curves, VG1, VG2, Vd_target, atol=0.03):
    """Find measured I_D at (VG1,VG2,Vd_target). Returns NaN if not found."""
    for c in curves:
        if abs(c["VG1"] - VG1) < 1e-3 and abs(c["VG2"] - VG2) < 1e-3:
            Vd = c["Vd"].numpy()
            Id = c["Id"].numpy()
            i = int(np.argmin(np.abs(Vd - Vd_target)))
            if abs(Vd[i] - Vd_target) < atol + 0.05:
                return float(Vd[i]), float(Id[i])
    return float("nan"), float("nan")


def solve_one(cfg, model_M1, model_M2, bjt, Vd_val, VG1_val, VG2_val,
              P_M1, P_M2, sd_M1, sd_M2, Vd_seq_full):
    """Sweep full Vd vector to get warm-started solution, return per-point dict
    at the target Vd along with full components from _residuals."""
    Vd_seq = torch.as_tensor(Vd_seq_full, dtype=torch.float64)
    VG1 = torch.tensor(VG1_val, dtype=torch.float64)
    VG2 = torch.tensor(VG2_val, dtype=torch.float64)

    # Walk Vd_seq one-by-one with warm-starting (mirrors forward_2t low-level path)
    Vsint_warm = torch.tensor(0.0, dtype=torch.float64)
    Vb_warm = torch.tensor(0.0, dtype=torch.float64)
    target_idx = int(np.argmin(np.abs(Vd_seq.numpy() - Vd_val)))
    out_at_target = None

    # z425 applies per-bias overrides via patch_sd_scaled (modifies sd.scaled dict),
    # NOT via solve_2t's P_M1/P_M2 kwargs (which would call _override_sd with
    # attribute-name keys like 'etab' that don't exist on SizeDependParam).
    with torch.no_grad(), patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
        for i in range(len(Vd_seq)):
            Vd_i = Vd_seq[i:i + 1]
            out = solve_2t_with_homotopy(
                cfg, model_M1, bjt,
                Vd=Vd_i, VG1=VG1, VG2=VG2,
                P_M1=None, P_M2=None,
                Vsint_init=Vsint_warm.expand_as(Vd_i),
                Vb_init=Vb_warm.expand_as(Vd_i),
                verbose=False,
                model_M2=model_M2,
            )
            Vsint_warm = out["Vsint"].detach().squeeze(0)
            Vb_warm = out["Vb"].detach().squeeze(0)
            if i == target_idx:
                out_at_target = out
                R_S, R_B, comp = _residuals(
                    cfg, model_M1, bjt, Vd_i,
                    VG1, VG2,
                    out["Vsint"], out["Vb"],
                    P_M1=None, P_M2=None,
                    model_M2=model_M2,
                )
                return out, comp, R_S, R_B
    return out_at_target, None, None, None


def scalarize(t):
    if torch.is_tensor(t):
        return float(t.detach().squeeze().item()) if t.numel() == 1 else float(t.detach().squeeze()[0].item())
    return float(t)


def main():
    print("[z426] loading models + curves ...")
    model_M1, model_M2 = build_models()
    cfg, sd_M1, sd_M2 = make_cfg(model_M1, model_M2)
    curves = load_curves()
    sebas_rows = load_sebas_params()
    print(f"       {len(curves)} curves, {len(sebas_rows)} sebas rows")

    Vd_full = np.linspace(0.05, 2.0, 30)

    bias_points = [
        # (label, VG1, VG2, Vd_target)
        ("B1_vg1_0p6_vg2_0p0_vd_0p5", 0.6, 0.0, 0.5),
        ("B2_vg1_0p6_vg2_0p0_vd_1p0", 0.6, 0.0, 1.0),
        ("B3_vg1_0p6_vg2_0p0_vd_1p5", 0.6, 0.0, 1.5),
        ("B4_vg1_0p6_vg2_0p0_vd_2p0", 0.6, 0.0, 2.0),
        ("B5_vg1_0p2_vg2_0p1_vd_2p0", 0.2, 0.1, 2.0),
    ]

    # For waterfall: full Vd sweep at VG1=0.6, VG2=0.0
    waterfall_rows = []  # list of per-Vd component dict

    summary = {}
    for label, VG1, VG2, Vd_target in bias_points:
        print(f"\n[z426] bias {label}: VG1={VG1} VG2={VG2} Vd={Vd_target}")
        row = find_params(sebas_rows, VG1, VG2)
        if row is None or math.isnan(row.get("K1", float("nan"))):
            print(f"       no Sebas row for VG1={VG1} VG2={VG2}; skipping")
            summary[label] = {"error": "no sebas row"}
            continue
        P_M1, P_M2 = make_overrides(row)
        bjt = make_bjt(row)
        Vd_meas, Id_meas = measured_at(curves, VG1, VG2, Vd_target)
        print(f"       measured Vd={Vd_meas:.3f}  I_D={Id_meas:.3e}")

        try:
            out, comp, R_S, R_B = solve_one(
                cfg, model_M1, model_M2, bjt,
                Vd_target, VG1, VG2, P_M1, P_M2,
                sd_M1, sd_M2, Vd_full,
            )
        except Exception as e:
            print(f"       SOLVE ERROR: {e}")
            summary[label] = {"error": str(e)}
            continue

        Id_pred = scalarize(out["Id"])
        Vsint = scalarize(out["Vsint"])
        Vb = scalarize(out["Vb"])

        def g(name, default=0.0):
            if comp is None or name not in comp:
                return default
            v = comp[name]
            return scalarize(v) if v is not None else default

        # Compute Mario Ipos from the PWL coefficients exactly as the cell does
        # (R_B side dump for diagnostic)
        try:
            xs, ys = PWL["a"]; a_v = float(np.interp(VG1, xs, ys))
            xs, ys = PWL["b"]; b_v = float(np.interp(VG1, xs, ys))
            xs, ys = PWL["d"]; d_v = float(np.interp(VG1, xs, ys))
            xs, ys = PWL["e"]; e_v = float(np.interp(VG1, xs, ys))
            xs, ys = PWL["f"]; f_v = float(np.interp(VG1, xs, ys))
            C_c = -2.4
            I_exp = a_v * math.exp(max(-60.0, min(40.0, (Vd_target + C_c) * b_v)))
            arg_pow = Vd_target + f_v
            I_pow = d_v * (arg_pow ** e_v) if arg_pow > 0 else 0.0
            I_pos_body = max(0.0, min(1.0e-2, I_exp + I_pow))
        except Exception:
            I_exp = I_pow = I_pos_body = float("nan")

        row_out = {
            "label": label,
            "VG1": VG1, "VG2": VG2, "Vd": Vd_target,
            "V_D": Vd_target,
            "V_G1": VG1, "V_G2": VG2,
            "V_Source": 0.0,
            "V_Sint": Vsint,
            "V_B": Vb,
            "Id_measured": Id_meas,
            "Id_predicted_total": Id_pred,
            "R_Sint": scalarize(R_S) if R_S is not None else float("nan"),
            "R_B": scalarize(R_B) if R_B is not None else float("nan"),
            # M1 components
            "Ids_M1": g("Ids_M1"),
            "Iii_M1": g("Iii_M1"),
            "Igidl_M1": g("Igidl_M1"),
            "Igisl_M1": g("Igisl_M1"),
            "Ibs_M1": g("Ibs_M1"),
            "Ibd_M1": g("Ibd_M1"),
            "Igb_M1": g("Igb_M1"),
            # M2 components
            "Ids_M2": g("Ids_M2"),
            "Iii_M2": g("Iii_M2"),
            "Igidl_M2": g("Igidl_M2"),
            "Ibs_M2": g("Ibs_M2"),
            "Ibd_M2": g("Ibd_M2"),
            # BJT Q1 (vertical parasitic at body)
            "Ic_Q1": g("Ic_Q1"),
            "Ib_Q1": g("Ib_Q1"),
            "Ie_Q1": g("Ie_Q1"),
            # BJT Q2 (lateral)
            "Ic_Q2": g("Ic_Q2"),
            "Ib_Q2": g("Ib_Q2"),
            "Ie_Q2": g("Ie_Q2"),
            "Ic_lat": g("Ic_lat"),
            "Ic_avalanche": g("Ic_avalanche"),
            # vertical NPN to DNW
            "Ic_vert": g("Ic_vert"),
            "Ib_vert": g("Ib_vert"),
            "Ie_vert": g("Ie_vert"),
            # body diodes / well
            "I_well_body": g("I_well_body"),
            "I_body_pdiode": g("I_body_pdiode"),
            "I_tat": g("I_tat"),
            "I_subdiode": g("I_subdiode"),
            # snapback subcircuit
            "I_snap_d": g("I_snap_d"),
            "I_snap_b": g("I_snap_b"),
            # body leak / Mario
            "I_leak_body": g("I_leak_body"),
            "Mario_Iexp": I_exp,
            "Mario_Ipow": I_pow,
            "Mario_Ipos_body": I_pos_body,
        }

        # Re-derive Id from constituent sum the way nsram_cell_2T does it
        Id_recon = (
            row_out["Ids_M1"] + row_out["Ic_Q1"] + row_out["Ic_Q2"]
            + row_out["Ic_lat"] + row_out["Ic_avalanche"]
            + row_out["Igidl_M1"] - row_out["Ibd_M1"]
            - row_out["Ie_vert"] + row_out["I_snap_d"]
        )
        row_out["Id_reconstructed"] = Id_recon

        # Gap in decades
        if Id_meas > 0 and abs(Id_pred) > 0:
            row_out["gap_dec"] = math.log10(Id_meas / abs(Id_pred))
        else:
            row_out["gap_dec"] = float("nan")

        summary[label] = row_out
        print(f"       Vb={Vb:+.3f}  Vsint={Vsint:+.3f}  "
              f"Ids_M1={row_out['Ids_M1']:+.3e}  Ic_Q1={row_out['Ic_Q1']:+.3e}  "
              f"Ipos={I_pos_body:+.3e}  Id_pred={Id_pred:+.3e}  meas={Id_meas:.3e}  "
              f"gap={row_out['gap_dec']:+.2f}dec")

    # ---- Waterfall: full Vd sweep at VG1=0.6, VG2=0.0 ----
    print("\n[z426] waterfall sweep VG1=0.6, VG2=0.0 ...")
    row = find_params(sebas_rows, 0.6, 0.0)
    if row is not None and not math.isnan(row.get("K1", float("nan"))):
        P_M1, P_M2 = make_overrides(row)
        bjt = make_bjt(row)
        Vd_seq = torch.as_tensor(Vd_full, dtype=torch.float64)
        VG1_t = torch.tensor(0.6, dtype=torch.float64)
        VG2_t = torch.tensor(0.0, dtype=torch.float64)
        Vsint_warm = torch.tensor(0.0, dtype=torch.float64)
        Vb_warm = torch.tensor(0.0, dtype=torch.float64)
        with torch.no_grad(), patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
            for i in range(len(Vd_seq)):
                Vd_i = Vd_seq[i:i + 1]
                out = solve_2t_with_homotopy(
                    cfg, model_M1, bjt,
                    Vd=Vd_i, VG1=VG1_t, VG2=VG2_t,
                    P_M1=None, P_M2=None,
                    Vsint_init=Vsint_warm.expand_as(Vd_i),
                    Vb_init=Vb_warm.expand_as(Vd_i),
                    verbose=False,
                    model_M2=model_M2,
                )
                Vsint_warm = out["Vsint"].detach().squeeze(0)
                Vb_warm = out["Vb"].detach().squeeze(0)
                _, _, comp = _residuals(
                    cfg, model_M1, bjt, Vd_i, VG1_t, VG2_t,
                    out["Vsint"], out["Vb"],
                    P_M1=None, P_M2=None, model_M2=model_M2,
                )
                waterfall_rows.append({
                    "Vd": float(Vd_seq[i].item()),
                    "Vb": scalarize(out["Vb"]),
                    "Ids_M1": scalarize(comp["Ids_M1"]),
                    "Ic_Q1": scalarize(comp["Ic_Q1"]),
                    "Iii_M1": scalarize(comp["Iii_M1"]),
                    "Igidl_M1": scalarize(comp["Igidl_M1"]),
                    "Ibd_M1": scalarize(comp["Ibd_M1"]),
                    "Ic_lat": scalarize(comp["Ic_lat"]),
                    "Ic_avalanche": scalarize(comp["Ic_avalanche"]),
                    "I_snap_d": scalarize(comp.get("I_snap_d", torch.zeros(1))),
                    "Id_pred": scalarize(out["Id"]),
                })

    # ---- Write summary.json ----
    out_obj = {
        "bias_points": summary,
        "waterfall_VG1_0p6_VG2_0p0": waterfall_rows,
    }
    with open(RESULTS / "summary.json", "w") as f:
        json.dump(out_obj, f, indent=2, default=str)
    print(f"\n[z426] wrote {RESULTS/'summary.json'}")

    # ---- Markdown breakdown ----
    write_markdown(summary)

    # ---- Waterfall plot ----
    try:
        write_waterfall_plot(waterfall_rows, curves)
    except Exception as e:
        print(f"[z426] plot failed: {e}")

    # ---- Diagnosis ----
    write_diagnosis(summary, waterfall_rows)
    print(f"[z426] DONE — artifacts in {RESULTS}")


def fmt(x):
    if isinstance(x, float):
        if math.isnan(x):
            return "NaN"
        if x == 0:
            return "0"
        return f"{x:+.2e}"
    return str(x)


def mark(val, abs_val_meas):
    """Tag a current as DOMINANT / NEGLIGIBLE / BUG?."""
    a = abs(val)
    if math.isnan(a) or abs_val_meas == 0 or math.isnan(abs_val_meas):
        return ""
    ratio = a / abs_val_meas
    if ratio > 0.5:
        return "DOMINANT"
    if ratio > 0.05:
        return "MAJOR"
    if ratio < 1e-3:
        return "negligible"
    return ""


def write_markdown(summary):
    keys = [
        "V_D", "V_G1", "V_G2", "V_B", "V_Sint",
        "Id_measured", "Id_predicted_total", "Id_reconstructed", "gap_dec",
        "Ids_M1", "Iii_M1", "Igidl_M1", "Ibd_M1", "Ibs_M1", "Igb_M1",
        "Ic_Q1", "Ib_Q1", "Ie_Q1",
        "Ic_Q2", "Ic_lat", "Ic_avalanche",
        "Ic_vert", "Ib_vert", "Ie_vert",
        "Mario_Iexp", "Mario_Ipow", "Mario_Ipos_body",
        "I_well_body", "I_body_pdiode", "I_tat", "I_subdiode",
        "Ids_M2", "Iii_M2", "Igidl_M2", "Ibd_M2",
        "I_snap_d", "I_snap_b", "I_leak_body",
        "R_Sint", "R_B",
    ]
    lines = ["# z426 — Current Breakdown\n",
             "All currents in A, voltages in V. Sign: positive = INTO that node/terminal.\n",
             "`DOMINANT`/`MAJOR`/`negligible` tags computed vs |Id_measured|.\n"]
    labels = list(summary.keys())
    header = "| field |" + "|".join(f" {lbl} " for lbl in labels) + "|"
    sep = "|---|" + "|".join("---" for _ in labels) + "|"
    lines.append(header)
    lines.append(sep)
    for k in keys:
        cells = []
        for lbl in labels:
            r = summary[lbl]
            if "error" in r:
                cells.append("ERR")
                continue
            v = r.get(k, float("nan"))
            tag = ""
            if k not in ("V_D", "V_G1", "V_G2", "V_B", "V_Sint",
                         "gap_dec", "Id_measured", "Id_predicted_total",
                         "Id_reconstructed", "R_Sint", "R_B"):
                tag = mark(v if isinstance(v, float) else 0.0,
                           abs(r.get("Id_measured", 0.0)))
            cells.append(f"{fmt(v)} {tag}".strip())
        lines.append(f"| `{k}` | " + " | ".join(cells) + " |")
    (RESULTS / "current_breakdown.md").write_text("\n".join(lines) + "\n")
    print(f"[z426] wrote {RESULTS/'current_breakdown.md'}")


def write_waterfall_plot(rows, curves):
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Vd = [r["Vd"] for r in rows]
    fields = ["Ids_M1", "Ic_Q1", "Iii_M1", "Igidl_M1", "Ic_lat",
              "Ic_avalanche", "I_snap_d", "Id_pred"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for fld in fields:
        y = np.abs([r[fld] for r in rows]) + 1e-30
        ax.semilogy(Vd, y, marker=".", label=fld)
    # Overlay measured
    for c in curves:
        if abs(c["VG1"] - 0.6) < 1e-3 and abs(c["VG2"] - 0.0) < 1e-3:
            ax.semilogy(c["Vd"].numpy(), c["Id"].numpy(), "k--", lw=2,
                        label="MEASURED")
            break
    ax.set_xlabel("V_D [V]")
    ax.set_ylabel("|I| [A]")
    ax.set_title("z426 waterfall: VG1=0.6, VG2=0.0 — all current components")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "waterfall_VG1_0p6.png", dpi=120)
    plt.close(fig)
    print(f"[z426] wrote {RESULTS/'waterfall_VG1_0p6.png'}")


def write_diagnosis(summary, waterfall):
    lines = ["# z426 Diagnosis — Which term is wrong?\n"]
    # Hypothesis A: Ic_Q1 huge in pyport but not aggregated into I_D.
    # Check by comparing |Ic_Q1| to Id_predicted_total and Id_reconstructed.
    lines.append("## Per-bias breakdown (key signals)\n")
    lines.append("| bias | V_B | Ids_M1 | Ic_Q1 | Mario_Ipos | Id_pred | Id_meas | gap[dec] |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for lbl, r in summary.items():
        if "error" in r:
            lines.append(f"| {lbl} | ERR |  |  |  |  |  |  |")
            continue
        lines.append(
            f"| {lbl} | {r['V_B']:+.3f} | {fmt(r['Ids_M1'])} | {fmt(r['Ic_Q1'])} | "
            f"{fmt(r['Mario_Ipos_body'])} | {fmt(r['Id_predicted_total'])} | "
            f"{fmt(r['Id_measured'])} | {r['gap_dec']:+.2f} |"
        )

    # Test each hypothesis numerically
    lines.append("\n## Hypothesis test\n")
    # Pick the post-snapback point (Vd=2.0, VG1=0.6) where gap should be large
    key = "B4_vg1_0p6_vg2_0p0_vd_2p0"
    if key in summary and "error" not in summary[key]:
        r = summary[key]
        Id_meas = r["Id_measured"]
        Ic_Q1 = r["Ic_Q1"]
        Ids_M1 = r["Ids_M1"]
        Ipos = r["Mario_Ipos_body"]
        Vb = r["V_B"]
        Id_pred = r["Id_predicted_total"]

        lines.append(f"### Post-snapback diagnostic point: {key}\n")
        lines.append(f"- Measured I_D = {Id_meas:.3e} A")
        lines.append(f"- Predicted I_D = {Id_pred:.3e} A")
        lines.append(f"- Gap = {r['gap_dec']:+.2f} decades")
        lines.append(f"- V_B = {Vb:+.3f} V (BJT threshold ~0.7 V)")
        lines.append(f"- Ic_Q1 (BJT collector) = {Ic_Q1:+.3e} A")
        lines.append(f"- Ids_M1 (channel)      = {Ids_M1:+.3e} A")
        lines.append(f"- Mario Ipos at body    = {Ipos:+.3e} A\n")

        # Hypothesis A check: is Ic_Q1 huge relative to Id_pred?
        if abs(Ic_Q1) > 10 * abs(Id_pred):
            lines.append("**Hypothesis A SUPPORTED**: Ic_Q1 huge but NOT propagating to Id "
                         "→ aggregator bug in dc.py/_residuals → forward Id formula.\n")
        elif abs(Ic_Q1) < 1e-3 * abs(Id_meas) and Vb > 0.6:
            lines.append("**Hypothesis B SUPPORTED**: V_B above BJT turn-on (0.7 V) "
                         "but Ic_Q1 still tiny → bug in `compute_bjt` "
                         "(wrong Vt? wrong sign? `be_oneway` gate clamping?).\n")
            # Compute what Ic_Q1 SHOULD be analytically
            Vt = 0.02585  # kT/q at 300 K
            Vsint = r["V_Sint"]
            Vbe = Vb - Vsint
            try:
                exp_term = math.exp(min(80.0, Vbe / Vt))
                Is = 5e-9 * 1e-6  # with area=1e-6
                Bf = 10000.0
                Ic_theory_no_kqb = Is * exp_term
                lines.append(f"- Theoretical Icc (Is·exp(Vbe/Vt), area=1e-6, "
                             f"Vbe={Vbe:.3f}) ≈ {Ic_theory_no_kqb:.3e} A\n")
            except Exception:
                pass
        elif abs(Ids_M1) > 0.5 * abs(Id_pred) and abs(Ic_Q1) < abs(Ids_M1):
            lines.append("**Hypothesis C plausible**: channel dominates predicted Id, "
                         "BJT amplification too weak to swing the µA budget.\n")
        else:
            lines.append("**Hypothesis D plausible**: all components reasonable but "
                         "still 2+ dec below measurement → missing mechanism "
                         "(latch-up, parasitic SCR, lateral BJT not captured).\n")

        # Sanity: Mario Ipos should be O(1e-6) and stay at body (NOT show up in Id directly)
        if abs(Ipos) > 0 and abs(Id_pred) > 0 and abs(Ipos - abs(Id_pred)) < 1e-12:
            lines.append("**ROUTING BUG**: Id_pred numerically equals Mario_Ipos → "
                         "Ipos is being added directly to drain, not body.\n")

    lines.append("\n## Sign-convention sanity (V_D=0 expected: all currents ≈ 0)\n")
    if waterfall:
        r0 = waterfall[0]
        lines.append(f"- At Vd={r0['Vd']:.3f}: Ids_M1={fmt(r0['Ids_M1'])}, "
                     f"Ic_Q1={fmt(r0['Ic_Q1'])}, Iii_M1={fmt(r0['Iii_M1'])}\n")

    lines.append("\n## Suggested 5-line fix (if applicable)\n")
    lines.append("Inspect output for which hypothesis lit up. If A: check dc.py / "
                 "Id assembly in nsram_cell_2T.py:1910. If B: check bjt.py "
                 "`compute_bjt` (Vbe sign, gate=sigmoid((Vbe-0.7)/0.05) "
                 "may be clamping Icc to ~0 when V_BE just barely exceeds 0.7).\n")

    (RESULTS / "diagnosis.md").write_text("\n".join(lines) + "\n")
    print(f"[z426] wrote {RESULTS/'diagnosis.md'}")


if __name__ == "__main__":
    main()
