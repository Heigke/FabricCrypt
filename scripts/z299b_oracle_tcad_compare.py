"""z299b — Compare oracle-extracted TCAD curves vs pyport replay (z299).

Maps each oracle-extracted I-V curve (slides 6, 8, 9, 14, 15) to the
nearest pyport-replay sweep (BV / IdVds / IdVgs / IdVgs1), interpolates
both onto the overlap region of the independent variable, and computes
log-RMSE on |I|.

Gate (relaxed):  any cmd-file < 0.8 dec log-RMSE  → PASS
Ambitious gate:  any cmd-file < 0.3 dec log-RMSE  → AMBITIOUS PASS
"""
from __future__ import annotations
import json, re, math
from pathlib import Path
import numpy as np

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
ORACLE_MD = ROOT / "research_plan/oracle_queries/O45_tcad_curve_extract/openai_response.md"
Z299 = ROOT / "results/z299_tcad_replay"
OUT = ROOT / "results/z299b_oracle_tcad_compare"
OUT.mkdir(parents=True, exist_ok=True)


def load_oracle():
    txt = ORACLE_MD.read_text()
    m = re.search(r"```json\s*(.*?)```", txt, re.S)
    return json.loads(m.group(1))


def load_replay(name: str):
    d = np.load(Z299 / f"replay_{name}.npz")
    return d["V"], d["Id"]


def log_rmse(I_a, I_b):
    """RMS of log10(|I_a|) - log10(|I_b|), floored at 1e-15."""
    floor = 1e-15
    a = np.log10(np.maximum(np.abs(I_a), floor))
    b = np.log10(np.maximum(np.abs(I_b), floor))
    return float(np.sqrt(np.mean((a - b) ** 2)))


def log_rmse_shape(I_a, I_b):
    """log-RMSE after removing constant log-offset (i.e., best vertical
    shift). Measures curve SHAPE agreement, not absolute level."""
    floor = 1e-15
    a = np.log10(np.maximum(np.abs(I_a), floor))
    b = np.log10(np.maximum(np.abs(I_b), floor))
    offset = float(np.mean(a - b))
    return float(np.sqrt(np.mean((a - b - offset) ** 2))), offset


def interp_overlap(samples, V_rep, I_rep):
    """samples = list of [V, I]. Interpolate replay onto sample Vs over
    the overlap range, also interpolate oracle samples onto the same Vs."""
    samples = np.array(samples, dtype=float)
    Vo, Io = samples[:, 0], samples[:, 1]
    vmin = max(np.min(Vo), np.min(V_rep))
    vmax = min(np.max(Vo), np.max(V_rep))
    if vmax <= vmin:
        return None
    # query points: oracle samples that lie inside the overlap
    mask = (Vo >= vmin) & (Vo <= vmax)
    if mask.sum() < 2:
        return None
    Vq = Vo[mask]
    Io_q = Io[mask]
    # Interpolate replay onto Vq (handle non-monotonic V by sorting)
    order = np.argsort(V_rep)
    Ir_q = np.interp(Vq, V_rep[order], I_rep[order])
    return Vq, Io_q, Ir_q


# -- mapping: which oracle curves correspond to which replay sweeps?
# Slide 6: I-V family at multiple VG1; x=Vd → maps to IdVds (V=Vd 0..2)
# Slide 9: 3-corner overlay; same idea, IdVds
# Slide 8: I-V noise band envelopes; IdVds
# Slide 14: Bulk current vs ?; assume vs Vd or Vg — try IdVds and IdVgs
# Slide 15: Transient V_D ramps over time → BV (Vd ramp 0..5 V)
# Slides 1-4: parameter vs VG → NOT a TCAD I-V output, skip
# Slide 21: pdiode dynamic response (V or I vs t) → BV (Vd ramp 0..5)

# Heuristic: based on x-axis units and value range
def axis_is_voltage(s):
    return "V" in s and "[" not in s.split("V")[0][-3:].lower() or "Vd" in s or "Vg" in s


def classify_curve(e):
    """Return list of candidate replay names to try, or [] to skip."""
    xa = e.get("x_axis", "").lower()
    ya = e.get("y_axis", "").lower()
    slide = e["slide"]
    # transient (time on x-axis) → try BV (which is a Vd ramp)
    if "t (" in xa or "time" in xa or xa.startswith("t "):
        return ["BV_des"]
    # current on y axis vs voltage on x → I-V
    is_current = ("a)" in ya or "[a]" in ya or " a" in ya or "id" in ya or "current" in ya)
    is_voltage_x = any(tok in xa for tok in ["v ", "v(", "vd", "vg", "vds", "vgs", "(v)"])
    if not (is_current and is_voltage_x):
        return []
    # detect Vd vs Vg from axis label (handle "v_d", "vd", "vds", "drain")
    has_vd = any(tok in xa for tok in ["vd", "v_d", "vds", "v_ds", "drain"])
    has_vg = any(tok in xa for tok in ["vg", "v_g", "vgs", "v_gs", "gate"])
    if has_vd and not has_vg:
        return ["IdVds_des", "BV_des"]
    if has_vg and not has_vd:
        return ["IdVgs_des", "IdVgs1_des"]
    # generic "Voltage (V)" + drain-style range → try all I-V vs V replays
    return ["IdVds_des", "BV_des", "IdVgs_des", "IdVgs1_des"]


def main():
    data = load_oracle()
    extractions = data["extractions"]

    replays = {n: load_replay(n) for n in ["BV_des", "IdVds_des", "IdVgs_des", "IdVgs1_des"]}

    comparisons = []
    for e in extractions:
        candidates = classify_curve(e)
        if not candidates:
            continue
        best = None
        for rep in candidates:
            V_r, I_r = replays[rep]
            res = interp_overlap(e["samples"], V_r, I_r)
            if res is None:
                continue
            Vq, Io_q, Ir_q = res
            rmse = log_rmse(Io_q, Ir_q)
            shape_rmse, offset = log_rmse_shape(Io_q, Ir_q)
            entry = {
                "rep": rep,
                "n_overlap": int(len(Vq)),
                "V_overlap": [float(np.min(Vq)), float(np.max(Vq))],
                "log_rmse_dec": rmse,
                "log_rmse_shape_dec": shape_rmse,
                "log_offset_dec": offset,
            }
            if best is None or rmse < best["log_rmse_dec"]:
                best = entry
        if best is None:
            continue
        comparisons.append({
            "slide": e["slide"],
            "curve_label": e["curve_label"],
            "confidence": e.get("confidence"),
            "x_axis": e.get("x_axis"),
            "y_axis": e.get("y_axis"),
            "best_match": best,
        })

    # Best per replay
    by_rep = {}
    for c in comparisons:
        rep = c["best_match"]["rep"]
        rmse = c["best_match"]["log_rmse_dec"]
        if rep not in by_rep or rmse < by_rep[rep]["log_rmse_dec"]:
            by_rep[rep] = {
                "log_rmse_dec": rmse,
                "slide": c["slide"],
                "curve_label": c["curve_label"],
                "n_overlap": c["best_match"]["n_overlap"],
            }

    any_rmse = [c["best_match"]["log_rmse_dec"] for c in comparisons]
    any_shape = [c["best_match"]["log_rmse_shape_dec"] for c in comparisons]
    best_overall = min(any_rmse) if any_rmse else float("inf")
    best_shape = min(any_shape) if any_shape else float("inf")

    gate_relaxed = best_overall < 0.8
    gate_ambitious = best_overall < 0.3
    gate_shape_relaxed = best_shape < 0.8

    summary = {
        "experiment": "z299b_oracle_tcad_compare",
        "oracle_packet": "research_plan/oracle_queries/O45_tcad_curve_extract",
        "oracle_provider": "openai (gpt-5)",
        "n_oracle_curves_extracted": len(extractions),
        "n_curves_compared": len(comparisons),
        "best_overall_log_rmse_dec": best_overall,
        "best_overall_shape_log_rmse_dec": best_shape,
        "best_per_replay": by_rep,
        "gate_relaxed_lt_0p8": gate_relaxed,
        "gate_ambitious_lt_0p3": gate_ambitious,
        "gate_shape_relaxed_lt_0p8": gate_shape_relaxed,
        "comparisons": comparisons,
        "could_not_extract": data.get("could_not_extract", []),
        "notes": [
            "Oracle samples are visually estimated from small slide plots (±0.3 dec).",
            "pyport replay uses simplified analytical surrogate, not Sentaurus binaries.",
            "Large log-RMSE expected because (a) oracle uncertainty + (b) replay is",
            "stub physics. Goal is order-of-magnitude sanity, not exact match.",
        ],
    }

    out_file = OUT / "summary.json"
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"[z299b] wrote {out_file}")
    print(f"[z299b] curves compared: {len(comparisons)}")
    print(f"[z299b] best log-RMSE: {best_overall:.3f} dec")
    print(f"[z299b] gate <0.8 dec: {gate_relaxed}")
    print(f"[z299b] gate <0.3 dec: {gate_ambitious}")
    print("[z299b] best per replay:")
    for rep, info in by_rep.items():
        print(f"   {rep}: {info['log_rmse_dec']:.3f} dec (slide {info['slide']}: {info['curve_label'][:60]})")


if __name__ == "__main__":
    main()
