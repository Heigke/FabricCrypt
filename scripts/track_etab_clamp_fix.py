#!/usr/bin/env python3
"""Track etab/T3_d gated upper-clamp validation.

BSIM4 v4.5 b4ld.c §1107-1117 only has a LOWER soft-clamp on
  T3 = eta0 + etab*Vbseff
via the rational regularizer when T3 < 1e-4. There is NO upper clamp in
canonical BSIM4. The proposed upper clamp (max=2.0 / etc.) is therefore a
deliberate non-BSIM4 modification, motivated by Sebas card's anomalous
etab≈1.8-2.5 (canonical ≈-0.07), which lets T3 explode to 2-4 at snapback
(Vbseff→1V) and over-suppress Vth in DIBL_Sft.

This script gates the upper clamp behind the model-card flag
`dibl_upper_clamp` (default OFF — all existing fits unchanged) and
validates the ON branch against:
  (1) full-33 fwd+bwd median_dec  (must improve over 0.461)
  (2) ngspice 9-bias xval mean gap (must improve over 0.808)
  (3) VG1=0.6 knee shift to ≤1.2V from prior ~1.5V

Plus baseline conditions are MEASURED IN THIS RUN (not taken from stale
prior dirs) so the comparison is apples-to-apples.

Stack-under-test (both A and B):
  - K1@VG1=0.6 = 0.53825
  - ALPHA0 = 7.83756e-4
  - tlpe1_disable = True (both M1, M2)
B adds:
  - dibl_upper_clamp = True, dibl_upper_clamp_max = 1.0 (M1, M2)

The "max=2.0" originally suggested would not bind at snapback since
T3_d there can reach ~2-4 with etab=2.5 and Vbs→1V — so max=2.0 still lets
T3 hit the ceiling at 2.0. We use max=1.0 by default (T3≤1.0, theta0vb0≤1
so DIBL_Sft ≤ Vds), which is a defensible physical-ish bound: DIBL shift
≤ Vds itself. Documented in verdict.
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import sys, json, time, traceback, importlib.util
from pathlib import Path
import numpy as np

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results/track_etab_clamp_fix"
OUT.mkdir(parents=True, exist_ok=True)

# Import pillar
sp = importlib.util.spec_from_file_location("pillar_I", ROOT / "scripts/pillar_I_C3_jts_tat.py")
pillar = importlib.util.module_from_spec(sp); sp.loader.exec_module(pillar)

# Import ngspice xval module
sp2 = importlib.util.spec_from_file_location("ngx", ROOT / "scripts/track_ngspice_xval.py")
ngx = importlib.util.module_from_spec(sp2); sp2.loader.exec_module(ngx)

# Stack
K1_OVR = 0.53825
ALPHA0_OVR = 7.83756e-4
UPPER_MAX = 1.0   # see header note


def apply_model_flags(M1, M2, enable_clamp: bool):
    """Set tlpe1_disable and (optionally) dibl_upper_clamp on M1+M2."""
    for M in (M1, M2):
        M._values["tlpe1_disable"] = True
        if enable_clamp:
            M._values["dibl_upper_clamp"] = True
            M._values["dibl_upper_clamp_max"] = UPPER_MAX
        else:
            M._values.pop("dibl_upper_clamp", None)
            M._values.pop("dibl_upper_clamp_max", None)


def clear_model_flags(M1, M2):
    for M in (M1, M2):
        M._values.pop("tlpe1_disable", None)
        M._values.pop("dibl_upper_clamp", None)
        M._values.pop("dibl_upper_clamp_max", None)


# -----------------------------------------------------------------------------
# (1) Full-33 dec ablation
# -----------------------------------------------------------------------------
def run_full33(label: str, enable_clamp: bool, curves, sebas_rows):
    print(f"\n[full33] === {label}  clamp={enable_clamp} ===", flush=True)
    cfg, M1, M2, bjt = pillar.build_pyport_base()
    apply_model_flags(M1, M2, enable_clamp)

    # K1 + ALPHA0 like track_combo
    saved_branch_k1 = pillar.BRANCH_FLAT[0.6]["K1"]
    pillar.BRANCH_FLAT[0.6]["K1"] = K1_OVR

    orig_make = pillar.make_overrides
    def patched_make(sebas_row):
        P_M1, P_M2 = orig_make(sebas_row)
        if P_M1 is None: P_M1 = {}
        if P_M2 is None: P_M2 = {}
        P_M1["alpha0"] = float(ALPHA0_OVR)
        P_M2["alpha0"] = float(ALPHA0_OVR)
        if sebas_row is not None and abs(sebas_row.get("VG1", float("nan")) - 0.6) < 1e-6:
            P_M1["k1"] = float(K1_OVR)
        return P_M1, P_M2
    pillar.make_overrides = patched_make

    try:
        t0 = time.time()
        rows, nan_count = pillar.run_grid(cfg, M1, M2, bjt, curves, sebas_rows,
                                          label, do_bwd=True)
        dt = time.time() - t0
    finally:
        pillar.make_overrides = orig_make
        pillar.BRANCH_FLAT[0.6]["K1"] = saved_branch_k1
        clear_model_flags(M1, M2)

    summ = pillar.summarize(rows, label)
    summ["runtime_s"] = dt
    summ["nan_count"] = int(nan_count)
    summ["n_rows"] = len(rows)
    # also per-VG1 breakdown
    for vg1 in (0.2, 0.4, 0.6):
        sub = [r["med_dec"] for r in rows
               if abs(r["VG1"] - vg1) < 1e-6 and np.isfinite(r["med_dec"])]
        summ[f"median_dec_VG1={vg1}_grouped"] = float(np.median(sub)) if sub else float("nan")
    print(f"[full33] {label}: median_dec_all = {summ['median_dec_all']['median']:.4f} "
          f"(n_finite={sum(1 for r in rows if np.isfinite(r['med_dec']))}/{len(rows)}, "
          f"{dt:.1f}s)", flush=True)
    return summ, rows


# -----------------------------------------------------------------------------
# (2) Knee shift
# -----------------------------------------------------------------------------
def knee_vd(Vd, Id, factor=10, base_window=(0.0, 0.3)):
    Id = np.asarray(Id); Vd = np.asarray(Vd)
    mask = (Vd >= base_window[0]) & (Vd <= base_window[1])
    base = np.median(Id[mask]) if mask.sum() else 1e-12
    above = Id > factor * max(base, 1e-13)
    idx = np.argmax(above)
    return float(Vd[idx]) if above.any() else float("nan")


def measure_knee(label: str, enable_clamp: bool, curves, sebas_rows):
    """For each VG2 at VG1=0.6, compute model knee position via fwd_2t."""
    import torch
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    cfg, M1, M2, bjt = pillar.build_pyport_base()
    apply_model_flags(M1, M2, enable_clamp)
    sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)

    knees = {}
    for c in curves:
        if abs(c["VG1"] - 0.6) > 1e-6:
            continue
        vg2 = float(c["VG2"])
        row, _ = pillar.find_or_impute_row(sebas_rows, c["VG1"], vg2)
        P_M1, P_M2 = pillar.make_overrides(row) if row else (None, None)
        if P_M1 is None: P_M1 = {}
        if P_M2 is None: P_M2 = {}
        P_M1["alpha0"] = float(ALPHA0_OVR); P_M2["alpha0"] = float(ALPHA0_OVR)
        P_M1["k1"] = float(K1_OVR)
        Vd_np = c["fwd_Vd"]; Id_data = c["fwd_Id"]
        Vd_t = torch.tensor(Vd_np, dtype=torch.float64)
        try:
            with pillar.patch_sd_scaled(sd_M1, P_M1), pillar.patch_sd_scaled(sd_M2, P_M2):
                out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt,
                                 Vd_seq=Vd_t,
                                 VG1=torch.tensor(c["VG1"], dtype=torch.float64),
                                 VG2=torch.tensor(vg2, dtype=torch.float64),
                                 warm_start=True)
            Id_model = np.abs(out["Id"].detach().cpu().numpy())
        except Exception as e:
            print(f"  knee VG2={vg2}: FAIL {e}")
            continue
        knees[vg2] = {
            "VG2": vg2,
            "knee_data": knee_vd(Vd_np, Id_data),
            "knee_model": knee_vd(Vd_np, Id_model),
            "n_Vd": len(Vd_np),
        }
    clear_model_flags(M1, M2)
    print(f"[knee] {label} clamp={enable_clamp}:")
    for vg2, k in sorted(knees.items()):
        print(f"  VG2={vg2:+.2f}  data={k['knee_data']:.3f}V  model={k['knee_model']:.3f}V")
    return knees


# -----------------------------------------------------------------------------
# (3) ngspice xval — re-use ngx but patch pyport builder to inject our flags
# -----------------------------------------------------------------------------
def run_ngspice_xval(label: str, enable_clamp: bool):
    """Run ngspice xval at 9 biases with our model flags applied to M1/M2.

    Strategy: monkey-patch ngx.build_pyport_with_hurkx so post-build it sets
    the flags.
    """
    orig_builder = ngx.build_pyport_with_hurkx

    def patched(use_hurkx=True):
        cfg, M1, M2, bjt = orig_builder(use_hurkx=use_hurkx)
        apply_model_flags(M1, M2, enable_clamp)
        return cfg, M1, M2, bjt

    ngx.build_pyport_with_hurkx = patched

    biases = [(vg1, vg2) for vg1 in ngx.VG1_GRID for vg2 in ngx.VG2_GRID]
    cfg_h, M1_h, M2_h, bjt_h = patched(use_hurkx=True)
    cfg_nh, M1_nh, M2_nh, bjt_nh = patched(use_hurkx=False)
    Vd_axis = np.arange(ngx.VD_LO, ngx.VD_HI + 1e-9, ngx.VD_STEP)

    results = []
    for (vg1, vg2) in biases:
        tag = (f"VG1={vg1:.2f}_VG2={vg2:.2f}_{label}"
               .replace("-", "m").replace(".", "p"))
        meas_csv = ngx.find_meas_csv(vg1, vg2)
        if meas_csv is None or not meas_csv.exists():
            print(f"  [ngx] VG1={vg1} VG2={vg2} skip (no meas)")
            continue
        Vd_meas, Id_meas = ngx.load_measured(meas_csv)
        Vd_ng, Id_ng, st = ngx.run_ngspice(vg1, vg2, tag)
        if Vd_ng is None:
            print(f"  [ngx] VG1={vg1} VG2={vg2} ngspice FAIL: {st}")
            continue
        Id_py_nh = ngx.run_pyport_sweep(cfg_nh, M1_nh, M2_nh, bjt_nh,
                                         vg1, vg2, Vd_axis)
        Id_py_nh_on_ng = np.interp(Vd_ng, Vd_axis, np.abs(Id_py_nh))
        m_dir = (Vd_ng > 0.3) & (Id_ng > 0) & (Id_py_nh_on_ng > 0)
        if m_dir.sum() >= 3:
            l_ng = np.log10(np.clip(Id_ng[m_dir], 1e-30, None))
            l_py = np.log10(np.clip(Id_py_nh_on_ng[m_dir], 1e-30, None))
            gap_med = float(np.median(np.abs(l_ng - l_py)))
        else:
            gap_med = float("nan")
        py_dec, _ = ngx.median_dec(Vd_meas, Id_meas, Vd_axis, Id_py_nh)
        print(f"  [ngx] VG1={vg1} VG2={vg2}: ng↔py med={gap_med:.3f}  "
              f"py_dec_vs_data={py_dec:.3f}")
        results.append({
            "VG1": vg1, "VG2": vg2,
            "ngspice_vs_pyport_nohurkx_med_dec": gap_med,
            "pyport_nohurkx_dec_vs_data": py_dec,
        })

    # restore
    ngx.build_pyport_with_hurkx = orig_builder
    clear_model_flags(M1_h, M2_h)
    clear_model_flags(M1_nh, M2_nh)

    gaps = [r["ngspice_vs_pyport_nohurkx_med_dec"] for r in results
            if np.isfinite(r["ngspice_vs_pyport_nohurkx_med_dec"])]
    mean_gap = float(np.mean(gaps)) if gaps else float("nan")
    print(f"[ngx] {label} clamp={enable_clamp}: mean gap = {mean_gap:.4f} "
          f"(n={len(gaps)}/{len(biases)})")
    return {"per_bias": results, "mean_gap": mean_gap, "n": len(gaps)}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("=== track_etab_clamp_fix ===")
    print(f"  Stack: K1={K1_OVR}, ALPHA0={ALPHA0_OVR}, tlpe1_disable=True, "
          f"dibl_upper_clamp_max={UPPER_MAX}")
    print()
    sebas_rows = pillar.load_sebas_params()
    curves = pillar.load_curves()
    print(f"loaded {len(curves)} curves, {len(sebas_rows)} sebas rows\n")

    summary = {}

    # (1) Full-33 dec, OFF and ON
    print("=" * 60)
    print("PHASE 1: full-33 dec ablation")
    print("=" * 60)
    summ_off, rows_off = run_full33("OFF_baseline", enable_clamp=False,
                                      curves=curves, sebas_rows=sebas_rows)
    summary["full33_off"] = summ_off
    with open(OUT / "ablation.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    summ_on, rows_on = run_full33("ON_clamp", enable_clamp=True,
                                    curves=curves, sebas_rows=sebas_rows)
    summary["full33_on"] = summ_on
    with open(OUT / "ablation.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # (2) Knee shift
    print("\n" + "=" * 60)
    print("PHASE 2: knee shift at VG1=0.6")
    print("=" * 60)
    knees_off = measure_knee("OFF_baseline", enable_clamp=False,
                               curves=curves, sebas_rows=sebas_rows)
    knees_on = measure_knee("ON_clamp", enable_clamp=True,
                              curves=curves, sebas_rows=sebas_rows)
    summary["knee_off"] = knees_off
    summary["knee_on"] = knees_on
    with open(OUT / "ablation.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # (3) ngspice xval
    print("\n" + "=" * 60)
    print("PHASE 3: ngspice xval (9 biases, gate OFF and ON)")
    print("=" * 60)
    ngx_off = run_ngspice_xval("OFF_baseline", enable_clamp=False)
    ngx_on = run_ngspice_xval("ON_clamp", enable_clamp=True)
    summary["ngspice_xval_off"] = ngx_off
    summary["ngspice_xval_on"] = ngx_on
    with open(OUT / "ngspice_xval.json", "w") as f:
        json.dump({"off": ngx_off, "on": ngx_on}, f, indent=2, default=str)
    with open(OUT / "ablation.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # -------- verdict.md ----------
    print("\n" + "=" * 60)
    print("Writing verdict.md")
    print("=" * 60)
    med_off = summ_off["median_dec_all"]["median"]
    med_on = summ_on["median_dec_all"]["median"]
    gap_off = ngx_off["mean_gap"]
    gap_on = ngx_on["mean_gap"]

    pass_dec  = med_on <= 0.4
    pass_gap  = gap_on <= 0.4

    # Knee shift summary: per VG2, did model knee move closer to data?
    knee_lines = []
    knee_pass_vals = []
    for vg2 in sorted(set(knees_off) & set(knees_on)):
        k_off = knees_off[vg2]["knee_model"]
        k_on  = knees_on[vg2]["knee_model"]
        k_dat = knees_off[vg2]["knee_data"]
        knee_pass_vals.append(k_on)
        knee_lines.append(
            f"| {vg2:+.2f} | {k_dat:.3f} | {k_off:.3f} | {k_on:.3f} | {k_on - k_off:+.3f} |")
    knee_max = float(np.nanmax(knee_pass_vals)) if knee_pass_vals else float("nan")
    pass_knee = knee_max <= 1.2

    overall = "PASS" if (pass_dec and pass_gap and pass_knee) else "FAIL"

    lines = []
    lines.append("# track_etab_clamp_fix — verdict\n")
    lines.append("## BSIM4 reference clamp (verified)\n")
    lines.append("Sources inspected:")
    lines.append("- `external/bsim4/code/b4ld.c` §1107-1117 (BSIM4 v4.5 reference)")
    lines.append("- `ngspice-42+ds/src/spicelib/devices/bsim4/b4ld.c` §1147-1153 (ngspice-42)\n")
    lines.append("```c")
    lines.append("T3 = here->BSIM4eta0 + pParam->BSIM4etab * Vbseff;")
    lines.append("if (T3 < 1.0e-4)")
    lines.append("{   T9 = 1.0 / (3.0 - 2.0e4 * T3);")
    lines.append("    T3 = (2.0e-4 - T3) * T9;")
    lines.append("    T4 = T9 * T9;")
    lines.append("}")
    lines.append("```\n")
    lines.append("**Canonical BSIM4 has ONLY a LOWER soft-clamp at 1e-4** (rational")
    lines.append("regularizer). There is **NO upper clamp** in BSIM4 v4.5 or ngspice-42.")
    lines.append("The proposed upper clamp is therefore a deliberate non-BSIM4")
    lines.append("modification, motivated by Sebas' anomalous etab≈1.8-2.5 (vs canonical ~-0.07).\n")
    lines.append(f"Patch is gated behind `model['dibl_upper_clamp']=True` with")
    lines.append(f"`model['dibl_upper_clamp_max']` (default 1.0; this run used {UPPER_MAX}).\n")
    lines.append("Why max=1.0 not 2.0? With etab≈2.5 and Vbseff→1V, T3_d can")
    lines.append("reach ~4; capping at 2.0 still allows DIBL_Sft = 2·θ0vb0·Vds")
    lines.append("which fully suppresses Vth. max=1.0 keeps the DIBL shift bounded")
    lines.append("by ~Vds (since θ0vb0≤1).\n")

    lines.append("## Patch location\n")
    lines.append("`nsram/nsram/bsim4_port/dc.py` after the canonical lower-clamp `T3_clamped` definition.\n")

    lines.append("## Results\n")
    lines.append("### Full-33 fwd+bwd median_dec (in-run baseline vs ON)\n")
    lines.append(f"- OFF (gate disabled, K1+ALPHA0+Tlpe1):  **{med_off:.4f} dec**")
    lines.append(f"- ON  (gate enabled,  K1+ALPHA0+Tlpe1):  **{med_on:.4f} dec**")
    lines.append(f"- Δ = {med_on - med_off:+.4f} dec")
    lines.append(f"- PASS gate (≤0.4): **{'PASS' if pass_dec else 'FAIL'}** ({med_on:.4f})\n")

    for vg1 in (0.2, 0.4, 0.6):
        lines.append(f"- per-VG1={vg1}:  "
                     f"OFF={summ_off[f'median_dec_VG1={vg1}_grouped']:.4f}, "
                     f"ON={summ_on[f'median_dec_VG1={vg1}_grouped']:.4f}")
    lines.append("")

    lines.append("### ngspice 9-bias mean gap (in-run baseline vs ON)\n")
    lines.append(f"- OFF: **{gap_off:.4f} dec** (n={ngx_off['n']})")
    lines.append(f"- ON:  **{gap_on:.4f} dec** (n={ngx_on['n']})")
    lines.append(f"- Δ = {gap_on - gap_off:+.4f} dec")
    lines.append(f"- PASS gate (≤0.4): **{'PASS' if pass_gap else 'FAIL'}**\n")

    lines.append("### Knee at VG1=0.6 (V where model Id crosses 10× baseline)\n")
    lines.append("| VG2 | data | model OFF | model ON | Δ(ON−OFF) |")
    lines.append("|----:|----:|----:|----:|----:|")
    lines.extend(knee_lines)
    lines.append(f"\n- max(model_ON across VG2) = **{knee_max:.3f}V**")
    lines.append(f"- PASS gate (≤1.2V): **{'PASS' if pass_knee else 'FAIL'}**\n")

    lines.append(f"## OVERALL: **{overall}**\n")
    if not pass_dec:
        lines.append(f"- full-33 dec did not improve below 0.4 (got {med_on:.4f}).")
    if not pass_gap:
        lines.append(f"- ngspice gap did not improve below 0.4 (got {gap_on:.4f}).")
    if not pass_knee:
        lines.append(f"- knee did not shift left to ≤1.2V (max {knee_max:.3f}V).")
    if overall == "FAIL":
        lines.append("\n**Interpretation**: the etab/T3_d upper clamp is NOT the")
        lines.append("right answer for the residual gap. The Sebas anomalous etab")
        lines.append("is the symptom of a different root cause (likely Vb attractor")
        lines.append("or charge-conservation issue in the snapback branch).")
    else:
        lines.append("\n**Interpretation**: clamping T3_d in DIBL_Sft addresses")
        lines.append("the Sebas etab anomaly's downstream effect on Vth in snapback.")
        lines.append("Gate remains OFF by default; enable per fit if desired.")

    (OUT / "verdict.md").write_text("\n".join(lines))
    print((OUT / "verdict.md").read_text())
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
