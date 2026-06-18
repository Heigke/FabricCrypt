#!/usr/bin/env python3
"""Track Newton-root fix — validate cfg.newton_root_seed='high_Vb_continuation'
against (a) full-33 K1×ALPHA0 ablation WITHOUT Hurkx and (b) 9-bias ngspice xval.

Forensic hypothesis: pyport Newton lands on a spurious low-Vb root; Vd
source-stepping continuation should push it to the physical high-Vb root.

Outputs:
  results/track_newton_root_fix/ablation.json    (K1×ALPHA0 grid, fix ON, no Hurkx)
  results/track_newton_root_fix/ngspice_xval.json (9-bias re-validation)
  results/track_newton_root_fix/verdict.md        (PASS/FAIL declaration)
"""
from __future__ import annotations
import os, sys, json, time, traceback
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
from pathlib import Path
import numpy as np

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

import importlib.util
sp = importlib.util.spec_from_file_location("pillar_I", ROOT / "scripts/pillar_I_C3_jts_tat.py")
pillar = importlib.util.module_from_spec(sp); sp.loader.exec_module(pillar)
sp = importlib.util.spec_from_file_location("xv", ROOT / "scripts/track_ngspice_xval.py")
xv = importlib.util.module_from_spec(sp); sp.loader.exec_module(xv)

OUT = ROOT / "results/track_newton_root_fix"
OUT.mkdir(parents=True, exist_ok=True)

# Same grids as track_combo_k1_alpha0
K1_BASELINE   = 0.41825
ALPHA0_CSV    = 7.842e-5
K1_CARD       = 0.53825
K1_PUSH       = 0.6459
ALPHA0_CARD   = 7.83756e-4
K1_GRID     = [K1_BASELINE, K1_CARD, K1_PUSH]
ALPHA0_GRID = [ALPHA0_CSV, ALPHA0_CARD]
BASELINE_MEDIAN_DEC = 1.163
PRIOR_BEST_DEC = 0.665  # K1_CARD + ALPHA0_CARD WITH Hurkx (from prior combo run)
SEED_MODE = "high_Vb_continuation"
SEED_N_STEPS = 40
SEED_VB_INIT = 0.95


def run_combo_grid():
    """Full 33-bias ablation grid with newton_root_seed ON, Hurkx OFF."""
    sebas_rows = pillar.load_sebas_params()
    curves = pillar.load_curves()
    print(f"[ablation] loaded {len(curves)} curves, {len(sebas_rows)} sebas rows", flush=True)

    results = {}
    Vb_log = {}  # per-condition per-bias converged Vb at high Vd
    for k1 in K1_GRID:
        for a0 in ALPHA0_GRID:
            tag = f"K1={k1:.5f}__ALPHA0={a0:.4e}"
            print(f"[ablation] === {tag} (fix={SEED_MODE}, Hurkx=OFF) ===", flush=True)
            cfg, M1, M2, bjt = pillar.build_pyport_base()
            cfg.hurkx_bbt_A = 0.0  # NO Hurkx fallback
            cfg.newton_root_seed = SEED_MODE
            cfg.newton_root_seed_n_steps = SEED_N_STEPS
            cfg.newton_root_seed_vb_init = SEED_VB_INIT

            saved_branch_k1 = pillar.BRANCH_FLAT[0.6]["K1"]
            pillar.BRANCH_FLAT[0.6]["K1"] = float(k1)

            orig_make = pillar.make_overrides
            def patched_make(sebas_row, _a0=float(a0), _k1=float(k1)):
                P_M1, P_M2 = orig_make(sebas_row)
                if P_M1 is None: P_M1 = {}
                if P_M2 is None: P_M2 = {}
                P_M1["alpha0"] = _a0
                P_M2["alpha0"] = _a0
                if sebas_row is not None and abs(sebas_row.get("VG1", float("nan")) - 0.6) < 1e-6:
                    P_M1["k1"] = _k1
                return P_M1, P_M2
            pillar.make_overrides = patched_make
            try:
                t0 = time.time()
                rows, nan_count = pillar.run_grid(cfg, M1, M2, bjt, curves, sebas_rows, tag, do_bwd=True)
                dt = time.time() - t0
            except Exception as e:
                print(f"  FAIL: {e}"); traceback.print_exc()
                results[tag] = {"label": tag, "error": str(e), "k1_vg1_0p6": k1, "alpha0": a0}
                pillar.make_overrides = orig_make
                pillar.BRANCH_FLAT[0.6]["K1"] = saved_branch_k1
                continue
            finally:
                pillar.make_overrides = orig_make
                pillar.BRANCH_FLAT[0.6]["K1"] = saved_branch_k1

            summ = pillar.summarize(rows, tag)
            summ["k1_vg1_0p6"] = float(k1); summ["alpha0"] = float(a0)
            summ["nan_count"] = int(nan_count); summ["runtime_s"] = float(dt)
            summ["n_rows"] = len(rows)
            finite = sum(1 for r in rows if np.isfinite(r["med_dec"]) and r["med_dec"] > 0)
            summ["n_finite"] = finite
            summ["convergence_rate"] = finite / max(len(rows), 1)
            results[tag] = summ
            with open(OUT / "ablation.json", "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"  median_dec_all={summ['median_dec_all']['median']:.3f}  "
                  f"conv={summ['convergence_rate']:.2f}  dt={dt:.1f}s", flush=True)

            # Check APU thermal
            try:
                temp = float(open("/sys/class/thermal/thermal_zone0/temp").read().strip()) / 1000.0
                if temp > 75:
                    print(f"  APU={temp:.1f}°C → pausing 30s", flush=True)
                    time.sleep(30)
            except Exception:
                pass
    return results


def run_ngspice_xval():
    """9-bias ngspice xval with newton_root_seed ON."""
    print("\n[xval] Building pyport (no Hurkx) with newton_root_seed=high_Vb_continuation")
    cfg_nh, M1_nh, M2_nh, bjt_nh = xv.build_pyport_with_hurkx(use_hurkx=False)
    cfg_nh.newton_root_seed = SEED_MODE
    cfg_nh.newton_root_seed_n_steps = SEED_N_STEPS
    cfg_nh.newton_root_seed_vb_init = SEED_VB_INIT

    Vd_axis = np.arange(xv.VD_LO, xv.VD_HI + 1e-9, xv.VD_STEP)
    biases = [(vg1, vg2) for vg1 in xv.VG1_GRID for vg2 in xv.VG2_GRID]

    out = {"config": {"K1": xv.K1_OVERRIDE, "ALPHA0": xv.ALPHA0_OVERRIDE,
                      "Hurkx": "OFF", "newton_root_seed": SEED_MODE,
                      "seed_vb_init": SEED_VB_INIT, "seed_n_steps": SEED_N_STEPS},
           "biases": []}

    for (vg1, vg2) in biases:
        tag = f"VG1={vg1:.2f}_VG2={vg2:.2f}".replace("-", "m").replace(".", "p")
        print(f"\n[xval] --- VG1={vg1} VG2={vg2} ---", flush=True)
        meas_csv = xv.find_meas_csv(vg1, vg2)
        if meas_csv is None:
            out["biases"].append({"VG1": vg1, "VG2": vg2, "skip": "no_meas"})
            continue
        Vd_meas, Id_meas = xv.load_measured(meas_csv)

        # ngspice (use existing decks if present)
        t0 = time.time()
        Vd_ng, Id_ng, ng_status = xv.run_ngspice(vg1, vg2, "newton_" + tag)
        ng_dt = time.time() - t0
        if Vd_ng is None:
            print(f"  ngspice FAIL: {ng_status}")
            out["biases"].append({"VG1": vg1, "VG2": vg2, "ng_status": ng_status})
            continue

        # pyport with fix
        t0 = time.time()
        Id_py = xv.run_pyport_sweep(cfg_nh, M1_nh, M2_nh, bjt_nh, vg1, vg2, Vd_axis)
        py_dt = time.time() - t0

        # Metrics
        ng_dec, ng_n = xv.median_dec(Vd_meas, Id_meas, Vd_ng, Id_ng)
        py_dec, py_n = xv.median_dec(Vd_meas, Id_meas, Vd_axis, Id_py)
        Id_py_on_ng = np.interp(Vd_ng, Vd_axis, np.abs(Id_py))
        m = (Vd_ng > 0.3) & (Id_ng > 0) & (Id_py_on_ng > 0)
        if m.sum() >= 3:
            l_ng = np.log10(np.clip(Id_ng[m], 1e-30, None))
            l_py = np.log10(np.clip(Id_py_on_ng[m], 1e-30, None))
            ng_vs_py_med = float(np.median(np.abs(l_ng - l_py)))
        else:
            ng_vs_py_med = float("nan")

        # Vb at Vd~2.0V from pyport
        i_top = int(np.argmin(np.abs(Vd_axis - 2.0)))
        Id_top = float(Id_py[i_top])
        print(f"  ngspice_dec={ng_dec:.3f}  pyport_dec={py_dec:.3f}  "
              f"ng↔py={ng_vs_py_med:.3f}  Id_py@2V={Id_top:.2e}  Id_ng@2V≈{Id_ng[np.argmin(np.abs(Vd_ng-2.0))]:.2e}")

        out["biases"].append({
            "VG1": vg1, "VG2": vg2,
            "ngspice_dec_vs_data": ng_dec, "pyport_dec_vs_data": py_dec,
            "ngspice_vs_pyport_med_dec": ng_vs_py_med,
            "Id_pyport_at_Vd2V": Id_top,
            "Id_ngspice_at_Vd2V": float(Id_ng[np.argmin(np.abs(Vd_ng-2.0))]),
        })
        with open(OUT / "ngspice_xval.json", "w") as f:
            json.dump(out, f, indent=2, default=str)

    # Aggregate
    deltas = [b["ngspice_vs_pyport_med_dec"] for b in out["biases"]
              if isinstance(b.get("ngspice_vs_pyport_med_dec"), float)
              and np.isfinite(b["ngspice_vs_pyport_med_dec"])]
    if deltas:
        out["aggregate"] = {
            "n": len(deltas),
            "mean_dec": float(np.mean(deltas)),
            "median_dec": float(np.median(deltas)),
            "max_dec": float(np.max(deltas)),
            "PASS": bool(np.mean(deltas) <= 0.5),
        }
    with open(OUT / "ngspice_xval.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    return out


def write_verdict(ablation, xval):
    lines = [f"# Newton-root fix verdict ({SEED_MODE}, Hurkx OFF)\n"]
    lines.append(f"Config: newton_root_seed={SEED_MODE}  n_steps={SEED_N_STEPS}  vb_init={SEED_VB_INIT}V\n")
    lines.append(f"Hurkx: DISABLED (no fallback)\n")
    lines.append(f"Canonical baseline (Hurkx=0, no fix): 1.163 dec  |  prior best WITH Hurkx (K1+ALPHA0): 0.665 dec\n")

    # Ablation table
    lines.append("## K1×ALPHA0 ablation (Hurkx OFF, fix ON)\n")
    lines.append("| K1@0.6 | ALPHA0 | median_dec_all | conv | Δ vs 1.163 |")
    lines.append("|---:|---:|---:|---:|---:|")
    best_tag = None; best_med = float("inf")
    for tag, s in ablation.items():
        if "error" in s:
            lines.append(f"| {s['k1_vg1_0p6']:.5f} | {s['alpha0']:.4e} | ERROR | | |")
            continue
        m = s["median_dec_all"]["median"]
        d = m - BASELINE_MEDIAN_DEC
        lines.append(f"| {s['k1_vg1_0p6']:.5f} | {s['alpha0']:.4e} | {m:.3f} | {s['convergence_rate']:.2f} | {d:+.3f} |")
        if np.isfinite(m) and m < best_med: best_med = m; best_tag = tag

    lines.append(f"\n**Best in grid**: {best_tag} → {best_med:.3f} dec\n")
    if best_med <= 0.5:
        lines.append("- Target ≤0.5 dec REACHED without Hurkx ✓\n")
    else:
        lines.append(f"- Target ≤0.5 dec NOT reached (gap = {best_med-0.5:+.3f} dec)\n")

    # Ngspice xval
    lines.append("\n## ngspice xval (9-bias, fix ON, Hurkx OFF)\n")
    lines.append("| VG1 | VG2 | ngspice dec vs data | pyport dec vs data | ng↔py med | Id_py@2V | Id_ng@2V |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for b in xval["biases"]:
        if "ngspice_vs_pyport_med_dec" not in b:
            lines.append(f"| {b['VG1']} | {b['VG2']} | SKIP | | | | |"); continue
        lines.append(f"| {b['VG1']} | {b['VG2']} | {b['ngspice_dec_vs_data']:.3f} | "
                     f"{b['pyport_dec_vs_data']:.3f} | {b['ngspice_vs_pyport_med_dec']:.3f} | "
                     f"{b['Id_pyport_at_Vd2V']:.2e} | {b['Id_ngspice_at_Vd2V']:.2e} |")
    if "aggregate" in xval:
        a = xval["aggregate"]
        lines.append(f"\n- mean ng↔py = {a['mean_dec']:.3f} dec  (PASS threshold 0.5)")
        lines.append(f"- median = {a['median_dec']:.3f}  max = {a['max_dec']:.3f}")
        verdict_str = "PASS" if a["PASS"] else "FAIL"
        lines.append(f"- **VERDICT: {verdict_str}**\n")

    (OUT / "verdict.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT/'verdict.md'}")


def main():
    print(f"=== track_newton_root_fix ({SEED_MODE}, Hurkx OFF) ===")
    print(f"  n_steps={SEED_N_STEPS}  vb_init={SEED_VB_INIT}\n")

    print("Phase A: K1×ALPHA0 ablation (33-bias × 2 sweeps × 6 cells)")
    t0 = time.time()
    ablation = run_combo_grid()
    print(f"\nPhase A done in {time.time()-t0:.1f}s\n")

    print("Phase B: 9-bias ngspice xval")
    xval = run_ngspice_xval()

    write_verdict(ablation, xval)


if __name__ == "__main__":
    main()
