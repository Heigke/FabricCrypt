#!/usr/bin/env python3
"""Track sub-threshold over-conduction in pyport vs ngspice at low Vd.

Workflow:
  1. Load ngspice DC sweeps from results/track_ngspice_xval/decks/*_dc.txt
     (sweep matches: Vd ∈ [0, 2] step 0.05).
  2. For each of 9 (VG1, VG2) biases, run pyport with the SAME config as
     track_ngspice_xval.py (well-diode ON, body_pdiode → vnwell).
  3. Identify the 3 (bias, Vd) points with the LARGEST low-Vd
     (Vd ∈ [0.05, 0.5]) |log10(Id_py/Id_ng)| where Id_py > Id_ng.
  4. Ablate at those 3 points:
        A: well_diode OFF + pdiode OFF
        B: well_diode OFF, pdiode ON
        C: well_diode ON,  pdiode OFF
        baseline: well_diode ON, pdiode ON (current xval config)
  5. Dump JSON + verdict.md ranked suspects.
"""
from __future__ import annotations
import os, sys, json, time, traceback
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

from pathlib import Path
import numpy as np

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results/track_subvt_over_conduction"
OUT.mkdir(parents=True, exist_ok=True)

DECKS = ROOT / "results/track_ngspice_xval/decks"

VG1_GRID = [0.2, 0.4, 0.6]
VG2_GRID = [-0.1, 0.0, 0.1]
VD_LO, VD_HI, VD_STEP = 0.0, 2.0, 0.05
LOW_VD_LO, LOW_VD_HI = 0.05, 0.5  # window for "low Vd" analysis

K1_OVERRIDE = 0.53825
ALPHA0_OVERRIDE = 7.83756e-4


def deck_tag(vg1, vg2):
    return f"VG1={vg1:.2f}_VG2={vg2:.2f}".replace("-", "m").replace(".", "p")


def load_ngspice(vg1, vg2):
    p = DECKS / f"{deck_tag(vg1, vg2)}_dc.txt"
    if not p.exists():
        return None
    d = np.loadtxt(p)
    return d[:, 0], np.abs(d[:, 1]), d[:, 3], d[:, 5]  # Vd, |Id|, Vsint, Vb


# -----------------------------------------------------------------------------
def build_pyport(cfg_flavor: str):
    """flavor in {'baseline','no_pdiode','no_well','no_both'}"""
    import importlib.util
    sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
    v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
    from nsram.bsim4_port.bjt import GummelPoonNPN
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                            newton_max_iters=120)
    cfg.bjt_emitter_to_gnd = True
    cfg.vnwell = 2.0
    cfg.hurkx_bbt_A = 0.0  # no Hurkx for ngspice match
    # body p-diode parameters (same as xval baseline)
    cfg.body_pdiode_Js = 5.3675e-7 / 22e-12
    cfg.body_pdiode_n = 1.0535
    cfg.body_pdiode_Rs = 1.0e6

    if cfg_flavor == "baseline":
        cfg.use_well_diode = True
        cfg.body_pdiode_to = "vnwell"
    elif cfg_flavor == "no_pdiode":
        cfg.use_well_diode = True
        cfg.body_pdiode_to = "off"
    elif cfg_flavor == "no_well":
        cfg.use_well_diode = False
        cfg.body_pdiode_to = "vnwell"
    elif cfg_flavor == "no_both":
        cfg.use_well_diode = False
        cfg.body_pdiode_to = "off"
    else:
        raise ValueError(cfg_flavor)

    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = 9000.0; bjt.Va = 0.55; bjt.Is = 1e-9
    return cfg, M1, M2, bjt


def run_pyport_sweep(cfg, M1, M2, bjt, VG1, VG2, Vd_axis):
    import torch
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    from contextlib import contextmanager
    sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)
    P_M1 = {"alpha0": float(ALPHA0_OVERRIDE)}
    P_M2 = {"alpha0": float(ALPHA0_OVERRIDE)}
    if abs(VG1 - 0.6) < 1e-6:
        P_M1["k1"] = float(K1_OVERRIDE)
    for k, v in {"k1": 0.63825, "k2": -0.070435, "etab": -0.086777, "beta0": 18.0}.items():
        P_M2.setdefault(k, float(v))

    @contextmanager
    def patch(sd, overrides):
        saved = {}
        try:
            for k, v in overrides.items():
                saved[k] = sd.scaled.get(k, None); sd.scaled[k] = float(v)
            yield
        finally:
            for k, v in saved.items():
                if v is None: sd.scaled.pop(k, None)
                else: sd.scaled[k] = v

    Vd_t = torch.tensor(Vd_axis, dtype=torch.float64)
    with patch(sd_M1, P_M1), patch(sd_M2, P_M2):
        out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd_t,
                         VG1=torch.tensor(VG1, dtype=torch.float64),
                         VG2=torch.tensor(VG2, dtype=torch.float64),
                         warm_start=True, multi_init=True,
                         hot_Vsint_init=0.2, hot_Vb_init=0.8)
    Id = np.abs(out["Id"].detach().cpu().numpy()).astype(np.float64)
    Vsint = out.get("Vsint")
    Vb = out.get("Vb")
    Vsint = Vsint.detach().cpu().numpy() if Vsint is not None else np.zeros_like(Id)
    Vb = Vb.detach().cpu().numpy() if Vb is not None else np.zeros_like(Id)
    return Id, Vsint, Vb


def main():
    print("=== track_subvt_over_conduction ===")
    print(f"  low-Vd window: [{LOW_VD_LO}, {LOW_VD_HI}] V")

    Vd_axis = np.arange(VD_LO, VD_HI + 1e-9, VD_STEP)

    # ---- Step 1: load ngspice for all 9 biases ----
    print("[1/4] Loading cached ngspice DC sweeps...")
    ng = {}
    for vg1 in VG1_GRID:
        for vg2 in VG2_GRID:
            res = load_ngspice(vg1, vg2)
            if res is None:
                print(f"  MISS (vg1={vg1} vg2={vg2})"); continue
            ng[(vg1, vg2)] = res
    print(f"  loaded {len(ng)} biases")

    # ---- Step 2: baseline pyport ----
    print("[2/4] Baseline pyport (well_diode ON + pdiode ON)...")
    cfg_b, M1_b, M2_b, bjt_b = build_pyport("baseline")
    py_baseline = {}
    t0 = time.time()
    for (vg1, vg2) in ng.keys():
        try:
            Id_py, Vs_py, Vb_py = run_pyport_sweep(cfg_b, M1_b, M2_b, bjt_b,
                                                    vg1, vg2, Vd_axis)
            py_baseline[(vg1, vg2)] = (Id_py, Vs_py, Vb_py)
        except Exception as e:
            print(f"  FAIL ({vg1},{vg2}): {e}")
            traceback.print_exc()
    print(f"  baseline done {time.time()-t0:.1f}s")

    # ---- Step 3: find worst low-Vd over-conduction points ----
    print("[3/4] Identifying worst low-Vd over-conduction points...")
    rankings = []
    for (vg1, vg2), (Vd_ng, Id_ng, Vsint_ng, Vb_ng) in ng.items():
        if (vg1, vg2) not in py_baseline: continue
        Id_py, Vs_py, Vb_py = py_baseline[(vg1, vg2)]
        # interpolate pyport onto ngspice Vd axis
        Id_py_i = np.interp(Vd_ng, Vd_axis, Id_py)
        Vb_py_i = np.interp(Vd_ng, Vd_axis, Vb_py)
        Vs_py_i = np.interp(Vd_ng, Vd_axis, Vs_py)
        mask = (Vd_ng >= LOW_VD_LO) & (Vd_ng <= LOW_VD_HI)
        for k in np.where(mask)[0]:
            ing = max(Id_ng[k], 1e-30)
            ipy = max(Id_py_i[k], 1e-30)
            dec = np.log10(ipy / ing)
            rankings.append({
                "VG1": vg1, "VG2": vg2, "Vd": float(Vd_ng[k]),
                "Id_ng": float(Id_ng[k]), "Id_py": float(Id_py_i[k]),
                "dec_py_over_ng": float(dec),
                "Vb_ng": float(Vb_ng[k]), "Vb_py": float(Vb_py_i[k]),
                "Vsint_ng": float(Vsint_ng[k]), "Vsint_py": float(Vs_py_i[k]),
            })
    # rank: largest positive (pyport over ngspice)
    rankings.sort(key=lambda r: -r["dec_py_over_ng"])
    top3 = rankings[:3]
    print("  TOP-3 over-conduction points (pyport higher than ngspice):")
    for r in top3:
        print(f"    VG1={r['VG1']} VG2={r['VG2']} Vd={r['Vd']:.2f}V  "
              f"Id_ng={r['Id_ng']:.2e}  Id_py={r['Id_py']:.2e}  "
              f"Δ={r['dec_py_over_ng']:+.2f} dec  "
              f"Vb_ng={r['Vb_ng']:.3f} Vb_py={r['Vb_py']:.3f}")

    # ---- Step 4: ablations at those 3 biases (full Vd sweep, all 4 flavors) ----
    print("[4/4] Ablations on top-3 biases...")
    ablation_biases = sorted({(r["VG1"], r["VG2"]) for r in top3})
    print(f"  unique biases for ablation: {ablation_biases}")
    flavors = ["baseline", "no_pdiode", "no_well", "no_both"]
    ablations = {}
    for flavor in flavors:
        if flavor == "baseline":
            results_flavor = py_baseline
        else:
            print(f"  building {flavor}...")
            cfg_f, M1_f, M2_f, bjt_f = build_pyport(flavor)
            results_flavor = {}
            for (vg1, vg2) in ablation_biases:
                try:
                    Id_py, Vs_py, Vb_py = run_pyport_sweep(cfg_f, M1_f, M2_f, bjt_f,
                                                            vg1, vg2, Vd_axis)
                    results_flavor[(vg1, vg2)] = (Id_py, Vs_py, Vb_py)
                except Exception as e:
                    print(f"    FAIL {flavor} ({vg1},{vg2}): {e}")
        # extract Id at each top3 (bias, Vd)
        for r in top3:
            key = (r["VG1"], r["VG2"])
            if key not in results_flavor: continue
            Id_py, Vs_py, Vb_py = results_flavor[key]
            Id_py_i = float(np.interp(r["Vd"], Vd_axis, Id_py))
            Vb_py_i = float(np.interp(r["Vd"], Vd_axis, Vb_py))
            Vs_py_i = float(np.interp(r["Vd"], Vd_axis, Vs_py))
            key_str = f"VG1={r['VG1']}_VG2={r['VG2']}_Vd={r['Vd']:.2f}"
            ablations.setdefault(key_str, {"ngspice_Id": r["Id_ng"],
                                            "ngspice_Vb": r["Vb_ng"],
                                            "ngspice_Vsint": r["Vsint_ng"]})
            ablations[key_str][flavor] = {
                "Id_py": Id_py_i, "Vb_py": Vb_py_i, "Vsint_py": Vs_py_i,
                "dec_vs_ng": float(np.log10(max(Id_py_i, 1e-30) / max(r["Id_ng"], 1e-30))),
            }

    # Also compute, for each bias, full low-Vd mean |dec| per flavor
    low_vd_mean = {}
    for flavor in flavors:
        if flavor == "baseline":
            results_flavor = py_baseline
        else:
            # rebuild only needed biases (already done above)
            results_flavor = {}
            cfg_f, M1_f, M2_f, bjt_f = build_pyport(flavor)
            for (vg1, vg2) in ablation_biases:
                try:
                    Id_py, Vs_py, Vb_py = run_pyport_sweep(cfg_f, M1_f, M2_f, bjt_f,
                                                            vg1, vg2, Vd_axis)
                    results_flavor[(vg1, vg2)] = (Id_py, Vs_py, Vb_py)
                except Exception:
                    pass
        for (vg1, vg2) in ablation_biases:
            if (vg1, vg2) not in results_flavor: continue
            if (vg1, vg2) not in ng: continue
            Vd_ng, Id_ng, _, _ = ng[(vg1, vg2)]
            Id_py = results_flavor[(vg1, vg2)][0]
            Id_py_i = np.interp(Vd_ng, Vd_axis, Id_py)
            mask = (Vd_ng >= LOW_VD_LO) & (Vd_ng <= LOW_VD_HI)
            if mask.sum() < 2: continue
            decs = np.abs(np.log10(np.clip(Id_py_i[mask], 1e-30, None)
                                   / np.clip(Id_ng[mask], 1e-30, None)))
            key = f"VG1={vg1}_VG2={vg2}"
            low_vd_mean.setdefault(key, {})[flavor] = float(np.mean(decs))

    output = {
        "config": {
            "low_vd_window": [LOW_VD_LO, LOW_VD_HI],
            "K1_override": K1_OVERRIDE, "ALPHA0_override": ALPHA0_OVERRIDE,
            "biases": [[v1, v2] for v1 in VG1_GRID for v2 in VG2_GRID],
        },
        "top3_worst_low_vd_over_conduction": top3,
        "ablations_at_top3": ablations,
        "low_vd_mean_abs_dec_per_bias": low_vd_mean,
        "all_low_vd_points_baseline_sorted_desc_by_over_conduction": rankings[:20],
    }
    out_path = OUT / "ablation.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nwrote {out_path}")

    # ---- verdict.md ----
    lines = ["# track_subvt_over_conduction — verdict\n"]
    lines.append(f"Window: Vd ∈ [{LOW_VD_LO}, {LOW_VD_HI}] V (low-Vd subthreshold regime).\n")
    lines.append("## Top-3 worst low-Vd over-conduction points (baseline pyport vs ngspice)\n")
    for i, r in enumerate(top3, 1):
        lines.append(f"{i}. VG1={r['VG1']}  VG2={r['VG2']}  Vd={r['Vd']:.2f}V")
        lines.append(f"   - Id_ng = {r['Id_ng']:.3e}, Id_py = {r['Id_py']:.3e},  Δ = {r['dec_py_over_ng']:+.2f} dec")
        lines.append(f"   - Vb_ng = {r['Vb_ng']:.3f}, Vb_py = {r['Vb_py']:.3f}  (Δ = {r['Vb_py']-r['Vb_ng']:+.3f} V)")
        lines.append(f"   - Vsint_ng = {r['Vsint_ng']:.3f}, Vsint_py = {r['Vsint_py']:.3f}\n")

    lines.append("## Ablation table (Id_py [A] and Δdec vs ngspice)\n")
    lines.append("| bias | ng Id | baseline | no_pdiode | no_well | no_both |")
    lines.append("|------|------|---------|-----------|---------|---------|")
    for key_str, ab in ablations.items():
        ing = ab["ngspice_Id"]
        row = [key_str, f"{ing:.2e}"]
        for flav in ["baseline", "no_pdiode", "no_well", "no_both"]:
            if flav in ab:
                row.append(f"{ab[flav]['Id_py']:.2e} ({ab[flav]['dec_vs_ng']:+.2f})")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Mean |dec| over low-Vd window per bias and ablation\n")
    lines.append("| bias | baseline | no_pdiode | no_well | no_both |")
    lines.append("|------|----------|-----------|---------|---------|")
    for key, d in low_vd_mean.items():
        row = [key]
        for flav in ["baseline", "no_pdiode", "no_well", "no_both"]:
            row.append(f"{d.get(flav, float('nan')):.3f}" if flav in d else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Ranked suspects (filled by inspection below)\n")
    lines.append("- See ablation table to determine effect of well_diode vs body_pdiode.\n")

    verdict = OUT / "verdict.md"
    verdict.write_text("\n".join(lines))
    print(f"wrote {verdict}")


if __name__ == "__main__":
    main()
