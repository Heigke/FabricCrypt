"""z361 — R-41: pdiode discharge fix + per-bias overrides with NaN-impute.

Apply R-40 three fixes:
  (1) cfg.body_pdiode_to = "vnwell"   (was default "off")
  (2) cfg.use_well_diode = True       (was default False)
  (3) For NaN Sebas rows (VG1=0.4 VG2<=-0.05, VG1=0.6 VG2<=-0.05), impute
      branch-flat ETAB/K1/BETA0 from the lowest-VG2 valid row in each VG1
      branch, and apply via make_overrides per-bias.

All prior fixes locked:
  R-20 BJT Vbc, R-29 lpe0/toxe, R-37 binunit, R-39 BBO best (Bf=991, Va=0.903,
  Is=5.95e-12, iii_body_gain=0.66).

Pre-registered gates (locked):
  INFRA           : 33/33 valid
  PASS            : cell-wide median < 1.5 dec
  AMBITIOUS       : cell-wide median < 0.95 dec
  HIGH_VG1        : VG1=0.60 median < 4.0 dec (vs z360's 5.6)

Probe at flagship (VG1=0.6, VG2=0.20, Vd=2.0): print Vsint, Vb, Id, Iii.
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
import sys, json, re, time, math, csv, importlib.util
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch


@contextmanager
def patch_sd_scaled(sd, overrides):
    """Override sd.scaled[name] entries (mirrors z91f.patch_sd_scaled)."""
    if not overrides:
        yield; return
    saved = {}
    try:
        for k, v in overrides.items():
            saved[k] = sd.scaled.get(k, None)
            sd.scaled[k] = float(v) if hasattr(v, "item") and not torch.is_tensor(v) else (v.item() if torch.is_tensor(v) else float(v))
        yield
    finally:
        for k, v in saved.items():
            if v is None: sd.scaled.pop(k, None)
            else: sd.scaled[k] = v

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "nsram"))
OUT = ROOT / "results/z361_pdiode_fix"; OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data/sebas_2026_04_22"


# --------------------------------------------------------------------------- #
# Sebas CSV loader + per-bias overrides with NaN-impute
# --------------------------------------------------------------------------- #
def load_sebas_params():
    path = DATA / "2Tcell_BSIM_param_DC.csv"
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = {}
            for k, v in r.items():
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = float("nan")
            rows.append(row)
    return rows


# Branch-flat imputation values (R-40 finding 3 / R-41 plan).
# For each VG1 branch, take ETAB/K1/BETA0/NFACTOR from the LOWEST-VG2 valid
# row in that branch and apply them flat across the NaN-row biases.
# VG1=0.4 first valid row VG2=0:    ETAB=1.9, K1=0.53825, BETA0=19, NFACTOR=6
# VG1=0.6 first valid row VG2=0:    ETAB=2.5, K1=0.41825, BETA0=20, NFACTOR=6
BRANCH_FLAT = {
    0.4: {"ETAB": 1.9,  "K1": 0.53825, "ALPHA0": 7.842e-05, "BETA0": 19.0, "NFACTOR": 6.0, "trise": 10.59},
    0.6: {"ETAB": 2.5,  "K1": 0.41825, "ALPHA0": 7.842e-05, "BETA0": 20.0, "NFACTOR": 6.0, "trise": 9.04},
}

# M2 static deltas from Sebas's card (constant across bias, mirrors z91f)
M2_STATIC_OVERRIDES = {
    "k1": 0.63825, "k2": -0.070435, "etab": -0.086777, "beta0": 18.0,
}


def find_or_impute_row(rows, VG1, VG2, atol=1e-3):
    """Find Sebas row; if found but NaN, impute from BRANCH_FLAT."""
    target = None
    for r in rows:
        if abs(r["VG1"] - VG1) < atol and abs(r["VG2"] - VG2) < atol:
            target = dict(r); break
    if target is None:
        return None, False
    # Check if K1 is NaN (proxy for whole row missing)
    if math.isnan(target.get("K1", float("nan"))):
        branch = BRANCH_FLAT.get(round(VG1, 2))
        if branch is None:
            return target, False
        for k, v in branch.items():
            target[k] = float(v)
        return target, True  # imputed
    return target, False


def make_overrides(sebas_row):
    """Returns plain-float override dicts targeting sd.scaled[...]."""
    if sebas_row is None:
        return None, None
    P_M1 = {}
    for csv_k, py_k in (("ETAB", "etab"), ("K1", "k1"),
                       ("ALPHA0", "alpha0"), ("BETA0", "beta0")):
        if not math.isnan(sebas_row.get(csv_k, float("nan"))):
            P_M1[py_k] = float(sebas_row[csv_k])
    P_M2 = {}
    if not math.isnan(sebas_row.get("NFACTOR", float("nan"))):
        P_M2["nfactor"] = float(sebas_row["NFACTOR"])
    for k, v in M2_STATIC_OVERRIDES.items():
        if k not in P_M2:
            P_M2[k] = float(v)
    return (P_M1 or None), (P_M2 or None)


# --------------------------------------------------------------------------- #
# Build models, cfg, bjt
# --------------------------------------------------------------------------- #
def build_pyport():
    sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
    v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
    from nsram.bsim4_port.bjt import GummelPoonNPN
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                            newton_max_iters=40)
    cfg.bjt_emitter_to_gnd = True
    cfg.iii_body_gain = 0.66  # R-39 best

    # R-41 fixes (1) + (2): pdiode discharge path to vnwell, well diode ON.
    cfg.body_pdiode_to = "vnwell"
    cfg.use_well_diode = True
    cfg.vnwell = 2.0
    # Sebas's pdiode card values (data/sebas_2026_04_22/.../pdiode.txt):
    #   is = 5.3675e-7 A (total), n = 1.0535, cj = 7.3279e-4 F/m²
    # Convention: cfg.body_pdiode_Js is per-area, multiplied by body_pdiode_area
    # in _residuals. Sebas's `is` is total → Js_per_area = is / area.
    cfg.body_pdiode_Js = 5.3675e-7 / 22e-12  # 2.44e4 A/m² (Sebas total / area)
    cfg.body_pdiode_n = 1.0535
    # Series-R on pdiode branch: enable a reasonable value rather than 1e10 Ω
    # disabled. Sebas's pdiode card has rs=7.4e-8 Ω (per device); but with the
    # huge Js above we need a meaningful Rs to limit current. Use 1e6 Ω as a
    # gentle limiter (still allows ~µA-mA discharge at Vb-Vnwell ~ -1V).
    cfg.body_pdiode_Rs = 1.0e6

    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = 991.0; bjt.Va = 0.903; bjt.Is = 5.95e-12  # R-39 eval 5 best
    return cfg, M1, M2, bjt


def load_curves():
    curves = []
    for sub in sorted(DATA.iterdir()):
        if not sub.is_dir(): continue
        m_vg1 = re.search(r"VG1=([\d.\-]+)", sub.name)
        if not m_vg1: continue
        vg1 = float(m_vg1.group(1))
        for f in sorted(sub.glob("*.csv")):
            m = re.search(r"VG2=([\-\d.]+)", f.name)
            if not m: continue
            vg2 = float(m.group(1))
            d = np.loadtxt(f, delimiter=",", skiprows=1)
            if d.ndim != 2 or d.shape[1] < 2: continue
            curves.append({"VG1": vg1, "VG2": vg2, "Vd": d[:, 0],
                          "Id": np.abs(d[:, 1]), "f": f.name})
    for f in sorted(DATA.glob("VG1*VG2*.csv")):
        m = re.search(r"VG1=([\d.\-]+)[_ ]*VG2=([\d.\-]+)", f.name)
        if not m: continue
        vg1 = float(m.group(1)); vg2 = float(m.group(2))
        d = np.loadtxt(f, delimiter=",", skiprows=1)
        curves.append({"VG1": vg1, "VG2": vg2, "Vd": d[:, 0],
                      "Id": np.abs(d[:, 1]), "f": f.name})
    return curves


def main():
    t0 = time.time()
    cfg, M1, M2, bjt = build_pyport()
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    print(f"[z361] cfg.body_pdiode_to={cfg.body_pdiode_to}  use_well_diode={cfg.use_well_diode}", flush=True)
    print(f"[z361] cfg.vnwell={cfg.vnwell}  Js={cfg.body_pdiode_Js:.3e}  n={cfg.body_pdiode_n}  Rs={cfg.body_pdiode_Rs:.1e}", flush=True)
    print(f"[z361] cfg.iii_body_gain={cfg.iii_body_gain}  bjt.Bf={bjt.Bf} Va={bjt.Va} Is={bjt.Is:.2e}", flush=True)

    sebas_rows = load_sebas_params()
    curves = load_curves()
    print(f"[z361] loaded {len(curves)} curves, {len(sebas_rows)} Sebas rows", flush=True)

    # Pre-build cached sd_M1 / sd_M2 (cfg caches them so forward_2t reuses).
    sd_M1 = cfg.size_dep_M1(M1)
    sd_M2 = cfg.size_dep_M2(M2)

    # ---- flagship probe (VG1=0.6, VG2=0.20, Vd=2.0) ----
    flagship_probe = None
    try:
        Vd_p = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float64)
        row, imp = find_or_impute_row(sebas_rows, 0.6, 0.20)
        P_M1, P_M2 = make_overrides(row)
        with patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
            out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd_p,
                             VG1=torch.tensor(0.6, dtype=torch.float64),
                             VG2=torch.tensor(0.20, dtype=torch.float64),
                             warm_start=True)
        idx = 3  # Vd=2.0
        comp = out.get("components", {})
        def _g(d, k):
            if k not in d: return None
            v = d[k]
            try:
                arr = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
                if arr.ndim == 0: return float(arr)
                return float(arr.flatten()[idx])
            except Exception:
                return None
        flagship_probe = {
            "VG1": 0.6, "VG2": 0.20, "Vd": 2.0,
            "Vsint": _g(out, "Vsint"), "Vb": _g(out, "Vb"), "Id": _g(out, "Id"),
            "Iii_M1": _g(comp, "Iii_M1"), "Iii": _g(comp, "Iii"),
            "Ic_Q1": _g(out, "Ic_Q1"), "Ib_Q1": _g(out, "Ib_Q1"),
        }
        print(f"[z361] FLAGSHIP probe @ Vd=2.0: Vsint={flagship_probe['Vsint']}  Vb={flagship_probe['Vb']}  Id={flagship_probe['Id']}  Iii_M1={flagship_probe['Iii_M1']}  Ic_Q1={flagship_probe['Ic_Q1']}", flush=True)
    except Exception as e:
        print(f"[z361] flagship probe FAILED: {type(e).__name__}: {e}", flush=True)
        flagship_probe = {"error": str(e)}

    # ---- 33-curve refit ----
    results = []; per_vg1 = {}; n_imputed = 0
    for c in curves:
        Vd = torch.tensor(c["Vd"], dtype=torch.float64)
        row, imp = find_or_impute_row(sebas_rows, c["VG1"], c["VG2"])
        if imp: n_imputed += 1
        P_M1, P_M2 = make_overrides(row)
        try:
            with patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
                out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd,
                                 VG1=torch.tensor(c["VG1"], dtype=torch.float64),
                                 VG2=torch.tensor(c["VG2"], dtype=torch.float64),
                                 warm_start=True)
            Id_pred = np.abs(out["Id"].detach().cpu().numpy())
            mask = (c["Id"] > 1e-15) & (Id_pred > 1e-15) & np.isfinite(Id_pred)
            rmse = float(np.sqrt(np.mean((np.log10(Id_pred[mask]) - np.log10(c["Id"][mask])) ** 2))) if mask.sum() >= 3 else float("nan")
        except Exception as e:
            rmse = float("nan")
            print(f"  err {c['f']}: {type(e).__name__}: {e}", flush=True)
        results.append({"VG1": c["VG1"], "VG2": c["VG2"], "log_rmse_dec": rmse, "imputed": imp})
        per_vg1.setdefault(c["VG1"], []).append(rmse)
        tag = " [IMP]" if imp else ""
        print(f"  VG1={c['VG1']:.2f} VG2={c['VG2']:+.2f}  rmse={rmse:.3f}{tag}", flush=True)

    valid = [r["log_rmse_dec"] for r in results if not math.isnan(r["log_rmse_dec"])]
    per_vg1_median = {f"{k:.2f}": float(np.median([x for x in v if not math.isnan(x)]))
                     for k, v in per_vg1.items()}
    median_cell = float(np.median(valid)) if valid else None
    high_vg1_med = per_vg1_median.get("0.60")

    summary = {
        "script": "z361_pdiode_fix_refit",
        "patches_active": [
            "R-20 BJT Vbc", "R-29 Vth/tox", "R-37 binunit", "R-39 BBO best (eval5)",
            "R-41 body_pdiode_to=vnwell", "R-41 use_well_diode=True",
            "R-41 NaN-row branch-flat impute (8/33)",
        ],
        "params": {
            "Bf": bjt.Bf, "Va": bjt.Va, "Is": bjt.Is, "iii_body_gain": cfg.iii_body_gain,
            "body_pdiode_to": cfg.body_pdiode_to, "use_well_diode": cfg.use_well_diode,
            "vnwell": cfg.vnwell, "body_pdiode_Js": cfg.body_pdiode_Js,
            "body_pdiode_n": cfg.body_pdiode_n, "body_pdiode_Rs": cfg.body_pdiode_Rs,
        },
        "n_curves_valid": len(valid), "n_curves_total": len(results),
        "n_imputed": n_imputed,
        "cell_wide_median_dec": median_cell,
        "per_VG1_median": per_vg1_median,
        "flagship_probe": flagship_probe,
        "baselines": {
            "z358_postR37": 4.28, "z360_R39_best": "see results/z360_final_refit/summary.json",
        },
        "elapsed_s": time.time() - t0,
        "per_curve": results,
    }
    # Gates
    summary["gate_INFRA_33_valid"] = (len(valid) == len(results)) and len(results) == 33
    summary["gate_PASS_lt_1p5"] = (median_cell is not None) and (median_cell < 1.5)
    summary["gate_AMBITIOUS_lt_0p95"] = (median_cell is not None) and (median_cell < 0.95)
    summary["gate_HIGH_VG1_lt_4p0"] = (high_vg1_med is not None) and (high_vg1_med < 4.0)

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- per-VG1 plots ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for vg1 in sorted(per_vg1.keys()):
            fig, ax = plt.subplots(figsize=(6, 4))
            vg2s = []; rmses = []
            for r in results:
                if abs(r["VG1"] - vg1) < 1e-3 and not math.isnan(r["log_rmse_dec"]):
                    vg2s.append(r["VG2"]); rmses.append(r["log_rmse_dec"])
            order = np.argsort(vg2s)
            vg2s = np.array(vg2s)[order]; rmses = np.array(rmses)[order]
            ax.plot(vg2s, rmses, "o-", label=f"VG1={vg1:.2f}")
            ax.axhline(1.5, ls="--", color="orange", label="PASS gate 1.5")
            ax.axhline(0.95, ls="--", color="green", label="AMBITIOUS gate 0.95")
            ax.set_xlabel("VG2 [V]"); ax.set_ylabel("log10 RMSE [dec]")
            ax.set_title(f"z361 pdiode_fix — VG1={vg1:.2f} (median={per_vg1_median[f'{vg1:.2f}']:.3f})")
            ax.legend(); ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(OUT / f"per_VG1_{vg1:.2f}.png", dpi=120)
            plt.close(fig)
        print(f"[z361] saved per-VG1 plots to {OUT}", flush=True)
    except Exception as e:
        print(f"[z361] plot failed: {type(e).__name__}: {e}", flush=True)

    print(f"\n[z361] DONE  median={median_cell}  high_VG1_med={high_vg1_med}", flush=True)
    print(f"[z361] per_VG1={per_vg1_median}", flush=True)
    print(f"[z361] gates: INFRA={summary['gate_INFRA_33_valid']}  PASS={summary['gate_PASS_lt_1p5']}  AMBITIOUS={summary['gate_AMBITIOUS_lt_0p95']}  HIGH_VG1={summary['gate_HIGH_VG1_lt_4p0']}", flush=True)
    print(f"[z361] elapsed {summary['elapsed_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
