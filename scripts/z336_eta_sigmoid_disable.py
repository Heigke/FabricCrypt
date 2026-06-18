"""z336 — R-20 two-pronged audit.

(A) Disable the iii_gain sigmoid (the `sigmoid(10*(Vd-1))` gate that
    kills Iii pre-knee) and re-run the 33-bias Sebas IV. Compare cell-wide
    median dec to z334 baseline (7.05 dec).

(B) BJT Q1 ground-truth OP audit. At ngspice OP (Vsint=0.382, Vb=0.267,
    Vd=2.0), compute pyport Q1 Ic and Ib in BOTH topologies:
      - default (emitter=Sint, collector=Drain) → LTSpice .asc wire trace
      - bjt_emitter_to_gnd=True (emitter=GND,  collector=Drain) → R-13 patch
    Compare to ngspice ground-truth: Ic=3.73e-11, Ib=3.58e-15, β≈10000.

Writes:
  results/z336_bjt_audit/bjt_terms.json
  results/z336_bjt_audit/eta_sigmoid_disable.json
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
import sys, json, re, time, importlib.util
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))

OUT = ROOT / "results/z336_bjt_audit"
OUT.mkdir(parents=True, exist_ok=True)

VG1_GT = 0.6
VG2_GT = 0.20
VD_GT  = 2.0
VSINT_GT = 0.382
VB_GT    = 0.267
DATA = ROOT / "data/sebas_2026_04_22"

NGSPICE_REF = {
    "Ic_at_q1": 3.73e-11,
    "Ib_at_q1": 3.58e-15,
    "beta":     1e4,
    "Vsint":    VSINT_GT,
    "Vb":       VB_GT,
    "Vd":       VD_GT,
    "VG1":      VG1_GT,
    "VG2":      VG2_GT,
    "topology": "Q1 vsint vb 0 (collector=Sint, base=Vb, emitter=GND)",
}


def build_pyport(eta_disable: bool = False):
    """Mirrors z334 build, with optional iii_gain sigmoid kill."""
    sp = importlib.util.spec_from_file_location(
        "v1", ROOT / "scripts/z96_narma10_pilot.py")
    v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
    from nsram.bsim4_port.bjt import GummelPoonNPN
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                             newton_max_iters=20)
    if eta_disable:
        # iii_gain = eta_max * sigmoid(eta_slope * (Vds - eta_vds_th))
        # set threshold far negative => sigmoid≈1.0 for all Vd ≥ 0 =>
        # iii_gain ≈ eta_max = 1.0 (full Iii flows to Vb).
        cfg.eta_vds_th = -100.0
        cfg.eta_slope  = 10.0
        cfg.eta_max    = 1.0
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = 9000.0; bjt.Va = 0.55; bjt.Is = 1e-9
    return cfg, M1, M2, bjt


# ──────────────────────────────────────────────────────────────────────
# (B) BJT audit at GT OP
# ──────────────────────────────────────────────────────────────────────
def bjt_audit():
    from nsram.bsim4_port.bjt import compute_bjt, GummelPoonNPN
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = 9000.0; bjt.Va = 0.55; bjt.Is = 1e-9

    Vsint = torch.tensor(VSINT_GT, dtype=torch.float64)
    Vb    = torch.tensor(VB_GT,    dtype=torch.float64)
    Vd    = torch.tensor(VD_GT,    dtype=torch.float64)

    out = {
        "ngspice_ref": NGSPICE_REF,
        "bjt_params": {
            "Is": bjt.Is, "Bf": bjt.Bf, "Va": bjt.Va, "Br": bjt.Br,
            "Nf": bjt.Nf, "Nr": bjt.Nr, "Ne": bjt.Ne, "Nc": bjt.Nc,
            "Ikr": bjt.Ikr, "Ikf": bjt.Ikf, "area": bjt.area,
        },
        "topologies": {},
    }

    # Topology A: pyport default (LTSpice .asc):  collector=D, base=B, emitter=Sint
    # → Vbe = Vb - Vsint,   Vbc = Vb - Vd
    Vbe_A = Vb - Vsint
    Vbc_A = Vb - Vd
    bjt_A = compute_bjt(bjt, Vbe=Vbe_A, Vbc=Vbc_A, T_K=300.15)
    out["topologies"]["A_default_LTSpice_emit=Sint_coll=D"] = {
        "Vbe": float(Vbe_A), "Vbc": float(Vbc_A),
        "Ic":  float(bjt_A["Ic"]),
        "Ib":  float(bjt_A["Ib"]),
        "Ie":  float(bjt_A["Ie"]),
        "Icc": float(bjt_A["Icc"]),
        "Iec": float(bjt_A["Iec"]),
        "kqb": float(bjt_A["kqb"]),
        "beta_eff": (float(bjt_A["Ic"]) / float(bjt_A["Ib"]))
                    if abs(float(bjt_A["Ib"])) > 1e-30 else None,
        "wiring_in_code": "lines 532-534 nsram_cell_2T.py",
    }

    # Topology B: R-13 patch (z330 ngspice deck): collector=D, base=B, emitter=GND
    # → Vbe = Vb,            Vbc = Vb - Vd
    # NOTE this is NOT identical to the ngspice deck which has
    # collector=Sint, but it's what cfg.bjt_emitter_to_gnd=True does.
    Vbe_B = Vb
    Vbc_B = Vb - Vd
    bjt_B = compute_bjt(bjt, Vbe=Vbe_B, Vbc=Vbc_B, T_K=300.15)
    out["topologies"]["B_bjt_emitter_to_gnd_coll=D"] = {
        "Vbe": float(Vbe_B), "Vbc": float(Vbc_B),
        "Ic":  float(bjt_B["Ic"]),
        "Ib":  float(bjt_B["Ib"]),
        "beta_eff": (float(bjt_B["Ic"]) / float(bjt_B["Ib"]))
                    if abs(float(bjt_B["Ib"])) > 1e-30 else None,
        "wiring_in_code": "lines 529-531 nsram_cell_2T.py",
    }

    # Topology C: ngspice deck verbatim:  collector=Sint, base=Vb, emitter=GND
    # → Vbe = Vb - 0 = Vb,   Vbc = Vb - Vsint
    Vbe_C = Vb
    Vbc_C = Vb - Vsint
    bjt_C = compute_bjt(bjt, Vbe=Vbe_C, Vbc=Vbc_C, T_K=300.15)
    out["topologies"]["C_ngspice_deck_coll=Sint_emit=GND"] = {
        "Vbe": float(Vbe_C), "Vbc": float(Vbc_C),
        "Ic":  float(bjt_C["Ic"]),
        "Ib":  float(bjt_C["Ib"]),
        "beta_eff": (float(bjt_C["Ic"]) / float(bjt_C["Ib"]))
                    if abs(float(bjt_C["Ib"])) > 1e-30 else None,
        "wiring_in_code": "NOT IMPLEMENTED in pyport",
    }

    # Comparison to ngspice ref
    for key, top in out["topologies"].items():
        Ic = top["Ic"]; Ib = top["Ib"]
        top["Ic_ratio_to_ngspice"] = Ic / NGSPICE_REF["Ic_at_q1"]
        top["Ib_ratio_to_ngspice"] = Ib / NGSPICE_REF["Ib_at_q1"]
        top["Ic_dec_gap"] = (np.log10(abs(Ic) + 1e-40)
                              - np.log10(abs(NGSPICE_REF["Ic_at_q1"])))

    return out


# ──────────────────────────────────────────────────────────────────────
# (A) eta_sigmoid disable test — re-run alpha0=baseline w/ iii_gain forced 1
# ──────────────────────────────────────────────────────────────────────
def load_sebas_curves():
    curves = []
    for d in sorted(DATA.glob("2vHCa-2 I-Vs@VG2 VG1=*")):
        mv = re.search(r"VG1=([\d.]+)", d.name)
        if not mv: continue
        VG1 = float(mv.group(1))
        for f in sorted(d.glob("*.csv")):
            mg = re.search(r"VG2=(-?\d+\.\d+)", f.name)
            if not mg: continue
            VG2 = float(mg.group(1))
            data = np.loadtxt(f, delimiter=",", skiprows=1, usecols=(0, 1))
            if data.ndim == 1: continue
            half = len(data) // 2
            Vd = data[:half, 0]
            Id = np.abs(data[:half, 1])
            mask = (Vd >= 0.05) & (Vd <= 2.0)
            Vd, Id = Vd[mask], Id[mask]
            if len(Vd) < 10: continue
            idx = np.linspace(0, len(Vd) - 1, 20).astype(int)
            curves.append({"VG1": VG1, "VG2": VG2,
                            "Vd": Vd[idx], "Id_meas": Id[idx]})
    return curves


def solve_one(cfg, M1, M2, bjt, Vd, VG1, VG2, Vb_fixed=0.267):
    from nsram.bsim4_port.nsram_cell_2T import _residuals
    Vd_t = torch.tensor(float(Vd), dtype=torch.float64)
    VG1_t = torch.tensor(float(VG1), dtype=torch.float64)
    VG2_t = torch.tensor(float(VG2), dtype=torch.float64)
    Vb_t  = torch.tensor(float(Vb_fixed), dtype=torch.float64)
    Vsint = (0.5 * Vd_t).clone()
    for it in range(cfg.newton_max_iters):
        R_S, R_B, comp = _residuals(cfg, M1, bjt, Vd_t, VG1_t, VG2_t,
                                     Vsint, Vb_t, model_M2=M2)
        if abs(float(R_S)) < 1e-12:
            break
        h = 1e-6
        R_Sp, _, _ = _residuals(cfg, M1, bjt, Vd_t, VG1_t, VG2_t,
                                  Vsint + h, Vb_t, model_M2=M2)
        dRdV = (R_Sp - R_S) / h
        if abs(float(dRdV)) < 1e-30:
            break
        dV = -R_S / dRdV
        dV = torch.clamp(dV, -0.5, 0.5)
        Vsint = Vsint + dV
    R_S, R_B, comp = _residuals(cfg, M1, bjt, Vd_t, VG1_t, VG2_t,
                                 Vsint, Vb_t, model_M2=M2)
    Id = (comp["Ids_M1"] + comp["Ic_Q1"]
          + comp.get("Ic_lat", 0.0) + comp.get("Ic_avalanche", 0.0)
          + comp["Igidl_M1"] - comp["Ibd_M1"])
    return float(abs(float(Id)))


def eta_disable_test():
    cfg, M1, M2, bjt = build_pyport(eta_disable=True)
    curves = load_sebas_curves()
    print(f"[z336-A] {len(curves)} curves, iii_gain forced ~1.0")
    log_eps = 1e-15

    # Diagnostic: confirm iii_gain ~ 1
    Vds_eff = VD_GT
    iii_gain_check = float(torch.sigmoid(torch.tensor(
        cfg.eta_slope * (Vds_eff - cfg.eta_vds_th))))
    print(f"[z336-A] iii_gain(Vd=2.0) = {iii_gain_check:.6f}")
    iii_gain_05 = float(torch.sigmoid(torch.tensor(
        cfg.eta_slope * (0.5 - cfg.eta_vds_th))))
    print(f"[z336-A] iii_gain(Vd=0.5) = {iii_gain_05:.6f}")

    per_curve = []
    pernc_VG1 = {}
    slope_signs = {}
    t0 = time.time()
    for c in curves:
        Id_pred = np.array([solve_one(cfg, M1, M2, bjt, vd, c["VG1"], c["VG2"])
                             for vd in c["Vd"]])
        Id_meas = c["Id_meas"]
        rmse = float(np.sqrt(np.mean(
            (np.log10(Id_pred + log_eps) -
             np.log10(Id_meas + log_eps))**2)))
        per_curve.append({"VG1": c["VG1"], "VG2": c["VG2"],
                           "log_rmse_dec": rmse})
        pernc_VG1.setdefault(c["VG1"], []).append(rmse)
        # Pre-knee slope at all VG1 rows
        mask = (c["Vd"] >= 0.05) & (c["Vd"] <= 0.5)
        if mask.sum() >= 3:
            slope_pred = np.polyfit(c["Vd"][mask],
                                      np.log10(Id_pred[mask]+log_eps), 1)[0]
            slope_meas = np.polyfit(c["Vd"][mask],
                                      np.log10(Id_meas[mask]+log_eps), 1)[0]
            slope_signs[f"VG1={c['VG1']:.2f}_VG2={c['VG2']:+.2f}"] = {
                "pred": float(slope_pred),
                "meas": float(slope_meas),
                "sign_match": bool(np.sign(slope_pred) == np.sign(slope_meas)),
            }
    arr = np.array([r["log_rmse_dec"] for r in per_curve
                     if np.isfinite(r["log_rmse_dec"])])
    median_dec = float(np.median(arr)) if len(arr) else float("inf")
    per_VG1 = {f"VG1={k:.2f}": float(np.median(v))
                for k, v in sorted(pernc_VG1.items())}
    sign_match_count = sum(1 for v in slope_signs.values() if v["sign_match"])

    return {
        "cfg": {
            "eta_vds_th": cfg.eta_vds_th,
            "eta_slope":  cfg.eta_slope,
            "eta_max":    cfg.eta_max,
            "iii_gain_at_Vd2.0": iii_gain_check,
            "iii_gain_at_Vd0.5": iii_gain_05,
        },
        "n_curves": len(per_curve),
        "median_log_rmse_dec_CELL": median_dec,
        "baseline_z334_median_dec": 7.05,
        "delta_vs_baseline": median_dec - 7.05,
        "per_VG1_median_dec": per_VG1,
        "preknee_slope_sign_match_count": f"{sign_match_count}/{len(slope_signs)}",
        "preknee_slope_signs": slope_signs,
        "elapsed_s": time.time() - t0,
    }


def main():
    print("=== z336 R-20 BJT audit + eta_sigmoid disable ===")

    # (B) BJT audit
    print("[z336-B] BJT Q1 audit at GT OP ...")
    audit = bjt_audit()
    (OUT / "bjt_terms.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))

    # (A) eta_sigmoid disable
    print("\n[z336-A] running 33-bias re-fit with iii_gain disabled ...")
    eta = eta_disable_test()
    (OUT / "eta_sigmoid_disable.json").write_text(json.dumps(eta, indent=2))
    print(json.dumps({k: v for k, v in eta.items()
                       if k != "preknee_slope_signs"}, indent=2))

    print(f"\n[z336] wrote {OUT}/bjt_terms.json")
    print(f"[z336] wrote {OUT}/eta_sigmoid_disable.json")


if __name__ == "__main__":
    main()
