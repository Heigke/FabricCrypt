"""z328 — R-12 solver fix: Vsint_init = 0 (NOT Vb_init).

R-10 (z326) set Vb_init = Vd - 0.7 but kept Vsint_init = 0.5*Vd (default).
Solver STILL converged to trivial basin (Vb~1.999, Vsint~1.867).
R-11 (z327) confirmed BSIM4 IIMOD math correct: at Vsint=0 we get
Iii_M1 = 1.18e-11 A (14 OoM > gate). So real fix is to force Newton
to *start* in the strong-saturation basin: Vsint=0 → Vgs_M1=0.6 →
M1 strong sat → Iii fires → body charges → NPN turns on → solver
locks into physical equilibrium.

Strategies (try in order):
  V1 — hard initial guess: Vsint_init = 0.0
  V2 — joint arclength (Vd, Vsint): ramp Vd from 0.1 to target with
        Vsint warm-carried (but explicit, not 0.5*Vd default)
  V3 — penalty-augmented residual on early iterations to force
        Vsint < 0.5*Vd in basin of attraction.

LOCKED gates (carry from R-10):
  INFRA  : Vsint < Vd at strong bias (VG1=0.6, Vd=2.0)
  INFRA-2: Iii_M1 > 1e-25 A at converged state (M1 ON)
  PASS   : cell-wide median < 0.95 dec
  AMBITIOUS: < 0.5 dec
  DIAGNOSTIC: VG1=0.6 < 0.7 dec
"""
from __future__ import annotations
import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import importlib.util
import json
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
SCRIPTS = ROOT / "scripts"
OUT_DIR = ROOT / "results/z328_solver_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PARTIAL = OUT_DIR / "partial.json"
SUMMARY = OUT_DIR / "summary.json"
PLOT_DIR = OUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "nsram"))

DEVICE = torch.device("cpu")  # solver is CPU-bound float64
DTYPE = torch.float64

ALPHA0_CONST = 7.842e-5
PDIODE_AREA = 2.2e-11
PDIODE_N = 1.0535
R_BODY_TABLE = {0.2: 1.0e10, 0.4: 1.0e9, 0.6: 1.0e8}
TAT_JTSS = 3.4e-7
TAT_NJTS = 20.0
TAT_VTSS = 10.0
TAT_XTSS = 0.02
LAT_BV_DISABLED = 1.0e6
BF_CARD = 10000.0


def _load_module(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


def configure_v5b_postfix(cfg, vg1):
    cfg.use_well_diode = False
    cfg.body_pdiode_to = "off"
    if hasattr(cfg, "z310_enable_vnwell_diode"):
        cfg.z310_enable_vnwell_diode = False
    cfg.body_pdiode_area = PDIODE_AREA
    cfg.body_pdiode_n = PDIODE_N
    cfg.Cbody = 7e-15
    cfg.body_pdiode_Rs = float(R_BODY_TABLE.get(round(vg1, 2), 1.0e9))
    cfg.vnwell_Rs = 1.0e30
    cfg.use_lateral_collector = False
    cfg.lat_BV = LAT_BV_DISABLED
    cfg.lat_BV_max = LAT_BV_DISABLED * 1.1
    cfg.enable_tat = True
    cfg.tat_jtss = TAT_JTSS
    cfg.tat_njts = TAT_NJTS
    cfg.tat_vtss = TAT_VTSS
    cfg.tat_xtss = TAT_XTSS
    if hasattr(cfg, "z313_enable_tat"):
        cfg.z313_enable_tat = False
    if hasattr(cfg, "invalidate"):
        cfg.invalidate()


@contextmanager
def patch_sd_scaled(sd, overrides):
    if not overrides:
        yield
        return
    saved = {}
    try:
        for k, v in overrides.items():
            saved[k] = sd.scaled.get(k, None)
            sd.scaled[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sd.scaled.pop(k, None)
            else:
                sd.scaled[k] = v


# ---------------------------------------------------------------------------
# Strategy V1: Vsint_init = 0.0 (hard).
# ---------------------------------------------------------------------------

def forward_2t_V1(cfg, model_M1, model_M2, bjt, Vd_seq, VG1, VG2):
    from nsram.bsim4_port.nsram_cell_2T import solve_2t_with_homotopy
    Vd_seq = Vd_seq.to(DTYPE)
    VG1 = torch.as_tensor(VG1, dtype=DTYPE)
    VG2 = torch.as_tensor(VG2, dtype=DTYPE)
    T = int(Vd_seq.shape[0])
    Ids, Vss, Vbs, convs = [], [], [], []
    for i in range(T):
        Vd_i = Vd_seq[i].unsqueeze(0)
        Vsint_init = torch.tensor([0.0], dtype=DTYPE)
        Vb_init = torch.tensor([0.0], dtype=DTYPE)
        out = solve_2t_with_homotopy(
            cfg, model_M1, bjt,
            Vd=Vd_i, VG1=VG1, VG2=VG2,
            P_M1=None, P_M2=None,
            Vsint_init=Vsint_init.expand_as(Vd_i),
            Vb_init=Vb_init.expand_as(Vd_i),
            model_M2=model_M2,
        )
        Ids.append(out["Id"].squeeze(0))
        Vss.append(out["Vsint"].squeeze(0))
        Vbs.append(out["Vb"].squeeze(0))
        convs.append(bool(torch.as_tensor(out["converged"]).all()))
    return {"Id": torch.stack(Ids), "Vsint": torch.stack(Vss),
            "Vb": torch.stack(Vbs), "converged": convs}


# ---------------------------------------------------------------------------
# Strategy V2: joint (Vd, Vsint) arclength continuation.
# Start at Vd=0.1 with Vsint=0, Vb=0; for each target Vd_target, ramp Vd
# upward in N steps; carry Vsint/Vb across.  Solver re-solves at each
# intermediate Vd which (combined with V1's Vsint=0 cold start at point 0)
# keeps us inside the physical basin.
# ---------------------------------------------------------------------------

def forward_2t_V3(cfg, model_M1, model_M2, bjt, Vd_seq, VG1, VG2):
    """V3: bypass gmin homotopy. Call solve_2t_steady_state DIRECTLY at the
    target gmin (=cfg.gmin), with Vsint_init=0. The gmin homotopy schedule
    [1e-3, 1e-5, ...] at gmin=1e-3 adds 2*gmin*(Vb-Vsint) to the Sint residual
    which strongly couples Vsint to Vb — that's the path that drags the solver
    into the trivial Vsint~Vd/2 basin. Skipping homotopy keeps the basin
    structure clean.
    """
    from nsram.bsim4_port.nsram_cell_2T import solve_2t_steady_state
    Vd_seq = Vd_seq.to(DTYPE)
    VG1 = torch.as_tensor(VG1, dtype=DTYPE)
    VG2 = torch.as_tensor(VG2, dtype=DTYPE)
    T = int(Vd_seq.shape[0])
    Ids, Vss, Vbs, convs = [], [], [], []
    for i in range(T):
        Vd_i = Vd_seq[i].unsqueeze(0)
        Vsint_init = torch.tensor([0.0], dtype=DTYPE)
        Vb_init = torch.tensor([0.0], dtype=DTYPE)
        out = solve_2t_steady_state(
            cfg, model_M1, bjt,
            Vd=Vd_i, VG1=VG1, VG2=VG2,
            P_M1=None, P_M2=None,
            Vsint_init=Vsint_init.expand_as(Vd_i),
            Vb_init=Vb_init.expand_as(Vd_i),
            model_M2=model_M2,
        )
        Ids.append(out["Id"].squeeze(0))
        Vss.append(out["Vsint"].squeeze(0))
        Vbs.append(out["Vb"].squeeze(0))
        convs.append(bool(torch.as_tensor(out["converged"]).all()))
    return {"Id": torch.stack(Ids), "Vsint": torch.stack(Vss),
            "Vb": torch.stack(Vbs), "converged": convs}


def forward_2t_V2(cfg, model_M1, model_M2, bjt, Vd_seq, VG1, VG2,
                   n_homotopy=8):
    from nsram.bsim4_port.nsram_cell_2T import solve_2t_with_homotopy
    Vd_seq = Vd_seq.to(DTYPE)
    VG1 = torch.as_tensor(VG1, dtype=DTYPE)
    VG2 = torch.as_tensor(VG2, dtype=DTYPE)
    T = int(Vd_seq.shape[0])
    Ids, Vss, Vbs, convs = [], [], [], []
    Vsint_warm = torch.tensor([0.0], dtype=DTYPE)
    Vb_warm = torch.tensor([0.0], dtype=DTYPE)
    Vd_cur = 0.1
    for i in range(T):
        Vd_target = float(Vd_seq[i].item())
        if i == 0:
            ramp = np.linspace(Vd_cur, Vd_target, n_homotopy + 1)[1:]
        else:
            prev = float(Vd_seq[i-1].item())
            ramp = np.linspace(prev, Vd_target, max(2, n_homotopy // 2) + 1)[1:]
        for vd_step in ramp:
            Vd_t = torch.tensor([vd_step], dtype=DTYPE)
            out = solve_2t_with_homotopy(
                cfg, model_M1, bjt,
                Vd=Vd_t, VG1=VG1, VG2=VG2,
                P_M1=None, P_M2=None,
                Vsint_init=Vsint_warm.expand_as(Vd_t),
                Vb_init=Vb_warm.expand_as(Vd_t),
                model_M2=model_M2,
            )
            Vsint_warm = out["Vsint"].detach().squeeze(0).unsqueeze(0)
            Vb_warm = out["Vb"].detach().squeeze(0).unsqueeze(0)
        Ids.append(out["Id"].squeeze(0))
        Vss.append(out["Vsint"].squeeze(0))
        Vbs.append(out["Vb"].squeeze(0))
        convs.append(bool(torch.as_tensor(out["converged"]).all()))
    return {"Id": torch.stack(Ids), "Vsint": torch.stack(Vss),
            "Vb": torch.stack(Vbs), "converged": convs}


# ---------------------------------------------------------------------------
# Diagnostic: at converged (Vsint, Vb), recompute Iii_M1.
# ---------------------------------------------------------------------------

def _diag_iii_at(cfg, model_M1, sd_M1, Vd, VG1, Vsint, Vb, P_M1):
    """Return Iii_M1 at the converged state."""
    from nsram.bsim4_port.nsram_cell_2T import _residuals
    from nsram.bsim4_port.bjt import GummelPoonNPN
    dummy_bjt = GummelPoonNPN.from_sebas_card()
    dummy_bjt.Bf = BF_CARD
    Vd_t = torch.as_tensor([Vd], dtype=DTYPE)
    VG1_t = torch.as_tensor(VG1, dtype=DTYPE).expand_as(Vd_t)
    Vsint_t = torch.as_tensor([Vsint], dtype=DTYPE)
    Vb_t = torch.as_tensor([Vb], dtype=DTYPE)
    VG2_t = torch.as_tensor(0.20, dtype=DTYPE).expand_as(Vd_t)
    with torch.no_grad(), patch_sd_scaled(sd_M1, P_M1 or {}):
        _, _, comp = _residuals(cfg, model_M1, dummy_bjt, Vd_t, VG1_t,
                                  VG2_t, Vsint_t, Vb_t, None, None)
    return float(comp.get("Iii_M1", torch.tensor(0.0)).abs().item())


# ---------------------------------------------------------------------------
# Single-bias test
# ---------------------------------------------------------------------------

def test_single_bias(strategy, cfg, M1, M2, sd_M1, sd_M2, sebas_rows, z91f,
                     disable_tat=False, body_pdiode_gnd=False,
                     body_pdiode_Rs=None):
    from nsram.bsim4_port.bjt import GummelPoonNPN
    VG1 = 0.6; VG2 = 0.20; Vd_val = 2.0
    configure_v5b_postfix(cfg, VG1)
    if disable_tat:
        cfg.enable_tat = False
    if body_pdiode_gnd:
        cfg.body_pdiode_to = "gnd"
    if body_pdiode_Rs is not None:
        cfg.body_pdiode_Rs = float(body_pdiode_Rs)
    if hasattr(cfg, "invalidate"): cfg.invalidate()
    sebas_row = next((r for r in sebas_rows
                      if abs(r["VG1"] - VG1) < 1e-3 and abs(r["VG2"] - VG2) < 1e-3),
                     None)
    P_M1 = {}
    if not math.isnan(sebas_row.get("ETAB", float("nan"))):
        P_M1["etab"] = torch.tensor(sebas_row["ETAB"], dtype=DTYPE)
    if not math.isnan(sebas_row.get("K1", float("nan"))):
        P_M1["k1"] = torch.tensor(sebas_row["K1"], dtype=DTYPE)
    P_M1["alpha0"] = torch.tensor(ALPHA0_CONST, dtype=DTYPE)
    if not math.isnan(sebas_row.get("BETA0", float("nan"))):
        P_M1["beta0"] = torch.tensor(sebas_row["BETA0"], dtype=DTYPE)
    P_M2 = {}
    if not math.isnan(sebas_row.get("NFACTOR", float("nan"))):
        P_M2["nfactor"] = torch.tensor(sebas_row["NFACTOR"], dtype=DTYPE)
    for k, v in z91f.M2_STATIC_OVERRIDES.items():
        if k not in P_M2:
            P_M2[k] = torch.tensor(float(v), dtype=DTYPE)
    bjt = GummelPoonNPN.from_sebas_card()
    if not math.isnan(sebas_row.get("IS", float("nan"))):
        bjt.Is = float(sebas_row["IS"])
    area = float(sebas_row.get("area", 1e-6))
    if math.isnan(area): area = 1e-6
    mbjt = float(sebas_row.get("mbjt", 1.0))
    if math.isnan(mbjt): mbjt = 1.0
    bjt.area = area * mbjt
    bjt.Bf = BF_CARD

    Vd_seq = torch.tensor([Vd_val], dtype=DTYPE)
    with torch.no_grad(), patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
        if strategy == "V1":
            out = forward_2t_V1(cfg, M1, M2, bjt, Vd_seq, VG1, VG2)
        elif strategy == "V2":
            out = forward_2t_V2(cfg, M1, M2, bjt, Vd_seq, VG1, VG2)
        elif strategy == "V3":
            out = forward_2t_V3(cfg, M1, M2, bjt, Vd_seq, VG1, VG2)
        else:
            raise ValueError(strategy)
        Iii = _diag_iii_at(cfg, M1, sd_M1, Vd_val, VG1,
                            float(out["Vsint"].squeeze()),
                            float(out["Vb"].squeeze()), P_M1)

    Vb = float(out["Vb"].squeeze()); Vsint = float(out["Vsint"].squeeze())
    Id = float(out["Id"].abs().squeeze()); conv = bool(out["converged"][0])
    infra_pass = (Vsint < Vd_val - 1e-4) and conv
    infra2_pass = Iii > 1e-25
    return {"strategy": strategy, "Vb": Vb, "Vsint": Vsint, "Id_A": Id,
            "Vd": Vd_val, "converged": conv, "Iii_M1_A": Iii,
            "infra_pass": infra_pass, "infra2_pass": infra2_pass}


# ---------------------------------------------------------------------------
# Full Sebas IV
# ---------------------------------------------------------------------------

def evaluate_full_iv(strategy, cfg, M1, M2, sd_M1, sd_M2, sebas_rows, curves,
                     z91f, z304, disable_tat=False, body_pdiode_gnd=False,
                     body_pdiode_Rs=None):
    from nsram.bsim4_port.bjt import GummelPoonNPN
    log_eps = 1e-15
    per_curve = []
    # Per VG1 group: store predicted curves for plotting
    plot_data = {0.2: [], 0.4: [], 0.6: []}
    for c in curves:
        VG1 = float(c["VG1"]); VG2 = float(c["VG2"])
        sebas_row = next((r for r in sebas_rows
                          if abs(r["VG1"]-VG1) < 1e-3 and abs(r["VG2"]-VG2) < 1e-3),
                         None)
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            continue
        configure_v5b_postfix(cfg, VG1)
        if disable_tat:
            cfg.enable_tat = False
        if body_pdiode_gnd:
            cfg.body_pdiode_to = "gnd"
        if body_pdiode_Rs is not None:
            cfg.body_pdiode_Rs = float(body_pdiode_Rs)
        if hasattr(cfg, "invalidate"): cfg.invalidate()
        P_M1, P_M2 = z304.make_row_overrides(sebas_row, ALPHA0_CONST,
                                              z91f.M2_STATIC_OVERRIDES)
        bjt = GummelPoonNPN.from_sebas_card()
        if not math.isnan(sebas_row.get("IS", float("nan"))):
            bjt.Is = float(sebas_row["IS"])
        area = float(sebas_row.get("area", 1e-6))
        if math.isnan(area): area = 1e-6
        mbjt = float(sebas_row.get("mbjt", 1.0))
        if math.isnan(mbjt): mbjt = 1.0
        bjt.area = area * mbjt
        bjt.Bf = BF_CARD

        try:
            with torch.no_grad(), patch_sd_scaled(sd_M1, P_M1), \
                  patch_sd_scaled(sd_M2, P_M2):
                if strategy == "V1":
                    out = forward_2t_V1(cfg, M1, M2, bjt, c["Vd"], VG1, VG2)
                elif strategy in ("V3", "V4", "V5", "V6"):
                    out = forward_2t_V3(cfg, M1, M2, bjt, c["Vd"], VG1, VG2)
                else:
                    out = forward_2t_V2(cfg, M1, M2, bjt, c["Vd"], VG1, VG2)
            Id_pred = out["Id"].abs()
            conv = torch.tensor([bool(x) for x in out["converged"]])
        except Exception as e:
            per_curve.append({"VG1": VG1, "VG2": VG2,
                              "log_rmse": float("inf"),
                              "signed_dec": float("nan"), "n_conv": 0,
                              "err": str(e)[:120]})
            continue

        log_p = torch.log10(Id_pred + log_eps)
        log_m = torch.log10(c["Id"] + log_eps)
        if conv.any():
            diff = (log_p[conv] - log_m[conv])
            rmse = float(torch.sqrt((diff**2).mean()))
            signed = float(torch.median(diff))
            Vsint_max = float(out["Vsint"].max())
            Vsint_lt_Vd_frac = float(((out["Vsint"] < (c["Vd"] - 1e-3)).float()).mean())
        else:
            rmse = float("inf"); signed = float("nan")
            Vsint_max = float("nan"); Vsint_lt_Vd_frac = float("nan")
        per_curve.append({"VG1": VG1, "VG2": VG2,
                          "log_rmse": rmse, "signed_dec": signed,
                          "n_conv": int(conv.sum()),
                          "Vsint_max": Vsint_max,
                          "Vsint_lt_Vd_frac": Vsint_lt_Vd_frac})
        plot_data.setdefault(VG1, []).append({
            "VG2": VG2, "Vd": c["Vd"].cpu().numpy().tolist(),
            "Id_meas": c["Id"].cpu().numpy().tolist(),
            "Id_pred": Id_pred.cpu().numpy().tolist(),
        })

    finite = [pc for pc in per_curve if math.isfinite(pc["log_rmse"])]
    rmses_all = np.array([pc["log_rmse"] for pc in finite])
    cell_median = float(np.median(rmses_all)) if len(rmses_all) else float("inf")

    per_vg1 = {}
    for vg1 in (0.2, 0.4, 0.6):
        rms = [pc["log_rmse"] for pc in finite if abs(pc["VG1"]-vg1) < 1e-3]
        per_vg1[f"{vg1:.1f}"] = (float(np.median(rms)) if rms else float("inf"))

    return {
        "strategy": strategy,
        "cell_median_log_rmse": cell_median,
        "per_vg1_median": per_vg1,
        "n_finite": len(finite),
        "n_total": len(per_curve),
        "per_curve": per_curve,
        "plot_data": plot_data,
    }


def _make_plot(plot_data, strategy, cell_median):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        return None
    paths = []
    for vg1, curves in plot_data.items():
        if not curves:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        cmap = plt.get_cmap("viridis")
        N = max(1, len(curves))
        for i, c in enumerate(sorted(curves, key=lambda x: x["VG2"])):
            color = cmap(i / N)
            Vd = c["Vd"]; Im = c["Id_meas"]; Ip = c["Id_pred"]
            ax.semilogy(Vd, np.abs(Im), "o", ms=3, color=color, alpha=0.55,
                         label=f"VG2={c['VG2']:.2f} meas")
            ax.semilogy(Vd, np.abs(Ip), "-", color=color, lw=1.4,
                         label=f"VG2={c['VG2']:.2f} pred")
        ax.set_xlabel("V_d (V)")
        ax.set_ylabel("|I_d| (A)")
        ax.set_title(f"z328 [{strategy}] VG1={vg1:.1f} — cell_median={cell_median:.3f} dec")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=6, ncol=2, loc="best")
        fig.tight_layout()
        p = PLOT_DIR / f"iv_vg1_{vg1:.1f}_{strategy}.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        paths.append(str(p))
    return paths


def save_partial(data):
    # strip plot_data (numpy) for partial json
    clean = json.loads(json.dumps(data, default=lambda o: str(o)))
    with open(PARTIAL, "w") as f:
        json.dump(clean, f, indent=2)


def main():
    t0 = time.time()
    print(f"[z328] device={DEVICE}", flush=True)

    z304 = _load_module("z304", SCRIPTS / "z304_sebas_three_branch_refit.py")
    z91f, cfg, M1, M2, sd_M1, sd_M2, _ = z304.build_models_once()
    sebas_rows = z304.load_sebas_params()
    curves = z304.load_curves()
    print(f"[z328] models built; {len(curves)} curves ({time.time()-t0:.1f}s)",
          flush=True)

    summary = {
        "script": "z328_solver_v2",
        "start_time": time.time(),
        "rationale": "Vsint_init=0 (not Vb_init); z327 confirmed Iii_M1=1.18e-11 at Vsint=0",
        "baselines": {"z304": 0.99, "v5b": 3.01, "z326_S3": 3.36},
        "single_bias_tests": {},
        "full_iv_tests": {},
        "gates": {},
        "loc_changes": {
            "file": "scripts/z328_solver_fix_v2.py (NEW)",
            "summary": ("No edits to nsram/bsim4_port — fix is a custom forward "
                          "wrapper that passes Vsint_init=0.0 (not 0.5*Vd default) "
                          "to solve_2t_with_homotopy. V2 adds joint Vd/Vsint "
                          "arclength continuation."),
            "key_lines": "forward_2t_V1 lines 115-141; forward_2t_V2 lines 152-188",
        },
    }
    save_partial(summary)

    # --- Strategy V1 ---
    print("\n[z328] === V1: Vsint_init = 0.0 ===", flush=True)
    v1 = test_single_bias("V1", cfg, M1, M2, sd_M1, sd_M2, sebas_rows, z91f)
    print(f"  V1: Vsint={v1['Vsint']:.4f} Vb={v1['Vb']:.4f} "
          f"Id={v1['Id_A']:.3e} Iii={v1['Iii_M1_A']:.3e} "
          f"conv={v1['converged']} infra={v1['infra_pass']} "
          f"infra2={v1['infra2_pass']}", flush=True)
    summary["single_bias_tests"]["V1"] = v1
    save_partial(summary)

    working = "V1" if (v1["infra_pass"] and v1["infra2_pass"]) else None

    if working is None:
        print("\n[z328] === V2: joint (Vd, Vsint) arclength ===", flush=True)
        v2 = test_single_bias("V2", cfg, M1, M2, sd_M1, sd_M2, sebas_rows, z91f)
        print(f"  V2: Vsint={v2['Vsint']:.4f} Vb={v2['Vb']:.4f} "
              f"Id={v2['Id_A']:.3e} Iii={v2['Iii_M1_A']:.3e} "
              f"conv={v2['converged']} infra={v2['infra_pass']} "
              f"infra2={v2['infra2_pass']}", flush=True)
        summary["single_bias_tests"]["V2"] = v2
        save_partial(summary)
        if v2["infra_pass"] and v2["infra2_pass"]:
            working = "V2"

    if working is None:
        print("\n[z328] === V3: bypass gmin homotopy, Vsint_init=0 ===",
              flush=True)
        v3 = test_single_bias("V3", cfg, M1, M2, sd_M1, sd_M2, sebas_rows, z91f)
        print(f"  V3: Vsint={v3['Vsint']:.4f} Vb={v3['Vb']:.4f} "
              f"Id={v3['Id_A']:.3e} Iii={v3['Iii_M1_A']:.3e} "
              f"conv={v3['converged']} infra={v3['infra_pass']} "
              f"infra2={v3['infra2_pass']}", flush=True)
        summary["single_bias_tests"]["V3"] = v3
        save_partial(summary)
        if v3["infra_pass"] and v3["infra2_pass"]:
            working = "V3"

    # NOTE: residual probing (see partial+log) shows that with v5b's TAT
    # enabled, R_B is positive everywhere at low Vb (~+3e-7 A from
    # tat_jtss=3.4e-7 leakage from D/S junctions PUMPING into body). No
    # physical basin with Vsint<Vd exists in the (Vsint, Vb) plane — TAT
    # alone forces Vb→Vd regardless of Iii. Try V4: disable TAT so the
    # physical Iii-driven equilibrium becomes the only root.

    if working is None:
        print("\n[z328] === V4: V3 + disable TAT (tat_jtss=0) ===", flush=True)
        v4 = test_single_bias("V3", cfg, M1, M2, sd_M1, sd_M2, sebas_rows, z91f,
                              disable_tat=True)
        v4["strategy"] = "V4_no_tat"
        print(f"  V4: Vsint={v4['Vsint']:.4f} Vb={v4['Vb']:.4f} "
              f"Id={v4['Id_A']:.3e} Iii={v4['Iii_M1_A']:.3e} "
              f"conv={v4['converged']} infra={v4['infra_pass']} "
              f"infra2={v4['infra2_pass']}", flush=True)
        summary["single_bias_tests"]["V4"] = v4
        save_partial(summary)
        if v4["infra_pass"] and v4["infra2_pass"]:
            working = "V4"

    # V5: no TAT + body_pdiode_to='gnd' with Rs sweep, clamps Vb in [0, ~Vd]
    if working is None:
        print("\n[z328] === V5: no TAT + body_pdiode_to='gnd' Rs sweep ===",
              flush=True)
        best_v5 = None; best_rs = None
        for rs in [1e8, 1e7, 1e6, 1e5, 1e4]:
            v5 = test_single_bias("V3", cfg, M1, M2, sd_M1, sd_M2, sebas_rows,
                                  z91f, disable_tat=True, body_pdiode_gnd=True,
                                  body_pdiode_Rs=rs)
            v5["strategy"] = f"V5_no_tat_gnd_Rs={rs:.0e}"
            print(f"  V5 Rs={rs:.0e}: Vsint={v5['Vsint']:.4f} Vb={v5['Vb']:.4f} "
                  f"Id={v5['Id_A']:.3e} Iii={v5['Iii_M1_A']:.3e} "
                  f"conv={v5['converged']} infra={v5['infra_pass']} "
                  f"infra2={v5['infra2_pass']}", flush=True)
            summary["single_bias_tests"][f"V5_Rs_{rs:.0e}"] = v5
            save_partial(summary)
            if v5["infra_pass"] and v5["infra2_pass"]:
                if best_v5 is None or v5["Id_A"] > best_v5["Id_A"]:
                    best_v5 = v5; best_rs = rs
        if best_v5 is not None:
            working = "V5"
            summary["V5_best_Rs"] = best_rs
            summary["single_bias_tests"]["V5"] = best_v5
            print(f"[z328] V5 best Rs={best_rs:.0e}", flush=True)

    # V6: KEEP TAT + body_pdiode_to='gnd' + Vsint_init=0 (z326 S3 with V3 init).
    # z326 S3 found this combination converges with Vb<Vd (Vb~1.97 Vsint~1.84)
    # but had Iii_M1~e-48. Try with V3 (no gmin homotopy) and Vsint_init=0 to
    # see if we can stay in the Iii-active sub-basin.
    if working is None:
        print("\n[z328] === V6: keep TAT + body_pdiode_to='gnd' + V3 init ===",
              flush=True)
        best_v6 = None; best_rs = None
        for rs in [1e8, 1e6, 1e4, 1e3, 1e2]:
            v6 = test_single_bias("V3", cfg, M1, M2, sd_M1, sd_M2, sebas_rows,
                                  z91f, disable_tat=False,
                                  body_pdiode_gnd=True, body_pdiode_Rs=rs)
            v6["strategy"] = f"V6_tat_gnd_Rs={rs:.0e}"
            print(f"  V6 Rs={rs:.0e}: Vsint={v6['Vsint']:.4f} Vb={v6['Vb']:.4f} "
                  f"Id={v6['Id_A']:.3e} Iii={v6['Iii_M1_A']:.3e} "
                  f"conv={v6['converged']} infra={v6['infra_pass']} "
                  f"infra2={v6['infra2_pass']}", flush=True)
            summary["single_bias_tests"][f"V6_Rs_{rs:.0e}"] = v6
            save_partial(summary)
            if v6["infra_pass"] and v6["infra2_pass"]:
                if best_v6 is None or v6["Id_A"] > best_v6["Id_A"]:
                    best_v6 = v6; best_rs = rs
        if best_v6 is not None:
            working = "V6"
            summary["V6_best_Rs"] = best_rs
            summary["single_bias_tests"]["V6"] = best_v6
            print(f"[z328] V6 best Rs={best_rs:.0e}", flush=True)

    if working is None:
        print("[z328] V1..V6 ALL failed INFRA — abort before full IV",
              flush=True)
        summary["status"] = "infra_fail"
        summary["elapsed_s"] = time.time() - t0
        with open(SUMMARY, "w") as f:
            json.dump({k: v for k, v in summary.items() if k != "full_iv_tests"},
                      f, indent=2, default=str)
        return

    summary["working_strategy"] = working
    save_partial(summary)

    # --- Full IV with working strategy ---
    print(f"\n[z328] === Full Sebas IV with {working} ===", flush=True)
    t_iv0 = time.time()
    iv_kwargs = {}
    if working == "V4":
        iv_kwargs["disable_tat"] = True
    elif working == "V5":
        iv_kwargs["disable_tat"] = True
        iv_kwargs["body_pdiode_gnd"] = True
        iv_kwargs["body_pdiode_Rs"] = summary.get("V5_best_Rs", 1e6)
    elif working == "V6":
        iv_kwargs["body_pdiode_gnd"] = True
        iv_kwargs["body_pdiode_Rs"] = summary.get("V6_best_Rs", 1e4)
    full = evaluate_full_iv(working, cfg, M1, M2, sd_M1, sd_M2,
                              sebas_rows, curves, z91f, z304, **iv_kwargs)
    print(f"  IV done ({time.time()-t_iv0:.1f}s) cell_median="
          f"{full['cell_median_log_rmse']:.3f} dec per_VG1={full['per_vg1_median']}",
          flush=True)
    summary["full_iv_tests"][working] = {k: v for k, v in full.items()
                                          if k != "plot_data"}
    save_partial(summary)

    # Make plots
    plot_paths = _make_plot(full["plot_data"], working,
                             full["cell_median_log_rmse"])
    summary["plot_paths"] = plot_paths or []
    print(f"[z328] plots saved: {plot_paths}", flush=True)

    # --- Gates ---
    cm = full["cell_median_log_rmse"]
    vg1d = full["per_vg1_median"]
    sb = summary["single_bias_tests"][working]
    gates = {
        "INFRA_Vsint_lt_Vd": bool(sb["infra_pass"]),
        "INFRA2_Iii_M1_gt_1e-25": bool(sb["infra2_pass"]),
        "PASS_lt_0_95": bool(cm < 0.95),
        "AMBITIOUS_lt_0_5": bool(cm < 0.5),
        "DIAGNOSTIC_VG1_0_6_lt_0_7": bool(vg1d.get("0.6", math.inf) < 0.7),
    }
    summary["gates"] = gates
    summary["status"] = "complete"
    summary["elapsed_s"] = time.time() - t0
    print(f"\n[z328] GATES: {json.dumps(gates, indent=2)}", flush=True)

    with open(SUMMARY, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[z328] Wrote {SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
