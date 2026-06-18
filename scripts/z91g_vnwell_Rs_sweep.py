"""z91g_Rs_sweep — find vnwell_Rs that minimises median log-RMSE.

Quick scan: vnwell_Rs ∈ {2e6, 1e7, 1e8, 1e9, 1e10}, all else as z91g.
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/sebas_2026_04_22"
OUT = ROOT / "results/z91g_vnwell_Rs_sweep"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "z91g_mod", ROOT / "scripts/z91g_two_model_validation.py")
z91g = importlib.util.module_from_spec(_spec)

# We don't run main, just pull helpers
from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.arclength import forward_2t_arclength_grad
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.geometry import Geometry

import importlib.util
_spec2 = importlib.util.spec_from_file_location(
    "z91f_mod", ROOT / "scripts/z91f_validate_with_sebas_params.py")
z91f = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(z91f)
load_curves = z91f.load_curves
load_sebas_params = z91f.load_sebas_params
find_params = z91f.find_params
patch_model_values = z91f.patch_model_values
patch_sd_scaled = z91f.patch_sd_scaled
make_overrides = z91f.make_overrides
make_bjt = z91f.make_bjt


def evaluate(cfg, model_M1, model_M2, sd_M1, sd_M2, curves, sebas_rows):
    log_eps = 1e-15
    rmses = []
    for c in curves:
        sebas_row = find_params(sebas_rows, c["VG1"], c["VG2"])
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            continue
        P_M1, P_M2 = make_overrides(sebas_row)
        if P_M2:
            for k in ("k1", "k2", "etab", "beta0"):
                P_M2.pop(k, None)
            if not P_M2:
                P_M2 = None
        bjt = make_bjt(sebas_row)
        mbjt = float(sebas_row.get("mbjt", 1.0))
        if math.isnan(mbjt):
            mbjt = 1.0
        cfg.vnwell_mbjt = mbjt
        try:
            with torch.no_grad(), patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
                out = forward_2t_arclength_grad(
                    cfg, model_M1=model_M1, model_M2=model_M2, bjt=bjt,
                    Vd_seq=c["Vd"],
                    VG1=torch.tensor(c["VG1"]), VG2=torch.tensor(c["VG2"]))
        except Exception:
            continue
        Id_pred = out["Id"].abs()
        conv = torch.tensor([bool(x) for x in out["converged"]])
        if not conv.any():
            continue
        log_p = torch.log10(Id_pred + log_eps)
        log_m = torch.log10(c["Id"] + log_eps)
        sq = (log_p - log_m) ** 2
        rmses.append(float(torch.sqrt(sq[conv].mean())))
    return float(np.median(rmses)) if rmses else float("inf"), \
           float(np.percentile(rmses, 90)) if rmses else float("inf")


def main():
    t0 = time.time()
    text_M1 = (DATA / "M1_130DNWFB.txt").read_text()
    model_M1 = BSIM4Model.from_spice(text_M1, model_type="nmos")
    patch_model_values(model_M1, type_n=True)
    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    model_M2 = BSIM4Model.from_spice(text_M2, model_type="nmos")
    patch_model_values(model_M2, type_n=True)

    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                             newton_max_iters=50)
    sd_M1 = compute_size_dep(model_M1, Geometry(L=cfg.Ln, W=cfg.Wn), T_C=cfg.T_C)
    sd_M2 = compute_size_dep(model_M2,
                              Geometry(L=cfg.Ln * cfg.M2_length_factor, W=cfg.Wn),
                              T_C=cfg.T_C)
    cfg._sd_M1 = sd_M1
    cfg._sd_M2 = sd_M2

    curves = load_curves()
    sebas_rows = load_sebas_params()

    Rs_grid = [2e6, 1e7, 3e7, 1e8, 3e8, 1e9, 1e10, 1e12]
    results = []
    for Rs in Rs_grid:
        cfg.vnwell_Rs = Rs
        med, p90 = evaluate(cfg, model_M1, model_M2, sd_M1, sd_M2, curves, sebas_rows)
        elapsed = time.time() - t0
        print(f"  Rs = {Rs:.0e}  median = {med:.3f}  p90 = {p90:.3f}  ({elapsed:.0f}s)", flush=True)
        results.append({"Rs": Rs, "median_log_rmse": med, "p90_log_rmse": p90})

    (OUT / "Rs_sweep.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved {OUT}/Rs_sweep.json")
    best = min(results, key=lambda r: r["median_log_rmse"])
    print(f"BEST: Rs = {best['Rs']:.0e}  median = {best['median_log_rmse']:.3f}  p90 = {best['p90_log_rmse']:.3f}")


if __name__ == "__main__":
    main()
