"""Probe v2 — VG1=0.4 V catastrophe single-bias deep dive.

Loads the same two-card setup as z91g, runs the arclength solver on the
worst-fitting bias (VG1=0.4 / VG2=+0.30, log-RMSE 3.25 dec), then walks
the converged path and dumps every component current + the body voltage
vs Vd. Saves a 2x2 diagnostic figure + JSON trace.

Goal: localise WHICH of the seven currents (Ids_M1, Ids_M2, Ic_Q1,
Iii_M1, Iii_M2, Igidl_M1, Igidl_M2) is responsible for the predicted
~7e-6 A "stuck-on" plateau where measurements show 1e-9 → 4e-6 sweep.
"""
from __future__ import annotations
import importlib.util, json, math, os, sys, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/sebas_2026_04_22"
OUT = ROOT / "research_plan/binning_audit/probe_v2_out"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "nsram"))

from nsram.bsim4_port.bjt import GummelPoonNPN  # noqa: E402
from nsram.bsim4_port.model_card import BSIM4Model, parse_param_blocks  # noqa: E402
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig, _residuals  # noqa: E402
from nsram.bsim4_port.arclength import forward_2t_arclength_grad  # noqa: E402
from nsram.bsim4_port.temp import compute_size_dep  # noqa: E402
from nsram.bsim4_port.geometry import Geometry  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "z91f_mod", ROOT / "scripts/z91f_validate_with_sebas_params.py")
z91f = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(z91f)
load_curves = z91f.load_curves
load_sebas_params = z91f.load_sebas_params
find_params = z91f.find_params
patch_model_values = z91f.patch_model_values
patch_sd_scaled = z91f.patch_sd_scaled
make_overrides = z91f.make_overrides
make_bjt = z91f.make_bjt


def main(VG1_target: float = 0.4, VG2_target: float = 0.30):
    t0 = time.time()
    print(f"[probe_v2] target bias: VG1={VG1_target} VG2={VG2_target}")

    text_M1 = (DATA / "M1_130DNWFB.txt").read_text()
    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    shared_params = parse_param_blocks(text_M2)

    model_M1 = BSIM4Model.from_spice(text_M1, model_type="nmos", params=shared_params)
    patch_model_values(model_M1, type_n=True)
    model_M2 = BSIM4Model.from_spice(text_M2, model_type="nmos", params=shared_params)
    patch_model_values(model_M2, type_n=True)

    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                            newton_max_iters=50)
    sd_M1 = compute_size_dep(model_M1, Geometry(L=cfg.Ln, W=cfg.Wn), T_C=cfg.T_C)
    sd_M2 = compute_size_dep(model_M2,
                             Geometry(L=cfg.Ln * cfg.M2_length_factor, W=cfg.Wn),
                             T_C=cfg.T_C)
    cfg._sd_M1 = sd_M1
    cfg._sd_M2 = sd_M2

    # Find the matching curve
    curves = load_curves()
    sebas_rows = load_sebas_params()
    target = None
    for c in curves:
        if abs(c["VG1"] - VG1_target) < 1e-3 and abs(c["VG2"] - VG2_target) < 1e-3:
            target = c
            break
    if target is None:
        raise SystemExit(f"no curve at VG1={VG1_target}/VG2={VG2_target}")

    sebas_row = find_params(sebas_rows, VG1_target, VG2_target)
    P_M1, P_M2 = make_overrides(sebas_row)
    if P_M2:
        for k in ("k1", "k2", "etab", "beta0"):
            P_M2.pop(k, None)
        if not P_M2:
            P_M2 = None
    bjt = make_bjt(sebas_row)
    bjt.Bf = 5.0e4
    mbjt = float(sebas_row.get("mbjt", 1.0))
    if math.isnan(mbjt):
        mbjt = 1.0
    cfg.vnwell_mbjt = mbjt
    if P_M1 is None:
        P_M1 = {}
    a0_csv = sebas_row.get("ALPHA0", 7.842e-5)
    if not math.isnan(a0_csv):
        P_M1["alpha0"] = torch.tensor(10.0 * a0_csv, dtype=torch.float64)

    print(f"[probe_v2] mbjt={mbjt} alpha0_eff={float(P_M1['alpha0']):.3e}")

    Vd_seq = target["Vd"]
    VG1 = torch.tensor(VG1_target)
    VG2 = torch.tensor(VG2_target)

    with torch.no_grad(), \
         patch_sd_scaled(sd_M1, P_M1), \
         patch_sd_scaled(sd_M2, P_M2):
        out = forward_2t_arclength_grad(
            cfg, model_M1=model_M1, model_M2=model_M2,
            bjt=bjt, Vd_seq=Vd_seq, VG1=VG1, VG2=VG2)
        Id_pred = out["Id"].abs().detach().cpu().numpy()
        Vsint = out["Vsint"].detach().cpu().numpy()
        Vb = out["Vb"].detach().cpu().numpy()
        # Re-evaluate components at the converged operating points
        Vd_t = Vd_seq.to(torch.float64)
        Vsint_t = torch.as_tensor(Vsint, dtype=torch.float64)
        Vb_t = torch.as_tensor(Vb, dtype=torch.float64)
        with patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
            # Pass P_M1=None / P_M2=None: the patch_sd_scaled ctx already
            # injected the overrides into sd.scaled[k]. Passing P_M1 again
            # would route through _override_sd which fails on attrs that
            # only live in sd.scaled (e.g. 'etab'). z91g uses the same idiom.
            _, _, comp = _residuals(
                cfg, model_M1, bjt, Vd_t,
                VG1.expand_as(Vd_t), VG2.expand_as(Vd_t),
                Vsint_t, Vb_t, None, None, model_M2=model_M2)

    Vd_np = Vd_seq.numpy()
    Im = target["Id"].numpy()

    keys = ["Ids_M1", "Ids_M2", "Ic_Q1", "Ib_Q1",
            "Iii_M1", "Iii_M2", "Igidl_M1", "Igidl_M2",
            "Ibs_M1", "Ibd_M1", "Ibs_M2", "Ibd_M2"]
    comps = {}
    for k in keys:
        if k in comp:
            comps[k] = comp[k].detach().cpu().numpy()

    # ── plot ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    ax.semilogy(Vd_np, Im, "ko", ms=4, label="measured")
    ax.semilogy(Vd_np, Id_pred, "r-", lw=1.5, label="predicted")
    ax.set_xlabel("Vd [V]"); ax.set_ylabel("|Id| [A]")
    ax.set_title(f"Total drain current  VG1={VG1_target} VG2={VG2_target:+.2f}")
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(1e-13, 1e-3)

    ax = axes[0, 1]
    for k in ["Ids_M1", "Ids_M2", "Ic_Q1", "Ib_Q1"]:
        if k in comps:
            ax.semilogy(Vd_np, np.abs(comps[k]) + 1e-30, lw=1.2, label=k)
    ax.semilogy(Vd_np, Im, "k--", lw=1.0, alpha=0.6, label="measured")
    ax.set_xlabel("Vd [V]"); ax.set_ylabel("|I| [A]")
    ax.set_title("Channel + bipolar components")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(1e-15, 1e-3)

    ax = axes[1, 0]
    for k in ["Iii_M1", "Iii_M2", "Igidl_M1", "Igidl_M2",
              "Ibs_M1", "Ibd_M1", "Ibs_M2", "Ibd_M2"]:
        if k in comps:
            ax.semilogy(Vd_np, np.abs(comps[k]) + 1e-30, lw=1.0, label=k)
    ax.set_xlabel("Vd [V]"); ax.set_ylabel("|I_body| [A]")
    ax.set_title("Body / leakage components")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3); ax.set_ylim(1e-25, 1e-3)

    ax = axes[1, 1]
    ax.plot(Vd_np, Vb, "b-", lw=1.5, label="Vb (body)")
    ax.plot(Vd_np, Vsint, "g-", lw=1.5, label="Vsint (M1.S = M2.D)")
    ax.set_xlabel("Vd [V]"); ax.set_ylabel("Voltage [V]")
    ax.set_title("Internal nodes")
    ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle(
        f"Probe v2 — VG1=0.4/VG2={VG2_target:+.2f} catastrophe diagnostic\n"
        f"meas range {Im.min():.2e}..{Im.max():.2e} · "
        f"pred range {Id_pred.min():.2e}..{Id_pred.max():.2e}",
        fontsize=11, weight="bold")
    fig.tight_layout()
    outpng = OUT / f"vg1_{VG1_target:.2f}_vg2_{VG2_target:+.2f}.png"
    fig.savefig(outpng, dpi=140); plt.close(fig)
    print(f"[probe_v2] saved {outpng}")

    # JSON dump of the trace
    trace = {
        "VG1": VG1_target, "VG2": VG2_target,
        "Vd": Vd_np.tolist(),
        "Id_meas": Im.tolist(),
        "Id_pred": Id_pred.tolist(),
        "Vsint": Vsint.tolist(),
        "Vb": Vb.tolist(),
        "components": {k: v.tolist() for k, v in comps.items()},
        "elapsed_s": time.time() - t0,
    }
    outjson = OUT / f"vg1_{VG1_target:.2f}_vg2_{VG2_target:+.2f}.json"
    outjson.write_text(json.dumps(trace, indent=2))
    print(f"[probe_v2] saved {outjson} ({time.time()-t0:.1f}s)")

    # Quick textual summary — which component dominates the predicted Id?
    idx_lo = 0
    idx_hi = len(Vd_np) - 1
    print("\n[probe_v2] Component breakdown (low Vd → high Vd):")
    print(f"  Vd[{idx_lo}]={Vd_np[idx_lo]:.3f}  Vb={Vb[idx_lo]:+.4f}  Vsint={Vsint[idx_lo]:+.4f}")
    for k in keys:
        if k in comps:
            print(f"     {k:>10s} = {comps[k][idx_lo]:+.3e}")
    print(f"  Vd[{idx_hi}]={Vd_np[idx_hi]:.3f}  Vb={Vb[idx_hi]:+.4f}  Vsint={Vsint[idx_hi]:+.4f}")
    for k in keys:
        if k in comps:
            print(f"     {k:>10s} = {comps[k][idx_hi]:+.3e}")
    print(f"\n  measured:   {Im[idx_lo]:.3e} → {Im[idx_hi]:.3e}")
    print(f"  predicted:  {Id_pred[idx_lo]:.3e} → {Id_pred[idx_hi]:.3e}")


if __name__ == "__main__":
    vg1 = float(os.environ.get("PROBE_VG1", "0.4"))
    vg2 = float(os.environ.get("PROBE_VG2", "0.30"))
    main(vg1, vg2)
