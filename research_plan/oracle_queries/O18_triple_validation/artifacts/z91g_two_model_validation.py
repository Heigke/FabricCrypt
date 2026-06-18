"""z91g — true two-card validation.

Builds on z91f. After the P2.2 refactor (forward_2t now accepts model_M1
and model_M2 as separate BSIM4Model instances), we can finally run the
M1 card on M1 and the M2 card on M2 — fixing the silent coherence break
where compute_dc(model, sd_M2, …) was reading M1's k3, lpe0, dvt0, kt1,
kt1l, kt2, etc. while computing M2.

Same .param post-load patch as z91f (vth0n=0.54153, vsatn=102230,
lpe0n=1.2439e-7, …) — the SPICE parser still misses + continuation lines
on .param directives, so the post-load fixup remains necessary.
"""
from __future__ import annotations
import json, math, os, re, csv, time
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/sebas_2026_04_22"
_out_suffix = os.environ.get("NSRAM_OUT_SUFFIX", "")
OUT = ROOT / f"results/z91g_two_model_validation{_out_suffix}"
OUT.mkdir(parents=True, exist_ok=True)

from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.model_card import BSIM4Model, parse_param_blocks
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig, forward_2t
from nsram.bsim4_port.arclength import forward_2t_arclength_grad
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.geometry import Geometry


# Reuse z91f's data + helper layer
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "z91f_mod", ROOT / "scripts/z91f_validate_with_sebas_params.py")
z91f = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(z91f)
load_curves = z91f.load_curves
load_sebas_params = z91f.load_sebas_params
find_params = z91f.find_params
patch_model_values = z91f.patch_model_values
patch_sd_scaled = z91f.patch_sd_scaled
make_overrides = z91f.make_overrides
make_bjt = z91f.make_bjt


def main():
    t0 = time.time()
    print(f"[z91g] starting at {time.strftime('%H:%M:%S')}", flush=True)

    # Load M1 and M2 cards as DISTINCT BSIM4Model instances. Apply the
    # .param post-load patch to each (parser drops + continuation lines on
    # .param blocks).
    # Stage 4 (2026-05-03): NSRAM_DISABLE_PATCH=1 skips patch_model_values to
    # verify the 1.00-dec match holds with faithful ngspice-equivalent parsing
    # after the model_card.py .param parser fix (log entry 12:42).
    # Stage 5 (2026-05-03 13:08): M1 card references symbols (vth0n, lintn,
    # lpe0n, etc.) that are defined only in M2's .param block — cross-file
    # scope. Pre-extract M2's .params and seed BOTH cards' from_spice calls,
    # mirroring ngspice's deck-wide .param scope.
    _disable_patch = bool(int(os.environ.get("NSRAM_DISABLE_PATCH", "0")))

    text_M1 = (DATA / "M1_130DNWFB.txt").read_text()
    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    shared_params = parse_param_blocks(text_M2)
    if _disable_patch:
        print(f"[z91g] cross-file .params from M2: {len(shared_params)} defs "
              f"(vth0n={shared_params.get('vth0n')}, "
              f"lintn={shared_params.get('lintn')}, "
              f"lpe0n={shared_params.get('lpe0n')})", flush=True)

    model_M1 = BSIM4Model.from_spice(text_M1, model_type="nmos",
                                       params=shared_params)
    if not _disable_patch:
        patch_model_values(model_M1, type_n=True)
    else:
        print("[z91g] NSRAM_DISABLE_PATCH=1: skipping patch_model_values(M1)", flush=True)
    # A.5.l (2026-05-02): M1 voff shift via env var (mirrors A.5.k for M2).
    _voff_m1_shift = float(os.environ.get("NSRAM_VOFF_M1_SHIFT", "0.0"))
    if _voff_m1_shift != 0.0:
        old = model_M1._values.get("voff", -0.1368)
        model_M1._values["voff"] = old + _voff_m1_shift
        print(f"[z91g] M1 voff shift: {old} -> {model_M1._values['voff']} (Δ={_voff_m1_shift:+.3f}V)", flush=True)
    print(f"[z91g] M1 card loaded; vth0={model_M1.get('vth0')} "
          f"vsat={model_M1.get('vsat')} k1={model_M1.get('k1')} "
          f"etab={model_M1.get('etab')} beta0={model_M1.get('beta0')}",
          flush=True)

    model_M2 = BSIM4Model.from_spice(text_M2, model_type="nmos",
                                       params=shared_params)
    if not _disable_patch:
        patch_model_values(model_M2, type_n=True)
    else:
        print("[z91g] NSRAM_DISABLE_PATCH=1: skipping patch_model_values(M2)", flush=True)
    # A.5.k (2026-05-02): apply NSRAM_VOFF_M2_SHIFT BEFORE compute_size_dep
    # so the shift propagates into sd_M2.voffcbn (which is cached at temp-time
    # and ignores post-hoc patch_sd_scaled overrides). Per A.5.j, the per-bias
    # P_M2["voff"] override path is plumbing-broken for voffcbn.
    _voff_shift = float(os.environ.get("NSRAM_VOFF_M2_SHIFT", "0.0"))
    if _voff_shift != 0.0:
        old = model_M2._values.get("voff", -0.1368)
        model_M2._values["voff"] = old + _voff_shift
        print(f"[z91g] M2 voff shift: {old} -> {model_M2._values['voff']} (Δ={_voff_shift:+.3f}V)", flush=True)
    print(f"[z91g] M2 card loaded; vth0={model_M2.get('vth0')} "
          f"vsat={model_M2.get('vsat')} k1={model_M2.get('k1')} "
          f"etab={model_M2.get('etab')} beta0={model_M2.get('beta0')}",
          flush=True)

    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                             newton_max_iters=50)
    vnwell_rs_override = os.environ.get("NSRAM_VNWELL_RS")
    if vnwell_rs_override is not None:
        cfg.vnwell_Rs = float(vnwell_rs_override)
        print(f"[z91g] vnwell_Rs override = {cfg.vnwell_Rs:g}", flush=True)
    m1_d_override = os.environ.get("NSRAM_M1_DIODE_SCALE")
    if m1_d_override is not None:
        cfg.m1_diode_scale = float(m1_d_override)
        print(f"[z91g] m1_diode_scale override = {cfg.m1_diode_scale}", flush=True)
    pdi_to = os.environ.get("NSRAM_PDI_TO")
    if pdi_to is not None:
        cfg.body_pdiode_to = pdi_to
        for k in ("AREA", "JS", "N", "VJ", "M"):
            v = os.environ.get(f"NSRAM_PDI_{k}")
            if v is not None:
                setattr(cfg, f"body_pdiode_{k.lower()}", float(v))
        print(f"[z91g] body_pdiode_to={cfg.body_pdiode_to} area={cfg.body_pdiode_area:g} "
              f"Js={cfg.body_pdiode_Js:g} n={cfg.body_pdiode_n}", flush=True)
    sd_M1 = compute_size_dep(model_M1, Geometry(L=cfg.Ln, W=cfg.Wn),
                              T_C=cfg.T_C)
    sd_M2 = compute_size_dep(model_M2,
                              Geometry(L=cfg.Ln * cfg.M2_length_factor,
                                       W=cfg.Wn), T_C=cfg.T_C)
    cfg._sd_M1 = sd_M1
    cfg._sd_M2 = sd_M2

    curves = load_curves()
    sebas_rows = load_sebas_params()
    print(f"[z91g] {len(curves)} measured curves, {len(sebas_rows)} CSV rows",
          flush=True)

    log_eps = 1e-15
    results = []
    for c in curves:
        sebas_row = find_params(sebas_rows, c["VG1"], c["VG2"])
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            results.append({"VG1": c["VG1"], "VG2": c["VG2"],
                            "skipped": True, "reason": "NaN row"})
            continue
        P_M1, P_M2 = make_overrides(sebas_row)
        if os.environ.get("NSRAM_DISABLE_OVERRIDES", "0") == "1":
            P_M1 = None
            P_M2 = None
        # The static M2_STATIC_OVERRIDES inside z91f.make_overrides puts
        # k1/etab/beta0 baselines in P_M2; with the proper M2 card now
        # loaded those baselines are already in sd_M2. Drop them so we
        # only override what the CSV says (NFACTOR).
        if P_M2:
            for k in ("k1", "k2", "etab", "beta0"):
                P_M2.pop(k, None)
            if not P_M2:
                P_M2 = None
        bjt = make_bjt(sebas_row)
        # z91h grid-search optimum (revisited post-A.1.s): Bf=5e4 + α0×10
        # gives lowest RMSE; previously these cut coverage 25→19 but the
        # robust arclength solver (A.1.s, tighter corrector tol + branch
        # detection) now keeps full coverage at these settings.
        bjt.Bf = float(os.environ.get("NSRAM_BJT_BF", "5.0e4"))
        # A.5.l: extra knobs to disambiguate M1/BJT/M2 contributions
        _bf_mult = float(os.environ.get("NSRAM_BJT_BF_MULT", "1.0"))
        _area_mult = float(os.environ.get("NSRAM_BJT_AREA_MULT", "1.0"))
        if _bf_mult != 1.0:
            bjt.Bf = bjt.Bf * _bf_mult
        if _area_mult != 1.0:
            bjt.area = bjt.area * _area_mult
        # Per-bias mbjt scales BOTH the BJT (already in make_bjt) AND the
        # well-body diode (cfg.vnwell_mbjt). At VG1=0.2 mbjt=0.001 → both
        # parasitic paths off; at VG1=0.4/0.6 mbjt=1 → fully on.
        mbjt = float(sebas_row.get("mbjt", 1.0))
        if math.isnan(mbjt):
            mbjt = 1.0
        cfg.vnwell_mbjt = mbjt
        # α0 multiplier — z91h grid found ×10 best at smooth-ramp regime,
        # but user feedback says shape is too smooth. Try ×100 to push
        # feedback loop gain higher and see if knee sharpens (env override).
        if P_M1 is None:
            P_M1 = {}
        a0_csv = sebas_row.get("ALPHA0", 7.842e-5)
        if not math.isnan(a0_csv):
            a0_mult = float(os.environ.get("NSRAM_A0_MULT", "10.0"))
            P_M1["alpha0"] = torch.tensor(a0_mult * a0_csv, dtype=torch.float64)
        # GPT-5 / O2 oracle injection-limited hypothesis test (A.1.q).
        # NSRAM_BETA0_TEST > 0 overrides M1 and M2 beta0 in compute_iimpact
        # to test if smaller β0 lights the body. Sebas's CSV says β0≈18-20;
        # if exp(-β0/Δ) at Δ≈0.27V is the killer, β0=1.5 → exp(-5.5)=0.004
        # vs current exp(-74)=e-32. Decisive single-variable experiment.
        BETA0_TEST = float(os.environ.get("NSRAM_BETA0_TEST", "0"))
        if BETA0_TEST > 0:
            if P_M1 is None:
                P_M1 = {}
            if P_M2 is None:
                P_M2 = {}
            P_M1["beta0"] = torch.tensor(BETA0_TEST, dtype=torch.float64)
            P_M2["beta0"] = torch.tensor(BETA0_TEST, dtype=torch.float64)
        # A.5.k: NSRAM_VOFF_M2_SHIFT now applied at model-load time, BEFORE
        # compute_size_dep. The per-bias P_M2["voff"] override path was
        # broken (didn't update voffcbn). See A.5.j log entry.
        try:
            with torch.no_grad(), \
                 patch_sd_scaled(sd_M1, P_M1), \
                 patch_sd_scaled(sd_M2, P_M2):
                out = forward_2t_arclength_grad(
                    cfg, model_M1=model_M1, model_M2=model_M2,
                    bjt=bjt, Vd_seq=c["Vd"],
                    VG1=torch.tensor(c["VG1"]),
                    VG2=torch.tensor(c["VG2"]))
            Id_pred = out["Id"].abs()
            conv = torch.tensor([bool(x) for x in out["converged"]])
        except Exception as e:
            results.append({"VG1": c["VG1"], "VG2": c["VG2"],
                            "skipped": True, "reason": f"forward error: {e}"})
            continue

        log_p = torch.log10(Id_pred + log_eps)
        log_m = torch.log10(c["Id"] + log_eps)
        if conv.any():
            sq = (log_p - log_m) ** 2
            rmse = float(torch.sqrt(sq[conv].mean()))
        else:
            rmse = float("inf")
        results.append({"VG1": c["VG1"], "VG2": c["VG2"], "skipped": False,
                        "log_rmse": rmse,
                        "n_converged": int(conv.sum()),
                        "n_total": int(len(conv)),
                        "Vd": c["Vd"].numpy().tolist(),
                        "Id_meas": c["Id"].numpy().tolist(),
                        "Id_pred": Id_pred.numpy().tolist(),
                        "converged": conv.numpy().tolist()})
        print(f"  VG1={c['VG1']:.2f} VG2={c['VG2']:+.2f}: "
              f"log_rmse={rmse:.3f}  conv={int(conv.sum())}/{len(conv)}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    rmses = [r["log_rmse"] for r in results
             if not r.get("skipped") and math.isfinite(r["log_rmse"])]
    median_rmse = float(np.median(rmses)) if rmses else float("inf")
    p90_rmse = float(np.percentile(rmses, 90)) if rmses else float("inf")

    summary = {
        "n_curves": len(curves),
        "n_evaluated": len(rmses),
        "n_skipped": sum(1 for r in results if r.get("skipped")),
        "median_log_rmse": median_rmse,
        "p90_log_rmse": p90_rmse,
        "elapsed_s": time.time() - t0,
        "vs_z91f_run1_median": 4.234,
        "vs_z91f_run2_median": 2.402,
        "note": "true two-model validation (M1 = 130DNWFB, M2 = 130bulkNSRAM)"
                " with Sebastian's per-bias CSV overrides",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "predictions.json").write_text(json.dumps(results, indent=2))
    print(f"\n[z91g] median log-RMSE = {median_rmse:.3f}  "
          f"p90 = {p90_rmse:.3f}  (z91f run2: median=2.40, p90=4.83)",
          flush=True)

    # Plot grid
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, vg1 in zip(axes, [0.2, 0.4, 0.6]):
        sel = [r for r in results
               if not r.get("skipped") and abs(r["VG1"] - vg1) < 1e-3]
        sel.sort(key=lambda r: r["VG2"])
        cmap = plt.cm.viridis(np.linspace(0, 1, max(len(sel), 1)))
        for color, r in zip(cmap, sel):
            Vd = np.array(r["Vd"])
            Im = np.array(r["Id_meas"])
            Ip = np.array(r["Id_pred"])
            cm = np.array(r["converged"])
            ax.semilogy(Vd, Im, "o", ms=3, color=color, alpha=0.5)
            Ip_plot = np.where(cm, Ip, np.nan)
            ax.semilogy(Vd, Ip_plot, "-", lw=1.0, color=color)
        ax.set_title(f"VG1 = {vg1} V")
        ax.set_xlabel("Vd [V]")
        ax.set_ylim(1e-13, 1e-3)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("|Id| [A]")
    fig.suptitle(
        f"z91g two-model validation — M1 = 130DNWFB, M2 = 130bulkNSRAM\n"
        f"o = measurement, line = prediction · "
        f"median log-RMSE = {median_rmse:.3f}  p90 = {p90_rmse:.3f}  "
        f"(z91f single-card: 2.40 / 4.83)",
        fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fit_vs_meas.png", dpi=140)
    plt.close(fig)
    print(f"[z91g] saved {OUT}/fit_vs_meas.png", flush=True)


if __name__ == "__main__":
    main()
