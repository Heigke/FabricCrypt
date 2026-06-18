"""z353 — R-34: warm-start pyport Newton from ngspice basin per Vd point.

R-33 z352 verdict: pyport's residual landscape HOSTS the right (high-Vb)
basin (R-16) but the solver locks into the low-Vb (subthreshold) basin
(Vb~0.185V vs ngspice 0.27V). T5 V_b-clamp eta_sigmoid did NOT help
(clamp never engages because Vb stays below threshold).

This script bypasses forward_2t's warm-start cascade. For each (VG1,VG2,Vd)
point we call solve_2t_steady_state DIRECTLY with Vsint_init/Vb_init taken
from `results/z340_ngspice_handover/per_bias_states.json` (which contains
ngspice's converged OP per bias). The Newton iteration is then evaluated
at the *real* operating point. If pyport's basin really hosts the
ngspice solution, Newton should stay there.

Pre-registered gates (LOCKED):
  INFRA   : Vb_final within 0.1V of ngspice Vb_init at ≥50% of biases
            (median over Vd per curve, then fraction across 33 curves)
  PASS    : cell-wide median log-RMSE < 1.5 dec
  AMBITIOUS: cell-wide median log-RMSE < 0.95 dec

Configuration (R-29/R-20 already in place):
  - eta_sigmoid = False (T5 falsified by R-33)
  - bjt_emitter_to_gnd = True
  - build_calibrated_models() (R-29 Vth/tox patch)
  - BJT z334 defaults

Output: results/z353_warmstart/summary.json + plots.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))

OUT = ROOT / "results/z353_warmstart"
OUT.mkdir(parents=True, exist_ok=True)

DATA = ROOT / "data/sebas_2026_04_22"
NGSPICE_HANDOVER = ROOT / "results/z340_ngspice_handover/per_bias_states.json"

BJT_BF = 9000.0
BJT_VA = 0.55
BJT_IS = 1.0e-9


def build_pyport():
    sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
    v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
    from nsram.bsim4_port.bjt import GummelPoonNPN

    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                            newton_max_iters=80)
    cfg.bjt_emitter_to_gnd = True
    cfg.eta_sigmoid = False  # R-33 falsified T5 clamp prescription

    M1, M2 = v1.build_calibrated_models()

    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = BJT_BF
    bjt.Va = BJT_VA
    bjt.Is = BJT_IS
    return cfg, M1, M2, bjt


def load_curves():
    curves = []
    for sub in DATA.iterdir():
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
            curves.append({"VG1": vg1, "VG2": vg2, "Vd": d[:,0],
                           "Id": np.abs(d[:,1]), "f": f.name})
    return curves


def load_ngspice_states():
    """Return dict {(VG1,VG2): np.ndarray shape (n_vd, 3) [Vd, Vsint, Vb]}."""
    raw = json.loads(NGSPICE_HANDOVER.read_text())
    out = {}
    for item in raw:
        key = (round(float(item["VG1"]), 3), round(float(item["VG2"]), 3))
        arr = np.array(item["states"], dtype=np.float64)
        out[key] = arr
    return out


def warm_init_for_vd(ng_states: np.ndarray, vd_targets: np.ndarray):
    """For each requested Vd, return the (Vsint*, Vb*) from ngspice at
    nearest Vd. ng_states[:,0]=Vd, [:,1]=Vsint, [:,2]=Vb.
    """
    vd_ng = ng_states[:, 0]
    Vs_ng = ng_states[:, 1]
    Vb_ng = ng_states[:, 2]
    # ngspice does up-then-down sweep so Vd is non-monotonic. Use nearest.
    Vs_init = np.empty(len(vd_targets), dtype=np.float64)
    Vb_init = np.empty(len(vd_targets), dtype=np.float64)
    for i, v in enumerate(vd_targets):
        j = int(np.argmin(np.abs(vd_ng - v)))
        Vs_init[i] = Vs_ng[j]
        Vb_init[i] = Vb_ng[j]
    return Vs_init, Vb_init


def run_curve(cfg, M1, M2, bjt, curve, ng_states):
    """Solve pyport per Vd point with warm-start from ngspice basin.

    Returns dict with Id_pred, Vsint_final, Vb_final (np arrays), basin_match
    (fraction of Vd points where |Vb_final - Vb_init| < 0.1V), log-RMSE.
    """
    from nsram.bsim4_port.nsram_cell_2T import solve_2t_steady_state

    Vd_arr = np.asarray(curve["Vd"], dtype=np.float64)
    Vs_init_np, Vb_init_np = warm_init_for_vd(ng_states, Vd_arr)

    Id_list = np.empty(len(Vd_arr))
    Vs_final = np.empty(len(Vd_arr))
    Vb_final = np.empty(len(Vd_arr))
    conv_list = np.zeros(len(Vd_arr), dtype=bool)
    niter_list = np.zeros(len(Vd_arr), dtype=int)

    VG1_t = torch.tensor(curve["VG1"], dtype=torch.float64)
    VG2_t = torch.tensor(curve["VG2"], dtype=torch.float64)

    for i, vd in enumerate(Vd_arr):
        Vd_t = torch.tensor([vd], dtype=torch.float64)
        Vs0  = torch.tensor([Vs_init_np[i]], dtype=torch.float64)
        Vb0  = torch.tensor([Vb_init_np[i]], dtype=torch.float64)
        try:
            out = solve_2t_steady_state(
                cfg, M1, bjt,
                Vd=Vd_t, VG1=VG1_t, VG2=VG2_t,
                Vsint_init=Vs0, Vb_init=Vb0,
                model_M2=M2,
            )
            Id_list[i] = float(torch.abs(out["Id"]).item())
            Vs_final[i] = float(out["Vsint"].item())
            Vb_final[i] = float(out["Vb"].item())
            conv_list[i] = bool(out["converged"].all().item())
            niter_list[i] = int(out["niter"])
        except Exception as e:
            Id_list[i] = np.nan
            Vs_final[i] = np.nan
            Vb_final[i] = np.nan
            conv_list[i] = False
            niter_list[i] = -1

    # log-RMSE
    Id_meas = np.abs(curve["Id"])
    mask = (Id_meas > 1e-15) & (Id_list > 1e-15) & np.isfinite(Id_list)
    if mask.sum() < 3:
        rmse = float("nan")
    else:
        logr = np.log10(Id_list[mask]) - np.log10(Id_meas[mask])
        rmse = float(np.sqrt(np.mean(logr ** 2)))

    # basin diagnostic
    valid = np.isfinite(Vb_final) & np.isfinite(Vb_init_np)
    drift = np.abs(Vb_final[valid] - Vb_init_np[valid])
    median_drift = float(np.median(drift)) if drift.size else float("nan")
    frac_close = float(np.mean(drift < 0.10)) if drift.size else float("nan")

    # Vb at flagship-like point (max Vd) — high-Vd, post-snapback regime
    vd_max_idx = int(np.argmax(Vd_arr))
    Vb_at_max = float(Vb_final[vd_max_idx]) if np.isfinite(Vb_final[vd_max_idx]) else None
    Vb_init_at_max = float(Vb_init_np[vd_max_idx])

    return {
        "VG1": curve["VG1"], "VG2": curve["VG2"], "f": curve["f"],
        "log_rmse_dec": rmse,
        "Id_pred": Id_list,
        "Vsint_final": Vs_final, "Vb_final": Vb_final,
        "Vsint_init": Vs_init_np, "Vb_init": Vb_init_np,
        "conv_frac": float(conv_list.mean()),
        "median_Vb_drift": median_drift,
        "frac_Vb_close_0p1": frac_close,
        "Vb_at_max_Vd": Vb_at_max,
        "Vb_init_at_max_Vd": Vb_init_at_max,
        "Vd": Vd_arr,
        "Id_meas": Id_meas,
        "niter_max": int(niter_list.max()) if niter_list.size else 0,
        "niter_mean": float(niter_list[niter_list >= 0].mean()) if (niter_list >= 0).any() else float("nan"),
    }


def probe_residual_at_ng_op(cfg, M1, M2, bjt, ng_states, VG1, VG2, vd_target=2.0):
    """Diagnostic: at flagship-like (vd=2.0V) bias, evaluate pyport
    residual at the ngspice OP. If R_S, R_B << physical currents the
    ngspice OP IS a pyport fixed point. If residuals are large it is NOT.
    """
    from nsram.bsim4_port.nsram_cell_2T import _residuals

    vd_ng = ng_states[:, 0]
    j = int(np.argmin(np.abs(vd_ng - vd_target)))
    Vs0 = ng_states[j, 1]
    Vb0 = ng_states[j, 2]
    vd  = ng_states[j, 0]

    Vd_t  = torch.tensor([vd],  dtype=torch.float64)
    VG1_t = torch.tensor(VG1,   dtype=torch.float64)
    VG2_t = torch.tensor(VG2,   dtype=torch.float64)
    Vs_t  = torch.tensor([Vs0], dtype=torch.float64)
    Vb_t  = torch.tensor([Vb0], dtype=torch.float64)

    R_S, R_B, comp = _residuals(cfg, M1, bjt, Vd_t, VG1_t, VG2_t, Vs_t, Vb_t,
                                None, None, model_M2=M2)
    # Physical scale
    keys = ["Ids_M1", "Ids_M2", "Ic_Q1", "Ib_Q1",
            "Iii_M1", "Iii_M2", "Igidl_M1", "Igidl_M2",
            "Ibs_M1", "Ibd_M1", "Ibs_M2", "Ibd_M2"]
    scale = 0.0
    parts = {}
    for k in keys:
        if k in comp:
            v = float(torch.abs(comp[k]).item())
            scale += v
            parts[k] = v
    R_S_v = float(torch.abs(R_S).item())
    R_B_v = float(torch.abs(R_B).item())
    return {
        "Vd_eval": float(vd), "Vsint_init": float(Vs0), "Vb_init": float(Vb0),
        "R_S": R_S_v, "R_B": R_B_v,
        "I_physical_scale": scale,
        "R_S_rel": R_S_v / max(scale, 1e-30),
        "R_B_rel": R_B_v / max(scale, 1e-30),
        "components": parts,
    }


def main():
    t0 = time.time()
    print("=" * 72, flush=True)
    print("[z353] R-34: warm-start Newton from ngspice basin (per-Vd init)",
          flush=True)
    print(f"[z353] Handover JSON: {NGSPICE_HANDOVER}", flush=True)
    print("=" * 72, flush=True)

    cfg, M1, M2, bjt = build_pyport()
    ng_all = load_ngspice_states()
    print(f"[z353] Loaded ngspice states for {len(ng_all)} biases", flush=True)

    curves = load_curves()
    print(f"[z353] Loaded {len(curves)} Sebas curves", flush=True)

    # Residual probes at high-VG1 (the failing regime)
    probes = []
    for (vg1, vg2) in [(0.6, 0.20), (0.6, 0.00), (0.4, 0.10), (0.2, 0.0)]:
        key = (round(vg1, 3), round(vg2, 3))
        if key in ng_all:
            try:
                p = probe_residual_at_ng_op(cfg, M1, M2, bjt, ng_all[key],
                                            vg1, vg2, vd_target=2.0)
                p["VG1"] = vg1; p["VG2"] = vg2
                probes.append(p)
                print(f"[z353] probe VG1={vg1} VG2={vg2}: "
                      f"R_S={p['R_S']:.3e} R_B={p['R_B']:.3e} "
                      f"I_phys={p['I_physical_scale']:.3e} "
                      f"R_S_rel={p['R_S_rel']:.2e} R_B_rel={p['R_B_rel']:.2e}",
                      flush=True)
            except Exception as e:
                print(f"[z353] probe err {e}", flush=True)

    results = []
    per_vg1 = {}
    for c in curves:
        key = (round(c["VG1"], 3), round(c["VG2"], 3))
        if key not in ng_all:
            print(f"[z353] no ngspice state for {key}, skip", flush=True)
            continue
        r = run_curve(cfg, M1, M2, bjt, c, ng_all[key])
        results.append(r)
        per_vg1.setdefault(c["VG1"], []).append(r)
        print(f"[z353] VG1={c['VG1']:.2f} VG2={c['VG2']:+.2f}: "
              f"dec={r['log_rmse_dec']:.3f}  "
              f"Vb@maxVd: pyport={r['Vb_at_max_Vd']:.3f} ng={r['Vb_init_at_max_Vd']:.3f}  "
              f"drift_med={r['median_Vb_drift']:.3f}  "
              f"frac_close={r['frac_Vb_close_0p1']:.2f}  "
              f"conv={r['conv_frac']:.2f}",
              flush=True)

    # Aggregate
    valid = [r["log_rmse_dec"] for r in results if not np.isnan(r["log_rmse_dec"])]
    cell_med = float(np.median(valid)) if valid else None
    per_vg1_med = {}
    for k, lst in per_vg1.items():
        vv = [r["log_rmse_dec"] for r in lst if not np.isnan(r["log_rmse_dec"])]
        per_vg1_med[f"{k:.2f}"] = float(np.median(vv)) if vv else None

    # Basin agreement: per curve, did MEDIAN Vb-drift stay under 0.1V?
    drift_meds = [r["median_Vb_drift"] for r in results
                  if not np.isnan(r["median_Vb_drift"])]
    frac_curves_basin_match = float(np.mean(np.asarray(drift_meds) < 0.10)) \
        if drift_meds else float("nan")

    gates = {
        "INFRA_n_total": len(results),
        "INFRA_basin_match_frac": frac_curves_basin_match,
        "INFRA_basin_PASS": frac_curves_basin_match >= 0.5,
        "PASS_cell_med_lt_1p5":
            (cell_med is not None) and (cell_med < 1.5),
        "AMBITIOUS_cell_med_lt_0p95":
            (cell_med is not None) and (cell_med < 0.95),
    }

    print("\n[z353] === Aggregate ===", flush=True)
    print(f"  cell_med_dec = {cell_med}", flush=True)
    print(f"  per_VG1_med  = {per_vg1_med}", flush=True)
    print(f"  basin match frac curves = {frac_curves_basin_match}", flush=True)
    print(f"  baselines: z346=4.08  z352=3.93", flush=True)
    print(f"  GATES: {gates}", flush=True)

    # JSON output (strip large arrays)
    def strip(r):
        out = {kk: vv for kk, vv in r.items()
               if kk not in ("Id_pred","Vsint_final","Vb_final",
                             "Vsint_init","Vb_init","Vd","Id_meas")}
        return out
    final = {
        "script": "z353_warmstart_refit",
        "config": {
            "bjt_emitter_to_gnd": True,
            "use_bjt": True,
            "eta_sigmoid": False,
            "M1_card": "ORIGINAL (build_calibrated_models) w/ R-29 Vth/tox patch",
            "warmstart_source": str(NGSPICE_HANDOVER),
            "bjt": {"Bf": BJT_BF, "Va": BJT_VA, "Is": BJT_IS},
            "newton_max_iters": 80,
        },
        "baselines": {
            "z346_cell_med_dec": 4.08,
            "z352_cell_med_dec": 3.93,
        },
        "n_curves": len(results),
        "cell_med_dec": cell_med,
        "per_VG1_med": per_vg1_med,
        "frac_curves_basin_match": frac_curves_basin_match,
        "gates_preregistered": gates,
        "ng_op_residual_probes": probes,
        "per_curve": [strip(r) for r in results],
        "elapsed_s": time.time() - t0,
    }
    (OUT / "summary.json").write_text(json.dumps(final, indent=2, default=str))

    # Plots: one per VG1
    for vg1, lst in per_vg1.items():
        lst.sort(key=lambda r: r["VG2"])
        med = per_vg1_med.get(f"{vg1:.2f}")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        title = f"z353 warm-start VG1={vg1:.2f}"
        if med is not None: title += f"  median dec={med:.2f}"
        fig.suptitle(title)
        cmap = plt.cm.viridis(np.linspace(0, 1, len(lst)))
        ax = axes[0]
        for col, r in zip(cmap, lst):
            ax.semilogy(r["Vd"], np.maximum(r["Id_meas"], 1e-16), color=col, lw=1.5,
                        label=f"VG2={r['VG2']:+.2f}")
            ax.semilogy(r["Vd"], np.maximum(r["Id_pred"], 1e-16),
                        "--", color=col, lw=1.0)
        ax.set_xlabel("Vd"); ax.set_ylabel("|Id|")
        ax.set_title("solid=silicon dashed=pyport(warm)")
        ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7, ncol=2)
        ax = axes[1]
        for col, r in zip(cmap, lst):
            ax.plot(r["Vd"], r["Vb_init"], color=col, lw=1.0, ls=":")
            ax.plot(r["Vd"], r["Vb_final"], color=col, lw=1.0)
        ax.set_xlabel("Vd"); ax.set_ylabel("V_b")
        ax.set_title("solid=pyport-converged  dotted=ngspice-init")
        ax.grid(True, alpha=0.3)
        ax = axes[2]
        vg2s = [r["VG2"] for r in lst]
        rmses = [r["log_rmse_dec"] for r in lst]
        drifts = [r["median_Vb_drift"] for r in lst]
        ax.plot(vg2s, rmses, "o-", label="log RMSE [dec]")
        ax.axhline(1.5, color="g", ls=":", label="PASS")
        ax.axhline(0.95, color="b", ls=":", label="AMBITIOUS")
        ax2 = ax.twinx()
        ax2.plot(vg2s, drifts, "s--", color="r", label="Vb drift [V]")
        ax2.axhline(0.1, color="r", ls=":", alpha=0.4)
        ax2.set_ylabel("median |Vb drift|", color="r")
        ax.set_xlabel("VG2"); ax.set_ylabel("log RMSE [dec]")
        ax.set_title("per-curve")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()
        fig.savefig(OUT / f"plot_VG1_{vg1:.2f}.png", dpi=110)
        plt.close(fig)

    print(f"\n[z353] DONE. Elapsed {time.time()-t0:.1f}s.  "
          f"Output: {OUT/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
