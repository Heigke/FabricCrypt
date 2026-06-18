"""z359 — Recalibrate BJT params against the (now correct) Iii.

Context (R-37→R-38):
- R-37 fixed binunit override; Iii is now ~ngspice (gap closed -3.43→-0.52 dec).
- R-38 (z358) cell-wide median=4.28 dec, VG1=0.20 improved 0.27, VG1=0.40/0.60
  WORSE by ~0.4/0.2 dec. Diagnosis: BJT params (Bf=9000, Va=0.55, Is=1e-9) were
  tuned to BROKEN weak Iii; with correct Iii (1000× larger) BJT over-pumps at
  high VG1.

R-39 plan:
- Small BBO (scipy.optimize.differential_evolution) over BJT.Bf, Va, Is and
  cfg.iii_body_gain ∈ [0.01, 1.0] (legacy multiplier, < 1 attenuates).
- Inner cost: cell-wide median log-RMSE on a 9-curve subset (3 per VG1).
- Final refit at best params on full 33-curve grid.

Patches active (locked):
- R-20 BJT Vbc=Vb-Vsint (cfg.bjt_emitter_to_gnd=True)
- R-29 lpe0=1.2439e-7, toxe=4e-9
- R-37 binunit override removed
- R-39 trigger relaxation for iii_body_gain != 1.0 (in nsram_cell_2T.py)

Gates (pre-registered, locked):
- INFRA: 33/33 valid curves
- PASS:   cell-wide median < 1.5 dec
- AMBITIOUS: cell-wide median < 0.95 dec
- HIGH_VG1: VG1=0.60 median < 5.64 (z358 baseline)

Budget: 90 min total.  9-curve eval ~ 130-150s, 24 evals + final 33 refit.
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
from scipy.optimize import differential_evolution

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))

OUT = ROOT / "results/z359_bjt_recalib"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data/sebas_2026_04_22"


def build_cfg_and_models():
    sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
    v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True, newton_max_iters=40)
    cfg.bjt_emitter_to_gnd = True
    cfg.eta_sigmoid = False  # T5 was no-op
    M1, M2 = v1.build_calibrated_models()
    return cfg, M1, M2


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
            curves.append({"VG1": vg1, "VG2": vg2, "Vd": d[:,0], "Id": np.abs(d[:,1]), "f": f.name})
    for f in sorted(DATA.glob("VG1*VG2*.csv")):
        m = re.search(r"VG1=([\d.\-]+)[_ ]*VG2=([\d.\-]+)", f.name)
        if not m: continue
        vg1 = float(m.group(1)); vg2 = float(m.group(2))
        d = np.loadtxt(f, delimiter=",", skiprows=1)
        curves.append({"VG1": vg1, "VG2": vg2, "Vd": d[:,0], "Id": np.abs(d[:,1]), "f": f.name})
    return curves


def pick_subset(curves, per_vg1=3):
    """Pick `per_vg1` curves per unique VG1, evenly spaced in VG2."""
    by_vg1 = {}
    for c in curves:
        by_vg1.setdefault(c["VG1"], []).append(c)
    subset = []
    for vg1, lst in sorted(by_vg1.items()):
        lst_sorted = sorted(lst, key=lambda x: x["VG2"])
        if len(lst_sorted) <= per_vg1:
            subset.extend(lst_sorted)
        else:
            idx = np.linspace(0, len(lst_sorted)-1, per_vg1).astype(int)
            subset.extend([lst_sorted[i] for i in idx])
    return subset


def eval_curves(cfg, M1, M2, bjt, curves):
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    rmses = []
    per = []
    for c in curves:
        Vd = torch.tensor(c["Vd"], dtype=torch.float64)
        try:
            out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd,
                             VG1=torch.tensor(c["VG1"], dtype=torch.float64),
                             VG2=torch.tensor(c["VG2"], dtype=torch.float64), warm_start=True)
            Id_pred = np.abs(out["Id"].detach().cpu().numpy())
            mask = (c["Id"] > 1e-15) & (Id_pred > 1e-15) & np.isfinite(Id_pred)
            r = float(np.sqrt(np.mean((np.log10(Id_pred[mask]) - np.log10(c["Id"][mask])) ** 2))) if mask.sum() >= 3 else float("nan")
        except Exception:
            r = float("nan")
        rmses.append(r)
        per.append({"VG1": c["VG1"], "VG2": c["VG2"], "log_rmse_dec": r})
    return rmses, per


_eval_counter = {"n": 0, "best": float("inf"), "best_x": None, "t0": None}


def make_cost(cfg, M1, M2, subset):
    from nsram.bsim4_port.bjt import GummelPoonNPN
    def cost(x):
        logBf, Va, logIs, log_gain = x
        bjt = GummelPoonNPN.from_sebas_card()
        bjt.Bf = float(10.0 ** logBf)
        bjt.Va = float(Va)
        bjt.Is = float(10.0 ** logIs)
        cfg.iii_body_gain = float(10.0 ** log_gain)
        rmses, _ = eval_curves(cfg, M1, M2, bjt, subset)
        valid = [r for r in rmses if not np.isnan(r) and np.isfinite(r)]
        if len(valid) < len(subset) // 2:
            c = 10.0
        else:
            c = float(np.median(valid))
        _eval_counter["n"] += 1
        if c < _eval_counter["best"]:
            _eval_counter["best"] = c
            _eval_counter["best_x"] = list(x)
        elapsed = time.time() - _eval_counter["t0"]
        print(f"  eval#{_eval_counter['n']:3d}  Bf={10**logBf:.1f} Va={Va:.3f} Is={10**logIs:.2e} gain={10**log_gain:.3f}  cost={c:.3f}  best={_eval_counter['best']:.3f}  t={elapsed:.0f}s", flush=True)
        return c
    return cost


def main():
    from nsram.bsim4_port.bjt import GummelPoonNPN
    t0 = time.time()
    cfg, M1, M2 = build_cfg_and_models()
    print(f"[z359] bjt_emitter_to_gnd={cfg.bjt_emitter_to_gnd}  eta_sigmoid={cfg.eta_sigmoid}", flush=True)
    print(f"[z359] M1 binunit={M1._values.get('binunit')} M2 binunit={M2._values.get('binunit')}", flush=True)

    curves = load_curves()
    print(f"[z359] loaded {len(curves)} curves", flush=True)
    subset = pick_subset(curves, per_vg1=2)
    print(f"[z359] subset {len(subset)} curves for BBO: {[(c['VG1'], c['VG2']) for c in subset]}", flush=True)

    # Timing probe (z358 baseline params)
    print("[z359] timing probe at z358 baseline params...", flush=True)
    t_probe = time.time()
    bjt0 = GummelPoonNPN.from_sebas_card(); bjt0.Bf=9000.0; bjt0.Va=0.55; bjt0.Is=1e-9
    cfg.iii_body_gain = 1.0  # no-op (default path)
    rmses0, _ = eval_curves(cfg, M1, M2, bjt0, subset)
    valid0 = [r for r in rmses0 if not np.isnan(r)]
    print(f"[z359] probe subset median={np.median(valid0):.3f} dec in {time.time()-t_probe:.1f}s ({len(valid0)}/{len(subset)} valid)", flush=True)

    bounds = [
        (np.log10(100.0),  np.log10(5000.0)),   # logBf
        (0.20, 2.00),                            # Va
        (np.log10(1e-12), np.log10(1e-7)),       # logIs
        (np.log10(0.01),  np.log10(1.0)),        # log_gain (iii_body_gain)
    ]
    print(f"[z359] bounds (for record): {bounds}", flush=True)

    _eval_counter["t0"] = time.time()
    cost = make_cost(cfg, M1, M2, subset)

    # R-39 budget pivot: 6-curve subset ~75s/eval. Use a focused manual grid
    # rather than scipy DE (which would need 96 evals = 2h on this subset).
    # Dominant knobs from R-38 diagnosis: Bf (over-pumping magnitude) and
    # iii_body_gain (attenuation). Va swept coarse, Is fixed at 1e-9.
    Bf_grid    = [100.0, 1000.0, 5000.0]
    gain_grid  = [0.05, 0.15, 0.5, 1.0]   # 1.0 = no attenuation (z358 baseline)
    Va_grid    = [0.55, 1.5]
    Is_grid    = [1e-9]
    grid_points = [(np.log10(b), va, np.log10(i), np.log10(g))
                   for b in Bf_grid for g in gain_grid for va in Va_grid for i in Is_grid]
    print(f"[z359] grid {len(grid_points)} pts: Bf={Bf_grid}, gain={gain_grid}, Va={Va_grid}, Is={Is_grid}", flush=True)
    grid_results = []
    for x in grid_points:
        c = cost(x)
        grid_results.append({"x": list(x), "cost": c})

    class _R: pass
    res = _R()
    res.x = np.array(_eval_counter["best_x"]) if _eval_counter["best_x"] is not None else np.array(grid_points[0])
    res.fun = float(_eval_counter["best"])
    res.nfev = _eval_counter["n"]
    print(f"[z359] BBO done: best={res.fun:.3f}  x={res.x}", flush=True)
    best_x = res.x
    if _eval_counter["best_x"] is not None and _eval_counter["best"] < res.fun:
        best_x = _eval_counter["best_x"]
        print(f"[z359] using best-tracked x (cost={_eval_counter['best']:.3f})", flush=True)

    logBf, Va, logIs, log_gain = best_x
    best_params = {
        "Bf": float(10.0 ** logBf),
        "Va": float(Va),
        "Is": float(10.0 ** logIs),
        "iii_body_gain": float(10.0 ** log_gain),
    }
    print(f"[z359] BEST params: {best_params}", flush=True)

    # Final 33-curve refit at best params
    print("[z359] final 33-curve refit...", flush=True)
    t_refit = time.time()
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = best_params["Bf"]; bjt.Va = best_params["Va"]; bjt.Is = best_params["Is"]
    cfg.iii_body_gain = best_params["iii_body_gain"]
    rmses_full, per_full = eval_curves(cfg, M1, M2, bjt, curves)
    valid_full = [r for r in rmses_full if not np.isnan(r)]
    per_vg1 = {}
    for r in per_full:
        per_vg1.setdefault(r["VG1"], []).append(r["log_rmse_dec"])
    per_vg1_median = {f"{k:.2f}": float(np.nanmedian(v)) for k,v in per_vg1.items()}
    cell_med = float(np.median(valid_full)) if valid_full else None
    print(f"[z359] final: cell-wide median = {cell_med}, per_VG1={per_vg1_median}", flush=True)
    print(f"[z359] refit took {time.time()-t_refit:.1f}s", flush=True)

    # Gates
    gates = {
        "INFRA_33_valid": len(valid_full) == 33,
        "PASS_lt_1.5":     cell_med is not None and cell_med < 1.5,
        "AMBITIOUS_lt_0.95": cell_med is not None and cell_med < 0.95,
        "HIGH_VG1_lt_5.64": ("0.60" in per_vg1_median) and per_vg1_median["0.60"] < 5.64,
    }
    summary = {
        "script": "z359_bjt_recalib",
        "patches_active": [
            "R-20 BJT Vbc=Vb-Vsint",
            "R-29 lpe0/toxe",
            "R-37 binunit override removed",
            "R-39 iii_body_gain trigger relaxed (< 1.0 active)",
        ],
        "M2_binunit": int(M2._values.get("binunit", -1)),
        "M2_binunit_was_overridden": False,
        "bbo": {
            "method": "manual_grid_24pts",
            "evals": _eval_counter["n"],
            "best_inner_cost": float(_eval_counter["best"]),
            "subset_n": len(subset),
            "grid_results": grid_results,
        },
        "best_params": best_params,
        "final_full_refit": {
            "n_curves_valid": len(valid_full),
            "n_curves_total": len(rmses_full),
            "cell_wide_median_dec": cell_med,
            "per_VG1_median": per_vg1_median,
            "per_curve": per_full,
        },
        "gates": gates,
        "baselines": {"z358_post_R37": 4.28, "z346": 4.08, "z352_best": 3.93,
                      "z358_per_VG1": {"0.20": 2.05, "0.40": 4.20, "0.60": 5.64}},
        "elapsed_s": time.time() - t0,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[z359] gates={gates}", flush=True)
    print(f"[z359] elapsed {summary['elapsed_s']:.1f}s", flush=True)

    # Plots per VG1
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for vg1, lst in sorted(per_vg1.items()):
            fig, ax = plt.subplots(figsize=(6,4))
            vg2_vals = sorted([c["VG2"] for c in curves if c["VG1"] == vg1])
            rmses_vg2 = [next(r["log_rmse_dec"] for r in per_full if r["VG1"]==vg1 and abs(r["VG2"]-v)<1e-6) for v in vg2_vals]
            ax.plot(vg2_vals, rmses_vg2, "o-", label=f"VG1={vg1}")
            ax.axhline(1.5, ls="--", c="g", alpha=0.5, label="PASS gate")
            ax.axhline(0.95, ls="--", c="b", alpha=0.5, label="AMBITIOUS")
            ax.set_xlabel("VG2"); ax.set_ylabel("log RMSE (dec)")
            ax.set_title(f"z359 R-39 best  VG1={vg1}  median={per_vg1_median[f'{vg1:.2f}']:.2f}")
            ax.legend(); ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(OUT / f"refit_VG1_{vg1:.2f}.png", dpi=100)
            plt.close(fig)
        print(f"[z359] plots written to {OUT}", flush=True)
    except Exception as e:
        print(f"[z359] plot err: {e}", flush=True)


if __name__ == "__main__":
    main()
