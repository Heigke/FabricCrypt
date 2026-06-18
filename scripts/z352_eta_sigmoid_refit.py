"""z352 — T5 prescription: V_b-dependent eta_sigmoid clamp on Iii-to-body
routing, on TOP of R-29 Vth/tox fix + R-20 BJT Vbc fix.

Mirrors z346 (R-30 baseline). Adds cfg.eta_sigmoid=True with T5 params:
   eta_0=0.6, eta_final=0.05, eta_k=30, eta_vturn=0.55

Pre-registered gates (LOCKED):
  INFRA: 33/33 valid, no NaN
  DIAGNOSTIC: V_b at flagship bias < 0.7 V (clamp active)
  PASS: cell-wide median < 1.5 dec (vs z346 baseline 4.08)
  AMBITIOUS: cell-wide median < 0.95 dec
  HIGH-VG1: VG1=0.60 median < 4.0 dec (vs z346=5.42)

Configuration:
  - bjt_emitter_to_gnd=True (R-20 fix)
  - build_calibrated_models() — ORIGINAL M1 card with R-29 Vth/tox patch
  - BJT z334 defaults: Bf=9000, Va=0.55, Is=1e-9
  - eta_sigmoid=True with sweep over eta_vturn in {0.40, 0.50, 0.55, 0.60, 0.70}

Output: results/z352_eta_sigmoid/summary.json + per-VG1 plots for best variant.
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

OUT = ROOT / "results/z352_eta_sigmoid"
OUT.mkdir(parents=True, exist_ok=True)

DATA = ROOT / "data/sebas_2026_04_22"

BJT_BF = 9000.0
BJT_VA = 0.55
BJT_IS = 1.0e-9

# T5 prescription (flagship)
ETA_0       = 0.6
ETA_FINAL   = 0.05
ETA_K       = 30.0
ETA_VTURN_FLAGSHIP = 0.55
ETA_VTURN_SWEEP = [0.40, 0.50, 0.55, 0.60, 0.70]


def build_pyport(eta_vturn: float):
    sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
    v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
    from nsram.bsim4_port.bjt import GummelPoonNPN

    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                            newton_max_iters=40)
    cfg.bjt_emitter_to_gnd = True
    # T5 eta_sigmoid clamp
    cfg.eta_sigmoid = True
    cfg.eta_0       = ETA_0
    cfg.eta_final   = ETA_FINAL
    cfg.eta_k       = ETA_K
    cfg.eta_vturn   = float(eta_vturn)

    M1, M2 = v1.build_calibrated_models()

    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = BJT_BF
    bjt.Va = BJT_VA
    bjt.Is = BJT_IS
    return cfg, M1, M2, bjt


def load_curves():
    curves = []
    for f in sorted(DATA.glob("VG1*VG2*.csv")):
        m = re.search(r"VG1=([\d.\-]+)[_ ]*VG2=([\d.\-]+)", f.name)
        if not m: continue
        vg1 = float(m.group(1)); vg2 = float(m.group(2))
        d = np.loadtxt(f, delimiter=",", skiprows=1)
        curves.append({"VG1": vg1, "VG2": vg2, "Vd": d[:,0],
                       "Id": np.abs(d[:,1]), "f": f.name})
    if not curves:
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


def probe_flagship_cell(cfg, M1, M2, bjt):
    """Run forward_2t at flagship bias, return Vsint, Vb, Id, eta_lat actual."""
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    VG1, VG2 = 0.6, 0.20
    # Flagship Vd ~ 2.0V (single point — wrap as sequence)
    Vd_seq = torch.tensor([2.0], dtype=torch.float64)
    out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt,
                     Vd_seq=Vd_seq,
                     VG1=torch.tensor(VG1, dtype=torch.float64),
                     VG2=torch.tensor(VG2, dtype=torch.float64),
                     warm_start=True)
    Vb_val = float(out["Vb"].detach().cpu().numpy().ravel()[0])
    Vsint_val = float(out["Vsint"].detach().cpu().numpy().ravel()[0])
    Id_val = float(out["Id"].detach().cpu().numpy().ravel()[0])
    # Compute realized eta_lat at flagship Vb
    eta_lat = ETA_FINAL + (ETA_0 - ETA_FINAL) * float(
        torch.sigmoid(torch.tensor(-ETA_K * (Vb_val - cfg.eta_vturn))))
    return {
        "VG1": VG1, "VG2": VG2, "Vd": 2.0,
        "Vsint": Vsint_val, "Vb": Vb_val, "Id": Id_val,
        "eta_lat_realized": eta_lat,
        "eta_vturn": cfg.eta_vturn,
    }


def run_one_variant(eta_vturn: float):
    t0 = time.time()
    cfg, M1, M2, bjt = build_pyport(eta_vturn)
    from nsram.bsim4_port.nsram_cell_2T import forward_2t

    # Probe at flagship FIRST
    try:
        probe = probe_flagship_cell(cfg, M1, M2, bjt)
    except Exception as e:
        probe = {"err": f"{type(e).__name__}: {e}"}

    curves = load_curves()
    results = []
    per_vg1 = {}
    preds = []
    for c in curves:
        Vd = torch.tensor(c["Vd"], dtype=torch.float64)
        try:
            out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt,
                             Vd_seq=Vd,
                             VG1=torch.tensor(c["VG1"], dtype=torch.float64),
                             VG2=torch.tensor(c["VG2"], dtype=torch.float64),
                             warm_start=True)
            Id_pred = np.abs(out["Id"].detach().cpu().numpy())
            mask = (c["Id"] > 1e-15) & (Id_pred > 1e-15) & np.isfinite(Id_pred)
            if mask.sum() < 3:
                rmse = float("nan")
            else:
                logr = np.log10(Id_pred[mask]) - np.log10(c["Id"][mask])
                rmse = float(np.sqrt(np.mean(logr ** 2)))
        except Exception as e:
            rmse = float("nan")
            Id_pred = None
            print(f"  [vturn={eta_vturn}] err {c['f']}: {type(e).__name__}: {e}", flush=True)
        results.append({"VG1": c["VG1"], "VG2": c["VG2"],
                        "log_rmse_dec": rmse, "f": c["f"]})
        per_vg1.setdefault(c["VG1"], []).append(rmse)
        preds.append({"c": c, "Id_pred": Id_pred, "rmse": rmse})

    valid = [r["log_rmse_dec"] for r in results if not np.isnan(r["log_rmse_dec"])]
    median_dec = float(np.median(valid)) if valid else None
    per_vg1_med = {f"{k:.2f}": (float(np.median([x for x in v if not np.isnan(x)]))
                                if any(not np.isnan(x) for x in v) else None)
                   for k, v in per_vg1.items()}
    summary = {
        "eta_vturn": eta_vturn,
        "n_curves_total": len(results),
        "n_curves_valid": len(valid),
        "cell_wide_median_dec": median_dec,
        "cell_wide_p25_dec": float(np.percentile(valid, 25)) if valid else None,
        "cell_wide_p75_dec": float(np.percentile(valid, 75)) if valid else None,
        "cell_wide_max_dec": float(np.max(valid)) if valid else None,
        "cell_wide_min_dec": float(np.min(valid)) if valid else None,
        "per_VG1_median": per_vg1_med,
        "flagship_probe": probe,
        "elapsed_s": time.time() - t0,
        "per_curve": results,
        "preds": preds,
    }
    return summary


def make_plots(variant_summary, tag="best"):
    preds = variant_summary["preds"]
    vg1_set = sorted(set(p["c"]["VG1"] for p in preds))
    for vg1 in vg1_set:
        these = [p for p in preds if abs(p["c"]["VG1"] - vg1) < 1e-6]
        if not these: continue
        these.sort(key=lambda p: p["c"]["VG2"])
        med = variant_summary["per_VG1_median"].get(f"{vg1:.2f}")
        title = (f"z352 eta_sigmoid (vturn={variant_summary['eta_vturn']}) — "
                 f"VG1={vg1:.2f}"
                 + (f"  (median dec={med:.2f})" if med is not None else ""))
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        fig.suptitle(title)
        ax = axes[0]
        cmap = plt.cm.viridis(np.linspace(0, 1, len(these)))
        for col, p in zip(cmap, these):
            c = p["c"]
            ax.semilogy(c["Vd"], np.maximum(c["Id"], 1e-16), color=col, lw=1.5,
                        label=f"VG2={c['VG2']:+.2f}")
            if p["Id_pred"] is not None:
                ax.semilogy(c["Vd"], np.maximum(p["Id_pred"], 1e-16),
                            "--", color=col, lw=1.0)
        ax.set_xlabel("Vd [V]"); ax.set_ylabel("|Id| [A]")
        ax.set_title("solid=silicon  dashed=pyport"); ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        ax = axes[1]
        vg2s = [p["c"]["VG2"] for p in these]
        rmses = [p["rmse"] for p in these]
        ax.plot(vg2s, rmses, "o-")
        ax.axhline(1.5, color="g", ls=":", label="PASS=1.5")
        ax.axhline(0.95, color="b", ls=":", label="AMBITIOUS=0.95")
        ax.set_xlabel("VG2 [V]"); ax.set_ylabel("log10 RMSE [dec]")
        ax.set_title("per-curve RMSE"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        ax = axes[2]
        for col, p in zip(cmap, these):
            if p["Id_pred"] is None: continue
            c = p["c"]
            mask = (c["Id"] > 1e-15) & (p["Id_pred"] > 1e-15) & np.isfinite(p["Id_pred"])
            if mask.sum() < 2: continue
            resid = np.log10(p["Id_pred"][mask]) - np.log10(c["Id"][mask])
            ax.plot(c["Vd"][mask], resid, color=col, lw=1.0,
                    label=f"VG2={c['VG2']:+.2f}")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("Vd [V]"); ax.set_ylabel("log10(pyport) − log10(silicon)")
        ax.set_title("log residual"); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"plot_{tag}_VG1_{vg1:.2f}.png", dpi=110)
        plt.close(fig)


def main():
    t0 = time.time()
    print("=" * 70, flush=True)
    print(f"[z352] T5 eta_sigmoid refit; sweep eta_vturn over {ETA_VTURN_SWEEP}",
          flush=True)
    print(f"[z352] eta_0={ETA_0} eta_final={ETA_FINAL} eta_k={ETA_K}", flush=True)
    print(f"[z352] BJT z334 defaults: Bf={BJT_BF} Va={BJT_VA} Is={BJT_IS:.2e}",
          flush=True)
    print("=" * 70, flush=True)

    variants = {}
    for vturn in ETA_VTURN_SWEEP:
        print(f"\n[z352] === variant eta_vturn={vturn} ===", flush=True)
        s = run_one_variant(vturn)
        med = s["cell_wide_median_dec"]
        per_vg1 = s["per_VG1_median"]
        probe = s["flagship_probe"]
        print(f"[z352] vturn={vturn}: median_dec={med}  per_VG1={per_vg1}",
              flush=True)
        print(f"[z352] vturn={vturn}: flagship probe={probe}", flush=True)
        variants[f"{vturn:.2f}"] = s

    # Pick best by cell-wide median
    valid_vars = {k: v for k, v in variants.items()
                  if v["cell_wide_median_dec"] is not None
                  and not np.isnan(v["cell_wide_median_dec"])}
    best_key = min(valid_vars.keys(),
                   key=lambda k: valid_vars[k]["cell_wide_median_dec"])
    best = valid_vars[best_key]
    print(f"\n[z352] BEST variant: eta_vturn={best_key}  "
          f"median_dec={best['cell_wide_median_dec']:.3f}", flush=True)

    # Pre-registered gates
    med = best["cell_wide_median_dec"]
    probe = best["flagship_probe"]
    Vb_flag = probe.get("Vb")
    high_vg1_med = best["per_VG1_median"].get("0.60")
    n_valid = best["n_curves_valid"]
    n_total = best["n_curves_total"]

    gates = {
        "INFRA_all33_valid": (n_valid == 33) and (n_total == 33),
        "DIAGNOSTIC_Vb_lt_0p7":
            (Vb_flag is not None) and (Vb_flag < 0.7),
        "PASS_median_lt_1p5":
            (med is not None) and (med < 1.5),
        "AMBITIOUS_median_lt_0p95":
            (med is not None) and (med < 0.95),
        "HIGH_VG1_lt_4p0":
            (high_vg1_med is not None) and (high_vg1_med < 4.0),
    }
    print(f"\n[z352] GATES @ best (vturn={best_key}):", flush=True)
    for g, v in gates.items():
        print(f"  {g}: {v}", flush=True)

    # Strip preds before JSON (large)
    final = {
        "script": "z352_eta_sigmoid_refit",
        "config": {
            "bjt_emitter_to_gnd": True,
            "use_bjt": True,
            "eta_sigmoid": True,
            "eta_0": ETA_0,
            "eta_final": ETA_FINAL,
            "eta_k": ETA_K,
            "eta_vturn_sweep": ETA_VTURN_SWEEP,
            "M1_card": "ORIGINAL (build_calibrated_models) w/ R-29 Vth/tox patch",
            "bjt": {"Bf": BJT_BF, "Va": BJT_VA, "Is": BJT_IS},
        },
        "baselines": {
            "z346_cell_wide_median_dec": 4.08,
            "z346_per_VG1_0.60": 5.42,
            "z346_per_VG1_0.40": 3.74,
            "z346_per_VG1_0.20": 2.32,
        },
        "variants": {
            k: {kk: vv for kk, vv in v.items() if kk != "preds"}
            for k, v in variants.items()
        },
        "best_eta_vturn": best_key,
        "best_median_dec": med,
        "best_per_VG1_median": best["per_VG1_median"],
        "best_flagship_probe": probe,
        "gates_preregistered": gates,
        "elapsed_s": time.time() - t0,
    }
    (OUT / "summary.json").write_text(json.dumps(final, indent=2, default=str))

    make_plots(best, tag=f"best_vturn_{best_key}")

    print(f"\n[z352] DONE. Total elapsed {time.time()-t0:.1f}s. "
          f"Output: {OUT/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
