"""z343 — Cell-wide refit using PATCHED M1 card (R-26 lalpha0=0 fix).

Loads M1 from M1_130DNWFB_LALPHA0_FIX.txt (alpha0=7.84e-4, lalpha0=0)
instead of the original M1_130DNWFB.txt where lalpha0=-9.84e-12 cancels
97% of alpha0 at L=0.13µm.

Config:
  - cfg.bjt_emitter_to_gnd = True (R-20 fix from z337)
  - z338 eval 21 best BJT params: Bf=2605, Va=0.357, Is=3.29e-10
  - lat_BV=4.02, body_pdiode_Rs=5.48e6 (from same eval)
  - NO alpha0 override (use the patched card value)

Baselines: z304=0.99, z337=4.16, z338_best=3.43, z339A=4.45
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

OUT = ROOT / "results/z343_iimod_fix_refit"
OUT.mkdir(parents=True, exist_ok=True)

DATA = ROOT / "data/sebas_2026_04_22"

# z338 eval 21 best (excluding alpha0 — card provides it via patch)
BJT_BF = 2605.2882016162002
BJT_VA = 0.3567358318716285
BJT_IS = 3.2906845928467974e-10
LAT_BV = 4.018266147002578
BODY_PDIODE_RS = 5480383.486345367


def build_pyport():
    # Pull helpers from z96 module
    sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
    v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
    from nsram.bsim4_port.model_card import BSIM4Model
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
    from nsram.bsim4_port.bjt import GummelPoonNPN
    # patch_model_values lives in z96 module under alias 'f'
    f = v1.f

    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                             newton_max_iters=40)
    cfg.bjt_emitter_to_gnd = True
    cfg.lat_BV = LAT_BV
    cfg.body_pdiode_Rs = BODY_PDIODE_RS

    # Build M1 from PATCHED card
    text_M1 = (DATA / "M1_130DNWFB_LALPHA0_FIX.txt").read_text()
    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    M1 = BSIM4Model.from_spice(text_M1, model_type="nmos")
    M2 = BSIM4Model.from_spice(text_M2, model_type="nmos")
    f.patch_model_values(M1, type_n=True)
    f.patch_model_values(M2, type_n=True)
    M1._values["voff"] = M1._values.get("voff", -0.1368)
    M2._values["voff"] = M2._values.get("voff", -0.1368)

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
        curves.append({"VG1": vg1, "VG2": vg2, "Vd": d[:,0], "Id": np.abs(d[:,1]), "f": f.name})
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
                curves.append({"VG1": vg1, "VG2": vg2, "Vd": d[:,0], "Id": np.abs(d[:,1]), "f": f.name})
    return curves


def main():
    t0 = time.time()
    cfg, M1, M2, bjt = build_pyport()
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    print(f"[z343] cfg.bjt_emitter_to_gnd={cfg.bjt_emitter_to_gnd}  use_bjt={cfg.use_bjt}", flush=True)
    print(f"[z343] M1 patched card; bjt Bf={bjt.Bf} Va={bjt.Va} Is={bjt.Is:.2e}", flush=True)
    print(f"[z343] cfg.lat_BV={cfg.lat_BV} cfg.body_pdiode_Rs={cfg.body_pdiode_Rs:.2e}", flush=True)

    # Verify alpha0 patch is in effect
    try:
        a0 = M1._values.get("alpha0")
        la0 = M1._values.get("lalpha0")
        print(f"[z343] M1 alpha0={a0} lalpha0={la0}", flush=True)
    except Exception:
        pass

    curves = load_curves()
    print(f"[z343] loaded {len(curves)} curves", flush=True)
    if not curves:
        print(f"[z343] no curves found in {DATA}", flush=True)
        return

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
            print(f"  err {c['f']}: {type(e).__name__}: {e}", flush=True)
        results.append({"VG1": c["VG1"], "VG2": c["VG2"], "log_rmse_dec": rmse, "f": c["f"]})
        per_vg1.setdefault(c["VG1"], []).append(rmse)
        preds.append({"c": c, "Id_pred": Id_pred, "rmse": rmse})
        print(f"  VG1={c['VG1']:.2f} VG2={c['VG2']:+.2f}  rmse={rmse:.3f}", flush=True)

    valid = [r["log_rmse_dec"] for r in results if not np.isnan(r["log_rmse_dec"])]
    summary = {
        "script": "z343_iimod_fix_refit",
        "patch": "M1 card lalpha0=0 + alpha0=7.84e-4 (R-26 fix); cfg.bjt_emitter_to_gnd=True; z338 eval21 BJT/lat_BV/Rs",
        "cfg": {
            "bjt_emitter_to_gnd": True, "use_bjt": True,
            "lat_BV": LAT_BV, "body_pdiode_Rs": BODY_PDIODE_RS,
        },
        "bjt": {"Bf": BJT_BF, "Va": BJT_VA, "Is": BJT_IS},
        "M1_card_file": "M1_130DNWFB_LALPHA0_FIX.txt",
        "n_curves_total": len(results),
        "n_curves_valid": len(valid),
        "cell_wide_median_dec": float(np.median(valid)) if valid else None,
        "cell_wide_p25_dec":    float(np.percentile(valid, 25)) if valid else None,
        "cell_wide_p75_dec":    float(np.percentile(valid, 75)) if valid else None,
        "cell_wide_max_dec":    float(np.max(valid)) if valid else None,
        "cell_wide_min_dec":    float(np.min(valid)) if valid else None,
        "per_VG1_median": {f"{k:.2f}": (float(np.median([x for x in v if not np.isnan(x)])) if any(not np.isnan(x) for x in v) else None) for k,v in per_vg1.items()},
        "baselines": {"z304_v4": 0.99, "z337": 4.16, "z338_best": 3.43, "z339A": 4.45},
        "elapsed_s": time.time() - t0,
        "per_curve": results,
    }
    if summary["cell_wide_median_dec"] is not None:
        summary["gate_PASS_lt_0.95"] = summary["cell_wide_median_dec"] < 0.95
        summary["gate_AMBITIOUS_lt_0.50"] = summary["cell_wide_median_dec"] < 0.50

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    # 3-panel plot per VG1
    vg1_set = sorted(set(r["VG1"] for r in results))
    for vg1 in vg1_set:
        these = [p for p in preds if abs(p["c"]["VG1"] - vg1) < 1e-6]
        if not these: continue
        # Sort by VG2
        these.sort(key=lambda p: p["c"]["VG2"])
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        fig.suptitle(f"z343 IIMOD fix refit — VG1={vg1:.2f}  (median dec for this VG1: "
                     f"{summary['per_VG1_median'].get(f'{vg1:.2f}'):.2f})" if summary["per_VG1_median"].get(f"{vg1:.2f}") is not None
                     else f"z343 IIMOD fix refit — VG1={vg1:.2f}")
        # Panel 1: all curves overlaid (silicon + pred)
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
        # Panel 2: per-curve log RMSE vs VG2
        ax = axes[1]
        vg2s = [p["c"]["VG2"] for p in these]
        rmses = [p["rmse"] for p in these]
        ax.plot(vg2s, rmses, "o-")
        ax.axhline(0.95, color="g", ls=":", label="PASS=0.95")
        ax.axhline(0.50, color="b", ls=":", label="AMBITIOUS=0.50")
        ax.set_xlabel("VG2 [V]"); ax.set_ylabel("log10 RMSE [dec]")
        ax.set_title("per-curve RMSE"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        # Panel 3: log residual vs Vd, all curves
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
        fig.savefig(OUT / f"plot_VG1_{vg1:.2f}.png", dpi=110)
        plt.close(fig)

    print(f"\n[z343] DONE median={summary['cell_wide_median_dec']}  per_VG1={summary['per_VG1_median']}", flush=True)
    print(f"[z343] gate_PASS={summary.get('gate_PASS_lt_0.95')}  gate_AMBITIOUS={summary.get('gate_AMBITIOUS_lt_0.50')}", flush=True)
    print(f"[z343] elapsed {summary['elapsed_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
