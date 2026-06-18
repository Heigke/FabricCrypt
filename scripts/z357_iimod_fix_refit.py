"""z346 — R-30: Cell-wide refit with R-29 Vth patch (lpe0/toxe).

The R-29 audit found a SECOND root cause for the cell-wide dec gap:
pyport Vth was +88 mV too high (0.8247 vs ngspice 0.7367), traced to
patch_model_values hardcoding stale BSIM4 defaults (lpe0=1.74e-7,
tox=3e-9). showmod on ngspice-42 actually resolves the card values
(lpe0=1.2439e-7, tox=4e-9). Patches were applied to
scripts/z91f_validate_with_sebas_params.py L174 and L218-220
(see CLAUDE.md log 2026-05-14 01:20).

Expected: Vth −88 mV → Ids_M1 +2.6 dec → Iii +2.6 dec → Vb lifts off
floor → Q1 collector activates → cell-wide dec drops from
z343=3.99 / z344C=4.61 → target ≤ 1.5.

Configuration (per user instruction):
  - cfg.bjt_emitter_to_gnd = True (R-20 fix, kept from z337)
  - build_calibrated_models() — uses ORIGINAL M1 card (M1_130DNWFB.txt)
    with patched lpe0/toxe values applied via patch_model_values.
    NOT the M1_LALPHA0_FIX card and NOT the z338 BBO BJT params,
    because those were tuned to a pyport with the +88 mV Vth bug.
  - BJT defaults: Bf=9000, Va=0.55, Is=1e-9 (z334 baseline)
  - No lat_BV / body_pdiode_Rs overrides (defaults)

Also probes pyport Iii_M1 at the flagship R-25 OP
(VG1=0.6, VG2=0.20, Vd=2.0) post-patch to confirm Vth dropped and
Ids increased.

Baselines: z304_v4=0.99, z337=4.16, z343=3.99, z344C=4.61.
Gate: PASS < 0.95, AMBITIOUS < 0.50.
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

OUT = ROOT / "results/z346_vth_fix_refit"
OUT.mkdir(parents=True, exist_ok=True)

DATA = ROOT / "data/sebas_2026_04_22"

# --- z334 defaults (pre-z338 BBO) ---
BJT_BF = 9000.0
BJT_VA = 0.55
BJT_IS = 1.0e-9


def build_pyport():
    sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
    v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
    from nsram.bsim4_port.bjt import GummelPoonNPN

    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                             newton_max_iters=40)
    cfg.bjt_emitter_to_gnd = True

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


def probe_flagship(M1, M2):
    """Probe pyport at R-25 OP to confirm Vth drop / Ids gain."""
    from nsram.bsim4_port import dc as dc_mod
    from nsram.bsim4_port.geometry import Geometry
    from nsram.bsim4_port.temp import compute_size_dep
    VG1, VG2, VD = 0.6, 0.20, 2.0
    VSINT, VB = 0.382, 0.267
    VGS = VG1 - VSINT
    VDS = VD - VSINT
    VBS = VB - VSINT
    geom = Geometry(L=0.13e-6, W=1e-6)
    sd = compute_size_dep(M1, geom, T_C=27.0)
    try:
        res = dc_mod.compute_dc(M1, sd,
                                Vgs=torch.tensor(VGS, dtype=torch.float64),
                                Vds=torch.tensor(VDS, dtype=torch.float64),
                                Vbs=torch.tensor(VBS, dtype=torch.float64))
        return {
            "OP": {"Vg": VG1, "Vd": VD, "Vsint": VSINT, "Vb": VB,
                   "Vgs": VGS, "Vds": VDS, "Vbs": VBS},
            "Ids_M1":  float(res.Ids),
            "Vth":     float(res.Vth),
            "Vdsat":   float(res.Vdsat),
            "Vgsteff": float(res.Vgsteff),
            "ng_reference": {"Ids_M1": 1.5e-11, "Vth": 0.7367},
            "delta_Vth_mV": float((res.Vth - 0.7367) * 1000),
        }
    except Exception as e:
        return {"err": f"{type(e).__name__}: {e}"}


def main():
    t0 = time.time()
    cfg, M1, M2, bjt = build_pyport()
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    print(f"[z346] cfg.bjt_emitter_to_gnd={cfg.bjt_emitter_to_gnd}  use_bjt={cfg.use_bjt}", flush=True)
    print(f"[z346] BJT defaults: Bf={bjt.Bf} Va={bjt.Va} Is={bjt.Is:.2e}", flush=True)
    print(f"[z346] M1 lpe0={M1._values.get('lpe0')}  toxe={M1._values.get('toxe')}  alpha0={M1._values.get('alpha0')}", flush=True)

    # Flagship probe
    probe = probe_flagship(M1, M2)
    print(f"[z346] FLAGSHIP probe: {probe}", flush=True)

    curves = load_curves()
    print(f"[z346] loaded {len(curves)} curves", flush=True)
    if not curves:
        print(f"[z346] no curves found in {DATA}", flush=True)
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
        results.append({"VG1": c["VG1"], "VG2": c["VG2"],
                        "log_rmse_dec": rmse, "f": c["f"]})
        per_vg1.setdefault(c["VG1"], []).append(rmse)
        preds.append({"c": c, "Id_pred": Id_pred, "rmse": rmse})
        print(f"  VG1={c['VG1']:.2f} VG2={c['VG2']:+.2f}  rmse={rmse:.3f}", flush=True)

    valid = [r["log_rmse_dec"] for r in results if not np.isnan(r["log_rmse_dec"])]
    summary = {
        "script": "z346_vth_fix_refit",
        "patch": "R-29 Vth fix (lpe0=1.2439e-7, toxe/toxp/toxm=4e-9). "
                 "build_calibrated_models (original M1 card). "
                 "cfg.bjt_emitter_to_gnd=True. BJT z334 defaults.",
        "cfg": {"bjt_emitter_to_gnd": True, "use_bjt": True},
        "bjt": {"Bf": BJT_BF, "Va": BJT_VA, "Is": BJT_IS},
        "flagship_probe": probe,
        "n_curves_total": len(results),
        "n_curves_valid": len(valid),
        "cell_wide_median_dec": float(np.median(valid)) if valid else None,
        "cell_wide_p25_dec":    float(np.percentile(valid, 25)) if valid else None,
        "cell_wide_p75_dec":    float(np.percentile(valid, 75)) if valid else None,
        "cell_wide_max_dec":    float(np.max(valid)) if valid else None,
        "cell_wide_min_dec":    float(np.min(valid)) if valid else None,
        "per_VG1_median": {f"{k:.2f}": (float(np.median([x for x in v if not np.isnan(x)])) if any(not np.isnan(x) for x in v) else None) for k,v in per_vg1.items()},
        "baselines": {"z304_v4": 0.99, "z337": 4.16, "z343": 3.99, "z344C": 4.61},
        "elapsed_s": time.time() - t0,
        "per_curve": results,
    }
    if summary["cell_wide_median_dec"] is not None:
        m = summary["cell_wide_median_dec"]
        summary["gate_PASS_lt_0.95"] = m < 0.95
        summary["gate_AMBITIOUS_lt_0.50"] = m < 0.50

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    # Plots per VG1
    vg1_set = sorted(set(r["VG1"] for r in results))
    for vg1 in vg1_set:
        these = [p for p in preds if abs(p["c"]["VG1"] - vg1) < 1e-6]
        if not these: continue
        these.sort(key=lambda p: p["c"]["VG2"])
        med = summary["per_VG1_median"].get(f"{vg1:.2f}")
        title = (f"z346 Vth-fix refit — VG1={vg1:.2f}"
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
        ax.axhline(0.95, color="g", ls=":", label="PASS=0.95")
        ax.axhline(0.50, color="b", ls=":", label="AMBITIOUS=0.50")
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
        fig.savefig(OUT / f"plot_VG1_{vg1:.2f}.png", dpi=110)
        plt.close(fig)

    print(f"\n[z346] DONE median={summary['cell_wide_median_dec']}  per_VG1={summary['per_VG1_median']}", flush=True)
    print(f"[z346] gate_PASS={summary.get('gate_PASS_lt_0.95')}  gate_AMBITIOUS={summary.get('gate_AMBITIOUS_lt_0.50')}", flush=True)
    print(f"[z346] elapsed {summary['elapsed_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
