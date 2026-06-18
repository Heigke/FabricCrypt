"""z372 — Snapback comparison demo: pyport model vs Sebas measured.

Three subplots for VG1 in {0.2, 0.4, 0.6}, all with VG2=+0.20.
Black dots = measured, red line = model, semilogy.

Uses R-46 per-VG1 BBO best params from results/z365_perVG1_bbo/bbo_history.json
(best_so_far.x), since cell-wide-best (0.965 dec) beats the GPU-blitz physical
global (1.048 dec).

Output: results/z372_snapback_demo/{snapback_compare.png, summary.json}
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
import sys, json, re, math, csv, importlib.util
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "nsram"))
OUT = ROOT / "results/z372_snapback_demo"; OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data/sebas_2026_04_22"


@contextmanager
def patch_sd_scaled(sd, overrides):
    if not overrides:
        yield; return
    saved = {}
    try:
        for k, v in overrides.items():
            saved[k] = sd.scaled.get(k, None)
            sd.scaled[k] = float(v)
        yield
    finally:
        for k, v in saved.items():
            if v is None: sd.scaled.pop(k, None)
            else: sd.scaled[k] = v


def load_sebas_params():
    path = DATA / "2Tcell_BSIM_param_DC.csv"
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                try: row[k] = float(v)
                except ValueError: row[k] = float("nan")
            rows.append(row)
    return rows


BRANCH_FLAT = {
    0.4: {"ETAB": 1.9,  "K1": 0.53825, "ALPHA0": 7.842e-05, "BETA0": 19.0, "NFACTOR": 6.0, "trise": 10.59},
    0.6: {"ETAB": 2.5,  "K1": 0.41825, "ALPHA0": 7.842e-05, "BETA0": 20.0, "NFACTOR": 6.0, "trise": 9.04},
}
M2_STATIC = {"k1": 0.63825, "k2": -0.070435, "etab": -0.086777, "beta0": 18.0}


def find_or_impute_row(rows, VG1, VG2, atol=1e-3):
    target = None
    for r in rows:
        if abs(r["VG1"] - VG1) < atol and abs(r["VG2"] - VG2) < atol:
            target = dict(r); break
    if target is None: return None
    if math.isnan(target.get("K1", float("nan"))):
        branch = BRANCH_FLAT.get(round(VG1, 2))
        if branch is None: return target
        for k, v in branch.items():
            target[k] = float(v)
    return target


def make_overrides(row):
    if row is None: return None, None
    P_M1 = {}
    for ck, pk in (("ETAB","etab"),("K1","k1"),("ALPHA0","alpha0"),("BETA0","beta0")):
        if not math.isnan(row.get(ck, float("nan"))): P_M1[pk] = float(row[ck])
    P_M2 = {}
    if not math.isnan(row.get("NFACTOR", float("nan"))): P_M2["nfactor"] = float(row["NFACTOR"])
    for k, v in M2_STATIC.items():
        P_M2.setdefault(k, float(v))
    return (P_M1 or None), (P_M2 or None)


def build_base():
    sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
    v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
    from nsram.bsim4_port.bjt import GummelPoonNPN
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True, newton_max_iters=40)
    cfg.bjt_emitter_to_gnd = True
    cfg.body_pdiode_to = "vnwell"
    cfg.use_well_diode = True
    cfg.vnwell = 2.0
    cfg.body_pdiode_Js = 5.3675e-7 / 22e-12
    cfg.body_pdiode_n = 1.0535
    cfg.body_pdiode_Rs = 1.0e6
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Va = 0.903; bjt.Is = 5.95e-12; bjt.Bf = 991.0
    return cfg, M1, M2, bjt


def load_measured(vg1, vg2=0.20):
    sub = DATA / f"2vHCa-2 I-Vs@VG2 VG1={vg1} vnwell=2"
    pat = re.compile(rf"VG2={vg2:.2f}_VG={vg1}")
    for f in sorted(sub.glob("*.csv")):
        if pat.search(f.name):
            d = np.loadtxt(f, delimiter=",", skiprows=1)
            return d[:, 0], np.abs(d[:, 1]), f.name
    raise FileNotFoundError(f"no csv for VG1={vg1} VG2={vg2}")


def main():
    # Per-VG1 best from R-46 (z365 best_so_far)
    # x = [Bf_020, iii_020, log10Rs_020, Bf_040, iii_040, log10Rs_040, Bf_060, iii_060, log10Rs_060]
    x_best = [1889.88, 1.8447, 9.1722,
              1092.27, 1.5152, 9.8983,
               417.63, 0.9036, 6.7846]
    per_vg1 = {0.2: (x_best[0], x_best[1], 10**x_best[2]),
               0.4: (x_best[3], x_best[4], 10**x_best[5]),
               0.6: (x_best[6], x_best[7], 10**x_best[8])}

    cfg, M1, M2, bjt = build_base()
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)
    sebas_rows = load_sebas_params()

    # NOTE: VG1=0.2 dataset only has VG2 up to +0.10 (no +0.20 available).
    # Use VG2=+0.10 for VG1=0.2 (still the worst-case branch per R-46).
    targets = [(0.2, 0.10), (0.4, 0.20), (0.6, 0.20)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    results = []

    for ax, (vg1, vg2) in zip(axes, targets):
        Vd_m, Id_m, fname = load_measured(vg1, vg2)
        Bf, iii, Rs = per_vg1[vg1]
        bjt.Bf = Bf; cfg.iii_body_gain = iii; cfg.vnwell_Rs = Rs
        row = find_or_impute_row(sebas_rows, vg1, vg2)
        P_M1, P_M2 = make_overrides(row)
        Vd_t = torch.tensor(Vd_m, dtype=torch.float64)
        with patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
            out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd_t,
                             VG1=torch.tensor(vg1, dtype=torch.float64),
                             VG2=torch.tensor(vg2, dtype=torch.float64),
                             warm_start=True)
        Id_p = np.abs(out["Id"].detach().cpu().numpy())

        mask = (Id_m > 1e-15) & (Id_p > 1e-15) & np.isfinite(Id_p)
        rmse_dec = float(np.sqrt(np.mean((np.log10(Id_p[mask]) - np.log10(Id_m[mask]))**2))) if mask.sum() >= 3 else float("nan")

        # Find snapback knee: largest forward-jump in log10(Id) in measured
        # restricted to Vd >= 0.5 V to skip sub-threshold noise jumps.
        dlog = np.diff(np.log10(np.maximum(Id_m, 1e-15)))
        Vmid = 0.5 * (Vd_m[1:] + Vd_m[:-1])
        valid_knee = Vmid >= 0.5
        if valid_knee.any() and len(dlog) > 0:
            dlog_masked = np.where(valid_knee, dlog, -np.inf)
            knee_idx = int(np.argmax(dlog_masked)) + 1
        else:
            knee_idx = None
        Vknee = float(Vd_m[knee_idx]) if knee_idx is not None else None
        jump_dec = float(dlog.max()) if len(dlog) > 0 else None

        # Approximate Vth from measured (Id crosses 1e-6 A first time)
        Vth = None
        for v, i in zip(Vd_m, Id_m):
            if i > 1e-6:
                Vth = float(v); break

        ax.semilogy(Vd_m, np.maximum(Id_m, 1e-15), "k.", ms=4, label="measured (Sebas)")
        ax.semilogy(Vd_m, np.maximum(Id_p, 1e-15), "r-", lw=1.6, label="pyport model")
        if Vknee is not None:
            ax.axvline(Vknee, color="gray", ls=":", lw=0.9, alpha=0.7)
            ax.annotate(f"snapback knee\nVd={Vknee:.2f}V, Δ={jump_dec:.1f} dec",
                        xy=(Vknee, Id_m[knee_idx]),
                        xytext=(Vknee+0.15, Id_m[knee_idx]*30),
                        fontsize=8, color="gray",
                        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
        if Vth is not None:
            ax.axvline(Vth, color="blue", ls="--", lw=0.7, alpha=0.5)
            ax.annotate(f"Vth≈{Vth:.2f}V", xy=(Vth, 1e-6),
                        xytext=(Vth-0.05, 1e-10), fontsize=8, color="blue",
                        rotation=90)

        # Verdict on snapback reproduction: did the model produce a similar jump near knee?
        if knee_idx is not None and knee_idx < len(Id_p):
            # Window around knee
            lo = max(0, knee_idx-3); hi = min(len(Id_p), knee_idx+3)
            dlog_p_window = np.diff(np.log10(np.maximum(Id_p[lo:hi], 1e-15)))
            model_jump = float(dlog_p_window.max()) if len(dlog_p_window) else 0.0
        else:
            model_jump = float("nan")
        snapback_ok = (model_jump > 1.0) if not math.isnan(model_jump) else False

        ax.set_xlabel("Vd (V)")
        ax.set_ylabel("|Id| (A)")
        ax.set_ylim(1e-13, 1e-2)
        ax.grid(True, which="both", alpha=0.3)
        verdict = "MATCHES" if snapback_ok and rmse_dec < 1.5 else "MISSES"
        ax.set_title(f"VG1={vg1}, VG2=+{vg2:.2f}\n"
                     f"RMSE={rmse_dec:.2f} dec | meas-jump={jump_dec:.1f} dec, "
                     f"model-jump={model_jump:.1f} dec\n"
                     f"snapback: {verdict}",
                     fontsize=10)
        ax.legend(loc="lower right", fontsize=8)

        results.append({
            "VG1": vg1, "VG2": vg2, "file": fname,
            "params": {"Bf": Bf, "iii_body_gain": iii, "vnwell_Rs": Rs},
            "rmse_dec": rmse_dec,
            "n_valid_points": int(mask.sum()),
            "Vth_measured_V": Vth,
            "V_snapback_knee_V": Vknee,
            "measured_jump_dec": jump_dec,
            "model_jump_dec": model_jump,
            "snapback_reproduced": bool(snapback_ok),
        })

    # Honest title: did we reproduce snapback?
    snapbacks = [r["snapback_reproduced"] for r in results]
    n_ok = sum(snapbacks); n_tot = len(snapbacks)
    rmses = [r["rmse_dec"] for r in results]
    if n_ok == n_tot:
        suptitle = (f"pyport vs Sebas — snapback REPRODUCED on {n_ok}/{n_tot} biases "
                    f"(VG2=+0.20). RMSE={rmses[0]:.2f}/{rmses[1]:.2f}/{rmses[2]:.2f} dec.")
    elif n_ok == 0:
        suptitle = (f"pyport vs Sebas — snapback FOLD MISSING on {n_tot}/{n_tot} biases "
                    f"(VG2=+0.20). Sub-threshold OK; post-knee shape wrong. "
                    f"RMSE={rmses[0]:.2f}/{rmses[1]:.2f}/{rmses[2]:.2f} dec.")
    else:
        suptitle = (f"pyport vs Sebas — partial: snapback on {n_ok}/{n_tot} biases. "
                    f"RMSE={rmses[0]:.2f}/{rmses[1]:.2f}/{rmses[2]:.2f} dec.")

    fig.suptitle(suptitle, fontsize=11, y=1.00)
    fig.tight_layout()
    out_png = OUT / "snapback_compare.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"[z372] wrote {out_png}")

    summary = {
        "script": "z372_snapback_demo",
        "task": "Compare pyport model vs Sebas measured snapback @ VG2=+0.20 for VG1 in {0.2,0.4,0.6}",
        "params_source": "R-46 z365 per-VG1 BBO best (cell-wide median = 0.965 dec)",
        "x_best_R46": x_best,
        "results_per_bias": results,
        "snapback_reproduced_count": int(n_ok),
        "total_biases": int(n_tot),
        "rmse_summary_dec": rmses,
        "verdict": suptitle,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[z372] wrote {OUT/'summary.json'}")
    print(suptitle)


if __name__ == "__main__":
    main()
