"""z387 — hypothesis 2: lateral parasitic NPN reorientation.

The vertical NPN Q1 in pyport (C=Vd or Sint, E=Vsint or GND, B=Vb) may not
be the missing physics for the VG1=0.6 snapback fold. Real NS-RAM 2T cells
have a parasitic LATERAL NPN spanning the shared body between M1.drain
(collector) and M2.source = GND (emitter), with base = shared body Vb.

This script tests 3 lateral-NPN configurations on 3 canonical biases using
the R-46 per-VG1 best params. Compared to z379 cold-mode baseline (no
lateral).

Pre-registered gates:
  INFRA       : no NaN, all configs converge, finite Ids, < 60 min
  DISCOVERY   : ANY lateral config gives VG1=0.6 fold > 0.5 dec
  AMBITIOUS   : cell-wide RMSE < 0.5 dec
  KILL-SHOT   : even lateral BJT gives no fold → BJT orientation is not it

Output: results/z387_lateral_bjt/{summary.json, ids_compare.png, run.log}
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import sys, json, math, csv, re, time, importlib.util
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "nsram"))
OUT = ROOT / "results/z387_lateral_bjt"; OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data/sebas_2026_04_22"

LOG_LINES = []
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    LOG_LINES.append(line)


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
    for r in rows:
        if abs(r["VG1"] - VG1) < atol and abs(r["VG2"] - VG2) < atol:
            target = dict(r); break
    else:
        return None
    if math.isnan(target.get("K1", float("nan"))):
        branch = BRANCH_FLAT.get(round(VG1, 2))
        if branch:
            for k, v in branch.items():
                target[k] = float(v)
    return target


def make_overrides(row):
    if row is None: return None, None
    P_M1 = {}
    for ck, pk in (("ETAB","etab"),("K1","k1"),("ALPHA0","alpha0"),("BETA0","beta0")):
        v = row.get(ck, float("nan"))
        if not math.isnan(v): P_M1[pk] = float(v)
    P_M2 = {}
    nf = row.get("NFACTOR", float("nan"))
    if not math.isnan(nf): P_M2["nfactor"] = float(nf)
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
    return cfg, M1, M2, bjt, GummelPoonNPN


def load_measured(vg1, vg2=0.20):
    sub = DATA / f"2vHCa-2 I-Vs@VG2 VG1={vg1} vnwell=2"
    pat = re.compile(rf"VG2={vg2:.2f}_VG={vg1}")
    for f in sorted(sub.glob("*.csv")):
        if pat.search(f.name):
            d = np.loadtxt(f, delimiter=",", skiprows=1)
            return d[:, 0], np.abs(d[:, 1]), f.name
    raise FileNotFoundError(f"no csv for VG1={vg1} VG2={vg2}")


def rmse_dec(Id_p, Id_m, floor=1e-15):
    p = np.asarray(Id_p, dtype=float); m = np.asarray(Id_m, dtype=float)
    mask = (m > floor) & (p > floor) & np.isfinite(p)
    if mask.sum() < 3: return float("nan"), 0
    return float(np.sqrt(np.mean((np.log10(p[mask]) - np.log10(m[mask]))**2))), int(mask.sum())


def fold_dec(Id, Vd, vmin=0.5):
    Id = np.maximum(np.asarray(Id, dtype=float), 1e-15)
    Vd = np.asarray(Vd, dtype=float)
    dl = np.diff(np.log10(Id))
    Vmid = 0.5 * (Vd[1:] + Vd[:-1])
    mask = Vmid >= vmin
    if not mask.any() or len(dl) == 0:
        return 0.0
    return float(np.where(mask, dl, -np.inf).max())


# --- R-46 per-VG1 best ---
X_BEST = [1889.88, 1.8447, 9.1722,
          1092.27, 1.5152, 9.8983,
           417.63, 0.9036, 6.7846]
PER_VG1 = {0.2: (X_BEST[0], X_BEST[1], 10**X_BEST[2]),
           0.4: (X_BEST[3], X_BEST[4], 10**X_BEST[5]),
           0.6: (X_BEST[6], X_BEST[7], 10**X_BEST[8])}


def run_config(forward_2t, cfg, M1, M2, bjt, bjt_lat, sd_M1, sd_M2, vg1, vg2,
               Vd_t, P_M1, P_M2, *, use_lateral, disable_vertical, mario_canonical):
    cfg.use_lateral_bjt = bool(use_lateral)
    cfg.disable_vertical_bjt = bool(disable_vertical)
    if use_lateral and mario_canonical:
        # Mario canonical BJT for the LATERAL device
        bjt_lat.Bf = 10000.0
        bjt_lat.Is = 5e-9
        bjt_lat.Va = 100.0
        bjt_lat.Var = 1e30
        bjt_lat.Nf = 1.0
        bjt_lat.Nr = 1.0
        cfg.bjt_lateral_instance = bjt_lat
    else:
        cfg.bjt_lateral_instance = None  # reuse main bjt for lateral
    kw = dict(cfg=cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd_t,
              VG1=torch.tensor(vg1, dtype=torch.float64),
              VG2=torch.tensor(vg2, dtype=torch.float64),
              warm_start=True)
    with patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
        out = forward_2t(**kw)
    return out


def main():
    t0 = time.time()
    log(f"z387 start. R-46 best per-VG1: {X_BEST}")
    cfg, M1, M2, bjt, GummelPoonNPN = build_base()
    bjt_lat = GummelPoonNPN.from_sebas_card()  # independent instance
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)
    sebas_rows = load_sebas_params()

    targets = [(0.2, 0.10), (0.4, 0.20), (0.6, 0.20)]
    configs = [
        ("baseline_no_lat", dict(use_lateral=False, disable_vertical=False, mario_canonical=False)),
        ("lateral_only",    dict(use_lateral=True,  disable_vertical=True,  mario_canonical=False)),
        ("both_lat_vert",   dict(use_lateral=True,  disable_vertical=False, mario_canonical=False)),
        ("lateral_mario",   dict(use_lateral=True,  disable_vertical=True,  mario_canonical=True)),
    ]
    cfg_colors = {"baseline_no_lat":"tab:gray", "lateral_only":"tab:red",
                  "both_lat_vert":"tab:orange", "lateral_mario":"tab:purple"}

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    results = []
    for ax, (vg1, vg2) in zip(axes, targets):
        Vd_m, Id_m, fname = load_measured(vg1, vg2)
        Bf_v, iii_v, Rs_v = PER_VG1[vg1]
        # Use R-46 best for the vertical BJT
        bjt.Bf = Bf_v
        cfg.iii_body_gain = iii_v
        cfg.vnwell_Rs = Rs_v
        row = find_or_impute_row(sebas_rows, vg1, vg2)
        P_M1, P_M2 = make_overrides(row)
        Vd_t = torch.tensor(Vd_m, dtype=torch.float64)
        meas_fold = fold_dec(Id_m, Vd_m, vmin=0.5)

        log(f"  VG1={vg1} VG2=+{vg2:.2f} ({len(Vd_m)} pts) meas-fold={meas_fold:.2f}d")
        per_cfg = {}
        for name, kw in configs:
            t1 = time.time()
            try:
                out = run_config(forward_2t, cfg, M1, M2, bjt, bjt_lat,
                                 sd_M1, sd_M2, vg1, vg2, Vd_t, P_M1, P_M2, **kw)
                Id_p = np.abs(out["Id"].detach().cpu().numpy())
                Ic_Q2 = np.abs(out["components"].get("Ic_Q2",
                                  torch.zeros(1)).detach().cpu().numpy()) if "components" in out else None
                conv = bool(np.all(out["converged"])) if not isinstance(out["converged"], bool) else out["converged"]
                finite = bool(np.all(np.isfinite(Id_p)))
                r_all, n = rmse_dec(Id_p, Id_m)
                fold = fold_dec(Id_p, Vd_m, vmin=0.5)
                dt = time.time() - t1
                per_cfg[name] = {"Id_p": Id_p, "rmse": r_all, "n": n, "fold": fold,
                                 "converged": conv, "finite": finite, "dt_s": dt,
                                 "Ic_Q2_max": float(np.nanmax(Ic_Q2)) if Ic_Q2 is not None else None}
                log(f"    {name:20s} fold={fold:+.2f}d rmse={r_all:.2f}d "
                    f"conv={conv} finite={finite} ({dt:.1f}s)"
                    + (f" Ic_Q2_max={per_cfg[name]['Ic_Q2_max']:.2e}A" if per_cfg[name]['Ic_Q2_max'] else ""))
                ax.semilogy(Vd_m, np.maximum(Id_p, 1e-15),
                            color=cfg_colors[name], lw=1.3,
                            label=f"{name} | fold={fold:.2f}d rmse={r_all:.2f}d")
            except Exception as e:
                dt = time.time() - t1
                log(f"    {name:20s} ERROR after {dt:.1f}s: {type(e).__name__}: {e}")
                per_cfg[name] = {"error": f"{type(e).__name__}: {e}", "dt_s": dt}

        ax.semilogy(Vd_m, np.maximum(Id_m, 1e-15), "k.", ms=4, label="measured (Sebas)")
        ax.set_xlabel("Vd (V)"); ax.set_ylabel("|Id| (A)")
        ax.set_ylim(1e-13, 1e-2)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(f"VG1={vg1}, VG2=+{vg2:.2f}  meas-fold={meas_fold:.2f}d")
        ax.legend(loc="lower right", fontsize=6.5)

        serial = {}
        for name, d in per_cfg.items():
            sd = dict(d); sd.pop("Id_p", None)
            serial[name] = sd
        results.append({"VG1": vg1, "VG2": vg2, "file": fname,
                        "measured_fold_dec": meas_fold,
                        "configs": serial,
                        "R46_params": {"Bf": Bf_v, "iii_body_gain": iii_v, "vnwell_Rs": Rs_v}})

    fig.suptitle("z387 hypothesis 2: lateral NPN (M1.D ↔ body ↔ M2.S=GND)",
                 fontsize=11, y=1.00)
    fig.tight_layout()
    out_png = OUT / "ids_compare.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    log(f"wrote {out_png}")

    # Gate evaluation
    cfg_names = [n for n, _ in configs]
    lat_names = [n for n in cfg_names if n != "baseline_no_lat"]
    def get_06(name):
        r06 = next(r for r in results if abs(r["VG1"] - 0.6) < 1e-3)
        return r06["configs"][name]

    infra = True
    for r in results:
        for name in cfg_names:
            c = r["configs"][name]
            if "error" in c or not c.get("finite", False):
                infra = False
    fold_06 = {name: get_06(name).get("fold", float("nan")) for name in lat_names}
    discovery = any(f > 0.5 for f in fold_06.values() if math.isfinite(f))

    # AMBITIOUS: cell-wide median rmse across configs
    cell_rmse = {}
    for name in cfg_names:
        vals = []
        for r in results:
            c = r["configs"][name]
            if "rmse" in c and math.isfinite(c["rmse"]):
                vals.append(c["rmse"])
        cell_rmse[name] = float(np.median(vals)) if vals else float("nan")
    ambitious = any(v < 0.5 for v in cell_rmse.values() if math.isfinite(v))
    kill_shot = not discovery

    log(f"GATE INFRA      : {'PASS' if infra else 'FAIL'}")
    log(f"GATE DISCOVERY  : {'PASS' if discovery else 'FAIL'}  fold06={fold_06}")
    log(f"GATE AMBITIOUS  : {'PASS' if ambitious else 'FAIL'}  cell_rmse={cell_rmse}")
    log(f"GATE KILL_SHOT  : {'PASS' if kill_shot else 'FAIL'} (BJT orientation not it)")

    summary = {
        "wallclock_s": time.time() - t0,
        "R46_best_x": X_BEST,
        "configs_tested": cfg_names,
        "results": results,
        "gates": {"infra": infra, "discovery": discovery,
                  "ambitious": ambitious, "kill_shot": kill_shot,
                  "fold_VG1_0.6": fold_06,
                  "cell_rmse_median": cell_rmse},
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    with open(OUT / "run.log", "w") as f:
        f.write("\n".join(LOG_LINES) + "\n")
    log(f"DONE in {time.time()-t0:.1f}s. wrote summary.json, run.log")


if __name__ == "__main__":
    main()
