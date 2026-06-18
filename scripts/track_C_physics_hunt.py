"""Track C — Physics Hunt: 5-axis sweep over candidate missing physics.

Axes (5 values per axis → 3125 cells total):
  selfheat_kappa  ∈ {0, 1e3, 5e3, 1e4, 1e5}    [K/W]   — Tj rise = κ·Pdiss
  hurkx_beta      ∈ {0, 1e-7, 1e-6, 1e-5, 1e-4}        — TAT enhancement on M1 alpha0
  bbt_alpha       ∈ {0, 1e-12, 1e-10, 1e-9, 1e-8}      — additive BBT current (A·V^-3)
  rb_nonuniform_f ∈ {0, 0.1, 0.3, 0.7, 1.0}            — NPN Bf log-falloff factor
  agidl_scale     ∈ {1, 3, 10, 20, 30}                  — GIDL prefactor multiplier on M2

Eval: 11 representative biases × decimated Vd (every 6th pt → ~14 pts each).
       Per cell ≈ 1-2s CPU. Parallel across N_WORKERS processes.

Outputs:
  results/physics_hunt_track_C/sweep_results.json  (all cells)
  results/physics_hunt_track_C/pareto.md           (top 20 + axis sensitivity)
"""
from __future__ import annotations
import os, sys, time, json, re, itertools, importlib.util
for _k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import multiprocessing as mp
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))

OUT = ROOT / "results/physics_hunt_track_C"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data/sebas_2026_04_22"

DTYPE = torch.float64

# Decimation: keep every K-th Vd sample
VD_STRIDE = 8
# Choose ~11 representative biases (subsample 33 → 11 spanning VG1∈{0.2,0.4,0.6} × VG2 grid)
REPRESENTATIVE_VG2 = [-0.10, 0.00, 0.10, 0.20, 0.30]  # 5 VG2 levels × 3 VG1 = 15 curves; only ones present


def _build_baseline():
    sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
    v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
    from nsram.bsim4_port.bjt import GummelPoonNPN
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                            newton_max_iters=25, Iabstol=1e-11)
    cfg.bjt_emitter_to_gnd = True
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = 9000.0; bjt.Va = 0.55; bjt.Is = 1e-9
    return cfg, M1, M2, bjt


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
            try:
                d = np.loadtxt(f, delimiter=",", skiprows=1)
            except Exception:
                continue
            if d.ndim != 2 or d.shape[1] < 2: continue
            curves.append({"VG1": vg1, "VG2": vg2, "Vd": d[:,0], "Id": np.abs(d[:,1]),
                           "label": f"VG1={vg1:.2f}_VG2={vg2:+.2f}"})
    return curves


def subsample_curves(curves):
    """Keep only representative VG2 values; decimate Vd by VD_STRIDE."""
    out = []
    for c in curves:
        # find nearest representative VG2
        if any(abs(c["VG2"]-x) < 0.005 for x in REPRESENTATIVE_VG2):
            sub = {
                "VG1": c["VG1"], "VG2": c["VG2"], "label": c["label"],
                "Vd": c["Vd"][::VD_STRIDE].copy(),
                "Id": c["Id"][::VD_STRIDE].copy(),
            }
            if len(sub["Vd"]) >= 5:
                out.append(sub)
    return out


# --------------------------------------------------------------------------
# Per-cell evaluation (worker)
# --------------------------------------------------------------------------
_W_STATE = {}

def _w_init():
    cfg, M1, M2, bjt = _build_baseline()
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    _W_STATE["cfg"] = cfg; _W_STATE["M1"] = M1; _W_STATE["M2"] = M2; _W_STATE["bjt"] = bjt
    _W_STATE["forward_2t"] = forward_2t
    _W_STATE["snap"] = {
        "M1_alpha0": M1._values.get("alpha0", 0.0),
        "M1_beta0":  M1._values.get("beta0", 0.0),
        "M2_agidl":  M2._values.get("agidl", 0.0),
        "bjt_Bf":    bjt.Bf,
        "bjt_Is":    bjt.Is,
    }
    _W_STATE["curves"] = subsample_curves(load_curves())


def _apply_cell(snap, kappa, hurkx_b, agidl_s, rb_f):
    cfg = _W_STATE["cfg"]; M1 = _W_STATE["M1"]; M2 = _W_STATE["M2"]; bjt = _W_STATE["bjt"]
    if hurkx_b > 0:
        boost = 1.0 + hurkx_b * 1e6
        M1._values["alpha0"] = snap["M1_alpha0"] * boost
    else:
        M1._values["alpha0"] = snap["M1_alpha0"]
    M1._values["beta0"] = snap["M1_beta0"]
    M2._values["agidl"] = snap["M2_agidl"] * agidl_s
    if rb_f > 0:
        bjt.Bf = snap["bjt_Bf"] * max(0.05, 1.0 - rb_f * 0.5)
    else:
        bjt.Bf = snap["bjt_Bf"]
    cfg.T_C = 27.0


def _eval_cell(kappa, hurkx_b, bbt_a, rb_f, agidl_s):
    cfg = _W_STATE["cfg"]; M1 = _W_STATE["M1"]; M2 = _W_STATE["M2"]; bjt = _W_STATE["bjt"]
    forward_2t = _W_STATE["forward_2t"]; curves = _W_STATE["curves"]
    snap = _W_STATE["snap"]
    _apply_cell(snap, kappa, hurkx_b, agidl_s, rb_f)

    per_curve_rmse = []
    n_conv = 0; n_pts = 0
    for c in curves:
        # Simple self-heat heuristic: at high Vd, T_C += κ·(Vd·1e-7)
        if kappa > 0:
            Vd_max = float(np.max(np.abs(c["Vd"])))
            cfg.T_C = 27.0 + float(np.clip(kappa * Vd_max * 1e-7, 0.0, 150.0))
        else:
            cfg.T_C = 27.0
        Vd_np = c["Vd"]; Id_meas = c["Id"]
        Vd_t = torch.tensor(Vd_np, dtype=DTYPE)
        try:
            out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd_t,
                             VG1=torch.tensor(c["VG1"], dtype=DTYPE),
                             VG2=torch.tensor(c["VG2"], dtype=DTYPE),
                             warm_start=True)
            Id_pred = torch.abs(out["Id"]).detach().cpu().numpy()
            conv = out.get("converged", None)
            if conv is not None:
                if hasattr(conv, "detach"):
                    cv = conv.detach().cpu().numpy()
                else:
                    cv = np.asarray(conv)
                n_conv += int(np.asarray(cv).sum()); n_pts += int(np.asarray(cv).size)
            else:
                n_pts += Id_pred.size; n_conv += Id_pred.size
            if bbt_a > 0:
                Id_pred = Id_pred + bbt_a * np.abs(Vd_np)**3
            mask = (Id_meas > 1e-15) & (Id_pred > 1e-15) & np.isfinite(Id_pred)
            if mask.sum() < 3:
                per_curve_rmse.append(np.nan); continue
            rmse = float(np.sqrt(np.mean((np.log10(Id_pred[mask]) - np.log10(Id_meas[mask]))**2)))
            if not np.isfinite(rmse) or rmse > 99.0:
                rmse = np.nan
            per_curve_rmse.append(rmse)
        except Exception:
            per_curve_rmse.append(np.nan)
    valid = [x for x in per_curve_rmse if not np.isnan(x)]
    if not valid:
        return {"median_dec": 99.9, "valid_fraction": 0.0, "conv_rate": 0.0, "INVALID": True}
    return {
        "median_dec": float(np.median(valid)),
        "valid_fraction": len(valid)/len(per_curve_rmse),
        "conv_rate": (n_conv / max(1, n_pts)),
        "INVALID": False,
    }


def _worker(task):
    idx, params = task
    r = _eval_cell(**params)
    return idx, {**params, **r}


# --------------------------------------------------------------------------
def main():
    t0 = time.time()
    n_workers = int(os.environ.get("TRACK_C_WORKERS", "16"))
    print(f"[track_C] start  n_workers={n_workers}", flush=True)

    # baseline
    _w_init()
    print(f"[track_C] loaded {len(_W_STATE['curves'])} subsampled biases (full=33)", flush=True)
    base = _eval_cell(kappa=0, hurkx_b=0, bbt_a=0, rb_f=0, agidl_s=1.0)
    print(f"[track_C] BASELINE dec={base['median_dec']:.4f} conv={base['conv_rate']:.3f} "
          f"valid={base['valid_fraction']:.2f}", flush=True)

    grid = {
        "kappa":   [0.0, 1e3, 5e3, 1e4, 1e5],
        "hurkx_b": [0.0, 1e-7, 1e-6, 1e-5, 1e-4],
        "bbt_a":   [0.0, 1e-12, 1e-10, 1e-9, 1e-8],
        "rb_f":    [0.0, 0.1, 0.3, 0.7, 1.0],
        "agidl_s": [1.0, 3.0, 10.0, 20.0, 30.0],
    }
    axes = list(grid.keys())
    points = list(itertools.product(*[grid[a] for a in axes]))
    tasks = [(i, dict(zip(axes, p))) for i, p in enumerate(points)]
    print(f"[track_C] sweep {len(tasks)} cells", flush=True)

    all_results = [None]*len(tasks)
    t_sweep = time.time()
    with mp.Pool(n_workers, initializer=_w_init) as pool:
        for k, (idx, r) in enumerate(pool.imap_unordered(_worker, tasks, chunksize=4)):
            all_results[idx] = r
            if (k+1) % 50 == 0:
                rate = (k+1)/(time.time()-t_sweep)
                eta = (len(tasks)-(k+1))/max(rate,1e-3)
                done = [x for x in all_results if x is not None and x.get("conv_rate",0)>=0.8]
                best = min((c["median_dec"] for c in done), default=99.9)
                print(f"[track_C] {k+1}/{len(tasks)} rate={rate:.2f}/s ETA={eta/60:.1f}m best={best:.3f}",
                      flush=True)
            if (k+1) % 500 == 0:
                (OUT/"sweep_results_partial.json").write_text(json.dumps({
                    "n_done": k+1, "cells": [c for c in all_results if c is not None],
                }, indent=1))

    all_results = [c for c in all_results if c is not None]
    valid_cells = sorted([c for c in all_results if c["conv_rate"]>=0.8 and not c["INVALID"]],
                         key=lambda c: c["median_dec"])
    if not valid_cells:
        valid_cells = sorted([c for c in all_results if not c["INVALID"]],
                             key=lambda c: c["median_dec"])

    best = valid_cells[0] if valid_cells else None
    sweep_blob = {
        "baseline": base, "axes": axes, "grid": grid,
        "n_total": len(all_results), "n_valid": len(valid_cells),
        "best": best,
        "subsample_info": {"n_biases_used": len(_W_STATE["curves"]), "VD_STRIDE": VD_STRIDE},
        "elapsed_sweep_s": time.time()-t_sweep,
        "cells": all_results,
    }
    (OUT/"sweep_results.json").write_text(json.dumps(sweep_blob, indent=1))
    print(f"[track_C] sweep done in {time.time()-t_sweep:.1f}s; best dec={best['median_dec']:.4f}" if best else "[track_C] no valid cells", flush=True)

    # ------------------------------------------------------------------
    # Axis sensitivity at best cell (single process, 20 pts each axis)
    # ------------------------------------------------------------------
    sensitivity = {}
    if best:
        ranges = {
            "kappa":   np.linspace(0.0, 1.5e5, 20),
            "hurkx_b": np.linspace(0.0, 2e-4, 20),
            "bbt_a":   np.concatenate([[0.0], np.geomspace(1e-14, 1e-7, 19)]),
            "rb_f":    np.linspace(0.0, 1.0, 20),
            "agidl_s": np.geomspace(1.0, 50.0, 20),
        }
        for ax in axes:
            t1 = time.time()
            curve_pts = []
            for v in ranges[ax]:
                d = {a: best[a] for a in axes}
                d[ax] = float(v)
                r = _eval_cell(**{k:d[k] for k in axes})
                curve_pts.append({"value": float(v), "dec": r["median_dec"], "conv": r["conv_rate"]})
            sensitivity[ax] = curve_pts
            print(f"[track_C] axis sens {ax}: {time.time()-t1:.1f}s", flush=True)
        sweep_blob["sensitivity_at_best"] = sensitivity
        (OUT/"sweep_results.json").write_text(json.dumps(sweep_blob, indent=1))

    # ------------------------------------------------------------------
    # Write pareto.md
    # ------------------------------------------------------------------
    md = ["# Physics Hunt — Track C — Pareto Results\n\n"]
    md.append(f"Sweep grid: 5^5 = {len(all_results)} cells; valid (conv≥0.8): {len(valid_cells)}\n\n")
    md.append(f"Subsample: {len(_W_STATE['curves'])}/33 biases (representative VG2 set), Vd decimated by stride={VD_STRIDE}\n\n")
    md.append(f"Baseline (axes off): **median_dec = {base['median_dec']:.4f}**, conv={base['conv_rate']:.3f}\n\n")
    if best:
        delta = base['median_dec'] - best['median_dec']
        md.append(f"Best cell: **median_dec = {best['median_dec']:.4f}** (Δ = {delta:+.4f} dec vs baseline)\n\n")
        md.append("Best parameters:\n\n")
        md.append(f"- selfheat_κ = {best['kappa']:.3g}\n")
        md.append(f"- hurkx_β = {best['hurkx_b']:.3g}\n")
        md.append(f"- bbt_α = {best['bbt_a']:.3g}\n")
        md.append(f"- rb_nonuniform_f = {best['rb_f']:.3g}\n")
        md.append(f"- agidl_scale = {best['agidl_s']:.3g}\n\n")

    md.append("## Top 20 cells (by median_dec, conv≥0.8)\n\n")
    md.append("| rank | dec | conv | valid | selfheat_κ | hurkx_β | bbt_α | rb_f | agidl_s |\n")
    md.append("|---|---|---|---|---|---|---|---|---|\n")
    for i, c in enumerate(valid_cells[:20]):
        md.append(f"| {i+1} | {c['median_dec']:.4f} | {c['conv_rate']:.3f} | "
                  f"{c['valid_fraction']:.2f} | {c['kappa']:.3g} | {c['hurkx_b']:.3g} | "
                  f"{c['bbt_a']:.3g} | {c['rb_f']:.3g} | {c['agidl_s']:.3g} |\n")

    # axis sensitivity table
    if best:
        md.append("\n## Axis-wise sensitivity at best cell (20 fine pts each)\n\n")
        md.append("| axis | dec_min | dec_max | Δ range | d(dec)/d(axis) near best |\n")
        md.append("|---|---|---|---|---|\n")
        for ax in axes:
            pts = sensitivity[ax]
            ys = np.array([p["dec"] for p in pts if p["dec"] < 50.0])
            xs = np.array([p["value"] for p in pts if p["dec"] < 50.0])
            if len(ys) < 3:
                md.append(f"| {ax} | n/a | n/a | n/a | n/a |\n"); continue
            v_best = best[ax]
            i_near = int(np.argmin(np.abs(xs - v_best)))
            i_lo = max(0, i_near-1); i_hi = min(len(xs)-1, i_near+1)
            dydx = (ys[i_hi] - ys[i_lo]) / (xs[i_hi] - xs[i_lo] + 1e-30) if i_hi != i_lo else float("nan")
            md.append(f"| {ax} | {ys.min():.4f} | {ys.max():.4f} | {ys.max()-ys.min():.4f} | {dydx:+.4g} |\n")

        # single-largest contributor
        md.append("\n### Single-largest contributor\n\n")
        contribs = []
        for ax in axes:
            ys = np.array([p["dec"] for p in sensitivity[ax] if p["dec"]<50])
            if len(ys): contribs.append((ax, ys.max()-ys.min()))
        contribs.sort(key=lambda x: -x[1])
        for ax, rng in contribs:
            md.append(f"- {ax}: Δdec range = {rng:.4f}\n")

    md.append(f"\n---\ntotal elapsed: {time.time()-t0:.1f} s\n")
    (OUT/"pareto.md").write_text("".join(md))
    print(f"[track_C] wrote pareto.md + sweep_results.json; total {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
