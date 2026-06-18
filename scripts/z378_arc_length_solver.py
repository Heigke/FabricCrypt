"""z378 — S2 arc-length continuation solver validation.

Compare cold-start 2D Newton (baseline `forward_2t`) vs pseudo-arclength
continuation (`forward_2t_arclength_grad`) at R-46 best per-VG1 params
across all 33 Sebas (VG1, VG2) biases.

S1 (z377) showed Ids(Vb=0)=3e-12 -> Ids(Vb=0.8)=1e-6 = 5.5 dec jump at
Vd=1.5V VG1=0.6 VG2=+0.2: the BSIM4 physics IS in the model, but plain
Newton from cold-start cannot navigate past the snapback fold. This script
verifies whether arc-length continuation actually crosses that fold.

Outputs results/z378_arc_length_solver/{summary.json, snapback_arc_length_compare.png, run.log}.
"""
from __future__ import annotations
import os, sys, json, re, math, csv, time, importlib.util, traceback
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "nsram"))
OUT = ROOT / "results/z378_arc_length_solver"; OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data/sebas_2026_04_22"

LOG_F = open(OUT / "run.log", "w", buffering=1)
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_F.write(line + "\n")


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
    rows = []
    with open(DATA / "2Tcell_BSIM_param_DC.csv") as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                try: row[k] = float(v)
                except ValueError: row[k] = float("nan")
            rows.append(row)
    return rows


BRANCH_FLAT = {
    0.4: {"ETAB": 1.9,  "K1": 0.53825, "ALPHA0": 7.842e-05, "BETA0": 19.0, "NFACTOR": 6.0},
    0.6: {"ETAB": 2.5,  "K1": 0.41825, "ALPHA0": 7.842e-05, "BETA0": 20.0, "NFACTOR": 6.0},
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


def list_all_biases():
    """Return list of (VG1, VG2, csv_path) for all 33 Sebas IV files."""
    biases = []
    for vg1 in (0.2, 0.4, 0.6):
        sub = DATA / f"2vHCa-2 I-Vs@VG2 VG1={vg1} vnwell=2"
        for f in sorted(sub.glob("*.csv")):
            m = re.search(r"VG2=(-?\d+\.\d+)_VG=", f.name)
            if not m: continue
            vg2 = float(m.group(1))
            biases.append((vg1, vg2, f))
    return biases


def load_csv(path):
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    return d[:, 0], np.abs(d[:, 1])


def decade_rmse(Id_pred, Id_meas, floor=1e-15):
    mask = (Id_meas > floor) & (Id_pred > floor) & np.isfinite(Id_pred)
    if mask.sum() < 3:
        return float("nan"), int(mask.sum())
    rm = float(np.sqrt(np.mean((np.log10(Id_pred[mask]) - np.log10(Id_meas[mask]))**2)))
    return rm, int(mask.sum())


def max_forward_jump(Id, Vd, Vd_min=0.5, floor=1e-15):
    if len(Id) < 2: return float("nan")
    dlog = np.diff(np.log10(np.maximum(Id, floor)))
    Vmid = 0.5 * (Vd[1:] + Vd[:-1])
    sel = Vmid >= Vd_min
    if not sel.any() or len(dlog) == 0: return float("nan")
    masked = np.where(sel, dlog, -np.inf)
    return float(masked.max())


PER_VG1 = {
    0.2: (1889.88, 1.8447, 9.1722),
    0.4: (1092.27, 1.5152, 9.8983),
    0.6: ( 417.63, 0.9036, 6.7846),
}


def run_baseline(cfg, M1, M2, bjt, Vd_t, vg1, vg2, P_M1, P_M2, sd_M1, sd_M2):
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    Bf, iii, Rs = PER_VG1[vg1]
    bjt.Bf = Bf; cfg.iii_body_gain = iii; cfg.vnwell_Rs = Rs
    with patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
        out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd_t,
                         VG1=torch.tensor(vg1, dtype=torch.float64),
                         VG2=torch.tensor(vg2, dtype=torch.float64),
                         warm_start=True)
    return out


def run_arclength(cfg, M1, M2, bjt, Vd_t, vg1, vg2, P_M1, P_M2, sd_M1, sd_M2):
    """Custom arclength runner: clamp Vd_start to >= 0.05 V (avoids _solve_initial_point
    failure at Vd~0) and interpolate at original Vd_seq. For Vd < Vd_start, fall back
    to plain cold-start 2D Newton (those points have Ids ~ 0 anyway)."""
    from nsram.bsim4_port.arclength import (trace_arclength, _trace_backward,
                                             interpolate_at_targets, _merge_paths)
    from nsram.bsim4_port.nsram_cell_2T import solve_2t_steady_state
    Bf, iii, Rs = PER_VG1[vg1]
    bjt.Bf = Bf; cfg.iii_body_gain = iii; cfg.vnwell_Rs = Rs
    VG1_t = torch.tensor(vg1, dtype=torch.float64)
    VG2_t = torch.tensor(vg2, dtype=torch.float64)
    Vd_min_f = float(Vd_t.min())
    Vd_max_f = float(Vd_t.max())
    Vd_start = max(0.05, Vd_min_f)

    with patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
        # SEED via production Newton (solve_2t_steady_state has damping/clamping
        # that the bare _solve_initial_point Newton lacks — at R-46 params with
        # large vnwell_Rs the bare Newton diverges to Vb=-1e10).
        # NOTE: pass P_M1=None to solve_2t/trace_arclength because we already
        # patched sd.scaled via patch_sd_scaled context.
        with torch.no_grad():
            seed_out = solve_2t_steady_state(
                cfg, M1, bjt,
                Vd=torch.tensor([Vd_start], dtype=torch.float64),
                VG1=VG1_t, VG2=VG2_t,
                P_M1=None, P_M2=None,
                Vsint_init=torch.tensor([0.5*Vd_start], dtype=torch.float64),
                Vb_init=torch.tensor([0.5], dtype=torch.float64),
                model_M2=M2)
            seed_Vsint = float(seed_out["Vsint"].squeeze().item())
            seed_Vb = float(seed_out["Vb"].squeeze().item())
            path = trace_arclength(cfg, M1, bjt, VG1_t, VG2_t,
                                    Vd_start=Vd_start, Vd_max=Vd_max_f,
                                    P_M1=None, P_M2=None, model_M2=M2,
                                    Vsint_init=seed_Vsint, Vb_init=seed_Vb)
            init_ok = path.get("init_ok", False)
            max_reached = max(path["path_Vd"]) if (init_ok and path["path_Vd"]) else -1e9
            if not init_ok or max_reached < Vd_max_f - 1e-3:
                bwd = _trace_backward(cfg, M1, bjt, VG1_t, VG2_t,
                                       Vd_start=Vd_max_f, Vd_min=Vd_start,
                                       P_M1=None, P_M2=None, model_M2=M2)
                if bwd.get("init_ok", False):
                    if not init_ok:
                        path = bwd
                    else:
                        path = _merge_paths(path, bwd)
            if not path.get("init_ok", False):
                return {"Id": torch.full_like(Vd_t, float("nan")),
                        "Vsint": torch.full_like(Vd_t, float("nan")),
                        "Vb": torch.full_like(Vd_t, float("nan")),
                        "converged": torch.zeros_like(Vd_t, dtype=torch.bool),
                        "arclen_n_steps": int(path.get("n_steps", 0)),
                        "arclen_n_folds": int(path.get("n_folds", 0))}
            warm = interpolate_at_targets(path, Vd_t)

        Ids_list, Vs_list, Vb_list, conv_list = [], [], [], []
        Vs_cold = (Vd_t * 0.5).detach()
        for i in range(int(Vd_t.shape[0])):
            Vd_i = Vd_t[i].unsqueeze(0)
            vd_scalar = float(Vd_i.item())
            if vd_scalar < Vd_start - 1e-6 or not bool(warm["converged"][i]):
                # Below arclength start: cold-start plain Newton
                Vs0 = (Vd_i * 0.5).detach()
                Vb0 = torch.tensor(0.5, dtype=torch.float64)
            else:
                Vs0 = warm["Vsint"][i].unsqueeze(0).detach()
                Vb0 = warm["Vb"][i].unsqueeze(0).detach()
            out = solve_2t_steady_state(
                cfg, M1, bjt, Vd=Vd_i, VG1=VG1_t, VG2=VG2_t,
                P_M1=None, P_M2=None,
                Vsint_init=Vs0, Vb_init=Vb0, model_M2=M2)
            Ids_list.append(out["Id"].squeeze(0))
            Vs_list.append(out["Vsint"].squeeze(0))
            Vb_list.append(out["Vb"].squeeze(0))
            conv_val = out["converged"]
            if isinstance(conv_val, torch.Tensor):
                conv_val = bool(conv_val.squeeze(0).all().item()) if conv_val.numel() > 0 else False
            conv_list.append(bool(conv_val))

    return {"Id": torch.stack(Ids_list),
            "Vsint": torch.stack(Vs_list),
            "Vb": torch.stack(Vb_list),
            "converged": torch.tensor(conv_list, dtype=torch.bool),
            "arclen_n_steps": int(path.get("n_steps", 0)),
            "arclen_n_folds": int(path.get("n_folds", 0))}


def main():
    t0 = time.time()
    log("z378 — S2 arc-length solver validation")
    log("Gates: INFRA=<30min,no nan; DISCOVERY cell<0.85dec AND any VG1 jump>0.5dec; "
        "AMBITIOUS cell<0.5dec AND VG1=0.6 jump>1.5dec; KILL-SHOT=arclength conv but no fold")
    cfg, M1, M2, bjt = build_base()
    sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)
    rows = load_sebas_params()
    biases = list_all_biases()
    log(f"Found {len(biases)} biases across VG1∈{{0.2,0.4,0.6}}")

    per_bias = []
    for k, (vg1, vg2, fpath) in enumerate(biases):
        try:
            Vd_m, Id_m = load_csv(fpath)
            Vd_t = torch.tensor(Vd_m, dtype=torch.float64)
            row = find_or_impute_row(rows, vg1, vg2)
            P_M1, P_M2 = make_overrides(row)

            t_b0 = time.time()
            out_base = run_baseline(cfg, M1, M2, bjt, Vd_t, vg1, vg2, P_M1, P_M2, sd_M1, sd_M2)
            t_b1 = time.time()
            Id_base = np.abs(out_base["Id"].detach().cpu().numpy())
            base_conv = out_base.get("converged", None)
            base_conv_n = int(sum(1 for c in base_conv if bool(c))) if base_conv is not None else -1

            t_a0 = time.time()
            try:
                out_arc = run_arclength(cfg, M1, M2, bjt, Vd_t, vg1, vg2, P_M1, P_M2, sd_M1, sd_M2)
                arc_err = None
                Id_arc = np.abs(out_arc["Id"].detach().cpu().numpy())
                arc_conv = out_arc.get("converged", None)
                arc_conv_n = int(arc_conv.sum().item()) if isinstance(arc_conv, torch.Tensor) else -1
                arc_n_steps = int(out_arc.get("arclen_n_steps", -1) or -1)
                arc_n_folds = int(out_arc.get("arclen_n_folds", -1) or -1)
            except Exception as e:
                arc_err = f"{type(e).__name__}: {e}"
                log(f"  ARC fail: {arc_err}")
                Id_arc = np.full_like(Id_m, np.nan)
                arc_conv_n = -1; arc_n_steps = -1; arc_n_folds = -1
            t_a1 = time.time()

            rmse_base, n_base = decade_rmse(Id_base, Id_m)
            rmse_arc,  n_arc  = decade_rmse(Id_arc,  Id_m)
            jump_meas = max_forward_jump(Id_m,    Vd_m)
            jump_base = max_forward_jump(Id_base, Vd_m)
            jump_arc  = max_forward_jump(Id_arc,  Vd_m)

            per_bias.append({
                "VG1": float(vg1), "VG2": float(vg2),
                "file": fpath.name,
                "n_points": int(len(Vd_m)),
                "rmse_base_dec": rmse_base,
                "rmse_arc_dec":  rmse_arc,
                "improvement_dec": (rmse_base - rmse_arc) if (math.isfinite(rmse_base) and math.isfinite(rmse_arc)) else float("nan"),
                "jump_meas_dec": jump_meas,
                "jump_base_dec": jump_base,
                "jump_arc_dec":  jump_arc,
                "base_conv_count": base_conv_n,
                "arc_conv_count":  arc_conv_n,
                "arc_n_steps":     arc_n_steps,
                "arc_n_folds":     arc_n_folds,
                "t_base_s": t_b1 - t_b0,
                "t_arc_s":  t_a1 - t_a0,
                "arc_err":  arc_err,
            })
            log(f"  [{k+1:2d}/{len(biases)}] VG1={vg1} VG2={vg2:+.2f}: "
                f"rmse base={rmse_base:.2f} -> arc={rmse_arc:.2f} dec | "
                f"jump meas={jump_meas:.2f} base={jump_base:.2f} arc={jump_arc:.2f} | "
                f"arc steps={arc_n_steps} folds={arc_n_folds} | "
                f"dt={t_a1-t_a0:.1f}s")
        except Exception as e:
            traceback.print_exc()
            log(f"  [{k+1}] FAIL: {e}")
            per_bias.append({"VG1": float(vg1), "VG2": float(vg2),
                             "file": fpath.name, "error": str(e)})

    # Aggregate
    def vals(key, finite_only=True):
        out = [b.get(key) for b in per_bias if isinstance(b.get(key), (int, float))]
        if finite_only:
            out = [v for v in out if math.isfinite(v)]
        return out

    rmse_base_all = vals("rmse_base_dec")
    rmse_arc_all  = vals("rmse_arc_dec")
    cellwide_base = float(np.median(rmse_base_all)) if rmse_base_all else float("nan")
    cellwide_arc  = float(np.median(rmse_arc_all))  if rmse_arc_all  else float("nan")

    # Per-VG1 max model jump (best snapback in arc result)
    jump_arc_by_vg1 = {}
    jump_meas_by_vg1 = {}
    for vg1 in (0.2, 0.4, 0.6):
        rows_v = [b for b in per_bias if b.get("VG1") == vg1 and isinstance(b.get("jump_arc_dec"), (int, float))]
        if rows_v:
            jump_arc_by_vg1[vg1] = max(b["jump_arc_dec"] for b in rows_v if math.isfinite(b["jump_arc_dec"]) ) if any(math.isfinite(b["jump_arc_dec"]) for b in rows_v) else float("nan")
            jump_meas_by_vg1[vg1] = max((b["jump_meas_dec"] for b in rows_v if math.isfinite(b.get("jump_meas_dec", float("nan")))), default=float("nan"))
        else:
            jump_arc_by_vg1[vg1] = float("nan")
            jump_meas_by_vg1[vg1] = float("nan")

    any_vg1_jump_gt_p5 = any(math.isfinite(v) and v > 0.5 for v in jump_arc_by_vg1.values())
    vg1_06_jump = jump_arc_by_vg1.get(0.6, float("nan"))
    nan_count = sum(1 for b in per_bias if not math.isfinite(b.get("rmse_arc_dec", float("nan"))))

    wall_s = time.time() - t0
    gate_infra = (nan_count == 0 and wall_s < 30 * 60)
    gate_disc = (math.isfinite(cellwide_arc) and cellwide_arc < 0.85 and any_vg1_jump_gt_p5)
    gate_amb = (math.isfinite(cellwide_arc) and cellwide_arc < 0.5
                and math.isfinite(vg1_06_jump) and vg1_06_jump > 1.5)
    # KILL-SHOT: arc convergence but no fold => arc reports n_folds=0 across the
    # board AND max jump small everywhere
    folds_total = sum(b.get("arc_n_folds", 0) or 0 for b in per_bias if isinstance(b.get("arc_n_folds"), int))
    gate_kill = (not any_vg1_jump_gt_p5 and folds_total == 0)

    if gate_amb: verdict = "AMBITIOUS"
    elif gate_disc: verdict = "DISCOVERY"
    elif gate_kill: verdict = "KILL_SHOT"
    elif gate_infra: verdict = "INFRA_ONLY"
    else: verdict = "INFRA_FAIL"

    summary = {
        "script": "z378_arc_length_solver",
        "wall_s": wall_s,
        "n_biases": len(biases),
        "cellwide_median_rmse_base_dec": cellwide_base,
        "cellwide_median_rmse_arc_dec":  cellwide_arc,
        "improvement_dec": cellwide_base - cellwide_arc if (math.isfinite(cellwide_base) and math.isfinite(cellwide_arc)) else float("nan"),
        "max_jump_arc_by_VG1": jump_arc_by_vg1,
        "max_jump_meas_by_VG1": jump_meas_by_vg1,
        "any_VG1_arc_jump_gt_0.5": any_vg1_jump_gt_p5,
        "vg1_06_arc_jump_dec": vg1_06_jump,
        "nan_count": nan_count,
        "folds_detected_total": folds_total,
        "gate_INFRA": gate_infra,
        "gate_DISCOVERY": gate_disc,
        "gate_AMBITIOUS": gate_amb,
        "gate_KILL_SHOT": gate_kill,
        "verdict": verdict,
        "per_VG1_params": {f"{k}": list(v) for k, v in PER_VG1.items()},
        "per_bias": per_bias,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log(f"\n=== SUMMARY ===")
    log(f"  Cell-wide median RMSE: baseline={cellwide_base:.3f} dec | arclength={cellwide_arc:.3f} dec")
    log(f"  Per-VG1 arc max jump: {jump_arc_by_vg1}")
    log(f"  Folds detected total: {folds_total}; NaN count: {nan_count}")
    log(f"  Verdict: {verdict}")
    log(f"  Wall: {wall_s:.1f}s")

    # Plot: 3-panel snapback compare (VG2=+0.20 representative biases),
    # measured vs baseline vs arc-length.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, vg1 in zip(axes, (0.2, 0.4, 0.6)):
        # VG1=0.2 only has VG2 up to +0.10
        vg2 = 0.10 if vg1 == 0.2 else 0.20
        cand = [b for b in per_bias if b.get("VG1") == vg1 and abs(b.get("VG2", 99) - vg2) < 1e-3]
        if not cand:
            ax.set_title(f"VG1={vg1} (no data)"); continue
        b = cand[0]
        # Re-load the curves to plot (we didn't store them in per_bias)
        fpath = DATA / f"2vHCa-2 I-Vs@VG2 VG1={vg1} vnwell=2" / b["file"]
        Vd_m, Id_m = load_csv(fpath)
        Vd_t = torch.tensor(Vd_m, dtype=torch.float64)
        row = find_or_impute_row(rows, vg1, vg2)
        P_M1, P_M2 = make_overrides(row)
        out_base = run_baseline(cfg, M1, M2, bjt, Vd_t, vg1, vg2, P_M1, P_M2, sd_M1, sd_M2)
        Id_base = np.abs(out_base["Id"].detach().cpu().numpy())
        try:
            out_arc = run_arclength(cfg, M1, M2, bjt, Vd_t, vg1, vg2, P_M1, P_M2, sd_M1, sd_M2)
            Id_arc = np.abs(out_arc["Id"].detach().cpu().numpy())
        except Exception:
            Id_arc = np.full_like(Id_m, np.nan)
        ax.semilogy(Vd_m, np.maximum(Id_m, 1e-15), "k.", ms=4, label="measured")
        ax.semilogy(Vd_m, np.maximum(Id_base, 1e-15), "b-", lw=1.2, alpha=0.7,
                    label=f"2D Newton (RMSE={b['rmse_base_dec']:.2f})")
        ax.semilogy(Vd_m, np.maximum(Id_arc, 1e-15), "r-", lw=1.6,
                    label=f"arc-length (RMSE={b['rmse_arc_dec']:.2f})")
        ax.set_xlabel("Vd (V)"); ax.set_ylabel("|Id| (A)")
        ax.set_ylim(1e-13, 1e-2); ax.grid(True, which="both", alpha=0.3)
        ax.set_title(f"VG1={vg1} VG2=+{vg2:.2f}\n"
                     f"jump meas={b['jump_meas_dec']:.1f} | base={b['jump_base_dec']:.1f} | arc={b['jump_arc_dec']:.1f} dec\n"
                     f"folds={b['arc_n_folds']} steps={b['arc_n_steps']}", fontsize=9)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(f"z378 — arc-length vs 2D Newton @ R-46 best | "
                 f"cell base={cellwide_base:.2f} -> arc={cellwide_arc:.2f} dec | verdict={verdict}",
                 fontsize=11, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "snapback_arc_length_compare.png", dpi=150, bbox_inches="tight")
    log(f"Wrote {OUT/'snapback_arc_length_compare.png'}")
    LOG_F.close()


if __name__ == "__main__":
    main()
