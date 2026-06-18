"""z13_gpu_solver.py — GPU-parallel branch-following Vbs solver.

Uses torch on the gfx1151 to evaluate thousands of parameter candidates
against all 33 curves simultaneously. Same physics as the numpy
version (full BSIM4 §2.2-§6.1 + parasitic NPN) but vectorised across a
batch dimension B = (param_candidates × curves).

Branch-following strategy:
  For each Vd step (sequentially along the sweep):
    1. Evaluate kcl_net(Vb, params, Vd) on a fine Vb grid for all B
    2. Find the lowest Vb root (or vmax if no root exists)
    3. This vectorised grid search is embarrassingly parallel on GPU
    4. Commit the found Vb, move to next Vd

Run with:
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/z13_gpu_solver.py

After loading z12's best DE parameters this script zooms around them
with a fine grid that would be infeasible on CPU.
"""
from __future__ import annotations

import csv, json, re, time
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from nsram.bsim4 import BSIM4_PRESETS


DATA_DIR = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
                "data/sebas_2026_04_22")
OUT_DIR = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
               "results/z13_gpu_solver")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {DEVICE}")
if DEVICE == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

BASE = BSIM4_PRESETS["ns_ram_130nm_pazos"]
VT = 0.02585         # thermal voltage at 300K
EPS_OX = 3.4528e-11  # ε₀ · ε_r,oxide
COX = EPS_OX / BASE.Toxe

VG_RE = re.compile(r"VG2=(-?\d+\.?\d*)_VG=(\d+\.\d+)")


def load_curves(n_ds: int = 20):
    cv = []
    for sub in sorted(DATA_DIR.iterdir()):
        if not sub.is_dir(): continue
        for fn in sorted(sub.glob("*.csv")):
            m = VG_RE.search(fn.name)
            if not m: continue
            rows = []
            with open(fn) as f:
                rdr = csv.reader(f); next(rdr)
                for r in rdr:
                    try: rows.append((float(r[2]), float(r[0]), float(r[1])))
                    except ValueError: continue
            rows.sort()
            Vd = np.array([r[1] for r in rows])
            Id = np.array([r[2] for r in rows])
            peak = int(np.argmax(Vd))
            Vd = Vd[:peak + 1]; Id = Id[:peak + 1]
            nvd = np.linspace(0.1, float(Vd.max()), n_ds)
            nid = np.interp(nvd, Vd, Id)
            cv.append((float(m.group(2)), float(m.group(1)), nvd, nid))
    return cv


# ──────────────────────────────────────────────────────────────
# Vectorised torch implementation of the physics (batch dim = B)
# ──────────────────────────────────────────────────────────────
def torch_vth(Vbs, Vds, p):
    Vbs_c = torch.minimum(Vbs, torch.as_tensor(BASE.PhiS - 1e-3, device=Vbs.device))
    delta_phi = torch.sqrt(BASE.PhiS - Vbs_c) - np.sqrt(BASE.PhiS)
    Vth = p["VTH0"] + BASE.K1 * delta_phi - BASE.K2 * Vbs_c
    Lr = BASE.Leff / p["LTW"]
    sce_factor = torch.exp(-BASE.DVT1 * Lr / 2.0) + 2.0 * torch.exp(-BASE.DVT1 * Lr)
    sce_amp = 1.0 + BASE.DVT2 * Vbs_c
    Vth = Vth - BASE.DVT0 * sce_factor * sce_amp * (BASE.VBI - BASE.PhiS)
    dibl_factor = torch.exp(-BASE.DSUB * Lr / 2.0)
    dibl_body = 1.0 + BASE.ETAB * Vbs_c
    Vth = Vth - BASE.ETA0 * Vds * dibl_factor * dibl_body
    Vth = Vth - BASE.PDIBLCB * Vbs_c * Vds
    return Vth


def torch_ids(Vgs, Vds, Vbs, p):
    Vth = torch_vth(Vbs, Vds, p)
    n = BASE.NFACTOR
    Vgt = Vgs - Vth - BASE.VOFF
    Vgt_pos = torch.clamp(Vgt, min=0.0)
    Vgt_sub = torch.clamp(Vgt, max=0.0)
    arg_sub = torch.clamp(Vgt_sub / (n * VT), min=-120.0, max=0.0)

    # Mobility degradation
    Eeff = (Vgt_pos + 2.0 * torch.abs(Vth)) / BASE.Toxe
    mob_denom = 1.0 + (BASE.UA_mob * Eeff + BASE.UB_mob * Eeff * Eeff) \
                       * (1.0 + BASE.UC_mob * Vbs)
    mu_eff = BASE.mu0 / torch.clamp(mob_denom, min=1e-3)
    beta = mu_eff * COX * BASE.Weff / BASE.Leff

    Vdsat = torch.clamp(Vgt_pos, min=BASE.Vdsat0 * 1e-3)
    Vdseff = torch.minimum(Vds, Vdsat)
    vds_fac = 1.0 - torch.exp(torch.clamp(-Vds / VT, min=-60.0, max=0.0))

    I_triode = beta * (Vgt_pos * Vdseff - 0.5 * Vdseff * Vdseff)
    I_sat = 0.5 * beta * Vgt_pos * Vgt_pos * (
        1.0 + BASE.lambda_clm * torch.clamp(Vds - Vdsat, min=0.0))
    I_strong = torch.where(Vds < Vdsat, I_triode, I_sat) * vds_fac
    I_sub = beta * (n * VT) ** 2 * np.e * vds_fac * torch.exp(arg_sub)
    Ids = I_strong + I_sub
    return Ids, Vdsat


def torch_iii(Vgs, Vds, Vbs, p):
    Ids, Vdsat = torch_ids(Vgs, Vds, Vbs, p)
    dv = torch.clamp(Vds - Vdsat, min=1e-9)
    prefactor = (p["ALPHA0"] + BASE.ALPHA1 * BASE.Leff) / BASE.Leff
    beta0_eff = BASE.BETA0 * BASE.Leff / (BASE.Leff + p["L_NONLOCAL"])
    exp_arg = torch.clamp(-beta0_eff / dv, min=-80.0, max=0.0)
    return prefactor * dv * torch.exp(exp_arg) * Ids


def torch_gidl(Vds, Vgs, Vbs, p):
    Vdg = Vds - Vgs - BASE.EGIDL
    mask = Vdg > 0
    Vdg_safe = torch.where(mask, Vdg, torch.ones_like(Vdg))
    field = Vdg_safe / (3.0 * BASE.Toxe)
    exp_arg = torch.clamp(-3.0 * BASE.Toxe * BASE.BGIDL / Vdg_safe, min=-80.0, max=0.0)
    Vdb = Vds - Vbs
    vb_term = Vdb ** 3 / (BASE.CGIDL + Vdb ** 3 + 1e-18)
    igidl = BASE.AGIDL * BASE.WeffCJ * BASE.Nf * field * torch.exp(exp_arg) * vb_term
    return torch.where(mask, igidl, torch.zeros_like(Vdg))


def torch_kcl(Vb, Vgs, Vds, p):
    """Net current INTO body node — zeros = steady-state roots."""
    Iii = torch_iii(Vgs, Vds, Vb, p)
    Igidl = torch_gidl(Vds, Vgs, Vb, p)
    IS_eff = BASE.BJT_IS * p["BJT_AREA"]
    Vb_c = torch.minimum(Vb, torch.tensor(BASE.BJT_VJE * 1.1, device=Vb.device))
    exp_arg = torch.clamp(Vb_c / (BASE.BJT_NE * VT), min=-60.0, max=60.0)
    Ib_out = (IS_eff / BASE.BJT_BF) * (torch.exp(exp_arg) - 1.0) + Vb / p["Rb"]
    return (Iii + Igidl) - Ib_out


def branch_follow_batch(p, Vgs_b, Vds_seq_b, n_vgrid: int = 121,
                         vmax: float = 0.85):
    """Vectorised branch-following across batch dim B.

    Inputs (all on DEVICE):
      p          dict of (B,) tensors — per-batch parameter values
      Vgs_b      (B,)
      Vds_seq_b  (B, T)  — Vd sweep per batch element

    Returns: Id_pred (B, T) — predicted drain current across the sweep.
    """
    B, T = Vds_seq_b.shape
    Vb = torch.zeros(B, device=Vds_seq_b.device)
    Id_out = torch.zeros(B, T, device=Vds_seq_b.device)
    v_grid = torch.linspace(0.0, vmax, n_vgrid, device=Vds_seq_b.device)  # (V,)
    for t in range(T):
        Vd_t = Vds_seq_b[:, t]   # (B,)
        # Expand for batched grid evaluation: (B, V)
        Vb_grid = v_grid.unsqueeze(0).expand(B, -1)        # (B, V)
        Vgs_exp = Vgs_b.unsqueeze(1).expand(-1, n_vgrid)   # (B, V)
        Vd_exp  = Vd_t.unsqueeze(1).expand(-1, n_vgrid)    # (B, V)
        p_exp = {k: v.unsqueeze(1).expand(-1, n_vgrid) if torch.is_tensor(v) else v
                  for k, v in p.items()}
        kcl = torch_kcl(Vb_grid, Vgs_exp, Vd_exp, p_exp)  # (B, V)
        # Find smallest Vb where kcl crosses zero from + to −
        # (stable low branch is the smallest positive-slope root)
        sign = torch.sign(kcl)
        sign_change = (sign[:, :-1] > 0) & (sign[:, 1:] <= 0)  # (B, V-1)
        # Index of first such crossing per batch (or -1 if none)
        first_cross = torch.where(
            sign_change.any(dim=1),
            sign_change.float().argmax(dim=1),
            torch.full((B,), n_vgrid - 1, device=Vds_seq_b.device),
        )
        # Fallback: if no crossing, Vb stays at vmax (body fully charged)
        any_cross = sign_change.any(dim=1)
        # Interpolate between v_grid[i] and v_grid[i+1] for sub-grid accuracy
        fi = first_cross.long()
        fi_plus = torch.clamp(fi + 1, max=n_vgrid - 1)
        kcl_lo = torch.gather(kcl, 1, fi.unsqueeze(1)).squeeze(1)
        kcl_hi = torch.gather(kcl, 1, fi_plus.unsqueeze(1)).squeeze(1)
        frac = kcl_lo / (kcl_lo - kcl_hi + 1e-20)
        v_lo = v_grid[fi]
        v_hi = v_grid[fi_plus]
        Vb_new = v_lo + frac * (v_hi - v_lo)
        Vb_new = torch.where(any_cross, Vb_new,
                              torch.full_like(Vb_new, vmax))
        # Branch-following: prefer a root near previous Vb; if our grid
        # search found something far below, keep the previous value
        # (body can't physically drop instantaneously)
        Vb = torch.maximum(Vb, Vb_new * 0 + Vb_new)  # just use Vb_new
        # Compute Id at this Vd
        Ids, _ = torch_ids(Vgs_b, Vd_t, Vb, p)
        # Bipolar collector current
        Iii = torch_iii(Vgs_b, Vd_t, Vb, p)
        Igidl = torch_gidl(Vd_t, Vgs_b, Vb, p)
        Ib_in = Iii + Igidl
        va_corr = 1.0 + Vd_t / BASE.BJT_VA
        IKF_eff = BASE.BJT_IKF * p["BJT_AREA"]
        Ic_raw = BASE.BJT_BF * Ib_in * va_corr
        Ic = Ic_raw / torch.sqrt(
            1.0 + torch.clamp(Ic_raw, min=0.0) / torch.clamp(IKF_eff, min=1e-18))
        Id_out[:, t] = Ids + torch.clamp(Ic, min=0.0)
    return Id_out


def main():
    """Demo: load z12's best, run a fine GPU zoom around it."""
    z12_json = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
                     "results/z12_optimize/summary.json")
    if not z12_json.exists():
        print("z12 summary not yet available — waiting for DE to finish.")
        print("Re-run this script after z12_optimize.py completes.")
        return
    best = json.loads(z12_json.read_text())
    print(f"[z12 best] VTH0={best['VTH0']:.3f} LTW={best['LTW_nm']:.1f}n "
          f"L_nl={best['L_NONLOCAL_nm']:.0f}n AREA={best['BJT_AREA']:.1e} "
          f"Rb={best['Rb']:.0e} ALPHA0={best['ALPHA0']:.2e}")

    curves = load_curves(20)
    n = len(curves)
    Vgs_np = np.array([c[0] for c in curves])
    Vg2_np = np.array([c[1] for c in curves])
    vd_list = [c[2] for c in curves]
    id_list = [c[3] for c in curves]
    T = vd_list[0].size
    Vds_seq = torch.tensor(np.stack(vd_list), dtype=torch.float32, device=DEVICE)
    Id_meas = torch.tensor(np.stack(id_list), dtype=torch.float32, device=DEVICE)
    Vgs_t = torch.tensor(Vgs_np, dtype=torch.float32, device=DEVICE)
    print(f"[curves] {n} × {T}")

    # Zoom grid around z12 best (±20% on Vth, ±50% on lengths, ±1 dec on area/Rb)
    vth_span  = np.linspace(best["VTH0"]*0.85, best["VTH0"]*1.15, 11)
    ltw_span  = np.linspace(max(10e-9, best["LTW_nm"]*0.5e-9),
                              best["LTW_nm"]*2e-9, 9)
    lnl_span  = np.linspace(max(0, best["L_NONLOCAL_nm"]*0.5e-9),
                              best["L_NONLOCAL_nm"]*2e-9 + 50e-9, 9)
    area_span = np.logspace(np.log10(best["BJT_AREA"])-0.75,
                             np.log10(best["BJT_AREA"])+0.75, 7)
    rb_span   = np.logspace(np.log10(best["Rb"])-0.5,
                             np.log10(best["Rb"])+0.5, 5)
    alpha_span = np.logspace(np.log10(best["ALPHA0"])-0.5,
                              np.log10(best["ALPHA0"])+0.5, 5)

    # Full Cartesian product is too big. Sample a random subset from the grid
    rng = np.random.default_rng(0)
    N_SAMPLE = 20000
    combos = []
    for _ in range(N_SAMPLE):
        combos.append([
            rng.choice(vth_span),
            rng.choice(ltw_span),
            rng.choice(lnl_span),
            rng.choice(area_span),
            rng.choice(rb_span),
            rng.choice(alpha_span),
        ])
    combos = np.array(combos, dtype=np.float32)
    print(f"[sweep] {len(combos)} parameter candidates × {n} curves on GPU")

    # Batch: B = Ncombo × Ncurves. For each combo we need Vgs for each curve.
    BATCH_LIM = 2000  # combos per GPU batch to manage memory
    best_rmse = np.inf
    best_params = None
    best_idx = 0
    t0 = time.time()
    total_evals = 0

    # Reshape so one batch = one combo × all curves
    log_meas = torch.log10(torch.clamp(Id_meas, min=1e-20))
    for start in range(0, len(combos), BATCH_LIM):
        end = min(start + BATCH_LIM, len(combos))
        chunk = combos[start:end]
        Nc = chunk.shape[0]
        B = Nc * n
        # Build per-batch parameter tensors
        vth_b = torch.tensor(np.repeat(chunk[:, 0], n), dtype=torch.float32, device=DEVICE)
        ltw_b = torch.tensor(np.repeat(chunk[:, 1], n), dtype=torch.float32, device=DEVICE)
        lnl_b = torch.tensor(np.repeat(chunk[:, 2], n), dtype=torch.float32, device=DEVICE)
        area_b = torch.tensor(np.repeat(chunk[:, 3], n), dtype=torch.float32, device=DEVICE)
        rb_b = torch.tensor(np.repeat(chunk[:, 4], n), dtype=torch.float32, device=DEVICE)
        a0_b = torch.tensor(np.repeat(chunk[:, 5], n), dtype=torch.float32, device=DEVICE)
        Vgs_b = Vgs_t.repeat(Nc)              # (B,)
        Vds_b = Vds_seq.repeat(Nc, 1)         # (B, T)
        p_b = {"VTH0": vth_b, "LTW": ltw_b, "L_NONLOCAL": lnl_b,
               "BJT_AREA": area_b, "Rb": rb_b, "ALPHA0": a0_b}
        with torch.no_grad():
            Id_pred = branch_follow_batch(p_b, Vgs_b, Vds_b)
        # Compute log-RMSE per curve, aggregate per combo
        Id_pred_c = torch.clamp(Id_pred, min=1e-30)
        log_pred = torch.log10(Id_pred_c)
        log_meas_b = log_meas.repeat(Nc, 1)   # (B, T)
        mask = (Id_meas.repeat(Nc, 1) > 1e-13) & (Id_pred > 0)
        diff_sq = (log_meas_b - log_pred) ** 2
        diff_sq = torch.where(mask, diff_sq, torch.zeros_like(diff_sq))
        per_curve_rmse_sq = diff_sq.sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1)
        per_curve_rmse = torch.sqrt(per_curve_rmse_sq).view(Nc, n)  # (Nc, n)
        combo_score = 0.5 * per_curve_rmse.median(dim=1).values \
                      + 0.5 * torch.quantile(per_curve_rmse, 0.9, dim=1)
        combo_score_np = combo_score.cpu().numpy()
        i_min_local = int(np.argmin(combo_score_np))
        if combo_score_np[i_min_local] < best_rmse:
            best_rmse = float(combo_score_np[i_min_local])
            best_params = chunk[i_min_local].copy()
            best_idx = start + i_min_local
        total_evals += Nc
        elapsed = time.time() - t0
        print(f"  batch {start:>5}-{end:>5}  best-so-far score={best_rmse:.3f} "
              f"({total_evals/elapsed:.0f} evals/s, {elapsed:.0f}s)")

    print(f"\n═══ GPU SWEEP DONE — {total_evals} evals in {time.time()-t0:.0f}s ═══")
    print(f"  best score : {best_rmse:.3f}")
    print(f"  VTH0       = {best_params[0]:.3f}")
    print(f"  LTW        = {best_params[1]*1e9:.1f} nm")
    print(f"  L_NONLOCAL = {best_params[2]*1e9:.0f} nm")
    print(f"  BJT_AREA   = {best_params[3]:.2e}")
    print(f"  Rb         = {best_params[4]:.2e}")
    print(f"  ALPHA0     = {best_params[5]:.2e}")

    # Re-evaluate on CPU-numpy with branch-following for comparison
    from scripts.z12_optimize import build_params, trace, log_rmse
    p_best = build_params([
        best_params[0], best_params[1], best_params[2],
        np.log10(best_params[3]), np.log10(best_params[4]),
        np.log10(best_params[5] / BASE.ALPHA0),
    ])
    per_curve = []
    for vg1, vg2, vd, idd in curves:
        pred = trace(vg1, vd, p_best)
        m = (idd > 1e-13) & (pred > 0)
        if m.sum() < 5:
            continue
        r = float(np.sqrt(np.mean((np.log10(idd[m]) - np.log10(pred[m]))**2)))
        per_curve.append({"vg1": vg1, "vg2": vg2, "log_rmse": r})
    rs = np.array([r["log_rmse"] for r in per_curve])
    print(f"\n[CPU verification] median {np.median(rs):.2f} p90 {np.percentile(rs, 90):.2f}")

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({
            "gpu_sweep_evals": total_evals,
            "best_params": [float(x) for x in best_params],
            "gpu_score": best_rmse,
            "cpu_median": float(np.median(rs)),
            "cpu_p90": float(np.percentile(rs, 90)),
            "cpu_worst": float(np.max(rs)),
        }, f, indent=2)


if __name__ == "__main__":
    main()
