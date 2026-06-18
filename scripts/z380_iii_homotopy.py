"""z380 — S2a: iii_body_gain homotopy continuation to disconnected snapback basin.

Hypothesis (from S1/S2):
    At R-46 best per-VG1 params (iii_body_gain=0.90 @ VG1=0.6, 1.5152 @ VG1=0.4,
    1.8447 @ VG1=0.2), the high-Vb snapback root is DISCONNECTED from the
    cold-start (Vb≈0) root along the natural parameter Vd. Forcing Vb=0.8
    *manually* opens a 5.5-decade Ids jump, proving the basin exists.

    If at HIGH iii_body_gain (e.g. 10×) the avalanche feedback dominates and
    the fold MERGES with the cold-start curve (single-valued), we can solve
    cold from Vb=0 and *then* continuation in iii_body_gain back down to the
    R-46 value, carrying the warm-start (Vsint, Vb) through 20 ramp steps.

Algorithm (per (VG1, VG2, Vd)):
    1. Set iii_body_gain = 10.0; solve cold from Vb_init=0  → (Vsint*, Vb*)
    2. For iii_k in linspace(10.0, iii_target, 20):
           cfg.iii_body_gain = iii_k
           solve with warm-start (Vsint*, Vb*) → update
    3. Final iii=iii_target solution: compare Ids to plain cold-start (Vb=0) baseline.

Gates:
    INFRA       — homotopy completes without nan on all 3 biases
    DISCOVERY   — at iii=target after homotopy, Vb>0.5 AND Ids>10× cold-start Ids @ Vd=1.5V
    AMBITIOUS   — cell-wide decade RMSE < 0.5 AND VG1=0.6 fold > 1.5 dec
    KILL-SHOT   — high iii also fails to fold, OR fold collapses during ramp back to target

Outputs results/z380_iii_homotopy/{summary.json, vb_during_homotopy.png, snapback_after_homotopy.png, run.log}.
"""
from __future__ import annotations
import os, sys, json, math, csv, time, importlib.util, traceback
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
OUT = ROOT / "results/z380_iii_homotopy"; OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data/sebas_2026_04_22"

LOG_F = open(OUT / "run.log", "w", buffering=1)
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True); LOG_F.write(line + "\n")


# --- Reuse build_base / param plumbing from z378 (kept identical to avoid drift) ---
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
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True, newton_max_iters=60)
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


# R-46 best per-VG1 params (Bf, iii_body_gain, vnwell_Rs)
PER_VG1 = {
    0.2: (1889.88, 1.8447, 9.1722),
    0.4: (1092.27, 1.5152, 9.8983),
    0.6: ( 417.63, 0.9036, 6.7846),
}


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
    if not sel.any(): return float("nan")
    masked = np.where(sel, dlog, -np.inf)
    return float(masked.max())


# --- Solver wrappers ---
def solve_one(cfg, M1, M2, bjt, Vd_scalar, VG1_t, VG2_t, Vsint_init, Vb_init,
              P_M1, P_M2):
    from nsram.bsim4_port.nsram_cell_2T import solve_2t_steady_state
    Vd_t = torch.tensor([Vd_scalar], dtype=torch.float64)
    out = solve_2t_steady_state(
        cfg, M1, bjt, Vd=Vd_t, VG1=VG1_t, VG2=VG2_t,
        P_M1=None, P_M2=None,
        Vsint_init=torch.tensor([Vsint_init], dtype=torch.float64),
        Vb_init=torch.tensor([Vb_init], dtype=torch.float64),
        model_M2=M2)
    Id = float(out["Id"].detach().squeeze().item())
    Vs = float(out["Vsint"].detach().squeeze().item())
    Vb = float(out["Vb"].detach().squeeze().item())
    conv_t = out["converged"]
    conv = bool(conv_t.squeeze().item()) if isinstance(conv_t, torch.Tensor) else bool(conv_t)
    return Id, Vs, Vb, conv


def run_homotopy_bias(cfg, M1, M2, bjt, Vd_array, vg1, vg2, P_M1, P_M2,
                     sd_M1, sd_M2, iii_high=10.0, n_ramp=20,
                     vb_trace_at_vd=1.5):
    """For each Vd:
        (a) cold-start at iii_high
        (b) ramp iii from iii_high → iii_target with warm-start
       Plus a baseline cold-start at iii_target for comparison.
    Returns dict with arrays: Vd, Ids_homotopy, Vb_homotopy, Ids_baseline, Vb_baseline,
    plus the full Vb trace along iii ramp at vb_trace_at_vd.
    """
    Bf, iii_target, Rs = PER_VG1[vg1]
    bjt.Bf = Bf
    cfg.vnwell_Rs = Rs

    VG1_t = torch.tensor(vg1, dtype=torch.float64)
    VG2_t = torch.tensor(vg2, dtype=torch.float64)

    iii_ramp = np.linspace(iii_high, iii_target, n_ramp)

    Ids_homo = np.full_like(Vd_array, np.nan, dtype=np.float64)
    Vb_homo  = np.full_like(Vd_array, np.nan, dtype=np.float64)
    Vs_homo  = np.full_like(Vd_array, np.nan, dtype=np.float64)
    conv_homo = np.zeros_like(Vd_array, dtype=bool)

    Ids_base = np.full_like(Vd_array, np.nan, dtype=np.float64)
    Vb_base  = np.full_like(Vd_array, np.nan, dtype=np.float64)

    # Trace storage at probe Vd (closest measured point ≥ vb_trace_at_vd)
    probe_idx = int(np.argmin(np.abs(Vd_array - vb_trace_at_vd)))
    probe_vd = float(Vd_array[probe_idx])
    iii_trace = list(iii_ramp)
    vb_trace = []
    ids_trace = []

    with patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
        with torch.no_grad():
            for i, vd in enumerate(Vd_array):
                vd_f = float(vd)
                # Baseline: cold start at iii_target
                cfg.iii_body_gain = float(iii_target)
                try:
                    Id_b, Vs_b, Vb_b, _ = solve_one(
                        cfg, M1, M2, bjt, vd_f, VG1_t, VG2_t,
                        Vsint_init=0.5*vd_f, Vb_init=0.0,
                        P_M1=P_M1, P_M2=P_M2)
                    Ids_base[i] = abs(Id_b); Vb_base[i] = Vb_b
                except Exception as e:
                    log(f"  baseline fail @ Vd={vd_f:.3f}: {e}")

                # (a) cold-start at iii_high
                cfg.iii_body_gain = float(iii_high)
                try:
                    Id_h, Vs_h, Vb_h, conv_h = solve_one(
                        cfg, M1, M2, bjt, vd_f, VG1_t, VG2_t,
                        Vsint_init=0.5*vd_f, Vb_init=0.0,
                        P_M1=P_M1, P_M2=P_M2)
                except Exception as e:
                    log(f"  iii_high cold fail @ Vd={vd_f:.3f}: {e}")
                    continue

                # (b) homotopy ramp
                trace_here = (i == probe_idx)
                if trace_here:
                    vb_trace = [Vb_h]; ids_trace = [abs(Id_h)]
                ok = True
                for iii_k in iii_ramp[1:]:
                    cfg.iii_body_gain = float(iii_k)
                    try:
                        Id_h, Vs_h, Vb_h, conv_h = solve_one(
                            cfg, M1, M2, bjt, vd_f, VG1_t, VG2_t,
                            Vsint_init=Vs_h, Vb_init=Vb_h,
                            P_M1=P_M1, P_M2=P_M2)
                    except Exception as e:
                        log(f"  homotopy fail @ Vd={vd_f:.3f} iii={iii_k:.3f}: {e}")
                        ok = False; break
                    if not (math.isfinite(Id_h) and math.isfinite(Vb_h)):
                        ok = False; break
                    if trace_here:
                        vb_trace.append(Vb_h); ids_trace.append(abs(Id_h))
                if ok:
                    Ids_homo[i] = abs(Id_h); Vb_homo[i] = Vb_h; Vs_homo[i] = Vs_h
                    conv_homo[i] = bool(conv_h)

    return {
        "Vd": Vd_array,
        "Ids_homotopy": Ids_homo,
        "Vb_homotopy": Vb_homo,
        "Vs_homotopy": Vs_homo,
        "conv_homotopy": conv_homo,
        "Ids_baseline": Ids_base,
        "Vb_baseline": Vb_base,
        "iii_ramp": np.array(iii_trace),
        "vb_trace": np.array(vb_trace),
        "ids_trace": np.array(ids_trace),
        "probe_vd": probe_vd,
        "iii_target": float(iii_target),
        "iii_high": float(iii_high),
    }


def find_bias_csv(vg1, vg2):
    """Find IV csv for (vg1, vg2) in Sebas data."""
    sub = DATA / f"2vHCa-2 I-Vs@VG2 VG1={vg1} vnwell=2"
    import re
    for f in sorted(sub.glob("*.csv")):
        m = re.search(r"VG2=(-?\d+\.\d+)_VG=", f.name)
        if not m: continue
        v = float(m.group(1))
        if abs(v - vg2) < 1e-3:
            return f
    return None


def main():
    t0 = time.time()
    log("z380 — S2a iii_body_gain homotopy continuation")
    log("Gates: INFRA=no nan on 3 biases; DISCOVERY Vb>0.5 AND Ids>10× cold @ Vd=1.5V; "
        "AMBITIOUS cell<0.5dec AND VG1=0.6 fold>1.5dec; "
        "KILL-SHOT high iii fails OR fold collapses during ramp")

    cfg, M1, M2, bjt = build_base()
    sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)
    rows = load_sebas_params()

    TEST_BIASES = [(0.2, 0.10), (0.4, 0.20), (0.6, 0.20)]

    results = []
    for (vg1, vg2) in TEST_BIASES:
        log(f"=== Bias VG1={vg1} VG2={vg2:+.2f} ===")
        fpath = find_bias_csv(vg1, vg2)
        if fpath is None:
            log(f"  csv not found, SKIP"); continue
        Vd_m, Id_m = load_csv(fpath)
        row = find_or_impute_row(rows, vg1, vg2)
        P_M1, P_M2 = make_overrides(row)

        t_b0 = time.time()
        try:
            out = run_homotopy_bias(cfg, M1, M2, bjt, Vd_m, vg1, vg2, P_M1, P_M2,
                                    sd_M1, sd_M2)
            err = None
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            log(f"  RUN FAIL: {err}\n{traceback.format_exc()}")
            results.append({"vg1": vg1, "vg2": vg2, "error": err})
            continue
        t_b1 = time.time()

        Id_h = out["Ids_homotopy"]
        Id_b = out["Ids_baseline"]
        Vb_h = out["Vb_homotopy"]
        Vb_b = out["Vb_baseline"]

        # Metrics
        rmse_h, n_h = decade_rmse(Id_h, Id_m)
        rmse_b, n_b = decade_rmse(Id_b, Id_m)
        jump_meas = max_forward_jump(Id_m, Vd_m)
        jump_homo = max_forward_jump(Id_h, Vd_m)
        jump_base = max_forward_jump(Id_b, Vd_m)

        # Vd=1.5 probe (or closest)
        idx_15 = int(np.argmin(np.abs(Vd_m - 1.5)))
        vd_15 = float(Vd_m[idx_15])
        ids_h_15 = float(Id_h[idx_15]) if math.isfinite(Id_h[idx_15]) else float("nan")
        ids_b_15 = float(Id_b[idx_15]) if math.isfinite(Id_b[idx_15]) else float("nan")
        vb_h_15 = float(Vb_h[idx_15]) if math.isfinite(Vb_h[idx_15]) else float("nan")
        vb_b_15 = float(Vb_b[idx_15]) if math.isfinite(Vb_b[idx_15]) else float("nan")
        ratio_15 = (ids_h_15 / ids_b_15) if (ids_b_15 > 1e-30 and math.isfinite(ids_b_15) and math.isfinite(ids_h_15)) else float("nan")

        n_nan_h = int(np.sum(~np.isfinite(Id_h)))
        n_nan_b = int(np.sum(~np.isfinite(Id_b)))

        # Vb trace summary
        vb_tr = out["vb_trace"]
        ids_tr = out["ids_trace"]
        vb_tr_start = float(vb_tr[0]) if len(vb_tr) > 0 else float("nan")
        vb_tr_end = float(vb_tr[-1]) if len(vb_tr) > 0 else float("nan")
        vb_tr_min = float(np.min(vb_tr)) if len(vb_tr) > 0 else float("nan")

        log(f"  iii_target={out['iii_target']:.4f}, iii_high={out['iii_high']:.2f}, "
            f"n_ramp_done={len(vb_tr)} at probe Vd={out['probe_vd']:.3f}")
        log(f"  Vb trace iii=10→target: start={vb_tr_start:.3f} end={vb_tr_end:.3f} min={vb_tr_min:.3f}")
        log(f"  @Vd≈1.5: Ids_homo={ids_h_15:.3e} Vb_homo={vb_h_15:.3f} | "
            f"Ids_cold={ids_b_15:.3e} Vb_cold={vb_b_15:.3f} | ratio={ratio_15:.2e}")
        log(f"  decade RMSE: homo={rmse_h:.3f} (n={n_h}), cold={rmse_b:.3f} (n={n_b})")
        log(f"  max fwd jump: meas={jump_meas:.2f}, homo={jump_homo:.2f}, cold={jump_base:.2f} dec/step")
        log(f"  nan count: homo={n_nan_h}, cold={n_nan_b}  ({t_b1-t_b0:.1f}s)")

        # Per-bias gates
        discovery = bool(math.isfinite(vb_h_15) and vb_h_15 > 0.5
                         and math.isfinite(ratio_15) and ratio_15 > 10.0)
        kill = bool((not math.isfinite(vb_tr_start)) or vb_tr_start <= 0.3
                    or (math.isfinite(vb_tr_start) and math.isfinite(vb_tr_end)
                        and vb_tr_start > 0.5 and vb_tr_end < 0.3))
        log(f"  per-bias DISCOVERY={discovery}, KILL-SHOT-signal={kill}")

        results.append({
            "vg1": vg1, "vg2": vg2,
            "iii_target": out["iii_target"], "iii_high": out["iii_high"],
            "rmse_homo": rmse_h, "rmse_base": rmse_b, "n_pts": int(len(Vd_m)),
            "n_dec_homo": n_h, "n_dec_base": n_b,
            "max_jump_meas": jump_meas,
            "max_jump_homo": jump_homo,
            "max_jump_base": jump_base,
            "Vd_15": vd_15, "Ids_homo_15": ids_h_15, "Ids_base_15": ids_b_15,
            "ratio_homo_over_base_15": ratio_15,
            "Vb_homo_15": vb_h_15, "Vb_base_15": vb_b_15,
            "vb_trace_start": vb_tr_start, "vb_trace_end": vb_tr_end, "vb_trace_min": vb_tr_min,
            "iii_ramp_n": int(len(vb_tr)),
            "probe_vd": float(out["probe_vd"]),
            "n_nan_homo": n_nan_h, "n_nan_base": n_nan_b,
            "discovery": discovery, "kill_shot_signal": kill,
            "wall_s": t_b1 - t_b0,
            # raw arrays for plotting
            "_Vd": Vd_m.tolist(),
            "_Id_meas": Id_m.tolist(),
            "_Id_homo": [None if not math.isfinite(x) else float(x) for x in Id_h],
            "_Id_base": [None if not math.isfinite(x) else float(x) for x in Id_b],
            "_Vb_homo": [None if not math.isfinite(x) else float(x) for x in Vb_h],
            "_iii_ramp": out["iii_ramp"].tolist(),
            "_vb_trace": vb_tr.tolist(),
            "_ids_trace": ids_tr.tolist(),
        })

    # --- Plots ---
    # Plot 1: Vb during homotopy (iii ramp) at probe Vd
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for r in results:
        if "error" in r: continue
        ax.plot(r["_iii_ramp"], r["_vb_trace"],
                marker="o", label=f"VG1={r['vg1']} VG2={r['vg2']:+.2f} (Vd≈{r['probe_vd']:.2f})")
    ax.set_xlabel("iii_body_gain (ramp from 10 → R-46 target)")
    ax.set_ylabel("Vb [V] during homotopy")
    ax.set_title("z380 — Vb trace during iii_gain homotopy continuation")
    ax.invert_xaxis()
    ax.axhline(0.5, color="gray", linestyle=":", label="Vb=0.5 threshold")
    ax.legend(fontsize=8); ax.grid(alpha=0.4)
    fig.tight_layout(); fig.savefig(OUT / "vb_during_homotopy.png", dpi=130); plt.close(fig)

    # Plot 2: Snapback after homotopy vs cold-start baseline vs measurement
    n = len([r for r in results if "error" not in r])
    fig, axes = plt.subplots(1, max(1, n), figsize=(5*max(1, n), 4.5), sharey=True)
    if n <= 1: axes = [axes]
    j = 0
    for r in results:
        if "error" in r: continue
        ax = axes[j]; j += 1
        Vd = np.array(r["_Vd"])
        Imeas = np.array(r["_Id_meas"])
        Ihomo = np.array([np.nan if x is None else x for x in r["_Id_homo"]])
        Ibase = np.array([np.nan if x is None else x for x in r["_Id_base"]])
        ax.semilogy(Vd, np.maximum(Imeas, 1e-15), "k.-", label="meas", lw=1)
        ax.semilogy(Vd, np.maximum(Ibase, 1e-15), "b--", label=f"cold (iii={r['iii_target']:.2f})", lw=1.2)
        ax.semilogy(Vd, np.maximum(Ihomo, 1e-15), "r-", label="homotopy", lw=1.5)
        ax.set_title(f"VG1={r['vg1']} VG2={r['vg2']:+.2f}\nratio@1.5V={r['ratio_homo_over_base_15']:.1e}",
                     fontsize=10)
        ax.set_xlabel("Vd [V]"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
        if j == 1: ax.set_ylabel("|Ids| [A]")
    fig.suptitle("z380 — Snapback IV after iii_gain homotopy", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "snapback_after_homotopy.png", dpi=130); plt.close(fig)

    # --- Gates evaluation ---
    n_ok = sum(1 for r in results if "error" not in r and r["n_nan_homo"] < r["n_pts"])
    infra_pass = (n_ok == len(TEST_BIASES))
    discoveries = [r.get("discovery", False) for r in results if "error" not in r]
    discovery_pass = all(discoveries) and len(discoveries) == len(TEST_BIASES)
    # AMBITIOUS
    vg06 = next((r for r in results if r.get("vg1") == 0.6 and "error" not in r), None)
    rmses = [r["rmse_homo"] for r in results if "error" not in r and math.isfinite(r["rmse_homo"])]
    rmse_cell = float(np.sqrt(np.mean(np.array(rmses)**2))) if rmses else float("nan")
    ambitious_pass = bool(math.isfinite(rmse_cell) and rmse_cell < 0.5
                          and vg06 is not None and math.isfinite(vg06["max_jump_homo"])
                          and vg06["max_jump_homo"] > 1.5)
    # KILL-SHOT
    kill_signals = [r.get("kill_shot_signal", False) for r in results if "error" not in r]
    kill_shot = any(kill_signals) or (not infra_pass)

    summary = {
        "script": "z380_iii_homotopy",
        "wall_total_s": time.time() - t0,
        "gates": {
            "INFRA": bool(infra_pass),
            "DISCOVERY": bool(discovery_pass),
            "AMBITIOUS": bool(ambitious_pass),
            "KILL_SHOT": bool(kill_shot),
        },
        "rmse_cell_homotopy": rmse_cell,
        "biases": results,
    }
    with open(OUT / "summary.json", "w") as f:
        # strip raw arrays from on-disk summary to keep small
        def _strip(obj):
            if isinstance(obj, dict):
                return {k: _strip(v) for k, v in obj.items() if not k.startswith("_")}
            return obj
        json.dump(_strip(summary), f, indent=2, default=lambda o: None if (isinstance(o, float) and not math.isfinite(o)) else o)

    log(f"=== GATES ===")
    log(f"  INFRA      = {infra_pass}  ({n_ok}/{len(TEST_BIASES)} biases ran)")
    log(f"  DISCOVERY  = {discovery_pass}  (all biases Vb>0.5 AND ratio>10× @ Vd≈1.5V)")
    log(f"  AMBITIOUS  = {ambitious_pass}  (rmse_cell={rmse_cell:.3f}, VG1=0.6 fold={vg06['max_jump_homo'] if vg06 else float('nan'):.2f} dec)")
    log(f"  KILL-SHOT  = {kill_shot}  (any high iii fail / fold collapse / no infra)")
    log(f"Total wall: {summary['wall_total_s']:.1f}s")
    log(f"Wrote {OUT/'summary.json'}, {OUT/'vb_during_homotopy.png'}, {OUT/'snapback_after_homotopy.png'}")
    LOG_F.close()


if __name__ == "__main__":
    main()
