"""z371 — DS-N18: GPU-batched massive parameter sweep on ikaros (AMD gfx1151).

Goal: saturate ROCm GPU by evaluating N_param parameter vectors in parallel
across all 33 Sebas DC curves. Each param vector becomes a "cell" in the
N-cell GPU batch of forward_2t_batched_gpu; we loop over the 33 curves
(each curve has its own Vd_seq, VG1, VG2, P_M1/P_M2) one GPU call at a time.

Per param vector, cell-wide median log10-RMSE is computed across 33 curves.
Top 10 are reported.

Param space (R-50 physical bounds, identical to z370):
  Bf            in [50, 50000]      (BJT forward beta)
  Va            in [0.5, 3.0] V     (Early voltage)
  Is            log-uniform [1e-13, 1e-7] A (BJT saturation current)
  vnwell_Rs     log-uniform [1e5, 1e9] ohm (well/body ohmic)
  iii_body_gain in [0.1, 1.0]       (eta_lat physical range)

Sampling: 500 Sobol + 500 LHS  (1000 total)

Pre-registered gates:
  INFRA       : 1000 samples complete < 30 min on ikaros GPU
  DISCOVERY   : any sample beats R-43 floor 1.131 dec
  BREAKTHROUGH: cell-wide median < 0.95 dec with PHYSICAL params

Output: results/z371_gpu_blitz/{summary.json, distribution.png, all_samples.json}
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
import sys, json, re, time, math, csv, importlib.util
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))
OUT = ROOT / "results/z371_gpu_blitz"; OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data/sebas_2026_04_22"


@contextmanager
def patch_sd_scaled(sd, overrides):
    if not overrides:
        yield; return
    saved = {}
    try:
        for k, v in overrides.items():
            saved[k] = sd.scaled.get(k, None)
            sd.scaled[k] = (v.item() if torch.is_tensor(v) else float(v))
        yield
    finally:
        for k, v in saved.items():
            if v is None: sd.scaled.pop(k, None)
            else: sd.scaled[k] = v


def load_sebas_params():
    path = DATA / "2Tcell_BSIM_param_DC.csv"
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
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
M2_STATIC_OVERRIDES = {
    "k1": 0.63825, "k2": -0.070435, "etab": -0.086777, "beta0": 18.0,
}


def find_or_impute_row(rows, VG1, VG2, atol=1e-3):
    target = None
    for r in rows:
        if abs(r["VG1"] - VG1) < atol and abs(r["VG2"] - VG2) < atol:
            target = dict(r); break
    if target is None:
        return None, False
    if math.isnan(target.get("K1", float("nan"))):
        branch = BRANCH_FLAT.get(round(VG1, 2))
        if branch is None:
            return target, False
        for k, v in branch.items():
            target[k] = float(v)
        return target, True
    return target, False


def make_overrides(sebas_row):
    if sebas_row is None:
        return None, None
    P_M1 = {}
    for csv_k, py_k in (("ETAB", "etab"), ("K1", "k1"),
                       ("ALPHA0", "alpha0"), ("BETA0", "beta0")):
        if not math.isnan(sebas_row.get(csv_k, float("nan"))):
            P_M1[py_k] = float(sebas_row[csv_k])
    P_M2 = {}
    if not math.isnan(sebas_row.get("NFACTOR", float("nan"))):
        P_M2["nfactor"] = float(sebas_row["NFACTOR"])
    for k, v in M2_STATIC_OVERRIDES.items():
        if k not in P_M2:
            P_M2[k] = float(v)
    return (P_M1 or None), (P_M2 or None)


def build_pyport_base():
    sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
    v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
    from nsram.bsim4_port.bjt import GummelPoonNPN
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True, newton_max_iters=40)
    cfg.bjt_emitter_to_gnd = True
    cfg.body_pdiode_to = "vnwell"
    cfg.use_well_diode = True
    cfg.vnwell = 2.0  # R-45 frozen
    cfg.body_pdiode_Js = 5.3675e-7 / 22e-12
    cfg.body_pdiode_n = 1.0535
    cfg.body_pdiode_Rs = 1.0e6
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
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
            d = np.loadtxt(f, delimiter=",", skiprows=1)
            if d.ndim != 2 or d.shape[1] < 2: continue
            curves.append({"VG1": vg1, "VG2": vg2, "Vd": d[:, 0],
                          "Id": np.abs(d[:, 1]), "f": f.name})
    for f in sorted(DATA.glob("VG1*VG2*.csv")):
        m = re.search(r"VG1=([\d.\-]+)[_ ]*VG2=([\d.\-]+)", f.name)
        if not m: continue
        vg1 = float(m.group(1)); vg2 = float(m.group(2))
        d = np.loadtxt(f, delimiter=",", skiprows=1)
        curves.append({"VG1": vg1, "VG2": vg2, "Vd": d[:, 0],
                      "Id": np.abs(d[:, 1]), "f": f.name})
    return curves


# 5-dim physical bounds (R-50)
PARAM_BOUNDS = np.array([
    [50.0,    50000.0],   # Bf (linear)
    [0.5,     3.0],       # Va (V)
    [-13.0,   -7.0],      # log10(Is) -> Is in [1e-13, 1e-7]
    [5.0,     9.0],       # log10(Rs) -> Rs in [1e5, 1e9]
    [0.1,     1.0],       # iii_body_gain
])
PARAM_NAMES = ["Bf", "Va", "log10_Is", "log10_Rs", "iii_body_gain"]


def sobol_samples(N: int, d: int, seed: int = 0) -> np.ndarray:
    """Scrambled Sobol via scipy."""
    from scipy.stats.qmc import Sobol
    s = Sobol(d=d, scramble=True, seed=seed)
    # Sobol balanced power-of-2; oversample then trim.
    n2 = int(2 ** math.ceil(math.log2(max(2, N))))
    pts = s.random_base2(int(math.log2(n2)))
    return pts[:N]


def lhs_samples(N: int, d: int, seed: int = 0) -> np.ndarray:
    from scipy.stats.qmc import LatinHypercube
    s = LatinHypercube(d=d, seed=seed)
    return s.random(n=N)


def make_param_batch(N: int, seed: int = 371) -> np.ndarray:
    """Return (N, 5) raw parameter array in physical bounds.
    Half Sobol, half LHS. Each row is [Bf, Va, Is, Rs, iii] in raw units."""
    n_sobol = N // 2
    n_lhs = N - n_sobol
    u_sobol = sobol_samples(n_sobol, 5, seed=seed)
    u_lhs = lhs_samples(n_lhs, 5, seed=seed + 1)
    u = np.vstack([u_sobol, u_lhs])
    lo = PARAM_BOUNDS[:, 0]; hi = PARAM_BOUNDS[:, 1]
    x_scaled = lo + u * (hi - lo)  # (N, 5) in [Bf, Va, log10Is, log10Rs, iii]
    out = np.zeros_like(x_scaled)
    out[:, 0] = x_scaled[:, 0]                       # Bf
    out[:, 1] = x_scaled[:, 1]                       # Va
    out[:, 2] = 10.0 ** x_scaled[:, 2]               # Is
    out[:, 3] = 10.0 ** x_scaled[:, 3]               # Rs
    out[:, 4] = x_scaled[:, 4]                       # iii_gain
    return out  # columns are [Bf, Va, Is, Rs, iii]


def main():
    t0 = time.time()
    parser_N = int(os.environ.get("Z371_N", "1000"))
    print(f"[z371] DS-N18: GPU param blitz  N={parser_N}", flush=True)
    print(f"[z371] device check: torch.cuda.is_available()={torch.cuda.is_available()}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[z371] device = {device}  name={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)

    from nsram.bsim4_port.forward_2t_batched_gpu import forward_2t_gpu_batched

    cfg, M1, M2, bjt = build_pyport_base()
    sebas_rows = load_sebas_params()
    curves = load_curves()
    print(f"[z371] loaded {len(curves)} curves, {len(sebas_rows)} Sebas rows", flush=True)
    sd_M1 = cfg.size_dep_M1(M1)
    sd_M2 = cfg.size_dep_M2(M2)

    # ---- Generate N parameter vectors ---------------------------------- #
    P = make_param_batch(parser_N, seed=371)
    N = P.shape[0]
    print(f"[z371] param batch: {N}×5", flush=True)

    # Move param tensors to device
    Bf_t  = torch.tensor(P[:, 0], dtype=torch.float64, device=device)
    Va_t  = torch.tensor(P[:, 1], dtype=torch.float64, device=device)
    Is_t  = torch.tensor(P[:, 2], dtype=torch.float64, device=device)
    Rs_t  = torch.tensor(P[:, 3], dtype=torch.float64, device=device)
    iii_t = torch.tensor(P[:, 4], dtype=torch.float64, device=device)

    # Tensorize bjt + cfg in-place. _residuals and compute_bjt read these as
    # tensors and broadcast over Vbe/Vbc/Vd (which we feed as shape (N,) per
    # GPU call). The fp64 dtype matches the Newton's working precision.
    bjt.Bf = Bf_t
    bjt.Va = Va_t
    bjt.Is = Is_t
    cfg.iii_body_gain = iii_t
    cfg.vnwell_Rs = Rs_t

    # ---- Loop over 33 curves, batching all N params per call ---------- #
    # log10-RMSE accumulator: shape (N, n_curves)
    n_curves = len(curves)
    rmse_NC = np.full((N, n_curves), np.nan, dtype=np.float64)

    # Warm-state propagation per-curve isn't possible across curves since
    # different VG1/VG2; but within a curve we propagate Vsint, Vb (batched
    # gpu kernel handles warm-start across Vd internally).

    print(f"[z371] starting {n_curves} GPU calls (each N={N} cells × ~{82} Vd points)...", flush=True)
    t_loop = time.time()
    for ci, c in enumerate(curves):
        Vd_seq = torch.tensor(c["Vd"], dtype=torch.float64, device=device)
        # Per-cell VG1/VG2 are identical (the curve's VG1/VG2) across N
        VG1_arr = torch.full((N,), float(c["VG1"]), dtype=torch.float64, device=device)
        VG2_arr = torch.full((N,), float(c["VG2"]), dtype=torch.float64, device=device)
        row, _ = find_or_impute_row(sebas_rows, c["VG1"], c["VG2"])
        P_M1, P_M2 = make_overrides(row)

        t_call = time.time()
        try:
            with patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
                out = forward_2t_gpu_batched(
                    cfg, model_M1=M1, model_M2=M2, bjt=bjt,
                    Vd_seq=Vd_seq, VG1_arr=VG1_arr, VG2_arr=VG2_arr,
                    max_iters=int(os.environ.get("Z371_MAXITER", "15")), tol=1e-9, damping=1.0,
                    step_clamp=0.5, eps=1e-5,
                    Vb_lo=-0.5, Vb_hi=1.2,
                    dtype=torch.float64, device=device,
                    early_stop=True, verbose=False,
                )
            Id_pred = out["Id"].detach().abs().cpu().numpy()  # (N, T)
            Id_meas = c["Id"][None, :]  # (1, T)
            valid = (Id_meas > 1e-15) & (Id_pred > 1e-15) & np.isfinite(Id_pred)
            # log10 ratio diff
            with np.errstate(invalid="ignore", divide="ignore"):
                log_diff = np.where(valid,
                                    np.log10(Id_pred) - np.log10(Id_meas),
                                    np.nan)
            # row-wise RMSE
            sq = log_diff ** 2
            n_per_row = np.sum(valid, axis=1)
            sumsq = np.nansum(sq, axis=1)
            rmse_row = np.where(n_per_row >= 3,
                                np.sqrt(sumsq / np.maximum(n_per_row, 1)),
                                np.nan)
            rmse_NC[:, ci] = rmse_row
        except Exception as e:
            print(f"  [curve {ci}] EXCEPTION: {e}", flush=True)

        if ci % 5 == 0 or ci == n_curves - 1:
            elapsed = time.time() - t_call
            valid_share = float(np.mean(~np.isnan(rmse_NC[:, ci])))
            print(f"  [curve {ci+1:2d}/{n_curves}] VG1={c['VG1']:.2f} VG2={c['VG2']:.2f} "
                  f"T={Vd_seq.numel():3d}  {elapsed:.2f}s  valid={valid_share:.3f}", flush=True)

    t_loop_done = time.time() - t_loop
    print(f"\n[z371] all curves done in {t_loop_done:.1f}s  ({t_loop_done/n_curves:.2f}s/curve)", flush=True)

    # ---- Aggregate per-param cell-wide median ------------------------- #
    # Replace NaNs with sentinel high cost (so we don't reward incomplete sweeps)
    valid_per_param = np.sum(~np.isnan(rmse_NC), axis=1)
    rmse_filled = np.where(np.isnan(rmse_NC), 10.0, rmse_NC)
    cell_med = np.median(rmse_filled, axis=1)        # (N,)
    n_valid_curves = valid_per_param                  # (N,)

    # Best ranking
    order = np.argsort(cell_med)
    top50 = order[:50]
    top10 = order[:10]

    # ---- Report results ------------------------------------------------ #
    R43_FLOOR = 1.1306581736187744
    R46_BEST = 0.965  # per-VG1 fit (NOT physical-global; for context)

    samples_summary = []
    for rank, i in enumerate(order):
        samples_summary.append({
            "rank": int(rank),
            "idx": int(i),
            "params": {
                "Bf": float(P[i, 0]),
                "Va": float(P[i, 1]),
                "Is": float(P[i, 2]),
                "vnwell_Rs": float(P[i, 3]),
                "iii_body_gain": float(P[i, 4]),
            },
            "cell_wide_median_dec": float(cell_med[i]),
            "n_valid_curves": int(n_valid_curves[i]),
            "n_total_curves": int(n_curves),
            "sampler": "sobol" if i < N // 2 else "lhs",
        })

    best = samples_summary[0]
    print("\n========== TOP 10 ==========")
    for s in samples_summary[:10]:
        p = s["params"]
        print(f"  rank {s['rank']:2d}  dec={s['cell_wide_median_dec']:.4f}  "
              f"valid={s['n_valid_curves']}/33  "
              f"Bf={p['Bf']:.1f}  Va={p['Va']:.3f}  Is={p['Is']:.2e}  "
              f"Rs={p['vnwell_Rs']:.2e}  iii={p['iii_body_gain']:.3f}")
    print("===========================\n")

    elapsed_total = time.time() - t0
    summary = {
        "script": "z371_gpu_param_blitz",
        "task": "DS-N18",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "N_samples": int(N),
        "n_sobol": int(N // 2),
        "n_lhs": int(N - N // 2),
        "n_curves": int(n_curves),
        "param_bounds_R50": {
            "Bf": [50.0, 50000.0],
            "Va": [0.5, 3.0],
            "Is": [1e-13, 1e-7],
            "vnwell_Rs": [1e5, 1e9],
            "iii_body_gain": [0.1, 1.0],
        },
        "patches_active": [
            "R-20 BJT Vbc", "R-29 Vth/tox", "R-37 binunit",
            "R-41 body_pdiode_to=vnwell + use_well_diode=True",
            "R-45 cfg.vnwell=2.0 frozen",
            "z371 tensorized bjt.Bf/Va/Is + cfg.iii_body_gain/vnwell_Rs",
        ],
        "elapsed_total_s": float(elapsed_total),
        "elapsed_loop_s": float(t_loop_done),
        "wall_time_min": float(elapsed_total / 60.0),
        "best": best,
        "top_10": samples_summary[:10],
        "top_50": samples_summary[:50],
        "baselines": {
            "z363_R43_global_floor": R43_FLOOR,
            "z365_R46_perVG1": R46_BEST,
            "ngspice_target_aspiration": 0.27,
        },
        "stats": {
            "median_dec": float(np.median(cell_med)),
            "p10_dec": float(np.percentile(cell_med, 10)),
            "p25_dec": float(np.percentile(cell_med, 25)),
            "min_dec": float(np.min(cell_med)),
            "max_dec": float(np.max(cell_med)),
            "n_below_R43_floor": int(np.sum(cell_med < R43_FLOOR)),
            "n_below_0p95": int(np.sum(cell_med < 0.95)),
            "n_below_0p50": int(np.sum(cell_med < 0.50)),
        },
        "gates": {
            "INFRA_under_30min": bool(elapsed_total < 30 * 60),
            "DISCOVERY_beat_R43": bool(best["cell_wide_median_dec"] < R43_FLOOR),
            "BREAKTHROUGH_under_0p95": bool(best["cell_wide_median_dec"] < 0.95),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    # All samples in a compact JSON (top 50 already in summary; full in all_samples)
    all_samples = {
        "params_columns": ["Bf", "Va", "Is", "vnwell_Rs", "iii_body_gain"],
        "params": P.tolist(),
        "cell_wide_median_dec": cell_med.tolist(),
        "n_valid_curves": n_valid_curves.tolist(),
        "rmse_per_curve_first_5_rows": rmse_NC[:5].tolist(),
    }
    (OUT / "all_samples.json").write_text(json.dumps(all_samples, indent=2))

    # Distribution plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ax = axes[0]
        ax.hist(cell_med[np.isfinite(cell_med) & (cell_med < 10)], bins=60,
                edgecolor="black", alpha=0.7)
        ax.axvline(R43_FLOOR, color="red", linestyle="--",
                   label=f"R-43 floor ({R43_FLOOR:.3f})")
        ax.axvline(R46_BEST, color="green", linestyle="--",
                   label=f"R-46 per-VG1 ({R46_BEST:.3f})")
        ax.axvline(best["cell_wide_median_dec"], color="blue", linestyle="-",
                   label=f"z371 best ({best['cell_wide_median_dec']:.3f})")
        ax.set_xlabel("cell-wide median dec")
        ax.set_ylabel("count")
        ax.set_title(f"z371 GPU blitz N={N}: dec distribution")
        ax.legend()
        # Pairwise scatter for top 100
        ax = axes[1]
        sc = ax.scatter(P[order[:200], 0], 10.0 * np.log10(P[order[:200], 3]),
                        c=cell_med[order[:200]], cmap="viridis", s=20)
        ax.set_xscale("log")
        ax.set_xlabel("Bf")
        ax.set_ylabel("10·log10(vnwell_Rs)")
        ax.set_title("Top 200: Bf vs Rs colored by dec")
        plt.colorbar(sc, ax=ax, label="dec")
        plt.tight_layout()
        plt.savefig(OUT / "distribution.png", dpi=120)
        plt.close()
    except Exception as e:
        print(f"  [plot] failed: {e}", flush=True)

    print(f"\n[z371] DONE")
    print(f"  best dec  = {best['cell_wide_median_dec']:.4f}")
    print(f"  N<R43     = {summary['stats']['n_below_R43_floor']}/{N}")
    print(f"  N<0.95    = {summary['stats']['n_below_0p95']}/{N}")
    print(f"  wall      = {elapsed_total:.1f}s  ({elapsed_total/60:.1f} min)")
    print(f"  gates     = INFRA:{summary['gates']['INFRA_under_30min']}  "
          f"DISCOVERY:{summary['gates']['DISCOVERY_beat_R43']}  "
          f"BREAKTHROUGH:{summary['gates']['BREAKTHROUGH_under_0p95']}")
    print(f"  out       = {OUT}")


if __name__ == "__main__":
    main()
