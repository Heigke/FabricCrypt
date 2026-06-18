"""J1 — v5.5 surrogate regen (100K pts, Tlpe1 fix ON).

Builds a dense 3D surrogate (VG1 × VG2 × Vd) with the v5.5 BSIM4 model
card corrections enabled (tlpe1_disable=True, ngspice_match well-diode).
Validates by comparing surrogate-vs-forward_2t over a 32-bias panel.

Outputs:
  results/SURROGATE_v55_2026-05-21/surrogate_v55.npz
  results/SURROGATE_v55_2026-05-21/surrogate_v55.json    (build meta)
  results/SURROGATE_v55_2026-05-21/parity_32bias.json    (acceptance metric)
"""
from __future__ import annotations
import os, sys, json, time
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "2")  # be gentle on APU
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/SURROGATE_v55_2026-05-21"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.vectorized import forward_2t_batched

# v5.5 model-card optimum (from best calibration)
OPT_BF, OPT_VA, OPT_IS = 9000.0, 0.55, 1e-9


def build_v55_models():
    M1, M2 = v1.build_calibrated_models()
    # v5.5 toggles (K1+ALPHA0 path + Tlpe1 fix + ngspice_match well diode)
    M1._values["tlpe1_disable"] = True
    M2._values["tlpe1_disable"] = True
    M1._values["dibl_upper_clamp"] = True
    M2._values["dibl_upper_clamp"] = True
    M1._values["well_diode_mode"] = "ngspice_match"
    M2._values["well_diode_mode"] = "ngspice_match"
    return M1, M2


def thermal_temp_c():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return -1.0


def wait_cool(threshold=75.0, target=50.0, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        t = thermal_temp_c()
        if t < threshold:
            return t
        print(f"  [thermal] T={t:.1f}C >= {threshold}C — sleeping 10s", flush=True)
        time.sleep(10)
    return thermal_temp_c()


def build_dense_grid(grid_size=(46, 46, 48)):
    """46*46*48 = 101,568 ~ 100K pts.  Vectorised: one forward_2t_batched
    sweep over all (VG1,VG2) pairs at all Vd. Memory ~ 100K*float64 ~ 1MB."""
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                            newton_max_iters=20)
    M1, M2 = build_v55_models()
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = OPT_BF; bjt.Va = OPT_VA; bjt.Is = OPT_IS

    NG1, NG2, NVD = grid_size
    vg1_axis = np.linspace(0.10, 0.80, NG1)
    vg2_axis = np.linspace(-0.10, 0.60, NG2)
    vd_axis  = np.linspace(0.10, 2.20, NVD)

    print(f"[J1] grid {NG1}x{NG2}x{NVD} = {NG1*NG2*NVD} pts ; "
          f"APU T0={thermal_temp_c():.1f}C", flush=True)

    VG1_flat = np.repeat(vg1_axis, NG2)
    VG2_flat = np.tile(vg2_axis, NG1)
    VG1_t = torch.tensor(VG1_flat); VG2_t = torch.tensor(VG2_flat)
    Vd_t = torch.tensor(vd_axis)

    t0 = time.time()
    # Chunk in (VG1,VG2) pairs to keep memory bounded + allow thermal checks
    N_pairs = NG1 * NG2
    CHUNK = 256
    Id_chunks = []
    for s in range(0, N_pairs, CHUNK):
        e = min(s + CHUNK, N_pairs)
        if thermal_temp_c() >= 75.0:
            t = wait_cool()
            print(f"  [thermal] resumed at T={t:.1f}C", flush=True)
        out = forward_2t_batched(cfg, M1, M2, bjt,
                                  Vd_t, VG1_t[s:e], VG2_t[s:e],
                                  max_iters=20, tol=1e-9, verbose=False)
        Id_chunks.append(out["Id"].abs().clamp(min=1e-15).log10().numpy())
        wall = time.time() - t0
        done = e
        eta = wall / done * (N_pairs - done)
        print(f"  pair {done}/{N_pairs}  wall={wall:.0f}s  eta={eta:.0f}s  "
              f"T={thermal_temp_c():.1f}C", flush=True)
    Id_flat = np.concatenate(Id_chunks, axis=0)  # (N_pairs, NVD)
    grid = Id_flat.reshape(NG1, NG2, NVD)
    wall = time.time() - t0
    out_npz = OUT / "surrogate_v55.npz"
    np.savez(out_npz, grid=grid, vg1_axis=vg1_axis, vg2_axis=vg2_axis,
             vd_axis=vd_axis, build_t=wall)
    meta = dict(
        version="v5.5",
        flags=dict(tlpe1_disable=True, dibl_upper_clamp=True,
                   well_diode_mode="ngspice_match"),
        opt=dict(Bf=OPT_BF, Va=OPT_VA, Is=OPT_IS),
        grid_size=list(grid_size), n_points=int(NG1*NG2*NVD),
        vg1_range=[0.10, 0.80], vg2_range=[-0.10, 0.60], vd_range=[0.10, 2.20],
        build_wall_s=wall,
        log_Id_range=[float(grid.min()), float(grid.max())],
    )
    with open(out_npz.with_suffix(".json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[J1] built {grid.size} pts in {wall:.1f}s ; "
          f"log_Id [{grid.min():.2f}, {grid.max():.2f}]", flush=True)
    return grid, vg1_axis, vg2_axis, vd_axis, cfg, M1, M2, bjt


def parity_32bias(grid, vg1_axis, vg2_axis, vd_axis, cfg, M1, M2, bjt):
    """Acceptance gate: surrogate-vs-forward_2t residual ≤ 0.39 dec
    across 32 reservoir biases."""
    rng = np.random.default_rng(2026_05_21)
    # 32 reservoir biases (VG1, VG2) randomly drawn in middle of axes
    VG1_b = rng.uniform(0.20, 0.65, size=32)
    VG2_b = rng.uniform(0.00, 0.50, size=32)
    Vd_test = np.array([0.5, 1.0, 1.5, 2.0])

    # Surrogate eval (trilinear)
    def trilin(VG1, VG2, Vd):
        def _idx(x, ax):
            n = len(ax)
            i = np.clip(np.searchsorted(ax, x) - 1, 0, n-2)
            f = np.clip((x-ax[i])/(ax[i+1]-ax[i]+1e-30), 0., 1.)
            return i, f
        i, fi = _idx(VG1, vg1_axis)
        j, fj = _idx(VG2, vg2_axis)
        k, fk = _idx(Vd, vd_axis)
        g = grid
        c00 = g[i,j,k]*(1-fi) + g[i+1,j,k]*fi
        c01 = g[i,j,k+1]*(1-fi) + g[i+1,j,k+1]*fi
        c10 = g[i,j+1,k]*(1-fi) + g[i+1,j+1,k]*fi
        c11 = g[i,j+1,k+1]*(1-fi) + g[i+1,j+1,k+1]*fi
        c0 = c00*(1-fj) + c10*fj
        c1 = c01*(1-fj) + c11*fj
        return c0*(1-fk) + c1*fk

    # Forward truth
    VG1_t = torch.tensor(np.repeat(VG1_b, len(Vd_test)))
    VG2_t = torch.tensor(np.repeat(VG2_b, len(Vd_test)))
    Vd_t  = torch.tensor(np.tile(Vd_test, 32))
    # forward_2t_batched expects Vd_seq (T,) common across batch.
    # Loop over Vd_test for cleanliness.
    truth_log = np.zeros((32, len(Vd_test)))
    surr_log  = np.zeros((32, len(Vd_test)))
    for kvd, vd in enumerate(Vd_test):
        out = forward_2t_batched(cfg, M1, M2, bjt,
                                  torch.tensor([vd]),
                                  torch.tensor(VG1_b), torch.tensor(VG2_b),
                                  max_iters=20, tol=1e-9, verbose=False)
        truth_log[:, kvd] = out["Id"].abs().clamp(min=1e-15).log10().numpy().ravel()
        surr_log[:, kvd]  = trilin(VG1_b, VG2_b, np.full(32, vd))
    residual_dec = np.abs(truth_log - surr_log)
    median = float(np.median(residual_dec))
    p95 = float(np.percentile(residual_dec, 95))
    max_d = float(residual_dec.max())
    mean_d = float(residual_dec.mean())
    out_json = OUT / "parity_32bias.json"
    with open(out_json, "w") as f:
        json.dump(dict(median_dec=median, p95_dec=p95, max_dec=max_d,
                       mean_dec=mean_d,
                       acceptance_gate=0.39,
                       PASS=bool(median <= 0.39),
                       n_biases=32, vd_panel=Vd_test.tolist()), f, indent=2)
    print(f"[J1] parity median={median:.4f}  p95={p95:.4f}  "
          f"max={max_d:.4f}  PASS={median <= 0.39}", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    print(f"[J1] start — v5.5 surrogate build, APU T={thermal_temp_c():.1f}C",
          flush=True)
    grid, vg1, vg2, vd, cfg, M1, M2, bjt = build_dense_grid((46, 46, 48))
    print(f"[J1] running parity check (32 biases × 4 Vd)", flush=True)
    parity_32bias(grid, vg1, vg2, vd, cfg, M1, M2, bjt)
    print(f"[J1] done — total {time.time()-t0:.1f}s", flush=True)
