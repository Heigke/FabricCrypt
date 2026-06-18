"""z283 / PTP — massively parallel GPU transient probe.

Batched body-state ODE simulation: 1000+ transients simultaneously.
Scales PMP-9 single-transient code path to (B, T) batched tensors.

Surrogate: results/z271_pmp3_dense_surrogate/surrogate_4d_v2.npz
  (same as z273 PMP-9, for exact validation match)

Pipeline:
  1) Validation: 16-point batch reproduces single PMP-9 calls within 1e-6.
  2) Production grid: V_G1 × V_G2 × t_rise × C_b = 6×4×5×6 = 720 transients
     run as one batch on GPU.
  3) 4D summary array + heatmaps.

Outputs:
  results/z283_ptp/summary.json
  results/z283_ptp/{S_fire,V_b_peak,S_relax,max_I_d}_heatmap.png
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
SURR = ROOT / "results/z271_pmp3_dense_surrogate/surrogate_4d_v2.npz"
OUT = ROOT / "results/z283_ptp"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
print(f"[z283] device={DEVICE} dtype={DTYPE}")


# ─────────────────────── surrogate I/O (same as z273) ───────────────────
def load_surrogate():
    d = np.load(SURR)
    Id = torch.tensor(d["Id"], dtype=DTYPE, device=DEVICE)
    Iii = torch.tensor(d["Iii"], dtype=DTYPE, device=DEVICE)
    Ileak = torch.tensor(d["Ileak"], dtype=DTYPE, device=DEVICE)
    axes = {
        "vg1": torch.tensor(d["vg1_axis"], dtype=DTYPE, device=DEVICE),
        "vg2": torch.tensor(d["vg2_axis"], dtype=DTYPE, device=DEVICE),
        "vd": torch.tensor(d["vd_axis"], dtype=DTYPE, device=DEVICE),
        "vb": torch.tensor(d["vb_axis"], dtype=DTYPE, device=DEVICE),
    }
    print(f"[z283] surrogate Id shape={tuple(Id.shape)} "
          f"vg1=[{axes['vg1'].min():.2f},{axes['vg1'].max():.2f}] "
          f"vg2=[{axes['vg2'].min():.2f},{axes['vg2'].max():.2f}] "
          f"vd=[{axes['vd'].min():.2f},{axes['vd'].max():.2f}] "
          f"vb=[{axes['vb'].min():.2f},{axes['vb'].max():.2f}]")
    return {"Id": Id, "Iii": Iii, "Ileak": Ileak, "axes": axes}


# ─────────────────── 4-D linear interp (batched, identical math) ──────
def _idx_frac(x: torch.Tensor, axis: torch.Tensor):
    n = axis.shape[0]
    idx = torch.searchsorted(axis, x.contiguous())
    idx = torch.clamp(idx, 1, n - 1)
    i0 = idx - 1
    i1 = idx
    a0 = axis[i0]
    a1 = axis[i1]
    w = (x - a0) / (a1 - a0 + 1e-30)
    w = torch.clamp(w, 0.0, 1.0)
    return i0, i1, w


def interp4d(table, axes, vg1, vg2, vd, vb):
    """Linear 4-D interp; all inputs are (B,)-shaped tensors (or broadcastable)."""
    i0_g1, i1_g1, w_g1 = _idx_frac(vg1, axes["vg1"])
    i0_g2, i1_g2, w_g2 = _idx_frac(vg2, axes["vg2"])
    i0_d, i1_d, w_d = _idx_frac(vd, axes["vd"])
    i0_b, i1_b, w_b = _idx_frac(vb, axes["vb"])

    def g(a, b, c, e):
        return table[a, b, c, e]

    out = 0.0
    for sg1, ig1 in ((1 - w_g1, i0_g1), (w_g1, i1_g1)):
        for sg2, ig2 in ((1 - w_g2, i0_g2), (w_g2, i1_g2)):
            for sd, id_ in ((1 - w_d, i0_d), (w_d, i1_d)):
                for sb, ib in ((1 - w_b, i0_b), (w_b, i1_b)):
                    out = out + sg1 * sg2 * sd * sb * g(ig1, ig2, id_, ib)
    return out


# ─────────────────────── batched V_d profile ──────────────────────────
def triangle_vd_batched(t_1d: torch.Tensor,
                        V_set_b: torch.Tensor,
                        t_rise_b: torch.Tensor,
                        t_set_b: torch.Tensor,
                        t_fall_b: torch.Tensor) -> torch.Tensor:
    """Per-batch triangle pulse. Returns (B, T)."""
    # broadcast: t (1, T), params (B, 1)
    t = t_1d.unsqueeze(0)                      # (1, T)
    V_set = V_set_b.unsqueeze(1)               # (B, 1)
    t_rise = t_rise_b.unsqueeze(1)
    t_set = t_set_b.unsqueeze(1)
    t_fall = t_fall_b.unsqueeze(1)

    r = t < t_rise
    h = (t >= t_rise) & (t < t_rise + t_set)
    f = (t >= t_rise + t_set) & (t < t_rise + t_set + t_fall)

    vd = torch.zeros_like(t.expand_as(V_set + t).contiguous())
    # rising
    vd = torch.where(r, V_set * (t / t_rise), vd)
    # hold
    vd = torch.where(h, V_set.expand_as(vd), vd)
    # falling
    vd = torch.where(f,
                     V_set * (1.0 - (t - t_rise - t_set) / t_fall),
                     vd)
    return vd  # (B, T)


# ─────────────────────── batched simulate ─────────────────────────────
def simulate_batch(surr,
                   VG1: torch.Tensor, VG2: torch.Tensor,
                   V_set: torch.Tensor,
                   t_rise: torch.Tensor,
                   t_set: torch.Tensor,
                   t_fall: torch.Tensor,
                   C_b_F: torch.Tensor,
                   dt_sim: float,
                   pad_factor: float = 0.3) -> dict:
    """Batched forward-Euler body-state. All input tensors are (B,)."""
    B = VG1.shape[0]
    # uniform global time axis sized for the SLOWEST transient in the batch
    T_total = (t_rise + t_set + t_fall + pad_factor * t_rise).max().item()
    n_steps = int(np.ceil(T_total / dt_sim)) + 1
    t = torch.arange(n_steps, dtype=DTYPE, device=DEVICE) * dt_sim   # (T,)

    V_d = triangle_vd_batched(t, V_set, t_rise, t_set, t_fall)        # (B, T)

    vb_min = surr["axes"]["vb"].min().item()
    vb_max = surr["axes"]["vb"].max().item()

    V_b = torch.zeros((B, n_steps), dtype=DTYPE, device=DEVICE)
    I_d = torch.zeros_like(V_b)
    I_ii = torch.zeros_like(V_b)
    I_leak = torch.zeros_like(V_b)

    vb = torch.zeros(B, dtype=DTYPE, device=DEVICE)
    railed = torch.zeros(B, dtype=torch.bool, device=DEVICE)
    invC = 1.0 / C_b_F                                                # (B,)

    for k in range(n_steps):
        V_b[:, k] = vb
        vd_k = V_d[:, k]                                              # (B,)
        id_k = interp4d(surr["Id"], surr["axes"], VG1, VG2, vd_k, vb)
        iii_k = interp4d(surr["Iii"], surr["axes"], VG1, VG2, vd_k, vb)
        ile_k = interp4d(surr["Ileak"], surr["axes"], VG1, VG2, vd_k, vb)
        I_d[:, k] = id_k
        I_ii[:, k] = iii_k
        I_leak[:, k] = ile_k
        if k + 1 < n_steps:
            dvb = (iii_k - ile_k) * invC * dt_sim                     # (B,)
            vb_new = vb + dvb
            # rail detection
            non_fin = ~torch.isfinite(vb_new)
            out_lo = vb_new < (vb_min - 0.05)
            out_hi = vb_new > (vb_max + 0.05)
            railed = railed | non_fin | out_lo | out_hi
            vb_new = torch.where(non_fin, vb, vb_new)
            vb = torch.clamp(vb_new, vb_min, vb_max)

    return {
        "t": t, "V_d": V_d, "V_b": V_b, "I_d": I_d,
        "I_ii": I_ii, "I_leak": I_leak,
        "dt": dt_sim, "n_steps": n_steps,
        "railed": railed,
    }


def extract_slopes_batched(res, t_rise, t_set, t_fall):
    """Per-batch S_fire/S_relax/V_b_peak/max_I_d. Returns dict of (B,) tensors."""
    t = res["t"]                       # (T,)
    I = res["I_d"]                     # (B, T)
    dI = torch.diff(I, dim=1)          # (B, T-1)
    dt = torch.diff(t)                 # (T-1,)
    slope = dI / dt.unsqueeze(0)       # (B, T-1)
    t_mid = 0.5 * (t[:-1] + t[1:])     # (T-1,)
    # per-batch masks
    rise_end = t_rise.unsqueeze(1)                                # (B,1)
    fall_start = (t_rise + t_set).unsqueeze(1)
    fall_end = (t_rise + t_set + t_fall).unsqueeze(1)
    tm = t_mid.unsqueeze(0)                                       # (1,T-1)
    rise_mask = tm < rise_end
    fall_mask = (tm >= fall_start) & (tm < fall_end)

    NEG_INF = torch.finfo(DTYPE).min
    slope_rise = torch.where(rise_mask, slope, torch.full_like(slope, NEG_INF))
    slope_fall = torch.where(fall_mask, -slope, torch.full_like(slope, NEG_INF))
    s_fire = slope_rise.max(dim=1).values
    s_relax = slope_fall.max(dim=1).values

    return dict(
        S_fire=s_fire,
        S_relax=s_relax,
        V_b_peak=res["V_b"].max(dim=1).values,
        max_I_d=I.max(dim=1).values,
        railed=res["railed"],
    )


# ─────────────────────── PMP-9 single-shot (validation oracle) ────────
def _single_pmp9(surr, VG1, VG2, V_set, t_rise, t_set, t_fall, C_b_F,
                 dt_sim):
    """Exact PMP-9 forward-Euler. Mirrors z273.simulate_ramp body."""
    T_total = t_rise + t_set + t_fall + 0.3 * t_rise
    n_steps = int(np.ceil(T_total / dt_sim)) + 1
    t = torch.arange(n_steps, dtype=DTYPE, device=DEVICE) * dt_sim
    # triangle (single)
    V_d = torch.zeros_like(t)
    r = t < t_rise
    h = (t >= t_rise) & (t < t_rise + t_set)
    f = (t >= t_rise + t_set) & (t < t_rise + t_set + t_fall)
    V_d = torch.where(r, V_set * (t / t_rise), V_d)
    V_d = torch.where(h, torch.full_like(V_d, V_set), V_d)
    V_d = torch.where(f, V_set * (1.0 - (t - t_rise - t_set) / t_fall), V_d)

    vb_min = surr["axes"]["vb"].min().item()
    vb_max = surr["axes"]["vb"].max().item()
    V_b = torch.zeros_like(t)
    I_d = torch.zeros_like(t)
    vb = torch.tensor(0.0, dtype=DTYPE, device=DEVICE)
    railed = False
    for k in range(n_steps):
        V_b[k] = vb
        vd_k = V_d[k:k + 1]
        vb_k = vb.unsqueeze(0)
        vg1_k = torch.full_like(vd_k, VG1)
        vg2_k = torch.full_like(vd_k, VG2)
        id_k = interp4d(surr["Id"], surr["axes"], vg1_k, vg2_k, vd_k, vb_k)
        iii_k = interp4d(surr["Iii"], surr["axes"], vg1_k, vg2_k, vd_k, vb_k)
        ile_k = interp4d(surr["Ileak"], surr["axes"], vg1_k, vg2_k, vd_k, vb_k)
        I_d[k] = id_k
        if k + 1 < n_steps:
            dvb = (iii_k - ile_k) / C_b_F
            vb = vb + dvb.squeeze() * dt_sim
            if not torch.isfinite(vb):
                railed = True
                break
            if vb.item() < vb_min - 0.05 or vb.item() > vb_max + 0.05:
                railed = True
                vb = torch.clamp(vb, vb_min, vb_max)
    return dict(V_d=V_d, V_b=V_b, I_d=I_d, railed=railed)


# ─────────────────────── validation ───────────────────────────────────
def run_validation(surr):
    print("[z283] === validation: 16-point batch vs single PMP-9 ===")
    torch.manual_seed(0)
    B = 16
    # diverse but in-grid params; slide-21 included as element 0
    rng = np.random.default_rng(0)
    vg1_v = np.concatenate([[0.45], rng.uniform(0.20, 0.60, B - 1)])
    vg2_v = np.concatenate([[0.30], rng.uniform(0.10, 0.40, B - 1)])
    trise_v = np.concatenate([[200e-6],
                              rng.choice([10e-6, 100e-6, 200e-6, 1e-3],
                                         B - 1)])
    cb_v = np.concatenate([[14e-15],
                           rng.choice([2e-15, 5e-15, 8e-15, 14e-15, 20e-15],
                                      B - 1)])
    V_set = 2.05
    t_set = 1e-6
    # use uniform dt_sim across batch (slowest dominates) — pick 5e-9 (fine)
    dt_sim = 5e-9

    VG1 = torch.tensor(vg1_v, dtype=DTYPE, device=DEVICE)
    VG2 = torch.tensor(vg2_v, dtype=DTYPE, device=DEVICE)
    Vset = torch.full((B,), V_set, dtype=DTYPE, device=DEVICE)
    trise = torch.tensor(trise_v, dtype=DTYPE, device=DEVICE)
    tset = torch.full((B,), t_set, dtype=DTYPE, device=DEVICE)
    tfall = trise.clone()
    Cb = torch.tensor(cb_v, dtype=DTYPE, device=DEVICE)

    # batched run
    t0 = time.time()
    batch_res = simulate_batch(surr, VG1, VG2, Vset, trise, tset, tfall, Cb,
                               dt_sim=dt_sim)
    batch_extract = extract_slopes_batched(batch_res, trise, tset, tfall)
    t_batch = time.time() - t0
    print(f"    batched 16: {t_batch:.2f}s  n_steps={batch_res['n_steps']}")

    # single-shot oracle for each
    t0 = time.time()
    max_abs_err_Id = 0.0
    max_abs_err_Vb = 0.0
    max_abs_err_Sfire = 0.0
    max_abs_err_VbPeak = 0.0
    for b in range(B):
        single = _single_pmp9(surr,
                              vg1_v[b], vg2_v[b], V_set,
                              trise_v[b], t_set, trise_v[b],
                              cb_v[b], dt_sim)
        # both use same global time axis (same dt, same T_total because same
        # t_rise=t_fall=longest_in_batch?) NO — single uses its own T_total.
        # Compare overlap region only.
        n_s = single["I_d"].shape[0]
        n_b = batch_res["I_d"].shape[1]
        n = min(n_s, n_b)
        eId = (batch_res["I_d"][b, :n] - single["I_d"][:n]).abs().max().item()
        eVb = (batch_res["V_b"][b, :n] - single["V_b"][:n]).abs().max().item()
        max_abs_err_Id = max(max_abs_err_Id, eId)
        max_abs_err_Vb = max(max_abs_err_Vb, eVb)

        # S_fire from single
        dI = torch.diff(single["I_d"])
        dt = dt_sim
        slope = dI / dt
        t_single = torch.arange(n_s, dtype=DTYPE, device=DEVICE) * dt_sim
        tm = 0.5 * (t_single[:-1] + t_single[1:])
        rm = tm < trise_v[b]
        s_fire_single = slope[rm].max().item() if rm.any() else float("nan")
        s_fire_batch = batch_extract["S_fire"][b].item()
        max_abs_err_Sfire = max(max_abs_err_Sfire,
                                abs(s_fire_single - s_fire_batch))
        vb_peak_single = single["V_b"].max().item()
        vb_peak_batch = batch_extract["V_b_peak"][b].item()
        max_abs_err_VbPeak = max(max_abs_err_VbPeak,
                                 abs(vb_peak_single - vb_peak_batch))
    t_single = time.time() - t0
    print(f"    16 single  : {t_single:.2f}s")
    print(f"    max |ΔI_d| = {max_abs_err_Id:.3e} A")
    print(f"    max |ΔV_b| = {max_abs_err_Vb:.3e} V")
    print(f"    max |ΔS_fire| = {max_abs_err_Sfire:.3e} A/s")
    print(f"    max |ΔV_b_peak| = {max_abs_err_VbPeak:.3e} V")
    return {
        "B": B,
        "t_batch_s": t_batch,
        "t_single_s": t_single,
        "speedup": t_single / t_batch if t_batch > 0 else float("inf"),
        "max_abs_err_Id_A": max_abs_err_Id,
        "max_abs_err_Vb_V": max_abs_err_Vb,
        "max_abs_err_Sfire_A_per_s": max_abs_err_Sfire,
        "max_abs_err_VbPeak_V": max_abs_err_VbPeak,
    }


# ─────────────────────── production grid ──────────────────────────────
VG1_AX = [0.20, 0.30, 0.40, 0.45, 0.50, 0.60]
VG2_AX = [0.10, 0.20, 0.30, 0.40]
TRISE_AX = [1e-6, 10e-6, 100e-6, 1e-3, 10e-3]
CB_AX = [2e-15, 5e-15, 8e-15, 14e-15, 20e-15, 50e-15]
V_SET = 2.05
T_SET = 1e-6


def run_grid(surr):
    print("[z283] === production grid 6×4×5×6 = 720 ===")
    g1, g2, tr, cb = np.meshgrid(VG1_AX, VG2_AX, TRISE_AX, CB_AX,
                                 indexing="ij")
    shape = g1.shape  # (6,4,5,6)
    vg1 = g1.ravel(); vg2 = g2.ravel(); trise = tr.ravel(); cbv = cb.ravel()
    B = vg1.size
    print(f"    B={B} shape={shape}")
    # dt_sim per-element ideal = min(t_rise,t_fall)/200; uniform dt is
    # constrained to the FINEST resolution required → dt = min(t_rise)/200
    # = 1e-6/200 = 5e-9. n_steps for longest (10e-3*2.3) = 4.6e6. Too big.
    # Strategy: bucket by t_rise so each bucket uses its own uniform dt.
    # 5 buckets total → 5 GPU batches, still massively parallel within each.
    summary = {
        "axes": {"V_G1": VG1_AX, "V_G2": VG2_AX,
                 "t_rise_s": TRISE_AX, "C_b_F": CB_AX},
        "V_set_V": V_SET, "t_set_s": T_SET,
        "shape": list(shape),
        "S_fire": np.zeros(shape, dtype=np.float64),
        "S_relax": np.zeros(shape, dtype=np.float64),
        "V_b_peak": np.zeros(shape, dtype=np.float64),
        "max_I_d": np.zeros(shape, dtype=np.float64),
        "railed": np.zeros(shape, dtype=bool),
    }

    # bucket by t_rise
    t_per_bucket = {}
    for i, tr_val in enumerate(TRISE_AX):
        mask = trise == tr_val
        idx = np.where(mask)[0]
        Bk = idx.size
        dt_sim = tr_val / 200.0
        T_total = tr_val + T_SET + tr_val + 0.3 * tr_val
        n_steps = int(np.ceil(T_total / dt_sim)) + 1
        print(f"    bucket t_rise={tr_val:.0e}  B={Bk}  dt={dt_sim:.1e}  "
              f"n_steps={n_steps}", flush=True)

        VG1 = torch.tensor(vg1[idx], dtype=DTYPE, device=DEVICE)
        VG2 = torch.tensor(vg2[idx], dtype=DTYPE, device=DEVICE)
        Vset = torch.full((Bk,), V_SET, dtype=DTYPE, device=DEVICE)
        trb = torch.full((Bk,), tr_val, dtype=DTYPE, device=DEVICE)
        tsb = torch.full((Bk,), T_SET, dtype=DTYPE, device=DEVICE)
        tfb = trb.clone()
        Cb = torch.tensor(cbv[idx], dtype=DTYPE, device=DEVICE)

        t0 = time.time()
        res = simulate_batch(surr, VG1, VG2, Vset, trb, tsb, tfb, Cb,
                             dt_sim=dt_sim)
        ext = extract_slopes_batched(res, trb, tsb, tfb)
        torch.cuda.synchronize() if DEVICE.type == "cuda" else None
        dt_wall = time.time() - t0
        t_per_bucket[f"{tr_val:.0e}"] = dt_wall
        print(f"        wall={dt_wall:.2f}s  "
              f"S_fire range [{ext['S_fire'].min():.2e},"
              f"{ext['S_fire'].max():.2e}]  "
              f"railed={int(ext['railed'].sum())}/{Bk}", flush=True)

        # scatter into 4D arrays
        s_fire_np = ext["S_fire"].cpu().numpy()
        s_relax_np = ext["S_relax"].cpu().numpy()
        vbp_np = ext["V_b_peak"].cpu().numpy()
        mid_np = ext["max_I_d"].cpu().numpy()
        rail_np = ext["railed"].cpu().numpy()
        # unravel idx → (i_g1, i_g2, i_tr, i_cb)
        for j, flat in enumerate(idx):
            i_g1, i_g2, i_tr, i_cb = np.unravel_index(flat, shape)
            summary["S_fire"][i_g1, i_g2, i_tr, i_cb] = s_fire_np[j]
            summary["S_relax"][i_g1, i_g2, i_tr, i_cb] = s_relax_np[j]
            summary["V_b_peak"][i_g1, i_g2, i_tr, i_cb] = vbp_np[j]
            summary["max_I_d"][i_g1, i_g2, i_tr, i_cb] = mid_np[j]
            summary["railed"][i_g1, i_g2, i_tr, i_cb] = rail_np[j]

        # free
        del res, ext, VG1, VG2, Vset, trb, tsb, tfb, Cb
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    summary["t_per_bucket_s"] = t_per_bucket
    # NaN check
    any_nan = (np.isnan(summary["S_fire"]).any() or
               np.isnan(summary["V_b_peak"]).any())
    summary["any_nan"] = bool(any_nan)
    return summary


# ─────────────────────── analysis & plots ─────────────────────────────
def analyze_and_plot(summary):
    Sf = summary["S_fire"]                # (6,4,5,6)
    Sr = summary["S_relax"]
    Vbp = summary["V_b_peak"]
    Imx = summary["max_I_d"]

    # Best (V_G1, V_G2, t_rise, C_b) for max S_fire
    flat = Sf.ravel()
    bi = int(np.argmax(flat))
    i_g1, i_g2, i_tr, i_cb = np.unravel_index(bi, Sf.shape)
    best = {
        "V_G1": VG1_AX[i_g1],
        "V_G2": VG2_AX[i_g2],
        "t_rise_s": TRISE_AX[i_tr],
        "C_b_F": CB_AX[i_cb],
        "S_fire_A_per_s": float(Sf[i_g1, i_g2, i_tr, i_cb]),
        "S_relax_A_per_s": float(Sr[i_g1, i_g2, i_tr, i_cb]),
        "V_b_peak_V": float(Vbp[i_g1, i_g2, i_tr, i_cb]),
        "max_I_d_A": float(Imx[i_g1, i_g2, i_tr, i_cb]),
    }
    print(f"[z283] BEST S_fire @ V_G1={best['V_G1']:.2f} "
          f"V_G2={best['V_G2']:.2f} t_rise={best['t_rise_s']:.0e} "
          f"C_b={best['C_b_F']:.0e} → S_fire={best['S_fire_A_per_s']:.3e} A/s")

    # slide-21 indices: V_G1=0.45 → idx 3, V_G2=0.30 → idx 2
    iG1_s21 = VG1_AX.index(0.45)
    iG2_s21 = VG2_AX.index(0.30)

    def heatmap(M2d, title, fname, log=False):
        fig, ax = plt.subplots(figsize=(7, 5))
        data = np.log10(np.abs(M2d) + 1e-30) if log else M2d
        im = ax.imshow(data, aspect="auto", origin="lower",
                       extent=[0, len(CB_AX), 0, len(TRISE_AX)],
                       cmap="viridis")
        ax.set_xticks(np.arange(len(CB_AX)) + 0.5)
        ax.set_xticklabels([f"{c*1e15:.0f}" for c in CB_AX])
        ax.set_yticks(np.arange(len(TRISE_AX)) + 0.5)
        ax.set_yticklabels([f"{t*1e6:.0f}" for t in TRISE_AX])
        ax.set_xlabel("C_b  [fF]")
        ax.set_ylabel("t_rise  [µs]")
        ax.set_title(title)
        cb = plt.colorbar(im, ax=ax)
        cb.set_label("log10 |val|" if log else "value")
        fig.tight_layout()
        fig.savefig(OUT / fname, dpi=130)
        plt.close(fig)

    # slice at slide-21 V_G1/V_G2 → 2D (t_rise, C_b)
    heatmap(Sf[iG1_s21, iG2_s21],
            "S_fire vs (t_rise, C_b) @ V_G1=0.45 V_G2=0.30",
            "S_fire_heatmap.png", log=True)
    heatmap(Vbp[iG1_s21, iG2_s21],
            "V_b_peak [V] vs (t_rise, C_b) @ V_G1=0.45 V_G2=0.30",
            "V_b_peak_heatmap.png", log=False)
    heatmap(Sr[iG1_s21, iG2_s21],
            "S_relax vs (t_rise, C_b) @ V_G1=0.45 V_G2=0.30",
            "S_relax_heatmap.png", log=True)
    heatmap(Imx[iG1_s21, iG2_s21],
            "max_I_d [A] vs (t_rise, C_b) @ V_G1=0.45 V_G2=0.30",
            "max_I_d_heatmap.png", log=True)

    # monotonicity checks (ambitious gate)
    sf_slice = Sf[iG1_s21, iG2_s21]               # (t_rise, C_b)
    vbp_slice = Vbp[iG1_s21, iG2_s21]
    # S_fire vs t_rise (fix C_b=14fF idx 3): should be monotonically *decreasing*
    iCb_s21 = CB_AX.index(14e-15)
    sf_vs_trise = sf_slice[:, iCb_s21]
    sf_mono_trise = bool(np.all(np.diff(sf_vs_trise) <= 0))
    # V_b_peak vs C_b (fix t_rise=200µs not in grid; use 100µs idx 2): should *decrease* with larger C_b
    iTr_pick = TRISE_AX.index(100e-6)
    vbp_vs_cb = vbp_slice[iTr_pick]
    vbp_mono_cb = bool(np.all(np.diff(vbp_vs_cb) <= 0))

    return {
        "best_S_fire": best,
        "monotonicity": {
            "S_fire_decreasing_in_t_rise_at_slide21_Cb14fF": sf_mono_trise,
            "S_fire_vs_t_rise_values": sf_vs_trise.tolist(),
            "V_b_peak_decreasing_in_C_b_at_trise100us": vbp_mono_cb,
            "V_b_peak_vs_C_b_values": vbp_vs_cb.tolist(),
        },
    }


# ─────────────────────── main ─────────────────────────────────────────
def main():
    t_start = time.time()
    surr = load_surrogate()

    val = run_validation(surr)
    if val["max_abs_err_Id_A"] > 1e-6 or val["max_abs_err_Vb_V"] > 1e-6:
        print(f"[z283] WARN validation exceeds 1e-6 tolerance — "
              f"I_d err {val['max_abs_err_Id_A']:.2e}, "
              f"V_b err {val['max_abs_err_Vb_V']:.2e}")

    grid = run_grid(surr)
    analysis = analyze_and_plot(grid)

    # JSON-safe dump
    out = {
        "validation": val,
        "axes": grid["axes"],
        "V_set_V": grid["V_set_V"],
        "t_set_s": grid["t_set_s"],
        "shape": grid["shape"],
        "S_fire": grid["S_fire"].tolist(),
        "S_relax": grid["S_relax"].tolist(),
        "V_b_peak": grid["V_b_peak"].tolist(),
        "max_I_d": grid["max_I_d"].tolist(),
        "railed": grid["railed"].astype(int).tolist(),
        "any_nan": grid["any_nan"],
        "t_per_bucket_s": grid["t_per_bucket_s"],
        "analysis": analysis,
        "wall_total_s": time.time() - t_start,
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[z283] DONE wall={out['wall_total_s']:.1f}s "
          f"any_nan={out['any_nan']}")


if __name__ == "__main__":
    main()
