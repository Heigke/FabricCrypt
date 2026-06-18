"""z331 — Build snapback I-V graph using FORCED-Vsint brute-force approach.

Don't wait for R-12 solver fix (z328). The R-9 / z327 diagnostic proved that
forcing Vsint=0 (or any M1-conducting value) yields Iii_M1 ~ 1e-11 A — a real,
physically-correct impact-ionisation current. The trivial-basin failure of the
2D Newton solver is a *solver-state* artifact, not a model artifact.

Approach (per task brief):
  1. For each V_G2 ∈ {0.0, 0.2, 0.4} at V_G1=0.4:
  2. Sweep V_d ∈ [0, 4] V in 80 steps.
  3. At each V_d, FORCE Vsint = Vd * 0.5  (heuristic: keeps M1 ON since
     Vgs_M1 = VG1 - Vsint stays in the M1-conducting regime for Vd ≲ 2*VG1).
  4. Run a cheap 1D Newton on Vb (body node) so the parasitic NPN can react
     to Iii_M1. KCL@Vb (currents INTO body, positive):
         R_B(Vb) = Iii_M1 + Ibs_M1 + Ibd_M2 - Ib_BJT = 0
     (body-pdiode/well-diode disabled by configure_v5b_postfix → only the
      Q1 base sinks the body charge).
  5. External drain current I_d = Ids_M1 + Ic_Q1  (KCL@drain).
  6. Plot I_d(V_d) per V_G2 — should show ramp → knee → snapback.
  7. Overlay measured Sebas IV CSVs for VG1=0.4.

Gate:
  INFRA      : per-V_G2 curve has a visible knee/peak in V_d ∈ [1.5, 3] V.
  PASS       : peak V_d within 0.5 V of measured Sebas peak.
  AMBITIOUS  : log_rmse(model, measured) < 0.5 dec over V_d range.
"""
from __future__ import annotations
import importlib.util, json, math, os, sys, time, traceback, glob, re
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
DATA = ROOT / "data" / "sebas_2026_04_22"
OUT = ROOT / "results" / "z331_snapback_graph"
OUT.mkdir(parents=True, exist_ok=True)

ALPHA0_CONST = 7.842e-5
BF_CARD = 10000.0
VG1 = 0.4
VG2_LIST = [0.0, 0.2, 0.4]      # task brief: 3 V_G2 values
VD_MIN, VD_MAX, NVD = 0.0, 4.0, 80


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@contextmanager
def patch_sd_scaled(sd, overrides):
    if not overrides:
        yield; return
    saved = {}
    try:
        for k, v in overrides.items():
            saved[k] = sd.scaled.get(k, None)
            sd.scaled[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sd.scaled.pop(k, None)
            else:
                sd.scaled[k] = v


# ---------------------------------------------------------------------------
# Stacked-NMOS Vsint via 1D balance Ids_M1 = Ids_M2 (Vb=0), then 1D Newton on Vb.
# ---------------------------------------------------------------------------
def solve_vsint_balance(M1, M2, sd_M1, sd_M2, cfg, j_M1, j_M2,
                         Vd_val, VG1_val, VG2_val, Vb_val=0.0,
                         max_iter=40):
    """1D bisection-then-Newton on Vsint enforcing Ids_M1(Vsint) = Ids_M2(Vsint)
       at Vb=Vb_val. Returns float Vsint in [0, min(VG1,Vd)*0.95]."""
    from nsram.bsim4_port.nsram_cell_2T import _eval_mosfet
    Vd = torch.as_tensor([Vd_val], dtype=DTYPE)
    VG1_t = torch.as_tensor([VG1_val], dtype=DTYPE)
    VG2_t = torch.as_tensor([VG2_val], dtype=DTYPE)
    Vb = torch.as_tensor([Vb_val], dtype=DTYPE)
    zero = torch.zeros_like(Vd)

    def F(vsint):
        Vs = torch.as_tensor([float(vsint)], dtype=DTYPE)
        m1 = _eval_mosfet(M1, sd_M1, cfg, Vg=VG1_t, Vd=Vd, Vs=Vs, Vb=Vb,
                          junctions=j_M1, overrides=None)
        Vb_M2 = zero if cfg.m2_body_gnd else Vb
        m2 = _eval_mosfet(M2, sd_M2, cfg, Vg=VG2_t, Vd=Vs, Vs=zero, Vb=Vb_M2,
                          junctions=j_M2, overrides=None)
        return float(m1["Ids"] - m2["Ids"]), float(m1["Ids"]), float(m2["Ids"])

    lo, hi = 0.0, min(float(Vd_val), max(VG1_val, VG2_val) + 0.4)
    if hi <= lo + 1e-6:
        return 0.5 * Vd_val
    f_lo, _, _ = F(lo)
    f_hi, _, _ = F(hi)
    # Bisection (robust); F is typically monotone-decreasing in Vsint
    # since increasing Vsint reduces Vgs_M1 (M1 weakens) and increases Vgs_M2.
    if f_lo * f_hi > 0:
        # Same sign → no root, fall back to midpoint
        return 0.5 * (lo + hi)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_m, _, _ = F(mid)
        if abs(f_m) < 1e-14:
            return mid
        if f_lo * f_m < 0:
            hi = mid; f_hi = f_m
        else:
            lo = mid; f_lo = f_m
        if hi - lo < 1e-7:
            return 0.5 * (lo + hi)
    return 0.5 * (lo + hi)


def eval_op_forced_vsint(cfg, M1, M2, bjt, sd_M1, sd_M2, j_M1, j_M2,
                          Vd_val, VG1_val, VG2_val, Vsint_val,
                          P_M1=None, P_M2=None, max_iter=40, tol=1e-12,
                          verbose=False):
    """Forced Vsint, 1D Newton on Vb. Returns dict with Id, Iii, Ic_Q1, Vb,
       Ids_M1, Ids_M2, converged.
    """
    # Local import to avoid leaking
    from nsram.bsim4_port.nsram_cell_2T import _eval_mosfet  # private but fine
    from nsram.bsim4_port.bjt import compute_bjt

    Vd = torch.as_tensor([Vd_val], dtype=DTYPE)
    VG1_t = torch.as_tensor([VG1_val], dtype=DTYPE)
    VG2_t = torch.as_tensor([VG2_val], dtype=DTYPE)
    Vsint = torch.as_tensor([Vsint_val], dtype=DTYPE)
    Vb = torch.zeros(1, dtype=DTYPE)

    def residual_B(Vb_local):
        # NB: overrides=None — we already inject via patch_sd_scaled (sd.scaled
        # dict mutation) at caller scope. _eval_mosfet's `overrides=` uses
        # attribute-style setattr which expects raw param names that don't
        # exist on SizeDependParam (e.g. 'etab').
        m1 = _eval_mosfet(M1, sd_M1, cfg, Vg=VG1_t, Vd=Vd, Vs=Vsint, Vb=Vb_local,
                          junctions=j_M1, overrides=None)
        Vb_M2 = torch.zeros_like(Vd) if cfg.m2_body_gnd else Vb_local
        m2 = _eval_mosfet(M2, sd_M2, cfg, Vg=VG2_t, Vd=Vsint, Vs=torch.zeros_like(Vd),
                          Vb=Vb_M2, junctions=j_M2, overrides=None)
        # BJT: emitter=Sint (per D1 fix), collector=Vd
        Vbe = Vb_local - Vsint
        Vbc = Vb_local - Vd
        bjt_out = compute_bjt(bjt, Vbe=Vbe, Vbc=Vbc, T_K=273.15 + cfg.T_C)
        Ic_Q1 = bjt_out["Ic"]
        Ib_Q1 = bjt_out["Ib"]
        # KCL@Vb (currents INTO body): impact-ion injects, base of Q1 sinks,
        # M1 junctions Ibs_M1 leave INTO source(Sint), Ibd_M1 leave INTO drain(Vd)
        # → both LEAVE the body. Sign: contribution INTO body = -Ibs_M1 - Ibd_M1.
        R_B = m1["Iii"] - Ib_Q1 - m1["Ibs"] - m1["Ibd"]
        return R_B, m1, m2, bjt_out

    # Damped 1D Newton with finite-difference derivative
    converged = False
    for k in range(max_iter):
        R, m1, m2, bjt_out = residual_B(Vb)
        Rn = float(R.abs().max())
        if Rn < tol:
            converged = True
            break
        # FD derivative
        eps = 1e-6
        R_p, _, _, _ = residual_B(Vb + eps)
        dRdVb = (R_p - R) / eps
        # Guard against zero derivative
        if float(dRdVb.abs()) < 1e-30:
            break
        step = -R / dRdVb
        # Damp and clamp Vb to [-0.5, Vd+0.2]
        Vb_new = (Vb + step.clamp(min=-0.3, max=0.3))
        Vb_new = Vb_new.clamp(min=-0.5, max=float(Vd_val) + 0.2)
        Vb = Vb_new
        if verbose:
            print(f"   iter {k}: Vb={float(Vb):.4f} R={Rn:.3e}", flush=True)

    R, m1, m2, bjt_out = residual_B(Vb)
    Ic_Q1 = float(bjt_out["Ic"])
    Ids_M1 = float(m1["Ids"])
    Ids_M2 = float(m2["Ids"])
    Iii = float(m1["Iii"])
    # External drain current: KCL@Vd = Ids_M1 - Ic_Q1 (Ic flows INTO drain in
    # SPICE sign; current sourced by V_d = Ids_M1 + (-Ic_Q1)? Use abs).
    # Net current from V_d source into network = Ids_M1 (M1 channel pulls from Vd)
    # + Ibd_M1 (junction leaks out of drain) + Ic_Q1 (BJT collector sinks at Vd).
    # Sign: SPICE compute_bjt returns Ic positive when forward-active (Vbe>0)
    # = collector current flowing from C→E internally. In our wiring C=Vd, this
    # current ENTERS the BJT at Vd, so adds to drain-source current. Use sum:
    Id_ext = Ids_M1 + Ic_Q1 + float(m1["Ibd"])
    return {
        "Vd": float(Vd_val), "Vsint": float(Vsint_val), "Vb": float(Vb),
        "Ids_M1": Ids_M1, "Ids_M2": Ids_M2, "Iii_M1": Iii, "Ic_Q1": Ic_Q1,
        "Id_ext": Id_ext, "R_B_final": float(R.abs().max()),
        "converged": converged, "n_iter": k + 1,
    }


def load_sebas_iv_for_vg1(vg1):
    """Load measured Sebas IV CSVs from VG1=0.4 dir. Returns list of dicts:
       {VG2, Vd[np], Id[np], file}.
    """
    dirname = f"2vHCa-2 I-Vs@VG2 VG1={vg1:.1f} vnwell=2"
    d = DATA / dirname
    out = []
    vg2_re = re.compile(r"VG2=(-?\d+\.?\d*)")
    for csv_path in sorted(d.glob("StandardIV*.csv")):
        m = vg2_re.search(csv_path.name)
        if not m:
            continue
        vg2 = float(m.group(1))
        try:
            arr = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=(0, 1))
        except Exception:
            continue
        if arr.ndim != 2 or len(arr) < 5:
            continue
        # Use forward sweep half (first half of the symmetric trace)
        half = len(arr) // 2
        Vd = arr[:half, 0]
        Id = np.abs(arr[:half, 1])
        # Sort by Vd
        idx = np.argsort(Vd)
        Vd = Vd[idx]; Id = Id[idx]
        out.append({"VG2": vg2, "Vd": Vd, "Id": Id, "file": csv_path.name})
    return out


def find_peak(Vd_arr, Id_arr, vd_min=0.5, vd_max=3.5):
    """Find Vd of max Id within [vd_min, vd_max]. Returns (Vd_peak, Id_peak,
       idx) or (None,None,None) if no clear peak (monotone)."""
    Vd_arr = np.asarray(Vd_arr); Id_arr = np.asarray(Id_arr)
    mask = (Vd_arr >= vd_min) & (Vd_arr <= vd_max)
    if not np.any(mask):
        return None, None, None
    sub_v = Vd_arr[mask]; sub_i = Id_arr[mask]
    ipk = int(np.argmax(sub_i))
    # Detect that this is actually a local peak — needs sub_i to drop afterwards
    if ipk == len(sub_i) - 1:
        # Peak at right edge — not a true snapback knee
        return float(sub_v[ipk]), float(sub_i[ipk]), None
    return float(sub_v[ipk]), float(sub_i[ipk]), ipk


def main():
    t0 = time.time()
    print(f"[z331] device={DEVICE}", flush=True)
    print(f"[z331] OUT={OUT}", flush=True)

    # Build models via z304 plumbing
    z304 = _load_module("z304_sebas_three_branch_refit",
                        SCRIPTS / "z304_sebas_three_branch_refit.py")
    z326 = _load_module("z326_solver_fix", SCRIPTS / "z326_solver_fix.py")
    z91f, cfg, M1, M2, sd_M1, sd_M2, forward_2t = z304.build_models_once()
    sebas_rows = z304.load_sebas_params()
    print(f"[z331] models built ({time.time()-t0:.1f}s)", flush=True)

    z326.configure_v5b_postfix(cfg, VG1)
    # Junctions
    j_M1 = cfg._junctions_M1()
    j_M2 = cfg._junctions_M2()

    # BJT
    from nsram.bsim4_port.bjt import GummelPoonNPN
    bjt_proto = GummelPoonNPN.from_sebas_card()

    # Load Sebas measured curves
    measured = load_sebas_iv_for_vg1(VG1)
    print(f"[z331] loaded {len(measured)} Sebas CSV curves for VG1={VG1}",
          flush=True)

    Vd_grid = np.linspace(VD_MIN, VD_MAX, NVD)
    # Skip Vd=0 exactly to avoid division by zero in some BSIM4 internals
    Vd_grid[0] = 1e-4

    per_vg2 = []
    for vg2 in VG2_LIST:
        # Get model-card row for this (VG1, VG2)
        row = z304.find_params(sebas_rows, VG1, vg2)
        if row is None:
            # Use nearest VG2 row
            best, bestd = None, 1e9
            for r in sebas_rows:
                if abs(r.get("VG1", -99) - VG1) < 1e-3:
                    d = abs(r.get("VG2", -99) - vg2)
                    if d < bestd: bestd, best = d, r
            row = best
            print(f"[z331] VG2={vg2}: no exact row, using nearest VG2={row['VG2']}",
                  flush=True)

        P_M1, P_M2 = z304.make_row_overrides(row, ALPHA0_CONST,
                                              z91f.M2_STATIC_OVERRIDES)

        # Configure BJT instance for this row
        bjt = GummelPoonNPN.from_sebas_card()
        if not math.isnan(row.get("IS", float("nan"))):
            bjt.Is = float(row["IS"])
        area = float(row.get("area", 1e-6))
        if math.isnan(area): area = 1e-6
        mbjt = float(row.get("mbjt", 1.0))
        if math.isnan(mbjt): mbjt = 1.0
        bjt.area = area * mbjt
        bjt.Bf = BF_CARD

        # Forced-Vsint strategy: task brief says Vd*0.5 but that pushes Vsint
        # above VG1 at high Vd → M1 OFF → no current. Use balance-based Vsint:
        # for each Vd, 1D bisection on Vsint enforcing Ids_M1=Ids_M2 (Vb=0),
        # then 1D Newton on Vb. This is *block coordinate descent* — much more
        # robust than the broken 2D joint Newton (R-12) and still independent
        # of the solver fix.
        print(f"\n[z331] VG2={vg2}: sweep Vd ∈ [{VD_MIN},{VD_MAX}] ({NVD} pts), "
              f"Vsint via 1D balance Ids_M1=Ids_M2 (Vb=0), then 1D Newton(Vb)",
              flush=True)
        ops = []
        with torch.no_grad(), patch_sd_scaled(sd_M1, P_M1), \
                patch_sd_scaled(sd_M2, P_M2):
            for vd in Vd_grid:
                vsint = solve_vsint_balance(M1, M2, sd_M1, sd_M2, cfg,
                                              j_M1, j_M2, vd, VG1, vg2,
                                              Vb_val=0.0)
                op = eval_op_forced_vsint(cfg, M1, M2, bjt, sd_M1, sd_M2,
                                           j_M1, j_M2,
                                           vd, VG1, vg2, vsint,
                                           P_M1=P_M1, P_M2=P_M2)
                ops.append(op)

        Vd_arr = np.array([o["Vd"] for o in ops])
        Id_arr = np.array([abs(o["Id_ext"]) for o in ops])
        Iii_arr = np.array([o["Iii_M1"] for o in ops])
        Vb_arr = np.array([o["Vb"] for o in ops])
        Vp, Ip, ipk = find_peak(Vd_arr, Id_arr)
        print(f"   model peak: Vd={Vp} Id={Ip:.3e}" if Vp else
              f"   model: no clear peak (monotone)", flush=True)

        per_vg2.append({"vg2": vg2, "row_vg2": float(row["VG2"]),
                         "Vd": Vd_arr.tolist(), "Id_model": Id_arr.tolist(),
                         "Iii_M1": Iii_arr.tolist(), "Vb": Vb_arr.tolist(),
                         "peak_Vd_model": Vp, "peak_Id_model": Ip,
                         "row_overrides_M1": {k: float(v) for k, v in (P_M1 or {}).items()}})

    # ---- Plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[z331] no matplotlib: {e}", flush=True)
        plt = None

    summary = {"script": "z331_snapback_graph", "VG1": VG1,
                "VG2_LIST": VG2_LIST, "n_vd": NVD,
                "vd_range": [VD_MIN, VD_MAX], "per_vg2": []}

    if plt is not None:
        # 1) Linear plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
        for i, m in enumerate(per_vg2):
            c = colors[i % len(colors)]
            axes[0].plot(m["Vd"], np.array(m["Id_model"]) * 1e6,
                         color=c, label=f"model VG2={m['vg2']}")
            # Match measured curve closest to this VG2
            best, bd = None, 1e9
            for meas in measured:
                d = abs(meas["VG2"] - m["vg2"])
                if d < bd: bd, best = d, meas
            if best is not None:
                axes[0].plot(best["Vd"], best["Id"] * 1e6,
                             "--", color=c, alpha=0.55,
                             label=f"meas VG2={best['VG2']}")
        axes[0].set_xlabel("V_d (V)"); axes[0].set_ylabel("I_d (uA)")
        axes[0].set_title(f"z331 forced-Vsint snapback (linear), VG1={VG1}")
        axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim([0, VD_MAX])

        # 2) Log plot
        for i, m in enumerate(per_vg2):
            c = colors[i % len(colors)]
            axes[1].semilogy(m["Vd"],
                              np.maximum(np.array(m["Id_model"]), 1e-15),
                              color=c, label=f"model VG2={m['vg2']}")
            best, bd = None, 1e9
            for meas in measured:
                d = abs(meas["VG2"] - m["vg2"])
                if d < bd: bd, best = d, meas
            if best is not None:
                axes[1].semilogy(best["Vd"], np.maximum(best["Id"], 1e-15),
                                  "--", color=c, alpha=0.55,
                                  label=f"meas VG2={best['VG2']}")
        axes[1].set_xlabel("V_d (V)"); axes[1].set_ylabel("|I_d| (A, log)")
        axes[1].set_title("log-scale comparison")
        axes[1].legend(fontsize=8); axes[1].grid(True, which="both", alpha=0.3)
        axes[1].set_xlim([0, VD_MAX]); axes[1].set_ylim([1e-13, 1e-2])

        # 3) Iii and Vb diagnostics
        for i, m in enumerate(per_vg2):
            c = colors[i % len(colors)]
            axes[2].semilogy(m["Vd"], np.maximum(np.array(m["Iii_M1"]), 1e-30),
                              color=c, label=f"Iii VG2={m['vg2']}")
        ax2b = axes[2].twinx()
        for i, m in enumerate(per_vg2):
            c = colors[i % len(colors)]
            ax2b.plot(m["Vd"], m["Vb"], ":", color=c, alpha=0.7)
        axes[2].set_xlabel("V_d (V)"); axes[2].set_ylabel("Iii_M1 (A, log)")
        ax2b.set_ylabel("Vb (V, dotted)")
        axes[2].set_title("diagnostics: Iii (solid), Vb (dotted)")
        axes[2].legend(fontsize=8); axes[2].grid(True, which="both", alpha=0.3)

        fig.tight_layout()
        png_path = OUT / "z331_snapback_graph.png"
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        print(f"[z331] saved {png_path}", flush=True)
        summary["png"] = str(png_path)

        # Per-VG2 individual plots
        for m in per_vg2:
            fig2, ax = plt.subplots(figsize=(7, 5))
            ax.plot(m["Vd"], np.array(m["Id_model"]) * 1e6,
                     "b-", lw=2, label=f"model VG2={m['vg2']}")
            best, bd = None, 1e9
            for meas in measured:
                d = abs(meas["VG2"] - m["vg2"])
                if d < bd: bd, best = d, meas
            if best is not None:
                ax.plot(best["Vd"], best["Id"] * 1e6,
                         "r--", lw=2, label=f"meas VG2={best['VG2']}")
            if m["peak_Vd_model"] is not None:
                ax.axvline(m["peak_Vd_model"], color="b", linestyle=":",
                            alpha=0.5, label=f"model peak Vd={m['peak_Vd_model']:.2f}")
            ax.set_xlabel("V_d (V)"); ax.set_ylabel("I_d (uA)")
            ax.set_title(f"z331 VG1={VG1}, VG2={m['vg2']}")
            ax.legend(); ax.grid(True, alpha=0.3)
            png_indiv = OUT / f"z331_VG2={m['vg2']:.2f}.png"
            fig2.savefig(png_indiv, dpi=120); plt.close(fig2)
            print(f"[z331] saved {png_indiv}", flush=True)

    # ---- Quant: compare peak Vd model vs measured ----
    for m in per_vg2:
        best, bd = None, 1e9
        for meas in measured:
            d = abs(meas["VG2"] - m["vg2"])
            if d < bd: bd, best = d, meas
        meas_peak_Vd, meas_peak_Id = None, None
        meas_peak_at_edge = False
        if best is not None:
            # Sebas only swept Vd to ~2.2V — true snapback knee may lie beyond.
            # Accept right-edge maximum as "measured peak ≥ this Vd".
            mp = find_peak(best["Vd"], best["Id"],
                            vd_min=0.5, vd_max=float(best["Vd"].max()))
            if mp[0] is not None:
                meas_peak_Vd, meas_peak_Id = mp[0], mp[1]
                if mp[2] is None:
                    meas_peak_at_edge = True

        knee_in_range = (m["peak_Vd_model"] is not None and
                          1.5 <= m["peak_Vd_model"] <= 3.0)
        peak_match = (m["peak_Vd_model"] is not None and
                       meas_peak_Vd is not None and
                       abs(m["peak_Vd_model"] - meas_peak_Vd) <= 0.5)
        # log_rmse over overlapping Vd range
        log_rmse = None
        if best is not None:
            try:
                vd_m = np.array(m["Vd"]); id_m = np.array(m["Id_model"])
                # Interpolate measured onto model grid
                lo, hi = max(vd_m.min(), best["Vd"].min()), \
                          min(vd_m.max(), best["Vd"].max())
                mask = (vd_m >= lo) & (vd_m <= hi)
                if mask.sum() > 5:
                    id_meas_interp = np.interp(vd_m[mask], best["Vd"], best["Id"])
                    log_m = np.log10(np.maximum(np.abs(id_m[mask]), 1e-14))
                    log_d = np.log10(np.maximum(np.abs(id_meas_interp), 1e-14))
                    log_rmse = float(np.sqrt(np.mean((log_m - log_d)**2)))
            except Exception:
                pass
        summary["per_vg2"].append({
            "vg2": m["vg2"], "row_vg2": m["row_vg2"],
            "model_peak_Vd": m["peak_Vd_model"],
            "model_peak_Id": m["peak_Id_model"],
            "measured_peak_Vd": meas_peak_Vd,
            "measured_peak_Id": meas_peak_Id,
            "measured_file": best["file"] if best else None,
            "measured_peak_at_sweep_edge": meas_peak_at_edge,
            "infra_knee_in_1p5_3": knee_in_range,
            "pass_peak_within_0p5V": peak_match,
            "log_rmse": log_rmse,
            "ambitious_log_rmse_lt_0p5": (log_rmse is not None and log_rmse < 0.5),
        })

    summary["gate"] = {
        "INFRA_all_knee_in_range": all(p["infra_knee_in_1p5_3"]
                                         for p in summary["per_vg2"]),
        "PASS_all_peak_within_0p5V": all(p["pass_peak_within_0p5V"]
                                           for p in summary["per_vg2"]),
        "AMBITIOUS_all_log_rmse_lt_0p5": all(p["ambitious_log_rmse_lt_0p5"]
                                               for p in summary["per_vg2"]),
    }
    summary["elapsed_s"] = round(time.time() - t0, 2)
    summary_path = OUT / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[z331] saved {summary_path} (elapsed {summary['elapsed_s']}s)",
          flush=True)
    print(f"[z331] GATE: {summary['gate']}", flush=True)
    for p in summary["per_vg2"]:
        print(f"   VG2={p['vg2']}: model peak Vd={p['model_peak_Vd']}, "
              f"meas={p['measured_peak_Vd']}, knee_in_range={p['infra_knee_in_1p5_3']}, "
              f"within_0p5={p['pass_peak_within_0p5V']}, log_rmse={p['log_rmse']}",
              flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
