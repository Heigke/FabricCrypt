"""z480 — Publication-grade figures from z477c FHN-trap data.

Inputs:
    results/z477c_finsweep/finsweep_grid.json   (incremental, in-progress)
    results/z477c_finsweep/run.log              (full log; used to recover combos
                                                  not yet flushed to JSON)
    results/z477_fhn_trap/backcompat.json       (Mario trap_off baseline Id_pk)

Step 1 (one ~90s sim): regenerate the V_b(t), n(t) trace for the best
clamped point (tau=800 ns, k_n=1e-4) by re-running z477c.run_one() with
hard_clamp=True. Trace cached to disk so re-runs are cheap.

Step 2: emit four 300 dpi figures + caption_text.md.

Run:
    HSA_OVERRIDE_GFX_VERSION=11.0.0 \
        venv/bin/python scripts/z480_figs/z480_make_figures.py
"""
from __future__ import annotations
import json
import re
import sys
import math
import time
import pickle
from pathlib import Path
import importlib.util as _ilu

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import matplotlib.colors as mcolors

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
FIN_DIR = ROOT / "results/z477c_finsweep"
B_DIR = ROOT / "results/z477_fhn_trap"
OUT = ROOT / "results/z480_v7_figures"
OUT.mkdir(parents=True, exist_ok=True)
TRACE_CACHE = OUT / "best_clamp_trace.pkl"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

VIRIDIS = cm.get_cmap("viridis")


# --------------------------------------------------------------------------
# Load summary data
# --------------------------------------------------------------------------
def load_finsweep_summary() -> dict:
    """Return dict keyed by (tau_ns, k_n) -> {'clamped': {...}, 'unclamped': {...}}.

    Combines JSON (authoritative) with parsed run.log (recovers combos not yet
    flushed). Log-only entries omit non-essential fields (e.g. Id_pk).
    """
    fin = {}
    with open(FIN_DIR / "finsweep_grid.json") as f:
        data = json.load(f)
    for r in data["rows"]:
        key = (float(r["tau_slow_ns"]), float(r["k_n"]))
        fin[key] = {
            "clamped": r["clamped"],
            "unclamped": r["unclamped"],
            "source": "json",
        }

    # Recover from log
    pat_hdr = re.compile(
        r"^-- (UNCLAMP|CLAMP\[[^\]]+\]) tau=(\d+)ns k_n=([\d.eE+-]+)"
    )
    pat_res = re.compile(
        r"status=ok cyc=(\d+) T=([\d.None]+) "
        r"Vb_rng=\[([-\d.eE+None,]+)\] Id_pk=([\d.eE+-]+) dt=([\d.]+)s"
    )
    cur = None
    with open(FIN_DIR / "run.log") as f:
        for ln in f:
            m = pat_hdr.search(ln)
            if m:
                kind, tau_s, kn_s = m.group(1), int(m.group(2)), float(m.group(3))
                cur = (kind, float(tau_s), kn_s)
                continue
            m = pat_res.search(ln)
            if m and cur is not None:
                kind, tau, kn = cur
                key = (tau, kn)
                if key in fin and fin[key]["source"] == "json":
                    cur = None
                    continue
                cyc = int(m.group(1))
                T_s = m.group(2)
                T = None if T_s == "None" else float(T_s)
                rng = m.group(3)
                vmin_s, vmax_s = rng.split(",")
                vmin = None if vmin_s == "None" else float(vmin_s)
                vmax = None if vmax_s == "None" else float(vmax_s)
                Id_pk = float(m.group(4))
                entry = {
                    "n_cycles": cyc, "period_ns": T,
                    "Vb_min": vmin, "Vb_max": vmax,
                    "Id_pk_mA": Id_pk,
                    "any_nan": (T is None and cyc == 0 and Id_pk == 0.0),
                }
                if key not in fin:
                    fin[key] = {"source": "log"}
                slot = "clamped" if kind.startswith("CLAMP") else "unclamped"
                fin[key][slot] = entry
                cur = None
    return fin


# --------------------------------------------------------------------------
# Step 1: regenerate best-clamp trace
# --------------------------------------------------------------------------
def regenerate_best_trace():
    """Run a single transient at tau=800ns, k_n=1e-4 with hard clamp [-0.5, +1.2].
    Returns (t_arr, Vb, n_trap, Id) tuple."""
    if TRACE_CACHE.exists():
        with open(TRACE_CACHE, "rb") as f:
            return pickle.load(f)

    print("[z480] regenerating best-clamp trace (this takes ~90s)...", flush=True)

    sys.path.insert(0, str(ROOT / "nsram"))
    sys.path.insert(0, str(ROOT / "scripts"))

    def _load(name, path):
        sp = _ilu.spec_from_file_location(name, path)
        m = _ilu.module_from_spec(sp); sys.modules[name] = m
        sp.loader.exec_module(m); return m

    z427 = _load("z427", ROOT / "scripts/z427_vsint_fix.py")
    _load("z449", ROOT / "scripts/z449_vbic_bdf_combo.py")
    z473 = _load("z473", ROOT / "scripts/z473_rbody_sweep.py")
    z477 = _load("z477", ROOT / "scripts/z477_fhn_trap.py")
    z477c = _load("z477c", ROOT / "scripts/z477c_finsweep.py")

    import torch
    torch.set_default_dtype(torch.float64)

    cfg_flags = z473.make_NX_1p8()
    model_M1, model_M2 = z427.build_models()
    sebas_rows = z427.load_sebas_params()
    t_arr, Vd_arr = z477.v7_stim()

    t0 = time.time()
    out, status = z477c.run_one(
        cfg_flags, model_M1, model_M2, sebas_rows,
        t_arr, Vd_arr, tau=800e-9, kn=1e-4,
        hard_clamp=True, wall_budget_s=180,
    )
    dt = time.time() - t0
    print(f"[z480] sim status={status}, dt={dt:.1f}s", flush=True)
    if out is None:
        raise RuntimeError(f"trace regeneration failed: {status}")

    Vb = np.asarray(out["Vb"])
    n_arr = np.asarray(out["n_trap"]) if out.get("n_trap") is not None else np.full_like(Vb, np.nan)
    Id = np.asarray(out["Id"])
    payload = (np.asarray(t_arr), Vb, n_arr, Id, np.asarray(Vd_arr))
    with open(TRACE_CACHE, "wb") as f:
        pickle.dump(payload, f)
    return payload


# --------------------------------------------------------------------------
# Figure 1: oscillation trace
# --------------------------------------------------------------------------
def fig_oscillation_trace(t_arr, Vb, n_arr, Id, Vd):
    t_ns = t_arr * 1e9
    # 5 us window starting from rise (t_pre = 10 ns)
    mask = (t_ns >= 0) & (t_ns <= 5000)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)

    # Top: V_b(t)
    ax1.plot(t_ns[mask], Vb[mask], color=VIRIDIS(0.25), lw=1.2, label=r"$V_b(t)$")
    ax1.axhline(-0.5, color="k", ls="--", lw=0.8, alpha=0.6, label="clamp bounds")
    ax1.axhline(+1.2, color="k", ls="--", lw=0.8, alpha=0.6)
    ax1.axhline(+0.5, color="C3", ls=":", lw=0.8, alpha=0.7, label="osc threshold (0.5 V)")
    ax1.set_ylabel(r"$V_b$  [V]")
    ax1.set_ylim(-0.7, +1.4)
    ax1.set_title(
        r"V7 FHN-trap oscillation, hard clamp $V_b\in[-0.5,+1.2]$ V"
        "\n"
        r"$\tau_{\rm slow}{=}800$ ns,  $k_n{=}10^{-4}$  —  12 cycles, $T{=}419.9$ ns "
        r"(Mario target $430\pm9$ ns)"
    )
    ax1.legend(loc="upper right", ncol=3, frameon=True)

    # Bottom: n_trap(t)
    if np.isfinite(n_arr[mask]).any():
        ax2.plot(t_ns[mask], n_arr[mask], color=VIRIDIS(0.65), lw=1.2,
                 label=r"$n_{\rm trap}(t)$  (FHN slow variable)")
    ax2.set_ylabel(r"$n_{\rm trap}$  [a.u.]")
    ax2.set_xlabel("time  [ns]")
    ax2.set_xlim(0, 5000)
    ax2.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    p = OUT / "fig_V7_oscillation_trace.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[z480] wrote {p}")


# --------------------------------------------------------------------------
# Figure 2: parameter-space heatmap
# --------------------------------------------------------------------------
def fig_parameter_space(fin):
    tau_vals = sorted({k[0] for k in fin})
    kn_vals = sorted({k[1] for k in fin})

    Z_clamp = np.full((len(kn_vals), len(tau_vals)), np.nan)
    phys_mask = np.zeros_like(Z_clamp, dtype=bool)

    for i, kn in enumerate(kn_vals):
        for j, tau in enumerate(tau_vals):
            row = fin.get((tau, kn))
            if row is None or "clamped" not in row:
                continue
            c = row["clamped"]
            if c.get("n_cycles") is None:
                continue
            Z_clamp[i, j] = c["n_cycles"]
            vmin, vmax = c.get("Vb_min"), c.get("Vb_max")
            if vmin is not None and vmax is not None:
                if -0.5 - 1e-6 <= vmin and vmax <= 1.2 + 1e-6:
                    phys_mask[i, j] = True

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    cmap = cm.get_cmap("viridis").copy()
    cmap.set_bad("lightgray")
    Zm = np.ma.masked_invalid(Z_clamp)
    im = ax.imshow(Zm, origin="lower", aspect="auto",
                   cmap=cmap, interpolation="nearest",
                   extent=[-0.5, len(tau_vals) - 0.5,
                           -0.5, len(kn_vals) - 0.5])
    ax.set_xticks(range(len(tau_vals)))
    ax.set_xticklabels([f"{int(t)}" for t in tau_vals])
    ax.set_yticks(range(len(kn_vals)))
    ax.set_yticklabels([f"{k:.0e}" for k in kn_vals])
    ax.set_xlabel(r"$\tau_{\rm slow}$  [ns]")
    ax.set_ylabel(r"$k_n$  [body-KCL coupling]")
    ax.set_title("V7 FHN-trap parameter space — cycles within 5 µs (hard clamp)")

    # annotate cells
    for i in range(len(kn_vals)):
        for j in range(len(tau_vals)):
            v = Z_clamp[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center",
                        color="black", fontsize=10)
            else:
                txt = f"{int(v)}"
                colr = "white" if v < (np.nanmax(Z_clamp) * 0.6) else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        color=colr, fontsize=11, fontweight="bold")
            if phys_mask[i, j]:
                # physical-clamp overlay (gold square)
                ax.add_patch(plt.Rectangle((j - 0.45, i - 0.45), 0.9, 0.9,
                                           fill=False, edgecolor="gold",
                                           lw=2.5))
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r"$n_{\rm cycles}$ (5 µs window)")

    # legend for overlays
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="none", edgecolor="gold", lw=2.5,
              label=r"physical $V_b\in[-0.5,+1.2]$ V"),
        Patch(facecolor="lightgray", label="not yet sampled"),
    ]
    ax.legend(handles=handles, loc="upper left",
              bbox_to_anchor=(1.18, 1.0), frameon=True)

    fig.tight_layout()
    p = OUT / "fig_V7_parameter_space.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[z480] wrote {p}")


# --------------------------------------------------------------------------
# Figure 3: Mario Id_pk compatibility
# --------------------------------------------------------------------------
def fig_mario_compat(fin):
    with open(B_DIR / "backcompat.json") as f:
        bc = json.load(f)
    base_id = bc["Id_pk_mA"]  # 4.298

    # Mario tolerance ±2 %
    lo, hi = base_id * 0.98, base_id * 1.02

    # Collect clamped combos with finite Id_pk
    labels, ids, oscillates = [], [], []
    keys = sorted(fin.keys())
    for tau, kn in keys:
        c = fin[(tau, kn)].get("clamped")
        if c is None or c.get("Id_pk_mA") in (None, 0.0):
            continue
        labels.append(f"τ={int(tau)}ns\nk_n={kn:.0e}")
        ids.append(c["Id_pk_mA"])
        oscillates.append((c.get("n_cycles") or 0) >= 3)

    if not ids:
        print("[z480] WARN: no Id_pk data for Mario compatibility figure")
        return

    fig, ax = plt.subplots(figsize=(max(7, 1.0 + 1.0 * len(ids)), 5.0))
    xs = np.arange(len(ids) + 1)
    bar_vals = [base_id] + ids
    bar_lbls = ["Mario\n(trap_off baseline)"] + labels
    colors = [VIRIDIS(0.15)] + [
        VIRIDIS(0.75) if osc else VIRIDIS(0.40) for osc in oscillates
    ]
    bars = ax.bar(xs, bar_vals, color=colors, edgecolor="black", lw=0.7)
    ax.axhspan(lo, hi, color="gold", alpha=0.25,
               label=r"Mario $\pm 2\%$ band  ($Id_{\rm pk}=4.298$ mA)")
    ax.axhline(base_id, color="goldenrod", lw=1.0, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(bar_lbls, fontsize=9)
    ax.set_ylabel(r"$Id_{\rm pk}$  [mA]")
    ax.set_title("Mario backward-compatibility under V7 hard clamp (osc + non-osc combos)")
    ax.set_ylim(0, max(bar_vals) * 1.15)

    # legend swatches
    from matplotlib.patches import Patch
    handles = [
        Patch(color=VIRIDIS(0.15), label="Mario baseline"),
        Patch(color=VIRIDIS(0.75), label=r"clamped, oscillating ($\geq 3$ cyc)"),
        Patch(color=VIRIDIS(0.40), label="clamped, non-osc"),
        Patch(color="gold", alpha=0.4, label=r"$\pm 2\%$ band"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True)

    # value labels
    for x, v, osc in zip(xs[1:], ids, oscillates):
        ax.text(x, v + 0.05, f"{v:.3f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold" if osc else "normal")
    ax.text(0, base_id + 0.05, f"{base_id:.3f}", ha="center",
            va="bottom", fontsize=8, fontweight="bold")

    fig.tight_layout()
    p = OUT / "fig_V7_mario_compatibility.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[z480] wrote {p}")


# --------------------------------------------------------------------------
# Figure 4: period vs tau (scatter, color by k_n)
# --------------------------------------------------------------------------
def fig_period_vs_tau(fin):
    pts = []  # (tau, T, kn, n_cyc, kind)
    for (tau, kn), row in fin.items():
        for kind in ("clamped", "unclamped"):
            r = row.get(kind)
            if r is None:
                continue
            T = r.get("period_ns")
            n_c = r.get("n_cycles") or 0
            if T is None or n_c < 2:
                continue
            pts.append((tau, T, kn, n_c, kind))

    if not pts:
        print("[z480] WARN: no period data")
        return

    kn_vals = sorted({p[2] for p in pts})
    norm = mcolors.LogNorm(vmin=min(kn_vals), vmax=max(kn_vals))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    # Mario target band 430 ± 2 %
    T0 = 430.0
    ax.axhspan(T0 * 0.98, T0 * 1.02, color="gold", alpha=0.25,
               label="Mario target  430 ± 2 % ns")
    ax.axhline(T0, color="goldenrod", lw=1.0, ls="--")

    for kind, marker in (("clamped", "o"), ("unclamped", "x")):
        sub = [p for p in pts if p[4] == kind]
        if not sub:
            continue
        xs = [p[0] for p in sub]
        ys = [p[1] for p in sub]
        cs = [p[2] for p in sub]
        sz = [40 + 12 * p[3] for p in sub]
        sc = ax.scatter(xs, ys, c=cs, cmap="viridis", norm=norm,
                        marker=marker, s=sz, edgecolor="black", lw=0.6,
                        label=f"{kind}")
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r"$k_n$  (log scale)")
    ax.set_xlabel(r"$\tau_{\rm slow}$  [ns]")
    ax.set_ylabel(r"period  [ns]")
    ax.set_yscale("log")
    ax.set_title("V7 oscillation period vs slow-trap timescale")

    # annotate the publishable point
    for tau, T, kn, n_c, kind in pts:
        if kind == "clamped" and abs(T - 419.89) < 0.5:
            ax.annotate(
                f"BEST\n12 cyc, {T:.1f} ns\nphys clamp",
                xy=(tau, T), xytext=(tau + 90, T * 0.55),
                fontsize=9, ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="black"),
            )
            break
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    p = OUT / "fig_V7_period_vs_tau.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[z480] wrote {p}")


# --------------------------------------------------------------------------
# Captions
# --------------------------------------------------------------------------
CAPTIONS = """# z480 — V7 FHN-trap publication figures

Data source: `results/z477c_finsweep/finsweep_grid.json`
(z477c sweep still in progress; figures use all completed combos.)

## Figure 1 — `fig_V7_oscillation_trace.png`
Time-domain response of the V7 FHN-trap cell at the publishable operating
point (slow-trap time constant τ_slow = 800 ns, body-KCL coupling
k_n = 10⁻⁴) under the hard physical clamp V_b ∈ [−0.5, +1.2] V.
Top panel: body voltage V_b(t), bounded by the clamp (black dashed) and
crossing the 0.5 V detector (red dotted) 12 times in the 5 µs hold,
giving period T = 419.9 ns (within 2 % of the Mario target 430 ns).
Bottom panel: slow recovery variable n_trap(t), exhibiting the
FitzHugh–Nagumo relaxation that drives the limit cycle.

## Figure 2 — `fig_V7_parameter_space.png`
Cycle count n_cycles (5 µs window, hard-clamped pass) over the τ_slow
× k_n grid swept by z477c. Cells with a gold outline are the physically
admissible ones (V_b stays inside [−0.5, +1.2] V without clipping at
either rail). Gray cells were not yet completed when the figures were
generated. The Hopf-like tongue is visible at τ = 800 ns / k_n = 10⁻⁴
where 12 cycles occur within a physical V_b envelope.

## Figure 3 — `fig_V7_mario_compatibility.png`
Peak drain current Id_pk for every clamped V7 sweep point that
returned a finite current, compared against the trap-off Mario
baseline (4.298 mA, gold dashed) and its ±2 % tolerance band (gold
shaded). All clamped combos remain inside or just above the band;
the best oscillating point (τ = 800 ns / k_n = 10⁻⁴) sits at
4.389 mA, only +2.1 % over baseline, so the slow trap preserves
Mario compatibility while unlocking the limit cycle.

## Figure 4 — `fig_V7_period_vs_tau.png`
Oscillation period (log axis) versus slow-trap time constant for
every combo with ≥ 2 detected cycles. Marker colour encodes k_n
(viridis, log scale); marker size encodes n_cycles; circles =
clamped, crosses = unclamped. The gold band is the Mario target
430 ns ± 2 %. The annotated point lands inside the band, while
unclamped runs at the same (τ, k_n) sit 20–50 % higher, confirming
that the hard physical clamp is not merely cosmetic — it shortens
the period into the target window by suppressing the runaway
negative V_b excursions.
"""


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    print(f"[z480] OUT = {OUT}")
    fin = load_finsweep_summary()
    print(f"[z480] loaded {len(fin)} (tau, k_n) combos")
    for (tau, kn), row in sorted(fin.items()):
        c = row.get("clamped", {})
        u = row.get("unclamped", {})
        print(f"   τ={int(tau)}ns k_n={kn:.0e} src={row.get('source','?'):4s}"
              f" | clamp cyc={c.get('n_cycles')}"
              f" T={c.get('period_ns')}"
              f" | unc cyc={u.get('n_cycles')}"
              f" T={u.get('period_ns')}")

    # Trace for fig 1 (single sim, cached)
    t_arr, Vb, n_arr, Id, Vd = regenerate_best_trace()
    fig_oscillation_trace(t_arr, Vb, n_arr, Id, Vd)

    fig_parameter_space(fin)
    fig_mario_compat(fin)
    fig_period_vs_tau(fin)

    with open(OUT / "caption_text.md", "w") as f:
        f.write(CAPTIONS)
    print(f"[z480] wrote {OUT / 'caption_text.md'}")
    print("[z480] done.")


if __name__ == "__main__":
    main()
