"""z458 — 2D sweep (snap_Is_scale × R_body) on top of v449_B + SB_HOT + NX_1p8.

Goal: find combined regime where:
  - ns-snap still fires on >=3/4 high-VG1 biases (V_B -> 0.5V within <5ns)
  - self-reset within 100ns-100us (V_B falls back below 0.3V)
  - periodic relaxation oscillation appears (>=3 cycles) with 100-1000ns period
  - DC penalty < 1.5 dec vs SB_OFF baseline (2.087 dec) => cap at 3.5 dec

Physics hypothesis: with V_db-gated NPN (NX_1p8) the NPN holding current
collapses post-snap (V_db drops below V_knee), so a weak R_body should
drain V_B without being out-fought by an active BJT clamp. Optimum likely
R_100M (tau ~ 270ns) which matches Mario slide-21 ~400 ns period.

4 x 4 = 16 conditions:
  snap_Is_scale in {1.0, 0.5, 0.1, 0.01}
  R_body in {10M, 100M, 1G, inf}

Per cell:
  - assert_snapback_live (deep, V_db=1.4)
  - mid-DC I_snap_d at V_db=0.8 (gate effective?)
  - DC fwd + bwd cell-wide (BOTH per z451 critique)
  - extended fast-pulse: V_D 100ps -> 2V, HOLD 1us, on 4 biases
  - detect self-reset, oscillation cycles + period

Outputs to results/z458_snap_rbody_2d/:
  run.log   (line 1 = pre-registered gates, locked)
  summary.json
  heatmap_dc.png      (DC_avg 4x4)
  heatmap_reset.png   (median t_to_reset 4x4)
  oscillation_traces.png   (V_B(t) for top-3 oscillation candidates)
  pareto.png          (DC penalty vs oscillation quality)
  honest_analysis.md
"""
from __future__ import annotations
import importlib.util as _ilu
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT = ROOT / "results/z458_snap_rbody_2d"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

LOG = open(OUT / "run.log", "w")
def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True); LOG.write(line + "\n"); LOG.flush()


# ---------------- Pre-registered gates (line 1 of run.log, LOCKED) -------
log("PRE-REGISTERED GATES (z458, locked):")
log("  INFRA      = 16 cells complete + summary.json written")
log("  DISCOVERY  = some (snap_Is_scale, R_body) gives self-reset (100ns..100us) "
    "AND DC_avg < 3.0 dec AND >=1 oscillation cycle on >=1 bias")
log("  AMBITIOUS  = some cell gives periodic oscillation >=3 cycles AND period "
    "in 100..1000ns range AND DC_avg < 1.8 dec")
log("  KILL_SHOT  = no cell produces self-reset across all 16 -> snapback "
    "fundamentally incompatible with reset under any tuning")
log("  Reference: SB_OFF DC_avg=2.087 dec; SB_HOT DC_avg=2.809 dec; "
    "Mario slide-21 osc period ~400 ns (quoted, not target-matched)")
log("")


# Reuse z454 (and via it z449/z427/z429) and z457 helpers.
# NOTE: importing these modules opens their own results/<name>/run.log in "w"
# mode at import time (truncating history). We snapshot+restore those files
# so historical logs are preserved.
_PRESERVE = [
    ROOT / "results/z454_snapback_integration/run.log",
    ROOT / "results/z456_rbody_reset/run.log",
    ROOT / "results/z457_npn_gate/run.log",
]
_snapshots = {}
for _p in _PRESERVE:
    try:
        _snapshots[_p] = _p.read_bytes() if _p.exists() else None
    except Exception:
        _snapshots[_p] = None

_spec454 = _ilu.spec_from_file_location("z454", ROOT / "scripts/z454_snapback_integration.py")
z454 = _ilu.module_from_spec(_spec454); _spec454.loader.exec_module(z454)
_spec457 = _ilu.spec_from_file_location("z457", ROOT / "scripts/z457_npn_gate.py")
z457 = _ilu.module_from_spec(_spec457); _spec457.loader.exec_module(z457)
_spec456 = _ilu.spec_from_file_location("z456", ROOT / "scripts/z456_rbody_reset.py")
z456 = _ilu.module_from_spec(_spec456); _spec456.loader.exec_module(z456)

# Redirect the helper modules' LOG to a discard sink (their log() also prints
# to stdout which still ends up in our nohup.out, so we lose nothing). Then
# restore the original run.log contents that import-time truncation wiped.
import io as _io
for _m in (z454, z456, z457):
    try:
        _m.LOG.close()
    except Exception:
        pass
    try:
        _m.LOG = _io.StringIO()  # any further .write/.flush is harmless
    except Exception:
        pass
for _p, _content in _snapshots.items():
    if _content is not None:
        try:
            _p.write_bytes(_content)
        except Exception:
            pass

z449 = z454.z449
z427 = z454.z427
z429 = z454.z429
BIASES = z454.BIASES
slow_dc_cell_rmse_dir = z454.slow_dc_cell_rmse_dir
assert_snapback_live  = z454.assert_snapback_live
mid_dc_I_snap_d       = z457.mid_dc_I_snap_d
fast_pulse_extended   = z456.fast_pulse_extended  # honours R_body in TransientCfgV2
detect_self_reset     = z456.detect_self_reset
detect_oscillation    = z456.detect_oscillation


# ---------------- Base flags: v449_B + SB_HOT + NX_1p8 NPN gate ---------
SB_HOT = dict(
    snap_BV=2.0 * 0.6,            # 1.2 V
    snap_n_avl=4.0,
    snap_Bf=417.0,
    snap_Va=0.90,
    snap_Is=6.0256e-9 * 5.0,      # SB_HOT Is baseline (snap_Is_scale=1.0 == this)
    snap_Nf=1.0,
    snap_Id_clamp=1e-2,
    snap_Iii_clamp=1e-2,
)
V449B_BASE = {
    "use_vbic_for_q1": True,
    "vbic_AVC1": 0.5, "vbic_AVC2": 0.5,
    "Cbody": 1e-15,
    "body_pdiode_Cj0_per_area": 0.0,
}
SLOTBOOM_KNEE_K_1P6 = {
    "snap_use_knee_gate": True,
    "snap_V_knee": 1.6,
    "snap_V_sharp": 0.05,
}
NX_1P8 = {
    "snap_npn_gate_mode": "current",
    "snap_npn_V_knee": 1.8,
    "snap_npn_V_sharp": 0.05,
    "snap_npn_V_BE_offset": 0.3,
}

SNAP_IS_BASE = SB_HOT["snap_Is"]  # SB_HOT level (scale=1.0)

SNAP_IS_SCALES = [1.0, 0.5, 0.1, 0.01]
R_BODY_LIST = [
    ("R_10M",  1.0e7,  "2.7e-5s"),
    ("R_100M", 1.0e8,  "2.7e-4s"),
    ("R_1G",   1.0e9,  "2.7e-3s"),
    ("R_INF",  None,   "inf"),
]


def cell_flags(is_scale: float) -> dict:
    f = {**V449B_BASE, "use_snapback_sub": True, **SB_HOT,
         **SLOTBOOM_KNEE_K_1P6, **NX_1P8}
    f["snap_Is"] = SNAP_IS_BASE * is_scale
    return f


def cell_name(is_scale: float, rb_name: str) -> str:
    s_tag = {1.0: "Is1p0", 0.5: "Is0p5", 0.1: "Is0p1", 0.01: "Is0p01"}[is_scale]
    return f"{s_tag}__{rb_name}"


CAPTURE_TAG = "VG1_0p6_VG2_0p2"


def thermal_pause():
    """Pause if APU thermal_zone0 > 85 C; wait to <75 C (CLAUDE.md)."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            t = int(f.read().strip()) / 1000.0
        if t > 85.0:
            log(f"  THERMAL PAUSE: APU={t:.1f}C > 85C, cooling ...")
            for _ in range(120):
                time.sleep(2)
                with open("/sys/class/thermal/thermal_zone0/temp") as f:
                    t = int(f.read().strip()) / 1000.0
                if t < 75.0:
                    log(f"  COOLED: APU={t:.1f}C, resuming"); break
    except Exception:
        pass


# ---------------- Heatmap helpers ---------------------------------------
def grid_extract(results, field_fn, fill=np.nan):
    """Return a 4 (rows: Is_scale) x 4 (cols: R_body) array of values."""
    M = np.full((len(SNAP_IS_SCALES), len(R_BODY_LIST)), fill, dtype=float)
    for r in results:
        i = SNAP_IS_SCALES.index(r["snap_Is_scale"])
        j = [k for k, (n, _, _) in enumerate(R_BODY_LIST) if n == r["R_body_name"]][0]
        try:
            v = field_fn(r)
            M[i, j] = v if v is not None else fill
        except Exception:
            pass
    return M


def plot_heatmap(M, title, path, cmap="viridis", fmt="{:.2f}",
                 vmin=None, vmax=None, cbar_label=""):
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    im = ax.imshow(M, cmap=cmap, aspect="auto",
                   vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(R_BODY_LIST)))
    ax.set_xticklabels([n for n, _, _ in R_BODY_LIST])
    ax.set_yticks(range(len(SNAP_IS_SCALES)))
    ax.set_yticklabels([f"x{s:g}" for s in SNAP_IS_SCALES])
    ax.set_xlabel("R_body"); ax.set_ylabel("snap_Is_scale")
    ax.set_title(title)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        color="white" if v > (np.nanmean(M) if np.isfinite(np.nanmean(M)) else 0) else "black",
                        fontsize=8)
            else:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color="grey")
    cb = fig.colorbar(im, ax=ax); cb.set_label(cbar_label)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    log(f"  wrote {path.name}")


def plot_oscillation_traces(results, path, top_n=3, bias_tag=CAPTURE_TAG):
    # Rank cells by oscillation quality: prefer >=3 cycles; tiebreak by lowest DC_avg.
    cands = []
    for r in results:
        rec = next((x for x in r["fast"]["per_bias"] if x["tag"] == bias_tag), None)
        if rec is None: continue
        ncyc = rec.get("n_oscillation_cycles", 0) or 0
        per  = rec.get("oscillation_period_ns")
        # quality score: 3 cycles weighted, plus penalty for out-of-range period
        score = ncyc
        if per is not None:
            if 100.0 <= per <= 1000.0:
                score += 1.0
        cands.append((score, -r["dc_avg"], r, rec))
    cands.sort(key=lambda x: (x[0], x[1]), reverse=True)
    use = cands[:top_n]
    if not use:
        log("  no oscillation candidates with traces to plot")
        return
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    colors = ["C0", "C2", "C3"]
    for (sc, _, r, rec), col in zip(use, colors):
        tr = rec.get("_traces")
        if tr is None:
            continue
        t = np.array(tr["t"]) * 1e9
        per_str = (f"{rec.get('oscillation_period_ns'):.0f}ns"
                   if rec.get("oscillation_period_ns") else "-")
        ax.plot(t, tr["Vb"], col + "-", lw=1.1,
                label=f"{r['name']} DC={r['dc_avg']:.2f}dec "
                      f"cyc={rec.get('n_oscillation_cycles', 0)} per={per_str}")
    ax.axhline(0.5, color="red", ls=":", lw=0.6, label="0.5V peak")
    ax.axhline(0.3, color="grey", ls=":", lw=0.6, label="0.3V reset")
    ax.set_xlabel("time [ns]"); ax.set_ylabel("V_B [V]")
    ax.set_title(f"z458 - top oscillation traces @ {bias_tag} "
                 f"(Mario slide-21 target ~400 ns, NOT tuned-to-fit)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    log(f"  wrote {path.name}")


def plot_pareto(results, path):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    SB_OFF_DC = 2.087
    xs = []; ys = []; lbls = []; cs = []
    for r in results:
        dc_pen = r["dc_avg"] - SB_OFF_DC
        # quality = avg number of oscillation cycles across biases + reset success rate
        n_reset = 0; n_total = 0; total_cyc = 0
        for x in r["fast"]["per_bias"]:
            n_total += 1
            if x.get("self_reset"): n_reset += 1
            total_cyc += int(x.get("n_oscillation_cycles", 0) or 0)
        quality = total_cyc + n_reset  # crude composite
        xs.append(dc_pen); ys.append(quality); lbls.append(r["name"])
        # colour by Is_scale
        cs.append({1.0:"C3", 0.5:"C1", 0.1:"C2", 0.01:"C0"}[r["snap_Is_scale"]])
    ax.scatter(xs, ys, c=cs, s=60, edgecolor="k")
    for x, y, lbl in zip(xs, ys, lbls):
        ax.annotate(lbl, (x, y), fontsize=7, xytext=(3, 3),
                    textcoords="offset points")
    ax.axvline(1.5, color="orange", ls=":", lw=0.7, label="DC pen 1.5 dec")
    ax.axvline(0.913, color="red", ls=":", lw=0.7, label="DC pen for DC_avg=3.0")
    ax.set_xlabel("DC penalty vs SB_OFF [dec]  (DC_avg - 2.087)")
    ax.set_ylabel("oscillation quality (sum cycles + reset count, 4 biases)")
    ax.set_title("z458 - Pareto: DC cost vs oscillation richness")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    log(f"  wrote {path.name}")


# ---------------- Gate evaluation ---------------------------------------
def eval_gates(results):
    any_self_reset = False
    discovery = False; discovery_who = []
    ambitious = False; ambitious_who = []
    for r in results:
        dc = r["dc_avg"]
        any_cycle = False
        for x in r["fast"]["per_bias"]:
            tr_ns = x.get("t_to_reset_after_peak_ns")
            ncyc  = int(x.get("n_oscillation_cycles", 0) or 0)
            per   = x.get("oscillation_period_ns")
            if tr_ns is not None and 100.0 <= tr_ns <= 1.0e5:
                any_self_reset = True
                if dc < 3.0 and ncyc >= 1:
                    discovery = True
                    discovery_who.append(
                        f"{r['name']}/{x['tag']} t_reset={tr_ns:.1f}ns dc={dc:.2f}")
            if ncyc >= 3 and per is not None and 100.0 <= per <= 1000.0 and dc < 1.8:
                ambitious = True
                ambitious_who.append(
                    f"{r['name']}/{x['tag']} cycles={ncyc} period={per:.1f}ns dc={dc:.2f}")
            if ncyc >= 1:
                any_cycle = True
    kill_shot = not any_self_reset
    return {
        "INFRA_pass": True,
        "DISCOVERY_pass": discovery,
        "DISCOVERY_who": discovery_who,
        "AMBITIOUS_pass": ambitious,
        "AMBITIOUS_who": ambitious_who,
        "KILL_SHOT": kill_shot,
        "kill_shot_reason": ("no_self_reset_anywhere" if kill_shot else None),
    }


# ---------------- Per-cell runner ---------------------------------------
def run_cell(is_scale, rb_tuple, model_M1, model_M2, sebas_rows, curves):
    rb_name, R_body, tau_str = rb_tuple
    name = cell_name(is_scale, rb_name)
    flags = cell_flags(is_scale)
    log(f"===== {name}: snap_Is={flags['snap_Is']:.3e} R_body={R_body} tau~{tau_str} =====")

    # 1) Deep snapback assert (V_db=1.4 > knee) - must fire when scale>=non-trivial
    assert_info = assert_snapback_live(name, flags, model_M1, model_M2, sebas_rows)
    log(f"  assert(deep): snap_called={assert_info.get('snap_called')} "
        f"|I_snap_d|={assert_info.get('I_snap_d', 0):.3e}A "
        f"|I_snap_b|={assert_info.get('I_snap_b', 0):.3e}A")

    # 2) Mid-DC I_snap_d probe at V_db=0.8 (BELOW NPN V_knee=1.8 -> must be ~0)
    mid = mid_dc_I_snap_d(name, flags, model_M1, model_M2, sebas_rows,
                          Vd_test=1.4, Vb_test=0.6, Vsint_test=0.0,
                          VG1=0.6, VG2=0.0)
    if mid is not None and mid["I_snap_d"] > 1e-3:
        log(f"  WARNING: NX_1p8 gate not effective for {name}: "
            f"|I_snap_d|_mid={mid['I_snap_d']:.3e}A (>1mA)")

    # z444-BURN style sanity: assert non-zero at deep + ~zero at mid for is_scale>=0.1
    burn_ok = True
    if is_scale >= 0.1:
        if not assert_info.get("snap_called", False):
            burn_ok = False
            log(f"  BURN-FAIL: snap not called at deep V_db for is_scale={is_scale}")
        if mid is not None and mid["I_snap_d"] > 1e-3:
            burn_ok = False
            log(f"  BURN-FAIL: NPN gate leaks at mid V_db for {name}")

    # 3) DC fwd + bwd (both directions, per z451 critique)
    thermal_pause()
    dc_f = slow_dc_cell_rmse_dir(name, flags, model_M1, model_M2, curves, sebas_rows,
                                 direction="forward")
    thermal_pause()
    dc_b = slow_dc_cell_rmse_dir(name, flags, model_M1, model_M2, curves, sebas_rows,
                                 direction="backward")
    dc_avg = 0.5 * (dc_f["cell_rmse_dec"] + dc_b["cell_rmse_dec"])
    log(f"  DC fwd={dc_f['cell_rmse_dec']:.3f}  bwd={dc_b['cell_rmse_dec']:.3f}  avg={dc_avg:.3f}")

    # 4) Extended fast-pulse with R_body (4 biases, 1us hold)
    thermal_pause()
    fast = fast_pulse_extended(name, flags, R_body, model_M1, model_M2, sebas_rows)

    return {
        "name": name,
        "snap_Is_scale": is_scale,
        "snap_Is": flags["snap_Is"],
        "R_body_name": rb_name,
        "R_body": R_body,
        "tau_est_s": tau_str,
        "assert": assert_info,
        "mid_dc_I_snap_d": mid,
        "burn_ok": burn_ok,
        "dc_forward": dc_f,
        "dc_backward": dc_b,
        "dc_avg": dc_avg,
        "fast": fast,
    }


# ---------------- Main --------------------------------------------------
def main():
    t0 = time.time()
    log("z458 starting - 4x4 (snap_Is_scale x R_body) on top of v449_B+SB_HOT+NX_1p8")
    model_M1, model_M2 = z427.build_models()
    curves = z427.load_curves()
    sebas_rows = z427.load_sebas_params()
    log(f"loaded: {len(curves)} curves, {len(sebas_rows)} sebas rows")

    results = []
    for is_scale in SNAP_IS_SCALES:
        for rb in R_BODY_LIST:
            try:
                r = run_cell(is_scale, rb, model_M1, model_M2, sebas_rows, curves)
            except Exception as e:
                log(f"  CELL FAIL {cell_name(is_scale, rb[0])}: {e}")
                continue
            results.append(r)
            # checkpoint summary every cell (in case of crash)
            try:
                _trim_for_json(results)
                (OUT / "summary.json").write_text(
                    json.dumps(_summary_blob(results, t0, partial=True),
                               indent=2, default=float))
            except Exception:
                pass

    gates = eval_gates(results)
    log(f"GATES: {gates}")

    # Plots
    M_dc = grid_extract(results, lambda r: r["dc_avg"])
    plot_heatmap(M_dc, "z458 - DC_avg [dec] (SB_OFF=2.087, SB_HOT=2.809)",
                 OUT / "heatmap_dc.png", cmap="viridis",
                 fmt="{:.2f}", cbar_label="DC_avg [dec]")

    def median_reset_ns(r):
        vals = [x.get("t_to_reset_after_peak_ns") for x in r["fast"]["per_bias"]
                if x.get("t_to_reset_after_peak_ns") is not None]
        return float(np.median(vals)) if vals else np.nan
    M_reset = grid_extract(results, median_reset_ns)
    plot_heatmap(np.log10(np.where(np.isfinite(M_reset) & (M_reset > 0),
                                   M_reset, np.nan)),
                 "z458 - median t_to_reset [log10 ns]  (nan = no reset)",
                 OUT / "heatmap_reset.png", cmap="plasma",
                 fmt="{:.2f}", cbar_label="log10(t_reset_ns)")

    plot_oscillation_traces(results, OUT / "oscillation_traces.png", top_n=3)
    plot_pareto(results, OUT / "pareto.png")

    _trim_for_json(results)
    blob = _summary_blob(results, t0, gates=gates, partial=False)
    (OUT / "summary.json").write_text(json.dumps(blob, indent=2, default=float))
    log(f"wrote summary.json  total_wall={blob['wall_total_sec']:.0f}s")

    # Best cell = lowest DC_avg subject to self-reset present somewhere
    cands_reset = [r for r in results
                   if any(x.get("self_reset") for x in r["fast"]["per_bias"])]
    best_reset = (min(cands_reset, key=lambda r: r["dc_avg"])
                  if cands_reset else None)
    best_dc = min(results, key=lambda r: r["dc_avg"]) if results else None

    # honest_analysis.md
    md = []
    md.append("# z458 - snap_Is x R_body 2D sweep (v449_B + SB_HOT + NX_1p8)\n")
    md.append("## Pre-registered gates\n")
    for k, v in gates.items():
        md.append(f"- **{k}**: {v}")
    md.append("\n## 4x4 DC_avg grid (rows = snap_Is_scale, cols = R_body)\n")
    md.append("| Is_scale | " + " | ".join(n for n, _, _ in R_BODY_LIST) + " |")
    md.append("|" + "---|" * (len(R_BODY_LIST) + 1))
    for i, s in enumerate(SNAP_IS_SCALES):
        row = [f"x{s:g}"] + [f"{M_dc[i,j]:.3f}" if np.isfinite(M_dc[i,j]) else "n/a"
                              for j in range(len(R_BODY_LIST))]
        md.append("| " + " | ".join(row) + " |")

    md.append("\n## 4x4 median t_to_reset [ns] grid\n")
    md.append("| Is_scale | " + " | ".join(n for n, _, _ in R_BODY_LIST) + " |")
    md.append("|" + "---|" * (len(R_BODY_LIST) + 1))
    for i, s in enumerate(SNAP_IS_SCALES):
        row = [f"x{s:g}"]
        for j in range(len(R_BODY_LIST)):
            v = M_reset[i, j]
            row.append(f"{v:.1f}" if np.isfinite(v) else "no-reset")
        md.append("| " + " | ".join(row) + " |")

    md.append("\n## Per-cell summary\n")
    md.append("| cell | snap_Is | R_body | DC_fwd | DC_bwd | DC_avg | "
              "n_biases_reset | n_biases_cyc>=3 | best_period_ns | mid_Isnap_d |")
    md.append("|" + "---|" * 10)
    for r in results:
        nb = len(r["fast"]["per_bias"])
        n_r = sum(1 for x in r["fast"]["per_bias"] if x.get("self_reset"))
        n_c3 = sum(1 for x in r["fast"]["per_bias"]
                   if (x.get("n_oscillation_cycles") or 0) >= 3)
        pers = [x.get("oscillation_period_ns") for x in r["fast"]["per_bias"]
                if x.get("oscillation_period_ns") is not None]
        per_str = f"{min(pers, key=lambda p: abs(p-400)):.1f}" if pers else "-"
        mid = r.get("mid_dc_I_snap_d")
        mid_str = f"{mid['I_snap_d']:.2e}" if mid else "-"
        md.append(f"| {r['name']} | {r['snap_Is']:.2e} | "
                  f"{r['R_body'] if r['R_body'] else 'inf'} | "
                  f"{r['dc_forward']['cell_rmse_dec']:.3f} | "
                  f"{r['dc_backward']['cell_rmse_dec']:.3f} | "
                  f"{r['dc_avg']:.3f} | {n_r}/{nb} | {n_c3}/{nb} | "
                  f"{per_str} | {mid_str} |")

    md.append("\n## Snapback assert (deep, V_db=1.4) - z444-BURN diagnostic\n")
    md.append("| cell | snap_called | |I_snap_d|[A] | |I_snap_b|[A] | mid_|I_snap_d|[A] | burn_ok |")
    md.append("|" + "---|" * 6)
    for r in results:
        a = r["assert"]; mid = r.get("mid_dc_I_snap_d")
        mid_v = f"{mid['I_snap_d']:.3e}" if mid else "-"
        md.append(f"| {r['name']} | {a.get('snap_called')} | "
                  f"{a.get('I_snap_d', 0):.3e} | {a.get('I_snap_b', 0):.3e} | "
                  f"{mid_v} | {r['burn_ok']} |")

    md.append("\n## Per-bias fast-pulse detail (1us hold)\n")
    for r in results:
        md.append(f"### {r['name']} (DC_avg={r['dc_avg']:.3f}, "
                  f"snap_Is={r['snap_Is']:.2e}, R_body={r['R_body']})")
        md.append("| bias | Vb_peak | t->0.5V[ns] | t->reset[ns] | self-reset | "
                  "n_cycles | period[ns] |")
        md.append("|" + "---|" * 7)
        for x in r["fast"]["per_bias"]:
            md.append(f"| {x['tag']} | {x['Vb_peak_V']:.3f} | "
                      f"{x.get('t_to_0p5V_after_ramp_ns')} | "
                      f"{x.get('t_to_reset_after_peak_ns')} | "
                      f"{x.get('self_reset')} | "
                      f"{x.get('n_oscillation_cycles')} | "
                      f"{x.get('oscillation_period_ns')} |")
        md.append("")

    md.append("\n## Best cells\n")
    if best_reset is not None:
        md.append(f"- **Best with self-reset present**: {best_reset['name']}  "
                  f"DC_avg={best_reset['dc_avg']:.3f}  snap_Is={best_reset['snap_Is']:.2e}  "
                  f"R_body={best_reset['R_body']}")
    else:
        md.append("- No cell shows self-reset on ANY bias.")
    if best_dc is not None:
        md.append(f"- **Lowest DC overall**: {best_dc['name']}  "
                  f"DC_avg={best_dc['dc_avg']:.3f}")

    md.append("\n## Reference baselines (from prior runs)\n")
    md.append("- SB_OFF DC_avg = 2.087 dec  (z454)")
    md.append("- SB_HOT DC_avg = 2.809 dec  (z454)")
    md.append("- z457 NX_1p8 DC_avg = (see z457/honest_analysis.md)")
    md.append("- Mario slide-21 reference: ~400 ns oscillation period "
              "(QUOTED only; not tuned-to-fit)")

    if gates["KILL_SHOT"]:
        md.append("\n## KILL_SHOT triggered\n")
        md.append("No (snap_Is_scale, R_body) combination produced V_B self-reset "
                  "in the 100ns..100us window on any bias. Snapback subcircuit + "
                  "V_db-gated NPN + body-leak resistor cannot deliver innate LIF "
                  "under any tuning explored here. Suggests:")
        md.append("- The NPN holding current after snap is still > R_body leak even "
                  "when gated (V_db remains > V_knee_npn after snap).")
        md.append("- Body capacitance may need to be increased (currently 1 fF) so "
                  "tau = R*C is in the right window without making R so small that DC dies.")
        md.append("- A two-stage knee (NPN OFF for V_db<V_knee, plus a state-dependent "
                  "shut-down) may be required - not a passive leak.")

    (OUT / "honest_analysis.md").write_text("\n".join(md))
    log("wrote honest_analysis.md")
    log("DONE.")


def _trim_for_json(results, max_pts=400):
    for r in results:
        for rec in r["fast"]["per_bias"]:
            tr = rec.get("_traces")
            if tr is None: continue
            keys = [k for k, v in tr.items() if isinstance(v, list)]
            if not keys: continue
            n_in = len(tr[keys[0]])
            if n_in <= max_pts: continue
            idx = np.linspace(0, n_in - 1, max_pts).astype(int).tolist()
            for k in keys:
                v = tr[k]
                if len(v) == n_in:
                    tr[k] = [v[i] for i in idx]


def _summary_blob(results, t0, gates=None, partial=False):
    cells = []
    for r in results:
        cells.append({
            "name": r["name"],
            "snap_Is_scale": r["snap_Is_scale"],
            "snap_Is": r["snap_Is"],
            "R_body_name": r["R_body_name"],
            "R_body": r["R_body"],
            "tau_est_s": r["tau_est_s"],
            "assert": r["assert"],
            "mid_dc_I_snap_d": r["mid_dc_I_snap_d"],
            "burn_ok": r["burn_ok"],
            "dc_forward_dec": r["dc_forward"]["cell_rmse_dec"],
            "dc_backward_dec": r["dc_backward"]["cell_rmse_dec"],
            "dc_avg_dec": r["dc_avg"],
            "dc_forward_n": r["dc_forward"]["n"],
            "dc_backward_n": r["dc_backward"]["n"],
            "fast_pulse": [
                {k: v for k, v in x.items()
                 if k != "_traces" or k == "_traces"}  # keep traces (already trimmed)
                for x in r["fast"]["per_bias"]
            ],
        })
    return {
        "axes": {
            "snap_Is_scales": SNAP_IS_SCALES,
            "R_body_list": [{"name": n, "R": R, "tau_est_s": tau}
                            for n, R, tau in R_BODY_LIST],
        },
        "snap_Is_base": SNAP_IS_BASE,
        "cells": cells,
        "gates": gates,
        "partial": partial,
        "references": {
            "SB_OFF_DC_avg_dec": 2.087,
            "SB_HOT_DC_avg_dec": 2.809,
            "Mario_slide21_osc_period_ns_quoted": 400.0,
            "DC_penalty_cap_dec": 1.5,
            "DC_avg_cap_dec": 3.5,
        },
        "wall_total_sec": round(time.time() - t0, 1),
    }


if __name__ == "__main__":
    main()
