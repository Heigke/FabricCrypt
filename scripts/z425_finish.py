#!/usr/bin/env python3
"""z425 finisher — runs remaining ablation variants + plots + summary.

Caches results from previous partial run via run.log parsing.
Re-runs ALL_FLAGS_ON with collect_traces=True for overlay/Vb plots.
Runs the remaining ablation variants (OFF_q1_be_oneway, OFF_mario_ipos,
OFF_sebas_per_bias, ONLY_mario_ipos, ALL_FLAGS_ON_HypA).
"""
from __future__ import annotations
import json, math, time, sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")

torch.set_default_dtype(torch.float64)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse z425's internals
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("z425", ROOT / "scripts/z425_ideal_floating_body.py")
z425 = _ilu.module_from_spec(_spec); _spec.loader.exec_module(z425)

OUT = ROOT / "results/z425_ideal_floating_body"

# Cached results from previous run.log (so we don't re-run)
CACHED = {
    "BASELINE": {
        "name": "BASELINE",
        "flags": {"use_sebas_per_bias_fits": True},
        "cell_rmse_dec": 4.357,
        "per_branch_rmse_dec": {"VG1_0.2": 2.330, "VG1_0.4": 4.286, "VG1_0.6": 5.291},
        "n_biases_evaluated": None,
        "convergence_failures": 0,
        "vb_max_overall": 0.994,
        "wall_sec": 204.9,
    },
    "OFF_suppress_bulk_diode": {
        "name": "OFF_suppress_bulk_diode",
        "flags": {"suppress_bulk_diode_forward": False, "q1_be_oneway": True,
                  "use_mario_ipos": True, "mario_ipos_param": "VG1",
                  "use_sebas_per_bias_fits": True},
        "cell_rmse_dec": 3.940,
        "per_branch_rmse_dec": {"VG1_0.2": 2.535, "VG1_0.4": 3.664, "VG1_0.6": 4.758},
        "n_biases_evaluated": None,
        "convergence_failures": 0,
        "vb_max_overall": 2.439,
        "wall_sec": 589.9,
    },
}

ALL_ON = dict(
    suppress_bulk_diode_forward=True,
    q1_be_oneway=True,
    use_mario_ipos=True,
    mario_ipos_param="VG1",
    use_sebas_per_bias_fits=True,
)

# Remaining variants to actually run
VARIANTS_TO_RUN = {
    "ALL_FLAGS_ON":            dict(ALL_ON),  # rerun with traces
    "OFF_q1_be_oneway":        {**ALL_ON, "q1_be_oneway": False},
    "OFF_mario_ipos":          {**ALL_ON, "use_mario_ipos": False},
    "OFF_sebas_per_bias":      {**ALL_ON, "use_sebas_per_bias_fits": False},
    "ONLY_mario_ipos":         dict(use_mario_ipos=True, mario_ipos_param="VG1",
                                     use_sebas_per_bias_fits=True),
    "ALL_FLAGS_ON_HypA":       {**ALL_ON, "mario_ipos_param": "VG2"},
}


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(OUT / "run.log", "a") as fh:
        fh.write(line + "\n")


def main():
    t_main = time.time()
    log("z425_finish starting")

    model_M1, model_M2 = z425.build_models()
    curves = z425.load_curves()
    sebas_rows = z425.load_sebas_params()
    log(f"{len(curves)} curves, {len(sebas_rows)} sebas rows")

    all_results = dict(CACHED)
    for name, flags in VARIANTS_TO_RUN.items():
        collect = (name == "ALL_FLAGS_ON")
        log(f"Running variant: {name}  flags={flags}")
        all_results[name] = z425.run_variant(name, flags, model_M1, model_M2,
                                              curves, sebas_rows,
                                              collect_traces=collect)

    headline = all_results["ALL_FLAGS_ON"]
    z425.plot_overlays(headline, OUT)
    z425.plot_vb_traces(headline, OUT)

    summary = {}
    for n, r in all_results.items():
        s = {k: v for k, v in r.items() if k not in ("per_bias", "traces")}
        summary[n] = s
    z425.plot_ablation_bar(summary, OUT / "ablation_bar.png")

    # Snapback in headline
    cell = headline["cell_rmse_dec"]
    vb_max = headline["vb_max_overall"]
    snapback_per_branch = {}
    for vg1 in [0.2, 0.4, 0.6]:
        sel = [r for r in headline.get("per_bias", []) if abs(r["VG1"] - vg1) < 1e-3]
        found = False
        for r in sel:
            Id = np.array(r["Id_pred"])
            if Id.size < 5:
                continue
            i_pk = int(np.argmax(Id))
            if i_pk < Id.size - 2 and Id[i_pk] > 1e-9:
                tail_min = Id[i_pk:].min()
                if tail_min > 0 and Id[i_pk] / tail_min > 2.0:
                    found = True; break
        snapback_per_branch[f"VG1_{vg1}"] = bool(found)

    base_cell = headline["cell_rmse_dec"]
    abl_deltas = {}
    for n in ("OFF_suppress_bulk_diode", "OFF_q1_be_oneway",
              "OFF_mario_ipos", "OFF_sebas_per_bias"):
        abl_deltas[n] = round(all_results[n]["cell_rmse_dec"] - base_cell, 3)
    each_flag_helps_03 = all(d >= 0.3 for d in abl_deltas.values())

    gates = {
        "INFRA": (all_results["ALL_FLAGS_ON"]["convergence_failures"] == 0),
        "DISCOVERY": (cell < 2.0 and vb_max > 0.7 and all(snapback_per_branch.values())),
        "AMBITIOUS": (cell < 1.0 and each_flag_helps_03),
        "KILL_SHOT": (cell > 3.5),
    }

    summary_out = {
        "variants": summary,
        "headline_variant": "ALL_FLAGS_ON",
        "headline_cell_rmse_dec": cell,
        "headline_vb_max": vb_max,
        "snapback_per_branch": snapback_per_branch,
        "ablation_deltas_dec": abl_deltas,
        "gates": gates,
        "elapsed_total_s": round(time.time() - t_main, 1),
        "note": "BASELINE and OFF_suppress_bulk_diode are cached from prior partial run (see run.log lines 1-5).",
    }
    (OUT / "summary.json").write_text(json.dumps(summary_out, indent=2))
    (OUT / "ablation.json").write_text(json.dumps(abl_deltas, indent=2))
    log(f"FINAL cell={cell:.3f}  Vb_max={vb_max:.3f}  gates={gates}")
    log(f"Ablation deltas: {abl_deltas}")
    return summary_out


if __name__ == "__main__":
    main()
