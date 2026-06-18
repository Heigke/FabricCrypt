"""z299: TCAD Zenodo replay.

Goal: parse Sentaurus sdevice `.cmd` files from the NS-RAM Zenodo bundle,
replay the bias setups with our pyport (BSIM4 + Gummel-Poon BJT), and
compare against TCAD outputs IF available.

HONEST DISCLOSURE:
  The Zenodo bundle ships ONLY the Sentaurus INPUTS (.cmd, .par, mesh .tdr,
  build logs). No trajectory outputs (.plt curves, .csv, simulated I(V)
  tables) are present. Without `tdr2plt`/`inspect` (Synopsys proprietary
  binaries, not installed here) we cannot decode the TCAD curves.

  -> We parse the cmd files, extract bias setups, run pyport sweeps for
     IdVgs and IdVds, and report HONESTLY that log-RMSE vs TCAD is
     UNAVAILABLE. Replay-OK is judged on Newton convergence + monotonic
     I-V shape (sanity).

Files parsed (per FloatBulk_Tsub/FloatBulk_Rsub):
  - IdVgs_des.cmd  : Vd=2.0 ramp + VG sweep -1 -> 2 V
  - IdVds_des.cmd  : VG=0.35, Vd sweep 0 -> 2 V
  - BV_des.cmd     : VG=0.35, Vd ramp toward 1e2 V (breakdown)
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import json
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TCAD_ROOT = ROOT / "data/nsram_zenodo/SimulationFiles/TCAD"
OUT_DIR = ROOT / "results/z299_tcad_replay"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "scripts"))


# ───────────────────────────── parse cmd ───────────────────────────────── #

DEFINE_RE = re.compile(r"#define\s+(\S+)\s+(.+?)\s*$", re.MULTILINE)
ELECTRODE_RE = re.compile(
    r'\{\s*Name="(\w+)"\s+Voltage=([^\s}]+)(?:\s+Resistor=\s*(\S+))?\s*\}'
)
GOAL_RE = re.compile(
    r'Goal\s*\{\s*Name="(\w+)"\s+Voltage=\s*([^\s}]+)\s*\}'
)
THERMODE_RE = re.compile(
    r'\{\s*Name="(\w+)"\s+Temperature=(\d+(?:\.\d+)?)'
)


def _resolve(expr: str, defines: dict) -> float:
    """Resolve a value: numeric literal, _NAME_, or @<expr>@."""
    expr = expr.strip().rstrip(",}")
    # @<...>@ inline expression
    m = re.match(r"@<(.+)>@", expr)
    if m:
        inner = m.group(1)
        for k, v in defines.items():
            inner = inner.replace(k, str(v))
        try:
            return float(eval(inner, {"__builtins__": {}}, {}))
        except Exception:
            return float("nan")
    # @NAME@ template placeholder
    if expr.startswith("@") and expr.endswith("@"):
        name = expr.strip("@")
        if name in defines:
            return float(defines[name])
        return float("nan")
    # _NAME_ define
    if expr in defines:
        try:
            return float(defines[expr])
        except Exception:
            return float("nan")
    # literal
    try:
        return float(expr)
    except Exception:
        return float("nan")


def parse_cmd(path: Path, gvars: dict) -> dict:
    txt = path.read_text(errors="replace")
    defines = dict(gvars)  # @VG@, @VG2@ etc come from gvars
    # Capture #define _Vdd_ 2.0 etc.
    for m in DEFINE_RE.finditer(txt):
        name = m.group(1).strip()
        val = m.group(2).strip().split()[0].rstrip(";")
        defines[name] = val
    # Electrodes (initial bias)
    electrodes = {}
    for m in ELECTRODE_RE.finditer(txt):
        name = m.group(1)
        v = _resolve(m.group(2), defines)
        electrodes[name] = v
    # Quasistationary Goals (ramp endpoints, in script order)
    goals = []
    for m in GOAL_RE.finditer(txt):
        goals.append((m.group(1), _resolve(m.group(2), defines)))
    # Thermode temp
    T = 300.0
    tm = THERMODE_RE.search(txt)
    if tm:
        T = float(tm.group(2))
    return {
        "file": str(path.relative_to(ROOT)),
        "defines": {k: defines[k] for k in defines if k.startswith("_") or k.startswith("V")},
        "electrodes_init": electrodes,
        "goals": goals,
        "temperature_K": T,
    }


def load_gvars(dir_path: Path) -> dict:
    """Load gtree.dat default param assignments."""
    gv = {}
    f = dir_path / "gtree.dat"
    if not f.exists():
        return gv
    # Lines look like: BV VG "0.35" {0.35 0.3} R
    for line in f.read_text().splitlines():
        m = re.match(r"\S+\s+(\w+)\s+\"([^\"]+)\"", line)
        if m:
            name, val = m.group(1), m.group(2)
            # Skip non-numeric (e.g. tmodel="DD")
            try:
                float(val)
                gv[name] = val
            except ValueError:
                gv[name] = val  # keep string
    return gv


# ───────────────────────────── pyport replay ───────────────────────────── #

def replay_idvgs(cfg, M1, M2, bjt, vd_max: float, vg_min: float, vg_max: float,
                  npts: int = 41) -> dict:
    from nsram_surrogate_4d import _solve_at_fixed_vb
    Vg = np.linspace(vg_min, vg_max, npts)
    Id = np.full(npts, np.nan)
    conv = np.zeros(npts, dtype=bool)
    for i, vg in enumerate(Vg):
        try:
            out = _solve_at_fixed_vb(cfg, M1, M2, bjt, vd_max, float(vg), 0.0, 0.0)
            Id[i] = out["Id"]
            conv[i] = out["converged"]
        except Exception:
            pass
    return {"V_sweep_name": "VG", "V_sweep": Vg.tolist(), "Id": Id.tolist(),
            "converged_frac": float(conv.mean()),
            "monotonic_inc": bool(np.all(np.diff(np.nan_to_num(Id)) >= -1e-9))}


def replay_idvds(cfg, M1, M2, bjt, vg: float, vd_max: float, npts: int = 41) -> dict:
    from nsram_surrogate_4d import _solve_at_fixed_vb
    Vd = np.linspace(0.0, vd_max, npts)
    Id = np.full(npts, np.nan)
    conv = np.zeros(npts, dtype=bool)
    for i, vd in enumerate(Vd):
        try:
            out = _solve_at_fixed_vb(cfg, M1, M2, bjt, float(vd), vg, 0.0, 0.0)
            Id[i] = out["Id"]
            conv[i] = out["converged"]
        except Exception:
            pass
    return {"V_sweep_name": "VD", "V_sweep": Vd.tolist(), "Id": Id.tolist(),
            "converged_frac": float(conv.mean()),
            "monotonic_inc": bool(np.all(np.diff(np.nan_to_num(Id)) >= -1e-9))}


def replay_bv(cfg, M1, M2, bjt, vg: float, vd_max: float, npts: int = 81) -> dict:
    """BV sweep — Vd 0 -> vd_max (TCAD pushes to 1e2 but pyport BSIM4 is
    valid up to ~5V; we cap at 5V which still captures snapback region)."""
    from nsram_surrogate_4d import _solve_at_fixed_vb
    Vd_cap = min(vd_max, 5.0)
    Vd = np.linspace(0.0, Vd_cap, npts)
    Id = np.full(npts, np.nan)
    conv = np.zeros(npts, dtype=bool)
    for i, vd in enumerate(Vd):
        try:
            out = _solve_at_fixed_vb(cfg, M1, M2, bjt, float(vd), vg, 0.0, 0.0)
            Id[i] = out["Id"]
            conv[i] = out["converged"]
        except Exception:
            pass
    Id_arr = np.array(Id)
    Id_arr_clean = np.nan_to_num(Id_arr)
    snapback = bool(np.any(np.diff(Id_arr_clean) < -1e-9))
    return {"V_sweep_name": "VD", "V_sweep": Vd.tolist(), "Id": Id.tolist(),
            "vd_cap_V": Vd_cap, "vd_max_requested": vd_max,
            "converged_frac": float(conv.mean()),
            "snapback_detected": snapback}


# ───────────────────────────── tdr/plt probe ───────────────────────────── #

def probe_tcad_outputs(dir_path: Path) -> dict:
    """Look for any TCAD trajectory output. We expect NONE in Zenodo bundle."""
    candidates = []
    for ext in ("plt", "csv", "tsv", "out", "txt"):
        candidates += [str(p.relative_to(ROOT)) for p in dir_path.glob(f"*.{ext}")]
    # tdr files in this bundle: only mesh (_msh.tdr, _bnd.tdr), not curve
    tdr_files = list(dir_path.glob("*.tdr"))
    tdr_curve = [p.name for p in tdr_files
                 if "_msh" not in p.name and "_bnd" not in p.name]
    # Probe for Synopsys binary tools
    have_tdr2plt = os.system("which tdr2plt > /dev/null 2>&1") == 0
    have_inspect = os.system("which inspect > /dev/null 2>&1") == 0
    return {
        "trajectory_candidates": candidates,
        "non_mesh_tdr": tdr_curve,
        "tdr2plt_available": have_tdr2plt,
        "inspect_available": have_inspect,
    }


# ───────────────────────────── main ────────────────────────────────────── #

def main():
    t0 = time.time()
    print(f"[z299] TCAD replay start; root={TCAD_ROOT}")

    # Build pyport models once
    from nsram_surrogate_4d import _build_pyport_models
    cfg, M1, M2, bjt = _build_pyport_models()
    print("[z299] pyport models built")

    summary = {
        "tcad_root": str(TCAD_ROOT.relative_to(ROOT)),
        "inventory": {},
        "replays": [],
        "outputs_probe": {},
        "gates": {},
        "wall_s": 0.0,
        "notes": [
            "Zenodo bundle ships Sentaurus INPUTS only.",
            "No .plt/.csv trajectory outputs present.",
            "Mesh .tdr (geometry) is present but is NOT an I(V) curve.",
            "Decoding requires Synopsys tdr2plt/inspect (proprietary, absent).",
            "Therefore log-RMSE vs TCAD is UNAVAILABLE / NOT COMPUTED.",
            "Replay-OK judged on (a) parse success, (b) Newton convergence,",
            "(c) qualitative shape (monotonic I-V, snapback presence).",
        ],
    }

    # Inventory both subdirs
    for sub in ("FloatBulk_Tsub", "FloatBulk_Rsub"):
        d = TCAD_ROOT / sub
        if not d.exists():
            continue
        all_files = sorted(p.name for p in d.iterdir() if p.is_file())
        by_ext = {}
        for f in all_files:
            ext = f.rsplit(".", 1)[-1] if "." in f else "noext"
            by_ext.setdefault(ext, []).append(f)
        summary["inventory"][sub] = {
            "n_files": len(all_files),
            "by_extension": {k: len(v) for k, v in by_ext.items()},
            "cmd_files": [f for f in all_files if f.endswith(".cmd")],
        }
        summary["outputs_probe"][sub] = probe_tcad_outputs(d)

    # Target cmd files (focus on Tsub which is the primary; Rsub uses same script)
    target_dir = TCAD_ROOT / "FloatBulk_Tsub"
    gvars = load_gvars(target_dir)
    summary["gvars_defaults"] = gvars

    targets = [
        ("IdVgs_des.cmd", "idvgs"),
        ("IdVds_des.cmd", "idvds"),
        ("BV_des.cmd",    "bv"),
        ("IdVgs1_des.cmd", "idvgs1"),
    ]

    n_parsed = 0
    n_replayed = 0
    log_rmse_per_file = {}

    for fname, kind in targets:
        fpath = target_dir / fname
        rec = {"file": fname, "kind": kind, "parsed_ok": False,
               "replayed_ok": False, "compared_ok": False,
               "log_rmse_dec": None, "error": None}
        if not fpath.exists():
            rec["error"] = "file_missing"
            summary["replays"].append(rec)
            continue
        try:
            parsed = parse_cmd(fpath, gvars)
            rec["parsed"] = parsed
            rec["parsed_ok"] = True
            n_parsed += 1
        except Exception as e:
            rec["error"] = f"parse: {e}\n{traceback.format_exc()[-300:]}"
            summary["replays"].append(rec)
            continue

        # Decide bias setup per kind
        try:
            goals = parsed["goals"]
            defines = parsed["defines"]
            vdd = float(defines.get("_Vdd_", 1.0))
            vginit = float(defines.get("_Vginit_", 0.0))
            if kind == "idvgs":
                # First goal: drain -> Vdd; second goal: gate -> 2*Vdd (or @_Vdd_@)
                vd_max = float(goals[0][1]) if goals else vdd
                vg_end = float(goals[1][1]) if len(goals) >= 2 else 2.0 * vdd
                rec["replay"] = replay_idvgs(cfg, M1, M2, bjt, vd_max, vginit, vg_end)
                rec["replay"]["bias"] = {"vd_hold_V": vd_max, "vg_range_V": [vginit, vg_end]}
                rec["replayed_ok"] = rec["replay"]["converged_frac"] > 0.5
            elif kind == "idvgs1":
                vd_max = float(goals[0][1]) if goals else vdd
                vg_end = float(goals[1][1]) if len(goals) >= 2 else 2.0 * vdd
                rec["replay"] = replay_idvgs(cfg, M1, M2, bjt, vd_max, vginit, vg_end)
                rec["replay"]["bias"] = {"vd_hold_V": vd_max, "vg_range_V": [vginit, vg_end]}
                rec["replayed_ok"] = rec["replay"]["converged_frac"] > 0.5
            elif kind == "idvds":
                # VG held at vginit, Vd ramped to Vdd
                vd_max = float(goals[0][1]) if goals else vdd
                rec["replay"] = replay_idvds(cfg, M1, M2, bjt, vginit, vd_max)
                rec["replay"]["bias"] = {"vg_hold_V": vginit, "vd_range_V": [0.0, vd_max]}
                rec["replayed_ok"] = rec["replay"]["converged_frac"] > 0.5
            elif kind == "bv":
                # BV: VG=@VG@ (gvar default 0.35), Vd ramp toward _vdd_=1e2
                vg_bv = float(gvars.get("VG", 0.35))
                # vdd is the literal _vdd_=1e2 in BV_des.cmd, but our pyport
                # cannot survive that. Cap is enforced inside replay_bv.
                rec["replay"] = replay_bv(cfg, M1, M2, bjt, vg_bv, 100.0)
                rec["replay"]["bias"] = {"vg_hold_V": vg_bv, "vd_range_V": [0.0, 100.0]}
                rec["replayed_ok"] = rec["replay"]["converged_frac"] > 0.5
            if rec["replayed_ok"]:
                n_replayed += 1
            # Comparison: not possible — no TCAD output. Log it.
            rec["compared_ok"] = False
            rec["log_rmse_dec"] = None
            rec["compare_note"] = "TCAD output trajectory absent in Zenodo bundle"
            log_rmse_per_file[fname] = None
        except Exception as e:
            rec["error"] = f"replay: {e}\n{traceback.format_exc()[-400:]}"
        summary["replays"].append(rec)
        print(f"[z299] {fname}: parsed={rec['parsed_ok']} replayed={rec['replayed_ok']}")

    # Gates (locked)
    # PASS-conservative: ≥1 cmd parsed AND replayed AND log-RMSE < 0.5 dec vs TCAD
    # AMBITIOUS:        ≥2 parsed/replayed both < 0.3 dec
    # Comparison was impossible. Document HONESTLY.
    summary["gates"] = {
        "conservative": {
            "passed": False,
            "reason": ("requires log-RMSE < 0.5 dec vs TCAD output; "
                       "TCAD output not extractable from Zenodo bundle "
                       "(only mesh .tdr + .cmd inputs; no .plt curves; "
                       "no Synopsys tdr2plt/inspect installed). "
                       f"Parsed: {n_parsed}; Replayed: {n_replayed}."),
        },
        "ambitious": {
            "passed": False,
            "reason": "same as conservative",
        },
        "alt_inputs_only": {
            "passed": n_parsed >= 2 and n_replayed >= 2,
            "definition": "≥2 cmd files parsed AND pyport replay converged",
            "n_parsed": n_parsed,
            "n_replayed": n_replayed,
        },
    }
    summary["wall_s"] = time.time() - t0

    # Save full summary
    out_path = OUT_DIR / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[z299] wrote {out_path} (wall={summary['wall_s']:.1f}s)")

    # Save per-replay I-V curves as npz for posterity
    for rec in summary["replays"]:
        if rec.get("replayed_ok") and "replay" in rec:
            tag = rec["file"].replace(".cmd", "")
            r = rec["replay"]
            np.savez(OUT_DIR / f"replay_{tag}.npz",
                     V=np.array(r["V_sweep"]), Id=np.array(r["Id"]),
                     converged_frac=r["converged_frac"])

    return summary


if __name__ == "__main__":
    main()
