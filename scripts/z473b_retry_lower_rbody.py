"""z473b — retry V6/V7/Mario at R_body=1e7 and 1e6 (skip slow V3 DC).
Writes results to results/z473_rbody_sweep/retry_lower.json.
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, str(ROOT / "scripts"))
import importlib.util as _ilu
def _load(name, path):
    sp=_ilu.spec_from_file_location(name,path); m=_ilu.module_from_spec(sp); sys.modules[name]=m; sp.loader.exec_module(m); return m
z473 = _load("z473", ROOT / "scripts/z473_rbody_sweep.py")

def main():
    out_dir = ROOT / "results/z473_rbody_sweep"
    cfg_flags = z473.make_NX_1p8()
    print("[retry] loading models", flush=True)
    model_M1, model_M2 = z473.z427.build_models()
    sebas_rows = z473.z427.load_sebas_params()

    results = {}
    for R in (1e7, 1e6):
        tag = f"{R:.0e}"
        print(f"[retry] === R_body={tag} ===", flush=True)
        log = lambda m: print(f"[{tag}] {m}", flush=True)
        # V6
        v6 = z473.run_V6(cfg_flags, model_M1, model_M2, sebas_rows, R, log)
        # V7
        v7, t7, vb7 = z473.run_V7(cfg_flags, model_M1, model_M2, sebas_rows, R, log)
        period = v7.get("period_ns") if isinstance(v7, dict) else None
        # Mario
        mario, _ = z473.mario_shape(cfg_flags, model_M1, model_M2, sebas_rows, R, period, log)
        results[tag] = {"R_body": R, "V6": v6, "V7": v7, "mario": mario}
        print(f"[retry] R={tag}: V6={v6.get('passed')} V7={v7.get('passed')} "
              f"mario={mario.get('n_metrics_matched','?')}/5", flush=True)
    (out_dir / "retry_lower.json").write_text(json.dumps(results, indent=2, default=float))
    print("[retry] wrote retry_lower.json", flush=True)

if __name__ == "__main__":
    main()
