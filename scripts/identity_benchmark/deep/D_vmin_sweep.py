"""D. DPM Vmin sweep. Sweep power_dpm_force_performance_level low/auto/high
with 30s thermal-equilibrate between. At each level run 100 reps of a small
deterministic matmul on GPU via torch+ROCm (HSA override needed).
Detect bit-flips per CU-equivalent (we use 80-tile partition of output matrix).
"""
import argparse, os, sys, time, numpy as np, hashlib
sys.path.insert(0, os.path.dirname(__file__))
from _common import (DPM_FILE, temp_c, wait_cool, abort_if_hot, save_json,
                     host_label, bootstrap_ci)

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION","11.0.0")

def set_dpm(level):
    if not DPM_FILE: return False
    try:
        with open(DPM_FILE,"w") as f: f.write(level)
        return True
    except Exception as e:
        print(f"[WARN] DPM write {level} failed: {e}", flush=True); return False

def get_dpm():
    try: return open(DPM_FILE).read().strip()
    except: return "?"

def hash_tile(arr):
    return hashlib.blake2b(arr.tobytes(), digest_size=8).hexdigest()

def run_level(torch, level, reps, n_tiles=80):
    """Returns per-tile dict[tile_idx] -> set of distinct hashes across reps."""
    device = "cuda"
    rng = torch.manual_seed(0x1151)
    A = torch.randn(1024,1024, dtype=torch.float32, device=device, generator=None)
    B = torch.randn(1024,1024, dtype=torch.float32, device=device)
    # warm
    for _ in range(3):
        _ = (A @ B); torch.cuda.synchronize()
    tile_hashes = [set() for _ in range(n_tiles)]
    # 1024x1024 -> 80 tiles ~ 128x128 grid; we use row-strips of ~12 rows
    strip = max(1, 1024 // n_tiles)
    times = []
    for r in range(reps):
        t0 = time.time()
        C = A @ B
        torch.cuda.synchronize()
        times.append(time.time()-t0)
        C_cpu = C.detach().cpu().numpy()
        for i in range(n_tiles):
            tile = C_cpu[i*strip:(i+1)*strip].copy()
            tile_hashes[i].add(hash_tile(tile))
        if abort_if_hot(72):
            print(f"[ABORT] rep {r} hot at {level}", flush=True); break
        time.sleep(0.02)
    flips = [len(s)-1 for s in tile_hashes]  # 0 = stable
    return dict(level=level, n_reps_done=len(times), per_tile_flips=flips,
                n_unstable_tiles=int(sum(1 for f in flips if f>0)),
                rep_time_mean_ms=float(np.mean(times)*1000),
                rep_time_std_ms=float(np.std(times)*1000))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    args=ap.parse_args()
    if args.smoke: args.reps=10

    info = dict(host=host_label(), dpm_initial=get_dpm())
    try:
        import torch
    except Exception as e:
        info["error"] = f"torch import: {e}"; save_json(args.out, info); return
    info["cuda_available"] = bool(torch.cuda.is_available())
    if not torch.cuda.is_available():
        info["error"]="no rocm/cuda"; save_json(args.out, info); return
    info["device"] = str(torch.cuda.get_device_name(0))

    levels = ["low","auto","high"]
    results = []
    t_start=time.time()
    for lvl in levels:
        ok = set_dpm(lvl)
        info_lvl = dict(set_ok=ok, actual=get_dpm())
        if not wait_cool(thresh=55, timeout=90):
            print(f"[WARN] cool timeout pre-{lvl}", flush=True)
        if abort_if_hot(72):
            info_lvl["aborted"]=True; results.append(info_lvl); continue
        # short pre-soak
        time.sleep(15)
        try:
            r = run_level(torch, lvl, args.reps)
            info_lvl.update(r)
        except Exception as e:
            info_lvl["error"]=str(e)
        results.append(info_lvl)
        print(f"[{lvl}] unstable_tiles={info_lvl.get('n_unstable_tiles','?')} t_ms={info_lvl.get('rep_time_mean_ms','?')}", flush=True)
    # restore
    set_dpm("auto")
    info["levels"]=results; info["wall_s"]=time.time()-t_start
    save_json(args.out, info)

if __name__=="__main__": main()
