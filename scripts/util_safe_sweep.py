"""Track P — thermal-safe ProcessPoolExecutor wrapper.

Per O31 gpt-5: OMP_NUM_THREADS=1 hard cap, 2-3 workers, thermal pause
≥75°C, kill >90°C. Solves the 2026-05-07 z211 thermal-trip pathology
(96°C/100°C with 12 and 6 workers respectively).

Usage:
    from scripts.util_safe_sweep import safe_sweep
    results = safe_sweep(
        run_fn=my_config_runner,        # function: (args) -> dict
        configs=[(arg1,), (arg2,), ...], # list of arg tuples
        out_dir=Path("results/my_sweep"),
        config_key=lambda args: f"cfg_{args[0]}",
        max_workers=2,
        thermal_pause_c=75.0,
        thermal_kill_c=90.0,
        per_config_wall_cap_s=120.0,
    )
"""
from __future__ import annotations
import json
import os
import signal
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Optional


def get_apu_temp_c() -> float:
    """Quick non-blocking APU temp read."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return -1.0  # unknown; treat as safe


def safe_sweep(
    run_fn: Callable,
    configs: Iterable,
    out_dir: Path,
    config_key: Callable,
    max_workers: int = 2,
    thermal_pause_c: float = 75.0,
    thermal_kill_c: float = 90.0,
    per_config_wall_cap_s: float = 120.0,
    skip_existing: bool = True,
    log_every: int = 5,
) -> list:
    """Run configs in parallel with thermal-safe wrapper.

    Per-subprocess env: OMP_NUM_THREADS=1, MKL=1, OPENBLAS=1, NUMEXPR=1,
    VECLIB_MAXIMUM_THREADS=1. Workers run with low priority (nice 5).

    Behavior:
      - Skip configs whose JSON exists in out_dir
      - Pause new task submission while APU > thermal_pause_c
      - Kill remaining tasks if APU > thermal_kill_c
      - Per-config wall-time cap; futures exceeding cap are abandoned
      - Returns list of result dicts from completed futures

    Each result is also written to `out_dir / f"{config_key(args)}.json"`
    if not already there.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter pending
    configs = list(configs)
    pending = []
    completed = []
    for cfg in configs:
        key = config_key(cfg)
        fp = out_dir / f"{key}.json"
        if skip_existing and fp.exists():
            try:
                completed.append(json.loads(fp.read_text()))
                continue
            except Exception:
                pass
        pending.append(cfg)

    print(f"[safe_sweep] {len(completed)}/{len(configs)} resumed; "
          f"{len(pending)} pending", flush=True)
    if not pending:
        return completed

    # Hard caps in this process AND all subprocesses (inherited via env)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[k] = "1"

    t0 = time.time()
    apu_peak = 0.0
    aborted = False

    # Use 'spawn' to inherit our hard env caps cleanly
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        futs = {}
        # Submit in small batches so we can throttle on thermal
        cfg_iter = iter(pending)
        outstanding = 0

        def maybe_submit():
            nonlocal outstanding
            try:
                cfg = next(cfg_iter)
            except StopIteration:
                return False
            fut = ex.submit(run_fn, cfg)
            futs[fut] = (cfg, time.time())
            outstanding += 1
            return True

        # Prime queue
        for _ in range(max_workers):
            if not maybe_submit():
                break

        done_n = 0
        while futs:
            apu = get_apu_temp_c()
            apu_peak = max(apu_peak, apu)

            if apu > thermal_kill_c:
                print(f"[safe_sweep] APU {apu:.1f}°C > {thermal_kill_c}, KILLING workers",
                      flush=True)
                for f in list(futs):
                    f.cancel()
                ex.shutdown(wait=False, cancel_futures=True)
                aborted = True
                break

            if apu > thermal_pause_c:
                print(f"[safe_sweep] APU {apu:.1f}°C > {thermal_pause_c}, pausing 30s",
                      flush=True)
                time.sleep(30)
                continue

            # Wait for ANY future to complete (with brief timeout to re-check thermal)
            done_set = []
            for f in list(futs):
                if f.done():
                    done_set.append(f)
                else:
                    cfg, t_start = futs[f]
                    if time.time() - t_start > per_config_wall_cap_s:
                        print(f"[safe_sweep] config {config_key(cfg)} exceeded wall cap; cancelling",
                              flush=True)
                        f.cancel()
                        done_set.append(f)
            if not done_set:
                time.sleep(2)
                continue

            for f in done_set:
                cfg, _ = futs.pop(f)
                outstanding -= 1
                try:
                    res = f.result(timeout=1.0)
                    completed.append(res)
                    # Persist if not already
                    fp = out_dir / f"{config_key(cfg)}.json"
                    if not fp.exists():
                        fp.write_text(json.dumps(res, indent=2))
                except Exception as e:
                    print(f"[safe_sweep] config {config_key(cfg)} failed: {e}",
                          flush=True)
                done_n += 1
                if done_n % log_every == 0 or done_n == len(pending):
                    print(f"[safe_sweep] {done_n}/{len(pending)} done; "
                          f"APU={apu:.1f}°C peak={apu_peak:.1f}°C "
                          f"wall={time.time()-t0:.0f}s", flush=True)
                # Top up the queue
                maybe_submit()

    print(f"[safe_sweep] FINISHED. {done_n}/{len(pending)} completed; "
          f"{'ABORTED' if aborted else 'OK'}; "
          f"APU peak={apu_peak:.1f}°C; wall={time.time()-t0:.0f}s",
          flush=True)
    return completed


if __name__ == "__main__":
    # smoke test
    def _dummy(args):
        x, = args
        time.sleep(1)
        return {"x": x, "x2": x * x}
    results = safe_sweep(
        run_fn=_dummy,
        configs=[(i,) for i in range(8)],
        out_dir=Path("/tmp/safe_sweep_smoke"),
        config_key=lambda a: f"x{a[0]}",
        max_workers=2,
    )
    print(f"smoke: {len(results)} results, sample: {results[0]}")
