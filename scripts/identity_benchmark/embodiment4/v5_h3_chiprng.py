"""V5-H3: Live substrate as entropy source vs PRNG for Bayesian sampling.

Hypothesis:
  /proc/interrupts deltas, hwmon temp jitter, and TSC jitter give us a
  hardware-derived RNG stream. For a Monte-Carlo / sampling task, this
  source should give equal-or-better statistical performance with strictly
  lower latency than a PRNG (no PRNG cost).

Reality check:
  Reading /proc/interrupts is itself ~50–200 us — typically SLOWER than
  numpy MT19937. To make this honest we batch reads: 1 read → 64 bytes
  of entropy → 16 float32s; PRNG amortized cost compared at matched batch.

Task:
  Monte-Carlo estimation of π via 1e6 (batched) samples. Compare:
   - PRNG (numpy default)
   - chip-RNG (sha256 of interrupts deltas + thermal jitter, repeatedly)
  Report: MC π estimate accuracy + total wall.

Gate: chip-RNG within 1% of PRNG accuracy AND chip-RNG ≥10% lower latency.
"""
from __future__ import annotations
import json, time, hashlib, os
from pathlib import Path
import numpy as np

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
import socket
HOST = socket.gethostname()
OUT = ROOT / f"results/IDENTITY_BENCHMARK_2026-05-30/embodiment4/v5_h3_{HOST}.json"

N_SAMPLES = 1_000_000
BATCH = 4096


def read_interrupts_bytes() -> bytes:
    try:
        return open("/proc/interrupts").read().encode()
    except Exception:
        return b""


def read_thermal_bytes() -> bytes:
    parts = []
    for i in range(8):
        try:
            parts.append(open(f"/sys/class/thermal/thermal_zone{i}/temp").read().strip().encode())
        except Exception:
            pass
    return b"|".join(parts)


def chip_rng_bytes(n_bytes: int) -> bytes:
    """SHA-256 chain seeded from interrupts + thermal + TSC."""
    out = bytearray()
    state = read_interrupts_bytes() + read_thermal_bytes() + str(time.perf_counter_ns()).encode()
    while len(out) < n_bytes:
        state = hashlib.sha256(state + str(time.perf_counter_ns()).encode()).digest()
        out.extend(state)
    return bytes(out[:n_bytes])


def chip_rng_floats(n: int) -> np.ndarray:
    raw = chip_rng_bytes(n * 4)
    arr = np.frombuffer(raw, dtype=np.uint32)
    return arr.astype(np.float64) / (2**32 - 1)


def mc_pi_prng(n_samples, batch) -> tuple:
    rng = np.random.default_rng(0xC0FFEE)
    t0 = time.perf_counter()
    inside = 0
    n_done = 0
    while n_done < n_samples:
        nb = min(batch, n_samples - n_done)
        x = rng.random(nb); y = rng.random(nb)
        inside += int(np.sum(x*x + y*y <= 1.0))
        n_done += nb
    elapsed = time.perf_counter() - t0
    return 4.0 * inside / n_samples, elapsed


def mc_pi_chip(n_samples, batch) -> tuple:
    t0 = time.perf_counter()
    inside = 0
    n_done = 0
    while n_done < n_samples:
        nb = min(batch, n_samples - n_done)
        floats = chip_rng_floats(nb * 2)
        x = floats[:nb]; y = floats[nb:nb*2]
        inside += int(np.sum(x*x + y*y <= 1.0))
        n_done += nb
    elapsed = time.perf_counter() - t0
    return 4.0 * inside / n_samples, elapsed


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_trials = 3
    prng_acc, prng_walls, chip_acc, chip_walls = [], [], [], []
    for trial in range(n_trials):
        p, wp = mc_pi_prng(N_SAMPLES, BATCH)
        c, wc = mc_pi_chip(N_SAMPLES, BATCH)
        prng_acc.append(abs(p - np.pi)); prng_walls.append(wp)
        chip_acc.append(abs(c - np.pi)); chip_walls.append(wc)
        print(f"[H3] trial={trial} PRNG: pi_est={p:.5f} err={abs(p-np.pi):.5f} wall={wp:.3f}s | chip: pi_est={c:.5f} err={abs(c-np.pi):.5f} wall={wc:.3f}s", flush=True)
    res = {
        "host": HOST, "n_samples": N_SAMPLES, "batch": BATCH,
        "prng_err_med": float(np.median(prng_acc)),
        "chip_err_med": float(np.median(chip_acc)),
        "prng_wall_med": float(np.median(prng_walls)),
        "chip_wall_med": float(np.median(chip_walls)),
    }
    res["latency_reduction_pct"] = 100.0 * (res["prng_wall_med"] - res["chip_wall_med"]) / res["prng_wall_med"]
    res["err_ratio"] = res["chip_err_med"] / max(1e-9, res["prng_err_med"])
    res["WIN"] = res["err_ratio"] <= 1.01 and res["latency_reduction_pct"] >= 10.0
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"[H3] err_ratio={res['err_ratio']:.3f} lat_reduction={res['latency_reduction_pct']:.1f}% WIN={res['WIN']}", flush=True)


if __name__ == "__main__":
    main()
