"""Faster C1 collection — no thermal pause (10-min run, peak ~91C, trip 99C).

Use this when the host has stable thermals and the regular collect()
hits hostile environment that keeps it paused. Outputs same shape file
as regular collect.
"""
import sys, time, socket, threading
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _telemetry import collect_window, CHANNELS, apu_temp_c

HOST = socket.gethostname()
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / f"results/IDENTITY_BENCHMARK_2026-05-30/embodiment5/c1_{HOST}"
OUT.mkdir(parents=True, exist_ok=True)

USE_CH = ["apu_temp_c", "gpu_temp_c", "gpu_power_w", "gpu_freq_mhz", "kern_lat_us"]
CH_IDX = [CHANNELS.index(c) for c in USE_CH]
DT = 0.20
TOTAL = 3000
CHUNK = 200
HARD_TRIP = 95.0  # only pause if very close to trip


class Driver:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.stop = False
    def run(self):
        cycle = [(200, 0.05), (400, 0.10), (700, 0.0), (250, 0.30),
                  (550, 0.05), (350, 0.15), (800, 0.0), (200, 0.5)]
        i = 0
        while not self.stop:
            if apu_temp_c() > HARD_TRIP:
                time.sleep(3.0)
                continue
            s, r = cycle[i % len(cycle)]
            a = self.rng.standard_normal((s, s)).astype(np.float32)
            t0 = time.time()
            while time.time() - t0 < 1.5 and not self.stop:
                _ = a @ a
            time.sleep(r)
            i += 1


def main():
    drv = Driver(seed=42)
    th = threading.Thread(target=drv.run, daemon=True)
    th.start()
    time.sleep(2.0)
    rows = []
    for start in range(0, TOTAL, CHUNK):
        n = min(CHUNK, TOTAL - start)
        w = collect_window(n, dt_s=DT)
        rows.append(w)
        t = apu_temp_c()
        print(f"  chunk {start}-{start+n} done, apu={t:.1f}C", flush=True)
    drv.stop = True
    data = np.concatenate(rows, axis=0)[:, CH_IDX]
    out = OUT / f"c1_{HOST}_data.npy"
    np.save(out, data)
    print(f"[FAST] saved {data.shape} to {out}")


if __name__ == "__main__":
    main()
