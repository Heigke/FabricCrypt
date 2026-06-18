#!/usr/bin/env python3
"""
Test deep GPU signals variance during training.

Monitors gpu_metrics binary blob to identify which low-level signals
are useful for embodiment (i.e., which ones actually vary with load).

Run this ON DAEDALUS while training is active.
"""

import struct
import time
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

# GPU metrics file path - adjust card number as needed
GPU_METRICS_PATH = Path("/sys/class/drm/card1/device/gpu_metrics")
GPU_BUSY_PATH = Path("/sys/class/drm/card1/device/gpu_busy_percent")


def read_gpu_metrics() -> Dict[str, float]:
    """Read and parse gpu_metrics binary blob."""
    try:
        with open(GPU_METRICS_PATH, "rb") as f:
            data = f.read()

        metrics = {}

        # Temperatures (0.01C units at offset 4, 6)
        metrics["temp_gfx"] = struct.unpack_from("<H", data, 4)[0] / 100.0
        metrics["temp_soc"] = struct.unpack_from("<H", data, 6)[0] / 100.0

        # Activity percentages
        metrics["gfx_activity"] = struct.unpack_from("<H", data, 42)[0]
        metrics["activity_62"] = struct.unpack_from("<H", data, 62)[0]
        metrics["mem_ctrl_activity"] = struct.unpack_from("<H", data, 66)[0]
        metrics["vcn_activity"] = struct.unpack_from("<H", data, 76)[0]
        metrics["activity_136"] = struct.unpack_from("<H", data, 136)[0]
        metrics["activity_138"] = struct.unpack_from("<H", data, 138)[0]

        # Clocks (MHz)
        metrics["gfxclk"] = struct.unpack_from("<H", data, 224)[0]
        metrics["socclk"] = struct.unpack_from("<H", data, 178)[0]
        metrics["memclk"] = struct.unpack_from("<H", data, 186)[0]
        metrics["fclk"] = struct.unpack_from("<H", data, 184)[0]

        # Throttle status
        metrics["throttle_status"] = struct.unpack_from("<I", data, 96)[0]

        # Store all 16-bit values for exploration
        for i in range(0, min(len(data)-2, 260), 2):
            metrics[f"raw_{i:03d}"] = struct.unpack_from("<H", data, i)[0]

        return metrics

    except Exception as e:
        print(f"Error reading gpu_metrics: {e}")
        return {}


def read_gpu_busy() -> int:
    """Read standard gpu_busy_percent for comparison."""
    try:
        return int(GPU_BUSY_PATH.read_text().strip())
    except:
        return 0


def main():
    print("=" * 70)
    print("DEEP GPU SIGNAL VARIANCE TEST")
    print("=" * 70)
    print("\nMonitoring gpu_metrics while training runs...")
    print("Press Ctrl+C to stop and see final analysis.\n")

    history = defaultdict(list)
    sample_interval = 0.1  # 100ms
    report_interval = 5.0  # Report every 5 seconds

    last_report = time.time()
    sample_count = 0

    try:
        while True:
            # Read metrics
            metrics = read_gpu_metrics()
            metrics["gpu_busy_std"] = read_gpu_busy()

            for name, value in metrics.items():
                history[name].append(float(value))

            sample_count += 1

            # Periodic report
            if time.time() - last_report >= report_interval:
                print(f"\n--- Sample {sample_count} ({sample_count * sample_interval:.1f}s) ---")
                print(f"Current: gfx={metrics.get('gfx_activity', 0)}% "
                      f"mem={metrics.get('mem_ctrl_activity', 0)}% "
                      f"clk={metrics.get('gfxclk', 0)}MHz "
                      f"T={metrics.get('temp_gfx', 0):.1f}C "
                      f"throttle=0x{metrics.get('throttle_status', 0):04x}")

                # Show top varying named signals
                print(f"\nNamed signal variance (last 50 samples):")
                for name in ["temp_gfx", "temp_soc", "gfx_activity", "mem_ctrl_activity", 
                             "vcn_activity", "gfxclk", "socclk", "memclk", "fclk", "throttle_status"]:
                    if name in history and len(history[name]) >= 10:
                        arr = np.array(history[name][-50:])
                        std = np.std(arr)
                        rng = np.max(arr) - np.min(arr)
                        cv = std / (np.mean(arr) + 1e-6)
                        useful = "***" if cv > 0.1 else "**" if cv > 0.05 else "*" if cv > 0.01 else ""
                        print(f"  {useful:3s} {name:20s}: mean={np.mean(arr):8.1f} std={std:6.2f} range={rng:6.1f} CV={cv:.4f}")

                last_report = time.time()

            time.sleep(sample_interval)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("FINAL ANALYSIS")
        print("=" * 70)

        print(f"\nTotal samples: {sample_count}")
        print(f"Duration: {sample_count * sample_interval:.1f}s")

        # Compute final stats for all signals
        stats = {}
        for name, values in history.items():
            if len(values) < 10:
                continue
            arr = np.array(values)
            stats[name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "range": float(np.max(arr) - np.min(arr)),
                "cv": float(np.std(arr) / (np.mean(arr) + 1e-6)),
            }

        # Find useful signals (with variance)
        useful = [(n, s) for n, s in stats.items() if s["cv"] > 0.005 or s["range"] > 0.5]
        useful.sort(key=lambda x: -x[1]["cv"])

        # Categorize signals
        print("\n=== NAMED SIGNALS WITH VARIANCE ===")
        named_useful = [(n, s) for n, s in useful if not n.startswith("raw_")]
        for name, s in named_useful:
            stars = "***" if s["cv"] > 0.1 else "**" if s["cv"] > 0.05 else "*"
            print(f"  {stars:3s} {name:20s}: mean={s['mean']:8.1f} std={s['std']:6.2f} "
                  f"range=[{s['min']:.1f}, {s['max']:.1f}] CV={s['cv']:.4f}")

        # Raw offsets with high variance
        print("\n=== RAW OFFSETS WITH HIGH VARIANCE (potential new signals) ===")
        raw_useful = [(n, s) for n, s in useful if n.startswith("raw_")]
        for name, s in raw_useful[:20]:  # Top 20
            offset = int(name.split("_")[1])
            stars = "***" if s["cv"] > 0.1 else "**" if s["cv"] > 0.05 else "*"
            print(f"  {stars:3s} offset {offset:3d}: mean={s['mean']:8.1f} std={s['std']:6.2f} "
                  f"range=[{s['min']:.1f}, {s['max']:.1f}] CV={s['cv']:.4f}")

        # Correlation with gpu_busy
        print("\n=== CORRELATION WITH gpu_busy_std ===")
        gpu_busy_vals = np.array(history["gpu_busy_std"])
        correlations = []
        for name, values in history.items():
            if name == "gpu_busy_std" or len(values) != len(gpu_busy_vals):
                continue
            arr = np.array(values)
            if np.std(arr) > 0.001 and np.std(gpu_busy_vals) > 0.001:
                corr = np.corrcoef(arr, gpu_busy_vals)[0, 1]
                if not np.isnan(corr):
                    correlations.append((name, corr))

        correlations.sort(key=lambda x: -abs(x[1]))
        print("Top signals correlated with GPU activity:")
        for name, corr in correlations[:20]:
            if name.startswith("raw_"):
                continue
            direction = "+" if corr > 0 else "-"
            print(f"  {direction} {name:20s}: r={corr:+.3f}")

        # Recommend signals for embodiment
        print("\n=== RECOMMENDED FOR EMBODIMENT ===")
        print("(High variance + correlated with activity = good for learning)")
        recommended = []
        corr_dict = dict(correlations)
        for name, s in named_useful:
            if name in corr_dict:
                score = abs(corr_dict[name]) * s["cv"] * 100
                recommended.append((name, s, corr_dict[name], score))

        recommended.sort(key=lambda x: -x[3])
        for name, s, corr, score in recommended[:10]:
            print(f"  [score={score:.2f}] {name}: CV={s['cv']:.4f}, corr={corr:+.3f}")

        # Save detailed results
        import json
        results = {
            "sample_count": sample_count,
            "duration_s": sample_count * sample_interval,
            "named_signals": {n: s for n, s in named_useful},
            "raw_offsets_with_variance": {n: s for n, s in raw_useful[:30]},
            "correlations_with_gpu_busy": {n: c for n, c in correlations[:30] if not n.startswith("raw_")},
            "recommended": [{"name": n, "cv": s["cv"], "corr": c, "score": sc} for n, s, c, sc in recommended[:10]],
        }

        out_path = Path(__file__).parent.parent / "results" / "deep_signal_analysis.json"
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {out_path}")


if __name__ == "__main__":
    main()
