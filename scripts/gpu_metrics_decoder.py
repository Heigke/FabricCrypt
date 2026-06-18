#!/usr/bin/env python3
"""
AMD GPU Metrics v3.0 Binary Decoder — Raw ADC-Level Telemetry
==============================================================
Decodes the gpu_metrics sysfs binary for AMD APUs (Ryzen AI + RDNA3.5).
Format v3.0 — 264 bytes, all fields documented.

Analog physics proxies available:
  - Temperature: GFX + SoC (centi-Celsius, 0.01°C resolution)
  - Power: socket/APU/GFX/dGPU/all-core/per-core (milliwatts)
  - Clocks: GFX/SoC/VPE/IPU/FCLK/VCLK/UCLK/MPIPU (MHz)
  - Per-core C0 residency (centipercent)
  - DRAM bandwidth (reads/writes)
  - Throttle residency counters (7 causes)
  - Filter time constant (microseconds)

Usage:
  python gpu_metrics_decoder.py                    # single shot
  python gpu_metrics_decoder.py --loop 0.1 --duration 60  # 100ms sampling for 60s
  python gpu_metrics_decoder.py --raw              # hex dump + decode
"""

import struct
import json
import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path


# ─── gpu_metrics_v3_0 struct layout ──────────────────────────────────────────
# From: drivers/gpu/drm/amd/include/kgd_pp_interface.h (kernel 6.14+)
# Total: 264 bytes (0x108), natural alignment with 3 padding slots

V3_0_LAYOUT = [
    # (offset, fmt, name, unit, scale_divisor)
    # Header
    (0x0000, 'H', 'structure_size', 'bytes', 1),
    (0x0002, 'B', 'format_revision', '', 1),
    (0x0003, 'B', 'content_revision', '', 1),
    # Temperature (centi-Celsius)
    (0x0004, 'H', 'temperature_gfx', 'cC', 100),
    (0x0006, 'H', 'temperature_soc', 'cC', 100),
    # Per-core temperatures (16 cores)
    *[(0x0008 + i*2, 'H', f'temperature_core_{i}', 'cC', 100) for i in range(16)],
    (0x0028, 'H', 'temperature_skin', 'cC', 100),
    # Utilization (centipercent)
    (0x002A, 'H', 'average_gfx_activity', 'c%', 100),
    (0x002C, 'H', 'average_vcn_activity', 'c%', 100),
    # IPU activity (8 units)
    *[(0x002E + i*2, 'H', f'average_ipu_activity_{i}', 'c%', 100) for i in range(8)],
    # Per-core C0 activity (16 cores)
    *[(0x003E + i*2, 'H', f'average_core_c0_activity_{i}', 'c%', 100) for i in range(16)],
    # DRAM/IPU bandwidth
    (0x005E, 'H', 'average_dram_reads', 'cnt', 1),
    (0x0060, 'H', 'average_dram_writes', 'cnt', 1),
    (0x0062, 'H', 'average_ipu_reads', 'cnt', 1),
    (0x0064, 'H', 'average_ipu_writes', 'cnt', 1),
    # Padding at 0x0066 (2 bytes for u64 alignment)
    # System clock counter (nanoseconds)
    (0x0068, 'Q', 'system_clock_counter', 'ns', 1),
    # Power (milliwatts, u32)
    (0x0070, 'I', 'average_socket_power', 'mW', 1000),
    (0x0074, 'H', 'average_ipu_power', 'mW', 1000),
    # Padding at 0x0076 (2 bytes for u32 alignment)
    (0x0078, 'I', 'average_apu_power', 'mW', 1000),
    (0x007C, 'I', 'average_gfx_power', 'mW', 1000),
    (0x0080, 'I', 'average_dgpu_power', 'mW', 1000),
    (0x0084, 'I', 'average_all_core_power', 'mW', 1000),
    # Per-core power (milliwatts, u16)
    *[(0x0088 + i*2, 'H', f'average_core_power_{i}', 'mW', 1000) for i in range(16)],
    # System power and limits
    (0x00A8, 'H', 'average_sys_power', 'mW', 1000),
    (0x00AA, 'H', 'stapm_power_limit', 'mW', 1000),
    (0x00AC, 'H', 'current_stapm_power_limit', 'mW', 1000),
    # Average clocks (MHz)
    (0x00AE, 'H', 'average_gfxclk_frequency', 'MHz', 1),
    (0x00B0, 'H', 'average_socclk_frequency', 'MHz', 1),
    (0x00B2, 'H', 'average_vpeclk_frequency', 'MHz', 1),
    (0x00B4, 'H', 'average_ipuclk_frequency', 'MHz', 1),
    (0x00B6, 'H', 'average_fclk_frequency', 'MHz', 1),
    (0x00B8, 'H', 'average_vclk_frequency', 'MHz', 1),
    (0x00BA, 'H', 'average_uclk_frequency', 'MHz', 1),
    (0x00BC, 'H', 'average_mpipu_frequency', 'MHz', 1),
    # Current per-core clocks (MHz)
    *[(0x00BE + i*2, 'H', f'current_coreclk_{i}', 'MHz', 1) for i in range(16)],
    (0x00DE, 'H', 'current_core_maxfreq', 'MHz', 1),
    (0x00E0, 'H', 'current_gfx_maxfreq', 'MHz', 1),
    # Padding at 0x00E2 (2 bytes for u32 alignment)
    # Throttle residency counters
    (0x00E4, 'I', 'throttle_residency_prochot', 'cnt', 1),
    (0x00E8, 'I', 'throttle_residency_spl', 'cnt', 1),
    (0x00EC, 'I', 'throttle_residency_fppt', 'cnt', 1),
    (0x00F0, 'I', 'throttle_residency_sppt', 'cnt', 1),
    (0x00F4, 'I', 'throttle_residency_thm_core', 'cnt', 1),
    (0x00F8, 'I', 'throttle_residency_thm_gfx', 'cnt', 1),
    (0x00FC, 'I', 'throttle_residency_thm_soc', 'cnt', 1),
    # Filter time constant
    (0x0100, 'I', 'time_filter_alphavalue', 'us', 1),
    # Sentinel
    (0x0104, 'I', '_tail_sentinel', '', 1),
]


def decode_v3_0(data):
    """Decode gpu_metrics v3.0 binary data."""
    result = {}
    for offset, fmt, name, unit, divisor in V3_0_LAYOUT:
        sz = struct.calcsize(fmt)
        if offset + sz > len(data):
            break
        raw = struct.unpack_from(f'<{fmt}', data, offset)[0]
        result[name] = raw
        # Add scaled value for human-readable output
        if divisor > 1 and raw != 0 and raw != 0xFFFF and raw != 0xFFFFFFFF:
            if unit == 'cC':
                result[f'{name}_C'] = round(raw / divisor, 2)
            elif unit == 'c%':
                result[f'{name}_pct'] = round(raw / divisor, 2)
            elif unit == 'mW':
                result[f'{name}_W'] = round(raw / divisor, 3)
    return result


def find_gpu_metrics_path():
    """Find the gpu_metrics sysfs path."""
    for card in ['card0', 'card1', 'card2']:
        path = f'/sys/class/drm/{card}/device/gpu_metrics'
        if os.path.exists(path):
            return path, card
    return None, None


def read_gpu_metrics(path):
    """Read raw binary gpu_metrics."""
    with open(path, 'rb') as f:
        return f.read()


def print_decoded(decoded, verbose=False):
    """Pretty-print decoded metrics."""
    print(f"\n{'='*65}")
    print(f"AMD GPU Metrics v3.0 — Radeon 8060S (gfx1151)")
    print(f"{'='*65}")

    # Temperatures
    print(f"\n--- Temperatures ---")
    for key in ['temperature_gfx', 'temperature_soc', 'temperature_skin']:
        ckey = f'{key}_C'
        if ckey in decoded:
            print(f"  {key:35s}: {decoded[ckey]:7.2f} °C  (raw: {decoded[key]})")
    # Per-core temps
    core_temps = [(i, decoded.get(f'temperature_core_{i}_C', 0))
                  for i in range(16) if decoded.get(f'temperature_core_{i}', 0) > 0]
    if core_temps:
        print(f"  Per-core temps: {', '.join(f'C{i}={t:.1f}' for i,t in core_temps)}")

    # Power
    print(f"\n--- Power (milliwatts → Watts) ---")
    for key in ['average_socket_power', 'average_apu_power', 'average_gfx_power',
                'average_dgpu_power', 'average_all_core_power', 'average_ipu_power',
                'average_sys_power']:
        wkey = f'{key}_W'
        if wkey in decoded:
            print(f"  {key:35s}: {decoded[wkey]:8.3f} W  (raw: {decoded[key]} mW)")
        elif key in decoded:
            print(f"  {key:35s}: {decoded[key]} mW")

    # Per-core power
    core_powers = [(i, decoded.get(f'average_core_power_{i}_W', 0))
                   for i in range(16) if decoded.get(f'average_core_power_{i}', 0) > 0]
    if core_powers:
        print(f"  Per-core power: {', '.join(f'C{i}={p:.3f}W' for i,p in core_powers)}")

    # Power limits
    print(f"\n--- Power Limits ---")
    for key in ['stapm_power_limit', 'current_stapm_power_limit']:
        wkey = f'{key}_W'
        if wkey in decoded:
            print(f"  {key:35s}: {decoded[wkey]:8.3f} W")
        elif key in decoded:
            val = decoded[key]
            if val == 0xFFFF:
                print(f"  {key:35s}: UNLIMITED (0xFFFF)")
            else:
                print(f"  {key:35s}: {val} mW")

    # Clocks
    print(f"\n--- Average Clocks ---")
    for key in ['average_gfxclk_frequency', 'average_socclk_frequency',
                'average_vpeclk_frequency', 'average_ipuclk_frequency',
                'average_fclk_frequency', 'average_vclk_frequency',
                'average_uclk_frequency', 'average_mpipu_frequency']:
        if key in decoded:
            print(f"  {key:35s}: {decoded[key]:5d} MHz")

    print(f"\n--- Current Core Clocks ---")
    core_clks = [(i, decoded.get(f'current_coreclk_{i}', 0))
                 for i in range(16) if decoded.get(f'current_coreclk_{i}', 0) > 0]
    if core_clks:
        for i, clk in core_clks:
            print(f"  core_{i:2d}: {clk:5d} MHz")
    if 'current_core_maxfreq' in decoded:
        print(f"  {'current_core_maxfreq':35s}: {decoded['current_core_maxfreq']:5d} MHz")
    if 'current_gfx_maxfreq' in decoded:
        print(f"  {'current_gfx_maxfreq':35s}: {decoded['current_gfx_maxfreq']:5d} MHz")

    # Activity
    print(f"\n--- Activity ---")
    for key in ['average_gfx_activity', 'average_vcn_activity']:
        pkey = f'{key}_pct'
        if pkey in decoded:
            print(f"  {key:35s}: {decoded[pkey]:6.2f} %")
    core_act = [(i, decoded.get(f'average_core_c0_activity_{i}_pct', 0))
                for i in range(16) if decoded.get(f'average_core_c0_activity_{i}', 0) > 0]
    if core_act:
        print(f"  Per-core C0: {', '.join(f'C{i}={a:.2f}%' for i,a in core_act)}")

    # DRAM bandwidth
    print(f"\n--- DRAM Bandwidth ---")
    for key in ['average_dram_reads', 'average_dram_writes',
                'average_ipu_reads', 'average_ipu_writes']:
        if key in decoded:
            print(f"  {key:35s}: {decoded[key]}")

    # Throttle residency
    print(f"\n--- Throttle Residency ---")
    any_throttle = False
    for key in ['throttle_residency_prochot', 'throttle_residency_spl',
                'throttle_residency_fppt', 'throttle_residency_sppt',
                'throttle_residency_thm_core', 'throttle_residency_thm_gfx',
                'throttle_residency_thm_soc']:
        val = decoded.get(key, 0)
        if val > 0:
            any_throttle = True
            print(f"  {key:35s}: {val}")
    if not any_throttle:
        print(f"  (no throttling)")

    # System
    print(f"\n--- System ---")
    ts = decoded.get('system_clock_counter', 0)
    print(f"  {'system_clock_counter':35s}: {ts} ns ({ts/1e9:.3f} s)")
    print(f"  {'time_filter_alphavalue':35s}: {decoded.get('time_filter_alphavalue', 0)} µs")

    # Derived: leakage estimation
    gfx_power = decoded.get('average_gfx_power', 0)
    gfx_activity = decoded.get('average_gfx_activity', 0)
    if gfx_activity == 0 and gfx_power > 0:
        print(f"\n--- Analog Proxy: Leakage Estimation ---")
        print(f"  GFX idle power (leakage proxy):  {gfx_power/1000:.3f} W")
        temp = decoded.get('temperature_gfx', 0) / 100.0
        print(f"  At GFX temperature:              {temp:.2f} °C")


def continuous_trace(path, interval=0.1, duration=10.0, output=None):
    """High-frequency sampling for time-series analysis."""
    samples = []
    t0 = time.time()
    count = 0

    print(f"Tracing gpu_metrics every {interval*1000:.0f}ms for {duration}s...")
    print(f"{'t(s)':>7s}  {'T_gfx':>6s}  {'T_soc':>6s}  {'P_sock':>7s}  {'P_gfx':>7s}  {'P_core':>7s}  {'GCLK':>5s}  {'FCLK':>5s}  {'UCLK':>5s}")
    print("-" * 75)

    while time.time() - t0 < duration:
        try:
            data = read_gpu_metrics(path)
            decoded = decode_v3_0(data)
            t = time.time() - t0
            decoded['_timestamp'] = t

            t_gfx = decoded.get('temperature_gfx_C', 0)
            t_soc = decoded.get('temperature_soc_C', 0)
            p_sock = decoded.get('average_socket_power_W', 0)
            p_gfx = decoded.get('average_gfx_power_W', 0)
            p_core = decoded.get('average_all_core_power_W', 0)
            gclk = decoded.get('average_gfxclk_frequency', 0)
            fclk = decoded.get('average_fclk_frequency', 0)
            uclk = decoded.get('average_uclk_frequency', 0)

            if count % 20 == 0 and count > 0:
                print(f"{'t(s)':>7s}  {'T_gfx':>6s}  {'T_soc':>6s}  {'P_sock':>7s}  {'P_gfx':>7s}  {'P_core':>7s}  {'GCLK':>5s}  {'FCLK':>5s}  {'UCLK':>5s}")

            print(f"{t:7.2f}  {t_gfx:6.2f}  {t_soc:6.2f}  {p_sock:7.3f}  {p_gfx:7.3f}  {p_core:7.3f}  {gclk:5d}  {fclk:5d}  {uclk:5d}")

            samples.append(decoded)
            count += 1
            time.sleep(interval)
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(interval)

    print(f"\nCollected {count} samples in {time.time()-t0:.1f}s")

    if output:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        # Convert to compact format
        compact = {
            'device': 'AMD Radeon 8060S (gfx1151)',
            'format': 'gpu_metrics_v3_0',
            'interval_s': interval,
            'n_samples': count,
            'timestamp': datetime.now().isoformat(),
            'fields': list(samples[0].keys()) if samples else [],
            'samples': samples,
        }
        with open(output, 'w') as f:
            json.dump(compact, f, indent=2, default=str)
        print(f"Trace saved to {output}")

    return samples


def main():
    parser = argparse.ArgumentParser(description='AMD GPU Metrics v3.0 Decoder')
    parser.add_argument('--card', default=None, help='DRM card (auto-detect if omitted)')
    parser.add_argument('--loop', type=float, default=0,
                        help='Continuous sampling interval in seconds (0=single shot)')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Duration for continuous mode')
    parser.add_argument('--output', default=None, help='Output JSON path')
    parser.add_argument('--raw', action='store_true', help='Print raw hex dump')
    args = parser.parse_args()

    path, card = find_gpu_metrics_path()
    if args.card:
        path = f'/sys/class/drm/{args.card}/device/gpu_metrics'
        card = args.card

    if not path or not os.path.exists(path):
        print("ERROR: No gpu_metrics found")
        sys.exit(1)

    print(f"Reading from {path} ({card})")

    data = read_gpu_metrics(path)

    if args.raw:
        print(f"\nRaw gpu_metrics ({len(data)} bytes):")
        for i in range(0, len(data), 16):
            hex_part = ' '.join(f'{b:02x}' for b in data[i:i+16])
            ascii_part = ''.join(chr(b) if 32 <= b < 127 else '.' for b in data[i:i+16])
            print(f"  {i:04x}: {hex_part:<48s}  {ascii_part}")

    # Check format
    fmt_rev = data[2] if len(data) > 2 else 0
    cnt_rev = data[3] if len(data) > 3 else 0

    if fmt_rev != 3:
        print(f"WARNING: Expected format v3.x, got v{fmt_rev}.{cnt_rev}")

    if args.loop > 0:
        if args.output is None:
            args.output = f'results/gpu_metrics_trace_{int(time.time())}.json'
        continuous_trace(path, args.loop, args.duration, args.output)
    else:
        decoded = decode_v3_0(data)
        print_decoded(decoded)

        if args.output:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(decoded, f, indent=2, default=str)
            print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
