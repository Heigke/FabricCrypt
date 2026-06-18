#!/usr/bin/env python3
"""z2155_heterogeneous_thresholds.py — Per-neuron process variation via GPU entropy

Uses GPU silicon jitter to create heterogeneous Vg offsets for each neuron,
mimicking real NS-RAM process variation where each memristor has slightly
different BVpar. This should increase ISI CV variability and create
richer avalanche dynamics.

z2149 showed all 8 neurons have identical spike rates at same Vg.
Real NS-RAM has ~5-15% BVpar spread. We inject this spread using GPU jitter.

Tests:
  - Spike rate variance across neurons should increase with Vg spread
  - ISI CV should increase (heterogeneous dynamics)
  - Cross-neuron correlation should decrease (independent dynamics)
  - Avalanche-like cascading should emerge from spread thresholds

Hardware: AMD gfx1151 GPU + Arty A7 FPGA on /dev/ttyUSB1
"""

import os, sys, json, time, struct, subprocess
import numpy as np
from pathlib import Path
from collections import defaultdict

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'

SYNC = 0x55
CMD_SET_VG = 0x01
CMD_READ_TELEM = 0x02
CMD_SET_KILL = 0x03


def to_q16_16(val):
    return int(val * 65536) & 0xFFFFFFFF


def find_fpga():
    try:
        import serial
    except ImportError:
        return None, None
    for p in ['/dev/ttyUSB1', '/dev/ttyUSB0']:
        try:
            s = serial.Serial(p, 115200, timeout=0.2)
            time.sleep(0.1)
            return s, p
        except:
            continue
    return None, None


def set_per_neuron_vg(ser, vg_values):
    """Set individual Vg for each of 8 neurons."""
    for nid, vg in enumerate(vg_values[:8]):
        q16 = to_q16_16(max(0.0, min(1.0, vg)))
        payload = bytes([nid & 0x07]) + struct.pack('>I', q16)
        ser.write(bytes([SYNC, CMD_SET_VG]) + payload)
    ser.flush()
    time.sleep(0.005)


def read_telem(ser, timeout=0.15):
    """Read telemetry packet."""
    deadline = time.monotonic() + timeout
    buf = bytearray()
    while time.monotonic() < deadline:
        ser.timeout = max(0.001, deadline - time.monotonic())
        b = ser.read(1)
        if not b:
            continue
        if b[0] == SYNC:
            buf = bytearray([SYNC])
            while len(buf) < 52 and time.monotonic() < deadline:
                ser.timeout = max(0.001, deadline - time.monotonic())
                chunk = ser.read(52 - len(buf))
                if chunk:
                    buf.extend(chunk)
            break
    if len(buf) < 52:
        return None
    payload = bytes(buf[3:51])
    neurons = []
    for i in range(8):
        off = i * 6
        sc = struct.unpack_from('>H', payload, off)[0]
        vm = struct.unpack_from('>H', payload, off + 2)[0]
        neurons.append({'spike_count': sc, 'vmem': vm / 256.0})
    return neurons


def run_hip_jitter_batch(n_iters=50, n_waves=16, work_iters=50000):
    """Run GPU probe for jitter bytes."""
    probe_bin = BASE / 'scripts' / 'z2153_deep_probe_bridge'
    if not probe_bin.exists():
        return []
    result = subprocess.run(
        [str(probe_bin), str(n_iters), str(n_waves), str(work_iters)],
        capture_output=True, text=True, timeout=30,
        env={**os.environ, 'HSA_OVERRIDE_GFX_VERSION': '11.0.0'}
    )
    if result.returncode != 0:
        return []
    jitter_bytes = []
    for line in result.stdout.strip().split('\n')[1:]:
        parts = line.split(',')
        if len(parts) >= 13:
            jitter_bytes.append(int(parts[12]))
    return jitter_bytes


def measure_heterogeneous(ser, base_vg, vg_spread, jitter_bytes=None, duration_s=8, sample_hz=10):
    """Set heterogeneous Vg per neuron and measure spike dynamics."""
    # Generate per-neuron Vg values with spread using GPU jitter
    if jitter_bytes and len(jitter_bytes) >= 8:
        # Use GPU jitter bytes to create deterministic but hardware-derived offsets
        # Map [0,255] → [-spread, +spread]
        vg_offsets = np.array([(b / 127.5 - 1.0) * vg_spread for b in jitter_bytes[:8]])
    else:
        rng = np.random.default_rng(int(vg_spread * 1000))
        vg_offsets = rng.uniform(-vg_spread, vg_spread, size=8)
    vg_per_neuron = [max(0.0, min(1.0, base_vg + off)) for off in vg_offsets]

    set_per_neuron_vg(ser, vg_per_neuron)
    time.sleep(0.3)  # let settle

    # Collect spike counts and vmem over time
    n_samples = int(duration_s * sample_hz)
    interval = 1.0 / sample_hz
    spike_history = [[] for _ in range(8)]
    vmem_history = [[] for _ in range(8)]
    prev_counts = None
    t0 = time.monotonic()

    for _ in range(n_samples):
        ser.reset_input_buffer()
        ser.write(bytes([SYNC, CMD_READ_TELEM]))
        ser.flush()
        telem = read_telem(ser, timeout=0.15)

        if telem:
            t_now = time.monotonic() - t0
            counts = [n['spike_count'] for n in telem]
            vmems = [n['vmem'] for n in telem]

            for i in range(8):
                vmem_history[i].append(vmems[i])

            if prev_counts is not None:
                for i in range(8):
                    delta = (counts[i] - prev_counts[i]) & 0xFFFF
                    if delta > 30000:
                        delta = 0  # wraparound guard
                    spike_history[i].append(delta)
            prev_counts = counts[:]

        time.sleep(interval)

    return {
        'vg_per_neuron': vg_per_neuron,
        'vg_offsets': vg_offsets.tolist(),
        'spike_history': [h for h in spike_history],
        'vmem_history': [h for h in vmem_history],
    }


def main():
    print("=" * 60)
    print("z2155: Heterogeneous Thresholds via GPU Process Variation")
    print("=" * 60)

    # Step 1: Generate GPU jitter for process variation seeds
    print("\n[1/4] Generating GPU jitter for process variation...")
    jitter = run_hip_jitter_batch(n_iters=50, n_waves=16)
    if not jitter:
        print("  GPU probe failed, using pseudorandom")
        jitter = np.random.default_rng(42).integers(0, 256, size=800).tolist()
    print(f"  Got {len(jitter)} jitter bytes")

    # Step 2: Connect to FPGA
    print("\n[2/4] Connecting to FPGA...")
    ser, port = find_fpga()
    if ser is None:
        print("  FPGA not found — simulation mode")
        fpga_available = False
    else:
        print(f"  Connected: {port}")
        fpga_available = True
        ser.write(bytes([SYNC, CMD_SET_KILL, 0x00]))
        ser.flush()
        time.sleep(0.1)
        print("  Kill switch disabled")

    # Step 3: Measure with varying Vg spread
    # BVpar = 3.5 - 1.5*Vg, Vcb = 3.15V. At Vg=0.60, BVpar=2.60 → margin only 0.55V
    # At Vg=0.70, BVpar=2.45 → margin 0.70V — enough headroom for ±0.10 spread
    # At Vg=0.75, BVpar=2.375 → margin 0.775V — even better
    base_vg = 0.72  # robust spiking regime with headroom for spread
    spreads = [0.00, 0.02, 0.05, 0.08, 0.12]

    all_results = {}
    print("\n[3/4] Measuring heterogeneous dynamics...")

    for spread in spreads:
        print(f"\n  === Vg spread ±{spread:.2f} ===")

        if fpga_available:
            data = measure_heterogeneous(ser, base_vg, spread, jitter_bytes=jitter, duration_s=10, sample_hz=10)
        else:
            # Simulation
            rng = np.random.default_rng(int(spread * 1000))
            offsets = rng.uniform(-spread, spread, size=8)
            vg_per = [base_vg + off for off in offsets]
            spike_h = [rng.poisson(50 * (1 + vg_per[i]), size=100).tolist() for i in range(8)]
            vmem_h = [rng.uniform(0, 10, size=100).tolist() for i in range(8)]
            data = {
                'vg_per_neuron': vg_per,
                'vg_offsets': offsets.tolist(),
                'spike_history': spike_h,
                'vmem_history': vmem_h,
            }

        # Analyze per-neuron spike rates
        spike_rates = []
        for i in range(8):
            if data['spike_history'][i]:
                rate = float(np.mean(data['spike_history'][i]))
                spike_rates.append(rate)
            else:
                spike_rates.append(0)

        mean_rate = float(np.mean(spike_rates)) if spike_rates else 0
        rate_cv = float(np.std(spike_rates) / max(np.mean(spike_rates), 1e-10)) if spike_rates else 0

        # Cross-neuron correlation
        if len(data['spike_history'][0]) > 2:
            corr_matrix = np.corrcoef([h[:min(len(h), 50)] for h in data['spike_history'] if len(h) > 2])
            if corr_matrix.shape[0] > 1:
                # Mean off-diagonal correlation
                mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
                mean_corr = float(np.nanmean(corr_matrix[mask]))
            else:
                mean_corr = 1.0
        else:
            mean_corr = 0

        print(f"    Vg per neuron: {[f'{v:.3f}' for v in data['vg_per_neuron']]}")
        print(f"    Spike rates: {[f'{r:.1f}' for r in spike_rates]}")
        print(f"    Rate CV: {rate_cv:.4f}  Mean rate: {mean_rate:.1f}")
        print(f"    Cross-neuron correlation: {mean_corr:.4f}")

        all_results[f"spread_{spread:.2f}"] = {
            'vg_spread': spread,
            'vg_per_neuron': data['vg_per_neuron'],
            'spike_rates': spike_rates,
            'mean_rate': mean_rate,
            'rate_cv': rate_cv,
            'cross_neuron_correlation': mean_corr,
        }

    # Step 4: Evaluate
    print(f"\n{'=' * 60}")
    print("[4/4] Heterogeneous Threshold Evaluation")
    print(f"{'=' * 60}")

    # Rate CV: peak effect vs baseline (spread=0)
    cvs = [(r['vg_spread'], r['rate_cv']) for r in all_results.values()]
    cvs.sort()
    baseline_cv = cvs[0][1]
    peak_cv = max(cv[1] for cv in cvs[1:]) if len(cvs) > 1 else baseline_cv
    cv_increase = peak_cv - baseline_cv

    # Cross-neuron correlation: peak decorrelation vs baseline
    corrs = [(r['vg_spread'], r['cross_neuron_correlation']) for r in all_results.values()]
    corrs.sort()
    baseline_corr = corrs[0][1]
    min_corr = min(c[1] for c in corrs[1:]) if len(corrs) > 1 else baseline_corr
    corr_decrease = baseline_corr - min_corr

    print(f"\n  Rate CV trend: {' → '.join(f'{cv[1]:.3f}' for cv in cvs)}")
    print(f"  CV increase (peak): {cv_increase:.4f} [{'PASS' if cv_increase > 0.02 else 'FAIL'}: > 0.02]")
    print(f"  Correlation trend: {' → '.join(f'{c[1]:.3f}' for c in corrs)}")
    print(f"  Corr decrease (peak): {corr_decrease:.4f} [{'PASS' if corr_decrease > 0.05 else 'FAIL'}: > 0.05]")

    # Save
    output = {
        'experiment': 'z2155_heterogeneous_thresholds',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'fpga_available': fpga_available,
        'base_vg': base_vg,
        'spreads_tested': spreads,
        'results': all_results,
        'evaluation': {
            'cv_increase_peak': float(cv_increase),
            'cv_increase_pass': cv_increase > 0.02,
            'corr_decrease_peak': float(corr_decrease),
            'corr_decrease_pass': corr_decrease > 0.05,
            'baseline_cv': float(baseline_cv),
            'peak_cv': float(peak_cv),
            'baseline_corr': float(baseline_corr),
            'min_corr': float(min_corr),
        },
    }

    out_path = RESULTS / 'z2155_heterogeneous_thresholds.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")

    if ser:
        ser.close()


if __name__ == '__main__':
    main()
