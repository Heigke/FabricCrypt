#!/usr/bin/env python3
"""z2204_raw_firmware_neuromorphic.py — Raw Firmware Physics as Neuromorphic Substrate

THE GOLDEN NUGGET EXPERIMENT: Can raw GPU firmware register dynamics — with ZERO
signal processing, ZERO filtering, ZERO artificial engineering — drive FPGA neurons
to perform useful computation?

Hypothesis: The GPU's internal physics (VRM switching, thermal regulation, clock
management) has dynamics that are INHERENTLY neuromorphic — they exhibit temporal
correlations, 1/f spectra, and homeostatic regulation that happen to be exactly
what biological neural systems use.

5 "Raw" noise sources from firmware, with NO processing:
  RAW_POWER:     hwmon power1_average → direct Vg mapping (VRM switching noise)
  RAW_SMN:       SMN thermal registers → direct Vg mapping (silicon thermal noise)
  RAW_PM_TABLE:  PM table bytes → direct Vg mapping (firmware regulatory dynamics)
  RAW_CLOCK:     gpu_metrics GFX clock → direct Vg mapping (DVFS dynamics)
  RAW_PERF:      PERF_SNAPSHOT jitter → direct Vg mapping (instruction scheduling)

3 Control conditions:
  WHITE:         np.random.randn → Vg (artificial white noise, no physics)
  SYNTHETIC_1F:  Voss-McCartney 1/f → Vg (artificially engineered 1/f)
  NO_NOISE:      constant Vg (no noise at all)

Processing: NONE. Raw values are linearly mapped to [0.05, 0.95] Vg range.
No filtering, no IIR, no spectral shaping, no normalization beyond min-max scaling.

Task: 3-class waveform classification (120 trials, 25 steps, 5-fold CV)

Tests T291-T298:
  T291: At least one RAW source > WHITE (raw physics > artificial noise)
  T292: Best RAW source > NO_NOISE (raw physics helps)
  T293: Best RAW source accuracy > 45% (genuinely useful, not barely above chance)
  T294: RAW_POWER PSD slope in [-2.5, -0.3] (native 1/f without filtering)
  T295: RAW_SMN temporal autocorrelation > 0.5 at lag=1 (inherent memory)
  T296: At least 2 RAW sources > 40% (multiple firmware subsystems are useful)
  T297: RAW_POWER > SYNTHETIC_1F (natural 1/f > engineered 1/f)
  T298: Inter-trial variance lower for RAW sources than WHITE (physics is structured)

Hardware: AMD gfx1151 GPU + Arty A7 FPGA on /dev/ttyUSB*
Firmware access: ryzen_smu kernel module for SMN/PM table
"""

import os, sys, json, time, struct, argparse
import numpy as np
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
RESULTS = BASE / 'results'
FIGURES = RESULTS / 'FEEL_paper_update' / 'FEEL__Functionally_Embodied_Emergent_Learning__13_-5' / 'figures'

SYNC = 0x55
CMD_SET_VG = 0x01
CMD_READ_TELEM = 0x02
CMD_SET_KILL = 0x03

N_NEURONS = 8
STEPS_PER_TRIAL = 25
SAMPLE_HZ = 20
N_TRIALS = 120
N_FOLDS = 5
BASE_VG = 0.45
ALPHA = 0.15

HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"
SMN_PATH = "/sys/kernel/ryzen_smu_drv/smn"
PM_TABLE_PATH = "/sys/kernel/ryzen_smu_drv/pm_table"
GPU_METRICS = None  # Will try to find

# SMN thermal register addresses (from z2200)
SMN_ADDRS = [0x59800, 0x59804, 0x59808, 0x5980C, 0x59810, 0x59814, 0x59818, 0x5981C]


def find_fpga():
    import serial
    import glob as gl
    ports = sorted(gl.glob('/dev/ttyUSB*'), reverse=True)
    for p in ports:
        try:
            s = serial.Serial(p, 115200, timeout=0.3)
            time.sleep(0.1)
            s.write(bytes([SYNC, CMD_SET_KILL, 0x00]))
            s.flush()
            time.sleep(0.05)
            s.write(bytes([SYNC, CMD_READ_TELEM]))
            s.flush()
            resp = s.read(52)
            if len(resp) >= 52 and resp[0] == SYNC and resp[1] == CMD_READ_TELEM:
                return s, p
            s.close()
        except Exception:
            continue
    return None, None

def send_set_vg(ser, neuron_id, vg_val):
    vg_q16 = int(np.clip(vg_val, 0.05, 0.95) * 65536) & 0xFFFFFFFF
    pkt = bytes([SYNC, CMD_SET_VG, neuron_id & 0x07]) + struct.pack('>I', vg_q16)
    ser.write(pkt)
    ser.flush()

def read_telem(ser):
    ser.write(bytes([SYNC, CMD_READ_TELEM]))
    ser.flush()
    resp = ser.read(52)
    if len(resp) < 52 or resp[0] != SYNC or resp[1] != CMD_READ_TELEM:
        return None
    payload = resp[2:51]
    data = {}
    data['delta_spikes'] = [payload[i] for i in range(N_NEURONS)]
    data['vmem'] = [struct.unpack('>H', payload[8+i*2:10+i*2])[0] for i in range(N_NEURONS)]
    return data


# ─── Raw Firmware Noise Sources ───

def read_raw_power():
    """Raw VRM switching noise — no processing."""
    try:
        with open(HWMON_POWER) as f:
            return int(f.read().strip()) / 1e6  # µW → W
    except:
        return 0.0

def read_raw_smn(addr):
    """Raw SMN register value — no processing."""
    try:
        with open(SMN_PATH, 'rb') as f:
            f.seek(addr)
            return struct.unpack('<I', f.read(4))[0]
    except:
        return 0

def read_raw_pm_table(offset, n_bytes=4):
    """Raw PM table bytes — no processing."""
    try:
        with open(PM_TABLE_PATH, 'rb') as f:
            f.seek(offset)
            return struct.unpack('<I', f.read(n_bytes))[0]
    except:
        return 0

def read_raw_clock():
    """Raw GPU clock from gpu_metrics — no processing."""
    try:
        import glob as gl
        paths = gl.glob('/sys/class/drm/card*/device/gpu_metrics')
        if paths:
            with open(paths[0], 'rb') as f:
                data = f.read()
            if len(data) >= 30:
                # gfx_clock at offset 26 (uint16)
                return struct.unpack('<H', data[26:28])[0]
    except:
        pass
    return 0

def read_raw_perf():
    """Raw performance counter jitter — timing of reads."""
    t0 = time.perf_counter_ns()
    # Do a minimal read operation
    try:
        with open('/proc/self/stat') as f:
            _ = f.read()
    except:
        pass
    t1 = time.perf_counter_ns()
    return (t1 - t0)  # nanoseconds


def collect_raw_noise(source_name, n_samples):
    """Collect raw noise from a source, return as numpy array. NO FILTERING."""
    samples = []
    for _ in range(n_samples):
        if source_name == 'RAW_POWER':
            samples.append(read_raw_power())
        elif source_name == 'RAW_SMN':
            # Read all 8 SMN registers, take mean
            vals = [read_raw_smn(a) for a in SMN_ADDRS]
            samples.append(np.mean(vals))
        elif source_name == 'RAW_PM_TABLE':
            # Read regulatory state from PM table
            vals = [read_raw_pm_table(off) for off in [0x10, 0x14, 0x18, 0x1C]]
            samples.append(np.mean(vals))
        elif source_name == 'RAW_CLOCK':
            samples.append(read_raw_clock())
        elif source_name == 'RAW_PERF':
            samples.append(read_raw_perf())
        time.sleep(1.0 / SAMPLE_HZ)
    return np.array(samples, dtype=float)


def raw_to_vg(raw_value, raw_min, raw_max):
    """Linear map from raw range to [0.05, 0.95]. NO filtering, NO normalization tricks."""
    if raw_max == raw_min:
        return 0.50
    normalized = (raw_value - raw_min) / (raw_max - raw_min)
    return 0.05 + 0.90 * np.clip(normalized, 0.0, 1.0)


def voss_mcartney_1f(n, rng):
    """Voss-McCartney 1/f noise generator."""
    n_rows = 16
    rows = rng.randn(n_rows)
    total = rows.sum()
    output = np.zeros(n)
    for i in range(n):
        output[i] = total
        # Update one random row
        j = 0
        k = i
        while k % 2 == 0 and j < n_rows - 1:
            k //= 2
            j += 1
        old = rows[j]
        rows[j] = rng.randn()
        total += rows[j] - old
    return output


def generate_waveforms(n_trials, steps, hz, rng):
    labels = np.repeat([0, 1, 2], (n_trials + 2) // 3)[:n_trials]
    rng.shuffle(labels)
    t = np.linspace(0, steps / hz, steps)
    signals = []
    for label in labels:
        freq = rng.uniform(0.5, 2.0)
        if label == 0:
            sig = np.sin(2 * np.pi * freq * t)
        elif label == 1:
            sig = 2 * np.abs(2 * (t * freq % 1) - 1) - 1
        else:
            sig = np.sign(np.sin(2 * np.pi * freq * t))
        signals.append(sig * 0.3)
    return np.array(signals), labels


def compute_psd_slope(signal, fs=20):
    """Compute PSD slope via linear regression in log-log space."""
    from scipy.signal import welch
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    mask = freqs > 0
    if mask.sum() < 2:
        return 0.0
    log_f = np.log10(freqs[mask])
    log_p = np.log10(psd[mask] + 1e-20)
    coeffs = np.polyfit(log_f, log_p, 1)
    return coeffs[0]  # slope


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=N_TRIALS)
    args = parser.parse_args()

    print("=" * 65)
    print("z2204: Raw Firmware Physics as Neuromorphic Substrate")
    print("=" * 65)
    print("  THE GOLDEN NUGGET: Zero processing, raw firmware → FPGA spikes → classification")

    ser, port = find_fpga()
    if ser is None:
        print("[ERR] FPGA not found")
        sys.exit(1)
    print(f"[HW] FPGA on {port}")

    # Check firmware access
    smn_ok = os.path.exists(SMN_PATH)
    pm_ok = os.path.exists(PM_TABLE_PATH)
    print(f"[HW] SMN access: {smn_ok}, PM table: {pm_ok}")

    rng = np.random.RandomState(42)

    # ─── Phase 1: Collect raw noise samples for calibration ───
    print("\n[1/4] Calibrating raw firmware noise sources (30s each)...")

    raw_sources = ['RAW_POWER', 'RAW_SMN', 'RAW_PM_TABLE', 'RAW_CLOCK', 'RAW_PERF']
    calibration = {}
    raw_traces = {}

    for src in raw_sources:
        print(f"  Collecting {src}...")
        trace = collect_raw_noise(src, 600)  # 30s at 20Hz
        raw_traces[src] = trace
        calibration[src] = {
            'min': float(np.min(trace)),
            'max': float(np.max(trace)),
            'mean': float(np.mean(trace)),
            'std': float(np.std(trace)),
        }
        print(f"    range: [{calibration[src]['min']:.2f}, {calibration[src]['max']:.2f}], "
              f"std={calibration[src]['std']:.4f}")

    # ─── Phase 2: Run reservoir for each condition ───
    print(f"\n[2/4] Running reservoir ({args.n_trials} trials × 8 conditions)...")

    signals, labels = generate_waveforms(args.n_trials, STEPS_PER_TRIAL, SAMPLE_HZ, rng)
    w_in = rng.randn(N_NEURONS)
    w_in /= np.linalg.norm(w_in)

    all_conditions = raw_sources + ['WHITE', 'SYNTHETIC_1F', 'NO_NOISE']
    results = {}

    for cond in all_conditions:
        print(f"\n  === Condition: {cond} ===")

        # Pre-generate noise for controls
        if cond == 'WHITE':
            noise_pool = rng.randn(args.n_trials * STEPS_PER_TRIAL * 2)
        elif cond == 'SYNTHETIC_1F':
            noise_pool = voss_mcartney_1f(args.n_trials * STEPS_PER_TRIAL * 2, rng)
            noise_pool = (noise_pool - noise_pool.mean()) / max(noise_pool.std(), 1e-6)
        else:
            noise_pool = None

        all_features = []
        noise_idx = 0
        all_raw_vgs = []  # track the actual Vg values used

        for ti in range(args.n_trials):
            if (ti + 1) % 20 == 0:
                print(f"    trial {ti+1}/{args.n_trials}")

            trial_features = []
            trial_vgs = []

            for step_i in range(STEPS_PER_TRIAL):
                inp = signals[ti][step_i]

                # Get noise value based on condition
                if cond == 'NO_NOISE':
                    noise_vg_offset = 0.0
                elif cond == 'WHITE':
                    noise_vg_offset = noise_pool[noise_idx] * 0.10
                    noise_idx += 1
                elif cond == 'SYNTHETIC_1F':
                    noise_vg_offset = noise_pool[noise_idx] * 0.10
                    noise_idx += 1
                elif cond in raw_sources:
                    # RAW: read firmware value RIGHT NOW, linear map to Vg offset
                    if cond == 'RAW_POWER':
                        raw_val = read_raw_power()
                    elif cond == 'RAW_SMN':
                        raw_val = np.mean([read_raw_smn(a) for a in SMN_ADDRS])
                    elif cond == 'RAW_PM_TABLE':
                        raw_val = np.mean([read_raw_pm_table(off) for off in [0x10, 0x14, 0x18, 0x1C]])
                    elif cond == 'RAW_CLOCK':
                        raw_val = read_raw_clock()
                    elif cond == 'RAW_PERF':
                        raw_val = read_raw_perf()
                    else:
                        raw_val = 0

                    # Direct linear map: no filtering!
                    cal = calibration[cond]
                    vg_from_raw = raw_to_vg(raw_val, cal['min'], cal['max'])
                    noise_vg_offset = (vg_from_raw - 0.50) * 0.20  # ±0.10 range

                # Set Vg per neuron
                for n in range(N_NEURONS):
                    vg = BASE_VG + ALPHA * inp * w_in[n] + noise_vg_offset
                    vg = np.clip(vg, 0.05, 0.95)
                    send_set_vg(ser, n, vg)
                    if n == 0:
                        trial_vgs.append(vg)

                time.sleep(1.0 / SAMPLE_HZ)

                telem = read_telem(ser)
                if telem:
                    spikes = np.array(telem['delta_spikes'], dtype=float)
                    vmem = np.array(telem['vmem'], dtype=float) / 256.0
                    trial_features.append(np.concatenate([spikes, vmem]))
                else:
                    trial_features.append(np.zeros(N_NEURONS * 2))

            tf = np.array(trial_features)
            pooled = np.concatenate([tf.mean(0), tf.max(0), tf.std(0)])
            all_features.append(pooled)
            all_raw_vgs.extend(trial_vgs)

        X = np.array(all_features)
        y = labels

        # 5-fold stratified CV
        from sklearn.linear_model import RidgeClassifier
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        accs = []
        for train_idx, test_idx in skf.split(X, y):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            clf = RidgeClassifier(alpha=1.0)
            clf.fit(X_train, y[train_idx])
            accs.append(clf.score(X_test, y[test_idx]))

        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))

        # Inter-trial variance
        trial_var = float(np.mean(np.var(X, axis=0)))

        # PSD slope of noise trace (for raw sources)
        psd_slope = 0.0
        acf_lag1 = 0.0
        if cond in raw_sources and len(raw_traces[cond]) > 10:
            psd_slope = compute_psd_slope(raw_traces[cond])
            trace = raw_traces[cond]
            trace_centered = trace - trace.mean()
            if np.std(trace_centered) > 1e-10:
                acf = np.correlate(trace_centered, trace_centered, mode='full')
                acf = acf[len(acf)//2:]
                acf_lag1 = float(acf[1] / (acf[0] + 1e-10))

        print(f"    accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        if cond in raw_sources:
            print(f"    PSD slope: {psd_slope:.3f}, ACF(1): {acf_lag1:.3f}")

        results[cond] = {
            'accuracy_mean': mean_acc,
            'accuracy_std': std_acc,
            'fold_accs': [float(a) for a in accs],
            'trial_variance': trial_var,
            'psd_slope': psd_slope,
            'acf_lag1': acf_lag1,
        }

    # ─── Phase 3: Tests T291-T298 ───
    print("\n" + "=" * 65)
    print("  Tests T291-T298: Raw Firmware Neuromorphic Computation")
    print("=" * 65)

    tests = {}
    white_acc = results['WHITE']['accuracy_mean']
    nonoise_acc = results['NO_NOISE']['accuracy_mean']
    synth_acc = results['SYNTHETIC_1F']['accuracy_mean']

    raw_accs = {s: results[s]['accuracy_mean'] for s in raw_sources}
    best_raw = max(raw_accs, key=raw_accs.get)
    best_raw_acc = raw_accs[best_raw]

    # T291: at least one RAW > WHITE
    any_raw_beats_white = any(raw_accs[s] > white_acc for s in raw_sources)
    t291 = any_raw_beats_white
    tests['T291'] = {'pass': t291, 'raw_accs': raw_accs, 'white_acc': white_acc}
    raw_gt_white = [s for s in raw_sources if raw_accs[s] > white_acc]
    print(f"\n  T291 raw>white:         {'PASS' if t291 else 'FAIL'}  raw_beats_white: {raw_gt_white}")

    # T292: best RAW > NO_NOISE
    t292 = best_raw_acc > nonoise_acc
    tests['T292'] = {'pass': t292, 'best_raw': best_raw, 'best_acc': best_raw_acc, 'nonoise': nonoise_acc}
    print(f"  T292 raw>nonoise:       {'PASS' if t292 else 'FAIL'}  {best_raw}={best_raw_acc:.4f} vs NONOISE={nonoise_acc:.4f}")

    # T293: best RAW > 45%
    t293 = best_raw_acc > 0.45
    tests['T293'] = {'pass': t293, 'best_raw_acc': best_raw_acc}
    print(f"  T293 useful_acc:        {'PASS' if t293 else 'FAIL'}  best_raw={best_raw_acc:.4f}")

    # T294: RAW_POWER PSD slope in biological range
    power_slope = results['RAW_POWER']['psd_slope']
    t294 = -2.5 < power_slope < -0.3
    tests['T294'] = {'pass': t294, 'slope': power_slope}
    print(f"  T294 power_1f:          {'PASS' if t294 else 'FAIL'}  slope={power_slope:.3f} (need [-2.5, -0.3])")

    # T295: RAW_SMN ACF(1) > 0.5
    smn_acf = results['RAW_SMN']['acf_lag1']
    t295 = smn_acf > 0.5
    tests['T295'] = {'pass': t295, 'acf': smn_acf}
    print(f"  T295 smn_memory:        {'PASS' if t295 else 'FAIL'}  ACF(1)={smn_acf:.3f}")

    # T296: at least 2 RAW > 40%
    n_raw_above_40 = sum(1 for s in raw_sources if raw_accs[s] > 0.40)
    t296 = n_raw_above_40 >= 2
    tests['T296'] = {'pass': t296, 'n_above_40': n_raw_above_40}
    print(f"  T296 multiple_useful:   {'PASS' if t296 else 'FAIL'}  {n_raw_above_40} sources > 40%")

    # T297: RAW_POWER > SYNTHETIC_1F
    t297 = raw_accs['RAW_POWER'] > synth_acc
    tests['T297'] = {'pass': t297, 'raw_power': raw_accs['RAW_POWER'], 'synth': synth_acc}
    print(f"  T297 natural>synth:     {'PASS' if t297 else 'FAIL'}  POWER={raw_accs['RAW_POWER']:.4f} vs SYNTH={synth_acc:.4f}")

    # T298: RAW inter-trial variance < WHITE
    raw_vars = [results[s]['trial_variance'] for s in raw_sources]
    mean_raw_var = np.mean(raw_vars)
    white_var = results['WHITE']['trial_variance']
    t298 = mean_raw_var < white_var
    tests['T298'] = {'pass': t298, 'raw_var': float(mean_raw_var), 'white_var': white_var}
    print(f"  T298 structured:        {'PASS' if t298 else 'FAIL'}  raw_var={mean_raw_var:.4f} vs white_var={white_var:.4f}")

    n_pass = sum(1 for t in tests.values() if t['pass'])
    print(f"\n  TOTAL: {n_pass}/8 PASS")

    # Print accuracy summary table
    print("\n  Accuracy Summary:")
    print(f"  {'Source':<20} {'Accuracy':>10} {'PSD slope':>12} {'ACF(1)':>10}")
    print(f"  {'-'*55}")
    for c in all_conditions:
        r = results[c]
        flag = " ***" if c == best_raw else ""
        print(f"  {c:<20} {r['accuracy_mean']:>10.4f} {r['psd_slope']:>12.3f} {r['acf_lag1']:>10.3f}{flag}")

    # ─── Save ───
    out = {
        'experiment': 'z2204_raw_firmware_neuromorphic',
        'n_trials': args.n_trials,
        'conditions': results,
        'calibration': calibration,
        'tests': tests,
        'n_pass': n_pass,
        'n_total': 8,
        'best_raw_source': best_raw,
    }
    out_path = RESULTS / 'z2204_raw_firmware_neuromorphic.json'
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Results: {out_path}")

    # ─── Figure ───
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Bar chart: accuracy by condition
        conds = all_conditions
        accs_m = [results[c]['accuracy_mean'] for c in conds]
        accs_s = [results[c]['accuracy_std'] for c in conds]
        colors = ['#e74c3c' if c in raw_sources else '#3498db' for c in conds]
        axes[0,0].bar(range(len(conds)), accs_m, yerr=accs_s, color=colors, capsize=3)
        axes[0,0].axhline(0.333, ls='--', color='gray', alpha=0.5, label='chance')
        axes[0,0].set_xticks(range(len(conds)))
        axes[0,0].set_xticklabels(conds, rotation=45, ha='right', fontsize=8)
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].set_title('Waveform Classification: Raw Firmware (red) vs Controls (blue)')
        axes[0,0].legend()

        # PSD slopes
        slopes = [results[s]['psd_slope'] for s in raw_sources]
        axes[0,1].bar(raw_sources, slopes, color='#e74c3c')
        axes[0,1].axhline(-1.0, ls='--', color='green', alpha=0.5, label='ideal 1/f')
        axes[0,1].axhline(-0.3, ls=':', color='orange', alpha=0.5)
        axes[0,1].axhline(-2.5, ls=':', color='orange', alpha=0.5)
        axes[0,1].set_ylabel('PSD Slope')
        axes[0,1].set_title('Power Spectral Density Slopes (native, unfiltered)')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].legend()

        # Raw noise traces
        for src in raw_sources[:3]:
            trace = raw_traces[src][:200]
            trace_norm = (trace - trace.mean()) / max(trace.std(), 1e-6)
            axes[1,0].plot(trace_norm, alpha=0.7, label=src, linewidth=0.8)
        axes[1,0].set_xlabel('Sample')
        axes[1,0].set_ylabel('Normalized Value')
        axes[1,0].set_title('Raw Firmware Noise Traces (first 200 samples)')
        axes[1,0].legend(fontsize=8)

        # ACF comparison
        acf_vals = [results[s]['acf_lag1'] for s in raw_sources]
        axes[1,1].bar(raw_sources, acf_vals, color='#e74c3c')
        axes[1,1].axhline(0.5, ls='--', color='green', alpha=0.5, label='memory threshold')
        axes[1,1].set_ylabel('ACF at lag=1')
        axes[1,1].set_title('Temporal Autocorrelation (inherent memory)')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].legend()

        plt.tight_layout()
        FIGURES.mkdir(parents=True, exist_ok=True)
        fig_path = FIGURES / 'z2204_raw_firmware_neuromorphic.png'
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"  Figure: {fig_path}")
    except Exception as e:
        print(f"  [WARN] Figure failed: {e}")

    ser.close()


if __name__ == '__main__':
    main()
