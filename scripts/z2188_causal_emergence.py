#!/usr/bin/env python3
"""z2188_causal_emergence.py — Causal Emergence in GPU-Noise-Driven FPGA Reservoir

Measures whether the GPU-FPGA system creates macro-level causal structure
that doesn't exist at the micro level.  Compares effective information (EI)
at micro level (individual neurons) vs macro level (coarse-grained neuron
groups).  If EI_macro > EI_micro, the system exhibits causal emergence —
macro descriptions are more causally powerful.

Coarse graining:
  Micro  : 8 binary neuron states  → 256 micro-states
  Pairs  : 4 pairs (0-1,2-3,4-5,6-7), state=sum → 3^4=81 macro-states
  Quads  : 2 quads (0-3,4-7), state=sum → 5^2=25 macro-states
  Whole  : 1 system total spikes   → 9 macro-states

4 conditions:
  FULL     — GPU 1/f noise → FPGA neurons
  WHITE    — white noise → FPGA neurons
  NO_NOISE — deterministic FPGA (beta=0)
  SHUFFLED — temporally shuffled 1/f noise → FPGA neurons

Tests T219-T224:
  T219: EI_macro(pairs, FULL) > EI_micro(FULL)
  T220: EI_macro(quads, FULL) > EI_micro(FULL)
  T221: EI_macro(FULL) > EI_macro(WHITE)
  T222: EI_macro(FULL) > EI_macro(NO_NOISE)
  T223: Emergence ratio = EI_macro/EI_micro > 1.0 for FULL
  T224: Causal emergence index (max EI_macro - EI_micro) > 0.05 bits for FULL

Hardware: AMD gfx1151 GPU + Arty A7 FPGA on /dev/ttyUSB*
"""

import os, sys, json, time, struct, argparse
import numpy as np
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'
FIGURES = BASE / 'figures'

# ─── FPGA Protocol ───
SYNC = 0x55
CMD_SET_VG = 0x01
CMD_READ_TELEM = 0x02
CMD_SET_KILL = 0x03

HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"

# ─── Default Parameters ───
N_NEURONS = 8
DEFAULT_BASE_VG = 0.55
DEFAULT_ALPHA = 0.15
DEFAULT_BETA = 0.10
DEFAULT_SAMPLE_HZ = 20
DEFAULT_N_STEPS = 2000


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ═══════════════════════════════════════════════════════════
# FPGA Communication  (same patterns as z2177)
# ═══════════════════════════════════════════════════════════

def to_q16_16(val: float) -> int:
    return int(val * 65536) & 0xFFFFFFFF


def find_fpga():
    try:
        import serial
    except ImportError:
        return None, None
    for p in ['/dev/ttyUSB1', '/dev/ttyUSB0', '/dev/ttyUSB2']:
        try:
            s = serial.Serial(p, 115200, timeout=0.2)
            time.sleep(0.1)
            return s, p
        except Exception:
            continue
    return None, None


def connect_fpga(ser):
    """Disable kill switch and flush."""
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    # Send kill-switch OFF
    ser.write(bytes([SYNC, CMD_SET_KILL, 0x00]))
    ser.flush()
    time.sleep(0.05)
    ser.reset_input_buffer()


def set_per_neuron_vg(ser, vg_values):
    """Set individual Vg for each of 8 neurons."""
    for nid, vg in enumerate(vg_values[:8]):
        q16 = to_q16_16(max(0.0, min(1.0, vg)))
        payload = bytes([nid & 0x07]) + struct.pack('>I', q16)
        ser.write(bytes([SYNC, CMD_SET_VG]) + payload)
    ser.flush()
    time.sleep(0.005)


def read_telem(ser, timeout=0.15):
    """Read telemetry packet: [0x55][0x02][0x30][48B][CRC8] = 52 bytes."""
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


def crc8(data):
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ 0x07
            else:
                crc <<= 1
            crc &= 0xFF
    return crc


# ═══════════════════════════════════════════════════════════
# Noise Sources
# ═══════════════════════════════════════════════════════════

def read_hwmon_power():
    """Read hwmon power1_average (uW -> W)."""
    try:
        return int(open(HWMON_POWER).read().strip()) / 1e6
    except Exception:
        return None


def collect_power_noise(duration_s=15, sample_hz=50):
    """Collect GPU power rail time series for 1/f noise source."""
    n_samples = int(duration_s * sample_hz)
    interval = 1.0 / sample_hz
    powers = []
    for _ in range(n_samples):
        p = read_hwmon_power()
        if p is not None:
            powers.append(p)
        time.sleep(interval)
    return np.array(powers) if powers else None


def iir_filter_noise(noise_samples, alpha_iir=0.85):
    """Apply IIR low-pass: y[t] = alpha*y[t-1] + (1-alpha)*x[t]."""
    filtered = np.zeros(len(noise_samples))
    filtered[0] = noise_samples[0]
    for t in range(1, len(noise_samples)):
        filtered[t] = alpha_iir * filtered[t - 1] + (1 - alpha_iir) * noise_samples[t]
    std = max(np.std(filtered), 1e-6)
    return filtered / std


class VossMcCartneyFilter:
    """10-octave Voss-McCartney 1/f noise generator."""

    def __init__(self, n_octaves=10):
        self.n_octaves = n_octaves
        self.values = np.zeros(n_octaves)
        self.counters = np.zeros(n_octaves, dtype=int)
        self.step = 0

    def process(self, white_sample):
        x = (white_sample / 127.5) - 1.0
        for k in range(self.n_octaves):
            period = 1 << k
            if self.step % period == 0:
                self.values[k] = x
        self.step += 1
        total = np.sum(self.values) / self.n_octaves
        return np.clip(total, -1.0, 1.0)


# ═══════════════════════════════════════════════════════════
# Effective Information Computation
# ═══════════════════════════════════════════════════════════

def build_tpm(state_sequence, n_states):
    """Build transition probability matrix from state sequence.

    TPM[i, j] = P(X_{t+1}=j | X_t=i) with Laplace smoothing.
    """
    tpm = np.ones((n_states, n_states)) * 1e-10  # Laplace smoothing
    for t in range(len(state_sequence) - 1):
        s_t = state_sequence[t]
        s_t1 = state_sequence[t + 1]
        if 0 <= s_t < n_states and 0 <= s_t1 < n_states:
            tpm[s_t, s_t1] += 1.0
    # Normalize rows
    row_sums = tpm.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    tpm = tpm / row_sums
    return tpm


def compute_ei(tpm):
    """Compute Effective Information from a transition probability matrix.

    EI = I(X_t ; X_{t+1}) under do(X_t ~ Uniform)
       = H(column marginals under uniform) - mean(H per row of TPM)

    Returns EI in bits.
    """
    n_states = tpm.shape[0]

    # Mean row entropy: average conditional entropy H(X_{t+1} | X_t = x)
    row_entropies = np.zeros(n_states)
    for i in range(n_states):
        row = tpm[i]
        row = row[row > 0]
        row_entropies[i] = -np.sum(row * np.log(row))
    mean_row_entropy = np.mean(row_entropies)

    # Column marginals under uniform input
    col_marginals = tpm.mean(axis=0)  # uniform over rows
    col_marginals = col_marginals[col_marginals > 0]
    h_marginals = -np.sum(col_marginals * np.log(col_marginals))

    # EI in nats → bits
    ei_nats = h_marginals - mean_row_entropy
    ei_bits = ei_nats / np.log(2)
    return max(ei_bits, 0.0)


# ═══════════════════════════════════════════════════════════
# Coarse Graining
# ═══════════════════════════════════════════════════════════

def binarize_spikes(spike_counts, median_threshold=None):
    """Binarize spike counts: spike_count > median → 1, else 0."""
    arr = np.array(spike_counts)
    if median_threshold is None:
        median_threshold = np.median(arr[arr > 0]) if np.any(arr > 0) else 0.0
    return (arr > median_threshold).astype(int)


def micro_state(binary_neurons):
    """8 binary neurons → integer in [0, 255]."""
    state = 0
    for i in range(N_NEURONS):
        state |= (binary_neurons[i] & 1) << i
    return state


def pair_state(binary_neurons):
    """4 pairs → state in mixed-radix (radix 3 per pair)."""
    pairs = [
        binary_neurons[0] + binary_neurons[1],
        binary_neurons[2] + binary_neurons[3],
        binary_neurons[4] + binary_neurons[5],
        binary_neurons[6] + binary_neurons[7],
    ]
    state = 0
    for i, p in enumerate(pairs):
        state += int(p) * (3 ** i)
    return state


def quad_state(binary_neurons):
    """2 quads → state in mixed-radix (radix 5 per quad)."""
    q0 = sum(binary_neurons[0:4])
    q1 = sum(binary_neurons[4:8])
    return int(q0) * 5 + int(q1)


def whole_state(binary_neurons):
    """Whole system → total spikes in [0, 8]."""
    return int(sum(binary_neurons))


N_MICRO_STATES = 256   # 2^8
N_PAIR_STATES = 81     # 3^4
N_QUAD_STATES = 25     # 5^2
N_WHOLE_STATES = 9     # 0..8


# ═══════════════════════════════════════════════════════════
# FPGA Data Collection
# ═══════════════════════════════════════════════════════════

def generate_drive_signal(n_steps, sample_hz):
    """Low-amplitude sine at mixed frequencies (0.5 Hz + 1.0 Hz)."""
    t = np.arange(n_steps) / sample_hz
    signal = 0.3 * np.sin(2 * np.pi * 0.5 * t) + 0.2 * np.sin(2 * np.pi * 1.0 * t)
    return signal


def run_condition(ser, condition, n_steps, sample_hz, base_vg, alpha, beta,
                  noise_1f, noise_white, noise_shuffled, drive_signal):
    """Run one condition and collect binary spike states per timestep.

    Returns: (n_steps,) array of delta spike counts per neuron → (n_steps, 8).
    """
    interval = 1.0 / sample_hz
    delta_spikes = np.zeros((n_steps, N_NEURONS))
    prev_counts = None
    power_mean = 11.0

    for t in range(n_steps):
        # Determine noise value for this timestep
        if condition == 'FULL':
            # Use live hwmon noise
            p = read_hwmon_power()
            noise_val = (p - power_mean) / 2.0 if p else 0.0
        elif condition == 'WHITE':
            noise_val = noise_white[t % len(noise_white)] if len(noise_white) > 0 else 0.0
        elif condition == 'SHUFFLED':
            noise_val = noise_shuffled[t % len(noise_shuffled)] if len(noise_shuffled) > 0 else 0.0
        else:  # NO_NOISE
            noise_val = 0.0

        # Compute per-neuron Vg
        vg_values = np.full(N_NEURONS, base_vg)
        vg_values += alpha * drive_signal[t]
        if condition != 'NO_NOISE':
            vg_values += beta * noise_val * np.array([1.0, -0.7, 0.5, -0.3,
                                                       0.8, -0.6, 0.4, -0.9])
        vg_values = np.clip(vg_values, 0.05, 0.95)

        set_per_neuron_vg(ser, vg_values)
        time.sleep(interval * 0.3)

        ser.reset_input_buffer()
        ser.write(bytes([SYNC, CMD_READ_TELEM]))
        ser.flush()
        telem = read_telem(ser, timeout=0.15)

        if telem:
            counts = [n['spike_count'] for n in telem]
            if prev_counts is not None:
                for i in range(N_NEURONS):
                    delta = (counts[i] - prev_counts[i]) & 0xFFFF
                    if delta > 30000:
                        delta = 0
                    delta_spikes[t, i] = delta
            prev_counts = counts[:]

        time.sleep(max(0, interval * 0.5 - 0.01))

        if (t + 1) % 200 == 0:
            print(f"  [{condition}] step {t+1}/{n_steps}")

    return delta_spikes


def simulate_condition(condition, n_steps, sample_hz, base_vg, alpha, beta,
                       noise_1f, noise_white, noise_shuffled, drive_signal):
    """Software LIF simulation fallback when FPGA not connected."""
    dt = 1.0 / sample_hz
    vmem = np.zeros(N_NEURONS)
    delta_spikes = np.zeros((n_steps, N_NEURONS))
    v_thresh = 1.0
    tau_m = 0.02
    w_noise = np.array([1.0, -0.7, 0.5, -0.3, 0.8, -0.6, 0.4, -0.9])

    rng = np.random.default_rng(42 if condition == 'FULL' else
                                 43 if condition == 'WHITE' else
                                 44 if condition == 'SHUFFLED' else 45)

    for t in range(n_steps):
        if condition == 'FULL':
            noise_val = noise_1f[t % len(noise_1f)] if len(noise_1f) > 0 else rng.standard_normal() * 0.3
        elif condition == 'WHITE':
            noise_val = noise_white[t % len(noise_white)] if len(noise_white) > 0 else rng.standard_normal()
        elif condition == 'SHUFFLED':
            noise_val = noise_shuffled[t % len(noise_shuffled)] if len(noise_shuffled) > 0 else rng.standard_normal() * 0.3
        else:
            noise_val = 0.0

        vg = np.full(N_NEURONS, base_vg)
        vg += alpha * drive_signal[t]
        if condition != 'NO_NOISE':
            vg += beta * noise_val * w_noise
        vg = np.clip(vg, 0.05, 0.95)

        I_in = vg * 5.0
        dvdt = (-vmem + I_in) / tau_m
        vmem += dvdt * dt

        for i in range(N_NEURONS):
            if vmem[i] >= v_thresh:
                delta_spikes[t, i] = 1
                vmem[i] = 0.0

    return delta_spikes


# ═══════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════

def analyze_condition(delta_spikes):
    """Compute EI at all coarse-graining levels for one condition.

    Returns dict with ei_micro, ei_pairs, ei_quads, ei_whole (all in bits).
    """
    n_steps = delta_spikes.shape[0]

    # Binarize: spike > median → 1
    binary = np.zeros_like(delta_spikes, dtype=int)
    for i in range(N_NEURONS):
        col = delta_spikes[:, i]
        med = np.median(col[col > 0]) if np.any(col > 0) else 0.0
        binary[:, i] = (col > med).astype(int)

    # Build state sequences at each level
    micro_seq = np.array([micro_state(binary[t]) for t in range(n_steps)])
    pair_seq = np.array([pair_state(binary[t]) for t in range(n_steps)])
    quad_seq = np.array([quad_state(binary[t]) for t in range(n_steps)])
    whole_seq = np.array([whole_state(binary[t]) for t in range(n_steps)])

    # Build TPMs and compute EI
    tpm_micro = build_tpm(micro_seq, N_MICRO_STATES)
    tpm_pairs = build_tpm(pair_seq, N_PAIR_STATES)
    tpm_quads = build_tpm(quad_seq, N_QUAD_STATES)
    tpm_whole = build_tpm(whole_seq, N_WHOLE_STATES)

    ei_micro = compute_ei(tpm_micro)
    ei_pairs = compute_ei(tpm_pairs)
    ei_quads = compute_ei(tpm_quads)
    ei_whole = compute_ei(tpm_whole)

    # State occupancy stats
    n_micro_visited = len(np.unique(micro_seq))
    n_pair_visited = len(np.unique(pair_seq))
    n_quad_visited = len(np.unique(quad_seq))
    n_whole_visited = len(np.unique(whole_seq))

    return {
        'ei_micro': float(ei_micro),
        'ei_pairs': float(ei_pairs),
        'ei_quads': float(ei_quads),
        'ei_whole': float(ei_whole),
        'states_visited': {
            'micro': int(n_micro_visited),
            'pairs': int(n_pair_visited),
            'quads': int(n_quad_visited),
            'whole': int(n_whole_visited),
        },
        'total_spikes': float(np.sum(delta_spikes)),
        'mean_firing_rate': float(np.mean(delta_spikes.sum(axis=1))),
    }


# ═══════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════

def make_figure(all_results, tests):
    """Create 2x2 figure: EI bars, emergence ratios, TPM heatmap, test summary."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping figure")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('z2188: Causal Emergence in GPU-FPGA System', fontsize=14, fontweight='bold')

    conditions = ['FULL', 'WHITE', 'NO_NOISE', 'SHUFFLED']
    levels = ['ei_micro', 'ei_pairs', 'ei_quads', 'ei_whole']
    level_labels = ['Micro (8 neurons)', 'Pairs (4 groups)', 'Quads (2 groups)', 'Whole (1 group)']
    cond_colors = {'FULL': '#2196F3', 'WHITE': '#FF9800', 'NO_NOISE': '#9E9E9E', 'SHUFFLED': '#4CAF50'}

    # ─── Panel 1: EI at each level per condition ───
    ax = axes[0, 0]
    x = np.arange(len(levels))
    width = 0.18
    for ci, cond in enumerate(conditions):
        if cond in all_results:
            vals = [all_results[cond].get(lv, 0) for lv in levels]
            ax.bar(x + ci * width, vals, width, label=cond, color=cond_colors[cond])
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(level_labels, fontsize=8, rotation=15)
    ax.set_ylabel('Effective Information (bits)')
    ax.set_title('EI at Each Coarse-Graining Level')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # ─── Panel 2: Emergence ratios ───
    ax = axes[0, 1]
    for ci, cond in enumerate(conditions):
        if cond in all_results:
            r = all_results[cond]
            ei_micro = r.get('ei_micro', 1e-10)
            if ei_micro < 1e-10:
                ei_micro = 1e-10
            ratios = [
                r.get('ei_pairs', 0) / ei_micro,
                r.get('ei_quads', 0) / ei_micro,
                r.get('ei_whole', 0) / ei_micro,
            ]
            ax.bar(np.arange(3) + ci * width, ratios, width, label=cond, color=cond_colors[cond])
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Emergence threshold')
    ax.set_xticks(np.arange(3) + 1.5 * width)
    ax.set_xticklabels(['Pairs/Micro', 'Quads/Micro', 'Whole/Micro'], fontsize=9)
    ax.set_ylabel('EI Ratio (macro/micro)')
    ax.set_title('Emergence Ratios')
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

    # ─── Panel 3: TPM heatmap for FULL condition at pair level ───
    ax = axes[1, 0]
    if 'FULL' in all_results and 'tpm_pairs' in all_results['FULL']:
        tpm = np.array(all_results['FULL']['tpm_pairs'])
        im = ax.imshow(tpm, cmap='viridis', aspect='auto', interpolation='nearest')
        ax.set_title('TPM (Pair Level, FULL condition)')
        ax.set_xlabel('State t+1')
        ax.set_ylabel('State t')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.text(0.5, 0.5, 'TPM data\nnot available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title('TPM (Pair Level, FULL condition)')

    # ─── Panel 4: Test summary ───
    ax = axes[1, 1]
    ax.axis('off')
    test_lines = []
    for t in tests:
        mark = 'PASS' if t['pass'] else 'FAIL'
        color = 'green' if t['pass'] else 'red'
        test_lines.append(f"{t['name']}: {mark}  ({t['detail']})")
    test_text = '\n'.join(test_lines)

    n_pass = sum(1 for t in tests if t['pass'])
    n_total = len(tests)
    header = f"Tests: {n_pass}/{n_total} PASS\n{'='*40}\n"

    ax.text(0.05, 0.95, header + test_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('Test Results (T219-T224)')

    plt.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES / 'z2188_causal_emergence.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Figure saved: {fig_path}")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='z2188 Causal Emergence')
    parser.add_argument('--n-steps', type=int, default=DEFAULT_N_STEPS)
    parser.add_argument('--sample-hz', type=int, default=DEFAULT_SAMPLE_HZ)
    parser.add_argument('--base-vg', type=float, default=DEFAULT_BASE_VG)
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA)
    parser.add_argument('--beta', type=float, default=DEFAULT_BETA)
    args = parser.parse_args()

    n_steps = args.n_steps
    sample_hz = args.sample_hz
    base_vg = args.base_vg
    alpha = args.alpha
    beta = args.beta

    print(f"=== z2188 Causal Emergence ===")
    print(f"n_steps={n_steps}, sample_hz={sample_hz}, base_vg={base_vg}, alpha={alpha}, beta={beta}")

    # ─── Prepare noise sources ───
    print("[1/5] Collecting GPU power noise for 1/f source...")
    noise_duration = max(20, n_steps / sample_hz * 1.5)
    raw_noise = collect_power_noise(duration_s=noise_duration, sample_hz=50)
    if raw_noise is not None and len(raw_noise) > 10:
        noise_1f = iir_filter_noise(raw_noise, alpha_iir=0.85)
        print(f"  Collected {len(raw_noise)} power samples → {len(noise_1f)} filtered")
    else:
        print("  [WARN] hwmon power not available, generating synthetic 1/f noise")
        rng = np.random.default_rng(99)
        raw_synth = rng.standard_normal(n_steps * 2)
        vmf = VossMcCartneyFilter()
        noise_1f = np.array([vmf.process((s * 50 + 127.5)) for s in raw_synth])

    # White noise
    rng = np.random.default_rng(123)
    noise_white = rng.standard_normal(n_steps * 2)

    # Shuffled 1/f noise (same distribution, destroyed temporal structure)
    noise_shuffled = noise_1f.copy()
    rng2 = np.random.default_rng(456)
    rng2.shuffle(noise_shuffled)

    # Drive signal
    drive_signal = generate_drive_signal(n_steps, sample_hz)

    # ─── Try FPGA connection ───
    print("[2/5] Connecting to FPGA...")
    ser, port = find_fpga()
    use_fpga = ser is not None
    if use_fpga:
        print(f"  FPGA found on {port}")
        connect_fpga(ser)
    else:
        print("  [WARN] No FPGA found, using software LIF simulation")

    # ─── Run 4 conditions ───
    conditions = ['FULL', 'WHITE', 'NO_NOISE', 'SHUFFLED']
    all_spikes = {}

    print("[3/5] Running conditions...")
    for cond in conditions:
        print(f"  Running condition: {cond}")
        t0 = time.monotonic()
        if use_fpga:
            spikes = run_condition(ser, cond, n_steps, sample_hz, base_vg, alpha, beta,
                                   noise_1f, noise_white, noise_shuffled, drive_signal)
        else:
            spikes = simulate_condition(cond, n_steps, sample_hz, base_vg, alpha, beta,
                                         noise_1f, noise_white, noise_shuffled, drive_signal)
        elapsed = time.monotonic() - t0
        all_spikes[cond] = spikes
        total = np.sum(spikes)
        print(f"    Done in {elapsed:.1f}s, total spikes: {total:.0f}")

    if ser:
        ser.close()

    # ─── Analyze ───
    print("[4/5] Computing effective information...")
    all_results = {}
    for cond in conditions:
        r = analyze_condition(all_spikes[cond])
        all_results[cond] = r
        print(f"  {cond}: EI_micro={r['ei_micro']:.4f}  EI_pairs={r['ei_pairs']:.4f}  "
              f"EI_quads={r['ei_quads']:.4f}  EI_whole={r['ei_whole']:.4f} bits")

    # Store TPM for FULL condition pair level (for figure)
    binary_full = np.zeros_like(all_spikes['FULL'], dtype=int)
    for i in range(N_NEURONS):
        col = all_spikes['FULL'][:, i]
        med = np.median(col[col > 0]) if np.any(col > 0) else 0.0
        binary_full[:, i] = (col > med).astype(int)
    pair_seq_full = np.array([pair_state(binary_full[t]) for t in range(n_steps)])
    tpm_pairs_full = build_tpm(pair_seq_full, N_PAIR_STATES)
    all_results['FULL']['tpm_pairs'] = tpm_pairs_full.tolist()

    # ─── Tests T219-T224 ───
    print("[5/5] Evaluating tests T219-T224...")
    full = all_results['FULL']
    white = all_results.get('WHITE', {})
    no_noise = all_results.get('NO_NOISE', {})

    ei_micro_full = full['ei_micro']
    ei_pairs_full = full['ei_pairs']
    ei_quads_full = full['ei_quads']
    ei_whole_full = full['ei_whole']

    # Best macro EI for FULL
    best_macro_full = max(ei_pairs_full, ei_quads_full, ei_whole_full)
    emergence_ratio = best_macro_full / ei_micro_full if ei_micro_full > 1e-10 else 0.0
    causal_emergence_index = best_macro_full - ei_micro_full

    tests = []

    # T219: EI_macro(pairs, FULL) > EI_micro(FULL)
    t219 = ei_pairs_full > ei_micro_full
    tests.append({
        'name': 'T219',
        'pass': t219,
        'detail': f'EI_pairs={ei_pairs_full:.4f} > EI_micro={ei_micro_full:.4f}',
    })

    # T220: EI_macro(quads, FULL) > EI_micro(FULL)
    t220 = ei_quads_full > ei_micro_full
    tests.append({
        'name': 'T220',
        'pass': t220,
        'detail': f'EI_quads={ei_quads_full:.4f} > EI_micro={ei_micro_full:.4f}',
    })

    # T221: EI_macro(FULL) > EI_macro(WHITE) — using best macro for each
    best_macro_white = max(white.get('ei_pairs', 0), white.get('ei_quads', 0),
                           white.get('ei_whole', 0))
    t221 = best_macro_full > best_macro_white
    tests.append({
        'name': 'T221',
        'pass': t221,
        'detail': f'EI_macro_FULL={best_macro_full:.4f} > EI_macro_WHITE={best_macro_white:.4f}',
    })

    # T222: EI_macro(FULL) > EI_macro(NO_NOISE)
    best_macro_nonoise = max(no_noise.get('ei_pairs', 0), no_noise.get('ei_quads', 0),
                              no_noise.get('ei_whole', 0))
    t222 = best_macro_full > best_macro_nonoise
    tests.append({
        'name': 'T222',
        'pass': t222,
        'detail': f'EI_macro_FULL={best_macro_full:.4f} > EI_macro_NONOISE={best_macro_nonoise:.4f}',
    })

    # T223: Emergence ratio > 1.0
    t223 = emergence_ratio > 1.0
    tests.append({
        'name': 'T223',
        'pass': t223,
        'detail': f'ratio={emergence_ratio:.4f} > 1.0',
    })

    # T224: Causal emergence index > 0.05 bits
    t224 = causal_emergence_index > 0.05
    tests.append({
        'name': 'T224',
        'pass': t224,
        'detail': f'CEI={causal_emergence_index:.4f} > 0.05 bits',
    })

    n_pass = sum(1 for t in tests if t['pass'])
    n_total = len(tests)
    print(f"\n{'='*50}")
    for t in tests:
        mark = 'PASS' if t['pass'] else 'FAIL'
        print(f"  {t['name']}: {mark}  {t['detail']}")
    print(f"{'='*50}")
    print(f"  TOTAL: {n_pass}/{n_total} PASS")

    # ─── Save results ───
    RESULTS.mkdir(parents=True, exist_ok=True)
    result_data = {
        'experiment': 'z2188_causal_emergence',
        'params': {
            'n_steps': n_steps,
            'sample_hz': sample_hz,
            'base_vg': base_vg,
            'alpha': alpha,
            'beta': beta,
            'n_neurons': N_NEURONS,
            'use_fpga': use_fpga,
        },
        'conditions': {},
        'emergence_summary': {
            'emergence_ratio': float(emergence_ratio),
            'causal_emergence_index': float(causal_emergence_index),
            'best_macro_level': (
                'pairs' if best_macro_full == ei_pairs_full else
                'quads' if best_macro_full == ei_quads_full else 'whole'
            ),
        },
        'tests': tests,
        'pass_count': n_pass,
        'total_tests': n_total,
    }

    # Store per-condition results (without bulky TPM)
    for cond in conditions:
        r = all_results[cond].copy()
        r.pop('tpm_pairs', None)  # Don't store full TPM in JSON
        result_data['conditions'][cond] = r

    result_path = RESULTS / 'z2188_causal_emergence.json'
    with open(result_path, 'w') as f:
        json.dump(result_data, f, indent=2, cls=NpEncoder)
    print(f"\n[INFO] Results saved: {result_path}")

    # ─── Figure ───
    make_figure(all_results, tests)

    return n_pass, n_total


if __name__ == '__main__':
    main()
