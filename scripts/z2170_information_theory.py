#!/usr/bin/env python3
"""z2170_information_theory.py — Information-Theoretic Analysis of GPU→FPGA Coupling

Measures causal and informational relationships between GPU noise sources and
FPGA LIF neuron spike trains using information-theoretic metrics computed with
numpy (no external info-theory libraries).

Conditions:
  FULL:     GPU 1/f noise driving via hwmon power + IIR filter
  WHITE:    Random white noise driving
  NO_NOISE: Deterministic (no noise modulation)
  SHUFFLED: Same 1/f noise values but temporally shuffled

Per condition: 5000 timesteps at 20 Hz (250 seconds).

Metrics:
  Transfer Entropy (TE): TE(X→Y) = H(Y_t|Y_past) - H(Y_t|Y_past, X_past)
  Mutual Information (MI): Between binned noise and binned spike counts
  Active Information Storage (AIS): MI(Y_past_k=3, Y_t)
  Phi approximation: MI(full 8-neuron) vs sum of 2 partitions (4+4)

Tests (T109-T114):
  T109: TE(FULL) > TE(WHITE)
  T110: TE(FULL) > TE(SHUFFLED)
  T111: MI(FULL) > MI(NO_NOISE)
  T112: AIS(FULL) > AIS(WHITE)
  T113: Phi(FULL) > Phi(NO_NOISE)
  T114: TE(NO_NOISE) < 0.01 bits

Hardware: AMD gfx1151 GPU + Arty A7 FPGA on /dev/ttyUSB{0,1}
"""

import os, sys, json, time, struct, argparse
import numpy as np
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'
FIGURES = RESULTS / 'FEEL_paper_update' / 'FEEL__Functionally_Embodied_Emergent_Learning__13_-5' / 'figures'

# ─── FPGA Protocol ───
SYNC = 0x55
CMD_SET_VG = 0x01
CMD_READ_TELEM = 0x02
CMD_SET_KILL = 0x03

HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"

# ─── Parameters ───
BASE_VG = 0.58       # near BVpar cliff
ALPHA_NOISE = 0.25   # noise coupling strength
N_NEURONS = 8
SAMPLE_HZ = 20
N_STEPS = 5000       # per condition (250 seconds at 20 Hz)
IIR_ALPHA = 0.85     # IIR filter coefficient for temporal memory

# ─── Info-theory parameters ───
N_BINS_NOISE = 8     # bins for noise discretisation
N_BINS_SPIKE = 8     # bins for spike count discretisation
TE_HISTORY = 3       # past steps for transfer entropy
AIS_HISTORY = 3      # past steps for active info storage


# ═══════════════════════════════════════════════════════════
# FPGA Communication
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


def reconnect_fpga(port):
    import serial
    try:
        ser = serial.Serial(port, 115200, timeout=0.2)
        time.sleep(0.1)
        ser.write(bytes([SYNC, CMD_SET_KILL, 0x00]))
        ser.flush()
        time.sleep(0.1)
        return ser
    except Exception:
        return None


def set_per_neuron_vg(ser, vg_values):
    for nid, vg in enumerate(vg_values[:8]):
        q16 = to_q16_16(max(0.0, min(1.0, vg)))
        payload = bytes([nid & 0x07]) + struct.pack('>I', q16)
        ser.write(bytes([SYNC, CMD_SET_VG]) + payload)
    ser.flush()
    time.sleep(0.005)


def read_telem(ser, timeout=0.15):
    """Read telemetry: [0x55][0x02][0x30][48B][CRC8] = 52 bytes."""
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


def safe_fpga_step(ser, port, vg_values, interval):
    import serial as serial_mod
    try:
        set_per_neuron_vg(ser, vg_values)
        time.sleep(interval * 0.3)
        ser.reset_input_buffer()
        ser.write(bytes([SYNC, CMD_READ_TELEM]))
        ser.flush()
        telem = read_telem(ser, timeout=0.15)
        return ser, telem
    except (serial_mod.SerialException, OSError) as e:
        print(f"    [!] Serial error: {e}, reconnecting...")
        try:
            ser.close()
        except Exception:
            pass
        time.sleep(0.5)
        new_ser = reconnect_fpga(port)
        if new_ser is None:
            print("    [!] Reconnection failed")
            return None, None
        print("    [!] Reconnected successfully")
        return new_ser, None


# ═══════════════════════════════════════════════════════════
# Noise Sources
# ═══════════════════════════════════════════════════════════

def read_hwmon_power():
    """Read hwmon power1_average (uW -> W). Rich 1/f dynamics ~11W +/- 1.5W."""
    try:
        return int(open(HWMON_POWER).read().strip()) / 1e6
    except Exception:
        return None


def normalize_noise(samples):
    arr = np.array(samples, dtype=float)
    if len(arr) == 0:
        return arr
    mu = arr.mean()
    std = max(arr.std(), 1e-6)
    return (arr - mu) / std


def iir_filter_noise(noise_samples, alpha_iir=IIR_ALPHA):
    """IIR low-pass: y[t] = alpha*y[t-1] + (1-alpha)*x[t]. Creates temporal memory."""
    if len(noise_samples) == 0:
        return noise_samples
    filtered = np.zeros(len(noise_samples))
    filtered[0] = noise_samples[0]
    for t in range(1, len(noise_samples)):
        filtered[t] = alpha_iir * filtered[t-1] + (1 - alpha_iir) * noise_samples[t]
    std = max(np.std(filtered), 1e-6)
    return filtered / std


def generate_synthetic_1f(n_samples, rng):
    """Voss-McCartney 1/f generator."""
    noise = np.zeros(n_samples)
    n_octaves = 8
    octaves = np.zeros(n_octaves)
    for i in range(n_samples):
        for j in range(n_octaves):
            if i % (1 << j) == 0:
                octaves[j] = rng.standard_normal()
        noise[i] = octaves.sum()
    return normalize_noise(noise)


# ═══════════════════════════════════════════════════════════
# LIF Simulator (fallback when no FPGA)
# ═══════════════════════════════════════════════════════════

class LIFSimulator:
    """Software LIF bank matching FPGA neuron behaviour."""

    def __init__(self, n_neurons=8, seed=42):
        self.n = n_neurons
        self.rng = np.random.default_rng(seed)
        # Per-neuron heterogeneous thresholds (mimicking FPGA process variation)
        self.v_thresh = 0.55 + 0.1 * self.rng.uniform(-1, 1, n_neurons)
        self.v_rest = 0.0
        self.v_reset = 0.0
        self.tau_m = 0.02  # 20 ms membrane time constant
        self.vmem = np.zeros(n_neurons)
        self.spike_counts = np.zeros(n_neurons, dtype=int)
        self.dt = 1.0 / SAMPLE_HZ

    def step(self, vg_values):
        """One timestep: integrate-and-fire with Vg as input current."""
        vg = np.array(vg_values[:self.n])
        # Leaky integration
        dv = (-self.vmem + vg) / self.tau_m * self.dt
        self.vmem += dv
        # Add small noise for realism
        self.vmem += 0.005 * self.rng.standard_normal(self.n)
        # Fire
        fired = self.vmem >= self.v_thresh
        spikes = fired.astype(int)
        self.spike_counts += spikes
        self.vmem[fired] = self.v_reset
        neurons = []
        for i in range(self.n):
            neurons.append({
                'spike_count': int(self.spike_counts[i]),
                'vmem': float(self.vmem[i])
            })
        return neurons


# ═══════════════════════════════════════════════════════════
# Information-Theoretic Metrics (pure numpy)
# ═══════════════════════════════════════════════════════════

def _entropy(counts):
    """Shannon entropy from histogram counts (in bits)."""
    p = counts / counts.sum()
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def _joint_entropy(x_bins, y_bins, n_x_bins, n_y_bins):
    """Joint entropy H(X,Y) from bin indices."""
    joint = np.zeros((n_x_bins, n_y_bins), dtype=int)
    for xi, yi in zip(x_bins, y_bins):
        joint[xi, yi] += 1
    p = joint.flatten() / joint.sum()
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def _conditional_entropy(target_bins, cond_bins, n_target, n_cond):
    """H(target|cond) = H(target, cond) - H(cond)."""
    h_joint = _joint_entropy(target_bins, cond_bins, n_target, n_cond)
    h_cond, _ = np.histogram(cond_bins, bins=n_cond, range=(0, n_cond))
    return h_joint - _entropy(h_cond)


def _bin_series(x, n_bins):
    """Discretise continuous series into bin indices using uniform quantiles."""
    if len(x) == 0:
        return np.array([], dtype=int)
    # Use quantile-based binning for robustness
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(x, percentiles)
    edges[-1] += 1e-10  # include max
    # Make edges strictly increasing
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = edges[i-1] + 1e-10
    bins = np.digitize(x, edges[1:-1])  # 0 to n_bins-1
    return np.clip(bins, 0, n_bins - 1)


def _encode_history(bins, k):
    """Encode k-step history into single integer index.
    history[t] encodes (bins[t-k], bins[t-k+1], ..., bins[t-1]).
    Returns (history_indices, n_history_bins, valid_start_index).
    """
    n = len(bins)
    n_history_bins = N_BINS_SPIKE ** k
    history = np.zeros(n, dtype=int)
    for t in range(k, n):
        idx = 0
        for j in range(k):
            idx = idx * N_BINS_SPIKE + bins[t - k + j]
        history[t] = idx
    return history, n_history_bins, k


def compute_transfer_entropy(noise_series, spike_series, k=TE_HISTORY,
                              n_bins_noise=N_BINS_NOISE, n_bins_spike=N_BINS_SPIKE):
    """Transfer entropy TE(X→Y) where X=noise, Y=spikes.

    TE(X→Y) = H(Y_t | Y_past) - H(Y_t | Y_past, X_past)

    Uses binned histograms for estimation.
    """
    # Bin both series
    x_bins = _bin_series(noise_series, n_bins_noise)
    y_bins = _bin_series(spike_series, n_bins_spike)

    n = min(len(x_bins), len(y_bins))
    x_bins = x_bins[:n]
    y_bins = y_bins[:n]

    if n < k + 2:
        return 0.0

    # Encode histories
    y_hist, n_y_hist, _ = _encode_history(y_bins, k)
    x_hist, n_x_hist, _ = _encode_history(x_bins, k)

    # Joint history (Y_past, X_past) encoded as single index
    joint_hist = y_hist * n_x_hist + x_hist
    n_joint_hist = n_y_hist * n_x_hist

    # Valid range: t >= k
    valid = slice(k, n)
    y_t = y_bins[valid]
    y_h = y_hist[valid]
    x_h = x_hist[valid]
    jh = joint_hist[valid]

    # H(Y_t | Y_past) = H(Y_t, Y_past) - H(Y_past)
    h_yt_ypast = _conditional_entropy(y_t, y_h, n_bins_spike, n_y_hist)

    # H(Y_t | Y_past, X_past) = H(Y_t, (Y_past, X_past)) - H(Y_past, X_past)
    h_yt_joint = _conditional_entropy(y_t, jh, n_bins_spike, n_joint_hist)

    te = max(0.0, h_yt_ypast - h_yt_joint)
    return te


def compute_mutual_information(noise_series, spike_series,
                                n_bins_noise=N_BINS_NOISE, n_bins_spike=N_BINS_SPIKE):
    """MI(X;Y) = H(X) + H(Y) - H(X,Y) using binned histograms."""
    n = min(len(noise_series), len(spike_series))
    if n < 2:
        return 0.0

    x_bins = _bin_series(noise_series[:n], n_bins_noise)
    y_bins = _bin_series(spike_series[:n], n_bins_spike)

    h_x_counts, _ = np.histogram(x_bins, bins=n_bins_noise, range=(0, n_bins_noise))
    h_y_counts, _ = np.histogram(y_bins, bins=n_bins_spike, range=(0, n_bins_spike))
    h_x = _entropy(h_x_counts)
    h_y = _entropy(h_y_counts)
    h_xy = _joint_entropy(x_bins, y_bins, n_bins_noise, n_bins_spike)

    mi = max(0.0, h_x + h_y - h_xy)
    return mi


def compute_active_info_storage(spike_series, k=AIS_HISTORY, n_bins=N_BINS_SPIKE):
    """AIS = MI(Y_past_k; Y_t) = H(Y_t) + H(Y_past_k) - H(Y_t, Y_past_k).

    Measures how much a neuron's past predicts its own future.
    """
    y_bins = _bin_series(spike_series, n_bins)
    n = len(y_bins)
    if n < k + 2:
        return 0.0

    y_hist, n_hist, _ = _encode_history(y_bins, k)

    valid = slice(k, n)
    y_t = y_bins[valid]
    y_h = y_hist[valid]

    h_yt, _ = np.histogram(y_t, bins=n_bins, range=(0, n_bins))
    h_yh, _ = np.histogram(y_h, bins=n_hist, range=(0, n_hist))
    h_yt_val = _entropy(h_yt)
    h_yh_val = _entropy(h_yh)
    h_joint = _joint_entropy(y_t, y_h, n_bins, n_hist)

    ais = max(0.0, h_yt_val + h_yh_val - h_joint)
    return ais


def compute_phi_approx(spike_matrix):
    """Phi approximation: compare MI of full 8-neuron system vs 2 partitions (4+4).

    Phi ~ MI(full_system) - [MI(partition_A) + MI(partition_B)]

    We compute MI between concatenated state histories:
      Full: MI(neurons_0-7_past ; neurons_0-7_present)
      Parts: MI(neurons_0-3_past ; neurons_0-3_present) + MI(neurons_4-7_past ; neurons_4-7_present)

    Spike_matrix shape: (n_steps, 8) — delta spike counts per neuron per step.
    """
    n_steps, n_neurons = spike_matrix.shape
    if n_steps < 10 or n_neurons < 8:
        return 0.0

    k = 1  # single-step history for tractability

    def _system_ais(cols):
        """Compute system-level AIS by encoding multi-neuron state as single index."""
        # Bin each neuron independently
        binned = np.zeros((n_steps, len(cols)), dtype=int)
        n_bins_per = 4  # fewer bins per neuron to keep joint space manageable
        for j, c in enumerate(cols):
            binned[:, j] = _bin_series(spike_matrix[:, c], n_bins_per)

        # Encode multi-neuron state as single index
        n_total_bins = n_bins_per ** len(cols)
        state = np.zeros(n_steps, dtype=int)
        for t in range(n_steps):
            idx = 0
            for j in range(len(cols)):
                idx = idx * n_bins_per + binned[t, j]
            state[t] = idx

        # AIS of the encoded state
        if n_steps < k + 2:
            return 0.0

        past = state[:-k]
        present = state[k:]
        n = len(past)

        # MI(past; present) via joint histogram
        h_past, _ = np.histogram(past, bins=min(n_total_bins, 256), range=(0, n_total_bins))
        h_present, _ = np.histogram(present, bins=min(n_total_bins, 256), range=(0, n_total_bins))
        h_p = _entropy(h_past)
        h_pr = _entropy(h_present)

        # Joint
        joint = np.zeros((min(n_total_bins, 256), min(n_total_bins, 256)), dtype=int)
        n_b = min(n_total_bins, 256)
        for i in range(n):
            pi = past[i] % n_b
            pri = present[i] % n_b
            joint[pi, pri] += 1
        p_j = joint.flatten() / max(joint.sum(), 1)
        p_j = p_j[p_j > 0]
        h_joint = -np.sum(p_j * np.log2(p_j))

        return max(0.0, h_p + h_pr - h_joint)

    # Full system (8 neurons) — use 2 bins per neuron (2^8=256 states)
    mi_full = _system_ais(list(range(8)))

    # Partition A: neurons 0-3, Partition B: neurons 4-7
    mi_a = _system_ais([0, 1, 2, 3])
    mi_b = _system_ais([4, 5, 6, 7])

    phi = max(0.0, mi_full - (mi_a + mi_b))
    return phi


# ═══════════════════════════════════════════════════════════
# Data Collection
# ═══════════════════════════════════════════════════════════

def collect_condition(ser, port, sim, noise_series, condition, n_steps, rng):
    """Collect spike data under a given noise condition.

    Returns:
      noise_applied: (n_steps,) noise values actually applied
      spike_matrix: (n_steps, 8) delta spike counts
      total_spikes: (n_steps,) sum across neurons
    """
    interval = 1.0 / SAMPLE_HZ
    spike_matrix = np.zeros((n_steps, N_NEURONS))
    noise_applied = np.zeros(n_steps)
    prev_counts = None
    use_fpga = (ser is not None)

    print(f"  Collecting {condition} ({n_steps} steps, {n_steps/SAMPLE_HZ:.0f}s)...")

    for t in range(n_steps):
        # Determine noise value
        if condition == 'FULL':
            if use_fpga:
                # Live hwmon read + IIR
                raw = read_hwmon_power()
                if raw is None:
                    raw = 11.0 + 0.5 * rng.standard_normal()
            else:
                raw = noise_series[t % len(noise_series)] if len(noise_series) > 0 else 0.0
            noise_val = raw
        elif condition == 'WHITE':
            noise_val = rng.standard_normal()
        elif condition == 'NO_NOISE':
            noise_val = 0.0
        elif condition == 'SHUFFLED':
            noise_val = noise_series[t % len(noise_series)] if len(noise_series) > 0 else 0.0
        else:
            noise_val = 0.0

        noise_applied[t] = noise_val

        # Compute Vg
        vg = np.full(N_NEURONS, BASE_VG)
        if condition != 'NO_NOISE':
            # Normalise noise_val for coupling
            vg += ALPHA_NOISE * noise_val * 0.01 * np.ones(N_NEURONS)
        vg = np.clip(vg, 0.05, 0.95)

        # Drive FPGA or simulator
        if use_fpga:
            ser, telem = safe_fpga_step(ser, port, vg, interval)
            if ser is None:
                print("    [!] FPGA lost, falling back to simulator")
                use_fpga = False
                sim = LIFSimulator(seed=t)
                telem = sim.step(vg)
            elif telem is None:
                telem = [{'spike_count': 0, 'vmem': 0.0}] * N_NEURONS
        else:
            telem = sim.step(vg)
            time.sleep(max(0, interval - 0.002))

        # Extract delta spikes
        counts = np.array([n['spike_count'] for n in telem])
        if prev_counts is not None:
            delta = counts - prev_counts
            delta[delta < 0] = counts[delta < 0]  # handle counter wrap
            spike_matrix[t] = delta
        prev_counts = counts.copy()

        # Progress
        if (t + 1) % 1000 == 0:
            rate = spike_matrix[max(0, t-99):t+1].sum(axis=1).mean()
            print(f"    step {t+1}/{n_steps}, avg spike rate: {rate:.2f}/step")

    total_spikes = spike_matrix.sum(axis=1)
    return noise_applied, spike_matrix, total_spikes, ser


# ═══════════════════════════════════════════════════════════
# IIR Filtering for Live Noise
# ═══════════════════════════════════════════════════════════

def apply_iir_online(noise_applied, alpha_iir=IIR_ALPHA):
    """Apply IIR filter to collected noise series (post-hoc for FULL condition)."""
    filtered = np.zeros(len(noise_applied))
    filtered[0] = noise_applied[0]
    for t in range(1, len(noise_applied)):
        filtered[t] = alpha_iir * filtered[t-1] + (1 - alpha_iir) * noise_applied[t]
    return filtered


# ═══════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════

def run_experiment(n_steps=N_STEPS, skip_fpga=False):
    rng = np.random.default_rng(2170)

    # ── Find FPGA ──
    ser, port = (None, None) if skip_fpga else find_fpga()
    sim = None
    if ser is not None:
        print(f"[FPGA] Connected on {port}")
        # Disable kill switch
        ser.write(bytes([SYNC, CMD_SET_KILL, 0x00]))
        ser.flush()
        time.sleep(0.1)
    else:
        print("[SIM] No FPGA found, using LIF simulator")
        sim = LIFSimulator(seed=2170)

    # ── Pre-generate noise for SHUFFLED condition ──
    # Collect some 1/f noise samples first (or generate synthetic)
    print("\n=== Pre-generating 1/f noise for SHUFFLED condition ===")
    if ser is not None:
        # Collect real hwmon power samples
        print("  Reading hwmon power samples...")
        power_samples = []
        for i in range(n_steps):
            p = read_hwmon_power()
            if p is not None:
                power_samples.append(p)
            if (i + 1) % 1000 == 0:
                print(f"    {i+1}/{n_steps}")
            time.sleep(1.0 / SAMPLE_HZ)
        noise_1f_raw = np.array(power_samples)
    else:
        noise_1f_raw = generate_synthetic_1f(n_steps, rng)

    # IIR filter the 1/f noise
    noise_1f_filtered = iir_filter_noise(noise_1f_raw, IIR_ALPHA)

    # Shuffled version: same values, random order (destroys temporal structure)
    noise_shuffled = noise_1f_filtered.copy()
    rng.shuffle(noise_shuffled)

    # ── Run conditions ──
    conditions = ['FULL', 'WHITE', 'NO_NOISE', 'SHUFFLED']
    data = {}

    for cond in conditions:
        print(f"\n=== Condition: {cond} ===")

        # Reset simulator state between conditions
        if sim is not None:
            sim = LIFSimulator(seed=2170 + hash(cond) % 10000)

        # Select noise source
        if cond == 'SHUFFLED':
            noise_source = noise_shuffled
        elif cond == 'FULL':
            noise_source = noise_1f_filtered
        else:
            noise_source = noise_1f_filtered  # not used for WHITE/NO_NOISE

        noise_applied, spike_matrix, total_spikes, ser = collect_condition(
            ser, port, sim, noise_source, cond, n_steps, rng
        )

        # For FULL condition, apply IIR to the raw live readings
        if cond == 'FULL':
            noise_filtered = apply_iir_online(noise_applied, IIR_ALPHA)
        elif cond == 'WHITE':
            noise_filtered = noise_applied  # white noise has no IIR
        elif cond == 'SHUFFLED':
            noise_filtered = noise_applied  # already shuffled
        else:
            noise_filtered = noise_applied  # zeros for NO_NOISE

        data[cond] = {
            'noise': noise_filtered,
            'spike_matrix': spike_matrix,
            'total_spikes': total_spikes,
        }

    # ── Compute metrics ──
    print("\n=== Computing Information-Theoretic Metrics ===")
    metrics = {}

    for cond in conditions:
        d = data[cond]
        noise = d['noise']
        total = d['total_spikes']
        sm = d['spike_matrix']

        # TE: average over all 8 neurons (noise → each neuron)
        te_per_neuron = []
        for n_id in range(N_NEURONS):
            te = compute_transfer_entropy(noise, sm[:, n_id])
            te_per_neuron.append(te)
        te_mean = float(np.mean(te_per_neuron))

        # MI: between noise and total spike count
        mi = compute_mutual_information(noise, total)

        # AIS: average over all neurons
        ais_per_neuron = []
        for n_id in range(N_NEURONS):
            ais = compute_active_info_storage(sm[:, n_id])
            ais_per_neuron.append(ais)
        ais_mean = float(np.mean(ais_per_neuron))

        # Phi approximation
        phi = compute_phi_approx(sm)

        metrics[cond] = {
            'TE_mean': te_mean,
            'TE_per_neuron': [float(x) for x in te_per_neuron],
            'MI': float(mi),
            'AIS_mean': ais_mean,
            'AIS_per_neuron': [float(x) for x in ais_per_neuron],
            'Phi': float(phi),
            'spike_rate_mean': float(total.mean()),
            'spike_rate_std': float(total.std()),
        }

        print(f"\n  {cond}:")
        print(f"    TE(mean)  = {te_mean:.6f} bits")
        print(f"    MI        = {mi:.6f} bits")
        print(f"    AIS(mean) = {ais_mean:.6f} bits")
        print(f"    Phi       = {phi:.6f} bits")
        print(f"    spike rate= {total.mean():.2f} +/- {total.std():.2f}")

    # ── Evaluate tests ──
    print("\n=== Test Results ===")
    tests = {}

    # T109: TE(FULL) > TE(WHITE)
    te_full = metrics['FULL']['TE_mean']
    te_white = metrics['WHITE']['TE_mean']
    t109 = te_full > te_white
    tests['T109'] = {
        'description': 'TE(FULL) > TE(WHITE) — 1/f noise has more causal influence',
        'pass': bool(t109),
        'TE_FULL': te_full,
        'TE_WHITE': te_white,
        'ratio': te_full / max(te_white, 1e-10),
    }
    print(f"  T109: TE(FULL)={te_full:.6f} > TE(WHITE)={te_white:.6f} → {'PASS' if t109 else 'FAIL'}")

    # T110: TE(FULL) > TE(SHUFFLED)
    te_shuf = metrics['SHUFFLED']['TE_mean']
    t110 = te_full > te_shuf
    tests['T110'] = {
        'description': 'TE(FULL) > TE(SHUFFLED) — temporal structure matters',
        'pass': bool(t110),
        'TE_FULL': te_full,
        'TE_SHUFFLED': te_shuf,
        'ratio': te_full / max(te_shuf, 1e-10),
    }
    print(f"  T110: TE(FULL)={te_full:.6f} > TE(SHUFFLED)={te_shuf:.6f} → {'PASS' if t110 else 'FAIL'}")

    # T111: MI(FULL) > MI(NO_NOISE)
    mi_full = metrics['FULL']['MI']
    mi_no = metrics['NO_NOISE']['MI']
    t111 = mi_full > mi_no
    tests['T111'] = {
        'description': 'MI(FULL) > MI(NO_NOISE) — noise carries information to spikes',
        'pass': bool(t111),
        'MI_FULL': mi_full,
        'MI_NO_NOISE': mi_no,
        'delta': mi_full - mi_no,
    }
    print(f"  T111: MI(FULL)={mi_full:.6f} > MI(NO_NOISE)={mi_no:.6f} → {'PASS' if t111 else 'FAIL'}")

    # T112: AIS(FULL) > AIS(WHITE)
    ais_full = metrics['FULL']['AIS_mean']
    ais_white = metrics['WHITE']['AIS_mean']
    t112 = ais_full > ais_white
    tests['T112'] = {
        'description': 'AIS(FULL) > AIS(WHITE) — 1/f creates more temporal memory',
        'pass': bool(t112),
        'AIS_FULL': ais_full,
        'AIS_WHITE': ais_white,
        'ratio': ais_full / max(ais_white, 1e-10),
    }
    print(f"  T112: AIS(FULL)={ais_full:.6f} > AIS(WHITE)={ais_white:.6f} → {'PASS' if t112 else 'FAIL'}")

    # T113: Phi(FULL) > Phi(NO_NOISE)
    phi_full = metrics['FULL']['Phi']
    phi_no = metrics['NO_NOISE']['Phi']
    t113 = phi_full > phi_no
    tests['T113'] = {
        'description': 'Phi(FULL) > Phi(NO_NOISE) — noise increases integration',
        'pass': bool(t113),
        'Phi_FULL': phi_full,
        'Phi_NO_NOISE': phi_no,
        'delta': phi_full - phi_no,
    }
    print(f"  T113: Phi(FULL)={phi_full:.6f} > Phi(NO_NOISE)={phi_no:.6f} → {'PASS' if t113 else 'FAIL'}")

    # T114: TE(NO_NOISE) < 0.01 bits
    te_no = metrics['NO_NOISE']['TE_mean']
    t114 = te_no < 0.01
    tests['T114'] = {
        'description': 'TE(NO_NOISE) < 0.01 bits — no noise → no transfer',
        'pass': bool(t114),
        'TE_NO_NOISE': te_no,
        'threshold': 0.01,
    }
    print(f"  T114: TE(NO_NOISE)={te_no:.6f} < 0.01 → {'PASS' if t114 else 'FAIL'}")

    n_pass = sum(1 for t in tests.values() if t['pass'])
    n_total = len(tests)
    print(f"\n  === TOTAL: {n_pass}/{n_total} PASS ===")

    # ── Build results ──
    result = {
        'experiment': 'z2170_information_theory',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'hardware': 'FPGA' if port else 'SIMULATED',
        'fpga_port': port,
        'parameters': {
            'n_steps': n_steps,
            'sample_hz': SAMPLE_HZ,
            'base_vg': BASE_VG,
            'alpha_noise': ALPHA_NOISE,
            'iir_alpha': IIR_ALPHA,
            'n_neurons': N_NEURONS,
            'n_bins_noise': N_BINS_NOISE,
            'n_bins_spike': N_BINS_SPIKE,
            'te_history': TE_HISTORY,
            'ais_history': AIS_HISTORY,
            'conditions': conditions,
        },
        'metrics': metrics,
        'tests': tests,
        'summary': {
            'pass_count': n_pass,
            'total_tests': n_total,
            'pass_rate': n_pass / n_total,
        }
    }

    # ── Save results ──
    RESULTS.mkdir(parents=True, exist_ok=True)
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, (np.bool_,)): return bool(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

    out_path = RESULTS / 'z2170_information_theory.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, cls=NpEncoder)
    print(f"\n[SAVED] {out_path}")

    # ── Generate figure ──
    try:
        generate_figure(metrics, tests, data)
    except Exception as e:
        print(f"[WARN] Figure generation failed: {e}")

    # ── Cleanup FPGA ──
    if ser is not None:
        try:
            ser.write(bytes([SYNC, CMD_SET_KILL, 0x00]))
            ser.flush()
            ser.close()
        except Exception:
            pass

    return result


# ═══════════════════════════════════════════════════════════
# Visualisation
# ═══════════════════════════════════════════════════════════

def generate_figure(metrics, tests, data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('z2170: Information-Theoretic Analysis — GPU→FPGA Coupling',
                 fontsize=14, fontweight='bold')

    conditions = ['FULL', 'WHITE', 'NO_NOISE', 'SHUFFLED']
    cond_colors = {'FULL': '#2196F3', 'WHITE': '#FF9800', 'NO_NOISE': '#9E9E9E', 'SHUFFLED': '#E91E63'}
    bar_x = np.arange(len(conditions))
    bar_w = 0.6

    # (0,0) Transfer Entropy
    ax = axes[0, 0]
    te_vals = [metrics[c]['TE_mean'] for c in conditions]
    bars = ax.bar(bar_x, te_vals, bar_w, color=[cond_colors[c] for c in conditions], edgecolor='black')
    ax.set_xticks(bar_x)
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel('Transfer Entropy (bits)')
    ax.set_title('Transfer Entropy: Noise → Spikes')
    t109_str = 'PASS' if tests['T109']['pass'] else 'FAIL'
    t110_str = 'PASS' if tests['T110']['pass'] else 'FAIL'
    ax.text(0.02, 0.95, f'T109(FULL>WHITE): {t109_str}\nT110(FULL>SHUF): {t110_str}',
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # (0,1) Mutual Information
    ax = axes[0, 1]
    mi_vals = [metrics[c]['MI'] for c in conditions]
    ax.bar(bar_x, mi_vals, bar_w, color=[cond_colors[c] for c in conditions], edgecolor='black')
    ax.set_xticks(bar_x)
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel('Mutual Information (bits)')
    ax.set_title('MI: Noise ↔ Spike Counts')
    t111_str = 'PASS' if tests['T111']['pass'] else 'FAIL'
    ax.text(0.02, 0.95, f'T111(FULL>NO_NOISE): {t111_str}',
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # (0,2) Active Information Storage
    ax = axes[0, 2]
    ais_vals = [metrics[c]['AIS_mean'] for c in conditions]
    ax.bar(bar_x, ais_vals, bar_w, color=[cond_colors[c] for c in conditions], edgecolor='black')
    ax.set_xticks(bar_x)
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel('AIS (bits)')
    ax.set_title('Active Information Storage')
    t112_str = 'PASS' if tests['T112']['pass'] else 'FAIL'
    ax.text(0.02, 0.95, f'T112(FULL>WHITE): {t112_str}',
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # (1,0) Phi Approximation
    ax = axes[1, 0]
    phi_vals = [metrics[c]['Phi'] for c in conditions]
    ax.bar(bar_x, phi_vals, bar_w, color=[cond_colors[c] for c in conditions], edgecolor='black')
    ax.set_xticks(bar_x)
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel('Phi (bits)')
    ax.set_title('Integrated Information (Phi approx)')
    t113_str = 'PASS' if tests['T113']['pass'] else 'FAIL'
    ax.text(0.02, 0.95, f'T113(FULL>NO_NOISE): {t113_str}',
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # (1,1) Per-neuron TE heatmap
    ax = axes[1, 1]
    te_matrix = np.array([metrics[c]['TE_per_neuron'] for c in conditions])
    im = ax.imshow(te_matrix, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(N_NEURONS))
    ax.set_xticklabels([f'N{i}' for i in range(N_NEURONS)])
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(conditions, fontsize=9)
    ax.set_xlabel('Neuron')
    ax.set_title('Per-Neuron Transfer Entropy')
    fig.colorbar(im, ax=ax, label='TE (bits)')

    # (1,2) Test summary
    ax = axes[1, 2]
    ax.axis('off')
    test_names = ['T109', 'T110', 'T111', 'T112', 'T113', 'T114']
    summary_text = "TEST RESULTS\n" + "=" * 40 + "\n\n"
    for tn in test_names:
        t = tests[tn]
        status = 'PASS' if t['pass'] else 'FAIL'
        marker = '[+]' if t['pass'] else '[-]'
        summary_text += f"{marker} {tn}: {status}\n    {t['description']}\n\n"
    n_pass = sum(1 for t in tests.values() if t['pass'])
    summary_text += f"\nTOTAL: {n_pass}/{len(tests)} PASS"
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES / 'fig_z2170_information_theory.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {fig_path}")


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='z2170: Information-Theoretic Analysis')
    parser.add_argument('--steps', type=int, default=N_STEPS,
                        help=f'Timesteps per condition (default: {N_STEPS})')
    parser.add_argument('--sim', action='store_true',
                        help='Force simulator mode (skip FPGA)')
    args = parser.parse_args()

    print("=" * 60)
    print("z2170: Information-Theoretic Analysis — GPU→FPGA Coupling")
    print("=" * 60)
    print(f"  Steps/condition: {args.steps}")
    print(f"  Duration/cond:   {args.steps/SAMPLE_HZ:.0f}s")
    print(f"  Total duration:  ~{4 * args.steps/SAMPLE_HZ:.0f}s (4 conditions)")
    print(f"  Metrics: TE, MI, AIS, Phi")
    print(f"  Tests: T109-T114")
    print()

    result = run_experiment(n_steps=args.steps, skip_fpga=args.sim)

    print("\n" + "=" * 60)
    print(f"DONE: {result['summary']['pass_count']}/{result['summary']['total_tests']} PASS")
    print("=" * 60)
