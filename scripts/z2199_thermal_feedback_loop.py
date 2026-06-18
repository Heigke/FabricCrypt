#!/usr/bin/env python3
"""z2199_thermal_feedback_loop.py — Thermal Feedback Loop: FPGA ↔ GPU Homeostasis

Architecture:
  FPGA Spikes ──→ Workload Controller ──→ GPU HIP Kernels
       ↑                                       ↓
       │                                  GPU Heats Up/Cools
       │                                       ↓
       └──── Vg Modulation ←──── SMN Thermal ADC + hwmon temp

A slow homeostatic loop (~2 Hz) where FPGA spike activity modulates GPU workload
intensity, GPU thermal state feeds back to FPGA gate voltages, and the system
self-organizes toward a thermal attractor.

Phases:
  A (100 steps, ~50s): Baseline with fixed workload
  B (200 steps, ~100s): Feedback loop active — FPGA controls GPU workload
  C (100 steps, ~50s): Feedback disabled, observe thermal decay/memory

Conditions (3):
  FEEDBACK:       Full thermal feedback loop in Phase B
  FIXED_WORKLOAD: Same FPGA input but GPU workload fixed at 50%
  NO_THERMAL:     FPGA driven by power noise only, no thermal channel

Tests T287-T292:
  T287: Spike-thermal correlation > 0.2 in FEEDBACK
  T288: Thermal variance FEEDBACK < FIXED (homeostatic regulation)
  T289: Power std FEEDBACK > FIXED by > 10% (FPGA-driven modulation)
  T290: Thermal memory > 5 steps in Phase C
  T291: SMN thermal vs hwmon correlation r > 0.5
  T292: CV_spikes FEEDBACK < CV_spikes FIXED (more stable spike rate)

Hardware: AMD gfx1151 GPU + Arty A7 FPGA on /dev/ttyUSB*
"""

import os, sys, json, time, struct, argparse, math
import numpy as np
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
RESULTS = BASE / 'results'
FIGURES = RESULTS / 'FEEL_paper_update' / 'FEEL__Functionally_Embodied_Emergent_Learning__13_-5' / 'figures'

from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry

# ─── FPGA Protocol ───
SYNC = 0x55
CMD_SET_VG = 0x01
CMD_READ_TELEM = 0x02
CMD_SET_KILL = 0x03

# ─── Hardware Paths ───
HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"
HWMON_TEMP = "/sys/class/hwmon/hwmon7/temp1_input"
PM_TABLE_PATH = "/sys/kernel/ryzen_smu_drv/pm_table"
PP_DPM_SCLK = "/sys/class/drm/card0/device/pp_dpm_sclk"

# ─── Parameters ───
N_NEURONS = 8
BASE_VG = 0.58
ALPHA_INPUT = 0.20       # input coupling
BETA_THERMAL = 0.15      # thermal → Vg coupling
BETA_POWER = 0.08        # power noise → Vg coupling
IIR_ALPHA = 0.85         # IIR filter coefficient for noise smoothing

PHASE_A_STEPS = 100      # baseline
PHASE_B_STEPS = 200      # feedback active
PHASE_C_STEPS = 100      # decay observation
STEP_INTERVAL = 0.5      # 2 Hz loop rate


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


# ═══════════════════════════════════════════════════════════
# Thermal Reading
# ═══════════════════════════════════════════════════════════

def read_hwmon_temp():
    """Read hwmon temp1_input (millidegrees → C). SOC temp on kernel 6.14."""
    try:
        return int(open(HWMON_TEMP).read().strip()) / 1000.0
    except Exception:
        return None


def read_hwmon_power():
    """Read hwmon power1_average (uW → W). Rich 1/f dynamics."""
    try:
        return int(open(HWMON_POWER).read().strip()) / 1e6
    except Exception:
        return None


def read_smn_pm_table():
    """Read SMN PM table, extract thermal proxy.

    Reads 1024 bytes, parses as uint32 array, computes mean of
    float-castable values in range [20.0, 105.0] as temperature proxy.
    """
    if not os.path.exists(PM_TABLE_PATH):
        return None
    try:
        with open(PM_TABLE_PATH, 'rb') as f:
            data = f.read(1024)
        if len(data) < 64:
            return None
        # Parse all 4-byte chunks as float32, keep those in thermal range
        n_words = len(data) // 4
        thermal_vals = []
        for i in range(n_words):
            try:
                v = struct.unpack_from('<f', data, i * 4)[0]
                if not (math.isnan(v) or math.isinf(v)) and 20.0 <= v <= 105.0:
                    thermal_vals.append(v)
            except Exception:
                continue
        if thermal_vals:
            return float(np.mean(thermal_vals))
        return None
    except Exception:
        return None


def read_clock_freq():
    """Read current SCLK frequency from pp_dpm_sclk (active line marked with *)."""
    try:
        text = open(PP_DPM_SCLK).read()
        for line in text.strip().split('\n'):
            if '*' in line:
                # e.g. "1: 1000Mhz *"
                parts = line.split()
                for p in parts:
                    p_clean = p.replace('Mhz', '').replace('MHz', '')
                    try:
                        return float(p_clean)
                    except ValueError:
                        continue
        return None
    except Exception:
        return None


def read_thermal_state():
    """Read all thermal channels, return dict."""
    temp = read_hwmon_temp()
    power = read_hwmon_power()
    smn_temp = read_smn_pm_table()
    clock = read_clock_freq()
    return {
        'temp': temp if temp is not None else 50.0,
        'power': power if power is not None else 10.0,
        'smn_temp': smn_temp,
        'clock': clock,
    }


# ═══════════════════════════════════════════════════════════
# GPU Workload via HIP (torch)
# ═══════════════════════════════════════════════════════════

def launch_gpu_workload(intensity, torch_device):
    """Launch GPU matmul kernel with intensity-proportional matrix size.

    intensity in [0, 1]: N = 256 + int(768 * intensity), so N in [256, 1024].
    """
    try:
        import torch
        N = 256 + int(768 * max(0.0, min(1.0, intensity)))
        a = torch.randn(N, N, device=torch_device)
        b = torch.randn(N, N, device=torch_device)
        _ = a @ b
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


# ═══════════════════════════════════════════════════════════
# Software LIF Simulation Fallback
# ═══════════════════════════════════════════════════════════

def simulate_lif_step(vmem, vg_values, dt=0.5):
    """Single LIF step for 8 neurons. Returns (spike_counts, new_vmem)."""
    v_thresh = 1.0
    v_rest = 0.0
    tau_m = 0.02
    spikes = np.zeros(N_NEURONS, dtype=int)
    vmem_out = vmem.copy()
    # Sub-step at finer resolution
    n_sub = 50
    sub_dt = dt / n_sub
    for _ in range(n_sub):
        I_in = np.array(vg_values) * 5.0
        dvdt = (-vmem_out + I_in) / tau_m
        vmem_out += dvdt * sub_dt
        for i in range(N_NEURONS):
            if vmem_out[i] >= v_thresh:
                spikes[i] += 1
                vmem_out[i] = v_rest
    return spikes, vmem_out


# ═══════════════════════════════════════════════════════════
# IIR Filter
# ═══════════════════════════════════════════════════════════

class IIRFilter:
    """Stateful IIR low-pass: y[t] = alpha*y[t-1] + (1-alpha)*x[t]."""
    def __init__(self, alpha=0.85):
        self.alpha = alpha
        self.y = 0.0

    def step(self, x):
        self.y = self.alpha * self.y + (1.0 - self.alpha) * x
        return self.y


# ═══════════════════════════════════════════════════════════
# Main Experiment Loop
# ═══════════════════════════════════════════════════════════

def run_condition(condition_name, ser, fpga, w_in, w_therm, w_out, rng,
                  torch_device, args):
    """Run one full condition (phases A, B, C). Returns trajectory dict."""
    total_steps = PHASE_A_STEPS + PHASE_B_STEPS + PHASE_C_STEPS
    base_vg = args.base_vg
    alpha_in = args.alpha_input
    beta_th = args.beta_thermal

    # Trajectories
    temps = []
    powers = []
    smn_temps = []
    spike_sums = []
    workloads = []
    phases = []
    timestamps = []
    per_neuron_spikes = []

    # State
    prev_counts = None
    iir_temp = IIRFilter(alpha=IIR_ALPHA)
    iir_power = IIRFilter(alpha=IIR_ALPHA)
    prev_temp = None
    prev_power = None
    lif_vmem = np.zeros(N_NEURONS)  # for simulation fallback

    t0 = time.monotonic()

    for step in range(total_steps):
        # Determine phase
        if step < PHASE_A_STEPS:
            phase = 'A'
        elif step < PHASE_A_STEPS + PHASE_B_STEPS:
            phase = 'B'
        else:
            phase = 'C'

        # 1. Read thermal state
        thermal = read_thermal_state()
        temp = thermal['temp']
        power = thermal['power']
        smn_t = thermal['smn_temp']

        # Filtered values
        temp_f = iir_temp.step(temp)
        power_f = iir_power.step(power)

        # Deltas
        temp_delta = (temp - prev_temp) if prev_temp is not None else 0.0
        power_delta = (power - prev_power) if prev_power is not None else 0.0
        prev_temp = temp
        prev_power = power

        # 2. Compute thermal features (zero-centered)
        # Normalize around expected operating point
        temp_norm = (temp_f - 55.0) / 20.0   # center ~55C, range ~35-75
        power_norm = (power_f - 12.0) / 5.0  # center ~12W, range ~7-17

        # 3. Drive FPGA neurons
        # input signal: slow sine for consistent excitation
        t_now = time.monotonic() - t0
        input_sig = np.sin(2 * np.pi * 0.1 * t_now) * 0.5  # 0.1 Hz sine

        vg_vals = np.full(N_NEURONS, base_vg)
        vg_vals += alpha_in * input_sig * w_in

        if condition_name == 'FEEDBACK' and phase == 'B':
            # Full thermal feedback
            vg_vals += beta_th * temp_norm * w_therm
            vg_vals += args.beta_power * power_norm * w_therm * 0.5
        elif condition_name == 'NO_THERMAL':
            # Power noise only, no thermal
            vg_vals += args.beta_power * power_norm * w_therm
        elif condition_name == 'FIXED_WORKLOAD':
            # Same thermal input to FPGA as FEEDBACK
            if phase == 'B':
                vg_vals += beta_th * temp_norm * w_therm

        vg_vals = np.clip(vg_vals, 0.05, 0.95)

        # 4. Send to FPGA and read spikes
        if fpga:
            set_per_neuron_vg(ser, vg_vals)
            ser.reset_input_buffer()
            ser.write(bytes([SYNC, CMD_READ_TELEM]))
            ser.flush()
            telem = read_telem(ser, timeout=0.15)

            if telem:
                counts = [n['spike_count'] for n in telem]
                if prev_counts is not None:
                    deltas = []
                    for i in range(N_NEURONS):
                        d = (counts[i] - prev_counts[i]) & 0xFFFF
                        if d > 30000:
                            d = 0
                        deltas.append(d)
                else:
                    deltas = [0] * N_NEURONS
                prev_counts = counts[:]
            else:
                deltas = [0] * N_NEURONS
        else:
            # LIF simulation fallback
            deltas_arr, lif_vmem = simulate_lif_step(lif_vmem, vg_vals, dt=STEP_INTERVAL)
            deltas = deltas_arr.tolist()

        spike_sum = sum(deltas)

        # 5. Compute workload from spikes
        workload_raw = sigmoid(spike_sum * np.dot(np.array(deltas) / max(spike_sum, 1.0), w_out)
                               if spike_sum > 0 else 0.0)

        if condition_name == 'FIXED_WORKLOAD':
            workload = 0.5  # fixed
        elif phase == 'A' or phase == 'C':
            workload = 0.5  # baseline/decay
        else:
            workload = float(workload_raw)

        # 6. Launch GPU kernel
        launch_gpu_workload(workload, torch_device)

        # Record
        temps.append(float(temp))
        powers.append(float(power))
        smn_temps.append(float(smn_t) if smn_t is not None else None)
        spike_sums.append(int(spike_sum))
        workloads.append(float(workload))
        phases.append(phase)
        timestamps.append(float(t_now))
        per_neuron_spikes.append(deltas)

        # Print progress
        if step % 20 == 0 or step == total_steps - 1:
            print(f"    Step {step:3d}/{total_steps} [{phase}] "
                  f"T={temp:.1f}C P={power:.1f}W spikes={spike_sum:3d} "
                  f"work={workload:.2f}")

        # 7. Wait for thermal propagation
        elapsed = time.monotonic() - t0 - step * STEP_INTERVAL
        wait = STEP_INTERVAL - elapsed % STEP_INTERVAL
        if wait > 0.01:
            time.sleep(wait)

    return {
        'condition': condition_name,
        'temps': temps,
        'powers': powers,
        'smn_temps': smn_temps,
        'spike_sums': spike_sums,
        'workloads': workloads,
        'phases': phases,
        'timestamps': timestamps,
        'per_neuron_spikes': per_neuron_spikes,
    }


# ═══════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════

def analyze_condition(traj):
    """Compute metrics for one condition trajectory."""
    temps = np.array(traj['temps'])
    powers = np.array(traj['powers'])
    spikes = np.array(traj['spike_sums'])
    phases = traj['phases']
    smn_raw = traj['smn_temps']

    # Phase masks
    phase_a = np.array([p == 'A' for p in phases])
    phase_b = np.array([p == 'B' for p in phases])
    phase_c = np.array([p == 'C' for p in phases])

    # Basic stats
    stats = {
        'temp_mean': float(temps.mean()),
        'temp_std': float(temps.std()),
        'power_mean': float(powers.mean()),
        'power_std': float(powers.std()),
        'spike_mean': float(spikes.mean()),
        'spike_std': float(spikes.std()),
        'spike_cv': float(spikes.std() / max(spikes.mean(), 1e-6)),
    }

    # Phase B stats
    if phase_b.sum() > 2:
        temps_b = temps[phase_b]
        powers_b = powers[phase_b]
        spikes_b = spikes[phase_b]
        stats['phase_b_temp_var'] = float(temps_b.var())
        stats['phase_b_power_std'] = float(powers_b.std())
        stats['phase_b_spike_cv'] = float(spikes_b.std() / max(spikes_b.mean(), 1e-6))
    else:
        stats['phase_b_temp_var'] = 0.0
        stats['phase_b_power_std'] = 0.0
        stats['phase_b_spike_cv'] = 0.0

    # Spike-thermal correlation (T287)
    # Does spike_sum[t] predict temp[t+1]?
    if len(spikes) > 5 and len(temps) > 5:
        # Lag-1 cross-correlation: spikes[:-1] vs temps[1:]
        s = spikes[:-1].astype(float)
        t = temps[1:]
        if s.std() > 0 and t.std() > 0:
            stats['spike_thermal_corr'] = float(np.corrcoef(s, t)[0, 1])
        else:
            stats['spike_thermal_corr'] = 0.0
    else:
        stats['spike_thermal_corr'] = 0.0

    # SMN vs hwmon correlation (T291)
    smn_vals = [v for v in smn_raw if v is not None]
    hwmon_vals = [temps[i] for i, v in enumerate(smn_raw) if v is not None]
    if len(smn_vals) > 5 and np.std(smn_vals) > 0 and np.std(hwmon_vals) > 0:
        stats['smn_hwmon_corr'] = float(np.corrcoef(smn_vals, hwmon_vals)[0, 1])
    else:
        stats['smn_hwmon_corr'] = 0.0

    # Thermal memory in Phase C (T290)
    # How many steps after feedback stops does temperature remain elevated/depressed?
    if phase_c.sum() > 5 and phase_b.sum() > 5:
        temp_end_b = temps[phase_b][-1]
        temps_c = temps[phase_c]
        temp_baseline_a = temps[phase_a].mean() if phase_a.sum() > 0 else temps_c[-1]
        # Memory = number of steps where temp is still > halfway between end-of-B and baseline-A
        threshold = (temp_end_b + temp_baseline_a) / 2.0
        if temp_end_b > temp_baseline_a:
            memory_steps = int(np.sum(temps_c > threshold))
        else:
            memory_steps = int(np.sum(temps_c < threshold))
        stats['thermal_memory_steps'] = memory_steps
    else:
        stats['thermal_memory_steps'] = 0

    return stats


def run_tests(condition_results):
    """Evaluate T287-T292."""
    fb = condition_results.get('FEEDBACK', {})
    fx = condition_results.get('FIXED_WORKLOAD', {})
    nt = condition_results.get('NO_THERMAL', {})

    tests = {}

    # T287: Spike-thermal correlation > 0.2 in FEEDBACK
    corr = abs(fb.get('spike_thermal_corr', 0))
    tests['T287_spike_thermal_corr'] = {
        'pass': corr > 0.2,
        'value': corr,
        'threshold': 0.2,
        'description': 'Spike-thermal correlation > 0.2 in FEEDBACK',
    }

    # T288: Thermal variance FEEDBACK < FIXED (homeostatic regulation)
    tv_fb = fb.get('phase_b_temp_var', 999)
    tv_fx = fx.get('phase_b_temp_var', 0)
    tests['T288_homeostatic_regulation'] = {
        'pass': tv_fb < tv_fx if tv_fx > 0 else False,
        'value_feedback': tv_fb,
        'value_fixed': tv_fx,
        'description': 'Thermal variance FEEDBACK < FIXED (homeostasis)',
    }

    # T289: Power std FEEDBACK > FIXED by >10%
    ps_fb = fb.get('phase_b_power_std', 0)
    ps_fx = fx.get('phase_b_power_std', 0)
    ratio = ps_fb / max(ps_fx, 1e-6)
    tests['T289_power_modulation'] = {
        'pass': ratio > 1.10,
        'value_feedback': ps_fb,
        'value_fixed': ps_fx,
        'ratio': ratio,
        'description': 'Power std FEEDBACK > FIXED by >10%',
    }

    # T290: Thermal memory > 5 steps in Phase C
    mem = fb.get('thermal_memory_steps', 0)
    tests['T290_thermal_memory'] = {
        'pass': mem > 5,
        'value': mem,
        'threshold': 5,
        'description': 'Thermal memory > 5 steps in Phase C',
    }

    # T291: SMN vs hwmon correlation > 0.5
    smn_corr = fb.get('smn_hwmon_corr', 0)
    tests['T291_smn_hwmon_corr'] = {
        'pass': abs(smn_corr) > 0.5,
        'value': smn_corr,
        'threshold': 0.5,
        'description': 'SMN thermal vs hwmon correlation r > 0.5',
    }

    # T292: CV_spikes FEEDBACK < CV_spikes FIXED
    cv_fb = fb.get('phase_b_spike_cv', 999)
    cv_fx = fx.get('phase_b_spike_cv', 0)
    tests['T292_spike_stability'] = {
        'pass': cv_fb < cv_fx if cv_fx > 0 else False,
        'value_feedback': cv_fb,
        'value_fixed': cv_fx,
        'description': 'FEEDBACK spike rate more stable than FIXED',
    }

    return tests


# ═══════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════

def make_figure(all_trajectories, condition_stats, tests, outpath):
    """3x2 figure: thermal trajectories, power trajectories, spike rates,
    spike-thermal scatter, workload, and test summary."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping figure")
        return

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('z2199: Thermal Feedback Loop — FPGA ↔ GPU Homeostasis', fontsize=14)

    colors = {'FEEDBACK': '#E53935', 'FIXED_WORKLOAD': '#1E88E5', 'NO_THERMAL': '#43A047'}
    total_steps = PHASE_A_STEPS + PHASE_B_STEPS + PHASE_C_STEPS

    # Phase boundary lines
    def add_phase_lines(ax):
        ax.axvline(x=PHASE_A_STEPS, color='gray', linestyle='--', alpha=0.5, label='Phase B start')
        ax.axvline(x=PHASE_A_STEPS + PHASE_B_STEPS, color='gray', linestyle=':', alpha=0.5,
                   label='Phase C start')

    # (a) Temperature trajectories
    ax = axes[0, 0]
    for cond, traj in all_trajectories.items():
        ax.plot(traj['temps'], color=colors.get(cond, 'gray'), alpha=0.8, label=cond, linewidth=0.8)
    add_phase_lines(ax)
    ax.set_ylabel('Temperature (C)')
    ax.set_title('(a) GPU Temperature Trajectory')
    ax.legend(fontsize=7)
    ax.set_xlabel('Step')

    # (b) Power trajectories
    ax = axes[0, 1]
    for cond, traj in all_trajectories.items():
        ax.plot(traj['powers'], color=colors.get(cond, 'gray'), alpha=0.8, label=cond, linewidth=0.8)
    add_phase_lines(ax)
    ax.set_ylabel('Power (W)')
    ax.set_title('(b) GPU Power Trajectory')
    ax.legend(fontsize=7)
    ax.set_xlabel('Step')

    # (c) Spike sums over time
    ax = axes[1, 0]
    for cond, traj in all_trajectories.items():
        # Smooth with rolling mean
        spk = np.array(traj['spike_sums'], dtype=float)
        if len(spk) > 10:
            kernel = np.ones(10) / 10
            spk_smooth = np.convolve(spk, kernel, mode='same')
        else:
            spk_smooth = spk
        ax.plot(spk_smooth, color=colors.get(cond, 'gray'), alpha=0.8, label=cond, linewidth=0.8)
    add_phase_lines(ax)
    ax.set_ylabel('Spike sum (smoothed)')
    ax.set_title('(c) FPGA Spike Activity')
    ax.legend(fontsize=7)
    ax.set_xlabel('Step')

    # (d) Workload over time
    ax = axes[1, 1]
    for cond, traj in all_trajectories.items():
        ax.plot(traj['workloads'], color=colors.get(cond, 'gray'), alpha=0.8, label=cond, linewidth=0.8)
    add_phase_lines(ax)
    ax.set_ylabel('Workload intensity')
    ax.set_title('(d) GPU Workload (FPGA-driven)')
    ax.legend(fontsize=7)
    ax.set_xlabel('Step')

    # (e) Spike-thermal scatter (FEEDBACK, Phase B)
    ax = axes[2, 0]
    fb_traj = all_trajectories.get('FEEDBACK')
    if fb_traj:
        phase_b_mask = [p == 'B' for p in fb_traj['phases']]
        spk_b = np.array(fb_traj['spike_sums'])[:-1]
        tmp_b = np.array(fb_traj['temps'])[1:]
        mask_b = np.array(phase_b_mask[:-1])
        if mask_b.sum() > 5:
            ax.scatter(spk_b[mask_b], tmp_b[mask_b], alpha=0.3, s=10, color=colors['FEEDBACK'])
            # Fit line
            if np.std(spk_b[mask_b]) > 0:
                z = np.polyfit(spk_b[mask_b], tmp_b[mask_b], 1)
                x_fit = np.linspace(spk_b[mask_b].min(), spk_b[mask_b].max(), 50)
                ax.plot(x_fit, np.polyval(z, x_fit), 'k--', alpha=0.6)
    ax.set_xlabel('Spike sum (t)')
    ax.set_ylabel('Temperature (t+1)')
    ax.set_title('(e) Spike→Thermal Prediction (FEEDBACK Phase B)')

    # (f) Test results summary
    ax = axes[2, 1]
    ax.axis('off')
    test_names = sorted(tests.keys())
    y_pos = 0.95
    for tn in test_names:
        t = tests[tn]
        status = 'PASS' if t['pass'] else 'FAIL'
        color = 'green' if t['pass'] else 'red'
        desc = t.get('description', '')
        ax.text(0.02, y_pos, f"{tn}: {status}", fontsize=10, fontweight='bold',
                color=color, transform=ax.transAxes, va='top', fontfamily='monospace')
        ax.text(0.35, y_pos, desc, fontsize=8, transform=ax.transAxes, va='top')
        y_pos -= 0.14
    ax.set_title('(f) Test Results (T287-T292)')

    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(outpath), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: {outpath}")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='z2199: Thermal Feedback Loop — FPGA ↔ GPU Homeostasis')
    parser.add_argument('--base-vg', type=float, default=BASE_VG, help='Base gate voltage')
    parser.add_argument('--alpha-input', type=float, default=ALPHA_INPUT, help='Input coupling')
    parser.add_argument('--beta-thermal', type=float, default=BETA_THERMAL, help='Thermal coupling')
    parser.add_argument('--beta-power', type=float, default=BETA_POWER, help='Power noise coupling')
    parser.add_argument('--step-interval', type=float, default=STEP_INTERVAL, help='Loop interval (s)')
    args = parser.parse_args()

    print("=" * 70)
    print("z2199: Thermal Feedback Loop — FPGA Spikes ↔ GPU Thermal Homeostasis")
    print("=" * 70)
    print(f"  Base Vg: {args.base_vg}  Alpha: {args.alpha_input}  "
          f"Beta_thermal: {args.beta_thermal}  Beta_power: {args.beta_power}")
    print(f"  Phases: A={PHASE_A_STEPS} B={PHASE_B_STEPS} C={PHASE_C_STEPS} "
          f"steps @ {1.0/args.step_interval:.1f} Hz")

    rng = np.random.default_rng(2199)
    w_in = rng.uniform(-1, 1, size=N_NEURONS)
    w_therm = rng.uniform(-1, 1, size=N_NEURONS)
    w_out = rng.uniform(-1, 1, size=N_NEURONS)

    # ─── Init torch ───
    torch_device = 'cpu'
    try:
        import torch
        if torch.cuda.is_available():
            torch_device = 'cuda'
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("  WARN: CUDA not available, GPU workload will be CPU-only")
    except ImportError:
        print("  WARN: torch not installed, GPU workload disabled")

    # ─── Init telemetry ───
    try:
        telem = SysfsHwmonTelemetry()
        print(f"  SysfsHwmonTelemetry initialized")
    except Exception as e:
        print(f"  WARN: SysfsHwmonTelemetry failed: {e}")

    # ─── Check SMN PM table ───
    smn_available = os.path.exists(PM_TABLE_PATH)
    print(f"  SMN PM table: {'available' if smn_available else 'NOT available'}")

    # ─── Check thermal channels ───
    init_thermal = read_thermal_state()
    print(f"  Initial thermal: T={init_thermal['temp']:.1f}C "
          f"P={init_thermal['power']:.1f}W "
          f"SMN={'%.1f' % init_thermal['smn_temp'] if init_thermal['smn_temp'] else 'N/A'}")

    # ─── Connect FPGA ───
    print("\n[1/5] Connecting to FPGA...")
    ser, port = find_fpga()
    fpga = ser is not None
    simulated = not fpga

    if fpga:
        print(f"  Connected: {port}")
        ser.write(bytes([SYNC, CMD_SET_KILL, 0x00]))
        ser.flush()
        time.sleep(0.1)
        print("  Kill switch disabled")
    else:
        print("  FPGA not found — using LIF simulation fallback")

    results = {
        'experiment': 'z2199_thermal_feedback_loop',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'base_vg': args.base_vg,
            'alpha_input': args.alpha_input,
            'beta_thermal': args.beta_thermal,
            'beta_power': args.beta_power,
            'step_interval': args.step_interval,
            'phase_a_steps': PHASE_A_STEPS,
            'phase_b_steps': PHASE_B_STEPS,
            'phase_c_steps': PHASE_C_STEPS,
            'n_neurons': N_NEURONS,
            'w_in': w_in.tolist(),
            'w_therm': w_therm.tolist(),
            'w_out': w_out.tolist(),
        },
        'simulated': simulated,
        'smn_available': smn_available,
        'torch_device': torch_device,
        'conditions': {},
        'tests': {},
    }

    # ─── Run conditions ───
    conditions = ['FEEDBACK', 'FIXED_WORKLOAD', 'NO_THERMAL']
    all_trajectories = {}
    condition_stats = {}

    for idx, cond in enumerate(conditions):
        print(f"\n[{idx+2}/5] Running condition: {cond}")
        print(f"  ({PHASE_A_STEPS + PHASE_B_STEPS + PHASE_C_STEPS} steps, "
              f"~{(PHASE_A_STEPS + PHASE_B_STEPS + PHASE_C_STEPS) * args.step_interval:.0f}s)")

        traj = run_condition(cond, ser, fpga, w_in, w_therm, w_out, rng,
                             torch_device, args)
        all_trajectories[cond] = traj

        stats = analyze_condition(traj)
        condition_stats[cond] = stats
        results['conditions'][cond] = stats

        print(f"  Results: T={stats['temp_mean']:.1f}±{stats['temp_std']:.2f}C "
              f"P={stats['power_mean']:.1f}±{stats['power_std']:.2f}W "
              f"spikes={stats['spike_mean']:.1f}±{stats['spike_std']:.1f} "
              f"CV={stats['spike_cv']:.3f}")
        print(f"  Spike-thermal corr: {stats['spike_thermal_corr']:.4f}")
        print(f"  Thermal memory: {stats['thermal_memory_steps']} steps")
        if stats['smn_hwmon_corr'] != 0:
            print(f"  SMN-hwmon corr: {stats['smn_hwmon_corr']:.4f}")

    # ─── Run tests ───
    print("\n[5/5] Evaluating tests T287-T292...")
    tests = run_tests(condition_stats)
    results['tests'] = tests

    n_pass = sum(1 for t in tests.values() if t['pass'])
    n_total = len(tests)
    print(f"\n{'=' * 70}")
    print(f"RESULTS: {n_pass}/{n_total} tests PASS")
    print(f"{'=' * 70}")
    for tn in sorted(tests.keys()):
        t = tests[tn]
        status = 'PASS' if t['pass'] else 'FAIL'
        print(f"  {tn}: {status} — {t.get('description', '')}")
        # Print key values
        for k, v in t.items():
            if k not in ('pass', 'description'):
                print(f"    {k}: {v}")

    # ─── Save results ───
    RESULTS.mkdir(parents=True, exist_ok=True)
    outfile = RESULTS / 'z2199_thermal_feedback_loop.json'
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\nResults saved: {outfile}")

    # ─── Figure ───
    try:
        FIGURES.mkdir(parents=True, exist_ok=True)
        figpath = FIGURES / 'z2199_thermal_feedback_loop.png'
        make_figure(all_trajectories, condition_stats, tests, figpath)
    except Exception as e:
        print(f"  Figure generation failed: {e}")

    # ─── Cleanup ───
    if fpga and ser:
        # Set neurons to safe resting Vg
        set_per_neuron_vg(ser, [0.3] * N_NEURONS)
        ser.close()
        print("  FPGA connection closed")

    print(f"\nDone. {n_pass}/{n_total} PASS")


if __name__ == '__main__':
    main()
