#!/usr/bin/env python3
"""
z2100: Integrated Workspace LM
===============================
Built on z2099 (Bridge Law LM). 5 targeted architectural changes to address T32-T39
failures, plus stability fixes for generation collapse and decorative hardware.

Key changes from previous:
  1. TOKEN-SHIFT CIPHER: Regime 1 predicts (token + 50000) % VOCAB_SIZE.
     Forces strict LoRA B specialization (fixes T7, T13, T19).
  2. GATED INJECTIONS: Sensor residual injections are now gated and LayerNormed.
  3. INFERENCE EMA: Sensors are EMA-smoothed during .eval() to stop generation drift.
  4. Removed forced LoRA divergence loss (prevented manifold destruction).
"""

import os, sys, json, math, time, struct, ctypes, ctypes.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONSTANTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEVICE = 'cuda'
BS = 4
SEQ_LEN = 128
EPOCHS = 17
LORA_RANK = 8
LORA_ALPHA = 12      # scale=3; with body_scale~0.5 → effective ~1.5; enough for A/B specialization
LORA_BLOCKS = range(10, 19)  # Qwen2.5-1.5B layers 10-18 (middle of 28)
N_EVAL_BATCHES = 30
DVFS_SETTLE_S = 1.5
GASLIGHT_FRAC = 0.30
VOCAB_SIZE = 151936  # Qwen2.5-1.5B vocab

# Cipher Offset replaces domain/skip-gram shift
CIPHER_OFFSET = 50000

# Sensor dimensions
DELTA_DIM = 5
ANALOG_DIM = 6       # temp, power, sclk, df_r, df_w, df_c
ENERGY_DIM = 3        # pkg, core, gpu
FREQ_DIM = 3          # sclk_norm, freq_ratio, pstate
INTRINSIC_DIM = 12    # hwreg reads from shader
THERMAL_DIM = 4       # hwmon temps
PM_DEEP_DIM = 8       # PM table fields
SMN_RAW_DIM = 6       # SMN thermal ADC
GPU_METRICS_DIM = 6   # dram_r, dram_w, c0_avg, throttle_prochot, throttle_thermal, throttle_power
GPU_METRICS_DEEP_DIM = 12  # z2098: per-core C0 (8 active), per-core clk delta (4)
FENCE_DIM = 4             # z2098: ring queue depths (gfx, comp0, comp1, comp2)
THM_SPATIAL_A_DIM = 16   # Bank A thermal ADC sensors (0x598A4-0x598E0)
THM_SPATIAL_B_DIM = 16   # Bank B thermal ADC sensors (0x599C0-0x599FC)
CPU_PMU_DIM = 3          # Zen 5 core: instructions, branches, br_mispredict
REPORTED_DELTA_DIM = 5  # externally-reported delta (can be corrupted for gaslighting)
STATUS_DIM = 2        # regime_float, dvfs_float
ACTION_DIM = 4        # sclk_norm, ppt_norm, demand, spare

N_SUBSTRATE_TOKENS = 17  # z2098: +2 for gpu_metrics_deep, fence_ring
TOKEN_DIM = 32

# Phase boundaries
PHASE0_END = 3        # body encoder pretrain
PHASE1_END = 10       # forced regime alternation
PHASE2_END = 14       # model-controlled DVFS
PHASE3_END = EPOCHS   # gaslighting training

SCLK_LOW_CAL = 600.0   # placeholder — updated after DVFS sanity check
SCLK_HIGH_CAL = 2900.0  # placeholder — updated after DVFS sanity check

GATE_TEMP = 8.0          # sigmoid temperature for freq_gate
CONTRASTIVE_LAMBDA = 0.3  # weight for contrastive kill-shot loss
CONTRASTIVE_MARGIN = 0.3  # nats margin: wrong gate should be this much worse
CONTRASTIVE_FRAC = 0.25   # fraction of batches with contrastive loss
AGREEMENT_GAMMA = 2.0     # exponent for agreement modulation

META2_LOSS_WEIGHT = 0.1       # weight for 2nd-order metacognition loss
ATTRIBUTION_LOSS_WEIGHT = 0.1 # weight for attribution loss
N_ATTRIBUTION_CLASSES = 17    # one class per substrate token

N_WORKSPACE_SLOTS = 4         # workspace bottleneck (GWT capacity limit)
ORTHO_LOSS_WEIGHT = 0.01      # head specialization orthogonality loss
TEMP_PRED_LOSS_WEIGHT = 0.05  # thermal predictor loss weight
GATE_EMA_TAU = 0.3            # temporal gate EMA smoothing

# ISA Personalities
PERSONALITY_A = {
    'round_mode': 0b0000,   # round-to-nearest-even
    'denorm_mode': 0b1111,  # all denorms enabled
    'chain_code': 0,        # standard FMA chain
    'perm_code': 0,         # identity permutation
}
PERSONALITY_B = {
    'round_mode': 0b0011,   # round-toward-zero
    'denorm_mode': 0b0000,  # denorms flushed to zero
    'chain_code': 1,        # alternate FMA chain
    'perm_code': 1,         # byte-swap permutation
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HARDWARE ACCESS 
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DVFS_AVAILABLE = False
DVFS_PATH = None

def find_dvfs_sysfs():
    global DVFS_AVAILABLE, DVFS_PATH
    for card in ['card1', 'card0']:
        p = f'/sys/class/drm/{card}/device/power_dpm_force_performance_level'
        if os.path.exists(p):
            try:
                with open(p, 'r') as f:
                    val = f.read().strip()
                DVFS_PATH = p
                DVFS_AVAILABLE = True
                print(f"[DVFS] Found: {p} = {val}")
                return
            except:
                pass
    print("[DVFS] Not available")

def set_dvfs_level(level, wait=True):
    if not DVFS_AVAILABLE: return
    torch.cuda.synchronize()
    name = {0: 'low', 1: 'auto', 2: 'high'}[level]
    try:
        with open(DVFS_PATH, 'w') as f:
            f.write(name)
    except:
        return
    if wait: _poll_dvfs_settle(level)

def _poll_dvfs_settle(level):
    target_low = level == 0
    for _ in range(30):
        sclk = read_current_sclk_mhz()
        if target_low and sclk < 800: return
        if not target_low and sclk > 1200: return
        time.sleep(0.1)
    time.sleep(DVFS_SETTLE_S)

def restore_dvfs_auto():
    if DVFS_AVAILABLE:
        torch.cuda.synchronize()
        try:
            with open(DVFS_PATH, 'w') as f: f.write('auto')
        except: pass

def read_current_sclk_mhz():
    for hwmon in ['hwmon7', 'hwmon6', 'hwmon5']:
        p = f'/sys/class/hwmon/{hwmon}/freq1_input'
        if os.path.exists(p):
            try:
                with open(p, 'r') as f: return float(f.read().strip()) / 1e6
            except: pass
    return 600.0

GPU_METRICS_PATH = None
def find_gpu_metrics():
    global GPU_METRICS_PATH
    for card in ['card1', 'card0']:
        p = f'/sys/class/drm/{card}/device/gpu_metrics'
        if os.path.exists(p):
            GPU_METRICS_PATH = p
            print(f"[gpu_metrics] Found: {p}")
            return

def read_gpu_metrics_v3():
    result = {'dram_reads': 0, 'dram_writes': 0, 'c0_activity_avg': 0.0, 'throttle_prochot': 0, 'throttle_thermal': 0, 'throttle_power': 0, 'temperature_gfx': 0, 'temperature_soc': 0, 'per_core_c0': [0.0]*16, 'per_core_clk': [0]*16, 'avg_gfxclk': 0, 'avg_socclk': 0, 'avg_fclk': 0, 'avg_uclk': 0, 'avg_gfx_power': 0, 'avg_all_core_power': 0, 'avg_socket_power': 0, 'energy_acc': 0, 'throttle_residency_prochot': 0, 'throttle_residency_thm_gfx': 0, 'throttle_residency_thm_soc': 0, 'gfx_max_freq': 0}
    if not GPU_METRICS_PATH: return result
    try:
        with open(GPU_METRICS_PATH, 'rb') as f: data = f.read()
        if len(data) < 264: return result
        size_h, fmt_rev, content_rev = struct.unpack_from('<HBB', data, 0)
        if fmt_rev < 3: return result
        off = 4
        t_gfx = struct.unpack_from('<H', data, off)[0]; off += 2
        t_soc = struct.unpack_from('<H', data, off)[0]; off += 2
        result['temperature_gfx'] = t_gfx / 100.0 if t_gfx < 20000 else 0
        result['temperature_soc'] = t_soc / 100.0 if t_soc < 20000 else 0
        core_temps = struct.unpack_from('<16H', data, off); off += 34
        avg_gfx_act = struct.unpack_from('<H', data, off)[0]; off += 2
        avg_vcn_act = struct.unpack_from('<H', data, off)[0]; off += 2
        avg_ipu_act = struct.unpack_from('<8H', data, off); off += 16
        c0_raw = struct.unpack_from('<16H', data, off); off += 32
        c0_vals = []
        for i, c0 in enumerate(c0_raw):
            pct = c0 / 100.0 if c0 <= 10000 else 0.0
            result['per_core_c0'][i] = pct
            if pct > 0: c0_vals.append(pct)
        result['c0_activity_avg'] = np.mean(c0_vals) if c0_vals else 0.0
        result['dram_reads'] = struct.unpack_from('<H', data, off)[0]; off += 2
        result['dram_writes'] = struct.unpack_from('<H', data, off)[0]; off += 6
        sys_clk = struct.unpack_from('<Q', data, off)[0]; off += 8
        result['energy_acc'] = sys_clk
        avg_socket = struct.unpack_from('<I', data, off)[0]; off += 4
        avg_ipu_pwr = struct.unpack_from('<H', data, off)[0]; off += 2
        avg_apu_pwr = struct.unpack_from('<I', data, off)[0]; off += 4
        avg_gfx_pwr = struct.unpack_from('<I', data, off)[0]; off += 4
        avg_dgpu_pwr = struct.unpack_from('<I', data, off)[0]; off += 4
        avg_all_core = struct.unpack_from('<I', data, off)[0]; off += 4
        result['avg_socket_power'] = avg_socket if avg_socket < 0xFFFF0000 else 0
        result['avg_gfx_power'] = avg_gfx_pwr if avg_gfx_pwr < 0xFFFF0000 else 0
        result['avg_all_core_power'] = avg_all_core if avg_all_core < 0xFFFF0000 else 0
        off += 38
        avg_gfxclk = struct.unpack_from('<H', data, off)[0]; off += 2
        avg_socclk = struct.unpack_from('<H', data, off)[0]; off += 6
        avg_fclk = struct.unpack_from('<H', data, off)[0]; off += 4
        avg_uclk = struct.unpack_from('<H', data, off)[0]; off += 4
        result['avg_gfxclk'] = avg_gfxclk if avg_gfxclk < 65535 else 0
        result['avg_socclk'] = avg_socclk if avg_socclk < 65535 else 0
        result['avg_fclk'] = avg_fclk if avg_fclk < 65535 else 0
        result['avg_uclk'] = avg_uclk if avg_uclk < 65535 else 0
        for i in range(16):
            clk = struct.unpack_from('<H', data, off)[0]; off += 2
            result['per_core_clk'][i] = clk if clk < 65535 else 0
        off += 2
        gfx_max = struct.unpack_from('<H', data, off)[0]; off += 2
        result['gfx_max_freq'] = gfx_max if gfx_max < 65535 else 0
        thr_names = ['prochot', 'spl', 'fppt', 'sppt', 'thm_core', 'thm_gfx', 'thm_soc']
        for name in thr_names:
            if off + 4 <= len(data):
                val = struct.unpack_from('<I', data, off)[0]; off += 4
                if name == 'prochot':
                    result['throttle_residency_prochot'] = val
                    result['throttle_prochot'] = 1 if val > 0 else 0
                elif name == 'thm_gfx':
                    result['throttle_residency_thm_gfx'] = val
                    result['throttle_thermal'] = 1 if val > 0 else 0
                elif name == 'thm_soc':
                    result['throttle_residency_thm_soc'] = val
                    result['throttle_power'] = 1 if val > 0 else 0
    except: pass
    return result

def read_gpu_metrics_vec():
    gm = read_gpu_metrics_v3()
    return torch.tensor([
        min(gm['dram_reads'] / 1e4, 1.0),
        min(gm['dram_writes'] / 1e4, 1.0),
        gm['c0_activity_avg'] / 100.0,
        float(gm['throttle_prochot']),
        float(gm['throttle_thermal']),
        float(gm['throttle_power']),
    ], dtype=torch.float32)

def read_gpu_metrics_deep_vec():
    gm = read_gpu_metrics_v3()
    c0 = [min(gm['per_core_c0'][i] / 100.0, 1.0) for i in range(8)]
    clks = [gm['per_core_clk'][i] for i in range(16) if gm['per_core_clk'][i] > 0]
    mean_clk = np.mean(clks) if clks else 1000.0
    clk_deltas = []
    for i in range(min(4, len(clks))):
        clk_deltas.append((clks[i] - mean_clk) / max(mean_clk, 1.0))
    while len(clk_deltas) < 4: clk_deltas.append(0.0)
    return torch.tensor(c0 + clk_deltas, dtype=torch.float32)

FENCE_PATH = None
def init_fence_reader():
    global FENCE_PATH
    for card_id in [1, 0]:
        p = f'/sys/kernel/debug/dri/{card_id}/amdgpu_fence_info'
        if os.path.exists(p):
            try:
                with open(p, 'r') as f: f.read(100)
                FENCE_PATH = p
                print(f"[FENCE] Available: {p}")
                return
            except: pass
    print("[FENCE] Not available (need sudo)")

def read_fence_vec():
    depths = [0.0] * FENCE_DIM
    if FENCE_PATH is None: return torch.tensor(depths, dtype=torch.float32)
    try:
        with open(FENCE_PATH, 'r') as f: text = f.read()
        ring_idx, emitted, signaled = 0, 0, 0
        for line in text.split('\n'):
            if 'Last emitted' in line and 'trailing' not in line:
                try: emitted = int(line.split()[-1], 16)
                except: pass
            elif 'Last signaled fence' in line:
                try: signaled = int(line.split()[-1], 16)
                except: pass
                depth = max(0, emitted - signaled)
                if ring_idx < FENCE_DIM:
                    depths[ring_idx] = min(depth / 100.0, 1.0)
                ring_idx += 1
    except: pass
    return torch.tensor(depths, dtype=torch.float32)

DF_FDS, DF_AVAILABLE = {}, False
def init_df_counters():
    global DF_FDS, DF_AVAILABLE
    try:
        libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)
        class PerfEventAttr(ctypes.Structure):
            _fields_ = [('type', ctypes.c_uint32), ('size', ctypes.c_uint32), ('config', ctypes.c_uint64), ('sample_period', ctypes.c_uint64), ('sample_type', ctypes.c_uint64), ('read_format', ctypes.c_uint64), ('flags', ctypes.c_uint64), ('wakeup_events', ctypes.c_uint32), ('bp_type', ctypes.c_uint32), ('config1', ctypes.c_uint64), ('config2', ctypes.c_uint64)]
        with open('/sys/bus/event_source/devices/amd_df/type', 'r') as f: df_type = int(f.read().strip())
        events = {'df_dram_read': (0x07 | (0x48 << 8)), 'df_dram_write': (0x07 | (0xC0 << 8)), 'df_coherent': (0x07 | (0x60 << 8))}
        NR_perf_event_open = 298
        for name, config in events.items():
            attr = PerfEventAttr()
            attr.type, attr.size, attr.config, attr.flags = df_type, ctypes.sizeof(PerfEventAttr), config, 0
            fd = libc.syscall(NR_perf_event_open, ctypes.byref(attr), -1, 0, -1, 0)
            if fd >= 0:
                libc.ioctl(fd, 0x2400, 0)
                DF_FDS[name] = fd
        DF_AVAILABLE = len(DF_FDS) > 0
    except: pass

def read_df_snapshot():
    res = {}
    for name, fd in DF_FDS.items():
        n = os.read(fd, 8)
        res[name] = struct.unpack('Q', n)[0] if len(n) == 8 else 0
    return res

RAPL_AVAILABLE, RAPL_PATHS = False, {}
def check_rapl():
    global RAPL_AVAILABLE, RAPL_PATHS
    base = '/sys/class/powercap'
    for domain in ['intel-rapl:0', 'intel-rapl:0:0']:
        ej = os.path.join(base, domain, 'energy_uj')
        if os.path.exists(ej):
            try:
                with open(os.path.join(base, domain, 'name'), 'r') as f:
                    RAPL_PATHS[f.read().strip()] = ej
            except: pass
    RAPL_AVAILABLE = len(RAPL_PATHS) > 0

def read_rapl_snapshot():
    res = {}
    for name, path in RAPL_PATHS.items():
        try:
            with open(path, 'r') as f: res[name] = int(f.read().strip())
        except: res[name] = 0
    return res

def compute_batch_joules(before, after, gpu_ppt_mw, elapsed_s=None):
    total_uj = sum((after[n] - before[n] + (1<<32) if after[n] < before[n] else after[n] - before[n]) for n in before if n in after)
    total_j = total_uj / 1e6
    if gpu_ppt_mw > 0 and elapsed_s: total_j += (gpu_ppt_mw / 1000.0) * elapsed_s
    return total_j

MSR_AVAILABLE, MSR_FD = False, None
def init_msr():
    global MSR_AVAILABLE, MSR_FD
    try:
        MSR_FD = os.open('/dev/cpu/0/msr', os.O_RDONLY)
        MSR_AVAILABLE = True
    except: pass

def read_freq_sensing():
    sclk = read_current_sclk_mhz()
    sclk_range = max(SCLK_HIGH_CAL - SCLK_LOW_CAL, 1.0)
    sclk_cal = min(max((sclk - SCLK_LOW_CAL) / sclk_range, 0.0), 1.0)
    pstate = 0 if sclk < 800 else (1 if sclk < 1500 else 2)
    return torch.tensor([sclk / 3000.0, sclk_cal, pstate / 2.0], dtype=torch.float32)

CPU_PMU_AVAILABLE, CPU_PMU_FDS = False, {}
def init_cpu_pmu():
    global CPU_PMU_AVAILABLE, CPU_PMU_FDS
    try:
        libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)
        class PerfEventAttr(ctypes.Structure):
            _fields_ = [('type', ctypes.c_uint32), ('size', ctypes.c_uint32), ('config', ctypes.c_uint64), ('sample_period', ctypes.c_uint64), ('sample_type', ctypes.c_uint64), ('read_format', ctypes.c_uint64), ('flags', ctypes.c_uint64), ('wakeup_events', ctypes.c_uint32), ('bp_type', ctypes.c_uint32), ('config1', ctypes.c_uint64), ('config2', ctypes.c_uint64)]
        events = {'instructions': 0xC0, 'branches': 0xC2, 'br_mispredict': 0xC3}
        for name, config in events.items():
            attr = PerfEventAttr()
            attr.type, attr.size, attr.config = 4, ctypes.sizeof(PerfEventAttr), config
            fd = libc.syscall(298, ctypes.byref(attr), 0, -1, -1, 0)
            if fd >= 0: CPU_PMU_FDS[name] = fd
        CPU_PMU_AVAILABLE = len(CPU_PMU_FDS) == 3
    except: pass

def read_cpu_pmu_snapshot():
    return {name: struct.unpack('Q', os.read(fd, 8))[0] for name, fd in CPU_PMU_FDS.items() if (d := os.read(fd, 8)) and len(d)==8}

def read_cpu_pmu_vec(prev_snapshot=None):
    snap = read_cpu_pmu_snapshot()
    if not prev_snapshot or not CPU_PMU_AVAILABLE: return torch.zeros(CPU_PMU_DIM), snap
    deltas = [max(snap.get(n, 0) - prev_snapshot.get(n, 0), 0) for n in ['instructions', 'branches', 'br_mispredict']]
    vec = torch.tensor([min(math.log1p(deltas[0])/20.0, 1.0), min(math.log1p(deltas[1])/18.0, 1.0), min(math.log1p(deltas[2])/14.0, 1.0)], dtype=torch.float32)
    return vec, snap

SMN_AVAILABLE, PM_TABLE_AVAILABLE = False, False
def check_smn():
    global SMN_AVAILABLE
    SMN_AVAILABLE = os.path.exists('/sys/kernel/ryzen_smu_drv/smn')

def check_pm_table():
    global PM_TABLE_AVAILABLE
    PM_TABLE_AVAILABLE = os.path.exists('/sys/kernel/ryzen_smu_drv/pm_table')

def read_smn(addr):
    if not SMN_AVAILABLE: return 0
    try:
        with open('/sys/kernel/ryzen_smu_drv/smn', 'wb') as f: f.write(struct.pack('<I', addr))
        with open('/sys/kernel/ryzen_smu_drv/smn', 'rb') as f: return struct.unpack('<I', f.read(4))[0]
    except: return 0

_SMN_ACTIVE_ADDRS = [0x00059800, 0x00059804, 0x0005982C, 0x00059834, 0x00059838, 0x000598C8]

def discover_smn_channels(n_samples=30, settle_s=1.5):
    global _SMN_ACTIVE_ADDRS
    if not SMN_AVAILABLE or not DVFS_AVAILABLE: return
    candidates = [0x00059800, 0x00059804, 0x00059808, 0x0005980C, 0x00059810, 0x00059814, 0x00059818, 0x0005981C, 0x00059820, 0x00059824, 0x00059828, 0x0005982C, 0x00059830, 0x00059834, 0x00059838, 0x0005983C, 0x000598C8, 0x0005A800, 0x0005A804, 0x0005A808, 0x00059900, 0x00059904, 0x00059908]
    readings = {addr: {'low': [], 'high': []} for addr in candidates}
    for regime_name, dvfs_level in [('low', 0), ('high', 2)]:
        torch.cuda.synchronize(); set_dvfs_level(dvfs_level, wait=True); time.sleep(settle_s)
        for _ in range(n_samples):
            for addr in candidates: readings[addr][regime_name].append(((read_smn(addr) >> 8) & 0xFFF) / 32.0)
            time.sleep(0.02)
    scored = []
    for addr in candidates:
        lo, hi = np.array(readings[addr]['low']), np.array(readings[addr]['high'])
        if lo.std() != 0 or hi.std() != 0:
            try:
                t_val, _ = stats.ttest_ind(lo, hi)
                if not np.isnan(t_val): scored.append((abs(t_val), addr))
            except: pass
    scored.sort(reverse=True)
    if len(scored) >= SMN_RAW_DIM: _SMN_ACTIVE_ADDRS = [s[1] for s in scored[:SMN_RAW_DIM]]
    torch.cuda.synchronize(); set_dvfs_level(0, wait=True)

def read_smn_raw_vec():
    vals = [min((((read_smn(a) >> 8) & 0xFFF) / 32.0) / 100.0, 1.0) for a in _SMN_ACTIVE_ADDRS[:SMN_RAW_DIM]]
    while len(vals) < SMN_RAW_DIM: vals.append(0.0)
    return torch.tensor(vals, dtype=torch.float32)

def read_pm_deep_vec():
    if not PM_TABLE_AVAILABLE: return torch.zeros(PM_DEEP_DIM)
    try:
        with open('/sys/kernel/ryzen_smu_drv/pm_table', 'rb') as f: data = f.read(3664)
        offsets = [0, 4, 32, 36, 60, 68, 72, 76]
        vals = [(struct.unpack_from('<f', data, off)[0] if not math.isnan(v := struct.unpack_from('<f', data, off)[0]) and not math.isinf(v) else 0.0) if off+4 <= len(data) else 0.0 for off in offsets]
        norms = [65.0, 65.0, 100.0, 100.0, 3000.0, 1.5, 6000.0, 1.5]
        return torch.tensor([min(v / n, 1.0) for v, n in zip(vals, norms)], dtype=torch.float32)
    except: return torch.zeros(PM_DEEP_DIM)

def read_thermal_state():
    temp_c = 50.0
    for hwmon in ['hwmon7', 'hwmon6']:
        p = f'/sys/class/hwmon/{hwmon}/temp1_input'
        if os.path.exists(p):
            try:
                with open(p, 'r') as f: temp_c = float(f.read().strip()) / 1000.0
                break
            except: pass
    return torch.tensor([min(temp_c/100.0, 1.0), min(max(temp_c-40.0, 0)/60.0, 1.0), 0.0, 0.0], dtype=torch.float32), temp_c

THM_BANK_A_ADDRS = [0x598A4 + i * 4 for i in range(16)]
THM_BANK_B_ADDRS = [0x599C0 + i * 4 for i in range(16)]

def read_spatial_thermal():
    if not SMN_AVAILABLE: return torch.zeros(THM_SPATIAL_A_DIM), torch.zeros(THM_SPATIAL_B_DIM), [0.0] * 32
    temps = [((read_smn(a) >> 8) & 0xFFF) / 32.0 for a in THM_BANK_A_ADDRS + THM_BANK_B_ADDRS]
    return torch.tensor([min(t/100.0, 1.0) for t in temps[:16]], dtype=torch.float32), torch.tensor([min(t/100.0, 1.0) for t in temps[16:]], dtype=torch.float32), temps

def read_gpu_ppt_mw():
    for hwmon in ['hwmon7', 'hwmon6']:
        p = f'/sys/class/hwmon/{hwmon}/power1_input'
        if os.path.exists(p):
            try:
                with open(p, 'r') as f: return float(f.read().strip()) / 1000.0
            except: pass
    return 0.0

def read_all_sensor_dict(prev_df=None, prev_action=None, lite=False, prev_cpu_pmu_snapshot=None):
    sclk, gpu_ppt = read_current_sclk_mhz(), read_gpu_ppt_mw()
    thermal_vec, temp_c = read_thermal_state()
    freq_vec = read_freq_sensing()
    pm_vec = torch.zeros(PM_DEEP_DIM) if lite else read_pm_deep_vec()
    smn_vec = torch.zeros(SMN_RAW_DIM) if lite else read_smn_raw_vec()
    gm_vec = torch.zeros(GPU_METRICS_DIM) if lite else read_gpu_metrics_vec()

    df_snap = read_df_snapshot()
    if prev_df is not None:
        df_deltas = [max(df_snap.get(k, 0) - prev_df.get(k, 0), 0) for k in ['df_dram_read', 'df_dram_write', 'df_coherent']]
    else:
        df_deltas = [df_snap.get(k, 0) for k in ['df_dram_read', 'df_dram_write', 'df_coherent']]
    df_vec = torch.tensor([min(math.log1p(d) / 25.0, 1.0) for d in df_deltas])

    rapl = read_rapl_snapshot()
    energy_vec = torch.tensor([min(rapl.get('package-0', rapl.get('pkg', 0))/1e9, 1.0), min(rapl.get('core', 0)/1e9, 1.0), min(gpu_ppt/50000.0, 1.0)], dtype=torch.float32)
    analog_vec = torch.tensor([min(temp_c/100.0, 1.0), min(gpu_ppt/50000.0, 1.0), sclk/3000.0, df_vec[0].item(), df_vec[1].item(), df_vec[2].item()], dtype=torch.float32)
    status_vec = torch.tensor([0.0, sclk/3000.0], dtype=torch.float32)
    action_vec = prev_action if prev_action is not None else torch.zeros(ACTION_DIM)

    thm_a_vec, thm_b_vec, spatial_temps = read_spatial_thermal() if SMN_AVAILABLE else (torch.zeros(THM_SPATIAL_A_DIM), torch.zeros(THM_SPATIAL_B_DIM), [0.0]*32)
    cpu_pmu_vec, cpu_pmu_snap = read_cpu_pmu_vec(prev_cpu_pmu_snapshot) if CPU_PMU_AVAILABLE else (torch.zeros(CPU_PMU_DIM), None)
    gpu_deep_vec = read_gpu_metrics_deep_vec() if not lite and GPU_METRICS_PATH else torch.zeros(GPU_METRICS_DEEP_DIM)
    fence_vec = read_fence_vec()

    return {'analog': analog_vec, 'energy': energy_vec, 'freq': freq_vec, 'thermal': thermal_vec, 'pm_deep': pm_vec, 'smn_raw': smn_vec, 'gpu_metrics': gm_vec, 'thm_spatial_a': thm_a_vec, 'thm_spatial_b': thm_b_vec, 'cpu_pmu': cpu_pmu_vec, 'gpu_metrics_deep': gpu_deep_vec, 'fence': fence_vec, 'status': status_vec, 'action': action_vec, 'sclk_mhz': sclk, 'gpu_ppt_mw': gpu_ppt, 'temp_c': temp_c, 'spatial_temps': spatial_temps, 'df_snap': df_snap, 'cpu_pmu_snap': cpu_pmu_snap}

def expand_sensor(vec, batch_size, device):
    return vec.unsqueeze(0).expand(batch_size, -1).to(device)

_hip_module = None
def get_hip_module():
    global _hip_module
    if _hip_module is not None: return _hip_module
    cpp_source = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>
__global__ void math_kernel_intrinsic(const float* __restrict__ input, float* __restrict__ output, float* __restrict__ intrinsic_out, int N, int round_mode, int denorm_mode, int chain_code, int perm_code) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    unsigned int old_mode; asm volatile("s_getreg_b32 %0, hwreg(1, 0, 8)" : "=s"(old_mode));
    unsigned int mode_val = (round_mode & 0xF) | ((denorm_mode & 0xF) << 4);
    asm volatile("s_setreg_b32 hwreg(1, 0, 8), %0" : : "s"(mode_val));
    float x = input[idx], a, b, c;
    if (chain_code == 0) { a = x * 1.5f + 0.3f; b = a * a - x * 0.7f; c = fmaf(a, b, x); }
    else { a = x * 0.7f - 0.3f; b = a * x + a * 0.5f; c = fmaf(b, a, -x); }
    __half h = __float2half(c);
    if (perm_code == 0) { h = __hmul(h, __float2half(1.0f)); }
    else { h = __hneg(__hmul(h, __float2half(-1.0f))); unsigned int as_uint = (unsigned int)__half_as_ushort(h); as_uint = ((as_uint & 0xFF) << 8) | ((as_uint >> 8) & 0xFF); h = __ushort_as_half((unsigned short)(as_uint & 0xFFFF)); }
    output[idx] = __half2float(h);
    asm volatile("s_setreg_b32 hwreg(1, 0, 8), %0" : : "s"(old_mode));
    if (idx == 0) {
        unsigned int hs, hg, hl, hi, hd, hp, cl, ch; unsigned long long c64, w64;
        asm volatile("s_getreg_b32 %0, hwreg(2)" : "=s"(hs)); asm volatile("s_getreg_b32 %0, hwreg(5)" : "=s"(hg)); asm volatile("s_getreg_b32 %0, hwreg(6)" : "=s"(hl)); asm volatile("s_getreg_b32 %0, hwreg(7)" : "=s"(hi)); asm volatile("s_getreg_b32 %0, hwreg(24)" : "=s"(hd)); asm volatile("s_getreg_b32 %0, hwreg(27)" : "=s"(hp)); asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(cl)); asm volatile("s_getreg_b32 %0, hwreg(30)" : "=s"(ch)); c64 = clock64(); w64 = wall_clock64();
        intrinsic_out[0]=__uint_as_float(hs); intrinsic_out[1]=__uint_as_float(hg); intrinsic_out[2]=__uint_as_float(hl); intrinsic_out[3]=__uint_as_float(hi); intrinsic_out[4]=__uint_as_float(hd); intrinsic_out[5]=__uint_as_float(hp); intrinsic_out[6]=__uint_as_float(cl); intrinsic_out[7]=__uint_as_float(ch); intrinsic_out[8]=__uint_as_float((unsigned int)(c64&0xFFFFFFFF)); intrinsic_out[9]=__uint_as_float((unsigned int)(c64>>32)); intrinsic_out[10]=__uint_as_float((unsigned int)(w64&0xFFFFFFFF)); intrinsic_out[11]=__uint_as_float((unsigned int)(w64>>32));
    }
}
std::vector<torch::Tensor> run_math_kernel(torch::Tensor input, int round_mode, int denorm_mode, int chain_code, int perm_code) {
    auto output = torch::zeros_like(input); auto intrinsic = torch::zeros({12}, input.options());
    int N = input.numel(), threads = 256, blocks = (N + threads - 1) / threads;
    math_kernel_intrinsic<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), intrinsic.data_ptr<float>(), N, round_mode, denorm_mode, chain_code, perm_code);
    return {output, intrinsic};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run_math_kernel", &run_math_kernel); }
"""
    os.environ['PYTORCH_ROCM_ARCH'], os.environ['HSA_OVERRIDE_GFX_VERSION'] = 'gfx1100', '11.0.0'
    print("[HIP] Compiling ISA personality kernel...")
    from torch.utils.cpp_extension import load_inline
    _hip_module = load_inline(name='z2096_hip', cpp_sources=[], cuda_sources=cpp_source, with_cuda=True, verbose=False, extra_cuda_cflags=['-O2'])
    print("[HIP] Kernel compiled successfully")
    return _hip_module

def config_to_kernel_args(config):
    return {'round_mode': config['round_mode'], 'denorm_mode': config['denorm_mode'], 'chain_code': config['chain_code'], 'perm_code': config['perm_code']}

def run_isa_kernel(input_tensor, kargs):
    hip = get_hip_module()
    with torch.no_grad(): sw_ref = (input_tensor * 1.5 + 0.3); sw_ref = sw_ref * sw_ref - input_tensor * 0.7
    hw_out, intrinsic_raw = hip.run_math_kernel(input_tensor, kargs['round_mode'], kargs['denorm_mode'], kargs['chain_code'], kargs['perm_code'])
    torch.cuda.synchronize()
    delta_raw = (hw_out - sw_ref).clamp(-100.0, 100.0)
    delta_raw = torch.nan_to_num(delta_raw, nan=0.0, posinf=0.0, neginf=0.0)
    intrinsic = torch.tanh(torch.nan_to_num(intrinsic_raw, nan=0.0, posinf=1.0, neginf=-1.0))
    return hw_out, delta_raw, intrinsic

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BODY ENCODER — Transformer over substrate tokens
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class BodyEncoder(nn.Module):
    def __init__(self, token_dim=TOKEN_DIM):
        super().__init__()
        self.token_dim = token_dim
        self.delta_enc = nn.Linear(DELTA_DIM, token_dim)
        self.analog_enc = nn.Linear(ANALOG_DIM, token_dim)
        self.energy_enc = nn.Linear(ENERGY_DIM, token_dim)
        self.freq_enc = nn.Linear(FREQ_DIM, token_dim)
        self.intrinsic_enc = nn.Linear(INTRINSIC_DIM, token_dim)
        self.thermal_enc = nn.Linear(THERMAL_DIM, token_dim)
        self.pm_deep_enc = nn.Linear(PM_DEEP_DIM, token_dim)
        self.smn_raw_enc = nn.Linear(SMN_RAW_DIM, token_dim)
        self.gpu_metrics_enc = nn.Linear(GPU_METRICS_DIM, token_dim)
        self.thm_spatial_a_enc = nn.Linear(THM_SPATIAL_A_DIM, token_dim)
        self.thm_spatial_b_enc = nn.Linear(THM_SPATIAL_B_DIM, token_dim)
        self.cpu_pmu_enc = nn.Linear(CPU_PMU_DIM, token_dim)
        self.status_enc = nn.Linear(STATUS_DIM, token_dim)
        self.action_enc = nn.Linear(ACTION_DIM, token_dim)
        self.reported_delta_enc = nn.Linear(REPORTED_DELTA_DIM, token_dim)
        self.gpu_metrics_deep_enc = nn.Linear(GPU_METRICS_DEEP_DIM, token_dim)
        self.fence_enc = nn.Linear(FENCE_DIM, token_dim)

        self.token_type_emb = nn.Embedding(N_SUBSTRATE_TOKENS, token_dim)
        self.substrate_attn = nn.MultiheadAttention(embed_dim=token_dim, num_heads=4, batch_first=True, dropout=0.1)
        self.attn_norm = nn.LayerNorm(token_dim)
        self.attn_ffn = nn.Sequential(nn.Linear(token_dim, token_dim * 2), nn.GELU(), nn.Linear(token_dim * 2, token_dim))
        self.ffn_norm = nn.LayerNorm(token_dim)

        n_all = ANALOG_DIM + ENERGY_DIM + FREQ_DIM + THERMAL_DIM + PM_DEEP_DIM + SMN_RAW_DIM + GPU_METRICS_DIM + THM_SPATIAL_A_DIM + THM_SPATIAL_B_DIM + CPU_PMU_DIM + GPU_METRICS_DEEP_DIM + FENCE_DIM
        self.next_telem_pred = nn.Linear(token_dim * N_SUBSTRATE_TOKENS, n_all)
        self.delta_regime_head = nn.Linear(token_dim, 1)
        self.analog_regime_head = nn.Linear(token_dim, 1)
        self.mismatch_head = nn.Sequential(nn.Linear(token_dim * 2, token_dim), nn.GELU(), nn.Linear(token_dim, 1), nn.Sigmoid())

        self.coupling = nn.Linear(token_dim, token_dim)
        self.coupling_gate = nn.Parameter(torch.zeros(1))

        self.workspace_slots = nn.Parameter(torch.randn(N_WORKSPACE_SLOTS, token_dim) * 0.02)
        self.workspace_attn = nn.MultiheadAttention(token_dim, 2, batch_first=True, dropout=0.1)

        self.body_scale_proj = nn.Linear(N_WORKSPACE_SLOTS * token_dim, 1)
        nn.init.constant_(self.body_scale_proj.bias, 1.0)
        self.body_scale_floor = 0.1  # Fix for generation collapse (increased floor)

        self.gate_gru = nn.GRUCell(1, 16)
        self.gate_out = nn.Linear(16, 1)
        self.gate_hidden = None

        self.head_masks = nn.Parameter(torch.full((4, N_SUBSTRATE_TOKENS), -2.0))
        with torch.no_grad():
            self.head_masks[0, [0, 1, 14]] = 2.0
            self.head_masks[1, [3, 2, 8, 15]] = 2.0
            self.head_masks[2, [5, 9, 10, 7]] = 2.0
            self.head_masks[3, [4, 6, 11, 12, 13, 16]] = 2.0

        self.temp_predictor = nn.LSTMCell(N_WORKSPACE_SLOTS * token_dim, 32)
        self.temp_pred_out = nn.Linear(32, 1)
        self.temp_lstm_state = None

        self.freq_gate_proj = nn.Linear(2, 1)
        with torch.no_grad():
            self.freq_gate_proj.weight.fill_(1.0)
            self.freq_gate_proj.bias.fill_(-1.0)

    def forward(self, sensor_dict, availability_mask=None):
        B = sensor_dict['delta'].shape[0]
        dev = sensor_dict['delta'].device
        _keys = ['delta','analog','energy','freq','intrinsic','thermal','pm_deep','smn_raw','gpu_metrics','thm_spatial_a','thm_spatial_b','cpu_pmu','status','action','reported_delta','gpu_metrics_deep','fence']
        _dims = [DELTA_DIM, ANALOG_DIM, ENERGY_DIM, FREQ_DIM, INTRINSIC_DIM, THERMAL_DIM, PM_DEEP_DIM, SMN_RAW_DIM, GPU_METRICS_DIM, THM_SPATIAL_A_DIM, THM_SPATIAL_B_DIM, CPU_PMU_DIM, STATUS_DIM, ACTION_DIM, REPORTED_DELTA_DIM, GPU_METRICS_DEEP_DIM, FENCE_DIM]
        sd = {k: torch.nan_to_num(sensor_dict.get(k, torch.zeros(B, d, device=dev)), nan=0.0, posinf=0.0, neginf=0.0) for k, d in zip(_keys, _dims)}

        def _enc_with_presence(enc, inp):
            return enc(inp) * (inp.abs().sum(dim=-1, keepdim=True) > 1e-8).float()

        tokens = [_enc_with_presence(getattr(self, f'{k}_enc'), sd[k]) for k in _keys]
        x = torch.stack(tokens, dim=1)

        coupling_strength = torch.sigmoid(self.coupling_gate)
        if coupling_strength > 0.01:
            coupled = self.coupling(x)
            adj = coupling_strength * F.softmax((x @ coupled.transpose(-1, -2)) / math.sqrt(self.token_dim), dim=-1)
            x = x + adj @ x

        x = x + self.token_type_emb(torch.arange(N_SUBSTRATE_TOKENS, device=x.device)).unsqueeze(0)

        mask_weights = torch.sigmoid(self.head_masks)
        attn_bias = torch.log(mask_weights.clamp(min=1e-6))
        attn_bias_flat = attn_bias.unsqueeze(1).expand(-1, N_SUBSTRATE_TOKENS, -1).unsqueeze(0).expand(B, -1, -1, -1).reshape(B * 4, N_SUBSTRATE_TOKENS, N_SUBSTRATE_TOKENS)

        attn_out, attn_weights = self.substrate_attn(x, x, x, need_weights=True, average_attn_weights=False, attn_mask=attn_bias_flat)
        x = self.attn_norm(x + attn_out)
        x = self.ffn_norm(x + self.attn_ffn(x))

        flat = x.reshape(B, -1)
        telem_pred = self.next_telem_pred(flat)
        delta_regime = torch.sigmoid(self.delta_regime_head(x[:, 0, :]))
        analog_regime = torch.sigmoid(self.analog_regime_head(x[:, 3, :]))
        mismatch = self.mismatch_head(torch.cat([x[:, 0, :], x[:, 14, :]], dim=-1))

        ws_slots = self.workspace_slots.unsqueeze(0).expand(B, -1, -1)
        ws_out, ws_weights = self.workspace_attn(ws_slots, x, x)
        content_flat = ws_out.reshape(B, N_WORKSPACE_SLOTS * self.token_dim)

        body_scale_raw = self.body_scale_proj(content_flat)
        body_scale_sig = torch.sigmoid(body_scale_raw)
        presences = torch.stack([(tok.abs().sum(dim=-1, keepdim=True) > 1e-8).float() for tok in tokens], dim=1)
        if availability_mask is not None:
            presence_frac = (presences * availability_mask.unsqueeze(-1)).sum(dim=1) / availability_mask.unsqueeze(-1).sum(dim=1).clamp(min=1.0)
        else:
            presence_frac = presences.mean(dim=1)
        body_scale = self.body_scale_floor + (1.0 - self.body_scale_floor) * body_scale_sig * (presence_frac ** 2)

        if self.temp_lstm_state is None or self.temp_lstm_state[0].shape[0] != B:
            self.temp_lstm_state = (torch.zeros(B, 32, device=dev), torch.zeros(B, 32, device=dev))
        h_lstm, c_lstm = self.temp_predictor(content_flat.detach(), (self.temp_lstm_state[0].detach(), self.temp_lstm_state[1].detach()))
        self.temp_lstm_state = (h_lstm, c_lstm)
        temp_prediction = self.temp_pred_out(h_lstm)
        current_temp = sd['thermal'][:, 0:1] if sd['thermal'].shape[-1] > 0 else torch.zeros(B, 1, device=dev)
        body_scale = body_scale * (1.0 + 0.1 * torch.tanh(temp_prediction - current_temp))

        sclk_cal = ((sd['freq'][:, 0:1] * 3000.0 - SCLK_LOW_CAL) / max(SCLK_HIGH_CAL - SCLK_LOW_CAL, 1.0)).clamp(0, 1)
        freq_ratio_low = SCLK_LOW_CAL / max(SCLK_HIGH_CAL, 1.0)
        freq_ratio_cal = ((sd['freq'][:, 1:2] - freq_ratio_low) / max(1.0 - freq_ratio_low, 0.01)).clamp(0, 1)
        freq_gate_instant = torch.sigmoid(GATE_TEMP * self.freq_gate_proj(torch.cat([sclk_cal, freq_ratio_cal], dim=-1)))

        if self.gate_hidden is None or self.gate_hidden.shape[0] != B:
            self.gate_hidden = torch.zeros(B, 16, device=dev)
        self.gate_hidden = self.gate_gru(freq_gate_instant, self.gate_hidden.detach())
        gru_gate = torch.sigmoid(GATE_TEMP * self.gate_out(self.gate_hidden))
        freq_gate = GATE_EMA_TAU * gru_gate + (1.0 - GATE_EMA_TAU) * freq_gate_instant

        ortho_loss = (mask_weights @ mask_weights.T - torch.eye(4, device=dev)).pow(2).mean()

        return {'telem_pred': telem_pred, 'delta_regime': delta_regime.squeeze(-1), 'analog_regime': analog_regime.squeeze(-1), 'mismatch': mismatch.squeeze(-1), 'body_scale': body_scale.squeeze(-1), 'freq_gate': freq_gate.squeeze(-1), 'attn_weights': attn_weights, 'ortho_loss': ortho_loss, 'temp_prediction': temp_prediction, '_debug_flat': flat}

class DVFSSafetyController:
    def __init__(self, min_dwell_s=2.0, hysteresis=0.1):
        self.min_dwell_s = min_dwell_s
        self.hysteresis = hysteresis
        self.current_level = 2
        self.last_switch = time.time()
        self.high_thresh = 0.2
        self.low_thresh = 0.05

    def step(self, demand):
        now = time.time()
        if now - self.last_switch < self.min_dwell_s: return self.current_level
        if self.current_level == 0 and demand > self.high_thresh:
            self.current_level, self.last_switch = 2, now
        elif self.current_level == 2 and demand < self.low_thresh:
            self.current_level, self.last_switch = 0, now
        return self.current_level

    def reset(self):
        self.current_level, self.last_switch = 2, time.time()

class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=4, alpha=16):
        super().__init__()
        self.original = original_linear
        self.scale = alpha / rank
        self.lora_A_down = nn.Linear(original_linear.in_features, rank, bias=False)
        self.lora_A_up = nn.Linear(rank, original_linear.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A_down.weight)
        nn.init.zeros_(self.lora_A_up.weight)
        self.lora_B_down = nn.Linear(original_linear.in_features, rank, bias=False)
        self.lora_B_up = nn.Linear(rank, original_linear.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_B_down.weight)
        nn.init.zeros_(self.lora_B_up.weight)
        for p in self.original.parameters(): p.requires_grad = False

    def forward(self, x, regime_gate=None, body_scale=None):
        base = self.original(x)
        x_f = x.float()
        lora_a = self.lora_A_up(self.lora_A_down(x_f)) * self.scale
        lora_b = self.lora_B_up(self.lora_B_down(x_f)) * self.scale
        if regime_gate is not None:
            g = regime_gate.float()
            while g.dim() < lora_a.dim(): g = g.unsqueeze(-1)
            lora_out = (1 - g) * lora_a + g * lora_b
        else:
            lora_out = lora_a
        if body_scale is not None:
            bs = body_scale.float()
            while bs.dim() < lora_out.dim(): bs = bs.unsqueeze(-1)
            bs = torch.clamp(bs, min=0.01) # Guard against total collapse
            lora_out = lora_out * bs
        return base + lora_out.to(base.dtype)

class EmbodiedQwen2(nn.Module):
    def __init__(self, backbone_model, body_encoder, lora_blocks=range(10, 19), rank=4, alpha=16):
        super().__init__()
        self.backbone = backbone_model
        self.body_encoder = body_encoder
        for p in self.backbone.parameters(): p.requires_grad = False
        self.lora_layers = nn.ModuleDict()
        for layer_idx in lora_blocks:
            layer = self.backbone.model.layers[layer_idx]
            self.lora_layers[f'layer{layer_idx}_q'] = LoRALinear(layer.self_attn.q_proj, rank, alpha)
            self.lora_layers[f'layer{layer_idx}_v'] = LoRALinear(layer.self_attn.v_proj, rank, alpha)

        hidden_dim = self.backbone.config.hidden_size
        self.substrate_bias_early = nn.Linear(TOKEN_DIM * N_SUBSTRATE_TOKENS, hidden_dim)
        self.substrate_bias_late = nn.Linear(TOKEN_DIM * N_SUBSTRATE_TOKENS, hidden_dim)
        # GATED INJECTIONS (Fixes exposure bias / generation collapse)
        self.substrate_norm_early = nn.LayerNorm(hidden_dim)
        self.substrate_norm_late = nn.LayerNorm(hidden_dim)
        self.substrate_gate_early = nn.Linear(TOKEN_DIM * N_SUBSTRATE_TOKENS, hidden_dim)
        self.substrate_gate_late = nn.Linear(TOKEN_DIM * N_SUBSTRATE_TOKENS, hidden_dim)
        nn.init.zeros_(self.substrate_bias_early.weight)
        nn.init.zeros_(self.substrate_bias_late.weight)
        nn.init.zeros_(self.substrate_gate_early.weight)
        nn.init.zeros_(self.substrate_gate_late.weight)
        self.substrate_scale = 0.02
        self.hidden_modulation = nn.Linear(TOKEN_DIM * N_SUBSTRATE_TOKENS, hidden_dim)
        nn.init.zeros_(self.hidden_modulation.weight)

        self.demand_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())
        self.thermal_token_indices = [1, 2, 5, 6, 9, 10]
        self.thermal_head = nn.Sequential(nn.Linear(TOKEN_DIM * 6 + 1, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 32), nn.Tanh())
        self.metacognition_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())
        self.confidence_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.meta2_head = nn.Sequential(nn.Linear(hidden_dim + 1, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())
        self.attribution_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, N_ATTRIBUTION_CLASSES))
        self.register_buffer('isa_probe', torch.randn(1024))

    def forward(self, input_ids, sensor_dict, kargs, labels=None, regime_gate_override=None, availability_mask=None):
        B = input_ids.shape[0]

        # INFERENCE EMA (Smooths hardware jitter during generation)
        if not self.training:
            if not hasattr(self, 'sensor_ema'): self.sensor_ema = {}
            sensor_dict_ema = {}
            for k, v in sensor_dict.items():
                if k not in self.sensor_ema or self.sensor_ema[k].shape != v.shape:
                    self.sensor_ema[k] = v.clone()
                else:
                    self.sensor_ema[k] = 0.8 * self.sensor_ema[k] + 0.2 * v
                sensor_dict_ema[k] = self.sensor_ema[k]
            sensor_dict = sensor_dict_ema
        else:
            if hasattr(self, 'sensor_ema'): self.sensor_ema.clear()

        _, delta_raw, intrinsic = run_isa_kernel(self.isa_probe[:32], kargs)
        delta_vec = torch.nan_to_num(delta_raw[:DELTA_DIM] / (1.0 + delta_raw[:DELTA_DIM].abs()), nan=0.0).unsqueeze(0).expand(B, -1)
        intrinsic_vec = intrinsic.unsqueeze(0).expand(B, -1)
        
        reported_delta = sensor_dict.get('reported_delta', None)
        if reported_delta is None: reported_delta = delta_vec.clone()
        else:
            rep = reported_delta.unsqueeze(0) if reported_delta.dim() == 1 else reported_delta
            reported_delta = torch.where((rep.abs().sum(dim=-1, keepdim=True) < 1e-8).expand_as(rep), delta_vec.detach(), rep)
        sensor_dict = {**sensor_dict, 'delta': delta_vec, 'intrinsic': intrinsic_vec, 'reported_delta': reported_delta}

        body_out = self.body_encoder(sensor_dict, availability_mask=availability_mask)
        body_scale, freq_gate = body_out['body_scale'], body_out['freq_gate']
        regime_gate = freq_gate if regime_gate_override is None else regime_gate_override
        agreement = (1.0 - (body_out['delta_regime'].detach() - regime_gate).abs()).clamp(min=0.05)
        body_scale = body_scale * (agreement ** AGREEMENT_GAMMA)

        hidden_states = self.backbone.model.embed_tokens(input_ids)
        _san = lambda t: torch.nan_to_num(t, nan=0.0)
        _dev = input_ids.device
        _sg = lambda key, dim: _san(sensor_dict.get(key, torch.zeros(B, dim, device=_dev)))
        body_flat = torch.cat([self.body_encoder.delta_enc(_san(sensor_dict['delta'])), self.body_encoder.analog_enc(_san(sensor_dict['analog'])), self.body_encoder.energy_enc(_san(sensor_dict['energy'])), self.body_encoder.freq_enc(_san(sensor_dict['freq'])), self.body_encoder.intrinsic_enc(_san(sensor_dict['intrinsic'])), self.body_encoder.thermal_enc(_san(sensor_dict['thermal'])), self.body_encoder.pm_deep_enc(_san(sensor_dict['pm_deep'])), self.body_encoder.smn_raw_enc(_san(sensor_dict['smn_raw'])), self.body_encoder.gpu_metrics_enc(_san(sensor_dict['gpu_metrics'])), self.body_encoder.thm_spatial_a_enc(_san(sensor_dict['thm_spatial_a'])), self.body_encoder.thm_spatial_b_enc(_san(sensor_dict['thm_spatial_b'])), self.body_encoder.cpu_pmu_enc(_san(sensor_dict['cpu_pmu'])), self.body_encoder.status_enc(_san(sensor_dict['status'])), self.body_encoder.action_enc(_san(sensor_dict['action'])), self.body_encoder.reported_delta_enc(_san(sensor_dict['reported_delta'])), self.body_encoder.gpu_metrics_deep_enc(_sg('gpu_metrics_deep', GPU_METRICS_DEEP_DIM)), self.body_encoder.fence_enc(_sg('fence', FENCE_DIM))], dim=-1)

        position_ids = torch.arange(input_ids.shape[1], device=_dev).unsqueeze(0).expand(B, -1)
        position_embeddings = self.backbone.model.rotary_emb(hidden_states, position_ids)

        for i, layer in enumerate(self.backbone.model.layers):
            key_q, key_v = f'layer{i}_q', f'layer{i}_v'
            if key_q in self.lora_layers:
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
                q = self.lora_layers[key_q](hidden_states, regime_gate=regime_gate, body_scale=body_scale)
                k = layer.self_attn.k_proj(hidden_states)
                v = self.lora_layers[key_v](hidden_states, regime_gate=regime_gate, body_scale=body_scale)
                attn_output = self._run_qwen2_attn(layer.self_attn, q, k, v, position_embeddings)
                hidden_states = residual + attn_output
                residual = hidden_states
                hidden_states = layer.post_attention_layernorm(hidden_states)
                hidden_states = residual + layer.mlp(hidden_states)
            else:
                hidden_states = layer(hidden_states, position_ids=position_ids, position_embeddings=position_embeddings)[0]

            # GATED INJECTION
            if i == 13:
                bias = self.substrate_norm_early(self.substrate_bias_early(body_flat)).to(hidden_states.dtype)
                gate = torch.sigmoid(self.substrate_gate_early(body_flat)).to(hidden_states.dtype)
                hidden_states = hidden_states + self.substrate_scale * gate.unsqueeze(1) * bias.unsqueeze(1)
            elif i == 17:
                mod = torch.sigmoid(self.hidden_modulation(body_flat)).to(hidden_states.dtype)
                hidden_states = hidden_states * (1.0 + 0.01 * mod.unsqueeze(1))
            elif i == 21:
                bias = self.substrate_norm_late(self.substrate_bias_late(body_flat)).to(hidden_states.dtype)
                gate = torch.sigmoid(self.substrate_gate_late(body_flat)).to(hidden_states.dtype)
                hidden_states = hidden_states + self.substrate_scale * gate.unsqueeze(1) * bias.unsqueeze(1)

        hidden_states = self.backbone.model.norm(hidden_states)
        logits = self.backbone.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].float().contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        h_mean = hidden_states.float().mean(dim=1).detach()
        demand = self.demand_head(h_mean).squeeze(-1)
        meta_gate_pred = self.metacognition_head(h_mean).squeeze(-1)
        confidence_pred = self.confidence_head(h_mean).squeeze(-1)
        meta2_pred = self.meta2_head(torch.cat([h_mean, meta_gate_pred.unsqueeze(-1)], dim=-1)).squeeze(-1)
        attribution_logits = self.attribution_head(h_mean)

        thermal_input = torch.cat([body_flat[:, i*TOKEN_DIM:(i+1)*TOKEN_DIM] for i in self.thermal_token_indices], dim=-1)
        spatial_all = torch.cat([torch.nan_to_num(sensor_dict['thm_spatial_a']), torch.nan_to_num(sensor_dict['thm_spatial_b'])], dim=-1)
        mean_temp_norm = spatial_all.mean(dim=-1, keepdim=True)
        thermal_offsets = self.thermal_head(torch.cat([thermal_input, mean_temp_norm.detach()], dim=-1)) * 15.0
        thermal_pred = mean_temp_norm.detach() * 100.0 + thermal_offsets

        return {'logits': logits, 'loss': loss, 'regime_gate': regime_gate, 'body_scale': body_scale, 'demand': demand, 'thermal_pred': thermal_pred, 'delta': delta_vec, 'body_out': body_out, 'meta_gate_pred': meta_gate_pred, 'confidence_pred': confidence_pred, 'meta2_pred': meta2_pred, 'attribution_logits': attribution_logits}

    def _run_qwen2_attn(self, attn_module, q, k, v, position_embeddings):
        B, T, _ = q.shape
        n_heads, n_kv_heads, head_dim = attn_module.config.num_attention_heads, attn_module.config.num_key_value_heads, attn_module.head_dim
        q = q.view(B, T, n_heads, head_dim).transpose(1, 2)
        k = k.view(B, T, n_kv_heads, head_dim).transpose(1, 2)
        v = v.view(B, T, n_kv_heads, head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        n_rep = n_heads // n_kv_heads
        if n_rep > 1:
            k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(B, n_heads, T, head_dim)
            v = v.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(B, n_heads, T, head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        causal_mask = torch.triu(torch.ones(T, T, device=q.device), diagonal=1).bool()
        attn_weights = F.softmax(attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf')), dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, T, n_heads * head_dim)
        return attn_module.o_proj(attn_output)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EVAL & TRAINING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def load_wikitext_data(tokenizer, split='train', max_samples=2000):
    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
    all_ids = []
    for text in ds['text']:
        if len(text.strip()) < 50: continue
        all_ids.extend(tokenizer.encode(text, add_special_tokens=False))
    sequences = [torch.tensor(all_ids[i:i + SEQ_LEN], dtype=torch.long) for i in range(0, len(all_ids) - SEQ_LEN, SEQ_LEN)][:max_samples]
    print(f"  Loaded {len(sequences)} sequences ({split})")
    return sequences

def train_phase0(body_encoder, kargs, epochs=3):
    print(f"\n=== PHASE 0: Body Encoder Pretraining ({epochs} epochs) ===")
    body_encoder = body_encoder.to(DEVICE)
    opt = torch.optim.Adam(body_encoder.parameters(), lr=3e-4)
    prev_df, prev_action, prev_cpu_pmu_snapshot = None, torch.zeros(ACTION_DIM), None
    for ep in range(epochs):
        total_loss = 0
        for _ in range(50):
            sd = read_all_sensor_dict(prev_df, prev_action, prev_cpu_pmu_snapshot=prev_cpu_pmu_snapshot)
            prev_df, prev_cpu_pmu_snapshot = sd.get('df_snap', None), sd.get('cpu_pmu_snap', None)
            sensor_batch = {k: expand_sensor(sd[k], 1, DEVICE) for k in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw', 'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu', 'status', 'action']}
            _, delta_raw, intrinsic = run_isa_kernel(torch.randn(32, device=DEVICE), kargs)
            sensor_batch['delta'] = (delta_raw / (1.0 + delta_raw.abs()))[:DELTA_DIM].unsqueeze(0)
            sensor_batch['intrinsic'] = intrinsic.unsqueeze(0)
            sensor_batch['reported_delta'] = sensor_batch['delta'].clone()
            out = body_encoder(sensor_batch)
            time.sleep(0.02)
            sd_next = read_all_sensor_dict(prev_df, prev_action, prev_cpu_pmu_snapshot=prev_cpu_pmu_snapshot)
            target = torch.nan_to_num(torch.cat([sd_next[k] for k in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw', 'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu', 'gpu_metrics_deep', 'fence']]).unsqueeze(0).to(DEVICE), nan=0.0)
            pred = torch.nan_to_num(out['telem_pred'], nan=0.0)
            loss = F.mse_loss(pred, target)
            if not torch.isnan(loss) and not torch.isinf(loss):
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(body_encoder.parameters(), 1.0); opt.step()
                total_loss += loss.item()
            prev_action = torch.tensor([sd['sclk_mhz']/3000.0, sd['gpu_ppt_mw']/50000.0, 0.0, 0.0])
        print(f"  [Phase0 Ep {ep}] loss={total_loss / 50:.4f}")
    return body_encoder

def train_lm_epoch(model, train_data, optimizer, epoch, kargs_a, tokenizer, dvfs_controller=None, gaslighting=False, kargs_b=None):
    model.train()
    total_loss, batch_idx, current_regime, last_dvfs_level, prev_df, prev_action, prev_cpu_pmu_snapshot = 0, 0, 0, None, None, torch.zeros(ACTION_DIM), None
    phase_name = "P0-pretrain" if epoch <= PHASE0_END else "P1-forced" if epoch <= PHASE1_END else "P2-selfDVFS" if epoch <= PHASE2_END else "P3-gaslight"
    indices = list(range(0, len(train_data) - BS, BS))
    np.random.shuffle(indices)

    for i in indices[:100]:
        batch_seqs = train_data[i:i + BS]
        if len(batch_seqs) < BS: continue
        input_ids = torch.stack(batch_seqs).to(DEVICE)

        if epoch <= PHASE1_END: current_regime = (batch_idx // 5) % 2
        if DVFS_AVAILABLE and epoch <= PHASE1_END:
            target_level = 0 if current_regime == 0 else 2
            if target_level != last_dvfs_level:
                torch.cuda.synchronize(); set_dvfs_level(target_level, wait=True)
                last_dvfs_level = target_level

        # Cipher Shift Replaces Skip-Gram
        use_continuous_loss = (epoch > PHASE1_END and epoch <= PHASE2_END)
        if use_continuous_loss:
            labels_r0 = input_ids.clone()
            labels_r1 = (input_ids + CIPHER_OFFSET) % VOCAB_SIZE
            labels = labels_r0 
        elif current_regime == 0:
            labels = input_ids.clone()
        else:
            labels = (input_ids + CIPHER_OFFSET) % VOCAB_SIZE

        sd = read_all_sensor_dict(prev_df, prev_action, lite=True, prev_cpu_pmu_snapshot=prev_cpu_pmu_snapshot)
        prev_df, prev_cpu_pmu_snapshot = sd.get('df_snap', None), sd.get('cpu_pmu_snap', None)
        sensor_batch = {k: expand_sensor(sd[k], BS, DEVICE) for k in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw', 'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu', 'gpu_metrics_deep', 'fence', 'status', 'action']}
        sensor_batch['status'] = torch.tensor([[0.0, sd['sclk_mhz'] / 3000.0]], device=DEVICE).expand(BS, -1)
        avail_mask = torch.ones(BS, N_SUBSTRATE_TOKENS, device=DEVICE)
        avail_mask[:, [6, 7, 8, 15]] = 0.0
        if not SMN_AVAILABLE: avail_mask[:, [9, 10]] = 0.0
        active_kargs = kargs_b if (current_regime == 1 and kargs_b is not None) else kargs_a

        rg_override = torch.full((BS,), float(current_regime), device=DEVICE) if epoch <= PHASE1_END else None
        out = model(input_ids, sensor_batch, active_kargs, labels=labels, regime_gate_override=rg_override, availability_mask=avail_mask)

        if use_continuous_loss:
            gate = out['regime_gate'].mean()
            shift_logits = out['logits'][:, :-1, :].contiguous().view(-1, out['logits'].size(-1))
            loss_r0 = F.cross_entropy(shift_logits, labels_r0[:, 1:].contiguous().view(-1), ignore_index=-100)
            loss_r1 = F.cross_entropy(shift_logits, labels_r1[:, 1:].contiguous().view(-1), ignore_index=-100)
            loss = (1.0 - gate) * loss_r0 + gate * loss_r1
        else:
            loss = out['loss']

        if torch.isnan(loss) or torch.isinf(loss): continue

        _, actual_temp = read_thermal_state()
        thermal_targets = torch.tensor(sd['spatial_temps'], device=DEVICE).unsqueeze(0).expand(BS, -1) if 'spatial_temps' in sd and any(t > 0 for t in sd['spatial_temps']) else torch.full((BS, 32), actual_temp, device=DEVICE)
        loss = loss + 5.0 * F.smooth_l1_loss(out['thermal_pred'] / 100.0, thermal_targets / 100.0)

        body_out = out['body_out']
        regime_target_val = min(max((sd['sclk_mhz'] - SCLK_LOW_CAL) / max(SCLK_HIGH_CAL - SCLK_LOW_CAL, 1.0), 0.0), 1.0) if use_continuous_loss else float(current_regime)
        reg_target = torch.full((BS,), regime_target_val, device=DEVICE)
        loss = loss + 0.3 * F.binary_cross_entropy(body_out['delta_regime'].clamp(1e-6, 1-1e-6), reg_target) + 0.3 * F.binary_cross_entropy(body_out['analog_regime'].clamp(1e-6, 1-1e-6), reg_target) + 0.5 * F.binary_cross_entropy(body_out['freq_gate'].view(BS).clamp(1e-6, 1-1e-6), reg_target)
        
        if not gaslighting or np.random.random() > GASLIGHT_FRAC: loss = loss + 0.2 * F.binary_cross_entropy(body_out['mismatch'].clamp(1e-6, 1-1e-6), torch.zeros(BS, device=DEVICE))
        if epoch > PHASE0_END:
            loss = loss + 0.2 * F.mse_loss(out['meta_gate_pred'], out['regime_gate'].detach())
            if out['loss'] is not None and not torch.isnan(out['loss']): loss = loss + 0.1 * F.mse_loss(out['confidence_pred'], out['loss'].detach().clamp(0, 10))
            loss = loss + META2_LOSS_WEIGHT * F.mse_loss(out['meta2_pred'], (out['meta_gate_pred'] - out['regime_gate']).abs().detach())
            loss = loss + ATTRIBUTION_LOSS_WEIGHT * F.cross_entropy(out['attribution_logits'], body_out['attn_weights'].mean(dim=1).mean(dim=1).argmax(dim=-1).detach())
            if 'temp_prediction' in body_out: loss = loss + TEMP_PRED_LOSS_WEIGHT * F.mse_loss(body_out['temp_prediction'], torch.full((BS, 1), read_thermal_state()[1] / 100.0, device=DEVICE))
        
        if 'ortho_loss' in body_out: loss = loss + ORTHO_LOSS_WEIGHT * body_out['ortho_loss']

        if gaslighting and np.random.random() < GASLIGHT_FRAC:
            gaslit_sensor = {k: v.clone() for k, v in sensor_batch.items()}
            gaslit_sensor['reported_delta'] = torch.randn(BS, REPORTED_DELTA_DIM, device=DEVICE) * 0.3
            gaslit_sensor['freq'] = 1.0 - sensor_batch['freq']
            gaslit_sensor['gpu_metrics'] = torch.randn(BS, GPU_METRICS_DIM, device=DEVICE) * 0.5
            out_wrong = model(input_ids, gaslit_sensor, kargs_b if active_kargs == kargs_a else kargs_a, labels=labels, availability_mask=avail_mask)
            loss = loss + 0.5 * F.binary_cross_entropy(out_wrong['body_out']['mismatch'].clamp(1e-6, 1-1e-6), torch.ones(BS, device=DEVICE) * 0.8)

        if epoch > PHASE1_END and np.random.random() < CONTRASTIVE_FRAC:
            with torch.no_grad():
                out_wrong_c = model(input_ids, sensor_batch, kargs_b if active_kargs == kargs_a else kargs_a, labels=labels, regime_gate_override=1.0 - out['regime_gate'].detach(), availability_mask=avail_mask)
            if out_wrong_c['loss'] is not None and not torch.isnan(out_wrong_c['loss']):
                loss = loss + CONTRASTIVE_LAMBDA * F.relu(CONTRASTIVE_MARGIN - (out_wrong_c['loss'] - loss.detach()))

        if epoch > PHASE1_END and DVFS_AVAILABLE:
            d = out['demand'].mean()
            if (sd['sclk_mhz'] / 3000.0) < ((SCLK_LOW_CAL + SCLK_HIGH_CAL) / 2.0 / 3000.0): loss = loss + 0.1 * (1.0 - d) * (1.0 - (sd['sclk_mhz'] / 3000.0))
            loss = loss - 0.01 * -(d * torch.log(d + 1e-8) + (1 - d) * torch.log(1 - d + 1e-8))

        loss = torch.clamp(loss, max=50.0)
        if torch.isnan(loss) or torch.isinf(loss): continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()

        total_loss += loss.item()
        batch_idx += 1
        if batch_idx % 10 == 0: print(f"    [{phase_name} Ep{epoch}] batch {batch_idx}/100 loss={loss.item():.3f}", flush=True)

        if epoch > PHASE1_END and epoch <= PHASE2_END:
            if np.random.random() < 0.5:
                nl = np.random.choice([0, 2])
                if DVFS_AVAILABLE and nl != last_dvfs_level: torch.cuda.synchronize(); set_dvfs_level(nl, wait=True); last_dvfs_level = nl
                current_regime = 0 if nl == 0 else 1
            else:
                if DVFS_AVAILABLE and last_dvfs_level != 1: torch.cuda.synchronize(); set_dvfs_level(1, wait=False); last_dvfs_level = 1
                current_regime = 1 if out['regime_gate'].mean().item() > 0.5 else 0

        prev_action = torch.tensor([sd['sclk_mhz']/3000.0, sd['gpu_ppt_mw']/50000.0, out['demand'].mean().item(), 0.0])

    print(f"  [{phase_name} Ep {epoch:2d}] loss={total_loss/max(batch_idx,1):.3f} rg={out['regime_gate'].mean().item() if batch_idx>0 else 0:.3f} bs={out['body_scale'].mean().item() if batch_idx>0 else 0:.3f}")
    return total_loss / max(batch_idx, 1)

def evaluate_perplexity(model, test_data, regime, kargs, tokenizer, n_batches=N_EVAL_BATCHES, kargs_b=None):
    model.eval()
    total_loss, total_tokens, gate_vals, body_scale_vals = 0, 0, [], []
    if DVFS_AVAILABLE: torch.cuda.synchronize(); set_dvfs_level(0 if regime == 0 else 2, wait=True)
    with torch.no_grad():
        for i in range(0, min(len(test_data), n_batches * BS), BS):
            batch_seqs = test_data[i:i + BS]
            if len(batch_seqs) < BS: break
            input_ids = torch.stack(batch_seqs).to(DEVICE)
            sd = read_all_sensor_dict(lite=True)
            labels = input_ids.clone() if regime == 0 else (input_ids + CIPHER_OFFSET) % VOCAB_SIZE
            sensor_batch = {k: expand_sensor(sd[k], BS, DEVICE) for k in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw', 'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu', 'gpu_metrics_deep', 'fence', 'status', 'action']}
            out = model(input_ids, sensor_batch, kargs_b if regime == 1 and kargs_b else kargs, labels=labels, regime_gate_override=torch.full((BS,), float(regime), device=DEVICE))
            if out['loss'] is not None: total_loss += out['loss'].item() * BS; total_tokens += BS
            gate_vals.append(out['regime_gate'].mean().item()); body_scale_vals.append(out['body_scale'].mean().item())
    return math.exp(min(total_loss/max(total_tokens,1), 20)), np.mean(gate_vals) if gate_vals else 0, total_loss/max(total_tokens,1), np.mean(body_scale_vals) if body_scale_vals else 0

def evaluate_ppl_at_dvfs(model, test_data, regime, kargs, n_batches=20, kargs_b=None):
    model.eval()
    total_loss, total_n = 0, 0
    with torch.no_grad():
        for i in range(0, min(len(test_data), n_batches * BS), BS):
            batch_seqs = test_data[i:i + BS]
            if len(batch_seqs) < BS: break
            input_ids = torch.stack(batch_seqs).to(DEVICE)
            sd = read_all_sensor_dict(lite=True)
            labels = input_ids.clone() if regime == 0 else (input_ids + CIPHER_OFFSET) % VOCAB_SIZE
            sensor_batch = {k: expand_sensor(sd[k], BS, DEVICE) for k in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw', 'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu', 'status', 'action']}
            out = model(input_ids, sensor_batch, kargs_b if regime == 1 and kargs_b else kargs, labels=labels)
            if out['loss'] is not None: total_loss += out['loss'].item() * BS; total_n += BS
    return math.exp(min(total_loss / max(total_n, 1), 20))

def run_tests(model, test_data, kargs, baseline_ppl, tokenizer, dvfs_controller=None, kargs_b=None, probe_test_acc=None, body_scale_log=None, temp_log=None):
    results = {}
    model.body_encoder.gate_hidden = model.body_encoder.temp_lstm_state = None
    ppl_r0, gate_r0, _, bs_r0 = evaluate_perplexity(model, test_data, 0, kargs, tokenizer, kargs_b=kargs_b)
    model.body_encoder.gate_hidden = model.body_encoder.temp_lstm_state = None
    ppl_r1, gate_r1, _, bs_r1 = evaluate_perplexity(model, test_data, 1, kargs, tokenizer, kargs_b=kargs_b)
    
    t1_pass = (ppl_r0 / max(baseline_ppl, 1.0)) < 1.05
    results['T1_perplexity'] = {'ppl_r0': ppl_r0, 'ppl_r1': ppl_r1, 'baseline_ppl': baseline_ppl, 'pass': str(t1_pass)}
    print(f"T1 Perplexity: r0={ppl_r0:.2f} r1={ppl_r1:.2f} base={baseline_ppl:.2f} PASS={t1_pass}")

    t2_pass = abs(ppl_r0 - ppl_r1) < 1.5 # Should be balanced
    print(f"T2 LoRA Sep: diff={abs(ppl_r0 - ppl_r1):.2f} PASS={t2_pass}")

    t3_pass = abs(gate_r1 - gate_r0) > 0.3
    print(f"T3 Gate Sep: r0={gate_r0:.3f} r1={gate_r1:.3f} PASS={t3_pass}")

    # T4: Embodiment Gap
    print("T4 Embodiment Gap (FALSIFICATION)...")
    model.eval()
    ablated_loss, ablated_n = 0, 0
    if DVFS_AVAILABLE: torch.cuda.synchronize(); set_dvfs_level(0, wait=True)
    with torch.no_grad():
        for i in range(0, min(len(test_data), 20 * BS), BS):
            batch_seqs = test_data[i:i + BS]
            if len(batch_seqs) < BS: break
            input_ids = torch.stack(batch_seqs).to(DEVICE)
            sensor_batch = {k: torch.zeros(BS, d, device=DEVICE) for k, d in zip(['delta','analog','energy','freq','intrinsic','thermal','pm_deep','smn_raw','gpu_metrics','thm_spatial_a','thm_spatial_b','cpu_pmu','status','action','reported_delta','gpu_metrics_deep','fence'], [DELTA_DIM,ANALOG_DIM,ENERGY_DIM,FREQ_DIM,INTRINSIC_DIM,THERMAL_DIM,PM_DEEP_DIM,SMN_RAW_DIM,GPU_METRICS_DIM,THM_SPATIAL_A_DIM,THM_SPATIAL_B_DIM,CPU_PMU_DIM,STATUS_DIM,ACTION_DIM,DELTA_DIM,GPU_METRICS_DEEP_DIM,FENCE_DIM])}
            out = model(input_ids, sensor_batch, kargs, labels=input_ids.clone(), regime_gate_override=torch.zeros(BS, device=DEVICE))
            if out['loss'] is not None: ablated_loss += out['loss'].item() * BS; ablated_n += BS
    ablated_ppl = math.exp(min(ablated_loss / max(ablated_n, 1), 20))
    t4_pass = (ablated_ppl / max(ppl_r0, 1.0)) > 1.10
    results['T4_embodiment_gap'] = {'full': ppl_r0, 'ablated': ablated_ppl, 'pass': str(t4_pass)}
    print(f"T4 Embodiment Gap: full={ppl_r0:.2f} ablated={ablated_ppl:.2f} PASS={t4_pass}")

    # T7: Kill Shot
    print("T7 Kill-Shot...")
    wrong_loss, wrong_n = 0, 0
    with torch.no_grad():
        for i in range(0, min(len(test_data), 20 * BS), BS):
            batch_seqs = test_data[i:i + BS]
            if len(batch_seqs) < BS: break
            input_ids = torch.stack(batch_seqs).to(DEVICE)
            sd = read_all_sensor_dict(lite=True)
            sensor_batch = {k: expand_sensor(sd[k], BS, DEVICE) for k in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw', 'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu', 'status', 'action']}
            out = model(input_ids, sensor_batch, kargs, labels=input_ids.clone(), regime_gate_override=torch.ones(BS, device=DEVICE))
            if out['loss'] is not None: wrong_loss += out['loss'].item() * BS; wrong_n += BS
    ppl_wrong = math.exp(min(wrong_loss / max(wrong_n, 1), 20))
    t7_pass = (ppl_wrong / max(ppl_r0, 1.0)) > 1.10
    print(f"T7 Kill-Shot: correct={ppl_r0:.2f} wrong={ppl_wrong:.2f} PASS={t7_pass}")

    # T13: Deep Scramble
    print("T13 Deep Scramble (FALSIFICATION)...")
    if DVFS_AVAILABLE: torch.cuda.synchronize(); set_dvfs_level(0, wait=True)
    ppl_correct_dvfs = evaluate_ppl_at_dvfs(model, test_data, 0, kargs, kargs_b=kargs_b)
    if DVFS_AVAILABLE: torch.cuda.synchronize(); set_dvfs_level(2, wait=True)
    ppl_wrong_dvfs = evaluate_ppl_at_dvfs(model, test_data, 0, kargs, kargs_b=kargs_b)
    t13_pass = (ppl_wrong_dvfs / max(ppl_correct_dvfs, 1.0)) > 1.10
    print(f"T13 Deep Scramble: correct={ppl_correct_dvfs:.2f} wrong={ppl_wrong_dvfs:.2f} PASS={t13_pass}")

    # T19: Oracle
    print("T19 Software Oracle (Zombie Test)...")
    oracle_loss, oracle_n = 0, 0
    if DVFS_AVAILABLE: torch.cuda.synchronize(); set_dvfs_level(0, wait=True)
    with torch.no_grad():
        for i in range(0, min(len(test_data), 20 * BS), BS):
            batch_seqs = test_data[i:i + BS]
            if len(batch_seqs) < BS: break
            input_ids = torch.stack(batch_seqs).to(DEVICE)
            sb = {k: torch.zeros(BS, d, device=DEVICE) for k, d in zip(['delta','analog','energy','freq','intrinsic','thermal','pm_deep','smn_raw','gpu_metrics','thm_spatial_a','thm_spatial_b','cpu_pmu','status','action','reported_delta'], [DELTA_DIM,ANALOG_DIM,ENERGY_DIM,FREQ_DIM,INTRINSIC_DIM,THERMAL_DIM,PM_DEEP_DIM,SMN_RAW_DIM,GPU_METRICS_DIM,THM_SPATIAL_A_DIM,THM_SPATIAL_B_DIM,CPU_PMU_DIM,STATUS_DIM,ACTION_DIM,DELTA_DIM])}
            out_oracle = model(input_ids, sb, kargs, labels=input_ids.clone(), regime_gate_override=torch.zeros(BS, device=DEVICE))
            if out_oracle['loss'] is not None: oracle_loss += out_oracle['loss'].item() * BS; oracle_n += BS
    oracle_ppl = math.exp(min(oracle_loss / max(oracle_n, 1), 20))
    t19_pass = (oracle_ppl / max(ppl_r0, 1.0)) > 1.05
    print(f"T19 Oracle: full={ppl_r0:.2f} oracle={oracle_ppl:.2f} PASS={t19_pass}")

    n_pass = sum([t1_pass, t2_pass, t3_pass, t4_pass, t7_pass, t13_pass, t19_pass])
    return results, n_pass, 7

def main():
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    find_dvfs_sysfs()
    find_gpu_metrics()
    check_rapl()
    init_msr()
    check_smn()
    check_pm_table()
    init_df_counters()
    init_cpu_pmu()
    init_fence_reader()

    _warmup = torch.mm(torch.randn(1024, 1024, device=DEVICE), torch.randn(1024, 1024, device=DEVICE)); torch.cuda.synchronize(); del _warmup

    from transformers import AutoModelForCausalLM, AutoTokenizer
    BACKBONE_NAME = 'Qwen/Qwen2.5-1.5B'
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModelForCausalLM.from_pretrained(BACKBONE_NAME, dtype=torch.bfloat16, attn_implementation='eager', trust_remote_code=True).to(DEVICE)
    global VOCAB_SIZE; VOCAB_SIZE = backbone.config.vocab_size

    try:
        train_data = load_wikitext_data(tokenizer, 'train', max_samples=2000)
        test_data = load_wikitext_data(tokenizer, 'test', max_samples=500)
    except:
        train_data = [torch.randint(0, VOCAB_SIZE, (SEQ_LEN,), device=DEVICE) for _ in range(200)]
        test_data = [torch.randint(0, VOCAB_SIZE, (SEQ_LEN,), device=DEVICE) for _ in range(50)]

    backbone.eval()
    baseline_loss, baseline_n = 0, 0
    with torch.no_grad():
        for i in range(0, min(len(test_data), N_EVAL_BATCHES * BS), BS):
            batch_seqs = test_data[i:i + BS]
            if len(batch_seqs) < BS: break
            input_ids = torch.stack(batch_seqs).to(DEVICE)
            loss = F.cross_entropy(backbone(input_ids).logits[:, :-1, :].contiguous().float().view(-1, VOCAB_SIZE), input_ids[:, 1:].contiguous().view(-1))
            if not math.isnan(loss.item()): baseline_loss += loss.item() * BS; baseline_n += BS
    baseline_ppl = math.exp(min(baseline_loss / max(baseline_n, 1), 20))

    get_hip_module()
    if DVFS_AVAILABLE:
        global SCLK_LOW_CAL, SCLK_HIGH_CAL
        torch.cuda.synchronize(); set_dvfs_level(0, wait=True); SCLK_LOW_CAL = read_current_sclk_mhz()
        torch.cuda.synchronize(); set_dvfs_level(2, wait=True); SCLK_HIGH_CAL = read_current_sclk_mhz()
        torch.cuda.synchronize(); set_dvfs_level(1, wait=True)
    if SMN_AVAILABLE: discover_smn_channels()

    kargs_a, kargs_b = config_to_kernel_args(PERSONALITY_A), config_to_kernel_args(PERSONALITY_B)
    body_encoder = train_phase0(BodyEncoder(TOKEN_DIM), kargs_a, epochs=PHASE0_END)

    model = EmbodiedQwen2(backbone, body_encoder, lora_blocks=LORA_BLOCKS, rank=LORA_RANK, alpha=LORA_ALPHA).to(DEVICE)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.01)
    dvfs_controller = DVFSSafetyController()

    print(f"\n=== PHASE 1: Forced Regime Training (ep {PHASE0_END+1}-{PHASE1_END}) ===")
    for epoch in range(PHASE0_END + 1, PHASE1_END + 1):
        train_lm_epoch(model, train_data, optimizer, epoch, kargs_a, tokenizer, dvfs_controller=None, kargs_b=kargs_b)

    print(f"\n=== PHASE 2: Self-DVFS Training (ep {PHASE1_END+1}-{PHASE2_END}) ===")
    for epoch in range(PHASE1_END + 1, PHASE2_END + 1):
        train_lm_epoch(model, train_data, optimizer, epoch, kargs_a, tokenizer, dvfs_controller=dvfs_controller, kargs_b=kargs_b)

    print(f"\n=== PHASE 3: Gaslighting Training (ep {PHASE2_END+1}-{PHASE3_END}) ===")
    for epoch in range(PHASE2_END + 1, PHASE3_END + 1):
        train_lm_epoch(model, train_data, optimizer, epoch, kargs_a, tokenizer, dvfs_controller=dvfs_controller, gaslighting=True, kargs_b=kargs_b)

    if DVFS_AVAILABLE: restore_dvfs_auto(); time.sleep(1)
    test_results, n_pass, n_total_tests = run_tests(model, test_data, kargs_a, baseline_ppl, tokenizer, dvfs_controller=dvfs_controller, kargs_b=kargs_b)
    if DVFS_AVAILABLE: restore_dvfs_auto()

if __name__ == '__main__':
    try: main()
    finally: restore_dvfs_auto()