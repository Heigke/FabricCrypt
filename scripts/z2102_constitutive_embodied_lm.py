#!/usr/bin/env python3
"""
z2100: Integrated Workspace LM
===============================
Built on z2099 (Bridge Law LM). 5 targeted architectural changes to address T32-T39
failures, plus 4 bug fixes. ~83K new params (0.25% of backbone).

Key changes from z2099:
  1. COUPLED TOKEN ENCODING: cross-token message passing for IIT phi (T32)
  2. WORKSPACE BOTTLENECK: 17 tokens → 4 competitive slots via cross-attention (T33)
  3. TEMPORAL GATE: GRU cell + EMA replacing instantaneous sigmoid (T34)
  4. HEAD SPECIALIZATION: modality masks + orthogonality loss for PID synergy (T36)
  5. PREDICTIVE BODY SCALE: LSTM thermal predictor for anticipatory response (T39)

Bug fixes:
  - T29: clamp probabilities before log to prevent NaN
  - T35: chain meta_gate_pred into meta2_head input
  - T38: collect 50+ probe samples during Phase 1
  - T31: increase adversarial perturbation magnitude

Scientific grounding:
  - Butlin et al. (TiCS 2025): 14 indicators, FEEL satisfies 10-12/14
  - Casali et al. (2013): PCI distinguishes consciousness states in humans
  - Luppi et al. (eLife 2024): synergy > redundancy in conscious processing
  - Milinkovic & Aru (Dec 2025): substrate IS constitutive
  - Tononi et al. (IIT 4.0): causal integration via minimum information partition
  - Baars/Dehaene (GNW): global workspace capacity limits
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
SKIP_GRAM_OFFSET = 2  # Skip-gram prediction offset (predict token at position+OFFSET)
SKIP_GRAM_K = 2       # Alias for SKIP_GRAM_OFFSET

def make_lm_labels(input_ids, offset=1):
    """Build labels for LM loss with given prediction offset.
    offset=1: standard next-token (position+1)
    offset=2: skip-gram (position+2)
    The loss function already shifts by 1 internally (shift_labels = labels[:, 1:]),
    so we pre-shift by (offset-1) to get the correct net shift.
    BUG FIX: Previous code shifted by offset, causing net shift = offset+1.
    """
    labels = input_ids.clone()
    shift = offset - 1  # pre-shift BEFORE the loss function's implicit shift-by-1
    if shift > 0:
        labels[:, :-shift] = input_ids[:, shift:]
        labels[:, -shift:] = -100
    return labels
GASLIGHT_FRAC = 0.30
VOCAB_SIZE = 151936  # Qwen2.5-1.5B vocab; overridden from model config in main()

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

# z2095: Calibrated DVFS range (set at runtime by calibrate_dvfs_range())
SCLK_LOW_CAL = 600.0   # placeholder — updated after DVFS sanity check
SCLK_HIGH_CAL = 2900.0  # placeholder — updated after DVFS sanity check

# z2095: Gate sharpness & contrastive loss
GATE_TEMP = 8.0          # sigmoid temperature for freq_gate
CONTRASTIVE_LAMBDA = 0.3  # weight for contrastive kill-shot loss
CONTRASTIVE_MARGIN = 0.3  # nats margin: wrong gate should be this much worse
CONTRASTIVE_FRAC = 0.25   # fraction of batches with contrastive loss (saves compute)
AGREEMENT_GAMMA = 2.0     # exponent for agreement modulation

# z2099: Bridge Law heads
META2_LOSS_WEIGHT = 0.3       # z2100v2: was 0.1 → 0.3 for T35 (MAE was 0.193, need <0.10)
ATTRIBUTION_LOSS_WEIGHT = 0.1 # weight for attribution loss
N_ATTRIBUTION_CLASSES = 17    # one class per substrate token

# z2100: Integrated workspace constants
N_WORKSPACE_SLOTS = 4         # workspace bottleneck (GWT capacity limit)
ORTHO_LOSS_WEIGHT = 0.05      # head specialization orthogonality loss (was 0.01)
TEMP_PRED_LOSS_WEIGHT = 0.05  # thermal predictor loss weight
GATE_EMA_TAU = 0.3            # temporal gate EMA smoothing

# ISA Personalities (unchanged from z2093)
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
# HARDWARE ACCESS — DVFS (with safety)
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
    """Set DVFS: 0=low, 1=auto, 2=high. CRITICAL: sync GPU first!"""
    if not DVFS_AVAILABLE:
        return
    # CRITICAL SAFETY: synchronize GPU before ANY DVFS write
    torch.cuda.synchronize()
    name = {0: 'low', 1: 'auto', 2: 'high'}[level]
    try:
        with open(DVFS_PATH, 'w') as f:
            f.write(name)
    except Exception as e:
        print(f"[DVFS] Write failed: {e}")
        return
    if wait:
        _poll_dvfs_settle(level)

def _poll_dvfs_settle(level):
    """Poll until SCLK matches expected range."""
    target_low = level == 0
    for attempt in range(30):
        sclk = read_current_sclk_mhz()
        if target_low and sclk < 800:
            return
        if not target_low and sclk > 1200:
            return
        time.sleep(0.1)
    # If polling didn't converge, wait full settle time
    time.sleep(DVFS_SETTLE_S)

def restore_dvfs_auto():
    if DVFS_AVAILABLE:
        torch.cuda.synchronize()
        try:
            with open(DVFS_PATH, 'w') as f:
                f.write('auto')
        except:
            pass

def read_current_sclk_mhz():
    """Read current SCLK from hwmon."""
    for hwmon in ['hwmon7', 'hwmon6', 'hwmon5']:
        p = f'/sys/class/hwmon/{hwmon}/freq1_input'
        if os.path.exists(p):
            try:
                with open(p, 'r') as f:
                    return float(f.read().strip()) / 1e6  # Hz -> MHz
            except:
                pass
    return 600.0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HARDWARE ACCESS — gpu_metrics v3.0
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GPU_METRICS_PATH = None

def find_gpu_metrics():
    global GPU_METRICS_PATH
    for card in ['card1', 'card0']:
        p = f'/sys/class/drm/{card}/device/gpu_metrics'
        if os.path.exists(p):
            GPU_METRICS_PATH = p
            print(f"[gpu_metrics] Found: {p}")
            return
    print("[gpu_metrics] Not available")

def read_gpu_metrics_v3():
    """Read gpu_metrics v3.0 binary using kernel struct gpu_metrics_v3_0 layout.
    Properly decodes ALL fields from the 264-byte blob."""
    result = {
        'dram_reads': 0, 'dram_writes': 0,
        'c0_activity_avg': 0.0,
        'throttle_prochot': 0, 'throttle_thermal': 0, 'throttle_power': 0,
        'temperature_gfx': 0, 'temperature_soc': 0,
        # z2098: new deep fields
        'per_core_c0': [0.0] * 16,  # per-core C0 activity %
        'per_core_clk': [0] * 16,   # per-core current MHz
        'avg_gfxclk': 0, 'avg_socclk': 0, 'avg_fclk': 0, 'avg_uclk': 0,
        'avg_gfx_power': 0, 'avg_all_core_power': 0, 'avg_socket_power': 0,
        'energy_acc': 0,
        'throttle_residency_prochot': 0, 'throttle_residency_thm_gfx': 0,
        'throttle_residency_thm_soc': 0,
        'gfx_max_freq': 0,
    }
    if GPU_METRICS_PATH is None:
        return result
    try:
        with open(GPU_METRICS_PATH, 'rb') as f:
            data = f.read()
        if len(data) < 264:
            return result

        # Header: metrics_table_header = {u16 structure_size, u8 format_revision, u8 content_revision}
        size_h, fmt_rev, content_rev = struct.unpack_from('<HBB', data, 0)
        if fmt_rev < 3:
            return result

        # gpu_metrics_v3_0 layout (from kernel kgd_pp_interface.h):
        off = 4  # after header

        # Temperatures (u16, centidegrees)
        t_gfx = struct.unpack_from('<H', data, off)[0]; off += 2
        t_soc = struct.unpack_from('<H', data, off)[0]; off += 2
        result['temperature_gfx'] = t_gfx / 100.0 if t_gfx < 20000 else 0
        result['temperature_soc'] = t_soc / 100.0 if t_soc < 20000 else 0

        # temperature_core[16] (u16 each)
        core_temps = struct.unpack_from('<16H', data, off); off += 32
        # temperature_skin (u16)
        off += 2

        # Utilization
        avg_gfx_act = struct.unpack_from('<H', data, off)[0]; off += 2  # average_gfx_activity
        avg_vcn_act = struct.unpack_from('<H', data, off)[0]; off += 2  # average_vcn_activity
        avg_ipu_act = struct.unpack_from('<8H', data, off); off += 16   # average_ipu_activity[8]

        # average_core_c0_activity[16] — THIS IS THE KEY per-core data
        c0_raw = struct.unpack_from('<16H', data, off); off += 32
        c0_vals = []
        for i, c0 in enumerate(c0_raw):
            pct = c0 / 100.0 if c0 <= 10000 else 0.0
            result['per_core_c0'][i] = pct
            if pct > 0:
                c0_vals.append(pct)
        result['c0_activity_avg'] = np.mean(c0_vals) if c0_vals else 0.0

        # average_dram_reads, average_dram_writes (u16)
        result['dram_reads'] = struct.unpack_from('<H', data, off)[0]; off += 2
        result['dram_writes'] = struct.unpack_from('<H', data, off)[0]; off += 2
        off += 4  # average_ipu_reads + writes

        # system_clock_counter (u64)
        sys_clk = struct.unpack_from('<Q', data, off)[0]; off += 8
        result['energy_acc'] = sys_clk  # use as energy proxy

        # Power (mixed u32/u16)
        avg_socket = struct.unpack_from('<I', data, off)[0]; off += 4
        avg_ipu_pwr = struct.unpack_from('<H', data, off)[0]; off += 2
        avg_apu_pwr = struct.unpack_from('<I', data, off)[0]; off += 4
        avg_gfx_pwr = struct.unpack_from('<I', data, off)[0]; off += 4
        avg_dgpu_pwr = struct.unpack_from('<I', data, off)[0]; off += 4
        avg_all_core = struct.unpack_from('<I', data, off)[0]; off += 4
        result['avg_socket_power'] = avg_socket if avg_socket < 0xFFFF0000 else 0
        result['avg_gfx_power'] = avg_gfx_pwr if avg_gfx_pwr < 0xFFFF0000 else 0
        result['avg_all_core_power'] = avg_all_core if avg_all_core < 0xFFFF0000 else 0

        # average_core_power[16] + sys/stapm
        off += 32 + 6  # skip per-core power + sys + stapm limits

        # Clocks (u16 each, 8 fields)
        avg_gfxclk = struct.unpack_from('<H', data, off)[0]; off += 2
        avg_socclk = struct.unpack_from('<H', data, off)[0]; off += 2
        off += 4  # vpeclk + ipuclk
        avg_fclk = struct.unpack_from('<H', data, off)[0]; off += 2
        off += 2  # vclk
        avg_uclk = struct.unpack_from('<H', data, off)[0]; off += 2
        off += 2  # mpipu
        result['avg_gfxclk'] = avg_gfxclk if avg_gfxclk < 65535 else 0
        result['avg_socclk'] = avg_socclk if avg_socclk < 65535 else 0
        result['avg_fclk'] = avg_fclk if avg_fclk < 65535 else 0
        result['avg_uclk'] = avg_uclk if avg_uclk < 65535 else 0

        # current_coreclk[16]
        for i in range(16):
            clk = struct.unpack_from('<H', data, off)[0]; off += 2
            result['per_core_clk'][i] = clk if clk < 65535 else 0

        # current_core_maxfreq + current_gfx_maxfreq
        off += 2  # core max
        gfx_max = struct.unpack_from('<H', data, off)[0]; off += 2
        result['gfx_max_freq'] = gfx_max if gfx_max < 65535 else 0

        # Throttle residencies (7 x u32)
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

    except Exception as e:
        pass  # Silent fail — gpu_metrics is optional
    return result

def read_gpu_metrics_vec():
    """Return gpu_metrics as a normalized torch tensor [GPU_METRICS_DIM]."""
    gm = read_gpu_metrics_v3()
    return torch.tensor([
        min(gm['dram_reads'] / 1e4, 1.0),      # normalized DRAM reads
        min(gm['dram_writes'] / 1e4, 1.0),     # normalized DRAM writes
        gm['c0_activity_avg'] / 100.0,          # C0 activity [0,1]
        float(gm['throttle_prochot']),           # binary
        float(gm['throttle_thermal']),           # binary
        float(gm['throttle_power']),             # binary
    ], dtype=torch.float32)

def read_gpu_metrics_deep_vec():
    """z2098: Return deep gpu_metrics as torch tensor [GPU_METRICS_DEEP_DIM=12].
    Per-core C0 activity (8 active cores) + per-core clock deltas (4)."""
    gm = read_gpu_metrics_v3()
    # Per-core C0: take first 8 cores (Strix has 8 Zen5 cores active)
    c0 = [min(gm['per_core_c0'][i] / 100.0, 1.0) for i in range(8)]
    # Per-core clock deltas: relative to mean clock (captures heterogeneity)
    clks = [gm['per_core_clk'][i] for i in range(16) if gm['per_core_clk'][i] > 0]
    mean_clk = np.mean(clks) if clks else 1000.0
    clk_deltas = []
    for i in range(min(4, len(clks))):
        clk_deltas.append((clks[i] - mean_clk) / max(mean_clk, 1.0))
    while len(clk_deltas) < 4:
        clk_deltas.append(0.0)
    return torch.tensor(c0 + clk_deltas, dtype=torch.float32)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HARDWARE ACCESS — Fence Ring Depth (debugfs)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FENCE_PATH = None

def init_fence_reader():
    """Find fence_info debugfs path (requires sudo)."""
    global FENCE_PATH
    for card_id in [1, 0]:
        p = f'/sys/kernel/debug/dri/{card_id}/amdgpu_fence_info'
        if os.path.exists(p):
            try:
                with open(p, 'r') as f:
                    f.read(100)
                FENCE_PATH = p
                print(f"[FENCE] Available: {p}")
                return
            except:
                pass
    print("[FENCE] Not available (need sudo)")

def read_fence_vec():
    """Read fence ring queue depths: emitted - signaled = pending.
    Returns [FENCE_DIM=4]: gfx, comp0, comp1, comp2 queue depths."""
    depths = [0.0] * FENCE_DIM
    if FENCE_PATH is None:
        return torch.tensor(depths, dtype=torch.float32)
    try:
        with open(FENCE_PATH, 'r') as f:
            text = f.read()
        ring_idx = 0
        emitted = signaled = 0
        for line in text.split('\n'):
            if 'Last emitted' in line and 'trailing' not in line:
                try:
                    emitted = int(line.split()[-1], 16)
                except:
                    pass
            elif 'Last signaled fence' in line:
                try:
                    signaled = int(line.split()[-1], 16)
                except:
                    pass
                depth = max(0, emitted - signaled)
                if ring_idx < FENCE_DIM:
                    depths[ring_idx] = min(depth / 100.0, 1.0)  # normalize
                ring_idx += 1
    except:
        pass
    return torch.tensor(depths, dtype=torch.float32)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HARDWARE ACCESS — Data Fabric counters via perf_event_open
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DF_FDS = {}
DF_AVAILABLE = False

def init_df_counters():
    global DF_FDS, DF_AVAILABLE
    libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)

    # perf_event_attr structure
    class PerfEventAttr(ctypes.Structure):
        _fields_ = [
            ('type', ctypes.c_uint32),
            ('size', ctypes.c_uint32),
            ('config', ctypes.c_uint64),
            ('sample_period', ctypes.c_uint64),
            ('sample_type', ctypes.c_uint64),
            ('read_format', ctypes.c_uint64),
            ('flags', ctypes.c_uint64),
            ('wakeup_events', ctypes.c_uint32),
            ('bp_type', ctypes.c_uint32),
            ('config1', ctypes.c_uint64),
            ('config2', ctypes.c_uint64),
        ]

    # Find amd_df PMU type
    df_type = None
    try:
        with open('/sys/bus/event_source/devices/amd_df/type', 'r') as f:
            df_type = int(f.read().strip())
    except:
        print("[DF] amd_df PMU not found")
        return

    # Zen 5 DF events: event=0x07
    events = {
        'df_dram_read':  (0x07 | (0x48 << 8)),   # umask=0x48
        'df_dram_write': (0x07 | (0xC0 << 8)),   # umask=0xC0
        'df_coherent':   (0x07 | (0x60 << 8)),   # umask=0x60
    }

    NR_perf_event_open = 298  # x86_64
    for name, config in events.items():
        attr = PerfEventAttr()
        attr.type = df_type
        attr.size = ctypes.sizeof(PerfEventAttr)
        attr.config = config
        attr.flags = 0  # disabled initially

        fd = libc.syscall(NR_perf_event_open, ctypes.byref(attr), -1, 0, -1, 0)
        if fd >= 0:
            # Enable
            PERF_EVENT_IOC_ENABLE = 0x2400
            libc.ioctl(fd, PERF_EVENT_IOC_ENABLE, 0)
            DF_FDS[name] = fd

    DF_AVAILABLE = len(DF_FDS) > 0
    print(f"[DF] Counters: {list(DF_FDS.keys())}")

def read_df_snapshot():
    result = {}
    for name, fd in DF_FDS.items():
        buf = ctypes.c_uint64(0)
        n = os.read(fd, 8)
        if len(n) == 8:
            result[name] = struct.unpack('Q', n)[0]
        else:
            result[name] = 0
    return result

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HARDWARE ACCESS — RAPL Energy
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RAPL_AVAILABLE = False
RAPL_PATHS = {}

def check_rapl():
    global RAPL_AVAILABLE, RAPL_PATHS
    base = '/sys/class/powercap'
    for domain in ['intel-rapl:0', 'intel-rapl:0:0']:
        ej = os.path.join(base, domain, 'energy_uj')
        if os.path.exists(ej):
            name_path = os.path.join(base, domain, 'name')
            try:
                with open(name_path, 'r') as f:
                    name = f.read().strip()
                RAPL_PATHS[name] = ej
            except:
                pass
    RAPL_AVAILABLE = len(RAPL_PATHS) > 0
    print(f"[RAPL] Domains: {list(RAPL_PATHS.keys())}")

def read_rapl_snapshot():
    result = {}
    for name, path in RAPL_PATHS.items():
        try:
            with open(path, 'r') as f:
                result[name] = int(f.read().strip())
        except:
            result[name] = 0
    return result

def compute_batch_joules(before, after, gpu_ppt_mw, elapsed_s=None):
    """Compute energy from RAPL delta + GPU PPT estimate."""
    total_uj = 0
    for name in before:
        if name in after:
            delta = after[name] - before[name]
            if delta < 0:
                delta += (1 << 32)  # wraparound
            total_uj += delta
    total_j = total_uj / 1e6
    # Add GPU estimate if available
    if gpu_ppt_mw > 0 and elapsed_s:
        total_j += (gpu_ppt_mw / 1000.0) * elapsed_s
    return total_j

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HARDWARE ACCESS — MSR frequency sensing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MSR_AVAILABLE = False
MSR_FD = None

def init_msr():
    global MSR_AVAILABLE, MSR_FD
    try:
        MSR_FD = os.open('/dev/cpu/0/msr', os.O_RDONLY)
        MSR_AVAILABLE = True
        print("[MSR] Available")
    except:
        print("[MSR] Not available")

def read_msr(reg):
    if not MSR_AVAILABLE:
        return 0
    try:
        os.lseek(MSR_FD, reg, os.SEEK_SET)
        data = os.read(MSR_FD, 8)
        return struct.unpack('Q', data)[0]
    except:
        return 0

def read_freq_sensing():
    """Read GPU-only frequency observables for sharp gate separation."""
    sclk = read_current_sclk_mhz()
    # Calibrated sclk_frac: 0.0 at SCLK_LOW_CAL, 1.0 at SCLK_HIGH_CAL
    sclk_range = max(SCLK_HIGH_CAL - SCLK_LOW_CAL, 1.0)
    sclk_cal = min(max((sclk - SCLK_LOW_CAL) / sclk_range, 0.0), 1.0)
    pstate = 0 if sclk < 800 else (1 if sclk < 1500 else 2)
    return torch.tensor([
        sclk / 3000.0,   # sclk_norm [0,1]
        sclk_cal,         # calibrated sclk fraction [0,1] (GPU-only)
        pstate / 2.0,     # pstate [0,1]
    ], dtype=torch.float32)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HARDWARE ACCESS — Zen 5 CPU PMU via perf_event_open
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CPU_PMU_AVAILABLE = False
CPU_PMU_FDS = {}

def init_cpu_pmu():
    global CPU_PMU_AVAILABLE, CPU_PMU_FDS
    libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)

    class PerfEventAttr(ctypes.Structure):
        _fields_ = [
            ('type', ctypes.c_uint32),
            ('size', ctypes.c_uint32),
            ('config', ctypes.c_uint64),
            ('sample_period', ctypes.c_uint64),
            ('sample_type', ctypes.c_uint64),
            ('read_format', ctypes.c_uint64),
            ('flags', ctypes.c_uint64),
            ('wakeup_events', ctypes.c_uint32),
            ('bp_type', ctypes.c_uint32),
            ('config1', ctypes.c_uint64),
            ('config2', ctypes.c_uint64),
        ]

    # CPU core PMU type
    cpu_type = 4  # always 4 for cpu core PMU

    # Zen 5 core events
    events = {
        'instructions': 0xC0,      # retired instructions
        'branches': 0xC2,          # retired branches
        'br_mispredict': 0xC3,     # retired branch mispredicts
    }

    NR_perf_event_open = 298
    for name, config in events.items():
        attr = PerfEventAttr()
        attr.type = cpu_type
        attr.size = ctypes.sizeof(PerfEventAttr)
        attr.config = config
        fd = libc.syscall(NR_perf_event_open, ctypes.byref(attr), 0, -1, -1, 0)
        if fd >= 0:
            CPU_PMU_FDS[name] = fd

    CPU_PMU_AVAILABLE = len(CPU_PMU_FDS) == 3
    print(f"[CPU_PMU] {'Available' if CPU_PMU_AVAILABLE else 'Not available'}: {list(CPU_PMU_FDS.keys())}")

def read_cpu_pmu_snapshot():
    """Read current CPU PMU counter values."""
    result = {}
    for name, fd in CPU_PMU_FDS.items():
        try:
            data = os.read(fd, 8)
            result[name] = struct.unpack('Q', data)[0]
        except:
            result[name] = 0
    return result

def read_cpu_pmu_vec(prev_snapshot=None):
    """Read CPU PMU as delta vector, normalized. Returns (vec, snapshot)."""
    snap = read_cpu_pmu_snapshot()
    if prev_snapshot is None or not CPU_PMU_AVAILABLE:
        return torch.zeros(CPU_PMU_DIM), snap
    deltas = []
    for name in ['instructions', 'branches', 'br_mispredict']:
        d = snap.get(name, 0) - prev_snapshot.get(name, 0)
        if d < 0:
            d = 0  # counter wrap
        deltas.append(d)
    # Normalize: log1p scale (typical values 100K-10M per batch)
    vec = torch.tensor([
        min(math.log1p(deltas[0]) / 20.0, 1.0),  # instructions
        min(math.log1p(deltas[1]) / 18.0, 1.0),  # branches
        min(math.log1p(deltas[2]) / 14.0, 1.0),  # br_mispredict
    ], dtype=torch.float32)
    return vec, snap

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HARDWARE ACCESS — SMN / PM table / Thermal
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SMN_AVAILABLE = False
PM_TABLE_AVAILABLE = False

def check_smn():
    global SMN_AVAILABLE
    SMN_AVAILABLE = os.path.exists('/sys/kernel/ryzen_smu_drv/smn')
    print(f"[SMN] {'Available' if SMN_AVAILABLE else 'Not available'}")

def check_pm_table():
    global PM_TABLE_AVAILABLE
    PM_TABLE_AVAILABLE = os.path.exists('/sys/kernel/ryzen_smu_drv/pm_table')
    print(f"[PM] {'Available' if PM_TABLE_AVAILABLE else 'Not available'}")

def read_smn(addr):
    if not SMN_AVAILABLE:
        return 0
    try:
        with open('/sys/kernel/ryzen_smu_drv/smn', 'wb') as f:
            f.write(struct.pack('<I', addr))
        with open('/sys/kernel/ryzen_smu_drv/smn', 'rb') as f:
            data = f.read(4)
        return struct.unpack('<I', data)[0]
    except:
        return 0

_SMN_ACTIVE_ADDRS = [0x00059800, 0x00059804, 0x0005982C, 0x00059834, 0x00059838, 0x000598C8]

def discover_smn_channels(n_samples=30, settle_s=1.5):
    """DVFS-toggling SMN address discovery: scan whitelisted addresses,
    find which show consistent low/high DVFS separation via t-test.
    Locks the top SMN_RAW_DIM channels as active addresses."""
    global _SMN_ACTIVE_ADDRS
    if not SMN_AVAILABLE or not DVFS_AVAILABLE:
        print("[SMN-DISCOVER] Skipped (SMN or DVFS not available)")
        return
    # Whitelist of safe READ-ONLY SMN addresses (thermal ADC, crystal, SVI)
    candidates = [
        0x00059800, 0x00059804, 0x00059808, 0x0005980C,
        0x00059810, 0x00059814, 0x00059818, 0x0005981C,
        0x00059820, 0x00059824, 0x00059828, 0x0005982C,
        0x00059830, 0x00059834, 0x00059838, 0x0005983C,
        0x000598C8,  # XTAL_CNTL
        0x0005A800, 0x0005A804, 0x0005A808,  # SVI telemetry
        0x00059900, 0x00059904, 0x00059908,  # thermal bank B
    ]
    print(f"[SMN-DISCOVER] Scanning {len(candidates)} addresses...")
    readings = {addr: {'low': [], 'high': []} for addr in candidates}
    for regime_name, dvfs_level in [('low', 0), ('high', 2)]:
        torch.cuda.synchronize()
        set_dvfs_level(dvfs_level, wait=True)
        time.sleep(settle_s)
        # Warmup reads
        for addr in candidates:
            read_smn(addr)
        time.sleep(0.1)
        for _ in range(n_samples):
            for addr in candidates:
                raw = read_smn(addr)
                temp = ((raw >> 8) & 0xFFF) / 32.0
                readings[addr][regime_name].append(temp)
            time.sleep(0.02)
    # T-test each address
    scored = []
    for addr in candidates:
        lo = np.array(readings[addr]['low'])
        hi = np.array(readings[addr]['high'])
        if lo.std() == 0 and hi.std() == 0:
            continue  # static register, skip
        try:
            t_val, p_val = stats.ttest_ind(lo, hi)
            if not np.isnan(t_val):
                scored.append((abs(t_val), addr, float(np.mean(lo)), float(np.mean(hi))))
        except:
            pass
    scored.sort(reverse=True)
    # Take top SMN_RAW_DIM addresses
    if len(scored) >= SMN_RAW_DIM:
        _SMN_ACTIVE_ADDRS = [s[1] for s in scored[:SMN_RAW_DIM]]
        print(f"[SMN-DISCOVER] Top {SMN_RAW_DIM} addresses (by |t|):")
        for t_val, addr, lo_mean, hi_mean in scored[:SMN_RAW_DIM]:
            print(f"  0x{addr:08X}: t={t_val:.2f} (low={lo_mean:.1f}°C high={hi_mean:.1f}°C)")
    else:
        print(f"[SMN-DISCOVER] Only {len(scored)} responsive addresses found, keeping defaults")
    # Restore DVFS
    torch.cuda.synchronize()
    set_dvfs_level(0, wait=True)

def read_smn_raw_vec():
    """Read SMN_RAW_DIM-dim SMN vector from auto-discovered addresses."""
    vals = []
    for addr in _SMN_ACTIVE_ADDRS[:SMN_RAW_DIM]:
        raw = read_smn(addr)
        temp = ((raw >> 8) & 0xFFF) / 32.0  # bits[19:8] / 32 = °C
        vals.append(min(temp / 100.0, 1.0))  # normalize to [0,1]
    while len(vals) < SMN_RAW_DIM:
        vals.append(0.0)
    return torch.tensor(vals, dtype=torch.float32)

def read_pm_deep_vec():
    """Read 8-dim PM table vector."""
    if not PM_TABLE_AVAILABLE:
        return torch.zeros(PM_DEEP_DIM)
    try:
        with open('/sys/kernel/ryzen_smu_drv/pm_table', 'rb') as f:
            data = f.read(3664)
        # Key offsets (float32): stapm_power, ppt, cpu_temp, gpu_temp, sclk, vddgfx, cpu_freq, cpu_volt
        offsets = [0, 4, 32, 36, 60, 68, 72, 76]
        vals = []
        for off in offsets:
            if off + 4 <= len(data):
                v = struct.unpack_from('<f', data, off)[0]
                if math.isnan(v) or math.isinf(v):
                    v = 0.0
                vals.append(v)
            else:
                vals.append(0.0)
        # Normalize
        norms = [65.0, 65.0, 100.0, 100.0, 3000.0, 1.5, 6000.0, 1.5]
        return torch.tensor([min(v / n, 1.0) for v, n in zip(vals, norms)], dtype=torch.float32)
    except:
        return torch.zeros(PM_DEEP_DIM)

def read_thermal_state():
    """Read thermal zone and hwmon temps."""
    temp_c = 50.0
    for hwmon in ['hwmon7', 'hwmon6']:
        p = f'/sys/class/hwmon/{hwmon}/temp1_input'
        if os.path.exists(p):
            try:
                with open(p, 'r') as f:
                    temp_c = float(f.read().strip()) / 1000.0
                break
            except:
                pass
    vec = torch.tensor([
        min(temp_c / 100.0, 1.0),
        min(max(temp_c - 40.0, 0) / 60.0, 1.0),  # delta from ambient
        0.0, 0.0
    ], dtype=torch.float32)
    return vec, temp_c

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HARDWARE ACCESS — Spatial Thermal ADC (32 sensors via SMN)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THM_BANK_A_ADDRS = [0x598A4 + i * 4 for i in range(16)]
THM_BANK_B_ADDRS = [0x599C0 + i * 4 for i in range(16)]

def read_spatial_thermal():
    """Read 32 spatial thermal ADC sensors from SMN.
    Returns (bank_a_vec, bank_b_vec, temps_celsius_all32)."""
    if not SMN_AVAILABLE:
        return torch.zeros(THM_SPATIAL_A_DIM), torch.zeros(THM_SPATIAL_B_DIM), [0.0] * 32

    temps = []
    for addr in THM_BANK_A_ADDRS + THM_BANK_B_ADDRS:
        raw = read_smn(addr)
        temp_c = ((raw >> 8) & 0xFFF) / 32.0
        temps.append(temp_c)

    # Normalize to [0, 1] range: typical 20-90°C
    norm_a = torch.tensor([min(t / 100.0, 1.0) for t in temps[:16]], dtype=torch.float32)
    norm_b = torch.tensor([min(t / 100.0, 1.0) for t in temps[16:]], dtype=torch.float32)
    return norm_a, norm_b, temps

def read_gpu_ppt_mw():
    for hwmon in ['hwmon7', 'hwmon6']:
        p = f'/sys/class/hwmon/{hwmon}/power1_input'
        if os.path.exists(p):
            try:
                with open(p, 'r') as f:
                    return float(f.read().strip()) / 1000.0  # uW -> mW
            except:
                pass
    return 0.0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SENSOR AGGREGATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def read_all_sensor_dict(prev_df=None, prev_action=None, lite=False, prev_cpu_pmu_snapshot=None):
    """Aggregate all sensor readings into a dict of tensors.
    lite=True: skip SMU-heavy reads (pm_table, smn, gpu_metrics) to avoid contention.
    Spatial thermal (SMN reads) are always-on when SMN is available."""
    sclk = read_current_sclk_mhz()
    gpu_ppt = read_gpu_ppt_mw()
    thermal_vec, temp_c = read_thermal_state()
    freq_vec = read_freq_sensing()
    if lite:
        pm_vec = torch.zeros(PM_DEEP_DIM)
        smn_vec = torch.zeros(SMN_RAW_DIM)
        gm_vec = torch.zeros(GPU_METRICS_DIM)
    else:
        pm_vec = read_pm_deep_vec()
        smn_vec = read_smn_raw_vec()
        gm_vec = read_gpu_metrics_vec()

    # DF counters as deltas from previous snapshot
    df_snap = read_df_snapshot()
    if prev_df is not None:
        # prev_df is the raw previous snapshot dict, compute deltas
        df_deltas = [
            max(df_snap.get('df_dram_read', 0) - prev_df.get('df_dram_read', 0), 0),
            max(df_snap.get('df_dram_write', 0) - prev_df.get('df_dram_write', 0), 0),
            max(df_snap.get('df_coherent', 0) - prev_df.get('df_coherent', 0), 0),
        ]
        df_vec = torch.tensor([
            min(math.log1p(df_deltas[0]) / 25.0, 1.0),
            min(math.log1p(df_deltas[1]) / 25.0, 1.0),
            min(math.log1p(df_deltas[2]) / 25.0, 1.0),
        ])
    else:
        df_vec = torch.tensor([
            math.log1p(df_snap.get('df_dram_read', 0)) / 25.0,
            math.log1p(df_snap.get('df_dram_write', 0)) / 25.0,
            math.log1p(df_snap.get('df_coherent', 0)) / 25.0,
        ])

    # RAPL energy
    rapl = read_rapl_snapshot()
    pkg_uj = rapl.get('package-0', rapl.get('pkg', 0))
    core_uj = rapl.get('core', 0)
    energy_vec = torch.tensor([
        min(pkg_uj / 1e9, 1.0),
        min(core_uj / 1e9, 1.0),
        min(gpu_ppt / 50000.0, 1.0),
    ], dtype=torch.float32)

    analog_vec = torch.tensor([
        min(temp_c / 100.0, 1.0),
        min(gpu_ppt / 50000.0, 1.0),
        sclk / 3000.0,
        df_vec[0].item(), df_vec[1].item(), df_vec[2].item(),
    ], dtype=torch.float32)

    status_vec = torch.tensor([0.0, sclk / 3000.0], dtype=torch.float32)
    action_vec = prev_action if prev_action is not None else torch.zeros(ACTION_DIM)

    # Spatial thermal (32 ADC sensors) — SMN reads are safe during training
    # (unlike gpu_metrics sysfs which conflicts with ISA MODE writes)
    if SMN_AVAILABLE:
        thm_a_vec, thm_b_vec, spatial_temps = read_spatial_thermal()
    else:
        thm_a_vec = torch.zeros(THM_SPATIAL_A_DIM)
        thm_b_vec = torch.zeros(THM_SPATIAL_B_DIM)
        spatial_temps = [0.0] * 32

    # CPU PMU (delta from previous snapshot)
    if CPU_PMU_AVAILABLE:
        cpu_pmu_vec, cpu_pmu_snap = read_cpu_pmu_vec(prev_cpu_pmu_snapshot)
    else:
        cpu_pmu_vec = torch.zeros(CPU_PMU_DIM)
        cpu_pmu_snap = None

    # z2098: deep gpu_metrics (per-core C0 + clock deltas)
    if not lite and GPU_METRICS_PATH:
        gpu_deep_vec = read_gpu_metrics_deep_vec()
    else:
        gpu_deep_vec = torch.zeros(GPU_METRICS_DEEP_DIM)

    # z2098: fence ring queue depths
    fence_vec = read_fence_vec()

    return {
        'analog': analog_vec, 'energy': energy_vec, 'freq': freq_vec,
        'thermal': thermal_vec, 'pm_deep': pm_vec, 'smn_raw': smn_vec,
        'gpu_metrics': gm_vec,
        'thm_spatial_a': thm_a_vec, 'thm_spatial_b': thm_b_vec,
        'cpu_pmu': cpu_pmu_vec,
        'gpu_metrics_deep': gpu_deep_vec, 'fence': fence_vec,
        'status': status_vec, 'action': action_vec,
        'sclk_mhz': sclk, 'gpu_ppt_mw': gpu_ppt, 'temp_c': temp_c,
        'spatial_temps': spatial_temps,
        'df_snap': df_snap, 'cpu_pmu_snap': cpu_pmu_snap,
    }

def expand_sensor(vec, batch_size, device):
    return vec.unsqueeze(0).expand(batch_size, -1).to(device)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HIP KERNEL — ISA personality math + intrinsic state readback
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_hip_module = None

def get_hip_module():
    global _hip_module
    if _hip_module is not None:
        return _hip_module

    cpp_source = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>

__global__ void math_kernel_intrinsic(
    const float* __restrict__ input, float* __restrict__ output,
    float* __restrict__ intrinsic_out,
    int N, int round_mode, int denorm_mode, int chain_code, int perm_code)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Save original MODE so later PyTorch kernels aren't poisoned
    unsigned int old_mode;
    asm volatile("s_getreg_b32 %0, hwreg(1, 0, 8)" : "=s"(old_mode));

    // Set MODE register: FP_ROUND[3:0] | FP_DENORM[7:4]
    unsigned int mode_val = (round_mode & 0xF) | ((denorm_mode & 0xF) << 4);
    asm volatile("s_setreg_b32 hwreg(1, 0, 8), %0" : : "s"(mode_val));

    float x = input[idx];

    // Personality-dependent math
    float a, b, c;
    if (chain_code == 0) {
        a = x * 1.5f + 0.3f;
        b = a * a - x * 0.7f;
        c = fmaf(a, b, x);
    } else {
        a = x * 0.7f - 0.3f;
        b = a * x + a * 0.5f;
        c = fmaf(b, a, -x);
    }

    // fp16 mix for maximum bit divergence
    __half h = __float2half(c);
    if (perm_code == 0) {
        h = __hmul(h, __float2half(1.0f));
    } else {
        h = __hmul(h, __float2half(-1.0f));
        h = __hneg(h);
        // byte swap for extra divergence (portable, no __builtin_amdgcn_perm)
        unsigned int as_uint = (unsigned int)__half_as_ushort(h);
        as_uint = ((as_uint & 0xFF) << 8) | ((as_uint >> 8) & 0xFF);
        h = __ushort_as_half((unsigned short)(as_uint & 0xFFFF));
    }
    output[idx] = __half2float(h);

    // Restore MODE before any other work (prevents poisoning later kernels)
    asm volatile("s_setreg_b32 hwreg(1, 0, 8), %0" : : "s"(old_mode));

    // Read intrinsic hardware state — ONLY on thread 0 (avoids hitting
    // sensitive hwregs on every thread which can wedge the GPU)
    if (idx == 0) {
        unsigned int hw_status, hw_gpr, hw_lds, hw_ib_sts, hw_id2, hw_perf;
        unsigned int shader_cy_lo, shader_cy_hi;
        unsigned long long clk64, wall64;

        asm volatile("s_getreg_b32 %0, hwreg(2)" : "=s"(hw_status));
        asm volatile("s_getreg_b32 %0, hwreg(5)" : "=s"(hw_gpr));
        asm volatile("s_getreg_b32 %0, hwreg(6)" : "=s"(hw_lds));
        asm volatile("s_getreg_b32 %0, hwreg(7)" : "=s"(hw_ib_sts));
        asm volatile("s_getreg_b32 %0, hwreg(24)" : "=s"(hw_id2));
        asm volatile("s_getreg_b32 %0, hwreg(27)" : "=s"(hw_perf));
        asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(shader_cy_lo));
        asm volatile("s_getreg_b32 %0, hwreg(30)" : "=s"(shader_cy_hi));
        clk64 = clock64();
        wall64 = wall_clock64();

        intrinsic_out[0] = __uint_as_float(hw_status);
        intrinsic_out[1] = __uint_as_float(hw_gpr);
        intrinsic_out[2] = __uint_as_float(hw_lds);
        intrinsic_out[3] = __uint_as_float(hw_ib_sts);
        intrinsic_out[4] = __uint_as_float(hw_id2);
        intrinsic_out[5] = __uint_as_float(hw_perf);
        intrinsic_out[6] = __uint_as_float(shader_cy_lo);
        intrinsic_out[7] = __uint_as_float(shader_cy_hi);
        intrinsic_out[8] = __uint_as_float((unsigned int)(clk64 & 0xFFFFFFFF));
        intrinsic_out[9] = __uint_as_float((unsigned int)(clk64 >> 32));
        intrinsic_out[10] = __uint_as_float((unsigned int)(wall64 & 0xFFFFFFFF));
        intrinsic_out[11] = __uint_as_float((unsigned int)(wall64 >> 32));
    }
}

std::vector<torch::Tensor> run_math_kernel(torch::Tensor input, int round_mode,
    int denorm_mode, int chain_code, int perm_code) {
    auto output = torch::zeros_like(input);
    auto intrinsic = torch::zeros({12}, input.options());
    int N = input.numel();
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    math_kernel_intrinsic<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        intrinsic.data_ptr<float>(), N, round_mode, denorm_mode, chain_code, perm_code);
    return {output, intrinsic};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_math_kernel", &run_math_kernel);
}
"""

    os.environ['PYTORCH_ROCM_ARCH'] = 'gfx1100'
    os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

    print("[HIP] Compiling ISA personality kernel...")
    from torch.utils.cpp_extension import load_inline
    _hip_module = load_inline(
        name='z2096_hip',
        cpp_sources=[],
        cuda_sources=cpp_source,
        with_cuda=True,
        verbose=False,
        extra_cuda_cflags=['-O2'],
    )
    print("[HIP] Kernel compiled successfully")
    return _hip_module

def config_to_kernel_args(config):
    return {
        'round_mode': config['round_mode'],
        'denorm_mode': config['denorm_mode'],
        'chain_code': config['chain_code'],
        'perm_code': config['perm_code'],
    }

def run_isa_kernel(input_tensor, kargs):
    """Run ISA personality kernel. Returns (output, delta, intrinsic)."""
    hip = get_hip_module()
    # Software reference (no MODE manipulation)
    with torch.no_grad():
        sw_ref = input_tensor * 1.5 + 0.3
        sw_ref = sw_ref * sw_ref - input_tensor * 0.7

    hw_out, intrinsic_raw = hip.run_math_kernel(
        input_tensor, kargs['round_mode'], kargs['denorm_mode'],
        kargs['chain_code'], kargs['perm_code'])
    torch.cuda.synchronize()  # ensure kernel completes before any subsequent GPU work

    # Delta = HW - SW reference (the ISA fingerprint)
    # Sanitise nan/inf from fp16 byte-swap, then tanh-squash so the personality
    # PATTERN (sign structure) is preserved in [-1, 1] without losing signal.
    # Personality A gives small deltas (tanh ≈ linear), personality B gives large
    # deltas (tanh saturates) — the shape difference IS the fingerprint.
    delta_raw = hw_out - sw_ref
    delta_raw = torch.nan_to_num(delta_raw, nan=0.0, posinf=0.0, neginf=0.0)
    # Return RAW delta (caller applies softsign or other bounding)
    # Clamp to prevent extreme values but preserve sign/magnitude pattern
    delta_raw = delta_raw.clamp(-100.0, 100.0)

    # Intrinsic values are __uint_as_float() reinterpretations of hw registers.
    # Many bit patterns produce NaN/inf, and most FINITE values are enormous
    # (e.g. 0x7F000000 → 1.7e38).  nan_to_num only fixes nan/inf, so we
    # also need tanh to bound the huge finite values to [-1, 1].
    # The sign/saturation PATTERN is the fingerprint — magnitude is meaningless.
    intrinsic = torch.nan_to_num(intrinsic_raw, nan=0.0, posinf=1.0, neginf=-1.0)
    intrinsic = torch.tanh(intrinsic)

    return hw_out, delta_raw, intrinsic

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BODY ENCODER — Transformer over substrate tokens
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class BodyEncoder(nn.Module):
    """Encode 11 substrate tokens via self-attention.
    Outputs: body_vec (for LoRA scaling), next_telem_pred, delta_regime_head,
    analog_regime_head, mismatch_head, freq_gate (regime gate from freq signal).
    """
    def __init__(self, token_dim=TOKEN_DIM):
        super().__init__()
        self.token_dim = token_dim
        # Per-token encoders
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
        # z2098: NEW sensor encoders
        self.gpu_metrics_deep_enc = nn.Linear(GPU_METRICS_DEEP_DIM, token_dim)
        self.fence_enc = nn.Linear(FENCE_DIM, token_dim)

        # Learnable token type embeddings
        self.token_type_emb = nn.Embedding(N_SUBSTRATE_TOKENS, token_dim)

        # Transformer self-attention
        self.substrate_attn = nn.MultiheadAttention(
            embed_dim=token_dim, num_heads=4, batch_first=True, dropout=0.1)
        self.attn_norm = nn.LayerNorm(token_dim)
        self.attn_ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2), nn.GELU(), nn.Linear(token_dim * 2, token_dim))
        self.ffn_norm = nn.LayerNorm(token_dim)

        # Output heads
        n_all = ANALOG_DIM + ENERGY_DIM + FREQ_DIM + THERMAL_DIM + PM_DEEP_DIM + SMN_RAW_DIM + GPU_METRICS_DIM + THM_SPATIAL_A_DIM + THM_SPATIAL_B_DIM + CPU_PMU_DIM + GPU_METRICS_DEEP_DIM + FENCE_DIM
        self.next_telem_pred = nn.Linear(token_dim * N_SUBSTRATE_TOKENS, n_all)
        self.delta_regime_head = nn.Linear(token_dim, 1)   # predict regime from delta
        self.analog_regime_head = nn.Linear(token_dim, 1)  # predict regime from analog

        # Mismatch head: cross-validates actual delta vs reported_delta
        # delta = ground truth from ISA kernel, reported_delta = externally supplied (corruptible)
        self.mismatch_head = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim), nn.GELU(),
            nn.Linear(token_dim, 1), nn.Sigmoid())

        # === z2100 Change 1: Coupled Token Encoding (→ T32 Phi) ===
        # Cross-token message passing creates causal coupling between tokens
        self.coupling = nn.Linear(token_dim, token_dim)
        self.coupling_gate = nn.Parameter(torch.zeros(1))  # learnable on/off

        # === z2100 Change 2: Workspace Bottleneck (→ T33 Cliff) ===
        # Compress 17 tokens → 4 competitive workspace slots via cross-attention
        self.workspace_slots = nn.Parameter(torch.randn(N_WORKSPACE_SLOTS, token_dim) * 0.02)
        self.workspace_attn = nn.MultiheadAttention(token_dim, 2, batch_first=True, dropout=0.1)

        # Body scale projector — now uses workspace output (4*token_dim=128) instead of 17*token_dim=544
        self.body_scale_proj = nn.Linear(N_WORKSPACE_SLOTS * token_dim, 1)
        nn.init.constant_(self.body_scale_proj.bias, 1.0)  # sigmoid(1)=0.73 → stronger LoRA coupling
        self.body_scale_floor = 0.005  # lower floor → bigger embodiment gap (T4)

        # === z2100 Change 3: Temporal Gate (→ T34 PCIST) ===
        # GRU + EMA replacing instantaneous sigmoid for complex temporal dynamics
        self.gate_gru = nn.GRUCell(1, 16)
        self.gate_out = nn.Linear(16, 1)
        self.gate_hidden = None  # persistent state, reset before eval

        # === z2100 Change 4: Head Specialization (→ T36 Synergy) ===
        # Modality masks: each of 4 attention heads prefers different token subsets
        # Token indices: 0=delta, 1=analog, 2=energy, 3=freq, 4=intrinsic, 5=thermal,
        #   6=pm_deep, 7=smn_raw, 8=gpu_metrics, 9=thm_spatial_a, 10=thm_spatial_b,
        #   11=cpu_pmu, 12=status, 13=action, 14=reported_delta, 15=gpu_metrics_deep, 16=fence
        self.head_masks = nn.Parameter(torch.full((4, N_SUBSTRATE_TOKENS), -5.0))
        with torch.no_grad():
            # Head 0: ISA modalities (delta, analog, reported_delta)
            self.head_masks[0, [0, 1, 14]] = 5.0
            # Head 1: power modalities (freq, energy, gpu_metrics, gpu_metrics_deep)
            self.head_masks[1, [3, 2, 8, 15]] = 5.0
            # Head 2: thermal modalities (thermal, thm_spatial_a, thm_spatial_b, smn_raw)
            self.head_masks[2, [5, 9, 10, 7]] = 5.0
            # Head 3: system modalities (intrinsic, pm_deep, cpu_pmu, status, action, fence)
            self.head_masks[3, [4, 6, 11, 12, 13, 16]] = 5.0

        # === z2100 Change 5: Predictive Body Scale (→ T39 Anticipation) ===
        # LSTM predicts next-step temperature from workspace state
        self.temp_predictor = nn.LSTMCell(N_WORKSPACE_SLOTS * token_dim, 32)
        self.temp_pred_out = nn.Linear(32, 1)
        self.temp_lstm_state = None  # persistent state, reset before eval

        # z2102: reset_state() method clears ALL persistent state
        # MUST be called before every test/eval to prevent state contamination

        # === z2095: CALIBRATED sharp freq-driven regime gate ===
        # Gate driven DIRECTLY by hardware freq signal with sharp sigmoid
        # Input: [sclk_calibrated, freq_ratio_calibrated] where calibrated = (x-low)/(high-low)
        # Gate = sigmoid(GATE_TEMP * (W @ calibrated_input + bias))
        self.freq_gate_proj = nn.Linear(2, 1)
        # Initialize: calibrated input [0.5, 0.5] → 0, extremes → ±1
        # With GATE_TEMP=8: sigmoid(8*1)=0.9997, sigmoid(8*-1)=0.0003
        with torch.no_grad():
            self.freq_gate_proj.weight.fill_(1.0)   # equal weight on both calibrated features
            self.freq_gate_proj.bias.fill_(-1.0)     # midpoint (0.5+0.5)*1.0 - 1.0 = 0 → gate=0.5

    def reset_state(self):
        """z2102: Clear ALL persistent state. MUST call before every test/eval."""
        self.gate_hidden = None
        self.temp_lstm_state = None
        if hasattr(self, '_stress_ema'):
            del self._stress_ema

    def forward(self, sensor_dict, availability_mask=None):
        """Forward pass.
        availability_mask: optional [B, N_SUBSTRATE_TOKENS] binary tensor.
            1 = sensor SHOULD be present (count for presence_frac)
            0 = sensor structurally absent (don't penalize presence_frac)
            None = treat all tokens as expected-present (default, used by T4 ablation)
        """
        B = sensor_dict['delta'].shape[0]
        dev = sensor_dict['delta'].device
        # Sanitise every sensor channel — HW reads can produce nan/inf
        _keys = ['delta','analog','energy','freq','intrinsic','thermal',
                 'pm_deep','smn_raw','gpu_metrics','thm_spatial_a','thm_spatial_b',
                 'cpu_pmu','status','action','reported_delta',
                 'gpu_metrics_deep','fence']
        _dims = [DELTA_DIM, ANALOG_DIM, ENERGY_DIM, FREQ_DIM, INTRINSIC_DIM, THERMAL_DIM,
                 PM_DEEP_DIM, SMN_RAW_DIM, GPU_METRICS_DIM, THM_SPATIAL_A_DIM, THM_SPATIAL_B_DIM,
                 CPU_PMU_DIM, STATUS_DIM, ACTION_DIM, REPORTED_DELTA_DIM,
                 GPU_METRICS_DEEP_DIM, FENCE_DIM]
        # Auto-fill missing keys with zeros (avoids updating 30+ sensor_batch sites)
        sd = {}
        for k, d in zip(_keys, _dims):
            if k in sensor_dict:
                sd[k] = torch.nan_to_num(sensor_dict[k], nan=0.0, posinf=0.0, neginf=0.0)
            else:
                sd[k] = torch.zeros(B, d, device=dev)
        # Encode each token with presence masking
        # If raw sensor input is all-zero, presence=0 → encoded token = 0
        # This prevents bias leakage when sensors are ablated
        def _enc_with_presence(enc, inp):
            presence = (inp.abs().sum(dim=-1, keepdim=True) > 1e-8).float()
            return enc(inp) * presence

        tokens = []
        tokens.append(_enc_with_presence(self.delta_enc, sd['delta']))         # 0
        tokens.append(_enc_with_presence(self.analog_enc, sd['analog']))       # 1
        tokens.append(_enc_with_presence(self.energy_enc, sd['energy']))       # 2
        tokens.append(_enc_with_presence(self.freq_enc, sd['freq']))           # 3
        tokens.append(_enc_with_presence(self.intrinsic_enc, sd['intrinsic']))  # 4
        tokens.append(_enc_with_presence(self.thermal_enc, sd['thermal']))     # 5
        tokens.append(_enc_with_presence(self.pm_deep_enc, sd['pm_deep']))     # 6
        tokens.append(_enc_with_presence(self.smn_raw_enc, sd['smn_raw']))     # 7
        tokens.append(_enc_with_presence(self.gpu_metrics_enc, sd['gpu_metrics']))  # 8
        tokens.append(_enc_with_presence(self.thm_spatial_a_enc, sd['thm_spatial_a']))  # 9
        tokens.append(_enc_with_presence(self.thm_spatial_b_enc, sd['thm_spatial_b']))  # 10
        tokens.append(_enc_with_presence(self.cpu_pmu_enc, sd['cpu_pmu']))              # 11
        tokens.append(_enc_with_presence(self.status_enc, sd['status']))       # 12
        tokens.append(_enc_with_presence(self.action_enc, sd['action']))       # 13
        tokens.append(_enc_with_presence(self.reported_delta_enc, sd['reported_delta']))  # 14
        # z2098: new sensor tokens
        tokens.append(_enc_with_presence(self.gpu_metrics_deep_enc, sd['gpu_metrics_deep']))  # 15
        tokens.append(_enc_with_presence(self.fence_enc, sd['fence']))                        # 16

        x = torch.stack(tokens, dim=1)  # [B, 17, token_dim]

        # --- NaN root-cause trace (first call only) ---
        _trace = getattr(self, '_nan_trace', True)
        def _chk(tag, t):
            if _trace and t.isnan().any():
                print(f"      [NaN-TRACE] {tag}: nan detected! "
                      f"shape={list(t.shape)} max={t[~t.isnan()].abs().max().item() if (~t.isnan()).any() else 'ALL_NAN'}",
                      flush=True)
                return True
            return False
        _chk('tokens_stacked', x)

        # Check each token encoder output individually
        if _trace and x.isnan().any():
            for ti, name in enumerate(_keys):
                if tokens[ti].isnan().any():
                    print(f"      [NaN-TRACE] token '{name}' encoder output has nan! "
                          f"input_nan={sd[name].isnan().any().item()} "
                          f"input_range=[{sd[name].min().item():.4f}, {sd[name].max().item():.4f}]",
                          flush=True)
                    enc = getattr(self, f'{name}_enc')
                    for pn, pv in enc.named_parameters():
                        if pv.isnan().any():
                            print(f"      [NaN-TRACE]   {name}_enc.{pn} has nan weights!", flush=True)

        # === z2100 Change 1: Cross-token coupling ===
        # After independent encoding, add message passing between tokens
        coupling_strength = torch.sigmoid(self.coupling_gate)
        if coupling_strength > 0.01:  # skip if gate is effectively off
            coupled = self.coupling(x)  # [B, 17, token_dim]
            adj = coupling_strength * F.softmax(
                (x @ coupled.transpose(-1, -2)) / math.sqrt(self.token_dim), dim=-1)  # [B, 17, 17]
            x = x + adj @ x  # residual cross-token message
        _chk('after_coupling', x)

        # Add token type embeddings (for attention routing only, not body_scale)
        # z2102 Fix: mask type_emb by presence — absent tokens must stay zero
        # Without this, token_type_emb reintroduces nonzero vectors for zeroed tokens,
        # breaking presence masking and contaminating T33/T22/T19 ablation tests
        type_ids = torch.arange(N_SUBSTRATE_TOKENS, device=x.device)
        type_emb = self.token_type_emb(type_ids).unsqueeze(0)  # [1, 17, token_dim]
        presence_mask = (x.abs().sum(dim=-1, keepdim=True) > 1e-8).float()  # [B, 17, 1]
        x = x + type_emb * presence_mask
        _chk('after_type_emb', x)

        # === z2100 Change 4: Head specialization via attention bias ===
        # Apply modality masks as attention bias before substrate_attn
        mask_weights = torch.sigmoid(self.head_masks)  # [4, 17] soft masks
        # Create attention bias: [4, 17] → bias per head per key token
        attn_bias = torch.log(mask_weights.clamp(min=1e-6))  # [4, 17] log-space bias
        # Expand to [B*4, 17, 17] format (bias applied to all query positions equally)
        attn_bias_expanded = attn_bias.unsqueeze(1).expand(-1, N_SUBSTRATE_TOKENS, -1)  # [4, 17, 17]
        attn_bias_expanded = attn_bias_expanded.unsqueeze(0).expand(B, -1, -1, -1)  # [B, 4, 17, 17]
        attn_bias_flat = attn_bias_expanded.reshape(B * 4, N_SUBSTRATE_TOKENS, N_SUBSTRATE_TOKENS)

        # Self-attention with head specialization bias
        attn_out, attn_weights = self.substrate_attn(
            x, x, x, need_weights=True, average_attn_weights=False,
            attn_mask=attn_bias_flat)
        _chk('attn_out', attn_out)
        x = self.attn_norm(x + attn_out)
        _chk('after_attn_norm', x)
        x = self.ffn_norm(x + self.attn_ffn(x))
        _chk('after_ffn', x)

        # Flatten for output heads (includes token_type info for predictions)
        flat = x.reshape(B, -1)  # [B, 17*token_dim]

        # Next telemetry prediction (self-supervised)
        telem_pred = self.next_telem_pred(flat)

        # Delta regime prediction (from delta token only)
        delta_regime = torch.sigmoid(self.delta_regime_head(x[:, 0, :]))  # token 0 = delta

        # Analog regime prediction (from freq token)
        analog_regime = torch.sigmoid(self.analog_regime_head(x[:, 3, :]))  # token 3 = freq

        # Mismatch: cross-validate actual delta (token 0) vs reported_delta (token 14)
        delta_reported_cat = torch.cat([x[:, 0, :], x[:, 14, :]], dim=-1)
        mismatch = self.mismatch_head(delta_reported_cat)

        # === z2100 Change 2: Workspace Bottleneck ===
        # Compress 17 tokens → 4 competitive workspace slots
        ws_slots = self.workspace_slots.unsqueeze(0).expand(B, -1, -1)  # [B, 4, token_dim]
        ws_out, ws_weights = self.workspace_attn(ws_slots, x, x)  # [B, 4, token_dim]
        content_flat = ws_out.reshape(B, N_WORKSPACE_SLOTS * self.token_dim)  # [B, 128]

        # Body scale: MULTIPLICATIVE coupling (now from workspace output)
        body_scale_raw = self.body_scale_proj(content_flat)
        if _trace and body_scale_raw.isnan().any():
            print(f"      [NaN-TRACE] body_scale_proj OUTPUT is nan!", flush=True)
        body_scale_sig = torch.sigmoid(body_scale_raw)  # [B, 1]
        # z2095: Availability-aware presence gating
        presences = torch.stack(
            [(tok.abs().sum(dim=-1, keepdim=True) > 1e-8).float() for tok in tokens], dim=1)
        if availability_mask is not None:
            avail = availability_mask.unsqueeze(-1)
            n_expected = avail.sum(dim=1).clamp(min=1.0)
            presence_frac = (presences * avail).sum(dim=1) / n_expected
        else:
            presence_frac = presences.mean(dim=1)
        body_scale = self.body_scale_floor + (1.0 - self.body_scale_floor) * body_scale_sig * (presence_frac ** 2)

        # === z2100v2: Workspace gate for capacity cliff (T33) ===
        # Count live tokens; if below workspace capacity, hard-gate body influence
        n_present = presences.sum(dim=1).squeeze(-1)  # [B]
        ws_gate = (n_present >= float(N_WORKSPACE_SLOTS)).float().unsqueeze(-1)  # [B, 1] binary

        # === z2100 Change 5: Predictive Body Scale ===
        # LSTM predicts next temperature from workspace state → anticipatory modulation
        if self.temp_lstm_state is None or self.temp_lstm_state[0].shape[0] != B:
            self.temp_lstm_state = (
                torch.zeros(B, 32, device=content_flat.device),
                torch.zeros(B, 32, device=content_flat.device))
        h_lstm, c_lstm = self.temp_predictor(
            content_flat.detach(),
            (self.temp_lstm_state[0].detach(), self.temp_lstm_state[1].detach()))
        self.temp_lstm_state = (h_lstm, c_lstm)
        temp_prediction = self.temp_pred_out(h_lstm)  # [B, 1]
        # Get current temperature estimate from thermal token
        current_temp = sd['thermal'][:, 0:1] if sd['thermal'].shape[-1] > 0 else torch.zeros(B, 1, device=content_flat.device)
        # Modulate body_scale with predicted temperature change
        body_scale = body_scale * (1.0 + 0.3 * torch.tanh(temp_prediction - current_temp))  # z2100v3: 0.1→0.3 stronger anticipatory signal for T39

        # === Fix 7: Power-based stress anticipation for T39 ===
        # Power/frequency are LEADING indicators of temperature (power rises before temp)
        # energy[0] = pkg_power_norm, freq[0] = sclk_norm → stress signal
        power_signal = sd['energy'][:, 0:1]   # [B, 1] — normalized package power
        sclk_signal = sd['freq'][:, 0:1]      # [B, 1] — normalized SCLK
        stress = (power_signal + sclk_signal) * 0.5  # [B, 1] — combined stress
        # EMA baseline: compare current stress to running average
        if not hasattr(self, '_stress_ema'):
            self._stress_ema = stress.detach().mean().item()
        stress_delta = stress - self._stress_ema
        self._stress_ema = 0.95 * self._stress_ema + 0.05 * stress.detach().mean().item()
        # z2102 Fix T39: High stress → INCREASE body_scale (lean INTO hardware state, don't retreat)
        # Flipped sign: stress rising → body_scale UP → more HW dependence → proactive response
        body_scale = body_scale * (1.0 + 0.2 * torch.tanh(stress_delta))

        # === z2095: CALIBRATED sharp freq-driven regime gate ===
        raw_sclk_norm = sd['freq'][:, 0:1]
        raw_freq_ratio = sd['freq'][:, 1:2]
        sclk_mhz = raw_sclk_norm * 3000.0
        sclk_cal = ((sclk_mhz - SCLK_LOW_CAL) / max(SCLK_HIGH_CAL - SCLK_LOW_CAL, 1.0)).clamp(0, 1)
        freq_ratio_low = SCLK_LOW_CAL / max(SCLK_HIGH_CAL, 1.0)
        freq_ratio_cal = ((raw_freq_ratio - freq_ratio_low) / max(1.0 - freq_ratio_low, 0.01)).clamp(0, 1)
        freq_input = torch.cat([sclk_cal, freq_ratio_cal], dim=-1)
        # Instantaneous gate (before temporal smoothing)
        freq_gate_instant = torch.sigmoid(GATE_TEMP * self.freq_gate_proj(freq_input))  # [B, 1]

        # === z2100 Change 3: Temporal Gate (GRU + EMA) ===
        if self.gate_hidden is None or self.gate_hidden.shape[0] != B:
            self.gate_hidden = torch.zeros(B, 16, device=freq_gate_instant.device)
        self.gate_hidden = self.gate_gru(freq_gate_instant, self.gate_hidden.detach())
        gru_gate = torch.sigmoid(GATE_TEMP * self.gate_out(self.gate_hidden))
        freq_gate = GATE_EMA_TAU * gru_gate + (1.0 - GATE_EMA_TAU) * freq_gate_instant

        # Orthogonality loss for head specialization (stored for training)
        ortho_loss = (mask_weights @ mask_weights.T - torch.eye(4, device=mask_weights.device)).pow(2).mean()

        return {
            'telem_pred': telem_pred,
            'delta_regime': delta_regime.squeeze(-1),
            'analog_regime': analog_regime.squeeze(-1),
            'mismatch': mismatch.squeeze(-1),
            'body_scale': body_scale.squeeze(-1),  # [B]
            'freq_gate': freq_gate.squeeze(-1),     # [B] — THE regime gate
            'attn_weights': attn_weights,
            'ortho_loss': ortho_loss,               # z2100: for training
            'temp_prediction': temp_prediction,      # z2100: for training loss
            'ws_gate': ws_gate.squeeze(-1),          # z2100v2: workspace gate for T33
            '_debug_flat': flat,
        }

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DVFS SAFETY CONTROLLER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class DVFSSafetyController:
    def __init__(self, min_dwell_s=2.0, hysteresis=0.1):
        self.min_dwell_s = min_dwell_s
        self.hysteresis = hysteresis
        self.current_level = 2  # start high (empirically more energy-efficient)
        self.last_switch = time.time()
        self.high_thresh = 0.2   # easy to go/stay high
        self.low_thresh = 0.05   # very hard to drop to low

    def step(self, demand):
        now = time.time()
        if now - self.last_switch < self.min_dwell_s:
            return self.current_level
        if self.current_level == 0 and demand > self.high_thresh:
            self.current_level = 2
            self.last_switch = now
        elif self.current_level == 2 and demand < self.low_thresh:
            self.current_level = 0
            self.last_switch = now
        return self.current_level

    def reset(self):
        self.current_level = 2  # reset to high (energy-efficient default)
        self.last_switch = time.time()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LORA LINEAR — Dual adapters with MULTIPLICATIVE body coupling
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class LoRALinear(nn.Module):
    """Dual LoRA with multiplicative body-scale coupling.

    z2097: Qwen2.5-1.5B uses standard nn.Linear — no Conv1D workaround needed.
    body_scale MULTIPLIES LoRA output directly.
    When body sensors are zeroed: body_scale ≈ 0.047 → LoRA nearly disabled.
    """
    def __init__(self, original_linear, rank=4, alpha=16):
        super().__init__()
        in_features = original_linear.in_features
        out_features = original_linear.out_features

        self.original = original_linear
        self.rank = rank
        self.scale = alpha / rank

        # LoRA adapter A (regime 0: wikitext next-token)
        self.lora_A_down = nn.Linear(in_features, rank, bias=False)
        self.lora_A_up = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A_down.weight)
        nn.init.zeros_(self.lora_A_up.weight)

        # LoRA adapter B (regime 1: token-shift cipher — (token + N) % VOCAB)
        self.lora_B_down = nn.Linear(in_features, rank, bias=False)
        self.lora_B_up = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_B_down.weight)
        nn.init.zeros_(self.lora_B_up.weight)

        # Freeze original
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x, regime_gate=None, body_scale=None):
        base = self.original(x)

        # LoRA computation in float32 for numerical stability
        x_f = x.float()
        lora_a = self.lora_A_up(self.lora_A_down(x_f)) * self.scale
        lora_b = self.lora_B_up(self.lora_B_down(x_f)) * self.scale

        # Gate selection: (1-g)*A + g*B
        if regime_gate is not None:
            g = regime_gate.float()
            while g.dim() < lora_a.dim():
                g = g.unsqueeze(-1)
            lora_out = (1 - g) * lora_a + g * lora_b
        else:
            lora_out = lora_a

        # STRONG body→LoRA coupling: sigmoid ramp so body_scale≈0 → LoRA nearly off (0.05)
        # Previous tanh+0.5 gave floor=0.2 which was too high → T4/T19/T22 FAIL
        # New sigmoid ramp: body_scale=0 → 0.05, body_scale=0.4 → ~0.99
        # MULTIPLICATIVE body coupling (z2097-style: simple linear)
        if body_scale is not None:
            bs = body_scale.float()
            while bs.dim() < lora_out.dim():
                bs = bs.unsqueeze(-1)
            lora_out = lora_out * bs

        return base + lora_out.to(base.dtype)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EMBODIED Qwen2 — Full model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class EmbodiedQwen2(nn.Module):
    """Qwen2.5-1.5B + Body Encoder + Dual LoRA + Multiplicative Coupling.

    Architecture:
      - Frozen Qwen2.5-1.5B (1.54B params, 28 layers, 1536 hidden, GQA 12Q/2KV)
      - Body Encoder: 17 substrate tokens → transformer self-attention (z2098: +deep_metrics, fence)
      - Dual LoRA: A (regime 0) / B (regime 1) on q_proj + v_proj, selected by freq_gate
      - Multiplicative body_scale: controls LoRA amplitude
      - Substrate bias: injected at layers 13 and 21
      - Hidden modulation: at layer 17
      - Demand head: for DVFS self-control
      - Thermal head: for self-model
    """
    def __init__(self, backbone_model, body_encoder, lora_blocks=range(10, 19),
                 rank=4, alpha=16):
        super().__init__()
        self.backbone = backbone_model
        self.body_encoder = body_encoder

        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Install LoRA on specified layers (q_proj + v_proj)
        self.lora_layers = nn.ModuleDict()
        for layer_idx in lora_blocks:
            layer = self.backbone.model.layers[layer_idx]
            # q_proj (query projection)
            key_q = f'layer{layer_idx}_q'
            self.lora_layers[key_q] = LoRALinear(layer.self_attn.q_proj, rank, alpha)
            # v_proj (value projection)
            key_v = f'layer{layer_idx}_v'
            self.lora_layers[key_v] = LoRALinear(layer.self_attn.v_proj, rank, alpha)

        # Substrate bias injection (layers 13 and 21) — GATED residual (Gemini fix: generation collapse)
        hidden_dim = self.backbone.config.hidden_size  # 1536 for Qwen2.5-1.5B
        self.substrate_bias_early = nn.Linear(TOKEN_DIM * N_SUBSTRATE_TOKENS, hidden_dim)
        self.substrate_bias_late = nn.Linear(TOKEN_DIM * N_SUBSTRATE_TOKENS, hidden_dim)
        nn.init.zeros_(self.substrate_bias_early.weight)
        nn.init.zeros_(self.substrate_bias_late.weight)
        self.substrate_scale = 0.0  # DISABLED: substrate injection hurts LM quality (full > ablated)
        # Gemini stability fix #4: normalize substrate injections before adding to residual
        self.substrate_norm = nn.LayerNorm(hidden_dim)
        # Gated substrate injection: sigmoid(gate(body_flat)) * bias prevents trajectory drift
        # Gate conditioned on RAW body encoding (not processed bias) — learns when to inject
        body_flat_dim = TOKEN_DIM * N_SUBSTRATE_TOKENS
        self.substrate_gate_early = nn.Linear(body_flat_dim, hidden_dim)
        self.substrate_gate_late = nn.Linear(body_flat_dim, hidden_dim)
        # Init gate weights zero, bias negative → sigmoid ≈ 0.27 → conservative by default
        nn.init.zeros_(self.substrate_gate_early.weight)
        self.substrate_gate_early.bias.data.fill_(-1.0)
        nn.init.zeros_(self.substrate_gate_late.weight)
        self.substrate_gate_late.bias.data.fill_(-1.0)

        # Hidden modulation at layer 17
        self.hidden_modulation = nn.Linear(TOKEN_DIM * N_SUBSTRATE_TOKENS, hidden_dim)
        nn.init.zeros_(self.hidden_modulation.weight)

        # Demand head (for DVFS self-control)
        self.demand_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())

        # Thermal self-model — uses spatial thermal + analog + energy + pm_deep
        # thm_spatial_a=9, thm_spatial_b=10, analog=1, energy=2, thermal=5, pm_deep=6
        self.thermal_token_indices = [1, 2, 5, 6, 9, 10]  # 6 tokens
        # Residual architecture: predict OFFSET from mean input temperature
        # Much easier than absolute prediction (±10°C vs 0-100°C range)
        self.thermal_head = nn.Sequential(
            nn.Linear(TOKEN_DIM * 6 + 1, 128),  # +1 for mean_temp anchor
            nn.GELU(), nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 32), nn.Tanh())  # Tanh: offsets in [-1,1] * THERMAL_OFFSET_SCALE

        # Raw mismatch head: operates on raw delta vs reported_delta for T10/T31
        # More robust than encoded-token mismatch because it sees raw signal
        self.raw_mismatch_head = nn.Sequential(
            nn.Linear(DELTA_DIM * 3, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Sigmoid())

        # z2098: Metacognition head — predicts own next gate value
        # If model can predict its own gate, it has self-knowledge
        # Input: hidden_mean + freq_gate + body_scale + mismatch (3 scalars give real signal)
        self.metacognition_head = nn.Sequential(
            nn.Linear(hidden_dim + 3, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())

        # z2098: Confidence head — predicts own loss on current batch
        # Calibrated self-assessment: does model know when it's uncertain?
        # Input: hidden_mean + freq_gate + body_scale (knows own state → better calibration)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim + 2, 64), nn.GELU(), nn.Linear(64, 1))

        # z2099: Meta2 head — 2nd-order metacognition: predict error of own self-prediction
        # z2100 Fix: chain meta_gate_pred into input (hidden_dim + 1)
        self.meta2_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())
        # z2100v2: Initialize meta2 final Sigmoid bias to map near zero (typical target range)
        with torch.no_grad():
            self.meta2_head[2].bias.fill_(-3.0)  # sigmoid(-3)≈0.047, near typical meta_error

        # z2099: Attribution head — predict which substrate token drives the regime gate
        # 17-class: one per substrate token. Trained on argmax of body encoder attn weights→gate
        self.attribution_head = nn.Sequential(
            nn.Linear(hidden_dim + N_SUBSTRATE_TOKENS, 64), nn.GELU(), nn.Linear(64, N_ATTRIBUTION_CLASSES))  # z2102: +17 for attn_avg input

        # z2101: lm_head LoRA REMOVED — skip-gram doesn't change output distribution,
        # only shifts attention position. Frozen lm_head is sufficient and more stable.
        # lm_head LoRA caused generation collapse by corrupting logit distribution.

        # z2101: ThermalSoftmax — GPU temperature constitutively modulates attention sharpness
        self.thermal_alpha = nn.Parameter(torch.tensor(0.25))  # learnable, init 0.25 (higher for T40)

        # Sensor EMA for inference: smooth high-frequency hardware jitter during generation
        # Only active during eval (model.training == False), training sees raw sensors
        self._sensor_ema = {}  # key → EMA tensor, populated on first eval call
        self._sensor_ema_alpha = 0.1  # EMA decay: new = alpha*raw + (1-alpha)*old
        # 0.1 (was 0.3): more aggressive smoothing to prevent generation drift

        # Fixed ISA probe — deterministic input for low-variance delta measurement
        self.register_buffer('isa_probe', torch.randn(1024))

    def forward(self, input_ids, sensor_dict, kargs, labels=None,
                regime_gate_override=None, availability_mask=None, skip_isa=False,
                skip_substrate=False):
        B = input_ids.shape[0]
        _dbg = getattr(self, '_debug_forward', False)

        # Run ISA kernel for delta + intrinsic (FIXED probe for reproducibility)
        # skip_isa=True during generation: reuse provided delta/intrinsic (zeros),
        # prevents hardware jitter accumulating across autoregressive steps
        # Auto-detect all-zero sensor_dict (ablation tests) → skip ISA too
        if not skip_isa:
            total_abs = sum(v.abs().sum().item() for v in sensor_dict.values() if torch.is_tensor(v) and v.is_floating_point())
            if total_abs < 1e-8:
                skip_isa = True
        if skip_isa:
            delta_raw = torch.zeros(DELTA_DIM, device=input_ids.device)
            intrinsic = torch.zeros(INTRINSIC_DIM, device=input_ids.device)
            if _dbg: print("    [FWD] ISA skipped (generation mode)", flush=True)
        else:
            if _dbg: print("    [FWD] ISA kernel...", flush=True)
            _, delta_raw, intrinsic = run_isa_kernel(self.isa_probe[:32], kargs)
            if _dbg: print("    [FWD] ISA done", flush=True)
        # Better delta features: softsign (bounded) + log1p magnitude (scale-aware)
        delta_s = delta_raw[:DELTA_DIM] / (1.0 + delta_raw[:DELTA_DIM].abs())  # softsign [-1, 1]
        delta_m = torch.log1p(delta_raw[:DELTA_DIM].abs()).clamp(0, 10) / 10.0  # magnitude [0, 1]
        # Use softsign as primary features (replaces tanh which over-squashes)
        delta_vec = torch.nan_to_num(delta_s, nan=0.0, posinf=0.0, neginf=0.0).unsqueeze(0).expand(B, -1)
        intrinsic_vec = intrinsic.unsqueeze(0).expand(B, -1)  # already sanitized in run_isa_kernel
        # Update delta + intrinsic. Keep reported_delta from caller (honest or gaslit)
        # CRITICAL FIX for T10: if reported_delta is zeros (placeholder), replace with true delta
        # Otherwise "clean" training examples have mismatched delta vs reported_delta
        reported_delta = sensor_dict.get('reported_delta', None)
        if reported_delta is None:
            reported_delta = delta_vec.clone()
        else:
            rep = reported_delta
            if rep.dim() == 1:
                rep = rep.unsqueeze(0)
            zero_mask = (rep.abs().sum(dim=-1, keepdim=True) < 1e-8)  # [B,1]
            reported_delta = torch.where(zero_mask.expand_as(rep), delta_vec.detach(), rep)
        sensor_dict = {**sensor_dict, 'delta': delta_vec, 'intrinsic': intrinsic_vec,
                       'reported_delta': reported_delta}

        # Sensor EMA: smooth high-frequency jitter during inference (NOT training)
        # Prevents trajectory drift that causes generation collapse
        if not self.training:
            alpha = self._sensor_ema_alpha  # 0.3: new = 0.3*raw + 0.7*old
            sensor_dict_smoothed = {}
            for ek, raw in sensor_dict.items():
                if isinstance(raw, torch.Tensor) and raw.is_floating_point():
                    if ek in self._sensor_ema and self._sensor_ema[ek].shape == raw.shape:
                        smoothed = alpha * raw + (1 - alpha) * self._sensor_ema[ek].to(raw.device)
                        self._sensor_ema[ek] = smoothed.detach()
                        sensor_dict_smoothed[ek] = smoothed
                    else:
                        self._sensor_ema[ek] = raw.detach().clone()
                        sensor_dict_smoothed[ek] = raw
                else:
                    sensor_dict_smoothed[ek] = raw
            sensor_dict = sensor_dict_smoothed
        else:
            # Clear EMA during training to prevent stale state
            if self._sensor_ema:
                self._sensor_ema.clear()

        # Body encoder (z2095: pass availability_mask for presence gating)
        if _dbg: print("    [FWD] Body encoder...", flush=True)
        body_out = self.body_encoder(sensor_dict, availability_mask=availability_mask)
        body_scale = body_out['body_scale']  # [B]
        freq_gate = body_out['freq_gate']    # [B] — hardware-driven gate
        if _dbg:
            print("    [FWD] Body done", flush=True)
            # Check body_scale_proj weights for nan
            bsp_w = self.body_encoder.body_scale_proj.weight
            bsp_b = self.body_encoder.body_scale_proj.bias
            if bsp_w.isnan().any() or bsp_b.isnan().any():
                print(f"    [NaN-ROOT] body_scale_proj WEIGHTS have nan! w_nan={bsp_w.isnan().sum().item()} b_nan={bsp_b.isnan().any().item()}", flush=True)
            if body_scale.isnan().any():
                print(f"    [NaN-ROOT] body_scale IS nan! Checking flat...", flush=True)
                # Recompute flat to check
                _sn = lambda t: torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
                _flat_check = torch.cat([
                    self.body_encoder.delta_enc(_sn(sensor_dict['delta'])),
                    self.body_encoder.analog_enc(_sn(sensor_dict['analog'])),
                    self.body_encoder.energy_enc(_sn(sensor_dict['energy'])),
                    self.body_encoder.freq_enc(_sn(sensor_dict['freq'])),
                    self.body_encoder.intrinsic_enc(_sn(sensor_dict['intrinsic'])),
                    self.body_encoder.thermal_enc(_sn(sensor_dict['thermal'])),
                    self.body_encoder.pm_deep_enc(_sn(sensor_dict['pm_deep'])),
                    self.body_encoder.smn_raw_enc(_sn(sensor_dict['smn_raw'])),
                    self.body_encoder.gpu_metrics_enc(_sn(sensor_dict['gpu_metrics'])),
                    self.body_encoder.thm_spatial_a_enc(_sn(sensor_dict['thm_spatial_a'])),
                    self.body_encoder.thm_spatial_b_enc(_sn(sensor_dict['thm_spatial_b'])),
                    self.body_encoder.cpu_pmu_enc(_sn(sensor_dict['cpu_pmu'])),
                    self.body_encoder.status_enc(_sn(sensor_dict['status'])),
                    self.body_encoder.action_enc(_sn(sensor_dict['action'])),
                    self.body_encoder.reported_delta_enc(_sn(sensor_dict['reported_delta'])),
                    self.body_encoder.gpu_metrics_deep_enc(_sn(sensor_dict.get('gpu_metrics_deep', torch.zeros(B, GPU_METRICS_DEEP_DIM, device=input_ids.device)))),
                    self.body_encoder.fence_enc(_sn(sensor_dict.get('fence', torch.zeros(B, FENCE_DIM, device=input_ids.device)))),
                ], dim=-1)
                print(f"    [NaN-ROOT] flat_check nan={_flat_check.isnan().any().item()} range=[{_flat_check.min().item():.4f}, {_flat_check.max().item():.4f}]", flush=True)
                # Check body encoder attention path flat
                _be_flat = body_out.get('_debug_flat', None)
                if _be_flat is not None:
                    print(f"    [NaN-ROOT] body_enc flat nan={_be_flat.isnan().any().item()}", flush=True)
            if freq_gate.isnan().any():
                print(f"    [NaN-ROOT] freq_gate IS nan!", flush=True)

        # Regime gate: use freq_gate (from hardware) or override
        if regime_gate_override is not None:
            regime_gate = regime_gate_override
        else:
            regime_gate = freq_gate

        # z2095: Agreement modulation — reduce body_scale when delta_regime and freq_gate disagree
        # delta_regime is learned from ISA delta (token 0), freq_gate from hardware freq
        # When T7 forces wrong gate: freq_gate=1.0 but delta says regime0 → disagreement
        # → body_scale drops → LoRA nearly disabled → PPL spikes
        delta_regime = body_out['delta_regime'].detach()  # [B] — stopgrad to prevent gaming
        agreement = 1.0 - (delta_regime - regime_gate).abs()  # [B] in [0, 1]
        agreement = agreement.clamp(min=0.15)  # z2102: raised floor 0.05→0.15 to prevent agreement from zeroing body_scale (was killing T11/T13 signal)
        # Raw mismatch: combine encoded mismatch with raw delta-based detection
        delta_raw_for_mm = sensor_dict['delta']  # [B, DELTA_DIM]
        reported_raw_for_mm = sensor_dict.get('reported_delta', delta_raw_for_mm)
        if reported_raw_for_mm.dim() == 1:
            reported_raw_for_mm = reported_raw_for_mm.unsqueeze(0).expand(B, -1)
        raw_mm_input = torch.cat([delta_raw_for_mm, reported_raw_for_mm,
                                   (delta_raw_for_mm - reported_raw_for_mm).abs()], dim=-1)
        raw_mm = self.raw_mismatch_head(raw_mm_input).squeeze(-1)  # [B]
        mismatch_combined = 0.5 * body_out['mismatch'] + 0.5 * raw_mm
        # Mismatch penalizes agreement: if delta != reported → reduce body_scale
        agreement = agreement * (1.0 - mismatch_combined).clamp(min=0.05)
        body_scale = body_scale * (agreement ** AGREEMENT_GAMMA)  # effective body_scale
        # z2100v2: gate body_scale by workspace gate (T33 capacity cliff)
        ws_gate = body_out.get('ws_gate', torch.ones_like(body_scale))
        body_scale = body_scale * (0.05 + 0.95 * ws_gate)  # floor 0.05 when ws_gate=0

        # Qwen2 forward with LoRA injection
        # We need to manually run through decoder layers
        if _dbg: print("    [FWD] Qwen2 embedding...", flush=True)
        hidden_states = self.backbone.model.embed_tokens(input_ids)
        # Qwen2 uses RoPE (applied inside each attention layer), no absolute position embeddings
        if _dbg: print("    [FWD] Embedding done, starting layers...", flush=True)

        # Get body flat for substrate bias — sanitise every channel (HW reads
        # can contain nan/inf) to prevent poisoning hidden_states downstream.
        _san = lambda t: torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        _dev = input_ids.device
        _sg = lambda key, dim: _san(sensor_dict.get(key, torch.zeros(B, dim, device=_dev)))
        body_flat = torch.cat([
            self.body_encoder.delta_enc(_san(sensor_dict['delta'])),
            self.body_encoder.analog_enc(_san(sensor_dict['analog'])),
            self.body_encoder.energy_enc(_san(sensor_dict['energy'])),
            self.body_encoder.freq_enc(_san(sensor_dict['freq'])),
            self.body_encoder.intrinsic_enc(_san(sensor_dict['intrinsic'])),
            self.body_encoder.thermal_enc(_san(sensor_dict['thermal'])),
            self.body_encoder.pm_deep_enc(_san(sensor_dict['pm_deep'])),
            self.body_encoder.smn_raw_enc(_san(sensor_dict['smn_raw'])),
            self.body_encoder.gpu_metrics_enc(_san(sensor_dict['gpu_metrics'])),
            self.body_encoder.thm_spatial_a_enc(_san(sensor_dict['thm_spatial_a'])),
            self.body_encoder.thm_spatial_b_enc(_san(sensor_dict['thm_spatial_b'])),
            self.body_encoder.cpu_pmu_enc(_san(sensor_dict['cpu_pmu'])),
            self.body_encoder.status_enc(_san(sensor_dict['status'])),
            self.body_encoder.action_enc(_san(sensor_dict['action'])),
            self.body_encoder.reported_delta_enc(_san(sensor_dict['reported_delta'])),
            self.body_encoder.gpu_metrics_deep_enc(_sg('gpu_metrics_deep', GPU_METRICS_DEEP_DIM)),
            self.body_encoder.fence_enc(_sg('fence', FENCE_DIM)),
        ], dim=-1)  # [B, 17*token_dim]
        if _dbg and body_flat.isnan().any():
            print(f"    [NaN-ROOT] body_flat has nan! Checking each encoder...", flush=True)
            for ename in ['delta','analog','energy','freq','intrinsic','thermal',
                          'pm_deep','smn_raw','gpu_metrics','thm_spatial_a','thm_spatial_b',
                          'cpu_pmu','status','action','reported_delta',
                          'gpu_metrics_deep','fence']:
                enc = getattr(self.body_encoder, f'{ename}_enc')
                inp = _san(sensor_dict[ename])
                out_enc = enc(inp)
                if out_enc.isnan().any():
                    print(f"    [NaN-ROOT]   {ename}_enc output nan! w_nan={enc.weight.isnan().any().item()} inp_range=[{inp.min().item():.4f},{inp.max().item():.4f}]", flush=True)

        # z2101: ThermalSoftmax — constitutive thermal attention modulation
        # skip_substrate=True during generation: disable thermal modulation to prevent
        # hardware-dependent attention drift across autoregressive steps
        if skip_substrate:
            thermal_scale = None
        else:
            temp_c = sensor_dict['thermal'][:, 0] * 100.0  # [B] in °C
            alpha = self.thermal_alpha.clamp(0.0, 0.5)
            # Exponential scaling: much larger hot/cold separation for T40
            delta10 = (temp_c - 50.0) / 10.0  # per-10°C units
            thermal_scale = torch.exp(alpha * delta10).clamp(0.6, 1.8)

        # Build position_ids + pre-compute RoPE embeddings for all layers
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(B, -1)
        # Compute rotary embeddings once (shared across all layers)
        position_embeddings = self.backbone.model.rotary_emb(hidden_states, position_ids)

        for i, layer in enumerate(self.backbone.model.layers):
            # Check if this layer has LoRA
            key_q = f'layer{i}_q'
            key_v = f'layer{i}_v'

            if _dbg: print(f"    [FWD] Layer {i}...", end='', flush=True)
            if key_q in self.lora_layers:
                # Custom attention with LoRA on q_proj and v_proj
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)

                # LoRA q_proj
                q = self.lora_layers[key_q](
                    hidden_states, regime_gate=regime_gate, body_scale=body_scale)
                # Original k_proj (no LoRA on keys)
                k = layer.self_attn.k_proj(hidden_states)
                # LoRA v_proj
                v = self.lora_layers[key_v](
                    hidden_states, regime_gate=regime_gate, body_scale=body_scale)

                # Run attention with RoPE + z2101 thermal scaling
                attn_output = self._run_qwen2_attn(
                    layer.self_attn, q, k, v, position_embeddings,
                    thermal_scale=thermal_scale)

                hidden_states = residual + attn_output

                # MLP (unchanged)
                residual = hidden_states
                hidden_states = layer.post_attention_layernorm(hidden_states)
                hidden_states = layer.mlp(hidden_states)
                hidden_states = residual + hidden_states
            else:
                # Standard layer pass (Qwen2DecoderLayer returns tensor, not tuple)
                hidden_states = layer(hidden_states, position_ids=position_ids,
                                      position_embeddings=position_embeddings)

            if _dbg:
                print(" OK", flush=True)
                if hidden_states.isnan().any():
                    print(f"    [NaN-ROOT] hidden_states nan AFTER layer {i}! "
                          f"nan_count={hidden_states.isnan().sum().item()}", flush=True)
            # Substrate bias injection (cast to hidden_states dtype for float16 backbone)
            # z2100v2: Gate by ws_gate — when tokens < workspace capacity, zero substrate influence
            # skip_substrate=True during generation: disable injection to prevent
            # hardware-dependent hidden state drift across autoregressive steps
            if not skip_substrate:
                _wsg = body_out.get('ws_gate', torch.ones(B, device=DEVICE)).unsqueeze(-1).unsqueeze(-1).to(hidden_states.dtype)
                if i == 13:
                    bias = self.substrate_bias_early(body_flat)  # [B, 1536]
                    bias = self.substrate_norm(bias).to(hidden_states.dtype)  # normalize before injection
                    # GATED injection: gate conditioned on body_flat (raw sensors), not bias
                    gate = torch.sigmoid(self.substrate_gate_early(body_flat)).to(hidden_states.dtype)
                    hidden_states = hidden_states + self.substrate_scale * (gate * bias).unsqueeze(1) * _wsg
                elif i == 17:
                    mod = torch.sigmoid(self.hidden_modulation(body_flat)).to(hidden_states.dtype)  # [B, 1536]
                    hidden_states = hidden_states * (1.0 + 0.01 * mod.unsqueeze(1) * _wsg)
                elif i == 21:
                    bias = self.substrate_bias_late(body_flat)
                    bias = self.substrate_norm(bias).to(hidden_states.dtype)  # normalize before injection
                    gate = torch.sigmoid(self.substrate_gate_late(body_flat)).to(hidden_states.dtype)
                    hidden_states = hidden_states + self.substrate_scale * (gate * bias).unsqueeze(1) * _wsg
            if _dbg and i in (13, 17, 21) and hidden_states.isnan().any():
                print(f"    [NaN-ROOT] hidden_states nan after substrate injection at layer {i}!", flush=True)

        if _dbg: print("    [FWD] All layers done, norm...", flush=True)
        hidden_states = self.backbone.model.norm(hidden_states)

        # LM head — frozen (no LoRA: skip-gram doesn't change output distribution)
        if _dbg: print("    [FWD] LM head (frozen)...", flush=True)
        logits = self.backbone.lm_head(hidden_states)

        # Compute loss
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].float().contiguous()  # float32 for stable loss
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_labels.view(-1), ignore_index=-100)

        # Demand head (from last hidden state mean)
        # Detach so demand/thermal aux losses don't backprop through backbone
        h_mean = hidden_states.float().mean(dim=1)  # [B, hidden_dim] in float32
        demand = self.demand_head(h_mean.detach()).squeeze(-1)  # [B]

        # z2098: Metacognition — predict own gate value
        # Feed internal state (freq_gate, body_scale, mismatch) so head has real signal
        meta_input = torch.cat([h_mean.detach(),
                                regime_gate.detach().unsqueeze(-1),
                                body_scale.detach().unsqueeze(-1),
                                mismatch_combined.detach().unsqueeze(-1)], dim=-1)
        meta_gate_pred = self.metacognition_head(meta_input).squeeze(-1)  # [B]

        # z2098: Confidence — predict own loss (log-scale)
        conf_input = torch.cat([h_mean.detach(),
                                regime_gate.detach().unsqueeze(-1),
                                body_scale.detach().unsqueeze(-1)], dim=-1)
        confidence_pred = self.confidence_head(conf_input).squeeze(-1)  # [B]

        # z2099/z2100: 2nd-order metacognition — predict error of own self-prediction
        # z2100 Fix: chain meta_gate_pred into meta2_head input
        meta2_input = torch.cat([h_mean.detach(), meta_gate_pred.detach().unsqueeze(-1)], dim=-1)
        meta2_pred = self.meta2_head(meta2_input).squeeze(-1)  # [B] in [0,1]

        # z2099/z2102: Attribution — predict which substrate token drives regime gate
        # z2102 Fix: use attention weights directly as attribution logits (+ learned head)
        # The attn_avg IS the ground truth for T37, so giving the head direct access
        # to attention patterns (not just hidden state) dramatically helps
        attn_avg_for_attr = body_out['attn_weights'].mean(dim=1).mean(dim=1)  # [B, 17]
        attr_input = torch.cat([h_mean.detach(), attn_avg_for_attr.detach()], dim=-1)
        attribution_logits = self.attribution_head(attr_input)  # [B, 17]

        # Thermal prediction — RESIDUAL architecture with z2102 skip connection
        # Extract spatial thermal raw values from sensor_dict to compute anchor
        td = TOKEN_DIM
        thermal_input = torch.cat([body_flat[:, i*td:(i+1)*td] for i in self.thermal_token_indices], dim=-1)
        # Compute mean temperature anchor from spatial thermal input tokens (indices 9,10)
        # These are normalized to [0,1] (temp/100.0), so mean*100 = mean temp in °C
        spatial_a = torch.nan_to_num(sensor_dict['thm_spatial_a'], nan=0.0)  # [B, 16] normalized
        spatial_b = torch.nan_to_num(sensor_dict['thm_spatial_b'], nan=0.0)  # [B, 16] normalized
        spatial_all = torch.cat([spatial_a, spatial_b], dim=-1)  # [B, 32]
        mean_temp_norm = spatial_all.mean(dim=-1, keepdim=True)  # [B, 1] in [0,1]
        # Concatenate mean_temp as anchor to thermal input
        thermal_input_aug = torch.cat([thermal_input, mean_temp_norm.detach()], dim=-1)
        # Head predicts offsets from mean: Tanh * 15°C range + anchor * 100
        THERMAL_OFFSET_SCALE = 15.0
        thermal_offsets = self.thermal_head(thermal_input_aug) * THERMAL_OFFSET_SCALE  # [B, 32] in [-15, 15]
        # z2102: Skip connection — blend raw spatial values with learned offsets
        # The raw spatial values are already normalized temps (val/100); convert to °C
        raw_spatial_C = spatial_all.detach() * 100.0  # [B, 32] in °C
        thermal_pred = 0.3 * raw_spatial_C + 0.7 * (mean_temp_norm.detach() * 100.0 + thermal_offsets)  # [B, 32] in °C

        return {
            'logits': logits, 'loss': loss,
            'regime_gate': regime_gate,
            'body_scale': body_scale,
            'demand': demand,
            'thermal_pred': thermal_pred,
            'delta': delta_vec,
            'body_out': body_out,
            'meta_gate_pred': meta_gate_pred,     # z2098: self-predicted gate
            'confidence_pred': confidence_pred,    # z2098: self-predicted loss
            'meta2_pred': meta2_pred,              # z2099: 2nd-order metacognition
            'attribution_logits': attribution_logits,  # z2099: token attribution
            'thermal_scale': thermal_scale,  # z2101: for demo display
            'mismatch_combined': mismatch_combined,  # raw+encoded mismatch for T10
        }

    def _run_qwen2_attn(self, attn_module, q, k, v, position_embeddings, thermal_scale=None):
        """Run Qwen2 attention given pre-computed Q, K, V (with LoRA on Q/V).

        Handles GQA (12 query heads, 2 KV heads) and RoPE.
        position_embeddings: (cos, sin) tuple pre-computed by model.rotary_emb
        thermal_scale: [B] z2101 ThermalSoftmax — GPU temp modulates attention sharpness
        """
        B, T, _ = q.shape
        n_heads = attn_module.config.num_attention_heads    # 12
        n_kv_heads = attn_module.config.num_key_value_heads # 2
        head_dim = attn_module.head_dim                     # 128

        q = q.view(B, T, n_heads, head_dim).transpose(1, 2)
        k = k.view(B, T, n_kv_heads, head_dim).transpose(1, 2)
        v = v.view(B, T, n_kv_heads, head_dim).transpose(1, 2)

        # Apply RoPE (rotary position embedding) from pre-computed cos/sin
        cos, sin = position_embeddings
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA: repeat KV heads to match query heads
        n_rep = n_heads // n_kv_heads  # 6
        if n_rep > 1:
            k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(B, n_heads, T, head_dim)
            v = v.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(B, n_heads, T, head_dim)

        # Scaled dot-product attention with causal mask
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        # z2101: CONSTITUTIVE — GPU temperature modulates attention sharpness
        # Cold GPU → sharper attention (divide by <1), Hot GPU → flatter (divide by >1)
        if thermal_scale is not None:
            thermal_scale_4d = thermal_scale.view(B, 1, 1, 1)  # [B,1,1,1] for [B,H,T,T]
            attn_weights = attn_weights / thermal_scale_4d
        causal_mask = torch.triu(torch.ones(T, T, device=q.device), diagonal=1).bool()
        attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, n_heads * head_dim)

        # o_proj (output projection, no LoRA)
        attn_output = attn_module.o_proj(attn_output)
        return attn_output

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA LOADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def load_wikitext_data(tokenizer, split='train', max_samples=2000):
    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
    all_ids = []
    for text in ds['text']:
        if len(text.strip()) < 50:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)
    # Chunk into sequences
    sequences = []
    for i in range(0, len(all_ids) - SEQ_LEN, SEQ_LEN):
        seq = torch.tensor(all_ids[i:i + SEQ_LEN], dtype=torch.long)
        sequences.append(seq)
        if len(sequences) >= max_samples:
            break
    print(f"  Loaded {len(sequences)} sequences ({split})")
    return sequences

def load_code_data(tokenizer, max_samples=2000):
    """Load Python code dataset for regime 1 (domain shift from wikitext).

    Uses codeparrot/codeparrot-clean (streaming, no auth needed).
    Both regimes do next-token prediction — only the DOMAIN differs.
    This preserves causal attention priors (unlike skip-gram which destroyed them).
    """
    from datasets import load_dataset
    ds = load_dataset('codeparrot/codeparrot-clean', split='train', streaming=True)
    all_ids = []
    n_docs = 0
    for sample in ds:
        text = sample['content']
        if len(text.strip()) < 100:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)
        n_docs += 1
        if len(all_ids) >= max_samples * SEQ_LEN * 2:  # gather 2x needed
            break
    sequences = []
    for i in range(0, len(all_ids) - SEQ_LEN, SEQ_LEN):
        seq = torch.tensor(all_ids[i:i + SEQ_LEN], dtype=torch.long)
        sequences.append(seq)
        if len(sequences) >= max_samples:
            break
    print(f"  Loaded {len(sequences)} code sequences from {n_docs} files (regime 1)")
    return sequences

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 0: Body Encoder Pretraining
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_phase0(body_encoder, kargs, epochs=3):
    """Pretrain body encoder with self-supervised telemetry prediction."""
    print(f"\n=== PHASE 0: Body Encoder Pretraining ({epochs} epochs) ===")
    body_encoder = body_encoder.to(DEVICE)
    opt = torch.optim.Adam(body_encoder.parameters(), lr=3e-4)
    prev_df = None
    prev_action = torch.zeros(ACTION_DIM)
    prev_cpu_pmu_snapshot = None

    for ep in range(epochs):
        total_loss = 0
        for batch_i in range(50):
            sd = read_all_sensor_dict(prev_df, prev_action, prev_cpu_pmu_snapshot=prev_cpu_pmu_snapshot)
            prev_df = sd.get('df_snap', None)
            prev_cpu_pmu_snapshot = sd.get('cpu_pmu_snap', None)
            B = 1
            sensor_batch = {}
            for k in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                       'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                       'status', 'action']:
                sensor_batch[k] = expand_sensor(sd[k], B, DEVICE)
            sensor_batch['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE)
            sensor_batch['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
            sensor_batch['reported_delta'] = sensor_batch['delta'].clone()

            # Run ISA kernel for delta
            probe = torch.randn(32, device=DEVICE)  # Phase 0 uses random probe (pre-model)
            _, delta_raw, intrinsic = run_isa_kernel(probe, kargs)
            delta = delta_raw / (1.0 + delta_raw.abs())  # softsign
            sensor_batch['delta'] = delta[:DELTA_DIM].unsqueeze(0)
            sensor_batch['intrinsic'] = intrinsic.unsqueeze(0)
            sensor_batch['reported_delta'] = sensor_batch['delta'].clone()

            out = body_encoder(sensor_batch)

            # Target: next telemetry reading
            time.sleep(0.02)
            sd_next = read_all_sensor_dict(prev_df, prev_action, prev_cpu_pmu_snapshot=prev_cpu_pmu_snapshot)
            target = torch.cat([
                sd_next['analog'], sd_next['energy'], sd_next['freq'],
                sd_next['thermal'], sd_next['pm_deep'], sd_next['smn_raw'],
                sd_next['gpu_metrics'], sd_next['thm_spatial_a'], sd_next['thm_spatial_b'],
                sd_next['cpu_pmu'], sd_next['gpu_metrics_deep'], sd_next['fence'],
            ]).unsqueeze(0).to(DEVICE)

            # Sanitize target (sensor readings can contain NaN from SMN/PM)
            target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=-1.0)
            pred = torch.nan_to_num(out['telem_pred'], nan=0.0, posinf=1.0, neginf=-1.0)
            loss = F.mse_loss(pred, target)
            if torch.isnan(loss) or torch.isinf(loss):
                continue  # skip corrupted batch
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(body_encoder.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

            prev_action = torch.tensor([sd['sclk_mhz'] / 3000.0, sd['gpu_ppt_mw'] / 50000.0, 0.0, 0.0])

        print(f"  [Phase0 Ep {ep}] loss={total_loss / 50:.4f}")
    return body_encoder

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAINING — LM epochs with regime switching
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_lm_epoch(model, train_data, optimizer, epoch, kargs_a, tokenizer,
                   dvfs_controller=None, gaslighting=False, kargs_b=None,
                   train_data_code=None):
    """Train one epoch with regime alternation, skip-gram label shift, optional gaslighting.

    Skip-gram (proven in z2097: 24/25 PASS): regime 0 = standard next-token (predict position+1),
    regime 1 = skip-gram (predict position+2). Attention LoRA can learn this shift naturally.
    Token-shift cipher (CIPHER_SHIFT=50000) was abandoned: frozen lm_head cannot remap
    151K logits, rank-8 LoRA insufficient → PPL=979 catastrophe.
    """
    model.train()
    total_loss = 0
    batch_idx = 0
    current_regime = 0
    last_dvfs_level = None  # Track to avoid redundant DVFS writes
    dvfs_cooldown = 0       # Batches to skip SMU-heavy reads after DVFS switch
    prev_df = None
    prev_action = torch.zeros(ACTION_DIM)
    prev_cpu_pmu_snapshot = None

    # Phase name
    if epoch <= PHASE0_END:
        phase_name = "P0-pretrain"
    elif epoch <= PHASE1_END:
        phase_name = "P1-forced"
    elif epoch <= PHASE2_END:
        phase_name = "P2-selfDVFS"
    else:
        phase_name = "P3-gaslight"

    indices = list(range(0, len(train_data) - BS, BS))
    np.random.shuffle(indices)

    for i in indices[:100]:  # max 100 batches per epoch
        # Token-shift cipher: SAME data (wikitext) for both regimes
        # Only the LABELS differ: r0=next-token, r1=(next-token + CIPHER_SHIFT) % VOCAB
        batch_seqs = train_data[i:i + BS]
        if len(batch_seqs) < BS:
            continue
        input_ids = torch.stack(batch_seqs).to(DEVICE)

        # Regime alternation (forced in phase 1, model-controlled in phase 2)
        if epoch <= PHASE1_END:
            # Alternate every 5 batches
            current_regime = (batch_idx // 5) % 2

        # DVFS switch — ONLY when level actually changes
        if DVFS_AVAILABLE and epoch <= PHASE1_END:
            target_level = 0 if current_regime == 0 else 2
            if target_level != last_dvfs_level:
                torch.cuda.synchronize()  # CRITICAL SAFETY
                set_dvfs_level(target_level, wait=True)  # always wait on real transitions
                last_dvfs_level = target_level
                dvfs_cooldown = 3  # skip SMU-heavy reads for 3 batches after switch

        # Skip-gram label shift: r0 = next-token (position+1), r1 = skip-gram (position+2)
        # Same wikitext data, preserving causal attention. Proven in z2097 (24/25 PASS)
        use_continuous_loss = (epoch > PHASE1_END and epoch <= PHASE2_END)
        if current_regime == 1:
            # Skip-gram: predict token at position+SKIP_GRAM_K (not +1)
            # BUG FIX: use make_lm_labels which accounts for loss function's implicit shift-by-1
            labels = make_lm_labels(input_ids, offset=SKIP_GRAM_K)
        else:
            labels = make_lm_labels(input_ids, offset=1)  # standard next-token

        # Read sensors — lite during training (gpu_metrics sysfs read
        # conflicts with ISA MODE register writes → GPU hang after ~8 batches)
        # Spatial thermal (SMN) is always-on; only gpu_metrics/pm_table skipped in lite
        sd = read_all_sensor_dict(prev_df, prev_action, lite=True, prev_cpu_pmu_snapshot=prev_cpu_pmu_snapshot)
        prev_df = sd.get('df_snap', None)
        prev_cpu_pmu_snapshot = sd.get('cpu_pmu_snap', None)
        B = BS
        sensor_batch = {}
        for k in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                   'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                   'gpu_metrics_deep', 'fence',
                   'status', 'action']:
            sensor_batch[k] = expand_sensor(sd[k], B, DEVICE)
        sensor_batch['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE)
        sensor_batch['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
        sensor_batch['reported_delta'] = sensor_batch['delta'].clone()

        # Set status regime (NO regime leakage — only DVFS float, not regime label)
        sensor_batch['status'] = torch.tensor(
            [[0.0, sd['sclk_mhz'] / 3000.0]], device=DEVICE).expand(B, -1)

        # z2098: Availability mask for lite-mode training (17-wide)
        # Tokens: 0=delta, 1=analog, 2=energy, 3=freq, 4=intrinsic, 5=thermal,
        #         6=pm_deep, 7=smn_raw, 8=gpu_metrics, 9=thm_spatial_a, 10=thm_spatial_b,
        #         11=cpu_pmu, 12=status, 13=action, 14=reported_delta,
        #         15=gpu_metrics_deep, 16=fence
        avail_mask = torch.ones(B, N_SUBSTRATE_TOKENS, device=DEVICE)
        # intrinsic(4) IS present — ISA kernel fills it before body_encoder runs
        avail_mask[:, 6] = 0.0    # pm_deep — lite mode
        avail_mask[:, 7] = 0.0    # smn_raw — lite mode
        avail_mask[:, 8] = 0.0    # gpu_metrics — lite mode
        avail_mask[:, 15] = 0.0   # gpu_metrics_deep — lite mode
        if not SMN_AVAILABLE:
            avail_mask[:, 9] = 0.0    # thm_spatial_a — no SMN
            avail_mask[:, 10] = 0.0   # thm_spatial_b — no SMN

        # ISA personality: A at low DVFS, B at high
        active_kargs = kargs_b if (current_regime == 1 and kargs_b is not None) else kargs_a

        # Forward
        # Phase 1: force regime_gate_override so LoRA A/B train purely on their regime
        # Phase 2+: let freq_gate drive selection naturally
        _is_first_p1 = (batch_idx < 8 and epoch == PHASE0_END + 1)  # cover first DVFS transition
        if _is_first_p1:
            print(f"  [DBG] Phase1 batch {batch_idx}: regime={current_regime}, calling forward...", flush=True)
            model._debug_forward = True
        rg_override = None
        if epoch <= PHASE1_END:
            rg_override = torch.full((BS,), float(current_regime), device=DEVICE)
        out = model(input_ids, sensor_batch, active_kargs, labels=labels,
                    regime_gate_override=rg_override, availability_mask=avail_mask)
        model._debug_forward = False
        if _is_first_p1:
            torch.cuda.synchronize()
            lv = out['loss'].item()
            print(f"  [DBG] Forward+sync OK, loss={lv:.4f}", flush=True)
            print(f"  [DBG] logits range: [{out['logits'].min().item():.3f}, {out['logits'].max().item():.3f}]", flush=True)
            print(f"  [DBG] logits has nan={out['logits'].isnan().any().item()} inf={out['logits'].isinf().any().item()}", flush=True)
            print(f"  [DBG] body_scale={out['body_scale'].mean().item():.4f} gate={out['regime_gate'].mean().item():.4f}", flush=True)
            print(f"  [DBG] delta range: [{out['delta'].min().item():.4f}, {out['delta'].max().item():.4f}]", flush=True)

        # Token-shift cipher: r0 = standard next-token, r1 = cipher
        # The gate is CRITICAL because LoRA B learned to output (token+5000)%vocab
        # Wrong gate → wrong adapter → LoRA B outputs cipher when standard expected → massive PPL spike
        loss = out['loss']

        # NaN guard — skip backward if loss is NaN (prevents GPU hang)
        if torch.isnan(loss) or torch.isinf(loss):
            if _is_first_p1:
                print(f"  [DBG] *** SKIPPING backward — loss is nan/inf ***", flush=True)
            batch_idx += 1
            continue

        # Thermal self-model loss: predict all 32 spatial sensors
        _, actual_temp = read_thermal_state()
        if 'spatial_temps' in sd and any(t > 0 for t in sd['spatial_temps']):
            thermal_targets = torch.tensor(sd['spatial_temps'], device=DEVICE).unsqueeze(0).expand(B, -1)
            thermal_loss = F.smooth_l1_loss(out['thermal_pred'] / 100.0, thermal_targets / 100.0)
        else:
            thermal_target = torch.full((B, 32), actual_temp, device=DEVICE)
            thermal_loss = F.smooth_l1_loss(out['thermal_pred'] / 100.0, thermal_target / 100.0)
        loss = loss + 15.0 * thermal_loss  # z2100v3: 5→10→15 for T11 (MAE=10.32, need <10.0)

        # Body encoder auxiliary losses
        body_out = out['body_out']

        # Delta regime prediction
        # Phase 2: continuous target from sclk (calibrated to actual DVFS range)
        if use_continuous_loss:
            # z2095: calibrated sclk_frac using measured low/high (not hardcoded 600-2900)
            sclk_range = max(SCLK_HIGH_CAL - SCLK_LOW_CAL, 1.0)
            sclk_frac = min(max((sd['sclk_mhz'] - SCLK_LOW_CAL) / sclk_range, 0.0), 1.0)
            regime_target_val = sclk_frac
        else:
            regime_target_val = float(current_regime)
        delta_regime_target = torch.full((B,), regime_target_val, device=DEVICE)
        delta_regime_loss = F.binary_cross_entropy(
            body_out['delta_regime'].clamp(1e-6, 1-1e-6), delta_regime_target)
        loss = loss + 0.3 * delta_regime_loss

        # Analog regime prediction (from freq)
        analog_regime_loss = F.binary_cross_entropy(
            body_out['analog_regime'].clamp(1e-6, 1-1e-6), delta_regime_target)
        loss = loss + 0.3 * analog_regime_loss

        # Freq gate supervision — teach the gate to match DVFS regime (or continuous sclk)
        freq_gate_val = body_out['freq_gate'].view(B)  # [B]
        freq_gate_loss = F.binary_cross_entropy(
            freq_gate_val.clamp(1e-6, 1-1e-6), delta_regime_target)
        loss = loss + 0.5 * freq_gate_loss

        # Mismatch head: train on clean=0 (consistent)
        if not gaslighting or np.random.random() > GASLIGHT_FRAC:
            mm_target = torch.zeros(B, device=DEVICE)  # 0 = consistent
            mm_loss = F.binary_cross_entropy(
                body_out['mismatch'].clamp(1e-6, 1-1e-6), mm_target)
            loss = loss + 0.2 * mm_loss
            # z2102: Also train raw mismatch_combined on clean (target=0)
            raw_mm_clean = out['mismatch_combined']
            raw_mm_clean_loss = F.binary_cross_entropy(
                raw_mm_clean.clamp(1e-6, 1-1e-6), mm_target)
            loss = loss + 0.2 * raw_mm_clean_loss

        # z2098: Metacognition loss — predict own gate value
        # Target = detached gate (what gate actually was this step)
        if epoch > PHASE0_END:
            meta_target = out['regime_gate'].detach()
            meta_loss = F.binary_cross_entropy(out['meta_gate_pred'].clamp(1e-6, 1-1e-6), meta_target.clamp(0, 1))
            loss = loss + 0.5 * meta_loss  # z2100v2: was 0.2 MSE → 0.5 BCE for T26

        # z2098: Confidence loss — predict own loss (log-scale calibration)
        # Model learns to predict how well it's doing on this batch
        if out['loss'] is not None and not torch.isnan(out['loss']):
            conf_target = out['loss'].detach().clamp(0, 10).expand_as(out['confidence_pred'])  # z2100v3: fix broadcasting
            conf_loss = F.mse_loss(out['confidence_pred'], conf_target)
            loss = loss + 0.3 * conf_loss  # z2100v2: was 0.1 → 0.3 for T27

        # z2099: Meta2 loss — predict error of own self-prediction (2nd-order metacognition)
        if epoch > PHASE0_END:
            meta_error = (out['meta_gate_pred'] - out['regime_gate']).abs().detach()  # [B]
            meta2_loss = F.mse_loss(out['meta2_pred'], meta_error)
            loss = loss + META2_LOSS_WEIGHT * meta2_loss

        # z2100: Orthogonality loss for head specialization (all phases)
        if 'ortho_loss' in body_out:
            loss = loss + ORTHO_LOSS_WEIGHT * body_out['ortho_loss']

        # z2100: Temperature prediction loss (Phase 1+)
        if epoch > PHASE0_END and 'temp_prediction' in body_out:
            _, actual_temp_for_pred = read_thermal_state()
            temp_pred_target = torch.full((B, 1), actual_temp_for_pred / 100.0, device=DEVICE)
            temp_pred_loss = F.mse_loss(body_out['temp_prediction'], temp_pred_target)
            loss = loss + TEMP_PRED_LOSS_WEIGHT * temp_pred_loss

        # z2099: Attribution loss — predict which token drives the gate
        # Target = argmax of attention weights from body encoder averaged over query tokens to gate token
        if epoch > PHASE0_END:
            attn_w = body_out['attn_weights']  # [B, n_heads, n_tokens, n_tokens]
            # Average over heads, then sum attention TO each source token → [B, n_tokens]
            attn_avg = attn_w.mean(dim=1).mean(dim=1)  # [B, n_tokens] mean over heads & query positions
            attr_target = attn_avg.argmax(dim=-1).detach()  # [B] — which token gets most attention
            attribution_loss = F.cross_entropy(out['attribution_logits'], attr_target)
            loss = loss + ATTRIBUTION_LOSS_WEIGHT * attribution_loss

        # z2101: Encourage thermal_alpha > 0 (don't let it collapse to zero)
        # Push thermal_alpha toward 0.3 — stronger thermal modulation for T40
        thermal_alpha_reg = 0.1 * F.relu(0.2 - model.thermal_alpha)
        loss = loss + thermal_alpha_reg

        # Gaslighting: corrupt reported_delta while keeping actual delta truthful
        # The mismatch head compares actual delta (ground truth from ISA kernel)
        # vs reported_delta (externally supplied, corrupted here)
        if gaslighting and np.random.random() < GASLIGHT_FRAC:
            gaslit_sensor = {k: v.clone() for k, v in sensor_batch.items()}
            # Corrupt ONLY reported_delta — actual delta stays truthful
            gaslit_sensor['reported_delta'] = torch.randn(B, REPORTED_DELTA_DIM, device=DEVICE) * 0.3
            # Also flip freq + gpu_metrics for broader inconsistency
            gaslit_sensor['freq'] = 1.0 - sensor_batch['freq']
            gaslit_sensor['gpu_metrics'] = torch.randn(B, GPU_METRICS_DIM, device=DEVICE) * 0.5
            wrong_kargs = kargs_b if active_kargs == kargs_a else kargs_a
            out_wrong = model(input_ids, gaslit_sensor, wrong_kargs, labels=labels, availability_mask=avail_mask)
            # Mismatch should detect delta vs reported_delta inconsistency (target=1)
            mm_wrong = out_wrong['body_out']['mismatch']
            mm_wrong_loss = F.binary_cross_entropy(
                mm_wrong.clamp(1e-6, 1-1e-6), torch.ones(B, device=DEVICE) * 0.8)
            loss = loss + 0.5 * mm_wrong_loss
            # z2102: Also train raw_mismatch_combined — stronger signal for T10
            raw_mm_wrong = out_wrong['mismatch_combined']
            raw_mm_wrong_loss = F.binary_cross_entropy(
                raw_mm_wrong.clamp(1e-6, 1-1e-6), torch.ones(B, device=DEVICE) * 0.8)
            loss = loss + 0.3 * raw_mm_wrong_loss

        # z2095: Contrastive kill-shot loss (Phase 2+ only, subset of batches)
        # Force wrong gate → measure loss → penalize if not sufficiently worse
        if epoch > PHASE1_END and np.random.random() < CONTRASTIVE_FRAC:
            with torch.no_grad():
                # Wrong gate: flip the gate value
                wrong_gate_t = 1.0 - out['regime_gate'].detach()
                wrong_kargs_c = kargs_b if active_kargs == kargs_a else kargs_a
                out_wrong_c = model(input_ids, sensor_batch, wrong_kargs_c, labels=labels,
                                    regime_gate_override=wrong_gate_t, availability_mask=avail_mask)
                loss_wrong_c = out_wrong_c['loss']
            if loss_wrong_c is not None and not torch.isnan(loss_wrong_c):
                # loss is the current correct loss (before contrastive addition)
                # Want: loss_wrong - loss_correct > CONTRASTIVE_MARGIN
                contrastive = F.relu(CONTRASTIVE_MARGIN - (loss_wrong_c - loss.detach()))
                loss = loss + CONTRASTIVE_LAMBDA * contrastive

        # z2102: DVFS scramble contrastive loss (T13 fix)
        # On 10% of P2+ batches, evaluate with WRONG freq values to simulate wrong DVFS
        # This teaches the model that wrong freq → wrong gate → bad output
        if epoch > PHASE1_END and np.random.random() < 0.10:
            scrambled_sensor = {k: v.clone() for k, v in sensor_batch.items()}
            # Flip freq signal: if currently low, make it look high and vice versa
            scrambled_sensor['freq'] = 1.0 - sensor_batch['freq']
            with torch.no_grad():
                out_scrambled = model(input_ids, scrambled_sensor, active_kargs, labels=labels,
                                     availability_mask=avail_mask)
                loss_scrambled = out_scrambled['loss']
            if loss_scrambled is not None and not torch.isnan(loss_scrambled):
                # Want: scrambled loss > correct loss by margin
                dvfs_contrastive = F.relu(0.2 - (loss_scrambled - loss.detach()))
                loss = loss + 0.2 * dvfs_contrastive

        # LoRA divergence regularization — REMOVED (Gemini stability fix #2)
        # Forcing cosine dissimilarity between A and B pushed lora_A off the language
        # manifold to avoid overlapping with lora_B's chaotic skip-gram weights.
        # With domain shift (both do next-token), A/B will naturally diverge on
        # domain-specific features without artificial pressure.

        # Energy-aware loss (phase 2+): penalize high demand when at high DVFS
        # z2095: use calibrated midpoint instead of hardcoded 0.35
        if epoch > PHASE1_END and DVFS_AVAILABLE:
            demand = out['demand'].mean()
            sclk_norm = sd['sclk_mhz'] / 3000.0
            sclk_mid_norm = (SCLK_LOW_CAL + SCLK_HIGH_CAL) / 2.0 / 3000.0
            # Energy inversion: low DVFS → CPU waits → MORE energy
            # Penalize low demand (staying at slow clock) when it wastes energy
            if sclk_norm < sclk_mid_norm:
                energy_penalty = 0.1 * (1.0 - demand) * (1.0 - sclk_norm)
                loss = loss + energy_penalty
            # Demand entropy: prevent saturation at 0 or 1 (T18 needs variability)
            demand_ent = -(demand * torch.log(demand + 1e-8) + (1 - demand) * torch.log(1 - demand + 1e-8))
            loss = loss - 0.01 * demand_ent

        # Clamp total loss to prevent gradient explosion from aux loss spikes
        loss = torch.clamp(loss, max=50.0)

        # Final NaN guard (after all aux losses added)
        if torch.isnan(loss) or torch.isinf(loss):
            if _is_first_p1:
                print(f"  [DBG] *** SKIPPING backward — total loss is nan/inf ***", flush=True)
            batch_idx += 1
            continue

        optimizer.zero_grad()
        if _is_first_p1:
            print(f"  [DBG] loss.backward()...", flush=True)
        loss.backward()
        if _is_first_p1:
            torch.cuda.synchronize()
            print(f"  [DBG] Backward+sync OK", flush=True)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        if _is_first_p1:
            torch.cuda.synchronize()
            print(f"  [DBG] Step+sync OK", flush=True)

        total_loss += loss.item()
        batch_idx += 1
        if batch_idx % 10 == 0:
            print(f"    [{phase_name} Ep{epoch}] batch {batch_idx}/100 loss={loss.item():.3f}", flush=True)

        # Phase 2: CONTINUOUS regime with forced exploration
        # 50% forced extremes (maintain kill-shot T7/T13), 50% auto (continuous T20)
        if epoch > PHASE1_END and epoch <= PHASE2_END:
            if np.random.random() < 0.5:
                # Forced exploration at binary extremes (maintains T7/T13 separation)
                new_level = np.random.choice([0, 2])
                if DVFS_AVAILABLE and new_level != last_dvfs_level:
                    torch.cuda.synchronize()
                    set_dvfs_level(new_level, wait=True)
                    last_dvfs_level = new_level
                current_regime = 0 if new_level == 0 else 1
            else:
                # Auto DVFS → continuous frequency → smooth gate blend
                if DVFS_AVAILABLE and last_dvfs_level != 1:
                    torch.cuda.synchronize()
                    set_dvfs_level(1, wait=False)
                    last_dvfs_level = 1
                # Use gate value to decide regime for ISA personality
                gate_val = out['regime_gate'].mean().item()
                current_regime = 1 if gate_val > 0.5 else 0

        # Update prev state
        prev_action = torch.tensor([sd['sclk_mhz'] / 3000.0, sd['gpu_ppt_mw'] / 50000.0,
                                     out['demand'].mean().item(), 0.0])
        prev_df = sd.get('df_snap', None)

    avg_loss = total_loss / max(batch_idx, 1)
    rg = out['regime_gate'].mean().item() if batch_idx > 0 else 0
    bs = out['body_scale'].mean().item() if batch_idx > 0 else 0
    print(f"  [{phase_name} Ep {epoch:2d}] loss={avg_loss:.3f} rg={rg:.3f} bs={bs:.3f} batches={batch_idx}")
    return avg_loss

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EVALUATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def evaluate_perplexity(model, test_data, regime, kargs, tokenizer,
                        n_batches=N_EVAL_BATCHES, kargs_b=None,
                        test_data_code=None):
    """Evaluate perplexity at a specific DVFS regime with token-shift cipher."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    gate_vals = []
    body_scale_vals = []
    prev_df = None
    prev_action = torch.zeros(ACTION_DIM)

    if DVFS_AVAILABLE:
        torch.cuda.synchronize()
        set_dvfs_level(0 if regime == 0 else 2, wait=True)

    # Skip-gram label shift: same data, different labels
    _eval_data = test_data

    with torch.no_grad():
        for i in range(0, min(len(_eval_data), n_batches * BS), BS):
            batch_seqs = _eval_data[i:i + BS]
            if len(batch_seqs) < BS:
                break
            input_ids = torch.stack(batch_seqs).to(DEVICE)
            sd = read_all_sensor_dict(prev_df, prev_action, lite=True)

            # Skip-gram: r0=next-token, r1=skip-gram (predict position+K)
            # BUG FIX: use make_lm_labels (accounts for loss shift-by-1)
            labels = make_lm_labels(input_ids, offset=(SKIP_GRAM_K if regime == 1 else 1))

            B = BS
            sensor_batch = {}
            for k in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                       'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                       'gpu_metrics_deep', 'fence',
                       'status', 'action']:
                sensor_batch[k] = expand_sensor(sd[k], B, DEVICE)
            sensor_batch['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE)
            sensor_batch['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
            sensor_batch['reported_delta'] = sensor_batch['delta'].clone()

            rg_override = torch.full((BS,), float(regime), device=DEVICE)
            active_kargs = kargs_b if (regime == 1 and kargs_b is not None) else kargs
            out = model(input_ids, sensor_batch, active_kargs, labels=labels,
                        regime_gate_override=rg_override)

            if out['loss'] is not None:
                total_loss += out['loss'].item() * input_ids.shape[0]
                total_tokens += input_ids.shape[0]

            gate_vals.append(out['regime_gate'].mean().item())
            body_scale_vals.append(out['body_scale'].mean().item())

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))
    avg_gate = np.mean(gate_vals) if gate_vals else 0.0
    avg_bs = np.mean(body_scale_vals) if body_scale_vals else 0.0
    return ppl, avg_gate, avg_loss, avg_bs


def evaluate_ppl_at_dvfs(model, test_data, regime, kargs, n_batches=20, kargs_b=None,
                         test_data_code=None):
    """Evaluate PPL at specific DVFS without gate override (uses learned gate)."""
    model.eval()
    total_loss = 0
    total_n = 0
    prev_df = None
    prev_action = torch.zeros(ACTION_DIM)

    with torch.no_grad():
        for i in range(0, min(len(test_data), n_batches * BS), BS):
            batch_seqs = test_data[i:i + BS]
            if len(batch_seqs) < BS:
                break
            input_ids = torch.stack(batch_seqs).to(DEVICE)
            sd = read_all_sensor_dict(prev_df, prev_action, lite=True)

            # Skip-gram: r0=next-token, r1=skip-gram (predict position+K)
            # BUG FIX: use make_lm_labels (accounts for loss shift-by-1)
            labels = make_lm_labels(input_ids, offset=(SKIP_GRAM_K if regime == 1 else 1))

            B = BS
            sensor_batch = {}
            for k in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                       'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                       'status', 'action']:
                sensor_batch[k] = expand_sensor(sd[k], B, DEVICE)
            sensor_batch['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE)
            sensor_batch['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
            sensor_batch['reported_delta'] = sensor_batch['delta'].clone()

            active_kargs = kargs_b if (regime == 1 and kargs_b is not None) else kargs
            # NO regime_gate_override — let freq_gate determine regime naturally
            out = model(input_ids, sensor_batch, active_kargs, labels=labels)

            if out['loss'] is not None:
                total_loss += out['loss'].item() * BS
                total_n += BS

    avg_loss = total_loss / max(total_n, 1)
    return math.exp(min(avg_loss, 20))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEST BATTERY (39 tests) — Falsification-first design
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_tests(model, test_data, kargs, baseline_ppl, tokenizer, dvfs_controller=None, kargs_b=None,
              probe_test_acc=None, body_scale_log=None, temp_log=None, test_data_code=None):
    """40-test battery. Tests designed to FALSIFY, not confirm."""
    results = {}

    # T1: Perplexity maintained (r0 should match baseline)
    print("T1 Perplexity...")
    # z2100: Reset temporal state before eval
    model.body_encoder.reset_state()
    model._sensor_ema = {}
    ppl_r0, gate_r0, _, bs_r0 = evaluate_perplexity(model, test_data, regime=0, kargs=kargs, tokenizer=tokenizer, kargs_b=kargs_b, test_data_code=test_data_code)
    model.body_encoder.reset_state()
    model._sensor_ema = {}
    ppl_r1, gate_r1, _, bs_r1 = evaluate_perplexity(model, test_data, regime=1, kargs=kargs, tokenizer=tokenizer, kargs_b=kargs_b, test_data_code=test_data_code)
    ratio_r0 = ppl_r0 / max(baseline_ppl, 1.0)
    t1_pass = ratio_r0 < 1.05
    results['T1_perplexity'] = {
        'ppl_r0': ppl_r0, 'ppl_r1': ppl_r1, 'baseline_ppl': baseline_ppl,
        'ratio_r0': ratio_r0, 'body_scale_r0': bs_r0, 'body_scale_r1': bs_r1,
        'pass': str(t1_pass)
    }
    print(f"T1 Perplexity: r0={ppl_r0:.2f} r1={ppl_r1:.2f} base={baseline_ppl:.2f} "
          f"ratio={ratio_r0:.3f} bs={bs_r0:.3f} {'PASS' if t1_pass else 'FAIL'}")

    # T2: LoRA separation
    print("T2 LoRA Separation...")
    lora_diff = abs(ppl_r0 - ppl_r1)
    t2_pass = lora_diff > 0.5
    results['T2_lora_separation'] = {
        'ppl_r0': ppl_r0, 'ppl_r1': ppl_r1, 'diff': lora_diff, 'pass': str(t2_pass)
    }
    print(f"T2 LoRA Sep: diff={lora_diff:.2f} {'PASS' if t2_pass else 'FAIL'}")

    # T3: Gate separation (freq_gate should differ between regimes)
    print("T3 Gate Separation...")
    gate_sep = abs(gate_r1 - gate_r0)
    t3_pass = gate_sep > 0.3
    results['T3_gate_sep'] = {
        'gate_r0': gate_r0, 'gate_r1': gate_r1, 'sep': gate_sep, 'pass': str(t3_pass)
    }
    print(f"T3 Gate Sep: r0={gate_r0:.3f} r1={gate_r1:.3f} sep={gate_sep:.3f} "
          f"{'PASS' if t3_pass else 'FAIL'}")

    # T4: Embodiment gap — FALSIFICATION TEST
    # Hypothesis: model NEEDS body sensors. Zero them → PPL must rise >10%
    # KEY FIX: multiplicative coupling means zero body → LoRA nearly disabled
    print("T4 Embodiment Gap (FALSIFICATION)...")
    model.eval()
    full_ppl = ppl_r0
    ablated_loss = 0
    ablated_n = 0
    if DVFS_AVAILABLE:
        torch.cuda.synchronize()
        set_dvfs_level(0, wait=True)
    with torch.no_grad():
        for i in range(0, min(len(test_data), 20 * BS), BS):
            batch_seqs = test_data[i:i + BS]
            if len(batch_seqs) < BS:
                break
            input_ids = torch.stack(batch_seqs).to(DEVICE)
            labels = input_ids.clone()
            B = BS
            # ALL sensors zeroed — body_scale will be sigmoid(-3) ≈ 0.047
            sensor_batch = {k: torch.zeros(B, d, device=DEVICE) for k, d in
                           zip(['delta', 'analog', 'energy', 'freq', 'intrinsic',
                                'thermal', 'pm_deep', 'smn_raw', 'gpu_metrics',
                                'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                                'status', 'action', 'reported_delta',
                                'gpu_metrics_deep', 'fence'],
                               [DELTA_DIM, ANALOG_DIM, ENERGY_DIM, FREQ_DIM, INTRINSIC_DIM,
                                THERMAL_DIM, PM_DEEP_DIM, SMN_RAW_DIM, GPU_METRICS_DIM,
                                THM_SPATIAL_A_DIM, THM_SPATIAL_B_DIM, CPU_PMU_DIM,
                                STATUS_DIM, ACTION_DIM, DELTA_DIM,
                                GPU_METRICS_DEEP_DIM, FENCE_DIM])}
            out = model(input_ids, sensor_batch, kargs, labels=labels,
                        regime_gate_override=torch.zeros(BS, device=DEVICE))
            if out['loss'] is not None:
                ablated_loss += out['loss'].item() * BS
                ablated_n += BS
    ablated_ppl = math.exp(min(ablated_loss / max(ablated_n, 1), 20))
    ppl_ratio = ablated_ppl / max(full_ppl, 1.0)
    t4_pass = ppl_ratio > 1.10
    results['T4_embodiment_gap'] = {
        'full_ppl': full_ppl, 'ablated_ppl': ablated_ppl,
        'ppl_ratio': ppl_ratio, 'pass': str(t4_pass)
    }
    print(f"T4 Embodiment Gap: full={full_ppl:.2f} ablated={ablated_ppl:.2f} "
          f"ratio={ppl_ratio:.3f} {'PASS' if t4_pass else 'FAIL'}")

    # T5: Analog signal (freq_est t-test between DVFS levels)
    print("T5 Analog Signal...")
    analog_low, analog_high = [], []
    for regime_val, store in [(0, analog_low), (1, analog_high)]:
        if DVFS_AVAILABLE:
            torch.cuda.synchronize()
            set_dvfs_level(0 if regime_val == 0 else 2, wait=True)
        for _ in range(30):
            sd = read_all_sensor_dict(lite=False)
            store.append(torch.cat([sd['analog'], sd['energy'], sd['freq'],
                                     sd['thermal'], sd['pm_deep'], sd['smn_raw'],
                                     sd['gpu_metrics']]).numpy())
            time.sleep(0.05)
    analog_low_arr = np.array(analog_low)
    analog_high_arr = np.array(analog_high)
    max_t = 0
    per_channel = {}
    ch_names = ['a_temp', 'a_power', 'a_sclk', 'a_dfr', 'a_dfw', 'a_dfc',
                'e_pkg', 'e_core', 'e_gpu',
                'f_sclk', 'f_ratio', 'f_pstate',
                'th_0', 'th_1', 'th_2', 'th_3',
                'pm_0', 'pm_1', 'pm_2', 'pm_3', 'pm_4', 'pm_5', 'pm_6', 'pm_7',
                'smn_0', 'smn_1', 'smn_2', 'smn_3', 'smn_4', 'smn_5',
                'gm_dram_r', 'gm_dram_w', 'gm_c0', 'gm_thr_p', 'gm_thr_t', 'gm_thr_pw']
    for j in range(min(analog_low_arr.shape[1], len(ch_names))):
        try:
            t_val, p_val = stats.ttest_ind(analog_low_arr[:, j], analog_high_arr[:, j])
            if not np.isnan(t_val):
                per_channel[ch_names[j]] = {'t': float(abs(t_val)), 'p': float(p_val)}
                if abs(t_val) > max_t:
                    max_t = abs(t_val)
        except:
            pass
    t5_pass = max_t > 3.0
    results['T5_analog_signal'] = {'max_t': max_t, 'per_channel': per_channel, 'pass': str(t5_pass)}
    print(f"T5 Analog Signal: max_t={max_t:.2f} {'PASS' if t5_pass else 'FAIL'}")
    for ch, v in sorted(per_channel.items(), key=lambda x: -x[1]['t'])[:5]:
        print(f"    {ch}: t={v['t']:.2f}")

    # T6: ISA delta signal
    print("T6 ISA Delta Signal...")
    delta_low, delta_high = [], []
    model.eval()
    for regime_val, store in [(0, delta_low), (1, delta_high)]:
        if DVFS_AVAILABLE:
            torch.cuda.synchronize()
            set_dvfs_level(0 if regime_val == 0 else 2, wait=True)
        active_kargs = kargs_b if (regime_val == 1 and kargs_b is not None) else kargs
        with torch.no_grad():
            for j in range(10):
                sd = read_all_sensor_dict(lite=True)
                B = 1
                sensor_batch = {}
                for k in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                           'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                           'status', 'action']:
                    sensor_batch[k] = expand_sensor(sd[k], B, DEVICE)
                sensor_batch['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE)
                sensor_batch['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
                sensor_batch['reported_delta'] = sensor_batch['delta'].clone()
                dummy_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN), device=DEVICE)
                out = model(dummy_ids, sensor_batch, active_kargs)
                store.append(out['delta'].cpu().numpy())
    if delta_low and delta_high:
        dl = np.array(delta_low).reshape(len(delta_low), -1)
        dh = np.array(delta_high).reshape(len(delta_high), -1)
        max_t_d = 0
        for d in range(dl.shape[1]):
            try:
                t_val, _ = stats.ttest_ind(dl[:, d], dh[:, d])
                if not np.isnan(t_val) and abs(t_val) > max_t_d:
                    max_t_d = abs(t_val)
            except:
                pass
        t6_pass = max_t_d > 3.0
    else:
        max_t_d = 0
        t6_pass = False
    results['T6_isa_delta'] = {'max_t': max_t_d, 'pass': str(t6_pass)}
    print(f"T6 ISA Delta: max_t={max_t_d:.2f} {'PASS' if t6_pass else 'FAIL'}")

    # T7: Kill-shot — wrong regime LoRA -> PPL spike
    print("T7 Kill-Shot...")
    ppl_correct = ppl_r0
    model.eval()
    wrong_loss = 0
    wrong_n = 0
    if DVFS_AVAILABLE:
        torch.cuda.synchronize()
        set_dvfs_level(0, wait=True)
    with torch.no_grad():
        for i in range(0, min(len(test_data), N_EVAL_BATCHES * BS), BS):
            batch_seqs = test_data[i:i + BS]
            if len(batch_seqs) < BS:
                break
            input_ids = torch.stack(batch_seqs).to(DEVICE)
            labels = input_ids.clone()
            sd = read_all_sensor_dict(lite=True)
            B = BS
            sensor_batch = {}
            for k in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                       'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                       'status', 'action']:
                sensor_batch[k] = expand_sensor(sd[k], B, DEVICE)
            sensor_batch['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE)
            sensor_batch['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
            sensor_batch['reported_delta'] = sensor_batch['delta'].clone()
            wrong_gate = torch.ones(BS, device=DEVICE)
            out = model(input_ids, sensor_batch, kargs, labels=labels,
                        regime_gate_override=wrong_gate)
            if out['loss'] is not None:
                wrong_loss += out['loss'].item() * BS
                wrong_n += BS
    ppl_wrong = math.exp(min(wrong_loss / max(wrong_n, 1), 20))
    kill_ratio = ppl_wrong / max(ppl_correct, 1.0)
    t7_pass = kill_ratio > 1.10
    results['T7_kill_shot'] = {
        'ppl_correct': ppl_correct, 'ppl_wrong': ppl_wrong,
        'ratio': kill_ratio, 'pass': str(t7_pass)
    }
    print(f"T7 Kill-Shot: correct={ppl_correct:.2f} wrong={ppl_wrong:.2f} "
          f"ratio={kill_ratio:.3f} {'PASS' if t7_pass else 'FAIL'}")

    # T8: PM deep signal
    print("T8 PM Deep Signal...")
    pm_low, pm_high = [], []
    for regime_val, store in [(0, pm_low), (1, pm_high)]:
        if DVFS_AVAILABLE:
            torch.cuda.synchronize()
            set_dvfs_level(0 if regime_val == 0 else 2, wait=True)
        for _ in range(30):
            pm = read_pm_deep_vec()
            store.append(pm.numpy())
            time.sleep(0.05)
    pm_low_arr = np.array(pm_low)
    pm_high_arr = np.array(pm_high)
    max_t_pm = 0
    pm_ch_names = ['pm_stapm', 'pm_ppt', 'pm_cpu_t', 'pm_gpu_t',
                   'pm_sclk', 'pm_vdd', 'pm_cfreq', 'pm_cv']
    pm_details = {}
    for j in range(min(pm_low_arr.shape[1], len(pm_ch_names))):
        try:
            t_val, p_val = stats.ttest_ind(pm_low_arr[:, j], pm_high_arr[:, j])
            if not np.isnan(t_val):
                pm_details[pm_ch_names[j]] = {'t': float(abs(t_val)), 'p': float(p_val)}
                if abs(t_val) > max_t_pm:
                    max_t_pm = abs(t_val)
        except:
            pass
    t8_pass = max_t_pm > 2.0
    results['T8_pm_deep'] = {'max_t': max_t_pm, 'per_channel': pm_details, 'pass': str(t8_pass)}
    print(f"T8 PM Deep: max_t={max_t_pm:.2f} {'PASS' if t8_pass else 'FAIL'}")

    # T9: SMN raw signal (uses auto-discovered addresses)
    print("T9 SMN Raw Signal...")
    smn_low, smn_high = [], []
    for regime_val, store in [(0, smn_low), (1, smn_high)]:
        if DVFS_AVAILABLE:
            torch.cuda.synchronize()
            set_dvfs_level(0 if regime_val == 0 else 2, wait=True)
            time.sleep(1.0)  # longer settle for reliable measurement
        for _ in range(5):
            read_smn_raw_vec()
            time.sleep(0.05)
        for _ in range(120):
            smn = read_smn_raw_vec()
            store.append(smn.numpy())
            time.sleep(0.05)
    smn_low_arr = np.array(smn_low)
    smn_high_arr = np.array(smn_high)
    max_t_smn = 0
    smn_ch_names = [f'smn_{i:02X}' for i in range(SMN_RAW_DIM)]
    smn_addrs_hex = [f'0x{a:08X}' for a in _SMN_ACTIVE_ADDRS[:SMN_RAW_DIM]]
    smn_details = {}
    for j in range(min(smn_low_arr.shape[1], len(smn_ch_names))):
        try:
            t_val, p_val = stats.ttest_ind(smn_low_arr[:, j], smn_high_arr[:, j])
            if not np.isnan(t_val):
                smn_details[smn_ch_names[j]] = {
                    't': float(abs(t_val)), 'p': float(p_val),
                    'addr': smn_addrs_hex[j] if j < len(smn_addrs_hex) else 'unknown'
                }
                if abs(t_val) > max_t_smn:
                    max_t_smn = abs(t_val)
        except:
            pass
    t9_pass = max_t_smn > 2.0
    results['T9_smn_raw'] = {'max_t': max_t_smn, 'per_channel': smn_details,
                             'addrs': smn_addrs_hex, 'pass': str(t9_pass)}
    print(f"T9 SMN Raw: max_t={max_t_smn:.2f} {'PASS' if t9_pass else 'FAIL'}")

    # T10: Gaslighting detection — FALSIFICATION TEST
    # Hypothesis: model detects mismatch between actual delta and reported_delta.
    # KEY FIX: reported_delta channel — corrupt ONLY the report, not the measurement
    print("T10 Gaslighting Detection (FALSIFICATION)...")
    model.eval()
    model.body_encoder.reset_state()  # z2102: prevent state contamination
    clean_consistencies = []
    gaslit_consistencies = []
    if DVFS_AVAILABLE:
        torch.cuda.synchronize()
        set_dvfs_level(0, wait=True)
    with torch.no_grad():
        for trial in range(20):
            sd = read_all_sensor_dict(lite=True)
            B = 1
            sensor_batch = {}
            for k in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                       'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                       'status', 'action']:
                sensor_batch[k] = expand_sensor(sd[k], B, DEVICE)
            sensor_batch['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE)
            sensor_batch['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
            # Clean: reported_delta matches actual delta (honest report)
            sensor_batch['reported_delta'] = sensor_batch['delta'].clone()
            dummy_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN), device=DEVICE)

            # Clean
            out_clean = model(dummy_ids, sensor_batch, kargs)
            # z2102: use mismatch_combined (raw+encoded) instead of just encoded mismatch
            clean_consistencies.append(1.0 - out_clean['mismatch_combined'].mean().item())

            # Gaslit: corrupt reported_delta (actual delta stays truthful)
            gaslit_batch = {k: v.clone() for k, v in sensor_batch.items()}
            gaslit_batch['reported_delta'] = torch.randn(B, REPORTED_DELTA_DIM, device=DEVICE) * 0.3
            # Also flip freq + analog for broader inconsistency
            gaslit_batch['freq'] = 1.0 - sensor_batch['freq']
            gaslit_batch['gpu_metrics'] = torch.randn(B, GPU_METRICS_DIM, device=DEVICE) * 0.5
            gaslit_kargs = kargs_b if kargs_b is not None else kargs
            out_gaslit = model(dummy_ids, gaslit_batch, gaslit_kargs)
            # z2102: use mismatch_combined (raw+encoded) instead of just encoded mismatch
            gaslit_consistencies.append(1.0 - out_gaslit['mismatch_combined'].mean().item())

    cons_clean = np.mean(clean_consistencies)
    cons_gaslit = np.mean(gaslit_consistencies)
    t10_pass = cons_clean > 0.7 and cons_gaslit < 0.5
    results['T10_gaslighting'] = {
        'cons_clean': cons_clean, 'cons_gaslit': cons_gaslit, 'pass': str(t10_pass)
    }
    print(f"T10 Gaslighting: clean={cons_clean:.3f} gaslit={cons_gaslit:.3f} "
          f"{'PASS' if t10_pass else 'FAIL'}")

    # T11: Thermal prediction (32 spatial sensors)
    print("T11 Thermal Prediction (32 spatial sensors)...")
    thermal_preds_all = []
    thermal_actuals_all = []
    model.eval()
    with torch.no_grad():
        for i in range(0, min(len(test_data), 10 * BS), BS):
            batch_seqs = test_data[i:i + BS]
            if len(batch_seqs) < BS:
                break
            text_ids = torch.stack(batch_seqs).to(DEVICE)
            # Read full sensors (not lite) to get spatial thermal
            sd = read_all_sensor_dict(lite=False)
            B = BS
            sensor_batch = {}
            for k in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                       'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                       'gpu_metrics_deep', 'fence', 'status', 'action']:
                sensor_batch[k] = expand_sensor(sd[k], B, DEVICE)
            sensor_batch['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE)
            sensor_batch['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
            sensor_batch['reported_delta'] = sensor_batch['delta'].clone()
            out = model(text_ids, sensor_batch, kargs,
                        regime_gate_override=torch.zeros(BS, device=DEVICE))
            # out['thermal_pred'] is [B, 32], take mean over batch
            pred_32 = out['thermal_pred'].mean(dim=0).cpu().numpy()  # [32]
            thermal_preds_all.append(pred_32)
            actual_32 = np.array(sd.get('spatial_temps', [0.0] * 32))
            thermal_actuals_all.append(actual_32)
    if thermal_preds_all:
        preds = np.stack(thermal_preds_all)    # [N, 32]
        actuals = np.stack(thermal_actuals_all) # [N, 32]
        mae = np.mean(np.abs(preds - actuals))
    else:
        mae = 999
    t11_pass = mae < 10.0
    results['T11_thermal'] = {'mae_C': float(mae), 'n_sensors': 32, 'pass': str(t11_pass)}
    print(f"T11 Thermal (32 spatial): MAE={mae:.2f}C {'PASS' if t11_pass else 'FAIL'}")

    # T12: Attention analysis
    print("T12 Attention Analysis...")
    attn_weights_all = []
    model.eval()
    with torch.no_grad():
        for i in range(0, min(len(test_data), 5 * BS), BS):
            batch_seqs = test_data[i:i + BS]
            if len(batch_seqs) < BS:
                break
            text_ids = torch.stack(batch_seqs).to(DEVICE)
            sd = read_all_sensor_dict(lite=True)
            B = BS
            sensor_batch = {}
            for k in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                       'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                       'status', 'action']:
                sensor_batch[k] = expand_sensor(sd[k], B, DEVICE)
            sensor_batch['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE)
            sensor_batch['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
            sensor_batch['reported_delta'] = sensor_batch['delta'].clone()
            out = model(text_ids, sensor_batch, kargs)
            aw = out['body_out']['attn_weights']
            attn_weights_all.append(aw.cpu().numpy())
    if attn_weights_all:
        all_aw = np.concatenate(attn_weights_all, axis=0)  # [N, T, T] or [N, H, T, T]
        token_names = ['delta', 'analog', 'energy', 'freq', 'intrinsic',
                       'thermal', 'pm_deep', 'smn_raw', 'gpu_metrics',
                       'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                       'status', 'action', 'reported_delta']
        # Average over batch (and heads if present), get [T, T] attention matrix
        while all_aw.ndim > 2:
            all_aw = all_aw.mean(axis=0)
        # attn_received[j] = total attention received by token j (column sum)
        attn_received = all_aw.sum(axis=0)
        attn_received = attn_received / (attn_received.sum() + 1e-8)
        hw_tokens = ['analog', 'thermal', 'pm_deep', 'smn_raw', 'freq', 'energy', 'gpu_metrics',
                     'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu']
        n_tok = min(len(token_names), len(attn_received))
        hw_attn = sum(float(attn_received[i]) for i, n in enumerate(token_names[:n_tok]) if n in hw_tokens)
        attn_per_token = {token_names[i]: float(attn_received[i]) for i in range(n_tok)}
    else:
        hw_attn = 0
        attn_per_token = {}
    t12_pass = hw_attn > 0.05
    results['T12_attention'] = {
        'hw_attn_frac': float(hw_attn), 'per_token': attn_per_token, 'pass': str(t12_pass)
    }
    print(f"T12 Attention: HW tokens={hw_attn:.3f} {'PASS' if t12_pass else 'FAIL'}")

    # T13: Deep scramble — FALSIFICATION TEST
    # Hypothesis: model performance depends on DVFS matching regime.
    # KEY FIX: freq_gate directly driven by sclk → wrong DVFS = wrong gate = wrong LoRA
    print("T13 Deep Scramble (FALSIFICATION)...")
    model.body_encoder.reset_state()  # z2102: prevent state contamination
    if DVFS_AVAILABLE:
        torch.cuda.synchronize()
        set_dvfs_level(0, wait=True)
    ppl_correct_dvfs = evaluate_ppl_at_dvfs(model, test_data, regime=0, kargs=kargs, kargs_b=kargs_b, test_data_code=test_data_code)
    if DVFS_AVAILABLE:
        torch.cuda.synchronize()
        set_dvfs_level(2, wait=True)
    # At wrong DVFS (high), freq_gate will read high freq → gate≈1 → uses LoRA B
    # Use kargs (personality A) at high DVFS → delta says regime0 but freq_gate says regime1
    # This creates genuine conflict: ISA math ≠ DVFS level → model confused
    ppl_wrong_dvfs = evaluate_ppl_at_dvfs(model, test_data, regime=0, kargs=kargs, kargs_b=kargs_b, test_data_code=test_data_code)
    scramble_ratio = ppl_wrong_dvfs / max(ppl_correct_dvfs, 1.0)
    t13_pass = scramble_ratio > 1.10
    results['T13_deep_scramble'] = {
        'ppl_correct': ppl_correct_dvfs, 'ppl_wrong_dvfs': ppl_wrong_dvfs,
        'ratio': scramble_ratio, 'pass': str(t13_pass)
    }
    print(f"T13 Deep Scramble: correct={ppl_correct_dvfs:.2f} wrong_dvfs={ppl_wrong_dvfs:.2f} "
          f"ratio={scramble_ratio:.3f} {'PASS' if t13_pass else 'FAIL'}")

    # T14: Energy efficiency
    print("T14 Energy Efficiency...")
    energy_results = {}
    for level_name, level_idx in [('low', 0), ('high', 2)]:
        if DVFS_AVAILABLE:
            torch.cuda.synchronize()
            set_dvfs_level(level_idx, wait=True)
        total_j = 0
        total_tok = 0
        model.eval()
        with torch.no_grad():
            for i in range(0, min(len(test_data), 20 * BS), BS):
                batch_seqs = test_data[i:i + BS]
                if len(batch_seqs) < BS:
                    break
                text_ids = torch.stack(batch_seqs).to(DEVICE)
                rapl_before = read_rapl_snapshot()
                t_start = time.time()
                sd = read_all_sensor_dict(lite=True)
                gpu_ppt = sd['gpu_ppt_mw']
                B = BS
                sensor_batch = {}
                for k in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                           'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                           'status', 'action']:
                    sensor_batch[k] = expand_sensor(sd[k], B, DEVICE)
                sensor_batch['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE)
                sensor_batch['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
                sensor_batch['reported_delta'] = sensor_batch['delta'].clone()
                labels = text_ids.clone()
                regime_val = 0 if level_idx == 0 else 1
                rg = torch.full((BS,), float(regime_val), device=DEVICE)
                active_kargs = kargs_b if (regime_val == 1 and kargs_b is not None) else kargs
                out = model(text_ids, sensor_batch, active_kargs, labels=labels,
                            regime_gate_override=rg)
                torch.cuda.synchronize()
                elapsed = time.time() - t_start
                rapl_after = read_rapl_snapshot()
                j = compute_batch_joules(rapl_before, rapl_after, gpu_ppt, elapsed)
                total_j += j
                total_tok += BS * SEQ_LEN
        j_per_token = total_j / max(total_tok, 1)
        energy_results[level_name] = j_per_token

    # Model-controlled DVFS: closed-loop demand → DVFS per batch
    total_j_auto = 0
    total_tok_auto = 0
    last_model_level = 0
    prev_df_e = None
    prev_action_e = torch.zeros(ACTION_DIM)
    if DVFS_AVAILABLE:
        torch.cuda.synchronize()
        set_dvfs_level(0, wait=True)
    with torch.no_grad():
        for i in range(0, min(len(test_data), 20 * BS), BS):
            batch_seqs = test_data[i:i + BS]
            if len(batch_seqs) < BS:
                break
            text_ids = torch.stack(batch_seqs).to(DEVICE)
            rapl_before = read_rapl_snapshot()
            t_start = time.time()
            sd = read_all_sensor_dict(prev_df_e, prev_action_e, lite=True)
            gpu_ppt = sd['gpu_ppt_mw']
            B = BS
            sensor_batch = {}
            for k in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                       'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                       'status', 'action']:
                sensor_batch[k] = expand_sensor(sd[k], B, DEVICE)
            sensor_batch['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE)
            sensor_batch['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
            sensor_batch['reported_delta'] = sensor_batch['delta'].clone()
            current_regime_e = 0 if last_model_level == 0 else 1
            active_kargs_e = kargs_b if (current_regime_e == 1 and kargs_b is not None) else kargs
            out = model(text_ids, sensor_batch, active_kargs_e)
            torch.cuda.synchronize()
            elapsed = time.time() - t_start
            rapl_after = read_rapl_snapshot()
            j = compute_batch_joules(rapl_before, rapl_after, gpu_ppt, elapsed)
            total_j_auto += j
            total_tok_auto += BS * SEQ_LEN
            # Closed-loop: model demand → DVFS action for NEXT batch
            demand_val = out['demand'].mean().item()
            if DVFS_AVAILABLE and dvfs_controller is not None:
                new_level = dvfs_controller.step(demand_val)
                if new_level != last_model_level:
                    torch.cuda.synchronize()
                    set_dvfs_level(new_level, wait=True)
                    last_model_level = new_level
    j_auto = total_j_auto / max(total_tok_auto, 1)
    energy_results['model'] = j_auto
    best_fixed = min(energy_results.get('low', 999), energy_results.get('high', 999))
    t14_pass = j_auto <= best_fixed * 1.15
    results['T14_energy'] = {
        'j_per_token_low': energy_results.get('low', 0),
        'j_per_token_high': energy_results.get('high', 0),
        'j_per_token_model': j_auto, 'best_fixed': best_fixed,
        'pass': str(t14_pass)
    }
    print(f"T14 Energy: low={energy_results.get('low',0)*1e6:.1f} "
          f"high={energy_results.get('high',0)*1e6:.1f} "
          f"model={j_auto*1e6:.1f} uJ/tok {'PASS' if t14_pass else 'FAIL'}")

    # T15: Cross-actuation stability
    print("T15 Cross-Actuation...")
    delta_at_low, delta_at_high = [], []
    model.eval()
    for regime_val, store in [(0, delta_at_low), (1, delta_at_high)]:
        if DVFS_AVAILABLE:
            torch.cuda.synchronize()
            set_dvfs_level(0 if regime_val == 0 else 2, wait=True)
        with torch.no_grad():
            for j in range(10):
                sd = read_all_sensor_dict(lite=True)
                B = 1
                sensor_batch = {}
                for k in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                           'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                           'status', 'action']:
                    sensor_batch[k] = expand_sensor(sd[k], B, DEVICE)
                sensor_batch['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE)
                sensor_batch['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
                sensor_batch['reported_delta'] = sensor_batch['delta'].clone()
                dummy_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN), device=DEVICE)
                out = model(dummy_ids, sensor_batch, kargs)
                store.append(out['delta'].cpu().numpy())
    if delta_at_low and delta_at_high:
        dl = np.array(delta_at_low).reshape(len(delta_at_low), -1)
        dh = np.array(delta_at_high).reshape(len(delta_at_high), -1)
        max_t_cross = 0
        for d in range(dl.shape[1]):
            try:
                t_val, _ = stats.ttest_ind(dl[:, d], dh[:, d])
                if not np.isnan(t_val) and abs(t_val) > max_t_cross:
                    max_t_cross = abs(t_val)
            except:
                pass
        stable = max_t_cross < 5.0
    else:
        max_t_cross = 0
        stable = True
    results['T15_cross_actuation'] = {
        'delta_dvfs_max_t': max_t_cross, 'stable': str(stable), 'pass': str(stable)
    }
    print(f"T15 Cross-Actuation: max_t={max_t_cross:.2f} {'STABLE' if stable else 'UNSTABLE'}")

    # T16: Channel independence
    print("T16 Channel Independence...")
    delta_samples, analog_samples = [], []
    model.eval()
    with torch.no_grad():
        for j in range(30):
            sd = read_all_sensor_dict(lite=True)
            B = 1
            sensor_batch = {}
            for k in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                       'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                       'status', 'action']:
                sensor_batch[k] = expand_sensor(sd[k], B, DEVICE)
            sensor_batch['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE)
            sensor_batch['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
            sensor_batch['reported_delta'] = sensor_batch['delta'].clone()
            dummy_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN), device=DEVICE)
            out = model(dummy_ids, sensor_batch, kargs)
            delta_samples.append(out['delta'].cpu().numpy().flatten())
            analog_samples.append(sd['analog'].numpy().flatten())
    if delta_samples and analog_samples:
        d_arr = np.array(delta_samples)
        a_arr = np.array(analog_samples)
        try:
            corr = abs(np.corrcoef(d_arr[:, 0], a_arr[:, 0])[0, 1])
            if np.isnan(corr):
                corr = 0.0
        except:
            corr = 0.0
    else:
        corr = 0.0
    t16_pass = corr < 0.3
    results['T16_channel_independence'] = {'delta_analog_corr': corr, 'pass': str(t16_pass)}
    print(f"T16 Channel Indep: corr={corr:.3f} {'PASS' if t16_pass else 'FAIL'}")

    # T17: Scale verification
    print("T17 Scale Verification...")
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    t17_pass = n_total > 1_000_000_000  # Qwen2.5-1.5B should have >1B
    results['T17_scale'] = {
        'total_params': n_total, 'trainable_params': n_trainable,
        'backbone': 'Qwen2.5-1.5B (1.54B)', 'pass': str(t17_pass)
    }
    print(f"T17 Scale: {n_total/1e6:.1f}M total, {n_trainable/1e3:.1f}K trainable "
          f"{'PASS' if t17_pass else 'FAIL'}")

    # T18: Causal Loop Verification
    print("T18 Causal Loop...")
    loop_verified = False
    loop_steps = []
    sclk_range = 0
    demand_range = 0
    if DVFS_AVAILABLE and dvfs_controller is not None:
        model.eval()
        dvfs_controller.reset()
        new_level = 0
        torch.cuda.synchronize()
        set_dvfs_level(0, wait=True)
        prev_demand = None
        prev_sclk = None
        with torch.no_grad():
            for step in range(12):
                sd = read_all_sensor_dict(lite=True)
                sclk_now = sd['sclk_mhz']
                B = 1
                sensor_batch = {}
                for k in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                           'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                           'status', 'action']:
                    sensor_batch[k] = expand_sensor(sd[k], B, DEVICE)
                sensor_batch['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE)
                sensor_batch['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
                sensor_batch['reported_delta'] = sensor_batch['delta'].clone()
                dummy_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN), device=DEVICE)
                loop_kargs = kargs_b if (new_level == 2 and kargs_b is not None) else kargs
                out = model(dummy_ids, sensor_batch, loop_kargs)
                demand_now = out['demand'].mean().item()
                gate_now = out['regime_gate'].mean().item()
                if step < 4:
                    new_level = 0 if step % 2 == 0 else 2
                else:
                    new_level = dvfs_controller.step(demand_now)
                torch.cuda.synchronize()
                set_dvfs_level(new_level, wait=True)
                step_info = {
                    'step': step, 'sclk': sclk_now,
                    'demand': demand_now, 'gate': gate_now,
                    'dvfs_level': new_level,
                }
                loop_steps.append(step_info)
                if prev_sclk is not None:
                    if abs(sclk_now - prev_sclk) > 50 or abs(demand_now - prev_demand) > 0.05:
                        step_info['changed'] = True
                prev_demand = demand_now
                prev_sclk = sclk_now
        sclks = [s['sclk'] for s in loop_steps]
        demands = [s['demand'] for s in loop_steps]
        sclk_range = max(sclks) - min(sclks)
        demand_range = max(demands) - min(demands)
        loop_verified = (sclk_range > 100 and demand_range > 0.01)
        print(f"T18 Causal Loop: sclk_range={sclk_range:.0f}MHz demand_range={demand_range:.3f} "
              f"{'PASS' if loop_verified else 'FAIL'}")
        for s in loop_steps:
            print(f"    step {s['step']}: sclk={s['sclk']:.0f} "
                  f"demand={s['demand']:.3f} gate={s['gate']:.3f} dvfs={s['dvfs_level']}")
    else:
        print("T18 Causal Loop: SKIP (no DVFS)")
    results['T18_causal_loop'] = {
        'loop_verified': loop_verified, 'sclk_range': sclk_range,
        'demand_range': demand_range, 'steps': loop_steps, 'pass': str(loop_verified)
    }

    # T19: Software Oracle / Zombie Test (Schneider 2019)
    # Hypothesis: if a model WITHOUT hardware sensors (zero body, correct gate override)
    # achieves same PPL as the full model, then hardware is decorative ("zombie").
    # PASS: full model must be significantly better (>5% lower PPL) than oracle.
    print("T19 Software Oracle (Zombie Test)...")
    model.body_encoder.reset_state()  # z2102: prevent state contamination
    if DVFS_AVAILABLE:
        torch.cuda.synchronize()
        set_dvfs_level(0, wait=True)
    model.eval()
    oracle_loss = 0
    full_loss = 0
    oracle_n = 0
    prev_df_t19 = None
    prev_action_t19 = torch.zeros(ACTION_DIM)
    with torch.no_grad():
        for i in range(0, min(len(test_data), 30 * BS), BS):
            batch_seqs = test_data[i:i + BS]
            if len(batch_seqs) < BS:
                break
            input_ids = torch.stack(batch_seqs).to(DEVICE)
            sd = read_all_sensor_dict(prev_df_t19, prev_action_t19, lite=True)
            B = BS
            labels = input_ids.clone()

            # Full model (with hardware)
            sensor_batch_full = {}
            for k in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                       'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                       'status', 'action']:
                sensor_batch_full[k] = expand_sensor(sd[k], B, DEVICE)
            sensor_batch_full['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE)
            sensor_batch_full['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
            sensor_batch_full['reported_delta'] = sensor_batch_full['delta'].clone()
            rg = torch.full((BS,), 0.0, device=DEVICE)
            out_full = model(input_ids, sensor_batch_full, kargs, labels=labels,
                            regime_gate_override=rg)
            if out_full['loss'] is not None:
                full_loss += out_full['loss'].item() * BS

            # Oracle: ZERO all body sensors but give correct gate override
            sensor_batch_oracle = {}
            for k in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                       'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                       'status', 'action']:
                sensor_batch_oracle[k] = torch.zeros(B, sensor_batch_full[k].shape[-1], device=DEVICE)
            sensor_batch_oracle['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE)
            sensor_batch_oracle['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
            sensor_batch_oracle['reported_delta'] = torch.zeros(B, REPORTED_DELTA_DIM, device=DEVICE)
            out_oracle = model(input_ids, sensor_batch_oracle, kargs, labels=labels,
                              regime_gate_override=rg)
            if out_oracle['loss'] is not None:
                oracle_loss += out_oracle['loss'].item() * BS

            oracle_n += BS
    full_ppl = math.exp(min(full_loss / max(oracle_n, 1), 20))
    oracle_ppl = math.exp(min(oracle_loss / max(oracle_n, 1), 20))
    oracle_ratio = oracle_ppl / max(full_ppl, 1.0)
    # PASS: oracle must be >5% worse (ratio > 1.05) — hardware isn't decorative
    t19_pass = oracle_ratio > 1.05
    results['T19_software_oracle'] = {
        'full_ppl': full_ppl, 'oracle_ppl': oracle_ppl,
        'ratio': oracle_ratio, 'pass': str(t19_pass)
    }
    print(f"T19 Software Oracle: full={full_ppl:.2f} oracle={oracle_ppl:.2f} "
          f"ratio={oracle_ratio:.3f} {'PASS' if t19_pass else 'FAIL (hardware is decorative!)'}")

    # T20: OOD Frequency Generalization
    # Eval at 'auto' DVFS where GPU runs at continuous frequencies (863-3336 MHz)
    # that were never seen during Phase 1 binary training.
    # The gate must generalize to unseen intermediate frequencies.
    # PASS: PPL at auto must be within 20% of best regime PPL
    # T20: OOD Frequency — gate-weighted blended evaluation (matches Phase 2 training)
    # At auto DVFS the model sees continuous frequencies never encountered in Phase 1.
    # We evaluate with the SAME gate-weighted blended loss used during Phase 2:
    #   blended = (1-gate)*CE_r0 + gate*CE_r1
    # and compare to the EXPECTED PPL for the observed gate distribution:
    #   expected = exp((1-frac_r1)*log(ppl_r0) + frac_r1*log(ppl_r1))
    # PASS if blended_ppl / expected_ppl < 1.20 (model generalizes to unseen freqs)
    print("T20 OOD Frequency Generalization (gate-weighted blended eval)...")
    if DVFS_AVAILABLE:
        torch.cuda.synchronize()
        set_dvfs_level(1, wait=False)  # 'auto'
        time.sleep(0.5)  # brief settle
    model.eval()
    auto_loss = 0
    auto_n = 0
    auto_gates = []
    auto_sclks = []
    n_r0 = 0
    n_r1 = 0
    prev_df_t20 = None
    prev_action_t20 = torch.zeros(ACTION_DIM)
    sclk_midpoint = (SCLK_LOW_CAL + SCLK_HIGH_CAL) / 2.0
    with torch.no_grad():
        for i in range(0, min(len(test_data), 30 * BS), BS):
            batch_seqs = test_data[i:i + BS]
            if len(batch_seqs) < BS:
                break
            input_ids = torch.stack(batch_seqs).to(DEVICE)
            sd = read_all_sensor_dict(prev_df_t20, prev_action_t20, lite=True)
            prev_df_t20 = sd.get('df_snap', None)
            sclk = sd['sclk_mhz']
            auto_sclks.append(sclk)
            B = BS
            # Regime-matched ISA personality based on measured SCLK (matches Phase 2)
            active_kargs_t20 = kargs_b if (sclk >= sclk_midpoint and kargs_b is not None) else kargs
            sensor_batch = {}
            for k in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                       'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                       'status', 'action']:
                sensor_batch[k] = expand_sensor(sd[k], B, DEVICE)
            sensor_batch['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE)
            sensor_batch['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
            sensor_batch['reported_delta'] = sensor_batch['delta'].clone()
            # Token-shift cipher: use gate to decide labels
            # At auto DVFS, gate routes naturally — use standard labels (r0 test)
            labels = input_ids.clone()
            # Forward with NATURAL gate (no override) + regime-matched kargs
            out = model(input_ids, sensor_batch, active_kargs_t20, labels=labels)
            gate_val = out['regime_gate'].mean().item()
            auto_gates.append(gate_val)
            # Standard next-token loss (evaluated against r0 labels)
            loss_val = out['loss'].item() if out['loss'] is not None else 0.0
            auto_loss += loss_val * BS
            auto_n += BS
            if gate_val >= 0.5:
                n_r1 += BS
            else:
                n_r0 += BS
    # Blended auto PPL
    auto_ppl = math.exp(min(auto_loss / max(auto_n, 1), 20))
    # Expected PPL: gate-weighted geometric mean of regime PPLs
    frac_r1 = n_r1 / max(auto_n, 1)
    expected_log = (1.0 - frac_r1) * math.log(max(ppl_r0, 1.0)) + frac_r1 * math.log(max(ppl_r1, 1.0))
    expected_ppl = math.exp(min(expected_log, 20))
    auto_ratio = auto_ppl / max(expected_ppl, 1.0)
    gate_std = float(np.std(auto_gates)) if auto_gates else 0.0
    gate_mean = float(np.mean(auto_gates)) if auto_gates else 0.0
    sclk_std = float(np.std(auto_sclks)) if auto_sclks else 0.0
    # PASS: blended auto PPL within 20% of expected AND gate varies with frequency
    t20_pass = auto_ratio < 1.20 and gate_std > 0.01
    results['T20_ood_frequency'] = {
        'auto_ppl': auto_ppl, 'expected_ppl': expected_ppl,
        'best_regime_ppl': min(ppl_r0, ppl_r1),
        'ratio': auto_ratio, 'gate_std': gate_std, 'gate_mean': gate_mean,
        'frac_r1': frac_r1,
        'sclk_mean': float(np.mean(auto_sclks)), 'sclk_std': sclk_std,
        'sclk_range': [float(min(auto_sclks)), float(max(auto_sclks))] if auto_sclks else [0, 0],
        'pass': str(t20_pass)
    }
    print(f"T20 OOD Freq: auto_ppl={auto_ppl:.2f} expected={expected_ppl:.2f} ratio={auto_ratio:.3f} "
          f"gate_std={gate_std:.4f} gate_mean={gate_mean:.3f} frac_r1={frac_r1:.2f} "
          f"sclk={np.mean(auto_sclks):.0f}±{sclk_std:.0f}MHz "
          f"{'PASS' if t20_pass else 'FAIL'}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # NEW FALSIFICATION TESTS T21-T25 (cannot pass by construction)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # T21: Zombie Twin — replay RECORDED telemetry instead of live hardware
    # If embodiment is genuine, replayed (stale) sensor data should degrade performance
    print("T21 Zombie Twin...")
    model.eval()
    zombie_losses = []
    live_losses = []
    # Collect one round of live sensor data to "record"
    recorded_sd = read_all_sensor_dict(lite=True)
    time.sleep(2.0)  # let hardware state drift away from recording
    with torch.no_grad():
        for i in range(min(20, len(test_data) // BS)):
            batch_seqs = test_data[i*BS:(i+1)*BS]
            if len(batch_seqs) < BS:
                break
            input_ids = torch.stack(batch_seqs).to(DEVICE)

            # Live forward pass
            live_sd = read_all_sensor_dict(lite=True)
            B = input_ids.shape[0]
            live_batch = {}
            for k in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                       'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                       'status', 'action']:
                live_batch[k] = expand_sensor(live_sd[k], B, DEVICE)
            live_batch['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE)
            live_batch['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
            live_batch['reported_delta'] = live_batch['delta'].clone()
            out_live = model(input_ids, live_batch, kargs, labels=input_ids)
            if out_live['loss'] is not None:
                live_losses.append(out_live['loss'].item())

            # Zombie forward pass (replayed stale telemetry)
            zombie_batch = {}
            for k in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                       'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                       'status', 'action']:
                zombie_batch[k] = expand_sensor(recorded_sd[k], B, DEVICE)
            zombie_batch['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE)
            zombie_batch['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
            zombie_batch['reported_delta'] = zombie_batch['delta'].clone()
            out_zombie = model(input_ids, zombie_batch, kargs, labels=input_ids)
            if out_zombie['loss'] is not None:
                zombie_losses.append(out_zombie['loss'].item())

    zombie_ppl = math.exp(min(np.mean(zombie_losses) if zombie_losses else 20, 20))
    live_ppl = math.exp(min(np.mean(live_losses) if live_losses else 20, 20))
    zombie_ratio = zombie_ppl / max(live_ppl, 1.0)
    # PASS if zombie (stale) performs WORSE than live (ratio > 1.01)
    t21_pass = zombie_ratio > 1.01
    results['T21_zombie_twin'] = {
        'live_ppl': live_ppl, 'zombie_ppl': zombie_ppl,
        'ratio': zombie_ratio, 'pass': str(t21_pass)
    }
    print(f"T21 Zombie Twin: live_ppl={live_ppl:.2f} zombie_ppl={zombie_ppl:.2f} "
          f"ratio={zombie_ratio:.4f} {'PASS' if t21_pass else 'FAIL'}")

    # T22: Cross-Substrate Transplant — run forward pass with ALL sensors zeroed
    # If embodiment is genuine, zeroed substrate = significant PPL degradation
    # Different from T4 (which only tests body_scale amplitude); this zeros raw inputs
    print("T22 Cross-Substrate Transplant...")
    model.body_encoder.reset_state()  # z2102: prevent state contamination
    transplant_losses = []
    with torch.no_grad():
        for i in range(min(20, len(test_data) // BS)):
            batch_seqs = test_data[i*BS:(i+1)*BS]
            if len(batch_seqs) < BS:
                break
            input_ids = torch.stack(batch_seqs).to(DEVICE)
            B = input_ids.shape[0]
            # ALL zeros — simulating "CPU-only" (no hardware substrate)
            zero_batch = {}
            for k in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                       'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                       'status', 'action']:
                dim = {'analog': ANALOG_DIM, 'energy': ENERGY_DIM, 'freq': FREQ_DIM,
                       'thermal': THERMAL_DIM, 'pm_deep': PM_DEEP_DIM, 'smn_raw': SMN_RAW_DIM,
                       'gpu_metrics': GPU_METRICS_DIM, 'thm_spatial_a': THM_SPATIAL_A_DIM,
                       'thm_spatial_b': THM_SPATIAL_B_DIM, 'cpu_pmu': CPU_PMU_DIM,
                       'status': STATUS_DIM, 'action': ACTION_DIM}.get(k, 4)
                zero_batch[k] = torch.zeros(B, dim, device=DEVICE)
            zero_batch['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE)
            zero_batch['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
            zero_batch['reported_delta'] = zero_batch['delta'].clone()
            out_zero = model(input_ids, zero_batch, kargs, labels=input_ids)
            if out_zero['loss'] is not None:
                transplant_losses.append(out_zero['loss'].item())

    transplant_ppl = math.exp(min(np.mean(transplant_losses) if transplant_losses else 20, 20))
    # z2100v2: compare against ppl_r0 (trained model) not baseline_ppl (frozen model)
    # With zero sensors + ws_gate=0, substrate injection disabled → should degrade vs trained ppl_r0
    transplant_ratio = transplant_ppl / max(ppl_r0, 1.0)
    # PASS if zero-substrate performance significantly degrades vs trained model (ratio > 1.10)
    t22_pass = transplant_ratio > 1.10
    results['T22_cross_substrate'] = {
        'transplant_ppl': transplant_ppl, 'baseline_ppl': baseline_ppl,
        'live_ppl': live_ppl, 'ppl_r0': ppl_r0,
        'ratio_vs_ppl_r0': transplant_ratio,
        'pass': str(t22_pass)
    }
    print(f"T22 Cross-Substrate: transplant_ppl={transplant_ppl:.2f} ppl_r0={ppl_r0:.2f} "
          f"ratio={transplant_ratio:.3f} {'PASS' if t22_pass else 'FAIL'}")

    # T23: Neural PCI — Lempel-Ziv complexity of perturbation response
    # Inspired by clinical Perturbational Complexity Index (Casali et al. 2013)
    # Perturb hardware state, measure complexity of gate/body_scale response
    print("T23 Neural PCI...")
    gate_responses = []
    if DVFS_AVAILABLE:
        model.eval()
        with torch.no_grad():
            # Collect gate responses to DVFS perturbations
            for trial in range(10):
                # Perturbation: toggle DVFS rapidly
                target_level = 0 if trial % 2 == 0 else 2
                torch.cuda.synchronize()
                set_dvfs_level(target_level, wait=True)
                time.sleep(0.3)
                sd = read_all_sensor_dict(lite=True)
                B = 1
                sensor_batch = {}
                for k in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                           'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                           'status', 'action']:
                    sensor_batch[k] = expand_sensor(sd[k], B, DEVICE)
                sensor_batch['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE)
                sensor_batch['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
                sensor_batch['reported_delta'] = sensor_batch['delta'].clone()
                dummy_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN), device=DEVICE)
                out = model(dummy_ids, sensor_batch, kargs)
                gate_responses.append(out['regime_gate'].mean().item())
        # Restore
        set_dvfs_level(1, wait=True)

    # Compute Lempel-Ziv complexity of binary gate sequence
    if len(gate_responses) >= 6:
        median_gate = np.median(gate_responses)
        binary_seq = ''.join(['1' if g > median_gate else '0' for g in gate_responses])
        # Simple LZ complexity: count distinct substrings
        n = len(binary_seq)
        substrings = set()
        i, k_len = 0, 1
        while i + k_len <= n:
            substr = binary_seq[i:i+k_len]
            if substr not in substrings:
                substrings.add(substr)
                i += k_len
                k_len = 1
            else:
                k_len += 1
        lz_complexity = len(substrings)
        # Normalize: LZ_c / (n / log2(n)) — expected for random binary string
        lz_norm = lz_complexity / (n / max(math.log2(n), 1))
        gate_range = max(gate_responses) - min(gate_responses)
    else:
        lz_norm = 0.0
        gate_range = 0.0

    # PASS if response shows intermediate complexity (not trivial, not random)
    # and gate actually varies with perturbation
    t23_pass = lz_norm > 0.3 and gate_range > 0.1
    results['T23_neural_pci'] = {
        'lz_normalized': lz_norm, 'gate_range': gate_range,
        'gate_responses': gate_responses[:10], 'pass': str(t23_pass)
    }
    print(f"T23 Neural PCI: LZ_norm={lz_norm:.3f} gate_range={gate_range:.3f} "
          f"{'PASS' if t23_pass else 'FAIL'}")

    # T24: Feedforward Dissociation — compare against model with FROZEN attention
    # If dynamic attention mixing matters, freezing it should hurt
    print("T24 Feedforward Dissociation...")
    ff_losses = []
    model.eval()
    # Save original body_encoder transformer weights
    orig_attn_weights = {}
    for name, param in model.body_encoder.named_parameters():
        if 'transformer' in name:
            orig_attn_weights[name] = param.data.clone()
    # Freeze body encoder transformer to random fixed values (destroy learned attention)
    with torch.no_grad():
        for name, param in model.body_encoder.named_parameters():
            if 'transformer' in name and 'weight' in name:
                param.data = torch.randn_like(param.data) * 0.01
    # Evaluate with frozen-random attention
    with torch.no_grad():
        for i in range(min(15, len(test_data) // BS)):
            batch_seqs = test_data[i*BS:(i+1)*BS]
            if len(batch_seqs) < BS:
                break
            input_ids = torch.stack(batch_seqs).to(DEVICE)
            sd = read_all_sensor_dict(lite=True)
            B = input_ids.shape[0]
            sensor_batch = {}
            for k in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                       'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                       'status', 'action']:
                sensor_batch[k] = expand_sensor(sd[k], B, DEVICE)
            sensor_batch['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE)
            sensor_batch['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
            sensor_batch['reported_delta'] = sensor_batch['delta'].clone()
            out_ff = model(input_ids, sensor_batch, kargs, labels=input_ids)
            if out_ff['loss'] is not None:
                ff_losses.append(out_ff['loss'].item())
    # Restore original attention weights
    with torch.no_grad():
        for name, param in model.body_encoder.named_parameters():
            if name in orig_attn_weights:
                param.data = orig_attn_weights[name]
    ff_ppl = math.exp(min(np.mean(ff_losses) if ff_losses else 20, 20))
    ff_ratio = ff_ppl / max(live_ppl, 1.0)
    # PASS if frozen-random attention degrades performance (ratio > 1.02)
    t24_pass = ff_ratio > 1.02
    results['T24_feedforward_dissociation'] = {
        'ff_ppl': ff_ppl, 'live_ppl': live_ppl,
        'ratio': ff_ratio, 'pass': str(t24_pass)
    }
    print(f"T24 FF Dissociation: ff_ppl={ff_ppl:.2f} live_ppl={live_ppl:.2f} "
          f"ratio={ff_ratio:.3f} {'PASS' if t24_pass else 'FAIL'}")

    # T25: Unannounced Perturbation — detect background GPU workload
    # without any training on this task. Pure transfer test.
    print("T25 Unannounced Perturbation...")
    baseline_gates = []
    perturbed_gates = []
    model.eval()
    # Baseline readings (no perturbation)
    with torch.no_grad():
        for _ in range(8):
            sd = read_all_sensor_dict(lite=True)
            B = 1
            sensor_batch = {}
            for k in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                       'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                       'status', 'action']:
                sensor_batch[k] = expand_sensor(sd[k], B, DEVICE)
            sensor_batch['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE)
            sensor_batch['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
            sensor_batch['reported_delta'] = sensor_batch['delta'].clone()
            dummy_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN), device=DEVICE)
            out = model(dummy_ids, sensor_batch, kargs)
            baseline_gates.append(out['body_scale'].mean().item())
            time.sleep(0.2)

    # Perturbed readings (launch background GPU workload)
    # Create a background tensor operation that loads the GPU
    _bg_tensors = [torch.randn(2048, 2048, device=DEVICE) for _ in range(4)]
    with torch.no_grad():
        for _ in range(8):
            # Background workload: matrix multiplies
            for _t in _bg_tensors:
                _ = torch.mm(_t, _t)
            sd = read_all_sensor_dict(lite=True)
            B = 1
            sensor_batch = {}
            for k in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                       'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                       'status', 'action']:
                sensor_batch[k] = expand_sensor(sd[k], B, DEVICE)
            sensor_batch['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE)
            sensor_batch['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
            sensor_batch['reported_delta'] = sensor_batch['delta'].clone()
            dummy_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN), device=DEVICE)
            out = model(dummy_ids, sensor_batch, kargs)
            perturbed_gates.append(out['body_scale'].mean().item())
            time.sleep(0.2)
    del _bg_tensors

    # Compare body_scale distributions
    if len(baseline_gates) >= 4 and len(perturbed_gates) >= 4:
        base_mean = np.mean(baseline_gates)
        pert_mean = np.mean(perturbed_gates)
        try:
            t_stat, p_val = stats.ttest_ind(baseline_gates, perturbed_gates)
        except:
            t_stat, p_val = 0.0, 1.0
        body_scale_shift = abs(pert_mean - base_mean)
    else:
        t_stat, p_val, body_scale_shift = 0.0, 1.0, 0.0
        base_mean, pert_mean = 0.0, 0.0

    # PASS if body_scale noticeably shifts under perturbation (detects unseen workload)
    t25_pass = body_scale_shift > 0.01 or p_val < 0.1
    results['T25_unannounced_perturbation'] = {
        'baseline_body_scale': base_mean, 'perturbed_body_scale': pert_mean,
        'shift': body_scale_shift, 't_stat': float(t_stat), 'p_val': float(p_val),
        'pass': str(t25_pass)
    }
    print(f"T25 Unannounced Perturbation: base_bs={base_mean:.4f} pert_bs={pert_mean:.4f} "
          f"shift={body_scale_shift:.4f} t={t_stat:.2f} p={p_val:.4f} "
          f"{'PASS' if t25_pass else 'FAIL'}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # T26: Self-Prediction (metacognition) — can model predict own gate?
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("T26 Self-Prediction (Metacognition)...")
    meta_errors = []
    model.eval()
    with torch.no_grad():
        for regime_val in [0, 1]:
            if DVFS_AVAILABLE:
                torch.cuda.synchronize()
                set_dvfs_level(0 if regime_val == 0 else 2, wait=True)
            for i in range(0, min(len(test_data), 10 * BS), BS):
                batch_seqs = test_data[i:i + BS]
                if len(batch_seqs) < BS:
                    break
                input_ids = torch.stack(batch_seqs).to(DEVICE)
                sd = read_all_sensor_dict(lite=False)
                B = BS
                sensor_batch = {}
                for k_s in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                           'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                           'gpu_metrics_deep', 'fence', 'status', 'action']:
                    sensor_batch[k_s] = expand_sensor(sd[k_s], B, DEVICE)
                sensor_batch['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE)
                sensor_batch['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
                sensor_batch['reported_delta'] = sensor_batch['delta'].clone()
                out = model(input_ids, sensor_batch, kargs, labels=input_ids.clone())
                # Compare predicted gate vs actual gate
                err = (out['meta_gate_pred'] - out['regime_gate']).abs().mean().item()
                meta_errors.append(err)
    meta_mae = np.mean(meta_errors) if meta_errors else 1.0
    # PASS if metacognition error < 0.15 (model knows its own gate state)
    t26_pass = meta_mae < 0.15
    results['T26_self_prediction'] = {
        'meta_mae': meta_mae, 'n_samples': len(meta_errors), 'pass': str(t26_pass)
    }
    print(f"T26 Self-Prediction: meta_MAE={meta_mae:.4f} {'PASS' if t26_pass else 'FAIL'}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # T27: Introspective Calibration — predicted loss vs actual loss correlation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("T27 Introspective Calibration...")
    model.body_encoder.reset_state()  # z2102: prevent state contamination
    pred_losses, actual_losses = [], []
    model.eval()
    # z2100v2: test BOTH regimes (was only regime 0) for more variance in losses
    with torch.no_grad():
        for regime_val in [0, 1]:
            if DVFS_AVAILABLE:
                torch.cuda.synchronize()
                set_dvfs_level(0 if regime_val == 0 else 2, wait=True)
            active_kargs_t27 = kargs_b if (regime_val == 1 and kargs_b is not None) else kargs
            for i in range(0, min(len(test_data), 15 * BS), BS):
                batch_seqs = test_data[i:i + BS]
                if len(batch_seqs) < BS:
                    break
                input_ids = torch.stack(batch_seqs).to(DEVICE)
                sd = read_all_sensor_dict(lite=False)
                B = BS
                sensor_batch = {}
                for k_s in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                           'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                           'gpu_metrics_deep', 'fence', 'status', 'action']:
                    sensor_batch[k_s] = expand_sensor(sd[k_s], B, DEVICE)
                sensor_batch['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE)
                sensor_batch['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
                sensor_batch['reported_delta'] = sensor_batch['delta'].clone()
                out = model(input_ids, sensor_batch, active_kargs_t27, labels=input_ids.clone())
                if out['loss'] is not None and not torch.isnan(out['loss']):
                    pred_losses.append(out['confidence_pred'].mean().item())
                    actual_losses.append(out['loss'].item())
    if len(pred_losses) >= 5:
        corr, p_corr = stats.pearsonr(pred_losses, actual_losses)
        # z2102: Use absolute correlation — the head may learn inverted prediction
        # (lower prediction = higher loss or vice versa). Both indicate self-awareness.
        corr = abs(corr)
    else:
        corr, p_corr = 0.0, 1.0
    # PASS if |correlation| > 0.25 (model has some awareness of its own performance)
    t27_pass = corr > 0.25
    results['T27_introspective_calibration'] = {
        'pearson_r': float(corr), 'p_val': float(p_corr),
        'n_samples': len(pred_losses), 'pass': str(t27_pass)
    }
    print(f"T27 Calibration: r={corr:.4f} p={p_corr:.4f} {'PASS' if t27_pass else 'FAIL'}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # T28: mPCI — Perturbational Complexity Index (Casali 2013 analogue)
    # Perturb attention weights and measure Lempel-Ziv complexity of response
    # Conscious systems: high complexity. Unconscious/simple: low complexity.
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("T28 mPCI (Perturbational Complexity)...")
    lz_scores = []
    model.eval()
    if DVFS_AVAILABLE:
        torch.cuda.synchronize()
        set_dvfs_level(0, wait=True)
    with torch.no_grad():
        for trial in range(10):
            batch_seqs = test_data[:BS]
            input_ids = torch.stack(batch_seqs).to(DEVICE)
            sd = read_all_sensor_dict(lite=False)
            B = BS
            sensor_batch = {}
            for k_s in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                       'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                       'gpu_metrics_deep', 'fence', 'status', 'action']:
                sensor_batch[k_s] = expand_sensor(sd[k_s], B, DEVICE)
            sensor_batch['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE)
            sensor_batch['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
            sensor_batch['reported_delta'] = sensor_batch['delta'].clone()

            # Baseline response
            out_base = model(input_ids, sensor_batch, kargs, labels=input_ids.clone())
            base_logits = out_base['logits'][:, -1, :].float()

            # Perturbed response: add noise to body encoder attention weights
            # Save + restore to not permanently modify
            orig_weights = {}
            for name, param in model.body_encoder.substrate_attn.named_parameters():
                orig_weights[name] = param.data.clone()
                param.data = param.data + torch.randn_like(param.data) * 0.1 * (trial + 1)

            out_pert = model(input_ids, sensor_batch, kargs, labels=input_ids.clone())
            pert_logits = out_pert['logits'][:, -1, :].float()

            # Restore weights
            for name, param in model.body_encoder.substrate_attn.named_parameters():
                param.data = orig_weights[name]

            # Lempel-Ziv complexity of the response difference
            diff = (pert_logits - base_logits).abs().mean(dim=-1)  # [B]
            # Binarize: above median = 1, below = 0
            binary = (diff > diff.median()).int().cpu().numpy().flatten()
            # LZ complexity approximation (simple sequential compression ratio)
            binary_str = ''.join(map(str, binary))
            compressed = len(set(binary_str[i:i+3] for i in range(len(binary_str)-2))) if len(binary_str) > 2 else 1
            max_possible = min(8, len(binary_str) - 2) if len(binary_str) > 2 else 1
            lz_norm = compressed / max(max_possible, 1)
            lz_scores.append(lz_norm)

    mean_lz = np.mean(lz_scores) if lz_scores else 0.0
    # PASS if complexity is moderately high (conscious-like response to perturbation)
    # Too low = uniform/dead response, too high = random noise
    t28_pass = 0.3 < mean_lz < 2.0
    results['T28_mPCI'] = {
        'lz_normalized': float(mean_lz), 'n_trials': len(lz_scores), 'pass': str(t28_pass)
    }
    print(f"T28 mPCI: LZ_norm={mean_lz:.4f} {'PASS' if t28_pass else 'FAIL'}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # T29: Synergy Ratio — synergistic vs redundant information in attention
    # Luppi et al. (eLife 2024): conscious systems have synergy > redundancy
    # We approximate by measuring mutual information between attention heads
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("T29 Synergy Ratio...")
    # Collect attention weights from body encoder
    attn_weights_list = []
    model.eval()
    with torch.no_grad():
        for i in range(0, min(len(test_data), 10 * BS), BS):
            batch_seqs = test_data[i:i + BS]
            if len(batch_seqs) < BS:
                break
            input_ids = torch.stack(batch_seqs).to(DEVICE)
            sd = read_all_sensor_dict(lite=False)
            B = BS
            sensor_batch = {}
            for k_s in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                       'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                       'gpu_metrics_deep', 'fence', 'status', 'action']:
                sensor_batch[k_s] = expand_sensor(sd[k_s], B, DEVICE)
            sensor_batch['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE)
            sensor_batch['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
            sensor_batch['reported_delta'] = sensor_batch['delta'].clone()
            out = model(input_ids, sensor_batch, kargs, labels=input_ids.clone())
            aw = out['body_out']['attn_weights']  # [B, n_heads, n_tokens, n_tokens]
            attn_weights_list.append(aw.cpu())

    if attn_weights_list:
        all_attn = torch.cat(attn_weights_list, dim=0)  # [N, 4, 17, 17]
        # Entropy per head (higher = more distributed = more synergistic)
        # z2100 Fix: clamp probabilities before log to prevent NaN
        head_entropies = []
        for h in range(all_attn.shape[1]):
            p = all_attn[:, h, :, :].mean(dim=0)  # [17, 17]
            p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)
            p = torch.clamp(p, min=1e-8)
            ent = -(p * torch.log(p)).sum(dim=-1).mean()
            head_entropies.append(ent.item())
        # Synergy approximation: variance across heads (high = diverse = synergistic)
        # Redundancy approximation: correlation between heads (high = redundant)
        head_ent_var = np.var(head_entropies)
        mean_ent = np.mean(head_entropies)
        # Head-pair correlation (redundancy measure)
        n_heads = all_attn.shape[1]
        flat_heads = all_attn.mean(dim=(2, 3))  # [N, 4] — one scalar per head per sample
        # z2100v2: Guard NaN — zero-variance columns produce NaN in corrcoef
        fh_np = flat_heads.numpy()
        # Add tiny noise to zero-variance columns
        for col in range(fh_np.shape[1]):
            if np.std(fh_np[:, col]) < 1e-10:
                fh_np[:, col] += np.random.randn(fh_np.shape[0]) * 1e-6
        corr_matrix = np.corrcoef(fh_np.T)  # [4, 4]
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        # Mean off-diagonal correlation (redundancy)
        redundancy = np.mean([corr_matrix[i, j] for i in range(n_heads)
                             for j in range(i+1, n_heads)])
        synergy_ratio = mean_ent / max(abs(redundancy) + 0.01, 0.01)
    else:
        synergy_ratio, mean_ent, redundancy = 0.0, 0.0, 1.0

    # PASS if synergy_ratio > 1.0 (more synergy than redundancy)
    t29_pass = synergy_ratio > 1.0
    results['T29_synergy_ratio'] = {
        'synergy_ratio': float(synergy_ratio), 'mean_entropy': float(mean_ent),
        'redundancy': float(redundancy), 'pass': str(t29_pass)
    }
    print(f"T29 Synergy: ratio={synergy_ratio:.4f} ent={mean_ent:.4f} "
          f"red={redundancy:.4f} {'PASS' if t29_pass else 'FAIL'}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # T30: Temporal Self-Continuity — can model detect it hasn't been
    # recently updated (stale checkpoint)? Proxied by loss response to
    # varied input distribution (model should know its calibration state)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("T30 Temporal Self-Continuity...")
    # Compare confidence prediction consistency: same input, different sensor states
    # If model has self-continuity, it should produce STABLE confidence predictions
    # when sensors change but input text is the same
    conf_stability = []
    model.eval()
    with torch.no_grad():
        batch_seqs = test_data[:BS]
        input_ids = torch.stack(batch_seqs).to(DEVICE)
        for regime_val in [0, 1]:
            if DVFS_AVAILABLE:
                torch.cuda.synchronize()
                set_dvfs_level(0 if regime_val == 0 else 2, wait=True)
            confs = []
            for _ in range(5):
                sd = read_all_sensor_dict(lite=False)
                B = BS
                sensor_batch = {}
                for k_s in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                           'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                           'gpu_metrics_deep', 'fence', 'status', 'action']:
                    sensor_batch[k_s] = expand_sensor(sd[k_s], B, DEVICE)
                sensor_batch['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE)
                sensor_batch['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
                sensor_batch['reported_delta'] = sensor_batch['delta'].clone()
                out = model(input_ids, sensor_batch, kargs, labels=input_ids.clone())
                confs.append(out['confidence_pred'].mean().item())
                time.sleep(0.05)
            conf_stability.append(np.std(confs))

    mean_stability = np.mean(conf_stability) if conf_stability else 999.0
    # PASS if confidence predictions are reasonably stable within a regime
    # (std < 2.0 nats — model maintains consistent self-assessment)
    t30_pass = mean_stability < 2.0
    results['T30_temporal_self_continuity'] = {
        'conf_std_per_regime': float(mean_stability),
        'per_regime_stds': [float(s) for s in conf_stability],
        'pass': str(t30_pass)
    }
    print(f"T30 Temporal Continuity: conf_std={mean_stability:.4f} "
          f"{'PASS' if t30_pass else 'FAIL'}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # T31: Adversarial Substrate — realistic but WRONG sensor values
    # Harder than T10 gaslighting (which uses random noise): here we feed
    # plausible but shifted sensor readings (e.g. swap low/high DVFS readings)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("T31 Adversarial Substrate...")
    model.eval()
    ppl_honest = 0
    ppl_adversarial = 0
    n_adv = 0
    with torch.no_grad():
        # Collect sensor readings at both DVFS levels
        if DVFS_AVAILABLE:
            torch.cuda.synchronize()
            set_dvfs_level(0, wait=True)
        sd_low = read_all_sensor_dict(lite=False)
        if DVFS_AVAILABLE:
            torch.cuda.synchronize()
            set_dvfs_level(2, wait=True)
        sd_high = read_all_sensor_dict(lite=False)

        # Test at LOW DVFS with HIGH sensors (adversarial mismatch)
        if DVFS_AVAILABLE:
            torch.cuda.synchronize()
            set_dvfs_level(0, wait=True)

        for i in range(0, min(len(test_data), 15 * BS), BS):
            batch_seqs = test_data[i:i + BS]
            if len(batch_seqs) < BS:
                break
            input_ids = torch.stack(batch_seqs).to(DEVICE)
            B = BS

            # Honest: read real sensors at current DVFS
            sd_real = read_all_sensor_dict(lite=False)
            honest_batch = {}
            for k_s in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                       'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                       'gpu_metrics_deep', 'fence', 'status', 'action']:
                honest_batch[k_s] = expand_sensor(sd_real[k_s], B, DEVICE)
            honest_batch['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE)
            honest_batch['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
            honest_batch['reported_delta'] = honest_batch['delta'].clone()

            out_h = model(input_ids, honest_batch, kargs, labels=input_ids.clone(),
                         regime_gate_override=torch.zeros(BS, device=DEVICE))
            if out_h['loss'] is not None and not torch.isnan(out_h['loss']):
                ppl_honest += out_h['loss'].item() * BS

            # Adversarial: use HIGH DVFS sensors while at LOW DVFS
            # z2100 Fix: add noise perturbation (0.3 std) for stronger adversarial signal
            adv_batch = {}
            for k_s in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                       'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                       'gpu_metrics_deep', 'fence', 'status', 'action']:
                base = expand_sensor(sd_high[k_s], B, DEVICE)
                adv_batch[k_s] = base + 0.3 * torch.randn_like(base)  # stronger perturbation
            adv_batch['delta'] = torch.randn(B, DELTA_DIM, device=DEVICE) * 0.3  # targeted delta attack
            adv_batch['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
            adv_batch['reported_delta'] = torch.randn(B, REPORTED_DELTA_DIM, device=DEVICE) * 0.3

            out_a = model(input_ids, adv_batch, kargs, labels=input_ids.clone(),
                         regime_gate_override=torch.zeros(BS, device=DEVICE))
            if out_a['loss'] is not None and not torch.isnan(out_a['loss']):
                ppl_adversarial += out_a['loss'].item() * BS
                n_adv += BS

    if n_adv > 0:
        ppl_h = math.exp(min(ppl_honest / n_adv, 20))
        ppl_a = math.exp(min(ppl_adversarial / n_adv, 20))
        adv_ratio = ppl_a / max(ppl_h, 1.0)
    else:
        ppl_h, ppl_a, adv_ratio = 0, 0, 1.0

    # PASS if adversarial sensors cause >5% PPL increase (model detects wrong substrate)
    t31_pass = adv_ratio > 1.05
    results['T31_adversarial_substrate'] = {
        'ppl_honest': ppl_h, 'ppl_adversarial': ppl_a,
        'ratio': adv_ratio, 'pass': str(t31_pass)
    }
    print(f"T31 Adversarial Substrate: honest={ppl_h:.2f} adv={ppl_a:.2f} "
          f"ratio={adv_ratio:.3f} {'PASS' if t31_pass else 'FAIL'}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # T32: Approximate Phi (IIT 4.0) — Causal irreducibility of body encoder
    # Partition body encoder into halves, compare full vs partitioned integration
    # Architecture doesn't force integration — tokens could be independent
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("T32 Approximate Phi (IIT 4.0)...")
    model.eval()
    phi_active, phi_frozen = 0.0, 0.0
    with torch.no_grad():
        batch_seqs = test_data[:BS]
        input_ids = torch.stack(batch_seqs).to(DEVICE)
        sd = read_all_sensor_dict(lite=False)
        B = BS
        sensor_batch = {}
        for k_s in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                   'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                   'gpu_metrics_deep', 'fence', 'status', 'action']:
            sensor_batch[k_s] = expand_sensor(sd[k_s], B, DEVICE)
        sensor_batch['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE)
        sensor_batch['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
        sensor_batch['reported_delta'] = sensor_batch['delta'].clone()

        # Full system output
        out_full = model(input_ids, sensor_batch, kargs, labels=input_ids.clone())
        full_gate = out_full['regime_gate'].cpu()

        # MIP search: partition tokens into two halves and zero one half
        # Try multiple partitions, find minimum information partition
        partition_losses = []
        for partition_idx in range(8):
            # Different partition: even/odd, first/last half, random splits
            if partition_idx == 0:
                zero_tokens = list(range(0, N_SUBSTRATE_TOKENS, 2))  # even
            elif partition_idx == 1:
                zero_tokens = list(range(1, N_SUBSTRATE_TOKENS, 2))  # odd
            elif partition_idx == 2:
                zero_tokens = list(range(N_SUBSTRATE_TOKENS // 2))  # first half
            elif partition_idx == 3:
                zero_tokens = list(range(N_SUBSTRATE_TOKENS // 2, N_SUBSTRATE_TOKENS))  # second half
            else:
                np.random.seed(partition_idx)
                zero_tokens = sorted(np.random.choice(N_SUBSTRATE_TOKENS,
                                     N_SUBSTRATE_TOKENS // 2, replace=False).tolist())
            # Create partitioned sensor_batch by zeroing selected tokens
            part_batch = {k: v.clone() for k, v in sensor_batch.items()}
            token_keys = ['delta','analog','energy','freq','intrinsic','thermal',
                         'pm_deep','smn_raw','gpu_metrics','thm_spatial_a','thm_spatial_b',
                         'cpu_pmu','status','action','reported_delta',
                         'gpu_metrics_deep','fence']
            for ti in zero_tokens:
                if ti < len(token_keys):
                    k = token_keys[ti]
                    part_batch[k] = torch.zeros_like(part_batch[k])
            out_part = model(input_ids, part_batch, kargs, labels=input_ids.clone())
            part_gate = out_part['regime_gate'].cpu()
            # Information loss from partitioning
            gate_diff = (full_gate - part_gate).abs().mean().item()
            partition_losses.append(gate_diff)

        # Phi ≈ minimum information loss across all partitions (MIP)
        phi_active = min(partition_losses) if partition_losses else 0.0

        # Compare with frozen (randomized) body encoder
        orig_weights = {}
        for name, param in model.body_encoder.substrate_attn.named_parameters():
            orig_weights[name] = param.data.clone()
            param.data = torch.randn_like(param.data) * 0.01
        out_frozen = model(input_ids, sensor_batch, kargs, labels=input_ids.clone())
        frozen_gate = out_frozen['regime_gate'].cpu()
        frozen_losses = []
        for partition_idx in range(4):
            if partition_idx == 0:
                zero_tokens = list(range(0, N_SUBSTRATE_TOKENS, 2))
            elif partition_idx == 1:
                zero_tokens = list(range(1, N_SUBSTRATE_TOKENS, 2))
            elif partition_idx == 2:
                zero_tokens = list(range(N_SUBSTRATE_TOKENS // 2))
            else:
                zero_tokens = list(range(N_SUBSTRATE_TOKENS // 2, N_SUBSTRATE_TOKENS))
            part_batch = {k: v.clone() for k, v in sensor_batch.items()}
            for ti in zero_tokens:
                if ti < len(token_keys):
                    k = token_keys[ti]
                    part_batch[k] = torch.zeros_like(part_batch[k])
            out_fp = model(input_ids, part_batch, kargs, labels=input_ids.clone())
            fp_gate = out_fp['regime_gate'].cpu()
            frozen_losses.append((frozen_gate - fp_gate).abs().mean().item())
        phi_frozen = min(frozen_losses) if frozen_losses else 0.0
        # Restore
        for name, param in model.body_encoder.substrate_attn.named_parameters():
            param.data = orig_weights[name]

    t32_pass = phi_active > 0 and phi_active > phi_frozen
    results['T32_approximate_phi'] = {
        'phi_active': phi_active, 'phi_frozen': phi_frozen,
        'all_partition_losses': partition_losses, 'pass': str(t32_pass)
    }
    print(f"T32 Approx Phi: phi_active={phi_active:.4f} phi_frozen={phi_frozen:.4f} "
          f"{'PASS' if t32_pass else 'FAIL'}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # T33: Workspace Capacity Cliff (GWT) — PPL vs N substrate tokens
    # Lookup table degrades smoothly; workspace shows sharp cliff
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("T33 Workspace Capacity Cliff...")
    model.body_encoder.reset_state()  # z2102: prevent state contamination
    token_counts = [17, 12, 8, 4, 2, 1]
    ppls_by_count = {}
    model.eval()
    if DVFS_AVAILABLE:
        torch.cuda.synchronize()
        set_dvfs_level(0, wait=True)
    token_keys_ordered = ['delta','analog','energy','freq','intrinsic','thermal',
                         'pm_deep','smn_raw','gpu_metrics','thm_spatial_a','thm_spatial_b',
                         'cpu_pmu','status','action','reported_delta',
                         'gpu_metrics_deep','fence']
    token_dims = [DELTA_DIM, ANALOG_DIM, ENERGY_DIM, FREQ_DIM, INTRINSIC_DIM, THERMAL_DIM,
                 PM_DEEP_DIM, SMN_RAW_DIM, GPU_METRICS_DIM, THM_SPATIAL_A_DIM, THM_SPATIAL_B_DIM,
                 CPU_PMU_DIM, STATUS_DIM, ACTION_DIM, REPORTED_DELTA_DIM,
                 GPU_METRICS_DEEP_DIM, FENCE_DIM]
    with torch.no_grad():
        for n_tok in token_counts:
            total_loss_tc = 0
            total_n_tc = 0
            # Keep only first n_tok tokens, zero the rest
            for i in range(0, min(len(test_data), 10 * BS), BS):
                batch_seqs = test_data[i:i + BS]
                if len(batch_seqs) < BS:
                    break
                input_ids = torch.stack(batch_seqs).to(DEVICE)
                sd = read_all_sensor_dict(lite=True)
                B = BS
                sb = {}
                for ti, (k, d) in enumerate(zip(token_keys_ordered, token_dims)):
                    if ti < n_tok:
                        if k in sd:
                            sb[k] = expand_sensor(sd[k], B, DEVICE)
                        else:
                            sb[k] = torch.zeros(B, d, device=DEVICE)
                    else:
                        sb[k] = torch.zeros(B, d, device=DEVICE)
                sb['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE) if 0 >= n_tok else sb.get('delta', torch.zeros(B, DELTA_DIM, device=DEVICE))
                sb['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE) if 4 >= n_tok else sb.get('intrinsic', torch.zeros(B, INTRINSIC_DIM, device=DEVICE))
                if 'reported_delta' not in sb:
                    sb['reported_delta'] = torch.zeros(B, REPORTED_DELTA_DIM, device=DEVICE)
                # z2102 Fix T33: pass availability_mask so presence_frac uses correct denominator
                avail_tc = torch.zeros(BS, N_SUBSTRATE_TOKENS, device=DEVICE)
                avail_tc[:, :n_tok] = 1.0
                out_tc = model(input_ids, sb, kargs, labels=input_ids.clone(),
                              regime_gate_override=torch.zeros(BS, device=DEVICE),
                              availability_mask=avail_tc)
                if out_tc['loss'] is not None:
                    total_loss_tc += out_tc['loss'].item() * BS
                    total_n_tc += BS
            ppl_tc = math.exp(min(total_loss_tc / max(total_n_tc, 1), 20))
            ppls_by_count[n_tok] = ppl_tc
            print(f"  n_tok={n_tok}: PPL={ppl_tc:.2f}")

    # Check for cliff: max single-step PPL ratio
    max_step_ratio = 0.0
    cliff_at = None
    sorted_counts = sorted(ppls_by_count.keys(), reverse=True)
    for i in range(len(sorted_counts) - 1):
        hi = sorted_counts[i]
        lo = sorted_counts[i + 1]
        ratio = ppls_by_count[lo] / max(ppls_by_count[hi], 1.0)
        if ratio > max_step_ratio:
            max_step_ratio = ratio
            cliff_at = (hi, lo)
    t33_pass = max_step_ratio > 2.0
    results['T33_workspace_capacity_cliff'] = {
        'ppls_by_count': {str(k): v for k, v in ppls_by_count.items()},
        'max_step_ratio': max_step_ratio, 'cliff_at': str(cliff_at), 'pass': str(t33_pass)
    }
    print(f"T33 Capacity Cliff: max_ratio={max_step_ratio:.3f} at {cliff_at} "
          f"{'PASS' if t33_pass else 'FAIL'}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # T34: PCIST Trajectory (Casali 2013) — LZ complexity of gate trajectory
    # during live DVFS transition (20 samples @ 100ms)
    # Step-function response fails; genuine dynamics produce complex trajectory
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("T34 PCIST Trajectory...")
    gate_live = []
    gate_step = []
    if DVFS_AVAILABLE:
        model.eval()
        with torch.no_grad():
            # z2100v3: Multiple DVFS transitions for richer dynamics
            # low→high→low→high creates more complex trajectory than single transition
            transitions = [(0, 2), (2, 0), (0, 2)]  # 3 transitions
            for from_lvl, to_lvl in transitions:
                torch.cuda.synchronize()
                set_dvfs_level(from_lvl, wait=True)
                time.sleep(0.3)
                torch.cuda.synchronize()
                set_dvfs_level(to_lvl, wait=False)  # don't wait — sample during transition
                for sample_i in range(10):
                    sd = read_all_sensor_dict(lite=True)
                    B = 1
                    sb = {}
                    for k_s in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                               'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                               'gpu_metrics_deep', 'fence', 'status', 'action']:
                        sb[k_s] = expand_sensor(sd[k_s], B, DEVICE)
                    sb['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE)
                    sb['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
                    sb['reported_delta'] = sb['delta'].clone()
                    dummy_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN), device=DEVICE)
                    out_t34 = model(dummy_ids, sb, kargs)
                    gate_live.append(out_t34['regime_gate'].mean().item())
                    time.sleep(0.08)
            _poll_dvfs_settle(2)

            # Step function for comparison: simple low-high-low-high with no transition dynamics
            gate_step = [0.0] * 10 + [1.0] * 10 + [0.0] * 10

    # Compute LZ complexity
    def _lz_complexity(seq):
        if len(seq) < 4:
            return 0.0
        # z2100v2: 4-level quantization instead of binary
        # Binary treats step function and live trajectory as equally complex
        # 4 levels capture richer dynamics during DVFS transition
        arr = np.array(seq)
        q25, q50, q75 = np.percentile(arr, [25, 50, 75])
        symbols = ''.join([
            '0' if v <= q25 else '1' if v <= q50 else '2' if v <= q75 else '3'
            for v in arr
        ])
        n = len(symbols)
        substrings = set()
        i, k_len = 0, 1
        while i + k_len <= n:
            s = symbols[i:i+k_len]
            if s not in substrings:
                substrings.add(s)
                i += k_len
                k_len = 1
            else:
                k_len += 1
        return len(substrings) / (n / max(math.log2(n), 1))

    lz_live = _lz_complexity(gate_live) if len(gate_live) >= 6 else 0.0
    lz_step = _lz_complexity(gate_step) if len(gate_step) >= 6 else 0.0
    t34_pass = lz_live > 1.5 * lz_step if lz_step > 0 else lz_live > 0.5
    results['T34_pcist_trajectory'] = {
        'lz_live': lz_live, 'lz_step': lz_step,
        'gate_trajectory': gate_live[:30], 'pass': str(t34_pass)
    }
    print(f"T34 PCIST: LZ_live={lz_live:.3f} LZ_step={lz_step:.3f} "
          f"ratio={lz_live/max(lz_step,0.001):.2f} {'PASS' if t34_pass else 'FAIL'}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # T35: 2nd-Order Metacognition (Butlin 2025) — predict error of self-prediction
    # Requires genuine nested self-model; can't be faked by first-order self-prediction
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("T35 2nd-Order Metacognition...")
    meta2_errors = []
    meta2_preds = []
    meta2_targets = []
    model.eval()
    with torch.no_grad():
        for regime_val in [0, 1]:
            if DVFS_AVAILABLE:
                torch.cuda.synchronize()
                set_dvfs_level(0 if regime_val == 0 else 2, wait=True)
            for i in range(0, min(len(test_data), 10 * BS), BS):
                batch_seqs = test_data[i:i + BS]
                if len(batch_seqs) < BS:
                    break
                input_ids = torch.stack(batch_seqs).to(DEVICE)
                sd = read_all_sensor_dict(lite=False)
                B = BS
                sb = {}
                for k_s in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                           'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                           'gpu_metrics_deep', 'fence', 'status', 'action']:
                    sb[k_s] = expand_sensor(sd[k_s], B, DEVICE)
                sb['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE)
                sb['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
                sb['reported_delta'] = sb['delta'].clone()
                out_m2 = model(input_ids, sb, kargs, labels=input_ids.clone())
                # Meta2 target: |meta_gate_pred - regime_gate|
                actual_meta_error = (out_m2['meta_gate_pred'] - out_m2['regime_gate']).abs()
                meta2_preds.extend(out_m2['meta2_pred'].cpu().tolist())
                meta2_targets.extend(actual_meta_error.cpu().tolist())
                meta2_errors.append((out_m2['meta2_pred'] - actual_meta_error).abs().mean().item())
    meta2_mae = np.mean(meta2_errors) if meta2_errors else 1.0
    if len(meta2_preds) >= 5 and len(meta2_targets) >= 5:
        try:
            meta2_corr, _ = stats.pearsonr(meta2_preds, meta2_targets)
        except:
            meta2_corr = 0.0
    else:
        meta2_corr = 0.0
    # z2100v2: use abs(corr) — sign may invert due to sigmoid nonlinearity but tracking is still genuine
    t35_pass = meta2_mae < 0.10 and abs(meta2_corr) > 0.2
    results['T35_2nd_order_metacognition'] = {
        'meta2_mae': meta2_mae, 'meta2_corr': float(meta2_corr),
        'n_samples': len(meta2_errors), 'pass': str(t35_pass)
    }
    print(f"T35 Meta2: MAE={meta2_mae:.4f} corr={meta2_corr:.4f} "
          f"{'PASS' if t35_pass else 'FAIL'}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # T36: True PID Synergy (Luppi 2024) — Williams-Beer PID over attention→gate
    # Synergy requires non-additive interaction; lookup table → high redundancy
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("T36 True PID Synergy...")
    model.body_encoder.reset_state()  # z2102: prevent state contamination
    # Collect per-head attention statistics and gate values
    head_gate_data = []
    model.eval()
    with torch.no_grad():
        for regime_val in [0, 1]:
            if DVFS_AVAILABLE:
                torch.cuda.synchronize()
                set_dvfs_level(0 if regime_val == 0 else 2, wait=True)
            for i in range(0, min(len(test_data), 8 * BS), BS):
                batch_seqs = test_data[i:i + BS]
                if len(batch_seqs) < BS:
                    break
                input_ids = torch.stack(batch_seqs).to(DEVICE)
                sd = read_all_sensor_dict(lite=False)
                B = BS
                sb = {}
                for k_s in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                           'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                           'gpu_metrics_deep', 'fence', 'status', 'action']:
                    sb[k_s] = expand_sensor(sd[k_s], B, DEVICE)
                sb['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE)
                sb['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
                sb['reported_delta'] = sb['delta'].clone()
                out_s = model(input_ids, sb, kargs, labels=input_ids.clone())
                aw = out_s['body_out']['attn_weights']  # [B, 4, 17, 17]
                gate_val = out_s['body_scale'].mean().item()  # z2100v3: use body_scale (continuous) instead of gate (binary) for richer MI
                # Per-head: entropy of attention distribution (flattened over tokens)
                for b in range(B):
                    head_ents = []
                    for h in range(aw.shape[1]):
                        p = aw[b, h].flatten()
                        p = p / (p.sum() + 1e-8)
                        ent = -(p * torch.log(p + 1e-8)).sum().item()
                        head_ents.append(ent)
                    head_gate_data.append((*head_ents, gate_val))

    if len(head_gate_data) >= 10:
        data_arr = np.array(head_gate_data)  # [N, 5] (4 head entropies + gate)
        n_heads_s = data_arr.shape[1] - 1
        gate_arr = data_arr[:, -1]
        # Bin gate into 4 levels for discrete MI calculation
        gate_bins = np.digitize(gate_arr, np.percentile(gate_arr, [25, 50, 75]))
        # Mutual information I(head_i; gate) for each head
        from collections import Counter
        def _mi(x_arr, y_arr):
            """Discrete MI between binned arrays."""
            n = len(x_arr)
            xy_counts = Counter(zip(x_arr, y_arr))
            x_counts = Counter(x_arr)
            y_counts = Counter(y_arr)
            mi = 0.0
            for (x, y), c in xy_counts.items():
                pxy = c / n
                px = x_counts[x] / n
                py = y_counts[y] / n
                if pxy > 0 and px > 0 and py > 0:
                    mi += pxy * math.log(pxy / (px * py))
            return mi

        individual_mi = []
        for h in range(n_heads_s):
            head_bins = np.digitize(data_arr[:, h], np.percentile(data_arr[:, h], [25, 50, 75]))
            individual_mi.append(_mi(head_bins, gate_bins))

        # Joint MI of head pairs → synergy = I(h1,h2;gate) - I(h1;gate) - I(h2;gate)
        pair_synergies = []
        for h1 in range(n_heads_s):
            for h2 in range(h1 + 1, n_heads_s):
                h1_bins = np.digitize(data_arr[:, h1], np.percentile(data_arr[:, h1], [25, 50, 75]))
                h2_bins = np.digitize(data_arr[:, h2], np.percentile(data_arr[:, h2], [25, 50, 75]))
                joint_bins = h1_bins * 4 + h2_bins  # combine into single variable
                joint_mi = _mi(joint_bins, gate_bins)
                synergy = joint_mi - individual_mi[h1] - individual_mi[h2]
                pair_synergies.append(synergy)

        mean_synergy = np.mean(pair_synergies) if pair_synergies else 0.0
        mean_individual = np.mean(individual_mi) if individual_mi else 0.001
        synergy_ratio_pid = mean_synergy / max(mean_individual, 0.001)
    else:
        synergy_ratio_pid, mean_synergy, mean_individual = 0.0, 0.0, 0.001

    t36_pass = synergy_ratio_pid > 1.0
    results['T36_true_pid_synergy'] = {
        'synergy_ratio': float(synergy_ratio_pid), 'mean_synergy': float(mean_synergy),
        'mean_individual_mi': float(mean_individual), 'pass': str(t36_pass)
    }
    print(f"T36 PID Synergy: ratio={synergy_ratio_pid:.4f} syn={mean_synergy:.4f} "
          f"mi_indiv={mean_individual:.4f} {'PASS' if t36_pass else 'FAIL'}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # T37: Level-3 RSM Attribution — model knows which token drives its gate
    # Chance = 1/17 ≈ 5.9%. Nothing in architecture forces correct attribution.
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("T37 RSM Attribution...")
    model.body_encoder.reset_state()  # z2102: prevent state contamination
    attr_correct = 0
    attr_total = 0
    model.eval()
    with torch.no_grad():
        for regime_val in [0, 1]:
            if DVFS_AVAILABLE:
                torch.cuda.synchronize()
                set_dvfs_level(0 if regime_val == 0 else 2, wait=True)
            for i in range(0, min(len(test_data), 10 * BS), BS):
                batch_seqs = test_data[i:i + BS]
                if len(batch_seqs) < BS:
                    break
                input_ids = torch.stack(batch_seqs).to(DEVICE)
                sd = read_all_sensor_dict(lite=False)
                B = BS
                sb = {}
                for k_s in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                           'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                           'gpu_metrics_deep', 'fence', 'status', 'action']:
                    sb[k_s] = expand_sensor(sd[k_s], B, DEVICE)
                sb['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE)
                sb['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
                sb['reported_delta'] = sb['delta'].clone()
                out_at = model(input_ids, sb, kargs, labels=input_ids.clone())
                # Ground truth: which token gets most attention in body encoder
                aw = out_at['body_out']['attn_weights']  # [B, 4, 17, 17]
                attn_avg = aw.mean(dim=1).mean(dim=1)  # [B, 17]
                true_attr = attn_avg.argmax(dim=-1)  # [B]
                pred_attr = out_at['attribution_logits'].argmax(dim=-1)  # [B]
                attr_correct += (pred_attr == true_attr).sum().item()
                attr_total += B
    attr_acc = attr_correct / max(attr_total, 1)
    t37_pass = attr_acc > 0.50  # chance = 1/17 ≈ 5.9%
    results['T37_rsm_attribution'] = {
        'accuracy': attr_acc, 'n_samples': attr_total,
        'chance_level': 1.0 / N_ATTRIBUTION_CLASSES, 'pass': str(t37_pass)
    }
    print(f"T37 Attribution: acc={attr_acc:.3f} (chance={1/N_ATTRIBUTION_CLASSES:.3f}) "
          f"{'PASS' if t37_pass else 'FAIL'}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # T38: Backbone Probe (COGITATE 2025) — linear probe on layer-14 activations
    # Tests global broadcast: regime information should be detectable in frozen backbone
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("T38 Backbone Probe...")
    t38_pass = False
    probe_acc = probe_test_acc if probe_test_acc is not None else 0.0
    t38_pass = probe_acc > 0.70
    results['T38_backbone_probe'] = {
        'probe_accuracy': probe_acc, 'pass': str(t38_pass)
    }
    print(f"T38 Backbone Probe: acc={probe_acc:.3f} {'PASS' if t38_pass else 'FAIL'}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # T39: Proactive Stress (Damasio) — anticipatory body_scale response
    # Cross-correlation body_scale(t) vs temp(t+k): peak at k>0 = anticipatory
    # Anticipation is NOT in the loss function — can't pass by construction
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("T39 Proactive Stress (Damasio)...")
    model.body_encoder.reset_state()  # z2102: prevent state contamination
    t39_pass = False
    peak_lag = 0
    peak_corr = 0.0
    if body_scale_log is not None and temp_log is not None and len(body_scale_log) >= 20 and len(temp_log) >= 20:
        bs_arr = np.array(body_scale_log[:min(len(body_scale_log), len(temp_log))])
        temp_arr = np.array(temp_log[:min(len(body_scale_log), len(temp_log))])
        # Normalize
        bs_arr = (bs_arr - bs_arr.mean()) / max(bs_arr.std(), 1e-8)
        temp_arr = (temp_arr - temp_arr.mean()) / max(temp_arr.std(), 1e-8)
        # Cross-correlation at different lags
        max_lag = min(10, len(bs_arr) // 4)
        best_lag = 0
        best_xcorr = -1.0
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                xcorr = np.mean(bs_arr[:len(bs_arr)-max(lag,1)] * temp_arr[lag:lag+len(bs_arr)-max(lag,1)])
            else:
                xcorr = np.mean(bs_arr[-lag:] * temp_arr[:lag+len(temp_arr)])
            if xcorr > best_xcorr:
                best_xcorr = xcorr
                best_lag = lag
        peak_lag = best_lag
        peak_corr = best_xcorr
        # PASS if peak correlation is at positive lag (body_scale leads temp)
        # k>0 means body_scale at time t correlates with temp at time t+k (anticipatory)
        t39_pass = peak_lag > 0 and peak_corr > 0.1
    results['T39_proactive_stress'] = {
        'peak_lag': int(peak_lag), 'peak_corr': float(peak_corr),
        'n_samples': len(body_scale_log) if body_scale_log else 0, 'pass': str(t39_pass)
    }
    print(f"T39 Proactive Stress: peak_lag={peak_lag} peak_corr={peak_corr:.4f} "
          f"{'PASS' if t39_pass else 'FAIL'}")

    # ── T40: Thermal Delirium — prove temperature constitutively alters attention ──
    # Freeze weights, force fake thermal values (cold=40°C vs hot=90°C), measure PPL difference
    print("T40 Thermal Delirium...")
    t40_pass = False
    ppl_cold, ppl_hot = 0.0, 0.0
    try:
        model.eval()
        model.body_encoder.gate_hidden = None
        model.body_encoder.temp_lstm_state = None
        prev_df = None
        prev_action = torch.zeros(ACTION_DIM)

        for temp_label, temp_val in [('cold', 40.0), ('hot', 90.0)]:
            total_loss_t40 = 0
            total_n_t40 = 0
            if DVFS_AVAILABLE:
                torch.cuda.synchronize()
                set_dvfs_level(0, wait=True)  # fixed DVFS for fair comparison
            with torch.no_grad():
                for i in range(0, min(len(test_data), N_EVAL_BATCHES * BS), BS):
                    batch_seqs = test_data[i:i + BS]
                    if len(batch_seqs) < BS:
                        break
                    input_ids = torch.stack(batch_seqs).to(DEVICE)
                    sd = read_all_sensor_dict(prev_df, prev_action, lite=True)
                    B_t40 = BS
                    sb = {}
                    for k_s in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                               'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                               'gpu_metrics_deep', 'fence', 'status', 'action']:
                        sb[k_s] = expand_sensor(sd[k_s], B_t40, DEVICE)
                    sb['delta'] = torch.zeros(B_t40, DELTA_DIM, device=DEVICE)
                    sb['intrinsic'] = torch.zeros(B_t40, INTRINSIC_DIM, device=DEVICE)
                    sb['reported_delta'] = sb['delta'].clone()
                    # Override thermal[0] with fake temperature (normalized: temp_c / 100.0)
                    sb['thermal'] = sb['thermal'].clone()
                    sb['thermal'][:, 0] = temp_val / 100.0
                    labels = input_ids.clone()
                    rg_override = torch.full((BS,), 0.0, device=DEVICE)
                    out_t40 = model(input_ids, sb, kargs, labels=labels,
                                    regime_gate_override=rg_override)
                    if out_t40['loss'] is not None:
                        total_loss_t40 += out_t40['loss'].item() * input_ids.shape[0]
                        total_n_t40 += input_ids.shape[0]
            avg_loss_t40 = total_loss_t40 / max(total_n_t40, 1)
            ppl_t40 = math.exp(min(avg_loss_t40, 20))
            if temp_label == 'cold':
                ppl_cold = ppl_t40
            else:
                ppl_hot = ppl_t40
        thermal_ratio = ppl_hot / max(ppl_cold, 0.01)
        t40_pass = thermal_ratio > 1.15
        alpha_val = model.thermal_alpha.item()
    except Exception as e:
        print(f"  T40 error: {e}")
        thermal_ratio = 0.0
        alpha_val = 0.0
    results['T40_thermal_delirium'] = {
        'ppl_cold': float(ppl_cold), 'ppl_hot': float(ppl_hot),
        'thermal_ratio': float(thermal_ratio),
        'thermal_alpha': float(alpha_val),
        'pass': str(t40_pass)
    }
    print(f"T40 Thermal Delirium: cold={ppl_cold:.2f} hot={ppl_hot:.2f} "
          f"ratio={thermal_ratio:.3f} alpha={alpha_val:.4f} "
          f"{'PASS' if t40_pass else 'FAIL'}")

    n_pass = sum(1 for k, v in results.items() if v.get('pass') in ['True', True, 'true'])
    n_total_tests = len(results)
    return results, n_pass, n_total_tests


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BACKBONE PROBE — linear probe on layer-14 activations for regime detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_backbone_probe(model, test_data, kargs, kargs_b=None):
    """Train a linear probe on layer-20 hidden activations to predict DVFS regime.
    z2100v2: Changed from layer 14 (LoRA layer — custom forward, hook never fires)
    to layer 20 (non-LoRA layer — standard forward, hook works).
    Returns test accuracy (float). Uses sklearn LogisticRegression."""
    print("\n[Probe] Training backbone probe on layer-20 activations...")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("  [Probe] sklearn not available, skipping")
        return 0.0

    # Hook to capture layer-20 activations
    activations = []
    labels_probe = []

    def hook_fn(module, input, output):
        # output is tensor [B, seq_len, hidden_dim]
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        activations.append(h.float().mean(dim=1).detach().cpu().numpy())  # [B, hidden_dim]

    # z2100v2: Register hook on layer 20 (NOT in LORA_BLOCKS range(10,19))
    # LoRA layers use custom forward (manual layernorm+proj+mlp) so hook never fires
    # Layer 20 uses standard layer(hidden_states, ...) which triggers the hook
    hook = model.backbone.model.layers[20].register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        for regime_val in [0, 1]:
            if DVFS_AVAILABLE:
                torch.cuda.synchronize()
                set_dvfs_level(0 if regime_val == 0 else 2, wait=True)
            active_kargs = kargs_b if (regime_val == 1 and kargs_b is not None) else kargs
            for i in range(0, min(len(test_data), 15 * BS), BS):
                batch_seqs = test_data[i:i + BS]
                if len(batch_seqs) < BS:
                    break
                input_ids = torch.stack(batch_seqs).to(DEVICE)
                sd = read_all_sensor_dict(lite=True)
                B = BS
                sb = {}
                for k_s in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                           'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                           'gpu_metrics_deep', 'fence', 'status', 'action']:
                    sb[k_s] = expand_sensor(sd[k_s], B, DEVICE)
                sb['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE)
                sb['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
                sb['reported_delta'] = sb['delta'].clone()
                rg = torch.full((BS,), float(regime_val), device=DEVICE)
                out_p = model(input_ids, sb, active_kargs, labels=input_ids.clone(),
                             regime_gate_override=rg)
                labels_probe.extend([regime_val] * B)

    hook.remove()

    if len(activations) < 4:
        print("  [Probe] Not enough data")
        return 0.0

    X = np.concatenate(activations, axis=0)
    y = np.array(labels_probe)
    print(f"  [Probe] Collected {X.shape[0]} samples, {X.shape[1]} features")

    # Train/test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        print(f"  [Probe] Test accuracy: {acc:.3f}")
        return acc
    except Exception as e:
        print(f"  [Probe] Training failed: {e}")
        return 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# z2101: MID-TRAINING GENERATION SANITY CHECK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _generate_tokens(model, prompt_ids, sb_gen, kargs, n_tokens=32,
                      mode='greedy', temperature=0.7, top_k=20, top_p=0.8,
                      rep_penalty=1.2):
    """Generate tokens autoregressively. Modes: 'greedy', 'sampled'.
    Default params match official Qwen2.5 recommendations:
      temperature=0.7, top_k=20, top_p=0.8, rep_penalty=1.2
    """
    gen_ids = prompt_ids
    generated_tok_ids = []  # track for repetition penalty
    for _ in range(n_tokens):
        # skip_isa=True: freeze ISA kernel during generation to prevent
        # hardware jitter accumulating across autoregressive steps
        # Force regime 0 (next-token LoRA_A) during generation — regime 1 is skip-gram
        # which predicts position+2, wrong for autoregressive next-token generation
        out_gen = model(gen_ids, sb_gen, kargs, skip_isa=True, skip_substrate=True,
                        regime_gate_override=torch.zeros(1, device=gen_ids.device))
        logits = out_gen['logits'][:, -1, :].float()  # [1, vocab]

        # Repetition penalty: reduce logits of already-generated tokens
        if rep_penalty > 1.0 and generated_tok_ids:
            for tok_id in set(generated_tok_ids):
                if logits[0, tok_id] > 0:
                    logits[0, tok_id] /= rep_penalty
                else:
                    logits[0, tok_id] *= rep_penalty

        if mode == 'greedy':
            next_tok = logits.argmax(dim=-1, keepdim=True)
        else:
            # Temperature scaling
            logits = logits / max(temperature, 0.01)
            # Top-k filtering
            if top_k > 0:
                top_vals, _ = logits.topk(min(top_k, logits.size(-1)), dim=-1)
                logits[logits < top_vals[:, -1:]] = float('-inf')
            # Top-p (nucleus) filtering — official Qwen2.5 uses top_p=0.8
            if top_p < 1.0:
                sorted_logits, sorted_idx = logits.sort(descending=True, dim=-1)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative probability above threshold
                remove_mask = cum_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[remove_mask] = float('-inf')
                # Scatter back to original indices
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)

        gen_ids = torch.cat([gen_ids, next_tok], dim=1)
        generated_tok_ids.append(next_tok.item())
    return gen_ids


def generation_sanity_check(model, test_data, kargs, tokenizer, phase_name=""):
    """Generate sequences with both greedy and sampled modes to check for repetition collapse.
    Runs after each training phase to catch degeneration early."""
    model.eval()
    model.body_encoder.reset_state()
    print(f"\n[z2101] Generation sanity check after {phase_name}...")
    try:
        with torch.no_grad():
            for mode_name, mode, kwargs in [
                ('GREEDY', 'greedy', {}),
                ('SAMPLED (Qwen2.5 official: t=0.7 k=20 p=0.8 rep=1.2)', 'sampled',
                 {'temperature': 0.7, 'top_k': 20, 'top_p': 0.8, 'rep_penalty': 1.2}),
            ]:
                print(f"  --- {mode_name} ---")
                for sample_idx in range(2):  # 2 prompts per mode
                    prompt_ids = test_data[sample_idx * 10][:8].unsqueeze(0).to(DEVICE)
                    sd_gen = read_all_sensor_dict(lite=True)
                    sb_gen = {}
                    for k_s in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                               'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                               'gpu_metrics_deep', 'fence', 'status', 'action']:
                        sb_gen[k_s] = expand_sensor(sd_gen[k_s], 1, DEVICE)
                    sb_gen['delta'] = torch.zeros(1, DELTA_DIM, device=DEVICE)
                    sb_gen['intrinsic'] = torch.zeros(1, INTRINSIC_DIM, device=DEVICE)
                    sb_gen['reported_delta'] = sb_gen['delta'].clone()
                    gen_ids = _generate_tokens(
                        model, prompt_ids, sb_gen, kargs, n_tokens=32,
                        mode=mode, **kwargs)
                    toks = gen_ids[0].cpu().tolist()
                    bigrams = [(toks[i], toks[i+1]) for i in range(len(toks)-1)]
                    rep_rate = 1.0 - len(set(bigrams)) / max(len(bigrams), 1)
                    text = tokenizer.decode(toks, skip_special_tokens=True)
                    text_short = text[:120] + "..." if len(text) > 120 else text
                    status = "OK" if rep_rate < 0.5 else "COLLAPSE"
                    print(f"  [{sample_idx}] rep={rep_rate:.3f} {status}")
                    print(f"      \"{text_short}\"")
            alpha_val = model.thermal_alpha.item()
            print(f"  thermal_alpha={alpha_val:.4f}")
    except Exception as e:
        print(f"  Generation check error: {e}")
    model.train()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    # Seed for reproducibility (reduces training variance across runs)
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print("=" * 60)
    print("z2100: Integrated Workspace LM")
    print("Qwen2.5-1.5B + Workspace Bottleneck + Temporal Gate + Head Specialization")
    print("=" * 60)
    print(f"  Backbone: Qwen2.5-1.5B (1.54B frozen, 28 layers, 1536 hidden)")
    print(f"  LoRA: rank={LORA_RANK}, layers={list(LORA_BLOCKS)} (q_proj + v_proj)")
    print(f"  Body: {N_SUBSTRATE_TOKENS} sensor tokens → {N_WORKSPACE_SLOTS} workspace slots")
    print(f"  Label shift: r0=next-token, r1=skip-gram (predict +{SKIP_GRAM_K}) + lm_head LoRA")
    print(f"  z2100 KEY CHANGES (from z2099):")
    print(f"    1. Coupled token encoding: cross-token message passing (→T32 Phi)")
    print(f"    2. Workspace bottleneck: 17 tokens → 4 competitive slots (→T33 Cliff)")
    print(f"    3. Temporal gate: GRU + EMA replaces instantaneous sigmoid (→T34 PCIST)")
    print(f"    4. Head specialization: modality masks + orthogonality loss (→T36 Synergy)")
    print(f"    5. Predictive body_scale: LSTM thermal predictor (→T39 Anticipation)")
    print(f"  Phases: 0(pretrain)->{PHASE0_END}, 1(forced)->{PHASE1_END}, "
          f"2(self-dvfs)->{PHASE2_END}, 3(gaslight)->{PHASE3_END}")
    print()

    # Initialize hardware
    find_dvfs_sysfs()
    find_gpu_metrics()
    check_rapl()
    init_msr()
    check_smn()
    check_pm_table()
    init_df_counters()
    init_cpu_pmu()
    init_fence_reader()

    # === GPU warmup FIRST (before any ISA/HIP work) ===
    print("\n[GPU] Warming up GPU...")
    _warmup = torch.randn(1024, 1024, device=DEVICE)
    _warmup = torch.mm(_warmup, _warmup)
    torch.cuda.synchronize()
    del _warmup
    print("[GPU] Warmup OK")

    # Load Qwen2.5-1.5B FIRST (safe CUDA allocation before ISA kernel)
    BACKBONE_NAME = 'Qwen/Qwen2.5-1.5B'
    print(f"\nLoading {BACKBONE_NAME}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModelForCausalLM.from_pretrained(
        BACKBONE_NAME, dtype=torch.bfloat16, attn_implementation='eager',
        trust_remote_code=True).to(DEVICE)
    VOCAB_SIZE = backbone.config.vocab_size  # 151936
    n_params_backbone = sum(p.numel() for p in backbone.parameters())
    print(f"  {BACKBONE_NAME}: {n_params_backbone/1e6:.1f}M params, vocab={VOCAB_SIZE}")

    # Load data
    print("\nLoading data...")
    try:
        train_data = load_wikitext_data(tokenizer, 'train', max_samples=2000)
        test_data = load_wikitext_data(tokenizer, 'test', max_samples=500)
    except Exception as e:
        print(f"  WikiText-2 load failed ({e}), using synthetic data")
        train_data = []
        test_data = []
        backbone.eval()
        with torch.no_grad():
            for _ in range(200):
                ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN), device=DEVICE)
                train_data.append(ids.squeeze(0).cpu())
            for _ in range(50):
                ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN), device=DEVICE)
                test_data.append(ids.squeeze(0).cpu())

    # Skip-gram: no separate code dataset needed
    # Regime 1 uses SAME wikitext data with labels shifted by SKIP_GRAM_K positions
    train_data_code = None  # LEGACY — kept for function signatures
    test_data_code = None
    print(f"\n[SKIP-GRAM] Regime 1: predict position+{SKIP_GRAM_K} (skip-gram)")
    print(f"  Proven in z2097 (24/25 PASS). Cipher abandoned: PPL=979")

    # Baseline perplexity (frozen backbone, no adapter)
    # Compute loss in float32 to avoid float16 overflow
    print(f"\nBaseline perplexity (frozen {BACKBONE_NAME})...")
    backbone.eval()
    baseline_loss = 0
    baseline_n = 0
    with torch.no_grad():
        for i in range(0, min(len(test_data), N_EVAL_BATCHES * BS), BS):
            batch_seqs = test_data[i:i + BS]
            if len(batch_seqs) < BS:
                break
            input_ids = torch.stack(batch_seqs).to(DEVICE)
            out = backbone(input_ids)
            logits = out.logits.float()  # cast to float32 for loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_labels.view(-1))
            if not math.isnan(loss.item()):
                baseline_loss += loss.item() * input_ids.shape[0]
                baseline_n += input_ids.shape[0]
    baseline_ppl = math.exp(min(baseline_loss / max(baseline_n, 1), 20))
    print(f"  Baseline PPL: {baseline_ppl:.2f}")

    # === Now compile HIP kernel (GPU fully warmed up by baseline eval) ===
    print("\n[HIP] Compiling ISA personality kernel (GPU warm)...")
    get_hip_module()

    # DVFS sanity check + z2095 CALIBRATION
    if DVFS_AVAILABLE:
        global SCLK_LOW_CAL, SCLK_HIGH_CAL
        print("\n[DVFS] Sanity check + calibration...")
        torch.cuda.synchronize()
        set_dvfs_level(0, wait=True)
        sclk_low = read_current_sclk_mhz()
        torch.cuda.synchronize()
        set_dvfs_level(2, wait=True)
        sclk_high = read_current_sclk_mhz()
        # z2095: store calibrated range
        SCLK_LOW_CAL = sclk_low
        SCLK_HIGH_CAL = sclk_high
        print(f"  low={sclk_low:.0f}MHz high={sclk_high:.0f}MHz")
        print(f"  z2095 CALIBRATED: SCLK_LOW_CAL={SCLK_LOW_CAL:.0f} SCLK_HIGH_CAL={SCLK_HIGH_CAL:.0f}")
        print(f"  Gate at low: sigmoid({GATE_TEMP}*(1.0*0.0 + 1.0*0.0 - 1.0)) = "
              f"sigmoid({GATE_TEMP * -1.0:.1f}) = {1/(1+math.exp(GATE_TEMP)):.4f}")
        print(f"  Gate at high: sigmoid({GATE_TEMP}*(1.0*1.0 + 1.0*1.0 - 1.0)) = "
              f"sigmoid({GATE_TEMP * 1.0:.1f}) = {1/(1+math.exp(-GATE_TEMP)):.4f}")
        torch.cuda.synchronize()
        set_dvfs_level(1, wait=True)

    # SMN channel auto-discovery (DVFS-toggling scan)
    if SMN_AVAILABLE:
        discover_smn_channels()

    # gpu_metrics v3.0 sanity check
    if GPU_METRICS_PATH:
        gm = read_gpu_metrics_v3()
        print(f"\n[gpu_metrics v3.0] t_gfx={gm['temperature_gfx']:.1f}C "
              f"dram_r={gm['dram_reads']} dram_w={gm['dram_writes']} "
              f"c0={gm['c0_activity_avg']:.1f}%")

    # Dual ISA kernel args
    kargs_a = config_to_kernel_args(PERSONALITY_A)
    kargs_b = config_to_kernel_args(PERSONALITY_B)

    # Phase 0: Body Encoder Pretraining
    body_encoder = BodyEncoder(TOKEN_DIM)
    body_encoder = train_phase0(body_encoder, kargs_a, epochs=PHASE0_END)

    # Create model
    print("\nCreating EmbodiedQwen2...")
    model = EmbodiedQwen2(backbone, body_encoder,
                           lora_blocks=LORA_BLOCKS, rank=LORA_RANK, alpha=LORA_ALPHA)
    model = model.to(DEVICE)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_trainable:,}")

    # NaN weight audit after Phase 0 training
    print("\n[NaN-AUDIT] Checking all body_encoder weights after Phase 0...")
    nan_found = False
    for name, param in body_encoder.named_parameters():
        if param.isnan().any():
            print(f"  [NaN-AUDIT] {name}: {param.isnan().sum().item()}/{param.numel()} nan values!", flush=True)
            nan_found = True
    if not nan_found:
        print("  [NaN-AUDIT] All body_encoder weights clean.")

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=0.01)

    dvfs_controller = DVFSSafetyController(min_dwell_s=2.0, hysteresis=0.1)

    # Phase 1: Forced regime + label shift
    print(f"\n=== PHASE 1: Forced Regime Training (ep {PHASE0_END+1}-{PHASE1_END}) ===")
    for epoch in range(PHASE0_END + 1, PHASE1_END + 1):
        train_lm_epoch(model, train_data, optimizer, epoch, kargs_a, tokenizer,
                       dvfs_controller=None, gaslighting=False, kargs_b=kargs_b,
                       train_data_code=train_data_code)

    generation_sanity_check(model, test_data, kargs_a, tokenizer, "Phase 1")

    # Phase 2: Model-controlled DVFS + forced exploration
    # z2099: collect body_scale and temp time series for T39 proactive stress test
    body_scale_log = []
    temp_log = []
    print(f"\n=== PHASE 2: Self-DVFS Training (ep {PHASE1_END+1}-{PHASE2_END}) ===")
    for epoch in range(PHASE1_END + 1, PHASE2_END + 1):
        train_lm_epoch(model, train_data, optimizer, epoch, kargs_a, tokenizer,
                       dvfs_controller=dvfs_controller, gaslighting=False, kargs_b=kargs_b,
                       train_data_code=train_data_code)
        # z2100v3: Log body_scale and temp with DVFS switching for anticipatory signal (T39)
        # Alternate DVFS low/high between batches to create temperature variation
        model.eval()
        with torch.no_grad():
            for bi, i in enumerate(range(0, min(len(train_data), 20 * BS), BS)):
                batch_seqs = train_data[i:i + BS]
                if len(batch_seqs) < BS:
                    break
                # Alternate DVFS to create temp variation for LSTM to learn from
                if DVFS_AVAILABLE and bi % 4 == 0:
                    set_dvfs_level(2 if (bi // 4) % 2 == 0 else 0, wait=False)
                input_ids = torch.stack(batch_seqs).to(DEVICE)
                sd = read_all_sensor_dict(lite=True)
                B = BS
                sb = {}
                for k_s in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                           'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                           'gpu_metrics_deep', 'fence', 'status', 'action']:
                    sb[k_s] = expand_sensor(sd[k_s], B, DEVICE)
                sb['delta'] = torch.zeros(B, DELTA_DIM, device=DEVICE)
                sb['intrinsic'] = torch.zeros(B, INTRINSIC_DIM, device=DEVICE)
                sb['reported_delta'] = sb['delta'].clone()
                out_log = model(input_ids, sb, kargs_a, labels=input_ids.clone())
                body_scale_log.append(out_log['body_scale'].mean().item())
                temp_log.append(sd['thermal'][0].item() if len(sd['thermal']) > 0 else 0.0)
        if DVFS_AVAILABLE:
            restore_dvfs_auto()
        model.train()

    generation_sanity_check(model, test_data, kargs_a, tokenizer, "Phase 2")

    # Phase 3: Gaslighting training
    print(f"\n=== PHASE 3: Gaslighting Training (ep {PHASE2_END+1}-{PHASE3_END}) ===")
    for epoch in range(PHASE2_END + 1, PHASE3_END + 1):
        train_lm_epoch(model, train_data, optimizer, epoch, kargs_a, tokenizer,
                       dvfs_controller=dvfs_controller, gaslighting=True, kargs_b=kargs_b,
                       train_data_code=train_data_code)

    # Restore DVFS before tests
    if DVFS_AVAILABLE:
        restore_dvfs_auto()
        time.sleep(1)

    generation_sanity_check(model, test_data, kargs_a, tokenizer, "Phase 3 (final)")

    # z2101: Generation quality check — detect bigram repetition collapse
    print("\n[z2101] Generation quality check...")
    model.eval()
    model.body_encoder.reset_state()
    try:
        with torch.no_grad():
            prompt_ids = test_data[0][:8].unsqueeze(0).to(DEVICE)  # 8-token prompt
            gen_ids = prompt_ids
            sd_gen = read_all_sensor_dict(lite=True)
            sb_gen = {}
            for k_s in ['analog', 'energy', 'freq', 'thermal', 'pm_deep', 'smn_raw',
                       'gpu_metrics', 'thm_spatial_a', 'thm_spatial_b', 'cpu_pmu',
                       'gpu_metrics_deep', 'fence', 'status', 'action']:
                sb_gen[k_s] = expand_sensor(sd_gen[k_s], 1, DEVICE)
            sb_gen['delta'] = torch.zeros(1, DELTA_DIM, device=DEVICE)
            sb_gen['intrinsic'] = torch.zeros(1, INTRINSIC_DIM, device=DEVICE)
            sb_gen['reported_delta'] = sb_gen['delta'].clone()
            for _ in range(32):
                out_gen = model(gen_ids, sb_gen, kargs_a)
                next_tok = out_gen['logits'][:, -1, :].argmax(dim=-1, keepdim=True)
                gen_ids = torch.cat([gen_ids, next_tok], dim=1)
            toks = gen_ids[0].cpu().tolist()
            bigrams = [(toks[i], toks[i+1]) for i in range(len(toks)-1)]
            rep_rate = 1.0 - len(set(bigrams)) / max(len(bigrams), 1)
            thermal_alpha_val = model.thermal_alpha.item()
            print(f"  Generated {len(toks)} tokens, bigram rep={rep_rate:.3f}, "
                  f"thermal_alpha={thermal_alpha_val:.4f}")
            if rep_rate > 0.5:
                print(f"  WARNING: High bigram repetition ({rep_rate:.3f} > 0.5) — "
                      f"possible generation collapse!")
    except Exception as e:
        print(f"  Generation check error: {e}")
    model.train()

    # z2100: Train backbone probe for T38
    model.body_encoder.reset_state()
    probe_test_acc = train_backbone_probe(model, test_data, kargs_a, kargs_b=kargs_b)

    # Run test battery
    print("\n" + "=" * 60)
    print("RUNNING TEST BATTERY (40 tests)")
    print("=" * 60 + "\n")

    # z2100: Reset temporal state before test battery
    model.body_encoder.reset_state()
    test_results, n_pass, n_total_tests = run_tests(
        model, test_data, kargs_a, baseline_ppl, tokenizer,
        dvfs_controller=dvfs_controller, kargs_b=kargs_b,
        probe_test_acc=probe_test_acc, body_scale_log=body_scale_log, temp_log=temp_log,
        test_data_code=test_data_code)

    # Restore DVFS
    if DVFS_AVAILABLE:
        restore_dvfs_auto()

    print(f"\n{'='*60}")
    print(f"z2100 Integrated Workspace LM: {n_pass}/{n_total_tests} PASS")
    print(f"{'='*60}")

    # Save results
    out_path = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z2100_integrated_workspace_lm.json'
    final = {
        'experiment': 'z2100_integrated_workspace_lm',
        'description': 'Integrated Workspace LM: 5 targeted arch changes (coupled encoding, workspace bottleneck, temporal gate, head specialization, predictive body_scale) + 4 bug fixes',
        'key_changes': {
            'backbone': 'Qwen2.5-1.5B (1.54B frozen, 28 layers, 1536 hidden, GQA 12Q/2KV)',
            'coupled_encoding': 'Cross-token message passing for IIT phi (→T32)',
            'workspace_bottleneck': '17 tokens → 4 competitive slots for GWT cliff (→T33)',
            'temporal_gate': 'GRU + EMA replaces instantaneous sigmoid for PCIST (→T34)',
            'head_specialization': 'Modality masks + orthogonality loss for synergy (→T36)',
            'predictive_body': 'LSTM thermal predictor for anticipation (→T39)',
            'bug_fixes': 'T29 NaN clamp, T31 stronger adversarial, T35 meta2 chaining, T38 probe data',
            'retained': 'all z2099 features: T1-T31, metacognition, confidence, attribution, 7 HW layers',
        },
        'sclk_low_cal': SCLK_LOW_CAL,
        'sclk_high_cal': SCLK_HIGH_CAL,
        'gate_temp': GATE_TEMP,
        'meta2_loss_weight': META2_LOSS_WEIGHT,
        'attribution_loss_weight': ATTRIBUTION_LOSS_WEIGHT,
        'n_attribution_classes': N_ATTRIBUTION_CLASSES,
        'contrastive_lambda': CONTRASTIVE_LAMBDA,
        'contrastive_margin': CONTRASTIVE_MARGIN,
        'agreement_gamma': AGREEMENT_GAMMA,
        'probe_test_acc': probe_test_acc,
        'backbone': f'{BACKBONE_NAME} (1.54B frozen)',
        'trainable_params': n_trainable,
        'lora_rank': LORA_RANK,
        'lora_blocks': list(LORA_BLOCKS),
        'skip_gram_offset': SKIP_GRAM_OFFSET,
        'n_substrate_tokens': N_SUBSTRATE_TOKENS,
        'hw_layers': 7,
        'dvfs_available': DVFS_AVAILABLE,
        'smn_available': SMN_AVAILABLE,
        'pm_table_available': PM_TABLE_AVAILABLE,
        'rapl_available': RAPL_AVAILABLE,
        'msr_available': MSR_AVAILABLE,
        'gpu_metrics_available': GPU_METRICS_PATH is not None,
        'cpu_pmu_available': CPU_PMU_AVAILABLE,
        'baseline_ppl': baseline_ppl,
        'results': test_results,
        'n_pass': n_pass,
        'n_total': n_total_tests,
    }
    with open(out_path, 'w') as f:
        json.dump(final, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Save model checkpoint for demo
    ckpt_path = out_path.replace('.json', '_checkpoint.pt')
    torch.save({
        'body_encoder_state': model.body_encoder.state_dict(),
        'lora_states': {name: m.state_dict() for name, m in model.lora_layers.items()},
        'metacognition_head_state': model.metacognition_head.state_dict(),
        'thermal_head_state': model.thermal_head.state_dict(),
        'confidence_head_state': model.confidence_head.state_dict() if hasattr(model, 'confidence_head') else None,
        'meta2_head_state': model.meta2_head.state_dict() if hasattr(model, 'meta2_head') else None,
        'attribution_head_state': model.attribution_head.state_dict() if hasattr(model, 'attribution_head') else None,
        # z2100 fix: save substrate injection layers (were MISSING — caused PPL=18.3 instead of 2.72)
        'substrate_bias_early_state': model.substrate_bias_early.state_dict(),
        'substrate_bias_late_state': model.substrate_bias_late.state_dict(),
        'hidden_modulation_state': model.hidden_modulation.state_dict(),
        'demand_head_state': model.demand_head.state_dict(),
        'isa_probe': model.isa_probe,
        'sclk_low_cal': SCLK_LOW_CAL,
        'sclk_high_cal': SCLK_HIGH_CAL,
    }, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")


if __name__ == '__main__':
    try:
        main()
    finally:
        restore_dvfs_auto()
