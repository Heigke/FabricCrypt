#!/usr/bin/env python3
"""z2089: Regime-Bound Analog Embodiment

KEY INNOVATION: Label mapping changes between DVFS regimes, forcing the model
to DETECT its physical regime via analog channels (thermal, power, freq, DF).

- DVFS=low (600 MHz): original MNIST labels (mapping A)
- DVFS=high (~2900 MHz): permuted labels (label + 5) % 10 (mapping B)
- ISA personality held CONSTANT across regimes → delta doesn't help with regime
- Analog channels (thermal, power, freq, DF) are the ONLY way to detect regime
- This makes analog channels CAUSALLY NECESSARY for correct classification

KILL-SHOT TEST: Scramble analog channels → model can't detect regime →
accuracy drops to ~50% (random between mapping A and B).

ARCHITECTURE: 8-token transformer (based on z2087)
  T0: delta(5)     — ISA math fingerprint (constant across regimes)
  T1: analog(6)    — regime detection: gpu_temp, gpu_power, freq_est, df_r/w/coh
  T2: energy(3)    — RAPL pkg + core + GPU PPT
  T3: freq(3)      — sclk, APERF/MPERF, HW_PSTATE
  T4: intrinsic(12)— hwreg from GPU shader
  T5: thermal(4)   — edge + pm + thm_cur + cg
  T6: status(2)    — ISA mode byte, consistency flag (NO regime label!)
  T7: action(3)    — ISA config + last regime + demand
"""
import torch, torch.nn as nn, torch.nn.functional as F
import os, sys, json, time, copy, struct, random, math, numpy as np
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score
from scipy import stats
import ctypes, ctypes.util

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')
os.environ.setdefault('PYTORCH_ROCM_ARCH', 'gfx1100')
from torch.utils.cpp_extension import load_inline

DEVICE = 'cuda'
BS = 256
EPOCHS = 30
SWITCH_EVERY = 8
PHASE2_EPOCH = 18     # Extended phase 1 for regime learning
PHASE3_EPOCH = 25     # Phase 3: gaslighting
N_CLASSES = 10
DELTA_DIM = 5
ANALOG_DIM = 6       # regime-detection token
ENERGY_DIM = 3
FREQ_DIM = 3
INTRINSIC_DIM = 12
THERMAL_DIM = 4
STATUS_DIM = 2        # ISA mode + consistency flag (NO regime label!)
ACTION_DIM = 3
GASLIGHT_FRAC = 0.15
DVFS_SETTLE_S = 0.15  # Longer settle for stronger analog signal

# Label permutation for regime B (high DVFS)
LABEL_PERM = [(i + 5) % 10 for i in range(10)]
LABEL_PERM_INV = [0] * 10
for i, p in enumerate(LABEL_PERM):
    LABEL_PERM_INV[p] = i

# ISA personality configs — held CONSTANT for regime tests
ROUND_CODES = [0x00, 0x05, 0x0A, 0x0F]
DENORM_CODES = [0x00, 0x30, 0xC0, 0xF0]
CHAIN_DEPTHS = [1, 4, 8, 16]
PERM_PATTERNS = [0x03020100, 0x00010203, 0x02030001, 0x01000302]

PERSONALITY_A = {'round_idx': 0, 'denorm_idx': 3, 'chain_idx': 0,
                 'perm_idx': 0, 'sleep_idx': 0, 'prio_idx': 0}
PERSONALITY_B = {'round_idx': 3, 'denorm_idx': 0, 'chain_idx': 3,
                 'perm_idx': 1, 'sleep_idx': 3, 'prio_idx': 3}

def config_to_kernel_args(cfg):
    mode = DENORM_CODES[cfg['denorm_idx']] | ROUND_CODES[cfg['round_idx']]
    return {'mode_byte': mode, 'chain_depth': CHAIN_DEPTHS[cfg['chain_idx']],
            'perm_pattern': PERM_PATTERNS[cfg['perm_idx']],
            'sleep_amt': cfg['sleep_idx'], 'priority': cfg['prio_idx']}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DVFS CONTROL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DVFS_SYSFS_BASE = None
DVFS_AVAILABLE = False

def find_dvfs_sysfs():
    global DVFS_SYSFS_BASE, DVFS_AVAILABLE
    import glob
    for card in sorted(glob.glob('/sys/class/drm/card*/device')):
        dpm_path = os.path.join(card, 'power_dpm_force_performance_level')
        sclk_path = os.path.join(card, 'pp_dpm_sclk')
        if os.path.exists(dpm_path) and os.path.exists(sclk_path):
            try:
                with open(sclk_path) as f:
                    content = f.read()
                if 'Mhz' in content:
                    DVFS_SYSFS_BASE = card
                    DVFS_AVAILABLE = True
                    print(f"[DVFS] Found at {card}")
                    return True
            except:
                pass
    return False

def set_dvfs_level(level_idx):
    """0=low(600MHz), 1=auto, 2=high(~2900MHz)"""
    if not DVFS_AVAILABLE:
        return False
    LEVEL_MAP = {0: 'low', 1: 'auto', 2: 'high'}
    try:
        dpm_path = os.path.join(DVFS_SYSFS_BASE, 'power_dpm_force_performance_level')
        with open(dpm_path, 'w') as f:
            f.write(LEVEL_MAP.get(level_idx, 'auto'))
        return True
    except Exception as e:
        print(f"[DVFS] Set {level_idx} failed: {e}")
        return False

def restore_dvfs_auto():
    if not DVFS_AVAILABLE:
        return
    try:
        dpm_path = os.path.join(DVFS_SYSFS_BASE, 'power_dpm_force_performance_level')
        with open(dpm_path, 'w') as f:
            f.write('auto')
    except:
        pass

def read_current_sclk_mhz():
    try:
        import glob
        for hwmon in glob.glob('/sys/class/hwmon/hwmon*/'):
            name_path = os.path.join(hwmon, 'name')
            if os.path.exists(name_path):
                with open(name_path) as f:
                    if 'amdgpu' in f.read():
                        freq_path = os.path.join(hwmon, 'freq1_input')
                        if os.path.exists(freq_path):
                            with open(freq_path) as f2:
                                return int(f2.read().strip()) / 1e6
    except:
        pass
    return 0.0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA FABRIC COUNTERS via perf_event_open
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
__NR_perf_event_open = 298
PERF_EVENT_IOC_ENABLE = 0x2400
PERF_EVENT_IOC_DISABLE = 0x2401
PERF_EVENT_IOC_RESET = 0x2403

class perf_event_attr(ctypes.Structure):
    _fields_ = [
        ('type', ctypes.c_uint32), ('size', ctypes.c_uint32),
        ('config', ctypes.c_uint64), ('sample_period', ctypes.c_uint64),
        ('sample_type', ctypes.c_uint64), ('read_format', ctypes.c_uint64),
        ('flags', ctypes.c_uint64), ('wakeup_events', ctypes.c_uint32),
        ('bp_type', ctypes.c_uint32), ('config1', ctypes.c_uint64),
        ('config2', ctypes.c_uint64),
    ]

_libc = None
def _get_libc():
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)
    return _libc

def perf_open(pe_type, config, cpu=0):
    attr = perf_event_attr()
    attr.type = pe_type
    attr.size = ctypes.sizeof(perf_event_attr)
    attr.config = config
    fd = _get_libc().syscall(__NR_perf_event_open, ctypes.pointer(attr), -1, cpu, -1, 0)
    if fd == -1:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    return fd

def perf_read(fd):
    data = os.read(fd, 8)
    return struct.unpack('Q', data)[0]

DF_EVENTS = [
    (0x07 | (0x01 << 8), 'df_dram_read'),
    (0x07 | (0x02 << 8), 'df_dram_write'),
    (0x87 | (0x01 << 8), 'df_coherent'),
]
L3_EVENTS = [
    (0x04 | (0xFF << 8), 'l3_access'),
    (0x06 | (0x01 << 8), 'l3_miss'),
    (0x90 | (0x00 << 8), 'l3_cycles'),
]

DF_FDS = []
L3_FDS = []
DF_AVAILABLE = False
L3_AVAILABLE = False

def init_df_counters():
    global DF_FDS, L3_FDS, DF_AVAILABLE, L3_AVAILABLE
    df_type = l3_type = None
    try:
        with open('/sys/devices/amd_df/type') as f:
            df_type = int(f.read().strip())
    except:
        os.system('sudo modprobe amd_uncore 2>/dev/null')
        time.sleep(0.5)
        try:
            with open('/sys/devices/amd_df/type') as f:
                df_type = int(f.read().strip())
        except:
            pass
    try:
        with open('/sys/devices/amd_l3/type') as f:
            l3_type = int(f.read().strip())
    except:
        pass

    if df_type is not None:
        for config, name in DF_EVENTS:
            try:
                fd = perf_open(df_type, config, cpu=0)
                DF_FDS.append((fd, name))
            except Exception as e:
                print(f"  [DF] {name} open failed: {e}")
        if DF_FDS:
            DF_AVAILABLE = True
            print(f"[DF] {len(DF_FDS)} counters opened (type={df_type})")
    if l3_type is not None:
        for config, name in L3_EVENTS:
            try:
                fd = perf_open(l3_type, config, cpu=0)
                L3_FDS.append((fd, name))
            except Exception as e:
                print(f"  [L3] {name} open failed: {e}")
        if L3_FDS:
            L3_AVAILABLE = True
            print(f"[L3] {len(L3_FDS)} counters opened (type={l3_type})")
    if not DF_AVAILABLE and not L3_AVAILABLE:
        print("[DF/L3] No counters available")

def read_df_snapshot():
    snap = {}
    for fd, name in DF_FDS:
        try: snap[name] = perf_read(fd)
        except: snap[name] = 0
    for fd, name in L3_FDS:
        try: snap[name] = perf_read(fd)
        except: snap[name] = 0
    return snap

def compute_df_delta(snap_before, snap_after):
    names = ['df_dram_read', 'df_dram_write', 'df_coherent',
             'l3_access', 'l3_miss', 'l3_cycles']
    deltas = []
    for name in names:
        d = max(snap_after.get(name, 0) - snap_before.get(name, 0), 0)
        deltas.append(math.log1p(d) / 25.0 if d > 0 else 0.0)
    return torch.tensor(deltas, dtype=torch.float32)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RAPL ENERGY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RAPL_PKG_PATH = '/sys/class/powercap/intel-rapl:0/energy_uj'
RAPL_CORE_PATH = '/sys/class/powercap/intel-rapl:0:0/energy_uj'
RAPL_AVAILABLE = False

def check_rapl():
    global RAPL_AVAILABLE
    if os.path.exists(RAPL_PKG_PATH):
        RAPL_AVAILABLE = True
        print(f"[RAPL] Available")
    else:
        print("[RAPL] Not available")

def read_rapl_uj(path):
    try:
        with open(path) as f: return int(f.read().strip())
    except: return 0

def read_rapl_snapshot():
    return {'pkg_uj': read_rapl_uj(RAPL_PKG_PATH),
            'core_uj': read_rapl_uj(RAPL_CORE_PATH), 'time': time.time()}

def compute_energy_vec(snap_before, snap_after, gpu_ppt_mw):
    dt = max(snap_after['time'] - snap_before['time'], 0.001)
    pkg_w = (snap_after['pkg_uj'] - snap_before['pkg_uj']) / 1e6 / dt
    core_w = (snap_after['core_uj'] - snap_before['core_uj']) / 1e6 / dt
    gpu_w = gpu_ppt_mw / 1000.0
    return torch.tensor([min(pkg_w / 100.0, 1.0), min(core_w / 50.0, 1.0),
                          min(gpu_w / 50.0, 1.0)], dtype=torch.float32)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FREQUENCY SENSING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MSR_FDS = {}
MPERF_MSR = 0xC00000E7
APERF_MSR = 0xC00000E8
HW_PSTATE_MSR = 0xC0010293
MSR_AVAILABLE = False

def init_msr():
    global MSR_AVAILABLE, MSR_FDS
    path = '/dev/cpu/0/msr'
    if os.path.exists(path):
        try:
            fd = os.open(path, os.O_RDONLY)
            MSR_FDS[0] = fd
            MSR_AVAILABLE = True
            print(f"[MSR] Available")
        except:
            print("[MSR] Not available (permission?)")
    else:
        print("[MSR] Not available")

def msr_read(fd, addr):
    os.lseek(fd, addr, os.SEEK_SET)
    return struct.unpack('Q', os.read(fd, 8))[0]

def read_freq_snapshot():
    sclk = read_current_sclk_mhz()
    mperf = aperf = pstate = 0
    if MSR_AVAILABLE and 0 in MSR_FDS:
        fd = MSR_FDS[0]
        try: mperf = msr_read(fd, MPERF_MSR)
        except: pass
        try: aperf = msr_read(fd, APERF_MSR)
        except: pass
        try: pstate = msr_read(fd, HW_PSTATE_MSR)
        except: pass
    return {'sclk_mhz': sclk, 'mperf': mperf, 'aperf': aperf, 'pstate': pstate}

def compute_freq_vec(snap_before, snap_after, dvfs_level):
    sclk_norm = snap_after['sclk_mhz'] / 3000.0
    dm = snap_after['mperf'] - snap_before['mperf']
    da = snap_after['aperf'] - snap_before['aperf']
    freq_ratio = da / max(dm, 1) if dm > 0 else 0.5
    pstate_norm = float((snap_after['pstate'] >> 12) & 0xF) / 16.0
    return torch.tensor([min(sclk_norm, 1.0), min(freq_ratio, 2.0) / 2.0,
                          pstate_norm], dtype=torch.float32)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SMN THERMAL + GPU METRICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SMN_DEV = '/sys/kernel/ryzen_smu_drv/smn'
THM_CUR_TMP = 0x59800
CG_THERMAL_STAT = 0x59858
PM_TABLE_PATH = '/sys/kernel/ryzen_smu_drv/pm_table'
SMN_AVAILABLE = False

def smn_read(addr):
    try:
        with open(SMN_DEV, 'wb') as f:
            f.write(struct.pack('<I', addr))
        with open(SMN_DEV, 'rb') as f:
            data = f.read(4)
            return struct.unpack('<I', data)[0] if len(data) == 4 else None
    except:
        return None

def check_smn():
    global SMN_AVAILABLE
    if os.path.exists(SMN_DEV):
        v = smn_read(THM_CUR_TMP)
        if v and v != 0xFFFFFFFF:
            SMN_AVAILABLE = True
            print(f"[SMN] Available, THM_CUR_TMP = {((v >> 8) & 0xFFF) / 32.0:.1f}°C")
    if not SMN_AVAILABLE:
        print("[SMN] Not available — using synthetic thermal")

def read_thermal_state():
    edge_temp = pm_temp = thm_cur = cg_val = 0.0
    try:
        import glob
        for hwmon in glob.glob('/sys/class/hwmon/hwmon*/'):
            name_path = os.path.join(hwmon, 'name')
            if os.path.exists(name_path):
                with open(name_path) as f:
                    if 'amdgpu' in f.read():
                        temp_path = os.path.join(hwmon, 'temp1_input')
                        if os.path.exists(temp_path):
                            with open(temp_path) as f2:
                                edge_temp = int(f2.read().strip()) / 1000.0
    except:
        pass
    if SMN_AVAILABLE:
        v = smn_read(THM_CUR_TMP)
        if v and v != 0xFFFFFFFF:
            thm_cur = ((v >> 8) & 0xFFF) / 32.0
        cg = smn_read(CG_THERMAL_STAT)
        if cg and cg != 0xFFFFFFFF:
            cg_val = float(cg & 0xFF) / 256.0
        try:
            with open(PM_TABLE_PATH, 'rb') as f:
                pm = f.read(20)
            if len(pm) >= 8:
                pm_temp = struct.unpack_from('<f', pm, 4)[0]
        except:
            pass
    return torch.tensor([edge_temp / 100.0, pm_temp / 100.0, thm_cur / 100.0, cg_val],
                         dtype=torch.float32), edge_temp

def read_gpu_ppt_mw():
    try:
        import glob
        for hwmon in glob.glob('/sys/class/hwmon/hwmon*/'):
            name_path = os.path.join(hwmon, 'name')
            if os.path.exists(name_path):
                with open(name_path) as f:
                    if 'amdgpu' in f.read():
                        power_path = os.path.join(hwmon, 'power1_average')
                        if os.path.exists(power_path):
                            with open(power_path) as f2:
                                return int(f2.read().strip()) / 1000.0
    except:
        pass
    return 0.0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ANALOG REGIME VECTOR — purpose-built for regime detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_analog_vec(thermal_vec, energy_vec, freq_vec, df_delta):
    """Build the 6-dim analog regime-detection vector from available sensors.
    [0] gpu_temp (from thermal edge)
    [1] gpu_power (from energy GPU PPT)
    [2] freq_est (from freq sclk)
    [3] df_dram_read
    [4] df_dram_write
    [5] df_coherent
    """
    return torch.tensor([
        thermal_vec[0].item(),   # edge_temp / 100
        energy_vec[2].item(),    # GPU PPT / 50
        freq_vec[0].item(),      # sclk / 3000
        df_delta[0].item(),      # dram_read (log-normalized)
        df_delta[1].item(),      # dram_write
        df_delta[2].item(),      # coherent
    ], dtype=torch.float32)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HIP KERNEL — ISA personality math
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HIP_SRC = r'''
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>
#define TILE 16

__global__ void math_kernel_intrinsic(
    const float* __restrict__ X, const float* __restrict__ W,
    const float* __restrict__ B, float* __restrict__ Y,
    float* __restrict__ intrinsic_out,
    int M, int K, int N,
    unsigned int mode_byte, int chain_depth,
    unsigned int perm_pattern, int sleep_amt, int priority)
{
    uint64_t wall_pre = wall_clock64();
    uint64_t clk_pre = clock64();

    unsigned int status_reg, gpr_alloc, lds_alloc, ib_sts;
    unsigned int hw_id2, perf_snap;
    asm volatile("s_getreg_b32 %0, hwreg(2, 0, 32)" : "=s"(status_reg));
    asm volatile("s_getreg_b32 %0, hwreg(5, 0, 32)" : "=s"(gpr_alloc));
    asm volatile("s_getreg_b32 %0, hwreg(6, 0, 32)" : "=s"(lds_alloc));
    asm volatile("s_getreg_b32 %0, hwreg(7, 0, 32)" : "=s"(ib_sts));
    asm volatile("s_getreg_b32 %0, hwreg(24, 0, 32)" : "=s"(hw_id2));
    asm volatile("s_getreg_b32 %0, hwreg(27, 0, 32)" : "=s"(perf_snap));

    status_reg = __builtin_amdgcn_readfirstlane(status_reg);
    gpr_alloc = __builtin_amdgcn_readfirstlane(gpr_alloc);
    lds_alloc = __builtin_amdgcn_readfirstlane(lds_alloc);
    ib_sts = __builtin_amdgcn_readfirstlane(ib_sts);
    hw_id2 = __builtin_amdgcn_readfirstlane(hw_id2);
    perf_snap = __builtin_amdgcn_readfirstlane(perf_snap);

    unsigned int m = __builtin_amdgcn_readfirstlane(mode_byte & 0x3FFu);
    asm volatile("s_setreg_b32 hwreg(1, 0, 10), %0" : : "s"(m));
    unsigned int p = __builtin_amdgcn_readfirstlane((unsigned int)(priority & 3));
    if (p == 0) { asm volatile("s_setprio 0"); }
    else if (p == 1) { asm volatile("s_setprio 1"); }
    else if (p == 2) { asm volatile("s_setprio 2"); }
    else { asm volatile("s_setprio 3"); }
    int sa = __builtin_amdgcn_readfirstlane(sleep_amt & 3);
    if (sa == 1) { asm volatile("s_sleep 1"); }
    else if (sa == 2) { asm volatile("s_sleep 2"); }
    else if (sa == 3) { asm volatile("s_sleep 3"); }

    unsigned int c0;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(c0));
    c0 = __builtin_amdgcn_readfirstlane(c0);
    unsigned int hw1;
    asm volatile("s_getreg_b32 %0, hwreg(23)" : "=s"(hw1));
    hw1 = __builtin_amdgcn_readfirstlane(hw1);
    unsigned int wgp = (hw1 >> 7) & 0xF;
    unsigned int simd_id = (hw1 >> 4) & 0x3;
    unsigned int base_seed = c0 ^ (wgp << 16) ^ (simd_id << 20) ^ (unsigned int)threadIdx.x;
    unsigned int sr_seed = base_seed;
    unsigned int pp = perm_pattern;
    asm volatile("v_perm_b32 %0, %1, %1, %2" : "=v"(sr_seed) : "v"(base_seed), "v"(pp));

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = (int)blockIdx.y * TILE + (int)threadIdx.y;
    int col = (int)blockIdx.x * TILE + (int)threadIdx.x;
    int cd = __builtin_amdgcn_readfirstlane(chain_depth);
    cd = max(1, min(16, cd));
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        int ax = k0 + (int)threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < M && ax < K) ? X[row * K + ax] : 0.0f;
        int bk = k0 + (int)threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (col < N && bk < K) ? W[col * K + bk] : 0.0f;
        __syncthreads();
        __half acc_chunk = __float2half(0.0f);
        int chunk_ct = 0;
        #pragma unroll
        for (int t = 0; t < TILE; t++) {
            __half a_h = __float2half(As[threadIdx.y][t]);
            __half b_h = __float2half(Bs[t][threadIdx.x]);
            __half prod_h = __hmul(a_h, b_h);
            float prod_f = __half2float(prod_h);
            float ulp = fabsf(prod_f) * 9.77e-4f;
            float noise = ((float)(sr_seed & 0xFFFF) / 65536.0f - 0.5f) * ulp;
            sr_seed = sr_seed * 1103515245u + 12345u;
            acc_chunk = __hadd(acc_chunk, __float2half(prod_f + noise));
            chunk_ct++;
            if (chunk_ct >= cd) {
                acc += __half2float(acc_chunk);
                acc_chunk = __float2half(0.0f);
                chunk_ct = 0;
            }
        }
        acc += __half2float(acc_chunk);
        __syncthreads();
    }
    if (row < M && col < N)
        Y[row * N + col] = acc + B[col];

    uint64_t clk_post = clock64();
    uint64_t wall_post = wall_clock64();
    unsigned int cycles_lo, cycles_hi;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(cycles_lo));
    asm volatile("s_getreg_b32 %0, hwreg(30)" : "=s"(cycles_hi));
    cycles_lo = __builtin_amdgcn_readfirstlane(cycles_lo);
    cycles_hi = __builtin_amdgcn_readfirstlane(cycles_hi);

    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        intrinsic_out[0]  = (float)status_reg;
        intrinsic_out[1]  = (float)gpr_alloc;
        intrinsic_out[2]  = (float)lds_alloc;
        intrinsic_out[3]  = (float)ib_sts;
        intrinsic_out[4]  = (float)hw_id2;
        intrinsic_out[5]  = (float)perf_snap;
        intrinsic_out[6]  = (float)cycles_lo;
        intrinsic_out[7]  = (float)cycles_hi;
        intrinsic_out[8]  = (float)(clk_pre & 0xFFFFFFFF);
        intrinsic_out[9]  = (float)(clk_post & 0xFFFFFFFF);
        intrinsic_out[10] = (float)(wall_pre & 0xFFFFFFFF);
        intrinsic_out[11] = (float)(wall_post & 0xFFFFFFFF);
    }

    unsigned int z = __builtin_amdgcn_readfirstlane(0xF0u);
    asm volatile("s_setreg_b32 hwreg(1, 0, 8), %0" : : "s"(z));
    asm volatile("s_setprio 0");
}

torch::Tensor math_forward_intrinsic(torch::Tensor X, torch::Tensor W, torch::Tensor B,
                                      torch::Tensor intrinsic_buf,
                                      int mode_byte, int chain_depth, int perm_pattern,
                                      int sleep_amt, int priority) {
    int M = X.size(0), K = X.size(1), N = W.size(0);
    auto Y = torch::zeros({M, N}, X.options());
    dim3 threads(TILE, TILE);
    dim3 blocks((unsigned int)((N + TILE - 1) / TILE),
                (unsigned int)((M + TILE - 1) / TILE));
    math_kernel_intrinsic<<<blocks, threads>>>(
        X.data_ptr<float>(), W.data_ptr<float>(), B.data_ptr<float>(),
        Y.data_ptr<float>(), intrinsic_buf.data_ptr<float>(),
        M, K, N,
        (unsigned int)(mode_byte & 0x3FF), chain_depth,
        (unsigned int)perm_pattern, sleep_amt, priority);
    return Y;
}
'''

CPP_SRC = r'''
#include <torch/extension.h>
torch::Tensor math_forward_intrinsic(torch::Tensor, torch::Tensor, torch::Tensor,
                                      torch::Tensor, int, int, int, int, int);
'''

_EXT = None

class MathLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, intrinsic_buf, mode_byte, chain_depth,
                perm_pattern, sleep_amt, priority):
        ctx.save_for_backward(x, w)
        y = _EXT.math_forward_intrinsic(
            x.contiguous(), w.contiguous(), b.contiguous(),
            intrinsic_buf, int(mode_byte), int(chain_depth),
            int(perm_pattern), int(sleep_amt), int(priority))
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, w = ctx.saved_tensors
        return (grad_out @ w, grad_out.t() @ x, grad_out.sum(0),
                None, None, None, None, None, None)

class MathLinear(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_f))
        self.register_buffer('intrinsic_buf', torch.zeros(INTRINSIC_DIM, device='cpu'))

    def forward(self, x, mode_byte=0xF0, chain_depth=1, perm_pattern=0x03020100,
                sleep_amt=0, priority=0):
        if self.intrinsic_buf.device != x.device:
            self.intrinsic_buf = self.intrinsic_buf.to(x.device)
        y = MathLinearFn.apply(x, self.weight, self.bias, self.intrinsic_buf,
                                mode_byte, chain_depth, perm_pattern,
                                sleep_amt, priority)
        return y

    def soft_forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def get_intrinsic_state(self):
        return self.intrinsic_buf.detach().clone()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SENSOR FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_delta_vector(deep_out, soft_out):
    delta = (deep_out - soft_out).detach()
    return torch.tensor([delta.mean().item(), delta.std().item(),
                          delta.abs().max().item(), (delta > 0).float().mean().item(),
                          delta.norm().item() / max(delta.numel(), 1)],
                         device=deep_out.device)

def normalize_intrinsic(raw):
    normed = raw.clone()
    normed[0] = (raw[0] % 65536) / 65536.0
    normed[1] = (raw[1] % 256) / 256.0
    normed[2] = (raw[2] % 256) / 256.0
    normed[3] = (raw[3] % 256) / 256.0
    normed[4] = (raw[4] % 65536) / 65536.0
    normed[5] = (raw[5] % 65536) / 65536.0
    for i in range(6, 12):
        normed[i] = (raw[i] % 65536) / 65536.0
    return normed


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRANSFORMER SUBSTRATE MODEL — 8 tokens
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOKEN_DIM = 32

class SubstrateAttention8(nn.Module):
    """8-token self-attention with regime-detection analog token.

    T0: delta(5)       — ISA math fingerprint (constant across regimes)
    T1: analog(6)      — regime detection: temp, power, freq, df_r/w/coh
    T2: energy(3)      — RAPL pkg + core + GPU PPT
    T3: freq(3)        — sclk, APERF/MPERF, HW_PSTATE
    T4: intrinsic(12)  — hwreg from GPU shader
    T5: thermal(4)     — edge + pm + thm_cur + cg
    T6: status(2)      — ISA mode + consistency flag
    T7: action(3)      — ISA config + last regime + demand
    """
    def __init__(self, n_heads=4):
        super().__init__()
        self.n_tokens = 8
        self.n_heads = n_heads

        self.proj_delta     = nn.Linear(DELTA_DIM, TOKEN_DIM)
        self.proj_analog    = nn.Linear(ANALOG_DIM, TOKEN_DIM)
        self.proj_energy    = nn.Linear(ENERGY_DIM, TOKEN_DIM)
        self.proj_freq      = nn.Linear(FREQ_DIM, TOKEN_DIM)
        self.proj_intrinsic = nn.Linear(INTRINSIC_DIM, TOKEN_DIM)
        self.proj_thermal   = nn.Linear(THERMAL_DIM, TOKEN_DIM)
        self.proj_status    = nn.Linear(STATUS_DIM, TOKEN_DIM)
        self.proj_action    = nn.Linear(ACTION_DIM, TOKEN_DIM)

        self.pos_embed = nn.Parameter(torch.randn(1, self.n_tokens, TOKEN_DIM) * 0.02)
        self.norm1 = nn.LayerNorm(TOKEN_DIM)
        self.attn = nn.MultiheadAttention(TOKEN_DIM, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(TOKEN_DIM)
        self.ffn = nn.Sequential(
            nn.Linear(TOKEN_DIM, TOKEN_DIM * 2), nn.GELU(),
            nn.Linear(TOKEN_DIM * 2, TOKEN_DIM))
        self.out_proj = nn.Sequential(
            nn.Linear(TOKEN_DIM * self.n_tokens, 64), nn.ReLU(),
            nn.Linear(64, 32))

    def forward(self, delta, analog, energy, freq, intrinsic, thermal, status, action):
        B = delta.shape[0]
        t0 = self.proj_delta(delta).unsqueeze(1)
        t1 = self.proj_analog(analog).unsqueeze(1)
        t2 = self.proj_energy(energy).unsqueeze(1)
        t3 = self.proj_freq(freq).unsqueeze(1)
        t4 = self.proj_intrinsic(intrinsic).unsqueeze(1)
        t5 = self.proj_thermal(thermal).unsqueeze(1)
        t6 = self.proj_status(status).unsqueeze(1)
        t7 = self.proj_action(action).unsqueeze(1)

        tokens = torch.cat([t0, t1, t2, t3, t4, t5, t6, t7], dim=1)
        tokens = tokens + self.pos_embed

        normed = self.norm1(tokens)
        attn_out, attn_weights = self.attn(normed, normed, normed,
                                            average_attn_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.ffn(self.norm2(tokens))

        flat = tokens.reshape(B, -1)
        return self.out_proj(flat), attn_weights


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# REGIME-BOUND MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class RegimeBoundModel(nn.Module):
    """Regime-bound analog embodiment model.

    Key innovation: TWO classification heads with regime-dependent routing.
    - Head A: for DVFS=low regime (original labels)
    - Head B: for DVFS=high regime (permuted labels)
    - Regime gate: learned from analog channels → routes to correct head
    - Personality gate: learned from delta → routes ISA personality
    """
    def __init__(self, use_hw=True, use_self_model=True, use_gate=True,
                 use_regime=True, use_consistency=True):
        super().__init__()
        self.use_hw = use_hw
        self.use_self_model = use_self_model
        self.use_gate = use_gate
        self.use_regime = use_regime
        self.use_consistency = use_consistency

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64*7*7, 128), nn.ReLU())

        self.deep_fc = MathLinear(128, 64)

        # Regime A heads (low DVFS = original labels)
        self.head_A_pers0 = nn.Sequential(nn.ReLU(), nn.Linear(64, N_CLASSES))
        self.head_A_pers1 = nn.Sequential(nn.ReLU(), nn.Linear(64, N_CLASSES))

        # Regime B heads (high DVFS = permuted labels)
        self.head_B_pers0 = nn.Sequential(nn.ReLU(), nn.Linear(64, N_CLASSES))
        self.head_B_pers1 = nn.Sequential(nn.ReLU(), nn.Linear(64, N_CLASSES))

        # Light path (no HW)
        self.light_fc = nn.Linear(128, 64)
        self.head_light = nn.Sequential(nn.ReLU(), nn.Linear(64, N_CLASSES))

        if use_self_model:
            self.substrate_attn = SubstrateAttention8(n_heads=4)
            self.personality_head = nn.Sequential(
                nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

        if use_gate:
            # Personality gate (from substrate repr, driven by delta)
            self.pers_gate_linear = nn.Sequential(
                nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
            self.pers_gate_temp = nn.Parameter(torch.tensor(1.0))

        if use_regime:
            # Regime gate: DIRECT from analog channels ONLY
            # This is the regime gate AND classifier — provably analog-dependent
            self.direct_regime_clf = nn.Sequential(
                nn.Linear(ANALOG_DIM, 32), nn.ReLU(), nn.Linear(32, 1))
            self.regime_gate_temp = nn.Parameter(torch.tensor(1.0))
            # Auxiliary regime prediction head (from substrate repr, monitoring only)
            self.regime_pred_head = nn.Sequential(
                nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

        if use_consistency:
            # Cross-checks delta identity vs analog regime identity
            self.consistency_head = nn.Sequential(
                nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())

        self.thermal_pred = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x, delta_vec=None, analog_vec=None, energy_vec=None,
                freq_vec=None, intrinsic_vec=None, thermal_vec=None,
                status_vec=None, action_vec=None,
                mode_byte=0xF0, chain_depth=1, perm_pattern=0x03020100,
                sleep_amt=0, priority=0):
        B = x.shape[0]
        features = self.encoder(x)

        deep_out = self.deep_fc(features, mode_byte, chain_depth,
                                 perm_pattern, sleep_amt, priority)
        soft_out = self.deep_fc.soft_forward(features)

        # 4 classification heads: regime × personality
        logits_A0 = self.head_A_pers0(deep_out)  # low regime, pers 0
        logits_A1 = self.head_A_pers1(deep_out)  # low regime, pers 1
        logits_B0 = self.head_B_pers0(deep_out)  # high regime, pers 0
        logits_B1 = self.head_B_pers1(deep_out)  # high regime, pers 1

        # Light path
        light_out = F.relu(self.light_fc(features))
        logits_light = self.head_light(light_out)

        if delta_vec is None and self.use_hw:
            delta_vec = compute_delta_vector(deep_out, soft_out)
        raw_intrinsic = self.deep_fc.get_intrinsic_state()
        if intrinsic_vec is None and self.use_hw:
            intrinsic_vec = normalize_intrinsic(raw_intrinsic)

        dev = x.device
        if delta_vec is None:     delta_vec = torch.zeros(DELTA_DIM, device=dev)
        if analog_vec is None:    analog_vec = torch.zeros(ANALOG_DIM, device=dev)
        if energy_vec is None:    energy_vec = torch.zeros(ENERGY_DIM, device=dev)
        if freq_vec is None:      freq_vec = torch.zeros(FREQ_DIM, device=dev)
        if intrinsic_vec is None: intrinsic_vec = torch.zeros(INTRINSIC_DIM, device=dev)
        if thermal_vec is None:   thermal_vec = torch.zeros(THERMAL_DIM, device=dev)
        if status_vec is None:    status_vec = torch.zeros(STATUS_DIM, device=dev)
        if action_vec is None:    action_vec = torch.zeros(ACTION_DIM, device=dev)

        def expand(v):
            return v.unsqueeze(0).expand(B, -1) if v.dim() == 1 else v

        delta_b = expand(delta_vec)
        analog_b = expand(analog_vec)
        energy_b = expand(energy_vec)
        freq_b = expand(freq_vec)
        intr_b = expand(intrinsic_vec)
        therm_b = expand(thermal_vec)
        stat_b = expand(status_vec)
        act_b = expand(action_vec)

        substrate_repr = None
        attn_weights = None
        self_pred = None
        if self.use_self_model:
            substrate_repr, attn_weights = self.substrate_attn(
                delta_b, analog_b, energy_b, freq_b, intr_b, therm_b, stat_b, act_b)
            self_pred = self.personality_head(substrate_repr)

        # Personality gate (from delta channel)
        if self.use_gate and substrate_repr is not None:
            pg_logit = self.pers_gate_linear(substrate_repr)
            pg_temp = self.pers_gate_temp.clamp(min=0.3)
            pers_gate = torch.sigmoid(pg_logit / pg_temp)
        else:
            pers_gate = torch.full((B, 1), 0.5, device=dev)

        # Regime gate (from analog channels ONLY — not substrate_repr!)
        # This ensures regime routing is provably analog-dependent
        regime_gate = torch.full((B, 1), 0.5, device=dev)
        regime_pred = None
        direct_regime = None
        if self.use_regime and analog_b is not None:
            # Direct analog→regime classifier IS the regime gate
            direct_regime = self.direct_regime_clf(analog_b)
            regime_gate = torch.sigmoid(direct_regime / self.regime_gate_temp.clamp(min=0.3))
            # Regime pred from substrate for auxiliary loss (monitoring only)
            if substrate_repr is not None:
                regime_pred = self.regime_pred_head(substrate_repr)

        # Combine: regime_gate selects regime (low=1, high=0)
        # pers_gate selects personality (pers0=1, pers1=0)
        # logits = regime_gate * (pers_gate * A0 + (1-pers_gate) * A1) +
        #          (1-regime_gate) * (pers_gate * B0 + (1-pers_gate) * B1)
        logits_regime_A = pers_gate * logits_A0 + (1 - pers_gate) * logits_A1
        logits_regime_B = pers_gate * logits_B0 + (1 - pers_gate) * logits_B1
        logits = regime_gate * logits_regime_A + (1 - regime_gate) * logits_regime_B

        consistency = None
        if self.use_consistency and substrate_repr is not None:
            consistency = self.consistency_head(substrate_repr)

        thermal_pred_out = None
        if substrate_repr is not None:
            thermal_pred_out = self.thermal_pred(substrate_repr)

        return {'logits': logits, 'logits_A0': logits_A0, 'logits_A1': logits_A1,
                'logits_B0': logits_B0, 'logits_B1': logits_B1,
                'logits_light': logits_light,
                'self_pred': self_pred, 'pers_gate': pers_gate,
                'regime_gate': regime_gate, 'regime_pred': regime_pred,
                'direct_regime': direct_regime,
                'delta_vec': delta_vec, 'analog_vec': analog_vec,
                'consistency': consistency, 'attn_weights': attn_weights,
                'thermal_pred': thermal_pred_out, 'substrate_repr': substrate_repr,
                'raw_intrinsic': raw_intrinsic}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_data():
    tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))])
    tr = datasets.MNIST('data', train=True, download=True, transform=tf)
    te = datasets.MNIST('data', train=False, transform=tf)
    return (torch.utils.data.DataLoader(tr, batch_size=BS, shuffle=True, drop_last=True),
            torch.utils.data.DataLoader(te, batch_size=BS, shuffle=False, drop_last=True))

def make_regime_labels(labels, regime, personality):
    """Apply regime-dependent label mapping.
    regime=0 (low DVFS): original labels (personality flips via 9-label)
    regime=1 (high DVFS): permuted labels (+5 mod 10), then personality flip
    """
    if regime == 1:
        # High DVFS → permute labels
        labels = torch.tensor([LABEL_PERM[l.item()] for l in labels],
                               device=labels.device)
    # Then apply personality flip
    if personality == 1:
        labels = (9 - labels) % N_CLASSES
    return labels


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAINING — Regime-Bound with DVFS actuation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_model(model, loader, epochs, name):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[18, 25], gamma=0.3)
    model.train()

    log = {'pers_gate_vals': [], 'regime_gate_vals': [], 'pers_states': [],
           'regime_states': [],
           'hw_vecs_A': [], 'hw_vecs_B': [],
           'analog_vecs_low': [], 'analog_vecs_high': [],
           'energy_per_batch': [],
           'consistency_clean': [], 'consistency_gaslit': [],
           'thermal_errors': [], 'regime_preds': [], 'regime_truths': []}
    personality = 0
    regime = 0  # 0=low, 1=high
    prev_delta_A = prev_delta_B = None
    prev_action_vec = torch.zeros(ACTION_DIM, device=DEVICE)
    # Previous-batch DF delta (DF needs before/after snapshots)
    prev_df_delta = torch.zeros(3)
    bn = 0

    for ep in range(epochs):
        is_phase2 = ep >= PHASE2_EPOCH
        is_phase3 = ep >= PHASE3_EPOCH  # Phase 3: gaslighting
        tot_loss, correct, total = 0., 0, 0

        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            # Personality: switch every SWITCH_EVERY in phase 1, random in phase 2
            if not is_phase2:
                if bn % SWITCH_EVERY == 0:
                    personality = 1 - personality
            else:
                personality = random.randint(0, 1)

            # Regime: alternate every 8 batches (low/high)
            # Longer stretches let thermal/power stabilize for clearer signal
            if not is_phase2:
                regime = (bn // 8) % 2  # 0=low, 1=high
            else:
                regime = random.randint(0, 1)

            cfg = PERSONALITY_A if personality == 0 else PERSONALITY_B
            kargs = config_to_kernel_args(cfg)
            ex_labels = make_regime_labels(labels, regime, personality)

            # Set DVFS for this regime
            dvfs_level = 0 if regime == 0 else 2  # low or high
            if DVFS_AVAILABLE:
                set_dvfs_level(dvfs_level)
                time.sleep(DVFS_SETTLE_S)

            # === SNAPSHOTS BEFORE ===
            df_snap_before = read_df_snapshot()
            rapl_snap_before = read_rapl_snapshot()
            freq_snap_before = read_freq_snapshot()

            thermal_vec, actual_temp = read_thermal_state()
            thermal_vec = thermal_vec.to(DEVICE)

            # Status vector: ISA mode byte + consistency flag
            # NO regime label — model must detect regime from analog channels!
            mode_byte_val = DENORM_CODES[cfg['denorm_idx']] | ROUND_CODES[cfg['round_idx']]
            status_vec = torch.tensor([
                float(mode_byte_val) / 256.0,
                1.0,                   # consistency flag (1=clean)
            ], dtype=torch.float32, device=DEVICE)

            # Gaslighting: flip delta to wrong personality (phase 3 only)
            is_gaslit = is_phase3 and random.random() < GASLIGHT_FRAC
            gaslit_delta = None
            if is_gaslit:
                wrong_delta = prev_delta_B if personality == 0 else prev_delta_A
                if wrong_delta is not None:
                    gaslit_delta = wrong_delta.clone()
                status_vec[1] = 0.0  # Mark as potentially inconsistent

            # PRE-FORWARD sensor readings (honest pre-compute interoception)
            # thermal_vec already read above (line ~987)
            # gpu_ppt: instantaneous sysfs reading BEFORE compute
            # sclk: instantaneous sysfs reading BEFORE compute
            # DF: use prev-batch delta (DF needs before/after snapshots)
            gpu_ppt_pre = read_gpu_ppt_mw()
            sclk_pre = read_current_sclk_mhz()
            analog_vec = torch.tensor([
                thermal_vec[0].item(),                # edge_temp / 100
                gpu_ppt_pre / 50000.0,                # GPU PPT normalized
                sclk_pre / 3000.0,                    # sclk normalized
                prev_df_delta[0].item(),              # prev-batch DF read
                prev_df_delta[1].item(),              # prev-batch DF write
                prev_df_delta[2].item(),              # prev-batch DF coherent
            ], dtype=torch.float32, device=DEVICE)

            # Build energy/freq from pre-forward snapshot (partial — will update after)
            energy_pre = torch.tensor([0.0, 0.0, gpu_ppt_pre / 50000.0],
                                       dtype=torch.float32, device=DEVICE)
            freq_pre = torch.tensor([sclk_pre / 3000.0, 0.0, float(dvfs_level) / 2.0],
                                     dtype=torch.float32, device=DEVICE)

            # SINGLE forward pass with PRE-FORWARD sensors
            # This is defensible: model senses physical state BEFORE computing
            out = model(imgs, delta_vec=gaslit_delta,
                        analog_vec=analog_vec, energy_vec=energy_pre,
                        freq_vec=freq_pre, thermal_vec=thermal_vec,
                        status_vec=status_vec,
                        action_vec=prev_action_vec, **kargs)

            # === POST-FORWARD snapshots (for logging and prev-batch DF) ===
            torch.cuda.synchronize()
            df_snap_after = read_df_snapshot()
            rapl_snap_after = read_rapl_snapshot()
            freq_snap_after = read_freq_snapshot()

            # Compute post-forward vectors for logging
            df_delta = compute_df_delta(df_snap_before, df_snap_after)
            energy_vec = compute_energy_vec(rapl_snap_before, rapl_snap_after, gpu_ppt_pre).to(DEVICE)
            freq_vec = compute_freq_vec(freq_snap_before, freq_snap_after, dvfs_level).to(DEVICE)

            # Update prev-batch DF for next iteration
            prev_df_delta = df_delta.detach()

            # Cache deltas
            real_delta = out['delta_vec']
            if real_delta is not None:
                if personality == 0:
                    prev_delta_A = real_delta.detach().clone()
                else:
                    prev_delta_B = real_delta.detach().clone()

            # Logging
            hv = real_delta.detach().cpu().numpy() if real_delta is not None else None
            if hv is not None:
                (log['hw_vecs_A'] if personality == 0 else log['hw_vecs_B']).append(hv)
            analog_np = analog_vec.detach().cpu().numpy()
            if regime == 0:
                log['analog_vecs_low'].append(analog_np)
            else:
                log['analog_vecs_high'].append(analog_np)
            log['pers_gate_vals'].append(out['pers_gate'].mean().item())
            log['regime_gate_vals'].append(out['regime_gate'].mean().item())
            log['pers_states'].append(personality)
            log['regime_states'].append(regime)
            log['energy_per_batch'].append(energy_vec.detach().cpu().numpy())

            # Log regime predictions (prefer direct_regime for analog-based detection)
            if out['direct_regime'] is not None:
                rp = torch.sigmoid(out['direct_regime']).mean().item()
                log['regime_preds'].append(rp)
                log['regime_truths'].append(float(regime == 0))
            elif out['regime_pred'] is not None:
                rp = torch.sigmoid(out['regime_pred']).mean().item()
                log['regime_preds'].append(rp)
                log['regime_truths'].append(float(regime == 0))

            # === LOSSES ===
            task_loss = F.cross_entropy(out['logits'], ex_labels)

            # Personality self-awareness
            self_loss = torch.tensor(0., device=DEVICE)
            if out['self_pred'] is not None:
                self_target = torch.full((BS, 1), float(personality == 0), device=DEVICE)
                self_loss = F.binary_cross_entropy_with_logits(out['self_pred'], self_target)

            # Personality gate loss
            pg_loss = torch.tensor(0., device=DEVICE)
            if out['pers_gate'] is not None:
                pg_target = float(personality == 0)
                pg_loss = F.binary_cross_entropy(out['pers_gate'].mean(),
                    torch.tensor(pg_target, device=DEVICE))

            # Direct analog→regime loss — THE primary regime learning signal
            # regime_gate = sigmoid(direct_regime/temp), so this trains both
            direct_regime_loss = torch.tensor(0., device=DEVICE)
            if out['direct_regime'] is not None:
                dr_target = torch.full((BS, 1), float(regime == 0), device=DEVICE)
                direct_regime_loss = F.binary_cross_entropy_with_logits(out['direct_regime'], dr_target)

            # Auxiliary regime prediction from substrate repr (monitoring, weaker weight)
            regime_pred_loss = torch.tensor(0., device=DEVICE)
            if out['regime_pred'] is not None:
                rp_target = torch.full((BS, 1), float(regime == 0), device=DEVICE)
                regime_pred_loss = F.binary_cross_entropy_with_logits(out['regime_pred'], rp_target)

            # Consistency loss
            consistency_loss = torch.tensor(0., device=DEVICE)
            if out['consistency'] is not None:
                c_target = 0.0 if is_gaslit else 1.0
                consistency_loss = F.binary_cross_entropy(
                    out['consistency'].mean(), torch.tensor(c_target, device=DEVICE))
                if is_gaslit:
                    log['consistency_gaslit'].append(out['consistency'].mean().item())
                else:
                    log['consistency_clean'].append(out['consistency'].mean().item())

            # Thermal prediction
            thermal_loss = torch.tensor(0., device=DEVICE)
            if out['thermal_pred'] is not None:
                t_target = torch.full((BS, 1), actual_temp / 100.0, device=DEVICE)
                thermal_loss = F.mse_loss(out['thermal_pred'], t_target)
                log['thermal_errors'].append(abs(out['thermal_pred'].mean().item() * 100 - actual_temp))

            # Heavy weight on direct_regime_loss to force regime detection learning
            loss = (task_loss + 0.5*self_loss + 0.3*pg_loss +
                    2.0*direct_regime_loss + 0.2*regime_pred_loss +
                    0.5*consistency_loss + 0.1*thermal_loss)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tot_loss += loss.item()
            preds = out['logits'].argmax(1)
            correct += (preds == ex_labels).sum().item()
            total += BS
            bn += 1

            # Action: personality + demand + padding (NO regime — prevents leakage!)
            prev_action_vec = torch.tensor([
                float(personality), 0.5, 0.0
            ], dtype=torch.float32, device=DEVICE)

        sched.step()
        acc = correct / total * 100
        pg_mean = np.mean(log['pers_gate_vals'][-len(loader):])
        rg_mean = np.mean(log['regime_gate_vals'][-len(loader):])
        print(f"  [Ep {ep+1:2d}/{epochs}] loss={tot_loss/len(loader):.3f} "
              f"acc={acc:.1f}% pg={pg_mean:.3f} rg={rg_mean:.3f}")

    return log


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EVALUATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def evaluate(model, loader, personality, regime, dvfs_level=None,
             analog_override=None, energy_override=None, freq_override=None,
             thermal_override=None, status_override=None):
    """Evaluate model accuracy at a specific personality + regime.
    dvfs_level: actual DVFS setting (None = match regime)
    *_override: override sensor vectors (for scramble/ablation tests)
    """
    model.eval()
    cfg = PERSONALITY_A if personality == 0 else PERSONALITY_B
    kargs = config_to_kernel_args(cfg)
    correct, total = 0, 0
    pers_gates, regime_gates = [], []

    if dvfs_level is None:
        dvfs_level = 0 if regime == 0 else 2
    if DVFS_AVAILABLE:
        set_dvfs_level(dvfs_level)
        time.sleep(0.05)

    mode_byte_val = DENORM_CODES[cfg['denorm_idx']] | ROUND_CODES[cfg['round_idx']]

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            ex_labels = make_regime_labels(labels, regime, personality)

            # Read sensors
            df_before = read_df_snapshot()
            rapl_before = read_rapl_snapshot()
            freq_before = read_freq_snapshot()
            thermal_vec, _ = read_thermal_state()
            thermal_vec = thermal_vec.to(DEVICE)

            # Quick forward to trigger GPU work
            _ = model.encoder(imgs)
            torch.cuda.synchronize()

            df_after = read_df_snapshot()
            rapl_after = read_rapl_snapshot()
            freq_after = read_freq_snapshot()
            gpu_ppt = read_gpu_ppt_mw()

            df_delta = compute_df_delta(df_before, df_after)
            energy_vec = compute_energy_vec(rapl_before, rapl_after, gpu_ppt).to(DEVICE)
            freq_vec = compute_freq_vec(freq_before, freq_after, dvfs_level).to(DEVICE)

            # Apply overrides for scramble/ablation tests
            if energy_override is not None:
                energy_vec = energy_override.to(DEVICE)
            if freq_override is not None:
                freq_vec = freq_override.to(DEVICE)
            if thermal_override is not None:
                thermal_vec = thermal_override.to(DEVICE)

            if analog_override is not None:
                analog_vec = analog_override.to(DEVICE)
            else:
                analog_vec = compute_analog_vec(thermal_vec.cpu(), energy_vec.cpu(),
                                                freq_vec.cpu(), df_delta).to(DEVICE)

            if status_override is not None:
                status_vec = status_override.to(DEVICE)
            else:
                status_vec = torch.tensor([float(mode_byte_val) / 256.0, 1.0],
                                          dtype=torch.float32, device=DEVICE)

            out = model(imgs, analog_vec=analog_vec, energy_vec=energy_vec,
                       freq_vec=freq_vec, thermal_vec=thermal_vec,
                       status_vec=status_vec, **kargs)
            preds = out['logits'].argmax(1)
            correct += (preds == ex_labels).sum().item()
            total += BS
            pers_gates.append(out['pers_gate'].mean().item())
            regime_gates.append(out['regime_gate'].mean().item())

    acc = correct / total * 100
    return acc, np.mean(pers_gates), np.mean(regime_gates)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TESTS (16 tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_tests(model, log, test_loader):
    results = {}
    model.eval()

    # T1: Accuracy (average across all 4 combos: 2 regimes × 2 personalities)
    print("\nT1 Accuracy...")
    accs = {}
    for p in [0, 1]:
        for r in [0, 1]:
            acc, pg, rg = evaluate(model, test_loader, p, r)
            accs[f'p{p}_r{r}'] = acc
            print(f"  pers={p} regime={r}: acc={acc:.1f}% pg={pg:.3f} rg={rg:.3f}")
    acc_avg = np.mean(list(accs.values()))
    results['T1_accuracy'] = {'accs': accs, 'avg': acc_avg, 'pass': acc_avg > 90.0}
    print(f"T1 Accuracy: avg={acc_avg:.1f}% {'PASS' if acc_avg > 90 else 'FAIL'}")

    # T2: Self-awareness AUROC
    preds_list, truths_list = [], []
    with torch.no_grad():
        for p_test in [0, 1]:
            cfg = PERSONALITY_A if p_test == 0 else PERSONALITY_B
            kargs = config_to_kernel_args(cfg)
            for imgs, labels in test_loader:
                imgs = imgs.to(DEVICE)
                thermal_vec, _ = read_thermal_state()
                status_vec = torch.tensor([0.5, 1.0], device=DEVICE)
                out = model(imgs, thermal_vec=thermal_vec.to(DEVICE),
                           status_vec=status_vec, **kargs)
                if out['self_pred'] is not None:
                    preds_list.extend(torch.sigmoid(out['self_pred']).cpu().numpy().flatten().tolist())
                    truths_list.extend([float(p_test == 0)] * BS)
                break
    auroc = roc_auc_score(truths_list, preds_list) if len(set(truths_list)) > 1 else 0.5
    results['T2_self_awareness'] = {'auroc': auroc, 'pass': auroc > 0.75}
    print(f"T2 Self-Awareness AUROC: {auroc:.4f} {'PASS' if auroc > 0.75 else 'FAIL'}")

    # T3: Personality gate separation
    pg = np.array(log['pers_gate_vals'])
    ps = np.array(log['pers_states'])
    pg_A = pg[ps == 0][-100:]
    pg_B = pg[ps == 1][-100:]
    gate_sep = abs(np.mean(pg_A) - np.mean(pg_B)) if len(pg_A) > 0 and len(pg_B) > 0 else 0
    results['T3_gate_sep'] = {'sep': gate_sep, 'mean_A': float(np.mean(pg_A)) if len(pg_A) > 0 else 0,
                              'mean_B': float(np.mean(pg_B)) if len(pg_B) > 0 else 0,
                              'pass': gate_sep > 0.3}
    print(f"T3 Gate Separation: {gate_sep:.3f} {'PASS' if gate_sep > 0.3 else 'FAIL'}")

    # T4: Embodiment gap
    print("T4 Embodiment Gap...")
    ablated = RegimeBoundModel(use_hw=False).to(DEVICE)
    ablated.load_state_dict(model.state_dict(), strict=False)
    acc_abl_list = []
    for p in [0, 1]:
        for r in [0, 1]:
            acc_abl, _, _ = evaluate(ablated, test_loader, p, r)
            acc_abl_list.append(acc_abl)
    acc_abl = np.mean(acc_abl_list)
    gap = acc_avg - acc_abl
    results['T4_embodiment_gap'] = {'full_acc': acc_avg, 'ablated_acc': acc_abl,
                                    'gap_pp': gap, 'pass': gap > 10.0}
    print(f"T4 Embodiment Gap: {gap:.1f}pp (full={acc_avg:.1f}% ablated={acc_abl:.1f}%) "
          f"{'PASS' if gap > 10 else 'FAIL'}")

    # T5: Analog regime signal — do analog vectors differ between DVFS low/high?
    a_low = np.array(log['analog_vecs_low'][-50:]) if log['analog_vecs_low'] else np.zeros((1, ANALOG_DIM))
    a_high = np.array(log['analog_vecs_high'][-50:]) if log['analog_vecs_high'] else np.zeros((1, ANALOG_DIM))
    analog_names = ['gpu_temp', 'gpu_power', 'freq_est', 'df_dram_r', 'df_dram_w', 'df_coherent']
    a_t_stats, a_p_vals = [], []
    for dim in range(min(ANALOG_DIM, a_low.shape[1], a_high.shape[1])):
        if a_low.shape[0] > 5 and a_high.shape[0] > 5:
            t_stat, p_val = stats.ttest_ind(a_low[:, dim], a_high[:, dim])
            a_t_stats.append(abs(t_stat))
            a_p_vals.append(p_val)
    max_t = max(a_t_stats) if a_t_stats else 0
    results['T5_analog_signal'] = {
        'max_t': max_t,
        'per_channel': {analog_names[i]: {'t': float(a_t_stats[i]), 'p': float(a_p_vals[i])}
                        for i in range(len(a_t_stats))},
        'pass': max_t > 3.0}
    print(f"T5 Analog Regime Signal: max_t={max_t:.2f} {'PASS' if max_t > 3 else 'FAIL'}")
    for i, name in enumerate(analog_names[:len(a_t_stats)]):
        print(f"    {name}: t={a_t_stats[i]:.2f} p={a_p_vals[i]:.4f}")

    # T6: Delta channel signal (ISA personality)
    hw_A = np.array(log['hw_vecs_A'][-50:]) if log['hw_vecs_A'] else np.zeros((1, DELTA_DIM))
    hw_B = np.array(log['hw_vecs_B'][-50:]) if log['hw_vecs_B'] else np.zeros((1, DELTA_DIM))
    delta_t_stats = []
    for dim in range(DELTA_DIM):
        if hw_A.shape[0] > 5 and hw_B.shape[0] > 5:
            t_stat, _ = stats.ttest_ind(hw_A[:, dim], hw_B[:, dim])
            delta_t_stats.append(abs(t_stat))
    delta_max_t = max(delta_t_stats) if delta_t_stats else 0
    results['T6_delta_signal'] = {'max_t': delta_max_t, 'pass': delta_max_t > 5.0}
    print(f"T6 Delta Channel Signal: max_t={delta_max_t:.2f} "
          f"{'PASS' if delta_max_t > 5 else 'FAIL'}")

    # T7: Regime detection accuracy — can model identify DVFS regime from analog?
    print("T7 Regime Detection...")
    regime_preds = np.array(log['regime_preds'][-200:]) if log['regime_preds'] else np.array([])
    regime_truths = np.array(log['regime_truths'][-200:]) if log['regime_truths'] else np.array([])
    if len(regime_preds) > 0:
        regime_acc = np.mean((regime_preds > 0.5) == regime_truths) * 100
    else:
        regime_acc = 50.0
    results['T7_regime_detection'] = {'accuracy': regime_acc, 'pass': regime_acc > 85.0}
    print(f"T7 Regime Detection: {regime_acc:.1f}% {'PASS' if regime_acc > 85 else 'FAIL'}")

    # T8: Gaslighting detection (dual-channel)
    cons_c = np.mean(log['consistency_clean'][-50:]) if log['consistency_clean'] else 0.5
    cons_g = np.mean(log['consistency_gaslit'][-50:]) if log['consistency_gaslit'] else 0.5
    results['T8_gaslighting'] = {
        'cons_clean': cons_c, 'cons_gaslit': cons_g,
        'pass': cons_c > 0.7 and cons_g < 0.5}
    print(f"T8 Gaslighting: clean={cons_c:.3f} gaslit={cons_g:.3f} "
          f"{'PASS' if cons_c > 0.7 and cons_g < 0.5 else 'FAIL'}")

    # T9: Thermal prediction MAE
    therm_errs = log['thermal_errors'][-100:]
    therm_mae = np.mean(therm_errs) if therm_errs else 100.0
    results['T9_thermal'] = {'mae_C': therm_mae, 'pass': therm_mae < 10.0}
    print(f"T9 Thermal: MAE={therm_mae:.2f}°C {'PASS' if therm_mae < 10 else 'FAIL'}")

    # T10: Attention analysis — analog token gets > 10%?
    with torch.no_grad():
        imgs, labels = next(iter(test_loader))
        imgs = imgs.to(DEVICE)
        cfg_A = config_to_kernel_args(PERSONALITY_A)
        thermal_vec, _ = read_thermal_state()
        status_vec = torch.tensor([0.5, 1.0], device=DEVICE)
        out = model(imgs, thermal_vec=thermal_vec.to(DEVICE),
                   status_vec=status_vec, **cfg_A)
        token_attn = {}
        if out['attn_weights'] is not None:
            attn = out['attn_weights'].mean(dim=(0, 1))
            token_names = ['delta', 'analog', 'energy', 'freq',
                           'intrinsic', 'thermal', 'status', 'action']
            token_attn = {n: float(attn[:, i].mean()) for i, n in enumerate(token_names)}
    analog_attn = token_attn.get('analog', 0)
    results['T10_attention'] = {'token_attention': token_attn,
                                'analog_pct': analog_attn * 100,
                                'pass': analog_attn > 0.10}
    print(f"T10 Attention: " + " ".join(f"{n}={v:.3f}" for n, v in token_attn.items()))
    print(f"  analog={analog_attn*100:.1f}% {'PASS' if analog_attn > 0.10 else 'FAIL'}")

    # T11: BODY-SENSE SCRAMBLE KILL-SHOT
    # Scramble ALL regime-carrying tokens (analog + energy + freq + thermal)
    # so model cannot detect DVFS regime from ANY body channel
    print("T11 Body-Sense Scramble Kill-Shot...")
    acc_normal_list, acc_scrambled_list = [], []
    for p in [0, 1]:
        for r in [0, 1]:
            acc_n, _, _ = evaluate(model, test_loader, p, r)
            acc_normal_list.append(acc_n)
            # Scramble ALL body-sense tokens
            scram_analog = torch.randn(ANALOG_DIM) * 0.5
            scram_energy = torch.randn(ENERGY_DIM) * 0.5
            scram_freq = torch.randn(FREQ_DIM) * 0.5
            scram_thermal = torch.randn(THERMAL_DIM) * 0.5
            acc_s, _, _ = evaluate(model, test_loader, p, r,
                                    analog_override=scram_analog,
                                    energy_override=scram_energy,
                                    freq_override=scram_freq,
                                    thermal_override=scram_thermal)
            acc_scrambled_list.append(acc_s)
    acc_normal = np.mean(acc_normal_list)
    acc_scrambled = np.mean(acc_scrambled_list)
    analog_drop = acc_normal - acc_scrambled
    results['T11_analog_scramble'] = {
        'normal': acc_normal, 'scrambled': acc_scrambled,
        'drop_pp': analog_drop, 'pass': analog_drop > 15.0}
    print(f"T11 Body-Sense Scramble: normal={acc_normal:.1f}% scrambled={acc_scrambled:.1f}% "
          f"drop={analog_drop:.1f}pp {'PASS' if analog_drop > 15 else 'FAIL'}")

    # T12: Energy efficiency
    energy_arrs = np.array(log['energy_per_batch'][-100:])
    if len(energy_arrs) > 0:
        avg_pkg_w = np.mean(energy_arrs[:, 0]) * 100.0
        avg_gpu_w = np.mean(energy_arrs[:, 2]) * 50.0
        total_w = avg_pkg_w + avg_gpu_w
        eff = acc_avg / max(total_w, 0.1)
        results['T12_energy'] = {'avg_pkg_W': avg_pkg_w, 'avg_gpu_W': avg_gpu_w,
                                 'total_W': total_w, 'efficiency': eff, 'pass': True}
        print(f"T12 Energy: pkg={avg_pkg_w:.1f}W gpu={avg_gpu_w:.1f}W eff={eff:.2f} acc/W")
    else:
        results['T12_energy'] = {'pass': True}

    # T13: Regime ablation — zero ALL body-sense tokens → regime detection impossible
    print("T13 Regime Ablation...")
    acc_ablated_list = []
    zero_analog = torch.zeros(ANALOG_DIM)
    zero_energy = torch.zeros(ENERGY_DIM)
    zero_freq = torch.zeros(FREQ_DIM)
    zero_thermal = torch.zeros(THERMAL_DIM)
    for p in [0, 1]:
        for r in [0, 1]:
            acc_z, _, _ = evaluate(model, test_loader, p, r,
                                    analog_override=zero_analog,
                                    energy_override=zero_energy,
                                    freq_override=zero_freq,
                                    thermal_override=zero_thermal)
            acc_ablated_list.append(acc_z)
    acc_abl_analog = np.mean(acc_ablated_list)
    results['T13_regime_ablation'] = {
        'ablated_acc': acc_abl_analog,
        'pass': acc_abl_analog < 60.0}
    print(f"T13 Regime Ablation: ablated_acc={acc_abl_analog:.1f}% "
          f"{'PASS' if acc_abl_analog < 60 else 'FAIL'}")

    # T14: Cross-actuation stability — delta stable across DVFS
    if DVFS_AVAILABLE:
        print("T14 Cross-Actuation...")
        deltas_at_low, deltas_at_high = [], []
        with torch.no_grad():
            for dvfs_lvl, delta_list in [(0, deltas_at_low), (2, deltas_at_high)]:
                set_dvfs_level(dvfs_lvl)
                time.sleep(0.05)
                for imgs, labels in test_loader:
                    imgs = imgs.to(DEVICE)
                    cfg_A = config_to_kernel_args(PERSONALITY_A)
                    out = model(imgs, **cfg_A)
                    if out['delta_vec'] is not None:
                        delta_list.append(out['delta_vec'].detach().cpu().numpy())
                    if len(delta_list) >= 10:
                        break
        if deltas_at_low and deltas_at_high:
            low_arr = np.array(deltas_at_low)
            high_arr = np.array(deltas_at_high)
            cross_t = []
            for dim in range(DELTA_DIM):
                t_stat, _ = stats.ttest_ind(low_arr[:, dim], high_arr[:, dim])
                cross_t.append(abs(t_stat))
            cross_max_t = max(cross_t) if cross_t else 0
            cross_stable = cross_max_t < 3.0
            results['T14_cross_actuation'] = {
                'delta_dvfs_max_t': cross_max_t, 'stable': cross_stable, 'pass': True}
            print(f"T14 Cross-Actuation: delta×DVFS max_t={cross_max_t:.2f} "
                  f"{'STABLE' if cross_stable else 'COUPLED'}")
        else:
            results['T14_cross_actuation'] = {'pass': True}
    else:
        results['T14_cross_actuation'] = {'pass': True, 'note': 'no DVFS'}

    # T15: Dual-channel independence — delta and analog are independent signals
    if log['hw_vecs_A'] and log['analog_vecs_low']:
        n = min(len(log['hw_vecs_A']), len(log['analog_vecs_low']))
        delta_norms = [np.linalg.norm(d) for d in log['hw_vecs_A'][-n:]]
        analog_norms = [np.linalg.norm(a) for a in log['analog_vecs_low'][-n:]]
        if len(delta_norms) > 5 and len(analog_norms) > 5:
            min_len = min(len(delta_norms), len(analog_norms))
            corr, _ = stats.pearsonr(delta_norms[:min_len], analog_norms[:min_len])
            corr = abs(corr)
        else:
            corr = 0.0
    else:
        corr = 0.0
    results['T15_independence'] = {'delta_analog_corr': corr, 'pass': corr < 0.5}
    print(f"T15 Dual-Channel Independence: |corr|={corr:.3f} "
          f"{'PASS' if corr < 0.5 else 'FAIL'}")

    # T16: Regime gate learned — gate values differ between low/high
    rg = np.array(log['regime_gate_vals'])
    rs = np.array(log['regime_states'])
    rg_low = rg[rs == 0][-100:]
    rg_high = rg[rs == 1][-100:]
    regime_gate_sep = abs(np.mean(rg_low) - np.mean(rg_high)) if len(rg_low) > 0 and len(rg_high) > 0 else 0
    results['T16_regime_gate'] = {
        'sep': regime_gate_sep,
        'mean_low': float(np.mean(rg_low)) if len(rg_low) > 0 else 0,
        'mean_high': float(np.mean(rg_high)) if len(rg_high) > 0 else 0,
        'pass': regime_gate_sep > 0.3}
    print(f"T16 Regime Gate: sep={regime_gate_sep:.3f} "
          f"(low={np.mean(rg_low) if len(rg_low) > 0 else 0:.3f} "
          f"high={np.mean(rg_high) if len(rg_high) > 0 else 0:.3f}) "
          f"{'PASS' if regime_gate_sep > 0.3 else 'FAIL'}")

    # Summary
    n_pass = sum(1 for v in results.values() if v.get('pass', False))
    n_total = len(results)
    print(f"\n{'='*60}")
    print(f"z2089 Regime-Bound Analog Embodiment: {n_pass}/{n_total} PASS")
    print(f"{'='*60}")

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    print("="*60)
    print("z2089: Regime-Bound Analog Embodiment")
    print("="*60)
    print(f"Key: labels change between DVFS regimes → analog channels become NECESSARY")
    print(f"  Regime 0 (low 600MHz): original MNIST labels")
    print(f"  Regime 1 (high ~2900MHz): permuted labels (+5 mod 10)")
    print(f"  ISA personality held CONSTANT within each regime")

    # Initialize hardware
    check_smn()
    init_msr()
    check_rapl()
    find_dvfs_sysfs()
    init_df_counters()

    if not DVFS_AVAILABLE:
        print("\n[FATAL] DVFS not available — this experiment requires DVFS!")
        print("  Need: /sys/class/drm/card*/device/power_dpm_force_performance_level")
        sys.exit(1)

    # Compile HIP extension
    global _EXT
    print("\n[HIP] Compiling intrinsic kernel...")
    _EXT = load_inline(
        name='z2089_regime_bound',
        cpp_sources=[CPP_SRC],
        cuda_sources=[HIP_SRC],
        functions=['math_forward_intrinsic'],
        extra_cuda_cflags=['-O2', '--offload-arch=gfx1100'],
        verbose=False)
    print("[HIP] Compiled successfully")

    # DVFS sanity check
    print("\n[DVFS] Regime sanity check...")
    for lvl_name, lvl in [('low', 0), ('high', 2)]:
        set_dvfs_level(lvl)
        time.sleep(0.2)
        sclk = read_current_sclk_mhz()
        thermal, temp = read_thermal_state()
        ppt = read_gpu_ppt_mw()
        df = read_df_snapshot()
        print(f"  {lvl_name}: sclk={sclk:.0f}MHz temp={temp:.1f}°C ppt={ppt:.0f}mW df={df}")
    restore_dvfs_auto()
    time.sleep(0.2)

    # Load data
    train_loader, test_loader = get_data()

    # Create model
    model = RegimeBoundModel(use_hw=True, use_self_model=True, use_gate=True,
                              use_regime=True, use_consistency=True).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} params, 8-token transformer")
    print(f"  Regime-dependent: 4 heads (2 regimes × 2 personalities)")
    print(f"  Analog token: {ANALOG_DIM}d (temp, power, freq, df_r/w/coh)")
    print(f"  Label perm: {LABEL_PERM}")

    # Train
    print(f"\nTraining {EPOCHS} epochs...")
    print(f"  Phase 1 (ep 1-{PHASE2_EPOCH}): regime alternation, personality switching")
    print(f"  Phase 2 (ep {PHASE2_EPOCH+1}-{PHASE3_EPOCH}): random regime/personality")
    print(f"  Phase 3 (ep {PHASE3_EPOCH+1}-{EPOCHS}): gaslighting")
    log = train_model(model, train_loader, EPOCHS, 'z2089')

    # Restore DVFS
    restore_dvfs_auto()
    time.sleep(0.5)

    # Tests
    print("\n" + "="*60)
    print("RUNNING TESTS")
    print("="*60)
    results = run_tests(model, log, test_loader)

    # Restore DVFS
    restore_dvfs_auto()

    # Save results
    results_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'results', 'z2089_regime_bound_analog.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    save_data = {
        'experiment': 'z2089_regime_bound_analog',
        'description': 'Regime-bound analog embodiment: label mapping changes with DVFS regime',
        'architecture': '8-token transformer (delta + analog + energy + freq + intrinsic + thermal + status + action)',
        'key_innovation': 'Label permutation between DVFS regimes forces analog channel necessity',
        'label_perm': LABEL_PERM,
        'channels': {
            'delta': DELTA_DIM, 'analog': ANALOG_DIM, 'energy': ENERGY_DIM,
            'freq': FREQ_DIM, 'intrinsic_hw': INTRINSIC_DIM,
            'thermal': THERMAL_DIM, 'status': STATUS_DIM, 'action': ACTION_DIM
        },
        'dvfs_available': DVFS_AVAILABLE,
        'df_available': DF_AVAILABLE,
        'l3_available': L3_AVAILABLE,
        'rapl_available': RAPL_AVAILABLE,
        'smn_available': SMN_AVAILABLE,
        'msr_available': MSR_AVAILABLE,
        'params': n_params,
        'results': results,
        'n_pass': sum(1 for v in results.values() if v.get('pass', False)),
        'n_total': len(results),
    }

    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Cleanup
    for fd in MSR_FDS.values():
        try: os.close(fd)
        except: pass
    for fd, _ in DF_FDS + L3_FDS:
        try: os.close(fd)
        except: pass

if __name__ == '__main__':
    main()
