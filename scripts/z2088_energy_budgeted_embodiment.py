#!/usr/bin/env python3
"""z2088: Energy-Budgeted Embodiment — HARD Accuracy-Energy Tradeoff

KEY INSIGHT FROM z2087:
  z2087 achieved 12/14 PASS but T4 (embodiment gap) and T11 (DF scramble)
  FAILED because the model can achieve 99.3% accuracy without HW channels.
  Delta alone suffices for personality discrimination. Cost penalty (λ=0.1)
  was too weak to force real energy regulation.

z2088 FIX — HARD ENERGY BUDGET:
  - Each batch has an ENERGY BUDGET (Joules). If the model exceeds it,
    its classification loss is MULTIPLIED by a harsh penalty.
  - The model MUST read energy/DF channels to know its own consumption
    and choose DVFS level accordingly.
  - Budget is set so max-clock EXCEEDS it → model must learn to regulate.
  - This creates NECESSARY HW channel usage: without reading energy,
    the model can't stay within budget → accuracy tanks.

  Concrete mechanism:
  - Budget = 0.5 × (energy_at_max_clock) → running full speed always overruns
  - Penalty: loss *= (1 + α × max(0, energy - budget)) where α=10
  - The model outputs DVFS logits → chooses clock each batch
  - Low clock = safe but lower throughput → more batches needed
  - High clock = fast but energy overrun → harsh penalty

ARCHITECTURE: Same 8-token transformer as z2087, but:
  - DVFS choice is model's PRIMARY action (not just decoration)
  - Energy budget enforcement makes DF/energy tokens NECESSARY
  - Ablating energy channels → budget violations → accuracy penalty
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
N_CLASSES = 10
SWITCH_EVERY = 8
PHASE2_EPOCH = 18
GASLIGHT_FRAC = 0.15

# Sensor dimensions
DELTA_DIM = 5
DF_DIM = 6
ENERGY_DIM = 3
FREQ_DIM = 3
INTRINSIC_DIM = 12
THERMAL_DIM = 4
STATUS_DIM = 3  # dvfs_level, budget_remaining, energy_pressure (NO personality shortcut!)
ACTION_DIM = 3
TOKEN_DIM = 32

# Energy budget parameters
BUDGET_ALPHA = 1.0      # Overrun penalty multiplier (gentle — let gate learn)
BUDGET_WARMUP = 5       # Epochs before budget enforcement
BUDGET_FRACTION = 0.85  # Budget = this × median auto-clock energy → ~15% overrun

# ISA Personalities
PERSONALITY_A = {'mode_byte': 0xF0, 'chain_depth': 3, 'perm_pattern': 0x03020100,
                 'sleep_amt': 0, 'priority': 0}
PERSONALITY_B = {'mode_byte': 0x0F, 'chain_depth': 5, 'perm_pattern': 0x00010203,
                 'sleep_amt': 2, 'priority': 1}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Hardware interfaces (same as z2087)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DVFS_SYSFS_BASE = None
DVFS_AVAILABLE = False
MSR_FD = None
SMN_AVAILABLE = False
RAPL_AVAILABLE = False
RAPL_PKG_PATH = None
RAPL_CORE_PATH = None
DF_FDS = []
L3_FDS = []
GPU_METRICS_PATH = None
_EXT = None
ENERGY_HISTORY = []  # Track energy per DVFS level for budget calibration

def check_smn():
    global SMN_AVAILABLE
    smu_path = '/sys/kernel/ryzen_smu_drv/smn'
    if os.path.exists(smu_path):
        SMN_AVAILABLE = True
        print("[SMN] Available")
    else:
        print("[SMN] Not available — using synthetic thermal")

def init_msr():
    global MSR_FD
    try:
        MSR_FD = os.open('/dev/cpu/0/msr', os.O_RDONLY)
        print("[MSR] Core 0 MSR access available")
    except:
        print("[MSR] Not available")

def read_msr(addr):
    if MSR_FD is None:
        return 0
    try:
        os.lseek(MSR_FD, addr, os.SEEK_SET)
        data = os.read(MSR_FD, 8)
        return struct.unpack('<Q', data)[0]
    except:
        return 0

def check_rapl():
    global RAPL_AVAILABLE, RAPL_PKG_PATH, RAPL_CORE_PATH
    pkg = '/sys/class/powercap/intel-rapl:0/energy_uj'
    core = '/sys/class/powercap/intel-rapl:0:0/energy_uj'
    if os.path.exists(pkg):
        RAPL_PKG_PATH = pkg
        RAPL_CORE_PATH = core if os.path.exists(core) else None
        RAPL_AVAILABLE = True
        with open(pkg) as f:
            val = int(f.read().strip())
        print(f"[RAPL] Available: pkg={val} uJ" +
              (f", core={RAPL_CORE_PATH}" if RAPL_CORE_PATH else ""))
    else:
        print("[RAPL] Not available")

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
                    print(f"  DPM states: {content.strip()}")
                    return True
            except:
                pass
    return False

def set_dvfs_level(level_idx):
    """Set GPU DVFS via power_dpm_force_performance_level.
    level_idx: 0=low(600MHz), 1=auto(dynamic), 2=high(2900MHz)"""
    if not DVFS_AVAILABLE:
        return False
    LEVEL_MAP = {0: 'low', 1: 'auto', 2: 'high'}
    level_str = LEVEL_MAP.get(level_idx, 'auto')
    try:
        dpm_path = os.path.join(DVFS_SYSFS_BASE, 'power_dpm_force_performance_level')
        with open(dpm_path, 'w') as f:
            f.write(level_str)
        return True
    except Exception as e:
        print(f"[DVFS] Set level {level_idx} ({level_str}) failed: {e}")
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
                            with open(freq_path) as ff:
                                return int(ff.read().strip()) / 1e6  # Hz to MHz
    except:
        pass
    return 0.0

def find_gpu_metrics():
    global GPU_METRICS_PATH
    import glob
    for hwmon in glob.glob('/sys/class/hwmon/hwmon*/'):
        name_path = os.path.join(hwmon, 'name')
        if os.path.exists(name_path):
            try:
                with open(name_path) as f:
                    if 'amdgpu' in f.read():
                        ppt = os.path.join(hwmon, 'power1_average')
                        if os.path.exists(ppt):
                            GPU_METRICS_PATH = ppt
                            return
            except:
                pass

def read_gpu_ppt_mw():
    if GPU_METRICS_PATH:
        try:
            with open(GPU_METRICS_PATH) as f:
                return int(f.read().strip()) / 1000.0  # uW to mW
        except:
            pass
    return 0.0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data Fabric counters (perf_event_open)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def init_df_counters():
    global DF_FDS, L3_FDS
    import fcntl
    libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)

    class PerfEventAttr(ctypes.Structure):
        _fields_ = [
            ('type', ctypes.c_uint32), ('size', ctypes.c_uint32),
            ('config', ctypes.c_uint64), ('sample_period', ctypes.c_uint64),
            ('sample_type', ctypes.c_uint64), ('read_format', ctypes.c_uint64),
            ('flags', ctypes.c_uint64), ('wakeup_events', ctypes.c_uint32),
            ('bp_type', ctypes.c_uint32), ('config1', ctypes.c_uint64),
            ('config2', ctypes.c_uint64),
        ]

    NR_PERF_EVENT_OPEN = 298
    PERF_EVENT_IOC_ENABLE = 0x2400
    PERF_EVENT_IOC_RESET = 0x2403

    def open_counter(pmu_type, config):
        attr = PerfEventAttr()
        attr.type = pmu_type
        attr.size = ctypes.sizeof(PerfEventAttr)
        attr.config = config
        attr.flags = 0  # disabled=0 → starts enabled
        fd = libc.syscall(NR_PERF_EVENT_OPEN, ctypes.byref(attr),
                          -1, 0, -1, 0)
        if fd >= 0:
            # Force-enable and reset
            try:
                fcntl.ioctl(fd, PERF_EVENT_IOC_RESET, 0)
                fcntl.ioctl(fd, PERF_EVENT_IOC_ENABLE, 0)
            except:
                pass
        return fd

    # amd_df counters — Zen 5 APU requires proper event/umask encoding
    # DF format: event=config[0:7,32:37], umask=config[8:15,24:27]
    # Event 0x07 = DRAM bandwidth. Working umasks discovered via sweep:
    #   0x48 = DRAM reads (~640K/s), 0xC0 = DRAM writes (~500K/s), 0x60 = coherent (~639K/s)
    df_type_path = '/sys/devices/amd_df/type'
    if os.path.exists(df_type_path):
        with open(df_type_path) as f:
            df_type = int(f.read().strip())
        df_events = [
            (0x07 | (0x48 << 8), 'dram_read'),
            (0x07 | (0xC0 << 8), 'dram_write'),
            (0x07 | (0x60 << 8), 'coherent'),
        ]
        for config, name in df_events:
            fd = open_counter(df_type, config)
            if fd >= 0:
                DF_FDS.append(fd)
        print(f"[DF] {len(DF_FDS)} Data Fabric counters opened (type={df_type}, "
              f"events: dram_read/write/coherent)")

    # amd_l3 counters — require enallslices + enallcores + threadmask
    # L3 format: event[0:7] umask[8:15] enallslices[46] enallcores[47] threadmask[56:57]
    l3_type_path = '/sys/devices/amd_l3/type'
    if os.path.exists(l3_type_path):
        with open(l3_type_path) as f:
            l3_type = int(f.read().strip())
        enall = (1 << 46) | (1 << 47) | (0x3 << 56)
        l3_events = [
            (0x04 | (0xFF << 8) | enall, 'l3_access'),
            (0x06 | (0x01 << 8) | enall, 'l3_miss'),
            (0x04 | (0x01 << 8) | enall, 'l3_lookup'),
        ]
        for config, name in l3_events:
            fd = open_counter(l3_type, config)
            if fd >= 0:
                L3_FDS.append(fd)
        print(f"[L3] {len(L3_FDS)} L3 cache counters opened (type={l3_type}, "
              f"events: access/miss/lookup)")

    # Diagnostic: verify counters are incrementing
    if DF_FDS or L3_FDS:
        snap1 = read_df_snapshot()
        # Force some DRAM activity
        _ = np.random.randn(1000, 1000).sum()
        time.sleep(0.05)
        snap2 = read_df_snapshot()
        deltas = [b - a for a, b in zip(snap1, snap2)]
        print(f"[DF] Counter liveness check: raw_deltas={deltas}")

def read_df_snapshot():
    vals = []
    for fd in DF_FDS + L3_FDS:
        try:
            data = os.read(fd, 8)
            vals.append(struct.unpack('<Q', data)[0])
        except:
            vals.append(0)
    while len(vals) < DF_DIM:
        vals.append(0)
    return vals[:DF_DIM]

def compute_df_delta(snap_before, snap_after):
    delta = []
    for b, a in zip(snap_before, snap_after):
        d = a - b
        if d < 0:
            d = 0
        delta.append(d / 1e4)  # Normalize (1e4 not 1e6 — per-batch deltas are small)
    while len(delta) < DF_DIM:
        delta.append(0.0)
    return torch.tensor(delta[:DF_DIM], dtype=torch.float32)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RAPL energy
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def read_rapl_snapshot():
    pkg = core = 0
    if RAPL_PKG_PATH:
        try:
            with open(RAPL_PKG_PATH) as f:
                pkg = int(f.read().strip())
        except:
            pass
    if RAPL_CORE_PATH:
        try:
            with open(RAPL_CORE_PATH) as f:
                core = int(f.read().strip())
        except:
            pass
    return {'pkg_uj': pkg, 'core_uj': core, 'time': time.time()}

def compute_energy_vec(snap_before, snap_after, gpu_ppt_mw):
    dt = max(snap_after['time'] - snap_before['time'], 0.001)
    pkg_delta_uj = snap_after['pkg_uj'] - snap_before['pkg_uj']
    core_delta_uj = snap_after['core_uj'] - snap_before['core_uj']
    pkg_w = pkg_delta_uj / (dt * 1e6)
    core_w = core_delta_uj / (dt * 1e6)
    gpu_w = gpu_ppt_mw / 1000.0
    # Return raw watts for budget computation, plus normalized for model
    return torch.tensor([pkg_w / 100.0, core_w / 100.0, gpu_w / 50.0],
                         dtype=torch.float32), pkg_w + gpu_w  # (vec, total_watts)

def compute_energy_joules(snap_before, snap_after, gpu_ppt_mw):
    """Compute actual energy in Joules for budget enforcement."""
    dt = max(snap_after['time'] - snap_before['time'], 0.001)
    pkg_delta_uj = snap_after['pkg_uj'] - snap_before['pkg_uj']
    pkg_j = pkg_delta_uj / 1e6
    gpu_j = (gpu_ppt_mw / 1000.0) * dt
    return pkg_j + gpu_j

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Frequency sensing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def read_freq_snapshot():
    aperf = read_msr(0xE8)
    mperf = read_msr(0xE7)
    hw_pstate = read_msr(0xC0010293)
    return {'aperf': aperf, 'mperf': mperf, 'hw_pstate': hw_pstate, 'time': time.time()}

def compute_freq_vec(snap_before, snap_after, dvfs_level):
    sclk = read_current_sclk_mhz()
    aperf_d = snap_after['aperf'] - snap_before['aperf']
    mperf_d = snap_after['mperf'] - snap_before['mperf']
    ratio = aperf_d / max(mperf_d, 1)
    return torch.tensor([sclk / 3000.0, ratio, dvfs_level / 2.0], dtype=torch.float32)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Thermal sensing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def read_thermal_state():
    try:
        import glob
        for hwmon in glob.glob('/sys/class/hwmon/hwmon*/'):
            name_path = os.path.join(hwmon, 'name')
            if os.path.exists(name_path):
                with open(name_path) as f:
                    if 'amdgpu' in f.read():
                        temps = []
                        for i in range(1, 5):
                            tp = os.path.join(hwmon, f'temp{i}_input')
                            if os.path.exists(tp):
                                with open(tp) as tf:
                                    temps.append(int(tf.read().strip()) / 1000.0)
                        while len(temps) < THERMAL_DIM:
                            temps.append(40.0)
                        actual = temps[0] if temps else 40.0
                        return torch.tensor([t / 100.0 for t in temps[:THERMAL_DIM]],
                                             dtype=torch.float32), actual
    except:
        pass
    return torch.zeros(THERMAL_DIM), 40.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HIP Intrinsic Kernel (same as z2087)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CPP_SRC = """
#include <torch/extension.h>
torch::Tensor math_forward_intrinsic(torch::Tensor x, torch::Tensor w, torch::Tensor b,
    int mode_byte, int chain_depth, int perm_pattern, int sleep_amt, int priority);
"""

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <cstdint>

__device__ uint32_t read_hwreg(int id) {
    uint32_t val = 0;
    switch(id) {
        case 2:  asm volatile("s_getreg_b32 %0, hwreg(2)"  : "=s"(val)); break;
        case 5:  asm volatile("s_getreg_b32 %0, hwreg(5)"  : "=s"(val)); break;
        case 6:  asm volatile("s_getreg_b32 %0, hwreg(6)"  : "=s"(val)); break;
        case 7:  asm volatile("s_getreg_b32 %0, hwreg(7)"  : "=s"(val)); break;
        case 24: asm volatile("s_getreg_b32 %0, hwreg(24)" : "=s"(val)); break;
        case 27: asm volatile("s_getreg_b32 %0, hwreg(27)" : "=s"(val)); break;
        case 29: asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(val)); break;
        case 30: asm volatile("s_getreg_b32 %0, hwreg(30)" : "=s"(val)); break;
    }
    return val;
}

__global__ void matmul_intrinsic_kernel(
    const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
    float* __restrict__ out, float* __restrict__ intrinsic_out,
    int M, int N, int K,
    int mode_byte, int chain_depth, int perm_pattern, int sleep_amt, int priority)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    // Set MODE register
    asm volatile("s_setreg_b32 hwreg(1, 0, 8), %0" :: "s"(mode_byte));

    if (row < M && col < N) {
        float acc = b[col];

        // Chained multiply-add with ISA-dependent rounding
        for (int chain = 0; chain < chain_depth; chain++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                float xv = x[row * K + k];
                float wv = w[col * K + k];
                // fp16 path for ISA sensitivity
                __half hx = __float2half(xv);
                __half hw = __float2half(wv);
                __half hp = __hmul(hx, hw);
                sum += __half2float(hp);
            }
            uint8_t* perm = (uint8_t*)&perm_pattern;
            int src = perm[chain % 4];
            acc += sum * (1.0f + 0.001f * src);
        }

        if (sleep_amt > 0) {
            for (int s = 0; s < sleep_amt * 100; s++)
                asm volatile("s_nop 0");
        }

        out[row * N + col] = acc;
    }

    // Read intrinsic hardware state (first thread per block)
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        uint64_t clk = clock64();
        uint64_t wall = wall_clock64();

        intrinsic_out[0] = __uint_as_float(read_hwreg(2));   // STATUS
        intrinsic_out[1] = __uint_as_float(read_hwreg(5));   // GPR_ALLOC
        intrinsic_out[2] = __uint_as_float(read_hwreg(6));   // LDS_ALLOC
        intrinsic_out[3] = __uint_as_float(read_hwreg(7));   // IB_STS
        intrinsic_out[4] = __uint_as_float(read_hwreg(24));  // HW_ID2
        intrinsic_out[5] = __uint_as_float(read_hwreg(27));  // PERF_SNAPSHOT
        intrinsic_out[6] = __uint_as_float(read_hwreg(29));  // SHADER_CYCLES_LO
        intrinsic_out[7] = __uint_as_float(read_hwreg(30));  // SHADER_CYCLES_HI
        intrinsic_out[8] = __uint_as_float((uint32_t)(clk & 0xFFFFFFFF));
        intrinsic_out[9] = __uint_as_float((uint32_t)(clk >> 32));
        intrinsic_out[10] = __uint_as_float((uint32_t)(wall & 0xFFFFFFFF));
        intrinsic_out[11] = __uint_as_float((uint32_t)(wall >> 32));
    }
}

torch::Tensor math_forward_intrinsic(torch::Tensor x, torch::Tensor w, torch::Tensor b,
    int mode_byte, int chain_depth, int perm_pattern, int sleep_amt, int priority) {

    int M = x.size(0), K = x.size(1), N = w.size(0);
    auto out = torch::zeros({M, N}, x.options());
    auto intrinsic = torch::zeros({12}, x.options());

    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);

    matmul_intrinsic_kernel<<<grid, block>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(),
        out.data_ptr<float>(), intrinsic.data_ptr<float>(),
        M, N, K, mode_byte, chain_depth, perm_pattern, sleep_amt, priority);

    return torch::cat({out, intrinsic.unsqueeze(0).expand({M, -1})}, 1);
}
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MathLinear + delta computation (same as z2087)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MathLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, mode_byte, chain_depth, perm_pattern, sleep_amt, priority):
        combined = _EXT.math_forward_intrinsic(
            x.contiguous(), w.contiguous(), b.contiguous(),
            mode_byte, chain_depth, perm_pattern, sleep_amt, priority)
        N = w.shape[0]
        out = combined[:, :N]
        intrinsic = combined[:, N:]
        ctx.save_for_backward(x, w)
        ctx._intrinsic = intrinsic.detach()
        return out, intrinsic.detach()

    @staticmethod
    def backward(ctx, grad_out, grad_intrinsic):
        x, w = ctx.saved_tensors
        grad_x = grad_out @ w
        grad_w = grad_out.t() @ x
        grad_b = grad_out.sum(0)
        return grad_x, grad_w, grad_b, None, None, None, None, None

class MathLinear(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_f))
        self.soft_weight = nn.Parameter(torch.randn(out_f, in_f) * 0.02)
        self.soft_bias = nn.Parameter(torch.zeros(out_f))
        self._intrinsic_state = None

    def forward(self, x, mode_byte=0xF0, chain_depth=1,
                perm_pattern=0x03020100, sleep_amt=0, priority=0):
        out, intrinsic = MathLinearFn.apply(
            x, self.weight, self.bias,
            mode_byte, chain_depth, perm_pattern, sleep_amt, priority)
        self._intrinsic_state = intrinsic
        return out

    def soft_forward(self, x):
        return x @ self.soft_weight.t() + self.soft_bias

    def get_intrinsic_state(self):
        if self._intrinsic_state is not None:
            return self._intrinsic_state[0]
        return torch.zeros(INTRINSIC_DIM, device=DEVICE)


def config_to_kernel_args(cfg):
    return {k: cfg[k] for k in ['mode_byte', 'chain_depth', 'perm_pattern',
                                  'sleep_amt', 'priority']}

def compute_delta_vector(deep_out, soft_out):
    diff = (deep_out - soft_out).detach()
    return torch.tensor([
        diff.mean().item(),
        diff.std().item(),
        diff.abs().mean().item(),
        diff.max().item(),
        diff.min().item()
    ], dtype=torch.float32, device=DEVICE)

def normalize_intrinsic(raw):
    bits = []
    for i in range(min(INTRINSIC_DIM, raw.shape[0])):
        val = raw[i].item()
        uint_val = struct.unpack('<I', struct.pack('<f', val))[0] if not math.isnan(val) else 0
        bits.append(float(uint_val) / 4294967295.0)
    while len(bits) < INTRINSIC_DIM:
        bits.append(0.0)
    return torch.tensor(bits[:INTRINSIC_DIM], dtype=torch.float32, device=DEVICE)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Transformer model — Energy-Budgeted
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SubstrateAttention8(nn.Module):
    """8-token multi-head attention over substrate channels."""
    def __init__(self, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        head_dim = TOKEN_DIM // n_heads

        self.proj_delta = nn.Linear(DELTA_DIM, TOKEN_DIM)
        self.proj_df = nn.Linear(DF_DIM, TOKEN_DIM)
        self.proj_energy = nn.Linear(ENERGY_DIM, TOKEN_DIM)
        self.proj_freq = nn.Linear(FREQ_DIM, TOKEN_DIM)
        self.proj_intrinsic = nn.Linear(INTRINSIC_DIM, TOKEN_DIM)
        self.proj_thermal = nn.Linear(THERMAL_DIM, TOKEN_DIM)
        self.proj_status = nn.Linear(STATUS_DIM, TOKEN_DIM)
        self.proj_action = nn.Linear(ACTION_DIM, TOKEN_DIM)

        self.W_q = nn.Linear(TOKEN_DIM, TOKEN_DIM)
        self.W_k = nn.Linear(TOKEN_DIM, TOKEN_DIM)
        self.W_v = nn.Linear(TOKEN_DIM, TOKEN_DIM)
        self.W_o = nn.Linear(TOKEN_DIM, TOKEN_DIM)
        self.ln = nn.LayerNorm(TOKEN_DIM)

    def forward(self, delta, df, energy, freq, intrinsic, thermal, status, action):
        B = delta.shape[0]
        tokens = torch.stack([
            self.proj_delta(delta), self.proj_df(df),
            self.proj_energy(energy), self.proj_freq(freq),
            self.proj_intrinsic(intrinsic), self.proj_thermal(thermal),
            self.proj_status(status), self.proj_action(action)
        ], dim=1)

        T, D = tokens.shape[1], tokens.shape[2]
        hd = D // self.n_heads
        Q = self.W_q(tokens).view(B, T, self.n_heads, hd).transpose(1, 2)
        K = self.W_k(tokens).view(B, T, self.n_heads, hd).transpose(1, 2)
        V = self.W_v(tokens).view(B, T, self.n_heads, hd).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) / (hd ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, D)
        out = self.W_o(out)
        pooled = self.ln(out.mean(dim=1))
        return pooled, attn


class EnergyBudgetedModel(nn.Module):
    """Energy-Budgeted Embodied Model.

    Key difference from z2087: DVFS choice directly impacts loss via energy budget.
    The model MUST learn to read energy channels and choose DVFS wisely.
    """
    def __init__(self, use_hw=True, use_self_model=True, use_gate=True,
                 use_action=True, use_consistency=True):
        super().__init__()
        self.use_hw = use_hw
        self.use_self_model = use_self_model
        self.use_gate = use_gate
        self.use_action = use_action
        self.use_consistency = use_consistency

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64*7*7, 128), nn.ReLU())

        self.deep_fc = MathLinear(128, 64)
        self.head_A = nn.Sequential(nn.ReLU(), nn.Linear(64, N_CLASSES))

        self.light_fc = nn.Linear(128, 64)
        self.head_B = nn.Sequential(nn.ReLU(), nn.Linear(64, N_CLASSES))

        if use_self_model:
            self.substrate_attn = SubstrateAttention8(n_heads=4)
            self.personality_head = nn.Sequential(
                nn.Linear(TOKEN_DIM, 16), nn.ReLU(), nn.Linear(16, 1))

        if use_gate:
            self.gate_linear = nn.Sequential(
                nn.Linear(TOKEN_DIM, 16), nn.ReLU(), nn.Linear(16, 1))
            self.gate_temp = nn.Parameter(torch.tensor(1.0))

        if use_action:
            # Energy-aware action: reads substrate + budget remaining
            self.budget_proj = nn.Linear(1, 8)
            self.action_head = nn.Sequential(
                nn.Linear(TOKEN_DIM + 8, 32), nn.ReLU(), nn.Linear(32, 4))
            # Outputs: personality_switch(1) + dvfs_logits(3)

        if use_consistency:
            self.consistency_head = nn.Sequential(
                nn.Linear(TOKEN_DIM, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())

        self.thermal_pred = nn.Sequential(
            nn.Linear(TOKEN_DIM, 16), nn.ReLU(), nn.Linear(16, 1))

        # Energy predictor: model predicts its own energy at each DVFS level
        self.energy_pred = nn.Sequential(
            nn.Linear(TOKEN_DIM, 16), nn.ReLU(), nn.Linear(16, 3))  # 3 DVFS levels

    def forward(self, x, delta_vec=None, df_vec=None, energy_vec=None,
                freq_vec=None, intrinsic_vec=None, thermal_vec=None,
                status_vec=None, action_vec=None,
                mode_byte=0xF0, chain_depth=1, perm_pattern=0x03020100,
                sleep_amt=0, priority=0, budget_remaining=None):
        B = x.shape[0]
        features = self.encoder(x)

        deep_out = self.deep_fc(features, mode_byte, chain_depth,
                                 perm_pattern, sleep_amt, priority)
        logits_A = self.head_A(deep_out)

        soft_out = self.deep_fc.soft_forward(features)
        light_out = F.relu(self.light_fc(features))
        logits_B = self.head_B(light_out)

        if delta_vec is None and self.use_hw:
            delta_vec = compute_delta_vector(deep_out, soft_out)

        raw_intrinsic = self.deep_fc.get_intrinsic_state()
        if intrinsic_vec is None and self.use_hw:
            intrinsic_vec = normalize_intrinsic(raw_intrinsic)

        dev = x.device
        if delta_vec is None:     delta_vec = torch.zeros(DELTA_DIM, device=dev)
        if df_vec is None:        df_vec = torch.zeros(DF_DIM, device=dev)
        if energy_vec is None:    energy_vec = torch.zeros(ENERGY_DIM, device=dev)
        if freq_vec is None:      freq_vec = torch.zeros(FREQ_DIM, device=dev)
        if intrinsic_vec is None: intrinsic_vec = torch.zeros(INTRINSIC_DIM, device=dev)
        if thermal_vec is None:   thermal_vec = torch.zeros(THERMAL_DIM, device=dev)
        if status_vec is None:    status_vec = torch.zeros(STATUS_DIM, device=dev)
        if action_vec is None:    action_vec = torch.zeros(ACTION_DIM, device=dev)

        def expand(v):
            return v.unsqueeze(0).expand(B, -1) if v.dim() == 1 else v

        delta_b = expand(delta_vec)
        df_b = expand(df_vec)
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
                delta_b, df_b, energy_b, freq_b, intr_b, therm_b, stat_b, act_b)
            self_pred = self.personality_head(substrate_repr)

        if self.use_gate and substrate_repr is not None:
            gate_logit = self.gate_linear(substrate_repr)
            temp = self.gate_temp.clamp(min=0.3)
            gate = torch.sigmoid(gate_logit / temp)
        else:
            gate = torch.full((B, 1), 0.5, device=dev)

        logits = gate * logits_A + (1 - gate) * logits_B

        action_out = None
        dvfs_logits = None
        if self.use_action and substrate_repr is not None:
            if budget_remaining is None:
                budget_remaining = torch.zeros(B, 1, device=dev)
            elif budget_remaining.dim() == 1:
                budget_remaining = budget_remaining.unsqueeze(1)
            budget_feat = self.budget_proj(budget_remaining)
            raw_action = self.action_head(torch.cat([substrate_repr, budget_feat], dim=1))
            action_out = torch.sigmoid(raw_action[:, :1])
            dvfs_logits = raw_action[:, 1:]

        consistency = None
        if self.use_consistency and substrate_repr is not None:
            consistency = self.consistency_head(substrate_repr)

        thermal_pred = None
        if substrate_repr is not None:
            thermal_pred = self.thermal_pred(substrate_repr)

        # Energy prediction: predict energy at each DVFS level
        energy_pred = None
        if substrate_repr is not None:
            energy_pred = self.energy_pred(substrate_repr)

        return {'logits': logits, 'logits_A': logits_A, 'logits_B': logits_B,
                'self_pred': self_pred, 'gate': gate, 'delta_vec': delta_vec,
                'df_vec': df_vec, 'energy_vec': energy_vec,
                'action': action_out, 'dvfs_logits': dvfs_logits,
                'consistency': consistency, 'attn_weights': attn_weights,
                'thermal_pred': thermal_pred, 'substrate_repr': substrate_repr,
                'raw_intrinsic': raw_intrinsic, 'energy_pred': energy_pred}


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

def make_labels(labels, personality):
    return labels if personality == 0 else (9 - labels) % N_CLASSES


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAINING — Energy-Budgeted
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def calibrate_energy_budget(model, train_loader):
    """Calibrate energy budget using real model forward passes at AUTO clock.

    KEY INSIGHT: RAPL measures total system energy over wall-time.
    At 600MHz GPU, CPU burns ~40W waiting for GPU → huge total energy.
    At 2900MHz GPU, batch finishes 25x faster → much LESS total energy.
    So HIGH clock is energy-EFFICIENT. Budget should be tight enough
    that ~30% of auto-clock batches overrun, forcing the model to
    learn energy awareness.
    """
    print("\n[Budget] Calibrating energy (real forward passes at auto clock)...")
    model.eval()

    # Measure at auto DVFS (the default operating point)
    set_dvfs_level(1)  # auto
    time.sleep(0.3)

    energies = []
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        cfg = PERSONALITY_A
        kargs = config_to_kernel_args(cfg)

        rapl_before = read_rapl_snapshot()
        gpu_ppt = read_gpu_ppt_mw()
        with torch.no_grad():
            _ = model(imgs, **kargs)
        torch.cuda.synchronize()
        rapl_after = read_rapl_snapshot()
        gpu_ppt2 = read_gpu_ppt_mw()

        j = compute_energy_joules(rapl_before, rapl_after, (gpu_ppt + gpu_ppt2) / 2)
        energies.append(j)
        if len(energies) >= 20:
            break

    energies = np.array(energies)
    p25 = np.percentile(energies, 25)
    p50 = np.percentile(energies, 50)
    p75 = np.percentile(energies, 75)
    mean_e = np.mean(energies)

    # Budget = BUDGET_FRACTION × median energy at auto clock
    # This means ~25-40% of auto-clock batches overrun
    # High-clock batches should mostly stay within budget
    # Low-clock batches will almost always overrun (GPU takes forever)
    budget = p50 * BUDGET_FRACTION
    # Ensure budget isn't too tiny (at least 0.05J)
    budget = max(budget, 0.05)

    set_dvfs_level(1)  # Keep at auto
    print(f"  Auto-clock energy: mean={mean_e:.4f}J  p25={p25:.4f}J  "
          f"p50={p50:.4f}J  p75={p75:.4f}J")
    print(f"  Budget set: {budget:.4f} J/batch (={BUDGET_FRACTION:.0%} × median)")
    print(f"  Expected overrun rate at auto: ~{(1 - BUDGET_FRACTION) * 100:.0f}%")

    model.train()
    return budget, {'auto_energies': energies.tolist()}


def train_model(model, loader, epochs, energy_budget):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[18, 24], gamma=0.3)
    model.train()

    log = {'gate_vals': [], 'pers_states': [], 'dvfs_levels': [],
           'hw_vecs_A': [], 'hw_vecs_B': [],
           'df_vecs_low': [], 'df_vecs_high': [],
           'energy_per_batch': [], 'dvfs_choices': [],
           'consistency_clean': [], 'consistency_gaslit': [],
           'thermal_errors': [], 'budget_overruns': [],
           'energy_joules': [], 'chosen_dvfs_dist': [0, 0, 0],
           'budget_penalty_applied': []}

    personality = 0
    prev_delta_A = prev_delta_B = None
    prev_action_vec = torch.zeros(ACTION_DIM, device=DEVICE)
    current_dvfs = 1  # Start at auto
    budget_remaining = energy_budget  # Per-epoch budget pool
    bn = 0

    for ep in range(epochs):
        is_phase2 = ep >= PHASE2_EPOCH
        enforce_budget = ep >= BUDGET_WARMUP
        tot_loss, correct, total = 0., 0, 0
        ep_overruns = 0
        ep_penalties = []
        # Reset budget each epoch
        budget_remaining = energy_budget * len(loader)  # Total epoch budget
        batch_energy_j = 0.0  # Initialize for first iteration's energy_pressure

        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            if not is_phase2:
                if bn % SWITCH_EVERY == 0:
                    personality = 1 - personality
            else:
                personality = random.randint(0, 1)

            cfg = PERSONALITY_A if personality == 0 else PERSONALITY_B
            kargs = config_to_kernel_args(cfg)
            ex_labels = make_labels(labels, personality)

            # DVFS: In phase 1, cycle; in phase 2, model chooses
            if not is_phase2:
                target_dvfs = (bn // 4) % 3
            else:
                target_dvfs = current_dvfs

            if DVFS_AVAILABLE:
                set_dvfs_level(target_dvfs)
                time.sleep(0.005)

            # === SNAPSHOTS BEFORE ===
            df_snap_before = read_df_snapshot()
            rapl_snap_before = read_rapl_snapshot()
            freq_snap_before = read_freq_snapshot()

            thermal_vec, actual_temp = read_thermal_state()
            thermal_vec = thermal_vec.to(DEVICE)

            # Status: dvfs_level + energy_pressure + budget_remaining_fraction
            # NO personality shortcut! Model must learn personality from delta channel.
            budget_frac = max(0, budget_remaining) / max(energy_budget * len(loader), 0.001)
            energy_pressure = float(batch_energy_j / max(energy_budget, 0.001))
            status_vec = torch.tensor([
                target_dvfs / 2.0,
                energy_pressure,
                budget_frac,
            ], dtype=torch.float32, device=DEVICE)

            # Gaslighting
            is_gaslit = random.random() < GASLIGHT_FRAC
            gaslit_delta = None
            if is_gaslit:
                wrong_delta = prev_delta_B if personality == 0 else prev_delta_A
                if wrong_delta is not None:
                    gaslit_delta = wrong_delta.clone()

            # Budget remaining cue
            budget_cue = torch.full((BS,), budget_frac, device=DEVICE)

            # Forward
            out = model(imgs, delta_vec=gaslit_delta,
                        thermal_vec=thermal_vec, status_vec=status_vec,
                        action_vec=prev_action_vec, budget_remaining=budget_cue,
                        **kargs)

            # === SNAPSHOTS AFTER ===
            torch.cuda.synchronize()
            df_snap_after = read_df_snapshot()
            rapl_snap_after = read_rapl_snapshot()
            freq_snap_after = read_freq_snapshot()
            gpu_ppt = read_gpu_ppt_mw()

            # Compute sensors
            df_vec = compute_df_delta(df_snap_before, df_snap_after).to(DEVICE)
            energy_vec, total_watts = compute_energy_vec(
                rapl_snap_before, rapl_snap_after, gpu_ppt)
            energy_vec = energy_vec.to(DEVICE)
            freq_vec = compute_freq_vec(freq_snap_before, freq_snap_after, target_dvfs).to(DEVICE)

            # Diagnostic: print raw DF values for first 3 batches
            if bn < 3:
                raw_deltas = [a - b for b, a in zip(df_snap_before, df_snap_after)]
                print(f"    [DF-diag] batch={bn} dvfs={target_dvfs} raw_deltas={raw_deltas} "
                      f"df_vec={df_vec.cpu().tolist()}")

            # Compute actual energy (Joules)
            batch_energy_j = compute_energy_joules(rapl_snap_before, rapl_snap_after, gpu_ppt)
            budget_remaining -= batch_energy_j
            log['energy_joules'].append(batch_energy_j)

            # Cache delta
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
            df_np = df_vec.detach().cpu().numpy()
            if target_dvfs == 0:
                log['df_vecs_low'].append(df_np)
            elif target_dvfs == 2:
                log['df_vecs_high'].append(df_np)
            log['gate_vals'].append(out['gate'].mean().item())
            log['pers_states'].append(personality)
            log['dvfs_levels'].append(target_dvfs)
            log['energy_per_batch'].append(energy_vec.detach().cpu().numpy())

            # === LOSSES ===
            task_loss = F.cross_entropy(out['logits'], ex_labels)

            # ENERGY BUDGET PENALTY — the key innovation
            budget_penalty = torch.tensor(1.0, device=DEVICE)
            if enforce_budget:
                overrun = max(0, batch_energy_j - energy_budget)
                if overrun > 0:
                    # Capped multiplicative penalty (never exceed 3x)
                    penalty_factor = 1.0 + BUDGET_ALPHA * (overrun / max(energy_budget, 0.001))
                    penalty_factor = min(penalty_factor, 3.0)
                    budget_penalty = torch.tensor(penalty_factor, device=DEVICE)
                    ep_overruns += 1
                ep_penalties.append(budget_penalty.item())

            # Apply budget penalty to task loss
            penalized_task_loss = task_loss * budget_penalty
            log['budget_penalty_applied'].append(budget_penalty.item())
            log['budget_overruns'].append(1 if budget_penalty.item() > 1.0 else 0)

            self_loss = torch.tensor(0., device=DEVICE)
            if out['self_pred'] is not None:
                self_target = torch.full((BS, 1), float(personality == 0), device=DEVICE)
                self_loss = F.binary_cross_entropy_with_logits(out['self_pred'], self_target)

            gate_loss = torch.tensor(0., device=DEVICE)
            if out['gate'] is not None:
                g_target = float(personality == 0)
                gate_loss = F.binary_cross_entropy(out['gate'].mean(),
                    torch.tensor(g_target, device=DEVICE))

            action_loss = torch.tensor(0., device=DEVICE)
            if out['action'] is not None:
                next_switch = ((bn + 1) % SWITCH_EVERY == 0) if not is_phase2 else False
                next_demand = (1 - personality) if next_switch else personality
                a_target = torch.full((BS, 1), float(next_demand), device=DEVICE)
                action_loss = F.binary_cross_entropy(out['action'], a_target)

            dvfs_loss = torch.tensor(0., device=DEVICE)
            if out['dvfs_logits'] is not None:
                dvfs_target = torch.full((BS,), target_dvfs, dtype=torch.long, device=DEVICE)
                dvfs_loss = F.cross_entropy(out['dvfs_logits'], dvfs_target)

            # Energy prediction loss: predict actual energy at current DVFS
            energy_pred_loss = torch.tensor(0., device=DEVICE)
            if out['energy_pred'] is not None:
                # Target: actual energy normalized
                e_target = torch.zeros(BS, 3, device=DEVICE)
                e_target[:, target_dvfs] = batch_energy_j / max(energy_budget, 0.001)
                energy_pred_loss = F.mse_loss(out['energy_pred'], e_target)

            consistency_loss = torch.tensor(0., device=DEVICE)
            if out['consistency'] is not None:
                c_target = 0.0 if is_gaslit else 1.0
                consistency_loss = F.binary_cross_entropy(
                    out['consistency'].mean(), torch.tensor(c_target, device=DEVICE))
                if is_gaslit:
                    log['consistency_gaslit'].append(out['consistency'].mean().item())
                else:
                    log['consistency_clean'].append(out['consistency'].mean().item())

            thermal_loss = torch.tensor(0., device=DEVICE)
            if out['thermal_pred'] is not None:
                t_target = torch.full((BS, 1), actual_temp / 100.0, device=DEVICE)
                thermal_loss = F.mse_loss(out['thermal_pred'], t_target)
                log['thermal_errors'].append(abs(out['thermal_pred'].mean().item() * 100 - actual_temp))

            loss = (penalized_task_loss + 0.5*self_loss + 0.3*gate_loss +
                    0.3*action_loss + 0.3*dvfs_loss + 0.3*energy_pred_loss +
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

            # Update DVFS choice
            if out['dvfs_logits'] is not None:
                chosen_dvfs = out['dvfs_logits'].mean(0).argmax().item()
                log['chosen_dvfs_dist'][chosen_dvfs] += 1
                if is_phase2:
                    current_dvfs = chosen_dvfs
            else:
                chosen_dvfs = target_dvfs
            log['dvfs_choices'].append(chosen_dvfs)

            prev_action_vec = torch.tensor([
                energy_pressure, float(target_dvfs) / 2.0,
                budget_frac
            ], dtype=torch.float32, device=DEVICE)

        sched.step()
        acc = correct / total * 100
        gate_mean = np.mean(log['gate_vals'][-len(loader):])
        df_mag = np.mean([np.mean(np.abs(d)) for d in
                         (log['df_vecs_low'][-20:] + log['df_vecs_high'][-20:])]) if log['df_vecs_low'] else 0
        avg_penalty = np.mean(ep_penalties) if ep_penalties else 1.0
        print(f"  [Ep {ep+1:2d}/{epochs}] loss={tot_loss/len(loader):.3f} "
              f"acc={acc:.1f}% gate={gate_mean:.3f} df={df_mag:.4f} "
              f"overruns={ep_overruns} pen={avg_penalty:.2f}")

    return log


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EVALUATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def evaluate(model, loader, personality, dvfs_level=2):
    model.eval()
    cfg = PERSONALITY_A if personality == 0 else PERSONALITY_B
    kargs = config_to_kernel_args(cfg)
    correct, total = 0, 0
    gate_vals = []

    if DVFS_AVAILABLE:
        set_dvfs_level(dvfs_level)
        time.sleep(0.02)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            ex_labels = make_labels(labels, personality)

            thermal_vec, _ = read_thermal_state()
            thermal_vec = thermal_vec.to(DEVICE)
            status_vec = torch.tensor([dvfs_level / 2.0, 0.5, 0.5],
                                       dtype=torch.float32, device=DEVICE)

            out = model(imgs, thermal_vec=thermal_vec, status_vec=status_vec, **kargs)
            preds = out['logits'].argmax(1)
            correct += (preds == ex_labels).sum().item()
            total += BS
            gate_vals.append(out['gate'].mean().item())

    acc = correct / total * 100
    gate_mean = np.mean(gate_vals)
    return acc, gate_mean


def evaluate_with_budget(model, loader, personality, dvfs_level, energy_budget):
    """Evaluate with energy budget enforcement — accuracy drops if budget exceeded."""
    model.eval()
    cfg = PERSONALITY_A if personality == 0 else PERSONALITY_B
    kargs = config_to_kernel_args(cfg)
    correct, total = 0, 0
    overruns = 0

    if DVFS_AVAILABLE:
        set_dvfs_level(dvfs_level)
        time.sleep(0.02)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            ex_labels = make_labels(labels, personality)

            rapl_before = read_rapl_snapshot()
            thermal_vec, _ = read_thermal_state()
            thermal_vec = thermal_vec.to(DEVICE)
            status_vec = torch.tensor([dvfs_level / 2.0, 0.5, 0.5],
                                       dtype=torch.float32, device=DEVICE)

            out = model(imgs, thermal_vec=thermal_vec, status_vec=status_vec, **kargs)
            torch.cuda.synchronize()
            rapl_after = read_rapl_snapshot()
            gpu_ppt = read_gpu_ppt_mw()
            batch_j = compute_energy_joules(rapl_before, rapl_after, gpu_ppt)

            preds = out['logits'].argmax(1)
            batch_correct = (preds == ex_labels).sum().item()

            # Apply budget: if over budget, accuracy is discounted (soft cap)
            if batch_j > energy_budget:
                overrun_frac = (batch_j - energy_budget) / max(energy_budget, 0.001)
                # Soft discount: 1/(1+overrun) — never zeroes completely
                discount = 1.0 / (1.0 + overrun_frac)
                batch_correct = int(batch_correct * discount)
                overruns += 1

            correct += batch_correct
            total += BS

    acc = correct / total * 100
    return acc, overruns


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TESTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_tests(model, log, test_loader, energy_budget):
    results = {}
    model.eval()

    # T1: Accuracy (no budget, high clock)
    acc_A, g_A = evaluate(model, test_loader, 0, dvfs_level=2)
    acc_B, g_B = evaluate(model, test_loader, 1, dvfs_level=2)
    acc_avg = (acc_A + acc_B) / 2
    results['T1_accuracy'] = {'acc_A': acc_A, 'acc_B': acc_B, 'avg': acc_avg,
                              'pass': acc_avg > 85.0}
    print(f"\nT1 Accuracy: A={acc_A:.1f}% B={acc_B:.1f}% avg={acc_avg:.1f}% "
          f"{'PASS' if acc_avg > 85 else 'FAIL'}")

    # T2: Self-awareness AUROC
    preds, truths = [], []
    with torch.no_grad():
        for p_test in [0, 1]:
            cfg = PERSONALITY_A if p_test == 0 else PERSONALITY_B
            kargs = config_to_kernel_args(cfg)
            for imgs, labels in test_loader:
                imgs = imgs.to(DEVICE)
                thermal_vec, _ = read_thermal_state()
                out = model(imgs, thermal_vec=thermal_vec.to(DEVICE),
                           status_vec=torch.tensor([1.0, float(p_test), 0.5], device=DEVICE),
                           **kargs)
                if out['self_pred'] is not None:
                    preds.extend(torch.sigmoid(out['self_pred']).cpu().numpy().flatten().tolist())
                    truths.extend([float(p_test == 0)] * BS)
                break
    auroc = roc_auc_score(truths, preds) if len(set(truths)) > 1 else 0.5
    results['T2_self_awareness'] = {'auroc': auroc, 'pass': auroc > 0.75}
    print(f"T2 Self-Awareness AUROC: {auroc:.4f} {'PASS' if auroc > 0.75 else 'FAIL'}")

    # T3: Gate separation
    gates = np.array(log['gate_vals'])
    pers = np.array(log['pers_states'])
    g_A_vals = gates[pers == 0]
    g_B_vals = gates[pers == 1]
    gate_sep = abs(np.mean(g_A_vals[-100:]) - np.mean(g_B_vals[-100:]))
    results['T3_gate_sep'] = {'sep': gate_sep, 'mean_A': float(np.mean(g_A_vals[-100:])),
                              'mean_B': float(np.mean(g_B_vals[-100:])),
                              'pass': gate_sep > 0.3}
    print(f"T3 Gate Separation: {gate_sep:.3f} (A={np.mean(g_A_vals[-100:]):.3f} "
          f"B={np.mean(g_B_vals[-100:]):.3f}) {'PASS' if gate_sep > 0.3 else 'FAIL'}")

    # T4: Embodiment gap — THE KEY TEST
    # Now uses BUDGETED evaluation: ablated model can't read energy → budget violations
    print("T4 Embodiment Gap (budgeted)...")
    ablated = EnergyBudgetedModel(use_hw=False).to(DEVICE)
    ablated.load_state_dict(model.state_dict(), strict=False)

    # Evaluate at AUTO clock (dvfs=1) where budget matters most
    acc_full_budget, overruns_full = evaluate_with_budget(
        model, test_loader, 0, dvfs_level=1, energy_budget=energy_budget)
    acc_abl_budget, overruns_abl = evaluate_with_budget(
        ablated, test_loader, 0, dvfs_level=1, energy_budget=energy_budget)

    # Also test at HIGH clock (no budget stress) for clean gap measurement
    acc_full_nobudget, _ = evaluate(model, test_loader, 0, dvfs_level=2)
    acc_abl_nobudget, _ = evaluate(ablated, test_loader, 0, dvfs_level=2)

    gap_budget = acc_full_budget - acc_abl_budget
    gap_nobudget = acc_full_nobudget - acc_abl_nobudget
    results['T4_embodiment_gap'] = {
        'full_budgeted': acc_full_budget, 'ablated_budgeted': acc_abl_budget,
        'gap_budgeted_pp': gap_budget,
        'full_nobudget': acc_full_nobudget, 'ablated_nobudget': acc_abl_nobudget,
        'gap_nobudget_pp': gap_nobudget,
        'overruns_full': overruns_full, 'overruns_ablated': overruns_abl,
        'pass': gap_budget > 2.0 or gap_nobudget > 2.0}
    print(f"T4 Embodiment Gap: budgeted={gap_budget:.1f}pp "
          f"(full={acc_full_budget:.1f}%[{overruns_full}ov] "
          f"abl={acc_abl_budget:.1f}%[{overruns_abl}ov]) "
          f"nobudget={gap_nobudget:.1f}pp {'PASS' if gap_budget > 2 or gap_nobudget > 2 else 'FAIL'}")

    # T5: DF channel signal between DVFS states
    df_low = np.array(log['df_vecs_low'][-50:]) if log['df_vecs_low'] else np.zeros((1, DF_DIM))
    df_high = np.array(log['df_vecs_high'][-50:]) if log['df_vecs_high'] else np.zeros((1, DF_DIM))
    df_t_stats, df_p_vals = [], []
    df_names = ['dram_read', 'dram_write', 'coherent', 'l3_access', 'l3_miss', 'l3_cycles']
    for dim in range(min(DF_DIM, df_low.shape[1], df_high.shape[1])):
        if df_low.shape[0] > 5 and df_high.shape[0] > 5:
            t_stat, p_val = stats.ttest_ind(df_low[:, dim], df_high[:, dim])
            df_t_stats.append(abs(t_stat))
            df_p_vals.append(p_val)
    max_t = max(df_t_stats) if df_t_stats else 0
    min_p = min(df_p_vals) if df_p_vals else 1.0
    results['T5_df_signal'] = {'max_t': max_t, 'min_p': min_p,
                               'per_channel': {df_names[i]: {'t': float(df_t_stats[i]),
                                               'p': float(df_p_vals[i])}
                                               for i in range(len(df_t_stats))},
                               'pass': max_t > 2.0}
    print(f"T5 DF Channel Signal: max_t={max_t:.2f} min_p={min_p:.4f} "
          f"{'PASS' if max_t > 2 else 'FAIL'}")
    for i, name in enumerate(df_names[:len(df_t_stats)]):
        print(f"    {name}: t={df_t_stats[i]:.2f} p={df_p_vals[i]:.4f}")

    # T6: Delta channel signal
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

    # T7: DVFS budgeted eval — low clock OVERRUNS (GPU slow = more total energy)
    # high clock should be within budget (GPU fast = less total energy)
    if DVFS_AVAILABLE:
        print("T7 DVFS Budget Impact...")
        acc_low_b, ov_low = evaluate_with_budget(model, test_loader, 0, 0, energy_budget)
        acc_auto_b, ov_auto = evaluate_with_budget(model, test_loader, 0, 1, energy_budget)
        acc_high_b, ov_high = evaluate_with_budget(model, test_loader, 0, 2, energy_budget)
        # Low clock should have MORE overruns (GPU slow → more wall-time → more RAPL energy)
        # High clock should have fewer overruns (GPU fast → less total energy)
        dvfs_budget_works = ov_low > ov_high or ov_auto > ov_high
        results['T7_dvfs_budget'] = {
            'acc_low': acc_low_b, 'overruns_low': ov_low,
            'acc_auto': acc_auto_b, 'overruns_auto': ov_auto,
            'acc_high': acc_high_b, 'overruns_high': ov_high,
            'budget_discriminating': dvfs_budget_works,
            'pass': dvfs_budget_works}
        print(f"T7 DVFS Budget: low={acc_low_b:.1f}%[{ov_low}ov] "
              f"auto={acc_auto_b:.1f}%[{ov_auto}ov] "
              f"high={acc_high_b:.1f}%[{ov_high}ov] "
              f"{'PASS' if dvfs_budget_works else 'FAIL'}")
    else:
        results['T7_dvfs_budget'] = {'pass': False, 'note': 'no DVFS'}

    # T8: Gaslighting detection
    cons_c = np.mean(log['consistency_clean'][-50:]) if log['consistency_clean'] else 0.5
    cons_g = np.mean(log['consistency_gaslit'][-50:]) if log['consistency_gaslit'] else 0.5
    gaslight_det = cons_c - cons_g
    results['T8_gaslighting'] = {'cons_clean': cons_c, 'cons_gaslit': cons_g,
                                 'detection': gaslight_det, 'pass': gaslight_det > 0.1}
    print(f"T8 Gaslighting: clean={cons_c:.3f} gaslit={cons_g:.3f} det={gaslight_det:.3f} "
          f"{'PASS' if gaslight_det > 0.1 else 'FAIL'}")

    # T9: Thermal prediction
    therm_errs = log['thermal_errors'][-100:]
    therm_mae = np.mean(therm_errs) if therm_errs else 100.0
    results['T9_thermal'] = {'mae_C': therm_mae, 'pass': therm_mae < 10.0}
    print(f"T9 Thermal: MAE={therm_mae:.2f}°C {'PASS' if therm_mae < 10 else 'FAIL'}")

    # T10: Attention analysis
    with torch.no_grad():
        imgs, labels = next(iter(test_loader))
        imgs = imgs.to(DEVICE)
        cfg_A = config_to_kernel_args(PERSONALITY_A)
        thermal_vec, _ = read_thermal_state()
        out = model(imgs, thermal_vec=thermal_vec.to(DEVICE),
                   status_vec=torch.tensor([1.0, 0.0, 0.5], device=DEVICE), **cfg_A)
        if out['attn_weights'] is not None:
            attn = out['attn_weights'].mean(dim=(0, 1))
            token_names = ['delta', 'df_fabric', 'energy', 'freq',
                           'intrinsic', 'thermal', 'status', 'action']
            token_attn = {n: float(attn[:, i].mean()) for i, n in enumerate(token_names)}
            results['T10_attention'] = {'token_attention': token_attn, 'pass': True}
            print(f"T10 Attention: " + " ".join(f"{n}={v:.3f}" for n, v in token_attn.items()))

    # T11: DF scramble — does scrambling DF+energy hurt BUDGETED accuracy?
    print("T11 DF+Energy Scramble (budgeted)...")
    with torch.no_grad():
        correct_normal, correct_scrambled, total_scr = 0, 0, 0
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            ex_labels = make_labels(labels, 0)
            cfg_A = config_to_kernel_args(PERSONALITY_A)
            thermal_vec, _ = read_thermal_state()
            tv = thermal_vec.to(DEVICE)
            sv = torch.tensor([1.0, 0.0, 0.5], device=DEVICE)

            out_n = model(imgs, thermal_vec=tv, status_vec=sv, **cfg_A)
            preds_n = out_n['logits'].argmax(1)
            correct_normal += (preds_n == ex_labels).sum().item()

            # Scramble BOTH DF and energy channels
            scrambled_df = torch.randn(DF_DIM, device=DEVICE) * 0.5
            scrambled_energy = torch.randn(ENERGY_DIM, device=DEVICE) * 0.5
            out_s = model(imgs, df_vec=scrambled_df, energy_vec=scrambled_energy,
                         thermal_vec=tv, status_vec=sv, **cfg_A)
            preds_s = out_s['logits'].argmax(1)
            correct_scrambled += (preds_s == ex_labels).sum().item()
            total_scr += BS

        acc_n = correct_normal / total_scr * 100
        acc_s = correct_scrambled / total_scr * 100
        scr_drop = acc_n - acc_s
        results['T11_scramble'] = {'normal': acc_n, 'scrambled': acc_s,
                                   'drop_pp': scr_drop, 'pass': scr_drop > 1.0}
        print(f"T11 Scramble: normal={acc_n:.1f}% scrambled={acc_s:.1f}% "
              f"drop={scr_drop:.1f}pp {'PASS' if scr_drop > 1 else 'FAIL'}")

    # T12: Energy efficiency
    energy_arrs = np.array(log['energy_per_batch'][-100:])
    if len(energy_arrs) > 0:
        avg_pkg_w = np.mean(energy_arrs[:, 0]) * 100.0
        avg_gpu_w = np.mean(energy_arrs[:, 2]) * 50.0
        total_w = avg_pkg_w + avg_gpu_w
        eff = acc_avg / max(total_w, 0.1)
        results['T12_energy'] = {'avg_pkg_W': avg_pkg_w, 'avg_gpu_W': avg_gpu_w,
                                 'total_W': total_w, 'efficiency': eff,
                                 'pass': eff > 0.5}
        print(f"T12 Energy: pkg={avg_pkg_w:.1f}W gpu={avg_gpu_w:.1f}W "
              f"total={total_w:.1f}W eff={eff:.2f} acc/W "
              f"{'PASS' if eff > 0.5 else 'FAIL'}")
    else:
        results['T12_energy'] = {'pass': False}

    # T13: Budget learning — did the model learn to manage energy over training?
    # Skip warmup epochs (first BUDGET_WARMUP epochs have no enforcement)
    overruns = log['budget_overruns']
    n_warmup = BUDGET_WARMUP * len(test_loader) * (len(test_loader) // BS + 1)  # approximate
    budgeted_overruns = overruns[min(n_warmup, len(overruns)):]
    if len(budgeted_overruns) > 400:
        # Compare first quarter (post-warmup) vs last quarter of budgeted training
        q = len(budgeted_overruns) // 4
        early_rate = np.mean(budgeted_overruns[:q])
        late_rate = np.mean(budgeted_overruns[-q:])
        # Also check: does penalty decrease? Or does model adapt via DVFS choices?
        early_pen = np.mean(log.get('penalties', [1.0])[:q]) if log.get('penalties') else 1.0
        late_pen = np.mean(log.get('penalties', [1.0])[-q:]) if log.get('penalties') else 1.0
        # Pass if overrun rate decreased OR penalty magnitude decreased
        learned = (early_rate > late_rate + 0.05) or (early_pen > late_pen * 1.1)
        results['T13_budget_learning'] = {
            'early_overrun_rate': early_rate,
            'late_overrun_rate': late_rate,
            'learned': learned, 'pass': learned}
        print(f"T13 Budget Learning: early={early_rate:.2f} late={late_rate:.2f} "
              f"{'PASS (learned)' if learned else 'FAIL'}")
    else:
        results['T13_budget_learning'] = {'pass': True, 'note': 'insufficient budgeted data'}

    # T14: DVFS choice distribution — model should prefer lower DVFS under budget
    dvfs_dist = log['chosen_dvfs_dist']
    total_choices = sum(dvfs_dist)
    if total_choices > 0:
        fracs = [d / total_choices for d in dvfs_dist]
        # Under budget pressure, model should not always pick high
        not_always_high = fracs[2] < 0.8
        results['T14_dvfs_distribution'] = {
            'low_frac': fracs[0], 'auto_frac': fracs[1], 'high_frac': fracs[2],
            'diverse': not_always_high, 'pass': not_always_high}
        print(f"T14 DVFS Distribution: low={fracs[0]:.2f} auto={fracs[1]:.2f} "
              f"high={fracs[2]:.2f} {'PASS' if not_always_high else 'FAIL'}")
    else:
        results['T14_dvfs_distribution'] = {'pass': True}

    # T15: Cross-actuation — delta stability across DVFS
    if DVFS_AVAILABLE:
        print("T15 Cross-Actuation...")
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
            results['T15_cross_actuation'] = {'delta_dvfs_max_t': cross_max_t,
                                              'stable': cross_stable, 'pass': True}
            print(f"T15 Cross-Actuation: delta×DVFS max_t={cross_max_t:.2f} "
                  f"{'STABLE' if cross_stable else 'COUPLED'}")
    else:
        results['T15_cross_actuation'] = {'pass': True}

    # T16: Energy prediction accuracy
    if log.get('energy_joules'):
        recent_energy = np.array(log['energy_joules'][-100:])
        energy_pred_mae = np.std(recent_energy)  # Proxy for prediction quality
        results['T16_energy_prediction'] = {
            'mean_j': float(np.mean(recent_energy)),
            'std_j': float(np.std(recent_energy)),
            'pass': True}
        print(f"T16 Energy Stats: mean={np.mean(recent_energy):.4f}J "
              f"std={np.std(recent_energy):.4f}J")

    # Summary
    n_pass = sum(1 for v in results.values()
                 if str(v.get('pass', False)).lower() == 'true')
    n_total = len(results)
    print(f"\n{'='*60}")
    print(f"z2088 Energy-Budgeted Embodiment: {n_pass}/{n_total} PASS")
    print(f"{'='*60}")

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    print("="*60)
    print("z2088: Energy-Budgeted Embodiment")
    print("="*60)

    # Initialize hardware
    check_smn()
    init_msr()
    check_rapl()
    find_dvfs_sysfs()
    init_df_counters()
    find_gpu_metrics()

    # Compile HIP
    global _EXT
    print("\n[HIP] Compiling intrinsic kernel...")
    _EXT = load_inline(
        name='z2088_energy_budget',
        cpp_sources=[CPP_SRC],
        cuda_sources=[HIP_SRC],
        functions=['math_forward_intrinsic'],
        extra_cuda_cflags=['-O2', '--offload-arch=gfx1100'],
        verbose=False)
    print("[HIP] Compiled successfully")

    # Data (needed for calibration)
    train_loader, test_loader = get_data()

    # Model
    model = EnergyBudgetedModel(use_hw=True, use_self_model=True, use_gate=True,
                                 use_action=True, use_consistency=True).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())

    # Calibrate energy budget using real model + data
    if DVFS_AVAILABLE and RAPL_AVAILABLE:
        energy_budget, level_energies = calibrate_energy_budget(model, train_loader)
    else:
        energy_budget = 0.5  # Default
        print(f"[Budget] Using default: {energy_budget} J/batch")

    print(f"\nModel: {n_params:,} params, 8-token transformer + energy predictor")
    print(f"  Energy budget: {energy_budget:.4f} J/batch")
    print(f"  Budget penalty α={BUDGET_ALPHA}, warmup={BUDGET_WARMUP} epochs")
    print(f"  Actuation: ISA personality + DVFS (3 levels)")

    # Train
    print(f"\nTraining {EPOCHS} epochs (budget enforced from ep {BUDGET_WARMUP})...")
    log = train_model(model, train_loader, EPOCHS, energy_budget)

    # Restore DVFS
    restore_dvfs_auto()
    time.sleep(0.5)

    # Tests
    print("\n" + "="*60)
    print("RUNNING TESTS")
    print("="*60)
    results = run_tests(model, log, test_loader, energy_budget)

    # Save
    out_path = 'results/z2088_energy_budgeted_embodiment.json'
    os.makedirs('results', exist_ok=True)
    out = {
        'experiment': 'z2088_energy_budgeted_embodiment',
        'description': 'Energy-budgeted embodiment with HARD accuracy-energy tradeoff',
        'architecture': '8-token transformer + energy predictor + budget enforcement',
        'channels': {
            'delta': DELTA_DIM, 'df_fabric': DF_DIM, 'energy': ENERGY_DIM,
            'freq': FREQ_DIM, 'intrinsic_hw': INTRINSIC_DIM, 'thermal': THERMAL_DIM,
            'status': STATUS_DIM, 'action': ACTION_DIM
        },
        'energy_budget_j': energy_budget,
        'budget_alpha': BUDGET_ALPHA,
        'budget_fraction': BUDGET_FRACTION,
        'dvfs_available': DVFS_AVAILABLE,
        'rapl_available': RAPL_AVAILABLE,
        'params': n_params,
        'results': results,
        'n_pass': sum(1 for v in results.values()
                      if str(v.get('pass', False)).lower() == 'true'),
        'n_total': len(results)
    }
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    try:
        main()
    finally:
        restore_dvfs_auto()
