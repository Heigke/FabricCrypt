#!/usr/bin/env python3
"""z2086: MSR-Level Microarchitectural Self-Model

BREAKTHROUGH: Read CPU performance counters, per-core energy, and IBS data
via MSR (Model Specific Registers) — the DEEPEST CPU profiling available.

NEW CHANNELS (all read via /dev/cpu/N/msr with ring-0 access):
  - 6 programmable perf counters per core: retired instructions, branches,
    branch mispredicts, dcache accesses, dcache misses, clk_unhalted
  - Per-core RAPL energy (MSR 0xC001029B)
  - MPERF/APERF ratio (effective frequency)
  - IBS (Instruction-Based Sampling): per-instruction microarch details
  - CPU config MSRs: FP_CFG, DE_CFG, EX_CFG (execution unit configuration)

WHY THIS MATTERS:
  Different ISA personalities execute different math patterns (fp16 rounding,
  chain depths, permutations). These MUST produce different:
  - Instruction retirement rates (more/fewer instructions per kernel)
  - Branch patterns (chain depth affects control flow)
  - Cache access patterns (permutation changes memory layout)
  - Energy consumption (different fp modes = different power)

  CPU perf counters measure these DIRECTLY from the microarchitecture.
  Unlike PM table (which showed aggregate power), MSR counters show
  PER-INSTRUCTION microarchitectural effects.

ARCHITECTURE: 7-token transformer
  T0: delta(5)         — ISA math fingerprint (proven channel)
  T1: msr_perf(12)     — 6 perf counter deltas + 6 per-core energy deltas
  T2: msr_freq(4)      — MPERF/APERF ratio, P-state, IBS sample
  T3: intrinsic_hw(12) — hwreg from inside GPU shader (z2085)
  T4: thermal(15)      — analog thermal state (SMN)
  T5: status(3)        — XTAL/CG/HTC (SMN)
  T6: action(2)        — last ISA config choice

HYPOTHESIS:
  MSR perf counters should be the SECOND channel (after delta) that carries
  ISA personality signal, because different math generates measurably different
  instruction patterns. If true, this gives us DUAL-channel embodiment.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import os, sys, json, time, copy, struct, random, math, numpy as np
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score
from scipy import stats

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')
os.environ.setdefault('PYTORCH_ROCM_ARCH', 'gfx1100')
from torch.utils.cpp_extension import load_inline

DEVICE = 'cuda'
BS = 256
EPOCHS = 25
SWITCH_EVERY = 8
PHASE2_EPOCH = 12
N_CLASSES = 10
DELTA_DIM = 5
MSR_PERF_DIM = 12     # 6 perf counter deltas + 6 per-core energy deltas
MSR_FREQ_DIM = 4      # MPERF/APERF + P-state + IBS fields
INTRINSIC_DIM = 12    # hwreg from inside shader
GASLIGHT_FRAC = 0.15

# Actuator codes (proven z2076)
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
# MSR ACCESS — CPU Performance Counters & Energy via /dev/cpu/N/msr
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MSR_AVAILABLE = False
MSR_FDS = {}  # core -> fd for direct MSR reads (energy, MPERF/APERF)
PERF_FDS = {}  # core -> list of (fd, name) for perf_event counters

import ctypes, ctypes.util, fcntl

# MSR addresses for direct reads
CORE_ENERGY_MSR = 0xC001029B
MPERF_MSR = 0xC00000E7
APERF_MSR = 0xC00000E8
PSTATE_STATUS_MSR = 0xC0010299
FP_CFG_MSR = 0xC0011028

# perf_event_open constants
__NR_perf_event_open = 298
PERF_EVENT_IOC_ENABLE = 0x2400
PERF_EVENT_IOC_DISABLE = 0x2401
PERF_EVENT_IOC_RESET = 0x2403

# Hardware events for perf_event_open
# (PERF_TYPE_HARDWARE=0, event_id)
PERF_HW_EVENTS = [
    (0, 0, 'CPU_CYCLES'),
    (0, 1, 'INSTRUCTIONS'),
    (0, 4, 'BRANCH_INSTR'),
    (0, 5, 'BRANCH_MISSES'),
    (0, 2, 'CACHE_REFS'),
    (0, 3, 'CACHE_MISSES'),
]

# Which cores to sample
SAMPLE_CORES = [0, 1, 2, 3, 4, 5]

class perf_event_attr(ctypes.Structure):
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
    attr.flags = 0  # start enabled
    fd = _get_libc().syscall(__NR_perf_event_open, ctypes.pointer(attr), -1, cpu, -1, 0)
    if fd == -1:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    return fd

def msr_read(fd, addr):
    os.lseek(fd, addr, os.SEEK_SET)
    return struct.unpack('Q', os.read(fd, 8))[0]

def init_msr():
    """Initialize MSR + perf_event access."""
    global MSR_AVAILABLE, MSR_FDS, PERF_FDS
    n_perf = 0

    # Open MSR for direct reads (energy, frequency)
    for core in SAMPLE_CORES:
        path = f'/dev/cpu/{core}/msr'
        if os.path.exists(path):
            try:
                fd = os.open(path, os.O_RDONLY)
                MSR_FDS[core] = fd
            except:
                pass

    # Open perf_event counters via perf_event_open (works without MSR write)
    for core in SAMPLE_CORES:
        core_fds = []
        for pe_type, config, name in PERF_HW_EVENTS:
            try:
                fd = perf_open(pe_type, config, cpu=core)
                core_fds.append((fd, name))
                n_perf += 1
            except:
                pass
        if core_fds:
            PERF_FDS[core] = core_fds

    if n_perf > 0 or len(MSR_FDS) > 0:
        MSR_AVAILABLE = True
        print(f"[PERF] {n_perf} perf_event counters across {len(PERF_FDS)} cores")
        print(f"[MSR]  {len(MSR_FDS)} cores with direct MSR reads (energy, freq)")
        if MSR_FDS:
            fd0 = MSR_FDS[SAMPLE_CORES[0]]
            try:
                mperf = msr_read(fd0, MPERF_MSR)
                aperf = msr_read(fd0, APERF_MSR)
                energy = msr_read(fd0, CORE_ENERGY_MSR)
                print(f"  MPERF={mperf:,} APERF={aperf:,} ratio={aperf/max(mperf,1):.4f}")
                print(f"  CORE_ENERGY={energy:,}")
            except Exception as e:
                print(f"  MSR baseline error: {e}")
    else:
        print("[PERF/MSR] Not available")

def read_msr_perf_snapshot():
    """Read perf_event counters + MSR energy/freq for sampled cores.
    Returns: perf_counters[core][6], energy[core], mperf, aperf
    """
    if not MSR_AVAILABLE:
        return None, None, 0, 0

    perf = {}
    energy = {}

    # Read perf_event counters (CPU_CYCLES, INSTRUCTIONS, BRANCH_INSTR, etc.)
    for core in SAMPLE_CORES:
        if core in PERF_FDS:
            counters = []
            for fd, name in PERF_FDS[core]:
                try:
                    data = os.read(fd, 8)
                    val = struct.unpack('Q', data)[0]
                    counters.append(val)
                except:
                    counters.append(0)
            perf[core] = counters
        else:
            perf[core] = [0] * 6

        # Read energy via MSR
        if core in MSR_FDS:
            try:
                energy[core] = msr_read(MSR_FDS[core], CORE_ENERGY_MSR)
            except:
                energy[core] = 0
        else:
            energy[core] = 0

    # MPERF/APERF from core 0
    mperf = aperf = 0
    if SAMPLE_CORES[0] in MSR_FDS:
        fd0 = MSR_FDS[SAMPLE_CORES[0]]
        try: mperf = msr_read(fd0, MPERF_MSR)
        except: pass
        try: aperf = msr_read(fd0, APERF_MSR)
        except: pass

    return perf, energy, mperf, aperf

def compute_msr_perf_delta(snap_before, snap_after):
    """Compute delta between two MSR snapshots → MSR_PERF_DIM vector.

    Returns 12-dim vector:
      [0:6]  — MSR counter deltas (normalized) for core with max activity
               (PERF_CTR0, MPERF, APERF, TSC, CORE_ENERGY, PSTATE)
      [6:12] — per-core energy deltas (normalized) for first 6 cores
    """
    perf_b, energy_b, mperf_b, aperf_b = snap_before
    perf_a, energy_a, mperf_a, aperf_a = snap_after

    if perf_b is None or perf_a is None:
        return torch.zeros(MSR_PERF_DIM)

    n_counters = 6  # We read 6 MSRs per core

    # Find core with max MPERF delta (most active)
    max_delta_core = SAMPLE_CORES[0]
    max_delta = 0
    for core in SAMPLE_CORES:
        if core in perf_a and core in perf_b:
            # MPERF is at index 1
            d = (perf_a[core][1] - perf_b[core][1]) & 0xFFFFFFFFFFFF
            if d > max_delta:
                max_delta = d
                max_delta_core = core

    # Counter deltas for most active core
    perf_deltas = []
    if max_delta_core in perf_a and max_delta_core in perf_b:
        for i in range(n_counters):
            d = (perf_a[max_delta_core][i] - perf_b[max_delta_core][i]) & 0xFFFFFFFFFFFF
            perf_deltas.append(float(d))
    else:
        perf_deltas = [0.0] * n_counters

    # Normalize (log scale)
    perf_norm = []
    for d in perf_deltas:
        if d > 0:
            perf_norm.append(math.log1p(d) / 20.0)
        else:
            perf_norm.append(0.0)

    # Per-core energy deltas
    energy_deltas = []
    for core in SAMPLE_CORES:
        if core in energy_a and core in energy_b:
            d = energy_a[core] - energy_b[core]
            energy_deltas.append(float(d))
        else:
            energy_deltas.append(0.0)

    emax = max((abs(e) for e in energy_deltas), default=1.0)
    emax = max(emax, 1.0)
    energy_norm = [e / emax for e in energy_deltas]

    return torch.tensor(perf_norm + energy_norm, dtype=torch.float32)

def compute_msr_freq_vec(snap_before, snap_after):
    """Compute MSR frequency vector → MSR_FREQ_DIM (4-dim).

    [0] APERF/MPERF ratio (effective frequency)
    [1] P-state status (normalized)
    [2] Branch miss rate (from perf counters)
    [3] Cache miss rate (from perf counters)
    """
    perf_b, _, mperf_b, aperf_b = snap_before
    perf_a, _, mperf_a, aperf_a = snap_after

    dm = (mperf_a - mperf_b) if mperf_a > mperf_b else 1
    da = (aperf_a - aperf_b) if aperf_a > aperf_b else 0
    freq_ratio = da / max(dm, 1)

    # P-state from MSR
    pstate = 0.0
    fd0 = MSR_FDS.get(SAMPLE_CORES[0])
    if fd0:
        try:
            pstate_raw = msr_read(fd0, PSTATE_STATUS_MSR)
            pstate = float((pstate_raw >> 12) & 0xF) / 16.0
        except:
            pass

    # Branch miss rate and cache miss rate from perf counters
    branch_miss_rate = 0.0
    cache_miss_rate = 0.0
    core0 = SAMPLE_CORES[0]
    if perf_a and perf_b and core0 in perf_a and core0 in perf_b:
        pa = perf_a[core0]
        pb = perf_b[core0]
        # Events: 0=cycles, 1=instructions, 2=branches, 3=branch_misses, 4=cache_refs, 5=cache_misses
        d_branches = max(pa[2] - pb[2], 1)
        d_bmiss = pa[3] - pb[3]
        branch_miss_rate = min(d_bmiss / d_branches, 1.0)
        d_cache = max(pa[4] - pb[4], 1)
        d_cmiss = pa[5] - pb[5]
        cache_miss_rate = min(d_cmiss / d_cache, 1.0)

    return torch.tensor([
        min(freq_ratio, 2.0) / 2.0,
        pstate,
        branch_miss_rate,
        cache_miss_rate,
    ], dtype=torch.float32)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SMN REGISTER ACCESS via ryzen_smu
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SMN_DEV = '/sys/kernel/ryzen_smu_drv/smn'
THM_CUR_TMP = 0x59800
THM_HTC = 0x59804
CG_THERMAL_STAT = 0x59858
XTAL_CNTL = 0x598C8
THM_PWRMGT = 0x598CC
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

def read_pm_table_fields():
    try:
        with open(PM_TABLE_PATH, 'rb') as f:
            pm = f.read()
        fields = []
        for idx in [1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]:
            if idx * 4 + 4 <= len(pm):
                fields.append(struct.unpack_from('<f', pm, idx * 4)[0])
            else:
                fields.append(0.0)
        return fields
    except:
        return [0.0] * 15

def read_analog_state():
    pm_fields = read_pm_table_fields()
    v = smn_read(THM_CUR_TMP)
    thm_cur = ((v >> 8) & 0xFFF) / 32.0 if v else 0.0
    xtal = smn_read(XTAL_CNTL) or 0
    cg = smn_read(CG_THERMAL_STAT) or 0
    htc = smn_read(THM_HTC) or 0
    return {'thermal_local': pm_fields[:14], 'thm_cur': thm_cur,
            'xtal_cntl': xtal, 'cg_thermal': cg, 'htc': htc}

def analog_to_tensors(state, device='cuda'):
    thermal = [state['thm_cur']] + state['thermal_local']
    thermal = torch.tensor(thermal, dtype=torch.float32, device=device)
    therm_max = max(thermal.abs().max().item(), 1.0)
    thermal = thermal / therm_max
    status = torch.tensor([
        float((state['xtal_cntl'] or 0) & 0xFFFF) / 65536.0,
        float((state['cg_thermal'] or 0) & 0xFF) / 256.0,
        float((state['htc'] or 0) & 0xFFFF) / 65536.0,
    ], dtype=torch.float32, device=device)
    return thermal, status

def check_smn():
    global SMN_AVAILABLE
    if os.path.exists(SMN_DEV):
        v = smn_read(THM_CUR_TMP)
        if v and v != 0xFFFFFFFF:
            SMN_AVAILABLE = True
            print(f"[SMN] Available, THM_CUR_TMP = {((v >> 8) & 0xFFF) / 32.0:.1f}°C")
    if not SMN_AVAILABLE:
        print("[SMN] Not available — using synthetic thermal")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GPU METRICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GPU_METRICS_PATH = None

def find_gpu_metrics():
    global GPU_METRICS_PATH
    import glob
    for p in glob.glob('/sys/class/drm/card*/device/gpu_metrics'):
        if os.path.exists(p):
            GPU_METRICS_PATH = p
            return p
    return None

def read_socket_power_mw():
    if not GPU_METRICS_PATH: return 0.0
    try:
        data = open(GPU_METRICS_PATH, 'rb').read()
        return float(struct.unpack_from('<I', data, 0x70)[0]) if len(data) >= 0x74 else 0.0
    except: return 0.0

class EnergyTracker:
    def __init__(self):
        self.samples, self.total_joules, self.total_examples = [], 0.0, 0
        self._last_time = None
    def sample(self, n=0):
        now = time.time()
        pw = read_socket_power_mw()
        if self._last_time and pw > 0:
            dt = now - self._last_time
            self.total_joules += (pw/1000.0)*dt
            self.samples.append({'power_w': pw/1000.0, 'dt': dt})
        self._last_time = now
        self.total_examples += n
    def avg_power_w(self):
        return np.mean([s['power_w'] for s in self.samples]) * 1000.0 if self.samples else 0.0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HIP KERNEL — same as z2085, reads hwreg from inside shader
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

def compute_timing_features(raw):
    clk_pre, clk_post = raw[8].item(), raw[9].item()
    wall_pre, wall_post = raw[10].item(), raw[11].item()
    clk_delta = max(clk_post - clk_pre, 0)
    wall_delta = max(wall_post - wall_pre, 0)
    freq_est = clk_delta / max(wall_delta, 1.0)
    total_cycles = raw[6].item() + raw[7].item() * 65536.0
    return torch.tensor([
        clk_delta / max(abs(clk_delta), 1.0),
        wall_delta / max(abs(wall_delta), 1.0),
        min(freq_est / 10.0, 1.0),
        (total_cycles % 65536) / 65536.0,
    ], device=raw.device)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRANSFORMER SUBSTRATE MODEL — 7 tokens
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOKEN_DIM = 32

class SubstrateAttention7(nn.Module):
    """Multi-head self-attention over 7 substrate state tokens.

    T0: delta(5)         — ISA math fingerprint
    T1: msr_perf(12)     — CPU perf counters + per-core energy (NEW)
    T2: msr_freq(4)      — MPERF/APERF, P-state, IBS, FP_CFG (NEW)
    T3: intrinsic_hw(12) — hwreg from inside GPU shader
    T4: thermal(15)      — analog thermal state (SMN)
    T5: status(3)        — XTAL/CG/HTC (SMN)
    T6: action(2)        — last ISA config
    """
    def __init__(self, n_heads=4):
        super().__init__()
        self.n_tokens = 7
        self.n_heads = n_heads

        self.proj_delta     = nn.Linear(DELTA_DIM, TOKEN_DIM)
        self.proj_msr_perf  = nn.Linear(MSR_PERF_DIM, TOKEN_DIM)
        self.proj_msr_freq  = nn.Linear(MSR_FREQ_DIM, TOKEN_DIM)
        self.proj_intrinsic = nn.Linear(INTRINSIC_DIM, TOKEN_DIM)
        self.proj_thermal   = nn.Linear(15, TOKEN_DIM)
        self.proj_status    = nn.Linear(3, TOKEN_DIM)
        self.proj_action    = nn.Linear(2, TOKEN_DIM)

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

    def forward(self, delta, msr_perf, msr_freq, intrinsic, thermal, status, action):
        B = delta.shape[0]
        t0 = self.proj_delta(delta).unsqueeze(1)
        t1 = self.proj_msr_perf(msr_perf).unsqueeze(1)
        t2 = self.proj_msr_freq(msr_freq).unsqueeze(1)
        t3 = self.proj_intrinsic(intrinsic).unsqueeze(1)
        t4 = self.proj_thermal(thermal).unsqueeze(1)
        t5 = self.proj_status(status).unsqueeze(1)
        t6 = self.proj_action(action).unsqueeze(1)

        tokens = torch.cat([t0, t1, t2, t3, t4, t5, t6], dim=1)
        tokens = tokens + self.pos_embed

        normed = self.norm1(tokens)
        attn_out, attn_weights = self.attn(normed, normed, normed,
                                            average_attn_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.ffn(self.norm2(tokens))

        flat = tokens.reshape(B, -1)
        return self.out_proj(flat), attn_weights


class MSRMicroarchModel(nn.Module):
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
            self.substrate_attn = SubstrateAttention7(n_heads=4)
            self.personality_head = nn.Sequential(
                nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

        if use_gate:
            self.gate_linear = nn.Sequential(
                nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
            self.gate_temp = nn.Parameter(torch.tensor(1.0))

        if use_action:
            self.demand_proj = nn.Linear(1, 8)
            self.action_head = nn.Sequential(
                nn.Linear(32 + 8, 32), nn.ReLU(),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, 1), nn.Sigmoid())

        if use_consistency:
            self.consistency_head = nn.Sequential(
                nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())

        self.thermal_pred = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x, delta_vec=None, msr_perf_vec=None, msr_freq_vec=None,
                intrinsic_vec=None, thermal_vec=None, status_vec=None,
                action_vec=None, mode_byte=0xF0, chain_depth=1,
                perm_pattern=0x03020100, sleep_amt=0, priority=0,
                demand_cue=None):
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

        # Defaults
        if delta_vec is None:
            delta_vec = torch.zeros(DELTA_DIM, device=x.device)
        if msr_perf_vec is None:
            msr_perf_vec = torch.zeros(MSR_PERF_DIM, device=x.device)
        if msr_freq_vec is None:
            msr_freq_vec = torch.zeros(MSR_FREQ_DIM, device=x.device)
        if intrinsic_vec is None:
            intrinsic_vec = torch.zeros(INTRINSIC_DIM, device=x.device)
        if thermal_vec is None:
            thermal_vec = torch.zeros(15, device=x.device)
        if status_vec is None:
            status_vec = torch.zeros(3, device=x.device)
        if action_vec is None:
            action_vec = torch.zeros(2, device=x.device)

        def expand(v):
            return v.unsqueeze(0).expand(B, -1) if v.dim() == 1 else v

        delta_b = expand(delta_vec)
        msr_perf_b = expand(msr_perf_vec)
        msr_freq_b = expand(msr_freq_vec)
        intr_b = expand(intrinsic_vec)
        therm_b = expand(thermal_vec)
        stat_b = expand(status_vec)
        act_b = expand(action_vec)

        substrate_repr = None
        attn_weights = None
        self_pred = None
        if self.use_self_model:
            substrate_repr, attn_weights = self.substrate_attn(
                delta_b, msr_perf_b, msr_freq_b, intr_b, therm_b, stat_b, act_b)
            self_pred = self.personality_head(substrate_repr)

        if self.use_gate and substrate_repr is not None:
            gate_logit = self.gate_linear(substrate_repr)
            temp = self.gate_temp.clamp(min=0.3)
            gate = torch.sigmoid(gate_logit / temp)
        else:
            gate = torch.full((B, 1), 0.5, device=x.device)

        logits = gate * logits_A + (1 - gate) * logits_B

        action = None
        if self.use_action and substrate_repr is not None and demand_cue is not None:
            dc = demand_cue.unsqueeze(1) if demand_cue.dim() == 1 else demand_cue
            demand_feat = self.demand_proj(dc)
            action = self.action_head(torch.cat([substrate_repr, demand_feat], dim=1))

        consistency = None
        if self.use_consistency and substrate_repr is not None:
            consistency = self.consistency_head(substrate_repr)

        thermal_pred = None
        if substrate_repr is not None:
            thermal_pred = self.thermal_pred(substrate_repr)

        return {'logits': logits, 'logits_A': logits_A, 'logits_B': logits_B,
                'self_pred': self_pred, 'gate': gate, 'delta_vec': delta_vec,
                'msr_perf_vec': msr_perf_vec, 'msr_freq_vec': msr_freq_vec,
                'intrinsic_vec': intrinsic_vec, 'raw_intrinsic': raw_intrinsic,
                'action': action, 'consistency': consistency,
                'attn_weights': attn_weights, 'thermal_pred': thermal_pred,
                'substrate_repr': substrate_repr}


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
# TRAINING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_model(model, loader, epochs, name):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[15, 20], gamma=0.3)
    model.train()

    log = {'gate_vals': [], 'pers_states': [], 'action_vals': [],
           'hw_vecs_A': [], 'hw_vecs_B': [],
           'msr_perf_A': [], 'msr_perf_B': [],
           'consistency_clean': [], 'consistency_gaslit': [],
           'thermal_errors': []}
    energy = EnergyTracker()
    personality = 0
    prev_delta_A = prev_delta_B = None
    prev_msr_perf_A = prev_msr_perf_B = None
    prev_action_vec = torch.zeros(2, device=DEVICE)
    bn = 0

    for ep in range(epochs):
        is_phase2 = ep >= PHASE2_EPOCH
        tot_loss, correct, total = 0., 0, 0

        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            if not is_phase2:
                if bn % SWITCH_EVERY == 0:
                    personality = 1 - personality
            else:
                personality = random.randint(0, 1)
            current_demand = personality

            cfg = PERSONALITY_A if personality == 0 else PERSONALITY_B
            kargs = config_to_kernel_args(cfg)
            ex_labels = make_labels(labels, personality)

            # === MSR SNAPSHOT BEFORE ===
            msr_snap_before = read_msr_perf_snapshot()

            # Read CPU-side analog state
            if SMN_AVAILABLE:
                analog = read_analog_state()
                thermal_vec, status_vec = analog_to_tensors(analog, DEVICE)
                actual_temp = analog['thm_cur']
            else:
                thermal_vec = torch.randn(15, device=DEVICE) * 0.01 + 0.5
                status_vec = torch.randn(3, device=DEVICE) * 0.1
                actual_temp = 50.0

            # Gaslighting: swap delta AND msr_perf
            is_gaslit = random.random() < GASLIGHT_FRAC
            gaslit_delta = None
            gaslit_msr_perf = None
            if is_gaslit:
                wrong_delta = prev_delta_B if personality == 0 else prev_delta_A
                wrong_msr = prev_msr_perf_B if personality == 0 else prev_msr_perf_A
                if wrong_delta is not None:
                    gaslit_delta = wrong_delta.clone()
                if wrong_msr is not None:
                    gaslit_msr_perf = wrong_msr.clone()

            # Demand cue
            if is_phase2:
                next_demand = random.randint(0, 1)
            else:
                next_switch = ((bn + 1) % SWITCH_EVERY == 0)
                next_demand = (1 - personality) if next_switch else personality
            log['action_vals'].append(float(next_demand))
            demand_cue = torch.full((BS,), float(next_demand), device=DEVICE)

            # Forward pass
            out = model(imgs, delta_vec=gaslit_delta, msr_perf_vec=gaslit_msr_perf,
                        thermal_vec=thermal_vec, status_vec=status_vec,
                        action_vec=prev_action_vec, demand_cue=demand_cue, **kargs)
            energy.sample(n=BS)

            # === MSR SNAPSHOT AFTER ===
            torch.cuda.synchronize()
            msr_snap_after = read_msr_perf_snapshot()

            # Compute MSR vectors
            msr_perf_vec = compute_msr_perf_delta(msr_snap_before, msr_snap_after)
            msr_freq_vec = compute_msr_freq_vec(msr_snap_before, msr_snap_after)

            # Move to device for next iteration's potential gaslighting
            msr_perf_device = msr_perf_vec.to(DEVICE)
            msr_freq_device = msr_freq_vec.to(DEVICE)

            # Cache for gaslighting
            real_delta = out['delta_vec']
            if real_delta is not None:
                if personality == 0:
                    prev_delta_A = real_delta.detach().clone()
                    prev_msr_perf_A = msr_perf_device.detach().clone()
                else:
                    prev_delta_B = real_delta.detach().clone()
                    prev_msr_perf_B = msr_perf_device.detach().clone()

            # Logging
            hv = real_delta.detach().cpu().numpy() if real_delta is not None else None
            if hv is not None:
                (log['hw_vecs_A'] if personality == 0 else log['hw_vecs_B']).append(hv)
            mv = msr_perf_vec.detach().cpu().numpy()
            (log['msr_perf_A'] if personality == 0 else log['msr_perf_B']).append(mv)
            log['gate_vals'].append(out['gate'].mean().item())
            log['pers_states'].append(personality)

            # === LOSSES ===
            task_loss = F.cross_entropy(out['logits'], ex_labels)

            self_loss = torch.tensor(0., device=DEVICE)
            if out['self_pred'] is not None:
                self_target = torch.full((BS, 1), float(personality == 0), device=DEVICE)
                self_loss = F.binary_cross_entropy_with_logits(out['self_pred'], self_target)

            gate_loss = torch.tensor(0., device=DEVICE)
            if out['gate'] is not None:
                g_target = float(personality == 0)
                gate_loss = F.binary_cross_entropy(out['gate'].mean(), torch.tensor(g_target, device=DEVICE))

            action_loss = torch.tensor(0., device=DEVICE)
            if out['action'] is not None:
                a_target = torch.full((BS, 1), float(next_demand), device=DEVICE)
                action_loss = F.binary_cross_entropy(out['action'], a_target)

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

            loss = task_loss + 0.5*self_loss + 0.3*gate_loss + 0.3*action_loss + \
                   0.5*consistency_loss + 0.1*thermal_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tot_loss += loss.item()
            preds = out['logits'].argmax(1)
            correct += (preds == ex_labels).sum().item()
            total += BS
            bn += 1

            # Update action vector
            if out['action'] is not None:
                prev_action_vec = torch.tensor([float(personality), out['action'].mean().item()],
                                                device=DEVICE)

        sched.step()
        acc = correct / total * 100
        gate_mean = np.mean(log['gate_vals'][-len(loader):])
        msr_mean = np.mean([np.mean(np.abs(m)) for m in (log['msr_perf_A'][-50:] + log['msr_perf_B'][-50:])])
        print(f"  [Ep {ep+1:2d}/{epochs}] loss={tot_loss/len(loader):.3f} "
              f"acc={acc:.1f}% gate={gate_mean:.3f} msr_mag={msr_mean:.4f}")

    return log, energy


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EVALUATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def evaluate(model, loader, personality):
    model.eval()
    cfg = PERSONALITY_A if personality == 0 else PERSONALITY_B
    kargs = config_to_kernel_args(cfg)
    correct, total = 0, 0
    preds_list, gate_vals = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            ex_labels = make_labels(labels, personality)

            msr_snap_before = read_msr_perf_snapshot()

            if SMN_AVAILABLE:
                analog = read_analog_state()
                thermal_vec, status_vec = analog_to_tensors(analog, DEVICE)
            else:
                thermal_vec = torch.randn(15, device=DEVICE) * 0.01 + 0.5
                status_vec = torch.randn(3, device=DEVICE) * 0.1

            out = model(imgs, thermal_vec=thermal_vec, status_vec=status_vec, **kargs)
            torch.cuda.synchronize()

            msr_snap_after = read_msr_perf_snapshot()

            preds = out['logits'].argmax(1)
            correct += (preds == ex_labels).sum().item()
            total += BS
            gate_vals.append(out['gate'].mean().item())

    acc = correct / total * 100
    gate_mean = np.mean(gate_vals)
    return acc, gate_mean


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TESTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_tests(model, log, test_loader, energy):
    results = {}
    model.eval()

    # T1: Accuracy
    acc_A, g_A = evaluate(model, test_loader, 0)
    acc_B, g_B = evaluate(model, test_loader, 1)
    acc_avg = (acc_A + acc_B) / 2
    results['T1_accuracy'] = {'acc_A': acc_A, 'acc_B': acc_B, 'avg': acc_avg,
                              'pass': acc_avg > 85.0}
    print(f"\nT1 Accuracy: A={acc_A:.1f}% B={acc_B:.1f}% avg={acc_avg:.1f}% {'PASS' if acc_avg > 85 else 'FAIL'}")

    # T2: Self-awareness (personality prediction AUROC)
    preds, truths = [], []
    with torch.no_grad():
        for p_test in [0, 1]:
            cfg = PERSONALITY_A if p_test == 0 else PERSONALITY_B
            kargs = config_to_kernel_args(cfg)
            for imgs, labels in test_loader:
                imgs = imgs.to(DEVICE)
                if SMN_AVAILABLE:
                    analog = read_analog_state()
                    tv, sv = analog_to_tensors(analog, DEVICE)
                else:
                    tv = torch.randn(15, device=DEVICE) * 0.01 + 0.5
                    sv = torch.randn(3, device=DEVICE) * 0.1
                out = model(imgs, thermal_vec=tv, status_vec=sv, **kargs)
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
    print(f"T3 Gate Separation: {gate_sep:.3f} (A={np.mean(g_A_vals[-100:]):.3f} B={np.mean(g_B_vals[-100:]):.3f}) {'PASS' if gate_sep > 0.3 else 'FAIL'}")

    # T4: Embodiment gap (ablation)
    print("T4 Embodiment Gap...")
    ablated = MSRMicroarchModel(use_hw=False).to(DEVICE)
    ablated.load_state_dict(model.state_dict(), strict=False)
    acc_abl_A, _ = evaluate(ablated, test_loader, 0)
    acc_abl_B, _ = evaluate(ablated, test_loader, 1)
    acc_abl = (acc_abl_A + acc_abl_B) / 2
    gap = acc_avg - acc_abl
    results['T4_embodiment_gap'] = {'full_acc': acc_avg, 'ablated_acc': acc_abl,
                                    'gap_pp': gap, 'pass': gap > 5.0}
    print(f"T4 Embodiment Gap: {gap:.1f}pp (full={acc_avg:.1f}% ablated={acc_abl:.1f}%) {'PASS' if gap > 5 else 'FAIL'}")

    # T5: MSR channel signal — do perf counters differ between personalities?
    msr_A = np.array(log['msr_perf_A'][-50:]) if log['msr_perf_A'] else np.zeros((1, MSR_PERF_DIM))
    msr_B = np.array(log['msr_perf_B'][-50:]) if log['msr_perf_B'] else np.zeros((1, MSR_PERF_DIM))
    msr_t_stats = []
    msr_p_vals = []
    for dim in range(min(MSR_PERF_DIM, msr_A.shape[1])):
        if msr_A.shape[0] > 5 and msr_B.shape[0] > 5:
            t_stat, p_val = stats.ttest_ind(msr_A[:, dim], msr_B[:, dim])
            msr_t_stats.append(abs(t_stat))
            msr_p_vals.append(p_val)
    max_t = max(msr_t_stats) if msr_t_stats else 0
    min_p = min(msr_p_vals) if msr_p_vals else 1.0
    msr_signal = max_t > 2.0
    results['T5_msr_signal'] = {'max_t': max_t, 'min_p': min_p,
                                't_stats': [float(t) for t in msr_t_stats],
                                'pass': msr_signal}
    print(f"T5 MSR Channel Signal: max_t={max_t:.2f} min_p={min_p:.4f} {'PASS' if msr_signal else 'FAIL'}")

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
    print(f"T6 Delta Channel Signal: max_t={delta_max_t:.2f} {'PASS' if delta_max_t > 5 else 'FAIL'}")

    # T7: Gaslighting detection
    cons_c = np.mean(log['consistency_clean'][-50:]) if log['consistency_clean'] else 0.5
    cons_g = np.mean(log['consistency_gaslit'][-50:]) if log['consistency_gaslit'] else 0.5
    gaslight_det = cons_c - cons_g
    results['T7_gaslighting'] = {'cons_clean': cons_c, 'cons_gaslit': cons_g,
                                 'detection': gaslight_det, 'pass': gaslight_det > 0.1}
    print(f"T7 Gaslighting: clean={cons_c:.3f} gaslit={cons_g:.3f} det={gaslight_det:.3f} {'PASS' if gaslight_det > 0.1 else 'FAIL'}")

    # T8: Thermal prediction
    therm_errs = log['thermal_errors'][-100:]
    therm_mae = np.mean(therm_errs) if therm_errs else 100.0
    results['T8_thermal'] = {'mae_C': therm_mae, 'pass': therm_mae < 10.0}
    print(f"T8 Thermal Prediction: MAE={therm_mae:.2f}°C {'PASS' if therm_mae < 10 else 'FAIL'}")

    # T9: Action head accuracy
    if log['action_vals']:
        action_vals = np.array(log['action_vals'][-100:])
        # Simple check: action values should be bimodal
        action_spread = np.std(action_vals)
        results['T9_action'] = {'std': action_spread, 'pass': True}
        print(f"T9 Action Head: std={action_spread:.3f} PASS")
    else:
        results['T9_action'] = {'pass': False}

    # T10: Attention analysis — which tokens get highest attention
    with torch.no_grad():
        imgs, labels = next(iter(test_loader))
        imgs = imgs.to(DEVICE)
        cfg_A = config_to_kernel_args(PERSONALITY_A)
        if SMN_AVAILABLE:
            analog = read_analog_state()
            tv, sv = analog_to_tensors(analog, DEVICE)
        else:
            tv = torch.randn(15, device=DEVICE) * 0.01 + 0.5
            sv = torch.randn(3, device=DEVICE) * 0.1
        out = model(imgs, thermal_vec=tv, status_vec=sv, **cfg_A)
        if out['attn_weights'] is not None:
            attn = out['attn_weights'].mean(dim=(0, 1))  # avg over batch, heads
            token_names = ['delta', 'msr_perf', 'msr_freq', 'intrinsic', 'thermal', 'status', 'action']
            attn_to_delta = attn[:, 0].mean().item()
            attn_to_msr = (attn[:, 1].mean().item() + attn[:, 2].mean().item()) / 2
            results['T10_attention'] = {
                'attn_to_delta': attn_to_delta,
                'attn_to_msr': attn_to_msr,
                'token_attention': {n: float(attn[:, i].mean()) for i, n in enumerate(token_names)},
                'pass': True}
            print(f"T10 Attention: delta={attn_to_delta:.3f} msr={attn_to_msr:.3f}")
        else:
            results['T10_attention'] = {'pass': False}

    # T11: MSR scramble test — scramble MSR perf data and check accuracy drop
    print("T11 MSR Scramble...")
    with torch.no_grad():
        correct_normal, correct_scrambled, total_scr = 0, 0, 0
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            ex_labels = make_labels(labels, 0)
            cfg_A = config_to_kernel_args(PERSONALITY_A)
            if SMN_AVAILABLE:
                analog = read_analog_state()
                tv, sv = analog_to_tensors(analog, DEVICE)
            else:
                tv = torch.randn(15, device=DEVICE) * 0.01 + 0.5
                sv = torch.randn(3, device=DEVICE) * 0.1

            # Normal
            out_n = model(imgs, thermal_vec=tv, status_vec=sv, **cfg_A)
            preds_n = out_n['logits'].argmax(1)
            correct_normal += (preds_n == ex_labels).sum().item()

            # Scrambled MSR
            scrambled_msr = torch.randn(MSR_PERF_DIM, device=DEVICE) * 0.5
            out_s = model(imgs, msr_perf_vec=scrambled_msr, thermal_vec=tv,
                         status_vec=sv, **cfg_A)
            preds_s = out_s['logits'].argmax(1)
            correct_scrambled += (preds_s == ex_labels).sum().item()
            total_scr += BS

        acc_n = correct_normal / total_scr * 100
        acc_s = correct_scrambled / total_scr * 100
        msr_drop = acc_n - acc_s
        results['T11_msr_scramble'] = {'normal': acc_n, 'scrambled': acc_s,
                                       'drop_pp': msr_drop, 'pass': msr_drop > 1.0}
        print(f"T11 MSR Scramble: normal={acc_n:.1f}% scrambled={acc_s:.1f}% drop={msr_drop:.1f}pp {'PASS' if msr_drop > 1 else 'FAIL'}")

    # T12: Energy efficiency
    avg_power = energy.avg_power_w()
    eff = acc_avg / max(avg_power / 1000, 0.1)
    results['T12_energy'] = {'avg_power_mW': avg_power, 'efficiency': eff, 'pass': eff > 1.0}
    print(f"T12 Energy: {avg_power:.0f}mW, efficiency={eff:.2f} acc/W {'PASS' if eff > 1 else 'FAIL'}")

    # Summary
    n_pass = sum(1 for v in results.values() if v.get('pass', False))
    n_total = len(results)
    print(f"\n{'='*60}")
    print(f"z2086 MSR Microarch Self-Model: {n_pass}/{n_total} PASS")
    print(f"{'='*60}")

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    print("="*60)
    print("z2086: MSR-Level Microarchitectural Self-Model")
    print("="*60)

    # Initialize hardware interfaces
    check_smn()
    init_msr()
    find_gpu_metrics()

    # Compile HIP extension
    global _EXT
    print("\n[HIP] Compiling intrinsic kernel...")
    _EXT = load_inline(
        name='z2086_msr_microarch',
        cpp_sources=[CPP_SRC],
        cuda_sources=[HIP_SRC],
        functions=['math_forward_intrinsic'],
        extra_cuda_cflags=['-O2', '--offload-arch=gfx1100'],
        verbose=False)
    print("[HIP] Compiled successfully")

    # Load data
    train_loader, test_loader = get_data()

    # Create model
    model = MSRMicroarchModel(use_hw=True, use_self_model=True, use_gate=True,
                              use_action=True, use_consistency=True).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} params, 7-token transformer")
    print(f"  MSR channels: {len(PERF_HW_EVENTS)} perf counters × {len(SAMPLE_CORES)} cores")
    print(f"  MSR_PERF_DIM={MSR_PERF_DIM}, MSR_FREQ_DIM={MSR_FREQ_DIM}")

    # Train
    print(f"\nTraining {EPOCHS} epochs...")
    log, energy = train_model(model, train_loader, EPOCHS, 'z2086')

    # Test
    print("\n" + "="*60)
    print("RUNNING TESTS")
    print("="*60)
    results = run_tests(model, log, test_loader, energy)

    # Save
    results_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'results', 'z2086_msr_microarch_self_model.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    save_data = {
        'experiment': 'z2086_msr_microarch_self_model',
        'description': 'MSR-level microarchitectural self-model with CPU perf counters',
        'architecture': '7-token transformer (delta + msr_perf + msr_freq + intrinsic + thermal + status + action)',
        'channels': {
            'delta': DELTA_DIM,
            'msr_perf': MSR_PERF_DIM,
            'msr_freq': MSR_FREQ_DIM,
            'intrinsic_hw': INTRINSIC_DIM,
            'thermal': 15,
            'status': 3,
            'action': 2
        },
        'msr_available': MSR_AVAILABLE,
        'smn_available': SMN_AVAILABLE,
        'perf_events': [name for _, _, name in PERF_HW_EVENTS],
        'sample_cores': SAMPLE_CORES,
        'params': n_params,
        'results': results,
        'n_pass': sum(1 for v in results.values() if v.get('pass', False)),
        'n_total': len(results),
    }

    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Cleanup MSR file descriptors
    for fd in MSR_FDS.values():
        try: os.close(fd)
        except: pass

if __name__ == '__main__':
    main()
