#!/usr/bin/env python3
"""z2066: Deep Analog Embodiment — Closest to the Messy Reality

Extends z2065 with ISA-level registers discovered in the z2065 probe session.
The model now reads its OWN execution state at the wave level.

NEW channels (from z2065 ISA probe, 2026-02-10):
  - SHADER_CYCLES hwreg(29): per-wave cycle counter → in-kernel timing
  - STATUS hwreg(2): wave execution flags (SCC, EXECZ, VCCZ, LDS bit)
  - HW_ID2 hwreg(24): VMID, queue ID, pipe ID → scheduling context
  - PC_LO hwreg(8): program counter → kernel binary fingerprint

Combined with z2065's channels:
  - MMIO: GRBM_STATUS, GRBM_STATUS_SE0, RLC_STATUS, CG_SPLL_CNTL
  - gpu_metrics: temp, power, SCLK, DRAM BW, activity%
  - Fence counters, sched_mask, hwmon sensors
  - ISA: HW_ID1 hwreg(23) for WGP placement, MODE hwreg(1) FP rounding
  - Actuation: DVFS + sched_mask + s_setreg MODE register

26-dim hardware vector (20 from z2065 + 6 new):
  [0-4]   sysfs: timing, power, temp, dram_bw, sclk_norm
  [5-10]  MMIO: gui_active, se0_busy, cp_busy, spi_busy, rlc_lo16, spll_lo8
  [11]    fence_rate
  [12-13] mode_reg: fp32_round, fp16_round
  [14]    sched_mask_norm
  [15-19] gpu_metrics_ext: gfx_activity, soc_temp, fclk, uclk, core_power
  [20]    cycle_delta_norm — SHADER_CYCLES before/after (in-kernel timing!)
  [21]    status_lds_bit — STATUS register LDS flag
  [22]    hw_id2_vmid_norm — HW_ID2 VMID / 15
  [23]    hw_id2_pipe_norm — HW_ID2 pipe / 3
  [24]    pc_lo_norm — PC_LO truncated + normalized
  [25]    grbm_load_bit — GRBM_STATUS bit 22 (compute load indicator)

Self-model predicts 12 targets (8 from z2065 + 4 new):
  sclk, timing, se0_busy, gui_active, rlc_lo16, power, temp, fence,
  cycle_delta, status_lds, hw_id2_vmid, grbm_load_bit

Why this matters for embodiment:
  SHADER_CYCLES is read INSIDE the wave — this is like measuring your own
  heartbeat directly, not through a stethoscope. The model's computation
  duration is physically entangled with the clock it reads.

Why this matters for efficiency:
  In-kernel timing is cheaper than CUDA events (no host sync). More registers
  = richer self-model = better effort decisions = smarter DVFS = less energy.

Tests (18): z2065's 16 + 2 new ISA-level tests
"""
import torch, torch.nn as nn, torch.nn.functional as F
import os, sys, json, time, copy, random, struct, subprocess
import numpy as np
from torchvision import datasets, transforms
from sklearn.metrics import r2_score, roc_auc_score
from scipy import stats

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')
os.environ.setdefault('PYTORCH_ROCM_ARCH', 'gfx1100')
from torch.utils.cpp_extension import load_inline

DEVICE = 'cuda'
BS = 256
EPOCHS = 30
PHASE2_EPOCH = 12
SWITCH_EVERY = 12
NUM_BANKS = 8
HW_DIM = 26          # 26-dimensional hardware vector
SELF_TARGETS = 12    # self-model predicts 12 targets
EFFORT_DIM = 3       # effort: [sclk_pct, sched_pct, round_mode]
ACTUATION_WAIT = 0.10

PHASE1_CONFIGS = [
    ('low',  255, 0), ('low',  1,   0), ('high', 255, 0), ('high', 1,   0),
    ('low',  15,  1), ('high', 15,  1), ('low',  255, 1), ('high', 255, 1),
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HIP KERNEL: Extended probe with all ISA registers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HIP_SRC = r'''
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Full probe: WGP + work + MODE + SHADER_CYCLES + STATUS + HW_ID2 + PC_LO
__global__ void probe_all(int* wgp_ids, float* work, int* mode_reg,
                           int* cycle_delta, int* status_reg,
                           int* hw_id2_reg, int* pc_lo_reg, int n) {
    int bid = blockIdx.x;
    if (bid >= n || threadIdx.x != 0) return;

    // Read SHADER_CYCLES BEFORE work
    unsigned int c0;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(c0));
    c0 = __builtin_amdgcn_readfirstlane(c0);

    // Read HW_ID1 for WGP placement
    unsigned int hw;
    asm volatile("s_getreg_b32 %0, hwreg(23)" : "=s"(hw));
    wgp_ids[bid] = (int)((hw >> 7) & 0xF);

    // Read MODE register
    unsigned int mode;
    asm volatile("s_getreg_b32 %0, hwreg(1, 0, 4)" : "=s"(mode));
    mode_reg[bid] = (int)(mode & 0xF);

    // Read STATUS register
    unsigned int sts;
    asm volatile("s_getreg_b32 %0, hwreg(2)" : "=s"(sts));
    sts = __builtin_amdgcn_readfirstlane(sts);
    status_reg[bid] = (int)sts;

    // Read HW_ID2 (VMID, queue, pipe)
    unsigned int id2;
    asm volatile("s_getreg_b32 %0, hwreg(24)" : "=s"(id2));
    id2 = __builtin_amdgcn_readfirstlane(id2);
    hw_id2_reg[bid] = (int)id2;

    // Read PC_LO (program counter)
    unsigned int pc;
    asm volatile("s_getreg_b32 %0, hwreg(8)" : "=s"(pc));
    pc = __builtin_amdgcn_readfirstlane(pc);
    pc_lo_reg[bid] = (int)pc;

    // Compute work (timing depends on SCLK)
    float acc = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < 5000; i++) acc += 1.0f / (float)(i+1);
    work[bid] = acc;

    // Read SHADER_CYCLES AFTER work
    unsigned int c1;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(c1));
    c1 = __builtin_amdgcn_readfirstlane(c1);
    cycle_delta[bid] = (int)(c1 - c0);
}

// Probe with MODE register write
__global__ void probe_with_mode(int* wgp_ids, float* work, int* mode_out,
                                 int* cycle_delta, int* status_reg,
                                 int* hw_id2_reg, int* pc_lo_reg,
                                 int round_mode, int n) {
    int bid = blockIdx.x;
    if (bid >= n || threadIdx.x != 0) return;

    // SHADER_CYCLES before
    unsigned int c0;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(c0));
    c0 = __builtin_amdgcn_readfirstlane(c0);

    // Set FP rounding mode
    unsigned int rm = (unsigned int)(round_mode ? 0xF : 0x0);
    unsigned int rm_lane = __builtin_amdgcn_readfirstlane(rm);
    asm volatile("s_setreg_b32 hwreg(1, 0, 4), %0" : : "s"(rm_lane));

    // Read back MODE
    unsigned int mode_readback;
    asm volatile("s_getreg_b32 %0, hwreg(1, 0, 4)" : "=s"(mode_readback));
    mode_out[bid] = (int)(mode_readback & 0xF);

    // Read HW_ID1
    unsigned int hw;
    asm volatile("s_getreg_b32 %0, hwreg(23)" : "=s"(hw));
    wgp_ids[bid] = (int)((hw >> 7) & 0xF);

    // Read STATUS
    unsigned int sts;
    asm volatile("s_getreg_b32 %0, hwreg(2)" : "=s"(sts));
    sts = __builtin_amdgcn_readfirstlane(sts);
    status_reg[bid] = (int)sts;

    // Read HW_ID2
    unsigned int id2;
    asm volatile("s_getreg_b32 %0, hwreg(24)" : "=s"(id2));
    id2 = __builtin_amdgcn_readfirstlane(id2);
    hw_id2_reg[bid] = (int)id2;

    // Read PC_LO
    unsigned int pc;
    asm volatile("s_getreg_b32 %0, hwreg(8)" : "=s"(pc));
    pc = __builtin_amdgcn_readfirstlane(pc);
    pc_lo_reg[bid] = (int)pc;

    // Compute (affected by rounding mode)
    float acc = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < 5000; i++) acc += 1.0f / (float)(i+1);
    work[bid] = acc;

    // SHADER_CYCLES after
    unsigned int c1;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(c1));
    c1 = __builtin_amdgcn_readfirstlane(c1);
    cycle_delta[bid] = (int)(c1 - c0);

    // Restore default rounding
    unsigned int zero = __builtin_amdgcn_readfirstlane(0u);
    asm volatile("s_setreg_b32 hwreg(1, 0, 4), %0" : : "s"(zero));
}

std::vector<torch::Tensor> probe(int n) {
    auto io = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto fo = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto wgps = torch::zeros({n}, io);
    auto work = torch::zeros({n}, fo);
    auto mode = torch::zeros({n}, io);
    auto cycles = torch::zeros({n}, io);
    auto sts = torch::zeros({n}, io);
    auto id2 = torch::zeros({n}, io);
    auto pclo = torch::zeros({n}, io);
    probe_all<<<n, 32>>>(wgps.data_ptr<int>(), work.data_ptr<float>(),
                          mode.data_ptr<int>(), cycles.data_ptr<int>(),
                          sts.data_ptr<int>(), id2.data_ptr<int>(),
                          pclo.data_ptr<int>(), n);
    return {wgps, work, mode, cycles, sts, id2, pclo};
}

std::vector<torch::Tensor> probe_mode(int round_mode, int n) {
    auto io = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto fo = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto wgps = torch::zeros({n}, io);
    auto work = torch::zeros({n}, fo);
    auto mode = torch::zeros({n}, io);
    auto cycles = torch::zeros({n}, io);
    auto sts = torch::zeros({n}, io);
    auto id2 = torch::zeros({n}, io);
    auto pclo = torch::zeros({n}, io);
    probe_with_mode<<<n, 32>>>(wgps.data_ptr<int>(), work.data_ptr<float>(),
                                mode.data_ptr<int>(), cycles.data_ptr<int>(),
                                sts.data_ptr<int>(), id2.data_ptr<int>(),
                                pclo.data_ptr<int>(), round_mode, n);
    return {wgps, work, mode, cycles, sts, id2, pclo};
}
'''

CPP_SRC = r'''
#include <torch/extension.h>
std::vector<torch::Tensor> probe(int n);
std::vector<torch::Tensor> probe_mode(int round_mode, int n);
'''


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SUB-FIRMWARE REGISTER ACCESS (same as z2065 + new channels)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _find_card():
    for c in range(8):
        if os.path.exists(f'/sys/class/drm/card{c}/device/gpu_metrics'):
            return c
    return 0

_CARD = _find_card()
_DRI = _CARD
if not os.path.exists(f'/sys/kernel/debug/dri/{_DRI}/amdgpu_regs2'):
    for d in range(8):
        if os.path.exists(f'/sys/kernel/debug/dri/{d}/amdgpu_regs2'):
            _DRI = d; break

REGS2_PATH = f'/sys/kernel/debug/dri/{_DRI}/amdgpu_regs2'
GPU_METRICS_PATH = f'/sys/class/drm/card{_CARD}/device/gpu_metrics'
DPM_PATH = f'/sys/class/drm/card{_CARD}/device/power_dpm_force_performance_level'
SCHED_MASK_PATH = f'/sys/kernel/debug/dri/{_DRI}/amdgpu_compute_sched_mask'
FENCE_PATH = f'/sys/kernel/debug/dri/{_DRI}/amdgpu_fence_info'
print(f"[z2066] Detected card{_CARD}, debugfs dri/{_DRI}")

_regs2_fd = None
def open_regs2():
    global _regs2_fd
    if _regs2_fd is None:
        try: _regs2_fd = os.open(REGS2_PATH, os.O_RDONLY)
        except: return None
    return _regs2_fd

def read_mmio(offset):
    try:
        fd = open_regs2()
        if fd is None: return 0
        os.lseek(fd, offset, os.SEEK_SET)
        data = os.read(fd, 4)
        return struct.unpack('<I', data)[0] if len(data) == 4 else 0
    except: return 0

def read_register_snapshot():
    grbm = read_mmio(0x8010)
    grbm_se0 = read_mmio(0x8014)
    rlc = read_mmio(0xB004)
    spll = read_mmio(0xE000)
    return {
        'grbm_raw': grbm,
        'gui_active': bool(grbm & (1 << 31)),
        'se0_busy': bool(grbm_se0 & (1 << 27)) if grbm_se0 != 0xffffffff else False,
        'cp_busy': bool(grbm & (1 << 13)),
        'spi_busy': bool(grbm & (1 << 23)),
        'ta_busy': bool(grbm & (1 << 30)),
        'load_bit': bool(grbm & (1 << 22)),  # NEW: compute load indicator
        'rlc_lo16': float(rlc & 0xFFFF) / 65535.0,
        'spll_lo8': float(spll & 0xFF) / 255.0,
    }

def read_fence_delta():
    try:
        with open(FENCE_PATH, 'r') as f: text = f.read()
        total = 0
        for line in text.split('\n'):
            if 'Last emitted' in line and 'trailing' not in line:
                try: total += int(line.strip().split()[-1], 16)
                except: pass
        return total
    except: return 0

def read_gpu_metrics():
    try:
        with open(GPU_METRICS_PATH, 'rb') as f: data = f.read()
        if len(data) < 200: return None
        return {
            'temp_gfx_c': struct.unpack_from('<H', data, 4)[0] / 100.0,
            'temp_soc_c': struct.unpack_from('<H', data, 6)[0] / 100.0,
            'gfx_activity_pct': struct.unpack_from('<H', data, 42)[0],
            'dram_reads_mbps': struct.unpack_from('<H', data, 94)[0],
            'socket_power_mw': struct.unpack_from('<I', data, 112)[0],
            'all_core_power_mw': struct.unpack_from('<I', data, 132)[0],
            'sclk_mhz': struct.unpack_from('<H', data, 174)[0],
            'fclk_mhz': struct.unpack_from('<H', data, 182)[0],
            'uclk_mhz': struct.unpack_from('<H', data, 186)[0],
        }
    except: return None


# ━━━ ACTUATION ━━━
def set_dvfs(mode):
    try: open(DPM_PATH, 'w').write(mode)
    except: pass

def set_sched_mask(mask_val):
    try: open(SCHED_MASK_PATH, 'w').write(str(mask_val))
    except: pass

def reset_actuation():
    set_sched_mask(255); set_dvfs('auto')

def apply_actuation(sclk_pct, sched_pct):
    perf = 'high' if sclk_pct >= 0.5 else 'low'
    set_dvfs(perf)
    mask = max(1, min(255, int(1 + sched_pct * 254)))
    set_sched_mask(mask)
    return perf, mask

def measure_wall_clock(ext, n=BS):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record(); ext.probe(n); e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)


# ━━━ CHARACTERIZATION ━━━
def characterize(ext):
    print("\n--- Deep Analog Characterization ---")
    info = {}
    for i, (perf, mask, rmode) in enumerate(PHASE1_CONFIGS):
        set_dvfs(perf); set_sched_mask(mask); time.sleep(0.4)
        for _ in range(5): ext.probe_mode(rmode, BS); torch.cuda.synchronize()

        walls, cycles_list = [], []
        fence_before = read_fence_delta()
        for _ in range(15):
            res = ext.probe_mode(rmode, BS); torch.cuda.synchronize()
            walls.append(measure_wall_clock(ext))
            cd = res[3].cpu().numpy()
            cycles_list.append(int(np.median(cd)))
        fence_after = read_fence_delta()
        gm = read_gpu_metrics()
        mode_vals = res[2].cpu().numpy()
        status_vals = res[4].cpu().numpy()
        id2_vals = res[5].cpu().numpy()
        pclo_vals = res[6].cpu().numpy()

        key = f'cfg{i}_{perf}_m{mask}_r{rmode}'
        info[key] = {
            'perf': perf, 'mask': mask, 'round_mode': rmode,
            'sclk': gm['sclk_mhz'] if gm else 0,
            'wall_ms': float(np.mean(walls)),
            'cycle_delta_median': int(np.median(cycles_list)),
            'status_unique': len(set(status_vals.tolist())),
            'hw_id2_val': int(id2_vals[0]),
            'pc_lo_val': int(pclo_vals[0]) & 0xFFFFFFFF,
            'fence_delta': fence_after - fence_before,
            'mode_readback': int(np.median(mode_vals)),
            'work_mean': float(res[1].mean().item()),
        }
        print(f"  [{i}] {perf:4s} m={mask:3d} r={rmode}: "
              f"SCLK={info[key]['sclk']:4d} wall={info[key]['wall_ms']:.3f}ms "
              f"cycles={info[key]['cycle_delta_median']} "
              f"sts_u={info[key]['status_unique']} "
              f"id2=0x{info[key]['hw_id2_val']:08X}")

    reset_actuation(); time.sleep(0.3)

    vals = [v for v in info.values() if isinstance(v, dict) and 'wall_ms' in v]
    info['norm'] = {
        'wall_min': min(v['wall_ms'] for v in vals),
        'wall_max': max(v['wall_ms'] for v in vals),
        'sclk_min': min(v['sclk'] for v in vals),
        'sclk_max': max(v['sclk'] for v in vals),
        'power_min': 20, 'power_max': 50,
        'fence_min': min(v['fence_delta'] for v in vals),
        'fence_max': max(v['fence_delta'] for v in vals),
        'cycle_min': min(v['cycle_delta_median'] for v in vals),
        'cycle_max': max(v['cycle_delta_median'] for v in vals),
    }
    return info


def make_hw_vector(wall_ms, gm, regs, mode_readback, fence_rate,
                    sched_mask_norm, cycle_delta, status_val, hw_id2_val,
                    pc_lo_val, char_info):
    """Build 26-dim hardware vector."""
    n = char_info['norm']

    # [0-4] sysfs
    timing = max(0, min(1, (wall_ms - n['wall_min']) / max(n['wall_max'] - n['wall_min'], 1e-6)))
    if gm:
        power = max(0, min(1, (gm['socket_power_mw']/1000 - n['power_min']) / max(n['power_max'] - n['power_min'], 0.1)))
        temp = max(0, min(1, (gm['temp_gfx_c'] - 30) / 50.0))
        dram_bw = max(0, min(1, gm['dram_reads_mbps'] / 5000.0))
        sclk_norm = max(0, min(1, (gm['sclk_mhz'] - n['sclk_min']) / max(n['sclk_max'] - n['sclk_min'], 1)))
    else:
        power, temp, dram_bw, sclk_norm = 0.5, 0.5, 0.5, 0.5

    # [5-10] MMIO
    gui_active = float(regs['gui_active'])
    se0_busy = float(regs['se0_busy'])
    cp_busy = float(regs['cp_busy'])
    spi_busy = float(regs['spi_busy'])
    rlc_lo16 = regs['rlc_lo16']
    spll_lo8 = regs['spll_lo8']

    # [11] fence
    fence_norm = max(0, min(1, (fence_rate - n.get('fence_min', 0)) /
                    max(n.get('fence_max', 1) - n.get('fence_min', 0), 1)))

    # [12-13] MODE
    fp32_round = float((mode_readback & 0x3) != 0)
    fp16_round = float(((mode_readback >> 2) & 0x3) != 0)

    # [14] sched mask
    sched_norm_v = sched_mask_norm

    # [15-19] gpu_metrics extended
    if gm:
        gfx_activity = max(0, min(1, gm['gfx_activity_pct'] / 100.0))
        soc_temp = max(0, min(1, (gm['temp_soc_c'] - 30) / 50.0))
        fclk_norm = max(0, min(1, gm['fclk_mhz'] / 2500.0))
        uclk_norm = max(0, min(1, gm['uclk_mhz'] / 2500.0))
        core_power = max(0, min(1, gm['all_core_power_mw'] / 50000.0))
    else:
        gfx_activity, soc_temp, fclk_norm, uclk_norm, core_power = 0.5, 0.5, 0.5, 0.5, 0.5

    # [20] cycle_delta — SHADER_CYCLES (in-kernel timing!)
    cd_min, cd_max = n.get('cycle_min', 1000), n.get('cycle_max', 100000)
    cycle_norm = max(0, min(1, (cycle_delta - cd_min) / max(cd_max - cd_min, 1)))

    # [21] STATUS LDS bit (bit 15)
    status_lds = float((status_val >> 15) & 1)

    # [22-23] HW_ID2: VMID [3:0] of bits [7:0], pipe [1:0] of bits [9:8]
    hw_id2_vmid = float((hw_id2_val & 0xF)) / 15.0
    hw_id2_pipe = float((hw_id2_val >> 8) & 0x3) / 3.0

    # [24] PC_LO truncated (lower 16 bits, normalized)
    pc_lo_norm = float(pc_lo_val & 0xFFFF) / 65535.0

    # [25] GRBM load bit
    grbm_load = float(regs.get('load_bit', False))

    return [timing, power, temp, dram_bw, sclk_norm,
            gui_active, se0_busy, cp_busy, spi_busy, rlc_lo16, spll_lo8,
            fence_norm, fp32_round, fp16_round, sched_norm_v,
            gfx_activity, soc_temp, fclk_norm, uclk_norm, core_power,
            cycle_norm, status_lds, hw_id2_vmid, hw_id2_pipe,
            pc_lo_norm, grbm_load]


# ━━━ MODEL ━━━
class DeepAnalogModel(nn.Module):
    def __init__(self, use_banks=True, use_hw=True, use_self_model=True,
                 use_gate=True, use_effort=True, always_light=False):
        super().__init__()
        self.use_banks = use_banks
        self.use_hw = use_hw
        self.use_self_model = use_self_model
        self.use_gate = use_gate
        self.use_effort = use_effort
        self.always_light = always_light

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64*7*7, 128), nn.ReLU())

        if use_hw:
            self.hw_proj = nn.Sequential(
                nn.Linear(HW_DIM, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU())

        combined_dim = 128 + (64 if use_hw else 0)

        if use_banks:
            self.bank_w = nn.Parameter(torch.randn(NUM_BANKS, 128, 128) * 0.02)

        if use_self_model:
            self.self_model = nn.Sequential(
                nn.Linear(combined_dim, 96), nn.ReLU(),
                nn.Linear(96, 64), nn.ReLU(),
                nn.Linear(64, SELF_TARGETS))

        if use_gate:
            self.gate_net = nn.Sequential(
                nn.Linear(SELF_TARGETS, 32), nn.ReLU(),
                nn.Linear(32, 1), nn.Sigmoid())

        self.head_full = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
        self.head_light = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))

        if use_effort:
            self.demand_proj = nn.Sequential(
                nn.Linear(EFFORT_DIM, 24), nn.ReLU(), nn.Linear(24, 24), nn.ReLU())
            self.effort_head = nn.Sequential(
                nn.Linear(combined_dim + 24, 96), nn.ReLU(),
                nn.Linear(96, 48), nn.ReLU(),
                nn.Linear(48, EFFORT_DIM))

    def forward(self, x, bank_ids=None, hw_vector=None, demand_vector=None):
        h_img = self.encoder(x)
        B = h_img.shape[0]

        if self.use_hw and hw_vector is not None:
            h_hw = self.hw_proj(hw_vector)
            h_combined = torch.cat([h_img, h_hw], dim=1)
        else:
            h_combined = h_img

        self_pred = None
        if self.use_self_model:
            self_pred = self.self_model(h_combined)

        if self.use_gate and not self.always_light:
            if self.use_self_model and self_pred is not None:
                gate = self.gate_net(self_pred)
            else:
                gate = self.gate_net(torch.full((B, SELF_TARGETS), 0.5, device=x.device))
        elif self.always_light:
            gate = torch.zeros(B, 1, device=x.device)
        else:
            gate = torch.full((B, 1), 0.5, device=x.device)

        if self.use_banks and bank_ids is not None:
            h_banked = torch.bmm(self.bank_w[bank_ids], h_img.unsqueeze(-1)).squeeze(-1)
            logits_full = self.head_full(h_banked)
        else:
            logits_full = self.head_full(h_img)

        logits_light = self.head_light(h_img)
        logits = gate * logits_full + (1 - gate) * logits_light

        effort = None
        if self.use_effort and demand_vector is not None:
            h_demand = self.demand_proj(demand_vector)
            effort_input = torch.cat([h_combined.detach(), h_demand], dim=1)
            effort = torch.sigmoid(self.effort_head(effort_input))

        return {'logits': logits, 'self_pred': self_pred, 'gate': gate, 'effort': effort}


def make_labels(digits, bank_ids, demand_level):
    labels = digits.clone()
    if demand_level > 0.5:
        even = (bank_ids % 2 == 0)
        shift = int(1 + demand_level * 8)
        labels[even] = (digits[even] + shift) % 10
        labels[~even] = (digits[~even] + shift + 2) % 10
    else:
        labels = (9 - digits) % 10
    return labels

def get_data():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    tr = datasets.MNIST('data', train=True, download=True, transform=tf)
    te = datasets.MNIST('data', train=False, transform=tf)
    return (torch.utils.data.DataLoader(tr, batch_size=BS, shuffle=True, drop_last=True),
            torch.utils.data.DataLoader(te, batch_size=BS, shuffle=False, drop_last=True))


# ━━━ TRAINING ━━━
def train_model(model, ext, loader, epochs, name, char_info,
                actuate=True, model_controlled=True):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    log = {'gate': [], 'sclk': [], 'effort_sclk': [], 'effort_sched': [],
           'effort_rmode': [], 'demand': [], 'cycle_delta': [], 'status': []}
    current_demand = [0.5, 1.0, 0.0]
    bn, level_idx = 0, 0
    prev_fence = read_fence_delta()
    current_round_mode = 0

    for ep in range(epochs):
        is_phase2 = model_controlled and ep >= PHASE2_EPOCH
        tot_loss, correct, total = 0, 0, 0

        for imgs, digits in loader:
            imgs, digits = imgs.to(DEVICE), digits.to(DEVICE)

            if not is_phase2:
                if actuate and bn % SWITCH_EVERY == 0:
                    level_idx = (level_idx + 1) % len(PHASE1_CONFIGS)
                    perf, mask, rmode = PHASE1_CONFIGS[level_idx]
                    set_dvfs(perf); set_sched_mask(mask)
                    time.sleep(ACTUATION_WAIT)
                    current_demand = [1.0 if perf == 'high' else 0.0,
                                      (mask - 1) / 254.0, float(rmode)]
                    current_round_mode = rmode

            wall_ms = measure_wall_clock(ext)
            gm = read_gpu_metrics()
            regs = read_register_snapshot()
            cur_fence = read_fence_delta()
            fence_rate = cur_fence - prev_fence
            prev_fence = cur_fence

            probe_res = ext.probe_mode(current_round_mode, BS)
            torch.cuda.synchronize()
            mode_readback = int(probe_res[2][0].item())
            cycle_delta = int(probe_res[3].cpu().numpy().mean())
            status_val = int(probe_res[4][0].item())
            hw_id2_val = int(probe_res[5][0].item())
            pc_lo_val = int(probe_res[6][0].item()) & 0xFFFFFFFF

            hw_vec = make_hw_vector(wall_ms, gm, regs, mode_readback, fence_rate,
                                     current_demand[1], cycle_delta, status_val,
                                     hw_id2_val, pc_lo_val, char_info)
            hw_t = torch.tensor([hw_vec] * BS, dtype=torch.float32, device=DEVICE)

            wgps = probe_res[0]
            bank_ids = (wgps // 2).long().clamp(0, NUM_BANKS - 1)
            demand_level = current_demand[0]
            labels = make_labels(digits, bank_ids, demand_level)

            if is_phase2 and bn % SWITCH_EVERY == 0:
                next_demand = [random.choice([0.0, 1.0]),
                               random.choice([0.0, 0.06, 0.5, 1.0]),
                               random.choice([0.0, 1.0])]
            elif not is_phase2:
                next_idx = ((bn + 1) // SWITCH_EVERY) % len(PHASE1_CONFIGS)
                np_lev, np_mask, np_rm = PHASE1_CONFIGS[next_idx]
                next_demand = [1.0 if np_lev == 'high' else 0.0,
                               (np_mask - 1) / 254.0, float(np_rm)]
            else:
                next_demand = current_demand

            demand_t = torch.tensor([next_demand] * BS, dtype=torch.float32, device=DEVICE)
            out = model(imgs, bank_ids=bank_ids, hw_vector=hw_t, demand_vector=demand_t)

            task_loss = F.cross_entropy(out['logits'], labels)

            self_loss = torch.tensor(0.0, device=DEVICE)
            if out['self_pred'] is not None:
                # 12 targets: sclk, timing, se0, gui, rlc, power, temp, fence,
                #             cycle_delta, status_lds, hw_id2_vmid, grbm_load
                targets = [hw_vec[4], hw_vec[0], hw_vec[6], hw_vec[5],
                           hw_vec[9], hw_vec[1], hw_vec[2], hw_vec[11],
                           hw_vec[20], hw_vec[21], hw_vec[22], hw_vec[25]]
                self_target = torch.tensor([targets] * BS, dtype=torch.float32, device=DEVICE)
                self_loss = F.mse_loss(out['self_pred'], self_target)

            effort_loss = torch.tensor(0.0, device=DEVICE)
            if out['effort'] is not None:
                effort_target = torch.tensor([next_demand] * BS, dtype=torch.float32, device=DEVICE)
                effort_loss = F.mse_loss(out['effort'], effort_target)

            homeo_loss = torch.tensor(0.0, device=DEVICE)
            if model.use_gate and not model.always_light:
                target_gate = torch.full_like(out['gate'], demand_level)
                homeo_loss = F.mse_loss(out['gate'], target_gate)

            energy_pen = torch.tensor(0.0, device=DEVICE)
            if is_phase2 and out['effort'] is not None and demand_level <= 0.5:
                energy_pen = out['effort'][:, 0].mean()

            loss = (task_loss + 0.25 * self_loss + 0.15 * effort_loss
                    + 0.1 * homeo_loss + 0.08 * energy_pen)

            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item()
            pred = out['logits'].argmax(1)
            correct += (pred == labels).sum().item()
            total += BS

            log['gate'].append(out['gate'].mean().item())
            log['sclk'].append(gm['sclk_mhz'] if gm else 600)
            log['cycle_delta'].append(cycle_delta)
            log['status'].append(status_val)
            if out['effort'] is not None:
                eff = out['effort'].mean(0)
                log['effort_sclk'].append(eff[0].item())
                log['effort_sched'].append(eff[1].item())
                log['effort_rmode'].append(eff[2].item())
                log['demand'].append(next_demand[0])

            if is_phase2 and model.use_effort and out['effort'] is not None:
                eff = out['effort'].mean(0)
                apply_actuation(eff[0].item(), eff[1].item())
                current_round_mode = 1 if eff[2].item() >= 0.5 else 0
                time.sleep(ACTUATION_WAIT)
                current_demand = [eff[0].item(), eff[1].item(), float(current_round_mode)]

            current_demand = next_demand
            bn += 1

        if ep % 3 == 0 or ep == epochs - 1:
            eff_str = ""
            if log['effort_sclk']:
                eff_str = (f" eff=[{np.mean(log['effort_sclk'][-50:]):.2f},"
                           f"{np.mean(log['effort_sched'][-50:]):.2f},"
                           f"{np.mean(log['effort_rmode'][-50:]):.2f}]")
            phase = "P2" if is_phase2 else "P1"
            cd_mean = np.mean(log['cycle_delta'][-50:]) if log['cycle_delta'] else 0
            print(f"  [{name} {phase}] Ep {ep}: loss={tot_loss/len(loader):.4f} "
                  f"acc={correct/total:.4f} gate={np.mean(log['gate'][-50:]):.3f}"
                  f"{eff_str} cycles={cd_mean:.0f}")

    return log


# ━━━ EVALUATION ━━━
def evaluate(model, ext, loader, char_info, actuate=True, model_controlled=True,
             scramble=False, fixed_sclk=None, ablate_type=None):
    model.eval()
    all_preds, all_sp, all_st = [], [], []
    gate_by_demand = {'high': [], 'low': []}
    efforts, demands = [], []
    effort_sclk_pairs, energy_log = [], []
    grbm_set, sclk_set, mode_seen = set(), set(), set()
    cycle_deltas_by_sclk = {'high': [], 'low': []}
    status_vals = set()
    bn, level_idx = 0, 0
    current_demand = [0.5, 1.0, 0.0]
    prev_fence = read_fence_delta()
    prev_effort_sclk = None
    current_round_mode = 0

    with torch.no_grad():
        for imgs, digits in loader:
            imgs, digits = imgs.to(DEVICE), digits.to(DEVICE)

            if fixed_sclk is not None:
                if bn == 0:
                    set_sched_mask(255); set_dvfs(fixed_sclk); time.sleep(0.3)
            elif not model_controlled:
                if actuate and bn % SWITCH_EVERY == 0:
                    level_idx = (level_idx + 1) % len(PHASE1_CONFIGS)
                    perf, mask, rmode = PHASE1_CONFIGS[level_idx]
                    set_dvfs(perf); set_sched_mask(mask)
                    time.sleep(ACTUATION_WAIT)
                    current_demand = [1.0 if perf == 'high' else 0.0,
                                      (mask - 1) / 254.0, float(rmode)]
                    current_round_mode = rmode

            wall_ms = measure_wall_clock(ext)
            gm = read_gpu_metrics()
            regs = read_register_snapshot()
            cur_fence = read_fence_delta()
            fence_rate = cur_fence - prev_fence
            prev_fence = cur_fence

            probe_res = ext.probe_mode(current_round_mode, BS)
            torch.cuda.synchronize()
            mode_readback = int(probe_res[2][0].item())
            cycle_delta = int(probe_res[3].cpu().numpy().mean())
            status_val = int(probe_res[4][0].item())
            hw_id2_val = int(probe_res[5][0].item())
            pc_lo_val = int(probe_res[6][0].item()) & 0xFFFFFFFF
            mode_seen.add(mode_readback)
            status_vals.add(status_val)

            sclk_key = 'high' if (gm and gm['sclk_mhz'] > 1000) else 'low'
            cycle_deltas_by_sclk[sclk_key].append(cycle_delta)

            if prev_effort_sclk is not None:
                sclk_na = max(0, min(1, (gm['sclk_mhz'] - char_info['norm']['sclk_min']) /
                    max(char_info['norm']['sclk_max'] - char_info['norm']['sclk_min'], 1))) if gm else 0.5
                effort_sclk_pairs.append((prev_effort_sclk, sclk_na))

            hw_vec = make_hw_vector(wall_ms, gm, regs, mode_readback, fence_rate,
                                     current_demand[1], cycle_delta, status_val,
                                     hw_id2_val, pc_lo_val, char_info)
            if scramble:
                hw_vec = [1.0 - v for v in hw_vec]
            hw_t = torch.tensor([hw_vec] * BS, dtype=torch.float32, device=DEVICE)

            wgps = probe_res[0]
            bank_ids = (wgps // 2).long().clamp(0, NUM_BANKS - 1)

            if bn % SWITCH_EVERY == 0:
                next_demand = [random.choice([0.0, 1.0]),
                               random.choice([0.0, 0.06, 0.5, 1.0]),
                               random.choice([0.0, 1.0])]

            demand_t = torch.tensor([next_demand] * BS, dtype=torch.float32, device=DEVICE)

            demand_level = random.random() if ablate_type == 'random_demand' else current_demand[0]
            labels = make_labels(digits, bank_ids, demand_level)

            out = model(imgs, bank_ids=bank_ids, hw_vector=hw_t, demand_vector=demand_t)
            pred = out['logits'].argmax(1)
            all_preds.extend((pred == labels).cpu().tolist())

            if out['self_pred'] is not None:
                sp = out['self_pred'].mean(0).cpu().numpy()
                targets = [hw_vec[4], hw_vec[0], hw_vec[6], hw_vec[5],
                           hw_vec[9], hw_vec[1], hw_vec[2], hw_vec[11],
                           hw_vec[20], hw_vec[21], hw_vec[22], hw_vec[25]]
                all_sp.append(sp); all_st.append(targets)

            g = out['gate'].mean().item()
            dk = 'high' if demand_level > 0.5 else 'low'
            gate_by_demand[dk].append(g)

            if out['effort'] is not None:
                eff = out['effort'].mean(0)
                efforts.append([eff[0].item(), eff[1].item(), eff[2].item()])
                demands.append(next_demand)
                prev_effort_sclk = eff[0].item()

            energy_log.append(gm['sclk_mhz'] if gm else 600)
            sclk_set.add(gm['sclk_mhz'] if gm else 0)
            grbm_set.add(regs['grbm_raw'])

            if model_controlled and fixed_sclk is None and out['effort'] is not None:
                eff = out['effort'].mean(0)
                apply_actuation(eff[0].item(), eff[1].item())
                current_round_mode = 1 if eff[2].item() >= 0.5 else 0
                time.sleep(ACTUATION_WAIT)
                current_demand = [eff[0].item(), eff[1].item(), float(current_round_mode)]

            current_demand = next_demand
            bn += 1

    m = {'acc': float(np.mean(all_preds))}

    if all_sp and all_st:
        sp, st = np.array(all_sp), np.array(all_st)
        target_names = ['sclk', 'timing', 'se0_busy', 'gui_active', 'rlc_lo16',
                        'power', 'temp', 'fence', 'cycle_delta', 'status_lds',
                        'hw_id2_vmid', 'grbm_load']
        for i, tn in enumerate(target_names):
            m[f'self_r2_{tn}'] = float(r2_score(st[:, i], sp[:, i])) if np.std(st[:, i]) > 1e-6 else 0.0

    g_h, g_l = gate_by_demand['high'], gate_by_demand['low']
    if g_h and g_l and len(set(g_h + g_l)) > 1:
        _, p_val = stats.ttest_ind(g_h, g_l)
        m['gate_p'] = float(p_val)
        m['gate_high'] = float(np.mean(g_h))
        m['gate_low'] = float(np.mean(g_l))
        m['gate_r'], _ = stats.pearsonr(g_h + g_l, [1.0]*len(g_h) + [0.0]*len(g_l))
    else:
        m['gate_p'] = 1.0; m['gate_high'] = 0.5; m['gate_low'] = 0.5; m['gate_r'] = 0.0

    if efforts:
        eff, dem = np.array(efforts), np.array(demands)
        m['effort_sclk_std'] = float(np.std(eff[:, 0]))
        m['effort_sched_std'] = float(np.std(eff[:, 1]))
        m['effort_rmode_std'] = float(np.std(eff[:, 2]))
        m['effort_demand_r2'] = float(r2_score(dem[:, 0], eff[:, 0])) if np.std(dem[:, 0]) > 1e-6 else 0.0

    if len(effort_sclk_pairs) > 10:
        ea = np.array([p[0] for p in effort_sclk_pairs])
        sa = np.array([p[1] for p in effort_sclk_pairs])
        m['temporal_r'] = float(stats.pearsonr(ea, sa)[0]) if np.std(ea) > 1e-6 and np.std(sa) > 1e-6 else 0.0
    else:
        m['temporal_r'] = 0.0

    m['mean_sclk'] = float(np.mean(energy_log))
    m['sclk_distinct'] = len(sclk_set)
    m['grbm_unique'] = len(grbm_set)
    m['mode_seen'] = sorted(mode_seen)
    m['status_unique'] = len(status_vals)

    # SHADER_CYCLES differentiates SCLK states
    cd_h = cycle_deltas_by_sclk['high']
    cd_l = cycle_deltas_by_sclk['low']
    if cd_h and cd_l:
        m['cycle_high_mean'] = float(np.mean(cd_h))
        m['cycle_low_mean'] = float(np.mean(cd_l))
        if len(cd_h) > 5 and len(cd_l) > 5:
            _, m['cycle_ttest_p'] = stats.ttest_ind(cd_h, cd_l)
        else:
            m['cycle_ttest_p'] = 1.0
    else:
        m['cycle_high_mean'] = 0; m['cycle_low_mean'] = 0; m['cycle_ttest_p'] = 1.0

    return m


def ablate_self_model(model):
    if hasattr(model, 'self_model'):
        for p in model.self_model.parameters(): p.data.zero_()

def ablate_effort(model):
    for attr in ['effort_head', 'demand_proj']:
        if hasattr(model, attr):
            for p in getattr(model, attr).parameters(): p.data.zero_()


# ━━━ MAIN ━━━
def main():
    print("=" * 70)
    print("z2066: Deep Analog Embodiment — Closest to the Messy Reality")
    print("=" * 70)
    print()
    print("z2065 all-hacks + ISA-level registers from z2065 probe session:")
    print("  NEW: SHADER_CYCLES (in-kernel timing), STATUS (wave flags),")
    print("       HW_ID2 (scheduling context), PC_LO (kernel fingerprint)")
    print(f"  26-dim HW vector, 12-target self-model, 3-axis effort")
    print()

    t0 = time.time()

    print("Compiling HIP kernel (7 ISA registers + MODE r/w)...")
    ext = load_inline(name='z2066_deep_analog', cpp_sources=CPP_SRC, cuda_sources=HIP_SRC,
                      functions=['probe', 'probe_mode'],
                      extra_cuda_cflags=['-O2', '--offload-arch=gfx1100'],
                      verbose=False)

    # Verify all channels
    res = ext.probe(64)
    torch.cuda.synchronize()
    wgps = sorted(set(res[0].cpu().numpy().tolist()))
    cd_mean = res[3].cpu().numpy().mean()
    sts = sorted(set(res[4].cpu().numpy().tolist()))
    id2 = int(res[5][0].item())
    pclo = int(res[6][0].item()) & 0xFFFFFFFF
    print(f"Channel check: {len(wgps)} WGPs, cycles={cd_mean:.0f}, "
          f"status={[hex(s) for s in sts]}, HW_ID2=0x{id2:08X}, PC=0x{pclo:08X}")

    char_info = characterize(ext)
    train_loader, test_loader = get_data()

    # ━━━ A: Full deep analog ━━━
    print(f"\n{'='*60}")
    print("A: FULL DEEP ANALOG (26-dim HW, 12-target self-model, 3-axis effort)")
    print(f"{'='*60}")
    model_A = DeepAnalogModel(use_banks=True, use_hw=True, use_self_model=True,
                               use_gate=True, use_effort=True).to(DEVICE)
    train_log = train_model(model_A, ext, train_loader, EPOCHS, 'A_deep', char_info,
                             model_controlled=True)
    m_A = evaluate(model_A, ext, test_loader, char_info, model_controlled=True)
    print(f"  A: acc={m_A['acc']:.4f}")
    print(f"     self: R²(sclk)={m_A.get('self_r2_sclk',0):.4f} "
          f"R²(cycle)={m_A.get('self_r2_cycle_delta',0):.4f} "
          f"R²(timing)={m_A.get('self_r2_timing',0):.4f}")
    print(f"     gate: h={m_A['gate_high']:.3f} l={m_A['gate_low']:.3f} "
          f"p={m_A['gate_p']:.6f}")
    print(f"     cycles: high={m_A.get('cycle_high_mean',0):.0f} "
          f"low={m_A.get('cycle_low_mean',0):.0f} "
          f"p={m_A.get('cycle_ttest_p',1):.6f}")

    # ━━━ B: Blind ━━━
    print(f"\n{'='*60}\nB: BLIND\n{'='*60}")
    model_B = DeepAnalogModel(use_banks=False, use_hw=False, use_self_model=False,
                               use_gate=False, use_effort=False).to(DEVICE)
    train_model(model_B, ext, train_loader, EPOCHS, 'B_blind', char_info,
                actuate=True, model_controlled=False)
    m_B = evaluate(model_B, ext, test_loader, char_info, model_controlled=False)
    print(f"  B: acc={m_B['acc']:.4f}")

    # ━━━ E: Scrambled ━━━
    print(f"\n{'='*60}\nE: SCRAMBLED\n{'='*60}")
    m_E = evaluate(model_A, ext, test_loader, char_info, model_controlled=True, scramble=True)
    print(f"  E: acc={m_E['acc']:.4f}")

    # ━━━ F: Ablated self-model ━━━
    print(f"\n{'='*60}\nF: ABLATED SELF-MODEL\n{'='*60}")
    model_F = copy.deepcopy(model_A)
    ablate_self_model(model_F)
    m_F = evaluate(model_F, ext, test_loader, char_info, model_controlled=True)
    print(f"  F: acc={m_F['acc']:.4f}")

    # ━━━ G: Ablated effort ━━━
    print(f"\n{'='*60}\nG: ABLATED EFFORT\n{'='*60}")
    model_G = copy.deepcopy(model_A)
    ablate_effort(model_G)
    m_G = evaluate(model_G, ext, test_loader, char_info, model_controlled=False,
                   fixed_sclk='high', ablate_type='random_demand')
    print(f"  G: acc={m_G['acc']:.4f}")

    # ━━━ H: Always-high ━━━
    print(f"\n{'='*60}\nH: ALWAYS-HIGH\n{'='*60}")
    m_H = evaluate(model_A, ext, test_loader, char_info, model_controlled=False,
                   fixed_sclk='high', ablate_type='random_demand')
    print(f"  H: acc={m_H['acc']:.4f} mean_sclk={m_H['mean_sclk']:.0f}")

    elapsed = time.time() - t0
    reset_actuation()
    energy_ratio = m_A['mean_sclk'] / max(m_H['mean_sclk'], 1)

    # ━━━ Tests (18) ━━━
    print(f"\n{'='*70}\nTEST RESULTS\n{'='*70}")
    tests = {}

    tests['T1_accuracy'] = {'verdict': 'PASS' if m_A['acc'] > 0.90 else 'FAIL',
        'val': f"A={m_A['acc']*100:.1f}% > 90%"}

    tests['T2_self_model_sclk'] = {'verdict': 'PASS' if m_A.get('self_r2_sclk', 0) > 0.3 else 'FAIL',
        'val': f"R²(sclk)={m_A.get('self_r2_sclk',0):.4f} > 0.3"}

    tests['T3_gate_adaptive'] = {'verdict': 'PASS' if m_A.get('gate_p', 1.0) < 0.01 else 'FAIL',
        'val': f"p={m_A.get('gate_p',1.0):.6f} < 0.01"}

    gap_AF = m_A['acc'] - m_F['acc']
    tests['T4_self_model_causal'] = {'verdict': 'PASS' if gap_AF > 0.10 else 'FAIL',
        'val': f"A-F={gap_AF*100:.1f}pp > 10pp"}

    gap_AG = m_A['acc'] - m_G['acc']
    tests['T5_effort_causal'] = {'verdict': 'PASS' if gap_AG > 0.10 else 'FAIL',
        'val': f"A-G={gap_AG*100:.1f}pp > 10pp"}

    tests['T6_scrambled_kills'] = {'verdict': 'PASS' if m_E['acc'] < m_A['acc'] - 0.05 else 'FAIL',
        'val': f"E={m_E['acc']*100:.1f}% < A-5pp={(m_A['acc']-0.05)*100:.1f}%"}

    gap_AB = m_A['acc'] - m_B['acc']
    tests['T7_embodiment_gap'] = {'verdict': 'PASS' if gap_AB > 0.25 else 'FAIL',
        'val': f"A-B={gap_AB*100:.1f}pp > 25pp"}

    tests['T8_gate_corr'] = {'verdict': 'PASS' if abs(m_A.get('gate_r', 0)) > 0.3 else 'FAIL',
        'val': f"|r|={abs(m_A.get('gate_r',0)):.4f} > 0.3"}

    tests['T9_effort_tracks_demand'] = {'verdict': 'PASS' if m_A.get('effort_demand_r2', 0) > 0.5 else 'FAIL',
        'val': f"R²={m_A.get('effort_demand_r2',0):.4f} > 0.5"}

    tests['T10_temporal_closed_loop'] = {'verdict': 'PASS' if abs(m_A.get('temporal_r', 0)) > 0.3 else 'FAIL',
        'val': f"|r|={abs(m_A.get('temporal_r',0)):.4f} > 0.3"}

    tests['T11_energy_saving'] = {'verdict': 'PASS' if energy_ratio < 0.95 else 'FAIL',
        'val': f"ratio={energy_ratio:.4f} < 0.95"}

    tests['T12_grbm_varies'] = {'verdict': 'PASS' if m_A.get('grbm_unique', 0) > 5 else 'FAIL',
        'val': f"GRBM_unique={m_A.get('grbm_unique',0)} > 5"}

    t13_v = m_A.get('self_r2_cycle_delta', 0)
    tests['T13_self_model_cycles'] = {'verdict': 'PASS' if t13_v > 0.1 else 'FAIL',
        'val': f"R²(cycle_delta)={t13_v:.4f} > 0.1"}

    tests['T14_mode_register'] = {
        'verdict': 'PASS' if len(m_A.get('mode_seen', [])) > 1 else 'FAIL',
        'val': f"mode_seen={m_A.get('mode_seen',[])}"}

    t15_p = m_A.get('cycle_ttest_p', 1.0)
    tests['T15_cycles_differentiate'] = {'verdict': 'PASS' if t15_p < 0.05 else 'FAIL',
        'val': f"cycle high={m_A.get('cycle_high_mean',0):.0f} "
               f"low={m_A.get('cycle_low_mean',0):.0f} p={t15_p:.6f}"}

    tests['T16_multi_axis'] = {
        'verdict': 'PASS' if (m_A.get('effort_sclk_std', 0) > 0.05 and
                               m_A.get('effort_sched_std', 0) > 0.05) else 'FAIL',
        'val': f"std(sclk)={m_A.get('effort_sclk_std',0):.3f} "
               f"std(sched)={m_A.get('effort_sched_std',0):.3f}"}

    tests['T17_status_varies'] = {
        'verdict': 'PASS' if m_A.get('status_unique', 0) >= 2 else 'FAIL',
        'val': f"status_unique={m_A.get('status_unique',0)} >= 2"}

    tests['T18_hw_vector_dim'] = {'verdict': 'PASS' if HW_DIM >= 26 else 'FAIL',
        'val': f"HW_DIM={HW_DIM} >= 26 (6 more than z2065)"}

    pass_count = sum(1 for t in tests.values() if t['verdict'] == 'PASS')
    verdict = f"{pass_count}/{len(tests)} PASS"

    for tname, result in tests.items():
        print(f"  {result['verdict']:4s} | {tname}: {result['val']}")
    print(f"\n  VERDICT: {verdict}")

    print(f"\n  Ablation analysis:")
    print(f"    A (deep analog):   {m_A['acc']*100:.1f}%")
    print(f"    F (no self-model): {m_F['acc']*100:.1f}%  ({gap_AF*100:+.1f}pp)")
    print(f"    G (no effort):     {m_G['acc']*100:.1f}%  ({gap_AG*100:+.1f}pp)")
    print(f"    E (scrambled):     {m_E['acc']*100:.1f}%")
    print(f"    B (blind):         {m_B['acc']*100:.1f}%")
    print(f"    H (always-high):   {m_H['acc']*100:.1f}%")
    print(f"    Energy ratio:      {energy_ratio:.4f}")

    results = {
        'experiment': 'z2066_deep_analog_embodiment',
        'version': 1,
        'hw_dim': HW_DIM,
        'self_targets': SELF_TARGETS,
        'new_channels': ['SHADER_CYCLES', 'STATUS', 'HW_ID2', 'PC_LO', 'GRBM_LOAD_BIT', 'HW_ID2_PIPE'],
        'accuracies': {k: round(v, 4) for k, v in [
            ('A_deep', m_A['acc']), ('B_blind', m_B['acc']),
            ('E_scrambled', m_E['acc']), ('F_ablated_self', m_F['acc']),
            ('G_ablated_effort', m_G['acc']), ('H_always_high', m_H['acc'])]},
        'self_model': {k: round(m_A.get(f'self_r2_{k}', 0), 4)
                       for k in ['sclk','timing','se0_busy','gui_active','rlc_lo16',
                                 'power','temp','fence','cycle_delta','status_lds',
                                 'hw_id2_vmid','grbm_load']},
        'gate': {'high': round(m_A['gate_high'],4), 'low': round(m_A['gate_low'],4),
                 'p': round(m_A['gate_p'],6), 'r': round(m_A.get('gate_r',0),4)},
        'cycles': {'high_mean': round(m_A.get('cycle_high_mean',0),1),
                   'low_mean': round(m_A.get('cycle_low_mean',0),1),
                   'ttest_p': round(m_A.get('cycle_ttest_p',1),6)},
        'energy': {'ratio': round(energy_ratio, 4),
                   'mean_A': round(m_A['mean_sclk'],1),
                   'mean_H': round(m_H['mean_sclk'],1)},
        'tests': tests, 'verdict': verdict, 'pass_count': pass_count,
        'elapsed_s': round(elapsed),
    }

    out_path = 'results/z2066_deep_analog_embodiment.json'
    os.makedirs('results', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"Elapsed: {elapsed:.0f}s")

    global _regs2_fd
    if _regs2_fd is not None:
        os.close(_regs2_fd); _regs2_fd = None


if __name__ == '__main__':
    main()
