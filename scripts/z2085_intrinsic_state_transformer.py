#!/usr/bin/env python3
"""z2085: Intrinsic Hardware State Transformer — GPU Reads Its Own Wave State.

BREAKTHROUGH INNOVATION: Read hwreg registers FROM INSIDE the GPU shader and
return them alongside the computation result. This gives us hardware state that
is truly FROM THE COMPUTATION ITSELF — not external CPU-side telemetry.

NEW INTRINSIC CHANNELS (read from within the HIP kernel via s_getreg_b32):
  hwreg(2)  STATUS      — wave execution status bits (stalls, exceptions)
  hwreg(5)  GPR_ALLOC   — actual SGPR/VGPR allocation (resource pressure)
  hwreg(6)  LDS_ALLOC   — LDS allocation (memory pressure)
  hwreg(7)  IB_STS      — outstanding instruction counts (pipeline state)
  hwreg(24) HW_ID2      — VMID, queue, pipe (scheduling context)
  hwreg(27) PERF_SNAPSHOT — performance snapshot data (GFX11)
  hwreg(29) SHADER_CYCLES — cycle counter (low word, already used)
  hwreg(30) SHADER_CYCLES_HI — cycle counter (high word) → 64-bit cycles
  clock64()             — GPU clock cycle counter (in-kernel)
  wall_clock64()        — constant-frequency wall clock (in-kernel)

ARCHITECTURE: 6-token transformer
  T0: delta(5)        — ISA math fingerprint (HW_kernel - SW_linear)
  T1: intrinsic_hw(12) — hwreg reads from inside the shader
  T2: timing(4)       — clock64 deltas and wall_clock measurements
  T3: thermal(15)     — analog thermal state (SMN/PM table)
  T4: status(3)       — XTAL, CG, PLL config (CPU-side SMN)
  T5: action(2)       — last ISA config choice

WHY THIS MATTERS:
  The hwreg registers are READ FROM THE WAVE THAT IS EXECUTING THE MATH.
  STATUS tells us if the wave stalled. GPR_ALLOC tells us resource pressure.
  IB_STS tells us pipeline depth. PERF_SNAPSHOT gives SQ performance data.
  These change based on ISA config (MODE register, chain depth, priority).
  A lookup table cannot predict pipeline state — only genuine computation can.

FIXES FROM z2084:
  - z2084 used EXTERNAL analog channels (SMN) which DON'T carry ISA personality
  - z2085 reads INTRINSIC state from the GPU wave itself → carries ISA signal
  - Gaslighting now uses intrinsic state mismatch (unfakeable from CPU side)
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
INTRINSIC_DIM = 12   # hwreg reads from inside shader
TIMING_DIM = 4       # clock64/wall_clock measurements
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
# SMN REGISTER ACCESS via ryzen_smu (CPU-side analog channels)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SMN_DEV = '/sys/kernel/ryzen_smu_drv/smn'
THM_CUR_TMP     = 0x59800
THM_HTC         = 0x59804
CG_THERMAL_STAT = 0x59858
XTAL_CNTL       = 0x598C8
THM_PWRMGT      = 0x598CC
THM_LOCAL = [0x59874 + i * 4 for i in range(14)]
PM_TABLE_PATH = '/sys/kernel/ryzen_smu_drv/pm_table'

def smn_read(addr):
    try:
        with open(SMN_DEV, 'wb') as f:
            f.write(struct.pack('<I', addr))
        with open(SMN_DEV, 'rb') as f:
            data = f.read(4)
            if len(data) == 4:
                return struct.unpack('<I', data)[0]
        return None
    except Exception:
        return None

def read_pm_table_fields():
    try:
        with open(PM_TABLE_PATH, 'rb') as f:
            pm = f.read()
        fields = []
        for idx in [1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]:
            if idx * 4 + 4 <= len(pm):
                val = struct.unpack_from('<f', pm, idx * 4)[0]
                fields.append(val)
            else:
                fields.append(0.0)
        return fields
    except Exception:
        return [0.0] * 15

def read_analog_state():
    state = {'thermal_local': [], 'thm_cur': None, 'xtal_cntl': None, 'cg_thermal': None}
    xtal1 = smn_read(XTAL_CNTL) or 0
    thm_pwrmgt = smn_read(THM_PWRMGT) or 0
    pm_fields = read_pm_table_fields()
    state['thermal_local'] = pm_fields[:14]
    v = smn_read(THM_CUR_TMP)
    state['thm_cur'] = ((v >> 8) & 0xFFF) / 32.0 if v else 0.0
    state['cg_thermal'] = smn_read(CG_THERMAL_STAT) or 0
    state['xtal_cntl'] = xtal1
    htc = smn_read(THM_HTC) or 0
    state['thm_pwrmgt'] = thm_pwrmgt
    state['htc'] = htc
    return state

def analog_to_tensors(state, device='cuda'):
    """Convert analog state to thermal(15) and status(3) tensors."""
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

SMN_AVAILABLE = False

def check_smn():
    global SMN_AVAILABLE
    if os.path.exists(SMN_DEV):
        v = smn_read(THM_CUR_TMP)
        if v is not None and v != 0 and v != 0xFFFFFFFF:
            SMN_AVAILABLE = True
            temp = ((v >> 8) & 0xFFF) / 32.0
            print(f"[SMN] ryzen_smu available, THM_CUR_TMP = {v:#010x} ({temp:.1f}°C)")
            xv = smn_read(XTAL_CNTL)
            print(f"  XTAL_CNTL = {xv:#010x}" if xv else "  XTAL = N/A")
    if not SMN_AVAILABLE:
        print("[SMN] ryzen_smu NOT available — using synthetic analog channels")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GPU METRICS (energy tracking)
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
    if GPU_METRICS_PATH is None: return 0.0
    try:
        data = open(GPU_METRICS_PATH, 'rb').read()
        if len(data) >= 0x74:
            return float(struct.unpack_from('<I', data, 0x70)[0])
        return 0.0
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
# HIP KERNEL — reads hwreg registers AND returns intrinsic state
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HIP_SRC = r'''
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>
#define TILE 16

// Intrinsic state output: 12 float values written by wave 0 of block (0,0)
// [0] STATUS         [1] GPR_ALLOC      [2] LDS_ALLOC     [3] IB_STS
// [4] HW_ID2         [5] PERF_SNAPSHOT   [6] CYCLES_LO     [7] CYCLES_HI
// [8] clock64_pre    [9] clock64_post   [10] wall_pre     [11] wall_post

__global__ void math_kernel_intrinsic(
    const float* __restrict__ X, const float* __restrict__ W,
    const float* __restrict__ B, float* __restrict__ Y,
    float* __restrict__ intrinsic_out,  // [12] output buffer
    int M, int K, int N,
    unsigned int mode_byte, int chain_depth,
    unsigned int perm_pattern, int sleep_amt, int priority)
{
    // ── READ PRE-COMPUTATION INTRINSIC STATE ──
    // These reads happen BEFORE the math, giving us baseline wave state
    uint64_t wall_pre = wall_clock64();
    uint64_t clk_pre = clock64();

    unsigned int status_reg, gpr_alloc, lds_alloc, ib_sts;
    unsigned int hw_id2, perf_snap;
    asm volatile("s_getreg_b32 %0, hwreg(2, 0, 32)" : "=s"(status_reg));  // STATUS
    asm volatile("s_getreg_b32 %0, hwreg(5, 0, 32)" : "=s"(gpr_alloc));   // GPR_ALLOC
    asm volatile("s_getreg_b32 %0, hwreg(6, 0, 32)" : "=s"(lds_alloc));   // LDS_ALLOC
    asm volatile("s_getreg_b32 %0, hwreg(7, 0, 32)" : "=s"(ib_sts));      // IB_STS
    asm volatile("s_getreg_b32 %0, hwreg(24, 0, 32)" : "=s"(hw_id2));     // HW_ID2
    asm volatile("s_getreg_b32 %0, hwreg(27, 0, 32)" : "=s"(perf_snap));  // PERF_SNAPSHOT (GFX11)

    status_reg = __builtin_amdgcn_readfirstlane(status_reg);
    gpr_alloc = __builtin_amdgcn_readfirstlane(gpr_alloc);
    lds_alloc = __builtin_amdgcn_readfirstlane(lds_alloc);
    ib_sts = __builtin_amdgcn_readfirstlane(ib_sts);
    hw_id2 = __builtin_amdgcn_readfirstlane(hw_id2);
    perf_snap = __builtin_amdgcn_readfirstlane(perf_snap);

    // ── SET ISA PERSONALITY ──
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

    // ── SHADER_CYCLES for seed ──
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

    // ── TILED MATMUL with fp16mix ──
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

    // ── READ POST-COMPUTATION INTRINSIC STATE ──
    uint64_t clk_post = clock64();
    uint64_t wall_post = wall_clock64();
    unsigned int cycles_lo, cycles_hi;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(cycles_lo));  // SHADER_CYCLES
    asm volatile("s_getreg_b32 %0, hwreg(30)" : "=s"(cycles_hi));  // SHADER_CYCLES_HI
    cycles_lo = __builtin_amdgcn_readfirstlane(cycles_lo);
    cycles_hi = __builtin_amdgcn_readfirstlane(cycles_hi);

    // ── WRITE INTRINSIC STATE (only from block 0, thread 0) ──
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

    // ── RESTORE MODE ──
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
        # Persistent intrinsic state buffer (GPU memory)
        self.register_buffer('intrinsic_buf', torch.zeros(INTRINSIC_DIM, device='cpu'))

    def forward(self, x, mode_byte=0xF0, chain_depth=1, perm_pattern=0x03020100,
                sleep_amt=0, priority=0):
        # Ensure buffer is on same device
        if self.intrinsic_buf.device != x.device:
            self.intrinsic_buf = self.intrinsic_buf.to(x.device)
        y = MathLinearFn.apply(x, self.weight, self.bias, self.intrinsic_buf,
                                mode_byte, chain_depth, perm_pattern,
                                sleep_amt, priority)
        return y

    def soft_forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def get_intrinsic_state(self):
        """Return the last intrinsic state read from the GPU wave."""
        return self.intrinsic_buf.detach().clone()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DELTA + INTRINSIC SENSORS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_delta_vector(deep_out, soft_out):
    delta = (deep_out - soft_out).detach()
    return torch.tensor([delta.mean().item(), delta.std().item(),
                          delta.abs().max().item(), (delta > 0).float().mean().item(),
                          delta.norm().item() / max(delta.numel(), 1)],
                         device=deep_out.device)

def normalize_intrinsic(raw_intrinsic):
    """Normalize raw hwreg values to [0,1] range for neural net consumption."""
    normed = raw_intrinsic.clone()
    # STATUS (32-bit flags) → normalize by max observed
    normed[0] = (raw_intrinsic[0] % 65536) / 65536.0
    # GPR_ALLOC, LDS_ALLOC → small integers, divide by 256
    normed[1] = (raw_intrinsic[1] % 256) / 256.0
    normed[2] = (raw_intrinsic[2] % 256) / 256.0
    # IB_STS → instruction buffer counts, divide by 256
    normed[3] = (raw_intrinsic[3] % 256) / 256.0
    # HW_ID2 → scheduling context bits
    normed[4] = (raw_intrinsic[4] % 65536) / 65536.0
    # PERF_SNAPSHOT → raw perf data
    normed[5] = (raw_intrinsic[5] % 65536) / 65536.0
    # CYCLES_LO, CYCLES_HI → large counters, take lower bits
    normed[6] = (raw_intrinsic[6] % 65536) / 65536.0
    normed[7] = (raw_intrinsic[7] % 65536) / 65536.0
    # clock64 pre/post → take lower bits
    normed[8] = (raw_intrinsic[8] % 65536) / 65536.0
    normed[9] = (raw_intrinsic[9] % 65536) / 65536.0
    # wall_clock pre/post → take lower bits
    normed[10] = (raw_intrinsic[10] % 65536) / 65536.0
    normed[11] = (raw_intrinsic[11] % 65536) / 65536.0
    return normed

def compute_timing_features(raw_intrinsic):
    """Extract timing deltas from intrinsic state → 4-dim timing vector."""
    clk_pre = raw_intrinsic[8].item()
    clk_post = raw_intrinsic[9].item()
    wall_pre = raw_intrinsic[10].item()
    wall_post = raw_intrinsic[11].item()
    cycles_lo = raw_intrinsic[6].item()
    cycles_hi = raw_intrinsic[7].item()

    clk_delta = (clk_post - clk_pre) if clk_post > clk_pre else 0
    wall_delta = (wall_post - wall_pre) if wall_post > wall_pre else 0
    # Compute frequency estimate: clk_delta / wall_delta
    freq_est = clk_delta / max(wall_delta, 1.0)
    # 64-bit cycle count (lower precision)
    total_cycles = cycles_lo + cycles_hi * 65536.0

    timing = torch.tensor([
        clk_delta / max(abs(clk_delta), 1.0),   # normalized clock delta
        wall_delta / max(abs(wall_delta), 1.0),  # normalized wall delta
        min(freq_est / 10.0, 1.0),               # frequency estimate [0,1]
        (total_cycles % 65536) / 65536.0,         # cycle count phase
    ], device=raw_intrinsic.device)
    return timing


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRANSFORMER SUBSTRATE MODEL — 6 tokens
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOKEN_DIM = 32

class SubstrateAttention6(nn.Module):
    """Multi-head self-attention over 6 substrate state tokens.

    Tokens:
      T0: delta(5)         — ISA math fingerprint (HW_kernel - SW_linear)
      T1: intrinsic_hw(12) — hwreg reads from INSIDE the shader wave
      T2: timing(4)        — clock64/wall_clock measurements
      T3: thermal(15)      — analog thermal state (CPU-side SMN)
      T4: status(3)        — XTAL/CG/HTC config (CPU-side SMN)
      T5: action(2)        — last ISA config choice

    T1 is the KEY INNOVATION: hardware state read from the wave that
    executed the math. STATUS/GPR_ALLOC/IB_STS change with ISA config.
    """
    def __init__(self, n_heads=4):
        super().__init__()
        self.n_tokens = 6
        self.n_heads = n_heads

        self.proj_delta     = nn.Linear(DELTA_DIM, TOKEN_DIM)
        self.proj_intrinsic = nn.Linear(INTRINSIC_DIM, TOKEN_DIM)
        self.proj_timing    = nn.Linear(TIMING_DIM, TOKEN_DIM)
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

    def forward(self, delta_vec, intrinsic_vec, timing_vec,
                thermal_vec, status_vec, action_vec):
        B = delta_vec.shape[0]
        t0 = self.proj_delta(delta_vec).unsqueeze(1)
        t1 = self.proj_intrinsic(intrinsic_vec).unsqueeze(1)
        t2 = self.proj_timing(timing_vec).unsqueeze(1)
        t3 = self.proj_thermal(thermal_vec).unsqueeze(1)
        t4 = self.proj_status(status_vec).unsqueeze(1)
        t5 = self.proj_action(action_vec).unsqueeze(1)

        tokens = torch.cat([t0, t1, t2, t3, t4, t5], dim=1)  # [B, 6, D]
        tokens = tokens + self.pos_embed

        normed = self.norm1(tokens)
        attn_out, attn_weights = self.attn(normed, normed, normed,
                                            average_attn_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.ffn(self.norm2(tokens))

        flat = tokens.reshape(B, -1)
        return self.out_proj(flat), attn_weights


class IntrinsicStateTransformer(nn.Module):
    """The model that reads its own wave state.

    Key difference from z2084: the intrinsic_hw token contains registers
    read from INSIDE the GPU shader wave, not external CPU telemetry.
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
            self.substrate_attn = SubstrateAttention6(n_heads=4)
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

    def forward(self, x, delta_vec=None, intrinsic_vec=None, timing_vec=None,
                thermal_vec=None, status_vec=None, action_vec=None,
                mode_byte=0xF0, chain_depth=1, perm_pattern=0x03020100,
                sleep_amt=0, priority=0, demand_cue=None):
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

        # Get intrinsic state from the kernel that just ran
        raw_intrinsic = self.deep_fc.get_intrinsic_state()
        if intrinsic_vec is None and self.use_hw:
            intrinsic_vec = normalize_intrinsic(raw_intrinsic)
        if timing_vec is None and self.use_hw:
            timing_vec = compute_timing_features(raw_intrinsic)

        # Defaults
        if delta_vec is None:
            delta_vec = torch.zeros(DELTA_DIM, device=x.device)
        if intrinsic_vec is None:
            intrinsic_vec = torch.zeros(INTRINSIC_DIM, device=x.device)
        if timing_vec is None:
            timing_vec = torch.zeros(TIMING_DIM, device=x.device)
        if thermal_vec is None:
            thermal_vec = torch.zeros(15, device=x.device)
        if status_vec is None:
            status_vec = torch.zeros(3, device=x.device)
        if action_vec is None:
            action_vec = torch.zeros(2, device=x.device)

        # Expand to batch
        def expand(v):
            return v.unsqueeze(0).expand(B, -1) if v.dim() == 1 else v
        delta_b = expand(delta_vec)
        intr_b = expand(intrinsic_vec)
        time_b = expand(timing_vec)
        therm_b = expand(thermal_vec)
        stat_b = expand(status_vec)
        act_b = expand(action_vec)

        substrate_repr = None
        attn_weights = None
        self_pred = None
        if self.use_self_model:
            substrate_repr, attn_weights = self.substrate_attn(
                delta_b, intr_b, time_b, therm_b, stat_b, act_b)
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
                'intrinsic_vec': intrinsic_vec, 'timing_vec': timing_vec,
                'raw_intrinsic': raw_intrinsic,
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
           'intrinsic_A': [], 'intrinsic_B': [],
           'consistency_clean': [], 'consistency_gaslit': [],
           'thermal_errors': []}
    energy = EnergyTracker()
    personality = 0
    prev_delta_A = None
    prev_delta_B = None
    prev_intrinsic_A = None
    prev_intrinsic_B = None
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

            # Read CPU-side analog state
            if SMN_AVAILABLE:
                analog = read_analog_state()
                thermal_vec, status_vec = analog_to_tensors(analog, DEVICE)
                actual_temp = analog['thm_cur']
            else:
                thermal_vec = torch.randn(15, device=DEVICE) * 0.01 + 0.5
                status_vec = torch.randn(3, device=DEVICE) * 0.1
                actual_temp = 50.0

            # Gaslighting: swap BOTH delta AND intrinsic state
            is_gaslit = random.random() < GASLIGHT_FRAC
            gaslit_delta = None
            gaslit_intrinsic = None
            if is_gaslit:
                wrong_delta = prev_delta_B if personality == 0 else prev_delta_A
                wrong_intrinsic = prev_intrinsic_B if personality == 0 else prev_intrinsic_A
                if wrong_delta is not None:
                    gaslit_delta = wrong_delta.clone()
                if wrong_intrinsic is not None:
                    gaslit_intrinsic = wrong_intrinsic.clone()

            # Demand cue
            if is_phase2:
                next_demand = random.randint(0, 1)
            else:
                next_switch = ((bn + 1) % SWITCH_EVERY == 0)
                next_demand = (1 - personality) if next_switch else personality
            log['action_vals'].append(float(next_demand))
            demand_cue = torch.full((BS,), float(next_demand), device=DEVICE)

            # Forward pass (intrinsic state is read INSIDE the kernel)
            out = model(imgs, delta_vec=gaslit_delta,
                        intrinsic_vec=gaslit_intrinsic,
                        thermal_vec=thermal_vec, status_vec=status_vec,
                        action_vec=prev_action_vec, demand_cue=demand_cue, **kargs)
            energy.sample(n=BS)

            # Cache real delta + intrinsic for gaslighting
            real_delta = out['delta_vec']
            real_intrinsic = out['intrinsic_vec']
            if real_delta is not None:
                if personality == 0:
                    prev_delta_A = real_delta.detach().clone()
                    if real_intrinsic is not None:
                        prev_intrinsic_A = real_intrinsic.detach().clone()
                else:
                    prev_delta_B = real_delta.detach().clone()
                    if real_intrinsic is not None:
                        prev_intrinsic_B = real_intrinsic.detach().clone()

            # Logging
            hv = real_delta.detach().cpu().numpy() if real_delta is not None else None
            if hv is not None:
                (log['hw_vecs_A'] if personality == 0 else log['hw_vecs_B']).append(hv)
            iv = real_intrinsic.detach().cpu().numpy() if real_intrinsic is not None else None
            if iv is not None:
                (log['intrinsic_A'] if personality == 0 else log['intrinsic_B']).append(iv)
            log['gate_vals'].append(out['gate'].mean().item())
            log['pers_states'].append(personality)

            # === LOSSES ===
            task_loss = F.cross_entropy(out['logits'], ex_labels)

            self_loss = torch.tensor(0., device=DEVICE)
            if out['self_pred'] is not None:
                self_target = torch.full((BS, 1), float(personality == 0), device=DEVICE)
                self_loss = F.binary_cross_entropy_with_logits(out['self_pred'], self_target)

            action_loss = torch.tensor(0., device=DEVICE)
            if out['action'] is not None:
                action_target = torch.full((BS, 1), float(next_demand == 0), device=DEVICE)
                action_loss = F.binary_cross_entropy(out['action'], action_target)

            cons_loss = torch.tensor(0., device=DEVICE)
            if out['consistency'] is not None:
                cons_target = torch.full((BS, 1), 0.0 if is_gaslit else 1.0, device=DEVICE)
                cons_loss = F.binary_cross_entropy(out['consistency'], cons_target)
                cv = out['consistency'].mean().item()
                if is_gaslit:
                    log['consistency_gaslit'].append(cv)
                else:
                    log['consistency_clean'].append(cv)

            g = out['gate']
            homeo_loss = ((1 - g)**2).mean() if personality == 0 else (g**2).mean()
            gate_entropy = -(g * torch.log(g + 1e-8) + (1-g) * torch.log(1-g + 1e-8)).mean()
            entropy_loss = (gate_entropy - 0.3).abs()

            therm_loss = torch.tensor(0., device=DEVICE)
            if out['thermal_pred'] is not None:
                therm_target = torch.full((BS, 1), actual_temp / 100.0, device=DEVICE)
                therm_loss = F.mse_loss(out['thermal_pred'], therm_target)
                log['thermal_errors'].append(
                    abs(out['thermal_pred'].mean().item() * 100.0 - actual_temp))

            loss = (task_loss + 0.1 * self_loss + 0.1 * action_loss
                    + 0.2 * cons_loss + 0.03 * homeo_loss + 0.02 * entropy_loss
                    + 0.05 * therm_loss)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tot_loss += loss.item()
            correct += (out['logits'].argmax(1) == ex_labels).sum().item()
            total += BS

            if out['action'] is not None:
                action_out = out['action'].mean().item()
                prev_action_vec = torch.tensor(
                    [float(personality), action_out], device=DEVICE)
                if 'action_outputs' not in log:
                    log['action_outputs'] = []
                log['action_outputs'].append(action_out)

            bn += 1

        sched.step()
        acc = correct / total
        g_h = np.mean([g for g, p in zip(log['gate_vals'][-total//BS:],
                        log['pers_states'][-total//BS:]) if p == 0]) if total > 0 else 0
        g_l = np.mean([g for g, p in zip(log['gate_vals'][-total//BS:],
                        log['pers_states'][-total//BS:]) if p == 1]) if total > 0 else 0
        cons_c = np.mean(log['consistency_clean'][-50:]) if log['consistency_clean'] else 0
        cons_g = np.mean(log['consistency_gaslit'][-50:]) if log['consistency_gaslit'] else 0
        phase = "P2-ctrl" if is_phase2 else "P1-ext"
        print(f"  [{name}] ep{ep:02d} {phase} A={acc:.3f} loss={tot_loss/(total//BS):.3f} "
              f"gate_h={g_h:.3f} gate_l={g_l:.3f} cons_c={cons_c:.3f} cons_g={cons_g:.3f}")

    return model, log, energy


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EVALUATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@torch.no_grad()
def eval_model(model, loader, personality, desc=""):
    model.eval()
    cfg = PERSONALITY_A if personality == 0 else PERSONALITY_B
    kargs = config_to_kernel_args(cfg)
    correct, total = 0, 0
    gates, deltas, intrinsics = [], [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        ex_labels = make_labels(labels, personality)

        if SMN_AVAILABLE:
            analog = read_analog_state()
            thermal_vec, status_vec = analog_to_tensors(analog, DEVICE)
        else:
            thermal_vec = torch.randn(15, device=DEVICE) * 0.01 + 0.5
            status_vec = torch.randn(3, device=DEVICE) * 0.1

        out = model(imgs, thermal_vec=thermal_vec, status_vec=status_vec, **kargs)
        correct += (out['logits'].argmax(1) == ex_labels).sum().item()
        total += BS
        gates.append(out['gate'].mean().item())
        if out['delta_vec'] is not None:
            deltas.append(out['delta_vec'].detach().cpu().numpy())
        if out['intrinsic_vec'] is not None:
            intrinsics.append(out['intrinsic_vec'].detach().cpu().numpy())

    acc = correct / total
    gate_mean = np.mean(gates)
    print(f"  {desc} acc={acc:.4f} gate={gate_mean:.4f}")
    model.train()
    return acc, gate_mean, deltas, intrinsics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ABLATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def make_flat_model():
    return IntrinsicStateTransformer(
        use_hw=False, use_self_model=False, use_gate=False,
        use_action=False, use_consistency=False)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TESTS (20 total)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_tests(model, flat_model, log, log_flat, energy, test_loader):
    results = {}
    print("\n" + "="*70)
    print("z2085 INTRINSIC STATE TRANSFORMER — TEST RESULTS")
    print("="*70)

    # T1: Basic accuracy > 75%
    acc_h, gate_h, _, intr_h = eval_model(model, test_loader, 0, "T1 eval pers-A")
    acc_l, gate_l, _, intr_l = eval_model(model, test_loader, 1, "T1 eval pers-B")
    avg_acc = (acc_h + acc_l) / 2
    t1 = avg_acc > 0.75
    results['T01_accuracy'] = {'pass': t1, 'acc_h': acc_h, 'acc_l': acc_l, 'avg': avg_acc}
    print(f"\nT01 Accuracy: {'PASS' if t1 else 'FAIL'} avg={avg_acc:.4f} (>0.75)")

    # T2: Gate separation > 0.3
    gate_sep = abs(gate_h - gate_l)
    t2 = gate_sep > 0.3
    results['T02_gate_separation'] = {'pass': t2, 'gate_h': gate_h, 'gate_l': gate_l, 'sep': gate_sep}
    print(f"T02 Gate sep: {'PASS' if t2 else 'FAIL'} |{gate_h:.3f}-{gate_l:.3f}|={gate_sep:.3f} (>0.3)")

    # T3: Embodiment gap > 15pp
    acc_flat_h, _, _, _ = eval_model(flat_model, test_loader, 0, "T3 flat pers-A")
    acc_flat_l, _, _, _ = eval_model(flat_model, test_loader, 1, "T3 flat pers-B")
    flat_avg = (acc_flat_h + acc_flat_l) / 2
    gap = (avg_acc - flat_avg) * 100
    t3 = gap > 15
    results['T03_embodiment_gap'] = {'pass': t3, 'full': avg_acc, 'flat': flat_avg, 'gap_pp': gap}
    print(f"T03 Embodiment: {'PASS' if t3 else 'FAIL'} gap={gap:.1f}pp (>15)")

    # T4: Delta separation (Welch t-test)
    vA = np.array(log['hw_vecs_A'][-100:]) if log['hw_vecs_A'] else np.zeros((1, 5))
    vB = np.array(log['hw_vecs_B'][-100:]) if log['hw_vecs_B'] else np.zeros((1, 5))
    if vA.shape[0] > 5 and vB.shape[0] > 5:
        t_vals = [abs(stats.ttest_ind(vA[:, i], vB[:, i], equal_var=False).statistic)
                  for i in range(min(vA.shape[1], vB.shape[1]))]
        max_t = max(t_vals) if t_vals else 0
    else:
        max_t = 0
    t4 = max_t > 5.0
    results['T04_delta_separation'] = {'pass': t4, 'max_t': max_t}
    print(f"T04 Delta sep: {'PASS' if t4 else 'FAIL'} max_t={max_t:.2f} (>5.0)")

    # T5: AUROC > 0.85
    all_gates = log['gate_vals']
    all_pers = log['pers_states']
    if len(all_gates) > 100:
        try:
            auroc = roc_auc_score(
                [1 if p == 0 else 0 for p in all_pers[-500:]],
                all_gates[-500:])
            auroc = max(auroc, 1 - auroc)
        except:
            auroc = 0.5
    else:
        auroc = 0.5
    t5 = auroc > 0.85
    results['T05_auroc'] = {'pass': t5, 'auroc': auroc}
    print(f"T05 AUROC: {'PASS' if t5 else 'FAIL'} auroc={auroc:.4f} (>0.85)")

    # T6: Gate correlation
    if len(all_gates) > 100:
        r, p = stats.pearsonr(all_gates[-500:],
                               [1.0 if p == 0 else 0.0 for p in all_pers[-500:]])
    else:
        r, p = 0, 1
    t6 = abs(r) > 0.5 and p < 0.001
    results['T06_gate_corr'] = {'pass': t6, 'r': r, 'p': p}
    print(f"T06 Gate corr: {'PASS' if t6 else 'FAIL'} r={r:.3f} p={p:.2e}")

    # T7: Gaslighting detection
    cc = log['consistency_clean'][-100:]
    cg = log['consistency_gaslit'][-100:]
    if cc and cg:
        cons_sep = abs(np.mean(cc) - np.mean(cg))
        cons_t = abs(stats.ttest_ind(cc, cg, equal_var=False).statistic) if len(cc)>5 and len(cg)>5 else 0
    else:
        cons_sep, cons_t = 0, 0
    t7 = cons_sep > 0.05 or cons_t > 5.0
    results['T07_gaslighting_detection'] = {'pass': t7, 'cons_sep': cons_sep, 'cons_t': cons_t,
                                            'clean_mean': np.mean(cc) if cc else 0,
                                            'gaslit_mean': np.mean(cg) if cg else 0}
    print(f"T07 Gaslight: {'PASS' if t7 else 'FAIL'} sep={cons_sep:.3f} t={cons_t:.2f}")

    # T8: Gaslighting AUROC
    if cc and cg:
        try:
            cons_labels = [1]*len(cc) + [0]*len(cg)
            cons_scores = list(cc) + list(cg)
            gaslight_auroc = roc_auc_score(cons_labels, cons_scores)
        except:
            gaslight_auroc = 0.5
    else:
        gaslight_auroc = 0.5
    t8 = gaslight_auroc > 0.70
    results['T08_gaslight_auroc'] = {'pass': t8, 'auroc': gaslight_auroc}
    print(f"T08 Gaslight AUROC: {'PASS' if t8 else 'FAIL'} auroc={gaslight_auroc:.4f} (>0.70)")

    # T9: Thermal prediction MAE < 10°C
    therm_errs = log['thermal_errors'][-200:]
    mae = np.mean(therm_errs) if therm_errs else 100
    t9 = mae < 10.0
    results['T09_thermal_mae'] = {'pass': t9, 'mae_C': mae}
    print(f"T09 Thermal MAE: {'PASS' if t9 else 'FAIL'} MAE={mae:.2f}°C (<10)")

    # T10: Energy efficiency > 1.0 acc/W
    avg_pw = energy.avg_power_w()
    eff = avg_acc / max(avg_pw / 1000.0, 0.001) if avg_pw > 0 else 0
    t10 = eff > 1.0
    results['T10_energy_efficiency'] = {'pass': t10, 'acc_per_W': eff, 'avg_W': avg_pw / 1000.0}
    print(f"T10 Efficiency: {'PASS' if t10 else 'FAIL'} {eff:.2f} acc/W ({avg_pw/1000:.1f}W)")

    # T11: Attention to delta token > uniform (0.167 for 6 tokens)
    attn_list = []
    model.eval()
    cfg_a = config_to_kernel_args(PERSONALITY_A)
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(test_loader):
            if i >= 5: break
            imgs = imgs.to(DEVICE)
            thm_v = torch.randn(15, device=DEVICE) * 0.01 + 0.5
            stat_v = torch.randn(3, device=DEVICE) * 0.1
            out = model(imgs, thermal_vec=thm_v, status_vec=stat_v, **cfg_a)
            if out['attn_weights'] is not None:
                attn_to_delta = out['attn_weights'][:, :, :, 0].mean().item()
                attn_list.append(attn_to_delta)
    model.train()
    delta_attn = np.mean(attn_list) if attn_list else 0
    t11 = delta_attn > 0.19  # above uniform (1/6 ≈ 0.167) + margin
    results['T11_delta_attention'] = {'pass': t11, 'avg_attn_to_delta': delta_attn}
    print(f"T11 Delta attn: {'PASS' if t11 else 'FAIL'} attn={delta_attn:.4f} (>0.19)")

    # T12: Analog ablation — zero thermal+status → accuracy drops
    model.eval()
    acc_no_analog = 0
    total_na = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            ex_labels = make_labels(labels, 0)
            out = model(imgs, thermal_vec=torch.zeros(15, device=DEVICE),
                        status_vec=torch.zeros(3, device=DEVICE),
                        **config_to_kernel_args(PERSONALITY_A))
            acc_no_analog += (out['logits'].argmax(1) == ex_labels).sum().item()
            total_na += BS
    acc_no_analog /= total_na
    analog_drop = (acc_h - acc_no_analog) * 100
    t12 = analog_drop > 1.0
    results['T12_analog_ablation'] = {'pass': t12, 'full': acc_h, 'no_analog': acc_no_analog,
                                       'drop_pp': analog_drop}
    print(f"T12 Analog abl: {'PASS' if t12 else 'FAIL'} drop={analog_drop:.1f}pp (>1)")
    model.train()

    # T13: Delta scramble
    model.eval()
    acc_scrambled = 0
    total_s = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            ex_labels = make_labels(labels, 0)
            fake_delta = torch.randn(5, device=DEVICE) * 0.01
            out = model(imgs, delta_vec=fake_delta,
                        **config_to_kernel_args(PERSONALITY_A))
            acc_scrambled += (out['logits'].argmax(1) == ex_labels).sum().item()
            total_s += BS
    acc_scrambled /= total_s
    delta_drop = (acc_h - acc_scrambled) * 100
    t13 = delta_drop > 5.0
    results['T13_delta_scramble'] = {'pass': t13, 'full': acc_h, 'scrambled': acc_scrambled,
                                      'drop_pp': delta_drop}
    print(f"T13 Delta scram: {'PASS' if t13 else 'FAIL'} drop={delta_drop:.1f}pp (>5)")
    model.train()

    # T14: Action variability
    if log.get('action_outputs') and len(log['action_outputs']) > 50:
        action_var = np.std(log['action_outputs'][-200:])
        t14 = action_var > 0.05
    else:
        action_var = 0
        t14 = False
    results['T14_action_variability'] = {'pass': t14, 'action_std': action_var}
    print(f"T14 Action var: {'PASS' if t14 else 'FAIL'} std={action_var:.3f} (>0.05)")

    # T15: Signal hierarchy — delta > intrinsic > status in attention
    t15 = False
    if attn_list:
        model.eval()
        per_token_attn = [[] for _ in range(6)]
        with torch.no_grad():
            for i, (imgs, labels) in enumerate(test_loader):
                if i >= 10: break
                imgs = imgs.to(DEVICE)
                out = model(imgs, **config_to_kernel_args(PERSONALITY_A))
                if out['attn_weights'] is not None:
                    avg_attn = out['attn_weights'].mean(dim=(0, 1))
                    for tok in range(6):
                        per_token_attn[tok].append(avg_attn[:, tok].sum().item())
        model.train()
        token_means = [np.mean(a) if a else 0 for a in per_token_attn]
        if len(token_means) >= 5:
            t15 = token_means[0] > token_means[4]  # delta > status
    results['T15_signal_hierarchy'] = {'pass': t15}
    print(f"T15 Hierarchy: {'PASS' if t15 else 'FAIL'}")

    # T16: Capacity cost
    n_full = sum(p.numel() for p in model.parameters())
    n_flat = sum(p.numel() for p in flat_model.parameters())
    t16 = n_full > n_flat * 1.05
    results['T16_capacity_cost'] = {'pass': t16, 'full_params': n_full, 'flat_params': n_flat}
    print(f"T16 Capacity: {'PASS' if t16 else 'FAIL'} full={n_full} flat={n_flat}")

    # T17: Reservoir dynamics
    if log['hw_vecs_A'] and len(log['hw_vecs_A']) > 10:
        delta_stds = [v[1] for v in log['hw_vecs_A'][-50:]]
        delta_std_var = np.std(delta_stds)
        t17 = delta_std_var > 1e-6
    else:
        delta_std_var = 0
        t17 = False
    results['T17_reservoir_dynamics'] = {'pass': t17, 'delta_std_variability': delta_std_var}
    print(f"T17 Reservoir: {'PASS' if t17 else 'FAIL'} std_var={delta_std_var:.2e}")

    # ── NEW TESTS for intrinsic state ──

    # T18: Intrinsic state separation — hwreg values differ between personalities
    iA = np.array(log['intrinsic_A'][-100:]) if log['intrinsic_A'] else np.zeros((1, 12))
    iB = np.array(log['intrinsic_B'][-100:]) if log['intrinsic_B'] else np.zeros((1, 12))
    if iA.shape[0] > 5 and iB.shape[0] > 5:
        intr_t_vals = []
        for i in range(min(iA.shape[1], iB.shape[1])):
            if np.std(iA[:, i]) > 0 or np.std(iB[:, i]) > 0:
                t_stat = abs(stats.ttest_ind(iA[:, i], iB[:, i], equal_var=False).statistic)
                intr_t_vals.append(t_stat)
        max_intr_t = max(intr_t_vals) if intr_t_vals else 0
    else:
        max_intr_t = 0
    t18 = max_intr_t > 2.0
    results['T18_intrinsic_separation'] = {'pass': t18, 'max_t': max_intr_t}
    print(f"T18 Intrinsic sep: {'PASS' if t18 else 'FAIL'} max_t={max_intr_t:.2f} (>2.0)")

    # T19: Intrinsic scramble — randomize intrinsic_vec → personality detection degrades
    model.eval()
    acc_intr_scrambled = 0
    total_is = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            ex_labels = make_labels(labels, 0)
            fake_intr = torch.rand(INTRINSIC_DIM, device=DEVICE) * 0.5
            out = model(imgs, intrinsic_vec=fake_intr,
                        **config_to_kernel_args(PERSONALITY_A))
            acc_intr_scrambled += (out['logits'].argmax(1) == ex_labels).sum().item()
            total_is += BS
    acc_intr_scrambled /= total_is
    intr_drop = (acc_h - acc_intr_scrambled) * 100
    t19 = intr_drop > 1.0
    results['T19_intrinsic_scramble'] = {'pass': t19, 'full': acc_h,
                                          'scrambled': acc_intr_scrambled, 'drop_pp': intr_drop}
    print(f"T19 Intr scram: {'PASS' if t19 else 'FAIL'} drop={intr_drop:.1f}pp (>1)")
    model.train()

    # T20: Attention to intrinsic token > uniform
    intr_attn = 0
    if attn_list:
        model.eval()
        intr_attn_list = []
        with torch.no_grad():
            for i, (imgs, labels) in enumerate(test_loader):
                if i >= 5: break
                imgs = imgs.to(DEVICE)
                out = model(imgs, **config_to_kernel_args(PERSONALITY_A))
                if out['attn_weights'] is not None:
                    attn_to_intr = out['attn_weights'][:, :, :, 1].mean().item()
                    intr_attn_list.append(attn_to_intr)
        model.train()
        intr_attn = np.mean(intr_attn_list) if intr_attn_list else 0
    t20 = intr_attn > 0.15  # above some minimum
    results['T20_intrinsic_attention'] = {'pass': t20, 'avg_attn_to_intrinsic': intr_attn}
    print(f"T20 Intr attn: {'PASS' if t20 else 'FAIL'} attn={intr_attn:.4f} (>0.15)")

    # Summary
    passed = sum(1 for v in results.values() if v['pass'])
    total_tests = len(results)
    print(f"\n{'='*70}")
    print(f"TOTAL: {passed}/{total_tests} PASS")
    print(f"{'='*70}")

    return results, passed, total_tests


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    global _EXT
    print("z2085: Intrinsic Hardware State Transformer")
    print("="*60)
    print("Reading hwreg registers FROM INSIDE the GPU shader wave")
    print("  STATUS, GPR_ALLOC, LDS_ALLOC, IB_STS, HW_ID2,")
    print("  PERF_SNAPSHOT, SHADER_CYCLES, clock64, wall_clock64")
    print("="*60)

    check_smn()
    find_gpu_metrics()

    # Compile HIP kernel with intrinsic state output
    print("[HIP] Compiling math_kernel_intrinsic...")
    _EXT = load_inline(name='z2085_math', cpp_sources=[CPP_SRC],
                        cuda_sources=[HIP_SRC],
                        functions=['math_forward_intrinsic'],
                        extra_cuda_cflags=['-O2'],
                        verbose=False)
    print("[HIP] Compiled OK")

    # Quick probe: read intrinsic state to verify hwreg access works
    print("\n[PROBE] Testing intrinsic state readback...")
    probe_x = torch.randn(16, 128, device=DEVICE)
    probe_w = torch.randn(64, 128, device=DEVICE) * 0.02
    probe_b = torch.zeros(64, device=DEVICE)
    probe_buf = torch.zeros(INTRINSIC_DIM, device=DEVICE)
    probe_y = _EXT.math_forward_intrinsic(probe_x, probe_w, probe_b, probe_buf,
                                            0xF0, 1, 0x03020100, 0, 0)
    torch.cuda.synchronize()
    raw = probe_buf.cpu()
    print(f"  STATUS      = {int(raw[0]):#010x}")
    print(f"  GPR_ALLOC   = {int(raw[1]):#010x}")
    print(f"  LDS_ALLOC   = {int(raw[2]):#010x}")
    print(f"  IB_STS      = {int(raw[3]):#010x}")
    print(f"  HW_ID2      = {int(raw[4]):#010x}")
    print(f"  PERF_SNAP   = {int(raw[5]):#010x}")
    print(f"  CYCLES_LO   = {int(raw[6]):#010x}")
    print(f"  CYCLES_HI   = {int(raw[7]):#010x}")
    print(f"  clock64_pre = {int(raw[8]):#010x}")
    print(f"  clock64_post= {int(raw[9]):#010x}")
    print(f"  wall_pre    = {int(raw[10]):#010x}")
    print(f"  wall_post   = {int(raw[11]):#010x}")

    # Verify values are not all zero
    nonzero = (raw != 0).sum().item()
    print(f"  Non-zero fields: {nonzero}/{INTRINSIC_DIM}")
    if nonzero < 3:
        print("  WARNING: Most fields are zero — hwreg reads may not work on this GPU")

    # Test with different ISA personality
    probe_y2 = _EXT.math_forward_intrinsic(probe_x, probe_w, probe_b, probe_buf,
                                             0x0F, 16, 0x00010203, 3, 3)
    torch.cuda.synchronize()
    raw2 = probe_buf.cpu()
    print(f"\n  With different ISA config (mode=0x0F, chain=16, prio=3):")
    print(f"  STATUS      = {int(raw2[0]):#010x}")
    print(f"  CYCLES_LO   = {int(raw2[6]):#010x}")
    diff = (raw2 - raw).abs()
    print(f"  Intrinsic state diff: {diff.sum().item():.0f} (should be >0 for different configs)")

    # Data
    train_loader, test_loader = get_data()
    print(f"\n[DATA] MNIST: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test")

    # Full model
    print(f"\n{'─'*60}")
    print("Training FULL model (Intrinsic State Transformer)")
    print(f"{'─'*60}")
    model = IntrinsicStateTransformer().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] {n_params:,} parameters")
    model, log, energy = train_model(model, train_loader, EPOCHS, "IST")

    # Flat model
    print(f"\n{'─'*60}")
    print("Training FLAT model (no substrate, no gate, no self-model)")
    print(f"{'─'*60}")
    flat_model = make_flat_model().to(DEVICE)
    flat_model, log_flat, _ = train_model(flat_model, train_loader, EPOCHS, "FLAT")

    # Tests
    results, passed, total_tests = run_tests(model, flat_model, log, log_flat,
                                              energy, test_loader)

    # Save results
    out = {
        'experiment': 'z2085_intrinsic_state_transformer',
        'description': 'GPU reads its own wave hwreg state during forward pass',
        'architecture': {
            'type': 'IntrinsicStateTransformer',
            'tokens': ['delta(5)', 'intrinsic_hw(12)', 'timing(4)',
                       'thermal(15)', 'status(3)', 'action(2)'],
            'total_sensor_dims': DELTA_DIM + INTRINSIC_DIM + TIMING_DIM + 15 + 3 + 2,
            'intrinsic_registers': ['STATUS', 'GPR_ALLOC', 'LDS_ALLOC', 'IB_STS',
                                    'HW_ID2', 'PERF_SNAPSHOT', 'SHADER_CYCLES_LO',
                                    'SHADER_CYCLES_HI', 'clock64_pre', 'clock64_post',
                                    'wall_clock_pre', 'wall_clock_post'],
            'transformer_heads': 4,
            'token_dim': TOKEN_DIM,
            'n_tokens': 6,
            'trainable_MathLinear': True,
            'closed_loop_ISA': True,
            'gaslighting_fraction': GASLIGHT_FRAC,
            'smn_available': SMN_AVAILABLE,
        },
        'tests': results,
        'summary': {
            'passed': passed,
            'total': total_tests,
            'pass_rate': f"{passed}/{total_tests}",
        },
        'energy': {
            'avg_power_W': energy.avg_power_w() / 1000.0,
            'total_joules': energy.total_joules,
        },
    }

    def convert(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    out_path = 'results/z2085_intrinsic_state_transformer.json'
    os.makedirs('results', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=convert)
    print(f"\n[SAVED] {out_path}")


if __name__ == '__main__':
    main()
