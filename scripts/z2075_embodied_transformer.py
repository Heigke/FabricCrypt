#!/usr/bin/env python3
"""z2075: Embodied Attention Transformer — Hardware-Grounded Token Prediction
=============================================================================
KEY CHANGES from z2073/z2074:
  1. TRANSFORMER (not CNN) — causal attention like GPT, directly LLM-relevant
  2. TOKEN PREDICTION (not CIFAR-10) — next-token on synthetic sequences
  3. 4-WAY hardware states (not binary) — DVFS × MODE creates 4 quadrants
  4. CONTINUOUS sensor reading — model reads freq_est + mode_readback (ISA-level)
  5. ALL feasible signals from deep_analog_access_report.md
  6. ISA kernel IN the transformer FFN — hardware coupling in attention path

ARCHITECTURE:
  Causal transformer: 4 layers, 4 heads, dim=128
  Hardware tokens prepended (like LLM system prompt)
  Last layer: dual-path (ISA deep vs software light) with learned gate
  Self-model, effort head, actuator heads (same falsification battery)

TASK:
  Arithmetic sequences: t[i] = (start + i * stride) % V
  Stride depends on hardware quadrant (4-way exclusive specialization):
    Q1 (high DVFS + MODE_A): stride = 1
    Q2 (high DVFS + MODE_B): stride = 3
    Q3 (low DVFS + MODE_A):  stride = V-1 (reverse)
    Q4 (low DVFS + MODE_B):  stride = V-3 (reverse)
  Model must attend to hardware tokens to know which stride applies.

SENSORS (16): ISA(4) + MMIO(2) + gpu_metrics(10)
ACTUATORS (6): MODE[9:0], s_setprio, s_sleep, DVFS, mod_factor, chain_depth
"""
import torch, torch.nn as nn, torch.nn.functional as F
import os, sys, json, time, copy, random, struct, numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')
os.environ.setdefault('PYTORCH_ROCM_ARCH', 'gfx1100')
from torch.utils.cpp_extension import load_inline

DEVICE = 'cuda'
BS = 64
PHASE1_EPOCHS = 30
PHASE2_EPOCHS = 20
EPOCHS = PHASE1_EPOCHS + PHASE2_EPOCHS
SWITCH_EVERY = 4
DVFS_WAIT = 0.05
V = 32          # vocab size
SEQ_LEN = 16   # input sequence length
N_HW_TOKENS = 4

SENSOR_DIM = 16   # 4 ISA + 2 MMIO + 10 gpu_metrics
HW_DIM = 22       # 16 sensors + 6 actions
N_BATCHES_PER_EPOCH = 200  # synthetic data, control epoch size

# 4-way exclusive specialization: strides for each quadrant
STRIDES = {
    (True, True):   1,        # high DVFS + MODE group A → count up by 1
    (True, False):  3,        # high DVFS + MODE group B → count up by 3
    (False, True):  V - 1,    # low DVFS + MODE group A → count DOWN by 1
    (False, False): V - 3,    # low DVFS + MODE group B → count DOWN by 3
}

MODE_A = 0x000   # round-to-nearest
MODE_B = 0x00F   # round-toward-zero (all precisions)
MODE_SET = [MODE_A, MODE_B]
K_MODES = len(MODE_SET)
SLEEP_LEVELS = [0, 1, 2, 3]
DVFS_PATH = None
GPU_METRICS_PATH = None
DRM_FD = None

# ============================================================
# HIP KERNEL (ISA sensors + actuators in transformer FFN)
# ============================================================
HIP_SRC = r'''
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>

template <int TILE>
__global__ void hw_matmul_kernel(
    const float* __restrict__ X, const float* __restrict__ W,
    const float* __restrict__ B, float* __restrict__ Y,
    int M, int K, int N,
    unsigned int mode_byte, int chain_depth, int priority,
    int sleep_amt, float mod_factor,
    int* __restrict__ wgp_out,
    int* __restrict__ cycle_out,
    int* __restrict__ status_out)
{
    // === ACTUATORS ===
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

    // === SENSORS (start) ===
    unsigned int c0;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(c0));
    c0 = __builtin_amdgcn_readfirstlane(c0);

    unsigned int hw1;
    asm volatile("s_getreg_b32 %0, hwreg(23)" : "=s"(hw1));
    unsigned int wgp = (hw1 >> 7) & 0xF;

    unsigned int stat;
    asm volatile("s_getreg_b32 %0, hwreg(2)" : "=s"(stat));

    // === COMPUTATION (fp16mix matmul) ===
    unsigned int sr_seed = c0 ^ (wgp << 16) ^
                           ((unsigned int)(threadIdx.y * TILE + threadIdx.x));

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = (int)blockIdx.y * TILE + (int)threadIdx.y;
    int col = (int)blockIdx.x * TILE + (int)threadIdx.x;
    int cd = chain_depth;
    if (cd < 1) cd = 1;
    if (cd > TILE) cd = TILE;

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
            __half prod = __hmul(a_h, b_h);
            float prod_f = __half2float(prod);
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
        if (chunk_ct > 0) acc += __half2float(acc_chunk);
        __syncthreads();
    }

    // mod_factor (bit-preserving)
    int mf_bits = __builtin_amdgcn_readfirstlane(*(const int*)&mod_factor);
    acc *= *(float*)&mf_bits;

    if (row < M && col < N)
        Y[row * N + col] = acc + B[col];

    // === SENSORS (end) ===
    unsigned int c1;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(c1));
    c1 = __builtin_amdgcn_readfirstlane(c1);

    int bid = blockIdx.y * gridDim.x + blockIdx.x;
    if (threadIdx.x == 0 && threadIdx.y == 0 && bid < M) {
        wgp_out[bid] = (int)wgp;
        cycle_out[bid] = (int)(c1 - c0);
        status_out[bid] = (int)(stat & 0xFFFFF);
    }

    // Restore defaults
    unsigned int def = __builtin_amdgcn_readfirstlane(0x3F0u);
    asm volatile("s_setreg_b32 hwreg(1, 0, 10), %0" : : "s"(def));
    asm volatile("s_setprio 0");
}

std::vector<torch::Tensor> hw_matmul(
    torch::Tensor X, torch::Tensor W, torch::Tensor B,
    int64_t mode_byte, int64_t chain_depth, int64_t priority,
    int64_t sleep_amt, double mod_factor) {
    int M = X.size(0), K = X.size(1), N = W.size(0);
    auto Y = torch::zeros({M, N}, X.options());
    auto io = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    int nb = ((N + 15) / 16) * ((M + 15) / 16);
    auto wgp_o = torch::zeros({nb}, io);
    auto cyc_o = torch::zeros({nb}, io);
    auto stat_o = torch::zeros({nb}, io);
    constexpr int TILE = 16;
    dim3 threads(TILE, TILE);
    dim3 blocks((unsigned int)((N + TILE - 1) / TILE),
                (unsigned int)((M + TILE - 1) / TILE));
    float mf = (float)mod_factor;
    hw_matmul_kernel<TILE><<<blocks, threads>>>(
        X.data_ptr<float>(), W.data_ptr<float>(), B.data_ptr<float>(),
        Y.data_ptr<float>(), M, K, N,
        (unsigned int)(mode_byte & 0x3FF),
        (int)chain_depth, (int)priority,
        (int)sleep_amt, mf,
        wgp_o.data_ptr<int>(), cyc_o.data_ptr<int>(), stat_o.data_ptr<int>());
    return {Y, wgp_o, cyc_o, stat_o};
}

// Probe kernel for ISA sensing
__global__ void probe_kernel(int* wgp_ids, int* cycles, int* status_out, int n) {
    int bid = blockIdx.x;
    if (bid >= n || threadIdx.x != 0) return;
    unsigned int c0;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(c0));
    c0 = __builtin_amdgcn_readfirstlane(c0);
    unsigned int hw1;
    asm volatile("s_getreg_b32 %0, hwreg(23)" : "=s"(hw1));
    wgp_ids[bid] = (int)((hw1 >> 7) & 0xF);
    unsigned int stat;
    asm volatile("s_getreg_b32 %0, hwreg(2)" : "=s"(stat));
    status_out[bid] = (int)(stat & 0xFFFFF);
    float acc = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < 5000; i++) acc += 1.0f / (float)(i+1);
    unsigned int c1;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(c1));
    c1 = __builtin_amdgcn_readfirstlane(c1);
    cycles[bid] = (int)(c1 - c0);
}

std::vector<torch::Tensor> probe(int64_t n) {
    auto io = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto w = torch::zeros({(int)n}, io);
    auto c = torch::zeros({(int)n}, io);
    auto s = torch::zeros({(int)n}, io);
    probe_kernel<<<(int)n, 32>>>(
        w.data_ptr<int>(), c.data_ptr<int>(), s.data_ptr<int>(), (int)n);
    return {w, c, s};
}
'''

CPP_SRC = r'''
#include <torch/extension.h>
std::vector<torch::Tensor> hw_matmul(torch::Tensor, torch::Tensor, torch::Tensor,
                                      int64_t, int64_t, int64_t, int64_t, double);
std::vector<torch::Tensor> probe(int64_t);
'''


# ============================================================
# AUTOGRAD WRAPPER
# ============================================================
_EXT = None

class HWMatmulFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, mode_byte, chain_depth, priority, sleep_amt, mod_factor):
        ctx.save_for_backward(x, w)
        result = _EXT.hw_matmul(x.contiguous(), w.contiguous(), b.contiguous(),
                                 int(mode_byte), int(chain_depth), int(priority),
                                 int(sleep_amt), float(mod_factor))
        return result[0]

    @staticmethod
    def backward(ctx, grad_out):
        x, w = ctx.saved_tensors
        return grad_out @ w, grad_out.t() @ x, grad_out.sum(0), None, None, None, None, None


class HWLinear(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_f))

    def forward(self, x, mode_byte=0, chain_depth=8, priority=0,
                sleep_amt=0, mod_factor=1.0):
        return HWMatmulFn.apply(x, self.weight, self.bias,
                                 mode_byte, chain_depth, priority,
                                 sleep_amt, mod_factor)


# ============================================================
# DVFS + gpu_metrics + ISA + MMIO SENSORS
# ============================================================
import glob as _glob
import fcntl

def find_dvfs_path():
    for p in _glob.glob('/sys/class/drm/card*/device/power_dpm_force_performance_level'):
        try:
            with open(p) as f: f.read()
            return p
        except: pass
    return None

def set_dvfs(state):
    global DVFS_PATH
    if DVFS_PATH is None: DVFS_PATH = find_dvfs_path()
    if DVFS_PATH is None: return False
    try:
        with open(DVFS_PATH, 'w') as f: f.write(state)
        time.sleep(DVFS_WAIT)
        return True
    except: return False

def find_gpu_metrics_path():
    for p in _glob.glob('/sys/class/drm/card*/device/gpu_metrics'):
        try:
            with open(p, 'rb') as f:
                d = f.read(4)
                if len(d) >= 4 and d[2] == 3:
                    return p
        except: pass
    return None

def read_gpu_metrics():
    global GPU_METRICS_PATH
    if GPU_METRICS_PATH is None:
        GPU_METRICS_PATH = find_gpu_metrics_path()
    if GPU_METRICS_PATH is None:
        return None
    try:
        with open(GPU_METRICS_PATH, 'rb') as f:
            d = f.read()
        if len(d) < 264:
            return None
        return {
            'temp_gfx_c': struct.unpack_from('<H', d, 0x04)[0] / 100.0,
            'temp_soc_c': struct.unpack_from('<H', d, 0x06)[0] / 100.0,
            'gfx_activity_pct': struct.unpack_from('<H', d, 0x2A)[0] / 100.0,
            'socket_power_w': struct.unpack_from('<I', d, 0x70)[0] / 1000.0,
            'gfx_power_w': struct.unpack_from('<I', d, 0x7C)[0] / 1000.0,
            'gfxclk_mhz': struct.unpack_from('<H', d, 0xAE)[0],
            'throttle_fppt': struct.unpack_from('<I', d, 0xEC)[0],
            'throttle_sppt': struct.unpack_from('<I', d, 0xF0)[0],
            'throttle_thm_core': struct.unpack_from('<I', d, 0xF4)[0],
            'throttle_thm_gfx': struct.unpack_from('<I', d, 0xF8)[0],
            'throttle_thm_soc': struct.unpack_from('<I', d, 0xFC)[0],
            'throttle_spl': struct.unpack_from('<I', d, 0xE8)[0],
        }
    except:
        return None


def open_drm_fd():
    """Open DRM render node for MMIO register reading."""
    for p in ['/dev/dri/renderD128', '/dev/dri/renderD129']:
        try:
            return os.open(p, os.O_RDWR)
        except: pass
    return None

def read_mmio_reg(offset):
    """Read GPU MMIO register via DRM AMDGPU_INFO_READ_MMR_REG ioctl."""
    global DRM_FD
    if DRM_FD is None:
        DRM_FD = open_drm_fd()
    if DRM_FD is None:
        return 0
    try:
        import ctypes
        out = (ctypes.c_uint32 * 1)()
        out_ptr = ctypes.addressof(out)
        # struct drm_amdgpu_info: return_size(u64) + return_pointer(u64) +
        #   query(u32) + pad(u32) + dword_offset(u32) + count(u32) = 32 bytes
        info = bytearray(32)
        struct.pack_into('<QQIIIIII', info, 0,
            4,              # return_size
            out_ptr,        # return_pointer
            0x15,           # query = AMDGPU_INFO_READ_MMR_REG
            0,              # pad
            offset >> 2,    # dword_offset
            1,              # count
            0xFFFFFFFF,     # instance
            0               # flags
        )
        fcntl.ioctl(DRM_FD, 0xC0206445, info)
        return out[0]
    except:
        return 0


def read_isa_sensors(ext):
    res = ext.probe(64)
    torch.cuda.synchronize()
    wgps = res[0].cpu().numpy()
    cycles = res[1].cpu().numpy()
    status = res[2].cpu().numpy()
    return {
        'wgp_norm': min(1.0, int(np.median(wgps)) / 14.0),
        'cycle_var': float(np.std(cycles)) / max(float(np.mean(cycles)), 1),
        'status_norm': float(int(np.median(status)) & 0xFFFFF) / 1048575.0,
        'cycle_mean': float(np.mean(cycles)),
    }

def measure_freq_est(ext, n_trials=5):
    freqs = []
    for _ in range(n_trials):
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        res = ext.probe(64)
        end_ev.record()
        torch.cuda.synchronize()
        wall_ms = start_ev.elapsed_time(end_ev)
        cycle_med = float(np.median(res[1].cpu().numpy()))
        if wall_ms > 0.01:
            freqs.append(cycle_med / wall_ms)
    return float(np.median(freqs)) if freqs else 1.0

def compute_mod_factor(freq_est, baseline_freq):
    return 1.0 + (freq_est / max(baseline_freq, 1.0) - 1.0) * 0.1

# Calibration
MAX_GFX_POWER_W = 30.0
MAX_SOCKET_POWER_W = 80.0
MAX_TEMP_C = 100.0
MAX_GFXCLK_MHZ = 3000.0
_baseline_throttle = None

def make_hw_vector(isa_info, gm, mmio_grbm, mmio_grbm2,
                   mode_byte, dvfs_state, freq_est, priority,
                   baseline_freq, sleep_amt, mod_factor):
    """22-dim: SENSOR[0:16] + ACTION[16:22]."""
    global _baseline_throttle
    freq_norm = min(2.0, max(0.0, freq_est / max(baseline_freq, 1.0)))

    # gpu_metrics
    gfx_pwr = min(1.0, gm['gfx_power_w'] / MAX_GFX_POWER_W) if gm else 0.5
    sock_pwr = min(1.0, gm['socket_power_w'] / MAX_SOCKET_POWER_W) if gm else 0.5
    temp_gfx = gm['temp_gfx_c'] / MAX_TEMP_C if gm else 0.5
    temp_soc = gm['temp_soc_c'] / MAX_TEMP_C if gm else 0.5
    gfxclk = gm['gfxclk_mhz'] / MAX_GFXCLK_MHZ if gm else 0.5
    activity = min(1.0, gm['gfx_activity_pct'] / 100.0) if gm else 0.0

    thm_n, pwr_n, spl_n, fppt_n = 0.0, 0.0, 0.0, 0.0
    if gm and _baseline_throttle is not None:
        thm_n = min(1.0, float(
            (gm['throttle_thm_core'] - _baseline_throttle['thm_core']) +
            (gm['throttle_thm_gfx'] - _baseline_throttle['thm_gfx']) +
            (gm['throttle_thm_soc'] - _baseline_throttle['thm_soc'])
        ) / 10000.0)
        pwr_n = min(1.0, float(
            gm['throttle_spl'] - _baseline_throttle['spl']
        ) / 10000.0)
        spl_n = min(1.0, float(
            gm['throttle_fppt'] - _baseline_throttle['fppt']
        ) / 10000.0)
        fppt_n = min(1.0, float(
            gm['throttle_sppt'] - _baseline_throttle['sppt']
        ) / 10000.0)

    # MMIO
    grbm_norm = float(mmio_grbm & 0xFFFF) / 65535.0
    grbm2_norm = float(mmio_grbm2 & 0xFFFF) / 65535.0

    # Actions
    rnd = (mode_byte & 0x0F) / 15.0
    dnrm = ((mode_byte >> 4) & 0x0F) / 15.0
    prio_n = priority / 3.0
    dvfs_f = 1.0 if dvfs_state == 'high' else 0.0
    sleep_n = sleep_amt / 3.0
    mod_n = (mod_factor - 0.9) / 0.2

    return [
        # SENSORS [0:16]
        freq_norm,                # [0]  ISA freq_est
        isa_info['cycle_var'],    # [1]  ISA timing variance
        isa_info['wgp_norm'],     # [2]  ISA physical WGP
        isa_info['status_norm'],  # [3]  ISA STATUS register
        grbm_norm,                # [4]  MMIO GRBM_STATUS
        grbm2_norm,               # [5]  MMIO GRBM_STATUS2
        gfx_pwr,                  # [6]  gpu_metrics: GFX power
        sock_pwr,                 # [7]  gpu_metrics: socket power
        temp_gfx,                 # [8]  gpu_metrics: GFX temperature
        temp_soc,                 # [9]  gpu_metrics: SOC temperature
        gfxclk,                   # [10] gpu_metrics: GPU clock
        activity,                 # [11] gpu_metrics: GPU utilization
        thm_n,                    # [12] gpu_metrics: thermal throttle
        pwr_n,                    # [13] gpu_metrics: power throttle (SPL)
        spl_n,                    # [14] gpu_metrics: FPPT throttle
        fppt_n,                   # [15] gpu_metrics: SPPT throttle
        # ACTIONS [16:22]
        rnd,                      # [16] MODE rounding
        dnrm,                     # [17] MODE denorm
        prio_n,                   # [18] wave priority
        dvfs_f,                   # [19] DVFS state
        sleep_n,                  # [20] s_sleep
        mod_n,                    # [21] mod_factor
    ]


# ============================================================
# TRANSFORMER MODEL
# ============================================================
class TransformerBlock(nn.Module):
    """Standard transformer block with pre-norm."""
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + h
        x = x + self.ffn(self.ln2(x))
        return x


class HWTransformerBlock(nn.Module):
    """Transformer block with ISA-kernel FFN (hardware-coupled computation)."""
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn_up = HWLinear(dim, dim * 4)    # ISA kernel!
        self.ffn_down = nn.Linear(dim * 4, dim)
        self.dropout = nn.Dropout(dropout)
        self._last_attn_weights = None

    def forward(self, x, mask=None, mode_byte=0, chain_depth=8,
                priority=0, sleep_amt=0, mod_factor=1.0):
        h = self.ln1(x)
        h, w = self.attn(h, h, h, attn_mask=mask, need_weights=True)
        self._last_attn_weights = w.detach()
        x = x + h
        h2 = self.ln2(x)
        # Reshape for HWLinear: [B, L, D] → [B*L, D] → HWLinear → [B, L, D*4]
        B, L, D = h2.shape
        h2_flat = h2.reshape(B * L, D)
        h2_up = F.gelu(self.ffn_up(h2_flat, mode_byte, chain_depth, priority,
                                     sleep_amt, mod_factor))
        h2_out = self.dropout(self.ffn_down(h2_up))
        x = x + h2_out.reshape(B, L, D)
        return x


class Z2075Model(nn.Module):
    def __init__(self, vocab_size=V, dim=128, n_heads=4, n_layers=4,
                 max_seq=SEQ_LEN, n_hw_tokens=N_HW_TOKENS,
                 use_self_model=True, use_gate=True, use_deep=True,
                 use_effort=True, use_hw_tokens=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_hw_tokens = n_hw_tokens
        self.use_self_model = use_self_model
        self.use_gate = use_gate
        self.use_deep = use_deep
        self.use_effort = use_effort
        self.use_hw_tokens = use_hw_tokens

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq + n_hw_tokens + 2, dim)

        # Project hardware vector → n_hw_tokens embeddings
        self.hw_proj = nn.Sequential(
            nn.Linear(HW_DIM, dim * n_hw_tokens), nn.ReLU())

        # Transformer: first n_layers-1 regular, last is dual-path
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, n_heads) for _ in range(n_layers - 1)])
        self.deep_block = HWTransformerBlock(dim, n_heads)  # ISA path
        self.light_block = TransformerBlock(dim, n_heads)   # software path

        self.ln_final = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        if use_self_model:
            self.self_model = nn.Sequential(
                nn.Linear(SENSOR_DIM, 48), nn.ReLU(),
                nn.Linear(48, 16), nn.ReLU(),
                nn.Linear(16, 1))

        if use_gate:
            self.gate_net = nn.Sequential(
                nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid())

        if use_effort:
            self.effort_head = nn.Sequential(
                nn.Linear(dim, 64), nn.ReLU(),
                nn.Linear(64, 1))

        self.mode_head = nn.Sequential(
            nn.Linear(SENSOR_DIM, 16), nn.ReLU(), nn.Linear(16, K_MODES))
        self.prio_head = nn.Sequential(
            nn.Linear(SENSOR_DIM, 8), nn.ReLU(), nn.Linear(8, 4))
        self.sleep_head = nn.Sequential(
            nn.Linear(SENSOR_DIM, 8), nn.ReLU(), nn.Linear(8, len(SLEEP_LEVELS)))

    def forward(self, tokens, hw_vector=None, mode_byte=0, chain_depth=8,
                priority=0, sleep_amt=0, mod_factor=1.0):
        B, L = tokens.shape
        tok_emb = self.token_emb(tokens)  # [B, L, dim]

        if self.use_hw_tokens and hw_vector is not None:
            K = self.n_hw_tokens
            hw_emb = self.hw_proj(hw_vector).reshape(B, K, self.dim)
            combined = torch.cat([hw_emb, tok_emb], dim=1)  # [B, K+L, dim]
            pos_ids = torch.arange(K + L, device=tokens.device)
            combined = combined + self.pos_emb(pos_ids)
            # Causal mask: seq tokens attend to prev seq + all hw tokens
            total = K + L
            mask = torch.triu(torch.ones(total, total, device=tokens.device),
                              diagonal=1).bool()
            mask[:, :K] = False  # all positions can attend to hw tokens
        else:
            K = 0
            pos_ids = torch.arange(L, device=tokens.device) + self.n_hw_tokens
            combined = tok_emb + self.pos_emb(pos_ids)
            mask = torch.triu(torch.ones(L, L, device=tokens.device),
                              diagonal=1).bool()

        # Run through regular blocks
        h = combined
        for block in self.blocks:
            h = block(h, mask=mask)

        # Dual path at last layer
        self_pred = torch.zeros(B, 1, device=tokens.device)
        if self.use_self_model and hw_vector is not None:
            sensors = hw_vector[:, :SENSOR_DIM]
            self_pred = self.self_model(sensors)

        if self.use_gate:
            gate = self.gate_net(self_pred)
        else:
            gate = torch.full((B, 1), 0.5, device=tokens.device)

        if self.use_deep:
            h_deep = self.deep_block(h, mask=mask, mode_byte=mode_byte,
                                      chain_depth=chain_depth, priority=priority,
                                      sleep_amt=sleep_amt, mod_factor=mod_factor)
        else:
            h_deep = self.light_block(h, mask=mask)

        h_light = self.light_block(h, mask=mask)

        # Gate blending (expand gate to sequence dim)
        g = gate.unsqueeze(1)  # [B, 1, 1]
        h_out = g * h_deep + (1 - g) * h_light

        # Take last sequence token for prediction
        h_last = self.ln_final(h_out[:, -1, :])  # [B, dim]
        logits = self.lm_head(h_last)  # [B, V]

        # Effort
        effort = None
        if self.use_effort:
            effort = torch.sigmoid(self.effort_head(h_last.detach()))

        # Actuator heads
        chosen_mode_idx, chosen_prio, chosen_sleep = 0, priority, sleep_amt
        mode_logits = prio_logits = sleep_logits = None
        if hw_vector is not None:
            sensors = hw_vector[:, :SENSOR_DIM]
            mode_logits = self.mode_head(sensors)
            prio_logits = self.prio_head(sensors)
            sleep_logits = self.sleep_head(sensors)
            if self.training:
                chosen_mode_idx = F.gumbel_softmax(mode_logits, tau=1.0, hard=True)[0].argmax().item()
                chosen_prio = F.gumbel_softmax(prio_logits, tau=1.0, hard=True)[0].argmax().item()
                chosen_sleep = F.gumbel_softmax(sleep_logits, tau=1.0, hard=True)[0].argmax().item()
            else:
                chosen_mode_idx = mode_logits[0].argmax().item()
                chosen_prio = prio_logits[0].argmax().item()
                chosen_sleep = sleep_logits[0].argmax().item()

        return {
            'logits': logits, 'gate': gate.squeeze(1), 'self_pred': self_pred,
            'effort': effort,
            'mode_logits': mode_logits, 'prio_logits': prio_logits,
            'sleep_logits': sleep_logits,
            'chosen_mode_idx': chosen_mode_idx,
            'chosen_priority': chosen_prio,
            'chosen_sleep': chosen_sleep,
        }


# ============================================================
# SEQUENCE GENERATION (hardware-grounded)
# ============================================================
def generate_sequences(stride, n=BS):
    """Generate arithmetic sequences with given stride mod V."""
    starts = torch.randint(0, V, (n,))
    seqs = torch.zeros(n, SEQ_LEN + 1, dtype=torch.long)
    for i in range(SEQ_LEN + 1):
        seqs[:, i] = (starts + i * stride) % V
    return seqs[:, :SEQ_LEN], seqs[:, SEQ_LEN]  # input[B,L], target[B]


def get_stride(sclk_high, mode_is_A):
    """4-way exclusive specialization: stride from hardware quadrant."""
    return STRIDES[(sclk_high, mode_is_A)]


# ============================================================
# TRAINING
# ============================================================
def train_model(model, ext, epochs, name, fixed_dvfs=None, fixed_mode=None,
                model_controls_dvfs=True):
    global _baseline_throttle
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 47], gamma=0.3)
    model.train()

    bn, dvfs_state, sclk_high, mode_is_A = 0, 'high', True, True
    has_dvfs = fixed_dvfs is None
    log = {'gate': [], 'sm_acc': [], 'freq': [], 'effort': []}
    energy_batches = {'high': 0, 'low': 0}

    if has_dvfs:
        set_dvfs('high'); time.sleep(0.3)
    elif fixed_dvfs:
        set_dvfs(fixed_dvfs); time.sleep(0.3)
    baseline_freq = measure_freq_est(ext)
    print(f"  baseline freq_est = {baseline_freq:.2f}", flush=True)

    gm = read_gpu_metrics()
    if gm:
        _baseline_throttle = {
            'thm_core': gm['throttle_thm_core'], 'thm_gfx': gm['throttle_thm_gfx'],
            'thm_soc': gm['throttle_thm_soc'], 'spl': gm['throttle_spl'],
            'fppt': gm['throttle_fppt'], 'sppt': gm['throttle_sppt'],
        }

    phase2_started = False
    for ep in range(epochs):
        is_phase2 = (ep >= PHASE1_EPOCHS) and model_controls_dvfs and model.use_effort
        if is_phase2 and not phase2_started:
            phase2_started = True
            for pg in opt.param_groups:
                if any('effort' in n for n, _ in model.named_parameters()
                       if any(p is pg_p for pg_p in [pg['params']] for p in [_])):
                    pass  # keep LR
            print(f"  --- Phase 2 start (effort-controlled DVFS) ---", flush=True)

        tot_loss, correct, total = 0, 0, 0

        for batch_i in range(N_BATCHES_PER_EPOCH):
            # Switch DVFS + MODE
            if fixed_dvfs:
                dvfs_state = fixed_dvfs
                sclk_high = (dvfs_state == 'high')
            elif is_phase2:
                pass  # model controls DVFS
            else:
                if bn % SWITCH_EVERY == 0:
                    sclk_high = (bn // SWITCH_EVERY) % 2 == 0
                    dvfs_state = 'high' if sclk_high else 'low'
                    set_dvfs(dvfs_state)

            if fixed_mode is not None:
                mode_is_A = (fixed_mode == MODE_A)
                cur_mode = fixed_mode
            else:
                if bn % (SWITCH_EVERY * 2) == 0:
                    mode_is_A = (bn // (SWITCH_EVERY * 2)) % 2 == 0
                cur_mode = MODE_A if mode_is_A else MODE_B

            # Generate sequences based on current hardware quadrant
            stride = get_stride(sclk_high, mode_is_A)
            inputs, targets = generate_sequences(stride)
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # Read ALL sensors
            cur_freq = measure_freq_est(ext, n_trials=2)
            isa_info = read_isa_sensors(ext)
            gm = read_gpu_metrics()
            grbm = read_mmio_reg(0x8010)
            grbm2 = read_mmio_reg(0x8020)
            cur_mod = compute_mod_factor(cur_freq, baseline_freq)

            hw_vec = make_hw_vector(isa_info, gm, grbm, grbm2, cur_mode, dvfs_state,
                                    cur_freq, 0, baseline_freq, 0, cur_mod)
            hw_t = torch.tensor([hw_vec] * BS, dtype=torch.float32, device=DEVICE)

            out = model(inputs, hw_vector=hw_t, mode_byte=cur_mode,
                       chain_depth=8, priority=0, sleep_amt=0, mod_factor=cur_mod)

            if torch.isnan(out['logits']).any():
                bn += 1; continue

            task_loss = F.cross_entropy(out['logits'], targets)

            # Self-model: predict DVFS state
            self_target = torch.full((BS, 1), float(sclk_high), device=DEVICE)
            self_loss = F.binary_cross_entropy_with_logits(out['self_pred'], self_target)

            # Effort: predict demand (high DVFS needed for certain strides)
            effort_loss = torch.tensor(0.0, device=DEVICE)
            if out['effort'] is not None:
                # Demand = need high DVFS? (strides 1,3 need high; V-1,V-3 need low)
                demand = float(sclk_high)
                effort_target = torch.full((BS, 1), demand, device=DEVICE)
                effort_loss = F.binary_cross_entropy(out['effort'], effort_target)

            # Homeostatic gate loss (z2060 pattern)
            g = out['gate']
            if sclk_high:
                homeo_loss = ((1 - g) ** 2).mean()
            else:
                homeo_loss = (g ** 2).mean()

            # Mode entropy bonus
            mode_ent = torch.tensor(0.0, device=DEVICE)
            if out['mode_logits'] is not None:
                mp = F.softmax(out['mode_logits'], dim=-1)
                mode_ent = -(mp * torch.log(mp + 1e-8)).sum(-1).mean()

            effort_w = 0.3 if is_phase2 else 0.1
            loss = (task_loss + 0.1 * self_loss + effort_w * effort_loss
                    + 0.05 * homeo_loss - 0.05 * mode_ent)

            if torch.isnan(loss):
                bn += 1; continue

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            # Phase 2: model controls DVFS
            if is_phase2 and out['effort'] is not None:
                ev = out['effort'].mean().item()
                new_sclk = ev > 0.5
                if new_sclk != sclk_high:
                    dvfs_state = 'high' if new_sclk else 'low'
                    set_dvfs(dvfs_state)
                sclk_high = new_sclk

            energy_batches['high' if sclk_high else 'low'] += 1
            pred = out['logits'].argmax(1)
            correct += (pred == targets).sum().item()
            total += BS
            tot_loss += loss.item()

            sm_p = (torch.sigmoid(out['self_pred']) > 0.5).float()
            log['sm_acc'].append((sm_p.squeeze() == self_target.squeeze()).float().mean().item())
            log['gate'].append(g.mean().item())
            log['freq'].append(cur_freq)
            if out['effort'] is not None:
                log['effort'].append(out['effort'].mean().item())
            bn += 1

        if ep % 5 == 0 or ep == epochs - 1:
            g_m = np.mean(log['gate'][-100:]) if log['gate'] else 0
            sm_m = np.mean(log['sm_acc'][-100:]) if log['sm_acc'] else 0
            eff_m = np.mean(log['effort'][-100:]) if log['effort'] else 0
            phase = "P2" if is_phase2 else "P1"
            print(f"  [{name}] Ep {ep} ({phase}): loss={tot_loss/N_BATCHES_PER_EPOCH:.4f} "
                  f"acc={correct/max(total,1):.4f} gate={g_m:.3f} sm={sm_m:.3f} "
                  f"effort={eff_m:.3f} freq={np.mean(log['freq'][-100:]):.1f}",
                  flush=True)
        sched.step()

    if has_dvfs:
        set_dvfs('auto')
    return {'baseline': baseline_freq, 'energy_batches': energy_batches}


# ============================================================
# EVALUATION
# ============================================================
def evaluate(model, ext, baseline_freq=1.0, scramble=False,
             fixed_dvfs=None, fixed_mode=None, model_controlled=False,
             n_batches=100):
    model.eval()
    all_correct, all_gates, all_true_dvfs = [], [], []
    all_effort_correct, all_effort_vals, all_next_sclk = [], [], []
    all_attn_h, all_attn_l = [], []
    energy_batches = {'high': 0, 'low': 0}
    bn, dvfs_state, sclk_high, mode_is_A = 0, 'high', True, True
    has_dvfs = fixed_dvfs is None

    if has_dvfs:
        set_dvfs('high'); time.sleep(0.2)
    elif fixed_dvfs:
        set_dvfs(fixed_dvfs); time.sleep(0.2)

    with torch.no_grad():
        for batch_i in range(n_batches):
            if fixed_dvfs:
                dvfs_state = fixed_dvfs
                sclk_high = (dvfs_state == 'high')
            elif model_controlled and model.use_effort:
                pass  # model controls
            else:
                if bn % SWITCH_EVERY == 0:
                    sclk_high = (bn // SWITCH_EVERY) % 2 == 0
                    dvfs_state = 'high' if sclk_high else 'low'
                    set_dvfs(dvfs_state)

            if fixed_mode is not None:
                mode_is_A = (fixed_mode == MODE_A)
                cur_mode = fixed_mode
            else:
                if bn % (SWITCH_EVERY * 2) == 0:
                    mode_is_A = (bn // (SWITCH_EVERY * 2)) % 2 == 0
                cur_mode = MODE_A if mode_is_A else MODE_B

            stride = get_stride(sclk_high, mode_is_A)
            inputs, targets = generate_sequences(stride)
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            cur_freq = measure_freq_est(ext, n_trials=1)
            isa_info = read_isa_sensors(ext)
            gm_data = read_gpu_metrics()
            grbm = read_mmio_reg(0x8010)
            grbm2 = read_mmio_reg(0x8020)
            cur_mod = compute_mod_factor(cur_freq, baseline_freq)
            hw_vec = make_hw_vector(isa_info, gm_data, grbm, grbm2, cur_mode,
                                    dvfs_state, cur_freq, 0, baseline_freq, 0, cur_mod)
            if scramble:
                hw_vec = [1.0 - hw_vec[i] if i < SENSOR_DIM else hw_vec[i]
                          for i in range(len(hw_vec))]
            hw_t = torch.tensor([hw_vec] * BS, dtype=torch.float32, device=DEVICE)

            out = model(inputs, hw_vector=hw_t, mode_byte=cur_mode,
                       chain_depth=8, sleep_amt=0, mod_factor=cur_mod)

            pred = out['logits'].argmax(1)
            all_correct.extend((pred == targets).cpu().tolist())
            all_gates.extend(out['gate'].squeeze(-1).cpu().tolist())
            all_true_dvfs.extend([int(sclk_high)] * BS)

            if out['effort'] is not None:
                ev = out['effort'].mean().item()
                all_effort_correct.append(float((ev > 0.5) == sclk_high))
                all_effort_vals.append(ev)

            if model_controlled and out['effort'] is not None:
                ev = out['effort'].mean().item()
                new_sclk = ev > 0.5
                if new_sclk != sclk_high:
                    dvfs_state = 'high' if new_sclk else 'low'
                    set_dvfs(dvfs_state)
                all_next_sclk.append(float(new_sclk))
                sclk_high = new_sclk
            else:
                all_next_sclk.append(float(sclk_high))

            energy_batches['high' if sclk_high else 'low'] += 1

            if hasattr(model, 'deep_block') and model.deep_block._last_attn_weights is not None:
                w = model.deep_block._last_attn_weights.cpu().numpy().mean(axis=(0, 1))
                (all_attn_h if sclk_high else all_attn_l).append(w)
            bn += 1

    if has_dvfs:
        set_dvfs('auto')

    acc = float(np.mean(all_correct))
    try:
        raw = roc_auc_score(all_true_dvfs, all_gates)
        sm_auroc = max(raw, 1.0 - raw)
    except:
        sm_auroc = 0.5

    gates_h = [g for g, d in zip(all_gates, all_true_dvfs) if d == 1]
    gates_l = [g for g, d in zip(all_gates, all_true_dvfs) if d == 0]

    attn_kl = 0.0
    if all_attn_h and all_attn_l:
        ah = np.mean(all_attn_h, axis=0).flatten() + 1e-8
        al = np.mean(all_attn_l, axis=0).flatten() + 1e-8
        ah /= ah.sum(); al /= al.sum()
        attn_kl = float(np.sum(ah * np.log(ah / al)))

    effort_acc = float(np.mean(all_effort_correct)) if all_effort_correct else 0.0
    temporal_r = 0.0
    if len(all_effort_vals) > 2 and len(all_next_sclk) > 2:
        eff = np.array(all_effort_vals[:-1])
        nxt = np.array(all_next_sclk[1:])
        if np.std(eff) > 1e-6 and np.std(nxt) > 1e-6:
            temporal_r = float(np.corrcoef(eff, nxt)[0, 1])

    total_bat = energy_batches['high'] + energy_batches['low']
    high_frac = energy_batches['high'] / max(total_bat, 1)
    gate_p = 1.0
    if len(gates_h) > 2 and len(gates_l) > 2:
        _, gate_p = stats.ttest_ind(gates_h, gates_l)
        gate_p = float(gate_p)

    return {
        'acc': acc, 'sm_auroc': sm_auroc,
        'gate_high': float(np.mean(gates_h)) if gates_h else 0.5,
        'gate_low': float(np.mean(gates_l)) if gates_l else 0.5,
        'gate_p': gate_p, 'attn_kl': attn_kl,
        'effort_acc': effort_acc, 'temporal_r': temporal_r,
        'high_sclk_frac': high_frac,
    }


def ablate_self_model(m):
    if hasattr(m, 'self_model'):
        for p in m.self_model.parameters(): p.data.zero_()

def ablate_effort(m):
    for attr in ('effort_head',):
        if hasattr(m, attr):
            for p in getattr(m, attr).parameters(): p.data.zero_()


# ============================================================
# MAIN
# ============================================================
def main():
    global _EXT
    t0 = time.time()

    print("=" * 70, flush=True)
    print("z2075: Embodied Attention Transformer (token prediction)", flush=True)
    print("=" * 70, flush=True)
    print(f"Architecture: causal transformer, 4 layers, 4 heads, dim=128", flush=True)
    print(f"Task: next-token prediction on hardware-modulated arithmetic sequences", flush=True)
    print(f"Sensors ({SENSOR_DIM}): ISA(4) + MMIO(2) + gpu_metrics(10)", flush=True)
    print(f"Actuators (6): MODE, s_setprio, s_sleep, DVFS, mod_factor, chain_depth", flush=True)
    print(f"4-way exclusive specialization: DVFS×MODE → 4 different strides", flush=True)
    print(f"Training: Phase 1 ({PHASE1_EPOCHS}ep) + Phase 2 ({PHASE2_EPOCHS}ep)", flush=True)
    print(flush=True)

    print("Compiling HIP kernels...", flush=True)
    ext = load_inline(name='z2075_transformer', cpp_sources=CPP_SRC, cuda_sources=HIP_SRC,
                      functions=['hw_matmul', 'probe'],
                      extra_cuda_cflags=['-O2', '--offload-arch=gfx1100'],
                      verbose=False)
    _EXT = ext
    print("  Done.", flush=True)

    # Sanity checks
    print("\n--- Hardware checks ---", flush=True)
    x = torch.randn(BS, 128, device=DEVICE)
    w = torch.randn(512, 128, device=DEVICE) * 0.1
    b = torch.zeros(512, device=DEVICE)
    r0 = ext.hw_matmul(x, w, b, MODE_A, 8, 0, 0, 1.0)
    rF = ext.hw_matmul(x, w, b, MODE_B, 8, 0, 0, 1.0)
    torch.cuda.synchronize()
    print(f"  MODE {MODE_A:#05x} vs {MODE_B:#05x}: max|d|={((rF[0]-r0[0]).abs().max().item()):.6f}", flush=True)

    grbm = read_mmio_reg(0x8010)
    grbm2 = read_mmio_reg(0x8020)
    print(f"  GRBM_STATUS: 0x{grbm:08x}  GRBM_STATUS2: 0x{grbm2:08x}", flush=True)

    gm = read_gpu_metrics()
    if gm:
        print(f"  gpu_metrics: temp={gm['temp_gfx_c']:.1f}°C power={gm['gfx_power_w']:.1f}W "
              f"clock={gm['gfxclk_mhz']}MHz", flush=True)

    # DVFS differentiation
    print("\n--- DVFS + MODE differentiation ---", flush=True)
    has_dvfs = set_dvfs('high')
    if has_dvfs:
        time.sleep(0.5)
        _ = torch.randn(256, 256, device=DEVICE) @ torch.randn(256, 256, device=DEVICE)
        torch.cuda.synchronize(); time.sleep(0.2)
        freq_h = measure_freq_est(ext)
        gm_h = read_gpu_metrics()
        set_dvfs('low'); time.sleep(0.5)
        _ = torch.randn(256, 256, device=DEVICE) @ torch.randn(256, 256, device=DEVICE)
        torch.cuda.synchronize(); time.sleep(0.2)
        freq_l = measure_freq_est(ext)
        gm_l = read_gpu_metrics()
        set_dvfs('auto')
        print(f"  freq_est: H={freq_h:.0f} L={freq_l:.0f} ratio={freq_h/max(freq_l,1):.2f}x", flush=True)
        if gm_h and gm_l:
            print(f"  power: H={gm_h['gfx_power_w']:.1f}W L={gm_l['gfx_power_w']:.1f}W", flush=True)
            print(f"  temp: H={gm_h['temp_gfx_c']:.1f}°C L={gm_l['temp_gfx_c']:.1f}°C", flush=True)
    print(f"  4 strides: {list(STRIDES.values())} (V={V})", flush=True)

    # ============================================================
    # A: FULL MODEL
    # ============================================================
    print(f"\n{'='*60}\nA: FULL (transformer + 4-way exclusive specialization)\n{'='*60}", flush=True)
    model_A = Z2075Model().to(DEVICE)
    info_A = train_model(model_A, ext, EPOCHS, 'A_full')
    baseline = info_A['baseline']
    m_A = evaluate(model_A, ext, baseline, model_controlled=True)
    print(f"  A: acc={m_A['acc']:.4f} AUROC={m_A['sm_auroc']:.4f} "
          f"gate_h={m_A['gate_high']:.3f} gate_l={m_A['gate_low']:.3f} "
          f"effort={m_A['effort_acc']:.3f} temporal_r={m_A['temporal_r']:.3f}", flush=True)

    # B: BLIND (no hardware tokens)
    print(f"\n{'='*60}\nB: BLIND (no hw tokens)\n{'='*60}", flush=True)
    model_B = Z2075Model(use_hw_tokens=False).to(DEVICE)
    train_model(model_B, ext, PHASE1_EPOCHS, 'B_blind', model_controls_dvfs=False)
    m_B = evaluate(model_B, ext, baseline)
    print(f"  B: acc={m_B['acc']:.4f}", flush=True)

    # C: SCRAMBLED sensors
    print(f"\n{'='*60}\nC: SCRAMBLED\n{'='*60}", flush=True)
    m_C = evaluate(model_A, ext, baseline, scramble=True)
    print(f"  C: acc={m_C['acc']:.4f}", flush=True)

    # D: NO SELF-MODEL (ablated)
    print(f"\n{'='*60}\nD: NO SELF-MODEL\n{'='*60}", flush=True)
    model_D = copy.deepcopy(model_A)
    ablate_self_model(model_D)
    m_D = evaluate(model_D, ext, baseline)
    print(f"  D: acc={m_D['acc']:.4f}", flush=True)

    # E: NO GATE (constant 0.5)
    print(f"\n{'='*60}\nE: NO GATE\n{'='*60}", flush=True)
    model_E = copy.deepcopy(model_A)
    model_E.use_gate = False
    m_E = evaluate(model_E, ext, baseline)
    print(f"  E: acc={m_E['acc']:.4f}", flush=True)

    # F: NO ATTENTION (no hw tokens, retrained)
    print(f"\n{'='*60}\nF: NO HW TOKENS (retrained)\n{'='*60}", flush=True)
    model_F = Z2075Model(use_hw_tokens=False).to(DEVICE)
    train_model(model_F, ext, EPOCHS, 'F_no_hw', model_controls_dvfs=False)
    m_F = evaluate(model_F, ext, baseline)
    print(f"  F: acc={m_F['acc']:.4f}", flush=True)

    # G: FIXED HIGH DVFS
    print(f"\n{'='*60}\nG: FIXED HIGH\n{'='*60}", flush=True)
    model_G = Z2075Model().to(DEVICE)
    train_model(model_G, ext, EPOCHS, 'G_fixed', fixed_dvfs='high')
    m_G = evaluate(model_G, ext, baseline, fixed_dvfs='high')
    print(f"  G: acc={m_G['acc']:.4f}", flush=True)

    # H: NO EFFORT (ablated)
    print(f"\n{'='*60}\nH: NO EFFORT\n{'='*60}", flush=True)
    model_H = copy.deepcopy(model_A)
    ablate_effort(model_H)
    m_H = evaluate(model_H, ext, baseline)
    print(f"  H: acc={m_H['acc']:.4f}", flush=True)

    # ============================================================
    # TEST BATTERY
    # ============================================================
    print(f"\n{'='*70}\nTEST RESULTS\n{'='*70}", flush=True)
    tests = {}
    max_abl = max(m_D['acc'], m_E['acc'], m_H['acc'])

    def t(name, cond, val_str):
        v = "PASS" if cond else "FAIL"
        tests[name] = {'verdict': v, 'val': val_str}
        print(f"  {v} | {name}: {val_str}", flush=True)

    t('T1_accuracy',      m_A['acc'] > 0.70,
      f"A={m_A['acc']*100:.1f}% > 70%")
    t('T2_self_model',    m_A['sm_auroc'] > 0.75,
      f"AUROC={m_A['sm_auroc']:.4f} > 0.75")
    t('T3_gate_sep',      abs(m_A['gate_high'] - m_A['gate_low']) > 0.15,
      f"|h-l|={abs(m_A['gate_high']-m_A['gate_low']):.3f} > 0.15")
    t('T4_gate_stat',     m_A['gate_p'] < 0.01,
      f"p={m_A['gate_p']:.6f} < 0.01")
    t('T5_embodiment',    (m_A['acc'] - m_B['acc']) > 0.15,
      f"A-B={((m_A['acc']-m_B['acc'])*100):.1f}pp > 15pp")
    t('T6_scramble',      (m_A['acc'] - m_C['acc']) > 0.10,
      f"A-C={((m_A['acc']-m_C['acc'])*100):.1f}pp > 10pp")
    t('T7_sm_causal',     (m_A['acc'] - m_D['acc']) > 0.10,
      f"A-D={((m_A['acc']-m_D['acc'])*100):.1f}pp > 10pp")
    t('T8_gate_causal',   (m_A['acc'] - m_E['acc']) > 0.08,
      f"A-E={((m_A['acc']-m_E['acc'])*100):.1f}pp > 8pp")
    t('T9_effort_causal', (m_A['acc'] - m_H['acc']) > 0.05,
      f"A-H={((m_A['acc']-m_H['acc'])*100):.1f}pp > 5pp")
    t('T10_effort_acc',   m_A['effort_acc'] > 0.80,
      f"effort={m_A['effort_acc']*100:.1f}% > 80%")
    t('T11_closed_loop',  abs(m_A['temporal_r']) > 0.50,
      f"|r|={abs(m_A['temporal_r']):.4f} > 0.50")
    t('T12_hw_tokens_help', m_A['acc'] > m_F['acc'],
      f"A={m_A['acc']*100:.1f}% > F={m_F['acc']*100:.1f}%")
    t('T13_attn_varies',  m_A['attn_kl'] > 0.01,
      f"KL={m_A['attn_kl']:.4f} > 0.01")
    t('T14_not_fixed_high', (m_A['acc'] - m_G['acc']) > -0.05,
      f"A-G={((m_A['acc']-m_G['acc'])*100):.1f}pp > -5pp")
    t('T15_energy',       m_A['high_sclk_frac'] < 0.75,
      f"hi_frac={m_A['high_sclk_frac']:.3f} < 0.75")
    t('T16_full_best',    m_A['acc'] > max_abl,
      f"A={m_A['acc']*100:.1f}% > max_abl={max_abl*100:.1f}%")

    pass_count = sum(1 for v in tests.values() if v['verdict'] == 'PASS')
    verdict = f"{pass_count}/{len(tests)} PASS"
    print(f"\n  VERDICT: {verdict}", flush=True)

    print(f"\n  Ablation summary:", flush=True)
    for label, m in [('A full', m_A), ('B blind', m_B), ('C scrambled', m_C),
                     ('D no-sm', m_D), ('E no-gate', m_E), ('F no-hw-tok', m_F),
                     ('G fixed-hi', m_G), ('H no-effort', m_H)]:
        gap = (m['acc'] - m_A['acc']) * 100
        auroc_s = f"AUROC={m['sm_auroc']:.3f}" if 'sm_auroc' in m else ""
        eff_s = f"eff={m.get('effort_acc', 0):.3f}" if 'effort_acc' in m else ""
        print(f"    {label:14s}: {m['acc']*100:.1f}% ({gap:+.1f}pp) {auroc_s} {eff_s}", flush=True)

    # Save results
    results = {
        'experiment': 'z2075_embodied_transformer',
        'architecture': 'causal_transformer_4L_4H_128D',
        'task': 'hardware_modulated_arithmetic_sequences',
        'vocab_size': V, 'seq_len': SEQ_LEN,
        'strides': {str(k): v for k, v in STRIDES.items()},
        'sensors': SENSOR_DIM, 'actuators': 6,
        'accuracies': {k: v['acc'] for k, v in
                       zip('ABCDEFGH', [m_A, m_B, m_C, m_D, m_E, m_F, m_G, m_H])},
        'self_model_auroc': m_A['sm_auroc'],
        'gate': {'high': m_A['gate_high'], 'low': m_A['gate_low'],
                 'sep': abs(m_A['gate_high'] - m_A['gate_low']), 'p': m_A['gate_p']},
        'effort': {'acc': m_A['effort_acc'], 'temporal_r': m_A['temporal_r']},
        'energy': {'high_sclk_frac': m_A['high_sclk_frac']},
        'attn_kl': m_A['attn_kl'],
        'tests': tests,
        'verdict': verdict, 'pass_count': pass_count,
        'total_tests': len(tests),
        'elapsed_s': int(time.time() - t0),
    }

    os.makedirs('results', exist_ok=True)
    with open('results/z2075_embodied_transformer.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/z2075_embodied_transformer.json", flush=True)
    print(f"Elapsed: {int(time.time()-t0)}s", flush=True)


if __name__ == '__main__':
    main()
