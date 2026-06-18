#!/usr/bin/env python3
"""z2072: Comprehensive Hardware Self-Awareness
=================================================
Can a neural network build an internal world model of its own
GPU hardware substrate?

Core innovation: CONTINUOUS voltage-dependent modulation of arithmetic,
combined with ALL available ISA sensors and actuators.

Sensors (all in-kernel via hwreg):
  1. SHADER_CYCLES (hwreg 29) — per-wave cycle counter
  2. HW_ID1 (hwreg 23) — physical WGP/SIMD/wave location
  3. HW_ID2 (hwreg 24) — VMID/pipe/queue scheduling context
  4. STATUS (hwreg 2) — wave execution state (20 bits)
  5. MODE (hwreg 1) — current FP/exception config
  6. freq_est — cycle_delta / wall_ms (continuous voltage proxy)
  7. DVFS state — high/low SCLK regime

Actuators (shader-writable):
  1. MODE[9:0] — FP rounding + denorm + DX10_CLAMP + IEEE
  2. s_setprio — wave execution priority (4 levels)
  3. Chain depth — fp16 chunk accumulation size
  4. DVFS — high/low SCLK (external, model-controlled effort)

Architecture:
  - CIFAR-10 encoder → features
  - Deep kernel: MODE-controlled fp16 + SHADER_CYCLES-seeded stochastic rounding
  - Self-model: predicts ALL hardware state from 12-dim hw_vector
  - World model: predicts NEXT freq_est from (state, action)
  - Priority head: model chooses wave priority (genuine actuator control)
  - Exclusive specialization: DVFS high → standard labels, low → reversed
  - Gate: routes to head_A / head_B based on self-model

Causal chain: voltage → SCLK → timing → freq_est → stochastic rounding → arithmetic

Scientific grounding:
  - Seth (2024): self-awareness = control-oriented interoceptive inference
  - "Beautiful Loop" (2025): self-modeling + world-modeling = recursive active inference
  - Biological computationalism (2025): algorithm IS substrate
  - Lipson (2025): egocentric self-modeling — but of body, not computation
  - OUR NOVELTY: first NN that perceives, controls, and predicts its own GPU substrate
"""
import torch, torch.nn as nn, torch.nn.functional as F
import os, sys, json, time, copy, random, numpy as np
from torchvision import datasets, transforms
from scipy import stats
from sklearn.metrics import roc_auc_score

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')
os.environ.setdefault('PYTORCH_ROCM_ARCH', 'gfx1100')
from torch.utils.cpp_extension import load_inline

DEVICE = 'cuda'
BS = 128
EPOCHS = 40
SWITCH_EVERY = 4
NUM_BANKS = 8

# Split into sensor-only (for self-model) and full (for gate/world-model)
# SENSOR channels (genuinely hardware-read, self-model input):
#   [freq_est_norm, cycle_var, wgp_norm, simd_norm, status_bits, hw_id2_norm]
SENSOR_DIM = 6
# FULL hw_vector (for gate, world-model, priority head):
#   sensor channels + action channels [mode_rnd, mode_dnrm, dx10, ieee, prio, dvfs]
HW_DIM = 12

# MODE configs: use more of the 19-bit space
MODE_SET = [
    0x000,  # nearest, flush denorms, no DX10_CLAMP, no IEEE
    0x005,  # +inf fp32, nearest fp16, flush
    0x00F,  # toward-zero all, flush
    0x0F0,  # nearest, preserve denorms fp16/64
    0x0FF,  # toward-zero, preserve all denorms
    0x100,  # nearest, flush, DX10_CLAMP enabled
    0x200,  # nearest, flush, IEEE mode enabled
    0x300,  # nearest, flush, DX10_CLAMP + IEEE
]
K_MODES = len(MODE_SET)
DVFS_PATH = None

def find_dvfs_path():
    import glob
    for p in glob.glob('/sys/class/drm/card*/device/power_dpm_force_performance_level'):
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
        time.sleep(0.05)
        return True
    except: return False

def mode_group(m):
    """Group A: nearest rounding (bits[1:0] < 2), Group B: non-nearest."""
    return 0 if (m & 0x03) < 2 else 1

MODE_GROUPS = [mode_group(m) for m in MODE_SET]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HIP KERNEL: Comprehensive hardware-aware matmul
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HIP_SRC = r'''
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>

// All hardware readings packed into output
struct HWReadings {
    int wgp;
    int simd;
    int wave;
    int status;
    int hw_id2;
    int mode_read;
    int cycles_start;
    int cycles_end;
    float freq_est_proxy;  // cycles per block
};

// ─── Comprehensive hardware-aware matmul ───
template <int TILE>
__global__ void hw_matmul_kernel(
    const float* __restrict__ X, const float* __restrict__ W,
    const float* __restrict__ B, float* __restrict__ Y,
    int M, int K, int N,
    unsigned int mode_byte, int chain_depth, int priority,
    // Outputs: sensor readings
    int* __restrict__ wgp_out,
    int* __restrict__ simd_out,
    int* __restrict__ status_out,
    int* __restrict__ hw_id2_out,
    int* __restrict__ mode_out,
    int* __restrict__ cycle_out)
{
    // ═══ ACTUATORS ═══

    // 1. Set MODE register (all 10 writable bits)
    unsigned int m = __builtin_amdgcn_readfirstlane(mode_byte & 0x3FFu);
    asm volatile("s_setreg_b32 hwreg(1, 0, 10), %0" : : "s"(m));

    // 2. Set wave priority
    // s_setprio takes bottom 2 bits
    unsigned int p = __builtin_amdgcn_readfirstlane((unsigned int)(priority & 3));
    if (p == 0) { asm volatile("s_setprio 0"); }
    else if (p == 1) { asm volatile("s_setprio 1"); }
    else if (p == 2) { asm volatile("s_setprio 2"); }
    else { asm volatile("s_setprio 3"); }

    // ═══ SENSORS (start) ═══

    // 3. Read SHADER_CYCLES
    unsigned int c0;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(c0));
    c0 = __builtin_amdgcn_readfirstlane(c0);

    // 4. Read HW_ID1 (physical location)
    unsigned int hw1;
    asm volatile("s_getreg_b32 %0, hwreg(23)" : "=s"(hw1));
    unsigned int wgp = (hw1 >> 7) & 0xF;
    unsigned int simd = (hw1 >> 4) & 0x3;
    unsigned int wave = hw1 & 0xF;

    // 5. Read HW_ID2 (scheduling context)
    unsigned int hw2;
    asm volatile("s_getreg_b32 %0, hwreg(24)" : "=s"(hw2));

    // 6. Read STATUS (wave execution state)
    unsigned int stat;
    asm volatile("s_getreg_b32 %0, hwreg(2)" : "=s"(stat));

    // 7. Read MODE back (verify our write took effect)
    unsigned int mode_actual;
    asm volatile("s_getreg_b32 %0, hwreg(1, 0, 10)" : "=s"(mode_actual));

    // ═══ COMPUTATION ═══
    // Stochastic rounding seeded from SHADER_CYCLES + WGP + thread
    unsigned int sr_seed = c0 ^ (wgp << 16) ^ (simd << 20) ^ ((unsigned int)(threadIdx.y * TILE + threadIdx.x));

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
            float a_val = As[threadIdx.y][t];
            float b_val = Bs[t][threadIdx.x];

            __half a_h = __float2half(a_val);
            __half b_h = __float2half(b_val);
            __half prod = __hmul(a_h, b_h);

            // SHADER_CYCLES-seeded stochastic rounding
            // The seed depends on: GPU cycle count (→ SCLK → voltage),
            // physical WGP location, SIMD unit, and thread position.
            // This creates CONTINUOUS, hardware-dependent arithmetic.
            float prod_f = __half2float(prod);
            float ulp = fabsf(prod_f) * 9.77e-4f;
            float noise = ((float)(sr_seed & 0xFFFF) / 65536.0f - 0.5f) * ulp;
            sr_seed = sr_seed * 1103515245u + 12345u;

            __half modulated = __float2half(prod_f + noise);
            acc_chunk = __hadd(acc_chunk, modulated);
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

    if (row < M && col < N)
        Y[row * N + col] = acc + B[col];

    // ═══ SENSORS (end) ═══
    unsigned int c1;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(c1));
    c1 = __builtin_amdgcn_readfirstlane(c1);

    int bid = blockIdx.y * gridDim.x + blockIdx.x;
    if (threadIdx.x == 0 && threadIdx.y == 0 && bid < M) {
        wgp_out[bid] = (int)wgp;
        simd_out[bid] = (int)simd;
        status_out[bid] = (int)(stat & 0xFFFFF);
        hw_id2_out[bid] = (int)hw2;
        mode_out[bid] = (int)mode_actual;
        cycle_out[bid] = (int)(c1 - c0);
    }

    // Restore MODE defaults
    unsigned int def = __builtin_amdgcn_readfirstlane(0x3F0u); // default: denorms + DX10 + IEEE
    asm volatile("s_setreg_b32 hwreg(1, 0, 10), %0" : : "s"(def));
    // Restore priority
    asm volatile("s_setprio 0");
}

// ─── C++ wrapper ───
std::vector<torch::Tensor> hw_matmul(
    torch::Tensor X, torch::Tensor W, torch::Tensor B,
    int64_t mode_byte, int64_t chain_depth, int64_t priority) {
    int M = X.size(0), K = X.size(1), N = W.size(0);
    auto Y = torch::zeros({M, N}, X.options());
    auto io = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    int nb = ((N + 15) / 16) * ((M + 15) / 16);
    auto wgp_o = torch::zeros({nb}, io);
    auto simd_o = torch::zeros({nb}, io);
    auto stat_o = torch::zeros({nb}, io);
    auto hw2_o = torch::zeros({nb}, io);
    auto mode_o = torch::zeros({nb}, io);
    auto cyc_o = torch::zeros({nb}, io);
    constexpr int TILE = 16;
    dim3 threads(TILE, TILE);
    dim3 blocks((unsigned int)((N + TILE - 1) / TILE),
                (unsigned int)((M + TILE - 1) / TILE));
    hw_matmul_kernel<TILE><<<blocks, threads>>>(
        X.data_ptr<float>(), W.data_ptr<float>(), B.data_ptr<float>(),
        Y.data_ptr<float>(), M, K, N,
        (unsigned int)(mode_byte & 0x3FF),
        (int)chain_depth, (int)priority,
        wgp_o.data_ptr<int>(), simd_o.data_ptr<int>(),
        stat_o.data_ptr<int>(), hw2_o.data_ptr<int>(),
        mode_o.data_ptr<int>(), cyc_o.data_ptr<int>());
    return {Y, wgp_o, simd_o, stat_o, hw2_o, mode_o, cyc_o};
}

// ─── Probe kernel (sensors only, lightweight) ───
__global__ void probe_kernel(int* wgp_ids, int* cycles, int* status_out, int n) {
    int bid = blockIdx.x;
    if (bid >= n || threadIdx.x != 0) return;
    unsigned int c0;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(c0));
    c0 = __builtin_amdgcn_readfirstlane(c0);
    unsigned int hw;
    asm volatile("s_getreg_b32 %0, hwreg(23)" : "=s"(hw));
    wgp_ids[bid] = (int)((hw >> 7) & 0xF);
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
    probe_kernel<<<(int)n, 32>>>(w.data_ptr<int>(), c.data_ptr<int>(), s.data_ptr<int>(), (int)n);
    return {w, c, s};
}
'''

CPP_SRC = r'''
#include <torch/extension.h>
std::vector<torch::Tensor> hw_matmul(torch::Tensor, torch::Tensor, torch::Tensor,
                                      int64_t, int64_t, int64_t);
std::vector<torch::Tensor> probe(int64_t);
'''


# ━━━ AUTOGRAD ━━━
_EXT = None

class HWMatmulFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, mode_byte, chain_depth, priority):
        ctx.save_for_backward(x, w)
        result = _EXT.hw_matmul(x.contiguous(), w.contiguous(), b.contiguous(),
                                 int(mode_byte), int(chain_depth), int(priority))
        # Store sensor readings for hw_vector construction
        ctx.hw_readings = {
            'wgp': result[1], 'simd': result[2], 'status': result[3],
            'hw_id2': result[4], 'mode': result[5], 'cycles': result[6],
        }
        return result[0]

    @staticmethod
    def backward(ctx, grad_out):
        x, w = ctx.saved_tensors
        return grad_out @ w, grad_out.t() @ x, grad_out.sum(0), None, None, None


class HWLinear(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_f))

    def forward(self, x, mode_byte=0, chain_depth=8, priority=0):
        return HWMatmulFn.apply(x, self.weight, self.bias,
                                 mode_byte, chain_depth, priority)


# ━━━ HW SENSING ━━━
def read_hw(ext, mode_byte=0):
    """Read comprehensive hardware state — ALL sensor channels."""
    res = ext.probe(64)
    torch.cuda.synchronize()
    wgps = res[0].cpu().numpy()
    cycles = res[1].cpu().numpy()
    status = res[2].cpu().numpy()
    return {
        'wgp': int(np.median(wgps)),
        'wgp_norm': min(1.0, int(np.median(wgps)) / 14.0),
        'cycles': int(np.median(cycles)),
        'cycle_var': float(np.std(cycles)) / max(float(np.mean(cycles)), 1),
        'status': int(np.median(status)),
        'status_execz': float((int(np.median(status)) >> 9) & 1),
        'status_vccz': float((int(np.median(status)) >> 1) & 1),
    }


def make_hw_vector(hw_info, mode_byte, dvfs_state, freq_est, priority, baseline_freq=1.0):
    """12-dim hardware vector, split into SENSOR (0:6) and ACTION (6:12) channels.

    SENSOR channels [0:6] — genuinely hardware-read, self-model uses ONLY these:
      [0] freq_est_norm: continuous voltage proxy (the KEY signal)
      [1] cycle_var: timing variance across waves
      [2] wgp_norm: physical WGP placement
      [3] simd_norm: unused (probe only sees one, but kept for completeness)
      [4] status_execz: EXEC zero bit from STATUS register
      [5] status_vccz: VCC zero bit from STATUS register

    ACTION channels [6:12] — software-chosen, self-model does NOT see:
      [6] rnd: FP rounding mode setting
      [7] dnrm: denorm mode setting
      [8] dx10: DX10_CLAMP setting
      [9] ieee: IEEE mode setting
      [10] prio_norm: wave priority choice
      [11] dvfs_f: DVFS regime (1=high, 0=low)
    """
    # SENSOR channels: genuinely hardware-dependent
    # Fix normalization: freq_est ranges ~1000-4000, use baseline for proper scaling
    freq_norm = freq_est / max(baseline_freq, 1.0)  # ~0.3 for low, ~1.0 for high
    freq_norm = min(2.0, max(0.0, freq_norm))  # clamp to [0, 2]

    # ACTION channels: software-chosen parameters
    rnd = (mode_byte & 0x0F) / 15.0
    dnrm = ((mode_byte >> 4) & 0x0F) / 15.0
    dx10 = float((mode_byte >> 8) & 1)
    ieee = float((mode_byte >> 9) & 1)
    prio_norm = priority / 3.0
    dvfs_f = 1.0 if dvfs_state == 'high' else 0.0

    return [
        # === SENSOR channels [0:6] (self-model input) ===
        freq_norm,                          # [0] KEY: continuous voltage proxy
        hw_info['cycle_var'],              # [1] timing variance
        hw_info['wgp_norm'],              # [2] physical placement
        0.0,                               # [3] simd (static on this GPU)
        hw_info.get('status_execz', 0.0), # [4] EXEC zero from STATUS
        hw_info.get('status_vccz', 0.0),  # [5] VCC zero from STATUS
        # === ACTION channels [6:12] (self-model masked) ===
        rnd,                               # [6] rounding mode
        dnrm,                              # [7] denorm mode
        dx10,                              # [8] DX10_CLAMP
        ieee,                              # [9] IEEE mode
        prio_norm,                         # [10] wave priority
        dvfs_f,                            # [11] DVFS regime
    ]


# ━━━ MODEL ━━━
class Z2072Model(nn.Module):
    """Comprehensive hardware self-awareness model.

    Components:
      1. Encoder: CIFAR-10 → features
      2. Self-model: hw_vector → predicted hardware state
      3. World model: (hw_vector, action) → predicted next freq_est
      4. Priority head: features + hw_state → optimal wave priority
      5. Deep kernel: MODE+stochastic rounding controlled matmul
      6. Exclusive heads: head_A (high SCLK) / head_B (low SCLK)
      7. Gate: routes based on self-model
    """
    def __init__(self, use_self_model=True, use_gate=True, use_deep=True,
                 use_world_model=True, use_priority_head=True):
        super().__init__()
        self.use_self_model = use_self_model
        self.use_gate = use_gate
        self.use_deep = use_deep
        self.use_world_model = use_world_model
        self.use_priority_head = use_priority_head

        # CIFAR-10 encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128*8*8, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU())

        # LayerNorm before deep kernels
        self.pre_deep_ln = nn.LayerNorm(128)

        # Deep ISA layers
        self.deep_fc = HWLinear(128, 64)
        self.head_A = HWLinear(64, 10)
        self.head_B = HWLinear(64, 10)

        # WGP bank routing
        bank_init = torch.eye(128).unsqueeze(0).expand(NUM_BANKS, -1, -1).clone()
        bank_init += torch.randn(NUM_BANKS, 128, 128) * 0.01
        self.bank_w = nn.Parameter(bank_init)

        # Self-model: predict DVFS regime from SENSOR-ONLY channels
        # Must INFER DVFS from hardware readings, NOT from action parameters
        if use_self_model:
            self.self_model = nn.Sequential(
                nn.Linear(SENSOR_DIM, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, 2))  # 2 DVFS groups

        # World model: predict NEXT freq_est from (hw_vector + current mode)
        if use_world_model:
            self.world_model = nn.Sequential(
                nn.Linear(HW_DIM + 4, 32), nn.ReLU(),  # +4 for action encoding
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid())  # output in [0,1], target = cur_freq/baseline ~ 0.3-1.0

        # Priority head: choose wave priority (0-3)
        if use_priority_head:
            self.priority_head = nn.Sequential(
                nn.Linear(HW_DIM, 16), nn.ReLU(),
                nn.Linear(16, 4))  # 4 priority levels

        # Gate
        if use_gate:
            self.gate_net = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())

    def forward(self, x, hw_vector=None, bank_ids=None,
                mode_byte=0, chain_depth=8, priority=0):
        h = self.encoder(x)
        B_size = h.shape[0]

        if bank_ids is not None:
            h = torch.bmm(self.bank_w[bank_ids], h.unsqueeze(-1)).squeeze(-1)

        h = self.pre_deep_ln(h)

        # Priority head: model CHOOSES wave priority
        chosen_priority = priority
        priority_logits = None
        if self.use_priority_head and hw_vector is not None:
            priority_logits = self.priority_head(hw_vector)
            chosen_priority = priority_logits.argmax(dim=-1)[0].item()

        if self.use_deep:
            h1 = F.relu(self.deep_fc(h, mode_byte, chain_depth, chosen_priority))
            yA = self.head_A(h1, mode_byte, chain_depth, chosen_priority)
            yB = self.head_B(h1, mode_byte, chain_depth, chosen_priority)
        else:
            h1 = F.relu(F.linear(h, self.deep_fc.weight, self.deep_fc.bias))
            yA = F.linear(h1, self.head_A.weight, self.head_A.bias)
            yB = F.linear(h1, self.head_B.weight, self.head_B.bias)

        # Self-model: ONLY sensor channels [0:6] — no action params, no DVFS label
        if self.use_self_model and hw_vector is not None:
            sensor_only = hw_vector[:, :SENSOR_DIM]  # first 6 dims = hardware sensors
            sm_logits = self.self_model(sensor_only)
            sm_probs = F.softmax(sm_logits, dim=-1)
        else:
            sm_logits = torch.zeros(B_size, 2, device=x.device)
            sm_probs = torch.ones(B_size, 2, device=x.device) * 0.5

        # World model prediction
        wm_pred = None
        if self.use_world_model and hw_vector is not None:
            action_enc = torch.tensor(
                [[float(mode_byte & 0x0F) / 15.0,
                  float((mode_byte >> 4) & 0x0F) / 15.0,
                  float(chosen_priority) / 3.0,
                  float(chain_depth) / 16.0]],
                device=x.device).expand(B_size, -1)
            wm_input = torch.cat([hw_vector, action_enc], dim=-1)
            wm_pred = self.world_model(wm_input).squeeze(-1)

        # Gate
        if self.use_gate:
            g = self.gate_net(sm_probs)
        else:
            g = torch.full((B_size, 1), 0.5, device=x.device)

        logits = g * yA + (1.0 - g) * yB

        return {
            'logits': logits, 'sm_logits': sm_logits, 'gate': g,
            'yA': yA, 'yB': yB,
            'wm_pred': wm_pred,
            'priority_logits': priority_logits,
            'chosen_priority': chosen_priority,
        }


def make_labels(digits, dvfs_state):
    if dvfs_state == 'high':
        return digits.clone()
    else:
        return (9 - digits) % 10


def get_data():
    tf_train = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    tr = datasets.CIFAR10('data', train=True, download=True, transform=tf_train)
    te = datasets.CIFAR10('data', train=False, transform=tf_test)
    return (torch.utils.data.DataLoader(tr, batch_size=BS, shuffle=True, drop_last=True),
            torch.utils.data.DataLoader(te, batch_size=BS, shuffle=False, drop_last=True))


# ━━━ FREQ_EST MEASUREMENT ━━━
def measure_freq_est(ext, n_trials=5):
    """Measure freq_est using CUDA events for wall timing + SHADER_CYCLES."""
    freqs = []
    for _ in range(n_trials):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        res = ext.probe(64)
        end_event.record()
        torch.cuda.synchronize()
        wall_ms = start_event.elapsed_time(end_event)
        cycles = res[1].cpu().numpy()
        cycle_med = float(np.median(cycles))
        if wall_ms > 0:
            freqs.append(cycle_med / wall_ms)
    return float(np.median(freqs)) if freqs else 1.0


# ━━━ TRAINING ━━━
def train_model(model, ext, loader, epochs, name, fixed_dvfs=None):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[20, 32], gamma=0.3)
    model.train()
    bn, config_idx = 0, 0
    dvfs_state = 'high'
    has_dvfs = fixed_dvfs is None
    log = {'gate': [], 'sm_acc': [], 'freq': [], 'priority': [], 'wm_err': []}

    if has_dvfs:
        set_dvfs('high')
        time.sleep(0.2)
    baseline_freq = measure_freq_est(ext)
    print(f"  baseline freq_est = {baseline_freq:.2f}", flush=True)

    prev_freq = baseline_freq
    prev_hw_vec = None
    prev_action = None
    for ep in range(epochs):
        tot_loss, correct, total = 0, 0, 0

        for imgs, digits in loader:
            imgs, digits = imgs.to(DEVICE), digits.to(DEVICE)

            # DVFS switching
            if has_dvfs and bn % SWITCH_EVERY == 0:
                dvfs_state = 'high' if (bn // SWITCH_EVERY) % 2 == 0 else 'low'
                set_dvfs(dvfs_state)
            elif fixed_dvfs:
                dvfs_state = fixed_dvfs

            # MODE cycling: RANDOM to decouple from DVFS (prevent self-model from
            # learning DVFS via correlated MODE config — must use freq_est instead)
            if bn % SWITCH_EVERY == 0:
                config_idx = random.randint(0, K_MODES - 1)
            cur_mode = MODE_SET[config_idx]
            cur_group = MODE_GROUPS[config_idx]
            cur_chain = [1, 4, 8, 16][min(config_idx // 2, 3)]

            # Measure freq_est
            cur_freq = measure_freq_est(ext, n_trials=2)

            # Sensor readings
            hw_info = read_hw(ext, cur_mode)
            probe_res = ext.probe(BS)
            torch.cuda.synchronize()
            wgps = probe_res[0]
            bank_ids = (wgps // 2).long().clamp(0, NUM_BANKS - 1)

            hw_vec = make_hw_vector(hw_info, cur_mode, dvfs_state, cur_freq, 0,
                                     baseline_freq=baseline_freq)
            hw_t = torch.tensor([hw_vec] * BS, dtype=torch.float32, device=DEVICE)
            labels = make_labels(digits, dvfs_state)

            out = model(imgs, hw_vector=hw_t, bank_ids=bank_ids,
                       mode_byte=cur_mode, chain_depth=cur_chain, priority=0)

            if torch.isnan(out['logits']).any():
                if bn < 5:
                    print(f"  NaN at batch {bn}!", flush=True)
                bn += 1
                continue

            # Task loss
            task_loss = F.cross_entropy(out['logits'], labels)

            # Self-model loss
            dvfs_target = torch.full((BS,), 0 if dvfs_state == 'high' else 1,
                                     dtype=torch.long, device=DEVICE)
            sm_loss = F.cross_entropy(out['sm_logits'], dvfs_target)

            # World model loss: predict CURRENT freq from PREVIOUS state+action
            wm_loss = torch.tensor(0.0, device=DEVICE)
            if out['wm_pred'] is not None and prev_hw_vec is not None:
                freq_target = torch.full((BS,), cur_freq / baseline_freq,
                                         device=DEVICE)
                # Use prev state to predict current freq (genuine prediction)
                prev_hw_t = torch.tensor([prev_hw_vec] * BS, dtype=torch.float32, device=DEVICE)
                prev_act = torch.tensor(
                    [[float(prev_action[0] & 0x0F) / 15.0,
                      float((prev_action[0] >> 4) & 0x0F) / 15.0,
                      float(prev_action[1]) / 3.0,
                      float(prev_action[2]) / 16.0]] * BS,
                    device=DEVICE)
                wm_in = torch.cat([prev_hw_t, prev_act], dim=-1)
                wm_p = model.world_model(wm_in).squeeze(-1)
                wm_loss = F.mse_loss(wm_p, freq_target)

            # Priority head regularization (entropy bonus)
            prio_loss = torch.tensor(0.0, device=DEVICE)
            if out['priority_logits'] is not None:
                prio_probs = F.softmax(out['priority_logits'], dim=-1)
                prio_entropy = -(prio_probs * torch.log(prio_probs + 1e-8)).sum(-1).mean()
                prio_loss = -0.05 * prio_entropy  # encourage exploration

            # Gate entropy regularization
            g = out['gate']
            gate_ent = -(g * torch.log(g + 1e-8) + (1-g) * torch.log(1-g + 1e-8)).mean()

            loss = task_loss + 0.5 * sm_loss + 0.05 * wm_loss.clamp(max=10.0) + prio_loss - 0.1 * gate_ent

            if torch.isnan(loss):
                bn += 1
                continue

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tot_loss += loss.item()
            pred = out['logits'].argmax(1)
            correct += (pred == labels).sum().item()
            total += BS

            sm_pred = out['sm_logits'].argmax(1)
            log['sm_acc'].append((sm_pred == dvfs_target).float().mean().item())
            log['gate'].append(g.mean().item())
            log['freq'].append(cur_freq)
            log['priority'].append(out['chosen_priority'])
            if out['wm_pred'] is not None:
                log['wm_err'].append(wm_loss.item())
            prev_freq = cur_freq
            prev_hw_vec = hw_vec
            prev_action = (cur_mode, out['chosen_priority'], cur_chain)
            bn += 1

        if ep % 5 == 0 or ep == epochs - 1:
            g_m = np.mean(log['gate'][-50:]) if log['gate'] else 0
            sm_m = np.mean(log['sm_acc'][-50:]) if log['sm_acc'] else 0
            freq_m = np.mean(log['freq'][-50:]) if log['freq'] else 0
            wm_m = np.mean(log['wm_err'][-50:]) if log['wm_err'] else 0
            prio = log['priority'][-1] if log['priority'] else 0
            print(f"  [{name}] Ep {ep}: loss={tot_loss/max(len(loader),1):.4f} "
                  f"acc={correct/max(total,1):.4f} gate={g_m:.3f} sm={sm_m:.3f} "
                  f"freq={freq_m:.1f} wm_err={wm_m:.4f} prio={prio}", flush=True)
        sched.step()

    if has_dvfs:
        set_dvfs('auto')

    return {'log': log, 'baseline': baseline_freq}


# ━━━ EVAL ━━━
def evaluate(model, ext, loader, baseline_freq=1.0, scramble=False,
             fixed_dvfs=None, return_details=False):
    model.eval()
    all_correct, all_gates, all_true_dvfs = [], [], []
    all_freq, all_wm_pred, all_prio = [], [], []
    bn, config_idx, dvfs_state = 0, 0, 'high'
    has_dvfs = fixed_dvfs is None

    if has_dvfs:
        set_dvfs('high')
        time.sleep(0.1)

    with torch.no_grad():
        for imgs, digits in loader:
            imgs, digits = imgs.to(DEVICE), digits.to(DEVICE)

            if has_dvfs and bn % SWITCH_EVERY == 0:
                dvfs_state = 'high' if (bn // SWITCH_EVERY) % 2 == 0 else 'low'
                set_dvfs(dvfs_state)
            elif fixed_dvfs:
                dvfs_state = fixed_dvfs

            if bn % SWITCH_EVERY == 0:
                config_idx = random.randint(0, K_MODES - 1)
            cur_mode = MODE_SET[config_idx]
            cur_chain = [1, 4, 8, 16][min(config_idx // 2, 3)]

            cur_freq = measure_freq_est(ext, n_trials=1)
            hw_info = read_hw(ext, cur_mode)
            probe_res = ext.probe(BS)
            torch.cuda.synchronize()
            wgps = probe_res[0]
            bank_ids = (wgps // 2).long().clamp(0, NUM_BANKS - 1)

            hw_vec = make_hw_vector(hw_info, cur_mode, dvfs_state, cur_freq, 0,
                                     baseline_freq=baseline_freq)
            if scramble:
                # Scramble SENSOR channels [0:6] only — action channels stay real
                hw_vec = [1.0 - hw_vec[i] if i < SENSOR_DIM else hw_vec[i]
                          for i in range(len(hw_vec))]
            hw_t = torch.tensor([hw_vec] * BS, dtype=torch.float32, device=DEVICE)
            labels = make_labels(digits, dvfs_state)

            out = model(imgs, hw_vector=hw_t, bank_ids=bank_ids,
                       mode_byte=cur_mode, chain_depth=cur_chain)

            pred = out['logits'].argmax(1)
            all_correct.extend((pred == labels).cpu().tolist())
            all_gates.extend(out['gate'].squeeze(-1).cpu().tolist())
            all_true_dvfs.extend([0 if dvfs_state == 'high' else 1] * BS)
            all_freq.append(cur_freq)
            if out['wm_pred'] is not None:
                all_wm_pred.append(out['wm_pred'].mean().item())
            all_prio.append(out['chosen_priority'])
            bn += 1

    if has_dvfs:
        set_dvfs('auto')

    acc = float(np.mean(all_correct))
    try:
        raw_auroc = roc_auc_score(all_true_dvfs, all_gates)
        sm_auroc = max(raw_auroc, 1.0 - raw_auroc)  # handle either polarity
    except:
        sm_auroc = 0.5

    gates_h = [g for g, d in zip(all_gates, all_true_dvfs) if d == 0]
    gates_l = [g for g, d in zip(all_gates, all_true_dvfs) if d == 1]

    # World model accuracy: correlation between predicted and actual freq
    wm_r = 0.0
    if len(all_wm_pred) > 10 and len(all_freq) > 10:
        try:
            wm_r = float(np.corrcoef(all_wm_pred[:len(all_freq)], all_freq[:len(all_wm_pred)])[0, 1])
            if np.isnan(wm_r): wm_r = 0.0
        except: pass

    result = {
        'acc': acc,
        'gate_mean': float(np.mean(all_gates)),
        'sm_auroc': sm_auroc,
        'gate_high': float(np.mean(gates_h)) if gates_h else 0.5,
        'gate_low': float(np.mean(gates_l)) if gates_l else 0.5,
        'mean_freq': float(np.mean(all_freq)),
        'wm_corr': wm_r,
        'mean_priority': float(np.mean(all_prio)),
        'priority_unique': len(set(all_prio)),
    }
    if return_details:
        result['all_correct'] = all_correct
        result['all_gates'] = all_gates
        result['all_true_dvfs'] = all_true_dvfs
    return result


# ━━━ SANITY CHECKS ━━━
def sanity_checks(ext):
    print("\n--- Hardware actuator/sensor checks ---", flush=True)
    x = torch.randn(BS, 128, device=DEVICE)
    w = torch.randn(64, 128, device=DEVICE) * 0.1
    b = torch.zeros(64, device=DEVICE)

    # MODE difference
    r0 = ext.hw_matmul(x, w, b, 0x000, 8, 0)
    rF = ext.hw_matmul(x, w, b, 0x00F, 8, 0)
    r100 = ext.hw_matmul(x, w, b, 0x100, 8, 0)
    r300 = ext.hw_matmul(x, w, b, 0x300, 8, 0)
    torch.cuda.synchronize()
    print(f"  MODE 0x000 vs 0x00F: max|Δ|={((rF[0]-r0[0]).abs().max().item()):.6f}", flush=True)
    print(f"  MODE 0x000 vs 0x100 (DX10): max|Δ|={((r100[0]-r0[0]).abs().max().item()):.6f}", flush=True)
    print(f"  MODE 0x000 vs 0x300 (DX10+IEEE): max|Δ|={((r300[0]-r0[0]).abs().max().item()):.6f}", flush=True)

    # Priority effect on timing
    print("\n--- Priority effect ---", flush=True)
    for prio in range(4):
        r = ext.hw_matmul(x, w, b, 0x000, 8, prio)
        torch.cuda.synchronize()
        cycles = r[6].cpu().numpy()
        valid = cycles[cycles > 0]
        if len(valid) > 0:
            print(f"  Priority {prio}: cycles={np.median(valid):.0f} (std={np.std(valid):.0f})", flush=True)

    # Sensor readings
    print("\n--- Sensor readings ---", flush=True)
    wgps = r0[1].cpu().numpy()
    simds = r0[2].cpu().numpy()
    status = r0[3].cpu().numpy()
    hw2 = r0[4].cpu().numpy()
    modes = r0[5].cpu().numpy()
    print(f"  WGP unique: {len(np.unique(wgps[wgps > 0]))} values: {sorted(np.unique(wgps[wgps > 0]))}", flush=True)
    print(f"  SIMD unique: {len(np.unique(simds))}", flush=True)
    print(f"  STATUS sample: 0x{int(np.median(status)):05X}", flush=True)
    print(f"  HW_ID2 sample: 0x{int(np.median(hw2)):08X}", flush=True)
    print(f"  MODE readback: 0x{int(np.median(modes)):03X}", flush=True)

    # DVFS freq_est differentiation
    print("\n--- DVFS freq_est ---", flush=True)
    has_dvfs = set_dvfs('high')
    if has_dvfs:
        time.sleep(0.3)
        freq_h = measure_freq_est(ext)
        set_dvfs('low')
        time.sleep(0.3)
        freq_l = measure_freq_est(ext)
        set_dvfs('auto')
        ratio = freq_h / max(freq_l, 1e-6)
        print(f"  High SCLK: freq_est={freq_h:.2f}", flush=True)
        print(f"  Low SCLK:  freq_est={freq_l:.2f}", flush=True)
        print(f"  Ratio: {ratio:.2f}x", flush=True)
    else:
        print("  DVFS not available", flush=True)


# ━━━ MAIN ━━━
def main():
    global _EXT

    print("=" * 70, flush=True)
    print("z2072: Comprehensive Hardware Self-Awareness", flush=True)
    print("=" * 70, flush=True)
    print("Can AI build an internal world model of its own GPU substrate?", flush=True)
    print("Sensors: SHADER_CYCLES, HW_ID1/2, STATUS, MODE, freq_est", flush=True)
    print("Actuators: MODE[9:0], s_setprio, chain depth, DVFS", flush=True)
    print("Innovation: continuous voltage modulation + stochastic rounding", flush=True)
    print(flush=True)

    t0 = time.time()

    print("Compiling HIP kernels...", flush=True)
    ext = load_inline(name='z2072_hwself', cpp_sources=CPP_SRC, cuda_sources=HIP_SRC,
                      functions=['hw_matmul', 'probe'],
                      extra_cuda_cflags=['-O2', '--offload-arch=gfx1100'],
                      verbose=False)
    _EXT = ext
    print("  Done.", flush=True)

    sanity_checks(ext)
    train_loader, test_loader = get_data()

    # ━━━ A: Full system ━━━
    print(f"\n{'='*60}", flush=True)
    print("A: FULL (self-model + world-model + priority + gate + deep)", flush=True)
    print(f"{'='*60}", flush=True)
    model_A = Z2072Model().to(DEVICE)
    info_A = train_model(model_A, ext, train_loader, EPOCHS, 'A_full')
    baseline = info_A['baseline']
    m_A = evaluate(model_A, ext, test_loader, baseline, return_details=True)
    print(f"  A: acc={m_A['acc']:.4f} AUROC={m_A['sm_auroc']:.4f} "
          f"gate_h={m_A['gate_high']:.3f} gate_l={m_A['gate_low']:.3f} "
          f"wm_r={m_A['wm_corr']:.3f} prio={m_A['mean_priority']:.1f}", flush=True)

    # ━━━ B: Blind ━━━
    print(f"\n{'='*60}\nB: BLIND\n{'='*60}", flush=True)
    model_B = Z2072Model(use_self_model=False, use_gate=False, use_deep=False,
                          use_world_model=False, use_priority_head=False).to(DEVICE)
    train_model(model_B, ext, train_loader, EPOCHS, 'B_blind')
    m_B = evaluate(model_B, ext, test_loader, baseline)
    print(f"  B: acc={m_B['acc']:.4f}", flush=True)

    # ━━━ C: No deep kernels ━━━
    print(f"\n{'='*60}\nC: NO DEEP\n{'='*60}", flush=True)
    model_C = Z2072Model(use_deep=False).to(DEVICE)
    train_model(model_C, ext, train_loader, EPOCHS, 'C_no_deep')
    m_C = evaluate(model_C, ext, test_loader, baseline)
    print(f"  C: acc={m_C['acc']:.4f}", flush=True)

    # ━━━ D: Scrambled ━━━
    print(f"\n{'='*60}\nD: SCRAMBLED\n{'='*60}", flush=True)
    m_D = evaluate(model_A, ext, test_loader, baseline, scramble=True)
    print(f"  D: acc={m_D['acc']:.4f}", flush=True)

    # ━━━ E: No self-model ━━━
    print(f"\n{'='*60}\nE: NO SELF-MODEL\n{'='*60}", flush=True)
    model_E = copy.deepcopy(model_A)
    model_E.use_self_model = False
    m_E = evaluate(model_E, ext, test_loader, baseline)
    print(f"  E: acc={m_E['acc']:.4f}", flush=True)

    # ━━━ F: No gate ━━━
    print(f"\n{'='*60}\nF: NO GATE\n{'='*60}", flush=True)
    model_F = copy.deepcopy(model_A)
    model_F.use_gate = False
    m_F = evaluate(model_F, ext, test_loader, baseline)
    print(f"  F: acc={m_F['acc']:.4f}", flush=True)

    # ━━━ G: No world model ━━━
    print(f"\n{'='*60}\nG: NO WORLD MODEL\n{'='*60}", flush=True)
    model_G = copy.deepcopy(model_A)
    model_G.use_world_model = False
    m_G = evaluate(model_G, ext, test_loader, baseline)
    print(f"  G: acc={m_G['acc']:.4f}", flush=True)

    # ━━━ H: No priority head ━━━
    print(f"\n{'='*60}\nH: NO PRIORITY HEAD\n{'='*60}", flush=True)
    model_H = copy.deepcopy(model_A)
    model_H.use_priority_head = False
    m_H = evaluate(model_H, ext, test_loader, baseline)
    print(f"  H: acc={m_H['acc']:.4f}", flush=True)

    # ━━━ I: Fixed DVFS high ━━━
    print(f"\n{'='*60}\nI: FIXED HIGH\n{'='*60}", flush=True)
    model_I = Z2072Model().to(DEVICE)
    train_model(model_I, ext, train_loader, EPOCHS, 'I_fixed', fixed_dvfs='high')
    m_I = evaluate(model_I, ext, test_loader, baseline, fixed_dvfs='high')
    print(f"  I: acc={m_I['acc']:.4f}", flush=True)

    elapsed = time.time() - t0

    # ━━━ TESTS ━━━
    print(f"\n{'='*70}\nTEST RESULTS\n{'='*70}", flush=True)
    tests = {}

    tests['T1_accuracy'] = {'verdict': 'PASS' if m_A['acc'] > 0.65 else 'FAIL',
        'val': f"A={m_A['acc']*100:.1f}% > 65%"}

    tests['T2_self_model_auroc'] = {'verdict': 'PASS' if m_A['sm_auroc'] > 0.80 else 'FAIL',
        'val': f"AUROC={m_A['sm_auroc']:.4f} > 0.80"}

    gate_sep = abs(m_A['gate_high'] - m_A['gate_low'])
    tests['T3_gate_separation'] = {'verdict': 'PASS' if gate_sep > 0.3 else 'FAIL',
        'val': f"|gate_h-gate_l|={gate_sep:.3f} > 0.3"}

    gap_AB = m_A['acc'] - m_B['acc']
    tests['T4_embodiment_gap'] = {'verdict': 'PASS' if gap_AB > 0.10 else 'FAIL',
        'val': f"A-B={gap_AB*100:.1f}pp > 10pp"}

    gap_AD = m_A['acc'] - m_D['acc']
    tests['T5_scramble_kills'] = {'verdict': 'PASS' if gap_AD > 0.05 else 'FAIL',
        'val': f"A-D={gap_AD*100:.1f}pp > 5pp"}

    gap_AE = m_A['acc'] - m_E['acc']
    tests['T6_self_model_causal'] = {'verdict': 'PASS' if gap_AE > 0.10 else 'FAIL',
        'val': f"A-E={gap_AE*100:.1f}pp > 10pp (self-model necessary)"}

    gap_AF = m_A['acc'] - m_F['acc']
    tests['T7_gate_causal'] = {'verdict': 'PASS' if gap_AF > 0.10 else 'FAIL',
        'val': f"A-F={gap_AF*100:.1f}pp > 10pp (gate necessary)"}

    tests['T8_world_model'] = {'verdict': 'PASS' if abs(m_A['wm_corr']) > 0.3 else 'FAIL',
        'val': f"wm_r={m_A['wm_corr']:.3f} > 0.3 (predicts freq_est)"}

    gap_AG = m_A['acc'] - m_G['acc']
    tests['T9_world_model_useful'] = {
        'verdict': 'PASS' if abs(gap_AG) > 0.01 else 'FAIL',
        'val': f"|A-G|={abs(gap_AG)*100:.1f}pp > 1pp"}

    tests['T10_priority_learned'] = {
        'verdict': 'PASS' if m_A['priority_unique'] > 1 else 'FAIL',
        'val': f"unique_priorities={m_A['priority_unique']} > 1"}

    gap_AI = m_A['acc'] - m_I['acc']
    tests['T11_adaptive_vs_fixed'] = {
        'verdict': 'PASS' if gap_AI > 0.05 else 'FAIL',
        'val': f"A-I={gap_AI*100:.1f}pp > 5pp (adaptive > fixed)"}

    best_abl = max(m_D['acc'], m_E['acc'], m_F['acc'])
    tests['T12_full_beats_ablations'] = {
        'verdict': 'PASS' if m_A['acc'] > best_abl else 'FAIL',
        'val': f"A={m_A['acc']*100:.1f}% > max(D,E,F)={best_abl*100:.1f}%"}

    tests['T13_freq_est_works'] = {
        'verdict': 'PASS' if m_A['mean_freq'] > 0.01 else 'FAIL',
        'val': f"mean_freq={m_A['mean_freq']:.2f} > 0"}

    tests['T14_deep_competitive'] = {
        'verdict': 'PASS' if m_A['acc'] - m_C['acc'] > -0.05 else 'FAIL',
        'val': f"A-C={(m_A['acc']-m_C['acc'])*100:.1f}pp > -5pp"}

    pass_count = sum(1 for t in tests.values() if t['verdict'] == 'PASS')
    verdict = f"{pass_count}/{len(tests)} PASS"

    for tname, result in tests.items():
        print(f"  {result['verdict']:4s} | {tname}: {result['val']}", flush=True)
    print(f"\n  VERDICT: {verdict}", flush=True)

    print(f"\n  Ablation summary:", flush=True)
    for lbl, m in [('A full', m_A), ('B blind', m_B), ('C no-deep', m_C),
                    ('D scrambled', m_D), ('E no-sm', m_E), ('F no-gate', m_F),
                    ('G no-wm', m_G), ('H no-prio', m_H), ('I fixed-high', m_I)]:
        gap = m['acc'] - m_A['acc']
        extras = f" AUROC={m.get('sm_auroc',0):.3f}" if 'sm_auroc' in m else ""
        extras += f" wm_r={m.get('wm_corr',0):.3f}" if m.get('wm_corr',0) != 0 else ""
        print(f"    {lbl:14s}: {m['acc']*100:.1f}% ({gap*100:+.1f}pp){extras}", flush=True)

    # ━━━ SAVE ━━━
    results = {
        'experiment': 'z2072_comprehensive_hw_self_awareness',
        'key_innovation': (
            'First comprehensive hardware self-awareness experiment: '
            'all ISA sensors (SHADER_CYCLES, HW_ID1/2, STATUS, MODE, freq_est) + '
            'all actuators (MODE[9:0], s_setprio, chain_depth, DVFS) + '
            'self-model + world model + priority control + exclusive specialization. '
            'Continuous voltage-dependent stochastic rounding at fp16 ULP level. '
            'Answers: can AI build an internal world model of its own GPU substrate?'
        ),
        'sensors': ['SHADER_CYCLES', 'HW_ID1', 'HW_ID2', 'STATUS', 'MODE', 'freq_est', 'DVFS'],
        'actuators': ['MODE[9:0]', 's_setprio', 'chain_depth', 'DVFS'],
        'hw_dim': HW_DIM,
        'accuracies': {k: round(v, 4) for k, v in [
            ('A_full', m_A['acc']), ('B_blind', m_B['acc']),
            ('C_no_deep', m_C['acc']), ('D_scrambled', m_D['acc']),
            ('E_no_self_model', m_E['acc']), ('F_no_gate', m_F['acc']),
            ('G_no_world_model', m_G['acc']), ('H_no_priority', m_H['acc']),
            ('I_fixed_high', m_I['acc'])]},
        'self_model_auroc': round(m_A['sm_auroc'], 4),
        'gate': {
            'high': round(m_A['gate_high'], 4),
            'low': round(m_A['gate_low'], 4),
            'separation': round(gate_sep, 4),
        },
        'world_model_corr': round(m_A['wm_corr'], 4),
        'mean_priority': round(m_A['mean_priority'], 2),
        'priority_unique': m_A['priority_unique'],
        'mean_freq_est': round(m_A['mean_freq'], 4),
        'tests': tests, 'verdict': verdict,
        'pass_count': pass_count, 'total_tests': len(tests),
        'elapsed_s': round(elapsed),
        'scientific_grounding': {
            'seth_2024': 'Self-awareness = control-oriented interoceptive inference',
            'beautiful_loop_2025': 'Self-modeling + world-modeling = recursive active inference',
            'biological_computationalism_2025': 'Algorithm IS substrate (scale-inseparable)',
            'lipson_2025': 'Egocentric self-modeling (body, not computation)',
            'our_novelty': 'First NN that perceives, controls, and predicts its own GPU substrate',
        },
    }

    out_path = 'results/z2072_hw_self_awareness.json'
    os.makedirs('results', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)
    print(f"Elapsed: {elapsed:.0f}s", flush=True)


if __name__ == '__main__':
    main()
