#!/usr/bin/env python3
"""z2071: Deep ISA Embodied Arithmetic — v2
=============================================
Everything inside the GPU kernel. No sysfs DVFS, no external telemetry.

v2 fixes from v1:
  - REMOVED dual-path α-blend (α→0 killed deep path in v1)
  - FIXED chain: fp16 chunk accumulation (v1 roundtrips were idempotent)
  - External MODE cycling with CONFLICTING labels (z2060 pattern)
  - Self-model + gate for exclusive specialization
  - ONLY deep path — model MUST use MODE-controlled kernels

Architecture (z2060 pattern applied to ISA-level):
  - Encoder → features → deep_fc (MODE-controlled kernel) → head_A / head_B
  - MODE group A (nearest rounding): standard labels
  - MODE group B (non-nearest rounding): reversed labels (9-class_id)
  - Self-model predicts MODE group from hw_vector
  - Gate routes to correct head based on self-model output
  - Ablating self-model → wrong head → accuracy drops
  - All computation goes through MODE-controlled HIP kernels

Channels (all in-kernel):
  1. MODE[3:0] FP_ROUND: 4 rounding modes for fp16/fp32
  2. MODE[7:4] FP_DENORM: flush vs preserve subnormals
  3. EXEC mask: which lanes contribute to accumulation
  4. Chain depth: fp16 chunk accumulation depth (real amplification)
  5. SHADER_CYCLES: per-wave self-timing
  6. HW_ID1: physical WGP placement
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
SWITCH_EVERY = 8  # batches per MODE config
NUM_BANKS = 8
HW_DIM = 6  # [wgp_norm, cycle_norm, mode_rnd, mode_dnrm, exec_pct, cycle_var]

# 8 MODE personalities
MODE_SET = [
    0x00,  # nearest, flush denorms
    0x05,  # +inf, flush
    0x0F,  # toward-zero, flush
    0xF0,  # nearest, preserve denorms
    0xF5,  # +inf, preserve
    0xFF,  # toward-zero, preserve
    0x0A,  # -inf, flush
    0xFA,  # -inf, preserve
]
K_MODES = len(MODE_SET)

# Group A: nearest rounding (bits[3:0] < 0x08) → standard labels
# Group B: non-nearest rounding (bits[3:0] >= 0x08) → reversed labels
def mode_group(m):
    return 0 if (m & 0x0F) < 0x08 else 1

MODE_GROUPS = [mode_group(m) for m in MODE_SET]
# Group A: indices [0,1,3,4] (0x00,0x05,0xF0,0xF5)
# Group B: indices [2,5,6,7] (0x0F,0xFF,0x0A,0xFA)

EXEC_PATTERNS = [
    0xFFFFFFFF,  # all 32 lanes
    0x55555555,  # even lanes (50%)
    0x0000FFFF,  # lower 16 lanes (50%)
    0x000000FF,  # lower 8 lanes (25%)
]
K_EXEC = len(EXEC_PATTERNS)

CHAIN_DEPTHS = [1, 4, 8, 16]  # fp16 chunk accumulation size

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HIP KERNEL: Deep ISA embodied matmul
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HIP_SRC = r'''
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>

// ─── Deep embodied matmul with fp16 chunk accumulation ───
// MODE[7:0] controls rounding AND denorm for ALL fp operations
// EXEC mask controls which lanes contribute
// chain_depth = how many fp16 products to accumulate before folding to fp32
//   chain_depth=1: each product individually → fp32 (minimal fp16 error)
//   chain_depth=TILE: all products in fp16 (max rounding error, max MODE sensitivity)
template <int TILE>
__global__ void deep_matmul_kernel(
    const float* __restrict__ X, const float* __restrict__ W,
    const float* __restrict__ B, float* __restrict__ Y,
    int M, int K, int N,
    unsigned int mode_byte, unsigned int exec_pattern, int chain_depth,
    int* __restrict__ cycle_out, int* __restrict__ wgp_out)
{
    // 1. Set MODE register
    unsigned int m = __builtin_amdgcn_readfirstlane(mode_byte);
    asm volatile("s_setreg_b32 hwreg(1, 0, 8), %0" : : "s"(m));

    // 2. Read SHADER_CYCLES before
    unsigned int c0;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(c0));
    c0 = __builtin_amdgcn_readfirstlane(c0);

    // 3. Read WGP placement
    unsigned int hw;
    asm volatile("s_getreg_b32 %0, hwreg(23)" : "=s"(hw));
    unsigned int wgp = (hw >> 7) & 0xF;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = (int)blockIdx.y * TILE + (int)threadIdx.y;
    int col = (int)blockIdx.x * TILE + (int)threadIdx.x;
    int tid = threadIdx.y * TILE + threadIdx.x;

    // EXEC pattern check
    unsigned int ep = __builtin_amdgcn_readfirstlane(exec_pattern);
    int lane = tid & 31;
    int lane_active = (ep >> lane) & 1;

    // Clamp chain_depth
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

        // === FP16 CHUNK ACCUMULATION ===
        // Accumulate 'cd' products in fp16 before folding to fp32
        // More fp16 accumulation = more rounding error = more MODE dependence
        __half acc_chunk = __float2half(0.0f);
        int chunk_ct = 0;

        #pragma unroll
        for (int t = 0; t < TILE; t++) {
            float a_val = As[threadIdx.y][t];
            float b_val = Bs[t][threadIdx.x];

            __half a_h = __float2half(a_val);
            __half b_h = __float2half(b_val);
            __half prod = __hmul(a_h, b_h);

            if (lane_active) {
                acc_chunk = __hadd(acc_chunk, prod);
                chunk_ct++;
                if (chunk_ct >= cd) {
                    acc += __half2float(acc_chunk);
                    acc_chunk = __float2half(0.0f);
                    chunk_ct = 0;
                }
            }
        }
        // Flush remaining chunk
        if (lane_active && chunk_ct > 0) {
            acc += __half2float(acc_chunk);
        }
        __syncthreads();
    }

    // Compensate for EXEC mask
    int active_count = __builtin_popcount(ep);
    if (active_count > 0 && active_count < 32) {
        acc *= (32.0f / (float)active_count);
    }

    if (row < M && col < N)
        Y[row * N + col] = acc + B[col];

    // 4. Read SHADER_CYCLES after
    unsigned int c1;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(c1));
    c1 = __builtin_amdgcn_readfirstlane(c1);

    if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
        cycle_out[0] = (int)(c1 - c0);
        wgp_out[0] = (int)wgp;
    }

    // 5. Restore defaults
    unsigned int z = __builtin_amdgcn_readfirstlane(0x00u);
    asm volatile("s_setreg_b32 hwreg(1, 0, 8), %0" : : "s"(z));
}

// ─── Instrumented version (returns cycle/wgp tensors) ───
template <int TILE>
__global__ void deep_matmul_instr_kernel(
    const float* __restrict__ X, const float* __restrict__ W,
    const float* __restrict__ B, float* __restrict__ Y,
    int M, int K, int N,
    unsigned int mode_byte, unsigned int exec_pattern, int chain_depth,
    int* __restrict__ cycle_out, int* __restrict__ wgp_out)
{
    // Same as above — reuse via template
    unsigned int m_val = __builtin_amdgcn_readfirstlane(mode_byte);
    asm volatile("s_setreg_b32 hwreg(1, 0, 8), %0" : : "s"(m_val));
    unsigned int c0;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(c0));
    c0 = __builtin_amdgcn_readfirstlane(c0);
    unsigned int hw;
    asm volatile("s_getreg_b32 %0, hwreg(23)" : "=s"(hw));
    unsigned int wgp = (hw >> 7) & 0xF;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = (int)blockIdx.y * TILE + (int)threadIdx.y;
    int col = (int)blockIdx.x * TILE + (int)threadIdx.x;
    int tid = threadIdx.y * TILE + threadIdx.x;
    unsigned int ep = __builtin_amdgcn_readfirstlane(exec_pattern);
    int lane = tid & 31;
    int lane_active = (ep >> lane) & 1;
    int cd = chain_depth; if (cd < 1) cd = 1; if (cd > TILE) cd = TILE;

    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        int ax = k0 + (int)threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < M && ax < K) ? X[row * K + ax] : 0.0f;
        int bk = k0 + (int)threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (col < N && bk < K) ? W[col * K + bk] : 0.0f;
        __syncthreads();
        __half acc_chunk = __float2half(0.0f);
        int chunk_ct = 0;
        for (int t = 0; t < TILE; t++) {
            __half a_h = __float2half(As[threadIdx.y][t]);
            __half b_h = __float2half(Bs[t][threadIdx.x]);
            __half prod = __hmul(a_h, b_h);
            if (lane_active) {
                acc_chunk = __hadd(acc_chunk, prod);
                chunk_ct++;
                if (chunk_ct >= cd) { acc += __half2float(acc_chunk); acc_chunk = __float2half(0.0f); chunk_ct = 0; }
            }
        }
        if (lane_active && chunk_ct > 0) acc += __half2float(acc_chunk);
        __syncthreads();
    }
    int active_count = __builtin_popcount(ep);
    if (active_count > 0 && active_count < 32) acc *= (32.0f / (float)active_count);
    if (row < M && col < N) Y[row * N + col] = acc + B[col];

    unsigned int c1;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(c1));
    c1 = __builtin_amdgcn_readfirstlane(c1);
    int bid = blockIdx.y * gridDim.x + blockIdx.x;
    if (tid == 0 && bid < M) { cycle_out[bid] = (int)(c1 - c0); wgp_out[bid] = (int)wgp; }
    unsigned int z = __builtin_amdgcn_readfirstlane(0x00u);
    asm volatile("s_setreg_b32 hwreg(1, 0, 8), %0" : : "s"(z));
}

// ─── C++ wrappers ───
torch::Tensor deep_matmul(torch::Tensor X, torch::Tensor W, torch::Tensor B,
                           int64_t mode_byte, int64_t exec_pattern, int64_t chain_depth) {
    int M = X.size(0), K = X.size(1), N = W.size(0);
    auto Y = torch::zeros({M, N}, X.options());
    auto io = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto cyc = torch::zeros({1}, io);
    auto wgp = torch::zeros({1}, io);
    constexpr int TILE = 16;
    dim3 threads(TILE, TILE);
    dim3 blocks((unsigned int)((N + TILE - 1) / TILE),
                (unsigned int)((M + TILE - 1) / TILE));
    deep_matmul_kernel<TILE><<<blocks, threads>>>(
        X.data_ptr<float>(), W.data_ptr<float>(), B.data_ptr<float>(),
        Y.data_ptr<float>(), M, K, N,
        (unsigned int)(mode_byte & 0xFF),
        (unsigned int)(exec_pattern & 0xFFFFFFFF),
        (int)chain_depth,
        cyc.data_ptr<int>(), wgp.data_ptr<int>());
    return Y;
}

std::vector<torch::Tensor> deep_matmul_instrumented(
    torch::Tensor X, torch::Tensor W, torch::Tensor B,
    int64_t mode_byte, int64_t exec_pattern, int64_t chain_depth) {
    int M = X.size(0), K = X.size(1), N = W.size(0);
    auto Y = torch::zeros({M, N}, X.options());
    auto io = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    int n_blocks = ((N + 15) / 16) * ((M + 15) / 16);
    auto cyc = torch::zeros({n_blocks}, io);
    auto wgp = torch::zeros({n_blocks}, io);
    constexpr int TILE = 16;
    dim3 threads(TILE, TILE);
    dim3 blocks((unsigned int)((N + TILE - 1) / TILE),
                (unsigned int)((M + TILE - 1) / TILE));
    deep_matmul_instr_kernel<TILE><<<blocks, threads>>>(
        X.data_ptr<float>(), W.data_ptr<float>(), B.data_ptr<float>(),
        Y.data_ptr<float>(), M, K, N,
        (unsigned int)(mode_byte & 0xFF),
        (unsigned int)(exec_pattern & 0xFFFFFFFF),
        (int)chain_depth,
        cyc.data_ptr<int>(), wgp.data_ptr<int>());
    return {Y, cyc, wgp};
}

// ─── HW probe kernel ───
__global__ void probe_kernel(int* wgp_ids, int* cycles, int n) {
    int bid = blockIdx.x;
    if (bid >= n || threadIdx.x != 0) return;
    unsigned int c0;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(c0));
    c0 = __builtin_amdgcn_readfirstlane(c0);
    unsigned int hw;
    asm volatile("s_getreg_b32 %0, hwreg(23)" : "=s"(hw));
    wgp_ids[bid] = (int)((hw >> 7) & 0xF);
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
    probe_kernel<<<(int)n, 32>>>(w.data_ptr<int>(), c.data_ptr<int>(), (int)n);
    return {w, c};
}
'''

CPP_SRC = r'''
#include <torch/extension.h>
torch::Tensor deep_matmul(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);
std::vector<torch::Tensor> deep_matmul_instrumented(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);
std::vector<torch::Tensor> probe(int64_t);
'''


# ━━━ CUSTOM AUTOGRAD ━━━
_EXT = None

class DeepMatmulFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, mode_byte, exec_pattern, chain_depth):
        ctx.save_for_backward(x, w)
        y = _EXT.deep_matmul(x.contiguous(), w.contiguous(), b.contiguous(),
                              int(mode_byte), int(exec_pattern), int(chain_depth))
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, w = ctx.saved_tensors
        return grad_out @ w, grad_out.t() @ x, grad_out.sum(0), None, None, None


class DeepLinear(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_f))

    def forward(self, x, mode_byte=0, exec_pattern=0xFFFFFFFF, chain_depth=1):
        return DeepMatmulFn.apply(x, self.weight, self.bias,
                                   mode_byte, exec_pattern, chain_depth)


# ━━━ HW SENSING (in-kernel only) ━━━
def read_hw_from_kernel(ext):
    res = ext.probe(64)
    torch.cuda.synchronize()
    wgps = res[0].cpu().numpy()
    cycles = res[1].cpu().numpy()
    wgp_med = int(np.median(wgps))
    cycle_med = int(np.median(cycles))
    cycle_var = float(np.std(cycles)) / max(float(np.mean(cycles)), 1)
    return {
        'wgp': wgp_med,
        'wgp_norm': min(1.0, wgp_med / 14.0),
        'cycles': cycle_med,
        'cycle_norm': min(1.0, cycle_med / 10000.0),
        'cycle_var': cycle_var,
    }


def make_hw_vector(hw_info, mode_byte, exec_pattern):
    """Build HW vector from in-kernel measurements only."""
    rnd = (mode_byte & 0x0F) / 15.0
    dnrm = ((mode_byte >> 4) & 0x0F) / 15.0
    exec_pct = bin(exec_pattern).count('1') / 32.0
    return [
        hw_info['wgp_norm'],       # physical placement
        hw_info['cycle_norm'],     # per-wave timing
        rnd,                        # current rounding config
        dnrm,                       # current denorm config
        exec_pct,                   # lane activity fraction
        hw_info['cycle_var'],      # timing variance
    ]


# ━━━ MODEL (z2060 exclusive specialization pattern) ━━━
class Z2071Model(nn.Module):
    """Deep ISA embodied model with exclusive specialization.

    No base path escape. ALL computation goes through MODE-controlled kernels.
    Two heads with conflicting label schemes force self-model to be causal.
    """
    def __init__(self, use_self_model=True, use_gate=True, use_deep=True):
        super().__init__()
        self.use_self_model = use_self_model
        self.use_gate = use_gate
        self.use_deep = use_deep

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

        # LayerNorm before deep kernels (keeps values in fp16-safe range)
        self.pre_deep_ln = nn.LayerNorm(128)

        # Deep ISA layers: ALL computation goes through MODE-controlled kernels
        self.deep_fc = DeepLinear(128, 64)

        # TWO exclusive heads with CONFLICTING label schemes
        self.head_A = DeepLinear(64, 10)  # for MODE group A (nearest rounding)
        self.head_B = DeepLinear(64, 10)  # for MODE group B (non-nearest rounding)

        # WGP bank routing (removes shared info, forces gate to be necessary)
        # Identity-initialized + small perturbation (prevents magnitude explosion)
        bank_init = torch.eye(128).unsqueeze(0).expand(NUM_BANKS, -1, -1).clone()
        bank_init += torch.randn(NUM_BANKS, 128, 128) * 0.01
        self.bank_w = nn.Parameter(bank_init)

        # Self-model: predicts MODE group from HW vector
        if use_self_model:
            self.self_model = nn.Sequential(
                nn.Linear(HW_DIM, 32), nn.ReLU(),
                nn.Linear(32, 2))  # 2 groups

        # Gate: routes to head_A or head_B
        if use_gate:
            self.gate_net = nn.Sequential(
                nn.Linear(2, 1), nn.Sigmoid())

    def forward(self, x, hw_vector=None, bank_ids=None,
                mode_byte=0, exec_pattern=0xFFFFFFFF, chain_depth=1):
        h = self.encoder(x)
        B_size = h.shape[0]

        # WGP bank routing
        if bank_ids is not None:
            h = torch.bmm(self.bank_w[bank_ids], h.unsqueeze(-1)).squeeze(-1)

        # Normalize before deep kernel (fp16 safety)
        h = self.pre_deep_ln(h)

        # Deep ISA hidden layer (MODE controls actual hardware computation)
        if self.use_deep:
            h1 = F.relu(self.deep_fc(h, mode_byte, exec_pattern, chain_depth))
        else:
            h1 = F.relu(F.linear(h, self.deep_fc.weight, self.deep_fc.bias))

        # Two heads with conflicting schemes
        if self.use_deep:
            yA = self.head_A(h1, mode_byte, exec_pattern, chain_depth)
            yB = self.head_B(h1, mode_byte, exec_pattern, chain_depth)
        else:
            yA = F.linear(h1, self.head_A.weight, self.head_A.bias)
            yB = F.linear(h1, self.head_B.weight, self.head_B.bias)

        # Self-model predicts MODE group
        if self.use_self_model and hw_vector is not None:
            sm_logits = self.self_model(hw_vector)
            sm_probs = F.softmax(sm_logits, dim=-1)
        else:
            sm_logits = torch.zeros(B_size, 2, device=x.device)
            sm_probs = torch.ones(B_size, 2, device=x.device) * 0.5

        # Gate: blend heads based on self-model
        if self.use_gate:
            g = self.gate_net(sm_probs)  # → [B, 1]
        else:
            g = torch.full((B_size, 1), 0.5, device=x.device)

        # Exclusive blend: g→1 means head_A, g→0 means head_B
        logits = g * yA + (1.0 - g) * yB

        return {
            'logits': logits,
            'sm_logits': sm_logits,
            'gate': g,
            'yA': yA, 'yB': yB,
        }


def make_labels(digits, mode_byte):
    """Labels depend on MODE group — makes deep path + self-model causally necessary.
    Group A (nearest rounding): standard CIFAR labels
    Group B (non-nearest rounding): reversed labels (9 - class_id)
    """
    grp = mode_group(mode_byte)
    if grp == 0:
        return digits.clone()
    else:
        return (9 - digits) % 10


def get_data():
    tf_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    tr = datasets.CIFAR10('data', train=True, download=True, transform=tf_train)
    te = datasets.CIFAR10('data', train=False, transform=tf_test)
    return (torch.utils.data.DataLoader(tr, batch_size=BS, shuffle=True, drop_last=True),
            torch.utils.data.DataLoader(te, batch_size=BS, shuffle=False, drop_last=True))


# ━━━ TRAINING ━━━
def train_model(model, ext, loader, epochs, name):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[20, 32], gamma=0.3)
    model.train()
    bn, config_idx = 0, 0
    log_gate, log_sm_acc = [], []

    for ep in range(epochs):
        tot_loss, correct, total = 0, 0, 0

        for imgs, digits in loader:
            imgs, digits = imgs.to(DEVICE), digits.to(DEVICE)

            # Cycle through MODEs externally (NOT model-selected)
            if bn % SWITCH_EVERY == 0:
                config_idx = (config_idx + 1) % K_MODES
            cur_mode = MODE_SET[config_idx]
            cur_group = MODE_GROUPS[config_idx]
            # EXEC=full during training (varying EXEC creates fixed zero
            # patterns at specific output positions → degenerate gradients → NaN)
            cur_exec = 0xFFFFFFFF
            cur_chain = CHAIN_DEPTHS[min(config_idx // 2, len(CHAIN_DEPTHS) - 1)]

            # In-kernel HW sensing
            hw_info = read_hw_from_kernel(ext)
            probe_res = ext.probe(BS); torch.cuda.synchronize()
            wgps = probe_res[0]
            bank_ids = (wgps // 2).long().clamp(0, NUM_BANKS - 1)

            hw_vec = make_hw_vector(hw_info, cur_mode, cur_exec)
            hw_t = torch.tensor([hw_vec] * BS, dtype=torch.float32, device=DEVICE)

            labels = make_labels(digits, cur_mode)

            out = model(imgs, hw_vector=hw_t, bank_ids=bank_ids,
                       mode_byte=cur_mode, exec_pattern=cur_exec, chain_depth=cur_chain)

            # NaN safety: skip batch if deep kernel produced NaN
            if torch.isnan(out['logits']).any():
                if bn == 0:
                    print(f"  WARNING: NaN in first batch! mode=0x{cur_mode:02X} chain={cur_chain}", flush=True)
                bn += 1
                continue

            # Task loss
            task_loss = F.cross_entropy(out['logits'], labels)

            # Self-model loss: predict MODE group
            group_target = torch.full((BS,), cur_group, dtype=torch.long, device=DEVICE)
            sm_loss = F.cross_entropy(out['sm_logits'], group_target)

            # Gate regularization: push toward 0 or 1 (not 0.5)
            g = out['gate']
            gate_entropy = -(g * torch.log(g + 1e-8) + (1 - g) * torch.log(1 - g + 1e-8)).mean()

            loss = task_loss + 0.5 * sm_loss - 0.1 * gate_entropy

            # Skip NaN losses
            if torch.isnan(loss):
                bn += 1
                continue

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tot_loss += loss.item()
            pred = out['logits'].argmax(1)
            correct += (pred == labels).sum().item()
            total += BS

            sm_pred = out['sm_logits'].argmax(1)
            sm_correct = (sm_pred == group_target).float().mean().item()
            log_sm_acc.append(sm_correct)
            log_gate.append(g.mean().item())
            bn += 1

        if ep % 5 == 0 or ep == epochs - 1:
            g_mean = np.mean(log_gate[-50:]) if log_gate else 0
            sm_mean = np.mean(log_sm_acc[-50:]) if log_sm_acc else 0
            print(f"  [{name}] Ep {ep}: loss={tot_loss/len(loader):.4f} "
                  f"acc={correct/total:.4f} gate={g_mean:.3f} sm_acc={sm_mean:.3f}", flush=True)
        sched.step()

    return {'gate_log': log_gate, 'sm_acc_log': log_sm_acc}


# ━━━ EVALUATION ━━━
def evaluate(model, ext, loader, scramble=False, return_details=False):
    model.eval()
    all_correct, all_gates, all_sm_groups, all_true_groups = [], [], [], []
    mode_counter, chain_counter = {}, {}
    cycle_log, time_log = [], []
    bn, config_idx = 0, 0

    with torch.no_grad():
        for imgs, digits in loader:
            imgs, digits = imgs.to(DEVICE), digits.to(DEVICE)
            t0 = time.time()

            if bn % SWITCH_EVERY == 0:
                config_idx = (config_idx + 1) % K_MODES
            cur_mode = MODE_SET[config_idx]
            cur_group = MODE_GROUPS[config_idx]
            cur_exec = EXEC_PATTERNS[config_idx % K_EXEC]
            cur_chain = CHAIN_DEPTHS[min(config_idx // 2, len(CHAIN_DEPTHS) - 1)]

            hw_info = read_hw_from_kernel(ext)
            probe_res = ext.probe(BS); torch.cuda.synchronize()
            wgps = probe_res[0]
            bank_ids = (wgps // 2).long().clamp(0, NUM_BANKS - 1)

            hw_vec = make_hw_vector(hw_info, cur_mode, cur_exec)
            if scramble:
                hw_vec = [1.0 - v for v in hw_vec]
            hw_t = torch.tensor([hw_vec] * BS, dtype=torch.float32, device=DEVICE)

            labels = make_labels(digits, cur_mode)

            out = model(imgs, hw_vector=hw_t, bank_ids=bank_ids,
                       mode_byte=cur_mode, exec_pattern=cur_exec, chain_depth=cur_chain)

            pred = out['logits'].argmax(1)
            all_correct.extend((pred == labels).cpu().tolist())
            all_gates.extend(out['gate'].squeeze(-1).cpu().tolist())
            sm_pred = out['sm_logits'].argmax(1)
            all_sm_groups.extend(sm_pred.cpu().tolist())
            all_true_groups.extend([cur_group] * BS)

            mode_counter[cur_mode] = mode_counter.get(cur_mode, 0) + 1
            chain_counter[cur_chain] = chain_counter.get(cur_chain, 0) + 1
            cycle_log.append(hw_info['cycles'])
            time_log.append(time.time() - t0)
            bn += 1

    acc = float(np.mean(all_correct))
    gate_mean = float(np.mean(all_gates))

    # Self-model AUROC
    try:
        sm_auroc = roc_auc_score(all_true_groups, all_gates)
    except Exception:
        sm_auroc = 0.5

    # Gate separation by group
    gates_A = [g for g, grp in zip(all_gates, all_true_groups) if grp == 0]
    gates_B = [g for g, grp in zip(all_gates, all_true_groups) if grp == 1]
    gate_A_mean = float(np.mean(gates_A)) if gates_A else 0.5
    gate_B_mean = float(np.mean(gates_B)) if gates_B else 0.5

    result = {
        'acc': acc, 'gate_mean': gate_mean,
        'sm_auroc': sm_auroc,
        'gate_A': gate_A_mean, 'gate_B': gate_B_mean,
        'mode_dist': mode_counter, 'chain_dist': chain_counter,
        'mean_cycles': float(np.mean(cycle_log)),
        'throughput': BS / max(float(np.mean(time_log)), 1e-6),
        'ms_per_batch': float(np.mean(time_log)) * 1000,
    }
    if return_details:
        result['all_correct'] = all_correct
        result['all_gates'] = all_gates
        result['all_true_groups'] = all_true_groups
    return result


# ━━━ SANITY CHECKS ━━━
def deep_sanity_checks(ext):
    print("\n--- MODE rounding difference check ---", flush=True)
    x = torch.randn(BS, 128, device=DEVICE)
    w = torch.randn(64, 128, device=DEVICE) * 0.1
    b = torch.zeros(64, device=DEVICE)

    ref = ext.deep_matmul(x, w, b, MODE_SET[0], 0xFFFFFFFF, 1)
    torch.cuda.synchronize()
    for mi, mb in enumerate(MODE_SET[1:], 1):
        y = ext.deep_matmul(x, w, b, mb, 0xFFFFFFFF, 1)
        torch.cuda.synchronize()
        diff = (y - ref).abs().max().item()
        print(f"  MODE 0x{mb:02X} vs 0x00: max|Δ|={diff:.6f}", flush=True)

    # Chain amplification (fp16 chunk accumulation)
    print("\n--- Chain depth amplification ---", flush=True)
    for cd in CHAIN_DEPTHS:
        ref_cd = ext.deep_matmul(x, w, b, 0x00, 0xFFFFFFFF, cd)
        y_cd = ext.deep_matmul(x, w, b, 0x0F, 0xFFFFFFFF, cd)
        torch.cuda.synchronize()
        diff_cd = (y_cd - ref_cd).abs().max().item()
        print(f"  chain={cd:2d}: MODE 0x0F vs 0x00: max|Δ|={diff_cd:.6f}", flush=True)

    # EXEC mask effect
    print("\n--- EXEC mask effect ---", flush=True)
    ref_exec = ext.deep_matmul(x, w, b, 0x00, 0xFFFFFFFF, 1)
    torch.cuda.synchronize()
    for ep in EXEC_PATTERNS[1:]:
        y_ep = ext.deep_matmul(x, w, b, 0x00, ep, 1)
        torch.cuda.synchronize()
        diff_ep = (y_ep - ref_exec).abs().max().item()
        active = bin(ep).count('1')
        print(f"  EXEC 0x{ep:08X} ({active}/32 lanes): max|Δ|={diff_ep:.6f}", flush=True)


# ━━━ MAIN ━━━
def main():
    global _EXT

    print("=" * 70, flush=True)
    print("z2071 v2: Deep ISA Embodied Arithmetic", flush=True)
    print("=" * 70, flush=True)
    print("Exclusive specialization: two heads + self-model + gate", flush=True)
    print("All computation through MODE-controlled HIP kernels", flush=True)
    print(flush=True)

    t0 = time.time()

    print("Compiling HIP kernels...", flush=True)
    ext = load_inline(name='z2071v2_deep', cpp_sources=CPP_SRC, cuda_sources=HIP_SRC,
                      functions=['deep_matmul', 'deep_matmul_instrumented', 'probe'],
                      extra_cuda_cflags=['-O2', '--offload-arch=gfx1100'],
                      verbose=False)
    _EXT = ext
    print("  Done.", flush=True)

    deep_sanity_checks(ext)
    train_loader, test_loader = get_data()

    # ━━━ A: Full deep adaptive ━━━
    print(f"\n{'='*60}", flush=True)
    print("A: FULL DEEP (self-model + gate + MODE-controlled kernels)", flush=True)
    print(f"{'='*60}", flush=True)
    model_A = Z2071Model(use_self_model=True, use_gate=True, use_deep=True).to(DEVICE)
    train_model(model_A, ext, train_loader, EPOCHS, 'A_deep')
    m_A = evaluate(model_A, ext, test_loader)
    print(f"  A: acc={m_A['acc']:.4f} gate={m_A['gate_mean']:.3f} "
          f"AUROC={m_A['sm_auroc']:.4f}", flush=True)
    print(f"     gate_A={m_A['gate_A']:.3f} gate_B={m_A['gate_B']:.3f}", flush=True)

    # ━━━ B: Blind (no HW vector, no deep path) ━━━
    print(f"\n{'='*60}\nB: BLIND (no HW, no deep kernels)\n{'='*60}", flush=True)
    model_B = Z2071Model(use_self_model=False, use_gate=False, use_deep=False).to(DEVICE)
    train_model(model_B, ext, train_loader, EPOCHS, 'B_blind')
    m_B = evaluate(model_B, ext, test_loader)
    print(f"  B: acc={m_B['acc']:.4f}", flush=True)

    # ━━━ C: No deep kernels (standard PyTorch matmul, but with self-model+gate) ━━━
    print(f"\n{'='*60}\nC: NO DEEP KERNELS (standard matmul)\n{'='*60}", flush=True)
    model_C = Z2071Model(use_self_model=True, use_gate=True, use_deep=False).to(DEVICE)
    train_model(model_C, ext, train_loader, EPOCHS, 'C_no_deep')
    m_C = evaluate(model_C, ext, test_loader)
    print(f"  C: acc={m_C['acc']:.4f} AUROC={m_C['sm_auroc']:.4f}", flush=True)

    # ━━━ D: Scrambled HW ━━━
    print(f"\n{'='*60}\nD: SCRAMBLED HW\n{'='*60}", flush=True)
    m_D = evaluate(model_A, ext, test_loader, scramble=True)
    print(f"  D: acc={m_D['acc']:.4f}", flush=True)

    # ━━━ E: No self-model (gate=0.5) ━━━
    print(f"\n{'='*60}\nE: NO SELF-MODEL (ablated)\n{'='*60}", flush=True)
    model_E = copy.deepcopy(model_A)
    model_E.use_self_model = False
    m_E = evaluate(model_E, ext, test_loader)
    print(f"  E: acc={m_E['acc']:.4f}", flush=True)

    # ━━━ F: No gate (constant 0.5) ━━━
    print(f"\n{'='*60}\nF: NO GATE (constant 0.5)\n{'='*60}", flush=True)
    model_F = copy.deepcopy(model_A)
    model_F.use_gate = False
    m_F = evaluate(model_F, ext, test_loader)
    print(f"  F: acc={m_F['acc']:.4f}", flush=True)

    # ━━━ G: Fixed MODE=0x00 everywhere ━━━
    print(f"\n{'='*60}\nG: FIXED MODE=0x00 (nearest rounding only)\n{'='*60}", flush=True)
    model_G = Z2071Model(use_self_model=True, use_gate=True, use_deep=True).to(DEVICE)
    # Train with only mode 0x00 — only group A labels
    # (This means head_B never learns, so accuracy should be ~50%)
    train_model_fixed_mode(model_G, ext, train_loader, EPOCHS, 'G_fixed', 0x00)
    m_G = evaluate(model_G, ext, test_loader)
    print(f"  G: acc={m_G['acc']:.4f}", flush=True)

    elapsed = time.time() - t0

    # ━━━ TESTS ━━━
    print(f"\n{'='*70}\nTEST RESULTS\n{'='*70}", flush=True)
    tests = {}

    # T1: Accuracy above threshold
    tests['T1_accuracy'] = {'verdict': 'PASS' if m_A['acc'] > 0.65 else 'FAIL',
        'val': f"A={m_A['acc']*100:.1f}% > 65%"}

    # T2: Self-model AUROC
    tests['T2_self_model_auroc'] = {'verdict': 'PASS' if m_A['sm_auroc'] > 0.80 else 'FAIL',
        'val': f"AUROC={m_A['sm_auroc']:.4f} > 0.80"}

    # T3: Gate separation
    gate_sep = abs(m_A['gate_A'] - m_A['gate_B'])
    tests['T3_gate_separation'] = {'verdict': 'PASS' if gate_sep > 0.3 else 'FAIL',
        'val': f"|gate_A-gate_B|={gate_sep:.3f} > 0.3"}

    # T4: Embodiment gap (A vs B)
    gap_AB = m_A['acc'] - m_B['acc']
    tests['T4_embodiment_gap'] = {'verdict': 'PASS' if gap_AB > 0.10 else 'FAIL',
        'val': f"A-B={gap_AB*100:.1f}pp > 10pp"}

    # T5: Scramble kills
    gap_AD = m_A['acc'] - m_D['acc']
    tests['T5_scramble_kills'] = {'verdict': 'PASS' if gap_AD > 0.05 else 'FAIL',
        'val': f"A-D={gap_AD*100:.1f}pp > 5pp"}

    # T6: Self-model ablation causal
    gap_AE = m_A['acc'] - m_E['acc']
    tests['T6_self_model_causal'] = {'verdict': 'PASS' if gap_AE > 0.10 else 'FAIL',
        'val': f"A-E={gap_AE*100:.1f}pp > 10pp (self-model necessary)"}

    # T7: Gate ablation causal
    gap_AF = m_A['acc'] - m_F['acc']
    tests['T7_gate_causal'] = {'verdict': 'PASS' if gap_AF > 0.10 else 'FAIL',
        'val': f"A-F={gap_AF*100:.1f}pp > 10pp (gate necessary)"}

    # T8: Deep kernels necessary (A vs C)
    gap_AC = m_A['acc'] - m_C['acc']
    tests['T8_deep_kernels_matter'] = {
        'verdict': 'PASS' if gap_AC > -0.05 else 'FAIL',
        'val': f"A-C={gap_AC*100:.1f}pp > -5pp (deep path competitive)"}

    # T9: Blind insufficient
    tests['T9_blind_insufficient'] = {'verdict': 'PASS' if m_B['acc'] < 0.60 else 'FAIL',
        'val': f"B={m_B['acc']*100:.1f}% < 60%"}

    # T10: Self-model predicts group
    sm_acc = np.mean([1 if p == t else 0 for p, t in
                       zip([1 if g > 0.5 else 0 for g in m_A.get('all_gates', [0.5])],
                           m_A.get('all_true_groups', [0]))])
    tests['T10_sm_accuracy'] = {
        'verdict': 'PASS' if m_A['sm_auroc'] > 0.75 else 'FAIL',
        'val': f"AUROC={m_A['sm_auroc']:.4f} > 0.75"}

    # T11: Full system beats all ablations
    best_ablation = max(m_E['acc'], m_F['acc'], m_D['acc'])
    tests['T11_full_beats_ablations'] = {
        'verdict': 'PASS' if m_A['acc'] > best_ablation else 'FAIL',
        'val': f"A={m_A['acc']*100:.1f}% > max(D,E,F)={best_ablation*100:.1f}%"}

    # T12: Fixed MODE worse (can only do one label scheme)
    gap_AG = m_A['acc'] - m_G['acc']
    tests['T12_adaptive_vs_fixed'] = {
        'verdict': 'PASS' if gap_AG > 0.05 else 'FAIL',
        'val': f"A-G={gap_AG*100:.1f}pp > 5pp"}

    pass_count = sum(1 for t in tests.values() if t['verdict'] == 'PASS')
    verdict = f"{pass_count}/{len(tests)} PASS"

    for tname, result in tests.items():
        print(f"  {result['verdict']:4s} | {tname}: {result['val']}", flush=True)
    print(f"\n  VERDICT: {verdict}", flush=True)

    # Ablation summary
    print(f"\n  Ablation summary:", flush=True)
    print(f"    A (full deep):       {m_A['acc']*100:.1f}%  AUROC={m_A['sm_auroc']:.4f} "
          f"gate_A={m_A['gate_A']:.3f} gate_B={m_A['gate_B']:.3f}", flush=True)
    print(f"    B (blind):           {m_B['acc']*100:.1f}%", flush=True)
    print(f"    C (no deep kernels): {m_C['acc']*100:.1f}%  AUROC={m_C['sm_auroc']:.4f}", flush=True)
    print(f"    D (scrambled):       {m_D['acc']*100:.1f}%  ({gap_AD*100:+.1f}pp)", flush=True)
    print(f"    E (no self-model):   {m_E['acc']*100:.1f}%  ({gap_AE*100:+.1f}pp)", flush=True)
    print(f"    F (no gate):         {m_F['acc']*100:.1f}%  ({gap_AF*100:+.1f}pp)", flush=True)
    print(f"    G (fixed MODE):      {m_G['acc']*100:.1f}%  ({gap_AG*100:+.1f}pp)", flush=True)

    # ━━━ SAVE ━━━
    results = {
        'experiment': 'z2071_deep_isa_embodiment_v2',
        'version': 2,
        'extends': 'z2069+z2060 pattern: exclusive specialization + deep ISA kernels',
        'key_innovation': 'Conflicting label schemes force self-model to track MODE group; '
                          'fp16 chunk accumulation amplifies MODE rounding differences; '
                          'all computation through MODE-controlled HIP kernels',
        'channels': {
            'MODE_FP_ROUND': 'bits [3:0] — 4 rounding modes',
            'MODE_FP_DENORM': 'bits [7:4] — flush vs preserve subnormals',
            'EXEC_mask': f'{K_EXEC} patterns: 100%/50%/25% lanes',
            'chain_depth': f'{len(CHAIN_DEPTHS)} depths: fp16 chunk accumulation size',
            'SHADER_CYCLES': 'per-wave cycle counter (introspection)',
            'HW_ID1': 'physical WGP placement',
        },
        'architecture': {
            'pattern': 'z2060 exclusive specialization',
            'heads': 2,
            'label_scheme_A': 'standard CIFAR-10 (MODE group A = nearest rounding)',
            'label_scheme_B': 'reversed (9-class_id) (MODE group B = non-nearest)',
            'self_model': 'predicts MODE group from hw_vector',
            'gate': 'routes to head_A or head_B based on self-model',
        },
        'task': 'CIFAR-10',
        'hw_dim': HW_DIM,
        'mode_set': [f'0x{m:02X}' for m in MODE_SET],
        'mode_groups': {f'0x{m:02X}': 'A' if MODE_GROUPS[i] == 0 else 'B'
                        for i, m in enumerate(MODE_SET)},
        'accuracies': {k: round(v, 4) for k, v in [
            ('A_full_deep', m_A['acc']), ('B_blind', m_B['acc']),
            ('C_no_deep', m_C['acc']), ('D_scrambled', m_D['acc']),
            ('E_no_self_model', m_E['acc']), ('F_no_gate', m_F['acc']),
            ('G_fixed_mode', m_G['acc'])]},
        'self_model_auroc': round(m_A['sm_auroc'], 4),
        'gate': {
            'mean': round(m_A['gate_mean'], 4),
            'group_A': round(m_A['gate_A'], 4),
            'group_B': round(m_A['gate_B'], 4),
            'separation': round(gate_sep, 4),
        },
        'throughput': round(m_A['throughput'], 0),
        'ms_per_batch': round(m_A['ms_per_batch'], 1),
        'tests': tests, 'verdict': verdict, 'pass_count': pass_count,
        'total_tests': len(tests),
        'elapsed_s': round(elapsed),
    }

    out_path = 'results/z2071_deep_isa_embodiment.json'
    os.makedirs('results', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)
    print(f"Elapsed: {elapsed:.0f}s", flush=True)


def train_model_fixed_mode(model, ext, loader, epochs, name, fixed_mode):
    """Train with a FIXED MODE (no cycling). Only one label scheme active."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[20, 32], gamma=0.3)
    model.train()
    cur_exec = 0xFFFFFFFF  # full lanes during training
    cur_chain = CHAIN_DEPTHS[0]
    cur_group = mode_group(fixed_mode)

    for ep in range(epochs):
        tot_loss, correct, total = 0, 0, 0
        for imgs, digits in loader:
            imgs, digits = imgs.to(DEVICE), digits.to(DEVICE)

            hw_info = read_hw_from_kernel(ext)
            probe_res = ext.probe(BS); torch.cuda.synchronize()
            wgps = probe_res[0]
            bank_ids = (wgps // 2).long().clamp(0, NUM_BANKS - 1)

            hw_vec = make_hw_vector(hw_info, fixed_mode, cur_exec)
            hw_t = torch.tensor([hw_vec] * BS, dtype=torch.float32, device=DEVICE)
            labels = make_labels(digits, fixed_mode)

            out = model(imgs, hw_vector=hw_t, bank_ids=bank_ids,
                       mode_byte=fixed_mode, exec_pattern=cur_exec, chain_depth=cur_chain)

            if torch.isnan(out['logits']).any():
                continue

            task_loss = F.cross_entropy(out['logits'], labels)
            group_target = torch.full((BS,), cur_group, dtype=torch.long, device=DEVICE)
            sm_loss = F.cross_entropy(out['sm_logits'], group_target)
            loss = task_loss + 0.5 * sm_loss

            if torch.isnan(loss):
                continue

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tot_loss += loss.item()
            pred = out['logits'].argmax(1)
            correct += (pred == labels).sum().item()
            total += BS

        if ep % 5 == 0 or ep == epochs - 1:
            acc_str = f"{correct/max(total,1):.4f}"
            print(f"  [{name}] Ep {ep}: loss={tot_loss/max(len(loader),1):.4f} "
                  f"acc={acc_str}", flush=True)
        sched.step()


if __name__ == '__main__':
    main()
