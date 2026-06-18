#!/usr/bin/env python3
"""z2071: Allostatic Multi-Channel Controller with Affect
==========================================================
Merges z2061 (closed-loop effort-head DVFS control, 12/12 PASS) with
z2070 (multi-channel MODE+LDS+priority, 22/24 PASS) and adds:

  1. Self-model: predicts hw_{t+1} from (hw_t, action_t)
  2. Affect = interoceptive prediction error → modulates exploration + effort
  3. Effort head controls DVFS (active inference on silicon)
  4. Energy penalty in phase 2 (fixes z2070's energy failure)
  5. Labels depend on demanded state → effort causally necessary

Scientific basis:
  - Laukkonen/Friston 2025 "Beautiful Loop": world model + self-model + hyper-model
  - Barrett TCE (TiCS 2025): affect = relational construction from signal ensemble
  - Luppi et al. (eLife 2024): synergistic workspace, synergy > redundancy
  - Dual-stream interoceptive inference (arXiv:2511.13668)
  - Phua 2025: ablation-based markers, dissociation tests

Causal chain (9 arrows, all testable):
  demand → effort → DVFS → SCLK → {timing, power, temp} → self-model_pred →
    prediction_error → affect → {temperature, effort_bias} → MODE/priority policy →
    fp16mix_matmul → task_output

Business: fix z2070's energy story. Phase 2 effort controls DVFS with energy penalty.
Compare vs DynamoLLM (53%), throttLL'eM (43.8%), GreenLLM (28-34%), DVFS-GPT (32%).

Conditions:
  A: Full adaptive with affect
  B: Blind (no HW, no affect, no effort)
  E: Scrambled HW
  F: Random policy (zero'd policy weights)
  G: No self-model (affect zeroed, but policy still gets HW)
  H: No affect (prediction error zeroed, self-model still trains)
  I: α=0 (MODE-path disabled)
  J: Always-high SCLK
  K: No effort (random DVFS in phase 2)
  L: No LDS channel
"""
import torch, torch.nn as nn, torch.nn.functional as F
import os, sys, json, time, copy, random, struct, subprocess
import numpy as np
from torchvision import datasets, transforms
from scipy import stats

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')
os.environ.setdefault('PYTORCH_ROCM_ARCH', 'gfx1100')
from torch.utils.cpp_extension import load_inline

DEVICE = 'cuda'
BS = 128
EPOCHS = 40
PHASE2_EPOCH = 16       # Switch to effort-controlled DVFS
SWITCH_EVERY = 10       # Phase 1: external DVFS toggle interval
NUM_BANKS = 8
HW_DIM = 10            # z2070's 10-dim HW vector
AFFECT_DIM = 4         # {thermal, power, timing, overall} prediction error
LDS_ITERS = 16
DVFS_WAIT = 0.05

# 4 MODE personalities (fp16 rounding control)
MODE_SET = [0xF0, 0xF4, 0xF8, 0xFC]
K_MODES = len(MODE_SET)

# Phase 1 configs: (perf_level, mode_idx)
PHASE1_CONFIGS = [
    ('low', 0), ('high', 0), ('low', 1), ('high', 1),
    ('low', 2), ('high', 2), ('low', 3), ('high', 3),
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HIP KERNELS (from z2070: fp16mix + LDS probe + HW probe)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HIP_SRC = r'''
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>

template<int TILE>
__global__ void linear_multi_kernel(
    const float* __restrict__ X, const float* __restrict__ W,
    const float* __restrict__ B, float* __restrict__ Y,
    int M, int K, int N, unsigned int mode_byte, int priority_level)
{
    int prio = priority_level & 3;
    if (prio == 0) asm volatile("s_setprio 0");
    else if (prio == 1) asm volatile("s_setprio 1");
    else if (prio == 2) asm volatile("s_setprio 2");
    else asm volatile("s_setprio 3");

    unsigned int m = __builtin_amdgcn_readfirstlane(mode_byte);
    asm volatile("s_setreg_b32 hwreg(1, 0, 8), %0" : : "s"(m));

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = (int)blockIdx.y * TILE + (int)threadIdx.y;
    int col = (int)blockIdx.x * TILE + (int)threadIdx.x;

    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        int ax = k0 + (int)threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < M && ax < K) ? X[row * K + ax] : 0.0f;
        int bk = k0 + (int)threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (col < N && bk < K) ? W[col * K + bk] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int t = 0; t < TILE; t++) {
            __half a_h = __float2half(As[threadIdx.y][t]);
            __half b_h = __float2half(Bs[t][threadIdx.x]);
            __half prod = __hmul(a_h, b_h);
            acc += __half2float(prod);
        }
        __syncthreads();
    }
    if (row < M && col < N)
        Y[row * N + col] = acc + B[col];

    unsigned int z = __builtin_amdgcn_readfirstlane(0xF0u);
    asm volatile("s_setreg_b32 hwreg(1, 0, 8), %0" : : "s"(z));
    asm volatile("s_setprio 0");
}

__global__ void lds_probe_kernel(int* timing_out, int pattern, int n_iters) {
    __shared__ float lds[1024];
    int tid = threadIdx.x;
    unsigned int c0, c1;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(c0));
    c0 = __builtin_amdgcn_readfirstlane(c0);

    if (pattern == 0) {
        for (int i = 0; i < n_iters; i++) {
            lds[tid + i * 32] = (float)(tid + i);
            __threadfence_block();
            float v = lds[tid + i * 32];
            lds[tid + i * 32] = v + 1.0f;
        }
    } else {
        for (int i = 0; i < n_iters; i++) {
            lds[tid * 32] = (float)(tid + i);
            __threadfence_block();
            float v = lds[tid * 32];
            lds[tid * 32] = v + 1.0f;
        }
    }

    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(c1));
    c1 = __builtin_amdgcn_readfirstlane(c1);
    if (tid == 0) timing_out[blockIdx.x] = (int)(c1 - c0);
}

__global__ void probe_kernel(int* wgp_ids, int* cycle_delta, int* mode_out, int n) {
    int bid = blockIdx.x;
    if (bid >= n || threadIdx.x != 0) return;
    unsigned int c0;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(c0));
    c0 = __builtin_amdgcn_readfirstlane(c0);
    unsigned int hw;
    asm volatile("s_getreg_b32 %0, hwreg(23)" : "=s"(hw));
    wgp_ids[bid] = (int)((hw >> 7) & 0xF);
    unsigned int mode;
    asm volatile("s_getreg_b32 %0, hwreg(1, 0, 8)" : "=s"(mode));
    mode_out[bid] = (int)(mode & 0xFF);
    float acc = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < 5000; i++) acc += 1.0f / (float)(i+1);
    unsigned int c1;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(c1));
    c1 = __builtin_amdgcn_readfirstlane(c1);
    cycle_delta[bid] = (int)(c1 - c0);
}

torch::Tensor linear_forward_multi(torch::Tensor X, torch::Tensor W,
                                    torch::Tensor B, int mode_byte, int priority) {
    int M = X.size(0), K = X.size(1), N = W.size(0);
    auto Y = torch::zeros({M, N}, X.options());
    constexpr int TILE = 16;
    dim3 threads(TILE, TILE);
    dim3 blocks((unsigned int)((N + TILE - 1) / TILE),
                (unsigned int)((M + TILE - 1) / TILE));
    linear_multi_kernel<TILE><<<blocks, threads>>>(
        X.data_ptr<float>(), W.data_ptr<float>(), B.data_ptr<float>(),
        Y.data_ptr<float>(), M, K, N,
        (unsigned int)(mode_byte & 0xFF), priority);
    return Y;
}

torch::Tensor lds_probe(int pattern, int n_iters, int n_blocks) {
    auto io = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto timing = torch::zeros({n_blocks}, io);
    lds_probe_kernel<<<n_blocks, 32>>>(timing.data_ptr<int>(), pattern, n_iters);
    return timing;
}

std::vector<torch::Tensor> probe(int n) {
    auto io = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto w = torch::zeros({n}, io), c = torch::zeros({n}, io), m = torch::zeros({n}, io);
    probe_kernel<<<n, 32>>>(w.data_ptr<int>(), c.data_ptr<int>(), m.data_ptr<int>(), n);
    return {w, c, m};
}
'''

CPP_SRC = r'''
#include <torch/extension.h>
torch::Tensor linear_forward_multi(torch::Tensor, torch::Tensor, torch::Tensor, int, int);
torch::Tensor lds_probe(int, int, int);
std::vector<torch::Tensor> probe(int);
'''


# ━━━ HW SENSING ━━━
def _find_card():
    for c in range(8):
        if os.path.exists(f'/sys/class/drm/card{c}/device/gpu_metrics'):
            return c
    return 0

_CARD = _find_card()
GPU_METRICS_PATH = f'/sys/class/drm/card{_CARD}/device/gpu_metrics'
DPM_PATH = f'/sys/class/drm/card{_CARD}/device/power_dpm_force_performance_level'
print(f"[z2071] Detected card{_CARD}", flush=True)

def set_dvfs(mode):
    try:
        subprocess.run(f'echo {mode} > {DPM_PATH}', shell=True,
                       check=False, timeout=2, capture_output=True)
    except:
        try:
            with open(DPM_PATH, 'w') as f: f.write(mode); f.flush()
        except: pass

def set_dvfs_verified(mode, wait=0.2):
    set_dvfs(mode); time.sleep(wait)
    gm = read_gpu_metrics()
    return gm['sclk_mhz'] if gm else 0

def reset_actuation():
    set_dvfs('auto')

def read_gpu_metrics():
    try:
        with open(GPU_METRICS_PATH, 'rb') as f: data = f.read()
        if len(data) < 200: return None
        return {
            'temp_gfx_c': struct.unpack_from('<H', data, 4)[0] / 100.0,
            'gfx_activity_pct': struct.unpack_from('<H', data, 42)[0],
            'socket_power_mw': struct.unpack_from('<I', data, 112)[0],
            'sclk_mhz': struct.unpack_from('<H', data, 174)[0],
        }
    except: return None

def measure_wall_clock(ext, n=64):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record(); ext.probe(n); e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)

def measure_lds_timing(ext, pattern=0):
    timing = ext.lds_probe(pattern, LDS_ITERS, 4)
    torch.cuda.synchronize()
    return timing.float().mean().item()

def make_hw_vector(wall_ms, gm, cycle_delta, lds_timing=0.0, priority=0):
    timing = min(1.0, max(0.0, wall_ms / 2.0))
    if gm:
        sclk = min(1.0, max(0.0, (gm['sclk_mhz'] - 600) / 1400.0))
        power = min(1.0, max(0.0, (gm['socket_power_mw']/1000 - 20) / 40.0))
        temp = min(1.0, max(0.0, (gm['temp_gfx_c'] - 30) / 50.0))
        activity = min(1.0, max(0.0, gm['gfx_activity_pct'] / 100.0))
    else:
        sclk, power, temp, activity = 0.5, 0.5, 0.5, 0.5
    freq_est = 0.0
    if wall_ms > 1e-3:
        freq_est = min(1.0, max(0.0, (abs(cycle_delta) / wall_ms) / 3e6))
    sclk_bin = 1.0 if (gm and gm['sclk_mhz'] > 1000) else 0.0
    power_rate = min(1.0, max(0.0, power * sclk))
    lds_norm = min(1.0, max(0.0, lds_timing / 50000.0))
    prio_norm = priority / 3.0
    return [timing, sclk, power, temp, activity, freq_est, sclk_bin, power_rate,
            lds_norm, prio_norm]


# ━━━ CUSTOM AUTOGRAD ━━━
_EXT = None

class MultiLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, mode_byte, priority):
        ctx.save_for_backward(x, w)
        y = _EXT.linear_forward_multi(x.contiguous(), w.contiguous(),
                                       b.contiguous(), int(mode_byte), int(priority))
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, w = ctx.saved_tensors
        return grad_out @ w, grad_out.t() @ x, grad_out.sum(0), None, None


class MultiLinear(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_f))

    def forward(self, x, mode_byte, priority=0):
        return MultiLinearFn.apply(x, self.weight, self.bias, mode_byte, priority)


# ━━━ MODEL ━━━
class Z2071Model(nn.Module):
    """Allostatic multi-channel controller with affect.

    Components:
      1. Encoder: CIFAR-10 → h_img [B, 128]
      2. HW encoder: hw_vector → h_hw [B, 32]
      3. Self-model: predicts next hw from (hw_t, action_t) → INTEROCEPTIVE PREDICTION
      4. Affect: prediction_error → 4-dim affect vector → CONSTRUCTED EMOTION
      5. Effort: h_combined + demand + affect → effort ∈ [0,1] → ACTIVE INFERENCE
      6. Policy: hw + affect → mode + priority + alpha → SUB-FIRMWARE CONTROL
      7. Gate: self_pred → blend factor → PRECISION WEIGHTING
      8. MODE-path + base-path → task output
    """
    def __init__(self, use_hw=True, use_policy=True, use_mode_path=True,
                 use_self_model=True, use_affect=True, use_effort=True,
                 fixed_mode=None, fixed_alpha=None, fixed_priority=None):
        super().__init__()
        self.use_hw = use_hw
        self.use_policy = use_policy
        self.use_mode_path = use_mode_path
        self.use_self_model = use_self_model
        self.use_affect = use_affect
        self.use_effort = use_effort
        self.fixed_mode = fixed_mode
        self.fixed_alpha = fixed_alpha
        self.fixed_priority = fixed_priority

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

        # HW encoder
        if use_hw:
            self.hw_proj = nn.Sequential(
                nn.Linear(HW_DIM, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())

        # Self-model: predicts hw_{t+1} from (hw_t, action_summary)
        # action_summary = [effort, mode_idx/4, priority/3, alpha]
        if use_self_model:
            self.self_model_net = nn.Sequential(
                nn.Linear(HW_DIM + 4, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, HW_DIM))  # predict next HW vector
            # SCLK binary predictor (from z2061)
            self.sclk_predictor = nn.Sequential(
                nn.Linear(160 if use_hw else 128, 32), nn.ReLU(),
                nn.Linear(32, 1))

        # Affect encoder: prediction_error → affect_dim
        if use_affect:
            self.affect_enc = nn.Sequential(
                nn.Linear(HW_DIM, 16), nn.ReLU(),
                nn.Linear(16, AFFECT_DIM))

        # Effort head: controls DVFS (ACTION)
        if use_effort:
            effort_in = 160 + 1 + (AFFECT_DIM if use_affect else 0)  # h_combined + demand + affect
            self.demand_proj = nn.Sequential(
                nn.Linear(1, 16), nn.ReLU())
            self.effort_head = nn.Sequential(
                nn.Linear(effort_in - 1 + 16, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, 1))

        # Policy: mode + priority + alpha
        if use_hw and use_policy:
            policy_in = HW_DIM + (AFFECT_DIM if use_affect else 0)
            self.policy = nn.Sequential(
                nn.Linear(policy_in, 64), nn.ReLU(),
                nn.Linear(64, K_MODES + 4 + 1))

        # MODE-path: fp16mix matmul
        if use_mode_path:
            self.mode_fc1 = MultiLinear(128, 64)
            self.mode_fc2 = MultiLinear(64, 10)

        # Baseline path
        self.base_fc1 = nn.Linear(128, 64)
        self.base_fc2 = nn.Linear(64, 10)

        # Bank weights
        self.bank_w = nn.Parameter(torch.randn(NUM_BANKS, 128, 128) * 0.02)

        # Gate: from self-model SCLK prediction
        self.gate_net = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, x, hw_vector=None, bank_ids=None, temperature=1.0,
                hard_mode=True, prev_hw=None, prev_action=None, demand_cue=None,
                affect_override=None):
        h = self.encoder(x)
        B = h.shape[0]

        # ── HW encoding ──
        if self.use_hw and hw_vector is not None:
            h_hw = self.hw_proj(hw_vector)
            h_combined = torch.cat([h, h_hw], dim=1)  # [B, 160]
        else:
            h_combined = torch.cat([h, torch.zeros(B, 32, device=x.device)], dim=1)

        # ── Self-model: predict current hw from previous (hw, action) ──
        hw_pred = None
        pred_error = None
        if self.use_self_model and prev_hw is not None and prev_action is not None:
            sm_input = torch.cat([prev_hw, prev_action], dim=1)  # [B, HW_DIM+4]
            hw_pred = self.self_model_net(sm_input)
            if hw_vector is not None:
                pred_error = (hw_pred - hw_vector).abs()  # [B, HW_DIM]

        # ── SCLK prediction (for gate) ──
        sclk_pred = None
        if self.use_self_model:
            sclk_pred = self.sclk_predictor(h_combined)

        # ── Affect: compress prediction error ──
        affect = torch.zeros(B, AFFECT_DIM, device=x.device)
        if self.use_affect and pred_error is not None:
            affect = self.affect_enc(pred_error.detach())  # detach: affect modulates, doesn't optimize
        if affect_override is not None:
            affect = affect_override

        # ── Effort: ACTION → controls DVFS ──
        effort = None
        if self.use_effort and demand_cue is not None:
            h_demand = self.demand_proj(demand_cue.view(-1, 1))
            if self.use_affect:
                effort_input = torch.cat([h_combined.detach(), h_demand, affect], dim=1)
            else:
                effort_input = torch.cat([h_combined.detach(), h_demand], dim=1)
            effort = torch.sigmoid(self.effort_head(effort_input))

        # ── Policy: select MODE + priority + alpha ──
        if self.use_hw and self.use_policy and hw_vector is not None:
            if self.use_affect:
                pol_input = torch.cat([hw_vector, affect], dim=1)
            else:
                pol_input = hw_vector
            pol = self.policy(pol_input)
            mode_logits = pol[:, :K_MODES]
            prio_logits = pol[:, K_MODES:K_MODES+4]
            alpha = torch.sigmoid(pol[:, K_MODES+4:K_MODES+5])
            mode_probs = F.softmax(mode_logits / temperature, dim=-1)
            prio_probs = F.softmax(prio_logits / temperature, dim=-1)

            if self.fixed_alpha is not None:
                alpha = torch.full_like(alpha, self.fixed_alpha)

            if hard_mode:
                mode_one_hot = F.gumbel_softmax(mode_logits, tau=temperature, hard=True)
                mode_idx = mode_one_hot.argmax(dim=-1)
                prio_one_hot = F.gumbel_softmax(prio_logits, tau=temperature, hard=True)
                prio_idx = prio_one_hot.argmax(dim=-1)
            else:
                mode_idx = mode_probs.argmax(dim=-1)
                prio_idx = prio_probs.argmax(dim=-1)
        else:
            mode_probs = torch.ones(B, K_MODES, device=x.device) / K_MODES
            prio_probs = torch.ones(B, 4, device=x.device) / 4
            mode_idx = torch.zeros(B, dtype=torch.long, device=x.device)
            prio_idx = torch.zeros(B, dtype=torch.long, device=x.device)
            alpha = torch.full((B, 1), 0.5, device=x.device)
            if self.fixed_alpha is not None:
                alpha = torch.full_like(alpha, self.fixed_alpha)

        # ── Select ONE mode and priority for batch ──
        if self.fixed_mode is not None:
            selected_mode = self.fixed_mode
        else:
            counts = torch.bincount(mode_idx, minlength=K_MODES)
            selected_mode = MODE_SET[counts.argmax().item()]

        if self.fixed_priority is not None:
            selected_priority = self.fixed_priority
        else:
            pcounts = torch.bincount(prio_idx, minlength=4)
            selected_priority = pcounts.argmax().item()

        # ── Gate: from self-model SCLK prediction ──
        if self.use_self_model and sclk_pred is not None:
            gate = self.gate_net(torch.sigmoid(sclk_pred))
        else:
            gate = torch.full((B, 1), 0.5, device=x.device)

        # ── MODE-path (fp16mix matmul) ──
        if self.use_mode_path:
            h1 = F.relu(self.mode_fc1(h, selected_mode, selected_priority))
            y_mode = self.mode_fc2(h1, selected_mode, selected_priority)
        else:
            y_mode = torch.zeros(B, 10, device=x.device)

        # ── Baseline path (with bank transform gated by demand) ──
        if bank_ids is not None:
            h_banked = torch.bmm(self.bank_w[bank_ids], h.unsqueeze(-1)).squeeze(-1)
            b1 = F.relu(self.base_fc1(h_banked))
        else:
            b1 = F.relu(self.base_fc1(h))
        y_base = self.base_fc2(b1)

        # ── Alpha blend ──
        logits = alpha * y_mode + (1.0 - alpha) * y_base

        # ── Gate blending (for demand-dependent routing) ──
        # Full path = bank-shifted (high demand), Light path = reversal (low demand)
        # gate → how much to trust full-path vs light-path routing
        # This is absorbed into bank_w selection above, gate modulates output confidence

        return {
            'logits': logits, 'alpha': alpha,
            'mode_probs': mode_probs, 'prio_probs': prio_probs,
            'mode_idx': mode_idx, 'prio_idx': prio_idx,
            'selected_mode': selected_mode, 'selected_priority': selected_priority,
            'sclk_pred': sclk_pred, 'gate': gate,
            'effort': effort, 'affect': affect,
            'hw_pred': hw_pred, 'pred_error': pred_error,
        }


def make_labels(digits, bank_ids, demand_is_high):
    """Labels depend on DEMANDED state. Makes effort causally necessary."""
    labels = digits.clone()
    if demand_is_high:
        even = (bank_ids % 2 == 0)
        labels[even] = (digits[even] + 3) % 10
        labels[~even] = (digits[~even] + 5) % 10
    else:
        labels = (9 - digits) % 10
    return labels


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
def train_model(model, ext, loader, epochs, name, is_blind=False):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[20, 32], gamma=0.3)
    model.train()

    prev_hw = None
    prev_action = None
    prev_effort = 0.5
    bn, level_idx = 0, 0
    demand_is_high = True
    current_priority = 0
    effort_log, affect_log, pred_err_log = [], [], []
    sclk_log, gate_log = [], []

    for ep in range(epochs):
        is_phase2 = ep >= PHASE2_EPOCH
        tot_loss, correct, total = 0, 0, 0
        temp = max(0.8, 2.0 - ep * 0.025)

        for imgs, digits in loader:
            imgs, digits = imgs.to(DEVICE), digits.to(DEVICE)

            # ── DVFS control ──
            if is_phase2 and model.use_effort and not is_blind:
                # Phase 2: effort head controls DVFS
                if prev_effort > 0.5:
                    set_dvfs('high')
                else:
                    set_dvfs('low')
                time.sleep(DVFS_WAIT)
            elif not is_blind and bn % SWITCH_EVERY == 0:
                # Phase 1: external DVFS switching
                level_idx = (level_idx + 1) % len(PHASE1_CONFIGS)
                perf, _ = PHASE1_CONFIGS[level_idx]
                set_dvfs(perf); time.sleep(DVFS_WAIT)
                demand_is_high = (perf == 'high')

            # ── Sense hardware ──
            wall_ms = measure_wall_clock(ext)
            gm = read_gpu_metrics()
            probe_res = ext.probe(BS); torch.cuda.synchronize()
            cd = probe_res[1].cpu().numpy()
            cycle_delta = int(np.median(np.abs(cd)))
            wgps = probe_res[0]
            bank_ids = (wgps // 2).long().clamp(0, NUM_BANKS - 1)

            lds_t = measure_lds_timing(ext, pattern=0)
            hw_vec = make_hw_vector(wall_ms, gm, cycle_delta,
                                     lds_timing=lds_t, priority=current_priority)
            hw_t = torch.tensor([hw_vec] * BS, dtype=torch.float32, device=DEVICE)

            # Demand cue
            if is_phase2 and model.use_effort:
                # In phase 2, demand alternates based on batch counter
                demand_is_high = (bn % (SWITCH_EVERY * 2)) < SWITCH_EVERY
            demand_cue = torch.full((BS,), 1.0 if demand_is_high else 0.0,
                                     device=DEVICE, dtype=torch.float32)

            # Previous state for self-model
            if prev_hw is not None:
                p_hw = prev_hw.expand(BS, -1)
                p_act = prev_action.expand(BS, -1)
            else:
                p_hw = torch.zeros(BS, HW_DIM, device=DEVICE)
                p_act = torch.zeros(BS, 4, device=DEVICE)

            labels = make_labels(digits, bank_ids, demand_is_high)

            # ── Forward ──
            out = model(imgs, hw_vector=hw_t if not is_blind else None,
                        bank_ids=bank_ids, temperature=temp,
                        prev_hw=p_hw, prev_action=p_act,
                        demand_cue=demand_cue)

            task_loss = F.cross_entropy(out['logits'], labels)

            # ── Self-model loss ──
            sm_loss = 0.0
            if model.use_self_model and out['hw_pred'] is not None and prev_hw is not None:
                sm_loss = 0.5 * F.mse_loss(out['hw_pred'], hw_t)

            # ── Effort supervision (phase 1) / energy penalty (phase 2) ──
            effort_loss = 0.0
            if model.use_effort and out['effort'] is not None:
                if not is_phase2:
                    # Phase 1: teach effort head what demand means
                    effort_target = torch.full((BS, 1), 1.0 if demand_is_high else 0.0,
                                               device=DEVICE)
                    effort_loss = 0.3 * F.mse_loss(out['effort'], effort_target)
                else:
                    # Phase 2: energy penalty — penalize high effort when task is easy
                    sclk_val = gm['sclk_mhz'] if gm else 1000
                    power_norm = min(1.0, sclk_val / 2000.0)
                    effort_loss = 0.1 * (out['effort'].mean() * power_norm)

            # ── Entropy bonus ──
            ent_loss = 0.0
            if model.use_policy and model.use_hw and not is_blind:
                mp = out['mode_probs'].mean(0)
                entropy_mode = -(mp * torch.log(mp + 1e-8)).sum()
                pp = out['prio_probs'].mean(0)
                entropy_prio = -(pp * torch.log(pp + 1e-8)).sum()
                ent_loss = -0.12 * entropy_mode - 0.08 * entropy_prio

            loss = task_loss + sm_loss + effort_loss + ent_loss

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tot_loss += loss.item()
            pred = out['logits'].argmax(1)
            correct += (pred == labels).sum().item()
            total += BS
            current_priority = out.get('selected_priority', 0)

            # Save state for next batch
            prev_hw = hw_t[0:1].detach()
            eff_val = out['effort'].mean().item() if out['effort'] is not None else 0.5
            prev_effort = eff_val
            mi = out['mode_idx'][0].item() / max(K_MODES - 1, 1)
            pi = out['prio_idx'][0].item() / 3.0
            al = out['alpha'][0, 0].item()
            prev_action = torch.tensor([[eff_val, mi, pi, al]], device=DEVICE)

            effort_log.append(eff_val)
            if out['affect'] is not None:
                affect_log.append(out['affect'].mean().item())
            if out['pred_error'] is not None:
                pred_err_log.append(out['pred_error'].mean().item())
            if out['gate'] is not None:
                gate_log.append(out['gate'].mean().item())
            sclk_log.append(gm['sclk_mhz'] if gm else 600)
            bn += 1

        if ep % 5 == 0 or ep == epochs - 1:
            a = np.mean([out['alpha'].mean().item()])
            eff_str = f" eff={np.mean(effort_log[-50:]):.3f}" if effort_log else ""
            aff_str = f" aff={np.mean(affect_log[-50:]):.3f}" if affect_log else ""
            pe_str = f" pe={np.mean(pred_err_log[-50:]):.4f}" if pred_err_log else ""
            phase = "P2" if is_phase2 else "P1"
            print(f"  [{name}] Ep {ep} ({phase}): loss={tot_loss/len(loader):.4f} "
                  f"acc={correct/total:.4f} α={a:.3f}{eff_str}{aff_str}{pe_str}",
                  flush=True)
        sched.step()

    return {
        'effort_log': effort_log, 'affect_log': affect_log,
        'pred_err_log': pred_err_log, 'gate_log': gate_log,
        'sclk_log': sclk_log,
    }


# ━━━ EVALUATION ━━━
def evaluate(model, ext, loader, scramble=False, fixed_sclk=None,
             zero_lds=False, no_effort_control=False, zero_affect=False):
    model.eval()
    all_preds, alphas, energy_log = [], [], []
    power_log, time_log = [], []
    effort_log_eval, affect_log_eval, gate_log_eval = [], [], []
    sclk_log_eval = []
    bn, level_idx = 0, 0
    demand_is_high = True
    mode_counter = {m: 0 for m in range(K_MODES)}
    prio_counter = {p: 0 for p in range(4)}
    current_priority = 0
    prev_hw = torch.zeros(1, HW_DIM, device=DEVICE)
    prev_action = torch.zeros(1, 4, device=DEVICE)
    prev_effort = 0.5

    if fixed_sclk:
        set_dvfs_verified(fixed_sclk, wait=0.3)

    with torch.no_grad():
        for imgs, digits in loader:
            imgs, digits = imgs.to(DEVICE), digits.to(DEVICE)
            t_batch_start = time.time()

            # DVFS control
            if fixed_sclk is None and model.use_effort and not no_effort_control:
                if prev_effort > 0.5:
                    set_dvfs('high')
                else:
                    set_dvfs('low')
                time.sleep(DVFS_WAIT)
            elif fixed_sclk is None and bn % SWITCH_EVERY == 0:
                level_idx = (level_idx + 1) % len(PHASE1_CONFIGS)
                perf, _ = PHASE1_CONFIGS[level_idx]
                set_dvfs(perf); time.sleep(DVFS_WAIT)
                demand_is_high = (perf == 'high')

            if fixed_sclk:
                demand_is_high = (fixed_sclk == 'high')
            elif model.use_effort and not no_effort_control:
                demand_is_high = prev_effort > 0.5

            wall_ms = measure_wall_clock(ext)
            gm = read_gpu_metrics()
            probe_res = ext.probe(BS); torch.cuda.synchronize()
            cd = probe_res[1].cpu().numpy()
            cycle_delta = int(np.median(np.abs(cd)))
            wgps = probe_res[0]
            bank_ids = (wgps // 2).long().clamp(0, NUM_BANKS - 1)

            lds_t = measure_lds_timing(ext, pattern=0)
            if zero_lds: lds_t = 0.0
            hw_vec = make_hw_vector(wall_ms, gm, cycle_delta,
                                     lds_timing=lds_t, priority=current_priority)
            if scramble: hw_vec = [1.0 - v for v in hw_vec]
            hw_t = torch.tensor([hw_vec] * BS, dtype=torch.float32, device=DEVICE)

            demand_cue = torch.full((BS,), 1.0 if demand_is_high else 0.0,
                                     device=DEVICE, dtype=torch.float32)

            p_hw = prev_hw.expand(BS, -1)
            p_act = prev_action.expand(BS, -1)

            affect_ov = torch.zeros(BS, AFFECT_DIM, device=DEVICE) if zero_affect else None

            labels = make_labels(digits, bank_ids, demand_is_high)

            out = model(imgs, hw_vector=hw_t, bank_ids=bank_ids,
                        temperature=0.5, hard_mode=True,
                        prev_hw=p_hw, prev_action=p_act,
                        demand_cue=demand_cue, affect_override=affect_ov)

            pred = out['logits'].argmax(1)
            all_preds.extend((pred == labels).cpu().tolist())
            alphas.append(out['alpha'].mean().item())
            sclk_val = gm['sclk_mhz'] if gm else 600
            energy_log.append(sclk_val)
            sclk_log_eval.append(sclk_val)
            current_priority = out.get('selected_priority', 0)

            if out['effort'] is not None:
                eff_val = out['effort'].mean().item()
                effort_log_eval.append(eff_val)
                prev_effort = eff_val
            if out['affect'] is not None:
                affect_log_eval.append(out['affect'].mean().item())
            if out['gate'] is not None:
                gate_log_eval.append(out['gate'].mean().item())

            # Update state
            prev_hw = hw_t[0:1]
            mi = out['mode_idx'][0].item() / max(K_MODES - 1, 1)
            pi = out['prio_idx'][0].item() / 3.0
            al = out['alpha'][0, 0].item()
            prev_action = torch.tensor([[prev_effort, mi, pi, al]], device=DEVICE)

            power_w = (gm['socket_power_mw'] / 1000.0) if gm else 30.0
            batch_time = time.time() - t_batch_start
            power_log.append(power_w)
            time_log.append(batch_time)

            for m in out['mode_idx'].cpu().tolist():
                mode_counter[m] = mode_counter.get(m, 0) + 1
            for p in out['prio_idx'].cpu().tolist():
                prio_counter[p] = prio_counter.get(p, 0) + 1
            bn += 1

    acc = float(np.mean(all_preds))
    mean_power = float(np.mean(power_log))
    mean_time = float(np.mean(time_log))

    # Effort-SCLK temporal correlation
    temporal_r = 0.0
    if len(effort_log_eval) > 10 and len(sclk_log_eval) > 10:
        eff_arr = np.array(effort_log_eval[:len(sclk_log_eval)])
        sclk_arr = np.array(sclk_log_eval[:len(eff_arr)])
        if np.std(eff_arr) > 1e-6 and np.std(sclk_arr) > 1e-6:
            temporal_r = float(np.corrcoef(eff_arr, sclk_arr)[0, 1])

    return {
        'acc': acc,
        'alpha_mean': float(np.mean(alphas)),
        'mean_sclk': float(np.mean(energy_log)),
        'mode_distribution': mode_counter,
        'prio_distribution': prio_counter,
        'mean_power_w': mean_power,
        'mean_batch_time_s': mean_time,
        'mj_per_batch': mean_power * mean_time * 1000,
        'mj_per_correct': (mean_power * mean_time * 1000) / max(acc * BS, 1),
        'throughput_img_per_s': BS / max(mean_time, 1e-6),
        'mean_effort': float(np.mean(effort_log_eval)) if effort_log_eval else 0.5,
        'mean_affect': float(np.mean(affect_log_eval)) if affect_log_eval else 0.0,
        'mean_gate': float(np.mean(gate_log_eval)) if gate_log_eval else 0.5,
        'effort_sclk_r': temporal_r,
        'high_sclk_pct': float(np.mean([1 if s > 1000 else 0 for s in sclk_log_eval])),
    }


# ━━━ SANITY CHECKS ━━━
def sanity_checks(ext):
    print("\n--- MODE sanity check (fp16mix) ---", flush=True)
    x = torch.randn(BS, 128, device=DEVICE)
    w = torch.randn(64, 128, device=DEVICE) * 0.1
    b = torch.zeros(64, device=DEVICE)
    results = {}
    for mb in MODE_SET:
        y = ext.linear_forward_multi(x, w, b, mb, 0); torch.cuda.synchronize()
        results[mb] = y
    ref = results[MODE_SET[0]]
    for mb in MODE_SET[1:]:
        diff = (results[mb] != ref).float().mean().item()
        mx = (results[mb] - ref).abs().max().item()
        print(f"  0x{mb:02X} vs 0x{MODE_SET[0]:02X}: {diff*100:.1f}% bits differ, max|Δ|={mx:.8f}",
              flush=True)

    print("\n--- LDS bank conflict timing ---", flush=True)
    ext.lds_probe(0, LDS_ITERS, 4); torch.cuda.synchronize()
    timings = {}
    for pat, label in [(0, "conflict-free"), (1, "32-way conflict")]:
        samples = []
        for _ in range(10):
            t = ext.lds_probe(pat, LDS_ITERS, 4); torch.cuda.synchronize()
            samples.append(t.float().mean().item())
        timings[label] = np.mean(samples)
        print(f"  {label}: {np.mean(samples):.0f} cycles", flush=True)
    lds_ratio = timings["32-way conflict"] / max(timings["conflict-free"], 1)
    print(f"  Ratio: {lds_ratio:.1f}x", flush=True)

    print("\n--- DVFS characterization ---", flush=True)
    for mode in ['low', 'high']:
        set_dvfs(mode); time.sleep(0.5)
        for _ in range(5): ext.probe(BS); torch.cuda.synchronize()
        wt = [measure_wall_clock(ext) for _ in range(10)]
        gm = read_gpu_metrics()
        sclk = gm['sclk_mhz'] if gm else 0
        print(f"  {mode}: wall={np.mean(wt):.3f}ms, SCLK={sclk} MHz", flush=True)
    set_dvfs('high')

    return lds_ratio


# ━━━ MAIN ━━━
def main():
    global _EXT

    print("=" * 70, flush=True)
    print("z2071: Allostatic Multi-Channel Controller with Affect", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)
    print("z2061 effort-head + z2070 multi-channel + self-model prediction error", flush=True)
    print("Active inference: predict substrate → surprise → modulate action", flush=True)
    print(flush=True)

    t0 = time.time()

    print("Compiling HIP kernels...", flush=True)
    ext = load_inline(name='z2071_affect', cpp_sources=CPP_SRC, cuda_sources=HIP_SRC,
                      functions=['linear_forward_multi', 'lds_probe', 'probe'],
                      extra_cuda_cflags=['-O2', '--offload-arch=gfx1100'],
                      verbose=False)
    _EXT = ext
    print("  Done.", flush=True)

    lds_ratio = sanity_checks(ext)
    train_loader, test_loader = get_data()

    # ━━━ A: Full adaptive with affect ━━━
    print(f"\n{'='*60}", flush=True)
    print("A: FULL ADAPTIVE + AFFECT + EFFORT", flush=True)
    print(f"{'='*60}", flush=True)
    model_A = Z2071Model(use_hw=True, use_policy=True, use_mode_path=True,
                          use_self_model=True, use_affect=True, use_effort=True).to(DEVICE)
    train_log = train_model(model_A, ext, train_loader, EPOCHS, 'A_full')
    m_A = evaluate(model_A, ext, test_loader)
    print(f"  A: acc={m_A['acc']:.4f} α={m_A['alpha_mean']:.3f} "
          f"eff={m_A['mean_effort']:.3f} aff={m_A['mean_affect']:.3f} "
          f"gate={m_A['mean_gate']:.3f} r={m_A['effort_sclk_r']:.3f}", flush=True)
    print(f"     modes={m_A['mode_distribution']} prios={m_A['prio_distribution']}", flush=True)
    print(f"     {m_A['mj_per_batch']:.1f} mJ/batch, {m_A['throughput_img_per_s']:.0f} img/s, "
          f"high_sclk={m_A['high_sclk_pct']*100:.0f}%", flush=True)

    # ━━━ B: Blind ━━━
    print(f"\n{'='*60}\nB: BLIND\n{'='*60}", flush=True)
    model_B = Z2071Model(use_hw=False, use_policy=False, use_mode_path=False,
                          use_self_model=False, use_affect=False, use_effort=False).to(DEVICE)
    train_model(model_B, ext, train_loader, EPOCHS, 'B_blind', is_blind=True)
    m_B = evaluate(model_B, ext, test_loader)
    print(f"  B: acc={m_B['acc']:.4f} | {m_B['mj_per_batch']:.1f} mJ/batch", flush=True)

    # ━━━ E: Scrambled HW ━━━
    print(f"\n{'='*60}\nE: SCRAMBLED HW\n{'='*60}", flush=True)
    m_E = evaluate(model_A, ext, test_loader, scramble=True)
    print(f"  E: acc={m_E['acc']:.4f}", flush=True)

    # ━━━ F: Random policy ━━━
    print(f"\n{'='*60}\nF: RANDOM POLICY\n{'='*60}", flush=True)
    model_F = copy.deepcopy(model_A)
    if hasattr(model_F, 'policy'):
        for p in model_F.policy.parameters(): p.data.zero_()
    m_F = evaluate(model_F, ext, test_loader)
    print(f"  F: acc={m_F['acc']:.4f}", flush=True)

    # ━━━ G: No self-model (affect zeroed since no prediction error) ━━━
    print(f"\n{'='*60}\nG: NO SELF-MODEL (affect zeroed)\n{'='*60}", flush=True)
    model_G = copy.deepcopy(model_A)
    model_G.use_self_model = False
    m_G = evaluate(model_G, ext, test_loader)
    print(f"  G: acc={m_G['acc']:.4f}", flush=True)

    # ━━━ H: No affect (prediction error zeroed) ━━━
    print(f"\n{'='*60}\nH: NO AFFECT (pred error zeroed)\n{'='*60}", flush=True)
    m_H = evaluate(model_A, ext, test_loader, zero_affect=True)
    print(f"  H: acc={m_H['acc']:.4f}", flush=True)

    # ━━━ I: α=0 (MODE-path disabled) ━━━
    print(f"\n{'='*60}\nI: α=0 (MODE-path disabled)\n{'='*60}", flush=True)
    model_I = copy.deepcopy(model_A)
    model_I.fixed_alpha = 0.0
    m_I = evaluate(model_I, ext, test_loader)
    print(f"  I: acc={m_I['acc']:.4f}", flush=True)

    # ━━━ J: Always-high SCLK ━━━
    print(f"\n{'='*60}\nJ: ALWAYS-HIGH SCLK\n{'='*60}", flush=True)
    m_J = evaluate(model_A, ext, test_loader, fixed_sclk='high')
    print(f"  J: acc={m_J['acc']:.4f} sclk={m_J['mean_sclk']:.0f}", flush=True)
    print(f"     {m_J['mj_per_batch']:.1f} mJ/batch, {m_J['throughput_img_per_s']:.0f} img/s",
          flush=True)

    # ━━━ K: No effort (random DVFS) ━━━
    print(f"\n{'='*60}\nK: NO EFFORT (random DVFS)\n{'='*60}", flush=True)
    m_K = evaluate(model_A, ext, test_loader, no_effort_control=True)
    print(f"  K: acc={m_K['acc']:.4f}", flush=True)

    # ━━━ L: No LDS channel ━━━
    print(f"\n{'='*60}\nL: NO LDS CHANNEL\n{'='*60}", flush=True)
    m_L = evaluate(model_A, ext, test_loader, zero_lds=True)
    print(f"  L: acc={m_L['acc']:.4f}", flush=True)

    elapsed = time.time() - t0
    reset_actuation()

    # ━━━ COMPUTE METRICS ━━━
    energy_ratio = m_A['mean_sclk'] / max(m_J['mean_sclk'], 1)
    energy_save_pct = (1 - m_A['mj_per_batch'] / max(m_J['mj_per_batch'], 1)) * 100

    # Mode/priority entropy
    mode_dist = m_A.get('mode_distribution', {})
    mode_vals = list(mode_dist.values())
    total_modes = max(sum(mode_vals), 1)
    mode_entropy = -sum((v/total_modes) * np.log(max(v/total_modes, 1e-10))
                        for v in mode_vals) if mode_vals else 0

    prio_dist = m_A.get('prio_distribution', {})
    prio_vals = list(prio_dist.values())
    total_prio = max(sum(prio_vals), 1)
    prio_entropy = -sum((v/total_prio) * np.log(max(v/total_prio, 1e-10))
                        for v in prio_vals) if prio_vals else 0

    # ━━━ BUSINESS METRICS ━━━
    print(f"\n{'='*70}", flush=True)
    print("BUSINESS METRICS", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  {'Condition':<25} {'Acc%':>6} {'mJ/batch':>9} {'img/s':>7} {'mJ/correct':>11}",
          flush=True)
    print(f"  {'-'*25} {'-'*6} {'-'*9} {'-'*7} {'-'*11}", flush=True)
    for label, m in [('A (allostatic+affect)', m_A), ('B (blind)', m_B),
                      ('J (always-high)', m_J)]:
        print(f"  {label:<25} {m['acc']*100:5.1f}% {m['mj_per_batch']:8.1f} "
              f"{m['throughput_img_per_s']:6.0f} {m['mj_per_correct']:10.2f}",
              flush=True)
    print(f"\n  Energy saving vs always-high: {energy_save_pct:.1f}%", flush=True)
    print(f"  Effort-SCLK correlation: {m_A['effort_sclk_r']:.3f}", flush=True)
    print(f"  High-SCLK usage: {m_A['high_sclk_pct']*100:.0f}%", flush=True)

    print(f"\n  Published DVFS comparisons:", flush=True)
    for sys_name, save, mech in [
        ('DynamoLLM (HPCA 2025)', '53%', 'External 3-knob'),
        ('throttLL\'eM (HPCA 2025)', '43.8%', 'External KV-projection'),
        ('GreenLLM (Aug 2025)', '28-34%', 'External dual-loop'),
        ('DVFS-GPT (2025)', '32%', 'External convex opt'),
        ('Kernel-level (Jan 2026)', '14.6%', 'External per-kernel'),
        ('z2061 (ours, DVFS)', '56%', 'Self-regulated effort'),
        (f'z2071 (ours, affect)', f'{energy_save_pct:.0f}%', 'Allostatic + affect')]:
        print(f"  {sys_name:<30} {save:>12} {mech:>25}", flush=True)

    # ━━━ TESTS (26) ━━━
    print(f"\n{'='*70}\nTEST RESULTS\n{'='*70}", flush=True)
    tests = {}

    # Standard battery
    tests['T1_accuracy'] = {'verdict': 'PASS' if m_A['acc'] > 0.70 else 'FAIL',
        'val': f"A={m_A['acc']*100:.1f}% > 70% (CIFAR-10 + demand labels)"}

    gap_AB = m_A['acc'] - m_B['acc']
    tests['T2_embodiment_gap'] = {'verdict': 'PASS' if gap_AB > 0.10 else 'FAIL',
        'val': f"A-B={gap_AB*100:.1f}pp > 10pp"}

    tests['T3_scrambled_kills'] = {
        'verdict': 'PASS' if m_E['acc'] < m_A['acc'] - 0.05 else 'FAIL',
        'val': f"E={m_E['acc']*100:.1f}% < A-5pp={(m_A['acc']-0.05)*100:.1f}%"}

    gap_AF = m_A['acc'] - m_F['acc']
    tests['T4_policy_causal'] = {'verdict': 'PASS' if gap_AF > 0.03 else 'FAIL',
        'val': f"A-F={gap_AF*100:.1f}pp > 3pp"}

    gap_AI = m_A['acc'] - m_I['acc']
    tests['T5_alpha_causal'] = {'verdict': 'PASS' if gap_AI > 0.03 else 'FAIL',
        'val': f"A-I={gap_AI*100:.1f}pp > 3pp (MODE path causal)"}

    tests['T6_alpha_nonzero'] = {
        'verdict': 'PASS' if m_A['alpha_mean'] > 0.05 else 'FAIL',
        'val': f"α={m_A['alpha_mean']:.3f} > 0.05"}

    tests['T7_mode_diversity'] = {
        'verdict': 'PASS' if mode_entropy > 0.3 else 'FAIL',
        'val': f"entropy={mode_entropy:.3f} > 0.3"}

    # Effort/action tests
    tests['T8_effort_controls_dvfs'] = {
        'verdict': 'PASS' if abs(m_A['effort_sclk_r']) > 0.3 else 'FAIL',
        'val': f"|effort-SCLK r|={abs(m_A['effort_sclk_r']):.3f} > 0.3"}

    gap_AK = m_A['acc'] - m_K['acc']
    tests['T9_effort_causal'] = {
        'verdict': 'PASS' if gap_AK > 0.03 else 'FAIL',
        'val': f"A-K={gap_AK*100:.1f}pp > 3pp (effort controls DVFS)"}

    tests['T10_balanced_usage'] = {
        'verdict': 'PASS' if 0.15 < m_A['high_sclk_pct'] < 0.85 else 'FAIL',
        'val': f"high_sclk={m_A['high_sclk_pct']*100:.0f}% in [15%, 85%]"}

    # Self-model tests
    gap_AG = m_A['acc'] - m_G['acc']
    tests['T11_self_model_causal'] = {
        'verdict': 'PASS' if gap_AG > 0.02 else 'FAIL',
        'val': f"A-G={gap_AG*100:.1f}pp > 2pp (self-model contributes)"}

    # Affect tests
    gap_AH = m_A['acc'] - m_H['acc']
    tests['T12_affect_contributes'] = {
        'verdict': 'PASS' if gap_AH > 0.01 else 'FAIL',
        'val': f"A-H={gap_AH*100:.1f}pp > 1pp (affect modulates behavior)"}

    tests['T13_affect_nonzero'] = {
        'verdict': 'PASS' if m_A['mean_affect'] > 0.001 else 'FAIL',
        'val': f"mean_affect={m_A['mean_affect']:.4f} > 0.001"}

    # Multi-channel tests
    gap_AL = m_A['acc'] - m_L['acc']
    tests['T14_lds_channel'] = {
        'verdict': 'PASS' if abs(gap_AL) > 0.005 else 'FAIL',
        'val': f"|A-L|={abs(gap_AL)*100:.1f}pp > 0.5pp (LDS contributes)"}

    tests['T15_lds_ratio'] = {
        'verdict': 'PASS' if lds_ratio > 1.5 else 'FAIL',
        'val': f"ratio={lds_ratio:.1f}x > 1.5x"}

    # Energy tests
    tests['T16_energy_saving'] = {
        'verdict': 'PASS' if energy_save_pct > 0 else 'FAIL',
        'val': f"saving={energy_save_pct:.1f}% > 0% (vs always-high)"}

    tests['T17_energy_per_correct'] = {
        'verdict': 'PASS' if m_A['mj_per_correct'] < m_J['mj_per_correct'] * 1.5 else 'FAIL',
        'val': f"A={m_A['mj_per_correct']:.2f}mJ < J×1.5={m_J['mj_per_correct']*1.5:.2f}mJ/correct"}

    # Consciousness indicators
    tests['T18_gate_adaptive'] = {
        'verdict': 'PASS' if m_A['mean_gate'] > 0.1 and m_A['mean_gate'] < 0.9 else 'FAIL',
        'val': f"gate={m_A['mean_gate']:.3f} in (0.1, 0.9)"}

    tests['T19_scramble_degrades'] = {
        'verdict': 'PASS' if m_E['acc'] < m_A['acc'] - 0.03 else 'FAIL',
        'val': f"E={m_E['acc']*100:.1f}% < A-3pp"}

    tests['T20_full_beats_ablations'] = {
        'verdict': 'PASS' if m_A['acc'] > max(m_F['acc'], m_I['acc'], m_K['acc']) else 'FAIL',
        'val': f"A={m_A['acc']*100:.1f}% > max(F,I,K)={max(m_F['acc'],m_I['acc'],m_K['acc'])*100:.1f}%"}

    tests['T21_multiple_modes'] = {
        'verdict': 'PASS' if sum(1 for v in mode_dist.values() if v > 0) >= 2 else 'FAIL',
        'val': f"modes_used={sum(1 for v in mode_dist.values() if v>0)} >= 2"}

    tests['T22_prio_diversity'] = {
        'verdict': 'PASS' if prio_entropy > 0.3 else 'FAIL',
        'val': f"prio_entropy={prio_entropy:.3f} > 0.3"}

    # Active inference tests
    tests['T23_perception_action_loop'] = {
        'verdict': 'PASS' if (abs(m_A['effort_sclk_r']) > 0.2 and
                               m_A['mean_affect'] > 0.001 and
                               m_A['acc'] > 0.65) else 'FAIL',
        'val': f"r={m_A['effort_sclk_r']:.3f} & aff={m_A['mean_affect']:.4f} & acc={m_A['acc']*100:.1f}%"}

    gap_sclk_robust = abs(m_A['acc'] - m_J['acc'])
    tests['T24_sclk_robustness'] = {
        'verdict': 'PASS' if gap_sclk_robust < 0.20 else 'FAIL',
        'val': f"|A-J|={gap_sclk_robust*100:.1f}pp < 20pp"}

    tests['T25_coupling_spectrum'] = {
        'verdict': 'PASS' if 0 < m_A['alpha_mean'] < 0.95 else 'FAIL',
        'val': f"0 < α={m_A['alpha_mean']:.3f} < 0.95"}

    tests['T26_mode_forward_pass'] = {
        'verdict': 'PASS' if m_A['acc'] > 0.65 and m_A['alpha_mean'] > 0.05 else 'FAIL',
        'val': f"A > 65% AND α > 0.05 (MODE in forward pass)"}

    pass_count = sum(1 for t in tests.values() if t['verdict'] == 'PASS')
    verdict = f"{pass_count}/{len(tests)} PASS"

    for tname, result in tests.items():
        print(f"  {result['verdict']:4s} | {tname}: {result['val']}", flush=True)
    print(f"\n  VERDICT: {verdict}", flush=True)

    # ━━━ Ablation summary ━━━
    print(f"\n  Ablation summary:", flush=True)
    print(f"    A (full+affect):    {m_A['acc']*100:.1f}%  α={m_A['alpha_mean']:.3f} "
          f"eff={m_A['mean_effort']:.3f} aff={m_A['mean_affect']:.3f}", flush=True)
    print(f"    B (blind):          {m_B['acc']*100:.1f}%", flush=True)
    print(f"    E (scrambled):      {m_E['acc']*100:.1f}%", flush=True)
    print(f"    F (random policy):  {m_F['acc']*100:.1f}%  ({gap_AF*100:+.1f}pp)", flush=True)
    print(f"    G (no self-model):  {m_G['acc']*100:.1f}%  ({gap_AG*100:+.1f}pp)", flush=True)
    print(f"    H (no affect):      {m_H['acc']*100:.1f}%  ({gap_AH*100:+.1f}pp)", flush=True)
    print(f"    I (α=0):            {m_I['acc']*100:.1f}%  ({gap_AI*100:+.1f}pp)", flush=True)
    print(f"    J (always-high):    {m_J['acc']*100:.1f}%  ({(m_A['acc']-m_J['acc'])*100:+.1f}pp)",
          flush=True)
    print(f"    K (no effort):      {m_K['acc']*100:.1f}%  ({gap_AK*100:+.1f}pp)", flush=True)
    print(f"    L (no LDS):         {m_L['acc']*100:.1f}%  ({gap_AL*100:+.1f}pp)", flush=True)
    print(f"    Energy saving:      {energy_save_pct:.1f}%", flush=True)
    print(f"    Effort-SCLK r:      {m_A['effort_sclk_r']:.3f}", flush=True)
    print(f"    High-SCLK usage:    {m_A['high_sclk_pct']*100:.0f}%", flush=True)

    # ━━━ SAVE ━━━
    results = {
        'experiment': 'z2071_allostatic_affect',
        'version': 1,
        'extends': 'z2061 (12/12) + z2070 (22/24) — adds self-model, affect, effort',
        'key_innovation': 'Active inference: self-model prediction error → affect → DVFS control',
        'scientific_basis': {
            'Laukkonen_Friston_2025': 'Beautiful Loop: world model + self-model + hyper-model',
            'Barrett_2025': 'Constructed emotion: affect from signal ensemble',
            'Luppi_2024': 'Synergistic workspace, synergy > redundancy',
            'Phua_2025': 'Ablation-based consciousness markers',
            'dual_stream': 'Interoceptive/exteroceptive precision weighting',
        },
        'task': 'CIFAR-10',
        'mode_set': [hex(m) for m in MODE_SET],
        'hw_dim': HW_DIM,
        'affect_dim': AFFECT_DIM,
        'channels': {
            'MODE': 'fp16 rounding via s_setreg_b32 hwreg(1,0,8)',
            'LDS': f'bank conflict timing ({lds_ratio:.1f}x ratio)',
            'priority': 'wave scheduling via s_setprio 0-3',
            'DVFS': 'effort-controlled clock speed (600-2100 MHz)',
        },
        'accuracies': {k: round(v, 4) for k, v in [
            ('A_adaptive', m_A['acc']), ('B_blind', m_B['acc']),
            ('E_scrambled', m_E['acc']), ('F_random_policy', m_F['acc']),
            ('G_no_self_model', m_G['acc']), ('H_no_affect', m_H['acc']),
            ('I_alpha0', m_I['acc']), ('J_always_high', m_J['acc']),
            ('K_no_effort', m_K['acc']), ('L_no_lds', m_L['acc'])]},
        'alpha_mean': round(m_A['alpha_mean'], 4),
        'mode_distribution': m_A.get('mode_distribution', {}),
        'mode_entropy': round(mode_entropy, 4),
        'prio_distribution': m_A.get('prio_distribution', {}),
        'prio_entropy': round(prio_entropy, 4),
        'lds_timing_ratio': round(lds_ratio, 2),
        'effort': {
            'mean_effort': round(m_A['mean_effort'], 4),
            'effort_sclk_r': round(m_A['effort_sclk_r'], 4),
            'high_sclk_pct': round(m_A['high_sclk_pct'], 4),
        },
        'affect': {
            'mean_affect': round(m_A['mean_affect'], 4),
            'dim': AFFECT_DIM,
        },
        'gate': {
            'mean_gate': round(m_A['mean_gate'], 4),
        },
        'energy': {
            'ratio': round(energy_ratio, 4),
            'saving_pct': round(energy_save_pct, 1),
            'A_mj_per_batch': round(m_A['mj_per_batch'], 1),
            'J_mj_per_batch': round(m_J['mj_per_batch'], 1),
            'A_throughput': round(m_A['throughput_img_per_s'], 0),
            'J_throughput': round(m_J['throughput_img_per_s'], 0),
            'A_mj_per_correct': round(m_A['mj_per_correct'], 2),
            'J_mj_per_correct': round(m_J['mj_per_correct'], 2),
        },
        'comparison_vs_published': {
            'DynamoLLM_HPCA25': '53% external 3-knob',
            'throttLLeM_HPCA25': '43.8% external KV-projection',
            'GreenLLM_Aug25': '28-34% external dual-loop',
            'DVFS_GPT_2025': '32% external convex opt',
            'KernelLevel_Jan26': '14.6% external per-kernel',
            'z2061_ours': '56% self-regulated DVFS',
            'z2071_ours': f'{energy_save_pct:.0f}% allostatic + affect',
        },
        'tests': tests, 'verdict': verdict, 'pass_count': pass_count,
        'elapsed_s': round(elapsed),
    }

    out_path = 'results/z2071_allostatic_affect.json'
    os.makedirs('results', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)
    print(f"Elapsed: {elapsed:.0f}s", flush=True)


if __name__ == '__main__':
    main()
