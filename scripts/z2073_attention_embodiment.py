#!/usr/bin/env python3
"""z2073: Attention-Based Hardware Self-Awareness + Closed-Loop Control
===========================================================================
ALL sensors + ALL actuators from the comprehensive inventory.

SENSORS (8 channels):
  freq_est, cycle_var, wgp, status, hw_id2, trapsts, gpr_alloc, wall_ratio

ACTUATORS (6 channels):
  MODE[9:0] (rounding+denorm), s_setprio, s_sleep, DVFS (effort-controlled),
  mod_factor (continuous freq_est->math coupling), chain_depth

KEY PATTERNS:
  z2060: homeostatic loss + exclusive specialization
  z2061: effort head -> DVFS control (closed sensorimotor loop)
  z2073: GWT attention + deep ISA kernel + world model + ALL sensors/actuators

Causal chain:
  demand -> effort -> DVFS -> SCLK -> freq_est -> mod_factor -> math output
  demand -> effort -> DVFS -> SCLK -> freq_est -> self-model -> gate -> routing
  Ablate effort -> stuck DVFS -> wrong labels -> crash
  Ablate self-model -> gate=0.5 -> blend conflicting heads -> crash
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
PHASE1_EPOCHS = 25
PHASE2_EPOCHS = 20
EPOCHS = PHASE1_EPOCHS + PHASE2_EPOCHS
SWITCH_EVERY = 4
DVFS_WAIT = 0.05

# --- Dimensions ---
SENSOR_DIM = 8   # freq_est, cycle_var, wgp, status, hwid2, trapsts, gpr, wall_ratio
HW_DIM = 14      # 8 sensors + 6 actions (mode_rnd, mode_dnrm, prio, dvfs_f, sleep, mod_factor)
N_ACTIONS = 5    # for world model prev_action input (mode_rnd, mode_dnrm, prio, sleep, mod_f)

MODE_SET = [0x000, 0x005, 0x00F, 0x0F0, 0x0FF, 0x100, 0x200, 0x300]
K_MODES = len(MODE_SET)
SLEEP_LEVELS = [0, 1, 2, 3]  # s_sleep 0..3 (~0, ~128, ~256, ~512 cycles)
DVFS_PATH = None

# ============================================================
# HIP KERNEL: ALL sensors + ALL actuators
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
    int* __restrict__ mode_out)
{
    // ═══ ACTUATORS ═══
    // A) MODE register (rounding + denorm + DX10/IEEE)
    unsigned int m = __builtin_amdgcn_readfirstlane(mode_byte & 0x3FFu);
    asm volatile("s_setreg_b32 hwreg(1, 0, 10), %0" : : "s"(m));

    // B) Wave priority
    unsigned int p = __builtin_amdgcn_readfirstlane((unsigned int)(priority & 3));
    if (p == 0) { asm volatile("s_setprio 0"); }
    else if (p == 1) { asm volatile("s_setprio 1"); }
    else if (p == 2) { asm volatile("s_setprio 2"); }
    else { asm volatile("s_setprio 3"); }

    // C) s_sleep (scheduling delay actuator)
    int sa = __builtin_amdgcn_readfirstlane(sleep_amt & 3);
    if (sa == 1) { asm volatile("s_sleep 1"); }
    else if (sa == 2) { asm volatile("s_sleep 2"); }
    else if (sa == 3) { asm volatile("s_sleep 3"); }

    // ═══ SENSORS (start) ═══
    unsigned int c0;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(c0));
    c0 = __builtin_amdgcn_readfirstlane(c0);

    unsigned int hw1;
    asm volatile("s_getreg_b32 %0, hwreg(23)" : "=s"(hw1));
    unsigned int wgp = (hw1 >> 7) & 0xF;
    unsigned int simd = (hw1 >> 4) & 0x3;

    unsigned int mode_actual;
    asm volatile("s_getreg_b32 %0, hwreg(1, 0, 10)" : "=s"(mode_actual));

    // ═══ COMPUTATION with stochastic rounding + mod_factor ═══
    unsigned int sr_seed = c0 ^ (wgp << 16) ^ (simd << 20) ^
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

    // D) mod_factor: continuous math coupling (freq_est -> output)
    int mf_bits = __builtin_amdgcn_readfirstlane(*(const int*)&mod_factor);
    acc *= *(float*)&mf_bits;

    if (row < M && col < N)
        Y[row * N + col] = acc + B[col];

    // ═══ SENSORS (end) ═══
    unsigned int c1;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(c1));
    c1 = __builtin_amdgcn_readfirstlane(c1);

    int bid = blockIdx.y * gridDim.x + blockIdx.x;
    if (threadIdx.x == 0 && threadIdx.y == 0 && bid < M) {
        wgp_out[bid] = (int)wgp;
        cycle_out[bid] = (int)(c1 - c0);
        mode_out[bid] = (int)mode_actual;
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
    auto mode_o = torch::zeros({nb}, io);
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
        wgp_o.data_ptr<int>(), cyc_o.data_ptr<int>(), mode_o.data_ptr<int>());
    return {Y, wgp_o, cyc_o, mode_o};
}

// ─── Probe kernel: reads ALL sensors ───
__global__ void probe_kernel(
    int* wgp_ids, int* cycles, int* status_out,
    int* hwid2_out, int* trapsts_out, int* gpr_out,
    int n)
{
    int bid = blockIdx.x;
    if (bid >= n || threadIdx.x != 0) return;

    // SHADER_CYCLES start
    unsigned int c0;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(c0));
    c0 = __builtin_amdgcn_readfirstlane(c0);

    // HW_ID1 (hwreg 23): WGP + SIMD
    unsigned int hw1;
    asm volatile("s_getreg_b32 %0, hwreg(23)" : "=s"(hw1));
    wgp_ids[bid] = (int)((hw1 >> 7) & 0xF);

    // STATUS (hwreg 2)
    unsigned int stat;
    asm volatile("s_getreg_b32 %0, hwreg(2)" : "=s"(stat));
    status_out[bid] = (int)(stat & 0xFFFFF);

    // HW_ID2 (hwreg 24): queue/pipe/engine context
    unsigned int hwid2;
    asm volatile("s_getreg_b32 %0, hwreg(24)" : "=s"(hwid2));
    hwid2_out[bid] = (int)hwid2;

    // TRAPSTS (hwreg 3): exception/trap status
    unsigned int trapsts;
    asm volatile("s_getreg_b32 %0, hwreg(3)" : "=s"(trapsts));
    trapsts_out[bid] = (int)trapsts;

    // GPR_ALLOC (hwreg 5): register allocation
    unsigned int gpr;
    asm volatile("s_getreg_b32 %0, hwreg(5)" : "=s"(gpr));
    gpr_out[bid] = (int)gpr;

    // Work loop for cycle timing
    float acc = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < 5000; i++) acc += 1.0f / (float)(i+1);

    // SHADER_CYCLES end
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
    auto h2 = torch::zeros({(int)n}, io);
    auto tr = torch::zeros({(int)n}, io);
    auto gp = torch::zeros({(int)n}, io);
    probe_kernel<<<(int)n, 32>>>(
        w.data_ptr<int>(), c.data_ptr<int>(), s.data_ptr<int>(),
        h2.data_ptr<int>(), tr.data_ptr<int>(), gp.data_ptr<int>(),
        (int)n);
    return {w, c, s, h2, tr, gp};
}
'''

CPP_SRC = r'''
#include <torch/extension.h>
std::vector<torch::Tensor> hw_matmul(torch::Tensor, torch::Tensor, torch::Tensor,
                                      int64_t, int64_t, int64_t, int64_t, double);
std::vector<torch::Tensor> probe(int64_t);
'''


# === AUTOGRAD ===
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


# === DVFS ===
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
        time.sleep(DVFS_WAIT)
        return True
    except: return False


# === HW SENSING (all 8 sensor channels) ===
def read_hw(ext):
    """Read ALL hardware sensors via probe kernel."""
    res = ext.probe(64)
    torch.cuda.synchronize()
    wgps = res[0].cpu().numpy()
    cycles = res[1].cpu().numpy()
    status = res[2].cpu().numpy()
    hwid2 = res[3].cpu().numpy()
    trapsts = res[4].cpu().numpy()
    gpr = res[5].cpu().numpy()

    cycle_mean = float(np.mean(cycles))
    return {
        'wgp_norm': min(1.0, int(np.median(wgps)) / 14.0),
        'cycle_var': float(np.std(cycles)) / max(cycle_mean, 1),
        'status_norm': float(int(np.median(status)) & 0xFFFFF) / 1048575.0,
        'hwid2_norm': float(int(np.median(hwid2)) & 0xFF) / 255.0,
        'trapsts_norm': min(1.0, float(int(np.median(trapsts))) / 255.0),
        'gpr_alloc_norm': float(int(np.median(gpr)) & 0xFF) / 255.0,
        'cycle_mean': cycle_mean,
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
        cycles = res[1].cpu().numpy()
        cycle_med = float(np.median(cycles))
        if wall_ms > 0.01:
            freqs.append(cycle_med / wall_ms)
    return float(np.median(freqs)) if freqs else 1.0

def compute_mod_factor(freq_est, baseline_freq):
    """Continuous math coupling: freq_est -> output scaling.
    Makes every deep kernel output depend on GPU frequency state."""
    freq_norm = freq_est / max(baseline_freq, 1.0)
    return 1.0 + (freq_norm - 1.0) * 0.1  # small: ~0.97 to ~1.03

def make_hw_vector(hw_info, mode_byte, dvfs_state, freq_est, priority,
                   baseline_freq, sleep_amt, mod_factor):
    """14-dim: SENSOR[0:8] + ACTION[8:14].
    Self-model sees ONLY sensor channels [0:8]."""
    freq_norm = freq_est / max(baseline_freq, 1.0)
    freq_norm = min(2.0, max(0.0, freq_norm))

    # wall_ratio: cycles / (freq_est * expected_time) — scheduling efficiency proxy
    # If GPU is contended, wall time increases relative to cycles
    wall_ratio = min(1.0, hw_info['cycle_mean'] / max(freq_est, 1.0))

    rnd = (mode_byte & 0x0F) / 15.0
    dnrm = ((mode_byte >> 4) & 0x0F) / 15.0
    prio_norm = priority / 3.0
    dvfs_f = 1.0 if dvfs_state == 'high' else 0.0
    sleep_norm = sleep_amt / 3.0
    mod_norm = (mod_factor - 0.9) / 0.2  # normalize [0.9, 1.1] -> [0, 1]

    return [
        # SENSORS [0:8]
        freq_norm,                      # [0] voltage/frequency proxy (KEY)
        hw_info['cycle_var'],          # [1] timing variance
        hw_info['wgp_norm'],           # [2] physical WGP placement
        hw_info['status_norm'],        # [3] STATUS register
        hw_info['hwid2_norm'],         # [4] HW_ID2 (queue context)
        hw_info['trapsts_norm'],       # [5] TRAPSTS (exceptions)
        hw_info['gpr_alloc_norm'],     # [6] GPR_ALLOC (register pressure)
        wall_ratio,                     # [7] wall timing ratio
        # ACTIONS [8:14]
        rnd,                            # [8] MODE rounding bits
        dnrm,                           # [9] MODE denorm bits
        prio_norm,                      # [10] wave priority
        dvfs_f,                         # [11] DVFS state (masked from self-model)
        sleep_norm,                     # [12] s_sleep amount
        mod_norm,                       # [13] mod_factor (math coupling)
    ]


# === GWT ATTENTION WORKSPACE ===
class HWAttentionBlock(nn.Module):
    """Global Workspace: feature slots + hardware token compete via self-attention."""
    def __init__(self, feat_dim=128, hw_dim=HW_DIM):
        super().__init__()
        self.n_slots = 4
        self.slot_dim = feat_dim // self.n_slots  # 32
        self.hw_proj = nn.Linear(hw_dim, self.slot_dim)
        self.attn = nn.MultiheadAttention(
            self.slot_dim, num_heads=2, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(self.slot_dim)
        self._last_weights = None

    def forward(self, features, hw_state):
        B = features.shape[0]
        slots = features.view(B, self.n_slots, self.slot_dim)
        hw_token = self.hw_proj(hw_state).unsqueeze(1)
        tokens = torch.cat([hw_token, slots], dim=1)  # (B, 5, 32)
        attended, weights = self.attn(tokens, tokens, tokens)
        attended = self.norm(tokens + attended)
        self._last_weights = weights.detach()
        return attended[:, 1:, :].reshape(B, -1)  # (B, 128)


# === MODEL ===
class Z2073Model(nn.Module):
    def __init__(self, use_self_model=True, use_gate=True, use_deep=True,
                 use_attention=True, use_world_model=True, use_effort=True):
        super().__init__()
        self.use_self_model = use_self_model
        self.use_gate = use_gate
        self.use_deep = use_deep
        self.use_attention = use_attention
        self.use_world_model = use_world_model
        self.use_effort = use_effort

        # CIFAR-10 encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128*8*8, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU())

        # GWT Attention workspace
        if use_attention:
            self.hw_attn = HWAttentionBlock(128, HW_DIM)

        # Deep kernel path (uses ISA actuators)
        self.pre_deep_ln = nn.LayerNorm(128)
        self.deep_fc = HWLinear(128, 64)
        self.head_A = HWLinear(64, 10)
        self.dropout = nn.Dropout(0.3)

        # Light path (pure software, no ISA actuators)
        self.light_fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3))
        self.head_B = nn.Linear(64, 10)

        # Self-model: SENSOR-ONLY[0:8] -> predict DVFS
        if use_self_model:
            self.self_model = nn.Sequential(
                nn.Linear(SENSOR_DIM, 32), nn.ReLU(),
                nn.Linear(32, 1))

        # Gate: self-model logit -> routing weight
        if use_gate:
            self.gate_net = nn.Sequential(
                nn.Linear(1, 8), nn.ReLU(),
                nn.Linear(8, 1), nn.Sigmoid())

        # World model: prev_hw + prev_actions -> predict next freq_est
        if use_world_model:
            self.world_model = nn.Sequential(
                nn.Linear(HW_DIM + N_ACTIONS, 32), nn.ReLU(),
                nn.Linear(32, 1), nn.Sigmoid())

        # Effort head: controls DVFS (z2061 closed-loop)
        if use_effort:
            self.demand_proj = nn.Sequential(
                nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU())
            self.effort_head = nn.Sequential(
                nn.Linear(128 + 16, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, 1))

        # Mode head: choose MODE config
        self.mode_head = nn.Sequential(
            nn.Linear(SENSOR_DIM, 16), nn.ReLU(), nn.Linear(16, K_MODES))

        # Priority head
        self.priority_head = nn.Sequential(
            nn.Linear(SENSOR_DIM, 8), nn.ReLU(), nn.Linear(8, 4))

        # Sleep head: choose s_sleep level
        self.sleep_head = nn.Sequential(
            nn.Linear(SENSOR_DIM, 8), nn.ReLU(), nn.Linear(8, len(SLEEP_LEVELS)))

    def forward(self, x, hw_vector=None, mode_byte=0, chain_depth=8,
                priority=0, demand_cue=None, sleep_amt=0, mod_factor=1.0):
        h = self.encoder(x)
        B_size = h.shape[0]

        # Deep attended path
        if self.use_attention and hw_vector is not None:
            h_deep = self.hw_attn(h, hw_vector)
        else:
            h_deep = h

        h_deep = self.pre_deep_ln(h_deep)

        if self.use_deep:
            h1 = self.dropout(F.relu(self.deep_fc(
                h_deep, mode_byte, chain_depth, priority, sleep_amt, mod_factor)))
            yA = self.head_A(h1, mode_byte, chain_depth, priority, sleep_amt, mod_factor)
        else:
            h1 = self.dropout(F.relu(F.linear(h_deep, self.deep_fc.weight, self.deep_fc.bias)))
            yA = F.linear(h1, self.head_A.weight, self.head_A.bias)

        # Light path
        h_light = self.light_fc(h)
        yB = self.head_B(h_light)

        # Self-model: sensor-only -> DVFS prediction
        self_pred = torch.zeros(B_size, 1, device=x.device)
        if self.use_self_model and hw_vector is not None:
            sensor_only = hw_vector[:, :SENSOR_DIM]
            self_pred = self.self_model(sensor_only)

        # Effort head (z2061): encoder.detach() + demand -> DVFS control
        effort = None
        if self.use_effort and demand_cue is not None:
            h_demand = self.demand_proj(demand_cue.unsqueeze(-1))
            effort_input = torch.cat([h.detach(), h_demand], dim=1)
            effort = torch.sigmoid(self.effort_head(effort_input))

        # Mode, priority, sleep heads
        chosen_mode_idx = 0
        chosen_priority = priority
        chosen_sleep = sleep_amt
        mode_logits = prio_logits = sleep_logits = None

        if hw_vector is not None:
            sensor_only = hw_vector[:, :SENSOR_DIM]
            mode_logits = self.mode_head(sensor_only)
            prio_logits = self.priority_head(sensor_only)
            sleep_logits = self.sleep_head(sensor_only)
            if self.training:
                mp = F.gumbel_softmax(mode_logits, tau=1.0, hard=True)
                chosen_mode_idx = mp[0].argmax().item()
                pp = F.gumbel_softmax(prio_logits, tau=1.0, hard=True)
                chosen_priority = pp[0].argmax().item()
                sp = F.gumbel_softmax(sleep_logits, tau=1.0, hard=True)
                chosen_sleep = sp[0].argmax().item()
            else:
                chosen_mode_idx = mode_logits[0].argmax().item()
                chosen_priority = prio_logits[0].argmax().item()
                chosen_sleep = sleep_logits[0].argmax().item()

        # Gate
        if self.use_gate:
            g = self.gate_net(self_pred)
        else:
            g = torch.full((B_size, 1), 0.5, device=x.device)

        logits = g * yA + (1.0 - g) * yB

        return {
            'logits': logits, 'gate': g, 'self_pred': self_pred,
            'effort': effort, 'yA': yA, 'yB': yB,
            'mode_logits': mode_logits, 'prio_logits': prio_logits,
            'sleep_logits': sleep_logits,
            'chosen_mode_idx': chosen_mode_idx,
            'chosen_priority': chosen_priority,
            'chosen_sleep': chosen_sleep,
        }


def make_labels(digits, demand_is_high):
    """Exclusive specialization: labels depend on DEMANDED state."""
    if demand_is_high:
        return (digits + 3) % 10
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


# === TRAINING (2-phase: z2061 pattern) ===
def train_model(model, ext, loader, epochs, name, fixed_dvfs=None,
                model_controls_dvfs=True):
    # Separate param groups: effort head gets LR boost in Phase 2
    effort_params = []
    other_params = []
    effort_names = {'effort_head', 'demand_proj'}
    for pname, p in model.named_parameters():
        if any(en in pname for en in effort_names):
            effort_params.append(p)
        else:
            other_params.append(p)
    opt = torch.optim.Adam([
        {'params': other_params, 'lr': 1e-3},
        {'params': effort_params, 'lr': 1e-3}  # group 1: effort
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[35, 42], gamma=0.3)
    model.train()
    bn, config_idx = 0, 0
    dvfs_state = 'high'
    sclk_high = True
    has_dvfs = fixed_dvfs is None
    log = {'gate': [], 'sm_acc': [], 'freq': [], 'wm_err': [], 'effort': []}
    energy_batches = {'high': 0, 'low': 0}

    if has_dvfs:
        set_dvfs('high')
        time.sleep(0.3)
    elif fixed_dvfs:
        set_dvfs(fixed_dvfs)
        time.sleep(0.3)
    baseline_freq = measure_freq_est(ext)
    print(f"  baseline freq_est = {baseline_freq:.2f}", flush=True)

    prev_hw_vec = None
    prev_action = None
    phase2_started = False

    for ep in range(epochs):
        is_phase2 = (ep >= PHASE1_EPOCHS) and model_controls_dvfs and model.use_effort
        # Boost effort head LR at Phase 2 start
        if is_phase2 and not phase2_started:
            phase2_started = True
            opt.param_groups[1]['lr'] = 3e-3  # 3x boost for effort head
        tot_loss, correct, total = 0, 0, 0

        for imgs, digits in loader:
            imgs, digits = imgs.to(DEVICE), digits.to(DEVICE)

            # --- DVFS + demand ---
            if fixed_dvfs:
                dvfs_state = fixed_dvfs
                sclk_high = (dvfs_state == 'high')
                current_demand = sclk_high
            elif is_phase2:
                current_demand = random.random() > 0.5
            else:
                if bn % SWITCH_EVERY == 0:
                    sclk_high = (bn // SWITCH_EVERY) % 2 == 0
                    dvfs_state = 'high' if sclk_high else 'low'
                    set_dvfs(dvfs_state)
                current_demand = sclk_high

            # MODE cycling: random (decoupled from DVFS)
            if bn % SWITCH_EVERY == 0:
                config_idx = random.randint(0, K_MODES - 1)
            cur_mode = MODE_SET[config_idx]
            cur_chain = [1, 4, 8, 16][min(config_idx // 2, 3)]

            # Read ALL sensors
            cur_freq = measure_freq_est(ext, n_trials=2)
            hw_info = read_hw(ext)
            cur_mod_factor = compute_mod_factor(cur_freq, baseline_freq)

            hw_vec = make_hw_vector(hw_info, cur_mode, dvfs_state, cur_freq, 0,
                                    baseline_freq, 0, cur_mod_factor)
            hw_t = torch.tensor([hw_vec] * BS, dtype=torch.float32, device=DEVICE)

            labels = make_labels(digits, current_demand)
            demand_cue = torch.full((BS,), float(current_demand), device=DEVICE)

            out = model(imgs, hw_vector=hw_t,
                       mode_byte=cur_mode, chain_depth=cur_chain, priority=0,
                       demand_cue=demand_cue, sleep_amt=0, mod_factor=cur_mod_factor)

            if torch.isnan(out['logits']).any():
                bn += 1
                continue

            # -- Losses (z2061 pattern) --
            task_loss = F.cross_entropy(out['logits'], labels)

            self_target = torch.full((BS, 1), float(sclk_high), device=DEVICE)
            self_loss = F.binary_cross_entropy_with_logits(out['self_pred'], self_target)

            effort_loss = torch.tensor(0.0, device=DEVICE)
            if out['effort'] is not None:
                effort_target = torch.full((BS, 1), float(current_demand), device=DEVICE)
                effort_loss = F.binary_cross_entropy(out['effort'], effort_target)

            g = out['gate']
            if current_demand:
                homeo_loss = ((1 - g) ** 2).mean()
            else:
                homeo_loss = (g ** 2).mean()

            wm_loss = torch.tensor(0.0, device=DEVICE)
            if model.use_world_model and prev_hw_vec is not None:
                freq_target = torch.full((BS,), cur_freq / max(baseline_freq, 1.0),
                                         device=DEVICE)
                prev_hw_t = torch.tensor([prev_hw_vec] * BS, dtype=torch.float32, device=DEVICE)
                pa = prev_action
                prev_act = torch.tensor(
                    [[pa[0], pa[1], pa[2], pa[3], pa[4]]] * BS, device=DEVICE)
                wm_in = torch.cat([prev_hw_t, prev_act], dim=-1)
                wm_p = model.world_model(wm_in).squeeze(-1)
                wm_loss = F.mse_loss(wm_p, freq_target)

            mode_ent = torch.tensor(0.0, device=DEVICE)
            if out['mode_logits'] is not None:
                mp = F.softmax(out['mode_logits'], dim=-1)
                mode_ent = -(mp * torch.log(mp + 1e-8)).sum(-1).mean()

            effort_w = 0.3 if is_phase2 else 0.1
            loss = (task_loss
                    + 0.1 * self_loss
                    + effort_w * effort_loss
                    + 0.05 * homeo_loss
                    + 0.05 * wm_loss.clamp(max=5.0)
                    - 0.05 * mode_ent)

            if torch.isnan(loss):
                bn += 1
                continue

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            # Phase 2: effort controls DVFS
            if is_phase2 and out['effort'] is not None:
                effort_val = out['effort'].mean().item()
                new_sclk_high = effort_val > 0.5
                if new_sclk_high != sclk_high:
                    dvfs_state = 'high' if new_sclk_high else 'low'
                    set_dvfs(dvfs_state)
                sclk_high = new_sclk_high

            energy_batches['high' if sclk_high else 'low'] += 1
            tot_loss += loss.item()
            pred = out['logits'].argmax(1)
            correct += (pred == labels).sum().item()
            total += BS

            sm_pred = (torch.sigmoid(out['self_pred']) > 0.5).float()
            log['sm_acc'].append((sm_pred.squeeze() == self_target.squeeze()).float().mean().item())
            log['gate'].append(g.mean().item())
            log['freq'].append(cur_freq)
            if wm_loss.item() > 0:
                log['wm_err'].append(wm_loss.item())
            if out['effort'] is not None:
                log['effort'].append(out['effort'].mean().item())

            # Store prev action for world model
            rnd = (cur_mode & 0x0F) / 15.0
            dnrm = ((cur_mode >> 4) & 0x0F) / 15.0
            prev_hw_vec = hw_vec
            prev_action = (rnd, dnrm, out['chosen_priority'] / 3.0,
                          out['chosen_sleep'] / 3.0,
                          (cur_mod_factor - 0.9) / 0.2)
            bn += 1

        if ep % 5 == 0 or ep == epochs - 1:
            g_m = np.mean(log['gate'][-50:]) if log['gate'] else 0
            sm_m = np.mean(log['sm_acc'][-50:]) if log['sm_acc'] else 0
            wm_m = np.mean(log['wm_err'][-20:]) if log['wm_err'] else 0
            eff_m = np.mean(log['effort'][-50:]) if log['effort'] else 0
            phase = "P2" if is_phase2 else "P1"
            print(f"  [{name}] Ep {ep} ({phase}): loss={tot_loss/max(len(loader),1):.4f} "
                  f"acc={correct/max(total,1):.4f} gate={g_m:.3f} sm={sm_m:.3f} "
                  f"effort={eff_m:.3f} freq={np.mean(log['freq'][-50:]):.1f} wm={wm_m:.4f}",
                  flush=True)
        sched.step()

    if has_dvfs:
        set_dvfs('auto')
    return {'baseline': baseline_freq, 'energy_batches': energy_batches}


# === EVAL ===
def evaluate(model, ext, loader, baseline_freq=1.0, scramble=False,
             fixed_dvfs=None, model_controlled=False,
             random_demand_labels=False):
    model.eval()
    all_correct, all_gates, all_true_dvfs = [], [], []
    all_effort_correct, all_effort_vals, all_next_sclk = [], [], []
    all_freq, all_prio, all_sleep = [], [], []
    all_attn_h, all_attn_l = [], []
    energy_batches = {'high': 0, 'low': 0}
    bn, config_idx, dvfs_state = 0, 0, 'high'
    sclk_high = True
    has_dvfs = fixed_dvfs is None

    if has_dvfs:
        set_dvfs('high')
        time.sleep(0.2)
    elif fixed_dvfs:
        set_dvfs(fixed_dvfs)
        time.sleep(0.2)

    with torch.no_grad():
        for imgs, digits in loader:
            imgs, digits = imgs.to(DEVICE), digits.to(DEVICE)

            if fixed_dvfs:
                dvfs_state = fixed_dvfs
                sclk_high = (dvfs_state == 'high')
                current_demand = random.random() > 0.5 if random_demand_labels else sclk_high
            elif model_controlled and model.use_effort:
                current_demand = random.random() > 0.5
            else:
                if bn % SWITCH_EVERY == 0:
                    sclk_high = (bn // SWITCH_EVERY) % 2 == 0
                    dvfs_state = 'high' if sclk_high else 'low'
                    set_dvfs(dvfs_state)
                current_demand = sclk_high

            if bn % SWITCH_EVERY == 0:
                config_idx = random.randint(0, K_MODES - 1)
            cur_mode = MODE_SET[config_idx]
            cur_chain = [1, 4, 8, 16][min(config_idx // 2, 3)]

            cur_freq = measure_freq_est(ext, n_trials=1)
            hw_info = read_hw(ext)
            cur_mod_factor = compute_mod_factor(cur_freq, baseline_freq)
            hw_vec = make_hw_vector(hw_info, cur_mode, dvfs_state, cur_freq, 0,
                                    baseline_freq, 0, cur_mod_factor)
            if scramble:
                hw_vec = [1.0 - hw_vec[i] if i < SENSOR_DIM else hw_vec[i]
                          for i in range(len(hw_vec))]
            hw_t = torch.tensor([hw_vec] * BS, dtype=torch.float32, device=DEVICE)
            labels = make_labels(digits, current_demand)
            demand_cue = torch.full((BS,), float(current_demand), device=DEVICE)

            out = model(imgs, hw_vector=hw_t, mode_byte=cur_mode,
                       chain_depth=cur_chain, demand_cue=demand_cue,
                       sleep_amt=0, mod_factor=cur_mod_factor)

            pred = out['logits'].argmax(1)
            all_correct.extend((pred == labels).cpu().tolist())
            all_gates.extend(out['gate'].squeeze(-1).cpu().tolist())
            all_true_dvfs.extend([int(sclk_high)] * BS)
            all_freq.append(cur_freq)
            all_prio.append(out['chosen_priority'])
            all_sleep.append(out['chosen_sleep'])

            if out['effort'] is not None:
                ev = out['effort'].mean().item()
                all_effort_correct.append(float((ev > 0.5) == current_demand))
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

            if hasattr(model, 'hw_attn') and model.hw_attn._last_weights is not None:
                w = model.hw_attn._last_weights.cpu().numpy().mean(axis=(0, 1))
                (all_attn_h if dvfs_state == 'high' else all_attn_l).append(w)

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
        'gate_p': gate_p, 'mean_freq': float(np.mean(all_freq)),
        'mean_priority': float(np.mean(all_prio)),
        'priority_unique': len(set(all_prio)),
        'sleep_unique': len(set(all_sleep)),
        'attn_kl': attn_kl, 'effort_acc': effort_acc,
        'temporal_r': temporal_r, 'high_sclk_frac': high_frac,
    }


def ablate_self_model(m):
    if hasattr(m, 'self_model'):
        for p in m.self_model.parameters(): p.data.zero_()

def ablate_effort(m):
    for attr in ('effort_head', 'demand_proj'):
        if hasattr(m, attr):
            for p in getattr(m, attr).parameters(): p.data.zero_()


# === SANITY CHECKS ===
def sanity_checks(ext):
    print("\n--- Hardware sensor/actuator checks ---", flush=True)
    x = torch.randn(BS, 128, device=DEVICE)
    w = torch.randn(64, 128, device=DEVICE) * 0.1
    b = torch.zeros(64, device=DEVICE)

    # MODE effect
    r0 = ext.hw_matmul(x, w, b, 0x000, 8, 0, 0, 1.0)
    rF = ext.hw_matmul(x, w, b, 0x00F, 8, 0, 0, 1.0)
    torch.cuda.synchronize()
    print(f"  MODE 0x000 vs 0x00F: max|d|={((rF[0]-r0[0]).abs().max().item()):.6f}", flush=True)

    # mod_factor effect
    r_lo = ext.hw_matmul(x, w, b, 0x000, 8, 0, 0, 0.95)
    r_hi = ext.hw_matmul(x, w, b, 0x000, 8, 0, 0, 1.05)
    torch.cuda.synchronize()
    print(f"  mod_factor 0.95 vs 1.05: max|d|={((r_hi[0]-r_lo[0]).abs().max().item()):.6f}", flush=True)

    # Sleep effect on cycles
    for sl in range(4):
        r = ext.hw_matmul(x, w, b, 0x000, 8, 0, sl, 1.0)
        torch.cuda.synchronize()
        cyc = r[2].cpu().numpy()
        valid = cyc[cyc > 0]
        if len(valid) > 0:
            print(f"  s_sleep {sl}: cycles={np.median(valid):.0f}", flush=True)

    # All sensors
    res = ext.probe(64)
    torch.cuda.synchronize()
    print(f"\n--- All sensor readings ---", flush=True)
    print(f"  WGP unique: {len(set(res[0].cpu().tolist()))} vals: {sorted(set(res[0].cpu().tolist()))[:5]}", flush=True)
    print(f"  STATUS sample: 0x{int(np.median(res[2].cpu().numpy())):05x}", flush=True)
    print(f"  HW_ID2 sample: 0x{int(np.median(res[3].cpu().numpy())):08x}", flush=True)
    print(f"  TRAPSTS sample: 0x{int(np.median(res[4].cpu().numpy())):04x}", flush=True)
    print(f"  GPR_ALLOC sample: 0x{int(np.median(res[5].cpu().numpy())):04x}", flush=True)

    # DVFS
    print(f"\n--- DVFS freq_est ---", flush=True)
    has_dvfs = set_dvfs('high')
    if has_dvfs:
        time.sleep(0.5)
        freq_h = measure_freq_est(ext)
        set_dvfs('low')
        time.sleep(0.5)
        freq_l = measure_freq_est(ext)
        set_dvfs('auto')
        ratio = freq_h / max(freq_l, 1e-6)
        mf_h = compute_mod_factor(freq_h, freq_h)
        mf_l = compute_mod_factor(freq_l, freq_h)
        print(f"  High: {freq_h:.1f} (mod={mf_h:.4f})  Low: {freq_l:.1f} (mod={mf_l:.4f})  Ratio: {ratio:.2f}x", flush=True)
    else:
        print("  DVFS not available!", flush=True)


# === MAIN ===
def main():
    global _EXT
    t0 = time.time()

    print("=" * 70, flush=True)
    print("z2073: Attention + Closed-Loop + ALL Sensors/Actuators", flush=True)
    print("=" * 70, flush=True)
    print(f"Sensors ({SENSOR_DIM}): freq_est, cycle_var, wgp, status, hwid2, trapsts, gpr, wall_ratio", flush=True)
    print(f"Actuators (6): MODE[9:0], s_setprio, s_sleep, DVFS(effort), mod_factor, chain_depth", flush=True)
    print(f"Training: Phase 1 ({PHASE1_EPOCHS}ep) perception, Phase 2 ({PHASE2_EPOCHS}ep) closed-loop (effort_w=0.3)", flush=True)
    print(flush=True)

    print("Compiling HIP kernels...", flush=True)
    ext = load_inline(name='z2073_full', cpp_sources=CPP_SRC, cuda_sources=HIP_SRC,
                      functions=['hw_matmul', 'probe'],
                      extra_cuda_cflags=['-O2', '--offload-arch=gfx1100'],
                      verbose=False)
    _EXT = ext
    print("  Done.", flush=True)

    sanity_checks(ext)
    train_loader, test_loader = get_data()

    # A: FULL
    print(f"\n{'='*60}\nA: FULL (all components, 2-phase)\n{'='*60}", flush=True)
    model_A = Z2073Model().to(DEVICE)
    info_A = train_model(model_A, ext, train_loader, EPOCHS, 'A_full')
    baseline = info_A['baseline']
    m_A = evaluate(model_A, ext, test_loader, baseline, model_controlled=True)
    print(f"  A: acc={m_A['acc']:.4f} AUROC={m_A['sm_auroc']:.4f} "
          f"gate_h={m_A['gate_high']:.3f} gate_l={m_A['gate_low']:.3f} "
          f"effort={m_A['effort_acc']:.3f} temporal_r={m_A['temporal_r']:.3f}", flush=True)

    # B: BLIND
    print(f"\n{'='*60}\nB: BLIND\n{'='*60}", flush=True)
    model_B = Z2073Model(use_self_model=False, use_gate=False, use_deep=False,
                          use_attention=False, use_world_model=False,
                          use_effort=False).to(DEVICE)
    train_model(model_B, ext, train_loader, EPOCHS, 'B_blind', model_controls_dvfs=False)
    m_B = evaluate(model_B, ext, test_loader, baseline)
    print(f"  B: acc={m_B['acc']:.4f}", flush=True)

    # C: SCRAMBLED
    print(f"\n{'='*60}\nC: SCRAMBLED\n{'='*60}", flush=True)
    m_C = evaluate(model_A, ext, test_loader, baseline, scramble=True)
    print(f"  C: acc={m_C['acc']:.4f}", flush=True)

    # D: NO SELF-MODEL
    print(f"\n{'='*60}\nD: NO SELF-MODEL\n{'='*60}", flush=True)
    model_D = copy.deepcopy(model_A)
    ablate_self_model(model_D)
    m_D = evaluate(model_D, ext, test_loader, baseline, model_controlled=True)
    print(f"  D: acc={m_D['acc']:.4f}", flush=True)

    # E: NO GATE
    print(f"\n{'='*60}\nE: NO GATE\n{'='*60}", flush=True)
    model_E = copy.deepcopy(model_A)
    model_E.use_gate = False
    m_E = evaluate(model_E, ext, test_loader, baseline, model_controlled=True)
    print(f"  E: acc={m_E['acc']:.4f}", flush=True)

    # F: NO ATTENTION
    print(f"\n{'='*60}\nF: NO ATTENTION\n{'='*60}", flush=True)
    model_F = Z2073Model(use_attention=False).to(DEVICE)
    train_model(model_F, ext, train_loader, EPOCHS, 'F_no_attn')
    m_F = evaluate(model_F, ext, test_loader, baseline, model_controlled=True)
    print(f"  F: acc={m_F['acc']:.4f}", flush=True)

    # G: FIXED HIGH
    print(f"\n{'='*60}\nG: FIXED HIGH\n{'='*60}", flush=True)
    model_G = Z2073Model().to(DEVICE)
    train_model(model_G, ext, train_loader, EPOCHS, 'G_fixed', fixed_dvfs='high')
    m_G = evaluate(model_G, ext, test_loader, baseline, fixed_dvfs='high')
    print(f"  G: acc={m_G['acc']:.4f}", flush=True)

    # H: EFFORT ABLATION
    print(f"\n{'='*60}\nH: ABLATED EFFORT\n{'='*60}", flush=True)
    model_H = copy.deepcopy(model_A)
    ablate_effort(model_H)
    m_H = evaluate(model_H, ext, test_loader, baseline,
                   fixed_dvfs='high', random_demand_labels=True)
    print(f"  H: acc={m_H['acc']:.4f}", flush=True)

    elapsed = time.time() - t0

    # === TESTS (16) ===
    print(f"\n{'='*70}\nTEST RESULTS\n{'='*70}", flush=True)
    tests = {}

    tests['T1_accuracy'] = {
        'verdict': 'PASS' if m_A['acc'] > 0.60 else 'FAIL',
        'val': f"A={m_A['acc']*100:.1f}% > 60%"}

    tests['T2_self_model'] = {
        'verdict': 'PASS' if m_A['sm_auroc'] > 0.75 else 'FAIL',
        'val': f"AUROC={m_A['sm_auroc']:.4f} > 0.75"}

    gate_sep = abs(m_A['gate_high'] - m_A['gate_low'])
    tests['T3_gate_sep'] = {
        'verdict': 'PASS' if gate_sep > 0.15 else 'FAIL',
        'val': f"|h-l|={gate_sep:.3f} > 0.15"}

    tests['T4_gate_stat'] = {
        'verdict': 'PASS' if m_A['gate_p'] < 0.01 else 'FAIL',
        'val': f"p={m_A['gate_p']:.6f} < 0.01"}

    gap_AB = m_A['acc'] - m_B['acc']
    tests['T5_embodiment'] = {
        'verdict': 'PASS' if gap_AB > 0.08 else 'FAIL',
        'val': f"A-B={gap_AB*100:.1f}pp > 8pp"}

    gap_AC = m_A['acc'] - m_C['acc']
    tests['T6_scramble'] = {
        'verdict': 'PASS' if gap_AC > 0.05 else 'FAIL',
        'val': f"A-C={gap_AC*100:.1f}pp > 5pp"}

    gap_AD = m_A['acc'] - m_D['acc']
    tests['T7_sm_causal'] = {
        'verdict': 'PASS' if gap_AD > 0.10 else 'FAIL',
        'val': f"A-D={gap_AD*100:.1f}pp > 10pp"}

    gap_AE = m_A['acc'] - m_E['acc']
    tests['T8_gate_causal'] = {
        'verdict': 'PASS' if gap_AE > 0.08 else 'FAIL',
        'val': f"A-E={gap_AE*100:.1f}pp > 8pp"}

    gap_AH = m_A['acc'] - m_H['acc']
    tests['T9_effort_causal'] = {
        'verdict': 'PASS' if gap_AH > 0.10 else 'FAIL',
        'val': f"A-H={gap_AH*100:.1f}pp > 10pp"}

    tests['T10_effort_acc'] = {
        'verdict': 'PASS' if m_A['effort_acc'] > 0.80 else 'FAIL',
        'val': f"effort={m_A['effort_acc']*100:.1f}% > 80%"}

    tests['T11_closed_loop'] = {
        'verdict': 'PASS' if abs(m_A['temporal_r']) > 0.50 else 'FAIL',
        'val': f"|r|={abs(m_A['temporal_r']):.4f} > 0.50"}

    tests['T12_attn_helps'] = {
        'verdict': 'PASS' if m_A['acc'] > m_F['acc'] else 'FAIL',
        'val': f"A={m_A['acc']*100:.1f}% > F={m_F['acc']*100:.1f}%"}

    tests['T13_attn_varies'] = {
        'verdict': 'PASS' if m_A['attn_kl'] > 0.01 else 'FAIL',
        'val': f"KL={m_A['attn_kl']:.4f} > 0.01"}

    gap_AG = m_A['acc'] - m_G['acc']
    tests['T14_adaptive'] = {
        'verdict': 'PASS' if gap_AG > 0.05 else 'FAIL',
        'val': f"A-G={gap_AG*100:.1f}pp > 5pp"}

    tests['T15_energy'] = {
        'verdict': 'PASS' if m_A['high_sclk_frac'] < 0.75 else 'FAIL',
        'val': f"hi_frac={m_A['high_sclk_frac']:.3f} < 0.75"}

    best_abl = max(m_C['acc'], m_D['acc'], m_E['acc'], m_H['acc'])
    tests['T16_full_best'] = {
        'verdict': 'PASS' if m_A['acc'] > best_abl else 'FAIL',
        'val': f"A={m_A['acc']*100:.1f}% > max_abl={best_abl*100:.1f}%"}

    pass_count = sum(1 for t in tests.values() if t['verdict'] == 'PASS')
    verdict = f"{pass_count}/{len(tests)} PASS"

    for tn, r in tests.items():
        print(f"  {r['verdict']:4s} | {tn}: {r['val']}", flush=True)
    print(f"\n  VERDICT: {verdict}", flush=True)

    print(f"\n  Ablation summary:", flush=True)
    for lbl, m in [('A full', m_A), ('B blind', m_B), ('C scrambled', m_C),
                    ('D no-sm', m_D), ('E no-gate', m_E), ('F no-attn', m_F),
                    ('G fixed-hi', m_G), ('H no-effort', m_H)]:
        gap = m['acc'] - m_A['acc']
        ex = f" AUROC={m.get('sm_auroc',0):.3f}" if 'sm_auroc' in m else ""
        if m.get('effort_acc', 0) > 0: ex += f" eff={m['effort_acc']:.3f}"
        print(f"    {lbl:14s}: {m['acc']*100:.1f}% ({gap*100:+.1f}pp){ex}", flush=True)

    # SAVE
    results = {
        'experiment': 'z2073_attention_embodiment_v2',
        'innovations': [
            'ALL 8 sensors: freq_est, cycle_var, wgp, status, hwid2, trapsts, gpr_alloc, wall_ratio',
            'ALL 6 actuators: MODE[9:0], s_setprio, s_sleep, DVFS(effort), mod_factor, chain_depth',
            'z2060 homeostatic loss + exclusive specialization',
            'z2061 effort head + 2-phase training + demand-driven labels',
            'GWT attention workspace (features + hw token)',
            'mod_factor: continuous freq_est->math coupling in ISA kernel',
            'World model: prev_hw + prev_action -> predict next freq_est',
        ],
        'accuracies': {k: round(v, 4) for k, v in [
            ('A', m_A['acc']), ('B', m_B['acc']), ('C', m_C['acc']),
            ('D', m_D['acc']), ('E', m_E['acc']), ('F', m_F['acc']),
            ('G', m_G['acc']), ('H', m_H['acc'])]},
        'self_model_auroc': round(m_A['sm_auroc'], 4),
        'gate': {'high': round(m_A['gate_high'], 4), 'low': round(m_A['gate_low'], 4),
                 'sep': round(gate_sep, 4), 'p': round(m_A['gate_p'], 6)},
        'effort': {'acc': round(m_A['effort_acc'], 4), 'temporal_r': round(m_A['temporal_r'], 4)},
        'energy': {'high_sclk_frac': round(m_A['high_sclk_frac'], 4)},
        'attn_kl': round(m_A['attn_kl'], 4),
        'tests': tests, 'verdict': verdict,
        'pass_count': pass_count, 'total_tests': len(tests),
        'elapsed_s': round(elapsed),
    }
    out_path = 'results/z2073_attention_embodiment.json'
    os.makedirs('results', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)
    print(f"Elapsed: {elapsed:.0f}s", flush=True)


if __name__ == '__main__':
    main()
