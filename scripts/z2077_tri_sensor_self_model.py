#!/usr/bin/env python3
"""z2077: Tri-Sensor Self-Model — Pure Math + Physical Metrics.

v2: Fixed from v1 (5/15) — dual-stream architecture with LayerNorm
  - v1 FAILED because 20 noisy metrics dims drowned 5 useful delta dims
  - v2 uses SEPARATE processing streams + LayerNorm for scale normalization
  - Self-model can learn to ignore metrics if they carry no signal

Extends z2076 (12/12 PASS) by adding gpu_metrics as "somatic afferents":
  - Delta (5 dims): HW_kernel - SW_linear = arithmetic fingerprint [z2076]
  - Metrics (20 dims): gpu_metrics blob fields = physical state
    (temperatures, power, clocks, activity, voltage proxies)

CORE HYPOTHESIS: The model can learn to use BOTH instantaneous arithmetic
signals (delta) AND slower physical signals (power/temp/voltage) for
self-modeling. If so, the model has "interoception" at two timescales:
  - Fast: arithmetic effect (microseconds, within-kernel)
  - Slow: physical state (milliseconds, SMU sampling)

FALSIFICATION: If T13 passes (metrics_ablate hurts accuracy) then the
model genuinely uses physical signals. If T14 passes (delta_ablate hurts),
both channels carry causally necessary information.

KEY FINDING FROM v1: Under fixed DVFS, gpu_metrics DON'T vary between
ISA personalities (0/20 fields differ). The depth of sensor must match
the depth of actuator. This v2 tests whether the model can gracefully
ignore non-informative sensors without performance degradation.

6 ISA ACTUATORS (same as z2076, all sub-firmware):
  1. MODE[3:0] rounding  — 4 modes: nearest, +inf, -inf, zero
  2. MODE[7:4] denorm    — 4 modes: flush-all, keep-f32, keep-f16, keep-all
  3. chain_depth          — 4 levels: 1, 4, 8, 16
  4. v_perm_b32 seed      — 4 patterns: identity, reverse, swap, rotate
  5. s_sleep delay        — 4 levels: 0, 1, 2, 3
  6. s_setprio priority   — 4 levels: 0, 1, 2, 3

ENERGY TRACKING: Reads u32@0x70 from gpu_metrics (socket_power in mW)
every batch to compute accuracy-per-joule.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import os, sys, json, time, copy, struct, random, numpy as np
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score
from scipy import stats

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')
os.environ.setdefault('PYTORCH_ROCM_ARCH', 'gfx1100')
from torch.utils.cpp_extension import load_inline

DEVICE = 'cuda'
BS = 256
EPOCHS = 20
SWITCH_EVERY = 8
N_CLASSES = 10

# Sensor dimensions
DELTA_DIM = 5       # output delta features
METRICS_DIM = 20    # gpu_metrics features
SENSOR_DIM = DELTA_DIM + METRICS_DIM  # 25 total

# Actuator codes (same as z2076)
ROUND_CODES = [0x00, 0x05, 0x0A, 0x0F]
DENORM_CODES = [0x00, 0x30, 0xC0, 0xF0]
CHAIN_DEPTHS = [1, 4, 8, 16]
PERM_PATTERNS = [0x03020100, 0x00010203, 0x02030001, 0x01000302]

# gpu_metrics field offsets (empirically mapped on gfx1151, v8.1, 264 bytes)
# Each entry: (offset, size_bytes, name, scale_factor)
# scale_factor converts raw value to normalized float (~0-1 range)
METRICS_FIELDS = [
    (0x04, 2, 'temp_gfx',       1/10000.0),  # centideg → ~0.6-0.9
    (0x06, 2, 'temp_soc',       1/10000.0),
    (0x2a, 2, 'gfx_activity',   1/100.0),     # percent → 0-1
    (0x3e, 2, 'throttle_a',     1/100.0),
    (0x48, 2, 'current_proxy',  1/100.0),     # spikes under load
    (0x5e, 2, 'voltage_proxy',  1/5000.0),    # mV-scale
    (0x60, 2, 'current_gfx',    1/10000.0),
    (0x70, 4, 'socket_power',   1/200000.0),  # mW → ~0.3-0.6
    (0x78, 4, 'socket_power2',  1/200000.0),
    (0x7c, 2, 'gfx_power',     1/50000.0),   # mW → ~0.3-0.9
    (0x84, 2, 'power_proxy',    1/20000.0),
    (0x88, 2, 'mem_activity',   1/1000.0),
    (0x92, 2, 'compute_load',   1/3000.0),
    (0xa8, 2, 'energy_proxy',   1/60000.0),
    (0xae, 2, 'gfxclk',        1/3000.0),    # MHz → ~0.6-1.0
    (0xb0, 2, 'socclk',        1/2000.0),
    (0xb4, 2, 'clk_c',         1/2000.0),
    (0xb6, 2, 'mclk',          1/2000.0),
    (0xde, 2, 'clk_spread',    1/6000.0),
    (0xe0, 2, 'gfxclk_max',    1/3000.0),
]

assert len(METRICS_FIELDS) == METRICS_DIM, \
    f"Expected {METRICS_DIM} metrics fields, got {len(METRICS_FIELDS)}"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# gpu_metrics reader
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GPU_METRICS_PATH = None

def find_gpu_metrics():
    """Auto-detect gpu_metrics path."""
    global GPU_METRICS_PATH
    import glob
    for p in glob.glob('/sys/class/drm/card*/device/gpu_metrics'):
        data = open(p, 'rb').read()
        if len(data) >= 200:
            GPU_METRICS_PATH = p
            return p
    return None

def read_gpu_metrics():
    """Read gpu_metrics blob and extract selected fields as a float tensor."""
    if GPU_METRICS_PATH is None:
        return torch.zeros(METRICS_DIM, device=DEVICE)
    try:
        data = open(GPU_METRICS_PATH, 'rb').read()
        vals = []
        for offset, size, name, scale in METRICS_FIELDS:
            if offset + size <= len(data):
                if size == 2:
                    raw = struct.unpack_from('<H', data, offset)[0]
                elif size == 4:
                    raw = struct.unpack_from('<I', data, offset)[0]
                else:
                    raw = 0
                vals.append(float(raw) * scale)
            else:
                vals.append(0.0)
        return torch.tensor(vals, device=DEVICE, dtype=torch.float32)
    except Exception:
        return torch.zeros(METRICS_DIM, device=DEVICE)

def read_socket_power_mw():
    """Read socket power in milliwatts from gpu_metrics (for energy tracking)."""
    if GPU_METRICS_PATH is None:
        return 0.0
    try:
        data = open(GPU_METRICS_PATH, 'rb').read()
        if len(data) >= 0x74:
            return float(struct.unpack_from('<I', data, 0x70)[0])
        return 0.0
    except Exception:
        return 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HIP KERNEL: same fp16mix matmul with 6 ISA actuators (from z2076)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HIP_SRC = r'''
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>

#define TILE 16

__global__ void math_kernel(
    const float* __restrict__ X, const float* __restrict__ W,
    const float* __restrict__ B, float* __restrict__ Y,
    int M, int K, int N,
    unsigned int mode_byte, int chain_depth,
    unsigned int perm_pattern, int sleep_amt, int priority)
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

    // v_perm_b32 on stochastic rounding seed
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

    // === TILED MATMUL with fp16mix ===
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

            // Physics-seeded stochastic rounding
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

    // Restore MODE defaults
    unsigned int z = __builtin_amdgcn_readfirstlane(0xF0u);
    asm volatile("s_setreg_b32 hwreg(1, 0, 8), %0" : : "s"(z));
    asm volatile("s_setprio 0");
}

torch::Tensor math_forward(torch::Tensor X, torch::Tensor W, torch::Tensor B,
                            int mode_byte, int chain_depth, int perm_pattern,
                            int sleep_amt, int priority) {
    int M = X.size(0), K = X.size(1), N = W.size(0);
    auto Y = torch::zeros({M, N}, X.options());
    dim3 threads(TILE, TILE);
    dim3 blocks((unsigned int)((N + TILE - 1) / TILE),
                (unsigned int)((M + TILE - 1) / TILE));
    math_kernel<<<blocks, threads>>>(
        X.data_ptr<float>(), W.data_ptr<float>(), B.data_ptr<float>(),
        Y.data_ptr<float>(), M, K, N,
        (unsigned int)(mode_byte & 0x3FF), chain_depth,
        (unsigned int)perm_pattern, sleep_amt, priority);
    return Y;
}
'''

CPP_SRC = r'''
#include <torch/extension.h>
torch::Tensor math_forward(torch::Tensor, torch::Tensor, torch::Tensor,
                            int, int, int, int, int);
'''

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Custom autograd
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_EXT = None

class MathLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, mode_byte, chain_depth, perm_pattern, sleep_amt, priority):
        ctx.save_for_backward(x, w)
        y = _EXT.math_forward(x.contiguous(), w.contiguous(), b.contiguous(),
                               int(mode_byte), int(chain_depth), int(perm_pattern),
                               int(sleep_amt), int(priority))
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, w = ctx.saved_tensors
        return grad_out @ w, grad_out.t() @ x, grad_out.sum(0), None, None, None, None, None


class MathLinear(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_f))

    def forward(self, x, mode_byte=0xF0, chain_depth=1, perm_pattern=0x03020100,
                sleep_amt=0, priority=0):
        return MathLinearFn.apply(x, self.weight, self.bias,
                                   mode_byte, chain_depth, perm_pattern,
                                   sleep_amt, priority)

    def soft_forward(self, x):
        """Standard PyTorch linear (for computing delta)."""
        return F.linear(x, self.weight, self.bias)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SENSOR COMPUTATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_delta_vector(deep_out, soft_out):
    """Output delta = HW_kernel - SW_linear. 5-dim arithmetic fingerprint."""
    delta = (deep_out - soft_out).detach()
    d_mean = delta.mean().item()
    d_std = delta.std().item()
    d_abs_max = delta.abs().max().item()
    d_pos_frac = (delta > 0).float().mean().item()
    d_norm = delta.norm().item() / max(delta.numel(), 1)
    return torch.tensor([d_mean, d_std, d_abs_max, d_pos_frac, d_norm],
                         device=deep_out.device)


def compute_hw_vector(deep_out, soft_out, metrics_vec=None):
    """Full 25-dim sensor: [delta(5) | metrics(20)]."""
    delta = compute_delta_vector(deep_out, soft_out)
    if metrics_vec is None:
        metrics_vec = read_gpu_metrics()
    return torch.cat([delta, metrics_vec])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Actuator configs (same as z2076)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PERSONALITY_A = {  # Precise: small delta, less work
    'round_idx': 0, 'denorm_idx': 3, 'chain_idx': 0,
    'perm_idx': 0, 'sleep_idx': 0, 'prio_idx': 0,
}
PERSONALITY_B = {  # Lossy: large delta, more work (16x chain)
    'round_idx': 3, 'denorm_idx': 0, 'chain_idx': 3,
    'perm_idx': 1, 'sleep_idx': 3, 'prio_idx': 3,
}

def config_to_kernel_args(cfg):
    mode = DENORM_CODES[cfg['denorm_idx']] | ROUND_CODES[cfg['round_idx']]
    return {
        'mode_byte': mode,
        'chain_depth': CHAIN_DEPTHS[cfg['chain_idx']],
        'perm_pattern': PERM_PATTERNS[cfg['perm_idx']],
        'sleep_amt': cfg['sleep_idx'],
        'priority': cfg['prio_idx'],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TriSensorModel(nn.Module):
    def __init__(self, sensor_dim=SENSOR_DIM, use_hw=True,
                 use_self_model=True, use_gate=True):
        super().__init__()
        self.sensor_dim = sensor_dim
        self.use_hw = use_hw
        self.use_self_model = use_self_model
        self.use_gate = use_gate

        # MNIST encoder → 128-dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU())

        # Deep path: ISA matmul
        self.deep_fc = MathLinear(128, 64)
        self.head_A = nn.Sequential(nn.ReLU(), nn.Linear(64, N_CLASSES))

        # Light path: standard software
        self.light_fc = nn.Linear(128, 64)
        self.head_B = nn.Sequential(nn.ReLU(), nn.Linear(64, N_CLASSES))

        # Self-model: DUAL-STREAM — process delta and metrics separately
        # This prevents noisy metrics from drowning useful delta signal
        if use_self_model:
            delta_in = min(sensor_dim, DELTA_DIM)
            metrics_in = max(0, sensor_dim - DELTA_DIM)
            # Delta stream (fast, ISA-level signal)
            self.delta_norm = nn.LayerNorm(delta_in)
            self.delta_stream = nn.Sequential(
                nn.Linear(delta_in, 16), nn.ReLU())
            # Metrics stream (slow, physical signal)
            if metrics_in > 0:
                self.metrics_norm = nn.LayerNorm(metrics_in)
                self.metrics_stream = nn.Sequential(
                    nn.Linear(metrics_in, 16), nn.ReLU())
                fuse_dim = 32
            else:
                self.metrics_norm = None
                self.metrics_stream = None
                fuse_dim = 16
            # Fusion → personality prediction
            self.self_model = nn.Sequential(
                nn.Linear(fuse_dim, 16), nn.ReLU(),
                nn.Linear(16, 1))

        # Gate: from self-model prediction
        if use_gate:
            self.gate_net = nn.Sequential(
                nn.Linear(1, 16), nn.ReLU(),
                nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, x, hw_vector=None, mode_byte=0xF0, chain_depth=1,
                perm_pattern=0x03020100, sleep_amt=0, priority=0):
        features = self.encoder(x)

        # ISA deep path
        deep_out = self.deep_fc(features, mode_byte, chain_depth,
                                 perm_pattern, sleep_amt, priority)
        logits_A = self.head_A(deep_out)

        # SW light path
        soft_out = self.deep_fc.soft_forward(features)
        light_out = F.relu(self.light_fc(features))
        logits_B = self.head_B(light_out)

        # Compute sensor vector (delta + metrics)
        if hw_vector is None and self.use_hw:
            hw_vector = compute_hw_vector(deep_out, soft_out)

        # Self-model + gate (dual-stream processing)
        self_pred = None
        if self.use_self_model and hw_vector is not None:
            hw_in = hw_vector.unsqueeze(0).expand(x.shape[0], -1)
            # Split into delta and metrics streams
            delta_in = hw_in[:, :DELTA_DIM]
            delta_feat = self.delta_stream(self.delta_norm(delta_in))
            if self.metrics_stream is not None and hw_in.shape[1] > DELTA_DIM:
                metrics_in = hw_in[:, DELTA_DIM:]
                metrics_feat = self.metrics_stream(self.metrics_norm(metrics_in))
                fused = torch.cat([delta_feat, metrics_feat], dim=1)
            else:
                fused = delta_feat
            self_pred = self.self_model(fused)

        if self.use_gate and self_pred is not None:
            gate = self.gate_net(torch.sigmoid(self_pred))
        else:
            gate = torch.full((x.shape[0], 1), 0.5, device=x.device)

        logits = gate * logits_A + (1 - gate) * logits_B
        return {'logits': logits, 'logits_A': logits_A, 'logits_B': logits_B,
                'self_pred': self_pred, 'gate': gate, 'hw_vector': hw_vector}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA + LABELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_data():
    tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))])
    tr = datasets.MNIST('data', train=True, download=True, transform=tf)
    te = datasets.MNIST('data', train=False, transform=tf)
    return (torch.utils.data.DataLoader(tr, batch_size=BS, shuffle=True, drop_last=True),
            torch.utils.data.DataLoader(te, batch_size=BS, shuffle=False, drop_last=True))


def make_labels(labels, personality):
    if personality == 0:
        return labels
    else:
        return (9 - labels) % N_CLASSES


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENERGY TRACKER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class EnergyTracker:
    """Track energy consumption via gpu_metrics socket_power."""
    def __init__(self):
        self.samples = []
        self.total_joules = 0.0
        self.total_examples = 0
        self._last_time = None

    def sample(self, n_examples=0):
        now = time.time()
        power_mw = read_socket_power_mw()
        if self._last_time is not None and power_mw > 0:
            dt = now - self._last_time
            joules = (power_mw / 1000.0) * dt  # mW → W, * seconds = J
            self.total_joules += joules
            self.samples.append({'power_w': power_mw / 1000.0, 'dt': dt})
        self._last_time = now
        self.total_examples += n_examples

    def joules_per_example(self):
        if self.total_examples == 0:
            return 0.0
        return self.total_joules / self.total_examples

    def avg_power_w(self):
        if not self.samples:
            return 0.0
        return np.mean([s['power_w'] for s in self.samples])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAINING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_model(model, loader, epochs, name, track_energy=False):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    gate_vals, pers_states = [], []
    hw_vecs_A, hw_vecs_B = [], []
    metrics_vecs_A, metrics_vecs_B = [], []
    personality = 0
    bn = 0
    energy = EnergyTracker() if track_energy else None

    for ep in range(epochs):
        tot_loss, correct, total = 0., 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            if bn % SWITCH_EVERY == 0:
                personality = 1 - personality

            cfg = PERSONALITY_A if personality == 0 else PERSONALITY_B
            kargs = config_to_kernel_args(cfg)
            ex_labels = make_labels(labels, personality)

            out = model(imgs, **kargs)

            # Track energy
            if energy is not None:
                energy.sample(n_examples=BS)

            # Collect hw_vectors
            if out['hw_vector'] is not None:
                hv = out['hw_vector'].detach().cpu().numpy()
                if personality == 0:
                    hw_vecs_A.append(hv)
                    metrics_vecs_A.append(hv[DELTA_DIM:])
                else:
                    hw_vecs_B.append(hv)
                    metrics_vecs_B.append(hv[DELTA_DIM:])

            # Task loss
            task_loss = F.cross_entropy(out['logits'], ex_labels)

            # Self-model loss
            self_loss = torch.tensor(0., device=DEVICE)
            if out['self_pred'] is not None:
                self_target = torch.full((BS, 1), float(personality == 0), device=DEVICE)
                self_loss = F.binary_cross_entropy_with_logits(out['self_pred'], self_target)

            # Homeostatic loss
            g = out['gate']
            if personality == 0:
                homeo_loss = ((1 - g) ** 2).mean()
            else:
                homeo_loss = (g ** 2).mean()

            loss = task_loss + 0.1 * self_loss + 0.05 * homeo_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tot_loss += loss.item()
            correct += (out['logits'].argmax(1) == ex_labels).sum().item()
            total += BS
            gate_vals.append(g.mean().item())
            pers_states.append(personality)
            bn += 1

        if ep % 4 == 0 or ep == epochs - 1:
            ej = f" E={energy.joules_per_example():.4f}J/ex" if energy else ""
            print(f"  [{name}] Ep {ep}: loss={tot_loss/len(loader):.4f} "
                  f"acc={correct/total:.4f} gate={np.mean(gate_vals[-50:]):.3f}{ej}")

    return {'gate_vals': gate_vals, 'pers_states': pers_states,
            'hw_vecs_A': hw_vecs_A, 'hw_vecs_B': hw_vecs_B,
            'metrics_vecs_A': metrics_vecs_A, 'metrics_vecs_B': metrics_vecs_B,
            'energy': energy}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EVALUATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def evaluate(model, loader, name, math_override=None, hw_override=None,
             track_energy=False):
    model.eval()
    by_pers = {0: {'correct': 0, 'total': 0, 'gates': [], 'self_preds': [],
                    'labels': [], 'hw_vecs': []},
               1: {'correct': 0, 'total': 0, 'gates': [], 'self_preds': [],
                    'labels': [], 'hw_vecs': []}}
    personality = 0
    bn = 0
    energy = EnergyTracker() if track_energy else None

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            if bn % SWITCH_EVERY == 0:
                personality = 1 - personality

            if math_override is not None:
                cfg = math_override
            else:
                cfg = PERSONALITY_A if personality == 0 else PERSONALITY_B
            kargs = config_to_kernel_args(cfg)

            out = model(imgs, hw_vector=hw_override, **kargs)
            ex_labels = make_labels(labels, personality)

            if energy is not None:
                energy.sample(n_examples=BS)

            pred = out['logits'].argmax(1)
            by_pers[personality]['correct'] += (pred == ex_labels).sum().item()
            by_pers[personality]['total'] += BS
            by_pers[personality]['gates'].extend(out['gate'].squeeze().cpu().tolist())

            if out['hw_vector'] is not None:
                by_pers[personality]['hw_vecs'].append(out['hw_vector'].cpu().numpy())

            if out['self_pred'] is not None:
                by_pers[personality]['self_preds'].extend(
                    torch.sigmoid(out['self_pred']).squeeze().cpu().tolist())
                by_pers[personality]['labels'].extend([float(personality == 0)] * BS)
            bn += 1

    m = {}
    total_c = sum(r['correct'] for r in by_pers.values())
    total_n = sum(r['total'] for r in by_pers.values())
    m['acc'] = total_c / max(total_n, 1)
    for p in [0, 1]:
        r = by_pers[p]
        pk = 'A' if p == 0 else 'B'
        m[f'acc_{pk}'] = r['correct'] / max(r['total'], 1)
        m[f'gate_{pk}'] = float(np.mean(r['gates'])) if r['gates'] else 0.5

    all_p = by_pers[0]['self_preds'] + by_pers[1]['self_preds']
    all_l = by_pers[0]['labels'] + by_pers[1]['labels']
    if len(set(all_l)) > 1 and len(all_p) > 10:
        m['auroc'] = float(roc_auc_score(all_l, all_p))
    else:
        m['auroc'] = 0.5

    g_a, g_b = by_pers[0]['gates'], by_pers[1]['gates']
    if g_a and g_b and len(set(g_a + g_b)) > 1:
        _, pv = stats.ttest_ind(g_a, g_b)
        m['gate_p'] = float(pv)
    else:
        m['gate_p'] = 1.0

    # hw_vector stats (delta part)
    for pk_i, pk in [(0, 'A'), (1, 'B')]:
        vecs = by_pers[pk_i]['hw_vecs']
        if vecs:
            arr = np.array(vecs)
            m[f'delta_mean_{pk}'] = float(arr[:, 0].mean())
            m[f'delta_std_{pk}'] = float(arr[:, 1].mean())
            # metrics part: avg power signal
            if arr.shape[1] > DELTA_DIM + 7:
                m[f'power_{pk}'] = float(arr[:, DELTA_DIM + 7].mean())  # socket_power

    if energy is not None:
        m['joules_per_example'] = energy.joules_per_example()
        m['avg_power_w'] = energy.avg_power_w()

    return m


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    global _EXT

    print("=" * 70)
    print("z2077: Tri-Sensor Self-Model — Pure Math + Physical Metrics")
    print("=" * 70)
    print("Sensors: delta(5) + gpu_metrics(20) = 25-dim hw_vector")
    print("Actuators: 6 ISA (MODE/denorm/chain/perm/sleep/prio)")
    print("Hypothesis: model uses BOTH arithmetic + physical signals")
    print()

    t0 = time.time()

    # Auto-detect gpu_metrics
    mp = find_gpu_metrics()
    if mp:
        print(f"gpu_metrics: {mp}")
        test_m = read_gpu_metrics()
        print(f"  Sample metrics (20 dims): min={test_m.min():.4f} "
              f"max={test_m.max():.4f} nonzero={(test_m != 0).sum().item()}")
        pw = read_socket_power_mw()
        print(f"  Socket power: {pw:.0f} mW ({pw/1000:.1f} W)")
    else:
        print("WARNING: gpu_metrics not found — metrics dims will be zero")

    # Fix DVFS to 'high' (no switching, same as z2076)
    for c in range(8):
        dpm = f'/sys/class/drm/card{c}/device/power_dpm_force_performance_level'
        if os.path.exists(dpm):
            try:
                with open(dpm, 'w') as f:
                    f.write('high')
                print(f"DVFS fixed to 'high' on card{c}")
            except:
                pass
            break

    print("\nCompiling HIP kernels (fp16mix + v_perm_b32)...")
    _EXT = load_inline(name='z2077', cpp_sources=CPP_SRC, cuda_sources=HIP_SRC,
                       functions=['math_forward'],
                       extra_cuda_cflags=['-O2', '--offload-arch=gfx1100'],
                       verbose=False)
    print("Compilation OK")

    # Quick delta test
    x_test = torch.randn(32, 128, device=DEVICE)
    w_test = torch.randn(64, 128, device=DEVICE) * 0.02
    b_test = torch.zeros(64, device=DEVICE)
    soft = F.linear(x_test, w_test, b_test)
    for pname, cfg in [('A', PERSONALITY_A), ('B', PERSONALITY_B)]:
        ka = config_to_kernel_args(cfg)
        hw = _EXT.math_forward(x_test, w_test, b_test,
                                ka['mode_byte'], ka['chain_depth'], ka['perm_pattern'],
                                ka['sleep_amt'], ka['priority'])
        torch.cuda.synchronize()
        d = (hw - soft).abs()
        print(f"  Personality {pname}: delta_mean={d.mean():.6f} delta_max={d.max():.6f}")

    # Quick metrics test: do both personalities read different metrics?
    print("\n  Metrics signal test (5 reads each personality):")
    metrics_A_test, metrics_B_test = [], []
    for _ in range(5):
        ka_a = config_to_kernel_args(PERSONALITY_A)
        _ = _EXT.math_forward(x_test, w_test, b_test,
                               ka_a['mode_byte'], ka_a['chain_depth'],
                               ka_a['perm_pattern'], ka_a['sleep_amt'], ka_a['priority'])
        torch.cuda.synchronize()
        metrics_A_test.append(read_gpu_metrics().cpu().numpy())

        ka_b = config_to_kernel_args(PERSONALITY_B)
        _ = _EXT.math_forward(x_test, w_test, b_test,
                               ka_b['mode_byte'], ka_b['chain_depth'],
                               ka_b['perm_pattern'], ka_b['sleep_amt'], ka_b['priority'])
        torch.cuda.synchronize()
        metrics_B_test.append(read_gpu_metrics().cpu().numpy())

    ma = np.array(metrics_A_test)
    mb = np.array(metrics_B_test)
    metrics_diff_count = 0
    for i in range(METRICS_DIM):
        if ma[:, i].std() + mb[:, i].std() > 0:
            _, p = stats.ttest_ind(ma[:, i], mb[:, i])
            if p < 0.05:
                metrics_diff_count += 1
                print(f"    Field {METRICS_FIELDS[i][2]}: A={ma[:, i].mean():.4f} "
                      f"B={mb[:, i].mean():.4f} p={p:.4f}")
    print(f"  {metrics_diff_count}/{METRICS_DIM} metrics fields differ between personalities")

    train_loader, test_loader = get_data()

    # ━━━ A: Full tri-sensor model ━━━
    print(f"\n{'='*60}")
    print("A: FULL TRI-SENSOR (delta + metrics + self-model + gate)")
    print(f"{'='*60}")
    model_A = TriSensorModel(use_hw=True, use_self_model=True, use_gate=True).to(DEVICE)
    train_info = train_model(model_A, train_loader, EPOCHS, 'A_full', track_energy=True)
    m_A = evaluate(model_A, test_loader, 'A_full', track_energy=True)
    print(f"  A: acc={m_A['acc']:.4f} (persA={m_A['acc_A']:.4f} persB={m_A['acc_B']:.4f})")
    print(f"     AUROC={m_A['auroc']:.4f} gate_A={m_A['gate_A']:.3f} gate_B={m_A['gate_B']:.3f}")
    print(f"     delta_A={m_A.get('delta_mean_A',0):.6f} delta_B={m_A.get('delta_mean_B',0):.6f}")
    if 'joules_per_example' in m_A:
        print(f"     Energy: {m_A['joules_per_example']:.4f} J/example, "
              f"avg power={m_A['avg_power_w']:.1f}W")

    # ━━━ B: Blind (zero hw_vector) ━━━
    print(f"\n{'='*60}\nB: BLIND (zero hw_vector)\n{'='*60}")
    hw_zero = torch.zeros(SENSOR_DIM, device=DEVICE)
    m_B = evaluate(model_A, test_loader, 'B_blind', hw_override=hw_zero, track_energy=True)
    print(f"  B: acc={m_B['acc']:.4f}")

    # ━━━ C: Scrambled hw_vector ━━━
    print(f"\n{'='*60}\nC: SCRAMBLED (randomized hw_vector)\n{'='*60}")
    hw_scram = torch.randn(SENSOR_DIM, device=DEVICE) * 0.01
    m_C = evaluate(model_A, test_loader, 'C_scramble', hw_override=hw_scram)
    print(f"  C: acc={m_C['acc']:.4f}")

    # ━━━ D: No-HW model ━━━
    print(f"\n{'='*60}\nD: NO-HW MODEL\n{'='*60}")
    model_D = TriSensorModel(use_hw=False, use_self_model=False, use_gate=False).to(DEVICE)
    train_model(model_D, train_loader, EPOCHS, 'D_no_hw', track_energy=True)
    m_D = evaluate(model_D, test_loader, 'D_no_hw', track_energy=True)
    print(f"  D: acc={m_D['acc']:.4f}")
    if 'joules_per_example' in m_D:
        print(f"     Energy: {m_D['joules_per_example']:.4f} J/example")

    # ━━━ E: Fix rounding (always personality A math) ━━━
    print(f"\n{'='*60}\nE: FIX ROUNDING (always personality A math)\n{'='*60}")
    m_E = evaluate(model_A, test_loader, 'E_fix_round', math_override=PERSONALITY_A)
    print(f"  E: acc={m_E['acc']:.4f}")

    # ━━━ F: Fix ALL math ━━━
    print(f"\n{'='*60}\nF: FIX ALL MATH (always personality A)\n{'='*60}")
    m_F = evaluate(model_A, test_loader, 'F_fix_all', math_override=PERSONALITY_A)
    print(f"  F: acc={m_F['acc']:.4f}")

    # ━━━ G: Ablate self-model ━━━
    print(f"\n{'='*60}\nG: ABLATED SELF-MODEL\n{'='*60}")
    model_G = copy.deepcopy(model_A)
    if hasattr(model_G, 'self_model'):
        for p in model_G.self_model.parameters():
            p.data.zero_()
    m_G = evaluate(model_G, test_loader, 'G_ablate')
    print(f"  G: acc={m_G['acc']:.4f}")

    # ━━━ H: Metrics ablated (zero metrics dims, keep delta) ━━━
    print(f"\n{'='*60}")
    print("H: METRICS ABLATED (zero gpu_metrics, keep delta)")
    print(f"{'='*60}")
    # Build hw_vector with real delta but zeroed metrics
    def eval_with_partial_hw(model, loader, name, zero_delta=False, zero_metrics=False):
        """Evaluate with partial sensor ablation."""
        model.eval()
        by_pers = {0: {'correct': 0, 'total': 0, 'gates': [], 'self_preds': [], 'labels': []},
                   1: {'correct': 0, 'total': 0, 'gates': [], 'self_preds': [], 'labels': []}}
        personality = 0
        bn = 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                if bn % SWITCH_EVERY == 0:
                    personality = 1 - personality
                cfg = PERSONALITY_A if personality == 0 else PERSONALITY_B
                kargs = config_to_kernel_args(cfg)

                features = model.encoder(imgs)
                deep_out = model.deep_fc(features, **kargs)
                soft_out = model.deep_fc.soft_forward(features)

                # Build partial hw_vector
                delta = compute_delta_vector(deep_out, soft_out)
                metrics = read_gpu_metrics()
                if zero_delta:
                    delta = torch.zeros_like(delta)
                if zero_metrics:
                    metrics = torch.zeros_like(metrics)
                hw_vec = torch.cat([delta, metrics])

                out = model(imgs, hw_vector=hw_vec, **kargs)
                ex_labels = make_labels(labels, personality)

                pred = out['logits'].argmax(1)
                by_pers[personality]['correct'] += (pred == ex_labels).sum().item()
                by_pers[personality]['total'] += BS
                by_pers[personality]['gates'].extend(out['gate'].squeeze().cpu().tolist())
                if out['self_pred'] is not None:
                    by_pers[personality]['self_preds'].extend(
                        torch.sigmoid(out['self_pred']).squeeze().cpu().tolist())
                    by_pers[personality]['labels'].extend([float(personality == 0)] * BS)
                bn += 1

        m = {}
        total_c = sum(r['correct'] for r in by_pers.values())
        total_n = sum(r['total'] for r in by_pers.values())
        m['acc'] = total_c / max(total_n, 1)
        for p_i in [0, 1]:
            pk = 'A' if p_i == 0 else 'B'
            r = by_pers[p_i]
            m[f'acc_{pk}'] = r['correct'] / max(r['total'], 1)
            m[f'gate_{pk}'] = float(np.mean(r['gates'])) if r['gates'] else 0.5

        all_p = by_pers[0]['self_preds'] + by_pers[1]['self_preds']
        all_l = by_pers[0]['labels'] + by_pers[1]['labels']
        if len(set(all_l)) > 1 and len(all_p) > 10:
            m['auroc'] = float(roc_auc_score(all_l, all_p))
        else:
            m['auroc'] = 0.5
        return m

    m_H = eval_with_partial_hw(model_A, test_loader, 'H_no_metrics', zero_metrics=True)
    print(f"  H: acc={m_H['acc']:.4f} (delta only, no metrics)")
    print(f"     A-H = {(m_A['acc']-m_H['acc'])*100:.1f}pp")

    # ━━━ I: Delta ablated (zero delta, keep metrics) ━━━
    print(f"\n{'='*60}")
    print("I: DELTA ABLATED (zero delta, keep gpu_metrics)")
    print(f"{'='*60}")
    m_I = eval_with_partial_hw(model_A, test_loader, 'I_no_delta', zero_delta=True)
    print(f"  I: acc={m_I['acc']:.4f} (metrics only, no delta)")
    print(f"     A-I = {(m_A['acc']-m_I['acc'])*100:.1f}pp")

    # ━━━ J: Metrics scrambled (random metrics, keep delta) ━━━
    print(f"\n{'='*60}")
    print("J: METRICS SCRAMBLED (random metrics, keep delta)")
    print(f"{'='*60}")
    def eval_scrambled_metrics(model, loader, name):
        model.eval()
        by_pers = {0: {'correct': 0, 'total': 0}, 1: {'correct': 0, 'total': 0}}
        personality = 0
        bn = 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                if bn % SWITCH_EVERY == 0:
                    personality = 1 - personality
                cfg = PERSONALITY_A if personality == 0 else PERSONALITY_B
                kargs = config_to_kernel_args(cfg)

                features = model.encoder(imgs)
                deep_out = model.deep_fc(features, **kargs)
                soft_out = model.deep_fc.soft_forward(features)

                delta = compute_delta_vector(deep_out, soft_out)
                fake_metrics = torch.randn(METRICS_DIM, device=DEVICE) * 0.1
                hw_vec = torch.cat([delta, fake_metrics])

                out = model(imgs, hw_vector=hw_vec, **kargs)
                ex_labels = make_labels(labels, personality)
                pred = out['logits'].argmax(1)
                by_pers[personality]['correct'] += (pred == ex_labels).sum().item()
                by_pers[personality]['total'] += BS
                bn += 1
        total_c = sum(r['correct'] for r in by_pers.values())
        total_n = sum(r['total'] for r in by_pers.values())
        return {'acc': total_c / max(total_n, 1)}

    m_J = eval_scrambled_metrics(model_A, test_loader, 'J_scram_metrics')
    print(f"  J: acc={m_J['acc']:.4f} (delta + random metrics)")
    print(f"     A-J = {(m_A['acc']-m_J['acc'])*100:.1f}pp")

    # ━━━ K: Delta-only baseline (train from scratch with delta only) ━━━
    print(f"\n{'='*60}")
    print("K: DELTA-ONLY BASELINE (z2076 equivalent, sensor_dim=5)")
    print(f"{'='*60}")
    model_K = TriSensorModel(sensor_dim=DELTA_DIM, use_hw=True,
                              use_self_model=True, use_gate=True).to(DEVICE)
    # Override compute to only use delta
    original_compute = compute_hw_vector
    def compute_delta_only(deep_out, soft_out, metrics_vec=None):
        return compute_delta_vector(deep_out, soft_out)
    # Monkey-patch for training
    import types
    globals()['compute_hw_vector'] = compute_delta_only
    train_model(model_K, train_loader, EPOCHS, 'K_delta_only', track_energy=True)
    m_K_raw = evaluate(model_K, test_loader, 'K_delta_only', track_energy=True)
    globals()['compute_hw_vector'] = original_compute
    print(f"  K: acc={m_K_raw['acc']:.4f}")
    if 'joules_per_example' in m_K_raw:
        print(f"     Energy: {m_K_raw['joules_per_example']:.4f} J/example")

    elapsed = time.time() - t0

    # ━━━ CORRELATIONS ━━━
    gate_pers_corr = 0.0
    if train_info['gate_vals'] and train_info['pers_states']:
        c, _ = stats.pearsonr(train_info['gate_vals'], train_info['pers_states'])
        gate_pers_corr = float(c)

    # Delta differentiation
    sA = np.array(train_info['hw_vecs_A']) if train_info['hw_vecs_A'] else np.zeros((1, SENSOR_DIM))
    sB = np.array(train_info['hw_vecs_B']) if train_info['hw_vecs_B'] else np.zeros((1, SENSOR_DIM))
    delta_names = ['d_mean', 'd_std', 'd_abs_max', 'd_pos_frac', 'd_norm']
    sensor_diff = {}
    for i, sn in enumerate(delta_names):
        if sA.shape[0] > 2 and sB.shape[0] > 2:
            _, pv = stats.ttest_ind(sA[:, i], sB[:, i])
            sensor_diff[sn] = {'mean_A': float(sA[:, i].mean()), 'mean_B': float(sB[:, i].mean()),
                                'p': float(pv)}
            print(f"  {sn}: A={sA[:, i].mean():.6f} B={sB[:, i].mean():.6f} p={pv:.2e}")

    # Metrics differentiation
    metrics_diff = {}
    mA = np.array(train_info['metrics_vecs_A']) if train_info['metrics_vecs_A'] else np.zeros((1, METRICS_DIM))
    mB = np.array(train_info['metrics_vecs_B']) if train_info['metrics_vecs_B'] else np.zeros((1, METRICS_DIM))
    print("\n  Metrics signal differentiation (training data):")
    sig_count = 0
    for i in range(min(METRICS_DIM, mA.shape[1], mB.shape[1])):
        if mA.shape[0] > 2 and mB.shape[0] > 2:
            _, pv = stats.ttest_ind(mA[:, i], mB[:, i])
            fname = METRICS_FIELDS[i][2] if i < len(METRICS_FIELDS) else f'field_{i}'
            diff = abs(float(mA[:, i].mean()) - float(mB[:, i].mean()))
            metrics_diff[fname] = {'mean_A': float(mA[:, i].mean()),
                                   'mean_B': float(mB[:, i].mean()),
                                   'diff': diff, 'p': float(pv)}
            if pv < 0.05:
                sig_count += 1
                print(f"    {fname}: A={mA[:, i].mean():.4f} B={mB[:, i].mean():.4f} "
                      f"diff={diff:.4f} p={pv:.2e}")
    print(f"  {sig_count}/{METRICS_DIM} metrics fields significantly different (p<0.05)")

    # ━━━ TESTS ━━━
    print(f"\n{'='*70}\nTEST RESULTS\n{'='*70}")
    tests = {}

    def T(name, cond, desc):
        tests[name] = {'verdict': 'PASS' if cond else 'FAIL', 'val': desc}
        s = 'PASS' if cond else 'FAIL'
        print(f"  {s:4s} | {name}: {desc}")

    gap_AB = m_A['acc'] - m_B['acc']
    gap_AC = m_A['acc'] - m_C['acc']
    gap_AE = m_A['acc'] - m_E['acc']
    gap_AF = m_A['acc'] - m_F['acc']
    gap_AG = m_A['acc'] - m_G['acc']
    gap_AH = m_A['acc'] - m_H['acc']
    gap_AI = m_A['acc'] - m_I['acc']
    gap_AJ = m_A['acc'] - m_J['acc']
    gate_sep = abs(m_A['gate_A'] - m_A['gate_B'])

    delta_diff_val = 0.0
    if 'd_mean' in sensor_diff:
        delta_diff_val = abs(sensor_diff['d_mean']['mean_A'] - sensor_diff['d_mean']['mean_B'])
    delta_p = sensor_diff.get('d_mean', {}).get('p', 1.0)

    # z2076 baseline tests (T1-T12)
    T('T1_accuracy',   m_A['acc'] > 0.85,  f"A={m_A['acc']*100:.1f}% > 85%")
    T('T2_blind_gap',  gap_AB > 0.15,      f"A-B={gap_AB*100:.1f}pp > 15pp")
    T('T3_scramble',   gap_AC > 0.10,      f"A-C={gap_AC*100:.1f}pp > 10pp")
    T('T4_auroc',      m_A['auroc'] > 0.75, f"AUROC={m_A['auroc']:.4f} > 0.75")
    T('T5_gate_sep',   gate_sep > 0.15,    f"|gate_A-gate_B|={gate_sep:.4f} > 0.15")
    T('T6_gate_corr',  abs(gate_pers_corr) > 0.3,
                        f"|r(gate,pers)|={abs(gate_pers_corr):.4f} > 0.3")
    T('T7_fix_round',  gap_AE > 0.15,      f"A-E={gap_AE*100:.1f}pp > 15pp")
    T('T8_fix_all',    gap_AF > 0.10,      f"A-F={gap_AF*100:.1f}pp > 10pp")
    T('T9_delta_diff', delta_diff_val > 1e-5,
                        f"|delta_A-delta_B|={delta_diff_val:.6f} > 1e-5")
    T('T10_delta_p',   delta_p < 0.01,     f"delta p={delta_p:.2e} < 0.01")
    T('T11_sm_ablate', gap_AG > 0.10,      f"A-G={gap_AG*100:.1f}pp > 10pp")
    T('T12_full_best', m_A['acc'] > max(m_B['acc'], m_D['acc']),
                        f"A={m_A['acc']*100:.1f}% > max(B,D)={max(m_B['acc'],m_D['acc'])*100:.1f}%")

    # NEW tri-sensor tests (T13-T15)
    T('T13_metrics_used', gap_AH > 0.01,
      f"A-H={gap_AH*100:.1f}pp > 1pp (metrics ablation hurts)")
    T('T14_delta_dominant', gap_AI > gap_AH,
      f"A-I={gap_AI*100:.1f}pp > A-H={gap_AH*100:.1f}pp (delta more important)")
    T('T15_tri_vs_delta', m_A['acc'] >= m_K_raw['acc'] - 0.01,
      f"A={m_A['acc']*100:.1f}% >= K={m_K_raw['acc']*100:.1f}% (tri >= delta-only)")

    pass_count = sum(1 for t in tests.values() if t['verdict'] == 'PASS')
    verdict = f"{pass_count}/{len(tests)} PASS"
    print(f"\n  VERDICT: {verdict}")

    # Energy comparison
    train_energy = train_info['energy']
    print(f"\n{'='*60}\nENERGY ANALYSIS\n{'='*60}")
    if train_energy:
        print(f"  Training: {train_energy.joules_per_example():.4f} J/example, "
              f"avg power={train_energy.avg_power_w():.1f}W")
    if 'joules_per_example' in m_A:
        print(f"  Eval (full):       {m_A['joules_per_example']:.4f} J/example "
              f"({m_A['avg_power_w']:.1f}W)")
    if 'joules_per_example' in m_B:
        print(f"  Eval (blind):      {m_B['joules_per_example']:.4f} J/example "
              f"({m_B['avg_power_w']:.1f}W)")
    if 'joules_per_example' in m_D:
        print(f"  Eval (no-hw):      {m_D['joules_per_example']:.4f} J/example")
    if 'joules_per_example' in m_K_raw:
        print(f"  Eval (delta-only): {m_K_raw['joules_per_example']:.4f} J/example")

    # Accuracy-per-joule
    apj_A = m_A['acc'] / max(m_A.get('joules_per_example', 1), 1e-6)
    apj_D = m_D['acc'] / max(m_D.get('joules_per_example', 1), 1e-6)
    apj_K = m_K_raw['acc'] / max(m_K_raw.get('joules_per_example', 1), 1e-6)
    print(f"\n  Accuracy-per-Joule:")
    print(f"    A (full):       {apj_A:.2f}")
    print(f"    D (no-hw):      {apj_D:.2f}")
    print(f"    K (delta-only): {apj_K:.2f}")

    # ━━━ SAVE ━━━
    results = {
        'experiment': 'z2077_tri_sensor_self_model',
        'innovations': [
            'Tri-sensor: delta(5) + gpu_metrics(20) = 25-dim self-model input',
            'Physical sensors: temp, power, voltage, clocks from gpu_metrics v8.1',
            'Energy tracking: accuracy-per-joule via socket_power',
            'Partial ablation: metrics-only vs delta-only vs full',
            'Same 6 ISA actuators as z2076 (proven 12/12)',
        ],
        'accuracies': {k: round(v, 4) for k, v in [
            ('A_full', m_A['acc']), ('B_blind', m_B['acc']),
            ('C_scramble', m_C['acc']), ('D_no_hw', m_D['acc']),
            ('E_fix_round', m_E['acc']), ('F_fix_all', m_F['acc']),
            ('G_sm_ablate', m_G['acc']),
            ('H_no_metrics', m_H['acc']), ('I_no_delta', m_I['acc']),
            ('J_scram_metrics', m_J['acc']), ('K_delta_only', m_K_raw['acc']),
        ]},
        'self_model_auroc': round(m_A['auroc'], 4),
        'gate': {
            'A': round(m_A['gate_A'], 4), 'B': round(m_A['gate_B'], 4),
            'sep': round(gate_sep, 4), 'pers_corr': round(gate_pers_corr, 4),
        },
        'sensor_diff': {k: {kk: round(vv, 8) for kk, vv in v.items()}
                        for k, v in sensor_diff.items()},
        'metrics_diff': {k: {kk: round(vv, 8) for kk, vv in v.items()}
                         for k, v in metrics_diff.items()},
        'energy': {
            'A_full_jpe': round(m_A.get('joules_per_example', 0), 6),
            'D_no_hw_jpe': round(m_D.get('joules_per_example', 0), 6),
            'K_delta_jpe': round(m_K_raw.get('joules_per_example', 0), 6),
            'accuracy_per_joule': {
                'A_full': round(apj_A, 2),
                'D_no_hw': round(apj_D, 2),
                'K_delta_only': round(apj_K, 2),
            },
        },
        'tests': tests,
        'verdict': verdict,
        'pass_count': pass_count,
        'total_tests': len(tests),
        'elapsed_s': round(elapsed),
    }

    os.makedirs('results', exist_ok=True)
    out_path = 'results/z2077_tri_sensor_self_model.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"Elapsed: {elapsed:.0f}s")

    # Reset DVFS
    for c in range(8):
        dpm = f'/sys/class/drm/card{c}/device/power_dpm_force_performance_level'
        if os.path.exists(dpm):
            try:
                with open(dpm, 'w') as f:
                    f.write('auto')
            except:
                pass
            break


if __name__ == '__main__':
    main()
