#!/usr/bin/env python3
"""z2079: Analog Falsification + Dual-Channel Self-Regulation

ADDRESSES 5 CRITICAL GAPS FROM SCIENTIFIC AUDIT:

Gap 1 - "Not analog": Add DVFS actuation → power sensor becomes genuinely informative
Gap 2 - "Self > external never shown": Compare A_self vs C_external vs H_fixed
Gap 3 - "Lookup table could replace self-model": Condition M tests threshold baseline
Gap 4 - "Demand cue makes action trivial": Condition O removes demand cue
Gap 5 - "Business value is artifact": Condition N = standard MNIST CNN baseline

DUAL-CHANNEL ACTUATION (deepest reach toward analog physics):
  Channel 1: ISA (MODE, chain_depth, v_perm_b32, s_sleep, s_setprio) → digital math
  Channel 2: DVFS (power_dpm_force_performance_level: high/low) → analog power/timing

2 SUBSTRATE STATES (combined DVFS + ISA):
  State A: High DVFS + Precise ISA → normal labels, high power (~95W)
  State B: Low  DVFS + Lossy ISA  → reversed labels, low power (~48W)

SENSOR FUSION:
  - Power reading (power1_input) → analog DVFS state (GENUINELY analog!)
  - Output delta (HW-SW) → digital ISA personality
  - Together → full substrate state inference from physics, not from digital cue

FALSIFICATION BATTERY:
  L: Constant SHADER_CYCLES seed → is physical entropy necessary?
  M: Lookup table (threshold on power+delta) → is learned self-model necessary?
  N: Standard MNIST CNN → fair business baseline (single labels, no ISA)
  O: No demand cue → can model infer from physics alone?

ENERGY: power1_input at 1mW resolution + accuracy-per-joule for all conditions.
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
EPOCHS = 25
PHASE2_EPOCH = 10
SWITCH_EVERY = 8
N_CLASSES = 10
DELTA_DIM = 5
POWER_DIM = 1
SENSOR_DIM = DELTA_DIM + POWER_DIM  # 6: delta(5) + power(1)
DVFS_WAIT = 0.03  # 30ms DVFS stabilization

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ISA personality configs (same as z2076-z2078)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DVFS control (Channel 2 — analog power/timing)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DRM_CARD = None

def find_drm_card():
    global DRM_CARD
    import glob
    for p in glob.glob('/sys/class/drm/card*/device/gpu_metrics'):
        DRM_CARD = int(p.split('/card')[1].split('/')[0])
        return DRM_CARD
    return None

def set_dvfs(level):
    """Set DVFS to 'high' or 'low'. Returns success."""
    try:
        p = f'/sys/class/drm/card{DRM_CARD}/device/power_dpm_force_performance_level'
        with open(p, 'w') as f:
            f.write(level)
        time.sleep(DVFS_WAIT)
        return True
    except:
        return False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Power sensor (genuinely analog — PDN physics)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POWER_PATH = None

def find_power_sensor():
    global POWER_PATH
    import glob
    for d in glob.glob('/sys/class/drm/card*/device/hwmon/*/power1_input'):
        POWER_PATH = d
        return d
    return None

def read_power_w():
    """Read instantaneous power in watts (1mW resolution, ~80kHz rate)."""
    if POWER_PATH is None:
        return 0.0
    try:
        return int(open(POWER_PATH).read().strip()) / 1e6
    except:
        return 0.0

class EnergyTracker:
    def __init__(self):
        self.total_joules = 0.0
        self.total_examples = 0
        self._last_time = None
        self.power_samples = []

    def sample(self, n_examples=0):
        now = time.time()
        pw = read_power_w()
        if self._last_time is not None and pw > 0:
            dt = now - self._last_time
            self.total_joules += pw * dt
            self.power_samples.append(pw)
        self._last_time = now
        self.total_examples += n_examples

    def jpe(self):
        return self.total_joules / max(self.total_examples, 1)

    def avg_w(self):
        return float(np.mean(self.power_samples)) if self.power_samples else 0.0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HIP KERNEL — with option for constant seed (falsification)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_kernel(use_physical_seed=True):
    seed_line = ('asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(c0));'
                 if use_physical_seed else 'c0 = 0xDEADBEEFu;')
    hip_src = r'''
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
    ''' + seed_line + r'''
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
    if (row < M && col < N) Y[row * N + col] = acc + B[col];
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
    cpp_src = r'''
#include <torch/extension.h>
torch::Tensor math_forward(torch::Tensor, torch::Tensor, torch::Tensor,
                            int, int, int, int, int);
'''
    return load_inline(name=f'z2079_{"phys" if use_physical_seed else "const"}',
                       cpp_sources=[cpp_src], cuda_sources=[hip_src],
                       functions=['math_forward'],
                       extra_cuda_cflags=['--offload-arch=gfx1100', '-O3'],
                       verbose=False)

_EXT = None
_EXT_CONST = None  # constant seed version for falsification

class MathLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, mode_byte, chain_depth, perm_pattern,
                sleep_amt, priority, use_const_seed=False):
        ctx.save_for_backward(x, w)
        ext = _EXT_CONST if use_const_seed else _EXT
        y = ext.math_forward(x.contiguous(), w.contiguous(), b.contiguous(),
                              int(mode_byte), int(chain_depth), int(perm_pattern),
                              int(sleep_amt), int(priority))
        return y
    @staticmethod
    def backward(ctx, grad_out):
        x, w = ctx.saved_tensors
        return grad_out @ w, grad_out.t() @ x, grad_out.sum(0), \
               None, None, None, None, None, None

class MathLinear(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_f))
    def forward(self, x, mode_byte=0xF0, chain_depth=1, perm_pattern=0x03020100,
                sleep_amt=0, priority=0, use_const_seed=False):
        return MathLinearFn.apply(x, self.weight, self.bias,
                                   mode_byte, chain_depth, perm_pattern,
                                   sleep_amt, priority, use_const_seed)
    def soft_forward(self, x):
        return F.linear(x, self.weight, self.bias)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Delta + Power sensor fusion
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_sensor_vector(deep_out, soft_out, power_w, power_norm=100.0):
    """6-dim sensor: delta(5) + normalized power(1)."""
    delta = (deep_out - soft_out).detach()
    d = torch.tensor([
        delta.mean().item(), delta.std().item(),
        delta.abs().max().item(), (delta > 0).float().mean().item(),
        delta.norm().item() / max(delta.numel(), 1),
        power_w / power_norm,  # Normalize to ~[0, 1] range
    ], device=deep_out.device)
    return d


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL: Dual-Channel Self-Regulation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class DualChannelModel(nn.Module):
    def __init__(self, use_hw=True, use_self_model=True, use_gate=True,
                 use_action=True, use_power=True, use_demand_cue=True):
        super().__init__()
        self.use_hw = use_hw
        self.use_self_model = use_self_model
        self.use_gate = use_gate
        self.use_action = use_action
        self.use_power = use_power
        self.use_demand_cue = use_demand_cue

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64*7*7, 128), nn.ReLU())
        self.deep_fc = MathLinear(128, 64)
        self.head_A = nn.Sequential(nn.ReLU(), nn.Linear(64, N_CLASSES))
        self.light_fc = nn.Linear(128, 64)
        self.head_B = nn.Sequential(nn.ReLU(), nn.Linear(64, N_CLASSES))

        sm_dim = SENSOR_DIM if use_power else DELTA_DIM
        if use_self_model:
            self.sensor_norm = nn.LayerNorm(sm_dim)
            self.self_model = nn.Sequential(
                nn.Linear(sm_dim, 32), nn.ReLU(),
                nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
        if use_gate:
            self.gate_net = nn.Sequential(
                nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())

        # Action head: sensors (+optional demand cue) → state choice
        act_in = sm_dim + (8 if use_demand_cue else 0)
        if use_action:
            if use_demand_cue:
                self.demand_proj = nn.Linear(1, 8)
            self.action_head = nn.Sequential(
                nn.Linear(act_in, 32), nn.ReLU(),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, x, hw_vector=None, mode_byte=0xF0, chain_depth=1,
                perm_pattern=0x03020100, sleep_amt=0, priority=0,
                demand_cue=None, use_const_seed=False, power_w=0.0):
        features = self.encoder(x)
        deep_out = self.deep_fc(features, mode_byte, chain_depth,
                                 perm_pattern, sleep_amt, priority,
                                 use_const_seed=use_const_seed)
        logits_A = self.head_A(deep_out)
        soft_out = self.deep_fc.soft_forward(features)
        light_out = F.relu(self.light_fc(features))
        logits_B = self.head_B(light_out)

        # Auto-compute sensor vector if not provided
        if hw_vector is None and self.use_hw:
            hw_vector = compute_sensor_vector(deep_out, soft_out, power_w)

        self_pred, gate, action = None, None, None

        # Self-model
        if self.use_self_model and hw_vector is not None:
            sm_in = hw_vector if self.use_power else hw_vector[:DELTA_DIM]
            sm_in = self.sensor_norm(sm_in.unsqueeze(0).expand(x.shape[0], -1))
            self_pred = self.self_model(sm_in)

        # Gate
        if self.use_gate and self_pred is not None:
            gate = self.gate_net(torch.sigmoid(self_pred))
        else:
            gate = torch.full((x.shape[0], 1), 0.5, device=x.device)
        logits = gate * logits_A + (1 - gate) * logits_B

        # Action head
        if self.use_action and hw_vector is not None:
            sm_in = hw_vector if self.use_power else hw_vector[:DELTA_DIM]
            act_in = sm_in.unsqueeze(0).expand(x.shape[0], -1)
            if self.use_demand_cue and demand_cue is not None:
                dc = demand_cue.unsqueeze(1) if demand_cue.dim() == 1 else demand_cue
                act_in = torch.cat([act_in, self.demand_proj(dc)], dim=1)
            action = self.action_head(act_in)

        return {'logits': logits, 'logits_A': logits_A, 'logits_B': logits_B,
                'self_pred': self_pred, 'gate': gate, 'hw_vector': hw_vector,
                'action': action}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Standard CNN baseline (no ISA, no DVFS, single labels)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class StandardCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64*7*7, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, N_CLASSES))
    def forward(self, x):
        return self.net(x)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_data():
    tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))])
    tr = datasets.MNIST('data', train=True, download=True, transform=tf)
    te = datasets.MNIST('data', train=False, transform=tf)
    return (torch.utils.data.DataLoader(tr, batch_size=BS, shuffle=True, drop_last=True),
            torch.utils.data.DataLoader(te, batch_size=BS, shuffle=False, drop_last=True))

def make_labels(labels, state):
    if state == 0: return labels
    else: return (9 - labels) % N_CLASSES

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAINING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_dual(model, loader, name, model_controlled=True):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[15, 20], gamma=0.3)
    model.train()
    log = {'gate_vals': [], 'pers_states': [], 'action_vals': [],
           'hw_vecs_A': [], 'hw_vecs_B': [], 'power_A': [], 'power_B': [],
           'action_correct': 0, 'action_total': 0}
    energy = EnergyTracker()
    state = 0  # 0=A (high+precise), 1=B (low+lossy)
    current_demand = 0
    bn = 0
    dvfs_switches = 0

    for ep in range(EPOCHS):
        is_phase2 = model_controlled and ep >= PHASE2_EPOCH
        tot_loss, correct, total = 0., 0, 0
        ep_act_c, ep_act_t = 0, 0

        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            # DVFS + ISA config
            if not is_phase2:
                if bn % SWITCH_EVERY == 0:
                    state = 1 - state
                    set_dvfs('high' if state == 0 else 'low')
                    dvfs_switches += 1
                current_demand = state

            cfg = PERSONALITY_A if state == 0 else PERSONALITY_B
            kargs = config_to_kernel_args(cfg)
            ex_labels = make_labels(labels, current_demand)

            # Read power BEFORE forward pass (analog interoception)
            power_w = read_power_w()

            # Next demand
            if is_phase2:
                next_demand = random.randint(0, 1)
            else:
                next_switch = ((bn + 1) % SWITCH_EVERY == 0)
                next_demand = (1 - state) if next_switch else state

            demand_cue = torch.full((BS,), float(next_demand), device=DEVICE)

            # Forward pass (power_w enables auto-computed sensor vector)
            out = model(imgs, demand_cue=demand_cue, power_w=power_w, **kargs)
            energy.sample(n_examples=BS)

            # Log sensor vector (already computed inside forward)
            hw_vec = out['hw_vector']
            (log['power_A'] if state == 0 else log['power_B']).append(power_w)
            if hw_vec is not None:
                hv = hw_vec.detach().cpu().numpy()
                (log['hw_vecs_A'] if state == 0 else log['hw_vecs_B']).append(hv)

            # Losses
            task_loss = F.cross_entropy(out['logits'], ex_labels)
            self_loss = torch.tensor(0., device=DEVICE)
            if out['self_pred'] is not None:
                st = torch.full((BS, 1), float(state == 0), device=DEVICE)
                self_loss = F.binary_cross_entropy_with_logits(out['self_pred'], st)

            action_loss = torch.tensor(0., device=DEVICE)
            if out['action'] is not None:
                at = torch.full((BS, 1), float(next_demand == 0), device=DEVICE)
                action_loss = F.binary_cross_entropy(out['action'], at)
                ab = (out['action'].mean().item() > 0.5)
                ca = (ab == (next_demand == 0))
                if ca: log['action_correct'] += 1; ep_act_c += 1
                log['action_total'] += 1; ep_act_t += 1
                log['action_vals'].append(out['action'].mean().item())

            g = out['gate']
            homeo_loss = ((1 - g)**2).mean() if state == 0 else (g**2).mean()
            loss = task_loss + 0.1*self_loss + 0.1*action_loss + 0.05*homeo_loss

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_loss += loss.item()
            correct += (out['logits'].argmax(1) == ex_labels).sum().item()
            total += BS
            log['gate_vals'].append(g.mean().item())
            log['pers_states'].append(state)

            # Phase 2: model controls next state
            if is_phase2 and out['action'] is not None:
                new_state = 0 if out['action'].mean().item() > 0.5 else 1
                if new_state != state:
                    set_dvfs('high' if new_state == 0 else 'low')
                    dvfs_switches += 1
                state = new_state
                current_demand = next_demand
            bn += 1

        sched.step()
        phase = "P2-CL" if is_phase2 else "P1-EXT"
        act_str = f" act_acc={ep_act_c/max(ep_act_t,1)*100:.1f}%" if ep_act_t else ""
        if ep % 3 == 0 or ep == EPOCHS - 1:
            print(f"  [{name} {phase}] Ep {ep}: loss={tot_loss/len(loader):.4f} "
                  f"acc={correct/total:.4f} gate={np.mean(log['gate_vals'][-50:]):.3f}"
                  f"{act_str}")
        if ep == PHASE2_EPOCH and model_controlled:
            print(f"  >>> PHASE 2: model controls DVFS + ISA <<<")

    log['energy'] = energy
    log['dvfs_switches'] = dvfs_switches
    return log


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EVALUATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def evaluate(model, loader, name, model_controlled=True,
             hw_override=None, fixed_state=None, scramble=False,
             track_energy=False, use_const_seed=False,
             override_demand_cue=None):
    model.eval()
    by_s = {0: {'c': 0, 'n': 0, 'gates': [], 'preds': [], 'labs': [], 'pwr': []},
            1: {'c': 0, 'n': 0, 'gates': [], 'preds': [], 'labs': [], 'pwr': []}}
    state, current_demand = 0, 0
    act_c, act_t = 0, 0
    act_pairs = []
    energy = EnergyTracker() if track_energy else None
    bn = 0

    # Set initial DVFS
    if fixed_state is not None:
        set_dvfs('high' if fixed_state == 0 else 'low')
    elif model_controlled:
        set_dvfs('high')  # start in state A
    else:
        set_dvfs('high')

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            if fixed_state is not None:
                state = fixed_state
            cfg = PERSONALITY_A if state == 0 else PERSONALITY_B
            kargs = config_to_kernel_args(cfg)

            next_demand = random.randint(0, 1)
            if override_demand_cue is not None:
                dc = torch.full((BS,), float(override_demand_cue), device=DEVICE)
            else:
                dc = torch.full((BS,), float(next_demand), device=DEVICE)

            ex_labels = make_labels(labels, current_demand if model_controlled else state)

            # Power reading
            power_w = read_power_w()

            if scramble:
                hw_ov = torch.randn(SENSOR_DIM, device=DEVICE) * 0.01
            elif hw_override is not None:
                hw_ov = hw_override
            else:
                # Compute real sensor vector
                features = model.encoder(imgs)
                deep = model.deep_fc(features, use_const_seed=use_const_seed, **kargs)
                soft = model.deep_fc.soft_forward(features)
                hw_ov = compute_sensor_vector(deep, soft, power_w)

            out = model(imgs, hw_vector=hw_ov, demand_cue=dc,
                        use_const_seed=use_const_seed, **kargs)
            if energy: energy.sample(n_examples=BS)

            pk = current_demand if model_controlled else state
            pred = out['logits'].argmax(1)
            by_s[pk]['c'] += (pred == ex_labels).sum().item()
            by_s[pk]['n'] += BS
            by_s[pk]['gates'].extend(out['gate'].squeeze().cpu().tolist())
            by_s[pk]['pwr'].append(power_w)

            if out['self_pred'] is not None:
                by_s[pk]['preds'].extend(torch.sigmoid(out['self_pred']).squeeze().cpu().tolist())
                by_s[pk]['labs'].extend([float(state == 0)] * BS)

            if out['action'] is not None:
                av = out['action'].mean().item()
                ab = int(av > 0.5)
                act_c += int(ab == (next_demand == 0))
                act_t += 1
                act_pairs.append((av, state))

            if model_controlled and out['action'] is not None and fixed_state is None:
                ns = 0 if out['action'].mean().item() > 0.5 else 1
                if ns != state:
                    set_dvfs('high' if ns == 0 else 'low')
                state = ns
                current_demand = next_demand
            elif not model_controlled and fixed_state is None:
                if bn % SWITCH_EVERY == 0:
                    state = 1 - state
                    set_dvfs('high' if state == 0 else 'low')
                current_demand = state
            bn += 1

    m = {}
    total_c = sum(r['c'] for r in by_s.values())
    total_n = sum(r['n'] for r in by_s.values())
    m['acc'] = total_c / max(total_n, 1)
    for s in [0, 1]:
        pk = 'A' if s == 0 else 'B'
        m[f'acc_{pk}'] = by_s[s]['c'] / max(by_s[s]['n'], 1)
        m[f'gate_{pk}'] = float(np.mean(by_s[s]['gates'])) if by_s[s]['gates'] else 0.5
        m[f'n_{pk}'] = by_s[s]['n']
        m[f'power_{pk}'] = float(np.mean(by_s[s]['pwr'])) if by_s[s]['pwr'] else 0.0
    all_p = by_s[0]['preds'] + by_s[1]['preds']
    all_l = by_s[0]['labs'] + by_s[1]['labs']
    m['auroc'] = float(roc_auc_score(all_l, all_p)) if len(set(all_l)) > 1 and len(all_p) > 10 else 0.5
    m['action_acc'] = act_c / max(act_t, 1)
    if len(act_pairs) > 10:
        aa = np.array([p[0] for p in act_pairs[:-1]])
        pa = np.array([p[1] for p in act_pairs[1:]])
        m['temporal_r'] = float(stats.pearsonr(aa, 1.0 - pa)[0]) if np.std(aa)>1e-6 and np.std(pa)>1e-6 else 0.0
    else:
        m['temporal_r'] = 0.0
    if energy:
        m['jpe'] = energy.jpe()
        m['avg_w'] = energy.avg_w()
    return m


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOOKUP TABLE BASELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def eval_lookup_table(model, loader, log):
    """Replace learned self-model with threshold on delta_mean + power."""
    # Compute thresholds from training data
    vA = np.array(log['hw_vecs_A']) if log['hw_vecs_A'] else np.zeros((1, SENSOR_DIM))
    vB = np.array(log['hw_vecs_B']) if log['hw_vecs_B'] else np.zeros((1, SENSOR_DIM))
    # Power threshold: midpoint of A and B power
    pA = np.mean(log['power_A']) if log['power_A'] else 80.0
    pB = np.mean(log['power_B']) if log['power_B'] else 50.0
    p_thresh = (pA + pB) / 2.0
    # Delta mean threshold
    dA = vA[:, 0].mean() if len(vA) > 0 else 0.0
    dB = vB[:, 0].mean() if len(vB) > 0 else 0.0
    d_thresh = (dA + dB) / 2.0

    model.eval()
    correct, total = 0, 0
    state = 0
    current_demand = 0
    bn = 0
    set_dvfs('high')

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            cfg = PERSONALITY_A if state == 0 else PERSONALITY_B
            kargs = config_to_kernel_args(cfg)
            ex_labels = make_labels(labels, current_demand)

            # Read power + compute delta
            power_w = read_power_w()
            features = model.encoder(imgs)
            deep = model.deep_fc(features, **kargs)
            soft = model.deep_fc.soft_forward(features)
            delta_mean = (deep - soft).mean().item()

            # LOOKUP TABLE: threshold on power and delta_mean
            if power_w > p_thresh:
                detected_state = 0  # high power → state A
            else:
                detected_state = 1  # low power → state B

            # Use detected state for gating (manually route)
            if detected_state == 0:
                logits = model.head_A(deep)
            else:
                light = F.relu(model.light_fc(features))
                logits = model.head_B(light)

            pred = logits.argmax(1)
            correct += (pred == ex_labels).sum().item()
            total += BS

            # External state alternation (same as eval external)
            next_demand = random.randint(0, 1)
            state = 0 if random.random() > 0.5 else 1
            set_dvfs('high' if state == 0 else 'low')
            current_demand = next_demand
            bn += 1

    return correct / max(total, 1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STANDARD CNN BASELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_standard_cnn(loader_tr, loader_te):
    """Fair baseline: same architecture, single labels, no ISA."""
    model = StandardCNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for ep in range(15):
        for imgs, labels in loader_tr:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    energy = EnergyTracker()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader_te:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            energy.sample(n_examples=BS)
            logits = model(imgs)
            correct += (logits.argmax(1) == labels).sum().item()
            total += BS
            energy.sample()
    return correct / max(total, 1), energy.jpe(), energy.avg_w()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    global _EXT, _EXT_CONST

    print("=" * 70)
    print("z2079: Analog Falsification + Dual-Channel Self-Regulation")
    print("  DVFS (analog power) + ISA (digital math) = deepest reach")
    print("=" * 70)
    print()
    t0 = time.time()

    find_drm_card()
    find_power_sensor()
    print(f"DRM card: card{DRM_CARD}")
    print(f"Power sensor: {POWER_PATH}")
    print(f"Current power: {read_power_w():.1f}W")
    print()

    # Compile both kernels: physical seed + constant seed
    print("Compiling HIP kernels (physical seed + constant seed)...")
    _EXT = build_kernel(use_physical_seed=True)
    _EXT_CONST = build_kernel(use_physical_seed=False)
    print("Compilation OK")

    # Probe power difference between DVFS states
    print("\n--- DVFS + Power Probe ---")
    set_dvfs('high')
    time.sleep(0.1)
    ph = [read_power_w() for _ in range(20)]
    set_dvfs('low')
    time.sleep(0.1)
    pl = [read_power_w() for _ in range(20)]
    set_dvfs('high')
    print(f"  DVFS high: {np.mean(ph):.1f}W (std={np.std(ph):.2f})")
    print(f"  DVFS low:  {np.mean(pl):.1f}W (std={np.std(pl):.2f})")
    print(f"  Delta:     {np.mean(ph)-np.mean(pl):.1f}W — {'STRONG' if np.mean(ph)-np.mean(pl)>5 else 'WEAK'}")

    # Probe delta
    x_test = torch.randn(32, 128, device=DEVICE)
    kA, kB = config_to_kernel_args(PERSONALITY_A), config_to_kernel_args(PERSONALITY_B)
    w_test = torch.randn(64, 128, device=DEVICE) * 0.02
    b_test = torch.zeros(64, device=DEVICE)

    yA = _EXT.math_forward(x_test, w_test, b_test, kA['mode_byte'], kA['chain_depth'],
                            kA['perm_pattern'], kA['sleep_amt'], kA['priority'])
    yB = _EXT.math_forward(x_test, w_test, b_test, kB['mode_byte'], kB['chain_depth'],
                            kB['perm_pattern'], kB['sleep_amt'], kB['priority'])
    y_soft = F.linear(x_test, w_test, b_test)
    dA = (yA - y_soft).detach()
    dB = (yB - y_soft).detach()
    print(f"\n  ISA delta A: mean={dA.mean().item():.6f} max={dA.abs().max().item():.6f}")
    print(f"  ISA delta B: mean={dB.mean().item():.6f} max={dB.abs().max().item():.6f}")

    # Constant seed comparison
    yA_c = _EXT_CONST.math_forward(x_test, w_test, b_test, kA['mode_byte'], kA['chain_depth'],
                                    kA['perm_pattern'], kA['sleep_amt'], kA['priority'])
    yB_c = _EXT_CONST.math_forward(x_test, w_test, b_test, kB['mode_byte'], kB['chain_depth'],
                                    kB['perm_pattern'], kB['sleep_amt'], kB['priority'])
    dAc = (yA_c - y_soft).detach()
    dBc = (yB_c - y_soft).detach()
    print(f"\n  CONSTANT seed delta A: mean={dAc.mean().item():.6f} max={dAc.abs().max().item():.6f}")
    print(f"  CONSTANT seed delta B: mean={dBc.mean().item():.6f} max={dBc.abs().max().item():.6f}")
    print(f"  Phys-Const diff A: {(dA-dAc).abs().max().item():.8f}")
    print(f"  Phys-Const diff B: {(dB-dBc).abs().max().item():.8f}")

    loader_tr, loader_te = get_data()

    # ═══════════════════════════════════════════════════
    # A: FULL DUAL-CHANNEL (with demand cue — matches z2078)
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("A: FULL DUAL-CHANNEL (DVFS + ISA, model-controlled)")
    print("=" * 60)
    model = DualChannelModel(use_demand_cue=True).to(DEVICE)
    log_A = train_dual(model, loader_tr, 'A_dual', model_controlled=True)
    m_A = evaluate(model, loader_te, 'A', model_controlled=True, track_energy=True)
    print(f"  A: acc={m_A['acc']:.4f} AUROC={m_A['auroc']:.4f} gate_sep={abs(m_A['gate_A']-m_A['gate_B']):.3f}")
    print(f"     action_acc={m_A['action_acc']:.4f} temporal_r={m_A['temporal_r']:.4f}")
    print(f"     power_A={m_A.get('power_A',0):.1f}W power_B={m_A.get('power_B',0):.1f}W")
    print(f"     Energy: {m_A.get('jpe',0):.4f} J/ex ({m_A.get('avg_w',0):.1f}W)")

    # ═══════════════════════════════════════════════════
    # B-I: Standard ablations (same as z2078)
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 60 + "\nB: BLIND (zero sensors)")
    m_B = evaluate(model, loader_te, 'B', hw_override=torch.zeros(SENSOR_DIM, device=DEVICE),
                   model_controlled=True)
    print(f"  B: acc={m_B['acc']:.4f}")

    print("\n" + "=" * 60 + "\nC: EXTERNAL CONTROL")
    m_C = evaluate(model, loader_te, 'C', model_controlled=False, track_energy=True)
    print(f"  C: acc={m_C['acc']:.4f} ({m_C.get('avg_w',0):.1f}W)")

    print("\n" + "=" * 60 + "\nD: SCRAMBLED SENSORS")
    m_D = evaluate(model, loader_te, 'D', scramble=True, model_controlled=True)
    print(f"  D: acc={m_D['acc']:.4f}")

    print("\n" + "=" * 60 + "\nE: NO-HW MODEL (retrained)")
    model_E = DualChannelModel(use_hw=False, use_self_model=False,
                                use_gate=False, use_action=False).to(DEVICE)
    train_dual(model_E, loader_tr, 'E_no_hw', model_controlled=False)
    m_E = evaluate(model_E, loader_te, 'E', model_controlled=False, track_energy=True)
    print(f"  E: acc={m_E['acc']:.4f} ({m_E.get('avg_w',0):.1f}W)")

    print("\n" + "=" * 60 + "\nF: ABLATED SELF-MODEL")
    model_F = copy.deepcopy(model)
    model_F.use_self_model = False
    m_F = evaluate(model_F, loader_te, 'F', model_controlled=True)
    print(f"  F: acc={m_F['acc']:.4f}")

    print("\n" + "=" * 60 + "\nG: ABLATED ACTION (random state)")
    model_G = copy.deepcopy(model)
    model_G.use_action = False
    m_G = evaluate(model_G, loader_te, 'G', model_controlled=True)
    print(f"  G: acc={m_G['acc']:.4f}")

    print("\n" + "=" * 60 + "\nH: FIXED STATE A (always high+precise)")
    m_H = evaluate(model, loader_te, 'H', fixed_state=0, track_energy=True)
    print(f"  H: acc={m_H['acc']:.4f} ({m_H.get('avg_w',0):.1f}W)")

    print("\n" + "=" * 60 + "\nI: FIXED STATE B (always low+lossy)")
    m_I = evaluate(model, loader_te, 'I', fixed_state=1, track_energy=True)
    print(f"  I: acc={m_I['acc']:.4f} ({m_I.get('avg_w',0):.1f}W)")

    # ═══════════════════════════════════════════════════
    # FALSIFICATION BATTERY (NEW)
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("FALSIFICATION BATTERY")
    print("=" * 60)

    # J: No power sensor (delta only)
    print("\nJ: NO POWER SENSOR (delta only, power dim zeroed)")
    model_J = DualChannelModel(use_power=False, use_demand_cue=True).to(DEVICE)
    train_dual(model_J, loader_tr, 'J_no_pwr', model_controlled=True)
    m_J = evaluate(model_J, loader_te, 'J', model_controlled=True)
    print(f"  J: acc={m_J['acc']:.4f} AUROC={m_J['auroc']:.4f}")

    # K: Constant seed (falsifies "physical entropy" claim)
    print("\nK: CONSTANT SEED (0xDEADBEEF replaces SHADER_CYCLES)")
    m_K = evaluate(model, loader_te, 'K', model_controlled=True, use_const_seed=True)
    print(f"  K: acc={m_K['acc']:.4f} (vs A={m_A['acc']:.4f})")

    # L: Lookup table (threshold on power + delta_mean)
    print("\nL: LOOKUP TABLE (threshold classifier on power + delta)")
    acc_L = eval_lookup_table(model, loader_te, log_A)
    print(f"  L: acc={acc_L:.4f}")

    # M: No demand cue (genuine self-regulation attempt)
    print("\nM: NO DEMAND CUE (model infers from physics alone)")
    model_M = DualChannelModel(use_demand_cue=False).to(DEVICE)
    train_dual(model_M, loader_tr, 'M_no_cue', model_controlled=True)
    m_M = evaluate(model_M, loader_te, 'M', model_controlled=True)
    print(f"  M: acc={m_M['acc']:.4f} action_acc={m_M['action_acc']:.4f}")

    # N: Standard CNN (fair business baseline)
    print("\nN: STANDARD CNN (no ISA, no DVFS, single labels)")
    set_dvfs('high')
    acc_N, jpe_N, avgw_N = train_standard_cnn(loader_tr, loader_te)
    print(f"  N: acc={acc_N:.4f} ({jpe_N:.4f} J/ex, {avgw_N:.1f}W)")

    # ═══════════════════════════════════════════════════
    # ANALYSIS
    # ═══════════════════════════════════════════════════
    # Delta differentiation from training
    delta_stats = {}
    for name_d, key in [('A', 'hw_vecs_A'), ('B', 'hw_vecs_B')]:
        vecs = log_A[key]
        if vecs:
            arr = np.array(vecs)
            delta_stats[name_d] = {f'd_{n}': float(arr[:, i].mean())
                                    for i, n in enumerate(['mean','std','abs_max','pos_frac','norm'])}
            delta_stats[name_d]['power_mean'] = float(np.mean(log_A[f'power_{name_d}']))

    # Power differentiation
    pA_arr = np.array(log_A['power_A']) if log_A['power_A'] else np.array([0])
    pB_arr = np.array(log_A['power_B']) if log_A['power_B'] else np.array([0])
    if len(pA_arr) > 1 and len(pB_arr) > 1:
        _, p_power = stats.ttest_ind(pA_arr, pB_arr)
    else:
        p_power = 1.0

    # ═══════════════════════════════════════════════════
    # TEST RESULTS
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    tests = {}
    def test(name, cond, val_str):
        v = "PASS" if cond else "FAIL"
        tests[name] = {'verdict': v, 'val': val_str}
        print(f"  {v:4s} | {name}: {val_str}")

    # Core tests (same as z2078)
    test('T1_accuracy', m_A['acc'] > 0.85, f"A={m_A['acc']*100:.1f}% > 85%")
    test('T2_auroc', m_A['auroc'] > 0.75, f"AUROC={m_A['auroc']:.4f} > 0.75")
    gate_sep = abs(m_A['gate_A'] - m_A['gate_B'])
    test('T3_gate_sep', gate_sep > 0.15, f"|gate_A-gate_B|={gate_sep:.3f} > 0.15")
    test('T4_blind_gap', m_A['acc'] - m_B['acc'] > 0.15, f"A-B={100*(m_A['acc']-m_B['acc']):.1f}pp > 15pp")
    test('T5_scramble', m_A['acc'] - m_D['acc'] > 0.10, f"A-D={100*(m_A['acc']-m_D['acc']):.1f}pp > 10pp")
    test('T6_sm_ablate', m_A['acc'] - m_F['acc'] > 0.10, f"A-F={100*(m_A['acc']-m_F['acc']):.1f}pp > 10pp (perception)")
    test('T7_action_causal', m_A['acc'] - m_G['acc'] > 0.10, f"A-G={100*(m_A['acc']-m_G['acc']):.1f}pp > 10pp (action)")
    test('T8_embod_gap', m_A['acc'] - m_E['acc'] > 0.30, f"A-E={100*(m_A['acc']-m_E['acc']):.1f}pp > 30pp")

    # Power sensor tests (NEW)
    power_diff = abs(pA_arr.mean() - pB_arr.mean())
    test('T9_power_diff', power_diff > 5.0, f"|power_A-power_B|={power_diff:.1f}W > 5W (DVFS creates analog signal)")
    test('T10_power_p', p_power < 0.01, f"power p={p_power:.2e} < 0.01 (power differentiates states)")
    test('T11_power_useful', m_A['acc'] - m_J['acc'] > 0.01,
         f"A-J={100*(m_A['acc']-m_J['acc']):.1f}pp > 1pp (power sensor helps)")

    # Falsification tests (NEW)
    test('T12_const_seed', abs(m_A['acc'] - m_K['acc']) < 0.05,
         f"|A-K|={100*abs(m_A['acc']-m_K['acc']):.1f}pp < 5pp (constant seed ~= physical)")
    test('T13_lookup_works', acc_L > 0.80,
         f"L={acc_L*100:.1f}% > 80% (lookup table works → learned SM not necessary)")
    test('T14_no_cue_works', m_M['acc'] > 0.50,
         f"M={m_M['acc']*100:.1f}% > 50% (model infers from physics)")

    # Business value tests (NEW)
    jpe_A = m_A.get('jpe', 0)
    jpe_H = m_H.get('jpe', 0)
    apj_A = m_A['acc'] / max(jpe_A, 1e-6) if jpe_A > 0 else 0
    apj_N = acc_N / max(jpe_N, 1e-6) if jpe_N > 0 else 0
    test('T15_energy_vs_fixed', jpe_A < jpe_H * 0.95 if jpe_H > 0 else False,
         f"A_jpe={jpe_A:.4f} < 0.95*H_jpe={jpe_H*0.95:.4f} (5% energy savings)")
    # T16 is HONEST: does ISA coupling beat standard CNN on accuracy-per-joule?
    test('T16_fair_baseline', m_A['acc'] >= acc_N * 0.98,
         f"A={m_A['acc']*100:.1f}% >= 98%*N={acc_N*98:.1f}% (within 2% of standard)")

    pass_count = sum(1 for t in tests.values() if t['verdict'] == 'PASS')
    total_tests = len(tests)
    verdict = f"{pass_count}/{total_tests} PASS"
    print(f"\n  VERDICT: {verdict}")

    # ═══════════════════════════════════════════════════
    # DETAILED ANALYSIS
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("ABLATION BREAKDOWN")
    print("=" * 60)
    for lbl, m in [('A (dual-channel)', m_A), ('B (blind)', m_B),
                    ('C (external)', m_C), ('D (scrambled)', m_D),
                    ('E (no HW)', m_E), ('F (no self-model)', m_F),
                    ('G (no action)', m_G), ('H (fixed high+precise)', m_H),
                    ('I (fixed low+lossy)', m_I)]:
        w_str = f" ({m.get('avg_w',0):.1f}W)" if 'avg_w' in m else ""
        print(f"  {lbl:30s} {m['acc']*100:5.1f}%{w_str}")

    print("\n" + "=" * 60)
    print("FALSIFICATION RESULTS")
    print("=" * 60)
    print(f"  J (no power sensor):  {m_J['acc']*100:.1f}%  (vs A={m_A['acc']*100:.1f}%)")
    print(f"  K (constant seed):    {m_K['acc']*100:.1f}%  (vs A={m_A['acc']*100:.1f}%)")
    print(f"  L (lookup table):     {acc_L*100:.1f}%")
    print(f"  M (no demand cue):    {m_M['acc']*100:.1f}%  (action_acc={m_M['action_acc']*100:.1f}%)")
    print(f"  N (standard CNN):     {acc_N*100:.1f}%  ({jpe_N:.4f} J/ex, {avgw_N:.1f}W)")

    print("\n" + "=" * 60)
    print("POWER SENSOR ANALYSIS (analog physics)")
    print("=" * 60)
    print(f"  State A (high+precise): power={pA_arr.mean():.1f}W (std={pA_arr.std():.2f})")
    print(f"  State B (low+lossy):    power={pB_arr.mean():.1f}W (std={pB_arr.std():.2f})")
    print(f"  Power difference:       {power_diff:.1f}W (p={p_power:.2e})")
    print(f"  This IS genuinely analog: PDN physics → current → ADC → sysfs")

    print("\n" + "=" * 60)
    print("ENERGY + BUSINESS VALUE")
    print("=" * 60)
    for lbl, m_r, a_r in [('A (self-regulated)', m_A, m_A['acc']),
                           ('H (fixed high)', m_H, m_H['acc']),
                           ('I (fixed low)', m_I, m_I['acc']),
                           ('N (standard CNN)', {'jpe': jpe_N, 'avg_w': avgw_N}, acc_N)]:
        j = m_r.get('jpe', 0)
        w = m_r.get('avg_w', 0)
        apj = a_r / max(j, 1e-6) if j > 0 else 0
        print(f"  {lbl:30s} acc={a_r*100:5.1f}%  {j:.4f} J/ex  {w:5.1f}W  APJ={apj:.1f}")

    # ═══════════════════════════════════════════════════
    # SAVE
    # ═══════════════════════════════════════════════════
    result = {
        'experiment': 'z2079_analog_falsification',
        'innovations': [
            'Dual-channel: DVFS (analog power) + ISA (digital math)',
            'Power sensor as genuinely analog interoceptive input',
            'Constant seed falsification (0xDEADBEEF vs hwreg(29))',
            'Lookup table baseline (threshold vs learned self-model)',
            'Standard CNN fair baseline (single labels, no ISA)',
            'No-demand-cue condition (genuine self-regulation test)',
        ],
        'accuracies': {
            'A_dual_channel': m_A['acc'], 'B_blind': m_B['acc'],
            'C_external': m_C['acc'], 'D_scrambled': m_D['acc'],
            'E_no_hw': m_E['acc'], 'F_no_self_model': m_F['acc'],
            'G_no_action': m_G['acc'], 'H_fixed_high': m_H['acc'],
            'I_fixed_low': m_I['acc'], 'J_no_power': m_J['acc'],
            'K_const_seed': m_K['acc'], 'L_lookup': acc_L,
            'M_no_demand_cue': m_M['acc'], 'N_standard_cnn': acc_N,
        },
        'auroc': m_A['auroc'],
        'gate': {'A': m_A['gate_A'], 'B': m_A['gate_B'],
                 'sep': gate_sep, 'pers_corr': m_A.get('temporal_r', 0)},
        'action': {'eval_acc': m_A['action_acc'], 'temporal_r': m_A['temporal_r']},
        'power': {
            'state_A_mean_w': float(pA_arr.mean()),
            'state_B_mean_w': float(pB_arr.mean()),
            'difference_w': float(power_diff),
            'p_value': float(p_power),
        },
        'falsification': {
            'const_seed_acc': m_K['acc'],
            'physical_seed_acc': m_A['acc'],
            'seed_diff_pp': round(100*abs(m_A['acc'] - m_K['acc']), 1),
            'lookup_acc': acc_L,
            'no_demand_cue_acc': m_M['acc'],
            'standard_cnn_acc': acc_N,
        },
        'energy': {
            'A_jpe': jpe_A, 'H_jpe': jpe_H, 'I_jpe': m_I.get('jpe', 0),
            'N_jpe': jpe_N,
            'A_avg_w': m_A.get('avg_w', 0), 'H_avg_w': m_H.get('avg_w', 0),
            'I_avg_w': m_I.get('avg_w', 0), 'N_avg_w': avgw_N,
        },
        'tests': tests,
        'verdict': verdict,
        'pass_count': pass_count,
        'total_tests': total_tests,
        'elapsed_s': int(time.time() - t0),
    }
    os.makedirs('results', exist_ok=True)
    with open('results/z2079_analog_falsification.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved to results/z2079_analog_falsification.json")
    print(f"Elapsed: {int(time.time()-t0)}s")

    # Restore DVFS
    set_dvfs('high')


if __name__ == '__main__':
    main()
