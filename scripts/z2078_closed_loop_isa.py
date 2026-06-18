#!/usr/bin/env python3
"""z2078: Closed-Loop ISA Self-Regulation — The network controls its own math.

THE TRUE CLOSED LOOP:
  batch t: model forward pass → delta → self-model → ACTION HEAD → personality choice
  batch t+1: ISA actuators set to the ACTION HEAD's choice → new forward pass → new delta → ...

  The math that computes the neural network's gradients IS the same math whose
  rounding/precision/noise is controlled by the neural network's own output.

CAUSAL CHAIN:
  demand(t) → action_head → ISA config → forward_pass(t+1) → delta → self-model → gate → accuracy

PHASE 1 (ep 0-9): External personality switching (like z2076, model learns routing)
PHASE 2 (ep 10-24): Model-controlled — action head picks ISA config for next batch
  - Demand alternates randomly
  - Labels depend on DEMANDED personality, not actual
  - Model MUST match action to demand or accuracy drops

INNOVATION vs z2076: z2076 has external schedule. z2078 closes the loop —
  the model CHOOSES its own substrate configuration.

INNOVATION vs z2061: z2061 controls DVFS (high-level). z2078 controls ISA math
  (sub-firmware level). The computation itself changes, not just the clock speed.

6 ISA ACTUATORS (same as z2076):
  1. MODE[3:0] rounding, 2. MODE[7:4] denorm, 3. chain_depth,
  4. v_perm_b32 seed, 5. s_sleep, 6. s_setprio

ENERGY TRACKING: gpu_metrics socket_power for accuracy-per-joule.
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
PHASE2_EPOCH = 10  # Switch from external to model-controlled
SWITCH_EVERY = 8
N_CLASSES = 10
SENSOR_DIM = 5  # delta only (z2077 showed metrics are noise under ISA-only)

# Actuator codes (same as z2076)
ROUND_CODES = [0x00, 0x05, 0x0A, 0x0F]
DENORM_CODES = [0x00, 0x30, 0xC0, 0xF0]
CHAIN_DEPTHS = [1, 4, 8, 16]
PERM_PATTERNS = [0x03020100, 0x00010203, 0x02030001, 0x01000302]

PERSONALITY_A = {  # Precise: small delta
    'round_idx': 0, 'denorm_idx': 3, 'chain_idx': 0,
    'perm_idx': 0, 'sleep_idx': 0, 'prio_idx': 0,
}
PERSONALITY_B = {  # Lossy: large delta
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# gpu_metrics reader (energy tracking only — z2077 showed metrics are noise)
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
    if GPU_METRICS_PATH is None:
        return 0.0
    try:
        data = open(GPU_METRICS_PATH, 'rb').read()
        if len(data) >= 0x74:
            return float(struct.unpack_from('<I', data, 0x70)[0])
        return 0.0
    except:
        return 0.0

class EnergyTracker:
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
            self.total_joules += (power_mw / 1000.0) * dt
            self.samples.append({'power_w': power_mw / 1000.0, 'dt': dt})
        self._last_time = now
        self.total_examples += n_examples

    def joules_per_example(self):
        return self.total_joules / max(self.total_examples, 1)

    def avg_power_w(self):
        return np.mean([s['power_w'] for s in self.samples]) if self.samples else 0.0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HIP KERNEL (same as z2076 — proven 12/12)
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
        return F.linear(x, self.weight, self.bias)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Delta sensor
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_delta_vector(deep_out, soft_out):
    delta = (deep_out - soft_out).detach()
    return torch.tensor([
        delta.mean().item(), delta.std().item(),
        delta.abs().max().item(), (delta > 0).float().mean().item(),
        delta.norm().item() / max(delta.numel(), 1)
    ], device=deep_out.device)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL: Closed-Loop ISA Self-Regulation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ClosedLoopISAModel(nn.Module):
    """Neural network that controls its own math substrate.

    Forward pass:
      1. Encoder → features
      2. MathLinear with CURRENT ISA config → deep_out
      3. Standard linear → soft_out
      4. Delta = deep_out - soft_out (senses current math config)
      5. Self-model(delta) → personality prediction
      6. Gate routes to head_A or head_B based on self-model
      7. Action head(delta, demand_cue) → next personality choice

    The ACTION HEAD's output from batch t sets ISA config for batch t+1.
    """
    def __init__(self, use_hw=True, use_self_model=True, use_gate=True,
                 use_action=True):
        super().__init__()
        self.use_hw = use_hw
        self.use_self_model = use_self_model
        self.use_gate = use_gate
        self.use_action = use_action

        # MNIST encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU())

        # Deep path: ISA matmul (the math that IS controlled)
        self.deep_fc = MathLinear(128, 64)
        self.head_A = nn.Sequential(nn.ReLU(), nn.Linear(64, N_CLASSES))

        # Light path: standard software
        self.light_fc = nn.Linear(128, 64)
        self.head_B = nn.Sequential(nn.ReLU(), nn.Linear(64, N_CLASSES))

        # Self-model: delta(5) → personality prediction (PERCEPTION)
        if use_self_model:
            self.delta_norm = nn.LayerNorm(SENSOR_DIM)
            self.self_model = nn.Sequential(
                nn.Linear(SENSOR_DIM, 32), nn.ReLU(),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, 1))

        # Gate: from self-model prediction
        if use_gate:
            self.gate_net = nn.Sequential(
                nn.Linear(1, 16), nn.ReLU(),
                nn.Linear(16, 1), nn.Sigmoid())

        # Action head: delta(5) + demand_cue(1) → personality choice (ACTION)
        # This IS the closed loop — the model chooses its own substrate config
        if use_action:
            self.demand_proj = nn.Linear(1, 8)
            self.action_head = nn.Sequential(
                nn.Linear(SENSOR_DIM + 8, 32), nn.ReLU(),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, x, hw_vector=None, mode_byte=0xF0, chain_depth=1,
                perm_pattern=0x03020100, sleep_amt=0, priority=0,
                demand_cue=None):
        features = self.encoder(x)

        # Deep path with CURRENT ISA config (set externally or by prev action)
        deep_out = self.deep_fc(features, mode_byte, chain_depth,
                                 perm_pattern, sleep_amt, priority)
        logits_A = self.head_A(deep_out)

        # SW light path
        soft_out = self.deep_fc.soft_forward(features)
        light_out = F.relu(self.light_fc(features))
        logits_B = self.head_B(light_out)

        # Compute delta sensor
        if hw_vector is None and self.use_hw:
            hw_vector = compute_delta_vector(deep_out, soft_out)

        # PERCEPTION: self-model predicts current personality
        self_pred = None
        if self.use_self_model and hw_vector is not None:
            hw_in = self.delta_norm(hw_vector.unsqueeze(0).expand(x.shape[0], -1))
            self_pred = self.self_model(hw_in)

        # ROUTING: gate based on self-model
        if self.use_gate and self_pred is not None:
            gate = self.gate_net(torch.sigmoid(self_pred))
        else:
            gate = torch.full((x.shape[0], 1), 0.5, device=x.device)

        logits = gate * logits_A + (1 - gate) * logits_B

        # ACTION: choose next ISA config based on current state + demand
        action = None
        if self.use_action and hw_vector is not None and demand_cue is not None:
            hw_in_act = hw_vector.unsqueeze(0).expand(x.shape[0], -1)
            dc = demand_cue.unsqueeze(1) if demand_cue.dim() == 1 else demand_cue
            demand_feat = self.demand_proj(dc)
            action_in = torch.cat([hw_in_act, demand_feat], dim=1)
            action = self.action_head(action_in)

        return {'logits': logits, 'logits_A': logits_A, 'logits_B': logits_B,
                'self_pred': self_pred, 'gate': gate, 'hw_vector': hw_vector,
                'action': action}


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

def make_labels(labels, personality):
    """Labels depend on DEMANDED personality, not actual."""
    if personality == 0:
        return labels
    else:
        return (9 - labels) % N_CLASSES


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAINING — Phase 1 (external) + Phase 2 (model-controlled)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_model(model, loader, epochs, name, model_controlled=True):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[15, 20], gamma=0.3)
    model.train()

    log = {'gate_vals': [], 'pers_states': [], 'action_vals': [],
           'demand_vals': [], 'action_correct': 0, 'action_total': 0,
           'hw_vecs_A': [], 'hw_vecs_B': [], 'phase_switch': []}
    energy = EnergyTracker()
    personality = 0  # current ISA config (0=A, 1=B)
    current_demand = 0  # what label scheme is expected
    bn = 0

    for ep in range(epochs):
        is_phase2 = model_controlled and ep >= PHASE2_EPOCH
        tot_loss, correct, total = 0., 0, 0
        ep_action_correct, ep_action_total = 0, 0

        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            # === DETERMINE ISA CONFIG ===
            if not is_phase2:
                # Phase 1: external personality switching
                if bn % SWITCH_EVERY == 0:
                    personality = 1 - personality
                current_demand = personality  # demand = actual in P1
            # (in Phase 2, personality was set by prev action head output)

            cfg = PERSONALITY_A if personality == 0 else PERSONALITY_B
            kargs = config_to_kernel_args(cfg)

            # Labels depend on DEMANDED personality
            ex_labels = make_labels(labels, current_demand)

            # Next demand (what the model should prepare for)
            if is_phase2:
                next_demand = random.randint(0, 1)  # random in P2
            else:
                # Phase 1: predict the external schedule
                next_switch = ((bn + 1) % SWITCH_EVERY == 0)
                next_demand = (1 - personality) if next_switch else personality

            demand_cue = torch.full((BS,), float(next_demand), device=DEVICE)

            # === FORWARD PASS (uses current ISA config) ===
            out = model(imgs, demand_cue=demand_cue, **kargs)
            energy.sample(n_examples=BS)

            # Collect vectors
            if out['hw_vector'] is not None:
                hv = out['hw_vector'].detach().cpu().numpy()
                (log['hw_vecs_A'] if personality == 0 else log['hw_vecs_B']).append(hv)

            # === LOSSES ===
            # Task loss
            task_loss = F.cross_entropy(out['logits'], ex_labels)

            # Self-model loss (predict current personality)
            self_loss = torch.tensor(0., device=DEVICE)
            if out['self_pred'] is not None:
                self_target = torch.full((BS, 1), float(personality == 0), device=DEVICE)
                self_loss = F.binary_cross_entropy_with_logits(out['self_pred'], self_target)

            # Action loss (predict correct next ISA config to match demand)
            action_loss = torch.tensor(0., device=DEVICE)
            if out['action'] is not None:
                action_target = torch.full((BS, 1), float(next_demand == 0), device=DEVICE)
                action_loss = F.binary_cross_entropy(out['action'], action_target)

                # Track action accuracy
                action_binary = (out['action'].mean().item() > 0.5)
                correct_action = (action_binary == (next_demand == 0))
                if correct_action:
                    log['action_correct'] += 1
                    ep_action_correct += 1
                log['action_total'] += 1
                ep_action_total += 1
                log['action_vals'].append(out['action'].mean().item())
                log['demand_vals'].append(float(next_demand == 0))

            # Homeostatic gate loss
            g = out['gate']
            if personality == 0:
                homeo_loss = ((1 - g) ** 2).mean()
            else:
                homeo_loss = (g ** 2).mean()

            loss = task_loss + 0.1 * self_loss + 0.1 * action_loss + 0.05 * homeo_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tot_loss += loss.item()
            correct += (out['logits'].argmax(1) == ex_labels).sum().item()
            total += BS
            log['gate_vals'].append(g.mean().item())
            log['pers_states'].append(personality)

            # === PHASE 2: MODEL CONTROLS NEXT ISA CONFIG ===
            if is_phase2 and out['action'] is not None:
                action_val = out['action'].mean().item()
                # action > 0.5 → personality A (precise), else personality B (lossy)
                personality = 0 if action_val > 0.5 else 1
                current_demand = next_demand  # labels will use next_demand
            else:
                current_demand = next_demand if is_phase2 else personality

            bn += 1

        sched.step()
        phase = "P2-CL" if is_phase2 else "P1-EXT"
        act_str = f" act_acc={ep_action_correct/max(ep_action_total,1)*100:.1f}%" if ep_action_total else ""
        if ep % 3 == 0 or ep == epochs - 1:
            print(f"  [{name} {phase}] Ep {ep}: loss={tot_loss/len(loader):.4f} "
                  f"acc={correct/total:.4f} gate={np.mean(log['gate_vals'][-50:]):.3f}"
                  f"{act_str}")
        if ep == PHASE2_EPOCH and model_controlled:
            log['phase_switch'].append(bn)
            print(f"  >>> PHASE 2 START: model now controls its own ISA config <<<")

    log['energy'] = energy
    return log


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EVALUATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def evaluate(model, loader, name, model_controlled=True,
             hw_override=None, math_override=None, fixed_action=None,
             scramble_delta=False, track_energy=False):
    """Evaluate with optional closed-loop ISA control.

    model_controlled: if True, action head sets ISA config for next batch.
    fixed_action: if set (0 or 1), forces specific personality regardless of action head.
    scramble_delta: if True, randomize delta before self-model.
    """
    model.eval()
    by_pers = {0: {'correct': 0, 'total': 0, 'gates': [], 'self_preds': [],
                    'labels': [], 'hw_vecs': []},
               1: {'correct': 0, 'total': 0, 'gates': [], 'self_preds': [],
                    'labels': [], 'hw_vecs': []}}

    personality = 0
    current_demand = 0
    actions_correct, actions_total = 0, 0
    action_pers_pairs = []  # (action_t, personality_{t+1})
    energy = EnergyTracker() if track_energy else None
    bn = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            # Determine ISA config
            if math_override is not None:
                cfg = math_override
            elif fixed_action is not None:
                personality = fixed_action
                cfg = PERSONALITY_A if personality == 0 else PERSONALITY_B
            else:
                cfg = PERSONALITY_A if personality == 0 else PERSONALITY_B
            kargs = config_to_kernel_args(cfg)

            # Demand = random in eval
            next_demand = random.randint(0, 1)
            demand_cue = torch.full((BS,), float(next_demand), device=DEVICE)
            ex_labels = make_labels(labels, current_demand if model_controlled else personality)

            if scramble_delta:
                hw_ov = torch.randn(SENSOR_DIM, device=DEVICE) * 0.01
            else:
                hw_ov = hw_override

            out = model(imgs, hw_vector=hw_ov, demand_cue=demand_cue, **kargs)

            if energy is not None:
                energy.sample(n_examples=BS)

            p_key = current_demand if model_controlled else personality
            pred = out['logits'].argmax(1)
            by_pers[p_key]['correct'] += (pred == ex_labels).sum().item()
            by_pers[p_key]['total'] += BS
            by_pers[p_key]['gates'].extend(out['gate'].squeeze().cpu().tolist())

            if out['hw_vector'] is not None:
                by_pers[p_key]['hw_vecs'].append(out['hw_vector'].cpu().numpy())

            if out['self_pred'] is not None:
                by_pers[p_key]['self_preds'].extend(
                    torch.sigmoid(out['self_pred']).squeeze().cpu().tolist())
                by_pers[p_key]['labels'].extend([float(personality == 0)] * BS)

            # Track action accuracy
            if out['action'] is not None:
                action_val = out['action'].mean().item()
                action_binary = int(action_val > 0.5)  # 1=personality A, 0=B
                correct_next = (action_binary == (next_demand == 0))
                actions_correct += int(correct_next)
                actions_total += 1

                # Record for temporal correlation
                action_pers_pairs.append((action_val, personality))

            # Model controls next ISA config
            if model_controlled and out['action'] is not None and fixed_action is None:
                new_pers = 0 if out['action'].mean().item() > 0.5 else 1
                personality = new_pers
                current_demand = next_demand
            elif not model_controlled:
                if bn % SWITCH_EVERY == 0:
                    personality = 1 - personality
                current_demand = personality

            bn += 1

    # Compute metrics
    m = {}
    total_c = sum(r['correct'] for r in by_pers.values())
    total_n = sum(r['total'] for r in by_pers.values())
    m['acc'] = total_c / max(total_n, 1)
    for p in [0, 1]:
        pk = 'A' if p == 0 else 'B'
        r = by_pers[p]
        m[f'acc_{pk}'] = r['correct'] / max(r['total'], 1)
        m[f'gate_{pk}'] = float(np.mean(r['gates'])) if r['gates'] else 0.5
        m[f'n_{pk}'] = r['total']

    # AUROC
    all_p = by_pers[0]['self_preds'] + by_pers[1]['self_preds']
    all_l = by_pers[0]['labels'] + by_pers[1]['labels']
    if len(set(all_l)) > 1 and len(all_p) > 10:
        m['auroc'] = float(roc_auc_score(all_l, all_p))
    else:
        m['auroc'] = 0.5

    # Action accuracy
    m['action_acc'] = actions_correct / max(actions_total, 1)

    # Temporal correlation: action_t → personality_{t+1}
    if len(action_pers_pairs) > 10:
        act_arr = np.array([p[0] for p in action_pers_pairs[:-1]])
        pers_arr = np.array([p[1] for p in action_pers_pairs[1:]])
        if np.std(act_arr) > 1e-6 and np.std(pers_arr) > 1e-6:
            m['temporal_r'], _ = stats.pearsonr(act_arr, 1.0 - pers_arr)
        else:
            m['temporal_r'] = 0.0
    else:
        m['temporal_r'] = 0.0

    # Delta stats
    for pk_i, pk in [(0, 'A'), (1, 'B')]:
        vecs = by_pers[pk_i]['hw_vecs']
        if vecs:
            arr = np.array(vecs)
            m[f'delta_mean_{pk}'] = float(arr[:, 0].mean())

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
    print("z2078: Closed-Loop ISA Self-Regulation")
    print("  The network controls its own math substrate.")
    print("=" * 70)
    print()
    print("CAUSAL CHAIN:")
    print("  demand → action_head → ISA_config → forward_pass → delta → self-model → gate → accuracy")
    print("                                  ↑                                                      |")
    print("                                  └──── THIS IS THE CLOSED LOOP ─────────────────────────┘")
    print()
    print(f"Phase 1 (ep 0-{PHASE2_EPOCH-1}): External ISA switching (learn routing)")
    print(f"Phase 2 (ep {PHASE2_EPOCH}-{EPOCHS-1}): Model-controlled ISA (learn action)")
    print()

    t0 = time.time()
    find_gpu_metrics()
    if GPU_METRICS_PATH:
        print(f"gpu_metrics: {GPU_METRICS_PATH} (energy tracking)")
        print(f"  Socket power: {read_socket_power_mw():.0f} mW")

    # Fix DVFS (we're testing ISA actuation, not DVFS)
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

    print("\nCompiling HIP kernels...")
    _EXT = load_inline(name='z2078', cpp_sources=CPP_SRC, cuda_sources=HIP_SRC,
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

    train_loader, test_loader = get_data()

    # ━━━ A: Full Closed-Loop ━━━
    print(f"\n{'='*60}")
    print("A: FULL CLOSED-LOOP (action head controls ISA config)")
    print(f"{'='*60}")
    model_A = ClosedLoopISAModel(use_hw=True, use_self_model=True,
                                  use_gate=True, use_action=True).to(DEVICE)
    train_log = train_model(model_A, train_loader, EPOCHS, 'A_closed', model_controlled=True)
    m_A = evaluate(model_A, test_loader, 'A_closed', model_controlled=True, track_energy=True)
    print(f"  A: acc={m_A['acc']:.4f} (A={m_A['acc_A']:.4f} B={m_A['acc_B']:.4f})")
    print(f"     AUROC={m_A['auroc']:.4f}  gate_A={m_A['gate_A']:.3f}  gate_B={m_A['gate_B']:.3f}")
    print(f"     action_acc={m_A['action_acc']:.4f}  temporal_r={m_A['temporal_r']:.4f}")
    print(f"     n_A={m_A['n_A']}  n_B={m_A['n_B']}")
    if 'joules_per_example' in m_A:
        print(f"     Energy: {m_A['joules_per_example']:.4f} J/ex, {m_A['avg_power_w']:.1f}W")

    # ━━━ B: Blind (zero delta) ━━━
    print(f"\n{'='*60}\nB: BLIND (zero delta)\n{'='*60}")
    hw_zero = torch.zeros(SENSOR_DIM, device=DEVICE)
    m_B = evaluate(model_A, test_loader, 'B_blind', model_controlled=True,
                   hw_override=hw_zero, track_energy=True)
    print(f"  B: acc={m_B['acc']:.4f}  action_acc={m_B['action_acc']:.4f}")

    # ━━━ C: External control only (no model action) ━━━
    print(f"\n{'='*60}\nC: EXTERNAL CONTROL (action head ignored)\n{'='*60}")
    m_C = evaluate(model_A, test_loader, 'C_external', model_controlled=False)
    print(f"  C: acc={m_C['acc']:.4f}")

    # ━━━ D: Scrambled delta ━━━
    print(f"\n{'='*60}\nD: SCRAMBLED DELTA\n{'='*60}")
    m_D = evaluate(model_A, test_loader, 'D_scramble', model_controlled=True,
                   scramble_delta=True)
    print(f"  D: acc={m_D['acc']:.4f}")

    # ━━━ E: No-HW model (no sensors, no gate, no action) ━━━
    print(f"\n{'='*60}\nE: NO-HW MODEL\n{'='*60}")
    model_E = ClosedLoopISAModel(use_hw=False, use_self_model=False,
                                  use_gate=False, use_action=False).to(DEVICE)
    train_model(model_E, train_loader, EPOCHS, 'E_no_hw', model_controlled=False)
    m_E = evaluate(model_E, test_loader, 'E_no_hw', model_controlled=False, track_energy=True)
    print(f"  E: acc={m_E['acc']:.4f}")

    # ━━━ F: Ablated self-model (perception fails) ━━━
    print(f"\n{'='*60}\nF: ABLATED SELF-MODEL (perception fails)\n{'='*60}")
    model_F = copy.deepcopy(model_A)
    if hasattr(model_F, 'self_model'):
        for p in model_F.self_model.parameters():
            p.data.zero_()
    m_F = evaluate(model_F, test_loader, 'F_no_self', model_controlled=True)
    print(f"  F: acc={m_F['acc']:.4f}")

    # ━━━ G: Ablated action head (random actions) ━━━
    print(f"\n{'='*60}\nG: ABLATED ACTION HEAD (random ISA config)\n{'='*60}")
    model_G = copy.deepcopy(model_A)
    if hasattr(model_G, 'action_head'):
        for p in model_G.action_head.parameters():
            p.data.zero_()
        # With zeroed weights, action ≈ sigmoid(0) = 0.5 → random personality
    m_G = evaluate(model_G, test_loader, 'G_no_action', model_controlled=True)
    print(f"  G: acc={m_G['acc']:.4f}  action_acc={m_G['action_acc']:.4f}")

    # ━━━ H: Fixed personality A (no switching) ━━━
    print(f"\n{'='*60}\nH: FIXED PERSONALITY A (always precise)\n{'='*60}")
    m_H = evaluate(model_A, test_loader, 'H_fixed_A', model_controlled=True,
                   fixed_action=0, track_energy=True)
    print(f"  H: acc={m_H['acc']:.4f}")

    elapsed = time.time() - t0

    # ━━━ CORRELATIONS & ANALYSIS ━━━
    gate_pers_corr = 0.0
    if train_log['gate_vals'] and train_log['pers_states']:
        c, _ = stats.pearsonr(train_log['gate_vals'], train_log['pers_states'])
        gate_pers_corr = float(c)

    train_action_acc = train_log['action_correct'] / max(train_log['action_total'], 1)

    # Delta differentiation
    sA = np.array(train_log['hw_vecs_A']) if train_log['hw_vecs_A'] else np.zeros((1, SENSOR_DIM))
    sB = np.array(train_log['hw_vecs_B']) if train_log['hw_vecs_B'] else np.zeros((1, SENSOR_DIM))
    delta_names = ['d_mean', 'd_std', 'd_abs_max', 'd_pos_frac', 'd_norm']
    sensor_diff = {}
    print(f"\n  Delta differentiation (training):")
    for i, sn in enumerate(delta_names):
        if sA.shape[0] > 2 and sB.shape[0] > 2:
            _, pv = stats.ttest_ind(sA[:, i], sB[:, i])
            sensor_diff[sn] = {'A': float(sA[:, i].mean()), 'B': float(sB[:, i].mean()), 'p': float(pv)}
            print(f"    {sn}: A={sA[:, i].mean():.6f} B={sB[:, i].mean():.6f} p={pv:.2e}")

    # ━━━ TESTS ━━━
    print(f"\n{'='*70}\nTEST RESULTS\n{'='*70}")
    tests = {}

    def T(name, cond, desc):
        tests[name] = {'verdict': 'PASS' if cond else 'FAIL', 'val': desc}
        s = 'PASS' if cond else 'FAIL'
        print(f"  {s:4s} | {name}: {desc}")

    gap_AB = m_A['acc'] - m_B['acc']
    gap_AD = m_A['acc'] - m_D['acc']
    gap_AE = m_A['acc'] - m_E['acc']
    gap_AF = m_A['acc'] - m_F['acc']
    gap_AG = m_A['acc'] - m_G['acc']
    gap_AH = m_A['acc'] - m_H['acc']
    gate_sep = abs(m_A['gate_A'] - m_A['gate_B'])

    delta_p = sensor_diff.get('d_mean', {}).get('p', 1.0)
    delta_diff_val = abs(sensor_diff.get('d_mean', {}).get('A', 0) - sensor_diff.get('d_mean', {}).get('B', 0))

    # z2076 baseline tests
    T('T1_accuracy',   m_A['acc'] > 0.85,    f"A={m_A['acc']*100:.1f}% > 85%")
    T('T2_auroc',      m_A['auroc'] > 0.75,  f"AUROC={m_A['auroc']:.4f} > 0.75")
    T('T3_gate_sep',   gate_sep > 0.15,      f"|gate_A-gate_B|={gate_sep:.3f} > 0.15")
    T('T4_blind_gap',  gap_AB > 0.15,        f"A-B={gap_AB*100:.1f}pp > 15pp")
    T('T5_scramble',   gap_AD > 0.10,        f"A-D={gap_AD*100:.1f}pp > 10pp")
    T('T6_delta_diff', delta_diff_val > 1e-5, f"|delta_A-B|={delta_diff_val:.6f} > 1e-5")
    T('T7_delta_p',    delta_p < 0.01,       f"delta p={delta_p:.2e} < 0.01")
    T('T8_sm_ablate',  gap_AF > 0.10,        f"A-F={gap_AF*100:.1f}pp > 10pp (perception causal)")

    # Closed-loop specific tests
    T('T9_action_acc',  m_A['action_acc'] > 0.70,
      f"action_acc={m_A['action_acc']*100:.1f}% > 70% (model picks right ISA)")
    T('T10_action_causal', gap_AG > 0.10,
      f"A-G={gap_AG*100:.1f}pp > 10pp (action head causally necessary)")
    T('T11_closed_loop', abs(m_A['temporal_r']) > 0.3,
      f"|temporal_r|={abs(m_A['temporal_r']):.4f} > 0.3 (action→ISA→delta chain)")
    T('T12_fixed_worse', gap_AH > 0.05,
      f"A-H={gap_AH*100:.1f}pp > 5pp (adaptive > fixed)")
    T('T13_full_best',  m_A['acc'] > max(m_B['acc'], m_E['acc']),
      f"A={m_A['acc']*100:.1f}% > max(B,E)={max(m_B['acc'],m_E['acc'])*100:.1f}%")
    T('T14_embod_gap',  gap_AE > 0.30,
      f"A-E={gap_AE*100:.1f}pp > 30pp (embodiment essential)")

    pass_count = sum(1 for t in tests.values() if t['verdict'] == 'PASS')
    verdict = f"{pass_count}/{len(tests)} PASS"
    print(f"\n  VERDICT: {verdict}")

    # Ablation breakdown
    print(f"\n{'='*60}\nABLATION BREAKDOWN\n{'='*60}")
    print(f"  A (closed-loop):     {m_A['acc']*100:.1f}%  (model controls ISA)")
    print(f"  B (blind delta):     {m_B['acc']*100:.1f}%  ({gap_AB*100:+.1f}pp)")
    print(f"  C (external ctrl):   {m_C['acc']*100:.1f}%  (action head unused)")
    print(f"  D (scrambled):       {m_D['acc']*100:.1f}%  ({gap_AD*100:+.1f}pp)")
    print(f"  E (no HW):           {m_E['acc']*100:.1f}%  ({gap_AE*100:+.1f}pp)")
    print(f"  F (no self-model):   {m_F['acc']*100:.1f}%  ({gap_AF*100:+.1f}pp — perception fails)")
    print(f"  G (no action head):  {m_G['acc']*100:.1f}%  ({gap_AG*100:+.1f}pp — action fails)")
    print(f"  H (fixed A):         {m_H['acc']*100:.1f}%  ({gap_AH*100:+.1f}pp — no adaptation)")

    # Sensorimotor loop
    print(f"\n{'='*60}\nSENSORIMOTOR LOOP\n{'='*60}")
    print(f"  Action accuracy (eval):  {m_A['action_acc']*100:.1f}%")
    print(f"  Action accuracy (train): {train_action_acc*100:.1f}%")
    print(f"  Temporal correlation:     {m_A['temporal_r']:.4f}")
    print(f"  Gate-personality corr:    {gate_pers_corr:.4f}")

    # Energy
    print(f"\n{'='*60}\nENERGY ANALYSIS\n{'='*60}")
    if 'joules_per_example' in m_A:
        print(f"  A (closed-loop): {m_A['joules_per_example']:.4f} J/ex ({m_A['avg_power_w']:.1f}W)")
    if 'joules_per_example' in m_B:
        print(f"  B (blind):       {m_B['joules_per_example']:.4f} J/ex ({m_B['avg_power_w']:.1f}W)")
    if 'joules_per_example' in m_E:
        print(f"  E (no HW):       {m_E['joules_per_example']:.4f} J/ex")
    if 'joules_per_example' in m_H:
        print(f"  H (fixed A):     {m_H['joules_per_example']:.4f} J/ex ({m_H['avg_power_w']:.1f}W)")

    apj_A = m_A['acc'] / max(m_A.get('joules_per_example', 1), 1e-6)
    apj_E = m_E['acc'] / max(m_E.get('joules_per_example', 1), 1e-6)
    print(f"  Accuracy-per-Joule: A={apj_A:.1f}  E={apj_E:.1f}")

    # ━━━ SAVE ━━━
    results = {
        'experiment': 'z2078_closed_loop_isa',
        'innovations': [
            'TRUE CLOSED LOOP: model action head controls ISA actuators for next batch',
            'The forward pass math IS the math whose config is set by the model',
            'Causal chain: demand → action → ISA → forward → delta → self-model → gate → accuracy',
            'Phase 1: external control (learn routing), Phase 2: model-controlled (learn action)',
            'Sub-firmware actuation: MODE rounding/denorm, chain, perm, sleep, prio',
        ],
        'accuracies': {k: round(v, 4) for k, v in [
            ('A_closed_loop', m_A['acc']), ('B_blind', m_B['acc']),
            ('C_external', m_C['acc']), ('D_scrambled', m_D['acc']),
            ('E_no_hw', m_E['acc']), ('F_no_self_model', m_F['acc']),
            ('G_no_action', m_G['acc']), ('H_fixed_A', m_H['acc']),
        ]},
        'self_model_auroc': round(m_A['auroc'], 4),
        'gate': {
            'A': round(m_A['gate_A'], 4), 'B': round(m_A['gate_B'], 4),
            'sep': round(gate_sep, 4), 'pers_corr': round(gate_pers_corr, 4),
        },
        'action': {
            'eval_accuracy': round(m_A['action_acc'], 4),
            'train_accuracy': round(train_action_acc, 4),
            'temporal_correlation': round(m_A['temporal_r'], 4),
            'G_action_acc': round(m_G['action_acc'], 4),
        },
        'sensor_diff': {k: {kk: round(vv, 8) for kk, vv in v.items()}
                        for k, v in sensor_diff.items()},
        'energy': {
            'A_jpe': round(m_A.get('joules_per_example', 0), 6),
            'E_jpe': round(m_E.get('joules_per_example', 0), 6),
            'H_jpe': round(m_H.get('joules_per_example', 0), 6),
            'accuracy_per_joule_A': round(apj_A, 2),
            'accuracy_per_joule_E': round(apj_E, 2),
        },
        'tests': tests,
        'verdict': verdict,
        'pass_count': pass_count,
        'total_tests': len(tests),
        'elapsed_s': round(elapsed),
    }

    os.makedirs('results', exist_ok=True)
    out_path = 'results/z2078_closed_loop_isa.json'
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
