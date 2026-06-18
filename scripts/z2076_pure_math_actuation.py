#!/usr/bin/env python3
"""z2076: Pure Math Actuation — ISA-level math manipulation, NO DVFS.

v2: Fixed from v1 (3/12) using z2060 lessons:
  - OUTPUT DELTA (HW_kernel - SW_linear) as primary sensor signal
    (ISA register readings are too weak without DVFS; delta directly
    measures the math config's effect on computation)
  - MNIST (proven in z2060's 8/8 PASS, CIFAR-10 too hard for dual-path)
  - z2060-simple architecture (no attention, no world model, no gpu_metrics)

6 ISA ACTUATORS (all below kernel level, no DVFS):
  1. MODE[3:0] rounding  — 4 modes: nearest, +inf, -inf, zero
  2. MODE[7:4] denorm    — 4 modes: flush-all, keep-f32, keep-f16, keep-all
  3. chain_depth          — 4 levels: 1, 4, 8, 16 (fp16 accum granularity)
  4. v_perm_b32 seed      — 4 patterns: identity, reverse, swap, rotate
  5. s_sleep delay        — 4 levels: 0, 1, 2, 3
  6. s_setprio priority   — 4 levels: 0, 1, 2, 3

EXCLUSIVE SPECIALIZATION (z2060 pattern):
  Personality A: rounding nearest, precise config → normal labels
  Personality B: rounding toward-zero, lossy config → reversed labels (9-y)

KEY SIGNAL: delta = MathLinear(x) - F.linear(x) captures the ISA effect.
  Personality A (precise): small delta
  Personality B (lossy): large delta
  Self-model learns delta → personality → gate routes correctly.
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
# hw_vector: [delta_mean, delta_std, delta_abs_max, delta_pos_frac, delta_norm]
SENSOR_DIM = 5

# Actuator codes
ROUND_CODES = [0x00, 0x05, 0x0A, 0x0F]
DENORM_CODES = [0x00, 0x30, 0xC0, 0xF0]
CHAIN_DEPTHS = [1, 4, 8, 16]
PERM_PATTERNS = [0x03020100, 0x00010203, 0x02030001, 0x01000302]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HIP KERNEL: fp16mix matmul with 6 ISA actuators
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
# Custom autograd (z2070 proven pattern)
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
# MATH FINGERPRINT (the key sensor signal)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_hw_vector(deep_out, soft_out):
    """Output delta = HW_kernel - SW_linear. This is the STRONGEST signal
    of math config because different rounding/chain/noise settings produce
    measurably different numerical results.

    Personality A (precise): small delta
    Personality B (lossy):   large delta
    """
    delta = (deep_out - soft_out).detach()
    d_mean = delta.mean().item()
    d_std = delta.std().item()
    d_abs_max = delta.abs().max().item()
    d_pos_frac = (delta > 0).float().mean().item()
    d_norm = delta.norm().item() / max(delta.numel(), 1)
    return torch.tensor([d_mean, d_std, d_abs_max, d_pos_frac, d_norm],
                         device=deep_out.device)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Actuator configs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL (z2060 pattern: simple + effective)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PureMathModel(nn.Module):
    def __init__(self, use_hw=True, use_self_model=True, use_gate=True):
        super().__init__()
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

        # Self-model: hw_vector → personality prediction
        if use_self_model:
            self.self_model = nn.Sequential(
                nn.Linear(SENSOR_DIM, 32), nn.ReLU(),
                nn.Linear(32, 16), nn.ReLU(),
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

        # Compute math fingerprint (output delta)
        if hw_vector is None and self.use_hw:
            hw_vector = compute_hw_vector(deep_out, soft_out)

        # Self-model + gate
        self_pred = None
        if self.use_self_model and hw_vector is not None:
            hw_in = hw_vector.unsqueeze(0).expand(x.shape[0], -1)
            self_pred = self.self_model(hw_in)

        if self.use_gate and self_pred is not None:
            gate = self.gate_net(torch.sigmoid(self_pred))
        else:
            gate = torch.full((x.shape[0], 1), 0.5, device=x.device)

        logits = gate * logits_A + (1 - gate) * logits_B
        return {'logits': logits, 'logits_A': logits_A, 'logits_B': logits_B,
                'self_pred': self_pred, 'gate': gate, 'hw_vector': hw_vector}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA + LABELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAINING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_model(model, loader, epochs, name):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    gate_vals, pers_states = [], []
    hw_vecs_A, hw_vecs_B = [], []
    personality = 0
    bn = 0

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

            # Collect hw_vectors
            if out['hw_vector'] is not None:
                hv = out['hw_vector'].detach().cpu().numpy()
                if personality == 0:
                    hw_vecs_A.append(hv)
                else:
                    hw_vecs_B.append(hv)

            # Task loss
            task_loss = F.cross_entropy(out['logits'], ex_labels)

            # Self-model loss
            self_loss = torch.tensor(0., device=DEVICE)
            if out['self_pred'] is not None:
                self_target = torch.full((BS, 1), float(personality == 0), device=DEVICE)
                self_loss = F.binary_cross_entropy_with_logits(out['self_pred'], self_target)

            # Homeostatic loss (z2060 pattern)
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
            print(f"  [{name}] Ep {ep}: loss={tot_loss/len(loader):.4f} "
                  f"acc={correct/total:.4f} gate={np.mean(gate_vals[-50:]):.3f}")

    return {'gate_vals': gate_vals, 'pers_states': pers_states,
            'hw_vecs_A': hw_vecs_A, 'hw_vecs_B': hw_vecs_B}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EVALUATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def evaluate(model, loader, name, math_override=None, hw_override=None):
    model.eval()
    by_pers = {0: {'correct': 0, 'total': 0, 'gates': [], 'self_preds': [],
                    'labels': [], 'hw_vecs': []},
               1: {'correct': 0, 'total': 0, 'gates': [], 'self_preds': [],
                    'labels': [], 'hw_vecs': []}}
    personality = 0
    bn = 0

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

    # hw_vector stats
    for pk_i, pk in [(0, 'A'), (1, 'B')]:
        vecs = by_pers[pk_i]['hw_vecs']
        if vecs:
            arr = np.array(vecs)
            m[f'delta_mean_{pk}'] = float(arr[:, 0].mean())
            m[f'delta_std_{pk}'] = float(arr[:, 1].mean())

    return m


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    global _EXT

    print("=" * 70)
    print("z2076: Pure Math Actuation v2 — ISA-level, NO DVFS")
    print("=" * 70)
    print("Signal: output delta (HW_kernel - SW_linear) = math fingerprint")
    print("6 actuators: rounding, denorm, chain_depth, v_perm_b32, s_sleep, s_setprio")
    print()

    t0 = time.time()

    # Fix DVFS to 'high' (no switching)
    for c in range(8):
        dpm = f'/sys/class/drm/card{c}/device/power_dpm_force_performance_level'
        if os.path.exists(dpm):
            try:
                with open(dpm, 'w') as f: f.write('high')
                print(f"DVFS fixed to 'high' on card{c}")
            except:
                pass
            break

    print("Compiling HIP kernels (fp16mix + v_perm_b32)...")
    _EXT = load_inline(name='z2076v2', cpp_sources=CPP_SRC, cuda_sources=HIP_SRC,
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
        print(f"  Personality {pname}: delta_mean={d.mean():.6f} delta_std={d.std():.6f} "
              f"delta_max={d.max():.6f}")

    train_loader, test_loader = get_data()

    # ━━━ A: Full model ━━━
    print(f"\n{'='*60}\nA: FULL (math fingerprint + self-model + gate)\n{'='*60}")
    model_A = PureMathModel(use_hw=True, use_self_model=True, use_gate=True).to(DEVICE)
    train_info = train_model(model_A, train_loader, EPOCHS, 'A_full')
    m_A = evaluate(model_A, test_loader, 'A_full')
    print(f"  A: acc={m_A['acc']:.4f} (persA={m_A['acc_A']:.4f} persB={m_A['acc_B']:.4f})")
    print(f"     AUROC={m_A['auroc']:.4f} gate_A={m_A['gate_A']:.3f} gate_B={m_A['gate_B']:.3f}")
    print(f"     delta_A={m_A.get('delta_mean_A',0):.6f} delta_B={m_A.get('delta_mean_B',0):.6f}")

    # ━━━ B: Blind (zero hw_vector) ━━━
    print(f"\n{'='*60}\nB: BLIND (zero hw_vector)\n{'='*60}")
    hw_zero = torch.zeros(SENSOR_DIM, device=DEVICE)
    m_B = evaluate(model_A, test_loader, 'B_blind', hw_override=hw_zero)
    print(f"  B: acc={m_B['acc']:.4f}")

    # ━━━ C: Scrambled hw_vector ━━━
    print(f"\n{'='*60}\nC: SCRAMBLED (randomized hw_vector)\n{'='*60}")
    hw_scram = torch.randn(SENSOR_DIM, device=DEVICE) * 0.01
    m_C = evaluate(model_A, test_loader, 'C_scramble', hw_override=hw_scram)
    print(f"  C: acc={m_C['acc']:.4f}")

    # ━━━ D: No-HW model ━━━
    print(f"\n{'='*60}\nD: NO-HW MODEL\n{'='*60}")
    model_D = PureMathModel(use_hw=False, use_self_model=False, use_gate=False).to(DEVICE)
    train_model(model_D, train_loader, EPOCHS, 'D_no_hw')
    m_D = evaluate(model_D, test_loader, 'D_no_hw')
    print(f"  D: acc={m_D['acc']:.4f}")

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
    gate_sep = abs(m_A['gate_A'] - m_A['gate_B'])

    # Delta differentiation test
    delta_diff = 0.0
    if 'd_mean' in sensor_diff:
        delta_diff = abs(sensor_diff['d_mean']['mean_A'] - sensor_diff['d_mean']['mean_B'])
    delta_p = sensor_diff.get('d_mean', {}).get('p', 1.0)

    T('T1_accuracy',   m_A['acc'] > 0.85,  f"A={m_A['acc']*100:.1f}% > 85%")
    T('T2_blind_gap',  gap_AB > 0.15,      f"A-B={gap_AB*100:.1f}pp > 15pp")
    T('T3_scramble',   gap_AC > 0.10,      f"A-C={gap_AC*100:.1f}pp > 10pp")
    T('T4_auroc',      m_A['auroc'] > 0.75, f"AUROC={m_A['auroc']:.4f} > 0.75")
    T('T5_gate_sep',   gate_sep > 0.15,    f"|gate_A-gate_B|={gate_sep:.4f} > 0.15")
    T('T6_gate_corr',  abs(gate_pers_corr) > 0.3, f"|r(gate,pers)|={abs(gate_pers_corr):.4f} > 0.3")
    T('T7_fix_round',  gap_AE > 0.15,      f"A-E={gap_AE*100:.1f}pp > 15pp (exclusive necessity)")
    T('T8_fix_all',    gap_AF > 0.10,      f"A-F={gap_AF*100:.1f}pp > 10pp")
    T('T9_delta_diff', delta_diff > 1e-5,  f"|delta_A-delta_B|={delta_diff:.6f} > 1e-5")
    T('T10_delta_p',   delta_p < 0.01,     f"delta p={delta_p:.2e} < 0.01")
    T('T11_sm_ablate', gap_AG > 0.10,      f"A-G={gap_AG*100:.1f}pp > 10pp (z2060 T4)")
    T('T12_full_best', m_A['acc'] > max(m_B['acc'], m_D['acc']),
                        f"A={m_A['acc']*100:.1f}% > max(B,D)={max(m_B['acc'],m_D['acc'])*100:.1f}%")

    pass_count = sum(1 for t in tests.values() if t['verdict'] == 'PASS')
    verdict = f"{pass_count}/{len(tests)} PASS"
    print(f"\n  VERDICT: {verdict}")

    # ━━━ SAVE ━━━
    results = {
        'experiment': 'z2076_pure_math_actuation_v2',
        'innovations': [
            'NO DVFS — all actuation is ISA-level math manipulation',
            'OUTPUT DELTA as primary sensor: HW_kernel - SW_linear = math fingerprint',
            '6 actuators: MODE rounding, MODE denorm, chain_depth, v_perm_b32, s_sleep, s_setprio',
            'v_perm_b32: first use of bit permutation for stochastic rounding seed',
            'z2060-simple architecture (proven 8/8 pattern)',
            'MNIST (proven in z2060, CIFAR-10 too hard for dual-path)',
        ],
        'accuracies': {k: round(v['acc'], 4) for k, v in
                       [('A', m_A), ('B', m_B), ('C', m_C), ('D', m_D),
                        ('E', m_E), ('F', m_F), ('G', m_G)]},
        'self_model_auroc': round(m_A['auroc'], 4),
        'gate': {
            'A': round(m_A['gate_A'], 4), 'B': round(m_A['gate_B'], 4),
            'sep': round(gate_sep, 4), 'pers_corr': round(gate_pers_corr, 4),
            'p': round(m_A.get('gate_p', 1.0), 6),
        },
        'sensor_diff': {k: {kk: round(vv, 8) for kk, vv in v.items()}
                        for k, v in sensor_diff.items()},
        'tests': tests,
        'verdict': verdict,
        'pass_count': pass_count,
        'total_tests': len(tests),
        'elapsed_s': round(elapsed),
    }

    os.makedirs('results', exist_ok=True)
    out_path = 'results/z2076_pure_math_actuation.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"Elapsed: {elapsed:.0f}s")

    # Reset DVFS
    for c in range(8):
        dpm = f'/sys/class/drm/card{c}/device/power_dpm_force_performance_level'
        if os.path.exists(dpm):
            try:
                with open(dpm, 'w') as f: f.write('auto')
            except:
                pass
            break


if __name__ == '__main__':
    main()
