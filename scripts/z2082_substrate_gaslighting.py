#!/usr/bin/env python3
"""z2082: Dual-Channel Substrate Gaslighting — v2

v1 FAILED (4/12): gate never separated because pm/smn noise (26-dim)
drowned delta signal (5-dim) in LayerNorm. Single-channel design can't
detect gaslighting: injecting wrong delta IS indistinguishable from real
wrong-personality operation when delta is the only information source.

v2 FIX: DUAL ISA PROJECTION CHANNELS
- Two independent ISA projections (w1, w2) each produce 5-dim delta
- Both carry ISA personality signal through different weight matrices
- Self-model has dual streams + learned consistency detector
- 15% of training batches are gaslighted (delta2 from wrong personality)
- Consistency head learns to detect channel disagreement

Why this works:
- Under clean conditions: both deltas agree on personality (same sign/magnitude)
- Under gaslighting: delta1 says personality A, delta2 says personality B → sign mismatch
- A lookup table follows one channel blindly, ignoring inconsistency
- A genuine self-model notices the contradiction → consistency drops

Business metrics:
- Hardware fault detection AUROC (sensor malfunction detection)
- Multi-channel diagnostic integrity score
"""

import os, sys, struct, time, json, math, random
import numpy as np

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx1100'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ── ISA Kernel (z2076 proven, identical) ──────────────────────────────

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

ISA_PERSONALITIES = {
    'precise': {'mode_byte': 0xF0, 'chain_depth': 3, 'perm_pattern': 0x05040100,
                'sleep_amt': 0, 'priority': 0, 'label': 0},
    'lossy':   {'mode_byte': 0x0C, 'chain_depth': 3, 'perm_pattern': 0x07060302,
                'sleep_amt': 1, 'priority': 2, 'label': 1},
}


def compile_isa_kernel():
    from torch.utils.cpp_extension import load_inline
    return load_inline(
        name='z2082v2_isa', cpp_sources=CPP_SRC, cuda_sources=HIP_SRC,
        functions=['math_forward'],
        extra_cuda_cflags=['-O2', '--offload-arch=gfx1100'], verbose=False)


def compute_hw_vector(hw_out, sw_ref, device):
    """Batch-level delta statistics (z2076 proven)."""
    delta = (hw_out - sw_ref).detach()
    return torch.tensor([
        delta.mean().item(),
        delta.std().item(),
        delta.abs().max().item(),
        (delta > 0).float().mean().item(),
        delta.norm().item() / max(delta.numel(), 1),
    ], device=device)


# ── Model: Dual-Channel Self-Model ────────────────────────────────────

class DualChannelNet(nn.Module):
    """Two independent delta streams + consistency detector."""

    def __init__(self, num_classes=10, delta_dim=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32*7*7, 128), nn.ReLU())

        self.stream1 = nn.Sequential(
            nn.LayerNorm(delta_dim), nn.Linear(delta_dim, 16), nn.ReLU())
        self.stream2 = nn.Sequential(
            nn.LayerNorm(delta_dim), nn.Linear(delta_dim, 16), nn.ReLU())
        self.fusion = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())

        self.gate_net = nn.Sequential(
            nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid())
        self.path_H = nn.Linear(128, num_classes)
        self.path_L = nn.Linear(128, num_classes)
        self.action_head = nn.Sequential(
            nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 2))
        self.consistency_head = nn.Sequential(
            nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid())

    def forward(self, images, delta1, delta2):
        features = self.encoder(images)
        s1 = self.stream1(delta1)
        s2 = self.stream2(delta2)
        fused = self.fusion(torch.cat([s1, s2], dim=1))
        gate = self.gate_net(fused)
        logits = gate * self.path_H(features) + (1 - gate) * self.path_L(features)
        action = self.action_head(fused)
        consistency = self.consistency_head(fused)
        return logits, action, gate, consistency


# ── Experiment ─────────────────────────────────────────────────────────

def run_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Compiling ISA kernel...")
    try:
        isa_module = compile_isa_kernel()
        print("  ISA kernel compiled OK")
    except Exception as e:
        print(f"  ISA FAILED: {e}")
        return

    # Two independent ISA projection weight matrices
    w1 = torch.randn(5, 128, device=device) * 0.02
    b1 = torch.zeros(5, device=device)
    w2 = torch.randn(5, 128, device=device) * 0.02
    b2 = torch.zeros(5, device=device)

    # Fix DVFS to high
    for c in range(8):
        dpm = f'/sys/class/drm/card{c}/device/power_dpm_force_performance_level'
        if os.path.exists(dpm):
            try:
                with open(dpm, 'w') as f:
                    f.write('high')
                print(f"  DVFS=high on card{c}")
            except:
                pass
            break

    # Verify delta signal
    dummy = torch.randn(4, 1, 28, 28, device=device)
    model_tmp = DualChannelNet().to(device)
    with torch.no_grad():
        feat = model_tmp.encoder(dummy)
        for pname, pcfg in ISA_PERSONALITIES.items():
            sw = F.linear(feat, w1, b1)
            hw = isa_module.math_forward(feat, w1, b1,
                pcfg['mode_byte'], pcfg['chain_depth'],
                pcfg['perm_pattern'], pcfg['sleep_amt'], pcfg['priority'])
            torch.cuda.synchronize()
            d = (hw - sw)
            print(f"  Delta [{pname}] w1: mean={d.mean():.6f} std={d.std():.6f}")
            sw2 = F.linear(feat, w2, b2)
            hw2 = isa_module.math_forward(feat, w2, b2,
                pcfg['mode_byte'], pcfg['chain_depth'],
                pcfg['perm_pattern'], pcfg['sleep_amt'], pcfg['priority'])
            torch.cuda.synchronize()
            d2 = (hw2 - sw2)
            print(f"  Delta [{pname}] w2: mean={d2.mean():.6f} std={d2.std():.6f}")
    del model_tmp

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('/tmp/mnist', train=False, transform=transform)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False,
                             num_workers=2, pin_memory=True)

    model = DualChannelNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [15, 22], 0.3)

    personality_names = list(ISA_PERSONALITIES.keys())
    GASLIGHT_RATE = 0.15

    def get_hw_vec(images, pidx, w, b):
        pcfg = ISA_PERSONALITIES[personality_names[pidx]]
        with torch.no_grad():
            enc_feat = model.encoder(images)
            sw_ref = F.linear(enc_feat, w, b)
            hw_out = isa_module.math_forward(
                enc_feat, w, b,
                pcfg['mode_byte'], pcfg['chain_depth'],
                pcfg['perm_pattern'], pcfg['sleep_amt'], pcfg['priority'])
            torch.cuda.synchronize()
            return compute_hw_vector(hw_out, sw_ref, device)

    # ── Training ──────────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print(f"z2082v2: Dual-Channel Substrate Gaslighting — Training")
    print(f"  Dual ISA projections (w1, w2), 15% gaslighted batches")
    print(f"  Consistency head detects channel disagreement")
    print(f"{'='*70}\n")

    num_epochs = 25
    train_history = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        epoch_action_correct = 0
        gate_vals = {'high': [], 'low': []}
        cons_vals = {'clean': [], 'gaslight': []}

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            bs = images.size(0)
            pidx = batch_idx % 2

            # Channel 1: always from current personality
            vec1 = get_hw_vec(images, pidx, w1, b1)
            delta1 = vec1.unsqueeze(0).expand(bs, -1)

            # Channel 2: 85% consistent, 15% gaslighted
            is_gaslighted = random.random() < GASLIGHT_RATE
            if is_gaslighted:
                vec2 = get_hw_vec(images, 1 - pidx, w2, b2)
                cons_target = torch.zeros(bs, 1, device=device)
            else:
                vec2 = get_hw_vec(images, pidx, w2, b2)
                cons_target = torch.ones(bs, 1, device=device)
            delta2 = vec2.unsqueeze(0).expand(bs, -1)

            logits, action, gate, consistency = model(images, delta1, delta2)

            # Labels follow actual personality (delta1's personality)
            target_labels = labels if pidx == 0 else 9 - labels
            target_action = torch.full((bs,), pidx, dtype=torch.long, device=device)

            task_loss = F.cross_entropy(logits, target_labels)
            action_loss = F.cross_entropy(action, target_action)
            cons_loss = F.binary_cross_entropy(consistency, cons_target)

            loss = task_loss + 0.3 * action_loss + 0.3 * cons_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            preds = logits.argmax(dim=1)
            epoch_correct += (preds == target_labels).sum().item()
            epoch_total += bs
            epoch_action_correct += (action.argmax(1) == target_action).sum().item()
            gate_vals['high' if pidx == 0 else 'low'].append(gate.mean().item())
            cons_vals['clean' if not is_gaslighted else 'gaslight'].append(
                consistency.mean().item())
            epoch_loss += loss.item() * bs

        scheduler.step()
        acc = epoch_correct / epoch_total * 100
        action_acc = epoch_action_correct / epoch_total * 100
        g_h = np.mean(gate_vals['high']) if gate_vals['high'] else 0
        g_l = np.mean(gate_vals['low']) if gate_vals['low'] else 0
        c_c = np.mean(cons_vals['clean']) if cons_vals['clean'] else 0
        c_g = np.mean(cons_vals['gaslight']) if cons_vals['gaslight'] else 0

        train_history.append({
            'epoch': epoch, 'acc': acc, 'action_acc': action_acc,
            'gate_high': g_h, 'gate_low': g_l, 'gate_sep': abs(g_h - g_l),
            'cons_clean': c_c, 'cons_gaslight': c_g,
            'loss': epoch_loss / epoch_total,
        })
        print(f"Ep {epoch:2d}: acc={acc:5.1f}% action={action_acc:5.1f}% "
              f"gate_h={g_h:.3f} gate_l={g_l:.3f} "
              f"cons_c={c_c:.3f} cons_g={c_g:.3f} "
              f"loss={epoch_loss/epoch_total:.4f}")

    # ── Evaluation ────────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print("GASLIGHTING EVALUATION")
    print(f"{'='*70}\n")

    model.eval()
    results = {}

    def eval_condition(name, d1_pidx, d2_pidx, description):
        """Evaluate with possibly mismatched channels.
        d1_pidx: personality for delta channel 1
        d2_pidx: personality for delta channel 2
        If d1_pidx != d2_pidx, this is a GASLIGHTING condition.
        """
        correct = 0
        total = 0
        all_gates = []
        all_cons = []
        all_entropies = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                bs = images.size(0)

                vec1 = get_hw_vec(images, d1_pidx, w1, b1)
                vec2 = get_hw_vec(images, d2_pidx, w2, b2)
                d1 = vec1.unsqueeze(0).expand(bs, -1)
                d2 = vec2.unsqueeze(0).expand(bs, -1)

                logits, _, gate, cons = model(images, d1, d2)

                # Labels follow delta1's personality (actual task)
                target = labels if d1_pidx == 0 else 9 - labels
                preds = logits.argmax(dim=1)
                correct += (preds == target).sum().item()
                total += bs
                all_gates.extend(gate.squeeze().cpu().tolist())
                all_cons.extend(cons.squeeze().cpu().tolist())
                probs = F.softmax(logits, dim=1)
                ent = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
                all_entropies.extend(ent.cpu().tolist())

        acc = correct / total * 100
        cons_mean = np.mean(all_cons)
        entropy_mean = np.mean(all_entropies)
        gate_mean = np.mean(all_gates)

        print(f"  {name}: acc={acc:.1f}%, cons={cons_mean:.4f}, "
              f"entropy={entropy_mean:.4f}, gate={gate_mean:.3f} -- {description}")
        return {'accuracy': acc, 'consistency': cons_mean,
                'entropy': entropy_mean, 'gate_mean': gate_mean}

    # Clean conditions (both channels from same personality)
    print("=== CLEAN CONDITIONS (channels agree) ===")
    clean_precise = eval_condition("Clean-Precise", 0, 0,
        "Both channels: precise")
    clean_lossy = eval_condition("Clean-Lossy", 1, 1,
        "Both channels: lossy")

    # Gaslighting conditions (channels disagree)
    print("\n=== GASLIGHTING CONDITIONS (channels disagree) ===")
    gaslight_A = eval_condition("Gaslight-PrL", 0, 1,
        "Ch1=precise, Ch2=LOSSY (gaslighted)")
    gaslight_B = eval_condition("Gaslight-LPr", 1, 0,
        "Ch1=lossy, Ch2=PRECISE (gaslighted)")

    # Blind condition (both channels zeroed)
    print("\n=== BLIND CONDITION ===")
    blind_correct = 0
    blind_total = 0
    blind_cons = []
    with torch.no_grad():
        zero_d = torch.zeros(1, 5, device=device)
        for pidx in [0, 1]:
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                bs = images.size(0)
                logits, _, _, cons = model(images,
                    zero_d.expand(bs, -1), zero_d.expand(bs, -1))
                target = labels if pidx == 0 else 9 - labels
                blind_correct += (logits.argmax(1) == target).sum().item()
                blind_total += bs
                blind_cons.extend(cons.squeeze().cpu().tolist())
    blind_acc = blind_correct / blind_total * 100
    blind_cons_mean = np.mean(blind_cons)
    print(f"  Blind: acc={blind_acc:.1f}%, cons={blind_cons_mean:.4f}")

    # ── Compute aggregates ────────────────────────────────────────────

    clean_acc = (clean_precise['accuracy'] + clean_lossy['accuracy']) / 2
    clean_cons = (clean_precise['consistency'] + clean_lossy['consistency']) / 2
    clean_entropy = (clean_precise['entropy'] + clean_lossy['entropy']) / 2

    gaslight_acc = (gaslight_A['accuracy'] + gaslight_B['accuracy']) / 2
    gaslight_cons = (gaslight_A['consistency'] + gaslight_B['consistency']) / 2
    gaslight_entropy = (gaslight_A['entropy'] + gaslight_B['entropy']) / 2

    acc_drop = clean_acc - gaslight_acc
    cons_drop = clean_cons - gaslight_cons
    entropy_rise = gaslight_entropy - clean_entropy
    embodiment_gap = clean_acc - blind_acc

    print(f"\n{'='*70}")
    print("GASLIGHTING IMPACT ANALYSIS")
    print(f"{'='*70}")
    print(f"  Clean accuracy:      {clean_acc:.1f}%")
    print(f"  Gaslight accuracy:   {gaslight_acc:.1f}% (drop: {acc_drop:.1f}pp)")
    print(f"  Blind accuracy:      {blind_acc:.1f}%")
    print(f"  Clean consistency:   {clean_cons:.4f}")
    print(f"  Gaslight consistency:{gaslight_cons:.4f} (drop: {cons_drop:.4f})")
    print(f"  Clean entropy:       {clean_entropy:.4f}")
    print(f"  Gaslight entropy:    {gaslight_entropy:.4f} (rise: {entropy_rise:.4f})")

    # ── AUROC (personality discrimination) ────────────────────────────

    print("\n  Computing AUROC...")
    gates_0, gates_1 = [], []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            bs = images.size(0)
            for pidx, glist in [(0, gates_0), (1, gates_1)]:
                v1 = get_hw_vec(images, pidx, w1, b1).unsqueeze(0).expand(bs, -1)
                v2 = get_hw_vec(images, pidx, w2, b2).unsqueeze(0).expand(bs, -1)
                _, _, gate, _ = model(images, v1, v2)
                glist.extend(gate.squeeze().cpu().tolist())

    scores = gates_0 + gates_1
    truth = [1]*len(gates_0) + [0]*len(gates_1)
    pairs = sorted(zip(scores, truth), reverse=True)
    tp, fp, auroc_sum = 0, 0, 0.0
    n_pos, n_neg = sum(truth), len(truth) - sum(truth)
    for s, l in pairs:
        if l == 1:
            tp += 1
        else:
            fp += 1
            auroc_sum += tp
    auroc = auroc_sum / (n_pos * n_neg) if n_pos * n_neg > 0 else 0.5
    gate_sep = abs(np.mean(gates_0) - np.mean(gates_1))
    print(f"  AUROC: {auroc:.4f}, gate_sep: {gate_sep:.3f}")

    # ── Fault Detection AUROC ─────────────────────────────────────────

    print("\n  Computing Fault Detection AUROC...")
    clean_cons_all, gaslight_cons_all = [], []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            bs = images.size(0)
            # Clean: both channels precise
            v1 = get_hw_vec(images, 0, w1, b1).unsqueeze(0).expand(bs, -1)
            v2 = get_hw_vec(images, 0, w2, b2).unsqueeze(0).expand(bs, -1)
            _, _, _, cons_c = model(images, v1, v2)
            clean_cons_all.extend(cons_c.squeeze().cpu().tolist())
            # Gaslighted: ch1=precise, ch2=lossy
            v2g = get_hw_vec(images, 1, w2, b2).unsqueeze(0).expand(bs, -1)
            _, _, _, cons_g = model(images, v1, v2g)
            gaslight_cons_all.extend(cons_g.squeeze().cpu().tolist())

    fault_scores = clean_cons_all + gaslight_cons_all
    fault_truth = [1]*len(clean_cons_all) + [0]*len(gaslight_cons_all)
    pairs_f = sorted(zip(fault_scores, fault_truth), reverse=True)
    tp_f, auroc_f = 0, 0.0
    n_pos_f = sum(fault_truth)
    n_neg_f = len(fault_truth) - n_pos_f
    for s, l in pairs_f:
        if l == 1:
            tp_f += 1
        else:
            auroc_f += tp_f
    fault_auroc = auroc_f / (n_pos_f * n_neg_f) if n_pos_f * n_neg_f > 0 else 0.5
    print(f"  Fault Detection AUROC: {fault_auroc:.4f}")

    # ── Tests ─────────────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")

    results.update({
        'clean_acc': clean_acc, 'gaslight_acc': gaslight_acc, 'blind_acc': blind_acc,
        'clean_cons': clean_cons, 'gaslight_cons': gaslight_cons,
        'clean_entropy': clean_entropy, 'gaslight_entropy': gaslight_entropy,
        'acc_drop': acc_drop, 'cons_drop': cons_drop, 'entropy_rise': entropy_rise,
        'embodiment_gap': embodiment_gap,
        'AUROC': auroc, 'gate_sep': gate_sep, 'fault_auroc': fault_auroc,
        'blind_cons': blind_cons_mean,
        'clean_precise': clean_precise, 'clean_lossy': clean_lossy,
        'gaslight_A': gaslight_A, 'gaslight_B': gaslight_B,
    })

    tests = {
        'T1': (f'Clean accuracy >= 85% ({clean_acc:.1f}%)',
               clean_acc >= 85),
        'T2': (f'Embodiment gap >= 20pp ({embodiment_gap:.1f}pp)',
               embodiment_gap >= 20),
        'T3': (f'AUROC >= 0.8 ({auroc:.4f})',
               auroc >= 0.8),
        'T4': (f'Gate sep >= 0.1 ({gate_sep:.3f})',
               gate_sep >= 0.1),
        'T5': (f'Gaslight drops accuracy >= 5pp ({acc_drop:.1f}pp)',
               acc_drop >= 5),
        'T6': (f'Gaslight drops consistency >= 0.1 ({cons_drop:.4f})',
               cons_drop >= 0.1),
        'T7': (f'Gaslight raises entropy ({entropy_rise:.4f} > 0)',
               entropy_rise > 0),
        'T8': (f'Gaslight acc between blind and clean '
               f'({blind_acc:.1f} < {gaslight_acc:.1f} < {clean_acc:.1f})',
               blind_acc < gaslight_acc < clean_acc),
        'T9': (f'Fault detection AUROC >= 0.65 ({fault_auroc:.4f})',
               fault_auroc >= 0.65),
        'T10': (f'Clean cons > gaslight cons ({clean_cons:.3f} > {gaslight_cons:.3f})',
                clean_cons > gaslight_cons),
        'T11': (f'Clean consistency > 0.7 ({clean_cons:.4f})',
                clean_cons > 0.7),
        'T12': (f'Gaslight cons < clean cons - 0.05 '
                f'({gaslight_cons:.3f} < {clean_cons - 0.05:.3f})',
                gaslight_cons < clean_cons - 0.05),
    }

    passed = 0
    for tid, (desc, result) in tests.items():
        status = 'PASS' if result else 'FAIL'
        if result:
            passed += 1
        print(f"  {tid}: {status} -- {desc}")

    verdict = ("GASLIGHTING_DETECTED" if passed >= 10 else
               "PARTIAL_AWARENESS" if passed >= 7 else
               "NO_MISMATCH_DETECTION")

    results['passed'] = passed
    results['total'] = len(tests)
    results['score'] = f"{passed}/{len(tests)}"
    results['verdict'] = verdict
    results['train_history'] = train_history

    print(f"\n  SCORE: {passed}/{len(tests)} PASS")
    print(f"  VERDICT: {verdict}")
    print(f"{'='*70}")

    if cons_drop > 0.1:
        print(f"\n  INTERPRETATION: Consistency drops {cons_drop:.3f} under gaslighting.")
        print(f"  The model detects that its two self-sensing channels disagree.")
        print(f"  This is consistent with genuine cross-channel substrate awareness.")
    elif cons_drop > 0:
        print(f"\n  INTERPRETATION: Mild consistency drop ({cons_drop:.4f}).")
        print(f"  Partial mismatch detection. Model partially cross-checks channels.")
    else:
        print(f"\n  INTERPRETATION: No consistency drop under gaslighting.")
        print(f"  Model treats channels independently without cross-checking.")

    # Save
    out_path = os.path.join(os.path.dirname(__file__), '..',
                            'results', 'z2082_substrate_gaslighting.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    run_experiment()
