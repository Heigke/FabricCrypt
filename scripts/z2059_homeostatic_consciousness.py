#!/usr/bin/env python3
"""z2059: Homeostatic Embodied Consciousness (v2)

Combines three scientific principles:
1. BIOLOGICAL COMPUTATIONALISM (Milinkovic & Aru 2026): algorithm IS substrate
   → WGP banking + timing make GPU silicon part of computation
2. HOMEOSTATIC REGULATION (Solms/Conscium): consciousness begins with NEED
   → Competing drives (accuracy vs efficiency) modulated by REAL GPU state
3. SELF-REFERENTIAL LOOP (Hofstadter/Anthropic): model predicts its own state
   → Self-model predicts SCLK state from combined representation

The strange loop:
  Hardware Timing → Combined Rep → Self-Model → Gate → Path Selection → Output
                         ↑                              ↓
                         └──────────────────────────────┘

v2 fix: Model receives wall-clock timing as input feature (like retinal input
includes both external scene AND blood-vessel shadows). The self-model then
introspects on the combined representation, not on images alone.

Tests (8):
  T1: A_homeostatic > 90% accuracy
  T2: Self-model AUROC > 0.8
  T3: Gate differs between states (p < 0.01)
  T4: Ablating self-model → accuracy drops > 10pp
  T5: A accuracy > D_always_light + 5pp
  T6: E_scrambled accuracy < A - 5pp
  T7: A - B_blind gap > 30pp
  T8: Gate-SCLK correlation |r| > 0.3
"""
import torch, torch.nn as nn, torch.nn.functional as F
import os, sys, json, time, numpy as np
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score
from scipy import stats

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')
os.environ.setdefault('PYTORCH_ROCM_ARCH', 'gfx1100')
from torch.utils.cpp_extension import load_inline

DEVICE = 'cuda'
BS = 256
EPOCHS = 12
SWITCH_EVERY = 20
NUM_BANKS = 8

# ━━━ HIP Kernel ━━━
HIP_SRC = r'''
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void read_wgp(int* wgp_ids, float* work, int n) {
    int bid = blockIdx.x;
    if (bid >= n || threadIdx.x != 0) return;
    uint32_t hw;
    asm volatile("s_getreg_b32 %0, hwreg(23)" : "=s"(hw));
    wgp_ids[bid] = (int)((hw >> 7) & 0xF);
    float acc = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < 5000; i++) acc += 1.0f / (float)(i+1);
    work[bid] = acc;
}

std::vector<torch::Tensor> probe(int n) {
    auto io = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto fo = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto wgps = torch::zeros({n}, io);
    auto work = torch::zeros({n}, fo);
    read_wgp<<<n, 32>>>(wgps.data_ptr<int>(), work.data_ptr<float>(), n);
    return {wgps, work};
}
'''
CPP_SRC = r'''
#include <torch/extension.h>
std::vector<torch::Tensor> probe(int n);
'''

# ━━━ DVFS ━━━
DPM = '/sys/class/drm/card0/device/power_dpm_force_performance_level'
SCLK_F = '/sys/class/drm/card0/device/pp_dpm_sclk'

def read_sclk():
    try: return [l.strip() for l in open(SCLK_F) if '*' in l][0]
    except: return '?'

def set_sclk(mode):
    try:
        open(DPM, 'w').write('low' if mode == 'low' else 'high')
        time.sleep(0.2); return True
    except: return False

def read_temp():
    try:
        for d in os.listdir('/sys/class/drm/card0/device/hwmon'):
            p = f'/sys/class/drm/card0/device/hwmon/{d}/temp1_input'
            if os.path.exists(p):
                return int(open(p).read().strip()) / 1000.0
    except: pass
    return 50.0

def measure_wall_clock(ext, n=BS):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record(); ext.probe(n); e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)


def characterize_sclk(ext):
    print("\n--- SCLK Characterization ---")
    info = {}
    for mode in ['low', 'high']:
        set_sclk(mode); time.sleep(0.5)
        for _ in range(10): ext.probe(BS); torch.cuda.synchronize()
        times = []
        for _ in range(30):
            times.append(measure_wall_clock(ext))
        info[mode] = {
            'mean': float(np.mean(times)), 'std': float(np.std(times)),
            'sclk': read_sclk(), 'temp': read_temp()
        }
        print(f"  {mode}: {info[mode]['mean']:.4f} +/- {info[mode]['std']:.4f} ms  "
              f"SCLK={info[mode]['sclk']}  T={info[mode]['temp']:.1f}C")
    ratio = info['low']['mean'] / max(info['high']['mean'], 1e-6)
    print(f"  Wall-clock ratio: {ratio:.2f}x")
    set_sclk('high')
    return info, ratio


def norm_timing(wall_ms, timing_info):
    """Normalize wall-clock timing to [0, 1] range. 0=fast(high SCLK), 1=slow(low SCLK)."""
    lo = timing_info['high']['mean']  # Fast = low ms
    hi = timing_info['low']['mean']   # Slow = high ms
    return max(0., min(1., (wall_ms - lo) / max(hi - lo, 1e-6)))


# ━━━ Model ━━━
class HomeostaticModel(nn.Module):
    """
    Model with self-referential loop and homeostatic regulation.

    Architecture:
      encoder(image) → h_img [B, 128]
      timing_proj(wall_clock) → h_hw [B, 32]
      h = concat(h_img, h_hw) → [B, 160]
      bank_transform(h, bank_ids) → h_banked [B, 128]  (full path)
      self_model(h) → P(SCLK_high)  (introspection)
      gate(self_pred) → [0,1]  (homeostatic regulation)
      output = gate * full_head(h_banked) + (1-gate) * light_head(h_img)

    The self-model predicts SCLK state from the COMBINED representation
    (image + timing). This is genuine introspection: the model detects
    its own hardware state by examining its own hidden representation.
    """
    def __init__(self, use_banks=True, use_timing=True, use_self_model=True,
                 use_gate=True, always_full=False, always_light=False):
        super().__init__()
        self.use_banks = use_banks
        self.use_timing = use_timing
        self.use_self_model = use_self_model
        self.use_gate = use_gate
        self.always_full = always_full
        self.always_light = always_light

        # Image encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64*7*7, 128), nn.ReLU())

        # Timing embedding (hardware signal → representation)
        h_dim = 128
        if use_timing:
            self.timing_proj = nn.Sequential(
                nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 32), nn.ReLU())
            h_dim = 128 + 32  # Combined dimension

        # Per-WGP weight banks: [N_BANKS, 128, h_dim] for bmm with [B, h_dim, 1] → [B, 128, 1]
        if use_banks:
            self.bank_w = nn.Parameter(torch.randn(NUM_BANKS, 128, h_dim) * 0.02)

        # Self-model: predict SCLK state from combined representation
        if use_self_model:
            self.self_model = nn.Sequential(
                nn.Linear(h_dim, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, 1))  # Logit for P(high_sclk)

        # Homeostatic gate
        if use_gate:
            self.gate_net = nn.Sequential(
                nn.Linear(1, 16), nn.ReLU(),
                nn.Linear(16, 1), nn.Sigmoid())

        # Full path head (after bank transform)
        self.head_full = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
        # Light path head (from image encoder only)
        self.head_light = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x, bank_ids=None, timing=None, sclk_state=None):
        h_img = self.encoder(x)  # [B, 128]

        # Build combined representation
        if self.use_timing and timing is not None:
            h_hw = self.timing_proj(timing.view(-1, 1))  # [B, 32]
            h = torch.cat([h_img, h_hw], dim=1)  # [B, 160]
        else:
            if self.use_timing:
                h = torch.cat([h_img, torch.zeros(h_img.shape[0], 32, device=h_img.device)], 1)
            else:
                h = h_img

        # Self-model: introspect on combined representation
        self_pred = None
        if self.use_self_model:
            self_pred = self.self_model(h)  # [B, 1]

        # Homeostatic gate
        if self.use_gate and not self.always_full and not self.always_light:
            if self.use_self_model and self_pred is not None:
                gate_input = torch.sigmoid(self_pred)
            else:
                gate_input = torch.full((h.shape[0], 1), 0.5, device=h.device)
            gate = self.gate_net(gate_input)
        elif self.always_full:
            gate = torch.ones(h.shape[0], 1, device=h.device)
        elif self.always_light:
            gate = torch.zeros(h.shape[0], 1, device=h.device)
        else:
            gate = torch.full((h.shape[0], 1), 0.5, device=h.device)

        # Full path: bank transform → classification
        if self.use_banks and bank_ids is not None:
            h_banked = torch.bmm(self.bank_w[bank_ids], h.unsqueeze(-1)).squeeze(-1)
            logits_full = self.head_full(h_banked)
        else:
            logits_full = self.head_full(h_img)

        # Light path: direct from image encoder
        logits_light = self.head_light(h_img)

        # Blend
        logits = gate * logits_full + (1 - gate) * logits_light

        return {
            'logits': logits,
            'self_pred': self_pred,
            'gate': gate,
        }


# ━━━ Labels ━━━
def make_labels(digits, bank_ids, sclk_is_high):
    """SCLK-dependent label permutation.

    SCLK low: identity (easy, light path sufficient)
    SCLK high: bank-dependent shift (hard, needs banks)
    """
    labels = digits.clone()
    if sclk_is_high:
        even_mask = (bank_ids % 2 == 0)
        odd_mask = ~even_mask
        labels[even_mask] = (digits[even_mask] + 3) % 10
        labels[odd_mask] = (digits[odd_mask] + 5) % 10
    return labels


def get_data():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    tr = datasets.MNIST('data', train=True, download=True, transform=tf)
    te = datasets.MNIST('data', train=False, transform=tf)
    return (torch.utils.data.DataLoader(tr, batch_size=BS, shuffle=True, drop_last=True),
            torch.utils.data.DataLoader(te, batch_size=BS, shuffle=False, drop_last=True))


# ━━━ Training ━━━
def train_model(model, ext, train_loader, epochs, name, timing_info,
                dvfs_enabled=True, scramble_timing=False):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    gate_vals, sclk_states, temps = [], [], []
    sclk_is_high = True
    batch_n = 0

    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for imgs, digits in train_loader:
            imgs, digits = imgs.to(DEVICE), digits.to(DEVICE)

            # DVFS cycling
            if dvfs_enabled and batch_n % SWITCH_EVERY == 0:
                sclk_is_high = not sclk_is_high
                set_sclk('high' if sclk_is_high else 'low')
                time.sleep(0.05)

            # Measure wall-clock timing (hardware signal)
            wall_ms = measure_wall_clock(ext)
            timing_norm = norm_timing(wall_ms, timing_info)
            if scramble_timing:
                timing_norm = 1.0 - timing_norm  # Invert signal

            timing_t = torch.full((BS,), timing_norm, device=DEVICE)

            # Get WGP bank IDs
            wgps = ext.probe(BS)[0]
            bank_ids = (wgps // 2).long().clamp(0, NUM_BANKS - 1)

            # Make labels
            labels = make_labels(digits, bank_ids, sclk_is_high)
            sclk_state_t = torch.full((BS,), float(sclk_is_high), device=DEVICE)

            # Forward
            out = model(imgs, bank_ids=bank_ids, timing=timing_t, sclk_state=sclk_state_t)

            # Task loss
            task_loss = F.cross_entropy(out['logits'], labels)

            # Self-model loss
            self_loss = torch.tensor(0.0, device=DEVICE)
            if out['self_pred'] is not None:
                self_target = torch.full((BS, 1), float(sclk_is_high), device=DEVICE)
                self_loss = F.binary_cross_entropy_with_logits(out['self_pred'], self_target)

            # Homeostatic loss: encourage gate to match task demands
            homeo_loss = torch.tensor(0.0, device=DEVICE)
            if model.use_gate and not model.always_full and not model.always_light:
                gate = out['gate']
                if sclk_is_high:
                    homeo_loss = ((1 - gate) ** 2).mean()
                else:
                    homeo_loss = (gate ** 2).mean()

            loss = task_loss + 0.1 * self_loss + 0.03 * homeo_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            pred = out['logits'].argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += BS

            gate_vals.append(out['gate'].mean().item())
            sclk_states.append(float(sclk_is_high))
            temps.append(read_temp())
            batch_n += 1

        acc = correct / total
        if epoch % 3 == 0 or epoch == epochs - 1:
            print(f"  [{name}] Epoch {epoch}: loss={total_loss/len(train_loader):.4f} "
                  f"acc={acc:.4f} gate={np.mean(gate_vals[-50:]):.3f}")

    return {'gate_vals': gate_vals, 'sclk_states': sclk_states, 'temps': temps}


# ━━━ Evaluation ━━━
def evaluate(model, ext, test_loader, timing_info, dvfs_enabled=True,
             scramble_timing=False):
    model.eval()
    by_state = {
        'high': {'correct': 0, 'total': 0, 'gates': [], 'self_preds': [], 'labels': []},
        'low':  {'correct': 0, 'total': 0, 'gates': [], 'self_preds': [], 'labels': []},
    }
    sclk_is_high = True
    batch_n = 0

    with torch.no_grad():
        for imgs, digits in test_loader:
            imgs, digits = imgs.to(DEVICE), digits.to(DEVICE)

            if dvfs_enabled and batch_n % SWITCH_EVERY == 0:
                sclk_is_high = not sclk_is_high
                set_sclk('high' if sclk_is_high else 'low')
                time.sleep(0.05)

            wall_ms = measure_wall_clock(ext)
            timing_norm = norm_timing(wall_ms, timing_info)
            if scramble_timing:
                timing_norm = 1.0 - timing_norm
            timing_t = torch.full((BS,), timing_norm, device=DEVICE)

            wgps = ext.probe(BS)[0]
            bank_ids = (wgps // 2).long().clamp(0, NUM_BANKS - 1)
            labels = make_labels(digits, bank_ids, sclk_is_high)

            out = model(imgs, bank_ids=bank_ids, timing=timing_t)

            sk = 'high' if sclk_is_high else 'low'
            pred = out['logits'].argmax(dim=1)
            by_state[sk]['correct'] += (pred == labels).sum().item()
            by_state[sk]['total'] += BS
            by_state[sk]['gates'].extend(out['gate'].squeeze().cpu().tolist())

            if out['self_pred'] is not None:
                by_state[sk]['self_preds'].extend(
                    torch.sigmoid(out['self_pred']).squeeze().cpu().tolist())
                by_state[sk]['labels'].extend([float(sclk_is_high)] * BS)

            batch_n += 1

    m = {}
    for s in ['high', 'low']:
        r = by_state[s]
        if r['total'] > 0:
            m[f'acc_{s}'] = r['correct'] / r['total']
            m[f'gate_{s}'] = float(np.mean(r['gates'])) if r['gates'] else 0.5

    total_c = sum(r['correct'] for r in by_state.values())
    total_n = sum(r['total'] for r in by_state.values())
    m['acc'] = total_c / max(total_n, 1)

    # Self-model AUROC
    all_p = by_state['high']['self_preds'] + by_state['low']['self_preds']
    all_l = by_state['high']['labels'] + by_state['low']['labels']
    if len(set(all_l)) > 1 and len(all_p) > 0:
        m['auroc'] = float(roc_auc_score(all_l, all_p))
    else:
        m['auroc'] = 0.5

    # Gate state difference
    g_h = by_state['high']['gates']
    g_l = by_state['low']['gates']
    if g_h and g_l and len(set(g_h + g_l)) > 1:
        t_stat, p_val = stats.ttest_ind(g_h, g_l)
        m['gate_p'] = float(p_val)
    else:
        m['gate_p'] = 1.0

    return m


# ━━━ Self-model ablation ━━━
def ablate_self_model(model):
    """Zero the self-model weights to test causal necessity."""
    if hasattr(model, 'self_model'):
        for p in model.self_model.parameters():
            p.data.zero_()


# ━━━ Main ━━━
def main():
    print("=" * 70)
    print("z2059: Homeostatic Embodied Consciousness (v2)")
    print("=" * 70)
    print()
    print("Scientific basis:")
    print("  1. Biological computationalism — algorithm IS substrate")
    print("  2. Homeostatic regulation — competing needs (accuracy vs efficiency)")
    print("  3. Self-referential loop — introspect own hardware state")
    print()

    t0 = time.time()

    # Compile HIP kernel
    print("Compiling HIP kernel...")
    ext = load_inline(name='z2059_homeo_v2', cpp_sources=CPP_SRC, cuda_sources=HIP_SRC,
                      functions=['probe'], extra_cuda_cflags=['-O2', '--offload-arch=gfx1100'],
                      verbose=False)

    wgps = ext.probe(1024)[0].cpu().numpy()
    unique_wgps = sorted(set(wgps.tolist()))
    print(f"WGP distribution: {len(unique_wgps)} unique values: {unique_wgps}")

    timing_info, wc_ratio = characterize_sclk(ext)
    train_loader, test_loader = get_data()

    # ━━━ A: Full Homeostatic (self-model + gate + banks + timing) ━━━
    print("\n" + "=" * 60)
    print("A: HOMEOSTATIC (banks + timing + self-model + gate)")
    print("=" * 60)
    model_A = HomeostaticModel(use_banks=True, use_timing=True,
                               use_self_model=True, use_gate=True).to(DEVICE)
    train_A = train_model(model_A, ext, train_loader, EPOCHS, 'A_homeo', timing_info)
    m_A = evaluate(model_A, ext, test_loader, timing_info)
    print(f"  A: acc={m_A['acc']:.4f}  AUROC={m_A['auroc']:.4f}  "
          f"gate_h={m_A.get('gate_high', 0):.3f}  gate_l={m_A.get('gate_low', 0):.3f}")

    # ━━━ B: Blind (no banks, no timing) ━━━
    print("\n" + "=" * 60)
    print("B: BLIND (no hardware signals)")
    print("=" * 60)
    model_B = HomeostaticModel(use_banks=False, use_timing=False,
                               use_self_model=False, use_gate=False).to(DEVICE)
    train_B = train_model(model_B, ext, train_loader, EPOCHS, 'B_blind', timing_info)
    m_B = evaluate(model_B, ext, test_loader, timing_info)
    print(f"  B: acc={m_B['acc']:.4f}")

    # ━━━ C: Banks + timing but no self-model (gate=0.5 constant) ━━━
    print("\n" + "=" * 60)
    print("C: NO SELF-MODEL (banks + timing, gate always 0.5)")
    print("=" * 60)
    model_C = HomeostaticModel(use_banks=True, use_timing=True,
                               use_self_model=False, use_gate=True).to(DEVICE)
    train_C = train_model(model_C, ext, train_loader, EPOCHS, 'C_no_sm', timing_info)
    m_C = evaluate(model_C, ext, test_loader, timing_info)
    print(f"  C: acc={m_C['acc']:.4f}  gate_h={m_C.get('gate_high', 0):.3f}  "
          f"gate_l={m_C.get('gate_low', 0):.3f}")

    # ━━━ D: Always light (banks available but gate=0) ━━━
    print("\n" + "=" * 60)
    print("D: ALWAYS LIGHT (ignore banks, light path only)")
    print("=" * 60)
    model_D = HomeostaticModel(use_banks=True, use_timing=True,
                               use_self_model=False, use_gate=True,
                               always_light=True).to(DEVICE)
    train_D = train_model(model_D, ext, train_loader, EPOCHS, 'D_light', timing_info)
    m_D = evaluate(model_D, ext, test_loader, timing_info)
    print(f"  D: acc={m_D['acc']:.4f}")

    # ━━━ E: Scrambled timing (inverted signal) ━━━
    print("\n" + "=" * 60)
    print("E: SCRAMBLED (trained A with inverted timing at eval)")
    print("=" * 60)
    m_E = evaluate(model_A, ext, test_loader, timing_info, scramble_timing=True)
    print(f"  E: acc={m_E['acc']:.4f}  gate_h={m_E.get('gate_high', 0):.3f}  "
          f"gate_l={m_E.get('gate_low', 0):.3f}")

    # ━━━ F: Self-model ablation (zero weights) ━━━
    print("\n" + "=" * 60)
    print("F: ABLATED SELF-MODEL (A with zeroed self-model)")
    print("=" * 60)
    import copy
    model_F = copy.deepcopy(model_A)
    ablate_self_model(model_F)
    m_F = evaluate(model_F, ext, test_loader, timing_info)
    print(f"  F: acc={m_F['acc']:.4f}  AUROC={m_F['auroc']:.4f}  "
          f"gate_h={m_F.get('gate_high', 0):.3f}  gate_l={m_F.get('gate_low', 0):.3f}")

    elapsed = time.time() - t0
    set_sclk('high')

    # ━━━ Correlations ━━━
    gate_sclk_corr = 0.0
    if train_A['gate_vals'] and train_A['sclk_states']:
        c, _ = stats.pearsonr(train_A['gate_vals'], train_A['sclk_states'])
        gate_sclk_corr = float(c)

    gate_temp_corr = 0.0
    if train_A['gate_vals'] and train_A['temps']:
        c, _ = stats.pearsonr(train_A['gate_vals'], train_A['temps'])
        gate_temp_corr = float(c)

    # Bank divergence
    bank_cos = []
    if model_A.use_banks:
        W = model_A.bank_w.data
        for i in range(NUM_BANKS):
            for j in range(i+1, NUM_BANKS):
                sim = F.cosine_similarity(W[i].flatten().unsqueeze(0),
                                          W[j].flatten().unsqueeze(0)).item()
                bank_cos.append(sim)

    # ━━━ Tests ━━━
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    tests = {}

    # T1: A > 90%
    t1 = m_A['acc'] > 0.90
    tests['T1_accuracy'] = {'verdict': 'PASS' if t1 else 'FAIL',
        'description': f"A={m_A['acc']*100:.1f}% {'>' if t1 else '<='} 90%"}

    # T2: Self-model AUROC > 0.8
    t2 = m_A['auroc'] > 0.80
    tests['T2_self_model'] = {'verdict': 'PASS' if t2 else 'FAIL',
        'description': f"AUROC={m_A['auroc']:.4f} {'>' if t2 else '<='} 0.80"}

    # T3: Gate differs between states
    t3 = m_A.get('gate_p', 1.0) < 0.01
    tests['T3_gate_adaptive'] = {'verdict': 'PASS' if t3 else 'FAIL',
        'description': f"p={m_A.get('gate_p', 1.0):.6f} {'<' if t3 else '>='} 0.01"}

    # T4: Ablating self-model drops accuracy > 10pp
    gap_AF = m_A['acc'] - m_F['acc']
    t4 = gap_AF > 0.10
    tests['T4_self_model_causal'] = {'verdict': 'PASS' if t4 else 'FAIL',
        'description': f"A-F={gap_AF*100:.1f}pp {'>' if t4 else '<='} 10pp"}

    # T5: A > D + 5pp
    t5 = m_A['acc'] > m_D['acc'] + 0.05
    tests['T5_full_path_needed'] = {'verdict': 'PASS' if t5 else 'FAIL',
        'description': f"A={m_A['acc']*100:.1f}% > D={m_D['acc']*100:.1f}%+5pp"}

    # T6: Scrambled < A - 5pp
    t6 = m_E['acc'] < m_A['acc'] - 0.05
    tests['T6_scrambled_kills'] = {'verdict': 'PASS' if t6 else 'FAIL',
        'description': f"E={m_E['acc']*100:.1f}% < A={m_A['acc']*100:.1f}%-5pp"}

    # T7: A-B gap > 30pp
    gap_AB = m_A['acc'] - m_B['acc']
    t7 = gap_AB > 0.30
    tests['T7_embodiment_gap'] = {'verdict': 'PASS' if t7 else 'FAIL',
        'description': f"A-B={gap_AB*100:.1f}pp {'>' if t7 else '<='} 30pp"}

    # T8: Gate-SCLK correlation
    t8 = abs(gate_sclk_corr) > 0.3
    tests['T8_gate_corr'] = {'verdict': 'PASS' if t8 else 'FAIL',
        'description': f"|r(gate,sclk)|={abs(gate_sclk_corr):.4f} {'>' if t8 else '<='} 0.3"}

    pass_count = sum(1 for t in tests.values() if t['verdict'] == 'PASS')
    verdict = f"{pass_count}/{len(tests)} PASS"

    for name, result in tests.items():
        print(f"  {result['verdict']:4s} | {name}: {result['description']}")
    print(f"\n  VERDICT: {verdict}")

    # ━━━ Save Results ━━━
    results = {
        'experiment': 'z2059_homeostatic_consciousness',
        'version': 2,
        'scientific_basis': {
            'biological_computationalism': 'Milinkovic & Aru 2026',
            'homeostatic_regulation': 'Solms/Conscium',
            'self_referential_loop': 'Hofstadter/Anthropic introspection',
        },
        'timing': timing_info,
        'wall_clock_ratio': round(wc_ratio, 2),
        'wgp_values': unique_wgps,
        'accuracies': {
            'A_homeostatic': round(m_A['acc'], 4),
            'B_blind': round(m_B['acc'], 4),
            'C_no_self_model': round(m_C['acc'], 4),
            'D_always_light': round(m_D['acc'], 4),
            'E_scrambled': round(m_E['acc'], 4),
            'F_ablated_self': round(m_F['acc'], 4),
        },
        'self_model_auroc': round(m_A['auroc'], 4),
        'self_model_auroc_ablated': round(m_F['auroc'], 4),
        'gate': {
            'high_state': round(m_A.get('gate_high', 0.5), 4),
            'low_state': round(m_A.get('gate_low', 0.5), 4),
            'diff_p_value': round(m_A.get('gate_p', 1.0), 6),
            'sclk_correlation': round(gate_sclk_corr, 4),
            'temp_correlation': round(gate_temp_corr, 4),
            'ablated_high': round(m_F.get('gate_high', 0.5), 4),
            'ablated_low': round(m_F.get('gate_low', 0.5), 4),
        },
        'bank_cos_sim_mean': round(float(np.mean(bank_cos)), 4) if bank_cos else None,
        'tests': tests,
        'verdict': verdict,
        'pass_count': pass_count,
        'elapsed_s': round(elapsed),
        'strange_loop': (
            'Hardware timing → Combined representation → Self-model predicts SCLK → '
            'Gate modulates full/light path → Output quality depends on correct gate → '
            'Training improves self-model → More accurate gate'
        ),
        'notes': (
            'v2: Added wall-clock timing as model input. Self-model now introspects on '
            'combined (image + timing) representation. Ablating self-model should break '
            'gate adaptation while preserving task ability through light path.'
        ),
    }

    out_path = 'results/z2059_homeostatic_consciousness.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"Elapsed: {elapsed:.0f}s")


if __name__ == '__main__':
    main()
