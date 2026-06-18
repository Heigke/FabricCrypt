#!/usr/bin/env python3
"""z2060: Exclusive Homeostatic Consciousness

Fixes z2059 T4 failure by making the self-model CAUSALLY NECESSARY.

Key architectural change from z2059:
  Full path:  bank_w[bank_id] @ h_img → NO timing info (can't distinguish SCLK)
  Light path: h_img only → NO bank info
  Self-model: predicts SCLK from h = concat(h_img, h_hw) → the ONLY timing-aware component
  Gate: sigmoid(self_pred) → routes to correct path

Task design (conflicting label schemes):
  SCLK high: label = (digit + {3,5}[bank_parity]) % 10  (needs banks → full path)
  SCLK low:  label = (9 - digit) % 10  (digit reversal → light path)

Why this fixes T4:
  - Full path learns bank-shifted labels only (gate≈1 during high SCLK training)
  - Light path learns reversed labels only (gate≈0 during low SCLK training)
  - Constant gate (ablated self-model) blends conflicting predictions → ~50% accuracy
  - Predicted drop: A(~97%) → F_ablated(~55%) = ~42pp gap >> 10pp threshold

Tests (8):
  T1: A_homeostatic > 90% accuracy
  T2: Self-model AUROC > 0.8
  T3: Gate differs between SCLK states (p < 0.01)
  T4: Ablating self-model → accuracy drops > 10pp  ← THE KEY TEST
  T5: A > D_always_light + 5pp
  T6: E_scrambled < A - 5pp
  T7: A - B_blind gap > 30pp
  T8: Gate-SCLK correlation |r| > 0.3
"""
import torch, torch.nn as nn, torch.nn.functional as F
import os, sys, json, time, copy, numpy as np
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score
from scipy import stats

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')
os.environ.setdefault('PYTORCH_ROCM_ARCH', 'gfx1100')
from torch.utils.cpp_extension import load_inline

DEVICE = 'cuda'
BS = 256
EPOCHS = 15
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
            if os.path.exists(p): return int(open(p).read().strip()) / 1000.0
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
        times = [measure_wall_clock(ext) for _ in range(30)]
        info[mode] = {'mean': float(np.mean(times)), 'std': float(np.std(times)),
                      'sclk': read_sclk(), 'temp': read_temp()}
        print(f"  {mode}: {info[mode]['mean']:.4f} +/- {info[mode]['std']:.4f} ms  "
              f"SCLK={info[mode]['sclk']}  T={info[mode]['temp']:.1f}C")
    ratio = info['low']['mean'] / max(info['high']['mean'], 1e-6)
    print(f"  Wall-clock ratio: {ratio:.2f}x")
    set_sclk('high')
    return info, ratio

def norm_timing(wall_ms, timing_info):
    lo = timing_info['high']['mean']
    hi = timing_info['low']['mean']
    return max(0., min(1., (wall_ms - lo) / max(hi - lo, 1e-6)))


# ━━━ Model ━━━
class ExclusiveHomeostaticModel(nn.Module):
    """
    Model with EXCLUSIVE path specialization.

    Architecture:
      encoder(image) → h_img [B, 128]
      timing_proj(wall_clock) → h_hw [B, 32]
      h_combined = concat(h_img, h_hw) → [B, 160]

      Full path:  bank_w[bank_id] @ h_img → h_banked [B, 128]  (NO timing!)
      Light path: h_img [B, 128]  (NO banks!)

      Self-model: h_combined → P(SCLK_high)  (ONLY timing-aware component)
      Gate: sigmoid(self_pred) → blend factor

      output = gate * head_full(h_banked) + (1-gate) * head_light(h_img)

    Key difference from z2059: timing does NOT enter bank transform.
    Full path cannot distinguish SCLK states → specializes on one label scheme.
    Light path has no banks → specializes on the other label scheme.
    Self-model → gate is the ONLY way to select the correct path.
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

        # Image encoder → h_img [B, 128]
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64*7*7, 128), nn.ReLU())

        # Timing projection → h_hw [B, 32]
        if use_timing:
            self.timing_proj = nn.Sequential(
                nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 32), nn.ReLU())

        # Per-WGP banks: [N, 128, 128] — transforms h_img ONLY (no timing!)
        if use_banks:
            self.bank_w = nn.Parameter(torch.randn(NUM_BANKS, 128, 128) * 0.02)

        # Self-model: predict SCLK from combined h = [h_img, h_hw]
        sm_dim = 160 if use_timing else 128
        if use_self_model:
            self.self_model = nn.Sequential(
                nn.Linear(sm_dim, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, 1))

        # Gate: from self-model prediction → blend factor
        if use_gate:
            self.gate_net = nn.Sequential(
                nn.Linear(1, 16), nn.ReLU(),
                nn.Linear(16, 1), nn.Sigmoid())

        # Full path head: from h_banked (bank-specific features)
        self.head_full = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
        # Light path head: from h_img (generic features)
        self.head_light = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x, bank_ids=None, timing=None):
        h_img = self.encoder(x)  # [B, 128]

        # Build combined representation (for self-model ONLY)
        if self.use_timing and timing is not None:
            h_hw = self.timing_proj(timing.view(-1, 1))  # [B, 32]
            h_combined = torch.cat([h_img, h_hw], dim=1)  # [B, 160]
        elif self.use_timing:
            h_combined = torch.cat([h_img, torch.zeros(h_img.shape[0], 32, device=h_img.device)], 1)
        else:
            h_combined = h_img

        # Self-model: introspect combined representation
        self_pred = None
        if self.use_self_model:
            self_pred = self.self_model(h_combined)  # [B, 1]

        # Gate
        if self.use_gate and not self.always_full and not self.always_light:
            if self.use_self_model and self_pred is not None:
                gate = self.gate_net(torch.sigmoid(self_pred))
            else:
                gate = self.gate_net(torch.full((h_img.shape[0], 1), 0.5, device=h_img.device))
        elif self.always_full:
            gate = torch.ones(h_img.shape[0], 1, device=h_img.device)
        elif self.always_light:
            gate = torch.zeros(h_img.shape[0], 1, device=h_img.device)
        else:
            gate = torch.full((h_img.shape[0], 1), 0.5, device=h_img.device)

        # Full path: bank transform on h_img ONLY (no timing!)
        if self.use_banks and bank_ids is not None:
            h_banked = torch.bmm(self.bank_w[bank_ids], h_img.unsqueeze(-1)).squeeze(-1)
            logits_full = self.head_full(h_banked)
        else:
            logits_full = self.head_full(h_img)

        # Light path: from h_img directly
        logits_light = self.head_light(h_img)

        # Blend: gate selects full vs light
        logits = gate * logits_full + (1 - gate) * logits_light

        return {'logits': logits, 'self_pred': self_pred, 'gate': gate}


# ━━━ Labels: CONFLICTING schemes ━━━
def make_labels(digits, bank_ids, sclk_is_high):
    """
    SCLK high: bank-dependent shift (needs banks → full path)
    SCLK low:  digit reversal (no banks needed → light path)

    These schemes CONFLICT: same digit → different labels per state.
    Only correct gate selection can solve both.
    """
    labels = digits.clone()
    if sclk_is_high:
        even = (bank_ids % 2 == 0)
        labels[even] = (digits[even] + 3) % 10
        labels[~even] = (digits[~even] + 5) % 10
    else:
        labels = (9 - digits) % 10
    return labels


def get_data():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    tr = datasets.MNIST('data', train=True, download=True, transform=tf)
    te = datasets.MNIST('data', train=False, transform=tf)
    return (torch.utils.data.DataLoader(tr, batch_size=BS, shuffle=True, drop_last=True),
            torch.utils.data.DataLoader(te, batch_size=BS, shuffle=False, drop_last=True))


# ━━━ Training ━━━
def train_model(model, ext, loader, epochs, name, timing_info, dvfs=True):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    gate_vals, sclk_states, temps = [], [], []
    sclk_high = True
    bn = 0

    for ep in range(epochs):
        tot_loss, correct, total = 0, 0, 0
        for imgs, digits in loader:
            imgs, digits = imgs.to(DEVICE), digits.to(DEVICE)

            if dvfs and bn % SWITCH_EVERY == 0:
                sclk_high = not sclk_high
                set_sclk('high' if sclk_high else 'low')
                time.sleep(0.05)

            wall_ms = measure_wall_clock(ext)
            timing_norm = norm_timing(wall_ms, timing_info)
            timing_t = torch.full((BS,), timing_norm, device=DEVICE)

            wgps = ext.probe(BS)[0]
            bank_ids = (wgps // 2).long().clamp(0, NUM_BANKS - 1)

            labels = make_labels(digits, bank_ids, sclk_high)

            out = model(imgs, bank_ids=bank_ids, timing=timing_t)

            # Task loss
            task_loss = F.cross_entropy(out['logits'], labels)

            # Self-model loss
            self_loss = torch.tensor(0.0, device=DEVICE)
            if out['self_pred'] is not None:
                self_target = torch.full((BS, 1), float(sclk_high), device=DEVICE)
                self_loss = F.binary_cross_entropy_with_logits(out['self_pred'], self_target)

            # Homeostatic loss: encourage correct gate (small regularizer)
            homeo_loss = torch.tensor(0.0, device=DEVICE)
            if model.use_gate and not model.always_full and not model.always_light:
                g = out['gate']
                if sclk_high:
                    homeo_loss = ((1 - g) ** 2).mean()
                else:
                    homeo_loss = (g ** 2).mean()

            loss = task_loss + 0.1 * self_loss + 0.05 * homeo_loss

            opt.zero_grad(); loss.backward(); opt.step()

            tot_loss += loss.item()
            pred = out['logits'].argmax(1)
            correct += (pred == labels).sum().item()
            total += BS
            gate_vals.append(out['gate'].mean().item())
            sclk_states.append(float(sclk_high))
            temps.append(read_temp())
            bn += 1

        if ep % 3 == 0 or ep == epochs - 1:
            print(f"  [{name}] Ep {ep}: loss={tot_loss/len(loader):.4f} "
                  f"acc={correct/total:.4f} gate={np.mean(gate_vals[-50:]):.3f}")

    return {'gate_vals': gate_vals, 'sclk_states': sclk_states, 'temps': temps}


# ━━━ Evaluation ━━━
def evaluate(model, ext, loader, timing_info, dvfs=True, scramble_timing=False):
    model.eval()
    by_state = {s: {'correct': 0, 'total': 0, 'gates': [], 'self_preds': [], 'labels': []}
                for s in ['high', 'low']}
    sclk_high = True
    bn = 0

    with torch.no_grad():
        for imgs, digits in loader:
            imgs, digits = imgs.to(DEVICE), digits.to(DEVICE)

            if dvfs and bn % SWITCH_EVERY == 0:
                sclk_high = not sclk_high
                set_sclk('high' if sclk_high else 'low')
                time.sleep(0.05)

            wall_ms = measure_wall_clock(ext)
            tn = norm_timing(wall_ms, timing_info)
            if scramble_timing:
                tn = 1.0 - tn
            timing_t = torch.full((BS,), tn, device=DEVICE)

            wgps = ext.probe(BS)[0]
            bank_ids = (wgps // 2).long().clamp(0, NUM_BANKS - 1)
            labels = make_labels(digits, bank_ids, sclk_high)

            out = model(imgs, bank_ids=bank_ids, timing=timing_t)

            sk = 'high' if sclk_high else 'low'
            pred = out['logits'].argmax(1)
            by_state[sk]['correct'] += (pred == labels).sum().item()
            by_state[sk]['total'] += BS
            by_state[sk]['gates'].extend(out['gate'].squeeze().cpu().tolist())

            if out['self_pred'] is not None:
                by_state[sk]['self_preds'].extend(
                    torch.sigmoid(out['self_pred']).squeeze().cpu().tolist())
                by_state[sk]['labels'].extend([float(sclk_high)] * BS)

            bn += 1

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
        _, p_val = stats.ttest_ind(g_h, g_l)
        m['gate_p'] = float(p_val)
    else:
        m['gate_p'] = 1.0

    return m


def ablate_self_model(model):
    if hasattr(model, 'self_model'):
        for p in model.self_model.parameters():
            p.data.zero_()


# ━━━ Main ━━━
def main():
    print("=" * 70)
    print("z2060: Exclusive Homeostatic Consciousness")
    print("=" * 70)
    print()
    print("Key fix from z2059: Full path sees h_img ONLY (no timing).")
    print("Label schemes CONFLICT: high=bank-shifted, low=digit-reversed.")
    print("Self-model → gate is the ONLY way to select correct path.")
    print()

    t0 = time.time()

    print("Compiling HIP kernel...")
    ext = load_inline(name='z2060_excl', cpp_sources=CPP_SRC, cuda_sources=HIP_SRC,
                      functions=['probe'], extra_cuda_cflags=['-O2', '--offload-arch=gfx1100'],
                      verbose=False)

    wgps = ext.probe(1024)[0].cpu().numpy()
    unique_wgps = sorted(set(wgps.tolist()))
    print(f"WGP distribution: {len(unique_wgps)} unique values: {unique_wgps}")

    timing_info, wc_ratio = characterize_sclk(ext)
    train_loader, test_loader = get_data()

    # ━━━ A: Full Homeostatic ━━━
    print(f"\n{'='*60}\nA: HOMEOSTATIC (banks + timing + self-model + gate)\n{'='*60}")
    model_A = ExclusiveHomeostaticModel(
        use_banks=True, use_timing=True, use_self_model=True, use_gate=True).to(DEVICE)
    train_A = train_model(model_A, ext, train_loader, EPOCHS, 'A_homeo', timing_info)
    m_A = evaluate(model_A, ext, test_loader, timing_info)
    print(f"  A: acc={m_A['acc']:.4f} (high={m_A.get('acc_high',0):.4f} low={m_A.get('acc_low',0):.4f})")
    print(f"     AUROC={m_A['auroc']:.4f}  gate_h={m_A.get('gate_high',0):.3f}  gate_l={m_A.get('gate_low',0):.3f}")

    # ━━━ B: Blind ━━━
    print(f"\n{'='*60}\nB: BLIND (no hardware signals)\n{'='*60}")
    model_B = ExclusiveHomeostaticModel(
        use_banks=False, use_timing=False, use_self_model=False, use_gate=False).to(DEVICE)
    train_model(model_B, ext, train_loader, EPOCHS, 'B_blind', timing_info)
    m_B = evaluate(model_B, ext, test_loader, timing_info)
    print(f"  B: acc={m_B['acc']:.4f}")

    # ━━━ C: No self-model (gate=constant) ━━━
    print(f"\n{'='*60}\nC: NO SELF-MODEL (banks + timing, gate=constant)\n{'='*60}")
    model_C = ExclusiveHomeostaticModel(
        use_banks=True, use_timing=True, use_self_model=False, use_gate=True).to(DEVICE)
    train_model(model_C, ext, train_loader, EPOCHS, 'C_no_sm', timing_info)
    m_C = evaluate(model_C, ext, test_loader, timing_info)
    print(f"  C: acc={m_C['acc']:.4f}")

    # ━━━ D: Always light ━━━
    print(f"\n{'='*60}\nD: ALWAYS LIGHT (ignore banks)\n{'='*60}")
    model_D = ExclusiveHomeostaticModel(
        use_banks=True, use_timing=True, use_self_model=False, use_gate=True,
        always_light=True).to(DEVICE)
    train_model(model_D, ext, train_loader, EPOCHS, 'D_light', timing_info)
    m_D = evaluate(model_D, ext, test_loader, timing_info)
    print(f"  D: acc={m_D['acc']:.4f}")

    # ━━━ E: Scrambled timing ━━━
    print(f"\n{'='*60}\nE: SCRAMBLED (A with inverted timing at eval)\n{'='*60}")
    m_E = evaluate(model_A, ext, test_loader, timing_info, scramble_timing=True)
    print(f"  E: acc={m_E['acc']:.4f}")

    # ━━━ F: Self-model ablation ━━━
    print(f"\n{'='*60}\nF: ABLATED SELF-MODEL (A with zeroed self-model)\n{'='*60}")
    model_F = copy.deepcopy(model_A)
    ablate_self_model(model_F)
    m_F = evaluate(model_F, ext, test_loader, timing_info)
    print(f"  F: acc={m_F['acc']:.4f} AUROC={m_F['auroc']:.4f} "
          f"gate_h={m_F.get('gate_high',0):.3f} gate_l={m_F.get('gate_low',0):.3f}")

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
    print(f"\n{'='*70}\nTEST RESULTS\n{'='*70}")
    tests = {}

    t1 = m_A['acc'] > 0.90
    tests['T1_accuracy'] = {'verdict': 'PASS' if t1 else 'FAIL',
        'description': f"A={m_A['acc']*100:.1f}% {'>' if t1 else '<='} 90%"}

    t2 = m_A['auroc'] > 0.80
    tests['T2_self_model'] = {'verdict': 'PASS' if t2 else 'FAIL',
        'description': f"AUROC={m_A['auroc']:.4f} {'>' if t2 else '<='} 0.80"}

    t3 = m_A.get('gate_p', 1.0) < 0.01
    tests['T3_gate_adaptive'] = {'verdict': 'PASS' if t3 else 'FAIL',
        'description': f"p={m_A.get('gate_p',1.0):.6f} {'<' if t3 else '>='} 0.01"}

    gap_AF = m_A['acc'] - m_F['acc']
    t4 = gap_AF > 0.10
    tests['T4_self_model_causal'] = {'verdict': 'PASS' if t4 else 'FAIL',
        'description': f"A-F={gap_AF*100:.1f}pp {'>' if t4 else '<='} 10pp"}

    t5 = m_A['acc'] > m_D['acc'] + 0.05
    tests['T5_full_path_needed'] = {'verdict': 'PASS' if t5 else 'FAIL',
        'description': f"A={m_A['acc']*100:.1f}% > D={m_D['acc']*100:.1f}%+5pp"}

    t6 = m_E['acc'] < m_A['acc'] - 0.05
    tests['T6_scrambled_kills'] = {'verdict': 'PASS' if t6 else 'FAIL',
        'description': f"E={m_E['acc']*100:.1f}% < A={m_A['acc']*100:.1f}%-5pp"}

    gap_AB = m_A['acc'] - m_B['acc']
    t7 = gap_AB > 0.30
    tests['T7_embodiment_gap'] = {'verdict': 'PASS' if t7 else 'FAIL',
        'description': f"A-B={gap_AB*100:.1f}pp {'>' if t7 else '<='} 30pp"}

    t8 = abs(gate_sclk_corr) > 0.3
    tests['T8_gate_corr'] = {'verdict': 'PASS' if t8 else 'FAIL',
        'description': f"|r(gate,sclk)|={abs(gate_sclk_corr):.4f} {'>' if t8 else '<='} 0.3"}

    pass_count = sum(1 for t in tests.values() if t['verdict'] == 'PASS')
    verdict = f"{pass_count}/{len(tests)} PASS"

    for name, result in tests.items():
        s = 'PASS' if result['verdict'] == 'PASS' else 'FAIL'
        print(f"  {s:4s} | {name}: {result['description']}")
    print(f"\n  VERDICT: {verdict}")

    # Per-state accuracy breakdown
    print(f"\n  Path specialization:")
    print(f"    A high SCLK (needs full path): {m_A.get('acc_high',0)*100:.1f}%")
    print(f"    A low SCLK (needs light path): {m_A.get('acc_low',0)*100:.1f}%")
    print(f"    F high SCLK (ablated):         {m_F.get('acc_high',0)*100:.1f}%")
    print(f"    F low SCLK (ablated):          {m_F.get('acc_low',0)*100:.1f}%")

    # ━━━ Save ━━━
    results = {
        'experiment': 'z2060_exclusive_homeostatic',
        'version': 1,
        'fixes': 'z2059 T4 failure — removes timing from bank transform, uses conflicting label schemes',
        'scientific_basis': {
            'biological_computationalism': 'Milinkovic & Aru 2026',
            'homeostatic_regulation': 'Solms/Conscium',
            'self_referential_loop': 'Hofstadter/Anthropic introspection',
            'exclusive_specialization': 'Each path handles one label scheme, gate selects',
        },
        'timing': timing_info,
        'wall_clock_ratio': round(wc_ratio, 2),
        'wgp_values': unique_wgps,
        'accuracies': {
            'A_homeostatic': round(m_A['acc'], 4),
            'A_high': round(m_A.get('acc_high', 0), 4),
            'A_low': round(m_A.get('acc_low', 0), 4),
            'B_blind': round(m_B['acc'], 4),
            'C_no_self_model': round(m_C['acc'], 4),
            'D_always_light': round(m_D['acc'], 4),
            'E_scrambled': round(m_E['acc'], 4),
            'F_ablated_self': round(m_F['acc'], 4),
            'F_high': round(m_F.get('acc_high', 0), 4),
            'F_low': round(m_F.get('acc_low', 0), 4),
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
        'architecture_note': (
            'Full path: bank_w[bank_id] @ h_img (NO timing). '
            'Light path: h_img only. '
            'Self-model → gate is the ONLY timing-aware routing. '
            'Ablating self-model → constant gate → conflicting path outputs → accuracy drops.'
        ),
    }

    out_path = 'results/z2060_exclusive_homeostatic.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"Elapsed: {elapsed:.0f}s")


if __name__ == '__main__':
    main()
