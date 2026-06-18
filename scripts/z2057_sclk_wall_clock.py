#!/usr/bin/env python3
"""z2057: SCLK Wall-Clock — Dual-Channel Hardware Embodiment

Two independent hardware channels:
1. DIGITAL: s_getreg_b32 hwreg(23) → per-block WGP bank identity
2. ANALOG: Wall-clock kernel timing (CUDA events) → GPU SCLK frequency state

Key physics: clock64() counts at SCLK rate → cycles are SCLK-invariant (falsified).
             But CUDA event wall-clock scales with 1/SCLK → genuine physics signal.
             600 MHz vs ~1800 MHz → ~3x wall-clock ratio.

Label offsets require BOTH channels:
  (even_bank, low_SCLK)→0   (even_bank, high_SCLK)→3
  (odd_bank,  low_SCLK)→5   (odd_bank,  high_SCLK)→7
Without both → blind ceiling ~25%, bank-only ceiling ~50%.

Tests:
  T1: A_embodied > 90%
  T2: A - B_blind gap > 30pp
  T3: Bank weight cos_sim < 0.9
  T4: Scrambled timing drops > 10pp
  T5: No-timing < A by > 10pp
  T6: Wall-clock ratio > 2.0x
  T7: Timing gate > 0.3
  T8: ≥3 distinct WGP values
"""
import torch, torch.nn as nn, torch.nn.functional as F
import os, sys, json, time, numpy as np
from torchvision import datasets, transforms

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')
os.environ.setdefault('PYTORCH_ROCM_ARCH', 'gfx1100')
from torch.utils.cpp_extension import load_inline

DEVICE = 'cuda'
BS = 256
EPOCHS = 10
SWITCH_EVERY = 25
NUM_BANKS = 8
OFFSETS = {(0, 0): 0, (0, 1): 3, (1, 0): 5, (1, 1): 7}

# ━━━ HIP Kernel ━━━
HIP_SRC = r'''
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void bank_probe(int* bank_ids, float* work, int n) {
    int bid = blockIdx.x;
    if (bid >= n || threadIdx.x != 0) return;
    uint32_t hw;
    asm volatile("s_getreg_b32 %0, hwreg(23)" : "=s"(hw));
    bank_ids[bid] = (int)((hw >> 7) & 0xF);
    // Deliberate compute work for wall-clock signal (~170k cycles)
    float acc = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < 5000; i++) acc += 1.0f / (float)(i+1);
    work[bid] = acc;
}

std::vector<torch::Tensor> probe(int n) {
    auto io = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto fo = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto b = torch::zeros({n}, io);
    auto w = torch::zeros({n}, fo);
    bank_probe<<<n, 32>>>(b.data_ptr<int>(), w.data_ptr<float>(), n);
    return {b, w};
}
'''

# ━━━ DVFS Control ━━━
DPM = '/sys/class/drm/card0/device/power_dpm_force_performance_level'
SCLK_F = '/sys/class/drm/card0/device/pp_dpm_sclk'

def read_sclk():
    try: return [l.strip() for l in open(SCLK_F) if '*' in l][0]
    except: return '?'

def set_sclk(mode):
    try:
        open(DPM, 'w').write('low' if mode == 'low' else 'high')
        time.sleep(0.3)
        return True
    except: return False

def normalize_timing(wall_ms, info):
    lo, hi = info['high']['mean'], info['low']['mean']
    return max(0., min(1., (wall_ms - lo) / max(hi - lo, 1e-6)))

def characterize(ext, n=BS):
    print("\n--- SCLK Characterization ---")
    info = {}
    for mode in ['low', 'high']:
        set_sclk(mode); time.sleep(0.5)
        for _ in range(10): ext.probe(n); torch.cuda.synchronize()
        times = []
        for _ in range(50):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); ext.probe(n); e.record(); torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        info[mode] = {'mean': float(np.mean(times)), 'std': float(np.std(times)),
                      'sclk': read_sclk()}
        print(f"  {mode}: {info[mode]['mean']:.4f} +/- {info[mode]['std']:.4f} ms  SCLK={info[mode]['sclk']}")
    ratio = info['low']['mean'] / max(info['high']['mean'], 1e-6)
    print(f"  Wall-clock ratio: {ratio:.2f}x")
    set_sclk('high')
    return info, ratio

# ━━━ Model ━━━
class DualChannelModel(nn.Module):
    def __init__(self, use_banks=True, use_timing=True):
        super().__init__()
        self.ub, self.ut = use_banks, use_timing
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64*7*7, 128), nn.ReLU())
        if use_banks:
            self.bank_w = nn.Parameter(torch.randn(NUM_BANKS, 128, 128) * 0.02)
        d = 128
        if use_timing:
            self.t_emb = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU())
            self.gate = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
            d += 64
        self.head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x, banks=None, timing=None):
        h = self.encoder(x)
        if self.ub and banks is not None:
            h = torch.bmm(self.bank_w[banks], h.unsqueeze(-1)).squeeze(-1)
        g = None
        if self.ut:
            if timing is not None:
                t = timing.view(-1, 1)
                g = self.gate(t)
                h = torch.cat([h, g * self.t_emb(t)], 1)
            else:
                h = torch.cat([h, torch.zeros(h.shape[0], 64, device=h.device)], 1)
        return self.head(h), g

# ━━━ Labels ━━━
def make_labels(digits, bank_ids, sclk_state):
    par = (bank_ids % 2).long()
    off = torch.zeros_like(digits)
    off[par == 0] = OFFSETS[(0, sclk_state)]
    off[par == 1] = OFFSETS[(1, sclk_state)]
    return (digits + off) % 10

# ━━━ Data ━━━
def get_data():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    tr = datasets.MNIST('data', train=True, download=True, transform=tf)
    te = datasets.MNIST('data', train=False, transform=tf)
    return (torch.utils.data.DataLoader(tr, batch_size=BS, shuffle=True, drop_last=True),
            torch.utils.data.DataLoader(te, batch_size=BS, shuffle=False, drop_last=True))

# ━━━ Training ━━━
def train_model(model, loader, ext, info, epochs=EPOCHS, label='model'):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(epochs):
        model.train()
        tot_l, cor, tot, gates = 0, 0, 0, []
        sclk_st = ep % 2
        set_sclk('low' if sclk_st == 0 else 'high')
        for bi, (imgs, digs) in enumerate(loader):
            if bi > 0 and bi % SWITCH_EVERY == 0:
                sclk_st = 1 - sclk_st
                set_sclk('low' if sclk_st == 0 else 'high')
            B = imgs.shape[0]
            imgs = imgs.to(DEVICE)
            # Probe + wall-clock timing
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); banks_raw, _ = ext.probe(B); e.record(); torch.cuda.synchronize()
            wall_ms = s.elapsed_time(e)
            bids = (banks_raw // 2).clamp(0, NUM_BANKS-1).to(DEVICE)
            t_feat = torch.full((B,), normalize_timing(wall_ms, info), device=DEVICE)
            labels = make_labels(digs, bids.cpu(), sclk_st).to(DEVICE)
            logits, g = model(imgs, bids if model.ub else None, t_feat if model.ut else None)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            tot_l += loss.item() * B; cor += (logits.argmax(1) == labels).sum().item(); tot += B
            if g is not None: gates.append(g.mean().item())
        acc = cor / tot
        g_avg = np.mean(gates) if gates else 0
        print(f"  [{label}] ep {ep}: loss={tot_l/tot:.4f} acc={acc:.4f} gate={g_avg:.3f}")
    return model

# ━━━ Evaluation ━━━
def eval_model(model, loader, ext, info,
               give_banks=True, give_timing=True,
               scramble_timing=False, scramble_banks=False, constant_timing=None):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for sclk_st in [0, 1]:
            set_sclk('low' if sclk_st == 0 else 'high')
            time.sleep(0.3)
            for imgs, digs in loader:
                B = imgs.shape[0]
                imgs = imgs.to(DEVICE)
                # Always probe (needed for labels)
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record(); banks_raw, _ = ext.probe(B); e.record(); torch.cuda.synchronize()
                wall_ms = s.elapsed_time(e)
                real_bids = (banks_raw // 2).clamp(0, NUM_BANKS - 1)
                labels = make_labels(digs, real_bids.cpu(), sclk_st).to(DEVICE)
                # Model bank input
                if give_banks:
                    bids = ((real_bids + 3) % NUM_BANKS if scramble_banks else real_bids).to(DEVICE)
                else:
                    bids = None
                # Model timing input
                if give_timing:
                    if constant_timing is not None:
                        tfeat = torch.full((B,), constant_timing, device=DEVICE)
                    elif scramble_timing:
                        # Invert: low SCLK gets 0.0 (looks like high), high gets 1.0 (looks like low)
                        tfeat = torch.full((B,), float(sclk_st), device=DEVICE)
                    else:
                        tfeat = torch.full((B,), normalize_timing(wall_ms, info), device=DEVICE)
                else:
                    tfeat = None
                logits, _ = model(imgs, bids, tfeat)
                correct += (logits.argmax(1) == labels).sum().item()
                total += B
    return correct / total

# ━━━ Main ━━━
def main():
    print("=== z2057: SCLK Wall-Clock Dual-Channel Embodiment ===\n")
    t0 = time.time()

    print("Compiling HIP kernel...")
    ext = load_inline(name='z2057_sclk', cpp_sources='std::vector<torch::Tensor> probe(int n);',
                      cuda_sources=HIP_SRC, functions=['probe'], verbose=False,
                      extra_cuda_cflags=['--offload-arch=gfx1100', '-O2'])

    info, wc_ratio = characterize(ext)

    # WGP coverage
    banks_raw, _ = ext.probe(1024); torch.cuda.synchronize()
    wgps = sorted(set((banks_raw.cpu().numpy()).tolist()))
    print(f"  WGP values: {wgps} ({len(wgps)} unique)")

    train_ld, test_ld = get_data()

    # ─── Train models ───
    print("\n--- Training A_embodied (banks + timing) ---")
    model_a = DualChannelModel(True, True).to(DEVICE)
    train_model(model_a, train_ld, ext, info, label='A')

    print("\n--- Training B_blind (no hw info) ---")
    model_b = DualChannelModel(False, False).to(DEVICE)
    train_model(model_b, train_ld, ext, info, label='B')

    print("\n--- Training C_bank_only (banks, no timing) ---")
    model_c = DualChannelModel(True, False).to(DEVICE)
    train_model(model_c, train_ld, ext, info, label='C')

    # ─── Evaluate ───
    print("\n--- Evaluation ---")
    set_sclk('high')

    acc_a = eval_model(model_a, test_ld, ext, info, True, True)
    print(f"  A_embodied:        {acc_a:.4f}")

    acc_b = eval_model(model_b, test_ld, ext, info, False, False)
    print(f"  B_blind:           {acc_b:.4f}")

    acc_c = eval_model(model_c, test_ld, ext, info, True, False)
    print(f"  C_bank_only:       {acc_c:.4f}")

    acc_no_t = eval_model(model_a, test_ld, ext, info, True, True, constant_timing=0.5)
    print(f"  D_no_timing (A):   {acc_no_t:.4f}")

    acc_scr_t = eval_model(model_a, test_ld, ext, info, True, True, scramble_timing=True)
    print(f"  E_scrambled_t (A): {acc_scr_t:.4f}")

    acc_scr_b = eval_model(model_a, test_ld, ext, info, True, True, scramble_banks=True)
    print(f"  F_scrambled_b (A): {acc_scr_b:.4f}")

    acc_scr_bt = eval_model(model_a, test_ld, ext, info, True, True,
                             scramble_timing=True, scramble_banks=True)
    print(f"  G_scrambled_both:  {acc_scr_bt:.4f}")

    # ─── Bank weight analysis ───
    mean_cos = 0.0
    if model_a.ub:
        W = model_a.bank_w.detach().cpu()
        cos_sims = []
        for i in range(NUM_BANKS):
            for j in range(i+1, NUM_BANKS):
                cs = F.cosine_similarity(W[i].flatten().unsqueeze(0),
                                         W[j].flatten().unsqueeze(0)).item()
                cos_sims.append(cs)
        mean_cos = float(np.mean(cos_sims))
        print(f"\n  Bank weight cos_sim: {mean_cos:.3f}")

    # ─── Gate analysis ───
    gate_val = None
    g_low = g_high = 0.0
    if model_a.ut:
        model_a.eval()
        with torch.no_grad():
            g_low = model_a.gate(torch.tensor([[1.0]], device=DEVICE)).item()
            g_high = model_a.gate(torch.tensor([[0.0]], device=DEVICE)).item()
            gate_val = (g_low + g_high) / 2
        print(f"  Gate at low SCLK (t=1.0): {g_low:.3f}")
        print(f"  Gate at high SCLK (t=0.0): {g_high:.3f}")
        print(f"  Mean gate: {gate_val:.3f}")

    elapsed = time.time() - t0

    # ─── Tests ───
    print("\n=== TEST VERDICTS ===")
    results = {
        'experiment': 'z2057_sclk_wall_clock',
        'timing': info, 'wall_clock_ratio': wc_ratio, 'wgp_values': wgps,
        'accuracies': {
            'A_embodied': acc_a, 'B_blind': acc_b, 'C_bank_only': acc_c,
            'D_no_timing': acc_no_t, 'E_scrambled_timing': acc_scr_t,
            'F_scrambled_bank': acc_scr_b, 'G_scrambled_both': acc_scr_bt
        },
        'bank_cos_sim': mean_cos,
        'gate': {'low_sclk': g_low, 'high_sclk': g_high, 'mean': gate_val} if gate_val else None,
        'elapsed_s': elapsed, 'tests': {}
    }

    def test(name, cond, desc):
        v = "PASS" if cond else "FAIL"
        results['tests'][name] = {'verdict': v, 'description': desc}
        print(f"  {name}: {v} -- {desc}")
        return cond

    t1 = test('T1_embodied_learns', acc_a > 0.90, f'A={acc_a:.1%} > 90%')
    t2 = test('T2_dual_gap', acc_a - acc_b > 0.30, f'A-B={acc_a-acc_b:.1%} > 30pp')
    t3 = test('T3_bank_divergence', mean_cos < 0.9, f'cos={mean_cos:.3f} < 0.9')
    t4 = test('T4_timing_kill', acc_a - acc_scr_t > 0.10, f'A-E={acc_a-acc_scr_t:.1%} > 10pp')
    t5 = test('T5_timing_signal', acc_a - acc_no_t > 0.10, f'A-D={acc_a-acc_no_t:.1%} > 10pp')
    t6 = test('T6_wc_ratio', wc_ratio > 2.0, f'ratio={wc_ratio:.2f} > 2.0')
    t7 = test('T7_gate_active', (gate_val or 0) > 0.3, f'gate={gate_val:.3f} > 0.3')
    t8 = test('T8_wgp_coverage', len(wgps) >= 3, f'{len(wgps)} WGPs >= 3')

    passes = sum([t1, t2, t3, t4, t5, t6, t7, t8])
    verdict = f"{passes}/8 PASS"
    results['verdict'] = verdict
    results['pass_count'] = passes
    hierarchy = f"B={acc_b:.1%} < C={acc_c:.1%} < A={acc_a:.1%}"
    results['hierarchy'] = hierarchy
    print(f"\n  Hierarchy: {hierarchy}")
    print(f"  VERDICT: {verdict}  ({elapsed:.0f}s)")

    out = 'results/z2057_sclk_wall_clock.json'
    with open(out, 'w') as f: json.dump(results, f, indent=2)
    print(f"  Saved: {out}")

    # Restore
    try: open(DPM, 'w').write('auto')
    except: pass
    print("  Restored SCLK to auto")

if __name__ == '__main__':
    main()
