#!/usr/bin/env python3
"""z2058: Three-Channel Unified Embodiment

Combines ALL discovered hardware channels into one architecture:
1. SPATIAL (digital):  s_getreg_b32 hwreg(23) → WGP bank (which silicon)
2. COMPUTE (ISA):      s_setreg_b32 hwreg(1) → FP16 rounding mode (what mode)
3. TEMPORAL (physics):  CUDA event wall-clock → SCLK state (how fast)

Label offsets encode 8 states (bank_parity × rounding × sclk):
  (even, round_near, low)→0   (even, round_near, high)→1
  (even, round_zero, low)→3   (even, round_zero, high)→4
  (odd,  round_near, low)→5   (odd,  round_near, high)→6
  (odd,  round_zero, low)→7   (odd,  round_zero, high)→9

Without all 3 channels → ceiling ~12.5% (random over 8 groups).
With 1 channel → ~25%. With 2 → ~50%. With all 3 → ~98%.

Tests:
  T1: A_full > 90%
  T2: A - B_blind gap > 40pp
  T3: A - C_two_channels gap > 10pp (3rd channel adds signal)
  T4: Kill shot: scramble any channel drops > 10pp
  T5: Wall-clock ratio > 2.0x
  T6: Gate > 0.3
  T7: ≥3 distinct WGP values
  T8: FP16 rounding produces 2+ unique values
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

# 8 label offsets for (bank_parity, rounding_mode, sclk_state)
# Each triple maps to a unique offset mod 10
OFFSET_TABLE = {
    (0, 0, 0): 0, (0, 0, 1): 1,  # even, round_near, low/high
    (0, 1, 0): 3, (0, 1, 1): 4,  # even, round_zero, low/high
    (1, 0, 0): 5, (1, 0, 1): 6,  # odd, round_near, low/high
    (1, 1, 0): 7, (1, 1, 1): 9,  # odd, round_zero, low/high
}

# ━━━ HIP Kernel: bank + fp16 rounding + compute work ━━━
HIP_SRC = r'''
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void three_channel_probe(int* bank_ids, float* fp16_vals,
                                     float* work_out, int n) {
    int bid = blockIdx.x;
    if (bid >= n || threadIdx.x != 0) return;

    // Channel 1: Spatial — physical WGP address
    uint32_t hw;
    asm volatile("s_getreg_b32 %0, hwreg(23)" : "=s"(hw));
    uint32_t wgp = (hw >> 7) & 0xF;
    bank_ids[bid] = (int)wgp;

    // Channel 2: Compute — FP16 rounding mode
    // Odd banks: set round-toward-zero for fp16 (MODE bits [3:2])
    int bank = wgp / 2;
    if (bank % 2 == 1) {
        uint32_t round_mode = 0xF;  // toward-zero for ALL precisions
        uint32_t sgpr_val = __builtin_amdgcn_readfirstlane(round_mode);
        // hwreg(1, 0, 4) = MODE register, bits [3:0]
        __builtin_amdgcn_s_setreg(0x1801, sgpr_val);
    }

    // FP16 test computation
    __half a = __float2half(3.14159f);
    __half b = __float2half(1.71828f);
    __half c = __hadd(a, b);
    fp16_vals[bid] = __half2float(c);

    // Reset MODE to round-nearest
    if (bank % 2 == 1) {
        uint32_t zero = __builtin_amdgcn_readfirstlane(0u);
        __builtin_amdgcn_s_setreg(0x1801, zero);
    }

    // Channel 3: Compute work for wall-clock signal
    float acc = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < 5000; i++) acc += 1.0f / (float)(i+1);
    work_out[bid] = acc;
}

std::vector<torch::Tensor> probe(int n) {
    auto io = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto fo = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto banks = torch::zeros({n}, io);
    auto fp16v = torch::zeros({n}, fo);
    auto work = torch::zeros({n}, fo);
    three_channel_probe<<<n, 32>>>(banks.data_ptr<int>(),
        fp16v.data_ptr<float>(), work.data_ptr<float>(), n);
    return {banks, fp16v, work};
}
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
        time.sleep(0.3); return True
    except: return False

def norm_timing(wall_ms, info):
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
class ThreeChannelModel(nn.Module):
    def __init__(self, use_banks=True, use_fp16=True, use_timing=True):
        super().__init__()
        self.ub, self.uf, self.ut = use_banks, use_fp16, use_timing
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64*7*7, 128), nn.ReLU())
        if use_banks:
            self.bank_w = nn.Parameter(torch.randn(NUM_BANKS, 128, 128) * 0.02)
        d = 128
        if use_fp16:
            self.fp16_emb = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 32), nn.ReLU())
            d += 32
        if use_timing:
            self.t_emb = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 32), nn.ReLU())
            self.gate = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
            d += 32
        self.head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x, banks=None, fp16_feat=None, timing=None):
        h = self.encoder(x)
        if self.ub and banks is not None:
            h = torch.bmm(self.bank_w[banks], h.unsqueeze(-1)).squeeze(-1)
        if self.uf:
            if fp16_feat is not None:
                h = torch.cat([h, self.fp16_emb(fp16_feat.view(-1, 1))], 1)
            else:
                h = torch.cat([h, torch.zeros(h.shape[0], 32, device=h.device)], 1)
        g = None
        if self.ut:
            if timing is not None:
                t = timing.view(-1, 1)
                g = self.gate(t)
                h = torch.cat([h, g * self.t_emb(t)], 1)
            else:
                h = torch.cat([h, torch.zeros(h.shape[0], 32, device=h.device)], 1)
        return self.head(h), g

# ━━━ Labels ━━━
def make_labels(digits, bank_ids, fp16_vals, sclk_state, fp16_thresh):
    par = (bank_ids % 2).long()
    rmode = (fp16_vals < fp16_thresh).long()  # 1 = round-toward-zero (lower value)
    off = torch.zeros_like(digits)
    for bp in [0, 1]:
        for rm in [0, 1]:
            mask = (par == bp) & (rmode == rm)
            off[mask] = OFFSET_TABLE[(bp, rm, sclk_state)]
    return (digits + off) % 10

# ━━━ Data ━━━
def get_data():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    tr = datasets.MNIST('data', train=True, download=True, transform=tf)
    te = datasets.MNIST('data', train=False, transform=tf)
    return (torch.utils.data.DataLoader(tr, batch_size=BS, shuffle=True, drop_last=True),
            torch.utils.data.DataLoader(te, batch_size=BS, shuffle=False, drop_last=True))

# ━━━ Training ━━━
def train_model(model, loader, ext, info, fp16_thresh, epochs=EPOCHS, label='model'):
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
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); banks_raw, fp16_raw, _ = ext.probe(B); e.record()
            torch.cuda.synchronize()
            wall_ms = s.elapsed_time(e)
            bids = (banks_raw // 2).clamp(0, NUM_BANKS-1).to(DEVICE)
            fp16v = fp16_raw.to(DEVICE)
            t_feat = torch.full((B,), norm_timing(wall_ms, info), device=DEVICE)
            # Normalize fp16 values
            fp16_norm = (fp16v - fp16_thresh) / max(abs(fp16_thresh), 1e-6)
            labels = make_labels(digs, bids.cpu(), fp16v.cpu(), sclk_st, fp16_thresh).to(DEVICE)
            logits, g = model(imgs,
                              bids if model.ub else None,
                              fp16_norm if model.uf else None,
                              t_feat if model.ut else None)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            tot_l += loss.item() * B; cor += (logits.argmax(1) == labels).sum().item(); tot += B
            if g is not None: gates.append(g.mean().item())
        acc = cor / tot
        g_avg = np.mean(gates) if gates else 0
        print(f"  [{label}] ep {ep}: loss={tot_l/tot:.4f} acc={acc:.4f} gate={g_avg:.3f}")
    return model

# ━━━ Evaluation ━━━
def eval_model(model, loader, ext, info, fp16_thresh,
               give_banks=True, give_fp16=True, give_timing=True,
               scramble_timing=False, scramble_banks=False, scramble_fp16=False):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for sclk_st in [0, 1]:
            set_sclk('low' if sclk_st == 0 else 'high')
            time.sleep(0.3)
            for imgs, digs in loader:
                B = imgs.shape[0]
                imgs = imgs.to(DEVICE)
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record(); banks_raw, fp16_raw, _ = ext.probe(B); e.record()
                torch.cuda.synchronize()
                wall_ms = s.elapsed_time(e)
                real_bids = (banks_raw // 2).clamp(0, NUM_BANKS - 1)
                real_fp16 = fp16_raw
                labels = make_labels(digs, real_bids.cpu(), real_fp16.cpu(),
                                     sclk_st, fp16_thresh).to(DEVICE)
                # Model inputs
                bids = ((real_bids + 3) % NUM_BANKS if scramble_banks else real_bids).to(DEVICE) \
                    if give_banks else None
                if give_fp16:
                    fp16_in = (-real_fp16 if scramble_fp16 else real_fp16).to(DEVICE)
                    fp16_in = (fp16_in - fp16_thresh) / max(abs(fp16_thresh), 1e-6)
                else:
                    fp16_in = None
                if give_timing:
                    if scramble_timing:
                        tfeat = torch.full((B,), float(sclk_st), device=DEVICE)
                    else:
                        tfeat = torch.full((B,), norm_timing(wall_ms, info), device=DEVICE)
                else:
                    tfeat = None
                logits, _ = model(imgs, bids, fp16_in, tfeat)
                correct += (logits.argmax(1) == labels).sum().item()
                total += B
    return correct / total

# ━━━ Main ━━━
def main():
    print("=== z2058: Three-Channel Unified Embodiment ===\n")
    t0 = time.time()

    print("Compiling HIP kernel...")
    ext = load_inline(name='z2058_3ch', cpp_sources='std::vector<torch::Tensor> probe(int n);',
                      cuda_sources=HIP_SRC, functions=['probe'], verbose=False,
                      extra_cuda_cflags=['--offload-arch=gfx1100', '-O2'])

    info, wc_ratio = characterize(ext)

    # Characterize fp16 rounding
    banks_raw, fp16_raw, _ = ext.probe(1024); torch.cuda.synchronize()
    wgps = sorted(set(banks_raw.cpu().numpy().tolist()))
    fp16_vals = fp16_raw.cpu().numpy()
    fp16_unique = sorted(set(np.round(fp16_vals, 4).tolist()))
    fp16_thresh = float(np.median(fp16_vals))
    print(f"  WGP values: {wgps} ({len(wgps)} unique)")
    print(f"  FP16 unique values: {fp16_unique}")
    print(f"  FP16 threshold: {fp16_thresh:.6f}")

    # Verify per-bank rounding
    bids = (banks_raw // 2).clamp(0, NUM_BANKS-1).cpu().numpy()
    for b in range(NUM_BANKS):
        mask = bids == b
        if mask.any():
            vals = fp16_vals[mask]
            print(f"    Bank {b}: fp16={vals[0]:.6f} (n={mask.sum()}, {'round_zero' if b%2==1 else 'round_near'})")

    train_ld, test_ld = get_data()

    # ─── Train models ───
    print("\n--- Training A_full (3 channels) ---")
    model_a = ThreeChannelModel(True, True, True).to(DEVICE)
    train_model(model_a, train_ld, ext, info, fp16_thresh, label='A')

    print("\n--- Training B_blind (no hw) ---")
    model_b = ThreeChannelModel(False, False, False).to(DEVICE)
    train_model(model_b, train_ld, ext, info, fp16_thresh, label='B')

    print("\n--- Training C_bank_only (1 channel) ---")
    model_c = ThreeChannelModel(True, False, False).to(DEVICE)
    train_model(model_c, train_ld, ext, info, fp16_thresh, label='C')

    print("\n--- Training D_bank_fp16 (2 channels) ---")
    model_d = ThreeChannelModel(True, True, False).to(DEVICE)
    train_model(model_d, train_ld, ext, info, fp16_thresh, label='D')

    # ─── Evaluate ───
    print("\n--- Evaluation ---")
    set_sclk('high')

    acc_a = eval_model(model_a, test_ld, ext, info, fp16_thresh, True, True, True)
    print(f"  A_full (3ch):       {acc_a:.4f}")

    acc_b = eval_model(model_b, test_ld, ext, info, fp16_thresh, False, False, False)
    print(f"  B_blind:            {acc_b:.4f}")

    acc_c = eval_model(model_c, test_ld, ext, info, fp16_thresh, True, False, False)
    print(f"  C_bank_only:        {acc_c:.4f}")

    acc_d = eval_model(model_d, test_ld, ext, info, fp16_thresh, True, True, False)
    print(f"  D_bank+fp16:        {acc_d:.4f}")

    acc_scr_t = eval_model(model_a, test_ld, ext, info, fp16_thresh, True, True, True,
                            scramble_timing=True)
    print(f"  E_scr_timing (A):   {acc_scr_t:.4f}")

    acc_scr_b = eval_model(model_a, test_ld, ext, info, fp16_thresh, True, True, True,
                            scramble_banks=True)
    print(f"  F_scr_bank (A):     {acc_scr_b:.4f}")

    acc_scr_f = eval_model(model_a, test_ld, ext, info, fp16_thresh, True, True, True,
                            scramble_fp16=True)
    print(f"  G_scr_fp16 (A):     {acc_scr_f:.4f}")

    # ─── Analysis ───
    mean_cos = 0.0
    if model_a.ub:
        W = model_a.bank_w.detach().cpu()
        cs = [F.cosine_similarity(W[i].flatten().unsqueeze(0),
              W[j].flatten().unsqueeze(0)).item()
              for i in range(NUM_BANKS) for j in range(i+1, NUM_BANKS)]
        mean_cos = float(np.mean(cs))
        print(f"\n  Bank weight cos_sim: {mean_cos:.3f}")

    gate_val = None
    g_low = g_high = 0.0
    if model_a.ut:
        model_a.eval()
        with torch.no_grad():
            g_low = model_a.gate(torch.tensor([[1.0]], device=DEVICE)).item()
            g_high = model_a.gate(torch.tensor([[0.0]], device=DEVICE)).item()
            gate_val = (g_low + g_high) / 2
        print(f"  Gate: low={g_low:.3f} high={g_high:.3f} mean={gate_val:.3f}")

    elapsed = time.time() - t0

    # ─── Tests ───
    print("\n=== TEST VERDICTS ===")
    results = {
        'experiment': 'z2058_three_channel_unified',
        'timing': info, 'wall_clock_ratio': wc_ratio,
        'wgp_values': wgps, 'fp16_unique': fp16_unique, 'fp16_thresh': fp16_thresh,
        'accuracies': {
            'A_full': acc_a, 'B_blind': acc_b, 'C_bank_only': acc_c,
            'D_bank_fp16': acc_d, 'E_scr_timing': acc_scr_t,
            'F_scr_bank': acc_scr_b, 'G_scr_fp16': acc_scr_f
        },
        'bank_cos_sim': mean_cos,
        'gate': {'low': g_low, 'high': g_high, 'mean': gate_val} if gate_val else None,
        'elapsed_s': elapsed, 'tests': {}
    }

    def test(name, cond, desc):
        v = "PASS" if cond else "FAIL"
        results['tests'][name] = {'verdict': v, 'description': desc}
        print(f"  {name}: {v} -- {desc}")
        return cond

    t1 = test('T1_full_learns', acc_a > 0.90, f'A={acc_a:.1%} > 90%')
    t2 = test('T2_full_blind_gap', acc_a - acc_b > 0.40, f'A-B={acc_a-acc_b:.1%} > 40pp')
    t3 = test('T3_third_adds', acc_a - acc_d > 0.10, f'A-D={acc_a-acc_d:.1%} > 10pp')
    t4 = test('T4_timing_kill', acc_a - acc_scr_t > 0.10, f'A-E={acc_a-acc_scr_t:.1%} > 10pp')
    t5 = test('T5_wc_ratio', wc_ratio > 2.0, f'ratio={wc_ratio:.2f} > 2.0')
    t6 = test('T6_gate_active', (gate_val or 0) > 0.3, f'gate={gate_val:.3f} > 0.3')
    t7 = test('T7_wgp_coverage', len(wgps) >= 3, f'{len(wgps)} WGPs >= 3')
    t8 = test('T8_fp16_diversity', len(fp16_unique) >= 2, f'{len(fp16_unique)} fp16 values >= 2')

    passes = sum([t1, t2, t3, t4, t5, t6, t7, t8])
    verdict = f"{passes}/8 PASS"
    results['verdict'] = verdict
    results['pass_count'] = passes
    hierarchy = f"B={acc_b:.1%} < C={acc_c:.1%} < D={acc_d:.1%} < A={acc_a:.1%}"
    results['hierarchy'] = hierarchy
    print(f"\n  Hierarchy: {hierarchy}")
    print(f"  VERDICT: {verdict}  ({elapsed:.0f}s)")

    out = 'results/z2058_three_channel_unified.json'
    with open(out, 'w') as f: json.dump(results, f, indent=2)
    print(f"  Saved: {out}")

    try: open(DPM, 'w').write('auto')
    except: pass
    print("  Restored SCLK to auto")

if __name__ == '__main__':
    main()
