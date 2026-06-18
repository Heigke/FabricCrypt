#!/usr/bin/env python3
"""
z2453: Mechanism curriculum v2 — proper staged training

z2452 issue: proxy trained with mechanisms ON → weights adapted to mechanisms
→ real ReLU-only (stage A) is 4.7pp worse because weights aren't for ReLU anymore.

Fix: Train in stages. Each stage locks in the previous mechanisms.
  Phase 1: Train as pure ReLU (alpha=0 fixed) → establish baseline weights
  Phase 2: Unfreeze alpha_shfl, continue training → shuffle earns its place
  Phase 3: Unfreeze alpha_branch → branch earns its place
  Phase 4: Unfreeze alpha_gate → gate earns its place

At each phase, validate on REAL kernel to confirm proxy≈real.
"""
import os, time, json
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

base = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy'

from torch.utils.cpp_extension import load_inline

HIP_SRC = r"""
#include <hip/hip_runtime.h>

__global__ void mech_kernel(
    const float* __restrict__ X, const float* __restrict__ W,
    const float* __restrict__ b, float* __restrict__ Y,
    int N, int in_d, int out_d,
    float alpha_shfl, float alpha_branch, float lambda_inh, float alpha_gate
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int s = idx / out_d, n = idx % out_d;
    if (s >= N) return;

    // Standard ReLU path
    float sum = b[n];
    float exc = 0, inh = 0;
    for (int k = 0; k < in_d; k++) {
        float p = W[n*in_d+k] * X[s*in_d+k];
        sum += p;
        if (p > 0) exc += p; else inh += p;
    }
    float relu_out = (sum > 0) ? sum : 0;

    float result = relu_out;

    // Shuffle residual
    if (alpha_shfl != 0) {
        float neighbor = __shfl_xor(relu_out, 1);
        result += alpha_shfl * (relu_out - neighbor) * 0.5f;
    }

    // Branch residual
    if (alpha_branch != 0) {
        float branch_sum = exc + inh * lambda_inh + b[n];
        float branch_relu = (branch_sum > 0) ? branch_sum : 0;
        result += alpha_branch * (branch_relu - relu_out);
    }

    // Soft gate residual
    if (alpha_gate != 0) {
        unsigned long long active = __ballot(relu_out > 0);
        int na = __popcll(active);
        float act_ratio = (float)na / 64.0f;
        float gate = 1.0f / (1.0f + expf(-10.0f * (0.5f - act_ratio)));
        result += alpha_gate * relu_out * (gate - 1.0f);
    }

    Y[s*out_d+n] = result;
}

torch::Tensor mech_layer(torch::Tensor X, torch::Tensor W, torch::Tensor b,
                          float as, float ab, float li, float ag) {
    int N=X.size(0), id=X.size(1), od=W.size(0);
    auto Y=torch::empty({N,od},X.options());
    mech_kernel<<<(N*od+255)/256,256>>>(X.data_ptr<float>(),W.data_ptr<float>(),
        b.data_ptr<float>(),Y.data_ptr<float>(),N,id,od,as,ab,li,ag);
    return Y;
}
"""
CPP = "torch::Tensor mech_layer(torch::Tensor X, torch::Tensor W, torch::Tensor b, float as, float ab, float li, float ag);"

print("Compiling...")
ext = load_inline(name='z2453', cpp_sources=CPP, cuda_sources=HIP_SRC,
                  functions=['mech_layer'], extra_cuda_cflags=['-O2','--offload-arch=gfx1100'], verbose=False)
print("OK\n")

device = torch.device('cuda')

# MNIST
with open(f'{base}/data/MNIST/raw/train-images-idx3-ubyte','rb') as f:
    f.read(16); tr_img = np.frombuffer(f.read(),dtype=np.uint8).reshape(-1,784).astype(np.float32)/255
with open(f'{base}/data/MNIST/raw/train-labels-idx1-ubyte','rb') as f:
    f.read(8); tr_lbl = np.frombuffer(f.read(),dtype=np.uint8)
with open(f'{base}/data/MNIST/raw/t10k-images-idx3-ubyte','rb') as f:
    f.read(16); te_img = np.frombuffer(f.read(),dtype=np.uint8).reshape(-1,784).astype(np.float32)/255
with open(f'{base}/data/MNIST/raw/t10k-labels-idx1-ubyte','rb') as f:
    f.read(8); te_lbl = np.frombuffer(f.read(),dtype=np.uint8)

X_tr = torch.tensor(tr_img, device=device)
y_tr = torch.tensor(tr_lbl, dtype=torch.long, device=device)
X_te = torch.tensor(te_img, device=device)
y_te = torch.tensor(te_lbl, dtype=torch.long, device=device)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(784, 128)
        self.w2 = nn.Linear(128, 10)
        self.alpha_shfl = nn.Parameter(torch.tensor(0.0))
        self.alpha_branch = nn.Parameter(torch.tensor(0.0))
        self.lambda_inh = nn.Parameter(torch.tensor(0.95))
        self.alpha_gate = nn.Parameter(torch.tensor(0.0))

    def forward_proxy(self, x):
        raw = self.w1(x)
        relu_out = F.relu(raw)
        r = relu_out
        # Shuffle proxy
        r = r + self.alpha_shfl * (relu_out - torch.roll(relu_out, 1, -1)) * 0.5
        # Branch proxy
        branch = F.leaky_relu(raw, self.lambda_inh.clamp(0.01,0.99).item())
        r = r + self.alpha_branch * (branch - relu_out)
        # Gate proxy
        af = (relu_out > 0).float().mean(-1, keepdim=True)
        gate = torch.sigmoid(-10 * (af - 0.5))
        r = r + self.alpha_gate * relu_out * (gate - 1)
        return self.w2(r)

    def forward_real(self, x):
        h = ext.mech_layer(x.contiguous(), self.w1.weight.contiguous(),
                           self.w1.bias.contiguous(),
                           self.alpha_shfl.item(), self.alpha_branch.item(),
                           self.lambda_inh.item(), self.alpha_gate.item())
        return self.w2(h)

def train_phase(model, opt, n_ep, label, freeze_alphas=None):
    """Train for n_ep epochs. freeze_alphas = list of param names to freeze."""
    if freeze_alphas:
        for name, p in model.named_parameters():
            if name in freeze_alphas:
                p.requires_grad = False
    for epoch in range(n_ep):
        model.train()
        perm = torch.randperm(len(X_tr), device=device)
        tl = 0; nb = 0
        for i in range(0, len(X_tr), 256):
            xb, yb = X_tr[perm[i:i+256]], y_tr[perm[i:i+256]]
            loss = F.cross_entropy(model.forward_proxy(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        model.eval()
        with torch.no_grad():
            ap = (model.forward_proxy(X_te).argmax(1)==y_te).float().mean().item()*100
            ar = (model.forward_real(X_te).argmax(1)==y_te).float().mean().item()*100
        if (epoch+1) == n_ep or epoch == 0:
            a = model
            print(f"  {label} ep{epoch+1}: loss={tl/nb:.4f} proxy={ap:.1f}% real={ar:.1f}% gap={ar-ap:+.1f}pp "
                  f"[s={a.alpha_shfl.item():.3f} b={a.alpha_branch.item():.3f} g={a.alpha_gate.item():.3f}]")
    # Unfreeze all
    for p in model.parameters():
        p.requires_grad = True
    return ap, ar

print("=" * 70)
print("z2453: STAGED MECHANISM CURRICULUM")
print("=" * 70)

model = Model().to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.001)

# Phase 1: Pure ReLU (all alphas frozen at 0)
print("\n--- Phase 1: Pure ReLU (alphas frozen = 0) ---")
ap1, ar1 = train_phase(model, opt, 10, "P1",
    freeze_alphas=['alpha_shfl','alpha_branch','lambda_inh','alpha_gate'])

# Phase 2: Unfreeze shuffle
print("\n--- Phase 2: + Shuffle (unfreeze alpha_shfl) ---")
ap2, ar2 = train_phase(model, opt, 10, "P2",
    freeze_alphas=['alpha_branch','alpha_gate'])

# Phase 3: Unfreeze branch
print("\n--- Phase 3: + Branch (unfreeze alpha_branch, lambda_inh) ---")
ap3, ar3 = train_phase(model, opt, 10, "P3",
    freeze_alphas=['alpha_gate'])

# Phase 4: Unfreeze gate
print("\n--- Phase 4: + Gate (all unfrozen) ---")
ap4, ar4 = train_phase(model, opt, 10, "P4",
    freeze_alphas=[])

# Noise robustness
print("\n--- Noise Robustness ---")
# Standard baseline
bl = nn.Sequential(nn.Linear(784,128), nn.ReLU(), nn.Linear(128,10)).to(device)
opt_bl = torch.optim.Adam(bl.parameters(), lr=0.001)
for ep in range(15):
    perm = torch.randperm(len(X_tr), device=device)
    for i in range(0, len(X_tr), 256):
        loss = F.cross_entropy(bl(X_tr[perm[i:i+256]]), y_tr[perm[i:i+256]])
        opt_bl.zero_grad(); loss.backward(); opt_bl.step()
bl.eval()
with torch.no_grad():
    acc_bl = (bl(X_te).argmax(1)==y_te).float().mean().item()*100
print(f"  Baseline: {acc_bl:.2f}%")

print(f"\n  {'σ':>5} {'Baseline':>10} {'Real(mech)':>12} {'Δ':>8}")
for sigma in [0.0, 0.1, 0.2, 0.3, 0.5]:
    noisy = (X_te + torch.randn_like(X_te)*sigma).clamp(0,1)
    with torch.no_grad():
        a_bl = (bl(noisy).argmax(1)==y_te).float().mean().item()*100
        a_r = (model.forward_real(noisy).argmax(1)==y_te).float().mean().item()*100
    print(f"  {sigma:>5.2f} {a_bl:>9.2f}% {a_r:>11.2f}% {a_r-a_bl:>+7.2f}pp")

# Run-to-run variance
print("\n--- Run-to-run variance (real kernel, 10 runs) ---")
accs = []
with torch.no_grad():
    for _ in range(10):
        a = (model.forward_real(X_te[:2000]).argmax(1)==y_te[:2000]).float().mean().item()*100
        accs.append(a)
print(f"  Accs: {[f'{a:.2f}' for a in accs]}")
print(f"  Mean={np.mean(accs):.2f}% Std={np.std(accs):.4f}%")

# Summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"  Phase 1 (ReLU only):      proxy={ap1:.1f}%  real={ar1:.1f}%  gap={ar1-ap1:+.1f}pp")
print(f"  Phase 2 (+ shuffle):      proxy={ap2:.1f}%  real={ar2:.1f}%  gap={ar2-ap2:+.1f}pp")
print(f"  Phase 3 (+ branch):       proxy={ap3:.1f}%  real={ar3:.1f}%  gap={ar3-ap3:+.1f}pp")
print(f"  Phase 4 (+ gate):         proxy={ap4:.1f}%  real={ar4:.1f}%  gap={ar4-ap4:+.1f}pp")
print(f"  Baseline MLP:             {acc_bl:.1f}%")
print(f"\n  Learned alphas: shfl={model.alpha_shfl.item():.4f} branch={model.alpha_branch.item():.4f} "
      f"λ={model.lambda_inh.item():.4f} gate={model.alpha_gate.item():.4f}")

# Key questions
gap_close = abs(ar4 - ap4) < 1.0
mech_helps = ar4 > ar1 + 0.5
beats_bl = ar4 > acc_bl - 1.0

print(f"\n  Proxy≈Real gap < 1pp?  {'YES ✓' if gap_close else 'NO ✗'}")
print(f"  Mechanisms help?       {'YES ✓' if mech_helps else 'NO ✗'} (real P4 vs real P1: {ar4-ar1:+.1f}pp)")
print(f"  Competitive with MLP?  {'YES ✓' if beats_bl else 'NO ✗'} (real P4 vs baseline: {ar4-acc_bl:+.1f}pp)")

results = {
    'phases': {'p1': [ap1,ar1], 'p2': [ap2,ar2], 'p3': [ap3,ar3], 'p4': [ap4,ar4]},
    'baseline': acc_bl,
    'alphas': {'shfl': model.alpha_shfl.item(), 'branch': model.alpha_branch.item(),
               'lambda': model.lambda_inh.item(), 'gate': model.alpha_gate.item()},
    'run_variance': {'mean': float(np.mean(accs)), 'std': float(np.std(accs))},
}
with open(f'{base}/results/z2453_curriculum_v2.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to results/z2453_curriculum_v2.json")
