#!/usr/bin/env python3
"""
z2454: Aligned proxy — fix proxy-real gap, then validate robustness

z2453 showed:
  - Mechanisms give +11.4pp noise robustness at σ=0.5
  - But proxy-real gap is 2.2pp (proxy doesn't match kernel)
  - Alpha values grow too large (shfl=1.5)

Fixes:
  1. Better proxy: __shfl_xor(v, 1) swaps adjacent PAIRS (0↔1, 2↔3, 4↔5...),
     not roll. Proxy should use index-based swap, not roll.
  2. Clamp alphas to [0, 0.3] to keep mechanisms RESIDUAL
  3. Calibration loss: penalize proxy-real mismatch every N batches
  4. Wider model (128 neurons) + 2 hidden layers for fairer comparison
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

    float sum = b[n];
    float exc = 0, inh = 0;
    for (int k = 0; k < in_d; k++) {
        float p = W[n*in_d+k] * X[s*in_d+k];
        sum += p;
        if (p > 0) exc += p; else inh += p;
    }
    float relu_out = (sum > 0) ? sum : 0;
    float result = relu_out;

    if (alpha_shfl != 0) {
        float neighbor = __shfl_xor(relu_out, 1);
        result += alpha_shfl * (relu_out - neighbor) * 0.5f;
    }
    if (alpha_branch != 0) {
        float bs = exc + inh * lambda_inh + b[n];
        float br = (bs > 0) ? bs : 0;
        result += alpha_branch * (br - relu_out);
    }
    if (alpha_gate != 0) {
        int na = __popcll(__ballot(relu_out > 0));
        float ar = (float)na / 64.0f;
        float gate = 1.0f / (1.0f + expf(-10.0f * (0.5f - ar)));
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
ext = load_inline(name='z2454', cpp_sources=CPP, cuda_sources=HIP_SRC,
                  functions=['mech_layer'], extra_cuda_cflags=['-O2','--offload-arch=gfx1100'], verbose=False)
print("OK\n")

device = torch.device('cuda')

# MNIST
with open(f'{base}/data/MNIST/raw/train-images-idx3-ubyte','rb') as f:
    f.read(16); tr_img=np.frombuffer(f.read(),dtype=np.uint8).reshape(-1,784).astype(np.float32)/255
with open(f'{base}/data/MNIST/raw/train-labels-idx1-ubyte','rb') as f:
    f.read(8); tr_lbl=np.frombuffer(f.read(),dtype=np.uint8)
with open(f'{base}/data/MNIST/raw/t10k-images-idx3-ubyte','rb') as f:
    f.read(16); te_img=np.frombuffer(f.read(),dtype=np.uint8).reshape(-1,784).astype(np.float32)/255
with open(f'{base}/data/MNIST/raw/t10k-labels-idx1-ubyte','rb') as f:
    f.read(8); te_lbl=np.frombuffer(f.read(),dtype=np.uint8)
X_tr=torch.tensor(tr_img,device=device); y_tr=torch.tensor(tr_lbl,dtype=torch.long,device=device)
X_te=torch.tensor(te_img,device=device); y_te=torch.tensor(te_lbl,dtype=torch.long,device=device)

def shfl_xor_proxy(x, shift=1):
    """Correct proxy for __shfl_xor(v, 1): swap adjacent pairs.
    Lane 0↔1, 2↔3, 4↔5... within each group of 64."""
    B, D = x.shape
    # Group into pairs and swap
    if D % 2 == 0:
        x2 = x.view(B, D//2, 2)
        x2 = x2.flip(-1)  # swap within each pair
        return x2.view(B, D)
    else:
        return torch.roll(x, 1, -1)

class MechLayer(nn.Module):
    def __init__(self, in_d, out_d, max_alpha=0.3):
        super().__init__()
        self.linear = nn.Linear(in_d, out_d)
        # Raw params (unclamped), will be clamped in forward
        self._alpha_shfl = nn.Parameter(torch.tensor(0.0))
        self._alpha_branch = nn.Parameter(torch.tensor(0.0))
        self._lambda_inh = nn.Parameter(torch.tensor(0.0))  # maps to [0.5, 1.0] via sigmoid
        self._alpha_gate = nn.Parameter(torch.tensor(0.0))
        self.max_alpha = max_alpha

    @property
    def alpha_shfl(self): return torch.tanh(self._alpha_shfl) * self.max_alpha
    @property
    def alpha_branch(self): return torch.tanh(self._alpha_branch) * self.max_alpha
    @property
    def alpha_gate(self): return torch.tanh(self._alpha_gate) * self.max_alpha
    @property
    def lambda_inh(self): return 0.5 + 0.5 * torch.sigmoid(self._lambda_inh)  # [0.5, 1.0]

    def forward_proxy(self, x):
        raw = self.linear(x)
        relu_out = F.relu(raw)
        r = relu_out

        # Shuffle: swap adjacent pairs (matches __shfl_xor(v,1))
        neighbor = shfl_xor_proxy(relu_out)
        r = r + self.alpha_shfl * (relu_out - neighbor) * 0.5

        # Branch: per-element E/I (approximate)
        li = self.lambda_inh.item()
        pos = F.relu(raw)
        neg = raw - pos  # negative part
        branch_out = F.relu(pos + neg * li + self.linear.bias)  # re-bias after split
        # Simplified: use LeakyReLU as proxy
        branch_out = F.leaky_relu(raw, li)
        r = r + self.alpha_branch * (branch_out - relu_out)

        # Gate: soft population regulation
        af = (relu_out > 0).float().mean(-1, keepdim=True)
        gate = torch.sigmoid(-10 * (af - 0.5))
        r = r + self.alpha_gate * relu_out * (gate - 1)

        return r

    def forward_real(self, x):
        return ext.mech_layer(x.contiguous(), self.linear.weight.contiguous(),
                              self.linear.bias.contiguous(),
                              self.alpha_shfl.item(), self.alpha_branch.item(),
                              self.lambda_inh.item(), self.alpha_gate.item())

    def alpha_summary(self):
        return f"s={self.alpha_shfl.item():.3f} b={self.alpha_branch.item():.3f} " \
               f"λ={self.lambda_inh.item():.3f} g={self.alpha_gate.item():.3f}"

class MechNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = MechLayer(784, 128, max_alpha=0.3)
        self.l2 = MechLayer(128, 128, max_alpha=0.3)
        self.head = nn.Linear(128, 10)

    def forward_proxy(self, x):
        return self.head(self.l2.forward_proxy(self.l1.forward_proxy(x)))

    def forward_real(self, x):
        return self.head(self.l2.forward_real(self.l1.forward_real(x)))

print("=" * 70)
print("z2454: ALIGNED PROXY — clamped alphas, corrected shuffle, 2 layers")
print("=" * 70)

model = MechNet().to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
np_model = sum(p.numel() for p in model.parameters())

# Baseline
bl = nn.Sequential(nn.Linear(784,128), nn.ReLU(), nn.Linear(128,128), nn.ReLU(), nn.Linear(128,10)).to(device)
opt_bl = torch.optim.Adam(bl.parameters(), lr=0.001)
for ep in range(15):
    perm=torch.randperm(len(X_tr),device=device)
    for i in range(0,len(X_tr),256):
        loss=F.cross_entropy(bl(X_tr[perm[i:i+256]]),y_tr[perm[i:i+256]])
        opt_bl.zero_grad();loss.backward();opt_bl.step()
bl.eval()
with torch.no_grad(): acc_bl=(bl(X_te).argmax(1)==y_te).float().mean().item()*100
np_bl = sum(p.numel() for p in bl.parameters())
print(f"Baseline: {acc_bl:.2f}% ({np_bl:,} params)")
print(f"Model:    {np_model:,} params\n")

# Phase 1: ReLU only
print("--- Phase 1: ReLU only (alphas frozen) ---")
for p in [model.l1._alpha_shfl, model.l1._alpha_branch, model.l1._alpha_gate,
          model.l2._alpha_shfl, model.l2._alpha_branch, model.l2._alpha_gate,
          model.l1._lambda_inh, model.l2._lambda_inh]:
    p.requires_grad = False

for epoch in range(10):
    model.train(); perm=torch.randperm(len(X_tr),device=device); tl=0;nb=0
    for i in range(0,len(X_tr),256):
        loss=F.cross_entropy(model.forward_proxy(X_tr[perm[i:i+256]]),y_tr[perm[i:i+256]])
        opt.zero_grad();loss.backward();opt.step();tl+=loss.item();nb+=1
model.eval()
with torch.no_grad():
    ap=(model.forward_proxy(X_te).argmax(1)==y_te).float().mean().item()*100
    ar=(model.forward_real(X_te).argmax(1)==y_te).float().mean().item()*100
print(f"  proxy={ap:.1f}% real={ar:.1f}% gap={ar-ap:+.1f}pp")

# Phase 2: Unfreeze all alphas
print("\n--- Phase 2: All mechanisms (clamped to ±0.3) ---")
for p in model.parameters(): p.requires_grad = True
# Add calibration: every 10 batches, compare proxy vs real on a small batch
calib_freq = 20
calib_losses = []

for epoch in range(20):
    model.train(); perm=torch.randperm(len(X_tr),device=device); tl=0;nb=0; cl=0; cn=0
    for i in range(0,len(X_tr),256):
        xb,yb = X_tr[perm[i:i+256]], y_tr[perm[i:i+256]]
        # Main loss
        loss = F.cross_entropy(model.forward_proxy(xb), yb)

        # Calibration loss: proxy-real MSE on activations
        if nb % calib_freq == 0:
            with torch.no_grad():
                real_out = model.forward_real(xb[:64])
            proxy_out = model.forward_proxy(xb[:64])
            calib = F.mse_loss(proxy_out, real_out.detach()) * 0.1
            loss = loss + calib
            cl += calib.item(); cn += 1

        opt.zero_grad(); loss.backward(); opt.step()
        tl+=loss.item();nb+=1

    model.eval()
    with torch.no_grad():
        ap=(model.forward_proxy(X_te[:2000]).argmax(1)==y_te[:2000]).float().mean().item()*100
        ar=(model.forward_real(X_te[:2000]).argmax(1)==y_te[:2000]).float().mean().item()*100
    calib_avg = cl/cn if cn > 0 else 0
    if (epoch+1)%5==0 or epoch==0:
        print(f"  ep{epoch+1:2d}: loss={tl/nb:.4f} calib={calib_avg:.5f} proxy={ap:.1f}% real={ar:.1f}% gap={ar-ap:+.1f}pp "
              f"[L1: {model.l1.alpha_summary()} | L2: {model.l2.alpha_summary()}]")

# Full eval
model.eval()
with torch.no_grad():
    acc_proxy=(model.forward_proxy(X_te).argmax(1)==y_te).float().mean().item()*100
    acc_real=(model.forward_real(X_te).argmax(1)==y_te).float().mean().item()*100

# Noise robustness
print(f"\n--- Noise Robustness ---")
print(f"  {'σ':>5} {'Baseline':>10} {'Mech(real)':>12} {'Δ':>8}")
robust = {}
for sigma in [0.0, 0.1, 0.2, 0.3, 0.5]:
    noisy=(X_te+torch.randn_like(X_te)*sigma).clamp(0,1)
    with torch.no_grad():
        a_bl=(bl(noisy).argmax(1)==y_te).float().mean().item()*100
        a_r=(model.forward_real(noisy).argmax(1)==y_te).float().mean().item()*100
    d=a_r-a_bl
    marker = "★" if d > 1 else ""
    print(f"  {sigma:>5.2f} {a_bl:>9.2f}% {a_r:>11.2f}% {d:>+7.2f}pp {marker}")
    robust[f's{sigma}'] = {'bl':a_bl, 'mech':a_r, 'delta':d}

# Summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"  Baseline MLP:    {acc_bl:.2f}% ({np_bl:,} params)")
print(f"  Mechanism proxy: {acc_proxy:.2f}%")
print(f"  Mechanism real:  {acc_real:.2f}% ({np_model:,} params)")
print(f"  Proxy-real gap:  {acc_real-acc_proxy:+.2f}pp")
print(f"\n  L1: {model.l1.alpha_summary()}")
print(f"  L2: {model.l2.alpha_summary()}")

r03 = robust.get('s0.3', {})
r05 = robust.get('s0.5', {})
if r03.get('delta',0) > 0 or r05.get('delta',0) > 0:
    print(f"\n  ROBUSTNESS ADVANTAGE:")
    if r03.get('delta',0) > 0: print(f"    σ=0.3: {r03['delta']:+.1f}pp")
    if r05.get('delta',0) > 0: print(f"    σ=0.5: {r05['delta']:+.1f}pp")
    print(f"    Mechanisms trade {acc_bl-acc_real:.1f}pp clean accuracy")
    print(f"    for {r05.get('delta',0):+.1f}pp robustness at high noise")

results = {'baseline':acc_bl, 'proxy':acc_proxy, 'real':acc_real,
           'gap':acc_real-acc_proxy, 'robust':robust,
           'alphas_l1':{'s':model.l1.alpha_shfl.item(),'b':model.l1.alpha_branch.item(),
                        'l':model.l1.lambda_inh.item(),'g':model.l1.alpha_gate.item()},
           'alphas_l2':{'s':model.l2.alpha_shfl.item(),'b':model.l2.alpha_branch.item(),
                        'l':model.l2.lambda_inh.item(),'g':model.l2.alpha_gate.item()}}
with open(f'{base}/results/z2454_aligned_proxy.json','w') as f:
    json.dump(results,f,indent=2)
print(f"\nSaved to results/z2454_aligned_proxy.json")
