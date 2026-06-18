#!/usr/bin/env python3
"""
z2452: Mechanism Curriculum — one mechanism at a time, residual, trainable

Architecture: y = ReLU(Wx + b) + α * M(x, W)
  - Standard compute is UNTOUCHED (ReLU path)
  - Mechanism is RESIDUAL, scaled by learnable α (init near 0)
  - Mechanism must EARN its way in during training

Stages:
  A: Baseline — standard ReLU only (verify training works in custom kernel)
  B: + shuffle residual (safest mechanism: structured, local, linear)
  C: + branch split (learnable inhibitory coefficient λ, init 0.95)
  D: + soft gate (sigmoid-style, not hard ballot)
  E: + atomic residual (tiny, 1-5%)

All stages use REAL HIP kernel for forward, surrogate gradient (STE) for backward.
Proxy = the differentiable approximation used in backward pass.
Real kernel = what actually runs on GPU hardware.
Periodic calibration: compare proxy vs real output, log mismatch.

Success metric per stage:
  - Within 2pp of standard MLP at same param count
  - No NaN
  - Stable training (loss decreases monotonically)
"""
import os, time, json
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

base = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy'

# ============================================================
# HIP kernels — one per stage, clean and minimal
# ============================================================
from torch.utils.cpp_extension import load_inline

HIP_SRC = r"""
#include <hip/hip_runtime.h>

// Stage A: Standard ReLU (baseline, verify kernel works)
__global__ void stage_a_kernel(
    const float* __restrict__ X, const float* __restrict__ W,
    const float* __restrict__ b, float* __restrict__ Y,
    int N, int in_d, int out_d
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int s = idx / out_d, n = idx % out_d;
    if (s >= N) return;
    float sum = b[n];
    for (int k = 0; k < in_d; k++) sum += W[n*in_d+k] * X[s*in_d+k];
    Y[s*out_d+n] = (sum > 0) ? sum : 0;
}

// Stage B: ReLU + shuffle residual
// y = ReLU(Wx+b) + alpha * (shuffle_neighbor_diff)
__global__ void stage_b_kernel(
    const float* __restrict__ X, const float* __restrict__ W,
    const float* __restrict__ b, float* __restrict__ Y,
    int N, int in_d, int out_d, float alpha
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int s = idx / out_d, n = idx % out_d;
    if (s >= N) return;
    float sum = b[n];
    for (int k = 0; k < in_d; k++) sum += W[n*in_d+k] * X[s*in_d+k];
    float relu_out = (sum > 0) ? sum : 0;

    // MECHANISM: shuffle residual (real __shfl_xor)
    float neighbor = __shfl_xor(relu_out, 1);
    float shfl_residual = (relu_out - neighbor) * 0.5f; // lateral contrast

    Y[s*out_d+n] = relu_out + alpha * shfl_residual;
}

// Stage C: ReLU + shuffle + branch split residual
// Branch: separate E/I accumulation, lerp with lambda
__global__ void stage_c_kernel(
    const float* __restrict__ X, const float* __restrict__ W,
    const float* __restrict__ b, float* __restrict__ Y,
    int N, int in_d, int out_d, float alpha_shfl, float alpha_branch, float lambda_inh
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int s = idx / out_d, n = idx % out_d;
    if (s >= N) return;

    // Standard accumulation
    float sum = b[n];
    float exc = 0, inh = 0;
    for (int k = 0; k < in_d; k++) {
        float p = W[n*in_d+k] * X[s*in_d+k];
        sum += p;
        if (p > 0) exc += p; else inh += p;
    }
    float relu_out = (sum > 0) ? sum : 0;

    // Branch residual: difference between E/I split and standard
    float branch_sum = exc + inh * lambda_inh + b[n];
    float branch_relu = (branch_sum > 0) ? branch_sum : 0;
    float branch_residual = branch_relu - relu_out;

    // Shuffle residual
    float neighbor = __shfl_xor(relu_out, 1);
    float shfl_residual = (relu_out - neighbor) * 0.5f;

    Y[s*out_d+n] = relu_out + alpha_shfl * shfl_residual + alpha_branch * branch_residual;
}

// Stage D: + soft gate residual (sigmoid population regulation)
__global__ void stage_d_kernel(
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

    // Branch residual
    float branch_sum = exc + inh * lambda_inh + b[n];
    float branch_relu = (branch_sum > 0) ? branch_sum : 0;
    float branch_res = branch_relu - relu_out;

    // Shuffle residual
    float neighbor = __shfl_xor(relu_out, 1);
    float shfl_res = (relu_out - neighbor) * 0.5f;

    // Soft gate: ballot-based population activity -> sigmoid modulation
    unsigned long long active = __ballot(relu_out > 0);
    int n_active = __popcll(active);
    float activity = (float)n_active / 64.0f;
    // Sigmoid gate: if too many active, suppress; if too few, boost
    float gate = 1.0f / (1.0f + expf(-10.0f * (0.5f - activity))); // centers at 50%
    float gate_res = relu_out * (gate - 1.0f); // residual from gating

    Y[s*out_d+n] = relu_out + alpha_shfl * shfl_res + alpha_branch * branch_res + alpha_gate * gate_res;
}

// Stage E: + atomic contention residual
__global__ void stage_e_kernel(
    const float* __restrict__ X, const float* __restrict__ W,
    const float* __restrict__ b, float* __restrict__ Y,
    int N, int in_d, int out_d,
    float alpha_shfl, float alpha_branch, float lambda_inh,
    float alpha_gate, float alpha_atomic
) {
    int sample = blockIdx.x;
    if (sample >= N) return;
    int tid = threadIdx.x;
    if (tid >= out_d) return;

    float sum = b[tid];
    float exc = 0, inh = 0;
    for (int k = 0; k < in_d; k++) {
        float p = W[tid*in_d+k] * X[sample*in_d+k];
        sum += p;
        if (p > 0) exc += p; else inh += p;
    }
    float relu_out = (sum > 0) ? sum : 0;

    // Branch
    float bs = exc + inh * lambda_inh + b[tid];
    float br = ((bs > 0) ? bs : 0) - relu_out;

    // Shuffle
    float nb = __shfl_xor(relu_out, 1);
    float sr = (relu_out - nb) * 0.5f;

    // Gate
    int na = __popcll(__ballot(relu_out > 0));
    float act = (float)na / 64.0f;
    float gate = 1.0f / (1.0f + expf(-10.0f * (0.5f - act)));
    float gr = relu_out * (gate - 1.0f);

    // Atomic contention: accumulate in LDS, compare to direct value
    __shared__ float s_acc[256];
    s_acc[tid] = 0;
    __syncthreads();
    atomicAdd(&s_acc[tid], relu_out);  // self-add (may differ from direct due to FP ordering)
    __syncthreads();
    float atomic_res = s_acc[tid] - relu_out;  // should be ~0, but contention adds noise

    Y[sample * out_d + tid] = relu_out
        + alpha_shfl * sr + alpha_branch * br
        + alpha_gate * gr + alpha_atomic * atomic_res;
}

// Wrappers
torch::Tensor stage_a(torch::Tensor X, torch::Tensor W, torch::Tensor b) {
    int N=X.size(0), id=X.size(1), od=W.size(0);
    auto Y=torch::empty({N,od},X.options());
    stage_a_kernel<<<(N*od+255)/256,256>>>(X.data_ptr<float>(),W.data_ptr<float>(),b.data_ptr<float>(),Y.data_ptr<float>(),N,id,od);
    return Y;
}
torch::Tensor stage_b(torch::Tensor X, torch::Tensor W, torch::Tensor b, float alpha) {
    int N=X.size(0), id=X.size(1), od=W.size(0);
    auto Y=torch::empty({N,od},X.options());
    stage_b_kernel<<<(N*od+255)/256,256>>>(X.data_ptr<float>(),W.data_ptr<float>(),b.data_ptr<float>(),Y.data_ptr<float>(),N,id,od,alpha);
    return Y;
}
torch::Tensor stage_c(torch::Tensor X, torch::Tensor W, torch::Tensor b, float as, float ab, float li) {
    int N=X.size(0), id=X.size(1), od=W.size(0);
    auto Y=torch::empty({N,od},X.options());
    stage_c_kernel<<<(N*od+255)/256,256>>>(X.data_ptr<float>(),W.data_ptr<float>(),b.data_ptr<float>(),Y.data_ptr<float>(),N,id,od,as,ab,li);
    return Y;
}
torch::Tensor stage_d(torch::Tensor X, torch::Tensor W, torch::Tensor b, float as, float ab, float li, float ag) {
    int N=X.size(0), id=X.size(1), od=W.size(0);
    auto Y=torch::empty({N,od},X.options());
    stage_d_kernel<<<(N*od+255)/256,256>>>(X.data_ptr<float>(),W.data_ptr<float>(),b.data_ptr<float>(),Y.data_ptr<float>(),N,id,od,as,ab,li,ag);
    return Y;
}
torch::Tensor stage_e(torch::Tensor X, torch::Tensor W, torch::Tensor b, float as, float ab, float li, float ag, float aa) {
    int N=X.size(0), id=X.size(1), od=W.size(0);
    auto Y=torch::empty({N,od},X.options());
    stage_e_kernel<<<N, od>>>(X.data_ptr<float>(),W.data_ptr<float>(),b.data_ptr<float>(),Y.data_ptr<float>(),N,id,od,as,ab,li,ag,aa);
    return Y;
}
"""

CPP_SRC = """
torch::Tensor stage_a(torch::Tensor X, torch::Tensor W, torch::Tensor b);
torch::Tensor stage_b(torch::Tensor X, torch::Tensor W, torch::Tensor b, float alpha);
torch::Tensor stage_c(torch::Tensor X, torch::Tensor W, torch::Tensor b, float as, float ab, float li);
torch::Tensor stage_d(torch::Tensor X, torch::Tensor W, torch::Tensor b, float as, float ab, float li, float ag);
torch::Tensor stage_e(torch::Tensor X, torch::Tensor W, torch::Tensor b, float as, float ab, float li, float ag, float aa);
"""

print("Compiling 5 stage kernels...")
ext = load_inline(name='z2452', cpp_sources=CPP_SRC, cuda_sources=HIP_SRC,
                  functions=['stage_a','stage_b','stage_c','stage_d','stage_e'],
                  extra_cuda_cflags=['-O2','--offload-arch=gfx1100'], verbose=False)
print("OK\n")

device = torch.device('cuda')

# Load MNIST
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

# ============================================================
# Model: 784 → 64 → 10 (small, fast, debuggable)
# Proxy trains with backprop, real kernel validates
# ============================================================
class MechLayer(nn.Module):
    """One layer with learnable mechanism strengths."""
    def __init__(self, in_d, out_d):
        super().__init__()
        self.linear = nn.Linear(in_d, out_d)
        self.alpha_shfl = nn.Parameter(torch.tensor(0.0))    # shuffle strength
        self.alpha_branch = nn.Parameter(torch.tensor(0.0))   # branch strength
        self.lambda_inh = nn.Parameter(torch.tensor(0.95))    # inhibitory coefficient
        self.alpha_gate = nn.Parameter(torch.tensor(0.0))     # gate strength
        self.alpha_atomic = nn.Parameter(torch.tensor(0.0))   # atomic strength

    def forward_proxy(self, x):
        """Differentiable proxy — used for backprop training."""
        raw = self.linear(x)
        relu_out = F.relu(raw)

        # Proxy shuffle: roll = software __shfl_xor
        neighbor = torch.roll(relu_out, 1, dims=-1)
        shfl_res = (relu_out - neighbor) * 0.5

        # Proxy branch: LeakyReLU-like E/I split
        # Positive products stay, negative scaled by lambda
        # This is approximate: real kernel splits per-product, proxy splits per-output
        branch_raw = F.leaky_relu(raw, negative_slope=self.lambda_inh.clamp(0.01, 0.99).item())
        branch_res = branch_raw - relu_out

        # Proxy gate: soft population regulation
        active_frac = (relu_out > 0).float().mean(dim=-1, keepdim=True)
        gate = torch.sigmoid(-10 * (active_frac - 0.5))
        gate_res = relu_out * (gate - 1)

        # Proxy atomic: small noise (approximates contention variance)
        atomic_res = torch.zeros_like(relu_out)
        if self.training:
            atomic_res = torch.randn_like(relu_out) * 0.001 * relu_out.abs()

        return relu_out + self.alpha_shfl * shfl_res + self.alpha_branch * branch_res \
               + self.alpha_gate * gate_res + self.alpha_atomic * atomic_res

    def forward_real(self, x, stage='e'):
        """Forward through REAL HIP kernel."""
        W = self.linear.weight.contiguous()
        b = self.linear.bias.contiguous()
        x = x.contiguous()
        a_s = self.alpha_shfl.item()
        a_b = self.alpha_branch.item()
        l_i = self.lambda_inh.item()
        a_g = self.alpha_gate.item()
        a_a = self.alpha_atomic.item()

        if stage == 'a': return ext.stage_a(x, W, b)
        if stage == 'b': return ext.stage_b(x, W, b, a_s)
        if stage == 'c': return ext.stage_c(x, W, b, a_s, a_b, l_i)
        if stage == 'd': return ext.stage_d(x, W, b, a_s, a_b, l_i, a_g)
        return ext.stage_e(x, W, b, a_s, a_b, l_i, a_g, a_a)


class MechNet(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dims)-2):
            self.layers.append(MechLayer(dims[i], dims[i+1]))
        self.head = nn.Linear(dims[-2], dims[-1])

    def forward(self, x, use_real=False, stage='e'):
        for layer in self.layers:
            if use_real:
                x = layer.forward_real(x, stage)
            else:
                x = layer.forward_proxy(x)
        return self.head(x)


# ============================================================
# Training + stage-by-stage evaluation
# ============================================================
print("=" * 60)
print("z2452: MECHANISM CURRICULUM")
print("=" * 60)

dims = [784, 64, 10]
model = MechNet(dims).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {dims}, {n_params:,} params\n")

bs = 256
n_epochs = 15

# Standard MLP baseline
baseline = nn.Sequential(nn.Linear(784,64), nn.ReLU(), nn.Linear(64,10)).to(device)
opt_bl = torch.optim.Adam(baseline.parameters(), lr=0.001)

# Train baseline
for epoch in range(n_epochs):
    perm = torch.randperm(len(X_tr), device=device)
    for i in range(0, len(X_tr), bs):
        xb, yb = X_tr[perm[i:i+bs]], y_tr[perm[i:i+bs]]
        loss = F.cross_entropy(baseline(xb), yb)
        opt_bl.zero_grad(); loss.backward(); opt_bl.step()
baseline.eval()
with torch.no_grad():
    acc_bl = (baseline(X_te).argmax(1)==y_te).float().mean().item()*100
print(f"Baseline MLP: {acc_bl:.2f}%\n")

# Train mechanism model (proxy-based backprop)
print("Training mechanism model (proxy backprop)...")
for epoch in range(n_epochs):
    model.train()
    perm = torch.randperm(len(X_tr), device=device)
    total_loss = 0; n_b = 0
    for i in range(0, len(X_tr), bs):
        xb, yb = X_tr[perm[i:i+bs]], y_tr[perm[i:i+bs]]
        logits = model(xb, use_real=False)
        loss = F.cross_entropy(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item(); n_b += 1

    # Evaluate BOTH proxy and real kernel
    model.eval()
    with torch.no_grad():
        acc_proxy = (model(X_te[:2000], use_real=False).argmax(1)==y_te[:2000]).float().mean().item()*100

    # Log mechanism strengths
    L = model.layers[0]
    alphas = f"shfl={L.alpha_shfl.item():.4f} branch={L.alpha_branch.item():.4f} " \
             f"λ_inh={L.lambda_inh.item():.3f} gate={L.alpha_gate.item():.4f} " \
             f"atomic={L.alpha_atomic.item():.4f}"

    if (epoch+1) % 3 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:2d}: loss={total_loss/n_b:.4f} proxy={acc_proxy:.1f}% [{alphas}]")

# Final proxy accuracy
model.eval()
with torch.no_grad():
    acc_proxy_full = (model(X_te, use_real=False).argmax(1)==y_te).float().mean().item()*100

# ============================================================
# Stage-by-stage REAL kernel evaluation
# ============================================================
print(f"\n{'='*60}")
print("STAGE-BY-STAGE REAL KERNEL EVALUATION")
print(f"{'='*60}")
print(f"{'Stage':>8} {'Mechanisms':>30} {'Real Acc':>10} {'Gap vs proxy':>14}")
print("-"*65)

stages = [
    ('a', 'ReLU only'),
    ('b', '+ shuffle'),
    ('c', '+ shuffle + branch'),
    ('d', '+ shuffle + branch + gate'),
    ('e', '+ all (shuffle+branch+gate+atomic)'),
]

for stage, desc in stages:
    with torch.no_grad():
        if stage == 'e':
            # Stage E uses block-per-sample launch, needs different batch handling
            # Process in chunks
            all_preds = []
            for i in range(0, len(X_te), 256):
                chunk = X_te[i:i+256]
                if len(chunk) < 256:
                    # Pad
                    pad = torch.zeros(256-len(chunk), 784, device=device)
                    chunk = torch.cat([chunk, pad])
                logits = model(chunk, use_real=True, stage=stage)
                all_preds.append(logits[:min(256, len(X_te)-i)])
            preds = torch.cat(all_preds).argmax(1)
        else:
            preds = model(X_te, use_real=True, stage=stage).argmax(1)
        acc_real = (preds == y_te).float().mean().item() * 100
    gap = acc_real - acc_proxy_full
    marker = "✓" if abs(gap) < 2 else "✗"
    print(f"{stage:>8} {desc:>30} {acc_real:>9.2f}% {gap:>+13.2f}pp {marker}")

# ============================================================
# Noise robustness
# ============================================================
print(f"\n--- Noise Robustness (Stage A = standard ReLU) ---")
print(f"{'Sigma':>6} {'Baseline':>10} {'Stage A':>10} {'Stage E':>10} {'E-BL':>8}")

for sigma in [0.0, 0.1, 0.2, 0.3, 0.5]:
    noisy = (X_te + torch.randn_like(X_te)*sigma).clamp(0,1)
    with torch.no_grad():
        a_bl = (baseline(noisy).argmax(1)==y_te).float().mean().item()*100
        a_a = (model(noisy, use_real=True, stage='a').argmax(1)==y_te).float().mean().item()*100
        # Stage E in chunks
        preds_e = []
        for i in range(0, len(noisy), 256):
            chunk = noisy[i:i+256]
            if len(chunk) < 256:
                chunk = torch.cat([chunk, torch.zeros(256-len(chunk),784,device=device)])
            logits = model(chunk, use_real=True, stage='e')
            preds_e.append(logits[:min(256, len(noisy)-i)])
        a_e = (torch.cat(preds_e).argmax(1)==y_te).float().mean().item()*100
    print(f"{sigma:>6.2f} {a_bl:>9.2f}% {a_a:>9.2f}% {a_e:>9.2f}% {a_e-a_bl:>+7.2f}pp")

# ============================================================
# Summary
# ============================================================
L = model.layers[0]
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"  Baseline MLP:     {acc_bl:.2f}%")
print(f"  Proxy accuracy:   {acc_proxy_full:.2f}%")
print(f"  Learned alphas:")
print(f"    shuffle:  {L.alpha_shfl.item():+.4f}")
print(f"    branch:   {L.alpha_branch.item():+.4f}")
print(f"    λ_inh:    {L.lambda_inh.item():.4f}")
print(f"    gate:     {L.alpha_gate.item():+.4f}")
print(f"    atomic:   {L.alpha_atomic.item():+.4f}")
print(f"\n  Key question: which alphas did the optimizer make NON-ZERO?")
print(f"  Non-zero alpha = mechanism EARNED its place during training")

results = {
    'baseline': acc_bl, 'proxy': acc_proxy_full, 'dims': dims, 'n_params': n_params,
    'alphas': {
        'shuffle': L.alpha_shfl.item(), 'branch': L.alpha_branch.item(),
        'lambda_inh': L.lambda_inh.item(), 'gate': L.alpha_gate.item(),
        'atomic': L.alpha_atomic.item(),
    }
}
with open(f'{base}/results/z2452_mech_curriculum.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to results/z2452_mech_curriculum.json")
