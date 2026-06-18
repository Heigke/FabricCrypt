#!/usr/bin/env python3
"""
z2451: Forward-Forward training THROUGH the real HIP kernel

z2450 showed: proxy trains to 87% but real kernel gives 28%.
Gap = proxy and kernel compute different things.

Fix: use REAL kernel in the FF goodness computation.
Since FF is layer-local and goodness is scalar, we can compute
the gradient via finite differences on the REAL kernel output.

For each weight w_ij:
  goodness(w + ε) - goodness(w - ε)
  grad ≈ --------------------------------
                    2ε

This is slow (2 forward passes per weight) but HONEST —
the gradient goes through the ACTUAL hardware mechanisms.

Optimization: use random projection gradient estimation instead of
per-weight finite diff. Sample k random directions, estimate gradient
in those directions. Much faster for high-dim parameters.
"""
import os, time, json
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

base = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy'

# ============================================================
# Load HIP kernel
# ============================================================
from torch.utils.cpp_extension import load_inline

HIP_SRC = r"""
#include <hip/hip_runtime.h>

__global__ void neuro_fwd(
    const float* __restrict__ X,
    const float* __restrict__ W,
    const float* __restrict__ b,
    float* __restrict__ Y,
    int N, int in_dim, int out_dim, float mix
) {
    int sample = blockIdx.x;
    if (sample >= N) return;
    int tid = threadIdx.x;
    if (tid >= out_dim) return;

    const float* x = X + sample * in_dim;
    const float* w = W + tid * in_dim;

    // Branch E/I accumulation
    float exc = 0, inh = 0;
    for (int k = 0; k < in_dim; k++) {
        float p = w[k] * x[k];
        if (p > 0) exc += p; else inh += p;
    }
    float raw = exc + inh * 0.2f + b[tid];

    // Atomic contention
    __shared__ float s[256];
    s[tid] = 0;
    __syncthreads();
    atomicAdd(&s[tid ^ (tid & 3)], raw * 0.25f);
    atomicAdd(&s[tid], raw * 0.75f);
    __syncthreads();
    float act = s[tid];

    // Shuffle mixing
    if (mix > 0) {
        float n1 = __shfl_xor(act, 1);
        float n2 = __shfl_xor(act, 2);
        act = act * (1+mix) - n1*(mix*0.5f) - n2*(mix*0.25f);
    }

    // Ballot gating
    int na = __popcll(__ballot(act > 0));
    float th = (na > 48) ? 0.1f : 0.0f;
    act = (act > th) ? act : act * 0.01f;

    Y[sample * out_dim + tid] = act;
}

torch::Tensor neuro_layer(torch::Tensor X, torch::Tensor W, torch::Tensor b, float mix) {
    int N = X.size(0), in_d = X.size(1), out_d = W.size(0);
    auto Y = torch::empty({N, out_d}, X.options());
    neuro_fwd<<<N, out_d>>>(X.data_ptr<float>(), W.data_ptr<float>(),
                             b.data_ptr<float>(), Y.data_ptr<float>(),
                             N, in_d, out_d, mix);
    return Y;
}
"""

CPP_SRC = "torch::Tensor neuro_layer(torch::Tensor X, torch::Tensor W, torch::Tensor b, float mix);"

print("Compiling HIP kernel...")
ext = load_inline(name='z2451', cpp_sources=CPP_SRC, cuda_sources=HIP_SRC,
                  functions=['neuro_layer'],
                  extra_cuda_cflags=['-O2', '--offload-arch=gfx1100'], verbose=False)
print("OK — REAL GPU mechanisms active")

device = torch.device('cuda')

# ============================================================
# Load MNIST
# ============================================================
with open(f'{base}/data/MNIST/raw/train-images-idx3-ubyte', 'rb') as f:
    f.read(16); train_img = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1,784).astype(np.float32)/255
with open(f'{base}/data/MNIST/raw/train-labels-idx1-ubyte', 'rb') as f:
    f.read(8); train_lbl = np.frombuffer(f.read(), dtype=np.uint8)
with open(f'{base}/data/MNIST/raw/t10k-images-idx3-ubyte', 'rb') as f:
    f.read(16); test_img = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1,784).astype(np.float32)/255
with open(f'{base}/data/MNIST/raw/t10k-labels-idx1-ubyte', 'rb') as f:
    f.read(8); test_lbl = np.frombuffer(f.read(), dtype=np.uint8)

X_train = torch.tensor(train_img, device=device)
y_train = torch.tensor(train_lbl, dtype=torch.long, device=device)
X_test = torch.tensor(test_img, device=device)
y_test = torch.tensor(test_lbl, dtype=torch.long, device=device)

# ============================================================
# Forward-Forward with REAL kernel + random projection gradients
# ============================================================
def embed_label(x, label):
    x2 = x.clone()
    x2[:, :10] = 0
    if isinstance(label, int):
        x2[:, label] = 1.0
    else:
        x2[torch.arange(len(label), device=x.device), label] = 1.0
    return x2

def real_forward(x, W, b, mix=0.1):
    """Run through REAL HIP kernel."""
    return ext.neuro_layer(x.contiguous(), W.contiguous(), b.contiguous(), mix)

def goodness(h):
    """FF goodness = mean squared activation per sample."""
    return (h ** 2).mean(dim=1)

def ff_loss(g_pos, g_neg, threshold=2.0):
    """FF loss: push positive above threshold, negative below."""
    return (torch.log(1 + torch.exp(-(g_pos - threshold))).mean() +
            torch.log(1 + torch.exp(g_neg - threshold)).mean())

# Layer parameters
dims = [(784, 256), (256, 128)]
weights = []
biases = []
for in_d, out_d in dims:
    W = torch.randn(out_d, in_d, device=device) * (2.0 / in_d) ** 0.5
    b = torch.zeros(out_d, device=device)
    weights.append(W)
    biases.append(b)

# ============================================================
# Training with random projection gradient estimation
#
# Instead of finite-diff per weight (too slow for 200k+ params),
# sample k random directions and estimate gradient component in each.
# This is "evolution strategies" / "random search" gradient estimation.
# ============================================================
print("\n" + "=" * 60)
print("z2451: FF training THROUGH real GPU mechanisms")
print(f"  Layer 0: {dims[0][0]}→{dims[0][1]} ({weights[0].numel()} weights)")
print(f"  Layer 1: {dims[1][0]}→{dims[1][1]} ({weights[1].numel()} weights)")
print("  Gradient: random projection (k=50 directions per step)")
print("=" * 60)

batch_size = 256
n_epochs = 30
k_directions = 50  # random directions per gradient step
eps = 0.01  # perturbation size
lr = 0.005

def evaluate(n_eval=2000):
    """Evaluate accuracy using real kernel."""
    correct = 0
    with torch.no_grad():
        for label in range(10):
            x_lab = embed_label(X_test[:n_eval], label)
            h = x_lab
            total_g = torch.zeros(n_eval, device=device)
            for L in range(len(weights)):
                h = real_forward(h, weights[L], biases[L], 0.1 if L > 0 else 0.05)
                total_g += goodness(h)
            if label == 0:
                best_g = total_g.clone()
                best_l = torch.zeros(n_eval, dtype=torch.long, device=device)
            else:
                mask = total_g > best_g
                best_g[mask] = total_g[mask]
                best_l[mask] = label
    return (best_l == y_test[:n_eval]).float().mean().item() * 100

# Initial accuracy
acc0 = evaluate()
print(f"\nInitial accuracy (random weights): {acc0:.1f}%")

for epoch in range(n_epochs):
    t0 = time.time()
    perm = torch.randperm(len(X_train), device=device)
    epoch_loss = 0
    n_steps = 0

    for bi in range(0, min(len(X_train), 10000), batch_size):
        xb = X_train[perm[bi:bi+batch_size]]
        yb = y_train[perm[bi:bi+batch_size]]
        if len(xb) < batch_size:
            continue

        x_pos = embed_label(xb, yb)
        wrong = (yb + torch.randint(1, 10, (len(yb),), device=device)) % 10
        x_neg = embed_label(xb, wrong)

        # Train each layer independently (FF is layer-local)
        h_pos = x_pos
        h_neg = x_neg

        for L in range(len(weights)):
            W = weights[L]
            b = biases[L]
            mix = 0.05 if L == 0 else 0.1

            # Current goodness
            with torch.no_grad():
                hp = real_forward(h_pos, W, b, mix)
                hn = real_forward(h_neg, W, b, mix)
                gp = goodness(hp).mean()
                gn = goodness(hn).mean()
                loss0 = ff_loss(goodness(hp), goodness(hn))

            # Random projection gradient estimation
            grad_W = torch.zeros_like(W)
            grad_b = torch.zeros_like(b)

            for _ in range(k_directions):
                # Random direction for W
                dW = torch.randn_like(W)
                dW = dW / (dW.norm() + 1e-10) * eps

                # Perturb W in +direction
                with torch.no_grad():
                    hp_p = real_forward(h_pos, W + dW, b, mix)
                    hn_p = real_forward(h_neg, W + dW, b, mix)
                    loss_p = ff_loss(goodness(hp_p), goodness(hn_p))

                # Gradient component in this direction
                g = (loss_p - loss0) / eps
                grad_W += g * dW

                # Random direction for b
                db = torch.randn_like(b)
                db = db / (db.norm() + 1e-10) * eps
                with torch.no_grad():
                    hp_b = real_forward(h_pos, W, b + db, mix)
                    hn_b = real_forward(h_neg, W, b + db, mix)
                    loss_b = ff_loss(goodness(hp_b), goodness(hn_b))
                g_b = (loss_b - loss0) / eps
                grad_b += g_b * db

            grad_W /= k_directions
            grad_b /= k_directions

            # Update
            weights[L] = W - lr * grad_W
            biases[L] = b - lr * grad_b

            # Propagate (detached, for next layer)
            with torch.no_grad():
                h_pos = real_forward(h_pos, weights[L], biases[L], mix)
                h_neg = real_forward(h_neg, weights[L], biases[L], mix)

            epoch_loss += loss0.item()
            n_steps += 1

    acc = evaluate()
    elapsed = time.time() - t0
    print(f"  Epoch {epoch+1:2d}: loss={epoch_loss/n_steps:.4f} acc={acc:.1f}% ({elapsed:.1f}s)")

# ============================================================
# Final comparison
# ============================================================
acc_final = evaluate(10000)
print(f"\n{'='*60}")
print(f"FINAL: FF through real mechanisms = {acc_final:.2f}%")
print(f"  (MLP backprop baseline from z2450 = 97.88%)")
print(f"{'='*60}")

results = {'ff_real_accuracy': acc_final, 'n_epochs': n_epochs,
           'k_directions': k_directions, 'lr': lr, 'eps': eps}
with open(f'{base}/results/z2451_ff_real_train.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved to results/z2451_ff_real_train.json")
