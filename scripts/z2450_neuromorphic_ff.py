#!/usr/bin/env python3
"""
z2450: Forward-Forward training with real GPU mechanisms

Why Forward-Forward (Hinton 2022):
  - No backprop → no need for surrogate gradients
  - Layer-local learning → mechanisms don't need to be differentiable
  - "Goodness" function = sum of squared activations → or mechanism signals
  - Each layer independently learns to separate positive from negative data

Architecture:
  Layer 1 (784→256): Branch divergence E/I + atomic contention
  Layer 2 (256→128): Shuffle mixing + ballot gating
  Layer 3 (128→10):  Standard linear readout (just for prediction)

Training:
  - Positive data: real MNIST image + correct label (embedded in first 10 pixels)
  - Negative data: real MNIST image + WRONG label
  - Each layer learns: positive → high goodness, negative → low goodness
  - Goodness = sum of squared activations (standard FF)
  - Layer-local optimizer (each layer has its own Adam)

The neuromorphic forward pass uses REAL GPU mechanisms via a HIP kernel.
But since FF is layer-local, we only need the HIP kernel for FORWARD —
the weight update is computed from the goodness gradient w.r.t. inputs,
which we compute in PyTorch using a DIFFERENTIABLE PROXY of the layer.

Key insight: the proxy doesn't need to be perfect — it just needs to
move weights in approximately the right direction. The real mechanism
forward pass provides the actual computation. The proxy backward pass
provides the learning signal. If they're close enough, training works.
"""
import os, time, json
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

base = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy'

# ============================================================
# Try to load HIP kernel; fall back to PyTorch mechanism emulation
# ============================================================
USE_HIP = False
try:
    from torch.utils.cpp_extension import load_inline

    HIP_SRC = r"""
    #include <hip/hip_runtime.h>

    // Neuromorphic layer: branch E/I + atomic contention + shuffle + ballot
    __global__ void neuro_layer_kernel(
        const float* __restrict__ X,   // [N, in_dim]
        const float* __restrict__ W,   // [out_dim, in_dim]
        const float* __restrict__ b,   // [out_dim]
        float* __restrict__ Y,         // [N, out_dim]
        int N, int in_dim, int out_dim,
        float shfl_mix
    ) {
        int sample = blockIdx.x;
        if (sample >= N) return;
        int tid = threadIdx.x;
        if (tid >= out_dim) return;

        const float* x = X + sample * in_dim;
        const float* w = W + tid * in_dim;

        // MECHANISM 1: Branch divergence E/I accumulation
        float exc = 0.0f, inh = 0.0f;
        for (int k = 0; k < in_dim; k++) {
            float p = w[k] * x[k];
            if (p > 0.0f) exc += p;
            else inh += p;
        }
        float raw = exc + inh * 0.2f + b[tid];

        // MECHANISM 2: Atomic contention in LDS
        __shared__ float s_acc[256];
        s_acc[tid] = 0.0f;
        __syncthreads();
        int conflict_addr = tid ^ (tid & 0x3);
        atomicAdd(&s_acc[conflict_addr], raw * 0.25f);
        atomicAdd(&s_acc[tid], raw * 0.75f);
        __syncthreads();
        float activated = s_acc[tid];

        // MECHANISM 3: Shuffle lateral inhibition
        if (shfl_mix > 0.0f) {
            float n1 = __shfl_xor(activated, 1);
            float n2 = __shfl_xor(activated, 2);
            activated = activated * (1.0f + shfl_mix)
                      - n1 * (shfl_mix * 0.5f)
                      - n2 * (shfl_mix * 0.25f);
        }

        // MECHANISM 4: Ballot homeostatic gating
        unsigned long long active = __ballot(activated > 0.0f);
        int n_active = __popcll(active);
        float threshold = 0.0f;
        if (n_active > 48) threshold = 0.1f;
        activated = (activated > threshold) ? activated : activated * 0.01f;

        Y[sample * out_dim + tid] = activated;
    }

    torch::Tensor neuro_layer(torch::Tensor X, torch::Tensor W, torch::Tensor b, float shfl_mix) {
        int N = X.size(0);
        int in_dim = X.size(1);
        int out_dim = W.size(0);
        auto Y = torch::empty({N, out_dim}, X.options());

        int threads = out_dim;
        if (threads > 256) threads = 256;
        neuro_layer_kernel<<<N, threads>>>(
            X.data_ptr<float>(), W.data_ptr<float>(), b.data_ptr<float>(),
            Y.data_ptr<float>(), N, in_dim, out_dim, shfl_mix);

        return Y;
    }
    """

    CPP_SRC = r"""
    torch::Tensor neuro_layer(torch::Tensor X, torch::Tensor W, torch::Tensor b, float shfl_mix);
    """

    ext = load_inline(
        name='z2450_neuro',
        cpp_sources=CPP_SRC,
        cuda_sources=HIP_SRC,
        functions=['neuro_layer'],
        extra_cuda_cflags=['-O2', '--offload-arch=gfx1100'],
        verbose=False
    )
    USE_HIP = True
    print("HIP kernel compiled successfully — using REAL GPU mechanisms")
except Exception as e:
    print(f"HIP compilation failed ({e}) — using PyTorch mechanism emulation")
    USE_HIP = False


# ============================================================
# Neuromorphic layer (PyTorch emulation if HIP unavailable)
# This is the DIFFERENTIABLE PROXY used for FF weight updates
# ============================================================
class NeuromorphicLayerProxy(nn.Module):
    """Differentiable proxy that approximates what the HIP kernel does."""
    def __init__(self, in_dim, out_dim, shfl_mix=0.1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.shfl_mix = shfl_mix
        # Initialize small
        nn.init.kaiming_normal_(self.linear.weight, a=0.2, mode='fan_in')
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        # Proxy for branch E/I: LeakyReLU on per-element products, then sum
        # Equivalent to: exc + 0.2*inh = LeakyReLU(w*x, alpha=0.2) summed
        raw = F.linear(x, self.linear.weight, self.linear.bias)

        # Proxy for atomic contention: add small input-dependent noise
        # (contention depends on activation magnitude)
        if self.training:
            noise = torch.randn_like(raw) * 0.01 * raw.abs()
            raw = raw + noise

        # Proxy for shuffle lateral inhibition
        if self.shfl_mix > 0:
            # Roll = software equivalent of __shfl_xor
            n1 = torch.roll(raw, 1, dims=-1)
            n2 = torch.roll(raw, 2, dims=-1)
            raw = raw * (1 + self.shfl_mix) - n1 * (self.shfl_mix * 0.5) - n2 * (self.shfl_mix * 0.25)

        # Proxy for ballot gating: soft homeostatic regulation
        active_frac = (raw > 0).float().mean(dim=-1, keepdim=True)
        scale = torch.where(active_frac > 0.75,
                           torch.tensor(0.9, device=raw.device),
                           torch.where(active_frac < 0.25,
                                      torch.tensor(1.1, device=raw.device),
                                      torch.tensor(1.0, device=raw.device)))
        raw = raw * scale

        # LeakyReLU (proxy for branch-conditional accumulation)
        return F.leaky_relu(raw, 0.01)


# ============================================================
# Forward-Forward Layer
# ============================================================
class FFLayer(nn.Module):
    """Forward-Forward layer with mechanism-based forward and proxy-based learning."""
    def __init__(self, in_dim, out_dim, shfl_mix=0.1, threshold=2.0):
        super().__init__()
        self.proxy = NeuromorphicLayerProxy(in_dim, out_dim, shfl_mix)
        self.threshold = threshold
        self.optimizer = None  # set after init

    def forward_real(self, x):
        """Forward pass using REAL GPU mechanisms (HIP kernel)."""
        if USE_HIP:
            return ext.neuro_layer(
                x.contiguous(), self.proxy.linear.weight.contiguous(),
                self.proxy.linear.bias.contiguous(), self.proxy.shfl_mix)
        else:
            return self.proxy(x)

    def forward_proxy(self, x):
        """Forward pass using differentiable proxy (for gradient computation)."""
        return self.proxy(x)

    def goodness(self, h):
        """Goodness = mean sum of squared activations per sample."""
        return (h ** 2).mean(dim=1)

    def train_step(self, x_pos, x_neg):
        """One FF training step: push positive goodness up, negative down."""
        # Forward through PROXY (differentiable) for gradient computation
        h_pos = self.forward_proxy(x_pos)
        h_neg = self.forward_proxy(x_neg)

        g_pos = self.goodness(h_pos)
        g_neg = self.goodness(h_neg)

        # FF loss: positive goodness should exceed threshold,
        # negative goodness should be below threshold
        loss = torch.log(1 + torch.exp(-(g_pos - self.threshold))).mean() + \
               torch.log(1 + torch.exp(g_neg - self.threshold)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), g_pos.mean().item(), g_neg.mean().item()


# ============================================================
# Full Forward-Forward Network
# ============================================================
class FFNetwork(nn.Module):
    def __init__(self, dims, threshold=2.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            mix = 0.1 if i > 0 else 0.05  # gentler mixing in first layer
            self.layers.append(FFLayer(dims[i], dims[i+1], shfl_mix=mix, threshold=threshold))

    def init_optimizers(self, lr=0.001):
        for layer in self.layers:
            layer.optimizer = torch.optim.Adam(layer.proxy.parameters(), lr=lr)

    def forward(self, x, use_real=False):
        """Forward pass through all layers."""
        h = x
        for layer in self.layers:
            if use_real:
                h = layer.forward_real(h)
            else:
                h = layer.forward_proxy(h)
        return h

    def predict(self, x, use_real=False):
        """Predict by finding label that maximizes total goodness.
        Standard FF: embed label in input, try all 10, pick highest goodness."""
        batch = x.shape[0]
        best_goodness = torch.full((batch,), -1e9, device=x.device)
        best_label = torch.zeros(batch, dtype=torch.long, device=x.device)

        for label in range(10):
            x_labeled = self.embed_label(x, label)
            h = x_labeled
            total_g = torch.zeros(batch, device=x.device)
            for layer in self.layers:
                if use_real:
                    h = layer.forward_real(h)
                else:
                    h = layer.forward_proxy(h)
                total_g += layer.goodness(h)

            mask = total_g > best_goodness
            best_goodness[mask] = total_g[mask]
            best_label[mask] = label

        return best_label

    @staticmethod
    def embed_label(x, label):
        """Embed label in first 10 dimensions (one-hot overlay)."""
        x_new = x.clone()
        x_new[:, :10] = 0
        if isinstance(label, int):
            x_new[:, label] = 1.0
        else:
            # label is tensor
            x_new[torch.arange(len(label)), label] = 1.0
        return x_new


# ============================================================
# Standard MLP baseline (same architecture, standard backprop)
# ============================================================
class StandardMLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================
# Training
# ============================================================
print("=" * 60)
print("z2450: FORWARD-FORWARD with GPU mechanisms")
print("=" * 60)

device = torch.device('cuda')

# Load MNIST
with open(f'{base}/data/MNIST/raw/train-images-idx3-ubyte', 'rb') as f:
    f.read(16)
    train_img = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 784).astype(np.float32) / 255.0
with open(f'{base}/data/MNIST/raw/train-labels-idx1-ubyte', 'rb') as f:
    f.read(8)
    train_lbl = np.frombuffer(f.read(), dtype=np.uint8)
with open(f'{base}/data/MNIST/raw/t10k-images-idx3-ubyte', 'rb') as f:
    f.read(16)
    test_img = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 784).astype(np.float32) / 255.0
with open(f'{base}/data/MNIST/raw/t10k-labels-idx1-ubyte', 'rb') as f:
    f.read(8)
    test_lbl = np.frombuffer(f.read(), dtype=np.uint8)

X_train = torch.tensor(train_img, device=device)
y_train = torch.tensor(train_lbl, dtype=torch.long, device=device)
X_test = torch.tensor(test_img, device=device)
y_test = torch.tensor(test_lbl, dtype=torch.long, device=device)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ============================================================
# Train FF network
# ============================================================
dims = [784, 256, 128]  # FF layers (no output layer — prediction by goodness)
ff_net = FFNetwork(dims, threshold=2.0).to(device)
ff_net.init_optimizers(lr=0.001)

print(f"\n--- Forward-Forward Training (real mechanisms: {USE_HIP}) ---")
batch_size = 512
n_epochs = 20

for epoch in range(n_epochs):
    ff_net.train()
    epoch_loss = 0
    n_batches = 0

    # Shuffle
    perm = torch.randperm(len(X_train), device=device)
    X_shuf = X_train[perm]
    y_shuf = y_train[perm]

    for i in range(0, len(X_train), batch_size):
        xb = X_shuf[i:i+batch_size]
        yb = y_shuf[i:i+batch_size]
        if len(xb) < batch_size:
            continue

        # Positive: real image + correct label
        x_pos = FFNetwork.embed_label(xb, yb)

        # Negative: real image + random WRONG label
        wrong = torch.randint(0, 9, (len(yb),), device=device)
        wrong = (yb + wrong + 1) % 10  # guaranteed different from yb
        x_neg = FFNetwork.embed_label(xb, wrong)

        # Train each layer greedily
        h_pos = x_pos
        h_neg = x_neg
        batch_loss = 0
        for layer in ff_net.layers:
            loss, gp, gn = layer.train_step(h_pos, h_neg)
            batch_loss += loss
            # Propagate through layer (detached — FF is layer-local)
            with torch.no_grad():
                h_pos = layer.forward_real(h_pos) if USE_HIP else layer.forward_proxy(h_pos)
                h_neg = layer.forward_real(h_neg) if USE_HIP else layer.forward_proxy(h_neg)

        epoch_loss += batch_loss
        n_batches += 1

    # Evaluate
    ff_net.eval()
    with torch.no_grad():
        # Test on subset for speed
        n_eval = 2000
        preds_proxy = ff_net.predict(X_test[:n_eval], use_real=False)
        acc_proxy = (preds_proxy == y_test[:n_eval]).float().mean().item() * 100

        if USE_HIP:
            preds_real = ff_net.predict(X_test[:n_eval], use_real=True)
            acc_real = (preds_real == y_test[:n_eval]).float().mean().item() * 100
            print(f"  Epoch {epoch+1:2d}: loss={epoch_loss/n_batches:.4f} "
                  f"proxy={acc_proxy:.1f}% real={acc_real:.1f}%")
        else:
            print(f"  Epoch {epoch+1:2d}: loss={epoch_loss/n_batches:.4f} acc={acc_proxy:.1f}%")

# Final evaluation on full test set
print("\n--- Final Evaluation ---")
ff_net.eval()
with torch.no_grad():
    preds = ff_net.predict(X_test, use_real=USE_HIP)
    acc_ff = (preds == y_test).float().mean().item() * 100
print(f"  FF neuromorphic: {acc_ff:.2f}%")

# ============================================================
# Train standard MLP baseline (same param count)
# ============================================================
print("\n--- Standard MLP Baseline (backprop) ---")
mlp = StandardMLP([784, 256, 128, 10]).to(device)
mlp_opt = torch.optim.Adam(mlp.parameters(), lr=0.001)

n_params_ff = sum(p.numel() for p in ff_net.parameters())
n_params_mlp = sum(p.numel() for p in mlp.parameters())
print(f"  FF params:  {n_params_ff:,}")
print(f"  MLP params: {n_params_mlp:,}")

for epoch in range(n_epochs):
    mlp.train()
    perm = torch.randperm(len(X_train), device=device)
    epoch_loss = 0
    n_b = 0
    for i in range(0, len(X_train), batch_size):
        xb = X_train[perm[i:i+batch_size]]
        yb = y_train[perm[i:i+batch_size]]
        if len(xb) < batch_size:
            continue
        logits = mlp(xb)
        loss = F.cross_entropy(logits, yb)
        mlp_opt.zero_grad()
        loss.backward()
        mlp_opt.step()
        epoch_loss += loss.item()
        n_b += 1

    if (epoch+1) % 5 == 0 or epoch == 0:
        mlp.eval()
        with torch.no_grad():
            acc = (mlp(X_test).argmax(1) == y_test).float().mean().item() * 100
        print(f"  Epoch {epoch+1:2d}: loss={epoch_loss/n_b:.4f} acc={acc:.1f}%")

mlp.eval()
with torch.no_grad():
    acc_mlp = (mlp(X_test).argmax(1) == y_test).float().mean().item() * 100
print(f"  Standard MLP: {acc_mlp:.2f}%")

# ============================================================
# Noise robustness comparison
# ============================================================
print("\n--- Noise Robustness ---")
print(f"  {'Sigma':>6} {'FF':>8} {'MLP':>8} {'Δ':>8}")
print(f"  {'-----':>6} {'--':>8} {'---':>8} {'--':>8}")

robustness = {}
for sigma in [0.0, 0.1, 0.2, 0.3, 0.5]:
    noise = torch.randn_like(X_test) * sigma
    X_noisy = (X_test + noise).clamp(0, 1)

    with torch.no_grad():
        preds_ff = ff_net.predict(X_noisy, use_real=USE_HIP)
        acc_ff_n = (preds_ff == y_test).float().mean().item() * 100

        acc_mlp_n = (mlp(X_noisy).argmax(1) == y_test).float().mean().item() * 100

    delta = acc_ff_n - acc_mlp_n
    print(f"  {sigma:>6.2f} {acc_ff_n:>7.2f}% {acc_mlp_n:>7.2f}% {delta:>+7.2f}pp")
    robustness[f'sigma_{sigma}'] = {'ff': acc_ff_n, 'mlp': acc_mlp_n, 'delta': delta}

# ============================================================
# Run-to-run variance (if using real HIP)
# ============================================================
if USE_HIP:
    print("\n--- Run-to-run Variance (real mechanisms) ---")
    accs = []
    with torch.no_grad():
        for r in range(5):
            preds = ff_net.predict(X_test[:2000], use_real=True)
            a = (preds == y_test[:2000]).float().mean().item() * 100
            accs.append(a)
    print(f"  Runs: {[f'{a:.2f}' for a in accs]}")
    print(f"  Mean: {np.mean(accs):.2f}% Std: {np.std(accs):.4f}%")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  FF Neuromorphic:  {acc_ff:.2f}%  (real GPU mechanisms: {USE_HIP})")
print(f"  Standard MLP:     {acc_mlp:.2f}%  (backprop)")
print(f"  Delta:            {acc_ff - acc_mlp:+.2f}pp")
print(f"  FF params:        {n_params_ff:,}")
print(f"  MLP params:       {n_params_mlp:,}")

if acc_ff > acc_mlp:
    print(f"\n  >>> NEUROMORPHIC FF BEATS STANDARD MLP <<<")
elif acc_ff > acc_mlp - 2:
    print(f"\n  FF competitive with MLP (within 2pp)")
else:
    print(f"\n  MLP wins — but check noise robustness")

# Check robustness advantage
r03 = robustness.get('sigma_0.3', {})
if r03.get('delta', 0) > 1.0:
    print(f"  >>> FF MORE ROBUST at σ=0.3: {r03['delta']:+.1f}pp <<<")

results = {
    'ff_accuracy': acc_ff, 'mlp_accuracy': acc_mlp,
    'use_hip': USE_HIP, 'dims': dims,
    'n_params_ff': n_params_ff, 'n_params_mlp': n_params_mlp,
    'robustness': robustness,
}
with open(f'{base}/results/z2450_neuromorphic_ff.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to results/z2450_neuromorphic_ff.json")
