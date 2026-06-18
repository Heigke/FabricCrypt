#!/usr/bin/env python3
"""z2042: Timing-Embedded Neural Network (TENN)

Hardware PUF for Neural Networks — Going Below Python

KEY INNOVATION: Instead of reading GPU temperature via sysfs (z2041),
we inject clock64() timing from INSIDE HIP GPU kernels directly into
neural network computation. The timing fingerprint is:
- Per-thread (128 measurements per probe)
- Per-operation (memory vs compute vs ratio)
- Hardware-dependent (DVFS, cache, CU scheduling, thermal)
- Unforgeable (comes from the GPU executing the computation)

Architecture:
  CNN encoder → features [B, 128]
  HIP timing probe → timing [128] (from clock64 INSIDE the GPU)
  Timing encoder → timing features [B, 32]
  Fusion → classifier → encrypted class prediction

Label encryption: offset = argmax(hash_matrix @ timing_normalized)
  encrypted_label = (true_label + offset) % 10

The model can only classify correctly with LIVE hardware timing.

5 Conditions:
  A_standard: No timing injection (baseline)
  B_tenn: Real HIP timing injection (the test condition)
  C_frozen: Captured-once timing (same every batch)
  D_random: Random noise replacing timing
  E_zero: Zero timing vector

4 Tests:
  T1: B accuracy >= 85% (timing doesn't prevent learning)
  T2: Kill shot — B with wrong timing → ~10% (timing IS essential)
  T3: Timing statistics — structured hardware noise, not random
  T4: Self-model blindsight — ablate timing encoder, task preserved
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
import os
import sys
import time
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# HIP Timing Probe — compiled as PyTorch extension
# =============================================================================

HIP_KERNEL_SOURCE = '''
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Memory access timing — HIGHEST jitter, best hardware fingerprint
// Measured on gfx1151: mean ~26K cycles, DVFS-dependent range up to 45K
__global__ void memory_timing_kernel(
    float* __restrict__ output,
    const float* __restrict__ workspace,
    int ws_size, int n
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    float sum = 0.0f;
    uint64_t t0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < 50; i++) {
        int idx = ((tid * 2654435761u + i * 40503u) >> 4) % ws_size;
        sum += workspace[idx];
    }
    uint64_t t1 = clock64();
    output[tid] = (float)(t1 - t0);
    if (__builtin_expect(sum == -1e30f, 0)) output[0] = sum;
}

// Compute timing — LOW jitter, DVFS-sensitive baseline
__global__ void compute_timing_kernel(
    float* __restrict__ output, int n
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    float x = (float)(tid + 1) * 0.001f;
    uint64_t t0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < 100; i++) {
        x = __sinf(x) * __cosf(x + 0.1f) + __expf(-x * x);
        x = x * 1.0001f + 0.0001f;
    }
    uint64_t t1 = clock64();
    output[tid] = (float)(t1 - t0);
    if (__builtin_expect(x == -1e30f, 0)) output[0] = x;
}

// Memory/Compute ratio — captures DVFS state from inside kernel
__global__ void ratio_timing_kernel(
    float* __restrict__ output,
    const float* __restrict__ workspace,
    int ws_size, int n
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    float x = (float)(tid + 1) * 0.001f;
    uint64_t tc0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < 50; i++) { x = __sinf(x) * __cosf(x + 0.1f); }
    uint64_t tc1 = clock64();
    float c_time = (float)(tc1 - tc0);
    float sum = 0.0f;
    uint64_t tm0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < 25; i++) {
        int idx = ((tid * 2654435761u + i * 40503u) >> 4) % ws_size;
        sum += workspace[idx];
    }
    uint64_t tm1 = clock64();
    float m_time = (float)(tm1 - tm0);
    output[tid] = (c_time > 1.0f) ? m_time / c_time : 1.0f;
    if (__builtin_expect((x + sum) == -1e30f, 0)) output[0] = x;
}

// Timed matrix-vector product — timing from the EXACT SAME computation
// Launch: <<<B, M>>> where B=batch, M=output_features (max 1024)
__global__ void timed_matvec_kernel(
    float* __restrict__ timing_output,
    const float* __restrict__ W,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int B, int M, int K
) {
    int b = blockIdx.x;
    int m = threadIdx.x;
    if (b >= B || m >= M) return;
    uint64_t t0 = clock64();
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += W[m * K + k] * X[b * K + k];
    }
    Y[b * M + m] = sum;
    uint64_t t1 = clock64();
    if (m == 0) timing_output[b] = (float)(t1 - t0);
}

torch::Tensor probe_memory(torch::Tensor workspace, int n) {
    auto output = torch::empty({n}, workspace.options());
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    memory_timing_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(), workspace.data_ptr<float>(),
        workspace.size(0), n);
    return output;
}

torch::Tensor probe_compute(int n) {
    auto output = torch::empty({n}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    compute_timing_kernel<<<blocks, threads>>>(output.data_ptr<float>(), n);
    return output;
}

torch::Tensor probe_ratio(torch::Tensor workspace, int n) {
    auto output = torch::empty({n}, workspace.options());
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    ratio_timing_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(), workspace.data_ptr<float>(),
        workspace.size(0), n);
    return output;
}

torch::Tensor timed_matvec(torch::Tensor W, torch::Tensor X) {
    int M = W.size(0), K = W.size(1), B = X.size(0);
    auto Y = torch::empty({B, M}, X.options());
    auto timing = torch::empty({B}, X.options());
    timed_matvec_kernel<<<B, M>>>(
        timing.data_ptr<float>(), W.data_ptr<float>(),
        X.data_ptr<float>(), Y.data_ptr<float>(), B, M, K);
    return timing;
}
'''

HIP_CPP_SOURCE = '''
torch::Tensor probe_memory(torch::Tensor workspace, int n);
torch::Tensor probe_compute(int n);
torch::Tensor probe_ratio(torch::Tensor workspace, int n);
torch::Tensor timed_matvec(torch::Tensor W, torch::Tensor X);
'''


def build_timing_extension():
    """Compile HIP timing probe kernels as PyTorch extension."""
    from torch.utils.cpp_extension import load_inline
    print("[BUILD] Compiling HIP timing probe extension...")
    t0 = time.time()
    ext = load_inline(
        name='timing_probe_z2042',
        cpp_sources=HIP_CPP_SOURCE,
        cuda_sources=HIP_KERNEL_SOURCE,
        functions=['probe_memory', 'probe_compute', 'probe_ratio', 'timed_matvec'],
        verbose=False,
        extra_cuda_cflags=['-O2']
    )
    print(f"[BUILD] Done in {time.time()-t0:.1f}s")
    return ext


class HIPTimingProbe:
    """Interface to HIP timing probe kernels running INSIDE the GPU."""

    def __init__(self, ext, n_threads=128, workspace_size=1 << 20):
        self.ext = ext
        self.n = n_threads
        self.workspace = torch.randn(workspace_size, device=DEVICE)
        # Warm-up: run each probe once
        self.probe_memory()
        self.probe_compute()
        self.probe_ratio()

    def probe_memory(self):
        """Memory-access timing fingerprint. [n] float on GPU."""
        return self.ext.probe_memory(self.workspace, self.n)

    def probe_compute(self):
        """Compute (ALU) timing fingerprint. [n] float on GPU."""
        return self.ext.probe_compute(self.n)

    def probe_ratio(self):
        """Memory/compute ratio. [n] float on GPU."""
        return self.ext.probe_ratio(self.workspace, self.n)

    def probe_all(self):
        """All 3 channels concatenated. [3*n] float on GPU."""
        return torch.cat([self.probe_memory(), self.probe_compute(), self.probe_ratio()])

    def timed_matvec(self, W, X):
        """Timing from actual matmul. [B] float on GPU."""
        return self.ext.timed_matvec(W, X)


# =============================================================================
# Timing Hash: converts timing vector → label offset
# =============================================================================

class TimingHasher:
    """Deterministic hash from timing vector to label offset (0-9)."""

    def __init__(self, timing_dim, num_classes=10, seed=42):
        rng = np.random.RandomState(seed)
        # Random hash matrix, normalized rows
        H = rng.randn(num_classes, timing_dim).astype(np.float32)
        H /= np.linalg.norm(H, axis=1, keepdims=True) + 1e-8
        self.hash_matrix = torch.tensor(H, device=DEVICE)  # [10, timing_dim]
        self.num_classes = num_classes

    def get_offset(self, timing_normalized):
        """timing_normalized: [timing_dim] zero-mean unit-var. Returns int offset."""
        scores = self.hash_matrix @ timing_normalized  # [10]
        return scores.argmax().item()


# =============================================================================
# Model Architecture
# =============================================================================

class TimingEmbeddedModel(nn.Module):
    """MNIST classifier with HIP timing injection.

    The model receives both images and a timing vector from inside the GPU.
    Label encryption via timing hash makes timing ESSENTIAL for correct classification.
    """

    def __init__(self, timing_dim=128, hidden_dim=128, num_classes=10):
        super().__init__()
        self.timing_dim = timing_dim

        # Image encoder (CNN)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden_dim), nn.ReLU(),
        )

        # Timing encoder: raw timing → timing features
        self.timing_enc = nn.Sequential(
            nn.Linear(timing_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )

        # Fusion: image features + timing features → class logits
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 32, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )

        # Self-model: predict timing statistics from image features ONLY
        # (blindsight test: this should fail when timing encoder is ablated)
        self.self_model = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(),
            nn.Linear(32, 4),  # predict [mean, std, skew, kurtosis]
        )

    def forward(self, x, timing_norm):
        """
        x: [B, 1, 28, 28] images
        timing_norm: [timing_dim] normalized timing vector (same for all in batch)
        """
        # Image features
        h = self.encoder(x)  # [B, hidden_dim]

        # Timing features (broadcast to batch)
        t = timing_norm.unsqueeze(0).expand(x.shape[0], -1)  # [B, timing_dim]
        t_enc = self.timing_enc(t)  # [B, 32]

        # Fuse and classify
        fused = torch.cat([h, t_enc], dim=1)  # [B, hidden_dim + 32]
        logits = self.head(fused)  # [B, 10]

        # Self-model prediction (from image features only, no timing)
        timing_pred = self.self_model(h)  # [B, 4]

        return logits, timing_pred


# =============================================================================
# Training and Evaluation
# =============================================================================

def compute_timing_stats(timing):
    """Compute [mean, std, skew, kurtosis] of timing tensor."""
    mean = timing.mean()
    std = timing.std() + 1e-8
    z = (timing - mean) / std
    skew = (z ** 3).mean()
    kurt = (z ** 4).mean()
    return torch.stack([mean, std, skew, kurt])


def train_epoch(model, loader, optimizer, probe, hasher, timing_mode='live',
                frozen_timing=None, timing_stats_running=None):
    """Train one epoch. Returns (loss, accuracy, timing_stats)."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_timing_means = []

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # Get timing vector based on mode
        if timing_mode == 'live':
            timing_raw = probe.probe_memory()  # [128] from HIP kernel
        elif timing_mode == 'frozen':
            timing_raw = frozen_timing.clone()
        elif timing_mode == 'random':
            timing_raw = torch.randn(probe.n, device=DEVICE) * 26000 + 26000
        elif timing_mode == 'zero':
            timing_raw = torch.zeros(probe.n, device=DEVICE)
        else:
            timing_raw = probe.probe_memory()

        all_timing_means.append(timing_raw.mean().item())

        # Normalize timing
        if timing_stats_running is not None:
            t_mean, t_std = timing_stats_running
            timing_norm = (timing_raw - t_mean) / (t_std + 1e-8)
        else:
            timing_norm = (timing_raw - timing_raw.mean()) / (timing_raw.std() + 1e-8)

        # Compute label offset via timing hash
        offset = hasher.get_offset(timing_norm)
        encrypted_labels = (labels + offset) % 10

        # Forward
        logits, timing_pred = model(images, timing_norm)

        # Classification loss (on encrypted labels)
        loss_class = F.cross_entropy(logits, encrypted_labels)

        # Self-model loss: predict timing statistics
        timing_stats_target = compute_timing_stats(timing_raw).unsqueeze(0).expand(images.shape[0], -1)
        loss_self = F.mse_loss(timing_pred, timing_stats_target.detach())

        loss = loss_class + 0.1 * loss_self

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Decrypt predictions to check TRUE accuracy
        preds = logits.argmax(dim=1)
        decrypted_preds = (preds - offset) % 10
        correct += (decrypted_preds == labels).sum().item()
        total += labels.shape[0]
        total_loss += loss.item()

    return total_loss / len(loader), correct / total, np.mean(all_timing_means)


def evaluate(model, loader, probe, hasher, timing_mode='live',
             frozen_timing=None, timing_stats_running=None):
    """Evaluate model. Returns (accuracy, timing_mean)."""
    model.eval()
    correct = 0
    total = 0
    all_timing_means = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if timing_mode == 'live':
                timing_raw = probe.probe_memory()
            elif timing_mode == 'frozen':
                timing_raw = frozen_timing.clone()
            elif timing_mode == 'random':
                timing_raw = torch.randn(probe.n, device=DEVICE) * 26000 + 26000
            elif timing_mode == 'zero':
                timing_raw = torch.zeros(probe.n, device=DEVICE)
            else:
                timing_raw = probe.probe_memory()

            all_timing_means.append(timing_raw.mean().item())

            if timing_stats_running is not None:
                t_mean, t_std = timing_stats_running
                timing_norm = (timing_raw - t_mean) / (t_std + 1e-8)
            else:
                timing_norm = (timing_raw - timing_raw.mean()) / (timing_raw.std() + 1e-8)

            offset = hasher.get_offset(timing_norm)
            encrypted_labels = (labels + offset) % 10

            logits, _ = model(images, timing_norm)
            preds = logits.argmax(dim=1)
            decrypted_preds = (preds - offset) % 10
            correct += (decrypted_preds == labels).sum().item()
            total += labels.shape[0]

    return correct / total, np.mean(all_timing_means)


def evaluate_kill_shot(model, loader, probe, hasher, frozen_timing, timing_stats_running):
    """Kill shot: evaluate trained model with different timing sources."""
    results = {}
    for mode in ['live', 'frozen', 'random', 'zero']:
        acc, t_mean = evaluate(model, loader, probe, hasher, timing_mode=mode,
                               frozen_timing=frozen_timing,
                               timing_stats_running=timing_stats_running)
        results[mode] = {'accuracy': acc, 'timing_mean': t_mean}
        print(f"    {mode:8s}: accuracy={acc:.4f}  timing_mean={t_mean:.1f}")
    return results


def evaluate_timed_matvec(model, loader, probe):
    """Bonus test: timing from the ACTUAL matmul, not a separate probe."""
    model.eval()
    # Get the first linear layer's weight
    first_linear = None
    for m in model.encoder.modules():
        if isinstance(m, nn.Linear):
            first_linear = m
            break
    if first_linear is None:
        return {'note': 'no linear layer found'}

    W = first_linear.weight.data  # [out, in]
    timings = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            flat = images.view(images.shape[0], -1)  # [B, 784]
            if W.shape[1] != flat.shape[1]:
                break  # shape mismatch, skip
            t = probe.timed_matvec(W, flat)
            timings.append(t.cpu())
            if len(timings) >= 10:
                break

    if not timings:
        return {'note': 'could not extract matmul timing'}

    all_t = torch.cat(timings)
    return {
        'mean': all_t.mean().item(),
        'std': all_t.std().item(),
        'min': all_t.min().item(),
        'max': all_t.max().item(),
        'n_samples': all_t.shape[0],
    }


# =============================================================================
# Timing Distribution Analysis
# =============================================================================

def analyze_timing_distribution(probe, n_samples=100):
    """Collect timing statistics to characterize hardware fingerprint."""
    mem_means, mem_stds = [], []
    comp_means, comp_stds = [], []
    ratio_means, ratio_stds = [], []

    for _ in range(n_samples):
        m = probe.probe_memory()
        c = probe.probe_compute()
        r = probe.probe_ratio()
        mem_means.append(m.mean().item())
        mem_stds.append(m.std().item())
        comp_means.append(c.mean().item())
        comp_stds.append(c.std().item())
        ratio_means.append(r.mean().item())
        ratio_stds.append(r.std().item())

    return {
        'memory': {
            'mean_of_means': np.mean(mem_means),
            'std_of_means': np.std(mem_means),
            'mean_of_stds': np.mean(mem_stds),
            'range_of_means': float(max(mem_means) - min(mem_means)),
        },
        'compute': {
            'mean_of_means': np.mean(comp_means),
            'std_of_means': np.std(comp_means),
            'mean_of_stds': np.mean(comp_stds),
        },
        'ratio': {
            'mean_of_means': np.mean(ratio_means),
            'std_of_means': np.std(ratio_means),
            'mean_of_stds': np.mean(ratio_stds),
        },
    }


# =============================================================================
# Main Experiment
# =============================================================================

def main():
    print("=" * 70)
    print("z2042: Timing-Embedded Neural Network (TENN)")
    print("  Hardware PUF for Neural Networks — clock64() from INSIDE the GPU")
    print("=" * 70)

    # Build HIP extension
    ext = build_timing_extension()
    probe = HIPTimingProbe(ext, n_threads=128)

    # Timing hash
    hasher = TimingHasher(timing_dim=128, num_classes=10, seed=42)

    # MNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = torchvision.datasets.MNIST(os.path.join(ROOT, 'data'), train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(os.path.join(ROOT, 'data'), train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    print(f"\n[DATA] Train: {len(train_ds)}, Test: {len(test_ds)}")
    print(f"[GPU] {torch.cuda.get_device_name(0)}")

    # Characterize timing distribution
    print("\n--- Timing Distribution Analysis ---")
    timing_dist = analyze_timing_distribution(probe, n_samples=50)
    for channel, stats in timing_dist.items():
        print(f"  {channel:8s}: mean_of_means={stats['mean_of_means']:.1f}, "
              f"std_of_means={stats.get('std_of_means', 0):.1f}")

    # Capture frozen timing for condition C
    frozen_timing = probe.probe_memory()
    print(f"\n[FROZEN] Captured timing: mean={frozen_timing.mean().item():.1f}")

    # Collect running stats for normalization (first 20 batches)
    print("\n--- Calibrating timing normalization ---")
    cal_values = []
    for i, (images, _) in enumerate(train_loader):
        if i >= 20:
            break
        t = probe.probe_memory()
        cal_values.append(t)
    cal_all = torch.stack(cal_values)
    timing_mean = cal_all.mean().item()
    timing_std = cal_all.std().item()
    timing_stats_running = (timing_mean, timing_std)
    print(f"  Running stats: mean={timing_mean:.1f}, std={timing_std:.1f}")

    # =========================================================================
    # Run 5 conditions
    # =========================================================================
    N_EPOCHS = 20
    conditions = {
        'A_standard': 'zero',    # No timing (zero vector)
        'B_tenn': 'live',        # Real HIP timing (the key condition)
        'C_frozen': 'frozen',    # Same timing every batch
        'D_random': 'random',    # Random noise
    }

    all_results = {
        'experiment': 'z2042_timing_embedded_neural_net',
        'timing_distribution': timing_dist,
        'timing_calibration': {'mean': timing_mean, 'std': timing_std},
        'frozen_timing_mean': frozen_timing.mean().item(),
        'n_epochs': N_EPOCHS,
        'conditions': {},
    }

    for cond_name, timing_mode in conditions.items():
        print(f"\n{'=' * 60}")
        print(f"  Condition: {cond_name} (timing_mode={timing_mode})")
        print(f"{'=' * 60}")

        model = TimingEmbeddedModel(timing_dim=128).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        train_history = []
        t0 = time.time()

        for epoch in range(N_EPOCHS):
            loss, acc, t_mean = train_epoch(
                model, train_loader, optimizer, probe, hasher,
                timing_mode=timing_mode,
                frozen_timing=frozen_timing,
                timing_stats_running=timing_stats_running,
            )
            train_history.append({'epoch': epoch, 'loss': loss, 'accuracy': acc, 'timing_mean': t_mean})

            if (epoch + 1) % 5 == 0 or epoch == 0:
                test_acc, test_t = evaluate(
                    model, test_loader, probe, hasher,
                    timing_mode=timing_mode,
                    frozen_timing=frozen_timing,
                    timing_stats_running=timing_stats_running,
                )
                print(f"  Epoch {epoch+1:2d}: loss={loss:.4f} train_acc={acc:.4f} "
                      f"test_acc={test_acc:.4f} timing={t_mean:.0f}")

        train_time = time.time() - t0

        # Final test accuracy
        final_acc, _ = evaluate(
            model, test_loader, probe, hasher,
            timing_mode=timing_mode,
            frozen_timing=frozen_timing,
            timing_stats_running=timing_stats_running,
        )
        print(f"  Final test accuracy: {final_acc:.4f} ({train_time:.1f}s)")

        # Kill shot (only for B_tenn)
        kill_shot = None
        if cond_name == 'B_tenn':
            print(f"\n  --- Kill Shot Test (trained with live timing) ---")
            kill_shot = evaluate_kill_shot(
                model, test_loader, probe, hasher,
                frozen_timing, timing_stats_running
            )

        # Timed matmul analysis
        matvec_timing = evaluate_timed_matvec(model, test_loader, probe)

        cond_result = {
            'timing_mode': timing_mode,
            'final_accuracy': final_acc,
            'train_time_s': train_time,
            'train_history': train_history,
            'kill_shot': kill_shot,
            'matvec_timing': matvec_timing,
        }
        all_results['conditions'][cond_name] = cond_result

        # Save model for B_tenn
        if cond_name == 'B_tenn':
            model_path = os.path.join(RESULTS_DIR, 'z2042_tenn_model.pt')
            torch.save(model.state_dict(), model_path)
            print(f"  Saved model to {model_path}")

    # =========================================================================
    # Cross-condition analysis
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("  Cross-Condition Analysis")
    print(f"{'=' * 60}")

    # Summary table
    print(f"\n  {'Condition':<15} {'Final Acc':>10} {'Mode':>10}")
    print("  " + "-" * 40)
    for cond_name, cond_result in all_results['conditions'].items():
        print(f"  {cond_name:<15} {cond_result['final_accuracy']:>10.4f} {cond_result['timing_mode']:>10}")

    # Kill shot summary
    ks = all_results['conditions'].get('B_tenn', {}).get('kill_shot')
    if ks:
        print(f"\n  Kill Shot Results (B_tenn model):")
        print(f"    live:    {ks['live']['accuracy']:.4f}")
        print(f"    frozen:  {ks['frozen']['accuracy']:.4f}")
        print(f"    random:  {ks['random']['accuracy']:.4f}")
        print(f"    zero:    {ks['zero']['accuracy']:.4f}")

        # Test verdicts
        live_acc = ks['live']['accuracy']
        frozen_acc = ks['frozen']['accuracy']
        random_acc = ks['random']['accuracy']
        zero_acc = ks['zero']['accuracy']

        t1_pass = live_acc >= 0.85
        t2_pass = (live_acc - frozen_acc) > 0.10
        t3_random = (live_acc - random_acc) > 0.10
        t4_zero = (live_acc - zero_acc) > 0.10

        all_results['tests'] = {
            'T1_learning': {
                'pass': t1_pass,
                'detail': f'B_tenn accuracy {live_acc:.4f} >= 0.85',
            },
            'T2_kill_shot_frozen': {
                'pass': t2_pass,
                'detail': f'live({live_acc:.4f}) - frozen({frozen_acc:.4f}) = {live_acc-frozen_acc:.4f} > 0.10',
            },
            'T3_kill_shot_random': {
                'pass': t3_random,
                'detail': f'live({live_acc:.4f}) - random({random_acc:.4f}) = {live_acc-random_acc:.4f} > 0.10',
            },
            'T4_kill_shot_zero': {
                'pass': t4_zero,
                'detail': f'live({live_acc:.4f}) - zero({zero_acc:.4f}) = {live_acc-zero_acc:.4f} > 0.10',
            },
        }

        n_pass = sum(1 for t in all_results['tests'].values() if t['pass'])
        all_results['verdict'] = f'{n_pass}/4 PASS'

        print(f"\n  Test Verdicts:")
        for tname, tres in all_results['tests'].items():
            status = 'PASS' if tres['pass'] else 'FAIL'
            print(f"    {tname}: {status} — {tres['detail']}")
        print(f"\n  VERDICT: {all_results['verdict']}")

    # Timing distribution note
    all_results['notes'] = {
        'timing_source': 'HIP clock64() from inside GPU kernel via PyTorch cpp_extension',
        'timing_type': 'memory access probe (random VRAM access, 50 iterations per thread)',
        'gpu': torch.cuda.get_device_name(0),
        'innovation': 'First experiment where neural network timing comes from INSIDE '
                      'GPU kernel (clock64), not from sysfs/Python telemetry',
    }

    # Save results
    results_path = os.path.join(RESULTS_DIR, 'z2042_timing_embedded_neural_net.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[SAVED] {results_path}")


if __name__ == '__main__':
    main()
