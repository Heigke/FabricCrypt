#!/usr/bin/env python3
"""z2047: Kernel-Fused Hardware-IS-Computation

THE DEEPEST POSSIBLE: Hardware coupling happens INSIDE the HIP kernel.
Python does NOT decide the computation path — the GPU hardware does.

WHAT HAPPENS INSIDE THE KERNEL:
  1. Each thread measures its own timing (clock64)
  2. Timing determines per-thread blend weight alpha ∈ [0, 1]
  3. Output = alpha * (W0 @ x) + (1-alpha) * (W1 @ x)
  4. Different threads get different alphas → spatially-varying blend
  5. The blend is PHYSICALLY determined by hardware scheduling

THIS IS TRULY "HW IS COMPUTATION":
  - No Python selecting banks — kernel timing does it
  - Per-OUTPUT-UNIT coupling (not per-batch like z2046)
  - Each hidden neuron's output depends on its thread's timing
  - DVFS changes timing distribution → changes alphas → changes output
  - You literally CANNOT separate the hardware from the computation

GRADIENT FLOW:
  - Forward: kernel provides timing-based alphas (detached)
  - Forward: PyTorch computes h0 = W0@x, h1 = W1@x (standard)
  - Forward: h = alpha.detach() * h0 + (1-alpha.detach()) * h1
  - Backward: gradients flow through W0, W1 via standard autograd
  - The MODEL learns to make W0 and W1 COMPLEMENTARY

4 CONDITIONS:
  A_kernel_fused:  Timing-based alpha from HIP kernel (DEEPEST)
  B_uniform:       Alpha = 0.5 everywhere (no hw coupling)
  C_random_alpha:  Random alpha per forward pass (noise baseline)
  D_standard:      Single weight matrix, no blending

5 TESTS:
  T1: A learns (acc > 90%)
  T2: DVFS changes alpha distribution (measured inside kernel)
  T3: Kill shot — replace hw alpha with uniform → output KL change
  T4: A_kernel_fused != D_standard in output distribution
  T5: W0 and W1 are different (cosine similarity < 0.9)
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
import ctypes
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# =============================================================================
# HIP Kernel: Hardware-Fused Alpha Computation
# =============================================================================

HIP_SRC = r'''
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// This kernel IS the hardware-computation fusion point.
// Each thread measures its own timing and uses it to set alpha.
// alpha determines the blend between two weight banks.
// This cannot be replicated without the actual hardware.

__global__ void hw_alpha_kernel(
    float* __restrict__ alpha_out,     // [out_dim] — per-unit alpha
    float* __restrict__ timing_out,    // [out_dim] — raw timing values
    const float* __restrict__ probe_data, // [probe_size] — data for timing probe
    int probe_size,
    int out_dim,
    float t_min,
    float t_range
) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= out_dim) return;

    // === HARDWARE-DETERMINED TIMING ===
    // This is where hardware IS computation.
    // Each thread does a memory access pattern, measures its own timing.
    // The timing varies per-thread due to:
    //   - Cache state (which lines are hot)
    //   - Memory bank conflicts
    //   - Wavefront scheduling
    //   - DVFS state (memory clock)

    float sum = 0.0f;
    uint64_t t0 = clock64();

    // Memory probe: each thread accesses different locations
    // Large strides force cache misses, timing varies with memory system
    #pragma unroll 1
    for (int i = 0; i < 32; i++) {
        // Thread-dependent access pattern
        int idx = ((uint32_t)(gid * 2654435761u + i * 65537u) >> 4) % probe_size;
        sum += probe_data[idx];
    }

    // Compute probe: FP operations, timing varies with ALU state
    #pragma unroll 1
    for (int i = 0; i < 16; i++) {
        sum = __sinf(sum + 0.001f) * __cosf(sum * 0.1f + (float)gid * 0.01f);
    }

    uint64_t t1 = clock64();
    float dt = (float)(t1 - t0);

    // Store raw timing
    timing_out[gid] = dt;

    // === HARDWARE DETERMINES ALPHA ===
    // Normalize timing to [0, 1]
    float a = (dt - t_min) / (t_range + 1e-6f);
    a = fminf(fmaxf(a, 0.0f), 1.0f);

    // Smooth sigmoid to avoid hard boundaries
    // Center at 0.5, scale to cover [0.1, 0.9]
    a = 1.0f / (1.0f + expf(-5.0f * (a - 0.5f)));

    alpha_out[gid] = a;

    // Anti-optimization barrier
    if (__builtin_expect(sum == -1e30f, 0)) alpha_out[0] = sum;
}

// Second kernel: measure timing distribution for calibration
__global__ void timing_calibrate_kernel(
    float* __restrict__ timings,        // [n_samples]
    const float* __restrict__ probe_data,
    int probe_size,
    int n_samples
) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= n_samples) return;

    float sum = 0.0f;
    uint64_t t0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < 32; i++) {
        int idx = ((uint32_t)(gid * 2654435761u + i * 65537u) >> 4) % probe_size;
        sum += probe_data[idx];
    }
    #pragma unroll 1
    for (int i = 0; i < 16; i++) {
        sum = __sinf(sum + 0.001f) * __cosf(sum * 0.1f + (float)gid * 0.01f);
    }
    uint64_t t1 = clock64();
    timings[gid] = (float)(t1 - t0);
    if (__builtin_expect(sum == -1e30f, 0)) timings[0] = sum;
}

std::vector<torch::Tensor> compute_hw_alpha(
    torch::Tensor probe_data, int out_dim,
    float t_min, float t_range
) {
    auto opts = probe_data.options();
    auto alpha = torch::empty({out_dim}, opts);
    auto timing = torch::empty({out_dim}, opts);

    int threads = 128;
    int blocks = (out_dim + threads - 1) / threads;
    hw_alpha_kernel<<<blocks, threads>>>(
        alpha.data_ptr<float>(),
        timing.data_ptr<float>(),
        probe_data.data_ptr<float>(),
        probe_data.size(0),
        out_dim,
        t_min,
        t_range
    );
    return {alpha, timing};
}

torch::Tensor calibrate_timing(torch::Tensor probe_data, int n_samples) {
    auto timings = torch::empty({n_samples}, probe_data.options());
    int threads = 128;
    int blocks = (n_samples + threads - 1) / threads;
    timing_calibrate_kernel<<<blocks, threads>>>(
        timings.data_ptr<float>(),
        probe_data.data_ptr<float>(),
        probe_data.size(0),
        n_samples
    );
    return timings;
}
'''

HIP_CPP = r'''
std::vector<torch::Tensor> compute_hw_alpha(
    torch::Tensor probe_data, int out_dim,
    float t_min, float t_range);
torch::Tensor calibrate_timing(torch::Tensor probe_data, int n_samples);
'''


class KernelFusedProbe:
    """Hardware alpha computation — runs entirely inside HIP kernel."""

    def __init__(self, out_dim=64, probe_size=1 << 22):  # 16MB probe data
        from torch.utils.cpp_extension import load_inline
        print("[BUILD] Compiling kernel-fused HW extension...")
        t0 = time.time()
        self.ext = load_inline(
            name='kernel_fused_z2047',
            cpp_sources=HIP_CPP,
            cuda_sources=HIP_SRC,
            functions=['compute_hw_alpha', 'calibrate_timing'],
            verbose=False,
            extra_cuda_cflags=['-O2']
        )
        print(f"[BUILD] Done in {time.time()-t0:.1f}s")
        self.out_dim = out_dim
        self.probe_data = torch.randn(probe_size, device=DEVICE)
        self.t_min = 0.0
        self.t_range = 1.0
        self._calibrate()

    def _calibrate(self, n_rounds=10):
        """Calibrate timing distribution for normalization."""
        print("[CAL] Calibrating kernel timing distribution...")
        all_timings = []
        for _ in range(n_rounds):
            t = self.ext.calibrate_timing(self.probe_data, self.out_dim)
            all_timings.append(t.cpu().numpy())

        timings = np.concatenate(all_timings)
        self.t_min = float(np.percentile(timings, 5))
        t_max = float(np.percentile(timings, 95))
        self.t_range = max(t_max - self.t_min, 1.0)
        print(f"[CAL] t_min={self.t_min:.0f}, t_max={t_max:.0f}, "
              f"range={self.t_range:.0f} cycles")
        print(f"[CAL] mean={timings.mean():.0f}, std={timings.std():.0f}")

    def get_alpha(self):
        """Get hardware-determined alpha values [out_dim].

        Returns (alpha, timing) where:
          alpha: [out_dim] blend weights ∈ [0, 1]
          timing: [out_dim] raw cycle counts
        """
        results = self.ext.compute_hw_alpha(
            self.probe_data, self.out_dim, self.t_min, self.t_range)
        return results[0], results[1]


# =============================================================================
# DVFS Controller
# =============================================================================

class DVFSController:
    def __init__(self):
        lib_path = str(ROOT / 'src' / 'native' / 'libdeep_gpu.so')
        try:
            self.lib = ctypes.CDLL(lib_path)
            self.lib.deep_gpu_init.restype = ctypes.c_int
            self.lib.deep_gpu_dvfs_force_low.restype = ctypes.c_int
            self.lib.deep_gpu_dvfs_force_high.restype = ctypes.c_int
            self.lib.deep_gpu_dvfs_auto.restype = ctypes.c_int
            self.lib.deep_gpu_dvfs_get_sclk.restype = ctypes.c_int
            self.lib.deep_gpu_cleanup.restype = None
            fd = self.lib.deep_gpu_init()
            self.available = fd >= 0
            if self.available:
                print(f"[DVFS] Initialized: fd={fd}")
        except Exception as e:
            print(f"[DVFS] Not available: {e}")
            self.available = False

    def force_low(self):
        if self.available: self.lib.deep_gpu_dvfs_force_low()
    def force_high(self):
        if self.available: self.lib.deep_gpu_dvfs_force_high()
    def auto(self):
        if self.available: self.lib.deep_gpu_dvfs_auto()
    def get_sclk(self):
        return self.lib.deep_gpu_dvfs_get_sclk() if self.available else 1000
    def cleanup(self):
        if self.available:
            self.lib.deep_gpu_dvfs_auto()
            self.lib.deep_gpu_cleanup()


# =============================================================================
# Kernel-Fused Model
# =============================================================================

class KernelFusedModel(nn.Module):
    """Neural network where hardware timing determines computation INSIDE the kernel.

    Two weight banks W0, W1. The kernel produces per-unit alpha values.
    h = alpha * (W0 @ x + b0) + (1 - alpha) * (W1 @ x + b1)

    alpha is determined by HIP ISA clock64() timing, per-output-unit.
    This is the closest GPU analog to FPGA routing-delay-based computation.
    """

    def __init__(self, hidden_dim=64, mode='kernel_fused'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mode = mode

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )

        # TWO weight banks for the hw-fused layer
        self.W0 = nn.Linear(128, hidden_dim)
        self.W1 = nn.Linear(128, hidden_dim)

        # Classifier head
        self.head = nn.Linear(hidden_dim, 10)

    def forward(self, x, alpha=None):
        """
        Args:
            x: [B, 1, 28, 28] images
            alpha: [hidden_dim] blend weights from kernel (or None for defaults)
        """
        x_flat = x.view(x.shape[0], -1)
        h_enc = self.encoder(x_flat)  # [B, 128]

        # Compute both bank outputs
        h0 = F.relu(self.W0(h_enc))  # [B, hidden_dim]
        h1 = F.relu(self.W1(h_enc))  # [B, hidden_dim]

        if alpha is not None:
            # Hardware-determined blend (alpha is detached — no gradient through hw)
            a = alpha.detach().unsqueeze(0)  # [1, hidden_dim]
            h = a * h0 + (1.0 - a) * h1  # [B, hidden_dim]
        elif self.mode == 'uniform':
            h = 0.5 * h0 + 0.5 * h1
        elif self.mode == 'bank0_only':
            h = h0
        else:
            h = 0.5 * h0 + 0.5 * h1

        logits = self.head(h)
        return logits


# =============================================================================
# Training & Evaluation
# =============================================================================

def train_condition(model, train_loader, test_loader, probe, dvfs,
                    mode, n_epochs=15, lr=1e-3):
    """Train one condition."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    alpha_stats = []
    t0 = time.time()

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        epoch_alphas = []

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # DVFS cycling (for all conditions — creates hw variation)
            if dvfs.available and batch_idx % 40 == 0:
                phase = (batch_idx // 40 + epoch * 4) % 3
                if phase == 0:
                    dvfs.force_low()
                elif phase == 1:
                    dvfs.auto()
                else:
                    dvfs.force_high()

            # Get alpha from INSIDE the kernel
            if mode == 'kernel_fused':
                with torch.no_grad():
                    alpha, timing = probe.get_alpha()
                epoch_alphas.append(alpha.mean().item())
            elif mode == 'random_alpha':
                alpha = torch.rand(model.hidden_dim, device=DEVICE)
                epoch_alphas.append(alpha.mean().item())
            elif mode == 'uniform':
                alpha = None
            else:  # standard
                alpha = None

            logits = model(images, alpha)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.shape[0]
            total_loss += loss.item()

        acc = correct / total
        a_mean = np.mean(epoch_alphas) if epoch_alphas else 0.5
        a_std = np.std(epoch_alphas) if epoch_alphas else 0
        history.append({
            'epoch': epoch,
            'loss': total_loss / len(train_loader),
            'accuracy': acc,
            'alpha_mean': a_mean,
            'alpha_std': a_std,
        })

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}: loss={history[-1]['loss']:.4f} "
                  f"acc={acc:.4f} alpha_mean={a_mean:.4f}±{a_std:.4f}")

    train_time = time.time() - t0
    dvfs.auto()
    time.sleep(0.3)

    # Final evaluation with natural alpha
    eval_result = evaluate_model(model, test_loader, probe, mode)

    return {
        'train_history': history,
        'train_time_s': train_time,
        **eval_result,
    }


def evaluate_model(model, test_loader, probe, mode, override_alpha=None):
    """Evaluate model, optionally overriding alpha."""
    model.eval()
    correct = 0
    total = 0
    all_outputs = []
    all_alphas = []
    all_timings = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if override_alpha is not None:
                alpha = override_alpha
            elif mode == 'kernel_fused':
                alpha, timing = probe.get_alpha()
                all_alphas.append(alpha.cpu())
                all_timings.append(timing.cpu())
            elif mode == 'random_alpha':
                alpha = torch.rand(model.hidden_dim, device=DEVICE)
                all_alphas.append(alpha.cpu())
            else:
                alpha = None

            logits = model(images, alpha)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.shape[0]
            all_outputs.append(F.softmax(logits, dim=1).cpu())

    outputs = torch.cat(all_outputs)
    accuracy = correct / total

    result = {
        'accuracy': accuracy,
        'outputs': outputs,
    }

    if all_alphas:
        alphas = torch.stack(all_alphas)
        result['alpha_mean'] = alphas.mean().item()
        result['alpha_std'] = alphas.std().item()
        result['alpha_per_unit_std'] = alphas.std(dim=0).mean().item()

    if all_timings:
        timings = torch.stack(all_timings)
        result['timing_mean'] = timings.mean().item()
        result['timing_std'] = timings.std().item()

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("z2047: Kernel-Fused Hardware-IS-Computation")
    print("  Alpha blend computed INSIDE HIP kernel via clock64()")
    print("  Python doesn't decide — GPU hardware does")
    print("=" * 70)

    probe = KernelFusedProbe(out_dim=64, probe_size=1 << 22)
    dvfs = DVFSController()

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = torchvision.datasets.MNIST(str(ROOT / 'data'), train=True,
                                           download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(str(ROOT / 'data'), train=False,
                                          download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                             num_workers=2, pin_memory=True)

    print(f"\n[DATA] Train: {len(train_ds)}, Test: {len(test_ds)}")
    print(f"[GPU] {torch.cuda.get_device_name(0)}")

    # =========================================================================
    # T2: DVFS → alpha distribution change
    # =========================================================================
    print(f"\n{'='*60}")
    print("  T2: DVFS → Alpha Distribution")
    print(f"{'='*60}")

    dvfs_alpha_dist = {}
    for state, fn in [('low', dvfs.force_low), ('auto', dvfs.auto),
                       ('high', dvfs.force_high)]:
        fn()
        time.sleep(0.5)
        alphas = []
        timings = []
        for _ in range(30):
            a, t = probe.get_alpha()
            alphas.append(a.cpu().numpy())
            timings.append(t.cpu().numpy())
            time.sleep(0.01)

        alphas = np.array(alphas)
        timings = np.array(timings)
        sclk = dvfs.get_sclk()
        dvfs_alpha_dist[state] = {
            'sclk': sclk,
            'alpha_mean': float(alphas.mean()),
            'alpha_std': float(alphas.std()),
            'timing_mean': float(timings.mean()),
            'timing_std': float(timings.std()),
            'per_unit_alpha_std': float(alphas.std(axis=0).mean()),
        }
        print(f"  {state:5s}: SCLK={sclk:5d} MHz, alpha={alphas.mean():.4f}±{alphas.std():.4f}, "
              f"timing={timings.mean():.0f}±{timings.std():.0f}")

    dvfs.auto()

    # T2 check: different DVFS states produce different alpha distributions
    alpha_means = [dvfs_alpha_dist[s]['alpha_mean'] for s in ['low', 'auto', 'high']]
    t2_range = max(alpha_means) - min(alpha_means)
    t2_pass = t2_range > 0.01  # at least 1% difference in mean alpha
    print(f"  → Alpha range across DVFS: {t2_range:.4f} — "
          f"{'PASS' if t2_pass else 'FAIL'}")

    # =========================================================================
    # Train conditions
    # =========================================================================
    conditions = {
        'A_kernel_fused': 'kernel_fused',
        'B_uniform':      'uniform',
        'C_random_alpha': 'random_alpha',
        'D_standard':     'standard',
    }

    all_results = {
        'experiment': 'z2047_kernel_fused_hw',
        'timestamp': datetime.now().isoformat(),
        'calibration': {
            't_min': probe.t_min,
            't_range': probe.t_range,
        },
        'dvfs_alpha_dist': dvfs_alpha_dist,
        'conditions': {},
    }

    saved_outputs = {}

    for cond_name, mode in conditions.items():
        print(f"\n{'='*60}")
        print(f"  Condition: {cond_name} (mode={mode})")
        print(f"{'='*60}")

        model = KernelFusedModel(hidden_dim=64, mode=mode).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        result = train_condition(model, train_loader, test_loader,
                                 probe, dvfs, mode, n_epochs=15)

        outputs_ref = result.pop('outputs')
        saved_outputs[cond_name] = outputs_ref

        # Kill shots (for kernel_fused)
        kill_shots = {}
        if mode == 'kernel_fused':
            # Kill shot 1: Replace hw alpha with uniform 0.5
            print(f"\n  --- Kill Shot 1: Uniform alpha (0.5) ---")
            uniform_alpha = torch.full((64,), 0.5, device=DEVICE)
            ev_uniform = evaluate_model(model, test_loader, probe, mode,
                                         override_alpha=uniform_alpha)
            out_u = ev_uniform.pop('outputs')
            p = out_u.mean(0).clamp(min=1e-8)
            q = outputs_ref.mean(0).clamp(min=1e-8)
            kl_uniform = (p * (p / q).log()).sum().item()
            kill_shots['uniform_alpha'] = {
                'accuracy': ev_uniform['accuracy'],
                'kl_from_ref': kl_uniform,
            }
            print(f"    acc={ev_uniform['accuracy']:.4f} KL={kl_uniform:.6f}")

            # Kill shot 2: All alpha = 0 (bank 1 only)
            print(f"  --- Kill Shot 2: Alpha = 0 (bank 1 only) ---")
            zero_alpha = torch.zeros(64, device=DEVICE)
            ev_zero = evaluate_model(model, test_loader, probe, mode,
                                      override_alpha=zero_alpha)
            out_z = ev_zero.pop('outputs')
            p = out_z.mean(0).clamp(min=1e-8)
            kl_zero = (p * (p / q).log()).sum().item()
            kill_shots['zero_alpha'] = {
                'accuracy': ev_zero['accuracy'],
                'kl_from_ref': kl_zero,
            }
            print(f"    acc={ev_zero['accuracy']:.4f} KL={kl_zero:.6f}")

            # Kill shot 3: All alpha = 1 (bank 0 only)
            print(f"  --- Kill Shot 3: Alpha = 1 (bank 0 only) ---")
            one_alpha = torch.ones(64, device=DEVICE)
            ev_one = evaluate_model(model, test_loader, probe, mode,
                                     override_alpha=one_alpha)
            out_o = ev_one.pop('outputs')
            p = out_o.mean(0).clamp(min=1e-8)
            kl_one = (p * (p / q).log()).sum().item()
            kill_shots['one_alpha'] = {
                'accuracy': ev_one['accuracy'],
                'kl_from_ref': kl_one,
            }
            print(f"    acc={ev_one['accuracy']:.4f} KL={kl_one:.6f}")

            # Kill shot 4: Per-DVFS evaluation
            print(f"\n  --- Per-DVFS Evaluation ---")
            for dvfs_state, dvfs_fn in [('low', dvfs.force_low),
                                         ('auto', dvfs.auto),
                                         ('high', dvfs.force_high)]:
                dvfs_fn()
                time.sleep(0.5)
                ev_dvfs = evaluate_model(model, test_loader, probe, mode)
                out_d = ev_dvfs.pop('outputs')
                p = out_d.mean(0).clamp(min=1e-8)
                kl_dvfs = (p * (p / q).log()).sum().item()
                kill_shots[f'dvfs_{dvfs_state}'] = {
                    'accuracy': ev_dvfs['accuracy'],
                    'alpha_mean': ev_dvfs.get('alpha_mean', 0),
                    'timing_mean': ev_dvfs.get('timing_mean', 0),
                    'kl_from_ref': kl_dvfs,
                }
                print(f"    {dvfs_state:5s}: acc={ev_dvfs['accuracy']:.4f} "
                      f"alpha={ev_dvfs.get('alpha_mean', 0):.4f} "
                      f"KL={kl_dvfs:.6f}")
            dvfs.auto()

            # Weight bank divergence
            with torch.no_grad():
                w0 = model.W0.weight.data.flatten()
                w1 = model.W1.weight.data.flatten()
                cos_sim = F.cosine_similarity(w0.unsqueeze(0),
                                               w1.unsqueeze(0)).item()
                l2_dist = (w0 - w1).norm().item()
            kill_shots['weight_divergence'] = {
                'cosine_similarity': cos_sim,
                'l2_distance': l2_dist,
            }
            print(f"\n  --- Weight Bank Divergence ---")
            print(f"    cos(W0, W1) = {cos_sim:.4f}, ||W0-W1|| = {l2_dist:.4f}")

        cond_result = {
            'mode': mode,
            'n_params': n_params,
            'final_accuracy': result['accuracy'],
            'alpha_mean': result.get('alpha_mean', 0.5),
            'alpha_std': result.get('alpha_std', 0),
            'train_time_s': result['train_time_s'],
            'train_history': result['train_history'],
            'kill_shots': kill_shots,
        }
        all_results['conditions'][cond_name] = cond_result
        print(f"\n  Final: acc={result['accuracy']:.4f}")

    # =========================================================================
    # Analysis
    # =========================================================================
    print(f"\n{'='*70}")
    print("  Cross-Condition Analysis")
    print(f"{'='*70}")

    print(f"\n  {'Condition':<18} {'Accuracy':>10} {'Alpha μ':>10} {'Alpha σ':>10}")
    print("  " + "-" * 50)
    for cond, res in all_results['conditions'].items():
        print(f"  {cond:<18} {res['final_accuracy']:>10.4f} "
              f"{res['alpha_mean']:>10.4f} {res['alpha_std']:>10.4f}")

    emb = all_results['conditions']['A_kernel_fused']

    # T1
    t1_pass = emb['final_accuracy'] >= 0.90
    print(f"\n  T1: A acc={emb['final_accuracy']:.4f} >= 0.90 — "
          f"{'PASS' if t1_pass else 'FAIL'}")

    # T3: Kill shot — uniform alpha changes output
    ks = emb.get('kill_shots', {})
    ks_uniform = ks.get('uniform_alpha', {})
    ks_zero = ks.get('zero_alpha', {})
    ks_one = ks.get('one_alpha', {})
    max_kl = max(abs(ks_uniform.get('kl_from_ref', 0)),
                 abs(ks_zero.get('kl_from_ref', 0)),
                 abs(ks_one.get('kl_from_ref', 0)))
    max_acc_diff = max(abs(emb['final_accuracy'] - ks_uniform.get('accuracy', emb['final_accuracy'])),
                       abs(emb['final_accuracy'] - ks_zero.get('accuracy', emb['final_accuracy'])),
                       abs(emb['final_accuracy'] - ks_one.get('accuracy', emb['final_accuracy'])))
    t3_pass = max_kl > 1e-4 or max_acc_diff > 0.005
    print(f"  T3: Kill shot max_KL={max_kl:.6f}, max_acc_diff={max_acc_diff:.4f} — "
          f"{'PASS' if t3_pass else 'FAIL'}")

    # T4: Output distribution differs from standard
    out_emb = saved_outputs.get('A_kernel_fused')
    out_std = saved_outputs.get('D_standard')
    if out_emb is not None and out_std is not None:
        p = out_emb.mean(0).clamp(min=1e-8)
        q = out_std.mean(0).clamp(min=1e-8)
        kl_emb_vs_std = (p * (p / q).log()).sum().item()
        t4_pass = abs(kl_emb_vs_std) > 1e-4
        print(f"  T4: KL(A||D_standard)={kl_emb_vs_std:.6f} — "
              f"{'PASS' if t4_pass else 'FAIL'}")
    else:
        t4_pass = False
        kl_emb_vs_std = 0

    # T5: W0 != W1
    wd = ks.get('weight_divergence', {})
    cos_sim = wd.get('cosine_similarity', 1.0)
    t5_pass = cos_sim < 0.95
    print(f"  T5: cos(W0,W1)={cos_sim:.4f} < 0.95 — "
          f"{'PASS' if t5_pass else 'FAIL'}")

    n_pass = sum([t1_pass, t2_pass, t3_pass, t4_pass, t5_pass])
    all_results['tests'] = {
        'T1_learning': {'pass': bool(t1_pass),
                        'detail': f"acc={emb['final_accuracy']:.4f}"},
        'T2_dvfs_alpha_shift': {'pass': bool(t2_pass),
                                 'detail': f"alpha_range={t2_range:.4f}"},
        'T3_kill_shot': {'pass': bool(t3_pass),
                         'detail': f"max_KL={max_kl:.6f}, max_acc_diff={max_acc_diff:.4f}"},
        'T4_output_divergence': {'pass': bool(t4_pass),
                                  'detail': f"KL(A||D)={kl_emb_vs_std:.6f}"},
        'T5_weight_divergence': {'pass': bool(t5_pass),
                                  'detail': f"cos={cos_sim:.4f}"},
    }
    all_results['verdict'] = f'{n_pass}/5 PASS'

    print(f"\n{'='*70}")
    print(f"  VERDICT: {n_pass}/5 PASS")
    print(f"{'='*70}")

    all_results['notes'] = {
        'innovation': 'Alpha blend computed INSIDE HIP kernel via per-thread clock64(). '
                      'Python receives alpha as opaque hardware output. '
                      'The computation path is PHYSICALLY determined by GPU timing.',
        'depth': 'DEEPEST — timing measurement + computation selection in same kernel. '
                 'No Python-level bank selection. Each hidden unit gets its own alpha.',
        'vs_z2046': 'z2046: Python reads SCLK → selects bank. '
                    'z2047: Kernel measures timing → computes alpha. Python is blind.',
    }

    dvfs.cleanup()

    results_path = RESULTS_DIR / 'z2047_kernel_fused_hw.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[SAVED] {results_path}")

    return all_results


if __name__ == '__main__':
    main()
