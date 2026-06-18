#!/usr/bin/env python3
"""z2048: Kernel-Dependent Task — Hardware Makes Classification IMPOSSIBLE Without Timing

THE PROBLEM WITH z2047: MNIST is too easy. Both banks solve it independently.
The hardware coupling is REAL but IRRELEVANT to the task.

THE FIX: Make the task REQUIRE hardware timing information.
Not by encrypting labels (z2042 failed), but by making the DRIFT target
hardware-dependent, computed INSIDE the same HIP kernel.

ARCHITECTURE:
  1. HIP kernel measures timing → computes alpha blend + timing_hash
  2. timing_hash determines a PERMUTATION of class labels
     (e.g., at timing_bin=0, digit 3→label 7; at timing_bin=1, digit 3→label 3)
  3. Model must learn: "what timing bin am I in?" to know the label mapping
  4. Standard model (no timing info) sees random permutations → ~10% accuracy

THIS IS THE GPU's "HARDWARE IS COMPUTATION" PROOF:
  - The correct answer DEPENDS on hardware timing
  - Without timing → can't solve the task
  - Kill shot: remove timing → immediate accuracy collapse
  - This is like z1315 drift prediction but AT THE KERNEL LEVEL

CRITICAL DESIGN: Use only 2 permutations (not 256) to be learnable:
  - Timing high (alpha > 0.5): identity mapping (digit 3 → label 3)
  - Timing low (alpha ≤ 0.5): shifted mapping (digit d → label (d+5)%10)
  The model MUST know its timing to predict the correct permuted label.

4 CONDITIONS:
  A_embodied:   Sees timing alpha → knows which permutation → high accuracy
  B_blind:      No timing info → sees random permutations → ~50% accuracy
  C_oracle:     Knows permutation directly (optimal baseline)
  D_frozen:     Fixed alpha=0.5 → ambiguous mapping → degraded

5 TESTS:
  T1: A_embodied acc > 80% (can learn permuted labels using timing)
  T2: B_blind acc < 60% (can't solve without timing)
  T3: A_embodied - B_blind gap > 20% (timing IS necessary)
  T4: Kill shot — ablate timing from trained A → accuracy collapses
  T5: DVFS changes which permutation is active
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
# HIP Kernel: Same as z2047 — timing determines alpha
# =============================================================================

HIP_SRC = r'''
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void hw_alpha_kernel(
    float* __restrict__ alpha_out,
    float* __restrict__ timing_out,
    const float* __restrict__ probe_data,
    int probe_size,
    int n,
    float t_min,
    float t_range
) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= n) return;

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
    float dt = (float)(t1 - t0);
    timing_out[gid] = dt;

    float a = (dt - t_min) / (t_range + 1e-6f);
    a = fminf(fmaxf(a, 0.0f), 1.0f);
    a = 1.0f / (1.0f + expf(-5.0f * (a - 0.5f)));
    alpha_out[gid] = a;

    if (__builtin_expect(sum == -1e30f, 0)) alpha_out[0] = sum;
}

__global__ void timing_calibrate_kernel(
    float* __restrict__ timings,
    const float* __restrict__ probe_data,
    int probe_size,
    int n
) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= n) return;
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
    torch::Tensor probe_data, int n, float t_min, float t_range) {
    auto opts = probe_data.options();
    auto alpha = torch::empty({n}, opts);
    auto timing = torch::empty({n}, opts);
    int threads = 128;
    int blocks = (n + threads - 1) / threads;
    hw_alpha_kernel<<<blocks, threads>>>(
        alpha.data_ptr<float>(), timing.data_ptr<float>(),
        probe_data.data_ptr<float>(), probe_data.size(0),
        n, t_min, t_range);
    return {alpha, timing};
}

torch::Tensor calibrate_timing(torch::Tensor probe_data, int n) {
    auto timings = torch::empty({n}, probe_data.options());
    int threads = 128;
    int blocks = (n + threads - 1) / threads;
    timing_calibrate_kernel<<<blocks, threads>>>(
        timings.data_ptr<float>(), probe_data.data_ptr<float>(),
        probe_data.size(0), n);
    return timings;
}
'''

HIP_CPP = r'''
std::vector<torch::Tensor> compute_hw_alpha(
    torch::Tensor probe_data, int n, float t_min, float t_range);
torch::Tensor calibrate_timing(torch::Tensor probe_data, int n);
'''


class KernelProbe:
    def __init__(self, n_threads=64, probe_size=1 << 22):
        from torch.utils.cpp_extension import load_inline
        print("[BUILD] Compiling kernel probe...")
        t0 = time.time()
        self.ext = load_inline(
            name='kernel_probe_z2048',
            cpp_sources=HIP_CPP,
            cuda_sources=HIP_SRC,
            functions=['compute_hw_alpha', 'calibrate_timing'],
            verbose=False,
            extra_cuda_cflags=['-O2']
        )
        print(f"[BUILD] Done in {time.time()-t0:.1f}s")
        self.n = n_threads
        self.probe_data = torch.randn(probe_size, device=DEVICE)
        self.t_min = 0.0
        self.t_range = 1.0
        self._calibrate()

    def _calibrate(self, n_rounds=10):
        all_t = []
        for _ in range(n_rounds):
            t = self.ext.calibrate_timing(self.probe_data, self.n)
            all_t.append(t.cpu().numpy())
        t = np.concatenate(all_t)
        self.t_min = float(np.percentile(t, 5))
        t_max = float(np.percentile(t, 95))
        self.t_range = max(t_max - self.t_min, 1.0)
        print(f"[CAL] t_min={self.t_min:.0f}, t_max={t_max:.0f}, "
              f"range={self.t_range:.0f}")

    def get_alpha(self):
        """Returns (alpha [n], timing [n])"""
        r = self.ext.compute_hw_alpha(self.probe_data, self.n,
                                       self.t_min, self.t_range)
        return r[0], r[1]

    def get_timing_bin(self):
        """Returns binary timing bin: 0 if alpha_mean ≤ 0.5, 1 if > 0.5"""
        alpha, _ = self.get_alpha()
        return 1 if alpha.mean().item() > 0.5 else 0


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
# Label Permutation Logic
# =============================================================================

def permute_labels(labels, timing_bin):
    """Apply timing-dependent label permutation.

    timing_bin=1 (high alpha): identity mapping
    timing_bin=0 (low alpha): shifted by 5: d → (d+5) % 10
    """
    if timing_bin == 1:
        return labels  # identity
    else:
        return (labels + 5) % 10  # shifted


# =============================================================================
# Model
# =============================================================================

class TimingAwareClassifier(nn.Module):
    """Classifier that receives timing alpha as an input feature.

    The model must learn: "when alpha > 0.5, use identity mapping;
    when alpha <= 0.5, use shifted mapping."
    """

    def __init__(self, alpha_dim=1, hidden_dim=128):
        super().__init__()
        # Image encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, hidden_dim), nn.ReLU(),
        )
        # Alpha (timing) integration — alpha_dim scalar features
        self.alpha_proj = nn.Linear(alpha_dim, hidden_dim, bias=False)
        nn.init.normal_(self.alpha_proj.weight, std=0.5)

        # Classifier (takes both image features and timing-modulated features)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x, alpha_scalar=None):
        """
        x: [B, 1, 28, 28]
        alpha_scalar: scalar float (mean alpha from kernel) or None
        """
        x_flat = x.view(x.shape[0], -1)
        h = self.encoder(x_flat)  # [B, hidden_dim]

        if alpha_scalar is not None:
            # Timing signal: project scalar alpha to hidden_dim
            a_input = torch.tensor([[alpha_scalar]], device=x.device)
            a_proj = self.alpha_proj(a_input)  # [1, hidden_dim]
            # Multiplicative modulation: timing scales the features
            h = h * (1.0 + torch.tanh(a_proj))  # [B, hidden_dim]

        logits = self.head(h)
        return logits


class BlindClassifier(nn.Module):
    """Same architecture but NO timing input."""

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, hidden_dim), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x, alpha_scalar=None):
        x_flat = x.view(x.shape[0], -1)
        h = self.encoder(x_flat)
        return self.head(h)


class OracleClassifier(nn.Module):
    """Directly receives timing_bin as a one-hot feature."""

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, hidden_dim), nn.ReLU(),
        )
        self.bin_proj = nn.Linear(2, hidden_dim, bias=False)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x, alpha_scalar=None, timing_bin=None):
        x_flat = x.view(x.shape[0], -1)
        h = self.encoder(x_flat)
        if timing_bin is not None:
            one_hot = torch.zeros(1, 2, device=x.device)
            one_hot[0, timing_bin] = 1.0
            h = h * (1.0 + torch.tanh(self.bin_proj(one_hot)))
        return self.head(h)


# =============================================================================
# Training
# =============================================================================

def train_condition(model, train_loader, test_loader, probe, dvfs,
                    mode, n_epochs=20, lr=1e-3):
    """
    Modes:
      'embodied': timing alpha from kernel, label permuted by timing bin
      'blind': no timing, label permuted by timing bin (impossible to solve)
      'oracle': timing bin directly given, label permuted by timing bin
      'frozen': fixed alpha=0.5, label permuted by timing bin
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    t0 = time.time()

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        bin_counts = [0, 0]

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # DVFS cycling
            if dvfs.available and batch_idx % 30 == 0:
                phase = (batch_idx // 30 + epoch * 5) % 3
                if phase == 0:
                    dvfs.force_low()
                elif phase == 1:
                    dvfs.auto()
                else:
                    dvfs.force_high()
                time.sleep(0.001)

            # Get timing from kernel
            with torch.no_grad():
                alpha, timing = probe.get_alpha()
            alpha_mean = alpha.mean().item()
            timing_bin = 1 if alpha_mean > 0.5 else 0
            bin_counts[timing_bin] += 1

            # Permute labels based on hardware timing
            perm_labels = permute_labels(labels, timing_bin)

            # Forward with or without timing info
            if mode == 'embodied':
                logits = model(images, alpha_scalar=alpha_mean)
            elif mode == 'oracle':
                logits = model(images, timing_bin=timing_bin)
            elif mode == 'frozen':
                logits = model(images, alpha_scalar=0.5)
            else:  # blind
                logits = model(images)

            loss = F.cross_entropy(logits, perm_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            correct += (preds == perm_labels).sum().item()
            total += labels.shape[0]
            total_loss += loss.item()

        acc = correct / total
        history.append({
            'epoch': epoch,
            'loss': total_loss / len(train_loader),
            'accuracy': acc,
            'bin_counts': bin_counts,
        })

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}: loss={history[-1]['loss']:.4f} "
                  f"acc={acc:.4f} bins={bin_counts}")

    train_time = time.time() - t0
    dvfs.auto()
    time.sleep(0.3)

    # Evaluate
    eval_result = evaluate(model, test_loader, probe, dvfs, mode)
    return {
        'train_history': history,
        'train_time_s': train_time,
        **eval_result,
    }


def evaluate(model, test_loader, probe, dvfs, mode,
             override_alpha=None, override_bin=None):
    """Evaluate with permuted labels."""
    model.eval()
    correct = 0
    total = 0
    bin_counts = [0, 0]

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Timing
            if override_alpha is not None:
                alpha_mean = override_alpha
                timing_bin = 1 if alpha_mean > 0.5 else 0
            elif override_bin is not None:
                alpha_mean = 0.8 if override_bin == 1 else 0.2
                timing_bin = override_bin
            else:
                alpha, timing = probe.get_alpha()
                alpha_mean = alpha.mean().item()
                timing_bin = 1 if alpha_mean > 0.5 else 0

            bin_counts[timing_bin] += 1
            perm_labels = permute_labels(labels, timing_bin)

            if mode == 'embodied':
                logits = model(images, alpha_scalar=alpha_mean)
            elif mode == 'oracle':
                logits = model(images, timing_bin=timing_bin)
            elif mode == 'frozen':
                logits = model(images, alpha_scalar=0.5)
            else:
                logits = model(images)

            preds = logits.argmax(dim=1)
            correct += (preds == perm_labels).sum().item()
            total += labels.shape[0]

    return {
        'accuracy': correct / total,
        'bin_counts': bin_counts,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("z2048: Kernel-Dependent Task — HW Timing Required for Labels")
    print("  Label permutation determined by HIP kernel timing")
    print("  Without timing info → can't solve the task")
    print("=" * 70)

    probe = KernelProbe(n_threads=64, probe_size=1 << 22)
    dvfs = DVFSController()

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

    # Check timing bin distribution
    print(f"\n--- Timing Bin Distribution Check ---")
    bin_counts = [0, 0]
    for _ in range(100):
        b = probe.get_timing_bin()
        bin_counts[b] += 1
        time.sleep(0.01)
    print(f"  Natural distribution: bin0={bin_counts[0]}, bin1={bin_counts[1]}")

    # Check DVFS effect on bins
    print(f"\n--- T5: DVFS → Timing Bin ---")
    dvfs_bins = {}
    for state, fn in [('low', dvfs.force_low), ('auto', dvfs.auto),
                       ('high', dvfs.force_high)]:
        fn()
        time.sleep(0.5)
        bins = []
        alphas = []
        for _ in range(30):
            a, _ = probe.get_alpha()
            am = a.mean().item()
            alphas.append(am)
            bins.append(1 if am > 0.5 else 0)
            time.sleep(0.01)
        dvfs_bins[state] = {
            'bin_distribution': [bins.count(0), bins.count(1)],
            'alpha_mean': float(np.mean(alphas)),
            'sclk': dvfs.get_sclk(),
        }
        print(f"  {state:5s}: SCLK={dvfs.get_sclk():5d} alpha={np.mean(alphas):.4f} "
              f"bins=[{bins.count(0)}, {bins.count(1)}]")
    dvfs.auto()

    # T5 check
    low_dominant = dvfs_bins['low']['bin_distribution']
    high_dominant = dvfs_bins['high']['bin_distribution']
    t5_pass = (low_dominant[0] != high_dominant[0]) or \
              abs(dvfs_bins['low']['alpha_mean'] - dvfs_bins['high']['alpha_mean']) > 0.05
    print(f"  T5: DVFS changes timing — {'PASS' if t5_pass else 'FAIL'}")

    # Train all conditions
    N_EPOCHS = 20
    conditions = {
        'A_embodied': ('embodied', TimingAwareClassifier),
        'B_blind':    ('blind', BlindClassifier),
        'C_oracle':   ('oracle', OracleClassifier),
        'D_frozen':   ('frozen', TimingAwareClassifier),
    }

    all_results = {
        'experiment': 'z2048_kernel_dependent_task',
        'timestamp': datetime.now().isoformat(),
        'dvfs_bins': dvfs_bins,
        'conditions': {},
    }

    for cond_name, (mode, model_cls) in conditions.items():
        print(f"\n{'='*60}")
        print(f"  Condition: {cond_name} (mode={mode})")
        print(f"{'='*60}")

        model = model_cls().to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        result = train_condition(model, train_loader, test_loader,
                                 probe, dvfs, mode, n_epochs=N_EPOCHS)

        # Kill shots for embodied
        kill_shots = {}
        if mode == 'embodied':
            print(f"\n  --- Kill Shot: Remove Timing (alpha=0.5) ---")
            ev = evaluate(model, test_loader, probe, dvfs, mode,
                          override_alpha=0.5)
            kill_shots['alpha_0.5'] = ev
            print(f"    acc={ev['accuracy']:.4f}")

            print(f"  --- Kill Shot: Force alpha=0.0 ---")
            ev0 = evaluate(model, test_loader, probe, dvfs, mode,
                           override_alpha=0.0)
            kill_shots['alpha_0.0'] = ev0
            print(f"    acc={ev0['accuracy']:.4f}")

            print(f"  --- Kill Shot: Force alpha=1.0 ---")
            ev1 = evaluate(model, test_loader, probe, dvfs, mode,
                           override_alpha=1.0)
            kill_shots['alpha_1.0'] = ev1
            print(f"    acc={ev1['accuracy']:.4f}")

            # Per-DVFS evaluation
            print(f"\n  --- Per-DVFS ---")
            for state, fn in [('low', dvfs.force_low), ('auto', dvfs.auto),
                               ('high', dvfs.force_high)]:
                fn()
                time.sleep(0.5)
                ev_d = evaluate(model, test_loader, probe, dvfs, mode)
                kill_shots[f'dvfs_{state}'] = ev_d
                print(f"    {state:5s}: acc={ev_d['accuracy']:.4f} bins={ev_d['bin_counts']}")
            dvfs.auto()

        all_results['conditions'][cond_name] = {
            'mode': mode,
            'n_params': n_params,
            'final_accuracy': result['accuracy'],
            'train_time_s': result['train_time_s'],
            'train_history': result['train_history'],
            'kill_shots': kill_shots,
        }
        print(f"  Final: acc={result['accuracy']:.4f}")

    # Analysis
    print(f"\n{'='*70}")
    print("  Cross-Condition Analysis")
    print(f"{'='*70}")

    print(f"\n  {'Condition':<14} {'Accuracy':>10}")
    print("  " + "-" * 30)
    for cond, res in all_results['conditions'].items():
        print(f"  {cond:<14} {res['final_accuracy']:>10.4f}")

    emb = all_results['conditions']['A_embodied']
    blind = all_results['conditions']['B_blind']
    oracle = all_results['conditions']['C_oracle']
    frozen = all_results['conditions']['D_frozen']

    t1_pass = emb['final_accuracy'] >= 0.80
    t2_pass = blind['final_accuracy'] < 0.60
    gap = emb['final_accuracy'] - blind['final_accuracy']
    t3_pass = gap > 0.20
    ks = emb.get('kill_shots', {})
    ks_acc = ks.get('alpha_0.5', {}).get('accuracy', emb['final_accuracy'])
    t4_drop = emb['final_accuracy'] - ks_acc
    t4_pass = t4_drop > 0.10

    print(f"\n  T1: A_embodied acc={emb['final_accuracy']:.4f} >= 0.80 — "
          f"{'PASS' if t1_pass else 'FAIL'}")
    print(f"  T2: B_blind acc={blind['final_accuracy']:.4f} < 0.60 — "
          f"{'PASS' if t2_pass else 'FAIL'}")
    print(f"  T3: Gap A-B={gap:.4f} > 0.20 — "
          f"{'PASS' if t3_pass else 'FAIL'}")
    print(f"  T4: Kill shot drop={t4_drop:.4f} > 0.10 — "
          f"{'PASS' if t4_pass else 'FAIL'}")
    print(f"  T5: DVFS → bin change — {'PASS' if t5_pass else 'FAIL'}")
    print(f"  C_oracle acc={oracle['final_accuracy']:.4f} (upper bound)")
    print(f"  D_frozen acc={frozen['final_accuracy']:.4f}")

    n_pass = sum([t1_pass, t2_pass, t3_pass, t4_pass, t5_pass])
    all_results['tests'] = {
        'T1_embodied_learns': {'pass': bool(t1_pass),
                                'detail': f"acc={emb['final_accuracy']:.4f}"},
        'T2_blind_fails': {'pass': bool(t2_pass),
                           'detail': f"acc={blind['final_accuracy']:.4f}"},
        'T3_gap': {'pass': bool(t3_pass), 'detail': f"gap={gap:.4f}"},
        'T4_kill_shot': {'pass': bool(t4_pass), 'detail': f"drop={t4_drop:.4f}"},
        'T5_dvfs_bins': {'pass': bool(t5_pass), 'detail': str(dvfs_bins)},
    }
    all_results['verdict'] = f'{n_pass}/5 PASS'

    print(f"\n{'='*70}")
    print(f"  VERDICT: {n_pass}/5 PASS")
    print(f"{'='*70}")

    all_results['notes'] = {
        'innovation': 'Label permutation determined by HIP kernel timing. '
                      'Task is IMPOSSIBLE without timing information.',
        'depth': 'Kernel-level timing → label mapping. '
                 'Model must decode timing to know which permutation is active.',
        'caveat': 'This is DESIGNED to require timing. The question is whether '
                  'the model can learn to USE the timing signal from the kernel.',
    }

    dvfs.cleanup()

    results_path = RESULTS_DIR / 'z2048_kernel_dependent_task.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[SAVED] {results_path}")


if __name__ == '__main__':
    main()
