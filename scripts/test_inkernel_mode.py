#!/usr/bin/env python3
"""Test: does MODE register change math when set INSIDE the compute kernel?"""
import torch, os, numpy as np
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')
os.environ.setdefault('PYTORCH_ROCM_ARCH', 'gfx1100')
from torch.utils.cpp_extension import load_inline

HIP_SRC = r'''
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void mode_linear_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int* __restrict__ side_out,
    int mode_byte, int B, int in_f, int out_f)
{
    int b = blockIdx.x;
    int o = blockIdx.y;
    if (b >= B || o >= out_f) return;

    unsigned int c0;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(c0));
    c0 = __builtin_amdgcn_readfirstlane(c0);

    // SET MODE — changes v_fma_f32/v_add_f32 rounding behavior
    unsigned int mb = __builtin_amdgcn_readfirstlane((unsigned int)(mode_byte & 0xFF));
    asm volatile("s_setreg_b32 hwreg(1, 0, 8), %0" : : "s"(mb));

    // Dot product UNDER the chosen MODE
    float acc = 0.0f;
    for (int i = (int)threadIdx.x; i < in_f; i += (int)blockDim.x) {
        acc += input[b * in_f + i] * weight[o * in_f + i];
    }

    // Warp reduction (also under our MODE)
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (threadIdx.x == 0) {
        output[b * out_f + o] = acc + bias[o];
    }

    unsigned int c1;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(c1));
    unsigned int mode_rb;
    asm volatile("s_getreg_b32 %0, hwreg(1, 0, 8)" : "=s"(mode_rb));

    // Restore default MODE
    unsigned int restore = __builtin_amdgcn_readfirstlane(0xF0u);
    asm volatile("s_setreg_b32 hwreg(1, 0, 8), %0" : : "s"(restore));

    if (threadIdx.x == 0 && o == 0) {
        side_out[b * 3 + 0] = (int)(c1 - c0);
        side_out[b * 3 + 1] = (int)(mode_rb & 0xFF);
        side_out[b * 3 + 2] = (int)(c0 & 0xFFFF);
    }
}

std::vector<torch::Tensor> mode_linear(torch::Tensor input, torch::Tensor weight,
                                         torch::Tensor bias, int mode_byte) {
    int B = input.size(0);
    int in_f = input.size(1);
    int out_f = weight.size(0);
    auto output = torch::zeros({B, out_f}, input.options());
    auto side = torch::zeros({B, 3}, torch::TensorOptions().dtype(torch::kInt32).device(input.device()));
    dim3 grid(B, out_f);
    dim3 block(32);
    mode_linear_kernel<<<grid, block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), side.data_ptr<int>(),
        mode_byte, B, in_f, out_f);
    return {output, side};
}
'''
CPP_SRC = r'''
#include <torch/extension.h>
std::vector<torch::Tensor> mode_linear(torch::Tensor input, torch::Tensor weight,
                                         torch::Tensor bias, int mode_byte);
'''

print("Compiling in-kernel MODE linear layer...")
ext = load_inline(name='mode_linear_test', cpp_sources=CPP_SRC, cuda_sources=HIP_SRC,
                  functions=['mode_linear'],
                  extra_cuda_cflags=['-O2', '--offload-arch=gfx1100'], verbose=False)

B, in_f, out_f = 64, 128, 64
torch.manual_seed(42)
x = torch.randn(B, in_f, device='cuda')
w = torch.randn(out_f, in_f, device='cuda')
b = torch.zeros(out_f, device='cuda')

# Reference: standard PyTorch
ref = torch.mm(x, w.t()) + b

# Test different MODE bytes
modes = {
    '0xF0 (nearest+noflush)': 0xF0,
    '0xFF (toward-zero+noflush)': 0xFF,
    '0xF5 (toward+inf SP)': 0xF5,
    '0xFA (toward-inf SP)': 0xFA,
    '0xF3 (toward-zero SP only)': 0xF3,
    '0x00 (nearest+flush-all)': 0x00,
    '0x0F (toward-zero+flush)': 0x0F,
}

results = {}
for name, mode in modes.items():
    out, side = ext.mode_linear(x, w, b, mode)
    torch.cuda.synchronize()
    results[name] = {
        'output': out,
        'mode_rb': side[0, 1].item(),
        'cycles': side[:, 0].float().mean().item(),
    }

baseline = results['0xF0 (nearest+noflush)']['output']

print("\n=== IN-KERNEL MODE TEST ===\n")
for name, r in results.items():
    diff = (baseline - r['output']).abs()
    out_bits = r['output'].view(-1).cpu().numpy().view(np.uint32)
    base_bits = baseline.view(-1).cpu().numpy().view(np.uint32)
    n_diff = np.sum(out_bits != base_bits)
    total = out_bits.size
    pct = n_diff / total * 100
    print(f"  {name}:")
    print(f"    MODE readback: 0x{r['mode_rb']:02X}")
    print(f"    max|diff|: {diff.max():.8f}  mean|diff|: {diff.mean():.8f}")
    print(f"    bits differ: {n_diff}/{total} ({pct:.1f}%)")
    print(f"    cycles: {r['cycles']:.0f}")
    print()

# Also check custom vs PyTorch reference
diff_ref = (ref - baseline).abs()
print(f"Custom(0xF0) vs PyTorch mm: max={diff_ref.max():.8f} mean={diff_ref.mean():.8f}")
print()

# Summary
any_changed = False
for name, r in results.items():
    if '0xF0' in name: continue
    out_bits = r['output'].view(-1).cpu().numpy().view(np.uint32)
    base_bits = baseline.view(-1).cpu().numpy().view(np.uint32)
    if np.sum(out_bits != base_bits) > 0:
        any_changed = True
        break

if any_changed:
    print(">>> CONFIRMED: MODE CHANGES MATH INSIDE THE KERNEL <<<")
    print(">>> ISA register → transistor-level rounding → different matmul output <<<")
    print(">>> This IS below firmware — direct hardware register control <<<")
else:
    print(">>> MODE did NOT change output — compiler may optimize away <<<")
