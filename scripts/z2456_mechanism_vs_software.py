#!/usr/bin/env python3
"""
z2456: The decisive experiment — mechanism vs software-matched control

For each mechanism, test THREE conditions with IDENTICAL parameter budget:
  A. Baseline: standard ReLU MLP
  B. Software control: differentiable module doing the SAME function
  C. Real HIP mechanism: actual GPU hardware instruction

If C > B → real mechanism value (hardware does something software can't)
If C ≈ B → mechanism is just regularization reimplemented in hardware
If C < B → mechanism hurts (proxy mismatch or hardware noise is destructive)

4 mechanisms tested individually:
  1. Shuffle lateral inhibition (__shfl_xor vs torch index swap)
  2. Branch E/I split (hardware branch divergence vs LeakyReLU)
  3. Ballot gating (__ballot population vote vs soft sigmoid gate)
  4. All three combined

Also test: unknown noise shift (train clean, test noisy)
  - This is the niche where implicit mechanism regularization
    might beat explicit software regularization.
"""
import os, time, json
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

base = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy'
device = torch.device('cuda')

from torch.utils.cpp_extension import load_inline

# Separate HIP kernels for each mechanism
HIP_SRC = r"""
#include <hip/hip_runtime.h>

// Mechanism 1: ONLY shuffle
__global__ void k_shfl(const float*X,const float*W,const float*b,float*Y,
                        int N,int id,int od,float alpha){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;int s=idx/od,n=idx%od;if(s>=N)return;
    float sum=b[n];for(int k=0;k<id;k++)sum+=W[n*id+k]*X[s*id+k];
    float ro=(sum>0)?sum:0;
    float nb=__shfl_xor(ro,1);
    Y[s*od+n]=ro+alpha*(ro-nb)*0.5f;
}

// Mechanism 2: ONLY branch E/I
__global__ void k_branch(const float*X,const float*W,const float*b,float*Y,
                          int N,int id,int od,float alpha,float lambda_inh){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;int s=idx/od,n=idx%od;if(s>=N)return;
    float sum=b[n],exc=0,inh=0;
    for(int k=0;k<id;k++){float p=W[n*id+k]*X[s*id+k];sum+=p;if(p>0)exc+=p;else inh+=p;}
    float ro=(sum>0)?sum:0;
    float bs=exc+inh*lambda_inh+b[n];float br=(bs>0)?bs:0;
    Y[s*od+n]=ro+alpha*(br-ro);
}

// Mechanism 3: ONLY ballot gate
__global__ void k_ballot(const float*X,const float*W,const float*b,float*Y,
                          int N,int id,int od,float alpha){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;int s=idx/od,n=idx%od;if(s>=N)return;
    float sum=b[n];for(int k=0;k<id;k++)sum+=W[n*id+k]*X[s*id+k];
    float ro=(sum>0)?sum:0;
    int na=__popcll(__ballot(ro>0));float ar=(float)na/64.0f;
    float g=1.0f/(1.0f+expf(-10.0f*(0.5f-ar)));
    Y[s*od+n]=ro+alpha*ro*(g-1.0f);
}

// All three combined
__global__ void k_all(const float*X,const float*W,const float*b,float*Y,
                       int N,int id,int od,float as_,float ab,float li,float ag){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;int s=idx/od,n=idx%od;if(s>=N)return;
    float sum=b[n],exc=0,inh=0;
    for(int k=0;k<id;k++){float p=W[n*id+k]*X[s*id+k];sum+=p;if(p>0)exc+=p;else inh+=p;}
    float ro=(sum>0)?sum:0,r=ro;
    float nb=__shfl_xor(ro,1);r+=as_*(ro-nb)*0.5f;
    float bs=exc+inh*li+b[n];float br=(bs>0)?bs:0;r+=ab*(br-ro);
    int na=__popcll(__ballot(ro>0));float ar=(float)na/64.0f;
    float g=1.0f/(1.0f+expf(-10.0f*(0.5f-ar)));r+=ag*ro*(g-1.0f);
    Y[s*od+n]=r;
}

torch::Tensor f_shfl(torch::Tensor X,torch::Tensor W,torch::Tensor b,float a){
    int N=X.size(0),id=X.size(1),od=W.size(0);auto Y=torch::empty({N,od},X.options());
    k_shfl<<<(N*od+255)/256,256>>>(X.data_ptr<float>(),W.data_ptr<float>(),b.data_ptr<float>(),Y.data_ptr<float>(),N,id,od,a);return Y;}
torch::Tensor f_branch(torch::Tensor X,torch::Tensor W,torch::Tensor b,float a,float li){
    int N=X.size(0),id=X.size(1),od=W.size(0);auto Y=torch::empty({N,od},X.options());
    k_branch<<<(N*od+255)/256,256>>>(X.data_ptr<float>(),W.data_ptr<float>(),b.data_ptr<float>(),Y.data_ptr<float>(),N,id,od,a,li);return Y;}
torch::Tensor f_ballot(torch::Tensor X,torch::Tensor W,torch::Tensor b,float a){
    int N=X.size(0),id=X.size(1),od=W.size(0);auto Y=torch::empty({N,od},X.options());
    k_ballot<<<(N*od+255)/256,256>>>(X.data_ptr<float>(),W.data_ptr<float>(),b.data_ptr<float>(),Y.data_ptr<float>(),N,id,od,a);return Y;}
torch::Tensor f_all(torch::Tensor X,torch::Tensor W,torch::Tensor b,float a,float ab,float li,float ag){
    int N=X.size(0),id=X.size(1),od=W.size(0);auto Y=torch::empty({N,od},X.options());
    k_all<<<(N*od+255)/256,256>>>(X.data_ptr<float>(),W.data_ptr<float>(),b.data_ptr<float>(),Y.data_ptr<float>(),N,id,od,a,ab,li,ag);return Y;}
"""

CPP = """
torch::Tensor f_shfl(torch::Tensor,torch::Tensor,torch::Tensor,float);
torch::Tensor f_branch(torch::Tensor,torch::Tensor,torch::Tensor,float,float);
torch::Tensor f_ballot(torch::Tensor,torch::Tensor,torch::Tensor,float);
torch::Tensor f_all(torch::Tensor,torch::Tensor,torch::Tensor,float,float,float,float);
"""

print("Compiling 4 mechanism kernels...")
ext = load_inline(name='z2456', cpp_sources=CPP, cuda_sources=HIP_SRC,
                  functions=['f_shfl','f_branch','f_ballot','f_all'],
                  extra_cuda_cflags=['-O2','--offload-arch=gfx1100'], verbose=False)
print("OK\n")

# MNIST
with open(f'{base}/data/MNIST/raw/train-images-idx3-ubyte','rb') as f:f.read(16);tri=np.frombuffer(f.read(),dtype=np.uint8).reshape(-1,784).astype(np.float32)/255
with open(f'{base}/data/MNIST/raw/train-labels-idx1-ubyte','rb') as f:f.read(8);trl=np.frombuffer(f.read(),dtype=np.uint8)
with open(f'{base}/data/MNIST/raw/t10k-images-idx3-ubyte','rb') as f:f.read(16);tei=np.frombuffer(f.read(),dtype=np.uint8).reshape(-1,784).astype(np.float32)/255
with open(f'{base}/data/MNIST/raw/t10k-labels-idx1-ubyte','rb') as f:f.read(8);tel=np.frombuffer(f.read(),dtype=np.uint8)
Xtr=torch.tensor(tri,device=device);ytr=torch.tensor(trl,dtype=torch.long,device=device)
Xte=torch.tensor(tei,device=device);yte=torch.tensor(tel,dtype=torch.long,device=device)

W = 128  # width
ALPHA = 0.15  # mechanism strength
LAMBDA = 0.75  # inhibitory coefficient
BS = 256
EP_PHASE1 = 10  # ReLU-only warmup
EP_PHASE2 = 15  # with mechanism/control
SEEDS = 3

def shfl_proxy(x):
    B,D=x.shape;return x.view(B,D//2,2).flip(-1).view(B,D)

# ============================================================
# Software-matched controls (differentiable, same function)
# ============================================================
def sw_shfl(x, alpha):
    """Software match for __shfl_xor: swap adjacent pairs, lateral diff."""
    nb = shfl_proxy(x)
    return x + alpha * (x - nb) * 0.5

def sw_branch(raw, relu_out, alpha, li):
    """Software match for branch E/I: LeakyReLU approximation."""
    branch = F.leaky_relu(raw, li)
    return relu_out + alpha * (branch - relu_out)

def sw_ballot(x, alpha):
    """Software match for ballot gating: soft population regulation."""
    af = (x > 0).float().mean(-1, keepdim=True)
    g = torch.sigmoid(-10 * (af - 0.5))
    return x + alpha * x * (g - 1)

# ============================================================
# Test harness
# ============================================================
def run_experiment(mech_name, proxy_fn, real_fn, seeds=SEEDS):
    """Train and evaluate one mechanism vs its software control."""
    results = {'A_baseline': [], 'B_software': [], 'C_mechanism': []}

    for seed in range(seeds):
        torch.manual_seed(seed)

        # Shared weights (same init for all three conditions)
        w1_init = torch.randn(W, 784, device=device) * (2/784)**0.5
        b1_init = torch.zeros(W, device=device)
        w2_init = torch.randn(W, W, device=device) * (2/W)**0.5
        b2_init = torch.zeros(W, device=device)
        wh_init = torch.randn(10, W, device=device) * (2/W)**0.5
        bh_init = torch.zeros(10, device=device)

        for cond in ['A_baseline', 'B_software', 'C_mechanism']:
            w1 = nn.Linear(784, W).to(device); w1.weight.data = w1_init.clone(); w1.bias.data = b1_init.clone()
            w2 = nn.Linear(W, W).to(device); w2.weight.data = w2_init.clone(); w2.bias.data = b2_init.clone()
            hd = nn.Linear(W, 10).to(device); hd.weight.data = wh_init.clone(); hd.bias.data = bh_init.clone()
            alpha_p = nn.Parameter(torch.tensor(0.0, device=device))

            all_p = list(w1.parameters()) + list(w2.parameters()) + list(hd.parameters())
            if cond != 'A_baseline':
                all_p.append(alpha_p)
            opt = torch.optim.Adam(all_p, lr=0.001)

            # Phase 1: pure ReLU (alpha frozen)
            alpha_p.requires_grad = False
            for ep in range(EP_PHASE1):
                pm = torch.randperm(len(Xtr), device=device)
                for i in range(0, len(Xtr), BS):
                    x = Xtr[pm[i:i+BS]]
                    h = F.relu(w1(x)); h = F.relu(w2(h))
                    l = F.cross_entropy(hd(h), ytr[pm[i:i+BS]])
                    opt.zero_grad(); l.backward(); opt.step()

            if cond == 'A_baseline':
                # No phase 2 for baseline
                pass
            else:
                alpha_p.requires_grad = True
                for ep in range(EP_PHASE2):
                    pm = torch.randperm(len(Xtr), device=device)
                    for i in range(0, len(Xtr), BS):
                        x = Xtr[pm[i:i+BS]]
                        a = torch.tanh(alpha_p) * ALPHA

                        # Layer 1
                        raw1 = w1(x); ro1 = F.relu(raw1)
                        h = proxy_fn(raw1, ro1, a) if cond == 'B_software' else ro1
                        # Layer 2
                        raw2 = w2(h); ro2 = F.relu(raw2)
                        h = proxy_fn(raw2, ro2, a) if cond == 'B_software' else ro2

                        l = F.cross_entropy(hd(h), ytr[pm[i:i+BS]])
                        opt.zero_grad(); l.backward(); opt.step()

            # Eval
            with torch.no_grad():
                a_val = (torch.tanh(alpha_p) * ALPHA).item() if cond != 'A_baseline' else 0

                # Clean + noisy eval (3 noise levels)
                accs = {}
                for sigma in [0.0, 0.2, 0.3]:
                    x = Xte if sigma == 0 else (Xte + torch.randn_like(Xte)*sigma).clamp(0,1)

                    if cond == 'C_mechanism':
                        h = real_fn(x.contiguous(), w1.weight.contiguous(), w1.bias.contiguous(), a_val)
                        h = real_fn(h.contiguous(), w2.weight.contiguous(), w2.bias.contiguous(), a_val)
                    elif cond == 'B_software':
                        raw1 = w1(x); ro1 = F.relu(raw1)
                        h = proxy_fn(raw1, ro1, a_val)
                        raw2 = w2(h); ro2 = F.relu(raw2)
                        h = proxy_fn(raw2, ro2, a_val)
                    else:
                        h = F.relu(w1(x)); h = F.relu(w2(h))

                    accs[f's{sigma}'] = (hd(h).argmax(1)==yte).float().mean().item()*100

                accs['alpha'] = a_val
                results[cond].append(accs)

    return results

# ============================================================
# Run all 4 mechanism tests
# ============================================================
print("=" * 80)
print("z2456: MECHANISM vs SOFTWARE-MATCHED CONTROL")
print("=" * 80)

# Mechanism 1: Shuffle
def proxy_shfl(raw, relu_out, alpha):
    return sw_shfl(relu_out, alpha)
def real_shfl(x, W, b, alpha):
    return ext.f_shfl(x, W, b, alpha)

# Mechanism 2: Branch
def proxy_branch(raw, relu_out, alpha):
    return sw_branch(raw, relu_out, alpha, LAMBDA)
def real_branch(x, W, b, alpha):
    return ext.f_branch(x, W, b, alpha, LAMBDA)

# Mechanism 3: Ballot
def proxy_ballot(raw, relu_out, alpha):
    return sw_ballot(relu_out, alpha)
def real_ballot(x, W, b, alpha):
    return ext.f_ballot(x, W, b, alpha)

# Mechanism 4: All combined
def proxy_all(raw, relu_out, alpha):
    h = sw_shfl(relu_out, alpha)
    h = h + alpha * (F.leaky_relu(raw, LAMBDA) - relu_out)
    af = (relu_out > 0).float().mean(-1, keepdim=True)
    g = torch.sigmoid(-10*(af-0.5))
    h = h + alpha * relu_out * (g-1)
    return h
def real_all(x, W, b, alpha):
    return ext.f_all(x, W, b, alpha, alpha, LAMBDA, alpha)

tests = [
    ('shuffle', proxy_shfl, real_shfl),
    ('branch', proxy_branch, real_branch),
    ('ballot', proxy_ballot, real_ballot),
    ('all_3', proxy_all, real_all),
]

all_results = {}
for mech_name, proxy_fn, real_fn in tests:
    print(f"\n--- {mech_name.upper()} ---")
    r = run_experiment(mech_name, proxy_fn, real_fn)
    all_results[mech_name] = r

    # Print table
    for sigma in ['s0.0', 's0.2', 's0.3']:
        sl = sigma.replace('s','σ=')
        a_vals = [x[sigma] for x in r['A_baseline']]
        b_vals = [x[sigma] for x in r['B_software']]
        c_vals = [x[sigma] for x in r['C_mechanism']]
        a_m, b_m, c_m = np.mean(a_vals), np.mean(b_vals), np.mean(c_vals)
        cb = c_m - b_m  # mechanism vs software
        ba = b_m - a_m  # software vs baseline
        marker = "★HW" if cb > 0.3 else "≈SW" if abs(cb) < 0.3 else "✗"
        print(f"  {sl}: A(base)={a_m:.1f}% B(sw)={b_m:.1f}%({ba:+.1f}) C(hw)={c_m:.1f}%({cb:+.1f}vs B) {marker}")

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*80}")
print("DECISIVE VERDICT — Does hardware mechanism beat software match?")
print(f"{'='*80}")
print(f"\n{'Mechanism':>10} | {'Clean C-B':>10} | {'σ=0.2 C-B':>10} | {'σ=0.3 C-B':>10} | Verdict")
print("-"*65)

for mech_name, r in all_results.items():
    deltas = {}
    for sigma in ['s0.0', 's0.2', 's0.3']:
        b_m = np.mean([x[sigma] for x in r['B_software']])
        c_m = np.mean([x[sigma] for x in r['C_mechanism']])
        deltas[sigma] = c_m - b_m

    # Verdict: HW wins if C > B at any noise level by > 0.5pp
    hw_wins = any(d > 0.5 for d in deltas.values())
    hw_loses = any(d < -1.0 for d in deltas.values())
    verdict = "HW WINS" if hw_wins and not hw_loses else "≈ SAME" if not hw_loses else "SW WINS"

    print(f"{mech_name:>10} | {deltas['s0.0']:>+9.2f}pp | {deltas['s0.2']:>+9.2f}pp | {deltas['s0.3']:>+9.2f}pp | {verdict}")

with open(f'{base}/results/z2456_mechanism_vs_software.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=float)
print(f"\nSaved to results/z2456_mechanism_vs_software.json")
