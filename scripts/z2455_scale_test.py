#!/usr/bin/env python3
"""
z2455: Scale test — depth × width × dataset × noise

z2454 confirmed: +7.3pp robustness at σ=0.3, 0pp clean cost, 5 seeds.
Now test if it scales:
  - Depths: 1, 2, 4 hidden layers
  - Widths: 64, 128, 256
  - Datasets: MNIST, Fashion-MNIST
  - Noise: σ=0, 0.2, 0.3
  - Overhead: time per batch
"""
import os, time, json
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

base = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy'

from torch.utils.cpp_extension import load_inline
HIP = r"""
#include <hip/hip_runtime.h>
__global__ void mk(const float*X,const float*W,const float*b,float*Y,int N,int id,int od,float as_,float ab,float li,float ag){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;int s=idx/od,n=idx%od;if(s>=N)return;
    float sum=b[n],exc=0,inh=0;
    for(int k=0;k<id;k++){float p=W[n*id+k]*X[s*id+k];sum+=p;if(p>0)exc+=p;else inh+=p;}
    float ro=(sum>0)?sum:0,r=ro;
    if(as_!=0){float nb=__shfl_xor(ro,1);r+=as_*(ro-nb)*0.5f;}
    if(ab!=0){float bs_=exc+inh*li+b[n];float br=(bs_>0)?bs_:0;r+=ab*(br-ro);}
    if(ag!=0){int na=__popcll(__ballot(ro>0));float ar=(float)na/64.0f;float g=1.0f/(1.0f+expf(-10.0f*(0.5f-ar)));r+=ag*ro*(g-1.0f);}
    Y[s*od+n]=r;
}
torch::Tensor ml(torch::Tensor X,torch::Tensor W,torch::Tensor b,float a,float ab,float li,float ag){
    int N=X.size(0),id=X.size(1),od=W.size(0);auto Y=torch::empty({N,od},X.options());
    mk<<<(N*od+255)/256,256>>>(X.data_ptr<float>(),W.data_ptr<float>(),b.data_ptr<float>(),Y.data_ptr<float>(),N,id,od,a,ab,li,ag);return Y;}
"""
ext = load_inline(name='z2455', cpp_sources='torch::Tensor ml(torch::Tensor X,torch::Tensor W,torch::Tensor b,float,float,float,float);',
                  cuda_sources=HIP, functions=['ml'], extra_cuda_cflags=['-O2','--offload-arch=gfx1100'], verbose=False)

device = torch.device('cuda')
MA = 0.3  # max alpha

def shfl_proxy(x):
    B,D=x.shape
    if D%2==0: return x.view(B,D//2,2).flip(-1).view(B,D)
    return torch.roll(x,1,-1)

def load_dataset(name):
    if name == 'mnist':
        p = f'{base}/data/MNIST/raw'
    else:
        p = f'{base}/data/FashionMNIST/raw'
    with open(f'{p}/train-images-idx3-ubyte','rb') as f: f.read(16); tri=np.frombuffer(f.read(),dtype=np.uint8).reshape(-1,784).astype(np.float32)/255
    with open(f'{p}/train-labels-idx1-ubyte','rb') as f: f.read(8); trl=np.frombuffer(f.read(),dtype=np.uint8)
    with open(f'{p}/t10k-images-idx3-ubyte','rb') as f: f.read(16); tei=np.frombuffer(f.read(),dtype=np.uint8).reshape(-1,784).astype(np.float32)/255
    with open(f'{p}/t10k-labels-idx1-ubyte','rb') as f: f.read(8); tel=np.frombuffer(f.read(),dtype=np.uint8)
    return (torch.tensor(tri,device=device), torch.tensor(trl,dtype=torch.long,device=device),
            torch.tensor(tei,device=device), torch.tensor(tel,dtype=torch.long,device=device))

def build_baseline(depth, width):
    layers = [nn.Linear(784, width), nn.ReLU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.ReLU()]
    layers.append(nn.Linear(width, 10))
    return nn.Sequential(*layers).to(device)

def train_and_eval(dataset_name, depth, width, n_epochs=15):
    Xtr, ytr, Xte, yte = load_dataset(dataset_name)

    # Baseline
    torch.manual_seed(42)
    bl = build_baseline(depth, width)
    ob = torch.optim.Adam(bl.parameters(), lr=0.001)
    for ep in range(n_epochs):
        pm = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(Xtr), 256):
            l = F.cross_entropy(bl(Xtr[pm[i:i+256]]), ytr[pm[i:i+256]])
            ob.zero_grad(); l.backward(); ob.step()

    # Mechanism model
    torch.manual_seed(42)
    linears = nn.ModuleList()
    linears.append(nn.Linear(784, width))
    for _ in range(depth - 1):
        linears.append(nn.Linear(width, width))
    head = nn.Linear(width, 10)
    linears = linears.to(device)
    head = head.to(device)

    alphas = []
    for _ in range(depth):
        a = [nn.Parameter(torch.tensor(0.0, device=device)) for _ in range(3)]
        alphas.append(a)
    all_params = list(linears.parameters()) + list(head.parameters())
    for a_list in alphas:
        all_params += a_list
    om = torch.optim.Adam(all_params, lr=0.001)

    # Phase 1: freeze alphas
    for a_list in alphas:
        for a in a_list: a.requires_grad = False
    for ep in range(n_epochs // 2):
        pm = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(Xtr), 256):
            x = Xtr[pm[i:i+256]]
            for lin in linears: x = F.relu(lin(x))
            l = F.cross_entropy(head(x), ytr[pm[i:i+256]])
            om.zero_grad(); l.backward(); om.step()

    # Phase 2: unfreeze
    for a_list in alphas:
        for a in a_list: a.requires_grad = True
    for ep in range(n_epochs):
        pm = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(Xtr), 256):
            x = Xtr[pm[i:i+256]]
            for li, (a_s, a_b, a_g) in zip(linears, alphas):
                raw = li(x)
                ro = F.relu(raw)
                r = ro
                r = r + torch.tanh(a_s)*MA*(ro - shfl_proxy(ro))*0.5
                r = r + torch.tanh(a_b)*MA*(F.leaky_relu(raw, 0.75) - ro)
                af = (ro>0).float().mean(-1, keepdim=True)
                g = torch.sigmoid(-10*(af-0.5))
                r = r + torch.tanh(a_g)*MA*ro*(g-1)
                x = r
            l = F.cross_entropy(head(x), ytr[pm[i:i+256]])
            om.zero_grad(); l.backward(); om.step()

    # Eval
    bl.eval()
    with torch.no_grad():
        # Clean
        a_bl = (bl(Xte).argmax(1)==yte).float().mean().item()*100

        # Real kernel
        x = Xte
        for lin, (a_s, a_b, a_g) in zip(linears, alphas):
            x = ext.ml(x.contiguous(), lin.weight.contiguous(), lin.bias.contiguous(),
                       (torch.tanh(a_s)*MA).item(), (torch.tanh(a_b)*MA).item(),
                       0.75, (torch.tanh(a_g)*MA).item())
        a_mech = (head(x).argmax(1)==yte).float().mean().item()*100

        # Noisy σ=0.2
        noisy2 = (Xte + torch.randn_like(Xte)*0.2).clamp(0,1)
        a_bl_n2 = (bl(noisy2).argmax(1)==yte).float().mean().item()*100
        x = noisy2
        for lin, (a_s, a_b, a_g) in zip(linears, alphas):
            x = ext.ml(x.contiguous(), lin.weight.contiguous(), lin.bias.contiguous(),
                       (torch.tanh(a_s)*MA).item(), (torch.tanh(a_b)*MA).item(),
                       0.75, (torch.tanh(a_g)*MA).item())
        a_mech_n2 = (head(x).argmax(1)==yte).float().mean().item()*100

        # Noisy σ=0.3
        noisy3 = (Xte + torch.randn_like(Xte)*0.3).clamp(0,1)
        a_bl_n3 = (bl(noisy3).argmax(1)==yte).float().mean().item()*100
        x = noisy3
        for lin, (a_s, a_b, a_g) in zip(linears, alphas):
            x = ext.ml(x.contiguous(), lin.weight.contiguous(), lin.bias.contiguous(),
                       (torch.tanh(a_s)*MA).item(), (torch.tanh(a_b)*MA).item(),
                       0.75, (torch.tanh(a_g)*MA).item())
        a_mech_n3 = (head(x).argmax(1)==yte).float().mean().item()*100

    # Timing
    torch.cuda.synchronize()
    x_bench = Xte[:256]

    t0 = time.perf_counter()
    for _ in range(100):
        bl(x_bench)
    torch.cuda.synchronize()
    t_bl = (time.perf_counter()-t0)/100*1000

    t0 = time.perf_counter()
    for _ in range(100):
        x = x_bench
        for lin, (a_s,a_b,a_g) in zip(linears, alphas):
            x = ext.ml(x.contiguous(), lin.weight.contiguous(), lin.bias.contiguous(),
                       (torch.tanh(a_s)*MA).item(), (torch.tanh(a_b)*MA).item(),
                       0.75, (torch.tanh(a_g)*MA).item())
        head(x)
    torch.cuda.synchronize()
    t_mech = (time.perf_counter()-t0)/100*1000

    alphas_vals = [(torch.tanh(a_s)*MA).item() for a_s,_,_ in alphas]

    return {
        'bl_clean': a_bl, 'mech_clean': a_mech, 'd_clean': a_mech-a_bl,
        'bl_n02': a_bl_n2, 'mech_n02': a_mech_n2, 'd_n02': a_mech_n2-a_bl_n2,
        'bl_n03': a_bl_n3, 'mech_n03': a_mech_n3, 'd_n03': a_mech_n3-a_bl_n3,
        't_bl': t_bl, 't_mech': t_mech, 'overhead': (t_mech-t_bl)/t_bl*100,
        'alpha_shfl': alphas_vals,
    }

print("=" * 80)
print("z2455: SCALING TEST — depth × width × dataset × noise")
print("=" * 80)

configs = [
    ('mnist', 1, 128), ('mnist', 2, 128), ('mnist', 4, 128),
    ('mnist', 2, 64), ('mnist', 2, 256),
    ('fashion', 1, 128), ('fashion', 2, 128), ('fashion', 4, 128),
]

print(f"\n{'Dataset':>8} {'D':>2} {'W':>4} | {'Clean':>12} {'σ=0.2':>14} {'σ=0.3':>14} | {'Time':>14} {'OH':>6}")
print(f"{'':>8} {'':>2} {'':>4} | {'BL→Mech Δ':>12} {'BL→Mech Δ':>14} {'BL→Mech Δ':>14} | {'BL→Mech':>14} {'':>6}")
print("-" * 85)

all_results = {}
for ds, depth, width in configs:
    r = train_and_eval(ds, depth, width)
    key = f"{ds}_d{depth}_w{width}"
    all_results[key] = r

    m_clean = "★" if r['d_clean'] > -0.5 else ""
    m_n02 = "★" if r['d_n02'] > 1 else ""
    m_n03 = "★" if r['d_n03'] > 1 else ""

    print(f"{ds:>8} {depth:>2} {width:>4} | "
          f"{r['bl_clean']:>5.1f}→{r['mech_clean']:>5.1f}{r['d_clean']:>+5.1f}{m_clean} | "
          f"{r['bl_n02']:>5.1f}→{r['mech_n02']:>5.1f}{r['d_n02']:>+5.1f}{m_n02} | "
          f"{r['bl_n03']:>5.1f}→{r['mech_n03']:>5.1f}{r['d_n03']:>+5.1f}{m_n03} | "
          f"{r['t_bl']:>5.2f}→{r['t_mech']:>5.2f}ms {r['overhead']:>+5.1f}%")

# Summary
print(f"\n{'='*80}")
print("SCALING SUMMARY")
print(f"{'='*80}")

d_cleans = [r['d_clean'] for r in all_results.values()]
d_n02s = [r['d_n02'] for r in all_results.values()]
d_n03s = [r['d_n03'] for r in all_results.values()]
overheads = [r['overhead'] for r in all_results.values()]

print(f"  Clean accuracy Δ: mean={np.mean(d_cleans):+.2f}pp (range {min(d_cleans):+.1f} to {max(d_cleans):+.1f})")
print(f"  σ=0.2 robustness Δ: mean={np.mean(d_n02s):+.2f}pp (range {min(d_n02s):+.1f} to {max(d_n02s):+.1f})")
print(f"  σ=0.3 robustness Δ: mean={np.mean(d_n03s):+.2f}pp (range {min(d_n03s):+.1f} to {max(d_n03s):+.1f})")
print(f"  Overhead: mean={np.mean(overheads):+.1f}% (range {min(overheads):+.1f} to {max(overheads):+.1f})")

n_robust = sum(1 for d in d_n03s if d > 0)
print(f"\n  Robustness positive in {n_robust}/{len(d_n03s)} configs at σ=0.3")
if n_robust == len(d_n03s):
    print(f"  >>> ROBUSTNESS SCALES ACROSS ALL CONFIGS <<<")

with open(f'{base}/results/z2455_scale_test.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to results/z2455_scale_test.json")
