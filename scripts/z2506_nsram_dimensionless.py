#!/usr/bin/env python3
"""z2506_nsram_dimensionless.py — Dimensionless NS-RAM SNN Reservoir

After 5 iterations (z2501-z2505) fighting with physical units, the lesson:
use dimensionless LIF like Brian2 actually does in slide 23.

Physics enters through:
  1. Per-neuron parameter heterogeneity (slide 16: die-to-die)
  2. Exponential I-V nonlinearity (avalanche-inspired activation)
  3. Charge trapping threshold modulation (SRH dynamics)
  4. Dale's law excitatory/inhibitory balance
  5. Inter-neuron synapses (what FPGA lacks!)
  6. Stochastic noise (SDE)

Dimensionless LIF:
  dv_i/dt = -v_i/τ_i + I_bg_i + Σ_j w_ij s_j + w_in_i u(t) + σ ξ(t)
  if v_i > θ_i: spike, v_i → v_reset, enter refractory
  ds_j/dt = -s_j/τ_syn + δ(spike_j)
  dQ_i/dt = k_cap(1-Q) rate_i - k_em Q
  θ_i(t) = θ_base_i - α Q_i     (charge trap modulates threshold)

Key: I_bg ≈ 0.95 × θ puts neurons NEAR threshold.
Input ±0.5 × W_in tips them over or under → rate coding.
"""

import torch
import numpy as np
import time, json, os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == 'cuda' else ""))


class NSRAMSNNReservoir:
    """Dimensionless spiking neural network reservoir with NS-RAM physics."""

    def __init__(self, N=128, n_inputs=1, connectivity='sparse',
                 spectral_radius=0.9, exc_frac=0.8, seed=42,
                 variability=0.10, bg_frac=0.95):
        self.N = N
        self.seed = seed
        rng = np.random.RandomState(seed)

        def var(base, frac=variability):
            return np.clip(base*(1+frac*rng.randn(N)), base*0.5, base*2.0).astype(np.float32)

        # ─── Dimensionless LIF ───
        self.tau_mem = torch.tensor(var(1.0, 0.15), device=DEVICE)   # ~1 timestep
        self.theta = torch.tensor(var(1.0, 0.05), device=DEVICE)     # Threshold ≈ 1.0
        self.v_reset = 0.0
        self.tau_ref = torch.tensor(var(0.05, 0.10), device=DEVICE)  # ~5% of τ

        # Background: I_bg ≈ bg_frac × θ (puts neurons NEAR threshold)
        self.I_bg = torch.tensor(
            var(bg_frac, 0.10) * self.theta.cpu().numpy(), device=DEVICE)

        # ─── Input weights ───
        self.W_in = torch.tensor(
            rng.randn(N, n_inputs).astype(np.float32) * 0.3, device=DEVICE)

        # ─── Recurrent weights (Dale's law) ───
        N_exc = int(N * exc_frac)
        nsign = np.ones(N, dtype=np.float32)
        nsign[N_exc:] = -1.0

        if connectivity == 'sparse':
            mask = rng.rand(N, N) < 0.15
            W = rng.randn(N, N).astype(np.float32) * mask
        elif connectivity == 'small_world':
            W = np.zeros((N, N), dtype=np.float32)
            for i in range(N):
                for k in [1, 2, 3, 4]:
                    W[i, (i+k)%N] = rng.randn() * 0.5
                    W[(i+k)%N, i] = rng.randn() * 0.5
                if rng.rand() < 0.10:
                    W[i, rng.randint(N)] = rng.randn()
        else:
            W = (rng.randn(N, N) / np.sqrt(N)).astype(np.float32)

        np.fill_diagonal(W, 0)
        W = np.abs(W) * nsign[:, None]
        eigs = np.abs(np.linalg.eigvals(W))
        if eigs.max() > 0:
            W = (W * spectral_radius / eigs.max()).astype(np.float32)
        self.W = torch.tensor(W, device=DEVICE)

        # ─── Synaptic ───
        self.tau_syn = torch.tensor(var(0.5, 0.20), device=DEVICE)  # Half-step decay
        self.syn_scale = 0.3  # Synaptic weight scaling

        # ─── Charge trapping (SRH, slide 17) ───
        # VG2-dependent: some neurons trap more than others
        vg2 = 0.35 + 0.12 * rng.rand(N).astype(np.float32)
        k_cap = (100.0 / (1.0 + np.exp((vg2 - 0.40) / 0.05))).astype(np.float32)
        self.k_cap = torch.tensor(k_cap, device=DEVICE)
        self.k_em = 50.0
        self.trap_shift = 0.2  # Max threshold shift from trapping

        # ─── Avalanche-inspired nonlinearity ───
        # Instead of linear LIF, use exponential integration near threshold
        # This matches the Chynoweth exponential I-V curve
        self.delta_T = torch.tensor(var(0.1, 0.15), device=DEVICE)  # Sharpness

    @torch.no_grad()
    def run(self, inputs_np, dt=1.0, noise_sigma=0.05):
        """Run reservoir. dt=1.0 means one timestep per input sample."""
        if inputs_np.ndim == 1:
            inputs_np = inputs_np[:, None]
        T = len(inputs_np)
        N = self.N
        inputs = torch.tensor(inputs_np, dtype=torch.float32, device=DEVICE)

        # State
        Vm = torch.zeros(N, device=DEVICE)
        syn = torch.zeros(N, device=DEVICE)
        Q = torch.zeros(N, device=DEVICE)
        refrac = torch.zeros(N, device=DEVICE)
        rate_est = torch.zeros(N, device=DEVICE)

        # Traces for richer readout
        fast_trace = torch.zeros(N, device=DEVICE)
        slow_trace = torch.zeros(N, device=DEVICE)

        states = torch.zeros(N, T, device=DEVICE)
        spike_out = torch.zeros(N, T, device=DEVICE)

        sqrt_dt = float(np.sqrt(dt))

        for t in range(T):
            u = inputs[t]

            # Input
            I_in = self.W_in @ u

            # Synaptic recurrence
            I_syn = self.syn_scale * (self.W.T @ syn)

            # Charge trap threshold modulation
            dQ = self.k_cap * (1.0 - Q) * rate_est - self.k_em * Q
            Q = torch.clamp(Q + dQ * dt * 0.01, 0, 1)
            theta_eff = torch.clamp(self.theta - Q * self.trap_shift, min=0.1)

            # Exponential LIF dynamics (AdEx-like, matches avalanche nonlinearity)
            # dv/dt = -v/τ + I_bg + I_in + I_syn + ΔT × exp((v - θ)/ΔT) [near threshold]
            active = (refrac <= 0).float()
            leak = -Vm / self.tau_mem
            exp_term = self.delta_T * torch.exp(
                torch.clamp((Vm - theta_eff) / self.delta_T, -10, 5))
            drive = self.I_bg + I_in + I_syn + exp_term
            noise = noise_sigma * sqrt_dt * torch.randn(N, device=DEVICE)

            Vm = Vm + active * (leak + drive) * dt + active * noise
            Vm = torch.clamp(Vm, -2.0, 5.0)

            # Spike
            spiked = (Vm >= theta_eff) & (refrac <= 0)
            if spiked.any():
                Vm[spiked] = self.v_reset
                refrac[spiked] = self.tau_ref[spiked]
                syn[spiked] = syn[spiked] + 1.0
                rate_est[spiked] = rate_est[spiked] + 5.0
                spike_out[spiked, t] = 1.0

            # Decay
            syn = syn * torch.exp(-dt / self.tau_syn)
            rate_est = rate_est * 0.95  # Smooth decay
            refrac = torch.clamp(refrac - dt, min=0)

            # Multi-timescale traces
            fast_trace = 0.8 * fast_trace + 0.2 * Vm
            slow_trace = 0.98 * slow_trace + 0.02 * Vm

            # State: combine membrane, spikes, traces, trap charge
            states[:, t] = Vm + spike_out[:, t] + 0.3 * fast_trace + 0.1 * slow_trace + 0.2 * Q

        return states.cpu().numpy(), spike_out.cpu().numpy()


class SoftESN:
    """Software ESN baseline (properly tuned, z2254j parameters)."""
    N = 128
    def __init__(self, N=128, sr=1.05, temp=0.65, seed=42):
        self.N = N; self.seed = seed; rng=np.random.RandomState(seed)
        W=rng.randn(N,N).astype(np.float32)/np.sqrt(N); np.fill_diagonal(W,0)
        e=np.abs(np.linalg.eigvals(W))
        self.W=torch.tensor((W*sr/e.max()).astype(np.float32),device=DEVICE)
        self.Win=torch.tensor(rng.randn(N,1).astype(np.float32)*0.5,device=DEVICE)
        self.t=temp
    @torch.no_grad()
    def run(self, u_np, **kw):
        u=torch.tensor(u_np[:,None] if u_np.ndim==1 else u_np,dtype=torch.float32,device=DEVICE)
        T=len(u); S=torch.zeros(self.N,T,device=DEVICE)
        v=torch.zeros(self.N,device=DEVICE); h=torch.zeros(self.N,device=DEVICE)
        s=torch.zeros(self.N,device=DEVICE)
        for t in range(T):
            pre=0.9*v+self.Win@u[t]+self.W@v
            v=torch.tanh(pre/self.t); h=0.93*h+0.07*v; s=0.99*s+0.01*v
            S[:,t]=v+0.3*h+0.1*s
        return S.cpu().numpy(), np.zeros((self.N,T))


# ═══════════════════════════════════════════════════════════════════════
# BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════

def ridge(X, y, alpha=1.0):
    return np.linalg.solve(X.T@X+alpha*np.eye(X.shape[1]), X.T@y)

def ev_xor(S,u,wo,tau):
    T=S.shape[1]; sp=wo+(T-wo)//2; X=S[:,wo+tau:].T
    y=((u[wo+tau:]>0)!=(u[wo:T-tau]>0)).astype(float); s=sp-wo-tau
    if s<20 or len(y)-s<20: return 0.5
    w=ridge(X[:s],y[:s]); a=((X[s:]@w>0.5)==(y[s:]>0.5)).mean(); return max(a,1-a)

def ev_mc(S,u,wo,md=15):
    T=S.shape[1]; sp=wo+(T-wo)//2; mc=0
    for d in range(1,md+1):
        X=S[:,wo+d:].T; y=u[wo:T-d]; s=sp-wo-d
        if s<20 or len(y)-s<20: continue
        w=ridge(X[:s],y[:s]); p=X[s:]@w; yt=y[s:]
        if np.std(yt)<1e-10 or np.std(p)<1e-10: continue
        mc+=np.corrcoef(p,yt)[0,1]**2
    return mc

def ev_narma(S,u,wo,order=5):
    T=min(S.shape[1],len(u)); y=np.zeros(T); uu=(u[:T]+1)/2*0.5
    for t in range(order,T):
        y[t]=0.3*y[t-1]+0.05*y[t-1]*np.sum(y[t-order:t])+1.5*uu[t-1]*uu[t-order]+0.1
        y[t]=np.tanh(y[t])
    sp=wo+(T-wo)//2; X=S[:,wo:T].T; yt=y[wo:T]; s=sp-wo
    if s<20 or len(yt)-s<20: return 0
    w=ridge(X[:s],yt[:s]); p=X[s:]@w; y2=yt[s:]
    r=np.sum((y2-p)**2); tt=np.sum((y2-y2.mean())**2)
    return max(0,1-r/tt) if tt>0 else 0

def ev_wave(S,u,wo,nc=4):
    T=S.shape[1]; sp=wo+(T-wo)//2
    b=np.linspace(-1,1,nc+1); l=np.digitize(u[:T],b[1:-1])
    X=S[:,wo:T].T; yl=l[wo:T]; s=sp-wo
    P=np.zeros((T-wo-s,nc))
    for c in range(nc): w=ridge(X[:s],(yl[:s]==c).astype(float)); P[:,c]=X[s:]@w
    return (np.argmax(P,1)==yl[s:]).mean()


def run_bench(name, res, inputs, wo=400, n_reps=5, **kw):
    reps=[]
    for rep in range(n_reps):
        t0=time.time(); S, spk=res.run(inputs, **kw); elapsed=time.time()-t0
        na=int((spk.sum(1)>0).sum()); ts=int(spk.sum()); T=S.shape[1]
        x1=ev_xor(S,inputs,wo,1); x2=ev_xor(S,inputs,wo,2); x5=ev_xor(S,inputs,wo,5)
        mc=ev_mc(S,inputs,wo); narma=ev_narma(S,inputs,wo); w4=ev_wave(S,inputs,wo)
        r={'xor1':x1,'xor2':x2,'xor5':x5,'mc':mc,'narma':narma,'wave4':w4,
           'active':na,'spikes':ts,'time':elapsed}
        reps.append(r)
        print(f"  [{name}] r{rep}: XOR1={x1:.1%} XOR2={x2:.1%} XOR5={x5:.1%} "
              f"MC={mc:.3f} NARMA={narma:.3f} W4={w4:.1%} | "
              f"{na}/{res.N}N, {ts}spk ({elapsed:.1f}s)")
    avg={k:np.mean([r[k] for r in reps]) for k in reps[0]}
    return {'name':name,'avg':avg,'reps':reps}


def main():
    print("="*75)
    print("  z2506: Dimensionless NS-RAM SNN Reservoir")
    print("  AdEx-LIF + synapses + charge trapping + noise")
    print("="*75)

    T=3000; wo=500
    rng=np.random.RandomState(42)
    inputs=rng.uniform(-1,1,T).astype(np.float64)

    ALL={}

    print("\n━━━ Software ESN Baselines ━━━")
    ALL['ESN_128']=run_bench("ESN_128",SoftESN(128),inputs,wo)
    ALL['ESN_64']=run_bench("ESN_64",SoftESN(64),inputs,wo)

    print("\n━━━ NS-RAM SNN Reservoir ━━━")
    configs=[
        # name,N,conn,sr,var,bg_frac,noise
        ("SNN_64_sparse",     64,  'sparse',     0.90, 0.10, 0.95, 0.05),
        ("SNN_128_sparse",    128, 'sparse',     0.90, 0.10, 0.95, 0.05),
        ("SNN_128_sw",        128, 'small_world', 0.90, 0.10, 0.95, 0.05),
        ("SNN_128_sr095",     128, 'sparse',     0.95, 0.10, 0.95, 0.05),
        ("SNN_128_dense",     128, 'dense',      0.90, 0.10, 0.95, 0.05),
        ("SNN_128_bg090",     128, 'sparse',     0.90, 0.10, 0.90, 0.05),
        ("SNN_128_bg099",     128, 'sparse',     0.90, 0.10, 0.99, 0.05),
        ("SNN_128_noise01",   128, 'sparse',     0.90, 0.10, 0.95, 0.01),
        ("SNN_128_noise20",   128, 'sparse',     0.90, 0.10, 0.95, 0.20),
        ("SNN_128_highvar",   128, 'sparse',     0.90, 0.20, 0.95, 0.05),
        ("SNN_256_sparse",    256, 'sparse',     0.90, 0.10, 0.95, 0.05),
    ]

    for name,N,conn,sr,var,bg,noise in configs:
        res=NSRAMSNNReservoir(N=N,connectivity=conn,spectral_radius=sr,
                               seed=42,variability=var,bg_frac=bg)
        ALL[name]=run_bench(name,res,inputs,wo,n_reps=5,noise_sigma=noise)

    print("\n"+"="*115)
    print(f"  {'Config':<22s}  {'XOR-1':>6s}  {'XOR-2':>6s}  {'XOR-5':>6s}  "
          f"{'MC':>6s}  {'NARMA':>6s}  {'W4':>5s}  {'Active':>7s}  {'Spk/N':>6s}")
    print("="*115)
    for name,r in ALL.items():
        a=r['avg']; N_val=int(name.split('_')[1]) if '_' in name else 128
        spk_per_n = a['spikes']/N_val if N_val > 0 else 0
        print(f"  {name:<22s}  {a['xor1']:>5.1%}  {a['xor2']:>5.1%}  {a['xor5']:>5.1%}  "
              f"{a['mc']:>6.3f}  {a['narma']:>6.3f}  {a['wave4']:>4.1%}  "
              f"{a['active']:>4.0f}/{N_val:<3d}  {spk_per_n:>6.0f}")

    out=os.path.join(os.path.dirname(__file__),'..','results','z2506_nsram_dimensionless.json')
    def ser(o):
        if isinstance(o,(np.integer,)):return int(o)
        if isinstance(o,(np.floating,np.float64)):return float(o)
        return o
    with open(out,'w') as f:
        json.dump({k:{'avg':{kk:ser(vv) for kk,vv in v['avg'].items()},
                       'reps':[{kk:ser(vv) for kk,vv in rep.items()} for rep in v['reps']]}
                   for k,v in ALL.items()},f,indent=2)
    print(f"\nSaved: {out}")


if __name__=='__main__':
    main()
