#!/usr/bin/env python3
"""z2505_nsram_edge_of_chaos.py — NS-RAM at the Edge of Firing

Key insight: NS-RAM computes at the boundary between spiking and silence.
  - Slides show 10⁴× firing range → very sharp cliff
  - At 300kHz ALL neurons fire identically → no info encoding
  - At 0 Hz nothing happens
  - The SWEET SPOT: most neurons near threshold, input tips them over/under

Physics fix: Set BVpar ≈ Vcb_peak so neurons are MARGINAL.
Then input Vg1 modulation (±0.08V → ±0.12V BVpar shift) determines
which neurons fire per Vcb cycle and which don't.

This is the edge-of-chaos / criticality regime that ALL reservoir
computing theory says is optimal.

Also: use Brian2-style LIF (slide 23) as the BASELINE spiking model
instead of avalanche-driven, since that's what actually achieved 72%
MNIST accuracy in the slides.
"""

import torch
import numpy as np
import time, json, os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == 'cuda' else ""))


class NSRAMLIFReservoir:
    """NS-RAM as Brian2-style LIF reservoir (slide 23 parameters).

    Instead of explicitly simulating Vcb self-oscillation + avalanche,
    we model the NET EFFECT: neurons receive input-dependent excitation
    and fire when Vm exceeds threshold. This is what Brian2 does in slide 23,
    and it achieved 72% MNIST accuracy.

    The NS-RAM physics enters through:
    1. Per-neuron heterogeneity (slide 16: die-to-die variability)
    2. Charge trapping modulates threshold (slide 17: SRH)
    3. Temperature dependence of time constants (implicit via variability)
    4. Inter-neuron synapses (what our FPGA lacks!)
    """

    def __init__(self, N=128, n_inputs=1, connectivity='sparse',
                 spectral_radius=0.95, exc_frac=0.8, seed=42,
                 variability=0.10):
        self.N = N
        self.seed = seed
        rng = np.random.RandomState(seed)

        def var(base, frac=variability):
            return np.clip(base * (1 + frac * rng.randn(N)), base*0.3, base*3.0).astype(np.float32)

        # ─── LIF parameters (slide 23: Brian2) ───
        # τ_mem = 10ms (functional timescale, not device-level 1μs)
        # This gives ~100Hz spiking rate — reasonable for RC benchmarks
        self.tau_mem = torch.tensor(var(10e-3, 0.15), device=DEVICE)  # 10ms ± 15%
        self.V_thresh_base = torch.tensor(var(1.0, 0.05), device=DEVICE)
        self.V_reset = 0.0
        self.tau_ref = torch.tensor(var(2e-3, 0.10), device=DEVICE)  # 2ms refractory

        # ─── Input weights (per-neuron, frequency coded) ───
        self.W_in = torch.tensor(
            rng.randn(N, n_inputs).astype(np.float32) * 0.5, device=DEVICE)

        # ─── Recurrent weights (Dale's law) ───
        N_exc = int(N * exc_frac)
        n_sign = np.ones(N, dtype=np.float32)
        n_sign[N_exc:] = -1.0

        if connectivity == 'sparse':
            mask = rng.rand(N, N) < 0.15
            W = rng.randn(N, N).astype(np.float32) * mask
        elif connectivity == 'small_world':
            W = np.zeros((N, N), dtype=np.float32)
            for i in range(N):
                for k in [1, 2, 3, 4]:
                    W[i, (i+k) % N] = rng.randn() * 0.5
                    W[(i+k) % N, i] = rng.randn() * 0.5
                if rng.rand() < 0.10:
                    W[i, rng.randint(N)] = rng.randn()
        else:
            W = (rng.randn(N, N) / np.sqrt(N)).astype(np.float32)

        np.fill_diagonal(W, 0)
        W = np.abs(W) * n_sign[:, None]
        eigs = np.abs(np.linalg.eigvals(W))
        if eigs.max() > 0:
            W = (W * spectral_radius / eigs.max()).astype(np.float32)
        self.W = torch.tensor(W, device=DEVICE)

        # Synaptic dynamics
        self.tau_syn = torch.tensor(var(5e-3, 0.20), device=DEVICE)  # 5ms synaptic τ
        self.syn_strength = 0.3  # mV equivalent per spike

        # Charge trapping (SRH, slide 17)
        k_cap = 500.0 / (1.0 + np.exp((var(0.40, 0.10) - 0.40) / 0.05))
        self.k_cap = torch.tensor(k_cap, device=DEVICE)
        self.k_em = 200.0
        self.Vth_shift_max = 0.3

        # Background current (sets neurons near threshold)
        # This is the NS-RAM analog: Vcb self-oscillation provides background
        # excitation that brings neurons NEAR threshold.
        # Input modulation then tips them over.
        self.I_bg = torch.tensor(var(0.8, 0.15), device=DEVICE)  # ~80% of threshold

    @torch.no_grad()
    def run(self, inputs_np, dt=0.5e-3, noise_sigma=0.1):
        """Euler-Maruyama LIF integration.

        dt=0.5ms, each input sample = 1 step → T steps total.
        This is the timescale of Brian2 simulation (slide 23).
        """
        if inputs_np.ndim == 1:
            inputs_np = inputs_np[:, None]

        T = len(inputs_np)
        N = self.N
        inputs = torch.tensor(inputs_np, dtype=torch.float32, device=DEVICE)

        # State
        Vm = torch.zeros(N, device=DEVICE)
        syn = torch.zeros(N, device=DEVICE)  # Synaptic activation
        Q = torch.zeros(N, device=DEVICE)     # Trapped charge
        refrac = torch.zeros(N, device=DEVICE)
        spike_rate_est = torch.zeros(N, device=DEVICE)

        # Output
        states = torch.zeros(N, T, device=DEVICE)
        spike_out = torch.zeros(N, T, device=DEVICE)

        sqrt_dt = float(np.sqrt(dt))

        for t in range(T):
            u = inputs[t]

            # ─── Input current ───
            I_in = self.W_in @ u

            # ─── Synaptic current (inter-neuron) ───
            I_syn = self.syn_strength * (self.W.T @ syn)

            # ─── Charge trap threshold modulation ───
            dQ = self.k_cap * (1.0 - Q) * spike_rate_est - self.k_em * Q
            Q = torch.clamp(Q + dQ * dt, 0, 1)
            V_th = torch.clamp(self.V_thresh_base - Q * self.Vth_shift_max, min=0.1)

            # ─── LIF dynamics ───
            # dVm/dt = (-Vm/τ_mem + I_bg + I_in + I_syn) + noise
            active = (refrac <= 0).float()
            leak = -Vm / self.tau_mem
            drive = self.I_bg + I_in + I_syn
            noise = noise_sigma * sqrt_dt * torch.randn(N, device=DEVICE)

            Vm = Vm + active * (leak + drive) * dt + active * noise
            Vm = torch.clamp(Vm, -1.0, 3.0)

            # ─── Spike detection ───
            spiked = (Vm >= V_th) & (refrac <= 0)
            n_spiked = spiked.sum().item()

            if n_spiked > 0:
                Vm[spiked] = self.V_reset
                refrac[spiked] = self.tau_ref[spiked]
                syn[spiked] = syn[spiked] + 1.0
                spike_rate_est[spiked] = spike_rate_est[spiked] + 10.0
                spike_out[spiked, t] = 1.0

            # Decay
            syn_decay = torch.exp(-dt / self.tau_syn)
            syn = syn * syn_decay
            spike_rate_est = spike_rate_est * float(np.exp(-dt / 0.1))  # 100ms τ
            refrac = torch.clamp(refrac - dt, min=0)

            # ─── Record state ───
            # Multi-feature: membrane + recent spikes + trap charge
            states[:, t] = Vm + 0.5 * spike_out[:, t] + 0.2 * Q

        return states.cpu().numpy(), spike_out.cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════
# RC BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════

def ridge(X, y, alpha=1.0):
    return np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ y)

def ev_xor(S, u, wo, tau):
    T=S.shape[1]; sp=wo+(T-wo)//2
    X=S[:,wo+tau:].T; y=((u[wo+tau:]>0)!=(u[wo:T-tau]>0)).astype(float)
    s=sp-wo-tau
    if s<20 or len(y)-s<20: return 0.5
    w=ridge(X[:s],y[:s]); a=((X[s:]@w>0.5)==(y[s:]>0.5)).mean()
    return max(a,1-a)

def ev_mc(S, u, wo, md=15):
    T=S.shape[1]; sp=wo+(T-wo)//2; mc=0
    for d in range(1,md+1):
        X=S[:,wo+d:].T; y=u[wo:T-d]; s=sp-wo-d
        if s<20 or len(y)-s<20: continue
        w=ridge(X[:s],y[:s]); p=X[s:]@w; yt=y[s:]
        if np.std(yt)<1e-10 or np.std(p)<1e-10: continue
        mc += np.corrcoef(p,yt)[0,1]**2
    return mc

def ev_narma(S, u, wo, order=5):
    T=min(S.shape[1],len(u)); y=np.zeros(T); uu=(u[:T]+1)/2*0.5
    for t in range(order,T):
        y[t]=0.3*y[t-1]+0.05*y[t-1]*np.sum(y[t-order:t])+1.5*uu[t-1]*uu[t-order]+0.1
        y[t]=np.tanh(y[t])
    sp=wo+(T-wo)//2; X=S[:,wo:T].T; yt=y[wo:T]; s=sp-wo
    if s<20 or len(yt)-s<20: return 0.0
    w=ridge(X[:s],yt[:s]); p=X[s:]@w; y2=yt[s:]
    r=np.sum((y2-p)**2); tt=np.sum((y2-y2.mean())**2)
    return max(0,1-r/tt) if tt>0 else 0

def ev_wave(S, u, wo, nc=4):
    T=S.shape[1]; sp=wo+(T-wo)//2
    b=np.linspace(-1,1,nc+1); l=np.digitize(u[:T],b[1:-1])
    X=S[:,wo:T].T; yl=l[wo:T]; s=sp-wo
    P=np.zeros((T-wo-s,nc))
    for c in range(nc): w=ridge(X[:s],(yl[:s]==c).astype(float)); P[:,c]=X[s:]@w
    return (np.argmax(P,1)==yl[s:]).mean()


def run_bench(name, res, inputs, wo=300, n_reps=5, **kw):
    reps = []
    for rep in range(n_reps):
        t0=time.time()
        S, spk = res.run(inputs, **kw)
        elapsed=time.time()-t0
        na=int((spk.sum(1)>0).sum()); ts=int(spk.sum())
        T=S.shape[1]
        if T > wo+100:
            x1=ev_xor(S,inputs,wo,1); x2=ev_xor(S,inputs,wo,2)
            x5=ev_xor(S,inputs,wo,5); mc=ev_mc(S,inputs,wo)
            narma=ev_narma(S,inputs,wo); w4=ev_wave(S,inputs,wo)
        else:
            x1=x2=x5=mc=narma=w4=0
        r={'xor1':x1,'xor2':x2,'xor5':x5,'mc':mc,'narma':narma,'wave4':w4,
           'active':na,'spikes':ts,'time':elapsed}
        reps.append(r)
        rate = ts/(na*T*0.5e-3) if na>0 and T>0 else 0
        print(f"  [{name}] r{rep}: XOR1={x1:.1%} XOR2={x2:.1%} XOR5={x5:.1%} "
              f"MC={mc:.3f} NARMA={narma:.3f} W4={w4:.1%} | "
              f"{na}N, {ts}spk, ~{rate:.0f}Hz ({elapsed:.1f}s)")
    avg={k:np.mean([r[k] for r in reps]) for k in reps[0]}
    return {'name':name,'avg':avg,'reps':reps}


def main():
    print("="*75)
    print("  z2505: NS-RAM Edge-of-Chaos LIF Reservoir")
    print("  Brian2-style LIF (slide 23) + synapses + charge trapping")
    print("="*75)

    T = 3000; wo = 500
    rng = np.random.RandomState(42)
    inputs = rng.uniform(-1, 1, T).astype(np.float64)

    ALL = {}

    # ─── Software ESN baseline (properly tuned) ───
    print("\n━━━ Software ESN Baseline ━━━")

    class SoftESN:
        N = 128
        def __init__(self, sr=1.05, temp=0.65, seed=42):
            self.seed=seed; rng=np.random.RandomState(seed)
            W=rng.randn(128,128).astype(np.float32)/np.sqrt(128)
            np.fill_diagonal(W,0)
            e=np.abs(np.linalg.eigvals(W))
            self.W=torch.tensor((W*sr/e.max()).astype(np.float32),device=DEVICE)
            self.Win=torch.tensor(rng.randn(128,1).astype(np.float32)*0.5,device=DEVICE)
            self.t=temp
        @torch.no_grad()
        def run(self, u_np, **kw):
            u=torch.tensor(u_np[:,None] if u_np.ndim==1 else u_np,dtype=torch.float32,device=DEVICE)
            T=len(u); S=torch.zeros(128,T,device=DEVICE)
            v=torch.zeros(128,device=DEVICE); h=torch.zeros(128,device=DEVICE)
            s=torch.zeros(128,device=DEVICE)
            for t in range(T):
                pre=0.9*v+self.Win@u[t]+self.W@v
                v=torch.tanh(pre/self.t)
                h=0.93*h+0.07*v; s=0.99*s+0.01*v
                S[:,t]=v+0.3*h+0.1*s
            return S.cpu().numpy(), np.zeros((128,T))

    esn = SoftESN()
    ALL['ESN_128'] = run_bench("ESN_128 (optimal)", esn, inputs, wo)

    # ─── NS-RAM LIF configs ───
    print("\n━━━ NS-RAM LIF Reservoir (edge-of-chaos) ━━━")

    configs = [
        # name, N, conn, sr, var, noise
        ("NSRAM_64_sparse",     64,  'sparse',     0.95, 0.10, 0.10),
        ("NSRAM_128_sparse",    128, 'sparse',     0.95, 0.10, 0.10),
        ("NSRAM_128_sw",        128, 'small_world', 0.95, 0.10, 0.10),
        ("NSRAM_128_sr105",     128, 'sparse',     1.05, 0.10, 0.10),
        ("NSRAM_128_dense",     128, 'dense',      0.95, 0.10, 0.10),
        ("NSRAM_128_highvar",   128, 'sparse',     0.95, 0.20, 0.10),
        ("NSRAM_128_lownoise",  128, 'sparse',     0.95, 0.10, 0.01),
        ("NSRAM_128_highnoise", 128, 'sparse',     0.95, 0.10, 0.30),
        ("NSRAM_256_sparse",    256, 'sparse',     0.95, 0.10, 0.10),
    ]

    for name, N, conn, sr, var, noise in configs:
        res = NSRAMLIFReservoir(N=N, connectivity=conn, spectral_radius=sr,
                                 seed=42, variability=var)
        ALL[name] = run_bench(name, res, inputs, wo, n_reps=5, noise_sigma=noise)

    # ─── Summary ───
    print("\n" + "="*110)
    print(f"  {'Config':<25s}  {'XOR-1':>6s}  {'XOR-2':>6s}  {'XOR-5':>6s}  "
          f"{'MC':>6s}  {'NARMA':>6s}  {'W4':>5s}  {'Active':>6s}  {'Spk/step':>8s}")
    print("="*110)
    for name, r in ALL.items():
        a=r['avg']
        N_val = 128 if '128' in name or 'ESN' in name else (256 if '256' in name else 64)
        spk_per_step = a['spikes'] / 3000 if a['spikes'] > 0 else 0
        print(f"  {name:<25s}  {a['xor1']:>5.1%}  {a['xor2']:>5.1%}  {a['xor5']:>5.1%}  "
              f"{a['mc']:>6.3f}  {a['narma']:>6.3f}  {a['wave4']:>4.1%}  "
              f"{a['active']:>6.0f}  {spk_per_step:>8.1f}")

    out = os.path.join(os.path.dirname(__file__), '..', 'results', 'z2505_nsram_edge_of_chaos.json')
    def ser(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating, np.float64)): return float(o)
        return o
    with open(out, 'w') as f:
        json.dump({k: {'avg':{kk:ser(vv) for kk,vv in v['avg'].items()},
                        'reps':[{kk:ser(vv) for kk,vv in rep.items()} for rep in v['reps']]}
                   for k,v in ALL.items()}, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
