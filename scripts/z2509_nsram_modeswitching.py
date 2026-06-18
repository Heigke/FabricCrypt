#!/usr/bin/env python3
"""z2509_nsram_modeswitching.py — NS-RAM Mode-Switching Reservoir

THE UNIQUE NS-RAM INNOVATION: Each cell dynamically switches between:
  - NEURON MODE (low VG2 resistance → avalanche spiking)
  - SYNAPSE MODE (high VG2 resistance → charge trapping → weight modification)

No other neuromorphic hardware does this. A cell that just spiked can
temporarily enter synapse mode, trapping charge that modifies its threshold
for future spikes. This creates:
  1. Activity-dependent plasticity (STDP-like, but from device physics)
  2. Self-organized criticality (trapping creates negative feedback on rate)
  3. Heterosynaptic metaplasticity (threshold shifts affect ALL inputs)

New mechanisms beyond z2508:
  A. Dynamic mode switching: VG2 controlled by network activity
  B. Synaptic weight modulation: trapped charge scales outgoing weights
  C. Homeostatic regulation: population rate controls VG2 globally
  D. Burst detection: consecutive spikes trigger mode switch
  E. Comparison against published LSM/SNN baselines

Physics:
  Mode switch: when neuron fires burst (≥2 spikes in 5 steps) → enter synapse mode
  In synapse mode: dQ/dt = k_cap × (1-Q) × input_drive - k_em × Q
  Q modulates: (1) threshold: θ_eff = θ - α_θ × Q
               (2) outgoing weights: W_eff = W × (1 + α_w × Q)
               (3) leak: τ_eff = τ × (1 + α_τ × Q)
  After τ_syn_mode steps → return to neuron mode

This is closer to the real 2T NS-RAM cell where VG2 (second transistor)
controls the operating regime.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time, json, os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'z2509_plots')
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Device: {DEVICE}")


class NSRAMModeSwitchReservoir:
    """NS-RAM reservoir with dynamic neuron↔synapse mode switching."""

    def __init__(self, N=128, n_inputs=1, connectivity='sparse',
                 spectral_radius=0.90, exc_frac=0.80, seed=42,
                 variability=0.10, bg_frac=0.95,
                 # Mode switching parameters
                 burst_threshold=2,      # spikes in window → switch to synapse mode
                 burst_window=5,         # steps to count burst
                 synapse_duration=10,    # steps in synapse mode
                 alpha_theta=0.15,       # Q → threshold modulation
                 alpha_weight=0.30,      # Q → outgoing weight boost
                 alpha_tau=0.20,         # Q → leak time constant boost
                 k_cap=5.0,             # charge capture rate in synapse mode
                 k_em=0.5,              # charge emission rate (slower = longer memory)
                 # Homeostatic
                 target_rate=0.3,        # target population spike fraction
                 homeo_speed=0.001,      # homeostatic VG2 adaptation speed
                 ):
        self.N = N; self.seed = seed
        self.burst_threshold = burst_threshold
        self.burst_window = burst_window
        self.synapse_duration = synapse_duration
        self.alpha_theta = alpha_theta
        self.alpha_weight = alpha_weight
        self.alpha_tau = alpha_tau
        self.target_rate = target_rate
        self.homeo_speed = homeo_speed

        rng = np.random.RandomState(seed)
        def var(base, frac=variability):
            return np.clip(base*(1+frac*rng.randn(N)), base*0.3, base*3.0).astype(np.float32)

        self.tau_mem_base = torch.tensor(var(1.0, 0.15), device=DEVICE)
        self.theta_base = torch.tensor(var(1.0, 0.05), device=DEVICE)
        self.tau_ref = torch.tensor(var(0.05, 0.10), device=DEVICE)
        self.I_bg_base = torch.tensor(var(bg_frac, 0.10) * self.theta_base.cpu().numpy(), device=DEVICE)
        self.W_in = torch.tensor(rng.randn(N, n_inputs).astype(np.float32) * 0.3, device=DEVICE)
        self.delta_T = torch.tensor(var(0.10, 0.15), device=DEVICE)

        N_exc = int(N * exc_frac)
        nsign = np.ones(N, dtype=np.float32); nsign[N_exc:] = -1.0
        self.neuron_type = nsign

        if connectivity == 'sparse':
            mask = rng.rand(N, N) < 0.15
            W = rng.randn(N, N).astype(np.float32) * mask
        elif connectivity == 'small_world':
            W = np.zeros((N, N), dtype=np.float32)
            for i in range(N):
                for k in [1,2,3,4]:
                    W[i,(i+k)%N] = rng.randn()*0.5; W[(i+k)%N,i] = rng.randn()*0.5
                if rng.rand() < 0.10: W[i, rng.randint(N)] = rng.randn()
        else:
            W = (rng.randn(N, N) / np.sqrt(N)).astype(np.float32)
        np.fill_diagonal(W, 0)
        W = np.abs(W) * nsign[:, None]
        eigs = np.abs(np.linalg.eigvals(W))
        if eigs.max() > 0: W = (W * spectral_radius / eigs.max()).astype(np.float32)
        self.W_base = torch.tensor(W, device=DEVICE)
        self.tau_syn_t = torch.tensor(var(0.50, 0.20), device=DEVICE)
        self.syn_scale = 0.30

        self.k_cap = k_cap
        self.k_em = k_em

    @torch.no_grad()
    def run(self, inputs_np, noise_sigma=0.01):
        if inputs_np.ndim == 1: inputs_np = inputs_np[:, None]
        T = len(inputs_np); N = self.N
        inputs = torch.tensor(inputs_np, dtype=torch.float32, device=DEVICE)

        # State
        Vm = torch.zeros(N, device=DEVICE)
        syn = torch.zeros(N, device=DEVICE)
        Q = torch.zeros(N, device=DEVICE)           # Trapped charge
        refrac = torch.zeros(N, device=DEVICE)
        mode = torch.zeros(N, device=DEVICE)         # 0=neuron, 1=synapse
        mode_timer = torch.zeros(N, device=DEVICE)   # Countdown in synapse mode
        spike_history = torch.zeros(N, self.burst_window, device=DEVICE)
        vg2_adapt = torch.zeros(N, device=DEVICE)    # Homeostatic VG2 shift
        ft = torch.zeros(N, device=DEVICE)
        st = torch.zeros(N, device=DEVICE)

        # Recording
        Vm_all = torch.zeros(N, T, device=DEVICE)
        spk_all = torch.zeros(N, T, device=DEVICE)
        Q_all = torch.zeros(N, T, device=DEVICE)
        mode_all = torch.zeros(N, T, device=DEVICE)
        states = torch.zeros(N, T, device=DEVICE)

        pop_rate_ema = 0.3  # Running estimate of population rate

        for t in range(T):
            u = inputs[t]
            I_in = self.W_in @ u

            # ── Mode-dependent dynamics ──
            is_neuron = (mode < 0.5)
            is_synapse = ~is_neuron

            # Charge trapping (SYNAPSE MODE ONLY)
            # In synapse mode, input drive accumulates charge instead of membrane
            input_drive = torch.abs(I_in) + torch.abs(self.syn_scale * (self.W_base.T @ syn))
            dQ_cap = self.k_cap * (1.0 - Q) * input_drive * is_synapse.float()
            dQ_em = self.k_em * Q  # Emission always active (detrapping)
            Q = torch.clamp(Q + (dQ_cap - dQ_em) * 0.01, 0, 1)

            # Q modulates threshold, weights, and leak
            theta_eff = torch.clamp(self.theta_base - self.alpha_theta * Q + vg2_adapt, min=0.1)
            tau_eff = self.tau_mem_base * (1.0 + self.alpha_tau * Q)
            # Weight modulation: neurons with more trapped charge have stronger outputs
            W_scale = 1.0 + self.alpha_weight * Q
            W_eff = self.W_base * W_scale.unsqueeze(1)  # Scale rows (outgoing)

            # Synaptic current with modulated weights
            I_syn = self.syn_scale * (W_eff.T @ syn)

            # ── Neuron dynamics (NEURON MODE ONLY) ──
            active = (refrac <= 0).float() * is_neuron.float()
            leak = -Vm / tau_eff
            exp_term = self.delta_T * torch.exp(torch.clamp((Vm - theta_eff) / self.delta_T, -10, 5))
            drive = self.I_bg_base + I_in + I_syn + exp_term
            noise = noise_sigma * torch.randn(N, device=DEVICE)
            Vm = Vm + active * (leak + drive) + active * noise

            # In synapse mode: membrane decays to rest (neuron is "off")
            Vm = Vm * (1.0 - 0.5 * is_synapse.float())  # Decay toward 0
            Vm = torch.clamp(Vm, -2, 5)

            # ── Spike detection (neuron mode only) ──
            spiked = (Vm >= theta_eff) & (refrac <= 0) & is_neuron
            if spiked.any():
                Vm[spiked] = 0
                refrac[spiked] = self.tau_ref[spiked]
                syn[spiked] += 1
                spk_all[spiked, t] = 1

            # ── Burst detection → mode switching ──
            # Shift spike history and add current spikes
            spike_history = torch.roll(spike_history, -1, dims=1)
            spike_history[:, -1] = spiked.float()
            burst_count = spike_history.sum(dim=1)

            # Switch to synapse mode if burst detected
            new_synapse = is_neuron & (burst_count >= self.burst_threshold)
            if new_synapse.any():
                mode[new_synapse] = 1.0
                mode_timer[new_synapse] = self.synapse_duration

            # Count down synapse mode timer
            mode_timer = torch.clamp(mode_timer - 1, min=0)
            back_to_neuron = is_synapse & (mode_timer <= 0)
            if back_to_neuron.any():
                mode[back_to_neuron] = 0.0

            # ── Homeostatic regulation ──
            # Global population rate drives VG2 adaptation
            current_rate = spiked.float().mean().item()
            pop_rate_ema = 0.99 * pop_rate_ema + 0.01 * current_rate
            rate_error = pop_rate_ema - self.target_rate
            # If rate too high → increase VG2 (raise threshold) → less spiking
            # If rate too low → decrease VG2 (lower threshold) → more spiking
            vg2_adapt = vg2_adapt + self.homeo_speed * rate_error

            # Decay
            syn *= torch.exp(-1 / self.tau_syn_t)
            refrac = torch.clamp(refrac - 1, min=0)
            ft = 0.8*ft + 0.2*Vm; st = 0.98*st + 0.02*Vm

            # Record
            Vm_all[:,t] = Vm; Q_all[:,t] = Q; mode_all[:,t] = mode
            # State includes mode information for readout
            states[:,t] = (Vm + spk_all[:,t] + 0.3*ft + 0.1*st
                          + 0.2*Q + 0.1*mode)  # Mode as feature

        return {k: v.cpu().numpy() for k, v in
                {'states': states, 'spikes': spk_all, 'Vm': Vm_all,
                 'Q': Q_all, 'mode': mode_all}.items()}


class StandardLIF:
    """Standard LIF reservoir (no NS-RAM features) for fair comparison."""
    def __init__(self, N=128, n_inputs=1, spectral_radius=0.90, seed=42):
        self.N = N; rng = np.random.RandomState(seed)
        self.tau = torch.tensor(np.clip(1.0+0.15*rng.randn(N),0.3,3).astype(np.float32), device=DEVICE)
        self.theta = torch.tensor(np.clip(1.0+0.05*rng.randn(N),0.5,2).astype(np.float32), device=DEVICE)
        self.tau_ref = torch.tensor(np.clip(0.05+0.01*rng.randn(N),0.01,0.2).astype(np.float32), device=DEVICE)
        self.I_bg = 0.95 * self.theta
        self.W_in = torch.tensor(rng.randn(N,n_inputs).astype(np.float32)*0.3, device=DEVICE)
        mask = rng.rand(N,N) < 0.15
        W = rng.randn(N,N).astype(np.float32) * mask; np.fill_diagonal(W,0)
        N_exc = int(N*0.8); ns = np.ones(N,dtype=np.float32); ns[N_exc:]=-1
        W = np.abs(W)*ns[:,None]
        eigs = np.abs(np.linalg.eigvals(W))
        if eigs.max()>0: W=(W*spectral_radius/eigs.max()).astype(np.float32)
        self.W = torch.tensor(W, device=DEVICE)
        self.tau_syn_t = torch.tensor(np.clip(0.5+0.1*rng.randn(N),0.1,2).astype(np.float32), device=DEVICE)

    @torch.no_grad()
    def run(self, inputs_np, noise_sigma=0.01):
        if inputs_np.ndim==1: inputs_np=inputs_np[:,None]
        T=len(inputs_np); N=self.N
        inputs=torch.tensor(inputs_np,dtype=torch.float32,device=DEVICE)
        Vm=torch.zeros(N,device=DEVICE); syn=torch.zeros(N,device=DEVICE)
        refrac=torch.zeros(N,device=DEVICE)
        ft=torch.zeros(N,device=DEVICE); st=torch.zeros(N,device=DEVICE)
        states=torch.zeros(N,T,device=DEVICE); spk=torch.zeros(N,T,device=DEVICE)
        for t in range(T):
            u=inputs[t]; I_in=self.W_in@u; I_syn=0.3*(self.W.T@syn)
            active=(refrac<=0).float()
            Vm = Vm + active*(-Vm/self.tau + self.I_bg + I_in + I_syn) + active*noise_sigma*torch.randn(N,device=DEVICE)
            Vm=torch.clamp(Vm,-2,5)
            spiked=(Vm>=self.theta)&(refrac<=0)
            if spiked.any():
                Vm[spiked]=0; refrac[spiked]=self.tau_ref[spiked]
                syn[spiked]+=1; spk[spiked,t]=1
            syn*=torch.exp(-1/self.tau_syn_t); refrac=torch.clamp(refrac-1,min=0)
            ft=0.8*ft+0.2*Vm; st=0.98*st+0.02*Vm
            states[:,t]=Vm+spk[:,t]+0.3*ft+0.1*st
        return {'states':states.cpu().numpy(),'spikes':spk.cpu().numpy(),
                'Vm':torch.zeros(1).numpy(),'Q':torch.zeros(1).numpy(),
                'mode':torch.zeros(1).numpy()}


class ContinuousESN:
    """Standard continuous ESN (tanh, no spikes) — literature baseline."""
    def __init__(self, N=128, n_inputs=1, spectral_radius=1.05, temp=0.65, seed=42):
        self.N=N; rng=np.random.RandomState(seed)
        W=rng.randn(N,N).astype(np.float32)/np.sqrt(N); np.fill_diagonal(W,0)
        e=np.abs(np.linalg.eigvals(W))
        self.W=torch.tensor((W*spectral_radius/e.max()).astype(np.float32),device=DEVICE)
        self.Win=torch.tensor(rng.randn(N,n_inputs).astype(np.float32)*0.5,device=DEVICE)
        self.t=temp

    @torch.no_grad()
    def run(self, u_np, noise_sigma=0.0):
        u=torch.tensor(u_np[:,None] if u_np.ndim==1 else u_np,dtype=torch.float32,device=DEVICE)
        T=len(u); S=torch.zeros(self.N,T,device=DEVICE)
        v=torch.zeros(self.N,device=DEVICE); h=torch.zeros(self.N,device=DEVICE)
        s=torch.zeros(self.N,device=DEVICE)
        for t in range(T):
            pre=0.9*v+self.Win@u[t]+self.W@v
            v=torch.tanh(pre/self.t); h=0.93*h+0.07*v; s=0.99*s+0.01*v
            S[:,t]=v+0.3*h+0.1*s
        return {'states':S.cpu().numpy(),'spikes':np.zeros((self.N,T)),
                'Vm':np.zeros(1),'Q':np.zeros(1),'mode':np.zeros(1)}


# ── Benchmarks ──
def ridge(X,y,a=1.0): return np.linalg.solve(X.T@X+a*np.eye(X.shape[1]),X.T@y)
def ev_xor(S,u,wo,tau):
    T=S.shape[1];sp=wo+(T-wo)//2;X=S[:,wo+tau:].T
    y=((u[wo+tau:]>0)!=(u[wo:T-tau]>0)).astype(float);s=sp-wo-tau
    if s<20 or len(y)-s<20: return .5
    w=ridge(X[:s],y[:s]);a=((X[s:]@w>.5)==(y[s:]>.5)).mean();return max(a,1-a)
def ev_mc(S,u,wo,md=15):
    T=S.shape[1];sp=wo+(T-wo)//2;mc=0
    for d in range(1,md+1):
        X=S[:,wo+d:].T;y=u[wo:T-d];s=sp-wo-d
        if s<20 or len(y)-s<20: continue
        w=ridge(X[:s],y[:s]);p=X[s:]@w;yt=y[s:]
        if np.std(yt)<1e-10 or np.std(p)<1e-10: continue
        mc+=np.corrcoef(p,yt)[0,1]**2
    return mc
def ev_narma(S,u,wo,order=5):
    T=min(S.shape[1],len(u));y=np.zeros(T);uu=(u[:T]+1)/2*0.5
    for t in range(order,T):
        y[t]=.3*y[t-1]+.05*y[t-1]*np.sum(y[t-order:t])+1.5*uu[t-1]*uu[t-order]+.1
        y[t]=np.tanh(y[t])
    sp=wo+(T-wo)//2;X=S[:,wo:T].T;yt=y[wo:T];s=sp-wo
    if s<20 or len(yt)-s<20: return 0
    w=ridge(X[:s],yt[:s]);p=X[s:]@w;y2=yt[s:]
    r=np.sum((y2-p)**2);tt=np.sum((y2-y2.mean())**2)
    return max(0,1-r/tt) if tt>0 else 0
def ev_wave(S,u,wo,nc=4):
    T=S.shape[1];sp=wo+(T-wo)//2;b=np.linspace(-1,1,nc+1);l=np.digitize(u[:T],b[1:-1])
    X=S[:,wo:T].T;yl=l[wo:T];s=sp-wo;P=np.zeros((T-wo-s,nc))
    for c in range(nc): w=ridge(X[:s],(yl[:s]==c).astype(float));P[:,c]=X[s:]@w
    return (np.argmax(P,1)==yl[s:]).mean()


def run_bench(name, res, inputs, wo=500, n_reps=5, noise=0.01):
    reps = []
    for rep in range(n_reps):
        out = res.run(inputs, noise_sigma=noise)
        S = out['states']; spk = out['spikes']
        m = {'xor1':ev_xor(S,inputs,wo,1), 'xor2':ev_xor(S,inputs,wo,2),
             'xor5':ev_xor(S,inputs,wo,5), 'mc':ev_mc(S,inputs,wo),
             'narma':ev_narma(S,inputs,wo), 'wave4':ev_wave(S,inputs,wo),
             'active':int((spk.sum(1)>0).sum()), 'spikes':int(spk.sum())}
        reps.append(m)
    avg = {k: np.mean([r[k] for r in reps]) for k in reps[0]}
    std = {k: np.std([r[k] for r in reps]) for k in reps[0]}
    print(f"  {name:<35s}: XOR1={avg['xor1']:.1%}±{std['xor1']:.1%}  "
          f"MC={avg['mc']:.3f}  NARMA={avg['narma']:.3f}  W4={avg['wave4']:.1%}  "
          f"({avg['active']:.0f}N, {avg['spikes']:.0f}spk)")
    return {'name': name, 'avg': avg, 'std': std, 'reps': reps, 'last_run': out}


def main():
    print("="*70)
    print("  z2509: NS-RAM Mode-Switching Reservoir")
    print("  Dynamic neuron↔synapse switching + homeostatic regulation")
    print("="*70)

    T = 3000; wo = 500
    rng = np.random.RandomState(42)
    inputs = rng.uniform(-1, 1, T).astype(np.float64)

    ALL = {}

    # ── Baselines ──
    print("\n━━━ Baselines ━━━")
    ALL['ESN'] = run_bench("Continuous ESN (tanh, SR=1.05)",
                            ContinuousESN(128, spectral_radius=1.05), inputs, wo)
    ALL['LIF'] = run_bench("Standard LIF (no NS-RAM)",
                            StandardLIF(128), inputs, wo)

    # ── NS-RAM without mode switching (z2506 baseline) ──
    print("\n━━━ NS-RAM without mode switching ━━━")
    ALL['NSRAM_base'] = run_bench("NS-RAM AdEx (no mode switch)",
                                   NSRAMModeSwitchReservoir(128, burst_threshold=999,
                                                             alpha_theta=0, alpha_weight=0,
                                                             alpha_tau=0, homeo_speed=0),
                                   inputs, wo)

    # ── NS-RAM with mode switching ──
    print("\n━━━ NS-RAM with mode switching ━━━")
    ALL['NSRAM_mode'] = run_bench("NS-RAM + mode switching",
                                   NSRAMModeSwitchReservoir(128), inputs, wo)
    ALL['NSRAM_mode_strong'] = run_bench("NS-RAM + strong trapping",
                                          NSRAMModeSwitchReservoir(128, alpha_theta=0.25,
                                                                    alpha_weight=0.50,
                                                                    k_cap=10.0, k_em=0.2),
                                          inputs, wo)
    ALL['NSRAM_mode_homeo'] = run_bench("NS-RAM + homeostatic",
                                         NSRAMModeSwitchReservoir(128, homeo_speed=0.005,
                                                                   target_rate=0.25),
                                         inputs, wo)
    ALL['NSRAM_mode_fast'] = run_bench("NS-RAM + fast switching",
                                        NSRAMModeSwitchReservoir(128, burst_threshold=1,
                                                                  synapse_duration=3),
                                        inputs, wo)
    ALL['NSRAM_mode_slow'] = run_bench("NS-RAM + slow switching",
                                        NSRAMModeSwitchReservoir(128, burst_threshold=3,
                                                                  synapse_duration=30),
                                        inputs, wo)

    # ── Scaling ──
    print("\n━━━ Mode-switching scaling ━━━")
    for N_val in [32, 64, 128, 256]:
        ALL[f'NSRAM_mode_{N_val}'] = run_bench(
            f"NS-RAM mode-switch N={N_val}",
            NSRAMModeSwitchReservoir(N_val), inputs, wo, n_reps=3)

    # ═══ PLOT: Comparison bar chart ═══
    print("\n━━━ Generating plots ━━━")

    # Main comparison
    compare_keys = ['ESN', 'LIF', 'NSRAM_base', 'NSRAM_mode', 'NSRAM_mode_strong']
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    x = np.arange(len(compare_keys))
    labels = [ALL[k]['name'][:25] for k in compare_keys]

    for ax, metric, title in [
        (axes[0,0], 'xor1', 'XOR τ=1'),
        (axes[0,1], 'mc', 'Memory Capacity'),
        (axes[1,0], 'narma', 'NARMA-5 R²'),
        (axes[1,1], 'wave4', '4-Class Waveform'),
    ]:
        vals = [ALL[k]['avg'][metric] for k in compare_keys]
        errs = [ALL[k]['std'][metric] for k in compare_keys]
        colors = ['#9E9E9E', '#2196F3', '#FF9800', '#4CAF50', '#E91E63']
        ax.bar(x, vals, 0.6, yerr=errs, color=colors, edgecolor='black',
               linewidth=0.5, capsize=3)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7, rotation=15)
        ax.set_ylabel(title, fontsize=10)
        for i, v in enumerate(vals):
            fmt = f'{v:.1%}' if metric in ('xor1','wave4') else f'{v:.3f}'
            ax.text(i, v + errs[i] + 0.01, fmt, ha='center', fontsize=7, fontweight='bold')
    fig.suptitle('NS-RAM Mode-Switching vs Baselines', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig1_comparison.png'), dpi=200)
    plt.close()
    print("  Saved fig1_comparison.png")

    # ═══ PLOT: Mode switching dynamics ═══
    ms_run = ALL['NSRAM_mode']['last_run']
    spk = ms_run['spikes']; Q = ms_run['Q']; mode_data = ms_run['mode']

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(5, 1, height_ratios=[0.1, 0.3, 0.2, 0.2, 0.2], hspace=0.3)

    t_range = slice(wo, wo+500)
    t_ax = np.arange(500)

    # Input
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(t_ax, inputs[t_range], 'b-', linewidth=0.5)
    ax0.set_ylabel('Input'); ax0.set_xticklabels([])

    # Raster
    ax1 = fig.add_subplot(gs[1])
    st, sn = np.where(spk[:, t_range].T > 0)
    ax1.scatter(st, sn, s=1, c='black', marker='.', rasterized=True)
    ax1.set_ylabel('Neuron #'); ax1.set_ylim(-1, 128); ax1.set_xticklabels([])

    # Mode map (neuron=white, synapse=red)
    ax2 = fig.add_subplot(gs[2])
    ax2.imshow(mode_data[:, t_range], aspect='auto', cmap='Reds', vmin=0, vmax=1,
               interpolation='none')
    ax2.set_ylabel('Mode\n(red=syn)'); ax2.set_xticklabels([])

    # Charge trap
    ax3 = fig.add_subplot(gs[3])
    ax3.imshow(Q[:, t_range], aspect='auto', cmap='viridis', interpolation='none')
    ax3.set_ylabel('Q_trap'); ax3.set_xticklabels([])

    # Population rate + mode fraction
    ax4 = fig.add_subplot(gs[4])
    win = 20
    pop_rate = np.convolve(spk.sum(0), np.ones(win)/win, mode='same')
    mode_frac = np.convolve(mode_data.mean(0), np.ones(win)/win, mode='same')
    ax4.plot(pop_rate[t_range], 'r-', linewidth=0.8, label='Pop spike rate')
    ax4t = ax4.twinx()
    ax4t.plot(mode_frac[t_range], 'b-', linewidth=0.8, label='Synapse fraction')
    ax4.set_ylabel('Spike rate', color='red')
    ax4t.set_ylabel('Syn. frac.', color='blue')
    ax4.set_xlabel('Time step')

    fig.suptitle('NS-RAM Dynamic Mode Switching (Neuron ↔ Synapse)',
                  fontsize=13, fontweight='bold')
    plt.savefig(os.path.join(OUT_DIR, 'fig2_mode_dynamics.png'), dpi=200)
    plt.close()
    print("  Saved fig2_mode_dynamics.png")

    # ═══ PLOT: Scaling ═══
    N_vals = [32, 64, 128, 256]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    xor_vals = [ALL[f'NSRAM_mode_{N}']['avg']['xor1'] for N in N_vals]
    mc_vals = [ALL[f'NSRAM_mode_{N}']['avg']['mc'] for N in N_vals]
    ax1.semilogx(N_vals, [x*100 for x in xor_vals], 'go-', linewidth=2, markersize=8,
                  label='NS-RAM mode-switch')
    ax1.axhline(50, color='gray', linestyle='--', label='Chance')
    ax1.set_xlabel('N neurons'); ax1.set_ylabel('XOR-1 (%)'); ax1.legend()
    ax1.set_title('XOR Scaling with Mode Switching'); ax1.grid(True, alpha=0.3)

    ax2.semilogx(N_vals, mc_vals, 'bo-', linewidth=2, markersize=8)
    ax2.set_xlabel('N neurons'); ax2.set_ylabel('MC')
    ax2.set_title('Memory Capacity Scaling'); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig3_scaling.png'), dpi=200)
    plt.close()
    print("  Saved fig3_scaling.png")

    # ═══ Summary table ═══
    print("\n" + "="*105)
    print(f"  {'Config':<35s}  {'XOR-1':>7s}  {'XOR-2':>7s}  {'MC':>7s}  "
          f"{'NARMA':>7s}  {'Wave4':>7s}  {'Active':>6s}")
    print("="*105)
    for k in ['ESN', 'LIF', 'NSRAM_base', 'NSRAM_mode', 'NSRAM_mode_strong',
              'NSRAM_mode_homeo', 'NSRAM_mode_fast', 'NSRAM_mode_slow']:
        a = ALL[k]['avg']
        print(f"  {ALL[k]['name']:<35s}  {a['xor1']:>6.1%}  {a['xor2']:>6.1%}  "
              f"{a['mc']:>7.3f}  {a['narma']:>7.3f}  {a['wave4']:>6.1%}  {a['active']:>6.0f}")

    # Save
    out_json = os.path.join(OUT_DIR, 'results.json')
    def ser(o):
        if isinstance(o,(np.integer,)):return int(o)
        if isinstance(o,(np.floating,np.float64)):return float(o)
        if isinstance(o, np.ndarray): return None  # Skip large arrays
        return o
    save_data = {}
    for k, v in ALL.items():
        save_data[k] = {'name': v['name'],
                         'avg': {kk:ser(vv) for kk,vv in v['avg'].items()},
                         'std': {kk:ser(vv) for kk,vv in v['std'].items()}}
    with open(out_json, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results: {out_json}")
    print(f"  Plots: {OUT_DIR}/")


if __name__ == '__main__':
    main()
