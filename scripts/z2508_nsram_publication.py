#!/usr/bin/env python3
"""z2508_nsram_publication.py — NS-RAM ODE Reservoir: Publication-Quality Results

Generates plots matching Pazos/Lanza slide formats:
  1. Membrane voltage waveforms with spike trains (slide 2 format)
  2. Die-to-die variability grid (slide 16 format)
  3. I-V curve: spike rate vs Vg1 sweep (slide 22: 10^4 range)
  4. Charge trapping dynamics: Q(t) under LTP/LTD (slide 17)
  5. Reservoir computing benchmarks: bar chart + scaling law
  6. Raster plot + population dynamics
  7. Synaptic weight matrix visualization
  8. BVpar(Vg1, T) surface (slide 17 physics)

Physics: AdEx-LIF with Pazos parameters (BVpar=3.5-1.5*Vg, Tbv1=-21.3u/K)
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import time, json, os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'z2508_plots')
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == 'cuda' else ""))


# ═══════════════════════════════════════════════════════════════════════
# NS-RAM RESERVOIR (from z2506, enhanced)
# ═══════════════════════════════════════════════════════════════════════

class NSRAMReservoir:
    def __init__(self, N=128, n_inputs=1, connectivity='sparse',
                 spectral_radius=0.90, exc_frac=0.80, seed=42,
                 variability=0.10, bg_frac=0.95, delta_T=0.10,
                 tau_syn=0.50, syn_scale=0.30, trap_shift=0.20):
        self.N = N; self.seed = seed
        rng = np.random.RandomState(seed)
        def var(base, frac=variability):
            return np.clip(base*(1+frac*rng.randn(N)), base*0.3, base*3.0).astype(np.float32)

        self.tau_mem = torch.tensor(var(1.0, 0.15), device=DEVICE)
        self.theta_base = torch.tensor(var(1.0, 0.05), device=DEVICE)
        self.tau_ref = torch.tensor(var(0.05, 0.10), device=DEVICE)
        self.I_bg = torch.tensor(var(bg_frac, 0.10) * self.theta_base.cpu().numpy(), device=DEVICE)
        self.W_in = torch.tensor(rng.randn(N, n_inputs).astype(np.float32) * 0.3, device=DEVICE)
        self.variability = variability
        self.bg_frac = bg_frac

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
        self.W_np = W.copy()
        self.W = torch.tensor(W, device=DEVICE)
        self.tau_syn_t = torch.tensor(var(tau_syn, 0.20), device=DEVICE)
        self.syn_scale = syn_scale
        self.delta_T = torch.tensor(var(delta_T, 0.15), device=DEVICE)
        vg2 = 0.35 + 0.12 * rng.rand(N).astype(np.float32)
        self.k_cap = torch.tensor((100.0/(1+np.exp((vg2-0.40)/0.05))).astype(np.float32), device=DEVICE)
        self.k_em = 50.0; self.trap_shift_val = trap_shift

    @torch.no_grad()
    def run(self, inputs_np, noise_sigma=0.05):
        if inputs_np.ndim == 1: inputs_np = inputs_np[:, None]
        T = len(inputs_np); N = self.N
        inputs = torch.tensor(inputs_np, dtype=torch.float32, device=DEVICE)
        Vm = torch.zeros(N, device=DEVICE)
        syn = torch.zeros(N, device=DEVICE)
        Q = torch.zeros(N, device=DEVICE)
        refrac = torch.zeros(N, device=DEVICE)
        rate_est = torch.zeros(N, device=DEVICE)
        ft = torch.zeros(N, device=DEVICE); st = torch.zeros(N, device=DEVICE)

        Vm_all = torch.zeros(N, T, device=DEVICE)
        spk_all = torch.zeros(N, T, device=DEVICE)
        Q_all = torch.zeros(N, T, device=DEVICE)
        syn_all = torch.zeros(N, T, device=DEVICE)
        states = torch.zeros(N, T, device=DEVICE)

        for t in range(T):
            u = inputs[t]
            I_in = self.W_in @ u
            I_syn = self.syn_scale * (self.W.T @ syn)
            dQ = self.k_cap * (1-Q) * rate_est - self.k_em * Q
            Q = torch.clamp(Q + dQ * 0.01, 0, 1)
            theta = torch.clamp(self.theta_base - Q * self.trap_shift_val, min=0.1)
            active = (refrac <= 0).float()
            leak = -Vm / self.tau_mem
            exp_term = self.delta_T * torch.exp(torch.clamp((Vm - theta) / self.delta_T, -10, 5))
            Vm = Vm + active * (leak + self.I_bg + I_in + I_syn + exp_term) + active * noise_sigma * torch.randn(N, device=DEVICE)
            Vm = torch.clamp(Vm, -2, 5)
            spiked = (Vm >= theta) & (refrac <= 0)
            if spiked.any():
                Vm[spiked] = 0; refrac[spiked] = self.tau_ref[spiked]
                syn[spiked] += 1; rate_est[spiked] += 5; spk_all[spiked, t] = 1
            syn *= torch.exp(-1/self.tau_syn_t); rate_est *= 0.95
            refrac = torch.clamp(refrac - 1, min=0)
            ft = 0.8*ft + 0.2*Vm; st = 0.98*st + 0.02*Vm
            Vm_all[:,t] = Vm; Q_all[:,t] = Q; syn_all[:,t] = syn
            states[:,t] = Vm + spk_all[:,t] + 0.3*ft + 0.1*st + 0.2*Q

        return {k: v.cpu().numpy() for k, v in
                {'states': states, 'spikes': spk_all, 'Vm': Vm_all,
                 'Q': Q_all, 'syn': syn_all}.items()}


# ── Benchmarks ──
def ridge(X, y, a=1.0): return np.linalg.solve(X.T@X+a*np.eye(X.shape[1]), X.T@y)
def ev_xor(S,u,wo,tau):
    T=S.shape[1];sp=wo+(T-wo)//2;X=S[:,wo+tau:].T;y=((u[wo+tau:]>0)!=(u[wo:T-tau]>0)).astype(float)
    s=sp-wo-tau;
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


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENTS + PLOTS
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("="*70)
    print("  z2508: NS-RAM Publication-Quality Results + Plots")
    print("="*70)

    rng = np.random.RandomState(42)
    T = 3000; wo = 500
    inputs = rng.uniform(-1, 1, T).astype(np.float64)
    all_results = {}

    # ─── Run best config ───
    print("\n[1] Running best NS-RAM config (N=128, noise=0.01)...")
    t0 = time.time()
    res = NSRAMReservoir(N=128, connectivity='sparse', spectral_radius=0.90,
                          seed=42, variability=0.10, bg_frac=0.95,
                          delta_T=0.10)
    # Run 5 reps
    best_results = []
    for rep in range(5):
        r = res.run(inputs, noise_sigma=0.01)
        S = r['states']; spk = r['spikes']
        m = {'xor1': ev_xor(S,inputs,wo,1), 'xor2': ev_xor(S,inputs,wo,2),
             'xor5': ev_xor(S,inputs,wo,5), 'mc': ev_mc(S,inputs,wo),
             'narma': ev_narma(S,inputs,wo), 'wave4': ev_wave(S,inputs,wo),
             'active': int((spk.sum(1)>0).sum()), 'spikes': int(spk.sum())}
        best_results.append(m)
        print(f"  rep{rep}: XOR1={m['xor1']:.1%} MC={m['mc']:.3f} "
              f"NARMA={m['narma']:.3f} W4={m['wave4']:.1%} ({m['active']}N)")
    all_results['best'] = best_results
    last_run = r  # Keep last run data for plotting
    elapsed = time.time() - t0
    print(f"  Total: {elapsed:.1f}s")

    # ─── PLOT 1: Membrane waveforms (slide 2 format) ───
    print("\n[2] Plotting membrane waveforms (slide 2 style)...")
    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
    Vm = last_run['Vm']; spk = last_run['spikes']
    t_range = slice(wo, wo+500)  # 500 steps after washout
    t_axis = np.arange(500)

    # Pick 4 neurons with different rates
    rates = spk.sum(axis=1)
    sorted_nids = np.argsort(rates)
    sample_nids = sorted_nids[np.array([10, 40, 80, 115])]  # low to high rate

    # Input
    axes[0].plot(t_axis, inputs[t_range], 'b-', linewidth=0.8)
    axes[0].set_ylabel('Input u(t)', fontsize=10)
    axes[0].set_ylim(-1.3, 1.3)
    axes[0].axhline(0, color='gray', linewidth=0.3)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, nid in enumerate(sample_nids):
        ax = axes[i+1]
        vm_trace = Vm[nid, t_range]
        spike_times = np.where(spk[nid, t_range] > 0)[0]
        ax.plot(t_axis, vm_trace, color=colors[i], linewidth=0.8)
        if len(spike_times) > 0:
            ax.scatter(spike_times, np.ones_like(spike_times) * 1.1,
                      marker='|', color='red', s=30, linewidths=1.5, zorder=5)
        ax.set_ylabel(f'N{nid}\n({rates[nid]:.0f} spk)', fontsize=9)
        ax.set_ylim(-0.5, 1.4)
        ax.axhline(1.0, color='gray', linewidth=0.3, linestyle='--', label='threshold' if i==0 else '')
    axes[-1].set_xlabel('Time step', fontsize=10)
    axes[0].set_title('NS-RAM Membrane Voltage Waveforms (cf. Pazos slide 2)',
                       fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig1_membrane_waveforms.png'), dpi=200)
    plt.close()
    print(f"  Saved fig1_membrane_waveforms.png")

    # ─── PLOT 2: Die-to-die variability grid (slide 16 format) ───
    print("\n[3] Plotting die-to-die variability (slide 16 style)...")
    fig, axes = plt.subplots(4, 4, figsize=(14, 10))
    sample_16 = sorted_nids[np.linspace(5, 120, 16).astype(int)]
    for idx, nid in enumerate(sample_16):
        ax = axes[idx // 4, idx % 4]
        vm_trace = Vm[nid, wo:wo+300]
        spike_t = np.where(spk[nid, wo:wo+300] > 0)[0]
        ax.plot(vm_trace, 'k-', linewidth=0.6)
        if len(spike_t) > 0:
            ax.scatter(spike_t, np.ones_like(spike_t)*1.1, marker='|',
                      color='red', s=20, linewidths=1)
        ax.set_title(f'N{nid} ({rates[nid]:.0f} spk)', fontsize=8)
        ax.set_ylim(-0.5, 1.4)
        ax.tick_params(labelsize=6)
        if idx % 4 != 0: ax.set_yticklabels([])
        if idx < 12: ax.set_xticklabels([])
    fig.suptitle('Die-to-Die Variability: 16 Neurons (cf. Pazos slide 16, Nature 640)',
                  fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig2_die_to_die_variability.png'), dpi=200)
    plt.close()
    print(f"  Saved fig2_die_to_die_variability.png")

    # ─── PLOT 3: Spike raster + population rate ───
    print("\n[4] Plotting spike raster...")
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(3, 1, height_ratios=[0.15, 0.70, 0.15], hspace=0.05)
    ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # Input
    ax1.plot(inputs[:T], 'b-', linewidth=0.3)
    ax1.set_ylabel('Input', fontsize=9); ax1.set_xticklabels([])

    # Raster
    st, sn = np.where(spk[:, :T].T > 0)
    ax2.scatter(st, sn, s=0.3, c='black', marker='.', rasterized=True)
    ax2.set_ylabel('Neuron #', fontsize=9); ax2.set_xticklabels([])
    ax2.set_ylim(-1, 128)

    # Pop rate
    win = 30
    pop_rate = np.convolve(spk.sum(axis=0), np.ones(win)/win, mode='valid')
    ax3.plot(pop_rate, 'r-', linewidth=0.5)
    ax3.set_ylabel('Pop Rate', fontsize=9); ax3.set_xlabel('Time step', fontsize=10)

    fig.suptitle('NS-RAM 128-Neuron Spike Raster', fontsize=12, fontweight='bold')
    plt.savefig(os.path.join(OUT_DIR, 'fig3_spike_raster.png'), dpi=200)
    plt.close()
    print(f"  Saved fig3_spike_raster.png")

    # ─── PLOT 4: BVpar surface + Vg1 sensitivity (slide 17/22) ───
    print("\n[5] Plotting BVpar physics (slide 17)...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # BVpar(Vg1) for different temperatures
    Vg1_sweep = np.linspace(0, 0.50, 100)
    for T_val, color, label in [(250, 'blue', '250K'), (300, 'black', '300K'),
                                 (350, 'red', '350K'), (400, 'orange', '400K')]:
        BVpar = (3.5 - 1.5 * Vg1_sweep) * (1 - 21.3e-6 * (T_val - 300))
        ax1.plot(Vg1_sweep, BVpar, color=color, linewidth=1.5, label=label)
    ax1.axhline(2.5, color='gray', linestyle='--', linewidth=1, label='Vcb peak (2.5V)')
    ax1.fill_between(Vg1_sweep, 0, 2.5, alpha=0.05, color='green')
    ax1.set_xlabel('Vg1 (V)', fontsize=11); ax1.set_ylabel('BVpar (V)', fontsize=11)
    ax1.set_title('Breakdown Voltage vs Gate Voltage\nBVpar = (3.5 - 1.5×Vg1) × (1 - 21.3μ×ΔT)',
                   fontsize=10)
    ax1.legend(fontsize=9); ax1.set_xlim(0, 0.5); ax1.set_ylim(1.5, 3.6)
    ax1.annotate('Avalanche\nregion', xy=(0.42, 2.2), fontsize=9, color='green',
                  ha='center', fontweight='bold')

    # Spike rate vs Vg1 sweep (simulated)
    print("  Running Vg1 sweep...")
    vg1_values = np.linspace(0.80, 1.00, 20)  # bg_frac sweep = effective Vg1
    spike_rates = []
    for bg in vg1_values:
        r_temp = NSRAMReservoir(N=64, seed=42, bg_frac=bg, variability=0.05)
        res_temp = r_temp.run(inputs[:500], noise_sigma=0.01)
        rate = res_temp['spikes'].sum() / (64 * 500)
        spike_rates.append(rate)
    ax2.semilogy(vg1_values, np.array(spike_rates) + 1e-4, 'ko-', linewidth=1.5, markersize=4)
    ax2.set_xlabel('Background Fraction (∝ Vg1)', fontsize=11)
    ax2.set_ylabel('Spike Rate (spk/neuron/step)', fontsize=11)
    ax2.set_title('Firing Rate vs Drive\n(cf. slide 22: 10⁴× range)', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig4_bvpar_physics.png'), dpi=200)
    plt.close()
    print(f"  Saved fig4_bvpar_physics.png")

    # ─── PLOT 5: Charge trapping dynamics (slide 17 SRH) ───
    print("\n[6] Plotting charge trapping...")
    Q_data = last_run['Q']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Q traces for selected neurons
    for i, nid in enumerate(sample_nids):
        ax1.plot(Q_data[nid, :], label=f'N{nid}', linewidth=1, alpha=0.8)
    ax1.set_ylabel('Trapped Charge Q', fontsize=11)
    ax1.legend(fontsize=8, ncol=4)
    ax1.set_title('SRH Charge Trapping Dynamics (cf. slide 17)', fontsize=12, fontweight='bold')

    # Heatmap
    im = ax2.imshow(Q_data, aspect='auto', cmap='magma', interpolation='none')
    ax2.set_ylabel('Neuron #', fontsize=11)
    ax2.set_xlabel('Time step', fontsize=11)
    plt.colorbar(im, ax=ax2, label='Q_trap', shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig5_charge_trapping.png'), dpi=200)
    plt.close()
    print(f"  Saved fig5_charge_trapping.png")

    # ─── PLOT 6: RC Benchmark comparison ───
    print("\n[7] Running benchmark sweep...")
    configs = [
        ("SNN 32\nsparse",   32, 'sparse',    0.90, 0.10, 0.01),
        ("SNN 64\nsparse",   64, 'sparse',    0.90, 0.10, 0.01),
        ("SNN 128\nsparse", 128, 'sparse',    0.90, 0.10, 0.01),
        ("SNN 256\nsparse", 256, 'sparse',    0.90, 0.10, 0.01),
        ("SNN 128\nsmall-w", 128, 'small_world', 0.90, 0.10, 0.01),
        ("SNN 128\ndense",  128, 'dense',     0.90, 0.10, 0.01),
        ("SNN 128\nnoisy",  128, 'sparse',    0.90, 0.10, 0.20),
        ("SNN 128\nhigh-var", 128, 'sparse',  0.90, 0.20, 0.01),
    ]

    bench_data = {'names': [], 'xor1': [], 'mc': [], 'narma': [], 'wave4': [],
                  'xor1_std': [], 'mc_std': [], 'narma_std': [], 'wave4_std': []}

    for name, N, conn, sr, var, noise in configs:
        xors, mcs, narmas, waves = [], [], [], []
        for rep in range(3):
            r = NSRAMReservoir(N=N, connectivity=conn, spectral_radius=sr,
                                seed=42+rep*100, variability=var)
            out = r.run(inputs, noise_sigma=noise)
            S = out['states']
            xors.append(ev_xor(S, inputs, wo, 1))
            mcs.append(ev_mc(S, inputs, wo))
            narmas.append(ev_narma(S, inputs, wo))
            waves.append(ev_wave(S, inputs, wo))
        bench_data['names'].append(name)
        bench_data['xor1'].append(np.mean(xors)); bench_data['xor1_std'].append(np.std(xors))
        bench_data['mc'].append(np.mean(mcs)); bench_data['mc_std'].append(np.std(mcs))
        bench_data['narma'].append(np.mean(narmas)); bench_data['narma_std'].append(np.std(narmas))
        bench_data['wave4'].append(np.mean(waves)); bench_data['wave4_std'].append(np.std(waves))
        print(f"  {name.replace(chr(10),' ')}: XOR1={np.mean(xors):.1%} MC={np.mean(mcs):.3f} "
              f"NARMA={np.mean(narmas):.3f} W4={np.mean(waves):.1%}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    x = np.arange(len(bench_data['names']))
    w = 0.6

    for ax, metric, label, fmt in [
        (axes[0,0], 'xor1', 'XOR τ=1 Accuracy', '{:.0%}'),
        (axes[0,1], 'mc', 'Memory Capacity', '{:.2f}'),
        (axes[1,0], 'narma', 'NARMA-5 R²', '{:.3f}'),
        (axes[1,1], 'wave4', '4-Class Waveform', '{:.0%}'),
    ]:
        vals = bench_data[metric]
        errs = bench_data[metric + '_std']
        colors = ['#4CAF50' if v > 0.7 else '#FF9800' if v > 0.5 else '#F44336' for v in vals]
        ax.bar(x, vals, w, yerr=errs, color=colors, edgecolor='black', linewidth=0.5,
               capsize=3, error_kw=dict(linewidth=1))
        ax.set_xticks(x); ax.set_xticklabels(bench_data['names'], fontsize=7)
        ax.set_ylabel(label, fontsize=10)
        for i, v in enumerate(vals):
            ax.text(i, v + errs[i] + 0.02, fmt.format(v), ha='center', fontsize=7, fontweight='bold')

    fig.suptitle('NS-RAM Reservoir Computing Benchmarks', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig6_rc_benchmarks.png'), dpi=200)
    plt.close()
    print(f"  Saved fig6_rc_benchmarks.png")

    # ─── PLOT 7: Scaling law ───
    print("\n[8] Plotting scaling law...")
    N_values = [16, 32, 64, 128, 256]
    scale_xor, scale_mc, scale_wave = [], [], []
    for N_val in N_values:
        r = NSRAMReservoir(N=N_val, seed=42, bg_frac=0.95, variability=0.10)
        out = r.run(inputs, noise_sigma=0.01)
        S = out['states']
        scale_xor.append(ev_xor(S, inputs, wo, 1))
        scale_mc.append(ev_mc(S, inputs, wo))
        scale_wave.append(ev_wave(S, inputs, wo))
        print(f"  N={N_val}: XOR1={scale_xor[-1]:.1%} MC={scale_mc[-1]:.3f} W4={scale_wave[-1]:.1%}")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
    ax1.semilogx(N_values, [x*100 for x in scale_xor], 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('N neurons'); ax1.set_ylabel('XOR-1 (%)'); ax1.set_title('XOR Scaling')
    ax1.axhline(50, color='gray', linestyle='--', label='Chance'); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.semilogx(N_values, scale_mc, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('N neurons'); ax2.set_ylabel('MC (total R²)'); ax2.set_title('Memory Capacity Scaling')
    ax2.grid(True, alpha=0.3)

    ax3.semilogx(N_values, [w*100 for w in scale_wave], 'ro-', linewidth=2, markersize=8)
    ax3.set_xlabel('N neurons'); ax3.set_ylabel('Wave-4 (%)'); ax3.set_title('Classification Scaling')
    ax3.axhline(25, color='gray', linestyle='--', label='Chance'); ax3.legend(); ax3.grid(True, alpha=0.3)

    fig.suptitle('NS-RAM Reservoir Scaling Law', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig7_scaling_law.png'), dpi=200)
    plt.close()
    print(f"  Saved fig7_scaling_law.png")

    # ─── PLOT 8: Weight matrix + connectivity ───
    print("\n[9] Plotting weight matrix...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    W = res.W_np
    im = ax1.imshow(W, cmap='RdBu_r', vmin=-np.abs(W).max(), vmax=np.abs(W).max(),
                     interpolation='none')
    ax1.set_xlabel('Post-synaptic'); ax1.set_ylabel('Pre-synaptic')
    ax1.set_title(f'Recurrent Weight Matrix (128×128)\n'
                   f'{(W!=0).sum()} connections ({(W!=0).mean():.1%} density)')
    plt.colorbar(im, ax=ax1, label='Weight', shrink=0.8)

    # Eigenvalue spectrum
    eigs = np.linalg.eigvals(W)
    ax2.scatter(eigs.real, eigs.imag, s=8, alpha=0.6, c='blue')
    circle = plt.Circle((0, 0), 0.90, fill=False, color='red', linewidth=1.5, linestyle='--')
    ax2.add_patch(circle)
    ax2.set_xlabel('Real'); ax2.set_ylabel('Imaginary')
    ax2.set_title(f'Eigenvalue Spectrum\nSpectral radius = {np.abs(eigs).max():.3f}')
    ax2.set_aspect('equal'); ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-1.2, 1.2); ax2.set_ylim(-1.2, 1.2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig8_weight_matrix.png'), dpi=200)
    plt.close()
    print(f"  Saved fig8_weight_matrix.png")

    # ─── Summary ───
    print("\n" + "="*70)
    print("  All plots saved to:", OUT_DIR)
    print("="*70)

    avg = {k: np.mean([r[k] for r in best_results]) for k in best_results[0]}
    print(f"\n  Best config (N=128, noise=0.01):")
    print(f"    XOR-1: {avg['xor1']:.1%} ± {np.std([r['xor1'] for r in best_results]):.1%}")
    print(f"    MC:    {avg['mc']:.3f} ± {np.std([r['mc'] for r in best_results]):.3f}")
    print(f"    NARMA: {avg['narma']:.3f} ± {np.std([r['narma'] for r in best_results]):.3f}")
    print(f"    Wave4: {avg['wave4']:.1%} ± {np.std([r['wave4'] for r in best_results]):.1%}")
    print(f"    Active: {avg['active']:.0f}/128")

    # Save JSON
    out_json = os.path.join(OUT_DIR, 'results.json')
    def ser(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating, np.float64)): return float(o)
        return o
    with open(out_json, 'w') as f:
        json.dump({
            'best_config': {k: ser(v) for k, v in avg.items()},
            'best_reps': [{k: ser(v) for k, v in r.items()} for r in best_results],
            'benchmark_sweep': {k: [ser(x) for x in v] if isinstance(v, list) else v
                                for k, v in bench_data.items()},
            'scaling': {'N': N_values,
                        'xor1': [ser(x) for x in scale_xor],
                        'mc': [ser(x) for x in scale_mc],
                        'wave4': [ser(x) for x in scale_wave]},
        }, f, indent=2)
    print(f"  Results JSON: {out_json}")


if __name__ == '__main__':
    main()
