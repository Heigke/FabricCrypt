#!/usr/bin/env python3
"""z2507_nsram_webapp.py — NS-RAM SDE Reservoir Interactive Webapp

Gradio 6.x compatible. Run sims, alter params, view raster plots, benchmarks.

Usage: python scripts/z2507_nsram_webapp.py
       → Opens browser at http://localhost:7860
"""

import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gradio as gr
import time, json, os, datetime

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class NSRAMReservoir:
    def __init__(self, N=128, n_inputs=1, connectivity='sparse',
                 spectral_radius=0.90, exc_frac=0.80, seed=42,
                 variability=0.10, bg_frac=0.95, delta_T=0.10,
                 tau_syn=0.50, syn_scale=0.30, trap_shift=0.20):
        self.N = N
        self.seed = seed
        self.params_dict = dict(N=N, connectivity=connectivity,
            spectral_radius=spectral_radius, exc_frac=exc_frac,
            variability=variability, bg_frac=bg_frac, delta_T=delta_T,
            tau_syn=tau_syn, syn_scale=syn_scale, trap_shift=trap_shift)
        rng = np.random.RandomState(seed)

        def var(base, frac=variability):
            return np.clip(base*(1+frac*rng.randn(N)), base*0.3, base*3.0).astype(np.float32)

        self.tau_mem = torch.tensor(var(1.0, 0.15), device=DEVICE)
        self.theta = torch.tensor(var(1.0, 0.05), device=DEVICE)
        self.tau_ref = torch.tensor(var(0.05, 0.10), device=DEVICE)
        self.I_bg = torch.tensor(var(bg_frac, 0.10) * self.theta.cpu().numpy(), device=DEVICE)
        self.W_in = torch.tensor(rng.randn(N, n_inputs).astype(np.float32) * 0.3, device=DEVICE)

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
        self.tau_syn_t = torch.tensor(var(tau_syn, 0.20), device=DEVICE)
        self.syn_scale = syn_scale
        self.delta_T = torch.tensor(var(delta_T, 0.15), device=DEVICE)

        vg2 = 0.35 + 0.12 * rng.rand(N).astype(np.float32)
        k_cap = (100.0 / (1.0 + np.exp((vg2 - 0.40) / 0.05))).astype(np.float32)
        self.k_cap = torch.tensor(k_cap, device=DEVICE)
        self.k_em = 50.0
        self.trap_shift_val = trap_shift

    @torch.no_grad()
    def run(self, inputs_np, noise_sigma=0.05):
        if inputs_np.ndim == 1:
            inputs_np = inputs_np[:, None]
        T = len(inputs_np)
        N = self.N
        inputs = torch.tensor(inputs_np, dtype=torch.float32, device=DEVICE)

        Vm = torch.zeros(N, device=DEVICE)
        syn = torch.zeros(N, device=DEVICE)
        Q = torch.zeros(N, device=DEVICE)
        refrac = torch.zeros(N, device=DEVICE)
        rate_est = torch.zeros(N, device=DEVICE)
        fast_trace = torch.zeros(N, device=DEVICE)
        slow_trace = torch.zeros(N, device=DEVICE)

        Vm_trace = torch.zeros(N, T, device=DEVICE)
        spike_trace = torch.zeros(N, T, device=DEVICE)
        Q_trace = torch.zeros(N, T, device=DEVICE)
        states = torch.zeros(N, T, device=DEVICE)

        for t in range(T):
            u = inputs[t]
            I_in = self.W_in @ u
            I_syn = self.syn_scale * (self.W.T @ syn)
            dQ = self.k_cap * (1.0 - Q) * rate_est - self.k_em * Q
            Q = torch.clamp(Q + dQ * 0.01, 0, 1)
            theta_eff = torch.clamp(self.theta - Q * self.trap_shift_val, min=0.1)

            active = (refrac <= 0).float()
            leak = -Vm / self.tau_mem
            exp_term = self.delta_T * torch.exp(
                torch.clamp((Vm - theta_eff) / self.delta_T, -10, 5))
            drive = self.I_bg + I_in + I_syn + exp_term
            noise = noise_sigma * torch.randn(N, device=DEVICE)
            Vm = Vm + active * (leak + drive) + active * noise
            Vm = torch.clamp(Vm, -2.0, 5.0)

            spiked = (Vm >= theta_eff) & (refrac <= 0)
            if spiked.any():
                Vm[spiked] = 0.0
                refrac[spiked] = self.tau_ref[spiked]
                syn[spiked] += 1.0
                rate_est[spiked] += 5.0
                spike_trace[spiked, t] = 1.0

            syn *= torch.exp(-1.0 / self.tau_syn_t)
            rate_est *= 0.95
            refrac = torch.clamp(refrac - 1.0, min=0)
            fast_trace = 0.8 * fast_trace + 0.2 * Vm
            slow_trace = 0.98 * slow_trace + 0.02 * Vm
            Vm_trace[:, t] = Vm
            Q_trace[:, t] = Q
            states[:, t] = Vm + spike_trace[:, t] + 0.3 * fast_trace + 0.1 * slow_trace + 0.2 * Q

        return (states.cpu().numpy(), spike_trace.cpu().numpy(),
                Vm_trace.cpu().numpy(), Q_trace.cpu().numpy())


# ── Benchmarks ──
def ridge(X, y, alpha=1.0):
    return np.linalg.solve(X.T@X + alpha*np.eye(X.shape[1]), X.T@y)

def ev_xor(S, u, wo, tau):
    T=S.shape[1]; sp=wo+(T-wo)//2; X=S[:,wo+tau:].T
    y=((u[wo+tau:]>0)!=(u[wo:T-tau]>0)).astype(float); s=sp-wo-tau
    if s<20 or len(y)-s<20: return 0.5
    w=ridge(X[:s],y[:s]); a=((X[s:]@w>0.5)==(y[s:]>0.5)).mean(); return max(a,1-a)

def ev_mc(S, u, wo, md=15):
    T=S.shape[1]; sp=wo+(T-wo)//2; mc=0
    for d in range(1,md+1):
        X=S[:,wo+d:].T; y=u[wo:T-d]; s=sp-wo-d
        if s<20 or len(y)-s<20: continue
        w=ridge(X[:s],y[:s]); p=X[s:]@w; yt=y[s:]
        if np.std(yt)<1e-10 or np.std(p)<1e-10: continue
        mc+=np.corrcoef(p,yt)[0,1]**2
    return mc

def ev_narma(S, u, wo, order=5):
    T=min(S.shape[1],len(u)); y=np.zeros(T); uu=(u[:T]+1)/2*0.5
    for t in range(order,T):
        y[t]=0.3*y[t-1]+0.05*y[t-1]*np.sum(y[t-order:t])+1.5*uu[t-1]*uu[t-order]+0.1
        y[t]=np.tanh(y[t])
    sp=wo+(T-wo)//2; X=S[:,wo:T].T; yt=y[wo:T]; s=sp-wo
    if s<20 or len(yt)-s<20: return 0
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


# ── Log ──
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'z2507_sim_log.jsonl')

def append_log(entry):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(entry, default=str) + '\n')

def load_log_text():
    if not os.path.exists(LOG_FILE):
        return "No runs yet. Click 'Run Simulation' to start."
    lines = []
    with open(LOG_FILE) as f:
        for line in f:
            try: lines.append(json.loads(line.strip()))
            except: pass
    if not lines:
        return "Log empty."
    rows = ["| # | Time | N | Conn | SR | Noise | XOR-1 | MC | NARMA | Wave4 | Spikes |",
            "|---|------|---|------|----|-------|-------|----|-------|-------|--------|"]
    for i, e in enumerate(lines[-15:]):
        m = e.get('metrics', {}); p = e.get('params', {})
        rows.append(f"| {len(lines)-14+i} | {e.get('ts','')[:16]} | {p.get('N','')} | "
                    f"{p.get('connectivity','')[:6]} | {p.get('spectral_radius','')} | "
                    f"{e.get('noise','')} | {m.get('xor1',0):.1%} | "
                    f"{m.get('mc',0):.3f} | {m.get('narma',0):.3f} | "
                    f"{m.get('wave4',0):.1%} | {e.get('spikes',0)} |")
    return '\n'.join(rows)


# ── Main simulation function ──
def run_sim(N, connectivity, spectral_radius, exc_frac, variability,
            bg_frac, delta_T, tau_syn, syn_scale, trap_shift,
            noise_sigma, n_steps, seed):
    N = int(N); n_steps = int(n_steps); seed = int(seed)
    t0 = time.time()

    res = NSRAMReservoir(N=N, connectivity=connectivity, spectral_radius=spectral_radius,
                          exc_frac=exc_frac, seed=seed, variability=variability,
                          bg_frac=bg_frac, delta_T=delta_T, tau_syn=tau_syn,
                          syn_scale=syn_scale, trap_shift=trap_shift)

    rng = np.random.RandomState(seed + 1)
    inputs = rng.uniform(-1, 1, n_steps).astype(np.float64)
    S, spk, Vm, Q = res.run(inputs, noise_sigma=noise_sigma)
    elapsed = time.time() - t0

    wo = min(500, n_steps // 4)
    xor1 = ev_xor(S, inputs, wo, 1)
    xor2 = ev_xor(S, inputs, wo, 2)
    xor5 = ev_xor(S, inputs, wo, 5)
    mc = ev_mc(S, inputs, wo)
    narma = ev_narma(S, inputs, wo)
    wave4 = ev_wave(S, inputs, wo)

    total_spk = int(spk.sum())
    n_active = int((spk.sum(axis=1) > 0).sum())
    T = S.shape[1]

    # ── Raster plot ──
    fig_raster = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                row_heights=[0.12, 0.76, 0.12],
                                vertical_spacing=0.03)
    fig_raster.add_trace(go.Scatter(y=inputs[:T], mode='lines',
                                     line=dict(color='#2196F3', width=1),
                                     name='Input'), row=1, col=1)
    st, sn = np.where(spk.T > 0)
    if len(st) > 50000:
        idx = np.random.choice(len(st), 50000, replace=False)
        st, sn = st[idx], sn[idx]
    if len(st) > 0:
        fig_raster.add_trace(go.Scattergl(x=st, y=sn, mode='markers',
                                           marker=dict(size=1.5, color='black'),
                                           name='Spikes'), row=2, col=1)
    win = 20
    if T > win:
        pr = np.convolve(spk.sum(axis=0), np.ones(win)/win, mode='valid')
        fig_raster.add_trace(go.Scatter(y=pr, mode='lines',
                                         line=dict(color='#F44336', width=1),
                                         name='Pop Rate'), row=3, col=1)
    fig_raster.update_layout(height=500, showlegend=False,
                              margin=dict(l=50, r=10, t=10, b=30),
                              plot_bgcolor='white')
    fig_raster.update_yaxes(title_text="Input", row=1, col=1)
    fig_raster.update_yaxes(title_text="Neuron", row=2, col=1)
    fig_raster.update_yaxes(title_text="Rate", row=3, col=1)
    fig_raster.update_xaxes(title_text="Time step", row=3, col=1)

    # ── Membrane traces ──
    rates = spk.sum(axis=1)
    sorted_idx = np.argsort(rates)
    nids = sorted_idx[np.linspace(0, N-1, min(8, N)).astype(int)]
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f']
    fig_vm = go.Figure()
    for i, nid in enumerate(nids):
        fig_vm.add_trace(go.Scatter(y=Vm[nid, :min(T,1000)], mode='lines',
                                     name=f'N{nid}', line=dict(color=colors[i%8], width=1),
                                     opacity=0.8))
    fig_vm.update_layout(height=350, margin=dict(l=50,r=10,t=30,b=30),
                          title="Membrane Voltage (first 1000 steps)",
                          xaxis_title="Step", yaxis_title="Vm", plot_bgcolor='white')

    # ── Charge trap heatmap ──
    fig_q = go.Figure(data=go.Heatmap(z=Q[:, :min(T,2000)], colorscale='Viridis',
                                       colorbar_title='Q'))
    fig_q.update_layout(height=250, margin=dict(l=50,r=10,t=30,b=30),
                         title="Charge Trap State", xaxis_title="Step", yaxis_title="Neuron")

    # ── Benchmark bars ──
    metrics = {'XOR-1': xor1, 'XOR-2': xor2, 'XOR-5': xor5,
               'MC': mc, 'NARMA': narma, 'Wave4': wave4}
    names = list(metrics.keys())
    vals = list(metrics.values())
    bar_colors = ['#4CAF50' if v > 0.7 else '#FF9800' if v > 0.5 else '#F44336'
                  for v in vals]
    fig_bench = go.Figure(data=go.Bar(
        x=names, y=vals, marker_color=bar_colors,
        text=[f'{v:.1%}' if v < 3 else f'{v:.2f}' for v in vals],
        textposition='outside'))
    fig_bench.update_layout(height=300, margin=dict(l=50,r=10,t=30,b=30),
                             title="RC Benchmarks", yaxis_title="Score",
                             plot_bgcolor='white')

    # ── Stats text ──
    ar = rates[rates > 0]
    stats = (f"## Results ({elapsed:.1f}s on {DEVICE})\n"
             f"- **Active neurons**: {n_active}/{N}\n"
             f"- **Total spikes**: {total_spk:,}\n"
             f"- **Mean rate**: {ar.mean():.1f} spk/step\n\n"
             f"| Metric | Score |\n|--------|-------|\n"
             f"| XOR-1 | **{xor1:.1%}** |\n"
             f"| XOR-2 | {xor2:.1%} |\n"
             f"| XOR-5 | {xor5:.1%} |\n"
             f"| MC | **{mc:.3f}** |\n"
             f"| NARMA-5 | **{narma:.3f}** |\n"
             f"| Wave-4 | **{wave4:.1%}** |\n")

    # ── Log ──
    append_log({'ts': datetime.datetime.now().isoformat(),
                'params': res.params_dict, 'noise': noise_sigma,
                'n_steps': n_steps, 'seed': seed,
                'metrics': {k: float(v) for k, v in metrics.items()},
                'spikes': total_spk, 'active': n_active, 'time': elapsed})

    return stats, fig_raster, fig_vm, fig_q, fig_bench, load_log_text()


# ── Gradio UI ──
def build_app():
    with gr.Blocks(title="NS-RAM Reservoir Simulator") as app:
        gr.Markdown("# NS-RAM Spiking Neural Network Reservoir\n"
                     "AdEx-LIF + Dale's law synapses + SRH charge trapping. "
                     "Based on Pazos/Lanza (Nature 640, 2025). "
                     f"GPU: {DEVICE}")

        with gr.Row():
            # Left: controls
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### Network")
                n_neurons = gr.Slider(16, 512, value=128, step=16, label="Neurons (N)")
                connectivity = gr.Dropdown(['sparse', 'small_world', 'dense'],
                                            value='sparse', label="Connectivity")
                spectral_radius = gr.Slider(0.5, 1.5, value=0.90, step=0.05,
                                             label="Spectral Radius")
                exc_frac = gr.Slider(0.5, 1.0, value=0.80, step=0.05,
                                      label="Exc. Fraction")

                gr.Markdown("### NS-RAM Physics")
                variability = gr.Slider(0.0, 0.30, value=0.10, step=0.02,
                                         label="Die-to-die variability")
                bg_frac = gr.Slider(0.80, 1.00, value=0.95, step=0.01,
                                     label="Background (frac of threshold)")
                delta_T = gr.Slider(0.01, 0.50, value=0.10, step=0.01,
                                     label="Avalanche sharpness (delta_T)")
                trap_shift = gr.Slider(0.0, 0.50, value=0.20, step=0.05,
                                        label="Charge trap shift")

                gr.Markdown("### Synapses")
                tau_syn = gr.Slider(0.1, 2.0, value=0.50, step=0.1, label="Synaptic tau")
                syn_scale = gr.Slider(0.0, 1.0, value=0.30, step=0.05, label="Synaptic scale")

                gr.Markdown("### Simulation")
                noise_sigma = gr.Slider(0.0, 0.50, value=0.05, step=0.01, label="Noise sigma")
                n_steps = gr.Slider(500, 10000, value=3000, step=500, label="Steps")
                seed = gr.Number(value=42, label="Seed", precision=0)

                run_btn = gr.Button("Run Simulation", variant="primary", size="lg")

            # Right: outputs
            with gr.Column(scale=2):
                stats_out = gr.Markdown("Click **Run Simulation** to start.")
                gr.Markdown("### Spike Raster")
                raster_out = gr.Plot()
                gr.Markdown("### Membrane Traces")
                vm_out = gr.Plot()

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Charge Trapping")
                        q_out = gr.Plot()
                    with gr.Column():
                        gr.Markdown("### Benchmarks")
                        bench_out = gr.Plot()

                gr.Markdown("### Simulation Log")
                log_out = gr.Markdown(load_log_text())

        all_inputs = [n_neurons, connectivity, spectral_radius, exc_frac,
                      variability, bg_frac, delta_T, tau_syn, syn_scale,
                      trap_shift, noise_sigma, n_steps, seed]
        all_outputs = [stats_out, raster_out, vm_out, q_out, bench_out, log_out]

        run_btn.click(fn=run_sim, inputs=all_inputs, outputs=all_outputs)

    return app


if __name__ == '__main__':
    print(f"NS-RAM Reservoir Simulator | Device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
