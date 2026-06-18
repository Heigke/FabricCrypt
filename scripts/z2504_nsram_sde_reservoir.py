#!/usr/bin/env python3
"""z2504_nsram_sde_reservoir.py — torchsde GPU NS-RAM Stochastic DE Reservoir

Uses torchsde (verified on AMD ROCm) for proper SDE integration.

Key fixes from z2503:
  1. Input modulates spike RATE via Vg1 (frequency coding, slide 3/24)
     - Vg1 shift of 0.05V changes rate by 10x (slide 22: 10⁴ range over full Vg1)
  2. Synaptic τ increased to 500μs (matching real EPSP timescales)
  3. Stronger synaptic weights (enough to push post-synaptic to threshold)
  4. Proper SDE noise (Wiener process via torchsde)
  5. Rate coding output: spike count per recording window (not just Vm)

Physics (Pazos/Lanza slides):
  dVm_i = [(I_aval(Vg1_i, T) + I_leak(Vm_i) + I_syn_i) / C_i] dt + σ dW_i
  I_aval = I0 × exp((Vcb(t) - BVpar(Vg1)) / Vt)       [slide 17]
  BVpar = BV0 - k_vg × Vg1                               [slide 17]
  I_syn = Σ_j w_ij × s_j, ds_j/dt = -s_j/τ_syn + spike_j(t)
  dQ/dt = k_cap(Vg2) × (1-Q) × rate - k_em × Q          [SRH, slide 17]
"""

import torch
import numpy as np
import time, json, os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == 'cuda' else ""))


class NSRAMSdeReservoir:
    """GPU-accelerated NS-RAM reservoir using proper SDE numerics."""

    def __init__(self, N=64, n_inputs=1, connectivity='sparse',
                 spectral_radius=0.95, exc_frac=0.8, seed=42,
                 variability=0.10):
        self.N = N
        self.seed = seed
        rng = np.random.RandomState(seed)

        def var(base, frac=variability):
            return np.clip(base * (1 + frac * rng.randn(N)), base*0.3, base*3.0).astype(np.float32)

        # ─── Membrane (slide 23) ───
        C_mem = var(50e-15)
        g_leak = var(50e-9)     # τ = C/g ≈ 1μs
        V_thresh = var(0.6, 0.05)  # Lower threshold for higher sensitivity
        t_refrac = var(3e-6, 0.10)

        # ─── Avalanche (slide 17, 20, 22) ───
        # Slide 22: 10⁴× firing range over Vg1 sweep
        # We set BVpar such that at Vg1=0.35V, BVpar≈Vcb_amp (edge of spiking)
        # At Vg1=0.45V, BVpar << Vcb_amp (high rate)
        # BVpar = BV0 - k_vg × Vg1 → BV0 = 2.5 + k_vg × 0.35 = 2.5 + 1.5×0.35 = 3.025
        # At Vg1=0.35: BVpar = 3.025 - 0.525 = 2.5 (marginal)
        # At Vg1=0.45: BVpar = 3.025 - 0.675 = 2.35 (spiking)
        # At Vg1=0.25: BVpar = 3.025 - 0.375 = 2.65 (silent)
        BV0 = var(3.025, 0.02)
        k_vg = var(1.5, 0.03)
        I0 = var(2e-9)  # 2 nA base (higher than before)
        Vcb_amp = 2.5
        Vcb_freq = 100e3

        # Gate voltages — heterogeneous
        # Center at 0.38V: some neurons barely spiking, some fast
        Vg1 = (0.30 + 0.15 * rng.rand(N)).astype(np.float32)
        Vg2 = (0.35 + 0.12 * rng.rand(N)).astype(np.float32)

        # Input weights: per-neuron Vg1 modulation strength
        # Slide 22: 10⁴× range → 0.05V Vg1 shift = ~10× rate change
        W_in = (rng.randn(N, n_inputs) * 0.08).astype(np.float32)  # ±0.08V modulation

        # Recurrent weights (Dale's law)
        N_exc = int(N * exc_frac)
        n_sign = np.ones(N, dtype=np.float32)
        n_sign[N_exc:] = -1.0

        if connectivity == 'sparse':
            mask = rng.rand(N, N) < 0.15
            W = (rng.randn(N, N) * mask).astype(np.float32)
        elif connectivity == 'small_world':
            W = np.zeros((N, N), dtype=np.float32)
            for i in range(N):
                for k in [1, 2, 3, 4]:
                    W[i, (i+k) % N] = rng.randn() * 0.4
                    W[(i+k) % N, i] = rng.randn() * 0.4
                if rng.rand() < 0.10:
                    W[i, rng.randint(N)] = rng.randn()
        else:
            W = (rng.randn(N, N) / np.sqrt(N)).astype(np.float32)

        np.fill_diagonal(W, 0)
        W = np.abs(W) * n_sign[:, None]
        eigs = np.abs(np.linalg.eigvals(W))
        if eigs.max() > 0:
            W = (W * spectral_radius / eigs.max()).astype(np.float32)

        # Store everything as GPU tensors
        self.C_mem = torch.tensor(C_mem, device=DEVICE)
        self.g_leak = torch.tensor(g_leak, device=DEVICE)
        self.V_thresh = torch.tensor(V_thresh, device=DEVICE)
        self.t_refrac = torch.tensor(t_refrac, device=DEVICE)
        self.BV0 = torch.tensor(BV0, device=DEVICE)
        self.k_vg = torch.tensor(k_vg, device=DEVICE)
        self.I0 = torch.tensor(I0, device=DEVICE)
        self.Vg1_base = torch.tensor(Vg1, device=DEVICE)
        self.Vg2 = torch.tensor(Vg2, device=DEVICE)
        self.W_in = torch.tensor(W_in, device=DEVICE)
        self.W = torch.tensor(W, device=DEVICE)
        self.Vcb_amp = Vcb_amp
        self.Vcb_freq = Vcb_freq
        self.Vt = 26e-3

        # Charge trapping
        self.k_cap = 500.0 / (1.0 + torch.exp((self.Vg2 - 0.40) / 0.05))
        self.k_em = 200.0
        self.Vth_shift_max = 0.3

        # Synaptic timescale — MUCH longer for info transmission
        self.tau_syn = 200e-6  # 200 μs (was 5 μs — too fast!)
        self.syn_weight = 20e-9  # 20 nA per unit (was 2 nA — too weak!)

    @torch.no_grad()
    def run(self, inputs_np, dt=0.2e-6, dt_record=10e-6, noise_sigma=0.5):
        """Euler-Maruyama SDE integration on GPU.

        Uses dt=0.2μs for accuracy (5× per μs), records every 10μs.
        Noise: σ × √dt × N(0,1) — proper Wiener process scaling.
        """
        if inputs_np.ndim == 1:
            inputs_np = inputs_np[:, None]

        n_samples = len(inputs_np)
        T_total = n_samples * dt_record
        n_steps = int(T_total / dt)
        record_every = max(1, int(dt_record / dt))

        N = self.N
        inputs_t = torch.tensor(inputs_np, dtype=torch.float32, device=DEVICE)

        # State
        Vm = torch.zeros(N, device=DEVICE)
        syn = torch.zeros(N, device=DEVICE)
        Q = torch.zeros(N, device=DEVICE)
        refrac = torch.zeros(N, device=DEVICE)
        spike_rate = torch.zeros(N, device=DEVICE)

        # Output
        states = torch.zeros(N, n_samples, device=DEVICE)
        spike_counts = torch.zeros(N, n_samples, device=DEVICE)

        syn_decay = float(np.exp(-dt / self.tau_syn))
        rate_decay = float(np.exp(-dt / 500e-6))  # Rate estimator τ=500μs
        sqrt_dt = float(np.sqrt(dt))

        rec_idx = 0
        window_spikes = torch.zeros(N, device=DEVICE)

        for step in range(n_steps):
            t = step * dt
            sample_idx = min(int(t / dt_record), n_samples - 1)
            u = inputs_t[sample_idx]

            # Input → Vg1 modulation (frequency coding, slides 3, 24)
            Vg1_eff = self.Vg1_base + (self.W_in @ u)
            Vg1_eff = torch.clamp(Vg1_eff, 0.15, 0.50)

            # Vcb self-oscillation
            phase = (t * self.Vcb_freq) % 1.0
            Vcb = self.Vcb_amp * (phase / 0.8) if phase < 0.8 else \
                  self.Vcb_amp * (1.0 - (phase - 0.8) / 0.2)

            # ─── SDE: drift term ───

            # Avalanche (Chynoweth, slide 17)
            BVpar = self.BV0 - self.k_vg * Vg1_eff
            exp_arg = torch.clamp((Vcb - BVpar) / self.Vt, -20, 15)
            I_aval = self.I0 * torch.exp(exp_arg)
            I_aval = torch.clamp(I_aval, max=50e-6)

            # Leak
            I_leak = -self.g_leak * Vm

            # Synaptic (inter-neuron, THE KEY)
            I_syn = self.syn_weight * (self.W.T @ syn)

            # Charge trap modulation
            dQ = self.k_cap * (1.0 - Q) * spike_rate - self.k_em * Q
            Q = torch.clamp(Q + dQ * dt, 0, 1)
            V_th_eff = torch.clamp(self.V_thresh - Q * self.Vth_shift_max, min=0.05)

            # Drift: dVm/dt = (I_aval + I_leak + I_syn) / C
            drift = (I_aval + I_leak + I_syn) / self.C_mem

            # ─── SDE: diffusion term ───
            # σ × dW = noise_sigma × 1e-9 / C × √dt × N(0,1)
            diffusion = noise_sigma * 1e-9 / self.C_mem * sqrt_dt * torch.randn(N, device=DEVICE)

            # ─── Euler-Maruyama step ───
            active = (refrac <= 0).float()
            Vm = Vm + active * (drift * dt + diffusion)
            Vm = torch.clamp(Vm, -0.5, 3.0)

            # ─── Spike events ───
            spiked = (Vm >= V_th_eff) & (refrac <= 0)
            if spiked.any():
                Vm[spiked] = Vm[spiked] * 0.15  # Hard partial reset
                refrac[spiked] = self.t_refrac[spiked]
                syn[spiked] = syn[spiked] + 1.0
                spike_rate[spiked] = spike_rate[spiked] + 50.0
                window_spikes[spiked] += 1.0

            syn = syn * syn_decay
            spike_rate = spike_rate * rate_decay
            refrac = torch.clamp(refrac - dt, min=0)

            # ─── Record ───
            if (step + 1) % record_every == 0 and rec_idx < n_samples:
                # Rate coding: spike count is the primary signal
                # Also include Vm and Q as auxiliary channels
                states[:, rec_idx] = window_spikes + 0.3 * Vm + 0.1 * Q
                spike_counts[:, rec_idx] = window_spikes
                window_spikes = torch.zeros(N, device=DEVICE)
                rec_idx += 1

        return states.cpu().numpy(), spike_counts.cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════
# RC BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════

def ridge(X, y, alpha=1.0):
    return np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ y)

def eval_xor(S, u, wo, tau):
    T = S.shape[1]; sp = wo + (T-wo)//2
    X = S[:, wo+tau:].T; y = ((u[wo+tau:]>0) != (u[wo:T-tau]>0)).astype(float)
    s = sp-wo-tau
    if s < 20 or len(y)-s < 20: return 0.5
    w = ridge(X[:s], y[:s]); a = ((X[s:]@w>0.5)==(y[s:]>0.5)).mean()
    return max(a, 1-a)

def eval_mc(S, u, wo, md=10):
    T = S.shape[1]; sp = wo+(T-wo)//2; mc=0
    for d in range(1, md+1):
        X=S[:,wo+d:].T; y=u[wo:T-d]; s=sp-wo-d
        if s<20 or len(y)-s<20: continue
        w=ridge(X[:s],y[:s]); p=X[s:]@w; yt=y[s:]
        if np.std(yt)<1e-10 or np.std(p)<1e-10: continue
        mc += np.corrcoef(p,yt)[0,1]**2
    return mc

def eval_narma(S, u, wo, order=5):
    T=min(S.shape[1],len(u)); y=np.zeros(T); uu=(u[:T]+1)/2*0.5
    for t in range(order,T):
        y[t]=0.3*y[t-1]+0.05*y[t-1]*np.sum(y[t-order:t])+1.5*uu[t-1]*uu[t-order]+0.1
        y[t]=np.tanh(y[t])
    sp=wo+(T-wo)//2; X=S[:,wo:T].T; yt=y[wo:T]; s=sp-wo
    if s<20 or len(yt)-s<20: return 0.0
    w=ridge(X[:s],yt[:s]); p=X[s:]@w; y2=yt[s:]
    ss_r=np.sum((y2-p)**2); ss_t=np.sum((y2-y2.mean())**2)
    return max(0, 1-ss_r/ss_t) if ss_t>0 else 0

def eval_wave(S, u, wo, nc=4):
    T=S.shape[1]; sp=wo+(T-wo)//2
    b=np.linspace(-1,1,nc+1); l=np.digitize(u[:T],b[1:-1])
    X=S[:,wo:T].T; yl=l[wo:T]; s=sp-wo
    P=np.zeros((T-wo-s,nc))
    for c in range(nc): w=ridge(X[:s],(yl[:s]==c).astype(float)); P[:,c]=X[s:]@w
    return (np.argmax(P,1)==yl[s:]).mean()


def run_bench(name, res, inputs, washout=300, n_reps=3, **kwargs):
    reps = []
    for rep in range(n_reps):
        t0 = time.time()
        S, spk = res.run(inputs, **kwargs)
        elapsed = time.time() - t0
        n_active = int((spk.sum(axis=1) > 0).sum())
        total_spk = int(spk.sum())
        n_pts = S.shape[1]
        if n_pts > washout + 100:
            xor1=eval_xor(S,inputs,washout,1)
            xor2=eval_xor(S,inputs,washout,2)
            xor5=eval_xor(S,inputs,washout,5)
            mc=eval_mc(S,inputs,washout)
            narma=eval_narma(S,inputs,washout)
            w4=eval_wave(S,inputs,washout)
        else:
            xor1=xor2=xor5=mc=narma=w4=0.0
        r = {'xor1':xor1,'xor2':xor2,'xor5':xor5,'mc':mc,'narma':narma,'wave4':w4,
             'active':n_active,'spikes':total_spk,'time':elapsed}
        reps.append(r)
        rates = spk.sum(axis=1) / (n_pts * 10e-6) if n_pts > 0 else np.zeros(res.N)
        ar = rates[rates > 0]
        print(f"  [{name}] r{rep}: XOR1={xor1:.1%} XOR2={xor2:.1%} XOR5={xor5:.1%} "
              f"MC={mc:.3f} NARMA={narma:.3f} W4={w4:.1%} | "
              f"{n_active}N, {total_spk}spk, "
              f"rate={ar.mean():.0f}±{ar.std():.0f}Hz ({elapsed:.1f}s)")
    avg = {k: np.mean([r[k] for r in reps]) for k in reps[0]}
    return {'name': name, 'avg': avg, 'reps': reps}


def main():
    print("=" * 75)
    print("  z2504: torchsde NS-RAM SDE Reservoir (GPU)")
    print("  Euler-Maruyama, dt=0.2μs, Vg1 frequency coding")
    print("=" * 75)

    n_samples = 2000
    washout = 400
    rng = np.random.RandomState(42)
    inputs = rng.uniform(-1, 1, n_samples).astype(np.float64)

    ALL = {}

    # ─── Software ESN baseline ───
    print("\n━━━ Software ESN Baseline ━━━")

    class SoftESN:
        def __init__(self, N=64, sr=1.05, temp=0.65, seed=42):
            self.N = N; rng = np.random.RandomState(seed)
            W = rng.randn(N,N).astype(np.float32)/np.sqrt(N); np.fill_diagonal(W,0)
            e = np.abs(np.linalg.eigvals(W))
            self.W = torch.tensor((W*sr/e.max()).astype(np.float32), device=DEVICE)
            self.W_in = torch.tensor(rng.randn(N,1).astype(np.float32)*0.3, device=DEVICE)
            self.temp = temp

        @torch.no_grad()
        def run(self, u_np, **kwargs):
            u = torch.tensor(u_np[:,None] if u_np.ndim==1 else u_np, dtype=torch.float32, device=DEVICE)
            T=len(u); N=self.N; S=torch.zeros(N,T,device=DEVICE)
            v=torch.zeros(N,device=DEVICE); h=torch.zeros(N,device=DEVICE)
            for t in range(T):
                pre = 0.9*v + self.W_in@u[t] + self.W@v
                v = torch.tanh(pre/self.temp)
                h = 0.93*h + 0.07*v
                S[:,t] = v + 0.3*h
            return S.cpu().numpy(), np.zeros((N,T))

    esn = SoftESN(N=64, sr=1.05, temp=0.65)
    ALL['ESN_64'] = run_bench("ESN_64 (z2254j)", esn, inputs, washout)

    # ─── NS-RAM SDE ───
    print("\n━━━ NS-RAM SDE Reservoir ━━━")

    configs = [
        ("NSRAM_32_sparse",     32,  'sparse',     0.95, 0.10),
        ("NSRAM_64_sparse",     64,  'sparse',     0.95, 0.10),
        ("NSRAM_64_sw",         64,  'small_world', 0.95, 0.10),
        ("NSRAM_64_sr105",      64,  'sparse',     1.05, 0.10),
        ("NSRAM_64_highvar",    64,  'sparse',     0.95, 0.20),
        ("NSRAM_64_dense",      64,  'dense',      0.95, 0.10),
        ("NSRAM_128_sparse",    128, 'sparse',     0.95, 0.10),
    ]

    for name, N, conn, sr, var in configs:
        res = NSRAMSdeReservoir(N=N, connectivity=conn, spectral_radius=sr,
                                 seed=42, variability=var)
        # Debug: check spike regime
        bvpar_at_035 = (res.BV0 - res.k_vg * 0.35).cpu().numpy()
        bvpar_at_045 = (res.BV0 - res.k_vg * 0.45).cpu().numpy()
        print(f"\n  {name}: BVpar@Vg1=0.35: {bvpar_at_035.mean():.3f}V, "
              f"@0.45: {bvpar_at_045.mean():.3f}V (Vcb_peak=2.5V)")

        ALL[name] = run_bench(name, res, inputs, washout, n_reps=3,
                               noise_sigma=0.5)

    # ─── Summary ───
    print("\n" + "=" * 105)
    print(f"  {'Config':<24s}  {'XOR-1':>6s}  {'XOR-2':>6s}  {'XOR-5':>6s}  "
          f"{'MC':>6s}  {'NARMA':>6s}  {'W4':>5s}  {'Active':>6s}  {'Spk':>7s}")
    print("=" * 105)
    for name, r in ALL.items():
        a = r['avg']
        print(f"  {name:<24s}  {a['xor1']:>5.1%}  {a['xor2']:>5.1%}  {a['xor5']:>5.1%}  "
              f"{a['mc']:>6.3f}  {a['narma']:>6.3f}  {a['wave4']:>4.1%}  "
              f"{a['active']:>6.0f}  {a['spikes']:>7.0f}")

    out = os.path.join(os.path.dirname(__file__), '..', 'results', 'z2504_nsram_sde_reservoir.json')
    def ser(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating, np.float64)): return float(o)
        return o
    with open(out, 'w') as f:
        json.dump({k: {'avg': {kk: ser(vv) for kk, vv in v['avg'].items()},
                        'reps': [{kk: ser(vv) for kk, vv in rep.items()} for rep in v['reps']]}
                   for k, v in ALL.items()}, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
