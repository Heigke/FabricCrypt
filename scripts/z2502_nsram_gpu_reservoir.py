#!/usr/bin/env python3
"""z2502_nsram_gpu_reservoir.py — GPU-accelerated NS-RAM ODE Reservoir

Fixes from z2501:
  1. Software ESN baseline was broken (wrong recurrence scaling)
  2. Input coupling too weak (needs proper scaling to spike regime)
  3. Now GPU-accelerated via PyTorch for speed

Physics from Pazos/Lanza slides:
  - Semi-empirical Chynoweth avalanche model (slide 17)
  - SRH charge trapping with VG2 dependence (slide 17)
  - Brian2 LIF: τ_mem=1μs, V_thresh=1.364V, t_refrac=1.6μs (slide 23)
  - Die-to-die variability (slide 16)
  - 21 fJ/spike, 60-360 kHz spiking range (slide 2)
  - VG1: 0-0.425V, VG2: 0.275-0.475V (Sebastian's SPICE parameters)
"""

import torch
import numpy as np
import time, json, os

DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif hasattr(torch, 'hip') or os.path.exists('/opt/rocm'):
    try:
        torch.zeros(1, device='cuda')
        DEVICE = 'cuda'
    except:
        DEVICE = 'cpu'

print(f"Using device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")


class NSRAMReservoirGPU:
    """GPU-accelerated NS-RAM reservoir with full Pazos/Lanza physics."""

    def __init__(self, N=128, n_inputs=1, connectivity='sparse',
                 spectral_radius=0.95, exc_frac=0.8, seed=42,
                 variability=0.10, dt=1e-6, substeps=10):
        self.N = N
        self.n_inputs = n_inputs
        self.seed = seed
        self.dt = dt
        self.substeps = substeps
        self.device = DEVICE

        rng = np.random.RandomState(seed)

        # ─── Per-neuron parameters (slide 16: die-to-die variability) ───
        def var(base, frac=variability):
            v = base * (1 + frac * rng.randn(N))
            return np.clip(v, base * 0.5, base * 2.0)

        # Membrane (slide 23)
        C_mem = var(102e-15)
        g_leak = C_mem / var(1e-6)  # τ_mem = C/g ≈ 1μs
        V_thresh = var(1.364, 0.05)  # 1.364V ± 5%
        t_refrac = var(1.6e-6, 0.10)

        # Avalanche (slide 17)
        BV0 = var(3.5, 0.03)
        k_vg = var(1.5, 0.05)
        I0 = var(0.5e-9)

        # Gate voltages (heterogeneous: slide VG1=0-0.425V, VG2=0.275-0.475V)
        Vg1 = 0.25 + 0.20 * rng.rand(N)  # Heterogeneous
        Vg2 = 0.30 + 0.17 * rng.rand(N)

        # Store as tensors
        self.C_mem = torch.tensor(C_mem, dtype=torch.float32, device=DEVICE)
        self.g_leak = torch.tensor(g_leak, dtype=torch.float32, device=DEVICE)
        self.V_thresh_base = torch.tensor(V_thresh, dtype=torch.float32, device=DEVICE)
        self.t_refrac = torch.tensor(t_refrac, dtype=torch.float32, device=DEVICE)
        self.BV0 = torch.tensor(BV0, dtype=torch.float32, device=DEVICE)
        self.k_vg = torch.tensor(k_vg, dtype=torch.float32, device=DEVICE)
        self.I0 = torch.tensor(I0, dtype=torch.float32, device=DEVICE)
        self.Vg1 = torch.tensor(Vg1, dtype=torch.float32, device=DEVICE)
        self.Vg2 = torch.tensor(Vg2, dtype=torch.float32, device=DEVICE)

        # ─── Synaptic weights (Dale's law) ───
        N_exc = int(N * exc_frac)
        neuron_sign = np.ones(N)
        neuron_sign[N_exc:] = -1.0

        # Input weights (per-neuron — fixes FPGA single-MAC problem!)
        W_in = rng.randn(N, n_inputs).astype(np.float32) * 0.5

        # Recurrent weights
        if connectivity == 'sparse':
            mask = (rng.rand(N, N) < 0.10)
            W = rng.randn(N, N).astype(np.float32) * mask
        elif connectivity == 'small_world':
            W = np.zeros((N, N), dtype=np.float32)
            for i in range(N):
                for k in [1, 2, 3, 4]:
                    W[i, (i+k) % N] = rng.randn() * 0.5
                    W[(i+k) % N, i] = rng.randn() * 0.5
                if rng.rand() < 0.08:
                    j = rng.randint(N)
                    W[i, j] = rng.randn()
        else:
            W = (rng.randn(N, N) / np.sqrt(N)).astype(np.float32)

        np.fill_diagonal(W, 0)
        W = np.abs(W) * neuron_sign[:, None]
        eigs = np.abs(np.linalg.eigvals(W))
        if eigs.max() > 0:
            W = (W * spectral_radius / eigs.max()).astype(np.float32)

        self.W_in = torch.tensor(W_in, device=DEVICE)
        self.W = torch.tensor(W, device=DEVICE)
        self.neuron_sign = torch.tensor(neuron_sign, dtype=torch.float32, device=DEVICE)

        # ─── Charge trapping (slide 17: SRH + VG2-dependent) ───
        self.k_cap_max = 1000.0
        self.k_em = 370.0
        self.Vth_max_shift = 0.5

        # Vcb pulse parameters (slide 20: self-oscillation)
        self.Vcb_amp = 2.5
        self.Vcb_period = 10e-6

    def k_cap_vg2(self):
        """VG2-dependent capture rate (slide 17: β depends on VG2)."""
        return self.k_cap_max / (1.0 + torch.exp((self.Vg2 - 0.40) / 0.05))

    @torch.no_grad()
    def run(self, inputs_np, noise_sigma=0.05, input_scale=1.0):
        """Run reservoir. inputs_np: (T,) or (T, n_inputs) numpy array.
        Returns: states (N, T), spikes (N, T) as numpy arrays."""
        if inputs_np.ndim == 1:
            inputs_np = inputs_np[:, None]
        T_steps = len(inputs_np)
        N = self.N
        dt = self.dt

        inputs = torch.tensor(inputs_np, dtype=torch.float32, device=self.device)

        # State
        Vm = torch.zeros(N, device=self.device)
        syn = torch.zeros(N, device=self.device)
        Q_trap = torch.zeros(N, device=self.device)
        refrac = torch.zeros(N, device=self.device)
        spike_rate = torch.zeros(N, device=self.device)

        states = torch.zeros(N, T_steps, device=self.device)
        spike_out = torch.zeros(N, T_steps, device=self.device)

        tau_syn = 5e-6
        syn_decay = np.exp(-dt / tau_syn)
        rate_decay = np.exp(-dt / 1e-3)
        Vt = 26e-3

        k_cap = self.k_cap_vg2()

        for t in range(T_steps):
            u = inputs[t]
            I_input = (self.W_in @ u) * input_scale

            step_spikes = torch.zeros(N, device=self.device)

            # KEY PHYSICS FIX: Input modulates Vg1 (gate voltage),
            # NOT injected as current. This is how real NS-RAM works:
            # VG1 controls avalanche sensitivity via BVpar(Vg1).
            # Higher Vg1 → lower BVpar → more avalanche current → more spikes.
            # Slide: VG1 range 0-0.425V
            effective_Vg1 = self.Vg1 + I_input * input_scale * 0.1  # ±0.1V modulation
            effective_Vg1 = torch.clamp(effective_Vg1, 0.0, 0.50)

            for sub in range(self.substeps):
                # Vcb self-oscillation (slide 20)
                global_phase = ((t * self.substeps + sub) * dt % self.Vcb_period) / self.Vcb_period
                if global_phase < 0.8:
                    Vcb = self.Vcb_amp * (global_phase / 0.8)
                else:
                    Vcb = self.Vcb_amp * (1.0 - (global_phase - 0.8) / 0.2)

                # Avalanche current (slide 17: Chynoweth)
                # BVpar depends on Vg1 — input MODULATES this!
                bvpar = self.BV0 - self.k_vg * effective_Vg1
                exp_arg = torch.clamp((Vcb - bvpar) / Vt, -20, 20)
                I_aval = self.I0 * torch.exp(exp_arg)
                I_aval = torch.clamp(I_aval, max=100e-6)

                # Leak
                I_leak = -self.g_leak * Vm

                # Synaptic recurrence (THE KEY: inter-neuron connections)
                # Synaptic current also modulates effective Vg1 of target neuron
                I_syn_current = (self.W.T @ syn) * 5e-9  # Stronger coupling

                # Noise (SDE)
                I_noise = noise_sigma * 1e-9 * torch.randn(N, device=self.device)

                # Charge trap modulation
                dQ = k_cap * (1.0 - Q_trap) * spike_rate - self.k_em * Q_trap
                Q_trap = torch.clamp(Q_trap + dQ * dt, 0, 1)
                delta_Vth = Q_trap * self.Vth_max_shift
                eff_thresh = torch.clamp(self.V_thresh_base - delta_Vth, min=0.1)

                # Integration: avalanche + leak + synaptic + noise
                active = (refrac <= 0).float()
                I_total = I_leak + I_aval + I_syn_current + I_noise
                Vm = Vm + active * (I_total / self.C_mem) * dt
                Vm = torch.clamp(Vm, -1.0, 5.0)

                # Spike detection
                spiked = (Vm >= eff_thresh) & (refrac <= 0)
                if spiked.any():
                    Vm[spiked] = Vm[spiked] * 0.3
                    refrac[spiked] = self.t_refrac[spiked]
                    syn[spiked] = syn[spiked] + 1.0
                    spike_rate[spiked] = spike_rate[spiked] + 1.0
                    step_spikes[spiked] = step_spikes[spiked] + 1.0

                syn = syn * syn_decay
                spike_rate = spike_rate * rate_decay
                refrac = torch.clamp(refrac - dt, min=0)

            # Record: membrane + spike info + trap charge
            states[:, t] = Vm + 0.5 * step_spikes + 0.2 * Q_trap
            spike_out[:, t] = step_spikes

        return states.cpu().numpy(), spike_out.cpu().numpy()


class SoftwareESN:
    """Properly implemented software ESN baseline (matches z2254j)."""

    def __init__(self, N=128, n_inputs=1, spectral_radius=1.05,
                 temp=0.65, leak=0.1, seed=42):
        self.N = N
        self.seed = seed
        rng = np.random.RandomState(seed)

        W = rng.randn(N, N).astype(np.float32) / np.sqrt(N)
        np.fill_diagonal(W, 0)
        eigs = np.abs(np.linalg.eigvals(W))
        self.W = torch.tensor((W * spectral_radius / eigs.max()).astype(np.float32),
                               device=DEVICE)
        self.W_in = torch.tensor(rng.randn(N, n_inputs).astype(np.float32) * 0.3,
                                  device=DEVICE)
        self.temp = temp
        self.leak = leak

    @torch.no_grad()
    def run(self, inputs_np, noise_sigma=0.0, input_scale=1.0):
        if inputs_np.ndim == 1:
            inputs_np = inputs_np[:, None]
        T = len(inputs_np)
        N = self.N
        inputs = torch.tensor(inputs_np, dtype=torch.float32, device=DEVICE)

        states = torch.zeros(N, T, device=DEVICE)
        v = torch.zeros(N, device=DEVICE)
        h = torch.zeros(N, device=DEVICE)

        for t in range(T):
            u = inputs[t]
            pre = (1 - self.leak) * v + self.W_in @ u * input_scale + self.W @ v
            v = torch.tanh(pre / self.temp)
            h = 0.93 * h + 0.07 * v
            states[:, t] = v + 0.3 * h

        return states.cpu().numpy(), np.zeros((N, T))


# ═══════════════════════════════════════════════════════════════════════
# RC BENCHMARKS (same as z2501 but using numpy on CPU results)
# ═══════════════════════════════════════════════════════════════════════

def ridge(X, y, alpha=1.0):
    XtX = X.T @ X + alpha * np.eye(X.shape[1])
    Xty = X.T @ y
    return np.linalg.solve(XtX, Xty)

def eval_xor(states, inputs, washout, tau):
    T = states.shape[1]
    half = washout + (T - washout) // 2
    X = states[:, washout+tau:].T
    y = ((inputs[washout+tau:] > 0) != (inputs[washout:T-tau] > 0)).astype(float)
    sp = half - washout - tau
    if sp < 20 or len(y) - sp < 20: return 0.5
    w = ridge(X[:sp], y[:sp])
    pred = X[sp:] @ w
    acc = ((pred > 0.5) == (y[sp:] > 0.5)).mean()
    return max(acc, 1 - acc)

def eval_mc(states, inputs, washout, max_d=15):
    T = states.shape[1]
    half = washout + (T - washout) // 2
    mc = 0.0
    for d in range(1, max_d + 1):
        X = states[:, washout+d:].T
        y = inputs[washout:T-d]
        sp = half - washout - d
        if sp < 20 or len(y) - sp < 20: continue
        w = ridge(X[:sp], y[:sp])
        pred = X[sp:] @ w
        yt = y[sp:]
        if np.std(yt) < 1e-10 or np.std(pred) < 1e-10: continue
        r = np.corrcoef(pred, yt)[0, 1]
        mc += r**2
    return mc

def eval_narma(states, inputs, washout, order=5):
    T = min(states.shape[1], len(inputs))
    y = np.zeros(T)
    u = (inputs[:T] + 1) / 2 * 0.5
    for t in range(order, T):
        y[t] = 0.3*y[t-1] + 0.05*y[t-1]*np.sum(y[t-order:t]) + 1.5*u[t-1]*u[t-order] + 0.1
        y[t] = np.tanh(y[t])
    half = washout + (T - washout) // 2
    X = states[:, washout:T].T
    yt = y[washout:T]
    sp = half - washout
    if sp < 20 or len(yt) - sp < 20: return 0.0
    w = ridge(X[:sp], yt[:sp])
    pred = X[sp:] @ w
    yt2 = yt[sp:]
    ss_res = np.sum((yt2 - pred)**2)
    ss_tot = np.sum((yt2 - yt2.mean())**2)
    return max(0, 1 - ss_res / ss_tot) if ss_tot > 0 else 0

def eval_wave(states, inputs, washout, nc=4):
    T = states.shape[1]
    half = washout + (T - washout) // 2
    bounds = np.linspace(-1, 1, nc + 1)
    labels = np.digitize(inputs[:T], bounds[1:-1])
    X = states[:, washout:T].T
    yl = labels[washout:T]
    sp = half - washout
    preds = np.zeros((T - washout - sp, nc))
    for c in range(nc):
        yc = (yl == c).astype(float)
        w = ridge(X[:sp], yc[:sp])
        preds[:, c] = X[sp:] @ w
    return (np.argmax(preds, axis=1) == yl[sp:]).mean()


def run_bench(name, res, inputs, washout=200, n_reps=3, input_scale=1.0):
    results = []
    for rep in range(n_reps):
        t0 = time.time()
        if hasattr(res, 'rng'):
            res.rng = np.random.RandomState(res.seed + rep * 1000)

        states, spikes = res.run(inputs, noise_sigma=0.05, input_scale=input_scale)
        elapsed = time.time() - t0

        xor1 = eval_xor(states, inputs, washout, 1)
        xor2 = eval_xor(states, inputs, washout, 2)
        xor5 = eval_xor(states, inputs, washout, 5)
        mc = eval_mc(states, inputs, washout)
        narma = eval_narma(states, inputs, washout)
        wave = eval_wave(states, inputs, washout)
        active = int((spikes.sum(axis=1) > 0).sum())
        tot_spk = int(spikes.sum())

        r = {'xor1': xor1, 'xor2': xor2, 'xor5': xor5,
             'mc': mc, 'narma': narma, 'wave4': wave,
             'active': active, 'spikes': tot_spk, 'time': elapsed}
        results.append(r)

        print(f"  [{name}] r{rep}: XOR1={xor1:.1%} XOR2={xor2:.1%} XOR5={xor5:.1%} "
              f"MC={mc:.3f} NARMA={narma:.3f} W4={wave:.1%} "
              f"({active}N, {tot_spk} spk, {elapsed:.1f}s)")

    avg = {k: np.mean([r[k] for r in results]) for k in results[0]}
    return {'name': name, 'reps': results, 'avg': avg}


def main():
    print("=" * 75)
    print("  z2502: GPU-Accelerated NS-RAM ODE Reservoir")
    print("  Physics: Pazos/Lanza semi-empirical model (slides 17, 20-23)")
    print("=" * 75)

    steps = 1500
    washout = 300
    rng = np.random.RandomState(42)
    inputs = rng.uniform(-1, 1, steps).astype(np.float64)

    ALL = {}

    # ─── Baselines ───
    print("\n━━━ Software ESN Baselines ━━━")
    esn = SoftwareESN(N=128, spectral_radius=1.05, temp=0.65, leak=0.10)
    ALL['ESN_128'] = run_bench("ESN_128 (z2254j)", esn, inputs, washout)

    esn64 = SoftwareESN(N=64, spectral_radius=1.05, temp=0.65, leak=0.10)
    ALL['ESN_64'] = run_bench("ESN_64", esn64, inputs, washout)

    # ─── NS-RAM configurations ───
    print("\n━━━ NS-RAM ODE Reservoir (slide-matched physics) ━━━")

    configs = [
        # name, N, connectivity, sr, variability, substeps, input_scale
        ("NSRAM_64_sparse",     64,  'sparse',     0.95, 0.10, 10, 2.0),
        ("NSRAM_128_sparse",    128, 'sparse',     0.95, 0.10, 10, 2.0),
        ("NSRAM_128_sw",        128, 'small_world', 0.95, 0.10, 10, 2.0),
        ("NSRAM_128_novar",     128, 'sparse',     0.95, 0.00, 10, 2.0),
        ("NSRAM_128_highvar",   128, 'sparse',     0.95, 0.20, 10, 2.0),
        ("NSRAM_128_sr105",     128, 'sparse',     1.05, 0.10, 10, 2.0),
        ("NSRAM_128_dense",     128, 'dense',      0.95, 0.10, 10, 2.0),
        ("NSRAM_128_input5x",   128, 'sparse',     0.95, 0.10, 10, 5.0),
        ("NSRAM_128_input10x",  128, 'sparse',     0.95, 0.10, 10, 10.0),
        ("NSRAM_128_sub20",     128, 'sparse',     0.95, 0.10, 20, 2.0),
        ("NSRAM_256_sparse",    256, 'sparse',     0.95, 0.10, 10, 2.0),
    ]

    for name, N, conn, sr, var, sub, iscale in configs:
        res = NSRAMReservoirGPU(N=N, n_inputs=1, connectivity=conn,
                                 spectral_radius=sr, seed=42,
                                 variability=var, substeps=sub)
        ALL[name] = run_bench(name, res, inputs, washout, n_reps=3, input_scale=iscale)

    # ─── Summary ───
    print("\n" + "=" * 105)
    print(f"  {'Config':<28s}  {'XOR-1':>6s}  {'XOR-2':>6s}  {'XOR-5':>6s}  "
          f"{'MC':>6s}  {'NARMA':>6s}  {'Wave4':>6s}  {'Active':>6s}  {'Time':>5s}")
    print("=" * 105)

    for name, r in ALL.items():
        a = r['avg']
        print(f"  {name:<28s}  {a['xor1']:>5.1%}  {a['xor2']:>5.1%}  {a['xor5']:>5.1%}  "
              f"{a['mc']:>6.3f}  {a['narma']:>6.3f}  {a['wave4']:>5.1%}  "
              f"{a['active']:>6.0f}  {a['time']:>5.1f}s")

    # Save
    out = os.path.join(os.path.dirname(__file__), '..', 'results', 'z2502_nsram_gpu_reservoir.json')
    def ser(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating, np.float64)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    with open(out, 'w') as f:
        json.dump({k: {'avg': {kk: ser(vv) for kk, vv in v['avg'].items()},
                        'reps': [{kk: ser(vv) for kk, vv in rep.items()} for rep in v['reps']]}
                   for k, v in ALL.items()}, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
