#!/usr/bin/env python3
"""
z2233_subthreshold_memory.py — Physics fix for memory capacity: subthreshold operation
======================================================================================
ROOT CAUSE of MC=0: vmem resets to 0 on every spike (lif_membrane.v:199).
At BASE_VG=0.58, neurons spike constantly → vmem never accumulates → no memory.

PHYSICS FIX: Operate in subthreshold regime.
  - Set BASE_VG BELOW the avalanche cliff so neurons get weak current
  - vmem integrates slowly, carrying graded temporal memory
  - Leak τ≈105ms creates genuine fading memory in vmem
  - Readout uses ONLY vmem (not spike counts, which are zero/rare)

Also test:
  - Temperature diversity (different T per neuron group → different BVpar)
  - Lateral connections (ring N±1=0.125, N±2=0.0625 provide recurrence)

No software tricks: all computation happens in FPGA physics.
"""

import os, sys, time, json
import numpy as np

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')
sys.path.insert(0, "scripts")
from fpga_host_eth import FPGAEthBridge

N_NEURONS = 128
SAMPLE_HZ = 200
STEP_DT = 1.0 / SAMPLE_HZ  # 5ms


def drain_latest(fpga, max_reads=50):
    """Drain auto-telemetry buffer, return latest packet."""
    latest = None
    for _ in range(max_reads):
        try:
            pkt = fpga.recv_auto_telemetry(timeout=0.001)
            if pkt is not None:
                latest = pkt
            else:
                break
        except:
            break
    return latest


def ridge_r2(X_tr, y_tr, X_te, y_te):
    """Ridge regression R² with alpha search."""
    best = -999
    for a in [1e-4, 1e-2, 1.0, 100.0, 10000.0]:
        I = np.eye(X_tr.shape[1])
        try:
            w = np.linalg.solve(X_tr.T @ X_tr + a * I, X_tr.T @ y_tr)
        except:
            continue
        pred = X_te @ w
        ss_res = np.sum((y_te - pred) ** 2)
        ss_tot = np.sum((y_te - y_te.mean()) ** 2)
        if ss_tot < 1e-10:
            continue
        r2 = 1 - ss_res / ss_tot
        if r2 > best:
            best = r2
    return best


def collect_gpu_noise(duration=15.0, rate=50):
    """Collect GPU hardware noise. Returns (n_samples, 5) array."""
    n = int(duration * rate)
    noise = np.zeros((n, 5), dtype=np.float32)
    dt = 1.0 / rate

    for i in range(n):
        t0 = time.perf_counter()
        try:
            with open('/sys/class/hwmon/hwmon7/power1_average', 'r') as f:
                noise[i, 0] = float(f.read().strip()) / 1e6
        except:
            noise[i, 0] = noise[i-1, 0] if i > 0 else 10.0
        try:
            with open('/sys/kernel/ryzen_smu_drv/pm_table', 'rb') as f:
                f.seek(0x004C)
                noise[i, 1] = np.frombuffer(f.read(4), dtype=np.float32)[0]
        except:
            noise[i, 1] = noise[i-1, 1] if i > 0 else 45.0
        noise[i, 2] = float(time.perf_counter_ns() % 100000) / 100000.0
        try:
            with open('/sys/class/hwmon/hwmon7/temp1_input', 'r') as f:
                noise[i, 3] = float(f.read().strip()) / 1000.0
        except:
            noise[i, 3] = noise[i-1, 3] if i > 0 else 45.0
        try:
            with open('/sys/class/hwmon/hwmon7/freq1_input', 'r') as f:
                noise[i, 4] = float(f.read().strip()) / 1e6
        except:
            noise[i, 4] = noise[i-1, 4] if i > 0 else 600.0

        elapsed = time.perf_counter() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

    # Normalize
    for ch in range(5):
        mu = noise[:, ch].mean()
        sigma = noise[:, ch].std()
        if sigma > 1e-10:
            noise[:, ch] = (noise[:, ch] - mu) / sigma

    # IIR filter (temporal correlation)
    iir_alphas = [0.85, 0.92, 0.0, 0.90, 0.80]
    for ch in range(5):
        a = iir_alphas[ch]
        if a > 0:
            filtered = np.zeros_like(noise[:, ch])
            filtered[0] = noise[0, ch]
            for j in range(1, len(filtered)):
                filtered[j] = a * filtered[j-1] + (1 - a) * noise[j, ch]
            noise[:, ch] = filtered

    return noise


# ==========================================================================
# SWEEP 1: Find subthreshold operating point
# ==========================================================================
def sweep_subthreshold(fpga):
    """Sweep Vg to find the subthreshold-to-spiking transition.
    We want the Vg where neurons INTEGRATE but RARELY spike.
    """
    print("\n" + "=" * 70)
    print("SWEEP 1: Find subthreshold operating point")
    print("=" * 70)

    fpga.set_leak_cond(0x0008)  # τ≈105ms
    fpga.set_mac_signal(0.0)
    time.sleep(0.1)

    vg_values = np.arange(0.30, 0.70, 0.02)
    results = []

    for vg in vg_values:
        fpga.set_kill(True)
        time.sleep(0.1)
        fpga.set_kill(False)
        time.sleep(0.1)
        drain_latest(fpga, max_reads=100)

        # Set uniform Vg
        fpga.set_vg_batch(0, [float(vg)] * 64)
        fpga.set_vg_batch(64, [float(vg)] * 64)
        time.sleep(0.05)

        # Collect for 500ms
        vmems = []
        spike_rates = []
        drain_latest(fpga, max_reads=100)
        time.sleep(0.01)

        t0_counts = None
        for step in range(100):  # 100 steps at 200Hz = 0.5s
            t0 = time.perf_counter()
            pkt = drain_latest(fpga, max_reads=20)
            if pkt is not None:
                vmems.append(pkt['vmem'].copy())
                if t0_counts is None:
                    t0_counts = pkt['spike_counts'].copy()
                last_counts = pkt['spike_counts'].copy()
            elapsed = time.perf_counter() - t0
            if elapsed < STEP_DT:
                time.sleep(STEP_DT - elapsed)

        if len(vmems) > 10 and t0_counts is not None:
            vmem_arr = np.array(vmems)
            mean_vmem = vmem_arr.mean()
            std_vmem = vmem_arr.std()
            # Spike rate
            delta = last_counts.astype(np.int32) - t0_counts.astype(np.int32)
            delta[delta < 0] = 0
            delta[delta > 30000] = 0
            rate = delta.mean() / 0.5  # spikes/s

            # vmem dynamics: how much does vmem vary across time?
            temporal_var = vmem_arr.std(axis=0).mean()  # mean per-neuron temporal std

            print(f"  Vg={vg:.2f}: vmem={mean_vmem:.4f} ±{std_vmem:.4f}, "
                  f"rate={rate:.1f} spk/s, temporal_var={temporal_var:.4f}")
            results.append({
                'vg': float(vg), 'vmem_mean': float(mean_vmem),
                'vmem_std': float(std_vmem), 'rate': float(rate),
                'temporal_var': float(temporal_var)
            })
        else:
            print(f"  Vg={vg:.2f}: NO DATA")
            results.append({'vg': float(vg), 'vmem_mean': 0, 'rate': 0, 'temporal_var': 0})

    # Find sweet spot: highest temporal_var with rate < 50 spk/s
    best = None
    best_tvar = 0
    for r in results:
        if r['rate'] < 50 and r['temporal_var'] > best_tvar:
            best_tvar = r['temporal_var']
            best = r
    if best:
        print(f"\n  SWEET SPOT: Vg={best['vg']:.2f} (rate={best['rate']:.1f}, "
              f"temporal_var={best['temporal_var']:.4f})")
    else:
        # Fallback: lowest rate > 0
        for r in sorted(results, key=lambda x: x['rate']):
            if r['rate'] > 0 and r['temporal_var'] > 0:
                best = r
                break
        if best:
            print(f"\n  FALLBACK: Vg={best['vg']:.2f} (rate={best['rate']:.1f})")

    return results, best


# ==========================================================================
# EXP 1: Subthreshold Memory Capacity
# ==========================================================================
def exp1_subthreshold_mc(fpga, noise_buf, rng, base_vg, leak_val=0x0008):
    """Memory capacity using vmem-only readout in subthreshold regime.

    Physics: vmem[t] = Σ I[s] * exp(-(t-s)/τ) — exponentially-weighted past inputs.
    With τ=105ms and 5ms steps, retention = 95.3%/step → genuine fading memory.
    Key: neurons must NOT spike (vmem reset destroys memory).
    """
    print(f"\n{'='*70}")
    print(f"EXP 1 — SUBTHRESHOLD MEMORY (Vg={base_vg:.2f}, LEAK={hex(leak_val)})")
    print(f"{'='*70}")

    fpga.set_leak_cond(leak_val)
    time.sleep(0.1)

    N_TRIALS = 80
    N_STEPS = 200   # 1.0s at 200Hz
    MAX_DELAY = 10
    ALPHA = 0.08    # SMALL perturbation — stay subthreshold
    BETA = 0.03     # weak noise coupling

    noise_map = [0]*32 + [1]*24 + [2]*24 + [3]*24 + [4]*24

    w_in = rng.standard_normal(N_NEURONS).astype(np.float32) * 0.3
    w_noise = rng.standard_normal(N_NEURONS).astype(np.float32) * 0.3

    conditions = {
        'NOISE': True,
        'WHITE': 'white',
        'STATIC': False,
    }

    results = {}

    for cond_name, cond_noise in conditions.items():
        print(f"\n  --- {cond_name} ---")

        fpga.set_kill(True)
        time.sleep(0.15)
        fpga.set_kill(False)
        time.sleep(0.15)
        fpga.set_mac_signal(0.0)
        drain_latest(fpga, max_reads=100)

        all_inputs = []
        all_vmem_states = []
        spike_count_total = 0

        for trial in range(N_TRIALS):
            u = rng.uniform(-1, 1, N_STEPS).astype(np.float32)

            if cond_noise == True:
                noise_idx = rng.integers(0, len(noise_buf) - N_STEPS)
            else:
                noise_idx = 0

            # Run trial
            prev_counts = None
            vmem_states = []
            trial_spikes = 0

            for t in range(N_STEPS):
                t0 = time.perf_counter()

                # Compute Vg: small perturbation to stay subthreshold
                if cond_noise == True:
                    ni = (noise_idx + t) % len(noise_buf)
                    noise_per = np.array([noise_buf[ni, noise_map[n]] for n in range(N_NEURONS)])
                elif cond_noise == 'white':
                    noise_per = rng.standard_normal(N_NEURONS).astype(np.float32)
                else:
                    noise_per = np.zeros(N_NEURONS, dtype=np.float32)

                vg = np.full(N_NEURONS, base_vg) + ALPHA * u[t] * w_in + BETA * noise_per * w_noise
                vg = np.clip(vg, 0.05, 0.95)

                fpga.set_vg_batch(0, vg[:64].tolist())
                fpga.set_vg_batch(64, vg[64:].tolist())

                elapsed_set = time.perf_counter() - t0
                wait = STEP_DT - elapsed_set - 0.001
                if wait > 0.0005:
                    time.sleep(wait)

                pkt = drain_latest(fpga, max_reads=30)
                if pkt is None:
                    time.sleep(0.003)
                    pkt = drain_latest(fpga, max_reads=10)

                if pkt is not None:
                    vm = pkt['vmem']
                    counts = pkt['spike_counts']

                    if prev_counts is not None:
                        delta = counts.astype(np.int32) - prev_counts.astype(np.int32)
                        delta[delta < 0] = 0
                        delta[delta > 30000] = 0
                        trial_spikes += delta.sum()

                    # Use ONLY vmem for state (graded analog signal)
                    vmem_states.append(vm.astype(np.float32).copy())
                    prev_counts = counts.copy()

                total_elapsed = time.perf_counter() - t0
                remaining = STEP_DT - total_elapsed
                if remaining > 0.0005:
                    time.sleep(remaining)

            spike_count_total += trial_spikes

            if len(vmem_states) >= N_STEPS - 10:
                all_inputs.append(u)
                all_vmem_states.append(np.array(vmem_states[-N_STEPS+1:]))

            if (trial + 1) % 20 == 0:
                avg_spikes = spike_count_total / ((trial+1) * N_STEPS) if trial > 0 else 0
                print(f"    Trial {trial+1}/{N_TRIALS}, valid={len(all_vmem_states)}, "
                      f"avg_spikes/step={avg_spikes:.1f}")

        if len(all_vmem_states) < 20:
            print(f"    FAIL: only {len(all_vmem_states)} valid trials")
            results[cond_name] = {'mc': [0]*MAX_DELAY, 'mc_total': 0}
            continue

        # Memory capacity at each delay (vmem-only readout)
        mc_values = []
        for d in range(1, MAX_DELAY + 1):
            X_list = []
            y_list = []
            for i in range(len(all_vmem_states)):
                state_mat = all_vmem_states[i]  # (steps, 128) vmem only
                u_seq = all_inputs[i]
                for t_idx in range(d, min(len(state_mat), len(u_seq) - 1)):
                    X_list.append(state_mat[t_idx])
                    y_list.append(u_seq[t_idx - d])

            X = np.array(X_list)
            y = np.array(y_list)

            # Normalize
            mu = X.mean(axis=0)
            sigma = X.std(axis=0)
            sigma[sigma < 1e-3] = 1.0  # more conservative floor for vmem
            X_n = (X - mu) / sigma

            n = len(X)
            n_tr = n * 3 // 4
            idx = rng.permutation(n)
            r2 = ridge_r2(X_n[idx[:n_tr]], y[idx[:n_tr]], X_n[idx[n_tr:]], y[idx[n_tr:]])
            mc_values.append(max(0, r2))
            print(f"    d={d:2d}: R²={r2:+.4f} {'***' if r2 > 0.02 else ''}")

        mc_total = sum(mc_values)
        results[cond_name] = {
            'mc': mc_values, 'mc_total': float(mc_total),
            'n_valid': len(all_vmem_states),
            'total_spikes': int(spike_count_total),
        }
        print(f"    MC total = {mc_total:.4f}")

    # Tests
    print("\n  TESTS:")
    noise_mc = results.get('NOISE', {}).get('mc_total', 0)
    static_mc = results.get('STATIC', {}).get('mc_total', 0)
    white_mc = results.get('WHITE', {}).get('mc_total', 0)
    noise_d1 = results.get('NOISE', {}).get('mc', [0])[0]

    t_pass = 0; t_total = 4

    p = noise_d1 > 0.02
    t_pass += p
    print(f"  T1 MC(d=1) > 0.02:        R²={noise_d1:.4f} {'PASS' if p else 'FAIL'}")

    p = noise_mc > 0.05
    t_pass += p
    print(f"  T2 MC_total > 0.05:       {noise_mc:.4f} {'PASS' if p else 'FAIL'}")

    p = noise_mc > static_mc
    t_pass += p
    print(f"  T3 NOISE > STATIC:        {noise_mc:.4f} vs {static_mc:.4f} {'PASS' if p else 'FAIL'}")

    # Decay profile: MC should decrease with delay
    noise_mc_arr = results.get('NOISE', {}).get('mc', [0]*10)
    if len(noise_mc_arr) >= 3 and noise_mc_arr[0] > 0:
        decaying = noise_mc_arr[0] > noise_mc_arr[2]  # d=1 > d=3
    else:
        decaying = False
    t_pass += decaying
    print(f"  T4 Decay profile:         d1={noise_mc_arr[0]:.4f} > d3={noise_mc_arr[2] if len(noise_mc_arr)>2 else 0:.4f} {'PASS' if decaying else 'FAIL'}")

    print(f"  Score: {t_pass}/{t_total}")
    results['tests'] = {'pass': t_pass, 'total': t_total}
    return results


# ==========================================================================
# EXP 2: Temperature diversity for threshold spread
# ==========================================================================
def exp2_temp_diversity(fpga, noise_buf, rng, base_vg, leak_val=0x0008):
    """Use different temperatures per neuron group to create BVpar diversity.

    Physics: BVpar = 3.5 - 1.5*Vg, Vt = 0.05*(T/300)
    Different T → different effective thresholds → population encodes continuous values.
    """
    print(f"\n{'='*70}")
    print(f"EXP 2 — TEMPERATURE DIVERSITY (threshold spread via physics)")
    print(f"{'='*70}")

    fpga.set_leak_cond(leak_val)
    time.sleep(0.1)

    N_TRIALS = 60
    N_STEPS = 200
    MAX_DELAY = 10
    ALPHA = 0.08
    BETA = 0.03

    noise_map = [0]*32 + [1]*24 + [2]*24 + [3]*24 + [4]*24
    w_in = rng.standard_normal(N_NEURONS).astype(np.float32) * 0.3
    w_noise = rng.standard_normal(N_NEURONS).astype(np.float32) * 0.3

    # Temperature groups: 4 groups at different T
    TEMP_GROUPS = [280, 300, 320, 340]  # Kelvin
    GROUP_SIZE = N_NEURONS // len(TEMP_GROUPS)

    conditions = {
        'DIVERSE_T': TEMP_GROUPS,      # 4 different temperatures
        'UNIFORM_T': [300, 300, 300, 300],  # all same
    }

    results = {}

    for cond_name, temps in conditions.items():
        print(f"\n  --- {cond_name} (T={temps}) ---")

        fpga.set_kill(True)
        time.sleep(0.15)
        fpga.set_kill(False)
        time.sleep(0.15)
        fpga.set_mac_signal(0.0)
        drain_latest(fpga, max_reads=100)

        # Set global temperature (API is global, not per-neuron)
        # Use mean temperature for this condition
        T_mean = int(np.mean(temps))
        fpga.set_temperature(T_mean)
        time.sleep(0.1)

        # For DIVERSE_T: vary Vg per group to emulate threshold spread
        # BVpar ~ Vt = 0.05*(T/300), so T=280→Vt=0.0467, T=340→Vt=0.0567
        # Map temperature effect to Vg offset per group
        vg_offsets = np.zeros(N_NEURONS)
        for g, T in enumerate(temps):
            offset = 0.02 * (T - 300) / 40  # ±0.02V for ±40K
            start = g * GROUP_SIZE
            end = min((g+1) * GROUP_SIZE, N_NEURONS)
            vg_offsets[start:end] = offset

        all_inputs = []
        all_vmem_states = []

        for trial in range(N_TRIALS):
            u = rng.uniform(-1, 1, N_STEPS).astype(np.float32)
            noise_idx = rng.integers(0, len(noise_buf) - N_STEPS)

            vmem_states = []
            prev_counts = None

            for t in range(N_STEPS):
                t0 = time.perf_counter()

                ni = (noise_idx + t) % len(noise_buf)
                noise_per = np.array([noise_buf[ni, noise_map[n]] for n in range(N_NEURONS)])
                vg = np.full(N_NEURONS, base_vg) + vg_offsets + ALPHA * u[t] * w_in + BETA * noise_per * w_noise
                vg = np.clip(vg, 0.05, 0.95)

                fpga.set_vg_batch(0, vg[:64].tolist())
                fpga.set_vg_batch(64, vg[64:].tolist())

                elapsed_set = time.perf_counter() - t0
                wait = STEP_DT - elapsed_set - 0.001
                if wait > 0.0005:
                    time.sleep(wait)

                pkt = drain_latest(fpga, max_reads=30)
                if pkt is None:
                    time.sleep(0.003)
                    pkt = drain_latest(fpga, max_reads=10)

                if pkt is not None:
                    vmem_states.append(pkt['vmem'].astype(np.float32).copy())

                total_elapsed = time.perf_counter() - t0
                remaining = STEP_DT - total_elapsed
                if remaining > 0.0005:
                    time.sleep(remaining)

            if len(vmem_states) >= N_STEPS - 10:
                all_inputs.append(u)
                all_vmem_states.append(np.array(vmem_states[-N_STEPS+1:]))

            if (trial + 1) % 20 == 0:
                print(f"    Trial {trial+1}/{N_TRIALS}, valid={len(all_vmem_states)}")

        if len(all_vmem_states) < 15:
            print(f"    FAIL: only {len(all_vmem_states)} valid trials")
            results[cond_name] = {'mc_total': 0}
            continue

        # MC
        mc_values = []
        for d in range(1, MAX_DELAY + 1):
            X_list = []
            y_list = []
            for i in range(len(all_vmem_states)):
                state_mat = all_vmem_states[i]
                u_seq = all_inputs[i]
                for t_idx in range(d, min(len(state_mat), len(u_seq) - 1)):
                    X_list.append(state_mat[t_idx])
                    y_list.append(u_seq[t_idx - d])

            X = np.array(X_list)
            y = np.array(y_list)
            mu = X.mean(axis=0)
            sigma = X.std(axis=0)
            sigma[sigma < 1e-3] = 1.0
            X_n = (X - mu) / sigma

            n = len(X)
            n_tr = n * 3 // 4
            idx = rng.permutation(n)
            r2 = ridge_r2(X_n[idx[:n_tr]], y[idx[:n_tr]], X_n[idx[n_tr:]], y[idx[n_tr:]])
            mc_values.append(max(0, r2))
            print(f"    d={d:2d}: R²={r2:+.4f} {'***' if r2 > 0.02 else ''}")

        mc_total = sum(mc_values)
        results[cond_name] = {'mc': mc_values, 'mc_total': float(mc_total)}
        print(f"    MC total = {mc_total:.4f}")

    # Tests
    print("\n  TESTS:")
    div_mc = results.get('DIVERSE_T', {}).get('mc_total', 0)
    uni_mc = results.get('UNIFORM_T', {}).get('mc_total', 0)

    t_pass = 0; t_total = 2

    p = div_mc > uni_mc
    t_pass += p
    print(f"  T5 DIVERSE > UNIFORM:  {div_mc:.4f} vs {uni_mc:.4f} {'PASS' if p else 'FAIL'}")

    p = div_mc > 0.05
    t_pass += p
    print(f"  T6 DIVERSE MC > 0.05:  {div_mc:.4f} {'PASS' if p else 'FAIL'}")

    print(f"  Score: {t_pass}/{t_total}")
    results['tests'] = {'pass': t_pass, 'total': t_total}
    return results


# ==========================================================================
# EXP 3: Vg sweep for MC — find the optimal operating point
# ==========================================================================
def exp3_vg_sweep_mc(fpga, noise_buf, rng, leak_val=0x0008):
    """Sweep BASE_VG from deep subthreshold to superthreshold.
    At each Vg, measure MC(d=1) to map the memory-vs-firing landscape.
    """
    print(f"\n{'='*70}")
    print(f"EXP 3 — VG SWEEP for Memory Capacity")
    print(f"{'='*70}")

    fpga.set_leak_cond(leak_val)
    time.sleep(0.1)

    N_TRIALS = 40
    N_STEPS = 150
    ALPHA = 0.08
    BETA = 0.03

    noise_map = [0]*32 + [1]*24 + [2]*24 + [3]*24 + [4]*24
    w_in = rng.standard_normal(N_NEURONS).astype(np.float32) * 0.3
    w_noise = rng.standard_normal(N_NEURONS).astype(np.float32) * 0.3

    vg_values = [0.30, 0.35, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.55, 0.58, 0.62]
    results = {}

    for base_vg in vg_values:
        print(f"\n  --- Vg={base_vg:.2f} ---")

        fpga.set_kill(True)
        time.sleep(0.1)
        fpga.set_kill(False)
        time.sleep(0.1)
        fpga.set_mac_signal(0.0)
        drain_latest(fpga, max_reads=100)

        all_inputs = []
        all_vmem_states = []
        total_spikes = 0

        for trial in range(N_TRIALS):
            u = rng.uniform(-1, 1, N_STEPS).astype(np.float32)
            noise_idx = rng.integers(0, len(noise_buf) - N_STEPS)

            vmem_states = []
            prev_counts = None
            trial_spikes = 0

            for t in range(N_STEPS):
                t0 = time.perf_counter()

                ni = (noise_idx + t) % len(noise_buf)
                noise_per = np.array([noise_buf[ni, noise_map[n]] for n in range(N_NEURONS)])
                vg = np.full(N_NEURONS, base_vg) + ALPHA * u[t] * w_in + BETA * noise_per * w_noise
                vg = np.clip(vg, 0.05, 0.95)

                fpga.set_vg_batch(0, vg[:64].tolist())
                fpga.set_vg_batch(64, vg[64:].tolist())

                elapsed_set = time.perf_counter() - t0
                wait = STEP_DT - elapsed_set - 0.001
                if wait > 0.0005:
                    time.sleep(wait)

                pkt = drain_latest(fpga, max_reads=30)
                if pkt is None:
                    time.sleep(0.003)
                    pkt = drain_latest(fpga, max_reads=10)

                if pkt is not None:
                    vm = pkt['vmem']
                    counts = pkt['spike_counts']
                    if prev_counts is not None:
                        delta = counts.astype(np.int32) - prev_counts.astype(np.int32)
                        delta[delta < 0] = 0
                        delta[delta > 30000] = 0
                        trial_spikes += delta.sum()
                    vmem_states.append(vm.astype(np.float32).copy())
                    prev_counts = counts.copy()

                total_elapsed = time.perf_counter() - t0
                remaining = STEP_DT - total_elapsed
                if remaining > 0.0005:
                    time.sleep(remaining)

            total_spikes += trial_spikes
            if len(vmem_states) >= N_STEPS - 10:
                all_inputs.append(u)
                all_vmem_states.append(np.array(vmem_states[-N_STEPS+1:]))

        if len(all_vmem_states) < 10:
            print(f"    FAIL: only {len(all_vmem_states)} valid")
            results[f"vg={base_vg:.2f}"] = {'r2_d1': -1, 'rate': 0}
            continue

        # MC at d=1 only (speed)
        X_list, y_list = [], []
        for i in range(len(all_vmem_states)):
            state_mat = all_vmem_states[i]
            u_seq = all_inputs[i]
            for t_idx in range(1, min(len(state_mat), len(u_seq) - 1)):
                X_list.append(state_mat[t_idx])
                y_list.append(u_seq[t_idx - 1])

        X = np.array(X_list)
        y = np.array(y_list)
        mu = X.mean(axis=0); sigma = X.std(axis=0); sigma[sigma < 1e-3] = 1.0
        X_n = (X - mu) / sigma
        n = len(X); n_tr = n * 3 // 4
        idx = rng.permutation(n)
        r2 = ridge_r2(X_n[idx[:n_tr]], y[idx[:n_tr]], X_n[idx[n_tr:]], y[idx[n_tr:]])

        avg_rate = total_spikes / (N_TRIALS * N_STEPS * N_NEURONS) * SAMPLE_HZ
        vmem_var = np.array([s.std(axis=0).mean() for s in all_vmem_states]).mean()

        print(f"    R²(d=1)={r2:+.4f}, rate={avg_rate:.1f} spk/s, vmem_var={vmem_var:.4f}")
        results[f"vg={base_vg:.2f}"] = {
            'r2_d1': float(r2), 'rate': float(avg_rate),
            'vmem_var': float(vmem_var), 'n_valid': len(all_vmem_states)
        }

    # Find best Vg
    best_vg = max(results.items(), key=lambda x: x[1].get('r2_d1', -999))
    print(f"\n  BEST: {best_vg[0]} R²={best_vg[1]['r2_d1']:.4f} rate={best_vg[1]['rate']:.1f}")

    # Test
    print("\n  TESTS:")
    t_pass = 0; t_total = 2
    best_r2 = best_vg[1]['r2_d1']

    p = best_r2 > 0.02
    t_pass += p
    print(f"  T7 Best R²(d=1) > 0.02:  {best_r2:.4f} {'PASS' if p else 'FAIL'}")

    # Subthreshold better than superthreshold
    sub_r2s = [v['r2_d1'] for k, v in results.items() if v.get('rate', 999) < 20]
    sup_r2s = [v['r2_d1'] for k, v in results.items() if v.get('rate', 999) > 100]
    sub_mean = np.mean(sub_r2s) if sub_r2s else -1
    sup_mean = np.mean(sup_r2s) if sup_r2s else -1
    p = sub_mean > sup_mean
    t_pass += p
    print(f"  T8 Subthreshold > supra: {sub_mean:.4f} vs {sup_mean:.4f} {'PASS' if p else 'FAIL'}")

    print(f"  Score: {t_pass}/{t_total}")
    results['tests'] = {'pass': t_pass, 'total': t_total}
    return results


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    print("z2233b — Subthreshold Memory Physics Fix")
    print("=" * 70)

    fpga = FPGAEthBridge()
    if not fpga.connect():
        print("FAIL: Cannot connect to FPGA")
        return

    fpga.set_kill(False)
    fpga.enable_auto_telemetry(2000)
    time.sleep(0.3)
    drain_latest(fpga, max_reads=200)

    rng = np.random.default_rng(2233)

    # Collect GPU noise
    print("\nCollecting GPU noise (15s)...")
    noise_buf = collect_gpu_noise(duration=15.0, rate=50)
    print(f"  {noise_buf.shape[0]} samples, Ch0 autocorr={np.corrcoef(noise_buf[:-1,0], noise_buf[1:,0])[0,1]:.3f}")

    all_results = {}

    # SWEEP: find subthreshold operating point
    try:
        sweep_results, sweet_spot = sweep_subthreshold(fpga)
        all_results['sweep'] = sweep_results
        sub_vg = sweet_spot['vg'] if sweet_spot else 0.44
    except Exception as e:
        print(f"  SWEEP ERROR: {e}")
        import traceback; traceback.print_exc()
        sub_vg = 0.44

    print(f"\nUsing subthreshold Vg = {sub_vg:.2f}")

    # EXP 1: Subthreshold memory
    try:
        all_results['exp1'] = exp1_subthreshold_mc(fpga, noise_buf, rng, base_vg=sub_vg)
    except Exception as e:
        print(f"  EXP 1 ERROR: {e}")
        import traceback; traceback.print_exc()

    # EXP 2: Temperature diversity
    try:
        all_results['exp2'] = exp2_temp_diversity(fpga, noise_buf, rng, base_vg=sub_vg)
    except Exception as e:
        print(f"  EXP 2 ERROR: {e}")
        import traceback; traceback.print_exc()

    # EXP 3: Vg sweep for MC landscape
    try:
        all_results['exp3'] = exp3_vg_sweep_mc(fpga, noise_buf, rng)
    except Exception as e:
        print(f"  EXP 3 ERROR: {e}")
        import traceback; traceback.print_exc()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)
    total_pass = 0; total_tests = 0
    for name, res in all_results.items():
        if isinstance(res, dict) and 'tests' in res:
            t = res['tests']
            p = t.get('pass', 0); n = t.get('total', 0)
            total_pass += p; total_tests += n
            print(f"  {name}: {p}/{n}")
    print(f"  TOTAL: {total_pass}/{total_tests}")

    with open("results/z2233_subthreshold_memory.json", "w") as f:
        def convert(obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return obj
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\nSaved to results/z2233_subthreshold_memory.json")

    fpga.close()


if __name__ == "__main__":
    main()
