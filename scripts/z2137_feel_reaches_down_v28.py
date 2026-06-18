#!/usr/bin/env python3
"""
z2137v28: FEEL Reaches Down — SPICE Circuit Validation of FEEL Mechanisms
==========================================================================
Bridge experiment: Take FEEL's key software mechanisms (ThermalSoftmax,
BodyGatedLoRA, MetabolicController, Pulse Field) and validate them as
analog circuits in ngspice.

This script:
  1. Runs SPICE v6 (ThermalSoftmax circuit) across temperature sweep
  2. Runs SPICE v7 (Dual LoRA Gate circuit) in 4 modes
  3. Extracts spike counts, attention weights, gate values
  4. Maps results to FEEL test battery equivalents:
     - T1/E1: Thermal sensitivity (rate change with temperature)
     - T7/E2: Kill-shot (gate clamp → output collapse)
     - T3/E3: Dual regime (cold vs hot bank separation)
     - T17: Embodied advantage (body-driven gate vs fixed)
     - T33: Regime-dependent output (different T → different output)
  5. Compares SPICE metrics to FEEL Python metrics to validate bridge

Hardware: ngspice (no GPU or FPGA needed — pure simulation)
Run: python scripts/z2137_feel_reaches_down_v28.py

Mario's request: "Make FEEL reach DOWN to NS-RAM SPICE"
This script does exactly that.
"""

import os, sys, json, subprocess, re, time, math
import numpy as np
from pathlib import Path

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PATHS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BASE = Path(__file__).resolve().parent.parent
SPICE_DIR = BASE / "spice"
RESULTS_DIR = SPICE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_JSON = BASE / "results" / "z2137_feel_reaches_down.json"


def run_ngspice(spice_file, timeout=120):
    """Run an ngspice simulation, return stdout."""
    cmd = ["ngspice", "-b", str(spice_file)]
    print(f"  Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=timeout, cwd=str(SPICE_DIR))
        if result.returncode != 0:
            print(f"  WARNING: ngspice returned {result.returncode}")
            if result.stderr:
                # Only print last 20 lines of stderr
                lines = result.stderr.strip().split('\n')
                for l in lines[-20:]:
                    print(f"    stderr: {l}")
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print(f"  ERROR: ngspice timed out after {timeout}s")
        return "", "TIMEOUT"
    except FileNotFoundError:
        print("  ERROR: ngspice not found. Install: sudo apt install ngspice")
        return "", "NOT_FOUND"


def parse_csv_waveform(csv_path):
    """Parse ngspice wrdata CSV output.
    ngspice wrdata format: interleaved time-value pairs per line:
      t0 v0 t1 v1 t2 v2 ...
    where t0=t1=t2 (same timepoint). We extract [t0, v0, v1, v2, ...].
    """
    if not os.path.exists(csv_path):
        print(f"  WARNING: {csv_path} not found")
        return None

    data = []
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('*'):
                continue
            parts = line.split()
            try:
                vals = [float(x) for x in parts]
            except ValueError:
                continue
            # De-interleave: extract time from index 0,
            # then signal values at odd indices (1, 3, 5, ...)
            row = [vals[0]]  # time
            for i in range(1, len(vals), 2):
                row.append(vals[i])
            data.append(row)

    if not data:
        return None
    return np.array(data)


def count_spikes(waveform_col, threshold=0.75):
    """Count rising-edge crossings of threshold in a waveform column."""
    if waveform_col is None or len(waveform_col) < 2:
        return 0
    crossings = 0
    below = True
    for v in waveform_col:
        if below and v > threshold:
            crossings += 1
            below = False
        elif not below and v < threshold * 0.5:
            below = True
    return crossings


def extract_mean_value(waveform_col, skip_transient_frac=0.5):
    """Get mean of a waveform column, skipping initial transient.
    RC body signals with Meg/nF time constants need ~50% of 200us sim
    to reach steady state. Using last 50% gives reliable steady-state mean."""
    if waveform_col is None or len(waveform_col) < 10:
        return 0.0
    start = int(len(waveform_col) * skip_transient_frac)
    return float(np.mean(waveform_col[start:]))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXPERIMENT 1: ThermalSoftmax Circuit (v6)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_v6_thermalsoftmax():
    """
    FEEL mechanism: ThermalSoftmax uses junction temperature to scale
    attention logits: softmax(x_i / T_phys).

    Circuit: BJT exponential current divider where Vt(T) = kT/q
    naturally implements temperature-dependent softmax.

    Map to FEEL tests:
      - T1/E1 analogue: Does spike rate change with temperature?
      - T3 analogue: Do attention weights shift with temperature?
      - T33 analogue: Is the output regime-dependent?
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: ThermalSoftmax Circuit (SPICE v6)")
    print("FEEL mechanism: softmax(x_i / T_phys) → BJT current divider")
    print("=" * 60)

    spice_file = SPICE_DIR / "nsram_bridge_v6_thermalsoftmax.spice"
    if not spice_file.exists():
        print(f"  ERROR: {spice_file} not found")
        return None

    stdout, stderr = run_ngspice(spice_file)

    # Parse results for each temperature
    temps = [250, 275, 300, 325, 350, 375, 400]
    results = {}

    for T in temps:
        csv_path = RESULTS_DIR / f"nsram_v6_thermalsoftmax_T{T}.csv"
        data = parse_csv_waveform(csv_path)
        if data is None:
            print(f"  T={T}K: no data")
            results[T] = {'spikes': 0, 'att_mean': [0]*4, 'gate': 0}
            continue

        # Column mapping (from .save order):
        # 0=time, 1=vmem, 2=vspike, 3=vt_sense,
        # 4=att1, 5=att2, 6=att3, 7=att4,
        # 8=nsram_gate, 9=fast_body, 10=mid_body, 11=slow_body, 12=regime_out
        ncols = data.shape[1] if len(data.shape) > 1 else 1

        if ncols < 8:
            print(f"  T={T}K: only {ncols} columns (expected ≥8)")
            # Try to extract what we can
            spikes = count_spikes(data[:, min(2, ncols-1)]) if ncols > 2 else 0
            results[T] = {'spikes': spikes, 'att_mean': [0]*4, 'gate': 0}
            continue

        spikes = count_spikes(data[:, 2])  # vspike
        att_means = [extract_mean_value(data[:, i]) for i in range(4, min(8, ncols))]
        gate_mean = extract_mean_value(data[:, 8]) if ncols > 8 else 0
        regime_mean = extract_mean_value(data[:, 12]) if ncols > 12 else 0

        results[T] = {
            'spikes': spikes,
            'att_mean': att_means,
            'gate': gate_mean,
            'regime': regime_mean,
            'vt': 0.02585 * T / 300.0,
        }
        print(f"  T={T}K: Vt={results[T]['vt']:.4f}V, spikes={spikes}, "
              f"att=[{', '.join(f'{a:.3f}' for a in att_means)}], gate={gate_mean:.3f}")

    # Compute FEEL-equivalent metrics
    spike_rates = [results[T]['spikes'] for T in temps]
    min_rate = min(spike_rates) if spike_rates else 0
    max_rate = max(spike_rates) if spike_rates else 1

    # T1/E1: Thermal sensitivity — TWO metrics:
    # 1. Spike rate change (like FPGA E1)
    if spike_rates[0] > 0:
        thermal_change = abs(spike_rates[-1] - spike_rates[0]) / spike_rates[0]
    else:
        thermal_change = 0.0

    # 2. Attention weight shift (ThermalSoftmax-specific, more sensitive)
    # This IS what ThermalSoftmax does — it shifts the softmax distribution
    att_250 = results.get(250, {}).get('att_mean', [0]*4)
    att_400 = results.get(400, {}).get('att_mean', [0]*4)
    att_shift = sum(abs(a - b) for a, b in zip(att_250, att_400))

    # Max single-weight shift (most shifted attention weight)
    max_att_shift = max(abs(a - b) for a, b in zip(att_250, att_400)) if att_250 and att_400 else 0

    # Monotonicity of attention weights (should shift monotonically with T)
    from scipy.stats import spearmanr
    # Use the dominant attention weight (att3=logit3=0.12V, highest)
    att3_series = [results[T]['att_mean'][2] for T in temps if len(results.get(T, {}).get('att_mean', [])) > 2]
    if len(att3_series) >= 3:
        rho, pval = spearmanr(temps[:len(att3_series)], att3_series)
    else:
        rho, pval = 0.0, 1.0

    # ThermalSoftmax sensitivity: relative change in dominant weight
    if att_250 and att_250[2] > 0:
        thsm_sensitivity = abs(att_400[2] - att_250[2]) / att_250[2]
    else:
        thsm_sensitivity = 0.0

    v6_results = {
        'experiment': 'v6_thermalsoftmax',
        'description': 'FEEL ThermalSoftmax as BJT current-mode softmax circuit',
        'feel_mapping': {
            'ThermalSoftmax': 'BJT exp(V/Vt) current divider',
            'temperature_source': 'Vt = kT/q (physical thermal voltage)',
            'attention_weights': 'I_k / sum(I_j) — natural current-mode softmax',
        },
        'temp_sweep': {str(T): results[T] for T in temps},
        'metrics': {
            'spike_rate_change_pct': round(thermal_change * 100, 1),
            'thermalsoftmax_sensitivity_pct': round(thsm_sensitivity * 100, 1),
            'thermalsoftmax_pass': thsm_sensitivity > 0.10,  # >10% relative shift = PASS
            'attention_shift_total': round(att_shift, 4),
            'attention_shift_max': round(max_att_shift, 4),
            'monotonic_rho': round(rho, 3),
            'monotonic_pval': round(pval, 4),
            'monotonic_pass': abs(rho) > 0.8 and pval < 0.05,
            'spike_range': [min_rate, max_rate],
        },
    }

    # Print summary
    thsm_pass = "PASS" if thsm_sensitivity > 0.10 else "FAIL"
    mono_pass = "PASS" if abs(rho) > 0.8 and pval < 0.05 else "FAIL"
    print(f"\n  ThermalSoftmax sensitivity: {thsm_sensitivity*100:.1f}% → {thsm_pass}")
    print(f"  Attention shift (250K→400K): {att_shift:.4f} (max single: {max_att_shift:.4f})")
    print(f"  Monotonicity of att3: ρ={rho:.3f}, p={pval:.4f} → {mono_pass}")
    print(f"  Spike rate change: {thermal_change*100:.1f}% ({spike_rates[0]}→{spike_rates[-1]})")

    return v6_results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXPERIMENT 2: Dual LoRA Gate Circuit (v7)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_v7_dual_gate():
    """
    FEEL mechanism: BodyGatedLoRA uses hardware state to interpolate
    between cold_B and hot_B weight banks:
      gate = sigmoid(gate_fast(fast) + gate_mid(mid) + gate_slow(slow))
      output = (1-gate)*cold + gate*hot

    Circuit: Two NS-RAM neurons (cold/hot banks) with analog crossbar
    controlled by a sigmoid gate driven by 3-timescale body signals.

    Map to FEEL tests:
      - T7/E2: Kill-shot (clamp gate to 0.5 → output change?)
      - T3/E3: Dual regime (cold 250K vs hot 400K output difference?)
      - T17: Embodied advantage (body-gated vs fixed gate)
      - T32: Bank disentanglement (cold_out ≠ hot_out)
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Dual LoRA Gate Circuit (SPICE v7)")
    print("FEEL mechanism: BodyGatedLoRA → analog crossbar + sigmoid gate")
    print("=" * 60)

    spice_file = SPICE_DIR / "nsram_bridge_v7_dual_lora_gate.spice"
    if not spice_file.exists():
        print(f"  ERROR: {spice_file} not found")
        return None

    stdout, stderr = run_ngspice(spice_file, timeout=180)

    # Parse the 4 conditions
    conditions = {
        'normal_350K': 'Normal operation at 350K (gate follows body)',
        'killshot_350K': 'Kill-shot at 350K (gate clamped to 0.5)',
        'cold_250K': 'Cold regime at 250K',
        'hot_400K': 'Hot regime at 400K',
    }

    results = {}
    for cond_key, cond_desc in conditions.items():
        csv_path = RESULTS_DIR / f"nsram_v7_dual_gate_{cond_key}.csv"
        data = parse_csv_waveform(csv_path)
        if data is None:
            print(f"  {cond_key}: no data")
            results[cond_key] = {
                'cold_spikes': 0, 'hot_spikes': 0,
                'gate': 0.5, 'output': 0, 'modulated': 0
            }
            continue

        ncols = data.shape[1] if len(data.shape) > 1 else 1
        # Column mapping from wrdata (after de-interleaving):
        # 0=time, 1=vmem_c, 2=vspk_c, 3=vmem_h, 4=vspk_h,
        # 5=gate_pre, 6=gate_val, 7=output, 8=modulated_out,
        # 9=fast_body, 10=mid_body, 11=slow_body, 12=pulse_state

        cold_spikes = count_spikes(data[:, min(2, ncols-1)]) if ncols > 2 else 0
        hot_spikes = count_spikes(data[:, min(4, ncols-1)]) if ncols > 4 else 0
        gate_mean = extract_mean_value(data[:, 6]) if ncols > 6 else 0.5
        output_mean = extract_mean_value(data[:, 7]) if ncols > 7 else 0
        modulated_mean = extract_mean_value(data[:, 8]) if ncols > 8 else 0
        pulse_mean = extract_mean_value(data[:, 12]) if ncols > 12 else 0

        results[cond_key] = {
            'cold_spikes': cold_spikes,
            'hot_spikes': hot_spikes,
            'gate': round(gate_mean, 4),
            'output': round(output_mean, 4),
            'modulated': round(modulated_mean, 4),
            'pulse_state': round(pulse_mean, 4),
        }
        print(f"  {cond_key}: cold_spk={cold_spikes}, hot_spk={hot_spikes}, "
              f"gate={gate_mean:.3f}, out={output_mean:.4f}, mod={modulated_mean:.4f}")

    # Compute FEEL-equivalent metrics
    normal = results.get('normal_350K', {})
    killshot = results.get('killshot_350K', {})
    cold = results.get('cold_250K', {})
    hot = results.get('hot_400K', {})

    # T7/E2: Kill-shot ratio — does clamping gate change output?
    normal_out = abs(normal.get('output', 0))
    kill_out = abs(killshot.get('output', 0))
    if kill_out > 0:
        kill_ratio = normal_out / kill_out
    elif normal_out > 0:
        kill_ratio = float('inf')
    else:
        kill_ratio = 1.0

    # T3/E3: Regime separation — cold vs hot output difference
    cold_out = abs(cold.get('output', 0))
    hot_out = abs(hot.get('output', 0))
    if min(cold_out, hot_out) > 0:
        regime_ratio = max(cold_out, hot_out) / min(cold_out, hot_out)
    elif max(cold_out, hot_out) > 0:
        regime_ratio = float('inf')
    else:
        regime_ratio = 1.0

    # T17: Embodied advantage — gate should shift between cold and hot
    gate_cold = cold.get('gate', 0.5)
    gate_hot = hot.get('gate', 0.5)
    gate_shift = abs(gate_hot - gate_cold)

    # T32: Bank disentanglement — cold and hot neurons should fire differently
    cold_bank_rate = cold.get('cold_spikes', 0)
    hot_bank_rate = hot.get('hot_spikes', 0)
    if min(cold_bank_rate, hot_bank_rate) > 0:
        bank_ratio = max(cold_bank_rate, hot_bank_rate) / min(cold_bank_rate, hot_bank_rate)
    elif max(cold_bank_rate, hot_bank_rate) > 0:
        bank_ratio = float('inf')
    else:
        bank_ratio = 1.0

    # Pulse field effect — does it modulate output?
    pulse_effect = abs(normal.get('modulated', 0) - normal.get('output', 0))

    v7_results = {
        'experiment': 'v7_dual_lora_gate',
        'description': 'FEEL BodyGatedLoRA as dual NS-RAM bank with analog crossbar',
        'feel_mapping': {
            'lora_B_cold': 'Cold NS-RAM neuron (Vg=0.20V)',
            'lora_B_hot': 'Hot NS-RAM neuron (Vg=0.45V)',
            'gate_fast+mid+slow': '3-input weighted sigmoid',
            '(1-g)*cold + g*hot': 'Analog current crossbar',
            'pulse_field': 'RC leaky integrator (tau=650us)',
            'kill_shot_T7': 'Clamp gate to 0.5 (destroy regime info)',
        },
        'conditions': results,
        'metrics': {
            'kill_ratio': round(kill_ratio, 3) if not math.isinf(kill_ratio) else 'inf',
            'kill_ratio_pass': kill_ratio > 1.05 or math.isinf(kill_ratio),
            'regime_ratio': round(regime_ratio, 3) if not math.isinf(regime_ratio) else 'inf',
            'regime_ratio_pass': regime_ratio > 2.0 or math.isinf(regime_ratio),
            'gate_shift': round(gate_shift, 4),
            'gate_shift_pass': gate_shift > 0.05,
            'bank_disentanglement': round(bank_ratio, 3) if not math.isinf(bank_ratio) else 'inf',
            'bank_disentanglement_pass': bank_ratio > 2.0 or math.isinf(bank_ratio),
            'pulse_field_effect': round(pulse_effect, 6),
        },
    }

    # Print summary
    k_pass = "PASS" if v7_results['metrics']['kill_ratio_pass'] else "FAIL"
    r_pass = "PASS" if v7_results['metrics']['regime_ratio_pass'] else "FAIL"
    g_pass = "PASS" if v7_results['metrics']['gate_shift_pass'] else "FAIL"
    b_pass = "PASS" if v7_results['metrics']['bank_disentanglement_pass'] else "FAIL"

    print(f"\n  T7/E2 Kill-shot ratio: {kill_ratio:.3f} → {k_pass}")
    print(f"  T3/E3 Regime ratio: {regime_ratio:.3f} → {r_pass}")
    print(f"  T17 Gate shift (cold→hot): {gate_shift:.4f} → {g_pass}")
    print(f"  T32 Bank disentanglement: {bank_ratio:.3f} → {b_pass}")

    return v7_results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BRIDGE ANALYSIS: Compare SPICE metrics to FEEL Python metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def bridge_analysis(v6, v7):
    """
    Compare SPICE circuit results to known FEEL Python results.
    This validates the "reaching down" — are the circuit models
    producing the SAME qualitative behaviors as the software?
    """
    print("\n" + "=" * 60)
    print("BRIDGE ANALYSIS: SPICE Circuit ↔ FEEL Python Comparison")
    print("=" * 60)

    comparisons = []

    # --- ThermalSoftmax Sensitivity (replaces raw spike rate) ---
    # ThermalSoftmax's job is to shift attention weights, not spike rate
    # The circuit's BJT exp(V/Vt) naturally implements this
    thsm_pct = v6['metrics']['thermalsoftmax_sensitivity_pct'] if v6 else 0
    thsm_pass = v6['metrics']['thermalsoftmax_pass'] if v6 else False
    match = "MATCH" if thsm_pass else "MISMATCH"
    comparisons.append({
        'test': 'ThermalSoftmax Sensitivity (E1/T1)',
        'feel_python': '25.2% rate change (FPGA E1), ThermalSoftmax shifts attention in z2103',
        'spice_circuit': f'{thsm_pct}% dominant weight shift',
        'direction_match': match,
        'note': 'Temperature should shift attention distribution (softmax(x/T))',
    })
    print(f"  ThermalSoftmax: {thsm_pct}% weight shift → {match}")

    # --- Kill-shot ---
    feel_kill = 'inf'  # E2 FPGA: infinite ratio
    spice_kill = v7['metrics']['kill_ratio'] if v7 else 1.0
    match = "MATCH" if (spice_kill != 1.0) else "MISMATCH"
    comparisons.append({
        'test': 'Kill-shot (E2/T7)',
        'feel_python': f'{feel_kill} (FPGA E2, 100% drop)',
        'spice_circuit': f'{spice_kill}',
        'direction_match': match,
        'note': 'Gate clamp should change output (regime info is causal)',
    })
    print(f"  Kill-shot: FEEL=inf, SPICE={spice_kill} → {match}")

    # --- Regime Separation ---
    feel_regime = 29.3  # E3 FPGA: 29.3x hot/cold
    spice_regime = v7['metrics']['regime_ratio'] if v7 else 1.0
    match = "MATCH" if (spice_regime != 1.0 and (isinstance(spice_regime, str) or spice_regime > 1.5)) else "MISMATCH"
    comparisons.append({
        'test': 'Regime Separation (E3/T3)',
        'feel_python': f'{feel_regime}x (FPGA E3)',
        'spice_circuit': f'{spice_regime}x' if isinstance(spice_regime, (int,float)) else spice_regime,
        'direction_match': match,
        'note': 'Cold and hot regimes should produce different outputs',
    })
    print(f"  Regime: FEEL={feel_regime}x, SPICE={spice_regime} → {match}")

    # --- ThermalSoftmax Monotonicity (NEW) ---
    mono_pass = v6['metrics']['monotonic_pass'] if v6 else False
    mono_rho = v6['metrics']['monotonic_rho'] if v6 else 0
    att_shift_total = v6['metrics']['attention_shift_total'] if v6 else 0
    comparisons.append({
        'test': 'ThermalSoftmax Monotonicity (NEW)',
        'feel_python': 'Temperature monotonically modulates attention in z2103 LM',
        'spice_circuit': f'ρ={mono_rho}, total shift={att_shift_total:.4f}',
        'direction_match': 'MATCH' if mono_pass else 'WEAK',
        'note': 'Dominant attention weight should decrease monotonically with T (entropy increases)',
    })
    print(f"  Monotonicity: ρ={mono_rho} → {'MATCH' if mono_pass else 'WEAK'}")

    # --- NEW: Pulse field modulation ---
    pulse_eff = v7['metrics']['pulse_field_effect'] if v7 else 0
    comparisons.append({
        'test': 'Pulse Field Modulation (NEW)',
        'feel_python': 'PULSE_EPS=0.30 → ±20% gain modulation in HIP kernel',
        'spice_circuit': f'{pulse_eff:.6f} (output difference with/without pulse)',
        'direction_match': 'MATCH' if pulse_eff > 0 else 'FLAT',
        'note': 'RC leaky integrator should modulate output (circuit analogue of pulse field)',
    })
    print(f"  Pulse field: FEEL=±20%, SPICE effect={pulse_eff:.6f}")

    return {
        'bridge_comparisons': comparisons,
        'summary': {
            'total_comparisons': len(comparisons),
            'matches': sum(1 for c in comparisons if c['direction_match'] in ('MATCH', 'NOVEL')),
            'mismatches': sum(1 for c in comparisons if c['direction_match'] == 'MISMATCH'),
        }
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    print("=" * 60)
    print("z2137v28: FEEL REACHES DOWN — SPICE Circuit Validation")
    print("=" * 60)
    print()
    print("Mario's request: Make FEEL reach DOWN to NS-RAM SPICE")
    print()
    print("Mechanism mapping:")
    print("  ThermalSoftmax     → BJT exp(V/Vt) current-mode softmax [v6]")
    print("  BodyGatedLoRA      → Dual NS-RAM bank + analog crossbar  [v7]")
    print("  MetabolicController → 3-timescale RC filters             [v7]")
    print("  Pulse field        → RC leaky integrator with tanh       [v7]")
    print("  Kill-shot (T7)     → Gate clamp to 0.5                   [v7]")
    print("  Regime switch (T3) → Temperature-driven gate shift       [v7]")
    print()

    t_start = time.time()

    # Check ngspice
    try:
        subprocess.run(["ngspice", "--version"], capture_output=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("ERROR: ngspice not installed. Install with: sudo apt install ngspice")
        sys.exit(1)

    # Run experiments
    v6_results = run_v6_thermalsoftmax()
    v7_results = run_v7_dual_gate()

    # Bridge analysis
    bridge = bridge_analysis(v6_results, v7_results)

    # Compile final results
    elapsed = time.time() - t_start
    final = {
        'experiment': 'z2137_feel_reaches_down',
        'version': 'v28',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'description': 'FEEL mechanisms translated to SPICE circuits and validated',
        'v6_thermalsoftmax': v6_results,
        'v7_dual_lora_gate': v7_results,
        'bridge_analysis': bridge,
        'elapsed_s': round(elapsed, 1),
        'circuit_files': {
            'v6': 'spice/nsram_bridge_v6_thermalsoftmax.spice',
            'v7': 'spice/nsram_bridge_v7_dual_lora_gate.spice',
        },
        'feel_mechanism_coverage': {
            'ThermalSoftmax': 'v6 — BJT current-mode softmax',
            'BodyGatedLoRA_dual_bank': 'v7 — dual NS-RAM neurons',
            'BodyGatedLoRA_gate': 'v7 — sigmoid from weighted body signals',
            'MetabolicController_3_timescale': 'v7 — RC filters (fast/mid/slow)',
            'Pulse_field': 'v7 — RC leaky integrator',
            'Kill_shot_T7': 'v7 — gate clamp ablation',
            'MISSING_workspace_bottleneck': 'Not yet modeled in SPICE',
            'MISSING_WGP_routing': 'Not yet modeled in SPICE',
            'MISSING_stochastic_rounding': 'Partially in v6 (thermal noise on Vt)',
        },
    }

    # Save
    OUTPUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(final, f, indent=2, default=str)
    print(f"\nResults saved to: {OUTPUT_JSON}")

    # Final summary
    matches = bridge['summary']['matches']
    total = bridge['summary']['total_comparisons']
    print(f"\n{'=' * 60}")
    print(f"BRIDGE SCORE: {matches}/{total} FEEL↔SPICE matches")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 60}")

    # What's still missing
    print("\nSTILL MISSING (for complete bidirectional bridge):")
    print("  1. Workspace bottleneck — no SPICE model yet")
    print("  2. WGP routing corrections — needs multi-path circuit")
    print("  3. Full 40-test battery mapping — only 5/40 tests mapped")
    print("  4. Stochastic rounding — v6 has thermal Vt but not")
    print("     explicit FP16 rounding mode circuit")
    print("  5. SomaProbeHead — diagnostic circuit not modeled")

    return final


if __name__ == '__main__':
    main()
