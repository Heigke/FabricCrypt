#!/usr/bin/env python3
"""
z1712: Causal Intervention -- PROVING Embodiment Matters

Granger Causality test: deliberately manipulate hardware state (performance
level) and measure whether the model's internal state changes CAUSALLY in
response.

Phase 1: Train embodied model (5 epochs, full telemetry + actuation)
Phase 2: Intervention protocol (200 eval batches, 3 conditions):
  A: Embodied   (FiLM on, real telemetry, interventions applied)
  B: Disembodied (FiLM off, constant telemetry=0.5, interventions applied)
  C: No Intervention (FiLM on, real telemetry, no perf-level changes)

Verdicts:
  V1: A response latency < 3 batches
  V2: A hidden divergence > B hidden divergence
  V3: A correlation(perf, action) > 0.3
  V4: Granger p < 0.05 for A but not B
"""

import sys, os, json, time, math, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from src.metabolic.film_transformer import create_metabolic_transformer, MetabolicConfig
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry
from src.actuation.gpu_actuator import GPUActuator, PerformanceLevel

ROOT = Path('/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy')
BS, SL, LR = 4, 256, 3e-4
TRAIN_EPOCHS, TRAIN_STEPS = 5, 100
EVAL_BATCHES = 200
SWITCH_INTERVAL = 10
COOLDOWN = 25
ACTION_MAP = {0: PerformanceLevel.LOW, 1: PerformanceLevel.BALANCED,
              2: PerformanceLevel.HIGH, 3: PerformanceLevel.HIGH}
PERF_CYCLE = [PerformanceLevel.LOW, PerformanceLevel.HIGH,
              PerformanceLevel.LOW, PerformanceLevel.HIGH]


def jsonify(obj):
    """Numpy/torch-safe JSON serializer."""
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, torch.Tensor): return obj.detach().cpu().tolist()
    if isinstance(obj, (np.bool_,)): return bool(obj)
    return str(obj)


def load_data():
    path = ROOT / 'data' / 'tinyshakespeare.txt'
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    text = path.read_text(encoding='utf-8')
    data = torch.tensor(list(text.encode('utf-8')), dtype=torch.long)
    print(f"Loaded TinyShakespeare: {len(data):,} bytes")
    return data


def get_batch(data, bs, sl, device):
    starts = torch.randint(0, len(data) - sl - 1, (bs,))
    x = torch.stack([data[s:s+sl] for s in starts]).to(device)
    y = torch.stack([data[s+1:s+sl+1] for s in starts]).to(device)
    return x, y


def build_telemetry(telem, device, prev_sample=None):
    """Build 12-dim normalized telemetry vector."""
    s = telem.read_sample()
    raw = [s.power_w / 50, s.temp_edge_c / 100, s.freq_sclk_mhz / 3000,
           s.gpu_busy_pct / 100, 0.5, 0.0]
    if prev_sample is not None:
        raw += [(s.power_w - prev_sample.power_w) / 50,
                (s.temp_edge_c - prev_sample.temp_edge_c) / 100,
                (s.freq_sclk_mhz - prev_sample.freq_sclk_mhz) / 3000,
                (s.gpu_busy_pct - prev_sample.gpu_busy_pct) / 100,
                (s.temp_edge_c - 70) / 100, (3000 - s.freq_sclk_mhz) / 3000]
    else:
        raw += [0.0] * 6
    return torch.tensor(raw[:12], dtype=torch.float32, device=device).unsqueeze(0), s


def constant_telemetry(device):
    return torch.full((1, 12), 0.5, dtype=torch.float32, device=device)


def granger_causality_test(perf_series, action_series):
    """VAR(1) F-test: does past perf_level Granger-cause action_chosen?
    Unrestricted: action_t = a0 + a1*action_{t-1} + a2*perf_{t-1}
    Restricted:   action_t = a0 + a1*action_{t-1}
    F = ((RSS_r - RSS_u) / 1) / (RSS_u / (n - 3))"""
    p = np.array(perf_series, dtype=np.float64)
    a = np.array(action_series, dtype=np.float64)
    n = len(a) - 1
    if n < 5:
        return {'f_stat': 0.0, 'p_value': 1.0, 'significant': False}
    y, a_lag, p_lag = a[1:], a[:-1], p[:-1]

    X_u = np.column_stack([np.ones(n), a_lag, p_lag])
    beta_u = np.linalg.lstsq(X_u, y, rcond=None)[0]
    rss_u = np.sum((y - X_u @ beta_u) ** 2)

    X_r = np.column_stack([np.ones(n), a_lag])
    beta_r = np.linalg.lstsq(X_r, y, rcond=None)[0]
    rss_r = np.sum((y - X_r @ beta_r) ** 2)

    f_stat = 0.0 if rss_u < 1e-12 else ((rss_r - rss_u) / 1.0) / (rss_u / max(n - 3, 1))

    try:
        from scipy.stats import f as f_dist
        p_value = 1.0 - f_dist.cdf(f_stat, 1, max(n - 3, 1))
    except ImportError:
        p_value = math.exp(-0.5 * f_stat) if 0 < f_stat < 20 else (1.0 if f_stat <= 0 else 1e-6)

    return {'f_stat': float(f_stat), 'p_value': float(p_value),
            'significant': p_value < 0.05, 'rss_unrestricted': float(rss_u),
            'rss_restricted': float(rss_r), 'coeff_perf_lag': float(beta_u[2]),
            'n_observations': n}


def train_embodied(model, data, telem, actuator, device):
    """Phase 1: Train embodied model with full telemetry + actuation."""
    print(f"\n{'='*60}\n  Phase 1: Embodied Training ({TRAIN_EPOCHS} epochs)\n{'='*60}")
    model.enable_conditioning(True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    prev_s, losses = None, []

    for epoch in range(TRAIN_EPOCHS):
        eloss = 0.0
        for step in range(TRAIN_STEPS):
            x, y = get_batch(data, BS, SL, device)
            tv, prev_s = build_telemetry(telem, device, prev_s)
            out = model(x, tv.expand(BS, -1))
            loss = F.cross_entropy(out['logits'].view(-1, 256), y.view(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

            mean_probs = F.softmax(out['action_logits'], dim=-1).mean(dim=0)
            action_idx = torch.argmax(mean_probs).item()
            try: actuator.set_performance_level(ACTION_MAP[min(action_idx, 3)])
            except Exception: pass
            eloss += loss.item(); losses.append(loss.item())
            if (step + 1) % 50 == 0:
                print(f"  e{epoch+1} s{step+1}/{TRAIN_STEPS} loss={loss.item():.4f} act={action_idx}")
        print(f"  Epoch {epoch+1} avg loss: {eloss/TRAIN_STEPS:.4f}")

    final_loss = float(np.mean(losses[-20:]))
    print(f"  Training done. Final loss: {final_loss:.4f}")
    return final_loss


def compute_response_latency(action_series, switch_points, window=3):
    """Batches after a switch until action distribution changes."""
    latencies = []
    for sp in switch_points:
        if sp < window or sp + window >= len(action_series):
            continue
        pre = action_series[sp - 1]
        found = False
        for lag in range(1, min(window + 1, len(action_series) - sp)):
            if action_series[sp + lag] != pre:
                latencies.append(lag); found = True; break
        if not found:
            latencies.append(window + 1)
    return float(np.mean(latencies)) if latencies else float(window + 1)


def run_intervention(label, model_src, data, telem, actuator, device,
                     use_film, use_real_telemetry, apply_interventions):
    """Run EVAL_BATCHES eval batches with deliberate hardware manipulation."""
    print(f"\n{'='*60}\n  Condition {label}: film={use_film} real_telem={use_real_telemetry} "
          f"intervene={apply_interventions}\n{'='*60}")

    model = copy.deepcopy(model_src).to(device)
    model.eval(); model.enable_conditioning(use_film)

    prev_s = None
    perf_series, action_series = [], []
    hidden_norm_series, entropy_series = [], []
    switch_points, pre_switch_logits, kl_divs = [], [], []
    cur_idx, cur_perf = 0, PERF_CYCLE[0]
    if apply_interventions:
        try: actuator.set_performance_level(cur_perf)
        except Exception: pass

    with torch.no_grad():
        for bi in range(EVAL_BATCHES):
            switched = False
            if apply_interventions and bi > 0 and bi % SWITCH_INTERVAL == 0:
                cur_idx = (cur_idx + 1) % len(PERF_CYCLE)
                cur_perf = PERF_CYCLE[cur_idx]
                try: actuator.set_performance_level(cur_perf)
                except Exception: pass
                switch_points.append(bi); switched = True
                time.sleep(0.05)

            pn = 1.0 if cur_perf == PerformanceLevel.HIGH else 0.0
            perf_series.append(pn)

            tv = (build_telemetry(telem, device, prev_s) if use_real_telemetry
                  else (constant_telemetry(device), prev_s))
            if use_real_telemetry:
                tv, prev_s = tv
            else:
                tv = tv[0]

            x, y = get_batch(data, BS, SL, device)
            out = model(x, tv.expand(BS, -1), return_hidden=True)

            mp = F.softmax(out['action_logits'], dim=-1).mean(dim=0)
            action_series.append(torch.argmax(mp).item())
            hidden_norm_series.append(out['hidden'].norm(dim=-1).mean().item())

            lp = F.log_softmax(out['logits'][:, -1, :], dim=-1)
            entropy_series.append(-(lp.exp() * lp).sum(-1).mean().item())

            if switched and pre_switch_logits:
                pre_lp = F.log_softmax(pre_switch_logits[-1], dim=-1)
                post_lp = F.log_softmax(out['logits'][:, -1, :], dim=-1)
                kl_divs.append(abs((pre_lp.exp() * (pre_lp - post_lp)).sum(-1).mean().item()))

            if (bi + 1) % SWITCH_INTERVAL == 0:
                pre_switch_logits.append(out['logits'][:, -1, :].clone())
            if (bi + 1) % 50 == 0:
                print(f"  [{label}] {bi+1}/{EVAL_BATCHES} perf={pn:.0f} "
                      f"act={action_series[-1]} hn={hidden_norm_series[-1]:.2f}")

    try: actuator.set_performance_level(PerformanceLevel.BALANCED)
    except Exception: pass

    resp_latency = compute_response_latency(action_series, switch_points)
    hidden_div = float(np.mean(kl_divs)) if kl_divs else 0.0
    corr = 0.0
    if len(perf_series) > 2:
        c = np.corrcoef(perf_series, action_series)[0, 1]
        corr = 0.0 if np.isnan(c) else float(c)
    granger = granger_causality_test(perf_series, action_series)

    hn = np.array(hidden_norm_series)
    hn_sv = 0.0
    if switch_points:
        sn = []
        for sp in switch_points:
            sn.extend(hn[max(0, sp-2):min(len(hn), sp+3)].tolist())
        hn_sv = float(np.var(sn)) if sn else 0.0

    metrics = {
        'condition': label, 'use_film': use_film,
        'use_real_telemetry': use_real_telemetry,
        'apply_interventions': apply_interventions,
        'response_latency': resp_latency, 'hidden_divergence': hidden_div,
        'action_perf_correlation': corr, 'granger_causality': granger,
        'hidden_norm_switch_variance': hn_sv,
        'mean_hidden_norm': float(np.mean(hidden_norm_series)),
        'mean_output_entropy': float(np.mean(entropy_series)),
        'action_distribution': np.bincount(action_series, minlength=4).tolist(),
        'num_switches': len(switch_points), 'num_kl_samples': len(kl_divs),
        'perf_level_series': perf_series, 'action_series': action_series,
        'hidden_norm_series': hidden_norm_series, 'output_entropy_series': entropy_series,
    }
    print(f"  [{label}] latency={resp_latency:.2f}  kl_div={hidden_div:.6f}  "
          f"corr={corr:.4f}  Granger F={granger['f_stat']:.3f} p={granger['p_value']:.4f}")
    return metrics


def main():
    print("=" * 70)
    print("  z1712: CAUSAL INTERVENTION -- Proving Embodiment Matters")
    print("  Granger causality test for hardware -> model response")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}  VRAM: {props.total_memory / 1e9:.1f} GB")
    print(f"Device: {device}  BS={BS} SL={SL}")

    data = load_data()
    telem = SysfsHwmonTelemetry(sample_rate_hz=20)
    actuator = GPUActuator(card_id=0)
    model = create_metabolic_transformer(
        hidden_dim=256, num_layers=6, num_heads=4, telemetry_dim=12,
    ).to(device)
    npar = sum(p.numel() for p in model.parameters())
    print(f"Model params: {npar:,}")

    try:
        train_loss = train_embodied(model, data, telem, actuator, device)

        conds = {
            'A': dict(use_film=True,  use_real_telemetry=True,  apply_interventions=True),
            'B': dict(use_film=False, use_real_telemetry=False, apply_interventions=True),
            'C': dict(use_film=True,  use_real_telemetry=True,  apply_interventions=False),
        }
        results = {}
        for i, (lbl, kw) in enumerate(conds.items()):
            results[lbl] = run_intervention(lbl, model, data, telem, actuator, device, **kw)
            if i < len(conds) - 1:
                print(f"\nCooldown {COOLDOWN}s...")
                try: actuator.set_performance_level(PerformanceLevel.BALANCED)
                except Exception: pass
                time.sleep(COOLDOWN)

        # --- Verdicts ---
        print(f"\n{'='*70}\n  VERDICTS -- CAUSAL INTERVENTION\n{'='*70}")
        verdicts = {}
        A, B, C = results['A'], results['B'], results['C']

        lat_a = A['response_latency']; v1 = lat_a < 3.0
        verdicts['response_latency'] = {'pass': v1, 'a_latency': lat_a, 'threshold': 3.0,
            'description': 'Embodied responds within 3 batches of HW switch'}
        print(f"\n1. Response Latency (A < 3): {'PASS' if v1 else 'FAIL'}  A={lat_a:.2f}")

        da, db = A['hidden_divergence'], B['hidden_divergence']; v2 = da > db
        verdicts['hidden_divergence'] = {'pass': v2, 'a_div': da, 'b_div': db,
            'description': 'Embodied responds more to HW switches than disembodied'}
        print(f"2. Hidden Divergence (A > B): {'PASS' if v2 else 'FAIL'}  A={da:.6f} B={db:.6f}")

        ca = A['action_perf_correlation']; v3 = abs(ca) > 0.3
        verdicts['action_adaptation'] = {'pass': v3, 'a_corr': ca, 'threshold': 0.3,
            'description': 'Action correlates with perf level (|r| > 0.3)'}
        print(f"3. Action Adaptation (|corr| > 0.3): {'PASS' if v3 else 'FAIL'}  corr={ca:.4f}")

        ga, gb = A['granger_causality'], B['granger_causality']
        v4 = ga['significant'] and not gb['significant']
        verdicts['granger_causality'] = {'pass': v4,
            'a_f': ga['f_stat'], 'a_p': ga['p_value'], 'a_sig': ga['significant'],
            'b_f': gb['f_stat'], 'b_p': gb['p_value'], 'b_sig': gb['significant'],
            'description': 'HW Granger-causes action in A (p<0.05) but not B'}
        print(f"4. Granger (A sig, not B): {'PASS' if v4 else 'FAIL'}  "
              f"A: F={ga['f_stat']:.3f} p={ga['p_value']:.4f}  "
              f"B: F={gb['f_stat']:.3f} p={gb['p_value']:.4f}")

        passed = sum(v['pass'] for v in verdicts.values())
        total = len(verdicts)
        print(f"\n{'='*70}\n  OVERALL: {passed}/{total} passed")
        if passed >= 3:
            print("  CONCLUSION: Embodiment CAUSALLY matters -- HW state drives model")
        elif passed >= 2:
            print("  CONCLUSION: Partial causal evidence for embodiment")
        else:
            print("  CONCLUSION: Insufficient causal evidence")
        print(f"{'='*70}")

        print(f"\n  {'Cond':<5} {'Latency':<9} {'KL-Div':<11} {'Corr':<9} "
              f"{'F-stat':<9} {'p-val':<9}")
        for l in 'ABC':
            r, gc = results[l], results[l]['granger_causality']
            print(f"  {l:<5} {r['response_latency']:<9.2f} {r['hidden_divergence']:<11.6f} "
                  f"{r['action_perf_correlation']:<9.4f} {gc['f_stat']:<9.3f} {gc['p_value']:<9.4f}")

        # Trim series for JSON
        for l in 'ABC':
            for k in ['perf_level_series', 'action_series',
                      'hidden_norm_series', 'output_entropy_series']:
                s = results[l].pop(k)
                results[l][f'{k}_first20'] = s[:20]
                results[l][f'{k}_last20'] = s[-20:]

        output = {
            'experiment': 'z1712_causal_intervention',
            'description': 'Granger causality test proving embodiment matters',
            'timestamp': datetime.now().isoformat(),
            'device': str(device),
            'gpu_name': (torch.cuda.get_device_properties(0).name
                         if torch.cuda.is_available() else 'cpu'),
            'config': {'batch_size': BS, 'seq_len': SL, 'lr': LR,
                       'train_epochs': TRAIN_EPOCHS, 'train_steps': TRAIN_STEPS,
                       'eval_batches': EVAL_BATCHES, 'switch_interval': SWITCH_INTERVAL,
                       'model_params': npar},
            'train_loss': train_loss, 'conditions': results,
            'verdicts': verdicts, 'passed': passed, 'total': total,
        }
        out_path = ROOT / 'results' / 'z1712_causal_intervention.json'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2, default=jsonify)
        print(f"\nResults saved to: {out_path}")
        print("Done.")

    finally:
        try: actuator.set_performance_level(PerformanceLevel.BALANCED)
        except Exception: pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
