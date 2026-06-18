#!/usr/bin/env python3
"""
z1714: Multivariate Telemetry Granger Causality -- Fixing z1712

z1712 tested perf_level->action (p=0.71). But the model sees power/temp/freq
via 12-dim telemetry, NOT perf_level. Test the RIGHT causal variables.

Tests: power->action, freq->action, temp->hidden (Granger VAR lag=2),
       full telem PCA -> hidden (R^2), Transfer Entropy both directions.
Conditions: A=Embodied, B=Disembodied, C=Embodied+NoIntervention
Verdicts: (1) any Granger p<0.05 in A, (2) A more sig than B,
          (3) TE(telem->hidden) > TE(hidden->telem), (4) R^2 improve >5%
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
EVAL_BATCHES = 300
SWITCH_INTERVAL = 15
COOLDOWN = 20
ACTION_MAP = {0: PerformanceLevel.LOW, 1: PerformanceLevel.BALANCED,
              2: PerformanceLevel.HIGH, 3: PerformanceLevel.HIGH}
PERF_CYCLE = [PerformanceLevel.LOW, PerformanceLevel.HIGH,
              PerformanceLevel.LOW, PerformanceLevel.HIGH]


def jsonify(obj):
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


# ---------- Granger causality (VAR with lag=2) ----------

def granger_var2(cause_series, effect_series):
    """VAR(2) F-test: does cause Granger-cause effect?"""
    x = np.array(cause_series, dtype=np.float64)
    y = np.array(effect_series, dtype=np.float64)
    n = len(y) - 2
    if n < 8:
        return {'f_stat': 0.0, 'p_value': 1.0, 'significant': False, 'n': n}
    yt, y1, y2 = y[2:], y[1:-1], y[:-2]
    x1, x2 = x[1:-1], x[:-2]

    X_u = np.column_stack([np.ones(n), y1, y2, x1, x2])
    beta_u = np.linalg.lstsq(X_u, yt, rcond=None)[0]
    rss_u = np.sum((yt - X_u @ beta_u) ** 2)

    X_r = np.column_stack([np.ones(n), y1, y2])
    beta_r = np.linalg.lstsq(X_r, yt, rcond=None)[0]
    rss_r = np.sum((yt - X_r @ beta_r) ** 2)

    q, dof = 2, max(n - 5, 1)
    f_stat = 0.0 if rss_u < 1e-12 else ((rss_r - rss_u) / q) / (rss_u / dof)

    try:
        from scipy.stats import f as f_dist
        p_value = 1.0 - f_dist.cdf(f_stat, q, dof)
    except ImportError:
        p_value = math.exp(-0.5 * f_stat) if 0 < f_stat < 20 else (1.0 if f_stat <= 0 else 1e-6)

    return {'f_stat': float(f_stat), 'p_value': float(p_value),
            'significant': p_value < 0.05, 'rss_u': float(rss_u),
            'rss_r': float(rss_r), 'n': n}


# ---------- Multivariate R^2 test ----------

def multivariate_r2_test(telem_matrix, hidden_norms):
    """R^2 comparison: hidden ~ lag(hidden) vs hidden ~ lag(hidden) + lag(telem_pca)."""
    T = len(hidden_norms)
    if T < 10:
        return {'r2_restricted': 0.0, 'r2_unrestricted': 0.0, 'r2_improvement': 0.0}
    hn = np.array(hidden_norms, dtype=np.float64)
    tm = np.array(telem_matrix, dtype=np.float64)
    tm_c = tm - tm.mean(axis=0, keepdims=True)
    try:
        _, _, Vt = np.linalg.svd(tm_c, full_matrices=False)
        nc = min(3, Vt.shape[0])
        tpca = tm_c @ Vt[:nc].T
    except np.linalg.LinAlgError:
        tpca, nc = tm_c[:, :3], 3
    y, hn_lag, tp_lag = hn[1:], hn[:-1], tpca[:-1]
    n = len(y)
    ss_tot = max(np.sum((y - y.mean()) ** 2), 1e-12)

    X_r = np.column_stack([np.ones(n), hn_lag])
    r2_r = 1.0 - np.sum((y - X_r @ np.linalg.lstsq(X_r, y, rcond=None)[0]) ** 2) / ss_tot
    X_u = np.column_stack([np.ones(n), hn_lag, tp_lag])
    r2_u = 1.0 - np.sum((y - X_u @ np.linalg.lstsq(X_u, y, rcond=None)[0]) ** 2) / ss_tot

    imp = r2_u - r2_r
    return {'r2_restricted': float(r2_r), 'r2_unrestricted': float(r2_u),
            'r2_improvement': float(imp), 'pca_components': nc, 'pass': imp > 0.05}


# ---------- Transfer Entropy (binned estimator) ----------

def binned_entropy(counts):
    """H(X) from histogram counts."""
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def transfer_entropy(source, target, nb=10):
    """TE(source->target) via binned histogram. TE = H(Yt|Yt1) - H(Yt|Yt1,Xt1)."""
    s, t = np.array(source, dtype=np.float64), np.array(target, dtype=np.float64)
    N = min(len(s), len(t)) - 1
    if N < 20:
        return 0.0
    def dig(a):
        lo, hi = a.min(), a.max()
        if hi - lo < 1e-12: return np.zeros(len(a), dtype=int)
        return np.clip(((a - lo) / (hi - lo) * (nb - 1)).astype(int), 0, nb - 1)
    sb, tb = dig(s), dig(t)
    yt, yt1, xt1 = tb[1:], tb[:-1], sb[:-1]
    jyy = np.zeros((nb, nb))
    jyyx = np.zeros((nb, nb, nb))
    for i in range(N):
        jyy[yt[i], yt1[i]] += 1
        jyyx[yt[i], yt1[i], xt1[i]] += 1
    h1 = binned_entropy(jyy.ravel()) - binned_entropy(jyy.sum(axis=0))
    h2 = binned_entropy(jyyx.ravel()) - binned_entropy(jyyx.sum(axis=0).ravel())
    return float(max(0.0, h1 - h2))


# ---------- Training ----------

def train_embodied(model, data, telem, actuator, device):
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


# ---------- Condition evaluation ----------

def run_condition(label, model_src, data, telem, actuator, device,
                  use_film, use_real_telemetry, apply_interventions):
    """Collect rich time series under one condition."""
    print(f"\n{'='*60}\n  Condition {label}: film={use_film} real_telem={use_real_telemetry} "
          f"intervene={apply_interventions}\n{'='*60}")

    model = copy.deepcopy(model_src).to(device)
    model.eval(); model.enable_conditioning(use_film)

    prev_s = None
    telem_series, action_series = [], []
    hidden_norm_series, entropy_series, loss_series = [], [], []

    cur_idx, cur_perf = 0, PERF_CYCLE[0]
    if apply_interventions:
        try: actuator.set_performance_level(cur_perf)
        except Exception: pass

    with torch.no_grad():
        for bi in range(EVAL_BATCHES):
            if apply_interventions and bi > 0 and bi % SWITCH_INTERVAL == 0:
                cur_idx = (cur_idx + 1) % len(PERF_CYCLE)
                cur_perf = PERF_CYCLE[cur_idx]
                try: actuator.set_performance_level(cur_perf)
                except Exception: pass
                time.sleep(0.05)

            if use_real_telemetry:
                tv, prev_s = build_telemetry(telem, device, prev_s)
            else:
                tv = constant_telemetry(device)

            telem_series.append(tv.squeeze(0).cpu().numpy())

            x, y = get_batch(data, BS, SL, device)
            out = model(x, tv.expand(BS, -1), return_hidden=True)

            # Action
            mp = F.softmax(out['action_logits'], dim=-1).mean(dim=0)
            action_idx = torch.argmax(mp).item()
            action_series.append(action_idx)

            hidden_norm_series.append(out['hidden'].norm(dim=-1).mean().item())

            # Output entropy
            lp = F.log_softmax(out['logits'][:, -1, :], dim=-1)
            entropy_series.append(-(lp.exp() * lp).sum(-1).mean().item())

            # Loss
            loss = F.cross_entropy(out['logits'].view(-1, 256), y.view(-1))
            loss_series.append(loss.item())

            if (bi + 1) % 75 == 0:
                print(f"  [{label}] {bi+1}/{EVAL_BATCHES} act={action_idx} "
                      f"hn={hidden_norm_series[-1]:.2f} loss={loss_series[-1]:.4f}")

    try: actuator.set_performance_level(PerformanceLevel.BALANCED)
    except Exception: pass

    telem_matrix = np.array(telem_series)  # (T, 12)
    power_s = telem_matrix[:, 0].tolist()
    freq_s = telem_matrix[:, 2].tolist()
    temp_s = telem_matrix[:, 1].tolist()

    granger_power_action = granger_var2(power_s, action_series)
    granger_freq_action = granger_var2(freq_s, action_series)
    granger_temp_hidden = granger_var2(temp_s, hidden_norm_series)
    n_sig = sum(1 for g in [granger_power_action, granger_freq_action,
                             granger_temp_hidden] if g['significant'])
    r2_test = multivariate_r2_test(telem_matrix, hidden_norm_series)
    telem_mag = np.linalg.norm(telem_matrix, axis=1).tolist()
    te_telem_hidden = transfer_entropy(telem_mag, hidden_norm_series)
    te_hidden_telem = transfer_entropy(hidden_norm_series, telem_mag)

    te_dir = 'telem->hidden' if te_telem_hidden > te_hidden_telem else 'hidden->telem'
    metrics = {
        'condition': label, 'use_film': use_film,
        'use_real_telemetry': use_real_telemetry, 'apply_interventions': apply_interventions,
        'granger_power_action': granger_power_action,
        'granger_freq_action': granger_freq_action,
        'granger_temp_hidden': granger_temp_hidden,
        'n_significant_granger': n_sig, 'r2_test': r2_test,
        'transfer_entropy_telem_to_hidden': te_telem_hidden,
        'transfer_entropy_hidden_to_telem': te_hidden_telem,
        'te_causal_direction': te_dir,
        'mean_hidden_norm': float(np.mean(hidden_norm_series)),
        'mean_output_entropy': float(np.mean(entropy_series)),
        'mean_loss': float(np.mean(loss_series)),
        'action_distribution': np.bincount(action_series, minlength=4).tolist(),
        'power_series_first20': power_s[:20], 'action_series_first20': action_series[:20],
        'hidden_norm_first20': hidden_norm_series[:20],
        'power_series_last20': power_s[-20:], 'action_series_last20': action_series[-20:],
        'hidden_norm_last20': hidden_norm_series[-20:],
    }
    gpa, gfa, gth = granger_power_action, granger_freq_action, granger_temp_hidden
    print(f"  [{label}] Granger: pow->act F={gpa['f_stat']:.3f} p={gpa['p_value']:.4f} | "
          f"freq->act F={gfa['f_stat']:.3f} p={gfa['p_value']:.4f} | "
          f"temp->hid F={gth['f_stat']:.3f} p={gth['p_value']:.4f} | n_sig={n_sig}")
    print(f"  [{label}] R^2 imp={r2_test['r2_improvement']:.4f} | "
          f"TE(t->h)={te_telem_hidden:.4f} TE(h->t)={te_hidden_telem:.4f} -> {te_dir}")
    return metrics


# ---------- Main ----------

def main():
    print("=" * 70)
    print("  z1714: MULTIVARIATE TELEMETRY GRANGER CAUSALITY")
    print("  Fixing z1712: test telemetry FEATURES, not raw perf level")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}  VRAM: {props.total_memory / 1e9:.1f} GB")
    print(f"Device: {device}  BS={BS} SL={SL} EVAL_BATCHES={EVAL_BATCHES}")

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
            results[lbl] = run_condition(lbl, model, data, telem, actuator, device, **kw)
            if i < len(conds) - 1:
                print(f"\nCooldown {COOLDOWN}s...")
                try: actuator.set_performance_level(PerformanceLevel.BALANCED)
                except Exception: pass
                time.sleep(COOLDOWN)

        # --- Verdicts ---
        print(f"\n{'='*70}\n  VERDICTS -- TELEMETRY GRANGER CAUSALITY\n{'='*70}")
        verdicts = {}
        A, B, C = results['A'], results['B'], results['C']

        a_sig, b_sig = A['n_significant_granger'], B['n_significant_granger']
        te_fwd, te_bwd = A['transfer_entropy_telem_to_hidden'], A['transfer_entropy_hidden_to_telem']
        r2_imp = A['r2_test']['r2_improvement']

        v1 = a_sig > 0
        verdicts['granger_any_significant'] = {
            'pass': v1, 'a_n_sig': a_sig,
            'a_power_p': A['granger_power_action']['p_value'],
            'a_freq_p': A['granger_freq_action']['p_value'],
            'a_temp_p': A['granger_temp_hidden']['p_value'],
            'description': 'Any telemetry Granger-causes model variable in A (p<0.05)'}
        print(f"\n1. Any Granger sig (A): {'PASS' if v1 else 'FAIL'}  n_sig={a_sig} "
              f"(pow p={A['granger_power_action']['p_value']:.4f}, "
              f"freq p={A['granger_freq_action']['p_value']:.4f}, "
              f"temp p={A['granger_temp_hidden']['p_value']:.4f})")

        v2 = a_sig > b_sig
        verdicts['granger_a_more_than_b'] = {
            'pass': v2, 'a_n_sig': a_sig, 'b_n_sig': b_sig,
            'description': 'A has more significant Granger tests than B'}
        print(f"2. A more sig than B: {'PASS' if v2 else 'FAIL'}  A={a_sig} B={b_sig}")

        v3 = te_fwd > te_bwd
        verdicts['transfer_entropy_direction'] = {
            'pass': v3, 'te_telem_to_hidden': te_fwd, 'te_hidden_to_telem': te_bwd,
            'description': 'TE(telem->hidden) > TE(hidden->telem) in A'}
        print(f"3. TE direction: {'PASS' if v3 else 'FAIL'}  "
              f"TE(t->h)={te_fwd:.4f}  TE(h->t)={te_bwd:.4f}")

        v4 = r2_imp > 0.05
        verdicts['r2_improvement'] = {
            'pass': v4, 'r2_restricted': A['r2_test']['r2_restricted'],
            'r2_unrestricted': A['r2_test']['r2_unrestricted'],
            'r2_improvement': r2_imp,
            'description': 'R^2 improves >5% adding telemetry as predictor'}
        print(f"4. R^2 improve >5%: {'PASS' if v4 else 'FAIL'}  imp={r2_imp:.4f} "
              f"({A['r2_test']['r2_restricted']:.4f} -> {A['r2_test']['r2_unrestricted']:.4f})")

        passed = sum(v['pass'] for v in verdicts.values())
        total = len(verdicts)
        print(f"\n{'='*70}\n  OVERALL: {passed}/{total} passed")
        if passed >= 3:
            print("  CONCLUSION: Strong causal evidence -- telemetry FEATURES drive model state")
        elif passed >= 2:
            print("  CONCLUSION: Partial causal evidence for telemetry-driven embodiment")
        else:
            print("  CONCLUSION: Insufficient causal evidence")
        print(f"{'='*70}")

        # Summary table
        print(f"\n  {'Cond':<5} {'n_sig':<6} {'pow_p':<9} {'freq_p':<9} {'temp_p':<9} "
              f"{'R2_imp':<9} {'TE_fwd':<9} {'TE_bwd':<9}")
        for l in 'ABC':
            r = results[l]
            print(f"  {l:<5} {r['n_significant_granger']:<6} "
                  f"{r['granger_power_action']['p_value']:<9.4f} "
                  f"{r['granger_freq_action']['p_value']:<9.4f} "
                  f"{r['granger_temp_hidden']['p_value']:<9.4f} "
                  f"{r['r2_test']['r2_improvement']:<9.4f} "
                  f"{r['transfer_entropy_telem_to_hidden']:<9.4f} "
                  f"{r['transfer_entropy_hidden_to_telem']:<9.4f}")

        output = {
            'experiment': 'z1714_telemetry_granger',
            'description': 'Multivariate telemetry Granger causality -- fixing z1712 '
                           'by testing telemetry FEATURES not raw perf level',
            'timestamp': datetime.now().isoformat(),
            'device': str(device),
            'gpu_name': (torch.cuda.get_device_properties(0).name
                         if torch.cuda.is_available() else 'cpu'),
            'gpu_vram_bytes': (torch.cuda.get_device_properties(0).total_memory
                               if torch.cuda.is_available() else 0),
            'config': {'batch_size': BS, 'seq_len': SL, 'lr': LR,
                       'train_epochs': TRAIN_EPOCHS, 'train_steps': TRAIN_STEPS,
                       'eval_batches': EVAL_BATCHES, 'switch_interval': SWITCH_INTERVAL,
                       'granger_lag': 2, 'te_bins': 10, 'model_params': npar},
            'train_loss': train_loss,
            'conditions': results,
            'verdicts': verdicts,
            'passed': passed, 'total': total,
        }
        out_path = ROOT / 'results' / 'z1714_telemetry_granger.json'
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
