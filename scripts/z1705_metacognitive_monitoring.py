#!/usr/bin/env python3
"""
z1705: Metacognitive Monitoring

The model monitors its own prediction quality and adjusts processing
accordingly -- analogous to human metacognition ("I know that I don't know").

Architecture:
  A MetacognitiveHead sits on top of the MetabolicTransformer. It receives
  BOTH the last-token hidden state AND the 12-dim GPU telemetry vector, and
  outputs: (a) confidence in [0,1] -- will my prediction be correct?
           (b) predicted loss -- what will my cross-entropy be?

Training phases:
  Phase 1 (3 ep): Train base char-LM normally on TinyShakespeare
  Phase 2 (5 ep): Freeze base, train MetacognitiveHead to predict accuracy+loss
  Phase 3 (5 ep x 4): Full metacognitive loop under 4 conditions

4 Conditions:
  A: Full Metacognitive -- confidence drives perf level (low conf -> HIGH)
  B: Reactive Only      -- action = argmax(action_logits), no confidence
  C: Fixed Actions      -- always BALANCED, no metacognition
  D: Anti-calibrated    -- confidence inverted (high conf -> MORE compute)

Metrics: ECE, Brier, loss-prediction MSE, J/token, action entropy

Verdicts:
  1. PASS if ECE < 0.15          2. PASS if loss pred MSE < 0.5
  3. PASS if A J/tok < B J/tok   4. PASS if D ppl > A ppl
"""

import os, sys, time, json, copy
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

sys.path.insert(0, '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy')
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.metabolic.film_transformer import (
    MetabolicTransformer, MetabolicConfig, create_metabolic_transformer, get_best_device
)
from src.actuation.gpu_actuator import GPUActuator, PerformanceLevel
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry

DEVICE = get_best_device()
ROOT = Path('/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy')
BS, SL, LR, COOLDOWN = 4, 256, 3e-4, 30

ACTION_MAP = {
    0: PerformanceLevel.LOW, 1: PerformanceLevel.BALANCED,
    2: PerformanceLevel.HIGH, 3: PerformanceLevel.HIGH,
}


class MetacognitiveHead(nn.Module):
    """
    Monitors model's own prediction quality.

    Takes hidden_dim + telemetry_dim input because the model's confidence
    should depend on BOTH its internal state AND its body state. E.g.:
    "When the GPU is throttling AND I'm uncertain, predictions degrade."
    """
    def __init__(self, hidden_dim=256, telemetry_dim=12):
        super().__init__()
        # Will my next-token prediction be correct?
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim + telemetry_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
        # What will my cross-entropy loss be on this batch?
        self.loss_predictor = nn.Sequential(
            nn.Linear(hidden_dim + telemetry_dim, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.register_buffer('running_loss_mean', torch.tensor(3.0))
        self.register_buffer('running_loss_std', torch.tensor(1.0))

    def forward(self, hidden, telemetry):
        combined = torch.cat([hidden, telemetry], dim=-1)
        return {
            'confidence': self.confidence_head(combined),
            'predicted_loss': self.loss_predictor(combined),
        }


def load_data():
    path = ROOT / 'data' / 'tinyshakespeare.txt'
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    text = path.read_text(encoding='utf-8')
    data = torch.tensor(list(text.encode('utf-8')), dtype=torch.long)
    print(f"Loaded TinyShakespeare: {len(data):,} bytes")
    return data


def get_batch(data, bs, sl):
    starts = torch.randint(0, len(data) - sl - 1, (bs,))
    x = torch.stack([data[s:s+sl] for s in starts]).to(DEVICE)
    y = torch.stack([data[s+1:s+sl+1] for s in starts]).to(DEVICE)
    return x, y


def get_telemetry(telem):
    """Read GPU sensors into normalized 12-dim telemetry vector."""
    s = telem.read_sample()
    raw = [s.power_w/100, s.temp_edge_c/100, s.freq_sclk_mhz/3000,
           s.gpu_busy_pct/100, s.temp_junction_c/100, s.vram_used_gb/16]
    # Pad with zeros for derivative channels (not computed inline)
    raw += [0.0] * (12 - len(raw))
    return torch.tensor(raw, dtype=torch.float32, device=DEVICE).unsqueeze(0)


def ece(confs, accs, n_bins=10):
    """Expected Calibration Error: weighted |confidence - accuracy| across bins."""
    edges = np.linspace(0, 1, n_bins + 1)
    result, total = 0.0, len(confs)
    if total == 0: return 1.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confs >= lo) & (confs < hi)
        if mask.sum() == 0: continue
        result += mask.sum() / total * abs(confs[mask].mean() - accs[mask].mean())
    return float(result)


def brier(confs, outcomes):
    """Brier score: mean (confidence - outcome)^2. Lower is better."""
    return float(np.mean((confs - outcomes)**2)) if len(confs) > 0 else 1.0


def train_phase1(model, data, telem, n_epochs=3):
    """Phase 1: Train base LM normally."""
    print(f"\n--- Phase 1: Base LM training ({n_epochs} epochs) ---")
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    total, n = 0.0, 0
    for epoch in range(n_epochs):
        eloss, steps = 0.0, 80
        for step in range(steps):
            x, y = get_batch(data, BS, SL)
            tv = get_telemetry(telem).expand(BS, -1)
            out = model(x, telemetry=tv)
            loss = F.cross_entropy(out['logits'].view(-1, 256), y.view(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            eloss += loss.item(); n += 1
            if (step+1) % 50 == 0:
                print(f"  e{epoch+1} s{step+1}/{steps} loss={loss.item():.4f}")
        total += eloss
        print(f"  Epoch {epoch+1} avg loss: {eloss/steps:.4f}")
    return total / max(n, 1)


def train_phase2(model, meta, data, telem, n_epochs=5):
    """Phase 2: Train metacognitive head (base frozen)."""
    print(f"\n--- Phase 2: Metacognitive head training ({n_epochs} epochs) ---")
    for p in model.parameters(): p.requires_grad = False
    opt = torch.optim.Adam(meta.parameters(), lr=LR)
    all_c, all_a, lpe = [], [], []
    for epoch in range(n_epochs):
        ec, el, steps = 0.0, 0.0, 60
        for step in range(steps):
            x, y = get_batch(data, BS, SL)
            tv = get_telemetry(telem).expand(BS, -1)
            with torch.no_grad():
                out = model(x, telemetry=tv, return_hidden=True)
                hidden = out['hidden'][:, -1, :]
                tloss = F.cross_entropy(out['logits'].view(-1, 256), y.view(-1))
                bacc = (out['logits'].argmax(-1) == y).float().mean(-1, keepdim=True)
            mo = meta(hidden.detach(), tv.detach())
            closs = F.mse_loss(mo['confidence'], bacc)
            lloss = F.mse_loss(mo['predicted_loss'], tloss.detach().expand(BS, 1))
            total = closs + 0.5 * lloss
            opt.zero_grad(); total.backward(); opt.step()
            meta.running_loss_mean.lerp_(tloss.detach(), 0.1)
            meta.running_loss_std.lerp_((tloss.detach() - meta.running_loss_mean).abs(), 0.1)
            ec += closs.item(); el += lloss.item()
            all_c.extend(mo['confidence'].detach().cpu().numpy().flatten().tolist())
            all_a.extend(bacc.detach().cpu().numpy().flatten().tolist())
            lpe.append(lloss.item())
            if (step+1) % 50 == 0:
                print(f"  e{epoch+1} s{step+1}/{steps} cal={closs.item():.4f} lp={lloss.item():.4f}")
        print(f"  Epoch {epoch+1} calib={ec/steps:.4f} loss_pred={el/steps:.4f}")
    for p in model.parameters(): p.requires_grad = True
    c, a = np.array(all_c), np.array(all_a)
    r = {'ece': ece(c, a), 'brier': brier(c, a),
         'loss_pred_mse': float(np.mean(lpe[-20:])) if lpe else 1.0}
    print(f"  Phase 2 ECE={r['ece']:.4f} Brier={r['brier']:.4f} LP-MSE={r['loss_pred_mse']:.4f}")
    return r


def select_action(conf, anti=False):
    """Map confidence to performance level.
    Normal:  low confidence -> HIGH (more compute for hard inputs)
    Anti:    high confidence -> HIGH (deliberately wasteful)"""
    if anti:
        return 2 if conf > 0.6 else (1 if conf > 0.3 else 0)
    return 2 if conf < 0.3 else (1 if conf < 0.6 else 0)


def run_condition(label, model, meta, data, telem, actuator, mode, n_epochs=5):
    """
    Run one experimental condition with energy measurement.

    Each condition deep-copies the model so results are independent.
    Loss = task_loss + 0.3*calibration + 0.2*loss_prediction + energy_penalty
    """
    print(f"\n{'='*60}\n  Condition {label}: {mode.upper()}\n{'='*60}")
    cm = copy.deepcopy(model).to(DEVICE)
    cmeta = copy.deepcopy(meta).to(DEVICE)
    params = list(cm.parameters()) + list(cmeta.parameters())
    opt = torch.optim.Adam(params, lr=LR)
    confs, accs, losses, lp, la, acts = [], [], [], [], [], []
    telem.reset_accumulator()
    telem.start_continuous_sampling()
    t0, tokens = time.time(), 0

    for epoch in range(n_epochs):
        eloss, steps = 0.0, 60
        for step in range(steps):
            x, y = get_batch(data, BS, SL)
            tv = get_telemetry(telem).expand(BS, -1)
            out = cm(x, telemetry=tv, return_hidden=True)
            tloss = F.cross_entropy(out['logits'].view(-1, 256), y.view(-1))
            hidden = out['hidden'][:, -1, :]
            mo = cmeta(hidden.detach(), tv.detach())
            conf = mo['confidence'].mean().item()
            pred_l = mo['predicted_loss'].mean().item()
            correct = (out['logits'].argmax(-1) == y).float()
            bacc = correct.mean().item()

            # Action selection
            if mode == 'metacognitive':   action = select_action(conf)
            elif mode == 'anti':          action = select_action(conf, anti=True)
            elif mode == 'reactive':      action = min(out['action_logits'].argmax(-1)[0].item(), 3)
            else:                         action = 1
            try: actuator.set_performance_level(ACTION_MAP[action])
            except Exception: pass
            acts.append(action)

            # Losses
            bacc_t = correct.mean(-1, keepdim=True)
            closs = F.mse_loss(mo['confidence'], bacc_t.detach())
            lloss = F.mse_loss(mo['predicted_loss'], tloss.detach().expand(BS, 1))
            epenalty = torch.tensor(0.0, device=DEVICE)
            if mode in ('metacognitive', 'anti'):
                epenalty = 0.1 * mo['confidence'].mean() * (action / 3.0)
            total = tloss + 0.3*closs + 0.2*lloss + epenalty
            opt.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()

            eloss += tloss.item(); tokens += BS * SL
            confs.append(conf); accs.append(bacc)
            losses.append(tloss.item()); lp.append(pred_l); la.append(tloss.item())
            if (step+1) % 50 == 0:
                print(f"  [{label}] e{epoch+1} s{step+1}/{steps} "
                      f"loss={tloss.item():.4f} conf={conf:.3f} acc={bacc:.3f} act={action}")
        print(f"  [{label}] Epoch {epoch+1} avg_loss={eloss/steps:.4f}")

    elapsed = time.time() - t0
    telem.stop_continuous_sampling()
    energy = telem.get_accumulated_energy_j()
    try: actuator.set_performance_level(PerformanceLevel.BALANCED)
    except Exception: pass

    c, a, l = np.array(confs), np.array(accs), np.array(losses)
    lpa, laa = np.array(lp), np.array(la)
    ac = np.bincount(acts, minlength=4)
    ap = ac / max(len(acts), 1)
    aent = -np.sum(ap[ap > 0] * np.log(ap[ap > 0]))

    m = {
        'condition': label, 'mode': mode,
        'ece': ece(c, a), 'brier_score': brier(c, a),
        'loss_pred_mse': float(np.mean((lpa - laa)**2)),
        'final_perplexity': float(np.exp(np.mean(l[-20:]))),
        'final_loss': float(np.mean(l[-20:])),
        'energy_j': energy, 'j_per_token': energy / max(tokens, 1),
        'total_tokens': tokens, 'elapsed_s': elapsed,
        'action_distribution': ac.tolist(), 'action_entropy': float(aent),
        'mean_confidence': float(c.mean()), 'mean_accuracy': float(a.mean()),
    }
    print(f"  [{label}] ECE={m['ece']:.4f} Brier={m['brier_score']:.4f} "
          f"LP-MSE={m['loss_pred_mse']:.4f} PPL={m['final_perplexity']:.2f}")
    print(f"  [{label}] Energy={energy:.2f}J J/tok={m['j_per_token']:.6f} "
          f"Actions={ac.tolist()} H={aent:.4f}")
    return m


def main():
    print("="*70)
    print("  z1705: METACOGNITIVE MONITORING")
    print("  Model monitors its own prediction quality, adjusts computation")
    print("="*70)
    print(f"Device: {DEVICE}  BS={BS} SL={SL} Cooldown={COOLDOWN}s")

    data = load_data()
    telem = SysfsHwmonTelemetry(sample_rate_hz=20)
    actuator = GPUActuator(card_id=0)

    model = create_metabolic_transformer(
        hidden_dim=256, num_layers=6, num_heads=4, telemetry_dim=12).to(DEVICE)
    meta = MetacognitiveHead(hidden_dim=256, telemetry_dim=12).to(DEVICE)

    npar = sum(p.numel() for p in model.parameters())
    mpar = sum(p.numel() for p in meta.parameters())
    print(f"Base params: {npar:,}  Meta params: {mpar:,}")

    p1_loss = train_phase1(model, data, telem, n_epochs=3)
    print(f"Phase 1 done, avg loss: {p1_loss:.4f}")
    p2 = train_phase2(model, meta, data, telem, n_epochs=5)

    # Phase 3: 4 conditions
    conds = [('A','metacognitive'), ('B','reactive'), ('C','fixed'), ('D','anti')]
    results = {}
    for i, (lbl, mode) in enumerate(conds):
        results[lbl] = run_condition(lbl, model, meta, data, telem, actuator, mode, n_epochs=5)
        if i < len(conds) - 1:
            print(f"\nCooldown {COOLDOWN}s...")
            try: actuator.set_performance_level(PerformanceLevel.BALANCED)
            except Exception: pass
            time.sleep(COOLDOWN)
    try: actuator.set_performance_level(PerformanceLevel.BALANCED)
    except Exception: pass

    # Verdicts
    print(f"\n{'='*70}\n  VERDICTS\n{'='*70}")
    verdicts = {}

    ece_a = results['A']['ece']
    v1 = ece_a < 0.15
    verdicts['calibration'] = {'pass': v1, 'ece': ece_a, 'threshold': 0.15}
    print(f"\n1. Calibration (ECE < 0.15): {'PASS' if v1 else 'FAIL'}  ECE={ece_a:.4f}")

    lm = results['A']['loss_pred_mse']
    v2 = lm < 0.5
    verdicts['loss_prediction'] = {'pass': v2, 'mse': lm, 'threshold': 0.5}
    print(f"2. Loss prediction (MSE < 0.5): {'PASS' if v2 else 'FAIL'}  MSE={lm:.4f}")

    ja, jb = results['A']['j_per_token'], results['B']['j_per_token']
    v3 = ja < jb
    verdicts['energy_efficiency'] = {'pass': v3, 'a': ja, 'b': jb}
    print(f"3. Energy (A < B J/tok): {'PASS' if v3 else 'FAIL'}  A={ja:.6f} B={jb:.6f}")

    pa, pd = results['A']['final_perplexity'], results['D']['final_perplexity']
    v4 = pd > pa
    verdicts['anti_calibration'] = {'pass': v4, 'a_ppl': pa, 'd_ppl': pd}
    print(f"4. Anti-calibration hurts (D>A ppl): {'PASS' if v4 else 'FAIL'}  A={pa:.2f} D={pd:.2f}")

    passed = sum(v['pass'] for v in verdicts.values())
    print(f"\n{'='*70}\n  OVERALL: {passed}/{len(verdicts)} passed\n{'='*70}")

    # Metacognitive efficiency: accuracy gain per extra J (A vs C)
    ea, ec_ = results['A']['energy_j'], results['C']['energy_j']
    aa, ac = results['A']['mean_accuracy'], results['C']['mean_accuracy']
    meff = (aa - ac) / abs(ea - ec_) if abs(ea - ec_) > 0.01 else 0.0

    output = {
        'experiment': 'z1705_metacognitive_monitoring',
        'timestamp': datetime.now().isoformat(),
        'device': str(DEVICE),
        'config': {'batch_size': BS, 'seq_len': SL, 'lr': LR,
                   'hidden_dim': 256, 'num_layers': 6,
                   'base_params': npar, 'meta_params': mpar},
        'phase1_avg_loss': p1_loss,
        'phase2_metrics': p2,
        'conditions': results,
        'verdicts': verdicts,
        'passed': passed, 'total': len(verdicts),
        'metacognitive_efficiency': meff,
        'summary': {k: {'ppl': results[k]['final_perplexity'],
                         'j_tok': results[k]['j_per_token'],
                         'ece': results[k]['ece']}
                    for k in 'ABCD'},
    }

    out_path = ROOT / 'results' / 'z1705_metacognitive_monitoring.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    def jd(o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=jd)
    print(f"\nResults saved to: {out_path}")
    print("Done.")


if __name__ == '__main__':
    main()
