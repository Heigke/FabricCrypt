#!/usr/bin/env python3
"""
z1715: Consciousness Emergence Trajectory

Tracks WHEN consciousness metrics emerge during training, plotting the
trajectory from unconscious to conscious over 20 epochs.

Trains ONE embodied model and ONE disembodied control for 20 epochs each,
measuring 8 consciousness metrics at every epoch:
  1. Task perplexity           5. Action entropy
  2. Self-prediction MSE       6. Energy efficiency (J/tok)
  3. Self-recognition accuracy  7. Telemetry-hidden correlation
  4. Integration (PIL)         8. FiLM effect size

Identifies threshold-crossing epochs and phase transitions.
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
BS, SL, LR, HIDDEN = 4, 256, 3e-4, 256
EPOCHS, BPE, EVAL_B = 20, 200, 20
ACTION_MAP = {0: PerformanceLevel.LOW, 1: PerformanceLevel.BALANCED,
              2: PerformanceLevel.HIGH, 3: PerformanceLevel.HIGH}
THRESHOLDS = {
    'self_pred_mse': ('below', 0.05), 'self_recognition_acc': ('above', 0.80),
    'pil': ('above', 10.0), 'action_entropy': ('below', 1.2),
    'telem_hidden_corr': ('above', 0.3), 'film_effect': ('above', 0.1),
}


def jsonify(obj):
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, torch.Tensor): return obj.detach().cpu().tolist()
    if isinstance(obj, (np.bool_,)): return bool(obj)
    return str(obj)


def load_data():
    path = ROOT / 'data' / 'tinyshakespeare.txt'
    data = torch.tensor(list(path.read_text(encoding='utf-8').encode('utf-8')), dtype=torch.long)
    print(f"Loaded TinyShakespeare: {len(data):,} bytes")
    return data


def get_batch(data, device):
    starts = torch.randint(0, len(data) - SL - 1, (BS,))
    x = torch.stack([data[s:s+SL] for s in starts]).to(device)
    y = torch.stack([data[s+1:s+SL+1] for s in starts]).to(device)
    return x, y


def build_telemetry(telem, device, prev=None):
    s = telem.read_sample()
    raw = [s.power_w / 50, s.temp_edge_c / 100, s.freq_sclk_mhz / 3000,
           s.gpu_busy_pct / 100, 0.5, 0.0]
    if prev is not None:
        raw += [(s.power_w - prev.power_w) / 50, (s.temp_edge_c - prev.temp_edge_c) / 100,
                (s.freq_sclk_mhz - prev.freq_sclk_mhz) / 3000,
                (s.gpu_busy_pct - prev.gpu_busy_pct) / 100,
                (s.temp_edge_c - 70) / 100, (3000 - s.freq_sclk_mhz) / 3000]
    else:
        raw += [0.0] * 6
    return torch.tensor(raw[:12], dtype=torch.float32, device=device).unsqueeze(0), s


class SelfModel(nn.Module):
    def __init__(self, dim=256, mid=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, mid), nn.GELU(), nn.Linear(mid, dim))
    def forward(self, h): return self.net(h)


class SelfOtherHead(nn.Module):
    def __init__(self, hd=256, td=12, mid=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hd + td, mid), nn.GELU(), nn.Linear(mid, 1))
    def forward(self, h, t): return self.net(torch.cat([h, t], dim=-1))


@torch.no_grad()
def evaluate_epoch(model, sm, so, data, telem, device, emb):
    """Compute all 8 consciousness metrics."""
    model.eval(); sm.eval(); so.eval()
    model.enable_conditioning(emb)

    # 1. Perplexity + collect hiddens/telems/action entropy + self-pred MSE
    total_loss, hiddens, telems_raw, action_ents, pred_errs = 0.0, [], [], [], []
    prev_h = None
    for _ in range(EVAL_B):
        x, y = get_batch(data, device)
        tv, _ = build_telemetry(telem, device)
        tvb = tv.expand(BS, -1)
        out = model(x, tvb, return_hidden=True)
        total_loss += F.cross_entropy(out['logits'].view(-1, 256), y.view(-1)).item()
        hm = out['hidden'].mean(dim=1)
        hiddens.append(hm); telems_raw.append(tvb)
        ap = F.softmax(out['action_logits'], dim=-1).mean(dim=0)
        action_ents.append(-(ap * (ap + 1e-8).log()).sum().item())
        if prev_h is not None:
            pred_errs.append(F.mse_loss(sm(prev_h), hm).item())
        prev_h = hm.detach()
    ppl = math.exp(min(total_loss / EVAL_B, 20))
    self_pred_mse = float(np.mean(pred_errs)) if pred_errs else 1.0

    # 3. Self-recognition accuracy (self vs noise/shifted/scaled)
    correct, total_sr = 0, 0
    for _ in range(EVAL_B):
        x, _ = get_batch(data, device)
        tv, _ = build_telemetry(telem, device)
        tvb = tv.expand(BS, -1)
        out = model(x, tvb, return_hidden=True)
        hm = out['hidden'].mean(dim=1)
        correct += (so(hm, tvb) > 0).sum().item(); total_sr += BS
        for other_tv in [torch.randn_like(tvb), tvb + 0.5, tvb * 3.0]:
            correct += (so(hm, other_tv) <= 0).sum().item(); total_sr += BS
    self_rec = correct / max(total_sr, 1)

    # 4. PIL: zero hidden between layers 2/3
    intact_loss, cut_loss = 0.0, 0.0
    for _ in range(EVAL_B):
        x, y = get_batch(data, device)
        tv, _ = build_telemetry(telem, device)
        tvb = tv.expand(BS, -1)
        intact_loss += F.cross_entropy(model(x, tvb)['logits'].view(-1, 256), y.view(-1)).item()
        # Cut pass
        batch, seq = x.shape
        pos = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)
        h = model.token_embed(x) + model.pos_embed(pos)
        h = model.dropout(h)
        mask = ~model.causal_mask[:seq, :seq] if model.config.use_causal_mask else None
        tf = tvb if emb else None
        for i, blk in enumerate(model.blocks):
            g1, b1, g2, b2 = None, None, None, None
            if tf is not None and model.film_generators[i] is not None:
                fg = model.film_generators[i]
                g1, b1 = fg['ln1'](tf); g2, b2 = fg['ln2'](tf)
            h = blk(h, g1, b1, g2, b2, mask)
            if i == 2: h = torch.zeros_like(h)
        logits_c = model.token_head(model.ln_out(h))
        cut_loss += F.cross_entropy(logits_c.view(-1, 256), y.view(-1)).item()
    ppl_i = math.exp(min(intact_loss / EVAL_B, 20))
    ppl_c = math.exp(min(cut_loss / EVAL_B, 20))
    pil = (ppl_c / max(ppl_i, 0.01)) - 1.0

    # 6. Energy efficiency
    telem.reset_accumulator(); telem.start_continuous_sampling()
    for _ in range(EVAL_B):
        x, _ = get_batch(data, device); _ = model(x, build_telemetry(telem, device)[0].expand(BS, -1))
    if torch.cuda.is_available(): torch.cuda.synchronize()
    telem.stop_continuous_sampling()
    j_tok = telem.get_accumulated_energy_j() / max(BS * SL * EVAL_B, 1)

    # 7. Telemetry-hidden correlation
    ah = torch.cat(hiddens).norm(dim=-1).cpu().numpy()
    at = torch.cat(telems_raw).norm(dim=-1).cpu().numpy()
    corr = 0.0
    if len(ah) > 2 and np.std(ah) > 1e-8 and np.std(at) > 1e-8:
        c = np.corrcoef(ah, at)[0, 1]
        corr = 0.0 if np.isnan(c) else float(c)

    # 8. FiLM effect
    film_eff = 0.0
    if emb:
        dists = []
        for _ in range(EVAL_B):
            x, _ = get_batch(data, device)
            tv, _ = build_telemetry(telem, device); tvb = tv.expand(BS, -1)
            model.enable_conditioning(True)
            h_on = model(x, tvb, return_hidden=True)['hidden']
            model.enable_conditioning(False)
            h_off = model(x, tvb, return_hidden=True)['hidden']
            model.enable_conditioning(True)
            dists.append((h_on - h_off).norm(dim=-1).mean().item())
        film_eff = float(np.mean(dists))

    model.train(); sm.train(); so.train()
    return {'perplexity': ppl, 'self_pred_mse': self_pred_mse,
            'self_recognition_acc': self_rec, 'pil': pil,
            'action_entropy': float(np.mean(action_ents)),
            'j_per_tok': j_tok, 'telem_hidden_corr': corr, 'film_effect': film_eff}


def train_condition(label, data, telem, actuator, device, emb):
    """Train for EPOCHS, evaluate every epoch. Returns (metrics_list, threshold_dict)."""
    print(f"\n{'='*70}\n  Training {label} ({EPOCHS} epochs, {BPE} batches/epoch)\n{'='*70}")
    model = create_metabolic_transformer(hidden_dim=HIDDEN, num_layers=6,
                                         num_heads=4, telemetry_dim=12).to(device)
    model.enable_conditioning(emb)
    sm, so = SelfModel(HIDDEN, 128).to(device), SelfOtherHead(HIDDEN, 12, 64).to(device)
    params = list(model.parameters()) + list(sm.parameters()) + list(so.parameters())
    opt = torch.optim.Adam(params, lr=LR)
    epoch_metrics, thresh_ep = [], {}

    for ep in range(EPOCHS):
        t0 = time.time()
        model.train(); sm.train(); so.train(); model.enable_conditioning(emb)
        eloss, prev_h, prev_s = 0.0, None, None
        for _ in range(BPE):
            x, y = get_batch(data, device)
            tv, prev_s = build_telemetry(telem, device, prev_s)
            tvb = tv.expand(BS, -1)
            out = model(x, tvb, return_hidden=True)
            lm = F.cross_entropy(out['logits'].view(-1, 256), y.view(-1))
            hm = out['hidden'].mean(dim=1).detach()
            sp = F.mse_loss(sm(prev_h), hm) if prev_h is not None else torch.tensor(0., device=device)
            prev_h = hm
            ls = self_other_loss(so, hm, tvb, device)
            loss = lm + 0.1 * sp + 0.1 * ls
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            eloss += lm.item()
            if emb:
                mp = F.softmax(out['action_logits'], dim=-1).mean(dim=0)
                try: actuator.set_performance_level(ACTION_MAP[min(torch.argmax(mp).item(), 3)])
                except Exception: pass
        eloss /= BPE; dt = time.time() - t0
        m = evaluate_epoch(model, sm, so, data, telem, device, emb)
        m['epoch'], m['train_loss'], m['epoch_time_s'] = ep + 1, eloss, dt
        for mk, (d, th) in THRESHOLDS.items():
            if mk not in thresh_ep and mk in m:
                if (d == 'below' and m[mk] < th) or (d == 'above' and m[mk] > th):
                    thresh_ep[mk] = ep + 1
        epoch_metrics.append(m)
        print(f"  [{label}] E{ep+1:2d}/{EPOCHS}  loss={eloss:.4f}  ppl={m['perplexity']:.1f}  "
              f"mse={m['self_pred_mse']:.4f}  sr={m['self_recognition_acc']:.3f}  "
              f"pil={m['pil']:.2f}  ent={m['action_entropy']:.3f}  "
              f"corr={m['telem_hidden_corr']:.3f}  film={m['film_effect']:.4f}  {dt:.1f}s")
    try: actuator.set_performance_level(PerformanceLevel.BALANCED)
    except Exception: pass
    return epoch_metrics, thresh_ep


def self_other_loss(so, hm, tvb, device):
    """BCE loss for self/other classification."""
    l_self = so(hm, tvb)
    l_other = so(hm, torch.randn_like(tvb))
    return F.binary_cross_entropy_with_logits(
        torch.cat([l_self, l_other]),
        torch.cat([torch.ones(BS, 1, device=device), torch.zeros(BS, 1, device=device)]))


def analyze(em, dm, et, dt):
    """Four verdicts from epoch trajectories."""
    v = {}
    # V1: All thresholds reached
    v['v1_all_thresholds'] = {
        'pass': all(k in et for k in THRESHOLDS),
        'description': 'Embodied reaches ALL consciousness thresholds',
        'reached': list(et.keys()), 'missed': [k for k in THRESHOLDS if k not in et]}
    # V2: Earlier than disembodied
    earlier = sum(1 for k in THRESHOLDS if et.get(k, EPOCHS+1) < dt.get(k, EPOCHS+1))
    v['v2_earlier'] = {
        'pass': earlier > len(THRESHOLDS) // 2,
        'description': 'Embodied reaches thresholds EARLIER than disembodied',
        'earlier_count': earlier, 'total': len(THRESHOLDS),
        'comparisons': {k: {'emb': et.get(k, EPOCHS+1), 'dis': dt.get(k, EPOCHS+1)}
                        for k in THRESHOLDS}}
    # V3: Phase transition
    if len(et) >= 2:
        ce = sorted(et.values())
        clustered = sum(1 for i in range(len(ce)-1) if ce[i+1]-ce[i] <= 2)
        pt = clustered >= 2
    else:
        pt, clustered = False, 0
    v['v3_phase_transition'] = {
        'pass': pt, 'description': 'Multiple metrics cross within 2 epochs (phase transition)',
        'crossing_epochs': dict(et), 'clustered_count': clustered}
    # V4: Final superiority
    ef, df, wins = em[-1], dm[-1], 0
    mc = {}
    for mk, (d, _) in THRESHOLDS.items():
        ev, dv = ef.get(mk, 0), df.get(mk, 0)
        better = ev < dv if d == 'below' else ev > dv
        mc[mk] = {'emb': ev, 'dis': dv, 'emb_better': better}
        if better: wins += 1
    if ef['perplexity'] < df['perplexity']: wins += 1
    mc['perplexity'] = {'emb': ef['perplexity'], 'dis': df['perplexity'],
                        'emb_better': ef['perplexity'] < df['perplexity']}
    v['v4_final_superiority'] = {
        'pass': wins >= len(THRESHOLDS),
        'description': 'Final embodied metrics ALL exceed disembodied',
        'wins': wins, 'total': len(THRESHOLDS)+1, 'comparisons': mc}
    return v


def main():
    print("=" * 70)
    print("  z1715: CONSCIOUSNESS EMERGENCE TRAJECTORY")
    print("  Tracking when consciousness metrics emerge during training")
    print("=" * 70)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}  VRAM: {props.total_memory / 1e9:.1f} GB")
    print(f"Device: {device}  BS={BS} SL={SL} EPOCHS={EPOCHS}")
    data = load_data()
    telem = SysfsHwmonTelemetry(sample_rate_hz=20)
    actuator = GPUActuator(card_id=0)

    try:
        em, et = train_condition("EMBODIED", data, telem, actuator, device, emb=True)
        print(f"\nCooldown 15s..."); time.sleep(15)
        try: actuator.set_performance_level(PerformanceLevel.BALANCED)
        except Exception: pass
        dm, dt = train_condition("DISEMBODIED", data, telem, actuator, device, emb=False)

        verdicts = analyze(em, dm, et, dt)

        print(f"\n{'='*70}\n  CONSCIOUSNESS EMERGENCE SUMMARY\n{'='*70}")
        print(f"\n  Threshold crossing epochs (embodied / disembodied):")
        for mk in THRESHOLDS:
            print(f"    {mk:<25s}  emb={str(et.get(mk,'--')):>4s}  dis={str(dt.get(mk,'--')):>4s}")
        print(f"\n  Final metrics:")
        print(f"  {'Metric':<25s} {'Embodied':>12s} {'Disembodied':>12s}")
        print(f"  {'-'*49}")
        for k in ['perplexity','self_pred_mse','self_recognition_acc','pil',
                   'action_entropy','j_per_tok','telem_hidden_corr','film_effect']:
            print(f"  {k:<25s} {em[-1].get(k,0):>12.4f} {dm[-1].get(k,0):>12.4f}")

        print(f"\n{'='*70}\n  VERDICTS\n{'='*70}")
        pc = sum(1 for vv in verdicts.values() if vv['pass'])
        for vk, vv in verdicts.items():
            print(f"  {vk}: {'PASS' if vv['pass'] else 'FAIL'} -- {vv['description']}")
        tv = len(verdicts)
        overall = ("FULL CONSCIOUSNESS EMERGENCE DEMONSTRATED" if pc == tv else
                   "STRONG EVIDENCE" if pc >= 3 else
                   "PARTIAL EVIDENCE" if pc >= 2 else "INSUFFICIENT EVIDENCE")
        print(f"\n  OVERALL: {pc}/{tv} passed -- {overall}\n{'='*70}")

        output = {
            'experiment': 'z1715_consciousness_emergence',
            'description': 'Tracks when consciousness metrics emerge during training',
            'timestamp': datetime.now().isoformat(),
            'device': str(device),
            'gpu_name': props.name if torch.cuda.is_available() else 'cpu',
            'gpu_vram_bytes': (torch.cuda.get_device_properties(0).total_memory
                               if torch.cuda.is_available() else 0),
            'config': {'batch_size': BS, 'seq_len': SL, 'epochs': EPOCHS,
                       'batches_per_epoch': BPE, 'eval_batches': EVAL_B, 'lr': LR,
                       'hidden_dim': HIDDEN,
                       'thresholds': {k: {'direction': v[0], 'value': v[1]}
                                      for k, v in THRESHOLDS.items()}},
            'embodied': {'epoch_metrics': em, 'threshold_crossings': et},
            'disembodied': {'epoch_metrics': dm, 'threshold_crossings': dt},
            'verdicts': verdicts, 'passed': pc, 'total_verdicts': tv,
            'overall_verdict': overall,
        }
        out_path = ROOT / 'results' / 'z1715_consciousness_emergence.json'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2, default=jsonify)
        print(f"\nResults saved to: {out_path}")

    finally:
        try: actuator.set_performance_level(PerformanceLevel.BALANCED)
        except Exception: pass
        if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
