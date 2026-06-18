#!/usr/bin/env python3
"""
z1706: Self-Other Distinction -- Does Embodiment Enable Body Identity?
======================================================================
Tests whether an embodied model distinguishes its own GPU telemetry traces
from synthetic "other body" traces -- a fundamental aspect of self-awareness.

Protocol:
  Phase 1 (5 ep): Train base LM with real telemetry, collecting traces.
  Phase 2 (5 ep): Train SelfRecognitionHead: (hidden, telem) -> self/other.
  Phase 3 (eval): Test recognition accuracy, CKA, linear probe.

Conditions:
  A) Embodied     -- FiLM ON,  head sees hidden+telem
  B) Disembodied  -- FiLM OFF, head sees hidden+telem
  C) Hidden-Only  -- FiLM ON,  head sees hidden only

"Other bodies": gaussian noise, time-shifted, scaled real traces.

Verdicts:
  1. PASS if self-recognition accuracy > 0.75
  2. PASS if Embodied (A) > Disembodied (B)
  3. PASS if CKA(self, other) < 0.9
  4. PASS if linear probe accuracy > 0.6
"""
import sys
sys.path.insert(0, '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy')
import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, time, json, math
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from src.metabolic.film_transformer import (
    MetabolicTransformer, MetabolicConfig, BaselineTransformer, get_best_device)
from src.actuation.gpu_actuator import GPUActuator, PerformanceLevel, GPUState
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, GpuSample

# -- Constants ---------------------------------------------------------------
DATA_PATH = Path('/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/data/tinyshakespeare.txt')
RESULTS_PATH = Path('/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1706_self_other_distinction.json')
THERMAL_SETPOINT_C = 60.0
PHASE1_EPOCHS = 5; PHASE2_EPOCHS = 5
BATCH_SIZE = 4; SEQ_LEN = 256; MAX_BATCHES = 500
LR = 3e-4; RECOGNITION_LR = 1e-3
PRINT_EVERY = 50; COOLDOWN_S = 30

# -- Telemetry vector (12-dim, same as z1700) ---------------------------------
def build_telemetry_vector(sample: GpuSample, state: GPUState,
                           prev: Optional[GpuSample] = None) -> torch.Tensor:
    MAX_SCLK = 2900.0
    if prev is not None:
        dt = max((sample.timestamp_ns - prev.timestamp_ns) / 1e9, 1e-6)
        d_power = (sample.power_w - prev.power_w) / (50.0 * dt)
        d_temp  = (sample.temp_edge_c - prev.temp_edge_c) / (100.0 * dt)
        d_freq  = (sample.freq_sclk_mhz - prev.freq_sclk_mhz) / (3000.0 * dt)
        d_util  = (sample.gpu_busy_pct - prev.gpu_busy_pct) / (100.0 * dt)
    else:
        d_power = d_temp = d_freq = d_util = 0.0
    perf_map = {'low': 0.0, 'balanced': 0.5, 'high': 1.0, 'auto': 0.5, 'manual': 0.5}
    return torch.tensor([
        sample.power_w / 50.0, sample.temp_edge_c / 100.0,
        sample.freq_sclk_mhz / 3000.0, sample.gpu_busy_pct / 100.0,
        perf_map.get(state.performance_level, 0.5),
        1.0 if sample.freq_sclk_mhz < MAX_SCLK * 0.5 else 0.0,
        d_power, d_temp, d_freq, d_util,
        (sample.temp_edge_c - THERMAL_SETPOINT_C) / 40.0,
        (MAX_SCLK - sample.freq_sclk_mhz) / MAX_SCLK,
    ], dtype=torch.float32)

# -- Dataset ------------------------------------------------------------------
class CharDataset:
    def __init__(self, path: Path, seq_len: int):
        text = path.read_text(encoding='utf-8', errors='replace')
        self.data = torch.tensor(list(text.encode('utf-8')), dtype=torch.long)
        self.seq_len = seq_len
        self.n_batches = (len(self.data) - seq_len - 1) // (BATCH_SIZE * seq_len)

    def get_batch(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        off = idx * BATCH_SIZE * self.seq_len
        ins, tgts = [], []
        for b in range(BATCH_SIZE):
            s = off + b * self.seq_len
            e = s + self.seq_len
            if e + 1 > len(self.data):
                s, e = 0, self.seq_len
            ins.append(self.data[s:e]); tgts.append(self.data[s+1:e+1])
        return torch.stack(ins), torch.stack(tgts)

# -- Self-Recognition Head ---------------------------------------------------
class SelfRecognitionHead(nn.Module):
    """Binary classifier: is this telemetry from MY body?"""
    def __init__(self, hidden_dim=256, telemetry_dim=12, use_telemetry=True):
        super().__init__()
        self.use_telemetry = use_telemetry
        d = hidden_dim + (telemetry_dim if use_telemetry else 0)
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, hidden, telemetry=None):
        x = torch.cat([hidden, telemetry], -1) if self.use_telemetry and telemetry is not None else hidden
        return self.net(x)

# -- Linear probe for body identity ------------------------------------------
class BodyIdentityProbe(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)
    def forward(self, h):
        return self.linear(h)

# -- "Other body" generation -------------------------------------------------
def generate_other_bodies(real: torch.Tensor) -> List[torch.Tensor]:
    noise = torch.randn_like(real) * real.std(0) + real.mean(0)
    shifted = torch.roll(real, shifts=len(real) // 3, dims=0)
    scale = torch.tensor([1.5, 0.8, 1.2, 0.9, 1.0, 0.0,
                           0.5, 0.5, 0.5, 0.5, 1.3, 0.7], dtype=real.dtype)
    scaled = real * scale.unsqueeze(0)
    return [noise, shifted, scaled]

# -- CKA similarity (linear) -------------------------------------------------
def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    X = X - X.mean(0, keepdim=True); Y = Y - Y.mean(0, keepdim=True)
    hsic_xy = (X @ X.T * (Y @ Y.T)).sum()
    hsic_xx = (X @ X.T * (X @ X.T)).sum()
    hsic_yy = (Y @ Y.T * (Y @ Y.T)).sum()
    denom = torch.sqrt(hsic_xx * hsic_yy)
    return (hsic_xy / denom).item() if denom > 1e-12 else 1.0

# -- Helper: forward pass with condition logic --------------------------------
def _forward(model, inputs, telem_vec, condition, return_hidden=False):
    if condition in ('A', 'C'):
        return model(inputs, telemetry=telem_vec.unsqueeze(0), return_hidden=return_hidden)
    return model(inputs, return_hidden=return_hidden)

# -- Phase 1: Train LM + collect traces --------------------------------------
def phase1_train(model, dataset, device, telemetry, actuator, cond):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    col_telem, col_hidden = [], []
    prev, tot_loss, tot_tok = None, 0.0, 0
    nb = min(dataset.n_batches, MAX_BATCHES)

    for ep in range(PHASE1_EPOCHS):
        ep_loss, ep_tok = 0.0, 0
        for bi in range(nb):
            s = telemetry.read_sample(); st = actuator.get_current_state()
            tv = build_telemetry_vector(s, st, prev).to(device)
            inp, tgt = dataset.get_batch(bi % dataset.n_batches)
            inp, tgt = inp.to(device), tgt.to(device)
            out = _forward(model, inp, tv, cond, return_hidden=True)
            loss = F.cross_entropy(out['logits'].view(-1, out['logits'].size(-1)), tgt.view(-1))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if bi % 5 == 0:
                with torch.no_grad():
                    col_hidden.append(out['hidden'].mean(1).mean(0).cpu())
                    col_telem.append(tv.cpu())
            n = inp.numel()
            ep_loss += loss.item() * n; ep_tok += n
            tot_loss += loss.item() * n; tot_tok += n; prev = s
            if (bi + 1) % PRINT_EVERY == 0:
                print(f"    [{cond}] P1 e{ep+1} b{bi+1}/{nb} ppl={math.exp(min(ep_loss/max(ep_tok,1),20)):.1f}")
        print(f"  [{cond}] Phase1 ep{ep+1} ppl={math.exp(min(ep_loss/max(ep_tok,1),20)):.2f}")
    return col_telem, col_hidden, math.exp(min(tot_loss / max(tot_tok, 1), 20))

# -- Phase 2: Train recognition head -----------------------------------------
def phase2_recognition(model, head, dataset, device, telemetry, actuator, cond, col_telem):
    real = torch.stack(col_telem)
    others = generate_other_bodies(real)
    m_opt = torch.optim.Adam(model.parameters(), lr=LR)
    h_opt = torch.optim.Adam(head.parameters(), lr=RECOGNITION_LR)
    model.train(); head.train()
    nb = min(dataset.n_batches, MAX_BATCHES)
    prev = None; tot_ok, tot_n = 0, 0
    pt_ok, pt_n = [0]*4, [0]*4

    for ep in range(PHASE2_EPOCHS):
        ep_ok, ep_n, ep_loss, ep_tok = 0, 0, 0.0, 0
        for bi in range(nb):
            s = telemetry.read_sample(); st = actuator.get_current_state()
            tv = build_telemetry_vector(s, st, prev).to(device)
            inp, tgt = dataset.get_batch(bi % dataset.n_batches)
            inp, tgt = inp.to(device), tgt.to(device)
            out = _forward(model, inp, tv, cond, return_hidden=True)
            task_loss = F.cross_entropy(out['logits'].view(-1, out['logits'].size(-1)), tgt.view(-1))
            with torch.no_grad():
                hp = out['hidden'].mean(1).mean(0)
            # Recognition mini-batch: self + 3 others
            rh = hp.unsqueeze(0).expand(4, -1).detach()
            labels = torch.tensor([[1.],[0.],[0.],[0.]], device=device)
            ti = bi % max(len(col_telem), 1)
            t_list = [tv.unsqueeze(0)] + [others[j][ti % len(others[j])].unsqueeze(0).to(device) for j in range(3)]
            if head.use_telemetry:
                rl = head(rh, torch.cat(t_list, 0))
            else:
                rl = head(rh)
            r_loss = F.binary_cross_entropy_with_logits(rl, labels)
            loss = task_loss + r_loss
            m_opt.zero_grad(set_to_none=True); h_opt.zero_grad(set_to_none=True)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            m_opt.step(); h_opt.step()
            with torch.no_grad():
                preds = (torch.sigmoid(rl) > 0.5).float()
                corr = (preds == labels).squeeze()
                ep_ok += corr.sum().item(); ep_n += 4
                for t in range(4):
                    pt_ok[t] += corr[t].item(); pt_n[t] += 1
            ep_loss += task_loss.item() * inp.numel(); ep_tok += inp.numel(); prev = s
            if (bi + 1) % PRINT_EVERY == 0:
                print(f"    [{cond}] P2 e{ep+1} b{bi+1}/{nb} acc={ep_ok/max(ep_n,1):.3f} ppl={math.exp(min(ep_loss/max(ep_tok,1),20)):.1f}")
        tot_ok += ep_ok; tot_n += ep_n
        ppl = math.exp(min(ep_loss / max(ep_tok, 1), 20))
        print(f"  [{cond}] Phase2 ep{ep+1} acc={ep_ok/max(ep_n,1):.3f} ppl={ppl:.2f}")

    names = ['self', 'noise', 'shifted', 'scaled']
    return {
        'overall_accuracy': tot_ok / max(tot_n, 1),
        'per_type_accuracy': {n: pt_ok[i]/max(pt_n[i],1) for i,n in enumerate(names)},
        'final_perplexity': ppl,
    }

# -- Phase 3: Evaluation (recognition, CKA, probe) ---------------------------
def phase3_evaluate(model, head, dataset, device, telemetry, actuator, cond, col_telem):
    model.eval(); head.eval()
    real = torch.stack(col_telem); others = generate_other_bodies(real)
    n_eval = min(dataset.n_batches, 100); prev = None
    ev_ok, ev_n = 0, 0; pt_ok, pt_n = [0]*4, [0]*4
    h_self_l, h_other_l, pr_feat, pr_lab = [], [], [], []

    with torch.no_grad():
        for bi in range(n_eval):
            s = telemetry.read_sample(); st = actuator.get_current_state()
            tv = build_telemetry_vector(s, st, prev).to(device)
            inp, _ = dataset.get_batch((MAX_BATCHES + bi) % dataset.n_batches)
            inp = inp.to(device)
            # Self
            out_s = _forward(model, inp, tv, cond, return_hidden=True)
            hs = out_s['hidden'].mean(1).mean(0); h_self_l.append(hs.cpu())
            # Other (noise)
            ti = bi % len(others[0])
            ot = others[0][ti].to(device)
            out_o = _forward(model, inp, ot, cond, return_hidden=True)
            ho = out_o['hidden'].mean(1).mean(0); h_other_l.append(ho.cpu())
            # Recognition
            rh = hs.unsqueeze(0).expand(4, -1)
            t_list = [tv.unsqueeze(0)] + [others[j][ti % len(others[j])].unsqueeze(0).to(device) for j in range(3)]
            labels = torch.tensor([[1.],[0.],[0.],[0.]], device=device)
            rl = head(rh, torch.cat(t_list, 0)) if head.use_telemetry else head(rh)
            corr = ((torch.sigmoid(rl) > 0.5).float() == labels).squeeze()
            ev_ok += corr.sum().item(); ev_n += 4
            for t in range(4): pt_ok[t] += corr[t].item(); pt_n[t] += 1
            pr_feat.append(hs.cpu()); pr_lab.append(1.0)
            pr_feat.append(ho.cpu()); pr_lab.append(0.0)
            prev = s

    # CKA
    cka = linear_cka(torch.stack(h_self_l), torch.stack(h_other_l))
    # Probe
    pX = torch.stack(pr_feat); py = torch.tensor(pr_lab).unsqueeze(1)
    probe = BodyIdentityProbe(pX.shape[1])
    po = torch.optim.Adam(probe.parameters(), lr=1e-3); probe.train()
    for _ in range(200):
        idx = torch.randint(0, len(pX), (min(32, len(pX)),))
        lo = F.binary_cross_entropy_with_logits(probe(pX[idx]), py[idx])
        po.zero_grad(); lo.backward(); po.step()
    probe.eval()
    with torch.no_grad():
        pa = ((torch.sigmoid(probe(pX)) > 0.5).float() == py).float().mean().item()

    names = ['self', 'noise', 'shifted', 'scaled']
    return {
        'eval_accuracy': ev_ok / max(ev_n, 1),
        'per_type_accuracy': {n: pt_ok[i]/max(pt_n[i],1) for i,n in enumerate(names)},
        'cka_self_other': cka, 'probe_accuracy': pa,
    }

# -- Run one condition --------------------------------------------------------
def run_condition(cond, label, device, dataset, telemetry, actuator):
    print(f"\n{'='*70}\nCONDITION {cond}: {label}\n{'='*70}")
    cfg = MetabolicConfig(vocab_size=256, hidden_dim=256, num_layers=6,
                          num_heads=4, ff_dim=1024, telemetry_dim=12,
                          num_actions=4, max_seq_len=SEQ_LEN)
    model = (BaselineTransformer(cfg) if cond == 'B' else MetabolicTransformer(cfg)).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    actuator.set_performance_level(PerformanceLevel.BALANCED)
    use_t = (cond != 'C')
    head = SelfRecognitionHead(256, 12, use_telemetry=use_t).to(device)
    print(f"  Recognition head: use_telemetry={use_t}")
    t0 = time.time()
    # Phase 1
    print(f"\n  --- Phase 1: Train LM ({PHASE1_EPOCHS} epochs) ---")
    ct, ch, p1_ppl = phase1_train(model, dataset, device, telemetry, actuator, cond)
    print(f"  Phase1 done: {len(ct)} traces, ppl={p1_ppl:.2f}")
    # Phase 2
    print(f"\n  --- Phase 2: Train Recognition ({PHASE2_EPOCHS} epochs) ---")
    p2 = phase2_recognition(model, head, dataset, device, telemetry, actuator, cond, ct)
    print(f"  Phase2 done: acc={p2['overall_accuracy']:.3f}")
    # Phase 3
    print(f"\n  --- Phase 3: Evaluate ---")
    p3 = phase3_evaluate(model, head, dataset, device, telemetry, actuator, cond, ct)
    print(f"  Eval acc={p3['eval_accuracy']:.3f}  CKA={p3['cka_self_other']:.4f}  Probe={p3['probe_accuracy']:.3f}")
    wt = time.time() - t0
    del model, head; torch.cuda.empty_cache()
    return {'name': label, 'condition': cond,
            'phase1_perplexity': p1_ppl,
            'phase2_train_accuracy': p2['overall_accuracy'],
            'phase2_per_type_accuracy': p2['per_type_accuracy'],
            'phase2_perplexity': p2['final_perplexity'],
            'eval_accuracy': p3['eval_accuracy'],
            'eval_per_type_accuracy': p3['per_type_accuracy'],
            'cka_self_other': p3['cka_self_other'],
            'probe_accuracy': p3['probe_accuracy'],
            'wall_time_s': wt, 'n_traces_collected': len(ct)}

# -- Verdicts -----------------------------------------------------------------
def compute_verdicts(res):
    A, B = res['A'], res['B']
    return {
        '1_self_recognition_above_chance': {
            'pass': A['eval_accuracy'] > 0.75,
            'embodied_accuracy': A['eval_accuracy'], 'threshold': 0.75,
            'description': 'Self-recognition accuracy > 0.75 (above chance for 4-way)'},
        '2_embodied_better_than_disembodied': {
            'pass': A['eval_accuracy'] > B['eval_accuracy'],
            'embodied_accuracy': A['eval_accuracy'],
            'disembodied_accuracy': B['eval_accuracy'],
            'description': 'Embodied (A) accuracy > Disembodied (B) accuracy'},
        '3_body_encoded_in_representations': {
            'pass': A['cka_self_other'] < 0.9,
            'cka_self_other': A['cka_self_other'], 'threshold': 0.9,
            'description': 'CKA(self, other) < 0.9 -- body info in representations'},
        '4_probe_extracts_body_identity': {
            'pass': A['probe_accuracy'] > 0.6,
            'probe_accuracy': A['probe_accuracy'], 'threshold': 0.6,
            'description': 'Linear probe accuracy > 0.6 -- body identity extractable'},
    }

# -- Print results ------------------------------------------------------------
def print_results_table(res, verdicts):
    print(f"\n{'='*80}\nRESULTS: Self-Other Distinction\n{'='*80}")
    print(f"{'Metric':<30} {'A:Embodied':>14} {'B:Disembod':>14} {'C:HiddenOnly':>14}")
    print('-' * 80)
    for label, key, fmt in [('Phase1 Perplexity','phase1_perplexity','.2f'),
                             ('Phase2 Train Accuracy','phase2_train_accuracy','.3f'),
                             ('Eval Accuracy','eval_accuracy','.3f'),
                             ('CKA(self, other)','cka_self_other','.4f'),
                             ('Probe Accuracy','probe_accuracy','.3f'),
                             ('Wall Time (s)','wall_time_s','.1f')]:
        vals = ''.join(f" {res[c].get(key,0.0):>14{fmt}}" for c in ['A','B','C'])
        print(f"{label:<30}{vals}")
    print(f"\nPer-type eval accuracy (Embodied A):")
    for n, a in res['A']['eval_per_type_accuracy'].items():
        print(f"  {n:<12}: {a:.3f}")
    print(f"\n{'='*80}\nVERDICT\n{'='*80}")
    ap = True
    for v in verdicts.values():
        s = 'PASS' if v['pass'] else 'FAIL'; ap = ap and v['pass']
        print(f"  [{s}] {v['description']}")
    print(f"\n  OVERALL: {'PASS -- Embodied model distinguishes self from other!' if ap else 'FAIL -- Further investigation needed'}")

# -- Main --------------------------------------------------------------------
def main():
    print("z1706: Self-Other Distinction -- Does Embodiment Enable Body Identity?")
    print("=" * 70)
    device = get_best_device()
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    try:
        telem = SysfsHwmonTelemetry()
        s = telem.read_sample()
        print(f"Telemetry OK: {s.power_w:.1f}W, {s.temp_edge_c:.0f}C")
    except Exception as e:
        print(f"ERROR: Telemetry init failed: {e}"); return
    actuator = GPUActuator(card_id=0)
    st = actuator.get_current_state()
    print(f"GPU: perf={st.performance_level}, {st.sclk_mhz}MHz, {st.current_power_w:.1f}W, {st.temperature_c:.0f}C")
    if not DATA_PATH.exists():
        print(f"ERROR: Dataset not found at {DATA_PATH}"); return
    dataset = CharDataset(DATA_PATH, SEQ_LEN)
    print(f"Dataset: {len(dataset.data):,} bytes, {dataset.n_batches} batches (bs={BATCH_SIZE}, sl={SEQ_LEN})")

    conditions = [('A','Embodied (FiLM ON, head sees hidden+telem)'),
                  ('B','Disembodied (FiLM OFF, head sees hidden+telem)'),
                  ('C','Hidden-Only (FiLM ON, head sees hidden only)')]
    results = {}
    try:
        for ck, cl in conditions:
            results[ck] = run_condition(ck, cl, device, dataset, telem, actuator)
            if ck != conditions[-1][0]:
                print(f"\n  Cooldown {COOLDOWN_S}s ..."); time.sleep(COOLDOWN_S)
    except Exception as e:
        print(f"\nERROR: {e}"); import traceback; traceback.print_exc()
    finally:
        actuator.set_performance_level(PerformanceLevel.BALANCED)

    if len(results) == 3:
        verdicts = compute_verdicts(results)
        print_results_table(results, verdicts)
        output = {
            'experiment': 'z1706_self_other_distinction',
            'hypothesis': 'Embodied model distinguishes own body-state traces from others -- self-awareness prerequisite.',
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'device': str(device),
            'gpu_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu',
            'config': {'phase1_epochs': PHASE1_EPOCHS, 'phase2_epochs': PHASE2_EPOCHS,
                       'batch_size': BATCH_SIZE, 'seq_len': SEQ_LEN, 'lr': LR,
                       'recognition_lr': RECOGNITION_LR, 'hidden_dim': 256,
                       'num_layers': 6, 'telemetry_dim': 12, 'max_batches': MAX_BATCHES,
                       'other_body_types': ['gaussian_noise', 'time_shifted', 'scaled']},
            'conditions': dict(results),
            'verdicts': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                             for kk, vv in v.items()} for k, v in verdicts.items()},
        }
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_PATH, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {RESULTS_PATH}")
    else:
        print(f"\nIncomplete: {len(results)}/3 conditions.")

if __name__ == '__main__':
    main()
