#!/usr/bin/env python3
"""
z1709: RECURSIVE SELF-MODELING - Can the Model Predict Its Own Future Predictions?

Genuine self-awareness requires predicting one's OWN future states. We test
whether embodied hardware signals (FiLM + GPU telemetry) improve self-prediction.

Architecture on top of MetabolicTransformer:
  SelfModel: MLP (256->128->64->128->256) predicts next hidden state
  MetaModel: MLP (256->64->256) predicts SelfModel output (recursive)

Training: Phase 1 (3ep) base LM | Phase 2 (5ep) SelfModel | Phase 3 (5ep) MetaModel
Conditions: A=Embodied(268d) B=Disembodied(256d) C=SelfOnly(256d+FiLM)
"""

import sys, os, json, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from src.metabolic.film_transformer import create_metabolic_transformer, MetabolicConfig
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry
from src.actuation.gpu_actuator import GPUActuator, PerformanceLevel
import functools
print = functools.partial(print, flush=True)


# -- SelfModel & MetaModel ---------------------------------------------------

class SelfModel(nn.Module):
    """Predicts model's NEXT hidden state from current hidden."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 128), nn.GELU(),
            nn.Linear(128, 256),
        )
    def forward(self, x): return self.net(x)


class MetaModel(nn.Module):
    """Predicts what SelfModel will predict (meta-prediction)."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.GELU(), nn.Linear(64, 256),
        )
    def forward(self, x): return self.net(x)


# -- Helpers ------------------------------------------------------------------

def build_telemetry_12d(tel, prev=None):
    s = tel.read_sample()
    p, t, f, b = s.power_w, s.temp_edge_c, s.freq_sclk_mhz, s.gpu_busy_pct
    if prev:
        dp, dt, df, du = (p-prev[0])/50, (t-prev[1])/100, (f-prev[2])/3000, (b-prev[3])/100
    else:
        dp = dt = df = du = 0.0
    vec = torch.tensor([
        p/50, t/100, f/3000, b/100, 0.5, 1.0 if t > 90 else 0.0,
        dp, dt, df, du, (t-60)/40, max(0, (3000-f))/3000,
    ], dtype=torch.float32)
    return vec, (p, t, f, b)


def load_dataset(path, seq_len=256):
    with open(path, 'r') as f:
        text = f.read()
    data = torch.tensor([ord(c) % 256 for c in text], dtype=torch.long)
    n = (len(data) // seq_len) * seq_len
    return data[:n].view(-1, seq_len)


def batch_correlation(a, b):
    af, bf = a.detach().float().flatten(), b.detach().float().flatten()
    if af.std() < 1e-8 or bf.std() < 1e-8:
        return 0.0
    ac, bc = af - af.mean(), bf - bf.mean()
    return (ac * bc).sum().item() / (ac.norm().item() * bc.norm().item() + 1e-8)


# -- Run one condition --------------------------------------------------------

def run_condition(name, film_on, telem_heads, dataset, device, tel, bs=4):
    print(f"\n{'='*70}")
    print(f"  CONDITION {name}: film={film_on}, telem_to_heads={telem_heads}")
    print(f"{'='*70}")

    model = create_metabolic_transformer(
        hidden_dim=256, num_layers=6, num_heads=4, telemetry_dim=12,
    ).to(device)
    model.enable_conditioning(film_on)

    hdim = 256 + 12 if telem_heads else 256
    sm = SelfModel(hdim).to(device)
    mm = MetaModel(hdim).to(device)

    print(f"  Params: base={sum(p.numel() for p in model.parameters()):,}"
          f"  self={sum(p.numel() for p in sm.parameters()):,}"
          f"  meta={sum(p.numel() for p in mm.parameters()):,}")

    opt_b = torch.optim.AdamW(model.parameters(), lr=3e-4)
    opt_s = torch.optim.AdamW(sm.parameters(), lr=1e-3)
    opt_m = torch.optim.AdamW(mm.parameters(), lr=1e-3)

    nb = min(len(dataset), 400) // bs
    prev_raw = None
    log = []

    def fwd_batch(bi):
        nonlocal prev_raw
        batch = dataset[bi*bs:(bi+1)*bs].to(device)
        tv, prev_raw = build_telemetry_12d(tel, prev_raw)
        tb = tv.unsqueeze(0).expand(bs, -1).to(device)
        out = model(batch, tb, return_hidden=True)
        return batch, tb, out

    def lm_loss(out, batch):
        logits = out['logits'][:, :-1].contiguous()
        return F.cross_entropy(logits.view(-1, 256), batch[:, 1:].contiguous().view(-1))

    def sm_input(prev_h, tb):
        return torch.cat([prev_h.detach(), tb], dim=-1) if telem_heads else prev_h.detach()

    # Phase 1: Base LM (3 epochs)
    print(f"\n  [Phase 1] Base LM (3 ep, {nb} batches/ep)")
    for ep in range(3):
        model.train(); el = 0.0
        for bi in range(nb):
            batch, tb, out = fwd_batch(bi)
            loss = lm_loss(out, batch)
            opt_b.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_b.step(); el += loss.item()
        avg = el / nb; ppl = math.exp(min(avg, 20))
        print(f"    Ep {ep+1}/3  loss={avg:.4f}  ppl={ppl:.1f}")
        log.append({'phase': 1, 'epoch': ep+1, 'lm_loss': avg, 'ppl': ppl})

    # Phase 2: SelfModel (5 epochs)
    print(f"\n  [Phase 2] SelfModel (5 ep)")
    for ep in range(5):
        model.train(); sm.train()
        el, es, prev_h = 0.0, 0.0, None
        for bi in range(nb):
            batch, tb, out = fwd_batch(bi)
            hm = out['hidden'].mean(dim=1)
            ll = lm_loss(out, batch)
            if prev_h is not None:
                pred = sm(sm_input(prev_h, tb))
                sl = F.mse_loss(pred, hm.detach()); es += sl.item()
                total = ll + 0.5 * sl
            else:
                total = ll
            opt_b.zero_grad(); opt_s.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(sm.parameters(), 1.0)
            opt_b.step(); opt_s.step()
            el += ll.item(); prev_h = hm.detach()
        al, aself = el/nb, es/max(nb-1, 1)
        ppl = math.exp(min(al, 20))
        print(f"    Ep {ep+1}/5  lm={al:.4f}  self_mse={aself:.6f}  ppl={ppl:.1f}")
        log.append({'phase': 2, 'epoch': ep+1, 'lm_loss': al, 'self_mse': aself, 'ppl': ppl})

    # Phase 3: MetaModel (5 epochs)
    print(f"\n  [Phase 3] MetaModel (5 ep)")
    for ep in range(5):
        model.train(); sm.train(); mm.train()
        el, es, em, prev_h = 0.0, 0.0, 0.0, None
        for bi in range(nb):
            batch, tb, out = fwd_batch(bi)
            hm = out['hidden'].mean(dim=1)
            ll = lm_loss(out, batch)
            if prev_h is not None:
                inp = sm_input(prev_h, tb)
                with torch.no_grad():
                    sp = sm(inp)
                es += F.mse_loss(sp, hm.detach()).item()
                mp = mm(inp)
                ml = F.mse_loss(mp, sp.detach()); em += ml.item()
                total = ll + 0.3 * ml
            else:
                total = ll
            opt_b.zero_grad(); opt_m.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(mm.parameters(), 1.0)
            opt_b.step(); opt_m.step()
            el += ll.item(); prev_h = hm.detach()
        al = el/nb; aself = es/max(nb-1,1); ameta = em/max(nb-1,1)
        ppl = math.exp(min(al, 20))
        print(f"    Ep {ep+1}/5  lm={al:.4f}  self={aself:.6f}  meta={ameta:.6f}  ppl={ppl:.1f}")
        log.append({'phase':3, 'epoch':ep+1, 'lm_loss':al, 'self_mse':aself, 'meta_mse':ameta, 'ppl':ppl})

    # Evaluation
    print(f"\n  [Eval] Final metrics...")
    model.eval(); sm.eval(); mm.eval()
    s_list, m_list, cms_list, cma_list, var_list = [], [], [], [], []
    prev_h = None
    with torch.no_grad():
        for bi in range(min(60, nb)):
            batch, tb, out = fwd_batch(bi)
            hm = out['hidden'].mean(dim=1)
            if prev_h is not None:
                inp = sm_input(prev_h, tb)
                sp, mp = sm(inp), mm(inp)
                s_list.append(F.mse_loss(sp, hm).item())
                m_list.append(F.mse_loss(mp, sp).item())
                cms_list.append(batch_correlation(mp, sp))
                cma_list.append(batch_correlation(mp, hm))
                var_list.append(((sp - hm)**2).mean(dim=-1).var().item())
            prev_h = hm

    # Task perplexity on tail
    ev = dataset[max(0, len(dataset)-40):][:20]
    tl = 0.0
    nev = len(ev) // bs
    with torch.no_grad():
        for bi in range(nev):
            batch = ev[bi*bs:(bi+1)*bs].to(device)
            tv, prev_raw = build_telemetry_12d(tel, prev_raw)
            tb = tv.unsqueeze(0).expand(batch.size(0), -1).to(device)
            out = model(batch, tb)
            tl += F.cross_entropy(
                out['logits'][:,:-1].contiguous().view(-1,256),
                batch[:,1:].contiguous().view(-1)).item()
    fppl = math.exp(min(tl / max(nev, 1), 20))

    # Action check
    probs = F.softmax(out['action_logits'], dim=-1).mean(dim=0)
    aidx = torch.argmax(probs).item()
    anames = ['LOW', 'BALANCED', 'HIGH', 'MAX']

    r = {
        'condition': name, 'film_enabled': film_on, 'telem_to_heads': telem_heads,
        'self_pred_mse': float(np.mean(s_list)) if s_list else 999.0,
        'meta_pred_mse': float(np.mean(m_list)) if m_list else 999.0,
        'corr_meta_self': float(np.mean(cms_list)) if cms_list else 0.0,
        'corr_meta_actual': float(np.mean(cma_list)) if cma_list else 0.0,
        'recursive_coherence': float(np.mean(cms_list)) if cms_list else 0.0,
        'self_model_stability': float(np.mean(var_list)) if var_list else 999.0,
        'task_ppl': fppl, 'action_chosen': anames[aidx], 'metrics_log': log,
    }
    print(f"\n  Results {name}:")
    print(f"    Self-pred MSE:    {r['self_pred_mse']:.6f}")
    print(f"    Meta-pred MSE:    {r['meta_pred_mse']:.6f}")
    print(f"    Corr(meta,self):  {r['corr_meta_self']:.4f}")
    print(f"    Corr(meta,act):   {r['corr_meta_actual']:.4f}")
    print(f"    Stability:        {r['self_model_stability']:.8f}")
    print(f"    Task PPL:         {r['task_ppl']:.2f}")

    del model, sm, mm, opt_b, opt_s, opt_m
    torch.cuda.empty_cache()
    return r


# -- Main ---------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  z1709: RECURSIVE SELF-MODELING")
    print("  Can the model predict its own future predictions?")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {props.name}  VRAM: {props.total_memory / 1e9:.1f} GB")

    tel = SysfsHwmonTelemetry()
    actuator = GPUActuator()
    actuator.set_performance_level(PerformanceLevel.BALANCED)

    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'data', 'tinyshakespeare.txt')
    print(f"\nDataset: {data_path}")
    dataset = load_dataset(data_path, seq_len=256)
    print(f"  {len(dataset)} sequences, seq_len=256")

    torch.manual_seed(1709)
    dataset = dataset[torch.randperm(len(dataset))]

    try:
        res = {}
        res['A'] = run_condition('A_Embodied', True, True, dataset, device, tel)
        res['B'] = run_condition('B_Disembodied', False, False, dataset, device, tel)
        res['C'] = run_condition('C_SelfOnly', True, False, dataset, device, tel)

        # Verdicts
        print(f"\n{'='*70}\n  VERDICTS\n{'='*70}")
        a, b, c = res['A'], res['B'], res['C']

        verdicts = {
            'v1_embodied_self_pred_better': {
                'pass': a['self_pred_mse'] < b['self_pred_mse'],
                'A': a['self_pred_mse'], 'B': b['self_pred_mse'],
                'desc': 'Embodied predicts self better than disembodied',
            },
            'v2_embodied_meta_pred_better': {
                'pass': a['meta_pred_mse'] < b['meta_pred_mse'],
                'A': a['meta_pred_mse'], 'B': b['meta_pred_mse'],
                'desc': 'Embodied meta-predicts better than disembodied',
            },
            'v3_recursive_coherence_gt_0.5': {
                'pass': a['recursive_coherence'] > 0.5,
                'A_coh': a['recursive_coherence'],
                'desc': 'Meta-model coherent with self-model (corr>0.5)',
            },
            'v4_task_ppl_preserved': {
                'pass': abs(a['task_ppl'] - b['task_ppl']) / max(b['task_ppl'], 1) < 0.10,
                'A_ppl': a['task_ppl'], 'B_ppl': b['task_ppl'],
                'diff_pct': abs(a['task_ppl']-b['task_ppl'])/max(b['task_ppl'],1)*100,
                'desc': 'Task quality within 10% of disembodied',
            },
        }
        np_ = sum(1 for v in verdicts.values() if v['pass'])
        for vv in verdicts.values():
            print(f"  [{'PASS' if vv['pass'] else 'FAIL'}] {vv['desc']}")
        print(f"\n  Overall: {np_}/4 passed")

        # Comparison table
        print(f"\n{'='*70}\n  COMPARISON TABLE\n{'='*70}")
        print(f"  {'Metric':<22} {'A(Embod)':>11} {'B(Disemb)':>11} {'C(Self)':>11}")
        print(f"  {'-'*55}")
        for k in ['self_pred_mse','meta_pred_mse','corr_meta_self','corr_meta_actual',
                   'self_model_stability','task_ppl']:
            fmt = '.6f' if 'mse' in k or 'stab' in k else ('.4f' if 'corr' in k else '.2f')
            line = f"  {k:<22} "
            line += f"{a[k]:{'>11'+fmt}} {b[k]:{'>11'+fmt}} {c[k]:{'>11'+fmt}}"
            print(line)

        # Save
        output = {
            'experiment': 'z1709_recursive_self_modeling',
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'device': str(device),
            'gpu_name': props.name if device.type == 'cuda' else 'cpu',
            'gpu_vram_gb': props.total_memory / 1e9 if device.type == 'cuda' else 0,
            'config': {
                'vocab_size': 256, 'hidden_dim': 256, 'num_layers': 6,
                'num_heads': 4, 'ff_dim': 1024, 'telemetry_dim': 12,
                'batch_size': 4, 'seq_len': 256,
                'phase1_epochs': 3, 'phase2_epochs': 5, 'phase3_epochs': 5,
            },
            'conditions': {'A_Embodied': a, 'B_Disembodied': b, 'C_SelfOnly': c},
            'verdicts': verdicts, 'n_pass': np_,
            'summary': (f"Recursive self-modeling: {np_}/4 passed. "
                        f"Embodied self-pred={a['self_pred_mse']:.6f} vs "
                        f"Disembodied={b['self_pred_mse']:.6f}. "
                        f"Coherence={a['recursive_coherence']:.4f}."),
        }
        out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'results', 'z1709_recursive_self_modeling.json')
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2, default=float)
        print(f"\nResults saved to: {out_path}")

    finally:
        actuator.set_performance_level(PerformanceLevel.BALANCED)
        torch.cuda.empty_cache()
        print("\nGPU cleanup complete.")


if __name__ == '__main__':
    main()
