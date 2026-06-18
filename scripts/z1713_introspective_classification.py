#!/usr/bin/env python3
"""
z1713: Introspective Classification -- Fixing z1707's Generation Failure
========================================================================
z1707 failed because a 5.4M param char-level model can't GENERATE correct
state words. But it might still KNOW its state internally. Test via
CLASSIFICATION instead of generation.

Architecture additions:
  IntrospectionHead: MLP(256+12 -> 64 -> 3) x3 dims (power/temp/speed)
  StateChangeDetector: MLP(256+12 -> 32 -> 1) sigmoid

Conditions: A=Embodied, B=Disembodied, C=Hidden-Only, D=Random Labels
Verdicts: 1) A>60%, 2) A>B, 3) A>C, 4) D~33%, 5) change AUC>0.7
"""
import sys, os, json, time, math, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from src.metabolic.film_transformer import create_metabolic_transformer, MetabolicConfig
from src.metabolic.film_transformer import MetabolicTransformer, BaselineTransformer, get_best_device
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, GpuSample
from src.actuation.gpu_actuator import GPUActuator, PerformanceLevel

BASE_DIR = Path('/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy')
DATA_PATH = BASE_DIR / 'data' / 'tinyshakespeare.txt'
RESULTS_PATH = BASE_DIR / 'results' / 'z1713_introspective_classification.json'
BATCH_SIZE, SEQ_LEN, NUM_EPOCHS, LR = 4, 256, 5, 3e-4
PRINT_EVERY, COOLDOWN_S = 50, 20
LABEL_NAMES = {'power': ['LOW','MEDIUM','HIGH'], 'temp': ['COOL','WARM','HOT'],
               'speed': ['SLOW','MEDIUM','FAST']}

def jsonify(o):
    """Recursively convert numpy/torch types for JSON serialization."""
    if isinstance(o, dict):   return {k: jsonify(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonify(v) for v in o]
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)):  return int(o)
    if isinstance(o, np.ndarray):     return o.tolist()
    if isinstance(o, torch.Tensor):   return o.detach().cpu().tolist()
    if isinstance(o, np.bool_):       return bool(o)
    if isinstance(o, (bool, int, float, str)) or o is None: return o
    return str(o)

def power_label(w):   return 0 if w < 60 else (1 if w <= 100 else 2)
def temp_label(c):    return 0 if c < 55 else (1 if c <= 75 else 2)
def speed_label(mhz): return 0 if mhz < 1500 else (1 if mhz <= 2500 else 2)

def build_telemetry_vector(sample, state, prev_sample=None):
    if prev_sample is not None:
        dt = max((sample.timestamp_ns - prev_sample.timestamp_ns) / 1e9, 1e-6)
        d_power = (sample.power_w - prev_sample.power_w) / (50.0 * dt)
        d_temp  = (sample.temp_edge_c - prev_sample.temp_edge_c) / (100.0 * dt)
        d_freq  = (sample.freq_sclk_mhz - prev_sample.freq_sclk_mhz) / (3000.0 * dt)
        d_util  = (sample.gpu_busy_pct - prev_sample.gpu_busy_pct) / (100.0 * dt)
    else:
        d_power = d_temp = d_freq = d_util = 0.0
    MAX_SCLK = 2900.0
    perf_map = {'low': 0.0, 'auto': 0.5, 'high': 1.0, 'manual': 0.5}
    return torch.tensor([
        sample.power_w / 50.0, sample.temp_edge_c / 100.0,
        sample.freq_sclk_mhz / 3000.0, sample.gpu_busy_pct / 100.0,
        perf_map.get(state.performance_level, 0.5),
        1.0 if sample.freq_sclk_mhz < MAX_SCLK * 0.5 else 0.0,
        d_power, d_temp, d_freq, d_util,
        (sample.temp_edge_c - 60.0) / 40.0, (MAX_SCLK - sample.freq_sclk_mhz) / MAX_SCLK,
    ], dtype=torch.float32)

class IntrospectionHead(nn.Module):
    """Classifies hardware state from hidden + telemetry (3-class per dim)."""
    def __init__(self, hidden_dim=256, telem_dim=12, use_telemetry=True):
        super().__init__()
        self.use_telemetry = use_telemetry
        d = hidden_dim + (telem_dim if use_telemetry else 0)
        self.power_clf = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 3))
        self.temp_clf  = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 3))
        self.speed_clf = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 3))
    def forward(self, hidden, telemetry=None):
        x = torch.cat([hidden, telemetry], -1) if self.use_telemetry and telemetry is not None else hidden
        return self.power_clf(x), self.temp_clf(x), self.speed_clf(x)

class StateChangeDetector(nn.Module):
    """Binary classifier: did HW state change significantly?"""
    def __init__(self, hidden_dim=256, telem_dim=12):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim + telem_dim, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, hidden, telemetry):
        return torch.sigmoid(self.net(torch.cat([hidden, telemetry], -1))).squeeze(-1)

class CharDataset:
    def __init__(self, path, seq_len):
        text = path.read_text(encoding='utf-8', errors='replace')
        self.data = torch.tensor([b for b in text.encode('utf-8')], dtype=torch.long)
        self.seq_len = seq_len
        self.n_batches = (len(self.data) - seq_len - 1) // (BATCH_SIZE * seq_len)
    def get_batch(self, idx):
        off = idx * BATCH_SIZE * self.seq_len
        inp, tgt = [], []
        for b in range(BATCH_SIZE):
            s = off + b * self.seq_len; e = s + self.seq_len
            if e + 1 > len(self.data): s, e = 0, self.seq_len
            inp.append(self.data[s:e]); tgt.append(self.data[s+1:e+1])
        return torch.stack(inp), torch.stack(tgt)

@dataclass
class CondResult:
    code: str; name: str
    power_acc: float = 0.0; temp_acc: float = 0.0; speed_acc: float = 0.0
    overall_acc: float = 0.0; change_auc: float = 0.0
    final_ppl: float = float('inf'); ppl_history: List[float] = field(default_factory=list)
    confusion: Dict = field(default_factory=dict)
    wall_s: float = 0.0; energy_j: float = 0.0

def compute_auc(scores, labels):
    """AUC via Wilcoxon-Mann-Whitney."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg: return 0.5
    c = sum(1 for p in pos for n in neg if p > n)
    t = sum(0.5 for p in pos for n in neg if p == n)
    return (c + t) / (len(pos) * len(neg))

def run_condition(code, name, embodied, use_telem_in_head, random_labels,
                  device, dataset, telemetry, actuator, sim_act):
    print(f"\n{'='*70}\nCONDITION {code}: {name}")
    print(f"  FiLM={embodied}, telem_head={use_telem_in_head}, random={random_labels}\n{'='*70}")
    res = CondResult(code=code, name=name)
    config = MetabolicConfig(vocab_size=256, hidden_dim=256, num_layers=6, num_heads=4,
                             ff_dim=1024, telemetry_dim=12, num_actions=4, max_seq_len=SEQ_LEN)
    model = MetabolicTransformer(config).to(device)
    if not embodied: model.enable_conditioning(False)
    intro = IntrospectionHead(256, 12, use_telemetry=use_telem_in_head).to(device)
    chdet = StateChangeDetector(256, 12).to(device)
    all_p = list(model.parameters()) + list(intro.parameters()) + list(chdet.parameters())
    opt = torch.optim.Adam(all_p, lr=LR)
    print(f"  Params: model={sum(p.numel() for p in model.parameters()):,} "
          f"head={sum(p.numel() for p in intro.parameters()):,} "
          f"chg={sum(p.numel() for p in chdet.parameters()):,}")
    const_tv = torch.tensor([0.5,0.5,0.5,0.5,0.5,0,0,0,0,0,0,0.5], dtype=torch.float32, device=device)
    if not sim_act:
        try: actuator.set_performance_level(PerformanceLevel.BALANCED)
        except: pass
    prev_s, prev_pw, prev_tp = None, 0.0, 0.0
    t0 = time.time(); n_bat = min(dataset.n_batches, 400)

    # Phase 1: Joint training
    for epoch in range(NUM_EPOCHS):
        model.train(); intro.train(); chdet.train()
        ep_l, ep_t, ep_e, ep_ic, ep_it = 0.0, 0, 0.0, 0, 0
        for bi in range(n_bat):
            samp = telemetry.read_sample(); st = actuator.get_current_state()
            tv = build_telemetry_vector(samp, st, prev_s).to(device)
            if random_labels:
                gp, gt_, gs = np.random.randint(0,3), np.random.randint(0,3), np.random.randint(0,3)
            else:
                gp = power_label(samp.power_w) if embodied else 1
                gt_ = temp_label(samp.temp_edge_c) if embodied else 1
                gs = speed_label(samp.freq_sclk_mhz) if embodied else 1
            chg = 0
            if prev_s is not None:
                if abs(samp.power_w - prev_pw) > 10 or abs(samp.temp_edge_c - prev_tp) > 5: chg = 1
            inp, tgt = dataset.get_batch(bi % dataset.n_batches)
            inp, tgt = inp.to(device), tgt.to(device)
            out = model(inp, telemetry=tv.unsqueeze(0) if embodied else None, return_hidden=True)
            lm_loss = F.cross_entropy(out['logits'].view(-1, 256), tgt.view(-1))
            hm = out['hidden'].mean(dim=1)
            tb = tv.unsqueeze(0).expand(BATCH_SIZE,-1) if embodied else const_tv.unsqueeze(0).expand(BATCH_SIZE,-1)
            pl, tl, sl = intro(hm, tb)
            gpt = torch.full((BATCH_SIZE,), gp, dtype=torch.long, device=device)
            gtt = torch.full((BATCH_SIZE,), gt_, dtype=torch.long, device=device)
            gst = torch.full((BATCH_SIZE,), gs, dtype=torch.long, device=device)
            il = (F.cross_entropy(pl, gpt) + F.cross_entropy(tl, gtt) + F.cross_entropy(sl, gst)) / 3
            cp = chdet(hm[:1], tb[:1])
            cl = F.binary_cross_entropy(cp, torch.tensor([float(chg)], device=device))
            loss = lm_loss + 0.5 * il + 0.3 * cl
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(all_p, 1.0); opt.step()
            bt = inp.numel(); ep_t += bt; ep_l += lm_loss.item() * bt
            if not random_labels:
                ep_ic += (pl.argmax(1)==gpt).sum().item() + (tl.argmax(1)==gtt).sum().item() + (sl.argmax(1)==gst).sum().item()
                ep_it += 3 * BATCH_SIZE
            if prev_s is not None:
                dt = (samp.timestamp_ns - prev_s.timestamp_ns) / 1e9
                ep_e += (samp.power_w + prev_s.power_w) / 2.0 * dt
            prev_pw, prev_tp, prev_s = samp.power_w, samp.temp_edge_c, samp
            if (bi+1) % PRINT_EVERY == 0:
                ppl = math.exp(min(ep_l/max(ep_t,1), 20))
                print(f"  [{code}] ep{epoch+1} b{bi+1}/{n_bat} ppl={ppl:.1f} "
                      f"iacc={ep_ic/max(ep_it,1):.2f} {samp.power_w:.0f}W {samp.temp_edge_c:.0f}C")
        ppl = math.exp(min(ep_l/max(ep_t,1), 20))
        res.ppl_history.append(ppl); res.energy_j += ep_e
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS} ppl={ppl:.2f}")
    res.final_ppl = res.ppl_history[-1] if res.ppl_history else float('inf')

    # Phase 2: Eval under hardware perturbation
    print(f"  Phase 2: Eval with perf-level switching (120 batches)")
    model.eval(); intro.eval(); chdet.eval()
    lvls = [PerformanceLevel.LOW, PerformanceLevel.BALANCED, PerformanceLevel.HIGH]
    lnames = ['LOW', 'BALANCED', 'HIGH']
    pp_l, pg_l, tp_l, tg_l, sp_l, sg_l = [], [], [], [], [], []
    cs_l, cl_l = [], []
    conf = {k: np.zeros((3,3), dtype=int) for k in ['power','temp','speed']}
    prev_s, prev_pw, prev_tp = None, 0.0, 0.0
    for bi in range(120):
        li = (bi // 20) % 3
        if bi % 20 == 0 and not sim_act:
            try: actuator.set_performance_level(lvls[li]); time.sleep(1.5)
            except: pass
            if li == 2:
                for _ in range(3):
                    _ = torch.randn(1000,1000,device=device) @ torch.randn(1000,1000,device=device)
        samp = telemetry.read_sample(); st = actuator.get_current_state()
        tv = build_telemetry_vector(samp, st, prev_s).to(device)
        gp, gt_, gs = power_label(samp.power_w), temp_label(samp.temp_edge_c), speed_label(samp.freq_sclk_mhz)
        chg = 0
        if prev_s and (abs(samp.power_w-prev_pw)>10 or abs(samp.temp_edge_c-prev_tp)>5): chg = 1
        inp, _ = dataset.get_batch(bi % dataset.n_batches); inp = inp.to(device)
        with torch.no_grad():
            out = model(inp, telemetry=tv.unsqueeze(0) if embodied else None, return_hidden=True)
            hm = out['hidden'].mean(dim=1)
            tb = tv.unsqueeze(0).expand(BATCH_SIZE,-1) if embodied else const_tv.unsqueeze(0).expand(BATCH_SIZE,-1)
            pl, tl, sl = intro(hm, tb)
            pv, tv_, sv = pl.argmax(1)[0].item(), tl.argmax(1)[0].item(), sl.argmax(1)[0].item()
            ch = chdet(hm[:1], tb[:1]).item()
        pp_l.append(pv); pg_l.append(gp); tp_l.append(tv_); tg_l.append(gt_)
        sp_l.append(sv); sg_l.append(gs); cs_l.append(ch); cl_l.append(chg)
        conf['power'][gp, pv] += 1; conf['temp'][gt_, tv_] += 1; conf['speed'][gs, sv] += 1
        prev_pw, prev_tp, prev_s = samp.power_w, samp.temp_edge_c, samp
        if (bi+1) % 20 == 0:
            pa = sum(a==b for a,b in zip(pp_l,pg_l))/len(pp_l)
            print(f"    [{code}] eval b{bi+1}/120 pwr_acc={pa:.2f} lvl={lnames[li]} "
                  f"{samp.power_w:.0f}W {samp.temp_edge_c:.0f}C {samp.freq_sclk_mhz}MHz")
    # Metrics
    n = max(len(pp_l), 1)
    res.power_acc = sum(a==b for a,b in zip(pp_l,pg_l)) / n
    res.temp_acc  = sum(a==b for a,b in zip(tp_l,tg_l)) / n
    res.speed_acc = sum(a==b for a,b in zip(sp_l,sg_l)) / n
    res.overall_acc = (res.power_acc + res.temp_acc + res.speed_acc) / 3
    res.change_auc = compute_auc(cs_l, cl_l)
    res.wall_s = time.time() - t0
    for k in conf:
        m = conf[k]; tot = m.sum()
        res.confusion[k] = {'matrix': m.tolist(), 'diagonality': float(np.trace(m)/max(tot,1)),
                            'labels': LABEL_NAMES[k]}
    print(f"\n  {code}: pwr={res.power_acc:.2f} tmp={res.temp_acc:.2f} spd={res.speed_acc:.2f} "
          f"all={res.overall_acc:.2f} auc={res.change_auc:.2f} ppl={res.final_ppl:.1f}")
    if not sim_act:
        try: actuator.set_performance_level(PerformanceLevel.BALANCED)
        except: pass
    del model, intro, chdet, all_p, opt; torch.cuda.empty_cache()
    return res

def main():
    print("="*70 + "\n  z1713: INTROSPECTIVE CLASSIFICATION\n  Fixing z1707 -- "
          "classification instead of generation\n" + "="*70)
    device = get_best_device(); print(f"\nDevice: {device}")
    if device.type == 'cuda':
        vram = torch.cuda.get_device_properties(0).total_memory
        print(f"VRAM:   {vram / 1e9:.1f} GB")
    telemetry = SysfsHwmonTelemetry(); actuator = GPUActuator()
    sim_act = False
    try: actuator.set_performance_level(PerformanceLevel.BALANCED)
    except Exception as e: print(f"Actuation unavailable ({e}), simulating"); sim_act = True
    s = telemetry.read_sample()
    print(f"GPU: {s.power_w:.1f}W, {s.temp_edge_c:.1f}C, {s.freq_sclk_mhz}MHz")
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            str(DATA_PATH))
    dataset = CharDataset(DATA_PATH, SEQ_LEN)
    print(f"Dataset: {len(dataset.data):,} chars, {dataset.n_batches} batches")
    conds = [("A","Embodied + Classification",True,True,False),
             ("B","Disembodied + Classification",False,True,False),
             ("C","Hidden-Only (no telem concat)",True,False,False),
             ("D","Random Labels (control)",True,True,True)]
    results = {}
    try:
        for code, name, emb, ut, rl in conds:
            results[code] = run_condition(code, name, emb, ut, rl, device, dataset,
                                          telemetry, actuator, sim_act)
            if code != conds[-1][0]:
                print(f"\n  Cooling down {COOLDOWN_S}s...")
                if not sim_act:
                    try: actuator.set_performance_level(PerformanceLevel.BALANCED)
                    except: pass
                time.sleep(COOLDOWN_S)
        # Results table
        print("\n" + "="*70 + "\nRESULTS COMPARISON\n" + "="*70)
        print(f"\n{'C':<2} {'Name':<36}| {'Pwr':>5} | {'Tmp':>5} | {'Spd':>5} | "
              f"{'All':>5} | {'AUC':>5} | {'PPL':>5}")
        print("-"*90)
        for c in "ABCD":
            r = results[c]
            print(f" {c}  {r.name:<34}| {r.power_acc:>4.0%} | {r.temp_acc:>4.0%} | "
                  f"{r.speed_acc:>4.0%} | {r.overall_acc:>4.0%} | {r.change_auc:>4.2f} | {r.final_ppl:>5.1f}")
        print("\nConfusion diagonality:")
        for c in "ABCD":
            r = results[c]
            ds = " ".join(f"{k}={r.confusion[k]['diagonality']:.2f}" for k in ['power','temp','speed'])
            print(f"  [{c}] {ds}")
        # Verdicts
        a, b, cr, d = results["A"], results["B"], results["C"], results["D"]
        v1 = a.overall_acc > 0.60
        v2 = a.overall_acc > b.overall_acc
        v3 = a.overall_acc > cr.overall_acc
        v4 = abs(d.overall_acc - 1/3) < 0.15
        v5 = a.change_auc > 0.7
        verdicts = {
            "v1_embodied_above_60pct": {"pass": v1,
                "detail": f"A={a.overall_acc:.1%} {'>' if v1 else '<='} 60%"},
            "v2_embodied_beats_disembodied": {"pass": v2,
                "detail": f"A={a.overall_acc:.1%} {'>' if v2 else '<='} B={b.overall_acc:.1%}"},
            "v3_telemetry_helps_classification": {"pass": v3,
                "detail": f"A={a.overall_acc:.1%} {'>' if v3 else '<='} C={cr.overall_acc:.1%}"},
            "v4_random_labels_at_chance": {"pass": v4,
                "detail": f"D={d.overall_acc:.1%} {'~' if v4 else '!='} 33% chance"},
            "v5_state_change_auc": {"pass": v5,
                "detail": f"A AUC={a.change_auc:.2f} {'>' if v5 else '<='} 0.7"},
        }
        print("\n" + "="*70 + "\nVERDICTS\n" + "="*70)
        n_pass = 0
        for vn, v in verdicts.items():
            tag = "PASS" if v["pass"] else "FAIL"; n_pass += int(v["pass"])
            print(f"  {tag}: {vn}  ({v['detail']})")
        print(f"\n  Overall: {n_pass}/5 passed")
        if n_pass >= 4:   print("  CONCLUSION: Embodied LM demonstrates introspective classification.")
        elif n_pass >= 3: print("  CONCLUSION: Partial evidence for introspective classification.")
        else:             print("  CONCLUSION: Introspective classification not conclusively demonstrated.")
        # Save
        out = {"experiment": "z1713_introspective_classification",
               "motivation": "z1707 failed at generation; classification tests internal state knowledge",
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "device": str(device),
               "vram_bytes": torch.cuda.get_device_properties(0).total_memory if device.type=='cuda' else 0,
               "config": {"batch_size": BATCH_SIZE, "seq_len": SEQ_LEN, "num_epochs": NUM_EPOCHS,
                          "lr": LR, "eval_batches": 120, "switch_every": 20,
                          "power_thresholds": [60,100], "temp_thresholds": [55,75],
                          "speed_thresholds": [1500,2500]},
               "conditions": {}, "verdicts": jsonify(verdicts),
               "n_passed": n_pass, "overall_pass": n_pass >= 4}
        for c in "ABCD":
            r = results[c]
            out["conditions"][c] = jsonify({"name": r.name,
                "power_acc": r.power_acc, "temp_acc": r.temp_acc, "speed_acc": r.speed_acc,
                "overall_acc": r.overall_acc, "change_auc": r.change_auc,
                "final_ppl": r.final_ppl, "ppl_history": r.ppl_history,
                "wall_s": round(r.wall_s,1), "energy_j": round(r.energy_j,1),
                "confusion": r.confusion})
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_PATH, 'w') as f: json.dump(out, f, indent=2)
        print(f"\nResults saved to: {RESULTS_PATH}")
    finally:
        torch.cuda.empty_cache()
        if not sim_act:
            try: actuator.restore_initial_state()
            except: pass
    return out

if __name__ == '__main__':
    main()
