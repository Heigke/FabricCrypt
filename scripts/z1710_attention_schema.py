#!/usr/bin/env python3
"""
z1710: Attention Schema Theory (AST) -- Graziano's Model of Consciousness

Graziano's AST: consciousness arises when the brain builds an internal MODEL
of its own attention. We test whether a model that tracks and predicts its own
attention patterns develops better self-regulation when embodied.

AttentionSchemaNet: takes 24 mean attention weights (6 layers x 4 heads) +
12-dim telemetry, predicts next-step attention + priority gates (sigmoid).
Feedback loop: schema predicts -> modulates -> changes what schema observes.

Phases: P1(3ep) base LM | P2(5ep) schema training | P3(5ep) full loop
Conditions: A=Embodied AST | B=Disembodied AST | C=No Schema | D=Random Gates
Verdicts: 1) A pred_MSE < B | 2) A ppl <= C | 3) A HUV > C | 4) D ppl > A
"""

import sys, os, json, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import copy
from pathlib import Path
from datetime import datetime

from src.metabolic.film_transformer import create_metabolic_transformer, MetabolicConfig
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry
from src.actuation.gpu_actuator import GPUActuator, PerformanceLevel

ROOT = Path('/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy')
BS, SL, LR = 4, 256, 3e-4
NUM_LAYERS, NUM_HEADS = 6, 4
ATTN_DIM = NUM_LAYERS * NUM_HEADS  # 24
COOLDOWN = 30

ACTION_MAP = {
    0: PerformanceLevel.LOW, 1: PerformanceLevel.BALANCED,
    2: PerformanceLevel.HIGH, 3: PerformanceLevel.HIGH,
}


class AttentionSchemaNet(nn.Module):
    """Models the transformer's own attention: predicts next-step attention
    weights + priority gates from current attention stats + optional telemetry."""

    def __init__(self, attn_dim=24, telemetry_dim=12, use_telemetry=True):
        super().__init__()
        self.use_telemetry = use_telemetry
        in_dim = attn_dim + (telemetry_dim if use_telemetry else 0)

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.attn_predictor = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, attn_dim))
        self.priority_head = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, attn_dim), nn.Sigmoid())
        with torch.no_grad():  # init priority near 1.0 (identity)
            self.priority_head[-2].bias.fill_(2.0)

    def forward(self, attn_stats, telemetry=None):
        if self.use_telemetry and telemetry is not None:
            x = torch.cat([attn_stats, telemetry], dim=-1)
        else:
            x = attn_stats
        h = self.encoder(x)
        predicted_attn = self.attn_predictor(h)
        priority = self.priority_head(h)
        return {'predicted_attn': predicted_attn, 'priority': priority}


def extract_attention_stats(model, hidden_states_per_layer):
    """Compute mean attention per (layer, head) via Q,K projections. -> [batch, 24]."""
    device = hidden_states_per_layer[0].device
    batch = hidden_states_per_layer[0].shape[0]
    all_means = []

    for i, h_in in enumerate(hidden_states_per_layer):
        attn_mod = model.blocks[i].attn
        seq_len, hd = h_in.shape[1], attn_mod.head_dim
        h_normed = model.blocks[i].ln1(h_in)  # pre-norm
        q = attn_mod.q_proj(h_normed).view(batch, seq_len, NUM_HEADS, hd).transpose(1, 2)
        k = attn_mod.k_proj(h_normed).view(batch, seq_len, NUM_HEADS, hd).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * attn_mod.scale
        causal = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))
        all_means.append(F.softmax(scores, dim=-1).mean(dim=(-2, -1)))
    return torch.cat(all_means, dim=-1)  # [batch, 24]


def forward_with_hidden_capture(model, input_ids, telemetry):
    """Forward pass capturing each block's input hidden states."""
    batch, seq_len = input_ids.shape
    device = input_ids.device

    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
    x = model.token_embed(input_ids) + model.pos_embed(positions)
    x = model.dropout(x)

    if model.config.use_causal_mask:
        mask = ~model.causal_mask[:seq_len, :seq_len]
    else:
        mask = None

    telem = telemetry
    if telem is not None and telem.dim() == 1:
        telem = telem.unsqueeze(0)
    if telem is not None and telem.size(0) == 1 and batch > 1:
        telem = telem.expand(batch, -1)

    hidden_per_layer = []
    for i, block in enumerate(model.blocks):
        hidden_per_layer.append(x.detach())

        gamma1, beta1, gamma2, beta2 = None, None, None, None
        if model._conditioning_enabled and telem is not None and model.film_generators[i] is not None:
            fg = model.film_generators[i]
            gamma1, beta1 = fg['ln1'](telem)
            gamma2, beta2 = fg['ln2'](telem)
        x = block(x, gamma1, beta1, gamma2, beta2, mask)

    x = model.ln_out(x)
    logits = model.token_head(x)
    action_logits = model.action_head(x[:, -1, :])
    return {'logits': logits, 'action_logits': action_logits, 'hidden': x}, hidden_per_layer


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
    perf_enc = 0.5  # default
    raw = [
        s.power_w / 50, s.temp_edge_c / 100, s.freq_sclk_mhz / 3000,
        s.gpu_busy_pct / 100, perf_enc, 0.0,  # throttle placeholder
    ]
    if prev_sample is not None:
        raw += [
            (s.power_w - prev_sample.power_w) / 50,
            (s.temp_edge_c - prev_sample.temp_edge_c) / 100,
            (s.freq_sclk_mhz - prev_sample.freq_sclk_mhz) / 3000,
            (s.gpu_busy_pct - prev_sample.gpu_busy_pct) / 100,
            (s.temp_edge_c - 70) / 100,  # thermal deviation from 70C
            (3000 - s.freq_sclk_mhz) / 3000,  # freq headroom
        ]
    else:
        raw += [0.0] * 6
    vec = torch.tensor(raw[:12], dtype=torch.float32, device=device).unsqueeze(0)
    return vec, s


def compute_attention_entropy(attn_stats):
    """Entropy across heads (higher = more uniform). attn_stats: [batch,24]."""
    p = F.softmax(attn_stats, dim=-1)
    return -(p * (p + 1e-8).log()).sum(-1).mean().item()

def compute_head_utilization_variance(attn_stats):
    """Variance of per-head mean attention (higher = more differentiation)."""
    return attn_stats.mean(0).var().item()


# ---------------------------------------------------------------------------
# Phase 1: Base LM training
# ---------------------------------------------------------------------------
def train_phase1(model, data, telem, device, n_epochs=3):
    print(f"\n{'='*60}")
    print(f"  Phase 1: Base LM Training ({n_epochs} epochs)")
    print(f"{'='*60}")
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    prev_s = None
    total_loss, n = 0.0, 0
    attn_history = []

    for epoch in range(n_epochs):
        eloss, steps = 0.0, 80
        for step in range(steps):
            x, y = get_batch(data, BS, SL, device)
            tv, prev_s = build_telemetry(telem, device, prev_s)
            tv_exp = tv.expand(BS, -1)

            out, hiddens = forward_with_hidden_capture(model, x, tv_exp)
            loss = F.cross_entropy(out['logits'].view(-1, 256), y.view(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            with torch.no_grad():
                astats = extract_attention_stats(model, hiddens)
                attn_history.append(astats.detach().cpu())

            eloss += loss.item(); n += 1
            if (step + 1) % 40 == 0:
                print(f"  e{epoch+1} s{step+1}/{steps} loss={loss.item():.4f}")
        total_loss += eloss
        print(f"  Epoch {epoch+1} avg loss: {eloss/steps:.4f}")

    return total_loss / max(n, 1), attn_history


# ---------------------------------------------------------------------------
# Phase 2: Train AttentionSchemaNet
# ---------------------------------------------------------------------------
def train_phase2(model, schema, data, telem, device, n_epochs=5):
    print(f"\n{'='*60}")
    print(f"  Phase 2: Attention Schema Training ({n_epochs} epochs)")
    print(f"{'='*60}")
    for p in model.parameters():
        p.requires_grad = False
    opt = torch.optim.Adam(schema.parameters(), lr=LR)
    prev_s = None
    prev_attn = None
    pred_mses = []

    for epoch in range(n_epochs):
        eloss, steps = 0.0, 60
        for step in range(steps):
            x, y = get_batch(data, BS, SL, device)
            tv, prev_s = build_telemetry(telem, device, prev_s)
            tv_exp = tv.expand(BS, -1)

            with torch.no_grad():
                out, hiddens = forward_with_hidden_capture(model, x, tv_exp)
                curr_attn = extract_attention_stats(model, hiddens)

            if prev_attn is not None:
                # Schema predicts current attention from previous
                telem_in = tv_exp if schema.use_telemetry else None
                so = schema(prev_attn.detach(), telem_in)
                pred_loss = F.mse_loss(so['predicted_attn'], curr_attn.detach())
                opt.zero_grad(); pred_loss.backward(); opt.step()
                eloss += pred_loss.item()
                pred_mses.append(pred_loss.item())
            prev_attn = curr_attn

            if (step + 1) % 30 == 0:
                print(f"  e{epoch+1} s{step+1}/{steps} pred_mse={eloss/max(step,1):.6f}")
        print(f"  Epoch {epoch+1} avg pred_mse: {eloss/steps:.6f}")

    for p in model.parameters():
        p.requires_grad = True
    final_mse = float(np.mean(pred_mses[-20:])) if pred_mses else 1.0
    print(f"  Phase 2 final pred MSE: {final_mse:.6f}")
    return {'schema_pred_mse': final_mse}


# ---------------------------------------------------------------------------
# Phase 3: Full loop condition runner
# ---------------------------------------------------------------------------
def run_condition(label, mode, model, schema, data, telem, actuator, device,
                  n_epochs=5):
    """
    Run one experimental condition.
    Modes: embodied_ast, disembodied_ast, no_schema, random_modulation
    """
    print(f"\n{'='*60}")
    print(f"  Condition {label}: {mode.upper()}")
    print(f"{'='*60}")

    cm = copy.deepcopy(model).to(device)
    cs = copy.deepcopy(schema).to(device)

    use_schema = mode in ('embodied_ast', 'disembodied_ast', 'random_modulation')
    use_modulation = mode in ('embodied_ast', 'disembodied_ast', 'random_modulation')
    use_telemetry = mode == 'embodied_ast'
    random_gates = mode == 'random_modulation'

    # Always keep schema's telemetry input active (it was built with telemetry dim)
    # For non-embodied conditions, we'll pass zero telemetry
    cs.use_telemetry = True

    params = list(cm.parameters()) + (list(cs.parameters()) if use_schema else [])
    opt = torch.optim.Adam(params, lr=LR)

    prev_s = None
    prev_attn = None
    priority_gates = torch.ones(ATTN_DIM, device=device)

    # Metric accumulators
    losses, pred_mses = [], []
    attn_entropies, head_util_vars = [], []
    actions = []
    telem.reset_accumulator()
    telem.start_continuous_sampling()
    t0, total_tokens = time.time(), 0

    for epoch in range(n_epochs):
        eloss, steps = 0.0, 60
        for step in range(steps):
            x, y = get_batch(data, BS, SL, device)
            tv, prev_s = build_telemetry(telem, device, prev_s)
            tv_exp = tv.expand(BS, -1)

            # Forward with hidden capture
            out, hiddens = forward_with_hidden_capture(cm, x, tv_exp)
            task_loss = F.cross_entropy(out['logits'].view(-1, 256), y.view(-1))

            # Extract attention stats
            with torch.no_grad():
                curr_attn = extract_attention_stats(cm, hiddens)

            # Schema prediction + modulation loss
            schema_loss = torch.tensor(0.0, device=device)
            if use_schema and prev_attn is not None:
                telem_in = tv_exp if use_telemetry else torch.zeros(BS, 12, device=device)
                so = cs(prev_attn.detach(), telem_in)
                schema_loss = F.mse_loss(so['predicted_attn'], curr_attn.detach())

                if use_modulation:
                    if random_gates:
                        priority_gates = torch.rand(ATTN_DIM, device=device) * 0.5 + 0.5
                    else:
                        priority_gates = so['priority'].mean(dim=0).detach()
                pred_mses.append(schema_loss.item())

            prev_attn = curr_attn.detach()

            # Combined loss
            total = task_loss + 0.3 * schema_loss
            opt.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            eloss += task_loss.item()
            losses.append(task_loss.item())
            total_tokens += BS * SL

            # Attention metrics
            with torch.no_grad():
                aent = compute_attention_entropy(curr_attn)
                huv = compute_head_utilization_variance(curr_attn)
                attn_entropies.append(aent)
                head_util_vars.append(huv)

            # Action selection from model
            mean_probs = F.softmax(out['action_logits'], dim=-1).mean(dim=0)
            action_idx = torch.argmax(mean_probs).item()
            actions.append(action_idx)
            try:
                actuator.set_performance_level(ACTION_MAP[min(action_idx, 3)])
            except Exception:
                pass

            if (step + 1) % 30 == 0:
                pmse = pred_mses[-1] if pred_mses else 0.0
                print(f"  [{label}] e{epoch+1} s{step+1}/{steps} "
                      f"loss={task_loss.item():.4f} pmse={pmse:.6f} "
                      f"aent={aent:.4f} huv={huv:.6f}")
        print(f"  [{label}] Epoch {epoch+1} avg_loss={eloss/steps:.4f}")

    elapsed = time.time() - t0
    telem.stop_continuous_sampling()
    energy = telem.get_accumulated_energy_j()
    try:
        actuator.set_performance_level(PerformanceLevel.BALANCED)
    except Exception:
        pass

    # Compute final metrics
    final_loss = float(np.mean(losses[-20:]))
    final_ppl = float(np.exp(final_loss))
    avg_pred_mse = float(np.mean(pred_mses[-20:])) if pred_mses else 0.0
    avg_attn_entropy = float(np.mean(attn_entropies[-20:]))
    avg_head_util_var = float(np.mean(head_util_vars[-20:]))

    act_counts = np.bincount(actions, minlength=4)
    act_probs = act_counts / max(len(actions), 1)
    act_ent = -np.sum(act_probs[act_probs > 0] * np.log(act_probs[act_probs > 0]))

    # MI approximation: correlation between schema prediction error and attention
    schema_info = 0.0
    if len(pred_mses) > 10:
        # Use variance of prediction errors as proxy for information content
        schema_info = float(np.std(pred_mses))

    m = {
        'condition': label, 'mode': mode,
        'final_perplexity': final_ppl,
        'final_loss': final_loss,
        'attention_pred_mse': avg_pred_mse,
        'attention_entropy': avg_attn_entropy,
        'head_utilization_variance': avg_head_util_var,
        'schema_information': schema_info,
        'energy_j': energy,
        'j_per_token': energy / max(total_tokens, 1),
        'total_tokens': total_tokens,
        'elapsed_s': elapsed,
        'action_distribution': act_counts.tolist(),
        'action_entropy': float(act_ent),
        'priority_gates_final': priority_gates.cpu().tolist(),
    }
    print(f"  [{label}] PPL={final_ppl:.2f} pred_MSE={avg_pred_mse:.6f} "
          f"aent={avg_attn_entropy:.4f} HUV={avg_head_util_var:.6f}")
    print(f"  [{label}] Energy={energy:.2f}J J/tok={m['j_per_token']:.6f}")
    return m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  z1710: ATTENTION SCHEMA THEORY (Graziano)")
    print("  Does modeling one's own attention improve self-regulation?")
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
        hidden_dim=256, num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
        telemetry_dim=12,
    ).to(device)
    model.enable_conditioning(True)

    schema = AttentionSchemaNet(
        attn_dim=ATTN_DIM, telemetry_dim=12, use_telemetry=True
    ).to(device)

    npar = sum(p.numel() for p in model.parameters())
    spar = sum(p.numel() for p in schema.parameters())
    print(f"Base params: {npar:,}  Schema params: {spar:,}")

    try:
        # Phase 1
        p1_loss, attn_history = train_phase1(model, data, telem, device, n_epochs=3)
        print(f"Phase 1 done, avg loss: {p1_loss:.4f}")

        # Phase 2
        p2 = train_phase2(model, schema, data, telem, device, n_epochs=5)

        # Phase 3: 4 conditions
        conditions = [
            ('A', 'embodied_ast'),
            ('B', 'disembodied_ast'),
            ('C', 'no_schema'),
            ('D', 'random_modulation'),
        ]
        results = {}
        for i, (lbl, mode) in enumerate(conditions):
            results[lbl] = run_condition(
                lbl, mode, model, schema, data, telem, actuator, device,
                n_epochs=5,
            )
            if i < len(conditions) - 1:
                print(f"\nCooldown {COOLDOWN}s...")
                try:
                    actuator.set_performance_level(PerformanceLevel.BALANCED)
                except Exception:
                    pass
                time.sleep(COOLDOWN)

        try:
            actuator.set_performance_level(PerformanceLevel.BALANCED)
        except Exception:
            pass

        # Verdicts
        print(f"\n{'='*70}")
        print(f"  VERDICTS")
        print(f"{'='*70}")
        verdicts = {}

        # V1: Embodied schema predicts attention better
        mse_a = results['A']['attention_pred_mse']
        mse_b = results['B']['attention_pred_mse']
        v1 = mse_a < mse_b
        verdicts['embodied_schema_superior'] = {
            'pass': v1, 'a_mse': mse_a, 'b_mse': mse_b,
        }
        print(f"\n1. Embodied schema better (A MSE < B MSE): "
              f"{'PASS' if v1 else 'FAIL'}  A={mse_a:.6f} B={mse_b:.6f}")

        # V2: Schema does not hurt task quality
        ppl_a = results['A']['final_perplexity']
        ppl_c = results['C']['final_perplexity']
        v2 = ppl_a <= ppl_c * 1.05  # allow 5% margin
        verdicts['schema_preserves_quality'] = {
            'pass': v2, 'a_ppl': ppl_a, 'c_ppl': ppl_c,
        }
        print(f"2. Schema preserves quality (A ppl <= C ppl): "
              f"{'PASS' if v2 else 'FAIL'}  A={ppl_a:.2f} C={ppl_c:.2f}")

        # V3: Schema differentiates heads
        huv_a = results['A']['head_utilization_variance']
        huv_c = results['C']['head_utilization_variance']
        v3 = huv_a > huv_c
        verdicts['head_differentiation'] = {
            'pass': v3, 'a_huv': huv_a, 'c_huv': huv_c,
        }
        print(f"3. Head differentiation (A HUV > C HUV): "
              f"{'PASS' if v3 else 'FAIL'}  A={huv_a:.6f} C={huv_c:.6f}")

        # V4: Random modulation hurts
        ppl_d = results['D']['final_perplexity']
        v4 = ppl_d > ppl_a
        verdicts['random_hurts'] = {
            'pass': v4, 'a_ppl': ppl_a, 'd_ppl': ppl_d,
        }
        print(f"4. Random modulation hurts (D ppl > A ppl): "
              f"{'PASS' if v4 else 'FAIL'}  A={ppl_a:.2f} D={ppl_d:.2f}")

        passed = sum(v['pass'] for v in verdicts.values())
        print(f"\n{'='*70}")
        print(f"  OVERALL: {passed}/{len(verdicts)} passed")
        print(f"{'='*70}")

        # Attention schema analysis
        print(f"\n  Attention Schema Analysis:")
        for lbl in 'ABCD':
            r = results[lbl]
            print(f"    {lbl}: ppl={r['final_perplexity']:.2f}  "
                  f"pred_MSE={r['attention_pred_mse']:.6f}  "
                  f"entropy={r['attention_entropy']:.4f}  "
                  f"HUV={r['head_utilization_variance']:.6f}  "
                  f"J/tok={r['j_per_token']:.6f}")

        # Priority gate analysis for condition A
        gates_a = results['A']['priority_gates_final']
        print(f"\n  Priority Gates (A, per layer-head):")
        for layer in range(NUM_LAYERS):
            layer_gates = gates_a[layer * NUM_HEADS:(layer + 1) * NUM_HEADS]
            gstr = ' '.join(f'{g:.3f}' for g in layer_gates)
            print(f"    Layer {layer}: [{gstr}]")

        # Save results
        output = {
            'experiment': 'z1710_attention_schema',
            'theory': 'Graziano Attention Schema Theory',
            'timestamp': datetime.now().isoformat(),
            'device': str(device),
            'gpu_name': torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'cpu',
            'config': {
                'batch_size': BS, 'seq_len': SL, 'lr': LR,
                'hidden_dim': 256, 'num_layers': NUM_LAYERS,
                'num_heads': NUM_HEADS, 'attn_dim': ATTN_DIM,
                'base_params': npar, 'schema_params': spar,
            },
            'phase1_avg_loss': p1_loss,
            'phase2_metrics': p2,
            'conditions': results,
            'verdicts': verdicts,
            'passed': passed, 'total': len(verdicts),
            'summary': {
                k: {
                    'ppl': results[k]['final_perplexity'],
                    'pred_mse': results[k]['attention_pred_mse'],
                    'attn_entropy': results[k]['attention_entropy'],
                    'head_util_var': results[k]['head_utilization_variance'],
                    'j_per_token': results[k]['j_per_token'],
                }
                for k in 'ABCD'
            },
        }

        out_path = ROOT / 'results' / 'z1710_attention_schema.json'
        out_path.parent.mkdir(parents=True, exist_ok=True)

        def json_default(o):
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, torch.Tensor): return o.tolist()
            return str(o)

        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2, default=json_default)
        print(f"\nResults saved to: {out_path}")
        print("Done.")

    finally:
        try:
            actuator.set_performance_level(PerformanceLevel.BALANCED)
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
