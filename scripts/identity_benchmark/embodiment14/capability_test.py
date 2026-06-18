"""Phase 14 Task D — capability comparison vanilla vs embodied.

Loads:
   A: vanilla GPT-2 (no embodiment)         -> PPL_A
   B: embodied GPT-2 + trained MLPs (live signature)   -> PPL_B

Bootstrap 95% CI on per-batch losses. Pre-reg: B PPL <= 0.97 * A PPL.

Optionally also loads:
   B_zero: embodied GPT-2 with signature forced to zero  (control)
"""
from __future__ import annotations
import os, sys, json, math, time, argparse
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common14 import thermal_guard, save_json, hostname, get_apu_temp_c, wait_cool
from signature_io import LiveSignature
from embodied_gpt2 import EmbodiedGPT2, VanillaGPT2, load_tokenizer
from dataset import get_loaders

OUT_DIR = os.path.abspath(os.path.join(HERE, '..', '..', '..',
            'results', 'IDENTITY_BENCHMARK_2026-05-30', 'embodiment14'))


def eval_per_batch(model, loader, device, n_batches=200, label=''):
    model.eval()
    losses = []
    ntok_per = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches: break
            thermal_guard(abort_c=67, pause_c=58, cool_c=50)
            ids = batch['input_ids'].to(device)
            lbl = batch['labels'].to(device)
            out = model(ids, labels=lbl)
            losses.append(out.loss.item())
            ntok_per.append(lbl.numel())
            time.sleep(0.04)
            if i % 50 == 0:
                print(f"[eval/{label}] batch {i}/{n_batches} loss={out.loss.item():.3f} temp={get_apu_temp_c():.1f}C", flush=True)
    return np.asarray(losses), np.asarray(ntok_per)


def bootstrap_ppl(losses, B=2000, rng=None):
    rng = rng or np.random.default_rng(0)
    n = len(losses)
    avgs = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        avgs[b] = losses[idx].mean()
    ppls = np.exp(avgs)
    return float(np.mean(ppls)), float(np.percentile(ppls, 2.5)), float(np.percentile(ppls, 97.5))


def main(ckpt_path=None, n_eval_batches=200, batch_size=4, block_size=128):
    host = hostname()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if ckpt_path is None:
        ckpt_path = os.path.join(OUT_DIR, f'embodied_ckpt_{host}.pt')

    tok = load_tokenizer()
    _, eval_ld, _, nev = get_loaders(tok, block_size, batch_size,
                                     train_max=10_000, eval_max=80_000)
    print(f"[cap] eval chunks={nev}", flush=True)

    # --- Vanilla
    print(f"[cap] loading vanilla...", flush=True)
    vanilla = VanillaGPT2().to(device)
    a_losses, _ = eval_per_batch(vanilla, eval_ld, device, n_eval_batches, label='vanilla')
    del vanilla; torch.cuda.empty_cache()
    wait_cool(target_c=55, timeout_s=120)

    # --- Embodied (live sig)
    print(f"[cap] loading embodied + live signature...", flush=True)
    sig = LiveSignature()
    embodied = EmbodiedGPT2(sig_reader=sig).to(device)
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        embodied.mlp_temp.load_state_dict(ck['mlp_temp'])
        embodied.mlp_gamma.load_state_dict(ck['mlp_gamma'])
        embodied.mlp_gain.load_state_dict(ck['mlp_gain'])
        print(f"[cap] loaded ckpt from {ckpt_path}", flush=True)
    else:
        print(f"[cap] WARN no ckpt -> embodied MLPs at zero-init (control)", flush=True)
    b_losses, _ = eval_per_batch(embodied, eval_ld, device, n_eval_batches, label='embodied_live')
    wait_cool(target_c=55, timeout_s=120)

    # --- Embodied with signature forced to zero (control: same arch but no live signal)
    print(f"[cap] embodied with zero signature override...", flush=True)
    embodied.set_signature_override(torch.zeros(32))
    b0_losses, _ = eval_per_batch(embodied, eval_ld, device, n_eval_batches, label='embodied_zero')
    embodied.set_signature_override(None)
    del embodied; torch.cuda.empty_cache()

    # bootstrap
    ppl_A, A_lo, A_hi = bootstrap_ppl(a_losses)
    ppl_B, B_lo, B_hi = bootstrap_ppl(b_losses)
    ppl_B0, B0_lo, B0_hi = bootstrap_ppl(b0_losses)
    rel_drop = (ppl_A - ppl_B) / ppl_A * 100
    rel_drop_zero = (ppl_A - ppl_B0) / ppl_A * 100

    out = {
        'host': host,
        'n_eval_batches': n_eval_batches,
        'batch_size': batch_size, 'block_size': block_size,
        'ppl_A_vanilla': {'mean': ppl_A, 'ci95': [A_lo, A_hi]},
        'ppl_B_embodied_live': {'mean': ppl_B, 'ci95': [B_lo, B_hi]},
        'ppl_B0_embodied_zerosig': {'mean': ppl_B0, 'ci95': [B0_lo, B0_hi]},
        'relative_improvement_pct_live': rel_drop,
        'relative_improvement_pct_zerosig': rel_drop_zero,
        'pre_reg_gate_3pct_improvement': rel_drop >= 3.0,
        'a_losses_mean': float(a_losses.mean()),
        'b_losses_mean': float(b_losses.mean()),
        'b0_losses_mean': float(b0_losses.mean()),
        'a_losses_first20': a_losses[:20].tolist(),
        'b_losses_first20': b_losses[:20].tolist(),
        't_end': time.time(),
        'temp_end_c': get_apu_temp_c(),
    }
    save_json(os.path.join(OUT_DIR, f'capability_{host}.json'), out)
    print(json.dumps({k: out[k] for k in
        ('ppl_A_vanilla','ppl_B_embodied_live','ppl_B0_embodied_zerosig',
         'relative_improvement_pct_live','relative_improvement_pct_zerosig',
         'pre_reg_gate_3pct_improvement')}, indent=2))
    return out


if __name__ == '__main__':
    n = int(os.environ.get('NEVAL', '200'))
    main(n_eval_batches=n)
