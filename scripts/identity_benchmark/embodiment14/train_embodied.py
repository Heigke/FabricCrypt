"""Phase 14 Task C — train the embodiment MLPs on WikiText-2.

Backbone GPT-2 is frozen; only the 3 injection MLPs (mlp_temp, mlp_gamma,
mlp_gain) and their gating coefficients are trained. Forward pass refreshes
the live HW signature on every step so the model learns to use the signal.

Strict thermal & wall budget.
"""
from __future__ import annotations
import os, sys, time, json, math
import torch, torch.nn as nn, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common14 import thermal_guard, wait_cool, save_json, hostname, get_apu_temp_c
from signature_io import LiveSignature
from embodied_gpt2 import EmbodiedGPT2, load_tokenizer
from dataset import get_loaders

OUT_DIR = os.path.abspath(os.path.join(HERE, '..', '..', '..',
            'results', 'IDENTITY_BENCHMARK_2026-05-30', 'embodiment14'))
os.makedirs(OUT_DIR, exist_ok=True)


def evaluate(model, loader, device, n_batches=None):
    model.eval()
    total_loss = 0.0
    total_tok  = 0
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if n_batches is not None and i >= n_batches: break
            thermal_guard(abort_c=67, pause_c=58, cool_c=50)
            input_ids = batch['input_ids'].to(device)
            labels    = batch['labels'].to(device)
            out = model(input_ids, labels=labels)
            ntok = labels.numel()
            total_loss += out.loss.item() * ntok
            total_tok  += ntok
            losses.append(out.loss.item())
            time.sleep(0.05)
    avg = total_loss / max(1, total_tok)
    return avg, math.exp(avg), losses


def main(steps=200, batch_size=4, block_size=128, lr=3e-4,
         max_wall_s=300, eval_every=50):
    host = hostname()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[train] host={host} device={device}", flush=True)

    sig = LiveSignature()
    tok = load_tokenizer()
    print(f"[train] building dataset...", flush=True)
    train_ld, eval_ld, ntr, nev = get_loaders(tok, block_size, batch_size,
                                              train_max=200_000, eval_max=50_000)
    print(f"[train] train chunks={ntr} eval chunks={nev}", flush=True)

    model = EmbodiedGPT2(sig_reader=sig).to(device)
    n_trainable = sum(p.numel() for p in model.trainable_parameters)
    print(f"[train] embodied trainable params={n_trainable:,}", flush=True)

    opt = torch.optim.AdamW(model.trainable_parameters, lr=lr)

    t_start = time.time()
    losses = []
    step_times = []
    train_iter = iter(train_ld)
    model.train()
    thermal_aborted = False
    try:
      for step in range(steps):
        # tight thermal: abort 67, pause 58, cool 50
        thermal_guard(abort_c=67, pause_c=58, cool_c=50)
        if time.time() - t_start > max_wall_s:
            print(f"[train] wall-budget {max_wall_s}s exceeded at step {step}", flush=True)
            break
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_ld); batch = next(train_iter)
        t0 = time.time()
        input_ids = batch['input_ids'].to(device)
        labels    = batch['labels'].to(device)
        out = model(input_ids, labels=labels)
        loss = out.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        dt = time.time() - t0
        step_times.append(dt)
        losses.append(loss.item())
        # inter-step cooldown — APU rises ~3-5C per step; brief sleep keeps it sane
        time.sleep(0.08)
        if step % 10 == 0 or step == steps-1:
            print(f"[train] step={step:4d} loss={loss.item():.4f} dt={dt*1000:.1f}ms temp={get_apu_temp_c():.1f}C", flush=True)
        if (step+1) % eval_every == 0:
            avg, ppl, _ = evaluate(model, eval_ld, device, n_batches=40)
            print(f"[eval]  step={step:4d} avg_loss={avg:.4f} ppl={ppl:.2f}", flush=True)
            wait_cool(target_c=52, timeout_s=120)
            model.train()
    except SystemExit as e:
        print(f"[train] thermal abort caught: {e} — saving partial checkpoint", flush=True)
        thermal_aborted = True

    # SAVE FIRST so weights survive regardless of eval outcome
    _early_ckpt = {
        'mlp_temp':  model.mlp_temp.state_dict(),
        'mlp_gamma': model.mlp_gamma.state_dict(),
        'mlp_gain':  model.mlp_gain.state_dict(),
        'alpha': model.alpha, 'beta': model.beta, 'delta': model.delta,
        'sig_mu':    sig.mu.tolist(),
        'sig_sigma': sig.sigma.tolist(),
        'host': host,
    }
    _early_path = os.path.join(OUT_DIR, f'embodied_ckpt_{host}.pt')
    torch.save(_early_ckpt, _early_path)
    print(f"[train] early-saved ckpt -> {_early_path}", flush=True)

    # final eval (small, with thermal checks)
    wait_cool(target_c=52, timeout_s=90)
    try:
        avg, ppl, eval_losses = evaluate(model, eval_ld, device, n_batches=60)
        print(f"[train] FINAL avg_loss={avg:.4f} ppl={ppl:.2f}", flush=True)
    except SystemExit as e:
        print(f"[train] eval thermal abort: {e}", flush=True)
        avg, ppl, eval_losses = float('nan'), float('nan'), []

    # save
    ckpt = {
        'mlp_temp':  model.mlp_temp.state_dict(),
        'mlp_gamma': model.mlp_gamma.state_dict(),
        'mlp_gain':  model.mlp_gain.state_dict(),
        'alpha': model.alpha, 'beta': model.beta, 'delta': model.delta,
        'sig_mu':    sig.mu.tolist(),
        'sig_sigma': sig.sigma.tolist(),
        'host': host,
    }
    ck_path = os.path.join(OUT_DIR, f'embodied_ckpt_{host}.pt')
    torch.save(ckpt, ck_path)
    rep = {
        'host': host,
        'steps_done': len(losses),
        'final_train_losses_last10': losses[-10:],
        'final_eval_avg_loss': avg,
        'final_eval_ppl': ppl,
        'mean_step_ms': float(np.mean(step_times) * 1000) if step_times else None,
        'thermal_aborted': thermal_aborted,
        'wall_s': time.time() - t_start,
        'temp_end_c': get_apu_temp_c(),
        'eval_loss_per_batch': eval_losses[:50],
    }
    save_json(os.path.join(OUT_DIR, f'train_report_{host}.json'), rep)
    print(f"[train] saved ckpt -> {ck_path}", flush=True)
    return rep


if __name__ == '__main__':
    steps     = int(os.environ.get('STEPS', '200'))
    bs        = int(os.environ.get('BSZ',   '4'))
    bs_block  = int(os.environ.get('BLK',   '128'))
    wall      = int(os.environ.get('WALL',  '300'))
    main(steps=steps, batch_size=bs, block_size=bs_block, max_wall_s=wall)
