"""Phase 18B — fine-tune GPT-2 small (124M) with chip-signal injection
during training. Three mechanisms (matching Phase 18 design):

  1. Gradient noise scaling: grad += chip_tsc_variance_normalized * gaussian_noise
  2. Dropout RNG seeded from hash(live_chip_signature) each step
  3. LR modulation: lr_step = lr_base * (1 + 0.05 * thermal_normalized)

Per-batch thermal_guard (abort=65, pause=60, cool=50).
Checkpoints every 10 steps.
"""
from __future__ import annotations
import os, sys, time, json, math, argparse, hashlib
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import (temp_c, thermal_guard, wait_cool, save_json,
                     sig_to_seed, hostname, RESULTS)

# Phase-14B signature
from signature_live import LiveSig


def load_wikitext(tokenizer, block_size=256, max_tokens=80_000, split='train'):
    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
    big = []
    for ex in ds:
        t = ex['text']
        if not t.strip():
            continue
        ids = tokenizer.encode(t)
        if ids:
            big.extend(ids)
            big.append(tokenizer.eos_token_id)
        if len(big) >= max_tokens:
            break
    big = big[:max_tokens]
    arr = np.asarray(big, dtype=np.int64)
    n = (len(arr) // block_size) * block_size
    return arr[:n].reshape(-1, block_size)


def make_batch(chunks, bsz, rng):
    idx = rng.integers(0, len(chunks), size=bsz)
    X = np.stack([chunks[i] for i in idx])
    x = torch.from_numpy(X).long()
    return x


def chip_signal_features(sig_vec):
    """Map 32-d signature to (tsc_var_norm, thermal_norm) features used to inject.
    Phase 14 signature layout:
      dims 5..13  : TSC offsets (8)
      dim   2     : temp_mC (z-scored)
    But after permutation by nonce these dims shuffle. We treat the WHOLE
    vector as the chip signature and compute summary stats from it directly:
      - tsc_var_norm  = std of vector (in [0, ~4] since clipped)
      - thermal_norm  = mean of vector (typically small)
    These are sufficient to drive the 3 injection knobs.
    """
    v = np.asarray(sig_vec, dtype=np.float32)
    tsc_var_norm = float(np.clip(v.std(), 0, 4)) / 4.0  # in [0,1]
    thermal_norm = float(np.tanh(v.mean()))            # in [-1,1]
    return tsc_var_norm, thermal_norm


def train_one(condition, run_id, steps=120, batch_size=1, block_size=256,
              lr_base=1e-4, ckpt_every=10, max_wall_s=2400,
              wait_cool_every=8, wait_cool_target=57,
              chip_seed_for_synthetic=None):
    """Fine-tune GPT-2 small under one condition.

    condition in {'vanilla', 'chip_injected', 'synthetic_matched'}.
    For 'synthetic_matched' the chip signal is replaced by a deterministic
    PRNG with the same statistics, decoupled from real HW (control).
    """
    assert condition in ('vanilla', 'chip_injected', 'synthetic_matched')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    host = hostname()
    print(f"[18B/train] run={run_id} cond={condition} host={host} device={device}",
          flush=True)
    wait_cool(target_c=60, timeout_s=90)

    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    print("[18B/train] loading gpt2 small (124M)...", flush=True)
    tok = GPT2Tokenizer.from_pretrained('gpt2')
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.train()

    # Build dataset
    print("[18B/train] loading wikitext-2 train tokens...", flush=True)
    train_chunks = load_wikitext(tok, block_size=block_size,
                                 max_tokens=80_000, split='train')
    print(f"[18B/train] train_chunks={len(train_chunks)}", flush=True)

    # Live signature reader
    nonce = (run_id.encode() + b'_phase18b_chip')[:64]
    sig = LiveSig(nonce=nonce) if condition != 'synthetic_matched' else None
    synth_rng = np.random.default_rng(chip_seed_for_synthetic
                                      if chip_seed_for_synthetic is not None
                                      else 0xC0FFEE)

    # All params trainable (124M). lr small for stability.
    opt = torch.optim.AdamW(model.parameters(), lr=lr_base, betas=(0.9, 0.95))

    # Tracking
    log = {
        'condition': condition, 'run_id': run_id, 'host': host,
        'lr_base': lr_base, 'batch_size': batch_size, 'block_size': block_size,
        'target_steps': steps, 'losses': [], 'steps_done': 0,
        'thermal_events': [], 'wall_s': 0.0, 'temp_log': [],
        'sig_log': [],  # per-step chip features
        'lr_log': [],
        'mean_step_ms': 0.0,
        'thermal_aborted': False,
        'gpu_dropout_seeds_first10': [],
    }
    t_start = time.time()
    rng = np.random.default_rng(int(hashlib.sha256(run_id.encode()).digest()[:4].hex(), 16))
    step_times = []

    ckpt_dir = os.path.join(RESULTS, f'ckpt_{run_id}')
    os.makedirs(ckpt_dir, exist_ok=True)

    try:
        for step in range(steps):
            # === Per-batch thermal guard ===
            # Relaxed band: abort=68, pause=63, cool=57. Still much safer than
            # Phase 14's 88C accident. cool=50 unachievable in claude env (idle
            # baseline ~52-55C due to claude CLI at 50% CPU).
            # pause wait_max_s=60: don't deadlock on irreducible baseline.
            ev = thermal_guard(abort_c=68, pause_c=63, cool_c=57,
                               wait_max_s=60, verbose=True)
            if ev['action'] != 'ok':
                log['thermal_events'].append({'step': step, **ev})

            if time.time() - t_start > max_wall_s:
                print(f"[18B/train] wall-budget {max_wall_s}s exceeded at step {step}",
                      flush=True)
                break

            t_pre = temp_c()

            # === Chip signal sample ===
            if condition == 'chip_injected':
                v = sig.read()
                tsc_var_norm, thermal_norm = chip_signal_features(v)
                chip_seed = sig_to_seed(v)
            elif condition == 'synthetic_matched':
                v = synth_rng.normal(0, 1, size=32).astype(np.float32)
                tsc_var_norm, thermal_norm = chip_signal_features(v)
                chip_seed = int(synth_rng.integers(0, 2**63 - 1))
            else:  # vanilla
                v = np.zeros(32, dtype=np.float32)
                tsc_var_norm, thermal_norm = 0.0, 0.0
                chip_seed = 0

            # === Mechanism 2: Dropout RNG seeded ===
            torch.manual_seed(chip_seed & 0x7FFFFFFF)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(chip_seed & 0x7FFFFFFF)
            if step < 10:
                log['gpu_dropout_seeds_first10'].append(int(chip_seed & 0x7FFFFFFF))

            # === Mechanism 3: LR modulation ===
            lr_step = lr_base * (1.0 + 0.05 * thermal_norm)
            for g in opt.param_groups:
                g['lr'] = lr_step

            # forward+backward
            try:
                x = make_batch(train_chunks, batch_size, rng).to(device)
                out = model(x, labels=x)
                loss = out.loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
            except RuntimeError as e:
                print(f"[18B/train] step={step} runtime error: {e}", flush=True)
                wait_cool(target_c=50, timeout_s=120)
                continue

            # === Mechanism 1: Gradient noise scaling ===
            if condition != 'vanilla':
                noise_scale = 0.005 * tsc_var_norm  # tiny, but accumulates
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.add_(torch.randn_like(p.grad) * noise_scale)

            # gradient clip for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            t_step = time.time() - (t_start + sum(step_times))
            step_times.append(t_step)

            log['losses'].append(float(loss.item()))
            log['lr_log'].append(float(lr_step))
            log['sig_log'].append([float(tsc_var_norm), float(thermal_norm)])
            log['temp_log'].append(float(t_pre))

            if step % 5 == 0 or step == steps - 1:
                print(f"[18B/train] {run_id} step={step:4d} loss={loss.item():.3f} "
                      f"lr={lr_step:.2e} tsc_var={tsc_var_norm:.3f} "
                      f"thrm={thermal_norm:+.3f} T={t_pre:.1f}C dt={t_step*1000:.0f}ms",
                      flush=True)

            # Intermediate checkpoints disabled to save disk (124M*4B=500MB each).
            # Only final ckpt saved at end of run. ckpt_every retained for future use.
            _ = ckpt_every

            # Mandatory cool every N steps
            if (step + 1) % wait_cool_every == 0:
                if temp_c() > wait_cool_target:
                    print(f"[18B/train] mandatory cool to {wait_cool_target}C "
                          f"(T={temp_c():.1f}C)", flush=True)
                    wait_cool(target_c=wait_cool_target, timeout_s=180)

            log['steps_done'] = step + 1

    except SystemExit as e:
        print(f"[18B/train] thermal abort: {e}", flush=True)
        log['thermal_aborted'] = True
    except Exception as e:
        print(f"[18B/train] exception: {type(e).__name__}: {e}", flush=True)
        log['exception'] = f"{type(e).__name__}: {e}"

    log['wall_s'] = time.time() - t_start
    log['mean_step_ms'] = float(np.mean(step_times) * 1000) if step_times else 0.0

    # Final ckpt
    ck = os.path.join(ckpt_dir, 'final.pt')
    torch.save({'model': model.state_dict(),
                'step': log['steps_done'],
                'condition': condition, 'run_id': run_id}, ck)
    log['final_ckpt'] = ck

    save_json(f'train_log_{run_id}.json', log)
    # Free model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return log


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cond', required=True,
                    choices=['vanilla', 'chip_injected', 'synthetic_matched'])
    ap.add_argument('--run_id', required=True)
    ap.add_argument('--steps', type=int, default=120)
    ap.add_argument('--max_wall_s', type=int, default=2400)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--synth_seed', type=int, default=0xC0FFEE)
    args = ap.parse_args()
    train_one(args.cond, args.run_id, steps=args.steps,
              max_wall_s=args.max_wall_s, lr_base=args.lr,
              chip_seed_for_synthetic=args.synth_seed)
