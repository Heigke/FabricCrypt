"""Phase 21 — train chip-personality models.

Stronger chip injection than Phase 18B. Three mechanisms:
  1. LoRA-style chip perturbation injected DIRECTLY into attention weights
     every step. Δ = α * (u_chip ⊗ v_chip), low-rank, accumulating.
  2. Per-layer LR multiplier from chip-sig dims (each layer different).
  3. Chip-seeded data ordering (per-step batch index permutation).

Usage:
  python train_personality.py --cond chip_inject --run_id chip_dae_500 \
      --steps 500 --lr 1e-4 --model distilgpt2 \
      --out ~/phase21/results
"""
from __future__ import annotations
import os, sys, time, json, math, argparse, hashlib
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import (temp_c, thermal_guard, wait_cool, save_json, hostname,
                     LiveSig, sig_to_seed)


def load_wikitext(tokenizer, block_size=256, max_tokens=200_000, split='train'):
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


def get_attn_modules(model):
    """Return list of (name, c_attn linear module) for chip-perturbation injection."""
    mods = []
    for name, m in model.named_modules():
        # GPT-2 uses Conv1D for c_attn/c_proj. Both distilgpt2 and gpt2-medium.
        if name.endswith('.attn.c_attn') or name.endswith('.attn.c_proj'):
            mods.append((name, m))
    return mods


def inject_chip_perturbation(attn_mods, sig_vec, alpha=1e-3, rng=None):
    """Mechanism 1: low-rank chip-derived perturbation into attention weights.
    For each attention layer, build rank-1 update u v^T from chip-sig hash bits.
    `rng` is a torch.Generator seeded from chip-sig for this step.
    """
    # Build deterministic u,v from sig + layer index — chip-derived direction.
    # Use sig bytes to seed a fast generator per call.
    sig_bytes = np.asarray(sig_vec, dtype=np.float32).tobytes()
    h = hashlib.sha256(sig_bytes).digest()
    seed = int.from_bytes(h[:8], 'little') & 0x7FFFFFFF
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    with torch.no_grad():
        for i, (name, m) in enumerate(attn_mods):
            W = m.weight  # Conv1D weight shape (in, out)
            # rank-1: u (in,) v (out,) — small chip-derived direction
            u = torch.randn(W.shape[0], generator=g) * float(sig_vec[i % len(sig_vec)])
            v = torch.randn(W.shape[1], generator=g) * float(sig_vec[(i + 7) % len(sig_vec)])
            u = u.to(W.device, W.dtype)
            v = v.to(W.device, W.dtype)
            # normalize, then scale by chip-derived magnitude
            un = u / (u.norm() + 1e-6)
            vn = v / (v.norm() + 1e-6)
            chip_mag = float(np.tanh(np.sum(np.asarray(sig_vec[:8]) ** 2) / 4.0))
            delta = alpha * abs(chip_mag + 0.5) * torch.outer(un, vn)
            W.add_(delta)


def per_layer_lr(opt, model, sig_vec, lr_base, scale=0.20):
    """Mechanism 2: per-layer LR multiplier from chip-sig dims.
    Apply different LR to each transformer block via param_groups."""
    # Param groups were created per-block at init. Here we just scale them.
    blocks = list(model.transformer.h)
    n = len(blocks)
    for li in range(n):
        m = 1.0 + scale * float(np.tanh(sig_vec[li % len(sig_vec)]))
        # opt.param_groups[li] corresponds to block li (see opt construction)
        opt.param_groups[li]['lr'] = lr_base * m
    # last param group is the rest (head/embeddings/ln_f); keep lr_base
    opt.param_groups[-1]['lr'] = lr_base


def make_batch_chip_ordered(chunks, bsz, sig_vec, step, base_rng):
    """Mechanism 3: chip-seeded data ordering.
    Mix per-step chip seed into batch index draw."""
    sig_seed = sig_to_seed(sig_vec) ^ step
    mix = int(base_rng.integers(0, 2**31))
    g = np.random.default_rng((int(sig_seed) ^ mix) & 0x7FFFFFFFFFFFFFFF)
    idx = g.integers(0, len(chunks), size=bsz)
    X = np.stack([chunks[i] for i in idx])
    return torch.from_numpy(X).long()


def train_one(condition, run_id, steps, model_name='distilgpt2',
              lr_base=1e-4, batch_size=2, block_size=256,
              ckpt_every=100, alpha_lora=1e-3,
              thermal_band=(80, 72, 65),
              wait_cool_every=20, wait_cool_target=68,
              out_dir='./results', max_wall_s=14400):
    """Train one condition.

    condition: vanilla | chip_inject | synthetic_matched
    """
    assert condition in ('vanilla', 'chip_inject', 'synthetic_matched')
    abort_c, pause_c, cool_c = thermal_band
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    host = hostname()
    print(f"[21/train] run={run_id} cond={condition} host={host} model={model_name}",
          flush=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"[21/train] loading {model_name}...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[21/train] n_params={n_params/1e6:.1f}M, n_blocks={len(model.transformer.h)}",
          flush=True)

    # Build per-block param groups for per-layer LR
    param_groups = []
    block_params = set()
    for blk in model.transformer.h:
        params = list(blk.parameters())
        for p in params:
            block_params.add(id(p))
        param_groups.append({'params': params, 'lr': lr_base})
    rest = [p for p in model.parameters() if id(p) not in block_params]
    param_groups.append({'params': rest, 'lr': lr_base})
    opt = torch.optim.AdamW(param_groups, lr=lr_base, betas=(0.9, 0.95))

    attn_mods = get_attn_modules(model)
    print(f"[21/train] attn_modules to perturb: {len(attn_mods)}", flush=True)

    # Data
    print(f"[21/train] loading wikitext...", flush=True)
    train_chunks = load_wikitext(tok, block_size=block_size, max_tokens=200_000)
    print(f"[21/train] train_chunks={len(train_chunks)}", flush=True)

    nonce = (run_id.encode() + b'_phase21')[:64]
    sig = LiveSig(nonce=nonce) if condition != 'synthetic_matched' else None
    synth_rng = np.random.default_rng(0xC0FFEE if condition == 'synthetic_matched' else 0)

    log = {
        'condition': condition, 'run_id': run_id, 'host': host,
        'model_name': model_name, 'lr_base': lr_base,
        'batch_size': batch_size, 'block_size': block_size,
        'target_steps': steps, 'alpha_lora': alpha_lora,
        'n_params': int(n_params),
        'losses': [], 'lr_by_layer_log': [], 'sig_norm_log': [],
        'temp_log': [], 'thermal_events': [], 'mean_step_ms': 0.0,
        'thermal_aborted': False, 'steps_done': 0, 'wall_s': 0.0,
        'ckpts': [],
    }
    t_start = time.time()
    base_rng = np.random.default_rng(int(hashlib.sha256(run_id.encode()).digest()[:4].hex(), 16))
    step_times = []

    ckpt_dir = os.path.join(out_dir, f'ckpt_{run_id}')
    os.makedirs(ckpt_dir, exist_ok=True)

    try:
        for step in range(steps):
            ev = thermal_guard(abort_c=abort_c, pause_c=pause_c, cool_c=cool_c,
                               wait_max_s=120, verbose=False)
            if ev['action'] != 'ok':
                log['thermal_events'].append({'step': step, **ev})

            if time.time() - t_start > max_wall_s:
                print(f"[21/train] wall budget {max_wall_s}s exceeded at step {step}",
                      flush=True)
                break

            t_pre = temp_c()
            tstep0 = time.time()

            # ---- chip-sig sample ----
            if condition == 'chip_inject':
                v = sig.read()
            elif condition == 'synthetic_matched':
                v = synth_rng.normal(0, 1, size=32).astype(np.float32)
            else:
                v = np.zeros(32, dtype=np.float32)

            # ---- Mechanism 1: LoRA-style chip perturbation into attention weights ----
            if condition != 'vanilla':
                inject_chip_perturbation(attn_mods, v, alpha=alpha_lora)

            # ---- Mechanism 2: per-layer LR modulation ----
            if condition != 'vanilla':
                per_layer_lr(opt, model, v, lr_base, scale=0.20)

            # ---- Mechanism 3: chip-seeded data ordering ----
            if condition != 'vanilla':
                x = make_batch_chip_ordered(train_chunks, batch_size, v, step, base_rng).to(device)
            else:
                idx = base_rng.integers(0, len(train_chunks), size=batch_size)
                X = np.stack([train_chunks[i] for i in idx])
                x = torch.from_numpy(X).long().to(device)

            # forward+backward
            out = model(x, labels=x)
            loss = out.loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            step_times.append(time.time() - tstep0)
            log['losses'].append(float(loss.item()))
            log['sig_norm_log'].append(float(np.linalg.norm(v)))
            log['lr_by_layer_log'].append([float(g['lr']) for g in opt.param_groups])
            log['temp_log'].append(float(t_pre))

            if step % 25 == 0 or step == steps - 1:
                print(f"[21/train] {run_id} step={step:4d} loss={loss.item():.3f} "
                      f"|sig|={np.linalg.norm(v):.2f} T={t_pre:.1f}C "
                      f"dt={step_times[-1]*1000:.0f}ms", flush=True)

            # checkpoint
            if (step + 1) % ckpt_every == 0 or step == steps - 1:
                ck = os.path.join(ckpt_dir, f'step_{step+1}.pt')
                torch.save({'model': model.state_dict(),
                            'step': step + 1,
                            'condition': condition, 'run_id': run_id}, ck)
                log['ckpts'].append({'step': step + 1, 'path': ck})
                print(f"[21/train] ckpt {ck}", flush=True)

            # cool-down
            if (step + 1) % wait_cool_every == 0:
                if temp_c() > wait_cool_target:
                    wait_cool(target_c=wait_cool_target, timeout_s=60)

            log['steps_done'] = step + 1

    except SystemExit as e:
        print(f"[21/train] thermal abort: {e}", flush=True)
        log['thermal_aborted'] = True
    except Exception as e:
        import traceback; traceback.print_exc()
        log['exception'] = f"{type(e).__name__}: {e}"

    log['wall_s'] = time.time() - t_start
    log['mean_step_ms'] = float(np.mean(step_times) * 1000) if step_times else 0.0

    log_path = os.path.join(out_dir, f'train_log_{run_id}.json')
    save_json(log_path, log)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return log


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cond', required=True,
                    choices=['vanilla', 'chip_inject', 'synthetic_matched'])
    ap.add_argument('--run_id', required=True)
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--model', default='distilgpt2')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--alpha', type=float, default=1e-3)
    ap.add_argument('--bsz', type=int, default=2)
    ap.add_argument('--ckpt_every', type=int, default=100)
    ap.add_argument('--out', default='./results')
    ap.add_argument('--abort_c', type=float, default=80)
    ap.add_argument('--pause_c', type=float, default=72)
    ap.add_argument('--cool_c', type=float, default=65)
    ap.add_argument('--max_wall_s', type=int, default=14400)
    args = ap.parse_args()
    train_one(args.cond, args.run_id, args.steps, model_name=args.model,
              lr_base=args.lr, batch_size=args.bsz, alpha_lora=args.alpha,
              ckpt_every=args.ckpt_every, out_dir=args.out,
              thermal_band=(args.abort_c, args.pause_c, args.cool_c),
              max_wall_s=args.max_wall_s)
