"""Phase 21B — STRICT thermal training (distilgpt2, ONE mechanism).

Differences vs Phase 21:
  - distilgpt2 only (82M)
  - block_size=128 (vs 256), bsz=1 (vs 2)
  - 200 steps max, ckpt every 10
  - PER-STEP thermal_guard (vanilla too — fair comparison)
  - ONLY Mechanism 1: LoRA-style chip-derived attention perturbation
  - Resumable from latest ckpt
  - Thermal band default: 68/62/50

Conditions:
  vanilla  — no injection
  chip     — alpha=1e-3 LoRA-style per-step perturbation on c_attn/c_proj

Usage:
  python train.py --cond chip --run_id chip_dae --steps 200 \
      --out /home/daedalus/embodiment21b/results
"""
from __future__ import annotations
import os, sys, time, json, math, argparse, hashlib, glob
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import (temp_c, thermal_guard, wait_cool, save_json, hostname,
                     LiveSig)


def load_wikitext(tokenizer, block_size=128, max_tokens=10_000):
    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
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
    mods = []
    for name, m in model.named_modules():
        if name.endswith('.attn.c_attn') or name.endswith('.attn.c_proj'):
            mods.append((name, m))
    return mods


def inject_chip_perturbation(attn_mods, sig_vec, alpha=1e-3):
    sig_bytes = np.asarray(sig_vec, dtype=np.float32).tobytes()
    h = hashlib.sha256(sig_bytes).digest()
    seed = int.from_bytes(h[:8], 'little') & 0x7FFFFFFF
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    with torch.no_grad():
        for i, (name, m) in enumerate(attn_mods):
            W = m.weight  # Conv1D (in, out)
            u = torch.randn(W.shape[0], generator=g) * float(sig_vec[i % len(sig_vec)])
            v = torch.randn(W.shape[1], generator=g) * float(sig_vec[(i + 7) % len(sig_vec)])
            u = u.to(W.device, W.dtype)
            v = v.to(W.device, W.dtype)
            un = u / (u.norm() + 1e-6)
            vn = v / (v.norm() + 1e-6)
            chip_mag = float(np.tanh(np.sum(np.asarray(sig_vec[:8]) ** 2) / 4.0))
            delta = alpha * abs(chip_mag + 0.5) * torch.outer(un, vn)
            W.add_(delta)


def find_latest_ckpt(ckpt_dir):
    files = glob.glob(os.path.join(ckpt_dir, 'step_*.pt'))
    if not files:
        return None, 0
    files.sort(key=lambda p: int(os.path.basename(p).replace('step_', '').replace('.pt', '')))
    last = files[-1]
    step = int(os.path.basename(last).replace('step_', '').replace('.pt', ''))
    return last, step


def train_one(condition, run_id, steps, model_name='distilgpt2',
              lr_base=1e-4, batch_size=1, block_size=128,
              ckpt_every=10, alpha_lora=1e-3,
              abort_c=68, pause_c=62, cool_c=50,
              session_max_s=1200, out_dir='./results', max_wall_s=14400,
              resume=True):
    assert condition in ('vanilla', 'chip')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    host = hostname()
    print(f"[21b/train] run={run_id} cond={condition} host={host} model={model_name} "
          f"T={temp_c():.1f}C device={device}", flush=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[21b/train] n_params={n_params/1e6:.1f}M, n_blocks={len(model.transformer.h)}",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr_base, betas=(0.9, 0.95))
    attn_mods = get_attn_modules(model)

    train_chunks = load_wikitext(tok, block_size=block_size, max_tokens=10_000)
    print(f"[21b/train] train_chunks={len(train_chunks)}", flush=True)

    ckpt_dir = os.path.join(out_dir, f'ckpt_{run_id}')
    os.makedirs(ckpt_dir, exist_ok=True)
    start_step = 0
    if resume:
        last_ck, last_step = find_latest_ckpt(ckpt_dir)
        if last_ck:
            sd = torch.load(last_ck, map_location=device, weights_only=False)
            model.load_state_dict(sd['model'])
            if 'optimizer' in sd:
                try:
                    opt.load_state_dict(sd['optimizer'])
                except Exception:
                    pass
            start_step = last_step
            print(f"[21b/train] RESUME from step {start_step} ({last_ck})", flush=True)

    nonce = (run_id.encode() + b'_21b')[:64]
    sig = LiveSig(nonce=nonce) if condition == 'chip' else None
    base_rng = np.random.default_rng(int(hashlib.sha256(run_id.encode()).digest()[:4].hex(), 16))
    # advance rng to match resumed step
    for _ in range(start_step):
        base_rng.integers(0, len(train_chunks), size=batch_size)

    log_path = os.path.join(out_dir, f'train_log_{run_id}.json')
    if resume and os.path.exists(log_path):
        log = json.load(open(log_path))
    else:
        log = {
            'condition': condition, 'run_id': run_id, 'host': host,
            'model_name': model_name, 'lr_base': lr_base,
            'batch_size': batch_size, 'block_size': block_size,
            'target_steps': steps, 'alpha_lora': alpha_lora,
            'n_params': int(n_params),
            'losses': [], 'sig_norm_log': [], 'temp_log': [],
            'thermal_events': [], 'mean_step_ms': 0.0,
            'thermal_aborted': False, 'steps_done': 0, 'wall_s': 0.0,
            'ckpts': [],
        }
    t_start = time.time()
    step_times = []
    session_aborted = False

    try:
        for step in range(start_step, steps):
            # PER-STEP thermal guard
            ev = thermal_guard(abort_c=abort_c, pause_c=pause_c, cool_c=cool_c,
                               wait_max_s=300, verbose=False)
            if ev['action'] == 'abort':
                print(f"[21b/train] THERMAL ABORT at step {step} T={ev.get('t_start'):.1f}C",
                      flush=True)
                log['thermal_events'].append({'step': step, **ev})
                log['thermal_aborted'] = True
                session_aborted = True
                break
            if ev['action'] == 'pause':
                log['thermal_events'].append({'step': step, **ev})

            # Session budget
            if (time.time() - t_start) > session_max_s:
                print(f"[21b/train] SESSION CAP {session_max_s}s reached at step {step}",
                      flush=True)
                break
            if (time.time() - t_start) > max_wall_s:
                break

            t_pre = temp_c()
            tstep0 = time.time()

            if condition == 'chip':
                v = sig.read()
            else:
                v = np.zeros(32, dtype=np.float32)

            if condition == 'chip':
                inject_chip_perturbation(attn_mods, v, alpha=alpha_lora)

            idx = base_rng.integers(0, len(train_chunks), size=batch_size)
            X = np.stack([train_chunks[i] for i in idx])
            x = torch.from_numpy(X).long().to(device)

            out = model(x, labels=x)
            loss = out.loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            step_times.append(time.time() - tstep0)
            log['losses'].append(float(loss.item()))
            log['sig_norm_log'].append(float(np.linalg.norm(v)))
            log['temp_log'].append(float(t_pre))

            if step % 10 == 0 or step == steps - 1:
                print(f"[21b/train] {run_id} step={step:3d}/{steps} loss={loss.item():.3f} "
                      f"|sig|={np.linalg.norm(v):.2f} T={t_pre:.1f}C "
                      f"dt={step_times[-1]*1000:.0f}ms", flush=True)

            # Checkpoint EVERY 10 steps + final
            if (step + 1) % ckpt_every == 0 or step == steps - 1:
                ck = os.path.join(ckpt_dir, f'step_{step+1}.pt')
                torch.save({'model': model.state_dict(),
                            'optimizer': opt.state_dict(),
                            'step': step + 1,
                            'condition': condition, 'run_id': run_id}, ck)
                log['ckpts'].append({'step': step + 1, 'path': ck})

            log['steps_done'] = step + 1

    except SystemExit as e:
        print(f"[21b/train] SystemExit: {e}", flush=True)
        log['thermal_aborted'] = True
    except Exception as e:
        import traceback; traceback.print_exc()
        log['exception'] = f"{type(e).__name__}: {e}"

    log['wall_s'] = (log.get('wall_s') or 0) + (time.time() - t_start)
    log['mean_step_ms'] = float(np.mean(step_times) * 1000) if step_times else 0.0

    save_json(log_path, log)
    print(f"[21b/train] DONE steps_done={log['steps_done']}/{steps} "
          f"aborted={log.get('thermal_aborted')} T={temp_c():.1f}C", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return log, session_aborted


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cond', required=True, choices=['vanilla', 'chip'])
    ap.add_argument('--run_id', required=True)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--model', default='distilgpt2')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--alpha', type=float, default=1e-3)
    ap.add_argument('--bsz', type=int, default=1)
    ap.add_argument('--block_size', type=int, default=128)
    ap.add_argument('--ckpt_every', type=int, default=10)
    ap.add_argument('--out', default='./results')
    ap.add_argument('--abort_c', type=float, default=68)
    ap.add_argument('--pause_c', type=float, default=62)
    ap.add_argument('--cool_c', type=float, default=50)
    ap.add_argument('--session_max_s', type=int, default=1200)
    ap.add_argument('--max_wall_s', type=int, default=14400)
    ap.add_argument('--no_resume', action='store_true')
    args = ap.parse_args()
    train_one(args.cond, args.run_id, args.steps, model_name=args.model,
              lr_base=args.lr, batch_size=args.bsz, block_size=args.block_size,
              alpha_lora=args.alpha, ckpt_every=args.ckpt_every, out_dir=args.out,
              abort_c=args.abort_c, pause_c=args.pause_c, cool_c=args.cool_c,
              session_max_s=args.session_max_s, max_wall_s=args.max_wall_s,
              resume=not args.no_resume)
