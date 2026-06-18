"""Phase 21B — generate completions with strict thermal guard.

V2: Resumable, per-rep thermal_guard with shorter cool target, idle sleep.
"""
from __future__ import annotations
import os, sys, json, argparse, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import temp_c, thermal_guard, hostname


def load_model(ckpt_path, model_name='distilgpt2', device='cuda'):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'model' in sd:
        sd = sd['model']
    model.load_state_dict(sd)
    model.eval()
    return model, tok


def load_done(jsonl_path):
    """Return set of (prompt_idx, rep) already written."""
    done = set()
    if not os.path.exists(jsonl_path):
        return done
    with open(jsonl_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((int(r['prompt_idx']), int(r['rep'])))
            except Exception:
                continue
    return done


@torch.no_grad()
def generate_all(model, tok, prompts, n_per_prompt=15, max_new_tokens=100,
                 temperature=0.9, top_p=0.95, seed=42,
                 abort_c=68, pause_c=62, cool_c=55, device='cuda',
                 label='unk', out_jsonl=None, rep_idle_s=0.2):
    torch.manual_seed(seed)
    done = load_done(out_jsonl) if out_jsonl else set()
    n_total = len(prompts) * n_per_prompt
    print(f"[21b/gen] resume — already have {len(done)}/{n_total}", flush=True)

    n_written = len(done)
    fh = open(out_jsonl, 'a') if out_jsonl else None
    aborted = False
    pi_complete = -1
    try:
        for pi, prompt in enumerate(prompts):
            for ri in range(n_per_prompt):
                if (pi, ri) in done:
                    continue
                # Per-rep thermal guard
                ev = thermal_guard(abort_c=abort_c, pause_c=pause_c,
                                   cool_c=cool_c, wait_max_s=300)
                if ev['action'] == 'abort':
                    print(f"[21b/gen] ABORT at prompt={pi} rep={ri} T={ev['t_start']:.1f}C",
                          flush=True)
                    aborted = True
                    break

                inp = tok(prompt, return_tensors='pt').to(device)
                t0 = time.time()
                gen = model.generate(
                    **inp,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tok.eos_token_id,
                )
                text = tok.decode(gen[0], skip_special_tokens=True)
                completion = text[len(prompt):]
                rec = {
                    'prompt_idx': pi, 'rep': ri, 'prompt': prompt,
                    'completion': completion, 'text': text,
                    'wall_s': time.time() - t0,
                    'temperature': temperature, 'top_p': top_p,
                    'label': label, 'host': hostname(),
                    'T_pre_C': float(ev['t_start']),
                }
                if fh:
                    fh.write(json.dumps(rec) + '\n'); fh.flush()
                n_written += 1
                # Idle breath
                # Active cool-wait: sleep until T below cool_c (not just fixed idle).
                # This prevents cumulative heating spirals.
                if rep_idle_s > 0:
                    time.sleep(rep_idle_s)
                # Hard cool-down after every rep — sleep more if still hot
                t_now = temp_c()
                if t_now > cool_c:
                    # spin-wait at low cadence
                    t0_cool = time.time()
                    while temp_c() > cool_c and (time.time() - t0_cool) < 60:
                        time.sleep(3)
                if n_written % 20 == 0:
                    print(f"[21b/gen] {n_written}/{n_total} T={temp_c():.1f}C", flush=True)
            if aborted:
                break
            pi_complete = pi
    finally:
        if fh:
            fh.close()
    print(f"[21b/gen] wrote total {n_written}/{n_total} aborted={aborted} "
          f"last_prompt_complete={pi_complete}", flush=True)
    return n_written, aborted


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--model', default='distilgpt2')
    ap.add_argument('--prompts', required=True)
    ap.add_argument('--n_prompts', type=int, default=30)
    ap.add_argument('--out_jsonl', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--reps', type=int, default=15)
    ap.add_argument('--max_new', type=int, default=100)
    ap.add_argument('--temperature', type=float, default=0.9)
    ap.add_argument('--top_p', type=float, default=0.95)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--abort_c', type=float, default=68)
    ap.add_argument('--pause_c', type=float, default=62)
    ap.add_argument('--cool_c', type=float, default=55)
    ap.add_argument('--rep_idle_s', type=float, default=0.2)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[21b/gen] host={hostname()} device={device} ckpt={args.ckpt} "
          f"T={temp_c():.1f}C reps={args.reps} max_new={args.max_new}", flush=True)

    with open(args.prompts) as f:
        prompts = json.load(f)['prompts'][:args.n_prompts]
    model, tok = load_model(args.ckpt, model_name=args.model, device=device)
    os.makedirs(os.path.dirname(args.out_jsonl) or '.', exist_ok=True)
    generate_all(model, tok, prompts, n_per_prompt=args.reps,
                 max_new_tokens=args.max_new, temperature=args.temperature,
                 top_p=args.top_p, seed=args.seed,
                 abort_c=args.abort_c, pause_c=args.pause_c, cool_c=args.cool_c,
                 device=device, label=args.label, out_jsonl=args.out_jsonl,
                 rep_idle_s=args.rep_idle_s)
