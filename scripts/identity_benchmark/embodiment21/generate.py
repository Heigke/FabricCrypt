"""Phase 21 — generate long-form completions from trained checkpoints.

For each ckpt: 50 prompts x N completions x 200 tokens.
Saves JSONL with chip+step+temperature metadata.

Optionally injects chip signature into the FIRST forward pass logits (chip-bound
inference). Chip-bound mode is OFF by default since the personality should be
in the weights post Phase 21.
"""
from __future__ import annotations
import os, sys, json, argparse, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import temp_c, thermal_guard, hostname, LiveSig, wait_cool


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


@torch.no_grad()
def generate_completions(model, tok, prompts, n_per_prompt=1, max_new_tokens=200,
                         temperature=0.9, top_p=0.95, seed=42,
                         thermal_band=(80, 72, 65), device='cuda'):
    abort_c, pause_c, cool_c = thermal_band
    out = []
    torch.manual_seed(seed)
    for pi, prompt in enumerate(prompts):
        ev = thermal_guard(abort_c=abort_c, pause_c=pause_c, cool_c=cool_c,
                           wait_max_s=120)
        for ri in range(n_per_prompt):
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
            out.append({
                'prompt_idx': pi, 'rep': ri, 'prompt': prompt,
                'completion': completion, 'text': text,
                'wall_s': time.time() - t0,
                'temperature': temperature, 'top_p': top_p,
            })
        if pi % 5 == 0:
            print(f"[gen] {pi+1}/{len(prompts)} T={temp_c():.1f}C", flush=True)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--model', default='distilgpt2')
    ap.add_argument('--prompts', required=True)
    ap.add_argument('--out_jsonl', required=True)
    ap.add_argument('--label', required=True, help='chip / vanilla / synth identifier')
    ap.add_argument('--step', type=int, default=0)
    ap.add_argument('--reps', type=int, default=2)
    ap.add_argument('--max_new', type=int, default=200)
    ap.add_argument('--temperature', type=float, default=0.9)
    ap.add_argument('--top_p', type=float, default=0.95)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--abort_c', type=float, default=80)
    ap.add_argument('--pause_c', type=float, default=72)
    ap.add_argument('--cool_c', type=float, default=65)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[gen] host={hostname()} device={device} ckpt={args.ckpt}", flush=True)

    with open(args.prompts) as f:
        prompts = json.load(f)['prompts']

    model, tok = load_model(args.ckpt, model_name=args.model, device=device)
    completions = generate_completions(
        model, tok, prompts, n_per_prompt=args.reps,
        max_new_tokens=args.max_new, temperature=args.temperature,
        top_p=args.top_p, seed=args.seed,
        thermal_band=(args.abort_c, args.pause_c, args.cool_c),
        device=device,
    )

    os.makedirs(os.path.dirname(args.out_jsonl) or '.', exist_ok=True)
    with open(args.out_jsonl, 'w') as f:
        for rec in completions:
            rec['label'] = args.label
            rec['step'] = args.step
            rec['ckpt'] = args.ckpt
            rec['host'] = hostname()
            f.write(json.dumps(rec) + '\n')
    print(f"[gen] wrote {len(completions)} to {args.out_jsonl}", flush=True)
