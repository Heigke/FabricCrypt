"""G1: Personality emergence — generate token sequences across chips × variants.

Single-file, minimal-dependency, foreground-safe. Generates samples for
2 chips × 3 variants × N prompts × R reps.
"""
from __future__ import annotations
import os, sys, time, json, argparse, hashlib
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import numpy as np
import torch
torch.set_num_threads(4)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy'
RESULTS = os.path.join(REPO, 'results/IDENTITY_BENCHMARK_2026-05-30/embodiment17')
os.makedirs(RESULTS, exist_ok=True)
DAEDALUS_SIGS = os.path.join(REPO,
    'results/IDENTITY_BENCHMARK_2026-05-30/embodiment14b/daedalus_sigs.npz')
THERMAL = '/sys/class/thermal/thermal_zone0/temp'


def temp_c():
    try:
        return int(open(THERMAL).read()) / 1000.0
    except Exception:
        return 0.0


def thermal_guard(abort_c=82, pause_c=72, cool_c=60):
    t = temp_c()
    if t >= pause_c:
        print(f"[THERMAL PAUSE] {t:.1f}C cooling to {cool_c}", flush=True)
        t0 = time.time()
        while temp_c() > cool_c:
            if (time.time() - t0) > 180 and temp_c() >= abort_c:
                raise SystemExit(f"[ABORT] {temp_c():.1f}C")
            time.sleep(5)


def sig_to_seed(sig_vec):
    b = np.asarray(sig_vec, dtype=np.float64).tobytes()
    return int.from_bytes(hashlib.sha256(b).digest()[:8], 'little')


def load_prompts():
    with open(os.path.join(HERE, 'prompts.txt')) as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith('#')]


def make_provider(chip, variant, ref_sigs=None):
    """Returns a stateful callable that yields a 32-d vec per call."""
    if variant == 'vanilla':
        seed = abs(hash(('vanilla', chip))) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        def read():
            return rng.normal(size=32).astype(np.float32)
        return read
    if variant == 'embodied':
        if chip == 'ikaros':
            # use LIVE chip signature
            sys.path.insert(0, os.path.join(HERE, '..', 'embodiment14b'))
            sys.path.insert(0, os.path.join(HERE, '..', 'embodiment14'))
            from signature_live import LiveSig
            ls = LiveSig()
            def read():
                return ls.read()
            return read
        elif chip == 'daedalus':
            # use recorded daedalus signatures, cycled
            sigs = np.load(DAEDALUS_SIGS)['sigs'].astype(np.float32)
            idx = [0]
            def read():
                v = sigs[idx[0] % len(sigs)]; idx[0] += 1
                return v
            return read
    if variant == 'synthetic':
        # matched amplitude IID gaussian
        seed = abs(hash(('synthetic', chip))) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        mu = ref_sigs.mean(axis=0)
        sd = ref_sigs.std(axis=0) + 1e-6
        def read():
            return (mu + rng.normal(size=mu.shape) * sd).astype(np.float32)
        return read
    raise ValueError(variant)


def sample_one(model, tok, prompt, max_new_tokens, tau, seed_fn, device='cpu'):
    enc = tok(prompt, return_tensors='pt').to(device)
    input_ids = enc['input_ids']
    gen = []
    with torch.no_grad():
        past = None
        cur = input_ids
        for _ in range(max_new_tokens):
            out = model(cur, past_key_values=past, use_cache=True)
            logits = out.logits[:, -1, :] / max(tau, 1e-6)
            past = out.past_key_values
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            s = seed_fn()
            rng = np.random.default_rng(s & 0xFFFFFFFFFFFFFFFF)
            probs = probs / probs.sum()
            tok_id = int(rng.choice(len(probs), p=probs))
            gen.append(tok_id)
            cur = torch.tensor([[tok_id]], dtype=torch.long, device=device)
            if tok_id == tok.eos_token_id:
                break
    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chips', nargs='+', default=['ikaros', 'daedalus'])
    ap.add_argument('--variants', nargs='+', default=['vanilla', 'embodied', 'synthetic'])
    ap.add_argument('--n-reps', type=int, default=5)
    ap.add_argument('--max-new', type=int, default=15)
    ap.add_argument('--tau', type=float, default=0.8)
    ap.add_argument('--n-prompts', type=int, default=15)
    args = ap.parse_args()

    device = 'cpu'
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print('[G1] loading distilgpt2...', flush=True)
    tok = AutoTokenizer.from_pretrained('distilgpt2')
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained('distilgpt2').to(device).eval()
    for p in mdl.parameters(): p.requires_grad_(False)
    print(f'[G1] model {sum(p.numel() for p in mdl.parameters())/1e6:.1f}M params on {device}',
          flush=True)

    prompts = load_prompts()[:args.n_prompts]
    print(f'[G1] {len(prompts)} prompts × {args.n_reps} reps × {args.max_new} tokens',
          flush=True)
    ref_sigs = np.load(DAEDALUS_SIGS)['sigs']

    for chip in args.chips:
        for variant in args.variants:
            out_path = os.path.join(RESULTS, f'{chip}_{variant}_outputs.json')
            if os.path.exists(out_path):
                print(f'[skip] {out_path}', flush=True); continue
            print(f'\n[G1] === {chip} / {variant} ===', flush=True)
            read = make_provider(chip, variant, ref_sigs=ref_sigs)
            samples = []
            t0 = time.time()
            for rep in range(args.n_reps):
                rt0 = time.time()
                for pi, pr in enumerate(prompts):
                    thermal_guard()
                    def seed_fn(_r=read):
                        return sig_to_seed(_r())
                    ids = sample_one(mdl, tok, pr, args.max_new, args.tau,
                                     seed_fn, device=device)
                    samples.append({'prompt_idx': pi, 'rep': rep, 'token_ids': ids})
                print(f'  rep {rep+1}/{args.n_reps} dt={time.time()-rt0:.1f}s temp={temp_c():.1f}C',
                      flush=True)
            out = {'chip': chip, 'variant': variant, 'tau': args.tau,
                   'max_new_tokens': args.max_new, 'n_reps': args.n_reps,
                   'wall_s': time.time() - t0, 'samples': samples}
            with open(out_path, 'w') as f:
                json.dump(out, f)
            print(f'[save] {out_path}', flush=True)

    print('[G1] done', flush=True)


if __name__ == '__main__':
    main()
