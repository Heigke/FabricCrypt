"""Phase 17 shared utilities.

Tests the REFRAMED claim: AI outputs from identical weights on different chips
develop measurably different, consistent, clone-resistant behavioral patterns.

THERMAL: strict (abort=68 pause=63 cool=50 per spec)
"""
from __future__ import annotations
import os, sys, time, json, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy'
RESULTS = os.path.join(REPO, 'results/IDENTITY_BENCHMARK_2026-05-30/embodiment17')
os.makedirs(RESULTS, exist_ok=True)

sys.path.insert(0, os.path.join(HERE, '..', 'embodiment14b'))
sys.path.insert(0, os.path.join(HERE, '..', 'embodiment14'))

THERMAL = '/sys/class/thermal/thermal_zone0/temp'


def temp_c():
    try:
        return int(open(THERMAL).read()) / 1000.0
    except Exception:
        return 0.0


def thermal_guard(abort_c=82, pause_c=72, cool_c=60, verbose=False, wait_max_s=180):
    t = temp_c()
    if t >= abort_c:
        # try cooling first
        t0 = time.time()
        while temp_c() > cool_c:
            if (time.time() - t0) > wait_max_s:
                raise SystemExit(f"[THERMAL ABORT] {temp_c():.1f}C >= {abort_c}C after {wait_max_s}s cool")
            time.sleep(5)
        return
    if t >= pause_c:
        if verbose:
            print(f"[THERMAL PAUSE] {t:.1f}C, cooling to {cool_c}", flush=True)
        t0 = time.time()
        while temp_c() > cool_c:
            if (time.time() - t0) > wait_max_s:
                if temp_c() >= abort_c:
                    raise SystemExit(f"[THERMAL ABORT] still {temp_c():.1f}C after {wait_max_s}s")
                break
            time.sleep(5)


def save_json(name, obj):
    path = os.path.join(RESULTS, name)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"[save] {path}")
    return path


def bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n < 2:
        return float(arr.mean()), float(arr.mean()), float(arr.mean())
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(arr.mean()), float(lo), float(hi)


def sig_to_seed(sig_vec):
    """Hash a 32-d signature into a 64-bit RNG seed."""
    b = np.asarray(sig_vec, dtype=np.float64).tobytes()
    h = hashlib.sha256(b).digest()
    return int.from_bytes(h[:8], 'little')


def load_prompts():
    p = os.path.join(HERE, 'prompts.txt')
    with open(p) as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith('#')]


# ---- Tiny LM helpers ----
def load_tiny_lm(device='cpu'):
    """Load distilgpt2 (~82M, but used as fixed sampler). Phase 14b cached.

    Spec says ~2.7M tiny, but we use a frozen pre-trained small model since
    spec also says 'don't train'. distilgpt2 is the smallest cached HF LM
    that produces meaningful text. We use it as a fixed text generator and
    only manipulate the SAMPLING RNG (this is where embodiment enters).
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained('distilgpt2')
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained('distilgpt2').to(device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    return tok, mdl


def sample_one(model, tok, prompt, max_new_tokens, tau, seed_fn, device='cpu'):
    """Generate tokens. seed_fn() -> int seed re-called every step (embodied).

    Returns: list of token ids (the generated continuation only).
    """
    import torch
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
            # numerically stable sample
            probs = probs / probs.sum()
            tok_id = int(rng.choice(len(probs), p=probs))
            gen.append(tok_id)
            cur = torch.tensor([[tok_id]], dtype=torch.long, device=device)
            if tok_id == tok.eos_token_id:
                break
    return gen


# ---- Signature providers ----
class LiveSigProvider:
    """Live chip signature (ikaros real)."""
    def __init__(self):
        from signature_live import LiveSig
        self.sig = LiveSig()

    def read(self):
        return self.sig.read()


class RecordedSigProvider:
    """Cycles through recorded signatures (e.g. real daedalus sigs)."""
    def __init__(self, npz_path, key='sigs'):
        d = np.load(npz_path)
        self.sigs = d[key].astype(np.float32)
        self.idx = 0

    def read(self):
        v = self.sigs[self.idx % len(self.sigs)]
        self.idx += 1
        return v


class SyntheticSigProvider:
    """Matched-amplitude IID Gaussian, no chip information."""
    def __init__(self, ref_sigs=None, seed=12345):
        self.rng = np.random.default_rng(seed)
        if ref_sigs is not None:
            arr = np.asarray(ref_sigs)
            self.mu = arr.mean(axis=0)
            self.sd = arr.std(axis=0) + 1e-6
        else:
            self.mu = np.zeros(32, dtype=np.float32)
            self.sd = np.ones(32, dtype=np.float32)

    def read(self):
        return (self.mu + self.rng.normal(size=self.mu.shape) * self.sd).astype(np.float32)


class FixedPRNGProvider:
    """Vanilla: a single global PRNG. Returns 32-d but only its sequence matters."""
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def read(self):
        return self.rng.normal(size=32).astype(np.float32)
