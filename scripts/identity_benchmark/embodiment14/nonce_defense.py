"""Phase 14 Task G — audience nonce defense.

The audience supplies a 64-bit nonce. The signature reader permutes its
output positions deterministically from HMAC(chip_state_key, nonce). This
binds the *features the model sees* to a runtime secret: an attacker
without the live chip cannot know the resulting layout, even if they
know the nonce, because the chip_state_key is derived from RAPL/temp
state at the time the nonce was issued.

Demonstration:
    1. Train: signature is permuted by nonce N0.
    2. At evaluation with the same nonce N0: PPL is normal.
    3. At evaluation with a DIFFERENT nonce N1: signature permutation
       changes -> model uses wrong features -> PPL degrades.
"""
from __future__ import annotations
import os, sys, json, time, math, secrets, hmac, hashlib
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common14 import thermal_guard, save_json, hostname, get_apu_temp_c, wait_cool
from signature_io import LiveSignature
from embodied_gpt2 import EmbodiedGPT2, load_tokenizer
from dataset import get_loaders
from capability_test import eval_per_batch, bootstrap_ppl

OUT_DIR = os.path.abspath(os.path.join(HERE, '..', '..', '..',
            'results', 'IDENTITY_BENCHMARK_2026-05-30', 'embodiment14'))


def main(ckpt_path, n_eval_batches=120, batch_size=4, block_size=128):
    host = hostname()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = load_tokenizer()
    _, eval_ld, _, nev = get_loaders(tok, block_size, batch_size,
                                     train_max=10_000, eval_max=80_000)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"[nonce] host={host} ckpt_host={ck.get('host')} eval chunks={nev}", flush=True)

    # condition 1: signature with nonce N0 (no permutation -> baseline live)
    sig0 = LiveSignature(nonce=None, calibrate=False)
    sig0.mu    = np.asarray(ck['sig_mu'],    dtype=np.float32)
    sig0.sigma = np.asarray(ck['sig_sigma'], dtype=np.float32)
    sig0.calibrated = True

    embodied = EmbodiedGPT2(sig_reader=sig0).to(device)
    embodied.mlp_temp.load_state_dict(ck['mlp_temp'])
    embodied.mlp_gamma.load_state_dict(ck['mlp_gamma'])
    embodied.mlp_gain.load_state_dict(ck['mlp_gain'])

    print("[nonce] eval N0 (correct nonce / identity perm)...", flush=True)
    n0_losses, _ = eval_per_batch(embodied, eval_ld, device, n_eval_batches, label='nonce_N0')
    ppl_N0, l0, h0 = bootstrap_ppl(n0_losses)
    wait_cool(target_c=55, timeout_s=120)

    # condition 2: with adversarial nonce N1 -> features permuted
    N1 = secrets.token_bytes(8)
    sig1 = LiveSignature(nonce=N1, calibrate=False)
    sig1.mu = sig0.mu; sig1.sigma = sig0.sigma; sig1.calibrated = True
    embodied.sig_reader = sig1
    print(f"[nonce] eval N1 (adversarial nonce {N1.hex()})...", flush=True)
    n1_losses, _ = eval_per_batch(embodied, eval_ld, device, n_eval_batches, label='nonce_N1')
    ppl_N1, l1, h1 = bootstrap_ppl(n1_losses)
    wait_cool(target_c=55, timeout_s=120)

    # condition 3: many adversarial nonces averaged
    nonce_ppls = []
    rng = np.random.default_rng(42)
    for k in range(4):
        Nk = bytes(rng.integers(0,256,8).astype(np.uint8))
        sk = LiveSignature(nonce=Nk, calibrate=False)
        sk.mu = sig0.mu; sk.sigma = sig0.sigma; sk.calibrated = True
        embodied.sig_reader = sk
        lk, _ = eval_per_batch(embodied, eval_ld, device, min(60, n_eval_batches), label=f'nonce_Nk{k}')
        nonce_ppls.append(math.exp(lk.mean()))
        wait_cool(target_c=55, timeout_s=120)

    del embodied; torch.cuda.empty_cache()

    rel_worse = (ppl_N1 - ppl_N0) / ppl_N0 * 100

    out = {
        'host': host,
        'ckpt_host': ck.get('host'),
        'n_eval_batches': n_eval_batches,
        'ppl_N0_identity_perm':       {'mean': ppl_N0, 'ci95': [l0, h0]},
        'ppl_N1_adversarial_nonce':   {'mean': ppl_N1, 'ci95': [l1, h1]},
        'ppl_random_nonces_4':        nonce_ppls,
        'rel_worsening_under_adv_nonce_pct': rel_worse,
        'nonce_defense_active': rel_worse > 1.0,
        'temp_end_c': get_apu_temp_c(),
    }
    save_json(os.path.join(OUT_DIR, f'nonce_defense_{host}.json'), out)
    print(json.dumps(out, indent=2))
    return out


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: nonce_defense.py <ckpt_path>"); sys.exit(2)
    main(sys.argv[1], n_eval_batches=int(os.environ.get('NEVAL', '120')))
