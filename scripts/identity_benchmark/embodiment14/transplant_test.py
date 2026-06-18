"""Phase 14 Task E — transplant test.

Loads the ikaros-trained embodied checkpoint, evaluates on LOCAL host
(should be run on a DIFFERENT machine, e.g. daedalus). On the local host:
   - signature live-read is from a different chip
   - calibration (sig.mu, sig.sigma) loaded from ckpt to AVOID cheating
     (a transplant would not get a fresh calibration on the new chip)

Pre-reg: C PPL >> A_vanilla PPL by >= 30%.
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
from capability_test import eval_per_batch, bootstrap_ppl

OUT_DIR = os.path.abspath(os.path.join(HERE, '..', '..', '..',
            'results', 'IDENTITY_BENCHMARK_2026-05-30', 'embodiment14'))


def main(ckpt_path, n_eval_batches=200, batch_size=4, block_size=128):
    host = hostname()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = load_tokenizer()
    _, eval_ld, _, nev = get_loaders(tok, block_size, batch_size,
                                     train_max=10_000, eval_max=80_000)
    print(f"[transplant] running on host={host} eval chunks={nev}", flush=True)

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"[transplant] ckpt trained on host={ck.get('host')}", flush=True)

    # build sig reader USING THE TRAINING-HOST'S CALIBRATION
    # (so the live values are normalised by mu/sigma fit to the OTHER chip)
    sig = LiveSignature(calibrate=False)
    sig.mu    = np.asarray(ck['sig_mu'],    dtype=np.float32)
    sig.sigma = np.asarray(ck['sig_sigma'], dtype=np.float32)
    sig.calibrated = True

    # build model & load weights
    embodied = EmbodiedGPT2(sig_reader=sig).to(device)
    embodied.mlp_temp.load_state_dict(ck['mlp_temp'])
    embodied.mlp_gamma.load_state_dict(ck['mlp_gamma'])
    embodied.mlp_gain.load_state_dict(ck['mlp_gain'])

    # also vanilla baseline locally
    vanilla = VanillaGPT2().to(device)
    a_losses, _ = eval_per_batch(vanilla, eval_ld, device, n_eval_batches, label='vanilla_local')
    del vanilla; torch.cuda.empty_cache()
    wait_cool(target_c=55, timeout_s=120)

    c_losses, _ = eval_per_batch(embodied, eval_ld, device, n_eval_batches, label='embodied_transplant')
    del embodied; torch.cuda.empty_cache()

    ppl_A, A_lo, A_hi = bootstrap_ppl(a_losses)
    ppl_C, C_lo, C_hi = bootstrap_ppl(c_losses)
    relworse = (ppl_C - ppl_A) / ppl_A * 100

    out = {
        'local_host': host,
        'training_host': ck.get('host'),
        'n_eval_batches': n_eval_batches,
        'ppl_A_vanilla_local': {'mean': ppl_A, 'ci95': [A_lo, A_hi]},
        'ppl_C_embodied_transplanted': {'mean': ppl_C, 'ci95': [C_lo, C_hi]},
        'relative_worsening_pct': relworse,
        'pre_reg_gate_30pct_worse': relworse >= 30.0,
        'temp_end_c': get_apu_temp_c(),
    }
    fname = f'transplant_{ck.get("host")}_to_{host}.json'
    save_json(os.path.join(OUT_DIR, fname), out)
    print(json.dumps(out, indent=2))
    return out


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: transplant_test.py <ckpt_path>"); sys.exit(2)
    main(sys.argv[1], n_eval_batches=int(os.environ.get('NEVAL', '200')))
