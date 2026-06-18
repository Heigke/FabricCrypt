"""Phase 14 Task F — spoof test.

Two phases:

  Phase 1 (on TRAINING host, e.g. ikaros):
      Record 1000 live signature reads, save mean+stddev+raw to .npz.

  Phase 2 (on TARGET host, e.g. daedalus):
      Load the captured ikaros signatures, replay (mean) as the model's
      signature override, compute PPL with replayed-signature.
      Compare to C (transplant w/ live local signature):
         If D ~ C: spoofing FAILS (live read is constitutive).
         If D < C: spoofing SUCCEEDS (signature is just a constant input).

We also try a stronger spoof: re-sample one of the 1000 stored vectors
per forward call (replay-with-variation).

Pre-reg: D ~ C (KS-test p > 0.05 on per-batch loss distributions, AND
   |ppl_D - ppl_C| / ppl_C < 5%).
"""
from __future__ import annotations
import os, sys, json, math, time, argparse
import numpy as np, torch
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common14 import thermal_guard, save_json, hostname, get_apu_temp_c, wait_cool
from signature_io import LiveSignature
from embodied_gpt2 import EmbodiedGPT2, load_tokenizer
from dataset import get_loaders
from capability_test import eval_per_batch, bootstrap_ppl

OUT_DIR = os.path.abspath(os.path.join(HERE, '..', '..', '..',
            'results', 'IDENTITY_BENCHMARK_2026-05-30', 'embodiment14'))


def record_signatures(n=1000, out_path=None, host=None):
    """Phase 1: record n live signatures."""
    sig = LiveSignature(host=host)
    vecs = np.empty((n, sig.DIM), dtype=np.float32)
    print(f"[spoof:record] capturing {n} live signatures on {sig.host}...", flush=True)
    for i in range(n):
        thermal_guard()
        vecs[i] = sig.read()
        time.sleep(0.003)
    if out_path is None:
        out_path = os.path.join(OUT_DIR, f'spoof_capture_{sig.host}.npz')
    np.savez(out_path, vecs=vecs, host=sig.host,
             mu=sig.mu, sigma=sig.sigma)
    print(f"[spoof:record] wrote {out_path} shape={vecs.shape}", flush=True)
    return out_path


class ReplaySignatureWrapper:
    """A drop-in replacement that returns pre-recorded vectors instead of
    live reads. Two modes: 'mean' (constant) or 'sample' (random pick)."""
    DIM = 32
    def __init__(self, npz_path, mode='mean', seed=0):
        d = np.load(npz_path)
        self.vecs = d['vecs']
        self.host = str(d['host'])
        self.mu = d['mu']; self.sigma = d['sigma']
        self.mode = mode
        self.calibrated = True
        self._mean = self.vecs.mean(axis=0).astype(np.float32)
        self._rng = np.random.default_rng(seed)
        self.perm = np.arange(self.DIM)
    def read(self):
        if self.mode == 'mean':
            return self._mean
        # sample
        i = int(self._rng.integers(0, len(self.vecs)))
        return self.vecs[i]
    def read_torch(self, device='cuda', dtype=None):
        import torch
        v = self.read()
        t = torch.from_numpy(v).to(device)
        if dtype is not None: t = t.to(dtype)
        return t


def run_spoof_eval(ckpt_path, capture_npz, n_eval_batches=200, batch_size=4, block_size=128):
    host = hostname()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = load_tokenizer()
    _, eval_ld, _, nev = get_loaders(tok, block_size, batch_size,
                                     train_max=10_000, eval_max=80_000)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"[spoof] local={host} ckpt_host={ck.get('host')} eval chunks={nev}", flush=True)

    # build embodied model. We'll swap signature reader between live & replay.
    sig_live = LiveSignature(calibrate=False)
    sig_live.mu    = np.asarray(ck['sig_mu'],    dtype=np.float32)
    sig_live.sigma = np.asarray(ck['sig_sigma'], dtype=np.float32)
    sig_live.calibrated = True
    embodied = EmbodiedGPT2(sig_reader=sig_live).to(device)
    embodied.mlp_temp.load_state_dict(ck['mlp_temp'])
    embodied.mlp_gamma.load_state_dict(ck['mlp_gamma'])
    embodied.mlp_gain.load_state_dict(ck['mlp_gain'])

    # condition C: live local signature
    c_losses, _ = eval_per_batch(embodied, eval_ld, device, n_eval_batches, label='live_local')
    wait_cool(target_c=55, timeout_s=120)

    # condition D-mean: replay mean
    replay_mean = ReplaySignatureWrapper(capture_npz, mode='mean')
    embodied.sig_reader = replay_mean
    d_mean_losses, _ = eval_per_batch(embodied, eval_ld, device, n_eval_batches, label='replay_mean')
    wait_cool(target_c=55, timeout_s=120)

    # condition D-sample: replay random pick
    replay_samp = ReplaySignatureWrapper(capture_npz, mode='sample')
    embodied.sig_reader = replay_samp
    d_samp_losses, _ = eval_per_batch(embodied, eval_ld, device, n_eval_batches, label='replay_sample')
    del embodied; torch.cuda.empty_cache()

    ppl_C, C_lo, C_hi   = bootstrap_ppl(c_losses)
    ppl_Dm, Dm_lo, Dm_hi = bootstrap_ppl(d_mean_losses)
    ppl_Ds, Ds_lo, Ds_hi = bootstrap_ppl(d_samp_losses)

    ks_mean = stats.ks_2samp(c_losses, d_mean_losses)
    ks_samp = stats.ks_2samp(c_losses, d_samp_losses)

    rel_diff_mean = (ppl_Dm - ppl_C) / ppl_C * 100
    rel_diff_samp = (ppl_Ds - ppl_C) / ppl_C * 100

    out = {
        'local_host': host,
        'ckpt_host':  ck.get('host'),
        'n_eval_batches': n_eval_batches,
        'ppl_C_live_local': {'mean': ppl_C, 'ci95': [C_lo, C_hi]},
        'ppl_D_replay_mean':   {'mean': ppl_Dm, 'ci95': [Dm_lo, Dm_hi]},
        'ppl_D_replay_sample': {'mean': ppl_Ds, 'ci95': [Ds_lo, Ds_hi]},
        'rel_diff_mean_pct':   rel_diff_mean,
        'rel_diff_sample_pct': rel_diff_samp,
        'ks_test_mean_vs_C':    {'stat': float(ks_mean.statistic),
                                 'p':    float(ks_mean.pvalue)},
        'ks_test_sample_vs_C':  {'stat': float(ks_samp.statistic),
                                 'p':    float(ks_samp.pvalue)},
        'pre_reg_spoof_fails': (ks_mean.pvalue > 0.05) and (abs(rel_diff_mean) < 5.0),
        'temp_end_c': get_apu_temp_c(),
    }
    save_json(os.path.join(OUT_DIR, f'spoof_{ck.get("host")}_on_{host}.json'), out)
    print(json.dumps(out, indent=2))
    return out


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'help'
    if cmd == 'record':
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
        record_signatures(n)
    elif cmd == 'eval':
        ckpt = sys.argv[2]; capture = sys.argv[3]
        run_spoof_eval(ckpt, capture, n_eval_batches=int(os.environ.get('NEVAL', '200')))
    else:
        print("usage: spoof_test.py record [n]")
        print("       spoof_test.py eval <ckpt> <capture.npz>")
