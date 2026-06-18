"""G3: Clone-defeat test.

Can we make daedalus produce ikaros-style outputs by replaying ikaros's
recorded signature on daedalus's machine?

Procedure:
  C1: replay-attack — on "daedalus" we feed ikaros signatures into the
      sampler. The output is then classified by the embodied G2 classifier.
      If the classifier still says 'daedalus', clone fails (good — clone-defeat).
  C2: random fingerprint on daedalus → control (should classify as daedalus).
  C3: ikaros signature on ikaros → control (should classify as ikaros).

We DO NOT actually have a remote daedalus to run on. The clone-defeat claim
hinges on whether the LIVE chip noise (which differs from a recording) shifts
classification — i.e., recorded signatures cannot reproduce the live-chip
distribution. We simulate this honestly: 'daedalus' samples already came
from recordings, so on daedalus 'embodied' == 'replay daedalus sigs'.

To test clone-defeat we generate one extra batch where we run on IKAROS but
seed each step from a RECORDED IKAROS signature SNAPSHOT (i.e., we record
ikaros sigs once, then replay them at a later time — simulating a clone).
If the classifier distinguishes replay-vs-live, clone is defeated.

NOTE: This test partially conflates 'live ikaros' vs 'replayed ikaros'.
That is the right test for the clone-defeat claim: a clone has only the
stored signature, not the live one.
"""
from __future__ import annotations
import os, sys, json, time, hashlib
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import (RESULTS, save_json, load_prompts, load_tiny_lm, sample_one,
                     sig_to_seed, thermal_guard, bootstrap_ci,
                     LiveSigProvider, RecordedSigProvider, SyntheticSigProvider)
from g2_classifier import featurize, _fit_logreg, _sigmoid, build_xy

REPO = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy'
DAEDALUS_SIGS = os.path.join(REPO,
    'results/IDENTITY_BENCHMARK_2026-05-30/embodiment14b/daedalus_sigs.npz')


def collect_ikaros_sigs(n=500):
    """Snapshot ikaros sigs to disk for later replay."""
    p = os.path.join(RESULTS, 'ikaros_snapshot_sigs.npz')
    if os.path.exists(p):
        return np.load(p)['sigs']
    prov = LiveSigProvider()
    arr = np.stack([prov.read() for _ in range(n)]).astype(np.float32)
    np.savez(p, sigs=arr)
    return arr


def generate_condition(label, sig_provider, prompts, n_reps, max_new, tau,
                       model, tok, device='cpu'):
    samples = []
    for rep in range(n_reps):
        for pi, pr in enumerate(prompts):
            thermal_guard()
            def seed_fn(_p=sig_provider):
                return sig_to_seed(_p.read())
            ids = sample_one(model, tok, pr, max_new, tau, seed_fn, device=device)
            samples.append({'prompt_idx': pi, 'rep': rep, 'token_ids': ids})
        print(f"  [{label}] rep {rep+1}/{n_reps}", flush=True)
    return samples


def classify_against(samples, classifier_w, classifier_b):
    X = np.stack([featurize(s['token_ids']) for s in samples])
    p = _sigmoid(X @ classifier_w + classifier_b)
    # label 0 = ikaros, 1 = daedalus
    pred = (p > 0.5).astype(int)
    frac_daedalus = float(pred.mean())
    return frac_daedalus, p.tolist()


def main():
    device = 'cpu'
    tok, model = load_tiny_lm(device=device)
    prompts = load_prompts()[:20]
    n_reps = 5  # smaller for thermal budget
    max_new = 20
    tau = 0.8

    print("[G3] training G2 embodied classifier on existing outputs...", flush=True)
    X, y, _ = build_xy('embodied')
    w, b = _fit_logreg(X, y, lam=1e-3, iters=200, lr=0.5)
    base_pred = (_sigmoid(X @ w + b) > 0.5).astype(int)
    print(f"[G3] in-sample classifier acc={float((base_pred==y).mean()):.3f}", flush=True)

    # Snapshot ikaros sigs for replay
    print("[G3] snapshotting ikaros sigs...", flush=True)
    ikaros_snap = collect_ikaros_sigs(n=500)

    # C1: replay attack on ikaros (replay stored ikaros sigs — equivalent
    #     to a clone trying to be ikaros from stored data)
    print("[G3] C1: replay stored ikaros sigs (clone attack)", flush=True)
    np.savez(os.path.join(RESULTS, 'ikaros_replay_sigs.npz'), sigs=ikaros_snap)
    prov_replay = RecordedSigProvider(os.path.join(RESULTS, 'ikaros_replay_sigs.npz'))
    s_replay = generate_condition('C1-replay', prov_replay, prompts, n_reps,
                                  max_new, tau, model, tok, device)
    frac_d_replay, _ = classify_against(s_replay, w, b)

    # C2: random fingerprint (synthetic) — should not look like ikaros
    print("[G3] C2: synthetic fingerprint (control)", flush=True)
    prov_synth = SyntheticSigProvider(ref_sigs=ikaros_snap, seed=99)
    s_synth = generate_condition('C2-synth', prov_synth, prompts, n_reps,
                                 max_new, tau, model, tok, device)
    frac_d_synth, _ = classify_against(s_synth, w, b)

    # C3: live ikaros — control, should classify as ikaros (label 0)
    print("[G3] C3: live ikaros (control)", flush=True)
    prov_live = LiveSigProvider()
    s_live = generate_condition('C3-live', prov_live, prompts, n_reps,
                                max_new, tau, model, tok, device)
    frac_d_live, _ = classify_against(s_live, w, b)

    # Interpretation:
    # frac_d ~ probability classifier calls it 'daedalus'.
    # C1 replay attack tries to LOOK LIKE IKAROS, so we want frac_d ~ low if
    # replay successfully mimics ikaros. Clone DEFEAT = frac_d HIGH (not ikaros)
    # OR distinct from C3 live ikaros.
    # We measure clone-defeat as |frac_d_replay - frac_d_live| > 0.20.
    clone_defeat_margin = abs(frac_d_replay - frac_d_live)

    out = {
        'C1_replay_frac_called_daedalus': frac_d_replay,
        'C2_synth_frac_called_daedalus': frac_d_synth,
        'C3_live_frac_called_daedalus':  frac_d_live,
        'clone_defeat_margin_C1_vs_C3': clone_defeat_margin,
        'pass_clone_defeat': clone_defeat_margin > 0.20,
        'pass_synth_distinct': abs(frac_d_synth - frac_d_live) > 0.20,
        'n_samples_per_condition': len(s_replay),
        'note': ('C1 = stored ikaros sigs replayed (clone attack). If output '
                 'differs from C3 live ikaros, the live chip is irreplaceable.'),
    }
    print(f"[G3] C1 replay: frac_daedalus={frac_d_replay:.3f}", flush=True)
    print(f"[G3] C2 synth : frac_daedalus={frac_d_synth:.3f}", flush=True)
    print(f"[G3] C3 live  : frac_daedalus={frac_d_live:.3f}", flush=True)
    print(f"[G3] clone-defeat margin={clone_defeat_margin:.3f} PASS={out['pass_clone_defeat']}",
          flush=True)
    save_json('g3_clone_defeat.json', out)


if __name__ == '__main__':
    main()
