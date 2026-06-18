"""G5: Cross-chip divergence.

Same prompts, same model. Compare embodied output distributions across
ikaros vs daedalus.

Pre-reg: KS test p < 0.001 on output-token-distribution.

We aggregate token-IDs and run a KS test on the empirical CDF of token-ID-rank
across all (prompt, rep) outputs. Also report cosine of mean style vectors.
"""
from __future__ import annotations
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import RESULTS, save_json
from g2_classifier import featurize, load_outputs


def ks_2samp(a, b):
    """Two-sample KS test, returns (D, p). Asymptotic p-value."""
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    n1, n2 = len(a), len(b)
    data = np.concatenate([a, b])
    cdf1 = np.searchsorted(a, data, side='right') / n1
    cdf2 = np.searchsorted(b, data, side='right') / n2
    D = float(np.max(np.abs(cdf1 - cdf2)))
    en = np.sqrt(n1 * n2 / (n1 + n2))
    # Kolmogorov asymptotic
    lam = (en + 0.12 + 0.11 / en) * D
    j = np.arange(1, 101)
    p = 2 * np.sum((-1) ** (j - 1) * np.exp(-2 * (lam * j) ** 2))
    p = float(min(max(p, 0.0), 1.0))
    return D, p


def gather_tokens(variant, chip):
    d = load_outputs(chip, variant)
    toks = []
    for s in d['samples']:
        toks.extend(s['token_ids'])
    return np.asarray(toks, dtype=np.int64)


def style_vec(variant, chip):
    d = load_outputs(chip, variant)
    feats = [featurize(s['token_ids']) for s in d['samples']]
    return np.mean(np.stack(feats), axis=0)


def cosine(a, b):
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    out = {'gate_p_lt': 1e-3, 'results': {}}
    for variant in ('vanilla', 'embodied', 'synthetic'):
        a = gather_tokens(variant, 'ikaros')
        b = gather_tokens(variant, 'daedalus')
        D, p = ks_2samp(a, b)
        sa = style_vec(variant, 'ikaros')
        sb = style_vec(variant, 'daedalus')
        cos = cosine(sa, sb)
        out['results'][variant] = {
            'ks_D': D, 'ks_p': p,
            'style_cosine_ikaros_daedalus': cos,
            'n_ikaros_tokens': int(len(a)), 'n_daedalus_tokens': int(len(b)),
        }
        print(f"[G5] {variant:10s}  KS_D={D:.4f}  p={p:.3e}  style_cos={cos:.3f}",
              flush=True)

    emb_p = out['results']['embodied']['ks_p']
    out['pass_embodied_ks_lt_1e-3'] = bool(emb_p < 1e-3)
    out['embodied_more_divergent_than_synth'] = bool(
        out['results']['embodied']['ks_D'] > out['results']['synthetic']['ks_D'])
    print(f"[G5] PASS embodied KS p<1e-3: {out['pass_embodied_ks_lt_1e-3']}", flush=True)
    save_json('g5_divergence.json', out)


if __name__ == '__main__':
    main()
