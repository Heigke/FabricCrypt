"""Phase 18B H1-H5 tests over the 6 trained GPT-2 small checkpoints.

H1: Weight divergence (cosine) — within-condition vs between-condition.
H2: Output divergence — KS on token distributions (200 completions per model).
H3: Text classifier — chip-from-output: classify which model produced a text.
H4: Clone-defeat — replay ikarosA chip signals on hostB, check H1+H2 reproduction.
H5: PPL sanity — perplexity on wikitext-2 test.
"""
from __future__ import annotations
import os, sys, time, json, math, hashlib
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import (temp_c, thermal_guard, wait_cool, save_json,
                     bootstrap_ci, RESULTS)

RUN_IDS = [
    'ikarosA_vanilla', 'ikarosA_chip', 'ikarosA_synthmatched',
    'hostB_vanilla',   'hostB_chip',   'hostB_synthmatched',
]


def _load_state(run_id):
    p = os.path.join(RESULTS, f'ckpt_{run_id}', 'final.pt')
    if not os.path.exists(p):
        return None
    d = torch.load(p, map_location='cpu', weights_only=False)
    return d.get('model', d)


def _flatten_state(sd):
    parts = []
    for k in sorted(sd.keys()):
        v = sd[k]
        if torch.is_tensor(v) and v.dtype.is_floating_point:
            parts.append(v.detach().to(torch.float32).reshape(-1).numpy())
    return np.concatenate(parts) if parts else np.zeros(0)


def h1_weight_divergence():
    print("\n[H1] weight divergence (cosine)", flush=True)
    states = {}
    for rid in RUN_IDS:
        sd = _load_state(rid)
        if sd is None:
            print(f"[H1] missing ckpt for {rid}", flush=True)
            continue
        states[rid] = _flatten_state(sd)
        print(f"[H1] {rid}: |w|={states[rid].size:,}", flush=True)

    # Cosine distance against gpt2 baseline (load fresh)
    from transformers import GPT2LMHeadModel
    base = GPT2LMHeadModel.from_pretrained('gpt2')
    base_flat = _flatten_state(base.state_dict())
    del base

    results = {'ckpt_vs_base_cos': {}, 'pairwise_cos': {}}
    base_norm = np.linalg.norm(base_flat) + 1e-12
    for rid, w in states.items():
        # Δ from base
        d = w - base_flat
        dn = np.linalg.norm(d) + 1e-12
        cos_to_base = float(np.dot(w, base_flat) / (np.linalg.norm(w) * base_norm))
        results['ckpt_vs_base_cos'][rid] = {
            'cos_w_base': cos_to_base,
            'delta_norm': float(dn),
            'rel_delta': float(dn / base_norm),
        }
        print(f"[H1] {rid}: cos(w,base)={cos_to_base:.6f} ||Δ||={dn:.3f} "
              f"rel={dn/base_norm:.4f}", flush=True)

    # Pairwise on deltas (more informative than on full weights)
    deltas = {rid: states[rid] - base_flat for rid in states}
    rids = list(deltas.keys())
    for i, a in enumerate(rids):
        for b in rids[i+1:]:
            da, db = deltas[a], deltas[b]
            na, nb = np.linalg.norm(da) + 1e-12, np.linalg.norm(db) + 1e-12
            c = float(np.dot(da, db) / (na * nb))
            results['pairwise_cos'][f'{a}__{b}'] = c

    # Aggregate: chip-chip vs chip-vanilla deltas
    def _cond(rid):
        if rid.endswith('_chip'): return 'chip'
        if rid.endswith('_vanilla'): return 'vanilla'
        if rid.endswith('_synthmatched'): return 'synth'
        return 'other'
    buckets = {}
    for k, c in results['pairwise_cos'].items():
        a, b = k.split('__')
        ca, cb = _cond(a), _cond(b)
        key = '_'.join(sorted([ca, cb]))
        buckets.setdefault(key, []).append(c)
    results['agg'] = {k: float(np.mean(v)) for k, v in buckets.items()}
    save_json('h1_weight_divergence.json', results)
    return results


@torch.no_grad()
def _gen_completions(run_id, prompts, max_new=24, device='cuda', n_per_prompt=1):
    """Generate text completions deterministically (greedy then sample)."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained('gpt2')
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    sd = _load_state(run_id)
    if sd is None:
        return None, None
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.load_state_dict(sd)
    model.eval()
    outs = []
    token_ids_all = []
    for pi, p in enumerate(prompts):
        thermal_guard(abort_c=68, pause_c=63, cool_c=57, wait_max_s=60)
        enc = tok(p, return_tensors='pt').to(device)
        # use sampling for diversity (each model sees same prompt+seed)
        torch.manual_seed(1234 + pi)
        gen = model.generate(**enc, max_new_tokens=max_new,
                             do_sample=True, top_k=50, top_p=0.95,
                             pad_token_id=tok.eos_token_id,
                             num_return_sequences=n_per_prompt)
        for g in gen:
            new_ids = g[enc.input_ids.shape[1]:].tolist()
            token_ids_all.extend(new_ids)
            outs.append(tok.decode(new_ids, skip_special_tokens=True))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outs, token_ids_all


def h2_h5_output_and_ppl(n_prompts=30, n_per=2, max_new=20):
    """Generate completions and compute PPL on wikitext-2 test."""
    print("\n[H2/H5] output divergence + PPL", flush=True)
    prompts = [
        "The quick brown fox", "In the beginning", "The President said",
        "Once upon a time", "Scientists have discovered", "It is well known that",
        "After many years", "The committee decided", "On the morning of",
        "Despite the warnings", "A new study shows", "The future of",
        "When asked about", "The team worked", "Critics argue that",
        "Throughout history", "By the end of", "Recent reports indicate",
        "The economy is", "Many people believe", "The author writes",
        "In other news", "A statement released", "The first time",
        "Some researchers say", "In an interview", "The data suggests",
        "Last month", "Officials confirmed", "It remains unclear",
    ][:n_prompts]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    completions = {}
    token_ids = {}
    for rid in RUN_IDS:
        wait_cool(target_c=58, timeout_s=90)
        print(f"[H2] generating for {rid} (T={temp_c():.1f}C)", flush=True)
        outs, ids = _gen_completions(rid, prompts, max_new=max_new,
                                     device=device, n_per_prompt=n_per)
        if outs is None:
            print(f"[H2] skipping {rid} (no ckpt)", flush=True)
            continue
        completions[rid] = outs
        token_ids[rid] = ids
        print(f"[H2] {rid}: {len(outs)} completions, {len(ids)} tokens", flush=True)
        # Sample:
        if outs:
            print(f"     sample: {outs[0][:80]!r}", flush=True)

    # H2 metric: KS-style distance on token-id distributions
    from scipy.stats import ks_2samp
    h2 = {'pairwise_ks': {}, 'pairwise_js': {}}
    rids = list(token_ids.keys())
    for i, a in enumerate(rids):
        for b in rids[i+1:]:
            ka = np.asarray(token_ids[a])
            kb = np.asarray(token_ids[b])
            if ka.size == 0 or kb.size == 0:
                continue
            try:
                stat, pval = ks_2samp(ka, kb)
            except Exception:
                stat, pval = float('nan'), float('nan')
            h2['pairwise_ks'][f'{a}__{b}'] = {'stat': float(stat),
                                              'p': float(pval)}
            # JS divergence on token frequency over vocab
            V = 50257
            pa = np.bincount(ka, minlength=V).astype(np.float64) + 1e-6
            pb = np.bincount(kb, minlength=V).astype(np.float64) + 1e-6
            pa /= pa.sum(); pb /= pb.sum()
            m = 0.5 * (pa + pb)
            js = 0.5 * (np.sum(pa * np.log(pa / m)) +
                        np.sum(pb * np.log(pb / m)))
            h2['pairwise_js'][f'{a}__{b}'] = float(js)
    save_json('h2_output_divergence.json', h2)

    # H5: PPL on wikitext-2 test (first ~5k tokens)
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from datasets import load_dataset
    tok = GPT2Tokenizer.from_pretrained('gpt2')
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    txt = '\n'.join([t for t in ds['text'] if t.strip()])
    ids = tok(txt, return_tensors='pt').input_ids[0][:4096]
    h5 = {}
    block = 256
    for rid in RUN_IDS:
        sd = _load_state(rid)
        if sd is None:
            continue
        wait_cool(target_c=58, timeout_s=90)
        m = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
        m.load_state_dict(sd)
        m.eval()
        losses = []
        with torch.no_grad():
            for off in range(0, ids.size(0) - block, block):
                thermal_guard(abort_c=68, pause_c=63, cool_c=57, wait_max_s=60)
                x = ids[off:off+block].unsqueeze(0).to(device)
                out = m(x, labels=x)
                losses.append(float(out.loss.item()))
        avg = float(np.mean(losses)) if losses else float('nan')
        ppl = math.exp(avg) if not math.isnan(avg) else float('nan')
        h5[rid] = {'avg_nll': avg, 'ppl': ppl, 'n_blocks': len(losses)}
        print(f"[H5] {rid}: avg_nll={avg:.4f} ppl={ppl:.2f}", flush=True)
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    save_json('h5_ppl.json', h5)

    # Persist completions for inspection
    save_json('completions.json', completions)
    return h2, h5, completions, token_ids


def h3_text_classifier(token_ids):
    """Train classifier: which of the 6 models produced this completion?
    Features: vocab-bag (top 1000 most common tokens). Logistic regression.
    """
    print("\n[H3] text classifier", flush=True)
    if not token_ids or len(token_ids) < 2:
        save_json('h3_classifier.json', {'error': 'too few runs'})
        return {}
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    # Each "sample" = one completion = one prompt's generated token sequence
    # We need to split completions per run. We have n_prompts * n_per per run.
    # token_ids[rid] is a flat list of generated tokens across all completions.
    # We must re-split: assume each completion has exactly max_new tokens.
    # Recover via stored completions JSON instead.
    comp_path = os.path.join(RESULTS, 'completions.json')
    completions = json.load(open(comp_path))

    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained('gpt2')

    samples = []  # list of (token_id_list, label)
    label_map = {rid: i for i, rid in enumerate(sorted(completions.keys()))}
    for rid, comps in completions.items():
        for c in comps:
            ids = tok(c)['input_ids']
            samples.append((ids, label_map[rid]))

    if len(samples) < 12:
        save_json('h3_classifier.json', {'error': f'only {len(samples)} samples'})
        return {}

    # Feature: bag over top-K tokens
    K = 2000
    flat = []
    for ids, _ in samples:
        flat.extend(ids)
    vc = np.bincount(np.asarray(flat), minlength=50257)
    top_tok = np.argsort(-vc)[:K]
    tok2col = {int(t): i for i, t in enumerate(top_tok)}

    X = np.zeros((len(samples), K), dtype=np.float32)
    y = np.zeros(len(samples), dtype=np.int64)
    for i, (ids, lab) in enumerate(samples):
        for t in ids:
            if t in tok2col:
                X[i, tok2col[t]] += 1
        y[i] = lab

    # Stratified k-fold accuracy
    n_classes = len(label_map)
    accs = []
    kf = StratifiedKFold(n_splits=min(5, np.bincount(y).min() if len(np.bincount(y)) > 0 else 2),
                          shuffle=True, random_state=42)
    try:
        for tr, te in kf.split(X, y):
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(X[tr], y[tr])
            accs.append(float(clf.score(X[te], y[te])))
    except Exception as e:
        accs = []
        print(f"[H3] KF failed: {e}", flush=True)

    chance = 1.0 / n_classes
    mean_acc, lo, hi = (float(np.mean(accs)), float(np.min(accs)), float(np.max(accs))) \
                       if accs else (float('nan'), float('nan'), float('nan'))
    # Also: chip vs vanilla binary (collapsing across runs)
    inv_label_map = {v: k for k, v in label_map.items()}
    def _is_chip(lab):
        rid = inv_label_map[int(lab)]
        # only the run_ids ending in '_chip' are chip_injected
        return rid.endswith('_chip')
    bin_y = np.array([1 if _is_chip(lab) else 0 for lab in y])
    bin_accs = []
    if len(np.unique(bin_y)) == 2:
        for tr, te in StratifiedKFold(n_splits=5, shuffle=True,
                                      random_state=42).split(X, bin_y):
            clf = LogisticRegression(max_iter=2000)
            clf.fit(X[tr], bin_y[tr])
            bin_accs.append(float(clf.score(X[te], bin_y[te])))

    res = {
        'n_samples': len(samples), 'n_classes': n_classes, 'chance': chance,
        'multiclass_fold_accs': accs,
        'multiclass_mean_acc': mean_acc,
        'multiclass_acc_min': lo, 'multiclass_acc_max': hi,
        'binary_chip_vs_nonchip_fold_accs': bin_accs,
        'binary_chip_vs_nonchip_mean_acc': float(np.mean(bin_accs)) if bin_accs else float('nan'),
        'label_map': label_map,
    }
    save_json('h3_classifier.json', res)
    print(f"[H3] multiclass {n_classes}-way: mean={mean_acc:.3f} (chance={chance:.3f})",
          flush=True)
    print(f"[H3] binary chip-vs-nonchip: mean={res['binary_chip_vs_nonchip_mean_acc']:.3f}",
          flush=True)
    return res


def h4_clone_defeat(h1_res, h2_res):
    """H4: Clone defeat. Can a replay attacker reproduce ikarosA_chip's
    behaviour without access to ikaros's chip?

    Real test would: take ikarosA chip signals, replay on hostB hardware,
    train hostB with replayed signal, compare to ikarosA_chip. Since we
    don't have a second physical host (daedalus unreachable), we test
    the weaker property: does synthetic_matched (which sees deterministic
    noise of matching stats) reproduce the chip_injected delta?

    Metric: cos(Δ_ikarosA_chip, Δ_ikarosA_synthmatched) vs
            cos(Δ_ikarosA_chip, Δ_hostB_chip).
    If clone-defeat holds: chip-chip cos > synth-chip cos.
    """
    print("\n[H4] clone defeat", flush=True)
    p = h1_res.get('pairwise_cos', {})
    res = {'pairs_used': [], 'cos': {}, 'caveats': []}

    def get(a, b):
        return p.get(f'{a}__{b}', p.get(f'{b}__{a}'))

    # chip-chip (different hosts, both real chip)
    c_chip_chip = get('ikarosA_chip', 'hostB_chip')
    # chip-vs-synth_matched (same host)
    c_chip_synth_A = get('ikarosA_chip', 'ikarosA_synthmatched')
    c_chip_synth_B = get('hostB_chip', 'hostB_synthmatched')
    # vanilla pair as null
    c_van_van = get('ikarosA_vanilla', 'hostB_vanilla')
    res['cos'] = {
        'chip_chip_ikarosA_hostB': c_chip_chip,
        'chip_vs_synthmatched_ikarosA': c_chip_synth_A,
        'chip_vs_synthmatched_hostB': c_chip_synth_B,
        'vanilla_vanilla_ikarosA_hostB': c_van_van,
    }
    if c_chip_chip is not None and c_chip_synth_A is not None:
        # Clone-defeat = chip-chip pairs are MORE similar than chip-vs-synthmatched
        # Larger = clone defeat WEAKER (chip and synth are similar => no chip-specific signal)
        res['clone_defeat_score'] = float(c_chip_chip - c_chip_synth_A)
        res['interpretation'] = (
            'positive = chip-trained models more similar to each other than to '
            'synthetic-noise models (chip-bound identity signal exists). '
            'Negative = chip is indistinguishable from generic noise.'
        )
    res['caveats'].append(
        'Daedalus host unreachable; hostB runs were performed on ikaros with '
        'a different nonce permutation. True cross-chip clone defeat NOT tested.'
    )
    save_json('h4_clone_defeat.json', res)
    print(f"[H4] cos(chip_A, chip_B)={c_chip_chip} "
          f"cos(chip_A, synth_A)={c_chip_synth_A} "
          f"cos(van_A, van_B)={c_van_van}", flush=True)
    return res


def main():
    print(f"[18B/tests] start temp={temp_c():.1f}C", flush=True)
    wait_cool(target_c=58, timeout_s=90)

    h1 = h1_weight_divergence()
    wait_cool(target_c=58, timeout_s=90)

    h2, h5, completions, token_ids = h2_h5_output_and_ppl()
    wait_cool(target_c=58, timeout_s=90)

    h3 = h3_text_classifier(token_ids)
    h4 = h4_clone_defeat(h1, h2)

    summary = {
        'H1': h1.get('agg', {}),
        'H2_mean_js_chip_pair': None,
        'H3_multiclass_mean_acc': h3.get('multiclass_mean_acc'),
        'H3_binary_chip_acc': h3.get('binary_chip_vs_nonchip_mean_acc'),
        'H3_n_classes': h3.get('n_classes'),
        'H3_chance': h3.get('chance'),
        'H4_clone_defeat_score': h4.get('clone_defeat_score'),
        'H5_ppl': {rid: r.get('ppl') for rid, r in h5.items()},
        'verdict': {},
    }
    h3_acc = h3.get('multiclass_mean_acc')
    if h3_acc is not None and not math.isnan(h3_acc):
        n_cls = h3.get('n_classes', 6)
        chance = 1.0 / n_cls
        summary['verdict']['H3_above_chance'] = h3_acc > 2 * chance
        summary['verdict']['H3_above_80pct'] = h3_acc >= 0.80
    save_json('SUMMARY.json', summary)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
