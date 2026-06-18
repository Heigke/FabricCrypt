"""Partial analysis: compare what we have NOW between any pair of models.

Pairs are inferred from existing .pt files. This is a defensive analysis
that runs regardless of which models trained successfully.

For each pair (A, B):
  - H1: cosine distance between weights
  - H2: KS test on output token distributions
  - H3: text classifier accuracy

Special pair of interest:
  ikaros_vanilla  vs  ikaros_chip_injected
    -> measures whether TRAINING-TIME CHIP INJECTION on the same host
       perturbs the model identifiably (this is the cleanest test of the
       injection mechanism itself; isolates injection from host).
"""
from __future__ import annotations
import os, sys, json, itertools
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import RESULTS, save_json, temp_c, thermal_guard, wait_cool
from tiny_lm import TinyLM, build_tokenizer_and_data, get_batch, VOCAB, SEQ_LEN

CACHE_DATA = os.path.join(RESULTS, 'data_cache.npz')
PROMPTS = [
    "The cat sat on the", "Once upon a time,", "The sky is blue because",
    "In a distant galaxy,", "She opened the door and", "The recipe calls for",
    "He carefully placed the", "The ancient ruins stood",
    "On a cold winter morning,", "The detective examined the",
    "Music filled the air", "Beneath the mountain,",
    "The clock struck midnight", "Sunlight streamed through",
    "The general approached", "Across the river there",
    "Without warning,", "A small voice whispered",
    "The market was bustling", "Through the broken window",
]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def models_available():
    out = {}
    for fn in os.listdir(RESULTS):
        if fn.endswith('.pt') and fn != 'init_checkpoint.pt':
            tag = fn[:-3]
            out[tag] = os.path.join(RESULTS, fn)
    return out


def load_sd(p):
    return torch.load(p, map_location='cpu', weights_only=True)


def cosine_dist_state(sda, sdb):
    total = 0.0; cd = 0.0
    for k in sda:
        if k not in sdb: continue
        a = sda[k].flatten().float()
        b = sdb[k].flatten().float()
        if a.numel() < 2: continue
        cs = float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
        cd += (1 - cs) * a.numel()
        total += a.numel()
    return cd / total if total else 0.0


def tokenize_prompt(prompt, id_map):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('distilgpt2')
    ids = tok(prompt, return_tensors='np')['input_ids'][0]
    return id_map[ids]


def detok_ids(ids, id_map):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('distilgpt2')
    rev = {}
    for orig, new in enumerate(id_map):
        if new != 0 and new not in rev:
            rev[int(new)] = orig
    orig_ids = [rev.get(int(i), tok.unk_token_id or 0) for i in ids]
    return tok.decode(orig_ids, skip_special_tokens=True)


@torch.no_grad()
def generate_one(model, prompt_ids, n_tokens, temperature, seed, device):
    rng = np.random.default_rng(seed)
    cur = torch.tensor(prompt_ids[-SEQ_LEN:], dtype=torch.long, device=device).unsqueeze(0)
    out = []
    for _ in range(n_tokens):
        if cur.shape[1] > SEQ_LEN:
            cur = cur[:, -SEQ_LEN:]
        logits = model(cur)[0, -1, :] / max(temperature, 1e-3)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        probs = probs / probs.sum()
        nid = int(rng.choice(len(probs), p=probs))
        out.append(nid)
        cur = torch.cat([cur, torch.tensor([[nid]], device=device, dtype=torch.long)], dim=1)
    return out


def generate_for_tag(tag, path, id_map, n_per_prompt=5, n_tokens=25, base_seed=12345):
    cache = os.path.join(RESULTS, f"outputs_{tag}.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    print(f"[gen] {tag}", flush=True)
    m = TinyLM().to(DEVICE)
    sd = torch.load(path, map_location=DEVICE, weights_only=True)
    m.load_state_dict(sd); m.eval()
    out = []
    for pi, prm in enumerate(PROMPTS):
        pids = tokenize_prompt(prm, id_map)
        for rep in range(n_per_prompt):
            seed = (base_seed * 1000003 + pi * 37 + rep) & 0xFFFFFFFF
            gen_ids = generate_one(m, pids, n_tokens, 1.0, seed, DEVICE)
            text = detok_ids(gen_ids, id_map)
            out.append({'prompt': prm, 'rep': rep, 'gen_ids': gen_ids, 'text': text, 'seed': seed})
        thermal_guard(abort_c=85, pause_c=78, cool_c=55)
    del m
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    with open(cache, 'w') as f:
        json.dump(out, f)
    print(f"[gen] saved {cache} ({len(out)} samples)", flush=True)
    return out


def ks_test_tokens(a, b):
    from scipy import stats
    a = np.asarray(a); b = np.asarray(b)
    if len(a) < 5 or len(b) < 5:
        return 1.0, 1.0
    D, p = stats.ks_2samp(a, b)
    return float(D), float(p)


def text_classifier_acc(texts_a, texts_b, ids_a=None, ids_b=None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    X = list(texts_a) + list(texts_b)
    y = [0]*len(texts_a) + [1]*len(texts_b)
    if len(set(y)) < 2:
        return {'acc_char': 0.5, 'acc_token': 0.5}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    vec_ch = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2)
    try:
        Xch = vec_ch.fit_transform(X)
        acc_ch = float(cross_val_score(LogisticRegression(max_iter=500, C=1.0),
                                       Xch, y, cv=cv, scoring='accuracy').mean())
    except Exception:
        acc_ch = 0.5
    acc_tk = 0.5
    if ids_a is not None and ids_b is not None:
        Xtk = ([' '.join(map(str, t)) for t in ids_a] + [' '.join(map(str, t)) for t in ids_b])
        vec_tk = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), token_pattern=r"\S+", min_df=1)
        try:
            Xtkv = vec_tk.fit_transform(Xtk)
            acc_tk = float(cross_val_score(LogisticRegression(max_iter=500),
                                           Xtkv, y, cv=cv, scoring='accuracy').mean())
        except Exception:
            acc_tk = 0.5
    return {'acc_char': acc_ch, 'acc_token': acc_tk, 'best': max(acc_ch, acc_tk)}


def main():
    train_ids, val_ids, id_map = build_tokenizer_and_data(CACHE_DATA)
    models = models_available()
    print(f"[available models] {list(models.keys())}", flush=True)

    init_sd = load_sd(os.path.join(RESULTS, 'init_checkpoint.pt'))

    # filter to models that are actually trained: cos_dist_from_init > 1e-4
    trained = {}
    for tag, p in models.items():
        sd = load_sd(p)
        cd = cosine_dist_state(sd, init_sd)
        trained[tag] = {'path': p, 'cd_from_init': cd}
        print(f"  {tag}: cd_from_init={cd:.6f}", flush=True)
    use = {t: v for t, v in trained.items() if v['cd_from_init'] > 1e-4}
    print(f"[trained models w/ cd>1e-4] {list(use.keys())}", flush=True)

    # Generate outputs for each trained model
    outputs = {}
    for tag, info in use.items():
        outputs[tag] = generate_for_tag(tag, info['path'], id_map)
        wait_cool(target_c=60, timeout_s=90)

    # All pairs
    tags = sorted(use.keys())
    pair_results = []
    for a, b in itertools.combinations(tags, 2):
        sda = load_sd(use[a]['path'])
        sdb = load_sd(use[b]['path'])
        cd = cosine_dist_state(sda, sdb)
        toks_a = sum([o['gen_ids'] for o in outputs[a]], [])
        toks_b = sum([o['gen_ids'] for o in outputs[b]], [])
        D, p = ks_test_tokens(toks_a, toks_b)
        clf = text_classifier_acc(
            [o['text'] for o in outputs[a]],
            [o['text'] for o in outputs[b]],
            ids_a=[o['gen_ids'] for o in outputs[a]],
            ids_b=[o['gen_ids'] for o in outputs[b]],
        )
        rec = {
            'pair': (a, b),
            'H1_weight_cos_dist': cd,
            'H1_cd_a_from_init': trained[a]['cd_from_init'],
            'H1_cd_b_from_init': trained[b]['cd_from_init'],
            'H2_ks_D': D, 'H2_ks_p': p,
            'H3_classifier_acc_char': clf['acc_char'],
            'H3_classifier_acc_token': clf['acc_token'],
            'H3_classifier_best': clf['best'],
            'n_tokens_a': len(toks_a),
            'n_tokens_b': len(toks_b),
        }
        pair_results.append(rec)
        print(f"  PAIR ({a}, {b}): cd={cd:.4f} ks_p={p:.2e} clf={clf['best']:.3f}", flush=True)

    # Key comparison summary
    summary = {
        'trained_models': trained,
        'used_models': list(use.keys()),
        'pairs': pair_results,
    }

    # Identify key pair: ikaros_vanilla vs ikaros_chip_injected
    for rec in pair_results:
        if set(rec['pair']) == {'ikaros_vanilla', 'ikaros_chip_injected'}:
            summary['KEY_ikaros_vanilla_vs_chip'] = rec
        if set(rec['pair']) == {'ikaros_vanilla', 'ikaros_synthetic_matched'}:
            summary['KEY_ikaros_vanilla_vs_synth'] = rec
        if set(rec['pair']) == {'ikaros_chip_injected', 'ikaros_synthetic_matched'}:
            summary['KEY_ikaros_chip_vs_synth'] = rec

    # Verdict (partial)
    key = summary.get('KEY_ikaros_vanilla_vs_chip')
    if key:
        chip_cd = key['H1_weight_cos_dist']
        chip_ks_p = key['H2_ks_p']
        chip_acc = key['H3_classifier_best']
        summary['PARTIAL_VERDICT_chip_vs_vanilla'] = {
            'H1_weight_diverged_from_vanilla': bool(chip_cd > 1e-4),
            'H2_outputs_diverged_p_under_001': bool(chip_ks_p < 1e-3),
            'H3_classifier_beats_chance': bool(chip_acc > 0.60),
            'H3_classifier_pre_reg_passes': bool(chip_acc >= 0.80),
            'CHIP_WEIGHT_CD': chip_cd,
            'CHIP_KS_P': chip_ks_p,
            'CHIP_CLF_BEST_ACC': chip_acc,
        }

    save_json('partial_analysis.json', summary)
    print(json.dumps(summary.get('PARTIAL_VERDICT_chip_vs_vanilla'), indent=2))


if __name__ == '__main__':
    main()
