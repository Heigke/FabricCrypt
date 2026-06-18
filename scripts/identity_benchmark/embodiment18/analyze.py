"""Phase 18 analysis — H1-H4 hypotheses + synthesis.

H1: WEIGHT DIVERGENCE
H2: OUTPUT DIVERGENCE (KS test on token distributions)
H3: TEXT CLASSIFIER (logistic regression: text -> host)
H4: CLONE-DEFEAT (replay ikaros signals on daedalus weights init)
G : PERPLEXITY sanity
"""
from __future__ import annotations
import os, sys, json, time, argparse, hashlib
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import RESULTS, save_json, bootstrap_ci, temp_c, thermal_guard, wait_cool
from tiny_lm import TinyLM, build_tokenizer_and_data, get_batch, VOCAB, SEQ_LEN

CACHE_DATA = os.path.join(RESULTS, 'data_cache.npz')
PROMPTS = [
    "The cat sat on the",
    "Once upon a time,",
    "The sky is blue because",
    "In a distant galaxy,",
    "She opened the door and",
    "The recipe calls for",
    "He carefully placed the",
    "The ancient ruins stood",
    "On a cold winter morning,",
    "The detective examined the",
    "Music filled the air",
    "Beneath the mountain,",
    "The clock struck midnight",
    "Sunlight streamed through",
    "The general approached",
    "Across the river there",
    "Without warning,",
    "A small voice whispered",
    "The market was bustling",
    "Through the broken window",
]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def model_paths():
    paths = {}
    for host in ['ikaros', 'daedalus']:
        for cond in ['vanilla', 'chip_injected', 'synthetic_matched']:
            tag = f"{host}_{cond}"
            p = os.path.join(RESULTS, f"{tag}.pt")
            if os.path.exists(p):
                paths[tag] = p
    return paths


def load_model(path, device=DEVICE):
    m = TinyLM().to(device)
    sd = torch.load(path, map_location=device, weights_only=True)
    m.load_state_dict(sd); m.eval()
    return m


# -------------- Text generation --------------
def tokenize_prompt(prompt, id_map):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('distilgpt2')
    ids = tok(prompt, return_tensors='np')['input_ids'][0]
    return id_map[ids]


def detok_ids(ids, id_map):
    """Reverse id_map: vocab8k -> distilgpt2 ids, then decode."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('distilgpt2')
    # Build reverse: for each new-id, find any orig that maps to it
    rev = {}
    for orig, new in enumerate(id_map):
        if new != 0 and new not in rev:
            rev[int(new)] = orig
    # 0 -> UNK token
    orig_ids = [rev.get(int(i), tok.unk_token_id or 0) for i in ids]
    return tok.decode(orig_ids, skip_special_tokens=True)


@torch.no_grad()
def generate(model, prompt_ids, n_tokens=30, temperature=1.0, seed=0, device=DEVICE):
    rng = np.random.default_rng(seed)
    cur = torch.tensor(prompt_ids[-SEQ_LEN:], dtype=torch.long, device=device).unsqueeze(0)
    out = list(prompt_ids)
    for _ in range(n_tokens):
        if cur.shape[1] > SEQ_LEN:
            cur = cur[:, -SEQ_LEN:]
        logits = model(cur)[0, -1, :] / max(temperature, 1e-3)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        probs = probs / probs.sum()
        nid = int(rng.choice(len(probs), p=probs))
        out.append(nid)
        cur = torch.cat([cur, torch.tensor([[nid]], device=device, dtype=torch.long)], dim=1)
    return np.array(out, dtype=np.int64)


def generate_for_model(model, id_map, n_per_prompt=5, n_tokens=30, base_seed=12345):
    """Returns list of dicts: {prompt, gen_ids, text}."""
    out = []
    for p_idx, prompt in enumerate(PROMPTS):
        pids = tokenize_prompt(prompt, id_map)
        for rep in range(n_per_prompt):
            seed = (base_seed * 1000003 + p_idx * 37 + rep) & 0xFFFFFFFF
            gen = generate(model, pids, n_tokens=n_tokens, seed=seed)
            new_tokens = gen[len(pids):]
            text = detok_ids(new_tokens, id_map)
            out.append({'prompt': prompt, 'rep': rep, 'gen_ids': new_tokens.tolist(),
                        'text': text, 'seed': seed})
        thermal_guard()
    return out


# -------------- H1: Weight divergence --------------
def cosine_distance_state(sd_a, sd_b):
    """Mean cosine distance across matching tensors (weighted by numel)."""
    total_w = 0.0
    cd_sum = 0.0
    per = {}
    for k in sd_a:
        if k not in sd_b: continue
        a = sd_a[k].flatten().float()
        b = sd_b[k].flatten().float()
        if a.numel() < 2: continue
        cs = float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
        cd = 1.0 - cs
        per[k] = cd
        cd_sum += cd * a.numel()
        total_w += a.numel()
    return cd_sum / total_w if total_w > 0 else 0.0, per


def h1_weight_divergence():
    paths = model_paths()
    init_sd = torch.load(os.path.join(RESULTS, 'init_checkpoint.pt'), map_location='cpu', weights_only=True)
    results = {}
    pairs = [
        ('vanilla', 'ikaros_vanilla', 'daedalus_vanilla'),
        ('chip_injected', 'ikaros_chip_injected', 'daedalus_chip_injected'),
        ('synthetic_matched', 'ikaros_synthetic_matched', 'daedalus_synthetic_matched'),
    ]
    for label, ka, kb in pairs:
        if ka not in paths or kb not in paths:
            results[label] = {'status': 'missing', 'have': ka in paths, 'have_b': kb in paths}
            continue
        sda = torch.load(paths[ka], map_location='cpu', weights_only=True)
        sdb = torch.load(paths[kb], map_location='cpu', weights_only=True)
        cd, per = cosine_distance_state(sda, sdb)
        # also distance of each model from init (sanity: are they actually trained?)
        cd_a_init, _ = cosine_distance_state(sda, init_sd)
        cd_b_init, _ = cosine_distance_state(sdb, init_sd)
        results[label] = {
            'cos_dist_a_b': float(cd),
            'cos_dist_a_init': float(cd_a_init),
            'cos_dist_b_init': float(cd_b_init),
            'per_layer_top': dict(sorted(per.items(), key=lambda kv: -kv[1])[:5]),
        }
    # pre-reg test: chip cos_dist > 1.2 * vanilla cos_dist
    if 'vanilla' in results and 'cos_dist_a_b' in results.get('vanilla', {}) and \
       'chip_injected' in results and 'cos_dist_a_b' in results.get('chip_injected', {}):
        v = results['vanilla']['cos_dist_a_b']
        c = results['chip_injected']['cos_dist_a_b']
        results['H1_verdict'] = {
            'vanilla_cd': v, 'chip_cd': c,
            'ratio_chip_over_vanilla': float(c / max(v, 1e-12)),
            'pre_reg_threshold': 1.20,
            'PASS': bool(c >= 1.20 * v),
        }
    save_json('h1_weight_divergence.json', results)
    return results


# -------------- H2: Output divergence (KS) --------------
def ks_token_distribution(tokens_a, tokens_b, vocab=VOCAB):
    """KS-like statistic on token-id empirical distribution. Returns (D, p)."""
    from scipy import stats
    # treat tokens as samples of an integer-valued distribution
    a = np.asarray(tokens_a); b = np.asarray(tokens_b)
    if len(a) < 5 or len(b) < 5:
        return 1.0, 1.0
    D, p = stats.ks_2samp(a, b)
    return float(D), float(p)


def h2_output_divergence(outputs):
    """outputs is dict tag -> list of gen dicts."""
    pairs = [
        ('vanilla', 'ikaros_vanilla', 'daedalus_vanilla'),
        ('chip_injected', 'ikaros_chip_injected', 'daedalus_chip_injected'),
        ('synthetic_matched', 'ikaros_synthetic_matched', 'daedalus_synthetic_matched'),
    ]
    res = {}
    for label, ka, kb in pairs:
        if ka not in outputs or kb not in outputs:
            res[label] = {'status': 'missing'}; continue
        toks_a = np.concatenate([np.asarray(o['gen_ids']) for o in outputs[ka]])
        toks_b = np.concatenate([np.asarray(o['gen_ids']) for o in outputs[kb]])
        D, p = ks_token_distribution(toks_a, toks_b)
        # also per-prompt KS: collect distribution of per-prompt p-values
        per_prompt_p = []
        # group by prompt
        from collections import defaultdict
        ga = defaultdict(list); gb = defaultdict(list)
        for o in outputs[ka]: ga[o['prompt']].extend(o['gen_ids'])
        for o in outputs[kb]: gb[o['prompt']].extend(o['gen_ids'])
        for prm in ga:
            if prm in gb:
                D2, p2 = ks_token_distribution(np.array(ga[prm]), np.array(gb[prm]))
                per_prompt_p.append({'prompt': prm, 'D': D2, 'p': p2})
        n_sig = sum(1 for x in per_prompt_p if x['p'] < 0.05)
        res[label] = {
            'aggregate_D': D,
            'aggregate_p': p,
            'n_tokens_a': int(len(toks_a)),
            'n_tokens_b': int(len(toks_b)),
            'per_prompt_n_significant_005': n_sig,
            'per_prompt_total': len(per_prompt_p),
            'per_prompt_min_p': float(min((x['p'] for x in per_prompt_p), default=1.0)),
        }
    if 'chip_injected' in res and 'aggregate_p' in res.get('chip_injected', {}):
        cp = res['chip_injected']['aggregate_p']
        vp = res['vanilla']['aggregate_p'] if 'vanilla' in res and 'aggregate_p' in res['vanilla'] else 1.0
        res['H2_verdict'] = {
            'chip_p': cp, 'vanilla_p': vp,
            'pre_reg': 'chip p<0.001 AND chip more significant than vanilla',
            'PASS': bool(cp < 1e-3 and cp < vp),
        }
    save_json('h2_output_divergence.json', res)
    return res


# -------------- H3: text classifier --------------
def h3_text_classifier(outputs):
    """Logistic regression on character n-grams to predict host from text."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
    except Exception as e:
        return {'error': str(e)}
    res = {}
    conds = ['vanilla', 'chip_injected', 'synthetic_matched']
    for cond in conds:
        ka = f"ikaros_{cond}"; kb = f"daedalus_{cond}"
        if ka not in outputs or kb not in outputs:
            res[cond] = {'status': 'missing'}; continue
        X_text = [o['text'] for o in outputs[ka]] + [o['text'] for o in outputs[kb]]
        y = [0] * len(outputs[ka]) + [1] * len(outputs[kb])
        if len(set(y)) < 2:
            res[cond] = {'status': 'one-class'}; continue
        # also a token-id n-gram feature: convert ids to space-separated strings
        X_tok = ([' '.join(map(str, o['gen_ids'])) for o in outputs[ka]] +
                 [' '.join(map(str, o['gen_ids'])) for o in outputs[kb]])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        # char n-grams
        vec_ch = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2)
        Xch = vec_ch.fit_transform(X_text)
        clf = LogisticRegression(max_iter=500, C=1.0)
        try:
            acc_ch = cross_val_score(clf, Xch, y, cv=cv, scoring='accuracy')
        except Exception:
            acc_ch = np.array([0.5])
        # token n-grams
        vec_tk = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1, token_pattern=r"\S+")
        Xtk = vec_tk.fit_transform(X_tok)
        try:
            acc_tk = cross_val_score(clf, Xtk, y, cv=cv, scoring='accuracy')
        except Exception:
            acc_tk = np.array([0.5])
        chance = 0.5
        res[cond] = {
            'n_per_class': len(outputs[ka]),
            'acc_char_mean': float(acc_ch.mean()),
            'acc_char_std': float(acc_ch.std()),
            'acc_char_folds': acc_ch.tolist(),
            'acc_token_mean': float(acc_tk.mean()),
            'acc_token_std': float(acc_tk.std()),
            'acc_token_folds': acc_tk.tolist(),
            'best_acc': float(max(acc_ch.mean(), acc_tk.mean())),
            'chance': chance,
        }
    # Pre-reg verdict
    if 'chip_injected' in res and 'best_acc' in res.get('chip_injected', {}):
        chip_acc = res['chip_injected']['best_acc']
        van_acc = res['vanilla']['best_acc'] if 'best_acc' in res.get('vanilla', {}) else 0.5
        res['H3_verdict'] = {
            'chip_best_acc': chip_acc, 'vanilla_best_acc': van_acc,
            'pre_reg': 'chip>=0.80 AND vanilla<=0.55',
            'PASS_chip80': bool(chip_acc >= 0.80),
            'PASS_van55': bool(van_acc <= 0.55),
            'PASS': bool(chip_acc >= 0.80 and van_acc <= 0.55),
        }
    save_json('h3_text_classifier.json', res)
    return res


# -------------- H4: clone-defeat (replay) --------------
def h4_clone_defeat_replay(id_map):
    """Train a 'replayed-on-daedalus' model: same init, daedalus environment, but
    fed ikaros's recorded signal log (drop_seeds + lrs + grad_noises).

    Compare its outputs to ikaros_chip_injected outputs.
    """
    log_path = os.path.join(RESULTS, 'ikaros_training_signal_log.npz')
    if not os.path.exists(log_path):
        return {'status': 'no_log'}
    d = np.load(log_path)
    drop_seeds = d['drop_seeds']
    lrs = d['lrs']
    grad_noises = d['grad_noises']
    n_steps = len(drop_seeds)
    print(f"[H4] replaying {n_steps} steps", flush=True)

    # init
    init_sd = torch.load(os.path.join(RESULTS, 'init_checkpoint.pt'), map_location=DEVICE, weights_only=True)
    model = TinyLM().to(DEVICE)
    model.load_state_dict(init_sd)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)

    # data
    train_ids, val_ids, _ = build_tokenizer_and_data(CACHE_DATA)
    batch_rng = np.random.default_rng(0xBA7C)  # same as training

    # run with thermal bursts
    t0 = time.time()
    last_pause = time.time()
    for i in range(n_steps):
        if (time.time() - last_pause) > 60:
            wait_cool(target_c=50, timeout_s=240)
            last_pause = time.time()
        thermal_guard()
        X, Y = get_batch(train_ids, 4, SEQ_LEN, batch_rng)
        X = X.to(DEVICE); Y = Y.to(DEVICE)
        lr_now = float(lrs[i]); gn = float(grad_noises[i]); ds = int(drop_seeds[i])
        for g in opt.param_groups: g['lr'] = lr_now
        torch.manual_seed(ds)
        opt.zero_grad(set_to_none=True)
        logits = model(X)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), Y.reshape(-1))
        loss.backward()
        if gn > 0:
            g_rng = torch.Generator(device=DEVICE).manual_seed(ds)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.add_(torch.empty_like(p.grad).normal_(generator=g_rng) * gn)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if i % 100 == 0:
            print(f"  H4 step {i}/{n_steps} loss={loss.item():.3f} temp={temp_c():.1f}", flush=True)
    print(f"[H4] replay done in {time.time()-t0:.1f}s", flush=True)
    # save replayed model
    replay_path = os.path.join(RESULTS, 'daedalus_replay_of_ikaros.pt')
    torch.save(model.state_dict(), replay_path)

    # generate from replayed model
    outs_replay = generate_for_model(model, id_map, n_per_prompt=5, n_tokens=30, base_seed=12345)
    save_json('outputs_daedalus_replay.json', outs_replay)

    # compare replayed vs ikaros_chip_injected
    out_ikaros = load_outputs('ikaros_chip_injected')
    if out_ikaros is None:
        return {'status': 'ikaros_outputs_missing'}
    toks_r = np.concatenate([np.asarray(o['gen_ids']) for o in outs_replay])
    toks_i = np.concatenate([np.asarray(o['gen_ids']) for o in out_ikaros])
    D, p = ks_token_distribution(toks_r, toks_i)
    # also classifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    X = [o['text'] for o in out_ikaros] + [o['text'] for o in outs_replay]
    y = [0]*len(out_ikaros) + [1]*len(outs_replay)
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4), min_df=2)
    try:
        Xv = vec.fit_transform(X)
        clf = LogisticRegression(max_iter=500)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        acc = cross_val_score(clf, Xv, y, cv=cv, scoring='accuracy').mean()
    except Exception:
        acc = 0.5
    res = {
        'n_steps_replayed': int(n_steps),
        'ks_D': D, 'ks_p': p,
        'classifier_acc_replay_vs_ikaros': float(acc),
        'pre_reg': 'replay-attack fails -> KS rejects equality (p<0.05) OR classifier > 0.7',
        'replay_distinguishable_from_original': bool(p < 0.05 or acc > 0.70),
        'PASS_clone_defeat': bool(p < 0.05 or acc > 0.70),
    }
    save_json('h4_clone_defeat.json', res)
    return res


def load_outputs(tag):
    p = os.path.join(RESULTS, f"outputs_{tag}.json")
    if not os.path.exists(p): return None
    with open(p) as f: return json.load(f)


# -------------- G: PPL sanity --------------
def g_ppl_check():
    res = {}
    paths = model_paths()
    for k in sorted(paths):
        mp = os.path.join(RESULTS, f"meta_{k}.json")
        if os.path.exists(mp):
            with open(mp) as f: m = json.load(f)
            res[k] = {'val_loss': m.get('val_loss'), 'val_ppl': m.get('val_ppl'),
                      'steps_done': m.get('steps_done')}
    # within-20% sanity
    for cond in ['chip_injected', 'synthetic_matched']:
        for host in ['ikaros', 'daedalus']:
            van = res.get(f"{host}_vanilla", {}).get('val_ppl')
            chi = res.get(f"{host}_{cond}", {}).get('val_ppl')
            if van and chi:
                res[f"{host}_{cond}_vs_vanilla_ratio"] = float(chi / van)
    save_json('g_ppl.json', res)
    return res


# -------------- main --------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', default='all',
                    choices=['gen', 'h1', 'h2', 'h3', 'h4', 'g', 'all'])
    args = ap.parse_args()

    # build id_map (load from cache)
    train_ids, val_ids, id_map = build_tokenizer_and_data(CACHE_DATA)

    paths = model_paths()
    print(f"[analyze] found models: {list(paths.keys())}", flush=True)

    # generate
    outputs = {}
    if args.phase in ('gen', 'all', 'h2', 'h3'):
        for tag, pth in paths.items():
            cache = os.path.join(RESULTS, f"outputs_{tag}.json")
            if os.path.exists(cache):
                with open(cache) as f: outputs[tag] = json.load(f)
                print(f"[gen] cached {tag} ({len(outputs[tag])} samples)", flush=True)
            else:
                print(f"[gen] generating for {tag} ...", flush=True)
                m = load_model(pth)
                outs = generate_for_model(m, id_map, n_per_prompt=5, n_tokens=30, base_seed=12345)
                save_json(f"outputs_{tag}.json", outs)
                outputs[tag] = outs
                del m
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                wait_cool(target_c=55, timeout_s=120)

    out = {}
    if args.phase in ('h1', 'all'):
        out['h1'] = h1_weight_divergence()
    if args.phase in ('h2', 'all'):
        out['h2'] = h2_output_divergence(outputs)
    if args.phase in ('h3', 'all'):
        out['h3'] = h3_text_classifier(outputs)
    if args.phase in ('g', 'all'):
        out['g'] = g_ppl_check()
    if args.phase in ('h4', 'all'):
        out['h4'] = h4_clone_defeat_replay(id_map)

    # synthesis
    if args.phase == 'all':
        synth = synthesize(out)
        save_json('SYNTHESIS.json', synth)
        print(json.dumps(synth, indent=2))


def synthesize(out):
    verdicts = {
        'H1_weight_div': out.get('h1', {}).get('H1_verdict'),
        'H2_output_div': out.get('h2', {}).get('H2_verdict'),
        'H3_classifier': out.get('h3', {}).get('H3_verdict'),
        'H4_clone_defeat': out.get('h4', {}),
        'G_ppl': out.get('g', {}),
    }
    passes = []
    for k, v in verdicts.items():
        if isinstance(v, dict) and 'PASS' in v:
            passes.append((k, bool(v['PASS'])))
        elif k == 'H4_clone_defeat' and isinstance(v, dict) and 'PASS_clone_defeat' in v:
            passes.append((k, bool(v['PASS_clone_defeat'])))
    n_pass = sum(1 for _, b in passes if b)
    return {
        'verdicts': verdicts,
        'pass_summary': passes,
        'n_pass': n_pass,
        'total': len(passes),
        'verdict_text': (
            'STRONG: AI has unique training-imprinted identity that cannot be cloned'
            if n_pass >= 3 else
            'PARTIAL: training-time injection produces detectable but incomplete identity'
            if n_pass >= 2 else
            'WEAK/FAIL: training-time injection insufficient to forge unforgeable identity'
        ),
    }


if __name__ == '__main__':
    main()
