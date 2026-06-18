"""Phase 21 — stylometric analysis & chip-identity classifier.

Reads JSONL outputs from generate.py, computes per-sample feature vector:
  - Token-level: Zipf-alpha (top-1000 token rank vs freq slope)
  - Sentence length: mean, std, skewness
  - POS-lite: adjective/verb ratio (approx via suffix heuristic)
  - Punctuation frequencies: , . ; : ! ? — "
  - Top-50 most-favored tokens (per-class)
  - Bigram transition entropy
  - Avg word length
  - Lexical diversity (TTR)

Then trains:
  - Logistic regression (sklearn) on TF-IDF + handcrafted features
  - K-fold cross-validation accuracy + 1000-iter bootstrap CI
  - Per-class diagnostic: top tokens by log-ratio
"""
from __future__ import annotations
import os, sys, json, argparse, math, re
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


WORD_RE = re.compile(r"[A-Za-z']+")
SENT_RE = re.compile(r"[.!?]+")
PUNCS = [',', '.', ';', ':', '!', '?', '—', '"', "'", '(', ')']


def tokenize_words(text):
    return [w.lower() for w in WORD_RE.findall(text)]


def split_sentences(text):
    parts = SENT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def zipf_alpha(freqs):
    """Fit slope of log(freq) vs log(rank) on top tokens."""
    if len(freqs) < 5:
        return 0.0
    sorted_f = sorted(freqs.values(), reverse=True)[:200]
    sorted_f = [f for f in sorted_f if f > 0]
    if len(sorted_f) < 5:
        return 0.0
    ranks = np.arange(1, len(sorted_f) + 1)
    x = np.log(ranks); y = np.log(sorted_f)
    return float(-np.polyfit(x, y, 1)[0])


def bigram_entropy(words):
    if len(words) < 2:
        return 0.0
    bigs = Counter(zip(words[:-1], words[1:]))
    total = sum(bigs.values())
    H = 0.0
    for c in bigs.values():
        p = c / total
        H -= p * math.log(p + 1e-12)
    return H


def featurize(text):
    words = tokenize_words(text)
    if not words:
        return None
    sents = split_sentences(text)
    sent_lens = [len(tokenize_words(s)) for s in sents] or [0]
    sent_lens = np.asarray(sent_lens, dtype=np.float32)
    word_counts = Counter(words)

    # POS-lite: -ly adverbs, -ing/-ed verbs, -ous/-ful/-al adj
    n_adj = sum(1 for w in words if w.endswith(('ous', 'ful', 'al', 'ive', 'ic', 'ary')))
    n_verb = sum(1 for w in words if w.endswith(('ing', 'ed', 'es')))
    n_adv = sum(1 for w in words if w.endswith('ly'))

    feats = {
        'n_words': len(words),
        'n_sents': len(sents),
        'avg_word_len': float(np.mean([len(w) for w in words])),
        'ttr': len(set(words)) / max(1, len(words)),
        'sent_len_mean': float(sent_lens.mean()),
        'sent_len_std': float(sent_lens.std()),
        'sent_len_max': float(sent_lens.max()),
        'sent_len_min': float(sent_lens.min()),
        'zipf_alpha': zipf_alpha(word_counts),
        'bigram_H': bigram_entropy(words),
        'adj_ratio': n_adj / max(1, len(words)),
        'verb_ratio': n_verb / max(1, len(words)),
        'adv_ratio': n_adv / max(1, len(words)),
        'adj_per_verb': n_adj / max(1, n_verb),
    }
    for p in PUNCS:
        feats[f'punct_{p}'] = text.count(p) / max(1, len(words))
    return feats


def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def build_feature_matrix(records, feat_keys=None):
    X, y = [], []
    rows = []
    for r in records:
        text = r.get('completion') or r.get('text', '')
        if not text or len(text) < 50:
            continue
        f = featurize(text)
        if f is None:
            continue
        rows.append((f, r['label']))
    if not rows:
        return np.zeros((0, 0)), np.array([]), []
    if feat_keys is None:
        feat_keys = sorted(rows[0][0].keys())
    for f, lbl in rows:
        X.append([f[k] for k in feat_keys])
        y.append(lbl)
    return np.asarray(X, dtype=np.float32), np.asarray(y), feat_keys


def top_tokens_by_logratio(records_by_label, k=30, vocab_min=10):
    """Return top-k tokens most over-represented in each class."""
    label_counts = {}
    label_totals = {}
    for lbl, recs in records_by_label.items():
        c = Counter()
        for r in recs:
            text = r.get('completion') or r.get('text', '')
            c.update(tokenize_words(text))
        label_counts[lbl] = c
        label_totals[lbl] = sum(c.values())

    labels = sorted(records_by_label.keys())
    if len(labels) < 2:
        return {}
    out = {}
    # Pairwise top tokens
    for a in labels:
        a_freq = {w: c / label_totals[a] for w, c in label_counts[a].items()}
        rivals = [l for l in labels if l != a]
        rival_total = sum(label_totals[r] for r in rivals)
        rival_counts = Counter()
        for r in rivals:
            rival_counts.update(label_counts[r])
        rival_freq = {w: c / rival_total for w, c in rival_counts.items()}
        # log-ratio
        scored = []
        all_words = set(a_freq.keys()) | set(rival_freq.keys())
        for w in all_words:
            ca = label_counts[a].get(w, 0)
            cb = rival_counts.get(w, 0)
            if ca + cb < vocab_min:
                continue
            fa = (ca + 1) / (label_totals[a] + 1)
            fb = (cb + 1) / (rival_total + 1)
            scored.append((w, math.log(fa / fb), ca, cb))
        scored.sort(key=lambda x: x[1], reverse=True)
        out[a] = scored[:k]
    return out


def classify_kfold(X, y, k=5, seed=0, n_boot=1000):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    fold_accs = []
    all_preds, all_true = [], []
    coefs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        Xt = sc.transform(X[tr]); Xe = sc.transform(X[te])
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced')
        clf.fit(Xt, y[tr])
        pred = clf.predict(Xe)
        acc = (pred == y[te]).mean()
        fold_accs.append(float(acc))
        all_preds.extend(pred.tolist())
        all_true.extend(y[te].tolist())
        coefs.append(clf.coef_)
    rng = np.random.default_rng(seed)
    arr = np.asarray(fold_accs)
    n_boot_acc = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(arr), size=len(arr))
        n_boot_acc.append(arr[idx].mean())
    lo, hi = np.percentile(n_boot_acc, [2.5, 97.5])
    return {
        'fold_accs': fold_accs,
        'mean_acc': float(arr.mean()),
        'ci_lo': float(lo), 'ci_hi': float(hi),
        'preds': all_preds, 'true': all_true,
        'mean_coef': np.mean(np.stack(coefs), axis=0).tolist(),
    }


def analyze(jsonl_paths, out_dir, classes=None):
    os.makedirs(out_dir, exist_ok=True)
    records = []
    for p in jsonl_paths:
        records.extend(load_jsonl(p))
    print(f"[stylometry] loaded {len(records)} records from {len(jsonl_paths)} files")
    if classes:
        records = [r for r in records if r.get('label') in classes]
        print(f"[stylometry] filtered to {len(records)} records in classes {classes}")

    X, y, feat_keys = build_feature_matrix(records)
    print(f"[stylometry] X.shape={X.shape}, classes={sorted(set(y.tolist()))}")
    if X.shape[0] < 20 or len(set(y.tolist())) < 2:
        print("[stylometry] not enough data for classification")
        return None

    cls_result = classify_kfold(X, y, k=5)
    print(f"[stylometry] mean_acc={cls_result['mean_acc']:.3f} "
          f"CI=[{cls_result['ci_lo']:.3f},{cls_result['ci_hi']:.3f}]")

    # Per-class top tokens
    by_label = {}
    for r in records:
        by_label.setdefault(r['label'], []).append(r)
    top_tok = top_tokens_by_logratio(by_label, k=30, vocab_min=5)

    # Per-class mean features
    per_class_means = {}
    for lbl in sorted(set(y.tolist())):
        mask = (y == lbl)
        per_class_means[lbl] = {feat_keys[i]: float(X[mask, i].mean())
                                for i in range(X.shape[1])}

    result = {
        'n_records': len(records),
        'classes': sorted(set(y.tolist())),
        'n_per_class': {lbl: int((y == lbl).sum()) for lbl in sorted(set(y.tolist()))},
        'feat_keys': feat_keys,
        'classifier': cls_result,
        'top_tokens_by_class': {k: [(w, lr, ca, cb) for (w, lr, ca, cb) in v]
                                 for k, v in top_tok.items()},
        'per_class_means': per_class_means,
    }
    out_path = os.path.join(out_dir, 'stylometry_result.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[stylometry] wrote {out_path}")
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', nargs='+', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--classes', nargs='+', default=None,
                    help='filter to subset of labels')
    args = ap.parse_args()
    analyze(args.jsonl, args.out_dir, classes=args.classes)
