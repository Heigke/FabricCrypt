"""Phase 18 tiny LM definition + dataset.

4-layer transformer, d_model=128, n_heads=4, mlp_dim=512, vocab=8000.

Why vocab=8000: small to keep parameter count low (~2.7M) and to fit in
2-minute thermal-bounded bursts. We use a BPE-style approximation by
truncating distilgpt2 tokenizer to top-8000 most frequent tokens in our
training subset (with UNK mapping for the rest).
"""
from __future__ import annotations
import os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = 8000
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
MLP_DIM = 512
SEQ_LEN = 128


class TinyTransformerBlock(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, mlp_dim=MLP_DIM, drop=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=drop, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, d_model),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab=VOCAB, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, mlp_dim=MLP_DIM, seq_len=SEQ_LEN, drop=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([
            TinyTransformerBlock(d_model, n_heads, mlp_dim, drop)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        # weight-tie
        self.head.weight = self.tok_emb.weight
        self.drop = nn.Dropout(drop)
        self.register_buffer('mask', torch.triu(
            torch.full((seq_len, seq_len), float('-inf')), diagonal=1), persistent=False)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        h = self.drop(h)
        m = self.mask[:T, :T]
        for blk in self.blocks:
            h = blk(h, m)
        h = self.ln_f(h)
        return self.head(h)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# -------------- Tokenizer (8k subset) --------------
def build_tokenizer_and_data(cache_path, n_train_tokens=100_000, n_val_tokens=10_000):
    """Build (or load) vocab8k tokenizer from distilgpt2 + token streams."""
    if os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        return (d['train_ids'].astype(np.int64),
                d['val_ids'].astype(np.int64),
                d['id_map'].astype(np.int64))
    from transformers import AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained('distilgpt2')
    ds_tr = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    ds_va = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
    text_tr = '\n'.join([t for t in ds_tr['text'] if t.strip()])
    text_va = '\n'.join([t for t in ds_va['text'] if t.strip()])
    ids_tr_full = tok(text_tr, return_tensors='np')['input_ids'][0]
    ids_va_full = tok(text_va, return_tensors='np')['input_ids'][0]
    # frequency on training set
    uniq, cnt = np.unique(ids_tr_full, return_counts=True)
    order = np.argsort(-cnt)
    top = uniq[order][:VOCAB - 1]  # reserve 0 for UNK
    id_map = np.full(tok.vocab_size + 1, 0, dtype=np.int64)  # 0 == UNK
    for new_id, orig in enumerate(top, start=1):
        id_map[orig] = new_id
    train_ids = id_map[ids_tr_full][:n_train_tokens]
    val_ids = id_map[ids_va_full][:n_val_tokens]
    np.savez(cache_path, train_ids=train_ids, val_ids=val_ids, id_map=id_map)
    return train_ids.astype(np.int64), val_ids.astype(np.int64), id_map.astype(np.int64)


def get_batch(ids, batch_size=4, seq_len=SEQ_LEN, rng=None):
    if rng is None:
        rng = np.random
    n = len(ids) - seq_len - 1
    starts = rng.integers(0, n, size=batch_size) if hasattr(rng, 'integers') else rng.randint(0, n, size=batch_size)
    X = np.stack([ids[s:s + seq_len] for s in starts])
    Y = np.stack([ids[s + 1:s + 1 + seq_len] for s in starts])
    return torch.from_numpy(X).long(), torch.from_numpy(Y).long()


def init_model(seed=1337):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = TinyLM()
    return m


if __name__ == '__main__':
    m = init_model()
    print('params:', count_params(m))
    x = torch.randint(0, VOCAB, (2, 32))
    y = m(x)
    print('logits:', y.shape)
