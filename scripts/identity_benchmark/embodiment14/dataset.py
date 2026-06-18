"""WikiText-2 loader (wikitext-103 not in cache; wikitext-2 is sufficient
for short-horizon embodiment effect demonstration).

Tokenises with the GPT-2 BPE and produces fixed-length contiguous chunks.
"""
from __future__ import annotations
import os, sys, torch, numpy as np
from torch.utils.data import Dataset, DataLoader

def build_token_arrays(tokenizer, split='train', block_size=128, max_tokens=None):
    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
    # concatenate
    big = []
    for ex in ds:
        t = ex['text']
        if not t: continue
        ids = tokenizer.encode(t)
        if ids:
            big.extend(ids)
            big.append(tokenizer.eos_token_id)
        if max_tokens is not None and len(big) >= max_tokens:
            break
    if max_tokens is not None:
        big = big[:max_tokens]
    arr = np.asarray(big, dtype=np.int64)
    # trim to multiple of block_size
    n = (len(arr) // block_size) * block_size
    return arr[:n].reshape(-1, block_size)


class TokenChunkDataset(Dataset):
    def __init__(self, chunks):
        self.chunks = chunks
    def __len__(self): return len(self.chunks)
    def __getitem__(self, i):
        x = torch.from_numpy(self.chunks[i])
        return {'input_ids': x, 'labels': x.clone()}


def get_loaders(tokenizer, block_size=128, batch_size=4,
                train_max=200_000, eval_max=50_000):
    train_chunks = build_token_arrays(tokenizer, 'train', block_size, train_max)
    eval_chunks  = build_token_arrays(tokenizer, 'test',  block_size, eval_max)
    train_ds = TokenChunkDataset(train_chunks)
    eval_ds  = TokenChunkDataset(eval_chunks)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    eval_ld  = DataLoader(eval_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
    return train_ld, eval_ld, len(train_chunks), len(eval_chunks)
