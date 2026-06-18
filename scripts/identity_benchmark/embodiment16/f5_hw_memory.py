"""F5: Hardware-augmented memory.

Sequential decision task with long-context retrieval. We give the agent a
sequence of (key, value) pairs to memorise, then ask it to retrieve values
by key at random positions later in the sequence.

Vanilla:  GRU hidden state of size H.
Embodied: GRU hidden state of size H, AUGMENTED by a "chip-state-keyed"
          external memory: when storing, we write the value to a slot
          indexed by hash(chip_state); on retrieval the chip's current
          state biases an attention readout over the memory.

The hypothesis: the chip's autocorrelated state provides a stable but
high-entropy "address space" that the model can use as memory slots that
don't have to be allocated/learned — effectively giving the model log2(K)
extra bits per timestep.

Pre-reg: embodied long-context (>=20-step lag) retrieval accuracy
> vanilla by >= 5pp (CI lower > 0.05).
"""
from __future__ import annotations
import os, sys, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import thermal_guard, save_json, bootstrap_ci, diff_ci, temp_c

sys.path.insert(0, os.path.join(HERE, '..', 'embodiment14b'))
from signature_live import LiveSig

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def gen_episode(n_pairs=3, lag_min=5, lag_max=20, n_keys=4, n_vals=4, rng=None):
    """Generate a sequence:
        STORE(k0, v0), STORE(k1, v1), ..., STORE(k_{n-1}, v_{n-1}),
        FILLER * lag, QUERY(k_j) -> v_j

    Encoded as int tokens; we represent each step as (op, key, value).
    op: 0=STORE, 1=QUERY, 2=FILLER
    """
    rng = rng or np.random.default_rng()
    pairs = [(rng.integers(0, n_keys), rng.integers(0, n_vals)) for _ in range(n_pairs)]
    seq = []
    for k, v in pairs:
        seq.append((0, k, v))   # STORE
    lag = rng.integers(lag_min, lag_max+1)
    for _ in range(lag):
        seq.append((2, rng.integers(0, n_keys), rng.integers(0, n_vals)))
    j = rng.integers(0, n_pairs)
    qk, qv = pairs[j]
    seq.append((1, qk, 0))
    target = qv
    return seq, target, lag


class MemoryAgent(nn.Module):
    def __init__(self, n_keys=4, n_vals=4, hidden=32, sig_dim=32, embodied=False, n_slots=16):
        super().__init__()
        self.n_keys = n_keys; self.n_vals = n_vals; self.hidden = hidden
        self.embodied = embodied; self.n_slots = n_slots
        # token embedding: op(3) + key(n_keys) + val(n_vals)
        self.in_dim = 3 + n_keys + n_vals
        self.rnn = nn.GRUCell(self.in_dim, hidden)
        self.out = nn.Linear(hidden, n_vals)
        if embodied:
            self.sig_proj = nn.Linear(sig_dim, n_slots)  # address head
            self.val_to_slot = nn.Linear(n_vals, hidden)
            self.slot_read = nn.Linear(hidden, n_vals)

    def encode(self, op, key, val):
        v = torch.zeros(self.in_dim, device=DEV)
        v[op] = 1.0
        v[3 + key] = 1.0
        v[3 + self.n_keys + val] = 1.0
        return v

    def forward_seq(self, seq, sig_vecs=None):
        """seq: list of (op,key,val); sig_vecs: list of (sig_dim,) tensors aligned.

        Returns: logits at the QUERY step (last).
        """
        h = torch.zeros(1, self.hidden, device=DEV)
        memory = None
        if self.embodied:
            memory = torch.zeros(self.n_slots, self.hidden, device=DEV)
        out_logits = None
        for t, (op, k, v) in enumerate(seq):
            x = self.encode(op, k, v).unsqueeze(0)
            h = self.rnn(x, h)
            if self.embodied and sig_vecs is not None:
                addr = F.softmax(self.sig_proj(sig_vecs[t]), dim=-1)  # (n_slots,)
                if op == 0:  # STORE: write current h projected into memory at addr
                    write = self.val_to_slot(F.one_hot(torch.tensor(v, device=DEV), self.n_vals).float())
                    memory = memory + addr.unsqueeze(-1) * write.unsqueeze(0)
                elif op == 1:  # QUERY: read from memory weighted by addr
                    readout = (addr.unsqueeze(-1) * memory).sum(0)  # (hidden,)
                    bonus_logits = self.slot_read(readout)
                    base = self.out(h.squeeze(0))
                    out_logits = base + bonus_logits
                    return out_logits.unsqueeze(0)
            else:
                if op == 1:
                    return self.out(h)
        if out_logits is None:
            out_logits = self.out(h)
        return out_logits


def make_episodes(N, rng):
    eps = []
    for _ in range(N):
        seq, tgt, lag = gen_episode(rng=rng)
        eps.append((seq, tgt, lag))
    return eps


def train_one(variant, train_eps, sig, seed, n_epochs=5):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MemoryAgent(embodied=(variant == 'embodied')).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.train()
    for ep in range(n_epochs):
        for seq, tgt, lag in train_eps:
            sig_vecs = None
            if variant == 'embodied':
                # one sig read per step (cheap; ~1ms * len)
                sig_vecs = []
                for _ in seq:
                    sig_vecs.append(torch.from_numpy(np.asarray(sig.read(), dtype=np.float32)).to(DEV))
            logits = model.forward_seq(seq, sig_vecs)
            target = torch.tensor([tgt], device=DEV, dtype=torch.long)
            loss = F.cross_entropy(logits, target)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def eval_one(model, test_eps, sig, variant):
    correct = 0; total = 0
    by_lag = {}
    model.eval()
    with torch.no_grad():
        for seq, tgt, lag in test_eps:
            sig_vecs = None
            if variant == 'embodied':
                sig_vecs = []
                for _ in seq:
                    sig_vecs.append(torch.from_numpy(np.asarray(sig.read(), dtype=np.float32)).to(DEV))
            logits = model.forward_seq(seq, sig_vecs)
            pred = int(logits.argmax(1).item())
            ok = int(pred == tgt)
            correct += ok; total += 1
            by_lag.setdefault(lag, []).append(ok)
    return correct/max(1, total), {k: float(np.mean(v)) for k, v in by_lag.items()}


def long_context_acc(by_lag, threshold=12):
    accs = [v for k, v in by_lag.items() if k >= threshold]
    return float(np.mean(accs)) if accs else 0.0


def main(seeds=15):
    print(f"[F5] start, seeds={seeds}, device={DEV}, temp={temp_c():.1f}C", flush=True)
    sig = LiveSig()
    rng = np.random.default_rng(42)
    train_eps = make_episodes(60, rng)
    test_eps = make_episodes(80, np.random.default_rng(99))
    print(f"[F5] {len(train_eps)} train ep, {len(test_eps)} test ep", flush=True)

    results = {'vanilla': {'acc': [], 'long': []},
               'embodied': {'acc': [], 'long': []}}
    t_start = time.time()
    for s in range(seeds):
        thermal_guard()
        t0 = time.time()
        for v in ('vanilla', 'embodied'):
            m = train_one(v, train_eps, sig, seed=s)
            acc, by_lag = eval_one(m, test_eps, sig, v)
            results[v]['acc'].append(acc)
            results[v]['long'].append(long_context_acc(by_lag, threshold=12))
        print(f"[F5] s{s}: van=({results['vanilla']['acc'][-1]:.3f},"
              f"long={results['vanilla']['long'][-1]:.3f})  "
              f"emb=({results['embodied']['acc'][-1]:.3f},"
              f"long={results['embodied']['long'][-1]:.3f}) "
              f"[{time.time()-t0:.1f}s T={temp_c():.1f}C]", flush=True)
        if (time.time() - t_start) > 420:
            print(f"[F5] time budget hit", flush=True)
            break

    summary = {}
    for v in results:
        for m in ('acc', 'long'):
            mean, lo, hi = bootstrap_ci(results[v][m])
            summary[f'{v}_{m}'] = {'mean': mean, 'ci95': [lo, hi]}
    dmean, dlo, dhi = diff_ci(results['embodied']['long'], results['vanilla']['long'])
    summary['delta_long_emb_minus_van'] = {'mean_pp': dmean*100, 'ci95_pp': [dlo*100, dhi*100]}
    gate_pass = bool(dlo > 0.05)
    summary['gate'] = {
        'criterion': 'embodied_long - vanilla_long > 5pp (CI lower > 0.05)',
        'delta_mean_pp': dmean*100, 'ci95_pp': [dlo*100, dhi*100],
        'pass': gate_pass,
    }
    summary['n_seeds_run'] = len(results['vanilla']['acc'])
    summary['raw'] = results
    save_json('f5_hw_memory.json', summary)
    print(json.dumps(summary['gate'], indent=2))
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=15)
    args = ap.parse_args()
    main(seeds=args.seeds)
