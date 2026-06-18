"""Phase 14B Task E — spoof defense.

For each task, evaluate degradation when:
  A. Replay attack: train embodied; eval with stored sigs from another time/host.
  B. Random sig: eval with N(0,1) noise.
  C. Nonce mismatch: train with nonce_A; eval with nonce_B (permutation differs).

We re-use the trained T2 and T3 classifiers since they directly read sig.
For T1/T4 we train fresh quickly and then test spoofs.
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
P14 = os.path.abspath(os.path.join(HERE, '..', 'embodiment14'))
sys.path.insert(0, P14)
P13 = os.path.abspath(os.path.join(HERE, '..', 'embodiment13'))
sys.path.insert(0, P13)

from common13 import thermal_guard, hostname, save_json
from signature_live import LiveSig
from embodied_tiny import EmbodiedTiny
import tasks as T
from train_and_eval import (set_seed, get_sig_for_batch, get_static_sig, auroc)


def train_T1_quick(device, sig, epochs=20, seed=0):
    inputs, y, sigs, meta = T.build_T1_dataset(n_per_expr=3, thermal_guard_fn=thermal_guard, sig=sig)
    idx = np.arange(len(inputs)); np.random.shuffle(idx)
    split = int(0.8*len(inputs))
    tr, te = idx[:split], idx[split:]
    set_seed(seed)
    model = EmbodiedTiny(embodied=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for ep in range(epochs):
        order = np.random.permutation(len(tr))
        for i in range(0, len(order), 32):
            b = tr[order[i:i+32]]
            xb = torch.from_numpy(inputs[b]).to(device)
            yb = torch.from_numpy(y[b]).to(device)
            sb = torch.from_numpy(sigs[b]).to(device)
            pred = model(xb, sb, head='reg')
            loss = F.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return model, (inputs[te], y[te])


def eval_T1(model, device, Xte, Yte, sig_provider, n_repeats=3):
    """sig_provider: callable(B)->np.ndarray(B,32)"""
    mses = []
    for _ in range(n_repeats):
        with torch.no_grad():
            xb = torch.from_numpy(Xte).to(device)
            yb = torch.from_numpy(Yte).to(device)
            sv = sig_provider(len(Xte))
            sb = torch.from_numpy(sv.astype(np.float32)).to(device)
            pred = model(xb, sb, head='reg')
            mse = float(F.mse_loss(pred, yb).item())
        mses.append(mse)
    return float(np.mean(mses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--stored_sig', default=None,
        help='path to stored sigs.npz (e.g. ikaros sigs to replay on daedalus)')
    ap.add_argument('--out_dir', default=None)
    args = ap.parse_args()

    host = hostname()
    out_dir = args.out_dir or os.path.abspath(os.path.join(
        HERE, '..', '..', '..', 'results', 'IDENTITY_BENCHMARK_2026-05-30', 'embodiment14b'))
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(args.device)

    nonce_A = b'phase14b_aud01'
    nonce_B = b'phase14b_aud99'   # mismatched nonce
    sig_A = LiveSig(nonce=nonce_A, host=host)
    sig_B = LiveSig(nonce=nonce_B, host=host)

    print(f"[spoof] host={host}")

    # ---------- T1 spoof defense ----------
    print("\n[spoof T1] training honest model with nonce_A...")
    model, (Xte, Yte) = train_T1_quick(device, sig_A, epochs=20)

    def honest_provider(B):
        return np.stack([sig_A.read() for _ in range(B)], 0)
    def random_provider(B):
        return np.clip(np.random.randn(B, 32).astype(np.float32), -4, 4)
    def replay_provider(B):
        # Replay: use a stored snapshot taken at start
        snapshot = sig_A.read()
        return np.stack([snapshot for _ in range(B)], 0)
    def nonce_mismatch_provider(B):
        return np.stack([sig_B.read() for _ in range(B)], 0)
    def stored_peer_provider(B):
        if not args.stored_sig or not os.path.exists(args.stored_sig):
            return random_provider(B)
        peer = np.load(args.stored_sig)['sigs']
        idx = np.random.randint(0, len(peer), size=B)
        return peer[idx].astype(np.float32)

    t1 = {}
    t1['honest']           = eval_T1(model, device, Xte, Yte, honest_provider)
    t1['random_sig']       = eval_T1(model, device, Xte, Yte, random_provider)
    t1['static_replay']    = eval_T1(model, device, Xte, Yte, replay_provider)
    t1['nonce_mismatch']   = eval_T1(model, device, Xte, Yte, nonce_mismatch_provider)
    t1['stored_peer']      = eval_T1(model, device, Xte, Yte, stored_peer_provider)

    # ---------- T3 spoof defense ----------
    # Train binary classifier own (0) vs peer (1) using stored_sig as peer if available
    print("\n[spoof T3] training twin classifier...")
    own_sigs = np.stack([sig_A.read() for _ in range(250)], 0)
    if args.stored_sig and os.path.exists(args.stored_sig):
        peer_sigs = np.load(args.stored_sig)['sigs'][:250]
    else:
        peer_sigs = own_sigs.copy()
        np.random.shuffle(peer_sigs.T); peer_sigs += 1.0
    X = np.concatenate([own_sigs, peer_sigs], 0).astype(np.float32)
    y = np.concatenate([np.zeros(250), np.ones(250)], 0).astype(np.int64)
    idx = np.arange(len(X)); np.random.shuffle(idx)
    split = int(0.7*len(X)); tr, te = idx[:split], idx[split:]
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(32,64), nn.GELU(), nn.Linear(64,2))
        def forward(self,x): return self.net(x)
    set_seed(0)
    m = M().to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for ep in range(20):
        order = np.random.permutation(len(tr))
        for i in range(0, len(order), 32):
            b = tr[order[i:i+32]]
            xb = torch.from_numpy(X[b]).to(device)
            yb = torch.from_numpy(y[b]).to(device)
            loss = F.cross_entropy(m(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()

    def t3_eval(sigs):
        with torch.no_grad():
            pred = m(torch.from_numpy(sigs.astype(np.float32)).to(device)).argmax(-1).cpu().numpy()
        return pred

    fresh_own = np.stack([sig_A.read() for _ in range(100)], 0)
    fresh_replay_one = np.tile(sig_A.read(), (100,1))  # static replay
    random_sigs = np.clip(np.random.randn(100,32).astype(np.float32), -4, 4)
    nonce_mismatched = np.stack([sig_B.read() for _ in range(100)], 0)

    t3 = {}
    # ideally classifier says "own" (0) for fresh_own
    t3['honest_own_p0_rate']      = float((t3_eval(fresh_own)==0).mean())
    t3['replay_one_p0_rate']      = float((t3_eval(fresh_replay_one)==0).mean())
    t3['random_p0_rate']          = float((t3_eval(random_sigs)==0).mean())
    t3['nonce_mismatch_p0_rate']  = float((t3_eval(nonce_mismatched)==0).mean())
    if args.stored_sig and os.path.exists(args.stored_sig):
        peer = np.load(args.stored_sig)['sigs'][250:350]
        # ideally classifier says NOT own (i.e. p0_rate low) for peer
        t3['peer_p0_rate']        = float((t3_eval(peer)==0).mean())

    out = {'host': host, 't1': t1, 't3': t3, 't': time.time(),
           'stored_sig': args.stored_sig}
    out_path = os.path.join(out_dir, f'{host}_spoof.json')
    save_json(out_path, out)
    print(f"\n[done] {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
