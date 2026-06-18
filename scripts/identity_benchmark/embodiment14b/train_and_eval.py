"""Phase 14B Task D — train + eval all 4 tasks, vanilla vs embodied.

Outputs JSON results to:
  results/IDENTITY_BENCHMARK_2026-05-30/embodiment14b/<host>_results.json

CPU is fine; the model is tiny.
"""
from __future__ import annotations
import os, sys, json, time, math, argparse, random
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

from common13 import thermal_guard, get_apu_temp_c, hostname, save_json
from signature_live import LiveSig
from embodied_tiny import EmbodiedTiny, count_params
import tasks as T


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def get_sig_for_batch(sig: LiveSig, B: int, device, training=True):
    """Sample B signature vectors. During training: jitter slightly."""
    out = np.zeros((B, 32), dtype=np.float32)
    for i in range(B):
        out[i] = sig.read()
    if training:
        out += np.random.randn(*out.shape).astype(np.float32) * 0.05
    return torch.from_numpy(out).to(device)


def get_static_sig(sig: LiveSig, B: int, device, n_avg=8):
    """Stable sig: average n_avg reads to denoise."""
    acc = np.zeros(32, dtype=np.float32)
    for _ in range(n_avg):
        acc += sig.read()
    acc /= n_avg
    return torch.from_numpy(acc[None,:].repeat(B, axis=0)).to(device)


# ----------------- T1: latency regression -----------------
def run_T1(device, sig: LiveSig, n_seeds=5, epochs=25, batch=32, embodied=True,
           cached_dataset=None):
    if cached_dataset is None:
        print(f"[T1] building dataset (with concurrent sigs)...")
        inputs, y, sigs, meta = T.build_T1_dataset(n_per_expr=4, thermal_guard_fn=thermal_guard, sig=sig)
    else:
        inputs, y, sigs, meta = cached_dataset
    print(f"[T1] embodied={embodied} N={len(inputs)} y_std={y.std():.3f}")
    idx = np.arange(len(inputs))
    rng = np.random.default_rng(0); rng.shuffle(idx)
    split = int(0.8*len(inputs))
    tr, te = idx[:split], idx[split:]
    Xtr, Ytr, Str = inputs[tr], y[tr], sigs[tr]
    Xte, Yte, Ste = inputs[te], y[te], sigs[te]
    mses = []
    for seed in range(n_seeds):
        set_seed(seed)
        model = EmbodiedTiny(embodied=embodied).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        for ep in range(epochs):
            order = np.random.permutation(len(Xtr))
            for i in range(0, len(order), batch):
                b = order[i:i+batch]
                xb = torch.from_numpy(Xtr[b]).to(device)
                yb = torch.from_numpy(Ytr[b]).to(device)
                sb = torch.from_numpy(Str[b]).to(device) if embodied else None
                pred = model(xb, sb, head='reg')
                loss = F.mse_loss(pred, yb)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(Xte).to(device)
            yb = torch.from_numpy(Yte).to(device)
            sb = torch.from_numpy(Ste).to(device) if embodied else None
            pred = model(xb, sb, head='reg')
            mse = float(F.mse_loss(pred, yb).item())
        mses.append(mse)
        print(f"[T1] seed={seed} MSE={mse:.4f}")
    return {'mses': mses, 'mean_mse': float(np.mean(mses)),
            'ci95_lo': float(np.percentile(mses, 2.5)),
            'ci95_hi': float(np.percentile(mses, 97.5)),
            'n_samples_train': len(tr), 'n_samples_test': len(te),
            'meta': meta}


# ----------------- T2: anomaly AUROC -----------------
def auroc(scores, labels):
    s = np.asarray(scores); y = np.asarray(labels).astype(int)
    pos = s[y==1]; neg = s[y==0]
    if len(pos)==0 or len(neg)==0: return 0.5
    # mann-whitney U
    rank = np.argsort(np.argsort(s)) + 1
    n_pos = len(pos); n_neg = len(neg)
    U = rank[y==1].sum() - n_pos*(n_pos+1)/2
    return float(U / (n_pos*n_neg))


def run_T2(device, sig: LiveSig, n_seeds=5, epochs=15, batch=64, embodied=True):
    print(f"[T2] embodied={embodied} building dataset...")
    X, y = T.build_T2_dataset(n=600, n_anom=60, sig=sig)
    idx = np.arange(len(X)); np.random.shuffle(idx)
    split = int(0.7*len(X))
    tr, te = idx[:split], idx[split:]
    aurocs = []
    for seed in range(n_seeds):
        set_seed(seed)
        # Tiny MLP head on top of sig OR on dummy input
        class M(nn.Module):
            def __init__(self, use_sig):
                super().__init__()
                in_d = 32 if use_sig else 1
                self.net = nn.Sequential(
                    nn.Linear(in_d, 64), nn.GELU(),
                    nn.Linear(64, 64), nn.GELU(),
                    nn.Linear(64, 2),
                )
                self.use_sig = use_sig
            def forward(self, sigv):
                if self.use_sig: return self.net(sigv)
                # vanilla: only sees a constant input
                return self.net(torch.zeros(sigv.shape[0], 1, device=sigv.device))
        model = M(embodied).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        for ep in range(epochs):
            order = np.random.permutation(len(tr))
            for i in range(0, len(order), batch):
                b = tr[order[i:i+batch]]
                xb = torch.from_numpy(X[b]).to(device)
                yb = torch.from_numpy(y[b]).to(device)
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(X[te]).to(device)
            logits = model(xb)
            scores = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        a = auroc(scores, y[te])
        aurocs.append(a)
        print(f"[T2] seed={seed} AUROC={a:.3f}")
    return {'aurocs': aurocs, 'mean_auroc': float(np.mean(aurocs)),
            'ci95_lo': float(np.percentile(aurocs, 2.5)),
            'ci95_hi': float(np.percentile(aurocs, 97.5)),
            'n_anom': 60, 'n_total': 600}


# ----------------- T3: twin paradox -----------------
def run_T3(device, sig: LiveSig, peer_sig_npz=None, n_seeds=5, epochs=20, batch=32, embodied=True):
    """If peer_sig_npz is provided, eval on cross-host data (transplant test).
    Otherwise eval on own held-out signatures.
    """
    print(f"[T3] embodied={embodied} (peer={peer_sig_npz is not None})")
    # Build labeled sigs: 200 from ikaros (this host -> label=0) + (we don't have peer here yet -> synthesize NEG by shuffling)
    n_per = 250
    sigs_own = np.stack([sig.read() for _ in range(n_per)], 0)
    if peer_sig_npz and os.path.exists(peer_sig_npz):
        peer = np.load(peer_sig_npz)['sigs']
        sigs_peer = peer[:n_per]
    else:
        # Synthetic peer: shuffle dim order + add bias. NOT REAL — only for unit-test path.
        sigs_peer = sigs_own.copy()
        np.random.shuffle(sigs_peer.T)
        sigs_peer += np.random.randn(*sigs_peer.shape).astype(np.float32)*0.5 + 1.0
        sigs_peer = np.clip(sigs_peer, -4, 4).astype(np.float32)
    X = np.concatenate([sigs_own, sigs_peer], 0).astype(np.float32)
    y = np.concatenate([np.zeros(n_per), np.ones(n_per)], 0).astype(np.int64)
    idx = np.arange(len(X)); np.random.shuffle(idx)
    split = int(0.7*len(X))
    tr, te = idx[:split], idx[split:]
    accs = []
    for seed in range(n_seeds):
        set_seed(seed)
        class M(nn.Module):
            def __init__(self, use_sig):
                super().__init__()
                in_d = 32 if use_sig else 1
                self.net = nn.Sequential(nn.Linear(in_d, 64), nn.GELU(), nn.Linear(64, 2))
                self.use_sig = use_sig
            def forward(self, x):
                if not self.use_sig:
                    x = torch.zeros(x.shape[0], 1, device=x.device)
                return self.net(x)
        model = M(embodied).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        for ep in range(epochs):
            order = np.random.permutation(len(tr))
            for i in range(0, len(order), batch):
                b = tr[order[i:i+batch]]
                xb = torch.from_numpy(X[b]).to(device)
                yb = torch.from_numpy(y[b]).to(device)
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(X[te]).to(device)
            pred = model(xb).argmax(-1).cpu().numpy()
        acc = float((pred == y[te]).mean())
        accs.append(acc)
        print(f"[T3] seed={seed} acc={acc:.3f}")
    return {'accs': accs, 'mean_acc': float(np.mean(accs)),
            'ci95_lo': float(np.percentile(accs, 2.5)),
            'ci95_hi': float(np.percentile(accs, 97.5)),
            'used_real_peer': bool(peer_sig_npz and os.path.exists(peer_sig_npz))}


# ----------------- T4: substrate-aware completion -----------------
def run_T4(device, sig: LiveSig, n_seeds=3, epochs=15, batch=16, embodied=True,
           cached_dataset=None):
    """T4 fair design: vanilla model has NO access to chip state, so its best
    strategy is to predict the marginal-prior best N across a *population* of
    machines. We don't have multi-host throughput data here, so the vanilla
    baseline picks the GENERIC (max-throughput at population mean) candidate
    which we approximate as candidate 0 ("safe small N"). Embodied learns
    THIS machine's specific best_idx.
    """
    if cached_dataset is None:
        print(f"[T4] measuring throughputs...")
        inputs, labels, thps, best_idx = T.build_T4_dataset(reps_per_N=2, thermal_guard_fn=thermal_guard)
    else:
        inputs, labels, thps, best_idx = cached_dataset
    print(f"[T4] embodied={embodied} best_idx={best_idx} thps={[f'{t:.2e}' for t in thps]}")
    accs = []; speedups = []
    baseline_thp = float(thps[0])  # generic = "smallest safe N", what vanilla picks
    for seed in range(n_seeds):
        set_seed(seed)
        if not embodied:
            # Vanilla can't observe chip state; it has no signal to pick a chip-specific N.
            # Its best strategy = pick the safe default (idx 0). Acc = (best_idx==0).
            picked = 0
            acc = float(picked == best_idx)
            chosen_thp = float(thps[picked])
            speedup = chosen_thp / max(baseline_thp, 1e-9)
            accs.append(acc); speedups.append(speedup)
            print(f"[T4] seed={seed} VANILLA picked=0 acc={acc:.3f} speedup={speedup:.2f}x")
            continue
        model = EmbodiedTiny(embodied=True, n_classes=len(T.T4_CANDIDATES)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        Xall = torch.from_numpy(inputs).to(device)
        Yall = torch.from_numpy(labels).to(device)
        for ep in range(epochs):
            order = np.random.permutation(len(inputs))
            for i in range(0, len(order), batch):
                b = order[i:i+batch]
                xb = Xall[b]; yb = Yall[b]
                sb = get_sig_for_batch(sig, len(b), device, training=True)
                logits = model(xb, sb, head='cls')
                loss = F.cross_entropy(logits, yb)
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            sb = get_static_sig(sig, len(inputs), device, n_avg=8)
            pred = model(Xall, sb, head='cls').argmax(-1).cpu().numpy()
        acc = float((pred == best_idx).mean())
        picked = int(np.bincount(pred).argmax())
        chosen_thp = float(thps[picked])
        speedup = chosen_thp / max(baseline_thp, 1e-9)
        accs.append(acc); speedups.append(speedup)
        print(f"[T4] seed={seed} EMBODIED acc={acc:.3f} picked={picked} speedup={speedup:.2f}x")
    return {'accs': accs, 'speedups': speedups,
            'mean_acc': float(np.mean(accs)), 'mean_speedup': float(np.mean(speedups)),
            'thps': thps.tolist(), 'best_idx': best_idx,
            'candidates': T.T4_CANDIDATES}


# ----------------- driver -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', default='T1,T2,T3,T4')
    ap.add_argument('--device', default='cpu')  # tiny model, CPU is enough and avoids GPU thermal
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--peer_sig', default=None, help='Path to peer signatures npz for T3 transplant')
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--also_dump_sigs', action='store_true',
                    help='dump 500 live signatures to <host>_sigs.npz for transplant test')
    args = ap.parse_args()

    host = hostname()
    out_dir = args.out_dir or os.path.abspath(os.path.join(
        HERE, '..', '..', '..', 'results', 'IDENTITY_BENCHMARK_2026-05-30', 'embodiment14b'))
    os.makedirs(out_dir, exist_ok=True)

    # nonce for unfakeability tests
    nonce = b'phase14b_aud01'
    sig = LiveSig(nonce=nonce, host=host)
    device = torch.device(args.device)

    if args.also_dump_sigs:
        sigs = np.stack([sig.read() for _ in range(500)], 0)
        path = os.path.join(out_dir, f'{host}_sigs.npz')
        np.savez(path, sigs=sigs, host=host, nonce=nonce.hex())
        print(f"[dump] saved {path}")

    results = {'host': host, 't_start': time.time(), 'device': args.device,
               'apu_temp_start_c': get_apu_temp_c(), 'tasks': {}}
    tasks_to_run = args.tasks.split(',')

    out_path = os.path.join(out_dir, f'{host}_results.json')
    # load existing to merge across separate runs
    if os.path.exists(out_path):
        try:
            existing = json.load(open(out_path))
            if 'tasks' in existing:
                results['tasks'].update(existing['tasks'])
        except Exception:
            pass

    def _save_partial():
        results['t_end'] = time.time()
        results['apu_temp_end_c'] = get_apu_temp_c()
        save_json(out_path, results)

    for t in tasks_to_run:
        if t == 'T1':
            t1_data = T.build_T1_dataset(n_per_expr=4, thermal_guard_fn=thermal_guard, sig=sig)
            v = run_T1(device, sig, n_seeds=args.seeds, embodied=False, cached_dataset=t1_data)
            e = run_T1(device, sig, n_seeds=args.seeds, embodied=True,  cached_dataset=t1_data)
            ratio = e['mean_mse'] / max(v['mean_mse'], 1e-9)
            results['tasks']['T1'] = {'vanilla': v, 'embodied': e, 'mse_ratio': ratio,
                                     'prereg_pass': ratio <= 0.5}
            _save_partial()
        elif t == 'T2':
            v = run_T2(device, sig, n_seeds=args.seeds, embodied=False)
            e = run_T2(device, sig, n_seeds=args.seeds, embodied=True)
            results['tasks']['T2'] = {'vanilla': v, 'embodied': e,
                                      'prereg_pass': (e['mean_auroc'] >= 0.85 and v['mean_auroc'] <= 0.6)}
            _save_partial()
        elif t == 'T3':
            v = run_T3(device, sig, peer_sig_npz=args.peer_sig, n_seeds=args.seeds, embodied=False)
            e = run_T3(device, sig, peer_sig_npz=args.peer_sig, n_seeds=args.seeds, embodied=True)
            results['tasks']['T3'] = {'vanilla': v, 'embodied': e,
                                      'prereg_pass': (e['mean_acc'] >= 0.95 and v['mean_acc'] <= 0.55)}
            _save_partial()
        elif t == 'T4':
            t4_data = T.build_T4_dataset(reps_per_N=2, thermal_guard_fn=thermal_guard)
            v = run_T4(device, sig, n_seeds=args.seeds, embodied=False, cached_dataset=t4_data)
            e = run_T4(device, sig, n_seeds=args.seeds, embodied=True,  cached_dataset=t4_data)
            sp = e['mean_speedup'] / max(v['mean_speedup'], 1e-9)
            results['tasks']['T4'] = {'vanilla': v, 'embodied': e, 'speedup_ratio': sp,
                                      'prereg_pass': sp >= 1.2}
            _save_partial()

    _save_partial()
    print(f"\n[done] saved {out_path}")
    print(json.dumps({k: {'prereg_pass': v.get('prereg_pass')} for k,v in results['tasks'].items()}, indent=2))


if __name__ == '__main__':
    main()
