"""Phase 18 — Train tiny LM with TRAINING-TIME chip-signal injection.

Three injection mechanisms (per backward step):
  1. Gradient noise: scale derived from chip variance proxy
  2. Dropout pattern: dropout RNG seeded per-step from hash(chip_signature)
  3. LR modulation: lr_step = lr_base * (1 + 0.05 * normalized_thermal)

Conditions:
  - vanilla            : no chip injection (fixed PRNG)
  - chip_injected      : live chip signals
  - synthetic_matched  : same amplitudes from PRNG (control / ablation)

Hosts:
  - ikaros (live LiveSig)
  - daedalus (replayed from prior daedalus_sigs.npz capture, embodiment14b)

Usage:
  python train_chip.py --host ikaros --cond vanilla --steps 500 [--burst 60]
  python train_chip.py --host ikaros --cond chip_injected --steps 500
  python train_chip.py --host ikaros --cond synthetic_matched --steps 500
  python train_chip.py --host daedalus --cond chip_injected --steps 500   # uses recorded sigs
"""
from __future__ import annotations
import os, sys, time, json, argparse, hashlib
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import (RESULTS, temp_c, thermal_guard, wait_cool,
                     save_json, sig_to_seed)
from tiny_lm import (TinyLM, init_model, build_tokenizer_and_data,
                     get_batch, count_params, SEQ_LEN, VOCAB)

CACHE_DATA = os.path.join(RESULTS, 'data_cache.npz')
INIT_CKPT = os.path.join(RESULTS, 'init_checkpoint.pt')


# ---------------- Signal providers (cpu-only, no torch needed) ----------------
class LiveProvider:
    def __init__(self):
        sys.path.insert(0, os.path.join(HERE, '..', 'embodiment14b'))
        from signature_live import LiveSig
        self.sig = LiveSig()

    def read(self):
        v = self.sig.read()
        # thermal as last entry of feature; in 14 schema the live sig has thermal-ish
        # we proxy: use mean of |v| as "variance proxy", v[2] is typically thermal-related
        return v


class RecordedProvider:
    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.sigs = d['sigs'].astype(np.float32)
        self.idx = 0

    def read(self):
        v = self.sigs[self.idx % len(self.sigs)]
        self.idx += 1
        return v


class SyntheticProvider:
    def __init__(self, seed=12345, ref_sigs=None):
        self.rng = np.random.default_rng(seed)
        if ref_sigs is not None:
            self.mu = np.asarray(ref_sigs).mean(0).astype(np.float32)
            self.sd = (np.asarray(ref_sigs).std(0) + 1e-6).astype(np.float32)
        else:
            self.mu = np.zeros(32, dtype=np.float32)
            self.sd = np.ones(32, dtype=np.float32)

    def read(self):
        return (self.mu + self.rng.normal(size=32).astype(np.float32) * self.sd).astype(np.float32)


class FixedPRNGProvider:
    """For vanilla: still need 'something' to seed dropout deterministically per step."""
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def read(self):
        return self.rng.normal(size=32).astype(np.float32)


def make_provider(host, cond):
    daedalus_npz = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/IDENTITY_BENCHMARK_2026-05-30/embodiment14b/daedalus_sigs.npz'
    if cond == 'vanilla':
        # deterministic, host-independent seed; we still differ vanilla(ikaros) vs vanilla(daedalus) only by host string -> nope, same seed.
        # For vanilla we expect ZERO divergence between ikaros and daedalus — that's the null model.
        return FixedPRNGProvider(seed=42), 'fixed_prng'
    if cond == 'synthetic_matched':
        # both hosts use IID Gaussian matched to daedalus_sigs amplitudes; different seeds per host to add slight diversity
        ref = np.load(daedalus_npz)['sigs']
        seed = 1000 if host == 'ikaros' else 2000
        return SyntheticProvider(seed=seed, ref_sigs=ref), 'synthetic'
    if cond == 'chip_injected':
        if host == 'ikaros':
            return LiveProvider(), 'live_ikaros'
        else:
            return RecordedProvider(daedalus_npz), 'recorded_daedalus'
    raise ValueError(cond)


# ---------------- Three injection mechanisms ----------------
def chip_grad_noise_scale(sig_vec):
    """Variance proxy in [0, 1]. Larger -> larger gradient noise."""
    v = np.asarray(sig_vec, dtype=np.float64)
    # robust scale ~ std of last 16 entries (live entries)
    s = np.std(v[16:]) if len(v) >= 32 else np.std(v)
    # squash
    return float(1.0 / (1.0 + np.exp(-s)))  # 0..1


def chip_thermal_norm(sig_vec):
    """A scalar in roughly [-1,1] used to perturb LR. Use sum of v[:8] (thermal-ish entries)."""
    v = np.asarray(sig_vec, dtype=np.float64)
    z = np.tanh(np.mean(v[:8]) / 4.0)
    return float(z)


def chip_dropout_seed(sig_vec):
    return sig_to_seed(sig_vec) & 0xFFFFFFFF


# ---------------- Training core ----------------
def train_one_burst(model, opt, train_ids, provider, cond,
                    steps, lr_base, batch_size, device,
                    pretrained_steps_done=0, burst_secs=120,
                    log=None):
    """Run up to `steps` or until burst_secs elapsed or thermal pause.

    Returns (steps_done_this_burst, signal_log_entries).
    """
    if log is None:
        log = []
    model.train()
    t0 = time.time()
    step_local = 0
    # batch RNG seeded by host+cond so batches are identical across runs
    batch_rng = np.random.default_rng(0xBA7C)
    # need to fast-forward batch_rng deterministically to pretrained_steps_done
    for _ in range(pretrained_steps_done):
        batch_rng.integers(0, max(1, len(train_ids) - SEQ_LEN - 1), size=batch_size)

    while step_local < steps:
        thermal_guard(verbose=True)
        if (time.time() - t0) > burst_secs:
            print(f"[burst] time exceeded {burst_secs}s, breaking at step_local={step_local}", flush=True)
            break

        X, Y = get_batch(train_ids, batch_size=batch_size, seq_len=SEQ_LEN, rng=batch_rng)
        X = X.to(device); Y = Y.to(device)

        # --- read chip signal BEFORE this step ---
        sig = provider.read()
        # mechanism 3: LR modulation
        if cond == 'vanilla':
            lr_now = lr_base
            grad_noise = 0.0
            drop_seed = step_local + pretrained_steps_done  # deterministic
        else:
            lr_now = lr_base * (1.0 + 0.05 * chip_thermal_norm(sig))
            grad_noise = 5e-4 * chip_grad_noise_scale(sig)  # small
            drop_seed = chip_dropout_seed(sig)

        for g in opt.param_groups:
            g['lr'] = lr_now

        # mechanism 2: dropout pattern (seed torch RNG per step)
        torch.manual_seed(drop_seed)

        opt.zero_grad(set_to_none=True)
        logits = model(X)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), Y.reshape(-1))
        loss.backward()

        # mechanism 1: gradient noise
        if grad_noise > 0:
            # seed grad noise with the same drop_seed for reproducibility
            g_rng = torch.Generator(device=device).manual_seed(drop_seed)
            for p in model.parameters():
                if p.grad is not None:
                    n = torch.empty_like(p.grad).normal_(generator=g_rng) * grad_noise
                    p.grad.add_(n)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        log.append({
            'gstep': pretrained_steps_done + step_local,
            'sig': sig.tolist() if hasattr(sig, 'tolist') else list(sig),
            'lr': float(lr_now),
            'grad_noise': float(grad_noise),
            'drop_seed': int(drop_seed),
            'loss': float(loss.item()),
            'temp_c': temp_c(),
        })
        step_local += 1
        if step_local % 25 == 0:
            print(f"  step {pretrained_steps_done + step_local} loss={loss.item():.3f} "
                  f"temp={temp_c():.1f}C lr={lr_now:.2e}", flush=True)
    return step_local, log


def evaluate_ppl(model, val_ids, device, batch_size=8, n_eval=50):
    model.eval()
    rng = np.random.default_rng(99)
    losses = []
    with torch.no_grad():
        for _ in range(n_eval):
            X, Y = get_batch(val_ids, batch_size=batch_size, seq_len=SEQ_LEN, rng=rng)
            X = X.to(device); Y = Y.to(device)
            logits = model(X)
            l = F.cross_entropy(logits.reshape(-1, VOCAB), Y.reshape(-1))
            losses.append(float(l.item()))
    avg = float(np.mean(losses))
    return avg, float(np.exp(avg))


def ensure_init_ckpt(device):
    if os.path.exists(INIT_CKPT):
        return torch.load(INIT_CKPT, map_location=device, weights_only=True)
    print("[init] building fresh init checkpoint (seed=1337)", flush=True)
    m = init_model(seed=1337)
    sd = m.state_dict()
    torch.save(sd, INIT_CKPT)
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', required=True, choices=['ikaros', 'daedalus'])
    ap.add_argument('--cond', required=True, choices=['vanilla', 'chip_injected', 'synthetic_matched'])
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--device', default='cuda')  # GPU: tiny model runs cool on iGPU; CPU AdamW heats the APU much more
    ap.add_argument('--burst', type=int, default=120, help='burst seconds')
    ap.add_argument('--bursts', type=int, default=6)
    ap.add_argument('--resume', action='store_true', help='resume from existing checkpoint')
    ap.add_argument('--resume_steps', type=int, default=0, help='step counter offset if resuming')
    args = ap.parse_args()

    print(f"[train] host={args.host} cond={args.cond} device={args.device} steps={args.steps}", flush=True)

    # ensure data
    print("[data] loading wikitext-2 + 8k vocab", flush=True)
    train_ids, val_ids, id_map = build_tokenizer_and_data(CACHE_DATA)
    print(f"  train tokens={len(train_ids)} val tokens={len(val_ids)}", flush=True)

    # model init from fixed seed (or resume)
    device = args.device
    init_sd = ensure_init_ckpt(device)
    model = TinyLM().to(device)
    tag = f"{args.host}_{args.cond}"
    ckpt_path_resume = os.path.join(RESULTS, f"{tag}.pt")
    if args.resume and os.path.exists(ckpt_path_resume):
        print(f"[resume] loading {ckpt_path_resume}", flush=True)
        sd = torch.load(ckpt_path_resume, map_location=device, weights_only=True)
        model.load_state_dict(sd)
    else:
        model.load_state_dict(init_sd)
    print(f"[model] params={count_params(model)}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    # signal provider
    provider, prov_name = make_provider(args.host, args.cond)
    print(f"[provider] {prov_name}", flush=True)

    # multi-burst training
    sig_log = []
    steps_done = args.resume_steps if args.resume else 0
    bursts_done = 0
    ckpt_path = ckpt_path_resume
    aborted = False
    while steps_done < args.steps and bursts_done < args.bursts:
        # wait cool before each burst (lenient target)
        wait_cool(target_c=55, timeout_s=180)
        bursts_done += 1
        remaining = args.steps - steps_done
        print(f"[burst {bursts_done}/{args.bursts}] target {remaining} steps, "
              f"temp_start={temp_c():.1f}C", flush=True)
        try:
            delta, sig_log = train_one_burst(
                model, opt, train_ids, provider, args.cond,
                steps=remaining, lr_base=args.lr, batch_size=args.bs,
                device=device, pretrained_steps_done=steps_done,
                burst_secs=args.burst, log=sig_log)
        except SystemExit as e:
            print(f"[burst {bursts_done}] thermal abort: {e}", flush=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f"[save-partial] {ckpt_path}", flush=True)
            aborted = True
            wait_cool(target_c=52, timeout_s=300)
            continue
        steps_done += delta
        torch.save(model.state_dict(), ckpt_path)  # incremental
        print(f"[burst {bursts_done}] done: {delta} steps, total={steps_done}, "
              f"temp_end={temp_c():.1f}C (saved)", flush=True)
        if delta == 0:
            print("[burst] zero progress, aborting (probably thermal-bound)", flush=True)
            break

    # eval
    print("[eval] computing val PPL ...", flush=True)
    try:
        val_loss, val_ppl = evaluate_ppl(model, val_ids, device)
    except SystemExit as e:
        print(f"[eval] thermal abort during eval: {e}", flush=True)
        val_loss, val_ppl = float('nan'), float('nan')
    print(f"[eval] val_loss={val_loss} ppl={val_ppl}", flush=True)

    # final save (idempotent)
    torch.save(model.state_dict(), ckpt_path)
    print(f"[save] {ckpt_path}", flush=True)

    meta = {
        'host': args.host,
        'cond': args.cond,
        'provider': prov_name,
        'steps_requested': args.steps,
        'steps_done': steps_done,
        'bursts_done': bursts_done,
        'lr_base': args.lr,
        'batch_size': args.bs,
        'val_loss': val_loss,
        'val_ppl': val_ppl,
        'params': count_params(model),
        'final_temp_c': temp_c(),
        'thermal_aborted': bool(aborted),
    }
    save_json(f"meta_{tag}.json", meta)

    # save signal log only for the ikaros chip_injected run (needed for clone-defeat replay)
    if args.host == 'ikaros' and args.cond == 'chip_injected':
        np.savez(os.path.join(RESULTS, 'ikaros_training_signal_log.npz'),
                 sigs=np.array([e['sig'] for e in sig_log], dtype=np.float32),
                 drop_seeds=np.array([e['drop_seed'] for e in sig_log], dtype=np.uint32),
                 lrs=np.array([e['lr'] for e in sig_log], dtype=np.float32),
                 grad_noises=np.array([e['grad_noise'] for e in sig_log], dtype=np.float32))
        print(f"[save] ikaros_training_signal_log.npz ({len(sig_log)} steps)", flush=True)


if __name__ == '__main__':
    main()
