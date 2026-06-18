"""H7 — ANALOG load-bearing embodiment: the per-die voltage-droop transient reservoir IS the LM's
only nonlinearity. This is the upgrade over the digital cache-XOR version (h7_rooted_lm_embodied.py):
the cache gate was load-bearing but GENERIC across dies (xor_sim worked). The analog droop transient
is PER-DIE silicon (process variation in di/dt droop + settling), so it can be load-bearing AND unique.

Grounding (already-measured, not simulated): h7_transient_vdroop.py drove the die with a sharp-edge
GPU burst stream u[t] and recorded the settling transient as NTAP=12 time-multiplexed virtual nodes x
N_CH=10 channels -> Tn (L,12,10). Its result: a LINEAR readout of these die features does temporal XOR
(u[t-1]^u[t-2]) at ~0.80 while the u-window control sits at chance (0.50) -> "die_needed: true".

This script wires that reservoir into an LM-style predictor:
  logits[t] = BYPASS(u-window lags)  +  DIE_HEAD(standardized die reservoir features at t)
  - BYPASS is a full LINEAR escape hatch over recent input bits. A linear map CANNOT compute XOR, so the
    model COULD ignore the die but then fails the task -> any dependence on the die is EARNED, not forced.
  - DIE_HEAD is a linear readout of the PHYSICAL reservoir state. The reservoir's fading memory + droop
    nonlinearity is the only thing that can supply XOR. Plain cross-entropy. No margin/spoof loss.

Held-out temporal split (train first 70% of timesteps, eval last 30%). Conditions:
  native   real die features on held-out timesteps        -> must SOLVE (load-bearing)
  shuffled die features row-permuted (right stats, wrong binding/timestep) -> must BREAK
  zero     die features zeroed (reservoir removed)         -> bypass only -> chance
  foreign  ANOTHER die's recorded features for the SAME u  -> if it BREAKS, the computation is UNIQUE
           to this die (h7_transient_vdroop uses a fixed seed, so u/targets match across dies).

native >> shuffled/zero  => DEEP load-bearing.   native >> foreign => UNIQUE (per-die).
Pure numpy/torch on recorded Tn — no root, no GPU bursts here. (Fresh-collection eval is a stronger
follow-up; daedalus Tn must be collected to run the foreign/uniqueness arm.)
Out: results/IDENTITY_H7_2026-06-09/embodied_analog_{host}.json
"""
from __future__ import annotations
import sys, json, time, socket, argparse, math
from pathlib import Path
import numpy as np

HOST = socket.gethostname()
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results/IDENTITY_H7_2026-06-09"
WASHOUT = 150


def load_die(host):
    p = OUT / f"transient_vdroop_raw_{host}.npz"
    if not p.exists():
        return None, None
    d = np.load(p)
    return d["u"].astype(int), d["Tn"].astype(np.float32)   # u:(L,)  Tn:(L,12,10)


def lag(x, k):
    y = np.zeros_like(x)
    if k > 0:
        y[k:] = x[:-k]; return y
    return x.copy()


def die_features(Tn):
    """Reservoir state = transient taps flattened + 2 step-lags (same construction as the harness)."""
    flat = Tn.reshape(Tn.shape[0], -1)                       # (L, 120)
    return np.hstack([flat, lag(flat, 1), lag(flat, 2)])     # (L, 360)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="XOR_t1t2", choices=["XOR_t1t2", "XOR_t1t3", "PAR3_123"])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--uwin", type=int, default=6, help="bypass linear window over input bit lags")
    ap.add_argument("--die_rank", type=int, default=0, help="0=full die features; >0 = PCA rank cap")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--foreign_host", default=None, help="other die host for the uniqueness arm")
    a = ap.parse_args()

    import torch, torch.nn as nn
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    u, Tn = load_die(HOST)
    if u is None:
        print(f"[{HOST}] missing transient_vdroop_raw_{HOST}.npz — run h7_transient_vdroop.py first"); sys.exit(2)
    L = len(u)

    # targets (the die-solvable temporal functions)
    if a.task == "XOR_t1t2": y = (lag(u, 1) ^ lag(u, 2))
    elif a.task == "XOR_t1t3": y = (lag(u, 1) ^ lag(u, 3))
    else: y = (lag(u, 1) ^ lag(u, 2) ^ lag(u, 3))
    y = y.astype(np.int64)

    Xdie = die_features(Tn)                                   # (L, 360)
    Uwin = np.stack([lag(u.astype(np.float32), k) for k in range(1, a.uwin + 1)], 1)  # (L, uwin)

    cut = WASHOUT + int(0.7 * (L - WASHOUT))
    tr = np.arange(WASHOUT, cut); te = np.arange(cut, L)

    # standardize on TRAIN stats only
    mu, sd = Xdie[tr].mean(0), Xdie[tr].std(0) + 1e-9
    Xz = (Xdie - mu) / sd
    umu, usd = Uwin[tr].mean(0), Uwin[tr].std(0) + 1e-9
    Uz = (Uwin - umu) / usd

    # optional rank cap on die features (fit PCA on train) — mirrors the harness rank-limited probe
    proj = None
    if a.die_rank > 0:
        Xc = Xz[tr] - Xz[tr].mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        proj = Vt[:a.die_rank]                                # (rank, 360)
    def die_proj(M):
        return M if proj is None else (M - Xz[tr].mean(0)) @ proj.T

    Xz_use = die_proj(Xz)
    ddim = Xz_use.shape[1]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bypass = nn.Linear(a.uwin, 2).to(dev)                    # LINEAR escape hatch over input bits
    die_head = nn.Linear(ddim, 2, bias=False).to(dev)        # readout of the PHYSICAL reservoir state
    opt = torch.optim.AdamW(list(bypass.parameters()) + list(die_head.parameters()), lr=a.lr)
    lossf = nn.CrossEntropyLoss()

    Ut = torch.from_numpy(Uz).float().to(dev)
    Dt = torch.from_numpy(Xz_use).float().to(dev)
    yt = torch.from_numpy(y).to(dev)
    tri = torch.from_numpy(tr).to(dev)

    for step in range(a.steps):
        logits = bypass(Ut[tri]) + die_head(Dt[tri])
        loss = lossf(logits, yt[tri])
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 1000 == 0:
            acc = float((logits.argmax(-1) == yt[tri]).float().mean())
            print(f"  step {step+1:5d} ce={loss.item():.4f} train_acc={acc:.3f}", flush=True)

    @torch.no_grad()
    def acc_with(Dmat):
        lg = bypass(Ut[torch.from_numpy(te).to(dev)]) + die_head(torch.from_numpy(Dmat[te]).float().to(dev))
        return float((lg.argmax(-1).cpu().numpy() == y[te]).mean())

    @torch.no_grad()
    def acc_bypass_only():
        lg = bypass(Ut[torch.from_numpy(te).to(dev)])        # die contribution dropped
        return float((lg.argmax(-1).cpu().numpy() == y[te]).mean())

    rng = np.random.default_rng(7)
    res = {}
    res["native"] = round(acc_with(Xz_use), 3)
    Xsh = Xz_use.copy(); perm = te.copy(); rng.shuffle(perm); Xsh[te] = Xz_use[perm]
    res["shuffled"] = round(acc_with(Xsh), 3)
    res["zero"] = round(acc_bypass_only(), 3)

    # uniqueness arm: another die's recorded features (same u/targets), ikaros-trained heads
    foreign = a.foreign_host or ("daedalus" if HOST == "ikaros" else "ikaros")
    uf, Tnf = load_die(foreign)
    if Tnf is not None and len(uf) == L and np.array_equal(uf, u):
        Xf = (die_features(Tnf) - mu) / sd
        res["foreign"] = round(acc_with(die_proj(Xf)), 3)
        res["foreign_host"] = foreign
    elif Tnf is not None:
        res["foreign"] = None; res["foreign_host"] = f"{foreign} (u mismatch — re-collect same seed)"

    chance = float(max(y[te].mean(), 1 - y[te].mean()))
    nat = res["native"]; corrupt = max(res["shuffled"], res["zero"])
    out = {
        "host": HOST, "task": a.task, "chance": round(chance, 3), "uwin": a.uwin,
        "die_rank": a.die_rank, "die_dim": ddim, "n_train": len(tr), "n_eval": len(te),
        "results": res,
        "LOAD_BEARING": bool(nat > chance + 0.10 and nat - corrupt > 0.10),
    }
    if "foreign" in res and res["foreign"] is not None:
        out["UNIQUE_per_die"] = bool(nat - res["foreign"] > 0.10)
        out["uniqueness_gap"] = round(nat - res["foreign"], 3)
    out["verdict"] = (
        ("DEEP+ANALOG load-bearing" if out["LOAD_BEARING"] else "NOT load-bearing (die not needed)")
        + (f"; UNIQUE per-die (gap {out.get('uniqueness_gap')})" if out.get("UNIQUE_per_die")
           else ("; NOT unique (foreign solves it too)" if "UNIQUE_per_die" in out
                 else "; uniqueness UNTESTED (need foreign-die data, same seed)"))
    )
    OUT.mkdir(parents=True, exist_ok=True)
    jp = OUT / f"embodied_analog_{HOST}.json"; jp.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)
    print(f"\n[{HOST}] VERDICT: {out['verdict']}", flush=True)
    print(f"  native={nat} shuffled={res['shuffled']} zero={res['zero']} "
          f"foreign={res.get('foreign','NA')} chance={chance:.2f}", flush=True)
    print(f"  saved {jp}", flush=True)


if __name__ == "__main__":
    main()
