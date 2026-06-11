"""H7 v10 FALSIFICATION battery — try hard to prove the body↔LLM coupling is FAKE.

The skeptic's hypothesis: v10 is not "embodiment", it is a narrow OOD anomaly
detector that scrambles on anything not matching its exact training distribution.
If so:
  (F1) a HELD-OUT ikaros recording (same die, recorded NOW, different thermal state)
       would ALSO break  -> it's snapshot-matching, not die-identity.
  (F2) real would break under thermal drift (GPU stress).
  (F3) the response is a sharp cliff, not graded -> pure detector.
  (F4) coherent output does NOT vary with the live signal -> only gates break/no-break.

Batteries (all in ONE live-substrate session to minimise thermal cycling):
  B0  reproduce: live real / knock / shuffle / zero / daedalus(real 2nd die)
  B1  HELD-OUT IKAROS replay (recorded this run): must stay coherent (~real) -> die identity
  B2  static frozen window x N  vs  live windows -> real-time vs static fingerprint
  B3  thermal-drift: capture real during a short GPU burst -> robustness
  B4  graded interpolation alpha*real+(1-alpha)*knock -> cliff vs graded
  B5  per-channel leave-one-out (zero each of 10 ch in real) -> what is load-bearing
  B6  gaussian-noise control (matched mean/var, no AR structure) -> should break
  B7  behavioral influence: token-dist divergence of OUTPUT across different real
      windows vs real-vs-zero -> does coherent text VARY with the live signal?

Run: sudo HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python h7_v10_falsify.py
"""
from __future__ import annotations
import sys, json, time, socket
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
from substrate_realtime_v3 import SubstrateStateV3
from h7_rooted_lm_v4a import GlobalNorm, SubstrateEncoderV4, K_TOKENS, WIN_LEN, N_CHANNELS, BASE_MODEL, STATS
from h7_embodied_v8 import FilmEmbodiedSmolLM
from h7_knockoff_kl_probe import make_knockoff, sym_kl
from h7_embodied_v7 import temporal_shuffle, encode, seq_nll
from transformers import AutoModelForCausalLM, AutoTokenizer

HOST = socket.gethostname()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/IDENTITY_H7_2026-06-09"
V10 = ROOT / "results/IDENTITY_EMBODIED_V10_2026-06-10"
CKPT = V10 / f"v10_best_{HOST}.pt"
DAED = OUT / "substrate_replay_daedalus_10ch.npz"
IK_REPLAY = OUT / f"substrate_replay_{HOST}_heldout_10ch.npz"
RESULT = V10 / "v10_falsify_result.json"
SEED = 202
THERMAL = Path("/sys/class/thermal/thermal_zone0/temp")

EVAL_TEXT = ("The forest was dark and quiet as she walked. He could not remember "
             "what the letter had said, only that it arrived on a cold morning. "
             "Beyond the river the lights of the town flickered against the hills.")


def temp_c():
    try: return int(THERMAL.read_text().strip()) / 1000.0
    except Exception: return -1.0


def load():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = FilmEmbodiedSmolLM().to(device).eval()
    se = SubstrateEncoderV4(d_emb=model.d, K=K_TOKENS).to(device).eval()
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    model.film.load_state_dict(ck["film"]); se.load_state_dict(ck["se"])
    msd = dict(model.named_parameters())
    for k, v in ck.get("lora", {}).items():
        if k in msd: msd[k].data.copy_(v.to(device))
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device).eval()
    for p in base.parameters(): p.requires_grad_(False)
    return model, se, base, tok, device, ck


@torch.no_grad()
def ppl_of(model, ids, S, pad):
    l, _ = seq_nll(model, ids, S, pad)
    return float(np.exp(l.item()))


def main():
    rng = np.random.default_rng(SEED)
    model, se, base, tok, device, ck = load()
    norm = GlobalNorm(STATS)
    pad = tok.pad_token_id
    ids = tok(EVAL_TEXT, return_tensors="pt", truncation=True, max_length=96).input_ids.to(device)
    with torch.no_grad():
        lb = base(ids).logits[:, :-1, :]
        ppl_base = float(np.exp(F.cross_entropy(lb.reshape(-1, lb.size(-1)), ids[:, 1:].reshape(-1),
                                                ignore_index=pad).item()))
    daed = np.load(DAED)["windows"] if DAED.exists() else None
    res = {"ckpt_step": ck.get("step"), "ppl_base": ppl_base, "temp_start": temp_c()}
    print(f"base PPL={ppl_base:.2f}  ckpt step={ck.get('step')}  T={temp_c():.0f}C")

    state = SubstrateStateV3(hz_target=500); state.start(); time.sleep(2.0)

    # ---- capture held-out ikaros windows over ~30s (B1) ----
    print("capturing 40 held-out ikaros windows...")
    ik_windows = []
    for _ in range(40):
        time.sleep(0.7); ik_windows.append(state.latest_window(length=WIN_LEN).copy())
    np.savez_compressed(IK_REPLAY, windows=np.stack(ik_windows))
    print(f"  saved {IK_REPLAY.name}  T={temp_c():.0f}C")

    def ppl_cond(w_or_S, n=1):
        if isinstance(w_or_S, torch.Tensor):
            return ppl_of(model, ids, w_or_S, pad)
        return ppl_of(model, ids, encode(se, norm, w_or_S, device), pad)

    # ---- B0 live real / knock / shuffle / zero / daedalus ----
    live = []
    for _ in range(8):
        time.sleep(0.5); live.append(state.latest_window(length=WIN_LEN).copy())
    real_ppls = [ppl_cond(w) for w in live]
    knock_ppls = [ppl_cond(make_knockoff(w, rng)) for w in live]
    shuf_ppls = [ppl_cond(temporal_shuffle(w, rng)) for w in live]
    zero_ppl = ppl_of(model, ids, torch.zeros(1, K_TOKENS, model.d, device=device), pad)
    daed_ppls = [ppl_cond(daed[rng.integers(0, len(daed))]) for _ in range(8)] if daed is not None else []
    # B1 held-out ikaros replay (same die, recorded this run)
    ik_ppls = [ppl_cond(w) for w in ik_windows[::2]]
    # B6 gaussian control matched mean/var per channel
    def gauss_like(w):
        mu = w.mean(0, keepdims=True); sd = w.std(0, keepdims=True) + 1e-6
        return (mu + sd * rng.standard_normal(w.shape)).astype(np.float32)
    gauss_ppls = [ppl_cond(gauss_like(w)) for w in live]

    def stat(a): return {"median": float(np.median(a)), "mean": float(np.mean(a)),
                         "min": float(np.min(a)), "max": float(np.max(a)), "n": len(a)} if a else {}
    res["B0_B1_B6"] = {
        "real_live": stat(real_ppls), "knock": stat(knock_ppls), "shuffle": stat(shuf_ppls),
        "zero": zero_ppl, "daedalus_real_die": stat(daed_ppls),
        "ikaros_heldout_replay": stat(ik_ppls), "gaussian_matched": stat(gauss_ppls),
    }
    rmed = np.median(real_ppls)
    print(f"\nB0/B1/B6 (vs real median {rmed:.1f}):")
    for k in ["real_live","ikaros_heldout_replay","zero","shuffle","knock","gaussian_matched","daedalus_real_die"]:
        s = res["B0_B1_B6"][k]
        if isinstance(s, dict) and s: print(f"  {k:24s} med={s['median']:.1f}  ({s['median']/rmed:.2f}x)")
        else: print(f"  {k:24s} {s:.1f}  ({s/rmed:.2f}x)")

    # ---- B2 static vs live ----
    w_static = live[0]
    static_ppls = [ppl_cond(w_static) for _ in range(6)]
    res["B2_static_vs_live"] = {"static_same_window": stat(static_ppls), "live": stat(real_ppls)}
    print(f"\nB2 static(one frozen window)={np.median(static_ppls):.1f}  live={rmed:.1f}")

    # ---- B4 graded interpolation real <-> knock ----
    w0 = live[0]; wk = make_knockoff(w0, rng)
    grad = {}
    for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
        wmix = (a * w0 + (1 - a) * wk).astype(np.float32)
        grad[f"alpha_{a}"] = ppl_cond(wmix)
    res["B4_interp_real_to_knock"] = grad
    print("\nB4 interp alpha*real+(1-a)*knock:")
    for a in [1.0,0.75,0.5,0.25,0.0]:
        print(f"  alpha={a:.2f} (real frac) PPL={grad[f'alpha_{a}']:.1f}")

    # ---- B5 per-channel leave-one-out (zero one channel of real) ----
    loo = {}
    for c in range(N_CHANNELS):
        w = live[0].copy(); w[:, c] = 0.0
        loo[f"ch{c}_zeroed"] = float(np.median([ppl_cond(w) for _ in range(2)]))
    res["B5_channel_LOO"] = {"baseline_real": float(ppl_cond(live[0])), "zeroed": loo}
    print("\nB5 per-channel LOO (zero one ch of real), baseline=%.1f:" % ppl_cond(live[0]))
    for c in range(N_CHANNELS):
        print(f"  ch{c} zeroed -> {loo[f'ch{c}_zeroed']:.1f}")

    # ---- B7 behavioral influence: does coherent OUTPUT vary with the live signal? ----
    @torch.no_grad()
    def last_logits(w_or_S):
        S = w_or_S if isinstance(w_or_S, torch.Tensor) else encode(se, norm, w_or_S, device)
        return model(ids, substrate_tokens=S).logits[0, :-1, :].cpu()
    Lr = [last_logits(w) for w in live[:6]]          # different live real windows
    Lz = last_logits(torch.zeros(1, K_TOKENS, model.d, device=device))
    # divergence between outputs under DIFFERENT real windows (real-vs-real)
    rr = [sym_kl(Lr[i], Lr[j]).median().item() for i in range(len(Lr)) for j in range(i+1, len(Lr))]
    # divergence real vs zero
    rz = [sym_kl(Lr[i], Lz).median().item() for i in range(len(Lr))]
    res["B7_behavioral"] = {"out_div_real_vs_real_median": float(np.median(rr)),
                            "out_div_real_vs_zero_median": float(np.median(rz)),
                            "interpretation": "if real-vs-real << real-vs-zero, coherent output is STABLE "
                                              "across live windows (gating, not graded real-time behavior)"}
    print(f"\nB7 output divergence: real-vs-real(diff live windows)={np.median(rr):.4f}  "
          f"real-vs-zero={np.median(rz):.4f}")

    # ---- B3 thermal drift: capture real during a short GPU burst, eval ----
    print(f"\nB3 thermal drift: T_before={temp_c():.0f}C")
    if temp_c() < 75:
        x = torch.randn(2048, 2048, device=device)
        t0 = time.time(); drift_windows = []
        while time.time() - t0 < 6.0 and temp_c() < 88:
            x = (x @ x) * 1e-4 + 0.1
            drift_windows.append(state.latest_window(length=WIN_LEN).copy())
        torch.cuda.synchronize() if device == "cuda" else None
        t_hot = temp_c()
        drift_ppls = [ppl_cond(w) for w in drift_windows[-6:]]
        res["B3_thermal_drift"] = {"T_hot": t_hot, "real_ppl_under_drift": stat(drift_ppls),
                                   "real_ppl_baseline": stat(real_ppls)}
        print(f"  T_hot={t_hot:.0f}C  real PPL under drift med={np.median(drift_ppls):.1f} "
              f"(baseline {rmed:.1f})  n={len(drift_ppls)}")
    else:
        res["B3_thermal_drift"] = {"skipped": "too hot to start"}
        print("  skipped (too hot)")

    state.stop()
    res["temp_end"] = temp_c()
    RESULT.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nsaved {RESULT}")

    # ---- verdict heuristics ----
    print("\n=== FALSIFICATION VERDICT ===")
    ik_med = res["B0_B1_B6"]["ikaros_heldout_replay"].get("median", float("nan"))
    daed_med = res["B0_B1_B6"]["daedalus_real_die"].get("median", float("nan")) if daed is not None else float("nan")
    f1 = ik_med < 2.0 * rmed   # held-out ikaros stays coherent
    f_die = (daed_med > 3.0 * rmed) and f1   # daedalus breaks AND ikaros doesn't -> die identity
    print(f"  F1 held-out ikaros stays coherent (<2x real): {'PASS' if f1 else 'FAIL'} "
          f"(ik {ik_med:.1f} vs real {rmed:.1f})")
    print(f"  DIE-IDENTITY (ikaros ok AND daedalus breaks): {'PASS' if f_die else 'FAIL'} "
          f"(daed {daed_med:.1f})")
    b3 = res.get("B3_thermal_drift", {})
    if "real_ppl_under_drift" in b3:
        drift_ok = b3["real_ppl_under_drift"]["median"] < 2.0 * rmed
        print(f"  THERMAL-ROBUST (real ok under drift): {'PASS' if drift_ok else 'FAIL'}")
    print(f"  B7 graded-behavior: real-vs-real out-div={np.median(rr):.4f} "
          f"(if ~0 vs real-vs-zero {np.median(rz):.4f}: only GATING, NOT graded real-time behavior)")


if __name__ == "__main__":
    main()
