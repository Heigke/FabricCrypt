"""H7 v10 HARDEN — multi-session / multi-load ikaros generalization vs daedalus.

Strongest open confound (oracle Grok): the gate may key on SESSION-specific window
statistics, not a stable physical property of the die. This records ikaros under
several DISTINCT operating regimes (idle / CPU-load / post-cooldown / time-separated)
WITHIN this boot and checks: does the model stay coherent on ALL novel ikaros regimes
(die identity) while daedalus (2nd die) still breaks?

NOTE: a TRUE cross-boot/cross-day test needs a reboot (cannot be done autonomously).
This covers time + thermal/load variation within one boot.

Run: sudo HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python h7_v10_multisession.py
"""
from __future__ import annotations
import sys, json, time, socket
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
from substrate_realtime_v3 import SubstrateStateV3
from h7_rooted_lm_v4a import GlobalNorm, SubstrateEncoderV4, K_TOKENS, WIN_LEN, BASE_MODEL, STATS
from h7_embodied_v8 import FilmEmbodiedSmolLM
from h7_embodied_v7 import encode, seq_nll
from transformers import AutoModelForCausalLM, AutoTokenizer

HOST = socket.gethostname()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/IDENTITY_H7_2026-06-09"
V10 = ROOT / "results/IDENTITY_EMBODIED_V10_2026-06-10"
CKPT = V10 / f"v10_best_{HOST}.pt"
DAED = OUT / "substrate_replay_daedalus_10ch.npz"
RESULT = V10 / "v10_multisession_result.json"
THERMAL = Path("/sys/class/thermal/thermal_zone0/temp")
EVAL_TEXT = ("The forest was dark and quiet as she walked. He could not remember "
             "what the letter had said, only that it arrived on a cold morning. "
             "Beyond the river the lights of the town flickered against the hills.")


def temp_c():
    try: return int(THERMAL.read_text().strip()) / 1000.0
    except Exception: return -1.0


def wait_cool(thr=55.0, timeout=90):
    t0 = time.time()
    while temp_c() > thr and time.time() - t0 < timeout:
        time.sleep(3)


def cpu_load(seconds):
    """Modest CPU/numpy load to shift thermal/clock regime (no GPU -> safer)."""
    t0 = time.time(); a = np.random.randn(512, 512)
    while time.time() - t0 < seconds and temp_c() < 85:
        a = a @ a; a = a / (np.linalg.norm(a) + 1e-9)


def main():
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
    norm = GlobalNorm(STATS); pad = tok.pad_token_id
    ids = tok(EVAL_TEXT, return_tensors="pt", truncation=True, max_length=96).input_ids.to(device)
    with torch.no_grad():
        lb = base(ids).logits[:, :-1, :]
        ppl_base = float(np.exp(F.cross_entropy(lb.reshape(-1, lb.size(-1)), ids[:, 1:].reshape(-1),
                                                ignore_index=pad).item()))

    @torch.no_grad()
    def ppl_w(w):
        l, _ = seq_nll(model, ids, encode(se, norm, w.astype(np.float32), device), pad)
        return float(np.exp(l.item()))

    state = SubstrateStateV3(hz_target=500); state.start(); time.sleep(2.0)

    def capture(n=8, gap=0.5):
        ws = []
        for _ in range(n):
            time.sleep(gap); ws.append(state.latest_window(length=WIN_LEN).copy())
        return ws

    sessions = {}
    # regime 1: idle now
    print(f"[idle] T={temp_c():.0f}C"); sessions["idle"] = (temp_c(), capture())
    # regime 2: under CPU load
    print("[cpu-load] spinning..."); cpu_load(12)
    print(f"[cpu-load] T={temp_c():.0f}C"); sessions["cpu_load"] = (temp_c(), capture())
    # regime 3: time-separated (wait ~60s) + cooled
    wait_cool(55); time.sleep(30)
    print(f"[t+90s cooled] T={temp_c():.0f}C"); sessions["delayed_cool"] = (temp_c(), capture())
    state.stop()

    daed = np.load(DAED)["windows"].astype(np.float32) if DAED.exists() else None
    res = {"ppl_base": ppl_base, "sessions": {}}
    print(f"\nbase PPL={ppl_base:.2f}")
    print(f"\n{'regime':16s} {'T(C)':>5s} {'PPL median':>11s} {'vs base':>8s}")
    print("-" * 46)
    all_ik = []
    for name, (T, ws) in sessions.items():
        ppls = [ppl_w(w) for w in ws]; med = float(np.median(ppls)); all_ik.extend(ppls)
        res["sessions"][name] = {"T": T, "ppl_median": med, "ppl_min": float(np.min(ppls)),
                                 "ppl_max": float(np.max(ppls)), "vs_base": med / ppl_base}
        print(f"{name:16s} {T:5.0f} {med:11.1f} {med/ppl_base:7.2f}x")
    daed_med = float(np.median([ppl_w(daed[i]) for i in range(min(8, len(daed)))])) if daed is not None else float("nan")
    res["daedalus_median"] = daed_med
    res["ikaros_all_median"] = float(np.median(all_ik))
    res["ikaros_all_max"] = float(np.max(all_ik))
    print(f"{'daedalus(2nd die)':16s} {'--':>5s} {daed_med:11.1f} {daed_med/ppl_base:7.2f}x")

    # verdict
    ik_max = float(np.max(all_ik)); ik_med = float(np.median(all_ik))
    cross = daed_med / ik_med
    ok = (ik_max < 3.0 * ppl_base) and (daed_med > 10.0 * ik_med)
    res["verdict"] = {"all_ikaros_regimes_coherent": bool(ik_max < 3.0 * ppl_base),
                      "daedalus_breaks_vs_ikaros": cross,
                      "die_not_session_supported_same_boot": bool(ok)}
    RESULT.write_text(json.dumps(res, indent=2, default=str))
    print(f"\n=== VERDICT (same-boot) ===")
    print(f"  all ikaros regimes coherent (<3x base): {'PASS' if ik_max < 3*ppl_base else 'FAIL'} (max {ik_max:.1f})")
    print(f"  daedalus breaks vs ikaros: {cross:.1f}x  ({'PASS' if cross > 10 else 'FAIL'})")
    print(f"  -> die-not-session (same boot): {'SUPPORTED' if ok else 'NOT SUPPORTED'}")
    print(f"  (cross-BOOT still untested — needs a reboot)")
    print(f"saved {RESULT}")


if __name__ == "__main__":
    main()
