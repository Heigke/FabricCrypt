"""H7 self-effect channel sweep — which substrate channel does the model's OWN compute move most?

The interoception path (v14 gaps 3/6) only works if the model's own computation perturbs the
substrate STRONGLY and READABLY. closed_loop_verify measured a weak ΔR² (~0.5-0.9%) on a few
thermal channels — but that may be the wrong, slow channels. Compute → power draw is near-
instant; compute → temperature is slow and smeared. This sweep alternates IDLE vs compute-BURST
and measures, per channel, the burst-vs-idle effect size (Cohen's d) and whether it's GRADED
with burst intensity (needed for content-coupled bursts).

Self-thermal-guarded: checks thermal_zone0 and cools before each burst (no external watchdog
needed; bursts are sized modest). Run as root, HSA_OVERRIDE_GFX_VERSION=11.0.0.
"""
from __future__ import annotations
import sys, json, time, socket
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent))
from substrate_realtime_v3 import SubstrateStateV3
from h7_rooted_lm_v4a import WIN_LEN, N_CHANNELS

HOST = socket.gethostname()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/IDENTITY_H7_2026-06-09"
ZONE = Path("/sys/class/thermal/thermal_zone0/temp")
HOT_C = 84.0       # cool down before a burst if above this
COOL_C = 70.0
N_CYCLES = 18
INTENSITIES = [1024, 2048, 4096]   # matmul size = burst intensity (graded test)


def temp_c():
    try: return int(ZONE.read_text()) / 1000.0
    except Exception: return 0.0


def cool_to(target=COOL_C, timeout=60):
    t0 = time.time()
    while temp_c() > target and time.time() - t0 < timeout:
        time.sleep(1.0)


def burst(device, n, secs=1.5):
    """Heavy matmul burst of size n for ~secs seconds — the model's 'thinking' proxy."""
    a = torch.randn(n, n, device=device); b = torch.randn(n, n, device=device)
    t0 = time.time(); k = 0
    while time.time() - t0 < secs:
        a = (a @ b).tanh() * 0.5 + 0.5; k += 1
    if device == "cuda": torch.cuda.synchronize()
    return k


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = SubstrateStateV3(hz_target=500); state.start()
    print(f"[{HOST}] warmup 6s..."); time.sleep(6.0)

    # --- effect-size sweep: idle vs fixed-intensity burst, N_CYCLES times ---
    idle_means, burst_means = [], []
    for c in range(N_CYCLES):
        if temp_c() > HOT_C: cool_to()
        time.sleep(1.0)
        idle_means.append(state.latest_window(length=WIN_LEN).mean(axis=0))   # quiet → per-channel mean
        burst(device, 2048, secs=1.5)                                          # the compute burst
        time.sleep(0.05)
        burst_means.append(state.latest_window(length=WIN_LEN).mean(axis=0))   # right after burst
        if (c + 1) % 6 == 0: print(f"  cycle {c+1}/{N_CYCLES}  temp={temp_c():.0f}C")
    idle_means = np.array(idle_means); burst_means = np.array(burst_means)     # (cycles, 10)

    # Cohen's d per channel (paired burst vs idle), and robust % shift
    diff = burst_means - idle_means
    pooled_sd = np.sqrt((idle_means.std(0) ** 2 + burst_means.std(0) ** 2) / 2) + 1e-9
    cohen_d = diff.mean(0) / pooled_sd
    pct_shift = diff.mean(0) / (np.abs(idle_means.mean(0)) + 1e-9) * 100

    # --- graded test: does each channel scale monotonically with burst intensity? ---
    grad_rows = []
    for n in INTENSITIES:
        vals = []
        for _ in range(5):
            if temp_c() > HOT_C: cool_to()
            time.sleep(0.5); burst(device, n, secs=1.2); time.sleep(0.05)
            vals.append(state.latest_window(length=WIN_LEN).mean(axis=0))
        grad_rows.append(np.array(vals).mean(0))
    grad = np.array(grad_rows)   # (3 intensities, 10)
    # monotonicity = corr(intensity rank, channel mean) per channel
    ranks = np.arange(len(INTENSITIES))
    mono = np.array([np.corrcoef(ranks, grad[:, ch])[0, 1] if grad[:, ch].std() > 1e-9 else 0.0
                     for ch in range(N_CHANNELS)])
    state.stop()

    order = np.argsort(-np.abs(cohen_d))
    res = {"host": HOST, "n_cycles": N_CYCLES, "cohen_d": cohen_d.tolist(),
           "pct_shift": pct_shift.tolist(), "monotonicity_vs_intensity": mono.tolist(),
           "ranked_channels_by_effect": order.tolist(),
           "best_channel": int(order[0]), "best_cohen_d": float(cohen_d[order[0]]),
           "best_monotonic_channel": int(np.argmax(np.abs(mono))),
           "best_monotonicity": float(mono[np.argmax(np.abs(mono))])}
    out = OUT / f"self_effect_sweep_{HOST}.json"; out.write_text(json.dumps(res, indent=2))
    print(f"\n=== SELF-EFFECT SWEEP [{HOST}] — does the model's compute move its body? ===")
    print("ch :  Cohen_d   %shift   monotonic(vs intensity)")
    for ch in order:
        print(f"  {ch}: {cohen_d[ch]:+8.3f}  {pct_shift[ch]:+8.2f}%   {mono[ch]:+.2f}")
    print(f">>> strongest self-effect channel = {order[0]}  (|d|={abs(cohen_d[order[0]]):.2f}); "
          f"most graded = ch{int(np.argmax(np.abs(mono)))} (mono={mono[np.argmax(np.abs(mono))]:+.2f})")
    print(f"INTEROCEPTION VIABLE if some |d|>0.8 AND |mono|>0.8.  saved {out}")


if __name__ == "__main__":
    main()
