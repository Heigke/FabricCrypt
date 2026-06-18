"""DS-N7b — Adversarial ablation of DS-N7 Memory Palace.

Goal: try to FALSIFY "96% recall depends on NS-RAM physics" by showing the
work is actually done by the hash table.

Ablations:
  A. Digital dict baseline (same hash fn, same load factors)
  B. Random Vb levels (no NS-RAM dynamics) — sanity floor
  C. Per-cell Vth-like process variation (sigma from S3, here mapped to
     VG1 offset since the LUT does not expose Vth0)
  D. Readout noise on Vb_read (1, 5, 10, 50 mV)
  E. Key bit-flip noise (1, 5, 10, 25, 50 %)
  F. Flash baseline (energy & latency from literature)

CLI:
    HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python scripts/DS_N7b_ablation.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from S2b_transient import IiiNetLUT
import DS_N7_memory_palace as dsn7
from DS_N7_memory_palace import (
    MemoryPalace, random_keys, key_addresses, encode_value,
    parallel_read, decode_levels, energy_per_read_J,
    V_LO, V_HI, DEFAULT_L, VG1_READ, VG2_READ, VD_READ, T_READ, DT_READ,
    CB_READ,
)

OUT = ROOT / "results" / "DS_N7b_ablation"
OUT.mkdir(parents=True, exist_ok=True)

# Reference benchmark config — "PASS" scale from DS-N7
N_CELLS = 10000
P_PAIRS = 1000
L = DEFAULT_L
D_KEY = 64
K_SDM = 1
SEED = 0

# Energy literature anchors
DIGITAL_RAM_J_PER_BYTE = 1e-9   # ~1 nJ/byte DRAM access
FLASH_J_PER_BYTE       = 1e-8   # ~10 nJ/byte NAND read
FLASH_PAGE_READ_S      = 25e-6  # 25 us page read

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def _gen(seed=SEED):
    rng = np.random.default_rng(seed)
    K = random_keys(P_PAIRS, D=D_KEY, seed=seed)
    V = rng.integers(0, L, size=P_PAIRS).astype(np.int32)
    return K, V, rng


def _baseline_run(lut):
    """Re-run the canonical DS-N7 PASS-scale to get a self-contained baseline."""
    K, V, _ = _gen()
    pal = MemoryPalace(N_cells=N_CELLS, lut=lut, L=L, k_sdm=K_SDM, seed=SEED)
    t0 = time.time(); enc = pal.encode_batch(K, V); t_enc = time.time() - t0
    t0 = time.time(); V_rec = pal.recall_batch(K);  t_rec = time.time() - t0
    acc = float(np.mean(V_rec == V))
    # Energy: read pulse per cell, baseline reads P_PAIRS cells once
    E = energy_per_read_J(P_PAIRS)
    return {"accuracy": acc, "t_encode_s": t_enc, "t_recall_s": t_rec,
            "energy_J": E,
            "bytes_stored": P_PAIRS * np.log2(L) / 8,
            "throughput_ops_s": P_PAIRS / max(t_rec, 1e-9)}


# ─────────────────────────────────────────────────────────────────────
# A. Digital hash table baseline
# ─────────────────────────────────────────────────────────────────────
def ablation_A_digital_dict():
    """Use the EXACT same address hash → plain dict, see if it matches."""
    K, V, _ = _gen()
    # Address map (k=1) under same hash
    addrs = key_addresses(K, N_CELLS, k=1, proj_seed=1234 + SEED).reshape(-1)
    table = {}
    t0 = time.time()
    for a, v in zip(addrs.tolist(), V.tolist()):
        table[a] = v
    t_enc = time.time() - t0
    t0 = time.time()
    V_rec = np.array([table[a] for a in addrs.tolist()], dtype=np.int32)
    t_rec = time.time() - t0
    acc = float(np.mean(V_rec == V))
    # Energy: ~1 nJ per byte fetched from DRAM; 1 byte/level is generous
    E = P_PAIRS * 1 * DIGITAL_RAM_J_PER_BYTE
    return {"accuracy": acc, "t_encode_s": t_enc, "t_recall_s": t_rec,
            "energy_J": E, "throughput_ops_s": P_PAIRS / max(t_rec, 1e-9),
            "n_unique_cells_touched": int(np.unique(addrs).size),
            "collision_rate": 1.0 - np.unique(addrs).size / addrs.size}


# ─────────────────────────────────────────────────────────────────────
# B. Random Vb levels — no NS-RAM dynamics in the codebook
# ─────────────────────────────────────────────────────────────────────
def ablation_B_random_levels(lut):
    """Same hash addressing, but stored Vb is RANDOMIZED (no encoding)."""
    K, V, rng = _gen()
    pal = MemoryPalace(N_cells=N_CELLS, lut=lut, L=L, k_sdm=K_SDM, seed=SEED)
    # write nonsense
    addrs = key_addresses(K, N_CELLS, k=1, proj_seed=pal.proj_seed).reshape(-1)
    pal.Vb[addrs] = rng.uniform(V_LO, V_HI, size=addrs.size)
    V_rec = pal.recall_batch(K)
    acc = float(np.mean(V_rec == V))
    return {"accuracy": acc, "chance": 1.0 / L,
            "interpretation": "should collapse to ~1/L" }


# ─────────────────────────────────────────────────────────────────────
# C. Process variation — per-cell VG1 offset (Vth-like, sigma=43 mV)
# ─────────────────────────────────────────────────────────────────────
def _read_with_jitter(lut, Vb0, dVG1, dVG2=None, sigma_read=0.0, rng=None):
    """parallel_read clone that adds per-cell VG1 offset and optional readout noise."""
    N = Vb0.size
    Vb = Vb0.astype(np.float64).copy()
    vg1 = np.full(N, VG1_READ) + dVG1
    vg2 = np.full(N, VG2_READ) + (dVG2 if dVG2 is not None else 0.0)
    vd  = np.full(N, VD_READ)
    inv_Cb = 1.0 / CB_READ
    for _ in range(T_READ):
        Inet = lut(vg1, vg2, vd, Vb)
        dVb = (Inet * inv_Cb) * DT_READ
        np.clip(dVb, -0.5, 0.5, out=dVb)
        Vb = np.clip(Vb + dVb, -0.5, 1.5)
    if sigma_read > 0:
        if rng is None: rng = np.random.default_rng(0)
        Vb = Vb + rng.normal(0.0, sigma_read, size=N)
    return Vb


def ablation_C_vth_variation(lut, sigma_V=0.043):
    """Per-cell Vth0 σ=43 mV (S3 distribution). Mapped to VG1 offset."""
    K, V, rng = _gen()
    pal = MemoryPalace(N_cells=N_CELLS, lut=lut, L=L, k_sdm=K_SDM, seed=SEED)
    pal.encode_batch(K, V)
    # Per-cell VG1 offset for the WHOLE memory (fixed at fab)
    dVG1_all = rng.normal(0.0, sigma_V, size=N_CELLS)
    addrs = key_addresses(K, N_CELLS, k=1, proj_seed=pal.proj_seed).reshape(-1)
    Vb0 = pal.Vb[addrs]
    dVG1 = dVG1_all[addrs]
    codes = _read_with_jitter(lut, Vb0, dVG1)
    # Codebook is calibrated on a NOMINAL cell — same one used at write
    lvls = decode_levels(codes, pal.codebook)
    acc = float(np.mean(lvls == V))
    return {"accuracy": acc, "sigma_VG1_V": sigma_V}


# ─────────────────────────────────────────────────────────────────────
# D. Readout noise robustness
# ─────────────────────────────────────────────────────────────────────
def ablation_D_readout_noise(lut, sigmas_mV=(1, 5, 10, 50)):
    K, V, _ = _gen()
    pal = MemoryPalace(N_cells=N_CELLS, lut=lut, L=L, k_sdm=K_SDM, seed=SEED)
    pal.encode_batch(K, V)
    addrs = key_addresses(K, N_CELLS, k=1, proj_seed=pal.proj_seed).reshape(-1)
    Vb0 = pal.Vb[addrs]
    out = []
    for s_mV in sigmas_mV:
        rng = np.random.default_rng(int(s_mV) + 99)
        codes = _read_with_jitter(lut, Vb0, dVG1=0.0,
                                    sigma_read=s_mV * 1e-3, rng=rng)
        lvls = decode_levels(codes, pal.codebook)
        acc = float(np.mean(lvls == V))
        out.append({"sigma_mV": int(s_mV), "accuracy": acc})
    return out


# ─────────────────────────────────────────────────────────────────────
# E. Key bit-flip robustness curve
# ─────────────────────────────────────────────────────────────────────
def ablation_E_keyflip(lut, fracs=(0.01, 0.05, 0.10, 0.25, 0.50)):
    K, V, rng = _gen()
    pal = MemoryPalace(N_cells=N_CELLS, lut=lut, L=L, k_sdm=K_SDM, seed=SEED)
    pal.encode_batch(K, V)
    out = []
    for f in fracs:
        Kf = K.copy()
        n_flip = max(1, int(round(f * D_KEY)))
        for i in range(P_PAIRS):
            idx = rng.choice(D_KEY, size=n_flip, replace=False)
            Kf[i, idx] = -Kf[i, idx]   # actual flip, not erase
        V_rec = pal.recall_batch(Kf)
        out.append({"flip_frac": f,
                    "n_bits_flipped": n_flip,
                    "accuracy": float(np.mean(V_rec == V))})
    return out


# ─────────────────────────────────────────────────────────────────────
# F. Flash baseline
# ─────────────────────────────────────────────────────────────────────
def ablation_F_flash():
    bits_per_cell = 1.0    # SLC reference
    bytes_stored  = P_PAIRS * np.log2(L) / 8
    energy_J      = bytes_stored * FLASH_J_PER_BYTE
    # P_PAIRS reads, but page-based: assume ~16 lookups per page (256 B page / 16 B)
    n_pages = max(1, int(np.ceil(bytes_stored / 256)))
    wall_s  = n_pages * FLASH_PAGE_READ_S
    return {"accuracy": 1.0,
            "bits_per_cell": bits_per_cell,
            "bytes_stored": float(bytes_stored),
            "energy_J": float(energy_J),
            "wall_s": float(wall_s),
            "ops_s": float(P_PAIRS / wall_s)}


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────
def _plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    ax = axes[0]
    sigmas = [r["sigma_mV"] for r in results["D_readout_noise"]]
    accs   = [r["accuracy"]  for r in results["D_readout_noise"]]
    ax.semilogx(sigmas, accs, "o-", lw=2)
    ax.axhline(1.0 / L, ls="--", c="grey", label=f"chance (1/{L})")
    ax.axhline(results["baseline"]["accuracy"], ls=":", c="green",
                label=f"DS-N7 baseline")
    ax.set_xlabel("Readout noise σ_Vb (mV)")
    ax.set_ylabel("Recall accuracy")
    ax.set_title("D. Readout-noise robustness")
    ax.legend(); ax.grid(alpha=.3)

    ax = axes[1]
    fracs = [r["flip_frac"]*100 for r in results["E_keyflip"]]
    accs  = [r["accuracy"]      for r in results["E_keyflip"]]
    ax.plot(fracs, accs, "s-", lw=2, color="crimson")
    ax.axhline(1.0 / L, ls="--", c="grey", label=f"chance (1/{L})")
    ax.set_xlabel("Key bits flipped (%)")
    ax.set_ylabel("Recall accuracy")
    ax.set_title("E. Key bit-flip robustness")
    ax.legend(); ax.grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(OUT / "robustness_curves.png", dpi=120)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    print("[DS-N7b] Loading LUT …", flush=True)
    lut = IiiNetLUT()

    results = {}
    print("[DS-N7b] Baseline …", flush=True);     results["baseline"]            = _baseline_run(lut)
    print("[DS-N7b] A digital-dict …", flush=True); results["A_digital_dict"]      = ablation_A_digital_dict()
    print("[DS-N7b] B random-levels …", flush=True); results["B_random_levels"]    = ablation_B_random_levels(lut)
    print("[DS-N7b] C Vth variation …", flush=True); results["C_vth_variation"]    = ablation_C_vth_variation(lut)
    print("[DS-N7b] D readout noise …", flush=True); results["D_readout_noise"]    = ablation_D_readout_noise(lut)
    print("[DS-N7b] E key-flip …", flush=True);     results["E_keyflip"]           = ablation_E_keyflip(lut)
    print("[DS-N7b] F flash …", flush=True);        results["F_flash"]             = ablation_F_flash()

    # Pre-registered verdicts
    base_acc = results["baseline"]["accuracy"]
    dict_acc = results["A_digital_dict"]["accuracy"]
    d50      = next(r for r in results["D_readout_noise"] if r["sigma_mV"] == 50)
    verdict = {
        "claim_retracted_digital_matches": dict_acc >= base_acc - 0.005,
        "defended_noise_robust_50mV":       d50["accuracy"] >= 0.80,
        "caveat_vth_kills":                 results["C_vth_variation"]["accuracy"] < 0.50,
    }
    if verdict["claim_retracted_digital_matches"]:
        verdict["overall"] = "RETRACTED: digital dict matches NS-RAM — claim is hash-table-in-disguise"
    elif verdict["defended_noise_robust_50mV"]:
        verdict["overall"] = "DEFENDED: analog tolerance survives 50mV readout noise — NS-RAM advantage real"
    elif verdict["caveat_vth_kills"]:
        verdict["overall"] = "CAVEAT: real process variation may destroy the system"
    else:
        verdict["overall"] = "INCONCLUSIVE: digital matches, noise tolerated, vth survives — re-check"

    results["verdict"] = verdict
    results["wall_total_s"] = time.time() - t_start

    (OUT / "ablation_table.json").write_text(json.dumps(results, indent=2))
    _plot(results)

    # Markdown summary
    md = []
    md.append("# DS-N7b Ablation Summary\n")
    md.append(f"Baseline (NS-RAM): acc={base_acc:.3f}  energy={results['baseline']['energy_J']:.3e} J  t_recall={results['baseline']['t_recall_s']:.3f} s\n")
    md.append(f"A. Digital dict:    acc={dict_acc:.3f}  energy={results['A_digital_dict']['energy_J']:.3e} J  t_recall={results['A_digital_dict']['t_recall_s']:.3f} s\n")
    md.append(f"B. Random levels:   acc={results['B_random_levels']['accuracy']:.3f}  (chance={1/L:.2f})\n")
    md.append(f"C. Vth σ=43mV:      acc={results['C_vth_variation']['accuracy']:.3f}\n")
    md.append("D. Readout noise:\n")
    for r in results["D_readout_noise"]:
        md.append(f"   σ={r['sigma_mV']:>3} mV → acc={r['accuracy']:.3f}\n")
    md.append("E. Key flip:\n")
    for r in results["E_keyflip"]:
        md.append(f"   {int(r['flip_frac']*100):>3} %  → acc={r['accuracy']:.3f}\n")
    f = results["F_flash"]
    md.append(f"F. Flash:           acc={f['accuracy']:.2f}  energy={f['energy_J']:.3e} J  t={f['wall_s']:.3e} s  ops/s={f['ops_s']:.1f}\n")
    md.append("\n## Verdict\n")
    for k, v in verdict.items():
        md.append(f"- {k}: {v}\n")
    (OUT / "ablation_summary.md").write_text("".join(md))

    print("\n===== DS-N7b summary =====")
    print("".join(md))
    print(f"[DS-N7b] done in {results['wall_total_s']:.1f} s")


if __name__ == "__main__":
    main()
