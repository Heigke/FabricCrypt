"""DS-N9 — Neuromorphic-forgetting memory on NS-RAM cells.

Premise (after DS-N7 retraction)
--------------------------------
DS-N7b showed that the "Memory Palace" was hash-table-in-disguise: a digital
dict matched the NS-RAM 0.957 recall exactly because *we never used the
analog dynamics*. DS-N7b recommendation was to "pair with z2213 multi-tau
persistence so loci NATURALLY fade".

This script does that — and ablates it RIGOROUSLY, so the answer is honest
even if the verdict is "no advantage".

Physics
-------
Each cell stores body-charge Vb. Real NS-RAM cells leak — Vb relaxes toward
its fixed point with a per-cell time-constant. From z2213 measured at 20 Hz,
the three observed decay regimes are:

  fast:  τ = 0.10 s    (decay-per-50ms ≈ 0.607)
  mid:   τ = 1.0 s
  slow:  τ = 5.0 s

We model the cell as Vb relaxing to V_FIXED with EXPONENTIAL retention:

    Vb(t) = V_FIXED + (Vb(0) - V_FIXED) * exp(-t / τ_cell)

τ_cell is sampled per cell from a mixture (40% fast, 35% mid, 25% slow) to
model device heterogeneity. This is the *substrate* multi-tau, not the
algorithmic one.

Ablations (all REQUIRED, not optional)
--------------------------------------
1.  digital_nodecay      — perfect dict, no decay (sanity ceiling)
2.  digital_match_decay  — dict + manual per-key exponential decay with
                            EXACT same τ as the NS-RAM cell. If this beats
                            NS-RAM the architecture has no value.
3.  digital_noise        — dict + gaussian readout noise per timestep.
4.  nsram_jitter         — NS-RAM with write-pulse jitter (timing noise).
5.  digital_jitter       — same write jitter applied to digital encoder.

Pre-registered HONEST gates
---------------------------
G1 DEMO: write 1000 items, read at t ∈ {0, 0.1, 1, 10, 100} s; report accuracy.
G2 HYPOTHESIS: NS-RAM retention curve matches z2213 expected within 20%
   (i.e. fraction-surviving at t=τ falls within [0.18, 0.55] — exp(-1)=0.368
   ±20% relative).
G3 ABLATION GATE: NS-RAM does NOT beat digital_match_decay by ≥3 pp on any
   t-slice → claim retracted (architecture is just decay).
G4 ROBUSTNESS GATE: NS-RAM with write-pulse jitter retains ≥ digital+jitter
   by ≥3 pp at t=1s.

Energy: tally J/write × P_pairs × refresh-rate-implied-by-decay over a
fair test window.

CLI
---
    HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python scripts/DS_N9_forgetting_memory.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from S2b_transient import IiiNetLUT
from DS_N7_memory_palace import (
    MemoryPalace, random_keys, key_addresses, encode_value,
    parallel_read, decode_levels, calibrate_codebook,
    V_LO, V_HI, DEFAULT_L, VG1_READ, VG2_READ, VD_READ, T_READ, DT_READ, CB_READ,
)

OUT = ROOT / "results" / "DS_N9_forgetting_memory"
OUT.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────
N_CELLS = 10_000
P_PAIRS = 1_000
L = DEFAULT_L
D_KEY = 64
SEED = 0

# Retention time-constants from z2213 measurements
TAU_FAST = 0.10   # s
TAU_MID  = 1.0
TAU_SLOW = 5.0
TAU_MIX  = np.array([0.40, 0.35, 0.25])   # weights fast/mid/slow

# Fixed point Vb relaxes toward. Below V_LO encoding → all reads decode → level 0.
V_FIXED = V_LO

# Read-times for retention curve (seconds)
T_READS = np.array([0.0, 0.1, 1.0, 10.0, 100.0])

# Energy literature anchors (DS-N7b-consistent)
DIGITAL_RAM_J_PER_BYTE = 1e-9        # ~1 nJ/byte DRAM access
DIGITAL_REFRESH_HZ      = 15.0       # 64ms DRAM refresh interval
DIGITAL_BYTES_PER_PAIR  = 8          # 8B per key→value entry

# NS-RAM write physics — set body-charge once via short pulse:
NSRAM_J_PER_WRITE = 4e-15            # 4 fJ ~ Cb·Vd·∆Vb
NSRAM_J_PER_READ  = 3.75e-11 / N_CELLS  # from DS-N7b energy (per-cell)


def sample_taus(N: int, rng) -> np.ndarray:
    """Sample τ per cell from heterogeneous mixture."""
    pick = rng.choice(3, size=N, p=TAU_MIX)
    return np.where(pick == 0, TAU_FAST, np.where(pick == 1, TAU_MID, TAU_SLOW))


# ─────────────────────────────────────────────────────────────────────
# NS-RAM forgetting palace
# ─────────────────────────────────────────────────────────────────────
class ForgettingPalace(MemoryPalace):
    """MemoryPalace whose cells leak Vb toward V_FIXED with per-cell τ."""

    def __init__(self, N_cells, lut, L=DEFAULT_L, seed=0, jitter_sigma=0.0):
        super().__init__(N_cells=N_cells, lut=lut, L=L, k_sdm=1, seed=seed)
        rng = np.random.default_rng(seed + 99)
        self.tau = sample_taus(N_cells, rng)
        self.write_time = np.zeros(N_cells, dtype=np.float64)
        self.jitter_sigma = jitter_sigma
        # Track Vb0 written (so we can compute exact analytic relaxation)
        self.Vb_written = np.full(N_cells, V_FIXED, dtype=np.float64)

    def encode_batch(self, K, V, t_now=0.0):
        addrs = key_addresses(K, self.N, k=self.k, proj_seed=self.proj_seed)
        vb_targets = encode_value(V, L=self.L)
        if self.jitter_sigma > 0.0:
            rng_j = np.random.default_rng(int(t_now * 1e6) ^ 0xABCD)
            vb_targets = vb_targets + rng_j.normal(0.0, self.jitter_sigma,
                                                   size=vb_targets.shape)
        addrs_flat = addrs.reshape(-1)
        vb_flat = np.repeat(vb_targets, self.k)
        self.Vb[addrs_flat] = vb_flat
        self.Vb_written[addrs_flat] = vb_flat
        self.write_time[addrs_flat] = t_now
        return {"addrs": addrs}

    def _relax_to(self, t_now: float):
        """Analytic exp-decay relaxation of every cell to t_now."""
        dt = np.maximum(t_now - self.write_time, 0.0)
        decay = np.exp(-dt / self.tau)
        self.Vb = V_FIXED + (self.Vb_written - V_FIXED) * decay

    def recall_batch_at(self, K, t_now: float):
        self._relax_to(t_now)
        return self.recall_batch(K)


# ─────────────────────────────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────────────────────────────
class DigitalDict:
    """Perfect digital dict — keyed by hash(K) → V."""
    def __init__(self, N_cells, L=DEFAULT_L, seed=0, jitter_p=0.0):
        self.N = N_cells; self.L = L
        self.proj_seed = 1234 + seed
        self.cells = np.full(N_cells, -1, dtype=np.int32)   # -1 = empty
        self.write_time = np.zeros(N_cells, dtype=np.float64)
        rng = np.random.default_rng(seed + 99)
        self.tau = sample_taus(N_cells, rng)
        self.jitter_p = jitter_p
        self.rng_j = np.random.default_rng(seed + 31)

    def encode_batch(self, K, V, t_now=0.0):
        addrs = key_addresses(K, self.N, k=1, proj_seed=self.proj_seed).reshape(-1)
        Vw = V.copy().astype(np.int32)
        if self.jitter_p > 0.0:
            flip = self.rng_j.random(Vw.size) < self.jitter_p
            Vw[flip] = self.rng_j.integers(0, self.L, size=int(flip.sum()))
        self.cells[addrs] = Vw
        self.write_time[addrs] = t_now

    def recall_batch_at(self, K, t_now: float, mode="nodecay", noise_sigma=0.0):
        addrs = key_addresses(K, self.N, k=1, proj_seed=self.proj_seed).reshape(-1)
        v = self.cells[addrs].copy()
        if mode == "nodecay":
            pass
        elif mode == "match_decay":
            # Survival prob = exp(-dt/τ); on "forget" produce uniform random
            # level (mimics readout collapsing toward fixed-point digit).
            dt = np.maximum(t_now - self.write_time[addrs], 0.0)
            p_survive = np.exp(-dt / self.tau[addrs])
            forget = self.rng_j.random(v.size) > p_survive
            # On forget → snap to level 0 (V_FIXED ≈ V_LO ≈ level 0)
            v[forget] = 0
        elif mode == "noise":
            # add gaussian noise per readout, snap to nearest level
            noisy = v.astype(np.float64) + self.rng_j.normal(0.0, noise_sigma,
                                                              size=v.size)
            v = np.clip(np.rint(noisy), 0, self.L - 1).astype(np.int32)
        else:
            raise ValueError(mode)
        # empty (-1) → forced wrong (snap to 0)
        v[v < 0] = 0
        return v


# ─────────────────────────────────────────────────────────────────────
# Experiment driver
# ─────────────────────────────────────────────────────────────────────
def gen_workload(seed=SEED):
    rng = np.random.default_rng(seed)
    K = random_keys(P_PAIRS, D=D_KEY, seed=seed)
    V = rng.integers(0, L, size=P_PAIRS).astype(np.int32)
    return K, V


def run_nsram(lut, t_reads, jitter_sigma=0.0):
    K, V = gen_workload()
    pal = ForgettingPalace(N_CELLS, lut=lut, L=L, seed=SEED,
                           jitter_sigma=jitter_sigma)
    pal.encode_batch(K, V, t_now=0.0)
    accs = []
    for t in t_reads:
        Vr = pal.recall_batch_at(K, t_now=t)
        accs.append(float(np.mean(Vr == V)))
    return accs


def run_digital(mode, t_reads, jitter_p=0.0, noise_sigma=0.0):
    K, V = gen_workload()
    dig = DigitalDict(N_CELLS, L=L, seed=SEED, jitter_p=jitter_p)
    dig.encode_batch(K, V, t_now=0.0)
    accs = []
    for t in t_reads:
        Vr = dig.recall_batch_at(K, t_now=t, mode=mode, noise_sigma=noise_sigma)
        accs.append(float(np.mean(Vr == V)))
    return accs


def measure_retention_match(nsram_curve, t_reads):
    """Check that NS-RAM accuracy decay matches expected mixture-exp within 20%.

    Mixture survival expected at t = sum_i w_i * exp(-t/τ_i).
    """
    s_expected = (TAU_MIX[0] * np.exp(-t_reads / TAU_FAST) +
                  TAU_MIX[1] * np.exp(-t_reads / TAU_MID) +
                  TAU_MIX[2] * np.exp(-t_reads / TAU_SLOW))
    # Map accuracy → survival: at t=0, acc=a0 (channel-limited).
    a0 = nsram_curve[0]
    a_chance = 1.0 / L
    s_observed = (np.array(nsram_curve) - a_chance) / max(1e-6, (a0 - a_chance))
    err = np.abs(s_observed - s_expected) / np.maximum(s_expected, 1e-3)
    return {
        "expected_survival": s_expected.tolist(),
        "observed_survival": s_observed.tolist(),
        "rel_err": err.tolist(),
        "match_within_20pct": bool(np.all(err < 0.20)),
    }


def main():
    t0_all = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] DS-N9 forgetting memory")
    print(f"  N_cells={N_CELLS}, P_pairs={P_PAIRS}, L={L}")
    print(f"  τ = (fast {TAU_FAST}s, mid {TAU_MID}s, slow {TAU_SLOW}s) mix {TAU_MIX}")

    lut = IiiNetLUT()
    t_reads = T_READS

    # ── G1 + ablations
    print("\nRunning NS-RAM (clean)…")
    nsram_clean = run_nsram(lut, t_reads, jitter_sigma=0.0)
    print(f"  acc(t) = {[f'{a:.3f}' for a in nsram_clean]}")

    print("Running NS-RAM (jitter σ=10mV)…")
    nsram_jit = run_nsram(lut, t_reads, jitter_sigma=0.010)
    print(f"  acc(t) = {[f'{a:.3f}' for a in nsram_jit]}")

    print("Running digital — nodecay (perfect ceiling)…")
    dig_nodecay = run_digital("nodecay", t_reads)
    print(f"  acc(t) = {[f'{a:.3f}' for a in dig_nodecay]}")

    print("Running digital — match_decay (KILL-SHOT ablation)…")
    dig_match  = run_digital("match_decay", t_reads)
    print(f"  acc(t) = {[f'{a:.3f}' for a in dig_match]}")

    print("Running digital — noise (σ=0.5 level)…")
    dig_noise = run_digital("noise", t_reads, noise_sigma=0.5)
    print(f"  acc(t) = {[f'{a:.3f}' for a in dig_noise]}")

    print("Running digital — jitter (write key-flip 5%)…")
    dig_jit = run_digital("nodecay", t_reads, jitter_p=0.05)
    print(f"  acc(t) = {[f'{a:.3f}' for a in dig_jit]}")

    # ── G2 retention match
    g2 = measure_retention_match(nsram_clean, t_reads)

    # ── G3 ablation gate
    a_nsram = np.array(nsram_clean)
    a_match = np.array(dig_match)
    delta = (a_nsram - a_match) * 100.0
    g3_pass = bool(np.any(delta >= 3.0))
    g3_best_dt = float(t_reads[int(np.argmax(delta))])
    g3_best_delta = float(np.max(delta))

    # ── G4 jitter robustness at t=1.0s
    idx_1s = int(np.argmin(np.abs(t_reads - 1.0)))
    g4_delta = (nsram_jit[idx_1s] - dig_jit[idx_1s]) * 100.0
    g4_pass = bool(g4_delta >= 3.0)

    # ── Energy comparison over a 100s window
    # NS-RAM: 1 write/pair + reads at t_reads
    nsram_E = (P_PAIRS * NSRAM_J_PER_WRITE
               + len(t_reads) * P_PAIRS * NSRAM_J_PER_READ)
    # Digital DRAM: refresh every 64ms over 100s, plus reads.
    window_s = 100.0
    n_refresh = window_s * DIGITAL_REFRESH_HZ
    dig_E = (P_PAIRS * DIGITAL_BYTES_PER_PAIR * DIGITAL_RAM_J_PER_BYTE
             * (1.0 + n_refresh)
             + len(t_reads) * P_PAIRS * DIGITAL_BYTES_PER_PAIR
                * DIGITAL_RAM_J_PER_BYTE)

    summary = {
        "t_reads_s": t_reads.tolist(),
        "nsram_clean_acc": nsram_clean,
        "nsram_jitter_acc": nsram_jit,
        "digital_nodecay_acc": dig_nodecay,
        "digital_match_decay_acc": dig_match,
        "digital_noise_acc": dig_noise,
        "digital_jitter_acc": dig_jit,
        "G2_retention_match": g2,
        "G3_ablation": {
            "delta_pp_per_t": delta.tolist(),
            "best_delta_pp": g3_best_delta,
            "best_dt_s": g3_best_dt,
            "PASS_nsram_beats_digital_match_3pp": g3_pass,
        },
        "G4_jitter_at_1s": {
            "nsram_jit_acc": nsram_jit[idx_1s],
            "digital_jit_acc": dig_jit[idx_1s],
            "delta_pp": g4_delta,
            "PASS": g4_pass,
        },
        "energy_J_over_100s": {
            "nsram_total": nsram_E,
            "digital_dram_total": dig_E,
            "ratio_digital_over_nsram": dig_E / max(nsram_E, 1e-30),
        },
        "config": {
            "N_CELLS": N_CELLS, "P_PAIRS": P_PAIRS, "L": L,
            "TAU_FAST": TAU_FAST, "TAU_MID": TAU_MID, "TAU_SLOW": TAU_SLOW,
            "TAU_MIX": TAU_MIX.tolist(),
            "NSRAM_J_PER_WRITE": NSRAM_J_PER_WRITE,
            "NSRAM_J_PER_READ_PER_CELL": NSRAM_J_PER_READ,
            "DIGITAL_RAM_J_PER_BYTE": DIGITAL_RAM_J_PER_BYTE,
            "DIGITAL_REFRESH_HZ": DIGITAL_REFRESH_HZ,
        },
        "wall_s": time.time() - t0_all,
    }

    # Verdict
    if g3_pass:
        verdict = ("NS-RAM beats digital+match_decay by "
                   f"{g3_best_delta:.1f}pp at t={g3_best_dt}s — "
                   "architecture-level signal present.")
    else:
        verdict = ("NS-RAM does NOT beat digital+match_decay anywhere "
                   "(max Δ = {:.2f}pp). "
                   "Retention curve alone is not a discriminator — "
                   "any decay-bounded dict matches.".format(g3_best_delta))
    summary["verdict"] = verdict
    print(f"\nVERDICT: {verdict}")

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.set_xscale("symlog", linthresh=0.05)
        ax.plot(t_reads, nsram_clean,    "o-", label="NS-RAM (clean)", lw=2)
        ax.plot(t_reads, nsram_jit,      "s--", label="NS-RAM (jitter σ=10mV)")
        ax.plot(t_reads, dig_nodecay,    "k:",  label="Digital (no decay)")
        ax.plot(t_reads, dig_match,      "v-",  label="Digital + matched decay")
        ax.plot(t_reads, dig_noise,      "^-",  label="Digital + noise σ=0.5")
        ax.plot(t_reads, dig_jit,        "x--", label="Digital + key-jitter 5%")
        ax.axhline(1.0 / L, ls=":", c="gray", label=f"chance (1/{L})")
        ax.set_xlabel("Time since write (s)")
        ax.set_ylabel("Recall accuracy")
        ax.set_title("DS-N9 Retention curves — NS-RAM vs digital baselines")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "retention_curve.png", dpi=120)
        plt.close(fig)
        print(f"Wrote {OUT/'retention_curve.png'}")
    except Exception as e:
        print(f"Plot skipped: {e}")

    # Ablation table
    lines = ["# DS-N9 Ablation Table\n",
             "| Condition | t=0 | t=0.1s | t=1s | t=10s | t=100s |",
             "|---|---|---|---|---|---|"]
    def row(name, vals):
        return f"| {name} | " + " | ".join(f"{v:.3f}" for v in vals) + " |"
    lines += [
        row("NS-RAM (clean)",          nsram_clean),
        row("NS-RAM (jitter)",         nsram_jit),
        row("Digital no-decay",        dig_nodecay),
        row("Digital match-decay",     dig_match),
        row("Digital + noise",         dig_noise),
        row("Digital + key-jitter 5%", dig_jit),
    ]
    lines += [
        "",
        "## Gates",
        f"- G2 retention-match within 20% : **{g2['match_within_20pct']}**",
        f"- G3 NS-RAM > digital+match by 3pp : **{g3_pass}** (best Δ = {g3_best_delta:.2f}pp @ t={g3_best_dt}s)",
        f"- G4 NS-RAM-jit > digital-jit at 1s : **{g4_pass}** (Δ = {g4_delta:.2f}pp)",
        "",
        "## Energy over 100s window",
        f"- NS-RAM total : {nsram_E:.2e} J",
        f"- DRAM total   : {dig_E:.2e} J (refresh × {n_refresh:.0f})",
        f"- Ratio (DRAM/NS-RAM) : {dig_E/max(nsram_E,1e-30):.1f}×",
        "",
        f"## Verdict\n{verdict}",
    ]
    (OUT / "ablation_table.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT/'ablation_table.md'}")
    print(f"Wall: {summary['wall_s']:.1f}s")


if __name__ == "__main__":
    main()
