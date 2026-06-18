"""DS-N7d — Method-of-Loci done right: pre-trained STDP substrate + wave recall.

Prior verdict (DS-N7c): all three "Memory Palace" architectures DEAD.
Hypothesis under test now: we misunderstood the architecture. Memory
champions don't hash — they use a *pre-trained* spatial substrate, and
recall is *trajectory dynamics* on that substrate. Items are PERTURBATIONS
of a learned associative manifold.

Concretely:
  1. Pre-train: N=10K NS-RAM cells in a 2D grid (ring-like torus). Drive
     with smooth synthetic "spatial walks". STDP forms synapse w_ij so a
     spike at cell i causes a spike at neighbour j after Δt. Result:
     activation propagates as a wave along learned paths.
  2. Encode: each (key,value) pair gets a "room" — a location on the
     grid. Hebbian binding ties key→location, location→value via fast
     plasticity. Multiple items live on the same substrate.
  3. Recall: cue activates entry cells → wave propagates along PRE-LEARNED
     synapses → reaches binding location → key→value readout from the
     spike pattern.

Brutal ablations:
  A. Randomize pre-trained synapses → if wave doesn't propagate, recall
     should fail. Confirms substrate role is causal.
  B. Digital baseline: numpy MLP (key→value classification). Same data,
     same input/output.
  C. Random-shuffle of stored items vs. ordered presentation.
  D. ALL energies include ADC + decision (honest accounting, lesson from
     DS-N7c).

PASS gates (pre-registered, before running):
  INFRA:        full-cue recall ≥80% on 1000 items in 10K substrate
  HYPOTHESIS:   NS-RAM beats digital MLP on PARTIAL CUE (25% bits) by
                ≥15pp — because dynamics fill missing bits via learned
                associations.
  KILL-SHOT:    if randomized-pretrain matches NS-RAM dynamics within
                3pp, retract — wave propagation does nothing.

CLI:
    HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python scripts/DS_N7d_loci_right.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from S2b_transient import IiiNetLUT  # noqa: E402

OUT = ROOT / "results" / "DS_N7d_loci_right"
OUT.mkdir(parents=True, exist_ok=True)

# ----- Energy anchors (same as DS-N7c, honest accounting) ------------------
NSRAM_J_READ = 3.75e-15      # per cell, per read step
NSRAM_J_WRITE = 4e-15        # per cell, per write step
ADC_J_PER_SAMPLE = 1e-12     # 1 pJ per ADC conversion (10-bit, conservative)
DIGITAL_MAC_J = 1e-12        # 1 pJ per multiply-add (digital MLP)
DIGITAL_J_PER_BYTE = 1e-9    # DRAM table

RNG = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# Substrate: 2D toroidal grid of NS-RAM cells with learned synapses
# ---------------------------------------------------------------------------
GRID_H, GRID_W = 100, 100      # 10K cells
N_CELLS = GRID_H * GRID_W
SYN_RADIUS = 3                 # local connectivity radius
V_REST = 0.30
V_SPIKE_THR = 0.55             # spike threshold on Vb
V_RESET = 0.30

def make_local_topology(H=GRID_H, W=GRID_W, r=SYN_RADIUS):
    """Each cell has local synaptic targets within radius r on torus."""
    pre, post = [], []
    for y in range(H):
        for x in range(W):
            i = y * W + x
            for dy in range(-r, r+1):
                for dx in range(-r, r+1):
                    if dy == 0 and dx == 0:
                        continue
                    j = ((y+dy) % H) * W + ((x+dx) % W)
                    pre.append(i); post.append(j)
    return np.array(pre, dtype=np.int32), np.array(post, dtype=np.int32)

PRE, POST = make_local_topology()
N_SYN = PRE.size  # ~480K synapses

def pretrain_substrate(n_walks=200, walk_steps=120, lut=None, seed=1):
    """Drive substrate with smooth 2D walks → STDP forms directional weights.

    Returns W: float32 array shape (N_SYN,) — learned synaptic weights.
    """
    rng = np.random.default_rng(seed)
    W = np.full(N_SYN, 0.05, dtype=np.float32)  # weak prior

    # Build per-cell out-synapse index table for fast lookup
    out_syn_idx = [[] for _ in range(N_CELLS)]
    for s in range(N_SYN):
        out_syn_idx[PRE[s]].append(s)
    out_syn_idx = [np.array(lst, dtype=np.int32) for lst in out_syn_idx]

    A_plus, A_minus = 0.02, 0.012
    tau_plus, tau_minus = 3.0, 5.0  # in step units
    last_spike = -np.ones(N_CELLS, dtype=np.float32) * 1e6

    for walk in range(n_walks):
        # smooth random walk in (y,x) torus
        cy, cx = rng.uniform(0, GRID_H), rng.uniform(0, GRID_W)
        vy, vx = rng.normal(0, 0.6), rng.normal(0, 0.6)
        for t in range(walk_steps):
            cy = (cy + vy) % GRID_H
            cx = (cx + vx) % GRID_W
            vy += rng.normal(0, 0.05); vx += rng.normal(0, 0.05)
            vy = np.clip(vy, -1.0, 1.0); vx = np.clip(vx, -1.0, 1.0)
            # spike all cells in radius-1 ball around (cy,cx)
            spiking = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    iy = int(cy + dy) % GRID_H
                    ix = int(cx + dx) % GRID_W
                    spiking.append(iy * GRID_W + ix)
            spiking = np.array(spiking, dtype=np.int32)
            # STDP: for each spiking cell i, look at recent post-cells' last_spike
            for i in spiking:
                # i just fired now. Strengthen pre→i where pre fired recently
                # (pre-before-post = LTP); weaken i→post where post fired
                # recently (post-before-pre = LTD).
                outs = out_syn_idx[i]
                if outs.size == 0:
                    continue
                post_ids = POST[outs]
                dt_post = t - last_spike[post_ids]
                # LTD on post that already spiked recently (post-before-pre)
                ltd = dt_post >= 0
                ltd &= dt_post < 4 * tau_minus
                if ltd.any():
                    W[outs[ltd]] -= A_minus * np.exp(-dt_post[ltd] / tau_minus)
                # LTP: look at incoming synapses to i (i is post)
                # find synapses with POST == i — precompute is expensive; use
                # simpler approximation: strengthen all outs that point to a
                # cell which is ALSO spiking now (co-spike = path consolidation)
                co_spike = np.isin(post_ids, spiking)
                if co_spike.any():
                    W[outs[co_spike]] += A_plus
            last_spike[spiking] = t
        # clip
        np.clip(W, 0.0, 1.0, out=W)
    return W

def run_wave(W_syn, seed_cells, n_steps=20, decay=0.92, thr=0.55, gain=0.7):
    """Propagate activation through pre-trained substrate.

    Returns spike_raster shape (n_steps, N_CELLS) bool, and final Vb.
    """
    N = (PRE.max() + 1) if PRE.size else N_CELLS
    N = max(N, N_CELLS)
    Vb = np.full(N, V_REST, dtype=np.float32)
    Vb[seed_cells] = V_SPIKE_THR + 0.20
    raster = np.zeros((n_steps, N), dtype=bool)
    refrac = np.zeros(N, dtype=np.int8)
    for t in range(n_steps):
        spiking = (Vb >= thr) & (refrac == 0)
        raster[t] = spiking
        if spiking.any():
            active_syn_mask = spiking[PRE]
            contrib_post = POST[active_syn_mask]
            contrib_w = W_syn[active_syn_mask]
            inj = np.bincount(contrib_post, weights=contrib_w, minlength=N)
            Vb = V_REST + decay * (Vb - V_REST) + gain * inj.astype(np.float32)
            Vb[spiking] = V_RESET
            refrac[spiking] = 2
        refrac = np.maximum(refrac - 1, 0)
        np.clip(Vb, 0.0, 1.2, out=Vb)
    return raster, Vb

# ---------------------------------------------------------------------------
# Encoding / recall
# ---------------------------------------------------------------------------
KEY_DIM = 32
VAL_DIM = 32
N_ITEMS = 1000

def make_dataset(n=N_ITEMS, seed=42):
    rng = np.random.default_rng(seed)
    keys = rng.integers(0, 2, size=(n, KEY_DIM)).astype(np.float32)
    vals = rng.integers(0, 2, size=(n, VAL_DIM)).astype(np.float32)
    return keys, vals

def hash_key_to_seed_cells(key, n_seeds=12, seed_base=7):
    """Deterministic mapping key→entry-point cells on grid."""
    h = int(''.join(map(str, key.astype(int).tolist())), 2) if key.size <= 60 else hash(key.tobytes())
    rng = np.random.default_rng((h ^ seed_base) & 0xFFFFFFFF)
    return rng.choice(N_CELLS, size=n_seeds, replace=False)

def hash_item_to_location(item_idx, n_loc_cells=16, seed_base=13):
    """Each stored item gets a unique 'room' = a set of cells."""
    rng = np.random.default_rng((item_idx * 2654435761) ^ seed_base)
    return rng.choice(N_CELLS, size=n_loc_cells, replace=False)

def encode_items(keys, vals, W_syn):
    """Build location→value readout matrix (Hebbian binding).

    For each item i: drive substrate from key[i]→seed cells, let wave
    propagate, capture the spike-rate vector across all cells, then
    Hebbian-tie that pattern to value[i] at the assigned room cells.

    Returns: readout matrix M of shape (N_CELLS, VAL_DIM) — for each cell,
    the expected value bits when that cell is active.
    """
    M = np.zeros((N_CELLS, VAL_DIM), dtype=np.float32)
    counts = np.zeros(N_CELLS, dtype=np.float32)
    for i, (k, v) in enumerate(zip(keys, vals)):
        seeds = hash_key_to_seed_cells(k)
        raster, _ = run_wave(W_syn, seeds, n_steps=15)
        spike_rate = raster.sum(axis=0).astype(np.float32)
        # Boost the assigned room
        room = hash_item_to_location(i)
        spike_rate[room] += 8.0
        # Hebbian write to readout: for each active cell, accumulate v
        weight = spike_rate / (spike_rate.max() + 1e-6)
        M += weight[:, None] * (v * 2.0 - 1.0)[None, :]  # bipolar value
        counts += weight
    # normalize
    M /= (counts[:, None] + 1e-3)
    return M

def recall(key_query, W_syn, M, n_steps=15):
    seeds = hash_key_to_seed_cells(key_query)
    raster, _ = run_wave(W_syn, seeds, n_steps=n_steps)
    spike_rate = raster.sum(axis=0).astype(np.float32)
    weight = spike_rate / (spike_rate.max() + 1e-6)
    # readout: weighted sum over cells of M, then sign
    score = weight @ M  # (VAL_DIM,)
    return (score > 0).astype(np.float32)

def partial_cue(key, frac=0.25, rng=None):
    rng = rng or np.random.default_rng()
    k = key.copy()
    n_keep = max(1, int(frac * key.size))
    keep_idx = rng.choice(key.size, size=n_keep, replace=False)
    out = np.zeros_like(k)
    out[keep_idx] = k[keep_idx]
    return out

# ---------------------------------------------------------------------------
# Digital MLP baseline
# ---------------------------------------------------------------------------
def train_digital_mlp(keys, vals, hidden=128, epochs=80, lr=0.05):
    """One-hidden-layer MLP, numpy, sigmoid output. Trained on full keys."""
    rng = np.random.default_rng(123)
    W1 = rng.normal(0, 0.2, (KEY_DIM, hidden)).astype(np.float32)
    b1 = np.zeros(hidden, dtype=np.float32)
    W2 = rng.normal(0, 0.2, (hidden, VAL_DIM)).astype(np.float32)
    b2 = np.zeros(VAL_DIM, dtype=np.float32)
    X = keys; Y = vals
    for e in range(epochs):
        z1 = X @ W1 + b1
        h = np.tanh(z1)
        z2 = h @ W2 + b2
        p = 1.0 / (1.0 + np.exp(-z2))
        dz2 = (p - Y) / X.shape[0]
        dW2 = h.T @ dz2; db2 = dz2.sum(0)
        dh = dz2 @ W2.T
        dz1 = dh * (1 - h*h)
        dW1 = X.T @ dz1; db1 = dz1.sum(0)
        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2
    def predict(K):
        return (1.0 / (1.0 + np.exp(-(np.tanh(K @ W1 + b1) @ W2 + b2)))) > 0.5
    # MAC count per query: KEY_DIM*hidden + hidden*VAL_DIM
    macs = KEY_DIM*hidden + hidden*VAL_DIM
    return predict, macs

# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print(f"[N7d] Substrate: {N_CELLS} cells, {N_SYN} synapses, "
          f"{N_ITEMS} items, key={KEY_DIM}b, val={VAL_DIM}b")
    lut = IiiNetLUT()

    # 1. Pre-train substrate via STDP on spatial walks
    print("[N7d] Pre-training substrate with STDP on smooth walks...")
    t = time.time()
    W_real = pretrain_substrate(n_walks=200, walk_steps=120, lut=lut, seed=1)
    print(f"    real W stats: mean={W_real.mean():.3f} std={W_real.std():.3f} "
          f"nonzero={np.mean(W_real > 0.1):.3f}  [{time.time()-t:.1f}s]")

    # 2. Build randomized counter-substrate
    rng = np.random.default_rng(99)
    W_rand = rng.uniform(0, W_real.max(), size=N_SYN).astype(np.float32)
    # match overall stats
    W_rand *= W_real.mean() / (W_rand.mean() + 1e-9)

    # 3. Dataset
    keys, vals = make_dataset(N_ITEMS)

    # 4. Encode items into substrate (real and randomized)
    print("[N7d] Encoding 1000 items into real substrate...")
    t = time.time()
    M_real = encode_items(keys, vals, W_real)
    print(f"    [{time.time()-t:.1f}s]")
    print("[N7d] Encoding into randomized substrate...")
    t = time.time()
    M_rand = encode_items(keys, vals, W_rand)
    print(f"    [{time.time()-t:.1f}s]")

    # 5. Train digital MLP baseline
    print("[N7d] Training digital MLP...")
    digital_predict, digital_macs = train_digital_mlp(keys, vals)

    # 6. Test on full cues + partial cues at multiple fractions
    cue_fracs = [1.0, 0.50, 0.25, 0.10]
    test_idx = RNG.choice(N_ITEMS, size=200, replace=False)
    results = {}
    for frac in cue_fracs:
        n7d_correct = 0; rand_correct = 0; dig_correct = 0
        n7d_bits = 0; rand_bits = 0; dig_bits = 0
        for i in test_idx:
            k = keys[i]
            if frac < 1.0:
                k_cue = partial_cue(k, frac=frac, rng=np.random.default_rng(i))
            else:
                k_cue = k
            # NS-RAM real
            pred = recall(k_cue, W_real, M_real)
            n7d_bits += (pred == vals[i]).sum()
            if np.array_equal(pred, vals[i]):
                n7d_correct += 1
            # NS-RAM randomized
            pred_r = recall(k_cue, W_rand, M_rand)
            rand_bits += (pred_r == vals[i]).sum()
            if np.array_equal(pred_r, vals[i]):
                rand_correct += 1
            # Digital
            pred_d = digital_predict(k_cue[None, :].astype(np.float32))[0]
            dig_bits += (pred_d == vals[i]).sum()
            if np.array_equal(pred_d, vals[i]):
                dig_correct += 1
        n_test = test_idx.size
        results[f"cue_{frac:.2f}"] = {
            "nsram_real_exact":  n7d_correct / n_test,
            "nsram_rand_exact":  rand_correct / n_test,
            "digital_exact":     dig_correct / n_test,
            "nsram_real_bit":    n7d_bits / (n_test * VAL_DIM),
            "nsram_rand_bit":    rand_bits / (n_test * VAL_DIM),
            "digital_bit":       dig_bits / (n_test * VAL_DIM),
        }
        print(f"  cue={frac:.2f}: real={n7d_correct/n_test:.3f} "
              f"rand={rand_correct/n_test:.3f} dig={dig_correct/n_test:.3f}  "
              f"(bit: {n7d_bits/(n_test*VAL_DIM):.3f} / "
              f"{rand_bits/(n_test*VAL_DIM):.3f} / {dig_bits/(n_test*VAL_DIM):.3f})")

    # 7. Shuffle (C ablation): re-encode keys in random order — should not
    # matter for digital MLP, but might matter for NS-RAM (sequence
    # interference). Use bit accuracy at full cue.
    perm = np.random.default_rng(7).permutation(N_ITEMS)
    M_shuf = encode_items(keys[perm], vals[perm], W_real)
    shuf_bits = 0
    for i in test_idx:
        pred = recall(keys[i], W_real, M_shuf)
        shuf_bits += (pred == vals[i]).sum()
    shuf_bit_acc = shuf_bits / (test_idx.size * VAL_DIM)
    print(f"[N7d] Shuffled-encoding bit acc (real W) = {shuf_bit_acc:.3f}")
    results["shuffle_real_bit"] = float(shuf_bit_acc)

    # 8. Energy accounting (per recall query)
    # NS-RAM per recall: 15 steps × N_CELLS reads + 1 ADC per cell + VAL_DIM
    # decision compares
    e_nsram = 15 * N_CELLS * NSRAM_J_READ + N_CELLS * ADC_J_PER_SAMPLE + VAL_DIM * 1e-13
    e_dig = digital_macs * DIGITAL_MAC_J
    results["energy_per_recall_J"] = {
        "nsram_real": float(e_nsram),
        "digital_mlp": float(e_dig),
        "ratio_dig_over_nsram": float(e_dig / e_nsram),
    }
    print(f"[N7d] Energy/recall: NS-RAM={e_nsram:.3e} J, digital={e_dig:.3e} J, "
          f"ratio={e_dig/e_nsram:.2e}")

    # 9. Wave propagation raster for a single query (visualization data)
    seeds_demo = hash_key_to_seed_cells(keys[0])
    raster_real, _ = run_wave(W_real, seeds_demo, n_steps=15)
    raster_rand, _ = run_wave(W_rand, seeds_demo, n_steps=15)
    np.savez(OUT / "wave_data.npz",
             raster_real=raster_real, raster_rand=raster_rand,
             W_real_hist=np.histogram(W_real, bins=32)[0],
             W_rand_hist=np.histogram(W_rand, bins=32)[0])

    # try plotting
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(raster_real.astype(int), aspect="auto", cmap="hot")
        axes[0].set_title(f"Wave: real STDP substrate\nactive cells/step "
                          f"min={raster_real.sum(1).min()}, "
                          f"max={raster_real.sum(1).max()}")
        axes[0].set_xlabel("cell idx"); axes[0].set_ylabel("step")
        axes[1].imshow(raster_rand.astype(int), aspect="auto", cmap="hot")
        axes[1].set_title(f"Wave: randomized substrate\nactive cells/step "
                          f"min={raster_rand.sum(1).min()}, "
                          f"max={raster_rand.sum(1).max()}")
        axes[1].set_xlabel("cell idx")
        plt.tight_layout()
        plt.savefig(OUT / "wave_propagation.png", dpi=110)
        plt.close()
        print(f"[N7d] Saved wave_propagation.png")
    except Exception as e:
        print(f"[N7d] plot skipped: {e}")

    # 10. Wave stats
    results["wave_stats"] = {
        "real_total_spikes":  int(raster_real.sum()),
        "rand_total_spikes":  int(raster_rand.sum()),
        "real_unique_cells":  int(raster_real.any(0).sum()),
        "rand_unique_cells":  int(raster_rand.any(0).sum()),
    }

    # 11. Gates
    full = results["cue_1.00"]
    partial = results["cue_0.25"]
    infra_pass = full["nsram_real_bit"] >= 0.80
    hypothesis_delta_pp = (partial["nsram_real_bit"] - partial["digital_bit"]) * 100
    hypothesis_pass = hypothesis_delta_pp >= 15.0
    killshot_delta_pp = (partial["nsram_real_bit"] - partial["nsram_rand_bit"]) * 100
    killshot_triggered = abs(killshot_delta_pp) < 3.0  # randomized matches real
    results["gates"] = {
        "infra_pass": bool(infra_pass),
        "hypothesis_delta_pp_vs_digital_partial": float(hypothesis_delta_pp),
        "hypothesis_pass": bool(hypothesis_pass),
        "killshot_delta_pp_real_minus_rand": float(killshot_delta_pp),
        "killshot_triggered": bool(killshot_triggered),
    }

    # 12. Verdict
    if killshot_triggered:
        verdict = "DEAD_RETRACT — randomized pre-training matches real → wave dynamics do nothing."
    elif hypothesis_pass and infra_pass:
        verdict = "ALIVE — NS-RAM beats digital on partial cue, kill-shot avoided."
    elif infra_pass and not hypothesis_pass:
        verdict = "PARTIAL — infra works, but no partial-cue advantage vs digital."
    else:
        verdict = "DEAD — infrastructure didn't reach 80% recall."
    results["verdict"] = verdict
    print(f"\n[N7d] VERDICT: {verdict}\n")

    results["wall_time_s"] = time.time() - t0
    (OUT / "summary.json").write_text(json.dumps(results, indent=2))

    # ablation table markdown
    md = ["# DS-N7d Ablation Table",
          "",
          "## Recall accuracy by cue fraction (exact / bit)",
          "",
          "| cue | NS-RAM real | NS-RAM rand | Digital MLP |",
          "|-----|-------------|-------------|-------------|"]
    for frac in cue_fracs:
        r = results[f"cue_{frac:.2f}"]
        md.append(f"| {frac:.2f} | {r['nsram_real_exact']:.3f} / "
                  f"{r['nsram_real_bit']:.3f} | {r['nsram_rand_exact']:.3f} / "
                  f"{r['nsram_rand_bit']:.3f} | {r['digital_exact']:.3f} / "
                  f"{r['digital_bit']:.3f} |")
    md += ["",
           "## Shuffle ablation (full cue, bit acc)",
           f"- Ordered encoding (real W): {results['cue_1.00']['nsram_real_bit']:.3f}",
           f"- Shuffled encoding (real W): {results['shuffle_real_bit']:.3f}",
           "",
           "## Energy per recall",
           f"- NS-RAM real: {results['energy_per_recall_J']['nsram_real']:.3e} J",
           f"- Digital MLP: {results['energy_per_recall_J']['digital_mlp']:.3e} J",
           f"- Ratio (dig/NS-RAM): {results['energy_per_recall_J']['ratio_dig_over_nsram']:.2e}",
           "",
           "## Wave stats",
           f"- Real substrate: {results['wave_stats']['real_total_spikes']} spikes / "
           f"{results['wave_stats']['real_unique_cells']} unique cells",
           f"- Random substrate: {results['wave_stats']['rand_total_spikes']} spikes / "
           f"{results['wave_stats']['rand_unique_cells']} unique cells",
           "",
           "## Gates",
           f"- INFRA (full-cue bit ≥80%): {'PASS' if infra_pass else 'FAIL'} "
           f"({full['nsram_real_bit']:.3f})",
           f"- HYPOTHESIS (partial-cue Δ ≥15pp vs digital): "
           f"{'PASS' if hypothesis_pass else 'FAIL'} ({hypothesis_delta_pp:+.1f}pp)",
           f"- KILL-SHOT (rand matches real within 3pp): "
           f"{'TRIGGERED → RETRACT' if killshot_triggered else 'avoided'} "
           f"({killshot_delta_pp:+.1f}pp)",
           "",
           f"## Verdict",
           f"**{verdict}**"]
    (OUT / "ablation_table.md").write_text("\n".join(md))

    # final verdict
    (OUT / "final_verdict.md").write_text(
        f"# DS-N7d Final Verdict\n\n**{verdict}**\n\n"
        f"## One-sentence physics finding\n\n" +
        ("Wave propagation through pre-trained STDP substrate is the same as "
         "propagation through randomized weights (no causal role) — final "
         "death certificate for Memory Palace on NS-RAM.\n"
         if killshot_triggered else
         "Pre-trained substrate enables partial-cue completion that digital "
         "MLP cannot match, confirming wave-propagation dynamics as the "
         "operative physical feature.\n"
         if hypothesis_pass else
         "Infrastructure works but no partial-cue advantage emerged; dynamics "
         "are present but functionally equivalent to digital lookup.\n")
    )
    print(f"[N7d] Done in {time.time()-t0:.1f}s. Results in {OUT}")

if __name__ == "__main__":
    main()
