#!/usr/bin/env python3
"""
z1703: Unified GPU + DRAM Embodiment with Memory Decay
=======================================================
Combines REAL GPU thermal dynamics with SIMULATED DRAM charge decay
for unified embodied intelligence. The model must simultaneously manage:
  1. GPU thermal budget  -- real power cap causes DVFS throttling
  2. DRAM charge decay   -- stored memory fades over time, needs refresh

4 experimental conditions:
  A: Full Embodied   -- GPU real + DRAM simulated, model senses both
  B: GPU Only        -- Only GPU telemetry in FiLM, no DRAM awareness
  C: DRAM Only       -- Only DRAM charge in telemetry, no GPU awareness
  D: No Embodiment   -- BaselineTransformer, no conditioning

Training: TinyShakespeare char-level, 10 epochs, batch=4, seq=256.
Model action head controls DRAM refresh rate (action 0..3 = low..aggressive).

Verdict criteria:
  1. Full Embodied (A) combined_efficiency > GPU Only (B)
  2. Full Embodied (A) memory_loss < No Embodiment (D)
  3. DRAM-aware (A,C) reduces refresh events vs (B,D)
  4. Model learns non-random action distribution

Author: Claude + ikaros
Date: 2026-02-04
"""
import os, sys, time, json, math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
import numpy as np

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy')
from src.metabolic.film_transformer import (
    MetabolicTransformer, BaselineTransformer, MetabolicConfig,
    create_metabolic_transformer, get_best_device,
)
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, GpuSample
from src.actuation.gpu_actuator import GPUActuator

# ============================================================================
# Constants
# ============================================================================
PROJECT_ROOT = Path('/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy')
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results'
SHAKESPEARE_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
                   "master/data/tinyshakespeare/input.txt")

NUM_EPOCHS = 10
BATCH_SIZE = 4
SEQ_LEN = 256
LR = 3e-4
DECAY_INTERVAL = 5        # apply DRAM decay every N batches
MEMORY_LOSS_WEIGHT = 0.01
PRINT_EVERY = 50
COOLDOWN_S = 30

# ============================================================================
# DRAM Charge Model (simulated)
# ============================================================================
class DRAMChargeModel:
    """Simulated DRAM with analog charge decay -- models ideal DDR3."""
    def __init__(self, num_cells: int = 1024, decay_rate: float = 0.01):
        self.charges = np.ones(num_cells, dtype=np.float64)
        self.decay_rate = decay_rate
        self.refresh_threshold = 0.3
        self.num_cells = num_cells
        self.total_refreshes = 0
        self.total_data_lost = 0.0

    def step(self, dt_s: float = 0.1):
        """Exponential charge decay across all cells."""
        before = self.charges.copy()
        self.charges *= np.exp(-self.decay_rate * dt_s)
        self.total_data_lost += np.maximum(0, before - self.charges).sum()

    def write(self, address: int, value: float, strength: float = 1.0):
        self.charges[address % self.num_cells] = value * strength

    def read(self, address: int) -> float:
        return self.charges[address % self.num_cells]

    def refresh(self, addresses: Optional[np.ndarray] = None) -> int:
        """Refresh cells below threshold. Returns count refreshed."""
        if addresses is None:
            addresses = np.where(self.charges < self.refresh_threshold)[0]
        if len(addresses) == 0:
            return 0
        self.charges[addresses] = 1.0
        self.total_refreshes += len(addresses)
        return len(addresses)

    def get_stats(self) -> Dict:
        return {
            'mean_charge': float(self.charges.mean()),
            'charge_std': float(self.charges.std()),
            'below_threshold': int((self.charges < self.refresh_threshold).sum()),
            'zero_count': int((self.charges < 0.01).sum()),
            'total_refreshes': self.total_refreshes,
            'total_data_lost': float(self.total_data_lost),
        }

    def reset(self):
        self.charges[:] = 1.0
        self.total_refreshes = 0
        self.total_data_lost = 0.0

# ============================================================================
# Body state helpers
# ============================================================================
def build_body_state_32(gp: Optional[GpuSample], dram: DRAMChargeModel,
                        gh: List[Dict], dh: List[Dict],
                        t_sp: float = 60.0, p_bud: float = 50.0) -> np.ndarray:
    """Build 32-dim body state from GPU sample + DRAM model."""
    ds = dram.get_stats()
    # GPU (8)
    pw = gp.power_w if gp else 0.0
    te = gp.temp_edge_c if gp else 0.0
    tj = gp.temp_junction_c if gp else 0.0
    sc = float(gp.freq_sclk_mhz) if gp else 0.0
    mc = float(gp.freq_mclk_mhz) if gp else 0.0
    bu = gp.gpu_busy_pct if gp else 0.0
    vr = gp.vram_used_gb * 1024 if gp else 0.0
    # Derivatives (8)
    pd = td = fd = ud = cd = 0.0
    if len(gh) >= 2:
        dt = gh[-1]['t'] - gh[-2]['t']
        if dt > 0:
            pd = (gh[-1]['p'] - gh[-2]['p']) / dt
            td = (gh[-1]['tc'] - gh[-2]['tc']) / dt
            fd = (gh[-1]['f'] - gh[-2]['f']) / dt
            ud = (gh[-1]['u'] - gh[-2]['u']) / dt
    if len(dh) >= 2:
        dt = dh[-1]['t'] - dh[-2]['t']
        if dt > 0:
            cd = (dh[-1]['mc'] - dh[-2]['mc']) / dt
    # Homeostatic (8)
    thd = (te - t_sp) / 20.0
    pvd = (pw - p_bud) / max(p_bud, 1.0)
    mpr = 1.0 - ds['mean_charge']
    dpr = dram.decay_rate * 10.0
    return np.array([
        pw, te, tj, sc, mc, bu, vr, 0.0,                          # GPU (8)
        25.0, ds['mean_charge'], ds['charge_std'], dram.decay_rate,  # DRAM (4)
        0, int((dram.charges > 0.99).sum()), ds['zero_count'], 0.0,  # DRAM (4)
        pd, td, fd, ud, cd, 0.0, 0.0, 0.0,                        # derivs (8)
        thd, pvd, mpr, dpr,                                        # homeo (4)
        max(0.0, thd) + max(0.0, pvd), mpr + dpr, 1.0, 0.0,       # homeo (4)
    ], dtype=np.float32)


def body_to_telemetry(b: np.ndarray) -> np.ndarray:
    """Project 32-dim BodyState to 12-dim telemetry for MetabolicTransformer."""
    return np.array([
        b[0]/50.0, b[1]/100.0, b[3]/3000.0, b[5]/100.0,  # GPU normalised
        b[9], b[10],                                         # DRAM charge
        b[16]*10.0, b[17]*10.0, b[20]*10.0,                 # derivatives
        b[24], b[25], b[26],                                 # homeostatic
    ], dtype=np.float32)


def gpu_only_telemetry(b: np.ndarray) -> np.ndarray:
    """GPU-only: zero out DRAM-related channels."""
    t = body_to_telemetry(b)
    t[4] = t[5] = t[8] = t[11] = 0.0
    return t


def dram_only_telemetry(b: np.ndarray) -> np.ndarray:
    """DRAM-only: zero out GPU-related channels."""
    t = body_to_telemetry(b)
    t[0] = t[1] = t[2] = t[3] = t[6] = t[7] = t[9] = t[10] = 0.0
    return t

# ============================================================================
# Data loading
# ============================================================================
def load_tiny_shakespeare() -> str:
    """Download / cache TinyShakespeare, return raw text."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / 'tinyshakespeare.txt'
    if path.exists():
        return path.read_text()
    print("[data] Downloading TinyShakespeare ...")
    import urllib.request
    urllib.request.urlretrieve(SHAKESPEARE_URL, str(path))
    return path.read_text()


def make_batches(text: str, batch_size: int, seq_len: int,
                 device: torch.device) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Char-level batches: input[t] -> target[t+1]."""
    data = np.frombuffer(text.encode('utf-8'), dtype=np.uint8).astype(np.int64)
    total_tokens = batch_size * seq_len
    n_batches = (len(data) - 1) // total_tokens
    if n_batches < 1:
        raise ValueError("Not enough data for even one batch")
    data = data[: n_batches * total_tokens + 1]
    inputs = data[:-1].reshape(n_batches, batch_size, seq_len)
    targets = data[1:].reshape(n_batches, batch_size, seq_len)
    return [(torch.from_numpy(inputs[i]).to(device),
             torch.from_numpy(targets[i]).to(device))
            for i in range(n_batches)]

# ============================================================================
# Training loop for one condition
# ============================================================================
REFRESH_FRACTIONS = [0.0, 0.05, 0.15, 0.40]  # action 0..3


def run_condition(condition: str,
                  batches: List[Tuple[torch.Tensor, torch.Tensor]],
                  device: torch.device,
                  telemetry: Optional[SysfsHwmonTelemetry],
                  actuator: Optional[GPUActuator]) -> Dict:
    """Train one condition and return metrics."""
    print(f"\n{'='*60}\n  CONDITION {condition}\n{'='*60}")

    is_baseline = (condition == 'D')
    model = create_metabolic_transformer(
        hidden_dim=256, num_layers=6, num_heads=4,
        telemetry_dim=12, baseline=is_baseline,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    dram = DRAMChargeModel(num_cells=1024, decay_rate=0.01)

    telem_fn = {'A': body_to_telemetry, 'B': gpu_only_telemetry,
                'C': dram_only_telemetry}.get(condition, None)

    gh: List[Dict] = []   # gpu history
    dh: List[Dict] = []   # dram history
    epoch_metrics: List[Dict] = []
    all_actions: List[int] = []
    throttle_count = 0
    total_tokens = 0
    total_energy_j = 0.0

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}  Batches/epoch: {len(batches)}")

    for epoch in range(NUM_EPOCHS):
        model.train()
        ep_loss = ep_mem = 0.0
        ep_tokens = 0
        ep_refreshes = 0
        ep_start = time.time()
        e_start_j = total_energy_j

        for bi, (x, y) in enumerate(batches):
            # --- Sense ---
            gpu_sample = None
            if telemetry:
                try:
                    gpu_sample = telemetry.read_sample()
                except Exception:
                    pass

            now = time.time()
            if gpu_sample:
                gh.append({'t': now, 'p': gpu_sample.power_w,
                           'tc': gpu_sample.temp_edge_c,
                           'f': float(gpu_sample.freq_sclk_mhz),
                           'u': gpu_sample.gpu_busy_pct})
                if len(gh) > 20:
                    gh.pop(0)
                dt = (gh[-1]['t'] - gh[-2]['t']) if len(gh) >= 2 else 0.02
                total_energy_j += gpu_sample.power_w * dt
                if gpu_sample.temp_edge_c > 85:
                    throttle_count += 1

            dh.append({'t': now, 'mc': dram.get_stats()['mean_charge']})
            if len(dh) > 20:
                dh.pop(0)

            body = build_body_state_32(gpu_sample, dram, gh, dh)

            # --- Build telemetry tensor ---
            if telem_fn is not None:
                tel = torch.from_numpy(telem_fn(body)).float().unsqueeze(0).to(device)
                tel = tel.expand(BATCH_SIZE, -1)
            else:
                tel = None

            # --- Forward ---
            output = model(x, telemetry=tel)
            logits = output['logits']
            task_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            # Memory loss: penalise data lost to decay
            ds = dram.get_stats()
            mlv = ds['total_data_lost'] / max(dram.num_cells, 1)
            loss = task_loss + MEMORY_LOSS_WEIGHT * torch.tensor(mlv, device=device)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # --- Act: use action head to decide refresh intensity ---
            with torch.no_grad():
                ap = F.softmax(output['action_logits'], dim=-1)
                act = torch.argmax(ap.mean(dim=0)).item()
                all_actions.append(act)

            frac = REFRESH_FRACTIONS[act]
            if frac > 0:
                n_ref = max(1, int(frac * dram.num_cells))
                weakest = np.argsort(dram.charges)[:n_ref]
                ep_refreshes += dram.refresh(weakest)

            # --- DRAM decay every DECAY_INTERVAL batches ---
            if (bi + 1) % DECAY_INTERVAL == 0:
                dram.step(dt_s=0.5)

            ep_loss += task_loss.item()
            ep_mem += mlv
            ep_tokens += BATCH_SIZE * SEQ_LEN
            total_tokens += BATCH_SIZE * SEQ_LEN

            if (bi + 1) % PRINT_EVERY == 0:
                ppl = math.exp(min(ep_loss / (bi + 1), 20))
                gt = gpu_sample.temp_edge_c if gpu_sample else 0
                gw = gpu_sample.power_w if gpu_sample else 0
                print(f"  [E{epoch+1} B{bi+1:>4d}] loss={task_loss.item():.3f} "
                      f"ppl={ppl:.1f} mc={ds['mean_charge']:.3f} act={act} "
                      f"T={gt:.0f}C P={gw:.1f}W")

        # Epoch summary
        ep_s = time.time() - ep_start
        avg_loss = ep_loss / len(batches)
        ppl = math.exp(min(avg_loss, 20))
        avg_mem = ep_mem / len(batches)
        ep_energy = total_energy_j - e_start_j
        jpt = ep_energy / max(ep_tokens, 1)
        ceff = 1.0 / (ppl * max(jpt, 1e-9) * (1.0 + avg_mem))

        em = {'epoch': epoch+1, 'loss': round(avg_loss, 4),
              'perplexity': round(ppl, 2), 'j_per_token': round(jpt, 8),
              'mean_charge': round(dram.get_stats()['mean_charge'], 4),
              'refresh_events': ep_refreshes,
              'memory_loss': round(avg_mem, 6),
              'combined_efficiency': round(ceff, 4),
              'epoch_time_s': round(ep_s, 2),
              'throttle_events': throttle_count}
        epoch_metrics.append(em)
        print(f"  Epoch {epoch+1}: ppl={ppl:.2f} j/tok={jpt:.6f} "
              f"mc={em['mean_charge']:.3f} refresh={ep_refreshes} "
              f"mem_loss={avg_mem:.4f} comb_eff={ceff:.4f}")

    # Action distribution analysis
    ad = Counter(all_actions)
    total_act = len(all_actions)
    act_pcts = {str(k): round(v/total_act*100, 1) for k, v in sorted(ad.items())}
    probs_arr = np.array([ad.get(i, 0) for i in range(4)], dtype=np.float64)
    probs_arr /= probs_arr.sum()
    entropy = float(-sum(p * np.log2(p + 1e-12) for p in probs_arr))
    non_random = entropy < 1.8  # < 90% of uniform entropy (2.0 bits)

    final = epoch_metrics[-1]
    return {
        'condition': condition,
        'final_perplexity': final['perplexity'],
        'final_j_per_token': final['j_per_token'],
        'final_mean_charge': final['mean_charge'],
        'final_memory_loss': final['memory_loss'],
        'final_combined_efficiency': final['combined_efficiency'],
        'total_refresh_events': sum(e['refresh_events'] for e in epoch_metrics),
        'total_throttle_events': throttle_count,
        'action_distribution_pct': act_pcts,
        'action_entropy_bits': round(entropy, 3),
        'non_random_actions': non_random,
        'total_tokens': total_tokens,
        'total_energy_j': round(total_energy_j, 4),
        'num_parameters': num_params,
        'epoch_metrics': epoch_metrics,
    }

# ============================================================================
# Verdict logic
# ============================================================================
def evaluate_verdicts(results: Dict[str, Dict]) -> List[Dict]:
    A, B, C, D = [results.get(k, {}) for k in 'ABCD']
    verdicts = []

    # 1. Full Embodied > GPU Only in combined efficiency
    av, bv = A.get('final_combined_efficiency', 0), B.get('final_combined_efficiency', 0)
    verdicts.append({'id': 1,
        'test': 'Full Embodied (A) combined_efficiency > GPU Only (B)',
        'A_val': av, 'B_val': bv, 'PASS': av > bv})

    # 2. Full Embodied memory_loss < No Embodiment
    am, dm = A.get('final_memory_loss', 1), D.get('final_memory_loss', 0)
    verdicts.append({'id': 2,
        'test': 'Full Embodied (A) memory_loss < No Embodiment (D)',
        'A_val': am, 'D_val': dm, 'PASS': am < dm})

    # 3. DRAM-aware (A,C) reduces refresh events vs (B,D)
    aware = (A.get('total_refresh_events', 0) + C.get('total_refresh_events', 0)) / 2.0
    unaware = (B.get('total_refresh_events', 0) + D.get('total_refresh_events', 0)) / 2.0
    verdicts.append({'id': 3,
        'test': 'DRAM-aware (A,C) fewer refresh events than (B,D)',
        'aware_mean': round(aware, 1), 'unaware_mean': round(unaware, 1),
        'PASS': aware < unaware})

    # 4. Non-random action distribution
    nr = any(results.get(k, {}).get('non_random_actions', False) for k in 'ABC')
    verdicts.append({'id': 4,
        'test': 'Model learns non-random action distribution',
        'A_entropy': A.get('action_entropy_bits', 2.0),
        'B_entropy': B.get('action_entropy_bits', 2.0),
        'C_entropy': C.get('action_entropy_bits', 2.0),
        'PASS': nr})

    return verdicts

# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print("  z1703: Unified GPU + DRAM Embodiment")
    print("  Memory Consolidation Under Energy Pressure")
    print("=" * 60)

    device = get_best_device()
    print(f"Device: {device}")

    # Hardware init
    telemetry = actuator = None
    try:
        telemetry = SysfsHwmonTelemetry(sample_rate_hz=50.0)
        s = telemetry.read_sample()
        print(f"GPU telemetry OK: {s.power_w:.1f}W, {s.temp_edge_c:.0f}C")
    except Exception as e:
        print(f"GPU telemetry unavailable: {e} (will run without)")

    try:
        actuator = GPUActuator(card_id=0)
        st = actuator.get_current_state()
        print(f"GPU actuator OK: cap={st.power_cap_w:.0f}W")
    except Exception as e:
        print(f"GPU actuator unavailable: {e}")

    # Data
    print("\nLoading TinyShakespeare ...")
    text = load_tiny_shakespeare()
    print(f"  {len(text):,} chars")
    batches = make_batches(text, BATCH_SIZE, SEQ_LEN, device)
    print(f"  {len(batches)} batches of {BATCH_SIZE}x{SEQ_LEN}")

    # Run conditions
    conditions = ['A', 'B', 'C', 'D']
    results: Dict[str, Dict] = {}
    t0 = time.time()

    for ci, cond in enumerate(conditions):
        if ci > 0:
            print(f"\n--- Cooldown {COOLDOWN_S}s before condition {cond} ---")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(COOLDOWN_S)
        results[cond] = run_condition(cond, batches, device, telemetry, actuator)

    total_time = time.time() - t0

    # Restore GPU
    if actuator:
        try:
            actuator.restore_initial_state()
            print("\nGPU state restored.")
        except Exception as e:
            print(f"\nWarning: could not restore GPU state: {e}")

    # Verdicts
    verdicts = evaluate_verdicts(results)
    n_pass = sum(1 for v in verdicts if v['PASS'])

    print(f"\n{'='*60}\n  VERDICTS  ({n_pass}/{len(verdicts)} PASS)\n{'='*60}")
    for v in verdicts:
        tag = "PASS" if v['PASS'] else "FAIL"
        print(f"  [{tag}] #{v['id']}: {v['test']}")
        for k, val in v.items():
            if k not in ('id', 'test', 'PASS'):
                print(f"         {k} = {val}")

    # Summary table
    print(f"\n{'='*60}\n  SUMMARY TABLE\n{'='*60}")
    hdr = f"  {'Cond':<5} {'PPL':>7} {'J/tok':>10} {'MC':>6} {'MemLoss':>8} {'CombEff':>9} {'Refr':>7}"
    print(hdr)
    print(f"  {'-'*5} {'-'*7} {'-'*10} {'-'*6} {'-'*8} {'-'*9} {'-'*7}")
    for c in conditions:
        r = results[c]
        print(f"  {c:<5} {r['final_perplexity']:>7.2f} "
              f"{r['final_j_per_token']:>10.6f} "
              f"{r['final_mean_charge']:>6.3f} "
              f"{r['final_memory_loss']:>8.4f} "
              f"{r['final_combined_efficiency']:>9.4f} "
              f"{r['total_refresh_events']:>7d}")
    for c in conditions:
        r = results[c]
        print(f"  {c} actions: {r['action_distribution_pct']}  "
              f"H={r['action_entropy_bits']:.2f} bits  "
              f"non-random={r['non_random_actions']}")

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    output = {
        'experiment': 'z1703_unified_gpu_dram',
        'description': 'Unified GPU thermal + DRAM charge decay embodiment',
        'device': str(device),
        'config': {
            'num_epochs': NUM_EPOCHS, 'batch_size': BATCH_SIZE,
            'seq_len': SEQ_LEN, 'lr': LR,
            'decay_interval': DECAY_INTERVAL,
            'memory_loss_weight': MEMORY_LOSS_WEIGHT,
            'dram_num_cells': 1024, 'dram_decay_rate': 0.01,
        },
        'conditions': {c: results[c] for c in conditions},
        'verdicts': verdicts,
        'num_pass': n_pass, 'num_total': len(verdicts),
        'total_time_s': round(total_time, 1),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    out_path = RESULTS_DIR / 'z1703_unified_gpu_dram.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    print(f"Total experiment time: {total_time/60:.1f} min")


if __name__ == '__main__':
    main()
