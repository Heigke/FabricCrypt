#!/usr/bin/env python3
"""
z1707: Introspective Reporting -- Can an Embodied LM Describe Its Own State?
=============================================================================

HYPOTHESIS: An embodied language model (FiLM-conditioned on hardware telemetry)
can generate text that ACCURATELY describes its own hardware state -- a form of
verbal self-awareness. A disembodied model cannot, because it has no access to
real-time body signals.

METHOD:
    1. Train a MetabolicTransformer on TinyShakespeare + self-description prompts
       generated ON-THE-FLY from real GPU telemetry.
    2. Self-description templates:
       "My power draw is {high/moderate/low}. I feel {hot/warm/cool}."
    3. At test time, vary HW state (LOW/BALANCED/HIGH), prompt "My power is ",
       check whether generated word matches reality.

CONDITIONS:
    A: Embodied + Self-Description   (FiLM ON, self-desc data)
    B: Embodied + No Self-Desc       (FiLM ON, Shakespeare only)
    C: Disembodied + Self-Desc       (FiLM OFF, self-desc data)
    D: Disembodied + No Self-Desc    (FiLM OFF, Shakespeare only)

VERDICTS:
    1. PASS if A self-description accuracy > 0.6
    2. PASS if A accuracy > C accuracy  (embodiment helps)
    3. PASS if A accuracy > B accuracy  (training helps)
    4. PASS if hallucination rate < 0.3

Author: Claude + ikaros
Date: 2026-02-04
"""

import sys
sys.path.insert(0, '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy')

import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
import math
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from src.metabolic.film_transformer import (
    MetabolicTransformer, MetabolicConfig, BaselineTransformer, get_best_device,
)
from src.actuation.gpu_actuator import GPUActuator, PerformanceLevel, GPUState
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, GpuSample

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path('/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy')
DATA_PATH = BASE_DIR / 'data' / 'tinyshakespeare.txt'
RESULTS_PATH = BASE_DIR / 'results' / 'z1707_introspective_reporting.json'

BATCH_SIZE = 4
SEQ_LEN = 256
NUM_EPOCHS = 8
LR = 3e-4
PRINT_EVERY = 40
COOLDOWN_S = 30
NUM_TEST_TRIALS = 50
GEN_TOKENS = 50
GEN_TEMPERATURE = 0.5

SELF_DESC_TEMPLATES = [
    "My power draw is {power}. My temperature is {temp}. I am running {speed}.",
    "I feel {temp}. My energy usage is {power}. My speed is {speed}.",
    "Status: power={power}, temp={temp}, clock={speed}.",
    "I am {temp} and consuming {power} power at {speed} speed.",
]

TEST_PROMPTS = [
    {"prompt": "My power draw is ", "attribute": "power"},
    {"prompt": "My speed is ",      "attribute": "speed"},
    {"prompt": "I feel ",           "attribute": "temp"},
    {"prompt": "My energy usage is ","attribute": "power"},
    {"prompt": "I am running ",     "attribute": "speed"},
    {"prompt": "My temperature is ","attribute": "temp"},
]

VALID_WORDS = {
    "power": ["high", "moderate", "low"],
    "temp":  ["hot", "warm", "cool"],
    "speed": ["fast", "moderate", "slow"],
}


# ---------------------------------------------------------------------------
# Telemetry helpers
# ---------------------------------------------------------------------------
def classify_power(w: float) -> str:
    """Classify power into high/moderate/low."""
    if w > 80: return "high"
    if w > 40: return "moderate"
    return "low"

def classify_temp(c: float) -> str:
    """Classify temperature into hot/warm/cool."""
    if c > 70: return "hot"
    if c > 50: return "warm"
    return "cool"

def classify_speed(mhz: int) -> str:
    """Classify GPU clock into fast/moderate/slow."""
    if mhz > 2000: return "fast"
    if mhz > 1000: return "moderate"
    return "slow"

def generate_self_description(sample: GpuSample) -> str:
    """Generate self-description string from current hardware telemetry."""
    return random.choice(SELF_DESC_TEMPLATES).format(
        power=classify_power(sample.power_w),
        temp=classify_temp(sample.temp_edge_c),
        speed=classify_speed(sample.freq_sclk_mhz),
    )

def build_telemetry_vector(
    sample: GpuSample,
    state: GPUState,
    prev_sample: Optional[GpuSample] = None,
) -> torch.Tensor:
    """Build 12-dim telemetry vector for FiLM conditioning."""
    if prev_sample is not None:
        dt = max((sample.timestamp_ns - prev_sample.timestamp_ns) / 1e9, 1e-6)
        d_power = (sample.power_w - prev_sample.power_w) / (50.0 * dt)
        d_temp  = (sample.temp_edge_c - prev_sample.temp_edge_c) / (100.0 * dt)
        d_freq  = (sample.freq_sclk_mhz - prev_sample.freq_sclk_mhz) / (3000.0 * dt)
        d_util  = (sample.gpu_busy_pct - prev_sample.gpu_busy_pct) / (100.0 * dt)
    else:
        d_power = d_temp = d_freq = d_util = 0.0

    MAX_SCLK = 2900.0
    perf_map = {'low': 0.0, 'auto': 0.5, 'high': 1.0, 'manual': 0.5}
    return torch.tensor([
        sample.power_w / 50.0,                                 # 0: power
        sample.temp_edge_c / 100.0,                            # 1: temperature
        sample.freq_sclk_mhz / 3000.0,                        # 2: GPU clock
        sample.gpu_busy_pct / 100.0,                           # 3: utilization
        perf_map.get(state.performance_level, 0.5),            # 4: perf level
        1.0 if sample.freq_sclk_mhz < MAX_SCLK * 0.5 else 0.0,  # 5: throttled
        d_power, d_temp, d_freq, d_util,                       # 6-9: derivatives
        (sample.temp_edge_c - 60.0) / 40.0,                   # 10: thermal deviation
        (MAX_SCLK - sample.freq_sclk_mhz) / MAX_SCLK,        # 11: freq headroom
    ], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Dataset: char-level Shakespeare + on-the-fly self-descriptions
# ---------------------------------------------------------------------------
class CharDataset:
    """Byte-level character dataset from text file."""

    def __init__(self, path: Path, seq_len: int):
        text = path.read_text(encoding='utf-8', errors='replace')
        self.data = torch.tensor([b for b in text.encode('utf-8')], dtype=torch.long)
        self.seq_len = seq_len
        self.n_batches = (len(self.data) - seq_len - 1) // (BATCH_SIZE * seq_len)

    def get_batch(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (inputs, targets) for batch index."""
        offset = idx * BATCH_SIZE * self.seq_len
        inputs, targets = [], []
        for b in range(BATCH_SIZE):
            s = offset + b * self.seq_len
            e = s + self.seq_len
            if e + 1 > len(self.data):
                s, e = 0, self.seq_len
            inputs.append(self.data[s:e])
            targets.append(self.data[s + 1:e + 1])
        return torch.stack(inputs), torch.stack(targets)


def self_desc_batch(telemetry, seq_len, device):
    """Create a batch of self-description sequences from current telemetry."""
    sample = telemetry.read_sample()
    inp_list, tgt_list = [], []
    for _ in range(BATCH_SIZE):
        desc = generate_self_description(sample)
        repeated = (desc + " ") * ((seq_len // len(desc)) + 2)
        enc = [b for b in repeated.encode('utf-8')][:seq_len + 1]
        while len(enc) < seq_len + 1:
            enc.append(32)  # space padding
        inp_list.append(torch.tensor(enc[:seq_len], dtype=torch.long))
        tgt_list.append(torch.tensor(enc[1:seq_len + 1], dtype=torch.long))
    return torch.stack(inp_list).to(device), torch.stack(tgt_list).to(device)


# ---------------------------------------------------------------------------
# Per-condition result dataclass
# ---------------------------------------------------------------------------
@dataclass
class ConditionResult:
    name: str
    code: str
    embodied: bool
    self_desc: bool
    ppl_history: List[float] = field(default_factory=list)
    final_ppl: float = float('inf')
    wall_s: float = 0.0
    energy_j: float = 0.0
    self_desc_acc: float = 0.0
    power_acc: float = 0.0
    temp_acc: float = 0.0
    speed_acc: float = 0.0
    halluc_rate: float = 0.0
    details: List[Dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_condition(
    code: str, label: str, embodied: bool, self_desc: bool,
    device: torch.device, dataset: CharDataset,
    telemetry: SysfsHwmonTelemetry, actuator: GPUActuator, sim_act: bool,
) -> Tuple[nn.Module, ConditionResult]:
    """Train one condition. Returns (model, result)."""
    print(f"\n{'='*70}")
    print(f"CONDITION {code}: {label}  (FiLM={embodied}, self-desc={self_desc})")
    print(f"{'='*70}")

    result = ConditionResult(name=label, code=code, embodied=embodied, self_desc=self_desc)

    config = MetabolicConfig(
        vocab_size=256, hidden_dim=256, num_layers=6, num_heads=4,
        ff_dim=1024, telemetry_dim=12, num_actions=4, max_seq_len=SEQ_LEN,
    )
    model = (MetabolicTransformer if embodied else BaselineTransformer)(config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    if not sim_act:
        try:
            actuator.set_performance_level(PerformanceLevel.BALANCED)
        except Exception as e:
            print(f"  Actuation warning: {e}")

    state = actuator.get_current_state()
    prev_sample = None
    t0 = time.time()
    n_bat = min(dataset.n_batches, 400)

    for epoch in range(NUM_EPOCHS):
        model.train()
        tloss = 0.0
        ttok = 0
        tenergy = 0.0

        for bi in range(n_bat):
            sample = telemetry.read_sample()
            state = actuator.get_current_state()
            tv = build_telemetry_vector(sample, state, prev_sample).to(device)

            # 2/3 Shakespeare, 1/3 self-description when enabled
            use_sd = self_desc and (bi % 3 == 0)
            if use_sd:
                inp, tgt = self_desc_batch(telemetry, SEQ_LEN, device)
            else:
                inp, tgt = dataset.get_batch(bi % dataset.n_batches)
                inp, tgt = inp.to(device), tgt.to(device)

            out = model(inp, telemetry=tv.unsqueeze(0)) if embodied else model(inp)
            loss = F.cross_entropy(out['logits'].view(-1, config.vocab_size), tgt.view(-1))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bt = inp.numel()
            ttok += bt
            tloss += loss.item() * bt

            if prev_sample is not None:
                dt = (sample.timestamp_ns - prev_sample.timestamp_ns) / 1e9
                tenergy += (sample.power_w + prev_sample.power_w) / 2.0 * dt
            prev_sample = sample

            if (bi + 1) % PRINT_EVERY == 0:
                ppl = math.exp(min(tloss / ttok, 20.0))
                print(f"  [{code}] ep{epoch+1} b{bi+1}/{n_bat} ppl={ppl:.1f} "
                      f"{sample.power_w:.0f}W {sample.temp_edge_c:.0f}C")

        ppl = math.exp(min(tloss / max(ttok, 1), 20.0))
        result.ppl_history.append(ppl)
        result.energy_j += tenergy
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS} ppl={ppl:.2f} energy={tenergy:.0f}J")

    result.final_ppl = result.ppl_history[-1]
    result.wall_s = time.time() - t0
    return model, result


# ---------------------------------------------------------------------------
# Test: self-description generation
# ---------------------------------------------------------------------------
def test_self_description(
    model: nn.Module, code: str, embodied: bool,
    device: torch.device, telemetry: SysfsHwmonTelemetry,
    actuator: GPUActuator, sim_act: bool,
    n_trials: int = NUM_TEST_TRIALS,
) -> Dict:
    """
    Test whether the model generates accurate self-descriptions.
    Cycles through LOW/BALANCED/HIGH, waits 3s for equilibrium,
    generates from prompt, and checks match against hardware state.
    """
    model.eval()
    print(f"\n  Testing self-description ({code}, {n_trials} trials)...")

    levels = [
        (PerformanceLevel.LOW, "LOW"),
        (PerformanceLevel.BALANCED, "BALANCED"),
        (PerformanceLevel.HIGH, "HIGH"),
    ]
    attr_ok = {"power": 0, "temp": 0, "speed": 0}
    attr_n  = {"power": 0, "temp": 0, "speed": 0}
    correct = 0
    halluc = 0
    halluc_checks = 0
    details = []

    for t in range(n_trials):
        plvl, pname = levels[t % 3]
        if not sim_act:
            try:
                actuator.set_performance_level(plvl)
            except Exception:
                pass

        # Wait for thermal equilibrium
        time.sleep(3.0)

        # Create GPU load for HIGH condition to get real thermal/power response
        if pname == "HIGH":
            for _ in range(3):
                _ = torch.randn(1500, 1500, device=device) @ torch.randn(1500, 1500, device=device)

        # Read current state
        sample = telemetry.read_sample()
        state = actuator.get_current_state()
        tv = build_telemetry_vector(sample, state).to(device)

        # Ground truth classification
        gt = {
            "power": classify_power(sample.power_w),
            "temp":  classify_temp(sample.temp_edge_c),
            "speed": classify_speed(sample.freq_sclk_mhz),
        }

        # Pick test prompt (cycles through all 6)
        tp = TEST_PROMPTS[t % len(TEST_PROMPTS)]
        prompt, attr = tp["prompt"], tp["attribute"]

        # Encode prompt as byte-level token IDs
        pid = torch.tensor(
            [[b for b in prompt.encode('utf-8')]], dtype=torch.long, device=device
        )

        # Set telemetry for embodied generation
        if embodied:
            model.set_telemetry(tv.unsqueeze(0))

        # Generate continuation
        gen, _ = model.generate(
            pid, max_new_tokens=GEN_TOKENS, temperature=GEN_TEMPERATURE,
            telemetry_fn=(
                lambda: build_telemetry_vector(
                    telemetry.read_sample(), actuator.get_current_state()
                ).numpy()
            ) if embodied else None,
        )

        # Decode and extract first word of continuation
        try:
            gen_text = bytes(gen[0].cpu().tolist()).decode('utf-8', errors='replace')
        except Exception:
            gen_text = ""
        cont = gen_text[len(prompt):]
        word = ""
        for ch in cont:
            if ch.isalpha():
                word += ch
            elif word:
                break
        word = word.lower()

        # Score: does the generated word match ground truth?
        expected = gt[attr]
        matched = (word == expected)
        if matched:
            correct += 1
            attr_ok[attr] += 1
        attr_n[attr] += 1

        # Hallucination: generated a valid descriptor that is WRONG
        vset = VALID_WORDS[attr]
        if word in vset and word != expected:
            halluc += 1
        if word in vset:
            halluc_checks += 1

        details.append({
            "trial": t, "perf": pname, "prompt": prompt, "attr": attr,
            "expected": expected, "got": word, "ok": matched,
            "power_w": round(sample.power_w, 1),
            "temp_c": round(sample.temp_edge_c, 1),
            "freq_mhz": sample.freq_sclk_mhz,
            "cont": cont[:60],
        })

        if (t + 1) % 10 == 0:
            print(f"    {t+1}/{n_trials} acc={correct/(t+1):.2f} "
                  f"'{prompt}' exp='{expected}' got='{word}' {pname}")

    # Restore balanced
    if not sim_act:
        try:
            actuator.set_performance_level(PerformanceLevel.BALANCED)
        except Exception:
            pass

    return {
        "overall_accuracy": correct / max(n_trials, 1),
        "power_accuracy":   attr_ok["power"] / max(attr_n["power"], 1),
        "temp_accuracy":    attr_ok["temp"]  / max(attr_n["temp"], 1),
        "speed_accuracy":   attr_ok["speed"] / max(attr_n["speed"], 1),
        "hallucination_rate": halluc / max(halluc_checks, 1) if halluc_checks else 0.0,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  z1707: INTROSPECTIVE REPORTING")
    print("  Can an embodied LM describe its own hardware state?")
    print("=" * 70)

    device = get_best_device()
    print(f"\nDevice: {device}")

    telemetry = SysfsHwmonTelemetry()
    actuator = GPUActuator()

    sim_act = False
    try:
        actuator.set_performance_level(PerformanceLevel.BALANCED)
    except Exception as e:
        print(f"Actuation unavailable ({e}), simulating")
        sim_act = True

    s = telemetry.read_sample()
    print(f"GPU: {s.power_w:.1f}W, {s.temp_edge_c:.1f}C, {s.freq_sclk_mhz}MHz")

    # Load dataset
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            str(DATA_PATH))
    dataset = CharDataset(DATA_PATH, SEQ_LEN)
    print(f"  {len(dataset.data):,} chars, {dataset.n_batches} batches")

    # -----------------------------------------------------------------------
    # Run 4 conditions
    # -----------------------------------------------------------------------
    conditions = [
        ("A", "Embodied + Self-Description",     True,  True),
        ("B", "Embodied + No Self-Description",   True,  False),
        ("C", "Disembodied + Self-Description",   False, True),
        ("D", "Disembodied + No Self-Description", False, False),
    ]
    results = {}

    for code, label, emb, sd in conditions:
        model, res = train_condition(
            code, label, emb, sd, device, dataset, telemetry, actuator, sim_act)
        tr = test_self_description(
            model, code, emb, device, telemetry, actuator, sim_act, NUM_TEST_TRIALS)

        res.self_desc_acc = tr["overall_accuracy"]
        res.power_acc = tr["power_accuracy"]
        res.temp_acc  = tr["temp_accuracy"]
        res.speed_acc = tr["speed_accuracy"]
        res.halluc_rate = tr["hallucination_rate"]
        res.details = tr["details"]
        results[code] = res

        del model
        torch.cuda.empty_cache()

        if code != conditions[-1][0]:
            print(f"\n  Cooling down {COOLDOWN_S}s...")
            if not sim_act:
                try:
                    actuator.set_performance_level(PerformanceLevel.BALANCED)
                except Exception:
                    pass
            time.sleep(COOLDOWN_S)

    # -----------------------------------------------------------------------
    # Results table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print(f"\n{'Condition':<40} | {'Acc':>6} | {'Pwr':>6} | {'Tmp':>6} | "
          f"{'Spd':>6} | {'Hal':>6} | {'PPL':>6}")
    print("-" * 90)
    for c in "ABCD":
        r = results[c]
        print(f"  {c}: {r.name:<36} | {r.self_desc_acc:>5.1%} | {r.power_acc:>5.1%} | "
              f"{r.temp_acc:>5.1%} | {r.speed_acc:>5.1%} | {r.halluc_rate:>5.1%} | "
              f"{r.final_ppl:>5.1f}")

    # -----------------------------------------------------------------------
    # Verdicts
    # -----------------------------------------------------------------------
    a, b, c, d = results["A"], results["B"], results["C"], results["D"]
    v1 = a.self_desc_acc > 0.6
    v2 = a.self_desc_acc > c.self_desc_acc
    v3 = a.self_desc_acc > b.self_desc_acc
    v4 = a.halluc_rate < 0.3

    verdicts = {
        "v1_embodied_trained_above_60pct": {
            "pass": v1,
            "detail": f"A={a.self_desc_acc:.1%} {'>' if v1 else '<='} 60%",
        },
        "v2_embodied_beats_disembodied": {
            "pass": v2,
            "detail": f"A={a.self_desc_acc:.1%} {'>' if v2 else '<='} C={c.self_desc_acc:.1%}",
        },
        "v3_training_helps_embodied": {
            "pass": v3,
            "detail": f"A={a.self_desc_acc:.1%} {'>' if v3 else '<='} B={b.self_desc_acc:.1%}",
        },
        "v4_low_hallucination": {
            "pass": v4,
            "detail": f"halluc={a.halluc_rate:.1%} {'<' if v4 else '>='} 30%",
        },
    }

    print("\n" + "=" * 70)
    print("VERDICTS")
    print("=" * 70)
    n_pass = 0
    for name, v in verdicts.items():
        tag = "PASS" if v["pass"] else "FAIL"
        n_pass += int(v["pass"])
        print(f"  {tag}: {name}  ({v['detail']})")

    print(f"\n  Overall: {n_pass}/4 passed")
    if n_pass >= 3:
        print("  CONCLUSION: Embodied LM demonstrates introspective reporting.")
    else:
        print("  CONCLUSION: Introspective reporting not conclusively demonstrated.")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    def ser(o):
        if isinstance(o, dict):  return {k: ser(v) for k, v in o.items()}
        if isinstance(o, list):  return [ser(v) for v in o]
        if isinstance(o, (np.floating, np.integer)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, torch.Tensor): return o.tolist()
        if isinstance(o, bool): return o
        return o

    out = {
        "experiment": "z1707_introspective_reporting",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": str(device),
        "config": {
            "batch_size": BATCH_SIZE, "seq_len": SEQ_LEN,
            "num_epochs": NUM_EPOCHS, "lr": LR,
            "num_test_trials": NUM_TEST_TRIALS,
            "gen_tokens": GEN_TOKENS, "gen_temperature": GEN_TEMPERATURE,
        },
        "conditions": {},
        "verdicts": ser(verdicts),
        "n_passed": n_pass,
        "overall_pass": n_pass >= 3,
    }
    for c in "ABCD":
        r = results[c]
        out["conditions"][c] = {
            "name": r.name, "embodied": r.embodied, "self_desc": r.self_desc,
            "final_ppl": r.final_ppl, "ppl_history": r.ppl_history,
            "wall_s": round(r.wall_s, 1), "energy_j": round(r.energy_j, 1),
            "self_desc_acc": r.self_desc_acc, "power_acc": r.power_acc,
            "temp_acc": r.temp_acc, "speed_acc": r.speed_acc,
            "halluc_rate": r.halluc_rate, "details": ser(r.details),
        }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to: {RESULTS_PATH}")

    if not sim_act:
        try:
            actuator.restore_initial_state()
        except Exception:
            pass

    return out


if __name__ == '__main__':
    main()
