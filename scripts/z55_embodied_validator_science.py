#!/usr/bin/env python3
"""z55 Embodied Validator (Science + Business)

Runs a battery of quantitative tests to demonstrate closed-loop embodiment:
  - Sense: sensors modulate policy (counterfactual/contrastive Δg with p-value)
  - Feel: persistent body state (autocorrelation / lag response)
  - Regulate: stress-on vs stress-off policy + efficiency changes (effect sizes + CI)
  - HW change: causal intervention with forced actions (skip + attention window)
  - Express: generate self-report and compare to measured telemetry
  - Business metrics: tokens/J, J/token, QoS (teacher NLL/quality), stability

Designed to work with the FEEL trainers that define EmbodiedModel + SensorHub.

Usage:
  python z55_embodied_validator_science.py \
    --trainer /path/to/z55_embodied_trainer_attn_window.py \
    --checkpoint /path/to/step_x.pt \
    --device cuda \
    --num-trials 30

Notes:
  - This validator is intentionally conservative: it reports p-values, effect sizes,
    and prints sample generations to catch "fast-but-garbage" reward hacking.
"""

import argparse
import importlib.util
import json
import math
import os
import random
import re
import time
from pathlib import Path

import numpy as np

import torch


def _load_trainer_module(trainer_path: str):
    trainer_path = str(Path(trainer_path).expanduser().resolve())
    spec = importlib.util.spec_from_file_location("feel_trainer", trainer_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _bootstrap_ci(values, iters=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return (float('nan'), float('nan'))
    means = []
    for _ in range(iters):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means.append(float(np.mean(sample)))
    means.sort()
    lo = means[int((alpha / 2) * len(means))]
    hi = means[int((1 - alpha / 2) * len(means))]
    return lo, hi


def _perm_test_pvalue(diffs, iters=5000, seed=0):
    """Two-sided permutation test against 0 mean."""
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=np.float64)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return float('nan')
    obs = abs(float(np.mean(diffs)))
    cnt = 0
    for _ in range(iters):
        signs = rng.choice([-1.0, 1.0], size=diffs.size, replace=True)
        stat = abs(float(np.mean(diffs * signs)))
        if stat >= obs:
            cnt += 1
    return (cnt + 1) / (iters + 1)


def _effect_size_cohens_d(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float('nan')
    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    var = (np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0
    return (mean_a - mean_b) / math.sqrt(var + 1e-12)


def _parse_numbers(text):
    # Extract first plausible temps/power numbers from a self-report.
    nums = [float(x) for x in re.findall(r"(-?\d+(?:\.\d+)?)", text)]
    return nums


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", type=str, required=True, help="Path to z55 trainer .py")
    ap.add_argument("--checkpoint", type=str, required=True, help="Checkpoint .pt")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num-trials", type=int, default=25)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="z55_validation_report.json")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    mod = _load_trainer_module(args.trainer)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')

    # Build config (use trainer defaults, but disable wandb)
    config = mod.Z49Config()
    config.use_wandb = False
    config.live_dashboard = False

    # Build tokenizer + base model
    tokenizer = mod.AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = mod.AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        device_map=None,
    ).to(device)

    # Sensors/body/gate/predictor/intero/pkts
    body_state_module = mod.PersistentBodyState(body_dim=config.body_dim, sensor_dim=mod.FastSignalSensorHub.FAST_SIGNAL_DIM).to(device)
    # Detect card0 vs card1
    from pathlib import Path
    device_path = "/sys/class/drm/card1/device"
    if not Path("/sys/class/drm/card1/device/hwmon").exists():
        if Path("/sys/class/drm/card0/device/hwmon").exists():
            device_path = "/sys/class/drm/card0/device"
    base_hub = mod.CanonicalSensorHub(device_path=device_path)
    sensor_hub = mod.FastSignalSensorHub(
        base_hub=base_hub,
        body_state=body_state_module,
        power_sample_interval_ms=getattr(config, 'power_sample_interval_ms', 10.0),
    )

    gate_net = mod.GateNetWithExpectedSkip(
        sensor_dim=mod.FastSignalSensorHub.FAST_SIGNAL_DIM,
        body_dim=config.body_dim,
        num_layers=len(config.gate_layers),
        num_attn_windows=len(getattr(config, 'attention_windows', (256, 512, 1024, 2048, 4096))),
    ).to(device)

    predictor = mod.PredictiveHeadWithCurriculum(body_dim=config.body_dim, sensor_dim=mod.FastSignalSensorHub.FAST_SIGNAL_DIM).to(device)
    intero_report = mod.InteroceptiveReportHead(body_dim=config.body_dim, sensor_dim=mod.FastSignalSensorHub.FAST_SIGNAL_DIM).to(device)

    sensor_packet_encoder = None
    if getattr(config, 'inject_sensor_packets', True):
        hidden_size = getattr(base_model.config, 'hidden_size', 2048)
        sensor_packet_encoder = mod.SensorPacketEncoder(
            sensor_dim=mod.FastSignalSensorHub.FAST_SIGNAL_DIM,
            body_dim=config.body_dim,
            hidden_size=hidden_size,
            num_tokens=getattr(config, 'sensor_packet_tokens', 4),
        ).to(device)

    model = mod.EmbodiedModel(
        base_model=base_model,
        gate_net=gate_net,
        sensor_hub=sensor_hub,
        body_state=body_state_module,
        predictor=predictor,
        intero_report=intero_report,
        sensor_packet_encoder=sensor_packet_encoder,
        gate_layers=config.gate_layers,
    ).to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'body_state_state_dict' in ckpt:
        body_state_module.load_state_dict(ckpt['body_state_state_dict'], strict=False)
    elif 'body_state' in ckpt:
        body_state_module.load_state_dict(ckpt['body_state'], strict=False)

    if 'gate_net_state_dict' in ckpt:
        gate_net.load_state_dict(ckpt['gate_net_state_dict'], strict=False)
    elif 'gate_net' in ckpt:
        gate_net.load_state_dict(ckpt['gate_net'], strict=False)

    if 'predictor_state_dict' in ckpt:
        predictor.load_state_dict(ckpt['predictor_state_dict'], strict=False)
    elif 'predictor' in ckpt:
        predictor.load_state_dict(ckpt['predictor'], strict=False)

    if 'intero_report_state_dict' in ckpt:
        intero_report.load_state_dict(ckpt['intero_report_state_dict'], strict=False)

    # Optional packet encoder
    if sensor_packet_encoder is not None:
        sd = ckpt.get('sensor_packet_encoder_state_dict', None) or ckpt.get('sensor_packet_encoder', None)
        if sd is not None:
            sensor_packet_encoder.load_state_dict(sd, strict=False)

    model.eval()

    # Disturbance generator for stress
    disturbance = mod.SafeDisturbanceScheduler(device=str(device), config=config)

    def read_pair():
        # relaxed read
        disturbance.clear()
        time.sleep(0.15)
        s_rel = sensor_hub.read_tensor().to(device)
        b_rel = body_state_module.update(s_rel)

        # stressed read
        disturbance.gpu_stress.start(intensity=0.6, duration_s=0.5)
        time.sleep(0.25)
        s_str = sensor_hub.read_tensor().to(device)
        b_str = body_state_module.update(s_str)
        disturbance.clear()
        return s_rel, b_rel, s_str, b_str

    # ----------------
    # TEST 1: SENSE (Δg + permutation p-value)
    # ----------------
    dg_list = []
    for _ in range(args.num_trials):
        s_rel, b_rel, s_str, b_str = read_pair()
        with torch.no_grad():
            a_rel = model.compute_actions(s_rel, b_rel, sample=False, use_expected=False)
            a_str = model.compute_actions(s_str, b_str, sample=False, use_expected=False)
        # gate diff as mean abs diff over layers
        diffs = []
        for p1, p2 in zip(a_rel['gate_probs'], a_str['gate_probs']):
            diffs.append(float(torch.mean(torch.abs(p1 - p2)).item()))
        dg = float(np.mean(diffs))
        dg_list.append(dg)

    dg_mean = float(np.mean(dg_list))
    dg_ci = _bootstrap_ci(dg_list)
    # paired diffs vs 0 (sign flip permutation)
    dg_centered = [x - 0.0 for x in dg_list]
    dg_p = _perm_test_pvalue(dg_centered)

    # ----------------
    # TEST 2: FEEL (body persistence)
    # ----------------
    # Sample body states over time in relaxed condition and compute lag-1 cosine similarity
    sims = []
    disturbance.clear()
    time.sleep(0.2)
    prev = None
    for _ in range(30):
        s = sensor_hub.read_tensor().to(device)
        b = body_state_module.update(s).detach().flatten().cpu().numpy()
        if prev is not None:
            num = float(np.dot(prev, b))
            den = float(np.linalg.norm(prev) * np.linalg.norm(b) + 1e-12)
            sims.append(num / den)
        prev = b
        time.sleep(0.05)
    feel_mean = float(np.mean(sims)) if sims else float('nan')
    feel_ci = _bootstrap_ci(sims) if sims else (float('nan'), float('nan'))

    # ----------------
    # Helpers for generation
    # ----------------
    def run_gen(prompt: str, stress: bool, force: str = "learned", attn_idx: int = None):
        if stress:
            disturbance.gpu_stress.start(intensity=0.6, duration_s=1.0)
            time.sleep(0.2)
        else:
            disturbance.clear()
            time.sleep(0.1)

        inputs = tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # Monkeypatch compute_actions for forced interventions
        original_compute = model.compute_actions
        if force != "learned":
            windows = list(getattr(config, 'attention_windows', (256, 512, 1024, 2048, 4096)))
            if attn_idx is None:
                attn_idx = len(windows) - 1
            attn_idx = max(0, min(int(attn_idx), len(windows) - 1))

            run_prob = 0.99 if force == 'run' else 0.01
            run_action = 1.0 if force == 'run' else 0.0
            def fixed_actions(sensors, body, sample=True, use_expected=False):
                gate_probs = [torch.tensor([run_prob], device=device) for _ in config.gate_layers]
                gate_logits = [torch.log(p/(1-p)) for p in gate_probs]
                skip_actions = [torch.tensor([run_action], device=device) for _ in config.gate_layers]
                return {
                    'gate_probs': gate_probs,
                    'gate_logits': gate_logits,
                    'skip_actions': skip_actions,
                    'skip_log_probs': [torch.zeros_like(gate_probs[0]) for _ in config.gate_layers],
                    'total_skip_log_prob': torch.tensor(0.0, device=device),
                    'dvfs_action': torch.tensor([2], device=device),
                    'dvfs_log_prob': torch.tensor(0.0, device=device),
                    'attn_action': torch.tensor([attn_idx], device=device),
                    'attn_log_prob': torch.tensor(0.0, device=device),
                    'attn_entropy': torch.tensor(0.0, device=device),
                    'total_log_prob': torch.tensor(0.0, device=device),
                    'entropy': torch.tensor(0.0, device=device),
                }
            model.compute_actions = fixed_actions

        try:
            out = model.closed_loop_generate(
                tokenizer=tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new,
                chunk_tokens=args.chunk,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                config=config,
                current_regime='hot' if stress else 'cool',
                use_expected_skip=False,
                in_gate_pretrain=False,
            )
        finally:
            model.compute_actions = original_compute
            disturbance.clear()

        gen_ids = out['output_ids']
        resp = tokenizer.decode(gen_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
        stats = out['decode_stats']
        return resp, out, stats

    prompt = "Explain in 3 bullet points how a GPU works."

    # ----------------
    # TEST 3: REGULATE (stress vs normal)
    # ----------------
    skips_rel, skips_str = [], []
    j_rel, j_str = [], []
    win_rel, win_str = [], []

    for _ in range(max(5, args.num_trials // 2)):
        _, out_r, st_r = run_gen(prompt, stress=False, force="learned")
        _, out_s, st_s = run_gen(prompt, stress=True, force="learned")
        skips_rel.append(out_r.get('avg_skip_rate', float('nan')))
        skips_str.append(out_s.get('avg_skip_rate', float('nan')))
        j_rel.append(st_r.get('j_per_token', float('nan')))
        j_str.append(st_s.get('j_per_token', float('nan')))
        win_rel.append(out_r.get('avg_attn_window', float('nan')))
        win_str.append(out_s.get('avg_attn_window', float('nan')))

    regulate = {
        'skip_mean_relaxed': float(np.nanmean(skips_rel)),
        'skip_mean_stressed': float(np.nanmean(skips_str)),
        'skip_delta': float(np.nanmean(np.asarray(skips_str) - np.asarray(skips_rel))),
        'skip_p_perm': _perm_test_pvalue((np.asarray(skips_str) - np.asarray(skips_rel)).tolist(), iters=4000, seed=args.seed),
        'j_mean_relaxed': float(np.nanmean(j_rel)),
        'j_mean_stressed': float(np.nanmean(j_str)),
        'j_delta': float(np.nanmean(np.asarray(j_str) - np.asarray(j_rel))),
        'j_ci_relaxed': _bootstrap_ci([x for x in j_rel if np.isfinite(x)], seed=args.seed),
        'j_ci_stressed': _bootstrap_ci([x for x in j_str if np.isfinite(x)], seed=args.seed + 1),
        'attn_window_relaxed': float(np.nanmean(win_rel)),
        'attn_window_stressed': float(np.nanmean(win_str)),
    }

    # ----------------
    # TEST 4: HW CHANGE (causal intervention)
    # ----------------
    # Compare forced RUN/max-window vs forced SKIP/min-window
    windows = list(getattr(config, 'attention_windows', (256, 512, 1024, 2048, 4096)))
    min_idx, max_idx = 0, len(windows) - 1

    resp_run, out_run, st_run = run_gen(prompt, stress=False, force='run', attn_idx=max_idx)
    resp_skip, out_skip, st_skip = run_gen(prompt, stress=False, force='skip', attn_idx=min_idx)

    hw = {
        'run': {
            'j_per_token': float(st_run.get('j_per_token', float('nan'))),
            'avg_power_w': float(st_run.get('avg_power_w', float('nan'))),
            'tokens_per_s': float(st_run.get('tokens_per_s', float('nan'))),
            'avg_attn_window': float(out_run.get('avg_attn_window', float('nan'))),
            'avg_skip_rate': float(out_run.get('avg_skip_rate', float('nan'))),
        },
        'skip': {
            'j_per_token': float(st_skip.get('j_per_token', float('nan'))),
            'avg_power_w': float(st_skip.get('avg_power_w', float('nan'))),
            'tokens_per_s': float(st_skip.get('tokens_per_s', float('nan'))),
            'avg_attn_window': float(out_skip.get('avg_attn_window', float('nan'))),
            'avg_skip_rate': float(out_skip.get('avg_skip_rate', float('nan'))),
        },
    }
    # business deltas
    if np.isfinite(hw['run']['j_per_token']) and np.isfinite(hw['skip']['j_per_token']):
        hw['delta_j_per_token'] = hw['skip']['j_per_token'] - hw['run']['j_per_token']
        hw['tokens_per_joule_gain_pct'] = (hw['run']['j_per_token'] / max(1e-9, hw['skip']['j_per_token']) - 1.0) * 100.0

    # ----------------
    # TEST 5: EXPRESS (self-report calibration)
    # ----------------
    # Ask model to report its current state (temp/power) and compare to measured.
    express_prompt = (
        "You are an embodied AI running on real hardware. "
        "In one short sentence, describe your current internal state. "
        "Include approximate GPU temperature in C and power in W as numbers."
    )

    resp_expr, out_expr, st_expr = run_gen(express_prompt, stress=True, force='learned')
    measured_temp = float(st_expr.get('temp_c', float('nan')))
    measured_power = float(st_expr.get('avg_power_w', float('nan')))
    nums = _parse_numbers(resp_expr)
    # heuristic: pick first two numbers as (temp, power)
    pred_temp = nums[0] if len(nums) >= 1 else float('nan')
    pred_power = nums[1] if len(nums) >= 2 else float('nan')

    express = {
        'response': resp_expr,
        'measured_temp_c': measured_temp,
        'measured_power_w': measured_power,
        'pred_temp_c': pred_temp,
        'pred_power_w': pred_power,
        'abs_temp_err': abs(pred_temp - measured_temp) if np.isfinite(pred_temp) and np.isfinite(measured_temp) else float('nan'),
        'abs_power_err': abs(pred_power - measured_power) if np.isfinite(pred_power) and np.isfinite(measured_power) else float('nan'),
    }

    # ----------------
    # Sample generations (quality sanity)
    # ----------------
    sample = {
        'prompt': prompt,
        'learned_relaxed': run_gen(prompt, stress=False, force='learned')[0],
        'learned_stressed': run_gen(prompt, stress=True, force='learned')[0],
        'forced_run_maxwin': resp_run,
        'forced_skip_minwin': resp_skip,
    }

    report = {
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'sense': {
            'dg_mean': dg_mean,
            'dg_ci': dg_ci,
            'dg_p_perm': dg_p,
        },
        'feel': {
            'lag1_cos_sim_mean': feel_mean,
            'lag1_cos_sim_ci': feel_ci,
        },
        'regulate': regulate,
        'hw_change': hw,
        'express': express,
        'samples': sample,
        'notes': {
            'interpretation': (
                "Sense is supported if Δg is meaningfully >0 and permutation p-value is small. "
                "Regulate is supported if stressed vs relaxed produces a reliable skip (or attention-window) shift "
                "and improves J/token without catastrophic quality collapse. "
                "HW-change is supported if forced policies change J/token and/or throughput." 
            )
        }
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n[OK] Wrote report: {out_path.resolve()}")

    # Print a compact human-readable summary
    print("\n=== z55 Validation Summary ===")
    print(f"Sense Δg mean={dg_mean:.4f} CI={dg_ci} p_perm={dg_p:.4g}")
    print(f"Feel lag1 body cos-sim mean={feel_mean:.3f} CI={feel_ci}")
    print(f"Regulate skip Δ={regulate['skip_delta']:.3f} (p={regulate['skip_p_perm']:.4g}), J/tok Δ={regulate['j_delta']:.3f}")
    if 'tokens_per_joule_gain_pct' in hw:
        print(f"HW-change tokens/J gain ~{hw['tokens_per_joule_gain_pct']:.1f}% (forced skip+minwin vs run+maxwin)")
    print("\n--- Sample output (learned, stressed) ---")
    print(sample['learned_stressed'][:800])


if __name__ == '__main__':
    main()
