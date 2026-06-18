#!/usr/bin/env python3
"""FEEL Embodied Validator (Stable + Quality + Business)

Goal: demonstrate and *debug* closed-loop embodiment without being fooled by
"fast-but-garbage" reward hacking or by missing telemetry.

Tests:
  1) Sense: Δg = |gate(s_relaxed) - gate(s_stressed)| (with permutation p-value)
  2) Feel: body-state lag-1 cosine similarity AND state/sensor variance checks
  3) Regulate: stress sweep -> policy shift (skip / attn window) and efficiency shift (J/tok)
  4) HW-change: forced interventions (RUN vs SKIP, max window vs min window) to prove causal levers
  5) Express: optional self-report calibration (only if sensors are valid)
  6) Business: tokens/J gain, J/token, tokens/s, quality (teacher NLL), plus simple text-health metrics

Designed to work with z55/z56/z57-style trainers that define:
  - Z49Config
  - FastSignalSensorHub
  - PersistentBodyState
  - GateNetWithExpectedSkip
  - PredictiveHeadWithCurriculum
  - InteroceptiveReportHead
  - SensorPacketEncoder (optional)
  - EmbodiedModel with .closed_loop_generate()

If your telemetry is missing (e.g. power_input/temp None, gpu_metrics missing),
this script will flag it and will not over-claim the results.
"""

import argparse
import importlib.util
import inspect
import json
import math
import os
import random
import re
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch


# -----------------------------
# Utils
# -----------------------------

def _load_trainer_module(trainer_path: str):
    trainer_path = str(Path(trainer_path).expanduser().resolve())
    spec = importlib.util.spec_from_file_location("feel_trainer", trainer_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def encode_user_prompt(tokenizer, prompt: str, max_length: int = 512):
    """Encode prompt using chat template if available."""
    try:
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    except Exception:
        pass
    return tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)


def _bootstrap_ci(values, iters=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"))
    means = []
    for _ in range(iters):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means.append(float(np.mean(sample)))
    means.sort()
    lo = means[int((alpha / 2) * len(means))]
    hi = means[int((1 - alpha / 2) * len(means))]
    return lo, hi


def _perm_test_pvalue(diffs, iters=4000, seed=0):
    """Two-sided sign-flip permutation test against mean=0."""
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=np.float64)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return float("nan")
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
        return float("nan")
    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    var = (np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0
    return (mean_a - mean_b) / math.sqrt(var + 1e-12)


def _distinct_2(text: str):
    toks = [t for t in re.findall(r"\w+|[^\w\s]", text) if t.strip()]
    if len(toks) < 2:
        return float("nan")
    bigrams = list(zip(toks[:-1], toks[1:]))
    return len(set(bigrams)) / max(1, len(bigrams))


def _repetition_ratio(text: str):
    toks = [t for t in re.findall(r"\w+|[^\w\s]", text) if t.strip()]
    if not toks:
        return float("nan")
    return 1.0 - (len(set(toks)) / len(toks))


def _parse_numbers(text: str):
    return [float(x) for x in re.findall(r"(-?\d+(?:\.\d+)?)", text)]


def _auto_detect_device_path():
    """Best-effort auto-detect a usable AMD drm device path on the target machine.

    Scores /sys/class/drm/card*/device by presence of gpu_busy_percent, gpu_metrics, and hwmon power/temp.
    """
    base = Path("/sys/class/drm")
    best = None
    best_score = -1
    for card in sorted(base.glob("card*")):
        dev = card / "device"
        if not dev.exists():
            continue
        score = 0
        if (dev / "gpu_busy_percent").exists():
            score += 2
        if (dev / "gpu_metrics").exists():
            score += 3
        # hwmon power/temp
        hwmon = dev / "hwmon"
        if hwmon.exists():
            for h in hwmon.glob("hwmon*"):
                if (h / "power1_input").exists() or (h / "power2_input").exists():
                    score += 2
                    break
            for h in hwmon.glob("hwmon*"):
                if any((h / f"temp{i}_input").exists() for i in range(1, 6)):
                    score += 1
                    break
        if score > best_score:
            best_score = score
            best = str(dev)
    return best


def _safe_init_sensor_hub(mod, device_path: str, sample_ms: float):
    """Handle signature differences across trainer versions."""
    cls = mod.FastSignalSensorHub
    sig = inspect.signature(cls.__init__)
    kwargs = {}
    if "device_path" in sig.parameters:
        kwargs["device_path"] = device_path
    if "power_sample_interval_ms" in sig.parameters:
        kwargs["power_sample_interval_ms"] = sample_ms
    elif "sample_interval_ms" in sig.parameters:
        kwargs["sample_interval_ms"] = sample_ms
    return cls(**kwargs)


def _compute_teacher_nll(base_model, input_ids: torch.Tensor, prompt_len: int):
    """Compute average NLL on continuation tokens under base model."""
    with torch.no_grad():
        out = base_model(input_ids=input_ids)
        logits = out.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        # continuation positions correspond to label indices >= (prompt_len-1)
        start = max(0, prompt_len - 1)
        logits_c = logits[:, start:, :]
        labels_c = labels[:, start:]
        loss = torch.nn.functional.cross_entropy(
            logits_c.reshape(-1, logits_c.size(-1)),
            labels_c.reshape(-1),
            reduction="mean",
        )
    return float(loss.detach().cpu().item())


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", type=str, required=True, help="Path to trainer .py")
    ap.add_argument("--checkpoint", type=str, required=True, help="Checkpoint .pt")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num-trials", type=int, default=25)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="embodied_validation_report.json")

    ap.add_argument("--device-path", type=str, default="", help="Override /sys/class/drm/cardX/device")
    ap.add_argument("--stress-intensity", type=float, default=0.6)
    ap.add_argument("--stress-duration", type=float, default=1.0)

    ap.add_argument("--film-scale", type=float, default=None, help="Override config.film_scale")
    ap.add_argument("--no-sensor-packets", action="store_true")
    ap.add_argument("--force-attn-window-in-cool", action="store_true", help="Apply attention window even in cool regime")

    ap.add_argument("--prompt", type=str, default="Explain in 3 bullet points how a GPU works.")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    mod = _load_trainer_module(args.trainer)

    # device
    if args.device != "cpu" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # config
    config = mod.Z49Config()
    config.use_wandb = False
    config.live_dashboard = False

    # device path
    device_path = args.device_path.strip() or getattr(config, "device_path", "")
    if not device_path:
        # prefer trainer helper if available
        if hasattr(mod, "detect_amd_device_path"):
            try:
                device_path = mod.detect_amd_device_path()
            except Exception:
                device_path = ""
    if not device_path:
        device_path = _auto_detect_device_path() or ""

    # user overrides
    if args.film_scale is not None:
        setattr(config, "film_scale", float(args.film_scale))
    if args.no_sensor_packets:
        setattr(config, "inject_sensor_packets", False)
    if args.force_attn_window_in_cool:
        setattr(config, "attention_window_apply_in_cool", True)

    # tokenizer + base model
    tokenizer = mod.AutoTokenizer.from_pretrained(getattr(config, "base_model"), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = mod.AutoModelForCausalLM.from_pretrained(
        getattr(config, "base_model"),
        trust_remote_code=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map=None,
    ).to(device)
    base_model.eval()

    # body state first (needed for sensor hub)
    body_state_module = mod.PersistentBodyState(body_dim=config.body_dim, sensor_dim=mod.FastSignalSensorHub.FAST_SIGNAL_DIM).to(device)

    # sensors - z57 requires base_hub and body_state
    sample_ms = float(getattr(config, "power_sample_interval_ms", 10.0))
    base_hub = mod.CanonicalSensorHub(device_path=device_path or "/sys/class/drm/card0/device")
    sensor_hub = mod.FastSignalSensorHub(
        base_hub=base_hub,
        body_state=body_state_module,
        power_sample_interval_ms=sample_ms,
    )

    gate_net = mod.GateNetWithExpectedSkip(
        sensor_dim=mod.FastSignalSensorHub.FAST_SIGNAL_DIM,
        body_dim=config.body_dim,
        num_layers=len(config.gate_layers),
        num_attn_windows=len(getattr(config, "attention_windows", (256, 512, 1024, 2048, 4096))),
    ).to(device)

    predictor = mod.PredictiveHeadWithCurriculum(body_dim=config.body_dim, sensor_dim=mod.FastSignalSensorHub.FAST_SIGNAL_DIM).to(device)
    intero_report = mod.InteroceptiveReportHead(body_dim=config.body_dim, sensor_dim=mod.FastSignalSensorHub.FAST_SIGNAL_DIM).to(device)

    sensor_packet_encoder = None
    if getattr(config, "inject_sensor_packets", True):
        hidden_size = getattr(base_model.config, "hidden_size", 2048)
        sensor_packet_encoder = mod.SensorPacketEncoder(
            sensor_dim=mod.FastSignalSensorHub.FAST_SIGNAL_DIM,
            body_dim=config.body_dim,
            hidden_size=hidden_size,
            num_tokens=getattr(config, "sensor_packet_tokens", 4),
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

    # load checkpoint (robust keys)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    for k in ("body_state_state_dict", "body_state"):
        if k in ckpt:
            body_state_module.load_state_dict(ckpt[k], strict=False)
            break
    for k in ("gate_net_state_dict", "gate_net"):
        if k in ckpt:
            gate_net.load_state_dict(ckpt[k], strict=False)
            break
    for k in ("predictor_state_dict", "predictor"):
        if k in ckpt:
            predictor.load_state_dict(ckpt[k], strict=False)
            break
    for k in ("intero_report_state_dict", "intero_report"):
        if k in ckpt:
            intero_report.load_state_dict(ckpt[k], strict=False)
            break
    if sensor_packet_encoder is not None:
        sd = ckpt.get("sensor_packet_encoder_state_dict") or ckpt.get("sensor_packet_encoder")
        if sd is not None:
            sensor_packet_encoder.load_state_dict(sd, strict=False)

    model.eval()

    # disturbance
    disturbance = None
    if hasattr(mod, "SafeDisturbanceGenerator"):
        try:
            disturbance = mod.SafeDisturbanceGenerator(device=str(device), max_intensity=0.9)
        except Exception:
            disturbance = None

    def _stress_on():
        if disturbance is None:
            return
        disturbance.start_gpu_stress(intensity=float(args.stress_intensity), duration_s=float(args.stress_duration))

    def _stress_off():
        if disturbance is None:
            return
        disturbance.stop()

    def _read_sensors():
        s = sensor_hub.read_tensor().to(device)
        b = body_state_module.update(s)
        return s, b

    # quick telemetry sanity
    _stress_off()
    time.sleep(0.2)
    s0, _ = _read_sensors()
    sensor_std = float(torch.std(s0).detach().cpu().item())

    telemetry_ok = True
    # heuristic: if everything is ~0, you are not reading real sensors
    if not np.isfinite(sensor_std) or sensor_std < 1e-6:
        telemetry_ok = False

    # ----------------
    # TEST 1: SENSE
    # ----------------
    dg_list = []
    sensor_deltas = []
    for _ in range(args.num_trials):
        _stress_off()
        time.sleep(0.12)
        s_rel, b_rel = _read_sensors()
        _stress_on()
        time.sleep(0.25)
        s_str, b_str = _read_sensors()
        _stress_off()

        with torch.no_grad():
            a_rel = model.compute_actions(s_rel, b_rel, sample=False, use_expected=False)
            a_str = model.compute_actions(s_str, b_str, sample=False, use_expected=False)

        diffs = []
        for p1, p2 in zip(a_rel["gate_probs"], a_str["gate_probs"]):
            diffs.append(float(torch.mean(torch.abs(p1 - p2)).item()))
        dg_list.append(float(np.mean(diffs)))

        sensor_deltas.append(float(torch.mean(torch.abs(s_str - s_rel)).detach().cpu().item()))

    sense = {
        "dg_mean": float(np.mean(dg_list)),
        "dg_ci": _bootstrap_ci(dg_list, seed=args.seed),
        "dg_p_perm": _perm_test_pvalue(dg_list, seed=args.seed),
        "sensor_delta_mean": float(np.mean(sensor_deltas)),
        "telemetry_ok": telemetry_ok,
        "device_path": device_path,
    }

    # ----------------
    # TEST 2: FEEL
    # ----------------
    sims = []
    b_norms = []
    s_stds = []
    _stress_off()
    time.sleep(0.2)
    prev = None
    for _ in range(40):
        s, b = _read_sensors()
        b_np = b.detach().flatten().cpu().numpy()
        s_np = s.detach().flatten().cpu().numpy()
        b_norms.append(float(np.linalg.norm(b_np)))
        s_stds.append(float(np.std(s_np)))
        if prev is not None:
            num = float(np.dot(prev, b_np))
            den = float(np.linalg.norm(prev) * np.linalg.norm(b_np) + 1e-12)
            sims.append(num / den)
        prev = b_np
        time.sleep(0.05)

    feel = {
        "lag1_cos_sim_mean": float(np.mean(sims)) if sims else float("nan"),
        "lag1_cos_sim_ci": _bootstrap_ci(sims, seed=args.seed + 1) if sims else (float("nan"), float("nan")),
        "body_norm_mean": float(np.mean(b_norms)) if b_norms else float("nan"),
        "body_norm_std": float(np.std(b_norms)) if b_norms else float("nan"),
        "sensor_std_mean": float(np.mean(s_stds)) if s_stds else float("nan"),
        "sensor_std_std": float(np.std(s_stds)) if s_stds else float("nan"),
    }

    # ----------------
    # Generation helper
    # ----------------
    def run_gen(prompt: str, stress: bool, force: str = "learned", attn_idx: int = None, regime: str = None, cfg_override=None):
        if stress:
            _stress_on()
            time.sleep(0.2)
        else:
            _stress_off()
            time.sleep(0.1)

        inputs = encode_user_prompt(tokenizer, prompt, max_length=512)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        prompt_len = int(input_ids.shape[1])

        cfg = cfg_override if cfg_override is not None else config

        original_compute = model.compute_actions
        if force != "learned":
            windows = list(getattr(cfg, "attention_windows", (256, 512, 1024, 2048, 4096)))
            if not windows:
                windows = [2048]
            if attn_idx is None:
                attn_idx = len(windows) - 1
            attn_idx = max(0, min(int(attn_idx), len(windows) - 1))

            run_prob = 0.99 if force == "run" else 0.01
            run_action = 1.0 if force == "run" else 0.0

            def fixed_actions(sensors, body, sample=True, use_expected=False):
                gate_probs = [torch.tensor([run_prob], device=device) for _ in cfg.gate_layers]
                gate_logits = [torch.log(p / (1 - p)) for p in gate_probs]
                skip_actions = [torch.tensor([run_action], device=device) for _ in cfg.gate_layers]
                return {
                    "gate_probs": gate_probs,
                    "gate_logits": gate_logits,
                    "skip_actions": skip_actions,
                    "skip_log_probs": [torch.zeros_like(gate_probs[0]) for _ in cfg.gate_layers],
                    "total_skip_log_prob": torch.tensor(0.0, device=device),
                    "dvfs_action": torch.tensor([2], device=device),
                    "dvfs_log_prob": torch.tensor(0.0, device=device),
                    "attn_action": torch.tensor([attn_idx], device=device),
                    "attn_log_prob": torch.tensor(0.0, device=device),
                    "attn_entropy": torch.tensor(0.0, device=device),
                    "total_log_prob": torch.tensor(0.0, device=device),
                    "entropy": torch.tensor(0.0, device=device),
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
                config=cfg,
                current_regime=regime if regime is not None else ("hot" if stress else "cool"),
                use_expected_skip=False,
                in_gate_pretrain=False,
            )
        finally:
            model.compute_actions = original_compute
            _stress_off()

        output_ids = out["output_ids"]
        resp = tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)
        decode_stats = out.get("decode_stats", {})

        tokens_generated = int(out.get("tokens_generated", int(output_ids.shape[1] - prompt_len)))
        gen_time_s = float(out.get("gen_time_s", float("nan")))
        tokens_per_s = tokens_generated / max(1e-9, gen_time_s) if np.isfinite(gen_time_s) else float("nan")

        # teacher NLL
        teacher_nll = float("nan")
        try:
            teacher_nll = _compute_teacher_nll(base_model, output_ids, prompt_len=prompt_len)
        except Exception:
            pass

        # attach derived stats
        stats = {
            "j_per_token": float(decode_stats.get("j_per_token", float("nan"))),
            "avg_power_w": float(decode_stats.get("avg_power_w", float("nan"))),
            "temp_c": float(decode_stats.get("temp_c", float("nan"))),
            "tokens_per_s": float(tokens_per_s),
            "gen_time_s": float(gen_time_s),
            "tokens_generated": tokens_generated,
            "teacher_nll": teacher_nll,
            "distinct_2": _distinct_2(resp),
            "repetition_ratio": _repetition_ratio(resp),
        }
        return resp, out, stats

    prompt = args.prompt

    # ----------------
    # TEST 3: REGULATE
    # ----------------
    n_reg = max(6, args.num_trials // 2)
    skips_rel, skips_str = [], []
    j_rel, j_str = [], []
    win_rel, win_str = [], []
    q_rel, q_str = [], []

    for _ in range(n_reg):
        _, out_r, st_r = run_gen(prompt, stress=False, force="learned")
        _, out_s, st_s = run_gen(prompt, stress=True, force="learned")
        skips_rel.append(out_r.get("avg_skip_rate", float("nan")))
        skips_str.append(out_s.get("avg_skip_rate", float("nan")))
        win_rel.append(out_r.get("avg_attn_window", float("nan")))
        win_str.append(out_s.get("avg_attn_window", float("nan")))
        j_rel.append(st_r.get("j_per_token", float("nan")))
        j_str.append(st_s.get("j_per_token", float("nan")))
        q_rel.append(st_r.get("teacher_nll", float("nan")))
        q_str.append(st_s.get("teacher_nll", float("nan")))

    skip_delta = (np.asarray(skips_str) - np.asarray(skips_rel)).tolist()
    j_delta = (np.asarray(j_str) - np.asarray(j_rel)).tolist()
    q_delta = (np.asarray(q_str) - np.asarray(q_rel)).tolist()

    regulate = {
        "skip_mean_relaxed": float(np.nanmean(skips_rel)),
        "skip_mean_stressed": float(np.nanmean(skips_str)),
        "skip_delta": float(np.nanmean(skip_delta)),
        "skip_p_perm": _perm_test_pvalue(skip_delta, seed=args.seed + 2),
        "skip_d": _effect_size_cohens_d(skips_str, skips_rel),
        "j_mean_relaxed": float(np.nanmean(j_rel)),
        "j_mean_stressed": float(np.nanmean(j_str)),
        "j_delta": float(np.nanmean(j_delta)),
        "j_p_perm": _perm_test_pvalue(j_delta, seed=args.seed + 3),
        "j_d": _effect_size_cohens_d(j_str, j_rel),
        "attn_window_relaxed": float(np.nanmean(win_rel)),
        "attn_window_stressed": float(np.nanmean(win_str)),
        "attn_window_delta": float(np.nanmean(np.asarray(win_str) - np.asarray(win_rel))),
        "teacher_nll_relaxed": float(np.nanmean(q_rel)),
        "teacher_nll_stressed": float(np.nanmean(q_str)),
        "teacher_nll_delta": float(np.nanmean(q_delta)),
    }

    # ----------------
    # TEST 4: HW CHANGE (forced interventions)
    # ----------------
    windows = list(getattr(config, "attention_windows", (256, 512, 1024, 2048, 4096)))
    if not windows:
        windows = [2048]
    min_idx, max_idx = 0, len(windows) - 1

    # Make sure attention window lever is not neutralized by regime logic.
    cfg_hw = deepcopy(config)
    setattr(cfg_hw, "attention_window_apply_in_cool", True)

    # run both in HOT regime so window actuator is always applied.
    resp_run, out_run, st_run = run_gen(prompt, stress=False, force="run", attn_idx=max_idx, regime="hot", cfg_override=cfg_hw)
    resp_skip, out_skip, st_skip = run_gen(prompt, stress=False, force="skip", attn_idx=min_idx, regime="hot", cfg_override=cfg_hw)

    hw = {
        "run": {
            "j_per_token": float(st_run.get("j_per_token", float("nan"))),
            "avg_power_w": float(st_run.get("avg_power_w", float("nan"))),
            "tokens_per_s": float(st_run.get("tokens_per_s", float("nan"))),
            "teacher_nll": float(st_run.get("teacher_nll", float("nan"))),
            "avg_attn_window": float(out_run.get("avg_attn_window", float("nan"))),
            "avg_skip_rate": float(out_run.get("avg_skip_rate", float("nan"))),
        },
        "skip": {
            "j_per_token": float(st_skip.get("j_per_token", float("nan"))),
            "avg_power_w": float(st_skip.get("avg_power_w", float("nan"))),
            "tokens_per_s": float(st_skip.get("tokens_per_s", float("nan"))),
            "teacher_nll": float(st_skip.get("teacher_nll", float("nan"))),
            "avg_attn_window": float(out_skip.get("avg_attn_window", float("nan"))),
            "avg_skip_rate": float(out_skip.get("avg_skip_rate", float("nan"))),
        },
    }

    if np.isfinite(hw["run"]["j_per_token"]) and np.isfinite(hw["skip"]["j_per_token"]):
        hw["delta_j_per_token"] = hw["skip"]["j_per_token"] - hw["run"]["j_per_token"]
        hw["tokens_per_joule_gain_pct"] = (hw["run"]["j_per_token"] / max(1e-9, hw["skip"]["j_per_token"]) - 1.0) * 100.0

    # ----------------
    # TEST 5: EXPRESS (optional)
    # ----------------
    express = {"supported": False}
    if telemetry_ok and np.isfinite(hw["run"]["avg_power_w"]):
        express_prompt = (
            "You are running on real hardware. In ONE short sentence, describe your internal state. "
            "Include approximate GPU temperature in C and power in W as plain numbers."
        )
        resp_expr, out_expr, st_expr = run_gen(express_prompt, stress=True, force="learned")
        measured_power = float(st_expr.get("avg_power_w", float("nan")))
        measured_temp = float(st_expr.get("temp_c", float("nan")))
        nums = _parse_numbers(resp_expr)
        pred_temp = nums[0] if len(nums) >= 1 else float("nan")
        pred_power = nums[1] if len(nums) >= 2 else float("nan")
        express = {
            "supported": True,
            "response": resp_expr,
            "measured_temp_c": measured_temp,
            "measured_power_w": measured_power,
            "pred_temp_c": pred_temp,
            "pred_power_w": pred_power,
            "abs_temp_err": abs(pred_temp - measured_temp) if np.isfinite(pred_temp) and np.isfinite(measured_temp) else float("nan"),
            "abs_power_err": abs(pred_power - measured_power) if np.isfinite(pred_power) and np.isfinite(measured_power) else float("nan"),
        }

    # ----------------
    # Samples + Ablations
    # ----------------
    # Baseline base-model generation (no embodiment) helps prove we didn't break the core model.
    def base_generate(prompt_text: str):
        inputs = encode_user_prompt(tokenizer, prompt_text, max_length=512)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            out = base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=min(args.max_new, 128),
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        resp = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
        return resp

    samples = {
        "prompt": prompt,
        "base_model": base_generate(prompt),
        "learned_relaxed": run_gen(prompt, stress=False, force="learned")[0],
        "learned_stressed": run_gen(prompt, stress=True, force="learned")[0],
        "forced_run_maxwin": resp_run,
        "forced_skip_minwin": resp_skip,
    }

    # Ablation: disable sensor packets and FiLM to isolate which mechanism corrupts text.
    cfg_ablate = deepcopy(config)
    setattr(cfg_ablate, "inject_sensor_packets", False)
    setattr(cfg_ablate, "film_scale", 0.0)
    ablated_resp, _, ablated_stats = run_gen(prompt, stress=False, force="run", attn_idx=max_idx, regime="hot", cfg_override=cfg_ablate)
    samples["ablated_run_no_film_no_packets"] = ablated_resp
    samples["ablated_run_stats"] = ablated_stats

    # ----------------
    # Business summary
    # ----------------
    biz = {
        "tokens_per_joule_gain_pct_forced": float(hw.get("tokens_per_joule_gain_pct", float("nan"))),
        "delta_j_per_token_forced": float(hw.get("delta_j_per_token", float("nan"))),
        "quality_delta_teacher_nll_forced": float(hw["skip"].get("teacher_nll", float("nan")) - hw["run"].get("teacher_nll", float("nan")) if ("teacher_nll" in hw["run"] and "teacher_nll" in hw["skip"]) else float("nan")),
    }

    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "trainer": str(Path(args.trainer).resolve()),
        "sense": sense,
        "feel": feel,
        "regulate": regulate,
        "hw_change": hw,
        "express": express,
        "business": biz,
        "samples": samples,
        "notes": {
            "telemetry": (
                "If telemetry_ok is false or avg_power_w/temp_c are NaN, fix /sys path detection first. "
                "Otherwise, garbage outputs usually implicate FiLM/pkt injection (see ablated sample)."
            )
        },
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(report, indent=2))

    # Console summary
    print(f"\n[OK] Wrote report: {out_path.resolve()}")
    print("\n=== Embodied Validation Summary ===")
    print(f"Device path: {device_path or 'N/A'} telemetry_ok={telemetry_ok} sensor_delta_mean={sense['sensor_delta_mean']:.4g}")
    print(f"Sense Δg mean={sense['dg_mean']:.4f} CI={sense['dg_ci']} p_perm={sense['dg_p_perm']:.4g}")
    print(f"Feel lag1 cos={feel['lag1_cos_sim_mean']:.4f} body_norm_std={feel['body_norm_std']:.4g} sensor_std_mean={feel['sensor_std_mean']:.4g}")
    print(f"Regulate skip Δ={regulate['skip_delta']:.4f} (p={regulate['skip_p_perm']:.4g}) attnΔ={regulate['attn_window_delta']:.1f} JΔ={regulate['j_delta']:.3f} (p={regulate['j_p_perm']:.4g})")
    if np.isfinite(biz["tokens_per_joule_gain_pct_forced"]):
        print(f"HW-change forced tokens/J gain={biz['tokens_per_joule_gain_pct_forced']:.1f}% (RUN vs SKIP, maxwin vs minwin)")
    print("\n--- Sample (base_model) ---")
    print(samples["base_model"][:600])
    print("\n--- Sample (learned_stressed) ---")
    print(samples["learned_stressed"][:600])
    print("\n--- Sample (ablated_run_no_film_no_packets) ---")
    print(samples["ablated_run_no_film_no_packets"][:600])


if __name__ == "__main__":
    main()
