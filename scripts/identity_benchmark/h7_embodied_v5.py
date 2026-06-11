"""H7 EMBODIED v5 — online-plastic LLM that drifts with its substrate.

This is NOT a conditional-LM. It is a living process.

Architecture:
  - SmolLM2-135M base. Bottom 20 layers FROZEN ("brainstem" — language syntax).
  - Top 10 layers: LoRA(r=16) on q_proj + v_proj — ONLINE PLASTIC.
  - Cross-attention with substrate tokens at layers 25, 28 — also plastic.
  - SubstrateEncoderV4 (Perceiver resampler → K=8 substrate tokens per window).

Online learning loop (every 32 tokens):
  - Read fresh substrate window.
  - LM hidden states predict next substrate frame.
  - local_loss = MSE(predicted_next_substrate, actual_next_substrate)
  - Backprop ONLY into LoRA + xattn + SE. Bounded grad-norm cap.
  - One optimizer step. Online learning rate η_online adapted by homeostatic critic.

4 homeostasis mechanisms:
  1. Per-update grad-norm cap (synaptic plasticity ceiling).
  2. Homeostatic critic on substrate-prediction-accuracy rolling mean
     (NOT PPL — that's RLHF bias. Embodiment means knowing your body.)
  3. Substrate baseline drift: 10-min rolling mean is "home". Drift away → costly.
     Acute shock (>3σ outside) → plasticity ratchets up (acclimatization).
  4. Sleep: every 1000 tokens, 30-step self-distillation on its own generations.

No anti-spoof margins. No KL anchor to base. No "reset" button.
The model is its current state. Two runs diverge after hours.

Reproducibility (classical sense) is DEAD. That's the price of embodiment.
"""
from __future__ import annotations
import argparse, json, math, os, socket, sys, time
from pathlib import Path
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from substrate_realtime_v3 import SubstrateStateV3, higher_moments
from h7_rooted_lm_v4a import (
    SubstrateEncoderV4, GatedCrossAttn, GlobalNorm,
    BASE_MODEL, K_TOKENS, N_CHANNELS, WIN_LEN, STATS,
)

HOST = socket.gethostname()
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results/IDENTITY_EMBODIED_2026-06-10"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT = OUT_DIR / f"embodied_v5_{HOST}.pt"
LOG_JSON = OUT_DIR / f"embodied_v5_{HOST}.jsonl"

# Plasticity bounds — synaptic-plasticity-cap analog
ETA_ONLINE_INIT  = 2e-4
ETA_ONLINE_MIN   = 1e-5
ETA_ONLINE_MAX   = 1e-3
GRAD_NORM_CAP    = 0.01     # bounded per update
UPDATE_EVERY     = 32       # tokens between online updates
SLEEP_EVERY      = 1000     # tokens between sleep cycles
SLEEP_STEPS      = 30       # self-distillation steps per sleep
BASELINE_LEN     = 1200     # ~20 min @ 1 update/sec from 32-token throughput
ACUTE_SIGMA      = 3.0      # outside this → acclimatization

INSERT_LAYERS = [25, 28]    # cross-attention layers (top 1/3 of 30)
LORA_LAYERS = list(range(20, 30))  # online-plastic layers
LORA_RANK = 16


# ---------------------------------------------------------------------------
# Minimal LoRA — applied to q_proj and v_proj of LLaMA-style layers
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with LoRA delta = α/r · B(A(x))."""
    def __init__(self, base: nn.Linear, r=LORA_RANK, alpha=32.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters(): p.requires_grad = False
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))   # zero-init → identity at start
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)


def inject_lora(model, layers, rank=LORA_RANK):
    """Replace q_proj and v_proj in named layers with LoRALinear."""
    lora_modules = []
    for layer_idx in layers:
        L = model.model.layers[layer_idx]
        attn = L.self_attn
        for name in ["q_proj", "v_proj"]:
            base = getattr(attn, name)
            wrapped = LoRALinear(base, r=rank)
            setattr(attn, name, wrapped)
            lora_modules.append(wrapped)
    return lora_modules


# ---------------------------------------------------------------------------
# Embodied wrapper — LoRA + xattn + substrate-prediction head
# ---------------------------------------------------------------------------
class EmbodiedSmolLM(nn.Module):
    def __init__(self, base_name=BASE_MODEL,
                 insert_layers=INSERT_LAYERS, lora_layers=LORA_LAYERS):
        super().__init__()
        self.base = AutoModelForCausalLM.from_pretrained(base_name)
        for p in self.base.parameters(): p.requires_grad = False
        self.d = self.base.config.hidden_size
        self.heads = self.base.config.num_attention_heads
        self.lora_mods = inject_lora(self.base, lora_layers)
        self.insert_layers = set(insert_layers)
        self.xattn = nn.ModuleDict({
            str(i): GatedCrossAttn(self.d, self.heads) for i in insert_layers
        })
        # Substrate prediction head: last hidden → next substrate frame (C channels)
        self.sub_pred = nn.Sequential(
            nn.Linear(self.d, 64), nn.GELU(), nn.Linear(64, N_CHANNELS)
        )
        self._S = None
        for i in insert_layers:
            self.base.model.layers[i].register_forward_hook(self._make_hook(i))

    def _make_hook(self, layer_idx):
        xattn = self.xattn[str(layer_idx)]
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            if self._S is not None:
                h = xattn(h, self._S)
            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h
        return hook

    def trainable_params(self):
        params = []
        for m in self.lora_mods:
            params += [m.A, m.B]
        params += list(self.xattn.parameters())
        params += list(self.sub_pred.parameters())
        return params

    def gate_alphas(self):
        return [self.xattn[str(i)].alpha.detach().item() for i in self.insert_layers]

    def forward(self, input_ids, substrate_tokens=None, output_hidden=False):
        self._S = substrate_tokens
        out = self.base(input_ids=input_ids, output_hidden_states=output_hidden)
        self._S = None
        return out


# ---------------------------------------------------------------------------
# Online stream — runs forever. Reads substrate at 500Hz, processes text in
# a never-ending loop, updates LoRA every UPDATE_EVERY tokens.
# ---------------------------------------------------------------------------
def online_loop(steps=10000, ctx=128, log_every=5, load_ckpt=None, tag=""):
    """steps is the number of online update events (each = 32 tokens consumed)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[v5 online] host={HOST} device={device} update_every={UPDATE_EVERY} steps={steps}")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    model = EmbodiedSmolLM().to(device)
    se = SubstrateEncoderV4(d_emb=model.d, K=K_TOKENS).to(device)

    # Optional: load from prior host's embodied checkpoint (transplant test)
    eta = ETA_ONLINE_INIT
    loaded_baseline = None
    loaded_substrate_baseline = None
    if load_ckpt and Path(load_ckpt).exists():
        ck = torch.load(load_ckpt, map_location=device, weights_only=False)
        print(f"[v5 online] LOADING transplant ckpt from {load_ckpt}")
        print(f"  trained on host={ck.get('host')}, step={ck.get('step')}, version={ck.get('version')}")
        # Load xattn (gate values from prior embodiment)
        model.xattn.load_state_dict(ck["xattn"])
        # Load substrate encoder
        se.load_state_dict(ck["se"])
        # Load substrate predictor
        model.sub_pred.load_state_dict(ck["sub_pred"])
        # Inherit baseline accuracy (so homeostat doesn't restart cold)
        loaded_baseline = ck.get("baseline_acc")
        eta = ck.get("eta", ETA_ONLINE_INIT)
        # Keep substrate_baseline so we can detect host-shift
        sb = ck.get("substrate_baseline")
        if sb is not None: loaded_substrate_baseline = np.array(sb, dtype=np.float32)
        print(f"  baseline_acc inherited: {loaded_baseline}")
        print(f"  starting η inherited:    {eta:.1e}")

    opt = torch.optim.AdamW(
        model.trainable_params() + list(se.parameters()),
        lr=eta, betas=(0.9, 0.999), weight_decay=0.01
    )

    norm = GlobalNorm(STATS)
    state = SubstrateStateV3(hz_target=500); state.start()

    # Deterministic but non-trivial text stream as input — same trick as v4a
    rng_text = np.random.default_rng(7 + ord(HOST[0]))  # host-specific seed
    chars = "the quick brown fox jumps over lazy dog she said hello world to me i think therefore i am the universe is made of stories not atoms a body is mind and mind is body all is one. "
    big_text = "".join(rng_text.choice(list(chars), size=400_000))
    ids = tok(big_text, return_tensors="pt").input_ids[0]
    N = ids.shape[0] - ctx - UPDATE_EVERY - 1
    cursor = 0

    rng = np.random.default_rng(123)
    pred_acc_buf = deque(maxlen=BASELINE_LEN)
    baseline_acc = loaded_baseline
    substrate_baseline = loaded_substrate_baseline
    substrate_var = (np.ones(N_CHANNELS) * 1.0) if loaded_substrate_baseline is not None else None
    log_path = LOG_JSON.parent / f"embodied_v5_{HOST}{'_' + tag if tag else ''}.jsonl"
    ckpt_path = CKPT.parent / f"embodied_v5_{HOST}{'_' + tag if tag else ''}.pt"
    log_f = open(log_path, "a")
    print(f"[v5 online] log → {log_path}")
    print(f"[v5 online] ckpt → {ckpt_path}")
    t0 = time.time()
    sleep_counter = 0

    for step in range(steps):
        # 1. Take next text chunk (ctx tokens of context + UPDATE_EVERY new tokens)
        if cursor + ctx + UPDATE_EVERY + 1 >= len(ids):
            cursor = 0
        ctx_ids = ids[cursor : cursor + ctx].unsqueeze(0).to(device)
        next_ids = ids[cursor + ctx : cursor + ctx + UPDATE_EVERY + 1].to(device)
        cursor += UPDATE_EVERY

        # 2. Read substrate windows: current (for SE input) and next (for prediction target)
        w_now = state.latest_window(length=WIN_LEN)
        time.sleep(0.05)   # let new substrate samples accumulate
        w_next = state.latest_window(length=WIN_LEN)
        target_next_frame = norm(w_next)[-1, :]   # last frame of next window

        # 3. Update substrate baseline (10-min EMA + std)
        w_now_n = norm(w_now)
        cur_frame = w_now_n[-1, :]
        if substrate_baseline is None:
            substrate_baseline = cur_frame.copy()
            substrate_var = np.ones_like(cur_frame) * 1.0
        else:
            ema_a = 0.01
            substrate_baseline = (1-ema_a) * substrate_baseline + ema_a * cur_frame
            substrate_var = (1-ema_a) * substrate_var + ema_a * (cur_frame - substrate_baseline)**2
        substrate_sd = np.sqrt(substrate_var + 1e-8)
        # Acute distance: max channel z-score
        acute = float(np.max(np.abs(cur_frame - substrate_baseline) / substrate_sd))

        # 4. Encode substrate → K tokens
        mom = higher_moments(w_now_n).astype(np.float32)
        sub_t = torch.from_numpy(w_now_n[None]).to(device)
        mom_t = torch.from_numpy(mom[None]).to(device)
        S = se(sub_t, mom_t)

        # 5. Forward LM with hidden states
        out = model(ctx_ids, substrate_tokens=S, output_hidden=True)
        last_hidden = out.hidden_states[-1][:, -1, :]   # (B=1, d)

        # 6. Substrate prediction loss (PRIMARY embodiment signal)
        pred = model.sub_pred(last_hidden)              # (1, C)
        target = torch.from_numpy(target_next_frame[None]).to(device).float()
        local_loss = F.mse_loss(pred, target)

        # 7. Diagnostic: LM PPL on next_ids (NOT used for training, just logged)
        with torch.no_grad():
            ce = F.cross_entropy(
                out.logits[0, -UPDATE_EVERY:, :], next_ids[1:UPDATE_EVERY+1]
            )
            lm_ppl = math.exp(min(ce.item(), 50.0))

        # 8. Embodiment accuracy metric: how close is pred to target in σ units?
        sd_t = torch.from_numpy(substrate_sd).to(device).float()
        err_in_sd = (pred[0] - target[0]).abs() / (sd_t + 1e-6)
        pred_acc = float((err_in_sd < 1.0).float().mean().item())   # fraction within 1σ
        pred_acc_buf.append(pred_acc)

        # 9. Update baseline accuracy (for homeostatic critic)
        if baseline_acc is None and len(pred_acc_buf) >= 50:
            baseline_acc = float(np.mean(pred_acc_buf))
        elif baseline_acc is not None:
            baseline_acc = 0.99 * baseline_acc + 0.01 * pred_acc

        # 10. Backprop ONLY through LoRA + xattn + SE + sub_pred (frozen base = no grads)
        opt.zero_grad()
        local_loss.backward()
        # Bounded grad-norm
        params = model.trainable_params() + list(se.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(params, GRAD_NORM_CAP)

        # 11. Homeostatic critic: adjust η based on whether pred_acc beats baseline
        if baseline_acc is not None:
            if pred_acc < baseline_acc - 0.05:
                eta *= 0.5      # struggling → lower plasticity (don't drift more)
            elif pred_acc > baseline_acc + 0.02:
                eta *= 1.05     # in groove → slight consolidation
            # Acute shock → ratchet up plasticity (acclimatization)
            if acute > ACUTE_SIGMA:
                eta *= 1.5
            eta = float(np.clip(eta, ETA_ONLINE_MIN, ETA_ONLINE_MAX))
            for g in opt.param_groups:
                g["lr"] = eta

        opt.step()

        # 12. Sleep cycle — every SLEEP_EVERY tokens, do self-distillation
        # (regenerate from own predictions, keep brain consolidated)
        sleep_counter += UPDATE_EVERY
        slept = False
        if sleep_counter >= SLEEP_EVERY:
            slept = True
            sleep_counter = 0
            with torch.no_grad():
                S_sleep = se(sub_t, mom_t).detach()
                cur = ctx_ids.clone()
                for _ in range(8):
                    o = model(cur[:, -ctx:], substrate_tokens=S_sleep)
                    next_tok = o.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                    cur = torch.cat([cur, next_tok], dim=-1)
                self_gen_x = cur[:, -ctx-1:-1]
                self_gen_y = cur[:, -ctx:]
            for _ in range(SLEEP_STEPS):
                S_step = se(sub_t, mom_t).detach()
                o2 = model(self_gen_x, substrate_tokens=S_step)
                ce2 = F.cross_entropy(
                    o2.logits.reshape(-1, o2.logits.shape[-1]),
                    self_gen_y.reshape(-1)
                )
                opt.zero_grad(); (ce2 * 0.1).backward()
                torch.nn.utils.clip_grad_norm_(params, GRAD_NORM_CAP)
                opt.step()

        # 13. Log
        if (step + 1) % log_every == 0:
            entry = {
                "step": step+1, "t": time.time()-t0,
                "local_loss": float(local_loss.item()),
                "pred_acc": pred_acc,
                "baseline_acc": baseline_acc if baseline_acc else 0.0,
                "lm_ppl": lm_ppl,
                "eta": eta,
                "grad_norm": float(grad_norm.item()),
                "acute_sigma": acute,
                "alphas": model.gate_alphas(),
                "slept": slept,
            }
            log_f.write(json.dumps(entry) + "\n"); log_f.flush()
            a = float(np.abs(model.gate_alphas()).mean())
            print(f"  s{step+1:6d}  ll={local_loss.item():.4f}  "
                  f"acc={pred_acc:.3f}/{baseline_acc or 0:.3f}  "
                  f"ppl={lm_ppl:8.1f}  η={eta:.1e}  "
                  f"|α|={a:.3f}  acute={acute:.1f}σ"
                  f"{'  SLEPT' if slept else ''}  t={time.time()-t0:.0f}s")

        # Save checkpoint periodically
        if (step + 1) % 500 == 0:
            torch.save({
                "lora": {f"layer{i}_q_A": model.lora_mods[2*j].A for j,i in enumerate(LORA_LAYERS)},
                "xattn": model.xattn.state_dict(),
                "se": se.state_dict(),
                "sub_pred": model.sub_pred.state_dict(),
                "eta": eta,
                "baseline_acc": baseline_acc,
                "substrate_baseline": substrate_baseline.tolist() if substrate_baseline is not None else None,
                "host": HOST, "step": step+1,
                "version": "5_embodied",
            }, ckpt_path)

    state.stop(); log_f.close()
    print(f"[v5 online] done. ckpt={ckpt_path} log={log_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--load-ckpt", type=str, default=None,
                    help="path to prior-host embodied ckpt for transplant test")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()
    if args.cmd == "run":
        online_loop(steps=args.steps, ctx=args.ctx,
                    load_ckpt=args.load_ckpt, tag=args.tag)


if __name__ == "__main__":
    main()
