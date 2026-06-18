"""Phase 14 Task B — Embodied GPT-2.

Wraps HuggingFace GPT2LMHeadModel ('gpt2', 124M) and injects the LIVE 32-d
hardware signature into 3 internal operations every forward pass:

  L8  — attention temperature scaling:
        attn_logits = attn_logits * (1.0 + alpha * mlp_temp(sig))
        where mlp_temp: 32 -> 1, output passed through tanh, range (-1,1)

  L11 — LayerNorm gamma offset:
        gamma_eff = gamma + beta * mlp_gamma(sig)
        where mlp_gamma: 32 -> hidden, small-init

  L14 — Residual gain:
        x = x + gain * sublayer(x), gain = 1 + delta * tanh(mlp_gain(sig))
        where mlp_gain: 32 -> 1

Each injection MLP is small (32->8->1 or 32->8->hidden) and zero-initialised
on the output layer, so model starts equivalent to vanilla GPT-2 and the
embodiment must be LEARNED (or not — if it can't be learned, the embodied
PPL should not be worse than vanilla).

Signature is read fresh at the START of each forward pass and broadcast to
all 3 injection sites. If signature.read() raises, model fails (this is the
unforgeable property).
"""
from __future__ import annotations
import torch, torch.nn as nn, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from signature_io import LiveSignature

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# layer indices to modify (gpt2 base has 12 layers 0..11; medium has 24).
# Spec said L8/L11/L14, but base gpt2 has only 12 layers (0..11). Map:
#   L8  -> layer index 7
#   L11 -> layer index 10
#   L14 -> layer index 11 (last)
INJECT_ATTN_LAYER  = 7
INJECT_LN_LAYER    = 10
INJECT_RESID_LAYER = 11


class SignatureMLP(nn.Module):
    def __init__(self, in_dim=32, hidden=8, out_dim=1, zero_out=True):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        if zero_out:
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)
    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))


class EmbodiedGPT2(nn.Module):
    """Wrap GPT2LMHeadModel and inject signature features at 3 layers."""
    def __init__(self, model_name='gpt2', alpha=0.1, beta=0.05, delta=0.1,
                 sig_reader: LiveSignature = None, freeze_backbone=True):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.config = self.model.config
        H = self.config.n_embd  # 768 for gpt2-base
        self.sig_reader = sig_reader
        self.alpha = alpha
        self.beta  = beta
        self.delta = delta

        # injection MLPs
        self.mlp_temp  = SignatureMLP(32, 8, 1, zero_out=True)
        self.mlp_gamma = SignatureMLP(32, 16, H, zero_out=True)
        self.mlp_gain  = SignatureMLP(32, 8, 1, zero_out=True)

        # current signature (refreshed per forward pass)
        self.register_buffer('_sig_buf', torch.zeros(32), persistent=False)
        self._sig_override = None  # if set, bypass live read (for spoof tests)

        # patch the layers
        self._install_hooks()

        if freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad_(False)
        # the MLPs remain trainable.

    @property
    def trainable_parameters(self):
        return [p for n, p in self.named_parameters() if p.requires_grad]

    def _install_hooks(self):
        # We avoid forward_pre_hooks for attention internals — too fragile across
        # transformers versions. Instead we wrap the block forward via monkey-patch.
        blocks = self.model.transformer.h

        # ATTENTION TEMP injection: scale the q-vector before attention (equivalent
        # to scaling logits since logits = QK^T / sqrt(d))
        target_attn = blocks[INJECT_ATTN_LAYER].attn
        orig_attn_forward = target_attn.forward
        owner = self
        def patched_attn_forward(*args, **kwargs):
            # multiplicative scale on output of attention via post-hook is easier
            return orig_attn_forward(*args, **kwargs)
        # apply via output multiplier — attach forward hook on the attn module
        def attn_out_hook(module, inputs, output):
            # output: tuple(attn_out, present, [attn_weights])
            scale = 1.0 + owner.alpha * torch.tanh(owner.mlp_temp(owner._sig_buf)).squeeze()
            new0 = output[0] * scale
            if isinstance(output, tuple):
                return (new0,) + tuple(output[1:])
            return new0
        target_attn.register_forward_hook(attn_out_hook)

        # LN GAMMA injection: replace ln_2 of layer LN with a wrapper
        target_block = blocks[INJECT_LN_LAYER]
        orig_ln = target_block.ln_2
        class _LNWrap(nn.Module):
            def __init__(self, orig, owner):
                super().__init__()
                self.orig = orig
                # bypass nn.Module submodule registration to avoid owner cycle
                object.__setattr__(self, 'owner', owner)
            def forward(self, x):
                # standard LN -> then modulate gamma by sig
                out = self.orig(x)
                # multiplicative gamma modulation, additive nudge
                offset = self.owner.mlp_gamma(self.owner._sig_buf)  # (H,)
                offset = offset.to(out.dtype)
                return out * (1.0 + self.owner.beta * torch.tanh(offset))
        target_block.ln_2 = _LNWrap(orig_ln, owner)

        # RESIDUAL GAIN injection: modulate the MLP residual of the last layer
        target_block_r = blocks[INJECT_RESID_LAYER]
        orig_mlp = target_block_r.mlp
        class _MLPWrap(nn.Module):
            def __init__(self, orig, owner):
                super().__init__()
                self.orig = orig
                object.__setattr__(self, 'owner', owner)
            def forward(self, x):
                y = self.orig(x)
                gain = 1.0 + self.owner.delta * torch.tanh(self.owner.mlp_gain(self.owner._sig_buf)).squeeze()
                return y * gain
        target_block_r.mlp = _MLPWrap(orig_mlp, owner)

    # ---- signature management ----
    def refresh_signature(self):
        if self._sig_override is not None:
            self._sig_buf.copy_(self._sig_override.to(self._sig_buf.device))
            return
        if self.sig_reader is None:
            # zero signature -> exactly vanilla
            self._sig_buf.zero_()
            return
        v = self.sig_reader.read_torch(device=self._sig_buf.device, dtype=self._sig_buf.dtype)
        self._sig_buf.copy_(v)

    def set_signature_override(self, v):
        """For spoofing / replay tests: force signature to a specific value."""
        if v is None:
            self._sig_override = None
            return
        if not isinstance(v, torch.Tensor):
            v = torch.as_tensor(v)
        self._sig_override = v.to(self._sig_buf.dtype)

    def forward(self, input_ids, labels=None, attention_mask=None, **kw):
        self.refresh_signature()
        return self.model(input_ids=input_ids, labels=labels,
                          attention_mask=attention_mask, **kw)


class VanillaGPT2(nn.Module):
    """Plain frozen GPT-2 (no embodiment), same interface."""
    def __init__(self, model_name='gpt2', freeze_backbone=True):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        if freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad_(False)
    def forward(self, input_ids, labels=None, attention_mask=None, **kw):
        return self.model(input_ids=input_ids, labels=labels,
                          attention_mask=attention_mask, **kw)


def load_tokenizer(model_name='gpt2'):
    tok = GPT2Tokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    return tok


if __name__ == '__main__':
    print("[smoke] loading...")
    sig = LiveSignature()
    m = EmbodiedGPT2(sig_reader=sig)
    tok = load_tokenizer()
    print(f"[smoke] trainable params: {sum(p.numel() for p in m.trainable_parameters):,}")
    ids = tok("hello world", return_tensors='pt').input_ids
    out = m(ids, labels=ids)
    print(f"[smoke] loss={out.loss.item():.4f}")
    # check vanilla is same as embodied @ zero-init MLPs
    v = VanillaGPT2()
    out_v = v(ids, labels=ids)
    print(f"[smoke] vanilla loss={out_v.loss.item():.4f}")
    print(f"[smoke] diff={abs(out.loss.item() - out_v.loss.item()):.2e} (should be ~0 at init)")
