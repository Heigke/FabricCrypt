# O101 synthesis — Conditioning a frozen LLM on hardware telemetry without language collapse

Date: 2026-06-10. 4 oracles answered (gpt-5 203s, gemini-2.5-pro 76s, grok-4 21s, deepseek-reasoner 72s).

## Convergence (4/4 agreement on the structural fix)

### What kills the v1/v2/v3 approach
**All four oracles confirm:** per-block multiplicative FiLM over the whole transformer stack is the wrong architecture. v1 ignored substrate; v2 blew up γ; v2.2 anti-rooted; v3 collapsed at eval. This is a structural failure mode, not a tuning failure.

| Oracle | Top architecture | Reason it solves v2.2's anti-rooting |
|---|---|---|
| gpt-5 | Flamingo gated x-attn into TOP 1/3 layers + Perceiver resampler → K=8-12 tokens | `tanh(α=0)` init = exact base behavior at start. KL anchor to base on zero-path. |
| gemini | Hypernet → LoRA (r=8-16) in mid-1/3 layers (deepseek backup) | Zero-input → zero ΔW by design. Mid layers preserve early-layer language. |
| grok | Flamingo gated x-attn into TOP 4-6 layers | Same as gpt-5, slightly more parsimonious. |
| deepseek | Hypernet → conditional LoRA in mid 1/3 (8-16 of 24) | Same as gemini. Functional separation: LM keeps language, hypernet provides condition-dependent perturbation. |

→ **Two camps with the same diagnosis**: 2/4 say gated-x-attn TOP layers, 2/4 say hypernet→LoRA MID layers. Both use:
1. **Identity-at-init**: zero-input must produce zero perturbation (tanh(α=0) gate, OR ΔW=hypernet(0)=0).
2. **Layer restriction**: NOT all-layer modulation. Top 1/3 (xattn) or middle 1/3 (LoRA) only. Early layers stay pristine for language.
3. **KL anchor to base** on the zero-substrate path. This is the missing piece in v2.2. We never had it.

### What anchors zero-substrate to base behavior (4/4 consensus)
**Explicit KL-divergence loss** between `P_θ(x | telem=0)` and `P_base(x)`. Weight 0.5-1.0. This is the load-bearing fix that we missed.

```python
# What v2.2/v3 missed:
L_zero_KL = KL(P_θ(x | telem=0) || P_base(x))   # weight 1.0
```

Without this term, the model is free to drift away from base behavior wholesale — which is exactly what we saw (zero PPL < native PPL = anti-rooted; v3 PPL=5e21 at eval = drift over the edge).

### What forces substrate to actually matter (4/4 consensus)
**Pairwise margin losses** on the SAME text sample with 4 conditions:
- native vs zero: native must beat zero by ≥0.5 nats (we want substrate to help, modestly)
- native vs spoof: native must beat matched-spectrum-spoof by ≥3 nats (substrate must be SPECIFIC to this die)
- native vs wrong-host: native must beat daedalus-telemetry by ≥3 nats (cross-host transplant collapse)

```python
L = L_native_CE
  + λ_zero_KL * KL(P_θ(·|t=0) || P_base(·))         # anchor — was MISSING
  + λ_m0    * relu(m0 - (CE_zero  - CE_native))     # native helps over null
  + λ_mw    * relu(mw - (CE_wrong - CE_native))     # wrong-host collapses
  + λ_ms    * relu(mw - (CE_spoof - CE_native))     # spoof collapses
  + λ_gate  * ||α||²                                # keep gates small unless needed
  + λ_c     * InfoNCE                               # optional contrastive on hidden
```

### What normalization to use (gpt-5 + grok agree, gemini disagrees)
- **gpt-5 + grok**: per-window z-score is FINE (current) — preserves cross-host signal AND removes absolute scale per window
- **gpt-5 explicit warning** (which is also our fear): "Do not per-window standardize at train/infer time if you want absolute level shifts preserved. Use frozen GLOBAL median/MAD."
- **gemini**: explicitly says use **fixed global** stats from training set, not per-window — for the same reason
- **deepseek**: doesn't push back

→ **Decision: switch to global median/MAD** (computed once on combined ikaros+daedalus replay buffer), not per-window. Our C07 cross-host gap d=66.7 is lost with per-window z-score. That's likely why v3 failed — it standardized away the strongest signal.

### Batch composition (4/4 close to our 1/4 split)
- gpt-5: 40% native / 20% zero / 20% wrong-host / 20% spoof
- gemini: 40% / 20% / 20% / 20% (cross-host + matched-spectrum-spoof split)
- grok: 40% / 25% / 20% / 15% (phase-shift)
- deepseek: similar

→ Use 40/20/20/20. Currently we use 25/25/25/25 (in v3). Increasing native fraction reduces variance on the primary objective.

### Falsification suite (all 4 propose ~5 tests, converge on these)

| # | Test | Pass | Catches failure |
|---|---|---|---|
| 1 | PPL on neutral text with telem=0 vs base | ratio ≤ 1.10 | (a) "LM destroyed" |
| 2 | PPL(native) vs PPL(zero) on native data | ΔNLL ≥ 0.3-0.5 nats | (b) "conditioning ignored" |
| 3 | KL(P_θ(·|t=0) ‖ P_base) | ≤ 0.05 nats/token | drift from base |
| 4 | PPL(spoof) vs PPL(native) | ratio ≥ 3× | (c) "not substrate-rooted" |
| 5 | PPL(wrong-host telem) vs PPL(native) | ratio ≥ 3× | (c) "not substrate-rooted" |
| 6 | Linear-probe AUC on hidden states (native vs wrong) | ≥ 0.9 | diagnostic |

**Strong rooting = pass all 6.** All four oracles agree. We can pre-register this.

### Closed-loop microkernel (gpt-5 + grok agree, gemini neutral, deepseek didn't address)

It's **additive**, not architecturally different. Add:
- **action head** on the same hidden state → discrete microkernel opcode + small continuous parameters
- **event embeddings** for post-action telemetry windows (separate positional code for "event time")
- Train action head with REINFORCE on intrinsic reward `r = log p_θ(y | post-action telem) - log p_θ(y | zero)`

→ Can ride directly on gated-x-attn architecture. Reserved for step 5 of H7_SCALE_PLAN.

## Convergence — failure-mode predictions

**Cheapest 1-day kill criterion (gpt-5 + grok agree on form):**
On SmolLM2-135M with the recommended architecture, train 3-8k steps, then:
- KILL if `PPL_zero(GL) > 1.25× PPL_base` (LM degraded by null conditioning → architecture broken)
- KILL if `ΔNLL_zero-native < 0.2 nats` after 3k steps and not trending up (conditioning is ignored → encoder underpowered or K too small)
- KILL if gates remain `|α| < 0.05` across all top layers (model couldn't find substrate utility — needs deeper encoder)

We can pre-register this on SmolLM2-135M as a 1-day go/no-go for the entire architecture family.

## Divergence — the key open choice

| | gpt-5 + grok | gemini + deepseek |
|---|---|---|
| Architecture | **Gated x-attn in TOP layers** | **Hypernet→LoRA in MID layers** |
| Reasoning | Substrate = "thought stream" at high level | Substrate = "weight reconfiguration" at middle |
| Risk | Substrate may be too easy to ignore (just close gates) | Hypernet → LoRA can blow up if not zero-anchored explicitly |
| Resolution | Both have identity-at-init guarantees. Both should be tried. |

→ **Decision**: start with **gpt-5's gated x-attn** because it has a stronger theoretical guarantee (gate truly zero at init), is well-attested in Flamingo lineage, and is simpler to implement. **Keep gemini/deepseek's hypernet→LoRA as Plan B** if x-attn fails the 1-day kill criterion.

## What we are doing this week (committed)

### v4 architecture (today + tomorrow)
- Base: **SmolLM2-135M** frozen (download to ikaros + daedalus)
- **Telemetry encoder**: 3× 1D-depthwise-sep-conv + 2-layer transformer + Perceiver-resampler → K=8 substrate tokens per 256-sample window
- **Insertion**: gated cross-attention into layers 9-12 (top 1/3 of 12-layer SmolLM2) ONLY
- **Gates**: `y = h + tanh(α) · CrossAttn(h, S)` with α=0 init per layer
- **Normalization**: **global median/MAD** (computed once on combined ikaros+daedalus replay), NOT per-window
- **Loss**: L_native + 1.0·L_zero_KL + 1.0·L_margin_zero(m=0.5) + 1.0·L_margin_wrong(m=3) + 1.0·L_margin_spoof(m=3) + 1e-3·L_gate + 0.05·L_InfoNCE
- **Batch**: 40% native / 20% zero / 20% wrong-host / 20% spoof, in groups-of-4 so the same text sample is evaluated in all 4 conditions
- **Schedule**: 2k phase-A (no margins, only L_native + L_zero_KL + L_gate) → 6k phase-B (full loss)

### Pre-register the falsification suite

Pass criteria (all 6 must hold for "rooted" claim):
1. PPL_zero(GL) / PPL_base(GL) ≤ 1.10
2. KL_zero ≤ 0.05 nats/token  
3. ΔNLL_zero-native(ND) ≥ 0.5 nats/token, 95%-CI lower ≥ 0.3
4. PPL_spoof / PPL_native ≥ 3.0
5. PPL_wrong-host / PPL_native ≥ 3.0
6. Linear probe AUC(native vs wrong) ≥ 0.9 on top layer hidden

### Kill criteria (cheap)
After 3k steps on SmolLM2-135M:
- PPL_zero(GL) > 1.25× PPL_base → KILL (architecture wrong)
- ΔNLL_zero-native < 0.2 nats and flat → KILL (conditioning ignored)
- All gates `|α| < 0.05` → KILL (encoder underpowered)

If all three pass kill, run full 8k → evaluate falsification suite.

## Decisions for Eric

1. **Architecture**: gpt-5's gated x-attn in top 1/3 layers (recommendation) vs gemini's hypernet→LoRA in mid 1/3 (backup) — **proceed with gpt-5's pick**?
2. **Base model**: SmolLM2-135M (faster, cheaper iteration) vs Qwen3-0.6B (better LM quality, we have z2103/z2107 precedent) — **proceed with SmolLM2-135M for v4 toy**, then port to Qwen3-0.6B for v5?
3. **Normalization fix**: switch from per-window z-score to global median/MAD (we lose absolute-level information at per-window today, which kills the strongest signal C07 d=66.7). **Adopt**?
4. **Drop v3 substrate state**: SubstrateStateV3's 10 channels are still good. Keep, but compute global stats first.

If yes to 1-4 I build v4 now.
