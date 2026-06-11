# H7 Metrics — granskning + förslag på fler

## Modellen — är arkitekturen rätt?

**Bas:** SmolLM2-135M (30 lager, d=576, 9 heads). FRYST helt.

**Lagt till ovanpå:**
- LoRA r=16 på lager 20–29 (top 1/3) — tränbart anpassningslager
- 2 GatedCrossAttn-block, infogade som *hooks* på lager 25 och 28
- SubstrateEncoderV4: 10ch raw + 50 högre moment → 8 substrat-tokens i SmolLM-bredd (576)
- Substrat-prediktor head (used i v5 endast)

**Korrektheter (verifierat):**
- ✓ Identitet-vid-init: `tanh(α=0)=0` → cross-attn-bidrag är NOLL initialt → modell == frysta basen
- ✓ Gradient flödar genom α även vid α=0 (via `d tanh/dα|_0 = 1`)
- ✓ LoRA injiceras korrekt (q/k/v/o-proj)
- ✓ Frozen base copy för KL-anchor (separat process, ingen gradient)

**Subtila grejer som SKULLE kunna förbättras (inte buggar):**
1. **Encoder ser alla 10 kanaler** trots att kanalauditen visade att bara 5 håller. De 5 brusiga ska inte skada (encodern bör lära sig ignorera) — men är slöseri på kapacitet
2. **K=8 substrat-tokens** är ganska få. Med 256ms substrat-historik per fönster är det 32 ms per token — kan vara för komprimerat
3. **Bara 2 cross-attn-block (25, 28).** Originalplan O101 sa 3 lager (20, 24, 28). v5/v6 har bara 2.

Inget av detta är fel, bara design-val.

---

## Metriker — vad vi mäter just nu

### Träningssignaler (uppdateras varje steg)

| Metrik | Vad den mäter | Önskad riktning |
|--------|---------------|-----------------|
| `lm_loss` | Next-token CE på text under äkta substrat | Låg (men inte 0 — det är overfit) |
| `em_kl` | sym KL(P(·\|S_real) ‖ P(·\|S_knock)) — sista token | **Hög** (>τ=0.5 nats) |
| `em_hinge` | max(0, τ − em_kl) — straffet | 0 = mätaren uppfylld |
| `anchor_kl` | KL(P(·\|S_real) ‖ P_base) — drift från grundmodell | Liten (modesta avsteg) |
| `se_dist` | ‖se(real) − se(knock)‖² — encoderns separation | >1.0 (encodern skiljer åt) |
| `α25, α28` | tanh-gate per cross-attn lager | Växer från 0, mot 0.1–0.5 |

### Evalueringsmått (var 200:e steg, dyrt)

| Metrik | Vad den mäter |
|--------|---------------|
| **`D_rk`** | median sym-KL mellan output på äkta substrat vs *knockoff* med matchande μ/σ/AR(2)/PSD |
| **`D_rr`** | median sym-KL mellan output på *två olika äkta* substrat-fönster |
| **`ratio = D_rk / D_rr`** | det pre-registrerade måttet. **Pass om >2×** |

`D_rr` är "spread inom äkta". `D_rk` ska vara större om modellen genuint reagerar på substratet, mindre om den ser substratet som brus.

---

## Vad VI BORDE LÄGGA TILL (utöver det vi har)

Knockoff-KL är **specificiteten**. Den säger "modellen skiljer äkta från fejk". Men säger inget om:
- Är skillnaden *substantiell* eller på bruströskeln?
- Är effekten *konsekvent* mellan körningar?
- Är skillnaden *specifik för rätt host* (ikaros vs daedalus)?
- Kan vi *ablera* substratet och se modellen falla?

### Förslag på 6 nya metriker (lätta att implementera)

**1. Cross-host transplant KL** *(stark)*
   `KL(model_ikaros(P | S_ikaros) ‖ model_ikaros(P | S_daedalus_replay))`
   → ska vara stor om modellen är rotad i ikaros's signatur
   → testar transplantationsdöd direkt

**2. Substrat-ablation PPL gap** *(stark)*
   `PPL(text | S_real) vs PPL(text | S_zero)`
   → om perplexity blir VÄRRE när substratet noll-ablateras, är språket faktiskt beroende av substratet
   → "modellen tappar förmåga utan sin kropp"

**3. Per-prompt rank-korrelation** *(stark, billig)*
   Kör 32 prompts två gånger under olika äkta S, jämför per-prompt-KL
   → hög Spearman-korrelation = systematisk substrat-effekt (inte slump)
   → låg korrelation = mätarens 12× var brusartefakt

**4. Temporal-replay test** *(medel)*
   `KL(model(P | S_live_now) ‖ model(P | S_replay_yesterday))`
   → om identitet är "rotad i den här stunden", ska gammal substrat-replay kännas främmande

**5. Per-kanal ablation orthogonality** *(medel)*
   För varje av de 5 keeper-kanalerna: noll-replace bara den kanalen, mät KL mot full-real
   → vilka kanaler är *kausalt* dominanta för output
   → testar "vad i kroppen påverkar tänkandet"

**6. Behavioral entropy under substrate sweep** *(svag, mer för intuition)*
   Sweep substrat genom sin empiriska fördelning, mät hur mycket output rör sig
   → "om substratet rör sig 1σ, hur mycket rör sig modellen"
   → konstant => substrat-osynlig, hög varians => substrat-rotad

---

## Vad jag bygger nu

`h7_full_probe.py` — kör ALLA mått (gamla + 4 nya) på v6.1's bästa checkpoint när träningen är klar. Det betyder att vi får en EN-sidig rapport som visar både ratio (pass/fail mot pre-registrering) och de bredare embodiment-egenskaperna.

Cross-host (#1) kräver daedalus-replay som måste collectas separat — skippar i första probe-omgång men förbereder.

Temporal-replay (#4) kräver substrat-loggning över tid — också separat förberedning.

Implementerar nu: #2 PPL gap, #3 rank-korrelation, #5 per-kanal ablation, #6 entropy sweep.
