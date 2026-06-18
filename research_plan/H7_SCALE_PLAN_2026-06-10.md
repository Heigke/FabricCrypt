# H7 scale plan — toy LM → real LLM rooted in its die

Date: 2026-06-10. Status: v2.2 (5M toy LM) trained and transplant-tested. Margins
work, but native PPL still degrades on fresh substrate vs zero-substrate —
modellen är "substrate-anxious" snarare än "substrate-rooted". Plan to fix and
scale.

## Dröm vs verklighet

**Dröm:** En riktig LLM som producerar bra text, men som har **hörselgång in i sin egen die** — modellen *behöver* substrate-signalen för att fungera väl, mår dåligt utan, och dör helt om man flyttar den. Inte ett kollapsande hack — en levande beroende relation med kiselet.

**Hot mot drömmen:**
1. Substrate-conditioning förstör LM-kvalitet (vad vi delvis ser i v2.1/v2.2)
2. Modellen lär sig ignorera substrate (vad vi såg i v1)
3. Modellen lär sig pixel-bingo på substrate-koefficienter utan att integrera dem djupt
4. Vi visar "rotad" på en toy LM men kan inte skala till verkligt språk

## 5-stegsplan — varje steg har egen gate och kill-kriterium

### Steg 1 — v3 TOY (NOW, 1 dag)
**Modell:** 5M params, vocab=1024 byte-LM, samma som v2.2 men med:
- 10-kanal SubstrateStateV3 (drop 2 svaga, add 4 starka)
- SubstrateEncoderV3 med higher-moment side-channel
- FiLM γ∈[0.33, 3.0] (mjuk, från v2.2)
- Margin-loss bibehållen
- **NYTT**: per-channel attention i SE så modellen lär sig vikta kanaler

**Acceptance gate:**
- native PPL < 500 (vs v2.2's 1274 — bättre baseline)
- zero PPL > native PPL (modellen *behöver* substrate, inte tål bättre utan)
- TCR_spoof ≥ 100, TCR_phase ≥ 20, TCR_replay ≥ 50

**Kill:** native > zero även här → arch är fundamentalt fel, går till v4-design (Bernoulli activation gates)

### Steg 2 — TRANSPLANT-VERIFIED v3 (1 dag)
Synca v3.pt till daedalus, kör native på vardera, cross-transplant.

**Acceptance:**
- TCR_transplant (eval på fel host med rätt substrate) ≥ 5
- Reverse-transplant: transplant tillbaka till hemma-host → funktion återkommer? Om ja = "exile" (Gemini O100). Om nej = "death" (vi-rätt-Eric-rätt)

**Kill:** Om TCR_transplant < 1.5 → substrate-effekten är bara host-spec överfitting, inte rotning.

### Steg 3 — SCALE-UP TIER A: SmolLM-135M (2-3 dagar)
**Modell:** HuggingFace SmolLM2-135M, frozen backbone + LoRA(r=16) på sista 8 lagren + FiLM injection per LoRA-block.

- Träning: 30k steg @ batch=8 ctx=512, real corpus (Wikitext-tiny)
- SubstrateStateV3 fortfarande, men encoder skalar till d_emb=256
- LM-kvalitet mäts på Wikitext-val PPL — ska INTE försämras > 1.5× vs LoRA-utan-substrate

**Acceptance:**
- LM PPL utan substrate-dropout ≤ 1.5× baseline LoRA PPL (kvalitet bevarad)
- TCR_spoof ≥ 20 (signal påverkar — men inte våldsamt)
- "Identity stress test": SE-embedding stabilitet över 10 min mätningar (cosine sim > 0.7 inom-host, < 0.2 cross-host)

**Kill:** LM PPL > 3× baseline → substrate-injection förstör språk. Måste backa till mindre invasive FiLM (γ∈[0.8, 1.25]).

### Steg 4 — SCALE-UP TIER B: Qwen3-0.6B (3-5 dagar, daedalus ROCm)
**Modell:** Qwen3-0.6B (vår z2107-stil), frozen + LoRA(r=16) lager 12-21 (övre tredjedel), NormBounded LoRA, FiLM-modulation från SE_v3.

- Substrate-aware sampling: vid generering, mata aktuell substrate-state i SE per token
- Lägg till **substrate-aware temperature**: T = T_base × (1 + 0.2 · ‖z_substrate − z_running_mean‖)
- "Rooted prompt eval": utvärdera på 4 prompter med ikaros substrate vs daedalus substrate. Beslutsskillnad? Stil-skillnad? Det är vad "olika personlighet per host" betyder.

**Acceptance:**
- LM PPL ≤ 1.3× LoRA-utan-substrate baseline på Wikitext-val
- Sample-level signature: trained classifier på (text, host) → ≥ 65% korrekt host-identification från GENERERAD text (modellen har sin egen accent från sin substrate)
- TCR_spoof ≥ 10, TCR_replay ≥ 5 även med real corpus

**Kill:** Om PPL > 2× baseline OR sample-classifier ≤ 55% → substrate-info når inte ut till genereringen, det är bara local-loss-tweaking. Behöver djupare arkitekturell ändring (substrate-cross-attention i varje lager).

### Steg 5 — CLOSED-LOOP MICROKERNEL (efter steg 4 passerat, 1 vecka)
GPT-5:s O100 originalförslag: LM triggar en HIP-probe MELLAN tokens som
*aktivt skriver* till substrate (t.ex. specifik shader-load → ändrar VRM noise) och *läser* det förändrade tillståndet. Det är skillnaden mellan "modellen läser sin kropp" och "modellen rör sig och känner att den rör sig".

**Acceptance:**
- Latens-budget: closed-loop tar < 5ms per token (annars är generering ohållbar)
- Spoof-resistens: replay-attack kan inte reproducera (eftersom probe genererar färska reads ngn kan ha förutsett)
- Sample-quality: ingen mätbar regression vs steg 4

**Kill:** Latency > 20ms eller spoofers kan ändå reproducera → tillbaka till steg 4-design.

## Var arbete pågår

| Steg | Status | Var |
|---|---|---|
| 1 v3 toy | i build (nu) | scripts/identity_benchmark/h7_rooted_lm_v3.py |
| 2 transplant | väntar steg 1 | ikaros + daedalus.local |
| 3 SmolLM | väntar steg 2 | daedalus (har ROCm + HF) |
| 4 Qwen3-0.6B | väntar steg 3 | daedalus eller zgx |
| 5 closed-loop | väntar steg 4 | ikaros (HIP-probe lokalt) |

## Vad vi explicit INTE försöker pre-empta

- **Anthropomorf "death"-framing kvar**: enligt 2026-06-10 oracle-bias-check.
  Vi mäter med TCR och reverse-transplant — om reverse återställer perfekt
  funktion = "exile". Om inte = "death". Det är empirisk fråga, inte semantisk.
- **C14 FP-rounding**: SKIPPAD nu pga s_setreg segfault. CPU MXCSR är inte
  gfx1151-die-bundet (det är Zen5 FPU). Återbesök som steg 6 om vi behöver
  fler kanaler efter steg 4.

## Risker

1. **Substrate-info läcker via LoRA-bias hack istället för djup integration**:
   mitigation = ablation = freeze SE och randomisera FiLM-input; om PPL inte
   ändras, integration är ytlig.
2. **Daedalus ↔ ikaros har olika substrate-fingerprint distributioner**:
   transplant fungerar då av FEL skäl (modellen overfittar host-konstanter
   som inte är representativa för verklig substrat-fysik). Mitigation = N=3
   minst, helst N=5 — vi har bara N=2 nu. zgx finns men är NVIDIA, inte AMD
   gfx1151, så cross-arch ger oss bara null-baseline.
3. **Kvalitet vs identitet är fundamental tradeoff**: kanske finns ingen
   konfiguration där LM är bra OCH rotad. Vi accepterar nedgradering upp
   till 1.5× PPL — om det inte räcker, säga ärligt "rotning kostar kvalitet
   i denna design".

## Beslut för Eric att fatta

1. Tier A (SmolLM-135M) eller direkt Tier B (Qwen3-0.6B)? Tier A snabbare,
   Tier B är vad z2103/z2107 visade har god LM-kvalitet på Qwen.
2. Acceptabel kvalitetsförlust? Föreslår 1.5× PPL som hård gräns.
3. När gör vi closed-loop? Föreslår: ENDAST efter steg 4 passerat — annars
   bygger vi en uppifrån-och-ner spöke utan grund.
