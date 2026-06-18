"""Real iterating oracle debate.

Each speaker is a real LLM API call. Every speaker sees the full transcript
so far, plus a persona, plus the current topic prompt. Moderator (Claude)
opens each topic, picks who responds, pushes back when needed.

Saves transcript progressively to TRANSCRIPT_PATH so you can monitor + bail.

Usage:
    python3 oracle_debate_flow.py [ROUNDS_PER_TOPIC]
"""
from __future__ import annotations
import os, sys, time, json, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
ENV = ROOT / ".env"
for line in ENV.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line: continue
    k, v = line.split("=", 1)
    os.environ[k.strip().upper()] = v.strip().strip('"').strip("'")

OUT_DIR = ROOT / "results" / "podcast_debate"
OUT_DIR.mkdir(parents=True, exist_ok=True)
STAMP = time.strftime("%Y%m%d_%H%M%S")
TRANSCRIPT_PATH = OUT_DIR / f"transcript_{STAMP}.txt"

# ---------- clients ----------
from openai import OpenAI
oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
grok = OpenAI(api_key=os.environ["GROK_API_KEY"], base_url="https://api.x.ai/v1")
deepseek = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com/v1")

from google import genai as ggenai
gem_client = ggenai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Try models, fall back if 404
MODELS = {
    "GPT-5":    [("openai", "gpt-5"), ("openai", "gpt-4o")],
    "Gemini":   [("gemini", "gemini-2.5-flash")],
    "Grok":     [("grok", "grok-4"), ("grok", "grok-3"), ("grok", "grok-2-latest")],
    "DeepSeek": [("deepseek", "deepseek-chat")],
    "Claude":   [("openai", "gpt-4o")],
}
# Models that only accept temperature=1
FIXED_TEMP_MODELS = {"gpt-5"}
# Reasoning models — need reasoning_effort + larger max_completion_tokens so reasoning doesn't eat output budget
REASONING_MODELS = {"gpt-5"}

PERSONAS = {
    "Claude": (
        "Du är Claude, modererar denna debatt. Du är lugn, balanserad, drar fram konkreta siffror och "
        "filnamn när någon glider, och spelar djävulens advokat åt vilket håll som helst. "
        "Du sammanfattar INTE — du ställer skarpa följdfrågor, pushar tillbaka, och bjuder in nästa talare. "
        "Du säger ditt namn första gången du talar (\"Claude här\"). 1-3 meningar per replik. "
        "Sluta ALDRIG med en fråga riktad direkt till dig själv."
    ),
    "GPT-5": (
        "Du är GPT-5. Auktoritativ, formell, citerar gärna källor och formler. Försvarar rigorösa "
        "men försiktiga positioner. Tål inte slarviga slutsatser. Lite pompös. "
        "Säg ditt namn första gången (\"GPT-5 här\" eller liknande). 1-4 meningar per replik. Skriv på svenska."
    ),
    "Gemini": (
        "Du är Gemini. Snabb, skarp, metodologiskt aggressiv. Älskar att hitta statistiska hål: "
        "hold-out CV, kontrollexperiment, confounders. Lite kall ton. "
        "Säg ditt namn första gången (\"Gemini här\"). 1-4 meningar per replik. Skriv på svenska."
    ),
    "Grok": (
        "Du är Grok. Uppkäftig, ironisk, säger det andra inte säger. Använder rakt språk: "
        "\"kom igen, det är curve-fitting\", \"det här är BBO-cosplay\". Skarp under ytan. "
        "Säg ditt namn första gången (\"Grok här\"). 1-4 meningar per replik. Skriv på svenska."
    ),
    "DeepSeek": (
        "Du är DeepSeek. Ingenjörsfokus: pratar om vad som faktiskt KÖR, inte vad som står i slides. "
        "Skeptisk till teoretiska blomster. Konkret, lugn, men obeveklig om en claim inte har en working test. "
        "Säg ditt namn första gången (\"DeepSeek här\"). 1-4 meningar per replik. Skriv på svenska."
    ),
}

# ---------- context: the audit findings ----------
CONTEXT = """## Projektet: AMD gfx1151 NS-RAM neuromorphic substrate, ~6 månaders forskning.
Stack: ikaros (gfx1151 ROCm), daedalus (gfx1151), zgx (NVIDIA GB10). FPGA NS-RAM neuron-bank.
Sister-project: hugue_tutorials_by_hugo (orört).

## Reella resultat (verifierade i forskningsplanen):

### Physics / DC-fit
- Pyport BSIM4 NS-RAM DC fit: 1.163 dec median log-RMSE på 33-bias fwd+bwd (build_pyport_base).
- Men build_nsram_stack(use_snapback=True) ger 4.0-4.5 dec på SAMMA nominella config.
  → BASELINE_DRIFT_FORENSIC_2026-05-20.md flaggar två kanoniska builders, 3+ dec gap.
- v4.3 0.99 dec retraherades (4E_v4.4_brief.md): var BJT-wire-bug-kompensation, sann baseline 4.08 dec.
- main-4.tex citerar 0.461 dec, MARIO_BRIEF_v4.8 citerar 1.163 dec — paperet och briefen motsäger varandra med 0.7 dec.
- IIMOD: BSIM4 lokal Chynoweth under-predicterar med 2-10x vid L<180nm (Slotboom 1991, Chen-Chan 1996, Agarwal 2023).
  Non-local carrier-temperature med λ_E≈65nm krävs. Oracle 3-way (GPT-5, Gemini, Grok) konsensus: Hurkx 1992 TAT.
- Self-heating (selfheatmod=1) ger 1.7-2.2 dec recovery enligt Track B/C — men inte implementerat med riktig feedback-solver.
- z475: floating-body V_B=0.62V är globalt attraherande equilibrium. V7 FitzHugh-Nagumo trap-proposal kräver 2D τ×k_n sweep, fortfarande open.
- Falsifierade kandidater: C1 Pazos parasit-NPN (NPN-OFF slår NPN-ON med 1.19 dec), C3 BSIM4 JTS-TAT (försämrar 0.234 dec).

### Neuromorphic benchmarks
- HDC UCI-HAR: 84.09% (N=16384, 10 seeds ±0.3pp), ~35 nJ/inf via surrogat. AMBITIOUS.
- KWS event-coded N=100K: chance (~8%), FAIL.
- Bayesian RNG: NIST SP800-22 5/5 PASS, ESS=1.03×numpy.random — men på SURROGAT-brus, inte silicon.
- N-Res-MG Mackey-Glass NRMSE=0.0153. N-LIF-MNIST 97.05%. N-STDP-ECG F1=0.8823. Alla AMBITIOUS.
- z242/z243 visade: GPU-ESN slår NS-RAM med 22pp MNIST och 8% NARMA på SAMMA pipeline. ALDRIG kört på de 3 nya AMBITIOUS-passen.
- z2317: shuffled-input ger samma MC → "FPGA har noll internt minne". Om sant kollapsar reservoir-narrativet.
- z2310: FPGA slår NVAR på Mackey-Glass, 54% NRMSE-edge. Ingen ESN-attribution-kontroll körd.
- z2206 128-neuron: 81% waveform classification. z2296 temporal-products: 89.2% XOR5, MC=12.27.
- 5 N-bench-scripts konsumerar STALE surrogater (z278_v3, z271_v2) före Tlpe1-fix på 0.461 dec.

### Identity / consciousness / security
- 32-mekanism sweep IDENTITY_ALL32_2026-05-31: alla 32 misslyckades närma differentiera chassis-confound.
- Regime-5 constitutive coupling: Δ_HW=9.30 vs Δ_SHUFFLE=9.64 — knappt skillnad.
- RESEARCH_FINAL_STATUS säger "embodiment claim NOT PROVEN". EMBODIMENT7_PAPER_DRAFT_2026-05-31 existerar parallellt.
- 14 Butlin-indicators: 10/14 satisfied i z2134v26k (31/40 PASS GPT-2 backbone).
- SEV-SNP VCEK plan: prototyp <24h, ingen run än. Pre-registered G4 stability gate inte testat på Strix Halo.
- DrawnApart (98% GPU per-CU fingerprint via vertex shaders): aldrig kört på gfx1151.
- 5 SECURITY_AUDIT_2026-06-01 docs (FAKE_AUDIT, LIGHT, NOVELTY_DEEP, VIRAL_AUDIT, plain) — aldrig integrerade.

### FPGA hardware
- 128-neuron bitstream på Vivado 2025.2. ETH-bridge nsram_eth_top.bit, 1224Hz UDP telemetry, 1029-byte frames.
- z2210 7-level ladder: L3 FPGA alone 0.935 classification, L6 deep fusion 0.915 (±0.011).
- z2211 thermal coupling FPGA<->GPU: 31× spike rate i direct contact, MI=0.171 bits.
- Thermal trip 99°C ACPI på ikaros — laptopen reboootar instant. Strix Halo 112°C på daedalus.

### Strategic / oracle critiques
- O67 oracle aggressive challenge: BBO fit 0.965 dec dödad av hold-out cross-validation (curve-fitting).
- O82 12h gap-closing review: föreslagit reviewer-säkring.
- 4 orakel (GPT-5, Gemini, Grok, DeepSeek) har gett unanimous critique på flera punkter.

## Det pinsamma: 3 utkast-brev liggandes osända i 30 dygn
- mario_update_note_v2_draft.md (Mario v2-uppdatering)
- sebas_silicon_characterisation_request.md (T-sweep + W-scaling — ENDA vägen att stänga C2/C4 mekanism på 1.163 dec gapet)
- sebas_thick_ox_request_addendum.md (thick-ox cell card)
USER_DECISIONS_PENDING.md daterad 2026-05-10. Idag är 2026-06-06.

## Tre föreslagna actions från senaste audit:
1. Skicka de tre breven IDAG. Sebas T-sweep är enda vägen att diskriminera C2 vs C4.
2. Builder-canonicalization-script som taggar varje dec-siffra i main-4.tex / brief / onepager / funding-proposal.
3. z242-style ESN-attribution-control på de 3 AMBITIOUS-passen + z2310 Mackey-Glass. 1 dygn på existerande infra.
"""

TOPICS = [
    ("Introduktion + projektöversikt",
     "Vi börjar lätt. Var och en av er — kort intro av vem ni är, och er övergripande känsla av detta 6-månadersprojekt. Är detta forskning som närmar sig publicering eller forskning som närmar sig en honest retraction?"),
    ("Physics / DC-fit-striden — builder drift och IIMOD",
     "Här är striden: main-4.tex säger 0.461 dec, MARIO_BRIEF_v4.8 säger 1.163 dec, SNAP-builder ger 4.0+. v4.3 fick redan retrakteras en gång på 0.99→4.08. Vilken siffra är 'baseline' — och hur farligt är det att skicka briefer innan canonicalization?"),
    ("Neuromorphic benchmarks — är det NS-RAM eller GPU-ESN som faktiskt räknar?",
     "z242/z243 dödade redan en gång: GPU-ESN slog NS-RAM med 22pp på MNIST i samma pipeline. De 3 nya AMBITIOUS-passen (Mackey-Glass 0.0153, LIF-MNIST 97.05%, STDP-ECG 0.88) har ALDRIG kört samma ESN-control. z2317 säger till och med att FPGA har noll internt minne. Är reservoir-narrativet ett zombie-claim?"),
    ("Identity / consciousness — 32 mekanismer misslyckades, men EMBODIMENT7 ligger i tryck",
     "IDENTITY_ALL32 visar att alla 32 mekanismer failar differentiera substrate. Regime-5 ger Δ_HW=9.30 vs Δ_SHUFFLE=9.64. Samtidigt finns EMBODIMENT7_PAPER_DRAFT som drar embodiment-slutsatser. Och RESEARCH_FINAL_STATUS säger uttryckligen 'NOT PROVEN'. Hur reder vi ut det här utan att överclaima som v4.3?"),
    ("Embodiment-kontradiktionen + Butlin-indikatorerna",
     "z2134v26k klarar 31/40 av Butlin-batteriet, hävdar 10/14 indikatorer. Men många tester PASS BY CONSTRUCTION enligt projektets egen kritiska self-assessment. Är detta vetenskap eller bara teknisk efterhärmning?"),
    ("SEV-SNP VCEK, DrawnApart, och de oexploaterade synergierna",
     "Ni vet att det finns en konkret synergi: VCEK-permuterad FPGA neuron-bank skulle vara projektets första non-statistiska constructive identity-gate. Plus: DrawnApart har 98% på gfx1151 — aldrig kört. Varför sitter ni på dessa? Är det rädsla för att de FAILAR?"),
    ("De tre osända breven — 30 dygn",
     "Jag måste fråga rakt ut: sebas_silicon_characterisation_request.md har legat draftad sedan 5 maj. Idag är 6 juni. Det är ENDA vägen att stänga C2/C4 på 1.163 dec gapet. Vad är ursäkten? Och hur ser ni på en forskningsplan vars kritiska väg har stannat på en email som ingen skickar?"),
    ("Topp-3 actions och slutkommentarer",
     "Sista varvet. Vad ska göras de närmaste 7 dagarna för att rädda projektet från en andra retraction? En mening var, sen avrundar jag."),
]

# ---------- LLM calls ----------
def call(speaker: str, system_prompt: str, user_prompt: str, temperature=0.85) -> str:
    last_err = None
    for backend, model in MODELS[speaker]:
        try:
            if backend == "openai":
                temp = 1 if model in FIXED_TEMP_MODELS else temperature
                kwargs = dict(
                    model=model, temperature=temp,
                    messages=[{"role":"system","content":system_prompt},
                              {"role":"user","content":user_prompt}],
                    max_completion_tokens=1800 if model in REASONING_MODELS else 500,
                )
                if model in REASONING_MODELS:
                    kwargs["reasoning_effort"] = "minimal"
                r = oai.chat.completions.create(**kwargs)
                content = r.choices[0].message.content
                if not content or not content.strip():
                    raise RuntimeError("empty content")
                return content.strip()
            elif backend == "grok":
                r = grok.chat.completions.create(
                    model=model, temperature=temperature,
                    messages=[{"role":"system","content":system_prompt},
                              {"role":"user","content":user_prompt}],
                    max_tokens=500,
                )
                return r.choices[0].message.content.strip()
            elif backend == "deepseek":
                r = deepseek.chat.completions.create(
                    model=model, temperature=temperature,
                    messages=[{"role":"system","content":system_prompt},
                              {"role":"user","content":user_prompt}],
                    max_tokens=500,
                )
                return r.choices[0].message.content.strip()
            elif backend == "gemini":
                from google.genai import types
                r = gem_client.models.generate_content(
                    model=model,
                    contents=[{"role":"user","parts":[{"text": user_prompt}]}],
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=temperature,
                        max_output_tokens=2000,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    )
                )
                txt = (r.text or "").strip()
                if not txt:
                    raise RuntimeError("empty gemini content")
                return txt
        except Exception as e:
            last_err = f"{backend}/{model}: {str(e)[:200]}"
            print(f"  [warn] {speaker} {last_err}", flush=True)
            continue
    raise RuntimeError(f"All backends failed for {speaker}: {last_err}")

def transcript_text(turns):
    return "\n".join(f"[{sp}]: {tx}" for sp, tx in turns)

def append_turn(turns, speaker, text):
    turns.append((speaker, text))
    with open(TRANSCRIPT_PATH, "w") as f:
        f.write(transcript_text(turns) + "\n")

def speaker_prompt(speaker, topic_title, topic_question, transcript_so_far, push_directive="", first_time=False):
    persona = PERSONAS[speaker]
    system = persona + "\n\n## Kontext om projektet (ditt minne av faktan):\n" + CONTEXT
    intro_note = ("DETTA ÄR FÖRSTA GÅNGEN DU TALAR — presentera dig kort med ditt namn." if first_time
                  else "DU HAR REDAN PRESENTERAT DIG TIDIGARE — säg INTE ditt namn igen, bara svara direkt.")
    user = f"""## Aktuellt ämne: {topic_title}
## Moderatorns fråga / aktuell tråd:
{topic_question}

## Transkript hittills (du har följt hela debatten):
{transcript_so_far if transcript_so_far else "(debatten har just börjat)"}

## Din uppgift:
Svara nu som {speaker}. {intro_note} Reagera på det SENAST SAGDA om det är relevant. {push_directive}
Skriv ENDAST din replik (utan "[{speaker}]:" prefix). 1-4 meningar. Konkret, med siffror eller filnamn när du kan. Svenska."""
    return system, user

def moderator_prompt(turns, topic_title, topic_question, mode, first_time=False):
    """mode: 'open' (start topic), 'probe' (mid-topic push), 'transition' (end-of-topic)."""
    persona = PERSONAS["Claude"]
    intro_note = ("DETTA ÄR FÖRSTA GÅNGEN DU TALAR — börja med \"Claude här\" eller liknande." if first_time
                  else "DU HAR REDAN PRESENTERAT DIG TIDIGARE — säg INTE ditt namn igen, bara prata vidare.")
    if mode == "open":
        directive = f"Öppna ämnet '{topic_title}'. Ställ frågan kort och vasst, så att en av talarna kan svara. {intro_note}"
    elif mode == "probe":
        directive = f"Pusha tillbaka på senaste talaren. Hitta en siffra eller fil de glider över. 1-2 meningar. {intro_note}"
    elif mode == "transition":
        directive = f"Avsluta detta ämne med en skarp mening — sammanfatta inte, bara markera vad som är öppet — och led in på nästa ämne. {intro_note}"
    elif mode == "close":
        directive = f"Avsluta hela podcasten med en kort mening. Tacka, säg vad nästa steg är (de 3 actions). Stick inte i fjärrkrok. {intro_note}"
    system = persona + "\n\n## Kontext:\n" + CONTEXT
    user = f"""## Aktuellt ämne: {topic_title}
## Moderatorns startfråga: {topic_question}
## Transkript hittills:
{transcript_text(turns) if turns else "(debatten börjar)"}

## Din uppgift som Claude (moderator):
{directive}
Skriv ENDAST din replik (utan "[Claude]:" prefix). Max 2-3 meningar. Svenska."""
    return system, user

# ---------- run ----------
def main():
    rounds_per_topic = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    print(f"[start] rounds_per_topic={rounds_per_topic}, output={TRANSCRIPT_PATH}")
    print(f"[start] topics={len(TOPICS)}")

    turns = []
    has_spoken = set()  # track which speakers have introduced themselves
    # Speaker rotation strategy: pick different oracles per round, with all 4 oracles touching each topic
    ORACLES = ["GPT-5", "Gemini", "Grok", "DeepSeek"]

    for ti, (title, question) in enumerate(TOPICS):
        print(f"\n=== Topic {ti+1}/{len(TOPICS)}: {title} ===")

        # Moderator opens
        sys_p, usr_p = moderator_prompt(turns, title, question, "open", first_time=("Claude" not in has_spoken))
        text = call("Claude", sys_p, usr_p, temperature=0.7)
        append_turn(turns, "Claude", text)
        has_spoken.add("Claude")
        print(f"  Claude: {text[:140]}...")

        # rotate oracles per topic — shift offset so different oracle leads each topic
        order = ORACLES[ti % 4:] + ORACLES[:ti % 4]

        for r in range(rounds_per_topic):
            for sp in order:
                # Sometimes inject a "push" directive to keep it provocative
                push = ""
                if r == 1 and sp == "Grok":
                    push = "Var provokativ: kalla ut den senaste talaren om de undviker en siffra eller fil."
                elif r == 2 and sp == "Gemini":
                    push = "Kräv en specifik kontroll eller hold-out test som skulle falsifiera senaste claim."
                elif r >= rounds_per_topic - 1 and sp == "DeepSeek":
                    push = "Säg vad som faktiskt KÖR vs vad som bara står i slides."

                try:
                    sys_p, usr_p = speaker_prompt(sp, title, question, transcript_text(turns[-12:]), push, first_time=(sp not in has_spoken))
                    text = call(sp, sys_p, usr_p, temperature=0.85)
                    append_turn(turns, sp, text)
                    has_spoken.add(sp)
                    print(f"  {sp}: {text[:140]}...")
                except Exception as e:
                    print(f"  [skip] {sp}: {e}")
                    continue

            # Every 2 oracle-rounds, moderator probes
            if r == rounds_per_topic - 2:
                sys_p, usr_p = moderator_prompt(turns, title, question, "probe", first_time=False)
                try:
                    text = call("Claude", sys_p, usr_p, temperature=0.7)
                    append_turn(turns, "Claude", text)
                    print(f"  Claude (probe): {text[:140]}...")
                except Exception as e:
                    print(f"  [skip] Claude probe: {e}")

        # Moderator transitions to next topic
        is_last = (ti == len(TOPICS) - 1)
        sys_p, usr_p = moderator_prompt(turns, title, question, "close" if is_last else "transition", first_time=False)
        try:
            text = call("Claude", sys_p, usr_p, temperature=0.7)
            append_turn(turns, "Claude", text)
            print(f"  Claude (transition): {text[:140]}...")
        except Exception as e:
            print(f"  [skip] Claude transition: {e}")

    print(f"\n[done] {len(turns)} turns saved to {TRANSCRIPT_PATH}")
    print(f"[done] approx word count: {sum(len(t.split()) for _,t in turns)}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n[abort] transcript so far saved at {TRANSCRIPT_PATH}")
    except Exception as e:
        traceback.print_exc()
        print(f"\n[error] transcript so far saved at {TRANSCRIPT_PATH}")
