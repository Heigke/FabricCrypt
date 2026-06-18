"""Generate a Swedish project-overview podcast.

Progressive depth: farmor (everyday metaphors) → mormor civilingenjör
(plain-technical Swedish) → engineer level. ~7-10 minutes.

Uses OpenAI TTS API (tts-1-hd) with a Swedish-friendly voice.
"""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / ".env"
OUT = ROOT / "results/podcast"
OUT.mkdir(parents=True, exist_ok=True)

# Load .env
for line in ENV.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    os.environ[k.strip().upper()] = v.strip().strip('"').strip("'")

SCRIPT = """
Hej. Det här är en uppdatering om FEEL × NS-RAM-projektet, hela vägen från start till nu. Jag tar det enkelt först, sen lite tekniskt, sen lite mer tekniskt. Lyssna så långt du vill.

Del ett — för farmor.

Vi försöker bygga en pytteliten datorhjärna. Inte en vanlig dator. En liten brick som både kan komma ihåg saker och räkna lite själv, ungefär som hur celler i hjärnan fungerar. En forskare i Saudiarabien, Sebastian Pazos, har tillverkat sådana bitar i kisel — riktiga fysiska små enheter. Vi i Sverige har byggt en exakt kopia av den lilla biten inne i datorn — i mjukvara — så vi kan testa massor av idéer utan att behöva tillverka nya kiselbitar varje gång. Det går mycket fortare så.

Jämför det med att bygga en modell av en bil i datorn istället för att svetsa ihop en ny prototyp varje gång du vill testa en idé. Vi har gjort vår mjukvarumodell tillräckligt bra för att den uppför sig precis som riktig kiselelektronik. Det tog flera veckor och vi hittade fem buggar i ett av världens vanligaste simulatorprogram som ingen annan upptäckt. Det är ungefär som att hitta fem stavfel i en ordbok som tusentals människor använt i tjugo år.

Sedan började vi testa: kan den här lilla biten lära sig saker? Vi gav den fyra olika uppgifter — minnas saker, räkna ut nästa siffra i en serie, känna igen mönster, klassificera vågformer. Den klarade alla fyra på en grundläggande nivå. Bra start. Sen frågade vi: vad händer om vi kopplar ihop hundra eller tvåhundra sådana här bitar? Då dyker det viktiga upp.

Det visade sig att hur man kopplar ihop bitarna spelar enormt mycket större roll än hur många man har. Vi testade fem olika sätt att koppla — rutnät där varje bit pratar med sina grannar, slumpmässiga kopplingar, glesa kopplingar, och så vidare. En särskild sort — slumpmässigt glest mönster där varje bit pratar med tio procent av alla andra — vann stort. Den gör ungefär tre gånger bättre ifrån sig på minnesuppgifter än andra mönster.

Vi skriver just nu ett rekommendationsbrev till en professor i Saudiarabien som ska tillverka nästa generation av dessa kiselbitar. Brevet ska skickas inom några dagar.

Del två — för mormor som var civilingenjör.

På NS-RAM-cellen sitter två transistorer, en bipolär parasit, och en flytande kropp. Pazos har visat att den flytande kroppens lavinurladdningsmekanism kan emulera neural fyrning — det är artikeln i Nature Electronics tjugohundratjugofem. Vår uppgift är att porta hans BSIM4 v fyra punkt åtta tre-spice-modell till PyTorch så vi kan göra automatisk differentiering, GPU-acceleration, och vid behov koppla på en lärande loop.

Phase A är stängd: median ett komma noll noll två dekader RMSE mot ngspice-mätningar över trettiotre bias-punkter. De fem ngspice-buggarna vi hittade är dokumenterade — främst tysta parser-fel där flerassigneringsrader på .model-rader får andra och tredje tilldelningarna tappade, och .param-substitutioner som tyst faller tillbaka till BSIM4-default-värden. Vår port måste imitera dessa buggar eftersom Sebastians kort ursprungligen kalibrerades mot ngspice — kortet förlitar sig implicit på den trasiga parsern.

I phase B-tester använde vi cellen som ett reservoir-computing-substrat: vi driver en pulserad spänning på drain-noden och drar ut log-strömmen som feature-vektor. Sen tränar vi en linjär utgångsläsare på fyra benchmark-uppgifter: minneskapacitet, NARMA-tio, temporal-XOR, och fyra-klass-vågformsklassificering.

Den stora fyndet kom när vi sedan testade fem kopplings-topologier. Vi körde två sweep-experiment z hundra nitton och z hundra tjugoett. Bägge visade att Erdős-Rényi-glest-koppling med tio-procent densitet slår fyra-grannars-rutnät med över femtio procent på minneskapacitet, och med tjugo-nio procent på temporal-XOR. Detta håller vid både rho-noll-komma-nio och kanonisk ett-genom-roten-ur-N spektral skalning, alltså W-rec-skalning-robust.

Mekanismen är att glesa slumpmässiga kopplingar funkar som en expander-graf — stort spektralt gap, snabb mixning, decorrelerade slumpmässiga projektioner. Rutnätet ger en nästan-Laplacian som har för smala Fourier-moder och därför hög feature-kollinearitet på grund av att grannar delar drain-common-mode. Det förklarar precis varför vi sett feature-collinearity-tak i tidigare N-skalning-experiment.

Del tre — engineer-nivå.

Vi körde en oracle-review (O thirteen) av OpenAI gpt-five och Gemini två-punkt-fem-pro. Båda gav grönt-ljus med ett huvudvillkor: stärk C-tre-formuleringen i brevet. Båda flaggade också att purt resistiv silikon-koppling är symmetrisk och icke-negativ; om sparse-fördelen kräver tecken-diversitet finns en silikon-implementeringsrisk.

I sista iterationen körde vi z hundra tjugotvå för att kvantifiera den risken. Tre villkor på ER-glest-topologi med kanonisk ett-genom-roten-N-skalning: A med båda tecken (z hundra tjugoett-baseline), B med endast positiva vikter, C med slumpmässigt teckenflippande över samma magnitud-fördelning som A.

Resultatet är skarpt: B kollapsar — minnes-kapacitet faller från två-komma-nittio till noll-komma-fyrtiotre, parad-t minus elva. XOR från noll-komma-åttiotvå till noll-komma-femtiosex, parad-t minus sju-komma-fem. C däremot är statistiskt ekvivalent med A — minus en-komma-en på minne, minus en-komma-noll på XOR, båda icke-signifikanta. Slutsatsen: tecken är väsentligt, men magnitud-fördelningen ensam räcker så länge båda tecken är representerade.

Detta påverkar tape-out-rekommendationen direkt. NS-RAM:s fysiska shared-bulk-koppling via R-bulk är intrinsiskt positiv-bara. Vi har därför uppdaterat C-tre-version-två-punkt-tre-tillägget med tre tecken-inverter-implementeringsalternativ: source-follower-inverter-cell (preferred, plus-trettio-procent-area), per-cell-signed-readout-pair (mindre area men mer komplex routing), eller input-side-dithering (billigaste men begränsar input-protokollet). M-nio-mask-layout-review är beslutsdeadline för vilket alternativ som väljs.

Brevet till Mario Lanza är nu i sitt starkaste försvarbara skick: phase A stängd, fem benchmarks rapporterade, monotonisk dichotomy-tabell, alla begränsningar dokumenterade, dubbel-oracle-grön-ljus, och nu också silikon-tecken-asymmetri-caveaten korrekt fångad och åtgärdad. Sex sidor, fyra hundra fem kilobyte. Skickas när du auktoriserar.

Det är allt för nu. Tack för att du lyssnade.
"""

# OpenAI TTS
from openai import OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

print(f"[podcast] script length: {len(SCRIPT)} chars")
print(f"[podcast] calling OpenAI tts-1-hd with voice=onyx (deep neutral)...")

# tts-1-hd has a 4096 char limit per call. Split if needed.
chunks = []
remaining = SCRIPT.strip()
while remaining:
    if len(remaining) <= 4000:
        chunks.append(remaining)
        break
    # Find a good split point (sentence boundary)
    split = remaining.rfind(". ", 0, 4000)
    if split < 1000:
        split = 4000
    chunks.append(remaining[:split+1])
    remaining = remaining[split+1:].lstrip()

print(f"[podcast] split into {len(chunks)} chunks")

import time
audio_parts = []
for i, ch in enumerate(chunks):
    print(f"  chunk {i+1}/{len(chunks)}: {len(ch)} chars")
    t0 = time.time()
    resp = client.audio.speech.create(
        model="tts-1-hd",
        voice="onyx",
        input=ch,
        response_format="mp3",
    )
    audio_parts.append(resp.content)
    print(f"    {len(resp.content)} bytes in {time.time()-t0:.1f}s")

# Concatenate MP3 chunks (simple binary append works for tts-1 output)
out_path = OUT / "feel_nsram_overview_2026_05_03.mp3"
with open(out_path, "wb") as f:
    for p in audio_parts:
        f.write(p)
print(f"\n[podcast] saved: {out_path}")
print(f"[podcast] total size: {out_path.stat().st_size / 1024:.0f} KB")
