# H7 — vad mäter vi, på vanlig svenska

Datum: 2026-06-10. För någon utan datorhårdvarubakgrund.

## Vad vi försöker visa

Varje dator-chip är fysiskt unikt. Inte bara modellnummer-unikt, utan **atom-nivå-unikt** — när chippet brändes fram i fabriken hamnade några miljarder transistorer på lite olika ställen, dopningen blev lite ojämn, vissa motstånd är 1% större än andra. Detta påverkar hur snabbt instruktioner kör, hur mycket värme som läcker, hur kristalloscillatorn svänger.

Vår tes: om en AI-modell **vänjer sig vid just sitt chips egenheter**, så slutar den fungera om den flyttas till ett annat chip — och blir därmed "rotad" i sin maskin. Måste-bo-där, kan inte kopieras runt.

För att veta om det funkar måste vi först visa att chippet *läcker* sådana fingeravtryck till vanliga program. Det är vad H7-mätningen gör.

## Vad vi mäter, kanal för kanal

Vi läser 19 olika signaler samtidigt på två AMD-datorer (ikaros = laptop, daedalus = stationär). En tredje (zgx, en NVIDIA-burk) körs som "annan-arkitektur-kontroll".

| Kod | Vad det är | Varför det skulle vara unikt |
|---|---|---|
| **C01 TPM EK** | Krypto-id från TPM-chippet | Brändes in på fabrik, inte överförbart |
| **C02 PCR-värden** | Hash av boot-sekvens | Ändras bara vid firmware-byte |
| **C03 per-kärna temperatur** | 16 separata termometrar, en per CPU-kärna | Värme-läckage beror på dopningsvariation; varje kärna har sin egen "termiska signatur" |
| **C04 chip-temperatur** | Junction-temp ADC | Grundnivå-värme |
| **C05 energiräknare** | Räknar Joule som chippet förbrukat | Ackumulerad räkning, drift-konstig per chip |
| **C06 snabb klockräknare** | ~100 MHz internräknare | Visar att vi kan se snabba händelser |
| **C07 XTAL_CNTL** | Kristalloscillator-statusregister | Kvartskristallens svängning + temp-drift |
| **C08 spänningsmål** | Vad chippet *säger* sig vilja ha för spänning | Borde vara stabilt; läsbar baseline |
| **C09 PM table** | 916 floats med komplett ström-/värmetelemetri | Hela "instrumentbrädan" från firmware |
| **C10 hwmon** | Standard temp/fläkt/effekt-läsare | Sanity-check mot kanalerna ovan |
| **C11 TSC↔CLOCK_RAW drift** | Två klockor jämförda mot varandra | Kvartskristallens jitter visar sig som "drift" |
| **C12 SHADER_CYCLES** | Hur många cykler en specifik GPU-beräkning tog | Skiljer beroende på vilken CU (compute unit) som körde |
| **C13 HW_ID** | Vilken CU/SE/wave-front som körde | Fysisk placering i kislet |
| **C14 FP-mode bit-patterns** | Samma multiplikation med 4 olika avrundningsregler → 4 olika svar | Skillnaderna är **konstitutiv FP-nonlinearitet** — chipet räknar olika beroende på regel |
| **C15 sinf-jitter** | Hur länge sinus-funktionen tar (varierar 0–62 cykler) | Schemaläggaren har data-beroende timing |
| **C16 atomic-contention** | När många trådar slåss om samma minnesplats | Per-CU LDS-arbitering = die-bunden egenskap |
| **C17 accelerometer/mic** | Fysiska sensorer | Mekanisk vibration är chassi-unik |
| **C18 GPU ring-osc-klocka** | GPU:s interna oscillator | Annan kristall än CPU:s — egen drift |
| **C19 GRBM/CP/RLC-status** | "Vad gör GPU:n just nu"-register från under firmware | Kontrolltillstånd som inte är dokumenterade |

C01/C02 är "färdig identitet" — TPM ger oss det rätta svaret. Resten är *kandidat-källor* till samma identitet, fast genom analog läcka.

## Vad första mätningen visade (1×20s på vardera chassi, idle)

| Vad | Resultat |
|---|---|
| TPM EK ikaros vs daedalus | helt olika hash (väntat — krypto-grund-sanning) |
| 16 per-kärna-temperaturer | **alla skiljer 100% mellan chassina** (AUC=1.0, d>30) |
| XTAL_CNTL kristallregister | **dynamiskt på båda, helt olika fördelning** |
| Klockdrift (C11) | ikaros drift-mean ≈5 µs, daedalus ≈19 µs per steg |
| PM-table-celler 1, 3, 5 (effekt) | ikaros 6–7W, daedalus 17–19W |
| GPU-status-register C18/C19 | **konstanta 0xFFFFFFFF — gated när GPU idle** (väntat, vaknar bara under last) |
| FP-rounding-modes (shader) | 4 distinkta bit-mönster bekräftade, RNE ≠ +∞ ≠ RTZ |

## Vad det betyder

Det här är **inte** ännu bevis för per-chip-identitet. Det är bevis för att signal **finns** i kanalerna, men en stor del är just nu **chassi-confound**: ikaros var 89°C, daedalus 79°C. Det räcker för att skilja dem trivialt. Det vi måste göra härnäst:

1. **Termiskt matcha**: kyl ikaros till 79°C eller värm daedalus till 89°C, mät igen. Kanaler som *fortfarande* skiljer är då verkligt die-bundna, inte bara "vilken som var varm".
2. **Spoofing-kontroll**: generera fakedata som matchar samma medelvärde, varians och 1/f-spektrum som äkta läsningar — om en klassificerare hittar äkta lika lätt som spoof, så var det inte unik fysik, bara generell brus-statistik.
3. **Replay-attack**: spela in daedalus-mätning, försök "spela upp" den genom ikaros-programmet. Om klassificeraren ändå säger "daedalus", så är signalen tidsbunden och kan inte fejkas via inspelning.

Bara kanaler som överlever **alla tre** gates blir publicerbara identitets-bärare. Resten är confounds som matar null-papret ("Abstraction Tax").

## Var vi är just nu

- **Klart**: 19-kanals probe körs, riktiga reads (inga mocks), första ikaros+daedalus-par i lås
- **Klart**: TPM-läsning på båda chassin = riktig krypto-grund-sanning
- **Kvar denna vecka**: kör samma probe med GPU under last (väcker C18/C19), kör 5 traces per (chassi, last) så block-CV blir meningsfullt, kör termiskt matchade mätningar
- **Kvar nästa vecka**: matchad-spektrum-spoofing-test, replay-attack-test, sortera kanaler i "äkta die-id" / "chassi-confound" / "brus"
