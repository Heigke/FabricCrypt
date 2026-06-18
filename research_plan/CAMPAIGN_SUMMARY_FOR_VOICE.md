# NS-RAM Modellering — Kampanjsammanfattning (för röstassistent)

## Bakgrund (vad är NS-RAM och varför)

NS-RAM = Neural State RAM. Mario Lanza och Sebastian Pazos byggde en specialcell i 130nm CMOS som beter sig som en neuron — den har en "snapback" där strömmen plötsligt hoppar 3 dekader när drain-spänningen når ett tröskelvärde. Detta är en relaxationsoscillator i kisel, publicerad i Nature 640:69-76 (2025).

Vi vill simulera den i pyport (vår Python-version av BSIM4 transistormodell på AMD GPU) så vi kan köra stora nätverk av sådana här neuroner för forskning.

## Vad cellen består av

- M1 = övre NMOS-transistor
- M2 = nedre NMOS-transistor (Ln=1.8µm, dvs 10× längre än M1)
- Q1 = parasit NPN-bipolär transistor (drain=kollektor, body=bas, source=emitter)
- Body är "flytande" — har bara liten kapacitans, ingen kontakt
- Två gate-spänningar: V_G1 (M1) och V_G2 (M2)

När V_D stiger genererar impact-ionization elektron-hål-par. Hål samlas i body, V_B stiger. När V_B = 0.7V slår Q1 på och förstärker 1000× → snapback knät.

## Vad vi gjorde och vart vi är (kronologi)

**Innan kampanjen**: Vi hade pyport med BSIM3-modell, ~5 dec cell-wide RMSE från Sebas's 33 mätta IV-kurvor. Modellen producerade ingen snapback alls.

**Dag 1 — Mario's Ipos-formel**: Digitiserade hans semi-empiriska formel `Iion = Iexp + Ipow` med 5 PWL-koefficienter från slide 12.26. Lade till som body-injection. Marginal förbättring.

**Dag 1-2 — Z419-Z425 kampanj**: Body laddades inte. Hittade att V_Sint (intern source) "runaway" till V_D, vilket gjorde BJT i deep saturation där V_BE = V_BC = Ic ≈ 0.

**STÖRSTA GENOMBROTTET (S18 z430)**: Hard pin V_Sint=0 (substrat-tap, fysiskt motiverat) → cell-wide 3.9 → 1.6 dec. Snapback började synas.

**Pseudo-transient z432**: Newton-solver hittade fel root. Ersatte med pseudo-transient body-integration. Backward sweep (V_D 2→0) väljer "latched" attractor → cell-wide 1.027 dec. **Hysteresis 0.45 dec mellan forward/backward = riktig bistabilitet, inte solver-trick**.

**Visuellt nu**: Snapback-formen finns i modellen vid V_G1=0.4/0.6 med låg V_G2. Knäet hamnar ca 0.2V för tidigt och magnitud ~1 dec för låg.

## Aktuella experiment (pågående)

1. **z438 knee-calibration** — sweepar alpha0 (impact ionization styrka) och Bf (BJT-förstärkning) i 4x4 grid. Försöker shifta knäet höger och höja magnituden. Bästa hittills 0.916 dec.
2. **z439 smooth pseudo-transient** — implicit Euler + BDF2 för att fixa numeriska zigzag i kurvorna.
3. **z441 V_G1-V_G2 sigmoid gate** — direkt test av S27's fynd att V_G1-V_G2≥0.20V är snapback-gränsen.

## Vad är ärligt rätt och vad är fortfarande fusk

**Rätt fysik (utan fusk)**:
- BSIM4 v4.8.3 ekvationerna kopierade exakt från ngspice
- Mario's Ipos formel digitiserad direkt från hans slide
- Snapback-fysiken är på rätt plats
- Bistabilitet finns i modellen som i mätningen

**Fortfarande fusk eller approximation**:
- V_Sint=0 hard pin (real silicon har 1-10Ω, ej 0Ω)
- Pseudo-transient C_B = 1e-18 F (Mario säger 1 fF = 1e-15 F) — vi använder det som solver-trick
- Knäet skiftat 0.2V och magnitud ~1 dec under

## Status just nu (cell-wide RMSE i decader)

| Version | Cell-wide |
|---|---|
| Innan kampanj | ~5 dec, ingen snapback |
| z430 V_SINT_PIN | 1.619 dec |
| z432 pseudo-transient backward | 1.027 dec |
| z438 knee-calib bästa hittills | **0.916 dec** |
| AMBITIOUS-mål | < 0.7 dec |

## Vad som händer härnäst

- Pågående experiment ETA ~2-3h
- Om någon ≤0.7 dec → publikationsklar modell
- Om alla 0.9-1.0 dec → acceptera och börja stora nätverkssimuleringar

## Vad vi använt för att hitta hit
- 4-oracle paneler (OpenAI GPT-5, Gemini 2.5 Pro, Grok-4, DeepSeek)
- Publicerade Nature-paper från Mario/Sebas
- Brute-force grid scans + bifurkations-analys
- Multipla solver-strategier
- ~70+ experimentella subagent-spår

Det är där vi står.
