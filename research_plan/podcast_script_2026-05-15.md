# NS-RAM Podcast — pedagogisk genomgång 2026-05-15

[del 1 — vad är NS-RAM]

Välkommen. Idag ska vi gå igenom ett pågående forskningsprojekt om något som kallas NS-RAM, eller "Neural State RAM". Vi börjar från grunden och bygger upp förståelse steg för steg, så att alla kan hänga med, oavsett om du är ingenjör eller bara nyfiken. Vi går också igenom var vi är just nu, vad som funkar, vad som inte funkar, och varför.

Föreställ dig en liten elektronisk komponent, mindre än en damm-tussa i ett dammkorn, som inte bara kommer ihåg ettor och nollor, utan kan uppföra sig lite som en hjärncell. Den kan integrera signaler över tid, spika, och glömma. Det är NS-RAM-cellen. Den är uppfunnen av en forskare som heter Mario Lanza, och hans doktorand Sebastian, vi kallar honom Sebas, hjälper oss att förstå hur den verkligen beter sig på kisel.

[del 2 — cellens fysik, enkelt]

Tänk dig två kranar bredvid varandra, kopplade till en vattenkanal. Den övre kranen heter VG1, den nedre VG2. Mellan kranarna finns en liten reservoar — en behållare som kan samla upp lite vatten. I verkligheten är "kranen" en transistor och "vattnet" är ström. Reservoaren mellan dem kallas för cellens "body", på svenska kanske kropp eller bål.

Det speciella med NS-RAM är att den här reservoaren är **flytande** — den är inte kopplad till någon avlopp. Det betyder att laddning som hamnar där stannar kvar tills något annat händer. Det är minnet. Cellen kommer ihåg hur många elektroner som finns i body, och beroende på det blir den lättare eller svårare att slå på.

När du sätter spänning på VG1 öppnar du den övre kranen lite. Vatten börjar rinna. Men det är inte allt — vid hög spänning händer något konstigt: vattnet börjar slå av sig själv i strömmen, det blir turbulens. På fysikspråk kallas det "impact ionization" — elektroner slår av nya elektron-hål-par. Vissa av de här hålen hamnar i reservoaren. Reservoaren laddas upp. När reservoaren är tillräckligt laddad, slår den på en parasitisk transistor — en så kallad bipolär — som plötsligt släpper igenom mycket mer ström. Det är **snapback**. Strömmen hoppar plötsligt 100 till 1000 gånger högre.

Snapback är som om vattnet i en damm plötsligt började koka. Det är inte mer vatten — det är mer turbulens. Och det är just den här effekten Mario tror kan användas för att bygga konstgjorda neuroner. Snapback ger en plötslig spike. Reservoaren töms långsamt. Då går det att spika igen.

[del 3 — vad vi vill göra]

Vårt projekt har två syften.

För det första: bygga en **datormodell** av cellen som beter sig som den verkliga. Då kan vi simulera tusentals eller miljoner celler innan vi tillverkar dem. Det sparar enormt mycket tid och pengar.

För det andra: visa att en sådan **substratbaserad beräkningsmaskin** kan göra något verkligt nyttigt. Klassificera tal. Förutsäga kaos. Bearbeta 5G-signaler. Och göra det med extremt lite energi — vi pratar pikojoule, alltså miljarddels miljonjoul, per operation. För jämförelse, en sökmotor använder ungefär miljarder gånger mer per sökning.

[del 4 — modellen, hur vi bygger den]

Hur bygger man en modell av en transistor? Det finns en industristandard som heter BSIM4. Det är en stor samling matematiska formler som beskriver hur ström flödar i en transistor som funktion av spänning, temperatur, transistorns storlek och så vidare. Det är, ärligt talat, ett monster. BSIM4 har över hundra parametrar.

Vi har skrivit en Python-version av BSIM4 — vi kallar den "pyport". Den är en port från industristandardens C-kod. Detta gör att vi kan köra simuleringar på GPU och miljonceller, vilket inte är möjligt med kommersiella SPICE-simulatorer.

Att porten faktiskt fungerar har varit ett projekt i sig. Vi har hittat bug efter bug — felaktiga parser-fall, fel inkluderade i HSPICE-syntax som ngspice tyst tappar, fel mappning mellan parametrar och formler. Vi har fixat över 50 stycken över tid.

[del 5 — fitten mot mätdata]

Sebas har mätt upp en faktisk NS-RAM-cell — alltså, han har lagt sondnålar på chippet, sveptat spänningarna och plottat ström. Vi har 33 mätkurvor: tre olika VG1-nivåer (0.2, 0.4 och 0.6 volt) och elva VG2-nivåer per VG1-gren.

Vårt mål är att simuleringen ska reproducera mätningen. Vi mäter felet i logaritmiska dekader. En "dec" är en faktor 10. Så 0.5 dec betyder vi är fel med en faktor 3. 1.0 dec, en faktor 10. 2.0 dec, en faktor 100. Och 3 dec, faktor 1000.

I januari var modellen fel med 4 dec på vissa biaspunkter. Det är fyra storleksordningar. Helt fel. Efter mycket arbete har vi nu nere på 1.13 dec globalt, och 0.965 dec när vi tillåter olika parametrar för varje VG1-gren. Det är runt en faktor 9 fel i ström. Acceptabelt men inte bra.

Vårt mål är under 0.95 dec. Vi är nära. Men det finns en hake.

[del 6 — snapback-folden vi inte lyckas reproducera]

Här kommer den kritiska delen. När vi tittar på mätdata noggrant ser vi att vid Vd ungefär 1 till 2 volt — drain-spänningen — så HOPPAR strömmen i mätningen 2 till 3 dekader uppåt. Det är snapbacken — den plötsliga turbulens-explosionen.

Vår modell? Den hoppar 0.02 till 0.06 dekader. Alltså mellan en hundradel och en femtiodel av vad mätningen visar. Modellens kurva följer mätdata fint nedanför knäet, men sedan slätar den helt över snapbacken. Som om vattendammen visade tecken på koksugning men aldrig faktiskt började koka.

Vår "0.965 dec fit" är alltså egentligen bara en bra anpassning av sub-threshold-delen, alltså den lugna delen innan turbulensen. Snapback-fysiken — det som faktiskt gör NS-RAM intressant — den missar vi.

[del 7 — sju misslyckade topologi-fixar]

Vi har försökt fixa det här. Sju gånger. Varje gång med en ny hypotes om vilken topologi-element som saknas, eller vilken koppling som måste förstärkas. Här är listan:

R-43: lägg till en multiplikator på impact-ionization-strömmen. Misslyckades.

R-45: svep parametrar för djup-N-well, alltså en annan parasitisk struktur. Misslyckades.

R-47: lägg till en sub-diod i body-kretsen. Misslyckades.

R-49: lägg till en drain-body avalanche-multiplikator. Misslyckades.

R-52: applicera multiplikatorn direkt på ström istället för injektion i body. Misslyckades.

R-53: kaskadera — multiplicera halvt och injicera halvt. Misslyckades.

R-55a: porta hela referenstopologin från Mario's Zenodo-arkiv — fem nya element samtidigt: en zener-diod, en sub-MOSFET för body-bias, korrekta BJT-parametrar, spänningsberoende breakdown-formler. Misslyckades.

Detta sista var det vi kallade "kill-shot test" — om även det misslyckades, var vi överens om att retraktera modell-programmet. Det misslyckades.

[del 8 — varför misslyckades alla?]

Vi konsulterade tre stora AI-modeller — OpenAI, Gemini och Grok — för att ge en hård kritik. Alla tre var överens. Den verkliga snapback-fysiken kräver en **regenerativ loop** som stänger på sig själv. Tänk dig en mikrofon mot en högtalare — det gnisslar för att ljudet hamnar i en feedback-slinga. På samma sätt behöver impact-ionization öka body-spänningen, som öppnar bipolen, som drar mer ström, som ger mer impact-ionization, och så vidare. Det är en lavin.

Vår modell har varje steg separat, men kopplingen är för svag. Body-spänningen stiger inte snabbt nog för att slå på bipolen innan strömmen klamras. Det är som att ha varje del i ett ekkosystem men inte tillåta dem att förstärka varandra.

Frågan är: är det vår simulator som är fundamentalt fel, eller är det fysiken vi har förstått fel? Den frågan har vi inte besvarat än.

[del 9 — vad vi gör nätverken med]

Parallellt med modelleringen har vi byggt **nätverkssimuleringar**. Det betyder: ta tusentals till miljonceller, koppla ihop dem i topologier — ringar, gitter, slumpmässiga grafer — och se om nätverket kan göra något användbart.

Och här har vi faktiskt vunnit ibland. Vi har identifierat fem applikationer där NS-RAM-substratet slår digitala baselines:

Ett. Sinus-frekvensigenkänning. Klassificera om en ton är 1 Hz, 2 Hz, 4 Hz eller 8 Hz. NS-RAM 97.8 procent rätt. Traditionell leaky-integrate-and-fire neuron, 86 procent. Slumpprojektion, 24 procent — alltså chans-nivå.

Två. Lorenz-attraktor prediktion. Att förutsäga ett kaotiskt system. NS-RAM gör fel ungefär en sjättedel mindre än en LSTM, ett standard neuralt nätverk för tidsserier.

Tre. Edge-cascade. En enkel ljudklassificerare som vaknar bara när något händer. Sparar 20 gånger energi jämfört med att alltid lyssna med en digital chip.

Fyra. Bayesisk slumptals-generator. Använd cellens fluktuationer som äkta slump. KL-divergens 3.1 gånger bättre än PyTorch standard.

Fem. 5G equalizer. Bearbeta radio-signaler. Matchar en tränat neuralt nätverk i kvalitet men beräknad teoretiskt 86 gånger lägre energi.

[del 10 — fem retraktioner ärligt]

Men. Vi har också retraherat fem påståenden. Här är de:

Ett. Memory palace. Idén var att lagra minnen i ett rumsligt mönster av celler. Vi fick 95.7 procent recall, vilket lät bra. Sedan ablation-testade vi: ersatte hela substrat-arkitekturen med en vanlig Python-ordbok som hash-tabell. Den fick exakt samma resultat. NS-RAM bidrog alltså inget. Retraherades.

Två. Tre alternativa minnesarkitekturer. Alla föll med 88 till 100 procentenheter när vi testade dem.

Tre. Method-of-Loci med STDP-förtränat substrat. STDP är synaptic plasticity. Den propagerade vågor genom celler men FÖRSÄMRADE diskrimineringen med 27.7 procentenheter jämfört med slumpmässiga vågor. Retraherades.

Fyra. Neuromorf glömmande minne. Tanken var att substratens långsamma sönderfall skulle ge naturlig glömska. Men en digital räknare med fönsterstorlek slog oss med 22 procentenheter. Retraherades.

Fem. STDP adaptiv EKG-filter. Substratet var för långsamt för QRS-skala plasticitet. Vb-spänningen mättades och alla celler spikade lika oavsett input. STDP-vikt-uppdateringen blev bit-identisk med frusna vikter. Retraherades.

Vi är stolta över retraktionerna. De är vetenskaplig hederlighet i praktiken. Vi hellre falsifierar oss själva nu än överskålar i en publikation.

[del 11 — UCI-HAR pinsam upptäckt]

För några timmar sedan körde vi ett publication-grade test på UCI-HAR, ett standardiserat dataset för aktivitetsigenkänning på smartwatches — sex klasser, 561 features.

Vårt NS-RAM HDC-system fick 76 procent vid N lika med 1024 hyperdimensioner. Vi var stolta — vi hade tidigare påstått 83.86 procent som låst resultat.

Men. När vi körde sklearn med en enda rad linjär ridge-regression som baseline, fick vi 96.2 procent. En enda rad. En kommandoraderad operation. 20 procentenheter över vårt fancy substrat.

Det är en hederlig och pinsam upptäckt. UCI-HAR är fel benchmark för NS-RAM — den är för enkel och redan löst av trivial linjär algebra. Vi har testat med fel motståndare.

[del 12 — produktionsskala]

En annan stor lärdom från ikväll. Vi körde DS-N17, ett massivt skalningstest. Mackey-Glass tidsserier-prediktion vid N lika med 10 tusen, 100 tusen och 1 miljon celler. Layered topology, alltså ett strukturerat nät.

NS-RAM skalar monotont. NRMSE 0.747 vid 1 miljon, ner från 0.873 vid 10 tusen. Det är bra — substratet bryr sig inte om skalan, det funkar.

Men LSTM, en vanlig digital arkitektur, fick 0.08 NRMSE. Tio gånger bättre. Vid alla skalor. NS-RAM är skalbart men inte konkurrenskraftigt på det här problemet.

Det är där vi är. Substratet har niche-styrkor — sinusklassificering, Lorenz, RNG, equalizer, edge cascade — men det är inte en universell hjärnersättare.

[del 13 — sju gap i fysisk korrekthet]

För att kunna säga "vår modell beskriver verkligheten korrekt" behöver vi sju saker. Vi har försökt med varje:

Gap 1: Snapback fold. Trasig efter sju försök. Diskuterat ovan.

Gap 2: Transient validering. Vi har bara DC-data från Sebas. Vi vet inte hur cellen beter sig över tid på mikrosekund-skala. Behövs för att modellera spikar korrekt.

Gap 3: Input-coupling. Vår LIF-demo visade att cellerna är **autonoma oscillatorer** i den bias-regim vi testat — de spikar 325 Hz oavsett input. Det är inte vad vi vill. Vi behöver hitta en bias-punkt där input faktiskt modulerar.

Gap 4: Temperatur. Vi har BSIM4-koefficienter för temperatur men har aldrig kopplat in dem. Vi vet inte om vår modell är giltig vid 25 grader, 50, eller 85.

Gap 5: Intrinsisk brus. Vi modellerar inte 1-över-f brus, shot-brus eller RTN — random telegraph noise. Det är just dessa fenomen som ger NS-RAM dess "neurala" karaktär enligt vissa hypoteser.

Gap 6: Aldring och drift. Floating body laddningar drifter över tid. NBTI och HCI är riktiga fenomen som påverkar långsiktig minneslagring. Vi modellerar inget av detta.

Gap 7: Wafer-variation. Vi har mätningar från **EN** cell. Riktiga chippar har miljoner celler, alla lite olika. Vi har syntetiserat en Vth0-distribution men den är inte fittad mot riktig wafer-statistik.

[del 14 — kvällens plan]

Just nu, medan denna podcast spelas upp, kör vi tre subagent-jobb parallellt på alla våra maskiner — Ikaros, Daedalus och Zgx. De försöker stänga dessa sju gap systematiskt.

Spår A: Snapback diagnostik. Forcera Vb till olika fasta värden, mät om Ids svarar med fold när vi sätter Vb tillräckligt högt. Om ja, så är det vår solver som är trasig, inte fysiken. Om nej, så är hela BSIM4-arkitekturen fel för den här regimen.

Spår B: Input-coupling tuning. Hitta bias-regimen där cellen INTE är autonom oscillator. Plus temperatur-integrering med BSIM4-koefficienter.

Spår C: Brus, aldring, wafer-variation. Implementera 1-över-f, RTN, HCI, NBTI, och svep variations-statistik.

Efter alla tre rapporterat dispatchar vi en fjärde oracle-call — vi kallar den O71 — för att brutalt kritisera resultaten och välja nästa enskilda experiment med högst värde.

[del 15 — pengar, tid, och hederlighet]

Det här projektet har pågått i månader. Vi har över 250 spårade uppgifter i vår uppgifts-databas, varav majoriteten är klara. Vi har över 400 numrerade experiment-script. Vi har skrivit 60 plus rapporter. Vi har konsulterat AI-oraklar dussintals gånger.

Och vi har faktiskt lärt oss något viktigt: **simulerad fysik utan kalibrerad transient data är värdelös för publikations-grade claims**. Vi kan säga "i den här regimen, under dessa antaganden, beter sig substratet enligt simulering så här". Vi kan INTE säga "i verklighet vid 85 grader efter en vecka kommer chippet ge X procent noggrannhet på Y benchmark".

Bryggan från "intressant simulering" till "användbar produkt" går genom tape-out — alltså, tillverka faktiska chippar och mäta dem i lab. Det är det Mario Lanza gör. Vi simulerar bara.

[del 16 — vad kommer härnäst]

Efter ikväll har vi tre möjliga vägar:

Väg 1, hederlig retraktion. Skriv en transparent rapport som säger: modell-programmet falsifierat enligt egna kriterier, applikations-programmet har fem niche-claims som behöver verifieras vid större skala och med rättvis baseline. Sluta lova produkt-grade fysik utan att ha det.

Väg 2, hardware-loop. Skippa simulator-spåret helt och hållet. Skicka tillbaka till Mario "vi har testat sju topologier, ingen reproducerar snapback i pyport, vänligen mät en cell med pulser så vi kan kalibrera transient direkt." Bygg sedan en hybridmiljö där NS-RAM-chippet är i loopen med en GPU.

Väg 3, fortsatt brute force. Försök fix 8, fix 9, fix 10 trots att O69-oraklen pakten säger retraktion. Det är inte vad vi gick med på.

Min rekommendation är väg 1 plus väg 2. Var ärliga, och gå direkt till verkliga mätningar.

[del 17 — varför detta är värdefullt ändå]

Och det här är viktigt. Även om vi inte lyckas reproducera snapback-folden, eller inte lyckas slå LSTM på Mackey-Glass, så har detta projekt lärt oss enormt mycket:

Hur man portar industri-SPICE-modeller till GPU.

Hur man dispatchar parallella jobb över ett kluster av tre maskiner.

Hur man brutalt motbevisar sina egna påståenden istället för att skydda dem.

Hur AI-oraklar kan användas för att hitta blindspots i forskning.

Hur man bygger ett autonom forsknings-loop med cron-jobb, sentinels, och pre-registrerade gates.

Och, kanske viktigast: vi har en hederlig, granulär, falsifierings-orienterad metodik som kan användas på alla framtida substrat-projekt.

[del 18 — slut och tack]

Det här är NS-RAM-projektet i grov översikt 15 maj 2026. Vi har modell-fit nere från 4 dec till runt 1 dec, men snapback-fysiken är trasig. Vi har fem applikations-vinster, alla niche. Vi har fem ärliga retraktioner. Vi har sju gap i fysisk korrekthet, som vi arbetar på just nu.

Allt detta är dokumenterat i vår 01_LOG.md, vår dagliga syntes, våra oracle-kritiska paket, och våra granulära per-experiment-summaries. Det finns plottar för varje claim och varje retraktion. Det är granskningsbart.

Det är inte en vinst-saga. Det är en hederlig forsknings-saga. Och det är, ärligt talat, det viktigaste vi har gjort.

Tack för att ni lyssnat. Sov gott — vi kör vidare i bakgrunden.

[del 19 — late-breaking upptäckt]

Vänta. Just nu, medan denna podcast genereras, kom ett kritiskt resultat in.

Vi gjorde ett diagnostiskt test som heter "body-strap". Vi forcerade body-spänningen Vb till olika fasta värden — istället för att låta solvern hitta den — och tittade på vad strömmen då gör.

Resultat: vid Vd lika med 1.5 volt, om vi forcerar Vb till 0.8 volt, då hoppar strömmen **5.5 dekader uppåt** jämfört med Vb lika med 0. Det är mer än vad mätningen visar.

Vad betyder det? Det betyder att **fysiken finns i modellen**. BSIM4 vet hur man genererar snapback-folden. Det är vår **solver** — den numeriska metoden som löser cirkelekvationerna — som inte hittar den höga Vb-basinen. Den hittar bara den låga, sub-threshold basinen.

Alla våra sju topologi-fixar var alltså inte fel om fysiken. Vi adderade element som inte behövdes, eftersom fysiken redan var där. Felet ligger i 2D Newton-solvern, som inte gör continuation eller homotopy för att gå förbi bifurkations-punkten.

Detta är massivt. KILL-SHOT-verdiktet — att retraktera modell-programmet — är upphävt. Fysik är bevarad. Vi behöver bygga en bättre solver, inte en bättre topologi.

Nästa steg är S2: implementera arc-length-continuation — en numerisk teknik där solvern följer lösningskurvan över bifurkationer istället för att hoppa över dem. Det här är standard i kraftsystemanalys och har använts för avalanche-modellering i LVTSCR ESD-litteratur.

Plus en till sak: en data-audit visade att vi har noll faktisk transient-data från Sebas. Bara DC-svep — kvasi-statiska. Vi har frågat efter pulsade Id(t)-spår nu. Det är låst hinder tills data anländer.

Och äldre påståenden som vi trodde var "fakta" — som att vi inte kunde reproducera snapback — visar sig vara en falsk falsifiering. Det visar att även vetenskaplig hederlighet kan vara fel. Man måste fortsätta sondera, inte bara godta första negativa resultatet.

[del 20 — verkligt slut]

Så där har vi det. Klockan är 14:35 på dagen. Modell-programmet är inte dött. Det är räddat. Vi har en konkret väg framåt med continuation-solver. Vi har sju spår igång parallellt över tre maskiner just nu. Vi vet vilken data vi saknar och kan be om den specifikt.

Det här är vad forskning ser ut som. Det är inte rätlinjigt. Det är inte alltid framåt. Men med rätt verktyg — pre-registrerade gates, oracle-kritik, brutala ablations-tester, och faktiskt INTE sluta i första negativa resultatet — kan man komma framåt.

Tack för att ni lyssnat. Vi kör vidare. Och nu, ärligt talat, vidare med solver-implementationen.
