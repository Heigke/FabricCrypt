# NS-RAM Campaign — Lessons Learned (LIVE doc)

Last updated: 2026-05-16 19:30

## Currently running (background)

| ID | Spår | Status | Förväntat |
|---|---|---|---|
| z438 knee-calib | grid 7-8/16, best 0.916 | running | -0.1 dec |
| z439 BDF2 | implicit Euler ej hjälpte | running | -0.05 dec |
| z441 VG-gate (S30) | restartad, grid 0/12 | running | -0.2 dec |
| z442 G9-fix | NFACTOR→M2 bugfix | running | -0.3 dec (VG1=0.2) |
| **z443 VBIC** | **DONE 1.311 dec** | -0.31 confirmed | ✓ |
| z444 BESD PNPN | dispatched | researching | -0.5 dec? |
| z445 Z²-FET | bailed (paywalled) | killed | n/a |
| z446 VBIC+PT (S31) | variant A done, B running | running | -0.5 dec? |
| z447 transient pipeline (S32) | dispatched | building | dynamics validation |

## KILL_SHOTS (saker som inte funkar)

| Spår | Varför fail |
|---|---|
| z431 BSIM4 GIDL | Redan PÅ, GIDL för svag (1e-16 vs mätt 1e-8) |
| z433 2D PWL | Strukturellt problem, ej parameter-lookup |
| z434 lateral PNP shunt | V_B kraschar till clamp |
| z435 λ-homotopy | Landar fel branch |
| z437 snapback subcircuit | NPN injicerar för mycket vid låg V_BE |
| z440 M2 body shunt parallell | Försämrar konvergens |

## BREAKTHROUGHS (saker som funkar)

| Fix | Δ cell-RMSE | Var i kod |
|---|---|---|
| V_SINT_PIN (hard pin V_Sint=0) | 3.9→1.6 dec | nsram_cell_2T.py |
| Pseudo-transient backward sweep | 1.6→1.0 dec | z432 script |
| VBIC + Kloosterman avalanche | -0.31 dec, VG1=0.2: -1.71! | vbic.py |
| G9 NFACTOR→M2 (canonical bug) | testas i z442 | poly_params.py:139 |

## Kvarvarande gap-källor

### Statisk fysik (DC) — 95% klar
- ✓ BSIM4 v4.8.3 (b4ld.c ekvivalent)
- ✓ Gummel-Poon + VBIC alternative
- ✓ Mario Ipos PWL injection
- ✓ V_SINT_PIN (substrat-tap)
- ⚠ Knee-position 0.2V skift (S26 grid+S30 gate försöker fixa)
- ⚠ VG1=0.2 high-VG2 false snapback (S30 gate)
- ❌ LDE/stress §13 (subprocent, ignorerar)

### Dynamik (transient) — 50% klar
- ✓ Pseudo-transient solver-trick (z432)
- ❌ Verklig C_B = 1 fF (S32 fixar)
- ❌ BJT τ_F/τ_R charge-storage (S32)
- ❌ Junction caps Cje/Cjc (S32)
- ❌ Self-heating (S32 optional)
- ❌ Sebas's transient mätdata (A.12 pending)

## Lessons om strategi

- **Oracle 4-panel ger ofta fel diagnos** (t.ex. O75 sa "M2 channel shunt" men z440 motbevisade)
- **Canonical-data-audit hittade bug oracle missade** (G9 NFACTOR)
- **Backward V_D-sweep löste bistability** (z432) — matchade Sebas's mät-protokoll
- **Pure parameter-fits hit a wall** ~1.0 dec — strukturell topologi-fix krävs (VBIC, BESD)
- **Mario använder samma stack som vi** — gap är strukturell, ej replikation
- **Pseudo-transient = solver-trick, inte fysik** — riktig transient kräver verklig C_B

## Nästa möjliga vinster (prioritetsordning)

1. **z446 VBIC+PT combo** — orthogonala mekanismer, väntad 0.7-0.8 dec
2. **z444 BESD PNPN** — enhetlig topologi, förväntat bäst
3. **z442 G9-fix** — NFACTOR-bugfix, bör fixa VG1=0.2 sub-thresh
4. **z447 real transient** — dynamics, för spike-baserad neuromorphism
5. **z441 V_G1-V_G2 gate** — surgical S27 boundary test
6. **z438 knee-calib** — marginell parameter-tuning

## Status mot mål

- DC publication target: < 0.5 dec cell-wide → vi är 1.0 dec borta från där
- Network-sim usability target: < 1.5 dec → **redan klar**
- Transient validation: blocked på Sebas's data (#A.12)
