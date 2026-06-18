12h NS-RAM project gap-closing review.

Today's net (since previous critique O51 at 01:22):

WINS:
- z312: HDC N=16384 → 83.91% σ=0 / 84.09% σ=0.05 UCI-HAR (n=4 each, headline-lock 
  to n=10 in flight)
- T2: 143 V_d>2V samples extracted from slides 15+21 via gpt-5 vision
- T4: 3 new physics candidates from 3-way oracle (N1 traps, N2 SRH, N3 rbodymod)
- M2 N1 trap stub: 6.2 decade hysteresis lift confirmed mechanism
- Adaptive GPU thermal governor installed on ikaros+daedalus (87→54°C in 8s)
- P2 unlock: BSIM4 TAT block PRE-CALIBRATED in Sebas cards (njts/vtss/xtss/jtss)
  + distributed Rbody 1MΩ-10GΩ in zenodo .asc + quantitative snapback law
  V_peak(V_G2) = 2.73 - 0.625·V_G2

FAILS (honest):
- z309 N3: rbodymod=1 flag is no-op in our pyport (parsed not consumed)
- z313 pyport_v4: 4-variant bisection showed cell-wide 2.91 dec (-1.92 vs z304
  baseline 0.99). All flags tested (R_body, avalanche) were INERT — only the
  polarity flip caused regression. z304's "original" polarity is correct;
  oracle P1 #1 recommendation was wrong premise.

Infrastructure gap exposed: cfg.vnwell_Rs and cfg.use_lateral_collector are
parsed but not consumed by _residuals. Multi-day code work needed for proper
DBR + avalanche implementation.

Three sharp questions:

Q1 GATE CROSSING: Of today's 3 AMBITIOUS PASS items (z312 84%, M2 traps 6.2-
dec hysteresis lift, T2 V_d>2V data harvest), which is most defensible as a
SHIPPABLE v4.4 BRIEF HEADLINE? Be honest — has any been overclaimed?

Q2 CHERRY-PICK: Today we ran z313 with 4 isolated variants and reported 
"all identical 2.91 dec" — implying infrastructure issue. But z304's "0.99 
dec baseline" is also a SINGLE-run number. Is reporting "0.99 dec baseline" 
without confidence interval cherry-picking?

Q3 NEXT HIGHEST-VALUE EXPERIMENT (1-3h wall): Given the infrastructure gap 
revealed today, what is the single highest-value experiment? Options:
  A. Audit and properly wire up cfg flags in pyport _residuals (multi-day, 
     would fix root cause)
  B. Test z311 trap stub at larger τ-spectrum (10 reservoirs µs→s) and 
     report hysteresis curve shape match to slide-21 quantitatively
  C. Snapback peak law sweep — run pyport at many V_G2 values and 
     quantitatively compare V_peak vs 2.73-0.625·V_G2 (gate already has 
     2/2 PASS on available V_G1=0.3 interp range)
  D. n=10 z312 lock now in flight — what to add WHILE waiting?

Be sharp, ≤400 words per oracle. NO-CHEAT.
