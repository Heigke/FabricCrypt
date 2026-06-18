# O72 — 12h Gap-Closing Review

## State of play (12h since O70)

Solver strategies tested (4 ways):
- S2 arc-length-on-Vd: KILL-SHOT (0 folds across 33 biases)
- S2a iii-gain homotopy: found Vb=+1.18V SATURATION root (non-physical well-clamp runaway)
- S2b two-branch hot-init Vb=0.8: relaxes back to cold-Vb root
- S2c PTC pseudo-transient: in flight, but bias 1 KILL-SHOT pattern (Vb→-1.0V both inits)

Physics hypotheses tested (S3-A through S3-D):
- H1 thermal static T=150°C: NO fold appears (0.000 dec)
- H2 lateral NPN reorientation: NO fold (0.04 dec max)
- H3 vnwell sweep + clamp-off: at VG1=0.2/0.4 gives 2.76/3.33 dec fold (matches measured!) — BUT
- H4 etab override + clamp-off: at VG1=0.6 still 0.025 dec
- H5 self-heating Rth: dT=1mK (nW dissipation), no fold

**S3-D BREAKTHROUGH (15 min ago)**: At VG1=0.4/0.6, Vsint pumps to 0.19-0.23V → Vgs_M1 in subthreshold (0.17-0.41V) → Ids collapses 100× → Iii starved → no fold. Mechanism IS in pyport but inverted: at low VG1 fold appears, at high VG1 Vsint-pump kills it.

Test C beta0=10 gave VG1=0.6 fold=1.91 dec (MATCHES MEAS) but transferred from VG1=0.2 (branch flip). Suggests bistability hidden in (Vsint, Vb) Newton.

## Three brutal questions (under 200 words each)

**Q1 (GATE CROSSING)**: We have multiple "near-success" results that flip branches:
- beta0=10: VG1=0.6 fold=1.91 dec but breaks VG1=0.2  
- clamp-off+etab=20: VG1=0.2/0.4 fold matches measured but VG1=0.6 stays flat
- multiple solver basins discovered (Vb=-1, Vb=0, Vb=+1.18 saturation)
Have we crossed a real gate, or are these all artifacts of an underspecified residual system? What's the FIRST falsification step before claiming "we found the snapback mechanism"?

**Q2 (CHERRY-PICK)**: We're framing S3-B "fold at VG1=0.2/0.4 with clamp-off+etab=20" as a success. But:
- This requires use_well_diode=FALSE (turning off a physical element)
- AND etab=20 (5-10× Mario canonical)
- AND it FAILS at VG1=0.6 where we most need it
Is this cherry-picking partial wins to avoid retract? What would honest framing of S3-B+S3-D look like in a Nature paper?

**Q3 (HIGHEST-VALUE NEXT)**: Given S3-D identified Vsint-pump as the proximate cause:
(a) Bisect what pumps Vsint (BJT, M2 source-follower, pdiode?) — solver-side
(b) Add Vsint clamp (non-physical regularizer) and refit
(c) Run TLP transient (industry standard for snapback)
(d) Just ask Sebas for measured Vsint at VG1=0.6 (closes ambiguity)
Which is the SINGLE highest-information action in next 4 hours? Be specific.

Terse, brutal. We have 4 hours before retract decision.
