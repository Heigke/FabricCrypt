* Ground-truth: load Sebas's actual M2 card and dump parameters
* Question: do ngspice's actual loaded values match the card's
* textual values, or do some land at BSIM4 defaults?

.title M2 actual-card parameter ground-truth

.include "../../data/sebas_2026_04_22/M2_130bulkNSRAM.txt"

* Trivial circuit to instantiate the model
m1 d g 0 0 NMOS  W=1u L=1.8u
vd  d  0  dc 1
vg  g  0  dc 0.5

.op
.print op v(d) v(g)
.end
