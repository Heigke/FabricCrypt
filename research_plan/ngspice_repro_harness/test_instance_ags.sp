* Stage 6b probe v2 — instance-level operating-point dump.
* `@m1[X]` only works for runtime OP outputs, not for BSIM4 binned-
* parameter internals. Capture what IS exposed and compare to pyport.
.title instance-level operating-point

.include "../../data/sebas_2026_04_22/M2_130bulkNSRAM.txt"

m1 d g 0 0 NMOS  W=1u L=0.234u
vd  d  0  dc 1.0
vg  g  0  dc 0.5

.control
op
print @m1[id]
print @m1[is]
print @m1[ig]
print @m1[ib]
print @m1[gm]
print @m1[gds]
print @m1[gmb]
print @m1[vdsat]
print @m1[vth]
print @m1[cgg]
print @m1[cgs]
print @m1[cgd]
print @m1[cdb]
print @m1[csb]
.endc
.end
