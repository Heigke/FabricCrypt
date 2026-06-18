.title M2 binning-term ground-truth

.include "../../data/sebas_2026_04_22/M2_130bulkNSRAM.txt"

m1 d g 0 0 NMOS  W=1u L=0.13u
vd  d  0  dc 1
vg  g  0  dc 0.5

.control
op
showmod m1
.endc
.end
