* B.1 — 2T cell ngspice op-point at PRODUCTION BJT params (Bf=9000, Va=0.55, Is=1e-9)
* Goal: certify which root (lumped's low-Id or quasi-2D's high-Id) is silicon-correct.

.title 2T cell B.1 ngspice cross-check at production params

.include "../../data/sebas_2026_04_22/M1_130DNWFB.txt"
.include "../../data/sebas_2026_04_22/M2_130bulkNSRAM.txt"

* Production BJT (overrides Sebas card with z91g production env values)
.model parasiticBJT NPN(is=1e-9 va=0.55 bf=9000 br=100 nc=2 ikr=100m rc=0.1
+ vje=0.7 re=0.1 cjc=1e-15 fc=0.5 cje=0.7e-15 ne=1.5 ise=0 tr=20e-12 tf=25e-12
+ itf=0.03 vtf=7 xtf=2)

.model Dwell_mod D(IS=3.4089e-19 N=1.017 RS=0)

.param vd_val=1.0
.param vg1_val=0.6
.param vg2_val=0.30

Vdd     vd       0       DC {vd_val}
Vg1     vg1      0       DC {vg1_val}
Vg2     vg2      0       DC {vg2_val}
Vnwell  vnwell   0       DC 2.0

* Devices (mirror NSRAMCell2TConfig with m2_body_gnd=True)
M1  vd vg1 vsint vb NMOSdnwfb L=0.13u W=1u
M2  vsint vg2 0 0 NMOS L=0.234u W=1u
Q1  vsint vb 0 parasiticBJT area=1u
Rwell  vnwell vnwell_x  10G
Dwell  vb     vnwell_x  Dwell_mod

.options gmin=1e-15 abstol=1e-12 reltol=1e-3

.control
echo "VG1,VG2,Vd,Vsint,Vb,Id,Iemit_q1" > b1_out.csv

foreach vg1_v 0.2 0.4 0.6
  foreach vg2_v 0.0 0.15 0.3
    alter vg1 dc=$vg1_v
    alter vg2 dc=$vg2_v
    alter vdd dc=1.0
    op
    let vsi = v(vsint)
    let vbi = v(vb)
    let idd = -i(vdd)
    let ie  = -i(q1.qpnp.cje)
    echo "$vg1_v,$vg2_v,1.0,$&vsi,$&vbi,$&idd,0" >> b1_out.csv
  end
end
quit
.endc

.end
