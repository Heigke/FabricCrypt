* F4 — 2T cell ngspice op-point cross-check vs pyport
* Full Sebas 2T cell: M1 (NMOSdnwfb, L=0.13u) + M2 (NMOS, L=0.234u) +
* parasitic NPN Q1 + well-body diode + body-pdiode.
* 12-bias spot grid (3 VG1 x 4 VG2) at 4 Vd values = 48 op-points.
*
* Topology (matches data/sebas_2026_04_22/2tnsram_simple.asc):
*   M1: D=Vd, G=VG1, S=Vsint, B=Vb (floating body)
*   M2: D=Vsint, G=VG2, S=GND, B=GND
*   Q1 (NPN parasitic): C=Vsint, B=Vb, E=GND, area=1u
*   Dwell (vnwell -> Vb): anode=Vb, cathode=vnwell @ +2 V
*   Dpdi (body pdiode): anode=Vb, cathode=Vsint
*
* Param values mirror NSRAMCell2TConfig defaults:
*   vnwell_Js=3.4089e-7 A/m^2, vnwell_n=1.017, vnwell_area=1e-12 m^2
*     -> Is_well = Js*A = 3.4089e-19 A
*   body_pdiode_Js=1e-6 A/m^2, body_pdiode_n=1.0535, area=22e-12 m^2
*     -> Is_pdi = 2.2e-17 A
*   (vnwell_Rs=1e10 ohm modelled as explicit Rwell)
*
* Parasitic NPN card (Sebas) included verbatim.

.title 2T cell op-point grid (F4 cross-check)

.include "../../data/sebas_2026_04_22/M1_130DNWFB.txt"
.include "../../data/sebas_2026_04_22/M2_130bulkNSRAM.txt"
.include "../../data/sebas_2026_04_22/parasiticBJT.txt"

* Well-body diode model: matches cfg.vnwell_Js/n with area=cfg.vnwell_area
.model Dwell_mod D(IS=3.4089e-19 N=1.017 RS=0)
* Body-pdiode model: matches cfg.body_pdiode_Js/n with area=cfg.body_pdiode_area
* Sebas's pdiode card uses level-1 LTSpice diode; we use the same Js*A path.
.model Dpdi_mod D(IS=2.2e-17 N=1.0535 RS=0)

* Voltage sources (parametric — alter'd in .control)
.param vd_val=1.0
.param vg1_val=0.6
.param vg2_val=0.30
.param vnwell_val=2.0

Vdd     vd       0       DC {vd_val}
Vg1     vg1      0       DC {vg1_val}
Vg2     vg2      0       DC {vg2_val}
Vnwell  vnwell   0       DC {vnwell_val}

* Devices
* M1: drain=vd gate=vg1 source=vsint body=vb (floating)
M1  vd vg1 vsint vb NMOSdnwfb L=0.13u W=1u

* M2: drain=vsint gate=vg2 source=0 body=0
M2  vsint vg2 0 0 NMOS L=0.234u W=1u

* Parasitic NPN: collector=vsint base=vb emitter=0
Q1  vsint vb 0 parasiticBJT area=1u

* Well-body diode + series Rs=1e10 (matches cfg.vnwell_Rs).
* anode = vb, cathode = vnwell node.
Rwell  vnwell vnwell_x  10G
Dwell  vb     vnwell_x  Dwell_mod

* Body-pdiode: anode=vb, cathode=vsint
Dpdi   vb     vsint     Dpdi_mod

* Convergence aids — floating body cell is hard.
.options gmin=1e-12 reltol=1e-4 abstol=1e-14 vntol=1e-7 itl1=300 itl2=200

.control
set wr_singlescale
set wr_vecnames
set filetype=ascii
set width=400

* Output table header — use ###ROW### marker so the parser can grep rows
echo "###HDR### VG1 VG2 Vd Vsint Vb Id"

* Sweep grid: VG1 in {0.2,0.4,0.6} x VG2 in {-0.10, 0.00, 0.15, 0.30}
* For each (VG1, VG2): step Vd in {0.5, 1.0, 1.5, 2.0}.
foreach vg1v 0.2 0.4 0.6
  alter Vg1 dc=$vg1v
  foreach vg2v -0.10 0.00 0.15 0.30
    alter Vg2 dc=$vg2v
    foreach vdv 0.5 1.0 1.5 2.0
      alter Vdd dc=$vdv
      op
      let vsint_v = v(vsint)
      let vb_v    = v(vb)
      let id_v    = -i(Vdd)
      echo "###ROW### vg1=$vg1v vg2=$vg2v vd=$vdv"
      print vsint_v vb_v id_v
    end
  end
end

quit
.endc

.end
