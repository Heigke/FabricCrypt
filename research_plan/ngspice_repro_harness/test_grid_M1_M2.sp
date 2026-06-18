* F2 grid validation: 90 bias pts × 2 devices (M1 dnwfb L=0.13u, M2 bulk L=0.234u).
* Loops .op over Vgs ∈ {0.2,0.4,0.6,0.8,1.0}, Vds ∈ {0.05,0.2,0.5,1.0,1.5,2.0},
* Vbs ∈ {-0.3, 0.0, +0.3}. For each point, prints Id, gm, gds, gmb, vth,
* vdsat, cgg, cgs, cgd, cdb, csb to grid_out.csv.
*
* Approach choice: caps are only available at the operating point (.op),
* not from .dc sweep print streams in our ngspice version.  Use ngspice's
* internal foreach to drive 90 .op runs per device in one process.
.title F2 90x2 grid

.include "../../data/sebas_2026_04_22/M2_130bulkNSRAM.txt"
.include "../../data/sebas_2026_04_22/M1_130DNWFB.txt"

* M1 device — 130 nm DNWFB card, L=0.13 µm.  Source=0, Body=b1 so Vbs=V(b1).
m1_M1 d1 g1 0 b1 NMOSdnwfb W=1u L=0.13u
* M2 device — 130 nm bulk NSRAM card, L=0.234 µm.  Source=0, Body=b2.
m1_M2 d2 g2 0 b2 NMOS      W=1u L=0.234u

vd1 d1 0 dc 1.0
vg1 g1 0 dc 0.5
vd2 d2 0 dc 1.0
vg2 g2 0 dc 0.5
vb1 b1 0 dc 0
vb2 b2 0 dc 0

.control
set wr_singlescale
set wr_vecnames
* CSV header
echo "device,vgs,vds,vbs,id,gm,gds,gmb,vth,vdsat,cgg,cgs,cgd,cdb,csb" > grid_out.csv

foreach vgs_v 0.2 0.4 0.6 0.8 1.0
  foreach vds_v 0.05 0.2 0.5 1.0 1.5 2.0
    foreach vbs_v -0.3 0.0 0.3
      alter vg1 dc=$vgs_v
      alter vd1 dc=$vds_v
      alter vg2 dc=$vgs_v
      alter vd2 dc=$vds_v
      * Body bias: re-attach M1/M2 body to a node driven by vb*.
      * Easier — recreate device with body=b{1,2}.  Done at deck level above
      * by pinning body to 0 — for Vbs ≠ 0 we instead shift Vgs/Vds frame:
      * since terminals are S=B=0, Vbs=0 always.  To get true Vbs we lift
      * source above body OR drop body below source.  We drop body:
      alter vb1 dc=$vbs_v
      alter vb2 dc=$vbs_v
      * Note: with body = vbs_v and source = 0, Vbs = vbs_v - 0 = vbs_v
      * but Vds, Vgs are still drain-to-source so unaffected.  The model
      * sees Vbs = body - source = vbs_v.
      op
      let id1   = @m1_m1[id]
      let gm1   = @m1_m1[gm]
      let gds1  = @m1_m1[gds]
      let gmb1  = @m1_m1[gmbs]
      let vth1  = @m1_m1[vth]
      let vdst1 = @m1_m1[vdsat]
      let cgg1  = @m1_m1[cgg]
      let cgs1  = @m1_m1[cgs]
      let cgd1  = @m1_m1[cgd]
      let cdb1  = @m1_m1[cdb]
      let csb1  = @m1_m1[csb]
      let id2   = @m1_m2[id]
      let gm2   = @m1_m2[gm]
      let gds2  = @m1_m2[gds]
      let gmb2  = @m1_m2[gmbs]
      let vth2  = @m1_m2[vth]
      let vdst2 = @m1_m2[vdsat]
      let cgg2  = @m1_m2[cgg]
      let cgs2  = @m1_m2[cgs]
      let cgd2  = @m1_m2[cgd]
      let cdb2  = @m1_m2[cdb]
      let csb2  = @m1_m2[csb]
      echo "M1,$vgs_v,$vds_v,$vbs_v,$&id1,$&gm1,$&gds1,$&gmb1,$&vth1,$&vdst1,$&cgg1,$&cgs1,$&cgd1,$&cdb1,$&csb1" >> grid_out.csv
      echo "M2,$vgs_v,$vds_v,$vbs_v,$&id2,$&gm2,$&gds2,$&gmb2,$&vth2,$&vdst2,$&cgg2,$&cgs2,$&cgd2,$&cdb2,$&csb2" >> grid_out.csv
      reset
    end
  end
end

quit
.endc
.end
