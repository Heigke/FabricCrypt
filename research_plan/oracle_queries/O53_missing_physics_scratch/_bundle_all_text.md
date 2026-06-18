# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: M1_130DNWFB.txt (9903 chars) ===
```
* Predictive Technology Model Beta Version
* 130nm NMOS SPICE Parametersv (normal one)
*  http://ptm.asu.edu/latest.html\
*+Lint = 2.5e-08 Tox = 3.3e-09
*+Vth0 = 0.395 Rdsw = 200

.model NMOSdnwfb NMOS

+Level = 14

+version = 4.5                 binunit = 2                   
+paramchk = 1                  mobmod = 0                    capmod = 2                    
+rdsmod = 0                    igcmod = 0                    igbmod = 0                    
+rbodymod = 0                  trnqsmod = 0                  acnqsmod = 0                  
+fnoimod = 1                   diomod = 1                    tempmod = 0                   
+permod = 1                    geomod = 0                    rgeomod = 0                   
+rgatemod = 0                  
+epsrox = 3.9                  toxe = toxn                   toxp = toxn                   
+toxm = toxn                   dtox = 0                      xj = 1.5e-7                   
+ndep = 1.7e17                 ngate = 1e23                  nsd = 1e20                    
+rsh = 1                       rshg = 0.1                    
+wint = wintn                  wl = 0                        wln = 1                       
+ww = -6.8e-15                 wwn = 1                       wwl = 0                       
+lint = lintn                  ll = 0                        lln = 1                       
+lw = 0                        lwn = 1                       lwl = 0                       
+llc = 0                       lwc = 0                       lwlc = 0                      
+wlc = 0                       wwc = 0                       wwlc = 0                      
+dwg = 0                       dwb = 0                       xl = 0                        
+xw = 0                        
+dmcg = 0                      dmdg = 0                      dmcgt = 0                     
+xgw = 0                       xgl = 0                       ngcon = 1                     
+vth0 = vth0n                  wvth0 = -1.6569e-8            pvth0 = pvth0n           
+phin = 0.05                   k1 = 0.53825                  k2 = -0.070435                
+k3 = k3n                      k3b = 6.37                    w0 = 2.5e-6                   
+lpe0 = lpe0n                  lpeb = -1.6512e-8             vbm = -3                      
+dvtp0 = 0                     dvtp1 = 0                     dvt0 = 1.9758                 
+dvt1 = 0.46322                dvt2 = -0.035558              dvt0w = -0.037131             
+dvt1w = 6.2805e5              dvt2w = -0.32774              vfbsdoff = 0                  
+u0 = 0.048317                 pu0 = -1.2e-16                ua = 5.0195e-11               
+ub = 1.7249e-18               uc = 1.1834e-10               ud = 1e14                     
+up = 0                        lp = 1e-8                     eu = 1.67                     
+vsat = vsatn                  pvsat = 1.03e-009             a0 = 1                        
+ags = 0.34914                 pags = 3e-013                 b0 = 6e-008                   b1 = 0                        
+keta = 0                      pketa = -3.4e-015             a1 = 0.9                      a2 = 0.95                     
+rdsw = 100-140*1e6*1u/int(1u/0.34u)     rdswmin = 35         rdw = 100             
+rdwmin = 0                    rsw = 100                     rswmin = 0                    
+prwb = -0.24                  prwg = 0                      wr = 1                        
+voff = -0.1368                wvoff = -5.6e-9               voffl = -5.5973e-9            
+minv = 0                      nfactor = 1.58                eta0 = 0.19998                
+etab = 1.8              dsub = 0.6412                 cit = 0                       
+cdsc = 2.4e-4                 cdscb = 0                     cdscd = 0                     
+pclm = 0.34476                pdiblc1 = 3.3832              pdiblc2 = 2e-3                
+pdiblcb = 0                   drout = 1.3536                pscbe1 = 5.331e8              
+pscbe2 = 1e-5                 pvag = 0.22                   delta = 0.01                  
+fprout = 0                    pdits = 0                     pditsl = 0                    
+pditsd = 0                    lambda = 0                    vtl = 2e5                     
+lc = 5e-9                     xn = 3                        alpha0 = 7.83756e-5           
+lalpha0 = -9.843026e-12       alpha1 = 0                    beta0 = 19                    
+lbeta0 = -9.5e-7              
+aigbacc = 0.43                bigbacc = 0.054               cigbacc = 0.075               
+nigbacc = 1                   aigbinv = 0.35                bigbinv = 0.03                
+cigbinv = 6e-3                eigbinv = 1.1                 nigbinv = 3                   
+aigc = 0.43                   bigc = 0.054                  cigc = 0.075                  
+aigsd = 0.43                  bigsd = 0.054                 cigsd = 0.075                 
+dlcig = 0                     nigc = 1                      poxedge = 1                   
+pigcd = 1                     ntox = 1                      toxref = toxn                 
+agidl = 1.99e-8               bgidl = 1.624e9               cgidl = 6.3                   
+egidl = 0.91                  
+noia = 3.3216e+41             noib = 1.0773239e+25          noic = -1.0624e+08                 
+em = 4.1e7                    ef = 0.96806                  lintnoi = 0                   
+xpart = 0                     cgso = rcgon*3.65e-10       cgdo = rcgon*3.65e-10               
+cgbo = 0                      ckappas = 0.6                 ckappad = 0.6                 
+cf = 0                        clc = 1e-7                    cle = 0.6                     
+dlc = 1.3737e-8               dwc = 0                       vfbcv = -1                    
+noff = 1                      lnoff = 2.2e-7                voffcv = -0.04464             
+lvoffcv = -2.8e-8             acde = 0.5535                 moin = 15                     
+cgsl = rcgon*2.98e-11         cgdl = rcgon*2.98e-11               
+ijthsrev = 0.1                ijthsfwd = 0.1                xjbvs = 1                     
+xjbvd = 1                     bvs = 10                      jss = 3.4089e-007                   
+jsws = 2.368e-013             jswgs = 0                     jtss = 0                      
+jtsd = 0                      jtssws = 0                    jtsswd = 0                    
+jtsswgs = 0                   jtsswgd = 0                   njts = 20                     
+njtssw = 20                   njtsswg = 20                  xtss = 0.02                   
+xtsd = 0.02                   xtssws = 0.02                 xtsswd = 0.02                 
+xtsswgs = 0.02                xtsswgd = 0.02                vtss = 10                     
+vtsd = 10                     vtssws = 10                   vtsswd = 10                   
+vtsswgs = 10                  vtsswgd = 10                  tnjts = 0                     
+tnjtssw = 0                   tnjtsswg = 0                  cjs = rcjn*0.0016995                
+mjs = 0.51829                 mjsws = 0.57223                         
+cjsws = rcjswn*2.9299e-011    cjswgs = rcjswgn*2.677e-010                
+mjswgs = 0.50288              pbs = 0.74883                 pbsws = 0.6836                     
+pbswgs = 0.70856                    
+xrcrg1 = 12                   xrcrg2 = 1                    rbpb = 50                     
+rbpd = 50                     rbps = 50                     rbdb = 50                     
+rbsb = 50                     rbps0 = 50                    rbpsl = 0                     
+rbpsw = 0                     rbpsnf = 0                    rbpd0 = 50                    
+rbpdl = 0                     rbpdw = 0                     rbpdnf = 0                    
+rbpbx0 = 100                  rbpbxl = 0                    rbpbxw = 0                    
+rbpbxnf = 0                   rbpby0 = 100                  rbpbyl = 0                    
+rbpbyw = 0                    rbpbynf = 0                   rbsbx0 = 100                  
+rbsby0 = 100                  rbdbx0 = 100                  rbdby0 = 100                  
+rbsdbxl = 0                   rbsdbxw = 0                   rbsdbxnf = 0                  
+rbsdbyl = 0                   gbmin = 1e-12                 
+tnom = 25                     ute = -1.785                  wute = 8e-8                   
+kt1 = -0.273                  kt1l = 3e-9                   kt2 = -0.034                  
+ua1 = 7.4e-10                 ub1 = -1e-18                  uc1 = -5.6e-11                
+lua1 = -8.88e-17
+ud1 = 0                       at = 4.6035e4                 prt = 0                    
+njs = 1.017                   xtis = 6.5                   tpb = 0                       
+tpbsw = 0                     tpbswg = 0                    tcj = 0                       
+tcjsw = 0                     tcjswg = 0                    tvoff = 0                     
+tvfbsdoff = 0                 
+saref = 1.04e-6               sbref = 1.04e-6               wlod = 0                      
+ku0 = -2.7e-8                 kvsat = 0.2                   kvth0 = 9.8e-9                
+tku0 = 0                      llodku0 = 0                   wlodku0 = 0                   
+llodvth = 0                   wlodvth = 0                   lku0 = 0                      
+wku0 = 0                      pku0 = 0                      lkvth0 = 0                    
+wkvth0 = 0                    pkvth0 = 0                    stk2 = 0                      
+lodk2 = 1                     steta0 = 0                    lodeta0 = 1                   
+web = 0                       wec = 0                       kvth0we = 0                   
+k2we = 0                      ku0we = 0                     scref = 1e-6         
```


=== FILE: M2_130bulkNSRAM.txt (10507 chars) ===
```
.param toxn    = 4e-009               toxp    = 4e-009
+lintn   = 1.219e-8             lintp   = -1.079e-8
+vth0n   = 0.54153              vth0p   = -1.106133
+lpe0n   = 1.2439e-007          lpe0p   = -7.833656e-8
+k3n     = 65.28                k3p     = -7.18419
+pvth0n  = -1.45e-015           pvth0p  = 5.543149e-16
+vsatn   = 102230               vsatp   = 8.07584e4
+wintn   = 4.7689e-008          wintp   = 4.268414e-9
+rcjn    = 1                    rcjp    = 1
+rcjswn  = 1                    rcjswp  = 1
+rcjswgn = 1                    rcjswgp = 1
+rcgon   = 1                    rcgop   = 1

* Predictive Technology Model Beta Version
* 130nm NMOS SPICE Parametersv (normal one)
*  http://ptm.asu.edu/latest.html\
*+Lint = 2.5e-08 Tox = 3.3e-09
*+Vth0 = 0.395 Rdsw = 200

.model NMOS NMOS

+Level = 14

+version = 4.5                 binunit = 2                   
+paramchk = 1                  mobmod = 0                    capmod = 2                    
+rdsmod = 0                    igcmod = 0                    igbmod = 0                    
+rbodymod = 0                  trnqsmod = 0                  acnqsmod = 0                  
+fnoimod = 1                   diomod = 1                    tempmod = 0                   
+permod = 1                    geomod = 0                    rgeomod = 0                   
+rgatemod = 0                  
+epsrox = 3.9                  toxe = toxn                   toxp = toxn                   
+toxm = toxn                   dtox = 0                      xj = 1.5e-7                   
+ndep = 1.7e17                 ngate = 1e23                  nsd = 1e20                    
+rsh = 1                       rshg = 0.1                    
+wint = wintn                  wl = 0                        wln = 1                       
+ww = -6.8e-15                 wwn = 1                       wwl = 0                       
+lint = lintn                  ll = 0                        lln = 1                       
+lw = 0                        lwn = 1                       lwl = 0                       
+llc = 0                       lwc = 0                       lwlc = 0                      
+wlc = 0                       wwc = 0                       wwlc = 0                      
+dwg = 0                       dwb = 0                       xl = 0                        
+xw = 0                        
+dmcg = 0                      dmdg = 0                      dmcgt = 0                     
+xgw = 0                       xgl = 0                       ngcon = 1                     
+vth0 = vth0n                  wvth0 = -1.6569e-8            pvth0 = pvth0n           
+phin = 0.05                   k1 = 0.63825                  k2 = -0.070435                
+k3 = k3n                      k3b = 6.37                    w0 = 2.5e-6                   
+lpe0 = lpe0n                  lpeb = -1.6512e-8             vbm = -3                      
+dvtp0 = 0                     dvtp1 = 0                     dvt0 = 1.9758                 
+dvt1 = 0.46322                dvt2 = -0.035558              dvt0w = -0.037131             
+dvt1w = 6.2805e5              dvt2w = -0.32774              vfbsdoff = 0                  
+u0 = 0.048317                 pu0 = -1.2e-16                ua = 5.0195e-11               
+ub = 1.7249e-18               uc = 1.1834e-10               ud = 1e14                     
+up = 0                        lp = 1e-8                     eu = 1.67                     
+vsat = vsatn                  pvsat = 1.03e-009             a0 = 1                        
+ags = 0.34914                 pags = 3e-013                 b0 = 6e-008                   b1 = 0                        
+keta = 0                      pketa = -3.4e-015             a1 = 0.9                      a2 = 0.95                     
+rdsw = 100-140*1e6*1u/int(1u/0.34u)     rdswmin = 35         rdw = 100             
+rdwmin = 0                    rsw = 100                     rswmin = 0                    
+prwb = -0.24                  prwg = 0                      wr = 1                        
+voff = -0.1368                wvoff = -5.6e-9               voffl = -5.5973e-9            
+minv = 0                      nfactor = 1.58                eta0 = 0.19998                
+etab = -0.086777              dsub = 0.6412                 cit = 0                       
+cdsc = 2.4e-4                 cdscb = 0                     cdscd = 0                     
+pclm = 0.34476                pdiblc1 = 3.3832              pdiblc2 = 2e-3                
+pdiblcb = 0                   drout = 1.3536                pscbe1 = 5.331e8              
+pscbe2 = 1e-5                 pvag = 0.22                   delta = 0.01                  
+fprout = 0                    pdits = 0                     pditsl = 0                    
+pditsd = 0                    lambda = 0                    vtl = 2e5                     
+lc = 5e-9                     xn = 3                        alpha0 = 7.83756e-5           
+lalpha0 = -9.843026e-12       alpha1 = 0                    beta0 = 18                    
+lbeta0 = -9.5e-7              
+aigbacc = 0.43                bigbacc = 0.054               cigbacc = 0.075               
+nigbacc = 1                   aigbinv = 0.35                bigbinv = 0.03                
+cigbinv = 6e-3                eigbinv = 1.1                 nigbinv = 3                   
+aigc = 0.43                   bigc = 0.054                  cigc = 0.075                  
+aigsd = 0.43                  bigsd = 0.054                 cigsd = 0.075                 
+dlcig = 0                     nigc = 1                      poxedge = 1                   
+pigcd = 1                     ntox = 1                      toxref = toxn                 
+agidl = 1.99e-8               bgidl = 1.624e9               cgidl = 6.3                   
+egidl = 0.91                  
+noia = 3.3216e+41             noib = 1.0773239e+25          noic = -1.0624e+08                 
+em = 4.1e7                    ef = 0.96806                  lintnoi = 0                   
+xpart = 0                     cgso = rcgon*3.65e-10       cgdo = rcgon*3.65e-10               
+cgbo = 0                      ckappas = 0.6                 ckappad = 0.6                 
+cf = 0                        clc = 1e-7                    cle = 0.6                     
+dlc = 1.3737e-8               dwc = 0                       vfbcv = -1                    
+noff = 1                      lnoff = 2.2e-7                voffcv = -0.04464             
+lvoffcv = -2.8e-8             acde = 0.5535                 moin = 15                     
+cgsl = rcgon*2.98e-11         cgdl = rcgon*2.98e-11               
+ijthsrev = 0.1                ijthsfwd = 0.1                xjbvs = 1                     
+xjbvd = 1                     bvs = 10                      jss = 3.4089e-007                   
+jsws = 2.368e-013             jswgs = 0                     jtss = 0                      
+jtsd = 0                      jtssws = 0                    jtsswd = 0                    
+jtsswgs = 0                   jtsswgd = 0                   njts = 20                     
+njtssw = 20                   njtsswg = 20                  xtss = 0.02                   
+xtsd = 0.02                   xtssws = 0.02                 xtsswd = 0.02                 
+xtsswgs = 0.02                xtsswgd = 0.02                vtss = 10                     
+vtsd = 10                     vtssws = 10                   vtsswd = 10                   
+vtsswgs = 10                  vtsswgd = 10                  tnjts = 0                     
+tnjtssw = 0                   tnjtsswg = 0                  cjs = rcjn*0.0016995                
+mjs = 0.51829                 mjsws = 0.57223                         
+cjsws = rcjswn*2.9299e-011    cjswgs = rcjswgn*2.677e-010                
+mjswgs = 0.50288              pbs = 0.74883                 pbsws = 0.6836                     
+pbswgs = 0.70856                    
+xrcrg1 = 12                   xrcrg2 = 1                    rbpb = 50                     
+rbpd = 50                     rbps = 50                     rbdb = 50                     
+rbsb = 50                     rbps0 = 50                    rbpsl = 0                     
+rbpsw = 0                     rbpsnf = 0                    rbpd0 = 50                    
+rbpdl = 0                     rbpdw = 0                     rbpdnf = 0                    
+rbpbx0 = 100                  rbpbxl = 0                    rbpbxw = 0                    
+rbpbxnf = 0                   rbpby0 = 100                  rbpbyl = 0                    
+rbpbyw = 0                    rbpbynf = 0                   rbsbx0 = 100                  
+rbsby0 = 100                  rbdbx0 = 100                  rbdby0 = 100                  
+rbsdbxl = 0                   rbsdbxw = 0                   rbsdbxnf = 0                  
+rbsdbyl = 0                   gbmin = 1e-12                 
+tnom = 25                     ute = -1.785                  wute = 8e-8                   
+kt1 = -0.273                  kt1l = 3e-9                   kt2 = -0.034                  
+ua1 = 7.4e-10                 ub1 = -1e-18                  uc1 = -5.6e-11                
+lua1 = -8.88e-17
+ud1 = 0                       at = 4.6035e4                 prt = 0                    
+njs = 1.017                   xtis = 6.5                   tpb = 0                       
+tpbsw = 0                     tpbswg = 0                    tcj = 0                       
+tcjsw = 0                     tcjswg = 0                    tvoff = 0                     
+tvfbsdoff = 0                 
+saref = 1.04e-6               sbref = 1.04e-6               wlod = 0                      
+ku0 = -2.7e-8                 kvsat = 0.2                   kvth0 = 9.8e-9                
+tku0 = 0                      llodku0 = 0                   wlodku0 = 0                   
+llodvth = 0                   wlodvth = 0                   lku0 = 0                      
+wku0 = 0                      pku0 = 0                      lkvth0 = 0                    
+wkvth0 = 0                    pkvth0 = 0                    stk2 = 0                      
+lodk2 = 1                     steta0 = 0                    lodeta0 = 1                   
+web = 0                       wec = 0                       kvth0we = 0                   
+k2we = 0                      ku0we = 0                     scref = 1e-6         
```


=== FILE: SA3_image_deep_extract.md (8776 chars) ===
```
# SA3 — Deep Image Extract from Sebas + Mario Slides (O48)

Date: 2026-05-12
Source: `research_plan/oracle_queries/O48_images_deep/openai_response.md` (gpt-5, 248 s, 22 images)
Status: Locked gate PASSED (>=3 topology insights new vs current pyport)

---

## 1. Inventory

22 images processed:
- 21 Mario+Sebas slides (`01..21_*.png/jpeg` from O44) — re-examined for SCHEMATIC content
- `22_sebas_2026_05_02_image-2.png` — NEW image from Sebas, 2 May 2026

No additional images were found anywhere in `data/sebas_2026_*` or `data/nsram_zenodo/`.
`nsram_zenodo` contains only `.cir`, `.txt`, `.csv` simulation files (no figures from the paper).

---

## 2. Per-Image Structural Summary

(Full per-image A–F breakdown lives in the oracle response; the highlights are
indexed here.)

| # | Type            | Topology / Content (one line)                                                                  |
|---|-----------------|------------------------------------------------------------------------------------------------|
| 01 | Param plot     | NFACTOR(M2) vs VG2, 3 VG1 branches (red 0.2, blue 0.4, black 0.6 V)                            |
| 02 | Param plot     | K1(M1) vs VG1, "for all VG2"                                                                   |
| 03 | Param plot     | ETAB(M1) vs VG2, 3 VG1 branches                                                                |
| 04 | Param plot     | BETA0(M1) vs VG2, 3 VG1 branches                                                               |
| 05 | Schematic + IV | 2T cell with explicit **VB–VG2 MOS capacitor**; ID-VD families at VG2=1.4 V and 0.1 V          |
| 06 | Schematic + IV | 2T cell inset; 3 panels of ID-VD vs VG2 sweep, symbols=meas, lines=sim                         |
| 07 | Param page     | 2×2 of (BETA0, ETAB, K1, NFACTOR) — consolidated extraction source                             |
| 08 | Transient      | VD ramp + IV cloud (dynamic trajectory); reference points at start/pre-knee/post-knee          |
| 09 | Meas vs SPICE  | 3 corners: (VG1, VG2) = (0.6, 0.35), (0.4, 0.25), (0.2, 0.0) — thick=meas, thin=sim            |
| 10 | Soma cell      | Cint=102 fF, VD→Vmem mapping, VB=Vspike output, ~0.2 pJ/spike, 111 µm²                         |
| 11 | Soma macro     | Starved-inverter front end, 21 fJ/spike, 60 µm², VG2=Vinteg=0.275, VG1=Vleak=0                 |
| 12 | E/I synapse    | Linear DAC synapses Vw_exc / Vw_inh → soma. Thick-ox stack, Vw range 2.5–3 V                   |
| 13 | Iion model     | Semi-empirical: I_ion = a·exp(b(VD+c)) + d(VD+f)^e ; a,b,d,e,f are PWL(VG); c constant         |
| 14 | Meas vs SPICE  | IB and ID vs VD, VG1 0.00→0.55 V family; **VB = 0 V fixed** (model-isolation step)             |
| 15 | Transient      | VD ramp at VG1=0.3 V, VG2 swept; meas (squares) vs sim (dashed)                                |
| 16 | Process slide  | 130 nm triple-well: deep N-well, isolated P-well; 3×3 µm² min cell                             |
| 17 | 1T cell        | 180 nm thick-ox, area 8 µm², VG up to 0.8 V, VNwell 3 or 5 V; pre-write −1 V widens DR         |
| 18 | 2T cell        | Thick-ox, area 17 µm², VG1<0.8 V, VG2<0.5 V (or floating), VD<3.5 V, VNwell>2 V                |
| 19 | Brian2 LIF     | Behavioural emulation: VG1=550 mV, VG2=500 mV, Cint=170 fF                                     |
| 20 | SNN results    | Confusion matrices: LIF 72% vs Poisson 85%                                                     |
| 21 | 2T + diode     | **Explicit parasitic N-well diode VNwell→VB**; trise sweep 10 µs / 100 µs / 1 ms               |
| 22 | Param page NEW | Source of `three_branch_params_extracted.json` — 2×2 panel (= Image 07 refreshed 2026-05-02)   |

---

## 3. Cross-Image Consistency Map

**Same device, multiple windows:**
- 2T floating-body cell: 05, 06, 08, 09, 13, 14, 15, 18, 21, 22 (10 slides)
- Param extraction summary: 01, 02, 03, 04, 07, 22 (the 4 single-panel slides + composite + new)
- Soma / system use: 10, 11, 12, 19, 20
- Process cross-section: 16, 17, 18

**Stable conventions across the corpus:**
- `VG1` = M1 gate (leak), `VG2` = M2 gate (integration / bleed)
- `VD` = M1 drain (= Vmem in soma circuits)
- `VB` = floating P-body (= M2 drain = Vspike in soma circuits)
- `VS = VNEG = 0 V`; `VNwell` separately biased
- Three colour code: red = VG1 0.20 V, blue = 0.40 V, black = 0.60 V

**Discrepancies (benign):**
- Image 05 uses VG2 up to **1.4 V** (older / overdrive experiment); later slides cap VG2 at 0.5 V
- Image 14 pins **VB = 0 V** to isolate the I_ion law — not a contradiction to floating-body operation elsewhere
- Image 22 supersedes 01–04 / 07 as the canonical parameter page (latest extraction, 2026-05-02)

**`image-2.png` confirmed identity:**
The NEW image is the **four-panel master parameter page** (BETA0, ETAB top row; K1, NFACTOR bottom row) — i.e., the direct source PNG behind `three_branch_params_extracted.json`. It is structurally the same as Image 07 but with refreshed numerics. Branch identities match the JSON exactly (red 0.20, blue 0.40, black 0.60 V VG1 branches).

---

## 4. Topology Insights — NEW vs Current Pyport

Current pyport (`scripts/z70_*`, `z83_*`, `nsram_fpga_bridge.py`, etc.) models the cell as **BSIM4 (M1) + Gummel-Poon parasitic BJT (M2 leak) + 3-branch I-V extraction**. It does NOT include explicit VB-node capacitances, the N-well diode, or input-driver pulse shaping.

Locked gate requires >=3 new insights. We have **7**:

1. **Parasitic N-well diode VNwell → VB is a separate circuit element**, not a transport-model add-on (Image 21 explicit, 16/17 implicit). Its junction capacitance + voltage-dependent leakage materially shifts the firing knee and is the dominant source of ramp-rate sensitivity (Images 08, 15, 21). **NOT in our pyport.**

2. **Deliberate VB–VG2 MOS coupling capacitor** (Image 05 schematic, drawn as an explicit cap) injects gate transitions onto the floating body and sets the spike rise time. This is a *designed* element, not a parasitic. **NOT in our pyport.**

3. **VB is an observable output node (Vspike)**, not just an internal state. In the soma cells (Image 10), M2's drain (= VB) is the read-out terminal. Our 3-branch I-V model treats VB only as an internal Vb-clamp parameter and never reports it; compact models that lack this node cannot reproduce self-reset waveforms.

4. **VD ↔ Vmem mapping inverts the role of the "drain ramp"**. In Image 10 the M1 drain is wired to the integrating capacitor Cint (102 fF) — i.e., what device characterisation calls a swept VD is in-circuit the *membrane potential driven by I_excit*. Our pyport currently sweeps VD as an external source; closed-loop integration with Cint at the drain is missing.

5. **NFACTOR(M2) depends on BOTH VG2 AND VG1** (Images 01, 22). VG1 couples into M2 only through VB. This proves the body node is a *shared state* between the two devices, not an isolated per-device parameter. Our current parameter table treats NFACTOR as VG2-only would lose the cross-coupling.

6. **Starved-inverter front-end at ~1 V swing** (Images 11–12) is part of the firing model. The soma fires correctly only with that pulse-shape input; our bridge tests have been driving with continuous-time signals. The starved inverter sets effective input impedance and rise-time of I_excit pulses.

7. **VNwell bias domain (≥2–5 V) + thick-oxide constraint** (Images 16–18) defines the legal operating window. Image 05's VG2=1.4 V case is outside the thick-ox regime used in 18/21/22 — when comparing old vs new datasets we must mark which oxide flavour was used. Our pyport does not track oxide flavour at all.

---

## 5. Implications for `pyport_v5`

Minimum to absorb before next fit cycle:
- Add a **D_NWELL** SPICE diode (with Cj) between VNwell and VB, parameters from Image 21's ramp-rate data
- Add an explicit **C_GS_M2 = C(VB, VG2)** coupling cap (Image 05); start at MOS-cap default (W·L·Cox of M2 gate area) and let the dynamic VD-ramp data refine it
- Expose VB as a **named output** of the cell model, not a hidden internal state — enables direct comparison to Vspike traces (Image 10)
- For soma-mode tests, wrap the cell in a Cint=102 fF loop at VD, drive with an I_excit current source, not a V_DD source

---

## 6. Pitfalls / Caveats Surfaced

- **Old VG2=1.4 V data (Image 05) is not in the thick-ox window**; do not include in the joint fit without flagging.
- **Image 14's VB=0 V** is a modelling crutch for isolating I_ion — do NOT confuse with floating-body operation.
- **Brian2 numbers (Image 19, Cint=170 fF) do not match silicon (Cint=102 fF, Image 10)** — these are different abstraction levels.
- Image 20 (LIF 72% vs Poisson 85%) — **the Poisson reference outperforms LIF**; not a "feature of NS-RAM", just a sanity check of the SNN training pipeline.

```
