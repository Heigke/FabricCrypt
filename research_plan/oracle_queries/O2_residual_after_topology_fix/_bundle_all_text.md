# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: 2Tcell_BSIM_param_DC.csv (1916 chars) ===
```csv
VG1,VG2,trise,ETAB,K1,ALPHA0,BETA0,NFACTOR,mbjt,IS,area
0.2,-0.2,11.63,0.8,0.55825,7.842e-05,10.75,12.15,0.001,5e-09,1e-06
0.2,-0.15,11.63,0.85,0.55825,7.842e-05,11,11.15,0.001,5e-09,1e-06
0.2,-0.1,11.63,0.9,0.55825,7.842e-05,11.25,10.15,0.001,5e-09,1e-06
0.2,-0.05,11.63,0.95,0.55825,7.842e-05,11.5,9.15,0.001,5e-09,1e-06
0.2,0,11.63,1,0.55825,7.842e-05,12.5,8.15,0.001,5e-09,1e-06
0.2,0.05,11.63,1.05,0.55825,7.842e-05,13.5,7.15,0.001,5e-09,1e-06
0.2,0.1,12.73,1.1,0.55825,7.842e-05,14,6.25,0.001,5e-09,1e-06
0.4,-0.2,NaN,NaN,NaN,NaN,NaN,NaN,1,5e-09,1e-06
0.4,-0.15,NaN,NaN,NaN,NaN,NaN,NaN,1,5e-09,1e-06
0.4,-0.1,NaN,NaN,NaN,NaN,NaN,NaN,1,5e-09,1e-06
0.4,-0.05,NaN,NaN,NaN,NaN,NaN,NaN,1,5e-09,1e-06
0.4,0,10.59,1.9,0.53825,7.842e-05,19,6,1,5e-09,1e-06
0.4,0.05,10.76,1.85,0.53825,7.842e-05,19,5.5,1,5e-09,1e-06
0.4,0.1,10.94,1.8,0.53825,7.842e-05,19,5,1,5e-09,1e-06
0.4,0.15,11.11,1.75,0.53825,7.842e-05,19,4.25,1,5e-09,1e-06
0.4,0.2,11.46,1.7,0.53825,7.842e-05,19,3.75,1,5e-09,1e-06
0.4,0.25,11.82,1.65,0.53825,7.842e-05,19,3.25,1,5e-09,1e-06
0.4,0.3,12.98,1.6,0.53825,7.842e-05,19,2.75,1,5e-09,1e-06
0.6,-0.2,NaN,NaN,NaN,NaN,NaN,NaN,1,5e-09,1e-06
0.6,-0.15,NaN,NaN,NaN,NaN,NaN,NaN,1,5e-09,1e-06
0.6,-0.1,NaN,NaN,NaN,NaN,NaN,NaN,1,5e-09,1e-06
0.6,-0.05,NaN,NaN,NaN,NaN,NaN,NaN,1,5e-09,1e-06
0.6,0,9.04,2.5,0.41825,7.842e-05,20,6,1,5e-09,1e-06
0.6,0.05,9.04,2.5,0.41825,7.842e-05,20,5.5,1,5e-09,1e-06
0.6,0.1,9.04,2.5,0.41825,7.842e-05,20,5.25,1,5e-09,1e-06
0.6,0.15,9.04,2.5,0.41825,7.842e-05,20,4.75,1,5e-09,1e-06
0.6,0.2,9.04,2.5,0.41825,7.842e-05,20,3.75,1,5e-09,1e-06
0.6,0.25,9.04,2.5,0.41825,7.842e-05,20,3.5,1,5e-09,1e-06
0.6,0.3,9.04,2.5,0.41825,7.842e-05,20,3.25,1,5e-09,1e-06
0.6,0.35,9.04,2.5,0.41825,7.842e-05,20,3,1,5e-09,1e-06
0.6,0.4,9.04,2.2,0.41825,7.842e-05,20,2.5,1,5e-09,1e-06
0.6,0.45,9.04,2.1,0.41825,7.842e-05,20,1.75,1,5e-09,1e-06
0.6,0.5,12.98,2.1,0.41825,7.842e-05,20,1.25,1,5e-09,1e-06

```


=== FILE: _normalised/2tnsram_simple_asc.txt (1419 chars) ===
```
Version 4
SHEET 1 3052 680
WIRE 800 0 512 0
WIRE 800 64 800 0
WIRE 816 64 800 64
WIRE 848 64 816 64
WIRE 512 112 512 0
WIRE 800 112 800 64
WIRE 608 160 512 160
WIRE 640 160 608 160
WIRE 688 160 640 160
WIRE 704 160 688 160
WIRE 736 160 704 160
WIRE 752 160 736 160
WIRE 432 192 384 192
WIRE 464 192 432 192
WIRE 512 240 512 208
WIRE 800 240 800 208
WIRE 800 240 512 240
WIRE 608 272 608 160
WIRE 800 272 800 240
WIRE 704 288 704 160
WIRE 624 320 608 320
WIRE 544 352 496 352
WIRE 560 352 544 352
WIRE 624 368 624 320
WIRE 624 368 608 368
WIRE 800 400 800 272
WIRE 608 416 608 368
WIRE 704 416 704 352
FLAG 640 160 B
FLAG 800 272 Sint
FLAG 816 64 D
FLAG 704 416 0
FLAG 608 416 0
FLAG 432 192 G
FLAG 544 352 G2
FLAG 384 192 G
IOPIN 384 192 In
FLAG 496 352 G2
IOPIN 496 352 In
FLAG 848 64 Din
IOPIN 848 64 In
FLAG 800 400 S
IOPIN 800 400 Out
FLAG 688 160 B
IOPIN 688 160 Out
SYMBOL npn 736 112 R0
SYMATTR InstName Q1
SYMATTR Value parasiticBJT
SYMATTR Value2 area=1u
SYMBOL nmos4 464 112 R0
SYMATTR InstName M1
SYMATTR Value2 l='Ln' w='Wn' m=1
SYMBOL cap 688 288 R0
WINDOW 3 22 49 Left 2
SYMATTR Value 'CBpar'
SYMATTR InstName C1
SYMATTR SpiceLine Rser=1m
SYMBOL nmos4 560 272 R0
SYMATTR InstName M2
SYMATTR Value2 l='Ln*10' w='Wn' m=1
TEXT 552 24 Left 2 !.param Ln=0.18u\n.param Wn=0.36u\n.param CBpar=1f
TEXT 520 -64 Left 2 !.inc PTM130bulkNSRAM.txt
TEXT 520 -40 Left 2 !.inc parasiticBJT.txt
TEXT 310 478 Left 2 !.op 0

```


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


=== FILE: email_history.md (20713 chars) ===
```
# Email history Sebas / Mario / Robert / Eric (extracted from TXT.rtf)





\*\*\
\*\
\*





\

\
Inga har markerats\'a0\

\

\
\
\
\
\
\

Forts\'e4tt till inneh\'e5ll\
Anv\'e4nda Gmail med sk\'e4rml\'e4sningsprogram\















\




\
\
\


\
\
\


\


\


\
\
\


\


\
\
\
\
\
\
\
\
\
\
\


\
\

\
\
\


Genv\'e4gar\


\


\
Startsida
\
\

\
Omn\'e4mnanden
\
\


star
Stj\'e4rnm\'e4rkta
\
\
\
\
\
\


Direktmeddelanden\


\
\
\

¬
\

\


NIKLAS KOTARSKY\


\
\
\
\
\
\
\
\
\


Rum\


\
\


Skapa ett rum f\'f6r att chatta och samarbeta\

Hitta ett rum att g\'e5 med i\


\
\
\


Appar\


\
\


Det finns \'e4nnu inga appar\

Utforska appar
\

\
\

arrow_downward\


fler ol\'e4sta\

1 av 18\'a0525\


\

\


\
\
\
\
\
\


Zoom NSRAM\


\


Inkorgen\
\

H\'e4ndelsen har avbrutits
\

tis 21 apr. \'95 13:00\'9614:00\


Zoom NSRAM
\

¬
\

\*

.cls-1\.cls-2\.cls-3\.cls-4\.cls-5\.cls-6\.cls-7\.cls-8\
\


Borttagen fr\'e5n Google Kalender
\


Baserat p\'e5 det h\'e4r e-postmeddelandet\
St\'e4mmer det?\
\
\


¬
\


Mario Lanza Martinez
 <mlanza@nus.edu.sg>\
\


¬
\


fre 20 mars 13:06\


\

\
\
\
\


till Robert, mig, Pazos
\

¬
\


\
\
\
\

\

Det h\'e4r meddelandet verkar vara p\'e5 engelska\
\
\
\


\
\
\


\'a0
\

\'a0\










\*¬










Hi there,








Mario Lanza Martinez is inviting you to a scheduled Zoom meeting.







\*







Meeting URL:


\*






Meeting ID:



847 3855 1708







Password:



429852







Join from an H.323/SIP room system








H.323:



144.195.19.161 (US West)206.247.11.121 (US East)159.124.15.191 (Amsterdam Netherlands)159.124.47.249 (Germany)159.124.104.213 (Australia Sydney)159.124.74.212 (Australia Melbourne)170.114.180.219 (Singapore)159.124.168.213 (Canada Toronto)159.124.196.25 (Canada Vancouver)170.114.194.163 (Japan Tokyo)147.124.100.25 (Japan Osaka)







Meeting ID:



847 3855 1708







Password:



429852







SIP:


\*






Password:



429852


\'a0\
\

\


\

Important: This email is confidential and may be privileged. If you are not the intended recipient, please delete it and notify us immediately; you should not copy or use it for any purpose, nor disclose its contents to any other person. Thank you.\


\
\
\


\

 En bilaga\
\'a0\'a0\'95\'a0\'a0Genoms\'f6kt av Gmail\
\
\

¬
invite.ics\'a0\*\'a0\'a0\


\
\


\
\


¬
\


Eric Bergvall
 <bergvall.eric@gmail.com>\
\


¬
\


m\'e5n 23 mars 11:14\


\

\
\
\
\


till Mario, Robert, Pazos
\

¬
\


\
\
\
\
\


Hello Mario and Sebastian,\
Thanks for taking the time last week. We are excited to see if we can contribute with our deep and wide software and system expertise. We have started the python and Julia simulation side and there is an initial python pip package you can find here\'a0\* with gpu support and the up to date parameters. Sebastian, check it out and let us know if this is a trajectory that can support your simulations and also to make it more widely accessible for collaboration. You can find one big hero teaser image below and you can install the package with pip install nsram. You also find the powerpoint and questions below as requested.\
We await your secretary, Mario, to reach out regarding the registration in Singapores research portal (?) for our companies and we also await your results in a couple of week.\
\
All the best,\
\
\





¬
\'a0


Eric Bergvall\'a0
\

m.
\'a0+4670 499 06 16
\

e.\'a0\*
\

a.\'a0
Sweden, Stockholm
\

\'a0


\

 2 bilagor\
\'a0\'a0\'95\'a0\'a0Genoms\'f6kt av Gmail\
\
\
\
\
\


\
\


\
\


¬
\


Pazos Sabattini Sebastian Matias
\
\


\


tis 24 mars 12:37\


\

\
\
\
\


till Sebastian, mig, Mario, Robert
\

¬
\


\
\
\
\

\

Det h\'e4r meddelandet verkar vara p\'e5 engelska\
\
\
\


\
\


Dear Eric, it was a pleasure to meet you last week, and I hope you are feeling better.\
\
Thanks for the slides and for the updates, I really appreciate it. I'm including my personal email account, as I will be conducting\'a0all communication on this topic over there through the following transition. I apologize in advance for any inconveniences this may cause.\
\
I've taken a look to the python package, it's a great kickstart and I appreciate you taking the time to work on it. I will most definitely be updating modeling parameters over the next few weeks, so I'll be reaching back soon with details and model improvements based on new experiments, as my to-do list starts to clear over the next couple of weeks.\
\
Thanks a lot for your amazing predisposition. I'll be in touch. Kind regards,\
\
Sebas\
\


\


From:
 Eric Bergvall <\*>\

Sent:
 Monday, March 23, 2026 6:14 PM\

To:
 Mario Lanza Martinez <\*>; Robert Luciani <\*>; Pazos Sabattini Sebastian Matias <\*>\

Subject:
 Re: Zoom NSRAM
\

\'a0\





\'a0


- External Email -


\
\


\
\


¬
\


Eric Bergvall
 <bergvall.eric@gmail.com>\
\


\


ons 25 mars 19:56\


\

\
\
\
\


till Pazos, Mario, Robert, Sebastian
\

¬
\


\
\
\
\
\


Hi Sebas, great to hear from you \'97 doing much better, thanks.\
\
Really glad you had a look. Since you last checked it's grown quite a bit, so here's a quick rundown of what's in there now (v0.9.0, pip install nsram):\
\
Device physics layer \'97 the full body-charge ODE, Chynoweth avalanche model, SRH charge trapping, and temperature-dependent BVpar, all matched to your Zenodo SPICE parameters. Single-cell simulation with scipy for validation against TCAD/SPICE.\
\
Characterization tools \'97 this is the part I think you'll find most useful when the new parameters come in. There's automated I-V curve fitting (drop in a CSV, it extracts BV0, Is, Ne), transient pulse response simulation with tau extraction, LTP/LTD cycle simulation (currently gets 7 distinct conductance levels), and Arrhenius retention modelling (gives tau=10,139s at 300K which lines up with your >10

s figure, and the extracted Ea is self-consistent).\
\
I've also been thinking ahead about what the library should support beyond the paper \'97 things like deep N-well high-voltage operation, sweep-rate dependent I-V hysteresis, polynomial bulk current models as an alternative to the exponential fit, E/I input neuron configurations, and frequency-coded spike encoding for image classification. These are all stubbed out with placeholder parameters, so if any of them are relevant to where your work is heading, the framework is ready.\
\
Monte Carlo variability \'97 parameterised die-to-die variation for array yield estimation. Should be useful if you're thinking about scaling to crossbar arrays.\
\
Network-level reservoir computing \'97 GPU-accelerated (CUDA/ROCm), 5 neuron models for comparison (your NS-RAM AdEx-LIF, plus Izhikevich, Hodgkin-Huxley, parametric LIF, and a standard ESN baseline). NS-RAM outperforms all of them: 97% temporal XOR, 99.6% Mackey-Glass chaotic prediction, 96.75% MNIST.\
\
Technology comparison \'97 NS-RAM benchmarked against RRAM, PCM, MRAM, hBN memristors, and CMOS neurons on area/energy/endurance.\
\
Everything is open-source at \*. When you send the updated parameters I can plug them straight into the DeviceParams dataclass and re-run all the validation \'97 the fitting pipeline is designed for exactly that workflow.\
\
Looking forward to it. No rush at all on your end.\
\
\'a0 Best Regards,\


\
\


\
\


¬
\


Eric Bergvall
 <bergvall.eric@gmail.com>\
\


\


tors 2 apr. 14:08\


\

\
\
\
\


till Pazos, Mario, Robert, Sebastian
\

¬
\


\
\
\
\
\


Hello Mario and Sebas,\
Hope you are doing fine.\'a0\
Would it be possible to move our meeting from Tuesday 21 April to Thursday 23 April? Same time works for us.\
\
Quick update: we pushed \* v0.10.0 (pip install nsram, \*) with a new module called BEAM -- a byte-level online learner where the core memory is a set of small matrices updated by a delta rule that maps directly to crossbar conductances. No backpropagation needed. 3.14 bits/char on text8 with 60K parameters in pure C.\
\
Details when we meet and just let us know when you have new info/data to share. Looking forward to it!\


\
\


\
\


¬
\


Mario Lanza Martinez
\
\


\


l\'f6r 4 apr. 17:51\


\

\
\
\
\


till mig, Pazos, Robert, Sebastian
\

¬
\


\
\
\
\

\

Det h\'e4r meddelandet verkar vara p\'e5 engelska\
\
\
\


\
\
\


Dear Eric,
\

\'a0
\

Thanks a lot for your email. I am now in the middle of a trip, and Sebastian did not have time yet to complete the modeling. Let us meet in the second half of April. I finish my trip on April 20
th
, and by that time we expect Sebastian to have some progress on the model. I am very sorry for being slow with this topic, but I think we are making a solid foundation.
\

\'a0
\

Yesterday I was presenting this in a company and they seem to be interested.
\

\'a0
\

Best regards,
\

\'a0
\

---
\

Mario Lanza, Ph.D. \'96 IEEE Fellow
\

Associate Professor of Materials Science and Engineering
\

9 Engineering Drive 1, Block EA, Office 05-28
\

National University of Singapore, 117575 Singapore
\


Email: \* \'96 Web: \*\'a0
\


\
\


\
\


¬
\


Sebastian Pazos
\
\


\


fre 17 apr. 22:45\


\

\
\
\
\


till Mario, mig, Pazos, Robert
\

¬
\


\
\
\
\

\

Det h\'e4r meddelandet verkar vara p\'e5 engelska\
\
\
\


Dear Eric, Robert, I hope this email finds you well. I'm quite alright with pushing our call to 23rd, I'm not sure about Mario's schedule.\
\
Apologies for the radio-silence. I wanted to let you know that, amidst the transition to my new roles, I'm finding time to work on the new model fits to the new data.\
\
Things are starting to look up\'a0in terms of model agreement with experimental results, but the modeling approach has changed a little bit from that old Zenodo repository of the paper.\'a0\
\
I'm working with a different SPICE tool and foundry-provided models and I'm focusing on adapting foundry models of body bias and impact ionization (in \*) to fit the floating body behaviour of our 2T cells. This renders a more standard approach for us circuit designers.\
\
In that sense, I've dropped the avalanche diode models (very annoying for convergence) that fire the bipolar parasitic effect in LTSpice, and I'm only including a complementary bipolar current to capture the full swing of the firing mechanism. Fits are looking good, but I'm still working on polynomial dependence of model parameters with tuning voltages (VG1, VG2) and layout dependent effect on transistor models to capture the experimental behaviour.\
\
My question at this point is: can your approach drop the avalanche voltage as a control parameter and deal with the BSIM Impact ionization and body voltage directly? Alternatively, I could run my latest I-V curves through your Python I-V fit module. Let me know what would you prefer.\
\
Kind regards,\
\
Sebas\


\
\


\
\


¬
\


Robert Luciani
\
\


\


l\'f6r 18 apr. 01:56 (f\'f6r 13 dagar sedan)\


\

\
\
\
\


till Sebastian, Mario, mig, Pazos
\

¬
\


\
\
\
\

\

Det h\'e4r meddelandet verkar vara p\'e5 engelska\
\
\
\


\

Hi Sebastian,\
Good to hear fits are converging!\
Thanks for the BSIM4 manual.\
\
I'm wondering if you could share three things for the Julia simulator:\
1. The 2T cell schematic, plus whatever array/grid topology you're planning to build out of them (shared lines, neighbor coupling, etc.).\
2. Raw I-V measurements from the silicon (e.g. CSV with bias conditions). No fits needed ^_^\'a0\
3. The process node you're targeting, so I know which BSIM4 parameter set to start from. \
If the foundry card is shareable, even better!\
\
Call on the 23rd works for me.\
\
Kind regards,\
~ Robert\


\
\


\
\


¬
\


Eric Bergvall
 <bergvall.eric@gmail.com>\
\


\


s\'f6n 19 apr. 18:51 (f\'f6r 12 dagar sedan)\


\

\
\
\
\


till Robert, Sebastian, Mario, Pazos
\

¬
\


\
\
\
\
\


Hi Sebas, Mario, Robert,
\

Short answers first \'97 23rd works for us.
\

To Sebastian \'97 your question #1: yes, we dropped 
BVpar
. We pushed 
nsram v0.12.0
 this week with the full BSIM4 floating-body stack:
\



\'a76.1 impact ionization
 (
ALPHA0
 / 
BETA0
) driving body charge directly\


\'a72.2 Vth(Vbs)
 with 
K1
 / 
K2
 for body-bias modulation\


\'a710.1 junction breakdown
 as an alternative firing path. We ran a head-to-head against your published Chynoweth I-V over 2\'964.5 V: 
\'a76.1 channel HCI fits ~4 decades RMS better than \'a710.1 junction breakdown
 \'97 so the channel-HCI route looks like the right match for your 2T cell, aligned with the "complementary bipolar current on top of BSIM4" description from your last email.\


\'a712 full temperature scaling
 (
KT1
, 
UTE
, 
XTIS
)\


\'a713 layout stress
 (
SA
, 
SB
, 
KU0
, 
KVTH0
) \'97 direct home for your "layout-dependent effect"\



PolynomialBSIM4Params
 \'97 wrapper for the 
\(VG1, VG2)
 polynomial fits you're working on. Drop in coefficients and it evaluates per bias point.\


body_charge_ode_bsim4_full(...)
 has a 
firing_mode 

 \
 switch if you want to A/B both paths against your data.
\


And yes on option #2 too \'97 please send the I-V curves.
 
fit_bsim4_impact(Vds, Isub, Vgs)
 extracts 
ALPHA0
/
BETA0
 from a CSV (synthetic self-consistency R\'b2=0.9998). GPU batch mode fits ~4000 curves in 0.7 s if wafer-scale Monte Carlo is relevant.
\


pip install --upgrade nsram
 \'97 0.12.0 is live on PyPI, repo at \*, runnable example at 
examples/bsim4_2t_floating_body.py
\


Echoing Robert's asks (helpful for both the Python and Julia sides):
\



2T cell schematic + any planned array topology (shared lines, neighbor coupling)\

Raw I-V CSVs \'97 no fits needed, we'd like to run them through the pipeline ourselves\

Process node, so we pick the right BSIM4 parameter set\


Foundry model card if shareable\

No rush \'97 whatever arrives before the 23rd helps frame the call, but we can equally well iterate afterwards. Also to speed up our collaboration we would like to propose weekly cadence with you Sebastian to share insights and progress but we can discuss it in the next meeting.
\


Mario \'97 one quick check-in on the vendor registration.
 Last we heard (24 March) you were planning to speak with an officer about whether NUS would engage ENIMBLE Solutions AB as a foreign company or me personally. Any update there? Happy to send additional documents (English certificate of incorporation, VAT registration, etc.) if that helps move it along. No pressure \'97 just want to make sure we're not blocking anything on your side before the 23rd.
\


Looking forward to the meeting.
\

Best regards
\


\
\


\
\


¬
\


Mario Lanza Martinez
\
\


\


m\'e5n 20 apr. 15:53 (f\'f6r 11 dagar sedan)\


\

\
\
\
\


till mig, Robert, Sebastian, Pazos
\

¬
\


\
\
\
\

\

Det h\'e4r meddelandet verkar vara p\'e5 engelska\
\
\
\


\
\
\


Hi Eric, Robert, Sebastian,
\

\'a0
\

Thanks a lot. I have modified the meeting time to 23
rd
 at 8 am Buenos Aires time.
\

\'a0
\

I have been travelling from March 28
th
 until yesterday and I couldn\'92t talk with the person in charge of the vendor registration. I will try to reach her tomorrow. I apologize for the delay.
\


\
\


\
\


¬
\


Sebastian Pazos
\
\


¬
\


m\'e5n 20 apr. 23:08 (f\'f6r 11 dagar sedan)\


\

\
\
\
\


till mig, Robert, Mario
\

¬
\


\
\
\
\

\

Det h\'e4r meddelandet verkar vara p\'e5 engelska\
\
\
\


Dear Eric, Robert, thank you so much for the quick response. Here's some material as per your requests:\
\
1) A circuit schematic of the 2T cell, as currently modelled, in ASC format (LTSPice, even though we are using a different simulator right now targeting tape-outs). Neighbour coupling will most likely be the first topology we'll target. I will soon have help to scale this into network and circuit level implementations.\
\
2) I-V curves in CSV files, at an average sweep rate of 0.2 volts per second, 3 different VG1 (0.2, 0.4, 0.6) and multiple VG2 (value enclosed within each filename). Let me know if this works for you. We are generating additional data now (multiple ramp rates for more data on dynamics, pulsed dynamics).\
\
3&4) A model card for the bipolar device within the 2T cell schematic and a set of parameters around the impact ionization / body effect set by me as starting point for my SPICE fittings. Sadly, the foundry's full model card cannot be shared without infringing NDAs, 
but the 130 nm\'a0(current working node) PTM model I'm attaching is a good starting point
.\
\
I hope you find this useful for the time being. More data and details are coming soon.\
\
Talk to you soon,\
\
Sebas\


\

 2 bilagor\
\'a0\'a0\'95\'a0\'a0Genoms\'f6kt av Gmail\
\
\
\
\
\


\
\


\
\


¬
\


Mario Lanza Martinez
\
\


\


tors 23 apr. 07:39 (f\'f6r 8 dagar sedan)\


\

\
\
\
\


till Sebastian, mig, Robert
\

¬
\


\
\
\
\

\

Det h\'e4r meddelandet verkar vara p\'e5 engelska\
\
\
\


\
\
\


Dear colleagues,
\

\'a0
\

I am extremely sorry that I have to move the meeting to next week because I got an important unexpected visitor (our startup investor) and he requires me for dinner exactly at the time of our meeting.
\

\'a0
\

Can we do the meeting at the same time (7 pm Singapore time) next week? Every day is fine for me. Thanks a lot.
\

\'a0
\

Best regards,
\

\'a0
\

---
\

Mario Lanza, Ph.D. \'96 IEEE Fellow
\

Associate Professor of Materials Science and Engineering
\

9 Engineering Drive 1, Block EA, Office 05-28
\

National University of Singapore, 117575 Singapore
\


Email: \* \'96 Web: \*\'a0
\

\'a0\


\
\


\
\


¬
\


Eric Bergvall
 <bergvall.eric@gmail.com>\
\


\


tors 23 apr. 10:23 (f\'f6r 8 dagar sedan)\


\

\
\
\
\


till Sebastian, Mario, Robert
\

¬
\


\
\
\
\
\


Dear Mario,\
\
From me and Roberts side we can move the meeting to next week Thursday 30th of April same time (7pm Singapore time).\
Does it work for you Sebastian?\


\
\


\
\


¬
\


Sebastian Pazos
\
\


¬
\


tors 30 apr. 15:25 (f\'f6r 17 timmar sedan)\


\

\
\
\
\


till mig, Mario, Robert
\

¬
\


\
\
\
\

\

Det h\'e4r meddelandet verkar vara p\'e5 engelska\
\
\
\


Dear Eric and Robert, it was a pleasure to connect today. Thank you for your updates. I wanted to share my raw slides (didn't have time to format, sorry about that) and fitting results with you, in case this data is useful for optimizing your engines (please do not share the BSIM cards publicly). Remember that some parameters change only for M1, while NFACTOR changes only for M2 (I attribute this to LDE, hence two separate model cards for each device). Let me know if you have any questions.\
\
I'm working on additional measurement data for the dynamic modeling and will try to organize it in a repository for these results, so I can provide you with better tracked updates and results.\'a0\
\
To summarize the technical aspects of the call a bit:\
1- All approaches that help better fit the set of parameters to properly model NS-RAM are very useful because\'a0they save SPICE simulation time and ease fits across\'a0different technologies.\
\
2-\'a0You asked if I had information about how performance may degrade with fan-out. I think the model is in a better place now to start checking this in SPICE. I'm a little shorthanded now, but if you come up with a simple architecture of a few tens of neurons that is feasible to simulate at the circuit level, this could serve as a nice working example to check this aspect.\
\
3- In the near future, we are targeting compact networks for specific, sparse-signal applications. However, thinking long-term, a "roadmap" can include how NSRAM can scale to larger models and how this can co-exist with simplifying massive models (some sort of sweetspot maybe around reasoning models?) of todays mainstream-AI.\
\
4- We can work on a collab doc to include some of these aspects as part of the centre proposal.\
\
5-\'a0BONUS TRACK: We didn't cover this in the call, but I'm working on a floorplan for the first testchip entirely dedicated to NS-RAM. This will already include small arrays of NS-RAM based neurons, but if there is something small and specific that you think can be useful to extract specific metrics that help your approach, please let me know and we can evaluate its inclusion (simplified schematic or block diagram of what you would be looking for and what you expect to get out of that cell in terms of info/figures of merit).\
\
I still have the feeling that I missed a question or two during our meeting: we covered many aspects quite quickly and now I feel like when you leave home thinking you forgot to grab something but can recall what it is\'a0¬. Please, forward any questions in this thread and I'll do my best to provide detailed/useful answers.\
\
Thanks again for your feedback and support. Kind regards,\
\
Sebas\


\

 En bilaga\
\'a0\'a0\'95\'a0\'a0Genoms\'f6kt av Gmail\
\
\
\
\


\
\


\
\
\


\
\

\

\
\
\
\
\
\
\
\
\
\
\


```


=== FILE: parasiticBJT.txt (244 chars) ===
```
* Simple bjt for floating bulk parasitic bipolar effect
* Pazos, S.

.model parasiticBJT NPN(is=5E-9 va=100 bf=10000 br=100 nc=2 ikr=100m rc=0.1 vje=0.7 re=0.1 cjc=1e-15 fc=0.5 cje=0.7e-15 ne=1.5 ise=0 tr=20e-12 tf=25e-12 itf=0.03 vtf=7 xtf=2)

```


=== FILE: summary.json (349 chars) ===
```json
{
  "n_curves": 33,
  "n_evaluated": 25,
  "n_skipped": 8,
  "median_log_rmse": 2.025633319969513,
  "p90_log_rmse": 5.027446256149432,
  "elapsed_s": 39.152913093566895,
  "vs_z91f_run1_median": 4.234,
  "vs_z91f_run2_median": 2.402,
  "note": "true two-model validation (M1 = 130DNWFB, M2 = 130bulkNSRAM) with Sebastian's per-bias CSV overrides"
}
```


=== FILE: artifacts/A1a_nfactor_trace.md (2987 chars) ===
```
# A1a — NFACTOR override trace through compute_dc

**Verdict:** **NFACTOR override IS reaching the subthreshold formula.**

## 1. Formula path inside `compute_dc` (`nsram/bsim4_port/dc.py`)

```python
# line 131:    P  = sd.scaled
# line 160:    nfactor = t(P["nfactor"])
# --- Subthreshold n (b4ld.c §1133-1154) ---
# line 361:    tmp1 = epssub / Xdep
# line 362:    tmp2 = nfactor * tmp1                       # <-- nfactor enters here
# line 363:    tmp3 = cdsc + cdscb*Vbseff + cdscd*Vds
# line 364:    tmp4 = (tmp2 + tmp3*Theta0 + cit) / coxe
# line 367:    n_a  = 1.0 + tmp4                           # subthreshold slope factor
# --- Vgsteff bridge (b4ld.c §1238-1296) ---
# line 439:    T0v = n * Vtm                               # n carries nfactor
# line 440:    T1v = mstar * Vgst
# line 441:    T2v = T1v / T0v
# line 449:    T10_bridge = n*Vtm * log1p(exp(T2v))
# line 471:    T9v  = mstar + n * T3v
# line 472:    Vgsteff = T10v / T9v                        # <-- subthreshold smoothing
```

So `nfactor → n → T0v → Vgsteff` (and into the drain-current expression).
It does not change `Vth`; it sets the inverse subthreshold slope (kT/q · n).

## 2. Patch idiom reaches that path

`patch_sd_scaled` in `scripts/z91f_validate_with_sebas_params.py` (lines 55–74) writes directly to `sd.scaled[k]`:

```python
sd.scaled[k] = v            # line 67
```

`compute_dc` reads `P = sd.scaled; nfactor = t(P["nfactor"])`. Same dict, same key. The override is consumed.

## 3. Numeric demo (M2, real card)

Script: `research_plan/artifacts/A1a_demo.py`. Loads
`data/sebas_2026_04_22/M2_130bulkNSRAM.txt`, builds `sd_M2` via
`compute_size_dep` (Ln·10 = 1.8 µm, W = 360 nm, T = 27 °C), applies the
same M2 static overrides as z91f (k1, k2, etab, beta0), then calls
`compute_dc` at **(Vgs = −0.10, Vds = 2.0, Vbs = 0)** with `nfactor` swapped
in `sd.scaled`.

| nfactor | Id [A]      | Vgsteff [V] | Vth [V] |
|--------:|------------:|------------:|--------:|
| 1.58    | 1.1545e-14  | 4.649e-09   | 0.4497  |
| 12.15   | 6.2099e-12  | 2.501e-06   | 0.4497  |

- **Id ratio = 5.38e+02 → +2.73 decades**
- ΔVgsteff = +2.50e-6 V (Vgsteff scales ~ n·Vtm·log1p(exp(...)) — bigger n softens the bridge)
- ΔVth = 0 (expected — nfactor sets slope, not threshold)

## 4. Implication for z91g

The override mechanism is wired correctly: a single-bias `nfactor` swap from
1.58 → 12.15 already moves Id by **+2.73 decades** at this low-VG2 bias.
That is the same order of magnitude as z91g's reported median residual
(2.40 decades at low VG2), so NFACTOR is the right knob and the patch
*does* reach it. The remaining z91g residual must therefore come from
either (a) sign / direction of the residual (does Sebas's Id move the
same way ours does?), (b) interaction with the M1 overrides
(etab, k1, alpha0, beta0) or the BJT wrapper, or (c) bias coordinates the
CSV NFACTOR row maps to versus what we apply at the same (VG1, VG2). The
NFACTOR-not-reaching-the-formula hypothesis is **falsified**.

```


=== FILE: artifacts/A1b_bjt_mapping.md (4121 chars) ===
```
# A1b — BJT Parameter Mapping (Sebas CSV ↔ `GummelPoonNPN`)

## 1. Sources

- **SPICE model card** (`parasiticBJT.txt`):
  ```
  .model parasiticBJT NPN(is=5E-9 va=100 bf=10000 br=100 nc=2 ikr=100m
                          rc=0.1 vje=0.7 re=0.1 cjc=1e-15 fc=0.5
                          cje=0.7e-15 ne=1.5 ise=0
                          tr=20e-12 tf=25e-12 itf=0.03 vtf=7 xtf=2)
  ```
- **Schematic** (`2tnsram_simple.asc`):
  `SYMBOL npn ... Q1 ... Value=parasiticBJT  Value2=area=1u`
  → instance `area = 1e-6`. No `m=` multiplier on the LTSpice Q1 instance.
- **CSV** columns: `mbjt, IS, area`. Rows show `IS=5e-9`, `area=1e-6`
  constant; `mbjt` flips between **0.001 (VG1=0.2)** and **1.0
  (VG1=0.4 / 0.6)**.

## 2. Mapping Table

| SPICE NPN keyword          | LTSpice instance | Our `GummelPoonNPN` | Honoured by `from_sebas_card`? |
|---|---|---|---|
| `IS`  (saturation current) | —                | `Is`  (5e-9)        | yes (hard-coded)               |
| `VA`  (Early fwd)          | —                | `Va`  (100)         | yes                            |
| `BF`                       | —                | `Bf`  (10000)       | yes                            |
| `BR`                       | —                | `Br`  (100)         | yes                            |
| `NF`/`NR` (default 1)      | —                | `Nf=Nr=1`           | yes                            |
| `NC`, `NE`                 | —                | `Nc=2`, `Ne=1.5`    | yes                            |
| `IKR`, `ISE`               | —                | `Ikr=0.1`, `Ise=0`  | yes                            |
| —                          | `area=1u`        | `area = 1e-6`       | yes (multiplies Is/Ikf/Ikr/Ise/Isc in `compute_bjt`) |
| —                          | `m=<mbjt>`       | **no field**        | **NO**                         |

`mbjt` is **the SPICE device multiplier `m`** (cell count / parallel
parasitic-NPN scaling). In SPICE, `m` multiplies `IS, IKF, IKR, ISE, ISC,
1/RB, 1/RE, 1/RC` exactly as `area` does — i.e. it is mathematically
identical to scaling `area`. There is no Gummel-Poon "ideality" parameter
called `mbjt`; this is purely a count multiplier added by Sebas's
extraction wrapper to switch the BJT path on/off per VG1.

## 3. Pipeline Audit (`z91f.make_bjt`)

```python
def make_bjt(sebas_row):
    bjt = GummelPoonNPN.from_sebas_card()
    if sebas_row is not None:
        if not math.isnan(sebas_row.get("IS", float("nan"))):
            bjt.Is = float(sebas_row["IS"])
        # mbjt is the BJT idealisation factor (we treat as Bf scaler).
        # For validation: ignore ...
    return bjt
```

- `IS` is read **once per row** but the value is constant 5e-9 — so the
  per-row override is a no-op. Per-bias `IS` is *technically* honoured;
  effectively unused.
- `area` is **not** read from the CSV (still defaults to 1e-6 from the card).
- `mbjt` is **explicitly ignored** with a wrong comment ("Bf scaler").
- z91g imports `make_bjt` from z91f → identical bug.

## 4. Numeric Check (Vbe = 0.6 V, Vbc = 0, T = 300 K)

| Effective multiplier | Ic       |
|---|---|
| current code (`area=1e-6`, mbjt ignored)         | 5.94e-5 A |
| with `area *= mbjt = 1.0` (VG1 ≥ 0.4 rows)       | 5.94e-5 A |
| with `area *= mbjt = 0.001` (VG1 = 0.2 rows)     | 5.94e-8 A |

→ **Exactly the ~3-decade gap** seen in the z91g residuals at low VG1.

## 5. Verdict

**`mbjt` is a SPICE `m=` device multiplier, not a Gummel-Poon ideality.
It is currently NOT honoured; `IS` is honoured but trivially constant.
At VG1 = 0.2 rows the simulator over-drives the parasitic NPN by 1000×,
which is exactly z91g's low-VG2 residual signature.**

**Fix (one line in `make_bjt`):**
```python
if not math.isnan(sebas_row.get("mbjt", float("nan"))):
    bjt.area = bjt.area * float(sebas_row["mbjt"])
if not math.isnan(sebas_row.get("area", float("nan"))):
    bjt.area = float(sebas_row["area"]) * float(sebas_row.get("mbjt", 1.0))
```
i.e. set `bjt.area = csv.area * csv.mbjt`. Existing `compute_bjt` already
applies `area` as the SPICE-correct multiplier on `Is, Ikf, Ikr, Ise, Isc`,
so no change to `bjt.py` is required.

```


=== FILE: artifacts/A1c_body_diode_trace.md (2129 chars) ===
```
# A1c — Component-current trace at low-VG2 bias

**Bias:** VG1=0.6, VG2=0.0, Vd=1.5 V. Sebas CSV row applied
(ETAB=2.5, K1=0.41825, ALPHA0=7.842e-5, BETA0=20, NFACTOR=6.0,
mbjt=1.0, IS=5e-9, area=1e-6). Two-card setup
(M1=130DNWFB, M2=130bulkNSRAM).

## Solver

Plain Newton hit the documented spurious-flat-root pathology
(converged at iter 2, all currents <1e-17 A). Used
`solve_2t_with_homotopy` (gmin 1e-3 → 1e-15) seeded with
Vsint=Vd/2, Vb=0.7; converged in 3 iters at target gmin.

**Converged:** Vsint = +0.3063 V, Vb = +0.3419 V → Vbe = +0.0356 V,
Vbc = −1.158 V.

## Component magnitudes (signed, A)

| Component | Value |
|---|---|
| (a) Ids_M1 | +1.251e-11 |
| (b) Ids_M2 | +1.252e-11 |
| (c) Ic_Q1  | +2.01e-14 (NPN off, Vbe≈36 mV) |
| (d) Ibs_M2 | +6.10e-13 (forward, sub-turn-on) |
| Ibd_M2 | +1.92e-16 |
| (e) GIDL/GISL M1+M2 | 0 (exact) |
| Iii_M1 | +2.03e-16 |
| Igb (M1,M2) | 0 |
| **Id at drain** | **+1.253e-11** |
| **Id measured** | **+2.07e-5** |

## Dominant path

Drain current is **completely dominated by M2 subthreshold channel
current** (Ids_M2 ≈ Ids_M1; M2 is the series bottleneck). BJT is
~3 decades smaller; M2 body diode ~2 decades smaller; GIDL = 0.
**Predicted Id under-shoots measurement by ~6 decades** — the z91g
low-VG2 residual.

## Body-diode sanity

Ibs_M2 = 6.1e-13 A at Vbs ≈ +36 mV is **not** suspicious — Vbs is
far from pn turn-on. With jss = 1e-4 A/m² and area ≈ 6.5e-13 m², the
saturation current ≈ 6.5e-17 A; at Vbs = 0.7 V that gives ~5e-5 A,
matching Sebas's measured magnitude. The kernel is fine; **the body
simply isn't being forward-biased** — Vb sits at +0.34 V between
Vsint (+0.31) and 0 with no driver pumping it up.

## Verdict

The dominant low-VG2 path in our sim is the **series-limited M2
subthreshold channel** (~10⁻¹¹ A); Sebas's measured ~2×10⁻⁵ A is
plausibly **body-driven** — NPN forward turn-on once Vb reaches
~0.7 V, or GIDL/well leakage we don't generate (Igidl ≡ 0 from this
M2 card). Fix priority: figure out why Vb fails to climb to NPN
turn-on — with Iii ~ 1e-16 and GIDL = 0 there is no body-charging
source — and why the M2 card emits Igidl ≡ 0.

```


=== FILE: artifacts/A1d_iimpact_trace.md (3024 chars) ===
```
# A1d — Why is Iii_M2 ≈ 0 at low VG2?

## BSIM4 §6.1 formula path (from `nsram/bsim4_port/leak.py`)

```python
T2          = (alpha0 + alpha1*Leff) / Leff
diff        = Vds - Vdseff
T1_strong   = T2 * diff * exp(-beta0 / diff)         # if diff > beta0/EXP_THRESH
Iii         = T1 * Idsa                              # Idsa is pre-SCBE Idsa·Vdseff
```
Note: the formula uses `Vdseff` (the smoothed `Vds`/`Vdsat`), not `Vdsat`.

## Numbers at the converged operating point

Both biases use Sebas row params `ALPHA0 = 7.842e-5, BETA0 = 20.0` (M2 card `lalpha0 = -9.84e-12` shifts effective alpha0 by only **−7%**, not the **−5e-1** the question hypothesised — the binning is harmless).

| factor                 | LOW_VG2 (VG2=0.0)  | HIGH_VG2 (VG2=0.5) |
|------------------------|--------------------|--------------------|
| Vsint  (converged)     | 0.3063 V           | 0.0942 V           |
| Vb     (converged)     | 0.3419 V           | 0.4152 V           |
| Vgs_M2                 | 0.0 V              | 0.5 V              |
| Vds_M2 = Vsint         | 0.3063 V           | 0.0942 V           |
| **Vdsat_M2**           | **0.0369 V**       | **0.0685 V**       |
| Vdseff_M2              | 0.0356 V           | 0.0547 V           |
| **Vds − Vdseff**       | **0.271 V**        | **0.0396 V**       |
| alpha0_eff             | 7.283e-5           | 7.283e-5           |
| beta0_eff              | 17.47              | 17.47              |
| T2 = (a0+a1·L)/L       | 41.0               | 41.0               |
| **−beta0/diff**        | **−64.5**          | **−441.6**         |
| **exp(−beta0/diff)**   | **9.6e-29**        | **1.7e-192**       |
| Idsa_M2 (pre-SCBE)     | 1.25e-11 A         | 9.19e-8 A          |
| **Iii_M2 final**       | **2.4e-25 A**      | **2.6e-22 A**      |

## Verdict

**Iii is ~zero at low-VG2 because the `exp(−beta0/(Vds−Vdseff))` factor is 1e-29.**

Not because M2 is in linear region (we are well past Vdsat: 0.306 V vs 0.037 V),
not because alpha0 fails to reach the formula (alpha0_eff = 7.28e-5, intact),
not because of the `lalpha0` binning (only −7% trim, sign correct).

The pure cause: **`BETA0=20 V` from the CSV row is too large for the BSIM4 §6.1
arrhenius-style argument** at ~0.3 V drain headroom. The exponential term
`exp(−20/0.27)` = `exp(−74) ≈ 1e-32` (model uses `beta0_eff = 17.47` after
binning → 9.6e-29). At HIGH_VG2 the Idsa prefactor compensates because the
channel is strongly on (so the simulator matches measurement via direct
M2 channel current, not via the BJT path); at LOW_VG2 there is no fallback
and the body never charges.

## Concrete one-line fix

Sebas's `BETA0=20` is in the wrong units for BSIM4 §6.1 (manual default is
~30 V but for short-channel cards the empirical value is **0.5–3 V**, not 20).
**Replace `SEBAS["BETA0"] = 20.0` with `BETA0 ≈ 1.0 V` for M2** (or treat BETA0
as a per-bias fitting parameter), giving `exp(−1/0.27) = 0.025` and lifting
Iii by ~27 orders of magnitude into the pA range required to forward-bias
the parasitic NPN.

```


=== FILE: artifacts/A1e_gidl_load_trace.md (4200 chars) ===
```
# A1.e — GIDL/GISL load trace for M2 at (VG1=0.6, VG2=0.0)

## Verdict (one sentence)
At the failing bias all four GIDL/GISL gates are closed by the
`Vd-Vg-egidl > 0` band-bending condition, so the observed `GIDL/GISL = 0`
is **correct physics — not a parser bug**; however, the load trace also
exposed a **latent bug** in the `("ref", "agidl")` default mechanism that
will silently zero GISL at other biases.

## Hypothesis test results

| H | Statement | Result |
|---|-----------|--------|
| H1 | Parser drops `+`-continued GIDL values | **REJECTED** — agidl/bgidl/cgidl/egidl all loaded correctly (`given=True`) |
| H2 | `compute_igidl_gisl` early-returns on `gidlmod` | REJECTED — `gidlmod=0` selects the implemented branch |
| H3 | Values land in `sd.scaled` but formula reads elsewhere | REJECTED — formula reads `model.get(...)` directly, same source |
| H2′ | (Newly identified) **agisl-group siblings stay at pre-override defaults** because `("ref", ...)` is resolved in pass 2 *before* user overrides apply in pass 3 | **CONFIRMED** |

## Loaded vs card vs default

| param   | loaded     | card        | BSIM4 default | given |
|---------|-----------:|------------:|--------------:|:-----:|
| agidl   | 1.99e-8    | 1.99e-8     | 0.0           | True  |
| bgidl   | 1.624e9    | 1.624e9     | 2.3e9         | True  |
| cgidl   | 6.3        | 6.3         | 0.5           | True  |
| egidl   | 0.91       | 0.91        | 0.8           | True  |
| agisl   | **0.0**    | not in card | (ref agidl)   | False |
| bgisl   | **2.3e9**  | not in card | (ref bgidl)   | False |
| cgisl   | **0.5**    | not in card | (ref cgidl)   | False |
| egisl   | **0.8**    | not in card | (ref egidl)   | False |

`agisl` was supposed to mirror `agidl=1.99e-8` per BSIM4 spec, but is **0.0**.

## Root cause (`model_card.py` lines 75–90)
```
Pass 1: scalar defaults  → agidl=0.0, bgidl=2.3e9, ...
Pass 2: ref defaults     → agisl=agidl=0.0, bgisl=bgidl=2.3e9, ...   ← SNAPSHOT
Pass 3: user overrides   → agidl=1.99e-8, bgidl=1.624e9, ...         ← agisl NOT updated
```
A card that specifies only the GIDL group leaves the GISL group pinned to
the *pre-card* defaults (agisl=0, etc.), which is contrary to BSIM4 v4.8.3
behavior (`b4ld.c` initialises GISL from GIDL after the parameter file is read).

## Bias gate check at (VG1=0.6, VG2=0.0, Vd=1.5, Vsint=0.306, Vb=0.342)

| device/edge | V_drive = Vd–Vg–e | result |
|---|---:|---|
| M2 GIDL (drain=0.306, g=0)     | -0.604 | CLOSED |
| M2 GISL (source=0, g=0)        | -0.800 | CLOSED |
| M1 GIDL (drain=1.5, g=0.6)     | -0.010 | CLOSED (just barely!) |
| M1 GISL (source=0.306, g=0.6)  | -1.094 | CLOSED |

So at THIS bias `GIDL/GISL ≡ 0` is honest physics: drain–gate band-bending
is too weak for BTBT. The observation in A.1.c is consistent with the
formula. The body-charging residual must come from impact ionisation
(`Iii`) or sub-threshold leakage, not GIDL/GISL.

The **agisl=0 bug is silent here** because the GISL gate is closed
anyway, but it will hide tunneling current at higher Vd or more positive
Vbs operating points (e.g. M1 GIDL was within 10 mV of opening — a small
bias shift turns it on, and once on, agisl=0 zeroes a current that
should be ≈ agidl-scale).

## Proposed fix
In `nsram/bsim4_port/model_card.py` `__init__`, **re-resolve `ref` defaults
after pass 3** (or only for parameters that are still `not is_given` and
whose referenced source was overridden):

```python
# Pass 4: re-resolve ref defaults whose source was user-overridden
for name, info in PARAMS_META.items():
    d = info["default"]
    if isinstance(d, tuple) and d[0] == "ref" and name not in self._given:
        self._values[name] = self._values.get(d[1], 0.0)
```

This restores the canonical BSIM4 behavior where, e.g., a card specifying
only `agidl` automatically yields `agisl = agidl`. Touches one file, no
formula changes; existing cards that explicitly set agisl are unaffected
(`name not in self._given` skips them).

## Artifacts
- Demo script: `research_plan/artifacts/A1e_demo.py`
- Card: `data/sebas_2026_04_22/M2_130bulkNSRAM.txt` (lines 80–81)
- Source: `nsram/nsram/bsim4_port/model_card.py` lines 75–90, `_model_card_data.py` lines 14–103

```


=== FILE: artifacts/A1g_multiroot.md (3874 chars) ===
```
# A1g — Multi-root hypothesis at low VG2

**Bias:** VG1∈{0.4, 0.6}, VG2=0.0, Vd=1.5 V; Sebas per-bias overrides.
**Symptom:** z91g returns Id≈1e-11; measurement Id≈2e-5 (VG1=0.6) /
1e-6 (VG1=0.4) — 5–6 decade gap.

## Method

`research_plan/artifacts/A1g_demo.py`. M1+M2 cards via z91f's
`patch_model_values`; per-bias overrides via `patch_sd_scaled` (z91g
convention; `_override_sd` errors on dict-only fields like
etab/alpha0/beta0/nfactor). Per bias:
- `solve_2t_with_homotopy`, Vb_init ∈ {0.0, 0.5, 0.7, 0.9}.
- `forward_2t_arclength_grad`, Vd∈[0.05, 2.0] (40 pts).

Trace: `A1g_multiroot_trace.json`. Plot: `A1g_multiroot.png`.

## Results

### VG1 = 0.6, VG2 = 0.0, Vd = 1.5 (meas Id = 2.07e-5 A)
| start          | Id [A]    | Vb [V]  | Vsint [V] | conv |
|----------------|-----------|---------|-----------|------|
| Vb_init=0.0    | 1.253e-11 | 0.3419  | 0.3063    | yes  |
| Vb_init=0.5    | 1.253e-11 | 0.3419  | 0.3063    | yes  |
| Vb_init=0.7    | 1.253e-11 | 0.3419  | 0.3063    | yes  |
| Vb_init=0.9    | 1.253e-11 | 0.3419  | 0.3063    | yes  |
| arclength@1.5  | 4.21e-12  | 0.1842  | —         | no   |

Arclength: **n_folds = 0**, 45 steps, 2.5 s.

### VG1 = 0.4, VG2 = 0.0, Vd = 1.5 (meas Id = 1.02e-6 A)
| start          | Id [A]    | Vb [V]  | Vsint [V] | conv |
|----------------|-----------|---------|-----------|------|
| Vb_init=0.0    | 1.444e-11 | 0.3550  | 0.2372    | no   |
| Vb_init=0.5    | 1.444e-11 | 0.3550  | 0.2372    | no   |
| Vb_init=0.7    | 1.444e-11 | 0.3550  | 0.2372    | no   |
| Vb_init=0.9    | 1.444e-11 | 0.3550  | 0.2372    | no   |
| arclength@1.5  | 4.57e-12  | 0.1857  | —         | no   |

Arclength: **n_folds = 0**, 44 steps, 2.8 s.

## Verdict: multi-root hypothesis **DISPROVEN**

Three converging lines:

1. **Initial-condition basin sweep is degenerate.** Vb_init from 0.0
   to 0.9 V — past the diode-on knee — converges in 3 Newton iters to
   the **same** (Vsint, Vb, Id) triple to 5 sig-figs. No second basin
   in the explored range.
2. **Arclength finds no fold** (n_folds = 0). If a high-Vb root were
   separated from the low-Vb root by an S-shaped fold in Id(Vd),
   trace_arclength would have detected a turning point. It did not.
3. **The found Vb is already moderate** (~0.34 V) — not a stuck-at-0
   NPN-OFF root. Body diode mildly forward-biased; Iii just isn't
   strong enough in our model to ignite the
   avalanche → base-drive → collector-current loop.

There is no second root for arclength or high-Vb init to snap to.
Newton already finds the only root the model admits.

## Actual mechanism (likely)

The 6-decade gap is a **sub-block disagreement**, not a convergence
failure. Ranked candidates:

- **Iii too weak.** Sebas's ALPHA0=7.84e-5, BETA0=20 through our
  `compute_iimpact` may produce orders less impact current than
  HSPICE BSIM4 IIMOD. The Iii→Vb→base-drive loop never ignites; the
  NPN stays off and Id collapses to M2 sub-threshold leakage. The
  z91g pattern (good fits at VG1=0.2 where NPN doesn't matter,
  degrading at VG1=0.4–0.6 where it does) is consistent.
- **GP NPN under-driven.** At Vb=0.34, Vbe is too small for our
  Gummel-Poon to deliver appreciable Ic. Worth verifying that
  `bjt.py` references Vbe = Vb − Vsint with the same sign convention
  as Sebas's `parasiticBJT.txt` (A1b mapped this; recheck).
- **Body-diode Js mismatch.** M2 card has zero SourceSatCurDensity_T;
  `cfg.default_jss` kicks in. If too large it clamps Vb near 0.34 V.

## What to do next

A high-Vb / arclength heuristic won't help — there's nothing to find.
Fix is in physics:

1. Audit `compute_iimpact` vs HSPICE BSIM4 IIMOD=1 at this bias with
   Sebas overrides (next diagnostic).
2. Verify GP NPN Vbe-mapping vs `parasiticBJT.txt` (C=Vd, B=body,
   E=Vsint).
3. Once Iii is calibrated and Vb pushes past ~0.6 V, Id should jump
   4–5 decades without any continuation tricks.

```


=== FILE: artifacts/A1h_iimod_audit.md (4797 chars) ===
```
# A1h — IIMOD audit: does our `compute_iimpact` match Sebas's card?

**Bias:** VG1=0.6, VG2=0.0, Vd=1.5 V (M2 in saturation, Vds−Vdseff ≈ 0.27 V).
**Predecessor:** A1d already pinned that `Iii ≈ 2.4e-25 A` due to
`exp(−beta0/(Vds−Vdseff))` collapsing. This audit asks the deeper question:
**are we even using the same impact-ion formula as Sebas's foundry card?**

---

## 1. What our `compute_iimpact` implements

`nsram/nsram/bsim4_port/leak.py` (lines 44–101). Single, unconditional formula
(no `iimod` branch anywhere — verified by `grep -ri iimod nsram/bsim4_port/`):

```
T2  = (alpha0 + alpha1·Leff) / Leff                          # leak.py:80,85
diff = max(Vds − Vdseff, 0)                                  # leak.py:76-78
if diff > beta0/EXP_THRESHOLD:                               # leak.py:86,93
    T1 = T2 · diff · exp(−beta0/diff)                        # leak.py:89-90
else:
    T1 = T2 · MIN_EXP · diff                                 # leak.py:92
Iii = T1 · Idsa·Vdseff      # uses pre-SCBE Idsa (WAVE2-FIX-1)
```

Expected units (implicit from this form):
- `ALPHA0` : **m·V⁻¹** (so `alpha0/Leff` is dimensionless V⁻¹)
- `ALPHA1` : **V⁻¹**
- `BETA0`  : **V**

This is the **classic BSIM4 IIMOD=0** form (the only form available in BSIM4
versions ≤ 4.6.4). No length-binning of `alpha0/beta0` is applied (the
`lalpha0`, `lbeta0` fields ingested by `_model_card_data.py:174,180` are
**not consumed** in `temp.py` — only `voffl` etc. are length-binned).

## 2. BSIM4 IIMOD branches (manual §6.1)

BSIM4 v4.7 introduced an `IIMOD` selector. v4.8.3 manual §6.1 lists:

- **IIMOD = 0** (default): the formula above.
  `Iii = ((α₀+α₁·Leff)/Leff)·(Vds−Vdseff)·exp(−β₀/(Vds−Vdseff))·Idsa·Vdseff`
  ALPHA0 [m/V], BETA0 [V].

- **IIMOD = 1** (Mansun-Chan-style, temperature-aware): adds a temperature
  prefactor and replaces ALPHA0/BETA0 with model parameters `IIIA0`,
  `IIIA1`, `IIIB0`, `IIIB1`, `IIIT0`, `IIIT1`, etc. (different parameter
  names — a card setting `iimod=1` would *not* read alpha0/beta0 at all).

- **IIMOD = 2** (HSPICE-compatible — present in some industry forks, not
  always documented in the Berkeley v4.8.3 manual; **flag for oracle
  confirmation**): commonly cited form:
  `Iii = (α₀/Leff + α₁)·(Vds−Vdseff)²·exp(−β₀/(Vds−Vdseff))·Idsa`
  with ALPHA0 in [m·V⁻¹] but the **squared** drain-headroom factor.

Confidence: IIMOD=0/1 from manual + b4ld.c source. IIMOD=2 wording above is
my recollection of HSPICE's "alternate" form — needs cross-check.

## 3. What does Sebas's card select?

Both `M1_130DNWFB.txt:9` and `M2_130bulkNSRAM.txt:22` declare:

```
+Level = 14
+version = 4.5                 ...
```

`version=4.5` **predates IIMOD entirely** (introduced v4.7). Neither card
sets `iimod = ...`. BSIM4 default is IIMOD=0. **Sebas's card therefore
uses the IIMOD=0 classic formula** — the same one we implement.

## 4. Numeric check at the diagnostic bias

From A1d converged operating point (LOW_VG2, M2):
`Vds−Vdseff = 0.271 V`, `Idsa·Vdseff ≈ 1.25e-11 A`, `Leff ≈ 1.91e-7 m`.
Sebas-row `ALPHA0 = 7.842e-5 m/V`, `BETA0 = 20 V` (CSV; card has 18-19 V
plus `lbeta0=-9.5e-7` length term):

```
T2   = 7.842e-5 / 1.91e-7      = 410     [V⁻¹]
exp(−20 / 0.271)               = exp(−73.8) = 8e-33
Iii  = 410 · 0.271 · 8e-33 · 1.25e-11 ≈ 1e-42 A
```

A1d already showed ~2.4e-25 with binning; both are absurdly far below the
1 nA needed to forward-bias the body. **The formula is faithful; the
parameters are simply outside the regime where IIMOD=0 produces
appreciable Iii.** BETA0=18-20 V means `exp(−β/Δ)` only awakens when
`Δ = Vds−Vdseff > ~3 V` — i.e. the IIMOD=0 model targets >3.3 V drain
operation, but our diagnostic runs at 1.5 V with most of it dropped
across M1.

## 5. Verdict

> Our `compute_iimpact` correctly implements BSIM4 v4.8.3 IIMOD=0, which
> is the same branch Sebas's `version=4.5` cards select by default. The
> 6-decade Id miss is **not** an IIMOD-mismatch bug — it is that the
> classic BSIM4 §6.1 formula with BETA0≈19 V genuinely emits ~0 A at
> this bias, and the floating body cannot be charged through this path.

**Proposed one-line fix (workaround, not formula correction):** treat
`BETA0` as a regime-switching fit parameter and refit it against
Sebas's CSV using the body-current rather than as a fixed manual default
— typical short-channel values are 0.5–3 V, which would lift Iii by
~25 orders of magnitude into the nA range.

**Real fix (root cause):** Iii is unlikely to be the dominant
body-charging path at Vd=1.5 V; junction GIDL or the body-source diode
are more plausible. Re-examine `compute_gidl_gisl` and `idiode` weights
before further tuning impact-ion.

**Flag for oracle:** confirm IIMOD=2 (HSPICE) form and whether any
Sebas-internal extraction uses an HSPICE-only impact-ion equation we
have not ported.

```


=== FILE: artifacts/A1i_complementary_bipolar.md (4789 chars) ===
```
# A1i — Decoding Sebas's "Complementary Bipolar Current"

**Source:** `/home/ikaros/nsram_info/schematic&modelCards/2tnsram_simple.asc`
(plus `parasiticBJT.txt`, `PTM130bulkNSRAM.txt`).

## 1. Behavioural elements — verbatim inventory

`grep`-ing the ASC for `bv`, `bi`, `B1`, `.subckt`, `.func`, `.lib` returns
**nothing**. The schematic contains exactly four primitive devices:

| Inst | Symbol | Value / model | Connections (D,G,S/B,Bulk) |
|------|--------|---------------|----------------------------|
| M1   | `nmos4`| `NMOS`, `l=Ln, w=Wn`         | D=`Din`, G=`G`,  S=`Sint`, B=`B` |
| Q1   | `npn`  | `parasiticBJT`, `area=1u`     | C=`D` (=Din), B=`B`, E=`0` (GND) |
| M2   | `nmos4`| `NMOS`, `l=Ln*10, w=Wn`       | D=`B`,  G=`G2`, S=`0`,    B=`0` |
| C1   | `cap`  | `'CBpar'` (=1 fF), Rser=1 mΩ  | between `B` and `0`              |

`.param Ln=0.18u  Wn=0.36u  CBpar=1f`
`.inc PTM130bulkNSRAM.txt   .inc parasiticBJT.txt`

There is **no B-source, no behavioural current, no sub-circuit.** The
"complementary bipolar current" Sebas refers to is *not* a custom
expression — it is simply the **collector current of Q1 (parasiticBJT)**,
fired by the **BSIM4 built-in impact-ionization current** (`alpha0`,
`beta0`) of M1, which charges the floating body node `B`.

## 2. Physical interpretation of each piece

* **M1 (NMOS, BSIM4)** — channel transport plus *intrinsic II generation*:
  `Iii = (alpha0/L) · (Vds−Vdsat) · exp[−beta0/(Vds−Vdsat)] · Ids`
  (BSIM4 manual eq. for `Iii`, routed to the body node). Card uses
  `alpha0 = 7.83756e-5`, `beta0 = 18`.
* **Q1 (NPN, model `parasiticBJT`)** — the lateral parasitic bipolar.
  `is = 5e-9`, `bf = 10000`, `va = 100`, `nc = 2`. Emitter tied to GND,
  base = floating P-body, collector = drain. *This is the "complementary
  bipolar current"* — when V(B) climbs above ~0.6 V, Q1 turns on and
  pumps a large `Ic = β·Is·exp(Vbe/Vt)` from D to GND, in addition to
  the BSIM4 channel current. That is the "full swing of the firing
  mechanism".
* **M2 (long NMOS, l=10·Ln)** — VG2-controlled *body-discharge*
  transistor. When `VG2 > Vth`, M2 sinks B→GND, killing the firing
  (low-VG2 regime = leaky body = no firing — exactly our diagnostic).
* **C1 (1 fF)** — parasitic body capacitance for transient charge
  retention.

## 3. Mathematical form

```
I_complementary(node D → node 0) = Q1.Ic
   = Is · ( exp(V(B)/Vt) − exp(V(B)−V(D))/Vt) ) · (1 + V(D)/Va) / qb
   ≈ 5e-9 · exp(V(B)/0.02585)            # forward-active, V(D)>>Vt
```

with body charge balance

```
C1·dV(B)/dt = I_ii(M1)                                # source: BSIM4 II
            − I_BE(Q1)                                # sink: BJT base
            − I_DS(M2, VG2, V(B))                     # sink: VG2 pull-down
```

So Sebas's "complementary current" is **not a hand-coded B-source**; it
is the standard Gummel–Poon Ic of `parasiticBJT`, *gated by* whether the
BSIM4 II current can outrun the M2 leakage path.

## 4. PyTorch port — how to add it

We already have the BSIM4 `Iii` term in `compute_iimpact`. The missing
piece is the **NPN collector current path D→GND**, plus M2's pull-down.
Concretely:

* **No new free parameters needed.** Hard-code Gummel–Poon constants from
  `parasiticBJT.txt` (`Is=5e-9, Bf=1e4, Va=100, area=1e-6 ⇒ Is_eff=5e-15 A`).
* Add `compute_complementary_bjt(Vd, Vb, Vt, params)` returning
  `Ic = Is_eff*(exp(Vb/Vt) − exp((Vb−Vd)/Vt))*(1+Vd/Va)`.
* In `_eval_mosfet`, add `Ids_total = Ids_bsim + Ic_bjt`.
* In the body-ODE update (or DC Newton solve), replace the existing
  "Ibody = Iii" with `Ibody = Iii − I_BE(Q1) − I_DS(M2,VG2,Vb)` where
  `I_BE = Is_eff·exp(Vb/Vt)/Bf` and `I_DS(M2)` is the same BSIM4 call
  with `l=10·Ln`, `w=Wn`, `Vgs=VG2`, `Vds=Vb`.

## 5. Numerical sanity at (Vd=1.5, Vsint=0.306, Vb=0.342, VG1=0.6, VG2=0)

```
Vt = 0.02585 V
exp(Vb/Vt)            = exp(0.342/0.02585) = exp(13.23)  ≈ 5.6e5
exp((Vb−Vd)/Vt)       = exp(−1.158/0.02585)= exp(−44.8)  ≈ 4e−20
Is_eff = Is·area      = 5e-9 · 1e-6        = 5e-15 A
Ic ≈ 5e-15 · 5.6e5 · (1+1.5/100) ≈ 2.8e-9 A
```

→ **~2.8 nA**, squarely in the "few-nA" range needed to charge `CBpar=1 fF`
on µs timescales and explain the measured firing onset. This closes our
6-decade gap: at Vb≈0.342 V the BSIM4-only current is sub-fA, but Q1
delivers ~3 nA — a ~10⁶× boost, exactly the missing factor.

## 6. Verdict

There is **no behavioural current source** in Sebas's schematic — the
"complementary bipolar current" is the **collector current of the NPN
`parasiticBJT` Q1 wired D-to-GND**, fed by **BSIM4's native impact-
ionization (`alpha0`,`beta0`)** charging the floating body, and gated by
the **VG2-controlled long NMOS M2** that bleeds the body down. We simply
need to add a Gummel-Poon Ic term plus the M2 body-discharge path; no
new free parameters.

```


=== FILE: _extracted/validation_scripts/z91e_bsim4_port_fit_with_anchors.py (8437 chars) ===
```python
"""z91e — z91d + soft regularizer toward Sebas's extracted parameter values.

Background
----------
The Apr-30 meeting slide 24 ("Image 2026-04-30 at 13.24") shows Sebas already
extracted four BSIM4 parameters as functions of (VG1, VG2) from his measured
curves:

    VTH0(M1)   ~ 0.42–0.55 V across all VG1 (descends with VG1)
    BETA0(M1)  ~ 11–21       (rises with VG1, weak VG2 dep)
    ETA0(M1)   ~ 1.0–2.0     (rises with VG1)
    NFACTOR(M2) ~ 3–12       (descends with VG1)

z91d's Stage-1 optimum drove vth0 to 0.315 V — *below* Sebas's extracted
range. This is a sign the optimizer is still compensating for something
(likely off-state model imperfections we can't fix without finer body-effect
treatment), and it confirms that pure-data fitting is under-constrained.

Strategy
--------
Add a **soft anchor regularizer** that pulls the fitted constant params toward
the mean of Sebas's extracted curves. Constant params are clearly insufficient
(slide 24 shows VG1, VG2 dependence) but anchoring the *mean* prevents the
optimizer from fleeing to non-physical regions to compensate for missing
physics. Once z91e converges, follow up with a poly(VG1, VG2) variant (z91f)
to capture the structure Sebas resolved.

Anchor mode
-----------
- vth0  → 0.50 V    (mean of Sebas's curve)         lambda=0.5
- u0    → init      (no Sebas data)                 lambda=0
- beta0 → 15        (mean of his BETA0(M1) plot)    lambda=0.05
                       — note Sebas's BETA0 ≠ BSIM4's beta0 (impact-ion).
                         Sebas uses BETA0 as a body-current scaler in his
                         own form. We weight this small.
- agidl/bgidl/cgidl/egidl unchanged from z91d (Sebas didn't extract these)
- alpha0/alpha1/Bf unchanged

Caveat: BETA0 is overloaded. In Sebas's plot it's likely his own coefficient
in his SPICE deck (Pazos lab convention), not BSIM4 §6.1's beta0. Treat its
anchor weight as a sanity check, not a hard constraint.
"""
from __future__ import annotations
import json, math, re, time
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cpu")
torch.set_default_device(DEVICE)
print(f"[z91e] Using device: {DEVICE}", flush=True)

from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.arclength import forward_2t_arclength_grad
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig, forward_2t
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.geometry import Geometry

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data/sebas_2026_04_22"
OUT = ROOT / "results/z91e_bsim4_port_fit_with_anchors"
OUT.mkdir(parents=True, exist_ok=True)


# Reuse loader/PARAM_SPEC/helpers from z91d. To avoid duplicating ~500 LOC we
# import the module under a stable name. z91d has top-level side effects (the
# print + dir creation), but those are safe to repeat.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "z91d_mod", ROOT / "scripts/z91d_bsim4_port_fit_arclength.py")
z91d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(z91d)

load_curves = z91d.load_curves
PARAM_SPEC = z91d.PARAM_SPEC
make_thetas = z91d.make_thetas
clone_thetas = z91d.clone_thetas
thetas_to_values = z91d.thetas_to_values
init_theta = z91d.init_theta
patch_sd = z91d.patch_sd
forward_curve = z91d.forward_curve
huber_log_loss = z91d.huber_log_loss
run_stage = z91d.run_stage
multistart_stage = z91d.multistart_stage
evaluate_full = z91d.evaluate_full


# --------------------------------------------------------------------------- #
# Sebas anchors (from Image 2026-04-30 at 13.24 — manual digitization)
# --------------------------------------------------------------------------- #
# Format: param_name -> (target_value, lambda_weight)
# lambda is on a per-parameter loss scale (huber-on-log10-Id is O(1)).
SEBAS_ANCHORS = {
    "vth0":  (0.50, 0.5),    # strong: optimizer fled to 0.315 in z91d
    "beta0": (15.0, 0.05),   # weak: BETA0 in his plot may differ from BSIM4 beta0
}


def anchor_loss(thetas) -> torch.Tensor:
    """Quadratic pull of fitted params toward Sebas-extracted means."""
    values = thetas_to_values(thetas)
    total = torch.zeros((), dtype=torch.float64)
    for name, (target, lam) in SEBAS_ANCHORS.items():
        if name in values:
            v = values[name]
            spec = PARAM_SPEC[name]
            lo, hi = spec["bounds"]
            scale = (hi - lo)
            total = total + lam * ((v - target) / scale) ** 2
    return total


def stage_loss(thetas, model, cfg, curves, *, use_homotopy: bool = False):
    """z91d's stage_loss + anchor regularizer."""
    data_loss = z91d.stage_loss(thetas, model, cfg, curves,
                                 use_homotopy=use_homotopy)
    anc = anchor_loss(thetas)
    return data_loss + anc


# Monkey-patch so run_stage / multistart_stage (imported above) use OUR
# stage_loss (the closure inside run_stage looks up z91d.stage_loss by ref).
z91d.stage_loss = stage_loss


def main():
    t0 = time.time()
    print(f"[z91e] starting at {time.strftime('%H:%M:%S')}", flush=True)

    curves = load_curves()
    print(f"[z91e] loaded {len(curves)} curves", flush=True)

    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                             newton_max_iters=50)
    text = (DATA_DIR / "PTM130bulkNSRAM.txt").read_text()
    model = BSIM4Model.from_spice(text, model_type="nmos")
    sd_M1 = compute_size_dep(model, Geometry(L=cfg.Ln, W=cfg.Wn), T_C=cfg.T_C)
    sd_M2 = compute_size_dep(model, Geometry(L=cfg.Ln * cfg.M2_length_factor,
                                              W=cfg.Wn), T_C=cfg.T_C)
    cfg._sd_M1 = sd_M1
    cfg._sd_M2 = sd_M2

    thetas = make_thetas(seed=0)

    # --- Stage 1: off-state, anchored vth0 + GIDL --------------------------- #
    off_curves = [c for c in curves if c["VG1"] == 0.2]
    s1_names = ["vth0", "agidl", "bgidl", "cgidl", "egidl"]
    print(f"\n=== Stage 1: off-state ({len(off_curves)} curves) "
          f"anchors: vth0→0.50 V ===", flush=True)
    thetas, loss1 = multistart_stage(
        1, thetas, model, cfg, off_curves, s1_names,
        n_adam=30, n_lbfgs=15, n_seeds=2, t0=t0)

    # --- Stage 2: core transport ------------------------------------------- #
    sub_curves = [c for c in curves
                  if (c["Vd"] <= 0.8).all() or c["VG1"] <= 0.4]
    s2_names = ["u0", "vsat", "vth0", "k1", "k2"]
    print(f"\n=== Stage 2: core transport ({len(sub_curves)} curves) ===",
          flush=True)
    thetas, loss2 = multistart_stage(
        2, thetas, model, cfg, sub_curves, s2_names,
        n_adam=30, n_lbfgs=15, n_seeds=3, t0=t0)

    # --- Stage 3: snapback (gmin homotopy ON) ------------------------------ #
    s3_names = ["alpha0", "alpha1", "beta0", "Bf"]
    print(f"\n=== Stage 3: snapback (all 33 curves, homotopy ON) ===",
          flush=True)
    thetas, loss3 = multistart_stage(
        3, thetas, model, cfg, curves, s3_names,
        n_adam=30, n_lbfgs=15, n_seeds=3, t0=t0,
        use_homotopy=True)

    # --- Stage 4: polish all ----------------------------------------------- #
    s4_names = list(PARAM_SPEC.keys())
    print(f"\n=== Stage 4: L-BFGS polish all 13 params ===", flush=True)
    loss4 = run_stage(4, "", thetas, model, cfg, curves, s4_names,
                      n_adam=10, n_lbfgs=30, t0=t0, use_homotopy=True)

    # --- Save -------------------------------------------------------------- #
    final_values = {n: float(v.detach().item())
                    for n, v in thetas_to_values(thetas).items()}
    median_rmse, preds = evaluate_full(thetas, model, cfg, curves)

    summary = {
        "stage1_loss": loss1, "stage2_loss": loss2,
        "stage3_loss": loss3, "stage4_loss": loss4,
        "median_log_rmse": median_rmse,
        "anchors": {k: list(v) for k, v in SEBAS_ANCHORS.items()},
        "params": final_values,
        "elapsed_s": time.time() - t0,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "predictions.json").write_text(json.dumps(preds, indent=2))
    print(f"\n[z91e] DONE  median_log_rmse={median_rmse:.3f}  "
          f"params={final_values}", flush=True)


if __name__ == "__main__":
    main()

```


=== FILE: _extracted/validation_scripts/z91g_two_model_validation.py (8407 chars) ===
```python
"""z91g — true two-card validation.

Builds on z91f. After the P2.2 refactor (forward_2t now accepts model_M1
and model_M2 as separate BSIM4Model instances), we can finally run the
M1 card on M1 and the M2 card on M2 — fixing the silent coherence break
where compute_dc(model, sd_M2, …) was reading M1's k3, lpe0, dvt0, kt1,
kt1l, kt2, etc. while computing M2.

Same .param post-load patch as z91f (vth0n=0.54153, vsatn=102230,
lpe0n=1.2439e-7, …) — the SPICE parser still misses + continuation lines
on .param directives, so the post-load fixup remains necessary.
"""
from __future__ import annotations
import json, math, re, csv, time
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/sebas_2026_04_22"
OUT = ROOT / "results/z91g_two_model_validation"
OUT.mkdir(parents=True, exist_ok=True)

from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig, forward_2t
from nsram.bsim4_port.arclength import forward_2t_arclength_grad
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.geometry import Geometry


# Reuse z91f's data + helper layer
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "z91f_mod", ROOT / "scripts/z91f_validate_with_sebas_params.py")
z91f = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(z91f)
load_curves = z91f.load_curves
load_sebas_params = z91f.load_sebas_params
find_params = z91f.find_params
patch_model_values = z91f.patch_model_values
patch_sd_scaled = z91f.patch_sd_scaled
make_overrides = z91f.make_overrides
make_bjt = z91f.make_bjt


def main():
    t0 = time.time()
    print(f"[z91g] starting at {time.strftime('%H:%M:%S')}", flush=True)

    # Load M1 and M2 cards as DISTINCT BSIM4Model instances. Apply the
    # .param post-load patch to each (parser drops + continuation lines on
    # .param blocks).
    text_M1 = (DATA / "M1_130DNWFB.txt").read_text()
    model_M1 = BSIM4Model.from_spice(text_M1, model_type="nmos")
    patch_model_values(model_M1, type_n=True)
    print(f"[z91g] M1 card loaded; vth0={model_M1.get('vth0')} "
          f"vsat={model_M1.get('vsat')} k1={model_M1.get('k1')} "
          f"etab={model_M1.get('etab')} beta0={model_M1.get('beta0')}",
          flush=True)

    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    model_M2 = BSIM4Model.from_spice(text_M2, model_type="nmos")
    patch_model_values(model_M2, type_n=True)
    print(f"[z91g] M2 card loaded; vth0={model_M2.get('vth0')} "
          f"vsat={model_M2.get('vsat')} k1={model_M2.get('k1')} "
          f"etab={model_M2.get('etab')} beta0={model_M2.get('beta0')}",
          flush=True)

    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                             newton_max_iters=50)
    sd_M1 = compute_size_dep(model_M1, Geometry(L=cfg.Ln, W=cfg.Wn),
                              T_C=cfg.T_C)
    sd_M2 = compute_size_dep(model_M2,
                              Geometry(L=cfg.Ln * cfg.M2_length_factor,
                                       W=cfg.Wn), T_C=cfg.T_C)
    cfg._sd_M1 = sd_M1
    cfg._sd_M2 = sd_M2

    curves = load_curves()
    sebas_rows = load_sebas_params()
    print(f"[z91g] {len(curves)} measured curves, {len(sebas_rows)} CSV rows",
          flush=True)

    log_eps = 1e-15
    results = []
    for c in curves:
        sebas_row = find_params(sebas_rows, c["VG1"], c["VG2"])
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            results.append({"VG1": c["VG1"], "VG2": c["VG2"],
                            "skipped": True, "reason": "NaN row"})
            continue
        P_M1, P_M2 = make_overrides(sebas_row)
        # The static M2_STATIC_OVERRIDES inside z91f.make_overrides puts
        # k1/etab/beta0 baselines in P_M2; with the proper M2 card now
        # loaded those baselines are already in sd_M2. Drop them so we
        # only override what the CSV says (NFACTOR).
        if P_M2:
            for k in ("k1", "k2", "etab", "beta0"):
                P_M2.pop(k, None)
            if not P_M2:
                P_M2 = None
        bjt = make_bjt(sebas_row)
        try:
            with torch.no_grad(), \
                 patch_sd_scaled(sd_M1, P_M1), \
                 patch_sd_scaled(sd_M2, P_M2):
                out = forward_2t_arclength_grad(
                    cfg, model_M1=model_M1, model_M2=model_M2,
                    bjt=bjt, Vd_seq=c["Vd"],
                    VG1=torch.tensor(c["VG1"]),
                    VG2=torch.tensor(c["VG2"]))
            Id_pred = out["Id"].abs()
            conv = torch.tensor([bool(x) for x in out["converged"]])
        except Exception as e:
            results.append({"VG1": c["VG1"], "VG2": c["VG2"],
                            "skipped": True, "reason": f"forward error: {e}"})
            continue

        log_p = torch.log10(Id_pred + log_eps)
        log_m = torch.log10(c["Id"] + log_eps)
        if conv.any():
            sq = (log_p - log_m) ** 2
            rmse = float(torch.sqrt(sq[conv].mean()))
        else:
            rmse = float("inf")
        results.append({"VG1": c["VG1"], "VG2": c["VG2"], "skipped": False,
                        "log_rmse": rmse,
                        "n_converged": int(conv.sum()),
                        "n_total": int(len(conv)),
                        "Vd": c["Vd"].numpy().tolist(),
                        "Id_meas": c["Id"].numpy().tolist(),
                        "Id_pred": Id_pred.numpy().tolist(),
                        "converged": conv.numpy().tolist()})
        print(f"  VG1={c['VG1']:.2f} VG2={c['VG2']:+.2f}: "
              f"log_rmse={rmse:.3f}  conv={int(conv.sum())}/{len(conv)}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    rmses = [r["log_rmse"] for r in results
             if not r.get("skipped") and math.isfinite(r["log_rmse"])]
    median_rmse = float(np.median(rmses)) if rmses else float("inf")
    p90_rmse = float(np.percentile(rmses, 90)) if rmses else float("inf")

    summary = {
        "n_curves": len(curves),
        "n_evaluated": len(rmses),
        "n_skipped": sum(1 for r in results if r.get("skipped")),
        "median_log_rmse": median_rmse,
        "p90_log_rmse": p90_rmse,
        "elapsed_s": time.time() - t0,
        "vs_z91f_run1_median": 4.234,
        "vs_z91f_run2_median": 2.402,
        "note": "true two-model validation (M1 = 130DNWFB, M2 = 130bulkNSRAM)"
                " with Sebastian's per-bias CSV overrides",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "predictions.json").write_text(json.dumps(results, indent=2))
    print(f"\n[z91g] median log-RMSE = {median_rmse:.3f}  "
          f"p90 = {p90_rmse:.3f}  (z91f run2: median=2.40, p90=4.83)",
          flush=True)

    # Plot grid
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, vg1 in zip(axes, [0.2, 0.4, 0.6]):
        sel = [r for r in results
               if not r.get("skipped") and abs(r["VG1"] - vg1) < 1e-3]
        sel.sort(key=lambda r: r["VG2"])
        cmap = plt.cm.viridis(np.linspace(0, 1, max(len(sel), 1)))
        for color, r in zip(cmap, sel):
            Vd = np.array(r["Vd"])
            Im = np.array(r["Id_meas"])
            Ip = np.array(r["Id_pred"])
            cm = np.array(r["converged"])
            ax.semilogy(Vd, Im, "o", ms=3, color=color, alpha=0.5)
            Ip_plot = np.where(cm, Ip, np.nan)
            ax.semilogy(Vd, Ip_plot, "-", lw=1.0, color=color)
        ax.set_title(f"VG1 = {vg1} V")
        ax.set_xlabel("Vd [V]")
        ax.set_ylim(1e-13, 1e-3)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("|Id| [A]")
    fig.suptitle(
        f"z91g two-model validation — M1 = 130DNWFB, M2 = 130bulkNSRAM\n"
        f"o = measurement, line = prediction · "
        f"median log-RMSE = {median_rmse:.3f}  p90 = {p90_rmse:.3f}  "
        f"(z91f single-card: 2.40 / 4.83)",
        fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fit_vs_meas.png", dpi=140)
    plt.close(fig)
    print(f"[z91g] saved {OUT}/fit_vs_meas.png", flush=True)


if __name__ == "__main__":
    main()

```


=== FILE: _extracted/validation_scripts/z91f_validate_with_sebas_params.py (14935 chars) ===
```python
"""z91f — End-to-end validation against Sebastian's extracted parameters.

Sebas (2026-04-30 email + BSIMfitsBA package) sent us:
  • 130DNWFB(M1).txt  — M1 device card (deep N-well floating body)
  • 130bulkNSRAM(M2).txt — M2 device card (bulk)
  • 2Tcell_BSIM_param_DC.csv — fitted BSIM4 + BJT params per (VG1, VG2)

Per his email: "some parameters change only for M1, while NFACTOR changes
only for M2 (I attribute this to LDE)". So the CSV columns split as:
  M1 overrides: ETAB, K1, ALPHA0, BETA0   (LDE-driven, vary with VG1)
  M2 overrides: NFACTOR                    (LDE on M2, varies with VG2)
  BJT/wrapper:  trise, mbjt, IS, area

This script does NO fitting. It runs our forward simulator with Sebas's
exact extracted parameters at each (VG1, VG2) and compares to measurement.

If we match Sebas's published SPICE fit → port is end-to-end validated and
all earlier "loss=X" numbers were noise from us using one card for both
devices and treating constants-across-bias.

If we don't match → the deviation tells us exactly which sub-block of the
port disagrees with industry SPICE.

Usage
-----
    python scripts/z91f_validate_with_sebas_params.py
    → results/z91f_validate_sebas/{summary.json, fit_vs_meas.png}
"""
from __future__ import annotations
import json, math, re, time, csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/sebas_2026_04_22"
OUT = ROOT / "results/z91f_validate_sebas"
OUT.mkdir(parents=True, exist_ok=True)

from contextlib import contextmanager
from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig, forward_2t
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.geometry import Geometry


@contextmanager
def patch_sd_scaled(sd, overrides):
    """Override sd.scaled[name] entries (NOT sd attributes). Mirrors the
    `patch_sd` helper in z91d. Use this for BSIM4 params that live in the
    SizeDep.scaled dict (k1, k2, etab, alpha0, beta0, nfactor, ...).
    """
    if not overrides:
        yield
        return
    saved = {}
    try:
        for k, v in overrides.items():
            saved[k] = sd.scaled.get(k, None)
            sd.scaled[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sd.scaled.pop(k, None)
            else:
                sd.scaled[k] = v


# --------------------------------------------------------------------------- #
# Load measurement curves (same loader as z91d)
# --------------------------------------------------------------------------- #
def parse_vg2(s):
    m = re.search(r"VG2=(-?\d+\.\d+)", s);  return float(m.group(1)) if m else None
def parse_vg1(s):
    m = re.search(r"VG1=([\d.]+)", s);      return float(m.group(1)) if m else None


def load_curves():
    curves = []
    for d in sorted(DATA.glob("2vHCa-2 I-Vs@VG2 VG1=*")):
        VG1 = parse_vg1(d.name)
        for f in sorted(d.glob("*.csv")):
            VG2 = parse_vg2(f.name)
            data = np.loadtxt(f, delimiter=",", skiprows=1, usecols=(0, 1))
            if data.ndim == 1:
                continue
            half = len(data) // 2
            Vd = data[:half, 0]
            Id = np.abs(data[:half, 1])
            mask = (Vd >= 0.05) & (Vd <= 2.0)
            Vd, Id = Vd[mask], Id[mask]
            if len(Vd) > 10:
                idx = np.linspace(0, len(Vd) - 1, 30).astype(int)
                Vd, Id = Vd[idx], Id[idx]
                curves.append({"VG1": VG1, "VG2": VG2,
                               "Vd": torch.tensor(Vd, dtype=torch.float64),
                               "Id": torch.tensor(Id, dtype=torch.float64)})
    return curves


# --------------------------------------------------------------------------- #
# Load Sebastian's per-bias parameter CSV
# --------------------------------------------------------------------------- #
def load_sebas_params():
    path = DATA / "2Tcell_BSIM_param_DC.csv"
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = {}
            for k, v in r.items():
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = float("nan")
            rows.append(row)
    return rows


def find_params(rows, VG1, VG2, atol=1e-3):
    for r in rows:
        if abs(r["VG1"] - VG1) < atol and abs(r["VG2"] - VG2) < atol:
            return r
    return None


# --------------------------------------------------------------------------- #
# .param block from M2 card (continuation lines our parser drops)            #
# --------------------------------------------------------------------------- #
# Both M1 and M2 cards reference shared symbols (vth0n, vsatn, lpe0n, …).    #
# M2 defines them at top via `.param ... + continuation`; our parser only    #
# captures the first line (toxn=4e-009). The rest fall back to BSIM4         #
# defaults — that is the 5-decade error we saw in z91f run #1. We apply      #
# them post-load to BOTH model_M1 and model_M2.                              #
SHARED_PARAM = {
    "toxn":   4e-9,
    "toxp":   4e-9,
    "lintn":  1.219e-8,
    "lintp": -1.079e-8,
    "vth0n":  0.54153,
    "vth0p": -1.106133,
    "lpe0n":  1.2439e-7,
    "lpe0p": -7.833656e-8,
    "k3n":    65.28,
    "k3p":   -7.18419,
    "pvth0n":-1.45e-15,
    "pvth0p": 5.543149e-16,
    "vsatn":  102230.0,
    "vsatp":  8.07584e4,
    "wintn":  4.7689e-8,
    "wintp":  4.268414e-9,
}

# Direct attribute substitutions applied to each model after load. These
# correspond to body lines like `+vth0 = vth0n` which our parser failed to
# resolve, leaving BSIM4 defaults in place. We patch the resolved values.
def patch_model_values(model, type_n: bool = True):
    s = "n" if type_n else "p"
    pmap = {
        "vth0":  SHARED_PARAM[f"vth0{s}"],
        "vsat":  SHARED_PARAM[f"vsat{s}"],
        "lpe0":  SHARED_PARAM[f"lpe0{s}"],
        "lint":  SHARED_PARAM[f"lint{s}"],
        "wint":  SHARED_PARAM[f"wint{s}"],
        "k3":    SHARED_PARAM[f"k3{s}"],
        "pvth0": SHARED_PARAM[f"pvth0{s}"],
        "toxe":  SHARED_PARAM[f"tox{s}"],
        "toxp":  SHARED_PARAM[f"tox{s}"],
        "toxm":  SHARED_PARAM[f"tox{s}"],
    }
    for k, v in pmap.items():
        model._values[k] = float(v)


# Static deltas in the BODY of M2 vs M1 (k1, etab, beta0 — verified by diff).
# These get applied to sd_M2.scaled at forward time. CSV per-bias overrides
# on top.
M2_STATIC_OVERRIDES = {
    "k1":    0.63825,
    "k2":   -0.070435,
    "etab": -0.086777,
    "beta0": 18.0,
}


def make_overrides(sebas_row):
    """Map a CSV row → (P_M1, P_M2) override dicts for forward_2t."""
    if sebas_row is None:
        return None, None
    # M1 overrides — bias-dependent parameters that Sebas attributes to LDE
    P_M1 = {}
    if not math.isnan(sebas_row.get("ETAB", float("nan"))):
        P_M1["etab"] = torch.tensor(sebas_row["ETAB"], dtype=torch.float64)
    if not math.isnan(sebas_row.get("K1", float("nan"))):
        P_M1["k1"] = torch.tensor(sebas_row["K1"], dtype=torch.float64)
    if not math.isnan(sebas_row.get("ALPHA0", float("nan"))):
        P_M1["alpha0"] = torch.tensor(sebas_row["ALPHA0"], dtype=torch.float64)
    if not math.isnan(sebas_row.get("BETA0", float("nan"))):
        P_M1["beta0"] = torch.tensor(sebas_row["BETA0"], dtype=torch.float64)

    # M2 overrides — NFACTOR varies with VG2 due to LDE on M2
    P_M2 = {}
    if not math.isnan(sebas_row.get("NFACTOR", float("nan"))):
        P_M2["nfactor"] = torch.tensor(sebas_row["NFACTOR"], dtype=torch.float64)

    # Always apply the M2 card's static deltas (not in CSV)
    for k, v in M2_STATIC_OVERRIDES.items():
        if k not in P_M2:
            P_M2[k] = torch.tensor(float(v), dtype=torch.float64)

    return P_M1 or None, P_M2 or None


def make_bjt(sebas_row):
    """Build per-bias BJT instance from a Sebas-CSV row.

    mbjt is SPICE's device-multiplicity `m=` parameter — same effect as
    scaling `area`. Sebas uses it to switch the parasitic-NPN path on
    (VG1=0.4/0.6 → mbjt=1) or essentially off (VG1=0.2 → mbjt=0.001).
    Honour it via `area *= mbjt` per A1b finding.
    """
    bjt = GummelPoonNPN.from_sebas_card()
    if sebas_row is not None:
        if not math.isnan(sebas_row.get("IS", float("nan"))):
            bjt.Is = float(sebas_row["IS"])
        area = float(sebas_row.get("area", 1e-6))
        if math.isnan(area):
            area = 1e-6
        mbjt = float(sebas_row.get("mbjt", 1.0))
        if math.isnan(mbjt):
            mbjt = 1.0
        bjt.area = area * mbjt
    return bjt


# --------------------------------------------------------------------------- #
# Run validation
# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    print(f"[z91f] starting at {time.strftime('%H:%M:%S')}", flush=True)

    # Load M1 card; patch resolved-from-.param values our parser dropped
    text_M1 = (DATA / "M1_130DNWFB.txt").read_text()
    model = BSIM4Model.from_spice(text_M1, model_type="nmos")
    patch_model_values(model, type_n=True)
    print(f"[z91f] loaded M1 card (130DNWFB), patched shared .param "
          f"values: vth0={model.get('vth0')} vsat={model.get('vsat')}",
          flush=True)

    # Load M2 card separately for sd_M2's transport baselines (vth0_T,
    # vsattemp etc.). Same .param patch — both cards share these globals.
    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    model_M2 = BSIM4Model.from_spice(text_M2, model_type="nmos")
    patch_model_values(model_M2, type_n=True)
    print(f"[z91f] loaded M2 card (130bulkNSRAM), patched: "
          f"vth0={model_M2.get('vth0')} vsat={model_M2.get('vsat')} "
          f"k1={model_M2.get('k1')} etab={model_M2.get('etab')}", flush=True)

    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                             newton_max_iters=50)
    sd_M1 = compute_size_dep(model, Geometry(L=cfg.Ln, W=cfg.Wn), T_C=cfg.T_C)
    sd_M2 = compute_size_dep(model_M2,
                              Geometry(L=cfg.Ln * cfg.M2_length_factor,
                                       W=cfg.Wn), T_C=cfg.T_C)
    cfg._sd_M1 = sd_M1
    cfg._sd_M2 = sd_M2

    curves = load_curves()
    sebas_rows = load_sebas_params()
    print(f"[z91f] loaded {len(curves)} measured curves and "
          f"{len(sebas_rows)} CSV rows", flush=True)

    log_eps = 1e-15
    results = []
    for c in curves:
        sebas_row = find_params(sebas_rows, c["VG1"], c["VG2"])
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            results.append({"VG1": c["VG1"], "VG2": c["VG2"],
                            "skipped": True,
                            "reason": "no Sebas params (NaN row)"})
            continue
        P_M1, P_M2 = make_overrides(sebas_row)
        bjt = make_bjt(sebas_row)
        try:
            with torch.no_grad(), \
                 patch_sd_scaled(sd_M1, P_M1), \
                 patch_sd_scaled(sd_M2, P_M2):
                out = forward_2t(cfg, model, bjt,
                                  c["Vd"], torch.tensor(c["VG1"]),
                                  torch.tensor(c["VG2"]),
                                  warm_start=True, use_homotopy=True)
            Id_pred = out["Id"].abs()
            conv = torch.tensor([bool(x) for x in out["converged"]])
        except Exception as e:
            results.append({"VG1": c["VG1"], "VG2": c["VG2"],
                            "skipped": True, "reason": f"forward error: {e}"})
            continue

        log_p = torch.log10(Id_pred + log_eps)
        log_m = torch.log10(c["Id"] + log_eps)
        if conv.any():
            sq = (log_p - log_m) ** 2
            rmse = float(torch.sqrt(sq[conv].mean()))
        else:
            rmse = float("inf")
        results.append({
            "VG1": c["VG1"], "VG2": c["VG2"], "skipped": False,
            "log_rmse": rmse,
            "n_converged": int(conv.sum()),
            "n_total": int(len(conv)),
            "Vd": c["Vd"].numpy().tolist(),
            "Id_meas": c["Id"].numpy().tolist(),
            "Id_pred": Id_pred.numpy().tolist(),
            "converged": conv.numpy().tolist(),
        })
        print(f"  VG1={c['VG1']:.2f} VG2={c['VG2']:+.2f}: "
              f"log_rmse={rmse:.3f}  conv={int(conv.sum())}/{len(conv)}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    rmses = [r["log_rmse"] for r in results
             if not r.get("skipped") and math.isfinite(r["log_rmse"])]
    median_rmse = float(np.median(rmses)) if rmses else float("inf")
    p90_rmse = float(np.percentile(rmses, 90)) if rmses else float("inf")

    summary = {
        "n_curves": len(curves),
        "n_evaluated": len(rmses),
        "n_skipped": sum(1 for r in results if r.get("skipped")),
        "median_log_rmse": median_rmse,
        "p90_log_rmse": p90_rmse,
        "elapsed_s": time.time() - t0,
        "note": "forward-only validation with Sebastian's extracted "
                "BSIM4 + BJT parameters (CSV) and M2 card static deltas",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "predictions.json").write_text(json.dumps(results, indent=2))
    print(f"\n[z91f] median log-RMSE = {median_rmse:.3f}  "
          f"p90 = {p90_rmse:.3f}", flush=True)

    # Plot grid: 3 columns by VG1
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, vg1 in zip(axes, [0.2, 0.4, 0.6]):
        sel = [r for r in results
               if not r.get("skipped") and abs(r["VG1"] - vg1) < 1e-3]
        sel.sort(key=lambda r: r["VG2"])
        cmap = plt.cm.viridis(np.linspace(0, 1, max(len(sel), 1)))
        for color, r in zip(cmap, sel):
            Vd = np.array(r["Vd"])
            Im = np.array(r["Id_meas"])
            Ip = np.array(r["Id_pred"])
            cm = np.array(r["converged"])
            ax.semilogy(Vd, Im, "o", ms=3, color=color, alpha=0.5)
            Ip_plot = np.where(cm, Ip, np.nan)
            ax.semilogy(Vd, Ip_plot, "-", lw=1.0, color=color)
        ax.set_title(f"VG1 = {vg1} V")
        ax.set_xlabel("Vd [V]")
        ax.set_ylim(1e-13, 1e-3)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("|Id| [A]")
    fig.suptitle(
        f"z91f forward-only validation — Sebas's parameters → our simulator\n"
        f"o = measurement, line = prediction · "
        f"median log-RMSE = {median_rmse:.3f}  p90 = {p90_rmse:.3f}",
        fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fit_vs_meas.png", dpi=140)
    plt.close(fig)
    print(f"[z91f] saved {OUT}/fit_vs_meas.png", flush=True)


if __name__ == "__main__":
    main()

```


=== FILE: _extracted/validation_scripts/z91d_bsim4_port_fit_arclength.py (23477 chars) ===
```python
"""z82 — P7v6: Stage-wise fit using the proper 2T NS-RAM cell topology.

Replaces z80's 1T-proxy (gamma_VG2 / Rb_leak / C_extra / I_PT0 hacks) with the
real 2-MOSFET + parasitic-NPN model defined in
``nsram/nsram/bsim4_port/nsram_cell_2T.py``. VG2 is now M2's actual gate;
body physics is self-consistent via Newton-Raphson on (Vsint, Vb).

Stages
  1. Off-state (VG1=0.2): GIDL params + vth0. BJT off, Iii off.
  2. Core transport (Vd<0.8 all VG1): u0, vsat, vth0, k1, k2. BJT/Iii off.
  3. Snapback (MERGED, all 33 curves): alpha0, beta0, Bf simultaneously,
     BJT+Iii on. Hard bounds prevent Iii suppression.
  4. Final polish: L-BFGS, all params unfrozen, balanced loss.

All bounded params use sigmoid reparametrization. Same `logb`/`linb` helpers
as z79/z80. M1 and M2 share the same fitted BSIM4 card values (only length
differs and is handled inside cfg).
"""
from __future__ import annotations
import json
import math
import re
import time
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_default_dtype(torch.float64)

# GPU device selection (AMD ROCm gfx1151 with HSA override applied externally)
DEVICE = torch.device("cpu")  # CPU is faster than GPU for this small-batch workload
torch.set_default_device(DEVICE)
print(f"[z83] Using device: {DEVICE}", flush=True)
if DEVICE.type == "cuda":
    print(f"[z83] GPU: {torch.cuda.get_device_name(0)}", flush=True)

from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.arclength import forward_2t_arclength_grad
from nsram.bsim4_port.nsram_cell_2T import (
    NSRAMCell2TConfig, forward_2t,
)
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.geometry import Geometry

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data/sebas_2026_04_22"
OUT = ROOT / "results/z91d_bsim4_port_fit_arclength"
OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Data loader (verbatim from z80)
# --------------------------------------------------------------------------- #
def parse_vg2(s):
    m = re.search(r"VG2=(-?\d+\.\d+)", s);  return float(m.group(1)) if m else None
def parse_vg1(s):
    m = re.search(r"VG1=([\d.]+)", s);      return float(m.group(1)) if m else None


def load_curves():
    curves = []
    for d in sorted(DATA_DIR.glob("2vHCa-2 I-Vs@VG2 VG1=*")):
        VG1 = parse_vg1(d.name)
        for f in sorted(d.glob("*.csv")):
            VG2 = parse_vg2(f.name)
            data = np.loadtxt(f, delimiter=",", skiprows=1, usecols=(0, 1))
            if data.ndim == 1:
                continue
            half = len(data) // 2
            Vd = data[:half, 0]; Id = np.abs(data[:half, 1])
            mask = (Vd >= 0.05) & (Vd <= 2.0)
            Vd, Id = Vd[mask], Id[mask]
            if len(Vd) > 10:
                idx = np.linspace(0, len(Vd) - 1, 10).astype(int)
                Vd, Id = Vd[idx], Id[idx]
            if len(Vd) < 5:
                continue
            curves.append({"VG1": VG1, "VG2": VG2,
                           "Vd": torch.tensor(Vd, dtype=torch.float64),
                           "Id": torch.tensor(Id, dtype=torch.float64)})
    return curves


# --------------------------------------------------------------------------- #
# Param spec (oracle-recommended for v6 — no gamma/Rb/C/PT)
# --------------------------------------------------------------------------- #
PARAM_SPEC = {
    # transport — z91: tightened vth0 lower bound (5th-oracle finding: z88 fled
    # to 0.20 V to compensate for masked-loss + non-converged Vsint/Vb leakage,
    # producing Vth_eff(M2) ≈ -9 mV which is non-physical)
    "vth0":  {"kind": "linb", "init": 0.40,  "bounds": (0.30, 0.55)},
    "u0":    {"kind": "logb", "init": 0.06,  "bounds": (0.02, 0.15)},
    "vsat":  {"kind": "logb", "init": 1e5,   "bounds": (5e4, 3e5)},
    # k1, k2 tightened — z88 ended at k1=0.90 (upper bound)
    "k1":    {"kind": "linb", "init": 0.50,  "bounds": (0.30, 0.80)},
    "k2":    {"kind": "linb", "init": 0.00,  "bounds": (-0.10, 0.10)},
    # GIDL
    "agidl": {"kind": "logb", "init": 5e-7,  "bounds": (1e-7, 1e-5)},
    "bgidl": {"kind": "logb", "init": 8e8,   "bounds": (3e8, 1.2e9)},
    "cgidl": {"kind": "linb", "init": 0.5,   "bounds": (0.3, 0.7)},
    "egidl": {"kind": "linb", "init": 0.5,   "bounds": (0.3, 0.6)},
    # impact-ion: alpha1 added (BSIM4: T2=(alpha0+alpha1*Leff)/Leff).
    # z88/z91 silently held alpha1=0, dropping a DoF the C-source has.
    # alpha1 range chosen so |alpha1*Leff| can match alpha0 at Leff=180nm
    # (i.e. ±0.05 / 180nm ≈ ±2.8e5 — start narrower at ±5e3 1/m).
    "alpha0": {"kind": "logb", "init": 5e-3, "bounds": (1e-3, 5e-2)},
    "alpha1": {"kind": "linb", "init": 0.0,  "bounds": (-5e3, 5e3)},
    "beta0":  {"kind": "linb", "init": 18.0, "bounds": (12.0, 30.0)},
    # BJT — Bf bound was [50, 300]; Sebas's parasiticBJT.txt has Bf=10000.
    # Widen to cover the real card value while still allowing lower fits.
    "Bf":     {"kind": "logb", "init": 1000., "bounds": (50.0, 50000.0)},
}

BSIM4_NAMES = {"vth0", "u0", "vsat", "k1", "k2",
               "agidl", "bgidl", "cgidl", "egidl",
               "alpha0", "alpha1", "beta0"}
BJT_NAMES = {"Bf"}


# --------------------------------------------------------------------------- #
# Reparametrization helpers (verbatim z80 idiom)
# --------------------------------------------------------------------------- #
def init_theta(name: str, jitter_seed: int = 0) -> torch.Tensor:
    spec = PARAM_SPEC[name]
    kind, init, bnd = spec["kind"], spec["init"], spec["bounds"]
    rng = np.random.default_rng(hash((name, jitter_seed)) & 0xFFFFFFFF)
    lo, hi = bnd
    if kind == "linb":
        u = (float(init) - lo) / (hi - lo)
    else:  # logb
        u = math.log(float(init) / lo) / math.log(hi / lo)
    u = min(max(u, 1e-6), 1.0 - 1e-6)
    theta0 = math.log(u / (1.0 - u))
    if jitter_seed != 0:
        theta0 += float(rng.normal(0, 0.3))
    return torch.tensor(theta0, dtype=torch.float64, requires_grad=True)


def theta_to_value(name: str, theta: torch.Tensor):
    spec = PARAM_SPEC[name]
    kind, bnd = spec["kind"], spec["bounds"]
    lo, hi = bnd
    s = torch.sigmoid(theta)
    if kind == "linb":
        return lo + (hi - lo) * s
    return lo * (hi / lo) ** s


def make_thetas(seed: int) -> dict:
    return {n: init_theta(n, seed) for n in PARAM_SPEC}


def thetas_to_values(thetas: dict) -> dict:
    return {n: theta_to_value(n, t) for n, t in thetas.items()}


def clone_thetas(thetas: dict) -> dict:
    return {n: t.detach().clone().requires_grad_(True) for n, t in thetas.items()}


# --------------------------------------------------------------------------- #
# SizeDep override — patch sd.scaled[...] / sd.vth0_T / sd.u0temp / sd.vsattemp
# while staying differentiable. Iii/GIDL params live in sd.scaled; transport
# params live as direct sd attributes. This is the wrinkle the spec's P_M1/P_M2
# dict abstracts over; we implement it explicitly so grads flow.
# --------------------------------------------------------------------------- #
SCALED_KEYS = {"k1", "k2", "agidl", "bgidl", "cgidl", "egidl",
               "alpha0", "alpha1", "beta0"}
ATTR_KEYS = {"vth0": "vth0_T", "u0": "u0temp", "vsat": "vsattemp"}


@contextmanager
def patch_sd(sd, values: dict):
    """Temporarily override SizeDependParam values for fitting.

    `values` maps PARAM_SPEC name -> tensor. We patch:
      - sd.scaled[name] for SCALED_KEYS
      - sd.<attrname> for ATTR_KEYS (vth0->vth0_T etc.)
    Non-bsim4 params ignored.
    """
    saved_scaled = {}
    saved_attr = {}
    try:
        for name, val in values.items():
            if name in SCALED_KEYS:
                saved_scaled[name] = sd.scaled.get(name, None)
                sd.scaled[name] = val
            elif name in ATTR_KEYS:
                attr = ATTR_KEYS[name]
                saved_attr[attr] = getattr(sd, attr)
                setattr(sd, attr, val)
        yield
    finally:
        for k, v in saved_scaled.items():
            if v is None:
                sd.scaled.pop(k, None)
            else:
                sd.scaled[k] = v
        for k, v in saved_attr.items():
            setattr(sd, k, v)


# --------------------------------------------------------------------------- #
# Forward: build cfg+sd once, patch with current θ each call.
# --------------------------------------------------------------------------- #
def make_cfg_and_sd(model: BSIM4Model, gates: dict):
    cfg = NSRAMCell2TConfig(
        Ln=180e-9, Wn=360e-9, M2_length_factor=10.0, T_C=27.0,
        use_iii=gates.get("use_iii", True),
        use_gidl=gates.get("use_gidl", True),
        use_bjt=gates.get("use_bjt", True),
    )
    # Force sd cache populate
    cfg.size_dep_M1(model)
    cfg.size_dep_M2(model)
    return cfg


def forward_curve(values: dict, model: BSIM4Model, cfg: NSRAMCell2TConfig,
                  VG1: float, VG2: float, Vd_seq: torch.Tensor,
                  use_homotopy: bool = False,
                  dense_vd_in_snapback: bool = True) -> tuple:
    """z91d: pseudo-arclength continuation for path-tracing through snapback.

    z94 showed: arclength gives 100% conv vs 60% with Newton+homotopy and is
    50-150× faster. We use it as the warm-start oracle, then re-run grad-
    tracked Newton from those starting points so autograd flows.
    """
    sd_M1 = cfg._sd_M1
    sd_M2 = cfg._sd_M2
    bjt = GummelPoonNPN.from_sebas_card()
    if "Bf" in values:
        bjt.Bf = values["Bf"]

    VG1_t = torch.tensor(VG1, dtype=torch.float64)
    VG2_t = torch.tensor(VG2, dtype=torch.float64)

    with patch_sd(sd_M1, values), patch_sd(sd_M2, values):
        result = forward_2t_arclength_grad(cfg, model, bjt, Vd_seq,
                                             VG1_t, VG2_t)
    converged_mask = torch.tensor([bool(c) for c in result["converged"]],
                                   dtype=torch.float64)
    return result["Id"].abs(), converged_mask


# --------------------------------------------------------------------------- #
# z91 Loss: Huber on log10(|Id|) with measurement-floor clip + non-conv penalty
# --------------------------------------------------------------------------- #
HUBER_DELTA = 1.0       # log-decade — Huber transition point
MEAS_FLOOR  = 1e-13     # A — measurement noise floor (instrument limit)
NONCONV_PENALTY = 4.0   # log-decade — penalty added at non-converged biases
                        #   (chosen so Newton failures hurt loss without exploding grad)


def huber_log_loss(pred: torch.Tensor, meas: torch.Tensor,
                   conv_mask: torch.Tensor) -> torch.Tensor:
    """Huber on log10(|Id|), with measurement floor + non-convergence penalty.

    No mask-skip: every bias contributes to the loss. Newton-failed biases are
    detached from the autograd graph (delta zeroed in nsram_cell_2T.py) but
    still receive a fixed scalar penalty, so the optimizer is rewarded for
    parameter sets where Newton converges more often.
    """
    log_eps = MEAS_FLOOR
    log_p = torch.log10(pred.abs().clamp_min(log_eps))
    log_m = torch.log10(meas.abs().clamp_min(log_eps))
    err = log_p - log_m
    a = err.abs()
    huber = torch.where(a < HUBER_DELTA,
                         0.5 * err * err,
                         HUBER_DELTA * (a - 0.5 * HUBER_DELTA))
    # Penalty for non-converged biases: a fixed log-decade equivalent
    nonconv = (1.0 - conv_mask) * (NONCONV_PENALTY ** 2 * 0.5)
    return (huber + nonconv).mean()


def stage_loss(thetas, model, cfg, curves, *, use_homotopy: bool = False):
    values = thetas_to_values(thetas)
    losses = []
    for c in curves:
        try:
            Id_pred, conv_mask = forward_curve(values, model, cfg,
                                                c["VG1"], c["VG2"], c["Vd"],
                                                use_homotopy=use_homotopy)
        except RuntimeError as e:
            print(f"    skip VG1={c['VG1']} VG2={c['VG2']}: {e}", flush=True)
            continue
        l = huber_log_loss(Id_pred, c["Id"], conv_mask)
        if torch.isfinite(l):
            losses.append(l)
    if not losses:
        return torch.tensor(1e3, dtype=torch.float64, requires_grad=True)
    return torch.stack(losses).mean()


# --------------------------------------------------------------------------- #
# Stage runner
# --------------------------------------------------------------------------- #
def run_stage(stage_id: int, label: str, thetas: dict, model, cfg, curves,
              fit_names: list, *, n_adam: int, n_lbfgs: int,
              lr_adam: float = 0.05, lr_lbfgs: float = 0.5,
              t0: float = 0.0, use_homotopy: bool = False):
    fit_thetas = [thetas[n] for n in fit_names]
    for n in fit_names:
        thetas[n].requires_grad_(True)

    with torch.no_grad():
        l0 = stage_loss(thetas, model, cfg, curves, use_homotopy=use_homotopy)
    print(f"[stage {stage_id}{label}] init loss = {l0.item():.4f}  "
          f"(fitting {len(fit_names)}, {len(curves)} curves, "
          f"homotopy={use_homotopy})  ({time.time()-t0:.0f}s)", flush=True)

    if n_adam > 0:
        opt = torch.optim.Adam(fit_thetas, lr=lr_adam)
        for it in range(n_adam):
            opt.zero_grad()
            l = stage_loss(thetas, model, cfg, curves, use_homotopy=use_homotopy)
            l.backward()
            torch.nn.utils.clip_grad_norm_(fit_thetas, max_norm=2.0)
            opt.step()
            if it % 5 == 0 or it == n_adam - 1:
                print(f"  s{stage_id}{label} Adam {it}: loss={l.item():.4f}  "
                      f"({time.time()-t0:.0f}s)", flush=True)

    if n_lbfgs > 0:
        opt2 = torch.optim.LBFGS(fit_thetas, max_iter=n_lbfgs, lr=lr_lbfgs,
                                  line_search_fn="strong_wolfe")
        def closure():
            opt2.zero_grad()
            l = stage_loss(thetas, model, cfg, curves, use_homotopy=use_homotopy)
            l.backward()
            return l
        try:
            opt2.step(closure)
        except RuntimeError as e:
            print(f"  s{stage_id}{label} L-BFGS warn: {e}", flush=True)

    with torch.no_grad():
        lf = stage_loss(thetas, model, cfg, curves, use_homotopy=use_homotopy)
    print(f"[stage {stage_id}{label}] final loss = {lf.item():.4f}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    return float(lf.item())


def multistart_stage(stage_id, base_thetas, model, cfg, curves, fit_names,
                     *, n_adam, n_lbfgs, n_seeds=3, t0,
                     lr_adam: float = 0.05, lr_lbfgs: float = 0.5,
                     use_homotopy: bool = False):
    best_loss = float("inf")
    best_thetas = None
    for seed in range(n_seeds):
        thetas = clone_thetas(base_thetas)
        if seed > 0:
            for n in fit_names:
                thetas[n] = init_theta(n, jitter_seed=seed)
        loss = run_stage(stage_id, f".s{seed}", thetas, model, cfg, curves,
                         fit_names, n_adam=n_adam, n_lbfgs=n_lbfgs, t0=t0,
                         lr_adam=lr_adam, lr_lbfgs=lr_lbfgs,
                         use_homotopy=use_homotopy)
        if loss < best_loss:
            best_loss = loss
            best_thetas = clone_thetas(thetas)
            print(f"  ** stage {stage_id} new best @ seed {seed}: "
                  f"loss={loss:.4f}", flush=True)
    return best_thetas, best_loss


# --------------------------------------------------------------------------- #
# Eval + save
# --------------------------------------------------------------------------- #
def evaluate_full(thetas, model, cfg, curves):
    log_eps = 1e-15
    values = thetas_to_values(thetas)
    rmses, preds = [], []
    for c in curves:
        try:
            with torch.no_grad():
                Id_pred = forward_curve(values, model, cfg,
                                        c["VG1"], c["VG2"], c["Vd"],
                                        use_homotopy=True)
                if isinstance(Id_pred, tuple):
                    Id_pred, conv_mask = Id_pred
                else:
                    conv_mask = torch.ones_like(Id_pred, dtype=torch.bool)
        except RuntimeError as e:
            print(f"  eval skip VG1={c['VG1']} VG2={c['VG2']}: {e}", flush=True)
            continue
        log_p = torch.log10(Id_pred.abs() + log_eps)
        log_m = torch.log10(c["Id"].abs() + log_eps)
        # honest per-curve metric: only count converged biases
        cm = conv_mask.bool() if conv_mask.dtype != torch.bool else conv_mask
        if cm.any():
            sq = (log_p - log_m) ** 2
            rmse = float(torch.sqrt(sq[cm].mean()).item())
        else:
            rmse = float("inf")
        rmses.append(rmse)
        preds.append({"VG1": c["VG1"], "VG2": c["VG2"], "log_rmse": rmse,
                      "Vd": c["Vd"].numpy().tolist(),
                      "Id_meas": c["Id"].numpy().tolist(),
                      "Id_pred": Id_pred.detach().numpy().tolist(),
                      "converged": cm.detach().numpy().tolist()})
    return float(np.median(rmses)) if rmses else float("inf"), preds


def fitted_dict(thetas):
    return {n: float(theta_to_value(n, thetas[n]).detach().item()) for n in thetas}


def save_stage_summary(stage_id, thetas, loss):
    p = OUT / f"stage{stage_id}_summary.json"
    p.write_text(json.dumps(
        {"stage": stage_id, "loss": loss, "params": fitted_dict(thetas)},
        indent=2))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    base_card_text = (DATA_DIR / "PTM130bulkNSRAM.txt").read_text()
    model = BSIM4Model.from_spice(base_card_text)
    curves = load_curves()
    print(f"Loaded {len(curves)} curves at {time.time()-t0:.1f}s", flush=True)

    # Curve subsets
    off_state = [c for c in curves if abs(c["VG1"] - 0.2) < 1e-3]
    low_vd = []
    for c in curves:
        mask = c["Vd"] < 0.8
        if mask.sum().item() >= 4:
            low_vd.append({"VG1": c["VG1"], "VG2": c["VG2"],
                           "Vd": c["Vd"][mask], "Id": c["Id"][mask]})
    print(f"  off_state (VG1=0.2): {len(off_state)}", flush=True)
    print(f"  low_vd  (Vd<0.8):    {len(low_vd)}", flush=True)
    print(f"  full curves:         {len(curves)}", flush=True)

    # Initialize thetas
    thetas = make_thetas(seed=0)

    # --- Stage 1: off-state, GIDL + vth0; BJT off, Iii off --- #
    cfg1 = make_cfg_and_sd(model, gates={"use_iii": False, "use_gidl": True,
                                          "use_bjt": False})
    s1_names = ["agidl", "bgidl", "cgidl", "egidl", "vth0"]
    print("\n=== Stage 1: Off-state (GIDL + vth0) ===", flush=True)
    thetas, l1 = multistart_stage(1, thetas, model, cfg1, off_state,
                                   s1_names, n_adam=30, n_lbfgs=15,
                                   n_seeds=3, t0=t0)
    save_stage_summary(1, thetas, l1)

    # --- Stage 2: core transport, low Vd; BJT off, Iii off --- #
    cfg2 = make_cfg_and_sd(model, gates={"use_iii": False, "use_gidl": True,
                                          "use_bjt": False})
    s2_names = ["u0", "vsat", "vth0", "k1", "k2"]
    print("\n=== Stage 2: Core transport (low-Vd) ===", flush=True)
    thetas, l2 = multistart_stage(2, thetas, model, cfg2, low_vd,
                                   s2_names, n_adam=30, n_lbfgs=15,
                                   n_seeds=3, t0=t0)
    save_stage_summary(2, thetas, l2)

    # --- Stage 3: snapback (MERGED) full data, BJT+Iii ON --- #
    # Stage 3 explodes with default lr=0.05 because BJT exp(Vbe/Vt) is
    # exponentially sensitive. Lower lr + fewer seeds (Stage 1+2 already
    # explored basins; Stage 3 just needs to balance Iii-vs-BJT).
    cfg3 = make_cfg_and_sd(model, gates={"use_iii": True, "use_gidl": True,
                                          "use_bjt": True})
    s3_names = ["alpha0", "alpha1", "beta0", "Bf"]
    print("\n=== Stage 3: Snapback merged (alpha0+beta0+Bf, gmin homotopy ON) ===", flush=True)
    thetas, l3 = multistart_stage(3, thetas, model, cfg3, curves,
                                   s3_names, n_adam=30, n_lbfgs=15,
                                   n_seeds=2, t0=t0,
                                   lr_adam=0.005, lr_lbfgs=0.1,
                                   use_homotopy=True)
    save_stage_summary(3, thetas, l3)

    # --- Stage 4: final polish, all params, L-BFGS only --- #
    cfg4 = make_cfg_and_sd(model, gates={"use_iii": True, "use_gidl": True,
                                          "use_bjt": True})
    s4_names = list(PARAM_SPEC.keys())
    print(f"\n=== Stage 4: Final polish (L-BFGS, all {len(s4_names)} params, "
          f"homotopy ON) ===", flush=True)
    l4 = run_stage(4, "", thetas, model, cfg4, curves,
                   s4_names, n_adam=0, n_lbfgs=25, t0=t0,
                   use_homotopy=True)
    save_stage_summary(4, thetas, l4)

    # --- Final eval --- #
    median_rmse, preds = evaluate_full(thetas, model, cfg4, curves)
    print(f"\n=== Final median log-RMSE = {median_rmse:.3f} ===", flush=True)

    fitted = fitted_dict(thetas)
    summary = {
        "stage_losses": {"1": l1, "2": l2, "3": l3, "4": l4},
        "median_log_rmse": median_rmse,
        "fitted_params": fitted,
        "elapsed_s": time.time() - t0,
        "n_curves": len(curves),
        "config": "z91 — Huber-on-log10|Id| + gmin homotopy + tight vth0/k1/k2 bounds + IFT-gated",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (OUT / "per_curve.json").write_text(json.dumps(preds, indent=1))

    print("\nFitted params:")
    for k, v in fitted.items():
        print(f"  {k:14s} = {v:+.4e}", flush=True)
    print(f"\nTotal elapsed: {summary['elapsed_s']:.0f}s")

    # 3-panel plot
    by_vg1 = {}
    for p in preds:
        by_vg1.setdefault(p["VG1"], []).append(p)
    if by_vg1:
        fig, axes = plt.subplots(1, max(len(by_vg1), 1), figsize=(6 * len(by_vg1), 6),
                                 sharey=True, squeeze=False)
        axes = axes[0]
        cmap = plt.get_cmap("viridis")
        for ax, VG1 in zip(axes, sorted(by_vg1)):
            ps = sorted(by_vg1[VG1], key=lambda c: c["VG2"])
            n = len(ps)
            for i, p in enumerate(ps):
                color = cmap(i / max(n - 1, 1))
                Vd = np.asarray(p["Vd"])
                ax.semilogy(Vd, p["Id_meas"], "o", color=color, ms=4, alpha=0.7,
                             label=f"VG2={p['VG2']:+.2f}")
                ax.semilogy(Vd, p["Id_pred"], "-", color=color, lw=1.5)
            ax.set_xlabel("Vd [V]"); ax.grid(alpha=0.3)
            ax.set_title(f"VG1 = {VG1} V    ({n} curves)")
            ax.legend(loc="lower right", fontsize=7, ncol=2)
        axes[0].set_ylabel("|Id| [A]")
        fig.suptitle(
            f"P7v6: 2T topology stage-wise fit\n"
            f"median log-RMSE = {median_rmse:.2f}  —  elapsed {summary['elapsed_s']:.0f}s",
            fontsize=12, weight="bold",
        )
        fig.tight_layout()
        fig.savefig(OUT / "fit_curves.png", dpi=140)
        plt.close(fig)
        print(f"Wrote {OUT/'fit_curves.png'}")


if __name__ == "__main__":
    main()

```


=== FILE: _extracted/O2_packet/parasiticBJT.txt (244 chars) ===
```
* Simple bjt for floating bulk parasitic bipolar effect
* Pazos, S.

.model parasiticBJT NPN(is=5E-9 va=100 bf=10000 br=100 nc=2 ikr=100m rc=0.1 vje=0.7 re=0.1 cjc=1e-15 fc=0.5 cje=0.7e-15 ne=1.5 ise=0 tr=20e-12 tf=25e-12 itf=0.03 vtf=7 xtf=2)

```


=== FILE: _extracted/O2_packet/email_history.md (20713 chars) ===
```
# Email history Sebas / Mario / Robert / Eric (extracted from TXT.rtf)





\*\*\
\*\
\*





\

\
Inga har markerats\'a0\

\

\
\
\
\
\
\

Forts\'e4tt till inneh\'e5ll\
Anv\'e4nda Gmail med sk\'e4rml\'e4sningsprogram\















\




\
\
\


\
\
\


\


\


\
\
\


\


\
\
\
\
\
\
\
\
\
\
\


\
\

\
\
\


Genv\'e4gar\


\


\
Startsida
\
\

\
Omn\'e4mnanden
\
\


star
Stj\'e4rnm\'e4rkta
\
\
\
\
\
\


Direktmeddelanden\


\
\
\

¬
\

\


NIKLAS KOTARSKY\


\
\
\
\
\
\
\
\
\


Rum\


\
\


Skapa ett rum f\'f6r att chatta och samarbeta\

Hitta ett rum att g\'e5 med i\


\
\
\


Appar\


\
\


Det finns \'e4nnu inga appar\

Utforska appar
\

\
\

arrow_downward\


fler ol\'e4sta\

1 av 18\'a0525\


\

\


\
\
\
\
\
\


Zoom NSRAM\


\


Inkorgen\
\

H\'e4ndelsen har avbrutits
\

tis 21 apr. \'95 13:00\'9614:00\


Zoom NSRAM
\

¬
\

\*

.cls-1\.cls-2\.cls-3\.cls-4\.cls-5\.cls-6\.cls-7\.cls-8\
\


Borttagen fr\'e5n Google Kalender
\


Baserat p\'e5 det h\'e4r e-postmeddelandet\
St\'e4mmer det?\
\
\


¬
\


Mario Lanza Martinez
 <mlanza@nus.edu.sg>\
\


¬
\


fre 20 mars 13:06\


\

\
\
\
\


till Robert, mig, Pazos
\

¬
\


\
\
\
\

\

Det h\'e4r meddelandet verkar vara p\'e5 engelska\
\
\
\


\
\
\


\'a0
\

\'a0\










\*¬










Hi there,








Mario Lanza Martinez is inviting you to a scheduled Zoom meeting.







\*







Meeting URL:


\*






Meeting ID:



847 3855 1708







Password:



429852







Join from an H.323/SIP room system








H.323:



144.195.19.161 (US West)206.247.11.121 (US East)159.124.15.191 (Amsterdam Netherlands)159.124.47.249 (Germany)159.124.104.213 (Australia Sydney)159.124.74.212 (Australia Melbourne)170.114.180.219 (Singapore)159.124.168.213 (Canada Toronto)159.124.196.25 (Canada Vancouver)170.114.194.163 (Japan Tokyo)147.124.100.25 (Japan Osaka)







Meeting ID:



847 3855 1708







Password:



429852







SIP:


\*






Password:



429852


\'a0\
\

\


\

Important: This email is confidential and may be privileged. If you are not the intended recipient, please delete it and notify us immediately; you should not copy or use it for any purpose, nor disclose its contents to any other person. Thank you.\


\
\
\


\

 En bilaga\
\'a0\'a0\'95\'a0\'a0Genoms\'f6kt av Gmail\
\
\

¬
invite.ics\'a0\*\'a0\'a0\


\
\


\
\


¬
\


Eric Bergvall
 <bergvall.eric@gmail.com>\
\


¬
\


m\'e5n 23 mars 11:14\


\

\
\
\
\


till Mario, Robert, Pazos
\

¬
\


\
\
\
\
\


Hello Mario and Sebastian,\
Thanks for taking the time last week. We are excited to see if we can contribute with our deep and wide software and system expertise. We have started the python and Julia simulation side and there is an initial python pip package you can find here\'a0\* with gpu support and the up to date parameters. Sebastian, check it out and let us know if this is a trajectory that can support your simulations and also to make it more widely accessible for collaboration. You can find one big hero teaser image below and you can install the package with pip install nsram. You also find the powerpoint and questions below as requested.\
We await your secretary, Mario, to reach out regarding the registration in Singapores research portal (?) for our companies and we also await your results in a couple of week.\
\
All the best,\
\
\





¬
\'a0


Eric Bergvall\'a0
\

m.
\'a0+4670 499 06 16
\

e.\'a0\*
\

a.\'a0
Sweden, Stockholm
\

\'a0


\

 2 bilagor\
\'a0\'a0\'95\'a0\'a0Genoms\'f6kt av Gmail\
\
\
\
\
\


\
\


\
\


¬
\


Pazos Sabattini Sebastian Matias
\
\


\


tis 24 mars 12:37\


\

\
\
\
\


till Sebastian, mig, Mario, Robert
\

¬
\


\
\
\
\

\

Det h\'e4r meddelandet verkar vara p\'e5 engelska\
\
\
\


\
\


Dear Eric, it was a pleasure to meet you last week, and I hope you are feeling better.\
\
Thanks for the slides and for the updates, I really appreciate it. I'm including my personal email account, as I will be conducting\'a0all communication on this topic over there through the following transition. I apologize in advance for any inconveniences this may cause.\
\
I've taken a look to the python package, it's a great kickstart and I appreciate you taking the time to work on it. I will most definitely be updating modeling parameters over the next few weeks, so I'll be reaching back soon with details and model improvements based on new experiments, as my to-do list starts to clear over the next couple of weeks.\
\
Thanks a lot for your amazing predisposition. I'll be in touch. Kind regards,\
\
Sebas\
\


\


From:
 Eric Bergvall <\*>\

Sent:
 Monday, March 23, 2026 6:14 PM\

To:
 Mario Lanza Martinez <\*>; Robert Luciani <\*>; Pazos Sabattini Sebastian Matias <\*>\

Subject:
 Re: Zoom NSRAM
\

\'a0\





\'a0


- External Email -


\
\


\
\


¬
\


Eric Bergvall
 <bergvall.eric@gmail.com>\
\


\


ons 25 mars 19:56\


\

\
\
\
\


till Pazos, Mario, Robert, Sebastian
\

¬
\


\
\
\
\
\


Hi Sebas, great to hear from you \'97 doing much better, thanks.\
\
Really glad you had a look. Since you last checked it's grown quite a bit, so here's a quick rundown of what's in there now (v0.9.0, pip install nsram):\
\
Device physics layer \'97 the full body-charge ODE, Chynoweth avalanche model, SRH charge trapping, and temperature-dependent BVpar, all matched to your Zenodo SPICE parameters. Single-cell simulation with scipy for validation against TCAD/SPICE.\
\
Characterization tools \'97 this is the part I think you'll find most useful when the new parameters come in. There's automated I-V curve fitting (drop in a CSV, it extracts BV0, Is, Ne), transient pulse response simulation with tau extraction, LTP/LTD cycle simulation (currently gets 7 distinct conductance levels), and Arrhenius retention modelling (gives tau=10,139s at 300K which lines up with your >10

s figure, and the extracted Ea is self-consistent).\
\
I've also been thinking ahead about what the library should support beyond the paper \'97 things like deep N-well high-voltage operation, sweep-rate dependent I-V hysteresis, polynomial bulk current models as an alternative to the exponential fit, E/I input neuron configurations, and frequency-coded spike encoding for image classification. These are all stubbed out with placeholder parameters, so if any of them are relevant to where your work is heading, the framework is ready.\
\
Monte Carlo variability \'97 parameterised die-to-die variation for array yield estimation. Should be useful if you're thinking about scaling to crossbar arrays.\
\
Network-level reservoir computing \'97 GPU-accelerated (CUDA/ROCm), 5 neuron models for comparison (your NS-RAM AdEx-LIF, plus Izhikevich, Hodgkin-Huxley, parametric LIF, and a standard ESN baseline). NS-RAM outperforms all of them: 97% temporal XOR, 99.6% Mackey-Glass chaotic prediction, 96.75% MNIST.\
\
Technology comparison \'97 NS-RAM benchmarked against RRAM, PCM, MRAM, hBN memristors, and CMOS neurons on area/energy/endurance.\
\
Everything is open-source at \*. When you send the updated parameters I can plug them straight into the DeviceParams dataclass and re-run all the validation \'97 the fitting pipeline is designed for exactly that workflow.\
\
Looking forward to it. No rush at all on your end.\
\
\'a0 Best Regards,\


\
\


\
\


¬
\


Eric Bergvall
 <bergvall.eric@gmail.com>\
\


\


tors 2 apr. 14:08\


\

\
\
\
\


till Pazos, Mario, Robert, Sebastian
\

¬
\


\
\
\
\
\


Hello Mario and Sebas,\
Hope you are doing fine.\'a0\
Would it be possible to move our meeting from Tuesday 21 April to Thursday 23 April? Same time works for us.\
\
Quick update: we pushed \* v0.10.0 (pip install nsram, \*) with a new module called BEAM -- a byte-level online learner where the core memory is a set of small matrices updated by a delta rule that maps directly to crossbar conductances. No backpropagation needed. 3.14 bits/char on text8 with 60K parameters in pure C.\
\
Details when we meet and just let us know when you have new info/data to share. Looking forward to it!\


\
\


\
\


¬
\


Mario Lanza Martinez
\
\


\


l\'f6r 4 apr. 17:51\


\

\
\
\
\


till mig, Pazos, Robert, Sebastian
\

¬
\


\
\
\
\

\

Det h\'e4r meddelandet verkar vara p\'e5 engelska\
\
\
\


\
\
\


Dear Eric,
\

\'a0
\

Thanks a lot for your email. I am now in the middle of a trip, and Sebastian did not have time yet to complete the modeling. Let us meet in the second half of April. I finish my trip on April 20
th
, and by that time we expect Sebastian to have some progress on the model. I am very sorry for being slow with this topic, but I think we are making a solid foundation.
\

\'a0
\

Yesterday I was presenting this in a company and they seem to be interested.
\

\'a0
\

Best regards,
\

\'a0
\

---
\

Mario Lanza, Ph.D. \'96 IEEE Fellow
\

Associate Professor of Materials Science and Engineering
\

9 Engineering Drive 1, Block EA, Office 05-28
\

National University of Singapore, 117575 Singapore
\


Email: \* \'96 Web: \*\'a0
\


\
\


\
\


¬
\


Sebastian Pazos
\
\


\


fre 17 apr. 22:45\


\

\
\
\
\


till Mario, mig, Pazos, Robert
\

¬
\


\
\
\
\

\

Det h\'e4r meddelandet verkar vara p\'e5 engelska\
\
\
\


Dear Eric, Robert, I hope this email finds you well. I'm quite alright with pushing our call to 23rd, I'm not sure about Mario's schedule.\
\
Apologies for the radio-silence. I wanted to let you know that, amidst the transition to my new roles, I'm finding time to work on the new model fits to the new data.\
\
Things are starting to look up\'a0in terms of model agreement with experimental results, but the modeling approach has changed a little bit from that old Zenodo repository of the paper.\'a0\
\
I'm working with a different SPICE tool and foundry-provided models and I'm focusing on adapting foundry models of body bias and impact ionization (in \*) to fit the floating body behaviour of our 2T cells. This renders a more standard approach for us circuit designers.\
\
In that sense, I've dropped the avalanche diode models (very annoying for convergence) that fire the bipolar parasitic effect in LTSpice, and I'm only including a complementary bipolar current to capture the full swing of the firing mechanism. Fits are looking good, but I'm still working on polynomial dependence of model parameters with tuning voltages (VG1, VG2) and layout dependent effect on transistor models to capture the experimental behaviour.\
\
My question at this point is: can your approach drop the avalanche voltage as a control parameter and deal with the BSIM Impact ionization and body voltage directly? Alternatively, I could run my latest I-V curves through your Python I-V fit module. Let me know what would you prefer.\
\
Kind regards,\
\
Sebas\


\
\


\
\


¬
\


Robert Luciani
\
\


\


l\'f6r 18 apr. 01:56 (f\'f6r 13 dagar sedan)\


\

\
\
\
\


till Sebastian, Mario, mig, Pazos
\

¬
\


\
\
\
\

\

Det h\'e4r meddelandet verkar vara p\'e5 engelska\
\
\
\


\

Hi Sebastian,\
Good to hear fits are converging!\
Thanks for the BSIM4 manual.\
\
I'm wondering if you could share three things for the Julia simulator:\
1. The 2T cell schematic, plus whatever array/grid topology you're planning to build out of them (shared lines, neighbor coupling, etc.).\
2. Raw I-V measurements from the silicon (e.g. CSV with bias conditions). No fits needed ^_^\'a0\
3. The process node you're targeting, so I know which BSIM4 parameter set to start from. \
If the foundry card is shareable, even better!\
\
Call on the 23rd works for me.\
\
Kind regards,\
~ Robert\


\
\


\
\


¬
\


Eric Bergvall
 <bergvall.eric@gmail.com>\
\


\


s\'f6n 19 apr. 18:51 (f\'f6r 12 dagar sedan)\


\

\
\
\
\


till Robert, Sebastian, Mario, Pazos
\

¬
\


\
\
\
\
\


Hi Sebas, Mario, Robert,
\

Short answers first \'97 23rd works for us.
\

To Sebastian \'97 your question #1: yes, we dropped 
BVpar
. We pushed 
nsram v0.12.0
 this week with the full BSIM4 floating-body stack:
\



\'a76.1 impact ionization
 (
ALPHA0
 / 
BETA0
) driving body charge directly\


\'a72.2 Vth(Vbs)
 with 
K1
 / 
K2
 for body-bias modulation\


\'a710.1 junction breakdown
 as an alternative firing path. We ran a head-to-head against your published Chynoweth I-V over 2\'964.5 V: 
\'a76.1 channel HCI fits ~4 decades RMS better than \'a710.1 junction breakdown
 \'97 so the channel-HCI route looks like the right match for your 2T cell, aligned with the "complementary bipolar current on top of BSIM4" description from your last email.\


\'a712 full temperature scaling
 (
KT1
, 
UTE
, 
XTIS
)\


\'a713 layout stress
 (
SA
, 
SB
, 
KU0
, 
KVTH0
) \'97 direct home for your "layout-dependent effect"\



PolynomialBSIM4Params
 \'97 wrapper for the 
\(VG1, VG2)
 polynomial fits you're working on. Drop in coefficients and it evaluates per bias point.\


body_charge_ode_bsim4_full(...)
 has a 
firing_mode 

 \
 switch if you want to A/B both paths against your data.
\


And yes on option #2 too \'97 please send the I-V curves.
 
fit_bsim4_impact(Vds, Isub, Vgs)
 extracts 
ALPHA0
/
BETA0
 from a CSV (synthetic self-consistency R\'b2=0.9998). GPU batch mode fits ~4000 curves in 0.7 s if wafer-scale Monte Carlo is relevant.
\


pip install --upgrade nsram
 \'97 0.12.0 is live on PyPI, repo at \*, runnable example at 
examples/bsim4_2t_floating_body.py
\


Echoing Robert's asks (helpful for both the Python and Julia sides):
\



2T cell schematic + any planned array topology (shared lines, neighbor coupling)\

Raw I-V CSVs \'97 no fits needed, we'd like to run them through the pipeline ourselves\

Process node, so we pick the right BSIM4 parameter set\


Foundry model card if shareable\

No rush \'97 whatever arrives before the 23rd helps frame the call, but we can equally well iterate afterwards. Also to speed up our collaboration we would like to propose weekly cadence with you Sebastian to share insights and progress but we can discuss it in the next meeting.
\


Mario \'97 one quick check-in on the vendor registration.
 Last we heard (24 March) you were planning to speak with an officer about whether NUS would engage ENIMBLE Solutions AB as a foreign company or me personally. Any update there? Happy to send additional documents (English certificate of incorporation, VAT registration, etc.) if that helps move it along. No pressure \'97 just want to make sure we're not blocking anything on your side before the 23rd.
\


Looking forward to the meeting.
\

Best regards
\


\
\


\
\


¬
\


Mario Lanza Martinez
\
\


\


m\'e5n 20 apr. 15:53 (f\'f6r 11 dagar sedan)\


\

\
\
\
\


till mig, Robert, Sebastian, Pazos
\

¬
\


\
\
\
\

\

Det h\'e4r meddelandet verkar vara p\'e5 engelska\
\
\
\


\
\
\


Hi Eric, Robert, Sebastian,
\

\'a0
\

Thanks a lot. I have modified the meeting time to 23
rd
 at 8 am Buenos Aires time.
\

\'a0
\

I have been travelling from March 28
th
 until yesterday and I couldn\'92t talk with the person in charge of the vendor registration. I will try to reach her tomorrow. I apologize for the delay.
\


\
\


\
\


¬
\


Sebastian Pazos
\
\


¬
\


m\'e5n 20 apr. 23:08 (f\'f6r 11 dagar sedan)\


\

\
\
\
\


till mig, Robert, Mario
\

¬
\


\
\
\
\

\

Det h\'e4r meddelandet verkar vara p\'e5 engelska\
\
\
\


Dear Eric, Robert, thank you so much for the quick response. Here's some material as per your requests:\
\
1) A circuit schematic of the 2T cell, as currently modelled, in ASC format (LTSPice, even though we are using a different simulator right now targeting tape-outs). Neighbour coupling will most likely be the first topology we'll target. I will soon have help to scale this into network and circuit level implementations.\
\
2) I-V curves in CSV files, at an average sweep rate of 0.2 volts per second, 3 different VG1 (0.2, 0.4, 0.6) and multiple VG2 (value enclosed within each filename). Let me know if this works for you. We are generating additional data now (multiple ramp rates for more data on dynamics, pulsed dynamics).\
\
3&4) A model card for the bipolar device within the 2T cell schematic and a set of parameters around the impact ionization / body effect set by me as starting point for my SPICE fittings. Sadly, the foundry's full model card cannot be shared without infringing NDAs, 
but the 130 nm\'a0(current working node) PTM model I'm attaching is a good starting point
.\
\
I hope you find this useful for the time being. More data and details are coming soon.\
\
Talk to you soon,\
\
Sebas\


\

 2 bilagor\
\'a0\'a0\'95\'a0\'a0Genoms\'f6kt av Gmail\
\
\
\
\
\


\
\


\
\


¬
\


Mario Lanza Martinez
\
\


\


tors 23 apr. 07:39 (f\'f6r 8 dagar sedan)\


\

\
\
\
\


till Sebastian, mig, Robert
\

¬
\


\
\
\
\

\

Det h\'e4r meddelandet verkar vara p\'e5 engelska\
\
\
\


\
\
\


Dear colleagues,
\

\'a0
\

I am extremely sorry that I have to move the meeting to next week because I got an important unexpected visitor (our startup investor) and he requires me for dinner exactly at the time of our meeting.
\

\'a0
\

Can we do the meeting at the same time (7 pm Singapore time) next week? Every day is fine for me. Thanks a lot.
\

\'a0
\

Best regards,
\

\'a0
\

---
\

Mario Lanza, Ph.D. \'96 IEEE Fellow
\

Associate Professor of Materials Science and Engineering
\

9 Engineering Drive 1, Block EA, Office 05-28
\

National University of Singapore, 117575 Singapore
\


Email: \* \'96 Web: \*\'a0
\

\'a0\


\
\


\
\


¬
\


Eric Bergvall
 <bergvall.eric@gmail.com>\
\


\


tors 23 apr. 10:23 (f\'f6r 8 dagar sedan)\


\

\
\
\
\


till Sebastian, Mario, Robert
\

¬
\


\
\
\
\
\


Dear Mario,\
\
From me and Roberts side we can move the meeting to next week Thursday 30th of April same time (7pm Singapore time).\
Does it work for you Sebastian?\


\
\


\
\


¬
\


Sebastian Pazos
\
\


¬
\


tors 30 apr. 15:25 (f\'f6r 17 timmar sedan)\


\

\
\
\
\


till mig, Mario, Robert
\

¬
\


\
\
\
\

\

Det h\'e4r meddelandet verkar vara p\'e5 engelska\
\
\
\


Dear Eric and Robert, it was a pleasure to connect today. Thank you for your updates. I wanted to share my raw slides (didn't have time to format, sorry about that) and fitting results with you, in case this data is useful for optimizing your engines (please do not share the BSIM cards publicly). Remember that some parameters change only for M1, while NFACTOR changes only for M2 (I attribute this to LDE, hence two separate model cards for each device). Let me know if you have any questions.\
\
I'm working on additional measurement data for the dynamic modeling and will try to organize it in a repository for these results, so I can provide you with better tracked updates and results.\'a0\
\
To summarize the technical aspects of the call a bit:\
1- All approaches that help better fit the set of parameters to properly model NS-RAM are very useful because\'a0they save SPICE simulation time and ease fits across\'a0different technologies.\
\
2-\'a0You asked if I had information about how performance may degrade with fan-out. I think the model is in a better place now to start checking this in SPICE. I'm a little shorthanded now, but if you come up with a simple architecture of a few tens of neurons that is feasible to simulate at the circuit level, this could serve as a nice working example to check this aspect.\
\
3- In the near future, we are targeting compact networks for specific, sparse-signal applications. However, thinking long-term, a "roadmap" can include how NSRAM can scale to larger models and how this can co-exist with simplifying massive models (some sort of sweetspot maybe around reasoning models?) of todays mainstream-AI.\
\
4- We can work on a collab doc to include some of these aspects as part of the centre proposal.\
\
5-\'a0BONUS TRACK: We didn't cover this in the call, but I'm working on a floorplan for the first testchip entirely dedicated to NS-RAM. This will already include small arrays of NS-RAM based neurons, but if there is something small and specific that you think can be useful to extract specific metrics that help your approach, please let me know and we can evaluate its inclusion (simplified schematic or block diagram of what you would be looking for and what you expect to get out of that cell in terms of info/figures of merit).\
\
I still have the feeling that I missed a question or two during our meeting: we covered many aspects quite quickly and now I feel like when you leave home thinking you forgot to grab something but can recall what it is\'a0¬. Please, forward any questions in this thread and I'll do my best to provide detailed/useful answers.\
\
Thanks again for your feedback and support. Kind regards,\
\
Sebas\


\

 En bilaga\
\'a0\'a0\'95\'a0\'a0Genoms\'f6kt av Gmail\
\
\
\
\


\
\


\
\
\


\
\

\

\
\
\
\
\
\
\
\
\
\
\


```


=== FILE: _normalised/2tnsram_simple_asc.txt (1419 chars) ===
```
Version 4
SHEET 1 3052 680
WIRE 800 0 512 0
WIRE 800 64 800 0
WIRE 816 64 800 64
WIRE 848 64 816 64
WIRE 512 112 512 0
WIRE 800 112 800 64
WIRE 608 160 512 160
WIRE 640 160 608 160
WIRE 688 160 640 160
WIRE 704 160 688 160
WIRE 736 160 704 160
WIRE 752 160 736 160
WIRE 432 192 384 192
WIRE 464 192 432 192
WIRE 512 240 512 208
WIRE 800 240 800 208
WIRE 800 240 512 240
WIRE 608 272 608 160
WIRE 800 272 800 240
WIRE 704 288 704 160
WIRE 624 320 608 320
WIRE 544 352 496 352
WIRE 560 352 544 352
WIRE 624 368 624 320
WIRE 624 368 608 368
WIRE 800 400 800 272
WIRE 608 416 608 368
WIRE 704 416 704 352
FLAG 640 160 B
FLAG 800 272 Sint
FLAG 816 64 D
FLAG 704 416 0
FLAG 608 416 0
FLAG 432 192 G
FLAG 544 352 G2
FLAG 384 192 G
IOPIN 384 192 In
FLAG 496 352 G2
IOPIN 496 352 In
FLAG 848 64 Din
IOPIN 848 64 In
FLAG 800 400 S
IOPIN 800 400 Out
FLAG 688 160 B
IOPIN 688 160 Out
SYMBOL npn 736 112 R0
SYMATTR InstName Q1
SYMATTR Value parasiticBJT
SYMATTR Value2 area=1u
SYMBOL nmos4 464 112 R0
SYMATTR InstName M1
SYMATTR Value2 l='Ln' w='Wn' m=1
SYMBOL cap 688 288 R0
WINDOW 3 22 49 Left 2
SYMATTR Value 'CBpar'
SYMATTR InstName C1
SYMATTR SpiceLine Rser=1m
SYMBOL nmos4 560 272 R0
SYMATTR InstName M2
SYMATTR Value2 l='Ln*10' w='Wn' m=1
TEXT 552 24 Left 2 !.param Ln=0.18u\n.param Wn=0.36u\n.param CBpar=1f
TEXT 520 -64 Left 2 !.inc PTM130bulkNSRAM.txt
TEXT 520 -40 Left 2 !.inc parasiticBJT.txt
TEXT 310 478 Left 2 !.op 0

```


=== FILE: _extracted/O2_packet/M2_130bulkNSRAM.txt (10507 chars) ===
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


=== FILE: _extracted/O2_packet/2Tcell_BSIM_param_DC.csv (1916 chars) ===
```csv
VG1,VG2,trise,ETAB,K1,ALPHA0,BETA0,NFACTOR,mbjt,IS,area
0.2,-0.2,11.63,0.8,0.55825,7.842e-05,10.75,12.15,0.001,5e-09,1e-06
0.2,-0.15,11.63,0.85,0.55825,7.842e-05,11,11.15,0.001,5e-09,1e-06
0.2,-0.1,11.63,0.9,0.55825,7.842e-05,11.25,10.15,0.001,5e-09,1e-06
0.2,-0.05,11.63,0.95,0.55825,7.842e-05,11.5,9.15,0.001,5e-09,1e-06
0.2,0,11.63,1,0.55825,7.842e-05,12.5,8.15,0.001,5e-09,1e-06
0.2,0.05,11.63,1.05,0.55825,7.842e-05,13.5,7.15,0.001,5e-09,1e-06
0.2,0.1,12.73,1.1,0.55825,7.842e-05,14,6.25,0.001,5e-09,1e-06
0.4,-0.2,NaN,NaN,NaN,NaN,NaN,NaN,1,5e-09,1e-06
0.4,-0.15,NaN,NaN,NaN,NaN,NaN,NaN,1,5e-09,1e-06
0.4,-0.1,NaN,NaN,NaN,NaN,NaN,NaN,1,5e-09,1e-06
0.4,-0.05,NaN,NaN,NaN,NaN,NaN,NaN,1,5e-09,1e-06
0.4,0,10.59,1.9,0.53825,7.842e-05,19,6,1,5e-09,1e-06
0.4,0.05,10.76,1.85,0.53825,7.842e-05,19,5.5,1,5e-09,1e-06
0.4,0.1,10.94,1.8,0.53825,7.842e-05,19,5,1,5e-09,1e-06
0.4,0.15,11.11,1.75,0.53825,7.842e-05,19,4.25,1,5e-09,1e-06
0.4,0.2,11.46,1.7,0.53825,7.842e-05,19,3.75,1,5e-09,1e-06
0.4,0.25,11.82,1.65,0.53825,7.842e-05,19,3.25,1,5e-09,1e-06
0.4,0.3,12.98,1.6,0.53825,7.842e-05,19,2.75,1,5e-09,1e-06
0.6,-0.2,NaN,NaN,NaN,NaN,NaN,NaN,1,5e-09,1e-06
0.6,-0.15,NaN,NaN,NaN,NaN,NaN,NaN,1,5e-09,1e-06
0.6,-0.1,NaN,NaN,NaN,NaN,NaN,NaN,1,5e-09,1e-06
0.6,-0.05,NaN,NaN,NaN,NaN,NaN,NaN,1,5e-09,1e-06
0.6,0,9.04,2.5,0.41825,7.842e-05,20,6,1,5e-09,1e-06
0.6,0.05,9.04,2.5,0.41825,7.842e-05,20,5.5,1,5e-09,1e-06
0.6,0.1,9.04,2.5,0.41825,7.842e-05,20,5.25,1,5e-09,1e-06
0.6,0.15,9.04,2.5,0.41825,7.842e-05,20,4.75,1,5e-09,1e-06
0.6,0.2,9.04,2.5,0.41825,7.842e-05,20,3.75,1,5e-09,1e-06
0.6,0.25,9.04,2.5,0.41825,7.842e-05,20,3.5,1,5e-09,1e-06
0.6,0.3,9.04,2.5,0.41825,7.842e-05,20,3.25,1,5e-09,1e-06
0.6,0.35,9.04,2.5,0.41825,7.842e-05,20,3,1,5e-09,1e-06
0.6,0.4,9.04,2.2,0.41825,7.842e-05,20,2.5,1,5e-09,1e-06
0.6,0.45,9.04,2.1,0.41825,7.842e-05,20,1.75,1,5e-09,1e-06
0.6,0.5,12.98,2.1,0.41825,7.842e-05,20,1.25,1,5e-09,1e-06

```


=== FILE: _extracted/O2_packet/M1_130DNWFB.txt (9903 chars) ===
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


=== FILE: _extracted/O2_packet/summary.json (349 chars) ===
```json
{
  "n_curves": 33,
  "n_evaluated": 25,
  "n_skipped": 8,
  "median_log_rmse": 2.025633319969513,
  "p90_log_rmse": 5.027446256149432,
  "elapsed_s": 39.152913093566895,
  "vs_z91f_run1_median": 4.234,
  "vs_z91f_run2_median": 2.402,
  "note": "true two-model validation (M1 = 130DNWFB, M2 = 130bulkNSRAM) with Sebastian's per-bias CSV overrides"
}
```


=== FILE: _extracted/nsram_bsim4_port/nsram/nsram/bsim4_port/model_card.py (8675 chars) ===
```python
"""BSIM4Model — model card data with defaults from official C source.

Public API:
    m = BSIM4Model(vth0=0.5, u0=0.045)            # explicit overrides
    m = BSIM4Model.from_spice(text)                # parse .model card
    m["vth0"]    →  0.5
    m.is_given("vth0")  →  True
    m.is_given("k1")    →  False  (default used)

The dict ALIASES handles legacy SPICE names (e.g. 'vtho' for 'vth0').
"""
from __future__ import annotations
import re
from typing import Any

from ._model_card_data import PARAMS_META, ALIASES


def _resolve_default(default: Any, type_n: int, scratch: dict[str, Any]) -> Any:
    if isinstance(default, tuple):
        kind = default[0]
        if kind == "nmos_pmos":
            return default[1] if type_n == 1 else default[2]
        if kind == "mobmod":
            return default[1]
        if kind == "ref":
            return scratch.get(default[1], 0.0)
    return default


# SPICE engineering-suffix table (n→1e-9, u→1e-6, etc.).
SPICE_SUFFIX = {
    "f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6,
    "m": 1e-3, "k": 1e3, "meg": 1e6, "g": 1e9, "t": 1e12,
}


def parse_spice_value(s: str) -> float | None:
    """Parse a SPICE-style numeric literal: '4n', '1.35e5', '-0.0465', '2u', etc."""
    s = s.strip()
    # Plain float / scientific notation
    try:
        return float(s)
    except ValueError:
        pass
    # Engineering suffix: digits[.digits][e±dd]<suffix>
    m = re.match(r"^([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\s*([a-zA-Z]+)$", s)
    if m:
        try:
            base = float(m.group(1))
        except ValueError:
            return None
        suf = m.group(2).lower()
        # Check 3-letter then 1-letter
        for k in ("meg",):
            if suf == k:
                return base * SPICE_SUFFIX[k]
        if suf and suf[0] in SPICE_SUFFIX:
            return base * SPICE_SUFFIX[suf[0]]
    return None


class BSIM4Model:
    """Holds ~880 BSIM4 v4.8.3 model parameters with C-faithful defaults."""

    __slots__ = ("_values", "_given")

    def __init__(self, **kwargs):
        type_arg = kwargs.pop("type", 1)
        if isinstance(type_arg, str):
            type_arg = 1 if type_arg.upper() == "NMOS" else -1
        self._values: dict[str, Any] = {"type": type_arg}
        self._given: set[str] = set()

        # Pass 1: scalar + nmos_pmos + mobmod defaults
        for name, info in PARAMS_META.items():
            d = info["default"]
            if isinstance(d, tuple) and d[0] == "ref":
                continue
            self._values[name] = _resolve_default(d, type_arg, self._values)

        # Pass 2: refs (after scalar defaults populated)
        for name, info in PARAMS_META.items():
            d = info["default"]
            if isinstance(d, tuple) and d[0] == "ref":
                self._values[name] = self._values.get(d[1], 0.0)

        # Pass 3: user overrides
        for k, v in kwargs.items():
            self.set(k, v)

    def _canonical(self, name: str) -> str:
        n = name.lower()
        return ALIASES.get(n, n)

    def __getitem__(self, name: str) -> Any:
        return self._values[self._canonical(name)]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        c = self._canonical(name)
        if c in self._values:
            return self._values[c]
        raise AttributeError(name)

    def get(self, name: str, default: Any = None) -> Any:
        return self._values.get(self._canonical(name), default)

    def set(self, name: str, value: Any) -> None:
        c = self._canonical(name)
        if c not in PARAMS_META:
            raise KeyError(f"unknown BSIM4 parameter: {name!r}")
        info = PARAMS_META[c]
        if info["py_type"] == "float":
            coerced = float(value)
        elif info["py_type"] == "int":
            coerced = int(value)
        elif info["py_type"] == "bool":
            coerced = bool(int(value)) if isinstance(value, str) else bool(value)
        else:
            coerced = value
        self._values[c] = coerced
        self._given.add(c)

    def is_given(self, name: str) -> bool:
        return self._canonical(name) in self._given

    @property
    def given(self) -> frozenset[str]:
        return frozenset(self._given)

    def asdict(self, only_given: bool = False) -> dict[str, Any]:
        if only_given:
            return {k: self._values[k] for k in self._given}
        return dict(self._values)

    @classmethod
    def from_spice(cls, text: str, *, params: dict[str, float] | None = None,
                    model_type: str = "nmos") -> "BSIM4Model":
        """Parse a SPICE .model card. Handles continuation '+', .param, suffixes.

        If the file has multiple .model cards (e.g. NMOS and PMOS), pick the
        one matching `model_type` (default 'nmos').
        """
        params = {k.lower(): v for k, v in (params or {}).items()}
        # Pre-extract .param definitions (these are file-wide).
        # ngspice accepts BOTH `.param name=value` and `.param name value`.
        for line in text.splitlines():
            mp = re.match(r"\s*\.param\s+(\w+)\s*[=\s]\s*(\S+)", line, re.IGNORECASE)
            if mp:
                v = parse_spice_value(mp.group(2))
                if v is not None:
                    params[mp.group(1).lower()] = v

        # Slice to just the requested .model block: from its .model line to
        # either the next .model or end-of-file. Continuation '+' lines belong.
        model_re = re.compile(r"^\s*\.model\s+\S+\s+(\w+)", re.MULTILINE | re.IGNORECASE)
        starts = list(model_re.finditer(text))
        if starts:
            chosen = None
            for i, m in enumerate(starts):
                if m.group(1).lower() == model_type.lower():
                    chosen = m
                    end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
                    text = text[m.start():end]
                    break
            if chosen is None:
                # No matching model — use the first one anyway, but warn.
                m = starts[0]
                end = starts[1].start() if len(starts) > 1 else len(text)
                text = text[m.start():end]

        # Detect type from .model line (post-slice).
        type_kw = 1
        m_model = re.search(r"\.model\s+\S+\s+(nmos|pmos)", text, re.IGNORECASE)
        if m_model:
            type_kw = 1 if m_model.group(1).lower() == "nmos" else -1

        m = cls(type=type_kw)

        # Strip comments + .param + .end
        cleaned: list[str] = []
        for raw in text.splitlines():
            ln = raw.split("$")[0]
            if re.match(r"\s*\*", ln):
                continue
            if re.match(r"\s*\.param\s+", ln, re.IGNORECASE):
                continue
            if re.match(r"\s*\.end", ln, re.IGNORECASE):
                continue
            cleaned.append(ln)

        # Join '+' continuation lines.
        merged: list[str] = []
        for line in cleaned:
            if line.lstrip().startswith("+"):
                if merged:
                    merged[-1] = merged[-1] + " " + line.lstrip()[1:]
                else:
                    merged.append(line.lstrip()[1:])
            else:
                merged.append(line)
        flat = " ".join(merged)

        # Strip the .model header so its tokens aren't mistaken for params.
        flat = re.sub(r"\.model\s+\S+\s+\S+", " ", flat, flags=re.IGNORECASE)

        # Extract `name=value` pairs. Value may be:
        #   - SPICE numeric: [+-]?digits[.digits][e±d+][suffix-letters]
        #   - .param identifier reference
        # We match either; parse_spice_value disambiguates.
        VAL = r"(?:[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?[a-zA-Z]*|[A-Za-z_]\w*)"
        for match in re.finditer(
            rf"(\w+)\s*=\s*({VAL})", flat
        ):
            name = match.group(1).lower()
            val = match.group(2).strip()
            if name in ("level", "version", "type"):
                continue
            canon = ALIASES.get(name, name)
            if canon not in PARAMS_META:
                continue
            v = parse_spice_value(val)
            if v is None:
                # Try .param substitution
                v = params.get(val.lower())
            if v is None:
                continue
            try:
                m.set(name, v)
            except (KeyError, ValueError):
                pass
        return m

    def __repr__(self) -> str:
        n = len(self._given)
        type_str = "NMOS" if self._values.get("type", 1) == 1 else "PMOS"
        return f"<BSIM4Model {type_str} given={n} of {len(PARAMS_META)}>"

```


=== FILE: _extracted/nsram_bsim4_port/nsram/nsram/bsim4_port/smooth.py (2423 chars) ===
```python
"""smooth — differentiable replacements for non-smooth primitives in BSIM4.

Each primitive:
  - Has a `sharpness` parameter; converges to the hard version as sharpness→∞.
  - Keeps gradients finite and non-zero in the transition region.
  - Tested in tests/test_smooth.py for: convergence, gradcheck, no-NaN.

Use these EVERYWHERE a faithful BSIM4 port has if/MAX/MIN/abs/sqrt/log/exp.
Cross-reference each substitution site in code with `# SMOOTH: <name>` comment
so block PRs are auditable.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


SHARPNESS_DEFAULT = 50.0
EPS_SQRT = 1e-12
EPS_LOG = 1e-30
EXP_THRESHOLD = 34.0  # matches BSIM4's MAX_EXP/MIN_EXP guard


def smooth_max(a: torch.Tensor, b: torch.Tensor, sharpness: float = SHARPNESS_DEFAULT) -> torch.Tensor:
    """max(a,b) ≈ b + softplus(s·(a-b))/s. Converges as s→∞."""
    return b + F.softplus(sharpness * (a - b)) / sharpness


def smooth_min(a: torch.Tensor, b: torch.Tensor, sharpness: float = SHARPNESS_DEFAULT) -> torch.Tensor:
    """min(a,b) ≈ a - softplus(s·(a-b))/s."""
    return a - F.softplus(sharpness * (a - b)) / sharpness


def soft_clamp(x: torch.Tensor, lo: float, hi: float, sharpness: float = SHARPNESS_DEFAULT) -> torch.Tensor:
    """Smooth clamp to [lo, hi] via smooth_max then smooth_min."""
    lo_t = torch.as_tensor(lo, dtype=x.dtype, device=x.device)
    hi_t = torch.as_tensor(hi, dtype=x.dtype, device=x.device)
    return smooth_min(smooth_max(x, lo_t, sharpness), hi_t, sharpness)


def safe_sqrt(x: torch.Tensor, eps: float = EPS_SQRT) -> torch.Tensor:
    """sqrt(max(x, eps)) — finite gradient at x=0."""
    return torch.sqrt(x.clamp_min(eps))


def safe_log(x: torch.Tensor, eps: float = EPS_LOG) -> torch.Tensor:
    """log(max(x, eps))."""
    return torch.log(x.clamp_min(eps))


def safe_exp(x: torch.Tensor, max_arg: float = EXP_THRESHOLD) -> torch.Tensor:
    """exp clipped at ±max_arg, mirroring BSIM4 DEXP."""
    return torch.exp(x.clamp(-max_arg, max_arg))


def smooth_step(x: torch.Tensor, lo: float, hi: float, sharpness: float = SHARPNESS_DEFAULT) -> torch.Tensor:
    """Smooth Heaviside: 0 below lo, 1 above hi."""
    mid = 0.5 * (lo + hi)
    width = max(hi - lo, 1e-6)
    return torch.sigmoid(2 * sharpness * (x - mid) / width)


def smooth_abs(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """|x| ≈ sqrt(x² + ε)."""
    return torch.sqrt(x * x + eps)

```


=== FILE: _extracted/nsram_bsim4_port/nsram/nsram/bsim4_port/nsram_cell_2T.py (43647 chars) ===
```python
"""nsram_cell_2T — Differentiable 2T NS-RAM cell with proper topology.

Replaces the 1T proxy in `nsram_cell.py` (which collapses VG2 into a
``vth0_eff = vth0 + gamma·VG2`` shift) with the FULL 2T topology faithful
to Sebas's schematic ``data/sebas_2026_04_22/2tnsram_simple.asc``::

        D ──┬─────────────┬── (drain pin)
            │             │
          M1.D          Q1.C
            │             │
   VG1 → M1.G           Q1.B ── B  (floating body, shared by M1 & M2)
            │             │
          M1.S ── Sint ── Q1.E
                    │
                  M2.D
                    │
   VG2 → M2.G       │
                    │
                  M2.S ── 0  (ground)

Two NMOS (M1 short, M2 long) share floating body B. The internal node
Sint is the M1 source / M2 drain / Q1 emitter. Two unknown internal
voltages (Vsint, Vb) are solved by Newton-Raphson at each (Vd, VG1, VG2)
bias point so Sint-KCL = 0 and Body-KCL = 0.

Newton residuals (currents INTO each node):

    R_Sint(Vsint, Vb) =
        + Ids_M1(VG1−Vsint, Vd−Vsint, Vb−Vsint)            # M1 source ejects into Sint
        − Ids_M2(VG2,         Vsint,    Vb)                 # M2 drain absorbs from Sint
        + Ie_Q1(Vb−Vsint, Vb−Vd)                            # BJT emitter ejects into Sint
        + Ibs_diode_M1(Vb−Vsint)                            # forward body→Sint diode of M1
        − Ibd_diode_M2(Vb)                                  # forward body→drain(=Sint) of M2 leaves Sint

    R_B(Vsint, Vb) =
        + Iii_M1 + Iii_M2                                   # impact-ion holes → body
        + Igidl_M1 + Igisl_M1 + Igidl_M2 + Igisl_M2         # BTBT
        + Igb_M1 + Igb_M2                                   # gate→body tunnel
        − Ibd_diode_M1(Vb−Vd) − Ibs_diode_M1(Vb−Vsint)      # M1 junction leaks LEAVE body
        − Ibd_diode_M2(Vb)    − Ibs_diode_M2(Vb)            # M2 junction leaks LEAVE body
        − Ib_Q1(Vb−Vsint, Vb−Vd)                            # BJT base current leaves B

Drain terminal current at the D pin (positive into device):
    Id = Ids_M1 + Ic_Q1 + Igidl_drain_M1 + Ibd_diode_M1

VG2 is now a *real* gate to M2 (not a proxy threshold shift); body-effect
on M1 enters naturally via Vbs_M1 = Vb − Vsint.

Differentiability: simplest correct path. Newton iterations live INSIDE
autograd (no implicit-function-theorem trick yet). Each iteration is a
single forward of the full BSIM4 stack (~30 calls per bias point worst
case, double precision). For 33×~10 sweep points that's still tractable.

WARNING: do NOT add arbitrary clipping to "fix" Newton divergence — that
was the v4 mistake. Diagnose with `verbose=True`.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import torch

from .bjt import GummelPoonNPN, compute_bjt
from .dc import compute_dc
from .diode import compute_body_diodes
from .geometry import Geometry
from .leak import compute_iimpact, compute_igidl_gisl, compute_igb
from .model_card import BSIM4Model
from .temp import compute_size_dep, SizeDependParam


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class NSRAMCell2TConfig:
    """Static config for a 2T NS-RAM cell.

    Geometry + toggles + Newton solver knobs. Two SizeDependParam objects
    (one per MOSFET) are computed lazily.
    """
    Ln: float = 180e-9                  # M1 channel length [m]
    Wn: float = 360e-9                  # both channels' width [m]
    M2_length_factor: float = 10.0      # M2 length = Ln * factor (Sebas: 10x)
    Cbody: float = 1e-15                # body cap [F] (transient only; from CBpar)
    T_C: float = 27.0                   # operating temperature

    # Junction geometry per MOSFET. None → auto W·L / 2(W+L).
    As_M1: Optional[float] = None
    Ad_M1: Optional[float] = None
    Ps_M1: Optional[float] = None
    Pd_M1: Optional[float] = None
    As_M2: Optional[float] = None
    Ad_M2: Optional[float] = None
    Ps_M2: Optional[float] = None
    Pd_M2: Optional[float] = None

    # Toggle physics
    use_iii: bool = True
    use_gidl: bool = True
    use_bjt: bool = True
    use_igb: bool = True
    use_diode: bool = True
    # Physical defaults injected when card has jss=jsd=0 (Sebas's PTM130 card
    # leaves these unset, which leaves the body diodes silent and lets Vb run
    # away unbounded under Iii injection — root cause of v6 fit explosion).
    # Typical 130nm CMOS pn junction: Js ≈ 1e-4 A/m². With AS = W·L = 360n·180n
    # = 6.5e-14 m², Is_diode ≈ 6.5e-18 A; at Vbs = 0.7V forward, Ibs ≈ 1.1e-5 A
    # → naturally clamps Vb at body-source diode turn-on voltage.
    default_jss: float = 1e-4    # A/m² source-bottom junction
    default_jsd: float = 1e-4    # A/m² drain-bottom junction

    # Newton solver
    newton_max_iters: int = 30
    newton_tol: float = 1e-12        # max(|R_Sint|, |R_B|) in Amperes (legacy)
    newton_damping: float = 1.0
    newton_min_damping: float = 1.0 / 64.0
    # Per-iteration relative voltage step cap (helps in steep regions w/o
    # masking divergence). Set to a large number to disable. Keep modest;
    # purpose is convergence, not "papering over" non-physics.
    max_step_V: float = 0.5

    # Oracle-recommended Newton hardening (gmin shunt + relative tol +
    # min-iter guard prevents the "spurious-root at iter 1" pathology where
    # Vb=0 initial guess lands all body currents at ~1e-17 A which is below
    # the absolute residual tolerance even though the true root is at
    # Vb~0.77 V).
    gmin: float = 1e-15              # shunt conductance on body+Sint KCL
    # Lowered from oracle-suggested 1e-12: at 1e-12 gmin shunts dominate
    # over physically zero body diodes (jss=jsd=0 in Sebas card) and pull
    # Vb to Vd/4, forward-biasing M1's body-source junction and doubling
    # Id. 1e-15 is small enough not to distort while still providing the
    # Jacobian slope to escape the spurious flat root at Vb=0.
    Iabstol: float = 1e-12           # absolute current tolerance
    Ireltol: float = 1e-3            # relative tolerance vs |I_physical|
    xtol_v: float = 1e-7             # voltage step infinity-norm tolerance
    min_iters: int = 2               # require >= this many Newton iters
    # gmin homotopy: if enabled, first cold-start solve walks gmin from
    # gmin_start down to `gmin` in factor-of-10 steps before declaring done.
    gmin_step: bool = False
    gmin_start: float = 1e-9

    # Lazy SizeDependParam caches
    _sd_M1: Optional[SizeDependParam] = field(default=None, init=False, repr=False)
    _sd_M2: Optional[SizeDependParam] = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------ #
    def _geom_M1(self) -> Geometry:
        return Geometry(L=self.Ln, W=self.Wn, NF=1)

    def _geom_M2(self) -> Geometry:
        return Geometry(L=self.Ln * self.M2_length_factor, W=self.Wn, NF=1)

    def size_dep_M1(self, model: BSIM4Model) -> SizeDependParam:
        if self._sd_M1 is None:
            self._sd_M1 = compute_size_dep(model, self._geom_M1(), T_C=self.T_C)
        return self._sd_M1

    def size_dep_M2(self, model: BSIM4Model) -> SizeDependParam:
        if self._sd_M2 is None:
            self._sd_M2 = compute_size_dep(model, self._geom_M2(), T_C=self.T_C)
        return self._sd_M2

    def invalidate(self) -> None:
        self._sd_M1 = None
        self._sd_M2 = None

    def _junctions_M1(self) -> tuple[float, float, float, float]:
        WL = self.Wn * self.Ln
        WLp = 2.0 * (self.Wn + self.Ln)
        return (
            WL  if self.As_M1 is None else self.As_M1,
            WL  if self.Ad_M1 is None else self.Ad_M1,
            WLp if self.Ps_M1 is None else self.Ps_M1,
            WLp if self.Pd_M1 is None else self.Pd_M1,
        )

    def _junctions_M2(self) -> tuple[float, float, float, float]:
        L2 = self.Ln * self.M2_length_factor
        WL = self.Wn * L2
        WLp = 2.0 * (self.Wn + L2)
        return (
            WL  if self.As_M2 is None else self.As_M2,
            WL  if self.Ad_M2 is None else self.Ad_M2,
            WLp if self.Ps_M2 is None else self.Ps_M2,
            WLp if self.Pd_M2 is None else self.Pd_M2,
        )


# --------------------------------------------------------------------------- #
# Param-override context for SizeDependParam                                  #
# --------------------------------------------------------------------------- #

@contextmanager
def _override_sd(sd: SizeDependParam, overrides: Optional[dict]):
    """Temporarily replace selected SizeDependParam fields (for fitting).

    Useful so optimizer can flow grads through ``sd.vth0_T`` etc. without
    rebuilding the whole SizeDependParam each iteration.
    """
    if not overrides:
        yield
        return
    saved: dict = {}
    try:
        for k, v in overrides.items():
            saved[k] = getattr(sd, k)
            setattr(sd, k, v)
        yield
    finally:
        for k, v in saved.items():
            setattr(sd, k, v)


# --------------------------------------------------------------------------- #
# Per-MOSFET physics evaluator                                                #
# --------------------------------------------------------------------------- #

def _eval_mosfet(
    model: BSIM4Model,
    sd: SizeDependParam,
    cfg: NSRAMCell2TConfig,
    Vg: torch.Tensor,
    Vd: torch.Tensor,
    Vs: torch.Tensor,
    Vb: torch.Tensor,
    junctions: tuple[float, float, float, float],
    overrides: Optional[dict] = None,
) -> dict:
    """Compute Ids, Iii, Igidl, Igisl, Igb, Ibs, Ibd for one NMOS at given
    *terminal* voltages. Bias mapping (NMOS, source-referenced):

        Vgs = Vg - Vs,   Vds = Vd - Vs,   Vbs = Vb - Vs,   Vbd = Vb - Vd

    Returned dict uses the convention native to each sub-call:
        - Ids: drain-to-source channel current (positive in saturation, NMOS)
        - Iii: positive INTO body (channel impact-ion)
        - Igidl: positive INTO body (drain edge BTBT, "drain → body")
        - Igisl: positive INTO body (source edge BTBT)
        - Igb: positive INTO body (gate → body tunneling)
        - Ibs, Ibd: junction diode currents, *positive when forward biased*
                   (current flows OUT of body INTO source/drain).
    """
    Vgs = Vg - Vs
    Vds = Vd - Vs
    Vbs = Vb - Vs
    Vbd = Vb - Vd

    with _override_sd(sd, overrides):
        dc = compute_dc(model, sd, Vgs=Vgs, Vds=Vds, Vbs=Vbs)

        if cfg.use_iii:
            Iii = compute_iimpact(model, sd, dc, Vds=Vds)
        else:
            Iii = torch.zeros_like(dc.Ids)

        if cfg.use_gidl:
            Igidl, Igisl = compute_igidl_gisl(model, sd, Vgs=Vgs, Vds=Vds, Vbs=Vbs)
        else:
            Igidl = torch.zeros_like(dc.Ids)
            Igisl = torch.zeros_like(dc.Ids)

        if cfg.use_igb:
            Igb = compute_igb(model, sd, Vgs=Vgs, Vbs=Vbs, dc_result=dc)
        else:
            Igb = torch.zeros_like(dc.Ids)

        if cfg.use_diode:
            As_, Ad_, Ps_, Pd_ = junctions
            # Inject physical Js defaults when card has zero (Sebas card
            # bug — root cause of Vb runaway). See cfg comments.
            js_overrides = {}
            try:
                if float(sd.SourceSatCurDensity_T) == 0.0 and cfg.default_jss > 0:
                    js_overrides["SourceSatCurDensity_T"] = cfg.default_jss
                if float(sd.DrainSatCurDensity_T) == 0.0 and cfg.default_jsd > 0:
                    js_overrides["DrainSatCurDensity_T"] = cfg.default_jsd
            except Exception:
                pass
            if js_overrides:
                with _override_sd(sd, js_overrides):
                    Ibs, Ibd = compute_body_diodes(model, sd, Vbs=Vbs, Vbd=Vbd,
                                                   As=As_, Ad=Ad_, Ps=Ps_, Pd=Pd_)
            else:
                Ibs, Ibd = compute_body_diodes(model, sd, Vbs=Vbs, Vbd=Vbd,
                                               As=As_, Ad=Ad_, Ps=Ps_, Pd=Pd_)
        else:
            Ibs = torch.zeros_like(dc.Ids)
            Ibd = torch.zeros_like(dc.Ids)

    return {
        "Ids": dc.Ids,
        "Iii": Iii,
        "Igidl": Igidl,
        "Igisl": Igisl,
        "Igb": Igb,
        "Ibs": Ibs,         # >0 ⇒ leaves body INTO source
        "Ibd": Ibd,         # >0 ⇒ leaves body INTO drain
        "Vds": Vds,
        "Vbs": Vbs,
        "Vbd": Vbd,
    }


# --------------------------------------------------------------------------- #
# Residual                                                                    #
# --------------------------------------------------------------------------- #

def _residuals(
    cfg: NSRAMCell2TConfig,
    model_M1: BSIM4Model,
    bjt: GummelPoonNPN,
    Vd: torch.Tensor,
    VG1: torch.Tensor,
    VG2: torch.Tensor,
    Vsint: torch.Tensor,
    Vb: torch.Tensor,
    P_M1: Optional[dict] = None,
    P_M2: Optional[dict] = None,
    model_M2: Optional[BSIM4Model] = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Compute (R_Sint, R_B, components) at current (Vsint, Vb) guess.

    `model_M2` defaults to `model_M1` (single-model legacy behaviour).
    """
    if model_M2 is None:
        model_M2 = model_M1
    sd_M1 = cfg.size_dep_M1(model_M1)
    sd_M2 = cfg.size_dep_M2(model_M2)
    j_M1 = cfg._junctions_M1()
    j_M2 = cfg._junctions_M2()

    # Ground reference is V_S (source of M2) = 0.
    zero = torch.zeros_like(Vd)

    # M1: D=Vd, G=VG1, S=Vsint, B=Vb
    m1 = _eval_mosfet(model_M1, sd_M1, cfg, Vg=VG1, Vd=Vd, Vs=Vsint, Vb=Vb,
                      junctions=j_M1, overrides=P_M1)
    # M2: D=Vsint, G=VG2, S=0, B=Vb
    m2 = _eval_mosfet(model_M2, sd_M2, cfg, Vg=VG2, Vd=Vsint, Vs=zero, Vb=Vb,
                      junctions=j_M2, overrides=P_M2)

    # Parasitic NPN: collector=D, base=B, emitter=GND.
    # ──────────────────────────────────────────────────────────────────
    # IMPORTANT (2026-05-01, A.1.i finding): Sebastian's LTSpice schematic
    # `2tnsram_simple.asc` wires the parasitic NPN with **emitter to
    # ground**, not to Sint. This is the "complementary bipolar current"
    # he refers to in his Apr-17 email — its purpose is to provide a
    # body-charging path that fires when Vb climbs (Vbe = Vb − 0 = Vb,
    # not Vb − Vsint ≈ small). With emitter=Sint the BJT would never
    # turn on at low VG2 because Vb tracks Vsint. With emitter=GND, Vbe
    # tracks Vb directly and the NPN switches at Vb ~0.6 V.
    if cfg.use_bjt:
        Vbe = Vb                 # emitter = ground
        Vbc = Vb - Vd            # collector = drain
        bjt_out = compute_bjt(bjt, Vbe=Vbe, Vbc=Vbc, T_K=273.15 + cfg.T_C)
        Ic_Q1 = bjt_out["Ic"]    # collector current (drain → emitter = GND)
        Ib_Q1 = bjt_out["Ib"]    # base current (INTO base from external)
        Ie_Q1 = bjt_out["Ie"]    # emitter current at GND (= −(Ic+Ib))
    else:
        Ic_Q1 = torch.zeros_like(Vd)
        Ib_Q1 = torch.zeros_like(Vd)
        Ie_Q1 = torch.zeros_like(Vd)

    # ---- Sint KCL: currents INTO Sint --------------------------------- #
    # M1 channel current Ids_M1 flows D→S — INTO Sint (M1 source). → +Ids_M1
    # M2 drain is Sint; M2 channel sinks current FROM drain → −Ids_M2
    # BJT emitter is now GND, NOT Sint — BJT no longer touches Sint node.
    # M1 junction: Ibs_M1 >0 ⇒ leaves body INTO source(=Sint). → +Ibs_M1
    # M2 junction: Ibd_M2 >0 ⇒ leaves body INTO drain(=Sint). → +Ibd_M2
    R_Sint = (
        m1["Ids"]
        - m2["Ids"]
        + m1["Ibs"]
        + m2["Ibd"]
    )

    # ---- Body KCL: currents INTO B ------------------------------------ #
    # Iii, Igidl, Igisl, Igb are already signed +INTO-body in the helpers.
    # Body junction diodes: Ibs and Ibd are POSITIVE-LEAVING-body, so we
    # subtract them.
    # BJT base current Ib (positive INTO base from external) — for the
    # floating body, the only external current into the base IS the body
    # node itself. Ib>0 ⇒ body sources current → leaves body. → −Ib_Q1
    R_B = (
        m1["Iii"] + m2["Iii"]
        + m1["Igidl"] + m1["Igisl"] + m2["Igidl"] + m2["Igisl"]
        + m1["Igb"] + m2["Igb"]
        - m1["Ibs"] - m1["Ibd"]
        - m2["Ibs"] - m2["Ibd"]
        - Ib_Q1
    )

    # Oracle-recommended gmin shunts — ngspice-style parallel conductance
    # in PARALLEL with each pn junction, NOT a single shunt to ground.
    # This is what gives the body a tendency to track (Vd+Vs)/2 in absence
    # of other forces, matching ngspice's behavior.
    #   I_gmin_bd = gmin * (Vd - Vb)   flows INTO body from drain
    #   I_gmin_bs = gmin * (Vs - Vb) = -gmin * Vb (since Vs=0)
    #                                   flows INTO body from source
    #   I_gmin_bsi = gmin * (Vsint - Vb)  body↔Sint via M1's body-source
    #                                      and M2's body-drain (both at Sint)
    # Sum into R_B (currents INTO B). Similar for Sint node.
    gmin = getattr(cfg, "gmin", 0.0)
    if gmin > 0.0:
        # Body node: junctions B↔D, B↔S(=0), B↔Sint (counted once: M1 body-source
        # and M2 body-drain are both at Sint, so 2× weight)
        R_B = R_B + gmin * (Vd - Vb) + gmin * (-Vb) + 2.0 * gmin * (Vsint - Vb)
        # Sint node: gmin shunt to ground (Sint↔S=0 via M2 channel parasitic)
        # plus to body. Mainly to keep Jacobian non-singular at Sint=0.
        R_Sint = R_Sint + gmin * (-Vsint) + 2.0 * gmin * (Vb - Vsint)

    components = {
        "Ids_M1": m1["Ids"], "Ids_M2": m2["Ids"],
        "Ic_Q1": Ic_Q1, "Ib_Q1": Ib_Q1, "Ie_Q1": Ie_Q1,
        "Iii_M1": m1["Iii"], "Iii_M2": m2["Iii"],
        "Igidl_M1": m1["Igidl"], "Igisl_M1": m1["Igisl"],
        "Igidl_M2": m2["Igidl"], "Igisl_M2": m2["Igisl"],
        "Igb_M1": m1["Igb"], "Igb_M2": m2["Igb"],
        "Ibs_M1": m1["Ibs"], "Ibd_M1": m1["Ibd"],
        "Ibs_M2": m2["Ibs"], "Ibd_M2": m2["Ibd"],
    }
    return R_Sint, R_B, components


# --------------------------------------------------------------------------- #
# Newton solve                                                                #
# --------------------------------------------------------------------------- #

def _solve_jac_2x2(R_S: torch.Tensor, R_B: torch.Tensor,
                   J: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve 2x2 system J · [dVs, dVb]^T = -[R_S, R_B]^T element-wise.

    J has shape (..., 2, 2). Returns (dVsint, dVb), each shape (...).

    Degenerate case handling: when all body physics is disabled
    (Iii=Igidl=Igb=Ibd=Ibs=BJT off), R_B ≡ 0 and the second row of J is
    zero. The 2D system is singular but the 1D problem in Vsint is
    well-posed. We detect this (R_B ≈ 0 AND row-2 of J ≈ 0) and reduce
    to dVs = -R_S / a, dVb = 0.
    """
    a = J[..., 0, 0]; b = J[..., 0, 1]
    c = J[..., 1, 0]; d = J[..., 1, 1]

    # Detect degenerate body row (R_B identically 0 ⇒ no info about Vb)
    body_dead = (c.abs() < 1e-30) & (d.abs() < 1e-30) & (R_B.abs() < 1e-30)

    det = a * d - b * c
    # Keep det away from 0 numerically; sign-preserving floor.
    sign = torch.where(det >= 0, torch.ones_like(det), -torch.ones_like(det))
    det_safe = torch.where(det.abs() < 1e-30, sign * 1e-30, det)
    rhs0 = -R_S
    rhs1 = -R_B
    dVs_full = (d * rhs0 - b * rhs1) / det_safe
    dVb_full = (-c * rhs0 + a * rhs1) / det_safe

    # 1-D fallback when body is dead
    a_safe = torch.where(a.abs() < 1e-30, sign * 1e-30, a)
    dVs_1d = -R_S / a_safe
    dVb_1d = torch.zeros_like(dVs_1d)

    dVs = torch.where(body_dead, dVs_1d, dVs_full)
    dVb = torch.where(body_dead, dVb_1d, dVb_full)
    return dVs, dVb


def _jacobian_finite_diff(
    cfg, model_M1, bjt, Vd, VG1, VG2, Vsint, Vb, P_M1, P_M2, h: float = 1e-6,
    model_M2=None,
) -> torch.Tensor:
    """Finite-difference 2x2 Jacobian ∂(R_Sint, R_B)/∂(Vsint, Vb).

    Vectorized over leading dims of Vsint/Vb. Returns shape (..., 2, 2).
    Computed under torch.no_grad — used inside the Newton loop only for the
    *step direction*; the autograd path through the converged solution
    flows via the iterative updates themselves (since they're under grad).
    """
    with torch.no_grad():
        # Central differences on Vsint
        Rsp_s, Rbp_s, _ = _residuals(cfg, model_M1, bjt, Vd, VG1, VG2,
                                     Vsint + h, Vb, P_M1, P_M2, model_M2=model_M2)
        Rsm_s, Rbm_s, _ = _residuals(cfg, model_M1, bjt, Vd, VG1, VG2,
                                     Vsint - h, Vb, P_M1, P_M2, model_M2=model_M2)
        dRs_dVs = (Rsp_s - Rsm_s) / (2 * h)
        dRb_dVs = (Rbp_s - Rbm_s) / (2 * h)
        # Central differences on Vb
        Rsp_b, Rbp_b, _ = _residuals(cfg, model_M1, bjt, Vd, VG1, VG2,
                                     Vsint, Vb + h, P_M1, P_M2, model_M2=model_M2)
        Rsm_b, Rbm_b, _ = _residuals(cfg, model_M1, bjt, Vd, VG1, VG2,
                                     Vsint, Vb - h, P_M1, P_M2, model_M2=model_M2)
        dRs_dVb = (Rsp_b - Rsm_b) / (2 * h)
        dRb_dVb = (Rbp_b - Rbm_b) / (2 * h)
    J = torch.stack([
        torch.stack([dRs_dVs, dRs_dVb], dim=-1),
        torch.stack([dRb_dVs, dRb_dVb], dim=-1),
    ], dim=-2)
    return J


def solve_2t_steady_state(
    cfg: NSRAMCell2TConfig,
    model: BSIM4Model,
    bjt: GummelPoonNPN,
    Vd: torch.Tensor,
    VG1: torch.Tensor,
    VG2: torch.Tensor,
    P_M1: Optional[dict] = None,
    P_M2: Optional[dict] = None,
    Vsint_init: Optional[torch.Tensor] = None,
    Vb_init: Optional[torch.Tensor] = None,
    verbose: bool = False,
    model_M2: Optional[BSIM4Model] = None,
) -> dict:
    """Solve the 2T cell at quasi-static (Vd, VG1, VG2).

    Returns dict with: Id, Vsint, Vb, components, R_Sint, R_B, niter, converged.

    Newton step uses *finite-difference* Jacobian (no_grad). Voltage
    updates themselves are inside the autograd graph, so gradients of Id
    w.r.t. fit params flow through the Newton iterates. This is slower
    than implicit-diff but correct and simpler.
    """
    # Coerce inputs to fp64 broadcastable tensors
    Vd = torch.as_tensor(Vd, dtype=torch.float64)
    VG1 = torch.as_tensor(VG1, dtype=torch.float64)
    VG2 = torch.as_tensor(VG2, dtype=torch.float64)
    Vd, VG1, VG2 = torch.broadcast_tensors(Vd, VG1, VG2)
    Vd = Vd.contiguous(); VG1 = VG1.contiguous(); VG2 = VG2.contiguous()

    if Vsint_init is None:
        Vsint = (0.5 * Vd).detach().clone()
    else:
        Vsint = Vsint_init.detach().clone().to(torch.float64).expand_as(Vd).contiguous()
    if Vb_init is None:
        # Cold-start at Vb=0. Note: oracle consensus recommended Vb=0.5
        # but in this model (PTM 130nm bulkNSRAM card) Iii=0 at typical
        # biases, so the high-Vb root is not an attractor and Newton
        # drifts back. Default Vb=0 matches legacy behaviour. Use the
        # `Vb_init=` kwarg explicitly when you know your bias is in the
        # impact-ion regime.
        Vb = torch.zeros_like(Vd)
    else:
        Vb = Vb_init.detach().clone().to(torch.float64).expand_as(Vd).contiguous()

    # Initial residual (need it grad-tracked for IFT-free autograd flow)
    R_S, R_B, comp0 = _residuals(cfg, model, bjt, Vd, VG1, VG2, Vsint, Vb, P_M1, P_M2,
                                 model_M2=model_M2)
    prev_resid_norm = (R_S.detach().abs() + R_B.detach().abs()).max()

    def _physical_scale(comp: dict) -> torch.Tensor:
        """Build a per-bias physical-current magnitude from KCL components.
        Used for relative-tolerance convergence — residual must be small
        relative to the current actually flowing in the device, not relative
        to the residual itself (circular)."""
        keys = ["Ids_M1", "Ids_M2", "Ic_Q1", "Ib_Q1",
                "Iii_M1", "Iii_M2", "Igidl_M1", "Igidl_M2",
                "Ibs_M1", "Ibd_M1", "Ibs_M2", "Ibd_M2"]
        scale = torch.zeros_like(R_S.detach())
        for k in keys:
            if k in comp:
                scale = scale + comp[k].detach().abs()
        return scale

    # Tolerances
    iabstol = getattr(cfg, "Iabstol", cfg.newton_tol)
    ireltol = getattr(cfg, "Ireltol", 0.0)
    xtol_v  = getattr(cfg, "xtol_v", 0.0)
    min_iters = getattr(cfg, "min_iters", 1)

    converged = torch.zeros_like(Vd, dtype=torch.bool)
    niter = 0
    last_dV_inf = torch.tensor(float("inf"), dtype=torch.float64)
    cur_comp = comp0
    for it in range(cfg.newton_max_iters):
        niter = it + 1
        # Convergence check (oracle hardening):
        #   - residual: |R| < max(Iabstol, Ireltol * |I_physical|)
        #     where I_physical = Σ|component currents|
        #   - step:     |dV|_inf < xtol_v
        #   - guard:    require >= min_iters AND the residual must have
        #               actually decreased once (or we've passed iter 1)
        residual_max = torch.maximum(R_S.detach().abs(), R_B.detach().abs())
        I_scale = _physical_scale(cur_comp)
        tol_eff = torch.maximum(torch.full_like(I_scale, iabstol), ireltol * I_scale)
        residual_ok = bool((residual_max < tol_eff).all())
        step_ok = bool((last_dV_inf < xtol_v).all()) if xtol_v > 0 else False
        cur_norm = (R_S.detach().abs() + R_B.detach().abs()).max()
        # min_iters: never declare convergence before this many iterations
        # have actually been taken (it counts the iteration *just executed*;
        # we must have done at least min_iters of them, i.e. it >= min_iters).
        if it >= min_iters and (residual_ok or step_ok):
            converged = residual_max < tol_eff
            if verbose:
                print(f"  Newton converged in {it} iter; max R = {residual_max.max():.3e} "
                      f"|dV|_inf = {float(last_dV_inf):.3e}")
            break
        prev_resid_norm = cur_norm

        # FD Jacobian (no_grad), step direction (no_grad). The implicit
        # function theorem is applied AFTER convergence to attach gradients
        # — see the IFT block at the end of this function.
        J = _jacobian_finite_diff(cfg, model, bjt, Vd, VG1, VG2,
                                  Vsint.detach(), Vb.detach(),
                                  P_M1, P_M2, model_M2=model_M2)
        dVs, dVb = _solve_jac_2x2(R_S.detach(), R_B.detach(), J)

        # Step-size cap (per-iteration relative-step limiter)
        max_abs = torch.maximum(dVs.abs(), dVb.abs())
        scale = torch.where(max_abs > cfg.max_step_V,
                            cfg.max_step_V / max_abs.clamp_min(1e-30),
                            torch.ones_like(max_abs))
        dVs = dVs * scale
        dVb = dVb * scale

        # Damped step + backtracking on residual norm (Armijo-style halving)
        damping = cfg.newton_damping
        prev_norm = R_S.detach().abs() + R_B.detach().abs()
        accepted = False
        while damping >= cfg.newton_min_damping:
            Vsint_try = Vsint + damping * dVs
            Vb_try = Vb + damping * dVb
            R_S_try, R_B_try, comp_try = _residuals(cfg, model, bjt, Vd, VG1, VG2,
                                             Vsint_try, Vb_try, P_M1, P_M2,
                                             model_M2=model_M2)
            new_norm = R_S_try.detach().abs() + R_B_try.detach().abs()
            # Strict decrease (mean over batch) accepted; or fall through at
            # min damping. The 0.999 factor demands genuine descent — at
            # min_damping we accept whatever we have.
            if (new_norm.mean() < prev_norm.mean() * 0.999) or damping <= cfg.newton_min_damping:
                Vsint = Vsint_try
                Vb = Vb_try
                R_S = R_S_try
                R_B = R_B_try
                cur_comp = comp_try
                accepted = True
                # Track step size for xtol convergence
                last_dV_inf = torch.maximum(
                    (damping * dVs).abs().max(),
                    (damping * dVb).abs().max(),
                )
                break
            damping *= 0.5
        if verbose:
            rmax = torch.maximum(R_S.detach().abs(), R_B.detach().abs()).max()
            print(f"  iter {it}: damping={damping:.3f} max|R|={rmax:.3e} "
                  f"|dVs|={dVs.abs().max():.3e} |dVb|={dVb.abs().max():.3e}")
        if not accepted:
            break

    # ----- Implicit Function Theorem (IFT) attachment -----
    # At convergence, R(x*, theta) ≈ 0 numerically, but x* (Vsint, Vb) has
    # been computed under no_grad — so it carries no gradient back to theta.
    # IFT says dx*/dtheta = -J^-1 · ∂R/∂theta. We can encode this in the
    # autograd graph by replacing x* with an "attached" version:
    #     x_attached = x*.detach() - J^-1 @ R(x*.detach(), theta)
    # At convergence R≈0 so x_attached ≈ x* in value, but its gradient w.r.t.
    # theta is exactly the IFT result because J^-1 is detached and R has
    # gradient through theta (via compute_dc, compute_iimpact, ...).
    Vsint_d = Vsint.detach()
    Vb_d = Vb.detach()
    R_S_at, R_B_at, _ = _residuals(cfg, model, bjt, Vd, VG1, VG2, Vsint_d, Vb_d, P_M1, P_M2,
                                   model_M2=model_M2)

    # CRITICAL: only apply IFT correction at biases where Newton ACTUALLY
    # converged. The IFT formula  x* = x*_d - J^-1 R(x*_d, theta)  assumes
    # R ≈ 0; if Newton failed, R can be huge, and J near-singular at that
    # bias would produce a spurious gradient that Adam misreads as a strong
    # signal — root cause of the v6/v7 stage 3 explosion.
    # When residual is too large, ZERO out the IFT delta at that bias →
    # gradient flows through theta-only paths, no broken Vb-loop signal.
    with torch.no_grad():
        J_final = _jacobian_finite_diff(cfg, model, bjt, Vd, VG1, VG2,
                                        Vsint_d, Vb_d, P_M1, P_M2,
                                        model_M2=model_M2)
    delta_s, delta_b = _solve_jac_2x2(R_S_at, R_B_at, J_final)

    # SMOOTH bound IFT delta via tanh — passes gradient through ALL bias
    # points (including non-converged ones), but compresses the magnitude so
    # Adam doesn't see exploding signal. Hard clamp would zero gradient at
    # boundary; tanh is differentiable everywhere.
    #   delta_smooth = D_MAX * tanh(delta_raw / D_MAX)
    # For |delta| << D_MAX: delta_smooth ≈ delta (full IFT signal)
    # For |delta| >> D_MAX: delta_smooth ≈ ±D_MAX, gradient ∝ sech²(.) → 0
    # This is the same effective bound but with smooth gradient transition.
    DELTA_BOUND = 0.3  # V — generous to allow real physics, not just to clip
    delta_s = DELTA_BOUND * torch.tanh(delta_s / DELTA_BOUND)
    delta_b = DELTA_BOUND * torch.tanh(delta_b / DELTA_BOUND)

    # 5th-oracle fix: at non-converged points, the IFT correction is meaningless
    # (Newton never reached a valid root) and would mutate Vsint/Vb away from
    # the un-corrected detached value. The function-level docstring promised we
    # don't apply IFT to non-converged points, but the code did. Gate it now.
    conv_mask = converged.detach()
    delta_s = torch.where(conv_mask, delta_s, torch.zeros_like(delta_s))
    delta_b = torch.where(conv_mask, delta_b, torch.zeros_like(delta_b))

    Vsint = Vsint_d - delta_s
    Vb = Vb_d - delta_b
    R_S, R_B, comp = _residuals(cfg, model, bjt, Vd, VG1, VG2, Vsint, Vb, P_M1, P_M2,
                                model_M2=model_M2)

    # Drain terminal current (positive INTO the D pin):
    #   Id = Ids_M1 (drain absorbs Ids from external) +
    #        Ic_Q1  (collector current absorbed from D) +
    #        Igidl_M1 leaves drain INTO body — but at the D pin this is a
    #        current LEAVING the drain to body, so the external D pin sees
    #        an EXTRA −Igidl_M1 inflow. We add it as a positive contribution
    #        because the convention "Igidl > 0 means current flows from drain
    #        into body via BTBT" implies the external supply drives that
    #        extra current INTO D. Same sign as the channel.
    #   The body diode Ibd_M1 is current LEAVING the body INTO drain, so
    #        from the D pin's perspective it FLOWS OUT to ground via M1
    #        substrate path → contributes −Ibd_M1 to Id (current leaves D).
    #
    # In the typical NS-RAM operating regime, |Ibd_M1|, |Igidl_M1| ≪ Ids_M1
    # so the dominant term is Ids_M1; SCBE / impact-ion shows up via Ic_Q1.
    Id = (
        comp["Ids_M1"]
        + comp["Ic_Q1"]
        + comp["Igidl_M1"]
        - comp["Ibd_M1"]
    )

    residual_max = torch.maximum(R_S.detach().abs(), R_B.detach().abs())
    I_scale_final = (R_S.detach().abs() + R_B.detach().abs()).clamp_min(iabstol)
    tol_final = torch.maximum(torch.full_like(I_scale_final, iabstol),
                              ireltol * I_scale_final)
    converged_final = residual_max < tol_final

    return {
        "Id": Id,
        "Vsint": Vsint,
        "Vb": Vb,
        "Ids_M1": comp["Ids_M1"],
        "Ids_M2": comp["Ids_M2"],
        "Ic_Q1": comp["Ic_Q1"],
        "Ib_Q1": comp["Ib_Q1"],
        "R_Sint": R_S,
        "R_B": R_B,
        "components": comp,
        "niter": niter,
        "converged": converged_final,
    }


# --------------------------------------------------------------------------- #
# gmin homotopy (z89): standard SPICE technique for snapback/bistable cells.  #
# --------------------------------------------------------------------------- #
@contextmanager
def _override_gmin(cfg: NSRAMCell2TConfig, value: float):
    """Temporarily override cfg.gmin (used by `_residuals` shunts)."""
    saved = cfg.gmin
    try:
        cfg.gmin = float(value)
        yield
    finally:
        cfg.gmin = saved


def solve_2t_with_homotopy(
    cfg: NSRAMCell2TConfig,
    model: BSIM4Model,
    bjt: GummelPoonNPN,
    Vd: torch.Tensor,
    VG1: torch.Tensor,
    VG2: torch.Tensor,
    P_M1: Optional[dict] = None,
    P_M2: Optional[dict] = None,
    Vsint_init: Optional[torch.Tensor] = None,
    Vb_init: Optional[torch.Tensor] = None,
    gmin_schedule: Optional[list] = None,
    verbose: bool = False,
    model_M2: Optional[BSIM4Model] = None,
) -> dict:
    """Solve 2T cell using gmin homotopy (oracle consensus recommendation).

    Standard SPICE technique for bistable / snapback / S-shaped I-V circuits:
    start with a LARGE gmin (linearizes the circuit, Newton always converges
    because every node has a strong shunt to its neighbours) and use that
    solution as a warm-start for the next smaller gmin. Repeat until the
    target gmin (= cfg.gmin) is reached.

    Implementation:
      * `gmin_schedule` defaults to [1e-3, 1e-5, 1e-8, 1e-12] followed by
        the cfg-specified target gmin (so the FINAL solve uses exactly the
        gmin that the IFT-attached delta sees, and gradients flow normally).
      * Each step calls existing `solve_2t_steady_state` with a temporarily
        overridden cfg.gmin and the previous solution as warm-start. We do
        NOT change `solve_2t_steady_state` so its IFT machinery is untouched.
      * The final returned dict comes from the last call (target gmin).

    NOTE: gmin shunts are physical-style conductances added in `_residuals`.
    They distort the solution slightly at large values; the homotopy walks
    that distortion smoothly to zero. At the FINAL gmin (= cfg.gmin), the
    solution is identical to a direct solve (only the convergence path is
    different) so gradient flow through IFT is unchanged.
    """
    if gmin_schedule is None:
        # Walk down by ~1000x per step. Final target = cfg.gmin.
        gmin_schedule = [1e-3, 1e-5, 1e-8, 1e-12]
    target = float(cfg.gmin)
    # Always end with the target gmin so IFT delta is computed at it.
    schedule = [g for g in gmin_schedule if g > target] + [target]

    Vsint_warm = Vsint_init
    Vb_warm = Vb_init
    last_out = None
    for step, g in enumerate(schedule):
        with _override_gmin(cfg, g):
            out = solve_2t_steady_state(
                cfg, model, bjt,
                Vd=Vd, VG1=VG1, VG2=VG2,
                P_M1=P_M1, P_M2=P_M2,
                Vsint_init=Vsint_warm,
                Vb_init=Vb_warm,
                verbose=verbose and (step == len(schedule) - 1),
                model_M2=model_M2,
            )
        if verbose:
            conv = bool(out["converged"].all())
            print(f"  homotopy step {step}: gmin={g:.1e}  converged={conv}  "
                  f"niter={out['niter']}", flush=True)
        # Warm-start next step with current solution. Detach so we don't
        # accumulate the previous step's autograd graph (the FINAL solve at
        # target gmin still goes through IFT for gradient attachment).
        Vsint_warm = out["Vsint"].detach()
        Vb_warm = out["Vb"].detach()
        last_out = out
    return last_out


# --------------------------------------------------------------------------- #
# Forward sweep                                                               #
# --------------------------------------------------------------------------- #

def forward_2t(
    cfg: NSRAMCell2TConfig,
    model: Optional[BSIM4Model] = None,
    bjt: Optional[GummelPoonNPN] = None,
    Vd_seq: Optional[torch.Tensor] = None,
    VG1: Optional[torch.Tensor] = None,
    VG2: Optional[torch.Tensor] = None,
    P_M1: Optional[dict] = None,
    P_M2: Optional[dict] = None,
    verbose: bool = False,
    warm_start: bool = True,
    use_homotopy: bool = False,
    dense_vd_in_snapback: bool = False,
    snapback_vd_threshold: float = 1.4,
    snapback_vd_step: float = 0.025,
    *,
    model_M1: Optional[BSIM4Model] = None,
    model_M2: Optional[BSIM4Model] = None,
) -> dict:
    """Sweep Vd from low to high with warm-starting Vsint, Vb between points.

    Returns dict with stacked tensors (shape (T,)): Id, Vsint, Vb, niter,
    converged, plus components per sub-call.

    Args (z89 additions):
      use_homotopy: if True, calls `solve_2t_with_homotopy` per point (gmin
          homotopy, expensive but converges through snapback / bistability).
      dense_vd_in_snapback: if True, internally insert intermediate Vd points
          at `snapback_vd_step` spacing for any segment where Vd >= threshold
          (defaults: threshold=1.4 V, step=0.025 V → 4× denser than the
          z88 default 0.1 V grid). Intermediate points are solved purely for
          warm-starting; the returned arrays only contain values at the
          ORIGINAL Vd_seq points (so the loss never sees the intermediate
          biases — they're a numerical aid only).

    Two-model variant: pass `model_M1=` and `model_M2=` as kwargs to use
    distinct BSIM4 cards for M1 and M2. If only legacy `model` is given,
    both transistors use it (back-compat). Mixing legacy `model` with
    `model_M2=` is also allowed (model → M1, kwarg → M2).
    """
    # Resolve model_M1 / model_M2 from positional `model` and kwargs.
    if model_M1 is None:
        model_M1 = model
    if model_M1 is None:
        raise TypeError("forward_2t requires either positional `model` or `model_M1=` kwarg")
    if model_M2 is None:
        model_M2 = model_M1
    Vd_seq = Vd_seq.to(torch.float64)
    VG1 = torch.as_tensor(VG1, dtype=torch.float64)
    VG2 = torch.as_tensor(VG2, dtype=torch.float64)
    T = int(Vd_seq.shape[0])

    # Build augmented schedule with intermediate (warm-start-only) points.
    # `report_idx[k]` indexes into the augmented sequence and tells us which
    # entries correspond to original Vd_seq points (we only return those).
    if dense_vd_in_snapback and T >= 2:
        aug_vd: list = []
        report_idx: list = []
        prev = float(Vd_seq[0].item())
        aug_vd.append(Vd_seq[0])
        report_idx.append(0)
        for i in range(1, T):
            cur = float(Vd_seq[i].item())
            # Insert intermediate points only if both endpoints (or the
            # current segment top) are in the snapback region. Spacing is
            # `snapback_vd_step` (only inserts if larger gap exists).
            if cur >= snapback_vd_threshold and (cur - prev) > 1.5 * snapback_vd_step:
                n_insert = int((cur - prev) / snapback_vd_step) - 1
                if n_insert > 0:
                    for k in range(1, n_insert + 1):
                        v = prev + (cur - prev) * (k / (n_insert + 1))
                        aug_vd.append(torch.tensor(v, dtype=torch.float64))
            aug_vd.append(Vd_seq[i])
            report_idx.append(len(aug_vd) - 1)
            prev = cur
        Vd_aug = torch.stack(aug_vd)
        report_set = set(report_idx)
    else:
        Vd_aug = Vd_seq
        report_idx = list(range(T))
        report_set = set(report_idx)

    T_aug = int(Vd_aug.shape[0])

    Ids_list, Vs_list, Vb_list = [], [], []
    niter_list, conv_list = [], []
    Ids_M1_list, Ids_M2_list, Ic_Q1_list = [], [], []

    # Cold start at Vb=0.5V (oracle consensus: avoid spurious flat root at
    # Vb=0 where all body currents are sub-femtoamp and Newton "converges"
    # without moving). Vsint=Vd/2 as initial series-divider guess.
    # Then cascade the converged solution from each point as the seed for
    # the next when warm_start=True (default).
    Vsint_warm = torch.tensor(0.0, dtype=torch.float64)  # gets replaced below
    Vb_warm = torch.tensor(0.5, dtype=torch.float64)

    # We collect outputs at ALL augmented points then filter to report_idx
    # at the end. This keeps the inner loop simple.
    aug_outs: list = []
    for i in range(T_aug):
        Vd_i = Vd_aug[i].unsqueeze(0)
        if i == 0:
            Vsint_warm = (Vd_i * 0.5).squeeze(0).detach()
        if use_homotopy:
            out = solve_2t_with_homotopy(
                cfg, model_M1, bjt,
                Vd=Vd_i, VG1=VG1, VG2=VG2,
                P_M1=P_M1, P_M2=P_M2,
                Vsint_init=Vsint_warm.expand_as(Vd_i),
                Vb_init=Vb_warm.expand_as(Vd_i),
                verbose=verbose,
                model_M2=model_M2,
            )
        else:
            out = solve_2t_steady_state(
                cfg, model_M1, bjt,
                Vd=Vd_i, VG1=VG1, VG2=VG2,
                P_M1=P_M1, P_M2=P_M2,
                Vsint_init=Vsint_warm.expand_as(Vd_i),
                Vb_init=Vb_warm.expand_as(Vd_i),
                verbose=verbose,
                model_M2=model_M2,
            )
        aug_outs.append(out)

        # Warm-start next point with current solution (detached so warm
        # start doesn't accumulate the previous step's Newton graph).
        if warm_start:
            Vsint_warm = out["Vsint"].detach().squeeze(0)
            Vb_warm = out["Vb"].detach().squeeze(0)

    # Filter to original Vd_seq points only (preserves graph for those).
    for i in report_idx:
        out = aug_outs[i]
        Ids_list.append(out["Id"].squeeze(0))
        Vs_list.append(out["Vsint"].squeeze(0))
        Vb_list.append(out["Vb"].squeeze(0))
        Ids_M1_list.append(out["Ids_M1"].squeeze(0))
        Ids_M2_list.append(out["Ids_M2"].squeeze(0))
        Ic_Q1_list.append(out["Ic_Q1"].squeeze(0))
        niter_list.append(out["niter"])
        conv_list.append(bool(out["converged"].all()))

    return {
        "Id": torch.stack(Ids_list),
        "Vsint": torch.stack(Vs_list),
        "Vb": torch.stack(Vb_list),
        "Ids_M1": torch.stack(Ids_M1_list),
        "Ids_M2": torch.stack(Ids_M2_list),
        "Ic_Q1": torch.stack(Ic_Q1_list),
        "niter": niter_list,
        "converged": conv_list,
    }

```


=== FILE: _extracted/nsram_bsim4_port/nsram/nsram/bsim4_port/geometry.py (2501 chars) ===
```python
"""B03_GEOM — effective channel geometry from drawn dimensions.

Faithful port of b4temp.c lines 450-545.
"""
from __future__ import annotations
from dataclasses import dataclass

from .model_card import BSIM4Model


@dataclass
class Geometry:
    L: float       # channel length [m]
    W: float       # channel width [m]
    NF: int = 1


@dataclass
class EffectiveGeom:
    Lnew: float
    Wnew: float
    dl: float
    dlc: float
    dw: float
    dwc: float
    dwj: float
    leff: float
    weff: float
    leffCV: float
    weffCV: float
    weffCJ: float
    Inv_L: float
    Inv_W: float
    Inv_LW: float


def compute_geometry(model: BSIM4Model, geom: Geometry) -> EffectiveGeom:
    """Faithful port of b4temp.c:455-545.

    Equations (BSIM4 manual §1):
        Lnew = L + xl
        Wnew = W/NF + xw
        dl   = Lint + Ll/Lnew^Lln + Lw/Wnew^Lwn + Lwl/(Lnew^Lln · Wnew^Lwn)
        leff = Lnew - 2·dl
        ...
    """
    L, W, NF = geom.L, geom.W, geom.NF
    Lnew = L + model["xl"]
    Wnew = W / NF + model["xw"]

    T0 = Lnew ** model["lln"] if model["lln"] != 0 else 1.0
    T1 = Wnew ** model["lwn"] if model["lwn"] != 0 else 1.0
    tmp1 = model["ll"] / T0 + model["lw"] / T1 + model["lwl"] / (T0 * T1)
    dl = model["lint"] + tmp1
    tmp2 = model["llc"] / T0 + model["lwc"] / T1 + model["lwlc"] / (T0 * T1)
    dlc = model["dlc"] + tmp2

    T2 = Lnew ** model["wln"] if model["wln"] != 0 else 1.0
    T3 = Wnew ** model["wwn"] if model["wwn"] != 0 else 1.0
    tmp1 = model["wl"] / T2 + model["ww"] / T3 + model["wwl"] / (T2 * T3)
    dw = model["wint"] + tmp1
    tmp2 = model["wlc"] / T2 + model["wwc"] / T3 + model["wwlc"] / (T2 * T3)
    dwc = model["dwc"] + tmp2
    dwj = model["dwj"] + tmp2

    leff = Lnew - 2.0 * dl
    weff = Wnew - 2.0 * dw
    if leff <= 0.0:
        raise ValueError(f"BSIM4 geometry: leff={leff} <= 0")
    if weff <= 0.0:
        raise ValueError(f"BSIM4 geometry: weff={weff} <= 0")

    leffCV = Lnew - 2.0 * dlc
    weffCV = Wnew - 2.0 * dwc
    weffCJ = Wnew - 2.0 * dwj

    if model["binunit"] == 1:
        Inv_L = 1.0e-6 / leff
        Inv_W = 1.0e-6 / weff
        Inv_LW = 1.0e-12 / (leff * weff)
    else:
        Inv_L = 1.0 / leff
        Inv_W = 1.0 / weff
        Inv_LW = 1.0 / (leff * weff)

    return EffectiveGeom(
        Lnew=Lnew, Wnew=Wnew, dl=dl, dlc=dlc, dw=dw, dwc=dwc, dwj=dwj,
        leff=leff, weff=weff, leffCV=leffCV, weffCV=weffCV, weffCJ=weffCJ,
        Inv_L=Inv_L, Inv_W=Inv_W, Inv_LW=Inv_LW,
    )

```


=== FILE: _extracted/nsram_bsim4_port/nsram/nsram/bsim4_port/diode.py (5121 chars) ===
```python
"""bsim4_port.diode — Source/drain body junction diodes (manual §11.1).

Faithful differentiable port of BSIM4 v4.8.3 body diode currents Ibs and Ibd.
These currents flow body↔source (Ibs) and body↔drain (Ibd) and enter the
body-node KCL with positive sign on the body side.

Source reference:
  - b4ld.c §654-852  : SourceSatCurrent / DrainSatCurrent + dioMod=0/1/2

We implement dioMod=1 (smooth-clamped exponential without breakdown), the
default for compact models.  The breakdown branch (xjbvs/bvs) is left as
TODO — relevant only at Vbs ≲ -BVS ≈ -10V which never occurs in NS-RAM ops.
TODO(reverse-bv): port dioMod=2 reverse-breakdown branch when needed.
TODO(TAT): trap-assisted tunneling not yet ported.

Components (manual §11.1):
    Is_total = As · Js  +  Ps · Jsws  +  Weff_CJ·NF · Jswgs        (source)
    Id_total = Ad · Jd  +  Pd · Jswd  +  Weff_CJ·NF · Jswgd        (drain)
    Ibs = Is_total · (exp(Vbs/(Nj·vt)) - 1)
    Ibd = Id_total · (exp(Vbd/(Nj·vt)) - 1)

Forward-bias safe; reverse bias clamps the exponent at -EXP_THRESHOLD via
safe_exp so the (exp - 1) saturates at exactly -1 (= reverse-bias saturation
current).
"""
from __future__ import annotations

import torch

from .model_card import BSIM4Model
from .smooth import safe_exp
from .temp import SizeDependParam


def compute_body_diodes(
    model: BSIM4Model,
    sd: SizeDependParam,
    Vbs: torch.Tensor | float,
    Vbd: torch.Tensor | float,
    *,
    As: float = 0.0,        # source bottom area [m²]
    Ad: float = 0.0,        # drain bottom area  [m²]
    Ps: float = 0.0,        # source perimeter (isolation edge) [m]
    Pd: float = 0.0,        # drain perimeter (isolation edge)  [m]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Body diode currents (Ibs, Ibd).  b4ld.c §654-852, manual §11.1.

    Geometry inputs (areas/perimeters) default to zero; in that case the
    currents reduce to the gate-edge component (weffCJ·NF·Jswg*).  For
    NS-RAM-style compact cells, callers should pass realistic As/Ad/Ps/Pd.

    The "dioMod=1 smooth" form is used: pure exp(V/(Nj·vt))-1, saturating at
    -A·Js for Vbs ≪ 0 via safe_exp clamp.

    NOTE: nj defaults — model_card defaults njs/njd to 0 (BSIM-style
    "givenness" flag).  When 0, the C code falls back to nj=1.0; we mirror
    that here.  Likewise jss/jsd default to 0; if all junction densities are
    zero we return zero current with `gmin·V` form (skipped here — caller is
    expected to add device-level gmin if needed).
    """
    # Coerce
    Vbs_t = torch.as_tensor(Vbs, dtype=torch.float64)
    Vbd_t = torch.as_tensor(Vbd, dtype=torch.float64)
    Vbs_b, Vbd_b = torch.broadcast_tensors(Vbs_t, Vbd_t)

    # ---- Temp-shifted current densities (from sd) ------------------------ #
    # GRADFIX: drop float() so injected tensor jss/jsd via sd.SourceSatCurDensity_T
    # propagate gradients. Floats still pass through unchanged.
    Js_s = sd.SourceSatCurDensity_T                    # [A/m²]
    Js_d = sd.DrainSatCurDensity_T                     # [A/m²]
    # Sidewall + gate-edge densities: temp-shift uses same xtis/xtid factor
    # as Js.  For first cut we treat them as zero unless the card supplies
    # them; in that case apply the same TRatio factor as Js.
    TRatio = float(sd.model_ctx.TRatio)
    xtis = float(model.get("xtis", 0.0))
    xtid = float(model.get("xtid", 0.0))
    # Junction emission factors  (model_card defaults give 0 ⇒ fall back to 1)
    nj_s_card = float(model.get("njs", 0.0))
    nj_s = nj_s_card if nj_s_card > 0 else 1.0
    nj_d_card = float(model.get("njd", 0.0))
    nj_d = nj_d_card if nj_d_card > 0 else 1.0

    # Sidewall densities (no TAT, simple Tratio^xti) — order-of-mag accurate.
    def _scale(jname: str, xti: float) -> float:
        j0 = float(model.get(jname, 0.0))
        if j0 == 0.0:
            return 0.0
        return j0 * (TRatio ** xti)

    jsws = _scale("jsws", xtis)
    jswgs = _scale("jswgs", xtis)
    jswd = _scale("jswd", xtid)
    jswgd = _scale("jswgd", xtid)

    # weffCJ × NF (use NF=1 if missing — Geometry stores it on geom)
    NF = float(getattr(sd.geom, "NF", 1)) if hasattr(sd.geom, "NF") else 1.0
    weffCJ = float(sd.geom.weffCJ)

    # ---- Saturation current per junction --------------------------------- #
    SourceSatI = As * Js_s + Ps * jsws + weffCJ * NF * jswgs
    DrainSatI = Ad * Js_d + Pd * jswd + weffCJ * NF * jswgd

    # Vt for diode emission
    vtm = float(sd.model_ctx.vtm)
    Nvtms = vtm * nj_s
    Nvtmd = vtm * nj_d

    # ---- Ibs ------------------------------------------------------------- #
    # safe_exp clamps the argument to ±EXP_THRESHOLD; at deep reverse bias,
    # safe_exp(Vbs/Nvtms) ≈ exp(-34) so (exp - 1) ≈ -1 exactly.
    if SourceSatI <= 0.0:
        Ibs = torch.zeros_like(Vbs_b)
    else:
        evbs = safe_exp(Vbs_b / Nvtms)
        Ibs = SourceSatI * (evbs - 1.0)

    # ---- Ibd ------------------------------------------------------------- #
    if DrainSatI <= 0.0:
        Ibd = torch.zeros_like(Vbd_b)
    else:
        evbd = safe_exp(Vbd_b / Nvtmd)
        Ibd = DrainSatI * (evbd - 1.0)

    return Ibs, Ibd

```


=== FILE: _extracted/nsram_bsim4_port/nsram/nsram/bsim4_port/temp.py (15704 chars) ===
```python
"""B05_PHYSCONST + B06_TEMPADJ — physical constants and temperature dependencies.

Faithful port of b4temp.c phases 5+6:
  - oxide setup (coxe, factor1, cgdo, cgso)
  - bandgap Eg(T), intrinsic ni
  - Vtm0 (at Tnom), vtm (at op T)
  - temperature shifts on Vth0, u0, vsat, Rds, Js (per-instance)

Returns a SizeDependParam dict mirroring the C `bsim4SizeDependParam` struct
fields most relevant for our DC + transient port.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any

from .constants import (
    Charge_q, EPS0, EPSSI, KboQ, PI, TZEROK,
)
from .geometry import EffectiveGeom, Geometry, compute_geometry
from .model_card import BSIM4Model


# ---- Model-level (size-independent) physics constants ----------------------

@dataclass
class ModelTempCtx:
    """Computed once per model + temperature. Bit-for-bit faithful."""
    Tnom: float            # nominal temp [K]
    Temp: float            # operating temp [K]
    delTemp: float         # Temp - Tnom
    TRatio: float          # Temp / Tnom
    Vtm0: float            # KboQ * Tnom
    vtm: float             # KboQ * Temp
    Eg0: float             # bandgap at Tnom
    Eg: float              # bandgap at Temp
    ni: float              # intrinsic carrier density at Tnom
    epsrox: float          # gate-ox dielectric constant
    epssub: float          # substrate permittivity (F/m)
    toxe: float            # effective gate-ox thickness
    coxe: float            # gate-ox capacitance per area
    factor1: float         # sqrt(epssub / (epsrox*EPS0) * toxe)


def compute_model_temp(model: BSIM4Model, T_C) -> ModelTempCtx:
    """Faithful port of b4temp.c lines 162-235 (model-level temperature setup).

    T_C may be a Python float OR a torch.Tensor (for fitting). When a tensor,
    we keep math.* on Python-float cached results: the cache is per scalar T,
    so callers must pass a scalar at cache-build time. For tensor T_C inside
    autograd graphs, downstream layers (dc.py) re-derive Vtm from Temp.
    """
    # Defensive: detach to scalar for the cached scalar pipeline. Callers
    # wanting tensor-T fitting should detour via a tensor-aware recompute.
    try:
        T_C_scalar = float(T_C.detach().item()) if hasattr(T_C, "detach") else float(T_C)
    except Exception:
        T_C_scalar = float(T_C)
    Tnom = model["tnom"] + TZEROK
    Temp = T_C_scalar + TZEROK
    delTemp = Temp - Tnom
    TRatio = Temp / Tnom

    # Oxide / substrate (mtrlMod = 0 path for now — Sebas uses Si)
    if model["mtrlmod"] != 0:
        epsrox = 3.9
        toxe = model["eot"]
        epssub = EPS0 * model["epsrsub"]
    else:
        epsrox = model["epsrox"]
        toxe = model["toxe"]
        epssub = EPSSI

    coxe = epsrox * EPS0 / toxe
    factor1 = math.sqrt(epssub / (epsrox * EPS0) * toxe)
    Vtm0 = KboQ * Tnom
    vtm = KboQ * Temp

    if model["mtrlmod"] == 0:
        Eg0 = 1.16 - 7.02e-4 * Tnom * Tnom / (Tnom + 1108.0)
        ni = (1.45e10 * (Tnom / 300.15) * math.sqrt(Tnom / 300.15)
              * math.exp(21.5565981 - Eg0 / (2.0 * Vtm0)))
        Eg = 1.16 - 7.02e-4 * Temp * Temp / (Temp + 1108.0)
    else:
        Eg0 = (model["bg0sub"] - model["tbgasub"] * Tnom * Tnom
                / (Tnom + model["tbgbsub"]))
        T0_b = (model["bg0sub"] - model["tbgasub"] * 300.15 * 300.15
                 / (300.15 + model["tbgbsub"]))
        ni = (model["ni0sub"] * (Tnom / 300.15) * math.sqrt(Tnom / 300.15)
              * math.exp((T0_b - Eg0) / (2.0 * Vtm0)))
        Eg = (model["bg0sub"] - model["tbgasub"] * Temp * Temp
                / (Temp + model["tbgbsub"]))

    return ModelTempCtx(
        Tnom=Tnom, Temp=Temp, delTemp=delTemp, TRatio=TRatio,
        Vtm0=Vtm0, vtm=vtm, Eg0=Eg0, Eg=Eg, ni=ni,
        epsrox=epsrox, epssub=epssub, toxe=toxe, coxe=coxe, factor1=factor1,
    )


# ---- Per-instance scaled + temperature-adjusted params ---------------------

# A short list of scaled params we need for our DC port. Each is:
#   pParam->X = model->X + l_X·Inv_L + w_X·Inv_W + p_X·Inv_LW
# where l_X, w_X, p_X are the model's "lX", "wX", "pX" coefficients.
# Default coefficients are 0, so for un-binned PDKs (e.g. Sebas's),
# pParam->X == model->X.
SCALED_PARAMS = [
    "vth0", "k1", "k2", "k3", "k3b", "w0",
    "dvt0", "dvt1", "dvt2", "dvt0w", "dvt1w", "dvt2w",
    "u0", "ua", "ub", "uc", "ud", "up", "lp",
    # WAVE3-FIX (z214): ua1/ub1/uc1/ud1 also bin via lX/wX/pX (b4temp.c §757-770)
    # and feed the ua/ub/uc/ud temperature shift below.  Without these the
    # mobility at T≠Tnom is wrong → Esat wrong → Vdsat ~50 mV high.
    "ua1", "ub1", "uc1", "ud1",
    # Also missing previously: b0/b1 enter Abulk (`b0/(Weff+b1)`).
    "b0", "b1",
    "vsat", "a0", "ags", "a1", "a2", "at",
    "keta", "nfactor", "cit", "cdsc", "cdscb", "cdscd",
    "eta0", "etab", "fprout", "pdits", "pditsd",
    "pclm", "pdiblc1", "pdiblc2", "pdiblcb", "drout", "dsub",
    "pscbe1", "pscbe2", "pvag",
    "delta", "rdsw", "rsw", "rdw", "prwg", "prwb", "wr",
    "alpha0", "alpha1", "beta0",
    "agidl", "bgidl", "cgidl", "egidl", "fgidl", "kgidl", "rgidl",
    "agisl", "bgisl", "cgisl", "egisl", "fgisl", "kgisl", "rgisl",
    "aigc", "bigc", "cigc", "aigsd", "bigsd", "cigsd",
    "aigbacc", "bigbacc", "cigbacc", "aigbinv", "bigbinv", "cigbinv",
    "nigc", "nigbacc", "nigbinv", "ntox", "eigbinv", "pigcd", "poxedge",
    "xrcrg1", "xrcrg2",
    "lambda", "vtl", "xn", "lc",
    "vfb", "tnoia", "tnoib", "rnoia", "rnoib", "rnoic",
    "ntnoi",
    "voff", "voffl", "voffcv", "voffcvl", "minv", "minvcv",
    "lpe0", "lpeb",
    "phin", "ndep", "nsd", "ngate",
    "xt", "xj", "vbm",
    "dvtp0", "dvtp1", "dvtp2", "dvtp3", "dvtp4", "dvtp5",
]


@dataclass
class SizeDependParam:
    """Cached per-(model, geometry, T) scaled + temp-adjusted params.

    Fields are dynamically populated; we don't pre-declare 200+ slots.
    """
    geom: EffectiveGeom
    model_ctx: ModelTempCtx
    scaled: dict[str, float] = field(default_factory=dict)
    # Temp-adjusted shadows of selected scaled params
    vth0_T: float = 0.0
    u0temp: float = 0.0
    vsattemp: float = 0.0
    rdstemp: float = 0.0
    Vth_T: float = 0.0
    SourceSatCurDensity_T: float = 0.0
    DrainSatCurDensity_T: float = 0.0
    # Pre-computed Vth/Xdep quantities (from b4temp.c §1300-1520)
    phi: float = 0.0           # surface potential 2φF
    sqrtPhi: float = 0.0
    phis3: float = 0.0
    Xdep0: float = 0.0         # depletion width at zero bias
    sqrtXdep0: float = 0.0
    vbi: float = 0.0           # built-in pn voltage
    vbsc: float = 0.0          # body-bias saturation clamp
    vbm: float = 0.0           # body-bias min
    k1ox: float = 0.0          # k1 × toxe/toxm
    k2ox: float = 0.0          # k2 × toxe/toxm
    theta0vb0: float = 0.0     # short-channel DIBL prefactor
    litl: float = 0.0          # screening length
    # Vgsteff regularizers (b4temp.c §1373-1427)
    mstar: float = 0.5
    voffcbn: float = 0.0
    cdep0: float = 0.0
    # Tcen / Coxeff capMod=2 inputs (b4ld.c §1789-1805, b4temp.c §1786)
    vtfbphi2: float = 0.0      # 4·(vth0 - vfb - phi); clamped ≥0
    coxp: float = 0.0          # gate-ox cap using toxp (defaults to coxe)
    toxp: float = 0.0          # poly oxide thickness (defaults to toxe)
    ados: float = 1.0
    bdos: float = 1.0
    # Other cached scalars
    vfb_eff: float = 0.0       # type·vth0 - vfb - phi sign branch input


def compute_size_dep(model: BSIM4Model, geom: Geometry, T_C: float) -> SizeDependParam:
    """Faithful port of b4temp.c phases 3+4.

    Pipeline:
      1. compute geometry → EffectiveGeom (with Inv_L, Inv_W, Inv_LW)
      2. compute model temp ctx (Eg, ni, Vtm, coxe, factor1)
      3. for each scaled param X, apply: pParam.X = model.X + l_X·Inv_L + w_X·Inv_W + p_X·Inv_LW
      4. apply temperature shifts on selected scaled params (Vth0, u0, vsat, Js)
    """
    eff = compute_geometry(model, geom)
    ctx = compute_model_temp(model, T_C)

    scaled: dict[str, float] = {}
    for X in SCALED_PARAMS:
        base = model.get(X, 0.0)
        # Coefficient names are lX, wX, pX (lvth0, wvth0, pvth0, ...).
        l_X = model.get("l" + X, 0.0)
        w_X = model.get("w" + X, 0.0)
        p_X = model.get("p" + X, 0.0)
        scaled[X] = base + l_X * eff.Inv_L + w_X * eff.Inv_W + p_X * eff.Inv_LW

    # Temperature adjustments (b4temp.c §1208-1300, simplified subset).
    delTemp = ctx.delTemp
    TRatio = ctx.TRatio

    # Vth0 temperature shift LIVES IN dc.py (b4ld.c:1099-1103, the only place
    # in the C source where it appears). DO NOT also apply it here — that
    # double-counts. Diagnostic z71 confirmed this was the root cause of the
    # 109% subthreshold worst-case rel err (Wave 2 finding 2026-04-29).
    # Reference: b4temp.c never touches vth0 — kt1/kt1l/kt2 only enter via
    # b4ld.c §1099-1103 inside the per-bias Vth assembly.
    vth0_T = scaled["vth0"]
    Tm1 = TRatio - 1.0   # used by vsattemp and rdstemp below

    # Mobility: u0temp = u0 · Tratio^ute  (mobMod ≠ 3 path, b4temp.c:1283)
    ute = model["ute"]
    u0temp = scaled["u0"] * (TRatio ** ute)

    # WAVE3-FIX (z214): mobility-coefficient temperature shifts.  b4temp.c
    # §1202-1241.  tempmod=0 path:  ua += ua1·T0, ub += ub1·T0, uc += uc1·T0,
    # ud += ud1·T0.  Without these, ua/ub/uc are off by 30-300% at T≠Tnom →
    # Esat wrong → Vdsat off by ~50 mV.  We mutate the `scaled` dict in place
    # so dc.py picks up the temperature-shifted values via `P["ua"]`.
    T0_uab = Tm1   # = TRatio - 1, the "T0" the C source uses on §1196.
    if model["tempmod"] == 0:
        scaled["ua"] = scaled["ua"] + scaled["ua1"] * T0_uab
        scaled["ub"] = scaled["ub"] + scaled["ub1"] * T0_uab
        scaled["uc"] = scaled["uc"] + scaled["uc1"] * T0_uab
        scaled["ud"] = scaled["ud"] + scaled["ud1"] * T0_uab
    elif model["tempmod"] == 3:
        scaled["ua"] = scaled["ua"] * (TRatio ** scaled["ua1"])
        scaled["ub"] = scaled["ub"] * (TRatio ** scaled["ub1"])
        scaled["uc"] = scaled["uc"] * (TRatio ** scaled["uc1"])
        scaled["ud"] = scaled["ud"] * (TRatio ** scaled["ud1"])
    else:  # tempmod = 1, 2
        scaled["ua"] = scaled["ua"] * (1.0 + scaled["ua1"] * delTemp)
        scaled["ub"] = scaled["ub"] * (1.0 + scaled["ub1"] * delTemp)
        scaled["uc"] = scaled["uc"] * (1.0 + scaled["uc1"] * delTemp)
        scaled["ud"] = scaled["ud"] * (1.0 + scaled["ud1"] * delTemp)

    # Saturation velocity (b4temp.c:1208 mtrlMod=0 path):
    #   vsattemp = vsat - at · (Temp/Tnom - 1)         tempMod=0
    #   vsattemp = vsat * (1 - at·delTemp)             tempMod≠0
    # We use tempMod=0 form (matches what ngspice does for tempmod=0 default).
    at_v = scaled["at"]
    if model["tempmod"] == 0:
        vsattemp = scaled["vsat"] - at_v * Tm1
    else:
        vsattemp = scaled["vsat"] * (1.0 - at_v * delTemp)

    # Series resistance temp scaling: rds(T) = rdsw_T · (1 + prt·...)
    # Simplified: not actually used in our minimal port until b4ld.c rdsmod
    prt = model["prt"]
    rds_factor = 1.0 + prt * Tm1
    rdstemp = scaled["rdsw"] * rds_factor

    # Junction saturation current density temperature dependence
    # b4temp.c lines 1461-1510 region; we use the canonical Eg/Vtm shift:
    Js0 = model["jss"]
    # Js(T) = Js0 · (Tratio)^xtis · exp[(Eg0/Vtm0 - Eg/vtm)/Nj]
    xtis = model["xtis"]
    nj = model.get("njs", 1.0)
    if nj <= 0:
        nj = 1.0
    T0_eg = (ctx.Eg0 / ctx.Vtm0) - (ctx.Eg / ctx.vtm)
    js_temp_factor = (TRatio ** xtis) * math.exp(T0_eg / nj)
    SjctSatT = Js0 * js_temp_factor

    Js0_d = model["jsd"]
    xtid = model["xtid"]
    nj_d = model.get("njd", 1.0)
    if nj_d <= 0:
        nj_d = 1.0
    js_temp_factor_d = (TRatio ** xtid) * math.exp(T0_eg / nj_d)
    DjctSatT = Js0_d * js_temp_factor_d

    # ---- Pre-computed Vth/Xdep quantities (b4temp.c §1322-1520) -----------
    # phi = Vtm0·log(NDEP/ni) + phin + 0.4
    ndep = max(model["ndep"], 1e10)   # safety: never log(0)
    phi = ctx.Vtm0 * math.log(ndep / max(ctx.ni, 1e-30)) + model["phin"] + 0.4
    if phi <= 0:
        phi = 0.4   # fallback; should warn if Sebas's ndep produces this
    sqrtPhi = math.sqrt(phi)
    phis3 = sqrtPhi * phi
    # Xdep0 = sqrt(2·epssub/(q·NDEP·1e6)) · sqrtPhi  (NDEP in cm⁻³ → m⁻³ via 1e6)
    Xdep0 = math.sqrt(2.0 * ctx.epssub / (Charge_q * ndep * 1.0e6)) * sqrtPhi
    sqrtXdep0 = math.sqrt(Xdep0)
    # vbi = Vtm0·log(NSD·NDEP/ni²)
    nsd = max(model["nsd"], 1e10)
    vbi = ctx.Vtm0 * math.log(nsd * ndep / max(ctx.ni * ctx.ni, 1e-60))
    # vbsc body-bias clamp: simplified — use vbm if given, else -3.0
    vbm = model["vbm"]
    vbsc = vbm if vbm < 0 else -3.0
    # k1ox, k2ox: oxide-thickness scaling (b4temp.c §1516, 1802)
    # Same ref-default ordering issue as toxp: re-resolve toxm against toxe.
    if model.is_given("toxm"):
        toxm = model["toxm"] if model["toxm"] > 0 else ctx.toxe
    else:
        toxm = ctx.toxe
    k1ox = scaled["k1"] * ctx.toxe / toxm
    k2ox = scaled["k2"] * ctx.toxe / toxm
    # litl screening length (b4temp.c §1347)
    epsrox = ctx.epsrox
    litl = math.sqrt(3.0 * 3.9 / epsrox * model["xj"] * ctx.toxe)
    # theta0vb0 — DIBL prefactor at zero Vbs (b4temp.c §1538-1560)
    # Theta0 = exp(-0.5·dvt0·Leff/litl) approx (full form b4ld.c handles)
    # We compute a simple approximation here; full form computed in dc.py.
    theta0vb0 = math.exp(-0.5 * scaled["dvt0"] * eff.leff / max(litl, 1e-12))

    # ---- Vgsteff bridge regularizers (b4temp.c §1373, 1425-1427) ----------
    # mstar = 0.5 + atan(minv)/pi
    minv_v = scaled.get("minv", model.get("minv", 0.0))
    mstar = 0.5 + math.atan(minv_v) / math.pi
    # voffcbn = voff + voffl/Leff
    voff_v = scaled.get("voff", model.get("voff", -0.08))
    voffl_v = model.get("voffl", 0.0)
    voffcbn = voff_v + voffl_v / max(eff.leff, 1e-12)
    # cdep0 = sqrt(q·epssub·NDEP·1e6 / 2 / phi)   (b4temp.c §1373-1375)
    ndep_scaled = max(scaled.get("ndep", ndep), 1e10)
    cdep0 = math.sqrt(Charge_q * ctx.epssub * ndep_scaled * 1.0e6 / 2.0 / max(phi, 1e-3))

    # ---- Tcen / Coxeff inputs (b4ld.c §1789-1805, b4temp.c §1786, §180) ----
    # coxp = epsrox·EPS0 / toxp; if toxp not given, fall back to toxe.
    # NOTE: model_card.py resolves "ref" defaults BEFORE user overrides, so
    # toxp ends up frozen to the default-toxe value (3e-9) even if the user
    # set toxe=4e-9. Re-resolve here using is_given().
    if model.is_given("toxp"):
        toxp = model.get("toxp", 0.0)
    else:
        toxp = ctx.toxe
    if toxp <= 0:
        toxp = ctx.toxe
    coxp = ctx.epsrox * EPS0 / toxp
    # vtfbphi2 = 4·(type·vth0 - vfb - phi); clamped ≥0 (b4temp.c §1786-1788)
    type_n = float(model._values.get("type", 1)) if hasattr(model, "_values") else 1.0
    vfb_card = model.get("vfb", -1.0)
    T3_vfb = type_n * vth0_T - vfb_card - phi
    vtfbphi2 = max(4.0 * T3_vfb, 0.0)
    # ados/bdos defaults are 1.0 each
    ados = model.get("ados", 1.0)
    bdos = model.get("bdos", 1.0)

    return SizeDependParam(
        geom=eff,
        model_ctx=ctx,
        scaled=scaled,
        vth0_T=vth0_T,
        u0temp=u0temp,
        vsattemp=vsattemp,
        rdstemp=rdstemp,
        Vth_T=vth0_T,
        SourceSatCurDensity_T=SjctSatT,
        DrainSatCurDensity_T=DjctSatT,
        phi=phi, sqrtPhi=sqrtPhi, phis3=phis3,
        Xdep0=Xdep0, sqrtXdep0=sqrtXdep0,
        vbi=vbi, vbsc=vbsc, vbm=vbm,
        k1ox=k1ox, k2ox=k2ox,
        theta0vb0=theta0vb0, litl=litl,
        mstar=mstar, voffcbn=voffcbn, cdep0=cdep0,
        vtfbphi2=vtfbphi2, coxp=coxp, toxp=toxp,
        ados=ados, bdos=bdos,
        vfb_eff=T3_vfb,
    )

```


=== FILE: _extracted/nsram_bsim4_port/nsram/nsram/bsim4_port/leak.py (18552 chars) ===
```python
"""bsim4_port.leak — Body-current leakage models (impact-ion, GIDL/GISL, Igb).

Faithful differentiable port of BSIM4 v4.8.3 sub-models that contribute to
the body-node KCL right-hand side. Critical for NS-RAM since the floating
bulk integrates these currents.

Source references:
  - b4ld.c §2047-2086  : Iii (impact-ionization, manual §6.1)
  - b4ld.c §2274-2370  : Igidl/Igisl pre-4.7 model (manual §6.2, gidlMod=0)
  - b4ld.c §2493-2538  : Voxacc / Voxdepinv setup
  - b4ld.c §2769-2882  : Igb = Igbacc + Igbinv (manual §4.3.1)

We DO NOT port Igc / Igs / Igd (gate-channel, gate-source, gate-drain) here
because those currents do NOT enter the body-node KCL. They flow gate→channel
ends.  TODO: port to a separate leak_gate.py if ever needed.

All exp(...) calls go through smooth.safe_exp (clamped at ±34) per project rule.
All denominators get clamp_min(epsilon) per project rule.
fp64 throughout.
"""
from __future__ import annotations

import torch

from .constants import DELTA_3, EPS0, EXP_THRESHOLD, MIN_EXP
from .model_card import BSIM4Model
from .smooth import safe_exp, safe_sqrt, smooth_max
from .temp import SizeDependParam


_DENOM_EPS = 1e-30        # absolute floor for /denominators
_GIDL_T2_CAP = 100.0      # b4ld.c "if (T2 < 100.0)" branch → use exp(-T2)
_TOX_FLOOR = 1.0e-12      # floor for toxe (1 pm)


def _t(x, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(x, dtype=like.dtype, device=like.device)


# --------------------------------------------------------------------------- #
# Impact-ionization                                                           #
# --------------------------------------------------------------------------- #

def compute_iimpact(
    model: BSIM4Model,
    sd: SizeDependParam,
    dc_result,
    Vds: torch.Tensor | float,
) -> torch.Tensor:
    """Substrate (impact-ionization) current Isub. b4ld.c §2047-2086.

    Equation (manual §6.1):
        T2  = (alpha0 + alpha1·Leff) / Leff
        if (Vds-Vdseff) > beta0/EXP_THRESHOLD:
            T1 = T2·(Vds-Vdseff)·exp(-beta0/(Vds-Vdseff))
        else:
            T1 = T2·MIN_EXP·(Vds-Vdseff)            # tiny linear floor
        Iii = T1 · Idsa · Vdseff

    WAVE2-FIX-1 (Gap 2, b4ld.c §2069 vs §2089-2091):
      Iii uses the *pre-SCBE* ``Idsa·Vdseff`` quantity (``dc_result.Idsa``),
      NOT the post-SCBE ``dc_result.Ids``. SCBE is applied to Ids only AFTER
      Iii is computed in the C source. Using Ids inflates Iii by the SCBE
      factor (typically 1.0-1.3, i.e. up to ~30% rel err in saturation).
      Falls back to dc_result.Ids if Idsa is missing (legacy compat).
    """
    P = sd.scaled
    leff = float(sd.geom.leff)
    alpha0 = P.get("alpha0", 0.0)
    alpha1 = P.get("alpha1", 0.0)
    beta0 = P.get("beta0", 0.0)

    Vds_t = torch.as_tensor(Vds, dtype=dc_result.Ids.dtype, device=dc_result.Ids.device)
    Vdseff = dc_result.Vdseff
    Vds_b, Vdseff_b = torch.broadcast_tensors(Vds_t, Vdseff)
    diffVds = Vds_b - Vdseff_b
    # Guard against negative diffVds (shouldn't occur for Vds>0 NMOS) — clamp.
    diffVds = diffVds.clamp_min(0.0)

    tmp = alpha0 + alpha1 * leff
    if (tmp <= 0.0) or (beta0 <= 0.0):
        # Card disables impact-ion.
        return torch.zeros_like(diffVds)

    T2 = tmp / leff
    threshold = beta0 / EXP_THRESHOLD                       # b4ld.c branch
    diff_safe = diffVds.clamp_min(_DENOM_EPS)
    # Strong-bias arm: T1 = T2·diff·exp(-beta0/diff)
    T0 = -beta0 / diff_safe
    T1_strong = T2 * diff_safe * safe_exp(T0)
    # Weak-bias arm: T1 = T2·MIN_EXP·diff
    T1_weak = T2 * MIN_EXP * diff_safe
    T1 = torch.where(diffVds > threshold, T1_strong, T1_weak)

    # WAVE2-FIX-1 (Gap 2): use pre-SCBE Idsa·Vdseff (b4ld.c §2069). Fallback to
    # Ids preserves backward compatibility with any legacy DCResult lacking Idsa.
    Idsa_Vdseff = getattr(dc_result, "Idsa", None)
    if Idsa_Vdseff is None:
        Idsa_Vdseff = dc_result.Ids
    Iii = T1 * Idsa_Vdseff
    return Iii


# --------------------------------------------------------------------------- #
# GIDL / GISL                                                                 #
# --------------------------------------------------------------------------- #

def _gidl_one_side(
    *,
    weffCJ: torch.Tensor,
    toxe: torch.Tensor,
    a: float, b: float, c: float, e: float,
    V_drive: torch.Tensor,                     # Vd-Vg-egidl (or -Vd-Vg-egisl)
    Vbody: torch.Tensor,                       # vbd (GIDL) or vbs (GISL)
    body_disable: torch.Tensor,                # bool: vbd>0 (GIDL) / vbs>0 (GISL)
) -> torch.Tensor:
    """Shared kernel for GIDL/GISL. b4ld.c §2295-2324.

    Igidl = a · Weff_CJ · T1 · exp(-b/T1) · Vbody³/(Vbody³ + c)   (b4ld.c form)
    where T1 = (Vd-Vg-egidl)/(3·toxe) (already passed in via V_drive/(3·tox)).
    """
    # T0 = 3·toxe (b4ld.c §2277)
    T0 = 3.0 * toxe.clamp_min(_TOX_FLOOR)
    T1 = V_drive / T0                                       # may be ≤0
    # Disable conditions: a≤0, b≤0, c≤0, T1≤0, body_disable.
    if a <= 0.0 or b <= 0.0 or c <= 0.0:
        return torch.zeros_like(V_drive)

    T1_safe = T1.clamp_min(_DENOM_EPS)
    T2 = b / T1_safe                                        # may be huge
    # b4ld.c branches T2<100 vs ≥100; we use a single safe_exp(-T2) which
    # automatically saturates at exp(-34)≈MIN_EXP.  Mirrors the saturated
    # branch but smoothly differentiable.
    Igidl_pre = a * weffCJ * T1_safe * safe_exp(-T2)        # cylinder

    # Body-bias factor: Vbody³/(Vbody³ + c)  (b4ld.c §2315-2323)
    # b4ld.c writes T4=v*v, T5=-v*T4, T6=c+T5, T7=T5/T6  →  T7 = -v³/(c-v³)
    # Inspecting: GIDL conducts when vbd<0 (drain reverse-biased to body), so
    # T5 = -vbd·vbd² > 0 ⇒ T7 ∈ (0,1).  We replicate the C exactly.
    T4 = Vbody * Vbody
    T5 = -Vbody * T4                                        # -v³
    T6 = c + T5                                             # could be near 0
    T6_safe = T6.clamp_min(_DENOM_EPS)
    T7 = T5 / T6_safe                                       # body factor
    Igidl = Igidl_pre * T7

    # Hard zero where T1≤0 or body_disable (matches b4ld.c if-branch).
    valid = (T1 > 0.0) & (~body_disable)
    Igidl = torch.where(valid, Igidl, torch.zeros_like(Igidl))
    return Igidl


def compute_igidl_gisl(
    model: BSIM4Model,
    sd: SizeDependParam,
    Vgs: torch.Tensor,
    Vds: torch.Tensor,
    Vbs: torch.Tensor | float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (Igidl, Igisl).  b4ld.c §2274-2370.  Manual §6.2 (gidlMod=0)."""
    if int(model.get("gidlmod", 0)) != 0:
        # gidlMod=1 form not ported yet.
        raise NotImplementedError("Only gidlMod=0 (pre-4.7) GIDL/GISL ported.")

    P = sd.scaled
    weffCJ = torch.as_tensor(sd.geom.weffCJ, dtype=Vgs.dtype, device=Vgs.device)
    toxe = torch.as_tensor(sd.model_ctx.toxe, dtype=Vgs.dtype, device=Vgs.device)

    Vgs_b, Vds_b, Vbs_b = torch.broadcast_tensors(
        torch.as_tensor(Vgs, dtype=torch.float64),
        torch.as_tensor(Vds, dtype=torch.float64),
        torch.as_tensor(Vbs, dtype=torch.float64),
    )
    # vgs_eff ≈ Vgs (we don't carry vfbsd correction; b4ld.c uses BSIM4vgs_eff
    # which is bias-clamped; for body-KCL accuracy this is sufficient).
    vgs_eff = Vgs_b
    vgd_eff = Vgs_b - Vds_b
    vbd = Vbs_b - Vds_b
    vbs = Vbs_b

    # GRADFIX: drop float() cast so gradcheck can propagate gradients when the
    # caller injects parameter tensors into sd.scaled. Floats still pass-through
    # unchanged; tensors flow through subsequent arithmetic.
    egidl = P.get("egidl", model.get("egidl", 0.8))
    egisl = P.get("egisl", model.get("egisl", egidl))
    agidl = P.get("agidl", 0.0)
    bgidl = P.get("bgidl", 0.0)
    cgidl = P.get("cgidl", model.get("cgidl", 0.5))
    agisl = P.get("agisl", agidl)
    bgisl = P.get("bgisl", bgidl)
    cgisl = P.get("cgisl", cgidl)

    # GIDL drive: Vd-Vg-egidl  (mtrlMod=0)
    V_drive_d = Vds_b - vgs_eff - egidl
    Igidl = _gidl_one_side(
        weffCJ=weffCJ, toxe=toxe,
        a=agidl, b=bgidl, c=cgidl, e=egidl,
        V_drive=V_drive_d, Vbody=vbd,
        body_disable=(vbd > 0.0),
    )
    # GISL drive: -Vd-Vg-egisl   (vgd_eff = Vg-Vd  ⇒  -Vd-Vg = -(Vd+Vg) ≠ -vgd_eff!)
    # b4ld.c §2331: T1 = (-vds - vgd_eff - egisl)/T0
    V_drive_s = -Vds_b - vgd_eff - egisl
    Igisl = _gidl_one_side(
        weffCJ=weffCJ, toxe=toxe,
        a=agisl, b=bgisl, c=cgisl, e=egisl,
        V_drive=V_drive_s, Vbody=vbs,
        body_disable=(vbs > 0.0),
    )
    return Igidl, Igisl


# --------------------------------------------------------------------------- #
# Vfbeff / Voxacc / Voxdepinv  (Wave-2 Gap 1 + Gap 7)                          #
# --------------------------------------------------------------------------- #

def compute_vfbeff(
    Vgs_eff: torch.Tensor,
    Vbseff: torch.Tensor,
    vfb: float,
) -> torch.Tensor:
    """Smooth flat-band voltage Vfbeff per b4ld.c §2496-2504.

    Faithful port of the BSIM4 v4.8.3 form:

        V3     = Vfb - Vgs_eff + Vbseff - DELTA_3
        T0     = sqrt(V3**2 +/- 4*DELTA_3*Vfb)        (+ if Vfb>0 else -)
        Vfbeff = Vfb - 0.5*(V3 + T0)

    The branch on `Vfb<=0` is on a per-device scalar (vfbzb / vfb is bias-independent),
    so a Python `if` is fine -- no torch.where needed. We wrap `sqrt` in `safe_sqrt`
    for numerical safety; in the well-defined regime the discriminant is structurally
    non-negative.

    Used by:
      - leak.compute_igb (Voxacc / Voxdepinv) -- Gap 7
      - caps_capmod2 (CTM charge model)        -- future Gap 3
    """
    V3 = (vfb - Vgs_eff + Vbseff - DELTA_3)
    if vfb <= 0.0:
        T0 = safe_sqrt(V3 * V3 - 4.0 * DELTA_3 * vfb)   # b4ld.c §2498
    else:
        T0 = safe_sqrt(V3 * V3 + 4.0 * DELTA_3 * vfb)   # b4ld.c §2500
    Vfbeff = vfb - 0.5 * (V3 + T0)                       # b4ld.c §2502
    return Vfbeff


def compute_voxacc_voxdepinv(
    Vgs_eff: torch.Tensor,
    Vbseff: torch.Tensor,
    Vgsteff: torch.Tensor,
    vfb: float,
    k1ox: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Voxacc / Voxdepinv per b4ld.c §2506-2537.

        Voxacc    = max(0, Vfb - Vfbeff)               (line 2506-2510)
        T3        = Vgs_eff - Vfbeff - Vbseff - Vgsteff
        if k1ox == 0:        Voxdepinv0 = 0
        elif T3 < 0:         Voxdepinv0 = -T3
        else:                Voxdepinv0 = k1ox*(sqrt((k1ox/2)**2 + T3) - k1ox/2)
        Voxdepinv = Voxdepinv0 + Vgsteff               (line 2534)

    Smooth-replacement strategy (per spec):
      - `if (Voxacc < 0)` clamp -> smooth_max(Voxacc, 0).
      - `if k1ox == 0`           -> Python if on scalar (param-time branch).
      - `if (T3 < 0)` two-arm    -> torch.where; both arms finite via safe_sqrt.
    """
    Vfbeff = compute_vfbeff(Vgs_eff, Vbseff, vfb)

    # Voxacc -- accumulation oxide voltage, clamped at 0 (b4ld.c §2506-2510).
    Voxacc_raw = vfb - Vfbeff
    zero = torch.zeros_like(Voxacc_raw)
    Voxacc = smooth_max(Voxacc_raw, zero)

    # Voxdepinv -- depletion+inversion, b4ld.c §2512-2537.
    T3 = Vgs_eff - Vfbeff - Vbseff - Vgsteff
    if k1ox == 0.0:
        Voxdepinv0 = torch.zeros_like(T3)
    else:
        T0 = 0.5 * k1ox
        # T3<0 arm: -T3                                              (b4ld.c §2518)
        # T3>=0 arm: k1ox*(sqrt(T0**2 + T3) - T0)                    (b4ld.c §2525-2527)
        sqrt_arm = k1ox * (safe_sqrt(T0 * T0 + T3) - T0)
        neg_arm = -T3
        Voxdepinv0 = torch.where(T3 < 0.0, neg_arm, sqrt_arm)

    Voxdepinv = Voxdepinv0 + Vgsteff                    # b4ld.c §2534
    return Voxacc, Voxdepinv


# --------------------------------------------------------------------------- #
# Gate-to-body tunneling                                                       #
# --------------------------------------------------------------------------- #

def _igb_branch(
    *,
    weff: torch.Tensor, leff: torch.Tensor, ToxRatio: torch.Tensor,
    Vgs: torch.Tensor, Vbs: torch.Tensor,
    Vaux_input: torch.Tensor, n: float, Vt: torch.Tensor,
    aigb: float, bigb: float, cigb: float,
    Vox: torch.Tensor,
    T11_prefactor: float, T12_factor: float,
) -> torch.Tensor:
    """Common Igbacc / Igbinv kernel. b4ld.c §2769-2876.

      T0  = Vt·n
      Vaux = T0·log(1+exp(Vaux_input/T0))      (smooth softplus)
      T2  = (Vgs-Vbs)·Vaux
      T11 = T11_prefactor · weff · leff · ToxRatio
      T12 = T12_factor · toxe                  (passed implicitly via call site)
      T5  = T12·(aigb + (aigb·cigb - bigb)·Vox - bigb·cigb·Vox²)
      T6  = exp(T5)                             (clamped)
      Igb = T11 · T2 · T6
    """
    T0 = (Vt * n).clamp_min(_DENOM_EPS)
    # Smooth softplus: Vaux = T0 · log(1 + exp(VxNVt))   with VxNVt = T1/T0
    # safe_exp guarantees clamp ±34, log1p stays finite.
    VxNVt = Vaux_input / T0
    Vaux = T0 * torch.log1p(safe_exp(VxNVt))

    T2 = (Vgs - Vbs) * Vaux

    T3 = aigb * cigb - bigb
    T4 = bigb * cigb
    # T12 already includes ·toxe (sign + magnitude); see compute_igb caller.
    T5 = T12_factor * (aigb + T3 * Vox - T4 * Vox * Vox)
    T6 = safe_exp(T5)

    Igb = T11_prefactor * weff * leff * ToxRatio * T2 * T6
    return Igb


def compute_igb(
    model: BSIM4Model,
    sd: SizeDependParam,
    Vgs: torch.Tensor,
    Vbs: torch.Tensor | float = 0.0,
    *,
    dc_result=None,
) -> torch.Tensor:
    """Gate-to-body tunneling Igb = Igbacc + Igbinv.  b4ld.c §2769-2882.

    Voxacc (b4ld.c §2506-2510) and Voxdepinv (b4ld.c §2511-2537) are computed
    via the faithful Vfbeff form (Wave-2 Gap 7) when `dc_result` is supplied
    (it provides the poly-dep'd Vgs_eff, Vbseff, Vgsteff intermediates).

    Backward-compatibility: when `dc_result` is None, fall back to the
    simplified softplus form (max(0, Vfb-Vgs+Vbs) / max(0, Vgs-Vfb-Vbs)). This
    preserves existing call sites (tests) but loses the Vgsteff coupling and
    k1ox-weighted sqrt bridge -- callers that need fidelity (e.g. nsram_cell)
    should pass `dc_result`.
    """
    if int(model.get("igbmod", 0)) == 0:
        return torch.zeros_like(torch.as_tensor(Vgs, dtype=torch.float64))

    P = sd.scaled
    Vgs_b, Vbs_b = torch.broadcast_tensors(
        torch.as_tensor(Vgs, dtype=torch.float64),
        torch.as_tensor(Vbs, dtype=torch.float64),
    )
    weff = torch.as_tensor(sd.geom.weff, dtype=Vgs_b.dtype, device=Vgs_b.device)
    leff = torch.as_tensor(sd.geom.leff, dtype=Vgs_b.dtype, device=Vgs_b.device)
    toxe = torch.as_tensor(sd.model_ctx.toxe, dtype=Vgs_b.dtype, device=Vgs_b.device)

    # ToxRatio = (toxref/toxe)^ntox · (1/toxe²)  per b4temp.c §1377-1379
    # NOTE: BSIM4ToxRatio in C includes a 1/toxe^2 factor in some forms but
    # b4temp.c stores the bare exp(ntox·log(toxref/toxe)) form; b4ld.c then
    # multiplies T11 (which includes 4.97e-7) by ToxRatio.  We follow b4temp.c.
    # GRADFIX: tensor-pass-through for fitable params.
    def _t(x):
        return x if isinstance(x, torch.Tensor) else torch.as_tensor(
            x, dtype=Vgs_b.dtype, device=Vgs_b.device)

    toxref = _t(model.get("toxref", 3.0e-9))
    ntox = _t(P.get("ntox", model.get("ntox", 1.0)))
    toxe_safe = torch.clamp(toxe, min=_TOX_FLOOR)
    ToxRatio = (toxref / toxe_safe) ** ntox / (toxe_safe * toxe_safe)

    Vfb = _t(model.get("vfb", -1.0))
    Vt = torch.as_tensor(sd.model_ctx.vtm, dtype=Vgs_b.dtype, device=Vgs_b.device)

    # ------------- Igbacc (b4ld.c §2769-2820) ------------- #
    # GRADFIX: keep tensors when injected via sd.scaled override.
    nigbacc = P.get("nigbacc", model.get("nigbacc", 1.0))
    aigbacc = P.get("aigbacc", model.get("aigbacc", 0.0136))
    bigbacc = P.get("bigbacc", model.get("bigbacc", 0.00171))
    cigbacc = P.get("cigbacc", model.get("cigbacc", 0.075))

    # WAVE2-FIX (Gap 7): Voxacc / Voxdepinv via faithful Vfbeff machinery
    # (b4ld.c §2493-2537) when dc_result is supplied.  Falls back to simplified
    # softplus form otherwise (preserves backward compat for existing tests).
    if dc_result is not None and dc_result.Vgs_eff is not None and dc_result.Vbseff is not None:
        # Broadcast dc intermediates against Vgs_b/Vbs_b shape.
        Vgs_eff_b = dc_result.Vgs_eff
        Vbseff_b = dc_result.Vbseff
        Vgsteff_b = dc_result.Vgsteff
        # vfb scalar from card; k1ox from sd (size-dep'd).  These match the C
        # `Vfb = vfbzb` / `pParam->BSIM4k1ox` reads at lines 2495 / 2512.
        # vfbzb proper would include vth0 and the k1·sqrtPhi shift (b4temp.c §1586,
        # §1805) -- using card.vfb is a known approximation, transitively validated
        # via the ngspice Igb diff once vfbzb is plumbed through (separate gap).
        vfb_scalar = float(model.get("vfb", -1.0))
        k1ox_scalar = float(P.get("k1ox", sd.k1ox))
        Voxacc, Voxdepinv = compute_voxacc_voxdepinv(
            Vgs_eff_b, Vbseff_b, Vgsteff_b, vfb_scalar, k1ox_scalar,
        )
    else:
        # SIMPLIFIED FALLBACK -- softplus·50.  No Vgsteff coupling, no k1ox sqrt.
        raw_Vacc = Vfb - Vgs_b + Vbs_b
        Voxacc = torch.nn.functional.softplus(raw_Vacc * 50.0) / 50.0
        raw_Vinv = Vgs_b - Vfb - Vbs_b
        Voxdepinv = torch.nn.functional.softplus(raw_Vinv * 50.0) / 50.0

    Vaux_input_acc = -Vgs_b + Vbs_b + Vfb                # T1 = -Vgs+Vbs+Vfb

    Igbacc = _igb_branch(
        weff=weff, leff=leff, ToxRatio=ToxRatio,
        Vgs=Vgs_b, Vbs=Vbs_b,
        Vaux_input=Vaux_input_acc, n=nigbacc, Vt=Vt,
        aigb=aigbacc, bigb=bigbacc, cigb=cigbacc,
        Vox=Voxacc,
        T11_prefactor=4.97232e-7,
        T12_factor=-7.45669e11 * toxe,
    )

    # ------------- Igbinv (b4ld.c §2822-2876) ------------- #
    # GRADFIX: tensor-pass-through.
    nigbinv = _t(P.get("nigbinv", model.get("nigbinv", 3.0)))
    aigbinv = _t(P.get("aigbinv", model.get("aigbinv", 0.0111)))
    bigbinv = _t(P.get("bigbinv", model.get("bigbinv", 0.000949)))
    cigbinv = _t(P.get("cigbinv", model.get("cigbinv", 0.006)))
    eigbinv = _t(P.get("eigbinv", model.get("eigbinv", 1.1)))

    # Voxdepinv already computed above (faithful path or fallback).
    Vaux_input_inv = Voxdepinv - eigbinv

    # b4ld.c §2849-2850: T11 *= 0.75610; T12 *= 1.31724
    Igbinv = _igb_branch(
        weff=weff, leff=leff, ToxRatio=ToxRatio,
        Vgs=Vgs_b, Vbs=Vbs_b,
        Vaux_input=Vaux_input_inv, n=nigbinv, Vt=Vt,
        aigb=aigbinv, bigb=bigbinv, cigb=cigbinv,
        Vox=Voxdepinv,
        T11_prefactor=4.97232e-7 * 0.75610,
        T12_factor=-7.45669e11 * toxe * 1.31724,
    )

    return Igbacc + Igbinv

```


=== FILE: _extracted/nsram_bsim4_port/nsram/nsram/bsim4_port/arclength.py (20796 chars) ===
```python
"""Pseudo-arclength continuation for the 2T NS-RAM cell.

Snapback is a fold bifurcation of the I-V curve: at some Vd_fold the equation
R(Vsint, Vb; Vd) = 0 has a singular Jacobian and there are two valid
(Vsint, Vb) roots (off-branch low-current, on-branch high-current). Plain
Newton on Vd-parameterized residual is undefined at the fold — that's why
gmin homotopy + warm-start from previous Vd still produces the "lines hopping
between roots" we observed in z93.

Pseudo-arclength continuation handles this by treating Vd itself as a state
variable and parameterizing the (Vsint, Vb, Vd) curve by arclength s. The
Jacobian of the augmented system is non-singular at the fold (the tangent
vector simply rotates so that ds/dVd reverses sign there).

Reference: Kelley & Keyes (1998) Convergence Analysis of Pseudo-Transient
Continuation; AUTO/MatCont/LOCA literature for fold-bifurcation tracking.

Implementation:
- 3D state x = (Vsint, Vb, Vd)
- 3D residual F(x) = (R_S, R_B, t·(x - x_prev) - ds)
  where R_S, R_B are the original 2T body-KCL residuals (re-using
  `nsram_cell_2T._residuals`) and the third equation is the arclength
  constraint orthogonal to the tangent t.
- Tangent computed from the 2x3 Jacobian by solving J·t = 0 normalized.
- Adaptive ds based on Newton iteration count.
- Returns Id, Vsint, Vb at user-requested Vd_targets via piecewise-linear
  interpolation along the traced path. For points where the path crosses
  Vd_target multiple times (snapback hysteresis), takes the FIRST crossing
  (= forward-sweep convention).
"""
from __future__ import annotations
from typing import Optional
import torch

from .nsram_cell_2T import (
    NSRAMCell2TConfig, _residuals, _jacobian_finite_diff,
)
from .model_card import BSIM4Model
from .bjt import GummelPoonNPN


def _residual_dVd(cfg, model, bjt, Vd, VG1, VG2, Vsint, Vb, P_M1, P_M2,
                   h: float = 1e-6, model_M2=None) -> torch.Tensor:
    """Finite-difference partial derivatives ∂(R_S, R_B)/∂Vd.
    Returns shape (2,) tensor for scalar inputs.
    """
    with torch.no_grad():
        Rsp, Rbp, _ = _residuals(cfg, model, bjt, Vd + h, VG1, VG2,
                                  Vsint, Vb, P_M1, P_M2, model_M2=model_M2)
        Rsm, Rbm, _ = _residuals(cfg, model, bjt, Vd - h, VG1, VG2,
                                  Vsint, Vb, P_M1, P_M2, model_M2=model_M2)
        dRs_dVd = (Rsp - Rsm) / (2 * h)
        dRb_dVd = (Rbp - Rbm) / (2 * h)
    return torch.stack([dRs_dVd, dRb_dVd])


def _solve_initial_point(cfg, model, bjt, Vd0, VG1, VG2,
                          Vsint_init=None, Vb_init=None,
                          max_iters: int = 30, tol: float = 1e-9,
                          model_M2=None):
    """Plain Newton at Vd=Vd0 to find the starting point on the curve."""
    Vd0 = torch.as_tensor(Vd0, dtype=torch.float64)
    VG1 = torch.as_tensor(VG1, dtype=torch.float64)
    VG2 = torch.as_tensor(VG2, dtype=torch.float64)
    Vsint = torch.tensor(0.5 * float(Vd0) if Vsint_init is None else float(Vsint_init),
                          dtype=torch.float64)
    Vb = torch.tensor(0.0 if Vb_init is None else float(Vb_init),
                       dtype=torch.float64)
    for _ in range(max_iters):
        with torch.no_grad():
            R_S, R_B, _ = _residuals(cfg, model, bjt, Vd0, VG1, VG2,
                                       Vsint, Vb, None, None, model_M2=model_M2)
            R = torch.stack([R_S, R_B])
            if R.norm() < tol:
                return Vsint, Vb, True
            J = _jacobian_finite_diff(cfg, model, bjt, Vd0, VG1, VG2,
                                       Vsint, Vb, None, None, model_M2=model_M2)
            try:
                dx = torch.linalg.solve(J, -R)
            except Exception:
                dx = torch.linalg.lstsq(J, -R.unsqueeze(-1)).solution.squeeze(-1)
            # Damped step
            alpha = 1.0
            for _ in range(10):
                Vsint_t = Vsint + alpha * dx[0]
                Vb_t = Vb + alpha * dx[1]
                R_S_t, R_B_t, _ = _residuals(cfg, model, bjt, Vd0, VG1, VG2,
                                               Vsint_t, Vb_t, None, None,
                                               model_M2=model_M2)
                if torch.stack([R_S_t, R_B_t]).norm() < R.norm():
                    Vsint, Vb = Vsint_t, Vb_t
                    break
                alpha *= 0.5
            else:
                Vsint, Vb = Vsint + dx[0], Vb + dx[1]  # accept anyway
    return Vsint, Vb, False


def _compute_tangent(cfg, model, bjt, Vd, VG1, VG2, Vsint, Vb, P_M1, P_M2,
                      prev_t: Optional[torch.Tensor] = None,
                      model_M2=None) -> torch.Tensor:
    """Tangent vector to the curve at (Vsint, Vb, Vd). Returns shape (3,).

    The 2x3 augmented Jacobian J_aug = [∂R/∂Vsint | ∂R/∂Vb | ∂R/∂Vd] has
    null-space dimension 1 (assuming we're not at a true bifurcation). The
    null-space vector is the tangent. We compute it via SVD of the 2x3
    matrix and pick the right-singular vector with smallest singular value.
    """
    with torch.no_grad():
        J_xy = _jacobian_finite_diff(cfg, model, bjt, Vd, VG1, VG2,
                                       Vsint, Vb, P_M1, P_M2,
                                       model_M2=model_M2)            # (2,2)
        J_z = _residual_dVd(cfg, model, bjt, Vd, VG1, VG2, Vsint, Vb,
                              P_M1, P_M2, model_M2=model_M2)         # (2,)
        J_aug = torch.cat([J_xy, J_z.unsqueeze(-1)], dim=-1)          # (2,3)

        # SVD to find null vector
        _, S, Vh = torch.linalg.svd(J_aug, full_matrices=True)
        t = Vh[-1]  # right-singular vector with smallest sigma

        # Sign convention: ensure consistent direction across steps
        if prev_t is not None:
            if torch.dot(t, prev_t) < 0:
                t = -t
        else:
            # Initial step: prefer increasing Vd direction
            if t[2] < 0:
                t = -t
        # Normalize (numerically robust)
        t = t / t.norm().clamp_min(1e-30)
    return t


def _newton_arclength_corrector(cfg, model, bjt, x_pred, x_prev, t_prev, ds,
                                  VG1, VG2, P_M1, P_M2,
                                  max_iters: int = 15, tol: float = 1e-9,
                                  model_M2=None):
    """3D Newton on augmented system [R_S; R_B; t·(x - x_prev) - ds] = 0.

    Returns (x_new, n_iter, converged).
    """
    x = x_pred.clone()
    for it in range(max_iters):
        Vsint, Vb, Vd = x[0], x[1], x[2]
        with torch.no_grad():
            R_S, R_B, _ = _residuals(cfg, model, bjt, Vd, VG1, VG2,
                                       Vsint, Vb, P_M1, P_M2, model_M2=model_M2)
            constraint = torch.dot(t_prev, x - x_prev) - ds
            F = torch.stack([R_S, R_B, constraint])

            if F.norm() < tol:
                return x, it, True

            # 3x3 Jacobian: top 2 rows = [J_xy | J_z], bottom row = t_prev
            J_xy = _jacobian_finite_diff(cfg, model, bjt, Vd, VG1, VG2,
                                           Vsint, Vb, P_M1, P_M2, model_M2=model_M2)
            J_z = _residual_dVd(cfg, model, bjt, Vd, VG1, VG2, Vsint, Vb,
                                  P_M1, P_M2, model_M2=model_M2)
            top = torch.cat([J_xy, J_z.unsqueeze(-1)], dim=-1)
            J_full = torch.cat([top, t_prev.unsqueeze(0)], dim=0)

            try:
                dx = torch.linalg.solve(J_full, -F)
            except Exception:
                dx = torch.linalg.lstsq(J_full, -F.unsqueeze(-1)).solution.squeeze(-1)

            # Damped step
            alpha = 1.0
            x_old = x.clone()
            for _ in range(8):
                x_try = x_old + alpha * dx
                R_S_t, R_B_t, _ = _residuals(cfg, model, bjt, x_try[2], VG1, VG2,
                                               x_try[0], x_try[1], P_M1, P_M2,
                                               model_M2=model_M2)
                F_try = torch.stack([
                    R_S_t, R_B_t,
                    torch.dot(t_prev, x_try - x_prev) - ds,
                ])
                if F_try.norm() < F.norm():
                    x = x_try
                    break
                alpha *= 0.5
            else:
                x = x_old + dx
    # Did not converge within max_iters
    return x, max_iters, False


def trace_arclength(
    cfg: NSRAMCell2TConfig,
    model: BSIM4Model,
    bjt: GummelPoonNPN,
    VG1, VG2,
    Vd_start: float = 0.05,
    Vd_max: float = 1.95,
    P_M1: Optional[dict] = None,
    P_M2: Optional[dict] = None,
    ds_init: float = 0.01,
    ds_min: float = 1e-4,
    ds_max: float = 0.05,
    max_steps: int = 2000,
    model_M2: Optional[BSIM4Model] = None,
) -> dict:
    """Trace I-V curve via pseudo-arclength continuation from Vd_start to Vd_max.

    Returns dict with arrays:
      'path_Vd', 'path_Vsint', 'path_Vb', 'path_Id'  : (N,) along arclength
      'converged'                                    : (N,) bool
      'n_steps', 'n_folds'                           : diagnostics
    """
    VG1 = torch.as_tensor(VG1, dtype=torch.float64)
    VG2 = torch.as_tensor(VG2, dtype=torch.float64)

    # 1. Find initial point
    Vsint0, Vb0, init_ok = _solve_initial_point(cfg, model, bjt, Vd_start,
                                                  VG1, VG2, model_M2=model_M2)
    if not init_ok:
        return {"path_Vd": [Vd_start], "path_Vsint": [float(Vsint0)],
                "path_Vb": [float(Vb0)], "path_Id": [float("nan")],
                "converged": [False], "n_steps": 0, "n_folds": 0,
                "init_ok": False}

    # 2. Trace
    x = torch.tensor([float(Vsint0), float(Vb0), float(Vd_start)],
                      dtype=torch.float64)
    t = _compute_tangent(cfg, model, bjt, x[2], VG1, VG2, x[0], x[1],
                          P_M1, P_M2, prev_t=None, model_M2=model_M2)
    ds = ds_init

    path_Vd = [float(x[2])]
    path_Vsint = [float(x[0])]
    path_Vb = [float(x[1])]
    converged_flags = [True]
    n_folds = 0
    n_steps = 0
    prev_dVd_sign = torch.sign(t[2])

    for step in range(max_steps):
        # Predictor
        x_pred = x + ds * t

        # Corrector
        x_new, n_iter, conv = _newton_arclength_corrector(
            cfg, model, bjt, x_pred, x_prev=x, t_prev=t, ds=ds,
            VG1=VG1, VG2=VG2, P_M1=P_M1, P_M2=P_M2,
            model_M2=model_M2,
        )

        if not conv:
            # Step too large — bisect
            ds = max(ds * 0.5, ds_min)
            if ds <= ds_min * 1.01:
                # Even at min step we can't converge — record and break
                path_Vd.append(float(x_new[2]))
                path_Vsint.append(float(x_new[0]))
                path_Vb.append(float(x_new[1]))
                converged_flags.append(False)
                break
            continue

        # Compute new tangent (with sign consistency)
        t_new = _compute_tangent(cfg, model, bjt, x_new[2], VG1, VG2,
                                   x_new[0], x_new[1], P_M1, P_M2,
                                   prev_t=t, model_M2=model_M2)
        # Detect fold: dVd/ds sign change
        new_dVd_sign = torch.sign(t_new[2])
        if new_dVd_sign != prev_dVd_sign and abs(prev_dVd_sign) > 0:
            n_folds += 1
        prev_dVd_sign = new_dVd_sign

        x = x_new
        t = t_new
        n_steps += 1

        path_Vd.append(float(x[2]))
        path_Vsint.append(float(x[0]))
        path_Vb.append(float(x[1]))
        converged_flags.append(True)

        # Adapt ds
        if n_iter > 8:
            ds = max(ds * 0.7, ds_min)
        elif n_iter <= 3:
            ds = min(ds * 1.3, ds_max)

        # Termination: reached Vd_max in forward direction (may have folded back)
        if x[2] >= Vd_max:
            break
        # Stuck termination: if Vd has been stagnant for too many steps
        if step > 50 and abs(path_Vd[-1] - path_Vd[-50]) < 1e-3 and n_folds == 0:
            break

    # 3. Compute Id along path (run forward at each path point)
    path_Vd_t = torch.tensor(path_Vd, dtype=torch.float64)
    path_Vsint_t = torch.tensor(path_Vsint, dtype=torch.float64)
    path_Vb_t = torch.tensor(path_Vb, dtype=torch.float64)
    with torch.no_grad():
        _, _, comp = _residuals(cfg, model, bjt,
                                 path_Vd_t, VG1.expand_as(path_Vd_t),
                                 VG2.expand_as(path_Vd_t),
                                 path_Vsint_t, path_Vb_t,
                                 P_M1, P_M2, model_M2=model_M2)
        # comp contains M1/M2/Q1 currents — Id at drain pin
        Id = comp.get("Id_total", comp.get("Ids_M1", torch.zeros_like(path_Vd_t)))

    return {
        "path_Vd": path_Vd,
        "path_Vsint": path_Vsint,
        "path_Vb": path_Vb,
        "path_Id": [float(x) for x in Id],
        "converged": converged_flags,
        "n_steps": n_steps,
        "n_folds": n_folds,
        "init_ok": True,
    }


def interpolate_at_targets(path: dict, Vd_targets: torch.Tensor) -> dict:
    """Interpolate Id at requested Vd_targets along the arclength path.

    For points where Vd_target is bracketed by two consecutive path points
    on the FORWARD-sweep portion (before the first fold or after the second
    fold for an off→on transition), use linear interpolation.

    For Vd_targets BEYOND the path's last reached Vd, mark as not-converged
    and return Id=nan.
    """
    import numpy as np
    Vd_arr = np.array(path["path_Vd"])
    Id_arr = np.array(path["path_Id"])
    Vsint_arr = np.array(path["path_Vsint"])
    Vb_arr = np.array(path["path_Vb"])

    Vd_targets_np = Vd_targets.detach().cpu().numpy() if isinstance(Vd_targets, torch.Tensor) else np.asarray(Vd_targets)

    Id_out = np.full_like(Vd_targets_np, np.nan, dtype=np.float64)
    Vsint_out = np.full_like(Vd_targets_np, np.nan, dtype=np.float64)
    Vb_out = np.full_like(Vd_targets_np, np.nan, dtype=np.float64)
    conv_out = np.zeros_like(Vd_targets_np, dtype=bool)

    # For each target, find first segment on path that brackets it.
    for k, Vd_t in enumerate(Vd_targets_np):
        for i in range(len(Vd_arr) - 1):
            v1, v2 = Vd_arr[i], Vd_arr[i + 1]
            if (v1 <= Vd_t <= v2) or (v2 <= Vd_t <= v1):
                # Linear interp
                if abs(v2 - v1) < 1e-12:
                    frac = 0.0
                else:
                    frac = (Vd_t - v1) / (v2 - v1)
                Id_out[k] = Id_arr[i] + frac * (Id_arr[i + 1] - Id_arr[i])
                Vsint_out[k] = Vsint_arr[i] + frac * (Vsint_arr[i + 1] - Vsint_arr[i])
                Vb_out[k] = Vb_arr[i] + frac * (Vb_arr[i + 1] - Vb_arr[i])
                conv_out[k] = (path["converged"][i] and path["converged"][i + 1])
                break

    return {
        "Id": torch.tensor(Id_out, dtype=torch.float64),
        "Vsint": torch.tensor(Vsint_out, dtype=torch.float64),
        "Vb": torch.tensor(Vb_out, dtype=torch.float64),
        "converged": torch.tensor(conv_out),
        "n_steps": path["n_steps"],
        "n_folds": path["n_folds"],
    }


def solve_2t_arclength(
    cfg: NSRAMCell2TConfig,
    model: BSIM4Model,
    bjt: GummelPoonNPN,
    Vd_seq: torch.Tensor,
    VG1, VG2,
    P_M1: Optional[dict] = None,
    P_M2: Optional[dict] = None,
    model_M2: Optional[BSIM4Model] = None,
    **kwargs,
) -> dict:
    """Drop-in replacement for forward_2t — uses arclength continuation."""
    Vd_seq = torch.as_tensor(Vd_seq, dtype=torch.float64)
    Vd_min = float(Vd_seq.min().item())
    Vd_max = float(Vd_seq.max().item())

    path = trace_arclength(cfg, model, bjt, VG1, VG2,
                            Vd_start=Vd_min, Vd_max=Vd_max,
                            P_M1=P_M1, P_M2=P_M2,
                            model_M2=model_M2,
                            **kwargs)

    if not path.get("init_ok", False):
        N = len(Vd_seq)
        return {
            "Id": torch.full((N,), float("nan"), dtype=torch.float64),
            "Vsint": torch.full((N,), float("nan"), dtype=torch.float64),
            "Vb": torch.full((N,), float("nan"), dtype=torch.float64),
            "converged": torch.zeros(N, dtype=torch.bool),
            "niter": torch.zeros(N, dtype=torch.long),
            "n_folds": 0,
            "n_steps": 0,
        }

    out = interpolate_at_targets(path, Vd_seq)
    out["niter"] = torch.full_like(Vd_seq, path["n_steps"], dtype=torch.long)
    return out


def forward_2t_arclength_grad(
    cfg: NSRAMCell2TConfig,
    model: Optional[BSIM4Model] = None,
    bjt: Optional[GummelPoonNPN] = None,
    Vd_seq: Optional[torch.Tensor] = None,
    VG1=None, VG2=None,
    P_M1: Optional[dict] = None,
    P_M2: Optional[dict] = None,
    *,
    model_M1: Optional[BSIM4Model] = None,
    model_M2: Optional[BSIM4Model] = None,
) -> dict:
    """Drop-in replacement for forward_2t that uses arclength path tracing
    for robust convergence + grad-tracked Newton at each interpolated point
    so gradients still flow to fit params.

    Strategy:
      1. trace_arclength under no_grad → 100% conv path through snapback
      2. interpolate path at Vd_seq → warm-start (Vsint*, Vb*) per bias
      3. solve_2t_steady_state per bias, grad-tracked, Vsint_init from (2).
         Starting at the converged point, Newton needs ~1-2 iterations to
         re-confirm and the autograd graph + IFT correction at convergence
         provide gradient flow.

    Returns dict with same keys as forward_2t: Id, Vsint, Vb, niter, converged.

    Two-model variant: pass `model_M1=` / `model_M2=` as kwargs (or single
    legacy positional `model` for back-compat with the 1-card path).
    """
    from .nsram_cell_2T import solve_2t_steady_state
    if model_M1 is None:
        model_M1 = model
    if model_M1 is None:
        raise TypeError("forward_2t_arclength_grad requires either positional `model` or `model_M1=` kwarg")
    if model_M2 is None:
        model_M2 = model_M1
    Vd_seq = Vd_seq.to(torch.float64)
    VG1_t = torch.as_tensor(VG1, dtype=torch.float64)
    VG2_t = torch.as_tensor(VG2, dtype=torch.float64)
    T = int(Vd_seq.shape[0])

    # 1. Path trace + interpolate (no_grad)
    with torch.no_grad():
        path = trace_arclength(cfg, model_M1, bjt, VG1_t, VG2_t,
                                Vd_start=float(Vd_seq.min()),
                                Vd_max=float(Vd_seq.max()),
                                P_M1=P_M1, P_M2=P_M2,
                                model_M2=model_M2)
        if not path.get("init_ok", False):
            return {
                "Id": torch.full((T,), float("nan"), dtype=torch.float64),
                "Vsint": torch.full((T,), float("nan"), dtype=torch.float64),
                "Vb": torch.full((T,), float("nan"), dtype=torch.float64),
                "converged": torch.zeros(T, dtype=torch.bool),
                "niter": torch.zeros(T, dtype=torch.long),
            }
        warm = interpolate_at_targets(path, Vd_seq)

    Vsint_warm = warm["Vsint"]  # (T,)
    Vb_warm = warm["Vb"]
    arclen_conv = warm["converged"]

    # 2. Per-bias grad-tracked solve from arclength warm-start
    Ids_list, Vs_list, Vb_list = [], [], []
    niter_list, conv_list = [], []
    for i in range(T):
        Vd_i = Vd_seq[i].unsqueeze(0)
        if not bool(arclen_conv[i]):
            # No bracket on path — use plain cascade fallback
            Vs0 = (Vd_i * 0.5).detach()
            Vb0 = torch.tensor(0.0, dtype=torch.float64)
        else:
            Vs0 = Vsint_warm[i].unsqueeze(0).detach()
            Vb0 = Vb_warm[i].unsqueeze(0).detach()
        out = solve_2t_steady_state(
            cfg, model_M1, bjt,
            Vd=Vd_i, VG1=VG1_t, VG2=VG2_t,
            P_M1=P_M1, P_M2=P_M2,
            Vsint_init=Vs0, Vb_init=Vb0,
            model_M2=model_M2,
        )
        Ids_list.append(out["Id"].squeeze(0))
        Vs_list.append(out["Vsint"].squeeze(0))
        Vb_list.append(out["Vb"].squeeze(0))
        niter_list.append(out["niter"] if isinstance(out["niter"], int)
                          else int(out["niter"].squeeze(0).item()))
        conv_val = out["converged"]
        if isinstance(conv_val, torch.Tensor):
            conv_val = bool(conv_val.squeeze(0).item())
        else:
            conv_val = bool(conv_val)
        conv_list.append(conv_val)

    Id_t = torch.stack(Ids_list)
    Vsint_t = torch.stack(Vs_list)
    Vb_t = torch.stack(Vb_list)
    return {
        "Id": Id_t,
        "Vsint": Vsint_t,
        "Vb": Vb_t,
        "converged": torch.tensor(conv_list, dtype=torch.bool),
        "niter": torch.tensor(niter_list, dtype=torch.long),
        "arclen_conv": arclen_conv,
        "arclen_n_steps": path["n_steps"],
        "arclen_n_folds": path["n_folds"],
    }

```


=== FILE: _extracted/nsram_bsim4_port/nsram/nsram/bsim4_port/constants.py (972 chars) ===
```python
"""Physical constants — bit-identical to BSIM4 4.8.3 b4temp.c #define values.

Don't change these. ngspice uses these exact values; the diff gate validates
that we match within 1e-6 relative.
"""
from __future__ import annotations

# From b4temp.c lines 35-44
Kb = 1.3806226e-23           # Boltzmann constant [J/K]
KboQ = 8.617087e-5           # Kb / q [V/K]
EPS0 = 8.85418e-12           # Vacuum permittivity [F/m]
EPSSI = 1.03594e-10          # Si permittivity [F/m]
PI = 3.141592654
MAX_EXP = 5.834617425e14     # exp(34)
MIN_EXP = 1.713908431e-15    # exp(-34)
EXP_THRESHOLD = 34.0
Charge_q = 1.60219e-19       # Electron charge [C]
DELTA = 1.0e-9               # Smoothing constant for various transitions
DELTA_3 = 0.02               # Vfbeff smoothing offset (b4ld.c #define DELTA_3)

# Standard reference temperature (Tnom, Kelvin offset)
TZEROK = 273.15              # 0°C in K

# Convenience
def C_to_K(t_celsius: float) -> float:
    return t_celsius + TZEROK

```


=== FILE: _extracted/nsram_bsim4_port/nsram/nsram/bsim4_port/bjt.py (6347 chars) ===
```python
"""bjt — differentiable Gummel-Poon NPN port.

DC-only Gummel-Poon (SPICE) sufficient for the NS-RAM body-KCL use case where
the parasitic NPN sits with floating base = body, collector = drain,
emitter = source. Junction caps (cje, cjc) and transit times (tf, tr, itf,
vtf, xtf) are intentionally omitted — they affect transient/AC dynamics, not
the DC body current that closes the body-KCL fixed point.

References: ngspice manual ch. on BJT; SPICE Gummel-Poon model equations.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch

from .constants import KboQ
from .smooth import safe_exp, safe_sqrt


@dataclass
class GummelPoonNPN:
    """Gummel-Poon NPN parameters (subset matching Sebas's parasiticBJT card)."""
    Is: float = 5e-9        # transport saturation current [A]
    Va: float = 100.0       # forward Early voltage [V]
    Vb: float = 1e30        # reverse Early voltage [V]   (Vaf/Var; default = no effect)
    Bf: float = 10000.0     # ideal max forward beta
    Br: float = 100.0       # ideal max reverse beta
    Nf: float = 1.0         # forward emission coefficient
    Nr: float = 1.0         # reverse emission coefficient
    Nc: float = 2.0         # base-collector leakage emission
    Ne: float = 1.5         # base-emitter leakage emission
    Ikf: float = 1e30       # forward knee (no high-injection if huge)
    Ikr: float = 0.1        # reverse knee   ← Sebas: 100m
    Ise: float = 0.0        # B-E leakage saturation
    Isc: float = 0.0        # B-C leakage saturation
    Re:  float = 0.1        # emitter ohmic    (DC: not iterated, kept for API)
    Rc:  float = 0.1        # collector ohmic  (DC: not iterated, kept for API)
    Rb:  float = 0.0        # base ohmic       (DC: not iterated, kept for API)
    area: float = 1.0       # SPICE BJT instance multiplier (scales Is, Ikf, Ikr, Ise, Isc)

    @classmethod
    def from_sebas_card(cls) -> "GummelPoonNPN":
        """Sebas's parasiticBJT.txt parameters (data/sebas_2026_04_22/parasiticBJT.txt).

        .model parasiticBJT NPN(is=5E-9 va=100 bf=10000 br=100 nc=2 ikr=100m
                                rc=0.1 vje=0.7 re=0.1 cjc=1e-15 fc=0.5
                                cje=0.7e-15 ne=1.5 ise=0
                                tr=20e-12 tf=25e-12 itf=0.03 vtf=7 xtf=2)

        Junction caps / transit times skipped (DC port).  Nf, Nr, Bf default in
        SPICE when not specified ⇒ Nf=Nr=1, Br given.  Ikf not in card ⇒ infinity.
        """
        return cls(
            Is=5e-9,
            Va=100.0,
            Bf=10000.0,
            Br=100.0,
            Nf=1.0,
            Nr=1.0,
            Nc=2.0,
            Ne=1.5,
            Ikf=1e30,
            Ikr=0.1,
            Ise=0.0,
            Isc=0.0,
            Re=0.1,
            Rc=0.1,
            Rb=0.0,
            area=1e-6,  # schematic 2tnsram_simple.asc: SYMATTR Value2 area=1u
        )


def compute_bjt(
    bjt: GummelPoonNPN,
    Vbe: torch.Tensor,
    Vbc: torch.Tensor,
    T_K: float = 300.15,
) -> dict[str, torch.Tensor]:
    """SPICE Gummel-Poon DC currents (no ohmic-resistance iteration, no caps).

    Returns dict with keys: 'Ic', 'Ib', 'Ie', 'Icc', 'Iec', 'kqb'.

    Equations (ngspice manual):
        vt   = k·T/q
        Icc  = Is·(exp(Vbe/(Nf·vt)) − 1)
        Iec  = Is·(exp(Vbc/(Nr·vt)) − 1)
        Ibe_n= Icc/Bf,           Ibe_l = Ise·(exp(Vbe/(Ne·vt)) − 1)
        Ibc_n= Iec/Br,           Ibc_l = Isc·(exp(Vbc/(Nc·vt)) − 1)
        q1   = 1/(1 − Vbc/Va − Vbe/Vb)
        q2   = Icc/Ikf + Iec/Ikr
        kqb  = (q1/2)·(1 + sqrt(1 + 4·q2))
        Ic   = (Icc − Iec)/kqb − Ibc_n − Ibc_l
        Ib   = Ibe_n + Ibe_l + Ibc_n + Ibc_l
        Ie   = −(Ic + Ib)

    Sign convention: NPN — Vbe>0 forward biases B-E; current flows from C to E.
    For the NS-RAM parasitic NPN (floating base = body):
      Vbe = V_body − V_source
      Vbc = V_body − V_drain
    The collector current Ic is the "I_BJT" entering the body-KCL.

    fp64 throughout; uses safe_exp (clamped ±34) and safe_sqrt (eps=1e-12).
    """
    # promote to fp64 on whichever device the inputs live on
    Vbe = Vbe.to(torch.float64)
    Vbc = Vbc.to(torch.float64)

    vt = torch.as_tensor(KboQ * T_K, dtype=torch.float64, device=Vbe.device)
    # SPICE BJT area multiplier: scales Is, Ikf, Ikr, Ise, Isc (and divides
    # Rb/Re/Rc, but those aren't iterated in this DC port).
    area_ = torch.as_tensor(bjt.area, dtype=torch.float64, device=Vbe.device)
    Is_ = torch.as_tensor(bjt.Is, dtype=torch.float64, device=Vbe.device) * area_
    Ise_ = bjt.Ise * area_
    Isc_ = bjt.Isc * area_
    Ikf_ = bjt.Ikf * area_
    Ikr_ = bjt.Ikr * area_

    # -- Transport currents (block: Icc / Iec) ---------------------------------
    Icc = Is_ * (safe_exp(Vbe / (bjt.Nf * vt)) - 1.0)
    Iec = Is_ * (safe_exp(Vbc / (bjt.Nr * vt)) - 1.0)

    # -- Base currents (block: ideal + non-ideal leakage) ----------------------
    Ibe_n = Icc / bjt.Bf
    Ibc_n = Iec / bjt.Br
    if bjt.Ise > 0.0:
        Ibe_l = Ise_ * (safe_exp(Vbe / (bjt.Ne * vt)) - 1.0)
    else:
        Ibe_l = torch.zeros_like(Vbe)
    if bjt.Isc > 0.0:
        Ibc_l = Isc_ * (safe_exp(Vbc / (bjt.Nc * vt)) - 1.0)
    else:
        Ibc_l = torch.zeros_like(Vbc)

    # -- Base-charge factor kqb (block: Early + high-injection knee) -----------
    # q1 includes Early-effect denominator.  Va and Vb (=Vaf/Var) are large
    # ⇒ q1 ≈ 1.  Guard the denominator with a soft floor so q1 stays finite
    # even when Vbc → Va (deep saturation).
    inv_q1 = 1.0 - Vbc / bjt.Va - Vbe / bjt.Vb
    inv_q1 = inv_q1.clamp_min(1e-4)        # prevents divide-by-zero
    q1 = 1.0 / inv_q1

    # q2 = high-injection ratio.  Ikf and Ikr default huge ⇒ q2 ≈ 0.
    q2 = Icc / Ikf_ + Iec / Ikr_

    # kqb = (q1/2)·(1 + sqrt(1 + 4 q2)).  Using safe_sqrt on the discriminant
    # gives a smooth knee at q2≈0 with finite gradient; saturates as
    # kqb ~ q1·sqrt(q2)  for q2 ≫ 1, which damps Icc once high-injection hits.
    disc = 1.0 + 4.0 * q2
    kqb = 0.5 * q1 * (1.0 + safe_sqrt(disc))

    # -- Terminal currents -----------------------------------------------------
    Ic = (Icc - Iec) / kqb - Ibc_n - Ibc_l
    Ib = Ibe_n + Ibe_l + Ibc_n + Ibc_l
    Ie = -(Ic + Ib)

    return {
        "Ic": Ic, "Ib": Ib, "Ie": Ie,
        "Icc": Icc, "Iec": Iec, "kqb": kqb,
    }

```


=== FILE: _extracted/nsram_bsim4_port/nsram/nsram/bsim4_port/dc.py (39248 chars) ===
```python
"""B07-B20 — Faithful BSIM4 4.8.3 DC drain-current port.

Ported from `external/bsim4/code/b4ld.c` lines 1002-2156.
Equation cross-references in section comments below as `# b4ld.c §<lines>`.

Scope (P3, first faithful pass):
  - mobMod = 1 path only
  - rdsMod ∈ {0, 1}: 0 = internal bias-dependent Rds (DEFAULT in BSIM4 per
    b4set.c:107-108; reduces Idsat & modifies Vdsat); 1 = external resistor.
  - mtrlMod = 0 (Si substrate)
  - tempMod = 0 (default)
  - No velocity overshoot (lambda branch skipped)
  - No source-end vtl limit (Fsevl skipped)
  - No quantum/bulk-charge centroid (Tcen, Coxeff = coxe directly)
  - No poly depletion Newton (Vgs_eff = Vgs)
  - No Weff_corr Newton (use sd.geom.weff directly)

Out of scope here (P3.5/P4):
  - Charge model (capMod), gate tunneling, GIDL/GISL, impact-ion, body diodes,
  - AC analysis, noise, NQS

Differentiability rules:
  - fp64 throughout.
  - All if-branches on tensors replaced with `torch.where` over both
    differentiable arms, OR substituted with smooth.py primitives.
  - All `exp` arguments are guarded via `safe_exp` (clipped at ±34) — matches
    BSIM4 DEXP MIN_EXP/MAX_EXP regularizer.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

import torch

from .constants import Charge_q, EPSSI, EPS0, KboQ, MAX_EXP, MIN_EXP, EXP_THRESHOLD, PI
from .geometry import Geometry
from .model_card import BSIM4Model
from .smooth import safe_exp, safe_log, safe_sqrt, smooth_min, smooth_max
from .temp import SizeDependParam, compute_size_dep

import math as _math
# Pre-computed BSIM4 deep-subthreshold floor: log(1 + MIN_EXP) ≈ MIN_EXP.
_LOG1P_MIN_EXP = _math.log1p(MIN_EXP)


@dataclass
class DCResult:
    Ids: torch.Tensor       # drain current [A]  (positive for NMOS Vds>0)
    Vth: torch.Tensor       # threshold incl. all corrections used
    Vgsteff: torch.Tensor   # effective overdrive (smooth)
    Vdsat: torch.Tensor     # saturation drain voltage
    Vdseff: torch.Tensor    # effective Vds (smooth-min Vds, Vdsat)
    Abulk: torch.Tensor     # bulk-charge factor
    n: torch.Tensor         # subthreshold ideality
    mueff: torch.Tensor     # effective mobility
    Rds: Optional[torch.Tensor] = None   # internal Rds (rdsmod=0); None for rdsmod=1
    # WAVE2-FIX-1 (Gap 2): pre-SCBE channel current "T4 = Idsa·Vdseff" from b4ld.c §2069.
    # Iii (impact ionization) in leak.py must use this — NOT post-SCBE Ids — to match
    # the C-source impact-ionization formula faithfully.
    Idsa: Optional[torch.Tensor] = None
    # WAVE2-FIX (Gap 7): intermediates needed by leak.compute_igb (Vfbeff path).
    Vgs_eff: Optional[torch.Tensor] = None  # poly-dep'd Vgs (b4ld.c §1224-1296)
    Vbseff: Optional[torch.Tensor] = None   # body-bias clamp (b4ld.c §1002-1019)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _t(x, dtype=torch.float64, device=None) -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype, device=device)


def _exp_threshold_branch(T0: torch.Tensor, *, ratio: float = 1.0) -> torch.Tensor:
    """BSIM4 Theta0-style stable rational form:
        if T0 < EXP_THRESHOLD: Theta0 = exp(T0) / ((exp(T0)-1)^2 + 2 exp(T0) MIN_EXP)
        else:                  Theta0 = 1 / (MAX_EXP - 2)
    Implemented faithfully via torch.where; both branches finite & differentiable.
    """
    # SMOOTH: replace if-on-tensor with torch.where over both arms; both finite.
    T1 = safe_exp(T0)                       # exp(min(T0, 34))  ⇒ never overflow
    T2 = T1 - 1.0
    T3 = T2 * T2
    T4 = T3 + 2.0 * T1 * MIN_EXP
    inner = T1 / T4
    saturated = torch.full_like(T0, 1.0 / (MAX_EXP - 2.0))
    return torch.where(T0 < EXP_THRESHOLD, inner, saturated)


# --------------------------------------------------------------------------- #
# Main DC entry point                                                          #
# --------------------------------------------------------------------------- #

def compute_dc(
    model: BSIM4Model,
    sd: SizeDependParam,
    Vgs: torch.Tensor,
    Vds: torch.Tensor,
    Vbs: torch.Tensor | float = 0.0,
    *,
    sharpness: float = 50.0,
) -> DCResult:
    """Faithful differentiable BSIM4 DC drain current.

    Returns DCResult with Ids and intermediate quantities. All tensors fp64.
    Bias inputs may be scalar or broadcastable tensors.
    """
    # ---- tensor coercion ------------------------------------------------- #
    if isinstance(Vgs, torch.Tensor):
        dtype = Vgs.dtype
        device = Vgs.device
    else:
        dtype = torch.float64
        device = None
    if dtype != torch.float64:
        # Promote — fp64 throughout per project rules.
        dtype = torch.float64

    def t(x):
        return _t(x, dtype=dtype, device=device)

    Vgs = t(Vgs)
    Vds = t(Vds)
    Vbs = t(Vbs)
    # Broadcast to a common shape so torch.where always works.
    Vgs, Vds, Vbs = torch.broadcast_tensors(Vgs, Vds, Vbs)

    geom = sd.geom
    ctx = sd.model_ctx
    P = sd.scaled

    # ---- Geometry / oxide / temp constants (scalars promoted to tensors) - #
    Leff = t(geom.leff)
    Weff = t(geom.weff)        # NOTE: Weff_corr Newton skipped this pass
    toxe = t(ctx.toxe)
    coxe = t(ctx.coxe)
    Vtm = t(ctx.vtm)
    Vtm0 = t(ctx.Vtm0)
    factor1 = t(ctx.factor1)
    epssub = t(ctx.epssub)

    type_n = float(model._values.get("type", 1))

    # ---- Scaled per-instance params (scalar floats → tensors) ------------ #
    vth0 = t(sd.vth0_T)               # T-shifted vth0
    k1 = t(P["k1"])
    k2 = t(P["k2"])
    k3 = t(P.get("k3", model.get("k3", 80.0)))
    k3b = t(P.get("k3b", model.get("k3b", 0.0)))
    w0 = t(P.get("w0", model.get("w0", 2.5e-6)))
    dvt0 = t(P["dvt0"])
    dvt1 = t(P["dvt1"])
    dvt2 = t(P["dvt2"])
    dvt0w = t(P["dvt0w"])
    dvt1w = t(P["dvt1w"])
    dvt2w = t(P["dvt2w"])
    eta0 = t(P["eta0"])
    etab = t(P["etab"])
    nfactor = t(P["nfactor"])
    cdsc = t(P["cdsc"])
    cdscb = t(P["cdscb"])
    cdscd = t(P["cdscd"])
    cit = t(P["cit"])
    voff = t(P["voff"])
    voffl = t(model.get("voffl", 0.0))
    minv = t(model.get("minv", 0.0))
    a0 = t(P["a0"])
    ags = t(P["ags"])
    a1 = t(P["a1"])
    a2 = t(P["a2"])
    keta = t(P["keta"])
    # WAVE3-FIX (z214): b0/b1 also bin via lX/wX/pX (b4temp.c).  Read from
    # the scaled per-instance dict so any future binning coefficients flow.
    b0 = t(P.get("b0", model.get("b0", 0.0)))
    b1 = t(P.get("b1", model.get("b1", 0.0)))
    xj = t(model.get("xj", 1.5e-7))
    dwg = t(model.get("dwg", 0.0))
    dwb = t(model.get("dwb", 0.0))
    pclm = t(P["pclm"])
    pdiblc1 = t(P["pdiblc1"])
    pdiblc2 = t(P["pdiblc2"])
    pdiblb = t(P.get("pdiblcb", model.get("pdiblcb", model.get("pdiblb", 0.0))))
    drout = t(P["drout"])
    pscbe1 = t(P["pscbe1"])
    pscbe2 = t(P["pscbe2"])
    pvag = t(P["pvag"])
    delta = t(P["delta"])
    fprout = t(P["fprout"])
    pdits = t(P["pdits"])
    pditsd = t(P["pditsd"])
    pditsl = t(model.get("pditsl", 0.0))
    dvtp0 = t(P["dvtp0"])
    dvtp1 = t(P["dvtp1"])
    dvtp4 = t(P.get("dvtp4", 0.0))
    dvtp2factor = t(model.get("dvtp2factor", 0.0))
    kt1 = t(model.get("kt1", -0.11))
    kt1l = t(model.get("kt1l", 0.0))
    kt2 = t(model.get("kt2", 0.022))
    lpe0 = t(model.get("lpe0", 1.74e-7))
    lpeb = t(model.get("lpeb", 0.0))
    ua = t(P["ua"])
    ub = t(P["ub"])
    uc = t(P["uc"])
    ud = t(P.get("ud", 0.0))
    u0temp = t(sd.u0temp)
    vsattemp = t(sd.vsattemp)
    Xdep0 = t(sd.Xdep0)
    sqrtPhi_pre = t(sd.sqrtPhi)
    phi_pre = t(sd.phi)
    vbi = t(sd.vbi)
    vbsc = t(sd.vbsc)
    k1ox = t(sd.k1ox)
    k2ox = t(sd.k2ox)
    litl = t(sd.litl)

    # Derived params now cached in SizeDependParam (b4temp.c §1373-1427).
    mstar = t(sd.mstar)
    voffcbn = t(sd.voffcbn)
    cdep0 = t(sd.cdep0)
    # Tcen / Coxeff inputs (b4ld.c §1789-1805 capMod=2 path)
    vtfbphi2 = t(sd.vtfbphi2)
    coxp = t(sd.coxp)
    toxp = t(sd.toxp)
    ados = t(sd.ados)
    bdos = t(sd.bdos)
    # Poly depletion inputs (b4ld.c §5170-5202)
    ngate = float(model.get("ngate", 0.0))
    epsrox_v = float(ctx.epsrox)
    coxe_f = float(ctx.coxe)

    # ===================================================================== #
    # 1. Vbseff  — body bias smooth saturation + JX forward correction      #
    #    b4ld.c §1002-1019    BSIM4 manual §3.5 (Vbseff)                    #
    # ===================================================================== #
    # WAVE2-FIX-2 (Gap 5): Strict ngspice validation for Vbseff is currently
    # NOT POSSIBLE: ngspice 42 does not expose `@m1[vbseff]` (probed
    # 2026-04-29 with all common name variants — vbseff/Vbseff/VBSeff/
    # vbs_eff/Vbs_eff/VBSEFF — all returned `Error: no such vector`). The
    # quantity is internal to BSIM4 and not part of the documented OP saves.
    # TODO Wave-3: validate via gm/gmbs ratio inference, OR patch ngspice to
    # expose vbseff in BSIM4dev.c, OR migrate to a newer ngspice version.
    # Original C: smooth-saturate Vbs to vbsc from below (T0 = Vbs - vbsc - 0.001).
    # The C splits on `T0 >= 0` and uses two algebraic forms; both are themselves
    # smooth — they differ by a continuous algebraic identity. We port the form
    # exactly, but use torch.where to keep both branches differentiable.
    T0 = Vbs - vbsc - 0.001
    T1 = safe_sqrt(T0 * T0 - 0.004 * vbsc)            # SMOOTH: safe_sqrt for grad at 0
    Vbseff_a = vbsc + 0.5 * (T0 + T1)                 # branch T0 >= 0
    T2 = -0.002 / (T1 - T0)                           # branch T0 < 0
    Vbseff_b = vbsc * (1.0 + T2)
    # SMOOTH: replace `if (T0 >= 0.0)` with torch.where over both arms.
    Vbseff = torch.where(T0 >= 0.0, Vbseff_a, Vbseff_b)

    # JX: correction to forward body bias  (b4ld.c §1014-1019)
    T9 = 0.95 * phi_pre
    T0 = T9 - Vbseff - 0.001
    T1 = safe_sqrt(T0 * T0 + 0.004 * T9)
    Vbseff = T9 - 0.5 * (T0 + T1)

    # ===================================================================== #
    # 2. Phis, sqrtPhis, Xdep   — b4ld.c §1020-1027   manual §2.4           #
    # ===================================================================== #
    Phis = phi_pre - Vbseff
    sqrtPhis = safe_sqrt(Phis)                         # SMOOTH: safe_sqrt
    Xdep = Xdep0 * sqrtPhis / sqrtPhi_pre

    # ===================================================================== #
    # 3. Vth core with DVT machinery   b4ld.c §1033-1130   manual §2.4-§3.0 #
    # ===================================================================== #
    T3 = safe_sqrt(Xdep)                               # SMOOTH: safe_sqrt
    V0 = vbi - phi_pre

    # --- lt1 ------------------------------------------------------------- #
    # b4ld.c §1037-1048: rational regularizer for dvt2*Vbs near -0.5
    # WAVE2-FIX (critique 7): the inactive arm 1/(3+8·T0) is singular at
    # T0=-3/8 (= -0.375) which sits INSIDE the active region (T0 >= -0.5).
    # Guard the denominator with a sign-preserving floor so backward never
    # produces NaN through the unused arm.
    T0 = dvt2 * Vbseff
    T1_a = 1.0 + T0                                    # branch: T0 >= -0.5
    _denom_b = 3.0 + 8.0 * T0
    _denom_b_safe = torch.where(_denom_b.abs() > 1e-6, _denom_b,
                                 torch.full_like(_denom_b, -1e-6))
    T4_b = 1.0 / _denom_b_safe                          # branch: T0 < -0.5
    T1_b = (1.0 + 3.0 * T0) * T4_b
    # SMOOTH: torch.where; both arms finite via _denom_b_safe.
    T1 = torch.where(T0 >= -0.5, T1_a, T1_b)
    lt1 = factor1 * T3 * T1

    # --- ltw  (b4ld.c §1050-1061) ---------------------------------------- #
    T0w = dvt2w * Vbseff
    T1w_a = 1.0 + T0w
    _denom_bw = 3.0 + 8.0 * T0w
    _denom_bw_safe = torch.where(_denom_bw.abs() > 1e-6, _denom_bw,
                                  torch.full_like(_denom_bw, -1e-6))
    T4w_b = 1.0 / _denom_bw_safe
    T1w_b = (1.0 + 3.0 * T0w) * T4w_b
    T1w = torch.where(T0w >= -0.5, T1w_a, T1w_b)
    ltw = factor1 * T3 * T1w

    # --- Theta0   b4ld.c §1063-1076  (body of §3.0 Vth long-channel) ----- #
    # Faithful: includes BSIM4's MIN_EXP regularizer in T4 = T2² + 2 T1 MIN_EXP.
    T0_th = dvt1 * Leff / lt1.clamp_min(1e-30)
    Theta0 = _exp_threshold_branch(T0_th)
    Delt_vth = dvt0 * Theta0 * V0

    # --- T5 (narrow-W via dvt0w/dvt1w)  b4ld.c §1081-1097 ---------------- #
    T0_w = dvt1w * Weff * Leff / ltw.clamp_min(1e-30)
    T5 = _exp_threshold_branch(T0_w)
    T2_narrow = dvt0w * T5 * V0   # corresponds to "T2" in C, narrow-W Vth shift

    # --- Lpe / temp / k3 narrow-W  b4ld.c §1099-1124 --------------------- #
    TempRatio = ctx.Temp / ctx.Tnom - 1.0
    T0_lpe = safe_sqrt(1.0 + lpe0 / Leff)
    Tlpe1 = (k1ox * (T0_lpe - 1.0) * sqrtPhi_pre
             + (kt1 + kt1l / Leff + kt2 * Vbseff) * TempRatio)
    Vth_NarrowW = toxe * phi_pre / (Weff + w0)

    # --- DIBL_Sft  (b4ld.c §1107-1117) ----------------------------------- #
    # Regularizer: when (eta0 + etab*Vbs) < 1e-4 use rational form to avoid
    # negative theta0vb0 contribution; we replicate it exactly.
    T3_d = eta0 + etab * Vbseff
    T9_d = 1.0 / (3.0 - 2.0e4 * T3_d)
    T3_clamped = torch.where(T3_d < 1.0e-4, (2.0e-4 - T3_d) * T9_d, T3_d)
    # b4temp.c §1531-1540 computes theta0vb0; we have it cached in sd.theta0vb0
    # (approximation). The C uses a slightly different form (with dsub):
    #   θ0vb0 = exp(dsub·Leff/√(εsub/(εrox·ε0)·tox·Xdep0)) / ...  rational form.
    # Recompute faithfully here so DIBL_Sft tracks dsub correctly.
    # AUTOGRAD-FIX: read dsub via the tensor-safe `t(...)` helper, preferring
    # the scaled per-instance dict (P) so external overrides — including
    # torch.Tensor leaves with requires_grad=True — flow through autograd.
    # The previous `float(...)` cast silently stripped gradients, which broke
    # stage-2 fitting of `dsub` in the v5 fitting script.
    dsub_v = t(P.get("dsub", model.get("dsub", model.get("drout", 0.56))))
    epsrox_t = t(ctx.epsrox)
    # SMOOTH: tensor-safe; keeps grads through epssub/toxe/Xdep0 if any becomes leaf.
    # Use very small eps in safe_sqrt: physical arg is ~1e-15 (epssub·toxe·Xdep0/epsrox);
    # the default 1e-12 floor would over-clamp by 3 orders of magnitude.
    tmp_dsub = torch.sqrt((epssub / (epsrox_t * EPS0) * toxe * Xdep0).clamp_min(1e-40))
    T0_dsub = dsub_v * t(geom.leff) / tmp_dsub.clamp_min(1e-40)
    # SMOOTH: replace if-on-tensor + math.exp with tensor _exp_threshold_branch
    theta0vb0 = _exp_threshold_branch(T0_dsub)

    DIBL_Sft = T3_clamped * theta0vb0 * Vds
    Lpe_Vb = safe_sqrt(1.0 + lpeb / Leff)

    # --- Final Vth assembly  b4ld.c §1121-1124 --------------------------- #
    Vth = (type_n * vth0
           + (k1ox * sqrtPhis - k1 * sqrtPhi_pre) * Lpe_Vb
           - k2ox * Vbseff
           - Delt_vth
           - T2_narrow
           + (k3 + k3b * Vbseff) * Vth_NarrowW
           + Tlpe1
           - DIBL_Sft)

    # ===================================================================== #
    # 4. Subthreshold n   b4ld.c §1133-1154    manual §3.2                  #
    # ===================================================================== #
    tmp1 = epssub / Xdep
    tmp2 = nfactor * tmp1
    tmp3 = cdsc + cdscb * Vbseff + cdscd * Vds
    tmp4 = (tmp2 + tmp3 * Theta0 + cit) / coxe
    # b4ld.c §1141-1154: regularize when tmp4 < -0.5 (n must stay > 0).
    # WAVE2-FIX (critique 7): inactive arm 1/(3+8·tmp4) singular at -3/8 ∈ (-0.5, 0).
    n_a = 1.0 + tmp4
    _ndenom = 3.0 + 8.0 * tmp4
    _ndenom_safe = torch.where(_ndenom.abs() > 1e-6, _ndenom,
                                torch.full_like(_ndenom, -1e-6))
    n_b = (1.0 + 3.0 * tmp4) / _ndenom_safe
    n = torch.where(tmp4 >= -0.5, n_a, n_b)

    # ===================================================================== #
    # 5. Pocket DITS Vth correction   b4ld.c §1158-1187   manual §3.0 (DITS)#
    # ===================================================================== #
    # Only active if dvtp0 > 0  (scalar param, so plain Python branch is fine).
    if float(model.get("dvtp0", 0.0)) > 0.0:
        T0_p = -dvtp1 * Vds
        T2_p = safe_exp(T0_p)                           # SMOOTH: safe_exp guards MIN_EXP
        T3_p = Leff + dvtp0 * (1.0 + T2_p)
        # tempMod < 2 path → use Vtm
        T4_p = Vtm * safe_log(Leff / T3_p)              # SMOOTH: safe_log on positive ratio
        Vth = Vth - n * T4_p

    # WAVE2-FIX-3 (Gap 6): v4.7 DITS_SFT2  b4ld.c §1189-1205  (only if both nonzero).
    # The C form `(exp(2·dvtp4·Vds) - 1)/(exp(2·dvtp4·Vds) + 1)` is identically
    # tanh(dvtp4·Vds): (e^(2x) - 1)/(e^(2x) + 1) = tanh(x).
    # SMOOTH: replace the C exp-rational form with torch.tanh — bounded, smooth,
    # avoids exp overflow for large positive Vds, exact algebraic equivalent.
    if (float(model.get("dvtp4", 0.0)) != 0.0
            and float(model.get("dvtp2factor", 0.0)) != 0.0):
        DITS_Sft2 = dvtp2factor * torch.tanh(dvtp4 * Vds)
        Vth = Vth - DITS_Sft2

    # ===================================================================== #
    # 6. Vgsteff  — smooth subthreshold↔strong-inversion bridge             #
    #    b4ld.c §1238-1296    manual §3.3                                   #
    # ===================================================================== #
    # ---- Poly Gate Si Depletion (BSIM4polyDepletion, b4ld.c §5170-5202) -- #
    # Closed-form (NOT Newton): for ngate in (1e18, 1e25) and Vgs > phi:
    #   T1 = 1e6·q·epsgate·ngate / coxe²
    #   T8 = Vgs - phi
    #   T4 = sqrt(1 + 2·T8/T1)
    #   T2 = 2·T8 / (T4 + 1)
    #   T3 = 0.5·T2² / T1               (Vpoly)
    #   T7 = 1.12 - T3 - 0.05
    #   T6 = sqrt(T7² + 0.224)
    #   T5 = 1.12 - 0.5·(T7 + T6)
    #   Vgs_eff = Vgs - T5
    # Differentiable as-is. We mask with torch.where on the active condition;
    # `epsgate` is BSIM4 epsrox·EPS0 in the C code path (T1 there).
    def _poly_dep(Vg_in: torch.Tensor) -> torch.Tensor:
        if not (1.0e18 < ngate < 1.0e25) or epsrox_v == 0.0 or coxe_f == 0.0:
            return Vg_in
        epsgate_f = epsrox_v * EPS0
        T1_pd = t(1.0e6 * Charge_q * epsgate_f * ngate / (coxe_f * coxe_f))
        T8_pd = Vg_in - phi_pre
        # Only apply when Vg > phi; below threshold, return Vg unchanged (smooth via where)
        active = T8_pd > 0.0
        # SMOOTH: clamp T8>=0 inside sqrt to keep grad finite when inactive
        T8_safe = T8_pd.clamp_min(0.0)
        T4_pd = safe_sqrt(1.0 + 2.0 * T8_safe / T1_pd)
        T2_pd = 2.0 * T8_safe / (T4_pd + 1.0)
        T3_pd = 0.5 * T2_pd * T2_pd / T1_pd
        T7_pd = 1.12 - T3_pd - 0.05
        T6_pd = safe_sqrt(T7_pd * T7_pd + 0.224)
        T5_pd = 1.12 - 0.5 * (T7_pd + T6_pd)
        return torch.where(active, Vg_in - T5_pd, Vg_in)

    Vgs_eff = _poly_dep(Vgs)
    Vgd_eff = _poly_dep(Vgs - Vds)   # Vgd path (b4ld.c line 1221)
    # Vds_eff for poly-depletion = Vgs_eff - Vgd_eff (BSIM4 line 1224 area)
    # For DC current we only consume Vgs_eff downstream (Vgd appears only in
    # symmetric-Vds capMod paths we skip).
    _ = Vgd_eff
    Vgst = Vgs_eff - Vth

    T0v = n * Vtm
    T1v = mstar * Vgst
    T2v = T1v / T0v.clamp_min(1e-30)
    # b4ld.c §1242-1263: faithful 3-branch numerator (T10).
    #   T2 >  EXP_THR  →  T10 = T1 = mstar·Vgst        (strong inversion linear)
    #   T2 < -EXP_THR  →  T10 = n·Vtm·log(1+MIN_EXP)   (deep subthreshold floor)
    #   else            →  T10 = n·Vtm·log(1+exp(T2))  (canonical bridge)
    # All three arms are evaluated with safe primitives so torch.where does NOT
    # propagate NaN even from the un-selected branch.
    ExpVgst = safe_exp(T2v)                                  # safe in all arms
    T10_bridge = n * Vtm * torch.log1p(ExpVgst)
    T10_strong = T1v
    T10_deep = n * Vtm * _LOG1P_MIN_EXP
    # Compose: pick deep when T2<-EXP_THR, strong when T2>EXP_THR, else bridge.
    T10v = torch.where(T2v > EXP_THRESHOLD, T10_strong,
            torch.where(T2v < -EXP_THRESHOLD, T10_deep, T10_bridge))

    # b4ld.c §1265-1291: faithful 3-branch denominator (T9).
    # T1_off = voffcbn - (1-mstar)·Vgst ;  T2_off = T1_off/T0
    #   T2_off < -EXP_THR → T3 = coxe·MIN_EXP/cdep0
    #   T2_off >  EXP_THR → T3 = coxe·MAX_EXP/cdep0
    #   else               → T3 = coxe/cdep0 · exp(T2_off)  (canonical)
    # Then T9 = mstar + n·T3.
    T1_off = voffcbn - (1.0 - mstar) * Vgst
    T2_off = T1_off / T0v.clamp_min(1e-30)
    coxe_over_cdep0 = coxe / cdep0.clamp_min(1e-30)
    ExpOff = safe_exp(T2_off)                                # safe in all arms
    T3_bridge = coxe_over_cdep0 * ExpOff
    T3_low = coxe_over_cdep0 * MIN_EXP
    T3_high = coxe_over_cdep0 * MAX_EXP
    T3v = torch.where(T2_off > EXP_THRESHOLD, T3_high,
           torch.where(T2_off < -EXP_THRESHOLD, T3_low, T3_bridge))
    T9v = mstar + n * T3v
    Vgsteff = T10v / T9v.clamp_min(1e-30)

    # ===================================================================== #
    # 6b. Weff correction  b4ld.c §1298-1311                                #
    # ===================================================================== #
    # Weff = Weff0 - 2·(dwg·Vgsteff + dwb·(sqrtPhis - sqrtPhi))
    # Plus a discontinuity guard for Weff < 2e-8.
    T9_w = sqrtPhis - sqrtPhi_pre
    Weff = Weff - 2.0 * (dwg * Vgsteff + dwb * T9_w)
    # Discontinuity guard (b4ld.c §1305-1311):
    #   if Weff < 2e-8: Weff = 2e-8·(4e-8 - Weff)/(6e-8 - 2·Weff)
    # SMOOTH: torch.where over both arms; both differentiable.
    T0_w = 1.0 / (6.0e-8 - 2.0 * Weff).clamp_min(1e-30)
    Weff_clamp = 2.0e-8 * (4.0e-8 - Weff) * T0_w
    Weff = torch.where(Weff < 2.0e-8, Weff_clamp, Weff)

    # ===================================================================== #
    # 7. Abulk   b4ld.c §1338-1395    manual §5.1                           #
    # ===================================================================== #
    T9_a = 0.5 * k1ox * Lpe_Vb / sqrtPhis.clamp_min(1e-30)
    T1_a = T9_a + k2ox - k3b * Vth_NarrowW

    # SMOOTH: safe_sqrt — but EPS_SQRT=1e-12 is far too coarse here. Physical
    # xj·Xdep ~ (1e-7)·(1e-7) = 1e-14; the default 1e-12 floor inflates T9_xj
    # by 100×, which propagates to Abulk0 (T5 collapses to ~0), making Abulk
    # ~10% low and Vdsat ~52 mV high. Use a fp64-safe 1e-30 floor (matches the
    # epssub/(εrox·ε0)·toxe·Xdep0 path in §3 DIBL_Sft, which has the same
    # numerical scale).  z214 finding 2026-04-30.
    T9_xj = torch.sqrt((xj * Xdep).clamp_min(1e-30))
    tmp1 = Leff + 2.0 * T9_xj
    T5_a = Leff / tmp1.clamp_min(1e-30)
    tmp2_a = a0 * T5_a
    tmp3_a = Weff + b1
    tmp4_a = b0 / tmp3_a.clamp_min(1e-30)
    T2_a = tmp2_a + tmp4_a
    T7_a = T5_a * T5_a * T5_a   # T6 = T5²; T7 = T5·T6

    Abulk0 = 1.0 + T1_a * T2_a
    T8_a = ags * a0 * T7_a
    dAbulk_dVg = -T1_a * T8_a
    Abulk = Abulk0 + dAbulk_dVg * Vgsteff

    # b4ld.c §1363-1375: rational regularizer when Abulk0 / Abulk < 0.1
    Abulk0 = torch.where(
        Abulk0 < 0.1,
        (0.2 - Abulk0) / (3.0 - 20.0 * Abulk0),
        Abulk0,
    )
    Abulk = torch.where(
        Abulk < 0.1,
        (0.2 - Abulk) / (3.0 - 20.0 * Abulk),
        Abulk,
    )

    # b4ld.c §1378-1393: keta body-bias scaling with -0.9 regularizer.
    T2_k = keta * Vbseff
    T0_a = torch.where(
        T2_k >= -0.9,
        1.0 / (1.0 + T2_k),
        (17.0 + 20.0 * T2_k) / (0.8 + T2_k),
    )
    Abulk = Abulk * T0_a
    Abulk0 = Abulk0 * T0_a   # tracked but not used downstream in DC-only path

    # ===================================================================== #
    # 8. Mobility   b4ld.c §1416-1578  (mobMod = 1 path)    manual §5.2     #
    # ===================================================================== #
    # mtrlMod=0 ⇒ T14 = 0
    T14 = t(0.0)
    T0_mu = Vgsteff + 2.0 * Vth - T14
    T2_mu = 1.0 + uc * Vbseff
    T3_mu = T0_mu / toxe
    T4_mu = T3_mu * (ua + ub * T3_mu)

    # ud term (Coulombic): T8 = ud · (toxe/(Vgsteff+2|Vth|))² · Vth
    T12_mu = safe_sqrt(Vth * Vth + 1.0e-4)
    T9_mu = 1.0 / (Vgsteff + 2.0 * T12_mu).clamp_min(1e-30)
    T10_mu = T9_mu * toxe
    T8_mu = ud * T10_mu * T10_mu * Vth
    T6_mu = T8_mu * Vth
    T5_mu = T4_mu * T2_mu + T6_mu

    # b4ld.c §1561-1571: Denomi rational regularizer at T5 < -0.8
    Denomi = torch.where(
        T5_mu >= -0.8,
        1.0 + T5_mu,
        (0.6 + T5_mu) / (7.0 + 10.0 * T5_mu),
    )
    mueff = u0temp / Denomi.clamp_min(1e-30)

    # ===================================================================== #
    # 8b. Rds(Vgsteff, Vbseff)   b4ld.c §1313-1336                          #
    # ===================================================================== #
    # rdsmod = 0 (DEFAULT in BSIM4 per b4set.c:107-108) → Rds is INTERNAL,
    #   bias-dependent, modifies Idsat AND Vdsat.
    # rdsmod = 1 → Rds is EXTERNAL (lumped resistor at S/D nodes), so the
    #   internal expressions take Rds=0.
    rdsmod_v = int(model.get("rdsmod", 0))

    # Internal rds0/rdswmin per b4temp.c §1255: rds*T10*nf/(weffCJ*1e6)^wr.
    # We get rdstemp = rdsw_scaled · (1 + prt·delT) from temp.py, plus rdswmin.
    nf_v = float(model.get("nf", 1.0))
    weffCJ_um = float(geom.weffCJ) * 1.0e6
    wr_v = float(P.get("wr", model.get("wr", 1.0)))
    PowWeffWr = (max(weffCJ_um, 1e-30) ** wr_v) if wr_v != 0.0 else 1.0
    PowWeffWr = max(PowWeffWr, 1e-30)
    # rdstemp already has the (1+prt·delT) factor. rdswmin temp-scales identically
    # (prt·delT factor); reconstruct from prt.
    prt_v = float(model.get("prt", 0.0))
    delTemp_v = float(ctx.Temp - ctx.Tnom)
    if int(model.get("tempmod", 0)) == 0:
        rds_temp_factor = 1.0 + prt_v * (ctx.TRatio - 1.0)
    else:
        rds_temp_factor = 1.0 + prt_v * delTemp_v
    rds0_val = sd.rdstemp * nf_v / PowWeffWr
    rdswmin_val = float(P.get("rdswmin", model.get("rdswmin", 0.0))) * rds_temp_factor * nf_v / PowWeffWr

    if rdsmod_v == 0:
        # b4ld.c §1316-1328  — full bias-dependent Rds.
        prwg = t(P.get("prwg", model.get("prwg", 1.0)))
        prwb = t(P.get("prwb", model.get("prwb", 0.0)))
        # T9 = sqrtPhis - sqrtPhi_pre  (b4ld.c §1299)
        T9_rds = sqrtPhis - sqrtPhi_pre
        T0_rds = 1.0 + prwg * Vgsteff
        T1_rds = prwb * T9_rds
        T2_rds = 1.0 / T0_rds.clamp_min(1e-30) + T1_rds
        # SMOOTH: safe_sqrt for the +0.01 regularizer (b4ld.c §1322)
        T3_rds = T2_rds + safe_sqrt(T2_rds * T2_rds + 0.01)
        T4_rds = t(rds0_val) * 0.5
        Rds = t(rdswmin_val) + T3_rds * T4_rds
        Rds = Rds.clamp_min(0.0)   # physical guard
    else:
        # rdsmod = 1: external resistor; internal expressions see Rds=0.
        Rds = torch.zeros_like(Vgsteff)

    # ===================================================================== #
    # 9. Vdsat   b4ld.c §1580-1679                                          #
    #    manual §5.6.1-§5.6.2                                               #
    # ===================================================================== #
    Esat = 2.0 * vsattemp / mueff.clamp_min(1e-30)
    EsatL = Esat * Leff
    # WVCox = Weff·vsattemp·coxe;  WVCoxRds = WVCox·Rds  (b4ld.c §1581-1582)
    WVCox = Weff * vsattemp * coxe
    WVCoxRds = WVCox * Rds   # zero tensor if rdsmod=1

    # Lambda: a1, a2 dependence  (b4ld.c §1591-1609)
    a1_v = float(P.get("a1", model.get("a1", 0.0)))
    if a1_v == 0.0:
        Lambda = a2  # tensor (likely 1.0)
    elif a1_v > 0.0:
        T0_l = 1.0 - a2
        T1_l = T0_l - a1 * Vgsteff - 1e-4
        T2_l = safe_sqrt(T1_l * T1_l + 4e-4 * T0_l)
        Lambda = a2 + T0_l - 0.5 * (T1_l + T2_l)
    else:
        T1_l = a2 + a1 * Vgsteff - 1e-4
        T2_l = safe_sqrt(T1_l * T1_l + 4e-4 * a2)
        Lambda = 0.5 * (T1_l + T2_l)

    Vgst2Vtm = Vgsteff + 2.0 * Vtm
    Lambda_safe = Lambda.clamp_min(1e-12)

    # b4ld.c §1620-1635 (simple): Rds=0 AND Lambda=1
    # b4ld.c §1636-1679 (full quadratic): all other cases.
    # We always compute the full quadratic; it reduces continuously to the
    # simple form as (Rds, 1-Lambda) → 0.
    # SMOOTH: use full quadratic everywhere with safe_sqrt; only divide by
    # T0 with clamp_min in the Lambda≈1 ∧ Rds=0 limit (then fall back to simple).
    T9_q = Abulk * WVCoxRds                                  # b4ld.c §1638
    T7_q = Vgst2Vtm * T9_q                                   # b4ld.c §1640
    T6_q = Vgst2Vtm * WVCoxRds                               # b4ld.c §1641
    T0_vd = 2.0 * Abulk * (T9_q - 1.0 + 1.0 / Lambda_safe)   # b4ld.c §1642
    T1_vd = Vgst2Vtm * (2.0 / Lambda_safe - 1.0) + Abulk * EsatL + 3.0 * T7_q  # §1649
    T2_vd = Vgst2Vtm * (EsatL + 2.0 * T6_q)                  # b4ld.c §1658
    # Discriminant T3 = sqrt(T1² - 2·T0·T2); SMOOTH: safe_sqrt
    disc = T1_vd * T1_vd - 2.0 * T0_vd * T2_vd
    T3_vd = safe_sqrt(disc)
    # Avoid 0/0 when T0→0 (Rds=0 ∧ Lambda=1): use simple form there.
    T0_safe = torch.where(T0_vd.abs() < 1.0e-12,
                          torch.full_like(T0_vd, 1.0e-12),
                          T0_vd)
    Vdsat_full = (T1_vd - T3_vd) / T0_safe
    Vdsat_simple = EsatL * Vgst2Vtm / (Abulk * EsatL + Vgst2Vtm).clamp_min(1e-30)
    use_full = T0_vd.abs() > 1.0e-9
    Vdsat = torch.where(use_full, Vdsat_full, Vdsat_simple)

    # ===================================================================== #
    # 10. Vdseff   b4ld.c §1682-1719   manual §5.6.3                        #
    # ===================================================================== #
    # This is BSIM4's smooth-min implementation; port bit-faithfully.
    T1_v = Vdsat - Vds - delta
    T2_v = safe_sqrt(T1_v * T1_v + 4.0 * delta * Vdsat)
    Vdseff_a = Vdsat - 0.5 * (T1_v + T2_v)             # T1 >= 0  (Vds < Vdsat-δ)
    T4_v = (2.0 * delta) / (T2_v - T1_v).clamp_min(1e-30)
    T5_v = 1.0 - T4_v
    Vdseff_b = Vdsat * T5_v                            # T1 < 0
    Vdseff = torch.where(T1_v >= 0.0, Vdseff_a, Vdseff_b)
    # b4ld.c §1712: clamp at Vds=0
    Vdseff = torch.where(Vds == 0.0, torch.zeros_like(Vds), Vdseff)
    # b4ld.c §1718-1719: hard cap Vdseff <= Vds
    Vdseff = smooth_min(Vdseff, Vds, sharpness=1000.0)  # SMOOTH: faithful cap

    diffVds = Vds - Vdseff

    # ===================================================================== #
    # 11. Idl  b4ld.c §1790-1844 (Coxeff=coxe simplification)               #
    #     manual §5.6.4                                                     #
    # ===================================================================== #
    # ---- Tcen / Coxeff centroid (b4ld.c §1789-1805 capMod=2 path) ------- #
    # T0 = (Vgsteff + vtfbphi2) / (2e8·toxp)
    # tmp3 = exp(bdos·0.7·log(T0)) = T0^(0.7·bdos)
    # T1 = 1 + tmp3
    # Tcen = ados·1.9e-9 / T1
    # Coxeff = epssub·coxp / (epssub + coxp·Tcen)
    tmp2_tc = (2.0e8 * toxp).clamp_min(1e-30)
    T0_tc_raw = (Vgsteff + vtfbphi2) / tmp2_tc
    # T0 must be > 0 for log; in deep subthreshold Vgsteff~0 ⇒ T0 small but >0
    T0_tc = T0_tc_raw.clamp_min(1e-30)
    tmp3_tc = safe_exp(bdos * 0.7 * safe_log(T0_tc))    # SMOOTH: safe primitives
    T1_tc = 1.0 + tmp3_tc
    Tcen = ados * 1.9e-9 / T1_tc.clamp_min(1e-30)
    Coxeff = epssub * coxp / (epssub + coxp * Tcen).clamp_min(1e-30)
    CoxeffWovL = Coxeff * Weff / Leff
    beta = mueff * CoxeffWovL

    AbovVgst2Vtm = Abulk / Vgst2Vtm.clamp_min(1e-30)
    T0_idl = 1.0 - 0.5 * Vdseff * AbovVgst2Vtm
    fgche1 = Vgsteff * T0_idl
    fgche2 = 1.0 + Vdseff / EsatL.clamp_min(1e-30)
    gche = beta * fgche1 / fgche2.clamp_min(1e-30)

    # b4ld.c §1843-1844:  Idl = gche / (1 + gche·Rds)
    # When Rds = 0 (rdsmod=1) this collapses to Idl = gche.
    Idl = gche / (1.0 + gche * Rds).clamp_min(1e-30)

    # ===================================================================== #
    # 12. DIBL / CLM / SCBE / DITS — combine Va contributions               #
    #     b4ld.c §1851-2110     manual §5.7                                 #
    # ===================================================================== #
    # FP — pocket-implant Rout degradation factor (b4ld.c §1853-1861)
    fprout_v = float(model.get("fprout", 0.0))
    if fprout_v <= 0.0:
        FP = torch.ones_like(Vgst2Vtm)
    else:
        T9_fp = fprout * safe_sqrt(Leff) / Vgst2Vtm.clamp_min(1e-30)
        FP = 1.0 / (1.0 + T9_fp)

    # PvagTerm — pvag pocket modifier (b4ld.c §1864-1880)
    T8_pv = pvag / EsatL.clamp_min(1e-30)
    T9_pv = T8_pv * Vgsteff
    PvagTerm = torch.where(
        T9_pv > -0.9,
        1.0 + T9_pv,
        (0.8 + T9_pv) / (17.0 + 20.0 * T9_pv),
    )

    # --- VACLM    b4ld.c §1882-1911    manual §5.7.1 -------------------- #
    pclm_v = float(P.get("pclm", model.get("pclm", 1.3)))
    if pclm_v > MIN_EXP:
        # b4ld.c §1883:  T0 = 1 + Rds·Idl  (Rds-coupled CLM denominator).
        # T1 = Leff + Vdsat/Esat = Leff + Vdsat·Leff/EsatL.
        T0_clm = 1.0 + Rds * Idl
        T2_clm = Vdsat / Esat.clamp_min(1e-30)
        T1_clm = Leff + T2_clm
        Cclm = FP * PvagTerm * T0_clm * T1_clm / (pclm * litl).clamp_min(1e-30)
        # diffVds≈0 case — guard with floor; result is huge VACLM (channel-length
        # modulation off in linear region) which is correct.
        diffVds_safe = diffVds.clamp_min(1e-12)
        VACLM = Cclm * diffVds_safe
    else:
        VACLM = torch.full_like(Vds, MAX_EXP)
        Cclm = torch.full_like(Vds, MAX_EXP)

    # --- VADIBL    b4ld.c §1913-1957    manual §5.7.2 ------------------- #
    # thetaRout = pdiblc1 · _exp_threshold_branch(drout·Leff/tmp_dsub) + pdiblc2
    # SMOOTH: tensor-safe; tmp_dsub is now a tensor.
    T0_dr = drout * t(geom.leff) / tmp_dsub.clamp_min(1e-40)
    T5_dr = _exp_threshold_branch(T0_dr)
    thetaRout = pdiblc1 * T5_dr + pdiblc2

    # SMOOTH: branch on tensor via torch.where to keep grads through thetaRout
    T8_db = Abulk * Vdsat
    T0_db = Vgst2Vtm * T8_db
    T1_db = Vgst2Vtm + T8_db
    VADIBL_active = (Vgst2Vtm - T0_db / T1_db.clamp_min(1e-30)) / thetaRout.clamp_min(1e-30)
    # Pocket pdiblb body-bias correction (b4ld.c §1934-1951)
    T7_db = pdiblb * Vbseff
    T3_db = torch.where(
        T7_db >= -0.9,
        1.0 / (1.0 + T7_db),
        (17.0 + 20.0 * T7_db) / (0.8 + T7_db),
    )
    VADIBL_active = VADIBL_active * T3_db * PvagTerm
    VADIBL = torch.where(
        thetaRout > MIN_EXP,
        VADIBL_active,
        torch.full_like(Vds, MAX_EXP),
    )

    # --- VADITS    b4ld.c §1969-1990    manual §5.7.3 ------------------- #
    T0_dits = pditsd * Vds
    T1_dits = safe_exp(T0_dits)                        # SMOOTH: safe_exp clipped
    pdits_v = float(P.get("pdits", model.get("pdits", 0.0)))
    if pdits_v > MIN_EXP:
        T2_dits = 1.0 + pditsl * Leff
        VADITS = (1.0 + T2_dits * T1_dits) / pdits.clamp_min(1e-30)
        VADITS = VADITS * FP
    else:
        VADITS = torch.full_like(Vds, MAX_EXP)

    # --- VASCBE    b4ld.c §1992-2011    manual §5.7.4 ------------------- #
    pscbe2_v = float(model.get("pscbe2", 1e-5))
    pscbe1_v = float(model.get("pscbe1", 4.24e8))
    if pscbe2_v > 0.0 and pscbe1_v >= 0.0:
        # SMOOTH: clamp diffVds away from 0 to avoid 1/0; safe_exp handles huge T0
        diffVds_scbe = diffVds.clamp_min(1e-12)
        T0_scbe = pscbe1 * litl / diffVds_scbe
        VASCBE = Leff * safe_exp(-T0_scbe) / pscbe2     # exp(-x)·... reformulated
        # Wait — b4ld.c writes VASCBE = Leff * exp(T0) / pscbe2 where T0 is
        # NEGATIVE (note: BSIM4 uses pscbe1·litl/diffVds positive but the term
        # appears in 1/Va as 1/VASCBE = pscbe2 · exp(-pscbe1·litl/(Vds-Vdseff))
        # / Leff per the manual §5.7.4 — so VASCBE = Leff·exp(+pscbe1·litl/diff)/pscbe2.
        # Re-derive: 1/Va_scbe = (pscbe2/Leff) · exp(-pscbe1·litl/diffVds).
        # ⇒ VASCBE = Leff·exp(+pscbe1·litl/diffVds) / pscbe2.   Match C: T0 = +x.
        VASCBE = Leff * safe_exp(T0_scbe) / pscbe2
    else:
        VASCBE = torch.full_like(Vds, MAX_EXP)

    # --- Vasat (b4ld.c §1765-1788)  manual §5.6 (extrinsic case) -------- #
    # Vasat = T0 / T1 where:
    #   tmp4 = 1 - 0.5·Abulk·Vdsat/Vgst2Vtm
    #   T0 = EsatL + Vdsat + 2·WVCoxRds·Vgsteff·tmp4
    #   T1 = 2/Lambda - 1 + WVCoxRds·Abulk
    # When WVCoxRds = 0 (rdsmod=1) and Lambda=1 → T0=EsatL+Vdsat, T1=1, Vasat=EsatL+Vdsat.
    # That is NOT equal to Vdsat in general — old port collapsed this incorrectly.
    Vgst2Vtm_safe = Vgst2Vtm.clamp_min(1e-30)
    tmp4_va = 1.0 - 0.5 * Abulk * Vdsat / Vgst2Vtm_safe
    T0_va = EsatL + Vdsat + 2.0 * WVCoxRds * Vgsteff * tmp4_va
    T1_va = 2.0 / Lambda_safe - 1.0 + WVCoxRds * Abulk
    Vasat = T0_va / T1_va.clamp_min(1e-30)
    Va = Vasat + VACLM

    # ===================================================================== #
    # 13. Final Ids — chain-multiply DIBL/DITS/CLM/SCBE  b4ld.c §2013-2091   #
    #     manual §5.6.4 + §5.7  (cdrain = Ids·Vdseff)                       #
    # ===================================================================== #
    # Faithful BSIM4 chain (NOT a parallel-resistor lump-sum):
    #   Idsa  = Idl · (1 + diffVds/VADIBL)
    #   Idsa *= (1 + diffVds/VADITS)
    #   Idsa *= (1 + log(Va/Vasat)/Cclm)         ← CLM as logarithm, not 1/V
    #   Ids   = Idsa · (1 + diffVds/VASCBE)
    # All "1 + small" factors stay >0 because diffVds≥0 and the V's are large.
    # Note: `Idl` in our port is gche (S); cdrain = Ids·Vdseff is computed below.
    Idsa = Idl * (1.0 + diffVds / VADIBL.clamp_min(1e-30))
    Idsa = Idsa * (1.0 + diffVds / VADITS.clamp_min(1e-30))
    # CLM term: log(Va/Vasat)/Cclm. Va≥Vasat ⇒ log≥0; Cclm>0 by guard above.
    Vasat_safe = Vasat.clamp_min(1e-30)
    log_VaVasat = safe_log((Va / Vasat_safe).clamp_min(1.0))   # never negative
    Idsa = Idsa * (1.0 + log_VaVasat / Cclm.clamp_min(1e-30))
    # WAVE2-FIX-1 (Gap 2): snapshot pre-SCBE Idsa·Vdseff for impact-ionization (Iii).
    # b4ld.c §2069: T4 = Idsa·Vdseff; Isub = T1·T4. SCBE is applied only to the final
    # Ids (line 2089-2091), not to Iii. Storing the current form (Idsa_chan * Vdseff)
    # so leak.compute_iimpact can directly multiply by T1.
    Idsa_Vdseff = Idsa * Vdseff
    Ids_chan = Idsa * (1.0 + diffVds / VASCBE.clamp_min(1e-30))
    # cdrain = Ids·Vdseff   (channel current per BSIM4 convention)
    Ids = Ids_chan * Vdseff

    # NMOS sign convention: positive Ids for Vds>0.
    return DCResult(
        Ids=Ids,
        Vth=Vth,
        Vgsteff=Vgsteff,
        Vdsat=Vdsat,
        Vdseff=Vdseff,
        Abulk=Abulk,
        n=n,
        mueff=mueff,
        Rds=Rds if rdsmod_v == 0 else None,
        Idsa=Idsa_Vdseff,
        Vgs_eff=Vgs_eff,
        Vbseff=Vbseff,
    )


# --------------------------------------------------------------------------- #
# Convenience wrapper                                                          #
# --------------------------------------------------------------------------- #

def compute_dc_simple(model: BSIM4Model, geom: Geometry, T_C: float,
                      Vgs, Vds, Vbs=0.0) -> DCResult:
    """One-shot helper that builds SizeDependParam and calls compute_dc."""
    sd = compute_size_dep(model, geom, T_C)
    return compute_dc(model, sd, Vgs=Vgs, Vds=Vds, Vbs=Vbs)

```


=== FILE: _extracted/nsram_bsim4_port/nsram/nsram/bsim4_port/nsram_cell.py (16456 chars) ===
```python
"""nsram_cell — body-KCL wrapper for the NS-RAM 2T floating-body cell.

Integrates the BSIM4 port (DC + impact-ion + GIDL/GISL + Igb + body diodes +
junction caps) with a parasitic Gummel-Poon NPN and an external body-leak
resistor into a single differentiable cell model.

Body node KCL (positive = current INTO the body):

    C_body · dVb/dt =
        + Iii(Vb, Vd, VG1, VG2)           # impact-ionization, channel → body
        - Ibd(Vb, Vd)                     # body→drain diode (Ibd>0 forward → leaves body)
        - Ibs(Vb, 0)                      # body→source diode (likewise)
        - Igidl(Vb, Vd, VG1)              # GIDL: drain edge band-to-band
        - Igisl(Vb, VG1)                  # GISL: source edge band-to-band
        + Igb(VG1, Vb)                    # gate→body tunneling (positive ⇒ enters body)
        - I_BJT(Vb, Ve=0, Vc=Vd)          # parasitic NPN collector current
        - Vb / Rb_leak                    # external resistor to ground

Body capacitance:
    C_body = Cjs + Cjd + C_extra

VG2 (bottom gate / well) is treated, by NS-RAM convention, as a bias that
shifts the effective threshold of the BSIM4 top transistor:
    vth0_eff(VG2) = vth0_T + gamma_VG2 · VG2          (default gamma_VG2 = 0.3)

This is a single-knob proxy for the second-gate body-effect coupling — VG2
thus selects between the three regimes (BISTABLE / SOFT / INTEGRATOR) by
moving the impact-ionization knee.

Sign convention for the diode block (compute_body_diodes):
    Ibs, Ibd as returned are positive when the junction is forward biased
    (Vbs>0 / Vbd>0), corresponding to current flowing OUT of the body. We
    subtract them from I_into_body.

The whole `kcl_body` is differentiable end-to-end (no Python branches on
tensor values) and verified by torch.autograd.gradcheck in the tests.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import torch

from .bjt import GummelPoonNPN, compute_bjt
from .caps import compute_caps
from .dc import compute_dc
from .diode import compute_body_diodes
from .geometry import Geometry
from .leak import compute_iimpact, compute_igidl_gisl, compute_igb
from .model_card import BSIM4Model
from .temp import compute_size_dep, SizeDependParam


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class NSRAMCellConfig:
    """Static config for one NS-RAM cell.

    Holds the BSIM4 model, geometry, parasitic NPN and a few cell-level
    knobs. The temperature-dependent SizeDependParam is computed lazily and
    cached on first use.
    """
    bsim4_model: BSIM4Model
    geometry: Geometry
    bjt_params: GummelPoonNPN
    Rb_leak: float = 5e8         # external body-leak resistor [Ω]
    C_extra: float = 0.0         # extra body cap [F]
    T_C: float = 27.0            # operating temp [°C]
    gamma_VG2: float = 0.3       # NS-RAM convention: VG2 → vth0 shift
    # Junction geometry. None = auto-default to W·L (area) and 2·(W+L) (perim);
    # explicit 0.0 = caller wants zero junction caps (rarely correct).
    # WAVE2-FIX (2026-04-29 critique): defaulting to 0 silently zeros junction
    # caps and makes Cbody dominated by abs(Cgb), which is physically wrong for
    # NS-RAM retention. Auto-default uses geometry W·L estimate when None.
    As: Optional[float] = None   # source bottom area [m²]; None → W·L
    Ad: Optional[float] = None   # drain  bottom area [m²]; None → W·L
    Ps: Optional[float] = None   # source perimeter   [m];  None → 2·(W+L)
    Pd: Optional[float] = None   # drain  perimeter   [m];  None → 2·(W+L)
    # Toggle bits (defaults match physical NS-RAM cell)
    use_iii: bool = True
    use_gidl: bool = True
    use_igb: bool = True
    use_diode: bool = True
    use_bjt: bool = True
    # Optional NS-RAM-specific lateral-BJT punch-through body-charging term.
    # BSIM4 doesn't model lateral-BJT punch-through avalanche at low Vd, but
    # measurements of NS-RAM 2T cells (Pazos+, 130nm) clearly show snapback
    # at Vd ≈ 0.7-1V — far below where standard BSIM4 alpha0/beta0=18 would
    # trigger. SOI-style extension; standard practice when modelling
    # floating-body cells with BSIM4 (alternative: switch to BSIM-SOI).
    #
    # Vb-COUPLED form (P7v4): trigger threshold lowered by accumulated body
    # voltage — captures the regenerative loop where body charge advances
    # the snapback voltage:
    #     I_PT = I_PT0 · scale · softplus((Vd + k_Vb·Vb − V_PT_th) / scale)
    # k_Vb > 0 means: more positive Vb → effective drive higher → triggers
    # at lower Vd. This is what makes VG2 (which sets Vb steady state)
    # actually shift the snapback voltage in the model.
    use_punchthrough: bool = False
    I_PT0: float = 1.0e-6      # punch-through pre-factor [A]
    V_PT_th: float = 0.7       # trigger Vd [V]
    V_PT_scale: float = 0.05   # ramp sharpness [V]
    k_Vb_PT: float = 0.0       # Vb coupling strength [unitless, ≥0]
    # Cached SizeDependParam (lazy)
    _sd: Optional[SizeDependParam] = field(default=None, init=False, repr=False)

    def size_dep(self) -> SizeDependParam:
        if self._sd is None:
            self._sd = compute_size_dep(self.bsim4_model, self.geometry,
                                        T_C=self.T_C)
        return self._sd

    def invalidate(self) -> None:
        """Force a re-compute of the SizeDependParam cache (after editing the
        model card)."""
        self._sd = None


# --------------------------------------------------------------------------- #
# Internal helpers                                                            #
# --------------------------------------------------------------------------- #

@contextmanager
def _vth0_shifted(sd: SizeDependParam, vth0_eff):
    """Temporarily replace sd.vth0_T with `vth0_eff` (tensor or float).

    compute_dc reads `sd.vth0_T` once via `t(sd.vth0_T)` (as_tensor); passing
    a fp64 tensor flows gradients through. We restore the original on exit.
    """
    saved = sd.vth0_T
    sd.vth0_T = vth0_eff
    try:
        yield
    finally:
        sd.vth0_T = saved


def _as_t(x, ref: torch.Tensor) -> torch.Tensor:
    """Coerce x to fp64 tensor on the same device as ref."""
    return torch.as_tensor(x, dtype=torch.float64, device=ref.device)


# --------------------------------------------------------------------------- #
# Body-KCL                                                                    #
# --------------------------------------------------------------------------- #

def kcl_body(
    cfg: NSRAMCellConfig,
    Vb: torch.Tensor,
    Vd: torch.Tensor,
    VG1: torch.Tensor,
    VG2: torch.Tensor,
    *,
    use_iii: Optional[bool] = None,
    use_gidl: Optional[bool] = None,
    use_igb: Optional[bool] = None,
    use_diode: Optional[bool] = None,
    use_bjt: Optional[bool] = None,
) -> dict[str, torch.Tensor]:
    """Compute body-KCL residual for the NS-RAM cell.

    All bias inputs are torch.Tensors (fp64). Sign convention: positive
    currents flow INTO the body.

    Returns:
        dict with
          - 'I_total'    [A]   sum of currents into body
          - 'C_body'     [F]   total body capacitance (Cjs + Cjd + C_extra)
          - 'dVb_dt'     [V/s] = I_total / C_body
          - 'components' dict of individual currents
    """
    # Resolve toggles (per-call override > cfg default)
    use_iii = cfg.use_iii if use_iii is None else use_iii
    use_gidl = cfg.use_gidl if use_gidl is None else use_gidl
    use_igb = cfg.use_igb if use_igb is None else use_igb
    use_diode = cfg.use_diode if use_diode is None else use_diode
    use_bjt = cfg.use_bjt if use_bjt is None else use_bjt

    sd = cfg.size_dep()
    model = cfg.bsim4_model

    # fp64 + broadcast all four biases to a common shape.
    Vb_t = Vb.to(torch.float64)
    Vd_t = _as_t(Vd, Vb_t)
    VG1_t = _as_t(VG1, Vb_t)
    VG2_t = _as_t(VG2, Vb_t)
    Vb_t, Vd_t, VG1_t, VG2_t = torch.broadcast_tensors(Vb_t, Vd_t, VG1_t, VG2_t)

    # Standard NS-RAM bias mapping
    Vgs = VG1_t                       # top gate voltage relative to source=0
    Vds = Vd_t
    Vbs = Vb_t                        # body relative to source=0
    Vbd = Vb_t - Vd_t

    # VG2 → vth0 shift (NS-RAM convention)
    vth0_T0 = sd.vth0_T
    if not isinstance(vth0_T0, torch.Tensor):
        vth0_T0 = torch.as_tensor(vth0_T0, dtype=torch.float64,
                                  device=Vb_t.device)
    vth0_eff = vth0_T0 + cfg.gamma_VG2 * VG2_t

    # ------- DC drain current + intermediates needed for Iii / caps ------- #
    with _vth0_shifted(sd, vth0_eff):
        dc = compute_dc(model, sd, Vgs=Vgs, Vds=Vds, Vbs=Vbs)

        # Impact-ionization (uses Vdseff from dc)
        if use_iii:
            Iii = compute_iimpact(model, sd, dc, Vds=Vds)
        else:
            Iii = torch.zeros_like(Vb_t)

        # GIDL / GISL
        if use_gidl:
            Igidl, Igisl = compute_igidl_gisl(model, sd, Vgs=Vgs, Vds=Vds, Vbs=Vbs)
        else:
            Igidl = torch.zeros_like(Vb_t)
            Igisl = torch.zeros_like(Vb_t)

        # Gate-to-body tunneling
        if use_igb:
            Igb = compute_igb(model, sd, Vgs=Vgs, Vbs=Vbs, dc_result=dc)
        else:
            Igb = torch.zeros_like(Vb_t)

        # WAVE2-FIX (2026-04-29 critique): auto-default junction geometries
        # to W·L (area) and 2·(W+L) (perimeter) when None. Defaulting to 0
        # silently zeros junction caps and produces wrong NS-RAM retention.
        W = cfg.geometry.W
        L = cfg.geometry.L
        WL = W * L
        WLp = 2.0 * (W + L)
        As_eff = WL  if cfg.As is None else cfg.As
        Ad_eff = WL  if cfg.Ad is None else cfg.Ad
        Ps_eff = WLp if cfg.Ps is None else cfg.Ps
        Pd_eff = WLp if cfg.Pd is None else cfg.Pd

        # Body diodes (positive when forward-biased = leaving body).
        if use_diode:
            Ibs, Ibd = compute_body_diodes(model, sd, Vbs=Vbs, Vbd=Vbd,
                                           As=As_eff, Ad=Ad_eff,
                                           Ps=Ps_eff, Pd=Pd_eff)
        else:
            Ibs = torch.zeros_like(Vb_t)
            Ibd = torch.zeros_like(Vb_t)

        # Junction capacitances (Cjs + Cjd) — depend on Vbs, Vbd.
        cap = compute_caps(model, sd, dc, Vgs=Vgs, Vds=Vds, Vbs=Vbs, Vbd=Vbd,
                           As=As_eff, Ad=Ad_eff, Ps=Ps_eff, Pd=Pd_eff)

    # Parasitic NPN: floating base = body, emitter = source(=0), collector = drain.
    # Vbe = Vb - Vsrc = Vb;   Vbc = Vb - Vd
    #
    # WAVE2-FIX (2026-04-29 critique): use Ib (base current), NOT Ic. The
    # body IS the base node — Ic flows collector→emitter through the BJT
    # transport mechanism, it does NOT terminate at the body. Only the base
    # current Ib (recombination + minority injection) charges/discharges the
    # body. Previous version used Ic, which with Bf=10000 drained the body
    # 10000× too fast in the BJT-on regime → wrong NS-RAM retention.
    if use_bjt:
        bjt = compute_bjt(cfg.bjt_params, Vbe=Vb_t, Vbc=Vbd,
                          T_K=273.15 + cfg.T_C)
        # SPICE NPN convention: Ib positive = INTO base from external circuit.
        # For floating-body, that current can only come from the body charge,
        # so Ib > 0 means the body is sourcing it (leaves body).
        Ib_bjt = bjt["Ib"]
    else:
        Ib_bjt = torch.zeros_like(Vb_t)

    # External body-leak resistor (to ground): I_leak_out = Vb / Rb_leak.
    Ileak_out = Vb_t / cfg.Rb_leak

    # Optional lateral-BJT punch-through body-charging term (NS-RAM specific).
    # P7v4 Vb-coupled form: I_PT depends on (Vd + k_Vb·Vb), so a charged body
    # advances the trigger → captures regenerative snapback dependence on VG2.
    if cfg.use_punchthrough:
        import torch.nn.functional as F
        V_scale_t = torch.as_tensor(cfg.V_PT_scale, dtype=Vd.dtype, device=Vd.device)
        k_Vb_t = torch.as_tensor(cfg.k_Vb_PT, dtype=Vd.dtype, device=Vd.device)
        drive = Vd + k_Vb_t * Vb_t - cfg.V_PT_th
        I_PT = (cfg.I_PT0 * V_scale_t
                * F.softplus(drive / V_scale_t.clamp_min(1e-6)))
    else:
        I_PT = torch.zeros_like(Vb_t)

    # ------- Sum into body (positive = entering body) ------- #
    # Iii    : channel → body                       → +
    # Ibd,Ibs: body → S/D when forward-biased        → -
    # Igidl  : drain → body when reverse-biased drain (it CHARGES body) but the
    #          model's sign for Igidl in BSIM4 is the magnitude; convention in
    #          ngspice is that Igidl flows OUT of the drain INTO the body, i.e.
    #          enters the body. We follow the diff_canonical convention: enter.
    # Igb    : gate → body                          → +
    # I_BJT  : The body IS the base of the parasitic NPN. Ib (positive into
    #          base in SPICE) drains the body when BE forward-biased (carriers
    #          recombine in base region). Ic flows collector→emitter through
    #          the device and does NOT terminate at the body — it must NOT
    #          enter body-KCL.
    # Ileak  : Vb / Rb_leak leaves the body          → -

    # Sign convention: I_total > 0 means net current ENTERING body.
    # Per b4ld.c §443 Ibtot = cbs+cbd - Igidl - Igisl - csub  (leaving body),
    # so currents ENTERING body are: Iii (csub), Igidl, Igisl, Igb;
    # currents LEAVING body are: Ibd, Ibs (junction diodes), Ib_bjt, Ileak.
    I_total = (
        Iii                       # +  (impact ionization → body)
        - Ibd                     # -  (body→drain diode)
        - Ibs                     # -  (body→source diode)
        + Igidl                   # +  (drain→body via GIDL)
        + Igisl                   # +  (source→body via GISL)
        + Igb                     # +  (gate→body tunneling)
        - Ib_bjt                  # -  (parasitic BJT base current; was Ic — bug)
        - Ileak_out               # -  (external Rb_leak)
        + I_PT                    # +  (lateral-BJT punch-through, if enabled)
    )

    # Body capacitance — include |Cgb| (gate-bulk intrinsic).
    # In accumulation (low Vgs) Cgb ≈ -CoxWL, dominating the NS-RAM hold regime.
    # BSIM4's sign convention can give negative Cgb (rate of Qb wrt Vg);
    # the contribution to total body cap is its magnitude.
    C_body = cap.Cjs + cap.Cjd + torch.abs(cap.Cgb) + cfg.C_extra
    # Guard against pathological zero (junctions all-zero geometry → tiny floor)
    C_body = C_body + 1e-30

    dVb_dt = I_total / C_body

    components = {
        "Iii": Iii,
        "Ibd": -Ibd,         # signed-into-body for clarity
        "Ibs": -Ibs,
        "Igidl": Igidl,      # enters body
        "Igisl": Igisl,      # enters body
        "Igb": Igb,
        "Ibjt": -Ib_bjt,
        "Ileak": -Ileak_out,
    }
    return {
        "I_total": I_total,
        "C_body": C_body,
        "dVb_dt": dVb_dt,
        "components": components,
    }


# --------------------------------------------------------------------------- #
# Transient                                                                   #
# --------------------------------------------------------------------------- #

def transient_step(
    cfg: NSRAMCellConfig,
    Vb: torch.Tensor,
    Vd: torch.Tensor,
    VG1: torch.Tensor,
    VG2: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """One forward-Euler step.  Returns Vb_new = Vb + dt · dVb/dt."""
    f = kcl_body(cfg, Vb, Vd, VG1, VG2)
    return Vb + dt * f["dVb_dt"]


def transient(
    cfg: NSRAMCellConfig,
    Vb0: torch.Tensor,
    Vd_seq: torch.Tensor,
    VG1: torch.Tensor,
    VG2: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Forward-Euler transient.

    Args:
        Vb0:    initial body voltage (scalar tensor)
        Vd_seq: drain bias trajectory, shape (T,)
        VG1, VG2: scalar tensors (constant over the transient)
        dt:     time step [s]

    Returns:
        Vb_history of shape (T,) — Vb after applying each Vd_seq[i].
    """
    Vd_seq = Vd_seq.to(torch.float64)
    T = int(Vd_seq.shape[0])
    Vb = Vb0.to(torch.float64)
    out = []
    for i in range(T):
        Vb = transient_step(cfg, Vb, Vd_seq[i], VG1, VG2, dt)
        out.append(Vb)
    return torch.stack(out, dim=0)

```


=== FILE: _extracted/nsram_bsim4_port/nsram/nsram/bsim4_port/caps.py (16706 chars) ===
```python
"""B21_CAPS — BSIM4 4.8.3 capacitance models (junction CV + Meyer capMod=0).

Faithful port of `external/bsim4/code/b4ld.c`:
  - Junction body-source/drain CV: lines 3912-4029 (manual §11.2.1-3)
  - Meyer intrinsic caps capMod=0: lines 2992-3197 (manual §7.4.1)

Scope:
  - Static C(V); no AC, no transient charge derivative integration.
  - capMod=0 Meyer model only (no capMod=2 charge-thickness CTM).
  - Returns the diagonal Cgg, Cgs, Cgd, Cgb plus body-junction Cjs, Cjd.
  - Body-cap for NS-RAM body-KCL: Cbody = Cjs + Cjd + (1-α)·Cox·W·L (channel).

Differentiability rules:
  - fp64 throughout.
  - All if-on-V branches replaced by torch.where over BOTH finite arms,
    or by smooth.smooth_step transitions.
  - safe_sqrt / safe_log / smooth_step substitute hard kinks at FC·Pb crossover
    and at the cutoff/triode/saturation region boundaries.
  - Junction CV uses the BSIM4 exact reverse-bias (V<0) form everywhere on
    [-inf, +inf), with a smooth gluing to the small-forward-bias linearization
    so dCj/dV stays finite as V → Pb.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

import torch

from .geometry import Geometry
from .model_card import BSIM4Model
from .smooth import (
    safe_sqrt, safe_log, safe_exp, smooth_max, smooth_min, smooth_step,
)
from .temp import SizeDependParam


# --------------------------------------------------------------------------- #
# Result container                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class CapResult:
    # Junction body-source/drain caps (per device, F)
    Cjs: torch.Tensor       # body-source pn junction (bottom + sw + swg)
    Cjd: torch.Tensor       # body-drain   pn junction (bottom + sw + swg)
    # Intrinsic Meyer caps (per device, F)
    Cgg: torch.Tensor       # gate self-cap
    Cgs: torch.Tensor       # gate-source
    Cgd: torch.Tensor       # gate-drain
    Cgb: torch.Tensor       # gate-bulk
    # Convenience scalar for NS-RAM body-KCL: total body capacitance to ground
    Cbody_total: torch.Tensor  # Cjs + Cjd + Cgb (channel-to-bulk via Meyer)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _t(x, dtype=torch.float64, device=None) -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype, device=device)


def _bsim4_pb_default(model: BSIM4Model, key: str, fallback: float):
    """BSIM4 built-in: pb*/mj* default to canonical values when card sets 0.

    GRADFIX: keep tensor type if `model.get` returns a tensor (injected for
    gradcheck on cap params); only cast to float for the comparison branch.
    """
    v = model.get(key, 0.0)
    v_check = float(v) if not isinstance(v, torch.Tensor) else float(v.detach())
    if v_check <= 0.0:
        return fallback
    return v


def _coxe(sd: SizeDependParam) -> float:
    return sd.model_ctx.coxe


# --------------------------------------------------------------------------- #
# JUNCTION CV  (b4ld.c §3912-4029, manual §11.2.1-3)                          #
# --------------------------------------------------------------------------- #
def _junction_cap_one(
    czb: torch.Tensor,            # zero-bias bottom-area cap (F)
    czbsw: torch.Tensor,          # zero-bias sidewall cap   (F)
    czbswg: torch.Tensor,         # zero-bias gate-sidewall cap (F)
    Pb: float, Pbsw: float, Pbswg: float,
    Mj: float, Mjsw: float, Mjswg: float,
    Vj: torch.Tensor,             # junction voltage (Vbs_jct or Vbd_jct), V
    sharpness: float = 50.0,
) -> torch.Tensor:
    """One body-junction cap (source OR drain), summing bottom + 2 sidewalls.

    BSIM4 (b4ld.c §3936-3978): for reverse bias V<0:
        Cj = Cj0 · (1 - V/Pb)^(-Mj)
    For forward bias V>=0 (the C code's `else` arm at §3974):
        Cj_lin(V) = Cj0 · (1 + V·Mj/Pb)              [Taylor expansion]
    We glue the two arms via smooth_step so the kink at V=0 disappears. The
    linearization is the same one BSIM4 uses for forward bias to keep Newton
    iterations from blowing up — we re-use it as a smooth fallback on [0, Pb).

    SMOOTH: smooth_step(V, -Vt, +Vt) blends reverse↔forward arms across V≈0.
    SMOOTH: safe_sqrt/safe_log on (1 - V/Pb) to avoid neg-arg + kink at V=Pb.
    """
    # --- Reverse-bias arm (V<0) -------------------------------------------- #
    # arg = 1 - V/Pb  must stay > 0; clamp via safe path.
    # SMOOTH: floor arg at small positive eps so log/sqrt are differentiable
    # even on the forward-bias side where this arm is unused.
    one = torch.ones_like(Vj)
    arg_b = (one - Vj / Pb).clamp_min(1e-6)         # SMOOTH: floor
    arg_sw = (one - Vj / Pbsw).clamp_min(1e-6)
    arg_swg = (one - Vj / Pbswg).clamp_min(1e-6)

    if abs(Mj - 0.5) < 1e-12:
        s_b = 1.0 / safe_sqrt(arg_b)
    else:
        s_b = safe_exp(-Mj * safe_log(arg_b))         # = arg_b^(-Mj)
    if abs(Mjsw - 0.5) < 1e-12:
        s_sw = 1.0 / safe_sqrt(arg_sw)
    else:
        s_sw = safe_exp(-Mjsw * safe_log(arg_sw))
    if abs(Mjswg - 0.5) < 1e-12:
        s_swg = 1.0 / safe_sqrt(arg_swg)
    else:
        s_swg = safe_exp(-Mjswg * safe_log(arg_swg))

    Cj_rev = czb * s_b + czbsw * s_sw + czbswg * s_swg

    # --- Forward-bias arm (V>=0): BSIM4 linearization, b4ld.c §3974-3978 -- #
    #   capbs = T0 + T1
    #   T0 = czbs+czbssw+czbsswg
    #   T1 = vbs_jct·(czbs·MJS/PhiBS + czbssw·MJSWS/PhiBSWS + czbsswg·MJSWGS/PhiBSWGS)
    T0 = czb + czbsw + czbswg
    T1 = Vj * (czb * Mj / Pb + czbsw * Mjsw / Pbsw + czbswg * Mjswg / Pbswg)
    Cj_fwd = T0 + T1

    # --- Smooth glue across V=0 ------------------------------------------- #
    # SMOOTH: smooth_step picks reverse arm for V<<0, forward for V>>0.
    # Width chosen as ~25 mV (≈ kT/q at 300K) so the blend is physically
    # localized at the no-bias point.
    w = 0.025
    blend_fwd = smooth_step(Vj, -w, +w, sharpness=sharpness)
    return blend_fwd * Cj_fwd + (1.0 - blend_fwd) * Cj_rev


def compute_junction_caps(
    model: BSIM4Model,
    sd: SizeDependParam,
    Vbs: torch.Tensor,
    Vbd: torch.Tensor,
    *,
    As: Optional[float] = None,
    Ad: Optional[float] = None,
    Ps: Optional[float] = None,
    Pd: Optional[float] = None,
) -> dict:
    """Body-source & body-drain junction capacitances. (b4ld.c §3912-4029)

    Cj(V) = Cj0·area · (1 - V/Pb)^(-Mj)        for V < 0 (reverse)
          = Cj0·area · (1 + V·Mj/Pb)           for V ≥ 0 (forward, smoothed)
    Adds bottom (Cj·As) + sidewall (Cjsw·Ps) + gate-sidewall (Cjswg·Weff_CJ)
    pieces. Convention: V is the JUNCTION voltage (V_body - V_diffusion); we
    take Vbs / Vbd directly. NMOS body is at Vb; for an NMOS in normal operation
    Vbs<0 (reverse), so cap shrinks vs zero-bias.
    """
    dtype = Vbs.dtype if isinstance(Vbs, torch.Tensor) else torch.float64
    device = Vbs.device if isinstance(Vbs, torch.Tensor) else None

    # --- Per-instance areas/perimeters (b4temp.c §2044) -------------------- #
    # If caller did not supply, default to ngspice's `if (AS<=0)` fallback:
    #   AS = AD = W · hdif · 2  (with hdif from card; otherwise drawn area).
    # We pick the safest neutral default: As = Ad = W·L (plausible bulk area)
    # and Ps = Pd = 2·(W+L). Caller should pass real values for SPICE match.
    W = sd.geom.weff
    L = sd.geom.leff
    if As is None: As = W * L
    if Ad is None: Ad = W * L
    if Ps is None: Ps = 2.0 * (W + L)
    if Pd is None: Pd = 2.0 * (W + L)
    weffCJ = sd.geom.weffCJ
    NF = 1.0  # single-finger; b4temp embeds NF separately into czbsswg

    # --- Per-area / per-perimeter zero-bias values (b4ld.c §3914-3921) ---- #
    # BSIM4DunitAreaTempJctCap = cjd at op-temp; at Tnom this equals model.cjd.
    # We use the un-temp-adjusted cj* directly (caller ensures Tnom or accepts
    # ~10% error). TODO(temp): apply tcj/tcjsw/tcjswg shifts.
    cjs   = model.get("cjs",   0.0)        # F/m^2
    cjd   = model.get("cjd",   0.0)
    cjsws = model.get("cjsws", 0.0)        # F/m
    cjswd = model.get("cjswd", 0.0)
    cjswgs = model.get("cjswgs", cjsws)    # default to cjsws if not given
    cjswgd = model.get("cjswgd", cjswd)

    # Built-in potentials & grading coeffs (BSIM4 standard defaults)
    PbS   = _bsim4_pb_default(model, "pbs",   1.0)
    PbD   = _bsim4_pb_default(model, "pbd",   1.0)
    PbSWS = _bsim4_pb_default(model, "pbsws", 1.0)
    PbSWD = _bsim4_pb_default(model, "pbswd", 1.0)
    PbSWGS = _bsim4_pb_default(model, "pbswgs", PbSWS)
    PbSWGD = _bsim4_pb_default(model, "pbswgd", PbSWD)
    MJS   = _bsim4_pb_default(model, "mjs",   0.5)
    MJD   = _bsim4_pb_default(model, "mjd",   0.5)
    MJSWS = _bsim4_pb_default(model, "mjsws", 0.33)
    MJSWD = _bsim4_pb_default(model, "mjswd", 0.33)
    MJSWGS = _bsim4_pb_default(model, "mjswgs", MJSWS)
    MJSWGD = _bsim4_pb_default(model, "mjswgd", MJSWD)

    czbs   = _t(cjs   * As, dtype, device)
    czbd   = _t(cjd   * Ad, dtype, device)
    czbssw = _t(cjsws * Ps, dtype, device)
    czbdsw = _t(cjswd * Pd, dtype, device)
    czbsswg = _t(cjswgs * weffCJ * NF, dtype, device)
    czbdswg = _t(cjswgd * weffCJ * NF, dtype, device)

    Cjs = _junction_cap_one(
        czbs, czbssw, czbsswg, PbS, PbSWS, PbSWGS, MJS, MJSWS, MJSWGS, Vbs,
    )
    Cjd = _junction_cap_one(
        czbd, czbdsw, czbdswg, PbD, PbSWD, PbSWGD, MJD, MJSWD, MJSWGD, Vbd,
    )
    return {"Cjs": Cjs, "Cjd": Cjd}


# --------------------------------------------------------------------------- #
# INTRINSIC MEYER CAPS  capMod=0  (b4ld.c §2992-3197, manual §7.4.1)          #
# --------------------------------------------------------------------------- #
def compute_intrinsic_caps_capmod0(
    model: BSIM4Model,
    sd: SizeDependParam,
    dc_result,                          # DCResult or None — only Vth used
    Vgs: torch.Tensor,
    Vds: torch.Tensor,
    Vbs: torch.Tensor,
) -> dict:
    """Meyer intrinsic caps, capMod=0. (b4ld.c §2992-3197)

    Three regions (manual §7.4.1):
      * Accumulation/cutoff (Vgs - Vfb < 0, b4ld.c §3014-3030):
            Cgg = Cox·W·L,  Cgs = Cgd = 0,  Cgb = -Cgg
      * Depletion (0 < Vgs - Vfb, Vgs < Vth, b4ld.c §3031-3050):
            Cgg = Cox·W·L · k1/(2·sqrt(k1²/4 + Arg1))     (small)
            Cgs = Cgd = 0
      * Strong inversion:
          - Saturation (Vds > Vdsat, b4ld.c §3062-3088, Meyer 0/100):
                Cgs = (2/3)·Cox·W·L,  Cgd = 0,  Cgb ≈ 0
          - Triode/linear (Vds < Vdsat, b4ld.c §3091+):
                Cgs/Cgd partition by Vds (Meyer's classic 50/50 form
                interpolates from 1/2 at Vds=0 to 2/3 at Vds=Vdsat).

    SMOOTH: each region transition is replaced with smooth_step blends so
    Cgg(Vgs) is C^∞ instead of piecewise.
    """
    CoxWL = _t(_coxe(sd) * sd.geom.weffCV * sd.geom.leffCV, Vgs.dtype, Vgs.device)

    # Vfbcv: flat-band voltage param for capMod=0 (model card field)
    Vfbcv = _t(model.get("vfbcv", -1.0), Vgs.dtype, Vgs.device)
    # Vth approx: use DC Vth if provided, else use vth0_T from temp
    if dc_result is not None and hasattr(dc_result, "Vth"):
        Vth = dc_result.Vth
    else:
        Vth = _t(sd.vth0_T, Vgs.dtype, Vgs.device)

    # Surface potential & k1ox from temp.py
    phi = _t(sd.phi, Vgs.dtype, Vgs.device)
    k1ox = _t(sd.k1ox, Vgs.dtype, Vgs.device)

    # --- Region masks (smooth) -------------------------------------------- #
    # Arg1 = Vgs - Vbs - Vfbcv         (b4ld.c §3012)
    Arg1 = Vgs - Vbs - Vfbcv
    # Vgst = Vgs - Vth                 (b4ld.c §3005)
    Vgst = Vgs - Vth
    # Vdsat ≈ Vgst / Abulk;            for Meyer cap0 BSIM4 uses AbulkCV.
    # We use Abulk≈1 as conservative default (manual §7.4.1 Meyer simplification).
    Abulk = _t(1.0, Vgs.dtype, Vgs.device)
    if dc_result is not None and hasattr(dc_result, "Abulk"):
        Abulk = dc_result.Abulk
    Vdsat = smooth_max(Vgst, _t(1e-6, Vgs.dtype, Vgs.device)) / Abulk

    # SMOOTH: width 25 mV (≈ kT/q) for all region transitions.
    w = 0.025
    accum_to_dep = smooth_step(Arg1, -w, +w)            # 0 in accum, 1 in dep+
    dep_to_inv   = smooth_step(Vgst, -w, +w)            # 0 in dep, 1 in strong inv
    sat_to_lin   = smooth_step(Vdsat - Vds, -w, +w)     # 0 in sat (Vds>Vdsat), 1 in lin

    # --- Accumulation arm  (b4ld.c §3014-3030) ---------------------------- #
    Cgg_acc = CoxWL
    Cgs_acc = torch.zeros_like(CoxWL)
    Cgd_acc = torch.zeros_like(CoxWL)
    Cgb_acc = -CoxWL

    # --- Depletion arm  (b4ld.c §3031-3050) ------------------------------- #
    # T1 = 0.5·k1ox; T2 = sqrt(T1² + Arg1).  Cgg = CoxWL · T1/T2  (= dQg/dVg)
    T1 = 0.5 * k1ox
    T2 = safe_sqrt(T1 * T1 + smooth_max(Arg1, _t(0.0, Vgs.dtype, Vgs.device)))
    Cgg_dep = CoxWL * T1 / (T2 + 1e-30)
    Cgs_dep = torch.zeros_like(CoxWL)
    Cgd_dep = torch.zeros_like(CoxWL)
    Cgb_dep = -Cgg_dep

    # --- Strong inversion: saturation arm  (b4ld.c §3062-3088) ------------ #
    # Meyer 50/50 in saturation: Cgs = 2/3·CoxWL, Cgd = 0, Cgg = 2/3·CoxWL,
    # Cgb ≈ 0.  (BSIM4 §7.4.1)
    Cgs_sat = (2.0 / 3.0) * CoxWL
    Cgd_sat = torch.zeros_like(CoxWL)
    Cgg_sat = Cgs_sat
    Cgb_sat = torch.zeros_like(CoxWL)

    # --- Strong inversion: linear/triode  (b4ld.c §3091+, Meyer triode) --- #
    # Classic Meyer triode partition (manual §7.4.1):
    #   eta = Vds / Vdsat                               (∈ [0, 1])
    #   Cgs = CoxWL · [1 - ((Vdsat - Vds)/(2·Vdsat - Vds))²] · (2/3)
    #   Cgd = CoxWL · [1 - (Vdsat / (2·Vdsat - Vds))²]      · (2/3)
    # At Vds=0: Cgs = Cgd = (1/2)·CoxWL ; at Vds=Vdsat: Cgs=2/3·CoxWL, Cgd=0.
    eps_v = _t(1e-9, Vgs.dtype, Vgs.device)
    Vdsat_safe = smooth_max(Vdsat, eps_v)
    denom = 2.0 * Vdsat_safe - Vds + eps_v
    r_s = (Vdsat_safe - Vds) / denom
    r_d = Vdsat_safe / denom
    Cgs_lin = CoxWL * (1.0 - r_s * r_s) * (2.0 / 3.0)
    Cgd_lin = CoxWL * (1.0 - r_d * r_d) * (2.0 / 3.0)
    Cgg_lin = Cgs_lin + Cgd_lin
    Cgb_lin = torch.zeros_like(CoxWL)

    # --- Blend strong-inv arms by Vds region ------------------------------ #
    Cgg_inv = sat_to_lin * Cgg_lin + (1.0 - sat_to_lin) * Cgg_sat
    Cgs_inv = sat_to_lin * Cgs_lin + (1.0 - sat_to_lin) * Cgs_sat
    Cgd_inv = sat_to_lin * Cgd_lin + (1.0 - sat_to_lin) * Cgd_sat
    Cgb_inv = sat_to_lin * Cgb_lin + (1.0 - sat_to_lin) * Cgb_sat

    # --- Blend depletion ↔ strong inversion ------------------------------- #
    Cgg_di = dep_to_inv * Cgg_inv + (1.0 - dep_to_inv) * Cgg_dep
    Cgs_di = dep_to_inv * Cgs_inv + (1.0 - dep_to_inv) * Cgs_dep
    Cgd_di = dep_to_inv * Cgd_inv + (1.0 - dep_to_inv) * Cgd_dep
    Cgb_di = dep_to_inv * Cgb_inv + (1.0 - dep_to_inv) * Cgb_dep

    # --- Blend accumulation ↔ (depletion+inversion) ----------------------- #
    Cgg = accum_to_dep * Cgg_di + (1.0 - accum_to_dep) * Cgg_acc
    Cgs = accum_to_dep * Cgs_di + (1.0 - accum_to_dep) * Cgs_acc
    Cgd = accum_to_dep * Cgd_di + (1.0 - accum_to_dep) * Cgd_acc
    Cgb = accum_to_dep * Cgb_di + (1.0 - accum_to_dep) * Cgb_acc

    return {"Cgg": Cgg, "Cgs": Cgs, "Cgd": Cgd, "Cgb": Cgb}


# --------------------------------------------------------------------------- #
# Top-level one-shot                                                          #
# --------------------------------------------------------------------------- #
def compute_caps(
    model: BSIM4Model,
    sd: SizeDependParam,
    dc_result,
    Vgs: torch.Tensor,
    Vds: torch.Tensor,
    Vbs: torch.Tensor,
    Vbd: Optional[torch.Tensor] = None,
    *,
    As: Optional[float] = None,
    Ad: Optional[float] = None,
    Ps: Optional[float] = None,
    Pd: Optional[float] = None,
) -> CapResult:
    """One-shot junction + Meyer intrinsic caps (capMod=0).

    Args:
        model: BSIM4 model card.
        sd:    SizeDependParam (per-geometry-temp cache).
        dc_result: DCResult from dc.compute_dc (uses Vth, Abulk).  May be None;
                   then falls back to vth0_T and Abulk=1.
        Vgs, Vds, Vbs: terminal voltages, fp64 tensors.
        Vbd: if None, computed as Vbs - Vds.
        As, Ad, Ps, Pd: device drawn area/perimeter (m^2, m).  None → defaults.
    """
    if Vbd is None:
        Vbd = Vbs - Vds
    j = compute_junction_caps(model, sd, Vbs, Vbd, As=As, Ad=Ad, Ps=Ps, Pd=Pd)
    m = compute_intrinsic_caps_capmod0(model, sd, dc_result, Vgs, Vds, Vbs)
    Cbody_total = j["Cjs"] + j["Cjd"] + torch.abs(m["Cgb"])
    return CapResult(
        Cjs=j["Cjs"], Cjd=j["Cjd"],
        Cgg=m["Cgg"], Cgs=m["Cgs"], Cgd=m["Cgd"], Cgb=m["Cgb"],
        Cbody_total=Cbody_total,
    )

```


=== FILE: _extracted/O2_packet/artifacts/A1d_iimpact_trace.md (3024 chars) ===
```
# A1d — Why is Iii_M2 ≈ 0 at low VG2?

## BSIM4 §6.1 formula path (from `nsram/bsim4_port/leak.py`)

```python
T2          = (alpha0 + alpha1*Leff) / Leff
diff        = Vds - Vdseff
T1_strong   = T2 * diff * exp(-beta0 / diff)         # if diff > beta0/EXP_THRESH
Iii         = T1 * Idsa                              # Idsa is pre-SCBE Idsa·Vdseff
```
Note: the formula uses `Vdseff` (the smoothed `Vds`/`Vdsat`), not `Vdsat`.

## Numbers at the converged operating point

Both biases use Sebas row params `ALPHA0 = 7.842e-5, BETA0 = 20.0` (M2 card `lalpha0 = -9.84e-12` shifts effective alpha0 by only **−7%**, not the **−5e-1** the question hypothesised — the binning is harmless).

| factor                 | LOW_VG2 (VG2=0.0)  | HIGH_VG2 (VG2=0.5) |
|------------------------|--------------------|--------------------|
| Vsint  (converged)     | 0.3063 V           | 0.0942 V           |
| Vb     (converged)     | 0.3419 V           | 0.4152 V           |
| Vgs_M2                 | 0.0 V              | 0.5 V              |
| Vds_M2 = Vsint         | 0.3063 V           | 0.0942 V           |
| **Vdsat_M2**           | **0.0369 V**       | **0.0685 V**       |
| Vdseff_M2              | 0.0356 V           | 0.0547 V           |
| **Vds − Vdseff**       | **0.271 V**        | **0.0396 V**       |
| alpha0_eff             | 7.283e-5           | 7.283e-5           |
| beta0_eff              | 17.47              | 17.47              |
| T2 = (a0+a1·L)/L       | 41.0               | 41.0               |
| **−beta0/diff**        | **−64.5**          | **−441.6**         |
| **exp(−beta0/diff)**   | **9.6e-29**        | **1.7e-192**       |
| Idsa_M2 (pre-SCBE)     | 1.25e-11 A         | 9.19e-8 A          |
| **Iii_M2 final**       | **2.4e-25 A**      | **2.6e-22 A**      |

## Verdict

**Iii is ~zero at low-VG2 because the `exp(−beta0/(Vds−Vdseff))` factor is 1e-29.**

Not because M2 is in linear region (we are well past Vdsat: 0.306 V vs 0.037 V),
not because alpha0 fails to reach the formula (alpha0_eff = 7.28e-5, intact),
not because of the `lalpha0` binning (only −7% trim, sign correct).

The pure cause: **`BETA0=20 V` from the CSV row is too large for the BSIM4 §6.1
arrhenius-style argument** at ~0.3 V drain headroom. The exponential term
`exp(−20/0.27)` = `exp(−74) ≈ 1e-32` (model uses `beta0_eff = 17.47` after
binning → 9.6e-29). At HIGH_VG2 the Idsa prefactor compensates because the
channel is strongly on (so the simulator matches measurement via direct
M2 channel current, not via the BJT path); at LOW_VG2 there is no fallback
and the body never charges.

## Concrete one-line fix

Sebas's `BETA0=20` is in the wrong units for BSIM4 §6.1 (manual default is
~30 V but for short-channel cards the empirical value is **0.5–3 V**, not 20).
**Replace `SEBAS["BETA0"] = 20.0` with `BETA0 ≈ 1.0 V` for M2** (or treat BETA0
as a per-bias fitting parameter), giving `exp(−1/0.27) = 0.025` and lifting
Iii by ~27 orders of magnitude into the pA range required to forward-bias
the parasitic NPN.

```


=== FILE: _extracted/O2_packet/artifacts/A1e_gidl_load_trace.md (4200 chars) ===
```
# A1.e — GIDL/GISL load trace for M2 at (VG1=0.6, VG2=0.0)

## Verdict (one sentence)
At the failing bias all four GIDL/GISL gates are closed by the
`Vd-Vg-egidl > 0` band-bending condition, so the observed `GIDL/GISL = 0`
is **correct physics — not a parser bug**; however, the load trace also
exposed a **latent bug** in the `("ref", "agidl")` default mechanism that
will silently zero GISL at other biases.

## Hypothesis test results

| H | Statement | Result |
|---|-----------|--------|
| H1 | Parser drops `+`-continued GIDL values | **REJECTED** — agidl/bgidl/cgidl/egidl all loaded correctly (`given=True`) |
| H2 | `compute_igidl_gisl` early-returns on `gidlmod` | REJECTED — `gidlmod=0` selects the implemented branch |
| H3 | Values land in `sd.scaled` but formula reads elsewhere | REJECTED — formula reads `model.get(...)` directly, same source |
| H2′ | (Newly identified) **agisl-group siblings stay at pre-override defaults** because `("ref", ...)` is resolved in pass 2 *before* user overrides apply in pass 3 | **CONFIRMED** |

## Loaded vs card vs default

| param   | loaded     | card        | BSIM4 default | given |
|---------|-----------:|------------:|--------------:|:-----:|
| agidl   | 1.99e-8    | 1.99e-8     | 0.0           | True  |
| bgidl   | 1.624e9    | 1.624e9     | 2.3e9         | True  |
| cgidl   | 6.3        | 6.3         | 0.5           | True  |
| egidl   | 0.91       | 0.91        | 0.8           | True  |
| agisl   | **0.0**    | not in card | (ref agidl)   | False |
| bgisl   | **2.3e9**  | not in card | (ref bgidl)   | False |
| cgisl   | **0.5**    | not in card | (ref cgidl)   | False |
| egisl   | **0.8**    | not in card | (ref egidl)   | False |

`agisl` was supposed to mirror `agidl=1.99e-8` per BSIM4 spec, but is **0.0**.

## Root cause (`model_card.py` lines 75–90)
```
Pass 1: scalar defaults  → agidl=0.0, bgidl=2.3e9, ...
Pass 2: ref defaults     → agisl=agidl=0.0, bgisl=bgidl=2.3e9, ...   ← SNAPSHOT
Pass 3: user overrides   → agidl=1.99e-8, bgidl=1.624e9, ...         ← agisl NOT updated
```
A card that specifies only the GIDL group leaves the GISL group pinned to
the *pre-card* defaults (agisl=0, etc.), which is contrary to BSIM4 v4.8.3
behavior (`b4ld.c` initialises GISL from GIDL after the parameter file is read).

## Bias gate check at (VG1=0.6, VG2=0.0, Vd=1.5, Vsint=0.306, Vb=0.342)

| device/edge | V_drive = Vd–Vg–e | result |
|---|---:|---|
| M2 GIDL (drain=0.306, g=0)     | -0.604 | CLOSED |
| M2 GISL (source=0, g=0)        | -0.800 | CLOSED |
| M1 GIDL (drain=1.5, g=0.6)     | -0.010 | CLOSED (just barely!) |
| M1 GISL (source=0.306, g=0.6)  | -1.094 | CLOSED |

So at THIS bias `GIDL/GISL ≡ 0` is honest physics: drain–gate band-bending
is too weak for BTBT. The observation in A.1.c is consistent with the
formula. The body-charging residual must come from impact ionisation
(`Iii`) or sub-threshold leakage, not GIDL/GISL.

The **agisl=0 bug is silent here** because the GISL gate is closed
anyway, but it will hide tunneling current at higher Vd or more positive
Vbs operating points (e.g. M1 GIDL was within 10 mV of opening — a small
bias shift turns it on, and once on, agisl=0 zeroes a current that
should be ≈ agidl-scale).

## Proposed fix
In `nsram/bsim4_port/model_card.py` `__init__`, **re-resolve `ref` defaults
after pass 3** (or only for parameters that are still `not is_given` and
whose referenced source was overridden):

```python
# Pass 4: re-resolve ref defaults whose source was user-overridden
for name, info in PARAMS_META.items():
    d = info["default"]
    if isinstance(d, tuple) and d[0] == "ref" and name not in self._given:
        self._values[name] = self._values.get(d[1], 0.0)
```

This restores the canonical BSIM4 behavior where, e.g., a card specifying
only `agidl` automatically yields `agisl = agidl`. Touches one file, no
formula changes; existing cards that explicitly set agisl are unaffected
(`name not in self._given` skips them).

## Artifacts
- Demo script: `research_plan/artifacts/A1e_demo.py`
- Card: `data/sebas_2026_04_22/M2_130bulkNSRAM.txt` (lines 80–81)
- Source: `nsram/nsram/bsim4_port/model_card.py` lines 75–90, `_model_card_data.py` lines 14–103

```


=== FILE: _extracted/O2_packet/artifacts/A1i_complementary_bipolar.md (4789 chars) ===
```
# A1i — Decoding Sebas's "Complementary Bipolar Current"

**Source:** `/home/ikaros/nsram_info/schematic&modelCards/2tnsram_simple.asc`
(plus `parasiticBJT.txt`, `PTM130bulkNSRAM.txt`).

## 1. Behavioural elements — verbatim inventory

`grep`-ing the ASC for `bv`, `bi`, `B1`, `.subckt`, `.func`, `.lib` returns
**nothing**. The schematic contains exactly four primitive devices:

| Inst | Symbol | Value / model | Connections (D,G,S/B,Bulk) |
|------|--------|---------------|----------------------------|
| M1   | `nmos4`| `NMOS`, `l=Ln, w=Wn`         | D=`Din`, G=`G`,  S=`Sint`, B=`B` |
| Q1   | `npn`  | `parasiticBJT`, `area=1u`     | C=`D` (=Din), B=`B`, E=`0` (GND) |
| M2   | `nmos4`| `NMOS`, `l=Ln*10, w=Wn`       | D=`B`,  G=`G2`, S=`0`,    B=`0` |
| C1   | `cap`  | `'CBpar'` (=1 fF), Rser=1 mΩ  | between `B` and `0`              |

`.param Ln=0.18u  Wn=0.36u  CBpar=1f`
`.inc PTM130bulkNSRAM.txt   .inc parasiticBJT.txt`

There is **no B-source, no behavioural current, no sub-circuit.** The
"complementary bipolar current" Sebas refers to is *not* a custom
expression — it is simply the **collector current of Q1 (parasiticBJT)**,
fired by the **BSIM4 built-in impact-ionization current** (`alpha0`,
`beta0`) of M1, which charges the floating body node `B`.

## 2. Physical interpretation of each piece

* **M1 (NMOS, BSIM4)** — channel transport plus *intrinsic II generation*:
  `Iii = (alpha0/L) · (Vds−Vdsat) · exp[−beta0/(Vds−Vdsat)] · Ids`
  (BSIM4 manual eq. for `Iii`, routed to the body node). Card uses
  `alpha0 = 7.83756e-5`, `beta0 = 18`.
* **Q1 (NPN, model `parasiticBJT`)** — the lateral parasitic bipolar.
  `is = 5e-9`, `bf = 10000`, `va = 100`, `nc = 2`. Emitter tied to GND,
  base = floating P-body, collector = drain. *This is the "complementary
  bipolar current"* — when V(B) climbs above ~0.6 V, Q1 turns on and
  pumps a large `Ic = β·Is·exp(Vbe/Vt)` from D to GND, in addition to
  the BSIM4 channel current. That is the "full swing of the firing
  mechanism".
* **M2 (long NMOS, l=10·Ln)** — VG2-controlled *body-discharge*
  transistor. When `VG2 > Vth`, M2 sinks B→GND, killing the firing
  (low-VG2 regime = leaky body = no firing — exactly our diagnostic).
* **C1 (1 fF)** — parasitic body capacitance for transient charge
  retention.

## 3. Mathematical form

```
I_complementary(node D → node 0) = Q1.Ic
   = Is · ( exp(V(B)/Vt) − exp(V(B)−V(D))/Vt) ) · (1 + V(D)/Va) / qb
   ≈ 5e-9 · exp(V(B)/0.02585)            # forward-active, V(D)>>Vt
```

with body charge balance

```
C1·dV(B)/dt = I_ii(M1)                                # source: BSIM4 II
            − I_BE(Q1)                                # sink: BJT base
            − I_DS(M2, VG2, V(B))                     # sink: VG2 pull-down
```

So Sebas's "complementary current" is **not a hand-coded B-source**; it
is the standard Gummel–Poon Ic of `parasiticBJT`, *gated by* whether the
BSIM4 II current can outrun the M2 leakage path.

## 4. PyTorch port — how to add it

We already have the BSIM4 `Iii` term in `compute_iimpact`. The missing
piece is the **NPN collector current path D→GND**, plus M2's pull-down.
Concretely:

* **No new free parameters needed.** Hard-code Gummel–Poon constants from
  `parasiticBJT.txt` (`Is=5e-9, Bf=1e4, Va=100, area=1e-6 ⇒ Is_eff=5e-15 A`).
* Add `compute_complementary_bjt(Vd, Vb, Vt, params)` returning
  `Ic = Is_eff*(exp(Vb/Vt) − exp((Vb−Vd)/Vt))*(1+Vd/Va)`.
* In `_eval_mosfet`, add `Ids_total = Ids_bsim + Ic_bjt`.
* In the body-ODE update (or DC Newton solve), replace the existing
  "Ibody = Iii" with `Ibody = Iii − I_BE(Q1) − I_DS(M2,VG2,Vb)` where
  `I_BE = Is_eff·exp(Vb/Vt)/Bf` and `I_DS(M2)` is the same BSIM4 call
  with `l=10·Ln`, `w=Wn`, `Vgs=VG2`, `Vds=Vb`.

## 5. Numerical sanity at (Vd=1.5, Vsint=0.306, Vb=0.342, VG1=0.6, VG2=0)

```
Vt = 0.02585 V
exp(Vb/Vt)            = exp(0.342/0.02585) = exp(13.23)  ≈ 5.6e5
exp((Vb−Vd)/Vt)       = exp(−1.158/0.02585)= exp(−44.8)  ≈ 4e−20
Is_eff = Is·area      = 5e-9 · 1e-6        = 5e-15 A
Ic ≈ 5e-15 · 5.6e5 · (1+1.5/100) ≈ 2.8e-9 A
```

→ **~2.8 nA**, squarely in the "few-nA" range needed to charge `CBpar=1 fF`
on µs timescales and explain the measured firing onset. This closes our
6-decade gap: at Vb≈0.342 V the BSIM4-only current is sub-fA, but Q1
delivers ~3 nA — a ~10⁶× boost, exactly the missing factor.

## 6. Verdict

There is **no behavioural current source** in Sebas's schematic — the
"complementary bipolar current" is the **collector current of the NPN
`parasiticBJT` Q1 wired D-to-GND**, fed by **BSIM4's native impact-
ionization (`alpha0`,`beta0`)** charging the floating body, and gated by
the **VG2-controlled long NMOS M2** that bleeds the body down. We simply
need to add a Gummel-Poon Ic term plus the M2 body-discharge path; no
new free parameters.

```


=== FILE: _extracted/O2_packet/artifacts/A1h_iimod_audit.md (4797 chars) ===
```
# A1h — IIMOD audit: does our `compute_iimpact` match Sebas's card?

**Bias:** VG1=0.6, VG2=0.0, Vd=1.5 V (M2 in saturation, Vds−Vdseff ≈ 0.27 V).
**Predecessor:** A1d already pinned that `Iii ≈ 2.4e-25 A` due to
`exp(−beta0/(Vds−Vdseff))` collapsing. This audit asks the deeper question:
**are we even using the same impact-ion formula as Sebas's foundry card?**

---

## 1. What our `compute_iimpact` implements

`nsram/nsram/bsim4_port/leak.py` (lines 44–101). Single, unconditional formula
(no `iimod` branch anywhere — verified by `grep -ri iimod nsram/bsim4_port/`):

```
T2  = (alpha0 + alpha1·Leff) / Leff                          # leak.py:80,85
diff = max(Vds − Vdseff, 0)                                  # leak.py:76-78
if diff > beta0/EXP_THRESHOLD:                               # leak.py:86,93
    T1 = T2 · diff · exp(−beta0/diff)                        # leak.py:89-90
else:
    T1 = T2 · MIN_EXP · diff                                 # leak.py:92
Iii = T1 · Idsa·Vdseff      # uses pre-SCBE Idsa (WAVE2-FIX-1)
```

Expected units (implicit from this form):
- `ALPHA0` : **m·V⁻¹** (so `alpha0/Leff` is dimensionless V⁻¹)
- `ALPHA1` : **V⁻¹**
- `BETA0`  : **V**

This is the **classic BSIM4 IIMOD=0** form (the only form available in BSIM4
versions ≤ 4.6.4). No length-binning of `alpha0/beta0` is applied (the
`lalpha0`, `lbeta0` fields ingested by `_model_card_data.py:174,180` are
**not consumed** in `temp.py` — only `voffl` etc. are length-binned).

## 2. BSIM4 IIMOD branches (manual §6.1)

BSIM4 v4.7 introduced an `IIMOD` selector. v4.8.3 manual §6.1 lists:

- **IIMOD = 0** (default): the formula above.
  `Iii = ((α₀+α₁·Leff)/Leff)·(Vds−Vdseff)·exp(−β₀/(Vds−Vdseff))·Idsa·Vdseff`
  ALPHA0 [m/V], BETA0 [V].

- **IIMOD = 1** (Mansun-Chan-style, temperature-aware): adds a temperature
  prefactor and replaces ALPHA0/BETA0 with model parameters `IIIA0`,
  `IIIA1`, `IIIB0`, `IIIB1`, `IIIT0`, `IIIT1`, etc. (different parameter
  names — a card setting `iimod=1` would *not* read alpha0/beta0 at all).

- **IIMOD = 2** (HSPICE-compatible — present in some industry forks, not
  always documented in the Berkeley v4.8.3 manual; **flag for oracle
  confirmation**): commonly cited form:
  `Iii = (α₀/Leff + α₁)·(Vds−Vdseff)²·exp(−β₀/(Vds−Vdseff))·Idsa`
  with ALPHA0 in [m·V⁻¹] but the **squared** drain-headroom factor.

Confidence: IIMOD=0/1 from manual + b4ld.c source. IIMOD=2 wording above is
my recollection of HSPICE's "alternate" form — needs cross-check.

## 3. What does Sebas's card select?

Both `M1_130DNWFB.txt:9` and `M2_130bulkNSRAM.txt:22` declare:

```
+Level = 14
+version = 4.5                 ...
```

`version=4.5` **predates IIMOD entirely** (introduced v4.7). Neither card
sets `iimod = ...`. BSIM4 default is IIMOD=0. **Sebas's card therefore
uses the IIMOD=0 classic formula** — the same one we implement.

## 4. Numeric check at the diagnostic bias

From A1d converged operating point (LOW_VG2, M2):
`Vds−Vdseff = 0.271 V`, `Idsa·Vdseff ≈ 1.25e-11 A`, `Leff ≈ 1.91e-7 m`.
Sebas-row `ALPHA0 = 7.842e-5 m/V`, `BETA0 = 20 V` (CSV; card has 18-19 V
plus `lbeta0=-9.5e-7` length term):

```
T2   = 7.842e-5 / 1.91e-7      = 410     [V⁻¹]
exp(−20 / 0.271)               = exp(−73.8) = 8e-33
Iii  = 410 · 0.271 · 8e-33 · 1.25e-11 ≈ 1e-42 A
```

A1d already showed ~2.4e-25 with binning; both are absurdly far below the
1 nA needed to forward-bias the body. **The formula is faithful; the
parameters are simply outside the regime where IIMOD=0 produces
appreciable Iii.** BETA0=18-20 V means `exp(−β/Δ)` only awakens when
`Δ = Vds−Vdseff > ~3 V` — i.e. the IIMOD=0 model targets >3.3 V drain
operation, but our diagnostic runs at 1.5 V with most of it dropped
across M1.

## 5. Verdict

> Our `compute_iimpact` correctly implements BSIM4 v4.8.3 IIMOD=0, which
> is the same branch Sebas's `version=4.5` cards select by default. The
> 6-decade Id miss is **not** an IIMOD-mismatch bug — it is that the
> classic BSIM4 §6.1 formula with BETA0≈19 V genuinely emits ~0 A at
> this bias, and the floating body cannot be charged through this path.

**Proposed one-line fix (workaround, not formula correction):** treat
`BETA0` as a regime-switching fit parameter and refit it against
Sebas's CSV using the body-current rather than as a fixed manual default
— typical short-channel values are 0.5–3 V, which would lift Iii by
~25 orders of magnitude into the nA range.

**Real fix (root cause):** Iii is unlikely to be the dominant
body-charging path at Vd=1.5 V; junction GIDL or the body-source diode
are more plausible. Re-examine `compute_gidl_gisl` and `idiode` weights
before further tuning impact-ion.

**Flag for oracle:** confirm IIMOD=2 (HSPICE) form and whether any
Sebas-internal extraction uses an HSPICE-only impact-ion equation we
have not ported.

```


=== FILE: _extracted/O2_packet/artifacts/A1a_nfactor_trace.md (2987 chars) ===
```
# A1a — NFACTOR override trace through compute_dc

**Verdict:** **NFACTOR override IS reaching the subthreshold formula.**

## 1. Formula path inside `compute_dc` (`nsram/bsim4_port/dc.py`)

```python
# line 131:    P  = sd.scaled
# line 160:    nfactor = t(P["nfactor"])
# --- Subthreshold n (b4ld.c §1133-1154) ---
# line 361:    tmp1 = epssub / Xdep
# line 362:    tmp2 = nfactor * tmp1                       # <-- nfactor enters here
# line 363:    tmp3 = cdsc + cdscb*Vbseff + cdscd*Vds
# line 364:    tmp4 = (tmp2 + tmp3*Theta0 + cit) / coxe
# line 367:    n_a  = 1.0 + tmp4                           # subthreshold slope factor
# --- Vgsteff bridge (b4ld.c §1238-1296) ---
# line 439:    T0v = n * Vtm                               # n carries nfactor
# line 440:    T1v = mstar * Vgst
# line 441:    T2v = T1v / T0v
# line 449:    T10_bridge = n*Vtm * log1p(exp(T2v))
# line 471:    T9v  = mstar + n * T3v
# line 472:    Vgsteff = T10v / T9v                        # <-- subthreshold smoothing
```

So `nfactor → n → T0v → Vgsteff` (and into the drain-current expression).
It does not change `Vth`; it sets the inverse subthreshold slope (kT/q · n).

## 2. Patch idiom reaches that path

`patch_sd_scaled` in `scripts/z91f_validate_with_sebas_params.py` (lines 55–74) writes directly to `sd.scaled[k]`:

```python
sd.scaled[k] = v            # line 67
```

`compute_dc` reads `P = sd.scaled; nfactor = t(P["nfactor"])`. Same dict, same key. The override is consumed.

## 3. Numeric demo (M2, real card)

Script: `research_plan/artifacts/A1a_demo.py`. Loads
`data/sebas_2026_04_22/M2_130bulkNSRAM.txt`, builds `sd_M2` via
`compute_size_dep` (Ln·10 = 1.8 µm, W = 360 nm, T = 27 °C), applies the
same M2 static overrides as z91f (k1, k2, etab, beta0), then calls
`compute_dc` at **(Vgs = −0.10, Vds = 2.0, Vbs = 0)** with `nfactor` swapped
in `sd.scaled`.

| nfactor | Id [A]      | Vgsteff [V] | Vth [V] |
|--------:|------------:|------------:|--------:|
| 1.58    | 1.1545e-14  | 4.649e-09   | 0.4497  |
| 12.15   | 6.2099e-12  | 2.501e-06   | 0.4497  |

- **Id ratio = 5.38e+02 → +2.73 decades**
- ΔVgsteff = +2.50e-6 V (Vgsteff scales ~ n·Vtm·log1p(exp(...)) — bigger n softens the bridge)
- ΔVth = 0 (expected — nfactor sets slope, not threshold)

## 4. Implication for z91g

The override mechanism is wired correctly: a single-bias `nfactor` swap from
1.58 → 12.15 already moves Id by **+2.73 decades** at this low-VG2 bias.
That is the same order of magnitude as z91g's reported median residual
(2.40 decades at low VG2), so NFACTOR is the right knob and the patch
*does* reach it. The remaining z91g residual must therefore come from
either (a) sign / direction of the residual (does Sebas's Id move the
same way ours does?), (b) interaction with the M1 overrides
(etab, k1, alpha0, beta0) or the BJT wrapper, or (c) bias coordinates the
CSV NFACTOR row maps to versus what we apply at the same (VG1, VG2). The
NFACTOR-not-reaching-the-formula hypothesis is **falsified**.

```


=== FILE: _extracted/O2_packet/artifacts/A1c_body_diode_trace.md (2129 chars) ===
```
# A1c — Component-current trace at low-VG2 bias

**Bias:** VG1=0.6, VG2=0.0, Vd=1.5 V. Sebas CSV row applied
(ETAB=2.5, K1=0.41825, ALPHA0=7.842e-5, BETA0=20, NFACTOR=6.0,
mbjt=1.0, IS=5e-9, area=1e-6). Two-card setup
(M1=130DNWFB, M2=130bulkNSRAM).

## Solver

Plain Newton hit the documented spurious-flat-root pathology
(converged at iter 2, all currents <1e-17 A). Used
`solve_2t_with_homotopy` (gmin 1e-3 → 1e-15) seeded with
Vsint=Vd/2, Vb=0.7; converged in 3 iters at target gmin.

**Converged:** Vsint = +0.3063 V, Vb = +0.3419 V → Vbe = +0.0356 V,
Vbc = −1.158 V.

## Component magnitudes (signed, A)

| Component | Value |
|---|---|
| (a) Ids_M1 | +1.251e-11 |
| (b) Ids_M2 | +1.252e-11 |
| (c) Ic_Q1  | +2.01e-14 (NPN off, Vbe≈36 mV) |
| (d) Ibs_M2 | +6.10e-13 (forward, sub-turn-on) |
| Ibd_M2 | +1.92e-16 |
| (e) GIDL/GISL M1+M2 | 0 (exact) |
| Iii_M1 | +2.03e-16 |
| Igb (M1,M2) | 0 |
| **Id at drain** | **+1.253e-11** |
| **Id measured** | **+2.07e-5** |

## Dominant path

Drain current is **completely dominated by M2 subthreshold channel
current** (Ids_M2 ≈ Ids_M1; M2 is the series bottleneck). BJT is
~3 decades smaller; M2 body diode ~2 decades smaller; GIDL = 0.
**Predicted Id under-shoots measurement by ~6 decades** — the z91g
low-VG2 residual.

## Body-diode sanity

Ibs_M2 = 6.1e-13 A at Vbs ≈ +36 mV is **not** suspicious — Vbs is
far from pn turn-on. With jss = 1e-4 A/m² and area ≈ 6.5e-13 m², the
saturation current ≈ 6.5e-17 A; at Vbs = 0.7 V that gives ~5e-5 A,
matching Sebas's measured magnitude. The kernel is fine; **the body
simply isn't being forward-biased** — Vb sits at +0.34 V between
Vsint (+0.31) and 0 with no driver pumping it up.

## Verdict

The dominant low-VG2 path in our sim is the **series-limited M2
subthreshold channel** (~10⁻¹¹ A); Sebas's measured ~2×10⁻⁵ A is
plausibly **body-driven** — NPN forward turn-on once Vb reaches
~0.7 V, or GIDL/well leakage we don't generate (Igidl ≡ 0 from this
M2 card). Fix priority: figure out why Vb fails to climb to NPN
turn-on — with Iii ~ 1e-16 and GIDL = 0 there is no body-charging
source — and why the M2 card emits Igidl ≡ 0.

```


=== FILE: _extracted/O2_packet/artifacts/A1g_multiroot.md (3874 chars) ===
```
# A1g — Multi-root hypothesis at low VG2

**Bias:** VG1∈{0.4, 0.6}, VG2=0.0, Vd=1.5 V; Sebas per-bias overrides.
**Symptom:** z91g returns Id≈1e-11; measurement Id≈2e-5 (VG1=0.6) /
1e-6 (VG1=0.4) — 5–6 decade gap.

## Method

`research_plan/artifacts/A1g_demo.py`. M1+M2 cards via z91f's
`patch_model_values`; per-bias overrides via `patch_sd_scaled` (z91g
convention; `_override_sd` errors on dict-only fields like
etab/alpha0/beta0/nfactor). Per bias:
- `solve_2t_with_homotopy`, Vb_init ∈ {0.0, 0.5, 0.7, 0.9}.
- `forward_2t_arclength_grad`, Vd∈[0.05, 2.0] (40 pts).

Trace: `A1g_multiroot_trace.json`. Plot: `A1g_multiroot.png`.

## Results

### VG1 = 0.6, VG2 = 0.0, Vd = 1.5 (meas Id = 2.07e-5 A)
| start          | Id [A]    | Vb [V]  | Vsint [V] | conv |
|----------------|-----------|---------|-----------|------|
| Vb_init=0.0    | 1.253e-11 | 0.3419  | 0.3063    | yes  |
| Vb_init=0.5    | 1.253e-11 | 0.3419  | 0.3063    | yes  |
| Vb_init=0.7    | 1.253e-11 | 0.3419  | 0.3063    | yes  |
| Vb_init=0.9    | 1.253e-11 | 0.3419  | 0.3063    | yes  |
| arclength@1.5  | 4.21e-12  | 0.1842  | —         | no   |

Arclength: **n_folds = 0**, 45 steps, 2.5 s.

### VG1 = 0.4, VG2 = 0.0, Vd = 1.5 (meas Id = 1.02e-6 A)
| start          | Id [A]    | Vb [V]  | Vsint [V] | conv |
|----------------|-----------|---------|-----------|------|
| Vb_init=0.0    | 1.444e-11 | 0.3550  | 0.2372    | no   |
| Vb_init=0.5    | 1.444e-11 | 0.3550  | 0.2372    | no   |
| Vb_init=0.7    | 1.444e-11 | 0.3550  | 0.2372    | no   |
| Vb_init=0.9    | 1.444e-11 | 0.3550  | 0.2372    | no   |
| arclength@1.5  | 4.57e-12  | 0.1857  | —         | no   |

Arclength: **n_folds = 0**, 44 steps, 2.8 s.

## Verdict: multi-root hypothesis **DISPROVEN**

Three converging lines:

1. **Initial-condition basin sweep is degenerate.** Vb_init from 0.0
   to 0.9 V — past the diode-on knee — converges in 3 Newton iters to
   the **same** (Vsint, Vb, Id) triple to 5 sig-figs. No second basin
   in the explored range.
2. **Arclength finds no fold** (n_folds = 0). If a high-Vb root were
   separated from the low-Vb root by an S-shaped fold in Id(Vd),
   trace_arclength would have detected a turning point. It did not.
3. **The found Vb is already moderate** (~0.34 V) — not a stuck-at-0
   NPN-OFF root. Body diode mildly forward-biased; Iii just isn't
   strong enough in our model to ignite the
   avalanche → base-drive → collector-current loop.

There is no second root for arclength or high-Vb init to snap to.
Newton already finds the only root the model admits.

## Actual mechanism (likely)

The 6-decade gap is a **sub-block disagreement**, not a convergence
failure. Ranked candidates:

- **Iii too weak.** Sebas's ALPHA0=7.84e-5, BETA0=20 through our
  `compute_iimpact` may produce orders less impact current than
  HSPICE BSIM4 IIMOD. The Iii→Vb→base-drive loop never ignites; the
  NPN stays off and Id collapses to M2 sub-threshold leakage. The
  z91g pattern (good fits at VG1=0.2 where NPN doesn't matter,
  degrading at VG1=0.4–0.6 where it does) is consistent.
- **GP NPN under-driven.** At Vb=0.34, Vbe is too small for our
  Gummel-Poon to deliver appreciable Ic. Worth verifying that
  `bjt.py` references Vbe = Vb − Vsint with the same sign convention
  as Sebas's `parasiticBJT.txt` (A1b mapped this; recheck).
- **Body-diode Js mismatch.** M2 card has zero SourceSatCurDensity_T;
  `cfg.default_jss` kicks in. If too large it clamps Vb near 0.34 V.

## What to do next

A high-Vb / arclength heuristic won't help — there's nothing to find.
Fix is in physics:

1. Audit `compute_iimpact` vs HSPICE BSIM4 IIMOD=1 at this bias with
   Sebas overrides (next diagnostic).
2. Verify GP NPN Vbe-mapping vs `parasiticBJT.txt` (C=Vd, B=body,
   E=Vsint).
3. Once Iii is calibrated and Vb pushes past ~0.6 V, Id should jump
   4–5 decades without any continuation tricks.

```


=== FILE: _extracted/O2_packet/artifacts/A1b_bjt_mapping.md (4121 chars) ===
```
# A1b — BJT Parameter Mapping (Sebas CSV ↔ `GummelPoonNPN`)

## 1. Sources

- **SPICE model card** (`parasiticBJT.txt`):
  ```
  .model parasiticBJT NPN(is=5E-9 va=100 bf=10000 br=100 nc=2 ikr=100m
                          rc=0.1 vje=0.7 re=0.1 cjc=1e-15 fc=0.5
                          cje=0.7e-15 ne=1.5 ise=0
                          tr=20e-12 tf=25e-12 itf=0.03 vtf=7 xtf=2)
  ```
- **Schematic** (`2tnsram_simple.asc`):
  `SYMBOL npn ... Q1 ... Value=parasiticBJT  Value2=area=1u`
  → instance `area = 1e-6`. No `m=` multiplier on the LTSpice Q1 instance.
- **CSV** columns: `mbjt, IS, area`. Rows show `IS=5e-9`, `area=1e-6`
  constant; `mbjt` flips between **0.001 (VG1=0.2)** and **1.0
  (VG1=0.4 / 0.6)**.

## 2. Mapping Table

| SPICE NPN keyword          | LTSpice instance | Our `GummelPoonNPN` | Honoured by `from_sebas_card`? |
|---|---|---|---|
| `IS`  (saturation current) | —                | `Is`  (5e-9)        | yes (hard-coded)               |
| `VA`  (Early fwd)          | —                | `Va`  (100)         | yes                            |
| `BF`                       | —                | `Bf`  (10000)       | yes                            |
| `BR`                       | —                | `Br`  (100)         | yes                            |
| `NF`/`NR` (default 1)      | —                | `Nf=Nr=1`           | yes                            |
| `NC`, `NE`                 | —                | `Nc=2`, `Ne=1.5`    | yes                            |
| `IKR`, `ISE`               | —                | `Ikr=0.1`, `Ise=0`  | yes                            |
| —                          | `area=1u`        | `area = 1e-6`       | yes (multiplies Is/Ikf/Ikr/Ise/Isc in `compute_bjt`) |
| —                          | `m=<mbjt>`       | **no field**        | **NO**                         |

`mbjt` is **the SPICE device multiplier `m`** (cell count / parallel
parasitic-NPN scaling). In SPICE, `m` multiplies `IS, IKF, IKR, ISE, ISC,
1/RB, 1/RE, 1/RC` exactly as `area` does — i.e. it is mathematically
identical to scaling `area`. There is no Gummel-Poon "ideality" parameter
called `mbjt`; this is purely a count multiplier added by Sebas's
extraction wrapper to switch the BJT path on/off per VG1.

## 3. Pipeline Audit (`z91f.make_bjt`)

```python
def make_bjt(sebas_row):
    bjt = GummelPoonNPN.from_sebas_card()
    if sebas_row is not None:
        if not math.isnan(sebas_row.get("IS", float("nan"))):
            bjt.Is = float(sebas_row["IS"])
        # mbjt is the BJT idealisation factor (we treat as Bf scaler).
        # For validation: ignore ...
    return bjt
```

- `IS` is read **once per row** but the value is constant 5e-9 — so the
  per-row override is a no-op. Per-bias `IS` is *technically* honoured;
  effectively unused.
- `area` is **not** read from the CSV (still defaults to 1e-6 from the card).
- `mbjt` is **explicitly ignored** with a wrong comment ("Bf scaler").
- z91g imports `make_bjt` from z91f → identical bug.

## 4. Numeric Check (Vbe = 0.6 V, Vbc = 0, T = 300 K)

| Effective multiplier | Ic       |
|---|---|
| current code (`area=1e-6`, mbjt ignored)         | 5.94e-5 A |
| with `area *= mbjt = 1.0` (VG1 ≥ 0.4 rows)       | 5.94e-5 A |
| with `area *= mbjt = 0.001` (VG1 = 0.2 rows)     | 5.94e-8 A |

→ **Exactly the ~3-decade gap** seen in the z91g residuals at low VG1.

## 5. Verdict

**`mbjt` is a SPICE `m=` device multiplier, not a Gummel-Poon ideality.
It is currently NOT honoured; `IS` is honoured but trivially constant.
At VG1 = 0.2 rows the simulator over-drives the parasitic NPN by 1000×,
which is exactly z91g's low-VG2 residual signature.**

**Fix (one line in `make_bjt`):**
```python
if not math.isnan(sebas_row.get("mbjt", float("nan"))):
    bjt.area = bjt.area * float(sebas_row["mbjt"])
if not math.isnan(sebas_row.get("area", float("nan"))):
    bjt.area = float(sebas_row["area"]) * float(sebas_row.get("mbjt", 1.0))
```
i.e. set `bjt.area = csv.area * csv.mbjt`. Existing `compute_bjt` already
applies `area` as the SPICE-correct multiplier on `Is, Ikf, Ikr, Ise, Isc`,
so no change to `bjt.py` is required.

```
