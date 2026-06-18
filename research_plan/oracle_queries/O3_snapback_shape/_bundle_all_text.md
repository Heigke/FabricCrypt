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


=== FILE: arclength.py (31233 chars) ===
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


def _solve_initial_point_single(cfg, model, bjt, Vd0, VG1, VG2,
                          Vsint_init=None, Vb_init=None,
                          max_iters: int = 30, tol: float = 1e-13,
                          model_M2=None):
    """Plain Newton at Vd=Vd0 from a single (Vsint, Vb) seed."""
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


def _solve_initial_point(cfg, model, bjt, Vd0, VG1, VG2,
                          Vsint_init=None, Vb_init=None,
                          max_iters: int = 30, tol: float = 1e-13,
                          model_M2=None):
    """Multi-restart wrapper around _solve_initial_point_single.

    Tries several (Vsint, Vb) seeds and returns the first that converges.
    Seeds cover both the off-branch (Vb≈0) and on-branch (Vb≈0.7-0.9)
    families so a hot snapback start is reachable from cold defaults.
    The user-supplied seed is tried first when given; otherwise the seed
    list contains all defaults.
    """
    Vd0_f = float(torch.as_tensor(Vd0))
    seeds = []
    if Vsint_init is not None or Vb_init is not None:
        seeds.append((Vsint_init, Vb_init))
    # Cold off-branch
    seeds.append((0.5 * Vd0_f, 0.0))
    # Mid Vsint, mid Vb
    seeds.append((0.3 * Vd0_f, 0.4))
    # Hot on-branch (BJT bias near snapback turn-on)
    seeds.append((0.2 * Vd0_f, 0.75))
    seeds.append((0.1 * Vd0_f, 0.85))
    # Near-zero Vsint (M1 in saturation)
    seeds.append((0.05, 0.0))
    best = None
    for s_v, b_v in seeds:
        Vsint, Vb, ok = _solve_initial_point_single(
            cfg, model, bjt, Vd0, VG1, VG2,
            Vsint_init=s_v, Vb_init=b_v,
            max_iters=max_iters, tol=tol, model_M2=model_M2)
        if ok:
            return Vsint, Vb, True
        if best is None:
            best = (Vsint, Vb)
    return best[0], best[1], False


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
                                  max_iters: int = 15, tol: float = 1e-13,
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
    consec_fail = 0    # consecutive corrector failures at ds_min
    max_consec_fail = 6

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
            # Step too large — bisect first, then try perturbed restarts.
            if ds > ds_min * 1.01:
                ds = max(ds * 0.5, ds_min)
                continue
            # At ds_min and still failing. Try perturbed predictors before
            # giving up: jitter Vb seed (the snapback fold is dominantly a
            # body-voltage instability), keep the current tangent.
            recovered = False
            for perturb_Vb in (0.05, -0.05, 0.15, -0.15, 0.30, -0.30):
                x_try = x_pred.clone()
                x_try[1] = x_try[1] + perturb_Vb
                x_pp, n_it_pp, conv_pp = _newton_arclength_corrector(
                    cfg, model, bjt, x_try, x_prev=x, t_prev=t, ds=ds_min,
                    VG1=VG1, VG2=VG2, P_M1=P_M1, P_M2=P_M2,
                    model_M2=model_M2,
                )
                if conv_pp:
                    x_new, n_iter, conv = x_pp, n_it_pp, True
                    recovered = True
                    break
            if not recovered:
                # Skip this region: take a tiny step along tangent and try
                # to re-acquire the path on the other side. Don't append a
                # bogus unconverged point (it pollutes interpolation).
                consec_fail += 1
                x = x + ds_min * t
                # Re-estimate tangent at the new (uncorrected) location
                try:
                    t = _compute_tangent(cfg, model, bjt, x[2], VG1, VG2,
                                           x[0], x[1], P_M1, P_M2,
                                           prev_t=t, model_M2=model_M2)
                except Exception:
                    pass
                if consec_fail >= max_consec_fail:
                    break
                # Allow ds to grow again on next iteration
                ds = max(ds_min * 4.0, ds)
                continue

        # Successful corrector
        consec_fail = 0

        # Compute new tangent (with sign consistency)
        t_new = _compute_tangent(cfg, model, bjt, x_new[2], VG1, VG2,
                                   x_new[0], x_new[1], P_M1, P_M2,
                                   prev_t=t, model_M2=model_M2)
        # Detect fold: dVd/ds sign change
        new_dVd_sign = torch.sign(t_new[2])
        if new_dVd_sign != prev_dVd_sign and abs(prev_dVd_sign) > 0:
            n_folds += 1
        prev_dVd_sign = new_dVd_sign

        # Tangent-rotation shrink: if the tangent rotates fast, we are
        # near a fold or sharp bend; force smaller ds for the next step
        # so the predictor doesn't shoot off the manifold.
        cos_rot = float(torch.dot(t, t_new))
        cos_rot = max(-1.0, min(1.0, cos_rot))

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
        # Override if tangent rotated > ~30°
        if cos_rot < 0.866:
            ds = max(ds * 0.4, ds_min)
        # Override hard near fold (> ~60° rotation): drop to ds_min*4
        if cos_rot < 0.5:
            ds = max(ds_min * 4.0, ds_min)

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


def _trace_backward(cfg, model, bjt, VG1, VG2,
                     Vd_start: float, Vd_min: float,
                     P_M1=None, P_M2=None, model_M2=None,
                     **kwargs) -> dict:
    """Trace from Vd_start downward to Vd_min using a hot on-branch seed.

    Implementation: re-uses trace_arclength with hot init seeds. We pick
    the seed at Vd_start so the initial point lives on the on-branch
    (post-fold) of the snapback. Then we let the natural arclength loop
    fold back down. Path is post-processed (reversed) so it reads in
    increasing-arclength order along Vd ascending where possible.
    """
    # Find a hot on-branch starting point at Vd_start by trying multiple
    # seeds biased toward on-branch (Vb≈0.7-0.9, Vsint near 0).
    Vsint0, Vb0, init_ok = _solve_initial_point(
        cfg, model, bjt, Vd_start, VG1, VG2,
        Vsint_init=0.1, Vb_init=0.85, model_M2=model_M2)
    if not init_ok:
        return {"path_Vd": [], "path_Vsint": [], "path_Vb": [],
                "path_Id": [], "converged": [], "n_steps": 0,
                "n_folds": 0, "init_ok": False}

    # Reuse trace_arclength but trick it into going backward by passing
    # Vd_start at the high end and Vd_max as the low end via internal
    # surgery: easier to just inline a mirror loop here.
    VG1 = torch.as_tensor(VG1, dtype=torch.float64)
    VG2 = torch.as_tensor(VG2, dtype=torch.float64)
    x = torch.tensor([float(Vsint0), float(Vb0), float(Vd_start)],
                      dtype=torch.float64)
    # Initial tangent: prefer DECREASING Vd direction
    t = _compute_tangent(cfg, model, bjt, x[2], VG1, VG2, x[0], x[1],
                          P_M1, P_M2, prev_t=None, model_M2=model_M2)
    if t[2] > 0:
        t = -t
    ds = 0.01
    ds_min = 1e-4
    ds_max = 0.05
    max_steps = 2000
    path_Vd = [float(x[2])]
    path_Vsint = [float(x[0])]
    path_Vb = [float(x[1])]
    converged_flags = [True]
    n_folds = 0
    n_steps = 0
    consec_fail = 0
    for step in range(max_steps):
        x_pred = x + ds * t
        x_new, n_iter, conv = _newton_arclength_corrector(
            cfg, model, bjt, x_pred, x_prev=x, t_prev=t, ds=ds,
            VG1=VG1, VG2=VG2, P_M1=P_M1, P_M2=P_M2,
            model_M2=model_M2,
        )
        if not conv:
            if ds > ds_min * 1.01:
                ds = max(ds * 0.5, ds_min)
                continue
            recovered = False
            for perturb_Vb in (0.05, -0.05, 0.15, -0.15):
                x_try = x_pred.clone(); x_try[1] = x_try[1] + perturb_Vb
                x_pp, n_it_pp, conv_pp = _newton_arclength_corrector(
                    cfg, model, bjt, x_try, x_prev=x, t_prev=t, ds=ds_min,
                    VG1=VG1, VG2=VG2, P_M1=P_M1, P_M2=P_M2,
                    model_M2=model_M2)
                if conv_pp:
                    x_new, n_iter, conv = x_pp, n_it_pp, True
                    recovered = True
                    break
            if not recovered:
                consec_fail += 1
                if consec_fail >= 6:
                    break
                x = x + ds_min * t
                continue
        consec_fail = 0
        t_new = _compute_tangent(cfg, model, bjt, x_new[2], VG1, VG2,
                                   x_new[0], x_new[1], P_M1, P_M2,
                                   prev_t=t, model_M2=model_M2)
        cos_rot = float(torch.dot(t, t_new))
        cos_rot = max(-1.0, min(1.0, cos_rot))
        x = x_new; t = t_new
        n_steps += 1
        path_Vd.append(float(x[2]))
        path_Vsint.append(float(x[0]))
        path_Vb.append(float(x[1]))
        converged_flags.append(True)
        if n_iter > 8: ds = max(ds * 0.7, ds_min)
        elif n_iter <= 3: ds = min(ds * 1.3, ds_max)
        if cos_rot < 0.866: ds = max(ds * 0.4, ds_min)
        # Termination conditions
        if x[2] <= Vd_min:
            break
        if step > 50 and abs(path_Vd[-1] - path_Vd[-50]) < 1e-3:
            break

    # Compute Id along path
    path_Vd_t = torch.tensor(path_Vd, dtype=torch.float64)
    path_Vsint_t = torch.tensor(path_Vsint, dtype=torch.float64)
    path_Vb_t = torch.tensor(path_Vb, dtype=torch.float64)
    with torch.no_grad():
        _, _, comp = _residuals(cfg, model, bjt,
                                 path_Vd_t, VG1.expand_as(path_Vd_t),
                                 VG2.expand_as(path_Vd_t),
                                 path_Vsint_t, path_Vb_t,
                                 P_M1, P_M2, model_M2=model_M2)
        Id = comp.get("Id_total", comp.get("Ids_M1", torch.zeros_like(path_Vd_t)))
    return {"path_Vd": path_Vd, "path_Vsint": path_Vsint,
            "path_Vb": path_Vb, "path_Id": [float(x) for x in Id],
            "converged": converged_flags, "n_steps": n_steps,
            "n_folds": n_folds, "init_ok": True}


def _merge_paths(fwd: dict, bwd: dict) -> dict:
    """Merge forward and backward arclength sweeps.

    The backward sweep is appended as-is to the forward path. The
    interpolator scans segments in order and uses the first bracket; for
    Vd values not reached by forward but reached by backward, the
    backward segments will provide the bracket. We do NOT reverse bwd —
    interpolation is direction-agnostic.
    """
    return {
        "path_Vd": list(fwd["path_Vd"]) + list(bwd["path_Vd"]),
        "path_Vsint": list(fwd["path_Vsint"]) + list(bwd["path_Vsint"]),
        "path_Vb": list(fwd["path_Vb"]) + list(bwd["path_Vb"]),
        "path_Id": list(fwd["path_Id"]) + list(bwd["path_Id"]),
        "converged": list(fwd["converged"]) + list(bwd["converged"]),
        "n_steps": fwd["n_steps"] + bwd["n_steps"],
        "n_folds": fwd["n_folds"] + bwd["n_folds"],
        "init_ok": True,
    }


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
        Vd_min_f = float(Vd_seq.min())
        Vd_max_f = float(Vd_seq.max())
        path = trace_arclength(cfg, model_M1, bjt, VG1_t, VG2_t,
                                Vd_start=Vd_min_f,
                                Vd_max=Vd_max_f,
                                P_M1=P_M1, P_M2=P_M2,
                                model_M2=model_M2)
        forward_init_ok = path.get("init_ok", False)
        # If forward path didn't span full Vd range, run a backward sweep
        # from Vd_max with a hot on-branch initial seed and merge.
        if forward_init_ok:
            max_Vd_reached = max(path["path_Vd"]) if path["path_Vd"] else Vd_min_f
        else:
            max_Vd_reached = -1e9
        if max_Vd_reached < Vd_max_f - 1e-3:
            # Trace from Vd_max downward. The hot on-branch seed (Vb≈0.85)
            # lives there, so the backward sweep typically picks up the
            # post-fold portion that the forward sweep missed.
            path_bwd = _trace_backward(
                cfg, model_M1, bjt, VG1_t, VG2_t,
                Vd_start=Vd_max_f, Vd_min=Vd_min_f,
                P_M1=P_M1, P_M2=P_M2, model_M2=model_M2)
            if path_bwd.get("init_ok", False):
                if not forward_init_ok:
                    path = path_bwd
                else:
                    path = _merge_paths(path, path_bwd)
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


=== FILE: bjt.py (6347 chars) ===
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


=== FILE: compute_body_diodes.py (5121 chars) ===
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


=== FILE: compute_dc.py (39248 chars) ===
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


=== FILE: compute_iimpact_and_gidl.py (18552 chars) ===
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


=== FILE: nsram_cell_2T.py (47404 chars) ===
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

    # Deep-N-well bias on M1 (130nm DNWFB device).
    # ──────────────────────────────────────────────────────────────────
    # IMPORTANT (2026-05-01, A.1.n finding): Sebas's measurement and
    # SPICE deck apply +2 V to the deep-N-well terminal of M1. The
    # well/body PN junction is forward-biased (well at +2 V vs floating
    # body at ~0 V), pumping current into the body. THIS is the missing
    # body-charging path that explains our 5-decade Id under-prediction
    # at low VG2. The schematic doesn't show the well node; the bias is
    # applied externally on the package pin, with the well capacitance
    # and series resistance internal.
    use_well_diode: bool = True
    vnwell: float = 2.0              # deep-N-well voltage [V]
    # Series-R production default = 1e9 Ω. Grid search (A.2 z91h_grid)
    # found tighter median at Rs=3e9 (0.79) and Rs=1e10 (0.69) but those
    # come with coverage loss — the arclength solver loses the snapback
    # fold mid-trace when vnwell coupling is strong. A.1.s solver work
    # (dual-direction sweep, branch detection) needed to unlock those
    # settings. Until then, Rs=1e9 gives the full 25/25 coverage at
    # honest median 1.19 / p90 2.88.
    vnwell_Rs: float = 1.0e10
    vnwell_area: float = 1.0e-12     # well-body junction area [m²] (~1 µm²; tiny)
    vnwell_Js: float = 3.4089e-7     # saturation current density [A/m²] (jss)
    vnwell_n: float = 1.017          # diode emission factor (njs)
    # mbjt-tracking: Sebas's CSV mbjt column scales the parasitic-NPN
    # area; physically the well-body junction belongs to the same parasitic
    # bipolar structure, so it should track the same multiplier. At
    # mbjt=0.001 (VG1=0.2 in his data) the well coupling effectively
    # disappears; at mbjt=1.0 (VG1=0.4/0.6) it's fully present.
    vnwell_mbjt: float = 1.0
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

    # Deep-N-well to body diode (A.1.n: this is the missing body-charging path).
    # ──────────────────────────────────────────────────────────────────
    # When vnwell > Vb, the N-well/P-body junction forward-biases and pumps
    # current INTO the body. Modelled as a Shockley diode with series R:
    #
    #     I_ideal  = Js·A · (exp((vnwell − Vb)/(n·Vt)) − 1)
    #     I_Rs     = (vnwell − Vb) / Rs   (when forward biased)
    #     I_well_b = harmonic_mean(I_ideal, I_Rs)   smooth transition
    #
    # Reverse-bias contribution is tiny (Js·A ~1e-15 A) — included for
    # completeness so derivatives are continuous through Vb crossing vnwell.
    if cfg.use_well_diode:
        Vt = 0.02585 * (273.15 + cfg.T_C) / 300.0   # thermal voltage at T
        V_drive = cfg.vnwell - Vb
        # Clamp exponent to avoid overflow when V_drive >> Vt
        exp_arg = (V_drive / (cfg.vnwell_n * Vt)).clamp(max=40.0)
        I_ideal = cfg.vnwell_Js * cfg.vnwell_area * (torch.exp(exp_arg) - 1.0)
        # Series-R limited current (only forward; reverse bias = 0 here)
        I_Rs = torch.relu(V_drive) / cfg.vnwell_Rs
        # Smooth min via harmonic mean (differentiable, transitions at the
        # smaller of the two without a hard kink)
        eps = 1e-30
        I_well_body = (I_ideal * I_Rs) / (I_ideal.abs() + I_Rs + eps)
        # Scale by mbjt — the well-body junction belongs to the same
        # parasitic bipolar structure as Q1, so it follows the same
        # device-multiplier. Without this scaling, VG1=0.2 (where
        # mbjt=0.001 keeps the BJT off) would still see full well
        # coupling and the body would float high.
        I_well_body = I_well_body * cfg.vnwell_mbjt
    else:
        I_well_body = torch.zeros_like(Vd)

    # ---- Body KCL: currents INTO B ------------------------------------ #
    # Iii, Igidl, Igisl, Igb are already signed +INTO-body in the helpers.
    # Body junction diodes: Ibs and Ibd are POSITIVE-LEAVING-body, so we
    # subtract them.
    # BJT base current Ib (positive INTO base from external) — for the
    # floating body, the only external current into the base IS the body
    # node itself. Ib>0 ⇒ body sources current → leaves body. → −Ib_Q1
    # Well-body diode I_well_body is +INTO body (well pumps body up). → +I_well_body
    R_B = (
        m1["Iii"] + m2["Iii"]
        + m1["Igidl"] + m1["Igisl"] + m2["Igidl"] + m2["Igisl"]
        + m1["Igb"] + m2["Igb"]
        - m1["Ibs"] - m1["Ibd"]
        - m2["Ibs"] - m2["Ibd"]
        - Ib_Q1
        + I_well_body
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
        "I_well_body": I_well_body,
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
  "median_log_rmse": 0.953504129997842,
  "p90_log_rmse": 1.9999951051223395,
  "elapsed_s": 48.60951018333435,
  "vs_z91f_run1_median": 4.234,
  "vs_z91f_run2_median": 2.402,
  "note": "true two-model validation (M1 = 130DNWFB, M2 = 130bulkNSRAM) with Sebastian's per-bias CSV overrides"
}
```


=== FILE: z91g_two_model_validation.py (10292 chars) ===
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
import json, math, os, re, csv, time
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
        # z91h grid-search optimum (revisited post-A.1.s): Bf=5e4 + α0×10
        # gives lowest RMSE; previously these cut coverage 25→19 but the
        # robust arclength solver (A.1.s, tighter corrector tol + branch
        # detection) now keeps full coverage at these settings.
        bjt.Bf = 5.0e4
        # Per-bias mbjt scales BOTH the BJT (already in make_bjt) AND the
        # well-body diode (cfg.vnwell_mbjt). At VG1=0.2 mbjt=0.001 → both
        # parasitic paths off; at VG1=0.4/0.6 mbjt=1 → fully on.
        mbjt = float(sebas_row.get("mbjt", 1.0))
        if math.isnan(mbjt):
            mbjt = 1.0
        cfg.vnwell_mbjt = mbjt
        # α0 multiplier — z91h grid found ×10 best at smooth-ramp regime,
        # but user feedback says shape is too smooth. Try ×100 to push
        # feedback loop gain higher and see if knee sharpens (env override).
        if P_M1 is None:
            P_M1 = {}
        a0_csv = sebas_row.get("ALPHA0", 7.842e-5)
        if not math.isnan(a0_csv):
            a0_mult = float(os.environ.get("NSRAM_A0_MULT", "10.0"))
            P_M1["alpha0"] = torch.tensor(a0_mult * a0_csv, dtype=torch.float64)
        # GPT-5 / O2 oracle injection-limited hypothesis test (A.1.q).
        # NSRAM_BETA0_TEST > 0 overrides M1 and M2 beta0 in compute_iimpact
        # to test if smaller β0 lights the body. Sebas's CSV says β0≈18-20;
        # if exp(-β0/Δ) at Δ≈0.27V is the killer, β0=1.5 → exp(-5.5)=0.004
        # vs current exp(-74)=e-32. Decisive single-variable experiment.
        BETA0_TEST = float(os.environ.get("NSRAM_BETA0_TEST", "0"))
        if BETA0_TEST > 0:
            if P_M1 is None:
                P_M1 = {}
            if P_M2 is None:
                P_M2 = {}
            P_M1["beta0"] = torch.tensor(BETA0_TEST, dtype=torch.float64)
            P_M2["beta0"] = torch.tensor(BETA0_TEST, dtype=torch.float64)
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
