# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: M2_card.txt (10507 chars) ===
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


=== FILE: compute_dc_vgsteff.py (2096 chars) ===
```python
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

```


=== FILE: compute_dc_vth.py (13582 chars) ===
```python
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
    # A.5.f (2026-05-02): use a SEPARATE Theta0 for the n branch with the
    # half-factor per BSIM4 v4.8.3 b4ld.c §1133-1154. Reusing the Vth-DVT
    # `Theta0` (T0_th = dvt1·Leff/lt1) makes the exponent too large for
    # long L, suppressing the cdsc contribution and pushing pyport `n`
    # ~12% high vs ngspice (z91m/O6 oracle convergence). The n branch
    # uses T0_th_n = 0.5·dvt1·Leff/lt1, giving a larger Theta0_n and
    # restoring the cdsc weighting.
    T0_th_n = 0.5 * dvt1 * Leff / lt1.clamp_min(1e-30)
    Theta0_n = _exp_threshold_branch(T0_th_n)
    tmp4 = (tmp2 + tmp3 * Theta0_n + cit) / coxe
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

```


=== FILE: temp_vgsteff_inputs.py (2981 chars) ===
```python
    DjctSatT = Js0_d * js_temp_factor_d

    # ---- Pre-computed Vth/Xdep quantities (b4temp.c §1322-1520) -----------
    # phi = 2·Vtm0·log(NDEP/ni) + phin   (BSIM4 v4.8.3, A.5.c fix 2026-05-01)
    # Prior code had `Vtm0·log(...) + phin + 0.4` — missing factor of 2 +
    # spurious +0.4 fudge. Three independent agents (A9 subagent, GPT-5,
    # Gemini-2.5-pro O5 packet) converged on this; produced ~60 mV uniform
    # Vth shift in compute_dc vs ngspice on isolated M2.
    ndep = max(model["ndep"], 1e10)   # safety: never log(0)
    phi = 2.0 * ctx.Vtm0 * math.log(ndep / max(ctx.ni, 1e-30)) + model["phin"]
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

```
