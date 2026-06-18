# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



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


=== FILE: dc.py (39248 chars) ===
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


=== FILE: model_card.py (9422 chars) ===
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

        # A.1.f fix: re-resolve `ref` defaults for parameters NOT explicitly
        # set by this card. BSIM4 v4.8.3 convention: source-side BTBT/GISL
        # parameters (agisl, bgisl, cgisl, egisl) mirror the drain-side
        # (agidl, bgidl, ...) when the card only specifies the agidl group.
        # Without this, Sebas's M2 card's nonzero agidl group leaves agisl
        # at pre-card defaults (agisl=0, bgisl=2.3e9, cgisl=0.5, egisl=0.8),
        # silently zeroing source-side BTBT at biases where it should fire.
        for name, info in PARAMS_META.items():
            d = info["default"]
            if isinstance(d, tuple) and d[0] == "ref" and name not in m._given:
                m._values[name] = m._values.get(d[1], 0.0)
        return m

    def __repr__(self) -> str:
        n = len(self._given)
        type_str = "NMOS" if self._values.get("type", 1) == 1 else "PMOS"
        return f"<BSIM4Model {type_str} given={n} of {len(PARAMS_META)}>"

```


=== FILE: model_card_data.py (108322 chars) ===
```python
"""Auto-generated by tools/bsim4_port/gen_model_card.py — do not edit by hand."""
from __future__ import annotations

# 880 canonical params, 4 alias names

PARAMS_META: dict[str, dict] = {
    "a0": {"c_field": "a0", "py_type": "float", "default": 1.0, "description": "Non-uniform depletion width effect coefficient."},
    "a1": {"c_field": "a1", "py_type": "float", "default": 0.0, "description": "Non-saturation effect coefficient"},
    "a2": {"c_field": "a2", "py_type": "float", "default": 1.0, "description": "Non-saturation effect coefficient"},
    "acde": {"c_field": "acde", "py_type": "float", "default": 1.0, "description": "Exponential coefficient for finite charge thickness"},
    "acnqsmod": {"c_field": "acnqsmod", "py_type": "int", "default": 0, "description": "AC NQS model selector"},
    "ados": {"c_field": "ados", "py_type": "float", "default": 1.0, "description": "Charge centroid parameter"},
    "af": {"c_field": "af", "py_type": "float", "default": 1.0, "description": "Flicker noise exponent"},
    "agidl": {"c_field": "agidl", "py_type": "float", "default": 0.0, "description": "Pre-exponential constant for GIDL"},
    "agisl": {"c_field": "agisl", "py_type": "float", "default": ("ref", "agidl"), "description": "Pre-exponential constant for GISL"},
    "ags": {"c_field": "ags", "py_type": "float", "default": 0.0, "description": "Gate bias  coefficient of Abulk."},
    "aigbacc": {"c_field": "aigbacc", "py_type": "float", "default": 0.0136, "description": "Parameter for Igb"},
    "aigbinv": {"c_field": "aigbinv", "py_type": "float", "default": 0.0111, "description": "Parameter for Igb"},
    "aigc": {"c_field": "aigc", "py_type": "float", "default": ("nmos_pmos", 1.36e-2, 9.80e-3), "description": "Parameter for Igc"},
    "aigd": {"c_field": "aigd", "py_type": "float", "default": ("nmos_pmos", 1.36e-2, 9.80e-3), "description": "Parameter for Igd"},
    "aigs": {"c_field": "aigs", "py_type": "float", "default": ("nmos_pmos", 1.36e-2, 9.80e-3), "description": "Parameter for Igs"},
    "aigsd": {"c_field": "aigsd", "py_type": "float", "default": 0.0, "description": "Parameter for Igs,d"},
    "alpha0": {"c_field": "alpha0", "py_type": "float", "default": 0.0, "description": "substrate current model parameter"},
    "alpha1": {"c_field": "alpha1", "py_type": "float", "default": 0.0, "description": "substrate current model parameter"},
    "at": {"c_field": "at", "py_type": "float", "default": 33000.0, "description": "Temperature coefficient of vsat"},
    "b0": {"c_field": "b0", "py_type": "float", "default": 0.0, "description": "Abulk narrow width parameter"},
    "b1": {"c_field": "b1", "py_type": "float", "default": 0.0, "description": "Abulk narrow width parameter"},
    "bdos": {"c_field": "bdos", "py_type": "float", "default": 1.0, "description": "Charge centroid parameter"},
    "beta0": {"c_field": "beta0", "py_type": "float", "default": 0.0, "description": "substrate current model parameter"},
    "bg0sub": {"c_field": "bg0sub", "py_type": "float", "default": 1.16, "description": "Band-gap of substrate at T=0K"},
    "bgidl": {"c_field": "bgidl", "py_type": "float", "default": 2300000000.0, "description": "Exponential constant for GIDL"},
    "bgisl": {"c_field": "bgisl", "py_type": "float", "default": ("ref", "bgidl"), "description": "Exponential constant for GISL"},
    "bigbacc": {"c_field": "bigbacc", "py_type": "float", "default": 0.00171, "description": "Parameter for Igb"},
    "bigbinv": {"c_field": "bigbinv", "py_type": "float", "default": 0.000949, "description": "Parameter for Igb"},
    "bigc": {"c_field": "bigc", "py_type": "float", "default": ("nmos_pmos", 1.71e-3, 7.59e-4), "description": "Parameter for Igc"},
    "bigd": {"c_field": "bigd", "py_type": "float", "default": ("nmos_pmos", 1.71e-3, 7.59e-4), "description": "Parameter for Igd"},
    "bigs": {"c_field": "bigs", "py_type": "float", "default": ("nmos_pmos", 1.71e-3, 7.59e-4), "description": "Parameter for Igs"},
    "bigsd": {"c_field": "bigsd", "py_type": "float", "default": 0.0, "description": "Parameter for Igs,d"},
    "binunit": {"c_field": "binunit", "py_type": "int", "default": 1, "description": "Bin  unit  selector"},
    "bvd": {"c_field": "bvd", "py_type": "float", "default": ("ref", "bvs"), "description": "Drain diode breakdown voltage"},
    "bvs": {"c_field": "bvs", "py_type": "float", "default": 10.0, "description": "Source diode breakdown voltage"},
    "capmod": {"c_field": "capmod", "py_type": "int", "default": 2, "description": "Capacitance model selector"},
    "cdsc": {"c_field": "cdsc", "py_type": "float", "default": 0.00024, "description": "Drain/Source and channel coupling capacitance"},
    "cdscb": {"c_field": "cdscb", "py_type": "float", "default": 0.0, "description": "Body-bias dependence of cdsc"},
    "cdscd": {"c_field": "cdscd", "py_type": "float", "default": 0.0, "description": "Drain-bias dependence of cdsc"},
    "cf": {"c_field": "cf", "py_type": "float", "default": 0.0, "description": "Fringe capacitance parameter"},
    "cgbo": {"c_field": "cgbo", "py_type": "float", "default": 0.0, "description": "Gate-bulk overlap capacitance per length"},
    "cgdl": {"c_field": "cgdl", "py_type": "float", "default": 0.0, "description": "New C-V model parameter"},
    "cgdo": {"c_field": "cgdo", "py_type": "float", "default": 0.0, "description": "Gate-drain overlap capacitance per width"},
    "cgidl": {"c_field": "cgidl", "py_type": "float", "default": 0.5, "description": "Parameter for body-bias dependence of GIDL"},
    "cgisl": {"c_field": "cgisl", "py_type": "float", "default": ("ref", "cgidl"), "description": "Parameter for body-bias dependence of GISL"},
    "cgsl": {"c_field": "cgsl", "py_type": "float", "default": 0.0, "description": "New C-V model parameter"},
    "cgso": {"c_field": "cgso", "py_type": "float", "default": 0.0, "description": "Gate-source overlap capacitance per width"},
    "cigbacc": {"c_field": "cigbacc", "py_type": "float", "default": 0.075, "description": "Parameter for Igb"},
    "cigbinv": {"c_field": "cigbinv", "py_type": "float", "default": 0.006, "description": "Parameter for Igb"},
    "cigc": {"c_field": "cigc", "py_type": "float", "default": ("nmos_pmos", 0.075, 0.03), "description": "Parameter for Igc"},
    "cigd": {"c_field": "cigd", "py_type": "float", "default": ("nmos_pmos", 0.075, 0.03), "description": "Parameter for Igd"},
    "cigs": {"c_field": "cigs", "py_type": "float", "default": ("nmos_pmos", 0.075, 0.03), "description": "Parameter for Igs"},
    "cigsd": {"c_field": "cigsd", "py_type": "float", "default": 0.0, "description": "Parameter for Igs,d"},
    "cit": {"c_field": "cit", "py_type": "float", "default": 0.0, "description": "Interface state capacitance"},
    "cjd": {"c_field": "cjd", "py_type": "float", "default": 0.0, "description": "Drain bottom junction capacitance per unit area"},
    "cjs": {"c_field": "cjs", "py_type": "float", "default": 0.0, "description": "Source bottom junction capacitance per unit area"},
    "cjswd": {"c_field": "cjswd", "py_type": "float", "default": 0.0, "description": "Drain sidewall junction capacitance per unit periphery"},
    "cjswgd": {"c_field": "cjswgd", "py_type": "float", "default": 0.0, "description": "Drain (gate side) sidewall junction capacitance per unit width"},
    "cjswgs": {"c_field": "cjswgs", "py_type": "float", "default": 0.0, "description": "Source (gate side) sidewall junction capacitance per unit width"},
    "cjsws": {"c_field": "cjsws", "py_type": "float", "default": 0.0, "description": "Source sidewall junction capacitance per unit periphery"},
    "ckappad": {"c_field": "ckappad", "py_type": "float", "default": ("ref", "ckappas"), "description": "D/G overlap C-V parameter"},
    "ckappas": {"c_field": "ckappas", "py_type": "float", "default": 0.6, "description": "S/G overlap C-V parameter "},
    "clc": {"c_field": "clc", "py_type": "float", "default": 1e-07, "description": "Vdsat parameter for C-V model"},
    "cle": {"c_field": "cle", "py_type": "float", "default": 0.6, "description": "Vdsat parameter for C-V model"},
    "cvchargemod": {"c_field": "cvchargemod", "py_type": "int", "default": 0, "description": "Capacitance Charge model selector"},
    "delta": {"c_field": "delta", "py_type": "float", "default": 0.01, "description": "Effective Vds parameter"},
    "diomod": {"c_field": "diomod", "py_type": "int", "default": 1, "description": "Diode IV model selector"},
    "dlc": {"c_field": "dlc", "py_type": "float", "default": ("ref", "lint"), "description": "Delta L for C-V model"},
    "dlcig": {"c_field": "dlcig", "py_type": "float", "default": ("ref", "lint"), "description": "Delta L for Ig model"},
    "dlcigd": {"c_field": "dlcigd", "py_type": "float", "default": 0.0, "description": "Delta L for Ig model drain side"},
    "dmcg": {"c_field": "dmcg", "py_type": "float", "default": 0.0, "description": "Distance of Mid-Contact to Gate edge"},
    "dmcgt": {"c_field": "dmcgt", "py_type": "float", "default": 0.0, "description": "Distance of Mid-Contact to Gate edge in Test structures"},
    "dmci": {"c_field": "dmci", "py_type": "float", "default": ("ref", "dmcg"), "description": "Distance of Mid-Contact to Isolation"},
    "dmdg": {"c_field": "dmdg", "py_type": "float", "default": 0.0, "description": "Distance of Mid-Diffusion to Gate edge"},
    "drout": {"c_field": "drout", "py_type": "float", "default": 0.56, "description": "DIBL coefficient of output resistance"},
    "dsub": {"c_field": "dsub", "py_type": "float", "default": ("ref", "drout"), "description": "DIBL coefficient in the subthreshold region"},
    "dtox": {"c_field": "dtox", "py_type": "float", "default": 0.0, "description": "Defined as (toxe - toxp) "},
    "dvt0": {"c_field": "dvt0", "py_type": "float", "default": 2.2, "description": "Short channel effect coeff. 0"},
    "dvt0w": {"c_field": "dvt0w", "py_type": "float", "default": 0.0, "description": "Narrow Width coeff. 0"},
    "dvt1": {"c_field": "dvt1", "py_type": "float", "default": 0.53, "description": "Short channel effect coeff. 1"},
    "dvt1w": {"c_field": "dvt1w", "py_type": "float", "default": 5300000.0, "description": "Narrow Width effect coeff. 1"},
    "dvt2": {"c_field": "dvt2", "py_type": "float", "default": -0.032, "description": "Short channel effect coeff. 2"},
    "dvt2w": {"c_field": "dvt2w", "py_type": "float", "default": -0.032, "description": "Narrow Width effect coeff. 2"},
    "dvtp0": {"c_field": "dvtp0", "py_type": "float", "default": 0.0, "description": "First parameter for Vth shift due to pocket"},
    "dvtp1": {"c_field": "dvtp1", "py_type": "float", "default": 0.0, "description": "Second parameter for Vth shift due to pocket"},
    "dvtp2": {"c_field": "dvtp2", "py_type": "float", "default": 0.0, "description": "3rd parameter for Vth shift due to pocket"},
    "dvtp3": {"c_field": "dvtp3", "py_type": "float", "default": 0.0, "description": "4th parameter for Vth shift due to pocket"},
    "dvtp4": {"c_field": "dvtp4", "py_type": "float", "default": 0.0, "description": "5th parameter for Vth shift due to pocket"},
    "dvtp5": {"c_field": "dvtp5", "py_type": "float", "default": 0.0, "description": "6th parameter for Vth shift due to pocket"},
    "dwb": {"c_field": "dwb", "py_type": "float", "default": 0.0, "description": "Width reduction parameter"},
    "dwc": {"c_field": "dwc", "py_type": "float", "default": ("ref", "wint"), "description": "Delta W for C-V model"},
    "dwg": {"c_field": "dwg", "py_type": "float", "default": 0.0, "description": "Width reduction parameter"},
    "dwj": {"c_field": "dwj", "py_type": "float", "default": ("ref", "dwc"), "description": "Delta W for S/D junctions"},
    "easub": {"c_field": "easub", "py_type": "float", "default": 4.05, "description": "Electron affinity of substrate"},
    "ef": {"c_field": "ef", "py_type": "float", "default": 1.0, "description": "Flicker noise frequency exponent"},
    "egidl": {"c_field": "egidl", "py_type": "float", "default": 0.8, "description": "Fitting parameter for Bandbending"},
    "egisl": {"c_field": "egisl", "py_type": "float", "default": ("ref", "egidl"), "description": "Fitting parameter for Bandbending"},
    "eigbinv": {"c_field": "eigbinv", "py_type": "float", "default": 1.1, "description": "Parameter for the Si bandgap for Igbinv"},
    "em": {"c_field": "em", "py_type": "float", "default": 41000000.0, "description": "Flicker noise parameter"},
    "eot": {"c_field": "eot", "py_type": "float", "default": 1.5e-09, "description": "Equivalent gate oxide thickness in meters"},
    "epsrgate": {"c_field": "epsrgate", "py_type": "float", "default": 11.7, "description": "Dielectric constant of gate relative to vacuum"},
    "epsrox": {"c_field": "epsrox", "py_type": "float", "default": 3.9, "description": "Dielectric constant of the gate oxide relative to vacuum"},
    "epsrsub": {"c_field": "epsrsub", "py_type": "float", "default": 11.7, "description": "Dielectric constant of substrate relative to vacuum"},
    "eta0": {"c_field": "eta0", "py_type": "float", "default": 0.08, "description": "Subthreshold region DIBL coefficient"},
    "etab": {"c_field": "etab", "py_type": "float", "default": -0.07, "description": "Subthreshold region DIBL coefficient"},
    "eu": {"c_field": "eu", "py_type": "float", "default": ("nmos_pmos", 1.67, 1.0), "description": "Mobility exponent"},
    "fgidl": {"c_field": "fgidl", "py_type": "float", "default": 0.0, "description": "GIDL vb parameter"},
    "fgisl": {"c_field": "fgisl", "py_type": "float", "default": ("ref", "fgidl"), "description": "GISL vb parameter"},
    "fnoimod": {"c_field": "fnoimod", "py_type": "int", "default": 1, "description": "Flicker noise model selector"},
    "fprout": {"c_field": "fprout", "py_type": "float", "default": 0.0, "description": "Rout degradation coefficient for pocket devices"},
    "gamma1": {"c_field": "gamma1", "py_type": "float", "default": 0.0, "description": "Vth body coefficient"},
    "gamma2": {"c_field": "gamma2", "py_type": "float", "default": 0.0, "description": "Vth body coefficient"},
    "gbmin": {"c_field": "gbmin", "py_type": "float", "default": 1e-12, "description": "Minimum body conductance"},
    "geomod": {"c_field": "geomod", "py_type": "int", "default": 0, "description": "Geometry dependent parasitics model selector"},
    "gidlclamp": {"c_field": "gidlclamp", "py_type": "float", "default": -1e-05, "description": "gidl clamp value"},
    "gidlmod": {"c_field": "gidlmod", "py_type": "int", "default": 0, "description": "parameter for GIDL selector"},
    "idovvds": {"c_field": "idovvds", "py_type": "float", "default": 0.0, "description": "noise clamping limit parameter"},
    "igbmod": {"c_field": "igbmod", "py_type": "int", "default": 0, "description": "Gate-to-body Ig model selector"},
    "igcmod": {"c_field": "igcmod", "py_type": "int", "default": 0, "description": "Gate-to-channel Ig model selector"},
    "ijthdfwd": {"c_field": "ijthdfwd", "py_type": "float", "default": ("ref", "ijthsfwd"), "description": "Forward drain diode forward limiting current"},
    "ijthdrev": {"c_field": "ijthdrev", "py_type": "float", "default": ("ref", "ijthsrev"), "description": "Reverse drain diode forward limiting current"},
    "ijthsfwd": {"c_field": "ijthsfwd", "py_type": "float", "default": 0.1, "description": "Forward source diode forward limiting current"},
    "ijthsrev": {"c_field": "ijthsrev", "py_type": "float", "default": 0.1, "description": "Reverse source diode forward limiting current"},
    "jsd": {"c_field": "jsd", "py_type": "float", "default": 0.0, "description": "Bottom drain junction reverse saturation current density"},
    "jss": {"c_field": "jss", "py_type": "float", "default": 0.0, "description": "Bottom source junction reverse saturation current density"},
    "jswd": {"c_field": "jswd", "py_type": "float", "default": 0.0, "description": "Isolation edge sidewall drain junction reverse saturation current density"},
    "jswgd": {"c_field": "jswgd", "py_type": "float", "default": 0.0, "description": "Gate edge drain junction reverse saturation current density"},
    "jswgs": {"c_field": "jswgs", "py_type": "float", "default": 0.0, "description": "Gate edge source junction reverse saturation current density"},
    "jsws": {"c_field": "jsws", "py_type": "float", "default": 0.0, "description": "Isolation edge sidewall source junction reverse saturation current density"},
    "jtsd": {"c_field": "jtsd", "py_type": "float", "default": ("ref", "jtss"), "description": "Drain bottom trap-assisted saturation current density"},
    "jtss": {"c_field": "jtss", "py_type": "float", "default": 0.0, "description": "Source bottom trap-assisted saturation current density"},
    "jtsswd": {"c_field": "jtsswd", "py_type": "float", "default": ("ref", "jtssws"), "description": "Drain STI sidewall trap-assisted saturation current density"},
    "jtsswgd": {"c_field": "jtsswgd", "py_type": "float", "default": ("ref", "jtsswgs"), "description": "Drain gate-edge sidewall trap-assisted saturation current density"},
    "jtsswgs": {"c_field": "jtsswgs", "py_type": "float", "default": 0.0, "description": "Source gate-edge sidewall trap-assisted saturation current density"},
    "jtssws": {"c_field": "jtssws", "py_type": "float", "default": 0.0, "description": "Source STI sidewall trap-assisted saturation current density"},
    "jtweff": {"c_field": "jtweff", "py_type": "float", "default": 0.0, "description": "TAT current width dependance"},
    "k1": {"c_field": "k1", "py_type": "float", "default": 0.0, "description": "Bulk effect coefficient 1"},
    "k2": {"c_field": "k2", "py_type": "float", "default": 0.0, "description": "Bulk effect coefficient 2"},
    "k2we": {"c_field": "k2we", "py_type": "float", "default": 0.0, "description": " K2 shift factor for well proximity effect "},
    "k3": {"c_field": "k3", "py_type": "float", "default": 80.0, "description": "Narrow width effect coefficient"},
    "k3b": {"c_field": "k3b", "py_type": "float", "default": 0.0, "description": "Body effect coefficient of k3"},
    "keta": {"c_field": "keta", "py_type": "float", "default": -0.047, "description": "Body-bias coefficient of non-uniform depletion width effect."},
    "ketac": {"c_field": "ketac", "py_type": "float", "default": ("ref", "keta"), "description": "Body-bias coefficient of non-uniform depletion width effect in dynamic evaluatio"},
    "kf": {"c_field": "kf", "py_type": "float", "default": 0.0, "description": "Flicker noise coefficient"},
    "kgidl": {"c_field": "kgidl", "py_type": "float", "default": 0.0, "description": "GIDL vb parameter"},
    "kgisl": {"c_field": "kgisl", "py_type": "float", "default": ("ref", "kgidl"), "description": "GISL vb parameter"},
    "kt1": {"c_field": "kt1", "py_type": "float", "default": -0.11, "description": "Temperature coefficient of Vth"},
    "kt1l": {"c_field": "kt1l", "py_type": "float", "default": 0.0, "description": "Temperature coefficient of Vth"},
    "kt2": {"c_field": "kt2", "py_type": "float", "default": 0.022, "description": "Body-coefficient of kt1"},
    "ku0": {"c_field": "ku0", "py_type": "float", "default": 0.0, "description": "Mobility degradation/enhancement coefficient for LOD"},
    "ku0we": {"c_field": "ku0we", "py_type": "float", "default": 0.0, "description": " Mobility degradation factor for well proximity effect "},
    "kvsat": {"c_field": "kvsat", "py_type": "float", "default": 0.0, "description": "Saturation velocity degradation/enhancement parameter for LOD"},
    "kvth0": {"c_field": "kvth0", "py_type": "float", "default": 0.0, "description": "Threshold degradation/enhancement parameter for LOD"},
    "kvth0we": {"c_field": "kvth0we", "py_type": "float", "default": 0.0, "description": "Threshold shift factor for well proximity effect"},
    "la0": {"c_field": "la0", "py_type": "float", "default": 0.0, "description": "Length dependence of a0"},
    "la1": {"c_field": "la1", "py_type": "float", "default": 0.0, "description": "Length dependence of a1"},
    "la2": {"c_field": "la2", "py_type": "float", "default": 0.0, "description": "Length dependence of a2"},
    "lacde": {"c_field": "lacde", "py_type": "float", "default": 0.0, "description": "Length dependence of acde"},
    "lagidl": {"c_field": "lagidl", "py_type": "float", "default": 0.0, "description": "Length dependence of agidl"},
    "lagisl": {"c_field": "lagisl", "py_type": "float", "default": ("ref", "lagidl"), "description": "Length dependence of agisl"},
    "lags": {"c_field": "lags", "py_type": "float", "default": 0.0, "description": "Length dependence of ags"},
    "laigbacc": {"c_field": "laigbacc", "py_type": "float", "default": 0.0, "description": "Length dependence of aigbacc"},
    "laigbinv": {"c_field": "laigbinv", "py_type": "float", "default": 0.0, "description": "Length dependence of aigbinv"},
    "laigc": {"c_field": "laigc", "py_type": "float", "default": 0.0, "description": "Length dependence of aigc"},
    "laigd": {"c_field": "laigd", "py_type": "float", "default": 0.0, "description": "Length dependence of aigd"},
    "laigs": {"c_field": "laigs", "py_type": "float", "default": 0.0, "description": "Length dependence of aigs"},
    "laigsd": {"c_field": "laigsd", "py_type": "float", "default": 0.0, "description": "Length dependence of aigsd"},
    "lalpha0": {"c_field": "lalpha0", "py_type": "float", "default": 0.0, "description": "Length dependence of alpha0"},
    "lalpha1": {"c_field": "lalpha1", "py_type": "float", "default": 0.0, "description": "Length dependence of alpha1"},
    "lambda": {"c_field": "lambda", "py_type": "float", "default": 0.0, "description": " Velocity overshoot parameter"},
    "lat": {"c_field": "lat", "py_type": "float", "default": 0.0, "description": "Length dependence of at"},
    "lb0": {"c_field": "lb0", "py_type": "float", "default": 0.0, "description": "Length dependence of b0"},
    "lb1": {"c_field": "lb1", "py_type": "float", "default": 0.0, "description": "Length dependence of b1"},
    "lbeta0": {"c_field": "lbeta0", "py_type": "float", "default": 0.0, "description": "Length dependence of beta0"},
    "lbgidl": {"c_field": "lbgidl", "py_type": "float", "default": 0.0, "description": "Length dependence of bgidl"},
    "lbgisl": {"c_field": "lbgisl", "py_type": "float", "default": ("ref", "lbgidl"), "description": "Length dependence of bgisl"},
    "lbigbacc": {"c_field": "lbigbacc", "py_type": "float", "default": 0.0, "description": "Length dependence of bigbacc"},
    "lbigbinv": {"c_field": "lbigbinv", "py_type": "float", "default": 0.0, "description": "Length dependence of bigbinv"},
    "lbigc": {"c_field": "lbigc", "py_type": "float", "default": 0.0, "description": "Length dependence of bigc"},
    "lbigd": {"c_field": "lbigd", "py_type": "float", "default": 0.0, "description": "Length dependence of bigd"},
    "lbigs": {"c_field": "lbigs", "py_type": "float", "default": 0.0, "description": "Length dependence of bigs"},
    "lbigsd": {"c_field": "lbigsd", "py_type": "float", "default": 0.0, "description": "Length dependence of bigsd"},
    "lc": {"c_field": "lc", "py_type": "float", "default": 5e-09, "description": " back scattering parameter"},
    "lcdsc": {"c_field": "lcdsc", "py_type": "float", "default": 0.0, "description": "Length dependence of cdsc"},
    "lcdscb": {"c_field": "lcdscb", "py_type": "float", "default": 0.0, "description": "Length dependence of cdscb"},
    "lcdscd": {"c_field": "lcdscd", "py_type": "float", "default": 0.0, "description": "Length dependence of cdscd"},
    "lcf": {"c_field": "lcf", "py_type": "float", "default": 0.0, "description": "Length dependence of cf"},
    "lcgdl": {"c_field": "lcgdl", "py_type": "float", "default": 0.0, "description": "Length dependence of cgdl"},
    "lcgidl": {"c_field": "lcgidl", "py_type": "float", "default": 0.0, "description": "Length dependence of cgidl"},
    "lcgisl": {"c_field": "lcgisl", "py_type": "float", "default": ("ref", "lcgidl"), "description": "Length dependence of cgisl"},
    "lcgsl": {"c_field": "lcgsl", "py_type": "float", "default": 0.0, "description": "Length dependence of cgsl"},
    "lcigbacc": {"c_field": "lcigbacc", "py_type": "float", "default": 0.0, "description": "Length dependence of cigbacc"},
    "lcigbinv": {"c_field": "lcigbinv", "py_type": "float", "default": 0.0, "description": "Length dependence of cigbinv"},
    "lcigc": {"c_field": "lcigc", "py_type": "float", "default": 0.0, "description": "Length dependence of cigc"},
    "lcigd": {"c_field": "lcigd", "py_type": "float", "default": 0.0, "description": "Length dependence of cigd"},
    "lcigs": {"c_field": "lcigs", "py_type": "float", "default": 0.0, "description": "Length dependence of cigs"},
    "lcigsd": {"c_field": "lcigsd", "py_type": "float", "default": 0.0, "description": "Length dependence of cigsd"},
    "lcit": {"c_field": "lcit", "py_type": "float", "default": 0.0, "description": "Length dependence of cit"},
    "lckappad": {"c_field": "lckappad", "py_type": "float", "default": 0.0, "description": "Length dependence of ckappad"},
    "lckappas": {"c_field": "lckappas", "py_type": "float", "default": 0.0, "description": "Length dependence of ckappas"},
    "lclc": {"c_field": "lclc", "py_type": "float", "default": 0.0, "description": "Length dependence of clc"},
    "lcle": {"c_field": "lcle", "py_type": "float", "default": 0.0, "description": "Length dependence of cle"},
    "ldelta": {"c_field": "ldelta", "py_type": "float", "default": 0.0, "description": "Length dependence of delta"},
    "ldrout": {"c_field": "ldrout", "py_type": "float", "default": 0.0, "description": "Length dependence of drout"},
    "ldsub": {"c_field": "ldsub", "py_type": "float", "default": 0.0, "description": "Length dependence of dsub"},
    "ldvt0": {"c_field": "ldvt0", "py_type": "float", "default": 0.0, "description": "Length dependence of dvt0"},
    "ldvt0w": {"c_field": "ldvt0w", "py_type": "float", "default": 0.0, "description": "Length dependence of dvt0w"},
    "ldvt1": {"c_field": "ldvt1", "py_type": "float", "default": 0.0, "description": "Length dependence of dvt1"},
    "ldvt1w": {"c_field": "ldvt1w", "py_type": "float", "default": 0.0, "description": "Length dependence of dvt1w"},
    "ldvt2": {"c_field": "ldvt2", "py_type": "float", "default": 0.0, "description": "Length dependence of dvt2"},
    "ldvt2w": {"c_field": "ldvt2w", "py_type": "float", "default": 0.0, "description": "Length dependence of dvt2w"},
    "ldvtp0": {"c_field": "ldvtp0", "py_type": "float", "default": 0.0, "description": "Length dependence of dvtp0"},
    "ldvtp1": {"c_field": "ldvtp1", "py_type": "float", "default": 0.0, "description": "Length dependence of dvtp1"},
    "ldvtp2": {"c_field": "ldvtp2", "py_type": "float", "default": 0.0, "description": "Length dependence of dvtp2"},
    "ldvtp3": {"c_field": "ldvtp3", "py_type": "float", "default": 0.0, "description": "Length dependence of dvtp3"},
    "ldvtp4": {"c_field": "ldvtp4", "py_type": "float", "default": 0.0, "description": "Length dependence of dvtp4"},
    "ldvtp5": {"c_field": "ldvtp5", "py_type": "float", "default": 0.0, "description": "Length dependence of dvtp5"},
    "ldwb": {"c_field": "ldwb", "py_type": "float", "default": 0.0, "description": "Length dependence of dwb"},
    "ldwg": {"c_field": "ldwg", "py_type": "float", "default": 0.0, "description": "Length dependence of dwg"},
    "leffeot": {"c_field": "leffeot", "py_type": "float", "default": 1.0, "description": " Effective length for extraction of EOT"},
    "legidl": {"c_field": "legidl", "py_type": "float", "default": 0.0, "description": "Length dependence of egidl"},
    "legisl": {"c_field": "legisl", "py_type": "float", "default": ("ref", "legidl"), "description": "Length dependence of egisl"},
    "leigbinv": {"c_field": "leigbinv", "py_type": "float", "default": 0.0, "description": "Length dependence for eigbinv"},
    "leta0": {"c_field": "leta0", "py_type": "float", "default": 0.0, "description": "Length dependence of eta0"},
    "letab": {"c_field": "letab", "py_type": "float", "default": -0.0, "description": "Length dependence of etab"},
    "leu": {"c_field": "leu", "py_type": "float", "default": 0.0, "description": " Length dependence of eu"},
    "lfgidl": {"c_field": "lfgidl", "py_type": "float", "default": 0.0, "description": "Length dependence of fgidl"},
    "lfgisl": {"c_field": "lfgisl", "py_type": "float", "default": ("ref", "lfgidl"), "description": "Length dependence of fgisl"},
    "lfprout": {"c_field": "lfprout", "py_type": "float", "default": 0.0, "description": "Length dependence of pdiblcb"},
    "lgamma1": {"c_field": "lgamma1", "py_type": "float", "default": 0.0, "description": "Length dependence of gamma1"},
    "lgamma2": {"c_field": "lgamma2", "py_type": "float", "default": 0.0, "description": "Length dependence of gamma2"},
    "lint": {"c_field": "lint", "py_type": "float", "default": 0.0, "description": "Length reduction parameter"},
    "lintnoi": {"c_field": "lintnoi", "py_type": "float", "default": 0.0, "description": "lint offset for noise calculation"},
    "lk1": {"c_field": "lk1", "py_type": "float", "default": 0.0, "description": "Length dependence of k1"},
    "lk2": {"c_field": "lk2", "py_type": "float", "default": 0.0, "description": "Length dependence of k2"},
    "lk2we": {"c_field": "lk2we", "py_type": "float", "default": 0.0, "description": " Length dependence of k2we "},
    "lk3": {"c_field": "lk3", "py_type": "float", "default": 0.0, "description": "Length dependence of k3"},
    "lk3b": {"c_field": "lk3b", "py_type": "float", "default": 0.0, "description": "Length dependence of k3b"},
    "lketa": {"c_field": "lketa", "py_type": "float", "default": 0.0, "description": "Length dependence of keta"},
    "lketac": {"c_field": "lketac", "py_type": "float", "default": ("ref", "lketa"), "description": "Length dependence of ketac"},
    "lkgidl": {"c_field": "lkgidl", "py_type": "float", "default": 0.0, "description": "Length dependence of kgidl"},
    "lkgisl": {"c_field": "lkgisl", "py_type": "float", "default": ("ref", "lkgidl"), "description": "Length dependence of kgisl"},
    "lkt1": {"c_field": "lkt1", "py_type": "float", "default": 0.0, "description": "Length dependence of kt1"},
    "lkt1l": {"c_field": "lkt1l", "py_type": "float", "default": 0.0, "description": "Length dependence of kt1l"},
    "lkt2": {"c_field": "lkt2", "py_type": "float", "default": 0.0, "description": "Length dependence of kt2"},
    "lku0": {"c_field": "lku0", "py_type": "float", "default": 0.0, "description": "Length dependence of ku0"},
    "lku0we": {"c_field": "lku0we", "py_type": "float", "default": 0.0, "description": " Length dependence of ku0we "},
    "lkvth0": {"c_field": "lkvth0", "py_type": "float", "default": 0.0, "description": "Length dependence of kvth0"},
    "lkvth0we": {"c_field": "lkvth0we", "py_type": "float", "default": 0.0, "description": "Length dependence of kvth0we"},
    "ll": {"c_field": "ll", "py_type": "float", "default": 0.0, "description": "Length reduction parameter"},
    "llambda": {"c_field": "llambda", "py_type": "float", "default": 0.0, "description": "Length dependence of lambda"},
    "llc": {"c_field": "llc", "py_type": "float", "default": ("ref", "ll"), "description": "Length reduction parameter for CV"},
    "lln": {"c_field": "lln", "py_type": "float", "default": 1.0, "description": "Length reduction parameter"},
    "llodku0": {"c_field": "llodku0", "py_type": "float", "default": 0.0, "description": "Length parameter for u0 LOD effect"},
    "llodvth": {"c_field": "llodvth", "py_type": "float", "default": 0.0, "description": "Length parameter for vth LOD effect"},
    "llp": {"c_field": "llp", "py_type": "float", "default": 0.0, "description": "Length dependence of lp"},
    "llpe0": {"c_field": "llpe0", "py_type": "float", "default": 0.0, "description": "Length dependence of lpe0"},
    "llpeb": {"c_field": "llpeb", "py_type": "float", "default": 0.0, "description": "Length dependence of lpeb"},
    "lmax": {"c_field": "lmax", "py_type": "float", "default": 1.0, "description": "Maximum length for the model"},
    "lmin": {"c_field": "lmin", "py_type": "float", "default": 0.0, "description": "Minimum length for the model"},
    "lminv": {"c_field": "lminv", "py_type": "float", "default": 0.0, "description": "Length dependence of minv"},
    "lminvcv": {"c_field": "lminvcv", "py_type": "float", "default": 0.0, "description": "Length dependence of minvcv"},
    "lmoin": {"c_field": "lmoin", "py_type": "float", "default": 0.0, "description": "Length dependence of moin"},
    "lndep": {"c_field": "lndep", "py_type": "float", "default": 0.0, "description": "Length dependence of ndep"},
    "lnfactor": {"c_field": "lnfactor", "py_type": "float", "default": 0.0, "description": "Length dependence of nfactor"},
    "lngate": {"c_field": "lngate", "py_type": "float", "default": 0.0, "description": "Length dependence of ngate"},
    "lnigbacc": {"c_field": "lnigbacc", "py_type": "float", "default": 0.0, "description": "Length dependence of nigbacc"},
    "lnigbinv": {"c_field": "lnigbinv", "py_type": "float", "default": 0.0, "description": "Length dependence of nigbinv"},
    "lnigc": {"c_field": "lnigc", "py_type": "float", "default": 0.0, "description": "Length dependence of nigc"},
    "lnoff": {"c_field": "lnoff", "py_type": "float", "default": 0.0, "description": "Length dependence of noff"},
    "lnsd": {"c_field": "lnsd", "py_type": "float", "default": 0.0, "description": "Length dependence of nsd"},
    "lnsub": {"c_field": "lnsub", "py_type": "float", "default": 0.0, "description": "Length dependence of nsub"},
    "lntox": {"c_field": "lntox", "py_type": "float", "default": 0.0, "description": "Length dependence of ntox"},
    "lodeta0": {"c_field": "lodeta0", "py_type": "float", "default": 1.0, "description": "eta0 shift modification factor for stress effect"},
    "lodk2": {"c_field": "lodk2", "py_type": "float", "default": 1.0, "description": "K2 shift modification factor for stress effect"},
    "lp": {"c_field": "lp", "py_type": "float", "default": 1e-08, "description": "Channel length exponential factor of mobility"},
    "lpclm": {"c_field": "lpclm", "py_type": "float", "default": 0.0, "description": "Length dependence of pclm"},
    "lpdiblc1": {"c_field": "lpdiblc1", "py_type": "float", "default": 0.0, "description": "Length dependence of pdiblc1"},
    "lpdiblc2": {"c_field": "lpdiblc2", "py_type": "float", "default": 0.0, "description": "Length dependence of pdiblc2"},
    "lpdiblcb": {"c_field": "lpdiblcb", "py_type": "float", "default": 0.0, "description": "Length dependence of pdiblcb"},
    "lpdits": {"c_field": "lpdits", "py_type": "float", "default": 0.0, "description": "Length dependence of pdits"},
    "lpditsd": {"c_field": "lpditsd", "py_type": "float", "default": 0.0, "description": "Length dependence of pditsd"},
    "lpe0": {"c_field": "lpe0", "py_type": "float", "default": 1.74e-07, "description": "Equivalent length of pocket region at zero bias"},
    "lpeb": {"c_field": "lpeb", "py_type": "float", "default": 0.0, "description": "Equivalent length of pocket region accounting for body bias"},
    "lphin": {"c_field": "lphin", "py_type": "float", "default": 0.0, "description": "Length dependence of phin"},
    "lpigcd": {"c_field": "lpigcd", "py_type": "float", "default": 0.0, "description": "Length dependence for pigcd"},
    "lpoxedge": {"c_field": "lpoxedge", "py_type": "float", "default": 0.0, "description": "Length dependence for poxedge"},
    "lprt": {"c_field": "lprt", "py_type": "float", "default": 0.0, "description": "Length dependence of prt "},
    "lprwb": {"c_field": "lprwb", "py_type": "float", "default": 0.0, "description": "Length dependence of prwb "},
    "lprwg": {"c_field": "lprwg", "py_type": "float", "default": 0.0, "description": "Length dependence of prwg "},
    "lpscbe1": {"c_field": "lpscbe1", "py_type": "float", "default": 0.0, "description": "Length dependence of pscbe1"},
    "lpscbe2": {"c_field": "lpscbe2", "py_type": "float", "default": 0.0, "description": "Length dependence of pscbe2"},
    "lpvag": {"c_field": "lpvag", "py_type": "float", "default": 0.0, "description": "Length dependence of pvag"},
    "lrdsw": {"c_field": "lrdsw", "py_type": "float", "default": 0.0, "description": "Length dependence of rdsw "},
    "lrdw": {"c_field": "lrdw", "py_type": "float", "default": 0.0, "description": "Length dependence of rdw"},
    "lrgidl": {"c_field": "lrgidl", "py_type": "float", "default": 0.0, "description": "Length dependence of rgidl"},
    "lrgisl": {"c_field": "lrgisl", "py_type": "float", "default": ("ref", "lrgidl"), "description": "Length dependence of rgisl"},
    "lrsw": {"c_field": "lrsw", "py_type": "float", "default": 0.0, "description": "Length dependence of rsw"},
    "lteta0": {"c_field": "lteta0", "py_type": "float", "default": 0.0, "description": "Length dependence of teta0"},
    "ltnfactor": {"c_field": "ltnfactor", "py_type": "float", "default": 0.0, "description": "Length dependence of tnfactor"},
    "ltvfbsdoff": {"c_field": "ltvfbsdoff", "py_type": "float", "default": 0.0, "description": "Length dependence of tvfbsdoff"},
    "ltvoff": {"c_field": "ltvoff", "py_type": "float", "default": 0.0, "description": "Length dependence of tvoff"},
    "ltvoffcv": {"c_field": "ltvoffcv", "py_type": "float", "default": 0.0, "description": "Length dependence of tvoffcv"},
    "lu0": {"c_field": "lu0", "py_type": "float", "default": 0.0, "description": "Length dependence of u0"},
    "lua": {"c_field": "lua", "py_type": "float", "default": 0.0, "description": "Length dependence of ua"},
    "lua1": {"c_field": "lua1", "py_type": "float", "default": 0.0, "description": "Length dependence of ua1"},
    "lub": {"c_field": "lub", "py_type": "float", "default": 0.0, "description": "Length dependence of ub"},
    "lub1": {"c_field": "lub1", "py_type": "float", "default": 0.0, "description": "Length dependence of ub1"},
    "luc": {"c_field": "luc", "py_type": "float", "default": 0.0, "description": "Length dependence of uc"},
    "luc1": {"c_field": "luc1", "py_type": "float", "default": 0.0, "description": "Length dependence of uc1"},
    "lucs": {"c_field": "lucs", "py_type": "float", "default": 0.0, "description": "Length dependence of lucs"},
    "lucste": {"c_field": "lucste", "py_type": "float", "default": 0.0, "description": "Length dependence of ucste"},
    "lud": {"c_field": "lud", "py_type": "float", "default": 0.0, "description": "Length dependence of ud"},
    "lud1": {"c_field": "lud1", "py_type": "float", "default": 0.0, "description": "Length dependence of ud1"},
    "lup": {"c_field": "lup", "py_type": "float", "default": 0.0, "description": "Length dependence of up"},
    "lute": {"c_field": "lute", "py_type": "float", "default": 0.0, "description": "Length dependence of ute"},
    "lvbm": {"c_field": "lvbm", "py_type": "float", "default": 0.0, "description": "Length dependence of vbm"},
    "lvbx": {"c_field": "lvbx", "py_type": "float", "default": 0.0, "description": "Length dependence of vbx"},
    "lvfb": {"c_field": "lvfb", "py_type": "float", "default": 0.0, "description": "Length dependence of vfb"},
    "lvfbcv": {"c_field": "lvfbcv", "py_type": "float", "default": 0.0, "description": "Length dependence of vfbcv"},
    "lvfbsdoff": {"c_field": "lvfbsdoff", "py_type": "float", "default": 0.0, "description": "Length dependence of vfbsdoff"},
    "lvoff": {"c_field": "lvoff", "py_type": "float", "default": 0.0, "description": "Length dependence of voff"},
    "lvoffcv": {"c_field": "lvoffcv", "py_type": "float", "default": 0.0, "description": "Length dependence of voffcv"},
    "lvsat": {"c_field": "lvsat", "py_type": "float", "default": 0.0, "description": "Length dependence of vsat"},
    "lvth0": {"c_field": "lvth0", "py_type": "float", "default": 0.0, "description": "Length dependence of vto"},
    "lvtl": {"c_field": "lvtl", "py_type": "float", "default": 0.0, "description": " Length dependence of vtl"},
    "lw": {"c_field": "lw", "py_type": "float", "default": 0.0, "description": "Length reduction parameter"},
    "lw0": {"c_field": "lw0", "py_type": "float", "default": 0.0, "description": "Length dependence of w0"},
    "lwc": {"c_field": "lwc", "py_type": "float", "default": ("ref", "lw"), "description": "Length reduction parameter for CV"},
    "lwl": {"c_field": "lwl", "py_type": "float", "default": 0.0, "description": "Length reduction parameter"},
    "lwlc": {"c_field": "lwlc", "py_type": "float", "default": ("ref", "lwl"), "description": "Length reduction parameter for CV"},
    "lwn": {"c_field": "lwn", "py_type": "float", "default": 1.0, "description": "Length reduction parameter"},
    "lwr": {"c_field": "lwr", "py_type": "float", "default": 0.0, "description": "Length dependence of wr"},
    "lxj": {"c_field": "lxj", "py_type": "float", "default": 0.0, "description": "Length dependence of xj"},
    "lxn": {"c_field": "lxn", "py_type": "float", "default": 0.0, "description": " Length dependence of xn"},
    "lxrcrg1": {"c_field": "lxrcrg1", "py_type": "float", "default": 0.0, "description": "Length dependence of xrcrg1"},
    "lxrcrg2": {"c_field": "lxrcrg2", "py_type": "float", "default": 0.0, "description": "Length dependence of xrcrg2"},
    "lxt": {"c_field": "lxt", "py_type": "float", "default": 0.0, "description": "Length dependence of xt"},
    "minv": {"c_field": "minv", "py_type": "float", "default": 0.0, "description": "Fitting parameter for moderate inversion in Vgsteff"},
    "minvcv": {"c_field": "minvcv", "py_type": "float", "default": 0.0, "description": "Fitting parameter for moderate inversion in Vgsteffcv"},
    "mjd": {"c_field": "mjd", "py_type": "float", "default": 0.0, "description": "Drain bottom junction capacitance grading coefficient"},
    "mjs": {"c_field": "mjs", "py_type": "float", "default": 0.0, "description": "Source bottom junction capacitance grading coefficient"},
    "mjswd": {"c_field": "mjswd", "py_type": "float", "default": 0.0, "description": "Drain sidewall junction capacitance grading coefficient"},
    "mjswgd": {"c_field": "mjswgd", "py_type": "float", "default": 0.0, "description": "Drain (gate side) sidewall junction capacitance grading coefficient"},
    "mjswgs": {"c_field": "mjswgs", "py_type": "float", "default": 0.0, "description": "Source (gate side) sidewall junction capacitance grading coefficient"},
    "mjsws": {"c_field": "mjsws", "py_type": "float", "default": 0.0, "description": "Source sidewall junction capacitance grading coefficient"},
    "mobmod": {"c_field": "mobmod", "py_type": "int", "default": 0, "description": "Mobility model selector"},
    "moin": {"c_field": "moin", "py_type": "float", "default": 15.0, "description": "Coefficient for gate-bias dependent surface potential"},
    "mtrlcompatmod": {"c_field": "mtrlcompatmod", "py_type": "int", "default": 0, "description": "New Material Mod backward compatibility selector"},
    "mtrlmod": {"c_field": "mtrlmod", "py_type": "int", "default": 0, "description": "parameter for non-silicon substrate or metal gate selector"},
    "ndep": {"c_field": "ndep", "py_type": "float", "default": 1.7e+17, "description": "Channel doping concentration at the depletion edge"},
    "nfactor": {"c_field": "nfactor", "py_type": "float", "default": 1.0, "description": "Subthreshold swing Coefficient"},
    "ngate": {"c_field": "ngate", "py_type": "float", "default": 0.0, "description": "Poly-gate doping concentration"},
    "ngcon": {"c_field": "ngcon", "py_type": "float", "default": 1.0, "description": "Number of gate contacts"},
    "ni0sub": {"c_field": "ni0sub", "py_type": "float", "default": 14500000000.0, "description": "Intrinsic carrier concentration of substrate at 300.15K"},
    "nigbacc": {"c_field": "nigbacc", "py_type": "float", "default": 1.0, "description": "Parameter for Igbacc slope"},
    "nigbinv": {"c_field": "nigbinv", "py_type": "float", "default": 3.0, "description": "Parameter for Igbinv slope"},
    "nigc": {"c_field": "nigc", "py_type": "float", "default": 1.0, "description": "Parameter for Igc slope"},
    "njd": {"c_field": "njd", "py_type": "float", "default": 0.0, "description": "Drain junction emission coefficient"},
    "njs": {"c_field": "njs", "py_type": "float", "default": 0.0, "description": "Source junction emission coefficient"},
    "njts": {"c_field": "njts", "py_type": "float", "default": 20.0, "description": "Non-ideality factor for bottom junction"},
    "njtsd": {"c_field": "njtsd", "py_type": "float", "default": 0.0, "description": "Non-ideality factor for bottom junction drain side"},
    "njtssw": {"c_field": "njtssw", "py_type": "float", "default": 20.0, "description": "Non-ideality factor for STI sidewall junction"},
    "njtsswd": {"c_field": "njtsswd", "py_type": "float", "default": 0.0, "description": "Non-ideality factor for STI sidewall junction drain side"},
    "njtsswg": {"c_field": "njtsswg", "py_type": "float", "default": 20.0, "description": "Non-ideality factor for gate-edge sidewall junction"},
    "njtsswgd": {"c_field": "njtsswgd", "py_type": "float", "default": 0.0, "description": "Non-ideality factor for gate-edge sidewall junction drain side"},
    "noff": {"c_field": "noff", "py_type": "float", "default": 1.0, "description": "C-V turn-on/off parameter"},
    "noia": {"c_field": "noia", "py_type": "float", "default": 0.0, "description": "Flicker noise parameter"},
    "noib": {"c_field": "noib", "py_type": "float", "default": 0.0, "description": "Flicker noise parameter"},
    "noic": {"c_field": "noic", "py_type": "float", "default": 0.0, "description": "Flicker noise parameter"},
    "nsd": {"c_field": "nsd", "py_type": "float", "default": 1e+20, "description": "S/D doping concentration"},
    "nsub": {"c_field": "nsub", "py_type": "float", "default": 6e+16, "description": "Substrate doping concentration"},
    "ntnoi": {"c_field": "ntnoi", "py_type": "float", "default": 1.0, "description": "Thermal noise parameter"},
    "ntox": {"c_field": "ntox", "py_type": "float", "default": 1.0, "description": "Exponent for Tox ratio"},
    "pa0": {"c_field": "pa0", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of a0"},
    "pa1": {"c_field": "pa1", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of a1"},
    "pa2": {"c_field": "pa2", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of a2"},
    "pacde": {"c_field": "pacde", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of acde"},
    "pagidl": {"c_field": "pagidl", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of agidl"},
    "pagisl": {"c_field": "pagisl", "py_type": "float", "default": ("ref", "pagidl"), "description": "Cross-term dependence of agisl"},
    "pags": {"c_field": "pags", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of ags"},
    "paigbacc": {"c_field": "paigbacc", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of aigbacc"},
    "paigbinv": {"c_field": "paigbinv", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of aigbinv"},
    "paigc": {"c_field": "paigc", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of aigc"},
    "paigd": {"c_field": "paigd", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of aigd"},
    "paigs": {"c_field": "paigs", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of aigs"},
    "paigsd": {"c_field": "paigsd", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of aigsd"},
    "palpha0": {"c_field": "palpha0", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of alpha0"},
    "palpha1": {"c_field": "palpha1", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of alpha1"},
    "paramchk": {"c_field": "paramchk", "py_type": "int", "default": 1, "description": "Model parameter checking selector"},
    "pat": {"c_field": "pat", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of at"},
    "pb0": {"c_field": "pb0", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of b0"},
    "pb1": {"c_field": "pb1", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of b1"},
    "pbd": {"c_field": "pbd", "py_type": "float", "default": 0.0, "description": "Drain junction built-in potential"},
    "pbeta0": {"c_field": "pbeta0", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of beta0"},
    "pbgidl": {"c_field": "pbgidl", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of bgidl"},
    "pbgisl": {"c_field": "pbgisl", "py_type": "float", "default": ("ref", "pbgidl"), "description": "Cross-term dependence of bgisl"},
    "pbigbacc": {"c_field": "pbigbacc", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of bigbacc"},
    "pbigbinv": {"c_field": "pbigbinv", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of bigbinv"},
    "pbigc": {"c_field": "pbigc", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of bigc"},
    "pbigd": {"c_field": "pbigd", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of bigd"},
    "pbigs": {"c_field": "pbigs", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of bigs"},
    "pbigsd": {"c_field": "pbigsd", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of bigsd"},
    "pbs": {"c_field": "pbs", "py_type": "float", "default": 0.0, "description": "Source junction built-in potential"},
    "pbswd": {"c_field": "pbswd", "py_type": "float", "default": 0.0, "description": "Drain sidewall junction capacitance built in potential"},
    "pbswgd": {"c_field": "pbswgd", "py_type": "float", "default": 0.0, "description": "Drain (gate side) sidewall junction capacitance built in potential"},
    "pbswgs": {"c_field": "pbswgs", "py_type": "float", "default": 0.0, "description": "Source (gate side) sidewall junction capacitance built in potential"},
    "pbsws": {"c_field": "pbsws", "py_type": "float", "default": 0.0, "description": "Source sidewall junction capacitance built in potential"},
    "pcdsc": {"c_field": "pcdsc", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of cdsc"},
    "pcdscb": {"c_field": "pcdscb", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of cdscb"},
    "pcdscd": {"c_field": "pcdscd", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of cdscd"},
    "pcf": {"c_field": "pcf", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of cf"},
    "pcgdl": {"c_field": "pcgdl", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of cgdl"},
    "pcgidl": {"c_field": "pcgidl", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of cgidl"},
    "pcgisl": {"c_field": "pcgisl", "py_type": "float", "default": ("ref", "pcgidl"), "description": "Cross-term dependence of cgisl"},
    "pcgsl": {"c_field": "pcgsl", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of cgsl"},
    "pcigbacc": {"c_field": "pcigbacc", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of cigbacc"},
    "pcigbinv": {"c_field": "pcigbinv", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of cigbinv"},
    "pcigc": {"c_field": "pcigc", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of cigc"},
    "pcigd": {"c_field": "pcigd", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of cigd"},
    "pcigs": {"c_field": "pcigs", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of cigs"},
    "pcigsd": {"c_field": "pcigsd", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of cigsd"},
    "pcit": {"c_field": "pcit", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of cit"},
    "pckappad": {"c_field": "pckappad", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of ckappad"},
    "pckappas": {"c_field": "pckappas", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of ckappas"},
    "pclc": {"c_field": "pclc", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of clc"},
    "pcle": {"c_field": "pcle", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of cle"},
    "pclm": {"c_field": "pclm", "py_type": "float", "default": 1.3, "description": "Channel length modulation Coefficient"},
    "pdelta": {"c_field": "pdelta", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of delta"},
    # WAVE2-FIX-WORST: BSIM4 b4set.c §425-428 defaults are 0.39 / 0.0086, NOT 0.0.
    # This was the dominant source of the 88.8% Vds-grown subthreshold error.
    "pdiblc1": {"c_field": "pdiblc1", "py_type": "float", "default": 0.39, "description": "Drain-induced barrier lowering coefficient"},
    "pdiblc2": {"c_field": "pdiblc2", "py_type": "float", "default": 0.0086, "description": "Drain-induced barrier lowering coefficient"},
    "pdiblcb": {"c_field": "pdiblcb", "py_type": "float", "default": 0.0, "description": "Body-effect on drain-induced barrier lowering"},
    "pdits": {"c_field": "pdits", "py_type": "float", "default": 0.0, "description": "Coefficient for drain-induced Vth shifts"},
    "pditsd": {"c_field": "pditsd", "py_type": "float", "default": 0.0, "description": "Vds dependence of drain-induced Vth shifts"},
    "pditsl": {"c_field": "pditsl", "py_type": "float", "default": 0.0, "description": "Length dependence of drain-induced Vth shifts"},
    "pdrout": {"c_field": "pdrout", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of drout"},
    "pdsub": {"c_field": "pdsub", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of dsub"},
    "pdvt0": {"c_field": "pdvt0", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of dvt0"},
    "pdvt0w": {"c_field": "pdvt0w", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of dvt0w"},
    "pdvt1": {"c_field": "pdvt1", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of dvt1"},
    "pdvt1w": {"c_field": "pdvt1w", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of dvt1w"},
    "pdvt2": {"c_field": "pdvt2", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of dvt2"},
    "pdvt2w": {"c_field": "pdvt2w", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of dvt2w"},
    "pdvtp0": {"c_field": "pdvtp0", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of dvtp0"},
    "pdvtp1": {"c_field": "pdvtp1", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of dvtp1"},
    "pdvtp2": {"c_field": "pdvtp2", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of dvtp2"},
    "pdvtp3": {"c_field": "pdvtp3", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of dvtp3"},
    "pdvtp4": {"c_field": "pdvtp4", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of dvtp4"},
    "pdvtp5": {"c_field": "pdvtp5", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of dvtp5"},
    "pdwb": {"c_field": "pdwb", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of dwb"},
    "pdwg": {"c_field": "pdwg", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of dwg"},
    "pegidl": {"c_field": "pegidl", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of egidl"},
    "pegisl": {"c_field": "pegisl", "py_type": "float", "default": ("ref", "pegidl"), "description": "Cross-term dependence of egisl"},
    "peigbinv": {"c_field": "peigbinv", "py_type": "float", "default": 0.0, "description": "Cross-term dependence for eigbinv"},
    "permod": {"c_field": "permod", "py_type": "int", "default": 1, "description": "Pd and Ps model selector"},
    "peta0": {"c_field": "peta0", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of eta0"},
    "petab": {"c_field": "petab", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of etab"},
    "peu": {"c_field": "peu", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of eu"},
    "pfgidl": {"c_field": "pfgidl", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of fgidl"},
    "pfgisl": {"c_field": "pfgisl", "py_type": "float", "default": ("ref", "pfgidl"), "description": "Cross-term dependence of fgisl"},
    "pfprout": {"c_field": "pfprout", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of pdiblcb"},
    "pgamma1": {"c_field": "pgamma1", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of gamma1"},
    "pgamma2": {"c_field": "pgamma2", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of gamma2"},
    "phig": {"c_field": "phig", "py_type": "float", "default": 4.05, "description": "Work function of gate"},
    "phin": {"c_field": "phin", "py_type": "float", "default": 0.0, "description": "Adjusting parameter for surface potential due to non-uniform vertical doping"},
    "pigcd": {"c_field": "pigcd", "py_type": "float", "default": 1.0, "description": "Parameter for Igc partition"},
    "pk1": {"c_field": "pk1", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of k1"},
    "pk2": {"c_field": "pk2", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of k2"},
    "pk2we": {"c_field": "pk2we", "py_type": "float", "default": 0.0, "description": " Cross-term dependence of k2we "},
    "pk3": {"c_field": "pk3", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of k3"},
    "pk3b": {"c_field": "pk3b", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of k3b"},
    "pketa": {"c_field": "pketa", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of keta"},
    "pketac": {"c_field": "pketac", "py_type": "float", "default": ("ref", "pketa"), "description": "Cross-term dependence of ketac"},
    "pkgidl": {"c_field": "pkgidl", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of kgidl"},
    "pkgisl": {"c_field": "pkgisl", "py_type": "float", "default": ("ref", "pkgidl"), "description": "Cross-term dependence of kgisl"},
    "pkt1": {"c_field": "pkt1", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of kt1"},
    "pkt1l": {"c_field": "pkt1l", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of kt1l"},
    "pkt2": {"c_field": "pkt2", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of kt2"},
    "pku0": {"c_field": "pku0", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of ku0"},
    "pku0we": {"c_field": "pku0we", "py_type": "float", "default": 0.0, "description": " Cross-term dependence of ku0we "},
    "pkvth0": {"c_field": "pkvth0", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of kvth0"},
    "pkvth0we": {"c_field": "pkvth0we", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of kvth0we"},
    "plambda": {"c_field": "plambda", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of lambda"},
    "plp": {"c_field": "plp", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of lp"},
    "plpe0": {"c_field": "plpe0", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of lpe0"},
    "plpeb": {"c_field": "plpeb", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of lpeb"},
    "pminv": {"c_field": "pminv", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of minv"},
    "pminvcv": {"c_field": "pminvcv", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of minvcv"},
    "pmoin": {"c_field": "pmoin", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of moin"},
    "pndep": {"c_field": "pndep", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of ndep"},
    "pnfactor": {"c_field": "pnfactor", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of nfactor"},
    "pngate": {"c_field": "pngate", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of ngate"},
    "pnigbacc": {"c_field": "pnigbacc", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of nigbacc"},
    "pnigbinv": {"c_field": "pnigbinv", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of nigbinv"},
    "pnigc": {"c_field": "pnigc", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of nigc"},
    "pnoff": {"c_field": "pnoff", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of noff"},
    "pnsd": {"c_field": "pnsd", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of nsd"},
    "pnsub": {"c_field": "pnsub", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of nsub"},
    "pntox": {"c_field": "pntox", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of ntox"},
    "poxedge": {"c_field": "poxedge", "py_type": "float", "default": 1.0, "description": "Factor for the gate edge Tox"},
    "ppclm": {"c_field": "ppclm", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of pclm"},
    "ppdiblc1": {"c_field": "ppdiblc1", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of pdiblc1"},
    "ppdiblc2": {"c_field": "ppdiblc2", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of pdiblc2"},
    "ppdiblcb": {"c_field": "ppdiblcb", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of pdiblcb"},
    "ppdits": {"c_field": "ppdits", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of pdits"},
    "ppditsd": {"c_field": "ppditsd", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of pditsd"},
    "pphin": {"c_field": "pphin", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of phin"},
    "ppigcd": {"c_field": "ppigcd", "py_type": "float", "default": 0.0, "description": "Cross-term dependence for pigcd"},
    "ppoxedge": {"c_field": "ppoxedge", "py_type": "float", "default": 0.0, "description": "Cross-term dependence for poxedge"},
    "pprt": {"c_field": "pprt", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of prt "},
    "pprwb": {"c_field": "pprwb", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of prwb "},
    "pprwg": {"c_field": "pprwg", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of prwg "},
    "ppscbe1": {"c_field": "ppscbe1", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of pscbe1"},
    "ppscbe2": {"c_field": "ppscbe2", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of pscbe2"},
    "ppvag": {"c_field": "ppvag", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of pvag"},
    "prdsw": {"c_field": "prdsw", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of rdsw "},
    "prdw": {"c_field": "prdw", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of rdw"},
    "prgidl": {"c_field": "prgidl", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of rgidl"},
    "prgisl": {"c_field": "prgisl", "py_type": "float", "default": ("ref", "prgidl"), "description": "Cross-term dependence of rgisl"},
    "prsw": {"c_field": "prsw", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of rsw"},
    "prt": {"c_field": "prt", "py_type": "float", "default": 0.0, "description": "Temperature coefficient of parasitic resistance "},
    "prwb": {"c_field": "prwb", "py_type": "float", "default": 0.0, "description": "Body-effect on parasitic resistance "},
    "prwg": {"c_field": "prwg", "py_type": "float", "default": 1.0, "description": "Gate-bias effect on parasitic resistance "},
    "pscbe1": {"c_field": "pscbe1", "py_type": "float", "default": 424000000.0, "description": "Substrate current body-effect coefficient"},
    "pscbe2": {"c_field": "pscbe2", "py_type": "float", "default": 1e-05, "description": "Substrate current body-effect coefficient"},
    "pteta0": {"c_field": "pteta0", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of teta0"},
    "ptnfactor": {"c_field": "ptnfactor", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of tnfactor"},
    "ptvfbsdoff": {"c_field": "ptvfbsdoff", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of tvfbsdoff"},
    "ptvoff": {"c_field": "ptvoff", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of tvoff"},
    "ptvoffcv": {"c_field": "ptvoffcv", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of tvoffcv"},
    "pu0": {"c_field": "pu0", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of u0"},
    "pua": {"c_field": "pua", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of ua"},
    "pua1": {"c_field": "pua1", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of ua1"},
    "pub": {"c_field": "pub", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of ub"},
    "pub1": {"c_field": "pub1", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of ub1"},
    "puc": {"c_field": "puc", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of uc"},
    "puc1": {"c_field": "puc1", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of uc1"},
    "pucs": {"c_field": "pucs", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of ucs"},
    "pucste": {"c_field": "pucste", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of ucste"},
    "pud": {"c_field": "pud", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of ud"},
    "pud1": {"c_field": "pud1", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of ud1"},
    "pup": {"c_field": "pup", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of up"},
    "pute": {"c_field": "pute", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of ute"},
    "pvag": {"c_field": "pvag", "py_type": "float", "default": 0.0, "description": "Gate dependence of output resistance parameter"},
    "pvbm": {"c_field": "pvbm", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of vbm"},
    "pvbx": {"c_field": "pvbx", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of vbx"},
    "pvfb": {"c_field": "pvfb", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of vfb"},
    "pvfbcv": {"c_field": "pvfbcv", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of vfbcv"},
    "pvfbsdoff": {"c_field": "pvfbsdoff", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of vfbsdoff"},
    "pvoff": {"c_field": "pvoff", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of voff"},
    "pvoffcv": {"c_field": "pvoffcv", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of voffcv"},
    "pvsat": {"c_field": "pvsat", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of vsat"},
    "pvth0": {"c_field": "pvth0", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of vto"},
    "pvtl": {"c_field": "pvtl", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of vtl"},
    "pw0": {"c_field": "pw0", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of w0"},
    "pwr": {"c_field": "pwr", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of wr"},
    "pxj": {"c_field": "pxj", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of xj"},
    "pxn": {"c_field": "pxn", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of xn"},
    "pxrcrg1": {"c_field": "pxrcrg1", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of xrcrg1"},
    "pxrcrg2": {"c_field": "pxrcrg2", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of xrcrg2"},
    "pxt": {"c_field": "pxt", "py_type": "float", "default": 0.0, "description": "Cross-term dependence of xt"},
    "rbdb": {"c_field": "rbdb", "py_type": "float", "default": 50.0, "description": "Resistance between bNode and dbNode"},
    "rbdbx0": {"c_field": "rbdbx0", "py_type": "float", "default": 100.0, "description": "Body resistance RBDBX  scaling"},
    "rbdby0": {"c_field": "rbdby0", "py_type": "float", "default": 100.0, "description": "Body resistance RBDBY  scaling"},
    "rbodymod": {"c_field": "rbodymod", "py_type": "int", "default": 0, "description": "Distributed body R model selector"},
    "rbpb": {"c_field": "rbpb", "py_type": "float", "default": 50.0, "description": "Resistance between bNodePrime and bNode"},
    "rbpbx0": {"c_field": "rbpbx0", "py_type": "float", "default": 100.0, "description": "Body resistance RBPBX  scaling"},
    "rbpbxl": {"c_field": "rbpbxl", "py_type": "float", "default": 0.0, "description": "Body resistance RBPBX L scaling"},
    "rbpbxnf": {"c_field": "rbpbxnf", "py_type": "float", "default": 0.0, "description": "Body resistance RBPBX NF scaling"},
    "rbpbxw": {"c_field": "rbpbxw", "py_type": "float", "default": 0.0, "description": "Body resistance RBPBX W scaling"},
    "rbpby0": {"c_field": "rbpby0", "py_type": "float", "default": 100.0, "description": "Body resistance RBPBY  scaling"},
    "rbpbyl": {"c_field": "rbpbyl", "py_type": "float", "default": 0.0, "description": "Body resistance RBPBY L scaling"},
    "rbpbynf": {"c_field": "rbpbynf", "py_type": "float", "default": 0.0, "description": "Body resistance RBPBY NF scaling"},
    "rbpbyw": {"c_field": "rbpbyw", "py_type": "float", "default": 0.0, "description": "Body resistance RBPBY W scaling"},
    "rbpd": {"c_field": "rbpd", "py_type": "float", "default": 50.0, "description": "Resistance between bNodePrime and bNode"},
    "rbpd0": {"c_field": "rbpd0", "py_type": "float", "default": 50.0, "description": "Body resistance RBPD scaling"},
    "rbpdl": {"c_field": "rbpdl", "py_type": "float", "default": 0.0, "description": "Body resistance RBPD L scaling"},
    "rbpdnf": {"c_field": "rbpdnf", "py_type": "float", "default": 0.0, "description": "Body resistance RBPD NF scaling"},
    "rbpdw": {"c_field": "rbpdw", "py_type": "float", "default": 0.0, "description": "Body resistance RBPD W scaling"},
    "rbps": {"c_field": "rbps", "py_type": "float", "default": 50.0, "description": "Resistance between bNodePrime and sbNode"},
    "rbps0": {"c_field": "rbps0", "py_type": "float", "default": 50.0, "description": "Body resistance RBPS scaling"},
    "rbpsl": {"c_field": "rbpsl", "py_type": "float", "default": 0.0, "description": "Body resistance RBPS L scaling"},
    "rbpsnf": {"c_field": "rbpsnf", "py_type": "float", "default": 0.0, "description": "Body resistance RBPS NF scaling"},
    "rbpsw": {"c_field": "rbpsw", "py_type": "float", "default": 0.0, "description": "Body resistance RBPS W scaling"},
    "rbsb": {"c_field": "rbsb", "py_type": "float", "default": 50.0, "description": "Resistance between bNode and sbNode"},
    "rbsbx0": {"c_field": "rbsbx0", "py_type": "float", "default": 100.0, "description": "Body resistance RBSBX  scaling"},
    "rbsby0": {"c_field": "rbsby0", "py_type": "float", "default": 100.0, "description": "Body resistance RBSBY  scaling"},
    "rbsdbxl": {"c_field": "rbsdbxl", "py_type": "float", "default": 0.0, "description": "Body resistance RBSDBX L scaling"},
    "rbsdbxnf": {"c_field": "rbsdbxnf", "py_type": "float", "default": 0.0, "description": "Body resistance RBSDBX NF scaling"},
    "rbsdbxw": {"c_field": "rbsdbxw", "py_type": "float", "default": 0.0, "description": "Body resistance RBSDBX W scaling"},
    "rbsdbyl": {"c_field": "rbsdbyl", "py_type": "float", "default": 0.0, "description": "Body resistance RBSDBY L scaling"},
    "rbsdbynf": {"c_field": "rbsdbynf", "py_type": "float", "default": 0.0, "description": "Body resistance RBSDBY NF scaling"},
    "rbsdbyw": {"c_field": "rbsdbyw", "py_type": "float", "default": 0.0, "description": "Body resistance RBSDBY W scaling"},
    "rdsmod": {"c_field": "rdsmod", "py_type": "int", "default": 0, "description": "Bias-dependent S/D resistance model selector"},
    "rdsw": {"c_field": "rdsw", "py_type": "float", "default": 200.0, "description": "Source-drain resistance per width"},
    "rdswmin": {"c_field": "rdswmin", "py_type": "float", "default": 0.0, "description": "Source-drain resistance per width at high Vg"},
    "rdw": {"c_field": "rdw", "py_type": "float", "default": 100.0, "description": "Drain resistance per width"},
    "rdwmin": {"c_field": "rdwmin", "py_type": "float", "default": 0.0, "description": "Drain resistance per width at high Vg"},
    "rgatemod": {"c_field": "rgatemod", "py_type": "int", "default": 0, "description": "Gate R model selector"},
    "rgidl": {"c_field": "rgidl", "py_type": "float", "default": 1.0, "description": "GIDL vg parameter"},
    "rgisl": {"c_field": "rgisl", "py_type": "float", "default": ("ref", "rgidl"), "description": "GISL vg parameter"},
    "rnoia": {"c_field": "rnoia", "py_type": "float", "default": 0.577, "description": "Thermal noise coefficient"},
    "rnoib": {"c_field": "rnoib", "py_type": "float", "default": 0.5164, "description": "Thermal noise coefficient"},
    "rnoic": {"c_field": "rnoic", "py_type": "float", "default": 0.395, "description": "Thermal noise coefficient"},
    "rsh": {"c_field": "rsh", "py_type": "float", "default": 0.0, "description": "Source-drain sheet resistance"},
    "rshg": {"c_field": "rshg", "py_type": "float", "default": 0.1, "description": "Gate sheet resistance"},
    "rsw": {"c_field": "rsw", "py_type": "float", "default": 100.0, "description": "Source resistance per width"},
    "rswmin": {"c_field": "rswmin", "py_type": "float", "default": 0.0, "description": "Source resistance per width at high Vg"},
    "saref": {"c_field": "saref", "py_type": "float", "default": 1e-06, "description": "Reference distance between OD edge to poly of one side"},
    "sbref": {"c_field": "sbref", "py_type": "float", "default": 1e-06, "description": "Reference distance between OD edge to poly of the other side"},
    "scref": {"c_field": "scref", "py_type": "float", "default": 1e-06, "description": " Reference distance to calculate SCA, SCB and SCC"},
    "steta0": {"c_field": "steta0", "py_type": "float", "default": 0.0, "description": "eta0 shift factor related to stress effect on vth"},
    "stk2": {"c_field": "stk2", "py_type": "float", "default": 0.0, "description": "K2 shift factor related to stress effect on vth"},
    "tbgasub": {"c_field": "tbgasub", "py_type": "float", "default": 0.000702, "description": "First parameter of band-gap change due to temperature"},
    "tbgbsub": {"c_field": "tbgbsub", "py_type": "float", "default": 1108.0, "description": "Second parameter of band-gap change due to temperature"},
    "tcj": {"c_field": "tcj", "py_type": "float", "default": 0.0, "description": "Temperature coefficient of cj"},
    "tcjsw": {"c_field": "tcjsw", "py_type": "float", "default": 0.0, "description": "Temperature coefficient of cjsw"},
    "tcjswg": {"c_field": "tcjswg", "py_type": "float", "default": 0.0, "description": "Temperature coefficient of cjswg"},
    "tempeot": {"c_field": "tempeot", "py_type": "float", "default": 300.15, "description": " Temperature for extraction of EOT"},
    "tempmod": {"c_field": "tempmod", "py_type": "int", "default": 0, "description": "Temperature model selector"},
    "teta0": {"c_field": "teta0", "py_type": "float", "default": 0.0, "description": "Temperature parameter for eta0"},
    "tku0": {"c_field": "tku0", "py_type": "float", "default": 0.0, "description": "Temperature coefficient of KU0"},
    "tnfactor": {"c_field": "tnfactor", "py_type": "float", "default": 0.0, "description": "Temperature parameter for nfactor"},
    "tnjts": {"c_field": "tnjts", "py_type": "float", "default": 0.0, "description": "Temperature coefficient for NJTS"},
    "tnjtsd": {"c_field": "tnjtsd", "py_type": "float", "default": 0.0, "description": "Temperature coefficient for NJTSD"},
    "tnjtssw": {"c_field": "tnjtssw", "py_type": "float", "default": 0.0, "description": "Temperature coefficient for NJTSSW"},
    "tnjtsswd": {"c_field": "tnjtsswd", "py_type": "float", "default": 0.0, "description": "Temperature coefficient for NJTSSWD"},
    "tnjtsswg": {"c_field": "tnjtsswg", "py_type": "float", "default": 0.0, "description": "Temperature coefficient for NJTSSWG"},
    "tnjtsswgd": {"c_field": "tnjtsswgd", "py_type": "float", "default": 0.0, "description": "Temperature coefficient for NJTSSWGD"},
    "tnoia": {"c_field": "tnoia", "py_type": "float", "default": 1.5, "description": "Thermal noise parameter"},
    "tnoib": {"c_field": "tnoib", "py_type": "float", "default": 3.5, "description": "Thermal noise parameter"},
    "tnoic": {"c_field": "tnoic", "py_type": "float", "default": 0.0, "description": "Thermal noise parameter"},
    "tnoimod": {"c_field": "tnoimod", "py_type": "int", "default": 0, "description": "Thermal noise model selector"},
    "tnom": {"c_field": "tnom", "py_type": "float", "default": 27.0, "description": "Parameter measurement temperature"},
    "toxe": {"c_field": "toxe", "py_type": "float", "default": 3e-09, "description": "Electrical gate oxide thickness in meters"},
    "toxm": {"c_field": "toxm", "py_type": "float", "default": ("ref", "toxe"), "description": "Gate oxide thickness at which parameters are extracted"},
    "toxp": {"c_field": "toxp", "py_type": "float", "default": ("ref", "toxe"), "description": "Physical gate oxide thickness in meters"},
    "toxref": {"c_field": "toxref", "py_type": "float", "default": 3e-09, "description": "Target tox value"},
    "tpb": {"c_field": "tpb", "py_type": "float", "default": 0.0, "description": "Temperature coefficient of pb"},
    "tpbsw": {"c_field": "tpbsw", "py_type": "float", "default": 0.0, "description": "Temperature coefficient of pbsw"},
    "tpbswg": {"c_field": "tpbswg", "py_type": "float", "default": 0.0, "description": "Temperature coefficient of pbswg"},
    "trnqsmod": {"c_field": "trnqsmod", "py_type": "int", "default": 0, "description": "Transient NQS model selector"},
    "tvfbsdoff": {"c_field": "tvfbsdoff", "py_type": "float", "default": 0.0, "description": "Temperature parameter for vfbsdoff"},
    "tvoff": {"c_field": "tvoff", "py_type": "float", "default": 0.0, "description": "Temperature parameter for voff"},
    "tvoffcv": {"c_field": "tvoffcv", "py_type": "float", "default": 0.0, "description": "Temperature parameter for tvoffcv"},
    "u0": {"c_field": "u0", "py_type": "float", "default": ("nmos_pmos", 0.067, 0.025), "description": "Low-field mobility at Tnom"},
    "ua": {"c_field": "ua", "py_type": "float", "default": ("mobmod", 1.0e-15, 1.0e-9), "description": "Linear gate dependence of mobility"},
    "ua1": {"c_field": "ua1", "py_type": "float", "default": 1e-09, "description": "Temperature coefficient of ua"},
    "ub": {"c_field": "ub", "py_type": "float", "default": 1e-19, "description": "Quadratic gate dependence of mobility"},
    "ub1": {"c_field": "ub1", "py_type": "float", "default": -1e-18, "description": "Temperature coefficient of ub"},
    "uc": {"c_field": "uc", "py_type": "float", "default": ("mobmod", -0.0465, -0.0465e-9), "description": "Body-bias dependence of mobility"},
    "uc1": {"c_field": "uc1", "py_type": "float", "default": ("mobmod", -0.056, -0.056e-9), "description": "Temperature coefficient of uc"},
    "ucs": {"c_field": "ucs", "py_type": "float", "default": ("nmos_pmos", 1.67, 1.0), "description": "Colombic scattering exponent"},
    "ucste": {"c_field": "ucste", "py_type": "float", "default": -0.004775, "description": "Temperature coefficient of colombic mobility"},
    "ud": {"c_field": "ud", "py_type": "float", "default": 0.0, "description": "Coulomb scattering factor of mobility"},
    "ud1": {"c_field": "ud1", "py_type": "float", "default": 0.0, "description": "Temperature coefficient of ud"},
    "up": {"c_field": "up", "py_type": "float", "default": 0.0, "description": "Channel length linear factor of mobility"},
    "ute": {"c_field": "ute", "py_type": "float", "default": -1.5, "description": "Temperature coefficient of mobility"},
    "vbm": {"c_field": "vbm", "py_type": "float", "default": -3.0, "description": "Maximum body voltage"},
    "vbx": {"c_field": "vbx", "py_type": "float", "default": 0.0, "description": "Vth transition body Voltage"},
    "vddeot": {"c_field": "vddeot", "py_type": "float", "default": ("nmos_pmos", 1.5, -1.5), "description": "Voltage for extraction of Equivalent gate oxide thickness"},
    "version": {"c_field": "version", "py_type": "float", "default": 4.83, "description": "parameter for model version"},
    "vfb": {"c_field": "vfb", "py_type": "float", "default": -1.0, "description": "Flat Band Voltage"},
    "vfbcv": {"c_field": "vfbcv", "py_type": "float", "default": -1.0, "description": "Flat Band Voltage parameter for capmod=0 only"},
    "vfbsdoff": {"c_field": "vfbsdoff", "py_type": "float", "default": 0.0, "description": "S/D flatband voltage offset"},
    "voff": {"c_field": "voff", "py_type": "float", "default": -0.08, "description": "Threshold voltage offset"},
    "voffcv": {"c_field": "voffcv", "py_type": "float", "default": 0.0, "description": "C-V lateral-shift parameter"},
    "voffcvl": {"c_field": "voffcvl", "py_type": "float", "default": 0.0, "description": "Length dependence parameter for Vth offset in CV"},
    "voffl": {"c_field": "voffl", "py_type": "float", "default": 0.0, "description": "Length dependence parameter for Vth offset"},
    "vsat": {"c_field": "vsat", "py_type": "float", "default": 80000.0, "description": "Saturation velocity at tnom"},
    "vth0": {"c_field": "vth0", "py_type": "float", "default": ("nmos_pmos", 0.7, -0.7), "description": "Threshold voltage"},
    "vtl": {"c_field": "vtl", "py_type": "float", "default": 200000.0, "description": " thermal velocity"},
    "vtsd": {"c_field": "vtsd", "py_type": "float", "default": ("ref", "vtss"), "description": "Drain bottom trap-assisted voltage dependent parameter"},
    "vtss": {"c_field": "vtss", "py_type": "float", "default": 10.0, "description": "Source bottom trap-assisted voltage dependent parameter"},
    "vtsswd": {"c_field": "vtsswd", "py_type": "float", "default": ("ref", "vtssws"), "description": "Drain STI sidewall trap-assisted voltage dependent parameter"},
    "vtsswgd": {"c_field": "vtsswgd", "py_type": "float", "default": ("ref", "vtsswgs"), "description": "Drain gate-edge sidewall trap-assisted voltage dependent parameter"},
    "vtsswgs": {"c_field": "vtsswgs", "py_type": "float", "default": 10.0, "description": "Source gate-edge sidewall trap-assisted voltage dependent parameter"},
    "vtssws": {"c_field": "vtssws", "py_type": "float", "default": 10.0, "description": "Source STI sidewall trap-assisted voltage dependent parameter"},
    "w0": {"c_field": "w0", "py_type": "float", "default": 2.5e-06, "description": "Narrow width effect parameter"},
    "wa0": {"c_field": "wa0", "py_type": "float", "default": 0.0, "description": "Width dependence of a0"},
    "wa1": {"c_field": "wa1", "py_type": "float", "default": 0.0, "description": "Width dependence of a1"},
    "wa2": {"c_field": "wa2", "py_type": "float", "default": 0.0, "description": "Width dependence of a2"},
    "wacde": {"c_field": "wacde", "py_type": "float", "default": 0.0, "description": "Width dependence of acde"},
    "wagidl": {"c_field": "wagidl", "py_type": "float", "default": 0.0, "description": "Width dependence of agidl"},
    "wagisl": {"c_field": "wagisl", "py_type": "float", "default": ("ref", "wagidl"), "description": "Width dependence of agisl"},
    "wags": {"c_field": "wags", "py_type": "float", "default": 0.0, "description": "Width dependence of ags"},
    "waigbacc": {"c_field": "waigbacc", "py_type": "float", "default": 0.0, "description": "Width dependence of aigbacc"},
    "waigbinv": {"c_field": "waigbinv", "py_type": "float", "default": 0.0, "description": "Width dependence of aigbinv"},
    "waigc": {"c_field": "waigc", "py_type": "float", "default": 0.0, "description": "Width dependence of aigc"},
    "waigd": {"c_field": "waigd", "py_type": "float", "default": 0.0, "description": "Width dependence of aigd"},
    "waigs": {"c_field": "waigs", "py_type": "float", "default": 0.0, "description": "Width dependence of aigs"},
    "waigsd": {"c_field": "waigsd", "py_type": "float", "default": 0.0, "description": "Width dependence of aigsd"},
    "walpha0": {"c_field": "walpha0", "py_type": "float", "default": 0.0, "description": "Width dependence of alpha0"},
    "walpha1": {"c_field": "walpha1", "py_type": "float", "default": 0.0, "description": "Width dependence of alpha1"},
    "wat": {"c_field": "wat", "py_type": "float", "default": 0.0, "description": "Width dependence of at"},
    "wb0": {"c_field": "wb0", "py_type": "float", "default": 0.0, "description": "Width dependence of b0"},
    "wb1": {"c_field": "wb1", "py_type": "float", "default": 0.0, "description": "Width dependence of b1"},
    "wbeta0": {"c_field": "wbeta0", "py_type": "float", "default": 0.0, "description": "Width dependence of beta0"},
    "wbgidl": {"c_field": "wbgidl", "py_type": "float", "default": 0.0, "description": "Width dependence of bgidl"},
    "wbgisl": {"c_field": "wbgisl", "py_type": "float", "default": ("ref", "wbgidl"), "description": "Width dependence of bgisl"},
    "wbigbacc": {"c_field": "wbigbacc", "py_type": "float", "default": 0.0, "description": "Width dependence of bigbacc"},
    "wbigbinv": {"c_field": "wbigbinv", "py_type": "float", "default": 0.0, "description": "Width dependence of bigbinv"},
    "wbigc": {"c_field": "wbigc", "py_type": "float", "default": 0.0, "description": "Width dependence of bigc"},
    "wbigd": {"c_field": "wbigd", "py_type": "float", "default": 0.0, "description": "Width dependence of bigd"},
    "wbigs": {"c_field": "wbigs", "py_type": "float", "default": 0.0, "description": "Width dependence of bigs"},
    "wbigsd": {"c_field": "wbigsd", "py_type": "float", "default": 0.0, "description": "Width dependence of bigsd"},
    "wcdsc": {"c_field": "wcdsc", "py_type": "float", "default": 0.0, "description": "Width dependence of cdsc"},
    "wcdscb": {"c_field": "wcdscb", "py_type": "float", "default": 0.0, "description": "Width dependence of cdscb"},
    "wcdscd": {"c_field": "wcdscd", "py_type": "float", "default": 0.0, "description": "Width dependence of cdscd"},
    "wcf": {"c_field": "wcf", "py_type": "float", "default": 0.0, "description": "Width dependence of cf"},
    "wcgdl": {"c_field": "wcgdl", "py_type": "float", "default": 0.0, "description": "Width dependence of cgdl"},
    "wcgidl": {"c_field": "wcgidl", "py_type": "float", "default": 0.0, "description": "Width dependence of cgidl"},
    "wcgisl": {"c_field": "wcgisl", "py_type": "float", "default": ("ref", "wcgidl"), "description": "Width dependence of cgisl"},
    "wcgsl": {"c_field": "wcgsl", "py_type": "float", "default": 0.0, "description": "Width dependence of cgsl"},
    "wcigbacc": {"c_field": "wcigbacc", "py_type": "float", "default": 0.0, "description": "Width dependence of cigbacc"},
    "wcigbinv": {"c_field": "wcigbinv", "py_type": "float", "default": 0.0, "description": "Width dependence of cigbinv"},
    "wcigc": {"c_field": "wcigc", "py_type": "float", "default": 0.0, "description": "Width dependence of cigc"},
    "wcigd": {"c_field": "wcigd", "py_type": "float", "default": 0.0, "description": "Width dependence of cigd"},
    "wcigs": {"c_field": "wcigs", "py_type": "float", "default": 0.0, "description": "Width dependence of cigs"},
    "wcigsd": {"c_field": "wcigsd", "py_type": "float", "default": 0.0, "description": "Width dependence of cigsd"},
    "wcit": {"c_field": "wcit", "py_type": "float", "default": 0.0, "description": "Width dependence of cit"},
    "wckappad": {"c_field": "wckappad", "py_type": "float", "default": 0.0, "description": "Width dependence of ckappad"},
    "wckappas": {"c_field": "wckappas", "py_type": "float", "default": 0.0, "description": "Width dependence of ckappas"},
    "wclc": {"c_field": "wclc", "py_type": "float", "default": 0.0, "description": "Width dependence of clc"},
    "wcle": {"c_field": "wcle", "py_type": "float", "default": 0.0, "description": "Width dependence of cle"},
    "wdelta": {"c_field": "wdelta", "py_type": "float", "default": 0.0, "description": "Width dependence of delta"},
    "wdrout": {"c_field": "wdrout", "py_type": "float", "default": 0.0, "description": "Width dependence of drout"},
    "wdsub": {"c_field": "wdsub", "py_type": "float", "default": 0.0, "description": "Width dependence of dsub"},
    "wdvt0": {"c_field": "wdvt0", "py_type": "float", "default": 0.0, "description": "Width dependence of dvt0"},
    "wdvt0w": {"c_field": "wdvt0w", "py_type": "float", "default": 0.0, "description": "Width dependence of dvt0w"},
    "wdvt1": {"c_field": "wdvt1", "py_type": "float", "default": 0.0, "description": "Width dependence of dvt1"},
    "wdvt1w": {"c_field": "wdvt1w", "py_type": "float", "default": 0.0, "description": "Width dependence of dvt1w"},
    "wdvt2": {"c_field": "wdvt2", "py_type": "float", "default": 0.0, "description": "Width dependence of dvt2"},
    "wdvt2w": {"c_field": "wdvt2w", "py_type": "float", "default": 0.0, "description": "Width dependence of dvt2w"},
    "wdvtp0": {"c_field": "wdvtp0", "py_type": "float", "default": 0.0, "description": "Width dependence of dvtp0"},
    "wdvtp1": {"c_field": "wdvtp1", "py_type": "float", "default": 0.0, "description": "Width dependence of dvtp1"},
    "wdvtp2": {"c_field": "wdvtp2", "py_type": "float", "default": 0.0, "description": "Width dependence of dvtp2"},
    "wdvtp3": {"c_field": "wdvtp3", "py_type": "float", "default": 0.0, "description": "Width dependence of dvtp3"},
    "wdvtp4": {"c_field": "wdvtp4", "py_type": "float", "default": 0.0, "description": "Width dependence of dvtp4"},
    "wdvtp5": {"c_field": "wdvtp5", "py_type": "float", "default": 0.0, "description": "Width dependence of dvtp5"},
    "wdwb": {"c_field": "wdwb", "py_type": "float", "default": 0.0, "description": "Width dependence of dwb"},
    "wdwg": {"c_field": "wdwg", "py_type": "float", "default": 0.0, "description": "Width dependence of dwg"},
    "web": {"c_field": "web", "py_type": "float", "default": 0.0, "description": "Coefficient for SCB"},
    "wec": {"c_field": "wec", "py_type": "float", "default": 0.0, "description": "Coefficient for SCC"},
    "weffeot": {"c_field": "weffeot", "py_type": "float", "default": 10.0, "description": "Effective width for extraction of EOT"},
    "wegidl": {"c_field": "wegidl", "py_type": "float", "default": 0.0, "description": "Width dependence of egidl"},
    "wegisl": {"c_field": "wegisl", "py_type": "float", "default": ("ref", "wegidl"), "description": "Width dependence of egisl"},
    "weigbinv": {"c_field": "weigbinv", "py_type": "float", "default": 0.0, "description": "Width dependence for eigbinv"},
    "weta0": {"c_field": "weta0", "py_type": "float", "default": 0.0, "description": "Width dependence of eta0"},
    "wetab": {"c_field": "wetab", "py_type": "float", "default": 0.0, "description": "Width dependence of etab"},
    "weu": {"c_field": "weu", "py_type": "float", "default": 0.0, "description": "Width dependence of eu"},
    "wfgidl": {"c_field": "wfgidl", "py_type": "float", "default": 0.0, "description": "Width dependence of fgidl"},
    "wfgisl": {"c_field": "wfgisl", "py_type": "float", "default": ("ref", "wfgidl"), "description": "Width dependence of fgisl"},
    "wfprout": {"c_field": "wfprout", "py_type": "float", "default": 0.0, "description": "Width dependence of pdiblcb"},
    "wgamma1": {"c_field": "wgamma1", "py_type": "float", "default": 0.0, "description": "Width dependence of gamma1"},
    "wgamma2": {"c_field": "wgamma2", "py_type": "float", "default": 0.0, "description": "Width dependence of gamma2"},
    "wint": {"c_field": "wint", "py_type": "float", "default": 0.0, "description": "Width reduction parameter"},
    "wk1": {"c_field": "wk1", "py_type": "float", "default": 0.0, "description": "Width dependence of k1"},
    "wk2": {"c_field": "wk2", "py_type": "float", "default": 0.0, "description": "Width dependence of k2"},
    "wk2we": {"c_field": "wk2we", "py_type": "float", "default": 0.0, "description": " Width dependence of k2we "},
    "wk3": {"c_field": "wk3", "py_type": "float", "default": 0.0, "description": "Width dependence of k3"},
    "wk3b": {"c_field": "wk3b", "py_type": "float", "default": 0.0, "description": "Width dependence of k3b"},
    "wketa": {"c_field": "wketa", "py_type": "float", "default": 0.0, "description": "Width dependence of keta"},
    "wketac": {"c_field": "wketac", "py_type": "float", "default": ("ref", "wketa"), "description": "Width dependence of ketac"},
    "wkgidl": {"c_field": "wkgidl", "py_type": "float", "default": 0.0, "description": "Width dependence of kgidl"},
    "wkgisl": {"c_field": "wkgisl", "py_type": "float", "default": ("ref", "wkgidl"), "description": "Width dependence of kgisl"},
    "wkt1": {"c_field": "wkt1", "py_type": "float", "default": 0.0, "description": "Width dependence of kt1"},
    "wkt1l": {"c_field": "wkt1l", "py_type": "float", "default": 0.0, "description": "Width dependence of kt1l"},
    "wkt2": {"c_field": "wkt2", "py_type": "float", "default": 0.0, "description": "Width dependence of kt2"},
    "wku0": {"c_field": "wku0", "py_type": "float", "default": 0.0, "description": "Width dependence of ku0"},
    "wku0we": {"c_field": "wku0we", "py_type": "float", "default": 0.0, "description": " Width dependence of ku0we "},
    "wkvth0": {"c_field": "wkvth0", "py_type": "float", "default": 0.0, "description": "Width dependence of kvth0"},
    "wkvth0we": {"c_field": "wkvth0we", "py_type": "float", "default": 0.0, "description": "Width dependence of kvth0we"},
    "wl": {"c_field": "wl", "py_type": "float", "default": 0.0, "description": "Width reduction parameter"},
    "wlambda": {"c_field": "wlambda", "py_type": "float", "default": 0.0, "description": "Width dependence of lambda"},
    "wlc": {"c_field": "wlc", "py_type": "float", "default": ("ref", "wl"), "description": "Width reduction parameter for CV"},
    "wln": {"c_field": "wln", "py_type": "float", "default": 1.0, "description": "Width reduction parameter"},
    "wlod": {"c_field": "wlod", "py_type": "float", "default": 0.0, "description": "Width parameter for stress effect"},
    "wlodku0": {"c_field": "wlodku0", "py_type": "float", "default": 0.0, "description": "Width parameter for u0 LOD effect"},
    "wlodvth": {"c_field": "wlodvth", "py_type": "float", "default": 0.0, "description": "Width parameter for vth LOD effect"},
    "wlp": {"c_field": "wlp", "py_type": "float", "default": 0.0, "description": "Width dependence of lp"},
    "wlpe0": {"c_field": "wlpe0", "py_type": "float", "default": 0.0, "description": "Width dependence of lpe0"},
    "wlpeb": {"c_field": "wlpeb", "py_type": "float", "default": 0.0, "description": "Width dependence of lpeb"},
    "wmax": {"c_field": "wmax", "py_type": "float", "default": 1.0, "description": "Maximum width for the model"},
    "wmin": {"c_field": "wmin", "py_type": "float", "default": 0.0, "description": "Minimum width for the model"},
    "wminv": {"c_field": "wminv", "py_type": "float", "default": 0.0, "description": "Width dependence of minv"},
    "wminvcv": {"c_field": "wminvcv", "py_type": "float", "default": 0.0, "description": "Width dependence of minvcv"},
    "wmoin": {"c_field": "wmoin", "py_type": "float", "default": 0.0, "description": "Width dependence of moin"},
    "wndep": {"c_field": "wndep", "py_type": "float", "default": 0.0, "description": "Width dependence of ndep"},
    "wnfactor": {"c_field": "wnfactor", "py_type": "float", "default": 0.0, "description": "Width dependence of nfactor"},
    "wngate": {"c_field": "wngate", "py_type": "float", "default": 0.0, "description": "Width dependence of ngate"},
    "wnigbacc": {"c_field": "wnigbacc", "py_type": "float", "default": 0.0, "description": "Width dependence of nigbacc"},
    "wnigbinv": {"c_field": "wnigbinv", "py_type": "float", "default": 0.0, "description": "Width dependence of nigbinv"},
    "wnigc": {"c_field": "wnigc", "py_type": "float", "default": 0.0, "description": "Width dependence of nigc"},
    "wnoff": {"c_field": "wnoff", "py_type": "float", "default": 0.0, "description": "Width dependence of noff"},
    "wnsd": {"c_field": "wnsd", "py_type": "float", "default": 0.0, "description": "Width dependence of nsd"},
    "wnsub": {"c_field": "wnsub", "py_type": "float", "default": 0.0, "description": "Width dependence of nsub"},
    "wntox": {"c_field": "wntox", "py_type": "float", "default": 0.0, "description": "Width dependence of ntox"},
    "wpclm": {"c_field": "wpclm", "py_type": "float", "default": 0.0, "description": "Width dependence of pclm"},
    "wpdiblc1": {"c_field": "wpdiblc1", "py_type": "float", "default": 0.0, "description": "Width dependence of pdiblc1"},
    "wpdiblc2": {"c_field": "wpdiblc2", "py_type": "float", "default": 0.0, "description": "Width dependence of pdiblc2"},
    "wpdiblcb": {"c_field": "wpdiblcb", "py_type": "float", "default": 0.0, "description": "Width dependence of pdiblcb"},
    "wpdits": {"c_field": "wpdits", "py_type": "float", "default": 0.0, "description": "Width dependence of pdits"},
    "wpditsd": {"c_field": "wpditsd", "py_type": "float", "default": 0.0, "description": "Width dependence of pditsd"},
    "wpemod": {"c_field": "wpemod", "py_type": "float", "default": 0.0, "description": " Flag for WPE model (WPEMOD=1 to activate this model) "},
    "wphin": {"c_field": "wphin", "py_type": "float", "default": 0.0, "description": "Width dependence of phin"},
    "wpigcd": {"c_field": "wpigcd", "py_type": "float", "default": 0.0, "description": "Width dependence for pigcd"},
    "wpoxedge": {"c_field": "wpoxedge", "py_type": "float", "default": 0.0, "description": "Width dependence for poxedge"},
    "wprt": {"c_field": "wprt", "py_type": "float", "default": 0.0, "description": "Width dependence of prt"},
    "wprwb": {"c_field": "wprwb", "py_type": "float", "default": 0.0, "description": "Width dependence of prwb "},
    "wprwg": {"c_field": "wprwg", "py_type": "float", "default": 0.0, "description": "Width dependence of prwg "},
    "wpscbe1": {"c_field": "wpscbe1", "py_type": "float", "default": 0.0, "description": "Width dependence of pscbe1"},
    "wpscbe2": {"c_field": "wpscbe2", "py_type": "float", "default": 0.0, "description": "Width dependence of pscbe2"},
    "wpvag": {"c_field": "wpvag", "py_type": "float", "default": 0.0, "description": "Width dependence of pvag"},
    "wr": {"c_field": "wr", "py_type": "float", "default": 1.0, "description": "Width dependence of rds"},
    "wrdsw": {"c_field": "wrdsw", "py_type": "float", "default": 0.0, "description": "Width dependence of rdsw "},
    "wrdw": {"c_field": "wrdw", "py_type": "float", "default": 0.0, "description": "Width dependence of rdw"},
    "wrgidl": {"c_field": "wrgidl", "py_type": "float", "default": 0.0, "description": "Width dependence of rgidl"},
    "wrgisl": {"c_field": "wrgisl", "py_type": "float", "default": ("ref", "wrgidl"), "description": "Width dependence of rgisl"},
    "wrsw": {"c_field": "wrsw", "py_type": "float", "default": 0.0, "description": "Width dependence of rsw"},
    "wteta0": {"c_field": "wteta0", "py_type": "float", "default": 0.0, "description": "Width dependence of teta0"},
    "wtnfactor": {"c_field": "wtnfactor", "py_type": "float", "default": 0.0, "description": "Width dependence of tnfactor"},
    "wtvfbsdoff": {"c_field": "wtvfbsdoff", "py_type": "float", "default": 0.0, "description": "Width dependence of tvfbsdoff"},
    "wtvoff": {"c_field": "wtvoff", "py_type": "float", "default": 0.0, "description": "Width dependence of tvoff"},
    "wtvoffcv": {"c_field": "wtvoffcv", "py_type": "float", "default": 0.0, "description": "Width dependence of tvoffcv"},
    "wu0": {"c_field": "wu0", "py_type": "float", "default": 0.0, "description": "Width dependence of u0"},
    "wua": {"c_field": "wua", "py_type": "float", "default": 0.0, "description": "Width dependence of ua"},
    "wua1": {"c_field": "wua1", "py_type": "float", "default": 0.0, "description": "Width dependence of ua1"},
    "wub": {"c_field": "wub", "py_type": "float", "default": 0.0, "description": "Width dependence of ub"},
    "wub1": {"c_field": "wub1", "py_type": "float", "default": 0.0, "description": "Width dependence of ub1"},
    "wuc": {"c_field": "wuc", "py_type": "float", "default": 0.0, "description": "Width dependence of uc"},
    "wuc1": {"c_field": "wuc1", "py_type": "float", "default": 0.0, "description": "Width dependence of uc1"},
    "wucs": {"c_field": "wucs", "py_type": "float", "default": 0.0, "description": "Width dependence of ucs"},
    "wucste": {"c_field": "wucste", "py_type": "float", "default": 0.0, "description": "Width dependence of ucste"},
    "wud": {"c_field": "wud", "py_type": "float", "default": 0.0, "description": "Width dependence of ud"},
    "wud1": {"c_field": "wud1", "py_type": "float", "default": 0.0, "description": "Width dependence of ud1"},
    "wup": {"c_field": "wup", "py_type": "float", "default": 0.0, "description": "Width dependence of up"},
    "wute": {"c_field": "wute", "py_type": "float", "default": 0.0, "description": "Width dependence of ute"},
    "wvbm": {"c_field": "wvbm", "py_type": "float", "default": 0.0, "description": "Width dependence of vbm"},
    "wvbx": {"c_field": "wvbx", "py_type": "float", "default": 0.0, "description": "Width dependence of vbx"},
    "wvfb": {"c_field": "wvfb", "py_type": "float", "default": 0.0, "description": "Width dependence of vfb"},
    "wvfbcv": {"c_field": "wvfbcv", "py_type": "float", "default": 0.0, "description": "Width dependence of vfbcv"},
    "wvfbsdoff": {"c_field": "wvfbsdoff", "py_type": "float", "default": 0.0, "description": "Width dependence of vfbsdoff"},
    "wvoff": {"c_field": "wvoff", "py_type": "float", "default": 0.0, "description": "Width dependence of voff"},
    "wvoffcv": {"c_field": "wvoffcv", "py_type": "float", "default": 0.0, "description": "Width dependence of voffcv"},
    "wvsat": {"c_field": "wvsat", "py_type": "float", "default": 0.0, "description": "Width dependence of vsat"},
    "wvth0": {"c_field": "wvth0", "py_type": "float", "default": 0.0, "description": "Width dependence of vto"},
    "wvtl": {"c_field": "wvtl", "py_type": "float", "default": 0.0, "description": "Width dependence of vtl"},
    "ww": {"c_field": "ww", "py_type": "float", "default": 0.0, "description": "Width reduction parameter"},
    "ww0": {"c_field": "ww0", "py_type": "float", "default": 0.0, "description": "Width dependence of w0"},
    "wwc": {"c_field": "wwc", "py_type": "float", "default": ("ref", "ww"), "description": "Width reduction parameter for CV"},
    "wwl": {"c_field": "wwl", "py_type": "float", "default": 0.0, "description": "Width reduction parameter"},
    "wwlc": {"c_field": "wwlc", "py_type": "float", "default": ("ref", "wwl"), "description": "Width reduction parameter for CV"},
    "wwn": {"c_field": "wwn", "py_type": "float", "default": 1.0, "description": "Width reduction parameter"},
    "wwr": {"c_field": "wwr", "py_type": "float", "default": 0.0, "description": "Width dependence of wr"},
    "wxj": {"c_field": "wxj", "py_type": "float", "default": 0.0, "description": "Width dependence of xj"},
    "wxn": {"c_field": "wxn", "py_type": "float", "default": 0.0, "description": "Width dependence of xn"},
    "wxrcrg1": {"c_field": "wxrcrg1", "py_type": "float", "default": 0.0, "description": "Width dependence of xrcrg1"},
    "wxrcrg2": {"c_field": "wxrcrg2", "py_type": "float", "default": 0.0, "description": "Width dependence of xrcrg2"},
    "wxt": {"c_field": "wxt", "py_type": "float", "default": 0.0, "description": "Width dependence of xt"},
    "xgl": {"c_field": "xgl", "py_type": "float", "default": 0.0, "description": "Variation in Ldrawn"},
    "xgw": {"c_field": "xgw", "py_type": "float", "default": 0.0, "description": "Distance from gate contact center to device edge"},
    "xj": {"c_field": "xj", "py_type": "float", "default": 1.5e-07, "description": "Junction depth in meters"},
    "xjbvd": {"c_field": "xjbvd", "py_type": "float", "default": ("ref", "xjbvs"), "description": "Fitting parameter for drain diode breakdown current"},
    "xjbvs": {"c_field": "xjbvs", "py_type": "float", "default": 1.0, "description": "Fitting parameter for source diode breakdown current"},
    "xl": {"c_field": "xl", "py_type": "float", "default": 0.0, "description": "L offset for channel length due to mask/etch effect"},
    "xn": {"c_field": "xn", "py_type": "float", "default": 3.0, "description": " back scattering parameter"},
    "xpart": {"c_field": "xpart", "py_type": "float", "default": 0.0, "description": "Channel charge partitioning"},
    "xrcrg1": {"c_field": "xrcrg1", "py_type": "float", "default": 12.0, "description": "First fitting parameter the bias-dependent Rg"},
    "xrcrg2": {"c_field": "xrcrg2", "py_type": "float", "default": 1.0, "description": "Second fitting parameter the bias-dependent Rg"},
    "xt": {"c_field": "xt", "py_type": "float", "default": 1.55e-07, "description": "Doping depth"},
    "xtid": {"c_field": "xtid", "py_type": "float", "default": 0.0, "description": "Drainjunction current temperature exponent"},
    "xtis": {"c_field": "xtis", "py_type": "float", "default": 0.0, "description": "Source junction current temperature exponent"},
    "xtsd": {"c_field": "xtsd", "py_type": "float", "default": ("ref", "xtss"), "description": "Power dependence of JTSD on temperature"},
    "xtss": {"c_field": "xtss", "py_type": "float", "default": 0.02, "description": "Power dependence of JTSS on temperature"},
    "xtsswd": {"c_field": "xtsswd", "py_type": "float", "default": ("ref", "xtssws"), "description": "Power dependence of JTSSWD on temperature"},
    "xtsswgd": {"c_field": "xtsswgd", "py_type": "float", "default": ("ref", "xtsswgs"), "description": "Power dependence of JTSSWGD on temperature"},
    "xtsswgs": {"c_field": "xtsswgs", "py_type": "float", "default": 0.02, "description": "Power dependence of JTSSWGS on temperature"},
    "xtssws": {"c_field": "xtssws", "py_type": "float", "default": 0.02, "description": "Power dependence of JTSSWS on temperature"},
    "xw": {"c_field": "xw", "py_type": "float", "default": 0.0, "description": "W offset for channel width due to mask/etch effect"},
}

ALIASES: dict[str, str] = {
    "lvtho": "lvth0",
    "pvtho": "pvth0",
    "vtho": "vth0",
    "wvtho": "wvth0",
    # Legacy SPICE BSIM4 short-name: `Tox=` is equivalent to `Toxe=`.
    # Without this alias, oracle smoke-test cards using `Tox=4n` silently
    # default to toxe=3e-9 in our model, causing ~40% Ids mismatch.
    "tox": "toxe",
}

```


=== FILE: temp.py (15704 chars) ===
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


=== FILE: vth_dibl_summary.json (482 chars) ===
```json
{
  "vds_0.05": {
    "Vth_ngspice": 0.5802127663120142,
    "Vth_pyport": 0.5217037466042274,
    "diff": -0.05850901970778688
  },
  "vds_0.5": {
    "Vth_ngspice": 0.572403338230139,
    "Vth_pyport": 0.512864729693589,
    "diff": -0.05953860853655002
  },
  "vds_2.0": {
    "Vth_ngspice": 0.5680161032442167,
    "Vth_pyport": 0.5082362367155965,
    "diff": -0.05977986652862022
  },
  "DIBL_ngspice_mV_per_V": 6.254699009126945,
  "DIBL_pyport_mV_per_V": 6.906415327503015
}
```


=== FILE: z91f_validate_with_sebas_params.py (14935 chars) ===
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


=== FILE: z91j_ngspice_isolated_m2.py (7009 chars) ===
```python
"""z91j — ngspice cross-validation on isolated M2.

Tests whether our PyTorch BSIM4 port (compute_dc) reproduces ngspice's
Berkeley BSIM4 (level 14) on the SAME M2 card with body=GND.

If they match (log-RMSE < 0.3 dec): our compute_dc is faithful, residual
in z91g is cell-level wiring/BJT/body-coupling, not BSIM4 itself.
If they diverge: a compute_dc bug is part of the cell-level residual.

Single bias point: VG2 in {-0.1, 0.0, 0.2}, Vd ∈ [0, 2V], Vbs=0, geometry
matches z91g M2 (L = 10× Ln, W = Wn).
"""
from __future__ import annotations
import subprocess, tempfile, json, re
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z91j_ngspice_iso_m2"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data/sebas_2026_04_22"

from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.dc import compute_dc
from nsram.bsim4_port.geometry import Geometry
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig

# Re-use z91f's post-load patcher (parser drops + continuation lines on .param)
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "z91f_mod", ROOT / "scripts/z91f_validate_with_sebas_params.py")
z91f = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(z91f)


def make_ngspice_card_inline() -> str:
    """Return self-contained ngspice .model NMOS body using Sebas's M2 params.

    We bake in the n-variant numeric values (parser already resolved them
    via .param continuation patch in z91f.patch_model_values).
    """
    # Use our parser to extract resolved values, then emit them as a flat
    # .model card ngspice can ingest. ngspice supports 'level=14' BSIM4.
    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    m = BSIM4Model.from_spice(text_M2, model_type="nmos")
    z91f.patch_model_values(m, type_n=True)
    # Pull out a curated set of params we know matter
    keys = ["vth0", "k1", "k2", "k3", "k3b", "w0", "nlx", "dvt0", "dvt1",
            "dvt2", "dvt0w", "dvt1w", "dvt2w", "u0", "ua", "ub", "uc",
            "vsat", "a0", "ags", "a1", "a2", "b0", "b1", "keta", "voff",
            "nfactor", "cdsc", "cdscb", "cdscd", "cit", "eta0", "etab",
            "dsub", "pclm", "pdiblc1", "pdiblc2", "pdiblcb", "drout",
            "pscbe1", "pscbe2", "pvag", "delta", "rdsw", "prwg", "prwb",
            "wr", "alpha0", "alpha1", "beta0", "agidl", "bgidl", "cgidl",
            "egidl", "agisl", "bgisl", "cgisl", "egisl", "tox", "xj",
            "nsub", "ngate", "ndep", "nsd", "lint", "wint", "xl", "xw",
            "rsh", "rdswmin", "rsw", "rdw"]
    pieces = []
    for k in keys:
        try:
            v = m[k]
            if isinstance(v, (int, float)):
                pieces.append(f"{k}={v:g}")
        except Exception:
            pass
    # Build .model card with proper ngspice continuations
    lines = [".model NMOSSEB NMOS (level=14"]
    for i in range(0, len(pieces), 6):
        lines.append("+ " + " ".join(pieces[i:i+6]))
    lines.append(")")
    return "\n".join(lines)


def run_ngspice_id_vd(vg2: float, geom: Geometry) -> tuple[np.ndarray, np.ndarray]:
    card = make_ngspice_card_inline()
    cir_text = f"""* z91j — isolated M2 BSIM4 cross-validation
{card}
VG G 0 DC {vg2:g}
VS S 0 DC 0
VB B 0 DC 0
VD D 0 DC 0
M1 D G S B NMOSSEB L={geom.L:g} W={geom.W:g}
.options gmin=1e-15 reltol=1e-6 abstol=1e-14
.control
dc Vd 0 2 0.05
wrdata {{tmpfile}}.dat i(vd) v(d)
quit
.endc
.end
"""
    with tempfile.NamedTemporaryFile("w", suffix=".cir", delete=False) as f:
        cir_text = cir_text.replace("{tmpfile}", f.name)
        f.write(cir_text)
        cir = f.name
    res = subprocess.run(["ngspice", "-b", cir], capture_output=True,
                         text=True, timeout=60)
    if not Path(cir + ".dat").exists():
        print("[z91j] ngspice stderr:", res.stderr[-500:])
        print("[z91j] ngspice stdout:", res.stdout[-500:])
        return np.array([]), np.array([])
    data = np.loadtxt(cir + ".dat")
    # ngspice wrdata cols: 0=sweep_x, 1=i(vd) real, 2=v(d) real
    Vd = data[:, 2]; Id = -data[:, 1]   # i(vd) negative-into-source
    return Vd, Id


def run_pyport_id_vd(vg2: float, geom: Geometry, model: BSIM4Model,
                      Vd_arr: np.ndarray) -> np.ndarray:
    sd = compute_size_dep(model, geom, T_C=27.0)
    Vd = torch.tensor(Vd_arr, dtype=torch.float64)
    out = compute_dc(model=model, sd=sd,
                     Vds=Vd, Vgs=torch.full_like(Vd, vg2),
                     Vbs=torch.zeros_like(Vd))
    return out.Ids.abs().numpy()


def main():
    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    model = BSIM4Model.from_spice(text_M2, model_type="nmos")
    z91f.patch_model_values(model, type_n=True)
    cfg = NSRAMCell2TConfig()
    geom = Geometry(L=cfg.Ln * cfg.M2_length_factor, W=cfg.Wn)
    print(f"[z91j] M2 geom: L={geom.L:g} W={geom.W:g}")

    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, vg2 in zip(axes, [-0.1, 0.0, 0.2]):
        Vd_ng, Id_ng = run_ngspice_id_vd(vg2, geom)
        if len(Vd_ng) == 0:
            print(f"[z91j] VG2={vg2}: ngspice failed")
            continue
        Id_py = run_pyport_id_vd(vg2, geom, model, Vd_ng)
        eps = 1e-15
        log_p = np.log10(np.abs(Id_py) + eps)
        log_n = np.log10(np.abs(Id_ng) + eps)
        rmse = float(np.sqrt(np.mean((log_p - log_n) ** 2)))
        print(f"[z91j] VG2={vg2:+.2f}  log-RMSE = {rmse:.3f}  "
              f"Id_py range [{Id_py.min():.2e}, {Id_py.max():.2e}]  "
              f"Id_ng range [{Id_ng.min():.2e}, {Id_ng.max():.2e}]")
        results[f"vg2={vg2}"] = {
            "log_rmse": rmse,
            "Vd": Vd_ng.tolist(),
            "Id_ngspice": Id_ng.tolist(),
            "Id_pyport": Id_py.tolist(),
        }
        ax.semilogy(Vd_ng, np.abs(Id_ng), "k-", label="ngspice")
        ax.semilogy(Vd_ng, Id_py, "r--", label="pyport")
        ax.set_title(f"VG2={vg2}  log-RMSE={rmse:.2f}")
        ax.set_xlabel("Vd"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_ylim(1e-13, 1e-3)
    axes[0].set_ylabel("|Id| [A]")
    fig.suptitle("z91j — isolated M2 BSIM4 cross-validation (ngspice vs pyport)")
    fig.tight_layout()
    fig.savefig(OUT / "iso_m2.png", dpi=140)
    plt.close(fig)
    rmses = [r["log_rmse"] for r in results.values()]
    summary = {"n": len(rmses), "median_log_rmse": float(np.median(rmses)) if rmses else float("nan"),
               "max_log_rmse": float(max(rmses)) if rmses else float("nan")}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "details.json").write_text(json.dumps(results, indent=2))
    print(f"[z91j] median log-RMSE = {summary['median_log_rmse']:.3f}, "
          f"max = {summary['max_log_rmse']:.3f}")


if __name__ == "__main__":
    main()

```


=== FILE: z91k_subthreshold_slope.py (5967 chars) ===
```python
"""z91k — subthreshold-slope diagnostic, isolated M2 BSIM4.

A.5.a: z91j showed pyport vs ngspice diverges by ~1 dec median on same
M2 card (subthreshold under, above-Vt over — polarity flip around Vth).
This tests whether the subthreshold-slope `n` is wrong.

Method: hold Vds=0.5V, sweep Vgs ∈ [-0.2, 0.8] at ΔVgs=0.025. Compute
log10|Id|(Vgs) for both engines. Subthreshold slope S = dVgs/d(log10 Id)
extracted by linear fit on the 1e-12 → 1e-9 decade.

S_theory at room T = n × ln(10) × kT/q ≈ n × 60 mV/dec.
S_perfect_MOS = 60 mV/dec at n=1.
Sebas's M2 card has nfactor=1.58 → expected S ≈ 95 mV/dec.

If our S << ngspice's S: our `n` is too small (subthreshold too steep
→ underpredicts Id at low Vg).
If our S >> ngspice's S: our `n` is too large (too lazy slope → over).
"""
from __future__ import annotations
import subprocess, tempfile, json
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z91k_subthreshold_slope"
OUT.mkdir(parents=True, exist_ok=True)

# Reuse z91j's helpers
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "z91j_mod", ROOT / "scripts/z91j_ngspice_isolated_m2.py")
z91j = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(z91j)

from nsram.bsim4_port.geometry import Geometry
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.dc import compute_dc
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.model_card import BSIM4Model

DATA = ROOT / "data/sebas_2026_04_22"


def run_ngspice_id_vgs(vd: float, geom: Geometry,
                        vgs_arr: np.ndarray) -> np.ndarray:
    card = z91j.make_ngspice_card_inline()
    vgs_lines = "\n".join(f"VG_{i} G_{i} 0 DC {v:g}" for i, v in enumerate(vgs_arr))
    # Single Vgs sweep via .dc on the gate voltage source
    cir_text = f"""* z91k Id-Vgs
{card}
VD D 0 DC {vd:g}
VG G 0 DC 0
VS S 0 DC 0
VB B 0 DC 0
M1 D G S B NMOSSEB L={geom.L:g} W={geom.W:g}
.options gmin=1e-15 reltol=1e-6 abstol=1e-16
.control
dc Vg {vgs_arr.min():g} {vgs_arr.max():g} {(vgs_arr[1]-vgs_arr[0]):g}
wrdata {{tmpfile}}.dat i(vd) v(g)
quit
.endc
.end
"""
    with tempfile.NamedTemporaryFile("w", suffix=".cir", delete=False) as f:
        cir_text = cir_text.replace("{tmpfile}", f.name)
        f.write(cir_text); cir = f.name
    res = subprocess.run(["ngspice", "-b", cir], capture_output=True,
                         text=True, timeout=60)
    if not Path(cir + ".dat").exists():
        print("[z91k] ngspice failed:", res.stderr[-300:])
        return np.array([])
    data = np.loadtxt(cir + ".dat")
    return -data[:, 1]   # i(vd) sign-flip


def run_pyport_id_vgs(vd: float, geom: Geometry, model: BSIM4Model,
                       vgs_arr: np.ndarray) -> np.ndarray:
    sd = compute_size_dep(model, geom, T_C=27.0)
    Vg = torch.tensor(vgs_arr, dtype=torch.float64)
    out = compute_dc(model=model, sd=sd,
                     Vgs=Vg, Vds=torch.full_like(Vg, vd),
                     Vbs=torch.zeros_like(Vg))
    return out.Ids.abs().numpy()


def extract_S(vgs: np.ndarray, Id: np.ndarray,
               id_lo=1e-12, id_hi=1e-9) -> float:
    """Subthreshold slope mV/dec by linear fit on log10(Id) range."""
    Id = np.maximum(np.abs(Id), 1e-30)
    mask = (Id > id_lo) & (Id < id_hi)
    if mask.sum() < 3:
        return float("nan")
    log_id = np.log10(Id[mask])
    v = vgs[mask]
    # Vgs vs log10(Id), slope = dVgs/d(log10 Id) → S in V/dec
    slope, _ = np.polyfit(log_id, v, 1)
    return float(slope * 1000.0)   # mV/dec


def main():
    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    model = BSIM4Model.from_spice(text_M2, model_type="nmos")
    z91j.z91f.patch_model_values(model, type_n=True)
    cfg = NSRAMCell2TConfig()
    geom = Geometry(L=cfg.Ln * cfg.M2_length_factor, W=cfg.Wn)
    nfactor = model.get("nfactor")
    print(f"[z91k] M2 card: nfactor={nfactor}, etab={model.get('etab')}, "
          f"cdsc={model.get('cdsc'):g}, cdscb={model.get('cdscb'):g}")
    print(f"[z91k] expected S ≈ {60 * (1 + nfactor):.1f} mV/dec  "
          f"(rough — ignores cdsc, cdscb)")

    vgs_arr = np.arange(-0.2, 0.81, 0.025)
    Vd = 0.5

    Id_ng = run_ngspice_id_vgs(Vd, geom, vgs_arr)
    if len(Id_ng) == 0:
        print("[z91k] ngspice failed, abort")
        return
    Id_py = run_pyport_id_vgs(Vd, geom, model, vgs_arr)

    S_ng = extract_S(vgs_arr, Id_ng)
    S_py = extract_S(vgs_arr, Id_py)
    print(f"[z91k] Vd={Vd}V")
    print(f"[z91k] ngspice S = {S_ng:.2f} mV/dec")
    print(f"[z91k] pyport  S = {S_py:.2f} mV/dec")
    print(f"[z91k] diff    = {S_py - S_ng:+.2f} mV/dec  ({(S_py-S_ng)/S_ng*100:+.1f}%)")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(vgs_arr, np.abs(Id_ng) + 1e-30, "k-", label=f"ngspice (S={S_ng:.1f} mV/dec)")
    ax.semilogy(vgs_arr, np.abs(Id_py) + 1e-30, "r--", label=f"pyport (S={S_py:.1f} mV/dec)")
    ax.set_xlabel("Vgs [V]")
    ax.set_ylabel("|Id| [A]")
    ax.set_title(f"z91k subthreshold slope — isolated M2, Vd={Vd}V, body=GND")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(1e-15, 1e-3)
    fig.tight_layout()
    fig.savefig(OUT / "id_vgs.png", dpi=140)

    summary = {
        "Vd": Vd,
        "S_ngspice_mV_per_dec": S_ng,
        "S_pyport_mV_per_dec": S_py,
        "S_diff": S_py - S_ng,
        "card_nfactor": nfactor,
        "card_etab": model.get("etab"),
        "card_cdsc": model.get("cdsc"),
        "card_cdscb": model.get("cdscb"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savetxt(OUT / "id_vgs.csv",
               np.column_stack([vgs_arr, Id_ng, Id_py]),
               header="Vgs,Id_ngspice,Id_pyport", delimiter=",", comments="")
    print(f"[z91k] saved {OUT}/id_vgs.png + summary.json")


if __name__ == "__main__":
    main()

```


=== FILE: z91l_vth_dibl.py (5412 chars) ===
```python
"""z91l — extract Vth and DIBL on isolated M2, ngspice vs pyport.

A.5.b: z91j showed pyport disagrees with ngspice by ~1 dec on isolated
M2 (subthreshold under, near-on over — polarity flip). z91k showed
subthreshold-slope `n` matches (72.5 vs 76.5 mV/dec). So the bug is
likely in Vth (DIBL/SCE).

Method: at Vds ∈ {0.5, 2.0}, run Id-Vgs sweep, extract Vth via the
constant-current criterion: Vth = Vgs at Id = (W/L) × 1e-7 A (standard
SPICE convention). DIBL = (Vth_low − Vth_high) / ΔVds in V/V.

If pyport and ngspice agree at Vds=0.5 but diverge at Vds=2.0, the bug
is in our DIBL term (pdiblc1 / pdiblc2 / pdiblcb / drout / dsub).
"""
from __future__ import annotations
import json, subprocess, tempfile
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z91l_vth_dibl"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "z91j_mod", ROOT / "scripts/z91j_ngspice_isolated_m2.py")
z91j = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(z91j)

_spec_k = importlib.util.spec_from_file_location(
    "z91k_mod", ROOT / "scripts/z91k_subthreshold_slope.py")
z91k = importlib.util.module_from_spec(_spec_k)
_spec_k.loader.exec_module(z91k)

from nsram.bsim4_port.geometry import Geometry
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.model_card import BSIM4Model

DATA = ROOT / "data/sebas_2026_04_22"


def vth_constant_current(vgs: np.ndarray, Id: np.ndarray,
                          geom: Geometry) -> float:
    """Vth = Vgs at Id_target = (W/L) * 1e-7 A. Linear interpolation."""
    Id_target = (geom.W / geom.L) * 1e-7
    Id = np.maximum(np.abs(Id), 1e-30)
    if Id.max() < Id_target or Id.min() > Id_target:
        return float("nan")
    # find first crossing
    log_id = np.log10(Id); log_t = np.log10(Id_target)
    for i in range(len(vgs) - 1):
        if (log_id[i] - log_t) * (log_id[i + 1] - log_t) <= 0:
            f = (log_t - log_id[i]) / (log_id[i + 1] - log_id[i] + 1e-30)
            return float(vgs[i] + f * (vgs[i + 1] - vgs[i]))
    return float("nan")


def main():
    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    model = BSIM4Model.from_spice(text_M2, model_type="nmos")
    z91j.z91f.patch_model_values(model, type_n=True)
    cfg = NSRAMCell2TConfig()
    geom = Geometry(L=cfg.Ln * cfg.M2_length_factor, W=cfg.Wn)
    print(f"[z91l] M2 geom: L={geom.L:g}  W={geom.W:g}  Id_target={(geom.W/geom.L)*1e-7:.3e}")
    print(f"[z91l] DIBL params: pdiblc1={model.get('pdiblc1'):g}, "
          f"pdiblc2={model.get('pdiblc2'):g}, pdiblcb={model.get('pdiblcb'):g}, "
          f"drout={model.get('drout'):g}, dsub={model.get('dsub'):g}")

    vgs_arr = np.arange(0.0, 1.21, 0.025)
    res = {}
    for Vds in [0.05, 0.5, 2.0]:
        Id_ng = z91k.run_ngspice_id_vgs(Vds, geom, vgs_arr)
        Id_py = z91k.run_pyport_id_vgs(Vds, geom, model, vgs_arr)
        Vth_ng = vth_constant_current(vgs_arr, Id_ng, geom)
        Vth_py = vth_constant_current(vgs_arr, Id_py, geom)
        print(f"[z91l] Vds={Vds:>4.2f}V  Vth_ng={Vth_ng:+.4f}V  Vth_py={Vth_py:+.4f}V  "
              f"diff={Vth_py - Vth_ng:+.4f}V")
        res[f"vds_{Vds}"] = {
            "Vth_ngspice": Vth_ng,
            "Vth_pyport": Vth_py,
            "diff": (Vth_py - Vth_ng) if not (np.isnan(Vth_ng) or np.isnan(Vth_py)) else None,
        }

    if all(not np.isnan(res[k].get("diff") or float("nan")) for k in res):
        # DIBL = -dVth/dVds (V/V), should be POSITIVE for n-MOS
        v_low = res["vds_0.05"]
        v_high = res["vds_2.0"]
        DIBL_ng = -(v_high["Vth_ngspice"] - v_low["Vth_ngspice"]) / (2.0 - 0.05)
        DIBL_py = -(v_high["Vth_pyport"]   - v_low["Vth_pyport"])  / (2.0 - 0.05)
        print(f"[z91l] DIBL_ng = {DIBL_ng*1000:+.1f} mV/V")
        print(f"[z91l] DIBL_py = {DIBL_py*1000:+.1f} mV/V")
        print(f"[z91l] diff    = {(DIBL_py - DIBL_ng)*1000:+.1f} mV/V")
        res["DIBL_ngspice_mV_per_V"] = DIBL_ng * 1000
        res["DIBL_pyport_mV_per_V"]  = DIBL_py * 1000

    (OUT / "summary.json").write_text(json.dumps(res, indent=2))

    # Plot id-vgs at three Vds for both engines
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, Vds in zip(axes, [0.05, 0.5, 2.0]):
        Id_ng = z91k.run_ngspice_id_vgs(Vds, geom, vgs_arr)
        Id_py = z91k.run_pyport_id_vgs(Vds, geom, model, vgs_arr)
        ax.semilogy(vgs_arr, np.abs(Id_ng) + 1e-30, "k-", label="ngspice")
        ax.semilogy(vgs_arr, np.abs(Id_py) + 1e-30, "r--", label="pyport")
        Vth_ng = vth_constant_current(vgs_arr, Id_ng, geom)
        Vth_py = vth_constant_current(vgs_arr, Id_py, geom)
        ax.axvline(Vth_ng, color="k", ls=":", alpha=0.4)
        ax.axvline(Vth_py, color="r", ls=":", alpha=0.4)
        ax.set_title(f"Vds={Vds}V  Vth_ng={Vth_ng:.3f}  Vth_py={Vth_py:.3f}")
        ax.set_xlabel("Vgs"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_ylim(1e-15, 1e-3)
    axes[0].set_ylabel("|Id| [A]")
    fig.suptitle(f"z91l Vth/DIBL — isolated M2, body=GND")
    fig.tight_layout()
    fig.savefig(OUT / "vth_dibl.png", dpi=140)
    print(f"[z91l] saved {OUT}/vth_dibl.png + summary.json")


if __name__ == "__main__":
    main()

```
