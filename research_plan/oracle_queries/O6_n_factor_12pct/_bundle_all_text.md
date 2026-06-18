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


=== FILE: temp.py (16018 chars) ===
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


=== FILE: z91k_summary.json (230 chars) ===
```json
{
  "Vd": 0.5,
  "S_ngspice_mV_per_dec": 72.47616926031559,
  "S_pyport_mV_per_dec": 76.22071097465532,
  "S_diff": 3.744541714339732,
  "card_nfactor": 1.58,
  "card_etab": -0.086777,
  "card_cdsc": 0.00024,
  "card_cdscb": 0.0
}
```


=== FILE: z91l_summary.json (484 chars) ===
```json
{
  "vds_0.05": {
    "Vth_ngspice": 0.5802127663120142,
    "Vth_pyport": 0.5242031870718897,
    "diff": -0.05600957924012451
  },
  "vds_0.5": {
    "Vth_ngspice": 0.572403338230139,
    "Vth_pyport": 0.5154000112524555,
    "diff": -0.05700332697768351
  },
  "vds_2.0": {
    "Vth_ngspice": 0.5680161032442167,
    "Vth_pyport": 0.5108006033580188,
    "diff": -0.057215499886197896
  },
  "DIBL_ngspice_mV_per_V": 6.254699009126945,
  "DIBL_pyport_mV_per_V": 6.873119853267144
}
```


=== FILE: z91m_summary.json (6043 chars) ===
```json
{
  "Vds": 0.05,
  "vgs_array": [
    0.3,
    0.325,
    0.35000000000000003,
    0.37500000000000006,
    0.4000000000000001,
    0.4250000000000001,
    0.4500000000000001,
    0.47500000000000014,
    0.5000000000000002,
    0.5250000000000001,
    0.5500000000000003,
    0.5750000000000002,
    0.6000000000000003,
    0.6250000000000002,
    0.6500000000000004,
    0.6750000000000003,
    0.7000000000000004,
    0.7250000000000003,
    0.7500000000000004,
    0.7750000000000004,
    0.8000000000000005,
    0.8250000000000004
  ],
  "fields": {
    "Ids": [
      6.387757041358975e-11,
      1.3432464117625584e-10,
      2.777103918607634e-10,
      5.617408690481961e-10,
      1.1063039299550919e-09,
      2.112982981900667e-09,
      3.905488639112844e-09,
      6.985369705250488e-09,
      1.210453681157643e-08,
      2.0333383216460197e-08,
      3.3053372084181533e-08,
      5.176585146778936e-08,
      7.768726460480345e-08,
      1.1131898249073311e-07,
      1.5228775545112951e-07,
      1.9954074861731318e-07,
      2.5170651579264277e-07,
      3.074119318379344e-07,
      3.6546760133977936e-07,
      4.2493343253692316e-07,
      4.851101779541707e-07,
      5.454984253386333e-07
    ],
    "Vgsteff": [
      6.120804806718976e-05,
      0.00012856877326628493,
      0.00026521623528694973,
      0.0005341232470848009,
      0.0010433096013429104,
      0.0019638279276744603,
      0.0035429135337310272,
      0.006104734016780871,
      0.010034064209180257,
      0.015742038043182787,
      0.023614375349277746,
      0.033945592396469826,
      0.046874035338269476,
      0.06234466694982093,
      0.08012149305088936,
      0.09984624865651315,
      0.1211157022089895,
      0.14354704258616605,
      0.16681605409948272,
      0.19066939005260125,
      0.2149201638529173,
      0.23943610581750846
    ],
    "Vdseff": [
      0.02778068080455088,
      0.027806754203115867,
      0.0278595523253468,
      0.02796308474452483,
      0.028157789355076655,
      0.028505337585784902,
      0.02908822906228234,
      0.029998299532434537,
      0.031309743105350875,
      0.033038460183690097,
      0.035102725733806274,
      0.037317890985021765,
      0.039453936826330215,
      0.04132885328928155,
      0.04286446803577321,
      0.04407173941872261,
      0.04500330572829784,
      0.04571957316497744,
      0.0462732423216895,
      0.046705576752561456,
      0.047047397395779955,
      0.04732128472219799
    ],
    "Vth": [
      0.5863888229042562,
      0.5863888229042562,
      0.5863888229042562,
      0.5863888229042562,
      0.5863888229042562,
      0.5863888229042562,
      0.5863888229042562,
      0.5863888229042562,
      0.5863888229042562,
      0.5863888229042562,
      0.5863888229042562,
      0.5863888229042562,
      0.5863888229042562,
      0.5863888229042562,
      0.5863888229042562,
      0.5863888229042562,
      0.5863888229042562,
      0.5863888229042562,
      0.5863888229042562,
      0.5863888229042562,
      0.5863888229042562,
      0.5863888229042562
    ],
    "n": [
      1.2298055204938627,
      1.2298055204938627,
      1.2298055204938627,
      1.2298055204938627,
      1.2298055204938627,
      1.2298055204938627,
      1.2298055204938627,
      1.2298055204938627,
      1.2298055204938627,
      1.2298055204938627,
      1.2298055204938627,
      1.2298055204938627,
      1.2298055204938627,
      1.2298055204938627,
      1.2298055204938627,
      1.2298055204938627,
      1.2298055204938627,
      1.2298055204938627,
      1.2298055204938627,
      1.2298055204938627,
      1.2298055204938627,
      1.2298055204938627
    ],
    "Vgs_eff": [
      0.3,
      0.325,
      0.35000000000000003,
      0.37500000000000006,
      0.4000000000000001,
      0.4250000000000001,
      0.4500000000000001,
      0.47500000000000014,
      0.5000000000000002,
      0.5250000000000001,
      0.5500000000000003,
      0.5750000000000002,
      0.6000000000000003,
      0.6250000000000002,
      0.6500000000000004,
      0.6750000000000003,
      0.7000000000000004,
      0.7250000000000003,
      0.7500000000000004,
      0.7750000000000004,
      0.8000000000000005,
      0.8250000000000004
    ],
    "log10_Id": [
      -10.194651610630572,
      -9.871844310896103,
      -9.556407868743278,
      -9.25046397819537,
      -8.956125544849126,
      -8.675104000768982,
      -8.408324621199188,
      -8.155810603622635,
      -7.917051824481245,
      -7.691790354269215,
      -7.480784227468886,
      -7.285956638124943,
      -7.109650169940134,
      -6.953430771981181,
      -6.817335014285664,
      -6.699968402771411,
      -6.599105541955955,
      -6.512279279872034,
      -6.437151117154857,
      -6.371679098528966,
      -6.314159613473373,
      -6.263206498728672
    ],
    "log10_Id_ngspice": [
      -11.227202580365667,
      -10.876199445517736,
      -10.527485155795707,
      -10.182081455360569,
      -9.84137330501301,
      -9.507142889979741,
      -9.181512018801966,
      -8.866731546492934,
      -8.564818287147641,
      -8.277210941136397,
      -8.004803426484756,
      -7.748597116560035,
      -7.510589553715533,
      -7.293923045916509,
      -7.101723305965511,
      -6.935427188757968,
      -6.794072250446757,
      -6.674905510893375,
      -6.574458803784156,
      -6.489337026502458,
      -6.416587774564818,
      -6.353794745622512
    ],
    "dec_diff": [
      1.0325509697350945,
      1.0043551346216333,
      0.9710772870524291,
      0.931617477165199,
      0.8852477601638835,
      0.8320388892107591,
      0.7731873976027774,
      0.7109209428702989,
      0.6477664626663966,
      0.5854205868671825,
      0.5240191990158696,
      0.46264047843509193,
      0.40093938377539917,
      0.340492273935328,
      0.28438829167984725,
      0.23545878598655712,
      0.1949667084908029,
      0.16262623102134022,
      0.13730768662929815,
      0.11765792797349217,
      0.1024281610914457,
      0.09058824689383993
    ]
  }
}
```


=== FILE: z91m_vgsteff_inspect.py (5427 chars) ===
```python
"""z91m — instrument the Vgsteff/subthreshold bridge.

A.5.d: A9 reported pyport diverges from ngspice at Vds=0.05V — pyport
+0.92 dec high in deep subthreshold (Vgs=0.40), -0.44 dec low near-on
(Vgs=0.58). The phi fix only moved Vth gap by 3 mV. The bug is in
the Vgsteff bridge (dc.py:397-472) or its downstream Id computation.

This dumps every intermediate of the Vgsteff bridge at a bias sweep
around Vth at Vds=0.05V, plots them, identifies which quantity has a
discontinuity / sign flip / wrong scaling.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z91m_vgsteff_inspect"
OUT.mkdir(parents=True, exist_ok=True)

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


def main():
    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    model = BSIM4Model.from_spice(text_M2, model_type="nmos")
    z91j.z91f.patch_model_values(model, type_n=True)
    cfg = NSRAMCell2TConfig()
    geom = Geometry(L=cfg.Ln * cfg.M2_length_factor, W=cfg.Wn)
    sd = compute_size_dep(model, geom, T_C=27.0)
    print(f"[z91m] M2 geom: L={geom.L:g} W={geom.W:g}")
    print(f"[z91m] sd vth0_eff = {sd.scaled.get('vth0', float('nan'))}")
    print(f"[z91m] sd nfactor = {sd.scaled.get('nfactor', float('nan'))}")
    print(f"[z91m] sd k1 = {sd.scaled.get('k1', float('nan'))}")
    print(f"[z91m] sd voff = {sd.scaled.get('voff', float('nan'))}")
    print(f"[z91m] ctx phi = {getattr(sd.model_ctx, 'phi', 'n/a')}")

    Vds = 0.05
    vgs_arr = np.arange(0.30, 0.85, 0.025)

    # Patch compute_dc to capture intermediates: monkey-patch by
    # re-implementing the relevant bits. Easier: run compute_dc and then
    # re-derive Vgsteff from its intermediate fields if it exposes them.
    # compute_dc returns DCResult; check what it exposes.
    Vg = torch.tensor(vgs_arr, dtype=torch.float64)
    Vd_t = torch.full_like(Vg, Vds)
    Vb_t = torch.zeros_like(Vg)
    out = compute_dc(model=model, sd=sd, Vgs=Vg, Vds=Vd_t, Vbs=Vb_t)

    # Inspect what's available
    avail = [a for a in dir(out) if not a.startswith("_")]
    print(f"[z91m] DCResult fields: {avail}")

    # Pull out fields we know about
    fields = {}
    for name in ("Ids", "Vgsteff", "Vdseff", "Vth", "n", "Vgst",
                  "T10", "T9", "Vgs_eff", "Vbsh"):
        if hasattr(out, name):
            v = getattr(out, name)
            if isinstance(v, torch.Tensor) and v.numel() == len(vgs_arr):
                fields[name] = v.detach().numpy()

    print(f"[z91m] captured fields: {list(fields.keys())}")
    # Also compute Id
    Id_py = out.Ids.abs().numpy()
    fields["log10_Id"] = np.log10(np.maximum(Id_py, 1e-30))

    # ngspice for reference
    Id_ng = z91j.run_ngspice_id_vd if False else None
    # Use z91k's Id-Vgs (we want at fixed Vds=0.05V)
    import importlib.util as iu
    _sk = iu.spec_from_file_location("z91k", ROOT / "scripts/z91k_subthreshold_slope.py")
    z91k = iu.module_from_spec(_sk); _sk.loader.exec_module(z91k)
    Id_ng = z91k.run_ngspice_id_vgs(Vds, geom, vgs_arr)
    fields["log10_Id_ngspice"] = np.log10(np.maximum(np.abs(Id_ng), 1e-30))
    fields["dec_diff"] = fields["log10_Id"] - fields["log10_Id_ngspice"]

    # Print table at key biases
    print(f"\n[z91m]  Vgs    log10_Id_py  log10_Id_ng  diff(dec)  "
          + "  ".join(k for k in ("Vgsteff", "Vth", "n", "Vbsh") if k in fields))
    for i, vgs in enumerate(vgs_arr):
        row = [f"{vgs:5.2f}",
               f"{fields['log10_Id'][i]:11.3f}",
               f"{fields['log10_Id_ngspice'][i]:11.3f}",
               f"{fields['dec_diff'][i]:+9.3f}"]
        for k in ("Vgsteff", "Vth", "n", "Vbsh"):
            if k in fields:
                row.append(f"{fields[k][i]:9.4f}")
        print("  " + "  ".join(row))

    # Plot dec_diff vs Vgs
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    axes[0].plot(vgs_arr, fields["log10_Id"], "r-", label="pyport")
    axes[0].plot(vgs_arr, fields["log10_Id_ngspice"], "k-", label="ngspice")
    axes[0].set_ylabel("log10 |Id|"); axes[0].grid(alpha=0.3); axes[0].legend()
    axes[0].set_title(f"z91m Vgsteff bridge inspect, Vds={Vds}V, body=GND")
    axes[1].plot(vgs_arr, fields["dec_diff"], "b-", lw=1.5)
    axes[1].axhline(0, color="k", ls=":")
    axes[1].set_ylabel("py - ngspice (dec)"); axes[1].set_xlabel("Vgs")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "vgsteff_inspect.png", dpi=140)

    # JSON summary
    summary = {
        "Vds": Vds,
        "vgs_array": vgs_arr.tolist(),
        "fields": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in fields.items()},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[z91m] saved {OUT}/vgsteff_inspect.png + summary.json")


if __name__ == "__main__":
    main()

```
