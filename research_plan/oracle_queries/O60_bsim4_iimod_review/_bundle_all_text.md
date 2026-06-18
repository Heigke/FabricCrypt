# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: 01_LOG_tail.md (4435 chars) ===
```

R-9 instrument verdict: Iii IS routed correctly to Vb residual (lines 816,
825, 838, 849 of nsram_cell_2T.py). Hypothesis (b) FALSE.

REAL root cause: solver converges with Vb=2.0V (= Vd) at the test bias.
At this fixed point:
- Vbe = Vb − Vsint = 0.13V → NPN can't fire
- Vds − Vdseff = 0.095V, beta0=20 → T1·exp(−211) = 1e-94 → Iii = 7.5e-48 A
- Body pinned to drain because no body-pull-down path is active

z304's 0.99 dec succeeded because the AVALANCHE PATH (which we removed
per Sebas mail) PULLED Vb DOWN from Vd. That enabled:
- Vbe > 0.5V → NPN fires
- Vds − Vdseff > 0.3V → Iii non-zero → snapback loop closes

v5b regression: we removed avalanche without installing alternative
body-pull-down. Solver finds trivial Vb=Vd fixed-point. All "physics"
silently dies.

**Next experiment**: force Vb-initial-guess away from Vd, OR re-enable
ONE path that pulls Vb down (slightly leaky vnwell→Vb diode at low
saturation current), to bootstrap the solver into the right basin.

This is the deepest insight today: model isn't missing physics — it's
finding the WRONG fixed point because no path is mathematically forcing
Vb < Vd. Sebas's LTSpice probably bootstraps differently via
DC-sweep continuation; our pyport solves cold-start at one bias.

R-9 closes the structural-investigation arc. The remaining question is
NUMERICAL: how to ensure solver finds the physical (Vb < Vd) basin.

## 2026-05-13 20:47 — :47 idle — ACTIVE: z323_v, APU=44C

## 2026-05-13 21:13 — R-phase progress check
R-1..R-9 done. R-9 = ROOT CAUSE: solver finds Vb=Vd trivial fixed-point;
no body-pull-down path active in current pyport. R-10 NOT dispatched
(numerical solver fix: homotopy + weak vnwell→Vb seed leak to bootstrap
solver to physical Vb<Vd basin). User-gated.

## 2026-05-13 21:47 — :47 idle — idle, APU=38C

## 2026-05-13 21:47 — deep-dive 2h cron: 4A-E closed, R-1..R-9 closed, R-10 user-gated, no auto-launch

## 2026-05-13 21:50 — 4h campaign check
P1-P8 master fix: CLOSED. R-1..R-9 topology rebuild: CLOSED with R-9 root cause.
R-10 numerical solver fix: pending user approval. No new ALERT.

## 2026-05-13 22:13 — R-phase progress check
R-1..R-9 closed. R-9 root cause: Vb pinned to Vd at solver fix-point.
R-10 (numerical solver homotopy fix) NOT dispatched, user-gated.
No ALERT.

## 2026-05-13 22:30 — 3h campaign cron: idle, APU=37C, R-10 user-gated, no auto-launch

## 2026-05-13 22:32 — R-10 numerical solver fix DISPATCHED

Pre-reg locked gates:
- INFRA: solver converges to Vb < Vd at V_G1=0.6, V_G2=0.20, V_d=2.0
- PASS: cell-wide median log-RMSE < 0.95 dec (beats z304 0.99)
- AMBITIOUS: < 0.5 dec
- DIAGNOSTIC: V_G1=0.2 < 2.5 dec AND V_G1=0.6 < 0.7 dec (per-branch parity)

## 2026-05-13 22:50 — R-10 ALL strategies FAILED — deeper bug

S1+S2+S3 all give Vb=2.0 (=Vd). Trivial fix-point is MATHEMATICALLY VALID
because residual at Vb=Vd is ~0 (all currents 1e-48). Initial-guess
doesn't help: solver finds the only equation-satisfying point.

Real issue: M1 at V_G1=0.6, V_d=2.0V should generate Iii ~1e-21 A by
BSIM4 IIMOD formula. Our pyport computes 1e-48. Either:
(a) BSIM4 IIMOD formula has implementation bug in pyport
(b) M1's Iii is routed to M1's own body, not the shared/floating Vb
(c) Pyport's body-node connectivity is wrong (M1 body and shared body are
    separated when they should be merged)

Next R-phase = code-level debug of BSIM4 _eval_mosfet → Iii computation
+ trace where M1's Iii goes in residual stamp.

Snapback graphs blocked until M1 Iii physically lives.

NO automatic graphs generated this run — R-10 halted at infra-gate.

## 2026-05-13 22:53 — 6h track audit
Phase A: A.1/A.2 ✓, A.3/A.4 deferred (model-blocked). 2/4.
Phase B: DS-N1✓ DS-N2✓ DS-N3✓(AMBITIOUS+NIST) DS-N5✓(LOCKED 83.86% n=10) DS-N4 in_progress DS-N6✓(FAIL). 5/6.
Phase C: 4A-E✓ brief compiled.
Topology campaign: R-1..R-9 closed (root cause: M1 Iii=1e-48 vs physics 1e-21).
R-10 solver-fix FAILED at infra. R-11 (BSIM4 Iii deep-trace) user-gated.
v4.4 status: HDC+RNG headlines locked, model-rebuild deeper than expected.

## 2026-05-13 23:00 — R-11 BSIM4 IIMOD deep-trace dispatched

Bug hypothesis: pyport's compute_iimpact gives 1e-48 A at M1 strong sat;
physics expects ~1e-21 A (27 OoM off). Subagent will:
- Read leak.py compute_iimpact
- Instrument at V_G1=0.6 V_d=2.0
- Compare to hand-calc step-by-step  
- Identify where divergence happens
- Fix and verify Iii > 1e-25 A
- Re-run V_G1=0.6 to see if cell-wide moves

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


=== FILE: leak.py (18552 chars) ===
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


=== FILE: nsram_cell_2T.py (84518 chars) ===
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
    # D2 fix (2026-05-13, R_deep_A audit): default OFF. LTSpice
    # `2tnsram_simple.asc` has ZERO explicit diodes. The N-well/body
    # junction is INSIDE BSIM4 PTM130bulkNSRAM (dnwell, source/drain-bulk
    # diodes). Explicit `use_well_diode=True` triple-counts the junction
    # (BSIM4 internal + explicit well_diode + body_pdiode). Rely on BSIM4
    # internal junction; expose vnwell knob only through model-card params.
    use_well_diode: bool = False
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
    # A.1.u: Wire M2 body to GND (Sebas's nmos4 with body unconnected
    # defaults to GND in LTSpice) instead of to the floating Vb. Default
    # True per oracle consensus + visual scan of his .asc deck.
    m2_body_gnd: bool = True
    # A.3.d: M1 body-diode scale factor. M1's body-source/body-drain diodes
    # clamp Vb at the Si forward voltage (~0.5V) before the parasitic NPN
    # can light at ~0.6-0.7V. Sebas's measured snap requires Vb past ~0.6V,
    # implying his deck either omits these diodes or has tiny jss.
    m1_diode_scale: float = 1.0
    # A.10 (2026-05-02 / refined 2026-05-03 / re-labelled PMP-6 2026-05-11):
    # Sebas's missing **parasitic N-well diode** at the floating P-body
    # (5×4.4 µm² = 22 µm²). Anode = floating P-body, cathode = V_Nwell
    # (Sebas slide 21 "Dynamic response with p-diode update"; O42 oracles
    # unanimous on topology). Voltage across the diode is V_b − V_Nwell.
    # SPICE card lives in `data/sebas_2026_05_02/pdiode.txt`. NOTE: prior
    # commentary called this a "P-body diode" with implicit cathode = GND
    # / source — that was the wrong topology label. The SPICE card itself
    # is for the N-well diode and was correct; only the narrative was
    # wrong. Field prefix `body_pdiode_*` retained for back-compat; read
    # as "body-side parasitic diode" not "P-body diode".
    # Default still OFF — the production fit (z91g 33-bias) has a
    # 1.00-decade median without it; turning it on is a forward action
    # for the post-tape-out validation phase, not a Phase-A re-fit.
    # Cathode candidate set: 'off' / 'vnwell' / 'gnd' / 'sint'.
    # The CORRECT cathode for the parasitic N-well diode is 'vnwell';
    # 'gnd' / 'sint' are kept as ablation knobs only.
    body_pdiode_to: str = "off"           # cathode node
    body_pdiode_area: float = 22e-12      # 5 µm × 4.4 µm (Sebas)
    # Sebas's pdiode.txt (LTspice diode model, level=1):
    #   is = 5.3675e-7 A,  n = 1.0535,  cj = 7.3279e-4 F/m²,
    #   vj = 0.21918,  m = 0.24097,  rs = 7.4e-8.
    # Js (per area) = is / area = 5.3675e-7 / 22e-12 = 2.44e4 A/m². Wait —
    # this is unexpectedly large. Sebas's `is` is the *total* saturation
    # current of the modelled junction (not per area). With the 22 µm²
    # area the per-area value is 24400 A/m², which is unphysical for a
    # well/body junction (typical ~1e-4 A/m²). Reading Sebas's card more
    # carefully: `is = 5.3675e-7` is the total — so we should NOT multiply
    # by area downstream. Implementation in `_residuals` already multiplies
    # `body_pdiode_Js * body_pdiode_area` → expected `body_pdiode_Js` is
    # *per area*. We therefore compute Js_per_area from Sebas's total:
    #   Js_per_area = is_total / area = 5.3675e-7 / 22e-12 = 2.44e4 A/m²
    # which is unrealistically large; the pdiode would inject huge body
    # current at any forward bias. We retain the prior `1e-6` "mid of
    # oracle estimates" default and document the discrepancy: Sebas's
    # `is` value is plausibly absorbing a series-resistance or scaling
    # factor we don't have access to without the full schematic. M9
    # measurement will calibrate this.
    body_pdiode_Js: float = 1e-6          # A/m² (mid-oracle default; Sebas implies 24400, see comment)
    body_pdiode_n: float = 1.0535         # ideality (Sebas's card)
    body_pdiode_Vj: float = 0.21918       # built-in (Sebas's pdiode card)
    body_pdiode_M: float = 0.24097        # grading (Sebas's card)
    body_pdiode_Cj0_per_area: float = 7.3279e-4  # F/m² zero-bias junction cap (Sebas's cj)
    # Sidewall (perimeter-junction) component — Sebas's 2026-05-02 card has
    # cjsw=1.0522e-10 F/m, ns=1.0851, isw=1.3664e-13 A/m, vjsw=0.65166,
    # mjsw=0.26029 (deep-scan log entry 2026-05-03 09:58). Defaults below
    # leave the perimeter contribution OFF (`body_pdiode_perim_length=0`) so
    # the existing 33-bias z91g 1.00-decade fit is unchanged. Set
    # `body_pdiode_perim_length = 18.8e-6` (m, = 2·(5+4.4)·1e-6 for 22 µm² body)
    # to enable. Numerical impact at Vb=0: ~12% extra C_body (16.1→18.1 fF
    # for the 22 µm² body), τ_body 2.1→~2.4 µs — within bullet 3 v4 envelope.
    body_pdiode_perim_length: float = 0.0       # m (set to 18.8e-6 for 22µm² body)
    body_pdiode_Js_sw: float = 1.3664e-13       # A/m (Sebas's isw)
    body_pdiode_n_sw: float = 1.0851            # ns ideality
    body_pdiode_Vj_sw: float = 0.65166          # vjsw built-in
    body_pdiode_M_sw: float = 0.26029           # mjsw grading
    body_pdiode_Cjsw_per_length: float = 1.0522e-10  # F/m zero-bias sidewall cap
    # Physical defaults injected when card has jss=jsd=0 (Sebas's PTM130 card
    # leaves these unset, which leaves the body diodes silent and lets Vb run
    # away unbounded under Iii injection — root cause of v6 fit explosion).
    # Typical 130nm CMOS pn junction: Js ≈ 1e-4 A/m². With AS = W·L = 360n·180n
    # = 6.5e-14 m², Is_diode ≈ 6.5e-18 A; at Vbs = 0.7V forward, Ibs ≈ 1.1e-5 A
    # → naturally clamps Vb at body-source diode turn-on voltage.
    default_jss: float = 1e-4    # A/m² source-bottom junction
    default_jsd: float = 1e-4    # A/m² drain-bottom junction

    # R-4 (2026-05-13): Series-R on the body-pdiode (parasitic N-well diode)
    # branch. R-3 audit found that when `use_well_diode=False` AND
    # `body_pdiode_to="vnwell"` (the z313 bisection topology), there was NO
    # series resistance on the body-charging path — the pdiode either ran
    # infinitesimal (Js=1e-6 × area=22e-12 ⇒ Is=2.2e-17 A) or unphysically
    # large when Js was bumped, with no R to limit it. This field mirrors
    # `vnwell_Rs` and is applied via a harmonic-mean limiter analogous to
    # lines 535-539. Default 1e10 Ω (effectively disabled).
    body_pdiode_Rs: float = 1.0e10

    # R-4 (2026-05-13): Core BSIM4 trap-assisted tunnelling (TAT) current
    # on the body-pdiode junction. Previously implemented as a monkey-patch
    # in `scripts/z313_pyport_v4.install_z313_tat_patch` — promoted here so
    # bisection variants and z320_pyport_v5 consume it natively.
    # Polarity: anode=V_b, cathode=V_Nwell (consistent with body_pdiode_to).
    #   I_TAT_leave = jtss · (exp((V_b − V_Nwell + xtss·(V_b−V_N)^2)
    #                              / (njts·Vt·acc_T)) − 1)
    # where acc_T = 1 + vtss·(T-300)/300 is the T-acceleration factor (so
    # vtss/xtss are no longer dead constants). Hard-clamped to ±10 mA.
    enable_tat: bool = False
    tat_jtss: float = 3.4e-7        # total TAT saturation current [A]
    tat_njts: float = 20.0          # TAT ideality (typically 10-30)
    tat_vtss: float = 10.0          # T-acceleration coeff (BSIM4 vtss)
    tat_xtss: float = 0.02          # V-acceleration coeff (BSIM4 xtss)

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

    # Quasi-2D body model (post-O25, gpt-5's #1 architecture option).
    # When False (default), solver runs lumped-Vb 2x2 Newton (existing path,
    # unchanged). When True, body splits into V_{b,S} (M1-side) and
    # V_{b,D} (M2-side) coupled through Rb_SD; solver expands to 3x3
    # Newton over (Vsint, Vb_S, Vb_D). Avalanche current Iii is split
    # (alpha, 1-alpha) between the two body nodes (M2-drain proximity to
    # the avalanche-generation region motivates alpha < 0.5 toward Vb_S
    # if M1 is the impact source -- but for the standard NS-RAM topology
    # M2 is where Iii originates, so alpha = 0.7 deposits more on Vb_D).
    quasi2d_body: bool = False
    Rb_SD: float = 1.0e6                # lateral spreading resistance [Ω]
    iii_split_alpha: float = 0.7        # fraction of Iii deposited on Vb_D side
    # B.2 (post-O28 oracle): branch protection. When True, reject
    # any quasi-2D Newton step where |ΔV_b,*| exceeds q2d_branch_max_dvb;
    # this prevents Newton from jumping past lumped's near-physical root
    # to the parasitic-NPN-latch alt-root (Id ≈ 1-3 µA bias-independent).
    # Implementation: damping is reduced (×0.5) until the step satisfies
    # the bound, with a min_damping floor.
    q2d_branch_protect: bool = False
    q2d_branch_max_dvb: float = 0.05    # max |ΔV_b| per Newton step [V]
    # B.3 (post-O28 oracle, gpt-5): tiny body-leak resistor as physical
    # regularizer — erases the spurious latch-up alt-root by giving the
    # body a high-impedance path to GND. 10-100 GΩ is silicon-realistic.
    # When > 0, an extra current term -V_b / Rb_leak is added to the
    # body residual (R_BS for Vb_S-side, R_BD for Vb_D-side, split
    # by iii_split_alpha for symmetry).
    q2d_body_leak_R: float = 0.0        # Rb_leak [Ω]; 0 = disabled

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
    # M2: D=Vsint, G=VG2, S=0, B=(Vb or GND, see cfg.m2_body_gnd)
    # ──────────────────────────────────────────────────────────────────
    # A.1.u (2026-05-01): Sebas's `2tnsram_simple.asc` uses the LTSpice
    # `nmos4` symbol with M2's body terminal **left unconnected** → it
    # defaults to GND. Wiring M2.B to the floating body Vb (our prior
    # behaviour) drains charge from the body through M2's bulk diodes /
    # Iii / GIDL, preventing Vb from rising enough to fire the parasitic
    # NPN that produces the snapback. cfg.m2_body_gnd=True restores
    # Sebas's topology.
    Vb_M2 = zero if cfg.m2_body_gnd else Vb
    m2 = _eval_mosfet(model_M2, sd_M2, cfg, Vg=VG2, Vd=Vsint, Vs=zero, Vb=Vb_M2,
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
        # D1 fix (2026-05-13, R_deep_A audit): emitter wired to Sint per
        # LTSpice 2tnsram_simple.asc wire trace (overrides prior A.1.i).
        # NPN R0 pins: C@(752,112)→D-net, E@(752,208)→Sint-net,
        # B@(~736,160)→B-net. Q1 fires only when Vb leads Vsint.
        Vbe = Vb - Vsint         # emitter = Sint (true LTSpice topology)
        Vbc = Vb - Vd            # collector = drain
        bjt_out = compute_bjt(bjt, Vbe=Vbe, Vbc=Vbc, T_K=273.15 + cfg.T_C)
        Ic_Q1 = bjt_out["Ic"]    # collector current (drain → emitter)
        Ib_Q1 = bjt_out["Ib"]    # base current (INTO base from external)
        Ie_Q1 = bjt_out["Ie"]    # emitter current INTO emitter (= −(Ic+Ib))
    else:
        Ic_Q1 = torch.zeros_like(Vd)
        Ib_Q1 = torch.zeros_like(Vd)
        Ie_Q1 = torch.zeros_like(Vd)
    # M3c.3 local-base node is patched in below, after iii_gain/eta_lat are
    # defined (those need m1, m2). When use_local_base=True we do an inner
    # 1D solve for Vb_local and overwrite Ic_Q1, Ib_Q1, Ie_Q1.
    Vb_local = Vb  # default: local = global (F1.v2 reduction)

    # ---- Sint KCL: currents INTO Sint --------------------------------- #
    # M1 channel current Ids_M1 flows D→S — INTO Sint (M1 source). → +Ids_M1
    # M2 drain is Sint; M2 channel sinks current FROM drain → −Ids_M2
    # D1 fix: BJT emitter now wired to Sint. Ie_Q1 = current INTO emitter
    # terminal (SPICE convention: Ie = -(Ic+Ib), forward-active → Ie<0).
    # Current INTO Sint from emitter node = -Ie_Q1.
    # M1 junction: Ibs_M1 >0 ⇒ leaves body INTO source(=Sint). → +Ibs_M1
    # M2 junction: Ibd_M2 >0 ⇒ leaves body INTO drain(=Sint). → +Ibd_M2
    R_Sint = (
        m1["Ids"]
        - m2["Ids"]
        + m1["Ibs"]
        + m2["Ibd"]
        - Ie_Q1
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
    # A.10 / PMP-6 (2026-05-11): extra **parasitic N-well diode** at the
    # floating P-body (Sebas's 2026-05-02 email + slide 21). Anode =
    # floating P-body B, cathode = V_Nwell (configurable via
    # `cfg.body_pdiode_to`; "vnwell" is the physically correct option,
    # default OFF for back-compat with the 33-bias z91g fit). Voltage
    # across the diode is V_b − V_Nwell when cathode = "vnwell".
    # Sign convention: I_nwell_diode = Js·area·(exp((Vb-Vc)/(n·Vt)) - 1),
    # positive when forward-biased (V_b > V_Nwell), leaves the body →
    # enters R_B with negative sign. Note: variable name retained as
    # `I_body_pdiode` for back-compat; topology = parasitic N-well diode.
    if cfg.body_pdiode_to != "off":
        Vt_body = 0.02585 * (273.15 + cfg.T_C) / 300.0
        if cfg.body_pdiode_to == "vnwell":
            Vc_pdi = cfg.vnwell
        elif cfg.body_pdiode_to == "gnd":
            Vc_pdi = 0.0
        elif cfg.body_pdiode_to == "sint":
            Vc_pdi = Vsint
        else:
            Vc_pdi = 0.0
        Vab = Vb - Vc_pdi
        exp_arg = (Vab / (cfg.body_pdiode_n * Vt_body)).clamp(-40.0, 40.0)
        I_body_pdiode = (cfg.body_pdiode_Js * cfg.body_pdiode_area
                          * (torch.exp(exp_arg) - 1.0))
        # Phase-B (2026-05-03 10:34): sidewall (perimeter) parallel branch.
        # Off when perim_length=0 (default). Same Vab, separate ideality and
        # saturation per Sebas's 2026-05-02 card (ns=1.0851, isw=1.3664e-13 A/m).
        if cfg.body_pdiode_perim_length > 0.0:
            exp_arg_sw = (Vab / (cfg.body_pdiode_n_sw * Vt_body)).clamp(-40.0, 40.0)
            I_body_pdiode = I_body_pdiode + (
                cfg.body_pdiode_Js_sw * cfg.body_pdiode_perim_length
                * (torch.exp(exp_arg_sw) - 1.0))
        # R-4 (2026-05-13): series-R limiter on the body-pdiode branch,
        # mirrors the well-diode path at lines 534-539. Without this, a
        # forward-biased pdiode is unbounded by anything except numerical
        # exp clamp — it cannot be tuned per V_G1 the way `vnwell_Rs` could
        # for the well-diode path. Harmonic mean transitions smoothly from
        # ideal-diode to Rs-limited as Vab climbs.
        Rs_pdi = float(getattr(cfg, "body_pdiode_Rs", 1.0e10))
        if Rs_pdi > 0.0 and Rs_pdi < 1.0e20:
            I_Rs_pdi = torch.relu(Vab) / Rs_pdi
            eps_pdi = 1e-30
            I_body_pdiode = (I_body_pdiode * I_Rs_pdi) / (
                I_body_pdiode.abs() + I_Rs_pdi + eps_pdi)
    else:
        I_body_pdiode = torch.zeros_like(Vd)

    # R-4 (2026-05-13): BSIM4 trap-assisted tunnelling on the body-pdiode
    # junction. Promoted from z313_pyport_v4 monkey-patch into the core
    # residual so all driver scripts (bisection, z320_v5, z304) see it.
    # vtss/xtss now actually enter the equation (T-acceleration and
    # V-acceleration); they used to be recorded but unused.
    if getattr(cfg, "enable_tat", False):
        Vt_tat = 0.02585 * (273.15 + cfg.T_C) / 300.0
        jtss = float(getattr(cfg, "tat_jtss", 3.4e-7))
        njts = float(getattr(cfg, "tat_njts", 20.0))
        vtss = float(getattr(cfg, "tat_vtss", 10.0))
        xtss = float(getattr(cfg, "tat_xtss", 0.02))
        # T-acceleration factor (BSIM4: vtss scales tunnel barrier with T).
        # At T_C=27 (300K), T-factor = 1; deviations scale linearly.
        T_K = 273.15 + cfg.T_C
        acc_T = 1.0 + vtss * (T_K - 300.0) / 300.0
        # V-acceleration: barrier-lowering term proportional to (V_drive)^2.
        # Polarity: anode=Vb, cathode=vnwell (matches body_pdiode_to=vnwell).
        V_drive_tat = Vb - cfg.vnwell
        V_eff = V_drive_tat + xtss * (V_drive_tat * V_drive_tat)
        arg_tat = (V_eff / (njts * Vt_tat * acc_T)).clamp(-40.0, 40.0)
        I_tat = jtss * (torch.exp(arg_tat) - 1.0)
        I_tat = I_tat.clamp(min=-1.0e-2, max=1.0e-2)  # ±10 mA hard ceiling
        # +TAT leaves body (anode=Vb), so adds to I_body_pdiode (same sign).
        I_body_pdiode = I_body_pdiode + I_tat
    else:
        I_tat = torch.zeros_like(Vd)

    # A.3.d: scale M1 body diodes (was clamping Vb at ~0.5V at VG1=0.4 row,
    # preventing parasitic NPN from lighting; controlled via cfg.m1_diode_scale,
    # default 1.0). Set <1 to weaken the diode shunt and let Vb climb.
    m1_d = float(cfg.m1_diode_scale)
    # M3b F1.v2 (post-O19): Iii→Vb collection efficiency η ∈ [0, 1].
    # Per O19 openai critique, the previous unbounded `iii_gain` was
    # itself a non-physical fudge factor that re-introduced an unphysical
    # gain path while we tried to clamp Bf. The correct form is a
    # bounded collection efficiency:
    #   η_eff = sigmoid(slope · (Vds − Vds_th)) ∈ [0, 1]
    # which models the fraction of channel-impact-ion holes that reach
    # the parasitic-NPN base laterally vs. diffuse to the bulk.
    # cfg.eta_max ∈ [0, 1] is a hard ceiling. Defaults: η_max=1.0,
    # slope=10/V, Vds_th=1.0 V → snapback regime.
    # If `iii_body_gain` is set explicitly (legacy), it overrides η
    # but is flagged as non-physical in the run log.
    iii_gain_legacy = getattr(cfg, "iii_body_gain", None)
    if iii_gain_legacy is not None and float(iii_gain_legacy) > 1.0 + 1e-9:
        # Legacy non-physical multiplier path. Used pre-O19; kept for
        # back-compat reproducibility, NOT for new fits.
        iii_gain = float(iii_gain_legacy) * torch.ones_like(Vd)
    else:
        eta_max = float(getattr(cfg, "eta_max", 1.0))
        eta_slope = float(getattr(cfg, "eta_slope", 10.0))
        eta_vds_th = float(getattr(cfg, "eta_vds_th", 1.0))
        Vds_eff = (Vd - 0.0)  # M2 source = GND, so Vds_M_full ≈ Vd
        iii_gain = eta_max * torch.sigmoid(eta_slope * (Vds_eff - eta_vds_th))

    # M3c.1 (charge-conserving electron–hole pair accounting). The
    # impact-ionised holes split into two destinations:
    #   η_lat   → fraction reaches the lateral parasitic-NPN base
    #   1−η_lat → fraction diffuses to the bulk body (existing F1.v2 path)
    # At η_lat=0 the routing reproduces F1.v2 exactly (regression gate).
    # The full M(Vbc)·Ids_M1 lateral collector formulation is M3c.2.
    eta_lat = float(getattr(cfg, "eta_lat", 0.0))
    # F6.v8 (post-O25 unanimous gemini): bias-dependent η_lat = η(Vbe).
    # Physical reasoning: as Vbe rises, the BJT base-current Ib starts
    # sinking the impact-ionized current — effectively reducing the
    # fraction reaching the lateral base. Constant η over-amplifies the
    # NPN ignition transition at VG1≈Vth (the persistent VG1=0.40
    # residual cluster). Sigmoid form per gemini:
    #   η(Vbe) = η_final + (η_0 - η_final) / (1 + exp(k·(Vbe - V_turn)))
    # In our topology (emitter=GND), Vbe = Vb. Triggered when
    # cfg.eta_sigmoid is truthy; reduces to constant η_lat otherwise.
    if getattr(cfg, "eta_sigmoid", False):
        eta_0 = float(getattr(cfg, "eta_0", 0.6))
        eta_final = float(getattr(cfg, "eta_final", 0.1))
        eta_k = float(getattr(cfg, "eta_k", 30.0))
        eta_vturn = float(getattr(cfg, "eta_vturn", 0.7))
        # Vb is the base voltage tensor at this Newton iter
        eta_lat = eta_final + (eta_0 - eta_final) * torch.sigmoid(
            -eta_k * (Vb - eta_vturn))
    iii_total_for_routing = m1["Iii"] if cfg.m2_body_gnd else (m1["Iii"] + m2["Iii"])
    Ib_lat_pair = eta_lat * iii_gain * iii_total_for_routing
    iii_to_body_factor = (1.0 - eta_lat)
    # M3c.2 path B (AUGMENT, post-O20 unanimous): the lateral-pair Ib
    # drives β·Ib of additional collector current that exits via the
    # drain (parasitic-NPN collector = drain). β = bjt.Bf (clamped at
    # the BJT's Bf parameter, which is honest-physical 100 in M3b).
    # KCL at NPN emitter (= GND): Ie_lat = Ic_lat + Ib_lat_pair.
    # GND absorbs both; drain pin sees +Ic_lat extra inflow.
    Ic_lat = float(getattr(bjt, "Bf", 100.0)) * Ib_lat_pair

    # M3c.3 local-base node (post-O21 unanimous α verdict). When
    # cfg.use_local_base=True, decouple the BJT base from the global
    # body Vb via a spread resistor Rb. Iii + GIDL inject into Vb_local;
    # BJT sees Vbe_local = Vb_local; body diodes see Vb_global.
    # Inner 1D damped-Newton solve (~6-8 iters) at each outer Newton step.
    # Default cfg.use_local_base=False → Vb_local = Vb → reduces to F1.v2.
    if cfg.use_bjt and getattr(cfg, "use_local_base", False):
        Rb = float(getattr(cfg, "lat_Rb", 1e6))
        # M3c.3 routing fix (post-O22 unanimous δ): only the lateral
        # fraction η_lat·iii_gain·Iii reaches Vb_local; the bulk
        # fraction (1−η_lat)·iii_gain·Iii stays at Vb_global. Do NOT
        # include Ib_lat_pair separately — it is the same η_lat·Iii
        # accounted via this routing.
        inflow_local = (
            eta_lat * iii_gain * iii_total_for_routing
            + m1["Igidl"] + m1["Igisl"]
        )
        Vb_local = Vb.clone().detach()  # warm start at the legacy answer
        for _it in range(10):
            # D1 fix: emitter at Sint (not GND); Vbe = Vb_local - Vsint
            bjt_l = compute_bjt(bjt, Vbe=Vb_local - Vsint, Vbc=Vb_local - Vd,
                                 T_K=273.15 + cfg.T_C)
            Ib_at_local = bjt_l["Ib"]
            spread = (Vb_local - Vb) / Rb
            f = inflow_local - Ib_at_local - spread
            # Finite-difference Jacobian (Ib_at_local has steep exponential)
            eps = 1e-4
            bjt_p = compute_bjt(bjt, Vbe=Vb_local + eps - Vsint,
                                 Vbc=Vb_local + eps - Vd,
                                 T_K=273.15 + cfg.T_C)
            dIb_dV = (bjt_p["Ib"] - Ib_at_local) / eps
            dfdV = -dIb_dV - 1.0 / Rb
            # Guard small derivatives + clamp step magnitude for stability
            step = -f / torch.where(dfdV.abs() > 1e-30, dfdV,
                                     torch.full_like(dfdV, -1.0 / Rb))
            step = torch.clamp(step, min=-0.1, max=0.1)
            Vb_local = Vb_local + step
            if float(step.abs().max()) < 1e-10:
                break
        # Recompute BJT outputs at the converged Vb_local
        # D1 fix: emitter at Sint (not GND); Vbe = Vb_local - Vsint
        bjt_out_local = compute_bjt(bjt, Vbe=Vb_local - Vsint, Vbc=Vb_local - Vd,
                                     T_K=273.15 + cfg.T_C)
        Ic_Q1 = bjt_out_local["Ic"]
        Ib_Q1 = bjt_out_local["Ib"]
        Ie_Q1 = bjt_out_local["Ie"]

    # M3c.2 path C (TOGGLE, post-O20): avalanche multiplier on the
    # channel current. Activated only when cfg.use_lateral_collector is
    # True. Default False → identical to path B baseline.
    # Vbc = Vb − Vd. In snapback regime Vd > Vb → Vbc < 0 (reverse-bias
    # B-C). We multiply only by the reverse-biased magnitude:
    #   M(Vbc) = 1 + (max(−Vbc, 0) / BV)^N      smoothed via softplus
    #   M_safe = 1 + (M − 1) · sigmoid((BV_max − |Vbc|) / δ)
    # so M saturates as |Vbc| → BV_max. Ic_avalanche is the EXTRA
    # collector current beyond Ids_M1 — added to drain only, never to
    # Ids_M1 itself (avoids double-count per O19 openai critique).
    if getattr(cfg, "use_lateral_collector", False):
        BV = float(getattr(cfg, "lat_BV", 6.0))
        N_av = float(getattr(cfg, "lat_N", 4.0))
        BV_max = float(getattr(cfg, "lat_BV_max", BV * 1.1))
        delta = float(getattr(cfg, "lat_M_smooth_delta", 0.5))
        # Reverse-bias magnitude: positive only when Vd > Vb
        Vbc = Vb - Vd
        rev_mag = torch.clamp(-Vbc, min=0.0)
        M_raw = 1.0 + (rev_mag / BV) ** N_av
        # Smooth saturation as |Vbc| approaches BV_max
        sat = torch.sigmoid((BV_max - rev_mag) / delta)
        M_safe = 1.0 + (M_raw - 1.0) * sat
        Ic_avalanche = (M_safe - 1.0) * m1["Ids"]
    else:
        Ic_avalanche = torch.zeros_like(Vd)

    if getattr(cfg, "use_local_base", False) and cfg.use_bjt:
        # M3c.3 (post-O22 routing fix): only the lateral fraction
        # η_lat·iii_gain·Iii routes to Vb_local. The bulk fraction
        # (1−η_lat)·iii_gain·Iii stays at Vb_global as in F1.v2.
        # Vb_global also receives spread current from local via Rb.
        # GIDL routes to local (per gpt-5 framework); body diodes /
        # well / Igb stay at global.
        Rb = float(getattr(cfg, "lat_Rb", 1e6))
        spread_in = (Vb_local - Vb) / Rb
        if cfg.m2_body_gnd:
            R_B = (
                iii_to_body_factor * iii_gain * m1["Iii"]   # bulk fraction stays
                + spread_in                                  # local feeds back via Rb
                + m1["Igb"]
                - m1_d * m1["Ibs"] - m1_d * m1["Ibd"]
                + I_well_body
                - I_body_pdiode
            )
        else:
            R_B = (
                iii_to_body_factor * iii_gain * (m1["Iii"] + m2["Iii"])
                + spread_in
                + m1["Igb"] + m2["Igb"]
                - m1["Ibs"] - m1["Ibd"]
                - m2["Ibs"] - m2["Ibd"]
                + I_well_body
                - I_body_pdiode
            )
    elif cfg.m2_body_gnd:
        # A.1.u: M2's body is GND, so its body-current contributions do
        # NOT enter the floating-body KCL — they flow between M2's nodes
        # and ground, not the floating Vb.
        R_B = (
            iii_to_body_factor * iii_gain * m1["Iii"]
            + m1["Igidl"] + m1["Igisl"]
            + m1["Igb"]
            - m1_d * m1["Ibs"] - m1_d * m1["Ibd"]
            - Ib_Q1
            - Ib_lat_pair
            + I_well_body
            - I_body_pdiode
        )
    else:
        R_B = (
            iii_to_body_factor * iii_gain * (m1["Iii"] + m2["Iii"])
            + m1["Igidl"] + m1["Igisl"] + m2["Igidl"] + m2["Igisl"]
            + m1["Igb"] + m2["Igb"]
            - m1["Ibs"] - m1["Ibd"]
            - m2["Ibs"] - m2["Ibd"]
            - Ib_Q1
            - Ib_lat_pair
            + I_well_body
            - I_body_pdiode
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
        "Ib_lat_pair": Ib_lat_pair,
        "Ic_lat": Ic_lat,
        "Ic_avalanche": Ic_avalanche,
        "Vb_local": Vb_local,
        "Igidl_M1": m1["Igidl"], "Igisl_M1": m1["Igisl"],
        "Igidl_M2": m2["Igidl"], "Igisl_M2": m2["Igisl"],
        "Igb_M1": m1["Igb"], "Igb_M2": m2["Igb"],
        "Ibs_M1": m1["Ibs"], "Ibd_M1": m1["Ibd"],
        "Ibs_M2": m2["Ibs"], "Ibd_M2": m2["Ibd"],
        "I_well_body": I_well_body,
        "I_body_pdiode": I_body_pdiode,
        "I_tat": I_tat,
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
        + comp.get("Ic_lat", 0.0)         # M3c.2 path B: lateral-pair-driven β·Ib
        + comp.get("Ic_avalanche", 0.0)   # M3c.2 path C: M(Vbc) avalanche on Ids
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


# =========================================================================== #
# QUASI-2D BODY MODEL (post-O25, gpt-5's #1 architecture upgrade)             #
# Plan A wrapper implementation (2026-05-07): split body Vb -> (Vb_S, Vb_D)   #
# coupled by Rb_SD lateral resistor; Iii avalanche current is split           #
# (alpha, 1-alpha) between Vb_D-side (M2-drain proximal to avalanche source)  #
# and Vb_S-side. M1/M2/well/diode/eta_lat/BJT machinery unchanged: this       #
# wrapper calls the existing _residuals at Vb=mean(Vb_S, Vb_D) and ADDS the   #
# split routing + coupling on top, so when Rb_SD is small enough that the     #
# coupling forces Vb_S = Vb_D, the solution reduces to the lumped-Vb fit.     #
# =========================================================================== #

def _residuals_quasi2d(
    cfg: NSRAMCell2TConfig,
    model_M1: BSIM4Model,
    bjt: GummelPoonNPN,
    Vd: torch.Tensor,
    VG1: torch.Tensor,
    VG2: torch.Tensor,
    Vsint: torch.Tensor,
    Vb_S: torch.Tensor,
    Vb_D: torch.Tensor,
    P_M1: Optional[dict] = None,
    P_M2: Optional[dict] = None,
    model_M2: Optional[BSIM4Model] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Quasi-2D body Newton residuals: returns (R_Sint, R_BS, R_BD, components).

    Plan A approximation: the existing single-Vb residual machinery is
    invoked at Vb = mean(Vb_S, Vb_D); the body residual R_B from that call
    is then split (alpha, 1-alpha) onto (R_BD, R_BS), and the lateral
    coupling current I_SD = (Vb_D - Vb_S) / Rb_SD is added with opposite
    signs.

    Conservation: R_BS + R_BD = R_B (lumped) at any (Vb_S, Vb_D), so KCL
    is preserved globally. When Vb_S = Vb_D (which Rb_SD -> small forces),
    the lumped fit is recovered exactly.
    """
    Vb_avg = 0.5 * (Vb_S + Vb_D)
    R_Sint, R_B, comps = _residuals(
        cfg, model_M1, bjt, Vd, VG1, VG2, Vsint, Vb_avg,
        P_M1=P_M1, P_M2=P_M2, model_M2=model_M2,
    )
    alpha = float(getattr(cfg, "iii_split_alpha", 0.7))
    Rb_SD = float(getattr(cfg, "Rb_SD", 1.0e6))
    # Coupling: I from D-side to S-side when Vb_D > Vb_S.
    #   R_BS gains current INTO B_S (positive contribution to R_BS as KCL):
    #     +(Vb_D - Vb_S)/Rb_SD
    #   R_BD loses the same current (it leaves B_D):
    #     -(Vb_D - Vb_S)/Rb_SD
    I_coup = (Vb_D - Vb_S) / Rb_SD
    R_BS = (1.0 - alpha) * R_B + I_coup
    R_BD = alpha * R_B - I_coup
    # B.3 body-leak regularizer (post-O28, gpt-5): tiny resistor from
    # each body node to GND erases the parasitic-NPN latch-up branch
    # that the model allows but silicon doesn't sit on. Split equally
    # so it doesn't introduce its own asymmetry.
    Rb_leak = float(getattr(cfg, "q2d_body_leak_R", 0.0))
    if Rb_leak > 0.0:
        I_leak_S = Vb_S / Rb_leak  # leaves Vb_S → GND, so subtract from R_BS
        I_leak_D = Vb_D / Rb_leak  # leaves Vb_D → GND, so subtract from R_BD
        R_BS = R_BS - I_leak_S
        R_BD = R_BD - I_leak_D
    comps = dict(comps)
    comps["Vb_avg"] = Vb_avg
    comps["I_coup"] = I_coup
    return R_Sint, R_BS, R_BD, comps


def _jacobian_finite_diff_quasi2d(
    cfg, model_M1, bjt, Vd, VG1, VG2, Vsint, Vb_S, Vb_D,
    P_M1, P_M2, h: float = 1e-6, model_M2=None,
) -> torch.Tensor:
    """3x3 finite-difference Jacobian for quasi-2D residuals.

    Returns shape (..., 3, 3). Columns: ∂/∂Vsint, ∂/∂Vb_S, ∂/∂Vb_D.
    Rows: R_Sint, R_BS, R_BD.
    """
    with torch.no_grad():
        # Vsint perturbation
        Rsp_s, Rbsp_s, Rbdp_s, _ = _residuals_quasi2d(
            cfg, model_M1, bjt, Vd, VG1, VG2,
            Vsint + h, Vb_S, Vb_D, P_M1, P_M2, model_M2=model_M2)
        Rsm_s, Rbsm_s, Rbdm_s, _ = _residuals_quasi2d(
            cfg, model_M1, bjt, Vd, VG1, VG2,
            Vsint - h, Vb_S, Vb_D, P_M1, P_M2, model_M2=model_M2)
        dRs_dVs = (Rsp_s - Rsm_s) / (2 * h)
        dRbs_dVs = (Rbsp_s - Rbsm_s) / (2 * h)
        dRbd_dVs = (Rbdp_s - Rbdm_s) / (2 * h)
        # Vb_S perturbation
        Rsp_bs, Rbsp_bs, Rbdp_bs, _ = _residuals_quasi2d(
            cfg, model_M1, bjt, Vd, VG1, VG2,
            Vsint, Vb_S + h, Vb_D, P_M1, P_M2, model_M2=model_M2)
        Rsm_bs, Rbsm_bs, Rbdm_bs, _ = _residuals_quasi2d(
            cfg, model_M1, bjt, Vd, VG1, VG2,
            Vsint, Vb_S - h, Vb_D, P_M1, P_M2, model_M2=model_M2)
        dRs_dVbs = (Rsp_bs - Rsm_bs) / (2 * h)
        dRbs_dVbs = (Rbsp_bs - Rbsm_bs) / (2 * h)
        dRbd_dVbs = (Rbdp_bs - Rbdm_bs) / (2 * h)
        # Vb_D perturbation
        Rsp_bd, Rbsp_bd, Rbdp_bd, _ = _residuals_quasi2d(
            cfg, model_M1, bjt, Vd, VG1, VG2,
            Vsint, Vb_S, Vb_D + h, P_M1, P_M2, model_M2=model_M2)
        Rsm_bd, Rbsm_bd, Rbdm_bd, _ = _residuals_quasi2d(
            cfg, model_M1, bjt, Vd, VG1, VG2,
            Vsint, Vb_S, Vb_D - h, P_M1, P_M2, model_M2=model_M2)
        dRs_dVbd = (Rsp_bd - Rsm_bd) / (2 * h)
        dRbs_dVbd = (Rbsp_bd - Rbsm_bd) / (2 * h)
        dRbd_dVbd = (Rbdp_bd - Rbdm_bd) / (2 * h)

    J = torch.stack([
        torch.stack([dRs_dVs,  dRs_dVbs,  dRs_dVbd ], dim=-1),
        torch.stack([dRbs_dVs, dRbs_dVbs, dRbs_dVbd], dim=-1),
        torch.stack([dRbd_dVs, dRbd_dVbs, dRbd_dVbd], dim=-1),
    ], dim=-2)
    return J


def solve_2t_quasi2d_steady_state(
    cfg: NSRAMCell2TConfig,
    model: BSIM4Model,
    bjt: GummelPoonNPN,
    Vd: torch.Tensor,
    VG1: torch.Tensor,
    VG2: torch.Tensor,
    P_M1: Optional[dict] = None,
    P_M2: Optional[dict] = None,
    Vsint_init: Optional[torch.Tensor] = None,
    Vb_S_init: Optional[torch.Tensor] = None,
    Vb_D_init: Optional[torch.Tensor] = None,
    verbose: bool = False,
    model_M2: Optional[BSIM4Model] = None,
) -> dict:
    """Solve the 2T cell with quasi-2D split body at quasi-static (Vd, VG1, VG2).

    Returns dict with: Id, Vsint, Vb_S, Vb_D, components, residuals, niter,
    converged. Uses 3x3 Newton with FD Jacobian, parallel structure to
    solve_2t_steady_state.

    Warm-start strategy: if Vb_S_init / Vb_D_init not provided, first run
    the lumped-Vb solver to convergence and use Vb := Vb_lumped for both
    initial guesses. This makes the new physics a perturbation around the
    lumped solution rather than a cold start.
    """
    Vd = torch.as_tensor(Vd, dtype=torch.float64)
    VG1 = torch.as_tensor(VG1, dtype=torch.float64)
    VG2 = torch.as_tensor(VG2, dtype=torch.float64)
    Vd, VG1, VG2 = torch.broadcast_tensors(Vd, VG1, VG2)
    Vd = Vd.contiguous(); VG1 = VG1.contiguous(); VG2 = VG2.contiguous()

    # Warm start from lumped-Vb solver
    if Vsint_init is None or Vb_S_init is None or Vb_D_init is None:
        warm = solve_2t_steady_state(
            cfg, model, bjt, Vd, VG1, VG2,
            P_M1=P_M1, P_M2=P_M2, model_M2=model_M2, verbose=False)
        if Vsint_init is None:
            Vsint_init = warm["Vsint"].detach().clone()
        if Vb_S_init is None:
            Vb_S_init = warm["Vb"].detach().clone()
        if Vb_D_init is None:
            Vb_D_init = warm["Vb"].detach().clone()
    Vsint = Vsint_init.clone()
    Vb_S = Vb_S_init.clone()
    Vb_D = Vb_D_init.clone()

    max_iters = cfg.newton_max_iters
    iabstol = getattr(cfg, "Iabstol", cfg.newton_tol)
    ireltol = getattr(cfg, "Ireltol", 0.0)
    xtol_v = getattr(cfg, "xtol_v", 0.0)
    min_iters = getattr(cfg, "min_iters", 1)
    converged = torch.zeros_like(Vd, dtype=torch.bool)
    last_dV_inf = torch.tensor(float("inf"), dtype=torch.float64)

    # Initial residual + components for the relative-tolerance scaling
    R_Sint, R_BS, R_BD, cur_comps = _residuals_quasi2d(
        cfg, model, bjt, Vd, VG1, VG2, Vsint, Vb_S, Vb_D,
        P_M1=P_M1, P_M2=P_M2, model_M2=model_M2)

    def _i_scale(comp):
        keys = ["Ids_M1", "Ids_M2", "Ic_Q1", "Ib_Q1",
                "Iii_M1", "Iii_M2", "Igidl_M1", "Igidl_M2",
                "Ibs_M1", "Ibd_M1", "Ibs_M2", "Ibd_M2"]
        s = torch.zeros_like(R_Sint.detach())
        for k in keys:
            if k in comp:
                s = s + comp[k].detach().abs()
        return s

    for it in range(max_iters):
        # Stop criterion mirrors solve_2t_steady_state: residual must be
        # small RELATIVE to physical-current scale, not just absolute.
        # This is what keeps lumped at the bias-dependent physical root
        # instead of walking past it to the well-body-saturation branch.
        residual_max = torch.maximum(
            torch.maximum(R_Sint.detach().abs(), R_BS.detach().abs()),
            R_BD.detach().abs())
        I_scale = _i_scale(cur_comps)
        tol_eff = torch.maximum(torch.full_like(I_scale, iabstol), ireltol * I_scale)
        residual_ok = bool((residual_max < tol_eff).all())
        step_ok = bool((last_dV_inf < xtol_v).all()) if xtol_v > 0 else False
        if verbose:
            print(f"  [q2d iter {it:2d}] max|R| = {residual_max.max().item():.3e} "
                  f"tol = {tol_eff.max().item():.3e} |dV| = {float(last_dV_inf):.3e}")
        if it >= min_iters and (residual_ok or step_ok):
            converged = residual_max < tol_eff
            break

        J = _jacobian_finite_diff_quasi2d(
            cfg, model, bjt, Vd, VG1, VG2, Vsint.detach(), Vb_S.detach(), Vb_D.detach(),
            P_M1, P_M2, model_M2=model_M2)
        with torch.no_grad():
            R_vec = torch.stack([R_Sint.detach(), R_BS.detach(), R_BD.detach()],
                                dim=-1).unsqueeze(-1)
            eye = torch.eye(3, dtype=J.dtype, device=J.device)
            J_safe = J + cfg.gmin * eye
            try:
                dV = torch.linalg.solve(J_safe, -R_vec).squeeze(-1)
            except Exception:
                dV = (torch.linalg.pinv(J_safe) @ (-R_vec)).squeeze(-1)
            # Per-step relative cap
            max_abs = dV.abs().max(dim=-1, keepdim=True).values
            scale = torch.where(max_abs > cfg.max_step_V,
                                cfg.max_step_V / max_abs.clamp_min(1e-30),
                                torch.ones_like(max_abs))
            dV = dV * scale

        # Armijo-style backtracking damping (mirrors lumped solver) +
        # B.2 branch protection (post-O28): reject |ΔV_b| > threshold
        # to prevent Newton from jumping past the physical root to the
        # parasitic-NPN-latch alt-root (which has smaller residual but
        # silicon doesn't sit there at the calibration biases).
        prev_norm = (R_Sint.detach().abs() + R_BS.detach().abs()
                     + R_BD.detach().abs())
        damping = cfg.newton_damping
        # If branch-protect: reduce initial damping until |ΔV_b| ≤ max_dvb
        if getattr(cfg, "q2d_branch_protect", False):
            max_dvb = float(getattr(cfg, "q2d_branch_max_dvb", 0.05))
            with torch.no_grad():
                step_b_max = torch.maximum(
                    (damping * dV[..., 1]).abs().max(),
                    (damping * dV[..., 2]).abs().max())
                while float(step_b_max) > max_dvb and damping > cfg.newton_min_damping:
                    damping *= 0.5
                    step_b_max = torch.maximum(
                        (damping * dV[..., 1]).abs().max(),
                        (damping * dV[..., 2]).abs().max())
        accepted = False
        while damping >= cfg.newton_min_damping:
            Vsint_try = Vsint + damping * dV[..., 0]
            Vb_S_try = Vb_S + damping * dV[..., 1]
            Vb_D_try = Vb_D + damping * dV[..., 2]
            R_S_try, R_BS_try, R_BD_try, comp_try = _residuals_quasi2d(
                cfg, model, bjt, Vd, VG1, VG2,
                Vsint_try, Vb_S_try, Vb_D_try,
                P_M1=P_M1, P_M2=P_M2, model_M2=model_M2)
            new_norm = (R_S_try.detach().abs() + R_BS_try.detach().abs()
                        + R_BD_try.detach().abs())
            if (new_norm.mean() < prev_norm.mean() * 0.999) or damping <= cfg.newton_min_damping:
                Vsint = Vsint_try
                Vb_S = Vb_S_try
                Vb_D = Vb_D_try
                R_Sint = R_S_try
                R_BS = R_BS_try
                R_BD = R_BD_try
                cur_comps = comp_try
                accepted = True
                last_dV_inf = (damping * dV).abs().max()
                break
            damping *= 0.5
        if not accepted:
            break

    # Final residual + convergence flag
    residual_max = torch.maximum(
        torch.maximum(R_Sint.detach().abs(), R_BS.detach().abs()),
        R_BD.detach().abs())
    I_scale = _i_scale(cur_comps)
    tol_eff = torch.maximum(torch.full_like(I_scale, iabstol), ireltol * I_scale)
    converged = residual_max < tol_eff
    # Final eval at converged voltages to get components
    R_Sint, R_BS, R_BD, comps = _residuals_quasi2d(
        cfg, model, bjt, Vd, VG1, VG2, Vsint, Vb_S, Vb_D,
        P_M1=P_M1, P_M2=P_M2, model_M2=model_M2)
    # Id formula matches solve_2t_steady_state (line ~1078)
    Id = (
        comps["Ids_M1"]
        + comps["Ic_Q1"]
        + comps.get("Ic_lat", 0.0)
        + comps.get("Ic_avalanche", 0.0)
        + comps["Igidl_M1"]
        - comps["Ibd_M1"]
    )
    return {
        "Id": Id,
        "Vsint": Vsint,
        "Vb_S": Vb_S,
        "Vb_D": Vb_D,
        "Vb_avg": 0.5 * (Vb_S + Vb_D),
        "components": comps,
        "R_Sint": R_Sint,
        "R_BS": R_BS,
        "R_BD": R_BD,
        "niter": it + 1,
        "converged": converged,
    }

```


=== FILE: summary.json (3121 chars) ===
```json
{
  "script": "z325_iii_instrument",
  "elapsed_s": 0.8961336612701416,
  "bias": {
    "V_G1": 0.6,
    "V_G2": 0.2,
    "V_d": 2.0
  },
  "solver": {
    "Id_A": 8.72958812357137e-13,
    "Vsint_V": 1.8663296959256948,
    "Vb_V": 1.9999999964432678,
    "converged": true
  },
  "components_at_Vb_node_A": {
    "Iii_M1_raw": 7.493425965531349e-48,
    "Iii_M2_raw": 9.056805997002132e-16,
    "Iii_M1_manual_recompute": 4.3050840189161375e-49,
    "Igidl_M1": 3.0941270272968556e-46,
    "Igisl_M1": 0.0,
    "Igidl_M2": 1.2378077412136202e-15,
    "Igisl_M2": 0.0,
    "Igb_M1": 0.0,
    "Igb_M2": 0.0,
    "Ibs_M1": 4.776078055035827e-17,
    "Ibd_M1": -8.911018032264905e-25,
    "Ibs_M2": 0.0,
    "Ibd_M2": -6.479999999999989e-17,
    "I_well_body": 0.0,
    "I_body_pdiode": -2.32625001705955e-15,
    "I_tat": -2.32625001705955e-15,
    "Ib_Q1": 8.729587428728817e-17,
    "Ic_Q1": 8.729588123562459e-13,
    "Ib_lat_pair": 0.0,
    "R_B_residual": -7.614723080867588e-17,
    "R_Sint_residual": 4.982727063481631e-13
  },
  "dominant_term": {
    "name": "-I_body_pdiode",
    "value_A": 2.32625001705955e-15
  },
  "iii_routing": {
    "routed_into_R_B": true,
    "code_path_used": "m2_body_gnd branch, nsram/nsram/bsim4_port/nsram_cell_2T.py lines 833-846",
    "term_in_residual": "iii_to_body_factor * iii_gain * m1[\"Iii\"]",
    "source_grep_hits_line_col": [
      [
        816,
        "iii_to_body_factor * iii_gain * m1[\"Iii\"]   # bulk fraction stays"
      ],
      [
        825,
        "iii_to_body_factor * iii_gain * (m1[\"Iii\"] + m2[\"Iii\"])"
      ],
      [
        838,
        "iii_to_body_factor * iii_gain * m1[\"Iii\"]"
      ],
      [
        849,
        "iii_to_body_factor * iii_gain * (m1[\"Iii\"] + m2[\"Iii\"])"
      ]
    ],
    "iii_gain_at_OP": 0.9999546021312976,
    "eta_lat_at_OP": 0.0,
    "iii_to_body_factor_at_OP": 1.0,
    "iii_effective_into_Vb_A": 7.493085779963236e-48
  },
  "bsim4_params": {
    "alpha0_used": 7.842e-05,
    "beta0_used": 20.0,
    "Vdseff": 0.03894898119058654,
    "Vds_minus_Vdseff": 0.0947213228837187,
    "Idsa": 5.262411857517794e-36
  },
  "verdict": {
    "Iii_M1_value_A": 7.493425965531349e-48,
    "Iii_manual_recompute_A": 4.3050840189161375e-49,
    "Iii_routed_into_R_B": true,
    "iii_gain_at_Vd_2_0": 0.9999546021312976,
    "iii_effective_into_Vb_A": 7.493085779963236e-48,
    "dominant_R_B_term": "-I_body_pdiode",
    "dominant_R_B_value_A": 2.32625001705955e-15,
    "hypothesis_b_status": "FALSE \u2014 Iii IS routed (m2_body_gnd branch, line ~838); Iii is zero at OP because Vds-Vdseff and Ids are tiny"
  },
  "code_changes": {
    "loc_changed": 0,
    "rationale": "Iii IS already routed into R_B via `iii_to_body_factor * iii_gain * m1[\"Iii\"]` at line ~838 (m2_body_gnd branch). Hypothesis (b) wiring-bug is FALSE. Hypothesis (ii) is true: Iii \u2248 0 at the OP because the BSIM4 formula `T1\u00b7Idsa\u00b7Vdseff` produces a tiny value when Vds-Vdseff is small and/or Ids is in subthreshold."
  },
  "z324_baseline_VG1_0_6_median_dec": 3.248,
  "post_fix_VG1_0_6_median_dec": null,
  "delta_dec": 0.0
}
```
