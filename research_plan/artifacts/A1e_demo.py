"""A1e: GIDL/GISL load trace + bias evaluation for M2.

Standalone script (no source edits). Demonstrates:
  1. agidl-group is loaded; agisl-group siblings retain pre-override defaults.
  2. The "ref" default mechanism is order-dependent in BSIM4Model.__init__,
     so user override of agidl does NOT propagate to agisl after pass 2.
  3. At (VG1=0.6, VG2=0.0), GIDL gate is closed (Vds-Vgs-egidl<0) — physics OK.
  4. M1 source-side GISL gate IS open (Vgs-Vsd-egisl > 0) but agisl=0
     so Igisl ≡ 0. THIS is the silent missing leakage path.
"""

import math, os
from nsram.bsim4_port.model_card import BSIM4Model

CARD = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/data/sebas_2026_04_22/M2_130bulkNSRAM.txt'
text = open(CARD).read()
m = BSIM4Model.from_spice(text, model_type='nmos')

print("=" * 60)
print("Loaded vs card-specified vs default")
print("=" * 60)
defaults = {'agidl': 0.0, 'bgidl': 2.3e9, 'cgidl': 0.5, 'egidl': 0.8,
            'agisl': '(ref agidl)', 'bgisl': '(ref bgidl)',
            'cgisl': '(ref cgidl)', 'egisl': '(ref egidl)'}
card_vals = {'agidl': 1.99e-8, 'bgidl': 1.624e9, 'cgidl': 6.3, 'egidl': 0.91}
for k in ('agidl', 'bgidl', 'cgidl', 'egidl', 'agisl', 'bgisl', 'cgisl', 'egisl'):
    loaded = m.get(k)
    card = card_vals.get(k, '— (not in card)')
    deflt = defaults[k]
    given = m.is_given(k)
    print(f"  {k:7s}  loaded={loaded:>12.4g}  card={str(card):>14s}  default={str(deflt):>14s}  given={given}")

print()
print(f"  gidlmod = {int(m.get('gidlmod', 0))}  (0 = pre-4.7 model active)")

# Hand-compute formula at the failing bias.
# M2 device: source = 0V, drain = Vsint = 0.306V, gate = VG2 = 0.0V, body = Vb = 0.342V
# M1 device: source = Vsint = 0.306V, drain = Vd = 1.5V, gate = VG1 = 0.6V, body = Vb = 0.342V
print()
print("=" * 60)
print("Bias gate check (b4ld.c §2295: V_drive = Vd-Vg-egidl, must be > 0)")
print("=" * 60)
egidl = m.get('egidl'); egisl = m.get('egisl')
# M2 GIDL (drain side of M2 = Vsint=0.306, gate=0, source=0)
Vds_M2, Vgs_M2 = 0.306, 0.0
Vd_drive_M2 = Vds_M2 - Vgs_M2 - egidl
print(f"  M2 GIDL  Vds={Vds_M2:.3f} Vgs={Vgs_M2:.3f} egidl={egidl:.3f}  V_drive={Vd_drive_M2:+.3f}  -> {'OPEN' if Vd_drive_M2>0 else 'CLOSED'}")
# M2 GISL (source side of M2 = 0V, with gate 0V, drain 0.306V)
# In BSIM4, GISL drive: -Vds-Vgd-egisl  with vgd=Vg-Vd
Vgd_M2 = Vgs_M2 - Vds_M2  # = -0.306
Vs_drive_M2 = -Vds_M2 - Vgd_M2 - egisl
print(f"  M2 GISL  Vds={Vds_M2:.3f} Vgd={Vgd_M2:.3f} egisl={egisl:.3f}  V_drive={Vs_drive_M2:+.3f}  -> {'OPEN' if Vs_drive_M2>0 else 'CLOSED'}")

# M1 GIDL (drain side = Vd=1.5, gate = VG1=0.6, source = Vsint=0.306)
Vds_M1, Vgs_M1 = 1.5 - 0.306, 0.6 - 0.306  # = 1.194, 0.294
Vd_drive_M1 = Vds_M1 - Vgs_M1 - egidl
print(f"  M1 GIDL  Vds={Vds_M1:.3f} Vgs={Vgs_M1:.3f} egidl={egidl:.3f}  V_drive={Vd_drive_M1:+.3f}  -> {'OPEN' if Vd_drive_M1>0 else 'CLOSED'}")
# M1 GISL (source side, where source = Vsint=0.306 — body-charging path!)
Vgd_M1 = Vgs_M1 - Vds_M1   # 0.294 - 1.194 = -0.9
Vs_drive_M1 = -Vds_M1 - Vgd_M1 - egisl
print(f"  M1 GISL  Vds={Vds_M1:.3f} Vgd={Vgd_M1:.3f} egisl={egisl:.3f}  V_drive={Vs_drive_M1:+.3f}  -> {'OPEN' if Vs_drive_M1>0 else 'CLOSED'}")

# What WOULD M1 GISL emit if agisl/bgisl/cgisl/egisl had been mirrored from agidl group?
print()
print("=" * 60)
print("Counterfactual: if agisl group had defaulted to agidl values from card")
print("=" * 60)
toxe = m.get('toxe', 3.4e-9)
weff_cj = 1e-7  # ~Weff_CJ scale; rough order
# Loaded (broken):
agisl_l, bgisl_l, cgisl_l, egisl_l = m.get('agisl'), m.get('bgisl'), m.get('cgisl'), m.get('egisl')
# Counterfactual (correct BSIM4 behavior):
agisl_c, bgisl_c, cgisl_c, egisl_c = m.get('agidl'), m.get('bgidl'), m.get('cgidl'), m.get('egidl')

def gisl(a, b, c, e, vdrive, vbs):
    if vdrive <= 0 or a == 0:
        return 0.0
    T1 = vdrive / (3.0 * toxe)
    T2 = b / T1 if T1 > 0 else 1e30
    if T2 > 100: return 0.0
    Igisl_pre = a * weff_cj * T1 * math.exp(-T2)
    vbs_term = vbs**3 / (vbs**3 + c) if vbs > 0 else 0.0  # body-bias modulation (simplified)
    return Igisl_pre

# Vbs for M1: Vb - Vsource = 0.342 - 0.306 = 0.036
Vbs_M1 = 0.036
Vs_drive_recomp = -Vds_M1 - Vgd_M1 - egisl_c
I_loaded = gisl(agisl_l, bgisl_l, cgisl_l, egisl_l, Vs_drive_M1, Vbs_M1)
I_counter = gisl(agisl_c, bgisl_c, cgisl_c, egisl_c, Vs_drive_recomp, Vbs_M1)
print(f"  loaded  agisl=0      bgisl=2.3e9    -> Igisl(M1) = {I_loaded:.3e} A")
print(f"  counter agisl=1.99e-8 bgisl=1.624e9 -> Igisl(M1) = {I_counter:.3e} A")
print()
print("If counter > 1e-15, the missing GISL is a real residual leak source.")
