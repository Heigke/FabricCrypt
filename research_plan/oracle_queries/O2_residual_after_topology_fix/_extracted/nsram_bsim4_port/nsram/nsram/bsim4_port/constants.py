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
