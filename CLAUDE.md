export DAEDALUS_PASS="daedalus"
export DAEDALUS_HOST="daedalus.local"
export DAEDALUS_USER="daedalus"
# Always use daedalus.local hostname (mDNS). Do NOT use 192.168.0.37 — IP varies/breaks.
and check in venvs for the torch-rocm venv

export MINOS_PASS="minos"
export MINOS_HOST="192.168.0.38"
export MINOS_USER="minos"

and we are locally on ikaros and venv is /home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/venv

# CRITICAL: gfx1151 (AMD Radeon 8060S) requires HSA override for torch compatibility
export HSA_OVERRIDE_GFX_VERSION=11.0.0
# Run scripts with: HSA_OVERRIDE_GFX_VERSION=11.0.0 python script.py

● Vivado 2025.2 is already installed at /opt/Xilinx/2025.2/Vivado/bin/vivado                                                                                                                            