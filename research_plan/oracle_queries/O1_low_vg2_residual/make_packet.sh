#!/usr/bin/env bash
# Build the upload zips for oracle query O1.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy

cp -f "$ROOT/data/sebas_2026_04_22/M1_130DNWFB.txt" ./ 2>/dev/null || true
cp -f "$ROOT/data/sebas_2026_04_22/M2_130bulkNSRAM.txt" ./ 2>/dev/null || true

# 1. BSIM4 port source — tight zip, no caches
cd "$ROOT"
zip -q -r research_plan/oracle_queries/O1_low_vg2_residual/nsram_bsim4_port.zip \
    nsram/nsram/bsim4_port/ \
    -x '*/__pycache__/*' '*.pyc' '*/tests/*'

# 2. Validation scripts subset
zip -q -j research_plan/oracle_queries/O1_low_vg2_residual/validation_scripts.zip \
    scripts/z91d_bsim4_port_fit_arclength.py \
    scripts/z91e_bsim4_port_fit_with_anchors.py \
    scripts/z91f_validate_with_sebas_params.py \
    scripts/z91g_two_model_validation.py

cd "$ROOT/research_plan/oracle_queries/O1_low_vg2_residual"
echo "Packet contents:"
ls -la
echo
echo "Total upload size:"
du -sh .
echo
echo "Recommended: zip the entire directory before upload:"
echo "  cd $(pwd) && zip -r O1_packet.zip ."
