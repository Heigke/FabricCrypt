#!/usr/bin/env bash
# Build O2 oracle packet — current state of the residual after emitter=GND fix.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy
NSRAM_INFO=/home/ikaros/nsram_info

# Stage individual files
cp -f "$ROOT/results/z91g_two_model_validation/fit_vs_meas.png" ./
cp -f "$ROOT/results/z91g_two_model_validation/summary.json" ./
cp -f "$ROOT/data/sebas_2026_04_22/M1_130DNWFB.txt" ./
cp -f "$ROOT/data/sebas_2026_04_22/M2_130bulkNSRAM.txt" ./
cp -f "$ROOT/data/sebas_2026_04_22/2Tcell_BSIM_param_DC.csv" ./
cp -f "$NSRAM_INFO/schematic&modelCards/parasiticBJT.txt" ./
cp -f "$NSRAM_INFO/schematic&modelCards/2tnsram_simple.asc" ./

# Stage diagnostic artifacts
mkdir -p artifacts
cp -f "$ROOT/research_plan/artifacts/"A1*.md artifacts/ 2>/dev/null || true
cp -f "$ROOT/research_plan/artifacts/email_history.md" ./

# 1. BSIM4 port — current state (post emitter=GND fix)
cd "$ROOT"
zip -q -r research_plan/oracle_queries/O2_residual_after_topology_fix/nsram_bsim4_port.zip \
    nsram/nsram/bsim4_port/ \
    -x '*/__pycache__/*' '*.pyc' '*/tests/*'

# 2. Validation scripts subset
zip -q -j research_plan/oracle_queries/O2_residual_after_topology_fix/validation_scripts.zip \
    scripts/z91d_bsim4_port_fit_arclength.py \
    scripts/z91e_bsim4_port_fit_with_anchors.py \
    scripts/z91f_validate_with_sebas_params.py \
    scripts/z91g_two_model_validation.py

cd "$ROOT/research_plan/oracle_queries/O2_residual_after_topology_fix"
echo "=== Packet contents ==="
ls -la
echo
echo "=== Total size ==="
du -sh .
echo
echo "=== Building O2_packet.zip ==="
rm -f O2_packet.zip
zip -q -r O2_packet.zip . -x O2_packet.zip make_packet.sh
echo "Built: $(ls -lh O2_packet.zip | awk '{print $5, $9}')"
echo
echo "Upload this single file to GPT-5 / Gemini 2.5 Pro and paste the prompt."
