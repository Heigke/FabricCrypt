#!/bin/bash
# auto_resume_after_reboot.sh — Auto-launch after reboot
# 1. Run diagnostics
# 2. Resume Claude Code conversation with results

cd /home/ikaros/Documents/claude_hive/AMD_gfx1151_energy

# Wait for system to settle (network, GPU driver, etc)
sleep 15

# Run diagnostics first
echo "Running post-reboot diagnostics..."
bash scripts/post_reboot_fw_direct_test.sh

# Get the conversation ID from the session file
SESSION_DIR="/home/ikaros/.claude/projects/-home-ikaros-Documents-claude-hive-AMD-gfx1151-energy"
# Find the most recent session
LATEST_SESSION=$(ls -t "$SESSION_DIR"/*.jsonl 2>/dev/null | head -1 | xargs -I{} basename {} .jsonl)

RESULTS_FILE="scripts/fw_direct_results.txt"

# Resume Claude Code with the results
echo "Resuming Claude Code..."
cd /home/ikaros/Documents/claude_hive/AMD_gfx1151_energy

# Use --print to pipe the prompt, --dangerously-skip-permissions for auto mode
claude --dangerously-skip-permissions -p "I just rebooted with amdgpu.fw_load_type=0. Read the diagnostics at /home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/scripts/fw_direct_results.txt and analyze whether DIRECT firmware loading worked. If the GPU initialized with DIRECT mode, proceed to write a kernel module that reads IC_BASE, MES_CNTL, and PC registers to confirm firmware state. If it failed, analyze why from dmesg and try alternative approaches. The full PSP bypass research context is in memory at psp_bypass_results.md."
