#!/bin/bash
# Phase 22 queue: replay light identity signals on daedalus once Phase 21b
# training is done.  Read-only / sysfs-only — does NOT load CPU or GPU.
#
# Pre-reqs (on daedalus):
#   - sudo -n dmidecode  (for S23) — optional, falls back to /sys/class/dmi/id
#   - sudo -n umr        (for S26) — optional, falls back to empty vector
#
# Usage (from ikaros):
#   bash scripts/identity_benchmark/embodiment22/queue_daedalus.sh
set -euo pipefail

: "${DAEDALUS_HOST:=192.168.0.37}"
: "${DAEDALUS_USER:=daedalus}"
: "${DAEDALUS_PASS:=daedalus}"

LOCAL_REPO="/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy"
REMOTE_REPO="/home/daedalus/AMD_gfx1151_energy"

# 1) rsync the embodiment22 scripts to daedalus
sshpass -p "$DAEDALUS_PASS" rsync -av --delete \
    "$LOCAL_REPO/scripts/identity_benchmark/embodiment22/" \
    "${DAEDALUS_USER}@${DAEDALUS_HOST}:${REMOTE_REPO}/scripts/identity_benchmark/embodiment22/"

# 2) ensure results dir exists
sshpass -p "$DAEDALUS_PASS" ssh "${DAEDALUS_USER}@${DAEDALUS_HOST}" \
    "mkdir -p ${REMOTE_REPO}/results/IDENTITY_BENCHMARK_2026-05-30/embodiment22"

# 3) run all 8 signals, 10 reps; uses daedalus' python (assumes numpy+scipy
#    in either system or torch-rocm venv).
sshpass -p "$DAEDALUS_PASS" ssh "${DAEDALUS_USER}@${DAEDALUS_HOST}" bash -lc "'
cd $REMOTE_REPO
source venvs/torch-rocm/bin/activate 2>/dev/null || true
python scripts/identity_benchmark/embodiment22/run_all.py 10 \
    > /tmp/embodiment22_daedalus.log 2>&1
tail -30 /tmp/embodiment22_daedalus.log
'"

# 4) pull results back to ikaros
sshpass -p "$DAEDALUS_PASS" rsync -av \
    "${DAEDALUS_USER}@${DAEDALUS_HOST}:${REMOTE_REPO}/results/IDENTITY_BENCHMARK_2026-05-30/embodiment22/daedalus_*" \
    "$LOCAL_REPO/results/IDENTITY_BENCHMARK_2026-05-30/embodiment22/"

# 5) cross-host KS analysis
cd "$LOCAL_REPO"
source venv/bin/activate
python scripts/identity_benchmark/embodiment22/analyze.py
