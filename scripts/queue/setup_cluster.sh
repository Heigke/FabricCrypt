#!/usr/bin/env bash
# Bring up the NS-RAM cluster job-queue workers.
#
# - Creates queue dirs on master
# - Verifies SSH to daedalus + zgx
# - Ships worker.sh to each node
# - Starts persistent workers in `screen` (zgx) or `tmux` (ikaros, daedalus)
# - Verifies all 3 workers are running
#
# Idempotent: safe to re-run. Existing live workers are left alone.

set -u
set -o pipefail

MASTER_REPO="/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy"
QUEUE="${MASTER_REPO}/research_plan/job_queue"
WORKER_SRC="${MASTER_REPO}/scripts/queue/worker.sh"

# NOTE (2026-05-17): daedalus IP corrected to 192.168.0.40 (was 192.168.0.37 in CLAUDE.md/README).
DAE_USER="daedalus"; DAE_HOST="192.168.0.40"; DAE_PASS="daedalus"
ZGX_USER="naorw";    ZGX_HOST="192.168.0.41"; ZGX_PASS="kernel"

# Use a NAMED session distinct from existing experiments (d2_a, d2_b on zgx).
SESSION="nsram_queue_worker"

say() { echo "[setup] $*"; }
die() { echo "[setup FATAL] $*" >&2; exit 1; }

# 1) dirs
say "creating queue directory structure"
mkdir -p "${QUEUE}/pending" "${QUEUE}/running" "${QUEUE}/done" "${QUEUE}/failed"
mkdir -p "${MASTER_REPO}/results/queue_logs"

# 2) verify SSH
say "verifying SSH to daedalus"
sshpass -p "$DAE_PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 \
    "${DAE_USER}@${DAE_HOST}" "echo OK_DAE; hostname" \
    | grep -q OK_DAE || die "daedalus SSH failed"

say "verifying SSH to zgx"
sshpass -p "$ZGX_PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 \
    "${ZGX_USER}@${ZGX_HOST}" "echo OK_ZGX; hostname" \
    | grep -q OK_ZGX || die "zgx SSH failed"

# 3) ship worker.sh
say "shipping worker.sh to daedalus"
sshpass -p "$DAE_PASS" ssh -o StrictHostKeyChecking=no "${DAE_USER}@${DAE_HOST}" "mkdir -p ~/nsram_queue"
sshpass -p "$DAE_PASS" scp -o StrictHostKeyChecking=no "$WORKER_SRC" "${DAE_USER}@${DAE_HOST}:~/nsram_queue/worker.sh"
sshpass -p "$DAE_PASS" ssh -o StrictHostKeyChecking=no "${DAE_USER}@${DAE_HOST}" "chmod +x ~/nsram_queue/worker.sh"

say "shipping worker.sh to zgx"
sshpass -p "$ZGX_PASS" ssh -o StrictHostKeyChecking=no "${ZGX_USER}@${ZGX_HOST}" "mkdir -p ~/nsram_queue ~/nsram_queue_sandbox"
sshpass -p "$ZGX_PASS" scp -o StrictHostKeyChecking=no "$WORKER_SRC" "${ZGX_USER}@${ZGX_HOST}:~/nsram_queue/worker.sh"
sshpass -p "$ZGX_PASS" ssh -o StrictHostKeyChecking=no "${ZGX_USER}@${ZGX_HOST}" "chmod +x ~/nsram_queue/worker.sh"

# 4) start workers (skip if already alive)
start_local_ikaros() {
  if pgrep -af "scripts/queue/worker.sh ikaros" | grep -v grep >/dev/null; then
    say "ikaros worker already alive — leaving as is"
    return 0
  fi
  say "starting ikaros worker in tmux session '$SESSION'"
  # Kill stale empty session of same name if it exists.
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  tmux new-session -d -s "$SESSION" \
      "bash ${WORKER_SRC} ikaros 0 2>&1 | tee -a ${MASTER_REPO}/results/queue_logs/worker_ikaros_0.tmuxlog"
}

start_remote_daedalus() {
  local check
  check=$(sshpass -p "$DAE_PASS" ssh -o StrictHostKeyChecking=no "${DAE_USER}@${DAE_HOST}" \
      "pgrep -af 'nsram_queue/worker.sh daedalus' | grep -v grep | wc -l")
  if [ "${check:-0}" -gt 0 ]; then
    say "daedalus worker already alive"
    return 0
  fi
  say "starting daedalus worker in tmux session '$SESSION'"
  sshpass -p "$DAE_PASS" ssh -o StrictHostKeyChecking=no "${DAE_USER}@${DAE_HOST}" \
      "tmux kill-session -t $SESSION 2>/dev/null; tmux new-session -d -s $SESSION 'bash ~/nsram_queue/worker.sh daedalus 0 2>&1 | tee -a ~/worker_daedalus.tmuxlog'"
}

start_remote_zgx() {
  # zgx has screen and existing d2_a/d2_b sessions. Use a different name.
  local check
  check=$(sshpass -p "$ZGX_PASS" ssh -o StrictHostKeyChecking=no "${ZGX_USER}@${ZGX_HOST}" \
      "pgrep -af 'nsram_queue/worker.sh zgx' | grep -v grep | wc -l")
  if [ "${check:-0}" -gt 0 ]; then
    say "zgx worker already alive"
    return 0
  fi
  say "starting zgx worker in screen session '$SESSION'"
  sshpass -p "$ZGX_PASS" ssh -o StrictHostKeyChecking=no "${ZGX_USER}@${ZGX_HOST}" \
      "screen -wipe >/dev/null 2>&1; screen -dmS $SESSION bash -c 'bash ~/nsram_queue/worker.sh zgx 0 2>&1 | tee -a ~/worker_zgx.screenlog'"
}

start_local_ikaros
start_remote_daedalus
start_remote_zgx

# 5) verify
sleep 3
say "verifying workers"
LOCAL_OK=0; DAE_OK=0; ZGX_OK=0
pgrep -af "scripts/queue/worker.sh ikaros" | grep -v grep >/dev/null && LOCAL_OK=1
sshpass -p "$DAE_PASS" ssh -o StrictHostKeyChecking=no "${DAE_USER}@${DAE_HOST}" \
    "pgrep -af 'nsram_queue/worker.sh daedalus' | grep -v grep | wc -l" \
    | grep -q '^[1-9]' && DAE_OK=1
sshpass -p "$ZGX_PASS" ssh -o StrictHostKeyChecking=no "${ZGX_USER}@${ZGX_HOST}" \
    "pgrep -af 'nsram_queue/worker.sh zgx' | grep -v grep | wc -l" \
    | grep -q '^[1-9]' && ZGX_OK=1

echo
echo "===== worker status ====="
echo "ikaros:   $([ $LOCAL_OK -eq 1 ] && echo ALIVE || echo DEAD)"
echo "daedalus: $([ $DAE_OK -eq 1 ] && echo ALIVE || echo DEAD)"
echo "zgx:      $([ $ZGX_OK -eq 1 ] && echo ALIVE || echo DEAD)"
echo
if [ $LOCAL_OK -eq 1 ] && [ $DAE_OK -eq 1 ] && [ $ZGX_OK -eq 1 ]; then
  say "ALL 3 WORKERS RUNNING"
  exit 0
else
  die "one or more workers failed to start"
fi
