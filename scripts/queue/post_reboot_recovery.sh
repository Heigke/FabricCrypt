#!/bin/bash
# Post-reboot recovery for ikaros.
# Run this in /home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/ after reboot.
set -u
ROOT=/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy
cd "$ROOT"

echo "=== POST-REBOOT RECOVERY $(date) ==="
echo

# 1. Restart sentinel (thermal watchdog)
echo "--- sentinel ---"
if pgrep -f sentinel.sh >/dev/null; then
  echo "sentinel already alive PID=$(pgrep -f sentinel.sh)"
else
  nohup bash $ROOT/scripts/sentinel.sh > /tmp/sentinel.out 2>&1 &
  echo "sentinel started PID=$!"
fi

# 2. Restart ikaros queue worker in tmux
echo
echo "--- queue worker (ikaros) ---"
tmux kill-session -t nsram_queue_worker 2>/dev/null
sleep 1
tmux new-session -d -s nsram_queue_worker "bash $ROOT/scripts/queue/worker.sh ikaros"
sleep 2
if tmux has-session -t nsram_queue_worker 2>/dev/null; then
  echo "ikaros worker UP in tmux session 'nsram_queue_worker'"
else
  echo "FAILED to start ikaros worker"
fi

# 3. Verify daedalus + zgx workers (they should have survived since they're remote)
echo
echo "--- remote workers ---"
sshpass -p daedalus ssh -o ConnectTimeout=5 daedalus@daedalus.local \
  "tmux has-session -t nsram_queue_worker 2>/dev/null && echo daedalus_UP || echo daedalus_DEAD" 2>&1 | tail -1
sshpass -p kernel ssh -o ConnectTimeout=5 naorw@192.168.0.41 \
  "screen -ls 2>/dev/null | grep -q nsram_queue_worker && echo zgx_UP || echo zgx_DEAD" 2>&1 | tail -1

# 4. Verify queue state intact
echo
echo "--- queue state ---"
venv/bin/python scripts/queue/status.py 2>&1 | head -10

# 5. Reminder for cron jobs (session-only, need re-creation by Claude)
echo
echo "--- CRON JOBS ---"
echo "Session-only cron jobs DIE on reboot."
echo "After Claude resumes, ask Claude to re-create the 8 jobs:"
echo "  - 21 */3 * * *   MEP+DS campaign every 3h"
echo "  - 47 * * * *     hourly idle check"
echo "  - 41 */6 * * *   oracle critique 6h"
echo "  - 37 11,23 * * * oracle synth 12h"
echo "  - 11 9,15,21 * * * track audit 6h"
echo "  - 43 4 * * *     baseline watchdog"
echo "  - 13 2 * * *     daily synth"
echo "  - 23 6 * * *     morning brief"

echo
echo "=== RECOVERY DONE $(date) ==="
echo
echo "--- AUTO-RESUMING CLAUDE WITH /remote-control ---"
SESSION_ID="83353dd8-59e1-4c51-a16d-0c4cceb1d1b4"
echo "Session: $SESSION_ID"
echo "Resuming in detached tmux session 'claude_session' with auto /remote-control..."

# Kill any existing claude_session tmux
tmux kill-session -t claude_session 2>/dev/null

# Start Claude in detached tmux so it has a TTY AND persists across SSH disconnects.
# The trailing slash-command "/remote-control" is sent as the initial prompt,
# activating Remote Control immediately on resume.
tmux new-session -d -s claude_session -c "$ROOT" \
  "claude --resume $SESSION_ID '/remote-control'"

sleep 3
if tmux has-session -t claude_session 2>/dev/null; then
  echo "claude_session tmux UP. Connect from phone via Remote Control."
  echo "  - To attach locally: tmux attach -t claude_session"
  echo "  - To detach without killing: Ctrl-B then D"
else
  echo "FAILED to start claude_session. Falling back to manual:"
  echo "  cd $ROOT && claude --resume $SESSION_ID '/remote-control'"
fi

echo
echo "Remote Control should now be active and reachable from phone."
echo "After connecting via phone, tell Claude:"
echo "    'recreate cron jobs from NOVEL_DS_PLAN'"
echo "to re-schedule the 8 session-only cron jobs that died with the prior Claude process."
