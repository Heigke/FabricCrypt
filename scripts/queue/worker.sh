#!/usr/bin/env bash
# NS-RAM cluster worker. Pure bash + ssh.
#
# Usage: worker.sh <node_name> [worker_slot]
#   node_name: ikaros | daedalus | zgx
#   worker_slot: integer suffix (default 0) — allow multiple workers per host
#
# Pulls jobs from master (ikaros) via atomic mv-claim, runs them, reports.
# Thermal-safe: warn at 85C, kill at 92C. GPU warn at 80C.
#
# IMPORTANT: Worker scripts are deliberately conservative. Failed jobs go to
# failed/ — never silently retried.
set -u  # NB: not -e; we want to handle errors explicitly
set -o pipefail

NODE="${1:?usage: worker.sh <node> [slot]}"
SLOT="${2:-0}"
WORKER_ID="${NODE}_${SLOT}"

# ---------- node-specific configuration ----------
MASTER_USER="ikaros"
MASTER_HOST="192.168.0.35"
MASTER_REPO="/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy"
QUEUE="${MASTER_REPO}/research_plan/job_queue"

case "$NODE" in
  ikaros)
    LOCAL_REPO="$MASTER_REPO"
    PYTHON_BIN="${LOCAL_REPO}/venv/bin/python"
    HAS_REPO=1
    HAS_GPU_AMD=1
    HAS_GPU_NVIDIA=0
    ;;
  daedalus)
    LOCAL_REPO="/home/daedalus/AMD_gfx1151_energy"
    PYTHON_BIN="/home/daedalus/venvs/torch-rocm/bin/python"
    HAS_REPO=1
    HAS_GPU_AMD=1
    HAS_GPU_NVIDIA=0
    ;;
  zgx)
    LOCAL_REPO="/home/naorw/nsram_queue_sandbox"
    PYTHON_BIN="/home/naorw/nsram_venv/bin/python"
    HAS_REPO=0       # repo not cloned; we scp script files per job
    HAS_GPU_AMD=0
    HAS_GPU_NVIDIA=1
    ;;
  *)
    echo "FATAL: unknown node $NODE" >&2
    exit 2
    ;;
esac

mkdir -p "$LOCAL_REPO" "$LOCAL_REPO/results" "$LOCAL_REPO/results/queue_logs" 2>/dev/null || true

POLL_INTERVAL=15
THERMAL_WARN=85000      # millicelsius
THERMAL_KILL=92000
GPU_WARN_NVIDIA=80
GPU_WARN_AMD=80
MAX_WALL_SEC=7200       # 2h hard cap

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ) $WORKER_ID] $*" | tee -a "$LOCAL_REPO/results/queue_logs/worker_${WORKER_ID}.log"
}

# Read APU thermal in millicelsius. Returns 0 on any error (treat as safe).
read_thermal_apu() {
  if [ -r /sys/class/thermal/thermal_zone0/temp ]; then
    cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo 0
  else
    echo 0
  fi
}

read_gpu_temp() {
  if [ "$HAS_GPU_NVIDIA" = "1" ]; then
    nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0
  elif [ "$HAS_GPU_AMD" = "1" ]; then
    # AMD hwmon edge temp in millidegrees
    local t
    t=$(cat /sys/class/drm/card?/device/hwmon/hwmon*/temp1_input 2>/dev/null | head -1)
    if [ -n "$t" ]; then echo $((t / 1000)); else echo 0; fi
  else
    echo 0
  fi
}

# Thermal gate. Returns 0 if safe to proceed, non-zero if we slept and should retry.
thermal_check() {
  local apu gpu
  apu=$(read_thermal_apu)
  gpu=$(read_gpu_temp)
  if [ "$apu" -ge "$THERMAL_KILL" ]; then
    log "THERMAL KILL: APU=${apu}mC >= ${THERMAL_KILL}mC. Killing python procs, sleep 120."
    pkill -9 -f "$PYTHON_BIN" 2>/dev/null || true
    sleep 120
    return 1
  fi
  if [ "$apu" -ge "$THERMAL_WARN" ]; then
    log "THERMAL WARN: APU=${apu}mC >= ${THERMAL_WARN}mC. Sleep 60."
    sleep 60
    return 1
  fi
  if [ "$gpu" -ge "$GPU_WARN_NVIDIA" ] && [ "$HAS_GPU_NVIDIA" = "1" ]; then
    log "GPU WARN: NVIDIA=${gpu}C. Sleep 30."
    sleep 30
    return 1
  fi
  if [ "$gpu" -ge "$GPU_WARN_AMD" ] && [ "$HAS_GPU_AMD" = "1" ]; then
    log "GPU WARN: AMD=${gpu}C. Sleep 30."
    sleep 30
    return 1
  fi
  return 0
}

# Wrappers for ssh-to-master. Master may BE this host (ikaros worker).
master_run() {
  if [ "$NODE" = "ikaros" ]; then
    bash -c "$1"
  else
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
        "${MASTER_USER}@${MASTER_HOST}" "$1"
  fi
}

master_scp_down() {
  # scp from master to local
  local remote="$1" local_path="$2"
  if [ "$NODE" = "ikaros" ]; then
    cp -f "$remote" "$local_path"
  else
    scp -q -o BatchMode=yes -o StrictHostKeyChecking=no \
        "${MASTER_USER}@${MASTER_HOST}:${remote}" "$local_path"
  fi
}

master_scp_up() {
  local local_path="$1" remote="$2"
  if [ "$NODE" = "ikaros" ]; then
    cp -f "$local_path" "$remote"
  else
    scp -q -o BatchMode=yes -o StrictHostKeyChecking=no \
        "$local_path" "${MASTER_USER}@${MASTER_HOST}:${remote}"
  fi
}

# Try to claim oldest pending job. Echoes claimed JOB_ID on stdout, empty if none.
claim_job() {
  # List pending sorted oldest first (mtime asc).
  local listing
  listing=$(master_run "ls -tr ${QUEUE}/pending/*.json 2>/dev/null | head -20" || true)
  [ -z "$listing" ] && return 0
  local jpath jid claimed
  while IFS= read -r jpath; do
    [ -z "$jpath" ] && continue
    jid=$(basename "$jpath" .json)
    # Skip if local pgrep finds a python proc already running this script name
    # (rough no-overlap protection by script-base-name).
    # Get script name first via a tiny ssh roundtrip is too expensive; rely on
    # post-claim check inside run_job() to detect dup and abort gracefully.
    claimed=$(master_run "mv ${QUEUE}/pending/${jid}.json ${QUEUE}/running/${jid}.${NODE}.json 2>/dev/null && echo OK" || true)
    if [ "$claimed" = "OK" ]; then
      echo "$jid"
      return 0
    fi
  done <<< "$listing"
  return 0
}

# Extract a JSON field (string) using python3 if available, else grep fallback.
json_field() {
  local file="$1" key="$2"
  python3 -c "import json,sys; d=json.load(open('$file')); v=d.get('$key'); print('' if v is None else (' '.join(v) if isinstance(v,list) else v))" 2>/dev/null
}

json_env_pairs() {
  python3 -c "import json; d=json.load(open('$1')); [print(f'{k}={v}') for k,v in (d.get('env_vars') or {}).items()]" 2>/dev/null
}

json_args_array() {
  # Print each arg on its own line, NUL-safe enough for our usage (no newlines in args)
  python3 -c "import json; d=json.load(open('$1')); [print(a) for a in (d.get('args') or [])]" 2>/dev/null
}

run_job() {
  local jid="$1"
  local json_remote="${QUEUE}/running/${jid}.${NODE}.json"
  local json_local="$LOCAL_REPO/results/queue_logs/${jid}.json"
  local logf="$LOCAL_REPO/results/queue_logs/${NODE}_${jid}.log"

  master_scp_down "$json_remote" "$json_local" || {
    log "ERROR: cannot scp job json for $jid; releasing"
    master_run "mv ${json_remote} ${QUEUE}/pending/${jid}.json 2>/dev/null" || true
    return 1
  }

  local script args_csv surrogate out_dir
  script=$(json_field "$json_local" script)
  out_dir=$(json_field "$json_local" out_dir)
  surrogate=$(json_field "$json_local" needs_surrogate)

  if [ -z "$script" ]; then
    log "ERROR: job $jid missing 'script' field"
    fail_job "$jid" "missing script field"
    return 1
  fi

  # No-overlap: skip if this script base-name already running locally
  local script_base
  script_base=$(basename "$script")
  if pgrep -af "$script_base" | grep -v "$$" | grep -v "worker.sh" | grep -q python; then
    log "DUP: $script_base already running on $NODE. Releasing $jid back to pending."
    master_run "mv ${json_remote} ${QUEUE}/pending/${jid}.json 2>/dev/null" || true
    return 1
  fi

  # Ensure script is available locally
  local script_local="$LOCAL_REPO/$script"
  if [ ! -f "$script_local" ]; then
    if [ "$HAS_REPO" = "1" ]; then
      log "WARNING: $script_local missing on $NODE (repo present). Fetching from master."
    fi
    mkdir -p "$(dirname "$script_local")"
    master_scp_down "${MASTER_REPO}/${script}" "$script_local" || {
      log "ERROR: cannot fetch $script from master"
      fail_job "$jid" "script not found on master: $script"
      return 1
    }
  fi

  # Fetch surrogate if requested
  if [ -n "$surrogate" ] && [ "$surrogate" != "None" ]; then
    local sur_local="$LOCAL_REPO/$surrogate"
    if [ ! -f "$sur_local" ]; then
      mkdir -p "$(dirname "$sur_local")"
      master_scp_down "${MASTER_REPO}/${surrogate}" "$sur_local" || {
        log "ERROR: cannot fetch surrogate $surrogate"
        fail_job "$jid" "surrogate fetch failed: $surrogate"
        return 1
      }
    fi
  fi

  # Build env + args
  local env_prefix=""
  while IFS= read -r kv; do
    [ -z "$kv" ] && continue
    env_prefix="$env_prefix $kv"
  done < <(json_env_pairs "$json_local")

  # Read args into a bash array
  local -a job_args=()
  while IFS= read -r a; do
    [ -z "$a" ] && continue
    job_args+=("$a")
  done < <(json_args_array "$json_local")

  log "RUN start jid=$jid script=$script args=(${job_args[*]}) env=($env_prefix)"
  local t0 t1 rc
  t0=$(date +%s)
  # Execute. Use timeout. Redirect to logfile. cd into LOCAL_REPO.
  (
    cd "$LOCAL_REPO" || exit 99
    # shellcheck disable=SC2086
    env $env_prefix timeout --kill-after=30 "$MAX_WALL_SEC" \
        "$PYTHON_BIN" "$script_local" "${job_args[@]}"
  ) > "$logf" 2>&1
  rc=$?
  t1=$(date +%s)
  log "RUN end jid=$jid rc=$rc duration=$((t1 - t0))s log=$logf"

  if [ "$rc" -eq 0 ]; then
    JPATH_ENV="$json_local" DUR_ENV="$((t1 - t0))" WORKER_ENV="$WORKER_ID" \
      LOGF_ENV="$logf" python3 - <<'PYEOF' 2>/dev/null || true
import json, os, time
p = os.environ['JPATH_ENV']
d = json.load(open(p))
d['completed_at'] = time.time()
d['completed_iso'] = time.strftime('%Y-%m-%dT%H:%M:%S')
d['duration_sec'] = int(os.environ['DUR_ENV'])
d['worker'] = os.environ['WORKER_ENV']
d['log_path'] = os.environ['LOGF_ENV']
d['return_code'] = 0
open(p, 'w').write(json.dumps(d, indent=2))
PYEOF
    master_scp_up "$json_local" "${QUEUE}/done/${jid}.json"
    master_run "rm -f ${json_remote}"
    log "DONE jid=$jid"
    return 0
  else
    fail_job "$jid" "rc=$rc see log $logf" "$logf" "$t0" "$t1"
    return 1
  fi
}

fail_job() {
  local jid="$1" msg="$2" logf="${3:-}" t0="${4:-0}" t1="${5:-0}"
  local json_remote="${QUEUE}/running/${jid}.${NODE}.json"
  local json_local="$LOCAL_REPO/results/queue_logs/${jid}.json"
  # Ensure local json exists (might be called before scp_down succeeded)
  if [ ! -f "$json_local" ]; then
    echo "{\"id\":\"$jid\"}" > "$json_local"
  fi
  MSG_ENV="$msg" LOGF_ENV="$logf" JID_ENV="$jid" WORKER_ENV="$WORKER_ID" \
    DUR_ENV="$((t1 - t0))" JPATH_ENV="$json_local" python3 - <<'PYEOF' 2>/dev/null || true
import json, os, time
p = os.environ['JPATH_ENV']
try:
    d = json.load(open(p))
except Exception:
    d = {'id': os.environ['JID_ENV']}
d.setdefault('retry_count', 0)
d['failed_at'] = time.time()
d['failed_iso'] = time.strftime('%Y-%m-%dT%H:%M:%S')
d['worker'] = os.environ['WORKER_ENV']
d['error'] = os.environ['MSG_ENV']
logf = os.environ.get('LOGF_ENV', '')
if logf and os.path.exists(logf):
    try:
        d['log_tail'] = open(logf, 'rb').read()[-4000:].decode('utf-8', errors='replace')
    except Exception as e:
        d['log_tail'] = f'<read err: {e}>'
else:
    d['log_tail'] = ''
try:
    d['duration_sec'] = int(os.environ['DUR_ENV'])
except Exception:
    pass
open(p, 'w').write(json.dumps(d, indent=2))
PYEOF
  master_scp_up "$json_local" "${QUEUE}/failed/${jid}.json" 2>/dev/null || true
  master_run "rm -f ${json_remote}"
  log "FAIL jid=$jid msg=$msg"
}

orphan_sweep() {
  # Only run from one place (the ikaros worker) — others skip.
  [ "$NODE" = "ikaros" ] || return 0
  local now
  now=$(date +%s)
  local f mt age jid base
  for f in "${QUEUE}/running/"*.json; do
    [ -e "$f" ] || continue
    mt=$(stat -c %Y "$f" 2>/dev/null || echo 0)
    age=$((now - mt))
    if [ "$age" -gt 10800 ]; then   # 3h
      base=$(basename "$f" .json)
      jid="${base%.*}"   # strip .nodeName suffix
      log "ORPHAN: $f age=${age}s -> recycle to pending as $jid (retry++)"
      JPATH_ENV="$f" JID_ENV="$jid" python3 - <<'PYEOF' 2>/dev/null || true
import json, os, time
p = os.environ['JPATH_ENV']
try:
    d = json.load(open(p))
except Exception:
    d = {'id': os.environ['JID_ENV']}
d['retry_count'] = d.get('retry_count', 0) + 1
d['orphaned_at'] = time.time()
open(p, 'w').write(json.dumps(d, indent=2))
PYEOF
      retries=$(JPATH_ENV="$f" python3 -c "import json,os; print(json.load(open(os.environ['JPATH_ENV'])).get('retry_count',0))" 2>/dev/null || echo 99)
      if [ "$retries" -le 3 ]; then
        mv "$f" "${QUEUE}/pending/${jid}.json" 2>/dev/null || true
      else
        mv "$f" "${QUEUE}/failed/${jid}.json" 2>/dev/null || true
        log "ORPHAN: $jid exhausted retries -> failed/"
      fi
    fi
  done
}

# ---------------- main loop ----------------
log "worker boot: node=$NODE slot=$SLOT python=$PYTHON_BIN repo=$LOCAL_REPO"

while true; do
  if ! thermal_check; then
    continue
  fi

  orphan_sweep

  jid=$(claim_job || true)
  if [ -z "$jid" ]; then
    sleep "$POLL_INTERVAL"
    continue
  fi

  log "CLAIMED $jid"
  run_job "$jid" || true
  # brief cooldown between jobs to let thermals settle
  sleep 5
done
