#!/usr/bin/env bash
# Chip-only gen with aggressive cool-down between reps.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-/home/daedalus/embodiment21b_results}"
PROMPTS="${HERE}/prompts.json"
PY="${PY:-/home/daedalus/venvs/torch-rocm/bin/python}"
read_t(){ awk '{printf "%.1f", $1/1000}' /sys/class/thermal/thermal_zone0/temp; }

wait_below(){
    local target=$1
    local timeout=${2:-600}
    local t0=$(date +%s)
    while :; do
        local t=$(read_t)
        local cur=$(date +%s)
        if [ "$((cur - t0))" -gt "$timeout" ]; then echo "[orch] timeout T=${t}C"; return 1; fi
        local ok=$(awk -v t="$t" -v tg="$target" 'BEGIN{print (t<=tg)?1:0}')
        if [ "$ok" = "1" ]; then echo "[orch] T=${t}C <= ${target}C — proceed"; return 0; fi
        echo "[orch] cooling T=${t}C target=${target}C dt=$((cur-t0))s"
        sleep 10
    done
}

LABEL=${1:-chip}
RUN=${2:-chip_dae_200}
TARGET=${3:-100}

CK="${OUT}/ckpt_${RUN}/step_200.pt"
JSONL="${OUT}/gen_${LABEL}.jsonl"

echo "===== CHIP GEN label=${LABEL} target=${TARGET} T=$(read_t)C ====="
for sess in $(seq 1 20); do
    n=$(wc -l < "${JSONL}" 2>/dev/null || echo 0)
    n=$(echo "${n}" | tr -d ' \n')
    echo "[orch] session ${sess}: ${n}/${TARGET} T=$(read_t)C"
    [ "${n}" -ge "${TARGET}" ] && { echo "[orch] DONE"; break; }
    wait_below 42 600 || { echo "[orch] cool fail"; break; }
    timeout 480 "${PY}" "${HERE}/generate.py" --ckpt "${CK}" --prompts "${PROMPTS}" \
        --n_prompts 30 --reps 7 --max_new 80 \
        --label "${LABEL}" \
        --out_jsonl "${JSONL}" \
        --abort_c 65 --pause_c 58 --cool_c 48 --rep_idle_s 3.0 2>&1 | tail -40
done
echo "===== CHIP GEN COMPLETE n=$(wc -l < "${JSONL}" 2>/dev/null) T=$(read_t)C ====="
