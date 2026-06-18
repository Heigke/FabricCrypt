#!/usr/bin/env bash
# Phase 21B — generation-only orchestrator. Runs many short gen sessions with
# cool periods between, resumes from existing JSONL.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-/home/daedalus/embodiment21b_results}"
PROMPTS="${HERE}/prompts.json"
PY="${PY:-/home/daedalus/venvs/torch-rocm/bin/python}"
mkdir -p "${OUT}"

read_t(){ awk '{printf "%.1f", $1/1000}' /sys/class/thermal/thermal_zone0/temp; }

wait_below(){
    local target=$1
    local timeout=${2:-600}
    local t0=$(date +%s)
    while :; do
        local t=$(read_t)
        local cur=$(date +%s)
        if [ "$((cur - t0))" -gt "$timeout" ]; then
            echo "[gen-orch] wait_below ${target}C TIMEOUT at T=${t}C"
            return 1
        fi
        local ok=$(awk -v t="$t" -v tg="$target" 'BEGIN{print (t<=tg)?1:0}')
        if [ "$ok" = "1" ]; then
            echo "[gen-orch] T=${t}C <= ${target}C — proceed"
            return 0
        fi
        echo "[gen-orch] cooling... T=${t}C target=${target}C dt=$((cur-t0))s"
        sleep 15
    done
}

run_gen_loop(){
    local label=$1 run_id=$2
    local target=$((30*7))  # 210 samples per model
    echo "===== GEN ${label} target=${target} T=$(read_t)C ====="
    local ck="${OUT}/ckpt_${run_id}/step_200.pt"
    if [ ! -f "${ck}" ]; then echo "[gen-orch] NO CKPT ${ck}"; return 1; fi
    local jsonl="${OUT}/gen_${label}.jsonl"
    # Start fresh: keep existing JSONL only if it has samples for the right model;
    # otherwise wipe. We pass --label so the resume logic only resumes its own.
    for sess in $(seq 1 12); do
        local n_done
        n_done=$(wc -l < "${jsonl}" 2>/dev/null || echo 0)
        echo "[gen-orch] ${label} session ${sess}: ${n_done}/${target} T=$(read_t)C"
        if [ "${n_done}" -ge "${target}" ]; then
            echo "[gen-orch] ${label} DONE"
            return 0
        fi
        wait_below 50 600 || { echo "[gen-orch] cool fail"; return 1; }
        timeout 600 "${PY}" "${HERE}/generate.py" --ckpt "${ck}" --prompts "${PROMPTS}" \
            --n_prompts 30 --reps 7 --max_new 80 \
            --label "${label}" \
            --out_jsonl "${jsonl}" \
            --abort_c 68 --pause_c 62 --cool_c 50 --rep_idle_s 2.0 2>&1 | tail -60
    done
}

echo "============================================="
echo " GEN-ONLY START host=$(hostname) T=$(read_t)C"
echo "============================================="

run_gen_loop vanilla vanilla_dae_200
run_gen_loop chip    chip_dae_200

echo "============================================="
echo " GEN-ONLY COMPLETE host=$(hostname) T=$(read_t)C"
wc -l "${OUT}"/gen_*.jsonl
