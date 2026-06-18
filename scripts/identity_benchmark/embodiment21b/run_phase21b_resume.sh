#!/usr/bin/env bash
# Phase 21B RESUME orchestrator — chip training (vanilla already done) + both gens.
# Tightened internal thermal: 65 abort / 58 pause / 45 cool (matches external wrapper 72/58/48).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-/home/daedalus/embodiment21b_results}"
PROMPTS="${HERE}/prompts.json"
PY="${PY:-/home/daedalus/venvs/torch-rocm/bin/python}"
mkdir -p "${OUT}"

read_t(){ awk '{printf "%.1f", $1/1000}' /sys/class/thermal/thermal_zone0/temp; }

wait_below(){
    local target=$1
    local timeout=${2:-900}
    local t0=$(date +%s)
    while :; do
        local t=$(read_t)
        local cur=$(date +%s)
        if [ "$((cur - t0))" -gt "$timeout" ]; then
            echo "[orch] wait_below ${target}C TIMEOUT at T=${t}C"
            return 1
        fi
        local ok=$(awk -v t="$t" -v tg="$target" 'BEGIN{print (t<=tg)?1:0}')
        if [ "$ok" = "1" ]; then
            echo "[orch] T=${t}C <= ${target}C — proceed"
            return 0
        fi
        echo "[orch] cooling... T=${t}C target=${target}C dt=$((cur-t0))s"
        sleep 15
    done
}

run_train(){
    local cond=$1 run_id=$2
    echo "===== TRAIN ${run_id} cond=${cond} T=$(read_t)C ====="
    wait_below 42 600 || { echo "[orch] cool fail; ship-what-we-got"; return 1; }
    for sess in 1 2 3 4 5 6 7 8; do
        echo "----- session ${sess} cond=${cond} T=$(read_t)C -----"
        "${PY}" "${HERE}/train.py" --cond "${cond}" --run_id "${run_id}" \
            --steps 200 --ckpt_every 10 --bsz 1 --block_size 128 \
            --abort_c 65 --pause_c 58 --cool_c 45 \
            --session_max_s 400 --out "${OUT}" 2>&1 | tail -200
        local steps_done
        steps_done=$(python3 -c "import json; d=json.load(open('${OUT}/train_log_${run_id}.json')); print(d.get('steps_done',0))" 2>/dev/null || echo 0)
        echo "[orch] ${run_id} session ${sess} done — steps_done=${steps_done}/200 T=$(read_t)C"
        if [ "${steps_done}" -ge "200" ]; then break; fi
        wait_below 42 600 || break
    done
}

run_gen(){
    local label=$1 run_id=$2
    echo "===== GEN label=${label} T=$(read_t)C ====="
    wait_below 42 600 || { echo "[orch] cool fail; skip gen"; return 1; }
    local ck=$(ls -t "${OUT}/ckpt_${run_id}/step_"*.pt 2>/dev/null | head -1)
    if [ -z "${ck}" ]; then echo "[orch] NO CKPT for ${run_id}"; return 1; fi
    echo "[orch] using ckpt ${ck}"
    "${PY}" "${HERE}/generate.py" --ckpt "${ck}" --prompts "${PROMPTS}" \
        --n_prompts 30 --reps 30 --max_new 200 \
        --label "${label}" \
        --out_jsonl "${OUT}/gen_${label}.jsonl" \
        --abort_c 65 --pause_c 58 --cool_c 45 2>&1 | tail -100
    echo "[orch] gen ${label} done T=$(read_t)C"
}

echo "============================================="
echo " Phase 21B RESUME host=$(hostname) T=$(read_t)C"
echo "============================================="

# vanilla already done — skip
if [ ! -f "${OUT}/train_log_chip_dae_200.json" ] || \
   [ "$(python3 -c "import json; print(json.load(open('${OUT}/train_log_chip_dae_200.json')).get('steps_done',0))" 2>/dev/null || echo 0)" -lt 200 ]; then
    run_train chip chip_dae_200
else
    echo "[orch] chip training already complete — skip"
fi

run_gen vanilla vanilla_dae_200
run_gen chip    chip_dae_200

echo "============================================="
echo " Phase 21B RESUME COMPLETE host=$(hostname) T=$(read_t)C"
ls -la "${OUT}"
