#!/usr/bin/env bash
# Phase 21B GENS-ONLY orchestrator — run both generations from existing ckpts.
# Internal thresholds RELAXED (rely on external thermal_wrap.sh as hard safety):
#   internal: abort 70, pause 60, cool 45
#   external: abort 72, pause 58, cool 48 (independent override)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-/home/daedalus/embodiment21b_results}"
PROMPTS="${HERE}/prompts.json"
PY="${PY:-/home/daedalus/venvs/torch-rocm/bin/python}"

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
        sleep 12
    done
}

run_gen(){
    local label=$1 run_id=$2
    echo "===== GEN label=${label} T=$(read_t)C ====="
    wait_below 42 600 || { echo "[orch] cool fail; skip gen"; return 1; }
    local ck=$(ls -t "${OUT}/ckpt_${run_id}/step_"*.pt 2>/dev/null | head -1)
    if [ -z "${ck}" ]; then echo "[orch] NO CKPT for ${run_id}"; return 1; fi
    echo "[orch] using ckpt ${ck}"
    # Use stdbuf so we see every line, no tail truncation
    stdbuf -oL -eL "${PY}" "${HERE}/generate.py" --ckpt "${ck}" --prompts "${PROMPTS}" \
        --n_prompts 30 --reps 30 --max_new 200 \
        --label "${label}" \
        --out_jsonl "${OUT}/gen_${label}.jsonl" \
        --abort_c 70 --pause_c 60 --cool_c 45 2>&1
    echo "[orch] gen ${label} done T=$(read_t)C"
}

echo "============================================="
echo " Phase 21B GENS-ONLY host=$(hostname) T=$(read_t)C"
echo "============================================="

# Clear partial files so we get clean 900-each runs
rm -f "${OUT}/gen_vanilla.jsonl" "${OUT}/gen_chip.jsonl"

# If chip gen runs slower (substrate inject), do it FIRST so it doesn't compound on a hot chip
run_gen chip    chip_dae_200
run_gen vanilla vanilla_dae_200

echo "============================================="
echo " Phase 21B GENS-ONLY COMPLETE host=$(hostname) T=$(read_t)C"
echo "vanilla: $(wc -l < ${OUT}/gen_vanilla.jsonl 2>/dev/null || echo 0) lines"
echo "chip:    $(wc -l < ${OUT}/gen_chip.jsonl    2>/dev/null || echo 0) lines"
ls -la "${OUT}"
