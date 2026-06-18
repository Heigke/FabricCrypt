#!/bin/bash
# Phase 21 end-to-end orchestration on daedalus.
# Trains 3 conditions (vanilla, chip_inject_dae, synthetic_matched),
# generates 50 prompts x 2 reps x 200 tokens per ckpt,
# runs stylometric classifier, writes results.
#
# Run from daedalus: ssh daedalus@daedalus.local 'bash ~/phase21/scripts/run_phase21.sh'

set -e
cd "$(dirname "$0")"
PY=$HOME/venvs/torch-rocm/bin/python
RES=$HOME/phase21/results
mkdir -p "$RES"

echo "[phase21] === HOST $(hostname) ==="
echo "[phase21] === sig sanity ==="
$PY _common.py

# ---- TRAIN (sequential, GPU constrained) ----
STEPS=${STEPS:-500}
MODEL=${MODEL:-distilgpt2}
ALPHA=${ALPHA:-1e-3}

echo "[phase21] === TRAIN 1/3: vanilla ==="
$PY train_personality.py --cond vanilla --run_id vanilla_dae \
    --steps $STEPS --model $MODEL --lr 1e-4 --bsz 2 --ckpt_every 100 \
    --out $RES --abort_c 80 --pause_c 72 --cool_c 65

echo "[phase21] === TRAIN 2/3: chip_inject (daedalus) ==="
$PY train_personality.py --cond chip_inject --run_id chip_dae \
    --steps $STEPS --model $MODEL --lr 1e-4 --bsz 2 --ckpt_every 100 \
    --alpha $ALPHA --out $RES --abort_c 80 --pause_c 72 --cool_c 65

echo "[phase21] === TRAIN 3/3: synthetic_matched ==="
$PY train_personality.py --cond synthetic_matched --run_id synth_dae \
    --steps $STEPS --model $MODEL --lr 1e-4 --bsz 2 --ckpt_every 100 \
    --alpha $ALPHA --out $RES --abort_c 80 --pause_c 72 --cool_c 65

# ---- GENERATE (final ckpts) ----
echo "[phase21] === GENERATE completions (each: 50 prompts x 2 reps x 200 tok) ==="
for run in vanilla_dae chip_dae synth_dae; do
    CKPT=$(ls -t $RES/ckpt_$run/step_*.pt 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        echo "[phase21] no ckpt for $run, skipping"
        continue
    fi
    echo "[phase21] generating for $run from $CKPT"
    $PY generate.py --ckpt "$CKPT" --model $MODEL --prompts prompts.json \
        --label "$run" --reps 2 --max_new 200 --temperature 0.9 \
        --out_jsonl $RES/gen_${run}.jsonl --seed 42
done

# ---- STYLOMETRY + CLASSIFIER ----
echo "[phase21] === STYLOMETRY (chip vs vanilla) ==="
$PY stylometry.py \
    --jsonl $RES/gen_chip_dae.jsonl $RES/gen_vanilla_dae.jsonl \
    --out_dir $RES/stylometry_chip_vs_vanilla \
    --classes chip_dae vanilla_dae

echo "[phase21] === STYLOMETRY (chip vs synth) — discriminates real chip from random noise ==="
$PY stylometry.py \
    --jsonl $RES/gen_chip_dae.jsonl $RES/gen_synth_dae.jsonl \
    --out_dir $RES/stylometry_chip_vs_synth \
    --classes chip_dae synth_dae

echo "[phase21] === STYLOMETRY (all 3 classes) ==="
$PY stylometry.py \
    --jsonl $RES/gen_chip_dae.jsonl $RES/gen_vanilla_dae.jsonl $RES/gen_synth_dae.jsonl \
    --out_dir $RES/stylometry_three_way

echo "[phase21] === DONE ==="
ls -lah $RES/
