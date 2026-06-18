#!/bin/bash
# FC-C5: Generate on IKAROS using daedalus-trained ckpts.
# Mirrors run_chip_gen.sh on daedalus but uses local paths.
set -e
cd "$(dirname "$0")/.."/identity_benchmark/embodiment21b

ROOT=/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy
GEN_OUT=$ROOT/results/FABRICCRYPT/c5_causal
CKPT_CHIP=$ROOT/models/FABRICCRYPT_c5/ckpt_chip_dae_200/step_200.pt
CKPT_VAN=$ROOT/models/FABRICCRYPT_c5/ckpt_vanilla_dae_200/step_200.pt
PROMPTS=$ROOT/scripts/identity_benchmark/embodiment21b/prompts.json

mkdir -p "$GEN_OUT"

export HSA_OVERRIDE_GFX_VERSION=11.0.0
PY=$ROOT/venv/bin/python

# Reps and prompts: match daedalus run (15 reps × 30 prompts = 450 max, but
# daedalus got 73/108 due to thermal; we mirror counts).
REPS=15
N_PROMPTS=30
SEED=42

echo "=== FC-C5 GEN @ ikaros: chip ckpt ==="
$PY $ROOT/scripts/identity_benchmark/embodiment21b/generate.py \
  --ckpt "$CKPT_CHIP" --model distilgpt2 \
  --prompts "$PROMPTS" --n_prompts $N_PROMPTS --reps $REPS --max_new 100 \
  --temperature 0.9 --top_p 0.95 --seed $SEED \
  --abort_c 68 --pause_c 62 --cool_c 55 --rep_idle_s 0.2 \
  --out_jsonl "$GEN_OUT/gen_chip_ikaros.jsonl" --label chip_ikaros

echo "=== FC-C5 GEN @ ikaros: vanilla ckpt ==="
$PY $ROOT/scripts/identity_benchmark/embodiment21b/generate.py \
  --ckpt "$CKPT_VAN" --model distilgpt2 \
  --prompts "$PROMPTS" --n_prompts $N_PROMPTS --reps $REPS --max_new 100 \
  --temperature 0.9 --top_p 0.95 --seed $SEED \
  --abort_c 68 --pause_c 62 --cool_c 55 --rep_idle_s 0.2 \
  --out_jsonl "$GEN_OUT/gen_vanilla_ikaros.jsonl" --label vanilla_ikaros

echo "=== FC-C5 GEN @ ikaros: DONE ==="
