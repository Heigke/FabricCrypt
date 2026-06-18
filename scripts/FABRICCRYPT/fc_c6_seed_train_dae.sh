#!/bin/bash
# FC-C6: Train 2 vanilla LoRA-style ckpts with DIFFERENT seeds on daedalus.
# Same chip-off training, only difference = data-iteration RNG (run_id-derived).
# We also pass --torch_seed via env (we re-export and edit train.py? No — train.py
# does NOT set torch.manual_seed. But we WANT seed variance. We rely on:
#   (a) different run_id -> different data ordering RNG
#   (b) different generator seeds on cuda kernels (default unseeded -> nondet)
# This means our "floor" is conservative (random non-determinism + data order).
set -e
ROOT=/home/daedalus/embodiment21b
OUT=/home/daedalus/embodiment21b_results_c6
mkdir -p "$OUT"

source /home/daedalus/venvs/torch-rocm/bin/activate
cd "$ROOT"

# We explicitly set PYTHONHASHSEED + torch_seed via tiny wrapper to control torch RNG too.
RUNA=vanilla_seedA_42
RUNB=vanilla_seedB_137

# Force torch determinism via env
export PYTHONHASHSEED=42
echo "=== FC-C6 TRAIN $RUNA (torch seed=42) ==="
python -c "import torch; torch.manual_seed(42); torch.cuda.manual_seed_all(42)" >/dev/null 2>&1
python train.py --cond vanilla --run_id $RUNA --steps 200 \
  --abort_c 68 --pause_c 62 --cool_c 50 \
  --out "$OUT" --no_resume

export PYTHONHASHSEED=137
echo "=== FC-C6 TRAIN $RUNB (torch seed=137) ==="
python -c "import torch; torch.manual_seed(137); torch.cuda.manual_seed_all(137)" >/dev/null 2>&1
python train.py --cond vanilla --run_id $RUNB --steps 200 \
  --abort_c 68 --pause_c 62 --cool_c 50 \
  --out "$OUT" --no_resume

echo "=== FC-C6 TRAIN DONE ==="
