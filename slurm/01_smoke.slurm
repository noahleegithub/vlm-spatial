#!/bin/bash
# 01_smoke.slurm -- ONE real model, 4 questions, on a GPU. Run this before any
# long job: it resolves the three unknowns that differ across transformers
# builds (image content key, processor class, chat-template quirks) in ~5 min.
#   sbatch --dependency=afterok:<data-job-id> slurm/01_smoke.slurm qwen3vl
#   sbatch --dependency=afterok:<data-job-id> slurm/01_smoke.slurm gemma4
#   sbatch --dependency=afterok:<data-job-id> slurm/01_smoke.slurm molmo2
#SBATCH --job-name=vsi_smoke
#SBATCH --time=00:40:00
#SBATCH --account=158685321433   # same allocation as ../trustfall; change if needed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --output=vsi_smoke_%j.out

set -euo pipefail
: "${SCRATCH:?SCRATCH is not set; submit from an ACES login shell}"
cd "$SLURM_SUBMIT_DIR"
MODEL="${1:?usage: sbatch slurm/01_smoke.slurm <qwen3vl|gemma4|molmo2>}"

export VSI_ROOT="$SCRATCH/vsi"
export HF_HOME="$SCRATCH/hf"
export HF_HUB_OFFLINE=1          # weights pre-fetched on the login node
export TOKENIZERS_PARALLELISM=false
export VSI_ATTN="${VSI_ATTN:-}"  # set to flash_attention_2 if flash-attn is installed

# One interpreter per model family. See envs/setup_envs.sh.
if command -v module >/dev/null 2>&1; then
  module purge
fi
source envs/aces_modules.sh
source envs/activate.sh "$MODEL"

echo "== env";  python -c "import torch,transformers;print('torch',torch.__version__,'tf',transformers.__version__,'cuda',torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

SMOKE_OUT="$VSI_ROOT/results/smoke/${MODEL}.jsonl"
mkdir -p "$(dirname "$SMOKE_OUT")"
rm -f "$SMOKE_OUT"                # smoke is a fresh 4-question check every time

# Keep the exit status so failures are visible in the job state, but still print
# the rows below to show which processor/content-key form failed.
set +e
python -u scripts/03_infer.py --model "$MODEL" \
    --tasks route_planning,object_abs_distance \
    --limit 2 --out "$SMOKE_OUT"
rc=$?
set -e

echo "== smoke rows"
if [ -f "$SMOKE_OUT" ]; then
    python - "$SMOKE_OUT" <<'PY' || rc=1
import json
import sys

path = sys.argv[1]
errors = 0
for line in open(path):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    if "error" in r:
        errors += 1
        print("ERROR", r.get("id"), r["error"])
    else:
        print(r["task"], "| GT", r["ground_truth"],
              "| pred", repr(r.get("raw_output"))[:90],
              "| key", r.get("content_key"))
sys.exit(1 if errors else 0)
PY
else
    echo "ERROR: smoke produced no output at $SMOKE_OUT" >&2
    rc=2
fi

exit "$rc"
