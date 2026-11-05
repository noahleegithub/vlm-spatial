#!/bin/bash
# 02_infer.slurm -- one model, all three tasks, resume-safe.
#   sbatch --dependency=afterok:<smoke-job-id> slurm/02_infer.slurm qwen3vl
#   sbatch --dependency=afterok:<smoke-job-id> slurm/02_infer.slurm gemma4
#   sbatch --dependency=afterok:<smoke-job-id> slurm/02_infer.slurm molmo2
#
# Resubmitting the same command continues where it stopped (results are
# appended per-question with fsync, and completed ids are skipped).  Previously
# failed rows are retried on the next run.
#SBATCH --job-name=vsi_infer
#SBATCH --time=04:00:00
#SBATCH --account=158685321433   # same allocation as ../trustfall; change if needed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --output=vsi_infer_%j.out

set -euo pipefail
: "${SCRATCH:?SCRATCH is not set; submit from an ACES login shell}"
cd "$SLURM_SUBMIT_DIR"
MODEL="${1:?usage: sbatch slurm/02_infer.slurm <qwen3vl|gemma4|molmo2>}"

export VSI_ROOT="$SCRATCH/vsi"
export HF_HOME="$SCRATCH/hf"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VSI_ATTN="${VSI_ATTN:-}"

if command -v module >/dev/null 2>&1; then
  module purge
fi
source envs/aces_modules.sh
source envs/activate.sh "$MODEL"
python -c "import torch,transformers;print('torch',torch.__version__,'tf',transformers.__version__)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python scripts/03_infer.py --model "$MODEL" \
    --tasks route_planning,object_abs_distance,room_size_estimation \
    --out "$VSI_ROOT/results/${MODEL}.jsonl"

# Scoring is intentionally not run here: scripts/04_score.py writes global
# results/scores.csv, table.md and stats.json, so concurrent model jobs would
# overwrite each other.  Run it once after all inference jobs finish (README step 6).
