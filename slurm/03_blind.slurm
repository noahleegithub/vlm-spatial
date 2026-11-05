#!/bin/bash
# 03_blind.slurm -- the control condition: same questions, NO video frames.
# If MRA barely drops without any visual input, the model is reciting object
# priors rather than measuring the room. This arm is cheap (no vision tokens)
# and is the strongest single piece of evidence in the report.
#   sbatch --dependency=afterok:<data-job-id> slurm/03_blind.slurm qwen3vl
#SBATCH --job-name=vsi_blind
#SBATCH --time=01:00:00
#SBATCH --account=158685321433   # same allocation as ../trustfall; change if needed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --output=vsi_blind_%j.out

set -euo pipefail
: "${SCRATCH:?SCRATCH is not set; submit from an ACES login shell}"
cd "$SLURM_SUBMIT_DIR"
MODEL="${1:?usage: sbatch slurm/03_blind.slurm <qwen3vl|gemma4|molmo2>}"

export VSI_ROOT="$SCRATCH/vsi"
export HF_HOME="$SCRATCH/hf"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

if command -v module >/dev/null 2>&1; then
  module purge
fi
source envs/aces_modules.sh
source envs/activate.sh "$MODEL"

python scripts/03_infer.py --model "$MODEL" --blind \
    --tasks object_abs_distance,room_size_estimation \
    --out "$VSI_ROOT/results/${MODEL}_blind.jsonl"

# Scoring is intentionally not run here: see the note in slurm/02_infer.slurm.
