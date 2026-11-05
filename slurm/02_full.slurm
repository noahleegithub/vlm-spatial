#!/bin/bash
# 02_full.slurm -- full run + blind control for one or more models in a single
# GPU allocation.
#
#   sbatch slurm/02_full.slurm "qwen3vl gemma4"        # ~2 h on one H100
#   sbatch --time=11:00:00 slurm/02_full.slurm molmo2  # ~6-7 h on one H100
#
# Grouping models into one allocation matters because queue waits dominate the
# runtime. Measure first (the smoke job prints seconds/question): on this
# benchmark qwen3vl runs at ~0.6 s/q, gemma4 ~2.8 s/q and molmo2 ~16 s/q, so
# molmo2 gets its own job with a long walltime rather than sharing with the pair.
#
# Everything is resume-safe: successful ids are skipped, previously failed rows
# are retried, so a walltime cut only costs a requeue.
#SBATCH --job-name=vsi_full
#SBATCH --time=03:00:00
#SBATCH --account=158685321433   # the PI's default allocation; confirm with `myproject`
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --output=vsi_full_%j.out

set -uo pipefail          # deliberately not -e: one model failing must not abort the rest
: "${SCRATCH:?SCRATCH is not set; submit from an ACES login shell}"
cd "$SLURM_SUBMIT_DIR"
MODELS="${1:-qwen3vl gemma4}"

export VSI_ROOT="$SCRATCH/vsi"
export HF_HOME="$SCRATCH/hf"
export HF_HUB_OFFLINE=1   # weights were pre-fetched on the login node
export TOKENIZERS_PARALLELISM=false
export VSI_ATTN="${VSI_ATTN:-}"

if command -v module >/dev/null 2>&1; then
  module purge
fi
source envs/aces_modules.sh
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

rc_total=0
for MODEL in $MODELS; do
  echo "=================== $MODEL: full run $(date -u +%H:%M:%S)"
  # shellcheck disable=SC1091
  source envs/activate.sh "$MODEL" >/dev/null
  python scripts/03_infer.py --model "$MODEL" \
      --tasks route_planning,object_abs_distance,room_size_estimation \
      --out "$VSI_ROOT/results/${MODEL}.jsonl"
  rc=$?; [ "$rc" -ne 0 ] && rc_total=$rc

  echo "------------------- $MODEL: blind arm $(date -u +%H:%M:%S)"
  python scripts/03_infer.py --model "$MODEL" --blind \
      --tasks object_abs_distance,room_size_estimation \
      --out "$VSI_ROOT/results/${MODEL}_blind.jsonl"
  rc=$?; [ "$rc" -ne 0 ] && rc_total=$rc

  deactivate 2>/dev/null || true
done

# Record the resolved versions and driver. The driver line only exists on a GPU
# node, which is why this runs here rather than during setup.
bash envs/freeze.sh >/dev/null 2>&1 || echo "WARN: freeze.sh failed"

echo "=================== done $(date -u +%H:%M:%S) rc=$rc_total"
exit "$rc_total"
