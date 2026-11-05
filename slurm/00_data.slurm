#!/bin/bash
# 00_data.slurm -- CPU only: download VSI-Bench, verify pairing, extract frames.
#   sbatch slurm/00_data.slurm
#SBATCH --job-name=vsi_data
#SBATCH --time=04:00:00
#SBATCH --account=158685321433   # same allocation as ../trustfall; change if needed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=cpu
#SBATCH --output=vsi_data.%j.out

set -euo pipefail
: "${SCRATCH:?SCRATCH is not set; submit from an ACES login shell}"
cd "$SLURM_SUBMIT_DIR"

# Home is 10GB / 10k files and NOT expandable -> everything on scratch.
export VSI_ROOT="$SCRATCH/vsi"
export HF_HOME="$SCRATCH/hf"
mkdir -p "$VSI_ROOT"/{data,results,cache} "$HF_HOME"

# ACES: Python for the venv; FFmpeg for 02_frames.py; WebProxy because compute
# nodes have no direct internet access (the ~5.7 GB dataset download runs here).
if command -v module >/dev/null 2>&1; then
  module purge
fi
source envs/aces_modules.sh
if command -v module >/dev/null 2>&1; then
  module load FFmpeg/7.0.2
  module load WebProxy
else
  echo "WARNING: module command not found; assuming ffmpeg and internet access exist." >&2
fi
for exe in ffmpeg ffprobe; do
  command -v "$exe" >/dev/null 2>&1 || {
    echo "ERROR: $exe not found. On ACES load FFmpeg/7.0.2." >&2
    exit 1
  }
done

# numpy is needed by the frame extractor and by common.py; the venv has it.
source envs/activate.sh qwen3vl

python scripts/01_prepare.py          # annotations + 3 video archives (~5.7 GB)
python scripts/02_frames.py           # 32 frames per scene -> frame cache
python scripts/01_prepare.py --no-download   # re-verify manifest/video pairing
