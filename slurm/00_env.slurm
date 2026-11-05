#!/bin/bash
# 00_env.slurm -- build the two Python interpreters on $SCRATCH.
#
# Run this instead of calling envs/setup_envs.sh on a login node: the torch
# wheel set is ~2.5 GB and ACES login nodes have a CPU-time cap that will kill
# the build mid-download. Compute nodes have no direct egress, so the job loads
# WebProxy before pip/uv touch the network.
#
#   sbatch slurm/00_env.slurm
#SBATCH --job-name=vsi_env
#SBATCH --time=02:00:00
#SBATCH --account=158685321433
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu
#SBATCH --output=vsi_env.%j.out

set -euo pipefail
: "${SCRATCH:?SCRATCH is not set; submit from an ACES login shell}"
cd "$SLURM_SUBMIT_DIR"

export VSI_ROOT="$SCRATCH/vsi"
export HF_HOME="$SCRATCH/hf"
mkdir -p "$VSI_ROOT" "$HF_HOME"

module purge
source envs/aces_modules.sh     # GCCcore + Python; does not purge (keeps WebProxy)
module load WebProxy            # compute nodes have no direct internet access

echo "== python"; python3 -V; command -v python3
echo "== proxy"; env | grep -i _proxy | sed 's/\(pass\|token\)[^ ]*/\1***/I' || true

bash envs/setup_envs.sh

# Record the resolved versions; quote these in the report. nvidia-smi is absent
# on cpu nodes, which freeze.sh handles.
bash envs/freeze.sh || echo "WARN: freeze.sh failed; run it again after the smoke test"

echo "== overlay check: the two envs must not share a transformers build"
for e in main molmo2; do
  # shellcheck disable=SC1091
  source "$SCRATCH/envs/$e/bin/activate"
  echo "$e -> $(python -c 'import transformers,sys;print(sys.version.split()[0], transformers.__version__)' 2>&1 | tail -1)"
  deactivate
done
