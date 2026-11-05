#!/bin/bash
# Freeze the exact interpreter and package versions after the smoke test.
#
#   bash envs/freeze.sh
#
# Writes envs/lock-<date>.txt. Nothing in setup_envs.sh pins transformers for
# the 'main' env (it installs the newest), so a rebuild weeks later can resolve
# differently; the lock file is what makes the reported numbers reproducible.
# Paste its torch/transformers lines into the report's setup table.
set -euo pipefail

: "${SCRATCH:?SCRATCH is not set; run this on an ACES login node}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/.."

# shellcheck source=envs/aces_modules.sh
source "$SCRIPT_DIR/aces_modules.sh"

OUT="envs/lock-$(date -u +%F).txt"
{
  echo "# generated $(date -u +%FT%TZ)"
  echo "# host      $(hostname 2>/dev/null || echo unknown)"
  echo "# modules   $(module list 2>&1 | tr '\n' ' ' | sed 's/  */ /g')"
  # On a cpu node nvidia-smi exists but cannot reach a driver; probe with -L and
  # record either the hardware or an explicit "none", never the driver error text.
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "# gpu       $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)"
  else
    echo "# gpu       none on this node (regenerate this lock on a GPU node for the driver line)"
  fi
  for e in main molmo2; do
    echo
    echo "## env: $e  ($SCRATCH/envs/$e)"
    # shellcheck disable=SC1091
    if ! source "$SCRATCH/envs/$e/bin/activate" 2>/dev/null; then
      echo "  MISSING -- run envs/setup_envs.sh"
      continue
    fi
    python -VV
    python - <<'PY'
import json
try:
    import torch
    print("torch", torch.__version__, "| cuda", torch.version.cuda,
          "| cuda_available", torch.cuda.is_available())
except Exception as e:
    print("torch MISSING:", e)
try:
    import transformers
    print("transformers", transformers.__version__)
except Exception as e:
    print("transformers MISSING:", e)
PY
    python -m pip freeze 2>/dev/null \
      | grep -iE '^(torch|torchvision|transformers|accelerate|pillow|numpy|huggingface|qwen-vl|molmo|decord)' || true
    deactivate
  done
} | tee "$OUT"

echo
echo "wrote $OUT -- put the torch/transformers lines in the report's setup table."
