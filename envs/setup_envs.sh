#!/bin/bash
# Build the two interpreters on $SCRATCH. Run on the LOGIN node
# (it needs network); the GPU jobs then run offline.
#
#   source envs/aces_modules.sh    # or run `module load ...` yourself
#   bash envs/setup_envs.sh
#
# ACES has no modules loaded by default and its OS Python is too old for
# current Torch/Transformers, so this script requires SCRATCH and a Python
# >= 3.11.  It sources envs/aces_modules.sh to load the expected module stack.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

: "${SCRATCH:?SCRATCH is not set; run this on an ACES login node with SCRATCH exported}"

# shellcheck source=envs/aces_modules.sh
source "$SCRIPT_DIR/aces_modules.sh"

BASE="$SCRATCH/envs"
mkdir -p "$BASE"

PY="${PYTHON_BIN:-python3}"
if ! PY="$(command -v "$PY")"; then
  echo "ERROR: Python interpreter not found: ${PYTHON_BIN:-python3}" >&2
  echo "       On ACES: module purge; module load GCCcore/13.3.0 Python/3.12.3" >&2
  exit 1
fi
if ! "$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
  echo "ERROR: $("$PY" --version 2>&1) is too old; Python >= 3.11 is required." >&2
  echo "       On ACES: module purge; module load GCCcore/13.3.0 Python/3.12.3" >&2
  exit 1
fi

CUDA_INDEX="https://download.pytorch.org/whl/cu126"

# Keep every cache on scratch.  Home is only 10 GB / 10,000 files on ACES.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SCRATCH/.cache/pip}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$SCRATCH/.cache/uv-python}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH/.cache}"
export HF_HOME="${HF_HOME:-$SCRATCH/hf}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export UV_PYTHON_DOWNLOADS=never
mkdir -p "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$XDG_CACHE_HOME" "$HF_HOME"

echo "python: $("$PY" --version)  ($PY)"
echo "scratch: $SCRATCH"
echo "hf home: $HF_HOME"

make_env() {   # $1 = name
  local d="$BASE/$1"
  if [ -d "$d" ]; then
    if [ -x "$d/bin/python" ] \
       && "$d/bin/python" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null \
       && "$d/bin/python" -m pip --version >/dev/null 2>&1; then
      echo "[skip] $d"
      return
    fi
    echo "[warn] $d exists but is incomplete or has an old Python; recreating" >&2
    rm -rf "$d"
  fi

  if command -v uv >/dev/null 2>&1; then
    # --seed is essential: a bare `uv venv` does not install pip.
    uv venv --seed --python "$PY" "$d" || true
  fi
  if [ ! -x "$d/bin/python" ] \
     || ! "$d/bin/python" -m pip --version >/dev/null 2>&1; then
    if command -v uv >/dev/null 2>&1; then
      echo "[warn] uv venv failed or left no pip; falling back to $PY -m venv" >&2
    fi
    rm -rf "$d"
    "$PY" -m venv "$d"
  fi
  if ! "$d/bin/python" -m pip --version >/dev/null 2>&1; then
    echo "ERROR: $d has no pip; use a Python build that includes ensurepip." >&2
    exit 1
  fi
  echo "[ok] $d"
  # shellcheck disable=SC1091
  source "$d/bin/activate"
  python -m pip install -q --upgrade pip setuptools wheel
  deactivate
}

install_common() {   # $1 = env dir
  # shellcheck disable=SC1091
  source "$1/bin/activate"
  python -m pip install -q --upgrade pip setuptools wheel
  python -m pip install -q torch torchvision --index-url "$CUDA_INDEX"
  python -m pip install -q pillow numpy accelerate "huggingface_hub[cli]"
  deactivate
}

echo "== env 'main'  (Qwen 3-VL + Gemma 4: recent transformers)"
make_env main
install_common "$BASE/main"
# shellcheck disable=SC1091
source "$BASE/main/bin/activate"
python -m pip install -q --upgrade transformers
python -m pip install -q qwen-vl-utils
echo "   transformers: $(python -c 'import transformers;print(transformers.__version__)')"
# Optional but worth it on H100 for 32-frame inputs (~30% faster prefill):
#   python -m pip install flash-attn --no-build-isolation
# then export VSI_ATTN=flash_attention_2
deactivate

echo "== env 'molmo2' (transformers pinned to the version Molmo 2 ships against)"
make_env molmo2
install_common "$BASE/molmo2"
# shellcheck disable=SC1091
source "$BASE/molmo2/bin/activate"
python -m pip install -q "transformers==4.57.1" molmo_utils
echo "   transformers: $(python -c 'import transformers;print(transformers.__version__)')"
deactivate

echo
echo "done. Activate with:  source envs/activate.sh <qwen3vl|gemma4|molmo2>"
echo "Pre-fetch weights on the login node (compute nodes may be offline):"
echo "  source envs/activate.sh qwen3vl"
echo "  export HF_HOME=\$SCRATCH/hf"
echo "  hf download Qwen/Qwen3-VL-8B-Instruct"
echo "  hf download google/gemma-4-12B-it"
echo "  hf download allenai/Molmo2-8B"
