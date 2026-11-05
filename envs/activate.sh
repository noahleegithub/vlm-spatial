#!/bin/bash
# usage: source envs/activate.sh <qwen3vl|gemma4|molmo2>
#
# On ACES, load the module stack first (envs/aces_modules.sh) so the venv's
# interpreter and the newer libstdc++ are available:
#
#   source envs/aces_modules.sh
#   source envs/activate.sh qwen3vl
#
# Two interpreters, not three: Gemma 4 and Qwen 3-VL both want a recent
# transformers, while Molmo 2 pins transformers==4.57.1 and needs
# trust_remote_code. Mixing them in one env breaks one of the two.
case "${1:-}" in
  molmo2) _envdir="${SCRATCH:-$HOME}/envs/molmo2" ;;
  qwen3vl|gemma4) _envdir="${SCRATCH:-$HOME}/envs/main" ;;
  *) echo "usage: source envs/activate.sh <qwen3vl|gemma4|molmo2>" >&2; return 1 2>/dev/null || exit 1 ;;
esac
# shellcheck disable=SC1091
source "$_envdir/bin/activate"
echo "activated $_envdir"
