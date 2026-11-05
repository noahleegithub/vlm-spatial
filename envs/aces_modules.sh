#!/bin/bash
# Load the ACES module stack used by this project.
#
# Source this from an interactive login shell or from a Slurm job before
# activating a venv:
#
#   source envs/aces_modules.sh
#
# ACES does not load any modules by default.  The venvs are built against the
# Python below, so the same toolchain must be present when they are used (Torch
# also needs the newer libstdc++ from GCCcore).  If HPRC changes versions, update
# the two module names after checking `module spider Python/3.12.3`.
#
# This intentionally does not `module purge`: setup may be running in a compute
# job with WebProxy already loaded.  Slurm scripts that want a clean stack purge
# first; interactive users should do the same.
#
# If `module` is unavailable (e.g. running on a laptop), this is a no-op and the
# caller is responsible for providing a Python >= 3.11 on PATH.

if [ "${VSI_SKIP_MODULES:-0}" = "1" ]; then
  return 0 2>/dev/null || exit 0
fi

if command -v module >/dev/null 2>&1; then
  module load GCCcore/13.3.0
  module load Python/3.12.3
else
  echo "WARNING: 'module' not found; assuming a Python >= 3.11 is already on PATH." >&2
fi
