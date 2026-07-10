#!/usr/bin/env bash
# Build the native extension into clpipe/.
#
# CUDA is detected automatically: pass -DCLPIPE_WITH_CUDA=OFF to force the CPU
# build on a machine that has the toolkit but no device worth using.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="${root}/build"

cmake -S "${root}" -B "${build}" \
    -DCMAKE_BUILD_TYPE=Release \
    -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
    "$@"

cmake --build "${build}" --parallel

echo
python - <<'PY'
from clpipe import native
print(native.describe())
PY
