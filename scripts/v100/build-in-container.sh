#!/usr/bin/env bash
#
# Build the looma/v100 wheel set inside scripts/v100/Dockerfile (see scripts/v100/build.sh):
#   - runtime wheels cp310..cp313, C++ extensions linked against torch 2.11.0+cu126 and
#     libcudart.so.12 (the upstream wheels link libcudart.so.13, which a cu126 torch lacks);
#   - one kernel-cache wheel with sm_70 SASS, compiled by the image's CUDA 12.6 nvcc.
#
# Expects the source tree at /src (read-only is fine) and writes wheels to /out.
set -euo pipefail

SRC="${FREETOKEN_V100_SRC:-/src}"
OUT="${FREETOKEN_V100_OUT:-/out}"
PYTHONS="${FREETOKEN_V100_PYTHONS:-3.10 3.11 3.12 3.13}"
CACHE_PYTHON="${FREETOKEN_V100_CACHE_PYTHON:-3.12}"
TORCH="torch==2.11.0+cu126"
INDEX=(
  --index-url https://download.pytorch.org/whl/cu126
  --extra-index-url https://pypi.org/simple
  --index-strategy unsafe-best-match
)

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }

mkdir -p "$OUT" /work
rsync -a --delete --exclude .git --exclude dist --exclude build --exclude .venv \
  --exclude __pycache__ --exclude jit_cache "$SRC"/ /work/src/
cd /work/src

for v in $PYTHONS; do
  venv="/work/venv$v"
  say "python $v: build venv with $TORCH"
  uv venv -q --clear "$venv" --python "$v"
  uv pip install -q --python "$venv/bin/python" "${INDEX[@]}" "$TORCH" "setuptools>=77" wheel ninja
  say "python $v: runtime wheel"
  uv build -q --wheel --no-build-isolation --python "$venv/bin/python" -o "$OUT" .
done

if [ "${FREETOKEN_V100_SKIP_KERNEL_CACHE:-0}" = 1 ]; then
  ls -la "$OUT"
  exit 0
fi

venv="/work/venv$CACHE_PYTHON"
runtime="$(ls "$OUT"/freetoken-*-cp${CACHE_PYTHON/./}-*.whl | head -1)"
say "kernel cache: installing $runtime into the $CACHE_PYTHON venv"
uv pip install -q --python "$venv/bin/python" "${INDEX[@]}" "$TORCH" "$runtime"
say "kernel cache: compiling for TVM_FFI_CUDA_ARCH_LIST=${FREETOKEN_KERNEL_CACHE_ARCHES:-7.0}"
(cd freetoken-kernel-cache && uv build -q --wheel --no-build-isolation --python "$venv/bin/python" -o "$OUT" .)

say "wheels in $OUT:"
ls -la "$OUT"
