#!/usr/bin/env bash
#
# Build the looma/v100 wheels in a linux/amd64 CUDA 12.6 container (works from an
# Apple Silicon Mac through emulation, just slower). Wheels land in ./dist/v100.
#
#   scripts/v100/build.sh
#
# FREETOKEN_V100_PYTHONS="3.12" narrows the runtime matrix for a quick iteration;
# FREETOKEN_V100_SKIP_KERNEL_CACHE=1 adds runtime wheels to dist/v100 without
# rebuilding the kernel cache (or removing what is already there).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
IMAGE="${FREETOKEN_V100_IMAGE:-looma-ft-v100-build}"
OUT="$ROOT/dist/v100"

docker build --platform linux/amd64 -t "$IMAGE" "$ROOT/scripts/v100"
mkdir -p "$OUT"
[ "${FREETOKEN_V100_SKIP_KERNEL_CACHE:-0}" = 1 ] || rm -f "$OUT"/*.whl
docker run --rm --platform linux/amd64 \
  -v "$ROOT":/src:ro -v "$OUT":/out -v looma-ft-v100-uv:/cache/uv \
  -e FREETOKEN_V100_PYTHONS -e FREETOKEN_KERNEL_CACHE_ARCHES -e FREETOKEN_V100_SKIP_KERNEL_CACHE \
  "$IMAGE" bash /src/scripts/v100/build-in-container.sh
