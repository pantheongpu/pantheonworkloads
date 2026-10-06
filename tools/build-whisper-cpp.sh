#!/usr/bin/env bash
# Builds whisper.cpp at a pinned tag for one backend and prints the directory holding
# whisper-cli and whisper-bench on the last stdout line.
#
#   tools/build-whisper-cpp.sh cpu|cuda|hip
#
#   PW_CACHE         build cache (default ~/.cache/pantheonworkloads)
#   PW_BUILD_JOBS    make parallelism (default 2: hosts are shared)
#   PW_CUDA_ARCH     CMAKE_CUDA_ARCHITECTURES for cuda (default: native)
#   PW_HIP_ARCH      AMDGPU_TARGETS for hip, e.g. gfx942 (required for hip)
#   PW_WHISPER_BIN_DIR  use an existing build instead (must hold whisper-cli, whisper-bench)
#
# Exit 77 when the build cannot be done here (missing tool, failed build), as the workload
# contract says; the reason is on stdout.
set -uo pipefail
TAG=v1.9.5
COMMIT=d1be6fde11ac6e0407606b4e42fe72d34add8037   # what the tag must resolve to
REPO=https://github.com/ggml-org/whisper.cpp
backend="${1:-cpu}"
skip() { echo "SKIP: $*"; exit 77; }

if [[ -n "${PW_WHISPER_BIN_DIR:-}" ]]; then
  [[ -x "$PW_WHISPER_BIN_DIR/whisper-cli" && -x "$PW_WHISPER_BIN_DIR/whisper-bench" ]] \
    || skip "PW_WHISPER_BIN_DIR has no whisper-cli/whisper-bench"
  echo "$PW_WHISPER_BIN_DIR"; exit 0
fi
cache="${PW_CACHE:-$HOME/.cache/pantheonworkloads}/whisper.cpp"
src="$cache/src-$TAG"
bld="$cache/build-$TAG-$backend"
if [[ -x "$bld/bin/whisper-cli" && -x "$bld/bin/whisper-bench" ]]; then echo "$bld/bin"; exit 0; fi

for t in git cmake c++; do command -v "$t" >/dev/null || skip "$t is not installed (needed to build whisper.cpp)"; done
flags=(-DCMAKE_BUILD_TYPE=Release -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_SERVER=OFF)
case "$backend" in
  cpu) ;;
  cuda)
    command -v nvcc >/dev/null || skip "nvcc is not installed (CUDA toolkit needed for the cuda backend)"
    flags+=(-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${PW_CUDA_ARCH:-native}") ;;
  hip)
    command -v hipconfig >/dev/null || skip "hipconfig is not installed (ROCm needed for the hip backend)"
    [[ -n "${PW_HIP_ARCH:-}" ]] || skip "set PW_HIP_ARCH to the GPU architecture, e.g. gfx942"
    export HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)"
    flags+=(-DGGML_HIP=ON -DAMDGPU_TARGETS="$PW_HIP_ARCH") ;;
  *) skip "unknown backend '$backend' (cpu, cuda or hip)" ;;
esac

mkdir -p "$cache" || skip "cannot create $cache"
if [[ ! -d "$src/.git" ]]; then
  git clone -q --depth 1 --branch "$TAG" "$REPO" "$src.tmp" >&2 || { rm -rf "$src.tmp"; skip "cannot clone $REPO at $TAG"; }
  mv "$src.tmp" "$src"
fi
[[ "$(git -C "$src" rev-parse HEAD)" == "$COMMIT" ]] || skip "$TAG in $src is not commit $COMMIT"
{ cmake -S "$src" -B "$bld" "${flags[@]}" \
  && cmake --build "$bld" -j "${PW_BUILD_JOBS:-2}" --target whisper-cli whisper-bench; } >&2 \
  || skip "whisper.cpp $TAG ($backend) failed to build"
echo "$bld/bin"
