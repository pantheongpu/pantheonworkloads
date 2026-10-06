#!/usr/bin/env bash
# Build llama.cpp at the pinned commit for one backend.
#
#   tools/llamacpp/build.sh <cpu|cuda|hip> <prefix>
#
# Installs llama-completion and llama-bench (and the ggml libraries beside them)
# into <prefix>/bin, <prefix>/lib. Exit 77 when this machine cannot build the
# backend (no cmake, compiler, git, network, nvcc or hipcc), so a workload
# calling this reports SKIP instead of FAIL.
#
# Environment:
#   LLAMACPP_REPO     git URL (default https://github.com/ggml-org/llama.cpp, MIT)
#   LLAMACPP_COMMIT   commit to build (default: the pin in tools/llamacpp/pin.env)
#   PW_BUILD_JOBS     parallel jobs (default 2)
#   PW_CUDA_ARCHS     CMAKE_CUDA_ARCHITECTURES for cuda (default: cmake's choice)
#   PW_HIP_ARCHS      AMDGPU_TARGETS for hip (default: gfx942, an MI300-class part;
#                     set it for the card you test on, for example gfx1100)
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=pin.env
. "$here/pin.env"
backend="${1:-}"; prefix="${2:-}"
[[ -n "$prefix" ]] || { echo "usage: build.sh <cpu|cuda|hip> <prefix>" >&2; exit 2; }
skip() { echo "SKIP: $*"; exit 77; }
for t in git cmake; do command -v "$t" >/dev/null || skip "$t is not installed"; done
command -v c++ >/dev/null || skip "no C++ compiler"
flags=(-DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
       -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_OPENSSL=OFF)
case "$backend" in
  cpu)  flags+=(-DGGML_NATIVE=OFF) ;;   # portable x86-64 build; reproducible across hosts
  cuda) command -v nvcc >/dev/null || skip "nvcc (CUDA toolkit) is not installed"
        flags+=(-DGGML_CUDA=ON)
        [[ -z "${PW_CUDA_ARCHS:-}" ]] || flags+=("-DCMAKE_CUDA_ARCHITECTURES=$PW_CUDA_ARCHS") ;;
  hip)  hipcc=$(command -v hipcc) || skip "hipcc (ROCm) is not installed"
        rocm=$(dirname "$(dirname "$(readlink -f "$hipcc")")")
        flags+=(-DGGML_HIP=ON "-DAMDGPU_TARGETS=${PW_HIP_ARCHS:-gfx942}"
                "-DCMAKE_PREFIX_PATH=$rocm" "-DCMAKE_C_COMPILER=$rocm/lib/llvm/bin/clang"
                "-DCMAKE_CXX_COMPILER=$rocm/lib/llvm/bin/clang++") ;;
  *) echo "unknown backend '$backend' (cpu, cuda, hip)" >&2; exit 2 ;;
esac
commit="${LLAMACPP_COMMIT:-$LLAMACPP_PINNED_COMMIT}"
repo="${LLAMACPP_REPO:-https://github.com/ggml-org/llama.cpp}"
work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT
src="$work/src"
git init -q "$src" && git -C "$src" fetch -q --depth 1 "$repo" "$commit" 2>"$work/git.err" \
  && git -C "$src" checkout -q FETCH_HEAD || { cat "$work/git.err" >&2; skip "cannot fetch llama.cpp $commit from $repo"; }
got=$(git -C "$src" rev-parse HEAD)
[[ "$got" == "$commit" || ${#commit} -lt 40 ]] || { echo "fetched $got, wanted $commit" >&2; exit 1; }
echo "building llama.cpp $got ($backend) into $prefix" >&2
cmake -S "$src" -B "$work/build" "${flags[@]}" -DCMAKE_INSTALL_PREFIX="$prefix" >"$work/cmake.log" 2>&1 \
  || { tail -20 "$work/cmake.log" >&2; skip "cmake configure for $backend failed"; }
cmake --build "$work/build" -j"${PW_BUILD_JOBS:-2}" --target llama-completion llama-bench >"$work/build.log" 2>&1 \
  || { tail -20 "$work/build.log" >&2; echo "build of $backend failed" >&2; exit 1; }
# Install only the two tools and the shared libraries they load.
mkdir -p "$prefix/bin" "$prefix/lib"
cp "$work/build/bin/llama-completion" "$work/build/bin/llama-bench" "$prefix/bin/"
find "$work/build" -name '*.so*' \( -type f -o -type l \) -exec cp -P {} "$prefix/lib/" \;
echo "$got" > "$prefix/COMMIT"
