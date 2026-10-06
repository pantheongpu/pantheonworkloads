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
#   LLAMACPP_EXTRA_TARGETS  more cmake targets to build and install beside the two (space separated,
#                     e.g. "llama-quantize llama-perplexity llama-debug"; the synthetic-GGUF workloads use these)
#   LLAMACPP_BUILD_PROBE  1: also compile probe/pw-probe.cpp (token ids + logit statistics; used by the
#                     llamacpp-synth-* workloads) against the build and install it as bin/pw-probe
#   LLAMACPP_BUILD_TESTS  1: configure with -DLLAMA_BUILD_TESTS=ON (for LLAMACPP_EXTRA_TARGETS="test-backend-ops", which
#                     compares every operator on a GPU backend with the CPU's)
#   PW_BUILD_DIR      keep the source and build tree here (default: a temporary directory, deleted at the end):
#                     a second run resumes where the first stopped, so a rebuild after a change takes seconds
#   PW_BUILD_JOBS     parallel jobs (default 2)
#   PW_CUDA_ARCHS     CMAKE_CUDA_ARCHITECTURES for cuda (default: cmake's choice)
#   PW_HIP_ARCHS      AMDGPU_TARGETS for hip (default: gfx942, an MI300-class part, or gfx90a when hipcc is
#                     ROCm 5.x, the newest CDNA part whose rocBLAS kernels Ubuntu's 5.5.1 ships; set it for the
#                     card you test on, for example gfx1100)
#   PW_HIP_EXTRA_PREFIX  <prefix>/usr made by tools/llamacpp/hip-apt-prefix.sh: rocBLAS and hipBLAS (headers,
#                     libraries, kernels for the architectures asked) when they are not installed
#   PW_HIP_FA, PW_HIP_GRAPHS  1: build FlashAttention kernels / HIP graphs (off with ROCm before 6.1, on otherwise)
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=pin.env
. "$here/pin.env"
backend="${1:-}"; prefix="${2:-}"
[[ -n "$prefix" ]] || { echo "usage: build.sh <cpu|cuda|hip> <prefix>" >&2; exit 2; }
skip() { echo "SKIP: $*"; exit 77; }
for t in git cmake; do command -v "$t" >/dev/null || skip "$t is not installed"; done
command -v c++ >/dev/null || skip "no C++ compiler"
flags=(-DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=${LLAMACPP_BUILD_TESTS:-0} -DLLAMA_BUILD_EXAMPLES=OFF
       -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_OPENSSL=OFF)
case "$backend" in
  cpu)  flags+=(-DGGML_NATIVE=OFF) ;;   # portable x86-64 build; reproducible across hosts
  cuda) command -v nvcc >/dev/null || skip "nvcc (CUDA toolkit) is not installed"
        flags+=(-DGGML_CUDA=ON)
        [[ -z "${PW_CUDA_ARCHS:-}" ]] || flags+=("-DCMAKE_CUDA_ARCHITECTURES=$PW_CUDA_ARCHS") ;;
  hip)  # separate from the cuda section: ROCm layouts differ, and Ubuntu's ROCm 5.7 needs a compatibility patch
        # shellcheck source=hip-config.sh
        . "$here/hip-config.sh"
        hip_detect || skip "$HIP_WHY"
        hip_prefix="$HIP_ROCM"; [[ -z "${PW_HIP_EXTRA_PREFIX:-}" ]] || hip_prefix="$HIP_ROCM;$PW_HIP_EXTRA_PREFIX"
        flags+=(-DGGML_HIP=ON "-DAMDGPU_TARGETS=${PW_HIP_ARCHS:-$( (( HIP_COMPAT )) && echo gfx90a || echo gfx942)}" "-DCMAKE_PREFIX_PATH=$hip_prefix"
                "-DCMAKE_C_COMPILER=$HIP_CLANG" "-DCMAKE_CXX_COMPILER=$HIP_CLANGXX"
                "-DCMAKE_HIP_COMPILER=$HIP_CLANGXX" "-DCMAKE_HIP_COMPILER_ROCM_ROOT=$HIP_ROCM"
                "-DCMAKE_HIP_COMPILER_ROCM_LIB=$HIP_CMAKE_LIB")
        if (( HIP_COMPAT )); then
          # ROCm before 6.1 (what apt gives on Ubuntu 24.04: 5.7.1): see patches/hip-rocm5.patch for the four
          # source changes, and below for the link flags. FlashAttention and HIP graphs are left off: they were
          # not built or run with 5.7, and FA is most of the compile time.
          flags+=(-DGGML_CUDA_FA=OFF -DGGML_HIP_GRAPHS=OFF)
          # ROCm 5.7's amd_hip_bf16.h defines its helpers without `inline`: one copy per object file, all identical
          link_extra="-Wl,--allow-multiple-definition"
        fi
        [[ "${PW_HIP_FA:-}" != 1 ]] || flags+=(-DGGML_CUDA_FA=ON)
        [[ "${PW_HIP_GRAPHS:-}" != 1 ]] || flags+=(-DGGML_HIP_GRAPHS=ON)
        if [[ -n "${PW_HIP_EXTRA_PREFIX:-}" ]]; then
          # the empty librocsolver stand-in of hip-apt-prefix.sh: the linker wants to see what libhipblas needs
          stub=$(cd "$PW_HIP_EXTRA_PREFIX/.." 2>/dev/null && pwd)/stub
          [[ -e "$stub/librocsolver.so.0" ]] && stub_flag="-Wl,-rpath-link,$stub"
        fi
        flags+=("-DCMAKE_SHARED_LINKER_FLAGS=${link_extra:-} ${stub_flag:-}" "-DCMAKE_EXE_LINKER_FLAGS=${link_extra:-} ${stub_flag:-}") ;;
  *) echo "unknown backend '$backend' (cpu, cuda, hip)" >&2; exit 2 ;;
esac
commit="${LLAMACPP_COMMIT:-$LLAMACPP_PINNED_COMMIT}"
repo="${LLAMACPP_REPO:-https://github.com/ggml-org/llama.cpp}"
if [[ -n "${PW_BUILD_DIR:-}" ]]; then work="$PW_BUILD_DIR"; mkdir -p "$work"; else work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT; fi
src="$work/src"
if [[ ! -e "$src/.pw-patched-for-$commit" ]]; then
  rm -rf "$src"
  git init -q "$src" && git -C "$src" fetch -q --depth 1 "$repo" "$commit" 2>"$work/git.err" \
    && git -C "$src" checkout -q FETCH_HEAD || { cat "$work/git.err" >&2; skip "cannot fetch llama.cpp $commit from $repo"; }
fi
got=$(git -C "$src" rev-parse HEAD)
if [[ "$backend" == hip ]] && (( HIP_COMPAT )) && [[ ! -e "$src/.pw-patched-for-$commit" ]]; then
  git -C "$src" apply "$here/patches/hip-rocm5.patch" 2>"$work/patch.err" \
    || { cat "$work/patch.err" >&2; echo "patches/hip-rocm5.patch does not apply to llama.cpp $got" >&2; exit 1; }
fi
[[ "$got" == "$commit" || ${#commit} -lt 40 ]] || { echo "fetched $got, wanted $commit" >&2; exit 1; }
touch "$src/.pw-patched-for-$commit"
echo "building llama.cpp $got ($backend) into $prefix" >&2
cmake -S "$src" -B "$work/build" "${flags[@]}" -DCMAKE_INSTALL_PREFIX="$prefix" >"$work/cmake.log" 2>&1 \
  || { tail -20 "$work/cmake.log" >&2; skip "cmake configure for $backend failed"; }
cmake --build "$work/build" -j"${PW_BUILD_JOBS:-2}" --target llama-completion llama-bench ${LLAMACPP_EXTRA_TARGETS:-} >"$work/build.log" 2>&1 \
  || { tail -20 "$work/build.log" >&2; echo "build of $backend failed" >&2; exit 1; }
# Install only the two tools and the shared libraries they load.
mkdir -p "$prefix/bin" "$prefix/lib"
cp "$work/build/bin/llama-completion" "$work/build/bin/llama-bench" "$prefix/bin/"
for t in ${LLAMACPP_EXTRA_TARGETS:-}; do cp "$work/build/bin/$t" "$prefix/bin/"; done
find "$work/build" -name '*.so*' \( -type f -o -type l \) -exec cp -P {} "$prefix/lib/" \;
if [[ "$backend" == hip && -n "${PW_HIP_EXTRA_PREFIX:-}" ]]; then
  # rocBLAS finds its kernels in <dir of librocblas.so>/rocblas/<version>/library
  cp -P "$PW_HIP_EXTRA_PREFIX"/lib/*-linux-gnu/lib{rocblas,hipblas}.so* "$prefix/lib/"
  cp "$PW_HIP_EXTRA_PREFIX/../stub/librocsolver.so.0" "$prefix/lib/" 2>/dev/null
  mkdir -p "$prefix/lib/rocblas"   # symlinks: the kernel libraries are hundreds of MB
  for d in "$PW_HIP_EXTRA_PREFIX"/lib/*-linux-gnu/rocblas/*; do ln -sfn "$d" "$prefix/lib/rocblas/$(basename "$d")"; done
fi
if [[ "${LLAMACPP_BUILD_PROBE:-0}" == 1 ]]; then
  # plain c++ against the public C API: no change to llama.cpp's own CMake files
  c++ -std=c++17 -O2 "$here/probe/pw-probe.cpp" -I"$src/include" -I"$src/ggml/include" -L"$work/build/bin" \
      -lllama -lggml -lggml-base ${stub_flag:-} -Wl,-rpath,'$ORIGIN/../lib' -o "$prefix/bin/pw-probe" \
    || { echo "compiling pw-probe failed" >&2; exit 1; }
fi
echo "$got" > "$prefix/COMMIT"
