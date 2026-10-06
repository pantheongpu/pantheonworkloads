# Sourced by build.sh for the hip backend. hip_detect finds the ROCm tree hipcc belongs to and the clang it uses,
# for both layouts: AMD's (/opt/rocm-X.Y/{bin,lib/llvm/bin}) and Ubuntu/Debian's (/usr/bin/hipcc with clang-17
# beside it and the CMake packages under /usr/lib/x86_64-linux-gnu/cmake). Sets:
#   HIP_ROCM      root of the install (for CMAKE_PREFIX_PATH)
#   HIP_CLANG, HIP_CLANGXX   the compilers hipcc itself uses
#   HIP_CMAKE_LIB the directory holding cmake/hip-lang/hip-lang-config.cmake
#   HIP_MAJOR, HIP_MINOR     from `hipcc --version` ("HIP version: 5.7.31921-0")
#   HIP_COMPAT    1 when llama.cpp's HIP backend needs tools/llamacpp/patches/hip-rocm5.patch: HIP before 6.1
# Returns 1 with a message in HIP_WHY when something is missing. HIPCC overrides the hipcc looked up on PATH.
hip_detect() {
  local hipcc="${HIPCC:-$(command -v hipcc || true)}" ver inst cver cand d n
  [[ -n "$hipcc" ]] || { HIP_WHY="hipcc (ROCm) is not installed"; return 1; }
  hipcc=$(readlink -f "$hipcc")
  HIP_ROCM=$(dirname "$(dirname "$hipcc")")
  ver=$("$hipcc" --version 2>/dev/null) || { HIP_WHY="$hipcc --version failed"; return 1; }
  HIP_MAJOR=$(sed -n 's/^HIP version: \([0-9]*\)\..*/\1/p' <<<"$ver" | head -1)
  HIP_MINOR=$(sed -n 's/^HIP version: [0-9]*\.\([0-9]*\).*/\1/p' <<<"$ver" | head -1)
  [[ -n "$HIP_MAJOR" && -n "$HIP_MINOR" ]] || { HIP_WHY="cannot read the HIP version from '$hipcc --version'"; return 1; }
  inst=$(sed -n 's/^InstalledDir: //p' <<<"$ver" | head -1)
  cver=$(sed -n 's/.*clang version \([0-9]*\)\..*/\1/p' <<<"$ver" | head -1)
  HIP_CLANG="" ; HIP_CLANGXX=""
  for cand in "$HIP_ROCM/lib/llvm/bin" "$HIP_ROCM/llvm/bin" "$inst" "$HIP_ROCM/lib/llvm-$cver/bin"; do
    [[ -n "$cand" ]] || continue
    # Debian keeps several LLVMs side by side: /usr/bin/clang++ may be another version than hipcc's
    for n in "clang++-$cver" clang++; do
      if [[ -x "$cand/$n" ]] && { [[ "$n" == "clang++-$cver" ]] || "$cand/$n" --version 2>/dev/null | grep -q "clang version $cver\."; }; then
        d=$(dirname "$(readlink -f "$cand/$n")"); HIP_CLANGXX="$d/clang++"; HIP_CLANG="$d/clang"; break 2
      fi
    done
  done
  [[ -x "$HIP_CLANGXX" && -x "$HIP_CLANG" ]] || { HIP_WHY="no clang $cver beside hipcc under $HIP_ROCM"; return 1; }
  HIP_CMAKE_LIB=""
  for cand in "$HIP_ROCM/lib" "$HIP_ROCM/lib64" "$HIP_ROCM"/lib/*-linux-gnu; do
    [[ -e "$cand/cmake/hip-lang/hip-lang-config.cmake" ]] && { HIP_CMAKE_LIB="$cand"; break; }
  done
  [[ -n "$HIP_CMAKE_LIB" ]] || { HIP_WHY="no cmake/hip-lang/hip-lang-config.cmake under $HIP_ROCM (libamdhip64-dev?)"; return 1; }
  HIP_COMPAT=0
  (( HIP_MAJOR < 6 || (HIP_MAJOR == 6 && HIP_MINOR < 1) )) && HIP_COMPAT=1
  return 0
}
