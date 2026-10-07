#!/usr/bin/env bash
# Prepare the two libraries llama.cpp's HIP backend links (rocBLAS, hipBLAS) from Ubuntu's apt packages,
# WITHOUT installing them: a full `apt install libhipblas-dev` pulls librocblas0 (2.3 GB installed, kernels
# for 10 architectures) and librocsolver0 (1.5 GB) onto a small disk. Only the pieces a build for the
# architectures asked for needs are unpacked into <prefix>.
#
#   tools/llamacpp/hip-apt-prefix.sh <prefix> [gfx90a gfx1030 gfx1100 ...]     (default: gfx90a)
#   PW_HIP_EXTRA_PREFIX=<prefix>/usr tools/llamacpp/build.sh hip <install-prefix>
#
# What goes into <prefix>:
#   usr/include, usr/lib/x86_64-linux-gnu/{librocblas,libhipblas}.so*, cmake config files   (headers and libraries)
#   usr/lib/x86_64-linux-gnu/rocblas/<ver>/library/   the Tensile kernel libraries of the architectures asked for
#                                                     (Ubuntu's rocBLAS 5.5.1 has none for gfx942/gfx950/gfx12xx:
#                                                     gfx803 gfx900 gfx906 gfx908 gfx90a gfx1010 gfx1030 gfx1100-1102)
#   stub/librocsolver.so.0   an EMPTY stand-in: libhipblas.so is linked with BIND_NOW against 68 rocsolver_* names,
#                            so the dynamic loader needs a library that defines them. Each returns
#                            rocblas_status_internal_error (2). llama.cpp never calls hipblas's solver entry
#                            points; if something ever does it fails loudly, it never computes.
# Exit 77 when apt cannot download the packages. Needs apt-get and dpkg-deb.
set -uo pipefail
prefix="${1:-}"; shift || true
[[ -n "$prefix" ]] || { echo "usage: hip-apt-prefix.sh <prefix> [gfx...]" >&2; exit 2; }
archs=("$@"); [[ ${#archs[@]} -gt 0 ]] || archs=(gfx90a)
for t in apt-get dpkg-deb gcc tar; do command -v "$t" >/dev/null || { echo "SKIP: $t is not installed"; exit 77; }; done
prefix=$(mkdir -p "$prefix" && cd "$prefix" && pwd)
work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
( cd "$work" && apt-get download librocblas0 librocblas-dev libhipblas0 libhipblas-dev >/dev/null 2>"$work/apt.err" ) \
  || { cat "$work/apt.err" >&2; echo "SKIP: apt-get download failed (is a deb source configured and reachable?)"; exit 77; }
for d in librocblas-dev libhipblas0 libhipblas-dev; do dpkg-deb -x "$work"/${d}_*.deb "$prefix"; done
# the runtime library, without the multi-GB kernel set: the shared object plus the asked-for architectures
rb=$(ls "$work"/librocblas0_*.deb)
pat=(--wildcards './usr/lib/x86_64-linux-gnu/librocblas.so*' './usr/lib/x86_64-linux-gnu/rocblas/*/library/TensileLibrary.dat')
for a in "${archs[@]}"; do
  pat+=("./usr/lib/x86_64-linux-gnu/rocblas/*/library/*${a}*")
done
dpkg-deb --fsys-tarfile "$rb" | tar -x -C "$prefix" "${pat[@]}" || { echo "unpacking rocBLAS failed" >&2; exit 1; }
mkdir -p "$prefix/stub"
nm -D --undefined-only "$prefix"/usr/lib/x86_64-linux-gnu/libhipblas.so.0 | awk '/ U rocsolver_/{print "int " $2 "(void){return 2;}"}' >"$work/stub.c"
[[ -s "$work/stub.c" ]] || { echo "libhipblas.so.0 needs no rocsolver names (changed package?): no stub made" >&2; }
gcc -shared -fPIC -Wl,-soname,librocsolver.so.0 -o "$prefix/stub/librocsolver.so.0" "$work/stub.c" || exit 1
echo "$prefix/usr"
