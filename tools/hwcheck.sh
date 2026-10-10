# Sourced by the model-catalog workloads' run.sh (docs/model-registry.md). Provides:
#   pw_require_gpus <count> <gib each> [nvidia|amd]   exit 77 with a message unless the host has that many GPUs of that memory
# The same numbers are in the workload's manifest.yaml under `requires:`; bin/pw checks them before it starts the
# workload, this repeats the check for a run.sh started by hand. PW_IGNORE_REQUIRES=1 turns it off (as in bin/pw).
# Detection: tools/hwcheck.py (nvidia-smi, then rocm-smi / sysfs for AMD).

#   pw_require_disk <GiB>                              exit 77 unless $PW_CACHE (default ~/.cache/pantheonworkloads) has that much room
pw_require_disk() {
  local need=$1 dir="${PW_CACHE:-$HOME/.cache/pantheonworkloads}" free
  mkdir -p "$dir" 2>/dev/null || true
  free=$(df -P --block-size=1G "$dir" 2>/dev/null | awk 'NR==2 {print $4}')
  if [[ -n "$free" ]] && (( free < need )); then
    echo "SKIP: the weights need about $need GiB under $dir; $free GiB are free (set PW_CACHE to a bigger disk)"
    exit 77
  fi
}

pw_require_gpus() {
  local want=$1 gib=$2 vendor=${3:-} root line have rest
  [[ "${PW_IGNORE_REQUIRES:-0}" != 1 ]] || return 0
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  command -v python3 >/dev/null || { echo "SKIP: python3 is not installed"; exit 77; }
  line=$(python3 -I "$root/hwcheck.py" "$gib" $vendor) || { echo "SKIP: cannot inspect the GPUs of this host"; exit 77; }
  have=${line%% *}; rest=${line#* }
  if (( have < want )); then
    echo "SKIP: needs $want GPU(s) with >= $gib GiB each${vendor:+ ($vendor)}; this host has $rest"
    exit 77
  fi
}
