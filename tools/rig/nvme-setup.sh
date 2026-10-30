#!/usr/bin/env bash
# Put the instance-store NVMe drives of a rig to use as a fast model cache (docs/rigs.md, "Direct download on a rig").
# Run on the rig with sudo (Deep Learning AMI, Ubuntu 22.04):  sudo tools/rig/nvme-setup.sh
#
#   * finds the NVMe disks whose model is "Amazon EC2 NVMe Instance Storage" (never the EBS root or other EBS volumes),
#     skipping any that is mounted or has a mounted partition;
#   * one disk: mkfs on it; several: an mdadm RAID0 (/dev/md/pwnvme) first;
#   * mkfs.xfs (ext4 when xfs is missing), label pwnvme, mounted at /mnt/nvme with noatime, owned by $PW_NVME_USER (default $SUDO_USER or ubuntu);
#   * prints the exports to use and writes them to /mnt/nvme/pw-env.sh (source it).
# Idempotent: already mounted -> prints the exports and changes nothing; a filesystem labelled pwnvme from an earlier run
# (instance store survives a reboot, not a stop) is mounted again rather than recreated. No instance store -> says so and exits 0
# (a g5/g6 or c6in rig has none: use gp3, which downloads at about 170 MiB/s). Instance-store data is lost when the instance stops
# or terminates; this is a cache, never the only copy of anything.
#
# Environment (for tests): PW_NVME_MNT (/mnt/nvme), PW_NVME_PROC_MOUNTS (/proc/mounts), PW_NVME_SKIP_ROOT_CHECK=1, PW_NVME_ID (id -u), PW_NVME_USER.
set -u
MNT="${PW_NVME_MNT:-/mnt/nvme}"
PROC_MOUNTS="${PW_NVME_PROC_MOUNTS:-/proc/mounts}"
OWNER="${PW_NVME_USER:-${SUDO_USER:-ubuntu}}"
LABEL=pwnvme

if [[ "${PW_NVME_SKIP_ROOT_CHECK:-0}" != 1 ]] && [[ "$(${PW_NVME_ID:-id -u})" != 0 ]]; then
  echo "nvme-setup.sh: must run as root (sudo $0)" >&2; exit 1
fi

exports() {
  cat <<EOT
export PW_CACHE=$MNT/pw-cache
export HF_HOME=$MNT/hf
export UV_CACHE_DIR=$MNT/uv-cache
export TMPDIR=$MNT/tmp
EOT
}

finish() {
  mkdir -p "$MNT/pw-cache" "$MNT/hf" "$MNT/uv-cache" "$MNT/tmp"
  chown "$OWNER" "$MNT" "$MNT/pw-cache" "$MNT/hf" "$MNT/uv-cache" "$MNT/tmp" 2>/dev/null || true
  exports > "$MNT/pw-env.sh" 2>/dev/null || true
  echo "Use it (llama.cpp builds and models live under PW_CACHE; vLLM/transformers weights under HF_HOME and PW_CACHE/hf):"
  exports
  echo "or: source $MNT/pw-env.sh"
  echo "Direct download: PW_FAST_FETCH=1 (see docs/rigs.md)."
  df -h "$MNT" 2>/dev/null | tail -1
}

is_mounted() { awk -v m="$1" '$2 == m {f=1} END {exit !f}' "$PROC_MOUNTS"; }
dev_busy() { grep -Eq "^$1(p?[0-9]+)? " "$PROC_MOUNTS"; }

if is_mounted "$MNT"; then
  echo "$MNT is already mounted; nothing to do."
  finish; exit 0
fi

devs=()
while read -r name type model; do
  [[ "$type" == disk ]] || continue
  model=$(printf '%b' "$model")
  [[ "$model" == *"Instance Storage"* ]] || continue
  if dev_busy "$name"; then echo "skipping $name: it or a partition is mounted" >&2; continue; fi
  devs+=("$name")
done < <(lsblk -d -n -p -r -o NAME,TYPE,MODEL 2>/dev/null)

if (( ${#devs[@]} == 0 )); then
  echo "no instance-store NVMe devices found (or all are in use): nothing to set up."
  echo "Fallback: keep PW_CACHE on the gp3 root volume. A single stream reaches ~145 MiB/s but gp3 writes cap near 170 MiB/s by default;"
  echo "raise the volume's throughput (up to 1000 MiB/s) and IOPS at launch if the download is the bottleneck."
  exit 0
fi

dev=$(blkid -L "$LABEL" 2>/dev/null) || dev=""
if [[ -z "$dev" && ${#devs[@]} -gt 1 ]]; then
  mdadm --assemble --scan >/dev/null 2>&1 || true
  dev=$(blkid -L "$LABEL" 2>/dev/null) || dev=""
fi
if [[ -z "$dev" ]]; then
  if (( ${#devs[@]} > 1 )); then
    echo "building RAID0 from ${#devs[@]} devices: ${devs[*]}"
    mdadm --create /dev/md/$LABEL --run --level=0 --raid-devices=${#devs[@]} --name=$LABEL "${devs[@]}" || { echo "mdadm failed" >&2; exit 1; }
    dev=/dev/md/$LABEL
  else
    dev="${devs[0]}"
  fi
  echo "making a filesystem on $dev"
  if command -v mkfs.xfs >/dev/null; then
    mkfs.xfs -f -L "$LABEL" "$dev" || { echo "mkfs.xfs failed" >&2; exit 1; }
  else
    mkfs.ext4 -F -L "$LABEL" -E lazy_itable_init=1,lazy_journal_init=1 "$dev" || { echo "mkfs.ext4 failed" >&2; exit 1; }
  fi
fi
mkdir -p "$MNT"
mount -o noatime,nodiscard "$dev" "$MNT" 2>/dev/null || mount -o noatime "$dev" "$MNT" || { echo "mount $dev $MNT failed" >&2; exit 1; }
chown "$OWNER" "$MNT" 2>/dev/null || true
echo "mounted $dev at $MNT (${#devs[@]} device(s))"
finish
