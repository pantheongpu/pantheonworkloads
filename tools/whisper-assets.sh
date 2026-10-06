#!/usr/bin/env bash
# Fetches, checksum-verified, the two inputs the whisper.cpp workloads need and prints the
# local path on the last stdout line. Nothing downloaded is ever committed.
#
#   tools/whisper-assets.sh model   ggml-tiny.en.bin (OpenAI Whisper tiny.en, MIT, converted by whisper.cpp)
#   tools/whisper-assets.sh clip    jfk.wav (11 s, 16 kHz mono)
#
#   PW_CACHE            cache directory (default ~/.cache/pantheonworkloads)
#   PW_WHISPER_MODEL    use this local file instead of downloading the model (still checksum-verified)
# Exit 77 when the file cannot be had here (offline, blocked host) or fails its checksum.
set -uo pipefail
cache="${PW_CACHE:-$HOME/.cache/pantheonworkloads}/whisper-assets"
skip() { echo "SKIP: $*"; exit 77; }
# Model: the sha1 is the one whisper.cpp publishes in models/README.md at tag v1.9.5.
MODEL_URL=https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin
MODEL_SHA1=c78c86eb1a8faa21b369bcd33207cc90d64ae9df
# Clip: pinned to the whisper.cpp commit of tag v1.9.5; sha256 computed when this was written.
CLIP_URL=https://raw.githubusercontent.com/ggml-org/whisper.cpp/d1be6fde11ac6e0407606b4e42fe72d34add8037/samples/jfk.wav
CLIP_SHA256=59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e

fetch() {  # fetch URL DEST ALGO SUM
  local url=$1 dest=$2 algo=$3 sum=$4
  if [[ ! -f "$dest" ]]; then
    mkdir -p "$cache" && curl -fsSL --retry 2 -m 600 -o "$dest.part" "$url" || { rm -f "$dest.part"; skip "cannot download $url"; }
    mv "$dest.part" "$dest"
  fi
  [[ "$("${algo}sum" "$dest" | cut -d' ' -f1)" == "$sum" ]] || skip "$dest does not match its pinned ${algo} ($sum); delete it to re-download"
  echo "$dest"
}

case "${1:-}" in
  model)
    if [[ -n "${PW_WHISPER_MODEL:-}" ]]; then
      [[ "$(sha1sum "$PW_WHISPER_MODEL" 2>/dev/null | cut -d' ' -f1)" == "$MODEL_SHA1" ]] || skip "PW_WHISPER_MODEL is missing or not ggml-tiny.en.bin (sha1 $MODEL_SHA1)"
      echo "$PW_WHISPER_MODEL"
    else
      fetch "$MODEL_URL" "$cache/ggml-tiny.en.bin" sha1 "$MODEL_SHA1"
    fi ;;
  clip) fetch "$CLIP_URL" "$cache/jfk-$CLIP_SHA256.wav" sha256 "$CLIP_SHA256" ;;
  *) echo "usage: $0 model|clip" >&2; exit 2 ;;
esac
