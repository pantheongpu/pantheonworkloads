#!/usr/bin/env bash
# Fetches, checksum-verified, the two inputs the whisper.cpp workloads need and prints the
# local path on the last stdout line. Nothing downloaded is ever committed.
#
#   tools/whisper-assets.sh model   ggml-tiny.en.bin (OpenAI Whisper tiny.en, MIT, converted by whisper.cpp)
#   tools/whisper-assets.sh model-large-v3   ggml-large-v3.bin (OpenAI Whisper large-v3, 3.1 GB, converted by whisper.cpp)
#   tools/whisper-assets.sh clip    jfk.wav (11 s, 16 kHz mono)
#
#   PW_CACHE            cache directory (default ~/.cache/pantheonworkloads)
#   PW_WHISPER_MODEL    (model only) use this local file instead of downloading the model (still checksum-verified)
# Exit 77 when the file cannot be had here (offline, blocked host) or fails its checksum.
set -uo pipefail
cache="${PW_CACHE:-$HOME/.cache/pantheonworkloads}/whisper-assets"
skip() { echo "SKIP: $*"; exit 77; }
# Model: pinned to the Hugging Face commit of ggerganov/whisper.cpp read on 2026-10-08; the sha256 is the file's LFS oid there.
# Its sha1, c78c86eb1a8faa21b369bcd33207cc90d64ae9df, is the one whisper.cpp publishes in models/README.md at v1.9.5, and matches.
MODEL_URL=https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-tiny.en.bin
MODEL_SHA256=921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f
# large-v3: the same HF commit; sha256 = the file's LFS oid there (3095033483 bytes).
LARGE_URL=https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-large-v3.bin
LARGE_SHA256=64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2
# large-v3-turbo (added with the model catalog, docs/model-registry.md; never downloaded when this was written): the same HF commit;
# sha256 = the file's LFS oid there, read from the Hub API on 2026-10-09 (see workloads/whisper-cpp-large-v3-turbo/model.sha256).
TURBO_URL=https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-large-v3-turbo.bin
TURBO_SHA256=1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69
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
      [[ "$(sha256sum "$PW_WHISPER_MODEL" 2>/dev/null | cut -d' ' -f1)" == "$MODEL_SHA256" ]] || skip "PW_WHISPER_MODEL is missing or not ggml-tiny.en.bin (sha256 $MODEL_SHA256)"
      echo "$PW_WHISPER_MODEL"
    else
      fetch "$MODEL_URL" "$cache/ggml-tiny.en.bin" sha256 "$MODEL_SHA256"
    fi ;;
  model-large-v3)
    if [[ -n "${PW_WHISPER_MODEL:-}" ]]; then
      [[ "$(sha256sum "$PW_WHISPER_MODEL" 2>/dev/null | cut -d' ' -f1)" == "$LARGE_SHA256" ]] || skip "PW_WHISPER_MODEL is missing or not ggml-large-v3.bin (sha256 $LARGE_SHA256)"
      echo "$PW_WHISPER_MODEL"
    else
      fetch "$LARGE_URL" "$cache/ggml-large-v3.bin" sha256 "$LARGE_SHA256"
    fi ;;
  model-large-v3-turbo)
    if [[ -n "${PW_WHISPER_MODEL:-}" ]]; then
      [[ "$(sha256sum "$PW_WHISPER_MODEL" 2>/dev/null | cut -d' ' -f1)" == "$TURBO_SHA256" ]] || skip "PW_WHISPER_MODEL is missing or not ggml-large-v3-turbo.bin (sha256 $TURBO_SHA256)"
      echo "$PW_WHISPER_MODEL"
    else
      fetch "$TURBO_URL" "$cache/ggml-large-v3-turbo.bin" sha256 "$TURBO_SHA256"
    fi ;;
  clip) fetch "$CLIP_URL" "$cache/jfk-$CLIP_SHA256.wav" sha256 "$CLIP_SHA256" ;;
  *) echo "usage: $0 model|model-large-v3|model-large-v3-turbo|clip" >&2; exit 2 ;;
esac
