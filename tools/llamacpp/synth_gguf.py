#!/usr/bin/env python3
"""Deterministic generator of tiny RANDOM-WEIGHT GGUF models for llama.cpp.

    python3 synth_gguf.py <arch> -o model-f16.gguf [--seed N] [--size tiny|s|m|l|xl] [--vocab N]
    python3 synth_gguf.py --text-out prompt.txt [--seed N] [--words N]     the fixed perplexity text
    python3 synth_gguf.py --list

Architectures: llama, mistral, mixtral (MoE), qwen2, gemma, phi3.

THESE ARE NOT TRAINED MODELS. Every weight is a seeded pseudo-random number. The files exist to drive
llama.cpp's graph for each architecture and its quantised matmul kernels (and the CUDA/HIP versions of
them) with no model licence and no download. They have nothing to say about the quality or behaviour of
Llama, Mistral, Mixtral, Qwen2, Gemma or Phi-3 models: only the tensor names, shapes and metadata follow
those architectures. "mistral" and "mixtral" are llama.cpp's `llama` architecture with the Mistral /
Mixtral shapes (grouped-query attention, rope base 1e6; 8 experts, 2 used), as in llama.cpp itself.

Determinism: the same (arch, seed, size, vocab) gives a byte-identical file. The weights come from a
counter-based generator that uses only integer arithmetic, a final division by a power of two and an
IEEE float16 cast, so they do not depend on numpy's or the platform's random/libm implementation.
Needs the MIT-licensed `gguf` package (pip install gguf) and numpy.
"""
import argparse
import sys
import zlib

import numpy as np

import gguf

GENERATOR_VERSION = 1
DEFAULT_SEED = 20260501
DISCLAIMER = ("RANDOM-WEIGHT synthetic model for llama.cpp kernel and graph coverage; seeded pseudo-random "
              "values, not trained, not a copy or approximation of any named model.")

# size -> (n_embd, n_layer, n_ff, head_dim); parameter counts are printed by the generator.
SIZES = {
    "tiny": (256, 2, 512, 32),      # functional workloads: every row length is a multiple of 256 (k-quants)
    "s": (512, 6, 1536, 64),
    "m": (1024, 12, 2816, 64),
    "l": (2048, 16, 5632, 128),
    "xl": (4096, 32, 11008, 128),
}
DEFAULT_VOCAB = 1024
BENCH_VOCAB = 32000               # the larger sizes pad the vocabulary with unused tokens to this size

# arch -> family settings. kv_div: n_head_kv = n_head // kv_div (gemma: MQA = 1 kv head).
ARCHS = {
    "llama":   dict(gguf_arch="llama",  kv_div=1, rope_base=10000.0, eps=1e-5, tok="spm"),
    "mistral": dict(gguf_arch="llama",  kv_div=4, rope_base=1000000.0, eps=1e-5, tok="spm"),
    "mixtral": dict(gguf_arch="llama",  kv_div=4, rope_base=1000000.0, eps=1e-5, tok="spm",
                    n_expert=8, n_expert_used=2),
    "qwen2":   dict(gguf_arch="qwen2",  kv_div=4, rope_base=1000000.0, eps=1e-6, tok="bpe", qkv_bias=True),
    "gemma":   dict(gguf_arch="gemma",  kv_div=0, rope_base=10000.0, eps=1e-6, tok="spm", tied=True, block_gain=8.0),
    "phi3":    dict(gguf_arch="phi3",   kv_div=1, rope_base=10000.0, eps=1e-5, tok="spm", fused_qkv=True),
}

# ---------------------------------------------------------------- counter-based seeded weights

_M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _splitmix64(x):
    """splitmix64 finaliser on a uint64 array (wraps modulo 2**64)."""
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def stream_key(seed, name):
    return (np.uint64(seed) << np.uint64(32)) ^ np.uint64(zlib.crc32(name.encode()))


def normals(key, start, count):
    """`count` approximately N(0,1) float32 values for stream positions start..start+count-1.

    Each value is the sum of twelve 16-bit uniform integers (Irwin-Hall, variance exactly 65536**2),
    centred and divided by 65536: integer arithmetic until one exact division by a power of two."""
    idx = np.arange(start, start + count, dtype=np.uint64) * np.uint64(3)
    words = np.empty((count, 3), dtype="<u8")
    with np.errstate(over="ignore"):
        for j in range(3):
            words[:, j] = _splitmix64(key + (idx + np.uint64(j)) * np.uint64(0xD1B54A32D192ED03))
    u16 = words.reshape(count, 3).view("<u2").reshape(count, 12)
    s = u16.sum(axis=1, dtype=np.int64) - 6 * 65535
    return s.astype(np.float32) / np.float32(65536.0)


ROW_SCALES = np.array([0.35, 0.5, 0.7, 1.0, 1.4, 2.8, 4.0], dtype=np.float32)


def row_scales(seed, name, rows):
    """Per-row multipliers from a fixed table (no exp/log, so identical on every platform): a few rows
    are louder than the rest, which widens the gap between the largest logits and keeps greedy decoding
    stable under quantisation noise."""
    z = normals(stream_key(seed, name + "#rows"), 0, rows)
    return ROW_SCALES[np.clip(np.floor(z + np.float32(3.5)).astype(np.int64), 0, 6)]


def weights(seed, name, shape, std, dtype=np.float16, chunk=1 << 22):
    """Seeded weights of `shape` with standard deviation `std`, cast to `dtype`."""
    n = int(np.prod(shape))
    key = stream_key(seed, name)
    out = np.empty(n, dtype=dtype)
    std = np.float32(std)
    for s in range(0, n, chunk):
        c = min(chunk, n - s)
        out[s:s + c] = (normals(key, s, c) * std).astype(dtype)
    return out.reshape(shape)


# ---------------------------------------------------------------- tokenizers (generated, no third-party vocab)

WORDS = """the of and to in is that it was for on are as with his they at be this from have or by one had not but what
all were when we there can an your which their said if do will each about how up out them then she many some so
these would other into has more her two like him see time could no make than first been its who now people my made
over did down only way find use may water long little very after words called just where most know get through back
much before go good new write our me man too any day same right look think also around another came come work three
must because does part even place well such here take why help put different away again off went old number great
tell men say small every found still between name should home big give air line set own under read last never us
left end along while might next sound below saw something thought both few those always show large often together
asked house world going want school important until form food keep children feet land side without boy once animal
life enough took sometimes four head above kind began almost live page got earth need far hand high year mother
light country father let night picture being study second soon story since white ever paper hard near sentence
better best across during today however sure knew try told young sun thing whole hear example heard several change
answer room sea against top turned learn point city play toward five using himself usually""".split()

SPECIALS_SPM = ["<unk>", "<s>", "</s>", "<|endoftext|>"]   # llama.cpp looks the last one up for phi3 models
SPACE = "▁"           # SentencePiece's visible space


def _filler(i):
    return f"<pad_{i}>"


def spm_vocab(n_vocab):
    """Return (tokens, scores, types) for a SentencePiece-style (llama) vocabulary of exactly n_vocab tokens."""
    tokens = list(SPECIALS_SPM) + [f"<0x{b:02X}>" for b in range(256)]
    types = [gguf.TokenType.UNKNOWN] + [gguf.TokenType.CONTROL] * 3 + [gguf.TokenType.BYTE] * 256
    scores = [0.0] * len(tokens)
    pieces = [SPACE]
    pieces += [chr(c) for c in range(33, 127) if chr(c) not in "<"]              # printable ASCII except '<'
    pieces += [SPACE + chr(c) for c in range(ord("a"), ord("z") + 1)]
    for w in WORDS:           # word pieces, more frequent words first; every prefix is a piece so SPM can merge up
        pieces += [SPACE + w[:k] for k in range(2, len(w) + 1)] + [w[:k] for k in range(2, len(w) + 1)]
    seen, rank = set(tokens), 0
    for p in pieces:
        if p in seen or len(tokens) >= n_vocab:
            continue
        seen.add(p)
        tokens.append(p)
        types.append(gguf.TokenType.NORMAL)
        # single characters merge last; longer word pieces and earlier words win
        scores.append(-1000.0 + 0.5 * min(len(p), 20) - rank * 0.01 if len(p) > 1 else -1000.0 - rank * 0.01)
        rank += 1
    while len(tokens) < n_vocab:
        tokens.append(_filler(len(tokens)))
        types.append(gguf.TokenType.USER_DEFINED)
        scores.append(-10000.0)
    return tokens[:n_vocab], scores[:n_vocab], types[:n_vocab]


def _byte_unicode():
    """GPT-2's reversible byte -> printable unicode map."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def bpe_vocab(n_vocab):
    """Return (tokens, types, merges) for a byte-level BPE vocabulary trained on WORDS (deterministic)."""
    b2u = _byte_unicode()
    base = [b2u[b] for b in range(256)]
    corpus = {}
    for i, w in enumerate(WORDS):
        for form, weight in (("Ġ" + w, 100 - i // 3), (w, 40 - i // 6)):     # Ġ = byte-level space
            corpus[tuple(form)] = max(weight, 1)
    tokens, merges = list(base), []
    n_special = 3
    words = {k: v for k, v in corpus.items()}
    while len(tokens) < n_vocab - n_special:
        pairs = {}
        for w, c in words.items():
            for a, b in zip(w, w[1:]):
                pairs[(a, b)] = pairs.get((a, b), 0) + c
        if not pairs:
            break
        a, b = max(pairs, key=lambda p: (pairs[p], p))      # most frequent; ties: largest pair
        new = a + b
        if new in tokens:
            break
        tokens.append(new)
        merges.append(f"{a} {b}")
        out = {}
        for w, c in words.items():
            lst, i = [], 0
            while i < len(w):
                if i + 1 < len(w) and w[i] == a and w[i + 1] == b:
                    lst.append(new)
                    i += 2
                else:
                    lst.append(w[i])
                    i += 1
            out[tuple(lst)] = out.get(tuple(lst), 0) + c
        words = out
    types = [gguf.TokenType.NORMAL] * len(tokens)
    for sp in ("<|endoftext|>", "<|im_start|>", "<|im_end|>"):
        tokens.append(sp)
        types.append(gguf.TokenType.CONTROL)
    while len(tokens) < n_vocab:
        tokens.append(_filler(len(tokens)))
        types.append(gguf.TokenType.USER_DEFINED)
    return tokens[:n_vocab], types[:n_vocab], merges


def synth_text(seed=DEFAULT_SEED, n_words=1200):
    """A fixed pseudo-random word sequence (for llama-perplexity): same seed, same text."""
    draw = _splitmix64(stream_key(seed, "text") + np.arange(n_words, dtype=np.uint64))
    sentence, out = 0, []
    for k in range(n_words):
        w = WORDS[int(draw[k] % np.uint64(len(WORDS)))]
        out.append(w)
        sentence += 1
        if sentence >= 9 + (k % 5):
            out[-1] += "."
            sentence = 0
    return " ".join(out) + "\n"


# ---------------------------------------------------------------- the models

def spec(arch, size="tiny", vocab=None):
    if arch not in ARCHS:
        raise SystemExit(f"unknown architecture {arch!r}; one of {', '.join(ARCHS)}")
    if size not in SIZES:
        raise SystemExit(f"unknown size {size!r}; one of {', '.join(SIZES)}")
    a = dict(ARCHS[arch])
    if vocab is None:                 # a tiny vocab for functional runs, a realistic output-layer size for benchmarks
        vocab = DEFAULT_VOCAB if size == "tiny" else BENCH_VOCAB
    embd, layers, ff, hd = SIZES[size]
    heads = embd // hd
    kv = 1 if a["kv_div"] == 0 else max(heads // a["kv_div"], 1)
    a.update(arch=arch, size=size, n_embd=embd, n_layer=layers, n_head=heads, n_head_kv=kv, head_dim=hd,
             n_ff=ff, n_vocab=vocab, n_ctx=2048 if size != "tiny" else 512)
    if a.get("n_expert"):
        a["n_ff"] = ff // 2 if size != "tiny" else 256       # per-expert width
    return a


def tensor_plan(a):
    """[(name, shape numpy order (out, in), std or None for a norm weight, dtype)] for the architecture."""
    T, N = gguf.MODEL_TENSOR, gguf.TENSOR_NAMES
    nm = lambda t, i=None: N[t].format(bid=i) if i is not None else N[t]
    e, ff, v = a["n_embd"], a["n_ff"], a["n_vocab"]
    q, kvw = a["n_head"] * a["head_dim"], a["n_head_kv"] * a["head_dim"]
    bg = a.get("block_gain", 1.0)      # louder attention/FFN outputs, so a tied embedding does not dominate the logits
    lin = lambda fan_in, gain=1.0: gain / np.sqrt(fan_in)
    plan = [(nm(T.TOKEN_EMBD) + ".weight", (v, e), 1.0 if a["arch"] != "gemma" else 0.06, np.float16)]
    plan.append((nm(T.OUTPUT_NORM) + ".weight", (e,), None, np.float32))
    if not a.get("tied"):
        plan.append((nm(T.OUTPUT) + ".weight", (v, e), 1.5 * lin(e), np.float16))
    for i in range(a["n_layer"]):
        plan.append((nm(T.ATTN_NORM, i) + ".weight", (e,), None, np.float32))
        if a.get("fused_qkv"):
            plan.append((nm(T.ATTN_QKV, i) + ".weight", (q + 2 * kvw, e), lin(e), np.float16))
        else:
            plan += [(nm(T.ATTN_Q, i) + ".weight", (q, e), lin(e), np.float16),
                     (nm(T.ATTN_K, i) + ".weight", (kvw, e), lin(e), np.float16),
                     (nm(T.ATTN_V, i) + ".weight", (kvw, e), lin(e), np.float16)]
        if a.get("qkv_bias"):
            plan += [(nm(T.ATTN_Q, i) + ".bias", (q,), 0.05, np.float32),
                     (nm(T.ATTN_K, i) + ".bias", (kvw,), 0.05, np.float32),
                     (nm(T.ATTN_V, i) + ".bias", (kvw,), 0.05, np.float32)]
        plan.append((nm(T.ATTN_OUT, i) + ".weight", (e, q), lin(q, bg), np.float16))
        plan.append((nm(T.FFN_NORM, i) + ".weight", (e,), None, np.float32))
        if a.get("n_expert"):
            n = a["n_expert"]
            plan += [(nm(T.FFN_GATE_INP, i) + ".weight", (n, e), lin(e, 8.0), np.float32),
                     (nm(T.FFN_GATE_EXP, i) + ".weight", (n, ff, e), lin(e), np.float16),
                     (nm(T.FFN_DOWN_EXP, i) + ".weight", (n, e, ff), lin(ff), np.float16),
                     (nm(T.FFN_UP_EXP, i) + ".weight", (n, ff, e), lin(e), np.float16)]
        elif a["arch"] == "phi3":      # gate and up fused into ffn_up (2*n_ff rows), as in the real architecture
            plan += [(nm(T.FFN_DOWN, i) + ".weight", (e, ff), lin(ff, bg), np.float16),
                     (nm(T.FFN_UP, i) + ".weight", (2 * ff, e), lin(e), np.float16)]
        else:
            plan += [(nm(T.FFN_GATE, i) + ".weight", (ff, e), lin(e), np.float16),
                     (nm(T.FFN_DOWN, i) + ".weight", (e, ff), lin(ff, bg), np.float16),
                     (nm(T.FFN_UP, i) + ".weight", (ff, e), lin(e), np.float16)]
    return plan


def n_params(a):
    return sum(int(np.prod(s)) for _, s, _, _ in tensor_plan(a))


def write_model(path, arch, seed=DEFAULT_SEED, size="tiny", vocab=None):
    a = spec(arch, size, vocab)
    w = gguf.GGUFWriter(path, a["gguf_arch"])
    w.add_name(f"pw-synth-{arch}-{size}-seed{seed}")
    w.add_description(DISCLAIMER)
    w.add_license("apache-2.0")
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
    w.add_string("pantheonworkloads.synth.disclaimer", DISCLAIMER)
    w.add_uint32("pantheonworkloads.synth.generator_version", GENERATOR_VERSION)
    w.add_uint64("pantheonworkloads.synth.seed", seed)
    w.add_string("pantheonworkloads.synth.family", arch)
    w.add_context_length(a["n_ctx"])
    w.add_embedding_length(a["n_embd"])
    w.add_block_count(a["n_layer"])
    w.add_feed_forward_length(a["n_ff"])
    w.add_head_count(a["n_head"])
    w.add_head_count_kv(a["n_head_kv"])
    w.add_layer_norm_rms_eps(a["eps"])
    w.add_rope_dimension_count(a["head_dim"])
    w.add_rope_freq_base(a["rope_base"])
    w.add_vocab_size(a["n_vocab"])
    if a["gguf_arch"] == "gemma":
        w.add_key_length(a["head_dim"])
        w.add_value_length(a["head_dim"])
    if a.get("n_expert"):
        w.add_expert_count(a["n_expert"])
        w.add_expert_used_count(a["n_expert_used"])
    if a["tok"] == "spm":
        tokens, scores, types = spm_vocab(a["n_vocab"])
        w.add_tokenizer_model("llama")
        w.add_token_list(tokens)
        w.add_token_scores(scores)
        w.add_token_types(types)
        w.add_bos_token_id(1)
        w.add_eos_token_id(2)
        w.add_unk_token_id(0)
        w.add_add_bos_token(True)
        w.add_add_eos_token(False)
        w.add_add_space_prefix(True)
    else:
        tokens, types, merges = bpe_vocab(a["n_vocab"])
        w.add_tokenizer_model("gpt2")
        w.add_tokenizer_pre("qwen2")
        w.add_token_list(tokens)
        w.add_token_types(types)
        w.add_token_merges(merges)
        eos = tokens.index("<|endoftext|>")
        w.add_eos_token_id(eos)
        w.add_pad_token_id(eos)
        w.add_add_bos_token(False)
    loud_rows = {"output.weight"}
    for name, shape, std, dtype in tensor_plan(a):
        if std is None:                      # norm scales: 1 + 0.05 * N(0,1)
            data = (np.float32(1.0) + weights(seed, name, shape, 0.05, np.float32)).astype(np.float32)
        else:
            data = weights(seed, name, shape, std, dtype)
            if name in loud_rows:
                data = (data.astype(np.float32) * row_scales(seed, name, shape[0])[:, None]).astype(dtype)
        w.add_tensor(name, data)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=False)
    w.close()
    return a


def write_imatrix(path, arch, seed=DEFAULT_SEED, size="tiny", vocab=None):
    """A seeded synthetic importance matrix (llama.cpp's GGUF imatrix layout) for the model's weight
    matrices. llama-quantize needs one for the IQ1/IQ2/IQ3_XXS/IQ3_XS and Q2_K_S types; with random weights
    there is no data to measure, so the per-input-channel statistics are seeded noise too."""
    a = spec(arch, size, vocab)
    w = gguf.GGUFWriter(path, "imatrix")
    w.add_array("imatrix.datasets", ["pantheonworkloads synthetic (seeded noise, not measured)"])
    w.add_uint32("imatrix.chunk_count", 1)
    w.add_uint32("imatrix.chunk_size", 512)
    for name, shape, std, _ in tensor_plan(a):
        if std is None or len(shape) < 2 or not name.endswith(".weight"):
            continue
        n_exp = shape[0] if len(shape) == 3 else 1
        n_in = shape[-1]
        z = weights(seed, name + "#imatrix", (n_exp, n_in), 1.0, np.float32)
        w.add_tensor(name + ".in_sum2", (z * z + np.float32(0.05)) * np.float32(1000.0))
        w.add_tensor(name + ".counts", np.full((n_exp,), 1000.0, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=False)
    w.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("arch", nargs="?", help="/".join(ARCHS))
    ap.add_argument("-o", "--output")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--size", default="tiny", help="/".join(SIZES))
    ap.add_argument("--vocab", type=int, default=None, help="default 1024 (tiny) or 32000 (larger sizes)")
    ap.add_argument("--imatrix-out", help="write the synthetic importance matrix for the model here instead of a model")
    ap.add_argument("--text-out", help="write the fixed perplexity text here instead of a model")
    ap.add_argument("--words", type=int, default=1200)
    ap.add_argument("--list", action="store_true", help="print the sizes and parameter counts")
    a = ap.parse_args(argv)
    if a.list:
        for arch in ARCHS:
            for size in SIZES:
                print(f"{arch:8} {size:5} {n_params(spec(arch, size, a.vocab)) / 1e6:9.1f} M params")
        return 0
    if a.imatrix_out:
        write_imatrix(a.imatrix_out, a.arch, a.seed, a.size, a.vocab)
        return 0
    if a.text_out:
        with open(a.text_out, "w", encoding="utf-8", newline="\n") as f:
            f.write(synth_text(a.seed, a.words))
        return 0
    if not a.arch or not a.output:
        ap.error("an architecture and -o are required")
    info = write_model(a.output, a.arch, a.seed, a.size, a.vocab)
    print(f"wrote {a.output}: {info['arch']} {info['size']} "
          f"{n_params(info) / 1e6:.2f}M random parameters (seed {a.seed})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
