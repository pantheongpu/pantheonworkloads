"""A BERT-style WordPiece tokenizer in pure Python (standard library only).

Used by the bertsquad and minilm workloads so that no tokenizer package is needed. It follows the
algorithm of the original BERT release (google-research/bert, tokenization.py: BasicTokenizer
with lower-casing, accent stripping, CJK splitting and punctuation splitting, then greedy
longest-match-first WordPiece). Written for this repository, not copied. The vocabulary comes from a
Hugging Face `tokenizer.json` (model.vocab) of an uncased BERT-vocabulary model.
"""
import json
import unicodedata

UNK, CLS, SEP, PAD = "[UNK]", "[CLS]", "[SEP]", "[PAD]"


def _is_whitespace(ch):
    return ch in " \t\n\r" or unicodedata.category(ch) == "Zs"


def _is_control(ch):
    return ch not in "\t\n\r" and unicodedata.category(ch).startswith("C")


def _is_punct(ch):
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:   # ASCII symbols count as punctuation
        return True
    return unicodedata.category(ch).startswith("P")


def _is_cjk(cp):
    return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF or 0x2A700 <= cp <= 0x2B73F
            or 0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF or 0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F)


def basic_tokens(text, lower=True):
    """Whitespace, punctuation and CJK-character split (after cleaning, lower-casing, accent stripping)."""
    out = []
    for ch in text:
        cp = ord(ch)
        if cp == 0 or cp == 0xFFFD or _is_control(ch):
            continue
        out.append(" " if _is_whitespace(ch) else (f" {ch} " if _is_cjk(cp) else ch))
    text = "".join(out)
    words = []
    for word in text.split():
        if lower:
            word = unicodedata.normalize("NFD", word.lower())
            word = "".join(c for c in word if unicodedata.category(c) != "Mn")
        cur = ""
        for ch in word:
            if _is_punct(ch):
                if cur:
                    words.append(cur)
                    cur = ""
                words.append(ch)
            else:
                cur += ch
        if cur:
            words.append(cur)
    return words


class WordPiece:
    def __init__(self, vocab, lower=True, max_chars=100):
        self.vocab, self.lower, self.max_chars = vocab, lower, max_chars
        self.inv = {i: t for t, i in vocab.items()}

    @classmethod
    def from_tokenizer_json(cls, path, lower=True):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f)["model"]["vocab"], lower=lower)

    def word_pieces(self, word):
        if len(word) > self.max_chars:
            return [UNK]
        pieces, start = [], 0
        while start < len(word):
            end, found = len(word), None
            while start < end:
                piece = word[start:end] if start == 0 else "##" + word[start:end]
                if piece in self.vocab:
                    found = piece
                    break
                end -= 1
            if found is None:
                return [UNK]
            pieces.append(found)
            start = end
        return pieces

    def tokenize(self, text):
        return [p for w in basic_tokens(text, self.lower) for p in self.word_pieces(w)]

    def ids(self, tokens):
        return [self.vocab[t] for t in tokens]

    def encode(self, text, max_len=None):
        """[CLS] pieces [SEP] as ids, cut to max_len (the closing [SEP] is kept)."""
        toks = self.tokenize(text)
        if max_len is not None:
            toks = toks[:max_len - 2]
        return self.ids([CLS] + toks + [SEP])

    @staticmethod
    def detokenize(tokens):
        """Pieces back to text: '##x' joins to the previous piece, others are space-separated."""
        text = ""
        for t in tokens:
            text += t[2:] if t.startswith("##") else (" " + t if text else t)
        return text
