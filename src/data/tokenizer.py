from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]


class BasicTokenizer:
    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.pad_id = vocab["[PAD]"]
        self.unk_id = vocab["[UNK]"]
        self.bos_id = vocab["[BOS]"]
        self.eos_id = vocab["[EOS]"]

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return text.lower().strip().split()

    def encode(self, text: str, max_len: int) -> List[int]:
        tokens = self.tokenize(text)
        ids = [self.bos_id] + [self.vocab.get(t, self.unk_id) for t in tokens] + [self.eos_id]
        if len(ids) > max_len:
            ids = ids[: max_len - 1] + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        tokens = [self.inv_vocab.get(i, "[UNK]") for i in ids]
        return " ".join(tokens)


def build_vocab(texts: Iterable[str], vocab_size: int) -> Dict[str, int]:
    counter = Counter()
    for t in texts:
        counter.update(BasicTokenizer.tokenize(t))
    most_common = [w for w, _ in counter.most_common(vocab_size - len(SPECIAL_TOKENS))]
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    offset = len(vocab)
    for i, w in enumerate(most_common):
        vocab[w] = offset + i
    return vocab


def save_vocab(vocab: Dict[str, int], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f)


def load_vocab(path: str | Path) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
