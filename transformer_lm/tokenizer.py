"""Byte-level BPE tokenizer (no regex pre-tokenization)."""

from __future__ import annotations

from collections import Counter
import re


def merge_ids(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace each adjacent occurrence of ``pair`` with ``new_id``."""
    merged: list[int] = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            merged.append(new_id)
            i += 2
        else:
            merged.append(ids[i])
            i += 1
    return merged


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[int, int]]]:
    special_tokens = special_tokens or []
    with open(input_path, encoding="utf-8", mode="r") as f:
        text = f.read()
    vocab = {i: bytes([i]) for i in range(256)}
    ids = list(text.encode("utf-8"))
    merges: list[tuple[int, int]] = []
    num_merges = vocab_size - 256 - len(special_tokens)
    for _ in range(num_merges):
        if len(ids) < 2:
            break
        pair_counts = Counter(
            (ids[i], ids[i + 1]) for i in range(len(ids) - 1)
        )
        pair = min(pair_counts, key=lambda p: (-pair_counts[p], p[0], p[1])) # Was using function, lambda suggested by cursor
        new_id = 256 + len(merges)
        merges.append(pair)
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        ids = merge_ids(ids, pair, new_id)  # adjacent replacement helper
    next_id = 256 + len(merges)
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1
    return vocab, merges


class BPETokenizer:
    """Byte-level BPE tokenizer."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[int, int]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = list(special_tokens or [])
        self.merge_rank = {pair: i for i, pair in enumerate(merges)}
        self.byte_to_id = {b: i for i, b in vocab.items()}
        self.special_to_id = {
            tok: self.byte_to_id[tok.encode("utf-8")]
            for tok in self.special_tokens
            if tok.encode("utf-8") in self.byte_to_id
        }

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs."""
        if text == "":
            return []

        if self.special_tokens:
            pattern = (
                "("
                + "|".join(
                    re.escape(tok)
                    for tok in sorted(self.special_tokens, key=len, reverse=True)
                )
                + ")"
            )
            chunks = re.split(pattern, text)
        else:
            chunks = [text]

        ids: list[int] = []
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.special_to_id:
                ids.append(self.special_to_id[chunk])
                continue

            chunk_ids = list(chunk.encode("utf-8"))
            for i, pair in enumerate(self.merges):
                chunk_ids = merge_ids(chunk_ids, pair, 256 + i)
            ids.extend(chunk_ids)
        return ids

    def decode(self, ids: list[int]) -> str:
        token_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return token_bytes.decode("utf-8", errors="replace")

