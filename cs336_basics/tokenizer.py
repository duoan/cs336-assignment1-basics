import json
import os
from functools import cache

import regex

from cs336_basics.consts import PAT


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes], tuple[bytes]],
        special_tokens: list[str] | None = None,
        **kwargs,
    ) -> None:
        self.idx2tokens = vocab
        self.token2idxs = {token: idx for idx, token in vocab.items()}
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {pair: i for i, pair in enumerate(merges)}
        self.merges = merges
        self.special_tokens = special_tokens

        self._special_tokens_set: set[str] = set(special_tokens) if special_tokens else set()
        self._special_pattern = None
        if special_tokens:
            sorted_tokens = sorted(special_tokens, key=len, reverse=True)

            # add the special token when miss them in vocab
            for special_token in sorted_tokens:
                special_token = special_token.encode("utf-8")
                if special_token not in self.token2idxs:
                    self.idx2tokens[len(self.idx2tokens)] = special_token
                    self.token2idxs[special_token] = self.idx2tokens[len(self.idx2tokens) - 1]

            self._special_pattern = regex.compile("(" + "|".join(regex.escape(tok) for tok in sorted_tokens) + ")")

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        with open(vocab_filepath) as vocab_file:
            vocab: dict[int, bytes] = {}
            for token, idx in json.load(vocab_file).items():
                vocab[token.encode("utf-8")] = idx

        with open(merges_filepath) as merges_file:
            merges: list[tuple[bytes, bytes]] = []
            for merged_tuple_line in merges_file.readlines():
                merged_tokens = merged_tuple_line.rstrip("\n").split(" ")
                merges.append((merged_tokens[0].encode("utf-8"), merged_tokens[1].encode("utf-8")))

        return Tokenizer(vocab, merges, special_tokens)

    @cache
    def _bpe(self, word: str) -> tuple[bytes, ...]:
        # 缓存命中:99% 的词都会命中(自然语言的 Zipf 分布)
        tokens = [bytes([b]) for b in word.encode("utf-8")]

        while len(tokens) >= 2:
            # 找出当前所有 pair 中 rank 最小的那个
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            best = min(
                pairs,
                key=lambda p: self.merge_ranks.get(p, float("inf")),
            )
            if best not in self.merge_ranks:
                break  # 没有可 merge 的 pair 了

            # 把所有的 best pair merge 掉
            new_tokens = []
            i = 0
            merged = best[0] + best[1]
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == best[0] and tokens[i + 1] == best[1]:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tuple(tokens)

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        # pre-tokenize
        chunks = [text] if self._special_pattern is None else self._special_pattern.split(text)

        idxs: list[int] = []

        for chunk in chunks:
            if not chunk:
                continue

            if self._special_tokens_set and chunk in self._special_tokens_set:
                idxs.append(self.token2idxs[chunk.encode("utf-8")])
                continue

            parts = regex.finditer(PAT, chunk)
            for part in parts:
                word = part.group()
                for token in self._bpe(word):
                    idxs.append(self.token2idxs[token])

        return idxs

    def encode_iterable(self, iterable):
        for line in iterable:
            yield from self.encode(line)

    def decode(self, idxs: list[int]) -> str:
        if not idxs:
            return ""

        tokens = b""
        for idx in idxs:
            tokens += self.idx2tokens.get(idx)

        return tokens.decode("utf-8", errors="replace")


if __name__ == "__main__":
    import pathlib

    FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "tests" / "fixtures"
    tokenizer = Tokenizer.from_files(
        vocab_filepath=FIXTURES_PATH / "train-bpe-reference-vocab.json",
        merges_filepath=FIXTURES_PATH / "train-bpe-reference-merges.txt",
        special_tokens=["<|endoftext|>"],
    )
    tokenizer.encode("hello cs226")
