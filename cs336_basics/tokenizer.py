import json
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor

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
        self.merges = merges
        self.special_tokens = special_tokens

        self._special_tokens_set: set[str] = set(special_tokens) if special_tokens else set()
        self._special_pattern = None
        if special_tokens:
            sorted_tokens = sorted(special_tokens, key=len, reverse=True)
            self._special_pattern = regex.compile("(" + "|".join(regex.escape(tok) for tok in sorted_tokens) + ")")

        self._pool = ThreadPoolExecutor(max(1, multiprocessing.cpu_count() - 1))

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

    def _merge_word_token_tuple(self, split):
        word_token_tuple = tuple(bytes([b]) for b in split.encode("utf-8"))
        for merge in self.merges:
            new_word_tokens: list[bytes] = []
            token = merge[0] + merge[1]
            i = 0
            while i < len(word_token_tuple):
                if (
                    i < len(word_token_tuple) - 1
                    and merge[0] == word_token_tuple[i]
                    and merge[1] == word_token_tuple[i + 1]
                ):
                    new_word_tokens.append(token)
                    i += 2
                else:
                    new_word_tokens.append(word_token_tuple[i])
                    i += 1
            word_token_tuple = tuple(new_word_tokens)
        return word_token_tuple

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        # pre-tokenize
        splits: list[str] = []
        chunks = [text] if self._special_pattern is None else self._special_pattern.split(text)
        for chunk in chunks:
            if not chunk:
                continue

            if self._special_tokens_set and chunk in self._special_tokens_set:
                splits.append(chunk)
                continue

            parts = regex.finditer(PAT, chunk)
            for part in parts:
                splits.append(part.group())

        # merge
        merged_word_tokens = []
        futures = []
        for split in splits:
            if split in self._special_tokens_set:
                futures.append(None)
            else:
                futures.append(self._pool.submit(self._merge_word_token_tuple, split))

        for i, future in enumerate(futures):
            if future is None:
                merged_word_tokens.append(splits[i].encode("utf-8"))
            else:
                merged_word_tokens.extend(list(future.result()))

        idxs: list[int] = []

        for token in merged_word_tokens:
            idxs.append(self.token2idxs.get(token, -1))

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
