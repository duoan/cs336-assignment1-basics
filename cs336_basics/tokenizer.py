import os


class Tokenizer:
    def __init__(
        self,
        vocab: dict[tuple[bytes, ...], int],
        merges: list[tuple[bytes], tuple[bytes]],
        special_tokens: list[str] | None = None,
        **kwargs,
    ) -> None:
        pass

    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        pass

    def encode(self, text: str) -> list[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        pass
