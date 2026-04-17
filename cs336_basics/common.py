import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import BinaryIO


def stream_chunks_by_special_token(
    file_path: str | os.PathLike,
    split_special_token: bytes = b"<|endoftext|>",
    read_buffer_size: int = 1 << 16,  # 64 KB
) -> Iterator[bytes]:
    """
    Stream the file, yielding one chunk per `split_special_token` occurrence.
    Each chunk contains the bytes BEFORE the token (the token itself is consumed).
    The final tail (after the last token, or whole file if no token) is also yielded.
    """
    assert split_special_token, "split_special_token must be non-empty"

    token = split_special_token
    token_len = len(token)
    buf = bytearray()

    with open(file_path, "rb") as f:
        while True:
            data = f.read(read_buffer_size)
            if not data:
                # EOF:剩下的全 yield
                if buf:
                    yield bytes(buf)
                return

            buf.extend(data)

            # 把 buf 里所有完整的 token 都切出来
            search_from = 0
            while True:
                idx = buf.find(token, search_from)
                if idx == -1:
                    # 没找到完整 token
                    # 留 token_len-1 字节,防止 token 跨 read 边界被漏掉
                    keep_from = max(0, len(buf) - (token_len - 1))
                    del buf[:keep_from]
                    break

                # ✅ 找到一个,yield token 前面的内容(不含 token 本身)
                yield bytes(buf[:idx])
                # 跳过 token,继续在剩余 buf 里找
                del buf[: idx + token_len]
                search_from = 0


def _find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


@contextmanager
def open_file_chunks(
    file_path: str | os.PathLike,
    num_processes: int = 4,
) -> Iterable[str]:
    ## Usage
    with open(file_path, "rb") as f:
        boundaries = _find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        def _gen():
            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                yield f.read(end - start).decode("utf-8", errors="ignore")

        yield _gen()
