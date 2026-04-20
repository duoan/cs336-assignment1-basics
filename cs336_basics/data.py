import os

import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    ix = np.random.randint(0, len(dataset) - context_length, size=(batch_size, 1))
    offsets = np.arange(context_length)
    x_indices = ix + offsets
    y_indices = ix + offsets + 1

    x_np = dataset[x_indices].astype(np.int64)
    y_np = dataset[y_indices].astype(np.int64)
    x_tensor = torch.from_numpy(x_np).to(device=device, non_blocking=True)
    y_tensor = torch.from_numpy(y_np).to(device=device, non_blocking=True)

    return x_tensor, y_tensor


class NumpyBinaryTokenDataset(torch.utils.data.Dataset):
    def __init__(self, input_path: str | os.PathLike, context_length: int):
        self.data = np.memmap(input_path, dtype=np.uint16, mode="r")
        self.context_length = context_length
        self.total_token = len(self.data)

    def __len__(self):
        return self.total_token - self.context_length

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.context_length + 1]
        chunk_tensor = torch.from_numpy(chunk.astype(np.int64))
        x = chunk_tensor[:-1]
        y = chunk_tensor[1:]

        return x, y
