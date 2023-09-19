from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import math
from typing import TypeVar, Optional, Iterator

import torch
import torch.distributed as dist
from torch import Tensor
from torch import nn
from torch.nn import functional as F

T_co = TypeVar("T_co", covariant=True)


class BaseLoader(DataLoader):
    pass


class BlockShuffleDistSampler(DistributedSampler):
    def __init__(self, block_size=1, ignore_steps=None, **kwargs):
        super().__init__(**kwargs)
        self.block_size = block_size
        self.ignore_steps = ignore_steps
        self.batch_size = 128

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            N = len(self.dataset)
            K = (N - 1) // self.block_size + 1  # number of blocks
            N_ = K * self.block_size
            indices = torch.arange(N_).reshape(K, self.block_size)
            indices = indices[torch.randperm(K, generator=g)].reshape(N_)  # type: ignore[arg-type]
            indices = indices[indices < N].tolist()
            assert len(indices) == N, "Invalid indicies!"
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples
        # Skip batch
        if self.ignore_steps is not None and self.ignore_steps >= 0:
            for i in range(
                min(
                    len(indices),
                    (self.ignore_steps + 1) * self.batch_size // self.num_replicas,
                )
            ):
                indices[i] = -100
        return iter(indices)
