# Copyright 2021 TUNiB Inc.

from operator import itemgetter
from typing import Iterator, Optional

from torch.utils.data import Dataset, DistributedSampler, Sampler


class DistributedProxySampler(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.

    This is copy of ``catalyst.data.sampler.DistributedSamplerWrapper``

    Args:
        sampler: Sampler used for subsampling
        num_replicas (int): Number of processes participating in distributed training
        rank (int): Rank of the current process within ``num_replicas``
        shuffle (bool): If true, sampler will shuffle the indices
        seed (int): random seed value
        drop_last (bool): If true, skip last batch
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        super(DistributedProxySampler, self).__init__(
            dataset=DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.

        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.

    This is copy of ``catalyst.data.dataset.DatasetFromSampler``

    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        return len(self.sampler)
