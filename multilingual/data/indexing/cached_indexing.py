# coding=utf-8
# Copyright 2021 TUNiB Inc.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE.apache-2.0 file in the root directory of this source tree.

from functools import lru_cache
from typing import List, Union

import numpy as np

from multilingual.data.indexing.lazy_indexing import IndexedDataset


class IndexedCachedDataset(IndexedDataset):
    """
    Copy of ``IndexedCachedDataset`` from ``fairseq``.

    Args:
        path (str): dataset path

    Attributes:
        cache (np.array): in-memory cached array
        cache_index (dict): indices of cached samples.
    """

    def __init__(self, path: str):
        super().__init__(path)
        self.cache = None
        self.cache_index = {}

    @property
    def supports_prefetch(self):
        """
        Check indexed dataset supports cache prefetching

        Returns:
            bool: whether support prefetching or not
        """

        return True

    def prefetch(self, indices: List[int]) -> None:
        """
        Prefetch dataset by given indices

        Args:
            indices (List[int]): dataset indices
        """
        if all(i in self.cache_index for i in indices):
            # If all indices are cached, quit method.
            return

        if not self.data_file:
            # If dataset is not loaded, load dataset from external memory.
            self.read_data(self.path)

        # Sort indices to compute total size from ``data_offsets``, contiguous array.
        indices = sorted(set(indices))

        total_size = 0
        for i in indices:
            total_size += self.data_offsets[i + 1] - self.data_offsets[i]

        # Create cache array
        self.cache = np.empty(
            total_size,
            dtype=self.dtype,
        )

        # Ensure cache_index is cleared array.
        self.cache_index.clear()
        ptx = 0

        for i in indices:
            # store total array size of from start to end.
            self.cache_index[i] = ptx

            # get slice from total cached array by ptx size
            size = self.data_offsets[i + 1] - self.data_offsets[i]
            array = self.cache[ptx : ptx + size]

            # sets the file's current position at the offset.
            self.data_file.seek(self.data_offsets[i] * self.element_size)

            # read data into array
            self.data_file.readinto(array)
            ptx += size

        if self.data_file:
            # close and delete data file after prefetch so we can pickle
            self.data_file.close()
            self.data_file = None

    @lru_cache(maxsize=8)
    def __getitem__(self, idx: Union[int, tuple]) -> Union[np.ndarray, List]:
        """
        Get item from given index or indices

        Args:
            idx (Union[int, tuple]: index or indices

        Returns:
            Union[np.ndarray, List]: loaded datasets
        """
        if isinstance(idx, int):
            # check index is valid
            self.check_index(idx)

            # compute tensor size
            tensor_size = self.sizes[self.dim_offsets[idx] : self.dim_offsets[idx + 1]]

            # create empty array to hold the data
            array = np.empty(tensor_size, dtype=self.dtype)

            # load data from cached array (not file access)
            ptx = self.cache_index[idx]

            # copy cached data to array
            np.copyto(array, self.cache[ptx : ptx + array.size])
            return array

        elif isinstance(idx, slice):
            # Hack just to make this work, can optimizer later if necessary
            return [self[i] for i in range(*idx.indices(len(self)))]
