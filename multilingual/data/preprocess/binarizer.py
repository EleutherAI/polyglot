# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from time import time
from typing import Iterable, Union

import torch

from multilingual.data.indexing import PathUtils
from multilingual.data.indexing.cached_indexing import IndexedCachedDataset
from multilingual.data.indexing.lazy_indexing import (
    IndexedDataset,
    IndexedDatasetBuilder,
)
from multilingual.data.indexing.mmap_indexing import (
    MMapIndexedDataset,
    MMapIndexedDatasetBuilder,
)


class DatasetBinarizer(object):
    """
    Dataset binarizer for preprocessing

    Args:
        impl (str): type of binarization implementation
    """

    def __init__(self, impl: str):
        impl = impl.lower()

        assert impl in [
            "lazy",
            "cached",
            "mmap",
        ], "Binarization implementation is must be one of [lazy, cached, mmap]"

        self.impl = impl
        self.builder = {
            "lazy": IndexedDataset.builder,
            "cached": IndexedCachedDataset.builder,
            "mmap": MMapIndexedDataset.builder,
        }

    def create_builder(self, path: str):
        """
        Create binarization builders

        Args:
            path (str): path of the dataset to be saved
        """

        data_path = PathUtils.data_path(path)
        index = PathUtils.idx_path(path)
        builder = self.builder[self.impl](data_path)

        return index, builder

    @staticmethod
    def binarize(
        iterator: Iterable,
        index_path: str,
        builder: Union[
            IndexedDatasetBuilder,
            MMapIndexedDatasetBuilder,
        ],
        log_interval: int,
    ) -> None:
        """Tokenize and encode raw text dataset for binarization"""
        start_time = time()
        total_bytes_processed = 0

        for i, (doc, byte) in enumerate(iterator, start=1):
            total_bytes_processed += byte * 4
            if len(doc) == 0:
                continue
            builder.add_item(torch.IntTensor(doc))
            builder.end_document()

            if i % log_interval == 0:
                elapsed = time() - start_time
                mbs = total_bytes_processed / elapsed / (1024 ** 2)

                print(
                    f" Processed {i} documents."
                    f" ({round(i/elapsed, 4)} docs/s, {round(mbs, 4)} MiB/s)."
                )

        builder.finalize(index_path)
