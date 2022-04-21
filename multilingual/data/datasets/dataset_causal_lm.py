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


import os

import numpy as np
import torch.distributed as dist
from torch.utils.data import Dataset

from multilingual.data.indexing.cached_indexing import IndexedCachedDataset
from multilingual.data.indexing.lazy_indexing import IndexedDataset
from multilingual.data.indexing.mmap_indexing import MMapIndexedDataset
from multilingual.utils import get_datasets_builder


class DatasetForCausalLM(Dataset):
    """
    Dataset for Causal Language Modeling

    Args:
        data_name (str): the name of dataset (data file)
        max_seq_length (int): max sequence length
        binarization_impl (str): type of binarization implementation. one of ['mmap', 'lazy', 'cached'].
        split_type (str): split type e.g. "train", "valid", "test"
        start_weight (float): dataset start weight, default is 0.0
        end_weight (float) : dataset end weight, default is 0.0
        seed (int): random seed value

    Notes:
        How to remove all the padding tokens?
            1. input: [doc_1, doc_2, doc_3, doc_4, doc_5, ..., doc_n]
            2. shuffle: [doc_1, doc_4, doc_2, doc_3, doc_5, ..., doc_n]
            3. concat: [doc_1 <eod> doc_4 <eod> doc_3 <eod>, ..., max_seq_length]

        What is binarization impl?
            This is how data is loaded. If we set ``binarization_impl`` to 'cached' it will load all dataset into memory.
            This is suitable for fine-tuning with relatively smaller dataset size than pre-training.
            But, if the size of the data is very large, we must load only the required amount of data at each step.
            If we set ``binarization_impl`` to 'lazy', we loads required amount of data from disk at each step.
            However, speed of disk is very slower than CPU and GPU. Therefore, we also support ``mmap`` method.
            if we set ``binarization_impl`` to 'mmap', it uses memory mapping. With the memory mapping data address is mapped to the virtual memory.
            Then, we can manipulate the data on the disk as if it exists in memory without disk I/O.
    """

    def __init__(
        self,
        data_name: str,
        max_seq_length: int,
        binarization_impl: str,
        split_type: str = "train",
        start_weight: float = 0.0,
        end_weight: float = 1.0,
        seed: int = None,
    ):
        assert binarization_impl in ["lazy", "mmap", "cached"], (
            "Param ``binarization_impl`` must be one of 'lazy', 'mmap' and 'cached'. "
            "For description of each method, please refer to the docstring."
        )

        assert 0.0 <= start_weight < 1.0, "Param ``start_weight`` must be 0.0 to 1.0."
        assert 0.0 < end_weight <= 1.0, "Param ``end_weight`` must be 0.0 to 1.0."
        assert (
            end_weight > start_weight
        ), "Param ``end_weight`` must be larger than ``start_weight``."
        assert (
            start_weight != end_weight
        ), "Param ``start_weight`` and ``end_weight`` must not be same."

        self.name = data_name
        self.index = {
            "lazy": IndexedDataset,
            "cached": IndexedCachedDataset,
            "mmap": MMapIndexedDataset,
        }[binarization_impl](data_name)

        num_samples = len(self.index)
        start_index = int(round(start_weight * float(num_samples)))
        end_index = int(round(end_weight * float(num_samples)))
        indexer = IndexingForCausalLM()

        self.doc_idx, self.sample_idx, self.shuffle_idx = indexer.build_index_mappings(
            data_name=data_name,
            start_index=start_index,
            end_index=end_index,
            split_type=split_type,
            sizes=self.index.sizes,
            num_samples=num_samples,
            max_seq_length=max_seq_length,
            seed=seed,
        )

    def __len__(self) -> int:
        """
        Length of dataset

        Returns:
            int: length of dataset
        """
        return self.sample_idx.shape[0] - 1

    def __getitem__(self, idx: int):
        """
        Get item from given index

        Args:
            idx (int): index of dataset

        Returns:
            Dict[str, np.ndarray]: sample by given index
        """
        # Get the shuffled index.
        idx = self.shuffle_idx[idx]

        # Start and end documents and offsets.
        doc_index_f = self.sample_idx[idx][0]
        doc_index_l = self.sample_idx[idx + 1][0]
        offset_f = self.sample_idx[idx][1]
        offset_l = self.sample_idx[idx + 1][1]

        # If we are within the same document, just extract the chunk.
        if doc_index_f == doc_index_l:
            sample = self.index.get(
                self.doc_idx[doc_index_f],
                offset=offset_f,
                length=offset_l - offset_f + 1,
            )

        else:
            # Otherwise, get the rest of the initial document.
            sample_list = [self.index.get(self.doc_idx[doc_index_f], offset=offset_f)]

            # Loop over all in between documents and add the entire document.
            for i in range(doc_index_f + 1, doc_index_l):
                sample_list.append(self.index.get(self.doc_idx[i]))

            # And finally add the relevant portion of last document.
            sample_list.append(
                self.index.get(self.doc_idx[doc_index_l], length=offset_l + 1)
            )

            sample = np.concatenate(sample_list)

        return np.array(sample, dtype=np.int64)


class IndexingForCausalLM(object):
    """
    Binary index mapper for CausalLM dataset
    Creates index mapping file between binary index and torch dataset

    Notes:
        DatasetForCausalLM works like the followings.
            1. shuffle dataset
            - shuffle input: [doc_1, doc_2, doc_3, doc_4, doc_5, ..., doc_n]
            - shuffle output: [doc_1, doc_4, doc_2, doc_3, doc_5, ..., doc_n]

            2. concat dataset (every sample has max token, e.g.=2048, so you don't have any pad tokens)
            - concat input: [doc_1, doc_4, doc_2, doc_3, doc_5, ..., doc_n]
            - concat output (no-eod, max=2048): doc_1 doc_4 doc_2 doc_3 doc_5 ... 2048 tokens
            - concat output (use-eod, max=2048): doc_1 <eod> doc_4 <eod> doc_3 <eod> ... 2048 tokens

        DatasetForCausalLM makes the following index files (.npy file)
            doc-idx: ordered array of documents to be used in training.
            sample-idx: the start index and offset for each training sample.
            shuffle-idx: the sample index into a random index into sample-idx.
    """

    def build_index_mappings(
        self,
        data_name: str,
        split_type: str,
        start_index: int,
        end_index: int,
        sizes: np.ndarray,
        num_samples: int,
        max_seq_length: int,
        seed: int,
    ):
        """
        Build doc-idx, sample-idx, and shuffle-idx.

        Notes:
            doc-idx: an ordered array of documents to be used in training
            sample-idx: the start document index and document offset for each training sample.
            shuffle-idx: the sample index into a random index into sample-idx.
        """

        local_rank = int(os.getenv("LOCAL_RANK", default=0))
        world_size = int(os.getenv("WORLD_SIZE", default=1))

        np_rng = np.random.RandomState(seed if seed is not None else 42)
        doc_indices = np.arange(start_index, end_index, dtype=np.int32)
        num_tokens = self._num_tokens(sizes, doc_indices)

        # Filename of the index mappings.
        # e.g. dataset_train_length=2048_seed=42_XXX_idx.npy
        _filename = data_name
        _filename += "_{}".format(split_type)
        _filename += "_samples={}".format(num_samples)
        _filename += "_length={}".format(max_seq_length)
        if seed is not None:
            _filename += "_seed={}".format(seed)

        doc_idx_filename = _filename + "_doc_idx.npy"
        sample_idx_filename = _filename + "_sample_idx.npy"
        shuffle_idx_filename = _filename + "_shuffle_idx.npy"

        if local_rank == 0:
            if (
                not os.path.isfile(doc_idx_filename)
                or not os.path.isfile(sample_idx_filename)
                or not os.path.isfile(shuffle_idx_filename)
            ):
                self._create_index_mappings(
                    doc_idx_filename=doc_idx_filename,
                    sample_idx_filename=sample_idx_filename,
                    shuffle_idx_filename=shuffle_idx_filename,
                    doc_indices=doc_indices,
                    sizes=sizes,
                    num_tokens=num_tokens,
                    max_seq_length=max_seq_length - 1,
                    np_rng=np_rng,
                )

        if world_size > 1:
            if not dist.is_initialized():
                dist.init_process_group("nccl")
            dist.barrier()

        doc_idx = np.load(doc_idx_filename, allow_pickle=True, mmap_mode="r")
        sample_idx = np.load(sample_idx_filename, allow_pickle=True, mmap_mode="r")
        shuffle_idx = np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode="r")

        return doc_idx, sample_idx, shuffle_idx

    @staticmethod
    def _num_tokens(sizes: np.ndarray, doc_indices: np.ndarray) -> np.ndarray:
        """Total number of tokens in the dataset."""
        return np.sum(sizes[doc_indices])

    def _create_index_mappings(
        self,
        doc_idx_filename,
        sample_idx_filename,
        shuffle_idx_filename,
        doc_indices,
        sizes,
        num_tokens,
        max_seq_length,
        np_rng,
    ):

        # 1. Build doc-idx.
        # e.g. [    89   9798 118303 ... 131932 146867 121958]
        doc_idx = self._build_doc_idx(doc_indices, np_rng)

        # 2. Build sample-idx
        # e.g. [[     0      0]
        #      [     1    222]
        #      [     1    734]
        #      ...
        #      [151655  13428]
        #      [151655  13940]
        #      [151655  14452]]
        assert doc_idx.dtype == np.int32
        assert sizes.dtype == np.int32
        cpp_builder = get_datasets_builder()

        if cpp_builder is not None:
            # Using C++ implementation for high speed.
            sample_idx = cpp_builder.build_sample_idx(
                sizes,
                doc_idx,
                max_seq_length,
                1,
                num_tokens,
            )
        else:
            sample_idx = self._build_sample_idx(
                sizes,
                doc_idx,
                max_seq_length,
                1,
                num_tokens,
            )

        # 3. Build Shuffle-idx.
        # e.g. [1395917  598303    6096 ...  991038  962985 1006638]
        shuffle_idx = self._build_shuffle_idx(
            total_size=sample_idx.shape[0] - 1,
            np_rng=np_rng,
        )

        np.save(doc_idx_filename, doc_idx, allow_pickle=True)
        np.save(sample_idx_filename, sample_idx, allow_pickle=True)
        np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)

    @staticmethod
    def _build_doc_idx(doc_indices, np_rng):
        indices = np.copy(doc_indices)
        np_rng.shuffle(indices)
        return indices

    @staticmethod
    def _build_sample_idx(
        sizes,
        doc_idx,
        seq_length,
        num_epochs,
        tokens_per_epoch,
    ):
        num_samples = (num_epochs * tokens_per_epoch - 1) // seq_length
        sample_idx = np.zeros([num_samples + 1, 2], dtype=np.int32)

        sample_index = 0
        doc_idx_index = 0
        doc_offset = 0

        sample_idx[sample_index][0] = doc_idx_index
        sample_idx[sample_index][1] = doc_offset
        sample_index += 1

        while sample_index <= num_samples:
            remaining_seq_length = seq_length + 1

            while remaining_seq_length != 0:
                doc_id = doc_idx[doc_idx_index]
                doc_length = sizes[doc_id] - doc_offset
                remaining_seq_length -= doc_length

                if remaining_seq_length <= 0:
                    doc_offset += remaining_seq_length + doc_length - 1
                    remaining_seq_length = 0
                else:
                    doc_idx_index += 1
                    doc_offset = 0

            sample_idx[sample_index][0] = doc_idx_index
            sample_idx[sample_index][1] = doc_offset
            sample_index += 1

        return sample_idx

    @staticmethod
    def _build_shuffle_idx(total_size, np_rng):
        dtype_ = np.uint32

        if total_size >= (np.iinfo(np.uint32).max - 1):
            dtype_ = np.int64

        shuffle_idx_first = np.arange(
            start=0,
            stop=total_size,
            step=1,
            dtype=dtype_,
        )

        np_rng.shuffle(shuffle_idx_first)
        return shuffle_idx_first
