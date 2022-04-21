# coding=utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE.apache-2.0 file in the root directory of this source tree.

import os
import struct
from itertools import accumulate
from typing import Union

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset

from multilingual.data.indexing import NumpyUtils, PathUtils


class IndexedDatasetBuilder(object):
    """
    Builder of Indexed dataset builder

    Args:
        output_file_path (str): output file path
    """

    def __init__(self, output_file_path):
        self.out_file = open(output_file_path, "wb")
        self.dtype = np.int32
        self.data_offsets = [0]
        self.dim_offsets = [0]
        self.sizes = []
        self.element_size = NumpyUtils.element_sizes[self.dtype]
        self.doc_idx = [0]

    def add_item(self, tensor: Tensor):
        """
        Add item to output file

        Args:
            tensor (Tensor): tensor to save
        """
        _bytes = np.array(tensor.numpy(), dtype=self.dtype)
        _bytes = self.out_file.write(_bytes)
        self.data_offsets.append(self.data_offsets[-1] + (_bytes / self.element_size))

        for s in tensor.size():
            self.sizes.append(s)

        self.dim_offsets.append(self.dim_offsets[-1] + len(tensor.size()))

    def end_document(self):
        self.doc_idx.append(len(self.sizes))

    def merge_file_(self, another_file_path):
        """
        Merge another file to current file

        Args:
            another_file_path (str): another file path to merge
        """
        index = IndexedDataset(another_file_path)
        assert index.dtype == self.dtype

        begin = self.data_offsets[-1]
        for offset in index.data_offsets[1:]:
            self.data_offsets.append(begin + offset)

        self.sizes.extend(index.sizes)
        begin = self.dim_offsets[-1]
        for dim_offset in index.dim_offsets[1:]:
            self.dim_offsets.append(begin + dim_offset)

        with open(PathUtils.data_path(another_file_path), "rb") as f:
            while True:
                data = f.read(1024)
                if data:
                    self.out_file.write(data)
                else:
                    break

    def finalize(self, index_file_path: str):
        """
        Write binary file

        Args:
            index_file_path (str): index file path
        """
        self.out_file.close()
        # 1. Create binary data
        index = open(index_file_path, "wb")
        magic = b"TNTIDX\x00\x00"
        version = struct.pack("<Q", 1)
        code_and_size = struct.pack(
            "<QQ",
            NumpyUtils.cody_by_dtypes[self.dtype],
            self.element_size,
        )
        offsets = struct.pack(
            "<QQ",
            len(self.data_offsets) - 1,
            len(self.sizes),
        )
        document_count = struct.pack(
            "<Q",
            len(self.doc_idx),
        )

        # 2. Write binary data to index file
        index.write(magic)
        index.write(version)
        index.write(code_and_size)
        index.write(offsets)
        index.write(document_count)

        # 3. Write array data to index file
        NumpyUtils.write_file_from_array(index, self.dim_offsets)
        NumpyUtils.write_file_from_array(index, self.data_offsets)
        NumpyUtils.write_file_from_array(index, self.sizes)
        NumpyUtils.write_file_from_array(index, self.doc_idx)
        index.close()


class IndexedDataset(Dataset):
    """
    Copy of ``IndexedDataset`` in ``fairseq``.

    Args:
        path (str): dataset path

    Attributes:
        _HDR_MAGIC: a key that determines the format of the dataset implementation (first 8 bytes of binary file)
        path: dataset path
        data_file: data from binarized file (path + '.bin')
        element_size: size of each element of sample's dtype
        _len: total length of data samples
        _sizes: total length of data samples
        sizes: the number of tokens of each sample
        doc_count: total number of documents
        dim_offsets: tensor dimension offsets
        data_offsets: data offsets
        doc_idx: array of document index

    References:
        https://github.com/pytorch/fairseq/blob/master/fairseq/data/indexed_dataset.py
    """

    def __init__(self, path: str):
        self._HDR_MAGIC = b"TNTIDX\x00\x00"
        self.path = path
        self.data_file = None
        self.element_size = None
        self.dtype = None
        self._len = None
        self._sizes = None
        self.sizes = None
        self.doc_count = None
        self.dim_offsets = None
        self.data_offsets = None
        self.doc_idx = None
        self.read_index(path)
        # read index file from path

    def read_index(self, path: str):
        """
        Read indexed dataset information from given file path

        Args:
            path (str): dataset path

        Notes:
            binary index file (.idx) structure:
            {
                magic (8: unsigned long long),
                version (8: unsigned long long),

                struct (16){
                    dtype_code (8: unsigned long long)
                    element_size (8: unsigned long long)
                },
                struct (16){
                    _len (8: unsigned long long)
                    sizes (8: unsigned long long)
                },
                struct (8){
                    doc_count (8: unsigned long long)
                },
                ...
            }

        References:
            https://docs.python.org/3/library/struct.html
            '<': little endian
            'Q': unsigned long long
        """

        with open(PathUtils.idx_path(path), "rb") as idx_file:
            # 1. Read first 8 bytes to detect format of dataset implementation
            # ex) ``magic``= b"TNTIDX\x00\x00"
            magic = idx_file.read(8)
            assert magic == self._HDR_MAGIC, (
                "Index file doesn't match expected format. "
                "Please check your configuration file."
            )

            # 2. Read second 8 bytes to detect version of dataset.
            # ex) ``version``=b'\x01\x00\x00\x00\x00\x00\x00\x00'
            idx_file.read(8)

            # 3. Read struct that consists dtype code and element size.
            # ex) type of token is np.int32 => ``dtype_code``=4, ``element_size``=4
            dtype_code, self.element_size = struct.unpack("<QQ", idx_file.read(16))
            self.dtype = NumpyUtils.dtypes_by_code[dtype_code]

            # 4. Read struct that consists length and sizes.
            # Note that ``_size (int)`` is difference with ``sizes (np.array)``.
            # ex) the number of samples is 41088 => ``len``=41088 ``_sizes``=41088
            self._len, self._sizes = struct.unpack("<QQ", idx_file.read(16))

            # 5. Read struct that consists document count.
            # ex) doc_count:tuple=(41089,)
            self.doc_count = struct.unpack("<Q", idx_file.read(8))

            # 6. Create ``dim_offsets`` and ``data_offsets`` arrays from ``_len``
            # ex) the number of samples is 41088,
            #     dim_offsets=[0, 1, 2, 3, ... 41086, 41087, 41088]
            #     data_offsets=[0, 28, 71, 107, ... 1311924, 1311955, 1311983]
            self.dim_offsets = NumpyUtils.create_array_from_file(
                idx_file,
                self._len + 1,
            )
            self.data_offsets = NumpyUtils.create_array_from_file(
                idx_file,
                self._len + 1,
            )

            # 7. Create ``sizes`` array from ``_sizes``
            # ex) sizes=[28 tokens, 43 tokens, 36 tokens, ...]
            self.sizes = NumpyUtils.create_array_from_file(
                idx_file,
                self._sizes,
            )

            # 8. Create ``doc_idx`` array from ``doc_count``
            # ex) doc_idx=[0, 1, 2, 3, ... 41086, 41087, 41088]
            self.doc_idx = NumpyUtils.create_array_from_file(
                idx_file,
                self.doc_count,
            )

    def read_data(self, path: str) -> None:
        """
        Read data file (.bin) from path

        Args:
            path (str): data path
        """
        self.data_file = open(
            PathUtils.data_path(path),
            "rb",
            buffering=0,
            # buffering off
        )

    def check_index(self, idx: int) -> None:
        """
        Check given index is valid.

        Args:
            idx (int): index

        Raises:
            IndexError: raise error if ``idx`` smaller than 0 or bigger than the number of total data samples.
        """
        if idx < 0 or idx >= self._len:
            raise IndexError("index out of range")

    def __getitem__(self, idx: Union[int, slice]) -> np.ndarray:
        """
        Get item from datasets index.

        Notes:
            Process of loading data is performed when this method is first called.
            That's why this dataset implementation is called ``lazy``.

        Args:
            idx (Union[int, slice]): index or slice

        Returns:
            np.array: Data by index
        """
        if not self.data_file:
            # Lazy data loading.
            self.read_data(self.path)

        if isinstance(idx, int):
            # check index is valid
            self.check_index(idx)

            # compute tensor size
            tensor_size = self.sizes[self.dim_offsets[idx] : self.dim_offsets[idx + 1]]

            # create empty array to hold the data
            array = np.empty(tensor_size, dtype=self.dtype)

            # sets the file's current position at the offset.
            self.data_file.seek(self.data_offsets[idx] * self.element_size)

            # read data into array
            self.data_file.readinto(array)
            return array

        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                # slices must be contiguous.
                raise ValueError("Slices into indexed_dataset must be contiguous")

            # compute tensor size and total tensor size
            sizes = self.sizes[self.dim_offsets[start] : self.dim_offsets[stop]]
            total_size = sum(sizes)

            # create empty array to hold the data
            array = np.empty(total_size, dtype=self.dtype)

            # sets the file's current position at the offset.
            self.data_file.seek(self.data_offsets[start] * self.element_size)
            self.data_file.readinto(array)

            # split datasets by the number of tokens
            offsets = list(accumulate(sizes))
            sentences = np.split(array, offsets[:-1])
            return sentences

    def __len__(self) -> int:
        """
        Returns the number of total datasets

        Returns:
            int: the number of total datasets
        """
        return self._len

    def num_tokens(self, index: Union[int, slice]) -> Union[int, tuple]:
        """
        Number of tokens by given index

        Args:
            index (Union[int, slice]): indices of dataset

        Returns:
            Union[int, slice]: the numbers of tokens by given indices
        """
        return self.sizes[index]

    def size(self, index: Union[int, slice]) -> Union[int, tuple]:
        """
        Returns data size by given index

        Args:
            index (Union[int, slice]): indices of dataset

        Returns:
            Union[int, tuple]: sizes of data samples by given indices
        """
        return self.sizes[index]

    @staticmethod
    def exists(path: str) -> bool:
        """
        Is exist file path of not

        Args:
            path (str): file path

        Returns:
            bool: exist or not
        """
        return os.path.exists(PathUtils.idx_path(path)) and os.path.exists(
            PathUtils.data_path(path)
        )

    @property
    def supports_prefetch(self) -> bool:
        """
        Check indexed dataset supports cache prefetching

        Returns:
            bool: whether support prefetching or not
        """
        return False

    @staticmethod
    def builder(path: str) -> IndexedDatasetBuilder:
        """
        Get build object to build lazy indexed dataset

        Args:
            path (str): output file path

        Returns:
            IndexedDatasetBuilder: indexed dataset builder object
        """
        return IndexedDatasetBuilder(path)
