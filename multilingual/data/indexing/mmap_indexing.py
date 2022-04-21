# coding=utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE.apache-2.0 file in the root directory of this source tree.

import os
import shutil
import struct
from functools import lru_cache
from itertools import accumulate
from typing import List, Tuple, Union

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset

from multilingual.data.indexing import NumpyUtils, PathUtils


class MMapIndexedDatasetBuilder(object):
    """
    Builder of MMap indexed dataset builder

    Args:
        output_file_path (str): output file path
    """

    def __init__(self, output_file_path: str):
        self._data_file = open(output_file_path, "wb")
        self._dtype = np.int32
        self._sizes = []
        self._doc_idx = [0]

    def add_item(self, tensor: Tensor) -> None:
        """
        Add item to output file

        Args:
            tensor (Tensor): tensor to save
        """
        np_array = np.array(tensor.numpy(), dtype=self._dtype)
        self._data_file.write(np_array.tobytes(order="C"))
        self._sizes.append(np_array.size)

    def end_document(self) -> None:
        """
        Append end of document index
        """
        self._doc_idx.append(len(self._sizes))

    def merge_file_(self, another_file_path: str) -> None:
        """
        Merge another file to current file

        Args:
            another_file_path (str): another file path to merge
        """
        # Concatenate index
        index = MmapIndex(PathUtils.idx_path(another_file_path))
        assert index.dtype == self._dtype

        for size in index.sizes:
            self._sizes.append(size)

        # Concatenate data
        with open(PathUtils.data_path(another_file_path), "rb") as f:
            shutil.copyfileobj(f, self._data_file)

    def finalize(self, index_file_path: str) -> None:
        """
        Write binary file

        Args:
            index_file_path (str): index file path
        """
        self._data_file.close()

        with MmapIndex.writer(index_file_path, self._dtype) as index:
            index.write(self._sizes, self._doc_idx)


class MMapIndexedDataset(Dataset):
    """
    Copy of ``MMapIndexedDataset`` in ``fairseq``.

    Args:
        path (str): data path
    """

    def __init__(self, path: str):
        super().__init__()
        self._path = None
        self._index = None
        self._bin_buffer = None
        self._do_init(path)

    def __getstate__(self) -> str:
        """
        Returns dataset path

        Returns:
            str: dataset path
        """
        return self._path

    def __setstate__(self, state: str) -> None:
        """
        Set dataset state

        Args:
            state (str): file path
        """
        self._do_init(state)

    def _do_init(self, path: str) -> None:
        """
        Initialize dataset

        Args:
            path (str): file path
        """
        self._path = path

        # 1. Create mmap index
        self._index = MmapIndex(PathUtils.idx_path(self._path))

        # 2. Create mmap array and memoryview for data (.bin)
        self._bin_buffer_mmap = np.memmap(
            PathUtils.data_path(self._path), mode="r", order="C"
        )

        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __len__(self) -> int:
        """
        Length of all dataset

        Returns:
            int: length of all dataset
        """
        return len(self._index)

    def __getitem__(self, idx: Union[int, slice]) -> np.ndarray:
        """
        Get item from memory map

        Args:
            idx (Union[int, slice]): index

        Returns:
            np.ndarray: loaded data sample
        """
        if isinstance(idx, int):
            # Get pointer from index
            ptr, size = self._index[idx]

            # Load data by pointer
            np_array = np.frombuffer(
                self._bin_buffer,
                dtype=self._index.dtype,
                count=size,
                offset=ptr,
            )
            return np_array

        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")

            # Get pointer from index
            ptr = self._index._pointers[start]
            sizes = self._index._sizes[idx]
            offsets = list(accumulate(sizes))
            total_size = sum(sizes)

            # Load data by pointer
            np_array = np.frombuffer(
                self._bin_buffer, dtype=self._index.dtype, count=total_size, offset=ptr
            )

            # Return sentences
            return np.split(np_array, offsets[:-1])

    def get(self, idx: int, offset: int = 0, length: int = None) -> np.ndarray:
        """
        Retrieves a single item from the dataset with
        the option to only return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.

        Args:
            idx (int): index
            offset (int): offset
            length (int): length of data

        Returns:
            np.ndarray: loaded data sample
        """

        # Get pointer from index
        ptr, size = self._index[idx]
        if length is None:
            length = size - offset

        ptr += offset * np.dtype(self._index.dtype).itemsize

        # Load data by pointer
        np_array = np.frombuffer(
            self._bin_buffer, dtype=self._index.dtype, count=length, offset=ptr
        )

        # Return single item
        return np_array

    @property
    def sizes(self):
        """
        Array of sizes of each data sample
        """
        return self._index.sizes

    @property
    def doc_idx(self):
        """Array of document indexes"""
        return self._index.doc_idx

    @property
    def supports_prefetch(self):
        """Whether support prefetch or not"""
        return False

    @staticmethod
    def exists(path: str):
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

    @staticmethod
    def builder(path) -> MMapIndexedDatasetBuilder:
        """
        Get build object to build mmap indexed dataset

        Args:
            path (str): output file path

        Returns:
            MMapIndexedDatasetBuilder: mmap indexed dataset builder object
        """
        return MMapIndexedDatasetBuilder(path)


class MmapIndex(object):
    """
    A class that handles mmap indexing

    Attributes:
        _HDR_MAGIC: a key to detect file format.
    """

    _HDR_MAGIC = b"MMIDIDX\x00\x00"

    @classmethod
    def writer(cls, path: str, dtype):
        """
        create index writer

        Args:
            path (str): file path

        Returns:
            _Writer: index writer
        """

        class _Writer(object):
            def __enter__(self):
                self._file = open(path, "wb")

                # 1. Write Magic string so we can check the file format then opening it again.
                self._file.write(cls._HDR_MAGIC)

                # 2. Write version number
                # Little endian unsigned 64 Bit integer
                self._file.write(struct.pack("<Q", 1))

                # 3. Little endian unsigned 8 Bit integer
                self._file.write(
                    struct.pack(
                        "<B",
                        NumpyUtils.cody_by_dtypes[dtype],
                    )
                )

                return self

            @staticmethod
            def _get_pointers(sizes: List[int]):
                """
                get addresses from data sizes list

                Args:
                    sizes (List[int]): sizes

                Notes:
                    examples of sizes:
                    ``sizes`` = [28 tokens, 43 tokens, 36 tokens, ...]

                Returns:
                    List[int]: list of address

                Examples:
                    >>> dtype = np.int8  # 8 bytes
                    >>> sizes = [28, 43, 36, ...]
                    >>> MmapIndex.writer("file_path.idx", dtype)._get_pointers(sizes)
                    [0*8, (0+28)*8, (0+28+43)*8, (0+28+43+36)*8, ...] = [0, 224, 568, 856, ...]
                """

                # start from 0
                address = 0
                dtype_size = dtype().itemsize
                pointers = []

                for size in sizes:
                    pointers.append(address)
                    address += size * dtype_size
                    # add (size * dtype_size) to previous address

                return pointers

            def write(self, sizes: np.array, doc_idx: np.array):
                """
                Write index file

                Args:
                    sizes (np.array): array of sizes of each data sample
                    doc_idx (np.array): array of document indexes
                """
                pointers = self._get_pointers(sizes)

                # 4. Write length of sizes array and document indexes
                self._file.write(struct.pack("<Q", len(sizes)))
                self._file.write(struct.pack("<Q", len(doc_idx)))

                # 5. Write sizes array
                sizes = np.array(sizes, dtype=np.int32)
                self._file.write(sizes.tobytes(order="C"))
                del sizes

                # 6. Write pointers array
                pointers = np.array(pointers, dtype=np.int64)
                self._file.write(pointers.tobytes(order="C"))
                del pointers

                # 7. Write document indexes array
                doc_idx = np.array(doc_idx, dtype=np.int64)
                self._file.write(doc_idx.tobytes(order="C"))

            def __exit__(self, exc_type, exc_val, exc_tb):
                """Close file"""
                self._file.close()

        return _Writer()

    @staticmethod
    def _warmup_mmap_file(path: str) -> None:
        """
        Read some bytes of page in the mapping to warm up the view.

        Args:
            path (str): file path
        """
        with open(path, "rb") as stream:
            while stream.read(100 * 1024 * 1024):
                pass

    def __init__(self, path: str):
        """
        Initialize index class

        Args:
            path (str): data path

        Notes:
              the format of mmap index (.idx) file:
              {
                magic (9, ????)

                struct (8){
                    version (8, unsigned long long)
                }

                struct (1){
                    dtype_code (1, unsigned char)
                }

                struct (8){
                    _len (8, unsigned long long)
                }

                struct (8){
                    _doc_count (8, unsigned long long)
                }
                ...
              }
        """
        with open(path, "rb") as stream:
            # 1. Read 9 bytes, magic string to detect dataset
            magic_test = stream.read(9)
            assert magic_test == self._HDR_MAGIC, (
                "Index file doesn't match expected format. "
                "Please check your configuration file."
            )

            version = struct.unpack("<Q", stream.read(8))
            assert (1,) == version

            # 2. Read 1 byte, dtype code
            (dtype_code,) = struct.unpack("<B", stream.read(1))
            self._dtype = NumpyUtils.dtypes_by_code[dtype_code]
            self._dtype_size = self._dtype().itemsize

            # 3. Read 8 byte, length of dataset
            self._len = struct.unpack("<Q", stream.read(8))[0]

            # 4. Read 8 byte, total count of dataset
            self._doc_count = struct.unpack("<Q", stream.read(8))[0]
            offset = stream.tell()

        # 5. Create mmap array and memoryview
        self._bin_buffer_mmap = np.memmap(path, mode="r", order="C")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

        # 6. Read sizes array from buffer
        self._sizes = np.frombuffer(
            self._bin_buffer, dtype=np.int32, count=self._len, offset=offset
        )

        # 7. Read pointers from buffer
        self._pointers = np.frombuffer(
            self._bin_buffer,
            dtype=np.int64,
            count=self._len,
            offset=offset + self._sizes.nbytes,
        )

        # 8. Read document indexes from buffer
        self._doc_idx = np.frombuffer(
            self._bin_buffer,
            dtype=np.int64,
            count=self._doc_count,
            offset=offset + self._sizes.nbytes + self._pointers.nbytes,
        )

    @property
    def dtype(self):
        """
        type of token
        """
        return self._dtype

    @property
    def sizes(self):
        """
        Array of sizes of each data sample
        """
        return self._sizes

    @property
    def doc_idx(self):
        """Array of document indexes"""
        return self._doc_idx

    @lru_cache(maxsize=8)
    def __getitem__(self, idx: Union[int, slice]) -> Tuple[int, int]:
        """
        Get pointer and size of sample by given index

        Args:
            idx (Union[int, slice]): index

        Returns:
            Tuple[int, int]: pointer and size of sample
        """
        return self._pointers[idx], self._sizes[idx]

    def __len__(self) -> int:
        """
        Total length of dataset

        Returns:
            int: length of total dataset
        """
        return self._len
