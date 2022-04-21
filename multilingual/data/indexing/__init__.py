# coding=utf-8
# Copyright 2021 TUNiB Inc.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE.apache-2.0 file in the root directory of this source tree.

from typing import Any, BinaryIO, Union

import numpy as np


class PathUtils(object):
    """
    Path making utils
    """

    @staticmethod
    def idx_path(path: str) -> str:
        """
        Make index file path (.idx)

        Args:
            path (str): index file path without extension

        Returns:
            str: index file path with extension
        """
        return path + ".idx"

    @staticmethod
    def data_path(path: str) -> str:
        """
        Make data file path (.bin)

        Args:
            path (str): data file path without extension

        Returns:
            str: data file path with extension
        """
        return path + ".bin"


class NumpyUtils(object):
    """
    Numpy data type converting utils
    """

    dtypes_by_code = {
        1: np.uint8,
        2: np.int8,
        3: np.int16,
        4: np.int32,
        5: np.int64,
        6: np.float,
        7: np.double,
        8: np.uint16,
    }

    cody_by_dtypes = {
        np.uint8: 1,
        np.int8: 2,
        np.int16: 3,
        np.int32: 4,
        np.int64: 5,
        np.float: 6,
        np.double: 7,
        np.uint16: 8,
    }
    element_sizes = {
        np.uint8: 1,
        np.int8: 1,
        np.int16: 2,
        np.int32: 4,
        np.int64: 8,
        np.float: 4,
        np.double: 8,
    }

    @staticmethod
    def create_array_from_file(file: BinaryIO, size: Union[int, tuple]) -> np.ndarray:
        """
        Create a numpy array from given binary file.

        Args:
            file (BinaryIO): binary file descriptor
            size (Union[int, tuple]: binary file sizes

        Returns:
            np.ndarray: numpy array that contains data from file.
        """
        # 1. create empty numpy array
        array = np.empty(size, dtype=np.int64)

        # 2. insert data of file into empty array
        file.readinto(array)
        return array

    @staticmethod
    def write_file_from_array(file: BinaryIO, array: Any) -> None:
        """
        Write a file from given array

        Args:
            file (BinaryIO): binary file descriptor
            array (Any): list that want to save
        """
        file.write(np.array(array, dtype=np.int64))
