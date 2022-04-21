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

from typing import List, Optional

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from multilingual.utils import get_datasets_builder


class DatasetBlender(Dataset):
    """
    Dataset blender for multiple datasets

    this is copied from Megatron-LM and modified.

    Args:
        datasets (List[Dataset]): list of datasets
        weights (Optional[List[float]]): list of dataset weights.
            if None, we make weight lists proportional to the length of each dataset automatically.
    """

    def __init__(self, datasets: List[Dataset], weights: Optional[List[float]] = None):
        super(DatasetBlender, self).__init__()

        self.datasets = datasets
        num_datasets = len(datasets)

        lengths = []
        for dataset in self.datasets:
            lengths.append(len(dataset))

        self.size = sum(lengths)

        if weights is None:
            # make weight automatically
            weights = [length / self.size for length in lengths]

        # Normalize weights.
        assert num_datasets == len(weights)
        weights = self._normalize_weight(weights)

        # Build indices.
        self.dataset_index = np.zeros(self.size, dtype=np.int32)
        self.dataset_sample_index = np.zeros(self.size, dtype=np.int64)
        cpp_builder = get_datasets_builder()

        if cpp_builder is not None:
            # Using C++ implementation for high speed.
            cpp_builder.build_blending_indices(
                self.dataset_index,
                self.dataset_sample_index,
                weights,
                num_datasets,
                self.size,
            )
        else:
            self._build_blending_indices(
                self.dataset_index,
                self.dataset_sample_index,
                weights,
                num_datasets,
                self.size,
            )

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        dataset_idx = self.dataset_index[idx]
        dataset_len = max(len(self.datasets[dataset_idx]), 1)
        sample_idx = self.dataset_sample_index[idx]
        return self.datasets[dataset_idx][sample_idx % dataset_len]

    @staticmethod
    def _build_blending_indices(
        dataset_index,
        dataset_sample_index,
        weights,
        num_datasets,
        size,
    ):
        """Python implementation of ``build_blending_indices``"""
        current_samples = [0] * num_datasets

        for sample_idx in tqdm(range(size)):
            sample_idx_double = max(sample_idx, 1.0)
            max_error_index = 0
            max_error = weights[0] * sample_idx_double - current_samples[0]

            for dataset_idx in range(num_datasets):
                error = (
                    weights[dataset_idx] * sample_idx_double
                    - current_samples[dataset_idx]
                )
                if error > max_error:
                    max_error = error
                    max_error_index = dataset_idx

            dataset_index[sample_idx] = max_error_index
            dataset_sample_index[sample_idx] = current_samples[max_error_index]
            current_samples[max_error_index] += 1

    @staticmethod
    def _normalize_weight(weights: List[float]) -> np.ndarray:
        """
        Normalize dataset weights into 0 to 1.

        Args:
            weights (List[float]): list of dataset weights

        Returns:
            np.ndarray: list of normalized dataset weights
        """
        weights = np.array(weights, dtype=np.float64)
        sum_weights = np.sum(weights)
        assert sum_weights > 0.0, (
            "sum of all the weights is zero. "
            "did you input zero length list of dataset?"
        )

        return weights / sum_weights
