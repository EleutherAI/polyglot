import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils import cpp_extension

DEFAULT_TORCH_EXTENSION_PATH = os.path.join(
    os.path.expanduser("~"),
    ".cache",
    "torch_extensions",
    "multilingual",
)

_DATASETS_BUILDER = None
_DATASETS_BUILDER_COMPILING_SUCCESS = None


class Binder(object):
    def __init__(self):
        self.compat = self.get_compatibility_version()

    @property
    def base_path(self):
        from multilingual import csrc

        return Path(csrc.__file__).parent.absolute()

    @property
    def name(self):
        return "multilingual"

    def includes(self):
        return [
            os.path.join(self.base_path, "includes"),
        ]

    def sources(self):
        return []

    @staticmethod
    def get_compatibility_version():
        a, b = torch.cuda.get_device_capability(torch.cuda.current_device())
        return int(str(a) + str(b))

    def bind(self):
        try:
            import ninja
            import pybind11

        except ImportError:
            raise ImportError(
                "Unable to compile C++ code due to ``ninja`` or ``pybind11`` not being installed. "
                "please install them using ``pip install ninja pybind11``."
            )

        # Ensure directory exists to prevent race condition in some cases
        ext_path = os.environ.get("TORCH_EXTENSIONS_DIR", DEFAULT_TORCH_EXTENSION_PATH)
        ext_path = os.path.join(ext_path, self.name)
        os.makedirs(ext_path, exist_ok=True)

        op_module = cpp_extension.load(
            name=self.name,
            sources=[os.path.join(self.base_path, path) for path in self.sources()],
            extra_include_paths=self.includes(),
            extra_cflags=self.cxx_args(),
            extra_cuda_cflags=self.nvcc_args(),
            verbose=False,
        )

        return op_module

    @staticmethod
    def cxx_args():
        if sys.platform == "win32":
            return [
                "-O2",
                "-Wno-reorder",
                "-Wno-deprecated",
                "-Wno-deprecated-declarations",
            ]
        else:
            return [
                "-O3",
                "-std=c++14",
                "-g",
                "-Wno-reorder",
                "-Wno-deprecated",
                "-Wno-deprecated-declarations",
            ]

    def nvcc_args(self, maxrregcount: int = None):
        nvcc_flags = [
            "-O3",
            "--use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
        ]

        additional_flags = [
            "-gencode",
            f"arch=compute_{self.compat},code=sm_{self.compat}",
        ]

        if maxrregcount:
            additional_flags.append(f"-maxrregcount={maxrregcount}")

        return nvcc_flags + additional_flags


class DatasetBinder(Binder):
    @property
    def name(self):
        return "datasets"

    def sources(self):
        return ["datasets.cpp"]


def get_datasets_builder():
    global _DATASETS_BUILDER, _DATASETS_BUILDER_COMPILING_SUCCESS

    if _DATASETS_BUILDER_COMPILING_SUCCESS is None:
        try:
            _DATASETS_BUILDER = DatasetBinder().bind()
            _DATASETS_BUILDER_COMPILING_SUCCESS = True
            return get_datasets_builder()
        except Exception as e:
            print(
                "Failed to launch C++ dataset builder... using slower python version. "
                f"Error message: {e}"
            )
            _DATASETS_BUILDER_COMPILING_SUCCESS = False
            return get_datasets_builder()

    elif _DATASETS_BUILDER_COMPILING_SUCCESS is True:
        assert _DATASETS_BUILDER is not None, "C++ dataset builder must not be None."
        return _DATASETS_BUILDER

    else:
        return None


def optimized_params(model, weight_decay):
    no_decay = ["bias", "LayerNorm.weight"]

    return [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
