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

import logging
from concurrent.futures.process import ProcessPoolExecutor

from transformers import PreTrainedTokenizer

from multilingual.data.preprocess.binarizer import DatasetBinarizer
from multilingual.data.preprocess.encoder import DatasetEncoder

logger = logging.getLogger(__name__)


class DatasetPreprocessor(object):
    """
    Dataset preprocessor (tokenizing + encoding + binarizing)
    This code is copied from Megatron-LM.

    Args:
        tokenizer (PreTrainedTokenizer): huggingface tokenizer object
        chunksize (int): chunk size for multiprocessing
        binarization_impl (str): type of binarization implementation. one of ['mmap', 'lazy', 'cached'].
        append_eod (bool): flag indication whether to apply append end of document token or not
        eod_token_id (int): id of end of document token.

    Notes:
        What is end of document token?
            For example, ``sentence_1`` is 'Hello I am Kevin' and ``sentence_2`` is 'I am Ryan'.
            Then, we concatenate all the sentences like 'Hello I am Kevin I am Ryan' to remove all the padding tokens.
            However, if we concatenate multiple sentences, the distinction between documents disappears.
            So, we append eod token to end of each document like 'Hello I am Kevin <eod> I am Ryan <eod>'.
            In general, models like GPT uses end of sentence (eos) token, and models like BERT uses seperator (SEP) token.

        What is binarization impl?
            This is how data is loaded. If we set ``binarization_impl`` to 'cached' it will load all dataset into memory.
            This is suitable for fine-tuning with relatively smaller dataset size than pre-training.
            But, if the size of the data is very large, we must load only the required amount of data at each step.
            If we set ``binarization_impl`` to 'lazy', we loads required amount of data from disk at each step.
            However, speed of disk is very slower than CPU and GPU. Therefore, we also support ``mmap`` method.
            if we set ``binarization_impl`` to 'mmap', it uses memory mapping. With the memory mapping data address is mapped to the virtual memory.
            Then, we can manipulate the data on the disk as if it exists in memory without disk I/O.

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained(...)
        >>> preprocessor = DatasetPreprocessor(tokenizer)
        >>> preprocessor.preprocess(data_paths)

    References:
        https://github.com/NVIDIA/Megatron-LM/blob/b31e1296354e979722627a6c4dedafe19b51fa97/tools/preprocess_data.py
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        chunksize: int = 1024,
        binarization_impl: str = "mmap",
        append_eod: bool = True,
        eod_token_id: int = None,
    ):
        assert binarization_impl in ["lazy", "mmap", "cached"], (
            "Param ``binarization_impl`` must be one of 'lazy', 'mmap' and 'cached'. "
            "For description of each method, please refer to the docstring."
        )

        self.tokenizer = tokenizer
        self.chunksize = chunksize
        self.binarization_impl = binarization_impl
        self.append_eod = append_eod

        if append_eod:
            assert append_eod is not None, (
                "You set ``append_eod=True`` but missing ``eod_token_id``."
                "Please input ``eod_token_ids` together."
            )
            self.eod_token_id = eod_token_id
        else:
            self.eod_token_id = eod_token_id

    def preprocess(
        self,
        iterable,
        save_file_name: str,
        log_interval: int = 1000,
    ) -> None:
        """
        Preprocess a dataset

        Args:
            iterable: iterable of string
            save_file_name (str): save file name
            log_interval (int) logging interval
        """

        encoder = DatasetEncoder(
            tokenizer=self.tokenizer,
            append_eod=self.append_eod,
            eod_token_id=self.eod_token_id,
        )
        binarizer = DatasetBinarizer(self.binarization_impl)
        index_path, builder = binarizer.create_builder(save_file_name)

        with ProcessPoolExecutor() as pool:
            iterator = pool.map(
                encoder.encode,
                iterable,
                chunksize=self.chunksize,
            )

            binarizer.binarize(
                iterator=iterator,
                builder=builder,
                index_path=index_path,
                log_interval=log_interval,
            )

    @staticmethod
    def open_jsonl(file, json_key):
        """
        Open jsonl file similar with Megatron-LM data format

        Examples:
            1 {'text': 'blah blah blah ...'}
            2 {'text': 'blah blah blah ...'}
            3 {'text': 'blah blah blah ...'}
            4 ...

            >>> DatasetPreprocessor.open_jsonl(
            ...     file=FILE_NAME, json_key='text'
            ... )
        """

        import json

        if file[-6:].lower() != ".jsonl":
            file = file + ".jsonl"

        source = open(file)

        while True:
            line = source.readline()

            if not line:
                break
            else:
                yield json.loads(line)[json_key]
