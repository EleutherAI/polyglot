# coding=utf-8
# Copyright 2021 TUNiB Inc.
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

from functools import lru_cache


class DatasetEncoder(object):
    """
    Dataset encoder for preprocessing.

    Args:
        tokenizer (TokenizationBase): Huggingface Tokenizers object
        append_eod (bool): whether append end of document token
        eod_token_id (int): identification of end of document token
    """

    def __init__(
        self,
        tokenizer,
        append_eod: bool,
        eod_token_id: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.append_eod = append_eod
        self.eod_token_id = eod_token_id

    @lru_cache(maxsize=1_000_000)
    def encode(self, text: str):
        """Encode input text to list of tokens"""

        # sentence_ids is 1D array
        sentence_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        if len(sentence_ids) > 0 and self.append_eod:
            sentence_ids.append(self.eod_token_id)

        return sentence_ids, len(sentence_ids)
