import time
import torch
from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast

import torch

import os

os.environ["WANDB_DISABLED"] = "true"

tokenizer = PreTrainedTokenizerFast.from_pretrained("hyunwoongko/kobart")
tokenizer.add_special_tokens({'additional_special_tokens':['<|endoftext|>']})

import pickle
from typing import Dict, List, Optional

from torch.utils.data.dataset import Dataset

from filelock import FileLock

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)

class TextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        overwrite_cache=False,
        cache_dir: Optional[str] = None,
    ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        # block_size = block_size - tokenizer.num_special_tokens_to_add(is_pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else directory,
            "cached_lm_{}_{}_{}".format(
                tokenizer.__class__.__name__,
                str(block_size),
                filename,
            ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")
                self.examples = []
                text = ""
                line_idx = 0
                total_length = 3748585 # wiki sentence size: 3748585
                with open(file_path, encoding="utf-8") as f:
                    single_example = []
                    while True:
                        if len(single_example) > block_size:
                            splited_examples = [single_example[i:i + block_size] for i in range(0, len(single_example), block_size)]
                            for splited_example in splited_examples:
                                self.examples.append(
                                        splited_example
                                )
                            single_example = []
                        if line_idx % 1000 == 0:
                            print(line_idx, '/', total_length, '(', line_idx / total_length * 100, '% )', '-', len(self.examples))
                        line_idx += 1
                        line = f.readline()
                        if line:
                            line = line.strip()
                            if len(line) < 1:
                                line = "<|endoftext|>"
                            tokenized_text = tokenizer.encode(line)
                            single_example.extend(tokenized_text)
                        else:
                            break
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should look for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)

dataset = TextDataset(
    tokenizer=tokenizer,
    file_path='/home/corpus/wiki_20190620.txt',
    block_size=512
)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='model_output',
    report_to=None,
    overwrite_output_dir=True,
    num_train_epochs=4,
    per_device_train_batch_size=16,  # xl 8
    save_steps=1000,    # ko; 1000
    save_total_limit=2,
    logging_steps=50,
)

model = GPT2LMHeadModel.from_pretrained("gpt2-medium", cache_dir="gpt2_model_cache")

for h in model.transformer.h:
    for p in h.parameters():
        p.requires_grad = False 

from torch import nn

torch.manual_seed(1)

### make new embedding layer!
make_embedding_layer = True
if make_embedding_layer:    
    my_input_embedding_layer = nn.Embedding(tokenizer.vocab_size+1, 1024)    # gpt-j = 50400/4096, gpt-xl: 1600, gpt-medium: 50257/1024
    my_input_embedding_layer.weight.data.uniform_(-1, 1)

    model.set_input_embeddings(my_input_embedding_layer)
    model.resize_token_embeddings(tokenizer.vocab_size+1)

model.transformer.wte.weight.requires_grad = True

print('load finish')

print('model params:', model.num_parameters())
print('trainable model params:', model.num_parameters(True))
print('trainable model params w/o embedding params:', model.num_parameters(True, True))

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

trainer.train()
trainer.save_model()