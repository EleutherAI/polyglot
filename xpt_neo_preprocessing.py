import json
import os
from argparse import ArgumentParser

from transformers import AutoTokenizer

from multilingual.data.preprocess.preprocessor import DatasetPreprocessor

parser = ArgumentParser()
parser.add_argument("--config", "-c", type=str, required=True)
config = json.load(open(parser.parse_args().config))

tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
os.environ["TOKENIZERS_PARALLELISM"] = "true"

preprocessor = DatasetPreprocessor(
    tokenizer=tokenizer,
    binarization_impl="mmap",
    eod_token_id=tokenizer.eos_token_id,
    append_eod=True,
)

for dataset in config["datasets"]:
    preprocessor.preprocess(
        open(dataset["name"] + ".txt"),
        save_file_name=dataset["name"],
        log_interval=100,
    )
