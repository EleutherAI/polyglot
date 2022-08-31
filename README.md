# multilingual
This repository records various multilingual related studies conducted by EleutherAI. 
Starting with GPT-3, researches about large-scale language model has been advanced so far, but most of it has been done in English.
There are many languages other than English in the world, so we wanted to make sure that non-English-speaking countries can also benefit from these researches.

Our ultimate goal is making open source language models trained with various languages. To this end, we plan to perform various experiments such as multilingual language transferring, non-English monolingual language model training and multilingual language model training.
If you have any questions or would like to participate in our research, [join our Discord](https://discord.com/invite/zBGx3azzUn).

## 1. GPT-NeoX-Ko
When we first started our research, most of our members were Korean, so we first targeted the Korean language. 
We collected about 1 TB of Korean dataset for this work, which was done with [TUNiB](https://tunib.ai/). 
In addition, we used the [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) codebase for model training and added 8 Korean tasks to [LM-Evaluation-Harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/multilingual-ko) for model evaluation.

| Size | Status |   Evaluation   |  Checkpoints   |
|:----:|:------:|:--------------:|:--------------:|
| 1.3B | Ready  | Coming soon... | Coming soon... |

## Citation
If you have found our words helpful in your work, you can cite this repository as
```
@misc{gpt-neox-ko,
  title = {{GPT-NeoX-Ko: Open-Source Korean Autoregressive Language Model}},
  author = {Hyunwoong, Ko and ...},
  url = {https://www.github.com/eleutherai/multilingual},
  month = {9},
  year = {2022},
}
```