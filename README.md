# multilingual
This repository contains various research output on multilingual language models which have been conducted by EleutherAI. Current large language model research such as GPT-3 have been mainly done in English therefore non-English speaking countries can not get a benefit from these researches directly. We would like to apply this knowledge to non English language models for users with languages other than English. Our ultimate goal is making language models learn other languages more than English as well. We are planning to perform various language-related experiments such as multilingual language transferring, non-English monolingual training and multilingual training, and we will be updating them here. If you have any questions or would like to participate in our research, please [join our Discord](https://discord.com/invite/zBGx3azzUn).

## 1. GPT-NeoX-Ko [WIP]
### 1.1. Introduction
We first targeted the Korean language because most of our contributors were Korean when we first started our research. We collected about 1.2TB of Korean dataset for this work, which was done with [TUNiB](https://tunib.ai/). In addition, we used the [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) framework for model training and added 8 Korean tasks to [LM-Evaluation-Harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/multilingual-ko) for model evaluation.

### 1.2. Models
| Size |                                           Status                                           | Evaluation  | Checkpoints |
|:----:|:------------------------------------------------------------------------------------------:|:-----------:|:-----------:|
| 1.3B | [Training](https://wandb.ai/eleutherai-oslo/gpt-neox-ko-1b?workspace=user-eleutherai-oslo) | Coming soon | Coming soon |
| 2.7B | [Training](https://wandb.ai/eleutherai-oslo/gpt-neox-ko-3b?workspace=user-eleutherai-oslo) | Coming soon | Coming soon |

### 1.3. Limitations and Biases
GPT-NeoX-Ko was trained as an autoregressive language model. This means that its core functionality is taking a string of text and predicting the next token. While language models are widely used for tasks other than this, there are a lot of unknowns with this work. Depending on your usecase GPT-NeoX-Ko may produce socially unacceptable text. As with all language models, it is hard to predict in advance how GPT-NeoX-Ko will respond to particular prompts and offensive content may occur without warning. We recommend having a human curate or filter the outputs before releasing them, both to censor undesirable content and to improve the quality of the results.


## Citation
If you have found our words helpful in your work, you can cite this repository as
```
@misc{gpt-neox-ko,
  title = {{GPT-NeoX-Ko: Open-Source Korean Autoregressive Language Model}},
  author = {Hyunwoong, Ko and Kichang, Yang and Minho, Ryu and Taekyun, Kim, ...},
  url = {https://www.github.com/eleutherai/multilingual},
  month = {9},
  year = {2022},
}
```