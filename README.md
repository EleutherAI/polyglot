# multilingual
This repository contains various research outputs on multilingual language models which have been conducted by EleutherAI. Current large-scale language model researches such as GPT-3 have been mainly done in English, and therefore it is hard to benefit from these researches directly in non-English speaking countries. We would like to apply this knowledge to non-English language models for users with languages other than English. Our ultimate goal is to make large-scale language models for low-resource languages, rather than just focusing on high-resource languages. We are planning to perform various language-related experiments such as multilingual language transferring, non-English monolingual training, and multilingual training for this, and we will be updating them here. If you have any questions or would like to participate in our research, please [join our Discord](https://discord.com/invite/zBGx3azzUn).

## 1. GPT-NeoX-Ko [WIP]
### 1.1. Introduction
We firstly targeted Korean language because most of our contributors were Korean when we started our research. We collected about 1.2TB Korean dataset for this work, which was done with TUNiB. In addition, we used the [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) framework for model training and added 8 Korean tasks to [LM-Evaluation-Harness](https://github.com/EleutherAI/lm-evaluation-harness) for model evaluation.

### 1.2. Models
| Size |                                           Status                                           | Evaluation  | Checkpoints |
|:----:|:------------------------------------------------------------------------------------------:|:-----------:|:-----------:|
| 1.3B | [Training](https://wandb.ai/eleutherai-oslo/gpt-neox-ko-1b?workspace=user-eleutherai-oslo) | Coming soon | Coming soon |
| 2.7B | [Training](https://wandb.ai/eleutherai-oslo/gpt-neox-ko-3b?workspace=user-eleutherai-oslo) | Coming soon | Coming soon |

### 1.3. Privacy considerations
General training algorithms for pretrained language model have many hazards that memorize personal information in training data. 
We added these three follow tokens to mitigate this problem

* `<|acc|>` : bank account number. 
* `<rnn>` : SSN(Social Security Number). 
* `<|tell|>` : phone number.

We replaced much personal information by using these tokens in data preprocessing steps.

### 1.4. Limitations and Biases
GPT-NeoX-Ko was trained as an autoregressive language model. This means that its core functionality is taking a string of text and predicting the next token. While language models are widely used for tasks other than this, there are a lot of unknowns with this work. Depending on your use case, GPT-NeoX-Ko may produce socially unacceptable text. As with all language models, it is hard to predict in advance how GPT-NeoX-Ko will respond to particular prompts, and offensive content may occur without warning. We recommend having a human curate or filter the outputs before releasing them, both to censor undesirable content and to improve the quality of the results.

### 1.5. Licensing
GPT-NeoX-Ko project is licensed under the terms of the Apache License 2.0.

Copyright © 2022, EleutherAI. All Rights Reserved.

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
