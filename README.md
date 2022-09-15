# multilingual
This repository contains various research outputs on multilingual language models which have been conducted by EleutherAI. Current large-scale language model researches such as GPT-3 have been mainly done in English, and therefore it is hard to benefit from these researches directly in non-English speaking countries. We would like to apply this knowledge to non-English language models for users with languages other than English. Our ultimate goal is to make large-scale language models for low-resource languages, rather than just focusing on high-resource languages. We are planning to perform various language-related experiments such as multilingual language transferring, non-English monolingual training, and multilingual training for this, and we will be updating them here. If you have any questions or would like to participate in our research, please [join our Discord](https://discord.com/invite/zBGx3azzUn).

## 1. GPT-NeoX-Ko [WIP]
### Model Description
GPT-NeoX-Ko is a Korean autoregressive language model made by EleutherAI multilingual team. We collected about 1.2TB Korean dataset for this work, which was done with [TUNiB](https://tunib.ai/). In addition, we used the GPT-NeoX framework for model training and added some Korean tasks to LM-Evaluation-Harness for model evaluation.

### Model Checkpoints
| Size |                                           Status                                            | Evaluation  |                                                      Checkpoints                                                      |
|:----:|:-------------------------------------------------------------------------------------------:|:-----------:|:---------------------------------------------------------------------------------------------------------------------:|
| 1.3B | [Published](https://wandb.ai/eleutherai-oslo/gpt-neox-ko-1b?workspace=user-eleutherai-oslo) | Coming soon | [Available](https://huggingface.co/[EleutherAI/gpt-neox-ko-1.3b](https://huggingface.co/EleutherAI/gpt-neox-ko-1.3b)) |
| 2.7B | [Training](https://wandb.ai/eleutherai-oslo/gpt-neox-ko-3b?workspace=user-eleutherai-oslo)  | Coming soon |                                                      Coming soon                                                      |

### Training data
GPT-NeoX-Ko was trained on 1.2TB Korean Dataset, a large-scale curated dataset created by [TUNiB](https://tunib.ai/).

### Training procedure
GPT-NeoX-Ko was trained for 213 billion tokens over 102,000 steps on 256 * A100 GPUs. It was trained as an autoregressive language model, using cross-entropy loss to maximize the likelihood of predicting the next token correctly.

### How to use

This model can be easily loaded using the `AutoModelForCausalLM` functionality:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("[EleutherAI/gpt-neox-ko-1.3b](https://huggingface.co/EleutherAI/gpt-neox-ko-1.3b)")
model = AutoModelForCausalLM.from_pretrained("[EleutherAI/gpt-neox-ko-1.3b](https://huggingface.co/EleutherAI/gpt-neox-ko-1.3b)")
```

### Privacy considerations and Limitations

GPT-NeoX-Ko learns an inner representation of the Korean that can be used to extract features useful for downstream tasks. The model is best at what it was pretrained for however, which is generating text from a prompt.

#### Privacy considerations
General training algorithms for pretrained language model have many hazards that memorize personal information in training data. We added the following tokens to vocabulary to mitigate privacy problem and replaced much personal information to these tokens in data preprocessing steps.

* `<|acc|>` : bank account number
* `<|rrn|>` : resident registration number
* `<|tell|>` : phone number

#### Limitations and Biases

The core functionality of GPT-NeoX-Ko is taking a string of text and predicting the next token. While language models are widely used for tasks other than this, there are a lot of unknowns with this work. When prompting GPT-NeoX-Ko it is important to remember that the statistically most likely next token is often not the token that produces the most "accurate" text. Never depend upon GPT-NeoX-Ko to produce factually accurate output.Depending upon use case GPT-NeoX-Ko may produce socially unacceptable text.

As with all language models, it is hard to predict in advance how GPT-NeoX-Ko will respond to particular prompts and offensive content may occur without warning. We recommend having a human curate or filter the outputs before releasing them, both to censor undesirable content and to improve the quality of the results.

### Citation

If you find our work useful, please consider citing:

```bibtex
@misc{gpt-neox-ko,
  title = {{GPT-NeoX-Ko: Open-Source Korean Autoregressive Language Model}},
  author = {Ko, Hyunwoong and Yang, Kichang and Ryu, Minho and Kim, Taekyun and Yang, Seungmu and Hyun, Jiwoong and Park, Sungho and Ryu, Myunghyun and Keum, Bitna and Oh, Saechan and Kim, Soohwan and Park, Kyubyong},
  url = {https://www.github.com/eleutherai/multilingual},
  month = {9},
  year = {2022},
}
```

### Licensing
GPT-NeoX-Ko project is licensed under the terms of the Apache License 2.0.

Copyright © 2022, EleutherAI. All Rights Reserved.
