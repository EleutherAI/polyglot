# Polyglot: Large Language Models of Well-balanced Competence in Multi-languages

## 1. Introduction
### Why another multilingual model?
Various multilingual models such as [mBERT](https://huggingface.co/bert-base-multilingual-cased), [BLOOM](https://huggingface.co/bigscience/bloom), and [XGLM](https://arxiv.org/abs/2112.10668) have been released.
Therefore, someone might ask, "why do we need to make multilingual models again?" Before answering the question, we would like to ask, "Why do people around the world make monolingual models in their language even though there are already many multilingual models?" We would like to point out there is a dissatisfaction with the non-English language performance of the current multilingual models as one of the most significant reason. So we want to make multilingual models with higher non-English language performance. This is the reason we need to make multilingual models again and why we name them ['Polyglot'](https://www.spanish.academy/blog/what-is-the-difference-between-a-polyglot-and-a-multilingual-person/).

### What do we focus on to make better multilingual models?
We will focus on the following two factors to make multilingual models which show better non-English performance.

#### Amount of data in each language and its balance
Most multilingual models are trained using data from fairly uneven distribution of languages. For example, BLOOM's training data is still English-centric. English data takes 30% of the data, however some languages such as Vietnamese and Indonesian are only 1-3% of data. XGLM has taken a step forward for mitigating this problem by data up-sampling, but we believe there is a limitation of data up-sampling. To resolve this problem, we will collect a large multilingual dataset with hundreds of billions of tokens per language and balance them so that the model can learn various languages in balance.

#### Language selection

Most multilingual models learned dozens of languages, including low-resource languages. For example, XGLM learned 30 languages, and BLOOM learned 42 languages. However, we plan to let go of the desire to be good at too many languages at once. The number of steps a model can learn is somewhat set, and the model converges when it exceeds that. So if one model takes too many languages, the training efficiency for each language decreases. Therefore, we want to train the model with languages in similar language families which enable synergy effect between them. In addition, we have excluded languages used by a few users use because it is difficult to collect a large amount of data. Therefore, we will only focus on high or middle-resource languages in our project.

## 2. Projects

### Polyglot-Ko [WIP]
When we started our research, we have already had 1.2TB of Korean data collected by [TUNiB](https://tunib.ai/). Before we collected a large amount of multilingual data, we decided to try Korean modeling with the dataset we already had. This Korean model can be used for performance comparison with the multilingual models, and this model itself would help many Korean companies and researchers.

| Size |                                      Training Status                                       |                           Model Card                            |                             Model Checkpoints                             |
|:----:|:------------------------------------------------------------------------------------------:|:---------------------------------------------------------------:|:-------------------------------------------------------------------------:|
| 1.3B | [Finished](https://wandb.ai/eleutherai-oslo/gpt-neox-ko-1b?workspace=user-eleutherai-oslo) | [Available](https://huggingface.co/EleutherAI/polyglot-ko-1.3b) | [Available](https://huggingface.co/EleutherAI/polyglot-ko-1.3b/tree/main) |
| 2.7B | [Finished](https://wandb.ai/eleutherai-oslo/gpt-neox-ko-3b?workspace=user-eleutherai-oslo) |                           Coming soon                           |                                Coming soon                                |
| 6.7B | [Training](https://wandb.ai/eleutherai-oslo/gpt-neox-ko-6b?workspace=user-eleutherai-oslo) |                           Coming soon                           |                                Coming soon                                |
| ...  |                                           Ready                                            |                           Coming soon                           |                                Coming soon                                |

### Polyglot-East-Asian [WIP]
We chose the East Asian language as our first multilingual dataset.
This model includes Korean, Chinese, Japanese, Indonesian, Malay, Vietnamese, Thai, and English.
We will train the model by collecting at least hundreds of billions tokens of data from each language and balancing them.
Some people may wonder why English is included on this list, but because English is now a global language, we believe it could synergize with any other language in the world.

| Size | Training Status | Model Card  | Model Checkpoints |
|:----:|:---------------:|:-----------:|:-----------------:|
| ...  |      Ready      | Coming soon |    Coming soon    |


## 3. Data Risks

Polyglot models learn an inner representation of the various languages that can be used to extract features useful for downstream tasks.
The model is best at what it was pre-trained for, however, generating text from a prompt.

### Privacy considerations
General training algorithms for pre-trained language models have many hazards, that memorize personal information in training data. We added the following tokens to vocabulary to mitigate privacy problems and replaced much personal information with these tokens in data preprocessing steps.

* `<|acc|>` : bank account number
* `<|rrn|>` : resident registration number
* `<|tell|>` : phone number

### Limitations and Biases
The core functionality of Polyglot is taking a string of text and predicting the next token. While language models are widely used for tasks other than this, there are a lot of unknowns with this work. When prompting Polyglot it is important to remember that the statistically most likely next token is often not the token that produces the most "accurate" text. Never depend upon Polyglot to produce factually accurate output. Depending upon the use case, Polyglot may produce socially unacceptable text.

As with all language models, it is hard to predict in advance how Polyglot will respond to particular prompts, and offensive content may occur without warning. We recommend having a human curate or filter the outputs before releasing them, both to censor undesirable content and to improve the quality of the results.

### Legal Restrictions
Since there are laws in many countries related to data collection, we will collect data with due regard to the laws of those countries.
Additionally, we plan to use the dataset to train our models, but we do not plan to make the dataset publicly available.

## 4. Citation and Related Information
### BibTeX entry
If you find our work useful, please consider citing:
```bibtex
@misc{polyglot-ko,
  title = {{Polyglot-Ko: Open-Source Korean Autoregressive Language Model}},
  author = {Ko, Hyunwoong and Yang, Kichang and Ryu, Minho and Kim, Taekyun and Yang, Seungmu and Hyun, jiwung and Park, Sungho},
  url = {https://www.github.com/eleutherai/polyglot},
  month = {9},
  year = {2022},
}
```

### Licensing
All our models are licensed under the terms of the Apache License 2.0.

```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

However, the model has the potential to generate unpredictable text as mentioned. Therefore, we are not responsible for any damages resulting from the use of the model.

### Acknowledgement

This project would not have been possible without the computing resources provided by [Stability.ai](https://stability.ai). Thanks for providing a large amount of GPU resources. Furthermore, thanks to [TUNiB](https://tunib.ai) for providing a large-scale Korean dataset for this work.
