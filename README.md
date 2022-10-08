# Polyglot: Large Language Models of Well-balanced Competence in Multi-languages

## 1. Introduction

### Why another multilingual model?
Various multilingual models such as [mBERT](https://huggingface.co/bert-base-multilingual-cased), [BLOOM](https://huggingface.co/bigscience/bloom), and [XGLM](https://arxiv.org/abs/2112.10668) have been released.
Therefore, someone might ask, "why do we need to make multilingual models again?" Before answering the question, we would like to ask, "Why do people around the world make monolingual models in their language even though there are already many multilingual models?" We would like to point out there is a dissatisfaction with the non-English language performance of the current multilingual models as one of the most significant reason. So we want to make multilingual models with higher non-English language performance. This is the reason we need to make multilingual models again and why we name them ['Polyglot'](https://www.spanish.academy/blog/what-is-the-difference-between-a-polyglot-and-a-multilingual-person/).

### What do we focus on to make better multilingual models?
We will focus on the following two factors to make multilingual models which show better non-English performance.

#### Amount of data in each language and its balance
Most multilingual models are trained using data from fairly uneven distribution of languages. For example, BLOOM's training data is still English-centric. English data takes 30% of the data, however some languages such as Vietnamese and Indonesian are only 1-3% of data. XGLM has taken a step forward for mitigating this problem by data up-sampling, but we believe there is a limitation of data up-sampling. To resolve this problem, we will collect a large multilingual dataset with hundreds billions tokens per language and balance them so that the model can learn various languages in balance.

#### Language selection
Most multilingual models learned dozens of languages, including low-resource languages. For example, XGLM learned 30 languages, and BLOOM learned 42 languages. However, we plan to let go of the desire to be good at too many languages at once. The number of steps a model can learn is somewhat set, and the model converges when it exceeds that. So if one model takes too many languages, the training efficiency for each language decreases. Therefore, we want to train the model with languages in similar language families which enable synergy effect between them. In addition, we have excluded languages used by a few users use because it is difficult to collect a large amount of data. Therefore, we will only focus on high or middle-resource languages in our project.

## 2. Projects

### Polyglot-Ko [WIP]
When we started our research, we have already had 1.2TB of Korean data collected by [TUNiB](https://tunib.ai/). Before we collected a large amount of multilingual data, we decided to try Korean modeling with the dataset we already had. This Korean model can be used for performance comparison with the multilingual models, and this model itself would help many Korean companies and researchers.

| Size |                                      Training Status                                       |                           Model Card                            |                             Model Checkpoints                             |                            Demo Server                             |
|:----:|:------------------------------------------------------------------------------------------:|:---------------------------------------------------------------:|:-------------------------------------------------------------------------:|:-------------------------------------------------------------------------:|
| 1.3B | [Finished](https://wandb.ai/eleutherai-oslo/polyglot-ko-1_3b?workspace=user-eleutherai-oslo) | [Available](https://huggingface.co/EleutherAI/polyglot-ko-1.3b) | [Available](https://huggingface.co/EleutherAI/polyglot-ko-1.3b/tree/main) | [Available](https://huggingface.co/spaces/EleutherAI/polyglot-ko-1.3b) |
| 3.8B | [Finished](https://wandb.ai/eleutherai-oslo/polyglot-ko-3_8b?workspace=user-eleutherai-oslo) | [Available](https://huggingface.co/EleutherAI/polyglot-ko-3.8b) | [Available](https://huggingface.co/EleutherAI/polyglot-ko-3.8b/tree/main) | N/A
| 5.8B | [Finished](https://wandb.ai/eleutherai-oslo/polyglot-ko-5_8b?workspace=user-eleutherai-oslo) |                           Coming soon                           |                                Coming soon                                | [Available](https://master-polyglot-deploy-jason9693.endpoint.ainize.ai/) |
|12.8B | [Training](https://wandb.ai/eleutherai-oslo/polyglot-ko-12_8b?workspace=user-eleutherai-oslo) |                           Coming soon                           |                                Coming soon                               | Comming soon
| ...  |                                           Ready                                            |                           Coming soon                           |                                Coming soon                                  | Comming soon

### Polyglot-East-Asian [WIP]
We chose the East Asian language as our first multilingual dataset.
This model includes Korean, Chinese, Japanese, Indonesian, Malay, Vietnamese, Thai, and English.
We will train the model by collecting at least hundreds of billions tokens of data from each language and balancing them.
Some people may wonder why English is included on this list, but because English is now a global language, we believe it could synergize with any other language in the world.

| Size | Training Status | Model Card  | Model Checkpoints |
|:----:|:---------------:|:-----------:|:-----------------:|
| ...  |      Ready      | Coming soon |    Coming soon    |


## 3. Data Risks

### Privacy considerations
In order to avoid the model memorizing and generating personally identifiable information (PII) in the training data, we masked out the following sensitive information in the pre-processing stage:

* `<|acc|>` : bank account number
* `<|rrn|>` : resident registration number
* `<|tell|>` : phone number

### Limitations and Biases
Polyglot has been trained to optimize next token prediction. Language models such as this are often used for a wide variety of tasks and it is important to be aware of possible unexpected outcomes. For instance, Polyglot will not always return the most factual or accurate response but the most statistically likely one. In addition, Polyglot may produce socially unacceptable or offensive content. We recommend having a human curator or other filtering mechanism to censor sensitive content.

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
This project was made possible thanks to the computing resources from [Stability.ai](https://stability.ai), thanks to [TUNiB](https://tunib.ai) for providing a large-scale Korean dataset, and thanks to [Common Computer](https://comcom.ai/en/) for providing a demo server(GPU) for this work.
