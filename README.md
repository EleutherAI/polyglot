# Polyglot: Large Language Models of Well-balanced Competence in Multi-languages

## 1. Introduction

### Why another multilingual model?
Various multilingual models such as [mBERT](https://huggingface.co/bert-base-multilingual-cased), [BLOOM](https://huggingface.co/bigscience/bloom), and [XGLM](https://arxiv.org/abs/2112.10668) have been released.
Therefore, someone might ask, "why do we need to make multilingual models again?" Before answering the question, we would like to ask, "Why do people around the world make monolingual models in their language even though there are already many multilingual models?" We would like to point out there is a dissatisfaction with the non-English language performance of the current multilingual models as one of the most significant reason. So we want to make multilingual models with higher non-English language performance. This is the reason we need to make multilingual models again and why we name them ['Polyglot'](https://www.spanish.academy/blog/what-is-the-difference-between-a-polyglot-and-a-multilingual-person/).

## 2. Projects

### 1) Polyglot-Ko [DONE]
When we started our research, we have already had 1.2TB of Korean data collected by [TUNiB](https://tunib.ai/). Before we collected a large amount of multilingual data, we decided to try Korean modeling with the dataset we already had. This Korean model can be used for performance comparison with the multilingual models, and this model itself would help many Korean companies and researchers.

| Size |                                      Training Status                                       |                           Model Card                            |                             Model Checkpoints                             |                            Demo                             |
|:----:|:------------------------------------------------------------------------------------------:|:---------------------------------------------------------------:|:-------------------------------------------------------------------------:|:-------------------------------------------------------------------------:|
| 1.3B | [Finished](https://wandb.ai/eleutherai/polyglot-ko/groups/polyglot-ko-1.3B) | [Available](https://huggingface.co/EleutherAI/polyglot-ko-1.3b) | [Available](https://huggingface.co/EleutherAI/polyglot-ko-1.3b/tree/main) | [Available](https://huggingface.co/spaces/EleutherAI/polyglot-ko-1.3b) |
| 3.8B | [Finished](https://wandb.ai/eleutherai/polyglot-ko/groups/polyglot-ko-3.8B) | [Available](https://huggingface.co/EleutherAI/polyglot-ko-3.8b) | [Available](https://huggingface.co/EleutherAI/polyglot-ko-3.8b/tree/main) | N/A                                                                    |
| 5.8B | [Finished](https://wandb.ai/eleutherai/polyglot-ko/groups/polyglot-ko-5.8B) |                           [Available](https://huggingface.co/EleutherAI/polyglot-ko-5.8b)                           |                                [Available](https://huggingface.co/EleutherAI/polyglot-ko-5.8b/tree/main)                                | N/A                                                           |
|12.8B | [Finished](https://wandb.ai/eleutherai-oslo/polyglot-ko-12_8b) |              [Available](https://huggingface.co/EleutherAI/polyglot-ko-12.8b)                           |                                [Available](https://huggingface.co/EleutherAI/polyglot-ko-12.8b/tree/main)                               | N/A

💡 We are collaborating with KoAlpaca team which is creating a series of Korean instruct fine-tuned models. As a result, we were able to release the Koalapca-Polyglot models. Please refer to [here](https://github.com/Beomi/KoAlpaca) to see more details.

## 3. Limitations and Biases
Polyglot has been trained to optimize next token prediction. Language models such as this are often used for a wide variety of tasks and it is important to be aware of possible unexpected outcomes. For instance, Polyglot will not always return the most factual or accurate response but the most statistically likely one. In addition, Polyglot may produce socially unacceptable or offensive content. We recommend having a human curator or other filtering mechanism to censor sensitive content.

## 4. Citation and Related Information

### BibTeX entry
If you find our work useful, please consider citing:
```bibtex
@misc{polyglot-ko,
  title = {{Polyglot-Ko: Open-Source Korean Autoregressive Language Model}},
  author = {Ko, Hyunwoong and Yang, Kichang and Ryu, Minho and Choi, Taekyoon and Yang, Seungmu and Hyun, jiwung and Park, Sungho},
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
This project was made possible thanks to the computing resources from [Stability.ai](https://stability.ai), thanks to [TUNiB](https://tunib.ai) for providing a large-scale Korean dataset.
