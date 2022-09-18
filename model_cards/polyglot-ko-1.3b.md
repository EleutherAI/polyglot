# Polyglot-Ko-1.3B

## Model Description
Polyglot-Ko is a Korean autoregressive language model made by EleutherAI polyglot team. We collected about 1.2TB Korean dataset for this work, which was done with [TUNiB](https://tunib.ai/). In addition, we used the [GPT-NeoX framework](https://github.com/EleutherAI/gpt-neox) for model training and added several Korean tasks to [LM-Evaluation-Harness](https://github.com/EleutherAI/lm-evaluation-harness) for model evaluation.

| Hyperparameter       | Value                                                                                                                                  |
|----------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| \\(n_{parameters}\\) | 13,3181,0304                                                                                                                           |
| \\(n_{layers}\\)     | 24                                                                                                                                     |
| \\(d_{model}\\)      | 2048                                                                                                                                   |
| \\(d_{ff}\\)         | 8192                                                                                                                                   |
| \\(n_{heads}\\)      | 16                                                                                                                                     |
| \\(d_{head}\\)       | 128                                                                                                                                    |
| \\(n_{ctx}\\)        | 2048                                                                                                                                   |
| \\(n_{vocab}\\)      | 30,000 / 30,080                                                                                                                        |
| Positional Encoding  | [Rotary Position Embedding (RoPE)](https://arxiv.org/abs/2104.09864)                                                                   |
| RoPE Dimensions      | [64](https://github.com/kingoflolz/mesh-transformer-jax/blob/f2aa66e0925de6593dcbb70e72399b97b4130482/mesh_transformer/layers.py#L223) |

The model consists of 24 transformer layers with a model dimension of 2048, and a feedforward dimension of 8192. The model
dimension is split into 16 heads, each with a dimension of 128. Rotary Position Embedding (RoPE) is applied to 64
dimensions of each head. The model is trained with a tokenization vocabulary of 30000.

## Training data

Polyglot-Ko was trained on 1.2TB Korean Dataset, a large-scale curated dataset created by [TUNiB](https://tunib.ai/).

## Training procedure

Polyglot-Ko was trained for 213 billion tokens over 102,000 steps on 256 * A100 GPUs. It was trained as an autoregressive language model, using cross-entropy loss to maximize the likelihood of predicting the next token correctly.

## How to use

This model can be easily loaded using the `AutoModelForCausalLM` functionality:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-1.3b")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/polyglot-ko-1.3b")
```

## Privacy considerations and Limitations

Polyglot-Ko learns an inner representation of the Korean that can be used to extract features useful for downstream tasks. The model is best at what it was pretrained for however, which is generating text from a prompt.

### Privacy considerations
General training algorithms for pretrained language model have many hazards that memorize personal information in training data. We added the following tokens to vocabulary to mitigate privacy problem and replaced much personal information to these tokens in data preprocessing steps.

* `<|acc|>` : bank account number
* `<|rrn|>` : resident registration number
* `<|tell|>` : phone number

### Limitations and Biases

The core functionality of Polyglot-Ko is taking a string of text and predicting the next token. While language models are widely used for tasks other than this, there are a lot of unknowns with this work. When prompting Polyglot-Ko it is important to remember that the statistically most likely next token is often not the token that produces the most "accurate" text. Never depend upon Polyglot-Ko to produce factually accurate output.Depending upon use case Polyglot-Ko may produce socially unacceptable text.

As with all language models, it is hard to predict in advance how Polyglot-Ko will respond to particular prompts and offensive content may occur without warning. We recommend having a human curate or filter the outputs before releasing them, both to censor undesirable content and to improve the quality of the results.

### Legal Restrictions
Since there are laws in many countries related to data collection, we will collect data with due regard to the laws of those countries.
Additionally, we plan to use dataset to train our models, but we do not plan to make the dataset publicly available.

## Evaluation results
We used the [KOBEST dataset](https://arxiv.org/abs/2204.04541), which consists of five Korean downstream tasks for model evaluation.
We added the corresponding tasks to [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and utilized prompt templates described in the paper.
The following tables show the evaluation results with the various number of few-shot examples. You can reproduce these results using [polyglot branch of lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/polyglot) and the following scripts.

```console
python main.py \
   --model gpt2 \
   --model_args pretrained='EleutherAI/polyglot-ko-1.3b' \
   --tasks kobest_boolq,kobest_copa,kobest_wic,kobest_hellaswag,kobest_sentineg \
   --num_fewshot $YOUR_NUM_FEWSHOT \
   --batch_size $YOUR_BATCH_SIZE \
   --device $YOUR_DEVICE \
   --output_path $/path/to/output/
```

- the number of few shot examples = 1

| Model                                                                                        | \\(n_{parameters}\\) | boolq (F1) | copa (F1)  | wic (F1)   | hellaswag (F1) | sentineg (F1) | average    |
|----------------------------------------------------------------------------------------------|----------------------|------------|------------|------------|----------------|---------------|------------|
| [skt/ko-gpt-trinity-1.2B-v0.5](https://huggingface.co/skt/ko-gpt-trinity-1.2B-v0.5) &dagger; | 1.2B                 | 0.4243     | 0.6773     | 0.328      | 0.4178         | 0.5587        | 0.48122    |
| [kakaobrain/kogpt](https://huggingface.co/kakaobrain/kogpt) &ast;                            | 6.0B                 | **0.5014** | **0.7446** | **0.4187** | **0.4524**     | 0.7419        | **0.5718** |
| [EleutherAI/polyglot-ko-1.3b](https://huggingface.co/EleutherAI/polyglot-ko-1.3b) (ours)     | 1.3B                 | 0.3986     | 0.7106     | 0.4116     | 0.3884         | **0.8509**    | 0.55202    |

- the number of few shot examples = 5

| Model                                                                                        | \\(n_{parameters}\\) | boolq (F1) | copa (F1)  | wic (F1)   | hellaswag (F1) | sentineg (F1) | average     |
|----------------------------------------------------------------------------------------------|----------------------|------------|------------|------------|----------------|---------------|-------------|
| [skt/ko-gpt-trinity-1.2B-v0.5](https://huggingface.co/skt/ko-gpt-trinity-1.2B-v0.5) &dagger; | 1.2B                 | 0.3346     | 0.6477     | 0.328      | 0.4            | 0.5186        | 0.44578     |
| [kakaobrain/kogpt](https://huggingface.co/kakaobrain/kogpt) &ast;                            | 6.0B                 | **0.5561** | **0.7287** | **0.3802** | **0.456**      | 0.7152        | **0.56724** |
| [EleutherAI/polyglot-ko-1.3b](https://huggingface.co/EleutherAI/polyglot-ko-1.3b) (ours)     | 1.3B                 | 0.5101     | 0.7193     | 0.328      | 0.3984         | **0.8057**    | 0.5523      |

- the number of few shot examples = 10

| Model                                                                                        | \\(n_{parameters}\\) | boolq (F1) | copa (F1)  | wic (F1)   | hellaswag (F1) | sentineg (F1) | average     |
|----------------------------------------------------------------------------------------------|----------------------|------------|------------|------------|----------------|---------------|-------------|
| [skt/ko-gpt-trinity-1.2B-v0.5](https://huggingface.co/skt/ko-gpt-trinity-1.2B-v0.5) &dagger; | 1.2B                 | 0.3402     | 0.6419     | 0.328      | 0.4011         | 0.529         | 0.44804     |
| [kakaobrain/kogpt](https://huggingface.co/kakaobrain/kogpt) &ast;                            | 6.0B                 | 0.4838     | **0.7277** | **0.3989** | **0.4616**     | 0.7422        | 0.56284     |
| [EleutherAI/polyglot-ko-1.3b](https://huggingface.co/EleutherAI/polyglot-ko-1.3b) (ours)     | 1.3B                 | **0.5262** | 0.7204     | 0.3314     | 0.417          | **0.8413**    | **0.56726** |

- the number of few shot examples = 50

| Model                                                                                        | \\(n_{parameters}\\) | boolq (F1) | copa (F1)  | wic (F1)   | hellaswag (F1) | sentineg (F1) | average     |
|----------------------------------------------------------------------------------------------|----------------------|------------|------------|------------|----------------|---------------|-------------|
| [skt/ko-gpt-trinity-1.2B-v0.5](https://huggingface.co/skt/ko-gpt-trinity-1.2B-v0.5) &dagger; | 1.2B                 | 0.3405     | 0.6514     | 0.328      | 0.4214         | 0.3798        | 0.42422     |
| [kakaobrain/kogpt](https://huggingface.co/kakaobrain/kogpt) &ast;                            | 6.0B                 | 0.4888     | **0.7479** | 0.4233     | **0.4754**     | **0.6757**    | **0.56222** |
| [EleutherAI/polyglot-ko-1.3b](https://huggingface.co/EleutherAI/polyglot-ko-1.3b) (ours)     | 1.3B                 | **0.5072** | 0.7206     | **0.4288** | 0.4416         | 0.6049        | 0.54062     |

- the number of few shot examples = 100

| Model                                                                                        | \\(n_{parameters}\\) | boolq (F1) | copa (F1)  | wic (F1)   | hellaswag (F1) | sentineg (F1) | average     |
|----------------------------------------------------------------------------------------------|----------------------|------------|------------|------------|----------------|---------------|-------------|
| [skt/ko-gpt-trinity-1.2B-v0.5](https://huggingface.co/skt/ko-gpt-trinity-1.2B-v0.5) &dagger; | 1.2B                 | 0.3381     | 0.6593     | 0.328      | 0.4187         | 0.3798        | 0.42478     |
| [kakaobrain/kogpt](https://huggingface.co/kakaobrain/kogpt) &ast;                            | 6.0B                 | 0.4755     | **0.7468** | 0.4225     | **0.458**      | **0.7081**    | **0.56218** |
| [EleutherAI/polyglot-ko-1.3b](https://huggingface.co/EleutherAI/polyglot-ko-1.3b) (ours)     | 1.3B                 | **0.4981** | 0.7343     | **0.4329** | 0.426          | 0.5948        | 0.53722     |

<p><strong>&dagger;</strong> The model card of this model provides evaluation results for the KOBEST dataset, but when we evaluated the model with the prompts described in the paper, we can't get similar results to it. Therefore, we checked the KOBEST paper and found that the results were similar to the fine-tuning results reported in the paper. Because we evaluated by prompt-based generation without fine-tuning the model, the results provided by the model card for the this model may differ.</p>

<p><strong>&ast;</strong> Since this model does not provide evaluation results with KOBEST dataset, we evaluated the model using lm-evaluation-harness ourselves. you can reproduce this result using the source code included in the polyglot branch of lm-evaluation-harness.</p>

## Citation and Related Information

### BibTeX entry
If you find our work useful, please consider citing:
```bibtex
@misc{polyglot-ko,
  title = {{Plyglot-Ko: Open-Source Korean Autoregressive Language Model}},
  author = {Ko, Hyunwoong and Yang, Kichang and Ryu, Minho and Kim, Taekyun and Yang, Seungmu and Hyun, jiwung and Park, Sungho},
  url = {https://www.github.com/eleutherai/polyglot},
  month = {9},
  year = {2022},
}
```

### Licensing
All our models are  licensed under the terms of the Apache License 2.0.

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

This project would not have been possible without compute generously provided by [Stability.ai](https://stability.ai), thanks them for providing a large amount of GPU resources. And thanks also go to [TUNiB](https://tunib.ai) for providing a large-scale Korean dataset for this work.

