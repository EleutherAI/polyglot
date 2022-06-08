import json
import logging
import math
import os
from argparse import ArgumentParser
from random import choice

import deepspeed
import torch
import torch.distributed as dist
import wandb
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    set_seed,
    GPT2TokenizerFast,
    GPTNeoConfig,
)
from transformers.adapters import AdapterCompositionBlock, Fuse
from transformers.adapters.configuration import AdapterConfig
from wandb import Table

from multilingual.data.datasets.dataset_causal_lm import DatasetForCausalLM
from multilingual.data.utils.blenders import DatasetBlender
from multilingual.models.mad_x.modeling_mad_x import GPTNeoForCausalLM
from multilingual.utils import optimized_params, get_lr

logger = logging.getLogger(__name__)

# Initialize program
SEED = 42
CURRENT_STEP = 0
dist.init_process_group("nccl")
torch.cuda.set_device(torch.distributed.get_rank())
os.environ["TOKENIZERS_PARALLELISM"] = "true"
set_seed(SEED)

# Parse config
parser = ArgumentParser()
parser.add_argument("--config", "-c", type=str, required=True)
parser.add_argument("--local_rank", type=int)
config = json.load(open(parser.parse_args().config))
args = config["experiment"]["args"]

model_config = GPTNeoConfig.from_pretrained(config["model_name"])
model_config.vocab_size = 30003

model = GPTNeoForCausalLM.from_pretrained(config["model_name"])
model.gradient_checkpointing_enable()

tokenizer = GPT2TokenizerFast.from_pretrained(config["tokenizer_name"])
model_config.eos_token_id = tokenizer.eos_token_id
task_name = f"{args['task_name']}_{args['language']}"

if task_name not in model.config.adapters:
    adapter_config = AdapterConfig.load(
        args["adapter_config"],
        non_linearity=args["adapter_non_linearity"],
        reduction_factor=args["adapter_reduction_factor"],
    )

    if args["load_adapter"]:
        model.load_adapter(args["load_adapter"], adapter_config, load_as=task_name)
    else:
        model.add_adapter(task_name, config=adapter_config)

    if args["load_lang_adapter"]:
        lang_adapter_config = AdapterConfig.load(
            args["lang_adapter_config"],
            non_linearity=args["lang_adapter_non_linearity"],
            reduction_factor=args["adapter_reduction_factor"],
        )

        lang_adapter_name = model.load_adapter(
            args["load_lang_adapter"],
            config=lang_adapter_config,
            load_as=args["language"],
        )
    else:
        lang_adapter_name = None

    model.train_adapter(task_name, train_embeddings=True)
else:
    if args["load_adapter"] or args["load_lang_adapter"]:
        raise ValueError

if args["embedding_strategies"] == "overlap-replace":
    orig_tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    model.add_embeddings(
        "lng_emb",
        tokenizer,
        reference_embedding="default",
        reference_tokenizer=orig_tokenizer,
    )
    model._active_embedding = "lng_emb"
    model.delete_embeddings("default")
    model.tie_weights()
elif args["embedding_strategies"] == "replace":
    model.resize_token_embeddings(len(tokenizer))

trainable_params = 0
frozen_params = 0
emb_params = 0
for name, param in model.named_parameters():
    if "word_embeddings" in name:
        param.requires_grad = True
        emb_params += param.numel()

    elif args["lang_adapt_strategies"] == "emb":
        param.requires_grad = False

    if not param.requires_grad:
        if dist.get_rank() == 0:
            print(f"🥶 Frozen layer '{name}'")
        frozen_params += param.numel()
    else:
        if dist.get_rank() == 0:
            print(f"🚀 Trainable layer '{name}'")
        trainable_params += param.numel()

    if "wte" in name and "wpe" in name:
        emb_params += param.numel()

if dist.get_rank() == 0:
    print(f"Total frozen parameters: {frozen_params}")
    print(f"Total emb parameters (wte, wpe): {emb_params}")
    print(f"Total trainable parameters: {trainable_params}")

model_frozen = getattr(model.base_model, "model_frozen", False)

if model_frozen and model.active_adapters:
    # Check if training AdapterFusion
    train_adapter_fusion = (
        isinstance(model.active_adapters, Fuse)
        or isinstance(model.active_adapters, AdapterCompositionBlock)
        and any([isinstance(child, Fuse) for child in model.active_adapters.children])
    )

if model.active_adapters is None:
    raise ValueError()


# Load datasets
train_sets, valid_sets = [], []
for dataset in config["datasets"]["names"]:
    train_sets.append(
        DatasetForCausalLM(
            data_name=os.path.join(dataset["name"], "train/merge"),
            max_seq_length=model.config.max_position_embeddings,
            binarization_impl=config["datasets"]["params"]["binarization_impl"],
            split_type="train",
            start_weight=config["datasets"]["params"]["train"]["start_weight"],
            end_weight=config["datasets"]["params"]["train"]["end_weight"],
            seed=SEED,
        )
    )
    valid_sets.append(
        DatasetForCausalLM(
            data_name=os.path.join(dataset["name"], "val/merge"),
            max_seq_length=model.config.max_position_embeddings,
            binarization_impl=config["datasets"]["params"]["binarization_impl"],
            split_type="valid",
            start_weight=config["datasets"]["params"]["valid"]["start_weight"],
            end_weight=config["datasets"]["params"]["valid"]["end_weight"],
            seed=SEED,
        )
    )


train_dataset = DatasetBlender(
    datasets=train_sets,
    weights=[d["weight"] for d in config["datasets"]["names"]],
)

valid_dataset = DatasetBlender(
    datasets=valid_sets,
    weights=[d["weight"] for d in config["datasets"]["names"]],
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config["training"]["train_micro_batch_size_per_gpu"],
    pin_memory=True,
    shuffle=False,
    num_workers=os.cpu_count() // dist.get_world_size(),
    sampler=DistributedSampler(train_dataset, shuffle=True),
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=config["training"]["train_micro_batch_size_per_gpu"],
    pin_memory=True,
    shuffle=False,
    num_workers=os.cpu_count() // dist.get_world_size(),
    sampler=DistributedSampler(valid_dataset, shuffle=False),
)


# Optimization
engine, optimizer, _, _ = deepspeed.initialize(
    config=config["training"],
    model=model,
    model_parameters=optimized_params(
        model=model,
        weight_decay=config["training"]["optimizer"]["params"]["weight_decay"],
    ),
)

# Initialize wandb
if dist.get_rank() == 0:
    wandb.init(
        config=config,
        name=f"{config['experiment']['type']}-{config['experiment']['name']}",
        project="MAD-X-Training",
    )


# Start training
wandb_generation_table = []
total_num_steps = config["training"]["scheduler"]["params"]["total_num_steps"]

while True:
    if CURRENT_STEP >= total_num_steps:
        break

    for train_data in train_loader:
        if CURRENT_STEP >= total_num_steps:
            break

        engine.train()
        train_data = train_data.cuda()
        loss = engine(
            input_ids=train_data,
            labels=train_data,
            use_cache=False,
        ).loss

        if dist.get_rank() == 0:
            ppl = math.exp(loss.item())
            wandb.log(
                data={
                    f"train_loss": loss.item(),
                    f"train_ppl": ppl,
                    "lr": get_lr(optimizer),
                },
                step=CURRENT_STEP,
            )
            print(
                f"STEP: {CURRENT_STEP}/" f"{total_num_steps}, " f"LOSS: {loss}, ",
                f"PPL: {ppl}",
            )

        engine.backward(loss)
        engine.step()

        if (
            CURRENT_STEP % config["experiment"]["eval_interval"] == 0
            and CURRENT_STEP != 0
        ):
            if dist.get_rank() == 0:
                print("START VALIDATION")

            with torch.no_grad():
                model.eval()
                val_losses, valid_samples = [], []

                if config["experiment"]["max_eval_steps"] is not None:
                    eval_total = config["experiment"]["max_eval_steps"]
                else:
                    eval_total = len(valid_loader) + 1

                for eval_steps, valid_data in enumerate(
                    tqdm(valid_loader, total=eval_total)
                ):
                    if config["experiment"]["max_eval_steps"] is not None:
                        if eval_steps > config["experiment"]["max_eval_steps"]:
                            break

                    valid_data = valid_data.cuda()
                    val_loss = engine(
                        input_ids=valid_data,
                        labels=valid_data,
                        use_cache=False,
                    ).loss
                    val_losses.append(val_loss.detach().item())
                    valid_samples.append(choice(valid_data))

                val_loss = sum(val_losses) / len(val_losses)

                num_prompt_tokens = 5
                sample = choice(valid_samples)[:num_prompt_tokens]
                generated_output = model.generate(sample.unsqueeze(0), max_length=25)
                generated_text = tokenizer.decode(generated_output[0])
                wandb_generation_table.append(
                    [CURRENT_STEP, tokenizer.decode(sample), generated_text]
                )

                if (
                    CURRENT_STEP
                    % config["experiment"]["checkpoint_params"]["save_interval"]
                    == 0
                    and CURRENT_STEP != 0
                ):
                    save_dir = os.path.join(
                        config["experiment"]["checkpoint_params"]["save_dir"],
                        f"{config['experiment']['type']}-"
                        f"{config['experiment']['name']}-"
                        f"steps={CURRENT_STEP}",
                    )

                if dist.get_rank() == 0:
                    val_ppl = math.exp(val_loss)
                    print("=" * 100)
                    print(
                        f"STEP: {CURRENT_STEP}/"
                        f"{total_num_steps}, "
                        f"LOSS: {val_loss}, ",
                        f"PPL: {val_ppl}",
                    )
                    print("=" * 100)

                    wandb.log(
                        data={
                            f"val_loss": val_loss,
                            f"val_ppl": math.exp(val_loss),
                            f"generation": Table(
                                columns=["Step", "Input Prompt", "Generated Text"],
                                data=wandb_generation_table,
                            ),
                        },
                        step=CURRENT_STEP,
                    )

                    if (
                        CURRENT_STEP
                        % config["experiment"]["checkpoint_params"]["save_interval"]
                        == 0
                        and CURRENT_STEP != 0
                    ):
                        model.save_pretrained(save_dir)

        CURRENT_STEP += 1
