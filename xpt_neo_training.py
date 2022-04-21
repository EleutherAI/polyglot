import json
import math
import os
from argparse import ArgumentParser
from random import choice

import deepspeed
import torch.cuda
import torch.distributed as dist
import wandb
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from wandb import Table

from datasets import tqdm
from multilingual.data.datasets.dataset_causal_lm import DatasetForCausalLM
from multilingual.data.utils.blenders import DatasetBlender
from multilingual.models.xpt_neo.modeling_xpt_neo import XPTNeoForCausalLM
from multilingual.utils import optimized_params, set_seed, get_lr

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
assert config["experiment"]["type"] == "xpt-neo", "Wrong experiment type."

# Create tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
model = XPTNeoForCausalLM.from_pretrained(config["model_name"])
model.gradient_checkpointing_enable()

# Initialize XPT training
model.initialize_xpt(
    new_embedding_size=len(tokenizer),
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    num_itl_layers=config["experiment"]["args"]["num_itl_layers"],
    pos_emb_requires_grad=config["experiment"]["args"]["pos_emb_requires_grad"],
)


# Load datasets
train_sets, valid_sets = [], []
for dataset in config["datasets"]:
    train_sets.append(
        DatasetForCausalLM(
            data_name=dataset["name"],
            max_seq_length=model.config.max_position_embeddings,
            binarization_impl="mmap",
            split_type="train",
            start_weight=0.0,
            end_weight=0.99,
            seed=SEED,
        )
    )
    valid_sets.append(
        DatasetForCausalLM(
            data_name=dataset["name"],
            max_seq_length=model.config.max_position_embeddings,
            binarization_impl="mmap",
            split_type="valid",
            start_weight=0.99,
            end_weight=1.0,
            seed=SEED,
        )
    )


train_dataset = DatasetBlender(
    datasets=train_sets,
    weights=[d["weight"] for d in config["datasets"]],
)

valid_dataset = DatasetBlender(
    datasets=valid_sets,
    weights=[d["weight"] for d in config["datasets"]],
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config["training"]["train_batch_size"],
    pin_memory=True,
    shuffle=False,
    num_workers=os.cpu_count() // dist.get_world_size(),
    sampler=DistributedSampler(train_dataset, shuffle=False),
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=config["training"]["train_batch_size"],
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
        name=f"{config['experiment']['type']}-{config['experiment']['name']}",
        project="XPT-Training",
    )


# Start training
wandb_generation_table = []

while True:
    if CURRENT_STEP >= config["training"]["scheduler"]["params"]["total_num_steps"]:
        break

    for train_data in train_loader:
        if CURRENT_STEP >= config["training"]["scheduler"]["params"]["total_num_steps"]:
            break

        engine.train()
        train_data = train_data.cuda()
        loss = engine(
            input_ids=train_data,
            labels=train_data,
            use_cache=False,
        ).loss

        if dist.get_rank() == 0:
            wandb.log(
                data={
                    "train_loss": loss.item(),
                    "train_ppl": math.exp(loss.item()),
                    "lr": get_lr(optimizer),
                },
                step=CURRENT_STEP,
            )
            print(
                f"STEP: {CURRENT_STEP}/"
                f"{config['training']['scheduler']['params']['total_num_steps']}, "
                f"LOSS: {loss}"
            )

        engine.backward(loss)
        engine.step()

        if CURRENT_STEP % config["experiment"]["eval_interval"] == 0:
            if dist.get_rank() == 0:
                print("START VALIDATION")

            with torch.no_grad():
                model.eval()
                val_losses, valid_samples = [], []

                for valid_data in tqdm(valid_loader):
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

                save_dir = os.path.join(
                    config["experiment"]["save_dir"],
                    f"{config['experiment']['type']}-"
                    f"{config['experiment']['name']}-"
                    f"steps={CURRENT_STEP}",
                )

                if dist.get_rank() == 0:
                    print("=" * 100)
                    print(
                        f"STEP: {CURRENT_STEP}/"
                        f"{config['training']['scheduler']['params']['total_num_steps']}, "
                        f"LOSS: {loss}"
                    )
                    print("=" * 100)

                    wandb.log(
                        data={
                            "val_loss": val_loss,
                            "val_ppl": math.exp(val_loss),
                            "generation": Table(
                                columns=["Step", "Input Prompt", "Generated Text"],
                                data=wandb_generation_table,
                            ),
                        },
                        step=CURRENT_STEP,
                    )

                    model.save_pretrained(save_dir)
                engine.save_checkpoint(os.path.join(save_dir, "deepspeed"))

        CURRENT_STEP += 1
