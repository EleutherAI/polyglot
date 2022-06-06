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

from tqdm import tqdm
from multilingual.data.datasets.dataset_causal_lm import DatasetForCausalLM
from multilingual.data.utils.blenders import DatasetBlender
from multilingual.models.xpt_neo.modeling_xpt_neo import XPTNeoForCausalLM
from multilingual.utils import optimized_params, set_seed, get_lr


class CheckPhase:
    def __init__(self, initial_phase):
        assert initial_phase in [
            "phase1",
            "phase2",
        ], "'initial_phase' must be 'phase1' or 'phase2'"
        self.phase = initial_phase

    def __call__(self):
        return self.phase

    def set_phase(self, current_phase):
        self.phase = current_phase

# TODO add EealryStop configuration
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0, min_steps=0):
        self.patience = patience
        self.min_delta = min_delta
        self.min_steps = min_steps
        self.min_val_loss = 999999
        self.patience_count = 0

        self.initial_patience = self.patience
        self.initial_min_delta = self.min_delta
        self.initial_min_steps = self.min_steps
        self.initial_min_val_loss = self.min_val_loss
        self.initial_patience_count = self.patience_count

    def __call__(self, current_steps, valid_loss):
        '''
         If the early termination condition is satisfied, return True.
         otherwise, return None
        '''

        if valid_loss - self.min_delta > self.min_val_loss:
            if current_steps > self.min_steps:
                self.patience_count += 1
        elif valid_loss - self.min_delta < self.min_val_loss:
            self.min_val_loss = valid_loss
            self.patience_count = 0

        if self.patience_count > self.patience:
            return True # finish train

    def reset_(self):
        self.patience = self.initial_patience
        self.min_delta = self.initial_min_delta
        self.min_steps = self.initial_min_steps
        self.min_val_loss = self.initial_min_val_loss
        self.patience_count = self.initial_patience_count

# Initialize program
SEED = 42
CURRENT_STEP = 0
REAL_STEP = 0
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
model = XPTNeoForCausalLM.from_pretrained(config["model_name"], 
                                          reorder_and_upcast_attn=config["experiment"]["reorder_and_upcast_attn"])
model.gradient_checkpointing_enable()

# Initialize XPT training
model.initialize_xpt(
    new_embedding_size=len(tokenizer),
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    num_itl_layers=config["experiment"]["args"]["num_itl_layers"],
    pos_emb_requires_grad=config["experiment"]["args"]["pos_emb_requires_grad"],
)

# Initialize config parse
total_num_steps = (
    config["experiment"]["args"]["phase1_steps"]
    + config["experiment"]["args"]["phase2_steps"]
)
config["training"]["scheduler"]["params"]["total_num_steps"] = total_num_steps
if config["experiment"]["args"]["phase1_steps"] > 0:
    phase_detector = CheckPhase("phase1")
else:
    phase_detector = CheckPhase("phase2")


# Load datasets
train_sets, valid_sets = [], []
for dataset in config["datasets"]["names"]:
    train_sets.append(
        DatasetForCausalLM(
            data_name=os.path.join(dataset["name"],"train/merge"),
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
            data_name=os.path.join(dataset["name"],"val/merge"),
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
    # batch_size=config["training"]["train_batch_size"] // dist.get_world_size(),
    batch_size=config["training"]["train_micro_batch_size_per_gpu"],
    pin_memory=True,
    shuffle=False,
    num_workers=os.cpu_count() // dist.get_world_size(),
    sampler=DistributedSampler(train_dataset, shuffle=True),
)

valid_loader = DataLoader(
    valid_dataset,
    # batch_size=config["training"]["train_batch_size"] // dist.get_world_size(),
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
        project="XPT-Training",
    )

# Set training args
early_stopping = EarlyStopping(patience=config['experiment']['early_stopping_args']['patience'], 
                                min_delta=config['experiment']['early_stopping_args']['min_delta'], 
                                min_steps=config['experiment']['early_stopping_args']['min_steps'])

# Start training
wandb_generation_table = []

while True:
    if CURRENT_STEP >= total_num_steps:
        break

    for train_data in train_loader:
        if CURRENT_STEP == config["experiment"]["args"]["phase1_steps"]:
            model.phase2()
            if dist.get_rank() == 0:
                phase_detector.set_phase(current_phase="phase2")
                print("START Phase-2")
        
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
                    f"{phase_detector()}/train_loss": loss.item(),
                    f"{phase_detector()}/train_ppl": ppl,
                    "lr": get_lr(optimizer),
                },
                step=REAL_STEP,
            )
            print(
                f"STEP: {REAL_STEP}/" f"{total_num_steps}, " f"LOSS: {loss}, ",
                f"PPL: {ppl}, ",
                f"CURRENT PHASE: {phase_detector().upper()}"
            )

        engine.backward(loss)
        engine.step()

        if REAL_STEP % config["experiment"]["eval_interval"] == 0 and REAL_STEP != 0:
            if dist.get_rank() == 0:
                print("START VALIDATION")

            with torch.no_grad():
                model.eval()
                val_losses, valid_samples = [], []

                if config["experiment"]["max_eval_steps"] is not None:
                    eval_total = config["experiment"]["max_eval_steps"]
                else:
                    eval_total = len(valid_loader) + 1

                for eval_steps, valid_data in enumerate(tqdm(valid_loader, total=eval_total)):
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
                generated_output = model.generate(sample.unsqueeze(0), max_length=25, pad_token_id=tokenizer.eos_token_id)
                generated_text = tokenizer.decode(generated_output[0])
                input_prompt = tokenizer.decode(sample)
                print(f"Input Prompt: {input_prompt}")
                print(f"Generated sentence: {generated_text}")
                wandb_generation_table.append(
                    [REAL_STEP, input_prompt, generated_text]
                )

            if dist.get_rank() == 0:
                val_ppl = math.exp(val_loss)
                print("=" * 100)
                print(
                    f"STEP: {REAL_STEP}/"
                    f"{total_num_steps}, "
                    f"VALID_LOSS: {val_loss}, ",
                    f"VALID_PPL: {val_ppl}, ",
                    f"CURRENT PHASE: {phase_detector().upper()}"
                )
                print("=" * 100)

                wandb.log(
                    data={
                        f"{phase_detector()}/val_loss": val_loss,
                        f"{phase_detector()}/val_ppl": math.exp(val_loss),
                        f"{phase_detector()}/generation": Table(
                            columns=["Step", "Input Prompt", "Generated Text"],
                            data=wandb_generation_table,
                        ),
                    },
                    step=REAL_STEP,
                )

                if (
                    REAL_STEP
                    % config["experiment"]["checkpoint_params"]["save_interval"]
                    == 0
                    and REAL_STEP != 0
                ):
                    print("SAVE CHECKPOINTS...")
                    save_dir = os.path.join(
                        config["experiment"]["checkpoint_params"]["save_dir"],
                        f"{phase_detector()}-"
                        f"{config['experiment']['type']}-"
                        f"{config['experiment']['name']}-"
                        f"steps={REAL_STEP}",
                    )
                    model.save_pretrained(save_dir)
                    # engine.save_checkpoint(os.path.join(save_dir, "deepspeed"))

            if early_stopping(current_steps=CURRENT_STEP, valid_loss=val_loss) and phase_detector() == 'phase1':
                # finish phase 1
                print(f"Ealry STOP in {phase_detector()}")
                save_dir = os.path.join(
                    config["experiment"]["checkpoint_params"]["save_dir"],
                    f"{phase_detector()}-"
                    f"{config['experiment']['type']}-"
                    f"{config['experiment']['name']}-"
                    f"steps={REAL_STEP}",
                )
                model.save_pretrained(save_dir)
                # engine.save_checkpoint(os.path.join(save_dir, "deepspeed"))
                CURRENT_STEP = config["experiment"]["args"]["phase1_steps"]
                early_stopping.reset_()

            elif early_stopping(current_steps=CURRENT_STEP, valid_loss=val_loss) and phase_detector() == 'phase2':
                # finish phase 2
                print(f"Ealry STOP in {phase_detector()}")
                save_dir = os.path.join(
                    config["experiment"]["checkpoint_params"]["save_dir"],
                    f"{phase_detector()}-"
                    f"{config['experiment']['type']}-"
                    f"{config['experiment']['name']}-"
                    f"steps={REAL_STEP}",
                )
                model.save_pretrained(save_dir)
                # engine.save_checkpoint(os.path.join(save_dir, "deepspeed"))
                CURRENT_STEP = total_num_steps

        CURRENT_STEP += 1
        REAL_STEP += 1
