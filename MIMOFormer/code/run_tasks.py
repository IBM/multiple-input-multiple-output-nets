#
# Copyright 2023- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

# ==================================================================================================
# IMPORTS
# ==================================================================================================
from model_wrapper import ModelForSC, ModelForSCDual
from dataset import LRADataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import time
import os, sys
import json
import numpy as np
import argparse
import math
import itertools
import lra_config
import random

# ==================================================================================================
# FUNCTIONS
# ==================================================================================================
def check_paths(args):
    try:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        new_log_dir = os.path.join(args.log_dir, time.ctime().replace(" ", "-"))
        args.log_dir = new_log_dir
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
    except OSError as e:
        print(e)
        sys.exit(1)

def step(component, step_idx):

    t0 = time.time()
    optimizer.zero_grad()

    _, batch = next(ds_iter[component])
    for key in batch:
        batch[key] = batch[key].cuda()

    if component == "train":
        outputs = {}

        partial_inputs_list = [{} for _ in range(accumu_steps)]
        for key in batch:
            for idx, inp in enumerate(torch.chunk(batch[key], accumu_steps, dim = 0)):
                partial_inputs_list[idx][key] = inp

        for partial_inputs in partial_inputs_list:
            partial_outputs = model(**partial_inputs)
            for key in partial_outputs:
                partial_outputs[key] = partial_outputs[key].mean() / accumu_steps
                if key not in outputs:
                    outputs[key] = partial_outputs[key]
                else:
                    outputs[key] += partial_outputs[key]
            amp_scaler.scale(partial_outputs["loss"]).backward()

        amp_scaler.step(optimizer)
        amp_scaler.update()
        lr_scheduler.step()
    else:
        with torch.no_grad():
            outputs = {}

            partial_inputs_list = [{} for _ in range(accumu_steps)]
            for key in batch:
                for idx, inp in enumerate(torch.chunk(batch[key], accumu_steps, dim = 0)):
                    partial_inputs_list[idx][key] = inp

            for partial_inputs in partial_inputs_list:
                partial_outputs = model(**partial_inputs)
                for key in partial_outputs:
                    partial_outputs[key] = partial_outputs[key].mean() / accumu_steps
                    if key not in outputs:
                        outputs[key] = partial_outputs[key]
                    else:
                        outputs[key] += partial_outputs[key]


    batch_size = batch[list(batch.keys())[0]].size(0)
    learning_rate = optimizer.param_groups[0]["lr"]
    loss =  outputs["loss"].data.item()
    accu = outputs["accu"].data.item()

    t1 = time.time()
    t_escape = t1 - t0
    time_since_start = time.time() - init_t
    print(f"step={step_idx}, tt={time_since_start:.1f}, t={t_escape:.3f}, bs={batch_size}, lr={learning_rate:.6f}, loss={loss:.4f}, accu={accu:.4f}\t\t\t\t", end = "\r", flush = True)

    summary[component]["t"] += t_escape
    summary[component]["loss"].append(loss)
    summary[component]["accu"].append(accu)

def print_summary(summary, save_if_improved, train_step_idx,mode="train", save_best = True):
    summary["loss"] = np.mean(summary["loss"])
    summary["accu"] = np.mean(summary["accu"])


    writer.add_scalar(f"accuracy/{mode}",summary["accu"],train_step_idx)
    writer.add_scalar(f"loss/{mode}",summary["loss"],train_step_idx)
    
    print()

    # Save every model
    save_name = os.path.join(args.checkpoint_dir, 'checkpoint.pth.tar')
    torch.save({"model_state_dict":model.module.state_dict()},save_name)
    
    # Save best
    if summary["accu"] > summary["best_accu"] and save_best:
        summary["best_accu"] = summary["accu"]
        if save_if_improved:
            best_accu = summary["best_accu"]
            save_name = os.path.join(args.checkpoint_dir, 'model_best.pth.tar')
            torch.save({"model_state_dict":model.module.state_dict()},save_name)
            print(f"best_accu={best_accu}. Saved best model")

    summary_round = {"train_step_idx":train_step_idx}
    for key in summary:
        if type(summary[key]) is str:
            summary_round[key] = summary[key]
        else:
            summary_round[key] = round(summary[key], 4)

    print(summary_round, flush = True)
    log_f.write(json.dumps(summary_round, sort_keys = True) + "\n")
    log_f.flush()

    summary["t"] = 0
    summary["loss"] = []
    summary["accu"] = []


# ==================================================================================================
# MAIN
# ==================================================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, help = "model", dest = "model", required = True)
parser.add_argument("--task", type = str, help = "task", dest = "task", required = True)
parser.add_argument("--skip-train", type = int, help = "skip_train", dest = "skip_train", default = 0)
parser.add_argument("--random-seed", type = int, default = 1234)
parser.add_argument("--datapath", type=str, default="data/")
parser.add_argument("--logpath", type=str, default="results/")
parser.add_argument("--expname", type=str, default="00_test")
parser.add_argument("-x", "--experiment_nr", type=int, default=0, 
                    help="useful for storing results in separate folders")


args = parser.parse_args()

attn_type = args.model
task = args.task

# Set the random seed manually 
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
# seed dataloader workers
def seed_worker(worker_id):
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
# seed generators
g = torch.Generator()
g.manual_seed(args.random_seed)


print(lra_config.config[task]["extra_attn_config"].keys(), flush = True)

model_config = lra_config.config[task]["model"]
model_config.update(lra_config.config[task]["extra_attn_config"][attn_type])

model_config["mixed_precision"] = True
model_config["attn_type"] = attn_type
model_config["max_seq_len"] = int(2 ** math.ceil(math.log2(model_config["max_seq_len"])))

training_config = lra_config.config[task]["training"]
gpu_memory_config = lra_config.config[task]["gpu_memory"]

device_ids = list(range(torch.cuda.device_count()))
print(f"GPU list: {device_ids}")

print(json.dumps([model_config, training_config], indent = 4))

# init model
if task == "retrieval":
    model = ModelForSCDual(model_config)
else:
    model = ModelForSC(model_config)

print(model)
print(f"parameter_size: {[weight.size() for weight in model.parameters()]}", flush = True)
print(f"num_parameter: {np.sum([np.prod(weight.size()) for weight in model.parameters()])}", flush = True)
ortho_regularization = model_config["ortho_regularization"]

# create experiment directory
args.exp_dir = args.logpath+args.expname+f"_{task}_{attn_type}_ortho_{ortho_regularization}_{model.extra_repr()}/{args.experiment_nr}/"
args.checkpoint_dir = args.exp_dir+"ckpt/"
args.save_dir = args.exp_dir+"save/"
args.log_dir = args.exp_dir+"log/"
check_paths(args)
# Tensorboard writer
writer = SummaryWriter(args.log_dir)

model = model.cuda()
model = nn.DataParallel(model, device_ids = device_ids)


# Create enumerators for the train/dev/test datasets
if task == 'pathfinder32':
    task = 'pathfinder32-curv_contour_length_14'


if task != "retrieval":
    ds_iter = {
        "train":enumerate(DataLoader(LRADataset(f"{args.datapath}/lra-{task}.train.pickle", True), batch_size = training_config["batch_size"], drop_last = True, worker_init_fn=seed_worker, generator=g)),
        "dev":enumerate(DataLoader(LRADataset(f"{args.datapath}/lra-{task}.dev.pickle", True), batch_size = training_config["batch_size"], drop_last = True, worker_init_fn=seed_worker, generator=g)),
        "test":enumerate(DataLoader(LRADataset(f"{args.datapath}/lra-{task}.test.pickle", False), batch_size = training_config["batch_size"], drop_last = True, worker_init_fn=seed_worker, generator=g)),
    }
else:
    ds_iter = {
        "train":enumerate(DataLoader(LRADataset(f"/dataset/acl_anthology/processed/{task}.train.pickle", True), batch_size = training_config["batch_size"], drop_last = True, worker_init_fn=seed_worker, generator=g)),
        "dev":enumerate(DataLoader(LRADataset(f"/dataset/acl_anthology/processed/{task}.dev.pickle", True), batch_size = training_config["batch_size"], drop_last = True, worker_init_fn=seed_worker, generator=g)),
        "test":enumerate(DataLoader(LRADataset(f"/dataset/acl_anthology/processed/{task}.test.pickle", False), batch_size = training_config["batch_size"], drop_last = True, worker_init_fn=seed_worker, generator=g)),
    }

# init optimzer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = training_config["learning_rate"],
    betas = (0.9, 0.999), eps = 1e-6, weight_decay = training_config["weight_decay"]
)

# init LR scheduler
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer = optimizer,
    max_lr = training_config["learning_rate"],
    pct_start = training_config["warmup"] / training_config["num_train_steps"],
    anneal_strategy = training_config["lr_decay"],
    total_steps = training_config["num_train_steps"]
)

amp_scaler = torch.cuda.amp.GradScaler() if model_config["mixed_precision"] else None

init_t = time.time()

# Dump config to output.log
log_f_path = os.path.join(args.log_dir, f"output.log")
log_f = open(log_f_path, "a+")
log_f.write(json.dumps(lra_config.config[task]) + "\n")
log_f.flush()

summary = {
    component:{"t":0, "loss":[], "accu":[], "best_accu":0, "component":component}
    for component in ["train", "dev", "test"]
}

accumu_steps = max(training_config["batch_size"] // len(device_ids) // gpu_memory_config[attn_type], 1)
print(f"accumu_steps={accumu_steps}")

# Main training loop
warmup = False
MIMO_warmup = model_config["MIMO_warmup"] if args.model=="mimoformer" else 0

if args.skip_train == 0:
    try:
        model.train()
        for train_step_idx in range(training_config["num_train_steps"]):

            # MIMO curriculum learning init 
            if MIMO_warmup!=0:
                if train_step_idx ==0:
                    model.module.model.start_MIMO_warmup()
                    warmup = True

            outputs = step("train", train_step_idx)

            if (train_step_idx + 1) % training_config["eval_frequency"] == 0:
                print_summary(summary["train"], False, train_step_idx,"train", save_best = not warmup)
                model.eval()
                for dev_step_idx in range(training_config["num_eval_steps"]):
                    outputs = step("dev", dev_step_idx)
                print_summary(summary["dev"], True, train_step_idx,"val", save_best = not warmup)
                model.train()

            # MIMO curriculum learning: change to standard N, M 
            if MIMO_warmup!=0:
                if (train_step_idx+1) %MIMO_warmup==0: 
                    model.module.model.stop_MIMO_warmup()
                    warmup = False

    except KeyboardInterrupt as e:
        print(e)
else: 
    train_step_idx = 0

# Validate the model 
save_name = os.path.join(args.checkpoint_dir, 'model_best.pth.tar')
checkpoint = torch.load(save_name, map_location = "cpu")
model.module.load_state_dict(checkpoint["model_state_dict"])
model.eval()
try:
    for test_step_idx in itertools.count():
        outputs = step("test", test_step_idx)
except StopIteration:
    print_summary(summary["test"], False, train_step_idx, "test")

writer.close()