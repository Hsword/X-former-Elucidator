from model_wrapper import ModelForSequenceClassification, ModelForSequenceClassificationDual
from dataset import LRADataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import time
import os
import json
import numpy as np
import argparse
import math
from transformers import set_seed
import itertools


parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, help = "model", dest = "model", required =True)
parser.add_argument("--task", type = str, help = "task", dest = "task", required = True)
parser.add_argument("--skip_train", type = int, help = "skip_train", dest = "skip_train", default = 0)
parser.add_argument("--batch_size", type = int, help = "batch_size", dest = "batch_size", default = 0)
parser.add_argument("--lr", type = float, help = "learning rate", dest = "lr", default=0)
parser.add_argument("--seq_len", type = int, help = "seq_len", dest = "seq_len", default = 0)
parser.add_argument("--cls_head", type = str, help = "classifier_head", dest = "cls_head", default = "cls")
parser.add_argument("--seed", type = int, help = "random seed", dest = "seed", default = 0)
args = parser.parse_args()

attn_type = args.model
task = args.task
cls_head = args.cls_head

curr_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.join(curr_path, os.pardir)

with open(os.path.join(root_path, 'configs', f'{task}.json'), 'r') as f:
    config = json.load(f)

attn_config = config['attn'][attn_type]
attn_config['attn_type'] = attn_type
model_config = config['model']
model_config.update(attn_config)
model_config['use_cls_token'] = True if cls_head.lower() == 'cls' else False
model_config['pooling_mode'] = cls_head.upper()
if args.seq_len != 0:
    model_config['max_seq_len'] = args.seq_len
model_config['max_seq_len'] = int(2 ** math.ceil(math.log2(model_config["max_seq_len"])))
model_config['mixed_precision'] = False

training_config = config['training']
if args.batch_size != 0:
    training_config['batch_size'] = args.batch_size
if args.lr != 0:
    training_config['learning_rate'] = args.lr

inst_per_gpu = config['inst_per_gpu'][attn_type]

checkpoint_dir = os.path.join(root_path, 'models', cls_head.lower(), attn_type.split('-')[0], 'model')
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
log_path = os.path.join(root_path, 'models', cls_head.lower(), attn_type.split('-')[0], 'logs')
if not os.path.exists(log_path):
    os.mkdir(log_path)
dataset_folder = os.path.join(root_path, 'datasets', task)

device_ids = list(range(torch.cuda.device_count()))
print(f"GPU list: {device_ids}")

print(json.dumps([model_config, training_config], indent = 4))
print(json.dumps({
    'attn_type': attn_type,
    'task': task,
    'batch size per device': inst_per_gpu,
    'random_seed': args.seed
}))

set_seed(args.seed)

########################### Loading Datasets ###########################

ds_loader = {
    'train':DataLoader(
        LRADataset(os.path.join(dataset_folder, f'{task}-train.pickle'), True),
        batch_size = training_config["batch_size"],
        drop_last = True),
    'dev':DataLoader(
        LRADataset(os.path.join(dataset_folder, f'{task}-dev.pickle'), True),
        batch_size = training_config["batch_size"],
        drop_last = True),
    'test':DataLoader(
        LRADataset(os.path.join(dataset_folder, f'{task}-test.pickle'), False),
        batch_size = training_config["batch_size"],
        drop_last = True)
}

ds_iter = { split: enumerate(ds_loader[split]) for split in ds_loader }

num_train_steps = training_config['num_train_steps']
num_eval_steps = training_config['num_eval_steps']

########################### Loading Model ###########################

if task == "retrieval":
    model = ModelForSequenceClassificationDual(model_config)
else:
    model = ModelForSequenceClassification(model_config)

print(model)
print(f"parameter_size: {[weight.size() for weight in model.parameters()]}", flush = True)
print(f"num_parameter: {np.sum([np.prod(weight.size()) for weight in model.parameters()])}", flush = True)

model = model.cuda()
model = nn.DataParallel(model, device_ids = device_ids)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = training_config["learning_rate"],
    betas = (0.9, 0.999), eps = 1e-6, weight_decay = training_config["weight_decay"]
)

lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer = optimizer,
    max_lr = training_config["learning_rate"],
    pct_start = training_config["warmup"] / training_config["num_train_steps"],
    anneal_strategy = training_config["lr_decay"],
    total_steps = training_config['num_train_steps']
)

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

            partial_outputs['loss'].backward()

        optimizer.step()
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

    t1 = time.time()

    batch_size = batch[list(batch.keys())[0]].size(0)
    t_escape = t1 - t0
    learning_rate = optimizer.param_groups[0]["lr"]
    loss = outputs["loss"].data.item()
    accu = outputs["accu"].data.item()
    time_since_start = time.time() - init_t

    print(f"step={step_idx}, tt={time_since_start:.1f}, t={t_escape:.3f}, bs={batch_size}, lr={learning_rate:.8f}, loss={loss:.4f}, accu={accu:.4f}\t\t\t\t", end = "\r", flush = True)

    summary[component]["t"] += t_escape
    summary[component]['total_time'] = time_since_start
    summary[component]["loss"].append(loss)
    summary[component]["accu"].append(accu)


def print_summary(summary, save_if_improved, train_step_idx):
    summary["loss"] = np.mean(summary["loss"])
    summary["accu"] = np.mean(summary["accu"])

    print()
    if summary["accu"] > summary["best_accu"]:
        summary["best_accu"] = summary["accu"]
        if save_if_improved:
            best_accu = summary["best_accu"]
            torch.save({"model_state_dict":model.module.state_dict()}, dump_path)
            print(f"best_accu={best_accu}. Saved best model to path {dump_path}")

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


init_t = time.time()

log_f_path = os.path.join(log_path, f"{task}_{args.seed}.log")
log_f = open(log_f_path, "a+")

dump_path = os.path.join(checkpoint_dir, f'{task}_{args.seed}.model')

summary = {
    component:{'total_time':0, "t":0, "loss":[], "accu":[], "best_accu":0, "component":component}
    for component in ["train", "dev", "test"]
}

accumu_steps = max(training_config["batch_size"] // len(device_ids) // inst_per_gpu, 1)
print(f"accumu_steps={accumu_steps}")

if args.skip_train == 0:
    try:
        model.train()
        for train_step_idx in range(training_config['num_train_steps']):
            outputs = step("train", train_step_idx)

            if (train_step_idx + 1) % training_config["eval_frequency"] == 0:
                print_summary(summary["train"], False, train_step_idx)
                model.eval()
                for dev_step_idx in range(training_config['num_eval_steps']):
                    outputs = step("dev", dev_step_idx)
                print_summary(summary["dev"], True, train_step_idx)
                model.train()
    except KeyboardInterrupt as e:
        print(e)

checkpoint = torch.load(dump_path, map_location = "cpu")
model.module.load_state_dict(checkpoint["model_state_dict"])
model.eval()
try:
    for test_step_idx in itertools.count():
        outputs = step("test", test_step_idx)
except StopIteration:
    print_summary(summary["test"], False, train_step_idx)
