from unittest.util import _MAX_LENGTH
from model_wrapper import ModelForSequenceClassification
from transformers import RobertaTokenizerFast
from torch.utils.data import DataLoader
from transformers.data.data_collator import default_data_collator
from datasets import load_dataset, load_metric
from transformers import set_seed, GlueDataset, GlueDataTrainingArguments
import torch
import torch.nn as nn
import time
import os
import json
import numpy as np
import argparse
import utils

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, help = "model", dest = "model", required = True)
parser.add_argument("--batch_size", type = int, help = "batch size", dest = "batch_size", required = True)
parser.add_argument("--lr", type = float, help = "learning rate", dest = "lr", required = True)
parser.add_argument("--epoch", type = int, help = "epoch", dest = "epoch", required = True)
parser.add_argument("--task", type = str, help = "downstream task", dest = "task", required = True)
parser.add_argument("--checkpoint", type = int, help = "checkpoint path", dest = "checkpoint", required = True)
parser.add_argument("--seed", type = int, help = "random seed", dest = "seed", required = False, default = 0)
parser.add_argument("--inst_per_gpu", type = int, help = "inst per gpu", dest = "inst_per_gpu", required = False, default = 0)
parser.add_argument("--seq_len", type = int, help = "seq_len", dest = "seq_len", required = False, default = 512)
args = parser.parse_args()

curr_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.join(curr_path, os.pardir)

with open(os.path.join(root_path, 'models', f"seq_len-{args.seq_len}", args.model, 'config.json'), 'r') as f:
    config = json.load(f)

model_config = config["model"]
model_config["use_cls_token"] = True
pretraining_config = config["pretraining_setting"]
gpu_config = config["gpu_setting"]
if args.inst_per_gpu != 0:
    gpu_config["inst_per_gpu"] = args.inst_per_gpu
checkpoint_dir = os.path.join(root_path, 'models', f"seq_len-{args.seq_len}", args.model, 'model')
glue_dataset_folder = os.path.join(root_path, "datasets", "glue")

device_ids = list(range(torch.cuda.device_count()))
print(f"GPU list: {device_ids}")

print(json.dumps([model_config, pretraining_config], indent = 4))
print(json.dumps({
    "batch size per device": gpu_config["inst_per_gpu"],
    "learning rate": args.lr,
    "training epoch": args.epoch,
    "task": args.task,
    "checkpoint resotred from": args.checkpoint,
    "random_seed": args.seed})
)

set_seed(args.seed)

########################### Loading Datasets ###########################

tokenizer = utils.get_tokenizer(model_config["max_seq_len"])
model_config["vocab_size"] = len(tokenizer.get_vocab())
raw_datasets = load_dataset("glue", args.task, cache_dir=os.path.join(root_path, "datasets"))

print("Raw Datasets:")
print(raw_datasets)

# Labels / Classes
is_regression = args.task == "stsb"
if not is_regression:
    label_list = raw_datasets["train"].features["label"].names
    print(label_list)
    model_config["num_classes"] = len(label_list)
else:
    model_config["num_classes"] = 1

# Preprocessing the raw_datasets
sentence1_key, sentence2_key = task_to_keys[args.task]

def preprocess_function(instances):
    # Tokenize the texts
    args = (
        (instances[sentence1_key],) if sentence2_key is None else (instances[sentence1_key], instances[sentence2_key])
    )
    result = tokenizer(*args, padding="max_length", max_length=model_config["max_seq_len"], truncation=True)
    
    if "label" in instances:
        result["labels"] = instances["label"]
    return result

processed_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
    load_from_cache_file=True,
)

print("Processed Datasets:")
print(processed_datasets)

train_dataset = processed_datasets["train"]
if args.task == "mnli":
    dev_datasets = {
        "dev": processed_datasets["validation_matched"],
        "dev-mm": processed_datasets["validation_mismatched"]
    }
    test_datasets = {
        "test": processed_datasets["test_matched"],
        "test-mm": processed_datasets["test_mismatched"]
    }
else:
    dev_datasets = {"dev": processed_datasets["validation"]}
    test_datasets = {"test": processed_datasets["test"]}

########################### Loading Model ###########################

model = ModelForSequenceClassification(model_config)
print(model)

model = model.cuda()
model = nn.DataParallel(model, device_ids = device_ids)

checkpoint_path = os.path.join(checkpoint_dir, f"cp-{args.checkpoint:04}.model")
checkpoint = torch.load(checkpoint_path, map_location = 'cpu')
missing_keys, unexpected_keys = model.module.load_state_dict(checkpoint['model_state_dict'], strict = False)
print(f"missing_keys = {missing_keys}")
print(f"unexpected_keys = {unexpected_keys}")
print("Model restored", checkpoint_path)

ckpt_store_path = os.path.join(checkpoint_dir, args.task)
if not os.path.exists(ckpt_store_path):
    os.mkdir(ckpt_store_path)

# Get the metric function
metric = load_metric("glue", args.task)

train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=default_data_collator,
    batch_size=args.batch_size
)

num_steps_per_epoch = len(train_dataloader)
print(f"num_steps_per_epoch: {num_steps_per_epoch}", flush = True)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = args.lr,
    betas = (0.9, 0.999), eps = 1e-6, weight_decay = 0.01
)

lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer = optimizer,
    max_lr = args.lr,
    pct_start = 0.05,
    anneal_strategy = "linear",
    epochs = args.epoch,
    steps_per_epoch = num_steps_per_epoch
)

if model_config["mixed_precision"]:
    amp_scaler = torch.cuda.amp.GradScaler()
else:
    amp_scaler = None

########################### Running Model ###########################

log_dir = os.path.join(root_path, "models", f"seq_len-{args.seq_len}", args.model, "logs")
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
log_file_name = f"glue-{args.task}-{args.checkpoint}-{args.lr}-{args.batch_size}-{args.epoch}.log"
print(f"Log file: {log_file_name}", flush = True)
log_f = open(os.path.join(log_dir, log_file_name), "a+")

partition_names = ["train"] + list(dev_datasets.keys())
accumu_steps = utils.compute_accumu_step(args.batch_size, len(device_ids), gpu_config["inst_per_gpu"])
print("accumu_steps", accumu_steps)

for epoch in range(args.epoch):
    for partition_name in partition_names:

        training = partition_name == "train"

        if training:
            data_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, collate_fn = default_data_collator)
            model.train()
        else:
            data_loader = DataLoader(dev_datasets[partition_name], batch_size = args.batch_size, shuffle = False, collate_fn = default_data_collator)
            model.eval()

        for batch_idx, batch in enumerate(data_loader):

            t0 = time.time()

            summary = {}
            if training:
                optimizer.zero_grad()

            for percent, inputs in utils.partition_inputs(batch, accumu_steps, True):

                if training:
                    outputs = model(**inputs)
                else:
                    with torch.no_grad():
                        outputs = model(**inputs)

                labels = inputs["labels"].cpu().data.tolist()
                predictions = (outputs["sent_scores"].argmax(dim=-1) if not is_regression else outputs["sent_scores"].squeeze()).cpu().data.tolist()
                metric.add_batch(
                    predictions=predictions,
                    references=labels
                )
                del outputs["sent_scores"]

                for key in outputs:
                    outputs[key] = outputs[key].mean() * percent
                if training:
                    utils.backward(outputs["loss"], amp_scaler)

                utils.add_output_to_summary(outputs, summary)

            if training:
                utils.optimizer_step(optimizer, lr_scheduler, amp_scaler)

            t1 = time.time()

            summary["batch_idx"] = batch_idx
            summary["epoch"] = epoch
            summary["time"] = round(t1 - t0, 4)
            summary["partition_name"] = partition_name
            if training:
                summary["learning_rate"] = round(optimizer.param_groups[0]["lr"], 8)

            log_f.write(json.dumps(summary, sort_keys = True) + "\n")
            if batch_idx % 10 == 0:
                print(json.dumps(summary, sort_keys = True), flush = True)
                log_f.flush()

        metric_result = metric.compute()
        metric_result["partition_name"] = partition_name
        metric_result["epoch"] = epoch
        print(json.dumps(metric_result, sort_keys = True), flush = True)
        log_f.write(json.dumps(metric_result, sort_keys = True) + "\n")
        log_f.flush()

        if training:
            dump_path = os.path.join(ckpt_store_path, f"cp-{epoch:02}-{args.checkpoint}-{args.lr}-{args.batch_size}-{args.epoch}.cp")
            torch.save({
                "model_state_dict":model.module.state_dict()
            }, dump_path.replace(".cp", ".model"))
            torch.save({
                "model_state_dict":model.module.state_dict(),
                "optimizer_state_dict":optimizer.state_dict(),
                "lr_scheduler_state_dict":lr_scheduler.state_dict(),
                "epoch":epoch,
            }, dump_path)
            print(f"Dump {dump_path}", flush = True)
