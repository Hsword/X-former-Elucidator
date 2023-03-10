from model_wrapper import ModelForMaskedLM
from dataset import CorpusDataset
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import time
import os
import json
import pickle
import numpy as np
import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, help = "model", dest = "model", required = True)
parser.add_argument("--seq_len", type = int, help = "seq len", dest = "seq_len", required = False, default = 512)
parser.add_argument("--start_epoch", type = int, help = "start epoch", dest = "start_epoch", required = False, default = None)
parser.add_argument("--end_epoch", type = int, help = "end epoch", dest = "end_epoch", required = False, default = None)
args = parser.parse_args()

curr_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.join(curr_path, os.pardir)

with open(os.path.join(root_path, "models", f"seq_len-{args.seq_len}", f"{args.model}", "config.json"), "r") as f:
    config = json.load(f)

model_config = config["model"]
pretraining_config = config["pretraining_setting"]
gpu_config = config["gpu_setting"]
checkpoint_dir = os.path.join(root_path, "models", f"seq_len-{args.seq_len}", f"{args.model}", "model")

if "dataset_folder" in config:
    data_folder = os.path.join(root_path, "datasets", config["data_folder"])
else:
    data_folder = os.path.join("datasets", f"{args.seq_len}-roberta")


if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

device_ids = list(range(torch.cuda.device_count()))
print(f"GPU list: {device_ids}")

print(json.dumps([model_config, pretraining_config], indent = 4))

########################### Loading Datasets ###########################

if "dataset" not in config:
    config["dataset"] = None

tokenizer = utils.get_tokenizer(model_config["max_seq_len"])
model_config["vocab_size"] = len(tokenizer.get_vocab())

dataset = CorpusDataset(folder_path = data_folder, file_json = "train.json", option = config["dataset"])
data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = True, mlm_probability = 0.15)
data_loader = DataLoader(dataset, batch_size = pretraining_config["batch_size"], shuffle = True, collate_fn = data_collator)
pretrain_dataloader_iter = enumerate(data_loader)

########################### Loading Model ###########################

model = ModelForMaskedLM(model_config)

print(model)
print(f"parameter_size: {[weight.size() for weight in model.parameters()]}", flush = True)
print(f"num_parameter: {np.sum([np.prod(weight.size()) for weight in model.parameters()])}", flush = True)

model = model.cuda()
model = nn.DataParallel(model, device_ids = device_ids)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = pretraining_config["learning_rate"],
    betas = (0.9, 0.999), eps = 1e-6, weight_decay = 0.01
)

lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer = optimizer,
    max_lr = pretraining_config["learning_rate"],
    pct_start = pretraining_config["warmup"],
    anneal_strategy = "linear",
    epochs = pretraining_config["epoch"],
    steps_per_epoch = pretraining_config["batches_per_epoch"]
)

amp_scaler = torch.cuda.amp.GradScaler() if model_config["mixed_precision"] else None

start_epoch = 0
inst_pass = 0
if args.start_epoch is not None:
    checkpoint_path = os.path.join(checkpoint_dir, f"cp-{args.start_epoch-1:04}.cp")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location = 'cpu')
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        start_epoch = checkpoint["epoch"] + 1
        inst_pass = checkpoint["inst_pass"]
        print("Model restored from", checkpoint_path)
else:
    for epoch in reversed(range(pretraining_config["epoch"])):
        checkpoint_path = os.path.join(checkpoint_dir, f"cp-{epoch:04}.cp")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location = 'cpu')
            model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            start_epoch = checkpoint["epoch"] + 1
            inst_pass = checkpoint["inst_pass"]
            print("Model restored from ", checkpoint_path)
            break

if start_epoch == 0:
    print("Model randomly initialized")


end_epoch = pretraining_config["epoch"]
if args.end_epoch is not None:
    end_epoch = args.end_epoch


########################### Running Model ###########################

log_dir = os.path.join(root_path, "models", f"seq_len-{args.seq_len}", f"{args.model}", "logs")
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
log_f = open(os.path.join(log_dir, "pretrain.log"), "a+")

accumu_steps = utils.compute_accumu_step(pretraining_config["batch_size"], len(device_ids), gpu_config["inst_per_gpu"])
print("accumu_steps", accumu_steps)

model.train()

init_t = time.time()
for epoch in range(start_epoch, end_epoch):

    for batch_idx in range(pretraining_config["batches_per_epoch"]):

        t0 = time.time()

        optimizer.zero_grad()

        _, batch = next(pretrain_dataloader_iter)

        inst_pass += list(batch.values())[0].size(0)
        summary = {}

        for percent, inputs in utils.partition_inputs(batch, accumu_steps, True):
            outputs = model(**inputs)
            for key in outputs:
                outputs[key] = outputs[key].mean() * percent
            utils.backward(outputs["loss"], amp_scaler)
            utils.add_output_to_summary(outputs, summary)

        utils.optimizer_step(optimizer, lr_scheduler, amp_scaler)
        del batch

        t1 = time.time()

        summary["idx"] = epoch * pretraining_config["batches_per_epoch"] + batch_idx
        summary["batch_idx"] = batch_idx
        summary["epoch"] = epoch
        summary["time"] = round(t1 - t0, 4)
        summary["inst_pass"] = inst_pass
        summary["learning_rate"] = round(optimizer.param_groups[0]["lr"], 8)
        summary["time_since_start"] = round(time.time() - init_t, 4)

        log_f.write(json.dumps(summary, sort_keys = True) + "\n")

        if batch_idx % pretraining_config["batches_per_report"] == 0:
            print(json.dumps(summary, sort_keys = True), flush = True)
            log_f.flush()

    print(f"finished epoch {epoch:04}", flush = True)

    dump_path = os.path.join(checkpoint_dir, f"cp-{epoch:04}.cp")
    torch.save({
        "model_state_dict":model.module.state_dict()
    }, dump_path.replace(".cp", ".model"))
    torch.save({
        "model_state_dict":model.module.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "lr_scheduler_state_dict":lr_scheduler.state_dict(),
        "epoch":epoch,
        "inst_pass":inst_pass
    }, dump_path)
    print(f"Dump {dump_path}", flush = True)
