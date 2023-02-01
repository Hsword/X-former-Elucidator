from model_wrapper import ModelForImageClassification
from torch.utils.data import DataLoader
from datasets import load_dataset, load_metric
from transformers import set_seed, ViTFeatureExtractor
from torchvision.transforms import Compose, Lambda, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor, Resize, CenterCrop
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import json
import numpy as np
import argparse
import utils
# from github/rwightman/pytorch-image-models/timm
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["labels"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


########################### Parsing and Printing ###########################

parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, help = "model", dest = "model", required = True)
parser.add_argument("--batch_size", type = int, help = "batch size", dest = "batch_size", required = True)
parser.add_argument("--seq_len", type = int, help = "seq_len", dest = "seq_len", required = False, default = 1024)
parser.add_argument("--lr", type = float, help = "learning rate", dest = "lr", required = True)
parser.add_argument("--epoch", type = int, help = "epoch", dest = "epoch", required = True)
parser.add_argument("--task", type = str, help = "downstream task", dest = "task", required = True)
parser.add_argument("--checkpoint", type = int, help = "checkpoint path", dest = "checkpoint", required = True)
parser.add_argument("--image_size", type = int, help="image size", dest="image_size", required=False, default=None)
parser.add_argument("--patch_size", type = int, help="patch size", dest="patch_size", required=False, default=None)
parser.add_argument("--mask_patch_size", type = int, help="mask patch size", dest="mask_patch_size", required=False, default=None)
parser.add_argument("--encoder_stride", type = int, help="encoder stride", dest="encoder_stride", required=False, default=None)
parser.add_argument("--seed", type = int, help = "random seed", dest = "seed", required = False, default = 0)
parser.add_argument("--anneal_strategy", type=str, help="anneal strategy", dest="anneal_strategy", required=False, default=None)
parser.add_argument("--weight_decay", type=float, help="weight decay", dest="weight_decay", required=False, default=0.05)
args = parser.parse_args()

curr_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.join(curr_path, os.pardir)

with open(os.path.join(root_path, 'models', f"seq_len-{args.seq_len}", args.model, 'config.json'), 'r') as f:
    config = json.load(f)

model_config = config["model"]
model_config["use_cls_token"] = True
model_config["use_mask_token"] = False
pretraining_config = config["pretraining_setting"]
gpu_config = config["gpu_setting"]
checkpoint_dir = os.path.join(root_path, 'models', f"seq_len-{args.seq_len}", args.model, 'model')

model_config.update(
    {
        "image_size": args.image_size if args.image_size is not None else model_config["image_size"],
        "patch_size": args.patch_size if args.patch_size is not None else model_config["patch_size"],
        "mask_patch_size": args.mask_patch_size if args.mask_patch_size is not None else model_config["mask_patch_size"],
        "encoder_stride": args.encoder_stride if args.encoder_stride is not None else model_config["encoder_stride"],
    }
)

pretraining_config.update(
    {
        "anneal_strategy": args.anneal_strategy if args.anneal_strategy is not None else pretraining_config["anneal_strategy"] if "anneal_strategy" in pretraining_config else "linear"
    }
)

device_ids = list(range(torch.cuda.device_count()))
print(f"GPU list: {device_ids}")

print(json.dumps([model_config, pretraining_config], indent = 4))
print(json.dumps({
    "total batch size": args.batch_size,
    "learning rate": args.lr,
    "anneal strategy": args.anneal_strategy,
    "weight decay": args.weight_decay,
    "training epoch": args.epoch,
    "task": args.task,
    "checkpoint resotred from": args.checkpoint,
    "image size": args.image_size,
    "patch size": args.patch_size,
    "mask patch size": args.mask_patch_size,
    "encoder stride": args.encoder_stride,
    "random_seed": args.seed})
)

set_seed(args.seed)

########################### Loading Datasets ###########################

raw_datasets = load_dataset(args.task if args.task != "tiny-imagenet-200" else "Maysee/tiny-imagenet", cache_dir=os.path.join(root_path, "datasets"), task="image-classification")

print("Raw Datasets:")
print(raw_datasets)

column_names = raw_datasets["train"].column_names
if "image" in column_names:
    image_column_name = "image"
elif "img" in column_names:
    image_column_name = "img"
else:
    image_column_name = column_names[0]

if "labels" in column_names:
    label_name = "labels"
elif "label" in column_names:
    label_name = "label"
else:
    label_name = column_names[-1]

# Labels / Classes
label_list = raw_datasets["train"].features[label_name].names
print(label_list)
model_config["num_classes"] = len(label_list)

feature_extractor = ViTFeatureExtractor(
    size=model_config["image_size"],
    image_mean=IMAGENET_DEFAULT_MEAN,
    image_std=IMAGENET_DEFAULT_STD
)

# Preprocessing the raw_datasets

normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
_train_transforms = Compose(
    [
        Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        RandomResizedCrop(feature_extractor.size, scale=(0.67, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)

_val_transforms = Compose(
    [
        Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        Resize(feature_extractor.size),
        CenterCrop(feature_extractor.size),
        ToTensor(),
        normalize,
    ]
)

def train_transforms(examples):
    """Apply _train_transforms across a batch."""

    examples["pixel_values"] = [_train_transforms(image) for image in examples[image_column_name]]
    return examples

def val_transforms(examples):
    """Apply _val_transforms across a batch."""

    examples["pixel_values"] = [_val_transforms(image) for image in examples[image_column_name]]
    return examples


raw_datasets["train"].set_transform(train_transforms)
train_dataset = raw_datasets["train"]

if "test" in raw_datasets:
    raw_datasets["test"].set_transform(val_transforms)
    test_dataset = raw_datasets["test"]
else:
    raw_datasets["valid"].set_transform(val_transforms)
    test_dataset = raw_datasets["valid"]

########################### Loading Model ###########################

model = ModelForImageClassification(model_config)
print(model)

model = model.cuda()
model = nn.DataParallel(model, device_ids = device_ids)

# Get the metric function
metric = load_metric("accuracy")

train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=collate_fn,
    batch_size=args.batch_size
)

num_steps_per_epoch = len(train_dataloader)
print(f"num_steps_per_epoch: {num_steps_per_epoch}", flush = True)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = args.lr,
    betas = (0.9, 0.999), eps = 1e-6, weight_decay = args.weight_decay
)

lr_scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer = optimizer,
    max_lr = args.lr,
    pct_start = pretraining_config["warmup"],
    anneal_strategy = pretraining_config["anneal_strategy"],
    epochs = args.epoch,
    steps_per_epoch = num_steps_per_epoch
)

if model_config["mixed_precision"]:
    amp_scaler = torch.cuda.amp.GradScaler()
else:
    amp_scaler = None

ckpt_store_path = os.path.join(checkpoint_dir, args.task)
if not os.path.exists(ckpt_store_path):
    os.mkdir(ckpt_store_path)

start_epoch = 0
restore_from_ckpt = False
for epoch in reversed(range(pretraining_config["epoch"])):
    checkpoint_path = os.path.join(ckpt_store_path, f"cp-{epoch:03}-{args.checkpoint}-{args.lr}-{args.batch_size}-{args.epoch}.cp")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location = 'cpu')
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        start_epoch = checkpoint["epoch"] + 1
        print("Model restored from", checkpoint_path)
        restore_from_ckpt = True
        break

if not restore_from_ckpt:
    checkpoint_path = os.path.join(checkpoint_dir, f"cp-{args.checkpoint:04}.model")
    checkpoint = torch.load(checkpoint_path, map_location = 'cpu')
    missing_keys, unexpected_keys = model.module.load_state_dict(checkpoint['model_state_dict'], strict = False)
    print(f"missing_keys = {missing_keys}")
    print(f"unexpected_keys = {unexpected_keys}")
    print("Model initialized", checkpoint_path)

########################### Running Model ###########################

log_dir = os.path.join(root_path, "models", f"seq_len-{args.seq_len}", args.model, "logs")
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
log_file_name = f"ic-{args.task}-{args.checkpoint}-{args.lr}-{args.batch_size}-{args.epoch}.log"
print(f"Log file: {log_file_name}", flush = True)
log_f = open(os.path.join(log_dir, log_file_name), "a+")

partition_names = ["train", "test"]
accumu_steps = utils.compute_accumu_step(args.batch_size, len(device_ids), gpu_config["inst_per_gpu"])
print("accumu_steps", accumu_steps)

init_t = time.time()
for epoch in range(start_epoch, args.epoch):
    for partition_name in partition_names:

        training = partition_name == "train"

        if training:
            data_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, collate_fn = collate_fn)
            model.train()
        else:
            data_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)
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
                predictions = outputs["logits"].argmax(dim=-1).cpu().data.tolist()
                metric.add_batch(
                    predictions=predictions,
                    references=labels
                )
                del outputs["logits"]

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
            summary["time_since_start"] = round(time.time() - init_t, 4)
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
            dump_path = os.path.join(ckpt_store_path, f"cp-{epoch:03}-{args.checkpoint}-{args.lr}-{args.batch_size}-{args.epoch}.cp")
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
