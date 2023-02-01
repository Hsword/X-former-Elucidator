from model_wrapper import ModelForMaskedIM
from transformers import set_seed, ViTFeatureExtractor
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor
import torch
import torch.nn as nn
import time
import os
import json
import pickle
import numpy as np
import argparse
import utils
# from github/rwightman/pytorch-image-models/timm
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# Implemented in huggingface transformers expamples for SimMIM
# Source:
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/image-pretraining/run_mim.py
class MaskGenerator:
    """
    A class to generate boolean masks for the pretraining task.

    A mask is a 1D tensor of shape (model_patch_size**2,) where the value is either 0 or 1,
    where 1 indicates "masked".
    """

    def __init__(self, input_size=64, mask_patch_size=8, model_patch_size=2, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        if self.input_size % self.mask_patch_size != 0:
            raise ValueError("Input size must be divisible by mask patch size")
        if self.mask_patch_size % self.model_patch_size != 0:
            raise ValueError("Mask patch size must be divisible by model patch size")

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size**2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[: self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return torch.tensor(mask.flatten())


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    mask = torch.stack([example["mask"] for example in examples])
    return {"pixel_values": pixel_values, "bool_masked_pos": mask}


########################### Parsing and Printing ###########################

parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, help = "model", dest = "model", required = True)
parser.add_argument("--seq_len", type = int, help = "seq_len", dest = "seq_len", required = False, default = 1024)
parser.add_argument("--image_size", type = int, help="image size", dest="image_size", required=False, default=None)
parser.add_argument("--patch_size", type = int, help="patch size", dest="patch_size", required=False, default=None)
parser.add_argument("--mask_patch_size", type = int, help="mask patch size", dest="mask_patch_size", required=False, default=None)
parser.add_argument("--encoder_stride", type = int, help="encoder stride", dest="encoder_stride", required=False, default=None)
parser.add_argument("--seed", type = int, help = "random seed", dest = "seed", required = False, default = 0)
parser.add_argument("--anneal_strategy", type=str, help="anneal strategy", dest="anneal_strategy", required=False, default=None)
args = parser.parse_args()


curr_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.join(curr_path, os.pardir)

with open(os.path.join(root_path, "models", f"seq_len-{args.seq_len}", args.model, "config.json"), "r") as f:
    config = json.load(f)

model_config = config["model"]
model_config["use_cls_token"] = True
model_config["use_mask_token"] = True
pretraining_config = config["pretraining_setting"]
gpu_config = config["gpu_setting"]
checkpoint_dir = os.path.join(root_path, "models", f"seq_len-{args.seq_len}", args.model, "model")
data_folder = os.path.join(root_path, "datasets", "tiny-imagenet-200")

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

if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

device_ids = list(range(torch.cuda.device_count()))
print(f"GPU list: {device_ids}")

print(json.dumps([model_config, pretraining_config], indent = 4))

set_seed(args.seed)

########################### Loading Datasets ###########################

if "dataset" not in config:
    config["dataset"] = None

raw_dataset = load_dataset("imagefolder", data_dir=os.path.join(data_folder, "train"))

feature_extractor = ViTFeatureExtractor(
    size=model_config["image_size"],
    image_mean=IMAGENET_DEFAULT_MEAN,
    image_std=IMAGENET_DEFAULT_STD
)

# transformations as done in original SimMIM paper
# source: https://github.com/microsoft/SimMIM/blob/main/data/data_simmim.py
transforms = Compose(
    [
        Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        RandomResizedCrop(model_config["image_size"], scale=(0.67, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
    ]
)

# create mask generator
mask_generator = MaskGenerator(
    input_size=model_config["image_size"],
    mask_patch_size=model_config["mask_patch_size"],
    model_patch_size=model_config["patch_size"],
    mask_ratio=0.6,
)

image_column_name = "image"

def preprocess_images(examples):
    """Preprocess a batch of images by applying transforms + creating a corresponding mask, indicating
    which patches to mask."""

    examples["pixel_values"] = [transforms(image) for image in examples[image_column_name]]
    examples["mask"] = [mask_generator() for i in range(len(examples[image_column_name]))]

    return examples

raw_dataset["train"].set_transform(preprocess_images)

########################### Loading Model ###########################

model = ModelForMaskedIM(model_config)

print(model)
print(f"parameter_size: {[weight.size() for weight in model.parameters()]}", flush = True)
print(f"num_parameter: {np.sum([np.prod(weight.size()) for weight in model.parameters()])}", flush = True)

model = model.cuda()
model = nn.DataParallel(model, device_ids = device_ids)

if "from_cp" in config:

    from_cp = config["from_cp"]
    checkpoint = torch.load(from_cp, map_location = 'cpu')

    missing_keys, unexpected_keys = model.module.load_state_dict(checkpoint['model_state_dict'], strict = False)
    print(f"missing_keys = {missing_keys}")
    print(f"unexpected_keys = {unexpected_keys}")
    print("Model initialized", from_cp, flush = True)

else:
    print("Model randomly initialized", flush = True)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = pretraining_config["learning_rate"],
    betas = (0.9, 0.999), eps = 1e-6, weight_decay = 0.05
)

lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer = optimizer,
    max_lr = pretraining_config["learning_rate"],
    pct_start = pretraining_config["warmup"],
    anneal_strategy = pretraining_config["anneal_strategy"],
    epochs = pretraining_config["epoch"],
    steps_per_epoch = pretraining_config["batches_per_epoch"]
)

amp_scaler = torch.cuda.amp.GradScaler() if model_config["mixed_precision"] else None

start_epoch = 0
inst_pass = 0
for epoch in reversed(range(pretraining_config["epoch"])):
    checkpoint_path = os.path.join(checkpoint_dir, f"cp-{epoch:04}.cp")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location = 'cpu')
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        start_epoch = checkpoint["epoch"] + 1
        inst_pass = checkpoint["inst_pass"]
        print("Model restored from", checkpoint_path)
        break

########################### Running Model ###########################

log_dir = os.path.join(root_path, "models", f"seq_len-{args.seq_len}", args.model, "logs")
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
log_f = open(os.path.join(log_dir, "pretrain.log"), "a+")

accumu_steps = utils.compute_accumu_step(pretraining_config["batch_size"], len(device_ids), gpu_config["inst_per_gpu"])
print("accumu_steps", accumu_steps)

model.train()

init_t = time.time()
for epoch in range(start_epoch, pretraining_config["epoch"]):

    data_loader = DataLoader(raw_dataset["train"], batch_size=pretraining_config["batch_size"], shuffle=True, collate_fn=collate_fn)
    pretrain_dataloader_iter = enumerate(data_loader)
    
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
