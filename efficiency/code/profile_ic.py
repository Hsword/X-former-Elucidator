import os
import sys

curr_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(curr_path))
sys.path.append(root_path)

from simmim.code.model_wrapper import ModelForImageClassification
import torch
import torch.nn as nn
import torch.cuda as cuda
from torch.utils.data import DataLoader
import copy
import json
import numpy as np


model_config = {
    "mixed_precision": False,
    "attention_grad_checkpointing": False,
    "gelu_grad_checkpointing": False,

    "embedding_dim": 512,
    "hidden_dim": 512,
    "ff_dim": 2048,
    "num_channels": 3,
    "image_size": 64,
    "patch_size": 4,
    "mask_patch_size": 8,
    "encoder_stride": 4,
    "use_mask_token": True,
    "num_layers": 8,
    "dropout_prob": 0.1,
    "num_head": 8,
    "head_dim": 64,

    "num_classes":10,
}
training_config = {
    "batch_size":64,
    "learning_rate":2e-4,
    "warmup":0.1,
    "inst_per_gpu": 64
}
attn_config = {
    "softmax":{},
    "longformer-128":{"window_size":128, "first_token_view":True},
    "bigbird-128":{"block_size":16, "num_random_blocks":1},
    "sparse-64-16":{"block_size":64, "stride_c":15, "use_cls_token":True, "is_decoder":False},
    "sparse-64-9":{"block_size":64, "stride_c":8, "use_cls_token":True, "is_decoder":False},
    "sparse-64-4":{"block_size":64, "stride_c":3, "use_cls_token":True, "is_decoder":False},
    "sparse-64-2":{"block_size":64, "stride_c":1, "use_cls_token":True, "is_decoder":False},
    "sparse-64-1":{"block_size":64, "stride_c":0, "use_cls_token":True, "is_decoder":False},
    "reformer-2-32":{"num_hash":2, "chunk_len":32},
    "clustered-100-16-32":{"num_clusters":100, "topk":16, "bits":32},
    "linear":{},
    "cosformer-elu":{"act_fun":"elu", "reweighting":True},
    "performer-relu-128":{"rp_dim":128, "generalized_attention":True, "kernel_type":"relu"},
    "linformer-128":{"linformer_k":128},
    "nystrom-32-13":{"num_landmarks":32, "conv_kernel_size":13},
    "synthesizer-FR-8":{"synthesizer_mode":"factorized_random", "factor_k": 8},
    "transnormer-128":{"block_size": 128, "kernel_type": "1+elu", "attn_norm_type": "simple_rms_norm"},
    "transnormer-128-droplast":{"block_size": 128, "kernel_type": "1+elu", "attn_norm_type": "simple_rms_norm", "drop_last": False}
}
len2sparse = {
    256: "sparse-64-16",
    512: "sparse-64-9",
    1024: "sparse-64-4",
    2048: "sparse-64-2",
    4096: "sparse-64-1",
    8192: "sparse-128",
    16384: "sparse-128",
    32768: "sparse-128"
}


def train(model, batch_size, image_size, inputs=None):
    if inputs is None:
        pixel_values = (torch.rand((batch_size, 3, image_size, image_size)) * 4. - 2.).float().cuda()
        labels = torch.randint(0, 10, (batch_size,)).long().cuda()
        out = model(pixel_values, labels)
    else:
        for key in inputs:
            inputs[key] = inputs[key].cuda()
        out = model(**inputs)
    out["loss"].mean().backward()


def profile_train(attn_type):
    print(attn_type, "train")
    results = dict()

    config = copy.deepcopy(model_config)
    seq_len = 256
    image_size = model_config["image_size"]
    if attn_type.startswith("sparse"):
        attn_type = len2sparse[seq_len]
    config['attn_type'] = attn_type
    # config["max_seq_len"] = seq_len
    config.update(attn_config[attn_type])

    batch_size = training_config["batch_size"]
    inst_per_gpu = training_config["inst_per_gpu"]
    device_ids = list(range(cuda.device_count()))
    accumu_steps = max(batch_size // cuda.device_count() // inst_per_gpu, 1)
    print(f"accumulate steps: {accumu_steps}")

    num_iter = 20
    # random generated data
    pixel_values = (torch.rand((num_iter * batch_size, 3, image_size, image_size)) * 4. - 2.).float()
    labels = torch.randint(0, 10, (num_iter * batch_size,)).long()
    dataset = [
        {"pixel_values":pixel_values[item], "labels":labels[item]} for item in range(num_iter * batch_size)
    ]
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    dataloader_iter = enumerate(data_loader)
    cuda.reset_peak_memory_stats()

    model = ModelForImageClassification(config).cuda()
    model.train()
    train(model, batch_size, image_size)

    start = cuda.Event(enable_timing=True)
    end = cuda.Event(enable_timing=True)
    time_list = []
    for _ in range(num_iter):
        _, inputs = next(dataloader_iter)
        cuda.synchronize()
        start.record()
        train(model, batch_size, image_size, inputs)
        end.record()
        cuda.synchronize()
        time_list.append(start.elapsed_time(end) / inst_per_gpu)

    peak_memory_all = cuda.max_memory_allocated()
    peak_memory_data = inst_per_gpu * (3 * image_size * image_size * 4 + 8)
    peak_memory_model = peak_memory_all - peak_memory_data

    per_inst_time_avg = np.mean(time_list)
    per_inst_time_std = np.std(time_list)
    memory_per_inst = peak_memory_model / inst_per_gpu / 1024 / 1024

    results = {
        "batch_size":batch_size,
        "inst_per_gpu":inst_per_gpu,
        "per_inst_time_avg (ms)":round(per_inst_time_avg, 3),
        "per_inst_time_std (ms)":round(per_inst_time_std, 3),
        "memory_per_inst (MB)":round(memory_per_inst, 3),
    }

    print(results)

    del model
    cuda.empty_cache()
    
    with open(os.path.join(root_path, "efficiency","cv_results", f"{attn_type}_ic.json"), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    models = ["softmax", "longformer-128", "bigbird-128", "sparse", "reformer-2-32", "clustered-100-16-32", "linear", "cosformer-elu", "performer-relu-128", "linformer-128", "nystrom-32-13", "synthesizer-FR-8", "transnormer-128"]
    for model in ["transnormer-128"]:
        profile_train(model)