import os
import sys

curr_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(curr_path))
sys.path.append(root_path)

from lra.code.model_wrapper import ModelForSequenceClassification
from torch.utils.data import DataLoader
from torch import cuda
import torch
import copy
import json
import numpy as np


model_config = {
    "embedding_dim":128,
    "hidden_dim":128,
    "ff_dim":256,
    "head_dim":64,
    "num_head":2,
    "num_layers":2,
    "vocab_size":512,
    "max_seq_len":1024,
    "dropout_prob":0.1,
    "pooling_mode":"MEAN",
    "num_classes": 2,
    "mixed_precision": False,
    "use_cls_token": True
}
training_config = {
    "batch_size":16,
    "learning_rate":5e-5,
    "warmup":175,
    "lr_decay":"cos",
    "weight_decay":0,
    "inst_per_gpu": 16
}
attn_config = {
    "softmax":{},
    "longformer-128":{"window_size":128, "first_token_view":False},
    "bigbird-128":{"block_size":16, "num_random_blocks":1},
    "sparse-64-4":{"block_size":64, "stride_c":3, "use_cls_token":True, "is_decoder":False},
    "sparse-64-2":{"block_size":64, "stride_c":1, "use_cls_token":True, "is_decoder":False},
    "sparse-64-1":{"block_size":64, "stride_c":0, "use_cls_token":True, "is_decoder":False},
    "reformer-2-32":{"num_hash":2, "chunk_len":32},
    "clustered-100-16-32":{"num_clusters":100, "topk":16, "bits":32},
    "linear":{},
    "cosformer":{"act_fun":"relu", "reweighting":True},
    "performer-relu-128":{"rp_dim":128, "generalized_attention":True, "kernel_type":"relu"},
    "linformer-128":{"linformer_k":128},
    "nystrom-32-35":{"num_landmarks":32, "conv_kernel_size":35},
    "synthesizer-FR-8":{"synthesizer_mode":"factorized_random", "factor_k": 8},
    "transnormer-128":{"block_size": 128, "kernel_type": "1+elu", "attn_norm_type": "simple_rms_norm"}
}
len2sparse = {
    512: "sparse-64-9",
    1024: "sparse-64-4",
    2048: "sparse-64-2",
    4096: "sparse-64-1",
    8192: "sparse-128",
    16384: "sparse-128",
    32768: "sparse-128"
}


def train(model, batch_size, seq_len, inputs=None):
    if inputs is None:
        input_ids_0 = torch.randint(0, 256, (batch_size, seq_len)).long().cuda()
        label = torch.randint(0, 2, (batch_size, )).long().cuda()
        mask_0 = torch.ones(batch_size, seq_len).float().cuda()
        out = model(input_ids_0, mask_0, label)
    else:
        for key in inputs:
            inputs[key] = inputs[key].cuda()
        out = model(**inputs)
    out["loss"].mean().backward()


def profile_train(attn_type):
    print(attn_type, "train")
    results = dict()

    config = copy.deepcopy(model_config)
    for seq_len in [1024, 2048, 4096]:
        if attn_type.startswith("sparse"):
            attn_type = len2sparse[seq_len]
        config['attn_type'] = attn_type
        config["max_seq_len"] = seq_len
        config.update(attn_config[attn_type])

        batch_size = training_config["batch_size"]
        inst_per_gpu = training_config["inst_per_gpu"]
        accumu_steps = max(training_config["batch_size"] // cuda.device_count() // inst_per_gpu, 1)
        print(f"accumulate steps: {accumu_steps}")

        num_iter = 20
        input_ids_0 = torch.randint(0, 512, (num_iter * batch_size, seq_len)).long()
        label = torch.randint(0, 2, (num_iter * batch_size,)).long()
        mask_0 = torch.ones(num_iter * batch_size, seq_len).float()
        dataset = [
            {"input_ids_0":input_ids_0[item], "mask_0": mask_0[item], "label":label[item]} for item in range(num_iter * batch_size)
        ]
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        dataloader_iter = enumerate(data_loader)

        cuda.reset_peak_memory_stats()

        model = ModelForSequenceClassification(config).cuda()
        model.train()
        train(model, batch_size, seq_len)

        start = cuda.Event(enable_timing=True)
        end = cuda.Event(enable_timing=True)
        time_list = []
        for _ in range(num_iter):
            _, inputs = next(dataloader_iter)
            cuda.synchronize()
            start.record()
            train(model, batch_size, seq_len, inputs)
            end.record()
            cuda.synchronize()
            time_list.append(start.elapsed_time(end) / batch_size)

        peak_memory_all = cuda.max_memory_allocated()
        peak_memory_data = batch_size * (seq_len * (8 + 4) + 8)
        peak_memory_model = peak_memory_all - peak_memory_data

        per_inst_time_avg = np.mean(time_list)
        per_inst_time_std = np.std(time_list)
        memory_per_inst = peak_memory_model / batch_size / 1024 / 1024

        results[seq_len] = {
            "batch_size":batch_size,
            "per_inst_time_avg (ms)":round(per_inst_time_avg, 3),
            "per_inst_time_std (ms)":round(per_inst_time_std, 3),
            "memory_per_inst (MB)":round(memory_per_inst, 3),
        }

        print(results[seq_len])

        del model
        cuda.empty_cache()
    
    with open(os.path.join(root_path, "efficiency","lra_results", f"{attn_type}_mean.json"), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    models = ["softmax", "longformer-128", "bigbird-128", "sparse", "reformer-2-32", "clustered-100-16-32", "linear", "cosformer", "performer-relu-128", "linformer-128", "nystrom-32-35", "synthesizer-FR-8", "transnormer-128"]
    for model in ["transnormer-128"]:
        profile_train(model)