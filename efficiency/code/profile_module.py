import os
import sys

curr_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(curr_path)

from torch.utils.data import DataLoader
from torch import cuda
import torch
import copy
import json
import numpy as np
from model.model import Model as ModelForModuleProfiling


model_config = {
    "vocab_size": 512,
    "max_seq_len": 512,
    "num_sen_type": 1,
    "embedding_dim": 768,
    "hidden_dim": 768,
    "ff_dim": 2048,
    "num_layers": 1,
    "num_head": 8,
    "head_dim": 64,
    "dropout_prob": 0.1,
    "attn_type": "softmax",
    "mode": "normal"
}


def eval(model, batch_size, seq_len, inputs=None):
    if inputs is None:
        input_ids = torch.randint(0, 512, (batch_size, seq_len)).long().cuda()
        with torch.no_grad():
            out = model(input_ids)
    else:
        for key in inputs:
            inputs[key] = inputs[key].cuda()
        out = model(**inputs)

num_iter = 20

results = dict()
for log_seq_len in range(7, 14):
    seq_len = int(2 ** log_seq_len)

    config = copy.deepcopy(model_config)
    config["max_seq_len"] = seq_len
    config["mode"] = "normal"

    batch_size = 1
    # model = ModelForModuleProfiling(config).cuda()
    try:
        cuda.reset_peak_memory_stats()
        model = ModelForModuleProfiling(config).cuda()
        model.eval()
        while True:
            eval(model, batch_size, seq_len)
            batch_size = batch_size * 2
    except Exception as e:
        if not str(e).startswith("CUDA out of memory"):
            print(e)
    finally:
        del model
        cuda.empty_cache()
        cuda.reset_accumulated_memory_stats()
    
    if batch_size == 1:
        print(f"seq_len {seq_len} is too long for batch size 1!")
        continue

    for _ in range(2):
        if batch_size > 1:
            batch_size = batch_size // 2
    print(f"seq len = {seq_len}, batch size = {batch_size}")
    
    for mode in ["normal", "no msa", "no ffn", "no msa and ffn", "no block"]:
        print("START", mode)

        if mode not in results:
            results[mode] = dict()
        input_ids = torch.randint(0, 512, (num_iter * batch_size, seq_len)).long()
        dataset = [{"input_ids": input_ids[item]} for item in range(num_iter * batch_size)]
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        dataloader_iter = enumerate(data_loader)

        cuda.reset_peak_memory_stats()

        config["mode"] = mode
        model = ModelForModuleProfiling(config).cuda()
        model.eval()
        eval(model, batch_size, seq_len)

        start = cuda.Event(enable_timing=True)
        end = cuda.Event(enable_timing=True)
        time_list = []
        for _ in range(num_iter):
            cuda.synchronize()
            start.record()
            _, inputs = next(dataloader_iter)
            eval(model, batch_size, seq_len, inputs)
            end.record()
            cuda.synchronize()
            time_list.append(start.elapsed_time(end) / batch_size)
        
        peak_memory_all = cuda.max_memory_allocated()
        peak_memory_data = batch_size * seq_len * 8
        peak_memory_model = peak_memory_all - peak_memory_data

        per_inst_time_avg = np.mean(time_list)
        per_inst_time_std = np.std(time_list)
        memory_per_inst = peak_memory_model / batch_size / 1024 / 1024

        results[mode][seq_len] = {
            "batch_size": batch_size,
            "per_inst_time_avg (ms)": round(per_inst_time_avg, 3),
            "per_inst_time_std (ms)": round(per_inst_time_std, 3),
            "memory_per_inst (MB)": round(memory_per_inst, 3),
        }
        
        print(results[mode][seq_len])

        del model
        cuda.empty_cache()

        print("END", mode)

with open(os.path.join(root_path, "module_result", "accurate_module_result.json"), "w") as f:
    json.dump(results, f, indent=2)
