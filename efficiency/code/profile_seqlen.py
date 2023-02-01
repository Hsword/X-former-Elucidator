import os
import sys

curr_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(curr_path))
sys.path.append(root_path)

from mlm.code.model_wrapper import ModelForSequenceClassification
import torch
import torch.cuda as cuda
from torch.utils.data import DataLoader
import copy
import time
import json
import numpy as np


model_config = {
    "mixed_precision": False,
    "attention_grad_checkpointing": False,
    "gelu_grad_checkpointing": False,

    "vocab_size": 512,
    "num_sen_type": 1,
    "max_seq_len": 512,

    "embedding_dim": 256,
    "hidden_dim": 256,
    "ff_dim": 1024,
    "num_layers": 4,
    "num_head": 4,
    "head_dim": 64,
    "dropout_prob": 0.1,
    "num_classes": 2,
    "use_cls_token": True
}
vocab_size = model_config["vocab_size"]

attn_config = {
    "softmax":{},
    "longformer-128":{"window_size":128, "first_token_view":True},
    "bigbird-128":{"block_size":16, "num_random_blocks":1},
    "sparse-64-9":{"block_size":64, "stride_c":8, "use_cls_token":True, "is_decoder":False},
    "sparse-64-4":{"block_size":64, "stride_c":3, "use_cls_token":True, "is_decoder":False},
    "sparse-64-2":{"block_size":64, "stride_c":1, "use_cls_token":True, "is_decoder":False},
    "sparse-64-1":{"block_size":64, "stride_c":0, "use_cls_token":True, "is_decoder":False},
    "sparse-128":{"block_size":128, "stride_c":-1, "use_cls_token":True, "is_decoder":False},
    "reformer-2-32":{"num_hash":2, "chunk_len":32},
    "clustered-100-16-32":{"num_clusters":100, "topk":16, "bits":32},
    "linear":{},
    "cosformer":{"act_fun":"relu", "reweighting":True},
    "performer-relu-128":{"rp_dim":128, "generalized_attention":True, "kernel_type":"relu"},
    "linformer-128":{"linformer_k":128},
    "nystrom-32-27":{"num_landmarks":32, "conv_kernel_size":27},
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
    32768: "sparse-128",
}

def train(model, batch_size, seq_len, inputs=None):
    if inputs is None:
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).long().cuda()
        attention_mask = torch.ones(batch_size, seq_len).float().cuda()
        labels = torch.randint(0, 2, (batch_size, )).long().cuda()
        out = model(input_ids, attention_mask, labels)
    else:
        for key in inputs:
            inputs[key] = inputs[key].cuda()
        out = model(**inputs)
    out["loss"].mean().backward()


def eval(model, batch_size, seq_len, inputs=None):
    if inputs is None:
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).long().cuda()
        attention_mask = torch.ones(batch_size, seq_len).float().cuda()
        labels = torch.randint(0, 2, (batch_size, )).long().cuda()
        with torch.no_grad():
            out = model(input_ids, attention_mask, labels)
    else:
        for key in inputs:
            inputs[key] = inputs[key].cuda()
        with torch.no_grad():
            out = model(**inputs)


num_iter = 20

def profile_seqlen(attn_type, mode, fn):
    print(attn_type, mode)
    results = dict()

    for log_seq_len in reversed(range(9, 16)):
        seq_len = int(2 ** log_seq_len)

        config = copy.deepcopy(model_config)
        if attn_type.startswith("sparse"):
            assert seq_len in len2sparse
            attn_type = len2sparse[seq_len]
        config["attn_type"] = attn_type
        config["max_seq_len"] = seq_len
        config.update(attn_config[attn_type])
        # vocab_size = model_config["vocab_size"]

        batch_size = 1
        
        try:
            cuda.reset_peak_memory_stats()
            model = ModelForSequenceClassification(config).cuda()
            if mode == "train":
                model.train()
            elif mode == "eval":
                model.eval()
            else:
                assert False

            while (True):
                fn(model, batch_size, seq_len)
                batch_size = batch_size * 2
        except Exception as e:
            if not str(e).startswith("CUDA out of memory"):
                print(e)
        finally:
            del model
            cuda.empty_cache()
            cuda.reset_peak_memory_stats()
        
        if batch_size == 1:
            print(f"seq_len {seq_len} is too long for batch size 1!")
            continue

        for _ in range(2):
            if batch_size > 1:
                batch_size = batch_size // 2

        input_ids = torch.randint(0, vocab_size, (num_iter * batch_size, seq_len)).long()
        attention_mask = torch.ones(num_iter * batch_size, seq_len).float()
        labels = torch.randint(0, 2, (num_iter * batch_size, )).long()
        dataset = [
            {"input_ids":input_ids[item], "attention_mask": attention_mask[item], "labels":labels[item]} for item in range(num_iter * batch_size)
        ]
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        dataloader_iter = enumerate(data_loader)

        cuda.reset_peak_memory_stats()

        model = ModelForSequenceClassification(config).cuda()
        if mode == "train":
            model.train()
        elif mode == "eval":
            model.eval()
        else:
            assert False
        fn(model, batch_size, seq_len)

        start = cuda.Event(enable_timing=True)
        end = cuda.Event(enable_timing=True)
        time_list = []
        for _ in range(num_iter):
            _, inputs = next(dataloader_iter)
            cuda.synchronize()
            start.record()
            # fn(model, batch_size, seq_len)
            fn(model, batch_size, seq_len, inputs)
            end.record()
            cuda.synchronize()
            time_list.append(start.elapsed_time(end) / batch_size)

        peak_memory_all = cuda.max_memory_allocated()
        peak_memory_data = batch_size * (seq_len * (8 + 4) + 8)
        peak_memory_model = peak_memory_all - peak_memory_data

        per_inst_time_avg = np.mean(time_list)
        per_inst_time_std = np.std(time_list)
        memory_per_inst = peak_memory_model / batch_size / 1024 / 1024
        if per_inst_time_std > 1.:
            print("ALERT: time deviation too large!")

        results[seq_len] = {
            "mode":mode,
            "seq_len":seq_len,
            "batch_size":batch_size,
            "per_inst_time_avg (ms)":round(per_inst_time_avg, 3),
            "per_inst_time_std (ms)":round(per_inst_time_std, 3),
            "memory_per_inst (MB)":round(memory_per_inst, 3),
        }

        print(results[seq_len])

        del model
        cuda.empty_cache()

        file_attn_type = attn_type if not attn_type.startswith("sparse") else "sparse-128"
        with open(os.path.join(root_path, "efficiency", mode, f"{file_attn_type}.json"), 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    models = ["softmax", "longformer-128", "bigbird-128", "sparse", "reformer-2-32", "clustered-100-16-32", "linear", "cosformer", "performer-relu-128", "linformer-128", "nystrom-32-27", "synthesizer-FR-8", "transnormer-128"]
    for model in models:
        profile_seqlen(attn_type=model, mode="train", fn=train)
        profile_seqlen(attn_type=model, mode="eval", fn=eval)
