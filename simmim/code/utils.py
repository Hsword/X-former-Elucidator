import torch
import math

def compute_accumu_step(batch_size, num_gpus, inst_per_gpu):
    size_per_gpu = batch_size // num_gpus
    return max(int(math.ceil(size_per_gpu / inst_per_gpu)), 1)

def backward(loss, amp_scaler):
    if amp_scaler is None:
        loss.backward()
    else:
        amp_scaler.scale(loss).backward()

def optimizer_step(optimizer, lr_scheduler, amp_scaler):
    if amp_scaler is None:
        optimizer.step()
    else:
        amp_scaler.step(optimizer)
        amp_scaler.update()
    lr_scheduler.step()

def add_output_to_summary(outputs, summary):
    for key in outputs:
        if key not in summary:
            summary[key] = 0
        summary[key] = round(summary[key] + outputs[key].data.item(), 6)

def partition_inputs(inputs, num_partitions, to_cuda):
    if to_cuda:
        for key in inputs:
            inputs[key] = inputs[key].cuda()

    inputs_list = [[None, {}] for _ in range(num_partitions)]
    valid_size = None
    batch_size = None

    for key in inputs:

        if batch_size is None:
            batch_size = inputs[key].size(0)
        else:
            assert batch_size == inputs[key].size(0)

        inps = torch.chunk(inputs[key], num_partitions, dim = 0)

        if valid_size is None:
            valid_size = len(inps)
        else:
            assert valid_size == len(inps)

        for idx in range(valid_size):
            inputs_list[idx][1][key] = inps[idx]
            if inputs_list[idx][0] is None:
                inputs_list[idx][0] = inps[idx].size(0) / batch_size
            else:
                assert inputs_list[idx][0] == inps[idx].size(0) / batch_size

    return inputs_list[:valid_size]
