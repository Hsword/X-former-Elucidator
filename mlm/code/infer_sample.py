from model_wrapper import ModelForMaskedLM
import torch
import os
import json
import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, help = "model", dest = "model", required = True)
args = parser.parse_args()

curr_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.join(curr_path, os.pardir)

with open(os.path.join(root_path, "models", args.model, "config.json"), "r") as f:
    config = json.load(f)

model_config = config["model"]
pretraining_config = config["pretraining_setting"]
gpu_config = config["gpu_setting"]
checkpoint_dir = os.path.join(root_path, "models", args.model, "model")

device_ids = list(range(torch.cuda.device_count()))

######################### Tokenize Inference ########################

inf_samples = [
    'He had a recurring role in ten episodes of the television series Casualty in 2010 , as " Kieron Fletcher" .',
    'Traditional Chinese literary criticism emphasized the life of the author when interpreting a work , a practice which Burton Watson attributes to " the close links that traditional Chinese thought posits between art and morality " .',
    'The filters were designed by Campbell for the purpose of separating multiplexed telephone channels on transmission lines , but their subsequent use has been much more widespread than that .'
]

tokenizer = utils.get_tokenizer(model_config["max_seq_len"])
model_config["vocab_size"] = len(tokenizer.get_vocab())

token_ids = []
for inst in [inf_samples[2]]:
    token_ids.append(tokenizer.encode(inst))

# print(token_ids)

########################### Loading Model ###########################

model_config["print_attn"] = True
model = ModelForMaskedLM(model_config)

# model = model.cuda()

########################### Running Model ###########################

checkpoint_path = os.path.join(checkpoint_dir, "cp-0189.model")
checkpoint = torch.load(checkpoint_path, map_location = 'cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model = model.model

model.eval()
with torch.no_grad():
    model(torch.tensor(token_ids))
