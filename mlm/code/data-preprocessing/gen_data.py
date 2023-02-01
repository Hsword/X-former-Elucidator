import os
import random
import pickle
from transformers import RobertaTokenizerFast
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seq_len", type=int, help="seq len", dest="seq_len", default=512)
args = parser.parse_args()

dump_folder = f"../datasets/{args.seq_len}-roberta"
if not os.path.exists(dump_folder):
    os.mkdir(dump_folder)

def process(dataset, SPLIT, tokenizer):
    data_folder = "./data"
    
    max_seq_len = args.seq_len
    per_batch_inst = 512 * 2048 // max_seq_len
    block_size = max_seq_len - tokenizer.num_special_tokens_to_add(pair = False)

    random.seed(hash(dataset))

    files = sorted([os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.endswith(".pickle") and file.find(SPLIT) != -1])
    data_buffer = []

    batch = []
    file_idx = 0

    for file in files:
        print(file)
        with open(file, "rb")as read_f:
            data = pickle.load(read_f)
        data_buffer.extend(data)
        for start_idx in range(0, len(data_buffer) - block_size + 1, block_size):
            block = data_buffer[start_idx: (start_idx + block_size)]
            assert len(block) == block_size

            block = tokenizer.build_inputs_with_special_tokens(block)
            assert len(block) == max_seq_len
            
            batch.append(block)
            if len(batch) == per_batch_inst:
                dump_path = os.path.join(dump_folder, f"{dataset}-{SPLIT}-{file_idx:03}.pickle")
                with open(dump_path, "wb") as dump_f:
                    pickle.dump(batch, dump_f)
                batch = []
                file_idx += 1
                print(f"Dumped {dump_path}", flush=True)
        
        data_buffer = data_buffer[(start_idx + block_size):]

    dump_path = os.path.join(dump_folder, f"{dataset}-{SPLIT}-{file_idx:03}.pickle")
    with open(dump_path, "wb") as dump_f:
        pickle.dump(batch, dump_f)
    batch = []
    file_idx += 1
    print(f"Dumped {dump_path}")


import json

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
for SPLIT in ["valid", "test", "train"]:
    process("wikitext", SPLIT, tokenizer)
    
    files = sorted([os.path.join(dump_folder, file) for file in os.listdir(dump_folder) if file.endswith(".pickle") and file.find(SPLIT) != -1])
    print(json.dumps(files, indent=4))
    random.seed(1)
    random.shuffle(files)

    if SPLIT is "valid":
        with open(os.path.join(dump_folder, "dev.json"), "w") as f:
            json.dump(files, f, indent=4)
    elif SPLIT is "test":
        with open(os.path.join(dump_folder, "test.json"), "w") as f:
            json.dump(files, f, indent=4)
    elif SPLIT is "train":
        with open(os.path.join(dump_folder, "train.json"), "w") as f:
            json.dump(files, f, indent=4)
