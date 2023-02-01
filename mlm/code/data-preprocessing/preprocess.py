import pickle
import os
from multiprocessing import Pool
from transformers import RobertaTokenizerFast

def tokenize(src_tgt_pair):
    src, tgt = src_tgt_pair

    if not os.path.exists(src):
        return

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    print("START", flush=True)
    with open(src, "r", encoding="utf=8") as read_f:
        text = read_f.read()
    print(f"Read {src}", flush=True)

    token_ids = tokenizer.encode(text)
    print(f"Encoded {src}", flush=True)
    # NOTE: same as the following
    # tokenize(text)
    # print("Tokenized {src}", flush=True)
    # del text

    # token_ids = tokenizer.convert_tokens_to_ids(tokens)
    # print("To Token IDs {src}", flush=True)

    with open(tgt, "wb") as dump_f:
        pickle.dump(token_ids, dump_f)
    print(f"Dumped {tgt}", flush=True)
    print("END", flush=True)


num_workers = 8
if num_workers > 1:
    pool = Pool(num_workers)

if not os.path.exists("./data"):
    os.mkdir("./data")
    
for SPLIT in ["valid", "test", "train"]:
    src_tgt_pairs = []

    for file_idx in range(100):
        file = f"data/wikitext-{SPLIT}-{file_idx:02}.txt"
        src_tgt_pairs.append((file, file.replace(".txt", "-roberta-base.pickle")))

        if len(src_tgt_pairs) == num_workers:
            if num_workers == 1:
                tokenize(src_tgt_pairs[0])
            else:
                pool.map(tokenize, src_tgt_pairs)
            src_tgt_pairs = []

if num_workers > 1:
    pool.close()
