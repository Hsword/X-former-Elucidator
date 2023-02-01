import torch
from torch.utils.data.dataset import Dataset
import os
import random
import json
import pickle
import numpy as np

class CorpusDataset(Dataset):
    def __init__(self, folder_path, file_json, option = None):

        with open(os.path.join(folder_path, file_json), "r") as f:
            files = json.load(f)

        self.files = [os.path.join(folder_path, file.split("/")[1]) for file in files]
        if option is not None:
            self.load_all_insts = option["load_all_insts"] if "load_all_insts" in option else True
            self.files_per_batch = option["files_per_batch"] if "files_per_batch" in option else 1024
            self.keep_prob = option["keep_prob"] if "keep_prob" in option else 0.1
            self.shuffle = option["shuffle"] if "shuffle" in option else False
        else:
            self.load_all_insts = True
            self.files_per_batch = 1024
            self.keep_prob = 0.1
            self.shuffle = True

        print(f"Number of Files: {len(self.files)}", flush = True)

        self.curr_idx = 0
        self.examples = []

    def load_files(self):

        if self.load_all_insts:
            if len(self.examples) == 0:
                for idx, file in enumerate(self.files):
                    print(f"Loading {idx} / {len(self.files)}: {file}", flush = True)
                    with open(file, "rb") as f:
                        self.examples.extend(pickle.load(f))

            self.curr_idx = 0
            if self.shuffle:
                random.shuffle(self.examples)
            print(f"Number of Instances: {len(self.examples)}", flush = True)
            print(f"Completed Loading", flush = True)
        else:
            del self.examples

            selected_files = np.random.choice(self.files, size = min(len(self.files), self.files_per_batch), replace = False)
            self.curr_idx = 0
            self.examples = []
            for idx, file in enumerate(selected_files):
                print(f"Loading {idx} / {len(selected_files)}: {file}", flush = True)
                with open(file, "rb") as f:
                    self.examples.extend([inst for inst in pickle.load(f) if random.random() < self.keep_prob])

            if self.shuffle:
                random.shuffle(self.examples)
            print(f"Number of Instances: {len(self.examples)}", flush = True)
            print(f"Completed Loading", flush = True)


    def __len__(self):
        return 100000000

    def __getitem__(self, i) -> torch.Tensor:
        if self.curr_idx >= len(self.examples):
            self.load_files()
        inst = self.examples[self.curr_idx]
        self.curr_idx += 1
        return torch.tensor(inst, dtype = torch.long)
