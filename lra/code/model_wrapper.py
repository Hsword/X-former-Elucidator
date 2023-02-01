import os
import sys

curr_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(curr_path)

import torch
import torch.nn as nn

def pooling(input, mode):
    if mode == "CLS":
        pooled = input[:, 0, :]
    elif mode == "MEAN":
        pooled = input.mean(dim = 1)
    else:
        raise Exception()
    return pooled


def append_cls(inp, mask, vocab_size):
    batch_size = inp.size(0)
    cls_id = ((vocab_size - 1) * torch.ones(batch_size, dtype = torch.long, device = inp.device)).long()
    cls_mask = torch.ones(batch_size, dtype = torch.float, device = mask.device)
    inp = torch.cat([cls_id[:, None], inp[:, :-1]], dim = -1)
    mask = torch.cat([cls_mask[:, None], mask[:, :-1]], dim = -1)
    return inp, mask


class SequenceClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.pooling_mode = config["pooling_mode"]
        self.ff = nn.Sequential(
            nn.Linear(config["hidden_dim"], config["ff_dim"]),
            nn.ReLU(),
            nn.Linear(config["ff_dim"], config["num_classes"])
        )

    def forward(self, input):
        seq_score = self.ff(pooling(input, self.pooling_mode))
        return seq_score

class ModelForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.enable_amp = config["mixed_precision"]
        self.pooling_mode = config["pooling_mode"]
        self.vocab_size = config["vocab_size"]

        self.attn_type = config["attn_type"]
        if self.attn_type.startswith("softmax"):
            from models.model import Model
            self.model = Model(config)
        elif self.attn_type.startswith("longformer"):
            from models.model_longformer import LongformerModel
            self.model = LongformerModel(config)
        elif self.attn_type.startswith("bigbird"):
            from models.model_bigbird import BigBirdModel
            self.model = BigBirdModel(config)
        elif self.attn_type.startswith("sparse"):
            from models.model_sparse import SparseModel
            self.model = SparseModel(config)
        elif self.attn_type.startswith("reformer"):
            from models.model_reformer import ReformerModel
            self.model = ReformerModel(config)
        elif self.attn_type.startswith("clustered"):
            from models.model_clustered import ClusteredModel
            self.model = ClusteredModel(config)        
        elif self.attn_type.startswith("linear"):
            from models.model_linear import LinearModel
            self.model = LinearModel(config)
        elif self.attn_type.startswith("cosformer"):
            from models.model_cosformer import CosformerModel
            self.model = CosformerModel(config)
        elif self.attn_type.startswith("performer"):
            from models.model_performer import PerformerModel
            self.model = PerformerModel(config)
        elif self.attn_type.startswith("transnormer"):
            from models.model_transnormer import TransNormerModel
            self.model = TransNormerModel(config)
        elif self.attn_type.startswith("linformer"):
            from models.model_linformer import LinformerModel
            self.model = LinformerModel(config)
        elif self.attn_type.startswith("nystrom"):
            from models.model_nystrom import NystromModel
            self.model = NystromModel(config)
        elif self.attn_type.startswith("synthesizer"):
            from models.model_synthesizer import SynthesizerModel
            self.model = SynthesizerModel(config)

        self.seq_classifer = SequenceClassificationHead(config)

        self.output_QK = "output_QK" in config and config["output_QK"]

    def forward(self, input_ids_0, mask_0, label):

        with torch.cuda.amp.autocast(enabled = self.enable_amp):

            if self.pooling_mode == "CLS":
                input_ids_0, mask_0 = append_cls(input_ids_0, mask_0, self.vocab_size)

            if self.output_QK:
                token_out, list_QK = self.model(input_ids_0, mask_0)
            else:
                token_out = self.model(input_ids_0, mask_0)
            seq_scores = self.seq_classifer(token_out)

            seq_loss = nn.CrossEntropyLoss(reduction = "none")(seq_scores, label)
            seq_accu = (seq_scores.argmax(dim = -1) == label).to(torch.float32)
            outputs = {}
            outputs["loss"] = seq_loss
            outputs["accu"] = seq_accu
            if self.output_QK:
                outputs["list_QK"] = list_QK

        return outputs


class SequenceClassificationHeadDual(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pooling_mode = config["pooling_mode"]
        self.ff = nn.Sequential(
            nn.Linear(config["hidden_dim"] * 4, config["ff_dim"]),
            nn.ReLU(),
            nn.Linear(config["ff_dim"], config["num_classes"])
        )

    def forward(self, input_0, input_1):
        X_0 = pooling(input_0, self.pooling_mode)
        X_1 = pooling(input_1, self.pooling_mode)
        seq_score = self.ff(torch.cat([X_0, X_1, X_0 * X_1, X_0 - X_1], dim = -1))
        return seq_score


class ModelForSequenceClassificationDual(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.enable_amp = config["mixed_precision"]
        self.pooling_mode = config["pooling_mode"]
        self.vocab_size = config["vocab_size"]

        self.attn_type = config["attn_type"]
        if self.attn_type.startswith("softmax"):
            from models.model import Model
            self.model = Model(config)
        elif self.attn_type.startswith("longformer"):
            from models.model_longformer import LongformerModel
            self.model = LongformerModel(config)
        elif self.attn_type.startswith("bigbird"):
            from models.model_bigbird import BigBirdModel
            self.model = BigBirdModel(config)
        elif self.attn_type.startswith("sparse"):
            from models.model_sparse import SparseModel
            self.model = SparseModel(config)
        elif self.attn_type.startswith("reformer"):
            from models.model_reformer import ReformerModel
            self.model = ReformerModel(config)
        elif self.attn_type.startswith("clustered"):
            from models.model_clustered import ClusteredModel
            self.model = ClusteredModel(config)        
        elif self.attn_type.startswith("linear"):
            from models.model_linear import LinearModel
            self.model = LinearModel(config)
        elif self.attn_type.startswith("cosformer"):
            from models.model_cosformer import CosformerModel
            self.model = CosformerModel(config)
        elif self.attn_type.startswith("performer"):
            from models.model_performer import PerformerModel
            self.model = PerformerModel(config)
        elif self.attn_type.startswith("transnormer"):
            from models.model_transnormer import TransNormerModel
            self.model = TransNormerModel(config)
        elif self.attn_type.startswith("linformer"):
            from models.model_linformer import LinformerModel
            self.model = LinformerModel(config)
        elif self.attn_type.startswith("nystrom"):
            from models.model_nystrom import NystromModel
            self.model = NystromModel(config)
        elif self.attn_type.startswith("synthesizer"):
            from models.model_synthesizer import SynthesizerModel
            self.model = SynthesizerModel(config)

        self.seq_classifer = SequenceClassificationHeadDual(config)

        self.output_QK = "output_QK" in config and config["output_QK"]

    def forward(self, input_ids_0, input_ids_1, mask_0, mask_1, label):

        with torch.cuda.amp.autocast(enabled = self.enable_amp):

            if self.pooling_mode == "CLS":
                input_ids_0, mask_0 = append_cls(input_ids_0, mask_0, self.vocab_size)
                input_ids_1, mask_1 = append_cls(input_ids_1, mask_1, self.vocab_size)

            if self.output_QK:
                token_out_0, list_QK_0 = self.model(input_ids_0, mask_0)
                token_out_1, list_QK_1 = self.model(input_ids_1, mask_1)
            else:
                token_out_0 = self.model(input_ids_0, mask_0)
                token_out_1 = self.model(input_ids_1, mask_1)
            seq_scores = self.seq_classifer(token_out_0, token_out_1)

            seq_loss = nn.CrossEntropyLoss(reduction = "none")(seq_scores, label)
            seq_accu = (seq_scores.argmax(dim = -1) == label).to(torch.float32)
            outputs = {}
            outputs["loss"] = seq_loss
            outputs["accu"] = seq_accu
            if self.output_QK:
                outputs["list_QK"] = list_QK_0 + list_QK_1

        return outputs
