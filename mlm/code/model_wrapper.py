import os
import sys

curr_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(curr_path)

import torch
import torch.nn as nn
from models.model import Approx_GeLU

class MLMHead(nn.Module):
    def __init__(self, config, embeddings):
        super().__init__()

        self.dense = nn.Linear(config["hidden_dim"], config["embedding_dim"])
        self.act = Approx_GeLU(config)
        self.norm = nn.LayerNorm(config["embedding_dim"])
        self.mlm_class = nn.Linear(config["embedding_dim"], config["vocab_size"])
        self.mlm_class.weight = embeddings.weight

    def forward(self, X):

        X = self.act(self.dense(X))
        X = self.norm(X)
        scores = self.mlm_class(X)

        return scores


class ModelForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.enable_amp = config["mixed_precision"]

        self.vocab_size = config["vocab_size"]
        
        self.attn_type = config["attn_type"]
        if self.attn_type.startswith("softmax"):
            from models.model import Model
            self.model = Model(config)
        elif self.attn_type.startswith("linformer"):
            from models.model_linformer import LinformerModel
            self.model = LinformerModel(config)
        elif self.attn_type.startswith("cosformer"):
            from models.model_cosformer import CosformerModel
            self.model = CosformerModel(config)
        elif self.attn_type.startswith("linear"):
            from models.model_linear import LinearModel
            self.model = LinearModel(config)
        elif self.attn_type.startswith("bigbird"):
            from models.model_bigbird import BigBirdModel
            self.model = BigBirdModel(config)
        elif self.attn_type.startswith("performer"):
            from models.model_performer import PerformerModel
            self.model = PerformerModel(config)
        elif self.attn_type.startswith("reformer"):
            from models.model_reformer import ReformerModel
            self.model = ReformerModel(config)
        elif self.attn_type.startswith("sparse"):
            from models.model_sparse import SparseModel
            self.model = SparseModel(config)
        elif self.attn_type.startswith("clustered"):
            from models.model_clustered import ClusteredModel
            self.model = ClusteredModel(config)
        elif self.attn_type.startswith("synthesizer"):
            from models.model_synthesizer import SynthesizerModel
            self.model = SynthesizerModel(config)
        elif self.attn_type.startswith("longformer"):
            from models.model_longformer import LongformerModel
            self.model = LongformerModel(config)
        elif self.attn_type.startswith("nystrom"):
            from models.model_nystrom import NystromModel
            self.model = NystromModel(config)
        elif self.attn_type.startswith("transnormer"):
            from models.model_transnormer import TransNormerModel
            self.model = TransNormerModel(config)
        
        self.mlm = MLMHead(config, self.model.embeddings.word_embeddings)

    def forward(self, input_ids, labels = None):

        with torch.cuda.amp.autocast(enabled = self.enable_amp):
            token_out = self.model(input_ids)

            mlm_scores = self.mlm(token_out)

            label_mask = (labels != -100).float()
            valid_count = torch.sum(label_mask) + 1e-6
            batch_size = torch.tensor(mlm_scores.size(0), dtype = torch.float, device = mlm_scores.device)

            mlm_loss_fct = nn.CrossEntropyLoss(reduction = "none")
            mlm_loss = mlm_loss_fct(mlm_scores.reshape(-1, self.vocab_size), labels.reshape(-1))
            mlm_loss = torch.sum(mlm_loss * label_mask.reshape(-1)) / valid_count

            mlm_correct = (mlm_scores.argmax(dim = -1) == labels).to(torch.float32)
            mlm_accu = torch.sum(mlm_correct * label_mask) / valid_count

            outputs = {
                "loss":mlm_loss, "mlm_accu":mlm_accu,
                "valid_count":valid_count, "batch_size_per_device":batch_size
            }

            outputs = {key:value[None] for key, value in outputs.items()}

        return outputs


class SequenceClassificationHead(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.dropout_0 = nn.Dropout(config["dropout_prob"])
        self.dense = nn.Linear(config["hidden_dim"], config["embedding_dim"])
        self.dropout_1 = nn.Dropout(config["dropout_prob"])
        self.out_proj = nn.Linear(config["embedding_dim"], config["num_classes"])

    def forward(self, X):

        sen_out = X[:, 0, :]
        sen_score = self.out_proj(self.dropout_1(torch.tanh(self.dense(self.dropout_0(sen_out)))))

        return sen_score


class ModelForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.enable_amp = config["mixed_precision"]

        self.vocab_size = config["vocab_size"]
        self.num_classes = config["num_classes"]

        self.attn_type = config["attn_type"]
        if self.attn_type.startswith("softmax"):
            from models.model import Model
            self.model = Model(config)
        elif self.attn_type.startswith("linformer"):
            from models.model_linformer import LinformerModel
            self.model = LinformerModel(config)
        elif self.attn_type.startswith("cosformer"):
            from models.model_cosformer import CosformerModel
            self.model = CosformerModel(config)
        elif self.attn_type.startswith("linear"):
            from models.model_linear import LinearModel
            self.model = LinearModel(config)
        elif self.attn_type.startswith("bigbird"):
            from models.model_bigbird import BigBirdModel
            self.model = BigBirdModel(config)
        elif self.attn_type.startswith("performer"):
            from models.model_performer import PerformerModel
            self.model = PerformerModel(config)
        elif self.attn_type.startswith("reformer"):
            from models.model_reformer import ReformerModel
            self.model = ReformerModel(config)
        elif self.attn_type.startswith("sparse"):
            from models.model_sparse import SparseModel
            self.model = SparseModel(config)
        elif self.attn_type.startswith("_sparse"):
            from models.model_sparse_ import SparseModel_
            self.model = SparseModel_(config)
        elif self.attn_type.startswith("clustered"):
            from models.model_clustered import ClusteredModel
            self.model = ClusteredModel(config)
        elif self.attn_type.startswith("synthesizer"):
            from models.model_synthesizer import SynthesizerModel
            self.model = SynthesizerModel(config)
        elif self.attn_type.startswith("informer"):
            from models.model_informer import InformerModel
            self.model = InformerModel(config)
        elif self.attn_type.startswith("longformer"):
            from models.model_longformer import LongformerModel
            self.model = LongformerModel(config)
        elif self.attn_type.startswith("nystrom"):
            from models.model_nystrom import NystromModel
            self.model = NystromModel(config)
        elif self.attn_type.startswith("flash_quad"):
            from models.model_gau import GAUModel
            self.model = GAUModel(config)
        elif self.attn_type.startswith("transnormer"):
            from models.model_transnormer import TransNormerModel
            self.model = TransNormerModel(config)
        elif self.attn_type.startswith("pernormer"):
            from models.model_pernormer import PerNormerModel
            self.model = PerNormerModel(config)
        
        self.mlm = MLMHead(config, self.model.embeddings.word_embeddings)
        self.sen_classifer = SequenceClassificationHead(config)

    def forward(self, input_ids, attention_mask, labels = None):

        with torch.cuda.amp.autocast(enabled = self.enable_amp):

            token_out = self.model(input_ids, attention_mask)

            mlm_scores = self.mlm(token_out)
            mlm_correct = (mlm_scores.argmax(dim = -1) == input_ids).to(torch.float32)
            mlm_accu = torch.sum(mlm_correct * attention_mask) / (torch.sum(attention_mask) + 1e-6)

            sent_scores = self.sen_classifer(token_out)

            outputs = {
                "mlm_accu":mlm_accu[None], "sent_scores":sent_scores
            }

            sen_loss = None
            if labels is not None:
                if self.num_classes == 1:
                    # We are doing regression
                    sen_loss_fct = nn.MSELoss()
                    sen_loss = sen_loss_fct(sent_scores.view(-1), labels.view(-1))
                else:
                    sen_loss_fct = nn.CrossEntropyLoss()
                    sen_loss = sen_loss_fct(sent_scores.view(-1, self.num_classes), labels.view(-1))

                sen_correct = (sent_scores.argmax(dim = -1) == labels).to(torch.float32)
                sen_accu = torch.mean(sen_correct)

                outputs["loss"] = sen_loss[None]
                outputs["accu"] = sen_accu[None]

        return outputs
