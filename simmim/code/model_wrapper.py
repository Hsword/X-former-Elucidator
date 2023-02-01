import os
import sys

curr_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(curr_path)

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.nn.functional import l1_loss


class Approx_GeLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.grad_checkpointing = config["gelu_grad_checkpointing"]
        self.act = nn.GELU()

    def func(self, x):
        return self.act(x)

    def forward(self, x):
        if self.grad_checkpointing:
            x = checkpoint(self.func, x)
        else:
            x = self.func(x)
        return x


class ModelForMaskedIM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.enable_amp = config["mixed_precision"]

        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.drop_last = False if "drop_last" not in config or not config["drop_last"] else True
        
        self.attn_type = config["attn_type"]
        if self.attn_type.startswith("softmax"):
            from vit_models.model import Model
            self.model = Model(config, use_mask_token=True)
        elif self.attn_type.startswith("linformer"):
            from vit_models.model_linformer import LinformerModel
            self.model = LinformerModel(config, use_mask_token=True)
        elif self.attn_type.startswith("cosformer"):
            from vit_models.model_cosformer import CosformerModel
            self.model = CosformerModel(config, use_mask_token=True)
        elif self.attn_type.startswith("linear"):
            from vit_models.model_linear import LinearModel
            self.model = LinearModel(config, use_mask_token=True)
        elif self.attn_type.startswith("bigbird"):
            from vit_models.model_bigbird import BigBirdModel
            self.model = BigBirdModel(config, use_mask_token=True)
        elif self.attn_type.startswith("performer"):
            from vit_models.model_performer import PerformerModel
            self.model = PerformerModel(config, use_mask_token=True)
        elif self.attn_type.startswith("reformer"):
            from vit_models.model_reformer import ReformerModel
            self.model = ReformerModel(config, use_mask_token=True)
        elif self.attn_type.startswith("sparse"):
            from vit_models.model_sparse import SparseModel
            self.model = SparseModel(config, use_mask_token=True)
        elif self.attn_type.startswith("clustered"):
            from vit_models.model_clustered import ClusteredModel
            self.model = ClusteredModel(config, use_mask_token=True)
        elif self.attn_type.startswith("synthesizer"):
            from vit_models.model_synthesizer import SynthesizerModel
            self.model = SynthesizerModel(config, use_mask_token=True)
        elif self.attn_type.startswith("longformer"):
            from vit_models.model_longformer import LongformerModel
            self.model = LongformerModel(config, use_mask_token=True)
        elif self.attn_type.startswith("nystrom"):
            from vit_models.model_nystrom import NystromModel
            self.model = NystromModel(config, use_mask_token=True)
        elif self.attn_type.startswith("transnormer"):
            from vit_models.model_transnormer import TransNormerModel
            self.model = TransNormerModel(config, use_mask_token=True)
        
        self.mim = nn.Sequential(
            nn.Conv2d(in_channels=config["hidden_dim"], out_channels=config["encoder_stride"]**2 * 3, kernel_size=1),
            nn.PixelShuffle(config["encoder_stride"]),
        )

        self.config = config

    def forward(self, pixel_values, bool_masked_pos = None):

        with torch.cuda.amp.autocast(enabled = self.enable_amp):
            token_out = self.model(pixel_values, bool_masked_pos)

            # Reshape to (batch_size, num_channels, height, width)
            if not self.drop_last:
                token_out = token_out[:, 1:]
            batch_size, seq_len, num_channels = token_out.size()
            height = width = int(seq_len**0.5)
            token_out = token_out.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)

            # Reconstruct pixel values
            reconstructed_pixel_values = self.mim(token_out)

            mim_loss = None
            if bool_masked_pos is not None:
                size = self.image_size // self.patch_size
                bool_masked_pos = bool_masked_pos.reshape(-1, size, size)
                mask = (
                    bool_masked_pos.repeat_interleave(self.patch_size, 1)
                    .repeat_interleave(self.patch_size, 2)
                    .unsqueeze(1)
                    .contiguous()
                )
                reconstruction_loss = l1_loss(pixel_values, reconstructed_pixel_values, reduction="none")
                mim_loss = (reconstruction_loss * mask).sum() / (mask.sum() + 1e-5) / self.num_channels

            outputs = {
                "loss":mim_loss[None]
            }

        return outputs


class ModelForImageClassification(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.enable_amp = config["mixed_precision"]

        self.num_classes = config["num_classes"]

        self.attn_type = config["attn_type"]
        if self.attn_type.startswith("softmax"):
            from vit_models.model import Model
            self.model = Model(config)
        elif self.attn_type.startswith("linformer"):
            from vit_models.model_linformer import LinformerModel
            self.model = LinformerModel(config)
        elif self.attn_type.startswith("cosformer"):
            from vit_models.model_cosformer import CosformerModel
            self.model = CosformerModel(config)
        elif self.attn_type.startswith("linear"):
            from vit_models.model_linear import LinearModel
            self.model = LinearModel(config)
        elif self.attn_type.startswith("bigbird"):
            from vit_models.model_bigbird import BigBirdModel
            self.model = BigBirdModel(config)
        elif self.attn_type.startswith("performer"):
            from vit_models.model_performer import PerformerModel
            self.model = PerformerModel(config)
        elif self.attn_type.startswith("reformer"):
            from vit_models.model_reformer import ReformerModel
            self.model = ReformerModel(config)
        elif self.attn_type.startswith("sparse"):
            from vit_models.model_sparse import SparseModel
            self.model = SparseModel(config)
        elif self.attn_type.startswith("clustered"):
            from vit_models.model_clustered import ClusteredModel
            self.model = ClusteredModel(config)
        elif self.attn_type.startswith("synthesizer"):
            from vit_models.model_synthesizer import SynthesizerModel
            self.model = SynthesizerModel(config)
        elif self.attn_type.startswith("longformer"):
            from vit_models.model_longformer import LongformerModel
            self.model = LongformerModel(config)
        elif self.attn_type.startswith("nystrom"):
            from vit_models.model_nystrom import NystromModel
            self.model = NystromModel(config)
        elif self.attn_type.startswith("transnormer"):
            from vit_models.model_transnormer import TransNormerModel
            self.model = TransNormerModel(config)

        self.img_classifier = nn.Linear(config["hidden_dim"], config["num_classes"])

        
    def forward(self, pixel_values, labels=None):
        with torch.cuda.amp.autocast(enabled = self.enable_amp):
            token_out = self.model(pixel_values)
            
            logits = self.img_classifier(token_out[:, 0, :])

            loss = None
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

            outputs = {
                "loss": loss[None], "logits": logits
            }

        return outputs
