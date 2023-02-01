import torch.nn as nn
import torch.nn.functional as F
from rms_norm import RMSNorm, SimpleRMSNorm, GatedRMSNorm


def get_norm_fn(norm_type="layer_norm"):
    if norm_type == "layer_norm":
        return nn.LayerNorm
    elif norm_type == "rms_norm":
        return RMSNorm
    elif norm_type == "simple_rms_norm":
        return SimpleRMSNorm
    elif norm_type == "gated_rms_norm":
        return GatedRMSNorm
    else:
        assert False


def get_act_fn(act_type="relu"):
    if act_type == "relu":
        return F.relu
    elif act_type == "gelu":
        return F.gelu
    elif act_type == "silu":
        return F.silu
    elif act_type == "1+elu":
        def f(x):
            return 1 + F.elu(x)
        return f
    else:
        assert False