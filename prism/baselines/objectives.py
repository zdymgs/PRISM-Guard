"""Label-free objectives and Tent's explicitly limited parameter scope."""

import torch
from torch import nn
from torch.nn import functional as F


def validate(logits, probability=None):
    if logits.ndim != 2 or not torch.isfinite(logits).all():
        raise ValueError("finite [batch,classes] logits required")
    if probability is not None:
        if (
            probability.shape != logits.shape
            or not torch.isfinite(probability).all()
            or (probability < 0).any()
            or not torch.allclose(
                probability.sum(-1), torch.ones_like(probability[:, 0]), atol=2e-5
            )
        ):
            raise ValueError("invalid content posterior")


def kd_soft_loss(logits, probabilities):
    validate(logits, probabilities)
    return F.kl_div(F.log_softmax(logits, -1), probabilities, reduction="batchmean")


def pl_conf_loss(logits, probabilities, threshold=0.9):
    validate(logits, probabilities)
    keep = probabilities.max(-1).values >= threshold
    count = int(keep.sum())
    if not count:
        return None, 0
    return F.cross_entropy(logits[keep], probabilities.argmax(-1)[keep]), count


def entropy_loss(logits):
    validate(logits)
    logp = F.log_softmax(logits, -1)
    return -(logp.exp() * logp).sum(-1).mean()


def configure_tent(model):
    model.eval()
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.train(True)
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
        if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.LayerNorm)):
            for name, parameter in module.named_parameters(recurse=False):
                if name in ("weight", "bias"):
                    parameter.requires_grad_(True)
    selected = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not selected or len({id(p) for _, p in selected}) != len(selected):
        raise ValueError("invalid Tent affine parameter set")
    return [p for _, p in selected], [n for n, _ in selected]
