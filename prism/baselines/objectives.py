"""Label-free objectives for the KD-soft and confidence pseudo-labeling baselines."""

import torch
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
