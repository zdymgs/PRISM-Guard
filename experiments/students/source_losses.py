"""Shared original source-side weighting and representation loss."""
from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
def class_weights(labels: np.ndarray, power: float, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=40).astype(np.float64)
    reference = counts[counts > 0].mean()
    weights = np.zeros(40, dtype=np.float32)
    present = counts > 0
    weights[present] = np.power(reference / counts[present], power).astype(np.float32)
    return torch.from_numpy(weights).to(device)

def relation_loss(student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    student_norm = F.normalize(student, dim=-1)
    teacher_norm = F.normalize(teacher, dim=-1)
    student_gram = torch.einsum("bwd,bvd->bwv", student_norm.flatten(1, 2), student_norm.flatten(1, 2))
    teacher_gram = torch.einsum("bwd,bvd->bwv", teacher_norm.flatten(1, 2), teacher_norm.flatten(1, 2))
    flat_mask = mask.flatten(1)
    pair_mask = flat_mask.unsqueeze(1) & flat_mask.unsqueeze(2)
    difference = (student_gram - teacher_gram).square() * pair_mask
    return difference.sum() / pair_mask.sum().clamp_min(1)
