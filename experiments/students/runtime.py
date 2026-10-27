"""Shared native model, checkpoint and data-loader adapters."""

from __future__ import annotations
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .cache import TensorCache, MOTION_FIELDS, aligned_values
from .source_fusion import visual_head_parameters

CLASSES = {"cuhkx": 40, "mmact": 37, "mmfit": 11}
SEEDS = {"cuhkx": 20260826, "mmact": 20260922, "mmfit": 20260927}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_student(dataset):
    if dataset == "cuhkx":
        from prism.student.cuhkx_model import build_cuhkx_student
        return build_cuhkx_student()
    if dataset == "mmact":
        from prism.student.activity_model import build_mmact_student
        return build_mmact_student()
    from prism.student.workout_model import build_student as build_mmfit
    return build_mmfit()


def load_checkpoint(model, path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a state dictionary")
    state = payload.get("model_state", payload.get("model", payload.get("state_dict", payload)))
    if not isinstance(state, dict) or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError("unsupported checkpoint: expected model_state, model or state_dict")
    model.load_state_dict(state, strict=True)
    return payload


def configure_adaptation(model):
    model.requires_grad_(False)
    fusion = list(model.motion_residual.parameters())
    visual = visual_head_parameters(model.visual)
    for parameter in fusion + visual:
        parameter.requires_grad_(True)
    if not fusion or not visual:
        raise ValueError("native adaptation parameter groups must be nonempty")
    return fusion, visual


def keep_cnn_eval(model):
    for name in ("stem", "layer1", "layer2", "layer3", "layer4"):
        getattr(model.visual, name).eval()


def model_forward(model, batch, raw=False):
    motion = {name: batch[name] for name in MOTION_FIELDS}
    if raw:
        return model(batch["images"], batch["view_valid"], batch["view_quality"],
                     batch["global_time_position"], motion)
    return model.forward_from_backbone_sequence(
        batch["backbone_sequence"], batch["view_valid"], batch["view_quality"],
        batch["global_time_position"], motion)


def move(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


class Context:
    def __init__(self, args, probability, *, raw=False):
        self.name = args.dataset
        self.device = torch.device(args.device)
        self.raw = raw
        kwargs = dict(cache_root=args.cache_root, targets=probability, raw=raw, fold=args.fold)
        self.base = TensorCache(args.inputs, **kwargs)
        self.train = TensorCache(args.inputs, augment=self.name == "cuhkx" and not raw, **kwargs)
        self.seed = args.seed if args.seed is not None else SEEDS[self.name]
        self.info = dict(rows=len(self.base), classes=CLASSES[self.name], seed=self.seed)
        self.model = build_student(self.name)
        load_checkpoint(self.model, args.c0)
        self.model.to(self.device)
        self.ids = self.base.ids

    def make_loader(self, dataset, shuffle, epoch=0):
        if self.name == "cuhkx":
            generator = None
        else:
            generator = (torch.Generator().manual_seed(self.seed + epoch)
                         if shuffle and self.name == "mmact" else
                         torch.Generator().manual_seed(self.seed) if self.name == "mmfit" else None)
        return DataLoader(dataset, batch_size=64, shuffle=shuffle, num_workers=0,
                          pin_memory=True, generator=generator)

    def row_positions(self, batch):
        return np.asarray(batch.pop("_row_index"), dtype=np.int64)

    def configure(self):
        return configure_adaptation(self.model)

    def keep_cnn_eval(self):
        if self.name != "cuhkx":
            keep_cnn_eval(self.model)

    def move(self, batch):
        return move(batch, self.device)

    def forward(self, batch):
        return model_forward(self.model, batch, raw=self.raw)["logits"]


def load_targets(args):
    # Reading identities does not open any tensor cache or source supervision.
    config = json.loads(Path(args.inputs).read_text(encoding="utf-8-sig"))
    if config["dataset"].lower() != args.dataset:
        raise ValueError("manifest dataset differs from --dataset")
    return aligned_values(args.targets, [str(s) for s in config["record_id"]],
                          key="targets", classes=CLASSES[args.dataset])


def require_cuda(args):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu for a small local check")
