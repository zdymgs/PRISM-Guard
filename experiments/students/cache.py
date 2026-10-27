"""Explicit tensor-cache interface for the unchanged native students."""

from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

MOTION_FIELDS = (
    "skeleton_features", "skeleton_feature_mask", "skeleton_joint_mask",
    "skeleton_relations", "skeleton_relation_mask", "skeleton_frame_quality",
    "imu_sequences", "imu_sequence_mask", "imu_bin_statistics", "imu_bin_mask",
    "imu_global_statistics", "imu_global_mask",
)
VISUAL_FIELDS = ("backbone_sequence", "images", "view_valid", "view_quality",
                 "global_time_position", "source_frame_indices")
SOURCE_FIELDS = ("teacher_features", "teacher_logits", "teacher_early_logits",
                 "teacher_late_logits", "teacher_temporal_delta_logits",
                 "imu_teacher_logits", "imu_teacher_valid", "anchor_logits")
FORBIDDEN = {"label", "labels", "truth", "class_id", "ground_truth", "class"}
FOLD_USERS = {
    0: {"user3", "user4", "user9", "user20", "user22", "user24"},
    1: {"user6", "user8", "user17", "user23"},
    2: {"user5", "user7", "user16", "user18", "user19"},
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def aligned_values(path, ids, key="targets", classes=None):
    value = read_json(path)
    other = [str(s) for s in value["record_id"]]
    if len(other) != len(set(other)) or set(other) != set(ids):
        raise ValueError("target/label record identities must match the input exactly")
    lookup = {s: i for i, s in enumerate(other)}
    result = np.asarray(value[key])[[lookup[s] for s in ids]]
    if key != "label":
        result = result.astype(np.float32)
        if (result.shape != (len(ids), classes) or not np.isfinite(result).all()
                or (result < 0).any() or not np.allclose(result.sum(1), 1, atol=2e-5)):
            raise ValueError("finite normalized [records, classes] target probabilities required")
    elif (result.shape != (len(ids),) or not np.issubdtype(result.dtype, np.integer)
          or (result < 0).any() or (result >= classes).any()):
        raise ValueError("source labels must be integer class indices")
    return result


def temporal_indices(steps):
    result = np.arange(steps, dtype=np.int64)
    if np.random.random() < 0.75:
        scale = float(np.random.uniform(0.86, 1.14))
        shift = float(np.random.uniform(-0.07, 0.07) * max(steps - 1, 1))
        base = np.arange(steps, dtype=np.float32)
        positions = np.clip((base - 0.5 * (steps - 1)) * scale
                            + 0.5 * (steps - 1) + shift, 0.0, float(steps - 1))
        result = np.rint(positions).astype(np.int64)
    return result


class TensorCache(Dataset):
    def __init__(self, path, *, cache_root=None, targets=None, source_labels=None,
                 augment=False, pixel_augment=False, raw=False, fold=None,
                 source=False, motion_only=False):
        path = Path(path)
        config = read_json(path)
        if config.get("schema") != "prism.student.v1":
            raise ValueError("schema must be prism.student.v1")
        if FORBIDDEN & set(config):
            raise ValueError("supply source labels through a separate --labels file")
        self.dataset = config["dataset"].lower()
        self.ids = [str(s) for s in config["record_id"]]
        if not self.ids or len(set(self.ids)) != len(self.ids):
            raise ValueError("nonempty unique record_id required")
        self.users = [str(s) for s in config.get("subject_id", [""] * len(self.ids))]
        self.workouts = [str(s) for s in config.get("workout", [""] * len(self.ids))]
        if len(self.users) != len(self.ids):
            raise ValueError("subject_id length must match record_id")
        if len(self.workouts) != len(self.ids):
            raise ValueError("workout length must match record_id")
        if self.dataset == "cuhkx":
            if fold not in FOLD_USERS or config.get("fold") != fold:
                raise ValueError("CUHK-X --fold must match the manifest fold")
            users = set(self.users)
            eligible = set().union(*FOLD_USERS.values())
            if not users or "" in users or not users <= eligible:
                raise ValueError("CUHK-X manifest must identify its eligible source/target subjects")
            if source and users & FOLD_USERS[fold]:
                raise ValueError("held-fold subject entered source training")
            if not source and not users <= FOLD_USERS[fold]:
                raise ValueError("target subjects do not belong to the requested fold")
        root = Path(cache_root or os.environ.get("PRISM_CACHE_ROOT") or path.parent)
        self.arrays = {}
        self.indices = np.asarray(config.get("indices", list(range(len(self.ids)))), dtype=np.int64)
        if self.indices.shape != (len(self.ids),) or (self.indices < 0).any():
            raise ValueError("indices must map each record to one nonnegative cache row")
        visual_allowed = () if motion_only else (
            ("images", "view_valid", "view_quality", "global_time_position", "source_frame_indices")
            if raw else ("backbone_sequence", "view_valid", "view_quality", "global_time_position"))
        allowed = set(MOTION_FIELDS + visual_allowed + (SOURCE_FIELDS if source else ()))
        if FORBIDDEN & set(config["tensors"]):
            raise ValueError("label tensor in input manifest")
        unknown = set(config["tensors"]) - set(MOTION_FIELDS + VISUAL_FIELDS + SOURCE_FIELDS)
        if unknown:
            raise ValueError(f"unknown tensor fields: {sorted(unknown)}")
        for key, spec in config["tensors"].items():
            if key not in allowed:
                continue  # Source-only arrays are never opened during adaptation.
            if isinstance(spec, str):
                spec = {"path": spec}
            file = Path(spec["path"])
            file = file if file.is_absolute() else root / file
            if file.suffix.lower() == ".npz":
                with np.load(file, allow_pickle=False) as archive:
                    array = archive[spec.get("key", key)].copy()
            else:
                array = np.load(file, mmap_mode="r", allow_pickle=False)
            if array.dtype.hasobject or len(array) <= int(self.indices.max()):
                raise ValueError(f"invalid tensor rows for {key}")
            self.arrays[key] = array
        visual = [] if motion_only else ["images" if raw else "backbone_sequence",
                                         "view_valid", "view_quality", "global_time_position"]
        required = set(MOTION_FIELDS + tuple(visual))
        if motion_only:
            required = set(MOTION_FIELDS)
        missing = required - set(self.arrays)
        if missing:
            raise ValueError(f"missing native tensor fields: {sorted(missing)}")
        self.targets = targets
        self.labels = source_labels
        if source != (source_labels is not None):
            raise ValueError("source datasets require explicit source labels")
        self.augment = augment
        self.pixel_augment = pixel_augment
        self.raw = raw
        self.source = source
        self.motion_only = motion_only
        self._validate_shapes()

    def _validate_shapes(self):
        shapes = {
            "skeleton_features": (2, 16, 17, 13), "skeleton_feature_mask": (2, 16, 17, 13),
            "skeleton_joint_mask": (2, 16, 17), "skeleton_relations": (2, 16, 18),
            "skeleton_relation_mask": (2, 16, 18), "skeleton_frame_quality": (2, 16),
        }
        if self.dataset == "cuhkx":
            shapes.update(imu_sequences=(2,16,5,4,16), imu_sequence_mask=(2,16,5,4),
                          imu_bin_statistics=(2,16,5,52), imu_bin_mask=(2,16,5),
                          imu_global_statistics=(5,48), imu_global_mask=(5,2),
                          view_valid=(2,16,3), view_quality=(2,16,3))
        else:
            shapes.update(imu_sequences=(2,16,5,4,6), imu_sequence_mask=(2,16,5,4,6),
                          imu_bin_statistics=(2,16,5,30), imu_bin_mask=(2,16,5,4),
                          imu_global_statistics=(2,5,30), imu_global_mask=(2,5),
                          view_valid=(2,3,16), view_quality=(2,3,16))
        shapes.update(backbone_sequence=(2,3,16,512), global_time_position=(2,16))
        for key, shape in shapes.items():
            if key in self.arrays and tuple(self.arrays[key].shape[1:]) != shape:
                raise ValueError(f"{key} must have trailing shape {shape}")
        if "images" in self.arrays:
            shape = self.arrays["images"].shape[1:]
            leading = (2,16,3) if self.dataset == "cuhkx" else (2,16,3,3)
            if tuple(shape[:-2]) != leading or self.arrays["images"].dtype != np.uint8:
                raise ValueError("raw images must be uint8 with the dataset's two-window, three-view layout")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        row = int(self.indices[index])
        item = {k: np.array(a[row], copy=True) for k, a in self.arrays.items()
                if k != "source_frame_indices"}
        if self.dataset == "cuhkx" and self.augment:
            if self.pixel_augment:
                from .augmentation import augment_ir
                item["images"], ii = augment_ir(item["images"])
            else:
                ii = temporal_indices(16)
            if "backbone_sequence" in item:
                item["backbone_sequence"] = item["backbone_sequence"][:, :, ii]
            for field in MOTION_FIELDS[:10]:
                item[field] = item[field][:, ii]
            for field in ("view_valid", "view_quality", "global_time_position"):
                item[field] = item[field][:, ii]
        for key, value in item.items():
            mask = (key.endswith("mask") and key != "imu_global_mask") or key in {
                "view_valid", "skeleton_joint_mask", "imu_teacher_valid"}
            dtype = torch.bool if mask else torch.uint8 if key == "images" else torch.float32
            item[key] = torch.as_tensor(np.ascontiguousarray(value), dtype=dtype)
        item["sample_id"] = self.ids[index]
        item["user_id"] = self.users[index]
        item["_row_index"] = index
        if self.labels is not None:
            item["label"] = torch.tensor(int(self.labels[index]), dtype=torch.long)
        if self.targets is not None:
            item["target_probability"] = torch.from_numpy(self.targets[index].copy())
            item["pseudo_probability"] = item["target_probability"]
            item["pseudo_confidence"] = item["target_probability"].max()
        return item
