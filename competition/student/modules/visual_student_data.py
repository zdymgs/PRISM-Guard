from __future__ import annotations

from pathlib import Path
from typing import Any
import csv

import numpy as np
import torch
from torch.utils.data import Dataset

from shared_dir_roi_data import SharedDIRFeatureDataset
from shared_dir_roi_model import MODALITY_NAMES, REGION_NAMES


WINDOW_NAMES = ("early", "late")
WINDOW_BOUNDS = ((0.0, 0.70), (0.30, 1.0))
VIEW_NAMES = ("scene", "person", "workspace")
FRAME_COUNT = 16


def window_indices(
    frame_count: int, low: float, high: float, samples: int = FRAME_COUNT
) -> np.ndarray:
    if frame_count < 1 or samples < 1 or not 0.0 <= low < high <= 1.0:
        raise ValueError("invalid temporal window request")
    end = frame_count - 1
    return np.rint(np.linspace(low * end, high * end, samples)).astype(np.int64)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path).resolve(), allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


class VisualStudentDataset(Dataset[dict[str, Any]]):
    """Early/late x scene/person/workspace IR features for one small student.

    roi_features already extracted one shared ImageNet ResNet18 pyramid for every frame,
    modality and ROI.  student phase one selects the exact foundation windows/views from
    those deployable small-backbone features; Large VideoMAE arrays are returned
    only as training targets and never as student inputs.
    """

    def __init__(
        self,
        feature_run: str | Path,
        teacher_features: str | Path,
        teacher_oof_logits: str | Path,
        sample_ids: set[str] | None = None,
    ) -> None:
        teacher = load_npz(teacher_features)
        logits = load_npz(teacher_oof_logits)
        teacher_ids = np.asarray(teacher["sample_ids"]).astype(str)
        teacher_source_ids = np.asarray(teacher["source_ids"]).astype(str)
        logit_ids = np.asarray(logits["sample_ids"]).astype(str)
        if len(set(teacher_ids.tolist())) != len(teacher_ids):
            raise RuntimeError("duplicate teacher feature sample IDs")
        if len(set(logit_ids.tolist())) != len(logit_ids):
            raise RuntimeError("duplicate teacher logit sample IDs")
        self.teacher_lookup = {sample_id: index for index, sample_id in enumerate(teacher_ids)}
        self.source_lookup = {
            source_id: index for index, source_id in enumerate(teacher_source_ids)
        }
        if len(self.source_lookup) != len(teacher_source_ids):
            raise RuntimeError("duplicate teacher source IDs")
        self.logit_lookup = {sample_id: index for index, sample_id in enumerate(logit_ids)}
        selected_source_ids = None
        if sample_ids is not None:
            missing = set(sample_ids) - set(self.teacher_lookup)
            if missing:
                raise RuntimeError(f"unknown canonical student sample IDs: {sorted(missing)[:3]}")
            selected_source_ids = {
                teacher_source_ids[self.teacher_lookup[sample_id]] for sample_id in sample_ids
            }
        self.base = SharedDIRFeatureDataset(
            feature_run, sample_ids=selected_source_ids
        )
        self.teacher_ids = teacher_ids
        self.teacher_values = np.asarray(teacher["features"], dtype=np.float16)
        self.teacher_labels = np.asarray(teacher["labels"], dtype=np.int64)
        self.teacher_users = np.asarray(teacher["users"]).astype(str)
        self.teacher_logits = np.asarray(logits["early_late_logits"], dtype=np.float32)
        self.logit_labels = np.asarray(logits["labels"], dtype=np.int64)
        self.logit_users = np.asarray(logits["users"]).astype(str)
        self.logit_folds = np.asarray(logits["folds"], dtype=np.int64)
        if self.teacher_values.shape != (2914, 2, 3, 1024):
            raise RuntimeError(f"unexpected teacher feature shape: {self.teacher_values.shape}")
        if self.teacher_logits.shape != (2914, 40):
            raise RuntimeError(f"unexpected teacher logit shape: {self.teacher_logits.shape}")
        if tuple(np.asarray(teacher["window_names"]).astype(str)) != WINDOW_NAMES:
            raise RuntimeError("teacher window order changed")
        if tuple(np.asarray(teacher["view_names"]).astype(str)) != VIEW_NAMES:
            raise RuntimeError("teacher view order changed")
        self.ir_index = MODALITY_NAMES.index("ir")
        self.region_indices = tuple(
            REGION_NAMES.index(name)
            for name in ("global_fallback", "full_body", "hand_workspace")
        )
        for row in self.base.rows:
            source_id = row["sample_id"]
            if source_id not in self.source_lookup:
                raise RuntimeError(f"missing student teacher source target: {source_id}")
            canonical_id = self.teacher_ids[self.source_lookup[source_id]]
            if canonical_id not in self.logit_lookup:
                raise RuntimeError(f"missing student teacher logit target: {canonical_id}")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base[index]
        source_id = str(item["sample_id"])
        teacher_index = self.source_lookup[source_id]
        sample_id = str(self.teacher_ids[teacher_index])
        logit_index = self.logit_lookup[sample_id]
        if int(item["class_id"]) != int(self.teacher_labels[teacher_index]):
            raise RuntimeError(f"roi_features/teacher label mismatch: {sample_id}")
        if int(item["class_id"]) != int(self.logit_labels[logit_index]):
            raise RuntimeError(f"roi_features/teacher-logit label mismatch: {sample_id}")
        if str(item["user_id"]) != self.teacher_users[teacher_index]:
            raise RuntimeError(f"roi_features/teacher user mismatch: {sample_id}")
        if str(item["user_id"]) != self.logit_users[logit_index]:
            raise RuntimeError(f"roi_features/teacher-logit user mismatch: {sample_id}")

        length = len(item["frame_ids"])
        chosen = np.stack(
            [window_indices(length, low, high) for low, high in WINDOW_BOUNDS]
        )
        chosen_tensor = torch.from_numpy(chosen)
        features = item["features"][
            chosen_tensor, self.ir_index
        ][:, :, self.region_indices]
        view_mask = item["roi_valid"][chosen_tensor][:, :, self.region_indices]
        view_quality = item["roi_quality"][chosen_tensor][:, :, self.region_indices]
        positions = (
            chosen_tensor.float() / float(max(length - 1, 1))
        )
        # The global fallback is a real full-scene crop.  Keep it available even
        # if an older ROI cache marked its quality metadata conservatively.
        view_mask[:, :, 0] = True
        view_quality[:, :, 0] = torch.maximum(
            view_quality[:, :, 0], torch.full_like(view_quality[:, :, 0], 0.5)
        )
        return {
            "sample_id": sample_id,
            "source_id": source_id,
            "user_id": str(item["user_id"]),
            "label": torch.tensor(int(item["class_id"]), dtype=torch.long),
            "fold": torch.tensor(int(self.logit_folds[logit_index]), dtype=torch.long),
            "features": features.float(),
            "view_mask": view_mask.bool(),
            "view_quality": view_quality.float(),
            "time_position": positions,
            "teacher_logits": torch.from_numpy(self.teacher_logits[logit_index].copy()),
            "teacher_features": torch.from_numpy(
                self.teacher_values[teacher_index].astype(np.float32)
            ),
        }


class CompactVisualStudentDataset(Dataset[dict[str, Any]]):
    """Memory-mapped value-equivalent student inputs with training-only teacher targets."""

    def __init__(
        self,
        compact_cache: str | Path,
        teacher_features: str | Path,
        teacher_oof_logits: str | Path,
    ) -> None:
        self.compact_cache = Path(compact_cache).resolve()
        with (self.compact_cache / "rows.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            self.rows = list(csv.DictReader(handle))
        if len(self.rows) != 2914:
            raise RuntimeError(f"student compact row universe changed: {len(self.rows)}")
        self.features = np.load(
            self.compact_cache / "features.npy", mmap_mode="r", allow_pickle=False
        )
        self.view_mask = np.load(
            self.compact_cache / "view_mask.npy", mmap_mode="r", allow_pickle=False
        )
        self.view_quality = np.load(
            self.compact_cache / "view_quality.npy", mmap_mode="r", allow_pickle=False
        )
        self.time_position = np.load(
            self.compact_cache / "time_position.npy", mmap_mode="r", allow_pickle=False
        )
        if self.features.shape != (2914, 2, FRAME_COUNT, 3, 896):
            raise RuntimeError(f"unexpected compact student shape: {self.features.shape}")
        teacher = load_npz(teacher_features)
        logits = load_npz(teacher_oof_logits)
        teacher_ids = np.asarray(teacher["sample_ids"]).astype(str)
        logit_ids = np.asarray(logits["sample_ids"]).astype(str)
        self.teacher_lookup = {sample_id: index for index, sample_id in enumerate(teacher_ids)}
        self.logit_lookup = {sample_id: index for index, sample_id in enumerate(logit_ids)}
        self.teacher_values = np.asarray(teacher["features"], dtype=np.float16)
        self.teacher_logits = np.asarray(logits["early_late_logits"], dtype=np.float32)
        self.logit_labels = np.asarray(logits["labels"], dtype=np.int64)
        self.logit_users = np.asarray(logits["users"]).astype(str)
        self.logit_folds = np.asarray(logits["folds"], dtype=np.int64)
        self.index_lookup: dict[str, int] = {}
        for index, row in enumerate(self.rows):
            sample_id = row["sample_id"]
            if int(row["row_index"]) != index or sample_id in self.index_lookup:
                raise RuntimeError("invalid student compact row ordering")
            if sample_id not in self.teacher_lookup or sample_id not in self.logit_lookup:
                raise RuntimeError(f"missing compact teacher target: {sample_id}")
            logit_index = self.logit_lookup[sample_id]
            if int(row["class_id"]) != int(self.logit_labels[logit_index]):
                raise RuntimeError(f"compact label mismatch: {sample_id}")
            if row["user_id"] != self.logit_users[logit_index]:
                raise RuntimeError(f"compact user mismatch: {sample_id}")
            if int(row["fold"]) != int(self.logit_folds[logit_index]):
                raise RuntimeError(f"compact fold mismatch: {sample_id}")
            self.index_lookup[sample_id] = index

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        sample_id = row["sample_id"]
        teacher_index = self.teacher_lookup[sample_id]
        logit_index = self.logit_lookup[sample_id]
        # copy() turns read-only memmap slices into writable tensors and avoids
        # PyTorch warnings without loading the full 0.5 GB cache into RAM.
        return {
            "sample_id": sample_id,
            "source_id": row["source_id"],
            "user_id": row["user_id"],
            "label": torch.tensor(int(row["class_id"]), dtype=torch.long),
            "fold": torch.tensor(int(row["fold"]), dtype=torch.long),
            "features": torch.from_numpy(self.features[index].astype(np.float32)),
            "view_mask": torch.from_numpy(self.view_mask[index].copy()),
            "view_quality": torch.from_numpy(self.view_quality[index].astype(np.float32)),
            "time_position": torch.from_numpy(self.time_position[index].copy()),
            "teacher_logits": torch.from_numpy(self.teacher_logits[logit_index].copy()),
            "teacher_features": torch.from_numpy(
                self.teacher_values[teacher_index].astype(np.float32)
            ),
        }


def collate_visual(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("empty student visual batch")
    return {
        "sample_id": [str(item["sample_id"]) for item in items],
        "user_id": [str(item["user_id"]) for item in items],
        "label": torch.stack([item["label"] for item in items]),
        "fold": torch.stack([item["fold"] for item in items]),
        "features": torch.stack([item["features"] for item in items]),
        "view_mask": torch.stack([item["view_mask"] for item in items]),
        "view_quality": torch.stack([item["view_quality"] for item in items]),
        "time_position": torch.stack([item["time_position"] for item in items]),
        "teacher_logits": torch.stack([item["teacher_logits"] for item in items]),
        "teacher_features": torch.stack([item["teacher_features"] for item in items]),
    }
