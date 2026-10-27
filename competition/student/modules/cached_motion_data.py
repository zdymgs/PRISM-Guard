from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from visual_pixel_data import VisualPixelDataset


MOTION_FIELDS = (
    "skeleton_features",
    "skeleton_feature_mask",
    "skeleton_joint_mask",
    "skeleton_relations",
    "skeleton_relation_mask",
    "skeleton_frame_quality",
    "imu_sequences",
    "imu_sequence_mask",
    "imu_bin_statistics",
    "imu_bin_mask",
    "imu_global_statistics",
    "imu_global_mask",
)
TEMPORAL_MOTION_FIELDS = MOTION_FIELDS[:10]
BOOLEAN_MOTION_FIELDS = {
    "skeleton_feature_mask",
    "skeleton_joint_mask",
    "skeleton_relation_mask",
    "imu_sequence_mask",
    "imu_bin_mask",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class CachedSequenceMotionDataset(Dataset[dict[str, Any]]):
    """Training-only MC3 sequence acceleration with exact aligned motion tokens."""

    def __init__(
        self,
        sequence_cache: str | Path,
        motion_cache: str | Path,
        pixel_cache: str | Path,
        teacher_features: str | Path,
        teacher_logits: str | Path,
        indices: np.ndarray | list[int] | None = None,
        temporal_augment: bool = False,
        imu_teacher_logits: str | Path | None = None,
        imu_event_features: str | Path | None = None,
        compact_sequence_cache: str | Path | None = None,
    ) -> None:
        self.sequence_cache = Path(sequence_cache).resolve()
        self.motion_cache = Path(motion_cache).resolve()
        sequence_rows = read_rows(self.sequence_cache / "rows.csv")
        motion_rows = read_rows(self.motion_cache / "rows.csv")
        pixel_rows = read_rows(Path(pixel_cache).resolve() / "rows.csv")
        row_ids = [[row["sample_id"] for row in rows] for rows in (sequence_rows, motion_rows, pixel_rows)]
        if row_ids[0] != row_ids[1] or row_ids[0] != row_ids[2]:
            raise RuntimeError("student sequence, motion and pixel cache row orders differ")
        self.rows = sequence_rows
        region_path = self.sequence_cache / "backbone_region_sequence_fp16.npy"
        sequence_path = self.sequence_cache / "backbone_sequence_fp16.npy"
        if region_path.exists() == sequence_path.exists():
            raise RuntimeError(
                "student sequence cache must contain exactly one compact or region sequence"
            )
        self.uses_spatial_regions = region_path.exists()
        self.backbone_sequence = np.load(
            region_path if self.uses_spatial_regions else sequence_path, mmap_mode="r"
        )
        self.compact_backbone_sequence: np.ndarray | None = None
        if compact_sequence_cache is not None:
            if not self.uses_spatial_regions:
                raise ValueError(
                    "a separate compact cache is valid only with a spatial cache"
                )
            compact_cache = Path(compact_sequence_cache).resolve()
            compact_rows = read_rows(compact_cache / "rows.csv")
            if [row["sample_id"] for row in compact_rows] != row_ids[0]:
                raise RuntimeError("student compact and spatial cache row orders differ")
            self.compact_backbone_sequence = np.load(
                compact_cache / "backbone_sequence_fp16.npy", mmap_mode="r"
            )
            compact_completed = np.load(
                compact_cache / "completed.npy", mmap_mode="r"
            )
            if (
                len(compact_completed) != len(self.rows)
                or not np.asarray(compact_completed).all()
            ):
                raise RuntimeError("student compact sequence cache is incomplete")
        self.anchor_logits = np.load(
            self.sequence_cache / "anchor_logits_fp16.npy", mmap_mode="r"
        )
        completed = np.load(self.sequence_cache / "completed.npy", mmap_mode="r")
        if len(completed) != len(self.rows) or not np.asarray(completed).all():
            raise RuntimeError("student MC3 sequence cache is incomplete")
        self.motion = {
            field: np.load(self.motion_cache / f"{field}.npy", mmap_mode="r")
            for field in MOTION_FIELDS
        }
        motion_completed = np.load(self.motion_cache / "completed.npy", mmap_mode="r")
        if len(motion_completed) != len(self.rows) or not np.asarray(motion_completed).all():
            raise RuntimeError("student motion cache is incomplete")
        self.view_valid = np.load(
            Path(pixel_cache).resolve() / "view_valid.npy", mmap_mode="r"
        )
        self.view_quality = np.load(
            Path(pixel_cache).resolve() / "view_quality.npy", mmap_mode="r"
        )
        self.source_frame_indices = np.load(
            Path(pixel_cache).resolve() / "source_frame_indices.npy", mmap_mode="r"
        )
        # Reuse the already-audited teacher alignment contract without reading pixels.
        reference = VisualPixelDataset(
            pixel_cache, teacher_features, teacher_logits, augment=False
        )
        self.teacher_features = reference.teacher_features
        self.teacher_logits = reference.teacher_logits
        self.teacher_early_logits = reference.teacher_early_logits
        self.teacher_late_logits = reference.teacher_late_logits
        self.teacher_temporal_delta_logits = reference.teacher_temporal_delta_logits
        (
            self.imu_teacher_logits,
            self.imu_teacher_valid,
        ) = self._load_partial_imu_teacher(imu_teacher_logits)
        (
            self.imu_event_features,
            self.imu_event_valid,
        ) = self._load_partial_event_features(imu_event_features)
        self.labels = reference.labels
        self.users = reference.users
        self.indices = (
            np.arange(len(self.rows), dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        self.temporal_augment = bool(temporal_augment)

    def _load_partial_imu_teacher(
        self, path: str | Path | None
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if path is None:
            return None, None
        with np.load(Path(path).resolve(), allow_pickle=False) as data:
            sample_ids = np.asarray(data["sample_ids"]).astype(str)
            key = "imu_logits" if "imu_logits" in data.files else "logits"
            logits = np.asarray(data[key], dtype=np.float32)
            source_valid = (
                np.asarray(data["valid"], dtype=bool)
                if "valid" in data.files
                else np.ones(len(sample_ids), dtype=bool)
            )
        lookup = {sample_id: index for index, sample_id in enumerate(sample_ids)}
        aligned = np.zeros((len(self.rows), logits.shape[1]), dtype=np.float32)
        valid = np.zeros(len(self.rows), dtype=bool)
        for row_index, row in enumerate(self.rows):
            teacher_index = lookup.get(row["sample_id"])
            if teacher_index is not None and source_valid[teacher_index]:
                aligned[row_index] = logits[teacher_index]
                valid[row_index] = True
        return aligned, valid

    def _load_partial_event_features(
        self, path: str | Path | None
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if path is None:
            return None, None
        with np.load(Path(path).resolve(), allow_pickle=False) as data:
            sample_ids = np.asarray(data["sample_ids"]).astype(str)
            features = np.asarray(data["features"], dtype=np.float32)
            source_valid = np.asarray(data["usable"], dtype=bool)
        lookup = {sample_id: index for index, sample_id in enumerate(sample_ids)}
        aligned = np.zeros((len(self.rows), features.shape[1]), dtype=np.float32)
        valid = np.zeros(len(self.rows), dtype=bool)
        for row_index, row in enumerate(self.rows):
            feature_index = lookup.get(row["sample_id"])
            if feature_index is not None and source_valid[feature_index]:
                aligned[row_index] = features[feature_index]
                valid[row_index] = True
        return aligned, valid

    @property
    def index_lookup(self) -> dict[str, int]:
        return {row["sample_id"]: index for index, row in enumerate(self.rows)}

    def __len__(self) -> int:
        return len(self.indices)

    @staticmethod
    def temporal_indices(steps: int) -> np.ndarray:
        result = np.arange(steps, dtype=np.int64)
        if np.random.random() < 0.75:
            scale = float(np.random.uniform(0.86, 1.14))
            shift = float(np.random.uniform(-0.07, 0.07) * max(steps - 1, 1))
            base = np.arange(steps, dtype=np.float32)
            positions = np.clip(
                (base - 0.5 * (steps - 1)) * scale + 0.5 * (steps - 1) + shift,
                0.0,
                float(steps - 1),
            )
            result = np.rint(positions).astype(np.int64)
        return result

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = int(self.indices[item])
        sequence = np.asarray(self.backbone_sequence[index], dtype=np.float32).copy()
        compact_sequence = (
            None
            if self.compact_backbone_sequence is None
            else np.asarray(
                self.compact_backbone_sequence[index], dtype=np.float32
            ).copy()
        )
        valid = np.asarray(self.view_valid[index], dtype=bool).copy()
        quality = np.asarray(self.view_quality[index], dtype=np.float32).copy()
        source_indices = np.asarray(self.source_frame_indices[index], dtype=np.int64).copy()
        frame_count = int(source_indices.max(initial=0)) + 1
        motion = {
            field: np.asarray(self.motion[field][index]).copy() for field in MOTION_FIELDS
        }
        steps = sequence.shape[2]
        temporal_index = (
            self.temporal_indices(steps)
            if self.temporal_augment
            else np.arange(steps, dtype=np.int64)
        )
        sequence = sequence[:, :, temporal_index]
        if compact_sequence is not None:
            compact_sequence = compact_sequence[:, :, temporal_index]
        valid = valid[:, temporal_index]
        quality = quality[:, temporal_index]
        source_indices = source_indices[:, temporal_index]
        for field in TEMPORAL_MOTION_FIELDS:
            motion[field] = motion[field][:, temporal_index]
        global_time = source_indices.astype(np.float32) / max(frame_count - 1, 1)
        result: dict[str, Any] = {
            (
                "backbone_region_sequence"
                if self.uses_spatial_regions
                else "backbone_sequence"
            ): torch.from_numpy(sequence),
            "view_valid": torch.from_numpy(valid),
            "view_quality": torch.from_numpy(quality),
            "global_time_position": torch.from_numpy(global_time),
            "teacher_features": torch.from_numpy(self.teacher_features[index]),
            "anchor_logits": torch.from_numpy(
                np.asarray(self.anchor_logits[index], dtype=np.float32).copy()
            ),
            "teacher_logits": torch.from_numpy(self.teacher_logits[index]),
            "teacher_early_logits": torch.from_numpy(self.teacher_early_logits[index]),
            "teacher_late_logits": torch.from_numpy(self.teacher_late_logits[index]),
            "teacher_temporal_delta_logits": torch.from_numpy(
                self.teacher_temporal_delta_logits[index]
            ),
            "label": torch.tensor(int(self.labels[index]), dtype=torch.long),
            "sample_id": self.rows[index]["sample_id"],
            "user_id": str(self.users[index]),
            "cache_index": index,
        }
        if compact_sequence is not None:
            result["backbone_sequence"] = torch.from_numpy(compact_sequence)
        if self.imu_teacher_logits is not None:
            result["imu_teacher_logits"] = torch.from_numpy(
                np.asarray(self.imu_teacher_logits[index], dtype=np.float32).copy()
            )
            result["imu_teacher_valid"] = torch.tensor(
                bool(self.imu_teacher_valid[index]), dtype=torch.bool
            )
        if self.imu_event_features is not None:
            result["imu_event_features"] = torch.from_numpy(
                np.asarray(self.imu_event_features[index], dtype=np.float32).copy()
            )
            result["imu_event_valid"] = torch.tensor(
                bool(self.imu_event_valid[index]), dtype=torch.bool
            )
        for field, value in motion.items():
            if field in BOOLEAN_MOTION_FIELDS:
                result[field] = torch.from_numpy(value.astype(bool, copy=False))
            else:
                result[field] = torch.from_numpy(value.astype(np.float32, copy=False))
        return result


def collate_cached_motion(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("empty cached-motion batch")
    output: dict[str, Any] = {}
    list_fields = {"sample_id", "user_id", "cache_index"}
    for key in items[0]:
        if key in list_fields:
            output[key] = [item[key] for item in items]
        else:
            output[key] = torch.stack([item[key] for item in items])
    return output
