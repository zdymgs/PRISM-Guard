from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from cached_motion_data import (
    BOOLEAN_MOTION_FIELDS,
    MOTION_FIELDS,
    TEMPORAL_MOTION_FIELDS,
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class TestPixelDataset(Dataset[dict[str, Any]]):
    """Label-free student pixel input contract for all official Test rows."""

    def __init__(
        self,
        cache_dir: str | Path,
        indices: np.ndarray | list[int] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir).resolve()
        self.rows = read_rows(self.cache_dir / "rows.csv")
        self.images = np.load(self.cache_dir / "images.npy", mmap_mode="r")
        self.completed = np.load(self.cache_dir / "completed.npy", mmap_mode="r")
        self.view_valid = np.load(self.cache_dir / "view_valid.npy", mmap_mode="r")
        self.view_quality = np.load(self.cache_dir / "view_quality.npy", mmap_mode="r")
        self.source_frame_indices = np.load(
            self.cache_dir / "source_frame_indices.npy", mmap_mode="r"
        )
        if len(self.rows) != len(self.images) or not np.asarray(self.completed).all():
            raise RuntimeError("challenge Test pixel cache is incomplete")
        if self.source_frame_indices.shape[:2] != self.images.shape[:2]:
            raise RuntimeError("challenge Test pixel source-index contract differs")
        self.indices = (
            np.arange(len(self.rows), dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )

    @property
    def index_lookup(self) -> dict[str, int]:
        return {row["sample_id"]: index for index, row in enumerate(self.rows)}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = int(self.indices[item])
        images = np.asarray(self.images[index]).copy()
        valid = np.asarray(self.view_valid[index], dtype=bool).copy()
        quality = np.asarray(self.view_quality[index], dtype=np.float32).copy()
        source_indices = np.asarray(
            self.source_frame_indices[index], dtype=np.int64
        ).copy()
        frame_count = int(source_indices.max(initial=0)) + 1
        global_time = source_indices.astype(np.float32) / max(frame_count - 1, 1)
        return {
            "images": torch.from_numpy(images),
            "view_valid": torch.from_numpy(valid),
            "view_quality": torch.from_numpy(quality),
            "global_time_position": torch.from_numpy(global_time),
            "sample_id": self.rows[index]["sample_id"],
            "user_id": "anonymous",
            "cache_index": index,
        }


class TestCachedSequenceMotionDataset(Dataset[dict[str, Any]]):
    """Label-free cached V+S+I inputs; contains no teacher or true-label fields."""

    def __init__(
        self,
        sequence_cache: str | Path,
        motion_cache: str | Path,
        pixel_cache: str | Path,
        indices: np.ndarray | list[int] | None = None,
        temporal_augment: bool = False,
    ) -> None:
        self.sequence_cache = Path(sequence_cache).resolve()
        self.motion_cache = Path(motion_cache).resolve()
        self.pixel_cache = Path(pixel_cache).resolve()
        sequence_rows = read_rows(self.sequence_cache / "rows.csv")
        motion_rows = read_rows(self.motion_cache / "rows.csv")
        pixel_rows = read_rows(self.pixel_cache / "rows.csv")
        row_ids = [
            [row["sample_id"] for row in rows]
            for rows in (sequence_rows, motion_rows, pixel_rows)
        ]
        if row_ids[0] != row_ids[1] or row_ids[0] != row_ids[2]:
            raise RuntimeError("challenge Test sequence, motion and pixel row orders differ")
        self.rows = sequence_rows
        sequence_path = self.sequence_cache / "backbone_sequence_fp16.npy"
        region_path = self.sequence_cache / "backbone_region_sequence_fp16.npy"
        if not sequence_path.is_file() or region_path.exists():
            raise RuntimeError("challenge final cache requires one compact MC3 sequence")
        self.backbone_sequence = np.load(sequence_path, mmap_mode="r")
        completed = np.load(self.sequence_cache / "completed.npy", mmap_mode="r")
        if len(completed) != len(self.rows) or not np.asarray(completed).all():
            raise RuntimeError("challenge Test MC3 sequence cache is incomplete")
        self.motion = {
            field: np.load(self.motion_cache / f"{field}.npy", mmap_mode="r")
            for field in MOTION_FIELDS
        }
        motion_completed = np.load(self.motion_cache / "completed.npy", mmap_mode="r")
        if len(motion_completed) != len(self.rows) or not np.asarray(motion_completed).all():
            raise RuntimeError("challenge Test motion cache is incomplete")
        self.view_valid = np.load(self.pixel_cache / "view_valid.npy", mmap_mode="r")
        self.view_quality = np.load(self.pixel_cache / "view_quality.npy", mmap_mode="r")
        self.source_frame_indices = np.load(
            self.pixel_cache / "source_frame_indices.npy", mmap_mode="r"
        )
        self.indices = (
            np.arange(len(self.rows), dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        self.temporal_augment = bool(temporal_augment)

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
                (base - 0.5 * (steps - 1)) * scale
                + 0.5 * (steps - 1)
                + shift,
                0.0,
                float(steps - 1),
            )
            result = np.rint(positions).astype(np.int64)
        return result

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = int(self.indices[item])
        sequence = np.asarray(self.backbone_sequence[index], dtype=np.float32).copy()
        valid = np.asarray(self.view_valid[index], dtype=bool).copy()
        quality = np.asarray(self.view_quality[index], dtype=np.float32).copy()
        source_indices = np.asarray(
            self.source_frame_indices[index], dtype=np.int64
        ).copy()
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
        valid = valid[:, temporal_index]
        quality = quality[:, temporal_index]
        source_indices = source_indices[:, temporal_index]
        for field in TEMPORAL_MOTION_FIELDS:
            motion[field] = motion[field][:, temporal_index]
        global_time = source_indices.astype(np.float32) / max(frame_count - 1, 1)
        result: dict[str, Any] = {
            "backbone_sequence": torch.from_numpy(sequence),
            "view_valid": torch.from_numpy(valid),
            "view_quality": torch.from_numpy(quality),
            "global_time_position": torch.from_numpy(global_time),
            "sample_id": self.rows[index]["sample_id"],
            "user_id": "anonymous",
            "cache_index": index,
        }
        for field, value in motion.items():
            if field in BOOLEAN_MOTION_FIELDS:
                result[field] = torch.from_numpy(value.astype(bool, copy=False))
            else:
                result[field] = torch.from_numpy(value.astype(np.float32, copy=False))
        return result


def collate_test(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("empty challenge Test batch")
    output: dict[str, Any] = {}
    for key in items[0]:
        if key in {"sample_id", "user_id", "cache_index"}:
            output[key] = [item[key] for item in items]
        else:
            output[key] = torch.stack([item[key] for item in items])
    if "label" in output or any(key.startswith("teacher") for key in output):
        raise RuntimeError("label or teacher field entered challenge Test batch")
    return output
