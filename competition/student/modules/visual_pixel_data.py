from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class VisualPixelDataset(Dataset[dict[str, Any]]):
    """Memory-mapped student pixels aligned exactly to foundation training-only targets."""

    def __init__(
        self,
        cache_dir: str | Path,
        teacher_features: str | Path,
        teacher_logits: str | Path,
        indices: np.ndarray | list[int] | None = None,
        augment: bool = False,
        augmentation_mode: str = "basic",
    ) -> None:
        self.cache_dir = Path(cache_dir).resolve()
        with (self.cache_dir / "rows.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            self.rows = list(csv.DictReader(handle))
        self.images = np.load(self.cache_dir / "images.npy", mmap_mode="r")
        self.completed = np.load(self.cache_dir / "completed.npy", mmap_mode="r")
        self.view_valid = np.load(self.cache_dir / "view_valid.npy", mmap_mode="r")
        self.view_quality = np.load(self.cache_dir / "view_quality.npy", mmap_mode="r")
        source_index_path = self.cache_dir / "source_frame_indices.npy"
        self.source_frame_indices = (
            np.load(source_index_path, mmap_mode="r") if source_index_path.exists() else None
        )
        if len(self.rows) != len(self.images) or not np.asarray(self.completed).all():
            raise RuntimeError("student pixel cache is incomplete")
        with np.load(Path(teacher_features).resolve(), allow_pickle=False) as data:
            feature_ids = np.asarray(data["sample_ids"]).astype(str)
            feature_index = {sample_id: index for index, sample_id in enumerate(feature_ids)}
            teacher_features_all = np.asarray(data["features"], dtype=np.float32)
        with np.load(Path(teacher_logits).resolve(), allow_pickle=False) as data:
            logit_ids = np.asarray(data["sample_ids"]).astype(str)
            logit_index = {sample_id: index for index, sample_id in enumerate(logit_ids)}
            logits_all = np.asarray(data["early_late_logits"], dtype=np.float32)
            early_logits_all = np.asarray(data["early_logits"], dtype=np.float32)
            late_logits_all = np.asarray(data["late_logits"], dtype=np.float32)
            temporal_delta_logits_all = np.asarray(
                data["temporal_delta_logits"], dtype=np.float32
            )
            labels_all = np.asarray(data["labels"], dtype=np.int64)
            users_all = np.asarray(data["users"]).astype(str)
            folds_all = np.asarray(data["folds"], dtype=np.int64)
        sample_ids = [row["sample_id"] for row in self.rows]
        missing = [
            sample_id
            for sample_id in sample_ids
            if sample_id not in feature_index or sample_id not in logit_index
        ]
        if missing:
            raise RuntimeError(f"teacher targets missing student pixels: {missing[:3]}")
        self.teacher_features = np.stack(
            [teacher_features_all[feature_index[sample_id]] for sample_id in sample_ids]
        )
        logit_rows = np.asarray([logit_index[sample_id] for sample_id in sample_ids], dtype=np.int64)
        self.teacher_logits = logits_all[logit_rows]
        self.teacher_early_logits = early_logits_all[logit_rows]
        self.teacher_late_logits = late_logits_all[logit_rows]
        self.teacher_temporal_delta_logits = temporal_delta_logits_all[logit_rows]
        self.labels = labels_all[logit_rows]
        self.users = users_all[logit_rows]
        self.folds = folds_all[logit_rows]
        manifest_labels = np.asarray([int(row["class_id"]) for row in self.rows])
        if not np.array_equal(manifest_labels, self.labels):
            raise RuntimeError("student pixel/teacher label mismatch")
        self.indices = (
            np.arange(len(self.rows), dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        self.augment = bool(augment)
        if augmentation_mode not in {
            "basic",
            "subject_robust",
            "subject_robust_no_flip",
        }:
            raise ValueError(f"unknown student augmentation mode: {augmentation_mode}")
        self.augmentation_mode = augmentation_mode

    @staticmethod
    def _subject_robust_augment(
        images: np.ndarray, allow_horizontal_flip: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply label-preserving transforms coherently across all six clips."""
        if allow_horizontal_flip and np.random.random() < 0.5:
            images = images[..., ::-1].copy()

        height, width = images.shape[-2:]
        pad = max(2, int(round(min(height, width) * 0.055)))
        padded = np.pad(
            images,
            ((0, 0), (0, 0), (0, 0), (pad, pad), (pad, pad)),
            mode="reflect",
        )
        top = int(np.random.randint(0, 2 * pad + 1))
        left = int(np.random.randint(0, 2 * pad + 1))
        images = padded[..., top : top + height, left : left + width]

        steps = images.shape[1]
        temporal_indices = np.arange(steps, dtype=np.int64)
        if np.random.random() < 0.75:
            scale = float(np.random.uniform(0.86, 1.14))
            shift = float(np.random.uniform(-0.07, 0.07) * max(steps - 1, 1))
            base = np.arange(steps, dtype=np.float32)
            positions = np.clip(
                (base - 0.5 * (steps - 1)) * scale + 0.5 * (steps - 1) + shift,
                0.0,
                float(steps - 1),
            )
            temporal_indices = np.rint(positions).astype(np.int64)
            images = images[:, temporal_indices]

        gain = float(np.random.uniform(0.82, 1.18))
        bias = float(np.random.uniform(-14.0, 14.0))
        # A 256-value LUT avoids allocating and transforming a multi-million
        # pixel float32 array for every sample. The transform remains coherent
        # across time and views, which is the intended sensor-domain shift.
        lookup = np.clip(np.arange(256, dtype=np.float32) * gain + bias, 0.0, 255.0)
        return lookup.astype(np.uint8)[images], temporal_indices

    @property
    def index_lookup(self) -> dict[str, int]:
        return {row["sample_id"]: index for index, row in enumerate(self.rows)}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = int(self.indices[item])
        # Explicit copies keep read-only memmaps from leaking into writable tensors.
        images = np.asarray(self.images[index]).copy()
        view_valid = np.asarray(self.view_valid[index], dtype=bool).copy()
        view_quality = np.asarray(self.view_quality[index], dtype=np.float32).copy()
        if self.source_frame_indices is not None:
            source_frame_index = np.asarray(
                self.source_frame_indices[index], dtype=np.int64
            ).copy()
            exact_source_time = True
        else:
            source_frame_index = np.broadcast_to(
                np.arange(images.shape[1], dtype=np.int64)[None],
                (images.shape[0], images.shape[1]),
            ).copy()
            exact_source_time = False
        # The late window includes the final trial frame. Capture the full count
        # before temporal augmentation so shifted clips do not get renormalized
        # to a false endpoint of 1.0.
        source_frame_count = int(source_frame_index.max(initial=0)) + 1
        temporal_source_index = np.arange(images.shape[1], dtype=np.int64)
        if self.augment:
            if self.augmentation_mode in {
                "subject_robust",
                "subject_robust_no_flip",
            }:
                images, temporal_source_index = self._subject_robust_augment(
                    images,
                    allow_horizontal_flip=self.augmentation_mode == "subject_robust",
                )
                view_valid = view_valid[:, temporal_source_index]
                view_quality = view_quality[:, temporal_source_index]
                source_frame_index = source_frame_index[:, temporal_source_index]
            elif np.random.random() < 0.5:
                images = images[..., ::-1].copy()
        global_time_position = source_frame_index.astype(np.float32) / max(
            source_frame_count - 1, 1
        )
        return {
            "images": torch.from_numpy(images),
            "view_valid": torch.from_numpy(view_valid),
            "view_quality": torch.from_numpy(view_quality),
            "temporal_source_index": torch.from_numpy(temporal_source_index),
            "source_frame_index": torch.from_numpy(source_frame_index),
            "global_time_position": torch.from_numpy(global_time_position),
            "source_frame_count": torch.tensor(source_frame_count, dtype=torch.long),
            "exact_source_time": torch.tensor(exact_source_time, dtype=torch.bool),
            "teacher_features": torch.from_numpy(self.teacher_features[index]),
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


def collate_pixels(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("empty student pixel batch")
    output: dict[str, Any] = {}
    for key in (
        "images",
        "view_valid",
        "view_quality",
        "teacher_features",
        "teacher_logits",
        "teacher_early_logits",
        "teacher_late_logits",
        "teacher_temporal_delta_logits",
        "temporal_source_index",
        "source_frame_index",
        "global_time_position",
        "source_frame_count",
        "exact_source_time",
        "label",
    ):
        output[key] = torch.stack([item[key] for item in items])
    for key in ("sample_id", "user_id", "cache_index"):
        output[key] = [item[key] for item in items]
    return output
