from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from shared_dir_roi_model import MODALITY_NAMES, REGION_NAMES


def safe_name(sample_id: str) -> Path:
    parts = sample_id.split("/")
    if len(parts) != 3 or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"unsafe sample_id: {sample_id}")
    return Path(*parts)


class SharedDIRFeatureDataset(Dataset[dict[str, Any]]):
    """Variable-length all-frame D/IR feature trials produced by roi_features."""

    def __init__(
        self,
        feature_run: str | Path,
        sample_ids: set[str] | None = None,
    ) -> None:
        self.feature_run = Path(feature_run).resolve()
        summary_path = self.feature_run / "trial_summary.csv"
        with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if sample_ids is not None:
            rows = [row for row in rows if row["sample_id"] in sample_ids]
        self.rows = sorted(
            rows,
            key=lambda row: (int(row["class_id"]), row["user_id"], row["trial_id"]),
        )
        if not self.rows:
            raise RuntimeError(f"no roi_features feature trials selected from {summary_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def cache_path(self, sample_id: str) -> Path:
        return (
            self.feature_run
            / "trial_feature_cache"
            / safe_name(sample_id).with_suffix(".npz")
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        with np.load(self.cache_path(row["sample_id"]), allow_pickle=False) as cache:
            modality_names = tuple(str(value) for value in cache["modality_names"])
            region_names = tuple(str(value) for value in cache["region_names"])
            if modality_names != MODALITY_NAMES or region_names != REGION_NAMES:
                raise RuntimeError(f"roi_features cache layout mismatch: {row['sample_id']}")
            features = torch.from_numpy(cache["features"].astype(np.float32))
            roi_valid = torch.from_numpy(cache["roi_valid"].astype(bool))
            roi_quality = torch.from_numpy(cache["roi_quality"].astype(np.float32))
            roi_source = torch.from_numpy(cache["roi_source"].astype(np.int64))
            clipped = torch.from_numpy(cache["roi_clipped_ratio"].astype(np.float32))
            pose_factor = torch.from_numpy(cache["pose_quality_factor"].astype(np.float32))
            frame_ids = [str(value) for value in cache["frame_ids"]]
        time_steps = len(frame_ids)
        time_position = (
            torch.linspace(0.0, 1.0, time_steps)
            if time_steps > 1
            else torch.zeros(1)
        )
        return {
            "sample_id": row["sample_id"],
            "class_id": int(row["class_id"]),
            "user_id": row["user_id"],
            "frame_ids": frame_ids,
            "features": features,
            "roi_valid": roi_valid,
            "roi_quality": roi_quality,
            "roi_source": roi_source,
            "roi_clipped_ratio": clipped,
            "pose_quality_factor": pose_factor,
            "time_position": time_position,
        }


def collate_trials(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("empty roi_features batch")
    batch_size = len(items)
    maximum = max(len(item["features"]) for item in items)
    feature_dim = items[0]["features"].shape[-1]
    features = torch.zeros(
        batch_size,
        maximum,
        len(MODALITY_NAMES),
        len(REGION_NAMES),
        feature_dim,
        dtype=torch.float32,
    )
    roi_valid = torch.zeros(batch_size, maximum, len(REGION_NAMES), dtype=torch.bool)
    roi_quality = torch.zeros(batch_size, maximum, len(REGION_NAMES))
    roi_source = torch.zeros(batch_size, maximum, len(REGION_NAMES), dtype=torch.long)
    clipped = torch.zeros(batch_size, maximum, len(REGION_NAMES))
    pose_factor = torch.zeros(batch_size, maximum)
    time_position = torch.zeros(batch_size, maximum)
    frame_mask = torch.zeros(batch_size, maximum, dtype=torch.bool)
    labels = torch.empty(batch_size, dtype=torch.long)
    for batch_index, item in enumerate(items):
        length = len(item["features"])
        features[batch_index, :length] = item["features"]
        roi_valid[batch_index, :length] = item["roi_valid"]
        roi_quality[batch_index, :length] = item["roi_quality"]
        roi_source[batch_index, :length] = item["roi_source"]
        clipped[batch_index, :length] = item["roi_clipped_ratio"]
        pose_factor[batch_index, :length] = item["pose_quality_factor"]
        time_position[batch_index, :length] = item["time_position"]
        frame_mask[batch_index, :length] = True
        labels[batch_index] = int(item["class_id"])
    return {
        "sample_id": [item["sample_id"] for item in items],
        "user_id": [item["user_id"] for item in items],
        "frame_ids": [item["frame_ids"] for item in items],
        "features": features,
        "roi_valid": roi_valid,
        "roi_quality": roi_quality,
        "roi_source": roi_source,
        "roi_clipped_ratio": clipped,
        "pose_quality_factor": pose_factor,
        "time_position": time_position,
        "frame_mask": frame_mask,
        "label": labels,
    }
