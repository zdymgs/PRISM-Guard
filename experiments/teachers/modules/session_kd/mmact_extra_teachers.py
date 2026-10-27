from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import hashlib
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from session_kd.mmact_imu import NativeIMUSchema, apply_channel_normalizer, assemble_part_grid, fit_channel_normalizer
from session_kd.mmact_motion_pretrain import build_mmact_motion_pretrain
from session_kd.mmact_skeleton import projected_skeleton_features
FITS = ('F2', 'F3', 'F4', 'final_source')
FAMILIES = ('mc3_motion', 'native_skeleton', 'sequential_imu')

def ordered_hash(values: Sequence[str]) -> str:
    return hashlib.sha256(('\n'.join(map(str, values)) + '\n').encode()).hexdigest()

def fit_predict_indices(source_indices: np.ndarray, source_folds: np.ndarray, fit_name: str):
    source_indices = np.asarray(source_indices, np.int64)
    source_folds = np.asarray(source_folds, str)
    if fit_name == 'final_source':
        return (source_indices.copy(), np.asarray([], np.int64))
    if fit_name not in {'F2', 'F3', 'F4'}:
        raise ValueError('unknown cross-fit name')
    fit = source_indices[source_folds != fit_name]
    predict = source_indices[source_folds == fit_name]
    if set(map(int, fit)) & set(map(int, predict)):
        raise AssertionError('cross-fit exclusion failed')
    return (fit, predict)

def ce_only_loss(outputs: dict[str, torch.Tensor], labels: torch.Tensor, skeleton_available: torch.Tensor, imu_available: torch.Tensor, class_weight: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Two supervised CE heads only: no KD, alignment, reconstruction or ancestor."""
    losses = {}
    if skeleton_available.any():
        losses['native_skeleton_ce'] = nn.functional.cross_entropy(outputs['skeleton_logits'][skeleton_available], labels[skeleton_available], weight=class_weight, label_smoothing=0.08)
    if imu_available.any():
        losses['sequential_imu_ce'] = nn.functional.cross_entropy(outputs['imu_logits'][imu_available], labels[imu_available], weight=class_weight, label_smoothing=0.08)
    if not losses:
        raise ValueError('batch has neither native Skeleton nor IMU supervision')
    return (sum(losses.values()), losses)

def raw_imu_grid(archive) -> tuple[np.ndarray, np.ndarray, NativeIMUSchema]:
    value = archive['imu_raw_phone_watch'].astype(np.float32)
    mask = archive['imu_channel_mask'].astype(bool)
    return assemble_part_grid(value[..., 0:3][None], value[..., 6:9][None], value[..., 3:6][None], mask[..., 0:3][None], mask[..., 6:9][None], mask[..., 3:6][None])

def fit_extra_normalizer(cache: Path, indices: np.ndarray, interval_ids: np.ndarray):
    values, masks = ([], [])
    for index in map(int, indices):
        with np.load(cache / f'interval_{index:06d}.npz', allow_pickle=False) as archive:
            grid, valid, schema = raw_imu_grid(archive)
        values.append(grid[0])
        masks.append(valid[0])
    x = np.stack(values)
    m = np.stack(masks)
    return fit_channel_normalizer(x, m, interval_ids, np.arange(len(x)), schema)

class NativeMotionDataset(Dataset):

    def __init__(self, cache: Path, indices: np.ndarray, ids: np.ndarray, labels: np.ndarray, normalizer):
        self.cache = cache
        self.indices = np.asarray(indices, np.int64)
        self.ids = np.asarray(ids, str)
        self.labels = np.asarray(labels, np.int64)
        self.normalizer = normalizer
        if not len(self.indices) == len(self.ids) == len(self.labels) or np.any(self.labels < 0):
            raise ValueError('native dataset alignment/label firewall failed')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = int(self.indices[position])
        path = self.cache / f'interval_{index:06d}.npz'
        with np.load(path, allow_pickle=False) as z:
            if int(z['global_interval_index']) != index or str(z['interval_id']) != self.ids[position]:
                raise ValueError('native cache identity mismatch')
            grid, valid, _ = raw_imu_grid(z)
            grid = apply_channel_normalizer(grid, valid, self.normalizer)[0]
            valid = valid[0]
            pose = z['pose_coco2d'].astype(np.float32)
            pose_mask = z['pose_joint_mask'].astype(bool)
            frame_available = z['frame_available'].astype(bool)
            times = z['source_time_ms'].astype(np.float64) / 1000.0
        valid_pose = pose_mask & frame_available[..., None]
        pose[..., 2] = np.where(valid_pose, pose[..., 2], 0.0)
        pose[..., :2] = np.where(valid_pose[..., None], pose[..., :2], 0.0)
        skeleton = []
        for window in range(2):
            skeleton.append(projected_skeleton_features(pose[window], times[window]))
        motion = {key: np.stack([value[key] for value in skeleton]) for key in skeleton[0]}
        motion = {key: value.astype(np.float32) if np.issubdtype(value.dtype, np.floating) else value.astype(bool) for key, value in motion.items()}
        motion.update({'imu_sequences': grid, 'imu_sequence_mask': valid})
        motion.update({'imu_bin_statistics': np.zeros((2, 16, 5, 30), np.float32), 'imu_bin_mask': np.zeros((2, 16, 5, 30), bool), 'imu_global_statistics': np.zeros((2, 5, 30), np.float32), 'imu_global_mask': np.zeros((2, 5, 30), bool)})
        return {**{k: torch.from_numpy(v) for k, v in motion.items()}, 'label': torch.tensor(self.labels[position]), 'skeleton_available': torch.tensor(bool(motion['skeleton_joint_mask'].any())), 'imu_available': torch.tensor(bool(valid.any())), 'global_index': torch.tensor(index)}

def build_extra_model(classes: int=37):
    """Always construct from scratch; callers must never load a C0/KD ancestor."""
    model = build_mmact_motion_pretrain(classes=classes)
    model.training_ancestry = 'from_scratch_ce_only'
    return model

def native_recipe() -> dict[str, object]:
    return {'epochs': 24, 'batch_size': 64, 'optimizer': 'AdamW', 'lr': 0.0004, 'minimum_lr': 2e-05, 'weight_decay': 0.05, 'class_weight_power': 0.35, 'label_smoothing': 0.08, 'losses': ['native_skeleton_ce', 'sequential_imu_ce'], 'kd_weight': 0.0, 'alignment_weight': 0.0, 'reconstruction_weight': 0.0, 'initialization': 'from_scratch'}
