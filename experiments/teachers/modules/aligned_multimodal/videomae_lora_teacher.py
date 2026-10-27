from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import csv
from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from aligned_multimodal.build_videomae_modality_cache import prepare_trial as prepare_modality_trial
from aligned_multimodal.build_videomae_multiclip_cache import WINDOW_BOUNDS, prepare_trial as prepare_ir_trial, window_indices
from aligned_multimodal.teacher_common import HERE, NUM_CLASSES, REPO_ROOT, classification_metrics, load_protocol, save_oof_artifact, seed_everything
FULL_MANIFEST = HERE / 'data' / 'single_split.csv'
ROI_RUN = HERE / 'runs' / 'dir_multiscale_roi_full'

def read_aligned_rows() -> list[dict[str, str]]:
    protocol = load_protocol()
    with FULL_MANIFEST.open('r', encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    lookup = {row['sample_id']: row for row in rows}
    if set(lookup) != set(protocol.sample_ids):
        raise ValueError('video full manifest does not match teacher master manifest')
    return [lookup[sample_id] for sample_id in protocol.sample_ids]

def prepare_clips(row: dict[str, str], modality: str) -> list[list[np.ndarray]]:
    if modality == 'ir':
        clips, _ = prepare_ir_trial(row, ROI_RUN)
    elif modality == 'depth':
        clips, _ = prepare_modality_trial(row, ROI_RUN, 'depth')
    else:
        raise ValueError(modality)
    return clips

class EvalTrialDataset(Dataset):

    def __init__(self, rows: list[dict[str, str]], labels: np.ndarray, modality: str) -> None:
        self.rows = rows
        self.labels = labels
        self.modality = modality

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {'videos': (clips := prepare_clips(self.rows[index], self.modality)), 'clip_indices': list(range(len(clips))), 'label': int(self.labels[index]), 'sample_id': self.rows[index]['sample_id']}

class VideoCollator:

    def __init__(self, processor: VideoMAEImageProcessor) -> None:
        self.processor = processor

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        views = len(items[0]['videos'])
        if any((len(item['videos']) != views for item in items)):
            raise ValueError('mixed clip count')
        videos = [video for item in items for video in item['videos']]
        clip_indices = [index for item in items for index in item['clip_indices']]
        pixel_values = self.processor(videos, return_tensors='pt').pixel_values
        return {'pixel_values': pixel_values, 'labels': torch.tensor([item['label'] for item in items], dtype=torch.long), 'clip_indices': torch.tensor(clip_indices, dtype=torch.long), 'sample_ids': [item['sample_id'] for item in items], 'views': views}
