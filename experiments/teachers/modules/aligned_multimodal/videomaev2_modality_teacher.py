from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import argparse
import json
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader, Dataset
from transformers import VideoMAEImageProcessor
from aligned_multimodal.build_videomae_modality_cache import prepare_trial as prepare_modality_trial, thermal_dir
from aligned_multimodal.teacher_common import REPO_ROOT, classification_metrics, load_protocol
from aligned_multimodal.videomae_lora_teacher import ROI_RUN, VideoCollator, read_aligned_rows
from aligned_multimodal.videomaev2_distilled_teacher import PROCESSOR_REPO, build_model, class_sample_weights
DEFAULT_OUTPUT = REPO_ROOT / 'runs/videomaev2_depth_fold0'

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', choices=('depth', 'thermal'), default='depth')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--trial-batch', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--reuse-cache', action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()

class ModalityTrialDataset(Dataset):

    def __init__(self, rows: list[dict[str, str]], labels: np.ndarray, modality: str) -> None:
        self.rows = rows
        self.labels = labels
        self.modality = modality

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        available = self.modality != 'thermal' or any(thermal_dir(row).glob('*.jpg'))
        if available:
            clips, _ = prepare_modality_trial(row, ROI_RUN, self.modality)
        else:
            black = np.zeros((224, 224, 3), dtype=np.uint8)
            clips = [[black] * 16 for _ in range(3)]
        if len(clips) != 3:
            raise ValueError(f'{self.modality} did not produce three ROI clips')
        return {'videos': clips, 'clip_indices': [0, 1, 2], 'label': int(self.labels[index]), 'sample_id': row['sample_id']}

@torch.inference_mode()
def extract(args: argparse.Namespace, cache: Path) -> dict[str, np.ndarray]:
    protocol = load_protocol()
    rows = read_aligned_rows()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, checkpoint = build_model(device)
    processor_snapshot = Path(snapshot_download(PROCESSOR_REPO, local_files_only=True))
    processor = VideoMAEImageProcessor.from_pretrained(processor_snapshot, local_files_only=True)
    dataset = ModalityTrialDataset(rows, protocol.labels, args.modality)
    loader = DataLoader(dataset, batch_size=args.trial_batch, shuffle=False, num_workers=args.num_workers, collate_fn=VideoCollator(processor), pin_memory=True)
    feature_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    sample_ids: list[str] = []
    started = time.time()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    for batch_number, batch in enumerate(loader):
        pixels = batch['pixel_values'].permute(0, 2, 1, 3, 4)
        pixels = pixels.to(device=device, dtype=torch.float16, non_blocking=True)
        features = model.forward_features(pixels)
        action_logits = model.head(features)
        trials = len(batch['sample_ids'])
        if features.shape[0] != trials * 3:
            raise ValueError(f'{args.modality} batch did not contain three clips per trial')
        feature_parts.append(features.reshape(trials, 3, 768).cpu().numpy())
        action_parts.append(action_logits.reshape(trials, 3, 710).cpu().numpy())
        sample_ids.extend(batch['sample_ids'])
        if (batch_number + 1) % 20 == 0:
            print(f'  {args.modality} extracted={len(sample_ids)}/{len(dataset)} elapsed_min={(time.time() - started) / 60:.1f}', flush=True)
    if sample_ids != protocol.sample_ids.tolist():
        raise ValueError('extraction order changed')
    features = np.concatenate(feature_parts)
    action_logits = np.concatenate(action_parts)
    modality_available = np.asarray([args.modality != 'thermal' or any(thermal_dir(row).glob('*.jpg')) for row in rows], dtype=np.uint8)
    features[modality_available == 0] = 0
    action_logits[modality_available == 0] = 0
    payload = {'sample_ids': protocol.sample_ids, 'labels': protocol.labels, 'users': protocol.users, 'fold_id': protocol.fold_id, 'features': features, 'action_logits': action_logits, 'modality_available': modality_available, 'modality': np.asarray(args.modality)}
    np.savez_compressed(cache, **payload)
    summary = {'modality': args.modality, 'checkpoint': str(checkpoint), 'clips': 'scene/person/workspace', 'samples': len(sample_ids), 'available_samples': int(modality_available.sum()), 'elapsed_seconds': time.time() - started, 'peak_cuda_gib': float(torch.cuda.max_memory_allocated() / 2 ** 30) if device.type == 'cuda' else 0.0}
    (cache.parent / 'cache_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload
