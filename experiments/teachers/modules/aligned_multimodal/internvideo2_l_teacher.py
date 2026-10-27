from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import argparse
import json
import sys
import time
import types
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from transformers import VideoMAEImageProcessor
from aligned_multimodal.teacher_common import REPO_ROOT, classification_metrics, load_protocol
from aligned_multimodal.videomae_lora_teacher import EvalTrialDataset, read_aligned_rows
MODEL_REPO = 'OpenGVLab/InternVideo2_distillation_models'
MODEL_FILE = Path(str(PRETRAINED_ROOT / 'huggingface/hub/')) / 'models--OpenGVLab--InternVideo2_distillation_models' / 'snapshots/449f7ea1d7d3b70b6b5630e70d238b44d3b7aaac' / 'stage1/L14/L14_ft_k710_ft_k400_f8/pytorch_model.bin'
OFFICIAL_ROOT = EXTERNAL_ROOT / 'InternVideo/InternVideo2/single_modality/models'
PROCESSOR_SNAPSHOT = Path(str(PRETRAINED_ROOT / 'huggingface/hub/models--MCG-NJU--videomae-large-finetuned-kinetics/snapshots/0f6adcd5f6902900aa0281f9daacfe52bb3c4ad4'))
DEFAULT_OUTPUT = REPO_ROOT / 'runs' / 'internvideo2_l_k400_teacher'

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--trial-batch', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--checkpoint-every', type=int, default=40)
    parser.add_argument('--reuse-cache', action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()

def import_official_model() -> Any:
    package = types.ModuleType('internvideo_models')
    package.__path__ = [str(OFFICIAL_ROOT)]
    sys.modules['internvideo_models'] = package
    from internvideo_models.internvideo2 import internvideo2_large_patch14_224
    return internvideo2_large_patch14_224

def build_model(device: torch.device) -> tuple[nn.Module, nn.Module]:
    constructor = import_official_model()
    model = constructor(num_classes=400, num_frames=8, tubelet_size=1, drop_path_rate=0.1, head_drop_path_rate=0.1, init_values=1e-05, use_flash_attn=False, use_fused_rmsnorm=False, use_fused_mlp=False)
    state = torch.load(MODEL_FILE, map_location='cpu', weights_only=False)
    model.load_state_dict(state, strict=True)
    head = model.head
    model.head = nn.Identity()
    model = model.eval().to(device=device, dtype=torch.bfloat16)
    head = head.eval().to(device=device, dtype=torch.bfloat16)
    return (model, head)

class EightFrameVideoCollator:

    def __init__(self, processor: VideoMAEImageProcessor) -> None:
        self.processor = processor

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        views = len(items[0]['videos'])
        videos = []
        for item in items:
            if len(item['videos']) != views:
                raise ValueError('mixed clip count')
            for video in item['videos']:
                positions = np.linspace(0, len(video) - 1, 8).round().astype(int)
                videos.append([video[position] for position in positions])
        pixels = self.processor(videos, return_tensors='pt').pixel_values
        return {'pixel_values': pixels, 'sample_ids': [item['sample_id'] for item in items]}

@torch.inference_mode()
def extract(args: argparse.Namespace, cache: Path) -> dict[str, np.ndarray]:
    protocol = load_protocol()
    rows = read_aligned_rows()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('InternVideo2-L fold-0 gate requires CUDA')
    model, head = build_model(device)
    processor = VideoMAEImageProcessor.from_pretrained(PROCESSOR_SNAPSHOT, local_files_only=True)
    dataset = EvalTrialDataset(rows, protocol.labels, 'ir')
    partial = cache.parent / 'partial_features.npz'
    feature_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    sample_ids: list[str] = []
    if partial.exists():
        with np.load(partial, allow_pickle=False) as source:
            sample_ids = source['sample_ids'].astype(str).tolist()
            feature_parts.append(np.asarray(source['features'], dtype=np.float16).reshape(-1, 6, 768))
            action_parts.append(np.asarray(source['action_logits'], dtype=np.float16).reshape(-1, 6, 400))
        expected = protocol.sample_ids[:len(sample_ids)].tolist()
        if sample_ids != expected:
            raise ValueError('InternVideo2 partial cache order changed')
        print(f'resuming partial trials={len(sample_ids)}/{len(dataset)}', flush=True)
    remaining_dataset = Subset(dataset, range(len(sample_ids), len(dataset)))
    loader = DataLoader(remaining_dataset, batch_size=args.trial_batch, shuffle=False, num_workers=args.num_workers, collate_fn=EightFrameVideoCollator(processor), pin_memory=True)
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for batch_number, batch in enumerate(loader):
        pixels = batch['pixel_values'].permute(0, 2, 1, 3, 4)
        pixels = pixels.to(device=device, dtype=torch.bfloat16, non_blocking=True)
        features = model(pixels)
        action_logits = head(features)
        trials = len(batch['sample_ids'])
        if features.shape != (trials * 6, 768):
            raise ValueError(f'Unexpected InternVideo2 feature shape: {features.shape}')
        feature_parts.append(features.reshape(trials, 6, 768).half().cpu().numpy())
        action_parts.append(action_logits.reshape(trials, 6, 400).half().cpu().numpy())
        sample_ids.extend(batch['sample_ids'])
        if (batch_number + 1) % 20 == 0:
            print(f'  extracted trials={len(sample_ids)}/{len(dataset)} elapsed_min={(time.time() - started) / 60:.1f}', flush=True)
        if (batch_number + 1) % args.checkpoint_every == 0:
            temporary = cache.parent / 'partial_features.tmp.npz'
            np.savez(temporary, sample_ids=np.asarray(sample_ids), features=np.concatenate(feature_parts).reshape(-1, 2, 3, 768), action_logits=np.concatenate(action_parts).reshape(-1, 2, 3, 400))
            temporary.replace(partial)
            with np.load(partial, allow_pickle=False) as source:
                feature_parts = [np.asarray(source['features']).reshape(-1, 6, 768)]
                action_parts = [np.asarray(source['action_logits']).reshape(-1, 6, 400)]
    if sample_ids != protocol.sample_ids.tolist():
        raise ValueError('InternVideo2 extraction order changed')
    payload = {'sample_ids': protocol.sample_ids, 'labels': protocol.labels, 'users': protocol.users, 'fold_id': protocol.fold_id, 'features': np.concatenate(feature_parts).reshape(-1, 2, 3, 768), 'action_logits': np.concatenate(action_parts).reshape(-1, 2, 3, 400)}
    np.savez_compressed(cache, **payload)
    metadata = {'model_repo': MODEL_REPO, 'model_file': str(MODEL_FILE), 'official_repo': str(DATA_ROOT / 'InternVideo'), 'official_commit': '3965eef16e2dadd0ea6c8d0cc29c8a3039df52e3', 'license': 'official GitHub repository Apache-2.0; HF weight repo lacks metadata', 'pretraining': 'K-Mash, distilled from InternVideo2 Stage2 1B, K710 then K400 fine-tuning', 'official_k400_top1': 90.4, 'parameters': int(sum((parameter.numel() for parameter in model.parameters())) + sum((parameter.numel() for parameter in head.parameters()))), 'frames_per_clip': 8, 'clips': 'early/late x scene/person/workspace', 'elapsed_seconds': time.time() - started, 'peak_cuda_gib': float(torch.cuda.max_memory_allocated() / 2 ** 30)}
    (cache.parent / 'cache_summary.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload
