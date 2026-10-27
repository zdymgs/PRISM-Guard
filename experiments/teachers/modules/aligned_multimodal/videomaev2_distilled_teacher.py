from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from torch.utils.data import DataLoader
from transformers import VideoMAEImageProcessor
from aligned_multimodal.teacher_common import REPO_ROOT, classification_metrics, load_protocol
from aligned_multimodal.videomae_lora_teacher import EvalTrialDataset, VideoCollator, read_aligned_rows
MODEL_REPO = 'OpenGVLab/VideoMAE2'
MODEL_FILE = 'distill/vit_b_k710_dl_from_giant.pth'
PROCESSOR_REPO = 'MCG-NJU/videomae-large-finetuned-kinetics'
EXTERNAL_REPO = EXTERNAL_ROOT / 'VideoMAEv2'
DEFAULT_OUTPUT = REPO_ROOT / 'runs' / 'videomaev2_distilled_teacher'

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--trial-batch', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--reuse-cache', action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()

def build_model(device: torch.device) -> tuple[torch.nn.Module, Path]:
    if not (EXTERNAL_REPO / 'models/modeling_finetune.py').exists():
        raise FileNotFoundError(f'Official VideoMAE V2 repository is missing: {EXTERNAL_REPO}')
    sys.path.insert(0, str(EXTERNAL_REPO.resolve()))
    from models.modeling_finetune import vit_base_patch16_224
    checkpoint = Path(hf_hub_download(MODEL_REPO, MODEL_FILE, local_files_only=True))
    model = vit_base_patch16_224(num_classes=710, all_frames=16, tubelet_size=2, use_mean_pooling=True)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)['module']
    model.load_state_dict(state, strict=True)
    model.eval().to(device=device, dtype=torch.float16)
    return (model, checkpoint)

@torch.inference_mode()
def extract(args: argparse.Namespace, cache: Path) -> dict[str, np.ndarray]:
    protocol = load_protocol()
    rows = read_aligned_rows()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, checkpoint = build_model(device)
    processor_snapshot = Path(snapshot_download(PROCESSOR_REPO, local_files_only=True))
    processor = VideoMAEImageProcessor.from_pretrained(processor_snapshot, local_files_only=True)
    dataset = EvalTrialDataset(rows, protocol.labels, 'ir')
    loader = DataLoader(dataset, batch_size=args.trial_batch, shuffle=False, num_workers=args.num_workers, collate_fn=VideoCollator(processor), pin_memory=True)
    feature_parts = []
    action_parts = []
    sample_ids: list[str] = []
    started = time.time()
    torch.cuda.reset_peak_memory_stats() if device.type == 'cuda' else None
    for batch_number, batch in enumerate(loader):
        pixels = batch['pixel_values'].permute(0, 2, 1, 3, 4)
        pixels = pixels.to(device=device, dtype=torch.float16, non_blocking=True)
        features = model.forward_features(pixels)
        action_logits = model.head(features)
        trials = len(batch['sample_ids'])
        if features.shape[0] != trials * 6:
            raise ValueError('IR batch did not contain six clips per trial')
        feature_parts.append(features.reshape(trials, 6, 768).cpu().numpy())
        action_parts.append(action_logits.reshape(trials, 6, 710).cpu().numpy())
        sample_ids.extend(batch['sample_ids'])
        if (batch_number + 1) % 20 == 0:
            print(f'  extracted trials={len(sample_ids)}/{len(dataset)} elapsed_min={(time.time() - started) / 60:.1f}', flush=True)
    if sample_ids != protocol.sample_ids.tolist():
        raise ValueError('VideoMAE V2 extraction order changed')
    payload = {'sample_ids': protocol.sample_ids, 'labels': protocol.labels, 'users': protocol.users, 'fold_id': protocol.fold_id, 'features': np.concatenate(feature_parts).reshape(-1, 2, 3, 768), 'action_logits': np.concatenate(action_parts).reshape(-1, 2, 3, 710)}
    np.savez_compressed(cache, **payload)
    extraction = {'model_repo': MODEL_REPO, 'model_file': MODEL_FILE, 'checkpoint': str(checkpoint), 'official_repo': str(DATA_ROOT / 'VideoMAEv2'), 'checkpoint_license': 'Hugging Face repository metadata: cc-by-nc-4.0', 'parameters': int(sum((parameter.numel() for parameter in model.parameters()))), 'clips': 'early/late x scene/person/workspace', 'processor': PROCESSOR_REPO, 'elapsed_seconds': time.time() - started, 'peak_cuda_gib': float(torch.cuda.max_memory_allocated() / 2 ** 30) if device.type == 'cuda' else 0.0, 'cache': str(cache)}
    (cache.parent / 'cache_summary.json').write_text(json.dumps(extraction, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload

def class_sample_weights(labels: np.ndarray, power: float) -> np.ndarray:
    counts = np.bincount(labels, minlength=40).astype(np.float64)
    reference = counts[counts > 0].mean()
    weights = np.zeros(40, dtype=np.float64)
    present = counts > 0
    weights[present] = np.power(reference / counts[present], power)
    values = weights[labels]
    return values / values.mean()
