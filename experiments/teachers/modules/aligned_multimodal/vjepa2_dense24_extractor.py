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
from transformers import AutoConfig, AutoVideoProcessor, VJEPA2ForVideoClassification
from aligned_multimodal.audit_yolo11_pose_skeleton import frame_map
from aligned_multimodal.build_shared_dir_roi_feature_cache import read_ir
from aligned_multimodal.build_videomae_cache import safe_relative, square_crop
from aligned_multimodal.teacher_common import REPO_ROOT, load_protocol
from aligned_multimodal.videomae_lora_teacher import ROI_RUN, read_aligned_rows
from aligned_multimodal.videomaev2_hand_teacher import centers, fallback_box, interaction_box
from aligned_multimodal.vjepa2_visual_teacher import bounded_indices, model_ready_video, one_trial
MODEL_REPO = 'facebook/vjepa2-vitl-fpc16-256-ssv2'
DEFAULT_OUTPUT = REPO_ROOT / 'runs/vjepa2_vitl_ssv2_dense24_fold0'
GLOBAL_BOUNDS = ((0.0, 1.0), (0.0, 0.5), (0.25, 0.75), (0.5, 1.0))
GLOBAL_NAMES = ('full', 'early', 'middle', 'late')
HAND_NAMES = ('full', 'early', 'late', 'motion_peak')
VIEW_COUNT = 24

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--clip-batch', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--flush-every', type=int, default=20)
    parser.add_argument('--max-samples', type=int, default=0)
    return parser.parse_args()

def roi_data(row: dict[str, str], roi_run: Path | None=None) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    root = ROI_RUN if roi_run is None else Path(roi_run)
    cache = root / 'trial_roi_cache' / safe_relative(row['source_id']).with_suffix('.npz')
    with np.load(cache, allow_pickle=False) as data:
        return (np.asarray(data['frame_ids']).astype(str), np.asarray(data['region_names']).astype(str).tolist(), np.asarray(data['roi_boxes_xyxy'], dtype=np.float32), np.asarray(data['roi_valid'], dtype=bool))

def ratio_indices(length: int, low: float, high: float, count: int) -> np.ndarray:
    start = int(round(low * length))
    stop = int(round(high * length))
    start = min(max(start, 0), max(length - 1, 0))
    stop = min(max(stop, start + 1), length)
    return bounded_indices(start, stop, count)

def dense_global_clips(row: dict[str, str], count: int, roi_run: Path | None=None) -> list[list[np.ndarray]]:
    frame_ids, names, boxes, valid = roi_data(row, roi_run)
    person = names.index('full_body')
    workspace = names.index('hand_workspace')
    paths = frame_map(Path(row['ir_dir']), 'ir')
    clips: list[list[np.ndarray]] = []
    for low, high in GLOBAL_BOUNDS:
        scene_video: list[np.ndarray] = []
        person_video: list[np.ndarray] = []
        workspace_video: list[np.ndarray] = []
        for frame in ratio_indices(len(frame_ids), low, high, count):
            image = read_ir(paths[frame_ids[frame]])
            person_box = boxes[frame, person] if valid[frame, person] else np.full(4, np.nan, dtype=np.float32)
            workspace_box = boxes[frame, workspace] if valid[frame, workspace] else person_box
            scene_video.append(image)
            person_video.append(square_crop(image, person_box, scale=1.15))
            workspace_video.append(square_crop(image, workspace_box, scale=1.4))
        clips.extend((scene_video, person_video, workspace_video))
    return clips

def dense_hand_clips(row: dict[str, str], count: int, roi_run: Path | None=None) -> list[list[np.ndarray]]:
    frame_ids, names, boxes, valid = roi_data(row, roi_run)
    left = names.index('left_hand')
    right = names.index('right_hand')
    workspace = names.index('hand_workspace')
    person = names.index('full_body')
    left_center = centers(boxes[:, left], valid[:, left])
    right_center = centers(boxes[:, right], valid[:, right])
    velocity = np.zeros(len(boxes), dtype=np.float32)
    if len(boxes) > 1:
        velocity[1:] = np.linalg.norm(np.diff(left_center, axis=0), axis=1)
        velocity[1:] += np.linalg.norm(np.diff(right_center, axis=0), axis=1)
    if len(velocity) >= 5:
        velocity = np.convolve(velocity, np.ones(5, dtype=np.float32) / 5.0, mode='same')
    center = int(np.argmax(velocity))
    span = min(len(boxes), max(count, int(round(len(boxes) * 0.45))))
    motion_low = min(max(center - span // 2, 0), max(len(boxes) - span, 0))
    windows = [ratio_indices(len(frame_ids), 0.0, 1.0, count), ratio_indices(len(frame_ids), 0.0, 0.5, count), ratio_indices(len(frame_ids), 0.5, 1.0, count), bounded_indices(motion_low, motion_low + span, count)]
    paths = frame_map(Path(row['ir_dir']), 'ir')
    clips: list[list[np.ndarray]] = []
    for chosen in windows:
        left_video: list[np.ndarray] = []
        right_video: list[np.ndarray] = []
        interaction_video: list[np.ndarray] = []
        for frame in chosen:
            image = read_ir(paths[frame_ids[frame]])
            left_box = fallback_box(boxes, valid, frame, left, workspace, person)
            right_box = fallback_box(boxes, valid, frame, right, workspace, person)
            both_box = interaction_box(boxes, valid, frame, left, right, workspace, person)
            left_video.append(square_crop(image, left_box, scale=1.85))
            right_video.append(square_crop(image, right_box, scale=1.85))
            interaction_video.append(square_crop(image, both_box, scale=1.55))
        clips.extend((left_video, right_video, interaction_video))
    return clips

class Dense24Dataset(Dataset[dict[str, Any]]):

    def __init__(self, rows: list[dict[str, str]], indices: np.ndarray, frames_per_clip: int, roi_run: Path | None=None) -> None:
        self.rows = rows
        self.indices = np.asarray(indices, dtype=np.int64)
        self.frames_per_clip = int(frames_per_clip)
        self.roi_run = Path(roi_run) if roi_run is not None else None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row_index = int(self.indices[index])
        row = self.rows[row_index]
        videos = [*dense_global_clips(row, self.frames_per_clip, self.roi_run), *dense_hand_clips(row, self.frames_per_clip, self.roi_run)]
        if len(videos) != VIEW_COUNT:
            raise ValueError(f'expected {VIEW_COUNT} views, got {len(videos)}')
        return {'row': row_index, 'sample_id': row['sample_id'], 'videos': videos}

def open_cache(output: Path, samples: int, hidden: int, labels: int) -> tuple[np.memmap, np.memmap, np.memmap]:
    specs = (('features.npy', np.float16, (samples, VIEW_COUNT, hidden)), ('ssv2_logits.npy', np.float16, (samples, VIEW_COUNT, labels)), ('done.npy', np.bool_, (samples,)))
    arrays = []
    for filename, dtype, shape in specs:
        path = output / filename
        if path.exists():
            value = np.lib.format.open_memmap(path, mode='r+')
            if value.shape != shape or value.dtype != dtype:
                raise ValueError(f'cache mismatch: {path} {value.shape}/{value.dtype}')
        else:
            value = np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)
            value[...] = False if dtype == np.bool_ else 0
            value.flush()
        arrays.append(value)
    return (arrays[0], arrays[1], arrays[2])

def write_progress(output: Path, snapshot: Path, done: np.ndarray, started: float, peak_gib: float, complete: bool) -> None:
    view_names = [*[f'global_{window}_{roi}' for window in GLOBAL_NAMES for roi in ('scene', 'person', 'workspace')], *[f'hand_{window}_{roi}' for window in HAND_NAMES for roi in ('left', 'right', 'interaction')]]
    (output / 'cache_summary.json').write_text(json.dumps({'model_repo': MODEL_REPO, 'snapshot': str(snapshot), 'label_free_extraction': True, 'view_count': VIEW_COUNT, 'view_names': view_names, 'completed_samples': int(done.sum()), 'total_samples': int(len(done)), 'complete': complete, 'elapsed_seconds_this_run': time.time() - started, 'peak_cuda_gib': peak_gib}, ensure_ascii=False, indent=2), encoding='utf-8')

@torch.inference_mode()
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol()
    rows = read_aligned_rows()
    if [row['sample_id'] for row in rows] != protocol.sample_ids.tolist():
        raise ValueError('row order differs from protocol')
    snapshot = Path(snapshot_download(MODEL_REPO, allow_patterns=('*.json', '*.txt', '*.safetensors'), local_files_only=True))
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    if 'VJEPA2ForVideoClassification' not in set(config.architectures or ()):
        raise ValueError('dense24 requires the cached SSV2 classification checkpoint')
    processor = AutoVideoProcessor.from_pretrained(snapshot, local_files_only=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    model = VJEPA2ForVideoClassification.from_pretrained(snapshot, local_files_only=True, dtype=dtype).to(device)
    model.eval()
    frames_per_clip = int(model.config.frames_per_clip)
    crop_size = int(model.config.crop_size)
    hidden = int(model.config.hidden_size)
    labels = int(model.config.num_labels)
    features, action_logits, done = open_cache(args.output_dir, len(protocol.labels), hidden, labels)
    pending = np.flatnonzero(~np.asarray(done, dtype=bool))
    if args.max_samples:
        pending = pending[:args.max_samples]
    if not len(pending):
        print(f'dense24 already complete={bool(np.asarray(done).all())}', flush=True)
        return
    loader = DataLoader(Dense24Dataset(rows, pending, frames_per_clip), batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=one_trial, persistent_workers=args.num_workers > 0)
    started = time.time()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    completed = 0
    for item in loader:
        row = int(item['row'])
        feature_parts = []
        logit_parts = []
        for low in range(0, VIEW_COUNT, args.clip_batch):
            prepared = [model_ready_video(video, frames_per_clip, crop_size) for video in item['videos'][low:low + args.clip_batch]]
            encoded = processor(prepared, return_tensors='pt', do_resize=False, do_center_crop=False)
            pixels = encoded.pixel_values_videos.to(device=device, dtype=dtype)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == 'cuda'):
                backbone = model.vjepa2(pixel_values_videos=pixels, skip_predictor=True)
                pooled = model.pooler(backbone.last_hidden_state)
                logits = model.classifier(pooled)
            feature_parts.append(pooled.float().cpu().numpy())
            logit_parts.append(logits.float().cpu().numpy())
        feature_value = np.concatenate(feature_parts)
        logit_value = np.concatenate(logit_parts)
        if feature_value.shape != (VIEW_COUNT, hidden):
            raise ValueError(f'unexpected feature shape {feature_value.shape}')
        if logit_value.shape != (VIEW_COUNT, labels):
            raise ValueError(f'unexpected logit shape {logit_value.shape}')
        features[row] = feature_value.astype(np.float16)
        action_logits[row] = logit_value.astype(np.float16)
        done[row] = True
        completed += 1
        if completed % args.flush_every == 0:
            features.flush()
            action_logits.flush()
            done.flush()
            peak = float(torch.cuda.max_memory_allocated() / 2 ** 30) if device.type == 'cuda' else 0.0
            write_progress(args.output_dir, snapshot, done, started, peak, False)
            print(f'dense24 extracted={int(np.asarray(done).sum())}/{len(done)} this_run={completed} elapsed_min={(time.time() - started) / 60:.1f} peak_gib={peak:.2f}', flush=True)
    features.flush()
    action_logits.flush()
    done.flush()
    complete = bool(np.asarray(done).all())
    peak = float(torch.cuda.max_memory_allocated() / 2 ** 30) if device.type == 'cuda' else 0.0
    write_progress(args.output_dir, snapshot, done, started, peak, complete)
    print(f'dense24 extraction finished complete={complete} done={int(np.asarray(done).sum())}/{len(done)}', flush=True)
