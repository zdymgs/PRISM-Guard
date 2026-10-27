from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models.video import mc3_18
from session_kd.mmact_strong_features import VideoMember, _atomic_json, _safe_scratch_file, extract_member, interval_frame_plan, inventory_archives, letterbox_rgb, load_all_intervals, probe_video
from session_kd.mmfit_strong_features import sha256_file
CHECKPOINT = Path(str(PRETRAINED_ROOT / 'torch/hub/checkpoints/mc3_18-a90a0ba3.pth'))
MEAN = np.asarray([0.43216, 0.394666, 0.37645], dtype=np.float32)
STD = np.asarray([0.22803, 0.22145, 0.216989], dtype=np.float32)
FEATURE_DIM = 512

def load_mc3(device: torch.device, checkpoint: Path=CHECKPOINT) -> nn.Module:
    model = mc3_18(weights=None)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.fc = nn.Identity()
    return model.eval().requires_grad_(False).to(device)

def choose_cameras(cache_root: Path, intervals: list, members: list[VideoMember]) -> tuple[np.ndarray, dict[str, VideoMember]]:
    identity = {member.video_id: member for member in members}
    available: dict[int, list[str]] = {index: [] for index in range(len(intervals))}
    for path in sorted((cache_root / 'videomaev2_large').glob('cam*/subject*/scene*/session*.npz')):
        with np.load(path, allow_pickle=False) as data:
            indices = data['global_interval_index'].astype(int)
            covered = data['video_covered']
        camera, subject, scene, session = (path.parts[-4], path.parts[-3], path.parts[-2], path.stem)
        video_id = f'{camera}:{subject}:{scene}:{session}'
        if video_id not in identity:
            raise ValueError(f'coverage cache has no archive member: {video_id}')
        for index in indices[covered]:
            available[int(index)].append(camera)
    chosen = np.empty(len(intervals), dtype='<U4')
    for index in range(len(intervals)):
        cameras = sorted(set(available[index]), key=lambda value: int(value[3:]))
        if not cameras:
            raise ValueError(f'annotation {index} has no covered camera')
        chosen[index] = 'cam1' if 'cam1' in cameras else cameras[0]
    return (chosen, identity)

@torch.inference_mode()
def encode_selected(video_path: Path, intervals: list, indices: list[int], model: nn.Module, device: torch.device, probe: dict, batch_size: int) -> tuple[np.ndarray, dict]:
    requests, covered, _ = interval_frame_plan(intervals, indices, probe)
    if not covered.all():
        raise ValueError('selected compact-cache video does not cover every assigned interval')
    pending = {row: [None] * 16 for row in range(len(indices))}
    features = np.full((len(indices), FEATURE_DIM), np.nan, dtype=np.float16)
    clips, rows = ([], [])
    capture = cv2.VideoCapture(str(video_path))
    cv2.setNumThreads(2)
    started = time.perf_counter()
    peak = 0.0

    def flush() -> None:
        nonlocal peak
        if not clips:
            return
        values = np.stack(clips).astype(np.float32) / 255.0
        values = (values - MEAN[None, None, None, None, :]) / STD[None, None, None, None, :]
        tensor = torch.from_numpy(values).permute(0, 4, 1, 2, 3).to(device, non_blocking=True)
        result = model(tensor).float().cpu().numpy()
        if result.shape != (len(clips), FEATURE_DIM) or not np.isfinite(result).all():
            raise ValueError(f'invalid MC3 output {result.shape}')
        features[np.asarray(rows)] = result.astype(np.float16)
        if device.type == 'cuda':
            peak = max(peak, torch.cuda.max_memory_allocated() / 2 ** 30)
        clips.clear()
        rows.clear()
    try:
        for frame_id in range(max(requests) + 1):
            if not capture.grab():
                raise RuntimeError(f'decode ended at frame {frame_id}')
            if frame_id not in requests:
                continue
            ok, bgr = capture.retrieve()
            if not ok or bgr is None:
                raise RuntimeError(f'retrieve failed at frame {frame_id}')
            prepared = letterbox_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), 112)
            for row, slot in requests[frame_id]:
                pending[row][slot] = prepared
                if all((value is not None for value in pending[row])):
                    clips.append(np.stack(pending.pop(row)))
                    rows.append(row)
                    if len(clips) >= batch_size:
                        flush()
        flush()
    finally:
        capture.release()
    if pending or not np.isfinite(features.astype(np.float32)).all():
        raise ValueError('incomplete/nonfinite MC3 cache shard')
    elapsed = time.perf_counter() - started
    return (features, {'rows': len(indices), 'elapsed_seconds': elapsed, 'rows_per_second': len(indices) / elapsed, 'peak_cuda_gib': peak, 'requested_unique_frames': len(requests)})

def run(dataset_root: Path, archive_dir: Path, teacher_cache: Path, output: Path, scratch: Path, batch_size: int, limit_videos: int | None, delete_extracted: bool) -> dict:
    intervals, _ = load_all_intervals(dataset_root)
    members = inventory_archives(archive_dir)
    chosen, identity = choose_cameras(teacher_cache, intervals, members)
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(intervals):
        key = f'{chosen[index]}:{row.subject}:{row.scene}:{row.session}'
        groups.setdefault(key, []).append(index)
    ordered = sorted(groups)
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / 'input_manifest.json', {'protocol': 'MMACT_MC3_STUDENT_FEATURES_V1', 'rows': len(intervals), 'checkpoint': str(CHECKPOINT), 'checkpoint_sha256': sha256_file(CHECKPOINT), 'feature_dim': FEATURE_DIM, 'camera_policy': 'cam1_if_covered_else_lowest_numbered_covered_camera', 'label_columns_used_for_selection': [], 'preprocess': 'RGB full-frame letterbox112; /255; Kinetics400 mean/std; 16 uniform frames', 'rows_manifest': [{'global_interval_index': index, 'interval_id': f'{row.annotation_file}:{row.line_number}', 'subject': row.subject, 'scene': row.scene, 'session': row.session, 'start_ms': row.start_ms, 'end_ms': row.end_ms, 'selected_camera': str(chosen[index])} for index, row in enumerate(intervals)]})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('MC3 cache pilot requires CUDA')
    model = load_mc3(device)
    summary = {'total_intervals': len(intervals), 'selected_videos': len(ordered), 'camera_counts': {camera: int(np.count_nonzero(chosen == camera)) for camera in sorted(set(chosen))}, 'completed_shards': 0, 'skipped_shards': 0, 'stats': []}
    for video_id in ordered[:limit_videos]:
        member, indices = (identity[video_id], groups[video_id])
        shard = output / 'shards' / (video_id.replace(':', '_') + '.npz')
        marker = shard.with_suffix('.complete.json')
        ids = [f'{intervals[i].annotation_file}:{intervals[i].line_number}' for i in indices]
        contract = {'video_id': video_id, 'source_crc32': member.crc32, 'checkpoint_sha256': sha256_file(CHECKPOINT), 'global_interval_indices': indices, 'ordered_id_sha256': hashlib.sha256(('\n'.join(ids) + '\n').encode()).hexdigest()}
        if shard.exists() and marker.exists():
            state = json.loads(marker.read_text())
            if all((state.get(k) == v for k, v in contract.items())) and sha256_file(shard) == state['cache_sha256']:
                summary['skipped_shards'] += 1
                if delete_extracted:
                    old_scratch = _safe_scratch_file(scratch, member)
                    if old_scratch.exists():
                        verified = extract_member(member, scratch)
                        if verified.resolve() != old_scratch.resolve():
                            raise ValueError('resume scratch deletion confinement failure')
                        verified.unlink()
                continue
            raise ValueError(f'resume contract mismatch: {video_id}')
        video = extract_member(member, scratch)
        probe = probe_video(video, member.end_ms)
        values, stats = encode_selected(video, intervals, indices, model, device, probe, batch_size)
        shard.parent.mkdir(parents=True, exist_ok=True)
        temporary = shard.with_suffix('.npz.building')
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, features=values, global_interval_index=np.asarray(indices, np.int32), interval_id=np.asarray(ids), selected_camera=np.asarray([chosen[i] for i in indices]))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(shard)
        with np.load(shard, allow_pickle=False) as check:
            if check['features'].shape != (len(indices), FEATURE_DIM) or not np.isfinite(check['features'].astype(np.float32)).all():
                raise ValueError('durable shard validation failed')
        _atomic_json(marker, {**contract, 'status': 'complete', 'cache_sha256': sha256_file(shard), 'probe': probe, 'stats': stats, 'preprocess': 'RGB full-frame letterbox112; /255; Kinetics400 mean/std; 16 uniform frames', 'label_columns_used_for_selection': []})
        summary['completed_shards'] += 1
        summary['stats'].append({'video_id': video_id, **stats})
        _atomic_json(output / 'run_summary.json', summary)
        print(json.dumps({'video_id': video_id, **stats}), flush=True)
        if delete_extracted:
            expected = _safe_scratch_file(scratch, member)
            if video.resolve() != expected:
                raise ValueError('scratch deletion confinement failure')
            video.unlink()
    if limit_videos is None and summary['completed_shards'] + summary['skipped_shards'] == len(ordered):
        complete = np.full((len(intervals), FEATURE_DIM), np.nan, dtype=np.float16)
        for shard in (output / 'shards').glob('*.npz'):
            with np.load(shard, allow_pickle=False) as data:
                complete[data['global_interval_index']] = data['features']
        if not np.isfinite(complete.astype(np.float32)).all():
            raise ValueError('incomplete aggregate MC3 cache')
        temporary = output / 'mc3_18_512_fp16.npy.building'
        with temporary.open('wb') as handle:
            np.save(handle, complete)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output / 'mc3_18_512_fp16.npy')
        summary['aggregate_sha256'] = sha256_file(output / 'mc3_18_512_fp16.npy')
    _atomic_json(output / 'run_summary.json', summary)
    return summary

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--archive-dir', type=Path, required=True)
    parser.add_argument('--teacher-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--limit-videos', type=int)
    parser.add_argument('--delete-extracted', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root, args.archive_dir, args.teacher_cache, args.output, args.scratch, args.batch_size, args.limit_videos, args.delete_extracted), indent=2))
