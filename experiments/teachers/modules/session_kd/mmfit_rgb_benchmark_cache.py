from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import csv
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from session_kd._rgb_encoder import encode_rgb_frames, load_local_resnet18_encoder
from session_kd._mmfit_windows import ACTION_TO_INDEX, ALL_WORKOUTS, PARTICIPANT_MAP, POSE_STRIDE, POSE_WINDOW, STUDENT_MODALITIES, TEST_WORKOUTS, TRAIN_WORKOUTS, VALIDATION_WORKOUTS, extract_sensor_window, load_labels, split_for_workout
RGB_OFFSETS = np.array([0, 50, 99, 149], dtype=np.int64)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def _label_at_center_source(center: float, intervals: list[tuple[int, int, int, str]]) -> int:
    found = [ACTION_TO_INDEX[a] for start, end, _, a in intervals if start < center < end]
    if len(found) > 1:
        raise ValueError('Overlapping source label intervals')
    return found[0] if found else ACTION_TO_INDEX['non_activity']

def enumerate_inputs(data_root: Path, workouts: tuple[str, ...]=ALL_WORKOUTS):
    """Enumerate inputs; TEST label files are neither opened nor named."""
    rows: list[dict[str, object]] = []
    source_labels: list[int] = []
    for workout in workouts:
        folder = data_root / workout
        pose = np.load(folder / f'{workout}_pose_3d.npy', mmap_mode='r')
        clock = np.asarray(pose[0, :, 0], dtype=np.float64)
        sensors = [np.load(folder / f'{workout}_{name}.npy', mmap_mode='r') for name in STUDENT_MODALITIES]
        intervals = None if workout in TEST_WORKOUTS else load_labels(folder / f'{workout}_labels.csv')
        for start_index in range(0, len(clock) - POSE_WINDOW + 1, POSE_STRIDE):
            start_frame = float(clock[start_index])
            if not all((extract_sensor_window(x, start_frame) is not None for x in sensors)):
                continue
            if start_frame != int(start_frame):
                raise ValueError('Nonintegral MM-Fit reference frame')
            center = float(clock[start_index + POSE_WINDOW // 2])
            rows.append({'segment_id': f'mmfit:{workout}:{start_index}', 'split': split_for_workout(workout), 'workout': workout, 'person': str(PARTICIPANT_MAP[workout]), 'start_pose_index': start_index, 'start_frame': int(start_frame), 'center_frame': int(center)})
            source_labels.append(-1 if intervals is None else _label_at_center_source(center, intervals))
    return (rows, np.asarray(source_labels, dtype=np.int64))

def validate_imu_alignment(rows: list[dict[str, object]], segments_path: Path) -> None:
    """Prove row-order equality while explicitly excluding label/activity columns."""
    columns = ['segment_id', 'split', 'workout', 'person', 'start_pose_index', 'start_frame', 'center_frame']
    legacy = pd.read_csv(segments_path, usecols=columns)
    if len(legacy) != len(rows):
        raise ValueError(f'IMU alignment row mismatch: {len(legacy)} != {len(rows)}')
    for index, row in enumerate(rows):
        expected = (str(row['segment_id']), str(row['split']), str(row['workout']), str(row['person']), int(row['start_pose_index']), int(row['start_frame']), int(row['center_frame']))
        actual_row = legacy.iloc[index]
        actual = (str(actual_row.segment_id), str(actual_row.split), str(actual_row.workout), str(actual_row.person), int(actual_row.start_pose_index), int(actual_row.start_frame), int(actual_row.center_frame))
        if actual != expected:
            raise ValueError(f'IMU alignment mismatch at row {index}: {actual!r} != {expected!r}')

def validate_derived_arrays(rgb: np.ndarray, pose: np.ndarray, chunk: int=512) -> None:
    if rgb.shape[0] != pose.shape[0] or rgb.shape[1:] != (4, 512) or pose.shape[1:] != (51, 150):
        raise ValueError('Derived cache shape mismatch')
    for start in range(0, len(rgb), chunk):
        r = np.asarray(rgb[start:start + chunk])
        p = np.asarray(pose[start:start + chunk])
        if not np.isfinite(r).all() or not np.isfinite(p).all():
            raise ValueError(f'Non-finite derived cache values in rows {start}:{start + chunk}')
        if np.max(np.abs(r.astype(np.float32))) > np.finfo(np.float16).max:
            raise ValueError('RGB feature exceeds FP16 range')

def _encode_one_video(video_path: Path, frame_to_slots: dict[int, list[tuple[int, int]]], features: np.memmap, encoder, device: torch.device, batch_size: int=64) -> dict[str, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f'Could not open {video_path}')
    wanted = set(frame_to_slots)
    maximum = max(wanted)
    batch_rgb: list[np.ndarray] = []
    batch_ids: list[int] = []
    decoded = selected = 0
    started = time.perf_counter()

    def flush() -> None:
        nonlocal selected
        if not batch_rgb:
            return
        encoded = encode_rgb_frames(encoder, np.stack(batch_rgb), device).cpu().numpy()
        for vector, frame_id in zip(encoded, batch_ids):
            for row_index, time_index in frame_to_slots[frame_id]:
                features[row_index, time_index] = vector.astype(np.float16)
            selected += 1
        batch_rgb.clear()
        batch_ids.clear()
    try:
        index = 0
        while index <= maximum:
            ok, bgr = capture.read()
            if not ok or bgr is None:
                raise RuntimeError(f'Decode ended at {index}, required through {maximum}: {video_path}')
            decoded += 1
            if index in wanted:
                batch_rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                batch_ids.append(index)
                if len(batch_rgb) >= batch_size:
                    flush()
            index += 1
        flush()
    finally:
        capture.release()
    elapsed = time.perf_counter() - started
    if selected != len(wanted):
        raise RuntimeError(f'Encoded {selected} unique frames but requested {len(wanted)}')
    return {'decoded_frames': decoded, 'unique_encoded_frames': selected, 'elapsed_seconds': elapsed, 'decoded_fps': decoded / elapsed}

def build_input_cache(data_root: Path, rgb_root: Path, output: Path, weights: Path, workouts: tuple[str, ...]=ALL_WORKOUTS, batch_size: int=64) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.perf_counter()
    rows, source_labels = enumerate_inputs(data_root, workouts)
    count = len(rows)
    rgb = np.lib.format.open_memmap(output / 'rgb_4x512_fp16.npy', mode='w+', dtype=np.float16, shape=(count, 4, 512))
    pose_cache = np.lib.format.open_memmap(output / 'pose_51x150_fp16.npy', mode='w+', dtype=np.float16, shape=(count, 51, 150))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = load_local_resnet18_encoder(weights, device)
    workout_stats = {}
    for workout in workouts:
        indices = [i for i, row in enumerate(rows) if row['workout'] == workout]
        pose = np.load(data_root / workout / f'{workout}_pose_3d.npy', mmap_mode='r')
        requests: dict[int, list[tuple[int, int]]] = {}
        for row_index in indices:
            start_index = int(rows[row_index]['start_pose_index'])
            xyz = np.asarray(pose[:, start_index:start_index + 150, 1:], dtype=np.float32)
            xyz -= xyz[:, :, :1]
            pose_cache[row_index] = xyz.transpose(0, 2, 1).reshape(51, 150).astype(np.float16)
            start_frame = int(rows[row_index]['start_frame'])
            for time_index, frame_id in enumerate(start_frame + RGB_OFFSETS):
                requests.setdefault(int(frame_id), []).append((row_index, time_index))
        workout_stats[workout] = _encode_one_video(rgb_root / f'{workout}_rgb.mp4', requests, rgb, encoder, device, batch_size)
        print(json.dumps({workout: workout_stats[workout]}), flush=True)
    rgb.flush()
    pose_cache.flush()
    validate_derived_arrays(rgb, pose_cache)
    with (output / 'inputs.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.save(output / 'source_labels_test_minus1.npy', source_labels)
    manifest = {'status': 'input_cache_complete_test_labels_never_opened', 'workouts': list(workouts), 'rows': count, 'rgb_offsets': RGB_OFFSETS.tolist(), 'weights': str(weights.resolve()), 'weights_sha256': sha256_file(weights), 'device': str(device), 'workout_stats': workout_stats, 'elapsed_seconds': time.perf_counter() - started, 'bytes': sum((p.stat().st_size for p in output.iterdir() if p.is_file()))}
    (output / 'cache_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest
