from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
TRAIN_WORKOUTS = ('w01', 'w02', 'w03', 'w04', 'w06', 'w07', 'w08', 'w16', 'w17', 'w18')
VALIDATION_WORKOUTS = ('w14', 'w15', 'w19')
TEST_WORKOUTS = ('w00', 'w05', 'w12', 'w13', 'w20')
ALL_WORKOUTS = TRAIN_WORKOUTS + VALIDATION_WORKOUTS + TEST_WORKOUTS
PARTICIPANT_MAP = {'w00': 2, 'w01': 0, 'w02': 1, 'w03': 0, 'w04': 1, 'w05': 2, 'w06': 0, 'w07': 1, 'w08': 0, 'w09': 1, 'w10': 0, 'w11': 1, 'w12': 3, 'w13': 4, 'w14': 0, 'w15': 1, 'w16': 5, 'w17': 6, 'w18': 7, 'w19': 8, 'w20': 9}
TEACHER_MODALITIES = ('sw_l_acc', 'sw_l_gyr', 'sw_r_acc', 'sw_r_gyr', 'sp_r_acc', 'sp_r_gyr', 'eb_l_acc', 'eb_l_gyr')
STUDENT_MODALITIES = ('sw_r_acc', 'sw_r_gyr')
ACTIONS = ('squats', 'lunges', 'bicep_curls', 'situps', 'pushups', 'tricep_extensions', 'dumbbell_rows', 'jumping_jacks', 'dumbbell_shoulder_press', 'lateral_shoulder_raises', 'non_activity')
ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTIONS)}
POSE_RATE = 30
WINDOW_SECONDS = 5
STRIDE_SECONDS = 1
TARGET_RATE = 50
POSE_WINDOW = POSE_RATE * WINDOW_SECONDS
POSE_STRIDE = POSE_RATE * STRIDE_SECONDS
SENSOR_WINDOW = TARGET_RATE * WINDOW_SECONDS

@dataclass(frozen=True)
class WindowSpec:
    workout: str
    start_pose_index: int
    start_frame: float
    center_frame: float
    label: int

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def split_for_workout(workout: str) -> str:
    if workout in TRAIN_WORKOUTS:
        return 'train'
    if workout in VALIDATION_WORKOUTS:
        return 'validation'
    if workout in TEST_WORKOUTS:
        return 'test'
    raise ValueError(workout)

def load_labels(path: Path) -> list[tuple[int, int, int, str]]:
    rows: list[tuple[int, int, int, str]] = []
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if len(row) != 4:
                raise ValueError(f'{path}:{line_number} must have four fields')
            start, end, repetitions = map(int, row[:3])
            action = row[3]
            if action not in ACTION_TO_INDEX:
                raise ValueError(f'Unknown MM-Fit action {action!r}')
            if start >= end or repetitions < 0:
                raise ValueError(f'Invalid MM-Fit label interval {row}')
            rows.append((start, end, repetitions, action))
    ordered = sorted(rows)
    for left, right in zip(ordered[:-1], ordered[1:]):
        if left[1] > right[0]:
            raise ValueError(f'Overlapping MM-Fit intervals in {path}')
    return rows

def label_at_center(center_frame: float, intervals: list[tuple[int, int, int, str]]) -> int:
    matches = [ACTION_TO_INDEX[action] for start, end, _, action in intervals if center_frame > start and center_frame < end]
    if len(matches) > 1:
        raise ValueError(f'Center frame {center_frame} belongs to multiple labels')
    return matches[0] if matches else ACTION_TO_INDEX['non_activity']

def validate_sensor_array(array: np.ndarray, context: str) -> None:
    if array.ndim != 2 or array.shape[1] < 5:
        raise ValueError(f'{context}: expected [rows, frame, time, xyz]')
    if len(array) < 2:
        raise ValueError(f'{context}: fewer than two sensor rows')
    if not np.isfinite(array[:, :5]).all():
        raise ValueError(f'{context}: non-finite frame/time/xyz')
    if np.any(np.diff(array[:, 0]) < 0) or np.any(np.diff(array[:, 1]) < 0):
        raise ValueError(f'{context}: frame or timestamp is not monotonic')

def extract_sensor_window(array: np.ndarray, start_frame: float, *, window_ms: float=WINDOW_SECONDS * 1000.0, target_length: int=SENSOR_WINDOW) -> np.ndarray | None:
    if array.ndim != 2 or array.shape[1] < 5:
        raise ValueError('sensor: expected [rows, frame, time, xyz]')
    start = int(np.searchsorted(array[:, 0], start_frame, side='left'))
    if start >= len(array) - 1:
        return None
    end_time = float(array[start, 1] + window_ms)
    stop = int(np.searchsorted(array[:, 1], end_time, side='left'))
    stop = min(stop, len(array))
    if stop - start < 2:
        return None
    values = np.asarray(array[start:stop, 2:5], dtype=np.float64).T
    if not np.isfinite(values).all():
        return None
    fraction = Fraction((target_length + 1) / values.shape[1]).limit_denominator(100)
    result = resample_poly(values, fraction.numerator, fraction.denominator, axis=1)
    if result.shape[1] < target_length:
        return None
    return result[:, :target_length].astype(np.float32)

def _paths(root: Path, workout: str) -> dict[str, Path]:
    workout_dir = root / workout
    if not workout_dir.is_dir():
        raise FileNotFoundError(workout_dir)
    result = {'labels': workout_dir / f'{workout}_labels.csv', 'pose_3d': workout_dir / f'{workout}_pose_3d.npy'}
    result.update({modality: workout_dir / f'{workout}_{modality}.npy' for modality in TEACHER_MODALITIES})
    for required in ('pose_3d', *STUDENT_MODALITIES) if workout in TEST_WORKOUTS else ('labels', 'pose_3d', *STUDENT_MODALITIES):
        if not result[required].is_file():
            raise FileNotFoundError(result[required])
    return result

def _load_pose_clock(path: Path) -> np.ndarray:
    pose = np.load(path, mmap_mode='r')
    if pose.ndim != 3 or pose.shape[0] < 1 or pose.shape[2] < 1:
        raise ValueError(f'Unexpected MM-Fit pose clock shape {pose.shape}')
    clock = np.asarray(pose[0, :, 0], dtype=np.float64)
    if len(clock) <= POSE_WINDOW + POSE_RATE or not np.isfinite(clock).all():
        raise ValueError(f'Invalid MM-Fit pose clock in {path}')
    if np.any(np.diff(clock) < 0):
        raise ValueError(f'Non-monotonic MM-Fit pose clock in {path}')
    return clock

def discover_window_specs(root: Path) -> tuple[list[WindowSpec], list[dict[str, Any]]]:
    specs: list[WindowSpec] = []
    exclusions: list[dict[str, Any]] = []
    for workout in ALL_WORKOUTS:
        paths = _paths(root, workout)
        clock = _load_pose_clock(paths['pose_3d'])
        intervals = None if workout in TEST_WORKOUTS else load_labels(paths['labels'])
        student_arrays = {modality: np.load(paths[modality], mmap_mode='r') for modality in STUDENT_MODALITIES}
        for modality, array in student_arrays.items():
            validate_sensor_array(array, f'{workout}/{modality}')
        for start_index in range(0, len(clock) - POSE_WINDOW + 1, POSE_STRIDE):
            start_frame = float(clock[start_index])
            center_frame = float(clock[start_index + POSE_WINDOW // 2])
            available = all((extract_sensor_window(array, start_frame) is not None for array in student_arrays.values()))
            if not available:
                exclusions.append({'workout': workout, 'start_pose_index': start_index, 'reason': 'missing right-smartwatch five-second coverage'})
                continue
            specs.append(WindowSpec(workout=workout, start_pose_index=start_index, start_frame=start_frame, center_frame=center_frame, label=-1 if intervals is None else label_at_center(center_frame, intervals)))
        print(f'MM-Fit preflight {workout}: {sum((spec.workout == workout for spec in specs))} retained', flush=True)
    return (specs, exclusions)

def build_cache(root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    root = root.resolve()
    specs, exclusions = discover_window_specs(root)
    if not specs:
        raise ValueError('No MM-Fit windows passed the frozen Student coverage gate')
    sensor = np.lib.format.open_memmap(output_dir / 'teacher_sensor_24x250.npy', mode='w+', dtype=np.float32, shape=(len(specs), len(TEACHER_MODALITIES) * 3, SENSOR_WINDOW))
    presence = np.lib.format.open_memmap(output_dir / 'modality_presence_8.npy', mode='w+', dtype=np.float32, shape=(len(specs), len(TEACHER_MODALITIES)))
    labels = np.asarray([spec.label for spec in specs], dtype=np.int64)
    metadata_rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    modality_coverage = {modality: 0 for modality in TEACHER_MODALITIES}
    row = 0
    for workout in ALL_WORKOUTS:
        paths = _paths(root, workout)
        arrays: dict[str, np.ndarray | None] = {}
        for modality in TEACHER_MODALITIES:
            path = paths[modality]
            if path.is_file():
                arrays[modality] = np.load(path, mmap_mode='r')
                validate_sensor_array(arrays[modality], f'{workout}/{modality}')
                source_hashes[path.relative_to(root).as_posix()] = sha256_file(path)
            else:
                arrays[modality] = None
        for key in ('pose_3d',) if workout in TEST_WORKOUTS else ('labels', 'pose_3d'):
            path = paths[key]
            source_hashes[path.relative_to(root).as_posix()] = sha256_file(path)
        workout_specs = [spec for spec in specs if spec.workout == workout]
        for spec in workout_specs:
            for modality_index, modality in enumerate(TEACHER_MODALITIES):
                array = arrays[modality]
                window = None if array is None else extract_sensor_window(array, spec.start_frame)
                start = modality_index * 3
                if window is None:
                    sensor[row, start:start + 3] = 0.0
                    presence[row, modality_index] = 0.0
                else:
                    sensor[row, start:start + 3] = window
                    presence[row, modality_index] = 1.0
                    modality_coverage[modality] += 1
            if not all((presence[row, TEACHER_MODALITIES.index(modality)] > 0.5 for modality in STUDENT_MODALITIES)):
                raise AssertionError('Retained MM-Fit row lost Student coverage')
            metadata_rows.append({'segment_id': f'mmfit:{workout}:{spec.start_pose_index}', 'split': split_for_workout(workout), 'workout': workout, 'person': str(PARTICIPANT_MAP[workout]), 'session_key': workout, 'start_pose_index': spec.start_pose_index, 'start_frame': spec.start_frame, 'center_frame': spec.center_frame, 'label': spec.label, 'activity': ACTIONS[spec.label] if spec.label >= 0 else 'unlabeled', 'duration_s': float(WINDOW_SECONDS)})
            row += 1
    if row != len(specs):
        raise AssertionError('MM-Fit cache row count changed during materialization')
    sensor.flush()
    presence.flush()
    metadata = pd.DataFrame(metadata_rows)
    durations = np.full(len(specs), WINDOW_SECONDS, dtype=np.float32)
    np.save(output_dir / 'labels.npy', labels)
    np.save(output_dir / 'durations_s.npy', durations)
    metadata.to_csv(output_dir / 'segments.csv', index=False)
    pd.DataFrame(exclusions).to_csv(output_dir / 'exclusions.csv', index=False)
    if set(metadata['split']) != {'train', 'validation', 'test'}:
        raise AssertionError('MM-Fit cache lacks a frozen split')
    train_people = set(metadata.loc[metadata['split'] == 'train', 'person'])
    test_people = set(metadata.loc[metadata['split'] == 'test', 'person'])
    if train_people & test_people:
        raise AssertionError('MM-Fit unseen TEST participant entered TRAIN')
    if set(metadata.loc[metadata['label'] >= 0, 'label']) != set(range(len(ACTIONS))):
        raise AssertionError('MM-Fit cache does not retain all eleven classes')
    audit = {'protocol': 'protocol/59_MMFIT_UNSEEN_SUBJECT_FROZEN_PROTOCOL.md', 'dataset_root': str(root), 'samples': int(len(metadata)), 'excluded_windows': int(len(exclusions)), 'shape': [len(metadata), len(TEACHER_MODALITIES) * 3, SENSOR_WINDOW], 'split_counts': {str(key): int(value) for key, value in metadata['split'].value_counts().sort_index().items()}, 'participant_counts': {str(key): int(value) for key, value in metadata['person'].value_counts().sort_index().items()}, 'class_counts': {str(key): int(value) for key, value in metadata['label'].value_counts().sort_index().items()}, 'modality_coverage': modality_coverage, 'source_sha256': source_hashes, 'window_seconds': WINDOW_SECONDS, 'stride_seconds': STRIDE_SECONDS, 'target_sensor_rate': TARGET_RATE, 'unseen_test_protocol_frozen_before_access': True, 'one_observation_gap_only': True, 'raw_data_redistributed': False}
    (output_dir / 'cache_metadata.json').write_text(json.dumps(audit, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps({key: audit[key] for key in ('samples', 'excluded_windows', 'split_counts', 'modality_coverage')}, indent=2), flush=True)
    return audit

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    build_cache(args.dataset_root, args.output_dir)
