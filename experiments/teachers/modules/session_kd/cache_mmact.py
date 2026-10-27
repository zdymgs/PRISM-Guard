from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from session_kd._file_hash import sha256_file
from session_kd.mmact import MMActInterval, audit_intervals, discover_annotation_files, parse_annotation_file
SENSORS = {'acc_phone_clip': {'prefix': 'phone_acc', 'rate': 100.0, 'scale': 9.8}, 'acc_watch_clip': {'prefix': 'watch_acc', 'rate': 100.0, 'scale': 9.8}, 'gyro_clip': {'prefix': 'phone_gyro', 'rate': 50.0, 'scale': 1.0}}

def _parse_sensor_time(value: str) -> float:
    text = str(value).strip()
    if '.' not in text:
        text += '.000'
    parsed = pd.to_datetime(text, format='%Y%m%d_%H:%M:%S.%f')
    parsed = parsed.tz_localize('Asia/Tokyo').tz_convert('UTC')
    return float(round(parsed.timestamp(), 2) * 1000.0)

def _sensor_directory(root: Path, sensor: str, interval: MMActInterval) -> Path:
    candidates = []
    for base in ('sensors', 'sensor'):
        for middle in (Path(), Path('trainval')):
            candidates.append(root / 'untrimmed' / base / middle / sensor / interval.subject / interval.scene / interval.session)
    found = [path for path in candidates if path.is_dir()]
    if len(found) != 1:
        raise FileNotFoundError(f'expected one {sensor} directory for {interval.subject}/{interval.scene}/{interval.session}, found {found}')
    return found[0]

def load_sensor_session(directory: Path, rate_hz: float, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Load MMAct chunks using the timestamp convention in the official toolkit."""
    time_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    for path in sorted(directory.glob('*.csv'), key=lambda item: item.name.lower()):
        try:
            frame = pd.read_csv(path, usecols=['time', 'x', 'y', 'z'])
        except ValueError:
            frame = pd.read_csv(path, names=['time', 'x', 'y', 'z'])
        if frame.empty:
            continue
        start_ms = _parse_sensor_time(str(frame.iloc[0]['time']))
        time_parts.append(start_ms + np.arange(len(frame), dtype=np.float64) * (1000.0 / rate_hz))
        values = frame.loc[:, ['x', 'y', 'z']].to_numpy(dtype=np.float32) / float(scale)
        if not np.isfinite(values).all():
            raise ValueError(f'non-finite sensor values in {path}')
        value_parts.append(values)
    if not time_parts:
        raise FileNotFoundError(f'no usable CSV chunks in {directory}')
    times = np.concatenate(time_parts)
    values = np.concatenate(value_parts)
    order = np.argsort(times, kind='stable')
    times = times[order]
    values = values[order]
    unique = np.concatenate((np.diff(times) > 0.0, [True]))
    return (times[unique], values[unique])

def resample_interval(times: np.ndarray, values: np.ndarray, start_ms: int, end_ms: int, length: int, coverage_tolerance_ms: float=25.0, max_gap_ms: float=100.0) -> np.ndarray:
    if length < 2:
        raise ValueError('sequence length must be at least two')
    if times.ndim != 1 or values.shape != (len(times), 3):
        raise ValueError('expected timestamps [N] and sensor values [N, 3]')
    if start_ms < times[0] - coverage_tolerance_ms or end_ms > times[-1] + coverage_tolerance_ms:
        raise ValueError(f'sensor does not cover annotation {start_ms}:{end_ms}; available {times[0]}:{times[-1]}')
    left = max(0, int(np.searchsorted(times, start_ms, side='right')) - 1)
    right = min(len(times), int(np.searchsorted(times, end_ms, side='left')) + 1)
    selected_times = times[left:right]
    largest_gap_ms = float(np.diff(selected_times).max()) if len(selected_times) >= 2 else float('inf')
    if largest_gap_ms > max_gap_ms:
        raise ValueError(f'sensor has {largest_gap_ms:.1f} ms internal gap over {max_gap_ms:.1f} ms limit')
    target = np.linspace(start_ms, end_ms, length, dtype=np.float64)
    return np.stack([np.interp(target, times, values[:, axis]) for axis in range(3)], axis=0).astype(np.float32)

def _ordered_intervals(intervals: list[MMActInterval]) -> list[MMActInterval]:
    return sorted(intervals, key=lambda item: (int(item.subject.removeprefix('subject')), int(item.scene.removeprefix('scene')), int(item.session.removeprefix('session')), item.start_ms, item.end_ms, item.line_number))

def _interval_key(interval: MMActInterval) -> tuple[str, str, str, int]:
    return (interval.subject, interval.scene, interval.session, interval.line_number)

def _load_complete_case_manifest(path: Path, intervals: list[MMActInterval]) -> dict[tuple[str, str, str, int], int]:
    frame = pd.read_csv(path)
    required = {'subject', 'scene', 'session', 'line_number', 'include_primary', 'block_index'}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f'coverage manifest is missing columns: {missing}')
    if len(frame) != len(intervals):
        raise ValueError(f'coverage manifest has {len(frame)} intervals; expected {len(intervals)}')
    manifest_keys = [(str(row.subject), str(row.scene), str(row.session), int(row.line_number)) for row in frame.itertuples(index=False)]
    if len(set(manifest_keys)) != len(manifest_keys):
        raise ValueError('coverage manifest interval keys are duplicated')
    expected_keys = {_interval_key(interval) for interval in intervals}
    if set(manifest_keys) != expected_keys:
        raise ValueError('coverage manifest interval keys do not match annotations')
    selected: dict[tuple[str, str, str, int], int] = {}
    for row, key in zip(frame.itertuples(index=False), manifest_keys):
        include = str(row.include_primary).strip().lower() in {'true', '1'}
        if not include:
            continue
        if pd.isna(row.block_index):
            raise ValueError(f'included interval has no block index: {key}')
        selected[key] = int(row.block_index)
    if not selected:
        raise ValueError('coverage manifest selects no intervals')
    return selected

def build_cache(dataset_root: Path, output_dir: Path, coverage_manifest: Path, sequence_length: int=96) -> None:
    files = discover_annotation_files(dataset_root)
    intervals = _ordered_intervals([item for path in files for item in parse_annotation_file(path, dataset_root)])
    audit = audit_intervals(intervals, n_folds=5)
    if not audit['all_checks_pass']:
        raise ValueError('MMAct interval audit failed; inspect overlaps before caching')
    selected_blocks = _load_complete_case_manifest(coverage_manifest, intervals)
    selected_intervals = [interval for interval in intervals if _interval_key(interval) in selected_blocks]
    label_names = sorted(audit['class_counts'])
    label_to_id = {label: index for index, label in enumerate(label_names)}
    subject_folds = audit['subject_folds']
    output_dir.mkdir(parents=True, exist_ok=False)
    x_path = output_dir / f'imu_accgyro_{sequence_length}.npy'
    array = np.lib.format.open_memmap(x_path, mode='w+', dtype=np.float32, shape=(len(selected_intervals), 9, sequence_length))
    metadata: list[dict[str, object]] = []
    grouped: dict[tuple[str, str, str], list[tuple[int, MMActInterval, int]]] = {}
    for index, interval in enumerate(selected_intervals):
        grouped.setdefault((interval.subject, interval.scene, interval.session), []).append((index, interval, selected_blocks[_interval_key(interval)]))
    for session_number, (_, group) in enumerate(grouped.items(), 1):
        representative = group[0][1]
        sensor_data: list[tuple[np.ndarray, np.ndarray]] = []
        for sensor, specification in SENSORS.items():
            directory = _sensor_directory(dataset_root, sensor, representative)
            sensor_data.append(load_sensor_session(directory, specification['rate'], specification['scale']))
        block_positions: dict[int, int] = {}
        for index, interval, block_index in group:
            try:
                channels = [resample_interval(times, values, interval.start_ms, interval.end_ms, sequence_length) for times, values in sensor_data]
            except ValueError as error:
                raise ValueError(f'failed to cache {interval.subject}/{interval.scene}/{interval.session} line {interval.line_number} ({interval.label}): {error}') from error
            array[index] = np.concatenate(channels, axis=0)
            source_session_key = f'{interval.subject}/{interval.scene}/{interval.session}'
            session_key = f'{source_session_key}/block{block_index:02d}'
            position = block_positions.get(block_index, 0)
            block_positions[block_index] = position + 1
            metadata.append({'segment_id': f'mmact:{source_session_key}:{interval.line_number}', 'user': interval.subject, 'person': interval.subject, 'outer_fold': f'F{subject_folds[interval.subject]}', 'session': f'{interval.scene}/{interval.session}', 'session_key': session_key, 'source_session_key': source_session_key, 'coverage_block': block_index, 'position': position, 'box': '', 'label': label_to_id[interval.label], 'operation': interval.label, 'duration_s': (interval.end_ms - interval.start_ms) / 1000.0, 'frames': sequence_length, 'start_ms': interval.start_ms, 'end_ms_exclusive': interval.end_ms})
        print(f'cached session {session_number}/{len(grouped)}', flush=True)
    array.flush()
    del array
    frame = pd.DataFrame(metadata)
    if len(frame) != len(selected_intervals) or frame['segment_id'].duplicated().any():
        raise AssertionError('MMAct cache metadata is incomplete or duplicated')
    np.save(output_dir / 'labels.npy', frame['label'].to_numpy(dtype=np.int64))
    np.save(output_dir / 'durations_s.npy', frame['duration_s'].to_numpy(dtype=np.float32))
    frame.to_csv(output_dir / 'segments.csv', index=False)
    cache_metadata = {'dataset': 'MMAct approved local copy', 'annotation_files': len(files), 'annotation_sha256': {path.relative_to(dataset_root).as_posix(): sha256_file(path) for path in files}, 'source_annotation_intervals': len(intervals), 'samples': len(frame), 'excluded_incomplete_intervals': len(intervals) - len(frame), 'shape': [len(frame), 9, sequence_length], 'feature_columns': [f'{specification['prefix']}_{axis}' for specification in SENSORS.values() for axis in ('x', 'y', 'z')], 'label_to_id': label_to_id, 'outer_fold_subjects': audit['outer_fold_subjects'], 'coverage_manifest': str(coverage_manifest), 'coverage_manifest_sha256': sha256_file(coverage_manifest), 'coverage_rule': 'all three primary sensors cover the annotation and have no internal timestamp gap over 100 ms', 'sequence_rule': 'excluded annotations split source sessions into independent contiguous coverage blocks', 'resampling': 'per-action linear interpolation in absolute timestamp space', 'acceleration_scale': 'm/s^2 divided by 9.8, matching official toolkit', 'redistributable': False, 'complete': True}
    (output_dir / 'cache_metadata.json').write_text(json.dumps(cache_metadata, indent=2, sort_keys=True), encoding='utf-8')

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--coverage-manifest', type=Path, required=True)
    parser.add_argument('--sequence-length', type=int, default=96)
    args = parser.parse_args()
    build_cache(args.dataset_root, args.output_dir, args.coverage_manifest, args.sequence_length)
